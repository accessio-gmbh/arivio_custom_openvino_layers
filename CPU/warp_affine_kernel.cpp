// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "warp_affine_kernel.hpp"
#include "warp_affine_op.hpp"
#include <details/ie_exception.hpp>
#include <ie_layouts.h>
#if defined(HAVE_SSE) || defined(HAVE_AVX2) || defined(HAVE_AVX512F)
#include <immintrin.h>
#endif
#include "ie_parallel.hpp"

using namespace CustomLayers;
using namespace InferenceEngine;

WarpAffineImpl::WarpAffineImpl(const std::shared_ptr<ngraph::Node> &node) {
    auto castedNode = std::dynamic_pointer_cast<WarpAffineOp>(node);
    if (!castedNode)
        THROW_IE_EXCEPTION << "Cannot create implementation for unknown operation!";
    if (castedNode->inputs().size() != 2 || castedNode->outputs().size() != 1)
        THROW_IE_EXCEPTION << "Cannot create implementation for operation with incorrect number of inputs or outputs!";
    if (castedNode->get_input_partial_shape(0).is_dynamic() || castedNode->get_output_partial_shape(0).is_dynamic())
        THROW_IE_EXCEPTION << "Cannot create implementation for op with dynamic shapes!";
    if ((castedNode->get_input_element_type(0) != ngraph::element::f32 && castedNode->get_input_element_type(0) != ngraph::element::u8)
            || castedNode->get_input_element_type(1) != ngraph::element::f32 || castedNode->get_output_element_type(0) != ngraph::element::f32)
        THROW_IE_EXCEPTION << "Operation supports only U8 or FP32 for input0 and FP32 for input1 and output0 tensors.";
    in_shape_img_ = castedNode->get_input_shape(0);
    in_shape_mat_ = castedNode->get_input_shape(1);
    out_shape_ = castedNode->get_output_shape(0);
}

StatusCode WarpAffineImpl::getSupportedConfigurations(std::vector<LayerConfig> &conf, ResponseDesc *resp) noexcept {
    std::vector<DataConfig> in_data_config;
    std::vector<DataConfig> out_data_config;

    SizeVector order_img(in_shape_img_.size());
    std::iota(order_img.begin(), order_img.end(), 0);
    size_t offset_img((std::numeric_limits<size_t>::max)());
    DataConfig img_conf;
    img_conf.desc = TensorDesc(Precision::FP32, in_shape_img_, {in_shape_img_, order_img, offset_img});
    in_data_config.push_back(img_conf);

    SizeVector order_mat(in_shape_mat_.size());
    std::iota(order_mat.begin(), order_mat.end(), 0);
    size_t offset_mat((std::numeric_limits<size_t>::max)());
    DataConfig mat_conf;
    mat_conf.desc = TensorDesc(Precision::FP32, in_shape_mat_, {in_shape_mat_, order_mat, offset_mat});
    in_data_config.push_back(mat_conf);

    // Output shape
    SizeVector order_out(out_shape_.size());
    std::iota(order_out.begin(), order_out.end(), 0);
    size_t offset_out((std::numeric_limits<size_t>::max)());
    DataConfig out_conf;
    out_conf.desc = TensorDesc(Precision::FP32, out_shape_, {out_shape_, order_out, offset_out});
    out_data_config.push_back(out_conf);

    LayerConfig layerConfig;
    layerConfig.inConfs = in_data_config;
    layerConfig.outConfs = out_data_config;

    conf.push_back(layerConfig);
    return StatusCode::OK;
}

StatusCode WarpAffineImpl::init(LayerConfig &config, ResponseDesc *resp) noexcept {
    try {
        if (config.inConfs.size() != 2 || config.outConfs.size() != 1) {
            THROW_IE_EXCEPTION << "Operation cannot be initialized with incorrect number of inputs/outputs!";
        }

        if (config.outConfs[0].desc.getPrecision() != Precision::FP32 ||
                (config.inConfs[0].desc.getPrecision() != Precision::FP32 &&
                 config.inConfs[0].desc.getPrecision() != Precision::U8) ||
                config.inConfs[1].desc.getPrecision() != Precision::FP32)  {
            THROW_IE_EXCEPTION << "Operation supports only FP32 precisions except U8 for input!";
        }
    } catch (details::InferenceEngineException& ex) {
        if (resp) {
            strncpy(resp->msg, error.c_str(), sizeof(resp->msg) - 1);
            resp->msg[sizeof(resp->msg)-1] = 0;
        }
        return GENERAL_ERROR;
    }
    return OK;
}

StatusCode WarpAffineImpl::execute(std::vector<Blob::Ptr> &inputs, std::vector<Blob::Ptr> &outputs, ResponseDesc *resp) noexcept {
    size_t IN = inputs[0]->getTensorDesc().getDims()[0];
    size_t IC = inputs[0]->getTensorDesc().getDims()[1];
    size_t IH = inputs[0]->getTensorDesc().getDims()[2];
    size_t IW = inputs[0]->getTensorDesc().getDims()[3];
    size_t OH = outputs[0]->getTensorDesc().getDims()[2];
    size_t OW = outputs[0]->getTensorDesc().getDims()[3];
    SizeVector strides_in = inputs[0]->getTensorDesc().getBlockingDesc().getStrides();
    SizeVector strides_out = outputs[0]->getTensorDesc().getBlockingDesc().getStrides();

    auto *dst_data = outputs[0]->buffer().as<float *>();

    switch (inputs[0]->getTensorDesc().getPrecision()) {
        case Precision::FP32:
        {
            interpolate(IN, IC, inputs[0]->buffer().as<const float *>(), IH, IW, dst_data, OH, OW, inputs[1]->buffer().as<const float *>(), strides_in, strides_out);
        }
            break;
        default:
            if (resp) {
                std::string errorMsg = "Incorrect input precision. Only FP32 is supported!";
                errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
            }
            return GENERAL_ERROR;
    }

    return OK;
}

void WarpAffineImpl::interpolate(const size_t N, const size_t C, const float* src, const size_t IH, const size_t IW, float* dst, const size_t OH,
                                 const size_t OW, const float* matrices, const SizeVector& strides_in, const SizeVector& strides_out){
    parallel_for3d(N, C, OH, [&](size_t n, size_t c, size_t h) {
        const float* matrix = matrices + 6 * n;
        const float *psrc_base = src + n * strides_in[0] + c * strides_in[1];
        float *pdst_base = dst + n * strides_out[0] + c * strides_out[1] + h * strides_out[2];

        for (size_t w = 0; w < OW; ++w) {
            float xi = w*matrix[0] + h*matrix[1] + matrix[2];
            float yi = w*matrix[3] + h*matrix[4] + matrix[5];

            float *pdst = pdst_base + w * strides_out[3];
            if(xi < -0.f || yi < -0.f || xi >= (IW-1.f) || yi >= (IH-1.f)){
                *pdst = 84.f;
            }
            else{
                int ih0 = (int)(yi);
                int ih1 = ih0 + 1;
                float h_lambda0 = yi - ih0;
                float h_lambda1 = 1.0f - h_lambda0;

                int iw0 = (int)(xi);
                int iw1 = iw0 + 1;
                float w_lambda0 = xi - iw0;
                float w_lambda1 = 1.0f - w_lambda0;

                const float *psrc00 = psrc_base + ih0 * strides_in[2] + iw0 * strides_in[3];
                const float *psrc01 = psrc_base + ih0 * strides_in[2] + iw1 * strides_in[3];
                const float *psrc10 = psrc_base + ih1 * strides_in[2] + iw0 * strides_in[3];
                const float *psrc11 = psrc_base + ih1 * strides_in[2] + iw1 * strides_in[3];

                *pdst =  h_lambda1 * (w_lambda1 * psrc00[c] + w_lambda0 * psrc01[c]) +
                         h_lambda0 * (w_lambda1 * psrc10[c] + w_lambda0 * psrc11[c]);
            }
        }
    });
}

