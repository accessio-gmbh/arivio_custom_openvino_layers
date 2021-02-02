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
    size_t IH = inputs[0]->getTensorDesc().getDims()[2];
    size_t IW = inputs[0]->getTensorDesc().getDims()[3];
    size_t OH = outputs[0]->getTensorDesc().getDims()[2];
    size_t OW = outputs[0]->getTensorDesc().getDims()[3];

    auto *dst_data = outputs[0]->buffer().as<float *>();

    switch (inputs[0]->getTensorDesc().getPrecision()) {
        case Precision::FP32:
        {
            size_t IC = inputs[0]->getTensorDesc().getBlockingDesc().getBlockDims()[1] *
                        inputs[0]->getTensorDesc().getBlockingDesc().getBlockDims()[4];
            interpolate(IN, IC, inputs[0]->buffer().as<const float *>(), IH, IW, dst_data, OH, OW, inputs[1]->buffer().as<const float *>());
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
                                 const size_t OW, const float* matrices){
#if defined(HAVE_AVX512F)
    const int block_size = 16;
#else
    const int block_size = 8;
#endif
    const float zero_float = 0.f;

    // Align channel number to block size to deal with channels padding in IE with multiple blobs
    size_t CB = (C + block_size - 1) & (-block_size);
    size_t CH = (C + block_size - 1) / block_size;

    parallel_for3d(N, CH, OH, [&](size_t n, size_t cb, size_t h) {
        const float* matrix = matrices + 6 * n;
        const float *psrc = src + n * CB * IH * IW;

        for (size_t w = 0; w < OW; ++w) {
            float xi = w*matrix[0] + h*matrix[1] + matrix[2];
            float yi = w*matrix[3] + h*matrix[4] + matrix[5];

            const float *psrc00, *psrc01, *psrc10, *psrc11;
            float h_lambda0 = 1.f, h_lambda1 = 0.f, w_lambda0 = 1.f, w_lambda1 = 0.f;
            if(xi < -0.f || yi < -0.f || xi >= (IW-1.f) || yi >= (IH-1.f)){
                psrc00 = psrc01 = psrc10 = psrc11 = &zero_float;
            }
            else{
                int ih0 = (int)(yi);
                int ih1 = ih0 + 1;
                h_lambda0 = yi - ih0;
                h_lambda1 = 1.0f - h_lambda0;

                int iw0 = (int)(xi);
                int iw1 = iw0 + 1;
                w_lambda0 = xi - iw0;
                w_lambda1 = 1.0f - w_lambda0;

                psrc00 = psrc + cb * block_size * IW * IH + ih0 * IW * block_size + iw0 * block_size;
                psrc01 = psrc + cb * block_size * IW * IH + ih0 * IW * block_size + iw1 * block_size;
                psrc10 = psrc + cb * block_size * IW * IH + ih1 * IW * block_size + iw0 * block_size;
                psrc11 = psrc + cb * block_size * IW * IH + ih1 * IW * block_size + iw1 * block_size;
            }
            float *pdst = dst + n * CB * OH * OW + cb * block_size * OW * OH + h * OW * block_size +
                          w * block_size;

#if defined(HAVE_AVX512F)
            __m512 vwl0 = _mm512_set1_ps(w_lambda0);
                        __m512 vwl1 = _mm512_set1_ps(w_lambda1);
                        __m512 vhl0 = _mm512_set1_ps(h_lambda0);
                        __m512 vhl1 = _mm512_set1_ps(h_lambda1);
                        __m512 vsrc00 = _mm512_loadu_ps(psrc00);
                        __m512 vsrc01 = _mm512_loadu_ps(psrc01);
                        __m512 vsrc10 = _mm512_loadu_ps(psrc10);
                        __m512 vsrc11 = _mm512_loadu_ps(psrc11);

                        __m512 vdst0 = _mm512_fmadd_ps(vwl1, vsrc00, _mm512_mul_ps(vwl0, vsrc01));
                        __m512 vdst1 = _mm512_fmadd_ps(vwl1, vsrc10, _mm512_mul_ps(vwl0, vsrc11));
                        __m512 vdst  = _mm512_fmadd_ps(vhl1, vdst0, _mm512_mul_ps(vhl0, vdst1));

                        _mm512_storeu_ps(pdst, vdst);
#elif defined(HAVE_AVX2)
            __m256 vwl0 = _mm256_set1_ps(w_lambda0);
                        __m256 vwl1 = _mm256_set1_ps(w_lambda1);
                        __m256 vhl0 = _mm256_set1_ps(h_lambda0);
                        __m256 vhl1 = _mm256_set1_ps(h_lambda1);
                        __m256 vsrc00 = _mm256_loadu_ps(psrc00);
                        __m256 vsrc01 = _mm256_loadu_ps(psrc01);
                        __m256 vsrc10 = _mm256_loadu_ps(psrc10);
                        __m256 vsrc11 = _mm256_loadu_ps(psrc11);

                       __m256 vdst0 = _mm256_fmadd_ps(vwl1, vsrc00, _mm256_mul_ps(vwl0, vsrc01));
                       __m256 vdst1 = _mm256_fmadd_ps(vwl1, vsrc10, _mm256_mul_ps(vwl0, vsrc11));
                       __m256 vdst  = _mm256_fmadd_ps(vhl1, vdst0, _mm256_mul_ps(vhl0, vdst1));

                       _mm256_storeu_ps(pdst, vdst);
#elif defined(HAVE_SSE)
            __m128 vwl0 = _mm_set1_ps(w_lambda0);
                        __m128 vwl1 = _mm_set1_ps(w_lambda1);
                        __m128 vhl0 = _mm_set1_ps(h_lambda0);
                        __m128 vhl1 = _mm_set1_ps(h_lambda1);
                        for (int i = 0; i < block_size/4; i++) {
                            __m128 vsrc00 = _mm_loadu_ps(psrc00 + i*block_size/2);
                            __m128 vsrc01 = _mm_loadu_ps(psrc01 + i*block_size/2);
                            __m128 vsrc10 = _mm_loadu_ps(psrc10 + i*block_size/2);
                            __m128 vsrc11 = _mm_loadu_ps(psrc11 + i*block_size/2);

                           __m128 vdst00 = _mm_mul_ps(vwl1, vsrc00);
                           __m128 vdst01 = _mm_mul_ps(vwl0, vsrc01);
                           __m128 vdst10 = _mm_mul_ps(vwl1, vsrc10);
                           __m128 vdst11 = _mm_mul_ps(vwl0, vsrc11);

                           __m128 vdst0 = _mm_add_ps(vdst00, vdst01);
                           __m128 vdst1 = _mm_add_ps(vdst10, vdst11);

                            __m128 vdst = _mm_add_ps(_mm_mul_ps(vhl1, vdst0), _mm_mul_ps(vhl0, vdst1));

                           _mm_storeu_ps(pdst + i*block_size/2, vdst);
                        }
#else
            for (int c = 0; c < block_size; ++c) {
                pdst[c] = h_lambda1 * (w_lambda1 * psrc00[c] + w_lambda0 * psrc01[c]) +
                          h_lambda0 * (w_lambda1 * psrc10[c] + w_lambda0 * psrc11[c]);
            }
#endif
        }
    });
}

