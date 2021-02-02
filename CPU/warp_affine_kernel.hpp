// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// source: https://github.com/openvinotoolkit/openvino/tree/master/docs/template_extension

//! [fft_kernel:header]
#pragma once

#include <ie_iextension.h>
#include <ngraph/ngraph.hpp>

namespace CustomLayers {

class WarpAffineImpl : public InferenceEngine::ILayerExecImpl {
public:
    explicit WarpAffineImpl(const std::shared_ptr<ngraph::Node>& node);
    InferenceEngine::StatusCode getSupportedConfigurations(std::vector<InferenceEngine::LayerConfig> &conf,
                                                           InferenceEngine::ResponseDesc *resp) noexcept override;
    InferenceEngine::StatusCode init(InferenceEngine::LayerConfig &config,
                                     InferenceEngine::ResponseDesc *resp) noexcept override;
    InferenceEngine::StatusCode execute(std::vector<InferenceEngine::Blob::Ptr> &inputs,
                                        std::vector<InferenceEngine::Blob::Ptr> &outputs,
                                        InferenceEngine::ResponseDesc *resp) noexcept override;
private:
    ngraph::Shape in_shape_img_;
    ngraph::Shape in_shape_mat_;
    ngraph::Shape out_shape_;
    std::string error;

    void interpolate(const size_t N, const size_t C,
                     const float *src, const size_t IH, const size_t IW,
                     float *dst, const size_t OH, const size_t OW,
                     const float *matrices);
};

}
//! [fft_kernel:header]
