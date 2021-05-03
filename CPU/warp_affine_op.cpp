// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//! [fft_op:implementation]
#include "warp_affine_op.hpp"

using namespace CustomLayers;

constexpr ngraph::NodeTypeInfo WarpAffineOp::type_info;

WarpAffineOp::WarpAffineOp(const ngraph::Output<ngraph::Node>& image,
                           const ngraph::Output<ngraph::Node>& transformation_matrix,
                           const ngraph::PartialShape& output_shape,
                           bool keep_scale_on_reshape,
                           const std::vector<float>& scale)
        : Op({image, transformation_matrix}), output_shape_(output_shape), keep_scale_on_reshape_(keep_scale_on_reshape), scale_(scale){
    constructor_validate_and_infer_types();
}

void WarpAffineOp::validate_and_infer_types() {
    if(keep_scale_on_reshape_){
        std::vector<ngraph::Dimension> out_shape(get_input_partial_shape(0));
        out_shape[2] = ngraph::Dimension(out_shape[2].get_min_length() * scale_[0], out_shape[2].get_max_length() * scale_[0]);
        out_shape[3] = ngraph::Dimension(out_shape[3].get_min_length() * scale_[1], out_shape[3].get_max_length() * scale_[1]);
        set_output_type(0, get_input_element_type(0), out_shape);
    }
    else {
        output_shape_[0] = ngraph::Dimension(get_input_partial_shape(0).get_min_shape()[0], get_input_partial_shape(0).get_max_shape()[0]);
        set_output_type(0, ngraph::element::Type(ngraph::element::Type_t::f32), output_shape_);
    }
}

std::shared_ptr<ngraph::Node> WarpAffineOp::clone_with_new_inputs(const ngraph::OutputVector &new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<WarpAffineOp>(new_args.at(0), new_args.at(1), output_shape_, keep_scale_on_reshape_, scale_);
}

bool WarpAffineOp::visit_attributes(ngraph::AttributeVisitor &visitor) {
    visitor.on_attribute("output_shape", output_shape_);
    visitor.on_attribute("keep_scale_on_reshape", keep_scale_on_reshape_);
    visitor.on_attribute("scale", scale_);
    return true;
}

