// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//! [fft_op:implementation]
#include "warp_affine_op.hpp"

using namespace CustomLayers;

constexpr ngraph::NodeTypeInfo WarpAffineOp::type_info;

WarpAffineOp::WarpAffineOp(const ngraph::Output<ngraph::Node>& image,
        const ngraph::Output<ngraph::Node>& transformation_matrix,
        const ngraph::Output<ngraph::Node>& output_shape) : Op({image, output_shape}){
    constructor_validate_and_infer_types();
}

void WarpAffineOp::validate_and_infer_types() {
    ngraph::PartialShape output_shape = ngraph::PartialShape(get_output_partial_shape(0));
    set_output_type(0, get_input_element_type(0), output_shape);
}

std::shared_ptr<ngraph::Node> WarpAffineOp::clone_with_new_inputs(const ngraph::OutputVector &new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<WarpAffineOp>(new_args.at(0), new_args.at(1), new_args.at(2));
}

bool WarpAffineOp::visit_attributes(ngraph::AttributeVisitor &visitor) {
    return true;
}

