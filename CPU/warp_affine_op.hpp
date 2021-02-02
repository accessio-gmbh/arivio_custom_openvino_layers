#pragma once

#include <ngraph/ngraph.hpp>

namespace CustomLayers {
    class WarpAffineOp : public ngraph::op::Op {
        public:
            static constexpr ngraph::NodeTypeInfo type_info{"WarpAffine", 0};
            const ngraph::NodeTypeInfo& get_type_info() const override { return type_info;  }

            WarpAffineOp() = default;
            WarpAffineOp(const ngraph::Output<ngraph::Node>& image,
                         const ngraph::Output<ngraph::Node>& transformation_matrix,
                         const ngraph::Output<ngraph::Node>& output_shape);
            void validate_and_infer_types() override;
            std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;
            bool visit_attributes(ngraph::AttributeVisitor& visitor) override;
    };
}
//! [fft_op:header]

