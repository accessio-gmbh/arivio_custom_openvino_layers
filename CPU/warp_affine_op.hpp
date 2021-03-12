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
                         const ngraph::PartialShape& output_shape,
                         bool keep_scale_on_reshape,
                         const std::vector<float>& scale);
            void validate_and_infer_types() override;
            std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;
            bool visit_attributes(ngraph::AttributeVisitor& visitor) override;
        protected:
            ngraph::PartialShape output_shape_;
            bool keep_scale_on_reshape_;
            std::vector<float> scale_;
    };
}
//! [fft_op:header]

