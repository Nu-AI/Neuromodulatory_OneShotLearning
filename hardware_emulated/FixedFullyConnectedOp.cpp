#include "tensorflow/core/public/version.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#if TF_MAJOR_VERSION == 1 && TF_MINOR_VERSION <= 12
// 1.12 and lower
#include "tensorflow/core/kernels/bounds_check.h"

#else
// 1.13 and above
#include "tensorflow/core/framework/bounds_check.h"

#endif

#include <cstdint>

#include "layers_fixed.h"

using namespace tensorflow;

REGISTER_OP("FixedFullyConnected")
        .Input("fc_input: T")
        .Input("fc_weight: T")
        .Input("bias_f: T")
        .Input("ipart: Tidx")
        .Input("fpart: Tidx")
        .Attr("T: numbertype")
        .Attr("Tidx: {int32, int64}")
        .Output("output: T")
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
            shape_inference::ShapeHandle a;
            TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &a));

            shape_inference::ShapeHandle b;
            TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &b));

            shape_inference::ShapeHandle d;
            TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &d));

            shape_inference::DimensionHandle output_rows = c->Dim(a, 0);
            shape_inference::DimensionHandle output_cols = c->Dim(b, 1);

            // Validate that the inner shapes are compatible.
            shape_inference::DimensionHandle inner_a = c->Dim(a, 1);
            shape_inference::DimensionHandle inner_b = c->Dim(b, 0);
            shape_inference::DimensionHandle merged_2d;
            TF_RETURN_IF_ERROR(c->Merge(inner_a, inner_b, &merged_2d));

            shape_inference::DimensionHandle merged_1d;
            TF_RETURN_IF_ERROR(c->Merge(output_cols, c->Dim(d, 0), &merged_1d));

            c->set_output(0, c->Matrix(output_rows, output_cols));
            return Status::OK();
        });

template<typename dtype, typename idx>
class FixedFullyConnectedOp : public OpKernel {
public:
    explicit FixedFullyConnectedOp(OpKernelConstruction *context) : OpKernel(context) {}

    void Compute(OpKernelContext *context) override {
        // Grab the input tensors
        const Tensor &fc_input_tensor = context->input(0);
        auto fc_input = fc_input_tensor.matrix<dtype>();
        const Tensor &fc_weight_tensor = context->input(1);
        auto fc_weight = fc_weight_tensor.matrix<dtype>();
        const Tensor &bias_tensor = context->input(2);
        auto bias = bias_tensor.vec<dtype>();

        OP_REQUIRES(context, fc_input_tensor.dims() == 2,
                    errors::InvalidArgument("fc_input must be 2-dimensional",
                                            fc_input_tensor.shape().DebugString()));
        OP_REQUIRES(context, fc_weight_tensor.dims() == 2,
                    errors::InvalidArgument("fc_weight must be 2-dimensional",
                                            fc_weight_tensor.shape().DebugString()));
        OP_REQUIRES(context, bias_tensor.dims() == 1,
                    errors::InvalidArgument("bias must be 1-dimensional",
                                            bias_tensor.shape().DebugString()));

        // Grab n and es
        const Tensor &n_tensor = context->input(3);
        const Tensor &es_tensor = context->input(4);

        OP_REQUIRES(context, TensorShapeUtils::IsScalar(n_tensor.shape()),
                    errors::InvalidArgument(
                            "n must be a scalar, but received tensor of shape: ",
                            n_tensor.shape().DebugString()));
        OP_REQUIRES(context, TensorShapeUtils::IsScalar(es_tensor.shape()),
                    errors::InvalidArgument(
                            "es must be a scalar, but received tensor of shape: ",
                            es_tensor.shape().DebugString()));

        auto n = internal::SubtleMustCopy(n_tensor.scalar<idx>()());
        auto es = internal::SubtleMustCopy(es_tensor.scalar<idx>()());

        OP_REQUIRES(context, n > 0,
                    errors::InvalidArgument("n must be > 0, received: ",
                                            n_tensor.DebugString()));
        OP_REQUIRES(context, es >= 0,
                    errors::InvalidArgument("es must be >= 0, received: ",
                                            es_tensor.DebugString()));

        // Create an output tensor
        idx n_output_row = fc_input_tensor.shape().dim_size(0);
        idx n_output_column = fc_weight_tensor.shape().dim_size(1);
        idx n_output_middle = fc_weight_tensor.shape().dim_size(0);

        Tensor *output_tensor = nullptr;
        TensorShape output_shape({n_output_row, n_output_column});
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,
                                                         &output_tensor));
        auto output = output_tensor->matrix<dtype>();

        // FC compute
        layers_fixed::fully_connected<dtype, idx>(
            fc_input, fc_weight, bias, output,
            n_output_row, n_output_column, n_output_middle, n, es);

//        OP_REQUIRES(context, n > es, errors::InvalidArgument("invalid ipart & fpart, received: ",
//                                                             n_tensor.DebugString(),
//                                                             es_tensor.DebugString()));
    }
};

// Register kernel with types needed
#define REGISTER_KERNEL(dtype, idx)   \
    REGISTER_KERNEL_BUILDER(          \
        Name("FixedFullyConnected")   \
        .Device(DEVICE_CPU)           \
        .TypeConstraint<dtype>("T")   \
        .TypeConstraint<idx>("Tidx"), \
        FixedFullyConnectedOp<dtype, idx>)

//REGISTER_KERNEL(int32, int32);
//REGISTER_KERNEL(int32, int64);
//REGISTER_KERNEL(int64, int32);
//REGISTER_KERNEL(int64, int64);
REGISTER_KERNEL(float, int32)
REGISTER_KERNEL(float, int64)
REGISTER_KERNEL(double, int32)
REGISTER_KERNEL(double, int64)

#undef REGISTER_KERNEL
