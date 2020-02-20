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

REGISTER_OP("FixedQuant")
        .Input("a: T")
        .Input("ipart: Tidx")
        .Input("fpart: Tidx")
        .Attr("T: numbertype")
        .Attr("Tidx: {int32, int64}")
        .Output("output: T")
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
            c->set_output(0, c->input(0));
            return Status::OK();
        });

template<typename dtype, typename idx>
class FixedQuantOp : public OpKernel {
    public:
        explicit FixedQuantOp(OpKernelConstruction *context) : OpKernel(context) {}

        void Compute(OpKernelContext *context) override {
            // Grab the input tensors
            const Tensor &input_tensor = context->input(0);
            auto input = input_tensor.flat<dtype>();

            // Grab n and es
            const Tensor &n_tensor = context->input(1);
            const Tensor &es_tensor = context->input(2);

            OP_REQUIRES(context, TensorShapeUtils::IsScalar(n_tensor.shape()),
                        errors::InvalidArgument(
                                "ipart must be a scalar, but received tensor of shape: ",
                                n_tensor.shape().DebugString()));
            OP_REQUIRES(context, TensorShapeUtils::IsScalar(es_tensor.shape()),
                        errors::InvalidArgument(
                                "fpart must be a scalar, but received tensor of shape: ",
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
            Tensor *output_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                             &output_tensor));
            auto output = output_tensor->flat<dtype>();

            layers_fixed::round_ftype<dtype, idx>(input, output, n, es, static_cast<idx>(input.size()));
        }
};

// Register kernel with types needed
#define REGISTER_KERNEL(dtype, idx)   \
    REGISTER_KERNEL_BUILDER(          \
        Name("FixedQuant")            \
        .Device(DEVICE_CPU)           \
        .TypeConstraint<dtype>("T")   \
        .TypeConstraint<idx>("Tidx"), \
        FixedQuantOp<dtype, idx>)

//REGISTER_KERNEL(int32, int32);
//REGISTER_KERNEL(int32, int64);
//REGISTER_KERNEL(int64, int32);
//REGISTER_KERNEL(int64, int64);
REGISTER_KERNEL(float, int32)
REGISTER_KERNEL(float, int64)
REGISTER_KERNEL(double, int32)
REGISTER_KERNEL(double, int64)

#undef REGISTER_KERNEL
