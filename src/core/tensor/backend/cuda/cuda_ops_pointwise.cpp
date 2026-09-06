/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "../facade_trace.hpp"
#include "../gpu_backend_ops.hpp"
#include "../pointwise_lowering.hpp"

#include "../../internal/tensor_impl.hpp"
#include "core/assert.hpp"
#include "core/tensor/backend/cuda/kernels/tensor_ops.hpp"

#include <cstdint>
#include <cuda_fp16.h>
#include <vector>

namespace lfs::core::internal {
    namespace {

        template <class T>
        T* cuda_pointer(const StorageRef storage) {
            LFS_ASSERT_MSG(storage.backend == GpuBackend::CUDA,
                           "CUDA pointwise adapter received non-CUDA storage");
            return reinterpret_cast<T*>(
                static_cast<unsigned char*>(storage.data) + storage.byte_offset);
        }

        template <class T>
        const T* cuda_const_pointer(const StorageRef storage) {
            return cuda_pointer<const T>(storage);
        }

        void validate_program(const PointwiseProgram& program,
                              const StorageRef input, const StorageRef output) {
            LFS_ASSERT_MSG(input.dtype == program.in_dtype,
                           "pointwise program input dtype does not match storage");
            LFS_ASSERT_MSG(output.dtype == program.out_dtype,
                           "pointwise program output dtype does not match storage");
        }

        template <class Op>
        void launch_unary_same(const PointwiseProgram& program,
                               const StorageRef input, const StorageRef output,
                               const size_t count, const ExecContext context,
                               const Op op = {}) {
            LFS_ASSERT_MSG(program.in_dtype == program.out_dtype,
                           "same-type unary operation requires matching dtypes");
            switch (program.in_dtype) {
            case DataType::Float32:
                tensor_ops::launch_unary_op_generic(
                    cuda_const_pointer<float>(input), cuda_pointer<float>(output),
                    count, op, context.cuda_stream);
                return;
            case DataType::Int32:
                tensor_ops::launch_unary_op_generic(
                    cuda_const_pointer<int>(input), cuda_pointer<int>(output),
                    count, op, context.cuda_stream);
                return;
            default:
                LFS_ASSERT_MSG(false, "unary op/dtype pair has no CUDA instantiation");
            }
        }

        template <class Op>
        void launch_unary_bool(const PointwiseProgram& program,
                               const StorageRef input, const StorageRef output,
                               const size_t count, const ExecContext context,
                               const Op op = {}) {
            LFS_ASSERT_MSG(program.out_dtype == DataType::Bool,
                           "predicate unary operation requires Bool output");
            switch (program.in_dtype) {
            case DataType::Float32:
                tensor_ops::launch_unary_op_generic(
                    cuda_const_pointer<float>(input), cuda_pointer<unsigned char>(output),
                    count, op, context.cuda_stream);
                return;
            case DataType::Int32:
                tensor_ops::launch_unary_op_generic(
                    cuda_const_pointer<int>(input), cuda_pointer<unsigned char>(output),
                    count, op, context.cuda_stream);
                return;
            case DataType::UInt8:
            case DataType::Bool:
                tensor_ops::launch_unary_op_generic(
                    cuda_const_pointer<unsigned char>(input),
                    cuda_pointer<unsigned char>(output), count, op, context.cuda_stream);
                return;
            default:
                LFS_ASSERT_MSG(false, "predicate unary op/dtype pair has no CUDA instantiation");
            }
        }

        template <class Op>
        void launch_binary_same(const PointwiseProgram& program,
                                const StorageRef lhs, const StorageRef rhs,
                                const StorageRef output, const size_t count,
                                const ExecContext context, const Op op = {}) {
            LFS_ASSERT_MSG(program.in_dtype == program.out_dtype,
                           "same-type binary operation requires matching dtypes");
            switch (program.in_dtype) {
            case DataType::Float32:
                tensor_ops::launch_binary_op_generic(
                    cuda_const_pointer<float>(lhs), cuda_const_pointer<float>(rhs),
                    cuda_pointer<float>(output), count, op, context.cuda_stream);
                return;
            case DataType::Float16:
                tensor_ops::launch_binary_op_generic(
                    cuda_const_pointer<__half>(lhs), cuda_const_pointer<__half>(rhs),
                    cuda_pointer<__half>(output), count, op, context.cuda_stream);
                return;
            case DataType::Int32:
                tensor_ops::launch_binary_op_generic(
                    cuda_const_pointer<int>(lhs), cuda_const_pointer<int>(rhs),
                    cuda_pointer<int>(output), count, op, context.cuda_stream);
                return;
            case DataType::Int64:
                tensor_ops::launch_binary_op_generic(
                    cuda_const_pointer<int64_t>(lhs), cuda_const_pointer<int64_t>(rhs),
                    cuda_pointer<int64_t>(output), count, op, context.cuda_stream);
                return;
            case DataType::UInt8:
                tensor_ops::launch_binary_op_generic(
                    cuda_const_pointer<uint8_t>(lhs), cuda_const_pointer<uint8_t>(rhs),
                    cuda_pointer<uint8_t>(output), count, op, context.cuda_stream);
                return;
            default:
                LFS_ASSERT_MSG(false, "binary op/dtype pair has no CUDA instantiation");
            }
        }

        template <class Op>
        void launch_binary_bool(const PointwiseProgram& program,
                                const StorageRef lhs, const StorageRef rhs,
                                const StorageRef output, const size_t count,
                                const ExecContext context, const Op op = {}) {
            LFS_ASSERT_MSG(program.out_dtype == DataType::Bool,
                           "predicate binary operation requires Bool output");
            switch (program.in_dtype) {
            case DataType::Float32:
                tensor_ops::launch_binary_op_generic(
                    cuda_const_pointer<float>(lhs), cuda_const_pointer<float>(rhs),
                    cuda_pointer<unsigned char>(output), count, op, context.cuda_stream);
                return;
            case DataType::Float16:
                tensor_ops::launch_binary_op_generic(
                    cuda_const_pointer<__half>(lhs), cuda_const_pointer<__half>(rhs),
                    cuda_pointer<unsigned char>(output), count, op, context.cuda_stream);
                return;
            case DataType::Int32:
                tensor_ops::launch_binary_op_generic(
                    cuda_const_pointer<int>(lhs), cuda_const_pointer<int>(rhs),
                    cuda_pointer<unsigned char>(output), count, op, context.cuda_stream);
                return;
            case DataType::Int64:
                tensor_ops::launch_binary_op_generic(
                    cuda_const_pointer<int64_t>(lhs), cuda_const_pointer<int64_t>(rhs),
                    cuda_pointer<unsigned char>(output), count, op, context.cuda_stream);
                return;
            case DataType::UInt8:
            case DataType::Bool:
                tensor_ops::launch_binary_op_generic(
                    cuda_const_pointer<unsigned char>(lhs),
                    cuda_const_pointer<unsigned char>(rhs),
                    cuda_pointer<unsigned char>(output), count, op, context.cuda_stream);
                return;
            default:
                LFS_ASSERT_MSG(false, "predicate binary op/dtype pair has no CUDA instantiation");
            }
        }

        template <class Op>
        void launch_broadcast_same(const PointwiseProgram& program,
                                   const StorageRef lhs, const StridedLayout& lhs_layout,
                                   const StorageRef rhs, const StridedLayout& rhs_layout,
                                   const StorageRef output, const StridedLayout& output_layout,
                                   const ExecContext context, const Op op = {}) {
            LFS_ASSERT_MSG(program.in_dtype == program.out_dtype,
                           "same-type broadcast operation requires matching dtypes");
#define LFS_LAUNCH_BROADCAST_SAME(Type, Dtype)                            \
    case DataType::Dtype:                                                 \
        tensor_ops::launch_broadcast_binary(                              \
            cuda_const_pointer<Type>(lhs), cuda_const_pointer<Type>(rhs), \
            cuda_pointer<Type>(output), lhs_layout.dims.data(),           \
            rhs_layout.dims.data(), output_layout.dims.data(),            \
            lhs_layout.rank, rhs_layout.rank, output_layout.rank,         \
            output_layout.element_count, op, context.cuda_stream);        \
        return
            switch (program.in_dtype) {
                LFS_LAUNCH_BROADCAST_SAME(float, Float32);
                LFS_LAUNCH_BROADCAST_SAME(__half, Float16);
                LFS_LAUNCH_BROADCAST_SAME(int, Int32);
                LFS_LAUNCH_BROADCAST_SAME(int64_t, Int64);
                LFS_LAUNCH_BROADCAST_SAME(uint8_t, UInt8);
            default:
                LFS_ASSERT_MSG(false, "broadcast op/dtype pair has no CUDA instantiation");
            }
#undef LFS_LAUNCH_BROADCAST_SAME
        }

        template <class Op>
        void launch_broadcast_bool(const PointwiseProgram& program,
                                   const StorageRef lhs, const StridedLayout& lhs_layout,
                                   const StorageRef rhs, const StridedLayout& rhs_layout,
                                   const StorageRef output, const StridedLayout& output_layout,
                                   const ExecContext context, const Op op = {}) {
            LFS_ASSERT_MSG(program.out_dtype == DataType::Bool,
                           "predicate broadcast operation requires Bool output");
#define LFS_LAUNCH_BROADCAST_BOOL(Type, Dtype)                            \
    case DataType::Dtype:                                                 \
        tensor_ops::launch_broadcast_binary(                              \
            cuda_const_pointer<Type>(lhs), cuda_const_pointer<Type>(rhs), \
            cuda_pointer<unsigned char>(output), lhs_layout.dims.data(),  \
            rhs_layout.dims.data(), output_layout.dims.data(),            \
            lhs_layout.rank, rhs_layout.rank, output_layout.rank,         \
            output_layout.element_count, op, context.cuda_stream);        \
        return
            switch (program.in_dtype) {
                LFS_LAUNCH_BROADCAST_BOOL(float, Float32);
                LFS_LAUNCH_BROADCAST_BOOL(__half, Float16);
                LFS_LAUNCH_BROADCAST_BOOL(int, Int32);
                LFS_LAUNCH_BROADCAST_BOOL(int64_t, Int64);
                LFS_LAUNCH_BROADCAST_BOOL(unsigned char, UInt8);
                LFS_LAUNCH_BROADCAST_BOOL(unsigned char, Bool);
            default:
                LFS_ASSERT_MSG(false, "predicate broadcast op/dtype pair has no CUDA instantiation");
            }
#undef LFS_LAUNCH_BROADCAST_BOOL
        }

        template <class BinaryOp, class Scalar>
        void launch_scalar_right_same(const PointwiseProgram& program,
                                      const StorageRef input, const StorageRef output,
                                      const size_t count, const ExecContext context,
                                      const Scalar scalar) {
            const ops::scalar_right_op<BinaryOp, Scalar> op(scalar);
            launch_unary_same(program, input, output, count, context, op);
        }

        template <class BinaryOp, class Scalar>
        void launch_scalar_left_same(const PointwiseProgram& program,
                                     const StorageRef input, const StorageRef output,
                                     const size_t count, const ExecContext context,
                                     const Scalar scalar) {
            const ops::scalar_left_op<BinaryOp, Scalar> op(scalar);
            launch_unary_same(program, input, output, count, context, op);
        }

        template <class BinaryOp, class Scalar>
        void launch_scalar_right_bool(const PointwiseProgram& program,
                                      const StorageRef input, const StorageRef output,
                                      const size_t count, const ExecContext context,
                                      const Scalar scalar) {
            const ops::scalar_right_op<BinaryOp, Scalar> op(scalar);
            launch_unary_bool(program, input, output, count, context, op);
        }

        template <class BinaryOp, bool SupportsScalarLeft = true>
        void dispatch_scalar_arithmetic(const PointwiseProgram& program,
                                        const StorageRef input, const StorageRef output,
                                        const size_t count, const ExecContext context) {
            if (program.scalar.scalar_on_right) {
                if (program.scalar.kind == ScalarKind::Float) {
                    launch_scalar_right_same<BinaryOp>(
                        program, input, output, count, context,
                        program.scalar.value.float_value);
                    return;
                }
                if (program.scalar.kind == ScalarKind::Int32) {
                    launch_scalar_right_same<BinaryOp>(
                        program, input, output, count, context,
                        program.scalar.value.int32_value);
                    return;
                }
            } else if constexpr (SupportsScalarLeft) {
                if (program.scalar.kind == ScalarKind::Float) {
                    launch_scalar_left_same<BinaryOp>(
                        program, input, output, count, context,
                        program.scalar.value.float_value);
                    return;
                }
            }
            LFS_ASSERT_MSG(false, "scalar arithmetic op has no CUDA instantiation");
        }

        template <class BinaryOp>
        void dispatch_scalar_integer_only(const PointwiseProgram& program,
                                          const StorageRef input, const StorageRef output,
                                          const size_t count, const ExecContext context) {
            LFS_ASSERT_MSG(program.scalar.scalar_on_right &&
                               program.scalar.kind == ScalarKind::Int32,
                           "scalar op has no CUDA instantiation for this scalar type");
            launch_scalar_right_same<BinaryOp>(
                program, input, output, count, context,
                program.scalar.value.int32_value);
        }

        template <class BinaryOp>
        void dispatch_scalar_comparison(const PointwiseProgram& program,
                                        const StorageRef input, const StorageRef output,
                                        const size_t count, const ExecContext context) {
            LFS_ASSERT_MSG(program.scalar.scalar_on_right,
                           "left scalar comparison has no CUDA instantiation");
            if (program.scalar.kind == ScalarKind::Float) {
                launch_scalar_right_bool<BinaryOp>(
                    program, input, output, count, context,
                    program.scalar.value.float_value);
                return;
            }
            if (program.scalar.kind == ScalarKind::Int32) {
                launch_scalar_right_bool<BinaryOp>(
                    program, input, output, count, context,
                    program.scalar.value.int32_value);
                return;
            }
            LFS_ASSERT_MSG(false, "scalar comparison has no CUDA instantiation");
        }

        template <class Op>
        void launch_direct_scalar(const PointwiseProgram& program,
                                  const StorageRef input, const StorageRef output,
                                  const size_t count, const ExecContext context,
                                  const Op op = {}) {
            LFS_ASSERT_MSG(program.in_dtype == DataType::Float32 &&
                               program.out_dtype == DataType::Float32 &&
                               program.scalar.kind == ScalarKind::Float,
                           "direct scalar op/dtype pair has no CUDA instantiation");
            tensor_ops::launch_scalar_op_generic(
                cuda_const_pointer<float>(input), program.scalar.value.float_value,
                cuda_pointer<float>(output), count, op, context.cuda_stream);
        }

        template <class Src, class Dst>
        void launch_conversion(const StorageRef input, const StorageRef output,
                               const size_t count, const ExecContext context) {
            tensor_ops::launch_convert_type(
                cuda_const_pointer<Src>(input), cuda_pointer<Dst>(output),
                count, context.cuda_stream);
        }

        template <class T>
        void launch_strided_fill(const StorageRef output,
                                 const StridedLayout& layout, const T value,
                                 const ExecContext context) {
            const std::vector<size_t> shape(
                layout.dims.begin(), layout.dims.begin() + layout.rank);
            const std::vector<size_t> strides(
                layout.strides.begin(), layout.strides.begin() + layout.rank);
            tensor_ops::launch_fill_strided(
                cuda_pointer<T>(output), value, shape, strides, 0,
                layout.element_count, context.cuda_stream);
        }

    } // namespace

    CudaBackendOps::~CudaBackendOps() = default;

    void CudaBackendOps::unary(const PointwiseProgram& program,
                               const StorageRef input, const StorageRef output,
                               const size_t count, const ExecContext context) {
        LFS_FACADE_TRACE(unary);
        validate_program(program, input, output);
        switch (program.op) {
#define LFS_UNARY_SAME_CASE(Id, Functor)                                      \
    case PointwiseOp::Id:                                                     \
        launch_unary_same(program, input, output, count, context, Functor{}); \
        return
            LFS_UNARY_SAME_CASE(Abs, ops::abs_op);
            LFS_UNARY_SAME_CASE(Neg, ops::neg_op);
            LFS_UNARY_SAME_CASE(Exp, ops::exp_op);
            LFS_UNARY_SAME_CASE(Log, ops::log_op);
            LFS_UNARY_SAME_CASE(Sqrt, ops::sqrt_op);
            LFS_UNARY_SAME_CASE(Sigmoid, ops::sigmoid_op);
            LFS_UNARY_SAME_CASE(Relu, ops::relu_op);
            LFS_UNARY_SAME_CASE(Square, ops::square_op);
            LFS_UNARY_SAME_CASE(Tanh, ops::tanh_op);
            LFS_UNARY_SAME_CASE(Rsqrt, ops::rsqrt_op);
            LFS_UNARY_SAME_CASE(Sign, ops::sign_op);
            LFS_UNARY_SAME_CASE(Reciprocal, ops::reciprocal_op);
            LFS_UNARY_SAME_CASE(Floor, ops::floor_op);
            LFS_UNARY_SAME_CASE(Ceil, ops::ceil_op);
            LFS_UNARY_SAME_CASE(Exp2, ops::exp2_op);
            LFS_UNARY_SAME_CASE(Log2, ops::log2_op);
            LFS_UNARY_SAME_CASE(Log10, ops::log10_op);
            LFS_UNARY_SAME_CASE(Log1p, ops::log1p_op);
            LFS_UNARY_SAME_CASE(Sin, ops::sin_op);
            LFS_UNARY_SAME_CASE(Cos, ops::cos_op);
            LFS_UNARY_SAME_CASE(Tan, ops::tan_op);
            LFS_UNARY_SAME_CASE(Asin, ops::asin_op);
            LFS_UNARY_SAME_CASE(Acos, ops::acos_op);
            LFS_UNARY_SAME_CASE(Atan, ops::atan_op);
            LFS_UNARY_SAME_CASE(Sinh, ops::sinh_op);
            LFS_UNARY_SAME_CASE(Cosh, ops::cosh_op);
            LFS_UNARY_SAME_CASE(Gelu, ops::gelu_op);
            LFS_UNARY_SAME_CASE(Swish, ops::swish_op);
            LFS_UNARY_SAME_CASE(Trunc, ops::trunc_op);
#undef LFS_UNARY_SAME_CASE
        case PointwiseOp::Round:
            if (program.in_dtype == DataType::Float32 &&
                program.out_dtype == DataType::Float32) {
                tensor_ops::launch_ieee_round_float(
                    cuda_const_pointer<float>(input), cuda_pointer<float>(output),
                    count, context.cuda_stream);
                return;
            }
            launch_unary_same(program, input, output, count, context, ops::round_op{});
            return;
#define LFS_UNARY_BOOL_CASE(Id, Functor)                                      \
    case PointwiseOp::Id:                                                     \
        launch_unary_bool(program, input, output, count, context, Functor{}); \
        return
            LFS_UNARY_BOOL_CASE(IsNan, ops::isnan_op);
            LFS_UNARY_BOOL_CASE(IsInf, ops::isinf_op);
            LFS_UNARY_BOOL_CASE(IsFinite, ops::isfinite_op);
            LFS_UNARY_BOOL_CASE(LogicalNot, ops::logical_not_op);
#undef LFS_UNARY_BOOL_CASE
        case PointwiseOp::AddScalar:
            dispatch_scalar_arithmetic<ops::add_op>(program, input, output, count, context);
            return;
        case PointwiseOp::SubScalar:
            dispatch_scalar_arithmetic<ops::sub_op>(program, input, output, count, context);
            return;
        case PointwiseOp::MulScalar:
            dispatch_scalar_arithmetic<ops::mul_op>(program, input, output, count, context);
            return;
        case PointwiseOp::DivScalar:
            dispatch_scalar_arithmetic<ops::div_op>(program, input, output, count, context);
            return;
        case PointwiseOp::PowScalar:
            dispatch_scalar_arithmetic<ops::pow_op, false>(program, input, output, count, context);
            return;
        case PointwiseOp::ModScalar:
            dispatch_scalar_arithmetic<ops::mod_op, false>(program, input, output, count, context);
            return;
        case PointwiseOp::MaximumScalar:
            dispatch_scalar_integer_only<ops::maximum_op>(program, input, output, count, context);
            return;
        case PointwiseOp::MinimumScalar:
            dispatch_scalar_integer_only<ops::minimum_op>(program, input, output, count, context);
            return;
        case PointwiseOp::EqualScalar:
            dispatch_scalar_comparison<ops::equal_op>(program, input, output, count, context);
            return;
        case PointwiseOp::NotEqualScalar:
            dispatch_scalar_comparison<ops::not_equal_op>(program, input, output, count, context);
            return;
        case PointwiseOp::LessScalar:
            dispatch_scalar_comparison<ops::less_op>(program, input, output, count, context);
            return;
        case PointwiseOp::LessEqualScalar:
            dispatch_scalar_comparison<ops::less_equal_op>(program, input, output, count, context);
            return;
        case PointwiseOp::GreaterScalar:
            dispatch_scalar_comparison<ops::greater_op>(program, input, output, count, context);
            return;
        case PointwiseOp::GreaterEqualScalar:
            dispatch_scalar_comparison<ops::greater_equal_op>(program, input, output, count, context);
            return;
        case PointwiseOp::ExpThenMulScalar: {
            LFS_ASSERT_MSG(program.scalar.kind == ScalarKind::Float,
                           "fused exp/mul requires a float scalar");
            const pointwise_exp_mul_functor op(
                ops::exp_op{}, pointwise_mul_scalar_functor(program.scalar.value.float_value));
            launch_unary_same(program, input, output, count, context, op);
            return;
        }
        case PointwiseOp::MulScalarThenAbs: {
            LFS_ASSERT_MSG(program.scalar.kind == ScalarKind::Float,
                           "fused mul/abs requires a float scalar");
            const pointwise_mul_abs_functor op(
                pointwise_mul_scalar_functor(program.scalar.value.float_value), ops::abs_op{});
            launch_unary_same(program, input, output, count, context, op);
            return;
        }
        case PointwiseOp::MulScalarThenRelu: {
            LFS_ASSERT_MSG(program.scalar.kind == ScalarKind::Float,
                           "fused mul/relu requires a float scalar");
            const pointwise_mul_relu_functor op(
                pointwise_mul_scalar_functor(program.scalar.value.float_value), ops::relu_op{});
            launch_unary_same(program, input, output, count, context, op);
            return;
        }
        default:
            LFS_ASSERT_MSG(false, "pointwise op is not a CUDA unary operation");
        }
    }

    void CudaBackendOps::binary(const PointwiseProgram& program,
                                const StorageRef lhs, const StorageRef rhs,
                                const StorageRef output, const size_t count,
                                const ExecContext context) {
        LFS_FACADE_TRACE(binary);
        validate_program(program, lhs, output);
        LFS_ASSERT_MSG(rhs.dtype == program.in_dtype,
                       "binary pointwise rhs dtype does not match program");
        switch (program.op) {
#define LFS_BINARY_SAME_CASE(Id, Functor)                                         \
    case PointwiseOp::Id:                                                         \
        launch_binary_same(program, lhs, rhs, output, count, context, Functor{}); \
        return
            LFS_BINARY_SAME_CASE(AddTensor, ops::add_op);
            LFS_BINARY_SAME_CASE(SubTensor, ops::sub_op);
            LFS_BINARY_SAME_CASE(MulTensor, ops::mul_op);
            LFS_BINARY_SAME_CASE(DivTensor, ops::div_op);
            LFS_BINARY_SAME_CASE(PowTensor, ops::pow_op);
            LFS_BINARY_SAME_CASE(ModTensor, ops::mod_op);
#undef LFS_BINARY_SAME_CASE
        case PointwiseOp::MaximumTensor:
            if (program.in_dtype == DataType::Float32) {
                tensor_ops::launch_ieee_maximum_float(
                    cuda_const_pointer<float>(lhs), cuda_const_pointer<float>(rhs),
                    cuda_pointer<float>(output), count, context.cuda_stream);
                return;
            }
            launch_binary_same(
                program, lhs, rhs, output, count, context, ops::maximum_op{});
            return;
        case PointwiseOp::MinimumTensor:
            if (program.in_dtype == DataType::Float32) {
                tensor_ops::launch_ieee_minimum_float(
                    cuda_const_pointer<float>(lhs), cuda_const_pointer<float>(rhs),
                    cuda_pointer<float>(output), count, context.cuda_stream);
                return;
            }
            launch_binary_same(
                program, lhs, rhs, output, count, context, ops::minimum_op{});
            return;
#define LFS_BINARY_BOOL_CASE(Id, Functor)                                         \
    case PointwiseOp::Id:                                                         \
        launch_binary_bool(program, lhs, rhs, output, count, context, Functor{}); \
        return
            LFS_BINARY_BOOL_CASE(EqualTensor, ops::equal_op);
            LFS_BINARY_BOOL_CASE(NotEqualTensor, ops::not_equal_op);
            LFS_BINARY_BOOL_CASE(LessTensor, ops::less_op);
            LFS_BINARY_BOOL_CASE(LessEqualTensor, ops::less_equal_op);
            LFS_BINARY_BOOL_CASE(GreaterTensor, ops::greater_op);
            LFS_BINARY_BOOL_CASE(GreaterEqualTensor, ops::greater_equal_op);
            LFS_BINARY_BOOL_CASE(LogicalAndTensor, ops::logical_and_op);
            LFS_BINARY_BOOL_CASE(LogicalOrTensor, ops::logical_or_op);
            LFS_BINARY_BOOL_CASE(LogicalXorTensor, ops::logical_xor_op);
#undef LFS_BINARY_BOOL_CASE
        default:
            LFS_ASSERT_MSG(false, "pointwise op is not a CUDA binary operation");
        }
    }

    void CudaBackendOps::scalar(const PointwiseProgram& program,
                                const StorageRef input, const StorageRef output,
                                const size_t count, const ExecContext context) {
        LFS_FACADE_TRACE(scalar);
        validate_program(program, input, output);
        switch (program.op) {
        case PointwiseOp::AddTensor:
        case PointwiseOp::AddScalar:
            launch_direct_scalar(program, input, output, count, context, ops::add_op{});
            return;
        case PointwiseOp::SubTensor:
        case PointwiseOp::SubScalar:
            launch_direct_scalar(program, input, output, count, context, ops::sub_op{});
            return;
        case PointwiseOp::MulTensor:
        case PointwiseOp::MulScalar:
            launch_direct_scalar(program, input, output, count, context, ops::mul_op{});
            return;
        case PointwiseOp::DivTensor:
        case PointwiseOp::DivScalar:
            launch_direct_scalar(program, input, output, count, context, ops::div_op{});
            return;
        default:
            LFS_ASSERT_MSG(false, "pointwise op is not a direct CUDA scalar operation");
        }
    }

    void CudaBackendOps::broadcast_binary(
        const PointwiseProgram& program,
        const StorageRef lhs, const StridedLayout& lhs_layout,
        const StorageRef rhs, const StridedLayout& rhs_layout,
        const StorageRef output, const StridedLayout& output_layout,
        const ExecContext context) {
        LFS_FACADE_TRACE(broadcast_binary);
        validate_program(program, lhs, output);
        LFS_ASSERT_MSG(rhs.dtype == program.in_dtype,
                       "broadcast rhs dtype does not match pointwise program");
        switch (program.op) {
#define LFS_BROADCAST_SAME_CASE(Id, Functor)                                     \
    case PointwiseOp::Id:                                                        \
        launch_broadcast_same(program, lhs, lhs_layout, rhs, rhs_layout, output, \
                              output_layout, context, Functor{});                \
        return
            LFS_BROADCAST_SAME_CASE(AddTensor, ops::add_op);
            LFS_BROADCAST_SAME_CASE(SubTensor, ops::sub_op);
            LFS_BROADCAST_SAME_CASE(MulTensor, ops::mul_op);
            LFS_BROADCAST_SAME_CASE(DivTensor, ops::div_op);
            LFS_BROADCAST_SAME_CASE(PowTensor, ops::pow_op);
            LFS_BROADCAST_SAME_CASE(ModTensor, ops::mod_op);
#undef LFS_BROADCAST_SAME_CASE
        case PointwiseOp::MaximumTensor:
            if (program.in_dtype == DataType::Float32) {
                tensor_ops::launch_ieee_maximum_float_broadcast(
                    cuda_const_pointer<float>(lhs), cuda_const_pointer<float>(rhs),
                    cuda_pointer<float>(output), lhs_layout.dims.data(),
                    rhs_layout.dims.data(), output_layout.dims.data(),
                    lhs_layout.rank, rhs_layout.rank, output_layout.rank,
                    output_layout.element_count, context.cuda_stream);
                return;
            }
            launch_broadcast_same(
                program, lhs, lhs_layout, rhs, rhs_layout, output, output_layout,
                context, ops::maximum_op{});
            return;
        case PointwiseOp::MinimumTensor:
            if (program.in_dtype == DataType::Float32) {
                tensor_ops::launch_ieee_minimum_float_broadcast(
                    cuda_const_pointer<float>(lhs), cuda_const_pointer<float>(rhs),
                    cuda_pointer<float>(output), lhs_layout.dims.data(),
                    rhs_layout.dims.data(), output_layout.dims.data(),
                    lhs_layout.rank, rhs_layout.rank, output_layout.rank,
                    output_layout.element_count, context.cuda_stream);
                return;
            }
            launch_broadcast_same(
                program, lhs, lhs_layout, rhs, rhs_layout, output, output_layout,
                context, ops::minimum_op{});
            return;
#define LFS_BROADCAST_BOOL_CASE(Id, Functor)                                     \
    case PointwiseOp::Id:                                                        \
        launch_broadcast_bool(program, lhs, lhs_layout, rhs, rhs_layout, output, \
                              output_layout, context, Functor{});                \
        return
            LFS_BROADCAST_BOOL_CASE(EqualTensor, ops::equal_op);
            LFS_BROADCAST_BOOL_CASE(NotEqualTensor, ops::not_equal_op);
            LFS_BROADCAST_BOOL_CASE(LessTensor, ops::less_op);
            LFS_BROADCAST_BOOL_CASE(LessEqualTensor, ops::less_equal_op);
            LFS_BROADCAST_BOOL_CASE(GreaterTensor, ops::greater_op);
            LFS_BROADCAST_BOOL_CASE(GreaterEqualTensor, ops::greater_equal_op);
            LFS_BROADCAST_BOOL_CASE(LogicalAndTensor, ops::logical_and_op);
            LFS_BROADCAST_BOOL_CASE(LogicalOrTensor, ops::logical_or_op);
            LFS_BROADCAST_BOOL_CASE(LogicalXorTensor, ops::logical_xor_op);
#undef LFS_BROADCAST_BOOL_CASE
        default:
            LFS_ASSERT_MSG(false, "pointwise op is not a CUDA broadcast operation");
        }
    }

    void CudaBackendOps::fused_pointwise_chain(
        const StorageRef input, const StorageRef output, const size_t count,
        const tensor_ops::FusedPointwiseOpChain& chain,
        std::span<const StorageRef>,
        const ExecContext context) {
        LFS_FACADE_TRACE(fused_pointwise_chain);
        LFS_ASSERT_MSG(input.dtype == DataType::Float32 &&
                           output.dtype == DataType::Float32,
                       "fused pointwise chain supports only Float32");
        tensor_ops::launch_fused_pointwise_chain(
            cuda_const_pointer<float>(input), cuda_pointer<float>(output),
            count, chain, context.cuda_stream);
    }

    void CudaBackendOps::clamp_scalar(
        const StorageRef data, const ScalarOperand minimum,
        const ScalarOperand maximum, const size_t count,
        const ExecContext context) {
        LFS_FACADE_TRACE(clamp_scalar);
        LFS_ASSERT_MSG(data.dtype == DataType::Float32 &&
                           minimum.kind == ScalarKind::Float &&
                           maximum.kind == ScalarKind::Float,
                       "floating clamp adapter requires Float32 operands");
        tensor_ops::launch_clamp_scalar(
            cuda_pointer<float>(data), minimum.value.float_value,
            maximum.value.float_value, count, context.cuda_stream);
    }

    void CudaBackendOps::clamp_fused(
        const StorageRef input, const StorageRef output,
        const ScalarOperand minimum, const ScalarOperand maximum,
        const size_t count, const ExecContext context) {
        LFS_FACADE_TRACE(clamp_fused);
        LFS_ASSERT_MSG(input.dtype == DataType::Float32 &&
                           output.dtype == DataType::Float32 &&
                           minimum.kind == ScalarKind::Float &&
                           maximum.kind == ScalarKind::Float,
                       "fused clamp adapter requires Float32 operands");
        tensor_ops::launch_clamp_fused(
            cuda_const_pointer<float>(input), cuda_pointer<float>(output),
            minimum.value.float_value, maximum.value.float_value,
            count, context.cuda_stream);
    }

    void CudaBackendOps::clamp_scalar_int(
        const StorageRef data, const ScalarOperand minimum,
        const ScalarOperand maximum, const size_t count,
        const ExecContext context) {
        LFS_FACADE_TRACE(clamp_scalar_int);
        LFS_ASSERT_MSG(data.dtype == DataType::Int32 &&
                           minimum.kind == ScalarKind::Int32 &&
                           maximum.kind == ScalarKind::Int32,
                       "integer clamp adapter requires Int32 operands");
        tensor_ops::launch_clamp_scalar_int(
            cuda_pointer<int>(data), minimum.value.int32_value,
            maximum.value.int32_value, count, context.cuda_stream);
    }

    void CudaBackendOps::convert_type(
        const StorageRef input, const StorageRef output,
        const size_t count, const ExecContext context) {
        LFS_FACADE_TRACE(convert_type);
        switch (input.dtype) {
        case DataType::Float32:
            switch (output.dtype) {
            case DataType::Float16: launch_conversion<float, __half>(input, output, count, context); return;
            case DataType::Int32: launch_conversion<float, int>(input, output, count, context); return;
            case DataType::Int64: launch_conversion<float, int64_t>(input, output, count, context); return;
            case DataType::UInt8: launch_conversion<float, uint8_t>(input, output, count, context); return;
            case DataType::UInt32: launch_conversion<float, uint32_t>(input, output, count, context); return;
            default: break;
            }
            break;
        case DataType::Float16:
            switch (output.dtype) {
            case DataType::Float32: launch_conversion<__half, float>(input, output, count, context); return;
            case DataType::Int32: launch_conversion<__half, int>(input, output, count, context); return;
            case DataType::Int64: launch_conversion<__half, int64_t>(input, output, count, context); return;
            case DataType::UInt8: launch_conversion<__half, uint8_t>(input, output, count, context); return;
            default: break;
            }
            break;
        case DataType::Int32:
            switch (output.dtype) {
            case DataType::Float32: launch_conversion<int, float>(input, output, count, context); return;
            case DataType::Float16: launch_conversion<int, __half>(input, output, count, context); return;
            case DataType::Int64: launch_conversion<int, int64_t>(input, output, count, context); return;
            case DataType::UInt8: launch_conversion<int, uint8_t>(input, output, count, context); return;
            default: break;
            }
            break;
        case DataType::Int64:
            switch (output.dtype) {
            case DataType::Float32: launch_conversion<int64_t, float>(input, output, count, context); return;
            case DataType::Float16: launch_conversion<int64_t, __half>(input, output, count, context); return;
            case DataType::Int32: launch_conversion<int64_t, int>(input, output, count, context); return;
            case DataType::UInt8: launch_conversion<int64_t, uint8_t>(input, output, count, context); return;
            default: break;
            }
            break;
        case DataType::UInt8:
        case DataType::Bool:
            switch (output.dtype) {
            case DataType::Float32: launch_conversion<uint8_t, float>(input, output, count, context); return;
            case DataType::Float16: launch_conversion<uint8_t, __half>(input, output, count, context); return;
            case DataType::Int32: launch_conversion<uint8_t, int>(input, output, count, context); return;
            case DataType::Int64: launch_conversion<uint8_t, int64_t>(input, output, count, context); return;
            case DataType::Bool: launch_conversion<uint8_t, bool>(input, output, count, context); return;
            default: break;
            }
            break;
        case DataType::UInt32:
            switch (output.dtype) {
            case DataType::Float32: launch_conversion<uint32_t, float>(input, output, count, context); return;
            case DataType::Int64: launch_conversion<uint32_t, int64_t>(input, output, count, context); return;
            default: break;
            }
            break;
        }
        LFS_ASSERT_MSG(false, "dtype conversion pair has no CUDA instantiation");
    }

    void CudaBackendOps::fill_strided(
        const StorageRef output, const StridedLayout& layout,
        const ScalarOperand value, const ExecContext context) {
        LFS_FACADE_TRACE(fill_strided);
        LFS_ASSERT_MSG(layout.rank <= MAX_TENSOR_RANK,
                       "strided fill rank exceeds MAX_TENSOR_RANK");
        switch (output.dtype) {
        case DataType::Float32:
            LFS_ASSERT_MSG(value.kind == ScalarKind::Float,
                           "Float32 fill requires a float scalar");
            launch_strided_fill(output, layout, value.value.float_value, context);
            return;
        case DataType::Int32:
            LFS_ASSERT_MSG(value.kind == ScalarKind::Int32,
                           "Int32 fill requires an Int32 scalar");
            launch_strided_fill(output, layout, value.value.int32_value, context);
            return;
        case DataType::Bool:
            LFS_ASSERT_MSG(value.kind == ScalarKind::Bool,
                           "Bool fill requires a bool scalar");
            launch_strided_fill(
                output, layout,
                static_cast<unsigned char>(value.value.bool_value ? 1 : 0), context);
            return;
        default:
            LFS_ASSERT_MSG(false, "strided fill dtype has no CUDA instantiation");
        }
    }

    void CudaBackendOps::load_fill(
        const StorageRef output, const size_t count,
        const ScalarOperand value, const ExecContext context) {
        LFS_FACADE_TRACE(load_fill);
        LFS_ASSERT_MSG(output.dtype == DataType::Float32 &&
                           value.kind == ScalarKind::Float,
                       "CUDA load fill supports only Float32");
        const size_t shape[] = {count};
        const float fill_value = value.value.float_value;
        tensor_ops::launch_load_op(
            cuda_pointer<float>(output), shape, 1, LoadOp::Const,
            &fill_value, output.dtype, context.cuda_stream);
    }

} // namespace lfs::core::internal
