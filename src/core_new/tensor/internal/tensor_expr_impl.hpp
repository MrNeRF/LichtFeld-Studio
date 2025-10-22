/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

// This file contains template method implementations that require the full Tensor definition
// It should be included at the END of tensor.hpp, after Tensor class is fully defined

#include "tensor_expr.hpp"
#include "tensor_functors.hpp" // For ops::compose

namespace lfs::core {

    // ============================================================================
    // TensorLeaf implementation - Needs full Tensor definition
    // ============================================================================

    inline TensorLeaf::TensorLeaf(Tensor tensor)
        : tensor_ptr_(std::make_shared<Tensor>(std::move(tensor))) {}

    inline Tensor TensorLeaf::eval_impl() const {
        // Materialize tensors with storage offsets (sliced views) or non-contiguous layout
        // Expression templates use ptr<T>() which returns the base pointer without offset
        // So we need to materialize sliced tensors even if they're technically "contiguous"
        if (tensor_ptr_->storage_offset() != 0 || !tensor_ptr_->is_contiguous()) {
            return tensor_ptr_->contiguous();
        }
        return *tensor_ptr_; // Return copy of the materialized tensor
    }

    inline const TensorShape& TensorLeaf::shape_impl() const {
        return tensor_ptr_->shape();
    }

    inline Device TensorLeaf::device_impl() const {
        return tensor_ptr_->device();
    }

    inline DataType TensorLeaf::dtype_impl() const {
        return tensor_ptr_->dtype();
    }

    // ============================================================================
    // TensorExpr::eval() - Needs full Tensor definition
    // ============================================================================

    template <typename Derived>
    Tensor TensorExpr<Derived>::eval() const {
        return derived().eval_impl();
    }

    template <typename Derived>
    TensorExpr<Derived>::operator Tensor() const {
        return eval();
    }

    // ============================================================================
    // UnaryExpr::eval_impl() - Needs Tensor::empty() and Tensor methods
    // ============================================================================

    // Helper struct for eval_impl dispatch based on operation return type
    namespace detail {
        // Default implementation: float -> float or Int32 -> Int32 operations
        template <typename InputExpr, typename UnaryOp, bool ReturnsBool>
        struct UnaryExprEvaluator {
            static Tensor eval(const UnaryExpr<InputExpr, UnaryOp>& /* expr */,
                               const InputExpr& input, const UnaryOp& op,
                               const TensorShape& shape, Device device, DataType dtype) {
                // Recursively evaluate input expression
                Tensor input_tensor = input.eval();

                // Create result tensor (needs Tensor::empty)
                Tensor result = Tensor::empty(shape, device, dtype);

                // Check dtype to determine correct template instantiation
                if (input_tensor.dtype() == DataType::Int32) {
                    // Int32 -> Int32 operations (abs, neg, sign, etc.)
                    if (device == Device::CUDA) {
                        tensor_ops::launch_unary_op_generic(
                            input_tensor.template ptr<int>(),
                            result.template ptr<int>(),
                            result.numel(), op, nullptr);
                    } else {
                        // CPU fallback
                        const int* in_ptr = input_tensor.template ptr<int>();
                        int* out_ptr = result.template ptr<int>();
                        size_t n = result.numel();
                        for (size_t i = 0; i < n; ++i) {
                            out_ptr[i] = op(in_ptr[i]);
                        }
                    }
                } else {
                    // Float -> Float operations (default case)
                    if (device == Device::CUDA) {
                        tensor_ops::launch_unary_op_generic(
                            input_tensor.template ptr<float>(),
                            result.template ptr<float>(),
                            result.numel(), op, nullptr);
                    } else {
                        // CPU fallback: apply operation element-wise
                        const float* in_ptr = input_tensor.template ptr<float>();
                        float* out_ptr = result.template ptr<float>();
                        size_t n = result.numel();
                        for (size_t i = 0; i < n; ++i) {
                            out_ptr[i] = op(in_ptr[i]);
                        }
                    }
                }

                return result;
            }
        };

        // Specialized implementation: Bool-returning operations
        template <typename InputExpr, typename UnaryOp>
        struct UnaryExprEvaluator<InputExpr, UnaryOp, true> {
            static Tensor eval(const UnaryExpr<InputExpr, UnaryOp>& /* expr */,
                               const InputExpr& input, const UnaryOp& op,
                               const TensorShape& shape, Device device, DataType dtype) {
                // Recursively evaluate input expression
                Tensor input_tensor = input.eval();

                // Create result tensor (Bool dtype)
                Tensor result = Tensor::empty(shape, device, dtype);

                // Check input dtype to determine correct template instantiation
                if (input_tensor.dtype() == DataType::Bool) {
                    // Bool input -> Bool output (e.g., logical_not on Bool tensor)
                    if (device == Device::CUDA) {
                        tensor_ops::launch_unary_op_generic(
                            input_tensor.template ptr<unsigned char>(),
                            result.template ptr<unsigned char>(),
                            result.numel(), op, nullptr);
                    } else {
                        // CPU fallback
                        const unsigned char* in_ptr = input_tensor.template ptr<unsigned char>();
                        unsigned char* out_ptr = result.template ptr<unsigned char>();
                        size_t n = result.numel();
                        for (size_t i = 0; i < n; ++i) {
                            out_ptr[i] = op(in_ptr[i]);
                        }
                    }
                } else if (input_tensor.dtype() == DataType::Int32) {
                    // Int32 input -> Bool output (e.g., comparisons on Int32 tensor)
                    if (device == Device::CUDA) {
                        tensor_ops::launch_unary_op_generic(
                            input_tensor.template ptr<int>(),
                            result.template ptr<unsigned char>(),
                            result.numel(), op, nullptr);
                    } else {
                        // CPU fallback
                        const int* in_ptr = input_tensor.template ptr<int>();
                        unsigned char* out_ptr = result.template ptr<unsigned char>();
                        size_t n = result.numel();
                        for (size_t i = 0; i < n; ++i) {
                            out_ptr[i] = op(in_ptr[i]);
                        }
                    }
                } else {
                    // Float input -> Bool output (e.g., isnan, isinf, isfinite)
                    if (device == Device::CUDA) {
                        tensor_ops::launch_unary_op_generic(
                            input_tensor.template ptr<float>(),
                            result.template ptr<unsigned char>(),
                            result.numel(), op, nullptr);
                    } else {
                        // CPU fallback
                        const float* in_ptr = input_tensor.template ptr<float>();
                        unsigned char* out_ptr = result.template ptr<unsigned char>();
                        size_t n = result.numel();
                        for (size_t i = 0; i < n; ++i) {
                            out_ptr[i] = op(in_ptr[i]);
                        }
                    }
                }

                return result;
            }
        };
    } // namespace detail

    template <typename InputExpr, typename UnaryOp>
    Tensor UnaryExpr<InputExpr, UnaryOp>::eval_impl() const {
        // Dispatch to appropriate evaluator based on whether op returns Bool
        return detail::UnaryExprEvaluator<InputExpr, UnaryOp, ops::returns_bool_v<UnaryOp>>::eval(
            *this, input_, op_, shape_, device_, dtype_);
    }

    // ============================================================================
    // UnaryExpr specialization for fusion - Needs Tensor::empty()
    // ============================================================================

    template <typename InnerInput, typename InnerOp, typename OuterOp>
    Tensor UnaryExpr<UnaryExpr<InnerInput, InnerOp>, OuterOp>::eval_impl() const {
        // Get the innermost input and operations
        const auto& innermost_input = inner_expr_.input_;
        const auto& inner_op = inner_expr_.op_;

        // Compose the operations: outer(inner(x)) - AUTOMATIC FUSION!
        auto fused_op = ops::compose(inner_op, outer_op_);

        // Evaluate the innermost expression only
        Tensor base = innermost_input.eval();

        // Create result tensor
        Tensor result = Tensor::empty(shape_, device_, dtype_);

        // Apply fused operation in a single pass!
        if (device_ == Device::CUDA) {
            tensor_ops::launch_unary_op_generic(
                base.template ptr<float>(),
                result.template ptr<float>(),
                result.numel(), fused_op, nullptr);
        } else {
            // CPU fallback: apply fused operation element-wise
            const float* in_ptr = base.template ptr<float>();
            float* out_ptr = result.template ptr<float>();
            size_t n = result.numel();
            for (size_t i = 0; i < n; ++i) {
                out_ptr[i] = fused_op(in_ptr[i]);
            }
        }

        return result;
    }

    // ============================================================================
    // BinaryExpr::eval_impl() - Needs Tensor::empty() and methods
    // ============================================================================

    namespace detail {
        // Default implementation: float,float -> float or Int32,Int32 -> Int32 operations
        template <typename LeftExpr, typename RightExpr, typename BinaryOp, bool ReturnsBool>
        struct BinaryExprEvaluator {
            static Tensor eval(const BinaryExpr<LeftExpr, RightExpr, BinaryOp>& /* expr */,
                               const LeftExpr& left, const RightExpr& right, const BinaryOp& op,
                               const TensorShape& shape, Device device, DataType dtype) {
                // Evaluate both sides
                Tensor left_tensor = left.eval();
                Tensor right_tensor = right.eval();

                // Create result tensor
                Tensor result = Tensor::empty(shape, device, dtype);

                // Determine if broadcasting is needed
                bool needs_broadcast = (left_tensor.shape() != shape) ||
                                       (right_tensor.shape() != shape);

                // Check input dtypes to determine correct template instantiation
                if (left_tensor.dtype() == DataType::Int32 && right_tensor.dtype() == DataType::Int32) {
                    // Int32,Int32 -> Int32 operations (add, sub, mul, div, etc.)
                    if (device == Device::CUDA) {
                        if (needs_broadcast) {
                            tensor_ops::launch_broadcast_binary(
                                left_tensor.template ptr<int>(),
                                right_tensor.template ptr<int>(),
                                result.template ptr<int>(),
                                left_tensor.shape().dims().data(),
                                right_tensor.shape().dims().data(),
                                shape.dims().data(),
                                left_tensor.shape().rank(), right_tensor.shape().rank(), shape.rank(),
                                result.numel(), op, nullptr);
                        } else {
                            tensor_ops::launch_binary_op_generic(
                                left_tensor.template ptr<int>(),
                                right_tensor.template ptr<int>(),
                                result.template ptr<int>(),
                                result.numel(), op, nullptr);
                        }
                    } else {
                        // CPU fallback
                        if (!needs_broadcast) {
                            const int* left_ptr = left_tensor.template ptr<int>();
                            const int* right_ptr = right_tensor.template ptr<int>();
                            int* out_ptr = result.template ptr<int>();
                            size_t n = result.numel();
                            for (size_t i = 0; i < n; ++i) {
                                out_ptr[i] = op(left_ptr[i], right_ptr[i]);
                            }
                        } else {
                            Tensor left_broadcast = left_tensor;
                            Tensor right_broadcast = right_tensor;
                            if (left_tensor.shape() != shape) {
                                left_broadcast = left_tensor.broadcast_to(shape);
                            }
                            if (right_tensor.shape() != shape) {
                                right_broadcast = right_tensor.broadcast_to(shape);
                            }
                            const int* left_ptr = left_broadcast.template ptr<int>();
                            const int* right_ptr = right_broadcast.template ptr<int>();
                            int* out_ptr = result.template ptr<int>();
                            size_t n = result.numel();
                            for (size_t i = 0; i < n; ++i) {
                                out_ptr[i] = op(left_ptr[i], right_ptr[i]);
                            }
                        }
                    }
                } else {
                    // Float,Float -> Float operations (default case)
                    if (device == Device::CUDA) {
                        if (needs_broadcast) {
                            // Use broadcast binary kernel
                            tensor_ops::launch_broadcast_binary(
                                left_tensor.template ptr<float>(),
                                right_tensor.template ptr<float>(),
                                result.template ptr<float>(),
                                left_tensor.shape().dims().data(),
                                right_tensor.shape().dims().data(),
                                shape.dims().data(),
                                left_tensor.shape().rank(), right_tensor.shape().rank(), shape.rank(),
                                result.numel(), op, nullptr);
                        } else {
                            // Element-wise binary operation (no broadcasting)
                            tensor_ops::launch_binary_op_generic(
                                left_tensor.template ptr<float>(),
                                right_tensor.template ptr<float>(),
                                result.template ptr<float>(),
                                result.numel(), op, nullptr);
                        }
                    } else {
                        // CPU fallback: apply operation element-wise
                        if (!needs_broadcast) {
                            // Simple element-wise operation
                            const float* left_ptr = left_tensor.template ptr<float>();
                            const float* right_ptr = right_tensor.template ptr<float>();
                            float* out_ptr = result.template ptr<float>();
                            size_t n = result.numel();
                            for (size_t i = 0; i < n; ++i) {
                                out_ptr[i] = op(left_ptr[i], right_ptr[i]);
                            }
                        } else {
                            // Broadcasting required - fallback to CPU broadcast logic
                            Tensor left_broadcast = left_tensor;
                            Tensor right_broadcast = right_tensor;
                            if (left_tensor.shape() != shape) {
                                left_broadcast = left_tensor.broadcast_to(shape);
                            }
                            if (right_tensor.shape() != shape) {
                                right_broadcast = right_tensor.broadcast_to(shape);
                            }
                            const float* left_ptr = left_broadcast.template ptr<float>();
                            const float* right_ptr = right_broadcast.template ptr<float>();
                            float* out_ptr = result.template ptr<float>();
                            size_t n = result.numel();
                            for (size_t i = 0; i < n; ++i) {
                                out_ptr[i] = op(left_ptr[i], right_ptr[i]);
                            }
                        }
                    }
                }

                return result;
            }
        };

        // Specialized implementation: Bool-returning binary operations
        template <typename LeftExpr, typename RightExpr, typename BinaryOp>
        struct BinaryExprEvaluator<LeftExpr, RightExpr, BinaryOp, true> {
            static Tensor eval(const BinaryExpr<LeftExpr, RightExpr, BinaryOp>& /* expr */,
                               const LeftExpr& left, const RightExpr& right, const BinaryOp& op,
                               const TensorShape& shape, Device device, DataType dtype) {
                // Evaluate both sides
                Tensor left_tensor = left.eval();
                Tensor right_tensor = right.eval();

                // Create result tensor (Bool dtype)
                Tensor result = Tensor::empty(shape, device, dtype);

                // Determine if broadcasting is needed
                bool needs_broadcast = (left_tensor.shape() != shape) ||
                                       (right_tensor.shape() != shape);

                // Check input dtypes to determine correct template instantiation
                if (left_tensor.dtype() == DataType::Bool && right_tensor.dtype() == DataType::Bool) {
                    // Bool,Bool -> Bool (logical operations: logical_and, logical_or, logical_xor)
                    if (device == Device::CUDA) {
                        if (needs_broadcast) {
                            tensor_ops::launch_broadcast_binary(
                                left_tensor.template ptr<unsigned char>(),
                                right_tensor.template ptr<unsigned char>(),
                                result.template ptr<unsigned char>(),
                                left_tensor.shape().dims().data(),
                                right_tensor.shape().dims().data(),
                                shape.dims().data(),
                                left_tensor.shape().rank(), right_tensor.shape().rank(), shape.rank(),
                                result.numel(), op, nullptr);
                        } else {
                            tensor_ops::launch_binary_op_generic(
                                left_tensor.template ptr<unsigned char>(),
                                right_tensor.template ptr<unsigned char>(),
                                result.template ptr<unsigned char>(),
                                result.numel(), op, nullptr);
                        }
                    } else {
                        // CPU fallback
                        if (!needs_broadcast) {
                            const unsigned char* left_ptr = left_tensor.template ptr<unsigned char>();
                            const unsigned char* right_ptr = right_tensor.template ptr<unsigned char>();
                            unsigned char* out_ptr = result.template ptr<unsigned char>();
                            size_t n = result.numel();
                            for (size_t i = 0; i < n; ++i) {
                                out_ptr[i] = op(left_ptr[i], right_ptr[i]);
                            }
                        } else {
                            Tensor left_broadcast = left_tensor;
                            Tensor right_broadcast = right_tensor;
                            if (left_tensor.shape() != shape) {
                                left_broadcast = left_tensor.broadcast_to(shape);
                            }
                            if (right_tensor.shape() != shape) {
                                right_broadcast = right_tensor.broadcast_to(shape);
                            }
                            const unsigned char* left_ptr = left_broadcast.template ptr<unsigned char>();
                            const unsigned char* right_ptr = right_broadcast.template ptr<unsigned char>();
                            unsigned char* out_ptr = result.template ptr<unsigned char>();
                            size_t n = result.numel();
                            for (size_t i = 0; i < n; ++i) {
                                out_ptr[i] = op(left_ptr[i], right_ptr[i]);
                            }
                        }
                    }
                } else if (left_tensor.dtype() == DataType::Int32 && right_tensor.dtype() == DataType::Int32) {
                    // Int32,Int32 -> Bool (comparison operations on Int32 tensors)
                    if (device == Device::CUDA) {
                        if (needs_broadcast) {
                            tensor_ops::launch_broadcast_binary(
                                left_tensor.template ptr<int>(),
                                right_tensor.template ptr<int>(),
                                result.template ptr<unsigned char>(),
                                left_tensor.shape().dims().data(),
                                right_tensor.shape().dims().data(),
                                shape.dims().data(),
                                left_tensor.shape().rank(), right_tensor.shape().rank(), shape.rank(),
                                result.numel(), op, nullptr);
                        } else {
                            tensor_ops::launch_binary_op_generic(
                                left_tensor.template ptr<int>(),
                                right_tensor.template ptr<int>(),
                                result.template ptr<unsigned char>(),
                                result.numel(), op, nullptr);
                        }
                    } else {
                        // CPU fallback
                        if (!needs_broadcast) {
                            const int* left_ptr = left_tensor.template ptr<int>();
                            const int* right_ptr = right_tensor.template ptr<int>();
                            unsigned char* out_ptr = result.template ptr<unsigned char>();
                            size_t n = result.numel();
                            for (size_t i = 0; i < n; ++i) {
                                out_ptr[i] = op(left_ptr[i], right_ptr[i]);
                            }
                        } else {
                            Tensor left_broadcast = left_tensor;
                            Tensor right_broadcast = right_tensor;
                            if (left_tensor.shape() != shape) {
                                left_broadcast = left_tensor.broadcast_to(shape);
                            }
                            if (right_tensor.shape() != shape) {
                                right_broadcast = right_tensor.broadcast_to(shape);
                            }
                            const int* left_ptr = left_broadcast.template ptr<int>();
                            const int* right_ptr = right_broadcast.template ptr<int>();
                            unsigned char* out_ptr = result.template ptr<unsigned char>();
                            size_t n = result.numel();
                            for (size_t i = 0; i < n; ++i) {
                                out_ptr[i] = op(left_ptr[i], right_ptr[i]);
                            }
                        }
                    }
                } else {
                    // Float,Float -> Bool (comparison operations: eq, ne, lt, le, gt, ge)
                    if (device == Device::CUDA) {
                        if (needs_broadcast) {
                            tensor_ops::launch_broadcast_binary(
                                left_tensor.template ptr<float>(),
                                right_tensor.template ptr<float>(),
                                result.template ptr<unsigned char>(),
                                left_tensor.shape().dims().data(),
                                right_tensor.shape().dims().data(),
                                shape.dims().data(),
                                left_tensor.shape().rank(), right_tensor.shape().rank(), shape.rank(),
                                result.numel(), op, nullptr);
                        } else {
                            tensor_ops::launch_binary_op_generic(
                                left_tensor.template ptr<float>(),
                                right_tensor.template ptr<float>(),
                                result.template ptr<unsigned char>(),
                                result.numel(), op, nullptr);
                        }
                    } else {
                        // CPU fallback
                        if (!needs_broadcast) {
                            const float* left_ptr = left_tensor.template ptr<float>();
                            const float* right_ptr = right_tensor.template ptr<float>();
                            unsigned char* out_ptr = result.template ptr<unsigned char>();
                            size_t n = result.numel();
                            for (size_t i = 0; i < n; ++i) {
                                out_ptr[i] = op(left_ptr[i], right_ptr[i]);
                            }
                        } else {
                            Tensor left_broadcast = left_tensor;
                            Tensor right_broadcast = right_tensor;
                            if (left_tensor.shape() != shape) {
                                left_broadcast = left_tensor.broadcast_to(shape);
                            }
                            if (right_tensor.shape() != shape) {
                                right_broadcast = right_tensor.broadcast_to(shape);
                            }
                            const float* left_ptr = left_broadcast.template ptr<float>();
                            const float* right_ptr = right_broadcast.template ptr<float>();
                            unsigned char* out_ptr = result.template ptr<unsigned char>();
                            size_t n = result.numel();
                            for (size_t i = 0; i < n; ++i) {
                                out_ptr[i] = op(left_ptr[i], right_ptr[i]);
                            }
                        }
                    }
                }

                return result;
            }
        };
    } // namespace detail

    template <typename LeftExpr, typename RightExpr, typename BinaryOp>
    Tensor BinaryExpr<LeftExpr, RightExpr, BinaryOp>::eval_impl() const {
        // Dispatch to appropriate evaluator based on whether op returns Bool
        return detail::BinaryExprEvaluator<LeftExpr, RightExpr, BinaryOp, ops::returns_bool_v<BinaryOp>>::eval(
            *this, left_, right_, op_, shape_, device_, dtype_);
    }

    // ============================================================================
    // ScalarUnaryExpr::eval_impl() - Needs Tensor::empty()
    // ============================================================================

    template <typename InputExpr, typename ScalarUnaryOp>
    Tensor ScalarUnaryExpr<InputExpr, ScalarUnaryOp>::eval_impl() const {
        Tensor input_tensor = input_.eval();
        Tensor result = Tensor::empty(shape_, device_, dtype_);

        if (device_ == Device::CUDA) {
            tensor_ops::launch_unary_op_generic(
                input_tensor.template ptr<float>(),
                result.template ptr<float>(),
                result.numel(), op_, nullptr);
        } else {
            // CPU fallback: apply scalar operation element-wise
            const float* in_ptr = input_tensor.template ptr<float>();
            float* out_ptr = result.template ptr<float>();
            size_t n = result.numel();
            for (size_t i = 0; i < n; ++i) {
                out_ptr[i] = op_(in_ptr[i]);
            }
        }

        return result;
    }

    // ============================================================================
    // PermutationExpr::eval_impl() - Lazy gather operation
    // ============================================================================

    template <typename InputExpr, typename IndexExpr>
    Tensor PermutationExpr<InputExpr, IndexExpr>::eval_impl() const {
        // Evaluate input and indices
        Tensor input_tensor = input_.eval();
        Tensor indices_tensor = indices_.eval();

        // Ensure indices are Int32
        if (indices_tensor.dtype() != DataType::Int32) {
            throw std::runtime_error("PermutationExpr: indices must be Int32 dtype");
        }

        // Use existing take() implementation (already optimized with thrust::gather)
        return input_tensor.flatten().take(indices_tensor).reshape(shape_);
    }

    // ============================================================================
    // UnaryExpr<PermutationExpr, UnaryOp>::eval_impl() - FUSED gather + unary!
    // ============================================================================

    template <typename InputExpr, typename IndexExpr, typename UnaryOp>
    Tensor UnaryExpr<PermutationExpr<InputExpr, IndexExpr>, UnaryOp>::eval_impl() const {
        // Evaluate the input data and indices from permutation
        Tensor input_tensor = perm_expr_.input_.eval();
        Tensor indices_tensor = perm_expr_.indices_.eval();

        if (indices_tensor.dtype() != DataType::Int32) {
            throw std::runtime_error("PermutationExpr: indices must be Int32 dtype");
        }

        // Flatten input for gather
        Tensor flat_input = input_tensor.flatten();

        // Create result tensor
        Tensor result = Tensor::empty(shape_, device_, dtype_);

        // OPTIMIZATION: Use fused gather+unary kernel!
        if (device_ == Device::CUDA) {
            tensor_ops::launch_gather_fused_unary(
                flat_input.template ptr<float>(),
                indices_tensor.template ptr<int>(),
                result.template ptr<float>(),
                flat_input.numel(),
                indices_tensor.numel(),
                op_,
                nullptr);
        } else {
            // CPU fallback: gather then apply operation
            const float* src = flat_input.template ptr<float>();
            const int* idx = indices_tensor.template ptr<int>();
            float* dst = result.template ptr<float>();
            size_t total = flat_input.numel();

            for (size_t i = 0; i < indices_tensor.numel(); ++i) {
                int pos = idx[i];
                if (pos < 0)
                    pos += total;
                dst[i] = (pos >= 0 && pos < static_cast<int>(total)) ? op_(src[pos]) : 0.0f;
            }
        }

        return result.reshape(shape_);
    }

} // namespace lfs::core
