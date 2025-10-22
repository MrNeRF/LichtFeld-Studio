/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef> // for size_t
#include <cstdint>
#include <memory>    // for std::shared_ptr
#include <stdexcept> // for std::runtime_error
#include <string>    // MUST be outside namespace!
#include <type_traits>
#include <utility>
#include <vector> // MUST be outside namespace!

namespace lfs::core {

    // Forward declarations only - actual definitions are in tensor.hpp
    // NOTE: This header is NOT meant to be included standalone!
    // It should ONLY be included from tensor.hpp or files that include tensor.hpp
    class Tensor;
    class TensorShape;
    enum class Device : uint8_t;
    enum class DataType : uint8_t;

    // Forward declare minimal ops namespace (actual definitions in tensor_functors.hpp)
    namespace ops {
        template <typename F, typename G>
        struct composed_unary_op;
    }

    // Forward declare tensor_ops - actual definitions in tensor_ops.hpp
    // Note: We don't forward declare here to avoid ambiguity issues
    // The actual declarations will be available when tensor_expr_impl.hpp is included

    // Forward declare UnaryExpr so TensorLeaf can reference it
    template <typename InputExpr, typename UnaryOp>
    class UnaryExpr;

    // ============================================================================
    // EXPRESSION TEMPLATE BASE CLASS (CRTP Pattern)
    // ============================================================================

    template <typename Derived>
    class TensorExpr {
    public:
        // CRTP pattern for static polymorphism
        const Derived& derived() const { return static_cast<const Derived&>(*this); }
        Derived& derived() { return static_cast<Derived&>(*this); }

        // Materialize the expression into a Tensor
        Tensor eval() const;

        // Allow implicit conversion (materializes on assignment)
        operator Tensor() const;

        // Get expression metadata
        const TensorShape& shape() const { return derived().shape_impl(); }
        Device device() const { return derived().device_impl(); }
        DataType dtype() const { return derived().dtype_impl(); }
    };

    // ============================================================================
    // LEAF EXPRESSION: Wraps an existing Tensor
    // ============================================================================

    class TensorLeaf : public TensorExpr<TensorLeaf> {
    private:
        // Use shared_ptr to avoid needing complete Tensor type in header
        std::shared_ptr<Tensor> tensor_ptr_;

    public:
        explicit TensorLeaf(Tensor tensor); // Implemented in .cpp

        Tensor eval_impl() const; // Implemented in .cpp

        const TensorShape& shape_impl() const;
        Device device_impl() const;
        DataType dtype_impl() const;

        // Enable chaining of operations (template methods must be defined in header)
        // Definition is after UnaryExpr is fully defined (see below line 240)
        template <typename UnaryOp>
        auto map(UnaryOp op) const -> UnaryExpr<TensorLeaf, UnaryOp>;
    };

    // Template method definition for TensorLeaf::map (must be after UnaryExpr is declared)
    // This will be defined after UnaryExpr class definition

    // ============================================================================
    // UNARY EXPRESSION: Applies a unary operation to an input expression
    // ============================================================================

    template <typename InputExpr, typename UnaryOp>
    class UnaryExpr : public TensorExpr<UnaryExpr<InputExpr, UnaryOp>> {
    private:
        InputExpr input_;
        UnaryOp op_;
        TensorShape shape_;
        Device device_;
        DataType dtype_;

        // Allow all UnaryExpr instantiations to access private members (needed for fusion)
        template <typename AnyInput, typename AnyOp>
        friend class UnaryExpr;

    public:
        UnaryExpr(InputExpr input, UnaryOp op, TensorShape shape, Device device, DataType dtype)
            : input_(std::move(input)),
              op_(op),
              shape_(std::move(shape)),
              device_(device),
              dtype_(dtype) {}

        // Evaluation: implemented in tensor_expr_impl.hpp (needs full Tensor definition)
        Tensor eval_impl() const;

        // Compose with another unary operation (fusion opportunity!)
        template <typename NewOp>
        auto map(NewOp new_op) const {
            // Instead of materializing, compose operations
            auto fused_op = ops::compose(op_, new_op);
            return UnaryExpr<InputExpr, decltype(fused_op)>(
                input_, fused_op, shape_, device_, dtype_);
        }

        const TensorShape& shape_impl() const { return shape_; }
        Device device_impl() const { return device_; }
        DataType dtype_impl() const { return dtype_; }
    };

    // ============================================================================
    // UNARY EXPRESSION SPECIALIZATION: Fuses nested unary operations
    // ============================================================================

    template <typename InnerInput, typename InnerOp, typename OuterOp>
    class UnaryExpr<UnaryExpr<InnerInput, InnerOp>, OuterOp>
        : public TensorExpr<UnaryExpr<UnaryExpr<InnerInput, InnerOp>, OuterOp>> {
    private:
        using InnerExpr = UnaryExpr<InnerInput, InnerOp>;
        InnerExpr inner_expr_;
        OuterOp outer_op_;
        TensorShape shape_;
        Device device_;
        DataType dtype_;

    public:
        UnaryExpr(InnerExpr inner, OuterOp outer, TensorShape shape, Device device, DataType dtype)
            : inner_expr_(std::move(inner)),
              outer_op_(outer),
              shape_(std::move(shape)),
              device_(device),
              dtype_(dtype) {}

        // FUSION DETECTED! Implemented in tensor_expr_impl.hpp
        Tensor eval_impl() const;

        // Continue fusion chain
        template <typename NewOp>
        auto map(NewOp new_op) const {
            // Fuse all three operations
            auto fused_op = ops::compose(inner_expr_.op_, outer_op_, new_op);
            return UnaryExpr<InnerInput, decltype(fused_op)>(
                inner_expr_.input_, fused_op, shape_, device_, dtype_);
        }

        const TensorShape& shape_impl() const { return shape_; }
        Device device_impl() const { return device_; }
        DataType dtype_impl() const { return dtype_; }
    };

    // ============================================================================
    // BINARY EXPRESSION: Combines two expressions with a binary operation
    // ============================================================================

    template <typename LeftExpr, typename RightExpr, typename BinaryOp>
    class BinaryExpr : public TensorExpr<BinaryExpr<LeftExpr, RightExpr, BinaryOp>> {
    private:
        LeftExpr left_;
        RightExpr right_;
        BinaryOp op_;
        TensorShape shape_;
        Device device_;
        DataType dtype_;

    public:
        BinaryExpr(LeftExpr left, RightExpr right, BinaryOp op,
                   TensorShape shape, Device device, DataType dtype)
            : left_(std::move(left)),
              right_(std::move(right)),
              op_(op),
              shape_(std::move(shape)),
              device_(device),
              dtype_(dtype) {}

        // Implemented in tensor_expr_impl.hpp
        Tensor eval_impl() const;

        // Apply unary operation to result (fuses with binary op)
        template <typename UnaryOp>
        auto map(UnaryOp unary_op) const {
            // Return a unary expression wrapping this binary expression
            return UnaryExpr<BinaryExpr, UnaryOp>(
                *this, unary_op, shape_, device_, dtype_);
        }

        const TensorShape& shape_impl() const { return shape_; }
        Device device_impl() const { return device_; }
        DataType dtype_impl() const { return dtype_; }
    };

    // ============================================================================
    // SCALAR UNARY EXPRESSION: Unary operation with scalar parameter
    // ============================================================================

    template <typename InputExpr, typename ScalarUnaryOp>
    class ScalarUnaryExpr : public TensorExpr<ScalarUnaryExpr<InputExpr, ScalarUnaryOp>> {
    private:
        InputExpr input_;
        ScalarUnaryOp op_;
        TensorShape shape_;
        Device device_;
        DataType dtype_;

    public:
        ScalarUnaryExpr(InputExpr input, ScalarUnaryOp op,
                        TensorShape shape, Device device, DataType dtype)
            : input_(std::move(input)),
              op_(op),
              shape_(std::move(shape)),
              device_(device),
              dtype_(dtype) {}

        // Implemented in tensor_expr_impl.hpp
        Tensor eval_impl() const;

        template <typename NewOp>
        auto map(NewOp new_op) const {
            auto fused_op = ops::compose(op_, new_op);
            return UnaryExpr<InputExpr, decltype(fused_op)>(
                input_, fused_op, shape_, device_, dtype_);
        }

        const TensorShape& shape_impl() const { return shape_; }
        Device device_impl() const { return device_; }
        DataType dtype_impl() const { return dtype_; }
    };

    // ============================================================================
    // PERMUTATION EXPRESSION: Lazy gather/indexing using permutation
    // ============================================================================

    template <typename InputExpr, typename IndexExpr>
    class PermutationExpr : public TensorExpr<PermutationExpr<InputExpr, IndexExpr>> {
    private:
        InputExpr input_;
        IndexExpr indices_;
        TensorShape shape_;
        Device device_;
        DataType dtype_;

        // Allow UnaryExpr specialization to access private members
        template <typename AnyInput, typename AnyOp>
        friend class UnaryExpr;

    public:
        PermutationExpr(InputExpr input, IndexExpr indices,
                        TensorShape shape, Device device, DataType dtype)
            : input_(std::move(input)),
              indices_(std::move(indices)),
              shape_(std::move(shape)),
              device_(device),
              dtype_(dtype) {}

        // Implemented in tensor_expr_impl.hpp
        Tensor eval_impl() const;

        // Apply unary operation to gathered result (fuses with gather!)
        template <typename UnaryOp>
        auto map(UnaryOp unary_op) const {
            return UnaryExpr<PermutationExpr, UnaryOp>(
                *this, unary_op, shape_, device_, dtype_);
        }

        const TensorShape& shape_impl() const { return shape_; }
        Device device_impl() const { return device_; }
        DataType dtype_impl() const { return dtype_; }
    };

    // ============================================================================
    // UNARY EXPRESSION SPECIALIZATION: Fuses gather + unary operation
    // ============================================================================

    template <typename InputExpr, typename IndexExpr, typename UnaryOp>
    class UnaryExpr<PermutationExpr<InputExpr, IndexExpr>, UnaryOp>
        : public TensorExpr<UnaryExpr<PermutationExpr<InputExpr, IndexExpr>, UnaryOp>> {
    private:
        using PermExpr = PermutationExpr<InputExpr, IndexExpr>;
        PermExpr perm_expr_;
        UnaryOp op_;
        TensorShape shape_;
        Device device_;
        DataType dtype_;

    public:
        UnaryExpr(PermExpr perm, UnaryOp op, TensorShape shape, Device device, DataType dtype)
            : perm_expr_(std::move(perm)),
              op_(op),
              shape_(std::move(shape)),
              device_(device),
              dtype_(dtype) {}

        // FUSED gather + unary! Implemented in tensor_expr_impl.hpp
        Tensor eval_impl() const;

        const TensorShape& shape_impl() const { return shape_; }
        Device device_impl() const { return device_; }
        DataType dtype_impl() const { return dtype_; }
    };

    // ============================================================================
    // TensorLeaf::map implementation (after UnaryExpr is fully defined)
    // ============================================================================

    template <typename UnaryOp>
    auto TensorLeaf::map(UnaryOp op) const -> UnaryExpr<TensorLeaf, UnaryOp> {
        return UnaryExpr<TensorLeaf, UnaryOp>(
            *this, op, shape_impl(), device_impl(), dtype_impl());
    }

} // namespace lfs::core
