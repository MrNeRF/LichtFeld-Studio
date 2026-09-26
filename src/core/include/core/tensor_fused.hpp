/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/export.hpp"
#include "core/tensor_fwd.hpp"

#include <initializer_list>
#include <memory>
#include <optional>
#include <vector>

// Fused kernels: a small expression language evaluated at every coordinate of an
// N-dimensional domain (rank 1 to 4) and compiled once per layout class into one GPU
// kernel on the tensor's backend. Sizes, strides and addresses are launch arguments, so
// changing shapes reuses the compiled kernel. A fold of length 1 to 64 is part of that
// class (at most 64 compiled variants plus one runtime-length kernel for 65..4096).
// Caller-provided UInt8, Bool and Float16 outputs own their full aligned tail word;
// outputs allocated by the kernel include this padding.
//
//     fused::Builder b(2);                                  // domain [height, width]
//     auto image = b.input(DataType::Float32, 2);
//     auto gain = b.input(DataType::Float32, 1);            // e.g. a CPU parameter vector
//     auto x = b.iota(1).cast(DataType::Float32) / b.extent(1).cast(DataType::Float32);
//     b.output(fused::clamp(image.load() * gain.at({0}) + x, 0.0f, 1.0f), DataType::Float32);
//     const fused::Kernel kernel(b);
//     Tensor result = kernel({height, width}, {image_tensor, gain_tensor})[0];
namespace lfs::core::fused {

    enum class Bounds : uint8_t { Clamp,
                                  Zero,
                                  Wrap };

    // Sum of Float32 values is compensated and differs from Tensor::sum in the last bits.
    enum class Fold : uint8_t { Sum,
                                Min,
                                Max,
                                And,
                                Or,
                                Xor,
                                Count };

    class Builder;

    // A value at one domain coordinate: Float32, Int32, UInt32 or Bool. Float16 inputs
    // load as Float32 and UInt8 inputs as UInt32. Binary operations take equal types;
    // arithmetic scalars convert to the other operand's type.
    class LFS_CORE_API Expr {
    public:
        [[nodiscard]] DataType dtype() const { return dtype_; }
        // Float <-> integer conversions truncate toward zero; Int32 <-> UInt32 keep the bits;
        // Bool converts to 0 or 1 and nonzero values convert to true.
        [[nodiscard]] Expr cast(DataType dtype) const;

    private:
        friend class Builder;
        friend class Input;
        friend class Operand;
        friend struct Nodes;
        Expr(Builder* builder, uint32_t id, DataType dtype) : builder_(builder), id_(id), dtype_(dtype) {}
        Builder* builder_;
        uint32_t id_;
        DataType dtype_;
    };

    class LFS_CORE_API Operand {
    public:
        Operand(const Expr& value) : expr_(value) {}
        Operand(float value) : scalar_(value) {}
        Operand(double value) : scalar_(value) {}
        Operand(int32_t value) : scalar_(value), scalar_type_(DataType::Int32) {}
        Operand(uint32_t value) : scalar_(value), scalar_type_(DataType::UInt32) {}
        Operand(bool value) : scalar_(value), scalar_type_(DataType::Bool) {}

    private:
        friend struct Nodes;
        std::optional<Expr> expr_;
        double scalar_ = 0;
        DataType scalar_type_ = DataType::Float32;
    };

    // A tensor bound at call time. GPU tensors are read in place; CPU tensors of 32-bit
    // types (at most 64 words over all CPU inputs) are copied into the launch arguments.
    class LFS_CORE_API Input {
    public:
        // Input dimension i is read along domain dimension dims[i] (by default the trailing
        // domain dimensions) and must match its extent; other domain dimensions broadcast.
        [[nodiscard]] Expr load(std::initializer_list<size_t> dims = {}) const;
        // One Int32 index per input dimension, outermost first, each bounded by its extent.
        [[nodiscard]] Expr gather(std::initializer_list<Expr> index, Bounds bounds = Bounds::Zero) const;
        // Constant indices; every call rejects indices outside the bound tensor.
        [[nodiscard]] Expr at(std::initializer_list<uint32_t> index) const;

    private:
        friend class Builder;
        Input(Builder* builder, uint32_t index) : builder_(builder), index_(index) {}
        Builder* builder_;
        uint32_t index_;
    };

    class LFS_CORE_API Builder {
    public:
        explicit Builder(size_t rank);
        ~Builder();
        Builder(const Builder&) = delete;
        Builder& operator=(const Builder&) = delete;

        Input input(DataType dtype, size_t rank);
        // Int32 coordinate along a domain dimension and the dimension's extent.
        Expr iota(size_t dim);
        Expr extent(size_t dim);
        Expr constant(float value);
        Expr constant(int32_t value);
        Expr constant(uint32_t value);
        Expr constant(bool value);
        // Reduces value along domain dimension dim (at most 4096 long). Every fold of a kernel
        // uses the same dimension, outputs span the domain without it, and values that use a
        // fold result are computed once per output element after the fold.
        Expr fold(Fold kind, const Expr& value, size_t dim);
        // Float32 values store to Float32 or Float16, Int32 and UInt32 values to Int32, UInt32
        // or UInt8 (low byte), Bool values to Bool or UInt8.
        void output(const Expr& value, DataType dtype);

    private:
        friend class Kernel;
        friend class Expr;
        friend class Input;
        friend struct Nodes;
        struct State;
        std::unique_ptr<State> state_;
    };

    class LFS_CORE_API Kernel {
    public:
        explicit Kernel(const Builder& builder);
        ~Kernel();

        // Outputs have the domain's shape without the fold dimension ({1} when nothing
        // remains) and live on the inputs' backend. An input may alias an output only at the
        // same elements. Compilation happens on the first call for a layout class.
        std::vector<Tensor> operator()(const std::vector<size_t>& domain, const std::vector<Tensor>& inputs) const;
        void operator()(const std::vector<size_t>& domain, const std::vector<Tensor>& inputs,
                        const std::vector<Tensor>& outputs) const;
        // Compiles and loads the kernel for the layouts of these arguments without running
        // it; empty outputs stand for allocated ones. Loading a CUDA module synchronizes the
        // CUDA context, so call this during setup rather than per frame.
        void prepare(const std::vector<size_t>& domain, const std::vector<Tensor>& inputs,
                     const std::vector<Tensor>& outputs) const;

    private:
        struct Compiled;
        void launch(const std::vector<size_t>& domain, const std::vector<Tensor>& inputs,
                    const std::vector<Tensor>& outputs, bool run) const;
        std::shared_ptr<const Compiled> compiled_;
    };

    LFS_CORE_API Expr operator+(const Operand& a, const Operand& b);
    LFS_CORE_API Expr operator-(const Operand& a, const Operand& b);
    LFS_CORE_API Expr operator*(const Operand& a, const Operand& b);
    // Integer division and remainder truncate toward zero; a zero divisor yields 0.
    LFS_CORE_API Expr operator/(const Operand& a, const Operand& b);
    LFS_CORE_API Expr operator%(const Operand& a, const Operand& b);
    LFS_CORE_API Expr operator-(const Expr& a);
    LFS_CORE_API Expr operator<(const Operand& a, const Operand& b);
    LFS_CORE_API Expr operator<=(const Operand& a, const Operand& b);
    LFS_CORE_API Expr operator>(const Operand& a, const Operand& b);
    LFS_CORE_API Expr operator>=(const Operand& a, const Operand& b);
    LFS_CORE_API Expr operator==(const Operand& a, const Operand& b);
    LFS_CORE_API Expr operator!=(const Operand& a, const Operand& b);
    LFS_CORE_API Expr operator&&(const Operand& a, const Operand& b);
    LFS_CORE_API Expr operator||(const Operand& a, const Operand& b);
    LFS_CORE_API Expr operator!(const Expr& a);
    LFS_CORE_API Expr operator&(const Operand& a, const Operand& b);
    LFS_CORE_API Expr operator|(const Operand& a, const Operand& b);
    LFS_CORE_API Expr operator^(const Operand& a, const Operand& b);
    LFS_CORE_API Expr operator~(const Expr& a);
    // Shifts are logical and use the low five bits of the shift amount.
    LFS_CORE_API Expr operator<<(const Operand& a, const Operand& b);
    LFS_CORE_API Expr operator>>(const Operand& a, const Operand& b);

    LFS_CORE_API Expr where(const Operand& condition, const Operand& yes, const Operand& no);
    LFS_CORE_API Expr clamp(const Operand& value, const Operand& low, const Operand& high);
    LFS_CORE_API Expr min(const Operand& a, const Operand& b);
    LFS_CORE_API Expr max(const Operand& a, const Operand& b);
    LFS_CORE_API Expr abs(const Expr& a);
    LFS_CORE_API Expr fma(const Operand& a, const Operand& b, const Operand& c);
    LFS_CORE_API Expr pow(const Operand& a, const Operand& b);
    LFS_CORE_API Expr atan2(const Operand& y, const Operand& x);
#define LFS_FUSED_UNARY(name) LFS_CORE_API Expr name(const Expr& a);
    LFS_FUSED_UNARY(exp)
    LFS_FUSED_UNARY(log)
    LFS_FUSED_UNARY(sqrt)
    LFS_FUSED_UNARY(sigmoid)
    LFS_FUSED_UNARY(relu)
    LFS_FUSED_UNARY(square)
    LFS_FUSED_UNARY(tanh)
    LFS_FUSED_UNARY(rsqrt)
    LFS_FUSED_UNARY(sign)
    LFS_FUSED_UNARY(reciprocal)
    LFS_FUSED_UNARY(floor)
    LFS_FUSED_UNARY(ceil)
    LFS_FUSED_UNARY(round)
    LFS_FUSED_UNARY(exp2)
    LFS_FUSED_UNARY(log2)
    LFS_FUSED_UNARY(log10)
    LFS_FUSED_UNARY(log1p)
    LFS_FUSED_UNARY(sin)
    LFS_FUSED_UNARY(cos)
    LFS_FUSED_UNARY(tan)
    LFS_FUSED_UNARY(asin)
    LFS_FUSED_UNARY(acos)
    LFS_FUSED_UNARY(atan)
    LFS_FUSED_UNARY(sinh)
    LFS_FUSED_UNARY(cosh)
    LFS_FUSED_UNARY(gelu)
    LFS_FUSED_UNARY(swish)
    LFS_FUSED_UNARY(trunc)
    LFS_FUSED_UNARY(isnan)
    LFS_FUSED_UNARY(isinf)
    LFS_FUSED_UNARY(isfinite)
#undef LFS_FUSED_UNARY

} // namespace lfs::core::fused
