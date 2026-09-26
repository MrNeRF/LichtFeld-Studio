/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/tensor/internal/private_access.hpp"

#include "core/tensor/backend/descriptors.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace lfs::core::internal {

    // Float opcodes read f32 bits; integer and logical opcodes read raw u32 values.
    // Gather, Gather2 and Gather3 take one index per table dimension, outermost first; the
    // policy bounds every index against its dimension. Checked indices are immediates that
    // every launch validates against the view. Fold reduces its operand along the
    // domain dimension in its policy field with the ExprReduce in aux, accumulating in the
    // DataType in immediate; all folds of a program share that dimension. Values that depend
    // on a fold are computed once per output element after the fold completes.
    enum class ExprOp : uint8_t {
        Load,
        Gather,
        Gather2,
        Gather3,
        Iota,
        Extent,
        Immediate,
        Store,
        Fold,
        Add,
        Sub,
        Mul,
        Div,
        Mod,
        Pow,
        Min,
        Max,
        Atan2,
        Fma,
        Clamp,
        Select,
        Abs,
        Neg,
        Exp,
        Log,
        Sqrt,
        Sigmoid,
        Relu,
        Square,
        Tanh,
        Rsqrt,
        Sign,
        Reciprocal,
        Floor,
        Ceil,
        Round,
        Exp2,
        Log2,
        Log10,
        Log1p,
        Sin,
        Cos,
        Tan,
        Asin,
        Acos,
        Atan,
        Sinh,
        Cosh,
        Gelu,
        Swish,
        Trunc,
        IsNan,
        IsInf,
        IsFinite,
        Equal,
        NotEqual,
        Less,
        LessEqual,
        Greater,
        GreaterEqual,
        LogicalAnd,
        LogicalOr,
        LogicalXor,
        LogicalNot,
        BitAnd,
        BitOr,
        BitXor,
        BitNot,
        ShiftLeft,
        ShiftRight,
        AddInt,
        SubInt,
        MulInt,
        IntDiv,
        IntMod,
        EqualInt,
        NotEqualInt,
        LessInt,
        LessEqualInt,
        GreaterInt,
        GreaterEqualInt,
        CastFloat,
        CastInt,
        CastUInt,
        UIntDiv,
        UIntMod,
        LessUInt,
        LessEqualUInt,
        GreaterUInt,
        GreaterEqualUInt,
        CastUnsignedFloat,
    };

    enum class ExprOob : uint8_t { Clamp,
                                   Zero,
                                   Wrap,
                                   Checked };
    enum class ExprReduce : uint8_t { Sum,
                                      Min,
                                      Max,
                                      And,
                                      Or,
                                      Xor,
                                      Count };

    // Register indices occupy byte fields; immediates retain all 32 bits.
    struct ExprInstruction {
        uint32_t op_dst_a_b;
        uint32_t c_aux_policy;
        uint32_t immediate;
    };
    static_assert(sizeof(ExprInstruction) == 12);

    constexpr uint32_t expr_pack(ExprOp op, uint8_t dst, uint8_t a, uint8_t b) {
        return static_cast<uint32_t>(op) | (uint32_t(dst) << 8) |
               (uint32_t(a) << 16) | (uint32_t(b) << 24);
    }

    constexpr uint32_t expr_arity(ExprOp op) {
        if (op == ExprOp::Load || op == ExprOp::Iota || op == ExprOp::Extent || op == ExprOp::Immediate)
            return 0;
        if (op == ExprOp::Gather3 || op == ExprOp::Fma || op == ExprOp::Clamp || op == ExprOp::Select)
            return 3;
        if ((op >= ExprOp::Abs && op <= ExprOp::IsFinite) ||
            op == ExprOp::Gather || op == ExprOp::Store || op == ExprOp::Fold ||
            op == ExprOp::LogicalNot || op == ExprOp::BitNot ||
            op == ExprOp::CastFloat || op == ExprOp::CastInt ||
            op == ExprOp::CastUInt || op == ExprOp::CastUnsignedFloat)
            return 1;
        return 2;
    }

    class LFS_CORE_API ExpressionProgram {
    public:
        static constexpr uint32_t max_instructions = 512;
        static constexpr uint32_t max_registers = 32;
        static constexpr uint32_t max_inputs = 12;
        static constexpr uint32_t max_outputs = 4;
        static constexpr uint32_t max_rank = 4;
        static constexpr uint32_t max_fold = 4096;
        static constexpr uint32_t max_parameters = 64;

        struct Analysis {
            std::vector<std::array<uint32_t, 3>> sources;
            std::array<bool, max_inputs> loaded{};
            std::array<uint32_t, max_inputs> arity{};
            std::array<std::array<uint32_t, max_rank>, max_inputs> checked_bounds{};
            uint32_t inputs = 0, stores = 0, rank = 1;
            int fold = -1;
            bool runtime_gathers = false;
        };
        [[nodiscard]] const Analysis& analysis() const { return analysis_; }
        [[nodiscard]] std::span<const ExprInstruction> instructions() const { return instructions_; }
        [[nodiscard]] std::vector<uint32_t> compact_inputs();

    private:
        friend class ExpressionBuilder;
        // Registers may be overwritten after their last use.
        void emit(ExprOp op, uint8_t dst, uint8_t a, uint8_t b,
                  uint8_t c, uint8_t aux, uint32_t immediate, ExprOob policy);

        void analyze();
        Analysis analysis_;
        std::vector<ExprInstruction> instructions_;
    };

    class LFS_CORE_API ExpressionBuilder {
    public:
        using Value = uint32_t;
        Value emit(ExprOp op, Value a = 0, Value b = 0, Value c = 0,
                   uint8_t aux = 0, uint32_t immediate = 0,
                   ExprOob policy = ExprOob::Zero);
        [[nodiscard]] ExpressionProgram build() const;

    private:
        struct Node {
            ExprOp op;
            Value operands[3];
            uint8_t aux;
            uint32_t immediate;
            ExprOob policy;
        };
        std::vector<Node> nodes_;
    };

    // A strided view. Loads index it with domain coordinates, and a nonzero stride requires
    // the domain extent in that dimension; gathers index it with explicit indices bounded by
    // dims. Host storage holds 32-bit parameters copied into the launch arguments.
    struct ExprInput {
        StorageRef storage;
        std::array<uint32_t, ExpressionProgram::max_rank> dims{1, 1, 1, 1};
        std::array<int32_t, ExpressionProgram::max_rank> strides{};
    };

    // Element strides over the domain dimensions; the fold dimension has none. Packed outputs
    // are contiguous and own their complete aligned words, including the tail word.
    struct ExprOutput {
        StorageRef storage;
        std::array<int32_t, ExpressionProgram::max_rank> strides{};
    };

    struct ExprShape {
        std::array<uint32_t, ExpressionProgram::max_rank> dims{1, 1, 1, 1};
        uint32_t rank = 1;
    };

    // A Store instruction names its output in aux and its value register in a. Sizes, strides
    // and addresses are launch arguments: programs compile once per layout class.
    LFS_CORE_API void run_expression(const ExpressionProgram& program,
                                     std::span<const ExprInput> inputs,
                                     std::span<const ExprOutput> outputs,
                                     const ExprShape& shape, ExecContext context = {});
    // Compiles and loads the program for the layout class of these bindings without running
    // it. Loading a CUDA module synchronizes the CUDA context.
    LFS_CORE_API void prepare_expression(const ExpressionProgram& program,
                                         std::span<const ExprInput> inputs,
                                         std::span<const ExprOutput> outputs,
                                         const ExprShape& shape);

} // namespace lfs::core::internal
