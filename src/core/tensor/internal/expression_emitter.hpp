/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/tensor/internal/private_access.hpp"
#include "expression_runtime.hpp"

namespace lfs::core::internal {
    // Values carry raw u32 bits. Mutable slots are local to one GPU invocation. Arguments are
    // the words of the ExpressionLayout block; `address` is the word offset of a 64-bit device
    // address in it.
    class ExpressionEmitter {
    public:
        using Value = uint32_t;
        virtual ~ExpressionEmitter() = default;
        virtual Value literal(uint32_t value) = 0;
        virtual Value math(ExprOp op, Value a, Value b = 0, Value c = 0) = 0;
        virtual Value mul_hi(Value a, Value b) = 0;
        virtual Value variable(Value initial) = 0;
        virtual Value read(Value slot) = 0;
        virtual void assign(Value slot, Value value) = 0;
        virtual void condition(Value predicate, const std::function<void()>& yes,
                               const std::function<void()>& no = {}) = 0;
        virtual void loop(Value first, Value end, Value step,
                          const std::function<void(Value)>& body) = 0;
        virtual Value argument(Value word) = 0;
        virtual Value load(uint32_t address, Value index, DataType dtype) = 0;
        // Two adjacent 32-bit elements at an 8-byte aligned address.
        virtual std::array<Value, 2> load_pair(uint32_t address, Value index) = 0;
        virtual Value output(uint32_t address, Value word) = 0;
        virtual void store(uint32_t address, Value index, Value value, DataType dtype = DataType::UInt32) = 0;
        virtual Value half(Value value, bool pack) = 0;
        virtual Value thread() = 0;
        virtual Value grid_stride() = 0;
        virtual bool cuda() const = 0;
        Value select(Value predicate, const std::function<Value()>& yes,
                     const std::function<Value()>& no) {
            auto slot = variable(literal(0));
            condition(predicate, [&] { assign(slot, yes()); }, [&] { assign(slot, no()); });
            return read(slot);
        }
    };
    void emit_expression(ExpressionEmitter& emitter, const ExpressionProgram& program,
                         const ExpressionSignature& signature);
    std::string expression_ptx(const ExpressionProgram& program, const ExpressionSignature& signature);
    std::vector<uint32_t> expression_spirv(const ExpressionProgram& program, const ExpressionSignature& signature);
    std::string expression_msl(const ExpressionProgram& program, const ExpressionSignature& signature);
    std::string_view expression_ptx_opcode(ExprOp op);
    std::string_view expression_ptx_half(bool pack);
    std::string_view expression_ptx_header();
} // namespace lfs::core::internal
