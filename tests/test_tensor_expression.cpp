/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/tensor.hpp"
#include "core/tensor/backend/gpu_backend_ops.hpp"
#include "core/tensor/backend/vulkan/vk_context.hpp"
#include "core/tensor/internal/expression_runtime.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_fused.hpp"
#include "expression_reference.hpp"
#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstring>
#include <future>
#include <gtest/gtest.h>
#include <string>
#include <thread>

void expression_native_atan2(const float*, const float*, float*, size_t);

namespace {
    using namespace expression_test;

    void CUDART_CB hold_stream(void* data) {
        static_cast<std::atomic<bool>*>(data)->wait(false);
    }

    Tensor host(const std::vector<uint32_t>& words, DataType dtype) {
        auto result = Tensor::empty_pageable_host({words.size()}, dtype);
        auto* bytes = static_cast<uint8_t*>(result.data_ptr());
        for (size_t i = 0; i < words.size(); ++i) {
            if (dtype == DataType::Float16)
                reinterpret_cast<__half*>(bytes)[i] = __float2half_rn(value(words[i]));
            else if (dtype == DataType::UInt8 || dtype == DataType::Bool)
                bytes[i] = uint8_t(words[i]);
            else
                std::memcpy(bytes + i * 4, &words[i], 4);
        }
        return result;
    }
    Tensor floats(std::initializer_list<float> values) {
        std::vector<uint32_t> words;
        for (float x : values)
            words.push_back(bits(x));
        return host(words, DataType::Float32);
    }
    ExprInput binding(const Tensor& tensor, std::array<int32_t, 4> strides) {
        ExprInput input;
        input.storage = tensor.device() == Device::CPU ? raw_storage_ref(const_cast<void*>(tensor.data_ptr()), tensor.dtype()) : storage_ref(tensor);
        input.strides = strides;
        for (size_t d = 0; d < 4; ++d) {
            if (!strides[d])
                continue;
            size_t span = tensor.numel();
            for (auto other : strides)
                if (other > strides[d])
                    span = std::min(span, size_t(other));
            input.dims[d] = uint32_t((span + strides[d] - 1) / strides[d]);
        }
        return input;
    }
    class ExpressionTest : public testing::TestWithParam<int> {
    protected:
        void SetUp() override {
            if (!gpu_backend_available(backend()))
                GTEST_SKIP();
        }
        void TearDown() override {
            if (backend() == GpuBackend::Vulkan && gpu_backend_available(backend())) {
                backend_ops(backend()).synchronize_device();
                for (const auto& message : vulkan_validation_messages_for_testing())
                    ADD_FAILURE() << message;
            }
        }
        GpuBackend backend() const { return GetParam() == 2 ? GpuBackend::Vulkan : GpuBackend::CUDA; }
        void check(const ExpressionProgram& p, const std::vector<Tensor>& cpu,
                   const std::vector<std::array<int32_t, 4>>& strides,
                   const ExprShape& shape, const std::vector<DataType>& types) {
            std::vector<ExprInput> input;
            for (size_t i = 0; i < cpu.size(); ++i)
                input.push_back(binding(cpu[i], strides[i]));
            auto expected = reference(p, input, shape, types);
            GpuBackendScope scope(backend());
            std::vector<Tensor> gpu;
            input.clear();
            for (size_t i = 0; i < cpu.size(); ++i) {
                gpu.push_back(cpu[i].to(Device::GPU));
                input.push_back(binding(gpu.back(), strides[i]));
            }
            size_t count = expected[0].size();
            std::vector<Tensor> output;
            std::vector<ExprOutput> refs;
            for (auto dtype : types) {
                auto initial = Tensor::empty_pageable_host({count + 4}, dtype);
                std::memset(initial.data_ptr(), 0xa5, (count + 4) * dtype_size(dtype));
                output.push_back(initial.to(Device::GPU));
                ExprOutput target{storage_ref(output.back())};
                int fold = -1;
                for (const auto ins : p.instructions())
                    if (ExprOp(ins.op_dst_a_b & 255) == ExprOp::Fold)
                        fold = (ins.c_aux_policy >> 16) & 255;
                int32_t stride = 1;
                for (int d = int(shape.rank) - 1; d >= 0; --d)
                    if (d != fold) {
                        target.strides[d] = stride;
                        stride *= shape.dims[d];
                    }
                refs.push_back(target);
            }
            run_expression(p, input, refs, shape);
            backend_ops(backend()).synchronize_device();
            for (size_t o = 0; o < output.size(); ++o) {
                auto actual = output[o].cpu();
                auto actual_input = binding(actual, {});
                for (size_t i = 0; i < count; ++i) {
                    SCOPED_TRACE("output=" + std::to_string(o) + " index=" + std::to_string(i));
                    uint32_t wanted = expected[o][i], got = load(actual_input, i);
                    if (types[o] == DataType::Float16)
                        wanted = bits(__half2float(__float2half_rn(value(wanted))));
                    if (types[o] == DataType::Float32 || types[o] == DataType::Float16) {
                        if (std::isnan(value(wanted)))
                            EXPECT_TRUE(std::isnan(value(got)));
                        else if (std::isinf(value(wanted)))
                            EXPECT_EQ(value(got), value(wanted));
                        else
                            EXPECT_NEAR(value(got), value(wanted), 3e-5f + 3e-5f * std::abs(value(wanted)));
                    } else
                        EXPECT_EQ(got, (types[o] == DataType::UInt8 || types[o] == DataType::Bool) ? wanted & 255 : wanted);
                }
                const auto* bytes = static_cast<const uint8_t*>(actual.data_ptr());
                for (size_t i = count * dtype_size(types[o]); i < (count + 4) * dtype_size(types[o]); ++i)
                    EXPECT_EQ(bytes[i], 0xa5);
            }
        }
    };
    TEST_P(ExpressionTest, IntegerDivisionBoundaries) {
        for (const auto op : {ExprOp::IntDiv, ExprOp::IntMod, ExprOp::UIntDiv, ExprOp::UIntMod}) {
            SCOPED_TRACE(uint32_t(op));
            const auto dtype = op == ExprOp::IntDiv || op == ExprOp::IntMod ? DataType::Int32 : DataType::UInt32;
            const auto lhs = host({0, 1, uint32_t(-7), 9, 127, 0x80000000u, 0xffffffffu}, dtype);
            const auto rhs = host({0, 0, 3, 4, 5, 0xffffffffu, 7}, dtype);
            ExpressionBuilder builder;
            auto x = builder.emit(ExprOp::Load), y = builder.emit(ExprOp::Load, 0, 0, 0, 1);
            builder.emit(ExprOp::Store, builder.emit(op, x, y));
            check(builder.build(), {lhs, rhs}, {{1, 0, 0, 0}, {1, 0, 0, 0}}, {{7, 1, 1, 1}}, {dtype});
        }
    }
    TEST_P(ExpressionTest, InterleavedOutput) {
        GpuBackendScope scope(backend());
        auto src = floats({1, 2, 3, 4, 5}).to(Device::GPU);
        auto dst = Tensor::empty({10}, Device::GPU);
        ExpressionBuilder b;
        auto x = b.emit(ExprOp::Load);
        b.emit(ExprOp::Store, x, 0, 0, 0);
        b.emit(ExprOp::Store, b.emit(ExprOp::Neg, x), 0, 0, 1);
        std::array<ExprInput, 1> in{binding(src, {1, 0, 0, 0})};
        std::array<ExprOutput, 2> out{{{storage_ref(dst), 2}, {offset_storage_ref(storage_ref(dst), 4), 2}}};
        run_expression(b.build(), in, out, {{5, 1, 1, 1}});
        auto actual = dst.cpu().to_vector();
        for (int i = 0; i < 5; ++i) {
            EXPECT_EQ(actual[2 * i], float(i + 1));
            EXPECT_EQ(actual[2 * i + 1], -float(i + 1));
        }
    }
    TEST_P(ExpressionTest, DagBroadcastStridesAndMultipleOutputs) {
        ExpressionBuilder b;
        auto x = b.emit(ExprOp::Load), y = b.emit(ExprOp::Load, 0, 0, 0, 1);
        auto sum = b.emit(ExprOp::Add, b.emit(ExprOp::Square, x), b.emit(ExprOp::Square, y));
        b.emit(ExprOp::Store, sum, 0, 0, 0);
        b.emit(ExprOp::Store, b.emit(ExprOp::Less, x, y), 0, 0, 1);
        b.emit(ExprOp::Store, b.emit(ExprOp::Iota, 0, 0, 0, 1), 0, 0, 2);
        b.emit(ExprOp::Store, b.emit(ExprOp::Immediate, 0, 0, 0, 0, bits(-3.5f)), 0, 0, 3);
        const auto p = b.build();
        std::vector<Tensor> input{floats({1, 99, 2, 99, 3}), floats({4, 5, 6, 7, 8})};
        std::vector<ExprInput> cpu{binding(input[0], {2, 0, 0, 0}), binding(input[1], {0, 1, 0, 0})};
        auto oracle = reference(p, cpu, {{3, 5, 1, 1}, 2}, std::array{DataType::Float32, DataType::Bool, DataType::UInt32, DataType::Float16});
        EXPECT_EQ(value(oracle[0][14]), 73.0f);
        EXPECT_EQ(oracle[2][14], 4u);
        EXPECT_EQ(value(oracle[3][14]), -3.5f);
        check(p, input, {{2, 0, 0, 0}, {0, 1, 0, 0}}, {{3, 5, 1, 1}, 2}, {DataType::Float32, DataType::Bool, DataType::UInt32, DataType::Float16});
    }
    TEST_P(ExpressionTest, GatherPoliciesAndStorageTypes) {
        auto indices = host({uint32_t(-8), uint32_t(-1), 0, 1, 2, 3, 8}, DataType::Int32);
        for (auto type : {DataType::Float32, DataType::Float16, DataType::Int32, DataType::UInt32, DataType::UInt8, DataType::Bool}) {
            auto table = host(type == DataType::Float32 || type == DataType::Float16 ? std::vector<uint32_t>{bits(.25f), bits(.5f), bits(.75f)} : std::vector<uint32_t>{1, 0, 1}, type);
            ExpressionBuilder b;
            auto index = b.emit(ExprOp::Load);
            for (uint8_t out = 0; out < 3; ++out) {
                auto gathered = b.emit(ExprOp::Gather, index, 0, 0, 1, 0, ExprOob(out));
                b.emit(ExprOp::Store, gathered, 0, 0, out);
            }
            check(b.build(), {indices, table}, {{1, 0, 0, 0}, {1, 0, 0, 0}}, {{7, 1, 1, 1}}, {type, type, type});
        }
    }
    TEST_P(ExpressionTest, TerminalReductionOverBroadcastDimension) {
        for (uint32_t code = 0; code <= uint32_t(ExprReduce::Count); ++code) {
            const auto reduction = ExprReduce(code);
            for (auto type : {DataType::Float32, DataType::Int32, DataType::UInt32}) {
                if (type == DataType::Float32 && code > 2)
                    continue;
                for (uint32_t dim = 0; dim < 2; ++dim) {
                    bool floating = type == DataType::Float32;
                    ExpressionBuilder b;
                    auto x = b.emit(ExprOp::Load), edge = b.emit(ExprOp::Iota, 0, 0, 0, uint8_t(dim));
                    if (floating)
                        edge = b.emit(ExprOp::CastFloat, edge);
                    b.emit(ExprOp::Store, b.emit(ExprOp::Fold, b.emit(floating ? ExprOp::Add : ExprOp::AddInt, x, edge), 0, 0, uint8_t(reduction), uint32_t(type), ExprOob(dim)));
                    auto input = floating ? floats({-4, -1, 0, 3, 4}) : type == DataType::Int32 ? host({uint32_t(-4), uint32_t(-1), 0, 3, 4}, type)
                                                                                                : host({0, 1, 2, 3, 4}, type);
                    ExprShape shape{{dim == 0 ? 7u : 5u, dim == 0 ? 5u : 7u, 1, 1}, 2};
                    check(b.build(), {input}, {{dim == 0 ? 0 : 1, dim == 0 ? 1 : 0, 0, 0}}, shape, {type});
                }
            }
        }
    }

    TEST_P(ExpressionTest, LongChainsAndCompensatedSums) {
        ExpressionBuilder chain;
        auto x = chain.emit(ExprOp::Load);
        for (int i = 0; i < 48; ++i)
            x = chain.emit(i % 3 == 2 ? ExprOp::Abs : ExprOp::Neg, x);
        chain.emit(ExprOp::Store, x);
        check(chain.build(), {floats({1, -2, 3.5f})}, {{1, 0, 0, 0}}, {{3, 1, 1, 1}}, {DataType::Float32});

        ExpressionBuilder sum;
        sum.emit(ExprOp::Store, sum.emit(ExprOp::Fold, sum.emit(ExprOp::Load), 0, 0, uint8_t(ExprReduce::Sum), uint32_t(DataType::Float32), ExprOob(1)));
        const ExprShape rows{{2, 4, 1, 1}, 2};
        const auto input = floats({16777216, 1, 1, -16777216, 16777216, 1, 1, -16777216});
        const std::vector<ExprInput> cpu{binding(input, {4, 1, 0, 0})};
        EXPECT_EQ(value(reference(sum.build(), cpu, rows, std::array{DataType::Float32})[0][0]), 2.0f);
        check(sum.build(), {input}, {{4, 1, 0, 0}}, rows, {DataType::Float32});
        const auto infinities = floats({INFINITY, 1, 2, 3, -INFINITY, 1, 2, 3});
        check(sum.build(), {infinities}, {{4, 1, 0, 0}}, rows, {DataType::Float32});
    }

    TEST_P(ExpressionTest, ParameterAndDeviceBindingsOfOneAllocationStaySeparate) {
        if (backend() != GpuBackend::CUDA)
            GTEST_SKIP() << "Host parameters alias device memory only through CUDA managed allocations";
        GpuBackendScope scope(backend());
        float* managed = nullptr;
        ASSERT_EQ(cudaMallocManaged(&managed, 4 * sizeof(float)), cudaSuccess);
        managed[0] = 10;
        managed[1] = 20;
        auto parameter = raw_storage_ref(managed, DataType::Float32);
        parameter.flags |= STORAGE_REF_HOST_MEMORY;
        auto device = raw_storage_ref(managed, DataType::Float32);
        device.byte_offset = sizeof(float);
        auto output = Tensor::zeros({1}, Device::GPU);
        ExpressionBuilder b;
        b.emit(ExprOp::Store, b.emit(ExprOp::Add, b.emit(ExprOp::Load), b.emit(ExprOp::Load, 0, 0, 0, 1)));
        std::array<ExprInput, 2> inputs{ExprInput{parameter}, ExprInput{device}};
        std::array<ExprOutput, 1> outputs{{{storage_ref(output), {1, 0, 0, 0}}}};
        run_expression(b.build(), inputs, outputs, {{1, 1, 1, 1}});
        EXPECT_EQ(output.cpu().to_vector(), std::vector<float>{30.0f});
        EXPECT_EQ(cudaFree(managed), cudaSuccess);
    }

    TEST_P(ExpressionTest, ByteTailAndMoreThan65535Workgroups) {
        GpuBackendScope scope(backend());
        constexpr uint32_t count = 65535u * 256u * 4u + 7;
        auto output = Tensor::empty({size_t(count) + 4}, Device::GPU, DataType::UInt8);
        ExpressionBuilder b;
        auto index = b.emit(ExprOp::Iota);
        b.emit(ExprOp::Store, index);
        std::array<ExprOutput, 1> refs{{{storage_ref(output), {1, 0, 0, 0}}}};
        run_expression(b.build(), {}, refs, {{count, 1, 1, 1}});
        backend_ops(backend()).synchronize_device();
        for (uint32_t i : {0u, 1u, count - 9, count - 8, count - 7, count - 2, count - 1}) {
            const auto bytes = output.slice(0, i, size_t(i) + 1).cpu().to_vector_uint8();
            ASSERT_EQ(bytes.size(), 1u);
            EXPECT_EQ(bytes[0], uint8_t(i));
        }
    }
    TEST_P(ExpressionTest, ExistingOperationsAreBitIdentical) {
        GpuBackendScope scope(backend());
        constexpr uint32_t count = 2048;
        std::vector<uint32_t> a(count), b(count);
        uint32_t random = 1234567;
        for (uint32_t i = 0; i < count; ++i) {
            random ^= random << 13;
            random ^= random >> 17;
            random ^= random << 5;
            a[i] = bits(float(int(random & 65535) - 32768) / 4096.0f);
            b[i] = bits(float(int((random >> 16) & 65535) - 32768) / 8192.0f);
        }
        const uint32_t specials[]{0, 0x80000000u, 1, 0x80000001u, 0x007fffffu, 0x807fffffu,
                                  0x00800000u, 0x80800000u, 0x7f800000u, 0xff800000u,
                                  0x7fc00000u, 0x7fc12345u, 0xffc12345u, 0x7f812345u,
                                  bits(.5f), bits(-.5f), bits(1.5f), bits(-1.5f), bits(2.5f),
                                  bits(-2.5f), bits(1), bits(-1), bits(0x1p24f), bits(-0x1p24f)};
        for (uint32_t i = 0; i < std::size(specials); ++i)
            for (uint32_t j = 0; j < std::size(specials); ++j) {
                a[i * std::size(specials) + j] = specials[i];
                b[i * std::size(specials) + j] = specials[j];
            }
        auto lhs = host(a, DataType::Float32).to(Device::GPU);
        auto rhs = host(b, DataType::Float32).to(Device::GPU);
        const std::pair<ExprOp, PointwiseOp> cases[]{
            {ExprOp::Abs, PointwiseOp::Abs},
            {ExprOp::Neg, PointwiseOp::Neg},
            {ExprOp::Exp, PointwiseOp::Exp},
            {ExprOp::Log, PointwiseOp::Log},
            {ExprOp::Sqrt, PointwiseOp::Sqrt},
            {ExprOp::Sigmoid, PointwiseOp::Sigmoid},
            {ExprOp::Relu, PointwiseOp::Relu},
            {ExprOp::Square, PointwiseOp::Square},
            {ExprOp::Tanh, PointwiseOp::Tanh},
            {ExprOp::Rsqrt, PointwiseOp::Rsqrt},
            {ExprOp::Sign, PointwiseOp::Sign},
            {ExprOp::Reciprocal, PointwiseOp::Reciprocal},
            {ExprOp::Floor, PointwiseOp::Floor},
            {ExprOp::Ceil, PointwiseOp::Ceil},
            {ExprOp::Round, PointwiseOp::Round},
            {ExprOp::Exp2, PointwiseOp::Exp2},
            {ExprOp::Log2, PointwiseOp::Log2},
            {ExprOp::Log10, PointwiseOp::Log10},
            {ExprOp::Log1p, PointwiseOp::Log1p},
            {ExprOp::Sin, PointwiseOp::Sin},
            {ExprOp::Cos, PointwiseOp::Cos},
            {ExprOp::Tan, PointwiseOp::Tan},
            {ExprOp::Asin, PointwiseOp::Asin},
            {ExprOp::Acos, PointwiseOp::Acos},
            {ExprOp::Atan, PointwiseOp::Atan},
            {ExprOp::Sinh, PointwiseOp::Sinh},
            {ExprOp::Cosh, PointwiseOp::Cosh},
            {ExprOp::Gelu, PointwiseOp::Gelu},
            {ExprOp::Swish, PointwiseOp::Swish},
            {ExprOp::Trunc, PointwiseOp::Trunc},
            {ExprOp::IsNan, PointwiseOp::IsNan},
            {ExprOp::IsInf, PointwiseOp::IsInf},
            {ExprOp::IsFinite, PointwiseOp::IsFinite},
            {ExprOp::Add, PointwiseOp::AddTensor},
            {ExprOp::Sub, PointwiseOp::SubTensor},
            {ExprOp::Mul, PointwiseOp::MulTensor},
            {ExprOp::Div, PointwiseOp::DivTensor},
            {ExprOp::Pow, PointwiseOp::PowTensor},
            {ExprOp::Mod, PointwiseOp::ModTensor},
            {ExprOp::Equal, PointwiseOp::EqualTensor},
            {ExprOp::NotEqual, PointwiseOp::NotEqualTensor},
            {ExprOp::Less, PointwiseOp::LessTensor},
            {ExprOp::LessEqual, PointwiseOp::LessEqualTensor},
            {ExprOp::Greater, PointwiseOp::GreaterTensor},
            {ExprOp::GreaterEqual, PointwiseOp::GreaterEqualTensor},
            {ExprOp::Min, PointwiseOp::MinimumTensor},
            {ExprOp::Max, PointwiseOp::MaximumTensor},
        };
        for (auto [op, primitive] : cases) {
            SCOPED_TRACE(pointwise_op_name(primitive));
            const bool predicate = op >= ExprOp::IsNan && op <= ExprOp::GreaterEqual;
            const auto dtype = predicate ? DataType::Bool : DataType::Float32;
            auto generated = Tensor::empty({count + 4}, Device::GPU, dtype);
            auto standalone = Tensor::empty({count + 4}, Device::GPU, dtype);
            ExpressionBuilder builder;
            auto x = builder.emit(ExprOp::Load), y = builder.emit(ExprOp::Load, 0, 0, 0, 1);
            builder.emit(ExprOp::Store, builder.emit(op, x, y));
            std::array<ExprInput, 2> inputs{binding(lhs, {1, 0, 0, 0}), binding(rhs, {1, 0, 0, 0})};
            std::array<ExprOutput, 1> outputs{{{storage_ref(generated), {1, 0, 0, 0}}}};
            run_expression(builder.build(), inputs, outputs, {{count, 1, 1, 1}});
            const PointwiseProgram pointwise{.op = primitive, .in_dtype = DataType::Float32, .out_dtype = dtype};
            if (expr_arity(op) == 1)
                backend_ops(backend()).unary(pointwise, storage_ref(lhs), storage_ref(standalone), count, {});
            else
                backend_ops(backend()).binary(pointwise, storage_ref(lhs), storage_ref(rhs), storage_ref(standalone), count, {});
            const auto actual = generated.cpu(), expected = standalone.cpu();
            unsigned different = 0, first = 0;
            for (uint32_t i = 0; i < count; ++i) {
                if (load(binding(actual, {}), i) != load(binding(expected, {}), i)) {
                    if (!different)
                        first = i;
                    ++different;
                }
            }
            EXPECT_EQ(different, 0u) << "first index=" << first
                                     << " generated=" << load(binding(actual, {}), first)
                                     << " existing=" << load(binding(expected, {}), first);
        }
    }

    TEST_P(ExpressionTest, IntegerAndCastByteIdentity) {
        GpuBackendScope scope(backend());
        constexpr uint32_t count = 2048;
        std::vector<uint32_t> a(count), b(count);
        uint32_t random = 0x13579bdf;
        const uint32_t edges[]{0, 1, 2, 31, 32, 127, 255, 65535, 0x7fffffffu, 0x80000000u, 0xffffffffu};
        for (uint32_t i = 0; i < count; ++i) {
            random ^= random << 13;
            random ^= random >> 17;
            random ^= random << 5;
            a[i] = i < std::size(edges) * std::size(edges) ? edges[i / std::size(edges)] : random;
            b[i] = i < std::size(edges) * std::size(edges) ? edges[i % std::size(edges)] : random ^ 0x9abcdef0;
        }
        for (uint32_t code = uint32_t(ExprOp::LogicalAnd); code <= uint32_t(ExprOp::CastUnsignedFloat); ++code) {
            const auto op = ExprOp(code);
            SCOPED_TRACE("opcode=" + std::to_string(code));
            const bool from_float = op == ExprOp::CastInt || op == ExprOp::CastUInt;
            const bool unsigned_input = op >= ExprOp::UIntDiv || (op >= ExprOp::BitAnd && op <= ExprOp::ShiftRight);
            const auto input_type = from_float ? DataType::Float32 : unsigned_input ? DataType::UInt32
                                                                                    : DataType::Int32;
            auto words = a;
            if (from_float)
                for (uint32_t i = 0; i < count; ++i)
                    words[i] = bits(op == ExprOp::CastUInt ? float(a[i] % 65536) * .5f : float(int32_t(a[i]) % 1000000) * .25f);
            auto rhs_words = b;
            // Native integer division has no defined C++ result for a zero divisor.
            if (op == ExprOp::IntDiv || op == ExprOp::UIntDiv)
                for (auto& value : rhs_words)
                    if (value == 0)
                        value = 1;
            auto lhs_cpu = host(words, input_type), rhs_cpu = host(rhs_words, input_type);
            auto lhs = lhs_cpu.to(Device::GPU), rhs = rhs_cpu.to(Device::GPU);
            const bool to_float = op == ExprOp::CastFloat || op == ExprOp::CastUnsignedFloat;
            const bool predicate = op <= ExprOp::LogicalNot || (op >= ExprOp::EqualInt && op <= ExprOp::GreaterEqualInt) ||
                                   (op >= ExprOp::LessUInt && op <= ExprOp::GreaterEqualUInt);
            const auto out_type = to_float ? DataType::Float32 : predicate                     ? DataType::Bool
                                                             : op == ExprOp::CastInt           ? DataType::Int32
                                                             : input_type == DataType::Float32 ? DataType::UInt32
                                                                                               : input_type;
            auto generated = Tensor::empty({count + 4}, Device::GPU, out_type);
            auto native = Tensor::empty({count + 4}, Device::GPU, out_type);
            ExpressionBuilder builder;
            auto x = builder.emit(ExprOp::Load), y = builder.emit(ExprOp::Load, 0, 0, 0, 1);
            builder.emit(ExprOp::Store, builder.emit(op, x, y));
            std::array<ExprInput, 2> inputs{binding(lhs, {1, 0, 0, 0}), binding(rhs, {1, 0, 0, 0})};
            std::array<ExprOutput, 1> outputs{{{storage_ref(generated), {1, 0, 0, 0}}}};
            run_expression(builder.build(), inputs, outputs, {{count, 1, 1, 1}});
            PointwiseOp primitive = PointwiseOp::Count;
            switch (op) {
            case ExprOp::AddInt: primitive = PointwiseOp::AddTensor; break;
            case ExprOp::SubInt: primitive = PointwiseOp::SubTensor; break;
            case ExprOp::MulInt: primitive = PointwiseOp::MulTensor; break;
            case ExprOp::IntDiv:
            case ExprOp::UIntDiv: primitive = PointwiseOp::DivTensor; break;
            case ExprOp::IntMod:
            case ExprOp::UIntMod: primitive = PointwiseOp::ModTensor; break;
            case ExprOp::EqualInt: primitive = PointwiseOp::EqualTensor; break;
            case ExprOp::NotEqualInt: primitive = PointwiseOp::NotEqualTensor; break;
            case ExprOp::LessInt:
            case ExprOp::LessUInt: primitive = PointwiseOp::LessTensor; break;
            case ExprOp::LessEqualInt:
            case ExprOp::LessEqualUInt: primitive = PointwiseOp::LessEqualTensor; break;
            case ExprOp::GreaterInt:
            case ExprOp::GreaterUInt: primitive = PointwiseOp::GreaterTensor; break;
            case ExprOp::GreaterEqualInt:
            case ExprOp::GreaterEqualUInt: primitive = PointwiseOp::GreaterEqualTensor; break;
            case ExprOp::LogicalAnd: primitive = PointwiseOp::LogicalAndTensor; break;
            case ExprOp::LogicalOr: primitive = PointwiseOp::LogicalOrTensor; break;
            case ExprOp::LogicalXor: primitive = PointwiseOp::LogicalXorTensor; break;
            case ExprOp::LogicalNot: primitive = PointwiseOp::LogicalNot; break;
            default: break;
            }
            const bool native_reference = primitive != PointwiseOp::Count || from_float || to_float;
            if (from_float || to_float)
                backend_ops(backend()).convert_type(storage_ref(lhs), storage_ref(native), count, {});
            else if (primitive != PointwiseOp::Count) {
                PointwiseProgram p{.op = primitive, .in_dtype = input_type, .out_dtype = out_type};
                if (op == ExprOp::LogicalNot)
                    backend_ops(backend()).unary(p, storage_ref(lhs), storage_ref(native), count, {});
                else
                    backend_ops(backend()).binary(p, storage_ref(lhs), storage_ref(rhs), storage_ref(native), count, {});
            }
            const auto actual = generated.cpu();
            const auto expected = native_reference ? native.cpu() : Tensor{};
            uint32_t different = 0, first = 0;
            for (uint32_t i = 0; i < count; ++i) {
                auto wanted = native_reference ? load(binding(expected, {}), i) : reference_op(op, words[i], rhs_words[i], 0);
                if (load(binding(actual, {}), i) != wanted) {
                    if (!different)
                        first = i;
                    ++different;
                }
            }
            EXPECT_EQ(different, 0u) << "first index=" << first;
        }
    }

    TEST_P(ExpressionTest, MultiOperationDagsAreBitIdentical) {
        GpuBackendScope scope(backend());
        constexpr uint32_t count = 2048;
        std::vector<uint32_t> a(count), b(count);
        uint32_t random = 1234567;
        for (uint32_t i = 0; i < count; ++i) {
            random ^= random << 13;
            random ^= random >> 17;
            random ^= random << 5;
            a[i] = bits(float(int(random & 65535) - 32768) / 16384.0f);
            b[i] = bits(float((random >> 16) & 65535) / 32768.0f);
        }
        auto lhs = host(a, DataType::Float32).to(Device::GPU), rhs = host(b, DataType::Float32).to(Device::GPU);
        for (auto [op, primitive] : {std::pair{ExprOp::Exp, PointwiseOp::Exp}, {ExprOp::Gelu, PointwiseOp::Gelu}, {ExprOp::Sigmoid, PointwiseOp::Sigmoid}, {ExprOp::Tanh, PointwiseOp::Tanh}, {ExprOp::Sin, PointwiseOp::Sin}, {ExprOp::Log1p, PointwiseOp::Log1p}}) {
            SCOPED_TRACE(pointwise_op_name(primitive));
            ExpressionBuilder builder;
            auto x = builder.emit(ExprOp::Load), y = builder.emit(ExprOp::Load, 0, 0, 0, 1);
            auto v = builder.emit(op, x);
            auto square = builder.emit(ExprOp::Square, v);
            auto product = builder.emit(ExprOp::Mul, square, x);
            builder.emit(ExprOp::Store, builder.emit(ExprOp::Add, product, y));
            builder.emit(ExprOp::Store, builder.emit(ExprOp::Select, builder.emit(ExprOp::Less, x, y), v, square), 0, 0, 1);
            auto generated = Tensor::empty({count}, Device::GPU), selected = Tensor::empty({count}, Device::GPU);
            std::array<ExprInput, 2> inputs{binding(lhs, {1, 0, 0, 0}), binding(rhs, {1, 0, 0, 0})};
            std::array<ExprOutput, 2> outputs{{{storage_ref(generated), {1, 0, 0, 0}}, {storage_ref(selected), {1, 0, 0, 0}}}};
            run_expression(builder.build(), inputs, outputs, {{count, 1, 1, 1}});
            auto native_v = Tensor::empty({count}, Device::GPU), native_square = Tensor::empty({count}, Device::GPU);
            auto native_product = Tensor::empty({count}, Device::GPU), native = Tensor::empty({count}, Device::GPU);
            auto& ops = backend_ops(backend());
            const auto unary = [&](PointwiseOp p, const Tensor& src, const Tensor& dst) {
                ops.unary({.op = p, .in_dtype = DataType::Float32, .out_dtype = DataType::Float32}, storage_ref(src), storage_ref(dst), count, {});
            };
            const auto binary = [&](PointwiseOp p, const Tensor& x, const Tensor& y, const Tensor& dst) {
                ops.binary({.op = p, .in_dtype = DataType::Float32, .out_dtype = DataType::Float32}, storage_ref(x), storage_ref(y), storage_ref(dst), count, {});
            };
            unary(primitive, lhs, native_v);
            unary(PointwiseOp::Square, native_v, native_square);
            binary(PointwiseOp::MulTensor, native_square, lhs, native_product);
            binary(PointwiseOp::AddTensor, native_product, rhs, native);
            const auto actual = generated.cpu(), chosen = selected.cpu(), expected = native.cpu();
            const auto value_cpu = native_v.cpu(), square_cpu = native_square.cpu();
            uint32_t different = 0;
            for (uint32_t i = 0; i < count; ++i) {
                different += load(binding(actual, {}), i) != load(binding(expected, {}), i);
                different += load(binding(chosen, {}), i) != load(binding(value(a[i]) < value(b[i]) ? value_cpu : square_cpu, {}), i);
            }
            EXPECT_EQ(different, 0u);
        }
    }

    TEST_P(ExpressionTest, TernaryAndAtan2Identity) {
        GpuBackendScope scope(backend());
        constexpr uint32_t count = 2048;
        std::vector<uint32_t> a(count), b(count), c(count), mask(count);
        uint32_t random = 0xabcdef;
        for (uint32_t i = 0; i < count; ++i) {
            random ^= random << 13;
            random ^= random >> 17;
            random ^= random << 5;
            a[i] = bits(float(int(random & 65535) - 32768) / 16384.0f);
            b[i] = bits(float(int(random >> 16) - 32768) / 8192.0f);
            c[i] = bits(float(int(random & 32767) - 16384) / 8192.0f);
            mask[i] = random & 1;
        }
        const uint32_t special[]{0, 0x80000000u, 1, 0x80000001u, 0x7f800000u, 0xff800000u, 0x7fc12345u};
        for (uint32_t i = 0; i < std::size(special); ++i)
            for (uint32_t j = 0; j < std::size(special); ++j) {
                a[i * std::size(special) + j] = special[i];
                b[i * std::size(special) + j] = special[j];
            }
        auto lhs = host(a, DataType::Float32).to(Device::GPU), rhs = host(b, DataType::Float32).to(Device::GPU);
        auto third = host(c, DataType::Float32).to(Device::GPU), condition = host(mask, DataType::Bool).to(Device::GPU);
        const StridedLayout layout{.rank = 1, .dims = {count}, .strides = {1}, .element_count = count};
        for (auto op : {ExprOp::Atan2, ExprOp::Fma, ExprOp::Clamp, ExprOp::Select}) {
            SCOPED_TRACE(uint32_t(op));
            auto generated = Tensor::empty({count}, Device::GPU), native = Tensor::empty({count}, Device::GPU);
            ExpressionBuilder builder;
            auto x = builder.emit(ExprOp::Load), y = builder.emit(ExprOp::Load, 0, 0, 0, 1), z = builder.emit(ExprOp::Load, 0, 0, 0, 2);
            if (op == ExprOp::Clamp) {
                y = builder.emit(ExprOp::Immediate, 0, 0, 0, 0, bits(-.5f));
                z = builder.emit(ExprOp::Immediate, 0, 0, 0, 0, bits(.75f));
            }
            builder.emit(ExprOp::Store, builder.emit(op, x, y, z));
            std::array<ExprInput, 3> inputs{binding(op == ExprOp::Select ? condition : lhs, {1, 0, 0, 0}), binding(rhs, {1, 0, 0, 0}), binding(third, {1, 0, 0, 0})};
            std::array<ExprOutput, 1> outputs{{{storage_ref(generated), {1, 0, 0, 0}}}};
            run_expression(builder.build(), inputs, outputs, {{count, 1, 1, 1}});
            auto& ops = backend_ops(backend());
            if (op == ExprOp::Fma) {
                // One rounding; subnormal lanes (CUDA flushes them) and NaN payloads are not compared.
                const auto generated_words = generated.cpu();
                std::vector<uint32_t> expected(count);
                for (uint32_t i = 0; i < count; ++i) {
                    const float result = std::fma(value(a[i]), value(b[i]), value(c[i]));
                    const bool subnormal = std::fpclassify(value(a[i])) == FP_SUBNORMAL || std::fpclassify(value(b[i])) == FP_SUBNORMAL ||
                                           std::fpclassify(value(c[i])) == FP_SUBNORMAL || std::fpclassify(result) == FP_SUBNORMAL;
                    const uint32_t produced = load(binding(generated_words, {}), i);
                    expected[i] = subnormal || (std::isnan(result) && std::isnan(value(produced))) ? produced : bits(result);
                }
                native = host(expected, DataType::Float32).to(Device::GPU);
            } else if (op == ExprOp::Clamp) {
                ScalarOperand lo, hi;
                lo.value.float_value = -.5f;
                hi.value.float_value = .75f;
                ops.clamp_fused(storage_ref(lhs), storage_ref(native), lo, hi, count, {});
            } else if (op == ExprOp::Select)
                ops.where(storage_ref(condition), storage_ref(rhs), storage_ref(third), storage_ref(native), layout, layout, layout, layout, {});
            else if (backend() == GpuBackend::CUDA)
                expression_native_atan2(lhs.ptr<float>(), rhs.ptr<float>(), native.ptr<float>(), count);
            const auto actual = generated.cpu();
            if (op == ExprOp::Atan2 && backend() == GpuBackend::Vulkan) {
                for (uint32_t i = std::size(special) * std::size(special); i < count; ++i)
                    EXPECT_NEAR(value(load(binding(actual, {}), i)), std::atan2(value(a[i]), value(b[i])), 2e-6f);
            } else {
                const auto expected = native.cpu();
                uint32_t different = 0, first = 0;
                for (uint32_t i = 0; i < count; ++i)
                    if (load(binding(actual, {}), i) != load(binding(expected, {}), i)) {
                        if (!different)
                            first = i;
                        ++different;
                    }
                EXPECT_EQ(different, 0u) << "first=" << first << " generated=" << load(binding(actual, {}), first) << " native=" << load(binding(expected, {}), first);
            }
        }
    }

    TEST_P(ExpressionTest, ParametersCacheAndConcurrentCompilation) {
        GpuBackendScope scope(backend());
        constexpr uint32_t count = 37;
        auto input = host(std::vector<uint32_t>(count, bits(1.25f)), DataType::Float32).to(Device::GPU);
        std::array<Tensor, 2> outputs{Tensor::empty({count}, Device::GPU), Tensor::empty({count}, Device::GPU)};
        ExpressionBuilder builder;
        auto x = builder.emit(ExprOp::Load);
        auto parameter = builder.emit(ExprOp::Load, 0, 0, 0, 1);
        builder.emit(ExprOp::Store, builder.emit(ExprOp::Mul, x, parameter));
        const auto program = builder.build();
        auto before = expression_cache_stats(backend());
        std::barrier start(2);
        std::array<std::future<void>, 2> threads;
        for (size_t i = 0; i < threads.size(); ++i) {
            threads[i] = std::async(std::launch::async, [&, i] {
                GpuBackendScope thread_scope(backend());
                if (backend() == GpuBackend::CUDA && cudaSetDevice(0) != cudaSuccess)
                    throw std::runtime_error("cannot select CUDA test device");
                const float value = float(i + 2);
                auto parameter_storage = raw_storage_ref(const_cast<float*>(&value), DataType::Float32);
                parameter_storage.flags |= STORAGE_REF_HOST_MEMORY;
                std::array<ExprInput, 2> inputs{binding(input, {1, 0, 0, 0}), ExprInput{parameter_storage}};
                std::array<ExprOutput, 1> output{{{storage_ref(outputs[i]), {1, 0, 0, 0}}}};
                start.arrive_and_wait();
                run_expression(program, inputs, output, {{count, 1, 1, 1}});
            });
        }
        for (auto& thread : threads)
            thread.get();
        for (size_t i = 0; i < outputs.size(); ++i)
            for (float actual : outputs[i].cpu().to_vector())
                EXPECT_EQ(actual, 1.25f * float(i + 2));
        const auto compiled = expression_cache_stats(backend());
        EXPECT_EQ((compiled.compilations + compiled.disk_hits) - (before.compilations + before.disk_hits), 1u);
        EXPECT_EQ(compiled.hits - before.hits, 1u);
        float value = 5;
        auto parameter_storage = raw_storage_ref(&value, DataType::Float32);
        parameter_storage.flags |= STORAGE_REF_HOST_MEMORY;
        std::array<ExprInput, 2> inputs{binding(input, {1, 0, 0, 0}), ExprInput{parameter_storage}};
        std::array<ExprOutput, 1> output{{{storage_ref(outputs[0]), {1, 0, 0, 0}}}};
        run_expression(program, inputs, output, {{count, 1, 1, 1}});
        for (float actual : outputs[0].cpu().to_vector())
            EXPECT_EQ(actual, 6.25f);
        const auto changed = expression_cache_stats(backend());
        EXPECT_EQ(changed.compilations, compiled.compilations);
        EXPECT_EQ(changed.disk_hits, compiled.disk_hits);
        EXPECT_EQ(changed.hits, compiled.hits + 1);
        inputs[0].dims[0] = 19;
        run_expression(program, inputs, output, {{19, 1, 1, 1}});
        inputs[0].dims[0] = count;
        const auto reshaped = expression_cache_stats(backend());
        EXPECT_EQ(reshaped.compilations + reshaped.disk_hits, changed.compilations + changed.disk_hits);

        ExpressionBuilder warm_builder;
        auto warm_x = warm_builder.emit(ExprOp::Load);
        auto warm_parameter = warm_builder.emit(ExprOp::Load, 0, 0, 0, 1);
        auto one = warm_builder.emit(ExprOp::Immediate, 0, 0, 0, 0, bits(1));
        warm_builder.emit(ExprOp::Store, warm_builder.emit(ExprOp::Add,
                                                           warm_builder.emit(ExprOp::Mul, warm_x, warm_parameter), one));
        const auto warm_program = warm_builder.build();
        prepare_expression(warm_program, inputs, output, {{count, 1, 1, 1}});
        value = 11;
        const auto warmed = expression_cache_stats(backend());
        for (float actual : outputs[0].cpu().to_vector())
            EXPECT_EQ(actual, 6.25f);
        run_expression(warm_program, inputs, output, {{count, 1, 1, 1}});
        for (float actual : outputs[0].cpu().to_vector())
            EXPECT_EQ(actual, 14.75f);
        EXPECT_EQ(expression_cache_stats(backend()).compilations, warmed.compilations);
    }

    TEST_P(ExpressionTest, StreamAndRecorderDependencies) {
        GpuBackendScope scope(backend());
        constexpr uint32_t count = 11;
        auto input = Tensor::zeros({count}, Device::GPU);
        auto output = Tensor::zeros({count}, Device::GPU);
        ExpressionBuilder builder;
        auto x = builder.emit(ExprOp::Load);
        auto one = builder.emit(ExprOp::Immediate, 0, 0, 0, 0, bits(1));
        builder.emit(ExprOp::Store, builder.emit(ExprOp::Add, x, one));
        const auto program = builder.build();
        std::array<ExprInput, 1> inputs{binding(input, {1, 0, 0, 0})};
        std::array<ExprOutput, 1> outputs{{{storage_ref(output), {1, 0, 0, 0}}}};
        const ExprShape shape{{count, 1, 1, 1}};
        prepare_expression(program, inputs, outputs, shape);
        if (backend() == GpuBackend::CUDA) {
            cudaStream_t stream = nullptr;
            ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
            ScalarOperand fill;
            fill.value.float_value = 2;
            backend_ops(backend()).load_fill(storage_ref(input), count, fill, {.cuda_stream = stream});
            run_expression(program, inputs, outputs, shape, {.cuda_stream = stream});
            ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
            fill.value.float_value = 0;
            backend_ops(backend()).load_fill(storage_ref(input), count, fill, {.cuda_stream = stream});
            ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
            std::atomic<bool> released = false;
            struct Release {
                std::atomic<bool>& flag;
                cudaStream_t stream;
                ~Release() {
                    flag.store(true);
                    flag.notify_all();
                    cudaStreamDestroy(stream);
                }
            } cleanup{released, stream};
            ASSERT_EQ(cudaLaunchHostFunc(stream, hold_stream, &released), cudaSuccess);
            fill.value.float_value = 2;
            backend_ops(backend()).load_fill(storage_ref(input), count, fill, {.cuda_stream = stream});
            run_expression(program, inputs, outputs, shape, {.cuda_stream = stream});
            ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
            released.store(true);
            released.notify_all();
            ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
            for (float actual : output.cpu().to_vector())
                EXPECT_EQ(actual, 3);
        } else {
            std::async(std::launch::async, [&] {
                run_expression(program, inputs, outputs, shape);
            }).get();
            auto result = Tensor::empty({count}, Device::GPU);
            inputs[0] = binding(output, {1, 0, 0, 0});
            outputs[0].storage = storage_ref(result);
            std::async(std::launch::async, [&] {
                run_expression(program, inputs, outputs, shape);
            }).get();
            for (float actual : result.cpu().to_vector())
                EXPECT_EQ(actual, 2);
        }
    }

    TEST(ExpressionCacheTest, PersistsAndBoundsArtifacts) {
        struct Kernel : CompiledExpression {
            char value;
        };
        const auto prefix = "test:" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
        unsigned compilations = 0;
        const auto compile = [&] { ++compilations; return std::vector<char>{42}; };
        const auto load = [](std::span<const char> bytes) {
            auto kernel = std::make_shared<Kernel>();
            kernel->value = bytes.front();
            return kernel;
        };
        {
            ExpressionCache cache;
            EXPECT_EQ(std::static_pointer_cast<Kernel>(cache.get(prefix, compile, load))->value, 42);
            cache.get(prefix, compile, load);
            EXPECT_EQ(compilations, 1u);
        }
        ExpressionCache cache;
        EXPECT_EQ(std::static_pointer_cast<Kernel>(cache.get(prefix, compile, load))->value, 42);
        EXPECT_EQ(compilations, 1u);
        EXPECT_EQ(cache.stats().disk_hits, 1u);
        for (int i = 0; i < 160; ++i)
            cache.get(prefix + ":" + std::to_string(i), compile, load);
        EXPECT_GT(cache.stats().evictions, 0u);
        EXPECT_LT(cache.stats().entries, 160u);
        EXPECT_EQ(std::static_pointer_cast<Kernel>(cache.get(prefix, compile, load))->value, 42);
    }

    TEST_P(ExpressionTest, TypedFrontendChangingShapesAndBroadcast) {
        GpuBackendScope scope(backend());
        fused::Builder b(2);
        auto a = b.input(DataType::Float32, 2).load();
        auto bias = b.input(DataType::Float32, 1).load();
        auto gain = b.input(DataType::Float32, 1).at({0});
        b.output(fused::fma(a, gain, bias) + b.iota(0).cast(DataType::Float32), DataType::Float32);
        fused::Kernel kernel(b);
        ExpressionCacheStats warmed;
        for (size_t n : {3, 5, 7, 11, 19}) {
            auto a_cpu = Tensor::empty_pageable_host({n, n * 2});
            auto bias_cpu = Tensor::empty_pageable_host({1});
            auto gain_cpu = floats({2});
            auto* data = a_cpu.ptr<float>();
            for (size_t i = 0; i < a_cpu.numel(); ++i)
                data[i] = float(i);
            bias_cpu.ptr<float>()[0] = 3;
            auto input = a_cpu.to(Device::GPU).slice(1, 0, n);
            auto bias_gpu = bias_cpu.to(Device::GPU);
            kernel.prepare({n, n}, {input, bias_gpu, gain_cpu}, {});
            if (n == 3)
                warmed = expression_cache_stats(backend());
            auto actual = kernel({n, n}, {input, bias_gpu, gain_cpu})[0].cpu().to_vector();
            for (size_t row = 0; row < n; ++row)
                for (size_t col = 0; col < n; ++col)
                    EXPECT_EQ(actual[row * n + col], float((row * n * 2 + col) * 2 + 3 + row));
            const auto stats = expression_cache_stats(backend());
            EXPECT_EQ(stats.compilations, warmed.compilations);
            EXPECT_EQ(stats.disk_hits, warmed.disk_hits);
            EXPECT_EQ(stats.loads, warmed.loads);
            EXPECT_EQ(stats.entries, warmed.entries);
        }
    }

    TEST_P(ExpressionTest, TypedGatherPerDimensionBounds) {
        GpuBackendScope scope(backend());
        for (auto bounds : {fused::Bounds::Clamp, fused::Bounds::Zero, fused::Bounds::Wrap}) {
            fused::Builder b(2);
            auto table = b.input(DataType::Float32, 2);
            b.output(table.gather({b.iota(0) - 1, b.iota(1) - 1}, bounds), DataType::Float32);
            fused::Kernel kernel(b);
            auto table_cpu = floats({1, 2, 3, 4, 5, 6}).reshape({2, 3});
            auto actual = kernel({4, 5}, {table_cpu.to(Device::GPU)})[0].cpu().to_vector();
            for (int y = 0; y < 4; ++y)
                for (int x = 0; x < 5; ++x) {
                    int row = y - 1, col = x - 1;
                    float wanted = 0;
                    if (bounds == fused::Bounds::Clamp) {
                        row = std::clamp(row, 0, 1);
                        col = std::clamp(col, 0, 2);
                    }
                    if (bounds == fused::Bounds::Wrap) {
                        row = (row % 2 + 2) % 2;
                        col = (col % 3 + 3) % 3;
                    }
                    if (row >= 0 && row < 2 && col >= 0 && col < 3)
                        wanted = float(row * 3 + col + 1);
                    EXPECT_EQ(actual[y * 5 + x], wanted);
                }
        }
    }

    TEST_P(ExpressionTest, TypedFoldEpilogueAndChangingLength) {
        GpuBackendScope scope(backend());
        for (auto kind : {fused::Fold::Sum, fused::Fold::Min, fused::Fold::Max, fused::Fold::And,
                          fused::Fold::Or, fused::Fold::Xor, fused::Fold::Count}) {
            fused::Builder b(2);
            auto x = b.input(DataType::Int32, 2).load();
            auto old = b.input(DataType::Int32, 1).load({0});
            auto reduced = b.fold(kind, x, 1);
            b.output(fused::where(reduced != 0, reduced, old), DataType::Int32);
            fused::Kernel kernel(b);
            ExpressionCacheStats warmed;
            for (size_t n : {2, 3, 5, 7, 11, 65, 67, 71, 73, 79}) {
                std::vector<uint32_t> data(n * 3);
                for (size_t i = 0; i < data.size(); ++i)
                    data[i] = i % 5;
                auto input = host(data, DataType::Int32).reshape({3, int(n)}).to(Device::GPU);
                auto old_gpu = host({99, 99, 99}, DataType::Int32).to(Device::GPU);
                kernel.prepare({3, n}, {input, old_gpu}, {});
                if (n <= 65)
                    warmed = expression_cache_stats(backend());
                auto actual = kernel({3, n}, {input, old_gpu})[0].cpu();
                for (size_t row = 0; row < 3; ++row) {
                    uint32_t acc = kind == fused::Fold::And ? ~0u : kind == fused::Fold::Min ? INT32_MAX
                                                                                             : 0;
                    for (size_t col = 0; col < n; ++col) {
                        auto v = data[row * n + col];
                        switch (kind) {
                        case fused::Fold::Sum: acc += v; break;
                        case fused::Fold::Min: acc = std::min(acc, v); break;
                        case fused::Fold::Max: acc = std::max(acc, v); break;
                        case fused::Fold::And: acc &= v; break;
                        case fused::Fold::Or: acc |= v; break;
                        case fused::Fold::Xor: acc ^= v; break;
                        case fused::Fold::Count: acc += v != 0; break;
                        }
                    }
                    EXPECT_EQ(actual.ptr<int32_t>()[row], int32_t(acc ? acc : 99));
                }
                auto stats = expression_cache_stats(backend());
                EXPECT_EQ(stats.compilations, warmed.compilations);
                EXPECT_EQ(stats.disk_hits, warmed.disk_hits);
                EXPECT_EQ(stats.loads, warmed.loads);
            }
        }
    }

    TEST_P(ExpressionTest, ShortFoldGatherBoundsAreCheckedEveryCall) {
        GpuBackendScope scope(backend());
        fused::Builder b(2);
        auto table = b.input(DataType::Int32, 1);
        for (auto bounds : {fused::Bounds::Clamp, fused::Bounds::Zero, fused::Bounds::Wrap})
            b.output(b.fold(fused::Fold::Sum, table.gather({b.iota(1)}, bounds), 1), DataType::Int32);
        fused::Kernel kernel(b);
        for (const size_t n : {4, 2, 5, 4}) {
            const std::vector<uint32_t> values{2, 3, 5, 7, 11};
            auto input = host(std::vector<uint32_t>(values.begin(), values.begin() + n), DataType::Int32).to(Device::GPU);
            auto outputs = kernel({2, 4}, {input});
            for (size_t o = 0; o < outputs.size(); ++o) {
                int32_t expected = 0;
                for (size_t i = 0; i < 4; ++i)
                    if (i < n || o != 1)
                        expected += values[o == 0 ? std::min(i, n - 1) : i % n];
                const auto actual = outputs[o].cpu();
                EXPECT_EQ(actual.ptr<int32_t>()[0], expected);
                EXPECT_EQ(actual.ptr<int32_t>()[1], expected);
            }
        }
    }

    TEST_P(ExpressionTest, RuntimeFoldConditionalBoundsAreCheckedEveryCall) {
        GpuBackendScope scope(backend());
        fused::Builder b(2);
        auto table = b.input(DataType::Int32, 1);
        auto index = fused::where(b.iota(1) == 0, b.extent(1) - 1, b.iota(1) - 1);
        for (auto bounds : {fused::Bounds::Clamp, fused::Bounds::Zero, fused::Bounds::Wrap})
            b.output(b.fold(fused::Fold::Sum, table.gather({index}, bounds), 1), DataType::Int32);
        fused::Kernel kernel(b);
        for (const size_t n : {65, 63, 70, 65}) {
            std::vector<uint32_t> values(n);
            for (size_t i = 0; i < n; ++i)
                values[i] = uint32_t(i + 1);
            auto input = host(values, DataType::Int32).gpu();
            kernel.prepare({2, 65}, {input}, {});
            auto outputs = kernel({2, 65}, {input});
            for (size_t o = 0; o < outputs.size(); ++o) {
                int32_t expected = 0;
                for (size_t i = 0; i < 65; ++i) {
                    const size_t j = i ? i - 1 : 64;
                    if (j < n || o != 1)
                        expected += values[o == 0 ? std::min(j, n - 1) : j % n];
                }
                EXPECT_EQ(outputs[o].cpu().to_vector_int(), (std::vector<int>{expected, expected}));
            }
        }
    }

    TEST_P(ExpressionTest, SharedConstrainedBoundsPrepareQuickly) {
        ExpressionBuilder b;
        const auto i = b.emit(ExprOp::Iota, 0, 0, 0, 1);
        const auto zero = b.emit(ExprOp::Immediate, 0, 0, 0, 0, 0);
        const auto predicate = b.emit(ExprOp::EqualInt, i, zero);
        auto repeated = i;
        for (int n = 0; n < 28; ++n)
            repeated = b.emit(ExprOp::AddInt, repeated, repeated);
        const auto index = b.emit(ExprOp::Select, predicate, repeated, zero);
        const auto value = b.emit(ExprOp::Gather, index, 0, 0, 0);
        const auto folded = b.emit(ExprOp::Fold, value, 0, 0, uint8_t(ExprReduce::Sum),
                                   uint32_t(DataType::Int32), ExprOob(1));
        b.emit(ExprOp::Store, folded);
        auto program = b.build();
        ASSERT_EQ(program.instructions().size(), 35u);
        std::array<std::array<uint32_t, ExpressionProgram::max_rank>, ExpressionProgram::max_inputs> bounds{};
        bounds[0][0] = 1;
        const ExprShape shape{{2, 65, 1, 1}, 2};
        const auto started = std::chrono::steady_clock::now();
        for (int n = 0; n < 100; ++n)
            ASSERT_TRUE(expression_gather_in_range(program, shape, bounds));
        const auto elapsed = std::chrono::steady_clock::now() - started;
        EXPECT_LT((std::chrono::duration<double, std::milli>(elapsed).count() / 100), 0.5);
        GpuBackendScope scope(backend());
        auto table = Tensor::from_vector(std::vector<int>{7}, {1}, Device::GPU);
        auto output = Tensor::zeros({2}, Device::GPU, DataType::Int32);
        ExprInput input = binding(table, {});
        input.dims[0] = 1;
        ExprOutput target{storage_ref(output)};
        target.strides[0] = 1;
        const std::array<ExprInput, 1> inputs{input};
        const std::array<ExprOutput, 1> outputs{target};
        prepare_expression(program, inputs, outputs, shape);
        const auto prepare_started = std::chrono::steady_clock::now();
        for (int n = 0; n < 20; ++n)
            prepare_expression(program, inputs, outputs, shape);
        const auto prepare_elapsed = std::chrono::steady_clock::now() - prepare_started;
        EXPECT_LT((std::chrono::duration<double, std::milli>(prepare_elapsed).count() / 20), 0.8);
        run_expression(program, inputs, outputs, shape);
        EXPECT_EQ(output.cpu().to_vector_int(), (std::vector<int>{455, 455}));
    }

    TEST_P(ExpressionTest, GatherPredicateSelectionPreservesBits) {
        GpuBackendScope scope(backend());
        const std::vector<uint32_t> patterns{0, 0x80000000u, 0x7fc12345u, 0xffc54321u, 0x3f800000u, 0xff800000u};
        auto input = host(patterns, DataType::Float32).gpu();
        auto indices = Tensor::from_vector(std::vector<int>{0, 1, 2, 3, -1, 99}, {6}, Device::GPU);
        auto table = Tensor::from_vector(std::vector<int>{0, 1, 1, 0}, {4}, Device::GPU);
        fused::Builder b(1);
        auto value = b.input(DataType::Float32, 1).load();
        auto index = b.input(DataType::Int32, 1).load();
        auto allowed = b.input(DataType::Int32, 1);
        b.output(fused::where(allowed.gather({index}) != 0, value, 0.f), DataType::Float32);
        fused::Kernel kernel(b);
        kernel.prepare({6}, {input, indices, table}, {});
        const auto output = kernel({6}, {input, indices, table})[0].cpu();
        const uint32_t expected[]{0, 0x80000000u, 0x7fc12345u, 0, 0, 0};
        EXPECT_EQ(std::memcmp(output.data_ptr(), expected, sizeof(expected)), 0);
    }

    TEST_P(ExpressionTest, HalfSelectionPreservesEveryPayload) {
        GpuBackendScope scope(backend());
        constexpr size_t count = 65536;
        auto host = Tensor::empty({count}, Device::CPU, DataType::Float16);
        auto* bits = static_cast<uint16_t*>(host.data_ptr());
        for (size_t i = 0; i < count; ++i)
            bits[i] = uint16_t(i);
        auto input = host.gpu();
        fused::Builder b(1);
        auto source = b.input(DataType::Float16, 1).load();
        auto mask = b.input(DataType::Bool, 1).load();
        b.output(fused::where(mask, -20.f, source), DataType::Float16);
        fused::Kernel kernel(b);
        auto condition = Tensor::zeros({count}, Device::GPU, DataType::Bool);
        kernel.prepare({count}, {input, condition}, {});
        auto output = kernel({count}, {input, condition})[0].cpu();
        EXPECT_EQ(std::memcmp(host.data_ptr(), output.data_ptr(), count * 2), 0);
        auto selected = Tensor::empty({count}, Device::CPU, DataType::Bool);
        auto* mask_bits = static_cast<uint8_t*>(selected.data_ptr());
        for (size_t i = 0; i < count; ++i)
            mask_bits[i] = i % 3 ? 0 : 7;
        condition = selected.gpu();
        output = kernel({count}, {input, condition})[0].cpu();
        const auto* actual = static_cast<const uint16_t*>(output.data_ptr());
        for (size_t i = 0; i < count; ++i)
            ASSERT_EQ(actual[i], mask_bits[i] ? 0xcd00u : uint16_t(i)) << i;
    }

    TEST_P(ExpressionTest, DeadBindingsDoNotAliasOrFillArguments) {
        GpuBackendScope scope(backend());
        fused::Builder b(1);
        (void)b.input(DataType::Float32, 1).load();
        b.output(b.constant(1.f), DataType::Float32);
        fused::Kernel kernel(b);
        auto output = Tensor::zeros({3}, Device::GPU);
        kernel({3}, {output}, {output});
        EXPECT_EQ(output.cpu().to_vector(), (std::vector<float>{1, 1, 1}));

        ExpressionBuilder raw;
        raw.emit(ExprOp::Load, 0, 0, 0, 0);
        const auto one = raw.emit(ExprOp::Immediate, 0, 0, 0, 0, bits(1.f));
        raw.emit(ExprOp::Store, one);
        auto program = raw.build();
        EXPECT_TRUE(program.compact_inputs().empty());
        ExpressionSignature signature;
        signature.inputs = 0;
        signature.outputs = 1;
        EXPECT_EQ(expression_layout(program, signature).words * 4, 16u);
        auto unused = host(std::vector<uint32_t>(64, bits(2.f)), DataType::Float32);
        kernel.prepare({3}, {unused}, {output});
        fused::Builder remapped(1);
        (void)remapped.input(DataType::Float32, 1).load();
        remapped.output(remapped.input(DataType::Float32, 1).load(), DataType::Float32);
        fused::Kernel copied(remapped);
        auto source = floats({4, 5, 6}).gpu();
        EXPECT_EQ(copied({3}, {unused, source})[0].cpu().to_vector(),
                  (std::vector<float>{4, 5, 6}));
    }

    TEST_P(ExpressionTest, PackedExternalTailRequiresCapacity) {
        GpuBackendScope scope(backend());
        for (auto dtype : {DataType::UInt8, DataType::Bool, DataType::Float16}) {
            fused::Builder b(1);
            b.output(dtype == DataType::Float16 ? b.constant(1.f) : dtype == DataType::Bool ? b.constant(true)
                                                                                            : b.constant(1u),
                     dtype);
            fused::Kernel kernel(b);
            if (backend() == GpuBackend::CUDA) {
                void* pointer = nullptr;
                ASSERT_EQ(cudaMalloc(&pointer, dtype_size(dtype)), cudaSuccess);
                auto external = Tensor::from_blob(pointer, {1}, Device::GPU, dtype);
                EXPECT_THROW(kernel.prepare({1}, {}, {external}), std::invalid_argument);
                ASSERT_EQ(cudaFree(pointer), cudaSuccess);
            } else {
                auto small = Tensor::empty({1}, Device::GPU, dtype);
                EXPECT_THROW(kernel.prepare({1}, {}, {small}), std::invalid_argument);
            }
            auto allocated = kernel({1}, {});
            EXPECT_EQ(allocated[0].numel(), 1u);
            EXPECT_EQ(load(binding(allocated[0].cpu(), {}), 0), dtype == DataType::Float16 ? bits(1.f) : 1u);
        }
    }

    TEST_P(ExpressionTest, AliasedOutputsReadOriginalInputs) {
        GpuBackendScope scope(backend());
        for (const bool reverse : {false, true}) {
            fused::Builder b(1);
            auto input = b.input(DataType::Float32, 1);
            if (reverse) {
                b.output(input.load(), DataType::Float32);
                b.output(b.constant(1.f), DataType::Float32);
            } else {
                b.output(b.constant(1.f), DataType::Float32);
                b.output(input.load(), DataType::Float32);
            }
            fused::Kernel kernel(b);
            auto original = floats({5, 6, 7}).to(Device::GPU);
            auto copied = Tensor::zeros({3}, Device::GPU);
            kernel({3}, {original}, reverse ? std::vector<Tensor>{copied, original} : std::vector<Tensor>{original, copied});
            EXPECT_EQ(original.cpu().to_vector(), (std::vector<float>{1, 1, 1}));
            EXPECT_EQ(copied.cpu().to_vector(), (std::vector<float>{5, 6, 7}));
        }
    }

    TEST_P(ExpressionTest, MultipleOutputHomeStreamsObserveWrites) {
        if (backend() != GpuBackend::CUDA)
            GTEST_SKIP() << "CUDA home streams; Vulkan uses storage timelines";
        GpuBackendScope scope(backend());
        struct Streams {
            std::atomic<bool> release = false;
            cudaStream_t a = nullptr, b = nullptr;
            cudaEvent_t done = nullptr;
            float* observed = nullptr;
            ~Streams() {
                release.store(true);
                release.notify_all();
                cudaStreamSynchronize(a);
                cudaStreamSynchronize(b);
                if (done)
                    cudaEventDestroy(done);
                if (observed)
                    cudaFreeHost(observed);
                if (a)
                    cudaStreamDestroy(a);
                if (b)
                    cudaStreamDestroy(b);
            }
        } streams;
        ASSERT_EQ(cudaStreamCreateWithFlags(&streams.a, cudaStreamNonBlocking), cudaSuccess);
        ASSERT_EQ(cudaStreamCreateWithFlags(&streams.b, cudaStreamNonBlocking), cudaSuccess);
        ASSERT_EQ(cudaEventCreateWithFlags(&streams.done, cudaEventDisableTiming), cudaSuccess);
        ASSERT_EQ(cudaMallocHost(&streams.observed, sizeof(float)), cudaSuccess);
        auto a = Tensor::zeros({1}, Device::GPU), b = Tensor::zeros({1}, Device::GPU);
        a.set_stream(streams.a);
        b.set_stream(streams.b);
        fused::Builder builder(1);
        builder.output(builder.constant(1.f), DataType::Float32);
        builder.output(builder.constant(2.f), DataType::Float32);
        fused::Kernel kernel(builder);
        kernel.prepare({1}, {}, {a, b});
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        ASSERT_EQ(cudaLaunchHostFunc(streams.a, hold_stream, &streams.release), cudaSuccess);
        kernel({1}, {}, {a, b});
        ASSERT_EQ(cudaMemcpyAsync(streams.observed, b.data_ptr(), sizeof(float), cudaMemcpyDeviceToHost, streams.b), cudaSuccess);
        ASSERT_EQ(cudaEventRecord(streams.done, streams.b), cudaSuccess);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
        cudaError_t status;
        do {
            status = cudaEventQuery(streams.done);
            std::this_thread::yield();
        } while (status == cudaErrorNotReady && std::chrono::steady_clock::now() < deadline);
        EXPECT_EQ(status, cudaErrorNotReady);
        streams.release.store(true);
        streams.release.notify_all();
        ASSERT_EQ(cudaEventSynchronize(streams.done), cudaSuccess);
        EXPECT_EQ(*streams.observed, 2.f);
        EXPECT_EQ(b.stream(), streams.b);
    }

    TEST_P(ExpressionTest, DeadFoldDoesNotReduceOutputShape) {
        GpuBackendScope scope(backend());
        fused::Builder b(2);
        (void)b.fold(fused::Fold::Sum, b.iota(1), 1);
        b.output(b.iota(0), DataType::Int32);
        fused::Kernel kernel(b);
        auto output = kernel({2, 3}, {})[0];
        EXPECT_EQ(output.size(0), 2u);
        EXPECT_EQ(output.size(1), 3u);
        auto actual = output.cpu();
        for (size_t i = 0; i < 6; ++i)
            EXPECT_EQ(actual.ptr<int32_t>()[i], int32_t(i / 3));
    }

    TEST_P(ExpressionTest, CpuScalarUsesInlineParameters) {
        GpuBackendScope scope(backend());
        fused::Builder b(1);
        b.output(b.input(DataType::Float32, 1).load() + 1.f, DataType::Float32);
        fused::Kernel kernel(b);
        auto scalar = floats({4});
        auto output = Tensor::empty({1}, Device::GPU);
        kernel.prepare({1}, {scalar}, {output});
        backend_ops(backend()).synchronize_device();
        Tensor::trim_device_memory_pool();
        const auto before = backend() == GpuBackend::Vulkan ? vulkan_live_vma_objects_for_testing() : 0;
        for (int i = 0; i < 3; ++i)
            kernel({1}, {scalar}, {output});
        if (backend() == GpuBackend::Vulkan)
            EXPECT_EQ(vulkan_live_vma_objects_for_testing(), before);
        EXPECT_EQ(output.cpu().to_vector(), (std::vector<float>{5}));
        EXPECT_EQ(kernel({3}, {floats({4, 6, 8})})[0].cpu().to_vector(), (std::vector<float>{5, 7, 9}));
        EXPECT_EQ(kernel({1}, {scalar})[0].cpu().to_vector(), (std::vector<float>{5}));
    }

    TEST_P(ExpressionTest, TypedGuardsAndPositionalAlias) {
        GpuBackendScope scope(backend());
        fused::Builder b(2);
        b.output(b.input(DataType::Float32, 2).load() + 1, DataType::Float32);
        fused::Kernel kernel(b);
        auto input = Tensor::zeros({3, 5}, Device::GPU);
        kernel({3, 5}, {input}, {input});
        for (float x : input.cpu().to_vector())
            EXPECT_EQ(x, 1);
        EXPECT_ANY_THROW(kernel({3, 4}, {input}));
        auto expanded = Tensor::zeros({1, 5}, Device::GPU).expand({3, 5});
        EXPECT_ANY_THROW(kernel({3, 5}, {input}, {expanded}));
        auto transposed = Tensor::zeros({5, 3}, Device::GPU).transpose(0, 1);
        kernel({3, 5}, {input}, {transposed});
        for (float x : transposed.contiguous().cpu().to_vector())
            EXPECT_EQ(x, 2);
        fused::Builder gather(1);
        auto table = gather.input(DataType::Float32, 1);
        gather.output(table.gather({gather.iota(0)}), DataType::Float32);
        fused::Kernel read(gather);
        auto flat = input.reshape({15});
        EXPECT_ANY_THROW(read({15}, {flat}, {flat}));
        fused::Builder fold(2);
        fold.output(fold.fold(fused::Fold::Sum, fold.iota(1), 1), DataType::Int32);
        fused::Kernel reduce(fold);
        EXPECT_ANY_THROW(reduce({1, 4097}, {}));
        EXPECT_ANY_THROW(kernel.prepare({size_t(INT32_MAX), 2}, {input}, {input}));
    }

    TEST_P(ExpressionTest, SharedValuesDominateConditionalBranches) {
        GpuBackendScope scope(backend());
        fused::Builder b(1);
        auto x = b.input(DataType::Float32, 1).load();
        auto shared = x * 2;
        b.output(fused::where(x > 0, shared + 1, shared - 1), DataType::Float32);
        b.output(shared + 3, DataType::Float32);
        b.output(fused::where(x > 0, 1, 0), DataType::Int32);
        fused::Kernel kernel(b);
        auto out = kernel({4}, {floats({-2, -1, 1, 2}).to(Device::GPU)});
        EXPECT_EQ(out[0].cpu().to_vector(), (std::vector<float>{-5, -3, 3, 5}));
        EXPECT_EQ(out[1].cpu().to_vector(), (std::vector<float>{-1, 1, 5, 7}));
        const auto flags = out[2].cpu();
        for (int i = 0; i < 4; ++i)
            EXPECT_EQ(flags.ptr<int32_t>()[i], i >= 2);
    }

    TEST_P(ExpressionTest, RuntimeCoordinatesAndGuards) {
        GpuBackendScope scope(backend());
        fused::Builder coordinates(4);
        for (size_t d = 0; d < 4; ++d)
            coordinates.output(coordinates.iota(d), DataType::Int32);
        fused::Kernel kernel(coordinates);
        ExpressionCacheStats warmed;
        for (size_t n : {2, 3, 7, 13, 31}) {
            kernel.prepare({3, 5, n, 11}, {}, {});
            if (n == 2)
                warmed = expression_cache_stats(backend());
            auto output = kernel({3, 5, n, 11}, {});
            for (size_t d = 0; d < 4; ++d) {
                auto actual = output[d].cpu();
                const size_t divisors[]{5 * n * 11, n * 11, 11, 1};
                const size_t extents[]{3, 5, n, 11};
                for (size_t i = 0; i < actual.numel(); ++i)
                    EXPECT_EQ(actual.ptr<int32_t>()[i], (i / divisors[d]) % extents[d]);
            }
            EXPECT_EQ(expression_cache_stats(backend()).loads, warmed.loads);
            EXPECT_EQ(expression_cache_stats(backend()).compilations, warmed.compilations);
        }
        ExpressionBuilder b;
        b.emit(ExprOp::Store, b.emit(ExprOp::Load));
        auto p = b.build();
        auto tensor = Tensor::zeros({16}, Device::GPU);
        std::array<ExprInput, 1> in{binding(tensor, {1, 0, 0, 0})};
        auto out_tensor = Tensor::empty_like(tensor);
        std::array<ExprOutput, 1> out{{{storage_ref(out_tensor), {1, 0, 0, 0}}}};
        prepare_expression(p, in, out, {{16, 1, 1, 1}});
        EXPECT_THROW(run_expression(p, in, out, {{15, 1, 1, 1}}), std::invalid_argument);
        in[0].dims[0] = 2;
        in[0].dims[1] = 2;
        in[0].strides = {INT32_MAX, INT32_MAX, 0, 0};
        EXPECT_THROW(run_expression(p, in, out, {{2, 2, 1, 1}, 2}), std::invalid_argument);
        in[0] = binding(tensor, {4, 1, 0, 0});
        out[0].strides = {1, 1, 0, 0};
        EXPECT_THROW(run_expression(p, in, out, {{4, 4, 1, 1}, 2}), std::invalid_argument);
        fused::Builder params(1);
        params.output(params.input(DataType::Float32, 1).at({0}), DataType::Float32);
        fused::Kernel parameter(params);
        auto host_bank = Tensor::empty_pageable_host({65});
        EXPECT_ANY_THROW(parameter({1}, {host_bank}));
    }

    INSTANTIATE_TEST_SUITE_P(Backends, ExpressionTest, testing::Values(1, 2), [](const auto& p) { return p.param == 1 ? "CUDA" : "Vulkan"; });
} // namespace
