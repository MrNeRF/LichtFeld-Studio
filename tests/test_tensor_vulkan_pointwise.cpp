/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "core/tensor/backend/gpu_backend_ops.hpp"
#include "core/tensor/backend/pointwise_lowering.hpp"
#include "core/tensor/backend/vulkan/vk_context.hpp"
#include "core/tensor/internal/lazy_executor.hpp"
#include "core/tensor_backend.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <string_view>
#include <vector>

namespace {
    using namespace lfs::core;

    Tensor upload_vulkan(const Tensor& cpu) {
        GpuBackendScope scope(GpuBackend::Vulkan);
        return cpu.to(Device::GPU);
    }

    Tensor upload_float(const std::vector<float>& values, const TensorShape& shape) {
        return upload_vulkan(Tensor::from_vector(values, shape, Device::CPU));
    }

    Tensor cpu_bytes(const std::vector<uint8_t>& values, const DataType dtype) {
        std::vector<uint8_t> copy = values;
        return Tensor::from_blob(copy.data(), {copy.size()}, Device::CPU, dtype).clone();
    }

    std::vector<uint8_t> guarded_payload(const std::vector<uint8_t>& payload,
                                         const size_t prefix, const size_t suffix) {
        std::vector<uint8_t> expected(prefix + payload.size() + suffix, 0xA5);
        std::copy(payload.begin(), payload.end(), expected.begin() + prefix);
        return expected;
    }

    std::vector<uint8_t> convert_into_guarded(const Tensor& cpu_input,
                                              const DataType output_dtype,
                                              const size_t prefix,
                                              const size_t suffix) {
        GpuBackendScope scope(GpuBackend::Vulkan);
        const size_t count = cpu_input.numel();
        const size_t total = prefix + count + suffix;
        const Tensor input = cpu_input.to(Device::GPU);
        Tensor dest = cpu_bytes(std::vector<uint8_t>(total, 0xA5), DataType::UInt8)
                          .to(Device::GPU);
        internal::StorageRef out =
            internal::offset_storage_ref(internal::storage_ref(dest), prefix);
        out.dtype = output_dtype;
        internal::backend_ops(GpuBackend::Vulkan)
            .convert_type(internal::storage_ref(input), out, count, {});
        return dest.cpu().to_vector_uint8();
    }

    void expect_close(const Tensor& actual, const Tensor& expected,
                      const float rtol = 2.0e-5f, const float atol = 2.0e-6f) {
        const auto actual_values = actual.cpu().to(DataType::Float32).to_vector();
        const auto expected_values = expected.cpu().to(DataType::Float32).to_vector();
        ASSERT_EQ(actual_values.size(), expected_values.size());
        for (size_t i = 0; i < actual_values.size(); ++i) {
            if (std::isnan(expected_values[i])) {
                EXPECT_TRUE(std::isnan(actual_values[i])) << "index=" << i;
            } else if (std::isinf(expected_values[i])) {
                EXPECT_EQ(actual_values[i], expected_values[i]) << "index=" << i;
            } else {
                EXPECT_NEAR(actual_values[i], expected_values[i],
                            atol + rtol * std::abs(expected_values[i]))
                    << "index=" << i;
            }
        }
    }

    class TensorVulkanPointwise : public testing::Test {
    protected:
        void SetUp() override {
            ASSERT_TRUE(gpu_backend_available(GpuBackend::Vulkan));
        }

        void TearDown() override {
            const auto status = shutdown_gpu_backend(GpuBackend::Vulkan);
            EXPECT_TRUE(status.has_value());
            for (const std::string& message :
                 internal::vulkan_validation_messages_for_testing()) {
                ADD_FAILURE() << message;
            }
        }
    };

    TEST_F(TensorVulkanPointwise, UnaryBoundarySizesCatchTailAndGridTruncation) {
        constexpr std::array sizes{size_t{1}, size_t{7}, size_t{4099}, size_t{1048581}};
        for (const size_t count : sizes) {
            std::vector<float> values(count);
            for (size_t i = 0; i < count; ++i) {
                values[i] = static_cast<float>(static_cast<int>(i % 31) - 15) / 16.0f;
            }
            const Tensor cpu = Tensor::from_vector(values, {count}, Device::CPU);
            const Tensor vulkan = upload_vulkan(cpu);
            Tensor result = vulkan.exp();
            expect_close(result, cpu.exp());
            static_cast<void>(result.data_ptr());
            EXPECT_EQ(gpu_backend_of(result), GpuBackend::Vulkan);
        }
    }

    TEST_F(TensorVulkanPointwise, EveryUnaryProgramCatchesMissingOpSpecializations) {
        struct Case {
            std::string_view name;
            std::function<Tensor(const Tensor&)> apply;
        };
        const std::array cases{
            Case{"abs", [](const Tensor& x) { return x.abs(); }},
            Case{"neg", [](const Tensor& x) { return x.neg(); }},
            Case{"exp", [](const Tensor& x) { return x.exp(); }},
            Case{"log", [](const Tensor& x) { return x.log(); }},
            Case{"sqrt", [](const Tensor& x) { return x.sqrt(); }},
            Case{"sigmoid", [](const Tensor& x) { return x.sigmoid(); }},
            Case{"relu", [](const Tensor& x) { return x.relu(); }},
            Case{"square", [](const Tensor& x) { return x.square(); }},
            Case{"tanh", [](const Tensor& x) { return x.tanh(); }},
            Case{"rsqrt", [](const Tensor& x) { return x.rsqrt(); }},
            Case{"sign", [](const Tensor& x) { return x.sign(); }},
            Case{"reciprocal", [](const Tensor& x) { return x.reciprocal(); }},
            Case{"floor", [](const Tensor& x) { return x.floor(); }},
            Case{"ceil", [](const Tensor& x) { return x.ceil(); }},
            Case{"round", [](const Tensor& x) { return x.round(); }},
            Case{"exp2", [](const Tensor& x) { return x.exp2(); }},
            Case{"log2", [](const Tensor& x) { return x.log2(); }},
            Case{"log10", [](const Tensor& x) { return x.log10(); }},
            Case{"log1p", [](const Tensor& x) { return x.log1p(); }},
            Case{"sin", [](const Tensor& x) { return x.sin(); }},
            Case{"cos", [](const Tensor& x) { return x.cos(); }},
            Case{"tan", [](const Tensor& x) { return x.tan(); }},
            Case{"asin", [](const Tensor& x) { return x.asin(); }},
            Case{"acos", [](const Tensor& x) { return x.acos(); }},
            Case{"atan", [](const Tensor& x) { return x.atan(); }},
            Case{"sinh", [](const Tensor& x) { return x.sinh(); }},
            Case{"cosh", [](const Tensor& x) { return x.cosh(); }},
            Case{"gelu", [](const Tensor& x) { return x.gelu(); }},
            Case{"swish", [](const Tensor& x) { return x.swish(); }},
            Case{"trunc", [](const Tensor& x) { return x.trunc(); }},
        };
        std::vector<float> values(4099);
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = 0.0625f + static_cast<float>(i % 29) / 32.0f;
        }
        const Tensor cpu = Tensor::from_vector(values, {values.size()}, Device::CPU);
        const Tensor vulkan = upload_vulkan(cpu);
        for (const Case& test : cases) {
            SCOPED_TRACE(test.name);
            expect_close(test.apply(vulkan), test.apply(cpu));
        }
    }

    TEST_F(TensorVulkanPointwise, IeeePoliciesCatchNaNZeroDenormalAndHalfEvenErrors) {
        const float nan_a = std::bit_cast<float>(0x7fc01234u);
        const float nan_b = std::bit_cast<float>(0x7fc05678u);
        const float denormal = std::numeric_limits<float>::denorm_min();
        const std::vector<float> lhs_values{
            nan_a, 1.0f, 0.0f, -0.0f, denormal, -denormal,
            0.5f, 1.5f, 2.5f, -0.5f, -1.5f, -2.5f,
            std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity()};
        const std::vector<float> rhs_values{
            3.0f, nan_b, -0.0f, 0.0f, -denormal, denormal,
            -7.0f, 7.0f, -7.0f, 7.0f, -7.0f, 7.0f, 0.0f, 0.0f};
        const Tensor lhs = upload_float(lhs_values, {lhs_values.size()});
        const Tensor rhs = upload_float(rhs_values, {rhs_values.size()});
        const auto maximum = lhs.maximum(rhs).to_vector();
        const auto minimum = lhs.minimum(rhs).to_vector();
        const auto rounded = lhs.round().to_vector();

        EXPECT_EQ(std::bit_cast<uint32_t>(maximum[0]), std::bit_cast<uint32_t>(nan_a));
        EXPECT_EQ(std::bit_cast<uint32_t>(maximum[1]), std::bit_cast<uint32_t>(nan_b));
        EXPECT_EQ(std::bit_cast<uint32_t>(maximum[2]), 0u);
        EXPECT_EQ(std::bit_cast<uint32_t>(maximum[3]), 0u);
        EXPECT_EQ(std::bit_cast<uint32_t>(minimum[2]), 0x80000000u);
        EXPECT_EQ(std::bit_cast<uint32_t>(minimum[3]), 0x80000000u);
        EXPECT_EQ(rounded[6], 0.0f);
        EXPECT_EQ(rounded[7], 2.0f);
        EXPECT_EQ(rounded[8], 2.0f);
        EXPECT_EQ(std::bit_cast<uint32_t>(rounded[9]), 0x80000000u);
        EXPECT_EQ(rounded[10], -2.0f);
        EXPECT_EQ(rounded[11], -2.0f);
        EXPECT_TRUE(lhs.isnan().to_vector_bool()[0]);
        EXPECT_TRUE(lhs.isinf().to_vector_bool()[12]);
        EXPECT_FALSE(lhs.isfinite().to_vector_bool()[13]);
    }

    TEST_F(TensorVulkanPointwise, BinaryScalarBroadcastAndLogicalCatchDtypeDispatch) {
        const Tensor cpu_a = Tensor::from_vector(
            std::vector<float>{-2.0f, -0.5f, 0.0f, 0.25f, 1.0f, 2.0f, 4.0f},
            {7, 1}, Device::CPU);
        const Tensor cpu_b = Tensor::from_vector(
            std::vector<float>{0.5f, 1.0f, 2.0f}, {1, 3}, Device::CPU);
        const Tensor a = upload_vulkan(cpu_a);
        const Tensor b = upload_vulkan(cpu_b);
        expect_close(a.add(b), cpu_a.add(cpu_b));
        expect_close(a.sub(b), cpu_a.sub(cpu_b));
        expect_close(a.mul(b), cpu_a.mul(cpu_b));
        expect_close(a.div(b), cpu_a.div(cpu_b));
        expect_close(a.maximum(b), cpu_a.maximum(cpu_b));
        expect_close(a.minimum(b), cpu_a.minimum(cpu_b));
        EXPECT_EQ(a.gt(b).to_vector_bool(), cpu_a.gt(cpu_b).to_vector_bool());

        const Tensor flat = a.flatten();
        expect_close(flat.add(0.25f), cpu_a.flatten().add(0.25f));
        expect_close(flat.sub(0.25f), cpu_a.flatten().sub(0.25f));
        expect_close(flat.mul(1.5f), cpu_a.flatten().mul(1.5f));
        expect_close(flat.div(2.0f), cpu_a.flatten().div(2.0f));
        expect_close(flat.pow(2.0f), cpu_a.flatten().pow(2.0f));
        EXPECT_EQ(flat.ge(0.0f).to_vector_bool(),
                  cpu_a.flatten().ge(0.0f).to_vector_bool());

        const Tensor bool_a = a.gt(0.0f);
        const Tensor bool_b = a.lt(2.0f);
        EXPECT_EQ(bool_a.logical_and(bool_b).to_vector_bool(),
                  std::vector<bool>({false, false, false, true, true, false, false}));
        EXPECT_EQ(bool_a.logical_or(bool_b).to_vector_bool(),
                  std::vector<bool>({true, true, true, true, true, true, true}));
        EXPECT_EQ(bool_a.logical_xor(bool_b).to_vector_bool(),
                  std::vector<bool>({true, true, true, false, false, true, true}));
    }

    TEST_F(TensorVulkanPointwise, ConversionCatchesTorchUInt8AndNonzeroBoolRules) {
        const std::vector<float> values{
            0.0f, 1.9f, -1.9f, 255.9f, 256.0f, -256.0f, 257.0f,
            std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::quiet_NaN()};
        const Tensor input = upload_float(values, {values.size()});
        EXPECT_EQ(input.to(DataType::UInt8).to_vector_uint8(),
                  std::vector<uint8_t>({0, 1, 255, 255, 0, 0, 1, 0, 0, 0}));

        const Tensor bytes = upload_vulkan(
            Tensor::from_vector(std::vector<int>{0, 1, 2, 127, 255},
                                {5}, Device::CPU)
                .to(DataType::UInt8));
        EXPECT_EQ(bytes.to(DataType::Bool).to_vector_bool(),
                  std::vector<bool>({false, true, true, true, true}));

        constexpr std::array dtypes{
            DataType::Float32, DataType::Float16, DataType::Int32,
            DataType::Int64, DataType::UInt8, DataType::Bool, DataType::UInt32};
        for (const DataType dtype : dtypes) {
            Tensor converted = input;
            if (dtype == DataType::Bool) {
                converted = bytes.to(dtype);
            } else {
                converted = input.to(dtype);
            }
            EXPECT_EQ(gpu_backend_of(converted), GpuBackend::Vulkan);
            EXPECT_EQ(converted.cpu().bytes(), converted.bytes());
        }
    }

    TEST_F(TensorVulkanPointwise, ConversionCatchesEveryCudaInstantiatedDtypePair) {
        struct Pair {
            DataType input;
            DataType output;
        };
        constexpr std::array pairs{
            Pair{DataType::Float32, DataType::Float16},
            Pair{DataType::Float32, DataType::Int32},
            Pair{DataType::Float32, DataType::Int64},
            Pair{DataType::Float32, DataType::UInt8},
            Pair{DataType::Float32, DataType::UInt32},
            Pair{DataType::Float16, DataType::Float32},
            Pair{DataType::Float16, DataType::Int32},
            Pair{DataType::Float16, DataType::Int64},
            Pair{DataType::Float16, DataType::UInt8},
            Pair{DataType::Int32, DataType::Float32},
            Pair{DataType::Int32, DataType::Float16},
            Pair{DataType::Int32, DataType::Int64},
            Pair{DataType::Int32, DataType::UInt8},
            Pair{DataType::Int64, DataType::Float32},
            Pair{DataType::Int64, DataType::Float16},
            Pair{DataType::Int64, DataType::Int32},
            Pair{DataType::Int64, DataType::UInt8},
            Pair{DataType::UInt8, DataType::Float32},
            Pair{DataType::UInt8, DataType::Float16},
            Pair{DataType::UInt8, DataType::Int32},
            Pair{DataType::UInt8, DataType::Int64},
            Pair{DataType::UInt8, DataType::Bool},
            Pair{DataType::Bool, DataType::Float32},
            Pair{DataType::Bool, DataType::Float16},
            Pair{DataType::Bool, DataType::Int32},
            Pair{DataType::Bool, DataType::Int64},
            Pair{DataType::Bool, DataType::Bool},
            Pair{DataType::UInt32, DataType::Float32},
            Pair{DataType::UInt32, DataType::Int64},
        };
        const Tensor base = Tensor::from_vector(
            std::vector<float>{0.0f, 1.0f, 2.0f, 3.0f, 127.0f},
            {5}, Device::CPU);
        for (const Pair pair : pairs) {
            SCOPED_TRACE(static_cast<int>(pair.input));
            SCOPED_TRACE(static_cast<int>(pair.output));
            const Tensor cpu_input = base.to(pair.input);
            if (pair.input == DataType::UInt32 &&
                pair.output == DataType::Int64) {
                GpuBackendScope scope(GpuBackend::Vulkan);
                const Tensor input = cpu_input.to(Device::GPU);
                Tensor actual = Tensor::empty(input.shape(), Device::GPU,
                                              pair.output);
                internal::backend_ops(GpuBackend::Vulkan)
                    .convert_type(internal::storage_ref(input),
                                  internal::storage_ref(actual), actual.numel(), {});
                EXPECT_EQ(actual.cpu().to_vector_int64(),
                          std::vector<int64_t>({0, 1, 2, 3, 127}));
                continue;
            }
            const Tensor actual = upload_vulkan(cpu_input).to(pair.output).cpu();
            const Tensor expected = cpu_input.to(pair.output);
            ASSERT_EQ(actual.bytes(), expected.bytes());
            EXPECT_EQ(std::memcmp(actual.data_ptr(), expected.data_ptr(), actual.bytes()), 0);
        }
    }

    TEST_F(TensorVulkanPointwise, UnalignedFacadeInputCatchesWrongVectorFastPath) {
        GpuBackendScope scope(GpuBackend::Vulkan);
        const Tensor base = Tensor::from_vector(
            std::vector<float>{99.0f, -1.0f, 2.0f, -3.0f, 4.0f,
                               -5.0f, 6.0f, -7.0f, 8.0f},
            {9}, Device::GPU);
        const Tensor input = base.slice(0, 1, 9);
        Tensor output = Tensor::empty({8}, Device::GPU);
        const auto program = internal::pointwise_program(
            DataType::Float32, DataType::Float32, ops::abs_op{});
        internal::backend_ops(GpuBackend::Vulkan).unary(program, internal::storage_ref(input), internal::storage_ref(output), output.numel(), {});
        EXPECT_EQ(output.to_vector(),
                  std::vector<float>({1, 2, 3, 4, 5, 6, 7, 8}));
    }

    TEST_F(TensorVulkanPointwise, ClampEntriesCatchInPlaceFusedAndIntegerBounds) {
        const Tensor cpu = Tensor::from_vector(
            std::vector<float>{-4.0f, -1.0f, -0.0f, 0.5f, 2.0f,
                               std::numeric_limits<float>::quiet_NaN()},
            {6}, Device::CPU);
        const Tensor input = upload_vulkan(cpu);
        expect_close(input.clamp(-1.0f, 1.0f), cpu.clamp(-1.0f, 1.0f));

        Tensor in_place = input.clone();
        in_place.clamp_(-1.0f, 1.0f);
        expect_close(in_place, cpu.clamp(-1.0f, 1.0f));

        const Tensor integers = upload_vulkan(
            Tensor::from_vector(std::vector<int>{-7, -2, 0, 3, 9},
                                {5}, Device::CPU));
        Tensor integer_in_place = integers.clone();
        integer_in_place.clamp_(-2, 3);
        EXPECT_EQ(integer_in_place.cpu().to_vector_int(),
                  std::vector<int>({-2, -2, 0, 3, 3}));
    }

    TEST_F(TensorVulkanPointwise, EagerExpressionCatchesAllSixteenChainDescriptors) {
        constexpr size_t count = 4099;
        std::vector<float> values(count);
        for (size_t i = 0; i < count; ++i)
            values[i] = 0.25f + static_cast<float>(i % 37) / 64.0f;
        const Tensor cpu = Tensor::from_vector(values, {count}, Device::CPU);
        const Tensor input = upload_vulkan(cpu);
        const Tensor identity = Tensor::zeros_like(input);

        internal::lazy_executor_reset_diagnostics_for_testing();
        Tensor actual = input.add(identity).add(0.25f).mul(1.5f).sub(0.5f).abs().neg().neg().square().sqrt().reciprocal().reciprocal().sigmoid().tanh().exp().log().relu();
        static_cast<void>(actual.data_ptr());
        Tensor expected = cpu.add(Tensor::zeros_like(cpu)).add(0.25f).mul(1.5f).sub(0.5f).abs().neg().neg().square().sqrt().reciprocal().reciprocal().sigmoid().tanh().exp().log().relu();
        expect_close(actual, expected);
        EXPECT_EQ(gpu_backend_of(actual), GpuBackend::Vulkan);
        EXPECT_GT(internal::lazy_executor_diagnostics_snapshot_for_testing()
                      .fused_launches,
                  0u);
    }

    TEST_F(TensorVulkanPointwise, MovementEntriesCatchStridesCatPadAndFill) {
        GpuBackendScope scope(GpuBackend::Vulkan);
        const Tensor base = Tensor::from_vector(
            std::vector<float>{1, 2, 3, 4, 5, 6}, {2, 3}, Device::GPU);
        EXPECT_EQ(base.transpose(0, 1).contiguous().to_vector(),
                  std::vector<float>({1, 4, 2, 5, 3, 6}));

        Tensor destination = Tensor::zeros({2, 3}, Device::GPU);
        destination.transpose(0, 1).copy_from(
            Tensor::from_vector(std::vector<float>{7, 8, 9, 10, 11, 12},
                                {3, 2}, Device::GPU));
        EXPECT_EQ(destination.to_vector(),
                  std::vector<float>({7, 9, 11, 8, 10, 12}));

        const Tensor rank5 = Tensor::from_vector(
            std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8},
            {1, 1, 1, 4, 2}, Device::GPU);
        EXPECT_EQ(rank5.transpose(3, 4).contiguous().to_vector(),
                  std::vector<float>({1, 3, 5, 7, 2, 4, 6, 8}));
        Tensor rank5_destination = Tensor::zeros({1, 1, 1, 4, 2}, Device::GPU);
        rank5_destination.transpose(3, 4).copy_from(
            Tensor::from_vector(std::vector<float>{8, 7, 6, 5, 4, 3, 2, 1},
                                {1, 1, 1, 2, 4}, Device::GPU));
        EXPECT_EQ(rank5_destination.to_vector(),
                  std::vector<float>({8, 4, 7, 3, 6, 2, 5, 1}));

        const Tensor host_view =
            Tensor::from_vector(std::vector<float>{1, 2, 3, 4, 5, 6},
                                {2, 3}, Device::CPU)
                .transpose(0, 1);
        EXPECT_EQ(host_view.to(Device::GPU).to_vector(), host_view.to_vector());

        Tensor float_destination = Tensor::zeros({2, 3}, Device::GPU);
        float_destination.transpose(0, 1).copy_from(
            Tensor::from_vector(std::vector<int>{1, 2, 3, 4, 5, 6},
                                {3, 2}, Device::GPU));
        EXPECT_EQ(float_destination.to_vector(),
                  std::vector<float>({1, 3, 5, 2, 4, 6}));

        Tensor filled = Tensor::zeros({3, 2}, Device::GPU);
        filled.transpose(0, 1).fill_(3.5f);
        EXPECT_EQ(filled.to_vector(), std::vector<float>(6, 3.5f));
        EXPECT_EQ(Tensor::full({7}, -2.25f, Device::GPU).to_vector(),
                  std::vector<float>(7, -2.25f));

        EXPECT_EQ(Tensor::cat({base, base}, 1).to_vector(),
                  std::vector<float>({1, 2, 3, 1, 2, 3,
                                      4, 5, 6, 4, 5, 6}));
        const Tensor rank3 = base.unsqueeze(0);
        EXPECT_EQ(Tensor::cat({rank3, rank3}, 1).to_vector(),
                  std::vector<float>({1, 2, 3, 4, 5, 6,
                                      1, 2, 3, 4, 5, 6}));

        MovementArgs pad_args;
        pad_args.args = std::vector<std::pair<int, int>>{{1, 1}, {2, 1}};
        const Tensor padded = base.movement(MovementOp::Pad, pad_args);
        EXPECT_EQ(padded.shape(), TensorShape({4, 6}));
        EXPECT_EQ(padded.slice(0, 1, 3).slice(1, 2, 5).to_vector(),
                  base.to_vector());
    }

    TEST_F(TensorVulkanPointwise, ConversionPackedByteOutputPreservesGuardsAtEveryAlignment) {
        std::vector<float> values(16);
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = static_cast<float>(static_cast<int>(i) * 17 - 40);
        }
        const Tensor cpu = Tensor::from_vector(values, {values.size()}, Device::CPU);
        constexpr std::array counts{size_t{1}, size_t{2}, size_t{3}, size_t{4},
                                    size_t{5}, size_t{7}, size_t{8}, size_t{15}};
        constexpr size_t kGuard = 8;
        for (const size_t count : counts) {
            const Tensor cpu_slice = cpu.slice(0, 0, count).contiguous();
            const std::vector<uint8_t> payload =
                cpu_slice.to(DataType::UInt8).to_vector_uint8();
            for (size_t align = 0; align < 4; ++align) {
                SCOPED_TRACE(count);
                SCOPED_TRACE(align);
                const size_t prefix = kGuard + align;
                const std::vector<uint8_t> actual =
                    convert_into_guarded(cpu_slice, DataType::UInt8, prefix, kGuard);
                EXPECT_EQ(actual, guarded_payload(payload, prefix, kGuard));
            }
        }

        std::vector<float> large(4099);
        for (size_t i = 0; i < large.size(); ++i) {
            large[i] = static_cast<float>(static_cast<int>(i % 511) - 200);
        }
        const Tensor cpu_large = Tensor::from_vector(large, {large.size()}, Device::CPU);
        const std::vector<uint8_t> large_payload =
            cpu_large.to(DataType::UInt8).to_vector_uint8();
        for (size_t align = 0; align < 4; ++align) {
            SCOPED_TRACE(align);
            const size_t prefix = kGuard + align;
            EXPECT_EQ(convert_into_guarded(cpu_large, DataType::UInt8, prefix, kGuard),
                      guarded_payload(large_payload, prefix, kGuard));
        }
    }

    TEST_F(TensorVulkanPointwise, ConversionPackedByteOutputMatchesDtypeReferenceAndUnalignedInput) {
        const std::vector<float> float_values{
            0.0f, 1.9f, -1.9f, 255.9f, 256.0f, -256.0f, 257.0f, -257.0f,
            511.0f, -0.1f, 127.0f, 128.0f,
            std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::quiet_NaN(),
            std::numeric_limits<float>::denorm_min()};
        const Tensor cpu_f32 =
            Tensor::from_vector(float_values, {float_values.size()}, Device::CPU);
        const std::vector<uint8_t> f32_payload =
            cpu_f32.to(DataType::UInt8).to_vector_uint8();
        EXPECT_EQ(upload_vulkan(cpu_f32).to(DataType::UInt8).to_vector_uint8(), f32_payload);
        EXPECT_EQ(convert_into_guarded(cpu_f32, DataType::UInt8, 9, 8),
                  guarded_payload(f32_payload, 9, 8));

        const Tensor cpu_f16 = cpu_f32.to(DataType::Float16);
        const std::vector<uint8_t> f16_payload =
            cpu_f16.to(DataType::UInt8).to_vector_uint8();
        EXPECT_EQ(upload_vulkan(cpu_f16).to(DataType::UInt8).to_vector_uint8(), f16_payload);
        EXPECT_EQ(convert_into_guarded(cpu_f16, DataType::UInt8, 1, 8),
                  guarded_payload(f16_payload, 1, 8));

        const Tensor cpu_i32 = Tensor::from_vector(
            std::vector<int>{-1, 0, 1, 255, 256, 257, -256, 511, -257},
            {9}, Device::CPU);
        const std::vector<uint8_t> i32_payload =
            cpu_i32.to(DataType::UInt8).to_vector_uint8();
        EXPECT_EQ(i32_payload, (std::vector<uint8_t>{255, 0, 1, 255, 0, 1, 0, 255, 255}));
        EXPECT_EQ(upload_vulkan(cpu_i32).to(DataType::UInt8).to_vector_uint8(), i32_payload);
        EXPECT_EQ(convert_into_guarded(cpu_i32, DataType::UInt8, 2, 8),
                  guarded_payload(i32_payload, 2, 8));

        const std::vector<int64_t> i64_values{
            0, -1, 256, 511, std::numeric_limits<int64_t>::min(),
            std::numeric_limits<int64_t>::max(), 0x100000001LL};
        const Tensor cpu_i64 =
            Tensor::from_blob(const_cast<int64_t*>(i64_values.data()),
                              {i64_values.size()}, Device::CPU, DataType::Int64)
                .clone();
        const std::vector<uint8_t> i64_payload =
            cpu_i64.to(DataType::UInt8).to_vector_uint8();
        EXPECT_EQ(upload_vulkan(cpu_i64).to(DataType::UInt8).to_vector_uint8(), i64_payload);
        EXPECT_EQ(convert_into_guarded(cpu_i64, DataType::UInt8, 3, 8),
                  guarded_payload(i64_payload, 3, 8));

        const std::vector<uint8_t> raw{0, 1, 2, 127, 255, 0, 3, 8};
        const Tensor cpu_u8 = cpu_bytes(raw, DataType::UInt8);
        const std::vector<uint8_t> bool_payload =
            cpu_u8.to(DataType::Bool).to_vector_uint8();
        EXPECT_EQ(bool_payload, (std::vector<uint8_t>{0, 1, 1, 1, 1, 0, 1, 1}));
        EXPECT_EQ(upload_vulkan(cpu_u8).to(DataType::Bool).to_vector_uint8(), bool_payload);
        EXPECT_EQ(convert_into_guarded(cpu_u8, DataType::Bool, 1, 8),
                  guarded_payload(bool_payload, 1, 8));

        GpuBackendScope scope(GpuBackend::Vulkan);
        for (size_t input_align = 0; input_align < 4; ++input_align) {
            SCOPED_TRACE(input_align);
            std::vector<uint8_t> padded(4 + raw.size(), 0x3C);
            std::copy(raw.begin(), raw.end(), padded.begin() + input_align);
            Tensor input = cpu_bytes(padded, DataType::UInt8).to(Device::GPU);
            const size_t out_prefix = 8 + input_align;
            constexpr size_t suffix = 8;
            Tensor dest =
                cpu_bytes(std::vector<uint8_t>(out_prefix + raw.size() + suffix, 0xA5),
                          DataType::UInt8)
                    .to(Device::GPU);
            internal::StorageRef in = internal::offset_storage_ref(
                internal::storage_ref(input), input_align);
            internal::StorageRef out = internal::offset_storage_ref(
                internal::storage_ref(dest), out_prefix);
            out.dtype = DataType::Bool;
            internal::backend_ops(GpuBackend::Vulkan)
                .convert_type(in, out, raw.size(), {});
            EXPECT_EQ(dest.cpu().to_vector_uint8(),
                      guarded_payload(bool_payload, out_prefix, suffix));
        }
    }

    TEST_F(TensorVulkanPointwise, ConversionPackedByteOutputInPlaceNormalizesWithoutClobber) {
        GpuBackendScope scope(GpuBackend::Vulkan);
        const std::vector<uint8_t> host{0xA5, 0xA5, 0, 2, 255, 1, 0, 7, 0xA5, 0xA5};
        Tensor tensor = cpu_bytes(host, DataType::UInt8).to(Device::GPU);
        internal::StorageRef region =
            internal::offset_storage_ref(internal::storage_ref(tensor), 2);
        internal::StorageRef as_bool = region;
        as_bool.dtype = DataType::Bool;
        internal::backend_ops(GpuBackend::Vulkan)
            .convert_type(region, as_bool, 6, {});
        EXPECT_EQ(tensor.cpu().to_vector_uint8(),
                  (std::vector<uint8_t>{0xA5, 0xA5, 0, 1, 1, 1, 0, 1, 0xA5, 0xA5}));

        Tensor bool_tensor = cpu_bytes(host, DataType::Bool).to(Device::GPU);
        internal::StorageRef bool_region =
            internal::offset_storage_ref(internal::storage_ref(bool_tensor), 2);
        internal::backend_ops(GpuBackend::Vulkan)
            .convert_type(bool_region, bool_region, 6, {});
        EXPECT_EQ(bool_tensor.cpu().to_vector_uint8(),
                  (std::vector<uint8_t>{0xA5, 0xA5, 0, 1, 1, 1, 0, 1, 0xA5, 0xA5}));
    }
} // namespace
