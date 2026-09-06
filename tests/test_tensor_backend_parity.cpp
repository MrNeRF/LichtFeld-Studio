/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
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
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace {
    using namespace lfs::core;

    constexpr uint64_t kSeed = 0x4c46535f50344733ULL;

    std::vector<float> seeded_values(const size_t count) {
        std::mt19937_64 generator(kSeed + count);
        std::uniform_real_distribution<float> distribution(-1.5f, 1.5f);
        std::vector<float> values(count);
        std::generate(values.begin(), values.end(), [&] { return distribution(generator); });
        return values;
    }

    Tensor upload(const std::vector<float>& values, const TensorShape& shape,
                  const GpuBackend backend) {
        const Tensor cpu = Tensor::from_vector(values, shape, Device::CPU);
        GpuBackendScope scope(backend);
        return cpu.to(Device::GPU);
    }

    uint32_t ordered_bits(const float value) {
        const uint32_t bits = std::bit_cast<uint32_t>(value);
        return (bits & 0x80000000u) != 0u ? ~bits : bits | 0x80000000u;
    }

    void expect_ulp(const Tensor& actual, const Tensor& expected,
                    const uint32_t maximum_ulp = 4u) {
        const auto a = actual.to_vector();
        const auto b = expected.to_vector();
        ASSERT_EQ(a.size(), b.size());
        for (size_t i = 0; i < a.size(); ++i) {
            if (std::isnan(a[i]) || std::isnan(b[i])) {
                EXPECT_TRUE(std::isnan(a[i]) && std::isnan(b[i])) << "index=" << i;
                continue;
            }
            const uint32_t lhs = ordered_bits(a[i]);
            const uint32_t rhs = ordered_bits(b[i]);
            const uint32_t distance = lhs > rhs ? lhs - rhs : rhs - lhs;
            EXPECT_LE(distance, maximum_ulp)
                << "index=" << i << " cuda=" << b[i]
                << " vulkan=" << a[i];
        }
    }

    void expect_absolute(const Tensor& actual, const Tensor& expected,
                         const float tolerance) {
        const auto a = actual.to_vector();
        const auto b = expected.to_vector();
        ASSERT_EQ(a.size(), b.size());
        for (size_t i = 0; i < a.size(); ++i) {
            if (std::isnan(a[i]) || std::isnan(b[i])) {
                EXPECT_TRUE(std::isnan(a[i]) && std::isnan(b[i]))
                    << "index=" << i;
                continue;
            }
            EXPECT_LE(std::abs(a[i] - b[i]), tolerance)
                << "index=" << i << " cuda=" << b[i]
                << " vulkan=" << a[i];
        }
    }

    class TensorBackendParity : public testing::Test {
    protected:
        void SetUp() override {
            ASSERT_TRUE(gpu_backend_available(GpuBackend::CUDA));
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

    Tensor run_pointwise_case(const GpuBackend backend, const bool opposite_scope) {
        constexpr size_t count = 4099;
        const auto lhs_values = seeded_values(count);
        auto rhs_values = seeded_values(count + 17);
        rhs_values.resize(count);
        const Tensor lhs = upload(lhs_values, {count}, backend);
        const Tensor rhs = upload(rhs_values, {count}, backend);
        Tensor output;
        if (opposite_scope) {
            GpuBackendScope scope(backend == GpuBackend::CUDA
                                      ? GpuBackend::Vulkan
                                      : GpuBackend::CUDA);
            output = lhs.mul(rhs).add(0.25f).sub(0.5f);
        } else {
            output = lhs.mul(rhs).add(0.25f).sub(0.5f);
        }
        static_cast<void>(output.data_ptr());
        EXPECT_EQ(gpu_backend_of(output), backend);
        return output.cpu();
    }

    TEST_F(TensorBackendParity, PointwiseUsesInputBackendWithoutActiveScope) {
        const Tensor cuda = run_pointwise_case(GpuBackend::CUDA, false);
        internal::lazy_executor_reset_diagnostics_for_testing();
        const Tensor vulkan = run_pointwise_case(GpuBackend::Vulkan, false);
        EXPECT_GT(internal::lazy_executor_diagnostics_snapshot_for_testing()
                      .fused_launches,
                  0u);
        expect_ulp(vulkan, cuda);
    }

    TEST_F(TensorBackendParity, PointwiseIgnoresOppositeActiveScope) {
        const Tensor cuda = run_pointwise_case(GpuBackend::CUDA, true);
        const Tensor vulkan = run_pointwise_case(GpuBackend::Vulkan, true);
        expect_ulp(vulkan, cuda);
    }

    TEST_F(TensorBackendParity, UnaryProgramsUseCudaToleranceClasses) {
        struct Case {
            std::string_view name;
            std::function<Tensor(const Tensor&)> apply;
            uint32_t maximum_ulp = 4;
            float absolute_tolerance = 0.0f;
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
            // Release CUDA lowers tanhf to MUFU.TANH under -use_fast_math.
            // The portable SPIR-V extended instruction is more accurate but
            // cannot reproduce that architecture-specific approximation.
            Case{"tanh", [](const Tensor& x) { return x.tanh(); }, 128},
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
            Case{"sin", [](const Tensor& x) { return x.sin(); }, 4, 0x1p-20f},
            Case{"cos", [](const Tensor& x) { return x.cos(); }, 4, 0x1p-20f},
            Case{"tan", [](const Tensor& x) { return x.tan(); }, 4, 0x1p-20f},
            Case{"asin", [](const Tensor& x) { return x.asin(); }, 4, 0x1p-19f},
            Case{"acos", [](const Tensor& x) { return x.acos(); }, 4, 0x1p-19f},
            Case{"atan", [](const Tensor& x) { return x.atan(); }, 4, 0x1p-19f},
            Case{"sinh", [](const Tensor& x) { return x.sinh(); }},
            Case{"cosh", [](const Tensor& x) { return x.cosh(); }},
            Case{"gelu", [](const Tensor& x) { return x.gelu(); }, 32},
            Case{"swish", [](const Tensor& x) { return x.swish(); }},
            Case{"trunc", [](const Tensor& x) { return x.trunc(); }},
        };
        std::vector<float> values(4099);
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = 0.125f + static_cast<float>(i % 31) / 64.0f;
        const Tensor cuda_input = upload(values, {values.size()}, GpuBackend::CUDA);
        const Tensor vulkan_input = upload(values, {values.size()}, GpuBackend::Vulkan);
        for (const Case& test : cases) {
            SCOPED_TRACE(test.name);
            const Tensor vulkan = test.apply(vulkan_input).cpu();
            const Tensor cuda = test.apply(cuda_input).cpu();
            if (test.absolute_tolerance != 0.0f) {
                expect_absolute(vulkan, cuda, test.absolute_tolerance);
            } else {
                expect_ulp(vulkan, cuda, test.maximum_ulp);
            }
        }
    }

    Tensor run_broadcast_case(const GpuBackend backend, const bool opposite_scope) {
        const auto lhs_values = seeded_values(33 * 127);
        const auto rhs_values = seeded_values(127);
        const Tensor lhs = upload(lhs_values, {33, 127}, backend);
        const Tensor rhs = upload(rhs_values, {1, 127}, backend);
        Tensor output;
        if (opposite_scope) {
            GpuBackendScope scope(backend == GpuBackend::CUDA
                                      ? GpuBackend::Vulkan
                                      : GpuBackend::CUDA);
            output = lhs.maximum(rhs);
        } else {
            output = lhs.maximum(rhs);
        }
        EXPECT_EQ(gpu_backend_of(output), backend);
        return output.cpu();
    }

    TEST_F(TensorBackendParity, BroadcastPreservesBackendAndIeeeValues) {
        for (const bool opposite_scope : {false, true}) {
            const Tensor cuda = run_broadcast_case(GpuBackend::CUDA, opposite_scope);
            const Tensor vulkan = run_broadcast_case(GpuBackend::Vulkan, opposite_scope);
            EXPECT_EQ(vulkan.to_vector(), cuda.to_vector());
        }
    }

    Tensor run_movement_case(const GpuBackend backend, const bool opposite_scope) {
        const auto values = seeded_values(2 * 3 * 17);
        const Tensor input = upload(values, {2, 3, 17}, backend);
        Tensor output;
        if (opposite_scope) {
            GpuBackendScope scope(backend == GpuBackend::CUDA
                                      ? GpuBackend::Vulkan
                                      : GpuBackend::CUDA);
            output = Tensor::cat({input.transpose(1, 2).contiguous(),
                                  input.transpose(1, 2).contiguous()},
                                 2);
        } else {
            output = Tensor::cat({input.transpose(1, 2).contiguous(),
                                  input.transpose(1, 2).contiguous()},
                                 2);
        }
        EXPECT_EQ(gpu_backend_of(output), backend);
        return output.cpu();
    }

    TEST_F(TensorBackendParity, MovementIsBitExactUnderOppositeScope) {
        for (const bool opposite_scope : {false, true}) {
            const Tensor cuda = run_movement_case(GpuBackend::CUDA, opposite_scope);
            const Tensor vulkan = run_movement_case(GpuBackend::Vulkan, opposite_scope);
            ASSERT_EQ(vulkan.bytes(), cuda.bytes());
            EXPECT_EQ(std::memcmp(vulkan.data_ptr(), cuda.data_ptr(), cuda.bytes()), 0);
        }
    }

    TEST_F(TensorBackendParity, Float16PointwiseUsesDocumentedTolerance) {
        constexpr size_t count = 4099;
        const auto values = seeded_values(count);
        auto run = [&](const GpuBackend backend) {
            const Tensor input = upload(values, {count}, backend).to(DataType::Float16);
            const Tensor result = input.add(input);
            EXPECT_EQ(gpu_backend_of(result), backend);
            return result.cpu().to(DataType::Float32).to_vector();
        };
        const auto cuda = run(GpuBackend::CUDA);
        const auto vulkan = run(GpuBackend::Vulkan);
        ASSERT_EQ(vulkan.size(), cuda.size());
        for (size_t i = 0; i < vulkan.size(); ++i) {
            EXPECT_NEAR(vulkan[i], cuda[i], 1.0e-3f) << "index=" << i;
        }
    }
} // namespace
