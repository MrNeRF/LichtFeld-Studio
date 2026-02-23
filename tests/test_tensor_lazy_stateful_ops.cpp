/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "core/tensor/internal/lazy_config.hpp"
#include "core/tensor/internal/lazy_executor.hpp"
#include "core/tensor/internal/lazy_ir.hpp"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <optional>
#include <vector>

using namespace lfs::core;

namespace {

    class LazyRuntimeGuard {
    public:
        explicit LazyRuntimeGuard(LazyMode mode) {
            internal::set_lazy_mode_override_for_testing(mode);
            internal::clear_lazy_ir_for_testing();
            internal::lazy_executor_clear_registry_for_testing();
            internal::lazy_executor_reset_diagnostics_for_testing();
            internal::lazy_executor_set_debug_dump_override_for_testing(std::nullopt);
            internal::lazy_executor_clear_debug_dump_cache_for_testing();
            internal::lazy_executor_set_pointwise_fusion_override_for_testing(std::nullopt);
            internal::lazy_executor_set_size_heuristic_override_for_testing(false);
            internal::lazy_executor_set_size_threshold_override_for_testing(std::nullopt);
            Tensor::reset_lazy_telemetry();
        }

        ~LazyRuntimeGuard() {
            internal::set_lazy_mode_override_for_testing(std::nullopt);
            internal::clear_lazy_ir_for_testing();
            internal::lazy_executor_clear_registry_for_testing();
            internal::lazy_executor_reset_diagnostics_for_testing();
            internal::lazy_executor_set_debug_dump_override_for_testing(std::nullopt);
            internal::lazy_executor_clear_debug_dump_cache_for_testing();
            internal::lazy_executor_set_pointwise_fusion_override_for_testing(std::nullopt);
            internal::lazy_executor_set_size_heuristic_override_for_testing(std::nullopt);
            internal::lazy_executor_set_size_threshold_override_for_testing(std::nullopt);
            Tensor::reset_lazy_telemetry();
        }
    };

    bool has_cuda_device() {
        int device_count = 0;
        const auto status = cudaGetDeviceCount(&device_count);
        return status == cudaSuccess && device_count > 0;
    }

} // namespace

TEST(TensorLazyStatefulOpsTest, RandIsEagerInLazyMode) {
    LazyRuntimeGuard guard(LazyMode::On);

    auto t = Tensor::rand({500}, Device::CPU, DataType::Float32);
    EXPECT_FALSE(t.has_lazy_expr());
    EXPECT_TRUE(t.is_valid());
    EXPECT_GT(t.numel(), 0u);

    const auto snapshot = Tensor::lazy_telemetry_snapshot();
    EXPECT_GE(snapshot.stateful_op_eager, 1u);
}

TEST(TensorLazyStatefulOpsTest, RandnIsEagerInLazyMode) {
    LazyRuntimeGuard guard(LazyMode::On);

    auto t = Tensor::randn({500}, Device::CPU, DataType::Float32);
    EXPECT_FALSE(t.has_lazy_expr());
    EXPECT_TRUE(t.is_valid());

    const auto snapshot = Tensor::lazy_telemetry_snapshot();
    EXPECT_GE(snapshot.stateful_op_eager, 1u);
}

TEST(TensorLazyStatefulOpsTest, RandintIsEagerInLazyMode) {
    LazyRuntimeGuard guard(LazyMode::On);

    auto t = Tensor::randint({500}, 0, 100, Device::CPU, DataType::Int32);
    EXPECT_FALSE(t.has_lazy_expr());
    EXPECT_TRUE(t.is_valid());

    const auto snapshot = Tensor::lazy_telemetry_snapshot();
    EXPECT_GE(snapshot.stateful_op_eager, 1u);
}

TEST(TensorLazyStatefulOpsTest, BernoulliIsEagerInLazyMode) {
    LazyRuntimeGuard guard(LazyMode::On);

    auto t = Tensor::bernoulli({500}, 0.5f, Device::CPU, DataType::Float32);
    EXPECT_FALSE(t.has_lazy_expr());
    EXPECT_TRUE(t.is_valid());

    const auto snapshot = Tensor::lazy_telemetry_snapshot();
    EXPECT_GE(snapshot.stateful_op_eager, 1u);
}

TEST(TensorLazyStatefulOpsTest, InplaceNormalIsEagerInLazyMode) {
    LazyRuntimeGuard guard(LazyMode::On);

    auto t = Tensor::empty({500}, Device::CPU, DataType::Float32);
    t.normal_(0.0f, 1.0f);
    EXPECT_FALSE(t.has_lazy_expr());
    EXPECT_TRUE(t.is_valid());
}

TEST(TensorLazyStatefulOpsTest, InplaceUniformIsEagerInLazyMode) {
    LazyRuntimeGuard guard(LazyMode::On);

    auto t = Tensor::empty({500}, Device::CPU, DataType::Float32);
    t.uniform_(0.0f, 1.0f);
    EXPECT_FALSE(t.has_lazy_expr());
    EXPECT_TRUE(t.is_valid());
}

TEST(TensorLazyStatefulOpsTest, ManualSeedReproducibilityWithLazyChain) {
    LazyRuntimeGuard guard(LazyMode::On);
    // Disable size heuristic so the lazy chain defers (500 floats > 4KB? No, 2000 bytes < 4096).
    // Actually 500 * 4 = 2000 bytes < 4096, so size heuristic will make it eager anyway.
    // Use larger tensor or disable heuristic.
    internal::lazy_executor_set_size_heuristic_override_for_testing(false);

    std::vector<float> run1;
    {
        Tensor::manual_seed(42);
        auto r = Tensor::rand({500}, Device::CPU, DataType::Float32);
        auto chain = r.add(1.0f);
        run1 = chain.to_vector();
    }

    Tensor::reset_lazy_telemetry();

    std::vector<float> run2;
    {
        Tensor::manual_seed(42);
        auto r = Tensor::rand({500}, Device::CPU, DataType::Float32);
        auto chain = r.add(1.0f);
        run2 = chain.to_vector();
    }

    ASSERT_EQ(run1.size(), run2.size());
    for (size_t i = 0; i < run1.size(); ++i) {
        EXPECT_FLOAT_EQ(run1[i], run2[i]) << "Mismatch at index " << i;
    }
}

TEST(TensorLazyStatefulOpsTest, InterleavedRandomOpsReproducible) {
    LazyRuntimeGuard guard(LazyMode::On);
    internal::lazy_executor_set_size_heuristic_override_for_testing(false);

    auto run = []() {
        Tensor::manual_seed(123);
        auto a = Tensor::rand({200}, Device::CPU, DataType::Float32).add(1.0f);
        auto b = Tensor::randn({200}, Device::CPU, DataType::Float32).mul(2.0f);
        auto c = a.add(b);
        return c.to_vector();
    };

    auto result1 = run();
    Tensor::reset_lazy_telemetry();
    auto result2 = run();

    ASSERT_EQ(result1.size(), result2.size());
    for (size_t i = 0; i < result1.size(); ++i) {
        EXPECT_FLOAT_EQ(result1[i], result2[i]) << "Mismatch at index " << i;
    }
}

TEST(TensorLazyStatefulOpsTest, GpuRandIsEagerInLazyMode) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required";
    }

    LazyRuntimeGuard guard(LazyMode::On);

    auto t = Tensor::rand({1000}, Device::CUDA, DataType::Float32);
    EXPECT_FALSE(t.has_lazy_expr());
    EXPECT_TRUE(t.is_valid());

    auto cpu = t.to(Device::CPU).to_vector();
    ASSERT_EQ(cpu.size(), 1000u);

    bool has_variation = false;
    for (size_t i = 1; i < cpu.size(); ++i) {
        if (cpu[i] != cpu[0]) {
            has_variation = true;
            break;
        }
    }
    EXPECT_TRUE(has_variation);

    const auto snapshot = Tensor::lazy_telemetry_snapshot();
    EXPECT_GE(snapshot.stateful_op_eager, 1u);
}

TEST(TensorLazyStatefulOpsTest, TelemetryCountsMultipleStatefulOps) {
    LazyRuntimeGuard guard(LazyMode::On);

    Tensor::rand({100}, Device::CPU, DataType::Float32);
    Tensor::randn({100}, Device::CPU, DataType::Float32);
    Tensor::randint({100}, 0, 10, Device::CPU, DataType::Int32);
    Tensor::bernoulli({100}, 0.5f, Device::CPU, DataType::Float32);

    const auto snapshot = Tensor::lazy_telemetry_snapshot();
    EXPECT_GE(snapshot.stateful_op_eager, 4u);
}

TEST(TensorLazyStatefulOpsTest, ShadowModeRecordsStatefulOps) {
    LazyRuntimeGuard guard(LazyMode::Shadow);

    Tensor::rand({100}, Device::CPU, DataType::Float32);
    Tensor::randn({100}, Device::CPU, DataType::Float32);

    const auto snapshot = Tensor::lazy_telemetry_snapshot();
    EXPECT_GE(snapshot.stateful_op_eager, 2u);
}
