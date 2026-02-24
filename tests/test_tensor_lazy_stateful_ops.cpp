/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "core/tensor/internal/lazy_ir.hpp"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

using namespace lfs::core;

namespace {

    class LazyTestGuard {
    public:
        LazyTestGuard() {
            internal::clear_lazy_ir_for_testing();
            Tensor::reset_lazy_telemetry();
        }
        ~LazyTestGuard() {
            internal::clear_lazy_ir_for_testing();
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
    LazyTestGuard guard;

    auto t = Tensor::rand({500}, Device::CPU, DataType::Float32);
    EXPECT_FALSE(t.has_lazy_expr());
    EXPECT_TRUE(t.is_valid());
    EXPECT_GT(t.numel(), 0u);
}

TEST(TensorLazyStatefulOpsTest, RandnIsEagerInLazyMode) {
    LazyTestGuard guard;

    auto t = Tensor::randn({500}, Device::CPU, DataType::Float32);
    EXPECT_FALSE(t.has_lazy_expr());
    EXPECT_TRUE(t.is_valid());
}

TEST(TensorLazyStatefulOpsTest, RandintIsEagerInLazyMode) {
    LazyTestGuard guard;

    auto t = Tensor::randint({500}, 0, 100, Device::CPU, DataType::Int32);
    EXPECT_FALSE(t.has_lazy_expr());
    EXPECT_TRUE(t.is_valid());
}

TEST(TensorLazyStatefulOpsTest, BernoulliIsEagerInLazyMode) {
    LazyTestGuard guard;

    auto t = Tensor::bernoulli({500}, 0.5f, Device::CPU, DataType::Float32);
    EXPECT_FALSE(t.has_lazy_expr());
    EXPECT_TRUE(t.is_valid());
}

TEST(TensorLazyStatefulOpsTest, InplaceNormalIsEagerInLazyMode) {
    LazyTestGuard guard;

    auto t = Tensor::empty({500}, Device::CPU, DataType::Float32);
    t.normal_(0.0f, 1.0f);
    EXPECT_FALSE(t.has_lazy_expr());
    EXPECT_TRUE(t.is_valid());
}

TEST(TensorLazyStatefulOpsTest, InplaceUniformIsEagerInLazyMode) {
    LazyTestGuard guard;

    auto t = Tensor::empty({500}, Device::CPU, DataType::Float32);
    t.uniform_(0.0f, 1.0f);
    EXPECT_FALSE(t.has_lazy_expr());
    EXPECT_TRUE(t.is_valid());
}

TEST(TensorLazyStatefulOpsTest, ManualSeedReproducibilityWithLazyChain) {
    LazyTestGuard guard;

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
    LazyTestGuard guard;

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

    LazyTestGuard guard;

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
}
