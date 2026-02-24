/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>

#include "core/tensor.hpp"
#include <thread>

using namespace lfs::core;

class TensorStorageVersionTest : public ::testing::Test {
protected:
    void SetUp() override {
        base_ = Tensor::zeros({100, 3}, Device::CUDA);
    }

    Tensor base_;
};

TEST_F(TensorStorageVersionTest, FreshAllocationHasStorageMeta) {
    auto t = Tensor::zeros({10, 3}, Device::CUDA);
    EXPECT_TRUE(t.is_valid());
}

TEST_F(TensorStorageVersionTest, ViewSharesStorageMetaWithBase) {
    auto view = base_.slice(0, 0, 50);
    EXPECT_TRUE(view.is_view());

    // Both should be accessible without assertion failure
    EXPECT_EQ(view.numel(), 50 * 3);
    EXPECT_NE(view.ptr<float>(), nullptr);
}

TEST_F(TensorStorageVersionTest, CloneGetsIndependentStorageMeta) {
    auto cloned = base_.clone();

    // Clone should have independent storage
    EXPECT_FALSE(cloned.is_view());
    EXPECT_NE(cloned.data_ptr(), base_.data_ptr());

    // Both should be independently accessible
    EXPECT_NE(base_.ptr<float>(), nullptr);
    EXPECT_NE(cloned.ptr<float>(), nullptr);
}

TEST_F(TensorStorageVersionTest, ReserveBumpsGenerationBeforeNewStorage) {
    auto with_capacity = Tensor::zeros({10, 3}, Device::CUDA);
    auto view_before = with_capacity.slice(0, 0, 5);

    // View should be valid before reserve
    EXPECT_NE(view_before.ptr<float>(), nullptr);

    // Reserve reallocates
    with_capacity.reserve(1000);

    // Base tensor should still be accessible
    EXPECT_NE(with_capacity.ptr<float>(), nullptr);
}

TEST_F(TensorStorageVersionTest, MutationWithoutReallocDoesNotInvalidateViews) {
    auto view = base_.slice(0, 0, 50);

    // In-place mutation does NOT reallocate
    base_.fill_(1.0f);

    // View should still be valid (shares same data_ pointer)
    EXPECT_NE(view.ptr<float>(), nullptr);
}

TEST_F(TensorStorageVersionTest, MultipleViewsShareSameGenerationCounter) {
    auto view1 = base_.slice(0, 0, 50);
    auto view2 = base_.slice(0, 50, 100);
    auto view3 = base_.reshape({300});

    // All views should be accessible
    EXPECT_NE(view1.ptr<float>(), nullptr);
    EXPECT_NE(view2.ptr<float>(), nullptr);
    EXPECT_NE(view3.ptr<float>(), nullptr);
}

TEST_F(TensorStorageVersionTest, NestedViewsShareSameGenerationCounter) {
    auto view1 = base_.slice(0, 0, 50);
    auto view2 = view1.slice(0, 0, 25);
    auto view3 = view2.reshape({75});

    // All nested views should be accessible
    EXPECT_NE(view1.ptr<float>(), nullptr);
    EXPECT_NE(view2.ptr<float>(), nullptr);
    EXPECT_NE(view3.ptr<float>(), nullptr);
}

TEST_F(TensorStorageVersionTest, PermuteProducesValidView) {
    auto t = Tensor::zeros({10, 3, 4}, Device::CUDA);
    auto permuted = t.permute({2, 0, 1});

    EXPECT_TRUE(permuted.is_view());
    EXPECT_NE(permuted.data_ptr(), nullptr);
}

TEST_F(TensorStorageVersionTest, TransposeProducesValidView) {
    auto t = Tensor::zeros({10, 20}, Device::CUDA);
    auto transposed = t.transpose(0, 1);

    EXPECT_TRUE(transposed.is_view());
    EXPECT_NE(transposed.data_ptr(), nullptr);
}

TEST_F(TensorStorageVersionTest, SqueezeUnsqueezeProduceValidViews) {
    auto t = Tensor::zeros({1, 10, 3}, Device::CUDA);

    auto squeezed = t.squeeze(0);
    EXPECT_NE(squeezed.data_ptr(), nullptr);

    auto unsqueezed = squeezed.unsqueeze(0);
    EXPECT_NE(unsqueezed.data_ptr(), nullptr);
}

TEST_F(TensorStorageVersionTest, CopyConstructorSharesMeta) {
    auto view = base_.slice(0, 0, 50);
    auto copy = view; // Copy constructor

    EXPECT_NE(copy.ptr<float>(), nullptr);
}

TEST_F(TensorStorageVersionTest, MoveConstructorTransfersMeta) {
    auto view = base_.slice(0, 0, 50);
    auto moved = std::move(view);

    EXPECT_NE(moved.ptr<float>(), nullptr);
}

TEST_F(TensorStorageVersionTest, ToDeviceGetsIndependentMeta) {
    auto cpu_tensor = base_.to(Device::CPU);

    // Should be independently valid
    EXPECT_NE(cpu_tensor.ptr<float>(), nullptr);
    EXPECT_NE(base_.ptr<float>(), nullptr);
}

TEST_F(TensorStorageVersionTest, ToDtypeGetsIndependentMeta) {
    auto int_tensor = base_.to(DataType::Int32);

    EXPECT_NE(int_tensor.ptr<int>(), nullptr);
    EXPECT_NE(base_.ptr<float>(), nullptr);
}

TEST_F(TensorStorageVersionTest, FromBlobHasStorageMeta) {
    float data[12] = {};
    auto t = Tensor::from_blob(data, {3, 4}, Device::CPU, DataType::Float32);

    EXPECT_NE(t.ptr<float>(), nullptr);
}

TEST_F(TensorStorageVersionTest, ZerosDirectHasStorageMeta) {
    auto t = Tensor::zeros_direct({10, 3}, 100, Device::CUDA);

    EXPECT_NE(t.ptr<float>(), nullptr);
}

// TensorRowProxy tests

class TensorRowProxyTest : public ::testing::Test {
protected:
    void SetUp() override {
        t_ = Tensor::from_vector({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, Device::CUDA);
    }

    Tensor t_;
};

TEST_F(TensorRowProxyTest, CudaReadReturnsCorrectValue) {
    float val = t_[0][1];
    EXPECT_FLOAT_EQ(val, 2.0f);

    float val2 = t_[1][2];
    EXPECT_FLOAT_EQ(val2, 6.0f);
}

TEST_F(TensorRowProxyTest, ConcurrentProxiesDoNotInterfere) {
    // Two proxies reading different values should not corrupt each other
    auto proxy_a = t_[0];
    auto proxy_b = t_[1];

    float a0 = proxy_a[0]; // 1.0
    float b0 = proxy_b[0]; // 4.0

    EXPECT_FLOAT_EQ(a0, 1.0f);
    EXPECT_FLOAT_EQ(b0, 4.0f);

    // Re-read through proxy_a — should still get correct value
    float a1 = proxy_a[1]; // 2.0
    EXPECT_FLOAT_EQ(a1, 2.0f);
}

TEST_F(TensorRowProxyTest, MultipleReadsFromSameProxy) {
    auto proxy = t_[0];

    float v0 = proxy[0];
    float v1 = proxy[1];
    float v2 = proxy[2];

    EXPECT_FLOAT_EQ(v0, 1.0f);
    EXPECT_FLOAT_EQ(v1, 2.0f);
    EXPECT_FLOAT_EQ(v2, 3.0f);
}

TEST_F(TensorRowProxyTest, CudaDoubleSubscriptAssignmentPersists) {
    t_[0][1] = 42.0f;
    t_[1][2] = -7.5f;

    auto cpu = t_.to(Device::CPU);
    auto values = cpu.to_vector();
    ASSERT_EQ(values.size(), 6u);
    EXPECT_FLOAT_EQ(values[1], 42.0f); // [0][1]
    EXPECT_FLOAT_EQ(values[5], -7.5f); // [1][2]
}

TEST_F(TensorRowProxyTest, CudaProxyFlushesAcrossMultipleElementWrites) {
    {
        auto row0 = t_[0];
        row0[0] = 10.0f;
        row0[1] = 20.0f;
        row0[2] = 30.0f;
    } // flush pending staged write for the last element

    auto cpu = t_.to(Device::CPU);
    auto values = cpu.to_vector();
    ASSERT_EQ(values.size(), 6u);
    EXPECT_FLOAT_EQ(values[0], 10.0f);
    EXPECT_FLOAT_EQ(values[1], 20.0f);
    EXPECT_FLOAT_EQ(values[2], 30.0f);
}

TEST_F(TensorRowProxyTest, CudaConstSubscriptSeesPendingWrite) {
    auto row0 = t_[0];
    row0[1] = 55.0f;

    const auto& row0_const = row0;
    const float read_back = row0_const[1];
    EXPECT_FLOAT_EQ(read_back, 55.0f);
}

TEST_F(TensorStorageVersionTest, StaleViewDetectedOnReserve) {
    auto with_capacity = Tensor::zeros({10, 3}, Device::CUDA);
    auto view = with_capacity.slice(0, 0, 5);

    // Reserve causes reallocation => bumps generation
    with_capacity.reserve(1000);

    // Accessing stale view should fail safely in all build types.
    EXPECT_THROW(
        { [[maybe_unused]] auto p = view.ptr<float>(); },
        std::runtime_error);
}

// CPU variant tests

TEST(TensorStorageVersionCPUTest, ViewAccessAfterMutationIsValid) {
    auto t = Tensor::zeros({100, 3}, Device::CPU);
    auto view = t.slice(0, 0, 50);

    t.fill_(42.0f);

    // View still points to same storage, should read mutated data
    float val = view.ptr<float>()[0];
    EXPECT_FLOAT_EQ(val, 42.0f);
}

TEST(TensorStorageVersionCPUTest, ReshapeViewIsValid) {
    auto t = Tensor::zeros({10, 3}, Device::CPU);
    auto view = t.reshape({30});

    EXPECT_NE(view.ptr<float>(), nullptr);
    EXPECT_EQ(view.numel(), 30);
}

TEST(TensorStorageVersionCPUTest, RowProxyConcurrentInstances) {
    auto t = Tensor::from_vector({10.0f, 20.0f, 30.0f, 40.0f}, {2, 2}, Device::CPU);

    auto p0 = t[0];
    auto p1 = t[1];

    EXPECT_FLOAT_EQ(p0[0], 10.0f);
    EXPECT_FLOAT_EQ(p1[1], 40.0f);
    EXPECT_FLOAT_EQ(p0[1], 20.0f);
}
