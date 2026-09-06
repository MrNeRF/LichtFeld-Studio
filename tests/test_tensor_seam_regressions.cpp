/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <vector>

namespace {

    using namespace lfs::core;

    bool has_cuda_device() {
        int device_count = 0;
        return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
    }

    // Catches read_scalar bounding a storage-linear index by numel(): a column
    // slice of a 4x4 matrix has numel 8 but its last element sits at linear
    // storage index 13.
    TEST(TensorSeamRegressions, ScalarReadsOnStridedViewsUseStorageExtent) {
        if (!has_cuda_device())
            GTEST_SKIP() << "CUDA device required";
        std::vector<float> values(16);
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = static_cast<float>(i);
        const Tensor matrix = Tensor::from_vector(values, {4, 4}, Device::CUDA);
        const Tensor columns = matrix.slice(1, 0, 2);
        ASSERT_EQ(columns.numel(), 8u);
        EXPECT_FLOAT_EQ(columns.at({3, 1}), 13.0f);
        EXPECT_FLOAT_EQ(columns.at({0, 1}), 1.0f);
        const Tensor column = matrix.slice(1, 1, 2);
        EXPECT_FLOAT_EQ(column.at({3, 0}), 13.0f);
    }

    // Catches in-place facade writes that skip lazy snapshot preservation: a
    // deferred expression must see the operand as it was when it was built.
    TEST(TensorSeamRegressions, InPlaceWritesPreserveLazySnapshots) {
        if (!has_cuda_device())
            GTEST_SKIP() << "CUDA device required";
        const std::vector<float> initial{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        Tensor x = Tensor::from_vector(initial, {8}, Device::CUDA);
        Tensor deferred = x.exp().mul(2.0f);
        x.add_(1.0f);
        const std::vector<float> observed = deferred.to_vector();
        ASSERT_EQ(observed.size(), initial.size());
        for (size_t i = 0; i < initial.size(); ++i)
            EXPECT_FLOAT_EQ(observed[i], 2.0f * std::exp(initial[i])) << "index " << i;

        Tensor y = Tensor::from_vector(initial, {8}, Device::CUDA);
        Tensor deferred_y = y.exp();
        y.clamp_(0.0f, 1.0f);
        const std::vector<float> observed_y = deferred_y.to_vector();
        for (size_t i = 0; i < initial.size(); ++i)
            EXPECT_FLOAT_EQ(observed_y[i], std::exp(initial[i])) << "index " << i;

        Tensor z = Tensor::from_vector(initial, {8}, Device::CUDA);
        Tensor deferred_z = z.exp();
        const Tensor mask = z.gt(4.0f);
        z.masked_fill_(mask, 0.0f);
        const std::vector<float> observed_z = deferred_z.to_vector();
        for (size_t i = 0; i < initial.size(); ++i)
            EXPECT_FLOAT_EQ(observed_z[i], std::exp(initial[i])) << "index " << i;
    }

    // Catches storage_ref dropping the stale-view check: a view taken before
    // reserve() must throw on a facade operation instead of reading the old storage.
    TEST(TensorSeamRegressions, StaleViewsAreRejectedByFacadeOperations) {
        if (!has_cuda_device())
            GTEST_SKIP() << "CUDA device required";
        Tensor base = Tensor::zeros({16}, Device::CUDA);
        const Tensor view = base.slice(0, 0, 8);
        base.reserve(1024);
        base.fill_(2.0f);
        EXPECT_THROW(static_cast<void>(view.sum_scalar()), std::runtime_error);
    }

} // namespace
