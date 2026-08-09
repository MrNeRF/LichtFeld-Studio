/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Do not call teardown_gpu_before_exit() in the parent process mid-suite; it
// shuts down the pool needed by later tests. CUDA death tests are also avoided
// because forked CUDA processes are unreliable.

#include "components/ppisp_controller.hpp"
#include "core/tensor.hpp"
#include "training/strategies/strategy_utils.hpp"

#include <gtest/gtest.h>

namespace {

    TEST(GpuTeardownOrderTest, ReleaseSharedBuffersIsIdempotent) {
        using lfs::training::PPISPController;
        PPISPController::preallocate_shared_buffers(32, 32);
        PPISPController::release_shared_buffers();
        PPISPController::release_shared_buffers();
        // Re-allocate after release must work (pool still alive in-process).
        PPISPController::preallocate_shared_buffers(32, 32);
        PPISPController::release_shared_buffers();
        SUCCEED();
    }

    TEST(GpuTeardownOrderTest, DensifyNScratchReleaseDropsStorage) {
        using lfs::core::Device;
        using lfs::training::DensifyNScratch;

        DensifyNScratch scratch;
        scratch.ensure_n(128, Device::CUDA);
        scratch.ensure_k(16, Device::CUDA);
        ASSERT_GT(scratch.n_capacity, 0u);
        ASSERT_GT(scratch.k_capacity, 0u);
        ASSERT_TRUE(scratch.f32_a.is_valid());
        ASSERT_TRUE(scratch.i64_a.is_valid());

        scratch.release();
        EXPECT_EQ(scratch.n_capacity, 0u);
        EXPECT_EQ(scratch.k_capacity, 0u);
        EXPECT_FALSE(scratch.f32_a.is_valid());
        EXPECT_FALSE(scratch.i64_a.is_valid());
    }

    TEST(GpuTeardownOrderTest, DensifyNScratchReleasesAfterPoolTouch) {
        // Exercise zeros_direct (f32/bool) + pool empty (i64) free paths in the
        // same scope as a normal pooled tensor — both free before process exit.
        using lfs::core::Device;
        using lfs::core::Tensor;
        using lfs::training::DensifyNScratch;

        DensifyNScratch scratch;
        scratch.ensure_n(4096, Device::CUDA);
        scratch.ensure_k(256, Device::CUDA);
        Tensor pooled = Tensor::empty({1024}, Device::CUDA);
        ASSERT_TRUE(pooled.is_valid());

        scratch.release();
        pooled = {};
        SUCCEED();
    }

} // namespace
