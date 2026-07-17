/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Phase 6B-2 P2 §6.2 — mandatory death-test proving a shared-tag
// LFS_CUDA_LAUNCH_CHECK inside a template-multiplied family
// (broadcast_binary_kernel via launch_broadcast_binary) fires and throws
// from a real instantiation under a poisoned CUDA sticky-error state.

#include "core/cuda_error_typed.hpp"
#include "core/tensor/internal/tensor_functors.hpp"
#include "core/tensor/internal/tensor_ops.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdlib>
#include <string>

namespace {

    class TensorLaunchCheckDeathTest : public ::testing::Test {
    protected:
        void SetUp() override {
            // The subprocess needs a WORKING CUDA context (malloc + real launch),
            // which fork() cannot guarantee once the parent holds a context.
            // threadsafe style re-execs the binary, so the child initializes
            // CUDA from scratch.
            saved_death_test_style_ = GTEST_FLAG_GET(death_test_style);
            GTEST_FLAG_SET(death_test_style, "threadsafe");
            lfs::core::reset_cuda_diagnostics_for_testing();
        }

        void TearDown() override {
            GTEST_FLAG_SET(death_test_style, saved_death_test_style_);
            (void)cudaGetLastError();
            lfs::core::reset_cuda_diagnostics_for_testing();
        }

    private:
        std::string saved_death_test_style_;
    };

    // Isolated so EXPECT_EXIT's macro argument parser is not fed template
    // commas (EXPECT_EXIT is a 3-arg function-style macro).
    [[noreturn]] void poisoned_broadcast_binary_subprocess() {
        int device = -1;
        if (cudaGetDevice(&device) != cudaSuccess) {
            (void)cudaGetLastError();
            std::_Exit(4);
        }

        // int,int,add_op forces the generic broadcast_binary_kernel path
        // (float→float takes specialized pattern kernels). Shared tag:
        // "tensor.broadcast.binary" (§9 sign-off 3).
        constexpr size_t a_n = 2 * 3 * 4;
        constexpr size_t b_n = 2 * 1 * 4;
        constexpr size_t c_n = 2 * 3 * 4;
        int* a = nullptr;
        int* b = nullptr;
        int* c = nullptr;
        if (cudaMalloc(&a, a_n * sizeof(int)) != cudaSuccess ||
            cudaMalloc(&b, b_n * sizeof(int)) != cudaSuccess ||
            cudaMalloc(&c, c_n * sizeof(int)) != cudaSuccess) {
            (void)cudaGetLastError();
            std::_Exit(5);
        }
        // Consume any sticky state from the allocations themselves.
        (void)cudaGetLastError();

        // Poison sticky CUDA error state (does not clear device identity).
        (void)cudaLaunchKernel(nullptr, dim3(1), dim3(1), nullptr, 0, nullptr);

        const size_t a_shape[] = {2, 3, 4};
        const size_t b_shape[] = {2, 1, 4};
        const size_t c_shape[] = {2, 3, 4};

        using AddOp = lfs::core::ops::add_op;
        try {
            lfs::core::tensor_ops::launch_broadcast_binary<int, int, AddOp>(
                a, b, c, a_shape, b_shape, c_shape,
                /*a_rank=*/3, /*b_rank=*/3, /*c_rank=*/3,
                /*c_elements=*/c_n, AddOp{}, /*stream=*/nullptr);
            std::_Exit(2);
        } catch (const lfs::Exception&) {
            std::_Exit(0);
        } catch (...) {
            std::_Exit(3);
        }
    }

} // namespace

TEST_F(TensorLaunchCheckDeathTest, PoisonedBroadcastBinaryTemplateLaunchThrows) {
    EXPECT_EXIT(poisoned_broadcast_binary_subprocess(),
                ::testing::ExitedWithCode(0),
                "");
}
