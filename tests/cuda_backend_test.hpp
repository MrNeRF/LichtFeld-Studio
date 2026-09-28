/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor_backend.hpp"

#include <gtest/gtest.h>

#include <optional>

namespace lfs::test {
    // --tensor-backend freezes the process default. Tests that switch it call this first.
    inline void reset_gpu_backend_for_testing() {
        core::internal::gpu_backend_reset_for_testing();
    }

    // Makes `backend` the process default, as a training session does, and
    // restores the previous default on destruction. The trainer resolves its
    // families through the default backend.
    class DefaultGpuBackendForTesting {
    public:
        explicit DefaultGpuBackendForTesting(const core::GpuBackend backend)
            : previous_(core::default_gpu_backend()) {
            core::internal::gpu_backend_reset_for_testing();
            switched_ = core::set_default_gpu_backend(backend).has_value();
            scope_.emplace(backend);
        }
        ~DefaultGpuBackendForTesting() {
            scope_.reset();
            core::internal::gpu_backend_reset_for_testing();
            (void)core::set_default_gpu_backend(previous_);
        }
        DefaultGpuBackendForTesting(const DefaultGpuBackendForTesting&) = delete;
        DefaultGpuBackendForTesting& operator=(const DefaultGpuBackendForTesting&) = delete;
        [[nodiscard]] bool switched() const { return switched_; }

    private:
        core::GpuBackend previous_;
        bool switched_ = false;
        std::optional<core::GpuBackendScope> scope_;
    };

    // Skips without a CUDA device; tensors stay on the backend --tensor-backend selects.
    class CudaDeviceTest : public ::testing::Test {
    protected:
        void SetUp() override {
            if (!core::gpu_backend_available(core::GpuBackend::CUDA)) {
                GTEST_SKIP() << "CUDA device unavailable";
            }
        }
    };

    // Worker threads and the test thread use CUDA until fixture destruction.
    class CudaBackendTest : public CudaDeviceTest {
    public:
        ~CudaBackendTest() override {
            if (previous_backend_) {
                core::internal::gpu_backend_reset_for_testing();
                (void)core::set_default_gpu_backend(*previous_backend_);
            }
        }

    protected:
        void SetUp() override {
            CudaDeviceTest::SetUp();
            if (!IsSkipped()) {
                previous_backend_ = core::default_gpu_backend();
                core::internal::gpu_backend_reset_for_testing();
                ASSERT_TRUE(core::set_default_gpu_backend(core::GpuBackend::CUDA));
                backend_scope_.emplace(core::GpuBackend::CUDA);
            }
        }

    private:
        std::optional<core::GpuBackend> previous_backend_;
        std::optional<core::GpuBackendScope> backend_scope_;
    };

} // namespace lfs::test

// GTEST_SKIP returns from CudaBackendTest::SetUp only, so the caller returns
// before allocating when no CUDA device is present.
#define LFS_CUDA_BACKEND_OR_RETURN() \
    do {                             \
        CudaBackendTest::SetUp();    \
        if (IsSkipped())             \
            return;                  \
    } while (0)
