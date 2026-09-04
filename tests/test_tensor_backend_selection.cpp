/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>

#include "core/tensor.hpp"
#include "core/tensor/internal/lazy_executor.hpp"
#include "core/tensor_backend.hpp"

#include <atomic>
#include <cstdlib>
#include <cuda_runtime.h>
#include <exception>
#include <string>
#include <thread>

namespace {

    using namespace lfs::core;

    void unset_backend_environment() {
#if defined(_WIN32)
        _putenv_s("LFS_TENSOR_BACKEND", "");
#else
        unsetenv("LFS_TENSOR_BACKEND");
#endif
    }

    void set_backend_environment(const char* value) {
#if defined(_WIN32)
        _putenv_s("LFS_TENSOR_BACKEND", value);
#else
        setenv("LFS_TENSOR_BACKEND", value, 1);
#endif
    }

    std::string exception_message(const auto& operation) {
        try {
            operation();
        } catch (const std::exception& error) {
            return error.what();
        }
        return {};
    }

    TEST(TensorBackendSelection, AProcessDefaultFreezesAfterFirstResolution) {
        unset_backend_environment();
        internal::gpu_backend_reset_for_testing();

        // Catches a missing CUDA fallback when no selector or environment is set.
        EXPECT_EQ(default_gpu_backend(), GpuBackend::CUDA);
        internal::gpu_backend_reset_for_testing();

        // Catches a thread scope freezing the otherwise untouched process default.
        {
            GpuBackendScope scope(GpuBackend::Vulkan);
            EXPECT_FALSE(exception_message([] {
                             (void)Tensor::empty({1}, Device::CUDA);
                         }).empty());
        }

        // Catches environment reads that override an explicit pre-resolution setter.
        const lfs::Status accepted = set_default_gpu_backend(GpuBackend::CUDA);
        ASSERT_TRUE(accepted.has_value());
        EXPECT_EQ(default_gpu_backend(), GpuBackend::CUDA);

        // Catches a mutable process default after its first resolution.
        const lfs::Status rejected = set_default_gpu_backend(GpuBackend::Vulkan);
        ASSERT_FALSE(rejected.has_value());
        const std::string message(rejected.error().user_message());
        EXPECT_NE(message.find("CUDA"), std::string::npos);
        EXPECT_NE(message.find("Vulkan"), std::string::npos);
        EXPECT_EQ(default_gpu_backend(), GpuBackend::CUDA);
    }

    TEST(TensorBackendSelection, ConfiguredDefaultPrecedesEnvironmentResolution) {
        internal::gpu_backend_reset_for_testing();
        set_backend_environment("vulkan");

        // Catches deferred environment resolution overriding an explicit selector.
        const lfs::Status accepted = set_default_gpu_backend(GpuBackend::CUDA);
        EXPECT_TRUE(accepted.has_value());
        EXPECT_EQ(default_gpu_backend(), GpuBackend::CUDA);

        unset_backend_environment();
        internal::gpu_backend_reset_for_testing();
    }

    TEST(TensorBackendSelection, ScopeNestsRestoresAndIsThreadLocal) {
        // Catches a scope implemented as a process-global mutable selector.
        std::atomic<bool> other_thread_used_cuda = false;
        {
            GpuBackendScope outer(GpuBackend::Vulkan);
            std::thread other_thread([&] {
                Tensor tensor = Tensor::empty({1}, Device::CUDA);
                other_thread_used_cuda = gpu_backend_of(tensor) == GpuBackend::CUDA;
            });
            other_thread.join();

            const std::string outer_error = exception_message([] {
                (void)Tensor::empty({1}, Device::CUDA);
            });
            EXPECT_NE(outer_error.find("Vulkan"), std::string::npos);
            EXPECT_NE(outer_error.find("unavailable"), std::string::npos);

            {
                GpuBackendScope inner(GpuBackend::CUDA);
                const Tensor tensor = Tensor::empty({1}, Device::CUDA);
                EXPECT_EQ(gpu_backend_of(tensor), GpuBackend::CUDA);
            }

            const std::string restored_error = exception_message([] {
                (void)Tensor::empty({1}, Device::CUDA);
            });
            EXPECT_NE(restored_error.find("Vulkan"), std::string::npos);
        }

        EXPECT_TRUE(other_thread_used_cuda.load());
        const Tensor after_scope = Tensor::empty({1}, Device::CUDA);
        EXPECT_EQ(gpu_backend_of(after_scope), GpuBackend::CUDA);
    }

    TEST(TensorBackendSelection, StorageIdentityCoversCpuGpuAndViews) {
        // Catches backend identity stored on tensor handles instead of shared storage.
        const Tensor cpu = Tensor::zeros({2, 2}, Device::CPU);
        EXPECT_EQ(gpu_backend_of(cpu), std::nullopt);
        EXPECT_EQ(gpu_backend_of(Tensor{}), std::nullopt);

        const Tensor gpu = Tensor::zeros({2, 2}, Device::CUDA);
        const Tensor view = gpu.slice(0, 0, 1);
        EXPECT_EQ(gpu_backend_of(gpu), GpuBackend::CUDA);
        EXPECT_EQ(gpu_backend_of(view), gpu_backend_of(gpu));
    }

    TEST(TensorBackendSelection, CudaFromBlobIgnoresVulkanScope) {
        // Catches from_blob incorrectly consulting the no-input factory selector.
        void* pointer = nullptr;
        ASSERT_EQ(cudaMalloc(&pointer, sizeof(float)), cudaSuccess);
        {
            GpuBackendScope scope(GpuBackend::Vulkan);
            const Tensor tensor = Tensor::from_blob(
                pointer, {1}, Device::CUDA, DataType::Float32);
            EXPECT_EQ(gpu_backend_of(tensor), GpuBackend::CUDA);
        }
        ASSERT_EQ(cudaFree(pointer), cudaSuccess);
    }

    TEST(TensorBackendSelection, VulkanFactoryFailsWithoutChangingDefault) {
        // Catches unavailable backends falling through to a CUDA allocation.
        EXPECT_FALSE(gpu_backend_available(GpuBackend::Vulkan));
        {
            GpuBackendScope scope(GpuBackend::Vulkan);
            const std::string message = exception_message([] {
                (void)Tensor::zeros({2}, Device::CUDA);
            });
            EXPECT_NE(message.find("Vulkan"), std::string::npos);
            EXPECT_NE(message.find("unavailable"), std::string::npos);
        }
        EXPECT_EQ(default_gpu_backend(), GpuBackend::CUDA);
    }

    TEST(TensorBackendSelection, OperationsInheritInputBackendAgainstScope) {
        // Catches output, reduction, and lazy allocations that consult the active scope.
        const Tensor a = Tensor::full({8}, 2.0f, Device::CUDA);
        const Tensor b = Tensor::full({8}, 3.0f, Device::CUDA);

        internal::lazy_executor_set_pointwise_fusion_override_for_testing(true);
        internal::lazy_executor_set_size_heuristic_override_for_testing(false);
        internal::lazy_executor_set_size_threshold_override_for_testing(0);
        {
            GpuBackendScope scope(GpuBackend::Vulkan);
            const Tensor added = a.add(b);
            const Tensor reduced = a.sum();
            const Tensor lazy = a.mul(b).add(a);
            EXPECT_EQ(gpu_backend_of(added), GpuBackend::CUDA);
            EXPECT_EQ(gpu_backend_of(reduced), GpuBackend::CUDA);
            EXPECT_EQ(gpu_backend_of(lazy), GpuBackend::CUDA);
            EXPECT_TRUE(lazy.has_lazy_expr());
            EXPECT_FLOAT_EQ(reduced.item<float>(), 16.0f);
            EXPECT_FLOAT_EQ(lazy.sum().item<float>(), 64.0f);
        }
        internal::lazy_executor_set_size_threshold_override_for_testing(std::nullopt);
        internal::lazy_executor_set_size_heuristic_override_for_testing(std::nullopt);
        internal::lazy_executor_set_pointwise_fusion_override_for_testing(std::nullopt);
    }

    TEST(TensorBackendSelection, CopyToCudaClonesAndVulkanFailsCleanly) {
        // Catches same-backend aliasing and cross-backend CUDA fallthrough.
        const Tensor source = Tensor::full({4}, 7.0f, Device::CUDA);
        const Tensor clone = internal::copy_to_backend(source, GpuBackend::CUDA);
        ASSERT_EQ(gpu_backend_of(clone), GpuBackend::CUDA);
        EXPECT_NE(clone.data_ptr(), source.data_ptr());
        EXPECT_EQ(clone.to_vector(), source.to_vector());

        const std::string message = exception_message([&] {
            (void)internal::copy_to_backend(source, GpuBackend::Vulkan);
        });
        EXPECT_NE(message.find("Vulkan"), std::string::npos);
        EXPECT_NE(message.find("unavailable"), std::string::npos);
    }

    TEST(TensorBackendSelection, BackendMemoryAndShutdownAreDefinedForBothBackends) {
        // Catches placeholder Vulkan services leaking CUDA statistics or errors.
        EXPECT_TRUE(gpu_backend_available(GpuBackend::CUDA));
        const MemoryInfo vulkan = gpu_backend_memory_info(GpuBackend::Vulkan);
        EXPECT_EQ(vulkan.free_bytes, 0u);
        EXPECT_EQ(vulkan.total_bytes, 0u);
        EXPECT_EQ(vulkan.allocated_bytes, 0u);
        EXPECT_EQ(vulkan.device_id, -1);
        EXPECT_TRUE(shutdown_gpu_backend(GpuBackend::CUDA).has_value());
        EXPECT_TRUE(shutdown_gpu_backend(GpuBackend::Vulkan).has_value());
    }

} // namespace
