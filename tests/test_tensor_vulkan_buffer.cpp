/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/error.hpp"
#include "core/tensor.hpp"
#include "core/tensor/backend/gpu_backend_ops.hpp"
#include "core/tensor/backend/vulkan/vk_context.hpp"
#include "core/tensor_backend.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

namespace {
    using namespace lfs::core;

    class TensorVulkanBufferQuery : public testing::Test {
    protected:
        void SetUp() override {
            ASSERT_TRUE(gpu_backend_available(GpuBackend::Vulkan));
        }

        void TearDown() override {
            const auto status = shutdown_gpu_backend(GpuBackend::Vulkan);
            EXPECT_TRUE(status.has_value());
            EXPECT_EQ(internal::vulkan_live_vma_objects_for_testing(), 0u);
            for (const std::string& message :
                 internal::vulkan_validation_messages_for_testing()) {
                ADD_FAILURE() << message;
            }
        }
    };

    void wait_timeline_value(const uint64_t value) {
        internal::backend_ops(GpuBackend::Vulkan).synchronize_device();
        ASSERT_NE(vulkan_backend_timeline(), nullptr);
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(5);
        uint64_t counter = 0;
        do {
            counter = internal::vulkan_completed_timeline_for_testing();
            if (counter >= value) {
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } while (std::chrono::steady_clock::now() < deadline);
        FAIL() << "vulkan backend timeline did not reach " << value
               << " (counter=" << counter << ")";
    }

    TEST_F(TensorVulkanBufferQuery, PendingValueIsSignalledAfterSynchronize) {
        GpuBackendScope scope(GpuBackend::Vulkan);
        const Tensor source = Tensor::full({256}, 1.25f, Device::GPU);
        const Tensor result = source.mul(3.0f);
        const auto buffer = tensor_vulkan_buffer(result);
        ASSERT_TRUE(buffer.has_value());
        EXPECT_NE(buffer->buffer, nullptr);
        EXPECT_GT(buffer->pending_timeline_value, 0u);
        EXPECT_EQ(buffer->bytes, result.bytes());
        EXPECT_EQ(buffer->device_address,
                  static_cast<uint64_t>(reinterpret_cast<uintptr_t>(result.data_ptr())));
        ASSERT_NE(vulkan_backend_timeline(), nullptr);
        wait_timeline_value(buffer->pending_timeline_value);
    }

    TEST_F(TensorVulkanBufferQuery, ViewOffsetBytesAndDeviceAddressMatchStorage) {
        GpuBackendScope scope(GpuBackend::Vulkan);
        const Tensor base = Tensor::from_vector(
            std::vector<float>{1, 2, 3, 4, 5, 6}, {2, 3}, Device::GPU);
        const Tensor sliced = base.slice(0, 1, 2);
        const auto buffer = tensor_vulkan_buffer(sliced);
        ASSERT_TRUE(buffer.has_value());
        EXPECT_EQ(buffer->offset, 3u * sizeof(float));
        EXPECT_EQ(buffer->bytes, 3u * sizeof(float));
        EXPECT_EQ(buffer->device_address,
                  static_cast<uint64_t>(reinterpret_cast<uintptr_t>(sliced.data_ptr())));
        EXPECT_EQ(buffer->buffer, tensor_vulkan_buffer(base)->buffer);
    }

    TEST_F(TensorVulkanBufferQuery, KeepAlivePinsStorageAfterTensorDrop) {
        {
            GpuBackendScope scope(GpuBackend::Vulkan);
            const Tensor warm = Tensor::ones({8}, Device::GPU);
            ASSERT_FLOAT_EQ(warm.sum_scalar(), 8.0f);
        }
        Tensor::trim_memory_pool();
        const uint64_t baseline = internal::vulkan_live_vma_objects_for_testing();
        std::shared_ptr<void> pin;
        {
            GpuBackendScope scope(GpuBackend::Vulkan);
            Tensor tensor = Tensor::ones({4096}, Device::GPU);
            const auto buffer = tensor_vulkan_buffer(tensor);
            ASSERT_TRUE(buffer.has_value());
            ASSERT_TRUE(static_cast<bool>(buffer->keep_alive));
            pin = buffer->keep_alive;
            tensor = Tensor();
            Tensor::trim_memory_pool();
            EXPECT_GT(internal::vulkan_live_vma_objects_for_testing(), baseline);
        }
        Tensor::trim_memory_pool();
        EXPECT_GT(internal::vulkan_live_vma_objects_for_testing(), baseline);
        pin.reset();
        internal::backend_ops(GpuBackend::Vulkan).synchronize_device();
        Tensor::trim_memory_pool();
        EXPECT_EQ(internal::vulkan_live_vma_objects_for_testing(), baseline);
    }

    TEST_F(TensorVulkanBufferQuery, CudaAndCpuTensorsReturnNullopt) {
        {
            GpuBackendScope cuda_scope(GpuBackend::CUDA);
            const Tensor cuda = Tensor::ones({8}, Device::GPU);
            EXPECT_EQ(gpu_backend_of(cuda), GpuBackend::CUDA);
            EXPECT_FALSE(tensor_vulkan_buffer(cuda).has_value());
        }
        const Tensor cpu = Tensor::ones({8}, Device::CPU);
        EXPECT_FALSE(tensor_vulkan_buffer(cpu).has_value());
    }

    TEST_F(TensorVulkanBufferQuery, TimelineIsNullAfterShutdown) {
        {
            GpuBackendScope scope(GpuBackend::Vulkan);
            const Tensor tensor = Tensor::ones({8}, Device::GPU);
            ASSERT_TRUE(tensor_vulkan_buffer(tensor).has_value());
            ASSERT_NE(vulkan_backend_timeline(), nullptr);
        }
        ASSERT_TRUE(shutdown_gpu_backend(GpuBackend::Vulkan).has_value());
        EXPECT_EQ(vulkan_backend_timeline(), nullptr);
    }

} // namespace
