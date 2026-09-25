/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/tensor/internal/private_access.hpp"

#include "core/tensor.hpp"
#include "core/tensor_vulkan_interop.hpp"

namespace lfs::core::internal {
    void cuda_where_into(Tensor& output, const Tensor& condition, float value, const Tensor& source);
    void vulkan_where_into(Tensor& output, const Tensor& condition, float value, const Tensor& source);
    void metal_where_into(Tensor& output, const Tensor& condition, float value, const Tensor& source);
    std::optional<TensorVulkanBuffer> native_vulkan_buffer(const Tensor& tensor);
    class TensorVulkanInteropBackend {
    public:
        virtual ~TensorVulkanInteropBackend() = default;
        virtual void shutdown() {}
        virtual void drain() = 0;
        virtual void release_timeline(void* semaphore) = 0;
        virtual std::shared_ptr<void> execution_scope() = 0;
        virtual Tensor empty_splat(TensorShape shape, size_t capacity, DataType dtype,
                                   std::string_view name, bool preserve_float_shN) = 0;
        virtual Tensor empty(TensorShape shape, DataType dtype, size_t capacity) = 0;
        virtual std::optional<TensorVulkanBuffer> buffer(const Tensor& tensor) = 0;
        virtual TensorCompletion ready(std::span<const Tensor* const> tensors) = 0;
        virtual void wait(std::span<const Tensor* const> tensors, VulkanTimelinePoint point) = 0;
    };

    std::shared_ptr<TensorVulkanInteropBackend> make_cuda_vulkan_interop(VulkanInteropDevice device);
    std::shared_ptr<TensorVulkanInteropBackend> make_vulkan_vulkan_interop(VulkanInteropDevice device);
    std::shared_ptr<TensorVulkanInteropBackend> make_metal_vulkan_interop(VulkanInteropDevice device);
} // namespace lfs::core::internal
