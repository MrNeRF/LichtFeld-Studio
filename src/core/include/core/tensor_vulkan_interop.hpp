/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/gpu_backend_fwd.hpp"
#include "core/tensor_completion.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string_view>

namespace lfs::core {
    class TensorShape;

    struct TensorVulkanBuffer {
        void* buffer = nullptr;
        uint64_t offset = 0;
        uint64_t device_address = 0;
        uint64_t bytes = 0;
        uint64_t pending_timeline_value = 0;
        std::shared_ptr<void> keep_alive;
        void* device = nullptr;
    };

    // Handles belong to the consumer. Its device and queue mutex must outlive
    // the interop session and all buffer / timeline leases it returns.
    struct VulkanInteropDevice {
        void* physical_device = nullptr;
        void* device = nullptr;
        void* sparse_queue = nullptr;
        std::mutex* queue_mutex = nullptr;
        std::array<uint32_t, 3> queue_families{};
        uint32_t queue_family_count = 0;
        bool sparse_binding = false;
        bool external_memory = false;
        bool external_semaphore = false;
        // VK_EXT_metal_objects: Metal tensors import into the device without copies.
        bool metal_objects = false;
    };

    class LFS_CORE_API TensorVulkanInterop {
    public:
        explicit TensorVulkanInterop(VulkanInteropDevice device);
        ~TensorVulkanInterop();
        TensorVulkanInterop(const TensorVulkanInterop&) = delete;
        TensorVulkanInterop& operator=(const TensorVulkanInterop&) = delete;

        [[nodiscard]] std::function<Tensor(TensorShape, size_t, DataType, std::string_view)>
        splat_allocator(bool preserve_float_shN = false);
        void drain(GpuBackend backend);
        // Runs `release` while no tensor work is in flight on the device.
        // MoltenVK keeps all device memory resident for every queue, so memory
        // freed while any queue has unfinished work faults.
        void run_while_idle(GpuBackend backend, const std::function<void()>& release);
        [[nodiscard]] std::shared_ptr<void> execution_scope(GpuBackend backend);
        [[nodiscard]] Tensor empty(TensorShape shape, DataType dtype,
                                   GpuBackend backend, size_t capacity = 0);
        [[nodiscard]] std::optional<TensorVulkanBuffer> buffer(const Tensor& tensor);
        void upload_host(Tensor& tensor, std::span<const std::byte> bytes);
        [[nodiscard]] TensorCompletion ready(std::span<const Tensor* const> tensors);
        void wait(std::span<const Tensor* const> tensors, VulkanTimelinePoint point);
        // The consumer calls this after its last submit, before destroying the semaphore.
        void release_timeline(void* semaphore);

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

    // How the Vulkan renderer receives splats a backend produced: drawn from that storage as is, moved
    // into renderer storage first, or copied so readers of the displayed storage never hold up the
    // producer's next use of it.
    enum class SplatPublication : uint8_t { Shared,
                                            RendererStorage,
                                            Copied };
    [[nodiscard]] LFS_CORE_API SplatPublication splat_publication(GpuBackend backend);

    // All operands are contiguous GPU tensors on one backend; output and source
    // have the same shape and Float32 or Float16 dtype. The mask has one Bool per element.
    LFS_CORE_API void where_into(Tensor& output, const Tensor& condition, float value, const Tensor& source);

    LFS_CORE_API std::optional<TensorVulkanBuffer> tensor_vulkan_buffer(const Tensor& tensor);
} // namespace lfs::core
