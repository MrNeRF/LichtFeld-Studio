/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vk_backend_ops.hpp"

#include "../../internal/tensor_impl.hpp"
#include "vk_context.hpp"
#include "vk_memory.hpp"
#include "vk_recorder.hpp"

namespace lfs::core::internal {

    StorageRef VulkanBackendOps::allocate(const size_t bytes, const size_t alignment,
                                          const ExecContext context) {
        return acquire_vulkan_context()->memory().allocate(bytes, alignment, context);
    }

    void VulkanBackendOps::deallocate(const StorageRef storage, ExecContext) noexcept {
        try {
            if (const auto context = try_live_vulkan_context()) {
                context->memory().deallocate(storage);
            }
        } catch (...) {
        }
    }

    void VulkanBackendOps::record_stream(StorageRef, ExecContext) {}

    void VulkanBackendOps::release_stream(ExecContext) {}

    void VulkanBackendOps::rehome_stream(StorageRef, ExecContext) {}

    void VulkanBackendOps::trim() {
        acquire_vulkan_context()->memory().trim();
    }

    void VulkanBackendOps::trim_if_reserved_unused_exceeds(
        const size_t threshold_bytes) {
        const auto context = acquire_vulkan_context();
        if (context->memory().cached_bytes() > threshold_bytes) {
            context->memory().trim();
        }
    }

    MemoryInfo VulkanBackendOps::stats() {
        return acquire_vulkan_context()->memory().stats();
    }

    void VulkanBackendOps::shutdown() {
        shutdown_vulkan_context();
    }

    void VulkanBackendOps::set_allocation_iteration(int) {}

    void VulkanBackendOps::record_tensor_allocation(
        StorageRef, const StridedLayout&, size_t) {}

    void VulkanBackendOps::copy_host_to_device(const CopyRequest& request) {
        acquire_vulkan_context()->memory().copy_host_to_device(request);
    }

    void VulkanBackendOps::copy_device_to_host(const CopyRequest& request) {
        acquire_vulkan_context()->memory().copy_device_to_host(request);
    }

    void VulkanBackendOps::copy_device_to_device(const CopyRequest& request) {
        acquire_vulkan_context()->memory().copy_device_to_device(request);
    }

    void VulkanBackendOps::memset(const FillRequest& request) {
        acquire_vulkan_context()->memory().memset(request);
    }

    void VulkanBackendOps::synchronize_stream(ExecContext) {
        const auto context = acquire_vulkan_context();
        context->recorders().wait_all();
        context->check_fault_buffer();
    }

    void VulkanBackendOps::synchronize_device() {
        synchronize_stream({});
    }

    void VulkanBackendOps::wait_for(const SyncToken token) {
        LFS_ASSERT_MSG(token.backend == GpuBackend::Vulkan,
                       "Vulkan sync service received a non-Vulkan token");
        const auto context = acquire_vulkan_context();
        context->recorders().flush_all();
        context->wait(token.value);
        context->check_fault_buffer();
    }

    SyncToken VulkanBackendOps::bridge(ExecContext, ExecContext) {
        const uint64_t value = acquire_vulkan_context()->recorders().flush_all();
        return SyncToken{
            .backend = GpuBackend::Vulkan,
            .value = value,
            .native = 0,
        };
    }

    PointerClass VulkanBackendOps::classify_pointer(const void* const pointer) {
        if (pointer == nullptr) {
            return PointerClass::Unknown;
        }
        if (const auto context = try_live_vulkan_context();
            context && context->memory().owns_address(pointer)) {
            return PointerClass::Device;
        }
        return PointerClass::Unknown;
    }

    bool VulkanBackendOps::stream_is_capturing(ExecContext) {
        return false;
    }

} // namespace lfs::core::internal
