/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/tensor_backend_vulkan.hpp"
#include "../../internal/tensor_impl.hpp"
#include "../tensor_vulkan_interop.hpp"
#include "vk_context.hpp"
#include "vk_memory.hpp"
#include "vk_recorder.hpp"
namespace lfs::core {
    void TensorVulkanInterop::upload_host(Tensor& tensor, std::span<const std::byte> bytes) {

        if (!tensor.is_valid() || tensor.device() != Device::GPU ||
            !tensor.is_contiguous() || bytes.size() > tensor.bytes())
            throw TensorError("Vulkan-visible host upload requires a contiguous GPU tensor with sufficient capacity");
        if (bytes.empty())
            return;
        internal::backend_ops_for(tensor).copy_host_to_device(internal::CopyRequest{
            .src = internal::raw_storage_ref(const_cast<std::byte*>(bytes.data())),
            .dst = internal::storage_ref(tensor),
            .bytes = bytes.size(),
            .synchronous = false,
            .context = internal::ExecContext{tensor.stream()},
        });
    }
    lfs::Status adopt_vulkan_device(const VulkanDeviceHandles& handles,
                                    const bool subgroup_size_control_enabled) {

        return internal::adopt_vulkan_context(internal::AdoptedDevice{
            .instance = static_cast<VkInstance>(handles.instance),
            .physical_device = static_cast<VkPhysicalDevice>(handles.physical_device),
            .device = static_cast<VkDevice>(handles.device),
            .queue = static_cast<VkQueue>(handles.queue),
            .queue_family = handles.queue_family,
            .sharing_queue_families = handles.sharing_queue_families,
            .sharing_queue_family_count = handles.sharing_queue_family_count,
            .shader_atomic_float = handles.shader_atomic_float,
            .memory_budget = handles.memory_budget,
            .shader_float64 = handles.shader_float64,
            .shader_float16 = handles.shader_float16,
            .vulkan_memory_model = handles.vulkan_memory_model,
            .vulkan_memory_model_device_scope = handles.vulkan_memory_model_device_scope,
            .cooperative_matrix = handles.cooperative_matrix,
            .external_memory = handles.external_memory,
            .external_semaphore = handles.external_semaphore,
#ifdef __APPLE__
            .metal_objects = handles.metal_objects,
#endif
            .subgroup_size_control = subgroup_size_control_enabled,
            .consumer_queue = static_cast<VkQueue>(handles.consumer_queue),
            .consumer_queue_mutex = handles.consumer_queue_mutex,
        });
    }
    bool vulkan_backend_adopted() {

        return internal::vulkan_context_adopted();
    }
    std::optional<TensorVulkanBuffer> tensor_vulkan_buffer(const Tensor& tensor) {

        auto result = internal::native_vulkan_buffer(tensor);
        if (result)
            internal::try_live_vulkan_context()->recorders().flush_storage(internal::storage_ref(tensor));
        return result;
    }
} // namespace lfs::core
