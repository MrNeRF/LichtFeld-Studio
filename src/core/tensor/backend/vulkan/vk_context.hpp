/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/export.hpp"

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

struct VmaAllocator_T;
using VmaAllocator = VmaAllocator_T*;
struct VmaAllocation_T;
using VmaAllocation = VmaAllocation_T*;

namespace lfs::core::internal {

    class VulkanMemory;
    class VulkanPipelines;
    class VulkanRecorderRegistry;

    struct VkDeviceCaps {
        std::array<uint8_t, VK_UUID_SIZE> device_uuid{};
        std::array<uint8_t, VK_UUID_SIZE> driver_uuid{};
        std::array<uint32_t, 3> max_workgroup_size{};
        uint32_t device_index = 0;
        uint32_t subgroup_size = 0;
        uint32_t max_workgroup_invocations = 0;
        uint32_t shared_memory_size = 0;
        float timestamp_period = 0.0f;
        bool shader_float16 = false;
        bool shader_atomic_float = false;
        bool memory_budget = false;
        bool host_visible_device_local = false;
        bool direct_host_uploads = false;
    };

    class VulkanContext final {
    public:
        VulkanContext();
        ~VulkanContext();

        VulkanContext(const VulkanContext&) = delete;
        VulkanContext& operator=(const VulkanContext&) = delete;

        [[nodiscard]] VkInstance instance() const noexcept { return instance_; }
        [[nodiscard]] VkPhysicalDevice physical_device() const noexcept {
            return physical_device_;
        }
        [[nodiscard]] VkDevice device() const noexcept { return device_; }
        [[nodiscard]] VkQueue queue() const noexcept { return queue_; }
        [[nodiscard]] uint32_t queue_family() const noexcept { return queue_family_; }
        [[nodiscard]] VkSemaphore timeline() const noexcept { return timeline_; }
        [[nodiscard]] VkPipelineCache pipeline_cache() const noexcept {
            return pipeline_cache_;
        }
        [[nodiscard]] VmaAllocator allocator() const noexcept { return allocator_; }
        [[nodiscard]] const VkDeviceCaps& caps() const noexcept { return caps_; }
        [[nodiscard]] const VkPhysicalDeviceMemoryProperties& memory_properties() const noexcept {
            return memory_properties_;
        }
        [[nodiscard]] uint32_t device_index() const noexcept { return device_index_; }
        [[nodiscard]] uint64_t context_id() const noexcept { return context_id_; }
        [[nodiscard]] bool accepting_work() const noexcept {
            return accepting_work_.load(std::memory_order_acquire);
        }
        [[nodiscard]] bool dead() const noexcept {
            return dead_.load(std::memory_order_acquire);
        }

        [[nodiscard]] uint64_t reserve_timeline_value();
        void submit(VkCommandBuffer command, uint64_t signal_value);
        void wait(uint64_t value);
        [[nodiscard]] uint64_t completed_timeline() const;
        void check_fault_buffer();
        void mark_device_lost_once();

        [[nodiscard]] VulkanMemory& memory();
        [[nodiscard]] VulkanRecorderRegistry& recorders();
        [[nodiscard]] VulkanPipelines& pipelines();

        void shutdown();

    private:
        void create_instance();
        void select_physical_device();
        void create_device();
        void create_allocator();
        void create_pipeline_cache();
        void create_fault_buffer();
        void save_pipeline_cache() noexcept;
        void destroy_fault_buffer() noexcept;

        VkInstance instance_ = VK_NULL_HANDLE;
        VkDebugUtilsMessengerEXT debug_messenger_ = VK_NULL_HANDLE;
        VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
        VkDevice device_ = VK_NULL_HANDLE;
        VkQueue queue_ = VK_NULL_HANDLE;
        uint32_t queue_family_ = 0;
        uint32_t device_index_ = 0;
        uint64_t context_id_ = 0;
        VkSemaphore timeline_ = VK_NULL_HANDLE;
        VkPipelineCache pipeline_cache_ = VK_NULL_HANDLE;
        VmaAllocator allocator_ = nullptr;
        VmaAllocation fault_allocation_ = nullptr;
        VkBuffer fault_buffer_ = VK_NULL_HANDLE;
        void* fault_mapped_ = nullptr;
        VkPhysicalDeviceProperties properties_{};
        VkPhysicalDeviceMemoryProperties memory_properties_{};
        VkDeviceCaps caps_{};
        std::atomic<uint64_t> next_timeline_{0};
        std::atomic<bool> accepting_work_{true};
        std::atomic<bool> dead_{false};
        std::atomic<bool> device_loss_reported_{false};
        std::mutex queue_mutex_;
        std::mutex shutdown_mutex_;
        std::unique_ptr<VulkanMemory> memory_;
        std::unique_ptr<VulkanRecorderRegistry> recorders_;
        std::unique_ptr<VulkanPipelines> pipelines_;
        std::string pipeline_cache_path_;
    };

    [[nodiscard]] bool vulkan_backend_probe_available() noexcept;
    [[nodiscard]] std::shared_ptr<VulkanContext> acquire_vulkan_context();
    [[nodiscard]] std::shared_ptr<VulkanContext> try_live_vulkan_context() noexcept;
    void shutdown_vulkan_context();

    [[nodiscard]] LFS_CORE_API std::vector<std::string> vulkan_validation_messages_for_testing();
    [[nodiscard]] LFS_CORE_API VkDeviceCaps vulkan_device_caps_for_testing();
    [[nodiscard]] LFS_CORE_API uint64_t vulkan_live_vma_objects_for_testing() noexcept;
    [[nodiscard]] LFS_CORE_API uint64_t vulkan_completed_timeline_for_testing();
    [[nodiscard]] LFS_CORE_API size_t vulkan_dead_recorder_count_for_testing();

    void vk_check(VulkanContext* context, VkResult result, const char* operation);

} // namespace lfs::core::internal
