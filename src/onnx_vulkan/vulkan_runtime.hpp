/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "lfs_onnx_vulkan/onnx_vulkan.hpp"

#include <vulkan/vulkan.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>

namespace lfs::onnx_vulkan::detail {

    enum class Kernel : std::uint8_t {
        Elementwise,
        Transform,
        PackFp16,
        MatMul,
        MatMulTiled,
        MatMulSmallK,
        MatMulCooperative,
        MatMulCooperativeFp16Weights,
        Conv,
        Conv1x1,
        ConvTiled,
        ConvCooperative,
        ConvCooperative3x3,
        ConvTransposeCooperative2x2,
        Im2ColFp16,
        ConvTranspose,
        ConvTransposeTiled,
        Reduce,
        ReduceSerial,
        Softmax,
        LayerNorm,
        Count,
    };

    struct Buffer {
        VkDevice device = VK_NULL_HANDLE;
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize size = 0;
        void* mapped = nullptr;

        Buffer() = default;
        Buffer(const Buffer&) = delete;
        Buffer& operator=(const Buffer&) = delete;
        Buffer(Buffer&& other) noexcept;
        Buffer& operator=(Buffer&& other) noexcept;
        ~Buffer();
    };

    struct BufferBinding {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceSize offset = 0;
        VkDeviceSize range = 0;
    };

    class VulkanRuntime final {
    public:
        VulkanRuntime(const VulkanRuntime&) = delete;
        VulkanRuntime& operator=(const VulkanRuntime&) = delete;
        ~VulkanRuntime();

        [[nodiscard]] static std::expected<std::unique_ptr<VulkanRuntime>, Error>
        create(const SessionOptions& options);

        [[nodiscard]] std::expected<Buffer, Error>
        create_buffer(VkDeviceSize size,
                      VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags required,
                      VkMemoryPropertyFlags preferred = 0) const;

        [[nodiscard]] std::expected<void, Error>
        upload(Buffer& destination, VkDeviceSize offset, std::span<const std::byte> bytes) const;

        [[nodiscard]] std::expected<void, Error>
        immediate_submit(const std::function<void(VkCommandBuffer)>& record) const;

        [[nodiscard]] std::expected<VkDescriptorPool, Error>
        create_descriptor_pool(std::uint32_t max_sets) const;

        [[nodiscard]] std::expected<VkDescriptorSet, Error>
        allocate_descriptor_set(VkDescriptorPool pool) const;

        void update_descriptor_set(VkDescriptorSet set,
                                   const std::array<BufferBinding, 5>& bindings) const;

        void bind_and_dispatch(VkCommandBuffer command,
                               Kernel kernel,
                               VkDescriptorSet descriptor_set,
                               std::uint64_t invocation_count,
                               std::array<std::uint32_t, 3> workgroups = {}) const;

        [[nodiscard]] VkDevice device() const noexcept { return device_; }
        [[nodiscard]] VkQueue queue() const noexcept { return queue_; }
        [[nodiscard]] VkCommandPool command_pool() const noexcept { return command_pool_; }
        [[nodiscard]] VkDescriptorSetLayout descriptor_layout() const noexcept { return descriptor_layout_; }
        [[nodiscard]] VkPipelineLayout pipeline_layout() const noexcept { return pipeline_layout_; }
        [[nodiscard]] VkDeviceSize storage_alignment() const noexcept { return storage_alignment_; }
        [[nodiscard]] VkDeviceSize maximum_storage_range() const noexcept { return maximum_storage_range_; }
        [[nodiscard]] std::uint32_t maximum_group_count_x() const noexcept { return maximum_group_count_x_; }
        [[nodiscard]] std::uint32_t maximum_group_count_y() const noexcept { return maximum_group_count_y_; }
        [[nodiscard]] std::uint32_t maximum_group_count_z() const noexcept { return maximum_group_count_z_; }
        [[nodiscard]] bool profiling_enabled() const noexcept { return profiling_enabled_; }
        [[nodiscard]] bool cooperative_matrix_enabled() const noexcept { return cooperative_matrix_enabled_; }
        [[nodiscard]] bool native_fp16_storage_enabled() const noexcept { return native_fp16_storage_enabled_; }
        [[nodiscard]] float timestamp_period() const noexcept { return timestamp_period_; }
        [[nodiscard]] std::uint32_t timestamp_valid_bits() const noexcept { return timestamp_valid_bits_; }
        [[nodiscard]] std::string_view device_name() const noexcept { return device_name_; }

    private:
        VulkanRuntime() = default;
        [[nodiscard]] std::expected<void, Error> initialize(const SessionOptions& options);
        [[nodiscard]] std::expected<void, Error> create_pipelines();
        [[nodiscard]] std::expected<std::uint32_t, Error>
        memory_type(std::uint32_t bits,
                    VkMemoryPropertyFlags required,
                    VkMemoryPropertyFlags preferred) const;
        void save_pipeline_cache() const noexcept;

        VkInstance instance_ = VK_NULL_HANDLE;
        VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
        VkDevice device_ = VK_NULL_HANDLE;
        VkQueue queue_ = VK_NULL_HANDLE;
        std::uint32_t queue_family_ = 0;
        VkCommandPool command_pool_ = VK_NULL_HANDLE;
        VkDescriptorSetLayout descriptor_layout_ = VK_NULL_HANDLE;
        VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
        VkPipelineCache pipeline_cache_ = VK_NULL_HANDLE;
        std::array<VkPipeline, static_cast<std::size_t>(Kernel::Count)> pipelines_{};
        VkPhysicalDeviceMemoryProperties memory_properties_{};
        VkDeviceSize storage_alignment_ = 1;
        VkDeviceSize maximum_storage_range_ = 0;
        std::uint32_t maximum_group_count_x_ = 1;
        std::uint32_t maximum_group_count_y_ = 1;
        std::uint32_t maximum_group_count_z_ = 1;
        bool profiling_enabled_ = false;
        bool cooperative_matrix_enabled_ = false;
        bool native_fp16_storage_enabled_ = false;
        float timestamp_period_ = 0.0f;
        std::uint32_t timestamp_valid_bits_ = 0;
        std::string device_name_;
        std::filesystem::path pipeline_cache_path_;
    };

} // namespace lfs::onnx_vulkan::detail
