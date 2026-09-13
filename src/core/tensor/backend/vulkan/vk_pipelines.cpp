/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vk_pipelines.hpp"

#include "core/assert.hpp"
#include "vk_context.hpp"
#include "vk_shader_table.hpp"

#include <algorithm>
#include <format>
#include <string_view>
#include <vector>

namespace lfs::core::internal {
    namespace {
        // Every capability a module may declare maps to a device feature the
        // context enables at creation; a capability outside this table means a
        // shader change that the runtime has not been taught to check.
        bool capability_provided(const VkDeviceCaps& caps, const std::string_view capability) {
            if (capability == "Shader" || capability == "Int64" || capability == "Int16" ||
                capability == "PhysicalStorageBufferAddresses" ||
                capability == "StorageBuffer16BitAccess" ||
                capability == "StorageBuffer8BitAccess") {
                return true;
            }
            if (capability == "SignedZeroInfNanPreserve") {
                return true;
            }
            if (capability == "Float16") {
                return caps.shader_float16;
            }
            if (capability == "AtomicFloat32AddEXT") {
                return caps.shader_atomic_float;
            }
            return false;
        }
    } // namespace

    VulkanPipelines::VulkanPipelines(VulkanContext& context)
        : context_(context) {}

    VulkanPipelines::~VulkanPipelines() {
        shutdown();
    }

    const VulkanPipeline& VulkanPipelines::specialized(
        const std::string& module, const uint32_t expected_push_constant_size,
        const std::span<const uint32_t> constants) {
        std::string key = module;
        for (const uint32_t constant : constants) {
            key += std::format(":{}", constant);
        }
        std::lock_guard lock(mutex_);
        LFS_ASSERT_MSG(!shutting_down_,
                       "Vulkan backend: pipeline creation during shutdown");
        auto iterator = pipelines_.find(key);
        if (iterator == pipelines_.end()) {
            iterator = pipelines_
                           .emplace(key, load(module, 256,
                                              expected_push_constant_size,
                                              constants))
                           .first;
        }
        return iterator->second;
    }

    VulkanPipeline VulkanPipelines::load(const std::string& module,
                                         const uint32_t expected_local_size_x,
                                         const uint32_t expected_push_constant_size,
                                         const std::span<const uint32_t> constants) {
        const EmbeddedShader* const shader = find_embedded_shader(module);
        LFS_ASSERT_MSG(shader != nullptr,
                       std::format("Vulkan backend: no embedded shader module named {}", module));
        LFS_ASSERT_MSG(shader->entry_point == "main",
                       "Vulkan backend: shader entry point mismatch");
        LFS_ASSERT_MSG(shader->local_size[0] == expected_local_size_x &&
                           shader->local_size[1] == 1 && shader->local_size[2] == 1,
                       "Vulkan backend: shader local size mismatch");
        LFS_ASSERT_MSG(shader->push_constant_size == expected_push_constant_size,
                       std::format("Vulkan backend: module {} declares a {} byte push constant block, "
                                   "the host passes {} bytes",
                                   module, shader->push_constant_size, expected_push_constant_size));
        for (const std::string_view capability : shader->capabilities) {
            LFS_ASSERT_MSG(capability_provided(context_.caps(), capability),
                           std::format("Vulkan backend: module {} declares SPIR-V capability {}, "
                                       "which this device or the backend contract does not provide",
                                       module, capability));
        }
        for (const uint32_t width : shader->float_widths) {
            LFS_ASSERT_MSG(std::ranges::find(shader->signed_zero_inf_nan_preserve, width) !=
                               shader->signed_zero_inf_nan_preserve.end(),
                           std::format("Vulkan backend: module {} computes in {}-bit floats without "
                                       "the SignedZeroInfNanPreserve {} execution mode",
                                       module, width, width));
        }
        const std::span<const uint32_t> words = shader->code;
        VulkanPipeline result;
        result.local_size_x = expected_local_size_x;
        result.push_constant_size = expected_push_constant_size;
        VkShaderModuleCreateInfo shader_info{
            VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        shader_info.codeSize = words.size() * sizeof(uint32_t);
        shader_info.pCode = words.data();
        vk_check(&context_, vkCreateShaderModule(context_.device(), &shader_info, nullptr, &result.shader),
                 "vkCreateShaderModule");
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.size = expected_push_constant_size;
        VkPipelineLayoutCreateInfo layout_info{
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout_info.pushConstantRangeCount = 1;
        layout_info.pPushConstantRanges = &push_range;
        vk_check(&context_, vkCreatePipelineLayout(context_.device(), &layout_info, nullptr, &result.layout),
                 "vkCreatePipelineLayout");
        VkPipelineShaderStageCreateInfo stage_info{
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage_info.module = result.shader;
        stage_info.pName = "main";
        std::vector<VkSpecializationMapEntry> entries(constants.size());
        for (size_t index = 0; index < entries.size(); ++index) {
            entries[index] = VkSpecializationMapEntry{
                .constantID = static_cast<uint32_t>(index),
                .offset = static_cast<uint32_t>(index * sizeof(uint32_t)),
                .size = sizeof(uint32_t),
            };
        }
        VkSpecializationInfo specialization{
            .mapEntryCount = static_cast<uint32_t>(entries.size()),
            .pMapEntries = entries.data(),
            .dataSize = constants.size_bytes(),
            .pData = constants.data(),
        };
        if (!constants.empty()) {
            stage_info.pSpecializationInfo = &specialization;
        }
        VkComputePipelineCreateInfo pipeline_info{
            VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipeline_info.stage = stage_info;
        pipeline_info.layout = result.layout;
        vk_check(&context_, vkCreateComputePipelines(context_.device(), context_.pipeline_cache(), 1, &pipeline_info, nullptr, &result.pipeline),
                 "vkCreateComputePipelines");
        return result;
    }

    void VulkanPipelines::shutdown() {
        std::lock_guard lock(mutex_);
        if (shutting_down_) {
            return;
        }
        shutting_down_ = true;
        for (auto& [name, pipeline] : pipelines_) {
            (void)name;
            if (pipeline.pipeline != VK_NULL_HANDLE) {
                vkDestroyPipeline(context_.device(), pipeline.pipeline, nullptr);
            }
            if (pipeline.layout != VK_NULL_HANDLE) {
                vkDestroyPipelineLayout(context_.device(), pipeline.layout, nullptr);
            }
            if (pipeline.shader != VK_NULL_HANDLE) {
                vkDestroyShaderModule(context_.device(), pipeline.shader, nullptr);
            }
        }
        pipelines_.clear();
    }

} // namespace lfs::core::internal
