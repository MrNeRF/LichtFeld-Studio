/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vk_pipelines.hpp"

#include "core/assert.hpp"
#include "vk_context.hpp"

#include <filesystem>
#include <format>
#include <fstream>
#include <nlohmann/json.hpp>
#include <vector>

namespace lfs::core::internal {
    namespace {
        std::vector<uint32_t> read_spirv(const std::filesystem::path& path) {
            std::ifstream stream(path, std::ios::binary | std::ios::ate);
            LFS_ASSERT_MSG(stream.good(),
                           std::string("Vulkan backend: cannot open SPIR-V module ") +
                               path.string());
            const std::streamoff byte_count = stream.tellg();
            LFS_ASSERT_MSG(byte_count > 0 && byte_count % 4 == 0,
                           "Vulkan backend: SPIR-V module has invalid byte size");
            std::vector<uint32_t> words(static_cast<size_t>(byte_count) / 4);
            stream.seekg(0);
            stream.read(reinterpret_cast<char*>(words.data()), byte_count);
            LFS_ASSERT_MSG(stream.good() && words.front() == 0x07230203u,
                           "Vulkan backend: SPIR-V module is invalid");
            return words;
        }

        nlohmann::json read_manifest(const std::filesystem::path& path) {
            std::ifstream stream(path);
            LFS_ASSERT_MSG(stream.good(),
                           std::string("Vulkan backend: cannot open shader manifest ") +
                               path.string());
            return nlohmann::json::parse(stream);
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
        const std::filesystem::path directory(LFS_TENSOR_SPV_DIR);
        const nlohmann::json manifest =
            read_manifest(directory / (module + ".json"));
        LFS_ASSERT_MSG(manifest.at("entry_point") == "main",
                       "Vulkan backend: shader manifest entry point mismatch");
        LFS_ASSERT_MSG(manifest.at("local_size").at(0).get<uint32_t>() ==
                               expected_local_size_x &&
                           manifest.at("local_size").at(1).get<uint32_t>() == 1 &&
                           manifest.at("local_size").at(2).get<uint32_t>() == 1,
                       "Vulkan backend: shader manifest local size mismatch");
        LFS_ASSERT_MSG(manifest.at("push_constant_size").get<uint32_t>() ==
                           expected_push_constant_size,
                       "Vulkan backend: shader manifest push constant size mismatch");
        const auto capabilities =
            manifest.at("required_capabilities").get<std::vector<std::string>>();
        std::vector<std::string> expected_capabilities{
            "Int64", "PhysicalStorageBufferAddresses"};
        if (module == "pointwise_half") {
            expected_capabilities.emplace_back("Float16");
            LFS_ASSERT_MSG(context_.caps().shader_float16,
                           "Vulkan native Float16 pipeline requires shaderFloat16");
        }
        LFS_ASSERT_MSG(capabilities == expected_capabilities,
                       "Vulkan backend: shader manifest capabilities mismatch");

        const std::vector<uint32_t> words =
            read_spirv(directory / (module + ".spv"));
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
