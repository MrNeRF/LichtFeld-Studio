/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../../internal/expression_emitter.hpp"
#include "../facade_trace.hpp"
#include "spirv_module.hpp"
#include "vk_backend_ops.hpp"
#include "vk_ops_common.hpp"
#include "vk_pipelines.hpp"
#include "vk_recorder.hpp"

#include <cstring>
#include <optional>
#include <stdexcept>

namespace lfs::core::internal {
    namespace {
        struct VulkanKernel final : CompiledExpression {
            VkDevice device;
            VulkanPipeline pipeline;
            uint32_t push_bytes = 0;
            explicit VulkanKernel(VkDevice owner) : device(owner) {}
            ~VulkanKernel() override {
                vkDestroyPipeline(device, pipeline.pipeline, nullptr);
                vkDestroyPipelineLayout(device, pipeline.layout, nullptr);
                vkDestroyShaderModule(device, pipeline.shader, nullptr);
            }
        };

        std::shared_ptr<CompiledExpression> load_kernel(VulkanContext& context,
                                                        const std::span<const char> artifact) {
            if (artifact.size() < 20 || artifact.size() % 4 != 0)
                throw std::runtime_error("Invalid cached expression SPIR-V");
            std::vector<uint32_t> words(artifact.size() / 4);
            std::memcpy(words.data(), artifact.data(), artifact.size());
            const auto facts = spirv::analyze(spirv::parse(words));
            if (facts.entry_name != "main" || facts.local_size != std::array<uint32_t, 3>{256, 1, 1} ||
                facts.push_constant_size == 0 || facts.push_constant_size > 128 || facts.push_constant_size % 4 != 0 ||
                facts.preserve_widths != facts.float_widths)
                throw std::runtime_error("Cached expression SPIR-V violates the tensor shader contract");
            for (const auto capability : facts.capabilities) {
                if (capability != spv::CapabilityShader && capability != spv::CapabilityInt64 &&
                    capability != spv::CapabilityPhysicalStorageBufferAddresses &&
                    capability != spv::CapabilitySignedZeroInfNanPreserve)
                    throw std::runtime_error(std::string("Vulkan expression requires an unsupported SPIR-V capability: ") +
                                             spv::CapabilityToString(capability));
            }
            if (facts.float_widths.contains(64) || facts.float_widths.contains(16))
                throw std::runtime_error("Vulkan expressions require float32 arithmetic");
            auto kernel = std::make_shared<VulkanKernel>(context.device());
            kernel->push_bytes = facts.push_constant_size;
            auto& pipeline = kernel->pipeline;
            VkShaderModuleCreateInfo shader{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            shader.codeSize = artifact.size();
            shader.pCode = words.data();
            vk_check(&context, vkCreateShaderModule(context.device(), &shader, nullptr, &pipeline.shader), "expression shader module");
            VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, kernel->push_bytes};
            VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            layout.pushConstantRangeCount = 1;
            layout.pPushConstantRanges = &range;
            vk_check(&context, vkCreatePipelineLayout(context.device(), &layout, nullptr, &pipeline.layout), "expression pipeline layout");
            VkComputePipelineCreateInfo create{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            create.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            create.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            create.stage.module = pipeline.shader;
            create.stage.pName = "main";
            create.layout = pipeline.layout;
            vk_check(&context, vkCreateComputePipelines(context.device(), VK_NULL_HANDLE, 1, &create, nullptr, &pipeline.pipeline), "expression pipeline");
            return kernel;
        }
        std::shared_ptr<VulkanKernel> compiled_kernel(const std::shared_ptr<VulkanContext>& context,
                                                      const ExpressionProgram& program,
                                                      const ExpressionSignature& signature) {
            std::string key = "vulkan-spirv:";
            const auto& caps = context->caps();
            key.append(reinterpret_cast<const char*>(caps.device_uuid.data()), caps.device_uuid.size());
            key.append(reinterpret_cast<const char*>(caps.driver_uuid.data()), caps.driver_uuid.size());
            key += expression_key(program, signature);
            return std::static_pointer_cast<VulkanKernel>(context->pipelines().expressions().get(key, [&] { const auto words = expression_spirv(program, signature);
                const auto* first = reinterpret_cast<const char*>(words.data());
                return std::vector<char>(first, first + words.size() * 4); }, [&](const std::span<const char> artifact) { return load_kernel(*context, artifact); }));
        }
    } // namespace

    void VulkanBackendOps::compiled_expression(const ExpressionLaunch& launch, ExecContext) {
        LFS_FACADE_TRACE(compiled_expression);
        const auto context = acquire_vulkan_context();
        const auto kernel = compiled_kernel(context, *launch.program, launch.signature);
        if (launch.prepare_only)
            return;
        const uint32_t bytes = launch.words * 4;
        const bool direct = bytes <= 128;
        std::array<StorageRef, ExpressionProgram::max_outputs + 1> writes{};
        std::copy_n(launch.writes.begin(), launch.write_count, writes.begin());
        std::optional<vk::ScopedAllocation> arguments;
        if (!direct) {
            arguments.emplace(*context, bytes);
            writes[launch.write_count] = arguments->storage();
        }
        const uint64_t address = direct ? 0 : vk::address(arguments->storage());
        const uint64_t work = (uint64_t(launch.count) + expression_packing(launch.signature) - 1) / expression_packing(launch.signature);
        context->recorders().record(std::span(launch.reads.data(), launch.read_count), std::span(writes.data(), launch.write_count + !direct), [&](VkCommandBuffer command) {
                if (!direct) {
                    const auto storage = arguments->storage();
                    vkCmdUpdateBuffer(command, VulkanMemory::buffer_for(storage), VulkanMemory::offset_for(storage), bytes, launch.arguments.data());
                    const VkMemoryBarrier2 written{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                                                   .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                                   .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                                   .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                                   .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT};
                    const VkDependencyInfo dependency{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &written};
                    vkCmdPipelineBarrier2(command, &dependency);
                }
                vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, kernel->pipeline.pipeline);
                vkCmdPushConstants(command, kernel->pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, kernel->push_bytes,
                                   direct ? static_cast<const void*>(launch.arguments.data()) : &address);
                vkCmdDispatch(command, vk::dispatch_groups(*context, work), 1, 1); }, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | (direct ? 0 : VK_PIPELINE_STAGE_2_TRANSFER_BIT), VK_WHOLE_SIZE, kernel);
    }
} // namespace lfs::core::internal
