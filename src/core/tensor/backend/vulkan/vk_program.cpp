/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../gpu_program.hpp"
#include "vk_context.hpp"
#include "vk_memory.hpp"
#include "vk_recorder.hpp"
#include <cstring>
#include <format>
#include <map>
#include <mutex>
#include <tuple>

namespace lfs::core::internal {
    namespace {
        using Module = GpuKernelModule;

        struct Pipeline {
            VkDevice device;
            VkPipeline pipeline = VK_NULL_HANDLE;
            VkPipelineLayout layout = VK_NULL_HANDLE;
            VkRenderPass pass = VK_NULL_HANDLE;
            ~Pipeline() {
                if (pipeline)
                    vkDestroyPipeline(device, pipeline, nullptr);
                if (layout)
                    vkDestroyPipelineLayout(device, layout, nullptr);
                if (pass)
                    vkDestroyRenderPass(device, pass, nullptr);
            }
        };
        struct Attachment {
            VkDevice device;
            VmaAllocator allocator;
            VkImage image = VK_NULL_HANDLE;
            VmaAllocation allocation = VK_NULL_HANDLE;
            VkImageView view = VK_NULL_HANDLE;
            VkImageAspectFlags aspect;
            ~Attachment() {
                if (view)
                    vkDestroyImageView(device, view, nullptr);
                if (image)
                    vmaDestroyImage(allocator, image, allocation);
            }
        };
        struct RasterResources {
            VkDevice device;
            VkFramebuffer framebuffer = VK_NULL_HANDLE;
            std::shared_ptr<Attachment> color, depth;
            ~RasterResources() {
                if (framebuffer)
                    vkDestroyFramebuffer(device, framebuffer, nullptr);
            }
        };

        void transition(VkCommandBuffer command, const Attachment& image, VkImageLayout from, VkImageLayout to) {
            VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            barrier.srcAccessMask = from == VK_IMAGE_LAYOUT_UNDEFINED ? 0 : VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
            barrier.oldLayout = from;
            barrier.newLayout = to;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = image.image;
            barrier.subresourceRange = {image.aspect, 0, 1, 0, 1};
            vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
        }

        class Program final : public GpuProgram {
        public:
            explicit Program(std::span<const Module::Entry> entries) : context_(acquire_vulkan_context()) {
                for (const auto& entry : entries) {
                    if (entry.backend != GpuBackend::Vulkan)
                        continue;
                    LFS_ASSERT_MSG(entry.code.size() % 4 == 0,
                                   std::format("SPIR-V entry '{}' size {} is not word aligned", entry.name, entry.code.size()));
                    std::vector<uint32_t> words(entry.code.size() / 4);
                    std::memcpy(words.data(), entry.code.data(), entry.code.size());
                    sources_.emplace(std::pair{std::string(entry.name), entry.stage}, std::move(words));
                }
            }
            bool supports_raster() const override {
                uint32_t count = 0;
                vkGetPhysicalDeviceQueueFamilyProperties(context_->physical_device(), &count, nullptr);
                std::vector<VkQueueFamilyProperties> families(count);
                vkGetPhysicalDeviceQueueFamilyProperties(context_->physical_device(), &count, families.data());
                return (families.at(context_->queue_family()).queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0;
            }
            uint64_t address(const Tensor& tensor) override {
                const auto storage = storage_ref(tensor);
                return storage.meta->gpu_descriptor.base_address + storage.byte_offset;
            }
            void dispatch(const Module::Dispatch& launch, const ProgramArguments& arguments) override {
                std::lock_guard lock(mutex_);
                const auto pipeline = compute(launch.function, arguments.parameters.size());
                std::vector<StorageRef> reads, writes;
                accesses(arguments, reads, writes);
                for (size_t i = 0; i < 3; ++i)
                    LFS_ASSERT_MSG(launch.groups[i] <= context_->caps().max_workgroup_count[i],
                                   std::format("Program dispatch dimension {}: {} groups exceeds {}", i, launch.groups[i], context_->caps().max_workgroup_count[i]));
                VkBuffer indirect = VK_NULL_HANDLE;
                VkDeviceSize indirect_offset = 0;
                if (launch.indirect) {
                    const auto storage = storage_ref(*launch.indirect);
                    indirect = VulkanMemory::buffer_for(storage);
                    indirect_offset = VulkanMemory::offset_for(storage) + launch.indirect_offset * sizeof(uint32_t);
                }
                // Indirect counts are read in the draw-indirect stage.
                const VkPipelineStageFlags2 stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                                    (indirect ? VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT : 0);
                context_->recorders().record(reads, writes, [&](VkCommandBuffer command) {
                    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline);
                    push(command, *pipeline, VK_SHADER_STAGE_COMPUTE_BIT, arguments);
                    if (indirect)
                        vkCmdDispatchIndirect(command, indirect, indirect_offset);
                    else
                        vkCmdDispatch(command, launch.groups[0], launch.groups[1], launch.groups[2]); }, stage, VK_WHOLE_SIZE, pipeline);
            }

            void draw(std::span<const Module::Draw> draws, std::span<const ProgramArguments> arguments) override {
                std::lock_guard lock(mutex_);
                const auto& first = draws.front();
                const uint32_t width = static_cast<uint32_t>(first.color->size(1)), height = static_cast<uint32_t>(first.color->size(0));
                const auto format = first.color->dtype() == DataType::UInt8 ? VK_FORMAT_R8G8B8A8_UNORM : VK_FORMAT_R32G32B32A32_SFLOAT;
                std::vector<std::shared_ptr<Pipeline>> pipelines;
                pipelines.reserve(draws.size());
                for (size_t i = 0; i < draws.size(); ++i)
                    pipelines.push_back(raster(draws[i], format, arguments[i].parameters.size()));
                // Every pipeline in a batch has the same attachment formats, so
                // their render passes are compatible with one framebuffer.
                auto resources = target(width, height, format, first.depth != nullptr, pipelines.front()->pass);
                std::vector<StorageRef> reads, writes;
                for (const auto& argument : arguments)
                    accesses(argument, reads, writes);
                writes.push_back(storage_ref(*first.color));
                if (first.depth)
                    writes.push_back(storage_ref(*first.depth));
                context_->recorders().record(reads, writes, [&](VkCommandBuffer command) {
                    const auto upload = [&](const Attachment& image, const Tensor& tensor, bool clear) {
                        transition(command, image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
                        if (!clear) {
                            auto copy = region(tensor, image.aspect, width, height);
                            vkCmdCopyBufferToImage(command, VulkanMemory::buffer_for(storage_ref(tensor)), image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
                        }
                        transition(command, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                   image.aspect == VK_IMAGE_ASPECT_COLOR_BIT ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
                    };
                    upload(*resources->color, *first.color, first.clear_color);
                    if (first.depth) upload(*resources->depth, *first.depth, first.clear_depth);
                    VkRenderPassBeginInfo begin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
                    begin.renderPass = pipelines.front()->pass;
                    begin.framebuffer = resources->framebuffer;
                    begin.renderArea.extent = {width, height};
                    vkCmdBeginRenderPass(command, &begin, VK_SUBPASS_CONTENTS_INLINE);
                    VkClearAttachment clear[2]{};
                    uint32_t clear_count = 0;
                    if (first.clear_color) {
                        clear[clear_count].aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                        std::copy(first.color_clear.begin(), first.color_clear.end(), clear[clear_count++].clearValue.color.float32);
                    }
                    if (first.depth && first.clear_depth) {
                        clear[clear_count].aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
                        clear[clear_count++].clearValue.depthStencil = {first.depth_clear, 0};
                    }
                    const VkClearRect rect{{{0, 0}, {width, height}}, 0, 1};
                    if (clear_count) vkCmdClearAttachments(command, clear_count, clear, 1, &rect);
                    for (size_t i = 0; i < draws.size(); ++i) {
                        const auto& draw = draws[i];
                        if (draw.vertex_count == 0) continue;
                        // Negative height makes NDC +Y point upward, like Metal.
                        const auto rect = draw.viewport.value_or(Module::Viewport{0, 0, float(width), float(height)});
                        const VkViewport viewport{rect.x, rect.y + rect.height, rect.width, -rect.height, 0, 1};
                        vkCmdSetViewport(command, 0, 1, &viewport);
                        const VkRect2D scissor = draw.scissor
                            ? VkRect2D{{static_cast<int32_t>(draw.scissor->x), static_cast<int32_t>(draw.scissor->y)},
                                       {draw.scissor->width, draw.scissor->height}}
                            : VkRect2D{{0, 0}, {width, height}};
                        vkCmdSetScissor(command, 0, 1, &scissor);
                        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines[i]->pipeline);
                        push(command, *pipelines[i], VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, arguments[i]);
                        vkCmdDraw(command, draw.vertex_count, draw.instance_count, draw.first_vertex, 0);
                    }
                    vkCmdEndRenderPass(command);
                    const auto download = [&](const Attachment& image, const Tensor& tensor) {
                        transition(command, image, image.aspect == VK_IMAGE_ASPECT_COLOR_BIT ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
                        auto copy = region(tensor, image.aspect, width, height);
                        vkCmdCopyImageToBuffer(command, image.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VulkanMemory::buffer_for(storage_ref(tensor)), 1, &copy);
                    };
                    download(*resources->color, *first.color);
                    if (first.depth) download(*resources->depth, *first.depth); }, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_WHOLE_SIZE, std::make_shared<std::pair<std::shared_ptr<RasterResources>, std::vector<std::shared_ptr<Pipeline>>>>(resources, pipelines));
            }

        private:
            void check(VkResult result, const char* operation) { vk_check(context_.get(), result, operation); }
            static void accesses(const ProgramArguments& arguments, std::vector<StorageRef>& reads, std::vector<StorageRef>& writes) {
                for (auto* tensor : arguments.reads)
                    reads.push_back(storage_ref(*tensor));
                for (auto* tensor : arguments.writes)
                    writes.push_back(storage_ref(*tensor));
            }
            static VkBufferImageCopy region(const Tensor& tensor, VkImageAspectFlags aspect, uint32_t width, uint32_t height) {
                VkBufferImageCopy copy{};
                copy.bufferOffset = VulkanMemory::offset_for(storage_ref(tensor));
                copy.imageSubresource = {aspect, 0, 0, 1};
                copy.imageExtent = {width, height, 1};
                return copy;
            }
            static void push(VkCommandBuffer command, const Pipeline& pipeline, VkShaderStageFlags stages, const ProgramArguments& arguments) {
                if (!arguments.parameters.empty())
                    vkCmdPushConstants(command, pipeline.layout, stages, 0, static_cast<uint32_t>(arguments.parameters.size()), arguments.parameters.data());
            }
            // A target is reused once the recorder has released it, i.e. after
            // the GPU finished the submission that last used it.
            std::shared_ptr<RasterResources> target(uint32_t width, uint32_t height, VkFormat format, bool depth, VkRenderPass pass) {
                const auto key = std::tuple{width, height, format, depth};
                auto& pool = targets_[key];
                for (const auto& cached : pool)
                    if (cached.use_count() == 1)
                        return cached;
                if (targets_.size() > 8) {
                    std::erase_if(targets_, [&](const auto& entry) { return entry.first != key; });
                }
                auto result = std::make_shared<RasterResources>();
                result->device = context_->device();
                result->color = attachment(width, height, format, VK_IMAGE_ASPECT_COLOR_BIT);
                if (depth)
                    result->depth = attachment(width, height, VK_FORMAT_D32_SFLOAT, VK_IMAGE_ASPECT_DEPTH_BIT);
                const VkImageView views[]{result->color->view, result->depth ? result->depth->view : VK_NULL_HANDLE};
                VkFramebufferCreateInfo framebuffer{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
                framebuffer.renderPass = pass;
                framebuffer.attachmentCount = depth ? 2 : 1;
                framebuffer.pAttachments = views;
                framebuffer.width = width;
                framebuffer.height = height;
                framebuffer.layers = 1;
                check(vkCreateFramebuffer(context_->device(), &framebuffer, nullptr, &result->framebuffer), "Create tensor framebuffer");
                pool.push_back(result);
                return result;
            }
            std::shared_ptr<Attachment> attachment(uint32_t width, uint32_t height, VkFormat format, VkImageAspectFlags aspect) {
                auto result = std::make_shared<Attachment>();
                result->device = context_->device();
                result->allocator = context_->allocator();
                result->aspect = aspect;
                VkImageCreateInfo image{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
                image.imageType = VK_IMAGE_TYPE_2D;
                image.format = format;
                image.extent = {width, height, 1};
                image.mipLevels = image.arrayLayers = 1;
                image.samples = VK_SAMPLE_COUNT_1_BIT;
                image.tiling = VK_IMAGE_TILING_OPTIMAL;
                image.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                              (aspect == VK_IMAGE_ASPECT_COLOR_BIT ? VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT : VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
                VmaAllocationCreateInfo allocation{};
                allocation.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
                check(vmaCreateImage(context_->allocator(), &image, &allocation, &result->image, &result->allocation, nullptr), "Allocate tensor raster attachment");
                VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
                view.image = result->image;
                view.viewType = VK_IMAGE_VIEW_TYPE_2D;
                view.format = format;
                view.subresourceRange = {aspect, 0, 1, 0, 1};
                check(vkCreateImageView(context_->device(), &view, nullptr, &result->view), "Create tensor attachment view");
                return result;
            }
            VkShaderModule shader(std::string_view name, Module::Stage stage) {
                const auto& words = sources_.at({std::string(name), stage});
                VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
                info.codeSize = words.size() * 4;
                info.pCode = words.data();
                VkShaderModule module;
                check(vkCreateShaderModule(context_->device(), &info, nullptr, &module), "Load Slang SPIR-V");
                return module;
            }
            std::shared_ptr<Pipeline> layout(size_t bytes, VkShaderStageFlags stages) {
                auto result = std::make_shared<Pipeline>();
                result->device = context_->device();
                VkPhysicalDeviceProperties properties{};
                vkGetPhysicalDeviceProperties(context_->physical_device(), &properties);
                LFS_ASSERT_MSG(bytes % 4 == 0 && bytes <= properties.limits.maxPushConstantsSize,
                               std::format("Program parameter size {} exceeds/alignment violates device limit {}", bytes, properties.limits.maxPushConstantsSize));
                const VkPushConstantRange range{stages, 0, static_cast<uint32_t>(bytes)};
                VkPipelineLayoutCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
                info.pushConstantRangeCount = bytes ? 1 : 0;
                info.pPushConstantRanges = &range;
                check(vkCreatePipelineLayout(context_->device(), &info, nullptr, &result->layout), "Create tensor program layout");
                return result;
            }
            std::shared_ptr<Pipeline> compute(std::string_view name, size_t bytes) {
                const auto key = std::pair{std::string(name), bytes};
                if (auto found = compute_.find(key); found != compute_.end())
                    return found->second;
                auto result = layout(bytes, VK_SHADER_STAGE_COMPUTE_BIT);
                const auto module = shader(name, Module::Stage::Compute);
                VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
                const std::string entry(name);
                info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, module, entry.c_str(), nullptr};
                info.layout = result->layout;
                const auto status = vkCreateComputePipelines(context_->device(), context_->pipeline_cache(), 1, &info, nullptr, &result->pipeline);
                vkDestroyShaderModule(context_->device(), module, nullptr);
                check(status, "Create tensor compute pipeline");
                compute_.emplace(key, result);
                return result;
            }
            std::shared_ptr<Pipeline> raster(const Module::Draw& draw, VkFormat format, size_t bytes) {
                const auto key = std::tuple{std::string(draw.vertex), std::string(draw.fragment), format, bytes, draw.blend, draw.depth != nullptr, draw.depth_compare, draw.depth_write, draw.cull};
                if (auto found = raster_.find(key); found != raster_.end())
                    return found->second;
                auto result = layout(bytes, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
                VkAttachmentDescription attachments[2]{};
                attachments[0] = {0, format, VK_SAMPLE_COUNT_1_BIT, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE,
                                  VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_DONT_CARE, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
                attachments[1] = {0, VK_FORMAT_D32_SFLOAT, VK_SAMPLE_COUNT_1_BIT, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE,
                                  VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_DONT_CARE, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
                const VkAttachmentReference color{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}, depth{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
                VkSubpassDescription subpass{};
                subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
                subpass.colorAttachmentCount = 1;
                subpass.pColorAttachments = &color;
                subpass.pDepthStencilAttachment = draw.depth ? &depth : nullptr;
                VkRenderPassCreateInfo pass{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
                pass.attachmentCount = draw.depth ? 2 : 1;
                pass.pAttachments = attachments;
                pass.subpassCount = 1;
                pass.pSubpasses = &subpass;
                check(vkCreateRenderPass(context_->device(), &pass, nullptr, &result->pass), "Create tensor render pass");
                const auto vertex = shader(draw.vertex, Module::Stage::Vertex), fragment = shader(draw.fragment, Module::Stage::Fragment);
                const std::string vertex_name(draw.vertex), fragment_name(draw.fragment);
                const VkPipelineShaderStageCreateInfo stages[]{
                    {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vertex, vertex_name.c_str(), nullptr},
                    {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fragment, fragment_name.c_str(), nullptr}};
                VkPipelineVertexInputStateCreateInfo input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
                VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
                assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
                VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
                viewport.viewportCount = viewport.scissorCount = 1;
                VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
                raster.polygonMode = VK_POLYGON_MODE_FILL;
                raster.cullMode = draw.cull == Module::Cull::Back ? VK_CULL_MODE_BACK_BIT
                                  : draw.cull == Module::Cull::Front ? VK_CULL_MODE_FRONT_BIT
                                                                     : VK_CULL_MODE_NONE;
                // Vulkan decides facing in framebuffer space; the negative-height
                // viewport maps NDC +Y up there, so NDC counter-clockwise stays front.
                raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
                raster.lineWidth = 1;
                VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
                multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
                VkPipelineDepthStencilStateCreateInfo depth_state{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
                depth_state.depthTestEnable = draw.depth != nullptr;
                depth_state.depthWriteEnable = draw.depth && draw.depth_write;
                depth_state.depthCompareOp = draw.depth_compare == Module::Compare::Less ? VK_COMPARE_OP_LESS : draw.depth_compare == Module::Compare::LessEqual ? VK_COMPARE_OP_LESS_OR_EQUAL
                                                                                                                                                                 : VK_COMPARE_OP_ALWAYS;
                VkPipelineColorBlendAttachmentState blend{};
                blend.blendEnable = draw.blend != Module::Blend::Opaque;
                blend.srcColorBlendFactor = draw.blend == Module::Blend::StraightAlpha ? VK_BLEND_FACTOR_SRC_ALPHA : VK_BLEND_FACTOR_ONE;
                blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
                blend.colorBlendOp = blend.alphaBlendOp = VK_BLEND_OP_ADD;
                blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
                blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
                blend.colorWriteMask = 15;
                VkPipelineColorBlendStateCreateInfo blending{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
                blending.attachmentCount = 1;
                blending.pAttachments = &blend;
                const VkDynamicState dynamic_states[]{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
                VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
                dynamic.dynamicStateCount = 2;
                dynamic.pDynamicStates = dynamic_states;
                VkGraphicsPipelineCreateInfo info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
                info.stageCount = 2;
                info.pStages = stages;
                info.pVertexInputState = &input;
                info.pInputAssemblyState = &assembly;
                info.pViewportState = &viewport;
                info.pRasterizationState = &raster;
                info.pMultisampleState = &multisample;
                info.pDepthStencilState = &depth_state;
                info.pColorBlendState = &blending;
                info.pDynamicState = &dynamic;
                info.layout = result->layout;
                info.renderPass = result->pass;
                const auto status = vkCreateGraphicsPipelines(context_->device(), context_->pipeline_cache(), 1, &info, nullptr, &result->pipeline);
                vkDestroyShaderModule(context_->device(), vertex, nullptr);
                vkDestroyShaderModule(context_->device(), fragment, nullptr);
                check(status, "Create tensor raster pipeline");
                raster_.emplace(key, result);
                return result;
            }

            std::shared_ptr<VulkanContext> context_;
            std::mutex mutex_;
            std::map<std::pair<std::string, Module::Stage>, std::vector<uint32_t>> sources_;
            std::map<std::pair<std::string, size_t>, std::shared_ptr<Pipeline>> compute_;
            std::map<std::tuple<std::string, std::string, VkFormat, size_t, Module::Blend, bool, Module::Compare, bool, Module::Cull>, std::shared_ptr<Pipeline>> raster_;
            std::map<std::tuple<uint32_t, uint32_t, VkFormat, bool>, std::vector<std::shared_ptr<RasterResources>>> targets_;
        };
    } // namespace

    std::unique_ptr<GpuProgram> make_vulkan_program(std::span<const GpuKernelModule::Entry> entries) {
        return std::make_unique<Program>(entries);
    }
} // namespace lfs::core::internal
