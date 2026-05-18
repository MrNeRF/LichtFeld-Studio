/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vulkan_viewport_pass.hpp"

#include "config.h"
#include "core/logger.hpp"
#include "vulkan_environment_pass.hpp"
#include "vulkan_mesh_pass.hpp"
#include "vulkan_scene_image_uploader.hpp"
#include "window/vulkan_context.hpp"

#ifdef LFS_VULKAN_VIEWER_ENABLED
#include "viewport/grid.frag.spv.h"
#include "viewport/grid.vert.spv.h"
#include "viewport/overlay.frag.spv.h"
#include "viewport/overlay.vert.spv.h"
#include "viewport/pivot.frag.spv.h"
#include "viewport/pivot.vert.spv.h"
#include "viewport/scene.frag.spv.h"
#include "viewport/screen_quad.vert.spv.h"
#include "viewport/shape_overlay.frag.spv.h"
#include "viewport/shape_overlay.vert.spv.h"
#include "viewport/textured_overlay.frag.spv.h"
#include "viewport/textured_overlay.vert.spv.h"
#include "viewport/vignette.frag.spv.h"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <span>
#include <vector>

namespace lfs::vis {

#ifdef LFS_VULKAN_VIEWER_ENABLED
    namespace {
        struct Vertex {
            glm::vec2 position;
            glm::vec2 uv;
        };

        struct FramebufferRect {
            std::int32_t x = 0;
            std::int32_t y = 0;
            std::uint32_t width = 0;
            std::uint32_t height = 0;
        };

        struct VignettePush {
            glm::vec4 viewport_intensity_radius{0.0f};
            glm::vec4 softness_padding{0.0f};
        };

        struct GridUniform {
            glm::mat4 view_projection{1.0f};
            glm::vec4 view_position_plane{0.0f};
            glm::vec4 opacity_padding{0.0f};
            glm::vec4 near_origin{0.0f};
            glm::vec4 near_x{0.0f};
            glm::vec4 near_y{0.0f};
            glm::vec4 far_origin{0.0f};
            glm::vec4 far_x{0.0f};
            glm::vec4 far_y{0.0f};
        };

        struct GridPush {
            std::int32_t grid_index = 0;
        };

        struct OverlayPush {
            glm::vec4 padding{0.0f};
        };

        struct PivotPush {
            glm::vec4 center_size{0.0f};
            glm::vec4 color_opacity{0.26f, 0.59f, 0.98f, 1.0f};
        };

        struct TexturedOverlayPush {
            glm::vec4 tint_opacity{1.0f, 1.0f, 1.0f, 0.8f};
            glm::vec4 effects{0.0f};
        };

        [[nodiscard]] VkDescriptorSet descriptorSetFromId(const std::uintptr_t texture_id) {
            return reinterpret_cast<VkDescriptorSet>(texture_id);
        }

        [[nodiscard]] FramebufferRect toFramebufferRect(
            const VulkanViewportPassParams& params,
            const VkExtent2D extent) {
            const float sx = params.framebuffer_scale.x > 0.0f ? params.framebuffer_scale.x : 1.0f;
            const float sy = params.framebuffer_scale.y > 0.0f ? params.framebuffer_scale.y : 1.0f;
            const int x0 = std::clamp(static_cast<int>(std::lround(params.viewport_pos.x * sx)),
                                      0, static_cast<int>(extent.width));
            const int y0 = std::clamp(static_cast<int>(std::lround(params.viewport_pos.y * sy)),
                                      0, static_cast<int>(extent.height));
            const int x1 = std::clamp(static_cast<int>(std::lround((params.viewport_pos.x + params.viewport_size.x) * sx)),
                                      0, static_cast<int>(extent.width));
            const int y1 = std::clamp(static_cast<int>(std::lround((params.viewport_pos.y + params.viewport_size.y) * sy)),
                                      0, static_cast<int>(extent.height));
            return {
                .x = x0,
                .y = y0,
                .width = static_cast<std::uint32_t>(std::max(x1 - x0, 0)),
                .height = static_cast<std::uint32_t>(std::max(y1 - y0, 0)),
            };
        }

        [[nodiscard]] FramebufferRect toFramebufferRect(
            const VulkanViewportPassParams& params,
            const VulkanViewportGridOverlay& grid,
            const VkExtent2D extent) {
            const float sx = params.framebuffer_scale.x > 0.0f ? params.framebuffer_scale.x : 1.0f;
            const float sy = params.framebuffer_scale.y > 0.0f ? params.framebuffer_scale.y : 1.0f;
            const int x0 = std::clamp(static_cast<int>(std::lround(grid.viewport_pos.x * sx)),
                                      0, static_cast<int>(extent.width));
            const int y0 = std::clamp(static_cast<int>(std::lround(grid.viewport_pos.y * sy)),
                                      0, static_cast<int>(extent.height));
            const int x1 = std::clamp(static_cast<int>(std::lround((grid.viewport_pos.x + grid.viewport_size.x) * sx)),
                                      0, static_cast<int>(extent.width));
            const int y1 = std::clamp(static_cast<int>(std::lround((grid.viewport_pos.y + grid.viewport_size.y) * sy)),
                                      0, static_cast<int>(extent.height));
            return {
                .x = x0,
                .y = y0,
                .width = static_cast<std::uint32_t>(std::max(x1 - x0, 0)),
                .height = static_cast<std::uint32_t>(std::max(y1 - y0, 0)),
            };
        }
    } // namespace
#endif

    struct VulkanViewportPass::Impl {
#ifdef LFS_VULKAN_VIEWER_ENABLED
        VkDevice device = VK_NULL_HANDLE;
        VulkanContext* context = nullptr;
        VmaAllocator allocator = VK_NULL_HANDLE;
        VkPipelineCache pipeline_cache = VK_NULL_HANDLE;
        VkQueue graphics_queue = VK_NULL_HANDLE;
        std::uint32_t graphics_queue_family = 0;
        VkFormat color_format = VK_FORMAT_UNDEFINED;
        VkFormat depth_stencil_format = VK_FORMAT_UNDEFINED;
        std::size_t frames_in_flight = 1;

        VkBuffer quad_buffer = VK_NULL_HANDLE;
        VmaAllocation quad_allocation = VK_NULL_HANDLE;
        bool quad_flip_y = false;
        bool quad_initialized = false;

        struct DynamicBuffer {
            VkBuffer buffer = VK_NULL_HANDLE;
            VmaAllocation allocation = VK_NULL_HANDLE;
            std::size_t capacity = 0;
            std::uint32_t count = 0;
        };

        struct FrameResources {
            DynamicBuffer overlay;
            DynamicBuffer shape_overlay;
            DynamicBuffer ui_shape_overlay;
            DynamicBuffer textured_overlay;
            DynamicBuffer grid_uniform;
            VkDescriptorSet scene_descriptor_set = VK_NULL_HANDLE;
            VkDescriptorSet grid_descriptor_set = VK_NULL_HANDLE;
        };
        std::vector<FrameResources> frame_resources;

        VkSampler scene_sampler = VK_NULL_HANDLE;
        VkDescriptorSetLayout scene_descriptor_layout = VK_NULL_HANDLE;
        VkDescriptorPool scene_descriptor_pool = VK_NULL_HANDLE;
        VulkanSceneImageUploader scene_image_uploader;
        VulkanMeshPass mesh_pass;
        VulkanEnvironmentPass environment_pass;
        VulkanDepthBlitPass depth_blit_pass;
        VulkanSplitViewPass split_view_pass;

        VkDescriptorSetLayout grid_descriptor_layout = VK_NULL_HANDLE;
        VkDescriptorPool grid_descriptor_pool = VK_NULL_HANDLE;

        VkPipelineLayout scene_pipeline_layout = VK_NULL_HANDLE;
        VkPipeline scene_pipeline = VK_NULL_HANDLE;
        VkPipelineLayout vignette_pipeline_layout = VK_NULL_HANDLE;
        VkPipeline vignette_pipeline = VK_NULL_HANDLE;
        VkPipelineLayout grid_pipeline_layout = VK_NULL_HANDLE;
        VkPipeline grid_pipeline = VK_NULL_HANDLE;
        VkPipelineLayout overlay_pipeline_layout = VK_NULL_HANDLE;
        VkPipeline overlay_pipeline = VK_NULL_HANDLE;
        VkPipelineLayout shape_overlay_pipeline_layout = VK_NULL_HANDLE;
        VkPipeline shape_overlay_pipeline = VK_NULL_HANDLE;
        VkPipelineLayout textured_overlay_pipeline_layout = VK_NULL_HANDLE;
        VkPipeline textured_overlay_pipeline = VK_NULL_HANDLE;
        VkPipelineLayout pivot_pipeline_layout = VK_NULL_HANDLE;
        VkPipeline pivot_pipeline = VK_NULL_HANDLE;

        [[nodiscard]] bool init(VulkanContext& context) {
            if (device != VK_NULL_HANDLE) {
                return true;
            }
            this->context = &context;
            device = context.device();
            allocator = context.allocator();
            pipeline_cache = context.pipelineCache();
            graphics_queue = context.graphicsQueue();
            graphics_queue_family = context.graphicsQueueFamily();
            color_format = context.swapchainFormat();
            depth_stencil_format = context.depthStencilFormat();
            frames_in_flight = std::max<std::size_t>(1, context.framesInFlight());
            frame_resources.resize(frames_in_flight);
            if (device == VK_NULL_HANDLE || allocator == VK_NULL_HANDLE ||
                graphics_queue == VK_NULL_HANDLE || color_format == VK_FORMAT_UNDEFINED ||
                depth_stencil_format == VK_FORMAT_UNDEFINED) {
                LOG_ERROR("Vulkan viewport pass requires an initialized Vulkan context");
                device = VK_NULL_HANDLE;
                return false;
            }

            if (!createSampler() || !scene_image_uploader.init(context, scene_sampler) ||
                !createSceneDescriptors() || !createGridResources() ||
                !createQuadBuffer() || !createPipelines()) {
                reset();
                return false;
            }
            if (!mesh_pass.init(context, color_format, depth_stencil_format)) {
                LOG_ERROR("Vulkan viewport pass: mesh sub-pass init failed");
                reset();
                return false;
            }
            if (!environment_pass.init(context, color_format, depth_stencil_format, quad_buffer)) {
                LOG_ERROR("Vulkan viewport pass: environment sub-pass init failed");
                reset();
                return false;
            }
            if (!depth_blit_pass.init(context, color_format, depth_stencil_format, quad_buffer)) {
                LOG_ERROR("Vulkan viewport pass: depth-blit sub-pass init failed");
                reset();
                return false;
            }
            if (!split_view_pass.init(context, color_format, depth_stencil_format, quad_buffer)) {
                LOG_ERROR("Vulkan viewport pass: split-view sub-pass init failed");
                reset();
                return false;
            }
            return true;
        }

        [[nodiscard]] bool createBuffer(const VkDeviceSize size,
                                        const VkBufferUsageFlags usage,
                                        VkBuffer& buffer,
                                        VmaAllocation& allocation) const {
            VkBufferCreateInfo buffer_info{};
            buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            buffer_info.size = size;
            buffer_info.usage = usage;
            buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            VmaAllocationCreateInfo allocation_info{};
            allocation_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
            allocation_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

            if (vmaCreateBuffer(allocator, &buffer_info, &allocation_info, &buffer, &allocation, nullptr) != VK_SUCCESS) {
                buffer = VK_NULL_HANDLE;
                allocation = VK_NULL_HANDLE;
                return false;
            }
            return true;
        }

        [[nodiscard]] bool writeAllocation(const VmaAllocation allocation,
                                           const void* const source,
                                           const VkDeviceSize size) const {
            if (allocation == VK_NULL_HANDLE || !source || size == 0) {
                return false;
            }
            void* mapped = nullptr;
            if (vmaMapMemory(allocator, allocation, &mapped) != VK_SUCCESS || !mapped) {
                return false;
            }
            std::memcpy(mapped, source, static_cast<std::size_t>(size));
            vmaFlushAllocation(allocator, allocation, 0, size);
            vmaUnmapMemory(allocator, allocation);
            return true;
        }

        [[nodiscard]] FrameResources& resourcesForFrame(const std::size_t frame_slot) {
            return frame_resources[frame_slot % frame_resources.size()];
        }

        [[nodiscard]] const FrameResources& resourcesForFrame(const std::size_t frame_slot) const {
            return frame_resources[frame_slot % frame_resources.size()];
        }

        void destroyDynamicBuffer(DynamicBuffer& resource) const {
            if (resource.buffer != VK_NULL_HANDLE) {
                vmaDestroyBuffer(allocator, resource.buffer, resource.allocation);
            }
            resource = {};
        }

        [[nodiscard]] bool ensureDynamicBuffer(DynamicBuffer& resource,
                                               const std::size_t element_count,
                                               const std::size_t element_size,
                                               const std::size_t initial_capacity,
                                               const VkBufferUsageFlags usage) const {
            if (element_count == 0) {
                resource.count = 0;
                return true;
            }
            if (resource.buffer != VK_NULL_HANDLE && resource.capacity >= element_count) {
                return true;
            }

            destroyDynamicBuffer(resource);
            std::size_t capacity = initial_capacity;
            while (capacity < element_count) {
                capacity *= 2;
            }
            if (!createBuffer(static_cast<VkDeviceSize>(element_size * capacity),
                              usage,
                              resource.buffer,
                              resource.allocation)) {
                resource = {};
                return false;
            }
            resource.capacity = capacity;
            return true;
        }

        [[nodiscard]] bool createSampler() {
            VkSamplerCreateInfo sampler_info{};
            sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            sampler_info.magFilter = VK_FILTER_LINEAR;
            sampler_info.minFilter = VK_FILTER_LINEAR;
            sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
            sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sampler_info.maxLod = 1.0f;
            return vkCreateSampler(device, &sampler_info, nullptr, &scene_sampler) == VK_SUCCESS;
        }

        [[nodiscard]] bool createSceneDescriptors() {
            VkDescriptorSetLayoutBinding binding{};
            binding.binding = 0;
            binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            binding.descriptorCount = 1;
            binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            VkDescriptorSetLayoutCreateInfo layout_info{};
            layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            layout_info.bindingCount = 1;
            layout_info.pBindings = &binding;
            if (vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &scene_descriptor_layout) != VK_SUCCESS) {
                return false;
            }

            const auto descriptor_count = static_cast<std::uint32_t>(frame_resources.size());
            VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, descriptor_count};
            VkDescriptorPoolCreateInfo pool_info{};
            pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            pool_info.maxSets = descriptor_count;
            pool_info.poolSizeCount = 1;
            pool_info.pPoolSizes = &pool_size;
            if (vkCreateDescriptorPool(device, &pool_info, nullptr, &scene_descriptor_pool) != VK_SUCCESS) {
                return false;
            }

            std::vector<VkDescriptorSetLayout> layouts(frame_resources.size(), scene_descriptor_layout);
            std::vector<VkDescriptorSet> sets(frame_resources.size(), VK_NULL_HANDLE);
            VkDescriptorSetAllocateInfo alloc_info{};
            alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            alloc_info.descriptorPool = scene_descriptor_pool;
            alloc_info.descriptorSetCount = descriptor_count;
            alloc_info.pSetLayouts = layouts.data();
            if (vkAllocateDescriptorSets(device, &alloc_info, sets.data()) != VK_SUCCESS) {
                return false;
            }
            for (std::size_t i = 0; i < frame_resources.size(); ++i) {
                frame_resources[i].scene_descriptor_set = sets[i];
            }
            return true;
        }

        [[nodiscard]] bool createGridResources() {
            VkDescriptorSetLayoutBinding binding{};
            binding.binding = 0;
            binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            binding.descriptorCount = 1;
            binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
            VkDescriptorSetLayoutCreateInfo layout_info{};
            layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            layout_info.bindingCount = 1;
            layout_info.pBindings = &binding;
            if (vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &grid_descriptor_layout) != VK_SUCCESS) {
                return false;
            }

            const auto descriptor_count = static_cast<std::uint32_t>(frame_resources.size());
            VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptor_count};
            VkDescriptorPoolCreateInfo pool_info{};
            pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            pool_info.maxSets = descriptor_count;
            pool_info.poolSizeCount = 1;
            pool_info.pPoolSizes = &pool_size;
            if (vkCreateDescriptorPool(device, &pool_info, nullptr, &grid_descriptor_pool) != VK_SUCCESS) {
                return false;
            }

            std::vector<VkDescriptorSetLayout> layouts(frame_resources.size(), grid_descriptor_layout);
            std::vector<VkDescriptorSet> sets(frame_resources.size(), VK_NULL_HANDLE);
            VkDescriptorSetAllocateInfo alloc_info{};
            alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            alloc_info.descriptorPool = grid_descriptor_pool;
            alloc_info.descriptorSetCount = descriptor_count;
            alloc_info.pSetLayouts = layouts.data();
            if (vkAllocateDescriptorSets(device, &alloc_info, sets.data()) != VK_SUCCESS) {
                return false;
            }
            for (std::size_t i = 0; i < frame_resources.size(); ++i) {
                frame_resources[i].grid_descriptor_set = sets[i];
            }

            return true;
        }

        [[nodiscard]] bool createQuadBuffer() {
            return createBuffer(sizeof(Vertex) * 6,
                                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                quad_buffer,
                                quad_allocation);
        }

        [[nodiscard]] VkShaderModule createShaderModule(const std::span<const std::uint32_t> spirv) const {
            VkShaderModuleCreateInfo create_info{};
            create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            create_info.codeSize = spirv.size() * sizeof(std::uint32_t);
            create_info.pCode = spirv.data();
            VkShaderModule module = VK_NULL_HANDLE;
            if (vkCreateShaderModule(device, &create_info, nullptr, &module) != VK_SUCCESS) {
                return VK_NULL_HANDLE;
            }
            return module;
        }

        enum class PipelineVertexLayout {
            ScreenQuad,
            ColorOverlay,
            TexturedOverlay,
            ShapeOverlay
        };

        [[nodiscard]] bool createPipeline(const std::span<const std::uint32_t> vertex_spv,
                                          const std::span<const std::uint32_t> fragment_spv,
                                          const char* label,
                                          VkDescriptorSetLayout descriptor_layout,
                                          const VkPushConstantRange* push_constant,
                                          bool enable_blend,
                                          PipelineVertexLayout vertex_layout,
                                          VkPipelineLayout& pipeline_layout,
                                          VkPipeline& pipeline) {
            VkShaderModule vertex_module = createShaderModule(vertex_spv);
            VkShaderModule fragment_module = createShaderModule(fragment_spv);
            if (vertex_module == VK_NULL_HANDLE || fragment_module == VK_NULL_HANDLE) {
                if (vertex_module != VK_NULL_HANDLE)
                    vkDestroyShaderModule(device, vertex_module, nullptr);
                if (fragment_module != VK_NULL_HANDLE)
                    vkDestroyShaderModule(device, fragment_module, nullptr);
                LOG_ERROR("Failed to create Vulkan viewport shader modules for {}", label);
                return false;
            }

            VkPipelineShaderStageCreateInfo stages[2]{};
            stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
            stages[0].module = vertex_module;
            stages[0].pName = "main";
            stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
            stages[1].module = fragment_module;
            stages[1].pName = "main";

            VkVertexInputBindingDescription binding{};
            binding.binding = 0;
            binding.stride = sizeof(Vertex);
            if (vertex_layout == PipelineVertexLayout::ColorOverlay) {
                binding.stride = sizeof(VulkanViewportOverlayVertex);
            } else if (vertex_layout == PipelineVertexLayout::TexturedOverlay) {
                binding.stride = sizeof(VulkanViewportTexturedOverlayVertex);
            } else if (vertex_layout == PipelineVertexLayout::ShapeOverlay) {
                binding.stride = sizeof(VulkanViewportShapeOverlayVertex);
            }
            binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
            std::array<VkVertexInputAttributeDescription, 6> attributes{};
            attributes[0].location = 0;
            attributes[0].binding = 0;
            attributes[0].format = VK_FORMAT_R32G32_SFLOAT;
            if (vertex_layout == PipelineVertexLayout::ColorOverlay) {
                attributes[0].offset = offsetof(VulkanViewportOverlayVertex, position);
            } else if (vertex_layout == PipelineVertexLayout::TexturedOverlay) {
                attributes[0].offset = offsetof(VulkanViewportTexturedOverlayVertex, position);
            } else if (vertex_layout == PipelineVertexLayout::ShapeOverlay) {
                attributes[0].offset = offsetof(VulkanViewportShapeOverlayVertex, position);
            } else {
                attributes[0].offset = offsetof(Vertex, position);
            }
            attributes[1].location = 1;
            attributes[1].binding = 0;
            attributes[1].format = vertex_layout == PipelineVertexLayout::ColorOverlay
                                       ? VK_FORMAT_R32G32B32A32_SFLOAT
                                       : VK_FORMAT_R32G32_SFLOAT;
            if (vertex_layout == PipelineVertexLayout::ColorOverlay) {
                attributes[1].offset = offsetof(VulkanViewportOverlayVertex, color);
            } else if (vertex_layout == PipelineVertexLayout::TexturedOverlay) {
                attributes[1].offset = offsetof(VulkanViewportTexturedOverlayVertex, uv);
            } else if (vertex_layout == PipelineVertexLayout::ShapeOverlay) {
                attributes[1].offset = offsetof(VulkanViewportShapeOverlayVertex, screen_position);
            } else {
                attributes[1].offset = offsetof(Vertex, uv);
            }
            if (vertex_layout == PipelineVertexLayout::ShapeOverlay) {
                attributes[2].location = 2;
                attributes[2].binding = 0;
                attributes[2].format = VK_FORMAT_R32G32_SFLOAT;
                attributes[2].offset = offsetof(VulkanViewportShapeOverlayVertex, p0);
                attributes[3].location = 3;
                attributes[3].binding = 0;
                attributes[3].format = VK_FORMAT_R32G32_SFLOAT;
                attributes[3].offset = offsetof(VulkanViewportShapeOverlayVertex, p1);
                attributes[4].location = 4;
                attributes[4].binding = 0;
                attributes[4].format = VK_FORMAT_R32G32B32A32_SFLOAT;
                attributes[4].offset = offsetof(VulkanViewportShapeOverlayVertex, color);
                attributes[5].location = 5;
                attributes[5].binding = 0;
                attributes[5].format = VK_FORMAT_R32G32B32A32_SFLOAT;
                attributes[5].offset = offsetof(VulkanViewportShapeOverlayVertex, params);
            }

            VkPipelineVertexInputStateCreateInfo vertex_input{};
            vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
            vertex_input.vertexBindingDescriptionCount = 1;
            vertex_input.pVertexBindingDescriptions = &binding;
            vertex_input.vertexAttributeDescriptionCount =
                vertex_layout == PipelineVertexLayout::ShapeOverlay ? 6u : 2u;
            vertex_input.pVertexAttributeDescriptions = attributes.data();

            VkPipelineInputAssemblyStateCreateInfo input_assembly{};
            input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

            VkPipelineViewportStateCreateInfo viewport_state{};
            viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
            viewport_state.viewportCount = 1;
            viewport_state.scissorCount = 1;

            VkPipelineRasterizationStateCreateInfo raster{};
            raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
            raster.polygonMode = VK_POLYGON_MODE_FILL;
            raster.cullMode = VK_CULL_MODE_NONE;
            raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
            raster.lineWidth = 1.0f;

            VkPipelineMultisampleStateCreateInfo multisample{};
            multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
            multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

            VkPipelineDepthStencilStateCreateInfo depth{};
            depth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
            depth.depthTestEnable = VK_FALSE;
            depth.depthWriteEnable = VK_FALSE;
            depth.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

            VkPipelineColorBlendAttachmentState blend_attachment{};
            blend_attachment.colorWriteMask =
                VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            blend_attachment.blendEnable = enable_blend ? VK_TRUE : VK_FALSE;
            blend_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
            blend_attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            blend_attachment.colorBlendOp = VK_BLEND_OP_ADD;
            blend_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            blend_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            blend_attachment.alphaBlendOp = VK_BLEND_OP_ADD;

            VkPipelineColorBlendStateCreateInfo blend{};
            blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
            blend.attachmentCount = 1;
            blend.pAttachments = &blend_attachment;

            std::array<VkDynamicState, 2> dynamic_states{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
            VkPipelineDynamicStateCreateInfo dynamic{};
            dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
            dynamic.dynamicStateCount = static_cast<std::uint32_t>(dynamic_states.size());
            dynamic.pDynamicStates = dynamic_states.data();

            VkPipelineLayoutCreateInfo layout_info{};
            layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            if (descriptor_layout != VK_NULL_HANDLE) {
                layout_info.setLayoutCount = 1;
                layout_info.pSetLayouts = &descriptor_layout;
            }
            if (push_constant) {
                layout_info.pushConstantRangeCount = 1;
                layout_info.pPushConstantRanges = push_constant;
            }
            if (vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
                vkDestroyShaderModule(device, vertex_module, nullptr);
                vkDestroyShaderModule(device, fragment_module, nullptr);
                return false;
            }

            VkPipelineRenderingCreateInfo rendering_info{};
            rendering_info.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
            rendering_info.colorAttachmentCount = 1;
            rendering_info.pColorAttachmentFormats = &color_format;
            rendering_info.depthAttachmentFormat = depth_stencil_format;
            rendering_info.stencilAttachmentFormat = depth_stencil_format;

            VkGraphicsPipelineCreateInfo pipeline_info{};
            pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            pipeline_info.pNext = &rendering_info;
            pipeline_info.stageCount = 2;
            pipeline_info.pStages = stages;
            pipeline_info.pVertexInputState = &vertex_input;
            pipeline_info.pInputAssemblyState = &input_assembly;
            pipeline_info.pViewportState = &viewport_state;
            pipeline_info.pRasterizationState = &raster;
            pipeline_info.pMultisampleState = &multisample;
            pipeline_info.pDepthStencilState = &depth;
            pipeline_info.pColorBlendState = &blend;
            pipeline_info.pDynamicState = &dynamic;
            pipeline_info.layout = pipeline_layout;
            pipeline_info.renderPass = VK_NULL_HANDLE;
            pipeline_info.subpass = 0;

            const bool ok =
                vkCreateGraphicsPipelines(device, pipeline_cache, 1, &pipeline_info, nullptr, &pipeline) == VK_SUCCESS;
            vkDestroyShaderModule(device, vertex_module, nullptr);
            vkDestroyShaderModule(device, fragment_module, nullptr);
            return ok;
        }

        [[nodiscard]] bool createPipelines() {
            VkPushConstantRange grid_push{};
            grid_push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
            grid_push.offset = 0;
            grid_push.size = sizeof(GridPush);
            VkPushConstantRange vignette_push{};
            vignette_push.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            vignette_push.offset = 0;
            vignette_push.size = sizeof(VignettePush);
            VkPushConstantRange pivot_push{};
            pivot_push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
            pivot_push.offset = 0;
            pivot_push.size = sizeof(PivotPush);
            VkPushConstantRange textured_overlay_push{};
            textured_overlay_push.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            textured_overlay_push.offset = 0;
            textured_overlay_push.size = sizeof(TexturedOverlayPush);
            using namespace viewport_shaders;

            return createPipeline(kScreenQuadVertSpv, kSceneFragSpv, "scene",
                                  scene_descriptor_layout, nullptr, true, PipelineVertexLayout::ScreenQuad,
                                  scene_pipeline_layout, scene_pipeline) &&
                   createPipeline(kScreenQuadVertSpv, kVignetteFragSpv, "vignette",
                                  VK_NULL_HANDLE, &vignette_push, true, PipelineVertexLayout::ScreenQuad,
                                  vignette_pipeline_layout, vignette_pipeline) &&
                   createPipeline(kGridVertSpv, kGridFragSpv, "grid",
                                  grid_descriptor_layout, &grid_push, true, PipelineVertexLayout::ScreenQuad,
                                  grid_pipeline_layout, grid_pipeline) &&
                   createPipeline(kOverlayVertSpv, kOverlayFragSpv, "overlay",
                                  VK_NULL_HANDLE, nullptr, true, PipelineVertexLayout::ColorOverlay,
                                  overlay_pipeline_layout, overlay_pipeline) &&
                   createPipeline(kShapeOverlayVertSpv, kShapeOverlayFragSpv, "shape_overlay",
                                  VK_NULL_HANDLE, nullptr, true, PipelineVertexLayout::ShapeOverlay,
                                  shape_overlay_pipeline_layout, shape_overlay_pipeline) &&
                   createPipeline(kTexturedOverlayVertSpv, kTexturedOverlayFragSpv, "textured_overlay",
                                  scene_descriptor_layout, &textured_overlay_push, true,
                                  PipelineVertexLayout::TexturedOverlay,
                                  textured_overlay_pipeline_layout, textured_overlay_pipeline) &&
                   createPipeline(kPivotVertSpv, kPivotFragSpv, "pivot",
                                  VK_NULL_HANDLE, &pivot_push, true, PipelineVertexLayout::ScreenQuad,
                                  pivot_pipeline_layout, pivot_pipeline);
        }

        void updateQuadBuffer(const bool flip_y) {
            if (quad_initialized && quad_flip_y == flip_y) {
                return;
            }
            const float top_v = flip_y ? 1.0f : 0.0f;
            const float bottom_v = flip_y ? 0.0f : 1.0f;
            const std::array<Vertex, 6> vertices{{
                {{-1.0f, -1.0f}, {0.0f, top_v}},
                {{1.0f, -1.0f}, {1.0f, top_v}},
                {{1.0f, 1.0f}, {1.0f, bottom_v}},
                {{-1.0f, -1.0f}, {0.0f, top_v}},
                {{1.0f, 1.0f}, {1.0f, bottom_v}},
                {{-1.0f, 1.0f}, {0.0f, bottom_v}},
            }};
            if (writeAllocation(quad_allocation, vertices.data(), sizeof(vertices))) {
                quad_flip_y = flip_y;
                quad_initialized = true;
            }
        }

        void updateOverlayBuffer(FrameResources& frame, const VulkanViewportPassParams& params) {
            frame.overlay.count = 0;
            if (params.overlay_triangles.empty()) {
                return;
            }
            if (!ensureDynamicBuffer(frame.overlay,
                                     params.overlay_triangles.size(),
                                     sizeof(VulkanViewportOverlayVertex),
                                     256,
                                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT)) {
                return;
            }
            const VkDeviceSize bytes =
                static_cast<VkDeviceSize>(sizeof(VulkanViewportOverlayVertex) * params.overlay_triangles.size());
            if (!writeAllocation(frame.overlay.allocation, params.overlay_triangles.data(), bytes)) {
                return;
            }
            frame.overlay.count = static_cast<std::uint32_t>(
                std::min<std::size_t>(params.overlay_triangles.size(), std::numeric_limits<std::uint32_t>::max()));
        }

        void updateShapeOverlayBuffer(const std::vector<VulkanViewportShapeOverlayVertex>& vertices,
                                      DynamicBuffer& resource) {
            resource.count = 0;
            if (vertices.empty()) {
                return;
            }
            if (!ensureDynamicBuffer(resource,
                                     vertices.size(),
                                     sizeof(VulkanViewportShapeOverlayVertex),
                                     256,
                                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT)) {
                return;
            }
            const VkDeviceSize bytes =
                static_cast<VkDeviceSize>(sizeof(VulkanViewportShapeOverlayVertex) * vertices.size());
            if (!writeAllocation(resource.allocation, vertices.data(), bytes)) {
                return;
            }
            resource.count = static_cast<std::uint32_t>(
                std::min<std::size_t>(vertices.size(), std::numeric_limits<std::uint32_t>::max()));
        }

        void updateTexturedOverlayBuffer(FrameResources& frame, const VulkanViewportPassParams& params) {
            frame.textured_overlay.count = 0;
            if (params.textured_overlays.empty()) {
                return;
            }
            const std::size_t vertex_count = params.textured_overlays.size() * 6u;
            if (!ensureDynamicBuffer(frame.textured_overlay,
                                     vertex_count,
                                     sizeof(VulkanViewportTexturedOverlayVertex),
                                     64,
                                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT)) {
                return;
            }

            std::vector<VulkanViewportTexturedOverlayVertex> vertices;
            vertices.reserve(vertex_count);
            for (const auto& overlay : params.textured_overlays) {
                vertices.insert(vertices.end(), overlay.vertices.begin(), overlay.vertices.end());
            }

            const VkDeviceSize bytes =
                static_cast<VkDeviceSize>(sizeof(VulkanViewportTexturedOverlayVertex) * vertices.size());
            if (!writeAllocation(frame.textured_overlay.allocation, vertices.data(), bytes)) {
                return;
            }
            frame.textured_overlay.count = static_cast<std::uint32_t>(
                std::min<std::size_t>(vertices.size(), std::numeric_limits<std::uint32_t>::max()));
        }

        void updateGridDescriptor(FrameResources& frame, const VkDeviceSize range) const {
            if (frame.grid_descriptor_set == VK_NULL_HANDLE || frame.grid_uniform.buffer == VK_NULL_HANDLE) {
                return;
            }
            VkDescriptorBufferInfo buffer_info{};
            buffer_info.buffer = frame.grid_uniform.buffer;
            buffer_info.offset = 0;
            buffer_info.range = range;
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = frame.grid_descriptor_set;
            write.dstBinding = 0;
            write.descriptorCount = 1;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.pBufferInfo = &buffer_info;
            vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
        }

        [[nodiscard]] bool ensureGridUniformBuffer(FrameResources& frame, const std::size_t grid_count) {
            if (grid_count == 0) {
                frame.grid_uniform.count = 0;
                return true;
            }
            if (frame.grid_uniform.buffer != VK_NULL_HANDLE && frame.grid_uniform.capacity >= grid_count) {
                return true;
            }

            std::size_t capacity = 1;
            while (capacity < grid_count) {
                capacity *= 2;
            }
            const VkDeviceSize bytes = static_cast<VkDeviceSize>(sizeof(GridUniform) * capacity);
            destroyDynamicBuffer(frame.grid_uniform);
            if (!createBuffer(bytes,
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                              frame.grid_uniform.buffer,
                              frame.grid_uniform.allocation)) {
                frame.grid_uniform = {};
                return false;
            }
            frame.grid_uniform.capacity = capacity;
            updateGridDescriptor(frame, bytes);
            return true;
        }

        [[nodiscard]] static GridUniform makeGridUniform(const VulkanViewportGridOverlay& grid) {
            const glm::mat4 view_inv = glm::inverse(grid.view);
            const glm::vec3 cam_pos = glm::vec3(view_inv[3]);
            const glm::vec3 cam_right = glm::vec3(view_inv[0]);
            const glm::vec3 cam_up = glm::vec3(view_inv[1]);
            const glm::vec3 cam_forward = -glm::vec3(view_inv[2]);

            glm::vec3 near_origin{0.0f};
            glm::vec3 near_x{0.0f};
            glm::vec3 near_y{0.0f};
            glm::vec3 far_origin{0.0f};
            glm::vec3 far_x{0.0f};
            glm::vec3 far_y{0.0f};
            if (grid.orthographic) {
                const float half_width = 1.0f / grid.projection[0][0];
                const float half_height = 1.0f / std::abs(grid.projection[1][1]);
                const glm::vec3 right_offset = cam_right * half_width;
                const glm::vec3 up_offset = cam_up * half_height;
                constexpr float kRayNear = -1000.0f;
                constexpr float kRayFar = 1000.0f;

                const glm::vec3 near_center = cam_pos + cam_forward * kRayNear;
                near_origin = near_center - right_offset - up_offset;
                near_x = right_offset * 2.0f;
                near_y = up_offset * 2.0f;

                const glm::vec3 far_center = cam_pos + cam_forward * kRayFar;
                far_origin = far_center - right_offset - up_offset;
                far_x = right_offset * 2.0f;
                far_y = up_offset * 2.0f;
            } else {
                const float fov_y = 2.0f * std::atan(1.0f / std::abs(grid.projection[1][1]));
                const float aspect = std::abs(grid.projection[1][1] / grid.projection[0][0]);
                const float half_height = std::tan(fov_y * 0.5f);
                const float half_width = half_height * aspect;
                const glm::vec3 far_center = cam_pos + cam_forward;
                const glm::vec3 right_offset = cam_right * half_width;
                const glm::vec3 up_offset = cam_up * half_height;
                const glm::vec3 far_bl = far_center - right_offset - up_offset;
                const glm::vec3 far_br = far_center + right_offset - up_offset;
                const glm::vec3 far_tl = far_center - right_offset + up_offset;

                near_origin = cam_pos;
                far_origin = far_bl;
                far_x = far_br - far_bl;
                far_y = far_tl - far_bl;
            }

            GridUniform uniform{};
            uniform.view_projection = grid.view_projection;
            uniform.view_position_plane = glm::vec4(grid.view_position,
                                                    static_cast<float>(std::clamp(grid.plane, 0, 2)));
            uniform.opacity_padding = glm::vec4(std::clamp(grid.opacity, 0.0f, 1.0f), 0.0f, 0.0f, 0.0f);
            uniform.near_origin = glm::vec4(near_origin, 0.0f);
            uniform.near_x = glm::vec4(near_x, 0.0f);
            uniform.near_y = glm::vec4(near_y, 0.0f);
            uniform.far_origin = glm::vec4(far_origin, 0.0f);
            uniform.far_x = glm::vec4(far_x, 0.0f);
            uniform.far_y = glm::vec4(far_y, 0.0f);
            return uniform;
        }

        [[nodiscard]] static std::vector<VulkanViewportGridOverlay> collectGridOverlays(
            const VulkanViewportPassParams& params) {
            if (!params.grid_overlays.empty()) {
                return params.grid_overlays;
            }
            if (!params.grid_enabled) {
                return {};
            }
            return {VulkanViewportGridOverlay{
                .viewport_pos = params.viewport_pos,
                .viewport_size = params.viewport_size,
                .render_size = {
                    std::max(static_cast<int>(std::lround(params.viewport_size.x)), 1),
                    std::max(static_cast<int>(std::lround(params.viewport_size.y)), 1)},
                .view = params.grid_view,
                .projection = params.grid_projection,
                .view_projection = params.grid_view_projection,
                .view_position = params.grid_view_position,
                .plane = params.grid_plane,
                .opacity = params.grid_opacity,
                .orthographic = params.grid_orthographic,
            }};
        }

        void updateGridUniforms(const VulkanViewportPassParams& params) {
            auto& frame = resourcesForFrame(params.frame_slot);
            const auto grids = collectGridOverlays(params);
            if (grids.empty()) {
                frame.grid_uniform.count = 0;
                return;
            }
            if (!ensureGridUniformBuffer(frame, grids.size())) {
                return;
            }

            std::vector<GridUniform> uniforms;
            uniforms.reserve(grids.size());
            for (const auto& grid : grids) {
                uniforms.push_back(makeGridUniform(grid));
            }
            if (uniforms.empty()) {
                frame.grid_uniform.count = 0;
                return;
            }

            const VkDeviceSize bytes =
                static_cast<VkDeviceSize>(sizeof(GridUniform) * uniforms.size());
            if (writeAllocation(frame.grid_uniform.allocation, uniforms.data(), bytes)) {
                frame.grid_uniform.count = static_cast<std::uint32_t>(
                    std::min<std::size_t>(uniforms.size(), std::numeric_limits<std::uint32_t>::max()));
            }
        }

        void uploadSceneImage(const VulkanViewportPassParams& params) {
            auto& frame = resourcesForFrame(params.frame_slot);
            scene_image_uploader.upload(params, frame.scene_descriptor_set);
        }

        void prepare(const VulkanViewportPassParams& params) {
            auto& frame = resourcesForFrame(params.frame_slot);
            updateQuadBuffer(params.scene_image_flip_y);
            updateGridUniforms(params);
            updateTexturedOverlayBuffer(frame, params);
            updateOverlayBuffer(frame, params);
            updateShapeOverlayBuffer(params.shape_overlay_triangles, frame.shape_overlay);
            updateShapeOverlayBuffer(params.ui_shape_overlay_triangles, frame.ui_shape_overlay);
            uploadSceneImage(params);

            VulkanMeshPassParams mesh_params{
                .view_projection = params.mesh_view_projection,
                .camera_position = params.mesh_camera_position,
                .items = params.mesh_items,
            };
            mesh_pass.prepare(*context, mesh_params);
            environment_pass.prepare(params.environment);
            depth_blit_pass.prepare(params.depth_blit);
            split_view_pass.prepare(params.split_view);
        }

        void bindViewport(VkCommandBuffer command_buffer, const FramebufferRect& rect) const {
            VkViewport viewport{};
            viewport.x = static_cast<float>(rect.x);
            viewport.y = static_cast<float>(rect.y);
            viewport.width = static_cast<float>(rect.width);
            viewport.height = static_cast<float>(rect.height);
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;
            VkRect2D scissor{};
            scissor.offset = {rect.x, rect.y};
            scissor.extent = {rect.width, rect.height};
            vkCmdSetViewport(command_buffer, 0, 1, &viewport);
            vkCmdSetScissor(command_buffer, 0, 1, &scissor);
        }

        void bindQuad(VkCommandBuffer command_buffer) const {
            const VkDeviceSize offset = 0;
            vkCmdBindVertexBuffers(command_buffer, 0, 1, &quad_buffer, &offset);
        }

        void clearViewport(VkCommandBuffer command_buffer, const FramebufferRect& rect, const glm::vec3 color) const {
            VkClearAttachment attachment{};
            attachment.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            attachment.colorAttachment = 0;
            attachment.clearValue.color = VkClearColorValue{{color.r, color.g, color.b, 1.0f}};
            VkClearRect clear_rect{};
            clear_rect.rect.offset = {rect.x, rect.y};
            clear_rect.rect.extent = {rect.width, rect.height};
            clear_rect.baseArrayLayer = 0;
            clear_rect.layerCount = 1;
            vkCmdClearAttachments(command_buffer, 1, &attachment, 1, &clear_rect);
        }

        void recordGridOverlays(VkCommandBuffer command_buffer,
                                const VkExtent2D extent,
                                const VulkanViewportPassParams& params,
                                const FramebufferRect& main_rect) const {
            const auto& frame = resourcesForFrame(params.frame_slot);
            if (frame.grid_uniform.count == 0 ||
                grid_pipeline == VK_NULL_HANDLE ||
                frame.grid_descriptor_set == VK_NULL_HANDLE ||
                frame.grid_uniform.buffer == VK_NULL_HANDLE) {
                return;
            }

            vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, grid_pipeline);
            vkCmdBindDescriptorSets(command_buffer,
                                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    grid_pipeline_layout,
                                    0,
                                    1,
                                    &frame.grid_descriptor_set,
                                    0,
                                    nullptr);
            const auto grids = collectGridOverlays(params);
            for (std::uint32_t i = 0;
                 i < std::min<std::uint32_t>(frame.grid_uniform.count, static_cast<std::uint32_t>(grids.size()));
                 ++i) {
                const auto& grid = grids[i];
                if (grid.viewport_size.x <= 0.0f || grid.viewport_size.y <= 0.0f ||
                    grid.render_size.x <= 0 || grid.render_size.y <= 0 ||
                    grid.opacity <= 0.0f) {
                    continue;
                }
                const FramebufferRect grid_rect = toFramebufferRect(params, grid, extent);
                if (grid_rect.width == 0 || grid_rect.height == 0) {
                    continue;
                }
                bindViewport(command_buffer, grid_rect);
                GridPush push{};
                push.grid_index = static_cast<std::int32_t>(i);
                vkCmdPushConstants(command_buffer,
                                   grid_pipeline_layout,
                                   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                   0,
                                   sizeof(push),
                                   &push);
                vkCmdDraw(command_buffer, 6, 1, 0, 0);
            }
            bindViewport(command_buffer, main_rect);
        }

        void recordShapeOverlays(VkCommandBuffer command_buffer,
                                 const DynamicBuffer& resource) const {
            if (resource.count == 0 ||
                resource.buffer == VK_NULL_HANDLE ||
                shape_overlay_pipeline == VK_NULL_HANDLE) {
                return;
            }
            const VkDeviceSize offset = 0;
            vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shape_overlay_pipeline);
            vkCmdBindVertexBuffers(command_buffer, 0, 1, &resource.buffer, &offset);
            vkCmdDraw(command_buffer, resource.count, 1, 0, 0);
            bindQuad(command_buffer);
        }

        void record(VkCommandBuffer command_buffer,
                    const VkExtent2D extent,
                    const VulkanViewportPassParams& params) {
            const auto& frame = resourcesForFrame(params.frame_slot);
            const FramebufferRect rect = toFramebufferRect(params, extent);
            if (rect.width == 0 || rect.height == 0 || quad_buffer == VK_NULL_HANDLE) {
                return;
            }
            bindViewport(command_buffer, rect);
            bindQuad(command_buffer);
            clearViewport(command_buffer, rect, params.background_color);

            // Environment background runs first so the splat scene quad and meshes draw
            // on top. Skipped when not enabled / no map loaded.
            if (params.environment.enabled && environment_pass.hasTexture()) {
                environment_pass.record(command_buffer,
                                        {static_cast<std::uint32_t>(rect.width),
                                         static_cast<std::uint32_t>(rect.height)},
                                        params.environment);
                bindQuad(command_buffer);
                bindViewport(command_buffer, rect);
            }

            const bool has_scene =
                scene_image_uploader.hasImage() &&
                frame.scene_descriptor_set != VK_NULL_HANDLE && scene_pipeline != VK_NULL_HANDLE;
            const bool split_active = params.split_view.enabled && split_view_pass.ready();
            if (split_active) {
                // content_rect arrives panel-local; lift it into framebuffer
                // coords so the shader's letterbox check matches gl_FragCoord.
                VulkanSplitViewParams adjusted = params.split_view;
                adjusted.content_rect.x += rect.x;
                adjusted.content_rect.y += rect.y;
                const VkRect2D panel_rect{
                    .offset = {rect.x, rect.y},
                    .extent = {static_cast<std::uint32_t>(rect.width),
                               static_cast<std::uint32_t>(rect.height)},
                };
                split_view_pass.record(command_buffer, panel_rect, adjusted);
                bindQuad(command_buffer);
                bindViewport(command_buffer, rect);
            } else if (has_scene) {
                vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, scene_pipeline);
                vkCmdBindDescriptorSets(command_buffer,
                                        VK_PIPELINE_BIND_POINT_GRAPHICS,
                                        scene_pipeline_layout,
                                        0,
                                        1,
                                        &frame.scene_descriptor_set,
                                        0,
                                        nullptr);
                vkCmdDraw(command_buffer, 6, 1, 0, 0);
            }
            // Splat depth → framebuffer depth attachment. Lets the mesh draws that
            // follow depth-test against the splat surface so meshes occluded by
            // splats render correctly.
            const bool split_mesh_panels_active =
                split_active && !params.mesh_items.empty() && !params.mesh_panels.empty();
            if (!split_active && !params.mesh_items.empty() && depth_blit_pass.hasDepth()) {
                const VkRect2D depth_rect{
                    .offset = {rect.x, rect.y},
                    .extent = {static_cast<std::uint32_t>(rect.width),
                               static_cast<std::uint32_t>(rect.height)},
                };
                depth_blit_pass.record(command_buffer, depth_rect, params.depth_blit);
                bindQuad(command_buffer);
                bindViewport(command_buffer, rect);
            }
            // Mesh sub-pass: GPU-rasterize meshes after the cached splat scene blit.
            if (!params.mesh_items.empty()) {
                VulkanMeshPassParams mesh_params{.items = params.mesh_items};
                if (split_mesh_panels_active) {
                    const int rect_min_x = rect.x;
                    const int rect_max_x = rect.x + static_cast<int>(rect.width);
                    for (const auto& panel : params.mesh_panels) {
                        const int x0 = std::clamp(
                            rect.x + static_cast<int>(std::lround(panel.start_position * static_cast<float>(rect.width))),
                            rect_min_x,
                            rect_max_x);
                        const int x1 = std::clamp(
                            rect.x + static_cast<int>(std::lround(panel.end_position * static_cast<float>(rect.width))),
                            rect_min_x,
                            rect_max_x);
                        if (x1 <= x0) {
                            continue;
                        }
                        mesh_params.view_projection = panel.view_projection;
                        mesh_params.camera_position = panel.camera_position;
                        const VkRect2D mesh_rect{
                            .offset = {x0, rect.y},
                            .extent = {static_cast<std::uint32_t>(x1 - x0),
                                       static_cast<std::uint32_t>(rect.height)},
                        };
                        mesh_pass.record(command_buffer, mesh_rect, mesh_params);
                        bindQuad(command_buffer);
                        bindViewport(command_buffer, rect);
                    }
                } else if (!split_active) {
                    mesh_params.view_projection = params.mesh_view_projection;
                    mesh_params.camera_position = params.mesh_camera_position;
                    const VkRect2D mesh_rect{
                        .offset = {rect.x, rect.y},
                        .extent = {static_cast<std::uint32_t>(rect.width),
                                   static_cast<std::uint32_t>(rect.height)},
                    };
                    mesh_pass.record(command_buffer, mesh_rect, mesh_params);
                    bindQuad(command_buffer);
                    bindViewport(command_buffer, rect);
                }
            }

            if (frame.textured_overlay.count > 0 && textured_overlay_pipeline != VK_NULL_HANDLE &&
                frame.textured_overlay.buffer != VK_NULL_HANDLE && !params.textured_overlays.empty()) {
                const VkDeviceSize offset = 0;
                vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, textured_overlay_pipeline);
                vkCmdBindVertexBuffers(command_buffer, 0, 1, &frame.textured_overlay.buffer, &offset);
                std::uint32_t first_vertex = 0;
                for (const auto& overlay : params.textured_overlays) {
                    if (overlay.texture_id == 0 || first_vertex + 6u > frame.textured_overlay.count) {
                        first_vertex += 6u;
                        continue;
                    }
                    const VkDescriptorSet descriptor_set = descriptorSetFromId(overlay.texture_id);
                    if (descriptor_set == VK_NULL_HANDLE) {
                        first_vertex += 6u;
                        continue;
                    }
                    TexturedOverlayPush push{};
                    push.tint_opacity = overlay.tint_opacity;
                    push.effects = overlay.effects;
                    vkCmdBindDescriptorSets(command_buffer,
                                            VK_PIPELINE_BIND_POINT_GRAPHICS,
                                            textured_overlay_pipeline_layout,
                                            0,
                                            1,
                                            &descriptor_set,
                                            0,
                                            nullptr);
                    vkCmdPushConstants(command_buffer,
                                       textured_overlay_pipeline_layout,
                                       VK_SHADER_STAGE_FRAGMENT_BIT,
                                       0,
                                       sizeof(push),
                                       &push);
                    vkCmdDraw(command_buffer, 6, 1, first_vertex, 0);
                    first_vertex += 6u;
                }
                bindQuad(command_buffer);
            }

            const std::uint32_t post_ui_overlay_vertices =
                std::min(params.post_ui_overlay_vertex_count, frame.overlay.count);
            const std::uint32_t overlay_vertices = frame.overlay.count - post_ui_overlay_vertices;

            // Most overlays sit below viewport UI shapes. A small tail segment is
            // reserved for full-viewport state overlays, such as export dimming.
            if (overlay_vertices > 0 && overlay_pipeline != VK_NULL_HANDLE &&
                frame.overlay.buffer != VK_NULL_HANDLE) {
                const VkDeviceSize offset = 0;
                vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, overlay_pipeline);
                vkCmdBindVertexBuffers(command_buffer, 0, 1, &frame.overlay.buffer, &offset);
                vkCmdDraw(command_buffer, overlay_vertices, 1, 0, 0);
                bindQuad(command_buffer);
            }

            recordShapeOverlays(command_buffer, frame.shape_overlay);

            if (!params.pivot_overlays.empty() && pivot_pipeline != VK_NULL_HANDLE) {
                bindQuad(command_buffer);
                vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pivot_pipeline);
                for (const auto& pivot : params.pivot_overlays) {
                    PivotPush push{};
                    push.center_size = {
                        pivot.center_ndc.x,
                        pivot.center_ndc.y,
                        pivot.size_ndc.x,
                        pivot.size_ndc.y,
                    };
                    push.color_opacity = {
                        pivot.color.r,
                        pivot.color.g,
                        pivot.color.b,
                        std::clamp(pivot.opacity, 0.0f, 1.0f),
                    };
                    vkCmdPushConstants(command_buffer,
                                       pivot_pipeline_layout,
                                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                       0,
                                       sizeof(push),
                                       &push);
                    vkCmdDraw(command_buffer, 6, 1, 0, 0);
                }
            }

            recordGridOverlays(command_buffer, extent, params, rect);

            if (params.vignette_enabled && vignette_pipeline != VK_NULL_HANDLE) {
                VignettePush push{};
                push.viewport_intensity_radius = {
                    static_cast<float>(rect.width),
                    static_cast<float>(rect.height),
                    params.vignette_intensity,
                    params.vignette_radius,
                };
                push.softness_padding = {params.vignette_softness, 0.0f, 0.0f, 0.0f};
                vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vignette_pipeline);
                vkCmdPushConstants(command_buffer,
                                   vignette_pipeline_layout,
                                   VK_SHADER_STAGE_FRAGMENT_BIT,
                                   0,
                                   sizeof(push),
                                   &push);
                vkCmdDraw(command_buffer, 6, 1, 0, 0);
            }

            recordShapeOverlays(command_buffer, frame.ui_shape_overlay);

            if (post_ui_overlay_vertices > 0 && overlay_pipeline != VK_NULL_HANDLE &&
                frame.overlay.buffer != VK_NULL_HANDLE) {
                const VkDeviceSize offset = 0;
                vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, overlay_pipeline);
                vkCmdBindVertexBuffers(command_buffer, 0, 1, &frame.overlay.buffer, &offset);
                vkCmdDraw(command_buffer, post_ui_overlay_vertices, 1, overlay_vertices, 0);
                bindQuad(command_buffer);
            }
        }

        void reset() {
            if (device != VK_NULL_HANDLE) {
                if (context != nullptr && !context->waitForSubmittedFrames()) {
                    LOG_WARN("Vulkan viewport pass shutdown could not wait for submitted frames: {}",
                             context->lastError());
                }
                scene_image_uploader.shutdown();
                mesh_pass.shutdown();
                environment_pass.shutdown();
                depth_blit_pass.shutdown();
                split_view_pass.shutdown();
                if (scene_pipeline != VK_NULL_HANDLE)
                    vkDestroyPipeline(device, scene_pipeline, nullptr);
                if (vignette_pipeline != VK_NULL_HANDLE)
                    vkDestroyPipeline(device, vignette_pipeline, nullptr);
                if (grid_pipeline != VK_NULL_HANDLE)
                    vkDestroyPipeline(device, grid_pipeline, nullptr);
                if (overlay_pipeline != VK_NULL_HANDLE)
                    vkDestroyPipeline(device, overlay_pipeline, nullptr);
                if (shape_overlay_pipeline != VK_NULL_HANDLE)
                    vkDestroyPipeline(device, shape_overlay_pipeline, nullptr);
                if (textured_overlay_pipeline != VK_NULL_HANDLE)
                    vkDestroyPipeline(device, textured_overlay_pipeline, nullptr);
                if (pivot_pipeline != VK_NULL_HANDLE)
                    vkDestroyPipeline(device, pivot_pipeline, nullptr);
                if (scene_pipeline_layout != VK_NULL_HANDLE)
                    vkDestroyPipelineLayout(device, scene_pipeline_layout, nullptr);
                if (vignette_pipeline_layout != VK_NULL_HANDLE)
                    vkDestroyPipelineLayout(device, vignette_pipeline_layout, nullptr);
                if (grid_pipeline_layout != VK_NULL_HANDLE)
                    vkDestroyPipelineLayout(device, grid_pipeline_layout, nullptr);
                if (overlay_pipeline_layout != VK_NULL_HANDLE)
                    vkDestroyPipelineLayout(device, overlay_pipeline_layout, nullptr);
                if (shape_overlay_pipeline_layout != VK_NULL_HANDLE)
                    vkDestroyPipelineLayout(device, shape_overlay_pipeline_layout, nullptr);
                if (textured_overlay_pipeline_layout != VK_NULL_HANDLE)
                    vkDestroyPipelineLayout(device, textured_overlay_pipeline_layout, nullptr);
                if (pivot_pipeline_layout != VK_NULL_HANDLE)
                    vkDestroyPipelineLayout(device, pivot_pipeline_layout, nullptr);
                if (quad_buffer != VK_NULL_HANDLE)
                    vmaDestroyBuffer(allocator, quad_buffer, quad_allocation);
                for (auto& frame : frame_resources) {
                    destroyDynamicBuffer(frame.overlay);
                    destroyDynamicBuffer(frame.shape_overlay);
                    destroyDynamicBuffer(frame.ui_shape_overlay);
                    destroyDynamicBuffer(frame.textured_overlay);
                    destroyDynamicBuffer(frame.grid_uniform);
                }
                if (scene_sampler != VK_NULL_HANDLE)
                    vkDestroySampler(device, scene_sampler, nullptr);
                if (scene_descriptor_pool != VK_NULL_HANDLE)
                    vkDestroyDescriptorPool(device, scene_descriptor_pool, nullptr);
                if (scene_descriptor_layout != VK_NULL_HANDLE)
                    vkDestroyDescriptorSetLayout(device, scene_descriptor_layout, nullptr);
                if (grid_descriptor_pool != VK_NULL_HANDLE)
                    vkDestroyDescriptorPool(device, grid_descriptor_pool, nullptr);
                if (grid_descriptor_layout != VK_NULL_HANDLE)
                    vkDestroyDescriptorSetLayout(device, grid_descriptor_layout, nullptr);
            }
            *this = {};
        }
#else
        [[nodiscard]] bool init(VulkanContext&) { return false; }
        void prepare(const VulkanViewportPassParams&) {}
        void record(VkCommandBuffer, VkExtent2D, const VulkanViewportPassParams&) {}
        void reset() {}
#endif
    };

    VulkanViewportPass::VulkanViewportPass() = default;

    VulkanViewportPass::~VulkanViewportPass() {
        shutdown();
    }

    bool VulkanViewportPass::init(VulkanContext& context) {
        if (!impl_) {
            impl_ = std::make_unique<Impl>();
        }
        return impl_->init(context);
    }

    void VulkanViewportPass::prepare(VulkanContext& context, const VulkanViewportPassParams& params) {
        if (!impl_ && !init(context)) {
            return;
        }
        impl_->prepare(params);
    }

    void VulkanViewportPass::record(VkCommandBuffer command_buffer,
                                    VkExtent2D framebuffer_extent,
                                    const VulkanViewportPassParams& params) {
        if (impl_) {
            impl_->record(command_buffer, framebuffer_extent, params);
        }
    }

    void VulkanViewportPass::shutdown() {
        if (impl_) {
            impl_->reset();
        }
    }

} // namespace lfs::vis
