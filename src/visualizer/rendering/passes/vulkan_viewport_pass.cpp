/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vulkan_viewport_pass.hpp"

#include "config.h"
#include "core/logger.hpp"
#include "core/tensor.hpp"
#include "rendering/image_layout.hpp"
#include "window/vulkan_context.hpp"
#include "window/vulkan_frame_graph.hpp"

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
#include <optional>
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

        [[nodiscard]] std::optional<std::vector<std::uint8_t>> tensorToRgba8(
            const lfs::core::Tensor& image,
            const glm::ivec2 expected_size) {
            if (!image.is_valid() || image.ndim() != 3 || expected_size.x <= 0 || expected_size.y <= 0) {
                return std::nullopt;
            }

            const auto layout = lfs::rendering::detectImageLayout(image);
            if (layout == lfs::rendering::ImageLayout::Unknown) {
                LOG_ERROR("Vulkan viewport pass received unsupported tensor shape [{}, {}, {}]",
                          image.size(0), image.size(1), image.size(2));
                return std::nullopt;
            }

            lfs::core::Tensor formatted = (layout == lfs::rendering::ImageLayout::HWC)
                                              ? image
                                              : image.permute({1, 2, 0}).contiguous();
            if (formatted.device() == lfs::core::Device::CUDA) {
                formatted = formatted.cpu();
            }
            if (formatted.dtype() != lfs::core::DataType::UInt8) {
                formatted = (formatted.clamp(0.0f, 1.0f) * 255.0f).to(lfs::core::DataType::UInt8);
            }
            formatted = formatted.contiguous();

            const int height = static_cast<int>(formatted.size(0));
            const int width = static_cast<int>(formatted.size(1));
            const int channels = static_cast<int>(formatted.size(2));
            if (width != expected_size.x || height != expected_size.y || !formatted.ptr<std::uint8_t>()) {
                LOG_ERROR("Vulkan viewport pass dimension mismatch: {}x{} vs {}x{}",
                          width, height, expected_size.x, expected_size.y);
                return std::nullopt;
            }
            if (channels != 1 && channels != 3 && channels != 4) {
                LOG_ERROR("Vulkan viewport pass received unsupported channel count {}", channels);
                return std::nullopt;
            }

            const std::uint8_t* const src = formatted.ptr<std::uint8_t>();
            std::vector<std::uint8_t> rgba(static_cast<std::size_t>(width) *
                                           static_cast<std::size_t>(height) * 4u);
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    const std::size_t src_offset =
                        (static_cast<std::size_t>(y) * width + x) * static_cast<std::size_t>(channels);
                    const std::size_t dst_offset = (static_cast<std::size_t>(y) * width + x) * 4u;
                    if (channels == 1) {
                        rgba[dst_offset + 0] = src[src_offset];
                        rgba[dst_offset + 1] = src[src_offset];
                        rgba[dst_offset + 2] = src[src_offset];
                        rgba[dst_offset + 3] = 255;
                    } else {
                        rgba[dst_offset + 0] = src[src_offset + 0];
                        rgba[dst_offset + 1] = src[src_offset + 1];
                        rgba[dst_offset + 2] = src[src_offset + 2];
                        rgba[dst_offset + 3] = channels >= 4 ? src[src_offset + 3] : 255;
                    }
                }
            }
            return rgba;
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
        VmaAllocator allocator = VK_NULL_HANDLE;
        VkPipelineCache pipeline_cache = VK_NULL_HANDLE;
        VkQueue graphics_queue = VK_NULL_HANDLE;
        std::uint32_t graphics_queue_family = 0;
        VkFormat color_format = VK_FORMAT_UNDEFINED;
        VkFormat depth_stencil_format = VK_FORMAT_UNDEFINED;
        std::size_t frames_in_flight = 1;

        VkCommandPool upload_command_pool = VK_NULL_HANDLE;
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
        VkImage scene_image = VK_NULL_HANDLE;
        VmaAllocation scene_image_allocation = VK_NULL_HANDLE;
        VkImageView scene_image_view = VK_NULL_HANDLE;
        VulkanFrameGraph scene_image_graph;
        glm::ivec2 scene_image_size{0, 0};
        const lfs::core::Tensor* uploaded_scene_tensor = nullptr;
        bool scene_image_external = false;
        std::uint64_t scene_image_external_generation = 0;

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

            VkCommandPoolCreateInfo command_pool_info{};
            command_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            command_pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            command_pool_info.queueFamilyIndex = graphics_queue_family;
            if (vkCreateCommandPool(device, &command_pool_info, nullptr, &upload_command_pool) != VK_SUCCESS) {
                LOG_ERROR("Failed to create Vulkan viewport upload command pool");
                reset();
                return false;
            }

            if (!createSampler() || !createSceneDescriptors() || !createGridResources() ||
                !createQuadBuffer() || !createPipelines()) {
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
                                  scene_descriptor_layout, nullptr, false, PipelineVertexLayout::ScreenQuad,
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

        [[nodiscard]] VkCommandBuffer beginUploadCommands() const {
            VkCommandBufferAllocateInfo alloc_info{};
            alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            alloc_info.commandPool = upload_command_pool;
            alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            alloc_info.commandBufferCount = 1;
            VkCommandBuffer command_buffer = VK_NULL_HANDLE;
            if (vkAllocateCommandBuffers(device, &alloc_info, &command_buffer) != VK_SUCCESS) {
                return VK_NULL_HANDLE;
            }
            VkCommandBufferBeginInfo begin_info{};
            begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            if (vkBeginCommandBuffer(command_buffer, &begin_info) != VK_SUCCESS) {
                vkFreeCommandBuffers(device, upload_command_pool, 1, &command_buffer);
                return VK_NULL_HANDLE;
            }
            return command_buffer;
        }

        [[nodiscard]] bool endUploadCommands(const VkCommandBuffer command_buffer) const {
            if (vkEndCommandBuffer(command_buffer) != VK_SUCCESS) {
                vkFreeCommandBuffers(device, upload_command_pool, 1, &command_buffer);
                return false;
            }
            VkSubmitInfo submit_info{};
            submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit_info.commandBufferCount = 1;
            submit_info.pCommandBuffers = &command_buffer;
            const VkResult result = vkQueueSubmit(graphics_queue, 1, &submit_info, VK_NULL_HANDLE);
            if (result == VK_SUCCESS) {
                vkQueueWaitIdle(graphics_queue);
            }
            vkFreeCommandBuffers(device, upload_command_pool, 1, &command_buffer);
            return result == VK_SUCCESS;
        }

        void clearSceneImageBinding() {
            scene_image_graph.forgetImage(scene_image);
            scene_image = VK_NULL_HANDLE;
            scene_image_allocation = VK_NULL_HANDLE;
            scene_image_view = VK_NULL_HANDLE;
            scene_image_size = {0, 0};
            uploaded_scene_tensor = nullptr;
            scene_image_external = false;
            scene_image_external_generation = 0;
        }

        void updateSceneDescriptor(FrameResources& frame,
                                   const VkImageView image_view,
                                   const VkImageLayout image_layout) const {
            if (frame.scene_descriptor_set == VK_NULL_HANDLE) {
                return;
            }
            VkDescriptorImageInfo descriptor_info{};
            descriptor_info.sampler = scene_sampler;
            descriptor_info.imageView = image_view;
            descriptor_info.imageLayout = image_layout;
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = frame.scene_descriptor_set;
            write.dstBinding = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write.descriptorCount = 1;
            write.pImageInfo = &descriptor_info;
            vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
        }

        void destroySceneImage() {
            if (scene_image_external) {
                clearSceneImageBinding();
                return;
            }
            if (scene_image_view != VK_NULL_HANDLE) {
                vkDestroyImageView(device, scene_image_view, nullptr);
            }
            if (scene_image != VK_NULL_HANDLE) {
                vmaDestroyImage(allocator, scene_image, scene_image_allocation);
            }
            clearSceneImageBinding();
        }

        [[nodiscard]] bool ensureSceneImage(const glm::ivec2 size, FrameResources& frame) {
            if (scene_image != VK_NULL_HANDLE && scene_image_size == size) {
                updateSceneDescriptor(frame, scene_image_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                return true;
            }
            destroySceneImage();

            VkImageCreateInfo image_info{};
            image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            image_info.imageType = VK_IMAGE_TYPE_2D;
            image_info.extent = {static_cast<std::uint32_t>(size.x), static_cast<std::uint32_t>(size.y), 1};
            image_info.mipLevels = 1;
            image_info.arrayLayers = 1;
            image_info.format = VK_FORMAT_R8G8B8A8_UNORM;
            image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
            image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            image_info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            image_info.samples = VK_SAMPLE_COUNT_1_BIT;
            image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            VmaAllocationCreateInfo allocation_info{};
            allocation_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            if (vmaCreateImage(allocator,
                               &image_info,
                               &allocation_info,
                               &scene_image,
                               &scene_image_allocation,
                               nullptr) != VK_SUCCESS) {
                destroySceneImage();
                return false;
            }
            vmaSetAllocationName(allocator, scene_image_allocation, "Viewport scene image");

            VkImageViewCreateInfo view_info{};
            view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            view_info.image = scene_image;
            view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
            view_info.format = VK_FORMAT_R8G8B8A8_UNORM;
            view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            view_info.subresourceRange.baseMipLevel = 0;
            view_info.subresourceRange.levelCount = 1;
            view_info.subresourceRange.baseArrayLayer = 0;
            view_info.subresourceRange.layerCount = 1;
            if (vkCreateImageView(device, &view_info, nullptr, &scene_image_view) != VK_SUCCESS) {
                destroySceneImage();
                return false;
            }

            scene_image_size = size;
            scene_image_graph.registerImage(scene_image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED);
            updateSceneDescriptor(frame, scene_image_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            return true;
        }

        [[nodiscard]] bool bindExternalSceneImage(const VulkanViewportPassParams& params, FrameResources& frame) {
            if (params.external_scene_image == VK_NULL_HANDLE ||
                params.external_scene_image_view == VK_NULL_HANDLE ||
                params.scene_image_size.x <= 0 ||
                params.scene_image_size.y <= 0) {
                return false;
            }
            if (scene_image_external &&
                scene_image == params.external_scene_image &&
                scene_image_view == params.external_scene_image_view &&
                scene_image_size == params.scene_image_size &&
                scene_image_graph.imageLayout(scene_image, VK_IMAGE_LAYOUT_UNDEFINED) == params.external_scene_image_layout &&
                scene_image_external_generation == params.external_scene_image_generation) {
                updateSceneDescriptor(frame, scene_image_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                return true;
            }

            destroySceneImage();
            scene_image = params.external_scene_image;
            scene_image_view = params.external_scene_image_view;
            scene_image_size = params.scene_image_size;
            uploaded_scene_tensor = params.scene_image.get();
            scene_image_external = true;
            scene_image_external_generation = params.external_scene_image_generation;
            scene_image_graph.registerImage(scene_image, VK_IMAGE_ASPECT_COLOR_BIT, params.external_scene_image_layout);
            updateSceneDescriptor(frame, scene_image_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            return true;
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
            if (!params.scene_image || params.scene_image_size.x <= 0 || params.scene_image_size.y <= 0) {
                uploaded_scene_tensor = nullptr;
                return;
            }
            if (params.external_scene_image != VK_NULL_HANDLE &&
                params.external_scene_image_view != VK_NULL_HANDLE) {
                if (!bindExternalSceneImage(params, frame)) {
                    LOG_ERROR("Failed to bind external Vulkan viewport scene image");
                }
                return;
            }
            if (scene_image_external) {
                destroySceneImage();
            }
            if (uploaded_scene_tensor == params.scene_image.get() && scene_image_size == params.scene_image_size &&
                scene_image_view != VK_NULL_HANDLE) {
                updateSceneDescriptor(frame, scene_image_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                return;
            }
            const auto rgba = tensorToRgba8(*params.scene_image, params.scene_image_size);
            if (!rgba || rgba->empty() || !ensureSceneImage(params.scene_image_size, frame)) {
                return;
            }

            VkBuffer staging_buffer = VK_NULL_HANDLE;
            VmaAllocation staging_allocation = VK_NULL_HANDLE;
            const VkDeviceSize upload_size = static_cast<VkDeviceSize>(rgba->size());
            if (!createBuffer(upload_size,
                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                              staging_buffer,
                              staging_allocation)) {
                return;
            }

            if (!writeAllocation(staging_allocation, rgba->data(), upload_size)) {
                vmaDestroyBuffer(allocator, staging_buffer, staging_allocation);
                return;
            }

            VkCommandBuffer command_buffer = beginUploadCommands();
            if (command_buffer == VK_NULL_HANDLE) {
                vmaDestroyBuffer(allocator, staging_buffer, staging_allocation);
                return;
            }
            scene_image_graph.transitionImage(command_buffer,
                                              scene_image,
                                              VK_IMAGE_ASPECT_COLOR_BIT,
                                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            VkBufferImageCopy copy{};
            copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            copy.imageSubresource.mipLevel = 0;
            copy.imageSubresource.baseArrayLayer = 0;
            copy.imageSubresource.layerCount = 1;
            copy.imageExtent = {
                static_cast<std::uint32_t>(params.scene_image_size.x),
                static_cast<std::uint32_t>(params.scene_image_size.y),
                1,
            };
            vkCmdCopyBufferToImage(command_buffer,
                                   staging_buffer,
                                   scene_image,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                   1,
                                   &copy);
            scene_image_graph.transitionImage(command_buffer,
                                              scene_image,
                                              VK_IMAGE_ASPECT_COLOR_BIT,
                                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            if (endUploadCommands(command_buffer)) {
                uploaded_scene_tensor = params.scene_image.get();
            }
            vmaDestroyBuffer(allocator, staging_buffer, staging_allocation);
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

            const bool has_scene =
                (params.scene_image || scene_image_external) && scene_image_view != VK_NULL_HANDLE &&
                frame.scene_descriptor_set != VK_NULL_HANDLE && scene_pipeline != VK_NULL_HANDLE;
            if (has_scene) {
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

            if (frame.overlay.count > 0 && overlay_pipeline != VK_NULL_HANDLE &&
                frame.overlay.buffer != VK_NULL_HANDLE) {
                const VkDeviceSize offset = 0;
                vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, overlay_pipeline);
                vkCmdBindVertexBuffers(command_buffer, 0, 1, &frame.overlay.buffer, &offset);
                vkCmdDraw(command_buffer, frame.overlay.count, 1, 0, 0);
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
        }

        void reset() {
            if (device != VK_NULL_HANDLE) {
                vkDeviceWaitIdle(device);
                destroySceneImage();
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
                if (upload_command_pool != VK_NULL_HANDLE)
                    vkDestroyCommandPool(device, upload_command_pool, nullptr);
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
