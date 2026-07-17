/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vulkan_viewport_pass.hpp"

#include "config.h"
#include "core/logger.hpp"
#include "diagnostics/vram_profiler.hpp"
#include "viewport_pass_graph.hpp"
#include "vulkan_environment_pass.hpp"
#include "vulkan_mesh_pass.hpp"
#include "vulkan_scene_image_uploader.hpp"
#include "rendering/vulkan_wait.hpp"
#include "window/vulkan_barrier2.hpp"
#include "window/vulkan_context.hpp"
#include "window/vulkan_result.hpp"

#include "viewport/frustum.vert.spv.h"
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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <span>
#include <stop_token>
#include <string>
#include <string_view>
#include <vector>

namespace lfs::vis {

    namespace {

        [[nodiscard]] const char* waitOutcomeLabel(const lfs::rendering::WaitOutcome outcome) noexcept {
            using lfs::rendering::WaitOutcome;
            switch (outcome) {
            case WaitOutcome::Ready: return "Ready";
            case WaitOutcome::Cancelled: return "Cancelled";
            case WaitOutcome::Shutdown: return "Shutdown";
            case WaitOutcome::Quarantined: return "Quarantined";
            }
            return "Unknown";
        }

        [[nodiscard]] std::string formatWaitFailure(
            const lfs::Result<lfs::rendering::WaitOutcome>& outcome) {
            if (outcome.has_value()) {
                return waitOutcomeLabel(*outcome);
            }
            return std::string(outcome.error().detail());
        }

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
            // x,y: viewport origin (framebuffer px). z,w: viewport size (framebuffer px).
            glm::vec4 viewport_rect{0.0f, 0.0f, 0.0f, 0.0f};
            // x: depth_available, y: flip-y, z/w unused.
            glm::vec4 depth_params{0.0f, 0.0f, 0.0f, 0.0f};
        };

        struct ShapeOverlayPush {
            // x,y: viewport origin (framebuffer px). z,w: viewport size (framebuffer px).
            glm::vec4 viewport_rect{0.0f, 0.0f, 0.0f, 0.0f};
            // x: depth_available (1.0 = sample splat depth, 0.0 = skip fade),
            // y: flip-y when sampling depth UV. z,w unused.
            glm::vec4 params{0.0f, 0.0f, 0.0f, 0.0f};
        };

        struct FrustumPush {
            glm::vec4 viewport_rect{0.0f, 0.0f, 0.0f, 0.0f};
            glm::vec4 params{0.0f, 0.0f, 0.0f, 0.0f};
            glm::mat4 view{1.0f};
            glm::vec4 viewport_panel{0.0f, 0.0f, 0.0f, 0.0f};
            glm::vec4 projection{0.0f, 0.0f, 0.0f, 0.0f};
        };

        constexpr std::uint32_t kFrustumVertexCount = 48;
        constexpr float kFrustumLineThickness = 1.5f;

        [[nodiscard]] VkDescriptorSet descriptorSetFromId(const std::uintptr_t texture_id) {
            return reinterpret_cast<VkDescriptorSet>(texture_id);
        }

        [[nodiscard]] FramebufferRect toFramebufferRect(
            const VulkanViewportPassParams& params,
            const glm::vec2& viewport_pos,
            const glm::vec2& viewport_size,
            const VkExtent2D extent) {
            const float sx = params.framebuffer_scale.x > 0.0f ? params.framebuffer_scale.x : 1.0f;
            const float sy = params.framebuffer_scale.y > 0.0f ? params.framebuffer_scale.y : 1.0f;
            const int x0 = std::clamp(static_cast<int>(std::lround(viewport_pos.x * sx)),
                                      0, static_cast<int>(extent.width));
            const int y0 = std::clamp(static_cast<int>(std::lround(viewport_pos.y * sy)),
                                      0, static_cast<int>(extent.height));
            const int x1 = std::clamp(static_cast<int>(std::lround((viewport_pos.x + viewport_size.x) * sx)),
                                      0, static_cast<int>(extent.width));
            const int y1 = std::clamp(static_cast<int>(std::lround((viewport_pos.y + viewport_size.y) * sy)),
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
            const VkExtent2D extent) {
            return toFramebufferRect(params, params.viewport_pos, params.viewport_size, extent);
        }

        [[nodiscard]] FramebufferRect toFramebufferRect(
            const VulkanViewportPassParams& params,
            const VulkanViewportGridOverlay& grid,
            const VkExtent2D extent) {
            return toFramebufferRect(params, grid.viewport_pos, grid.viewport_size, extent);
        }
    } // namespace

    struct VulkanViewportPass::Impl {
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
            DynamicBuffer frustum_instances;
            VkDescriptorSet scene_descriptor_set = VK_NULL_HANDLE;
            VkDescriptorSet grid_descriptor_set = VK_NULL_HANDLE;
            VkDescriptorSet frustum_descriptor_set = VK_NULL_HANDLE;
            VkDescriptorSet shape_overlay_descriptor_set = VK_NULL_HANDLE;
            VkImageView bound_shape_overlay_depth_view = VK_NULL_HANDLE;
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

        VkDescriptorSetLayout frustum_descriptor_layout = VK_NULL_HANDLE;
        VkDescriptorPool frustum_descriptor_pool = VK_NULL_HANDLE;

        VkSampler shape_overlay_depth_sampler = VK_NULL_HANDLE;
        VkImage shape_overlay_dummy_depth_image = VK_NULL_HANDLE;
        VmaAllocation shape_overlay_dummy_depth_alloc = VK_NULL_HANDLE;
        VkImageView shape_overlay_dummy_depth_view = VK_NULL_HANDLE;
        std::string shape_overlay_dummy_depth_vram_label;
        VkDescriptorSetLayout shape_overlay_descriptor_layout = VK_NULL_HANDLE;
        VkDescriptorPool shape_overlay_descriptor_pool = VK_NULL_HANDLE;

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
        VkPipelineLayout frustum_pipeline_layout = VK_NULL_HANDLE;
        VkPipeline frustum_pipeline = VK_NULL_HANDLE;

        // Declarative pass-graph: record() runs this ordered set of sub-passes, each gated by its
        // own active() condition, rebinding shared viewport/quad state between them.
        ViewportPassGraph graph_;
        std::vector<std::unique_ptr<ViewportSubPass>> graph_passes_;

        [[nodiscard]] bool init(VulkanContext& ctx) {
            if (device != VK_NULL_HANDLE) {
                return true;
            }
            this->context = &ctx;
            device = ctx.device();
            allocator = ctx.allocator();
            pipeline_cache = ctx.pipelineCache();
            graphics_queue = ctx.graphicsQueue();
            graphics_queue_family = ctx.graphicsQueueFamily();
            color_format = ctx.swapchainFormat();
            depth_stencil_format = ctx.depthStencilFormat();
            frames_in_flight = std::max<std::size_t>(1, ctx.framesInFlight());
            frame_resources.resize(frames_in_flight);
            if (device == VK_NULL_HANDLE || allocator == VK_NULL_HANDLE ||
                graphics_queue == VK_NULL_HANDLE || color_format == VK_FORMAT_UNDEFINED ||
                depth_stencil_format == VK_FORMAT_UNDEFINED) {
                LOG_ERROR("Vulkan viewport pass requires an initialized context (device={:#x}, allocator={:#x}, graphics_queue={:#x}, color_format={}, depth_stencil_format={}, frame_count={}) ({}:{})",
                          vkHandleValue(device),
                          reinterpret_cast<std::uintptr_t>(allocator),
                          vkHandleValue(graphics_queue),
                          static_cast<int>(color_format),
                          static_cast<int>(depth_stencil_format),
                          frames_in_flight,
                          __FILE__,
                          __LINE__);
                device = VK_NULL_HANDLE;
                return false;
            }

            if (!createSampler() || !scene_image_uploader.init(ctx, scene_sampler) ||
                !createSceneDescriptors() || !createGridResources() ||
                !createFrustumResources() ||
                !createShapeOverlayDescriptors() ||
                !createQuadBuffer() || !createPipelines()) {
                reset();
                return false;
            }
            if (!mesh_pass.init(ctx, color_format, depth_stencil_format)) {
                LOG_ERROR("Vulkan viewport pass: mesh sub-pass init failed");
                reset();
                return false;
            }
            if (!environment_pass.init(ctx, color_format, depth_stencil_format, quad_buffer)) {
                LOG_ERROR("Vulkan viewport pass: environment sub-pass init failed");
                reset();
                return false;
            }
            if (!depth_blit_pass.init(ctx, color_format, depth_stencil_format, quad_buffer)) {
                LOG_ERROR("Vulkan viewport pass: depth-blit sub-pass init failed");
                reset();
                return false;
            }
            if (!split_view_pass.init(ctx, color_format, depth_stencil_format, quad_buffer)) {
                LOG_ERROR("Vulkan viewport pass: split-view sub-pass init failed");
                reset();
                return false;
            }
            registerGraphPasses();
            graph_.finalize();
            return true;
        }

        // Registers a sub-pass into graph_. active() is the authoritative draw condition: the graph
        // skips the pass when it returns false, so the matching recordXxx asserts the precondition
        // rather than re-checking it.
        void addGraphPass(const char* name, ViewportPhase phase,
                          LambdaSubPass::ActiveFn active, LambdaSubPass::RecordFn record) {
            graph_passes_.push_back(
                std::make_unique<LambdaSubPass>(name, phase, std::move(active), std::move(record)));
            graph_.add(*graph_passes_.back());
        }

        void registerGraphPasses() {
            using P = ViewportPhase;
            const auto rect_of = [](const ViewportRecordContext& c) {
                return FramebufferRect{c.rect_x, c.rect_y, c.rect_w, c.rect_h};
            };
            addGraphPass(
                "environment", P::Background,
                [this](const VulkanViewportPassParams& p) {
                    return p.environment.enabled && environment_pass.hasTexture(p.frame_slot);
                },
                [this, rect_of](const ViewportRecordContext& c, const VulkanViewportPassParams& p) {
                    recordEnvironmentPass(c.cmd, rect_of(c), p);
                });
            addGraphPass(
                "scene", P::Scene,
                [this](const VulkanViewportPassParams& p) {
                    if (sceneSplitActive(p)) {
                        return true;
                    }
                    const auto& frame = resourcesForFrame(p.frame_slot);
                    return scene_image_uploader.hasImage() && scene_pipeline != VK_NULL_HANDLE &&
                           frame.scene_descriptor_set != VK_NULL_HANDLE;
                },
                [this, rect_of](const ViewportRecordContext& c, const VulkanViewportPassParams& p) {
                    recordScenePass(c.cmd, rect_of(c), p);
                });
            addGraphPass(
                "depth_blit", P::DepthBlit,
                [this](const VulkanViewportPassParams& p) {
                    return !sceneSplitActive(p) && !p.mesh_items.empty() &&
                           depth_blit_pass.hasDepth(p.frame_slot);
                },
                [this, rect_of](const ViewportRecordContext& c, const VulkanViewportPassParams& p) {
                    recordDepthBlitPass(c.cmd, rect_of(c), p);
                });
            addGraphPass(
                "mesh", P::Geometry,
                [](const VulkanViewportPassParams& p) { return !p.mesh_items.empty(); },
                [this, rect_of](const ViewportRecordContext& c, const VulkanViewportPassParams& p) {
                    recordMeshPass(c.cmd, rect_of(c), p);
                });
            addGraphPass(
                "textured_overlay", P::WorldOverlay,
                [this](const VulkanViewportPassParams& p) {
                    const auto& frame = resourcesForFrame(p.frame_slot);
                    return frame.textured_overlay.count > 0 && textured_overlay_pipeline != VK_NULL_HANDLE &&
                           frame.textured_overlay.buffer != VK_NULL_HANDLE && !p.textured_overlays.empty();
                },
                [this](const ViewportRecordContext& c, const VulkanViewportPassParams& p) {
                    recordTexturedOverlayPass(c, p);
                });
            addGraphPass(
                "base_overlay", P::WorldOverlay,
                [this](const VulkanViewportPassParams& p) {
                    const auto& frame = resourcesForFrame(p.frame_slot);
                    return overlaySplit(frame, p).base > 0 && overlay_pipeline != VK_NULL_HANDLE &&
                           frame.overlay.buffer != VK_NULL_HANDLE;
                },
                [this](const ViewportRecordContext& c, const VulkanViewportPassParams& p) {
                    recordBaseOverlayPass(c.cmd, p);
                });
            addGraphPass(
                "world_shape_overlay", P::WorldOverlay,
                [this](const VulkanViewportPassParams& p) {
                    const auto& frame = resourcesForFrame(p.frame_slot);
                    return frame.shape_overlay.count > 0 && shape_overlay_pipeline != VK_NULL_HANDLE &&
                           frame.shape_overlay.buffer != VK_NULL_HANDLE;
                },
                [this](const ViewportRecordContext& c, const VulkanViewportPassParams& p) {
                    recordWorldShapePass(c, p);
                });
            addGraphPass(
                "frustum", P::WorldOverlay,
                [this](const VulkanViewportPassParams& p) {
                    const auto& frame = resourcesForFrame(p.frame_slot);
                    return frame.frustum_instances.count > 0 && frustum_pipeline != VK_NULL_HANDLE &&
                           frame.frustum_descriptor_set != VK_NULL_HANDLE &&
                           frame.shape_overlay_descriptor_set != VK_NULL_HANDLE &&
                           frame.frustum_instances.buffer != VK_NULL_HANDLE && !p.frustum_batches.empty();
                },
                [this](const ViewportRecordContext& c, const VulkanViewportPassParams& p) {
                    recordFrustumPass(c, p);
                });
            addGraphPass(
                "pivot", P::WorldOverlay,
                [this](const VulkanViewportPassParams& p) {
                    return !p.pivot_overlays.empty() && pivot_pipeline != VK_NULL_HANDLE;
                },
                [this](const ViewportRecordContext& c, const VulkanViewportPassParams& p) {
                    recordPivotPass(c.cmd, p);
                });
            addGraphPass(
                "grid", P::WorldOverlay,
                [this](const VulkanViewportPassParams& p) {
                    const auto& frame = resourcesForFrame(p.frame_slot);
                    return frame.grid_uniform.count > 0 && grid_pipeline != VK_NULL_HANDLE &&
                           frame.grid_descriptor_set != VK_NULL_HANDLE &&
                           frame.grid_uniform.buffer != VK_NULL_HANDLE;
                },
                [this](const ViewportRecordContext& c, const VulkanViewportPassParams& p) {
                    recordGridOverlays(c.cmd, c.extent, p);
                });
            addGraphPass(
                "vignette", P::Effect,
                [this](const VulkanViewportPassParams& p) {
                    return p.vignette_enabled && vignette_pipeline != VK_NULL_HANDLE;
                },
                [this, rect_of](const ViewportRecordContext& c, const VulkanViewportPassParams& p) {
                    recordVignettePass(c.cmd, rect_of(c), p);
                });
            addGraphPass(
                "ui_shape_overlay", P::UiOverlay,
                [this](const VulkanViewportPassParams& p) {
                    const auto& frame = resourcesForFrame(p.frame_slot);
                    return frame.ui_shape_overlay.count > 0 && shape_overlay_pipeline != VK_NULL_HANDLE &&
                           frame.ui_shape_overlay.buffer != VK_NULL_HANDLE;
                },
                [this](const ViewportRecordContext& c, const VulkanViewportPassParams& p) {
                    recordUiShapePass(c, p);
                });
            addGraphPass(
                "post_ui_overlay", P::UiOverlay,
                [this](const VulkanViewportPassParams& p) {
                    const auto& frame = resourcesForFrame(p.frame_slot);
                    return overlaySplit(frame, p).post_ui > 0 && overlay_pipeline != VK_NULL_HANDLE &&
                           frame.overlay.buffer != VK_NULL_HANDLE;
                },
                [this](const ViewportRecordContext& c, const VulkanViewportPassParams& p) {
                    recordPostUiOverlayPass(c.cmd, p);
                });
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

            const VkResult result =
                vmaCreateBuffer(allocator, &buffer_info, &allocation_info, &buffer, &allocation, nullptr);
            if (result != VK_SUCCESS) {
                buffer = VK_NULL_HANDLE;
                allocation = VK_NULL_HANDLE;
                return reportVkFailure(
                    "vmaCreateBuffer(allocator, &buffer_info, &allocation_info, &buffer, &allocation, nullptr)",
                    result,
                    std::format("Viewport host-visible buffer allocation failed (allocator={:#x}, requested_size={}, usage={:#x})",
                                reinterpret_cast<std::uintptr_t>(allocator),
                                size,
                                static_cast<std::uint32_t>(usage)));
            }
            return true;
        }

        [[nodiscard]] bool writeAllocation(const VmaAllocation allocation,
                                           const void* const source,
                                           const VkDeviceSize size) const {
            if (allocation == VK_NULL_HANDLE || !source || size == 0) {
                return logVkFailure(std::format(
                    "Viewport buffer write requires an allocation, source pointer, and non-zero size (allocation={:#x}, source={:#x}, write_size={}) ({}:{})",
                    reinterpret_cast<std::uintptr_t>(allocation),
                    reinterpret_cast<std::uintptr_t>(source),
                    size,
                    __FILE__,
                    __LINE__));
            }
            void* mapped = nullptr;
            const VkResult map_result = vmaMapMemory(allocator, allocation, &mapped);
            if (map_result != VK_SUCCESS) {
                return reportVkFailure(
                    "vmaMapMemory(allocator, allocation, &mapped)",
                    map_result,
                    std::format("Viewport buffer allocation could not be mapped (allocator={:#x}, allocation={:#x}, source={:#x}, write_size={})",
                                reinterpret_cast<std::uintptr_t>(allocator),
                                reinterpret_cast<std::uintptr_t>(allocation),
                                reinterpret_cast<std::uintptr_t>(source),
                                size));
            }
            if (mapped == nullptr) {
                return logVkFailure(std::format(
                    "Viewport buffer mapping returned success with a null pointer (allocator={:#x}, allocation={:#x}, source={:#x}, write_size={}, mapped={:#x}) ({}:{})",
                    reinterpret_cast<std::uintptr_t>(allocator),
                    reinterpret_cast<std::uintptr_t>(allocation),
                    reinterpret_cast<std::uintptr_t>(source),
                    size,
                    reinterpret_cast<std::uintptr_t>(mapped),
                    __FILE__,
                    __LINE__));
            }
            std::memcpy(mapped, source, static_cast<std::size_t>(size));
            const VkResult flush_result = vmaFlushAllocation(allocator, allocation, 0, size);
            vmaUnmapMemory(allocator, allocation);
            if (flush_result != VK_SUCCESS) {
                return reportVkFailure(
                    "vmaFlushAllocation(allocator, allocation, 0, size)",
                    flush_result,
                    std::format("Viewport buffer flush failed (allocator={:#x}, allocation={:#x}, offset=0, flush_size={})",
                                reinterpret_cast<std::uintptr_t>(allocator),
                                reinterpret_cast<std::uintptr_t>(allocation),
                                size));
            }
            return true;
        }

        [[nodiscard]] FrameResources& resourcesForFrame(const std::size_t frame_slot) {
            if (frame_slot >= frame_resources.size()) [[unlikely]] {
                throw std::logic_error(std::format(
                    "Viewport frame slot is outside the resource ring (frame_slot={}, ring_size={}) ({}:{})",
                    frame_slot,
                    frame_resources.size(),
                    __FILE__,
                    __LINE__));
            }
            return frame_resources[frame_slot];
        }

        [[nodiscard]] const FrameResources& resourcesForFrame(const std::size_t frame_slot) const {
            if (frame_slot >= frame_resources.size()) [[unlikely]] {
                throw std::logic_error(std::format(
                    "Viewport frame slot is outside the resource ring (frame_slot={}, ring_size={}) ({}:{})",
                    frame_slot,
                    frame_resources.size(),
                    __FILE__,
                    __LINE__));
            }
            return frame_resources[frame_slot];
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
            if (!vk_try_bool(
                    vkCreateSampler(device, &sampler_info, nullptr, &scene_sampler),
                    "vkCreateSampler(device, &sampler_info, nullptr, &scene_sampler)",
                    lfs::rendering::formatVulkanDiagnostic(
                                "Viewport scene sampler creation failed (device={:#x}, mag_filter={}, min_filter={}, address_mode={})",
                                vkHandleValue(device),
                                static_cast<int>(sampler_info.magFilter),
                                static_cast<int>(sampler_info.minFilter),
                                static_cast<int>(sampler_info.addressModeU)
                            ),
                    std::source_location::current())) {
                return false;
            }
            context->setDebugObjectName(VK_OBJECT_TYPE_SAMPLER,
                                        scene_sampler,
                                        "viewport.scene.sampler");
            return true;
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
            if (!vk_try_bool(
                    vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &scene_descriptor_layout),
                    "vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &scene_descriptor_layout)",
                    lfs::rendering::formatVulkanDiagnostic(
                                "Viewport scene descriptor-set layout creation failed (device={:#x}, binding_count={}, descriptor_type={})",
                                vkHandleValue(device),
                                layout_info.bindingCount,
                                static_cast<int>(binding.descriptorType)
                            ),
                    std::source_location::current())) {
                return false;
            }
            context->setDebugObjectName(VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                                        scene_descriptor_layout,
                                        "viewport.scene.descriptor.layout");

            const auto descriptor_count = static_cast<std::uint32_t>(frame_resources.size());
            VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, descriptor_count};
            VkDescriptorPoolCreateInfo pool_info{};
            pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            pool_info.maxSets = descriptor_count;
            pool_info.poolSizeCount = 1;
            pool_info.pPoolSizes = &pool_size;
            if (!vk_try_bool(
                    vkCreateDescriptorPool(device, &pool_info, nullptr, &scene_descriptor_pool),
                    "vkCreateDescriptorPool(device, &pool_info, nullptr, &scene_descriptor_pool)",
                    lfs::rendering::formatVulkanDiagnostic(
                                "Viewport scene descriptor-pool creation failed (device={:#x}, frame_count={}, max_sets={}, descriptor_count={})",
                                vkHandleValue(device),
                                frame_resources.size(),
                                pool_info.maxSets,
                                pool_size.descriptorCount
                            ),
                    std::source_location::current())) {
                return false;
            }
            context->setDebugObjectName(VK_OBJECT_TYPE_DESCRIPTOR_POOL,
                                        scene_descriptor_pool,
                                        "viewport.scene.descriptor.pool");

            std::vector<VkDescriptorSetLayout> layouts(frame_resources.size(), scene_descriptor_layout);
            std::vector<VkDescriptorSet> sets(frame_resources.size(), VK_NULL_HANDLE);
            VkDescriptorSetAllocateInfo alloc_info{};
            alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            alloc_info.descriptorPool = scene_descriptor_pool;
            alloc_info.descriptorSetCount = descriptor_count;
            alloc_info.pSetLayouts = layouts.data();
            if (!vk_try_bool(
                    vkAllocateDescriptorSets(device, &alloc_info, sets.data()),
                    "vkAllocateDescriptorSets(device, &alloc_info, sets.data())",
                    lfs::rendering::formatVulkanDiagnostic(
                                "Viewport scene descriptor-set allocation failed (device={:#x}, descriptor_pool={:#x}, descriptor_layout={:#x}, requested_count={})",
                                vkHandleValue(device),
                                vkHandleValue(scene_descriptor_pool),
                                vkHandleValue(scene_descriptor_layout),
                                alloc_info.descriptorSetCount
                            ),
                    std::source_location::current())) {
                return false;
            }
            for (std::size_t i = 0; i < frame_resources.size(); ++i) {
                frame_resources[i].scene_descriptor_set = sets[i];
                context->setDebugObjectNamef(VK_OBJECT_TYPE_DESCRIPTOR_SET,
                                             sets[i],
                                             "viewport.scene.descriptor[{}]",
                                             i);
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
            if (!vk_try_bool(
                    vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &grid_descriptor_layout),
                    "vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &grid_descriptor_layout)",
                    lfs::rendering::formatVulkanDiagnostic(
                                "Viewport grid descriptor-set layout creation failed (device={:#x}, binding_count={}, descriptor_type={})",
                                vkHandleValue(device),
                                layout_info.bindingCount,
                                static_cast<int>(binding.descriptorType)
                            ),
                    std::source_location::current())) {
                return false;
            }
            context->setDebugObjectName(VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                                        grid_descriptor_layout,
                                        "viewport.grid.descriptor.layout");

            const auto descriptor_count = static_cast<std::uint32_t>(frame_resources.size());
            VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptor_count};
            VkDescriptorPoolCreateInfo pool_info{};
            pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            pool_info.maxSets = descriptor_count;
            pool_info.poolSizeCount = 1;
            pool_info.pPoolSizes = &pool_size;
            if (!vk_try_bool(
                    vkCreateDescriptorPool(device, &pool_info, nullptr, &grid_descriptor_pool),
                    "vkCreateDescriptorPool(device, &pool_info, nullptr, &grid_descriptor_pool)",
                    lfs::rendering::formatVulkanDiagnostic(
                                "Viewport grid descriptor-pool creation failed (device={:#x}, frame_count={}, max_sets={}, descriptor_count={})",
                                vkHandleValue(device),
                                frame_resources.size(),
                                pool_info.maxSets,
                                pool_size.descriptorCount
                            ),
                    std::source_location::current())) {
                return false;
            }
            context->setDebugObjectName(VK_OBJECT_TYPE_DESCRIPTOR_POOL,
                                        grid_descriptor_pool,
                                        "viewport.grid.descriptor.pool");

            std::vector<VkDescriptorSetLayout> layouts(frame_resources.size(), grid_descriptor_layout);
            std::vector<VkDescriptorSet> sets(frame_resources.size(), VK_NULL_HANDLE);
            VkDescriptorSetAllocateInfo alloc_info{};
            alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            alloc_info.descriptorPool = grid_descriptor_pool;
            alloc_info.descriptorSetCount = descriptor_count;
            alloc_info.pSetLayouts = layouts.data();
            if (!vk_try_bool(
                    vkAllocateDescriptorSets(device, &alloc_info, sets.data()),
                    "vkAllocateDescriptorSets(device, &alloc_info, sets.data())",
                    lfs::rendering::formatVulkanDiagnostic(
                                "Viewport grid descriptor-set allocation failed (device={:#x}, descriptor_pool={:#x}, descriptor_layout={:#x}, requested_count={})",
                                vkHandleValue(device),
                                vkHandleValue(grid_descriptor_pool),
                                vkHandleValue(grid_descriptor_layout),
                                alloc_info.descriptorSetCount
                            ),
                    std::source_location::current())) {
                return false;
            }
            for (std::size_t i = 0; i < frame_resources.size(); ++i) {
                frame_resources[i].grid_descriptor_set = sets[i];
                context->setDebugObjectNamef(VK_OBJECT_TYPE_DESCRIPTOR_SET,
                                             sets[i],
                                             "viewport.grid.descriptor[{}]",
                                             i);
            }

            return true;
        }

        [[nodiscard]] bool createFrustumResources() {
            VkDescriptorSetLayoutBinding binding{};
            binding.binding = 0;
            binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            binding.descriptorCount = 1;
            binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
            VkDescriptorSetLayoutCreateInfo layout_info{};
            layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            layout_info.bindingCount = 1;
            layout_info.pBindings = &binding;
            if (!vk_try_bool(
                    vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &frustum_descriptor_layout),
                    "vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &frustum_descriptor_layout)",
                    lfs::rendering::formatVulkanDiagnostic(
                                "Viewport frustum descriptor-set layout creation failed (device={:#x}, binding_count={}, descriptor_type={})",
                                vkHandleValue(device),
                                layout_info.bindingCount,
                                static_cast<int>(binding.descriptorType)
                            ),
                    std::source_location::current())) {
                return false;
            }
            context->setDebugObjectName(VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                                        frustum_descriptor_layout,
                                        "viewport.frustum.descriptor.layout");

            const auto descriptor_count = static_cast<std::uint32_t>(frame_resources.size());
            VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptor_count};
            VkDescriptorPoolCreateInfo pool_info{};
            pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            pool_info.maxSets = descriptor_count;
            pool_info.poolSizeCount = 1;
            pool_info.pPoolSizes = &pool_size;
            if (!vk_try_bool(
                    vkCreateDescriptorPool(device, &pool_info, nullptr, &frustum_descriptor_pool),
                    "vkCreateDescriptorPool(device, &pool_info, nullptr, &frustum_descriptor_pool)",
                    lfs::rendering::formatVulkanDiagnostic(
                                "Viewport frustum descriptor-pool creation failed (device={:#x}, frame_count={}, max_sets={}, descriptor_count={})",
                                vkHandleValue(device),
                                frame_resources.size(),
                                pool_info.maxSets,
                                pool_size.descriptorCount
                            ),
                    std::source_location::current())) {
                return false;
            }
            context->setDebugObjectName(VK_OBJECT_TYPE_DESCRIPTOR_POOL,
                                        frustum_descriptor_pool,
                                        "viewport.frustum.descriptor.pool");

            std::vector<VkDescriptorSetLayout> layouts(frame_resources.size(), frustum_descriptor_layout);
            std::vector<VkDescriptorSet> sets(frame_resources.size(), VK_NULL_HANDLE);
            VkDescriptorSetAllocateInfo alloc_info{};
            alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            alloc_info.descriptorPool = frustum_descriptor_pool;
            alloc_info.descriptorSetCount = descriptor_count;
            alloc_info.pSetLayouts = layouts.data();
            if (!vk_try_bool(
                    vkAllocateDescriptorSets(device, &alloc_info, sets.data()),
                    "vkAllocateDescriptorSets(device, &alloc_info, sets.data())",
                    lfs::rendering::formatVulkanDiagnostic(
                                "Viewport frustum descriptor-set allocation failed (device={:#x}, descriptor_pool={:#x}, descriptor_layout={:#x}, requested_count={})",
                                vkHandleValue(device),
                                vkHandleValue(frustum_descriptor_pool),
                                vkHandleValue(frustum_descriptor_layout),
                                alloc_info.descriptorSetCount
                            ),
                    std::source_location::current())) {
                return false;
            }
            for (std::size_t i = 0; i < frame_resources.size(); ++i) {
                frame_resources[i].frustum_descriptor_set = sets[i];
                context->setDebugObjectNamef(VK_OBJECT_TYPE_DESCRIPTOR_SET,
                                             sets[i],
                                             "viewport.frustum.descriptor[{}]",
                                             i);
            }
            return true;
        }

        // Submit a single command buffer to the graphics queue and wait for it.
        // Mirrors depth_blit_pass's pattern: lets init() do GPU-side setup even
        // when called during an active frame (host-side
        // transitionImageLayoutImmediate refuses in that case).
        [[nodiscard]] bool runOneShotGraphics(const std::function<void(VkCommandBuffer)>& record) {
            VkCommandPoolCreateInfo pool_info{};
            pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            pool_info.queueFamilyIndex = graphics_queue_family;
            pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
            VkCommandPool pool = VK_NULL_HANDLE;
            VkResult result = vkCreateCommandPool(device, &pool_info, nullptr, &pool);
            if (result != VK_SUCCESS) {
                return reportVkFailure(
                    "vkCreateCommandPool(device, &pool_info, nullptr, &pool)",
                    result,
                    std::format("One-shot graphics command-pool creation failed (device={:#x}, queue_family={})",
                                vkHandleValue(device),
                                graphics_queue_family));
            }
            context->setDebugObjectName(VK_OBJECT_TYPE_COMMAND_POOL,
                                        pool,
                                        "viewport.oneshot.pool");
            VkCommandBufferAllocateInfo cb_alloc{};
            cb_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            cb_alloc.commandPool = pool;
            cb_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cb_alloc.commandBufferCount = 1;
            VkCommandBuffer cb = VK_NULL_HANDLE;
            result = vkAllocateCommandBuffers(device, &cb_alloc, &cb);
            if (result != VK_SUCCESS) {
                const std::string error = formatVkCheckFailure(
                    "vkAllocateCommandBuffers(device, &cb_alloc, &cb)",
                    result,
                    std::format("One-shot graphics command-buffer allocation failed (device={:#x}, command_pool={:#x}, requested_count=1)",
                                vkHandleValue(device),
                                vkHandleValue(pool)),
                    __FILE__,
                    __LINE__);
                vkDestroyCommandPool(device, pool, nullptr);
                return logVkFailure(error);
            }
            context->setDebugObjectName(VK_OBJECT_TYPE_COMMAND_BUFFER,
                                        cb,
                                        "viewport.oneshot.command");
            VkCommandBufferBeginInfo begin{};
            begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            result = vkBeginCommandBuffer(cb, &begin);
            if (result != VK_SUCCESS) {
                const std::string error = formatVkCheckFailure(
                    "vkBeginCommandBuffer(cb, &begin)",
                    result,
                    std::format("One-shot graphics command buffer did not enter recording state (command_buffer={:#x}, command_pool={:#x})",
                                vkHandleValue(cb),
                                vkHandleValue(pool)),
                    __FILE__,
                    __LINE__);
                vkFreeCommandBuffers(device, pool, 1, &cb);
                vkDestroyCommandPool(device, pool, nullptr);
                return logVkFailure(error);
            }
            record(cb);
            result = vkEndCommandBuffer(cb);
            if (result != VK_SUCCESS) {
                const std::string error = formatVkCheckFailure(
                    "vkEndCommandBuffer(cb)",
                    result,
                    std::format("One-shot graphics command buffer did not leave recording state (command_buffer={:#x}, command_pool={:#x})",
                                vkHandleValue(cb),
                                vkHandleValue(pool)),
                    __FILE__,
                    __LINE__);
                vkFreeCommandBuffers(device, pool, 1, &cb);
                vkDestroyCommandPool(device, pool, nullptr);
                return logVkFailure(error);
            }
            VkSubmitInfo submit{};
            submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit.commandBufferCount = 1;
            submit.pCommandBuffers = &cb;
            VkFenceCreateInfo fence_info{};
            fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            VkFence fence = VK_NULL_HANDLE;
            result = vkCreateFence(device, &fence_info, nullptr, &fence);
            std::string error;
            bool submitted = false;
            bool wait_ready = false;
            if (result != VK_SUCCESS) {
                error = formatVkCheckFailure(
                    "vkCreateFence(device, &fence_info, nullptr, &fence)",
                    result,
                    std::format("One-shot graphics submission fence creation failed (device={:#x}, command_buffer={:#x})",
                                vkHandleValue(device),
                                vkHandleValue(cb)),
                    __FILE__,
                    __LINE__);
            } else {
                context->setDebugObjectName(VK_OBJECT_TYPE_FENCE,
                                            fence,
                                            "viewport.oneshot.fence");
                if (graphics_queue == VK_NULL_HANDLE || cb == VK_NULL_HANDLE ||
                    fence == VK_NULL_HANDLE || submit.commandBufferCount != 1 ||
                    submit.pCommandBuffers == nullptr || submit.pCommandBuffers[0] != cb) {
                    error = std::format(
                        "One-shot graphics submit requires a non-null queue, one expected command buffer, and a non-null fence (queue={:#x}, command_buffer={:#x}, fence={:#x}, command_buffer_count={}, command_buffer_array={:#x}, submitted_command_buffer={:#x}) ({}:{})",
                        vkHandleValue(graphics_queue),
                        vkHandleValue(cb),
                        vkHandleValue(fence),
                        submit.commandBufferCount,
                        reinterpret_cast<std::uintptr_t>(submit.pCommandBuffers),
                        submit.pCommandBuffers != nullptr
                            ? vkHandleValue(submit.pCommandBuffers[0])
                            : 0,
                        __FILE__,
                        __LINE__);
                } else {
                    result = vkQueueSubmit(graphics_queue, 1, &submit, fence);
                }
                if (error.empty() && result != VK_SUCCESS) {
                    error = formatVkCheckFailure(
                        "vkQueueSubmit(graphics_queue, 1, &submit, fence)",
                        result,
                        std::format("One-shot graphics submission failed (queue={:#x}, command_buffer={:#x}, command_buffer_count=1, wait_semaphore_count=0, signal_semaphore_count=0, fence={:#x})",
                                    vkHandleValue(graphics_queue),
                                    vkHandleValue(cb),
                                    vkHandleValue(fence)),
                        __FILE__,
                        __LINE__);
                } else if (error.empty()) {
                    submitted = true;
                    lfs::rendering::WaitContext wait_ctx;
                    wait_ctx.fingerprint = "pass.viewport.oneshot_wait";
                    auto wait_outcome = lfs::rendering::wait_fence_bounded(
                        device,
                        fence,
                        std::stop_token{},
                        lfs::rendering::VulkanWaitPolicy{},
                        wait_ctx);
                    if (wait_outcome.has_value() &&
                        *wait_outcome == lfs::rendering::WaitOutcome::Ready) {
                        wait_ready = true;
                    } else {
                        error = std::format(
                            "wait_fence_bounded(pass.viewport.oneshot_wait) did not reach Ready "
                            "(device={:#x}, fence={:#x}, command_buffer={:#x}): {} ({}:{})",
                            vkHandleValue(device),
                            vkHandleValue(fence),
                            vkHandleValue(cb),
                            formatWaitFailure(wait_outcome),
                            __FILE__,
                            __LINE__);
                    }
                }
            }
            // AMB-4: free one-shot resources only when never submitted or wait Ready.
            if (fence != VK_NULL_HANDLE) {
                if (!submitted || wait_ready) {
                    vkDestroyFence(device, fence, nullptr);
                } else {
                    LOG_ERROR(
                        "Vulkan: retaining viewport one-shot fence after non-Ready wait "
                        "(fence={:#x}, command_buffer={:#x})",
                        vkHandleValue(fence),
                        vkHandleValue(cb));
                }
            }
            if (!submitted || wait_ready) {
                vkFreeCommandBuffers(device, pool, 1, &cb);
                vkDestroyCommandPool(device, pool, nullptr);
            } else {
                LOG_ERROR(
                    "Vulkan: retaining viewport one-shot command pool/buffer after non-Ready wait "
                    "(pool={:#x}, command_buffer={:#x})",
                    vkHandleValue(pool),
                    vkHandleValue(cb));
            }
            if (!error.empty()) {
                return logVkFailure(std::move(error));
            }
            return true;
        }

        [[nodiscard]] bool createShapeOverlayDescriptors() {
            VkSamplerCreateInfo sampler_info{};
            sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            sampler_info.magFilter = VK_FILTER_NEAREST;
            sampler_info.minFilter = VK_FILTER_NEAREST;
            sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            if (!vk_try_bool(
                    vkCreateSampler(device, &sampler_info, nullptr, &shape_overlay_depth_sampler),
                    "vkCreateSampler(device, &sampler_info, nullptr, &shape_overlay_depth_sampler)",
                    lfs::rendering::formatVulkanDiagnostic(
                                "Viewport shape-overlay depth sampler creation failed (device={:#x}, mag_filter={}, min_filter={}, address_mode={})",
                                vkHandleValue(device),
                                static_cast<int>(sampler_info.magFilter),
                                static_cast<int>(sampler_info.minFilter),
                                static_cast<int>(sampler_info.addressModeU)
                            ),
                    std::source_location::current())) {
                return false;
            }
            context->setDebugObjectName(VK_OBJECT_TYPE_SAMPLER,
                                        shape_overlay_depth_sampler,
                                        "viewport.shape_overlay.depth.sampler");

            // 1x1 dummy depth image bound while the real splat depth view is not
            // available. The depth_available push flag is 0 in that case so the
            // frag never samples it, but the descriptor still needs a valid view
            // in SHADER_READ_ONLY layout.
            VkImageCreateInfo img_info{};
            img_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            img_info.imageType = VK_IMAGE_TYPE_2D;
            img_info.format = VK_FORMAT_R32_SFLOAT;
            img_info.extent = {1, 1, 1};
            img_info.mipLevels = 1;
            img_info.arrayLayers = 1;
            img_info.samples = VK_SAMPLE_COUNT_1_BIT;
            img_info.tiling = VK_IMAGE_TILING_OPTIMAL;
            img_info.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
            img_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            img_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            VmaAllocationCreateInfo alloc{};
            alloc.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            VmaAllocationInfo allocation_info{};
            if (!vk_try_bool(
                    vmaCreateImage(allocator,
                               &img_info,
                               &alloc,
                               &shape_overlay_dummy_depth_image,
                               &shape_overlay_dummy_depth_alloc,
                               &allocation_info),
                    "vmaCreateImage(allocator, &img_info, &alloc, &shape_overlay_dummy_depth_image, &shape_overlay_dummy_depth_alloc, &allocation_info)",
                    lfs::rendering::formatVulkanDiagnostic(
                                "Viewport shape-overlay dummy-depth image allocation failed (allocator={:#x}, requested_extent={}x{}, format={}, usage={:#x})",
                                reinterpret_cast<std::uintptr_t>(allocator),
                                img_info.extent.width,
                                img_info.extent.height,
                                static_cast<int>(img_info.format),
                                static_cast<std::uint32_t>(img_info.usage)
                            ),
                    std::source_location::current())) {
                return false;
            }
            context->setDebugObjectName(VK_OBJECT_TYPE_IMAGE,
                                        shape_overlay_dummy_depth_image,
                                        "viewport.shape_overlay.depth.dummy");
            shape_overlay_dummy_depth_vram_label = "r32_float:1x1";
            lfs::diagnostics::VramProfiler::instance().recordCurrentBytes(
                "vulkan.viewport.shape_overlay_dummy_depth",
                shape_overlay_dummy_depth_vram_label,
                static_cast<std::size_t>(allocation_info.size));
            VkImageViewCreateInfo view_info{};
            view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            view_info.image = shape_overlay_dummy_depth_image;
            view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
            view_info.format = VK_FORMAT_R32_SFLOAT;
            view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            if (!vk_try_bool(
                    vkCreateImageView(device, &view_info, nullptr, &shape_overlay_dummy_depth_view),
                    "vkCreateImageView(device, &view_info, nullptr, &shape_overlay_dummy_depth_view)",
                    lfs::rendering::formatVulkanDiagnostic(
                                "Viewport shape-overlay dummy-depth image-view creation failed (device={:#x}, image={:#x}, format={}, aspect_mask={:#x})",
                                vkHandleValue(device),
                                vkHandleValue(view_info.image),
                                static_cast<int>(view_info.format),
                                static_cast<std::uint32_t>(view_info.subresourceRange.aspectMask)
                            ),
                    std::source_location::current())) {
                return false;
            }
            context->setDebugObjectName(VK_OBJECT_TYPE_IMAGE_VIEW,
                                        shape_overlay_dummy_depth_view,
                                        "viewport.shape_overlay.depth.dummy.view");
            const bool transitioned = runOneShotGraphics([this](VkCommandBuffer cb) {
                cmdImageBarrier2(cb, shape_overlay_dummy_depth_image, VK_IMAGE_ASPECT_COLOR_BIT,
                                 VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                 VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE,
                                 VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
            });
            if (!transitioned) {
                return false;
            }

            VkDescriptorSetLayoutBinding binding{};
            binding.binding = 0;
            binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            binding.descriptorCount = 1;
            binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            VkDescriptorSetLayoutCreateInfo layout_info{};
            layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            layout_info.bindingCount = 1;
            layout_info.pBindings = &binding;
            if (!vk_try_bool(
                    vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &shape_overlay_descriptor_layout),
                    "vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &shape_overlay_descriptor_layout)",
                    lfs::rendering::formatVulkanDiagnostic(
                                "Viewport shape-overlay descriptor-set layout creation failed (device={:#x}, binding_count={}, descriptor_type={})",
                                vkHandleValue(device),
                                layout_info.bindingCount,
                                static_cast<int>(binding.descriptorType)
                            ),
                    std::source_location::current())) {
                return false;
            }
            context->setDebugObjectName(VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                                        shape_overlay_descriptor_layout,
                                        "viewport.shape_overlay.descriptor.layout");

            const auto descriptor_count = static_cast<std::uint32_t>(frame_resources.size());
            VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, descriptor_count};
            VkDescriptorPoolCreateInfo desc_pool_info{};
            desc_pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            desc_pool_info.maxSets = descriptor_count;
            desc_pool_info.poolSizeCount = 1;
            desc_pool_info.pPoolSizes = &pool_size;
            if (!vk_try_bool(
                    vkCreateDescriptorPool(device, &desc_pool_info, nullptr, &shape_overlay_descriptor_pool),
                    "vkCreateDescriptorPool(device, &desc_pool_info, nullptr, &shape_overlay_descriptor_pool)",
                    lfs::rendering::formatVulkanDiagnostic(
                                "Viewport shape-overlay descriptor-pool creation failed (device={:#x}, frame_count={}, max_sets={}, descriptor_count={})",
                                vkHandleValue(device),
                                frame_resources.size(),
                                desc_pool_info.maxSets,
                                pool_size.descriptorCount
                            ),
                    std::source_location::current())) {
                return false;
            }
            context->setDebugObjectName(VK_OBJECT_TYPE_DESCRIPTOR_POOL,
                                        shape_overlay_descriptor_pool,
                                        "viewport.shape_overlay.descriptor.pool");

            std::vector<VkDescriptorSetLayout> layouts(frame_resources.size(), shape_overlay_descriptor_layout);
            std::vector<VkDescriptorSet> sets(frame_resources.size(), VK_NULL_HANDLE);
            VkDescriptorSetAllocateInfo alloc_info{};
            alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            alloc_info.descriptorPool = shape_overlay_descriptor_pool;
            alloc_info.descriptorSetCount = descriptor_count;
            alloc_info.pSetLayouts = layouts.data();
            if (!vk_try_bool(
                    vkAllocateDescriptorSets(device, &alloc_info, sets.data()),
                    "vkAllocateDescriptorSets(device, &alloc_info, sets.data())",
                    lfs::rendering::formatVulkanDiagnostic(
                                "Viewport shape-overlay descriptor-set allocation failed (device={:#x}, descriptor_pool={:#x}, descriptor_layout={:#x}, requested_count={})",
                                vkHandleValue(device),
                                vkHandleValue(shape_overlay_descriptor_pool),
                                vkHandleValue(shape_overlay_descriptor_layout),
                                alloc_info.descriptorSetCount
                            ),
                    std::source_location::current())) {
                return false;
            }
            for (std::size_t i = 0; i < frame_resources.size(); ++i) {
                frame_resources[i].shape_overlay_descriptor_set = sets[i];
                context->setDebugObjectNamef(VK_OBJECT_TYPE_DESCRIPTOR_SET,
                                             sets[i],
                                             "viewport.shape_overlay.descriptor[{}]",
                                             i);
                bindShapeOverlayDepth(frame_resources[i], shape_overlay_dummy_depth_view);
            }
            return true;
        }

        void bindShapeOverlayDepth(FrameResources& frame, VkImageView view) {
            if (view == VK_NULL_HANDLE) {
                view = shape_overlay_dummy_depth_view;
            }
            if (frame.bound_shape_overlay_depth_view == view) {
                return;
            }
            VkDescriptorImageInfo di{};
            di.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            di.imageView = view;
            di.sampler = shape_overlay_depth_sampler;
            VkWriteDescriptorSet w{};
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet = frame.shape_overlay_descriptor_set;
            w.dstBinding = 0;
            w.descriptorCount = 1;
            w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w.pImageInfo = &di;
            vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);
            frame.bound_shape_overlay_depth_view = view;
        }

        [[nodiscard]] bool createQuadBuffer() {
            if (!createBuffer(sizeof(Vertex) * 6,
                              VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                              quad_buffer,
                              quad_allocation)) {
                return false;
            }
            context->setDebugObjectName(VK_OBJECT_TYPE_BUFFER,
                                        quad_buffer,
                                        "viewport.screen_quad.vertex");
            return true;
        }

        enum class PipelineVertexLayout {
            ScreenQuad,
            PositionOnly,
            ColorOverlay,
            TexturedOverlay,
            ShapeOverlay,
            Procedural
        };

        [[nodiscard]] bool createPipeline(const std::span<const std::uint32_t> vertex_spv,
                                          const std::span<const std::uint32_t> fragment_spv,
                                          const char* label,
                                          VkDescriptorSetLayout descriptor_layout,
                                          const VkPushConstantRange* push_constant,
                                          bool enable_blend,
                                          PipelineVertexLayout vertex_layout,
                                          VkPipelineLayout& pipeline_layout,
                                          VkPipeline& pipeline,
                                          VkDescriptorSetLayout extra_descriptor_layout = VK_NULL_HANDLE) {
            VkShaderModule vertex_module = lfs::vis::createShaderModule(device, vertex_spv, "Viewport");
            VkShaderModule fragment_module = lfs::vis::createShaderModule(device, fragment_spv, "Viewport");
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

            std::array<VkVertexInputAttributeDescription, 7> attributes{};
            VkVertexInputBindingDescription binding{};
            VkPipelineVertexInputStateCreateInfo vertex_input{};
            vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
            if (vertex_layout != PipelineVertexLayout::Procedural) {
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
                    attributes[6].location = 6;
                    attributes[6].binding = 0;
                    attributes[6].format = VK_FORMAT_R32_SFLOAT;
                    attributes[6].offset = offsetof(VulkanViewportShapeOverlayVertex, view_depth);
                } else if (vertex_layout == PipelineVertexLayout::TexturedOverlay) {
                    attributes[2].location = 2;
                    attributes[2].binding = 0;
                    attributes[2].format = VK_FORMAT_R32_SFLOAT;
                    attributes[2].offset = offsetof(VulkanViewportTexturedOverlayVertex, view_depth);
                }

                vertex_input.vertexBindingDescriptionCount = 1;
                vertex_input.pVertexBindingDescriptions = &binding;
                vertex_input.vertexAttributeDescriptionCount =
                    vertex_layout == PipelineVertexLayout::ShapeOverlay      ? 7u
                    : vertex_layout == PipelineVertexLayout::TexturedOverlay ? 3u
                    : vertex_layout == PipelineVertexLayout::PositionOnly    ? 1u
                                                                             : 2u;
                vertex_input.pVertexAttributeDescriptions = attributes.data();
            }

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
            std::array<VkDescriptorSetLayout, 2> set_layouts{descriptor_layout, extra_descriptor_layout};
            if (descriptor_layout != VK_NULL_HANDLE && extra_descriptor_layout != VK_NULL_HANDLE) {
                layout_info.setLayoutCount = 2;
                layout_info.pSetLayouts = set_layouts.data();
            } else if (descriptor_layout != VK_NULL_HANDLE) {
                layout_info.setLayoutCount = 1;
                layout_info.pSetLayouts = &descriptor_layout;
            }
            if (push_constant) {
                layout_info.pushConstantRangeCount = 1;
                layout_info.pPushConstantRanges = push_constant;
            }
            const VkResult layout_result =
                vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout);
            if (layout_result != VK_SUCCESS) {
                vkDestroyShaderModule(device, vertex_module, nullptr);
                vkDestroyShaderModule(device, fragment_module, nullptr);
                return reportVkFailure(
                    "vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout)",
                    layout_result,
                    std::format("Viewport pipeline-layout creation failed (label='{}', device={:#x}, descriptor_layout={:#x}, extra_descriptor_layout={:#x}, set_layout_count={}, push_constant_bytes={})",
                                label,
                                vkHandleValue(device),
                                vkHandleValue(descriptor_layout),
                                vkHandleValue(extra_descriptor_layout),
                                layout_info.setLayoutCount,
                                push_constant != nullptr ? push_constant->size : 0));
            }
            context->setDebugObjectNamef(VK_OBJECT_TYPE_PIPELINE_LAYOUT,
                                         pipeline_layout,
                                         "viewport.{}.pipeline.layout",
                                         label);

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

            const VkResult pipeline_result =
                vkCreateGraphicsPipelines(device, pipeline_cache, 1, &pipeline_info, nullptr, &pipeline);
            vkDestroyShaderModule(device, vertex_module, nullptr);
            vkDestroyShaderModule(device, fragment_module, nullptr);
            if (pipeline_result != VK_SUCCESS) {
                return reportVkFailure(
                    "vkCreateGraphicsPipelines(device, pipeline_cache, 1, &pipeline_info, nullptr, &pipeline)",
                    pipeline_result,
                    std::format("Viewport graphics pipeline creation failed (device={:#x}, pipeline_cache={:#x}, pipeline_layout={:#x}, color_format={}, depth_stencil_format={})",
                                vkHandleValue(device),
                                vkHandleValue(pipeline_cache),
                                vkHandleValue(pipeline_layout),
                                static_cast<int>(color_format),
                                static_cast<int>(depth_stencil_format)));
            }
            context->setDebugObjectNamef(VK_OBJECT_TYPE_PIPELINE,
                                         pipeline,
                                         "viewport.{}.pipeline",
                                         label);
            return true;
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
            VkPushConstantRange shape_overlay_push{};
            shape_overlay_push.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            shape_overlay_push.offset = 0;
            shape_overlay_push.size = sizeof(ShapeOverlayPush);
            VkPushConstantRange frustum_push{};
            frustum_push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
            frustum_push.offset = 0;
            frustum_push.size = sizeof(FrustumPush);
            using namespace viewport_shaders;

            return createPipeline(kScreenQuadVertSpv, kSceneFragSpv, "scene",
                                  scene_descriptor_layout, nullptr, true, PipelineVertexLayout::ScreenQuad,
                                  scene_pipeline_layout, scene_pipeline) &&
                   createPipeline(kScreenQuadVertSpv, kVignetteFragSpv, "vignette",
                                  VK_NULL_HANDLE, &vignette_push, true, PipelineVertexLayout::ScreenQuad,
                                  vignette_pipeline_layout, vignette_pipeline) &&
                   createPipeline(kGridVertSpv, kGridFragSpv, "grid",
                                  grid_descriptor_layout, &grid_push, true, PipelineVertexLayout::PositionOnly,
                                  grid_pipeline_layout, grid_pipeline) &&
                   createPipeline(kOverlayVertSpv, kOverlayFragSpv, "overlay",
                                  VK_NULL_HANDLE, nullptr, true, PipelineVertexLayout::ColorOverlay,
                                  overlay_pipeline_layout, overlay_pipeline) &&
                   createPipeline(kShapeOverlayVertSpv, kShapeOverlayFragSpv, "shape_overlay",
                                  shape_overlay_descriptor_layout, &shape_overlay_push, true,
                                  PipelineVertexLayout::ShapeOverlay,
                                  shape_overlay_pipeline_layout, shape_overlay_pipeline) &&
                   createPipeline(kTexturedOverlayVertSpv, kTexturedOverlayFragSpv, "textured_overlay",
                                  scene_descriptor_layout, &textured_overlay_push, true,
                                  PipelineVertexLayout::TexturedOverlay,
                                  textured_overlay_pipeline_layout, textured_overlay_pipeline,
                                  shape_overlay_descriptor_layout) &&
                   createPipeline(kPivotVertSpv, kPivotFragSpv, "pivot",
                                  VK_NULL_HANDLE, &pivot_push, true, PipelineVertexLayout::Procedural,
                                  pivot_pipeline_layout, pivot_pipeline) &&
                   createPipeline(kFrustumVertSpv, kShapeOverlayFragSpv, "frustum",
                                  shape_overlay_descriptor_layout, &frustum_push, true,
                                  PipelineVertexLayout::Procedural,
                                  frustum_pipeline_layout, frustum_pipeline,
                                  frustum_descriptor_layout);
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
            const VkBuffer previous_buffer = frame.overlay.buffer;
            if (!ensureDynamicBuffer(frame.overlay,
                                     params.overlay_triangles.size(),
                                     sizeof(VulkanViewportOverlayVertex),
                                     256,
                                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT)) {
                return;
            }
            if (frame.overlay.buffer != previous_buffer) {
                context->setDebugObjectNamef(VK_OBJECT_TYPE_BUFFER,
                                             frame.overlay.buffer,
                                             "viewport.overlay[{}].vertex[{}]",
                                             params.frame_slot,
                                             frame.overlay.capacity);
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
                                      DynamicBuffer& resource,
                                      const std::size_t frame_slot,
                                      const std::string_view label) {
            resource.count = 0;
            if (vertices.empty()) {
                return;
            }
            const VkBuffer previous_buffer = resource.buffer;
            if (!ensureDynamicBuffer(resource,
                                     vertices.size(),
                                     sizeof(VulkanViewportShapeOverlayVertex),
                                     256,
                                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT)) {
                return;
            }
            if (resource.buffer != previous_buffer) {
                context->setDebugObjectNamef(VK_OBJECT_TYPE_BUFFER,
                                             resource.buffer,
                                             "viewport.{}[{}].vertex[{}]",
                                             label,
                                             frame_slot,
                                             resource.capacity);
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
            const VkBuffer previous_buffer = frame.textured_overlay.buffer;
            if (!ensureDynamicBuffer(frame.textured_overlay,
                                     vertex_count,
                                     sizeof(VulkanViewportTexturedOverlayVertex),
                                     64,
                                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT)) {
                return;
            }
            if (frame.textured_overlay.buffer != previous_buffer) {
                context->setDebugObjectNamef(VK_OBJECT_TYPE_BUFFER,
                                             frame.textured_overlay.buffer,
                                             "viewport.textured_overlay[{}].vertex[{}]",
                                             params.frame_slot,
                                             frame.textured_overlay.capacity);
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

            // The guide-panel producer emits one grid normally and two for an
            // independent split view. Reserve both slots up front so toggling
            // split view does not replace a descriptor-backed buffer mid-run.
            std::size_t capacity = 2;
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
            const std::size_t frame_slot = static_cast<std::size_t>(&frame - frame_resources.data());
            context->setDebugObjectNamef(VK_OBJECT_TYPE_BUFFER,
                                         frame.grid_uniform.buffer,
                                         "viewport.grid[{}].uniform[{}]",
                                         frame_slot,
                                         capacity);
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

        void updateFrustumDescriptor(FrameResources& frame, const VkDeviceSize range) const {
            if (frame.frustum_descriptor_set == VK_NULL_HANDLE ||
                frame.frustum_instances.buffer == VK_NULL_HANDLE) {
                return;
            }
            VkDescriptorBufferInfo buffer_info{};
            buffer_info.buffer = frame.frustum_instances.buffer;
            buffer_info.offset = 0;
            buffer_info.range = range;
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = frame.frustum_descriptor_set;
            write.dstBinding = 0;
            write.descriptorCount = 1;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.pBufferInfo = &buffer_info;
            vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
        }

        [[nodiscard]] bool ensureFrustumInstanceBuffer(FrameResources& frame, const std::size_t instance_count) {
            if (instance_count == 0) {
                frame.frustum_instances.count = 0;
                return true;
            }
            if (frame.frustum_instances.buffer != VK_NULL_HANDLE &&
                frame.frustum_instances.capacity >= instance_count) {
                return true;
            }

            std::size_t capacity = 1;
            while (capacity < instance_count) {
                capacity *= 2;
            }
            const VkDeviceSize bytes =
                static_cast<VkDeviceSize>(sizeof(VulkanViewportFrustumInstance) * capacity);
            destroyDynamicBuffer(frame.frustum_instances);
            if (!createBuffer(bytes,
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                              frame.frustum_instances.buffer,
                              frame.frustum_instances.allocation)) {
                frame.frustum_instances = {};
                return false;
            }
            frame.frustum_instances.capacity = capacity;
            const std::size_t frame_slot = static_cast<std::size_t>(&frame - frame_resources.data());
            context->setDebugObjectNamef(VK_OBJECT_TYPE_BUFFER,
                                         frame.frustum_instances.buffer,
                                         "viewport.frustum[{}].instance[{}]",
                                         frame_slot,
                                         capacity);
            updateFrustumDescriptor(frame, bytes);
            return true;
        }

        void updateFrustumInstances(const VulkanViewportPassParams& params) {
            auto& frame = resourcesForFrame(params.frame_slot);
            if (params.frustum_instances.empty()) {
                frame.frustum_instances.count = 0;
                return;
            }
            if (!ensureFrustumInstanceBuffer(frame, params.frustum_instances.size())) {
                return;
            }
            const VkDeviceSize bytes = static_cast<VkDeviceSize>(
                sizeof(VulkanViewportFrustumInstance) * params.frustum_instances.size());
            if (writeAllocation(frame.frustum_instances.allocation, params.frustum_instances.data(), bytes)) {
                frame.frustum_instances.count = static_cast<std::uint32_t>(
                    std::min<std::size_t>(params.frustum_instances.size(), std::numeric_limits<std::uint32_t>::max()));
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
            updateFrustumInstances(params);
            updateTexturedOverlayBuffer(frame, params);
            updateOverlayBuffer(frame, params);
            updateShapeOverlayBuffer(params.shape_overlay_triangles,
                                     frame.shape_overlay,
                                     params.frame_slot,
                                     "shape_overlay");
            updateShapeOverlayBuffer(params.ui_shape_overlay_triangles,
                                     frame.ui_shape_overlay,
                                     params.frame_slot,
                                     "ui_shape_overlay");
            uploadSceneImage(params);

            VulkanMeshPassParams mesh_params{
                .view_projection = params.mesh_view_projection,
                .camera_position = params.mesh_camera_position,
                .items = params.mesh_items,
                .frame_slot = params.frame_slot,
                .draw_group_count = std::max<std::size_t>(1, params.mesh_panels.size()),
            };
            mesh_pass.prepare(*context, mesh_params);
            environment_pass.prepare(params.environment, params.frame_slot);
            depth_blit_pass.prepare(params.depth_blit, params.frame_slot);
            split_view_pass.prepare(params.split_view, params.frame_slot);
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
                                const VulkanViewportPassParams& params) const {
            const auto& frame = resourcesForFrame(params.frame_slot);
            LFS_VK_DEBUG_ASSERT(
                frame.grid_uniform.count > 0 && grid_pipeline != VK_NULL_HANDLE &&
                    frame.grid_descriptor_set != VK_NULL_HANDLE && frame.grid_uniform.buffer != VK_NULL_HANDLE,
                "Viewport grid pass requires uploaded uniforms, pipeline, descriptor set, and buffer (frame_slot={}, uniform_count={}, pipeline={:#x}, descriptor_set={:#x}, uniform_buffer={:#x})",
                params.frame_slot,
                frame.grid_uniform.count,
                vkHandleValue(grid_pipeline),
                vkHandleValue(frame.grid_descriptor_set),
                vkHandleValue(frame.grid_uniform.buffer));

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
        }

        void recordShapeOverlays(VkCommandBuffer command_buffer,
                                 const DynamicBuffer& resource,
                                 const FrameResources& frame,
                                 const ShapeOverlayPush& push) const {
            LFS_VK_DEBUG_ASSERT(
                resource.count > 0 && resource.buffer != VK_NULL_HANDLE &&
                    shape_overlay_pipeline != VK_NULL_HANDLE,
                "Viewport shape-overlay pass requires vertices, a vertex buffer, and a pipeline (vertex_count={}, vertex_buffer={:#x}, pipeline={:#x}, descriptor_set={:#x})",
                resource.count,
                vkHandleValue(resource.buffer),
                vkHandleValue(shape_overlay_pipeline),
                vkHandleValue(frame.shape_overlay_descriptor_set));
            const VkDeviceSize offset = 0;
            vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shape_overlay_pipeline);
            if (frame.shape_overlay_descriptor_set != VK_NULL_HANDLE) {
                vkCmdBindDescriptorSets(command_buffer,
                                        VK_PIPELINE_BIND_POINT_GRAPHICS,
                                        shape_overlay_pipeline_layout,
                                        0,
                                        1,
                                        &frame.shape_overlay_descriptor_set,
                                        0,
                                        nullptr);
            }
            vkCmdPushConstants(command_buffer,
                               shape_overlay_pipeline_layout,
                               VK_SHADER_STAGE_FRAGMENT_BIT,
                               0,
                               sizeof(ShapeOverlayPush),
                               &push);
            vkCmdBindVertexBuffers(command_buffer, 0, 1, &resource.buffer, &offset);
            vkCmdDraw(command_buffer, resource.count, 1, 0, 0);
        }

        // --- Sub-pass record steps -----------------------------------------------------------
        // One method per graph pass. The graph calls these only when the pass's active() gate holds
        // and rebinds shared viewport/quad state between passes, so each asserts its precondition and
        // does not restore shared state on exit.

        [[nodiscard]] bool sceneSplitActive(const VulkanViewportPassParams& params) const {
            return params.split_view.enabled && split_view_pass.ready(params.frame_slot);
        }

        struct OverlayVertexSplit {
            std::uint32_t base = 0;
            std::uint32_t post_ui = 0;
        };

        // Splits the shared overlay buffer into the world-space (pre-UI) head and the post-UI tail.
        [[nodiscard]] OverlayVertexSplit overlaySplit(const FrameResources& frame,
                                                      const VulkanViewportPassParams& params) const {
            const std::uint32_t post_ui = std::min(params.post_ui_overlay_vertex_count, frame.overlay.count);
            return {frame.overlay.count - post_ui, post_ui};
        }

        void recordEnvironmentPass(VkCommandBuffer command_buffer, const FramebufferRect& rect,
                                   const VulkanViewportPassParams& params) {
            LFS_VK_DEBUG_ASSERT(
                params.environment.enabled && environment_pass.hasTexture(params.frame_slot),
                "Viewport environment pass must be enabled and have a texture for its frame slot (frame_slot={}, enabled={}, texture_ready={})",
                params.frame_slot,
                params.environment.enabled,
                environment_pass.hasTexture(params.frame_slot));
            environment_pass.record(command_buffer,
                                    VkRect2D{
                                        .offset = {rect.x, rect.y},
                                        .extent = {static_cast<std::uint32_t>(rect.width),
                                                   static_cast<std::uint32_t>(rect.height)}},
                                    params.environment,
                                    params.frame_slot);
        }

        void recordScenePass(VkCommandBuffer command_buffer, const FramebufferRect& rect,
                             const VulkanViewportPassParams& params) {
            auto& frame = resourcesForFrame(params.frame_slot);
            const bool has_scene =
                scene_image_uploader.hasImage() &&
                frame.scene_descriptor_set != VK_NULL_HANDLE && scene_pipeline != VK_NULL_HANDLE;
            const bool split_active = sceneSplitActive(params);
            LFS_VK_DEBUG_ASSERT(
                split_active || has_scene,
                "Viewport scene pass requires either a ready split view or a complete scene-image binding (frame_slot={}, split_enabled={}, split_ready={}, scene_image_ready={}, descriptor_set={:#x}, pipeline={:#x})",
                params.frame_slot,
                params.split_view.enabled,
                split_active,
                scene_image_uploader.hasImage(),
                vkHandleValue(frame.scene_descriptor_set),
                vkHandleValue(scene_pipeline));
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
                split_view_pass.record(command_buffer, panel_rect, adjusted, params.frame_slot);
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
        }

        void recordDepthBlitPass(VkCommandBuffer command_buffer, const FramebufferRect& rect,
                                 const VulkanViewportPassParams& params) {
            LFS_VK_DEBUG_ASSERT(
                !sceneSplitActive(params) && !params.mesh_items.empty() &&
                    depth_blit_pass.hasDepth(params.frame_slot),
                "Viewport depth-blit pass requires non-split mesh rendering with a ready depth image (frame_slot={}, split_active={}, mesh_items={}, depth_ready={})",
                params.frame_slot,
                sceneSplitActive(params),
                params.mesh_items.size(),
                depth_blit_pass.hasDepth(params.frame_slot));
            const VkRect2D depth_rect{
                .offset = {rect.x, rect.y},
                .extent = {static_cast<std::uint32_t>(rect.width),
                           static_cast<std::uint32_t>(rect.height)},
            };
            depth_blit_pass.record(command_buffer, depth_rect, params.depth_blit, params.frame_slot);
        }

        void recordMeshPass(VkCommandBuffer command_buffer, const FramebufferRect& rect,
                            const VulkanViewportPassParams& params) {
            LFS_VK_DEBUG_ASSERT(
                !params.mesh_items.empty(),
                "Viewport mesh pass requires at least one mesh draw item (frame_slot={}, mesh_items={}, mesh_panels={}, split_enabled={})",
                params.frame_slot,
                params.mesh_items.size(),
                params.mesh_panels.size(),
                params.split_view.enabled);
            const bool split_active = sceneSplitActive(params);
            const bool split_mesh_panels_active = split_active && !params.mesh_panels.empty();
            VulkanMeshPassParams mesh_params{
                .items = params.mesh_items,
                .frame_slot = params.frame_slot,
                .draw_group_count = std::max<std::size_t>(1, params.mesh_panels.size()),
            };
            if (split_mesh_panels_active) {
                const int rect_min_x = rect.x;
                const int rect_max_x = rect.x + static_cast<int>(rect.width);
                for (std::size_t panel_index = 0; panel_index < params.mesh_panels.size(); ++panel_index) {
                    const auto& panel = params.mesh_panels[panel_index];
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
                    mesh_params.draw_group = panel_index;
                    const VkRect2D mesh_rect{
                        .offset = {x0, rect.y},
                        .extent = {static_cast<std::uint32_t>(x1 - x0),
                                   static_cast<std::uint32_t>(rect.height)},
                    };
                    mesh_pass.record(command_buffer, mesh_rect, mesh_params);
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
            }
        }

        void recordTexturedOverlayPass(const ViewportRecordContext& ctx,
                                       const VulkanViewportPassParams& params) {
            const VkCommandBuffer command_buffer = ctx.cmd;
            auto& frame = resourcesForFrame(params.frame_slot);
            LFS_VK_DEBUG_ASSERT(
                frame.textured_overlay.count > 0 && textured_overlay_pipeline != VK_NULL_HANDLE &&
                    frame.textured_overlay.buffer != VK_NULL_HANDLE && !params.textured_overlays.empty(),
                "Viewport textured-overlay pass requires vertices, overlays, a vertex buffer, and a pipeline (frame_slot={}, vertex_count={}, overlay_count={}, vertex_buffer={:#x}, pipeline={:#x})",
                params.frame_slot,
                frame.textured_overlay.count,
                params.textured_overlays.size(),
                vkHandleValue(frame.textured_overlay.buffer),
                vkHandleValue(textured_overlay_pipeline));
            const VkDeviceSize offset = 0;
            vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, textured_overlay_pipeline);
            vkCmdBindVertexBuffers(command_buffer, 0, 1, &frame.textured_overlay.buffer, &offset);
            if (frame.shape_overlay_descriptor_set != VK_NULL_HANDLE) {
                vkCmdBindDescriptorSets(command_buffer,
                                        VK_PIPELINE_BIND_POINT_GRAPHICS,
                                        textured_overlay_pipeline_layout,
                                        1,
                                        1,
                                        &frame.shape_overlay_descriptor_set,
                                        0,
                                        nullptr);
            }
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
                push.viewport_rect = ctx.viewport_rect_push;
                push.depth_params = ctx.world_depth_params_push;
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
        }

        void recordBaseOverlayPass(VkCommandBuffer command_buffer, const VulkanViewportPassParams& params) {
            auto& frame = resourcesForFrame(params.frame_slot);
            const std::uint32_t overlay_vertices = overlaySplit(frame, params).base;
            LFS_VK_DEBUG_ASSERT(
                overlay_vertices > 0 && overlay_pipeline != VK_NULL_HANDLE &&
                    frame.overlay.buffer != VK_NULL_HANDLE,
                "Viewport base-overlay pass requires vertices, a vertex buffer, and a pipeline (frame_slot={}, base_vertices={}, total_vertices={}, vertex_buffer={:#x}, pipeline={:#x})",
                params.frame_slot,
                overlay_vertices,
                frame.overlay.count,
                vkHandleValue(frame.overlay.buffer),
                vkHandleValue(overlay_pipeline));
            const VkDeviceSize offset = 0;
            vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, overlay_pipeline);
            vkCmdBindVertexBuffers(command_buffer, 0, 1, &frame.overlay.buffer, &offset);
            vkCmdDraw(command_buffer, overlay_vertices, 1, 0, 0);
        }

        void recordWorldShapePass(const ViewportRecordContext& ctx,
                                  const VulkanViewportPassParams& params) {
            auto& frame = resourcesForFrame(params.frame_slot);
            const ShapeOverlayPush world_shape_overlay_push{
                .viewport_rect = ctx.viewport_rect_push,
                .params = ctx.world_depth_params_push};
            recordShapeOverlays(ctx.cmd, frame.shape_overlay, frame, world_shape_overlay_push);
        }

        void recordFrustumPass(const ViewportRecordContext& ctx,
                               const VulkanViewportPassParams& params) {
            const VkCommandBuffer command_buffer = ctx.cmd;
            auto& frame = resourcesForFrame(params.frame_slot);
            LFS_VK_DEBUG_ASSERT(
                frame.frustum_instances.count > 0 && frustum_pipeline != VK_NULL_HANDLE &&
                    frame.frustum_descriptor_set != VK_NULL_HANDLE &&
                    frame.shape_overlay_descriptor_set != VK_NULL_HANDLE &&
                    frame.frustum_instances.buffer != VK_NULL_HANDLE,
                "Viewport frustum pass requires instances, pipeline, both descriptor sets, and an instance buffer (frame_slot={}, instance_count={}, batch_count={}, pipeline={:#x}, frustum_descriptor_set={:#x}, shape_descriptor_set={:#x}, instance_buffer={:#x})",
                params.frame_slot,
                frame.frustum_instances.count,
                params.frustum_batches.size(),
                vkHandleValue(frustum_pipeline),
                vkHandleValue(frame.frustum_descriptor_set),
                vkHandleValue(frame.shape_overlay_descriptor_set),
                vkHandleValue(frame.frustum_instances.buffer));
            vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, frustum_pipeline);
            const std::array<VkDescriptorSet, 2> sets{
                frame.shape_overlay_descriptor_set, frame.frustum_descriptor_set};
            vkCmdBindDescriptorSets(command_buffer,
                                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    frustum_pipeline_layout,
                                    0,
                                    static_cast<std::uint32_t>(sets.size()),
                                    sets.data(),
                                    0,
                                    nullptr);
            for (const auto& batch : params.frustum_batches) {
                if (batch.instance_count == 0 ||
                    batch.first_instance + batch.instance_count > frame.frustum_instances.count) {
                    continue;
                }
                const FramebufferRect rect =
                    toFramebufferRect(params, batch.viewport_pos, batch.viewport_size, ctx.extent);
                if (rect.width == 0 || rect.height == 0) {
                    continue;
                }
                bindViewport(command_buffer, rect);
                FrustumPush push{};
                const float projection_mode = batch.equirectangular ? 2.0f : (batch.orthographic ? 1.0f : 0.0f);
                push.viewport_rect = glm::vec4(static_cast<float>(rect.x),
                                               static_cast<float>(rect.y),
                                               static_cast<float>(rect.width),
                                               static_cast<float>(rect.height));
                push.params = glm::vec4(ctx.world_depth_params_push.x,
                                        ctx.world_depth_params_push.y,
                                        kFrustumLineThickness,
                                        projection_mode);
                push.view = batch.view;
                push.viewport_panel = glm::vec4(batch.viewport_pos, batch.viewport_size);
                push.projection = glm::vec4(batch.render_size, batch.focal_x, batch.focal_y);
                vkCmdPushConstants(command_buffer,
                                   frustum_pipeline_layout,
                                   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                   0,
                                   sizeof(push),
                                   &push);
                vkCmdDraw(command_buffer, kFrustumVertexCount, batch.instance_count, 0, batch.first_instance);
            }
        }

        void recordPivotPass(VkCommandBuffer command_buffer, const VulkanViewportPassParams& params) {
            LFS_VK_DEBUG_ASSERT(
                !params.pivot_overlays.empty() && pivot_pipeline != VK_NULL_HANDLE,
                "Viewport pivot pass requires at least one pivot and a valid pipeline (frame_slot={}, pivot_count={}, pipeline={:#x})",
                params.frame_slot,
                params.pivot_overlays.size(),
                vkHandleValue(pivot_pipeline));
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

        void recordVignettePass(VkCommandBuffer command_buffer, const FramebufferRect& rect,
                                const VulkanViewportPassParams& params) {
            LFS_VK_DEBUG_ASSERT(
                params.vignette_enabled && vignette_pipeline != VK_NULL_HANDLE,
                "Viewport vignette pass must be enabled and have a valid pipeline (frame_slot={}, enabled={}, pipeline={:#x}, intensity={}, radius={}, softness={})",
                params.frame_slot,
                params.vignette_enabled,
                vkHandleValue(vignette_pipeline),
                params.vignette_intensity,
                params.vignette_radius,
                params.vignette_softness);
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

        void recordUiShapePass(const ViewportRecordContext& ctx,
                               const VulkanViewportPassParams& params) {
            auto& frame = resourcesForFrame(params.frame_slot);
            // depth_available = 0 → UI shapes (gizmos, pivot) always render in front.
            const ShapeOverlayPush ui_shape_overlay_push{.viewport_rect = ctx.viewport_rect_push};
            recordShapeOverlays(ctx.cmd, frame.ui_shape_overlay, frame, ui_shape_overlay_push);
        }

        void recordPostUiOverlayPass(VkCommandBuffer command_buffer, const VulkanViewportPassParams& params) {
            auto& frame = resourcesForFrame(params.frame_slot);
            const OverlayVertexSplit split = overlaySplit(frame, params);
            LFS_VK_DEBUG_ASSERT(
                split.post_ui > 0 && overlay_pipeline != VK_NULL_HANDLE &&
                    frame.overlay.buffer != VK_NULL_HANDLE,
                "Viewport post-UI overlay pass requires trailing vertices, a vertex buffer, and a pipeline (frame_slot={}, post_ui_vertices={}, base_vertices={}, total_vertices={}, vertex_buffer={:#x}, pipeline={:#x})",
                params.frame_slot,
                split.post_ui,
                split.base,
                frame.overlay.count,
                vkHandleValue(frame.overlay.buffer),
                vkHandleValue(overlay_pipeline));
            const VkDeviceSize offset = 0;
            vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, overlay_pipeline);
            vkCmdBindVertexBuffers(command_buffer, 0, 1, &frame.overlay.buffer, &offset);
            vkCmdDraw(command_buffer, split.post_ui, 1, split.base, 0);
        }

        void record(VkCommandBuffer command_buffer,
                    const VkExtent2D extent,
                    const VulkanViewportPassParams& params) {
            auto& frame = resourcesForFrame(params.frame_slot);
            const FramebufferRect rect = toFramebufferRect(params, extent);
            if (rect.width == 0 || rect.height == 0 || quad_buffer == VK_NULL_HANDLE) {
                return;
            }
            const ScopedNvtxRange viewport_range{"viewport_pass_graph", 0xFFECEFF1};
            const glm::vec4 viewport_rect_push{
                static_cast<float>(rect.x),
                static_cast<float>(rect.y),
                static_cast<float>(rect.width),
                static_cast<float>(rect.height)};
            const bool depth_available = depth_blit_pass.hasDepth(params.frame_slot);
            const glm::vec4 world_depth_params_push{
                depth_available ? 1.0f : 0.0f,
                params.depth_blit.flip_y ? 1.0f : 0.0f,
                0.0f, 0.0f};
            bindShapeOverlayDepth(
                frame,
                depth_available ? depth_blit_pass.depthView(params.frame_slot) : VK_NULL_HANDLE);

            bindViewport(command_buffer, rect);
            bindQuad(command_buffer);
            clearViewport(command_buffer, rect, params.background_color);

            ViewportRecordContext rc{};
            rc.cmd = command_buffer;
            rc.extent = extent;
            rc.rect_x = rect.x;
            rc.rect_y = rect.y;
            rc.rect_w = rect.width;
            rc.rect_h = rect.height;
            rc.depth_available = depth_available;
            rc.frame_slot = params.frame_slot;
            rc.viewport_rect_push = viewport_rect_push;
            rc.world_depth_params_push = world_depth_params_push;
            graph_.record(rc, params, [this, command_buffer, rect]() {
                bindViewport(command_buffer, rect);
                bindQuad(command_buffer);
            });
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
                if (frustum_pipeline != VK_NULL_HANDLE)
                    vkDestroyPipeline(device, frustum_pipeline, nullptr);
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
                if (frustum_pipeline_layout != VK_NULL_HANDLE)
                    vkDestroyPipelineLayout(device, frustum_pipeline_layout, nullptr);
                if (quad_buffer != VK_NULL_HANDLE)
                    vmaDestroyBuffer(allocator, quad_buffer, quad_allocation);
                for (auto& frame : frame_resources) {
                    destroyDynamicBuffer(frame.overlay);
                    destroyDynamicBuffer(frame.shape_overlay);
                    destroyDynamicBuffer(frame.ui_shape_overlay);
                    destroyDynamicBuffer(frame.textured_overlay);
                    destroyDynamicBuffer(frame.grid_uniform);
                    destroyDynamicBuffer(frame.frustum_instances);
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
                if (frustum_descriptor_pool != VK_NULL_HANDLE)
                    vkDestroyDescriptorPool(device, frustum_descriptor_pool, nullptr);
                if (frustum_descriptor_layout != VK_NULL_HANDLE)
                    vkDestroyDescriptorSetLayout(device, frustum_descriptor_layout, nullptr);
                if (shape_overlay_descriptor_pool != VK_NULL_HANDLE)
                    vkDestroyDescriptorPool(device, shape_overlay_descriptor_pool, nullptr);
                if (shape_overlay_descriptor_layout != VK_NULL_HANDLE)
                    vkDestroyDescriptorSetLayout(device, shape_overlay_descriptor_layout, nullptr);
                if (shape_overlay_dummy_depth_view != VK_NULL_HANDLE)
                    vkDestroyImageView(device, shape_overlay_dummy_depth_view, nullptr);
                if (shape_overlay_dummy_depth_image != VK_NULL_HANDLE) {
                    if (!shape_overlay_dummy_depth_vram_label.empty()) {
                        lfs::diagnostics::VramProfiler::instance().recordCurrentBytes(
                            "vulkan.viewport.shape_overlay_dummy_depth",
                            shape_overlay_dummy_depth_vram_label,
                            0);
                    }
                    vmaDestroyImage(allocator, shape_overlay_dummy_depth_image, shape_overlay_dummy_depth_alloc);
                }
                if (shape_overlay_depth_sampler != VK_NULL_HANDLE)
                    vkDestroySampler(device, shape_overlay_depth_sampler, nullptr);
            }
            *this = {};
        }
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
