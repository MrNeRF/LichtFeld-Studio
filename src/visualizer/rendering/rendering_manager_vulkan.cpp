/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/logger.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "model_renderability.hpp"
#include "rendering/image_layout.hpp"
#include "rendering/rasterizer/rasterization/include/rasterization_config.h"
#include "rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "training/trainer.hpp"
#include "training/training_manager.hpp"
#include "viewport_appearance_correction.hpp"
#include "viewport_region_utils.hpp"
#include "viewport_request_builder.hpp"
#include "vksplat_viewport_renderer.hpp"
#include <algorithm>
#include <cmath>
#include <expected>
#include <format>
#include <shared_mutex>
#include <string>
#include <utility>
#include <vector>

namespace lfs::vis {

    namespace {
        [[nodiscard]] std::optional<std::shared_lock<std::shared_mutex>> acquireLiveModelRenderLock(
            const SceneManager* const scene_manager) {
            std::optional<std::shared_lock<std::shared_mutex>> lock;
            if (const auto* tm = scene_manager ? scene_manager->getTrainerManager() : nullptr) {
                if (const auto* trainer = tm->getTrainer()) {
                    lock.emplace(trainer->getRenderMutex());
                }
            }
            return lock;
        }

        [[nodiscard]] std::vector<ViewportInteractionPanel> buildVulkanInteractionPanels(
            const Viewport& primary_viewport,
            const SplitViewService& split_view_service,
            const RenderSettings& settings,
            const glm::vec2& screen_viewport_pos,
            const glm::vec2& screen_viewport_size) {
            std::vector<ViewportInteractionPanel> panels;
            if (screen_viewport_size.x <= 0.0f || screen_viewport_size.y <= 0.0f) {
                return panels;
            }

            const int full_screen_width = std::max(static_cast<int>(std::lround(screen_viewport_size.x)), 1);
            const int full_screen_height = std::max(static_cast<int>(std::lround(screen_viewport_size.y)), 1);
            const auto make_panel = [&](const SplitViewPanelId panel_id,
                                        const Viewport* const viewport,
                                        const float offset_x,
                                        const float width) {
                panels.push_back({
                    .panel = panel_id,
                    .viewport_data =
                        {.rotation = viewport->getRotationMatrix(),
                         .translation = viewport->getTranslation(),
                         .size = {
                             std::max(static_cast<int>(std::lround(width)), 1),
                             full_screen_height,
                         },
                         .focal_length_mm = settings.focal_length_mm,
                         .orthographic = settings.orthographic,
                         .ortho_scale = settings.ortho_scale},
                    .viewport_pos = {screen_viewport_pos.x + offset_x, screen_viewport_pos.y},
                    .viewport_size = {width, screen_viewport_size.y},
                });
            };

            const auto layouts = split_view_service.panelLayouts(settings, full_screen_width);
            if (!layouts || full_screen_width <= 1) {
                make_panel(SplitViewPanelId::Left, &primary_viewport, 0.0f, screen_viewport_size.x);
                return panels;
            }

            panels.reserve(layouts->size());
            make_panel(SplitViewPanelId::Left,
                       &primary_viewport,
                       static_cast<float>((*layouts)[0].x),
                       static_cast<float>((*layouts)[0].width));
            make_panel(SplitViewPanelId::Right,
                       &split_view_service.secondaryViewport(),
                       static_cast<float>((*layouts)[1].x),
                       static_cast<float>((*layouts)[1].width));
            return panels;
        }

        // CPU split-view composite kept ONLY for capture/screenshot — runs lazily on
        // demand when captureViewportImage() is called, never per-frame. Mirrors the
        // pixel-for-pixel result of the Vulkan split_view.frag shader so screenshots
        // match what the user sees on screen.
        [[nodiscard]] std::shared_ptr<lfs::core::Tensor> composeSplitViewCpu(
            const VulkanSplitViewParams& params,
            const glm::ivec2& output_size) {
            const auto load_panel = [](const std::shared_ptr<const lfs::core::Tensor>& image)
                -> std::optional<std::tuple<lfs::core::Tensor, int, int, int>> {
                if (!image || !image->is_valid() || image->ndim() != 3) {
                    return std::nullopt;
                }
                const auto layout = lfs::rendering::detectImageLayout(*image);
                if (layout == lfs::rendering::ImageLayout::Unknown) {
                    return std::nullopt;
                }
                lfs::core::Tensor t = *image;
                if (t.dtype() == lfs::core::DataType::UInt8) {
                    t = t.to(lfs::core::DataType::Float32) / 255.0f;
                } else if (t.dtype() != lfs::core::DataType::Float32) {
                    t = t.to(lfs::core::DataType::Float32);
                }
                if (layout == lfs::rendering::ImageLayout::HWC) {
                    t = t.permute({2, 0, 1}).contiguous();
                }
                t = t.cpu().contiguous();
                const int w = static_cast<int>(layout == lfs::rendering::ImageLayout::HWC
                                                   ? image->size(1)
                                                   : image->size(2));
                const int h = static_cast<int>(layout == lfs::rendering::ImageLayout::HWC
                                                   ? image->size(0)
                                                   : image->size(1));
                const int c = static_cast<int>(layout == lfs::rendering::ImageLayout::HWC
                                                   ? image->size(2)
                                                   : image->size(0));
                return std::make_tuple(std::move(t), w, h, c);
            };
            auto left_data = load_panel(params.left.image);
            auto right_data = load_panel(params.right.image);
            if (!left_data || !right_data) {
                return {};
            }

            const int width = output_size.x;
            const int height = output_size.y;
            if (width <= 0 || height <= 0) {
                return {};
            }
            const std::size_t pixel_count = static_cast<std::size_t>(width) * height;
            std::vector<float> output(3 * pixel_count, 0.0f);
            for (std::size_t i = 0; i < pixel_count; ++i) {
                output[i] = params.background.r;
                output[pixel_count + i] = params.background.g;
                output[2 * pixel_count + i] = params.background.b;
            }

            const auto sample = [](const lfs::core::Tensor& tensor, int w, int h,
                                   float u, float v, int channel) {
                const int x = std::clamp(static_cast<int>(std::lround(std::clamp(u, 0.0f, 1.0f) *
                                                                      static_cast<float>(w - 1))),
                                         0, w - 1);
                const int y = std::clamp(static_cast<int>(std::lround(std::clamp(v, 0.0f, 1.0f) *
                                                                      static_cast<float>(h - 1))),
                                         0, h - 1);
                const float* data = tensor.ptr<float>();
                return data[(static_cast<std::size_t>(channel) * h + y) * w + x];
            };

            const int rect_x = params.content_rect.x;
            const int rect_y = params.content_rect.y;
            const int rect_w = std::max(params.content_rect.z, 1);
            const int rect_h = std::max(params.content_rect.w, 1);
            const int divider = rect_x + splitViewDividerPixel(rect_w, params.split_position);
            const float split_x = static_cast<float>(rect_x) +
                                  std::clamp(params.split_position, 0.0f, 1.0f) *
                                      static_cast<float>(rect_w);
            const float center_y = static_cast<float>(rect_y) + static_cast<float>(rect_h) * 0.5f;

            constexpr glm::vec3 kDividerColor(0.29f, 0.33f, 0.42f);
            constexpr float kMinBarWidthPx = 4.0f;
            constexpr float kHandleHeightPx = 80.0f;
            constexpr float kHandleWidthPx = 24.0f;
            constexpr float kCornerRadiusPx = 6.0f;
            constexpr float kGripSpacingPx = 10.0f;
            constexpr float kGripWidthPx = 2.0f;
            constexpr float kGripLengthPx = 12.0f;
            constexpr int kGripLineCount = 2;

            const auto write = [&](std::size_t idx, const glm::vec3& c) {
                output[idx] = c.r;
                output[pixel_count + idx] = c.g;
                output[2 * pixel_count + idx] = c.b;
            };

            for (int y = rect_y; y < rect_y + rect_h; ++y) {
                const float v = rect_h > 1
                                    ? static_cast<float>(y - rect_y) / static_cast<float>(rect_h - 1)
                                    : 0.0f;
                for (int x = rect_x; x < rect_x + rect_w; ++x) {
                    const float u = rect_w > 1
                                        ? static_cast<float>(x - rect_x) / static_cast<float>(rect_w - 1)
                                        : 0.0f;
                    const bool use_left = x < divider;
                    const auto& panel = use_left ? params.left : params.right;
                    const auto& data = use_left ? *left_data : *right_data;
                    float panel_u = u;
                    if (panel.normalize_x_to_panel) {
                        const float span = std::max(panel.end_position - panel.start_position, 1e-6f);
                        panel_u = (u - panel.start_position) / span;
                    }
                    const float panel_v = panel.flip_y ? 1.0f - v : v;
                    const std::size_t idx = static_cast<std::size_t>(y) * width + x;
                    write(idx,
                          {sample(std::get<0>(data), std::get<1>(data), std::get<2>(data),
                                  panel_u, panel_v, 0),
                           sample(std::get<0>(data), std::get<1>(data), std::get<2>(data),
                                  panel_u, panel_v, 1),
                           sample(std::get<0>(data), std::get<1>(data), std::get<2>(data),
                                  panel_u, panel_v, 2)});

                    const float dist_from_split = std::abs(static_cast<float>(x) + 0.5f - split_x);
                    if (dist_from_split < kMinBarWidthPx * 0.5f) {
                        glm::vec3 color = kDividerColor;
                        const float dist_from_center =
                            std::abs(static_cast<float>(y) + 0.5f - center_y);
                        const float handle_h = std::min(kHandleHeightPx, static_cast<float>(rect_h));
                        const float handle_w = std::min(kHandleWidthPx, static_cast<float>(rect_w));
                        if (dist_from_center < handle_h * 0.5f &&
                            dist_from_split < handle_w * 0.5f) {
                            const float corner_radius =
                                std::min(kCornerRadiusPx, std::min(handle_w, handle_h) * 0.5f);
                            const glm::vec2 corner_dist =
                                glm::vec2(dist_from_split, dist_from_center) -
                                (glm::vec2(handle_w, handle_h) * 0.5f - glm::vec2(corner_radius));
                            if (corner_dist.x <= 0.0f || corner_dist.y <= 0.0f ||
                                glm::length(corner_dist) <= corner_radius) {
                                color = kDividerColor * 0.8f;
                                const float local_y = static_cast<float>(y) + 0.5f - center_y;
                                for (int i = -kGripLineCount; i <= kGripLineCount; ++i) {
                                    const float line_y = static_cast<float>(i) * kGripSpacingPx;
                                    if (std::abs(local_y - line_y) < kGripWidthPx &&
                                        dist_from_split < kGripLengthPx * 0.5f) {
                                        color = glm::vec3(0.9f);
                                        break;
                                    }
                                }
                            }
                        }
                        write(idx, color);
                    }
                }
            }

            auto tensor = lfs::core::Tensor::from_vector(
                              output,
                              {static_cast<std::size_t>(3),
                               static_cast<std::size_t>(height),
                               static_cast<std::size_t>(width)},
                              lfs::core::Device::CPU)
                              .cuda();
            return std::make_shared<lfs::core::Tensor>(std::move(tensor));
        }

        struct SplitCompositeContentRect {
            int x = 0;
            int y = 0;
            int width = 0;
            int height = 0;
        };

        [[nodiscard]] SplitCompositeContentRect resolveSplitCompositeContentRect(
            const glm::ivec2 output_size,
            const bool letterbox,
            const glm::ivec2 content_size) {
            SplitCompositeContentRect rect{
                .x = 0,
                .y = 0,
                .width = output_size.x,
                .height = output_size.y};
            if (!letterbox || content_size.x <= 0 || content_size.y <= 0 ||
                output_size.x <= 0 || output_size.y <= 0) {
                return rect;
            }

            const float content_aspect =
                static_cast<float>(content_size.x) / static_cast<float>(content_size.y);
            const float output_aspect =
                static_cast<float>(output_size.x) / static_cast<float>(output_size.y);
            if (content_aspect > output_aspect) {
                rect.width = output_size.x;
                rect.height = std::max(
                    static_cast<int>(std::lround(static_cast<float>(output_size.x) / content_aspect)),
                    1);
                rect.x = 0;
                rect.y = std::max((output_size.y - rect.height) / 2, 0);
            } else {
                rect.height = output_size.y;
                rect.width = std::max(
                    static_cast<int>(std::lround(static_cast<float>(output_size.y) * content_aspect)),
                    1);
                rect.x = std::max((output_size.x - rect.width) / 2, 0);
                rect.y = 0;
            }
            rect.width = std::clamp(rect.width, 1, output_size.x);
            rect.height = std::clamp(rect.height, 1, output_size.y);
            return rect;
        }

        [[nodiscard]] lfs::rendering::FrameMetadata makeSplitMetadata(
            const lfs::rendering::FrameMetadata& left,
            const lfs::rendering::FrameMetadata& right,
            const float split_position) {
            lfs::rendering::FrameMetadata metadata{
                .depth_panels =
                    {lfs::rendering::FramePanelMetadata{
                         .depth = left.depth_panel_count > 0 ? left.depth_panels[0].depth : nullptr,
                         .start_position = 0.0f,
                         .end_position = split_position,
                     },
                     lfs::rendering::FramePanelMetadata{
                         .depth = right.depth_panel_count > 0 ? right.depth_panels[0].depth : nullptr,
                         .start_position = split_position,
                         .end_position = 1.0f,
                     }},
                .depth_panel_count = 2,
                .valid = true,
                .far_plane = left.valid ? left.far_plane : right.far_plane,
                .orthographic = left.valid ? left.orthographic : right.orthographic};
            return metadata;
        }

        // Per-frame mesh-pass payload (without depth blit). Both the FastGs/Gut path
        // and the VkSplat path must publish this every frame so the viewport pass
        // sees the current camera; otherwise the mesh stays anchored to whichever
        // frame last set it. Depth blit (splat→mesh z-test) is path-specific and
        // attached by the caller.
        [[nodiscard]] RenderingManager::VulkanMeshFrame populateMeshFrame(
            const FrameContext& frame_ctx,
            const RenderSettings& settings,
            const VulkanSplitViewParams& split_view_params) {
            RenderingManager::VulkanMeshFrame frame;
            const auto vp_data = frame_ctx.makeViewportData();
            frame.view_projection = vp_data.getProjectionMatrix() * vp_data.getViewMatrix();
            frame.camera_position = vp_data.translation;
            frame.items.reserve(frame_ctx.scene_state.meshes.size());

            const bool any_selected_mesh = std::any_of(
                frame_ctx.scene_state.meshes.begin(),
                frame_ctx.scene_state.meshes.end(),
                [](const auto& mesh) { return mesh.is_selected; });
            const bool any_selected_node = std::any_of(
                frame_ctx.scene_state.selected_node_mask.begin(),
                frame_ctx.scene_state.selected_node_mask.end(),
                [](const bool selected) { return selected; });
            const bool dim_non_emphasized =
                settings.desaturate_unselected && (any_selected_mesh || any_selected_node);

            const glm::vec3 headlight_dir = glm::length(vp_data.translation) > 1e-6f
                                                ? glm::normalize(vp_data.translation)
                                                : settings.mesh_light_dir;

            for (const auto& mesh : frame_ctx.scene_state.meshes) {
                if (!mesh.mesh) {
                    continue;
                }
                lfs::vis::VulkanMeshDrawItem item{};
                item.mesh = mesh.mesh;
                item.model = mesh.transform;
                item.light_dir = headlight_dir;
                item.light_intensity = settings.mesh_light_intensity;
                item.ambient = settings.mesh_ambient;
                item.backface_culling = settings.mesh_backface_culling;
                item.is_emphasized = mesh.is_selected;
                item.dim_non_emphasized = dim_non_emphasized;
                item.flash_intensity = frame_ctx.selection_flash_intensity;
                item.wireframe_overlay = settings.mesh_wireframe;
                item.wireframe_color = settings.mesh_wireframe_color;
                item.wireframe_width = settings.mesh_wireframe_width;
                item.shadow_enabled = settings.mesh_shadow_enabled;
                item.shadow_map_resolution = settings.mesh_shadow_resolution;
                frame.items.push_back(item);
            }

            const auto frame_view = frame_ctx.makeFrameView();
            frame.environment.enabled = environmentBackgroundEnabled(settings);
            frame.environment.map_path = settings.environment_map_path;
            frame.environment.camera_to_world = vp_data.rotation;
            frame.environment.viewport_size =
                glm::vec2(static_cast<float>(frame_view.size.x), static_cast<float>(frame_view.size.y));
            if (frame_view.intrinsics_override.has_value() && !frame_view.orthographic) {
                const auto& intr = *frame_view.intrinsics_override;
                frame.environment.intrinsics =
                    glm::vec4(intr.focal_x, intr.focal_y, intr.center_x, intr.center_y);
            } else {
                const auto [fx, fy] =
                    lfs::rendering::computePixelFocalLengths(frame_view.size, frame_view.focal_length_mm);
                frame.environment.intrinsics =
                    glm::vec4(fx, fy, frame_view.size.x * 0.5f, frame_view.size.y * 0.5f);
            }
            frame.environment.exposure = settings.environment_exposure;
            frame.environment.rotation_radians = glm::radians(settings.environment_rotation_degrees);
            frame.environment.equirectangular_view = settings.equirectangular;

            frame.split_view = split_view_params;

            return frame;
        }
    } // namespace

    RenderingManager::VulkanFrameResult RenderingManager::renderVulkanFrame(const RenderContext& context) {
        LOG_TIMER("renderVulkanFrame");
        SceneManager* const scene_manager = context.scene_manager;

        if (!engine_) {
            engine_ = lfs::rendering::RenderingEngine::createRasterOnly();
        }
        if (!raster_initialized_) {
            if (auto init_result = engine_->initializeRasterOnly(); !init_result) {
                LOG_ERROR("Failed to initialize Vulkan raster path: {}", init_result.error());
                return {.image = vulkan_viewport_image_,
                        .size = vulkan_viewport_image_size_,
                        .flip_y = vulkan_viewport_image_flip_y_};
            }
            raster_initialized_ = true;
            initialized_ = true;
        }

        if (scene_manager && (dirty_mask_.load(std::memory_order_relaxed) & DirtyFlag::SELECTION)) {
            for (const auto& group : scene_manager->getScene().getSelectionGroups()) {
                lfs::rendering::config::setSelectionGroupColor(
                    group.id, make_float3(group.color.x, group.color.y, group.color.z));
            }
        }

        const auto framebuffer_region =
            resolveFramebufferViewportRegion(context.viewport, context.logical_screen_size, context.viewport_region);
        glm::ivec2 current_size = context.viewport.frameBufferSize;
        if (context.viewport_region) {
            current_size = framebuffer_region.size;
        }
        if (current_size.x <= 0 || current_size.y <= 0) {
            return {.image = vulkan_viewport_image_,
                    .size = vulkan_viewport_image_size_,
                    .flip_y = vulkan_viewport_image_flip_y_};
        }

        glm::vec2 screen_viewport_pos(0.0f, 0.0f);
        glm::vec2 screen_viewport_size(
            static_cast<float>(context.viewport.windowSize.x),
            static_cast<float>(context.viewport.windowSize.y));
        if (context.viewport_region) {
            screen_viewport_pos = {context.viewport_region->x, context.viewport_region->y};
            screen_viewport_size = {context.viewport_region->width, context.viewport_region->height};
        }
        const auto interaction_panels = buildVulkanInteractionPanels(
            context.viewport,
            split_view_service_,
            settings_,
            screen_viewport_pos,
            screen_viewport_size);
        viewport_interaction_context_.updatePickContext(interaction_panels);

        const auto resize_result = frame_lifecycle_service_.handleViewportResize(current_size);
        if (resize_result.dirty) {
            markDirty(resize_result.dirty);
        }

        auto render_lock = acquireLiveModelRenderLock(scene_manager);

        const lfs::core::SplatData* const model = scene_manager ? scene_manager->getModelForRendering() : nullptr;
        SceneRenderState scene_state;
        if (scene_manager) {
            LOG_TIMER("renderVulkanFrame.buildRenderState");
            scene_state = scene_manager->buildRenderState();
        }
        const bool has_renderable_model = hasRenderableGaussians(model);
        const bool has_point_cloud =
            scene_state.point_cloud != nullptr && scene_state.point_cloud->size() > 0;
        const bool has_meshes = std::any_of(scene_state.meshes.begin(),
                                            scene_state.meshes.end(),
                                            [](const auto& mesh) { return mesh.mesh != nullptr; });
        const bool has_environment = environmentBackgroundEnabled(settings_);
        const bool has_render_content = has_renderable_model || has_point_cloud || has_meshes || has_environment;
        const size_t model_ptr = reinterpret_cast<size_t>(model);

        if (const auto model_change = frame_lifecycle_service_.handleModelChange(model_ptr, viewport_artifact_service_);
            model_change.changed) {
            vulkan_viewport_image_.reset();
            vulkan_external_viewport_image_ = VK_NULL_HANDLE;
            vulkan_external_viewport_image_view_ = VK_NULL_HANDLE;
            vulkan_external_viewport_image_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
            vulkan_external_viewport_image_generation_ = 0;
            vulkan_viewport_image_size_ = {0, 0};
            vulkan_viewport_image_flip_y_ = false;
            if (vksplat_viewport_renderer_) {
                vksplat_viewport_renderer_->reset();
            }
            viewport_artifact_service_.clearViewportOutput();
            markDirty(DirtyFlag::ALL);
        }

        const bool is_training = scene_manager && scene_manager->hasDataset() &&
                                 scene_manager->getTrainerManager() &&
                                 scene_manager->getTrainerManager()->isRunning();
        if (const DirtyMask training_dirty = frame_lifecycle_service_.handleTrainingRefresh(
                is_training,
                framerate_controller_.getSettings().training_frame_refresh_time_sec);
            training_dirty) {
            markDirty(training_dirty);
        }

        if (const DirtyMask required_dirty = frame_lifecycle_service_.requiredDirtyMask(
                vulkan_viewport_image_ != nullptr || vulkan_external_viewport_image_ != VK_NULL_HANDLE,
                has_render_content,
                settings_.split_view_mode);
            required_dirty) {
            dirty_mask_.fetch_or(required_dirty, std::memory_order_relaxed);
        }

        DirtyMask frame_dirty = dirty_mask_.exchange(0);
        LOG_PERF("renderVulkanFrame.frame_dirty=0x{:x} model={} pc={} mesh={} env={}",
                 frame_dirty, has_renderable_model, has_point_cloud, has_meshes, has_environment);
        if (!has_render_content) {
            vulkan_viewport_image_.reset();
            vulkan_external_viewport_image_ = VK_NULL_HANDLE;
            vulkan_external_viewport_image_view_ = VK_NULL_HANDLE;
            vulkan_external_viewport_image_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
            vulkan_external_viewport_image_generation_ = 0;
            vulkan_viewport_image_size_ = {0, 0};
            vulkan_viewport_image_flip_y_ = false;
            viewport_artifact_service_.clearViewportOutput();
            render_lock.reset();
            return {};
        }

        if (frame_dirty == 0 &&
            (vulkan_viewport_image_ || vulkan_external_viewport_image_ != VK_NULL_HANDLE)) {
            LOG_PERF("renderVulkanFrame: cache HIT (returning cached image)");
            render_lock.reset();
            if (vulkan_external_viewport_image_ != VK_NULL_HANDLE) {
                return {.image = {},
                        .external_image = vulkan_external_viewport_image_,
                        .external_image_view = vulkan_external_viewport_image_view_,
                        .external_image_layout = vulkan_external_viewport_image_layout_,
                        .external_image_generation = vulkan_external_viewport_image_generation_,
                        .image_generation = vulkan_viewport_image_generation_,
                        .size = vulkan_viewport_image_size_,
                        .flip_y = vulkan_viewport_image_flip_y_};
            }
            return {.image = vulkan_viewport_image_,
                    .image_generation = vulkan_viewport_image_generation_,
                    .size = vulkan_viewport_image_size_,
                    .flip_y = vulkan_viewport_image_flip_y_};
        }

        framerate_controller_.beginFrame();

        const float scale = std::clamp(settings_.render_scale, 0.25f, 1.0f);
        glm::ivec2 render_size(
            std::max(static_cast<int>(std::lround(static_cast<float>(current_size.x) * scale)), 1),
            std::max(static_cast<int>(std::lround(static_cast<float>(current_size.y) * scale)), 1));

        const FrameContext frame_ctx{
            .viewport = context.viewport,
            .viewport_region = context.viewport_region,
            .render_lock_held = render_lock.has_value(),
            .scene_manager = scene_manager,
            .model = model,
            .scene_state = std::move(scene_state),
            .settings = settings_,
            .render_size = render_size,
            .viewport_pos = {0, 0},
            .frame_dirty = frame_dirty,
            .cursor_preview = viewport_overlay_service_.cursorPreview(),
            .gizmo = viewport_overlay_service_.makeFrameGizmoState(),
            .hovered_camera_id = camera_interaction_service_.hoveredCameraId(),
            .current_camera_id = camera_interaction_service_.currentCameraId(),
            .hovered_gaussian_id = viewport_overlay_service_.hoveredGaussianId(),
            .selection_flash_intensity = getSelectionFlashIntensity(),
            .view_panels = {}};

        std::shared_ptr<lfs::core::Tensor> rendered_image;
        lfs::rendering::FrameMetadata rendered_metadata{};
        std::string render_error;
        bool rendered_image_contains_ground_truth = false;
        glm::ivec2 rendered_gt_content_size{0, 0};
        std::optional<SplitViewInfo> rendered_split_info;
        VulkanSplitViewParams pending_split_view{};

        const auto populate_independent_split_mesh_panels =
            [&](VulkanMeshFrame& frame) {
                if (!pending_split_view.enabled ||
                    !splitViewUsesIndependentPanels(settings_.split_view_mode)) {
                    return;
                }
                const auto layouts = split_view_service_.panelLayouts(settings_, render_size.x);
                if (!layouts || render_size.y <= 0) {
                    return;
                }
                frame.panels.clear();
                const auto append_panel =
                    [&](const Viewport& viewport, const std::size_t index) {
                        const auto& layout = (*layouts)[index];
                        const glm::ivec2 panel_size{std::max(layout.width, 1), render_size.y};
                        const auto panel_view = frame_ctx.makeViewportData(viewport, panel_size);
                        frame.panels.push_back(lfs::vis::VulkanMeshViewportPanel{
                            .start_position = layout.start_position,
                            .end_position = layout.end_position,
                            .view_projection = panel_view.getProjectionMatrix() * panel_view.getViewMatrix(),
                            .camera_position = panel_view.translation});
                    };
                append_panel(context.viewport, 0);
                append_panel(split_view_service_.secondaryViewport(), 1);
            };

        struct RenderedPanel {
            std::shared_ptr<lfs::core::Tensor> image;
            lfs::rendering::FrameMetadata metadata;
        };

        const auto render_panel_image =
            [&](const Viewport& source_viewport,
                const glm::ivec2 panel_size,
                const std::optional<SplitViewPanelId> panel_id,
                const std::optional<std::vector<bool>>& node_visibility_override,
                const lfs::core::SplatData* model_override = nullptr,
                const std::vector<glm::mat4>* model_transforms_override = nullptr)
            -> std::expected<RenderedPanel, std::string> {
            const lfs::core::SplatData* const panel_model = model_override ? model_override : model;
            if (panel_size.x <= 0 || panel_size.y <= 0) {
                return std::unexpected("Invalid split-view panel size");
            }

            if ((settings_.point_cloud_mode || !hasRenderableGaussians(panel_model)) && has_point_cloud && !panel_model) {
                const std::vector<glm::mat4> point_cloud_transforms = {frame_ctx.scene_state.point_cloud_transform};
                const auto state = buildSplitViewPointCloudPanelRenderState(frame_ctx, panel_size, &source_viewport);
                lfs::rendering::PointCloudRenderRequest request{
                    .frame_view = state.frame_view,
                    .render = state.render,
                    .scene =
                        {.model_transforms = &point_cloud_transforms,
                         .transform_indices = nullptr,
                         .node_visibility_mask = {}},
                    .filters = state.filters,
                    .transparent_background = environmentBackgroundUsesTransparentViewerCompositing(settings_)};
                auto result = engine_->renderPointCloudImage(*frame_ctx.scene_state.point_cloud, request);
                if (!result || !result->image) {
                    return std::unexpected(result ? "Raw point-cloud panel render returned no image"
                                                  : result.error());
                }
                return RenderedPanel{.image = std::move(result->image), .metadata = std::move(result->metadata)};
            }

            if (!hasRenderableGaussians(panel_model)) {
                return std::unexpected("No renderable model for split-view panel");
            }

            if (settings_.point_cloud_mode) {
                const auto state = buildSplitViewPointCloudPanelRenderState(frame_ctx, panel_size, &source_viewport);
                std::vector<glm::mat4> transforms_storage;
                auto scene = state.scene;
                if (model_transforms_override) {
                    scene.model_transforms = model_transforms_override;
                } else if (!scene.model_transforms) {
                    transforms_storage = {glm::mat4(1.0f)};
                    scene.model_transforms = &transforms_storage;
                }
                if (node_visibility_override) {
                    scene.node_visibility_mask = *node_visibility_override;
                }
                lfs::rendering::PointCloudRenderRequest request{
                    .frame_view = state.frame_view,
                    .render = state.render,
                    .scene = scene,
                    .filters = state.filters,
                    .transparent_background = environmentBackgroundUsesTransparentViewerCompositing(settings_)};
                auto result = engine_->renderPointCloudImage(*panel_model, request);
                if (!result || !result->image) {
                    return std::unexpected(result ? "Point-cloud panel render returned no image"
                                                  : result.error());
                }
                return RenderedPanel{.image = std::move(result->image), .metadata = std::move(result->metadata)};
            }

            auto request = buildViewportRenderRequest(frame_ctx, panel_size, &source_viewport, panel_id);
            std::vector<glm::mat4> transforms_storage;
            if (model_transforms_override) {
                request.scene.model_transforms = model_transforms_override;
            } else if (!request.scene.model_transforms) {
                transforms_storage = {glm::mat4(1.0f)};
                request.scene.model_transforms = &transforms_storage;
            }
            if (node_visibility_override) {
                request.scene.node_visibility_mask = *node_visibility_override;
            }
            auto result = engine_->renderGaussiansImage(*panel_model, request);
            if (!result || !result->image) {
                return std::unexpected(result ? "Gaussian panel render returned no image"
                                              : result.error());
            }
            return RenderedPanel{.image = std::move(result->image), .metadata = std::move(result->metadata)};
        };

        if (splitViewUsesGTComparison(settings_.split_view_mode) && scene_manager &&
            (has_renderable_model || has_point_cloud)) {
            std::shared_ptr<lfs::core::Camera> camera;
            const auto cameras = scene_manager->getScene().getAllCameras();
            if (frame_ctx.current_camera_id >= 0) {
                for (const auto& candidate : cameras) {
                    if (candidate && candidate->uid() == frame_ctx.current_camera_id) {
                        camera = candidate;
                        break;
                    }
                }
            }
            if (!camera && !cameras.empty()) {
                for (const auto& candidate : cameras) {
                    if (candidate) {
                        camera = candidate;
                        break;
                    }
                }
            }

            if (camera && !camera->image_path().empty()) {
                try {
                    auto gt_tensor = camera->load_and_get_image(-1, render_size.x, false);
                    if (gt_tensor.is_valid() && gt_tensor.ndim() == 3) {
                        const auto gt_layout = lfs::rendering::detectImageLayout(gt_tensor);
                        if (gt_layout != lfs::rendering::ImageLayout::Unknown) {
                            gt_tensor = lfs::rendering::flipImageVertical(gt_tensor, gt_layout);
                            const glm::ivec2 gt_size{
                                lfs::rendering::imageWidth(gt_tensor, gt_layout),
                                lfs::rendering::imageHeight(gt_tensor, gt_layout)};

                            auto request = buildViewportRenderRequest(frame_ctx, gt_size);
                            const glm::mat4 scene_transform =
                                detail::currentSceneTransform(scene_manager, camera->uid());
                            const auto render_camera =
                                detail::buildGTRenderCamera(*camera, gt_size, scene_transform);
                            if (render_camera) {
                                request.frame_view.rotation = render_camera->rotation;
                                request.frame_view.translation = render_camera->translation;
                                request.frame_view.intrinsics_override = render_camera->intrinsics;
                                request.frame_view.orthographic = false;
                                request.frame_view.ortho_scale = lfs::rendering::DEFAULT_ORTHO_SCALE;
                                request.equirectangular = render_camera->equirectangular;
                            }

                            std::shared_ptr<lfs::core::Tensor> compare_image;
                            lfs::rendering::FrameMetadata compare_metadata{};
                            std::string compare_error;

                            const bool use_point_cloud_compare =
                                settings_.point_cloud_mode || !has_renderable_model;
                            if (use_point_cloud_compare && has_renderable_model) {
                                auto point_request = buildPointCloudRenderRequest(
                                    frame_ctx, gt_size, frame_ctx.scene_state.model_transforms);
                                point_request.frame_view = request.frame_view;
                                auto rendered = engine_->renderPointCloudImage(*model, point_request);
                                if (rendered && rendered->image) {
                                    compare_image = std::move(rendered->image);
                                    compare_metadata = std::move(rendered->metadata);
                                } else {
                                    compare_error = rendered ? "Point-cloud GT comparison render returned no image"
                                                             : rendered.error();
                                }
                            } else if (use_point_cloud_compare && has_point_cloud) {
                                const std::vector<glm::mat4> point_cloud_transforms = {
                                    frame_ctx.scene_state.point_cloud_transform};
                                auto point_request = buildPointCloudRenderRequest(
                                    frame_ctx, gt_size, point_cloud_transforms);
                                point_request.frame_view = request.frame_view;
                                auto rendered = engine_->renderPointCloudImage(
                                    *frame_ctx.scene_state.point_cloud, point_request);
                                if (rendered && rendered->image) {
                                    compare_image = std::move(rendered->image);
                                    compare_metadata = std::move(rendered->metadata);
                                } else {
                                    compare_error = rendered ? "Raw point-cloud GT comparison render returned no image"
                                                             : rendered.error();
                                }
                            } else if (has_renderable_model) {
                                auto rendered = engine_->renderGaussiansImage(*model, request);
                                if (rendered && rendered->image) {
                                    compare_image = std::move(rendered->image);
                                    compare_metadata = std::move(rendered->metadata);
                                } else {
                                    compare_error = rendered ? "GT comparison render returned no image" : rendered.error();
                                }
                            } else {
                                compare_error = "GT comparison requires a renderable Gaussian model or point cloud";
                            }

                            if (compare_image) {
                                compare_image = applyViewportAppearanceCorrection(
                                    std::move(compare_image),
                                    scene_manager,
                                    settings_,
                                    camera->uid());
                                auto gt_image = std::make_shared<lfs::core::Tensor>(std::move(gt_tensor));
                                const SplitCompositeContentRect rect =
                                    resolveSplitCompositeContentRect(render_size, true, gt_size);
                                pending_split_view.enabled = true;
                                pending_split_view.split_position = settings_.split_position;
                                pending_split_view.background = settings_.background_color;
                                pending_split_view.content_rect = {rect.x, rect.y, rect.width, rect.height};
                                pending_split_view.left = {std::move(gt_image), 0.0f, settings_.split_position, false, true};
                                pending_split_view.right = {compare_image, settings_.split_position, 1.0f, false, !compare_metadata.flip_y};
                                rendered_metadata = compare_metadata;
                                rendered_image_contains_ground_truth = true;
                                rendered_gt_content_size = gt_size;
                                rendered_split_info = SplitViewInfo{
                                    .enabled = true,
                                    .mode_label = "GT Compare",
                                    .detail_label = camera->image_name(),
                                    .left_name = "Ground Truth",
                                    .right_name = "Rendered"};
                            } else {
                                render_error = compare_error;
                            }
                        }
                    }
                } catch (const std::exception& e) {
                    render_error = std::format("GT comparison failed: {}", e.what());
                }
            }
        } else if (splitViewUsesIndependentPanels(settings_.split_view_mode)) {
            if (const auto layouts = split_view_service_.panelLayouts(settings_, render_size.x);
                layouts && render_size.x > 1) {
                auto left = render_panel_image(
                    context.viewport,
                    {std::max((*layouts)[0].width, 1), render_size.y},
                    SplitViewPanelId::Left,
                    std::nullopt);
                auto right = render_panel_image(
                    split_view_service_.secondaryViewport(),
                    {std::max((*layouts)[1].width, 1), render_size.y},
                    SplitViewPanelId::Right,
                    std::nullopt);
                if (left && right) {
                    pending_split_view.enabled = true;
                    pending_split_view.split_position = settings_.split_position;
                    pending_split_view.background = settings_.background_color;
                    pending_split_view.content_rect = {0, 0, render_size.x, render_size.y};
                    pending_split_view.left = {left->image, (*layouts)[0].start_position, (*layouts)[0].end_position, true, !left->metadata.flip_y};
                    pending_split_view.right = {right->image, (*layouts)[1].start_position, (*layouts)[1].end_position, true, !right->metadata.flip_y};
                    rendered_metadata = makeSplitMetadata(left->metadata, right->metadata, settings_.split_position);
                    rendered_split_info = SplitViewInfo{
                        .enabled = true,
                        .mode_label = "Split View",
                        .detail_label = "Primary | Secondary",
                        .left_name = "Primary View",
                        .right_name = "Secondary View"};
                } else {
                    render_error = left ? right.error() : left.error();
                }
            }
        } else if (splitViewUsesPLYComparison(settings_.split_view_mode) && scene_manager && has_renderable_model) {
            const auto visible_nodes = scene_manager->getScene().getVisibleNodes();
            if (visible_nodes.size() >= 2) {
                const size_t left_idx = settings_.split_view_offset % visible_nodes.size();
                const size_t right_idx = (settings_.split_view_offset + 1) % visible_nodes.size();
                std::vector<bool> left_mask(visible_nodes.size(), false);
                std::vector<bool> right_mask(visible_nodes.size(), false);
                left_mask[left_idx] = true;
                right_mask[right_idx] = true;

                auto left = render_panel_image(
                    context.viewport, render_size, std::nullopt, std::optional<std::vector<bool>>(left_mask));
                auto right = render_panel_image(
                    context.viewport, render_size, std::nullopt, std::optional<std::vector<bool>>(right_mask));
                if (left && right) {
                    pending_split_view.enabled = true;
                    pending_split_view.split_position = settings_.split_position;
                    pending_split_view.background = settings_.background_color;
                    pending_split_view.content_rect = {0, 0, render_size.x, render_size.y};
                    pending_split_view.left = {left->image, 0.0f, settings_.split_position, false, !left->metadata.flip_y};
                    pending_split_view.right = {right->image, settings_.split_position, 1.0f, false, !right->metadata.flip_y};
                    rendered_metadata = makeSplitMetadata(left->metadata, right->metadata, settings_.split_position);
                    rendered_split_info = SplitViewInfo{
                        .enabled = true,
                        .mode_label = "Split View",
                        .detail_label = std::format("{} | {}",
                                                    visible_nodes[left_idx]->name,
                                                    visible_nodes[right_idx]->name),
                        .left_name = visible_nodes[left_idx]->name,
                        .right_name = visible_nodes[right_idx]->name};
                } else {
                    render_error = left ? right.error() : left.error();
                }
            }
        }

        const bool render_point_cloud = settings_.point_cloud_mode || !has_renderable_model;

        if (rendered_image || pending_split_view.enabled) {
            // Split-view paths populate pending_split_view directly; skip the
            // full-viewport fallback that would set rendered_image to a wrong-
            // sized tensor and squash the left panel through the scene interop.
        } else if (render_point_cloud && has_renderable_model) {
            LOG_TIMER("renderVulkanFrame.renderPointCloudImage(splat)");
            auto request = buildPointCloudRenderRequest(frame_ctx, render_size, frame_ctx.scene_state.model_transforms);
            auto render_result = engine_->renderPointCloudImage(*model, request);
            if (render_result) {
                rendered_image = std::move(render_result->image);
                rendered_metadata = std::move(render_result->metadata);
            } else {
                render_error = render_result.error();
            }
        } else if (render_point_cloud && has_point_cloud) {
            LOG_TIMER("renderVulkanFrame.renderPointCloudImage(pc)");
            const std::vector<glm::mat4> point_cloud_transforms = {frame_ctx.scene_state.point_cloud_transform};
            auto request = buildPointCloudRenderRequest(frame_ctx, render_size, point_cloud_transforms);
            auto render_result = engine_->renderPointCloudImage(*frame_ctx.scene_state.point_cloud, request);
            if (render_result) {
                rendered_image = std::move(render_result->image);
                rendered_metadata = std::move(render_result->metadata);
            } else {
                render_error = render_result.error();
            }
        } else if (has_renderable_model) {
            auto request = buildViewportRenderRequest(frame_ctx, render_size);
            // VkSplat is editing/viewing only — splat files loaded with no
            // trainer attached. As soon as a dataset is present (ready, paused,
            // or running), its persistent Vulkan-side sort buffers can be
            // pinned to densification peaks, so we fall back to FastGs.
            const bool viewing_only =
                scene_manager && scene_manager->hasSplatFiles();
            if (request.raster_backend == lfs::rendering::GaussianRasterBackend::VkSplat &&
                !viewing_only) {
                request.raster_backend = lfs::rendering::GaussianRasterBackend::FastGs;
            }
            if (request.raster_backend == lfs::rendering::GaussianRasterBackend::VkSplat) {
                if (!context.vulkan_context) {
                    render_error = "VkSplat backend requires an active Vulkan context";
                } else {
                    if (!vksplat_viewport_renderer_) {
                        vksplat_viewport_renderer_ = std::make_unique<VksplatViewportRenderer>();
                    }
                    const bool force_input_upload = (frame_dirty & DirtyFlag::SPLATS) != 0;
                    LOG_TIMER("vksplat.render");
                    auto render_result = vksplat_viewport_renderer_->render(
                        *context.vulkan_context, *model, request, force_input_upload);
                    if (render_result) {
                        render_lock.reset();
                        vulkan_viewport_image_.reset();
                        vulkan_external_viewport_image_ = render_result->image;
                        vulkan_external_viewport_image_view_ = render_result->image_view;
                        vulkan_external_viewport_image_layout_ = render_result->image_layout;
                        vulkan_external_viewport_image_generation_ = render_result->generation;
                        vulkan_viewport_image_size_ = render_result->size;
                        vulkan_viewport_image_flip_y_ = render_result->flip_y;
                        vulkan_gt_comparison_content_size_ = {0, 0};
                        viewport_artifact_service_.clearViewportOutput();

                        if (resize_result.completed) {
                            frame_lifecycle_service_.noteResizeCompleted();
                            lfs::core::Tensor::trim_memory_pool();
                        }
                        queueCameraMetricsRefreshIfStale(scene_manager);
                        viewport_interaction_context_.scene_manager = scene_manager;
                        split_view_service_.updateInfo(FrameResources{});

                        // VkSplat returns early before the shared mesh-frame setup
                        // below. Republish it here so the viewport pass sees the
                        // current camera; vksplat doesn't expose splat depth, so
                        // depth_blit stays empty (mesh and splat composite without
                        // splat→mesh z-test, same as before this branch existed).
                        if (!frame_ctx.scene_state.meshes.empty() ||
                            environmentBackgroundEnabled(settings_)) {
                            auto mesh_frame = populateMeshFrame(frame_ctx, settings_, pending_split_view);
                            populate_independent_split_mesh_panels(mesh_frame);
                            setVulkanMeshFrame(std::move(mesh_frame));
                        } else {
                            clearVulkanMeshFrame();
                        }

                        return {.image = {},
                                .external_image = vulkan_external_viewport_image_,
                                .external_image_view = vulkan_external_viewport_image_view_,
                                .external_image_layout = vulkan_external_viewport_image_layout_,
                                .external_image_generation = vulkan_external_viewport_image_generation_,
                                .size = vulkan_viewport_image_size_,
                                .flip_y = vulkan_viewport_image_flip_y_};
                    }
                    render_error = render_result.error();
                }
            } else {
                auto render_result = engine_->renderGaussiansImage(*model, request);
                if (render_result) {
                    rendered_image = std::move(render_result->image);
                    rendered_metadata = std::move(render_result->metadata);
                } else {
                    render_error = render_result.error();
                }
            }
        }

        if (rendered_image && !rendered_image_contains_ground_truth) {
            rendered_image = applyViewportAppearanceCorrection(
                std::move(rendered_image),
                scene_manager,
                settings_,
                frame_ctx.current_camera_id);
        }

        if (frame_ctx.scene_state.meshes.empty()) {
            clearVulkanMeshFrame();
        }

        if ((rendered_image || render_error.empty() || pending_split_view.enabled) &&
            (environmentBackgroundEnabled(settings_) || !frame_ctx.scene_state.meshes.empty() ||
             pending_split_view.enabled)) {
            VulkanMeshFrame gpu_mesh_frame = populateMeshFrame(frame_ctx, settings_, pending_split_view);
            populate_independent_split_mesh_panels(gpu_mesh_frame);

            // Splat depth → mesh-pass z-test source. Only meaningful when the splat
            // produced a depth tensor (CUDA rasterizer path; vksplat doesn't yet).
            if (rendered_image && rendered_metadata.depth_panel_count > 0 &&
                rendered_metadata.depth_panels[0].depth &&
                rendered_metadata.depth_panels[0].depth->is_valid()) {
                gpu_mesh_frame.depth_blit.depth = rendered_metadata.depth_panels[0].depth;
                gpu_mesh_frame.depth_blit.depth_is_ndc = rendered_metadata.depth_is_ndc;
                gpu_mesh_frame.depth_blit.flip_y = !rendered_metadata.flip_y;
                gpu_mesh_frame.depth_blit.near_plane = rendered_metadata.near_plane > 0.0f
                                                           ? rendered_metadata.near_plane
                                                           : 0.1f;
                gpu_mesh_frame.depth_blit.far_plane = rendered_metadata.far_plane > 0.0f
                                                          ? rendered_metadata.far_plane
                                                          : 1000.0f;
            }

            setVulkanMeshFrame(std::move(gpu_mesh_frame));
        }
        render_lock.reset();

        const bool has_gpu_only_pass =
            !frame_ctx.scene_state.meshes.empty() ||
            environmentBackgroundEnabled(settings_) ||
            pending_split_view.enabled;

        if (!rendered_image && has_gpu_only_pass) {
            vulkan_viewport_image_.reset();
            vulkan_external_viewport_image_ = VK_NULL_HANDLE;
            vulkan_external_viewport_image_view_ = VK_NULL_HANDLE;
            vulkan_external_viewport_image_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
            vulkan_external_viewport_image_generation_ = 0;
            vulkan_viewport_image_size_ = render_size;
            vulkan_viewport_image_flip_y_ = false;
            vulkan_gt_comparison_content_size_ =
                rendered_image_contains_ground_truth ? rendered_gt_content_size : glm::ivec2{0, 0};
            FrameResources split_info_resources;
            if (rendered_split_info) {
                split_info_resources.split_view_executed = true;
                split_info_resources.split_info = *rendered_split_info;
            }
            split_view_service_.updateInfo(split_info_resources);

            if (pending_split_view.enabled) {
                viewport_artifact_service_.setLazyCapture(
                    [params = pending_split_view, render_size]() {
                        return composeSplitViewCpu(params, render_size);
                    },
                    rendered_metadata,
                    render_size);
            } else {
                viewport_artifact_service_.clearViewportOutput();
            }

            // Split-view: hand both panels to gui_manager so the existing scene
            // image interop covers the left panel and a parallel slot covers the right.
            // Both arrive in the viewport pass as external Vulkan image views; the
            // CPU staging path stays as a fallback for any frame where interop fails.
            VulkanFrameResult result{};
            result.image_generation = ++split_view_image_generation_;
            if (result.image_generation == 0) {
                result.image_generation = ++split_view_image_generation_;
            }
            result.size = vulkan_viewport_image_size_;
            result.flip_y = vulkan_viewport_image_flip_y_;

            const auto tensor_size = [](const lfs::core::Tensor& t) -> glm::ivec2 {
                const auto layout = lfs::rendering::detectImageLayout(t);
                if (layout == lfs::rendering::ImageLayout::Unknown) {
                    return {0, 0};
                }
                return {lfs::rendering::imageWidth(t, layout),
                        lfs::rendering::imageHeight(t, layout)};
            };

            if (pending_split_view.enabled) {
                if (auto left_tensor = pending_split_view.left.image) {
                    const auto sz = tensor_size(*left_tensor);
                    result.image = std::move(left_tensor);
                    result.size = sz;
                    result.flip_y = pending_split_view.left.flip_y;
                }
                if (auto right_tensor = pending_split_view.right.image) {
                    const auto sz = tensor_size(*right_tensor);
                    result.split_right_image = std::move(right_tensor);
                    result.split_right_size = sz;
                    result.split_right_flip_y = pending_split_view.right.flip_y;
                }
            }
            return result;
        }

        if (!rendered_image) {
            vulkan_gt_comparison_content_size_ = {0, 0};
            LOG_ERROR("Failed to render Vulkan viewport image: {}",
                      render_error.empty() ? "missing image payload" : render_error);
            vulkan_viewport_image_.reset();
            vulkan_external_viewport_image_ = VK_NULL_HANDLE;
            vulkan_external_viewport_image_view_ = VK_NULL_HANDLE;
            vulkan_external_viewport_image_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
            vulkan_external_viewport_image_generation_ = 0;
            vulkan_viewport_image_size_ = {0, 0};
            vulkan_viewport_image_flip_y_ = false;
            return {};
        }

        auto viewport_image = std::move(rendered_image);
        vulkan_viewport_image_ = viewport_image;
        ++vulkan_viewport_image_generation_;
        if (vulkan_viewport_image_generation_ == 0)
            ++vulkan_viewport_image_generation_;
        vulkan_external_viewport_image_ = VK_NULL_HANDLE;
        vulkan_external_viewport_image_view_ = VK_NULL_HANDLE;
        vulkan_external_viewport_image_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
        vulkan_external_viewport_image_generation_ = 0;
        vulkan_viewport_image_size_ = render_size;
        vulkan_viewport_image_flip_y_ = !rendered_metadata.flip_y;
        vulkan_gt_comparison_content_size_ =
            rendered_image_contains_ground_truth ? rendered_gt_content_size : glm::ivec2{0, 0};
        viewport_artifact_service_.updateFromImageOutput(
            std::move(viewport_image), rendered_metadata, render_size, true);

        if (resize_result.completed) {
            frame_lifecycle_service_.noteResizeCompleted();
            lfs::core::Tensor::trim_memory_pool();
        }

        queueCameraMetricsRefreshIfStale(scene_manager);
        viewport_interaction_context_.scene_manager = scene_manager;
        FrameResources split_info_resources;
        if (rendered_split_info) {
            split_info_resources.split_view_executed = true;
            split_info_resources.split_info = std::move(*rendered_split_info);
        }
        split_view_service_.updateInfo(split_info_resources);

        return {.image = vulkan_viewport_image_,
                .image_generation = vulkan_viewport_image_generation_,
                .size = vulkan_viewport_image_size_,
                .flip_y = vulkan_viewport_image_flip_y_};
    }

} // namespace lfs::vis
