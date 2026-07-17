/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/gui_manager.hpp"
#include "control/command_api.hpp"
#include "core/camera.hpp"
#include "core/cuda_error.hpp"
#include "core/cuda_version.hpp"
#include "core/environment.hpp"
#include "core/event_bridge/command_center_bridge.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "diagnostics/vram_profiler.hpp"
#include <ft2build.h>
#include FT_FREETYPE_H
#include "core/tensor.hpp"
#include "gui/bounds_gizmo.hpp"
#include "gui/editor/python_editor.hpp"
#include "gui/error_event_bridge.hpp"
#include "gui/layout_state.hpp"
#include "gui/line_renderer.hpp"
#include "gui/native_panels.hpp"
#include "gui/panel_input_utils.hpp"
#include "gui/panel_registry.hpp"
#include "gui/panels/python_console_panel.hpp"
#include "gui/rmlui/rml_panel_host.hpp"
#include "gui/rmlui/rml_theme.hpp"
#include "gui/rmlui/rmlui_system_interface.hpp"
#include "gui/rotation_gizmo.hpp"
#include "gui/scale_gizmo.hpp"
#include "gui/scene_panel_native.hpp"
#include "gui/string_keys.hpp"
#include "gui/translation_gizmo.hpp"
#include "gui/ui_widgets.hpp"
#include "gui/utils/file_association.hpp"
#include "gui/utils/native_file_dialog.hpp"
#include "gui/vulkan_ui_texture.hpp"
#include <implot.h>

#include "gui/gpu_memory_query.hpp"
#include "gui/gui_focus_state.hpp"
#include "gui/icon_cache.hpp"
#include "input/frame_input_buffer.hpp"
#include "input/input_controller.hpp"
#include "input/sdl_key_mapping.hpp"
#include "internal/resource_paths.hpp"
#include "tools/align_tool.hpp"

#include "core/events.hpp"
#include "core/parameters.hpp"
#include "core/scene.hpp"
#include "python/package_manager.hpp"
#include "python/python_runtime.hpp"
#include "python/runner.hpp"
#include "python/ui_hooks.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "rendering/cuda_vulkan_interop.hpp"
#include "rendering/image_layout.hpp"
#include "rendering/passes/vulkan_viewport_pass.hpp"
#include "rendering/rendering_manager.hpp"
#include "rendering/screen_overlay_renderer.hpp"
#include "scene/scene_manager.hpp"
#include "scene/scene_render_state.hpp"
#include "theme/theme.hpp"
#include "tools/selection_tool.hpp"
#include "training/trainer.hpp"
#include "training/training_manager.hpp"
#include "visualizer/app_store.hpp"
#include "visualizer/scene_coordinate_utils.hpp"
#include "visualizer_impl.hpp"
#include "window/vulkan_context.hpp"
#include "window/window_manager.hpp"
#include <OpenImageIO/imageio.h>
#include <SDL3/SDL.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <format>
#include <fstream>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <imgui_impl_sdl3.h>
#include <imgui_internal.h>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace lfs::vis {
    struct VulkanSceneInteropTarget {
        VulkanContext::ExternalImage image;
        VulkanContext::ExternalSemaphore semaphore;
        lfs::rendering::CudaVulkanInterop interop;
        glm::ivec2 size{0, 0};
        VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
        std::uint64_t timeline_value = 0;
        std::uint64_t generation = 0;
        // Generation of the source content (renderer-supplied) most recently
        // copied into this slot's external image. Used to skip re-uploads when
        // the renderer returns the same logical image (cache HIT) even though
        // it allocated a fresh Tensor pointer.
        std::uint64_t uploaded_source_generation = 0;

        void destroy(VulkanContext& context) {
            if (!context.waitForImmediateSubmits()) {
                LOG_ERROR("Could not drain Vulkan interop transitions before target destruction: {}",
                          context.lastError());
            }
            interop.reset();
            context.destroyExternalSemaphore(semaphore);
            context.destroyExternalImage(image);
            size = {0, 0};
            layout = VK_IMAGE_LAYOUT_UNDEFINED;
            timeline_value = 0;
            uploaded_source_generation = 0;
            ++generation;
        }
    };
} // namespace lfs::vis

namespace lfs::vis::gui {

    namespace {
        const FrameInputBuffer* s_frame_input = nullptr;
        constexpr auto kInteractiveTrainingPauseWait = std::chrono::milliseconds(300);
        constexpr auto kInteractiveTransitionGuardDuration = std::chrono::milliseconds(1200);
        constexpr auto kInteractiveIdleToggleMinInterval = std::chrono::milliseconds(750);
        constexpr auto kInteractiveTrainingToggleMinInterval = std::chrono::milliseconds(3000);
        constexpr auto kCachedImGuiResizeFrameWindow = std::chrono::milliseconds(160);

        [[nodiscard]] std::string formatLodCount(const std::size_t value) {
            constexpr double kThousand = 1'000.0;
            constexpr double kMillion = 1'000'000.0;
            constexpr double kBillion = 1'000'000'000.0;
            const auto numeric = static_cast<double>(value);
            if (numeric >= kBillion)
                return std::format("{:.2f}B", numeric / kBillion);
            if (numeric >= kMillion)
                return std::format("{:.2f}M", numeric / kMillion);
            if (numeric >= kThousand)
                return std::format("{:.1f}K", numeric / kThousand);
            return std::format("{}", value);
        }

        [[nodiscard]] std::string formatLodFloat(const float value) {
            if (!std::isfinite(value))
                return "--";
            const float abs_value = std::abs(value);
            if (abs_value > 0.0f && abs_value < 0.001f)
                return std::format("{:.2e}", value);
            if (abs_value >= 1000.0f)
                return std::format("{:.1f}", value);
            return std::format("{:.4f}", value);
        }

        [[nodiscard]] std::string formatLodPercent(const std::size_t numerator,
                                                   const std::size_t denominator) {
            if (denominator == 0)
                return "";
            const double ratio =
                (static_cast<double>(numerator) / static_cast<double>(denominator)) * 100.0;
            return std::format(" ({:.1f}%)", ratio);
        }

        [[nodiscard]] RmlViewportOverlay::LodStatsOverlayState makeLodStatsOverlayState(
            const SparkLodController::Stats& stats) {
            RmlViewportOverlay::LodStatsOverlayState state;
            state.visible = stats.available || stats.has_tree || stats.enabled;
            if (!state.visible)
                return state;

            if (stats.full_quality_reference) {
                state.status_text = "Full quality reference";
            } else if (stats.active && stats.gpu_selection) {
                state.status_text = "Active, GPU select";
            } else if (stats.active) {
                state.status_text = stats.async_result_ready ? "Active, async ready" : "Active";
            } else if (stats.has_tree || stats.available) {
                state.status_text = stats.enabled ? "Waiting for frame" : "Tree loaded, off";
            } else {
                state.status_text = stats.enabled ? "Enabled, no tree" : "No RAD LOD";
            }

            const std::size_t selected = stats.selected_splats > 0 ? stats.selected_splats : stats.output_size;
            const std::size_t full_quality = stats.full_quality_splats > 0
                                                 ? stats.full_quality_splats
                                                 : stats.model_splats;
            state.selected_text = std::format("{} / {}{}",
                                              formatLodCount(selected),
                                              formatLodCount(full_quality),
                                              formatLodPercent(selected, full_quality));
            if (stats.full_quality_reference) {
                state.budget_text = "leaf set";
            } else if (stats.budget_repair_active &&
                       stats.selected_splats > stats.max_splats) {
                state.budget_text = std::format("{} repair | {} max",
                                                formatLodCount(stats.selected_splats),
                                                formatLodCount(stats.max_splats));
            } else if (stats.requested_max_splats > 0 &&
                       stats.max_splats != stats.requested_max_splats) {
                state.budget_text = std::format("{} result | {} requested",
                                                formatLodCount(stats.max_splats),
                                                formatLodCount(stats.requested_max_splats));
            } else {
                state.budget_text = std::format("{} max", formatLodCount(stats.max_splats));
            }
            state.model_text = std::format("{} splats", formatLodCount(stats.model_splats));
            state.tree_text = std::format("{} nodes | {} internal | {} leaves",
                                          formatLodCount(stats.tree_nodes),
                                          formatLodCount(stats.non_leaf_nodes),
                                          formatLodCount(stats.full_quality_splats));

            std::string stop_reason = "complete";
            if (stats.gpu_selection) {
                stop_reason = stats.budget_limited ? "capacity clamp" : "pixel threshold";
            } else if (stats.full_quality_reference) {
                stop_reason = "leaf complete";
            } else if (stats.output_limited && stats.budget_limited) {
                stop_reason = "budget and output cap";
            } else if (stats.budget_limited) {
                stop_reason = "budget before expansion";
            } else if (stats.output_limited) {
                stop_reason = "output cap";
            } else if (stats.budget_repair_active && stats.budget_limited) {
                stop_reason = "repair cap";
            } else if (stats.budget_repair_active) {
                stop_reason = "budget repair";
            } else if (stats.budget_fill_active && stats.threshold_limited) {
                stop_reason = "budget-fill threshold";
            } else if (stats.budget_fill_active) {
                stop_reason = "budget fill";
            } else if (stats.threshold_limited) {
                stop_reason = "pixel threshold";
            }
            state.traversal_text = stats.gpu_selection
                                       ? std::format("{} output | per-frame GPU cut",
                                                     formatLodCount(stats.output_size))
                                       : std::format("{} output | {} frontier | {} leaf",
                                                     formatLodCount(stats.output_size),
                                                     formatLodCount(stats.frontier_size),
                                                     formatLodCount(stats.leaf_count));
            state.stop_text = stop_reason;
            state.chunks_text = stats.chunk_count > 0
                                    ? std::format("{} touched | {} resident | {} total @ {} splats",
                                                  formatLodCount(stats.touched_chunks),
                                                  formatLodCount(stats.resident_chunks),
                                                  formatLodCount(stats.chunk_count),
                                                  formatLodCount(stats.chunk_splats))
                                    : "--";
            if (stats.pool_pages > 0 && stats.chunk_splats > 0) {
                const std::size_t pool_splats = stats.pool_pages * stats.chunk_splats;
                std::string streaming =
                    stats.streaming_jobs > 0
                        ? std::format("{} pages loading", stats.streaming_jobs)
                        : (stats.resident_chunks >= stats.chunk_count ? "fully resident"
                                                                      : "idle");
                if (stats.deferred_requests > 0) {
                    streaming += std::format(" | {} deferred", stats.deferred_requests);
                }
                if (stats.admission_frozen) {
                    streaming += " | FROZEN";
                }
                const std::size_t resident_pages = std::min(stats.resident_chunks, stats.pool_pages);
                std::string utilization;
                if (stats.gpu_selection && resident_pages > 0) {
                    utilization = std::format(
                        " | util {:.1f}%",
                        100.0 * static_cast<double>(stats.selected_splats) /
                            static_cast<double>(resident_pages * stats.chunk_splats));
                }
                state.cache_text = std::format("{}/{} pages | {} splat pool{} | {}",
                                               formatLodCount(resident_pages),
                                               formatLodCount(stats.pool_pages),
                                               formatLodCount(pool_splats),
                                               utilization,
                                               streaming);
            } else {
                state.cache_text = "--";
            }
            state.selector_text =
                stats.gpu_selection
                    ? std::format("cap {} | overflow {} | feedback x{:.2f}",
                                  formatLodCount(stats.gpu_output_capacity),
                                  formatLodCount(stats.gpu_overflow),
                                  stats.gpu_pixel_scale_feedback)
                    : "CPU traversal";
            // min_pixel_scale and the per-node view counters come from the CPU
            // traversal and are frozen at the bootstrap cut in GPU mode.
            state.pixel_text = stats.gpu_selection
                                   ? std::format("limit {}", formatLodFloat(stats.pixel_scale_limit))
                                   : std::format("limit {} | min {}",
                                                 formatLodFloat(stats.pixel_scale_limit),
                                                 formatLodFloat(stats.min_pixel_scale));
            state.render_text = stats.full_quality_reference
                                    ? "leaf-only reference"
                                    : std::format("LOD scale x{:.1f}", stats.lod_render_scale);
            state.foveation_text = stats.gpu_selection
                                       ? std::format("cone {:.0f}/{:.0f} deg | edge x{:.2f} | behind x{:.2f}",
                                                     stats.cone_inner_degrees,
                                                     stats.cone_outer_degrees,
                                                     stats.cone_foveation,
                                                     stats.behind_camera_penalty)
                                       : std::format("cone {:.0f}/{:.0f} deg | edge x{:.2f} | behind x{:.2f} | view {}/{}",
                                                     stats.cone_inner_degrees,
                                                     stats.cone_outer_degrees,
                                                     stats.cone_foveation,
                                                     stats.behind_camera_penalty,
                                                     formatLodCount(stats.outside_view_nodes),
                                                     formatLodCount(stats.behind_view_nodes));
            state.hash_text = stats.gpu_selection
                                  ? "GPU selector"
                                  : std::format("CPU {:08x}",
                                                static_cast<std::uint32_t>(
                                                    stats.selection_hash & 0xffffffffull));
            return state;
        }

        void capturePressedKeysForRebinding(InputController& input_controller,
                                            const FrameInputBuffer& input) {
            auto& bindings = input_controller.getBindings();
            if (!bindings.isCapturing())
                return;

            const int mods = input::sdlModsToAppMods(input.key_mods);
            for (const SDL_Scancode scancode : input.keys_pressed) {
                const int physical_key = input::sdlScancodeToAppKey(scancode);
                int logical_key = input::sdlKeycodeToAppKey(
                    SDL_GetKeyFromScancode(scancode, SDL_KMOD_NONE, false));
                if (logical_key == input::KEY_UNKNOWN) {
                    logical_key = physical_key;
                }

                bindings.captureKey(physical_key, logical_key, mods);
                if (!bindings.isCapturing())
                    return;
            }
        }

        [[nodiscard]] bool isTransformGizmoOverOrUsing() {
            return isBoundsGizmoHovered() ||
                   isBoundsGizmoActive() ||
                   isRotationGizmoHovered() ||
                   isRotationGizmoActive() ||
                   isScaleGizmoHovered() ||
                   isScaleGizmoActive() ||
                   isTranslationGizmoHovered() ||
                   isTranslationGizmoActive();
        }

        struct VulkanGuidePanelTarget {
            SplitViewPanelId panel = SplitViewPanelId::Left;
            const Viewport* viewport = nullptr;
            glm::vec2 pos{0.0f};
            glm::vec2 size{0.0f};
            glm::ivec2 render_size{0};

            [[nodiscard]] bool valid() const {
                return viewport != nullptr && size.x > 0.0f && size.y > 0.0f &&
                       render_size.x > 0 && render_size.y > 0;
            }
        };

        [[nodiscard]] glm::vec4 guideColor(const ImVec4& color, const float alpha) {
            return {color.x, color.y, color.z, std::clamp(alpha, 0.0f, 1.0f)};
        }

        [[nodiscard]] glm::vec2 screenToViewportNdc(const glm::vec2& screen,
                                                    const glm::vec2& viewport_pos,
                                                    const glm::vec2& viewport_size) {
            const float x = viewport_size.x > 0.0f
                                ? (screen.x - viewport_pos.x) / viewport_size.x
                                : 0.0f;
            const float y = viewport_size.y > 0.0f
                                ? (screen.y - viewport_pos.y) / viewport_size.y
                                : 0.0f;
            return {x * 2.0f - 1.0f, y * 2.0f - 1.0f};
        }

        void appendShapeOverlayTriangle(std::vector<VulkanViewportShapeOverlayVertex>& out,
                                        const glm::vec2& viewport_pos,
                                        const glm::vec2& viewport_size,
                                        const glm::vec2& point,
                                        const glm::vec2& p0,
                                        const glm::vec2& p1,
                                        const glm::vec4& color,
                                        const glm::vec4& shape_params,
                                        const float view_depth = 0.0f) {
            out.push_back({
                .position = screenToViewportNdc(point, viewport_pos, viewport_size),
                .screen_position = point,
                .p0 = p0,
                .p1 = p1,
                .color = color,
                .params = shape_params,
                .view_depth = view_depth,
            });
        }

        void appendShapeOverlayQuad(std::vector<VulkanViewportShapeOverlayVertex>& out,
                                    const glm::vec2& viewport_pos,
                                    const glm::vec2& viewport_size,
                                    const glm::vec2& a,
                                    const glm::vec2& b,
                                    const glm::vec2& c,
                                    const glm::vec2& d,
                                    const glm::vec2& p0,
                                    const glm::vec2& p1,
                                    const glm::vec4& color,
                                    const glm::vec4& shape_params,
                                    const float depth_a = 0.0f,
                                    const float depth_b = 0.0f,
                                    const float depth_c = 0.0f,
                                    const float depth_d = 0.0f) {
            if (color.a <= 0.0f || viewport_size.x <= 0.0f || viewport_size.y <= 0.0f) {
                return;
            }
            appendShapeOverlayTriangle(out, viewport_pos, viewport_size, a, p0, p1, color, shape_params, depth_a);
            appendShapeOverlayTriangle(out, viewport_pos, viewport_size, b, p0, p1, color, shape_params, depth_b);
            appendShapeOverlayTriangle(out, viewport_pos, viewport_size, c, p0, p1, color, shape_params, depth_c);
            appendShapeOverlayTriangle(out, viewport_pos, viewport_size, a, p0, p1, color, shape_params, depth_a);
            appendShapeOverlayTriangle(out, viewport_pos, viewport_size, c, p0, p1, color, shape_params, depth_c);
            appendShapeOverlayTriangle(out, viewport_pos, viewport_size, d, p0, p1, color, shape_params, depth_d);
        }

        void appendShapeOverlayLine(std::vector<VulkanViewportShapeOverlayVertex>& out,
                                    const VulkanViewportPassParams& params,
                                    const glm::vec2& p0,
                                    const glm::vec2& p1,
                                    const glm::vec4& color,
                                    const float thickness,
                                    const float view_depth_p0 = 0.0f,
                                    const float view_depth_p1 = 0.0f) {
            if (color.a <= 0.0f) {
                return;
            }
            const glm::vec2 delta = p1 - p0;
            const float len = glm::length(delta);
            if (!std::isfinite(len) || len <= 1e-4f) {
                return;
            }
            const glm::vec2 dir = delta / len;
            const glm::vec2 normal(-dir.y, dir.x);
            const float extent = std::max(thickness, 1.0f) * 0.5f + 2.0f;
            appendShapeOverlayQuad(out,
                                   params.viewport_pos,
                                   params.viewport_size,
                                   p0 - dir * extent + normal * extent,
                                   p1 + dir * extent + normal * extent,
                                   p1 + dir * extent - normal * extent,
                                   p0 - dir * extent - normal * extent,
                                   p0,
                                   p1,
                                   color,
                                   {0.0f, std::max(thickness, 1.0f), 0.0f, 1.0f},
                                   view_depth_p0,
                                   view_depth_p1,
                                   view_depth_p1,
                                   view_depth_p0);
        }

        void appendShapeOverlayCircle(std::vector<VulkanViewportShapeOverlayVertex>& out,
                                      const VulkanViewportPassParams& params,
                                      const glm::vec2& center,
                                      const float radius,
                                      const glm::vec4& color) {
            if (radius <= 0.0f || color.a <= 0.0f) {
                return;
            }
            const float extent = radius + 2.0f;
            appendShapeOverlayQuad(out,
                                   params.viewport_pos,
                                   params.viewport_size,
                                   center + glm::vec2(-extent, -extent),
                                   center + glm::vec2(extent, -extent),
                                   center + glm::vec2(extent, extent),
                                   center + glm::vec2(-extent, extent),
                                   center,
                                   center,
                                   color,
                                   {1.0f, 0.0f, radius, 1.0f});
        }

        void appendShapeOverlayCircleOutline(std::vector<VulkanViewportShapeOverlayVertex>& out,
                                             const VulkanViewportPassParams& params,
                                             const glm::vec2& center,
                                             const float radius,
                                             const glm::vec4& color,
                                             const float thickness) {
            if (radius <= 0.0f || color.a <= 0.0f) {
                return;
            }
            const float extent = radius + std::max(thickness, 1.0f) * 0.5f + 2.0f;
            appendShapeOverlayQuad(out,
                                   params.viewport_pos,
                                   params.viewport_size,
                                   center + glm::vec2(-extent, -extent),
                                   center + glm::vec2(extent, -extent),
                                   center + glm::vec2(extent, extent),
                                   center + glm::vec2(-extent, extent),
                                   center,
                                   center,
                                   color,
                                   {2.0f, std::max(thickness, 1.0f), radius, 1.0f});
        }

        void appendScreenOverlayTriangle(std::vector<VulkanViewportOverlayVertex>& out,
                                         const VulkanViewportPassParams& params,
                                         const glm::vec2& p0,
                                         const glm::vec2& p1,
                                         const glm::vec2& p2,
                                         const glm::vec4& color) {
            if (color.a <= 0.0f || params.viewport_size.x <= 0.0f || params.viewport_size.y <= 0.0f) {
                return;
            }
            out.push_back({.position = screenToViewportNdc(p0, params.viewport_pos, params.viewport_size),
                           .color = color});
            out.push_back({.position = screenToViewportNdc(p1, params.viewport_pos, params.viewport_size),
                           .color = color});
            out.push_back({.position = screenToViewportNdc(p2, params.viewport_pos, params.viewport_size),
                           .color = color});
        }

        void appendViewportDimOverlay(VulkanViewportPassParams& params) {
            if (params.viewport_size.x <= 0.0f || params.viewport_size.y <= 0.0f) {
                return;
            }

            constexpr std::uint32_t kDimOverlayVertexCount = 6;
            const glm::vec2 a = params.viewport_pos;
            const glm::vec2 b = params.viewport_pos + glm::vec2(params.viewport_size.x, 0.0f);
            const glm::vec2 c = params.viewport_pos + params.viewport_size;
            const glm::vec2 d = params.viewport_pos + glm::vec2(0.0f, params.viewport_size.y);
            const glm::vec4 dim_color{0.08f, 0.09f, 0.11f, 0.62f};
            appendScreenOverlayTriangle(params.overlay_triangles, params, a, b, c, dim_color);
            appendScreenOverlayTriangle(params.overlay_triangles, params, a, c, d, dim_color);
            params.post_ui_overlay_vertex_count += kDimOverlayVertexCount;
        }

        void appendLineRendererCommandOverlays(VulkanViewportPassParams& params) {
            const auto commands = consumeLineRendererCommands();
            for (const auto& command : commands) {
                switch (command.type) {
                case LineRendererCommandType::Line:
                    appendShapeOverlayLine(params.ui_shape_overlay_triangles,
                                           params,
                                           command.p0,
                                           command.p1,
                                           command.color,
                                           command.thickness);
                    break;
                case LineRendererCommandType::Triangle:
                    appendScreenOverlayTriangle(params.overlay_triangles,
                                                params,
                                                command.p0,
                                                command.p1,
                                                command.p2,
                                                command.color);
                    break;
                case LineRendererCommandType::Circle:
                    appendShapeOverlayCircle(params.ui_shape_overlay_triangles,
                                             params,
                                             command.p0,
                                             command.thickness,
                                             command.color);
                    break;
                case LineRendererCommandType::CircleOutline:
                    appendShapeOverlayCircleOutline(params.ui_shape_overlay_triangles,
                                                    params,
                                                    command.p0,
                                                    command.radius,
                                                    command.color,
                                                    command.thickness);
                    break;
                }
            }
        }

        void appendTexturedOverlayQuad(VulkanViewportPassParams& params,
                                       std::uintptr_t texture_id,
                                       const std::array<glm::vec2, 4>& screen_points,
                                       const glm::vec2& uv_min,
                                       const glm::vec2& uv_max,
                                       const glm::vec4& tint_opacity,
                                       const glm::vec4& effects,
                                       const std::array<float, 4>& view_depths = {0.0f, 0.0f, 0.0f, 0.0f});

        // FreeType-baked overlay font atlas. Independent of ImGui's font system — the user
        // wants the vulkan branch to stop reaching into ImFont/io.Fonts. Atlas covers ASCII
        // printable range, baked at a single reference pixel size; runtime font_size scales
        // glyph quads.
        struct OverlayGlyph {
            glm::vec2 uv0{0.0f};
            glm::vec2 uv1{0.0f};
            glm::vec2 size_px{0.0f};
            glm::vec2 bearing_px{0.0f}; // x = bitmap_left, y = bitmap_top (positive up)
            float advance_px = 0.0f;
        };

        struct OverlayFontAtlas {
            VulkanUiTexture texture;
            std::array<OverlayGlyph, 128> glyphs{};
            float atlas_px_size = 0.0f;
            bool valid = false;
        };

        OverlayFontAtlas g_overlay_atlas;

        bool buildOverlayFontAtlas(const std::filesystem::path& font_path, const float pixel_size) {
            g_overlay_atlas = {};

            FT_Library ft = nullptr;
            if (FT_Init_FreeType(&ft) != 0) {
                LOG_WARN("Overlay font atlas: FT_Init_FreeType failed");
                return false;
            }
            struct FTLib {
                FT_Library lib;
                ~FTLib() {
                    if (lib)
                        FT_Done_FreeType(lib);
                }
            } ft_guard{ft};

            const std::string font_utf8 = lfs::core::path_to_utf8(font_path);
            FT_Face face = nullptr;
            if (FT_New_Face(ft, font_utf8.c_str(), 0, &face) != 0) {
                LOG_WARN("Overlay font atlas: FT_New_Face failed for {}", font_utf8);
                return false;
            }
            struct FTFace {
                FT_Face face;
                ~FTFace() {
                    if (face)
                        FT_Done_Face(face);
                }
            } face_guard{face};

            const FT_UInt px = static_cast<FT_UInt>(std::max(8.0f, pixel_size));
            FT_Set_Pixel_Sizes(face, 0, px);

            constexpr int kAtlasW = 1024;
            constexpr int kPad = 1;
            std::vector<std::uint8_t> rgba(static_cast<std::size_t>(kAtlasW) * kAtlasW * 4, 0);

            int pen_x = kPad;
            int pen_y = kPad;
            int row_h = 0;
            int max_y = 0;

            for (int code = 32; code < 128; ++code) {
                if (FT_Load_Char(face, static_cast<FT_ULong>(code), FT_LOAD_RENDER) != 0) {
                    continue;
                }
                const FT_GlyphSlot g = face->glyph;
                const int w = static_cast<int>(g->bitmap.width);
                const int h = static_cast<int>(g->bitmap.rows);

                if (pen_x + w + kPad > kAtlasW) {
                    pen_x = kPad;
                    pen_y += row_h + kPad;
                    row_h = 0;
                }
                if (pen_y + h + kPad > kAtlasW) {
                    LOG_WARN("Overlay font atlas: out of space at glyph {}", code);
                    break;
                }

                for (int y = 0; y < h; ++y) {
                    const std::uint8_t* src = g->bitmap.buffer + y * g->bitmap.pitch;
                    std::uint8_t* dst = rgba.data() + ((pen_y + y) * kAtlasW + pen_x) * 4;
                    for (int x = 0; x < w; ++x) {
                        const std::uint8_t a = src[x];
                        dst[x * 4 + 0] = 255;
                        dst[x * 4 + 1] = 255;
                        dst[x * 4 + 2] = 255;
                        dst[x * 4 + 3] = a;
                    }
                }

                OverlayGlyph& og = g_overlay_atlas.glyphs[code];
                og.uv0 = {static_cast<float>(pen_x) / kAtlasW,
                          static_cast<float>(pen_y) / kAtlasW};
                og.uv1 = {static_cast<float>(pen_x + w) / kAtlasW,
                          static_cast<float>(pen_y + h) / kAtlasW};
                og.size_px = {static_cast<float>(w), static_cast<float>(h)};
                og.bearing_px = {static_cast<float>(g->bitmap_left),
                                 static_cast<float>(g->bitmap_top)};
                og.advance_px = static_cast<float>(g->advance.x) / 64.0f;

                pen_x += w + kPad;
                row_h = std::max(row_h, h);
                max_y = std::max(max_y, pen_y + h);
            }

            if (!g_overlay_atlas.texture.upload(rgba.data(), kAtlasW, kAtlasW, 4)) {
                LOG_WARN("Overlay font atlas: Vulkan upload failed");
                return false;
            }

            g_overlay_atlas.atlas_px_size = static_cast<float>(px);
            g_overlay_atlas.valid = true;
            LOG_INFO("Overlay font atlas baked at {} px ({}x{} pixels used)",
                     px, kAtlasW, max_y + kPad);
            return true;
        }

        glm::vec2 measureOverlayText(std::string_view text, float size_px) {
            if (!g_overlay_atlas.valid || text.empty() || size_px <= 0.0f) {
                return {0.0f, 0.0f};
            }
            const float scale = size_px / g_overlay_atlas.atlas_px_size;
            float width = 0.0f;
            for (const char ch : text) {
                const auto code = static_cast<unsigned char>(ch);
                if (code < g_overlay_atlas.glyphs.size()) {
                    width += g_overlay_atlas.glyphs[code].advance_px;
                }
            }
            return {width * scale, size_px};
        }

        void appendTextOverlay(VulkanViewportPassParams& params,
                               const lfs::rendering::OverlayCommand& cmd) {
            if (!g_overlay_atlas.valid || cmd.text.empty() || cmd.font_size <= 0.0f ||
                cmd.color_premul.a <= 0.0f) {
                return;
            }
            const std::uintptr_t texture_id = g_overlay_atlas.texture.textureId();
            if (texture_id == 0) {
                return;
            }
            const float scale = cmd.font_size / g_overlay_atlas.atlas_px_size;
            const float baseline_y = cmd.p0.y + cmd.font_size; // top_left → baseline
            float pen_x = cmd.p0.x;

            for (const char ch : cmd.text) {
                const auto code = static_cast<unsigned char>(ch);
                if (code >= g_overlay_atlas.glyphs.size()) {
                    continue;
                }
                const OverlayGlyph& g = g_overlay_atlas.glyphs[code];
                if (g.size_px.x > 0.0f && g.size_px.y > 0.0f) {
                    const float x0 = pen_x + g.bearing_px.x * scale;
                    const float y0 = baseline_y - g.bearing_px.y * scale;
                    const float x1 = x0 + g.size_px.x * scale;
                    const float y1 = y0 + g.size_px.y * scale;
                    const std::array<glm::vec2, 4> pts = {
                        glm::vec2{x0, y0}, glm::vec2{x1, y0},
                        glm::vec2{x1, y1}, glm::vec2{x0, y1}};
                    appendTexturedOverlayQuad(params, texture_id, pts,
                                              g.uv0, g.uv1,
                                              cmd.color_premul, {1.0f, 0.0f, 0.0f, 0.0f});
                }
                pen_x += g.advance_px * scale;
            }
        }

        void appendScreenOverlayCommandOverlays(VulkanViewportPassParams& params,
                                                lfs::rendering::ScreenOverlayRenderer* overlay) {
            if (!overlay) {
                return;
            }
            const auto commands = overlay->consumeCommands();
            for (const auto& command : commands) {
                switch (command.type) {
                case lfs::rendering::OverlayCommandType::Line:
                    appendShapeOverlayLine(params.ui_shape_overlay_triangles,
                                           params,
                                           command.p0,
                                           command.p1,
                                           command.color_premul,
                                           command.thickness);
                    break;
                case lfs::rendering::OverlayCommandType::Triangle:
                    appendScreenOverlayTriangle(params.overlay_triangles,
                                                params,
                                                command.p0,
                                                command.p1,
                                                command.p2,
                                                command.color_premul);
                    break;
                case lfs::rendering::OverlayCommandType::CircleFilled:
                    appendShapeOverlayCircle(params.ui_shape_overlay_triangles,
                                             params,
                                             command.p0,
                                             command.radius,
                                             command.color_premul);
                    break;
                case lfs::rendering::OverlayCommandType::CircleOutline:
                    appendShapeOverlayCircleOutline(params.ui_shape_overlay_triangles,
                                                    params,
                                                    command.p0,
                                                    command.radius,
                                                    command.color_premul,
                                                    command.thickness);
                    break;
                case lfs::rendering::OverlayCommandType::Text:
                    appendTextOverlay(params, command);
                    break;
                }
            }
        }

        void addResolvedVulkanPanel(std::vector<VulkanGuidePanelTarget>& panels,
                                    const std::optional<RenderingManager::ViewerPanelInfo>& info_opt) {
            if (!info_opt || !info_opt->valid()) {
                return;
            }
            const auto& info = *info_opt;
            panels.push_back({
                .panel = info.panel,
                .viewport = info.viewport,
                .pos = {info.x, info.y},
                .size = {info.width, info.height},
                .render_size = {info.render_width, info.render_height},
            });
        }

        [[nodiscard]] std::vector<VulkanGuidePanelTarget> collectVulkanGuidePanels(
            const VisualizerImpl& viewer,
            const ViewportLayout& viewport_layout,
            const RenderingManager& rendering_manager) {
            std::vector<VulkanGuidePanelTarget> panels;
            panels.reserve(2);

            const auto& viewport = viewer.getViewport();
            if (rendering_manager.isIndependentSplitViewActive()) {
                addResolvedVulkanPanel(panels, rendering_manager.resolveViewerPanel(
                                                   viewport, viewport_layout.pos, viewport_layout.size,
                                                   std::nullopt, SplitViewPanelId::Left));
                addResolvedVulkanPanel(panels, rendering_manager.resolveViewerPanel(
                                                   viewport, viewport_layout.pos, viewport_layout.size,
                                                   std::nullopt, SplitViewPanelId::Right));
            }

            if (!panels.empty()) {
                return panels;
            }

            const glm::ivec2 render_size(
                std::max(static_cast<int>(std::round(viewport_layout.size.x)), 1),
                std::max(static_cast<int>(std::round(viewport_layout.size.y)), 1));
            panels.push_back({
                .panel = SplitViewPanelId::Left,
                .viewport = &viewport,
                .pos = viewport_layout.pos,
                .size = viewport_layout.size,
                .render_size = render_size,
            });
            return panels;
        }

        [[nodiscard]] glm::vec2 renderToPanelScreen(const VulkanGuidePanelTarget& panel,
                                                    const glm::vec2& projected) {
            const float sx = panel.size.x / static_cast<float>(std::max(panel.render_size.x, 1));
            const float sy = panel.size.y / static_cast<float>(std::max(panel.render_size.y, 1));
            return glm::vec2(panel.pos.x + projected.x * sx, panel.pos.y + projected.y * sy);
        }

        struct ProjectedSegment {
            glm::vec2 a{0.0f};
            glm::vec2 b{0.0f};
            float depth_a = 0.0f;
            float depth_b = 0.0f;
        };

        [[nodiscard]] std::optional<ProjectedSegment> projectSegmentToScreenClipped(
            const VulkanGuidePanelTarget& panel,
            const RenderSettings& settings,
            const glm::vec3& world_a,
            const glm::vec3& world_b) {
            if (settings.equirectangular) {
                const glm::mat3 rotation = panel.viewport->getRotationMatrix();
                const glm::vec3 translation = panel.viewport->getTranslation();

                struct EquirectProjected {
                    glm::vec2 screen;
                    float depth;
                };
                const auto project_equirect = [&](const glm::vec3& world) -> std::optional<EquirectProjected> {
                    const glm::vec3 view = glm::transpose(rotation) * (world - translation);
                    const float len = glm::length(view);
                    if (!std::isfinite(len) || len <= 1e-6f) {
                        return std::nullopt;
                    }
                    const glm::vec3 dir = view / len;
                    const float ndc_x = std::atan2(dir.x, -dir.z) / glm::pi<float>();
                    const float ndc_y = -std::asin(std::clamp(dir.y, -1.0f, 1.0f)) /
                                        (glm::pi<float>() * 0.5f);
                    if (!std::isfinite(ndc_x) || !std::isfinite(ndc_y)) {
                        return std::nullopt;
                    }
                    return EquirectProjected{
                        .screen = panel.pos + glm::vec2((ndc_x * 0.5f + 0.5f) * panel.size.x,
                                                        (ndc_y * 0.5f + 0.5f) * panel.size.y),
                        .depth = len,
                    };
                };

                const auto pa = project_equirect(world_a);
                const auto pb = project_equirect(world_b);
                if (!pa || !pb || std::abs((pa->screen.x - panel.pos.x) / panel.size.x - (pb->screen.x - panel.pos.x) / panel.size.x) > 0.5f) {
                    return std::nullopt;
                }
                return ProjectedSegment{.a = pa->screen, .b = pb->screen, .depth_a = pa->depth, .depth_b = pb->depth};
            }

            constexpr float kMinViewZ = -1e-4f;
            const glm::mat3 rotation = panel.viewport->getRotationMatrix();
            const glm::vec3 translation = panel.viewport->getTranslation();
            glm::vec3 view_a = glm::transpose(rotation) * (world_a - translation);
            glm::vec3 view_b = glm::transpose(rotation) * (world_b - translation);

            if (view_a.z >= kMinViewZ && view_b.z >= kMinViewZ) {
                return std::nullopt;
            }

            const auto clip_to_near = [](glm::vec3& inside, glm::vec3& outside) {
                const float denom = outside.z - inside.z;
                if (std::abs(denom) <= 1e-8f) {
                    return;
                }
                const float t = (-1e-4f - inside.z) / denom;
                outside = glm::mix(inside, outside, std::clamp(t, 0.0f, 1.0f));
                outside.z = -1e-4f;
            };
            if (view_a.z >= kMinViewZ) {
                clip_to_near(view_b, view_a);
            }
            if (view_b.z >= kMinViewZ) {
                clip_to_near(view_a, view_b);
            }

            const auto project_view = [&](const glm::vec3& view) -> std::optional<glm::vec2> {
                const float width = static_cast<float>(std::max(panel.render_size.x, 1));
                const float height = static_cast<float>(std::max(panel.render_size.y, 1));
                const float cx = width * 0.5f;
                const float cy = height * 0.5f;
                if (settings.orthographic) {
                    if (!std::isfinite(settings.ortho_scale) || settings.ortho_scale <= 0.0f) {
                        return std::nullopt;
                    }
                    return glm::vec2(cx + view.x * settings.ortho_scale,
                                     cy - view.y * settings.ortho_scale);
                }
                const auto [fx, fy] = lfs::rendering::computePixelFocalLengths(
                    panel.render_size, settings.focal_length_mm);
                const float depth = -view.z;
                if (depth <= 0.0f) {
                    return std::nullopt;
                }
                return glm::vec2(cx + view.x * fx / depth,
                                 cy - view.y * fy / depth);
            };

            const auto pa = project_view(view_a);
            const auto pb = project_view(view_b);
            if (!pa || !pb) {
                return std::nullopt;
            }
            return ProjectedSegment{
                .a = renderToPanelScreen(panel, *pa),
                .b = renderToPanelScreen(panel, *pb),
                .depth_a = -view_a.z,
                .depth_b = -view_b.z,
            };
        }

        [[nodiscard]] bool projectedQuadVisible(const std::array<glm::vec2, 4>& points,
                                                const VulkanGuidePanelTarget& panel) {
            glm::vec2 min_point(std::numeric_limits<float>::max());
            glm::vec2 max_point(-std::numeric_limits<float>::max());
            float area_twice = 0.0f;
            for (size_t i = 0; i < points.size(); ++i) {
                const glm::vec2& p = points[i];
                if (!std::isfinite(p.x) || !std::isfinite(p.y)) {
                    return false;
                }
                min_point = glm::min(min_point, p);
                max_point = glm::max(max_point, p);
                const glm::vec2& q = points[(i + 1) % points.size()];
                area_twice += p.x * q.y - q.x * p.y;
            }

            const glm::vec2 panel_min = panel.pos;
            const glm::vec2 panel_max = panel.pos + panel.size;
            if (max_point.x < panel_min.x || max_point.y < panel_min.y ||
                min_point.x > panel_max.x || min_point.y > panel_max.y) {
                return false;
            }

            const glm::vec2 extent = max_point - min_point;
            const float panel_limit = std::max(panel.size.x, panel.size.y) * 8.0f;
            return std::abs(area_twice) >= 2.0f &&
                   extent.x <= panel_limit &&
                   extent.y <= panel_limit;
        }

        void appendTexturedOverlayQuad(VulkanViewportPassParams& params,
                                       const std::uintptr_t texture_id,
                                       const std::array<glm::vec2, 4>& screen_points,
                                       const glm::vec2& uv_min,
                                       const glm::vec2& uv_max,
                                       const glm::vec4& tint_opacity,
                                       const glm::vec4& effects,
                                       const std::array<float, 4>& view_depths) {
            if (texture_id == 0 || tint_opacity.a <= 0.0f ||
                params.viewport_size.x <= 0.0f || params.viewport_size.y <= 0.0f) {
                return;
            }

            const auto ndc = [&](const glm::vec2& screen) {
                return screenToViewportNdc(screen, params.viewport_pos, params.viewport_size);
            };

            VulkanViewportTexturedOverlay overlay{};
            overlay.texture_id = texture_id;
            overlay.tint_opacity = tint_opacity;
            overlay.effects = effects;
            overlay.vertices = {{
                {.position = ndc(screen_points[0]), .uv = {uv_min.x, uv_min.y}, .view_depth = view_depths[0]},
                {.position = ndc(screen_points[1]), .uv = {uv_max.x, uv_min.y}, .view_depth = view_depths[1]},
                {.position = ndc(screen_points[2]), .uv = {uv_max.x, uv_max.y}, .view_depth = view_depths[2]},
                {.position = ndc(screen_points[0]), .uv = {uv_min.x, uv_min.y}, .view_depth = view_depths[0]},
                {.position = ndc(screen_points[2]), .uv = {uv_max.x, uv_max.y}, .view_depth = view_depths[2]},
                {.position = ndc(screen_points[3]), .uv = {uv_min.x, uv_max.y}, .view_depth = view_depths[3]},
            }};
            params.textured_overlays.push_back(overlay);
        }

        struct VulkanViewportGizmoMarker {
            int encoded_axis = -1;
            int axis = 0;
            bool negative = false;
            glm::vec2 screen_pos{0.0f};
            float radius = 0.0f;
            float depth = 0.0f;
            bool visible = false;
        };

        struct VulkanViewportGizmoLayout {
            glm::vec2 top_left{0.0f};
            glm::vec2 center{0.0f};
            float size = 0.0f;
            std::array<VulkanViewportGizmoMarker, 6> markers{};
        };

        constexpr std::array<glm::vec3, 3> kViewportGizmoAxisColors = {
            glm::vec3{0.89f, 0.15f, 0.21f},
            glm::vec3{0.54f, 0.86f, 0.20f},
            glm::vec3{0.17f, 0.48f, 0.87f}};
        constexpr float kViewportGizmoSize = 95.0f;
        constexpr float kViewportGizmoMarginX = 10.0f;
        constexpr float kViewportGizmoMarginY = 10.0f;
        constexpr float kViewportGizmoDistance = 2.8f;
        constexpr float kViewportGizmoFovDegrees = 38.0f;
        constexpr float kViewportGizmoSphereRadius = 0.198f;
        constexpr float kViewportGizmoLabelDistance = 0.63f;
        constexpr float kViewportGizmoHoverScale = 1.2f;
        constexpr float kViewportGizmoHoverBrightness = 1.3f;
        constexpr float kViewportGizmoHitRadiusScale = 2.5f;
        constexpr float kViewportGizmoRingInnerRadius = 0.55f;

        [[nodiscard]] glm::vec4 viewportGizmoAxisColor(const int axis,
                                                       const float alpha,
                                                       const float brightness = 1.0f) {
            const glm::vec3 c = glm::clamp(kViewportGizmoAxisColors[static_cast<size_t>(axis)] * brightness,
                                           glm::vec3(0.0f), glm::vec3(1.0f));
            return {c.r, c.g, c.b, std::clamp(alpha, 0.0f, 1.0f)};
        }

        [[nodiscard]] std::optional<VulkanViewportGizmoLayout> buildViewportGizmoLayout(
            const VulkanGuidePanelTarget& panel,
            const float size,
            const float margin_x,
            const float margin_y) {
            if (!panel.valid() || size <= 0.0f) {
                return std::nullopt;
            }

            VulkanViewportGizmoLayout layout;
            layout.size = size;
            layout.top_left = {
                panel.pos.x + panel.size.x - size - margin_x,
                panel.pos.y + margin_y,
            };
            layout.center = layout.top_left + glm::vec2(size * 0.5f);

            glm::mat4 view = lfs::rendering::makeViewMatrix(panel.viewport->getRotationMatrix(), glm::vec3(0.0f));
            view[3][2] = -kViewportGizmoDistance;
            const glm::mat4 proj =
                glm::perspective(glm::radians(kViewportGizmoFovDegrees), 1.0f, 0.1f, 10.0f);
            const float projected_marker_radius =
                kViewportGizmoSphereRadius *
                (1.0f / std::tan(glm::radians(kViewportGizmoFovDegrees) * 0.5f)) /
                kViewportGizmoDistance *
                size * 0.5f;

            const auto project_marker = [&](const int axis, const bool negative) {
                VulkanViewportGizmoMarker marker;
                marker.axis = axis;
                marker.negative = negative;
                marker.encoded_axis = axis + (negative ? 3 : 0);

                glm::vec3 position(0.0f);
                position[axis] = negative ? -kViewportGizmoLabelDistance : kViewportGizmoLabelDistance;

                const glm::vec4 clip = proj * view * glm::vec4(position, 1.0f);
                if (clip.w <= 0.0f) {
                    return marker;
                }

                const glm::vec3 ndc = glm::vec3(clip) / clip.w;
                const float local_x = (ndc.x * 0.5f + 0.5f) * size;
                const float local_y = (1.0f - (ndc.y * 0.5f + 0.5f)) * size;
                marker.screen_pos = layout.top_left + glm::vec2(local_x, local_y);
                marker.radius = projected_marker_radius;
                marker.depth = clip.z / clip.w;
                marker.visible = true;
                return marker;
            };

            for (int axis = 0; axis < 3; ++axis) {
                layout.markers[static_cast<size_t>(axis)] = project_marker(axis, false);
                layout.markers[static_cast<size_t>(axis + 3)] = project_marker(axis, true);
            }
            return layout;
        }

        [[nodiscard]] int hitTestViewportGizmoLayout(const VulkanViewportGizmoLayout& layout,
                                                     const glm::vec2& mouse_pos) {
            for (const auto& marker : layout.markers) {
                if (!marker.visible) {
                    continue;
                }
                const glm::vec2 delta = mouse_pos - marker.screen_pos;
                const float radius = marker.radius * kViewportGizmoHitRadiusScale;
                if (glm::dot(delta, delta) <= radius * radius) {
                    return marker.encoded_axis;
                }
            }
            return -1;
        }

        void appendViewportGizmoLabel(std::vector<VulkanViewportShapeOverlayVertex>& out,
                                      const VulkanViewportPassParams& params,
                                      const glm::vec2& center,
                                      const int axis,
                                      const float radius,
                                      const float ui_scale) {
            const glm::vec4 white(1.0f, 1.0f, 1.0f, 0.96f);
            const float s = std::max(radius * 0.42f, 3.0f * ui_scale);
            const float thickness = std::max(radius * 0.16f, 1.25f * ui_scale);
            if (axis == 0) {
                appendShapeOverlayLine(out, params, center + glm::vec2(-s, -s), center + glm::vec2(s, s),
                                       white, thickness);
                appendShapeOverlayLine(out, params, center + glm::vec2(-s, s), center + glm::vec2(s, -s),
                                       white, thickness);
            } else if (axis == 1) {
                appendShapeOverlayLine(out, params, center + glm::vec2(-s, -s), center, white, thickness);
                appendShapeOverlayLine(out, params, center + glm::vec2(s, -s), center, white, thickness);
                appendShapeOverlayLine(out, params, center, center + glm::vec2(0.0f, s), white, thickness);
            } else {
                appendShapeOverlayLine(out, params, center + glm::vec2(-s, -s), center + glm::vec2(s, -s),
                                       white, thickness);
                appendShapeOverlayLine(out, params, center + glm::vec2(s, -s), center + glm::vec2(-s, s),
                                       white, thickness);
                appendShapeOverlayLine(out, params, center + glm::vec2(-s, s), center + glm::vec2(s, s),
                                       white, thickness);
            }
        }

        void appendViewportGizmoLayout(std::vector<VulkanViewportShapeOverlayVertex>& out,
                                       const VulkanViewportPassParams& params,
                                       const VulkanViewportGizmoLayout& layout,
                                       const int hovered_axis) {
            const auto& t = theme();
            const float ui_scale = std::max(layout.size / kViewportGizmoSize, 0.01f);

            std::vector<const VulkanViewportGizmoMarker*> draw_order;
            draw_order.reserve(layout.markers.size());
            for (const auto& marker : layout.markers) {
                if (marker.visible) {
                    draw_order.push_back(&marker);
                }
            }
            std::sort(draw_order.begin(), draw_order.end(),
                      [](const auto* a, const auto* b) {
                          return a->depth > b->depth;
                      });

            for (const auto& marker : layout.markers) {
                if (!marker.visible || marker.negative) {
                    continue;
                }
                appendShapeOverlayLine(out, params, layout.center, marker.screen_pos,
                                       viewportGizmoAxisColor(marker.axis, 0.72f), 3.0f * ui_scale);
            }

            for (const auto* const marker_ptr : draw_order) {
                const auto& marker = *marker_ptr;
                const bool hovered = hovered_axis == marker.encoded_axis;
                const float radius = marker.radius * (hovered ? kViewportGizmoHoverScale : 1.0f);
                const float brightness = hovered ? kViewportGizmoHoverBrightness : 1.0f;
                const glm::vec4 color = viewportGizmoAxisColor(marker.axis,
                                                               marker.negative ? 0.88f : 1.0f,
                                                               brightness);
                if (marker.negative) {
                    const float ring_center_radius =
                        radius * (1.0f + kViewportGizmoRingInnerRadius) * 0.5f;
                    const float ring_thickness =
                        radius * (1.0f - kViewportGizmoRingInnerRadius);
                    appendShapeOverlayCircleOutline(out, params, marker.screen_pos, ring_center_radius,
                                                    color, ring_thickness);
                } else {
                    appendShapeOverlayCircle(out, params, marker.screen_pos, radius, color);
                    appendShapeOverlayCircleOutline(out, params, marker.screen_pos, radius,
                                                    guideColor(t.palette.background, 0.55f), 1.0f * ui_scale);
                    appendViewportGizmoLabel(out, params, marker.screen_pos, marker.axis, radius, ui_scale);
                }
            }
        }

        void appendVulkanViewportGizmoOverlay(VulkanViewportPassParams& params,
                                              VisualizerImpl& viewer,
                                              const ViewportLayout& viewport_layout,
                                              RenderingManager& rendering_manager,
                                              const float ui_scale,
                                              const bool dragging) {
            if (params.viewport_size.x <= 0.0f || params.viewport_size.y <= 0.0f) {
                return;
            }

            const auto panels = collectVulkanGuidePanels(viewer, viewport_layout, rendering_manager);
            if (panels.empty()) {
                return;
            }
            const float gizmo_scale = std::max(1.0f, ui_scale);
            const float gizmo_size = kViewportGizmoSize * gizmo_scale;
            const float gizmo_margin_x = kViewportGizmoMarginX * gizmo_scale;
            const float gizmo_margin_y = kViewportGizmoMarginY * gizmo_scale;

            int hovered_axis = -1;
            SplitViewPanelId hovered_panel = SplitViewPanelId::Left;
            bool has_hovered_panel = false;
            if (!guiFocusState().want_capture_mouse) {
                if (auto* const window_manager = viewer.getWindowManager()) {
                    const auto& frame_input = window_manager->frameInput();
                    const glm::vec2 mouse(frame_input.mouse_x, frame_input.mouse_y);
                    for (const auto& panel : panels) {
                        const float gizmo_x = panel.pos.x + panel.size.x -
                                              gizmo_size - gizmo_margin_x;
                        const float gizmo_y = panel.pos.y + gizmo_margin_y;
                        const bool mouse_in_gizmo = mouse.x >= gizmo_x &&
                                                    mouse.x <= gizmo_x + gizmo_size &&
                                                    mouse.y >= gizmo_y &&
                                                    mouse.y <= gizmo_y + gizmo_size;
                        if (!mouse_in_gizmo) {
                            continue;
                        }
                        if (const auto layout = buildViewportGizmoLayout(
                                panel, gizmo_size, gizmo_margin_x, gizmo_margin_y)) {
                            hovered_axis = hitTestViewportGizmoLayout(*layout, mouse);
                        }
                        hovered_panel = panel.panel;
                        has_hovered_panel = true;
                        break;
                    }
                }
            }

            for (const auto& panel : panels) {
                if (const auto layout = buildViewportGizmoLayout(
                        panel, gizmo_size, gizmo_margin_x, gizmo_margin_y)) {
                    appendViewportGizmoLayout(params.ui_shape_overlay_triangles,
                                              params,
                                              *layout,
                                              has_hovered_panel && hovered_panel == panel.panel ? hovered_axis : -1);
                    if (dragging && (!has_hovered_panel || hovered_panel == panel.panel)) {
                        appendShapeOverlayCircle(params.ui_shape_overlay_triangles,
                                                 params,
                                                 layout->center,
                                                 layout->size * 0.46f,
                                                 guideColor(theme().overlay.text_dim, 0.20f));
                    }
                }
            }
        }

        void addProjectedOverlayLine(VulkanViewportPassParams& params,
                                     const VulkanGuidePanelTarget& panel,
                                     const RenderSettings& settings,
                                     const glm::vec3& a,
                                     const glm::vec3& b,
                                     const glm::vec4& color,
                                     const float thickness,
                                     const bool depth_aware = false) {
            if (const auto projected = projectSegmentToScreenClipped(panel, settings, a, b)) {
                appendShapeOverlayLine(params.shape_overlay_triangles,
                                       params,
                                       projected->a,
                                       projected->b,
                                       color,
                                       thickness,
                                       depth_aware ? projected->depth_a : 0.0f,
                                       depth_aware ? projected->depth_b : 0.0f);
            }
        }

        [[nodiscard]] std::array<glm::vec3, 8> boxCorners(const glm::vec3& min,
                                                          const glm::vec3& max,
                                                          const glm::mat4& box_to_world) {
            const std::array local{
                glm::vec3(min.x, min.y, min.z),
                glm::vec3(max.x, min.y, min.z),
                glm::vec3(max.x, max.y, min.z),
                glm::vec3(min.x, max.y, min.z),
                glm::vec3(min.x, min.y, max.z),
                glm::vec3(max.x, min.y, max.z),
                glm::vec3(max.x, max.y, max.z),
                glm::vec3(min.x, max.y, max.z),
            };
            std::array<glm::vec3, 8> world{};
            for (size_t i = 0; i < local.size(); ++i) {
                world[i] = glm::vec3(box_to_world * glm::vec4(local[i], 1.0f));
            }
            return world;
        }

        void appendProjectedBox(VulkanViewportPassParams& params,
                                const VulkanGuidePanelTarget& panel,
                                const RenderSettings& settings,
                                const glm::vec3& min,
                                const glm::vec3& max,
                                const glm::mat4& box_to_world,
                                const glm::vec4& color,
                                const float thickness) {
            constexpr std::array<std::pair<int, int>, 12> edges{{
                {0, 1},
                {1, 2},
                {2, 3},
                {3, 0},
                {4, 5},
                {5, 6},
                {6, 7},
                {7, 4},
                {0, 4},
                {1, 5},
                {2, 6},
                {3, 7},
            }};

            const auto corners = boxCorners(min, max, box_to_world);
            for (const auto& [a, b] : edges) {
                addProjectedOverlayLine(params, panel, settings,
                                        corners[static_cast<size_t>(a)],
                                        corners[static_cast<size_t>(b)],
                                        color, thickness);
            }
        }

        [[nodiscard]] glm::vec4 cropGuideColor(const glm::vec3& base_color,
                                               const bool inverse,
                                               const float flash) {
            const glm::vec3 inverse_color(1.0f, 0.2f, 0.2f);
            const glm::vec3 color = glm::mix(inverse ? inverse_color : base_color,
                                             glm::vec3(1.0f),
                                             std::clamp(flash, 0.0f, 1.0f));
            return glm::vec4(color, 0.95f);
        }

        struct ThumbnailPlacement {
            std::uintptr_t texture_id = 0;
            glm::vec2 uv_min{0.0f};
            glm::vec2 uv_max{1.0f};
        };

        class CameraThumbnailCache {
        public:
            static constexpr int kThumbnailSize = 128;

            void beginFrame() {
                std::lock_guard lock(mutex_);
                ++frame_counter_;
            }

            void clear() {
                std::lock_guard lock(mutex_);
                load_queue_.clear();
                ready_queue_.clear();
                entries_.clear();
                pages_.clear();
                cv_.notify_all();
            }

            bool request(const lfs::core::Camera& camera) {
                const int uid = camera.uid();
                if (uid < 0) {
                    return false;
                }

                const std::filesystem::path path = camera.image_path();
                const std::string path_key = lfs::core::path_to_utf8(path);
                if (path_key.empty()) {
                    markFailed(uid, path_key, std::numeric_limits<uint64_t>::max());
                    return false;
                }

                {
                    std::lock_guard lock(mutex_);
                    ensureWorkersLocked();

                    Entry& entry = entries_[uid];
                    if (entry.path_key != path_key) {
                        releaseSlotLocked(entry);
                        entry = Entry{};
                        entry.path_key = path_key;
                    }
                    const bool waiting = entry.state == State::Queued ||
                                         entry.state == State::Loading ||
                                         entry.state == State::UploadReady;
                    if (entry.state == State::Ready || waiting) {
                        return false;
                    }
                    if (entry.state == State::Failed &&
                        entry.retry_frame > frame_counter_) {
                        return false;
                    }
                }

                std::error_code ec;
                if (!std::filesystem::is_regular_file(path, ec)) {
                    const bool path_is_missing =
                        !ec || ec == std::errc::no_such_file_or_directory ||
                        ec == std::errc::not_a_directory;
                    markFailed(uid, path_key, path_is_missing ? kMissingRetryFrames : kDecodeRetryFrames);
                    return false;
                }

                std::lock_guard lock(mutex_);
                Entry& entry = entries_[uid];
                if (entry.path_key != path_key) {
                    releaseSlotLocked(entry);
                    entry = Entry{};
                    entry.path_key = path_key;
                }
                const bool waiting = entry.state == State::Queued ||
                                     entry.state == State::Loading ||
                                     entry.state == State::UploadReady;
                if (entry.state == State::Ready || waiting ||
                    (entry.state == State::Failed && entry.retry_frame > frame_counter_)) {
                    return false;
                }
                entry.state = State::Queued;
                entry.generation += 1;
                load_queue_.push_back(Task{
                    .uid = uid,
                    .path = path,
                    .path_key = path_key,
                    .generation = entry.generation,
                });
                cv_.notify_one();
                return true;
            }

            std::optional<ThumbnailPlacement> placement(const int uid) {
                std::lock_guard lock(mutex_);
                const auto it = entries_.find(uid);
                if (it == entries_.end() ||
                    it->second.state != State::Ready ||
                    it->second.page_index < 0 ||
                    it->second.slot < 0 ||
                    static_cast<size_t>(it->second.page_index) >= pages_.size()) {
                    return std::nullopt;
                }
                const AtlasPage& page = pages_[static_cast<size_t>(it->second.page_index)];
                const std::uintptr_t texture_id = page.texture.textureId();
                if (texture_id == 0) {
                    return std::nullopt;
                }
                auto [uv_min, uv_max] = slotUv(it->second.slot);
                return ThumbnailPlacement{
                    .texture_id = texture_id,
                    .uv_min = uv_min,
                    .uv_max = uv_max,
                };
            }

            bool processReadyUploads(const size_t max_uploads) {
                bool progressed = false;
                for (size_t upload_index = 0; upload_index < max_uploads; ++upload_index) {
                    Decoded decoded;
                    int page_index = -1;
                    int slot = -1;
                    {
                        std::lock_guard lock(mutex_);
                        if (ready_queue_.empty()) {
                            break;
                        }
                        decoded = std::move(ready_queue_.front());
                        ready_queue_.pop_front();
                        cv_.notify_all();
                        const auto it = entries_.find(decoded.uid);
                        if (it == entries_.end() ||
                            it->second.path_key != decoded.path_key ||
                            it->second.generation != decoded.generation ||
                            it->second.state != State::UploadReady) {
                            progressed = true;
                            continue;
                        }
                        Entry& entry = it->second;
                        if (!allocateSlotLocked(entry)) {
                            entry.state = State::Failed;
                            entry.retry_frame = frame_counter_ + kUploadRetryFrames;
                            progressed = true;
                            continue;
                        }
                        page_index = entry.page_index;
                        slot = entry.slot;
                    }

                    const int slot_x = (slot % kAtlasSlotsPerAxis) * kThumbnailSize;
                    const int slot_y = (slot / kAtlasSlotsPerAxis) * kThumbnailSize;
                    const bool uploaded =
                        pages_[static_cast<size_t>(page_index)].texture.uploadRegion(decoded.pixels.data(),
                                                                                     kAtlasTextureSize,
                                                                                     kAtlasTextureSize,
                                                                                     slot_x,
                                                                                     slot_y,
                                                                                     decoded.width,
                                                                                     decoded.height,
                                                                                     decoded.channels);

                    {
                        std::lock_guard lock(mutex_);
                        const auto it = entries_.find(decoded.uid);
                        if (it == entries_.end() ||
                            it->second.path_key != decoded.path_key ||
                            it->second.generation != decoded.generation) {
                            progressed = true;
                            continue;
                        }

                        if (uploaded) {
                            it->second.state = State::Ready;
                            it->second.retry_frame = 0;
                        } else {
                            releaseSlotLocked(it->second);
                            it->second.state = State::Failed;
                            it->second.retry_frame = frame_counter_ + kUploadRetryFrames;
                        }
                    }
                    progressed = true;
                }
                return progressed;
            }

            bool hasPendingWork() const {
                std::lock_guard lock(mutex_);
                if (!load_queue_.empty() || !ready_queue_.empty()) {
                    return true;
                }
                for (const auto& [_, entry] : entries_) {
                    if (entry.state == State::Queued ||
                        entry.state == State::Loading ||
                        entry.state == State::UploadReady) {
                        return true;
                    }
                }
                return false;
            }

            void pruneTo(const std::unordered_set<int>& active_uids) {
                {
                    std::lock_guard lock(mutex_);
                    for (auto it = entries_.begin(); it != entries_.end();) {
                        if (active_uids.contains(it->first)) {
                            ++it;
                            continue;
                        }
                        releaseSlotLocked(it->second);
                        it = entries_.erase(it);
                    }
                    while (!pages_.empty() && pages_.back().live_slots == 0) {
                        pages_.pop_back();
                    }
                    cv_.notify_all();
                }
            }

        private:
            enum class State {
                Empty,
                Queued,
                Loading,
                UploadReady,
                Ready,
                Failed,
            };

            struct Entry {
                std::string path_key;
                State state = State::Empty;
                uint64_t generation = 0;
                uint64_t retry_frame = 0;
                int page_index = -1;
                int slot = -1;
            };

            struct AtlasPage {
                VulkanUiTexture texture;
                std::vector<int> free_slots;
                int next_slot = 0;
                int live_slots = 0;
            };

            struct Task {
                int uid = -1;
                std::filesystem::path path;
                std::string path_key;
                uint64_t generation = 0;
            };

            struct Decoded {
                int uid = -1;
                std::string path_key;
                uint64_t generation = 0;
                std::vector<std::uint8_t> pixels;
                int width = 0;
                int height = 0;
                int channels = 0;
            };

            // Thumbnail decode is OIIO read + CPU downscale; ~17-25 ms per
            // image at typical full-res. With only 2 workers a 50-image
            // dataset took ~600 ms of single-threaded decode dominating
            // viewport_pass_prepare_record. Bumped to 8 — modern desktop
            // CPUs idle most of these cores during dataset load anyway.
            static constexpr size_t kWorkerCount = 8;
            static constexpr uint64_t kDecodeRetryFrames = 180;
            static constexpr uint64_t kMissingRetryFrames = 1800;
            static constexpr uint64_t kUploadRetryFrames = 60;
            static constexpr size_t kMaxPendingUploads = 32;
            static constexpr int kAtlasSlotsPerAxis = 8;
            static constexpr int kAtlasSlotsPerPage = kAtlasSlotsPerAxis * kAtlasSlotsPerAxis;
            static constexpr int kAtlasTextureSize = kThumbnailSize * kAtlasSlotsPerAxis;

            void ensureWorkersLocked() {
                if (workers_started_) {
                    return;
                }
                workers_started_ = true;
                for (size_t i = 0; i < kWorkerCount; ++i) {
                    std::thread([this] {
                        workerLoop();
                    }).detach();
                }
            }

            void markFailed(const int uid, const std::string& path_key, const uint64_t retry_frames) {
                std::lock_guard lock(mutex_);
                Entry& entry = entries_[uid];
                if (entry.path_key != path_key) {
                    releaseSlotLocked(entry);
                    entry = Entry{};
                    entry.path_key = path_key;
                }
                entry.state = State::Failed;
                entry.retry_frame = retry_frames == std::numeric_limits<uint64_t>::max()
                                        ? std::numeric_limits<uint64_t>::max()
                                        : frame_counter_ + retry_frames;
            }

            [[nodiscard]] bool allocateSlotLocked(Entry& entry) {
                if (entry.page_index >= 0 && entry.slot >= 0) {
                    return true;
                }

                for (size_t i = 0; i < pages_.size(); ++i) {
                    AtlasPage& page = pages_[i];
                    if (!page.free_slots.empty()) {
                        entry.page_index = static_cast<int>(i);
                        entry.slot = page.free_slots.back();
                        page.free_slots.pop_back();
                        ++page.live_slots;
                        return true;
                    }
                    if (page.next_slot < kAtlasSlotsPerPage) {
                        entry.page_index = static_cast<int>(i);
                        entry.slot = page.next_slot++;
                        ++page.live_slots;
                        return true;
                    }
                }

                pages_.emplace_back();
                AtlasPage& page = pages_.back();
                entry.page_index = static_cast<int>(pages_.size() - 1);
                entry.slot = page.next_slot++;
                page.live_slots = 1;
                return true;
            }

            void releaseSlotLocked(Entry& entry) {
                if (entry.page_index < 0 ||
                    entry.slot < 0 ||
                    static_cast<size_t>(entry.page_index) >= pages_.size()) {
                    entry.page_index = -1;
                    entry.slot = -1;
                    return;
                }
                AtlasPage& page = pages_[static_cast<size_t>(entry.page_index)];
                page.free_slots.push_back(entry.slot);
                page.live_slots = std::max(0, page.live_slots - 1);
                entry.page_index = -1;
                entry.slot = -1;
            }

            [[nodiscard]] static std::pair<glm::vec2, glm::vec2> slotUv(const int slot) {
                const int slot_x = (slot % kAtlasSlotsPerAxis) * kThumbnailSize;
                const int slot_y = (slot / kAtlasSlotsPerAxis) * kThumbnailSize;
                constexpr float kInset = 0.5f;
                const float atlas = static_cast<float>(kAtlasTextureSize);
                return {
                    glm::vec2((static_cast<float>(slot_x) + kInset) / atlas,
                              (static_cast<float>(slot_y) + kInset) / atlas),
                    glm::vec2((static_cast<float>(slot_x + kThumbnailSize) - kInset) / atlas,
                              (static_cast<float>(slot_y + kThumbnailSize) - kInset) / atlas),
                };
            }

            [[nodiscard]] static std::vector<std::uint8_t> resizeToThumbnail(
                const unsigned char* pixels,
                const int width,
                const int height,
                const int channels) {
                std::vector<std::uint8_t> resized(kThumbnailSize * kThumbnailSize * 3u);
                if (!pixels || width <= 0 || height <= 0 || channels <= 0) {
                    return {};
                }
                for (int y = 0; y < kThumbnailSize; ++y) {
                    const int src_y_unflipped =
                        std::min(static_cast<int>((static_cast<int64_t>(y) * height) / kThumbnailSize),
                                 height - 1);
                    const int src_y = height - 1 - src_y_unflipped;
                    for (int x = 0; x < kThumbnailSize; ++x) {
                        const int src_x =
                            std::min(static_cast<int>((static_cast<int64_t>(x) * width) / kThumbnailSize),
                                     width - 1);
                        const size_t src = (static_cast<size_t>(src_y) * static_cast<size_t>(width) +
                                            static_cast<size_t>(src_x)) *
                                           static_cast<size_t>(channels);
                        const size_t dst = (static_cast<size_t>(y) * kThumbnailSize +
                                            static_cast<size_t>(x)) *
                                           3u;
                        if (channels == 1) {
                            resized[dst + 0] = pixels[src];
                            resized[dst + 1] = pixels[src];
                            resized[dst + 2] = pixels[src];
                        } else {
                            resized[dst + 0] = pixels[src + 0];
                            resized[dst + 1] = pixels[src + 1];
                            resized[dst + 2] = pixels[src + 2];
                        }
                    }
                }
                return resized;
            }

            bool claimTask(Task& task) {
                std::unique_lock lock(mutex_);
                ensureWorkersLocked();
                cv_.wait(lock, [&] {
                    return !load_queue_.empty();
                });
                task = std::move(load_queue_.front());
                load_queue_.pop_front();

                const auto it = entries_.find(task.uid);
                if (it == entries_.end() ||
                    it->second.path_key != task.path_key ||
                    it->second.generation != task.generation ||
                    it->second.state != State::Queued) {
                    return false;
                }
                it->second.state = State::Loading;
                return true;
            }

            void workerLoop() {
                while (true) {
                    Task task;
                    if (!claimTask(task)) {
                        continue;
                    }

                    Decoded decoded;
                    decoded.uid = task.uid;
                    decoded.path_key = task.path_key;
                    decoded.generation = task.generation;

                    try {
                        auto [raw_pixels, width, height, channels] =
                            lfs::core::load_image(task.path, -1, kThumbnailSize);
                        std::unique_ptr<unsigned char, decltype(&lfs::core::free_image)> pixels(
                            raw_pixels, &lfs::core::free_image);
                        if (!pixels || width <= 0 || height <= 0 || channels <= 0) {
                            throw std::runtime_error("decoded empty image");
                        }

                        decoded.pixels = resizeToThumbnail(pixels.get(), width, height, channels);
                        if (decoded.pixels.empty()) {
                            throw std::runtime_error("thumbnail resize failed");
                        }
                        decoded.width = kThumbnailSize;
                        decoded.height = kThumbnailSize;
                        decoded.channels = 3;

                        std::unique_lock lock(mutex_);
                        cv_.wait(lock, [&] {
                            return ready_queue_.size() < kMaxPendingUploads;
                        });
                        const auto it = entries_.find(task.uid);
                        if (it == entries_.end() ||
                            it->second.path_key != task.path_key ||
                            it->second.generation != task.generation) {
                            continue;
                        }
                        it->second.state = State::UploadReady;
                        ready_queue_.push_back(std::move(decoded));
                    } catch (const std::exception& e) {
                        std::lock_guard lock(mutex_);
                        const auto it = entries_.find(task.uid);
                        if (it != entries_.end() &&
                            it->second.path_key == task.path_key &&
                            it->second.generation == task.generation) {
                            it->second.state = State::Failed;
                            it->second.retry_frame = frame_counter_ + kDecodeRetryFrames;
                            LOG_DEBUG("Camera thumbnail load failed for '{}': {}", task.path_key, e.what());
                        }
                    }
                }
            }

            mutable std::mutex mutex_;
            std::condition_variable cv_;
            std::deque<Task> load_queue_;
            std::deque<Decoded> ready_queue_;
            std::unordered_map<int, Entry> entries_;
            std::vector<AtlasPage> pages_;
            uint64_t frame_counter_ = 0;
            bool workers_started_ = false;
        };

        CameraThumbnailCache& cameraThumbnailCache() {
            static auto* const cache = new CameraThumbnailCache();
            return *cache;
        }

        [[nodiscard]] std::optional<glm::mat4> cameraVisualizerTransform(
            const lfs::core::Camera& camera,
            const glm::mat4& scene_transform) {
            auto rotation_tensor = camera.R();
            auto translation_tensor = camera.T();
            if (!rotation_tensor.is_valid() || !translation_tensor.is_valid()) {
                return std::nullopt;
            }
            if (rotation_tensor.device() != lfs::core::Device::CPU) {
                rotation_tensor = rotation_tensor.cpu();
            }
            if (translation_tensor.device() != lfs::core::Device::CPU) {
                translation_tensor = translation_tensor.cpu();
            }
            if (rotation_tensor.dtype() != lfs::core::DataType::Float32 ||
                translation_tensor.dtype() != lfs::core::DataType::Float32 ||
                rotation_tensor.numel() < 9 || translation_tensor.numel() < 3) {
                return std::nullopt;
            }

            const float* const rotation = rotation_tensor.ptr<float>();
            const float* const translation = translation_tensor.ptr<float>();
            if (!rotation || !translation) {
                return std::nullopt;
            }

            glm::mat4 world_to_camera(1.0f);
            for (int row = 0; row < 3; ++row) {
                for (int col = 0; col < 3; ++col) {
                    world_to_camera[col][row] = rotation[row * 3 + col];
                }
                world_to_camera[3][row] = translation[row];
            }

            return scene_transform * glm::inverse(world_to_camera) *
                   lfs::rendering::DATA_TO_VISUALIZER_CAMERA_AXES_4;
        }

        [[nodiscard]] std::vector<glm::mat4> resolveCameraSceneTransforms(
            const SceneManager& scene_manager,
            const SceneRenderState* scene_state,
            const size_t camera_count) {
            if (scene_state && scene_state->camera_scene_transforms.size() == camera_count) {
                return scene_state->camera_scene_transforms;
            }

            auto transforms = scene_manager.getScene().getVisibleCameraSceneTransforms();
            if (transforms.size() != camera_count) {
                transforms.assign(camera_count, glm::mat4(1.0f));
            }
            for (auto& transform : transforms) {
                transform = lfs::rendering::dataWorldTransformToVisualizerWorld(transform);
            }
            return transforms;
        }

        [[nodiscard]] float cameraFrustumVisibilityAlpha(const glm::vec3& camera_position,
                                                         const glm::vec3& view_position,
                                                         const float scale,
                                                         const bool disabled) {
            constexpr float kFadeStartMultiplier = 5.0f;
            constexpr float kFadeEndMultiplier = 0.2f;
            constexpr float kMinVisibleMultiplier = 0.1f;
            constexpr float kMinVisibleAlpha = 0.05f;

            const float safe_scale = std::max(scale, 0.0f);
            const float fade_start = kFadeStartMultiplier * safe_scale;
            const float fade_end = kFadeEndMultiplier * safe_scale;
            const float min_visible = kMinVisibleMultiplier * safe_scale;
            const float distance = glm::length(camera_position - view_position);

            float alpha = 1.0f;
            if (distance < min_visible) {
                alpha = 0.0f;
            } else if (distance < fade_end) {
                alpha = kMinVisibleAlpha;
            } else if (distance < fade_start && fade_start > fade_end) {
                const float t = (distance - fade_end) / (fade_start - fade_end);
                alpha = kMinVisibleAlpha + (1.0f - kMinVisibleAlpha) * (t * t * (3.0f - 2.0f * t));
            }
            if (disabled) {
                alpha *= 0.4f;
            }
            return alpha;
        }

        [[nodiscard]] glm::vec4 cameraFrustumColor(const lfs::core::Camera& camera,
                                                   const size_t camera_index,
                                                   const RenderSettings& settings,
                                                   const std::span<const glm::vec3> per_camera_colors,
                                                   const float alpha,
                                                   const bool focused,
                                                   const bool disabled,
                                                   const bool emphasized) {
            const bool has_override = camera_index < per_camera_colors.size();
            const bool is_validation = camera.image_name().find("test") != std::string::npos;
            glm::vec3 color = is_validation ? settings.eval_camera_color : settings.train_camera_color;
            if (has_override) {
                const glm::vec3 override_color = per_camera_colors[camera_index];
                if (std::isfinite(override_color.x) &&
                    std::isfinite(override_color.y) &&
                    std::isfinite(override_color.z)) {
                    color = override_color;
                }
            }

            float final_alpha = alpha;
            if (emphasized) {
                color = glm::vec3(1.0f, 0.55f, 0.0f);
                final_alpha = std::min(1.0f, final_alpha + 0.4f);
            }
            if (focused) {
                color = is_validation ? glm::vec3(0.9f, 0.75f, 0.0f)
                                      : glm::vec3(1.0f, 0.55f, 0.0f);
                final_alpha = std::min(1.0f, final_alpha + 0.3f);
            }
            if (disabled) {
                color = glm::mix(color, glm::vec3(0.5f), 0.5f);
                final_alpha *= 0.5f;
            }
            return glm::vec4(color, std::clamp(final_alpha, 0.0f, 1.0f));
        }

        [[nodiscard]] std::optional<glm::mat4> cameraFrustumModelMatrix(
            const lfs::core::Camera& camera,
            const glm::mat4& visualizer_camera_to_world,
            const float scale) {
            const int image_width = camera.image_width() > 0 ? camera.image_width() : camera.camera_width();
            const int image_height = camera.image_height() > 0 ? camera.image_height() : camera.camera_height();
            if (image_width <= 0 || image_height <= 0 || scale <= 0.0f) {
                return std::nullopt;
            }

            constexpr float kEquirectangularDisplayFov = 1.0472f;
            const bool equirectangular =
                camera.camera_model_type() == lfs::core::CameraModelType::EQUIRECTANGULAR;
            if (!equirectangular && camera.focal_y() <= 0.0f) {
                return std::nullopt;
            }

            const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);
            const float fov_y = equirectangular
                                    ? kEquirectangularDisplayFov
                                    : lfs::core::focal2fov(camera.focal_y(), image_height);
            const float half_height = std::tan(fov_y * 0.5f);
            const float half_width = half_height * aspect;

            const glm::mat4 fov_scale = glm::scale(
                glm::mat4(1.0f),
                glm::vec3(half_width * 2.0f * scale, half_height * 2.0f * scale, scale));
            return visualizer_camera_to_world * fov_scale;
        }

        void appendEquirectangularCameraFrustum(VulkanViewportPassParams& params,
                                                const VulkanGuidePanelTarget& panel,
                                                const RenderSettings& settings,
                                                const glm::mat4& model,
                                                const glm::vec4& color) {
            constexpr int kLatSegments = 16;
            constexpr int kLonSegments = 24;
            constexpr float kRadius = 0.5f;

            const auto point = [&](const int lat, const int lon) {
                const float theta = static_cast<float>(lat) /
                                    static_cast<float>(kLatSegments) * glm::pi<float>();
                const float phi = static_cast<float>(lon) /
                                  static_cast<float>(kLonSegments) * 2.0f * glm::pi<float>();
                const float sin_theta = std::sin(theta);
                const float cos_theta = std::cos(theta);
                const glm::vec3 local(kRadius * sin_theta * std::sin(phi),
                                      kRadius * cos_theta,
                                      -kRadius * sin_theta * std::cos(phi));
                return glm::vec3(model * glm::vec4(local, 1.0f));
            };

            for (int lat = 0; lat <= kLatSegments; lat += 4) {
                glm::vec3 previous = point(lat, 0);
                for (int lon = 1; lon <= kLonSegments; ++lon) {
                    const glm::vec3 current = point(lat, lon);
                    addProjectedOverlayLine(params, panel, settings,
                                            previous, current, color, 1.5f, true);
                    previous = current;
                }
            }
            for (int lon = 0; lon < kLonSegments; lon += 4) {
                glm::vec3 previous = point(0, lon);
                for (int lat = 1; lat <= kLatSegments; ++lat) {
                    const glm::vec3 current = point(lat, lon);
                    addProjectedOverlayLine(params, panel, settings,
                                            previous, current, color, 1.5f, true);
                    previous = current;
                }
            }

            const glm::vec3 apex = glm::vec3(model * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
            for (int lon = 0; lon < kLonSegments; lon += kLonSegments / 4) {
                addProjectedOverlayLine(params,
                                        panel,
                                        settings,
                                        apex,
                                        point(kLatSegments / 2, lon),
                                        color,
                                        1.5f,
                                        true);
            }
        }

        void appendCameraFrustumOverlays(VulkanViewportPassParams& params,
                                         const VulkanGuidePanelTarget& panel,
                                         const RenderSettings& settings,
                                         RenderingManager& rendering_manager,
                                         const SceneManager& scene_manager,
                                         const SceneRenderState* scene_state) {
            if (!settings.show_camera_frustums || settings.camera_frustum_scale <= 0.0f) {
                return;
            }

            const auto cameras = scene_manager.getScene().getVisibleCameras();
            if (cameras.empty()) {
                cameraThumbnailCache().clear();
                return;
            }

            const std::vector<glm::mat4> scene_transforms =
                resolveCameraSceneTransforms(scene_manager, scene_state, cameras.size());
            const auto disabled_uids = scene_manager.getScene().getTrainingDisabledCameraUids();
            const int hovered_camera_id = rendering_manager.getHoveredCameraId();
            auto& thumbnail_cache = cameraThumbnailCache();
            thumbnail_cache.beginFrame();

            std::unordered_set<int> emphasized_uids;
            std::unordered_set<int> active_camera_uids;
            active_camera_uids.reserve(cameras.size());
            for (const auto& name : scene_manager.getSelectedNodeNames()) {
                const auto* node = scene_manager.getScene().getNode(name);
                if (node && node->type == lfs::core::NodeType::CAMERA && node->camera_uid >= 0) {
                    emphasized_uids.insert(node->camera_uid);
                }
            }
            for (const auto& camera : cameras) {
                if (camera && camera->uid() >= 0) {
                    active_camera_uids.insert(camera->uid());
                }
            }
            thumbnail_cache.pruneTo(active_camera_uids);
            thumbnail_cache.processReadyUploads(4);

            std::vector<glm::vec3> per_camera_colors;
            if (const auto* trainer_manager = scene_manager.getTrainerManager()) {
                if (const auto* trainer = trainer_manager->getTrainer()) {
                    std::vector<std::array<float, 3>> loss_colors;
                    if (trainer->fillCameraLossColors(cameras, loss_colors) &&
                        loss_colors.size() == cameras.size()) {
                        per_camera_colors.reserve(loss_colors.size());
                        for (const auto& color : loss_colors) {
                            per_camera_colors.emplace_back(color[0], color[1], color[2]);
                        }
                    }
                }
            }

            constexpr float kMinRenderAlpha = 0.01f;
            const glm::mat3 panel_rotation = panel.viewport->getRotationMatrix();
            const glm::vec3 panel_translation = panel.viewport->getTranslation();
            const glm::mat3 world_to_panel_rotation = glm::transpose(panel_rotation);
            const glm::vec3 view_position = panel_translation;
            const float panel_render_width = static_cast<float>(std::max(panel.render_size.x, 1));
            const float panel_render_height = static_cast<float>(std::max(panel.render_size.y, 1));
            const float panel_cx = panel_render_width * 0.5f;
            const float panel_cy = panel_render_height * 0.5f;
            const auto [panel_fx, panel_fy] =
                lfs::rendering::computePixelFocalLengths(panel.render_size, settings.focal_length_mm);
            const auto project_panel_point = [&](const glm::vec3& world) -> std::optional<glm::vec2> {
                const glm::vec3 view = world_to_panel_rotation * (world - panel_translation);
                if (settings.equirectangular) {
                    const float len = glm::length(view);
                    if (!std::isfinite(len) || len <= 1e-6f) {
                        return std::nullopt;
                    }
                    const glm::vec3 dir = view / len;
                    const float ndc_x = std::atan2(dir.x, -dir.z) / glm::pi<float>();
                    const float ndc_y = -std::asin(std::clamp(dir.y, -1.0f, 1.0f)) /
                                        (glm::pi<float>() * 0.5f);
                    if (!std::isfinite(ndc_x) || !std::isfinite(ndc_y)) {
                        return std::nullopt;
                    }
                    return panel.pos + glm::vec2((ndc_x * 0.5f + 0.5f) * panel.size.x,
                                                 (ndc_y * 0.5f + 0.5f) * panel.size.y);
                }

                constexpr float kMinViewZ = -1e-4f;
                if (view.z >= kMinViewZ) {
                    return std::nullopt;
                }
                if (settings.orthographic) {
                    if (!std::isfinite(settings.ortho_scale) || settings.ortho_scale <= 0.0f) {
                        return std::nullopt;
                    }
                    return renderToPanelScreen(panel, glm::vec2(panel_cx + view.x * settings.ortho_scale,
                                                                panel_cy - view.y * settings.ortho_scale));
                }

                const float depth = -view.z;
                if (depth <= 0.0f) {
                    return std::nullopt;
                }
                return renderToPanelScreen(panel, glm::vec2(panel_cx + view.x * panel_fx / depth,
                                                            panel_cy - view.y * panel_fy / depth));
            };
            size_t background_thumbnail_requests = 0;
            constexpr size_t kBackgroundThumbnailRequestsPerFrame = 16;
            const std::uint32_t frustum_first_instance =
                static_cast<std::uint32_t>(params.frustum_instances.size());
            params.frustum_instances.reserve(params.frustum_instances.size() + cameras.size());
            for (size_t i = 0; i < cameras.size(); ++i) {
                const auto& camera = cameras[i];
                if (!camera) {
                    continue;
                }

                const auto visualizer_camera_to_world =
                    cameraVisualizerTransform(*camera, scene_transforms[i]);
                if (!visualizer_camera_to_world) {
                    continue;
                }

                const bool disabled = disabled_uids.count(camera->uid()) > 0;
                const float alpha = cameraFrustumVisibilityAlpha(
                    glm::vec3((*visualizer_camera_to_world)[3]),
                    view_position,
                    settings.camera_frustum_scale,
                    disabled);
                if (alpha <= kMinRenderAlpha) {
                    continue;
                }

                const glm::vec4 color = cameraFrustumColor(*camera,
                                                           i,
                                                           settings,
                                                           per_camera_colors,
                                                           alpha,
                                                           camera->uid() == hovered_camera_id,
                                                           disabled,
                                                           emphasized_uids.count(camera->uid()) > 0);
                if (color.a <= kMinRenderAlpha) {
                    continue;
                }

                const auto model = cameraFrustumModelMatrix(*camera,
                                                            *visualizer_camera_to_world,
                                                            settings.camera_frustum_scale);
                if (!model) {
                    continue;
                }

                if (camera->camera_model_type() == lfs::core::CameraModelType::EQUIRECTANGULAR) {
                    appendEquirectangularCameraFrustum(params, panel, settings, *model, color);
                } else {
                    constexpr std::array image_corners{
                        glm::vec3(-0.5f, -0.5f, -1.0f),
                        glm::vec3(0.5f, -0.5f, -1.0f),
                        glm::vec3(0.5f, 0.5f, -1.0f),
                        glm::vec3(-0.5f, 0.5f, -1.0f),
                    };
                    std::array<glm::vec2, image_corners.size()> screen_points{};
                    std::array<float, image_corners.size()> corner_depths{};
                    bool quad_visible = true;
                    for (size_t corner = 0; corner < image_corners.size(); ++corner) {
                        const glm::vec3 world_point =
                            glm::vec3((*model) * glm::vec4(image_corners[corner], 1.0f));
                        const auto projected = project_panel_point(world_point);
                        if (!projected) {
                            quad_visible = false;
                            break;
                        }
                        screen_points[corner] = *projected;
                        const glm::vec3 view = world_to_panel_rotation * (world_point - panel_translation);
                        corner_depths[corner] = settings.equirectangular ? glm::length(view) : -view.z;
                    }
                    quad_visible = quad_visible && projectedQuadVisible(screen_points, panel);
                    if (quad_visible) {
                        thumbnail_cache.request(*camera);
                    } else if (background_thumbnail_requests < kBackgroundThumbnailRequestsPerFrame &&
                               thumbnail_cache.request(*camera)) {
                        ++background_thumbnail_requests;
                    }

                    const auto placement = thumbnail_cache.placement(camera->uid());
                    if (placement && quad_visible) {
                        const float opacity = std::clamp(color.a * 0.8f, 0.0f, 0.8f);
                        const float disabled_mix = disabled ? 0.5f : 0.0f;
                        const float emphasis_mix = emphasized_uids.count(camera->uid()) > 0 ? 0.18f : 0.0f;
                        appendTexturedOverlayQuad(params,
                                                  placement->texture_id,
                                                  screen_points,
                                                  placement->uv_min,
                                                  placement->uv_max,
                                                  {color.r, color.g, color.b, opacity},
                                                  {emphasis_mix, disabled_mix, 0.0f, 0.0f},
                                                  corner_depths);
                    }
                    params.frustum_instances.push_back(VulkanViewportFrustumInstance{
                        .model = *model,
                        .color = color,
                    });
                }
            }

            const std::uint32_t frustum_instance_count =
                static_cast<std::uint32_t>(params.frustum_instances.size()) - frustum_first_instance;
            if (frustum_instance_count > 0) {
                const glm::mat4 frustum_view =
                    lfs::rendering::makeViewMatrix(panel_rotation, panel_translation);
                params.frustum_batches.push_back(VulkanViewportFrustumBatch{
                    .view = frustum_view,
                    .viewport_pos = panel.pos,
                    .viewport_size = panel.size,
                    .render_size = glm::vec2(panel.render_size),
                    .focal_x = settings.orthographic ? settings.ortho_scale : panel_fx,
                    .focal_y = settings.orthographic ? settings.ortho_scale : panel_fy,
                    .orthographic = settings.orthographic,
                    .equirectangular = settings.equirectangular,
                    .first_instance = frustum_first_instance,
                    .instance_count = frustum_instance_count,
                });
            }

            if (thumbnail_cache.hasPendingWork()) {
                rendering_manager.markDirty(DirtyFlag::OVERLAY);
            }
        }

        void appendProjectedEllipsoid(VulkanViewportPassParams& params,
                                      const VulkanGuidePanelTarget& panel,
                                      const RenderSettings& settings,
                                      const glm::vec3& radii,
                                      const glm::mat4& ellipsoid_to_world,
                                      const glm::vec4& color,
                                      const float thickness) {
            constexpr int lat_segments = 24;
            constexpr int lon_segments = 32;

            const auto point = [&](const int lat, const int lon) {
                const float theta = static_cast<float>(lat) /
                                    static_cast<float>(lat_segments) * glm::pi<float>();
                const float phi = static_cast<float>(lon) /
                                  static_cast<float>(lon_segments) * 2.0f * glm::pi<float>();
                const float sin_theta = std::sin(theta);
                const glm::vec3 local(
                    sin_theta * std::cos(phi) * radii.x,
                    std::cos(theta) * radii.y,
                    sin_theta * std::sin(phi) * radii.z);
                return glm::vec3(ellipsoid_to_world * glm::vec4(local, 1.0f));
            };

            for (int lat = 0; lat < lat_segments; lat += 2) {
                glm::vec3 previous = point(lat, 0);
                for (int lon = 1; lon <= lon_segments; ++lon) {
                    const glm::vec3 current = point(lat, lon % lon_segments);
                    addProjectedOverlayLine(params, panel, settings, previous, current, color, thickness);
                    previous = current;
                }
            }
            for (int lon = 0; lon < lon_segments; lon += 2) {
                glm::vec3 previous = point(0, lon);
                for (int lat = 1; lat <= lat_segments; ++lat) {
                    const glm::vec3 current = point(lat, lon);
                    addProjectedOverlayLine(params, panel, settings, previous, current, color, thickness);
                    previous = current;
                }
            }
        }

        void appendCropAndFilterOverlays(VulkanViewportPassParams& params,
                                         const VulkanGuidePanelTarget& panel,
                                         const RenderSettings& settings,
                                         const SceneRenderState* scene_state,
                                         const SceneManager* scene_manager,
                                         const GizmoState& gizmo) {
            if (settings.depth_filter_enabled) {
                const glm::mat4 filter_to_world = settings.depth_filter_transform.toMat4();
                appendProjectedBox(params, panel, settings,
                                   settings.depth_filter_min,
                                   settings.depth_filter_max,
                                   filter_to_world,
                                   glm::vec4(0.0f, 0.0f, 0.0f, 0.85f),
                                   9.0f);
                appendProjectedBox(params, panel, settings,
                                   settings.depth_filter_min,
                                   settings.depth_filter_max,
                                   filter_to_world,
                                   glm::vec4(1.0f, 1.0f, 1.0f, 0.90f),
                                   6.0f);
                appendProjectedBox(params, panel, settings,
                                   settings.depth_filter_min,
                                   settings.depth_filter_max,
                                   filter_to_world,
                                   glm::vec4(1.0f, 1.0f, 1.0f, 1.0f),
                                   4.5f);
            }

            if (gizmo.cropbox_active) {
                appendProjectedBox(params, panel, settings,
                                   gizmo.cropbox_min,
                                   gizmo.cropbox_max,
                                   gizmo.cropbox_transform,
                                   cropGuideColor(glm::vec3(1.0f, 1.0f, 0.0f), false, 0.0f),
                                   2.0f);
            }

            if (gizmo.ellipsoid_active) {
                appendProjectedEllipsoid(params, panel, settings,
                                         gizmo.ellipsoid_radii,
                                         gizmo.ellipsoid_transform,
                                         cropGuideColor(glm::vec3(0.5f, 0.85f, 1.0f), false, 0.0f),
                                         2.0f);
            }

            if (!scene_state || !scene_manager) {
                return;
            }

            if (settings.show_crop_box) {
                const core::NodeId selected_id = scene_manager->getSelectedNodeCropBoxId();
                for (const auto& cb : scene_state->cropboxes) {
                    if (!cb.data) {
                        continue;
                    }
                    const bool selected = cb.node_id == selected_id;
                    const bool use_pending = selected && gizmo.cropbox_active;
                    const glm::vec3 box_min = use_pending ? gizmo.cropbox_min : cb.data->min;
                    const glm::vec3 box_max = use_pending ? gizmo.cropbox_max : cb.data->max;
                    const glm::mat4 world_transform = use_pending ? gizmo.cropbox_transform : cb.world_transform;
                    const float flash = selected ? std::clamp(cb.data->flash_intensity, 0.0f, 1.0f) : 0.0f;
                    appendProjectedBox(params, panel, settings,
                                       box_min,
                                       box_max,
                                       world_transform,
                                       cropGuideColor(cb.data->color, cb.data->inverse, flash),
                                       cb.data->line_width + flash * 4.0f);
                }
            }

            if (settings.show_ellipsoid) {
                const core::NodeId selected_id = scene_manager->getSelectedNodeEllipsoidId();
                for (const auto& el : scene_state->ellipsoids) {
                    if (!el.data) {
                        continue;
                    }
                    const bool selected = el.node_id == selected_id;
                    const bool use_pending = selected && gizmo.ellipsoid_active;
                    const glm::vec3 radii = use_pending ? gizmo.ellipsoid_radii : el.data->radii;
                    const glm::mat4 world_transform = use_pending ? gizmo.ellipsoid_transform : el.world_transform;
                    const float flash = selected ? std::clamp(el.data->flash_intensity, 0.0f, 1.0f) : 0.0f;
                    appendProjectedEllipsoid(params, panel, settings,
                                             radii,
                                             world_transform,
                                             cropGuideColor(el.data->color, el.data->inverse, flash),
                                             el.data->line_width + flash * 4.0f);
                }
            }
        }

        void appendPivotShaderOverlay(VulkanViewportPassParams& params,
                                      const VulkanGuidePanelTarget& panel,
                                      const RenderSettings& settings,
                                      const glm::vec3& pivot_world,
                                      const float opacity) {
            constexpr float kPivotSizePx = 50.0f;
            constexpr glm::vec3 kPivotColor{0.26f, 0.59f, 0.98f};

            const glm::mat4 view =
                lfs::rendering::makeViewMatrix(panel.viewport->getRotationMatrix(), panel.viewport->getTranslation());
            const glm::mat4 projection = lfs::rendering::createProjectionMatrixFromFocal(
                panel.render_size,
                settings.focal_length_mm,
                settings.orthographic,
                settings.ortho_scale,
                lfs::rendering::DEFAULT_NEAR_PLANE,
                settings.depth_clip_enabled ? settings.depth_clip_far : lfs::rendering::DEFAULT_FAR_PLANE);
            const glm::vec4 clip = projection * view * glm::vec4(pivot_world, 1.0f);
            if (!std::isfinite(clip.x) || !std::isfinite(clip.y) || !std::isfinite(clip.w) ||
                std::abs(clip.w) <= 1e-6f) {
                return;
            }

            const glm::vec2 gl_ndc = glm::vec2(clip) / clip.w;
            const glm::vec2 screen = panel.pos + glm::vec2(
                                                     (gl_ndc.x * 0.5f + 0.5f) * panel.size.x,
                                                     (1.0f - (gl_ndc.y * 0.5f + 0.5f)) * panel.size.y);
            const glm::vec2 framebuffer_scale(
                params.framebuffer_scale.x > 0.0f ? params.framebuffer_scale.x : 1.0f,
                params.framebuffer_scale.y > 0.0f ? params.framebuffer_scale.y : 1.0f);
            const glm::vec2 framebuffer_size = glm::max(params.viewport_size * framebuffer_scale,
                                                        glm::vec2(1.0f));
            params.pivot_overlays.push_back({
                .center_ndc = screenToViewportNdc(screen, params.viewport_pos, params.viewport_size),
                .size_ndc = (glm::vec2(kPivotSizePx) / framebuffer_size) * 2.0f,
                .color = kPivotColor,
                .opacity = std::clamp(opacity, 0.0f, 1.0f),
            });
        }

        void appendVulkanSceneGuideOverlays(VulkanViewportPassParams& params,
                                            const VisualizerImpl& viewer,
                                            const ViewportLayout& viewport_layout,
                                            const RenderSettings& settings,
                                            RenderingManager& rendering_manager,
                                            SceneManager* scene_manager,
                                            const SceneRenderState* scene_state,
                                            const GizmoState& gizmo) {
            if (params.viewport_size.x <= 0.0f || params.viewport_size.y <= 0.0f) {
                return;
            }

            const auto panels = collectVulkanGuidePanels(viewer, viewport_layout, rendering_manager);
            if (panels.empty()) {
                return;
            }

            constexpr std::array axis_colors{
                glm::vec4(1.0f, 0.0f, 0.0f, 1.0f),
                glm::vec4(0.0f, 1.0f, 0.0f, 1.0f),
                glm::vec4(0.0f, 0.0f, 1.0f, 1.0f),
            };
            constexpr std::array axes{
                glm::vec3(1.0f, 0.0f, 0.0f),
                glm::vec3(0.0f, 1.0f, 0.0f),
                glm::vec3(0.0f, 0.0f, 1.0f),
            };

            for (const auto& panel : panels) {
                if (!panel.valid()) {
                    continue;
                }

                appendCropAndFilterOverlays(params, panel, settings, scene_state, scene_manager, gizmo);
                if (scene_manager) {
                    appendCameraFrustumOverlays(params,
                                                panel,
                                                settings,
                                                rendering_manager,
                                                *scene_manager,
                                                scene_state);
                }

                if (settings.show_coord_axes) {
                    for (size_t axis = 0; axis < axes.size(); ++axis) {
                        if (settings.axes_visibility[axis]) {
                            addProjectedOverlayLine(params, panel, settings,
                                                    glm::vec3(0.0f),
                                                    axes[axis] * settings.axes_size,
                                                    axis_colors[axis], 3.0f);
                        }
                    }
                }

                constexpr float kPivotDurationSec = 0.5f;
                const float time_since_set = panel.viewport->camera.getSecondsSincePivotSet();
                const bool pivot_animation_active = time_since_set < kPivotDurationSec;
                if (pivot_animation_active) {
                    const auto remaining_ms = static_cast<int>(
                        (kPivotDurationSec - time_since_set) * 1000.0f);
                    rendering_manager.setPivotAnimationEndTime(
                        std::chrono::steady_clock::now() + std::chrono::milliseconds(remaining_ms));
                }
                if (settings.show_pivot || pivot_animation_active) {
                    const float opacity = settings.show_pivot
                                              ? 1.0f
                                              : 1.0f - std::clamp(time_since_set / kPivotDurationSec, 0.0f, 1.0f);
                    appendPivotShaderOverlay(params, panel, settings, panel.viewport->camera.getPivot(), opacity);
                }
            }
        }

        enum class DevResourceKind {
            None,
            Rml,
            Locale
        };

        [[nodiscard]] DevResourceKind devResourceKindForPath(const std::filesystem::path& path) {
            std::string ext = lfs::core::path_to_utf8(path.extension());
            std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char ch) {
                return static_cast<char>(std::tolower(ch));
            });

            if (ext == ".json")
                return DevResourceKind::Locale;
            if (ext == ".rml" || ext == ".rcss")
                return DevResourceKind::Rml;
            return DevResourceKind::None;
        }

        std::string makeRmlTabDomId(const std::string& id) {
            std::string result = "rp-tab-";
            result.reserve(result.size() + id.size());
            for (const char ch : id) {
                const bool keep = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
                                  (ch >= '0' && ch <= '9') || ch == '-' || ch == '_';
                result.push_back(keep ? ch : '-');
            }
            return result;
        }

        PanelInputState maskInputForBlockedUi(PanelInputState input) {
            input.mouse_x = -1.0e9f;
            input.mouse_y = -1.0e9f;
            for (auto& value : input.mouse_down)
                value = false;
            for (auto& value : input.mouse_clicked)
                value = false;
            for (auto& value : input.mouse_released)
                value = false;
            input.mouse_wheel = 0.0f;
            input.key_ctrl = false;
            input.key_shift = false;
            input.key_alt = false;
            input.key_super = false;
            input.viewport_keyboard_focus = false;
            input.keys_pressed.clear();
            input.keys_repeated.clear();
            input.keys_released.clear();
            input.text_codepoints.clear();
            input.text_inputs.clear();
            input.text_editing.clear();
            input.text_editing_start = -1;
            input.text_editing_length = -1;
            input.has_text_editing = false;
            return input;
        }

        [[nodiscard]] bool hasPointerActivity(const FrameInputBuffer& input) {
            return input.mouse_moved || input.mouse_wheel != 0.0f ||
                   input.mouse_down[0] || input.mouse_down[1] || input.mouse_down[2] ||
                   input.mouse_clicked[0] || input.mouse_clicked[1] || input.mouse_clicked[2] ||
                   input.mouse_released[0] || input.mouse_released[1] || input.mouse_released[2];
        }

        [[nodiscard]] bool hasKeyboardActivity(const FrameInputBuffer& input) {
            return !input.keys_pressed.empty() || !input.keys_repeated.empty() ||
                   !input.keys_released.empty() || !input.text_codepoints.empty() ||
                   !input.text_inputs.empty() || input.has_text_editing;
        }

        [[nodiscard]] bool hasMouseButtonDown(const FrameInputBuffer& input) {
            return input.mouse_down[0] || input.mouse_down[1] || input.mouse_down[2];
        }

        [[nodiscard]] bool hasMouseButtonClicked(const FrameInputBuffer& input) {
            return input.mouse_clicked[0] || input.mouse_clicked[1] || input.mouse_clicked[2];
        }

        [[nodiscard]] bool pointInRect(const float x, const float y,
                                       const glm::vec2 pos, const glm::vec2 size,
                                       const float extra = 0.0f) {
            return x >= pos.x - extra &&
                   x < pos.x + size.x + extra &&
                   y >= pos.y - extra &&
                   y < pos.y + size.y + extra;
        }

        void applyFrameInputCapture(RmlRightPanel* right_panel = nullptr) {
            const bool panel_hosts_want_keyboard = RmlPanelHost::consumeFrameWantsKeyboard();
            const bool panel_hosts_want_text_input = RmlPanelHost::consumeFrameWantsTextInput();
            if ((panel_hosts_want_keyboard || panel_hosts_want_text_input) && right_panel)
                right_panel->blurFocus();

            auto& focus = guiFocusState();
            if (panel_hosts_want_keyboard)
                focus.want_capture_keyboard = true;
            if (panel_hosts_want_text_input)
                focus.want_text_input = true;
        }

        void syncWindowTextInput(SDL_Window* window) {
            if (!window)
                return;

            const bool wants_text_input = guiFocusState().want_text_input;
            const bool text_input_active = SDL_TextInputActive(window);
            if (wants_text_input == text_input_active)
                return;

            if (wants_text_input)
                SDL_StartTextInput(window);
            else
                SDL_StopTextInput(window);
        }

        SDL_Cursor* systemCursorForImGuiCursor(const ImGuiMouseCursor cursor) {
            switch (cursor) {
            case ImGuiMouseCursor_TextInput: {
                static SDL_Cursor* const value = SDL_CreateSystemCursor(SDL_SYSTEM_CURSOR_TEXT);
                return value;
            }
            case ImGuiMouseCursor_Hand: {
                static SDL_Cursor* const value = SDL_CreateSystemCursor(SDL_SYSTEM_CURSOR_POINTER);
                return value;
            }
            case ImGuiMouseCursor_ResizeEW: {
                static SDL_Cursor* const value = SDL_CreateSystemCursor(SDL_SYSTEM_CURSOR_EW_RESIZE);
                return value;
            }
            case ImGuiMouseCursor_ResizeNS: {
                static SDL_Cursor* const value = SDL_CreateSystemCursor(SDL_SYSTEM_CURSOR_NS_RESIZE);
                return value;
            }
            case ImGuiMouseCursor_ResizeNWSE: {
                static SDL_Cursor* const value = SDL_CreateSystemCursor(SDL_SYSTEM_CURSOR_NWSE_RESIZE);
                return value;
            }
            case ImGuiMouseCursor_ResizeNESW: {
                static SDL_Cursor* const value = SDL_CreateSystemCursor(SDL_SYSTEM_CURSOR_NESW_RESIZE);
                return value;
            }
            case ImGuiMouseCursor_ResizeAll: {
                static SDL_Cursor* const value = SDL_CreateSystemCursor(SDL_SYSTEM_CURSOR_MOVE);
                return value;
            }
            case ImGuiMouseCursor_NotAllowed: {
                static SDL_Cursor* const value = SDL_CreateSystemCursor(SDL_SYSTEM_CURSOR_NOT_ALLOWED);
                return value;
            }
            default:
                return nullptr;
            }
        }

        SDL_Cursor* loadColorCursorFromAsset(const std::string& asset_name, int hot_x, int hot_y) {
            try {
                const auto path = lfs::vis::getAssetPath(asset_name);
                const std::string path_utf8 = lfs::core::path_to_utf8(path);
                std::unique_ptr<OIIO::ImageInput> in(OIIO::ImageInput::open(path_utf8));
                if (!in)
                    return nullptr;

                const OIIO::ImageSpec& spec = in->spec();
                const int width = spec.width;
                const int height = spec.height;
                const int channels = spec.nchannels;
                if (width <= 0 || height <= 0 || channels <= 0) {
                    in->close();
                    return nullptr;
                }

                const int read_channels = std::clamp(channels, 1, 4);
                std::vector<unsigned char> source_pixels(static_cast<size_t>(width) * height * read_channels);
                if (!in->read_image(0, 0, 0, read_channels, OIIO::TypeDesc::UINT8, source_pixels.data())) {
                    in->close();
                    return nullptr;
                }
                in->close();

                std::vector<unsigned char> rgba_pixels(static_cast<size_t>(width) * height * 4, 0);
                for (int i = 0; i < width * height; ++i) {
                    const size_t src = static_cast<size_t>(i) * read_channels;
                    const size_t dst = static_cast<size_t>(i) * 4;
                    switch (read_channels) {
                    case 1:
                        rgba_pixels[dst + 0] = source_pixels[src + 0];
                        rgba_pixels[dst + 1] = source_pixels[src + 0];
                        rgba_pixels[dst + 2] = source_pixels[src + 0];
                        rgba_pixels[dst + 3] = 255;
                        break;
                    case 2:
                        rgba_pixels[dst + 0] = source_pixels[src + 0];
                        rgba_pixels[dst + 1] = source_pixels[src + 0];
                        rgba_pixels[dst + 2] = source_pixels[src + 0];
                        rgba_pixels[dst + 3] = source_pixels[src + 1];
                        break;
                    case 3:
                        rgba_pixels[dst + 0] = source_pixels[src + 0];
                        rgba_pixels[dst + 1] = source_pixels[src + 1];
                        rgba_pixels[dst + 2] = source_pixels[src + 2];
                        rgba_pixels[dst + 3] = 255;
                        break;
                    default:
                        rgba_pixels[dst + 0] = source_pixels[src + 0];
                        rgba_pixels[dst + 1] = source_pixels[src + 1];
                        rgba_pixels[dst + 2] = source_pixels[src + 2];
                        rgba_pixels[dst + 3] = source_pixels[src + 3];
                        break;
                    }
                }

                SDL_Surface* surface = SDL_CreateSurfaceFrom(width, height, SDL_PIXELFORMAT_RGBA32,
                                                             rgba_pixels.data(), width * 4);
                if (!surface) {
                    return nullptr;
                }

                SDL_Cursor* cursor = SDL_CreateColorCursor(surface, hot_x, hot_y);
                SDL_DestroySurface(surface);
                return cursor;
            } catch (const std::exception& e) {
                LOG_WARN("Could not load cursor asset '{}': {}", asset_name, e.what());
                return nullptr;
            }
        }
    } // namespace

    GuiManager::GuiManager(VisualizerImpl* viewer)
        : viewer_(viewer),
          sequencer_ui_(viewer, sequencer_ui_state_, &rmlui_manager_),
          gizmo_manager_(viewer),
          async_tasks_(viewer) {

        panel_layout_.loadState();

        // Create components
        menu_bar_ = std::make_unique<MenuBar>();
        rml_modal_overlay_ = std::make_unique<RmlModalOverlay>(&rmlui_manager_);
        rml_toast_overlay_ = std::make_unique<RmlToastOverlay>(&rmlui_manager_);
        global_context_menu_ = std::make_unique<GlobalContextMenu>(&rmlui_manager_);
        lfs::python::set_global_context_menu(global_context_menu_.get());
        video_widget_ = lfs::gui::createVideoWidget();

        // Initialize window states
        window_states_["scene_panel"] = true;
        window_states_["system_console"] = false;
        window_states_["training_tab"] = false;
        window_states_["export_dialog"] = false;
        window_states_["python_console"] = false;

        lfs::python::set_modal_enqueue_callback(
            [this](lfs::core::ModalRequest req) { enqueueModal(std::move(req)); });

        setupEventHandlers();
        async_tasks_.setupEvents();
        sequencer_ui_.setupEvents();
        gizmo_manager_.setupEvents();
        checkCudaVersionAndNotify();
    }

    void GuiManager::checkCudaVersionAndNotify() {
        using namespace lfs::core;
        const auto info = check_cuda_version();
        if (!info.query_failed && !info.supported) {
            pending_cuda_warning_ = info;
        }
    }

    void GuiManager::promptFileAssociation() {
#ifdef _WIN32
        if (file_association_checked_)
            return;
        file_association_checked_ = true;

        LayoutState state;
        state.load();
        if (state.file_association == "declined")
            return;
        if (areFileAssociationsRegistered())
            return;

        using namespace lichtfeld::Strings;
        lfs::core::ModalRequest req;
        req.title = LOC(FileAssociation::TITLE);
        req.body_rml = "<p>" + std::string(LOC(FileAssociation::MESSAGE)) + "</p>";
        req.style = lfs::core::ModalStyle::Info;
        req.buttons = {
            {LOC(FileAssociation::YES), "primary"},
            {LOC(FileAssociation::NOT_NOW), "secondary"},
            {LOC(FileAssociation::DONT_ASK), "secondary"},
        };
        req.on_result = [](const lfs::core::ModalResult& result) {
            LayoutState ls;
            ls.load();

            if (result.button_label == LOC(FileAssociation::YES)) {
                registerFileAssociations();
                openFileAssociationSettings();
                return;
            } else if (result.button_label == LOC(FileAssociation::DONT_ASK)) {
                ls.file_association = "declined";
            } else {
                return;
            }
            ls.save();
        };

        enqueueModal(std::move(req));
#endif
    }

    GuiManager::~GuiManager() = default;

    void GuiManager::initCustomCursors() {
        if (!pipette_cursor_) {
            // The tip of the dropper sits near the lower-left corner in the 24x24 Tabler asset.
            pipette_cursor_ = loadColorCursorFromAsset("icon/color-picker.png", 4, 19);
            if (!pipette_cursor_)
                LOG_WARN("Could not create pipette cursor from icon/color-picker.png");
        }
    }

    void GuiManager::destroyCustomCursors() {
        if (pipette_cursor_) {
            SDL_SetCursor(SDL_GetDefaultCursor());
            SDL_DestroyCursor(pipette_cursor_);
            pipette_cursor_ = nullptr;
        }
    }

    void GuiManager::applyRmlCursorRequest(const RmlCursorRequest req) {
        if (req != RmlCursorRequest::Pipette && pipette_cursor_)
            SDL_SetCursor(SDL_GetDefaultCursor());

        switch (req) {
        case RmlCursorRequest::Arrow:
            ImGui::SetMouseCursor(ImGuiMouseCursor_Arrow);
            break;
        case RmlCursorRequest::TextInput:
            ImGui::SetMouseCursor(ImGuiMouseCursor_TextInput);
            break;
        case RmlCursorRequest::Hand:
            ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
            break;
        case RmlCursorRequest::Pipette:
            ImGui::SetMouseCursor(ImGuiMouseCursor_Arrow);
            if (pipette_cursor_)
                SDL_SetCursor(pipette_cursor_);
            break;
        case RmlCursorRequest::ResizeEW:
            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
            break;
        case RmlCursorRequest::ResizeNS:
            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
            break;
        case RmlCursorRequest::ResizeNWSE:
            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNWSE);
            break;
        case RmlCursorRequest::ResizeNESW:
            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNESW);
            break;
        case RmlCursorRequest::ResizeAll:
            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeAll);
            break;
        case RmlCursorRequest::NotAllowed:
            ImGui::SetMouseCursor(ImGuiMouseCursor_NotAllowed);
            break;
        case RmlCursorRequest::None:
            break;
        }
    }

    void GuiManager::initMenuBar() {
        menu_bar_->setOnShowPythonConsole([this]() {
            window_states_["python_console"] = !window_states_["python_console"];
        });
    }

    FontSet GuiManager::buildFontSet() const {
        FontSet fs{font_regular_, font_bold_, font_heading_, font_small_, font_section_, font_monospace_};
        for (int i = 0; i < FontSet::MONO_SIZE_COUNT; ++i) {
            fs.monospace_sized[i] = mono_fonts_[i];
            fs.monospace_sizes[i] = mono_font_scales_[i];
        }
        return fs;
    }

    void GuiManager::rebuildFonts(float scale) {
        ImGuiIO& io = ImGui::GetIO();

        io.Fonts->Clear();

        const auto& t = theme();
        try {
            const auto regular_path = lfs::vis::getAssetPath("fonts/" + t.fonts.regular_path);
            const auto bold_path = lfs::vis::getAssetPath("fonts/" + t.fonts.bold_path);
            const auto japanese_path = lfs::vis::getAssetPath("fonts/NotoSansJP-Regular.ttf");
            const auto korean_path = lfs::vis::getAssetPath("fonts/NotoSansKR-Regular.ttf");

            const auto is_font_valid = [](const std::filesystem::path& path) -> bool {
                constexpr size_t MIN_FONT_FILE_SIZE = 100;
                return std::filesystem::exists(path) && std::filesystem::file_size(path) >= MIN_FONT_FILE_SIZE;
            };

            const std::string lang =
                lfs::event::LocalizationManager::getInstance().getCurrentLanguage();
            const bool needs_jp_zh = (lang == "ja" || lang == "zh");
            const bool needs_kr = (lang == "ko");

            const auto load_font_latin_only =
                [&](const std::filesystem::path& path, const float size) -> ImFont* {
                if (!is_font_valid(path))
                    return nullptr;
                const std::string path_utf8 = lfs::core::path_to_utf8(path);
                ImFontConfig config;
                config.PixelSnapH = true;
                return io.Fonts->AddFontFromFileTTF(path_utf8.c_str(), size, &config);
            };

            const auto merge_cjk = [&](const float size) {
                if (needs_jp_zh && is_font_valid(japanese_path)) {
                    ImFontConfig config;
                    config.MergeMode = true;
                    config.OversampleH = 1;
                    config.PixelSnapH = true;
                    const std::string japanese_path_utf8 = lfs::core::path_to_utf8(japanese_path);
                    io.Fonts->AddFontFromFileTTF(japanese_path_utf8.c_str(), size, &config,
                                                 io.Fonts->GetGlyphRangesJapanese());
                    io.Fonts->AddFontFromFileTTF(japanese_path_utf8.c_str(), size, &config,
                                                 io.Fonts->GetGlyphRangesChineseSimplifiedCommon());
                }
                if (needs_kr && is_font_valid(korean_path)) {
                    ImFontConfig config;
                    config.MergeMode = true;
                    config.OversampleH = 1;
                    config.PixelSnapH = true;
                    const std::string korean_path_utf8 = lfs::core::path_to_utf8(korean_path);
                    io.Fonts->AddFontFromFileTTF(korean_path_utf8.c_str(), size, &config,
                                                 io.Fonts->GetGlyphRangesKorean());
                }
            };

            const auto load_font_with_cjk =
                [&](const std::filesystem::path& path, const float size) -> ImFont* {
                ImFont* font = load_font_latin_only(path, size);
                if (!font)
                    return nullptr;
                merge_cjk(size);
                return font;
            };

            font_regular_ = load_font_with_cjk(regular_path, t.fonts.base_size * scale);
            font_bold_ = load_font_with_cjk(bold_path, t.fonts.base_size * scale);
            font_heading_ = load_font_with_cjk(bold_path, t.fonts.heading_size * scale);
            font_small_ = load_font_with_cjk(regular_path, t.fonts.small_size * scale);
            font_section_ = load_font_with_cjk(bold_path, t.fonts.section_size * scale);

            const auto monospace_path = lfs::vis::getAssetPath("fonts/JetBrainsMono-Regular.ttf");
            if (is_font_valid(monospace_path)) {
                const std::string mono_path_utf8 = lfs::core::path_to_utf8(monospace_path);

                static constexpr ImWchar GLYPH_RANGES[] = {
                    0x0020,
                    0x00FF,
                    0x2190,
                    0x21FF,
                    0x2500,
                    0x257F,
                    0x2580,
                    0x259F,
                    0x25A0,
                    0x25FF,
                    0,
                };

                static constexpr float MONO_SCALES[] = {0.7f, 1.0f, 1.3f, 1.7f, 2.2f};
                static_assert(std::size(MONO_SCALES) == FontSet::MONO_SIZE_COUNT);

                for (int i = 0; i < FontSet::MONO_SIZE_COUNT; ++i) {
                    ImFontConfig config;
                    config.GlyphRanges = GLYPH_RANGES;
                    config.PixelSnapH = true;
                    const float size = t.fonts.base_size * scale * MONO_SCALES[i];
                    mono_fonts_[i] = io.Fonts->AddFontFromFileTTF(mono_path_utf8.c_str(), size, &config);
                    mono_font_scales_[i] = MONO_SCALES[i];
                }
                font_monospace_ = mono_fonts_[1];
            }
            if (!font_monospace_)
                font_monospace_ = font_regular_;

            const bool all_loaded = font_regular_ && font_bold_ && font_heading_ && font_small_ && font_section_;
            if (!all_loaded) {
                ImFont* const fallback = font_regular_ ? font_regular_ : io.Fonts->AddFontDefault();
                if (!font_regular_)
                    font_regular_ = fallback;
                if (!font_bold_)
                    font_bold_ = fallback;
                if (!font_heading_)
                    font_heading_ = fallback;
                if (!font_small_)
                    font_small_ = fallback;
                if (!font_section_)
                    font_section_ = fallback;
            }
        } catch (const std::exception& e) {
            LOG_ERROR("Font loading failed: {}", e.what());
            ImFont* const fallback = io.Fonts->AddFontDefault();
            font_regular_ = font_bold_ = font_heading_ = font_small_ = font_section_ = fallback;
        }

        io.Fonts->TexMinWidth = 2048;
        if (!io.Fonts->Build()) {
            LOG_ERROR("Font atlas build failed — CJK glyphs may be missing");
        }

        // Bake an independent FreeType atlas for ScreenOverlayRenderer Text commands.
        // Kept separate from ImGui's font atlas: the Vulkan branch has no ImGui renderer
        // backend, and reaching into io.Fonts trips the new RendererHasTextures contract.
        try {
            const auto regular_path = lfs::vis::getAssetPath("fonts/" + t.fonts.regular_path);
            const float atlas_size_px = std::round(t.fonts.large_size * scale);
            if (!buildOverlayFontAtlas(regular_path, atlas_size_px)) {
                LOG_WARN("Overlay font atlas not built; selection/tool labels won't render");
            }
        } catch (const std::exception& e) {
            LOG_WARN("Overlay font atlas: failed to resolve regular font path: {}", e.what());
        }
        lfs::rendering::ScreenOverlayRenderer::setTextMeasureFn(&measureOverlayText);
    }

    void GuiManager::applyUiScale(float scale) {
        scale = std::clamp(scale, 1.0f, 4.0f);
        const float previous_scale = current_ui_scale_;

        rmlui_manager_.setDpRatio(scale);
        lfs::vis::setThemeDpiScale(scale);
        lfs::python::set_shared_dpi_scale(scale);
        PanelRegistry::instance().rescale_floating_panels(previous_scale, scale);
        applyDefaultStyle();
        rebuildFonts(scale);
        current_ui_scale_ = scale;

        LOG_INFO("UI scale applied: {:.2f}", scale);
    }

    void GuiManager::loadImGuiSettings() {
        if (imgui_ini_path_.empty())
            return;

        try {
            if (!std::filesystem::exists(imgui_ini_path_))
                return;

            std::ifstream file;
            if (!lfs::core::open_file_for_read(imgui_ini_path_, std::ios::binary, file)) {
                LOG_WARN("Failed to open ImGui settings file: {}", lfs::core::path_to_utf8(imgui_ini_path_));
                return;
            }

            const std::string ini_data((std::istreambuf_iterator<char>(file)),
                                       std::istreambuf_iterator<char>());
            ImGui::LoadIniSettingsFromMemory(ini_data.c_str(), ini_data.size());
        } catch (const std::exception& e) {
            LOG_WARN("Failed to load ImGui settings: {}", e.what());
        } catch (...) {
            LOG_WARN("Failed to load ImGui settings: unknown error");
        }
    }

    void GuiManager::saveImGuiSettings() const {
        if (imgui_ini_path_.empty() || !ImGui::GetCurrentContext())
            return;

        try {
            std::filesystem::create_directories(imgui_ini_path_.parent_path());

            size_t ini_size = 0;
            const char* ini_data = ImGui::SaveIniSettingsToMemory(&ini_size);

            std::ofstream file;
            if (!lfs::core::open_file_for_write(imgui_ini_path_,
                                                std::ios::binary | std::ios::trunc,
                                                file)) {
                LOG_WARN("Failed to open ImGui settings for writing: {}",
                         lfs::core::path_to_utf8(imgui_ini_path_));
                return;
            }

            file.write(ini_data, static_cast<std::streamsize>(ini_size));
            if (!file) {
                LOG_WARN("Failed to write ImGui settings: {}",
                         lfs::core::path_to_utf8(imgui_ini_path_));
            }
        } catch (const std::exception& e) {
            LOG_WARN("Failed to save ImGui settings: {}", e.what());
        } catch (...) {
            LOG_WARN("Failed to save ImGui settings: unknown error");
        }
    }

    void GuiManager::persistImGuiSettingsIfNeeded() {
        ImGuiIO& io = ImGui::GetIO();
        if (!io.WantSaveIniSettings)
            return;

        saveImGuiSettings();
        io.WantSaveIniSettings = false;
    }

    void GuiManager::init() {
        // ImGui initialization
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImPlot::CreateContext();

        // Share ImGui state with Python module across DLL boundaries
        ImGuiContext* const ctx = ImGui::GetCurrentContext();
        lfs::python::set_imgui_context(ctx);

        ImGuiMemAllocFunc alloc_fn{};
        ImGuiMemFreeFunc free_fn{};
        void* alloc_user_data{};
        ImGui::GetAllocatorFunctions(&alloc_fn, &free_fn, &alloc_user_data);
        lfs::python::set_imgui_allocator_functions(
            reinterpret_cast<void*>(alloc_fn),
            reinterpret_cast<void*>(free_fn),
            alloc_user_data);
        lfs::python::set_implot_context(ImPlot::GetCurrentContext());

        vulkan_gui_ = viewer_ && viewer_->getWindowManager() && viewer_->getWindowManager()->isVulkan();

        lfs::python::set_ui_texture_service(
            [](const unsigned char*, int, int, int) -> lfs::python::TextureResult {
                return {0, 0, 0};
            },
            [](uint64_t) {},
            []() -> int {
                constexpr int FALLBACK_MAX_TEXTURE_SIZE = 4096;
                return FALLBACK_MAX_TEXTURE_SIZE;
            });

        ImGuiIO& io = ImGui::GetIO();
        imgui_ini_path_ = LayoutState::getConfigDir() / "imgui.ini";
        io.IniFilename = nullptr;
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable; // Enable Docking
        io.ConfigWindowsMoveFromTitleBarOnly = true;
        io.ConfigDragClickToInputText = true;
        loadImGuiSettings();

        // Platform initialization. Vulkan rendering is handled by the app-owned passes below.
        auto* vulkan_context = viewer_->getWindowManager()->getVulkanContext();
        if (!vulkan_context || !ImGui_ImplSDL3_InitForVulkan(viewer_->getWindow())) {
            throw std::runtime_error("Failed to initialize ImGui SDL3 platform backend");
        }
        setVulkanUiTextureContext(vulkan_context);
        if (!vulkan_interop_upload_stream_.init()) {
            LOG_ERROR("Could not create the non-blocking CUDA/Vulkan GUI upload stream: {}",
                      vulkan_interop_upload_stream_.lastError());
        }

        // Initialize localization system
        auto& loc = lfs::event::LocalizationManager::getInstance();
        std::filesystem::path locale_dir = lfs::core::getLocalesDir();
#ifdef LFS_DEV_LOCALE_SOURCE_DIR
        {
            const auto source_locale_dir = lfs::core::utf8_to_path(LFS_DEV_LOCALE_SOURCE_DIR);
            if (std::filesystem::exists(source_locale_dir) &&
                std::filesystem::is_directory(source_locale_dir)) {
                locale_dir = source_locale_dir;
                LOG_INFO("Localization dev source enabled: {}",
                         lfs::core::path_to_utf8(locale_dir));
            }
        }
#endif
        const std::string locale_path = lfs::core::path_to_utf8(locale_dir);
        if (!loc.initialize(locale_path)) {
            LOG_WARN("Failed to initialize localization system, using default strings");
        } else {
            LOG_INFO("Localization initialized with language: {}", loc.getCurrentLanguageName());
        }

        float saved_scale = lfs::vis::loadUiScalePreference();
        if (saved_scale <= 0.0f)
            saved_scale = SDL_GetWindowDisplayScale(viewer_->getWindow());
        current_ui_scale_ = std::clamp(saved_scale, 1.0f, 4.0f);

        lfs::python::set_shared_dpi_scale(current_ui_scale_);
        lfs::vis::setThemeDpiScale(current_ui_scale_);
        initCustomCursors();

        // Set application icon
        try {
            const auto icon_path = lfs::vis::getAssetPath("lichtfeld-icon.png");
            const auto [data, width, height, channels] = lfs::core::load_image_with_alpha(icon_path);

            SDL_Surface* icon_surface = SDL_CreateSurfaceFrom(width, height, SDL_PIXELFORMAT_RGBA32, data, width * 4);
            if (icon_surface) {
                SDL_SetWindowIcon(viewer_->getWindow(), icon_surface);
                SDL_DestroySurface(icon_surface);
            }
            lfs::core::free_image(data);
        } catch (const std::exception& e) {
            LOG_WARN("Could not load application icon: {}", e.what());
        }

        applyDefaultStyle();
        rebuildFonts(current_ui_scale_);

        initMenuBar();

        if (!drag_drop_.init(viewer_->getWindow())) {
            LOG_WARN("Native drag-drop initialization failed, drag-drop will use SDL events only");
        }
        drag_drop_.setFileDropCallback([this](const std::vector<std::string>& paths) {
            LOG_INFO("Files dropped via native drag-drop: {} file(s)", paths.size());
            if (auto* const ic = viewer_->getInputController()) {
                ic->handleFileDrop(paths);
            } else {
                LOG_ERROR("InputController not available for file drop handling");
            }
        });

        {
            auto* rml_vulkan_context = viewer_->getWindowManager()->getVulkanContext();
            if (!rml_vulkan_context || !rmlui_manager_.initVulkan(viewer_->getWindow(), *rml_vulkan_context, current_ui_scale_)) {
                throw std::runtime_error("Failed to initialize RmlUI Vulkan backend");
            }
        }
        lfs::vis::setThemeChangeCallback([this](const std::string& theme_id) {
            rmlui_manager_.activateTheme(theme_id);
            if (auto* const rendering = viewer_ ? viewer_->getRenderingManager() : nullptr) {
                rendering->markDirty(DirtyFlag::OVERLAY);
            }
        });
        lfs::python::set_rml_manager(&rmlui_manager_);
        initDevResourceHotReload();

        startup_overlay_.init(&rmlui_manager_);
        const bool startup_overlay_enabled = viewer_->options_.show_startup_overlay;
        if (!startup_overlay_enabled) {
            LOG_INFO("Startup overlay disabled");
            startup_overlay_.dismiss();
        }
        rml_shell_frame_.init(&rmlui_manager_);
        rml_right_panel_.init(&rmlui_manager_);
        rml_right_panel_.on_tab_changed = [this](const std::string& id) {
            panel_layout_.setActiveTab(id);
        };
        rml_right_panel_.on_tab_closed = [this](const std::string& id) {
            if (id.empty())
                return;
            PanelRegistry::instance().set_panel_enabled(id, false);
            if (panel_layout_.getActiveTab() == id)
                panel_layout_.setActiveTab({});
            if (focus_panel_name_ == id)
                focus_panel_name_.clear();
        };
        rml_right_panel_.on_splitter_delta = [this](float delta_y) {
            viewer_->getRenderingManager()->setViewportResizeActive(true);
            const auto* mvp = ImGui::GetMainViewport();
            ScreenState ss;
            ss.work_pos = {mvp->WorkPos.x, mvp->WorkPos.y};
            ss.work_size = {mvp->WorkSize.x, mvp->WorkSize.y};
            panel_layout_.adjustScenePanelRatio(delta_y, ss);
        };
        rml_right_panel_.on_splitter_end = [this]() {
            viewer_->getRenderingManager()->setViewportResizeActive(false);
        };
        rml_right_panel_.on_resize_delta = [this](float dx) {
            viewer_->getRenderingManager()->setViewportResizeActive(true);
            const auto* mvp = ImGui::GetMainViewport();
            ScreenState ss;
            ss.work_pos = {mvp->WorkPos.x, mvp->WorkPos.y};
            ss.work_size = {mvp->WorkSize.x, mvp->WorkSize.y};
            panel_layout_.applyResizeDelta(dx, ss);
        };
        rml_right_panel_.on_resize_end = [this]() {
            viewer_->getRenderingManager()->setViewportResizeActive(false);
        };
        rml_viewport_overlay_.init(&rmlui_manager_);
        rml_menu_bar_.init(&rmlui_manager_);
        rml_status_bar_.init(&rmlui_manager_);
        if (global_context_menu_)
            global_context_menu_->preload();
        if (rml_modal_overlay_)
            rml_modal_overlay_->preload();

        lfs::python::RmlPanelHostOps ops{};
        ops.create = [](void* mgr, const char* name, const char* rml,
                        const char* inline_rcss) -> void* {
            return new RmlPanelHost(static_cast<RmlUIManager*>(mgr),
                                    std::string(name), std::string(rml),
                                    inline_rcss ? std::string(inline_rcss) : std::string{});
        };
        ops.destroy = [](void* host) {
            if (lfs::python::on_graphics_thread()) {
                delete static_cast<RmlPanelHost*>(host);
            } else {
                lfs::python::schedule_graphics_callback([host]() {
                    delete static_cast<RmlPanelHost*>(host);
                });
            }
        };
        ops.draw = [](void* host, const void* ctx) {
            auto* h = static_cast<RmlPanelHost*>(host);
            float aw = ImGui::GetContentRegionAvail().x;
            float ah = ImGui::GetContentRegionAvail().y;
            ImVec2 pos = ImGui::GetCursorScreenPos();

            PanelInputState fallback;
            if (!h->hasInput() && s_frame_input) {
                fallback = buildPanelInputFromSDL(*s_frame_input);
                h->setInput(&fallback);
            }
            h->draw(*static_cast<const PanelDrawContext*>(ctx),
                    aw, ah, pos.x, pos.y);
            h->setInput(nullptr);
        };
        ops.draw_direct = [](void* host, float x, float y, float w, float h) {
            auto* hp = static_cast<RmlPanelHost*>(host);
            PanelInputState fallback;
            if (!hp->hasInput() && s_frame_input) {
                fallback = buildPanelInputFromSDL(*s_frame_input);
                auto* mvp = ImGui::GetMainViewport();
                fallback.bg_draw_list = ImGui::GetForegroundDrawList(mvp);
                fallback.fg_draw_list = ImGui::GetForegroundDrawList(mvp);
                hp->setInput(&fallback);
            }
            hp->drawDirect(x, y, w, h);
            hp->setInput(nullptr);
        };
        ops.draw_direct_cached = [](void* host, float x, float y, float w, float h) -> bool {
            return static_cast<RmlPanelHost*>(host)->drawDirectCached(x, y, w, h);
        };
        ops.prepare_direct = [](void* host, float w, float h) {
            auto* hp = static_cast<RmlPanelHost*>(host);
            PanelInputState fallback;
            if (!hp->hasInput() && s_frame_input) {
                fallback = buildPanelInputFromSDL(*s_frame_input);
                hp->setInput(&fallback);
            }
            hp->prepareDirect(w, h);
            hp->setInput(nullptr);
        };
        ops.prepare_layout = [](void* host, float w, float h) {
            static_cast<RmlPanelHost*>(host)->syncDirectLayout(w, h);
        };
        ops.get_document = [](void* host) -> void* {
            return static_cast<RmlPanelHost*>(host)->getDocument();
        };
        ops.is_loaded = [](void* host) -> bool {
            return static_cast<RmlPanelHost*>(host)->isDocumentLoaded();
        };
        ops.set_height_mode = [](void* host, int mode) {
            static_cast<RmlPanelHost*>(host)->setHeightMode(
                static_cast<PanelHeightMode>(mode));
        };
        ops.get_content_height = [](void* host) -> float {
            return static_cast<RmlPanelHost*>(host)->getContentHeight();
        };
        ops.ensure_context = [](void* host) -> bool {
            return static_cast<RmlPanelHost*>(host)->ensureContext();
        };
        ops.ensure_document = [](void* host) -> bool {
            return static_cast<RmlPanelHost*>(host)->ensureDocumentLoaded();
        };
        ops.reload_document = [](void* host) -> bool {
            return static_cast<RmlPanelHost*>(host)->reloadDocument();
        };
        ops.get_context = [](void* host) -> void* {
            return static_cast<RmlPanelHost*>(host)->getContext();
        };
        ops.set_foreground = [](void* host, bool fg) {
            static_cast<RmlPanelHost*>(host)->setForeground(fg);
        };
        ops.mark_content_dirty = [](void* host) {
            static_cast<RmlPanelHost*>(host)->markContentDirty();
        };
        ops.set_input_clip_y = [](void* host, float y_min, float y_max) {
            static_cast<RmlPanelHost*>(host)->setInputClipY(y_min, y_max);
        };
        ops.set_input = [](void* host, const void* input) {
            static_cast<RmlPanelHost*>(host)->setInput(
                static_cast<const PanelInputState*>(input));
        };
        ops.set_forced_height = [](void* host, float h) {
            static_cast<RmlPanelHost*>(host)->setForcedHeight(h);
        };
        ops.needs_animation = [](void* host) -> bool {
            return static_cast<RmlPanelHost*>(host)->needsAnimationFrame();
        };
        lfs::python::set_rml_panel_host_ops(ops);

        registerNativePanels();
    }

    void GuiManager::initDevResourceHotReload() {
        dev_resource_watch_ = {};

#if !defined(LFS_BUILD_PORTABLE) && (defined(LFS_DEV_RMLUI_SOURCE_DIR) || defined(LFS_DEV_LOCALE_SOURCE_DIR))
        if (!lfs::core::environment::flag("LFS_DEV_HOT_RELOAD", true))
            return;

#ifdef LFS_DEV_RMLUI_SOURCE_DIR
        {
            const auto dir = lfs::core::utf8_to_path(LFS_DEV_RMLUI_SOURCE_DIR);
            if (std::filesystem::exists(dir) && std::filesystem::is_directory(dir))
                dev_resource_watch_.rml_dir = dir;
        }
#endif
#ifdef LFS_DEV_LOCALE_SOURCE_DIR
        {
            const auto dir = lfs::core::utf8_to_path(LFS_DEV_LOCALE_SOURCE_DIR);
            if (std::filesystem::exists(dir) && std::filesystem::is_directory(dir))
                dev_resource_watch_.locale_dir = dir;
        }
#endif

        dev_resource_watch_.enabled =
            !dev_resource_watch_.rml_dir.empty() || !dev_resource_watch_.locale_dir.empty();
        if (!dev_resource_watch_.enabled)
            return;

        scanDevResourceFiles(false);
        dev_resource_watch_.next_scan = std::chrono::steady_clock::now() + std::chrono::seconds(1);
        LOG_INFO("Resource hot reload enabled (RmlUI: '{}', locales: '{}')",
                 dev_resource_watch_.rml_dir.empty() ? std::string("<disabled>")
                                                     : lfs::core::path_to_utf8(dev_resource_watch_.rml_dir),
                 dev_resource_watch_.locale_dir.empty() ? std::string("<disabled>")
                                                        : lfs::core::path_to_utf8(dev_resource_watch_.locale_dir));
#endif
    }

    GuiManager::DevResourceScanResult GuiManager::scanDevResourceFilesSnapshot(
        std::filesystem::path rml_dir,
        std::filesystem::path locale_dir,
        std::unordered_map<std::string, std::filesystem::file_time_type> previous_times,
        const bool detect_changes) {
        DevResourceScanResult result;
        std::unordered_map<std::string, std::filesystem::file_time_type> next_times;

        const auto scan_dir =
            [&](const std::filesystem::path& dir, const bool is_locale_dir) {
                if (dir.empty())
                    return;

                std::error_code ec;
                if (!std::filesystem::exists(dir, ec) || ec ||
                    !std::filesystem::is_directory(dir, ec) || ec) {
                    if (ec)
                        result.scan_failed = true;
                    return;
                }

                try {
                    std::filesystem::recursive_directory_iterator it(
                        dir, std::filesystem::directory_options::skip_permission_denied, ec);
                    const std::filesystem::recursive_directory_iterator end;
                    if (ec) {
                        result.scan_failed = true;
                        return;
                    }
                    for (; !ec && it != end; it.increment(ec)) {
                        std::error_code entry_ec;
                        if (!it->is_regular_file(entry_ec) || entry_ec)
                            continue;

                        const auto kind = devResourceKindForPath(it->path());
                        const bool watched = is_locale_dir
                                                 ? kind == DevResourceKind::Locale
                                                 : kind == DevResourceKind::Rml;
                        if (!watched)
                            continue;

                        const auto mtime = std::filesystem::last_write_time(it->path(), entry_ec);
                        if (entry_ec)
                            continue;

                        const std::string key = lfs::core::path_to_utf8(it->path().lexically_normal());
                        next_times[key] = mtime;

                        if (!detect_changes)
                            continue;

                        const auto old = previous_times.find(key);
                        if (old == previous_times.end() || old->second != mtime) {
                            if (is_locale_dir)
                                result.locale_changed = true;
                            else
                                result.rml_changed = true;
                        }
                    }
                    if (ec)
                        result.scan_failed = true;
                } catch (const std::filesystem::filesystem_error& e) {
                    result.scan_failed = true;
                    LOG_WARN("Resource hot reload scan skipped for '{}': {}",
                             lfs::core::path_to_utf8(dir), e.what());
                }
            };

        scan_dir(rml_dir, false);
        scan_dir(locale_dir, true);

        if (result.scan_failed)
            return result;

        if (detect_changes) {
            for (const auto& [key, unused] : previous_times) {
                (void)unused;
                if (next_times.contains(key))
                    continue;

                const auto kind = devResourceKindForPath(lfs::core::utf8_to_path(key));
                if (kind == DevResourceKind::Locale)
                    result.locale_changed = true;
                else if (kind == DevResourceKind::Rml)
                    result.rml_changed = true;
            }
        }

        result.file_times = std::move(next_times);
        return result;
    }

    GuiManager::DevResourceScanResult GuiManager::scanDevResourceFiles(const bool detect_changes) {
        auto result = scanDevResourceFilesSnapshot(dev_resource_watch_.rml_dir,
                                                   dev_resource_watch_.locale_dir,
                                                   dev_resource_watch_.file_times,
                                                   detect_changes);
        if (!result.scan_failed)
            dev_resource_watch_.file_times = result.file_times;
        return result;
    }

    void GuiManager::launchDevResourceScan() {
        if (dev_resource_watch_.scan_future.valid())
            return;

        auto rml_dir = dev_resource_watch_.rml_dir;
        auto locale_dir = dev_resource_watch_.locale_dir;
        auto previous_times = dev_resource_watch_.file_times;
        try {
            dev_resource_watch_.scan_future =
                std::async(std::launch::async,
                           [rml_dir = std::move(rml_dir),
                            locale_dir = std::move(locale_dir),
                            previous_times = std::move(previous_times)]() mutable {
                               return GuiManager::scanDevResourceFilesSnapshot(
                                   std::move(rml_dir),
                                   std::move(locale_dir),
                                   std::move(previous_times),
                                   true);
                           });
        } catch (const std::exception& e) {
            LOG_WARN("Resource hot reload async scan could not start: {}", e.what());
        }
    }

    bool GuiManager::consumeDevResourceScanResult() {
        if (!dev_resource_watch_.scan_future.valid())
            return false;

        using namespace std::chrono_literals;
        if (dev_resource_watch_.scan_future.wait_for(0ms) != std::future_status::ready)
            return false;

        DevResourceScanResult result;
        try {
            result = dev_resource_watch_.scan_future.get();
        } catch (const std::exception& e) {
            LOG_WARN("Resource hot reload async scan failed: {}", e.what());
            return true;
        }
        if (!result.scan_failed) {
            dev_resource_watch_.file_times = std::move(result.file_times);
            dev_resource_watch_.pending_rml_reload |= result.rml_changed;
            dev_resource_watch_.pending_locale_reload |= result.locale_changed;
        }
        return true;
    }

    bool GuiManager::reloadLocalizationResources() {
        if (dev_resource_watch_.locale_dir.empty())
            return false;

        auto& loc = lfs::event::LocalizationManager::getInstance();
        const std::string current_language = loc.getCurrentLanguage();
        const std::string locale_path = lfs::core::path_to_utf8(dev_resource_watch_.locale_dir);
        if (!loc.initialize(locale_path)) {
            LOG_WARN("Failed to reload localization resources from {}", locale_path);
            return false;
        }

        if (!current_language.empty() && current_language != loc.getCurrentLanguage()) {
            const auto available = loc.getAvailableLanguages();
            if (std::find(available.begin(), available.end(), current_language) != available.end()) {
                loc.setLanguage(current_language);
            }
        }

        lfs::vis::publish_language_generation();
        return true;
    }

    void GuiManager::reloadRmlResources() {
        rml_theme::invalidateBaseRcssCache();
        rml_theme::invalidateThemeMediaCache();

        startup_overlay_.reloadResources();
        rml_shell_frame_.reloadResources();
        rml_right_panel_.reloadResources();
        rml_viewport_overlay_.reloadResources();
        rml_menu_bar_.reloadResources();
        rml_status_bar_.reloadResources();
        sequencer_ui_.reloadRmlResources();
        PanelRegistry::instance().reload_rml_resources();

        if (rml_modal_overlay_)
            rml_modal_overlay_->reloadResources();
        if (rml_toast_overlay_)
            rml_toast_overlay_->reloadResources();
        if (global_context_menu_)
            global_context_menu_->reloadResources();

        if (auto* const rendering = viewer_ ? viewer_->getRenderingManager() : nullptr)
            rendering->markDirty(DirtyFlag::OVERLAY);
    }

    bool GuiManager::shouldDeferDevResourceHotReload() const {
        if (ImGui::GetCurrentContext()) {
            const ImGuiIO& io = ImGui::GetIO();
            if (io.WantTextInput || ImGui::IsAnyItemActive() ||
                ImGui::IsMouseDown(ImGuiMouseButton_Left) ||
                ImGui::IsMouseDown(ImGuiMouseButton_Right) ||
                ImGui::IsMouseDown(ImGuiMouseButton_Middle)) {
                return true;
            }
        }
        if (rmlui_manager_.wantsTextInput() || rmlui_manager_.anyItemActive())
            return true;

        if (rml_menu_bar_.isOpen())
            return true;
        if (global_context_menu_ && global_context_menu_->isOpen())
            return true;
        if (rml_modal_overlay_ && rml_modal_overlay_->isOpen())
            return true;

        return false;
    }

    void GuiManager::pollDevResourceHotReload() {
        if (!dev_resource_watch_.enabled)
            return;

        consumeDevResourceScanResult();

        if (dev_resource_watch_.pending_rml_reload ||
            dev_resource_watch_.pending_locale_reload) {
            if (shouldDeferDevResourceHotReload())
                return;

            const bool reload_rml = dev_resource_watch_.pending_rml_reload;
            const bool reload_locale = dev_resource_watch_.pending_locale_reload;
            dev_resource_watch_.pending_rml_reload = false;
            dev_resource_watch_.pending_locale_reload = false;

            if (reload_locale)
                reloadLocalizationResources();
            if (reload_rml || reload_locale)
                reloadRmlResources();

            LOG_INFO("Hot-reloaded dev resources{}{}",
                     reload_rml ? " (RmlUI)" : "",
                     reload_locale ? " (locales)" : "");
        }

        if (dev_resource_watch_.scan_future.valid())
            return;

        const auto now = std::chrono::steady_clock::now();
        if (dev_resource_watch_.next_scan != std::chrono::steady_clock::time_point{} &&
            now < dev_resource_watch_.next_scan) {
            return;
        }
        dev_resource_watch_.next_scan = now + std::chrono::seconds(1);
        launchDevResourceScan();
    }

    void GuiManager::shutdown() {
        endInteractiveTransitionGuard();
        ui_toggle_pending_ = false;
        fullscreen_toggle_pending_ = false;
        interactive_transition_guard_until_ = {};

        if (dev_resource_watch_.scan_future.valid())
            dev_resource_watch_.scan_future.wait();

        panel_layout_.saveState();

        if (video_widget_)
            video_widget_->shutdown();

        async_tasks_.shutdown();

        const bool need_gil = lfs::python::get_main_thread_state() != nullptr;
        if (need_gil)
            lfs::python::acquire_gil_main_thread();

        lfs::python::shutdown_python_ui_resources();
        lfs::python::set_modal_enqueue_callback({});
        lfs::python::set_global_context_menu(nullptr);

        rml_modal_overlay_.reset();
        rml_toast_overlay_.reset();
        global_context_menu_.reset();
        panels::ShutdownPythonConsoleRml();
        rml_status_bar_.shutdown();
        rml_menu_bar_.shutdown();
        rml_viewport_overlay_.shutdown();
        rml_right_panel_.shutdown();
        rml_shell_frame_.shutdown();
        startup_overlay_.shutdown();
        sequencer_ui_.destroyGraphicsResources();
        for (const auto& panel : native_panel_storage_) {
            if (panel)
                panel->releaseRendererResources();
        }
        PanelRegistry::instance().unregister_all_non_native();
        rmlui_manager_.shutdown();

        if (need_gil)
            lfs::python::release_gil_main_thread();

        drag_drop_.shutdown();
        destroyCustomCursors();
        lfs::rendering::ScreenOverlayRenderer::setTextMeasureFn({});
        g_overlay_atlas = {};
        // The process-wide camera cache owns device-local atlas pages. Dataset
        // camera thumbnails can span several pages, so release all of them while
        // the Vulkan context and its allocator are still alive.
        cameraThumbnailCache().clear();
        // IconCache is process-global, but its Vulkan textures belong to this
        // device. Release them while the context and allocator are still alive;
        // static destruction happens after VulkanContext::shutdown().
        IconCache::instance().clear();
        setVulkanUiTextureContext(nullptr);
        if (!vulkan_interop_upload_stream_.synchronize()) {
            LOG_WARN("CUDA/Vulkan GUI upload stream synchronization failed during shutdown: {}",
                     vulkan_interop_upload_stream_.lastError());
        }
        resetVulkanSceneInterop();
        resetVulkanSplitRightInterop();
        resetVulkanDepthBlitInterop();
        vulkan_interop_upload_stream_.reset();
        vulkan_scene_image_.reset();
        vulkan_viewport_pass_.reset();

        if (ImGui::GetCurrentContext()) {
            saveImGuiSettings();
            ImGui_ImplSDL3_Shutdown();
            ImPlot::DestroyContext();
            ImGui::DestroyContext();
        }
        vulkan_gui_ = false;
    }

    void GuiManager::registerNativePanels() {
        using namespace native_panels;
        auto& reg = PanelRegistry::instance();

        auto make_panel = [this](auto panel) -> std::shared_ptr<IPanel> {
            auto ptr = std::make_shared<decltype(panel)>(std::move(panel));
            native_panel_storage_.push_back(ptr);
            return ptr;
        };

        auto reg_panel = [&](const std::string& id, const std::string& label,
                             std::shared_ptr<IPanel> panel, PanelSpace space, int order,
                             uint32_t options = 0, float initial_width = 0, float initial_height = 0) {
            PanelInfo info;
            info.panel = std::move(panel);
            info.label = label;
            info.id = id;
            info.space = space;
            info.order = order;
            info.options = options;
            info.is_native = true;
            info.initial_width = initial_width;
            info.initial_height = initial_height;
            reg.register_panel(std::move(info));
        };

        // Floating panels (self-managed windows)
        {
            auto panel = std::static_pointer_cast<IPanel>(
                std::make_shared<NativeScenePanel>(&rmlui_manager_));
            native_panel_storage_.push_back(panel);
            reg_panel("lfs.scene", "Scene", panel, PanelSpace::SceneHeader, 0);
        }

        reg_panel("native.video_extractor", "Video Extractor",
                  make_panel(VideoExtractorPanel(video_widget_.get())),
                  PanelSpace::Floating, 11,
                  0,
                  1082.0f, 920.0f);
        reg.set_panel_enabled("native.video_extractor", false);

        // Viewport overlays (ordered by draw priority)
        reg_panel("native.selection_overlay", "Selection Overlay",
                  make_panel(SelectionOverlayPanel(this)),
                  PanelSpace::ViewportOverlay, 200);

        reg_panel("native.node_transform_gizmo", "Node Transform",
                  make_panel(NodeTransformGizmoPanel(&gizmo_manager_)),
                  PanelSpace::ViewportOverlay, 300);

        reg_panel("native.cropbox_gizmo", "Crop Box",
                  make_panel(CropBoxGizmoPanel(&gizmo_manager_)),
                  PanelSpace::ViewportOverlay, 301);

        reg_panel("native.ellipsoid_gizmo", "Ellipsoid",
                  make_panel(EllipsoidGizmoPanel(&gizmo_manager_)),
                  PanelSpace::ViewportOverlay, 302);

        reg_panel("native.sequencer", "Sequencer",
                  make_panel(SequencerPanel(&sequencer_ui_, &panel_layout_)),
                  PanelSpace::BottomDock, 500,
                  0, 8192.0f);

        reg_panel("native.python_overlay", "Python Overlay",
                  make_panel(PythonOverlayPanel(this)),
                  PanelSpace::ViewportOverlay, 500);

        reg_panel("native.viewport_decorations", "Viewport Decorations",
                  make_panel(ViewportDecorationsPanel(this)),
                  PanelSpace::ViewportOverlay, 800);

        reg_panel("native.viewport_gizmo", "Viewport Gizmo",
                  make_panel(ViewportGizmoPanel(&gizmo_manager_)),
                  PanelSpace::ViewportOverlay, 900);

        reg_panel("native.pie_menu", "Pie Menu",
                  make_panel(PieMenuPanel(&gizmo_manager_)),
                  PanelSpace::ViewportOverlay, 950);

        reg_panel("native.startup_overlay", "Startup Overlay",
                  make_panel(StartupOverlayPanel(&startup_overlay_, &drag_drop_hovering_)),
                  PanelSpace::ViewportOverlay, 0);
    }

    void GuiManager::setVulkanSceneImage(std::shared_ptr<const lfs::core::Tensor> image,
                                         const glm::ivec2 size,
                                         const bool flip_y,
                                         const std::uint64_t generation,
                                         const VkSemaphore completion_semaphore,
                                         const std::uint64_t completion_value) {
        const bool target_changed =
            vulkan_scene_image_.get() != image.get() ||
            vulkan_scene_image_size_ != size;
        if (target_changed) {
            vulkan_scene_interop_disabled_ = false;
        }
        vulkan_external_scene_image_ = VK_NULL_HANDLE;
        vulkan_external_scene_image_view_ = VK_NULL_HANDLE;
        vulkan_external_scene_image_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
        vulkan_external_scene_image_size_ = {0, 0};
        vulkan_frame_completion_semaphore_ = completion_semaphore;
        vulkan_frame_completion_value_ = completion_value;
        vulkan_scene_image_ = std::move(image);
        vulkan_scene_image_generation_ = generation;
        vulkan_scene_image_size_ = size;
        vulkan_scene_image_flip_y_ = flip_y;
    }

    void GuiManager::setVulkanExternalSceneImage(const VkImage image,
                                                 const VkImageView image_view,
                                                 const VkImageLayout layout,
                                                 const glm::ivec2 size,
                                                 const bool flip_y,
                                                 const std::uint64_t generation,
                                                 const VkSemaphore completion_semaphore,
                                                 const std::uint64_t completion_value) {
        vulkan_scene_image_.reset();
        vulkan_scene_image_size_ = size;
        vulkan_scene_image_flip_y_ = flip_y;
        vulkan_external_scene_image_ = image;
        vulkan_external_scene_image_view_ = image_view;
        vulkan_external_scene_image_layout_ = layout;
        vulkan_external_scene_image_size_ = size;
        vulkan_external_scene_image_flip_y_ = flip_y;
        vulkan_external_scene_image_generation_ = generation;
        vulkan_frame_completion_semaphore_ = completion_semaphore;
        vulkan_frame_completion_value_ = completion_value;
    }

    void GuiManager::setVulkanSplitRightImage(std::shared_ptr<const lfs::core::Tensor> image,
                                              const glm::ivec2 size,
                                              const bool flip_y,
                                              const std::uint64_t generation) {
        const bool target_changed =
            vulkan_split_right_image_.get() != image.get() ||
            vulkan_split_right_image_size_ != size;
        if (target_changed) {
            vulkan_split_right_interop_disabled_ = false;
        }
        vulkan_split_right_image_ = std::move(image);
        vulkan_split_right_image_generation_ = generation;
        vulkan_split_right_image_size_ = size;
        vulkan_split_right_image_flip_y_ = flip_y;
    }

    void GuiManager::clearVulkanSplitRightImage() {
        vulkan_split_right_image_.reset();
        vulkan_split_right_image_size_ = {0, 0};
        vulkan_split_right_image_flip_y_ = false;
        vulkan_split_right_image_generation_ = 0;
        vulkan_split_right_external_image_ = VK_NULL_HANDLE;
        vulkan_split_right_external_image_view_ = VK_NULL_HANDLE;
        vulkan_split_right_external_image_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
        vulkan_split_right_external_image_generation_ = 0;
    }

    void GuiManager::setVulkanDepthBlitImage(std::shared_ptr<const lfs::core::Tensor> depth,
                                             const glm::ivec2 size,
                                             const std::uint64_t generation) {
        const bool target_changed =
            vulkan_depth_blit_image_.get() != depth.get() ||
            vulkan_depth_blit_image_size_ != size;
        if (target_changed) {
            vulkan_depth_blit_interop_disabled_ = false;
        }
        vulkan_depth_blit_image_ = std::move(depth);
        vulkan_depth_blit_image_generation_ = generation;
        vulkan_depth_blit_image_size_ = size;
    }

    void GuiManager::clearVulkanDepthBlitImage() {
        vulkan_depth_blit_image_.reset();
        vulkan_depth_blit_image_size_ = {0, 0};
        vulkan_depth_blit_image_generation_ = 0;
        vulkan_depth_blit_external_image_ = VK_NULL_HANDLE;
        vulkan_depth_blit_external_image_view_ = VK_NULL_HANDLE;
        vulkan_depth_blit_external_image_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
        vulkan_depth_blit_external_image_generation_ = 0;
    }

    void GuiManager::resetVulkanSceneInterop() {
        if (vulkan_scene_interop_.empty()) {
            return;
        }
        auto* const window_manager = viewer_ ? viewer_->getWindowManager() : nullptr;
        auto* const vulkan_context = window_manager ? window_manager->getVulkanContext() : nullptr;
        // A previous frame's submit may still sample one of these slots; drain
        // before vkDestroyImage to avoid VK_ERROR_DEVICE_LOST.
        if (vulkan_context) {
            (void)vulkan_context->waitForSubmittedFrames();
        }
        for (auto& target : vulkan_scene_interop_) {
            if (!target) {
                continue;
            }
            if (vulkan_context) {
                target->destroy(*vulkan_context);
            } else {
                target->interop.reset();
            }
        }
        vulkan_scene_interop_.clear();
    }

    bool GuiManager::shouldDeferVulkanInteropResize() const {
        auto* const rendering = viewer_ ? viewer_->getRenderingManager() : nullptr;
        return rendering && rendering->isViewportResizeDeferring();
    }

    void GuiManager::prepareVulkanSceneInterop(VulkanContext& context) {
        if (vulkan_scene_interop_disabled_) {
            return;
        }
        if (vulkan_external_scene_image_ != VK_NULL_HANDLE) {
            return;
        }

        const auto fail_required_interop = [this](std::string message) -> void {
            vulkan_scene_interop_disabled_ = true;
            if (!vulkan_interop_upload_stream_.synchronize()) {
                message += std::format("; CUDA upload drain failed: {}",
                                       vulkan_interop_upload_stream_.lastError());
            }
            resetVulkanSceneInterop();
            LOG_ERROR("Required Vulkan/CUDA viewport interop failed: {}", message);
            throw std::runtime_error(std::move(message));
        };

        if (!vulkan_scene_image_ ||
            !vulkan_scene_image_->is_valid() ||
            vulkan_scene_image_->device() != lfs::core::Device::CUDA ||
            vulkan_scene_image_size_.x <= 0 ||
            vulkan_scene_image_size_.y <= 0) {
            if (!vulkan_scene_interop_.empty()) {
                resetVulkanSceneInterop();
            }
            return;
        }
        if (!vulkan_interop_upload_stream_.valid()) {
            fail_required_interop("non-blocking CUDA upload stream is unavailable");
        }

        const std::size_t frame_slot = context.currentFrameSlot();
        const bool slot_array_resize_needed = vulkan_scene_interop_.size() != context.framesInFlight();
        const bool resize_deferring = shouldDeferVulkanInteropResize();

        // Cache-HIT fast path: when nothing about the source image changed since the last
        // upload into THIS slot's interop target, there's no work to do — and crucially no
        // need to vkWaitForFences this slot. The previous unconditional wait was costing
        // ~kFrameDuration ms per frame (10–12 ms with kFramesInFlight=1) for no reason on
        // every renderer cache-HIT frame, which dominated gui_render time.
        if (!slot_array_resize_needed && frame_slot < vulkan_scene_interop_.size()) {
            const auto& target_ptr_const = vulkan_scene_interop_[frame_slot];
            const glm::ivec2 target_size = vulkan_scene_image_size_;
            const bool recreate_needed =
                !target_ptr_const ||
                target_ptr_const->size != target_size ||
                !target_ptr_const->interop.valid();
            if (!recreate_needed &&
                vulkan_scene_image_generation_ != 0 &&
                target_ptr_const->uploaded_source_generation == vulkan_scene_image_generation_ &&
                target_ptr_const->layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
                LOG_PERF("interop slot={} cache-HIT-skip cur_gen={} layout={}",
                         frame_slot, vulkan_scene_image_generation_,
                         static_cast<int>(target_ptr_const->layout));
                return;
            }
            if (resize_deferring && recreate_needed) {
                return;
            }
        } else if (resize_deferring) {
            return;
        }

        // Slow path: we will write to the interop image (recreate, transition, or copy).
        // Wait for any in-flight GPU use of this slot to finish before we touch it.
        {
            LOG_TIMER("interop.waitForCurrentFrameSlot");
            if (!context.waitForCurrentFrameSlot()) {
                fail_required_interop(std::format("frame slot wait failed: {}", context.lastError()));
            }
        }

        if (slot_array_resize_needed) {
            resetVulkanSceneInterop();
            vulkan_scene_interop_.resize(context.framesInFlight());
        }
        if (frame_slot >= vulkan_scene_interop_.size()) {
            fail_required_interop(std::format("invalid frame slot {}", frame_slot));
        }
        auto& target_ptr = vulkan_scene_interop_[frame_slot];
        const auto reset_frame_target = [&]() {
            if (target_ptr) {
                target_ptr->destroy(context);
                target_ptr.reset();
            }
        };

        const glm::ivec2 target_size = vulkan_scene_image_size_;
        const bool recreate =
            !target_ptr ||
            target_ptr->size != target_size ||
            !target_ptr->interop.valid();
        LOG_PERF("interop slot={} recreate={} cur_gen={} uploaded_gen={} layout={}",
                 frame_slot, recreate,
                 vulkan_scene_image_generation_,
                 target_ptr ? target_ptr->uploaded_source_generation : 0,
                 target_ptr ? static_cast<int>(target_ptr->layout) : -1);
        if (recreate) {
            reset_frame_target();
            auto target = std::make_unique<VulkanSceneInteropTarget>();
            const VkExtent2D extent{
                static_cast<std::uint32_t>(target_size.x),
                static_cast<std::uint32_t>(target_size.y),
            };
            if (!context.createExternalImage(extent,
                                             VK_FORMAT_R8G8B8A8_UNORM,
                                             target->image,
                                             "vulkan.gui.interop_image",
                                             std::format("scene.frame{}", frame_slot)) ||
                !context.createExternalTimelineSemaphore(0, target->semaphore)) {
                const std::string error = std::format("target creation failed: {}", context.lastError());
                if (target->image.image != VK_NULL_HANDLE || target->semaphore.semaphore != VK_NULL_HANDLE) {
                    target->destroy(context);
                }
                fail_required_interop(error);
            }
            const std::uint64_t vulkan_ready_value = ++target->timeline_value;
            if (!context.transitionImageLayoutImmediate(target->image.image,
                                                        VK_IMAGE_LAYOUT_UNDEFINED,
                                                        VK_IMAGE_LAYOUT_GENERAL,
                                                        VulkanContext::ImmediateTransitionOptions::signalAt(
                                                            {target->semaphore.semaphore, vulkan_ready_value}))) {
                const std::string error = std::format("image initialization failed: {}", context.lastError());
                target->destroy(context);
                fail_required_interop(error);
            }
            // Complete the one-time Vulkan initialization before exporting the
            // timeline to CUDA. Later handoffs remain asynchronous, but no
            // external producer may advance this semaphore past the pending
            // Vulkan signal that establishes its initial image ownership.
            if (!context.waitForImmediateSubmits()) {
                const std::string error = std::format(
                    "image initialization handoff failed: {}", context.lastError());
                target->destroy(context);
                fail_required_interop(error);
            }

            const auto memory_handle = context.releaseExternalImageNativeHandle(target->image);
            const auto semaphore_handle = context.releaseExternalSemaphoreNativeHandle(target->semaphore);
            lfs::rendering::CudaVulkanExternalImageImport image_import{
                .memory_handle = memory_handle,
                .allocation_size = static_cast<std::size_t>(target->image.allocation_size),
                .extent = {.width = extent.width, .height = extent.height},
                .format = lfs::rendering::CudaVulkanImageFormat::Rgba8Unorm,
                .dedicated_allocation = context.externalMemoryDedicatedAllocationEnabled(),
            };
            lfs::rendering::CudaVulkanExternalSemaphoreImport semaphore_import{
                .semaphore_handle = semaphore_handle,
                .initial_value = 0,
            };
            if (!target->interop.init(image_import, semaphore_import)) {
                const std::string error = std::format("CUDA import failed: {}", target->interop.lastError());
                target->destroy(context);
                fail_required_interop(error);
            }
            target->size = target_size;
            target->layout = VK_IMAGE_LAYOUT_GENERAL;
            target_ptr = std::move(target);
            LOG_INFO("Vulkan/CUDA viewport interop target initialized for frame slot {}: {}x{}",
                     frame_slot,
                     target_size.x,
                     target_size.y);
        }

        auto& target = *target_ptr;
        // Skip the upload (and the queue-blocking layout transitions inside it)
        // when this slot already holds the same content. Renderer cache-HIT
        // frames keep image_generation stable while alternating tensor pointers,
        // so identity-by-pointer is unsafe — use the source generation.
        if (vulkan_scene_image_generation_ != 0 &&
            target.uploaded_source_generation == vulkan_scene_image_generation_ &&
            target.layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            return;
        }

        if (target.layout != VK_IMAGE_LAYOUT_GENERAL) {
            LOG_TIMER("interop.transition_to_GENERAL");
            const std::uint64_t vulkan_ready_value = ++target.timeline_value;
            if (!context.transitionImageLayoutImmediate(target.image.image,
                                                        target.layout,
                                                        VK_IMAGE_LAYOUT_GENERAL,
                                                        VulkanContext::ImmediateTransitionOptions::signalAt(
                                                            {target.semaphore.semaphore, vulkan_ready_value}))) {
                fail_required_interop(std::format("image transition to GENERAL failed: {}", context.lastError()));
            }
            target.layout = VK_IMAGE_LAYOUT_GENERAL;
        }

        {
            LOG_TIMER("interop.copyTensorToSurface");
            assert(target.layout == VK_IMAGE_LAYOUT_GENERAL &&
                   "CUDA surf2Dwrite requires VK_IMAGE_LAYOUT_GENERAL");
            if (!target.interop.wait(target.timeline_value,
                                     vulkan_interop_upload_stream_.stream())) {
                fail_required_interop(std::format("CUDA wait for Vulkan image release failed: {}",
                                                  target.interop.lastError()));
            }
            if (!target.interop.copyTensorToSurface(*vulkan_scene_image_,
                                                    vulkan_interop_upload_stream_.stream())) {
                fail_required_interop(std::format("CUDA copy failed: {}", target.interop.lastError()));
            }
        }
        const std::uint64_t signal_value = ++target.timeline_value;
        {
            LOG_TIMER("interop.cuda_signal");
            if (!target.interop.signal(signal_value, vulkan_interop_upload_stream_.stream())) {
                fail_required_interop(std::format("CUDA signal failed: {}", target.interop.lastError()));
            }
        }
        {
            LOG_TIMER("interop.transition_to_READ_ONLY");
            if (!context.transitionImageLayoutImmediate(target.image.image,
                                                        VK_IMAGE_LAYOUT_GENERAL,
                                                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                                        VulkanContext::ImmediateTransitionOptions::waitOn(
                                                            {target.semaphore.semaphore, signal_value},
                                                            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT))) {
                fail_required_interop(std::format("Vulkan wait for CUDA signal failed: {}", context.lastError()));
            }
        }
        target.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        target.uploaded_source_generation = vulkan_scene_image_generation_;
        ++target.generation;
    }

    void GuiManager::resetVulkanSplitRightInterop() {
        vulkan_split_right_external_image_ = VK_NULL_HANDLE;
        vulkan_split_right_external_image_view_ = VK_NULL_HANDLE;
        vulkan_split_right_external_image_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
        vulkan_split_right_external_image_generation_ = 0;
        if (vulkan_split_right_interop_.empty()) {
            return;
        }
        auto* const window_manager = viewer_ ? viewer_->getWindowManager() : nullptr;
        auto* const vulkan_context = window_manager ? window_manager->getVulkanContext() : nullptr;
        if (vulkan_context) {
            (void)vulkan_context->waitForSubmittedFrames();
        }
        for (auto& target : vulkan_split_right_interop_) {
            if (!target) {
                continue;
            }
            if (vulkan_context) {
                target->destroy(*vulkan_context);
            } else {
                target->interop.reset();
            }
        }
        vulkan_split_right_interop_.clear();
    }

    void GuiManager::prepareVulkanSplitRightInterop(VulkanContext& context) {
        if (vulkan_split_right_interop_disabled_) {
            return;
        }

        const auto fail_required_interop = [this](std::string message) -> void {
            vulkan_split_right_interop_disabled_ = true;
            if (!vulkan_interop_upload_stream_.synchronize()) {
                message += std::format("; CUDA upload drain failed: {}",
                                       vulkan_interop_upload_stream_.lastError());
            }
            resetVulkanSplitRightInterop();
            LOG_ERROR("Required Vulkan/CUDA split-view interop failed: {}", message);
            throw std::runtime_error(std::move(message));
        };

        if (!vulkan_split_right_image_ ||
            !vulkan_split_right_image_->is_valid() ||
            vulkan_split_right_image_->device() != lfs::core::Device::CUDA ||
            vulkan_split_right_image_size_.x <= 0 ||
            vulkan_split_right_image_size_.y <= 0) {
            if (!vulkan_split_right_interop_.empty()) {
                resetVulkanSplitRightInterop();
            }
            vulkan_split_right_external_image_ = VK_NULL_HANDLE;
            vulkan_split_right_external_image_view_ = VK_NULL_HANDLE;
            vulkan_split_right_external_image_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
            vulkan_split_right_external_image_generation_ = 0;
            return;
        }
        if (!vulkan_interop_upload_stream_.valid()) {
            fail_required_interop("non-blocking CUDA upload stream is unavailable");
        }

        const std::size_t frame_slot = context.currentFrameSlot();
        const bool slot_array_resize_needed =
            vulkan_split_right_interop_.size() != context.framesInFlight();
        const auto clear_external_split_right = [this]() {
            vulkan_split_right_external_image_ = VK_NULL_HANDLE;
            vulkan_split_right_external_image_view_ = VK_NULL_HANDLE;
            vulkan_split_right_external_image_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
            vulkan_split_right_external_image_generation_ = 0;
        };
        const bool resize_deferring = shouldDeferVulkanInteropResize();

        if (!slot_array_resize_needed && frame_slot < vulkan_split_right_interop_.size()) {
            const auto& target_ptr_const = vulkan_split_right_interop_[frame_slot];
            const glm::ivec2 target_size = vulkan_split_right_image_size_;
            const bool recreate_needed =
                !target_ptr_const ||
                target_ptr_const->size != target_size ||
                !target_ptr_const->interop.valid();
            if (!recreate_needed &&
                vulkan_split_right_image_generation_ != 0 &&
                target_ptr_const->uploaded_source_generation == vulkan_split_right_image_generation_ &&
                target_ptr_const->layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
                vulkan_split_right_external_image_ = target_ptr_const->image.image;
                vulkan_split_right_external_image_view_ = target_ptr_const->image.view;
                vulkan_split_right_external_image_layout_ = target_ptr_const->layout;
                vulkan_split_right_external_image_generation_ = target_ptr_const->generation;
                return;
            }
            if (resize_deferring && recreate_needed) {
                clear_external_split_right();
                return;
            }
        } else if (resize_deferring) {
            clear_external_split_right();
            return;
        }

        if (!context.waitForCurrentFrameSlot()) {
            fail_required_interop(std::format("frame slot wait failed: {}", context.lastError()));
        }

        if (slot_array_resize_needed) {
            resetVulkanSplitRightInterop();
            vulkan_split_right_interop_.resize(context.framesInFlight());
        }
        if (frame_slot >= vulkan_split_right_interop_.size()) {
            fail_required_interop(std::format("invalid frame slot {}", frame_slot));
        }
        auto& target_ptr = vulkan_split_right_interop_[frame_slot];
        const auto reset_frame_target = [&]() {
            if (target_ptr) {
                target_ptr->destroy(context);
                target_ptr.reset();
            }
        };

        const glm::ivec2 target_size = vulkan_split_right_image_size_;
        const bool recreate =
            !target_ptr ||
            target_ptr->size != target_size ||
            !target_ptr->interop.valid();
        if (recreate) {
            reset_frame_target();
            auto target = std::make_unique<VulkanSceneInteropTarget>();
            const VkExtent2D extent{
                static_cast<std::uint32_t>(target_size.x),
                static_cast<std::uint32_t>(target_size.y),
            };
            if (!context.createExternalImage(extent,
                                             VK_FORMAT_R8G8B8A8_UNORM,
                                             target->image,
                                             "vulkan.gui.interop_image",
                                             std::format("split_right.frame{}", frame_slot)) ||
                !context.createExternalTimelineSemaphore(0, target->semaphore)) {
                const std::string error = std::format("target creation failed: {}", context.lastError());
                if (target->image.image != VK_NULL_HANDLE || target->semaphore.semaphore != VK_NULL_HANDLE) {
                    target->destroy(context);
                }
                fail_required_interop(error);
            }
            const std::uint64_t vulkan_ready_value = ++target->timeline_value;
            if (!context.transitionImageLayoutImmediate(target->image.image,
                                                        VK_IMAGE_LAYOUT_UNDEFINED,
                                                        VK_IMAGE_LAYOUT_GENERAL,
                                                        VulkanContext::ImmediateTransitionOptions::signalAt(
                                                            {target->semaphore.semaphore, vulkan_ready_value}))) {
                const std::string error = std::format("image initialization failed: {}", context.lastError());
                target->destroy(context);
                fail_required_interop(error);
            }
            if (!context.waitForImmediateSubmits()) {
                const std::string error = std::format(
                    "image initialization handoff failed: {}", context.lastError());
                target->destroy(context);
                fail_required_interop(error);
            }

            const auto memory_handle = context.releaseExternalImageNativeHandle(target->image);
            const auto semaphore_handle = context.releaseExternalSemaphoreNativeHandle(target->semaphore);
            lfs::rendering::CudaVulkanExternalImageImport image_import{
                .memory_handle = memory_handle,
                .allocation_size = static_cast<std::size_t>(target->image.allocation_size),
                .extent = {.width = extent.width, .height = extent.height},
                .format = lfs::rendering::CudaVulkanImageFormat::Rgba8Unorm,
                .dedicated_allocation = context.externalMemoryDedicatedAllocationEnabled(),
            };
            lfs::rendering::CudaVulkanExternalSemaphoreImport semaphore_import{
                .semaphore_handle = semaphore_handle,
                .initial_value = 0,
            };
            if (!target->interop.init(image_import, semaphore_import)) {
                const std::string error = std::format("CUDA import failed: {}", target->interop.lastError());
                target->destroy(context);
                fail_required_interop(error);
            }
            target->size = target_size;
            target->layout = VK_IMAGE_LAYOUT_GENERAL;
            target_ptr = std::move(target);
            LOG_INFO("Vulkan/CUDA split-view right-panel interop initialized for slot {}: {}x{}",
                     frame_slot, target_size.x, target_size.y);
        }

        auto& target = *target_ptr;
        if (vulkan_split_right_image_generation_ != 0 &&
            target.uploaded_source_generation == vulkan_split_right_image_generation_ &&
            target.layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            vulkan_split_right_external_image_ = target.image.image;
            vulkan_split_right_external_image_view_ = target.image.view;
            vulkan_split_right_external_image_layout_ = target.layout;
            vulkan_split_right_external_image_generation_ = target.generation;
            return;
        }

        if (target.layout != VK_IMAGE_LAYOUT_GENERAL) {
            const std::uint64_t vulkan_ready_value = ++target.timeline_value;
            if (!context.transitionImageLayoutImmediate(target.image.image,
                                                        target.layout,
                                                        VK_IMAGE_LAYOUT_GENERAL,
                                                        VulkanContext::ImmediateTransitionOptions::signalAt(
                                                            {target.semaphore.semaphore, vulkan_ready_value}))) {
                fail_required_interop(std::format("image transition to GENERAL failed: {}", context.lastError()));
            }
            target.layout = VK_IMAGE_LAYOUT_GENERAL;
        }

        assert(target.layout == VK_IMAGE_LAYOUT_GENERAL &&
               "CUDA surf2Dwrite requires VK_IMAGE_LAYOUT_GENERAL");
        if (!target.interop.wait(target.timeline_value,
                                 vulkan_interop_upload_stream_.stream())) {
            fail_required_interop(std::format("CUDA wait for Vulkan image release failed: {}",
                                              target.interop.lastError()));
        }
        if (!target.interop.copyTensorToSurface(*vulkan_split_right_image_,
                                                vulkan_interop_upload_stream_.stream())) {
            fail_required_interop(std::format("CUDA copy failed: {}", target.interop.lastError()));
        }
        const std::uint64_t signal_value = ++target.timeline_value;
        if (!target.interop.signal(signal_value, vulkan_interop_upload_stream_.stream())) {
            fail_required_interop(std::format("CUDA signal failed: {}", target.interop.lastError()));
        }
        if (!context.transitionImageLayoutImmediate(target.image.image,
                                                    VK_IMAGE_LAYOUT_GENERAL,
                                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                                    VulkanContext::ImmediateTransitionOptions::waitOn(
                                                        {target.semaphore.semaphore, signal_value},
                                                        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT))) {
            fail_required_interop(std::format("Vulkan wait for CUDA signal failed: {}", context.lastError()));
        }
        target.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        target.uploaded_source_generation = vulkan_split_right_image_generation_;
        ++target.generation;
        vulkan_split_right_external_image_ = target.image.image;
        vulkan_split_right_external_image_view_ = target.image.view;
        vulkan_split_right_external_image_layout_ = target.layout;
        vulkan_split_right_external_image_generation_ = target.generation;
    }

    void GuiManager::resetVulkanDepthBlitInterop() {
        vulkan_depth_blit_external_image_ = VK_NULL_HANDLE;
        vulkan_depth_blit_external_image_view_ = VK_NULL_HANDLE;
        vulkan_depth_blit_external_image_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
        vulkan_depth_blit_external_image_generation_ = 0;
        if (vulkan_depth_blit_interop_.empty()) {
            return;
        }
        auto* const window_manager = viewer_ ? viewer_->getWindowManager() : nullptr;
        auto* const vulkan_context = window_manager ? window_manager->getVulkanContext() : nullptr;
        if (vulkan_context) {
            (void)vulkan_context->waitForSubmittedFrames();
        }
        for (auto& target : vulkan_depth_blit_interop_) {
            if (!target) {
                continue;
            }
            if (vulkan_context) {
                target->destroy(*vulkan_context);
            } else {
                target->interop.reset();
            }
        }
        vulkan_depth_blit_interop_.clear();
    }

    void GuiManager::prepareVulkanDepthBlitInterop(VulkanContext& context) {
        if (vulkan_depth_blit_interop_disabled_) {
            return;
        }

        const auto fail_required_interop = [this](std::string message) -> void {
            vulkan_depth_blit_interop_disabled_ = true;
            if (!vulkan_interop_upload_stream_.synchronize()) {
                message += std::format("; CUDA upload drain failed: {}",
                                       vulkan_interop_upload_stream_.lastError());
            }
            resetVulkanDepthBlitInterop();
            LOG_ERROR("Required Vulkan/CUDA depth-blit interop failed: {}", message);
            throw std::runtime_error(std::move(message));
        };

        if (!vulkan_depth_blit_image_ ||
            !vulkan_depth_blit_image_->is_valid() ||
            vulkan_depth_blit_image_->device() != lfs::core::Device::CUDA ||
            vulkan_depth_blit_image_size_.x <= 0 ||
            vulkan_depth_blit_image_size_.y <= 0) {
            if (!vulkan_depth_blit_interop_.empty()) {
                resetVulkanDepthBlitInterop();
            }
            vulkan_depth_blit_external_image_ = VK_NULL_HANDLE;
            vulkan_depth_blit_external_image_view_ = VK_NULL_HANDLE;
            vulkan_depth_blit_external_image_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
            vulkan_depth_blit_external_image_generation_ = 0;
            return;
        }
        if (!vulkan_interop_upload_stream_.valid()) {
            fail_required_interop("non-blocking CUDA upload stream is unavailable");
        }

        const std::size_t frame_slot = context.currentFrameSlot();
        const bool slot_array_resize_needed =
            vulkan_depth_blit_interop_.size() != context.framesInFlight();
        const auto clear_external_depth_blit = [this]() {
            vulkan_depth_blit_external_image_ = VK_NULL_HANDLE;
            vulkan_depth_blit_external_image_view_ = VK_NULL_HANDLE;
            vulkan_depth_blit_external_image_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
            vulkan_depth_blit_external_image_generation_ = 0;
        };
        const bool resize_deferring = shouldDeferVulkanInteropResize();

        if (!slot_array_resize_needed && frame_slot < vulkan_depth_blit_interop_.size()) {
            const auto& target_ptr_const = vulkan_depth_blit_interop_[frame_slot];
            const glm::ivec2 target_size = vulkan_depth_blit_image_size_;
            const bool recreate_needed =
                !target_ptr_const ||
                target_ptr_const->size != target_size ||
                !target_ptr_const->interop.valid();
            if (!recreate_needed &&
                vulkan_depth_blit_image_generation_ != 0 &&
                target_ptr_const->uploaded_source_generation == vulkan_depth_blit_image_generation_ &&
                target_ptr_const->layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
                vulkan_depth_blit_external_image_ = target_ptr_const->image.image;
                vulkan_depth_blit_external_image_view_ = target_ptr_const->image.view;
                vulkan_depth_blit_external_image_layout_ = target_ptr_const->layout;
                vulkan_depth_blit_external_image_generation_ = target_ptr_const->generation;
                return;
            }
            if (resize_deferring && recreate_needed) {
                clear_external_depth_blit();
                return;
            }
        } else if (resize_deferring) {
            clear_external_depth_blit();
            return;
        }

        if (!context.waitForCurrentFrameSlot()) {
            fail_required_interop(std::format("frame slot wait failed: {}", context.lastError()));
        }

        if (slot_array_resize_needed) {
            resetVulkanDepthBlitInterop();
            vulkan_depth_blit_interop_.resize(context.framesInFlight());
        }
        if (frame_slot >= vulkan_depth_blit_interop_.size()) {
            fail_required_interop(std::format("invalid frame slot {}", frame_slot));
        }
        auto& target_ptr = vulkan_depth_blit_interop_[frame_slot];
        const auto reset_frame_target = [&]() {
            if (target_ptr) {
                target_ptr->destroy(context);
                target_ptr.reset();
            }
        };

        const glm::ivec2 target_size = vulkan_depth_blit_image_size_;
        const bool recreate =
            !target_ptr ||
            target_ptr->size != target_size ||
            !target_ptr->interop.valid();
        if (recreate) {
            reset_frame_target();
            auto target = std::make_unique<VulkanSceneInteropTarget>();
            const VkExtent2D extent{
                static_cast<std::uint32_t>(target_size.x),
                static_cast<std::uint32_t>(target_size.y),
            };
            if (!context.createExternalImage(extent,
                                             VK_FORMAT_R32_SFLOAT,
                                             target->image,
                                             "vulkan.gui.interop_image",
                                             std::format("depth_blit.frame{}", frame_slot)) ||
                !context.createExternalTimelineSemaphore(0, target->semaphore)) {
                const std::string error = std::format("target creation failed: {}", context.lastError());
                if (target->image.image != VK_NULL_HANDLE || target->semaphore.semaphore != VK_NULL_HANDLE) {
                    target->destroy(context);
                }
                fail_required_interop(error);
            }
            const std::uint64_t vulkan_ready_value = ++target->timeline_value;
            if (!context.transitionImageLayoutImmediate(target->image.image,
                                                        VK_IMAGE_LAYOUT_UNDEFINED,
                                                        VK_IMAGE_LAYOUT_GENERAL,
                                                        VulkanContext::ImmediateTransitionOptions::signalAt(
                                                            {target->semaphore.semaphore, vulkan_ready_value}))) {
                const std::string error = std::format("image initialization failed: {}", context.lastError());
                target->destroy(context);
                fail_required_interop(error);
            }
            if (!context.waitForImmediateSubmits()) {
                const std::string error = std::format(
                    "image initialization handoff failed: {}", context.lastError());
                target->destroy(context);
                fail_required_interop(error);
            }

            const auto memory_handle = context.releaseExternalImageNativeHandle(target->image);
            const auto semaphore_handle = context.releaseExternalSemaphoreNativeHandle(target->semaphore);
            lfs::rendering::CudaVulkanExternalImageImport image_import{
                .memory_handle = memory_handle,
                .allocation_size = static_cast<std::size_t>(target->image.allocation_size),
                .extent = {.width = extent.width, .height = extent.height},
                .format = lfs::rendering::CudaVulkanImageFormat::R32Sfloat,
                .dedicated_allocation = context.externalMemoryDedicatedAllocationEnabled(),
            };
            lfs::rendering::CudaVulkanExternalSemaphoreImport semaphore_import{
                .semaphore_handle = semaphore_handle,
                .initial_value = 0,
            };
            if (!target->interop.init(image_import, semaphore_import)) {
                const std::string error = std::format("CUDA import failed: {}", target->interop.lastError());
                target->destroy(context);
                fail_required_interop(error);
            }
            target->size = target_size;
            target->layout = VK_IMAGE_LAYOUT_GENERAL;
            target_ptr = std::move(target);
            LOG_INFO("Vulkan/CUDA depth-blit interop initialized for slot {}: {}x{}",
                     frame_slot, target_size.x, target_size.y);
        }

        auto& target = *target_ptr;
        if (vulkan_depth_blit_image_generation_ != 0 &&
            target.uploaded_source_generation == vulkan_depth_blit_image_generation_ &&
            target.layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            vulkan_depth_blit_external_image_ = target.image.image;
            vulkan_depth_blit_external_image_view_ = target.image.view;
            vulkan_depth_blit_external_image_layout_ = target.layout;
            vulkan_depth_blit_external_image_generation_ = target.generation;
            return;
        }

        if (target.layout != VK_IMAGE_LAYOUT_GENERAL) {
            const std::uint64_t vulkan_ready_value = ++target.timeline_value;
            if (!context.transitionImageLayoutImmediate(target.image.image,
                                                        target.layout,
                                                        VK_IMAGE_LAYOUT_GENERAL,
                                                        VulkanContext::ImmediateTransitionOptions::signalAt(
                                                            {target.semaphore.semaphore, vulkan_ready_value}))) {
                fail_required_interop(std::format("image transition to GENERAL failed: {}", context.lastError()));
            }
            target.layout = VK_IMAGE_LAYOUT_GENERAL;
        }

        assert(target.layout == VK_IMAGE_LAYOUT_GENERAL &&
               "CUDA surf2Dwrite requires VK_IMAGE_LAYOUT_GENERAL");
        if (!target.interop.wait(target.timeline_value,
                                 vulkan_interop_upload_stream_.stream())) {
            fail_required_interop(std::format("CUDA wait for Vulkan image release failed: {}",
                                              target.interop.lastError()));
        }
        if (!target.interop.copyTensorToSurface(*vulkan_depth_blit_image_,
                                                vulkan_interop_upload_stream_.stream())) {
            fail_required_interop(std::format("CUDA copy failed: {}", target.interop.lastError()));
        }
        const std::uint64_t signal_value = ++target.timeline_value;
        if (!target.interop.signal(signal_value, vulkan_interop_upload_stream_.stream())) {
            fail_required_interop(std::format("CUDA signal failed: {}", target.interop.lastError()));
        }
        if (!context.transitionImageLayoutImmediate(target.image.image,
                                                    VK_IMAGE_LAYOUT_GENERAL,
                                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                                    VulkanContext::ImmediateTransitionOptions::waitOn(
                                                        {target.semaphore.semaphore, signal_value},
                                                        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT))) {
            fail_required_interop(std::format("Vulkan wait for CUDA signal failed: {}", context.lastError()));
        }
        target.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        target.uploaded_source_generation = vulkan_depth_blit_image_generation_;
        ++target.generation;
        vulkan_depth_blit_external_image_ = target.image.image;
        vulkan_depth_blit_external_image_view_ = target.image.view;
        vulkan_depth_blit_external_image_layout_ = target.layout;
        vulkan_depth_blit_external_image_generation_ = target.generation;
    }

    VulkanViewportPassParams GuiManager::buildVulkanViewportParams(const VkExtent2D extent,
                                                                   const std::size_t frame_slot) const {
        const bool has_viewport_layout =
            viewport_layout_.size.x > 0.0f && viewport_layout_.size.y > 0.0f;
        const bool export_locked = isViewportExportLocked();

        VulkanViewportPassParams params{};
        params.frame_slot = frame_slot;
        params.viewport_pos = has_viewport_layout ? viewport_layout_.pos : glm::vec2(0.0f, 0.0f);
        params.viewport_size = has_viewport_layout
                                   ? viewport_layout_.size
                                   : glm::vec2(static_cast<float>(extent.width), static_cast<float>(extent.height));
        params.framebuffer_scale = {
            ImGui::GetIO().DisplayFramebufferScale.x,
            ImGui::GetIO().DisplayFramebufferScale.y,
        };

        params.scene_image = vulkan_scene_image_;
        params.scene_image_size = vulkan_scene_image_size_;
        params.scene_image_flip_y = vulkan_scene_image_flip_y_;
        if (vulkan_external_scene_image_ != VK_NULL_HANDLE &&
            vulkan_external_scene_image_view_ != VK_NULL_HANDLE &&
            vulkan_external_scene_image_size_.x > 0 &&
            vulkan_external_scene_image_size_.y > 0) {
            params.scene_image_size = vulkan_external_scene_image_size_;
            params.scene_image_flip_y = vulkan_external_scene_image_flip_y_;
            params.external_scene_image = vulkan_external_scene_image_;
            params.external_scene_image_view = vulkan_external_scene_image_view_;
            params.external_scene_image_layout = vulkan_external_scene_image_layout_;
            params.external_scene_image_generation = vulkan_external_scene_image_generation_;
        }
        const auto bind_cached_interop_slot = [&](const std::size_t slot) -> bool {
            if (slot >= vulkan_scene_interop_.size()) {
                return false;
            }
            const auto& target = vulkan_scene_interop_[slot];
            if (!target ||
                !target->interop.valid() ||
                target->layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL ||
                target->size != params.scene_image_size ||
                vulkan_scene_image_generation_ == 0 ||
                target->uploaded_source_generation != vulkan_scene_image_generation_) {
                return false;
            }

            params.external_scene_image = target->image.image;
            params.external_scene_image_view = target->image.view;
            params.external_scene_image_layout = target->layout;
            params.external_scene_image_generation = target->generation;
            return true;
        };
        if (params.external_scene_image == VK_NULL_HANDLE) {
            const bool bound_current_slot = bind_cached_interop_slot(frame_slot);
            if (!bound_current_slot && export_locked) {
                // Export mode freezes the viewport and skips new CUDA/Vulkan interop uploads.
                // Reuse any already-prepared slot so multi-buffered frames keep the same image.
                for (std::size_t slot = 0; slot < vulkan_scene_interop_.size(); ++slot) {
                    if (slot != frame_slot && bind_cached_interop_slot(slot)) {
                        break;
                    }
                }
            }
            params.preserve_scene_image_binding =
                params.external_scene_image == VK_NULL_HANDLE &&
                params.scene_image &&
                shouldDeferVulkanInteropResize();
        }

        if (auto* const rendering_manager = viewer_ ? viewer_->getRenderingManager() : nullptr) {
            const auto settings = rendering_manager->getSettings();
            params.background_color = settings.background_color;
            params.grid_enabled =
                settings.show_grid &&
                !splitViewUsesComparisonPanels(settings.split_view_mode) &&
                !settings.equirectangular &&
                settings.grid_opacity > 0.0f;
            params.grid_plane = std::clamp(settings.grid_plane, 0, 2);
            params.grid_opacity = std::clamp(settings.grid_opacity, 0.0f, 1.0f);

            if (params.grid_enabled && viewer_) {
                const auto panels = collectVulkanGuidePanels(*viewer_, viewport_layout_, *rendering_manager);
                params.grid_overlays.reserve(panels.size());
                for (const auto& panel : panels) {
                    if (!panel.valid()) {
                        continue;
                    }
                    const glm::mat4 view =
                        lfs::rendering::makeViewMatrix(panel.viewport->getRotationMatrix(),
                                                       panel.viewport->getTranslation());
                    const glm::mat4 proj = lfs::rendering::createProjectionMatrixFromFocal(
                        panel.render_size,
                        settings.focal_length_mm,
                        settings.orthographic,
                        settings.ortho_scale,
                        lfs::rendering::DEFAULT_NEAR_PLANE,
                        settings.depth_clip_enabled ? settings.depth_clip_far : lfs::rendering::DEFAULT_FAR_PLANE);
                    VulkanViewportGridOverlay grid{};
                    grid.viewport_pos = panel.pos;
                    grid.viewport_size = panel.size;
                    grid.render_size = panel.render_size;
                    grid.view = view;
                    grid.projection = proj;
                    grid.view_projection = proj * view;
                    grid.view_position = panel.viewport->getTranslation();
                    grid.plane = rendering_manager->getGridPlaneForPanel(panel.panel);
                    grid.opacity = params.grid_opacity;
                    grid.orthographic = settings.orthographic;
                    params.grid_overlays.push_back(grid);
                }
                if (!params.grid_overlays.empty()) {
                    const auto& first_grid = params.grid_overlays.front();
                    params.grid_plane = first_grid.plane;
                    params.grid_view = first_grid.view;
                    params.grid_projection = first_grid.projection;
                    params.grid_view_projection = first_grid.view_projection;
                    params.grid_view_position = first_grid.view_position;
                    params.grid_orthographic = first_grid.orthographic;
                }
            }

            if (viewer_) {
                SceneManager* const scene_manager = viewer_->getSceneManager();
                std::optional<SceneRenderState> overlay_scene_state;
                if (scene_manager && (settings.show_crop_box || settings.show_ellipsoid)) {
                    overlay_scene_state = scene_manager->buildRenderState();
                }
                const GizmoState gizmo_state = rendering_manager->getGizmoState();
                appendVulkanSceneGuideOverlays(params,
                                               *viewer_,
                                               viewport_layout_,
                                               settings,
                                               *rendering_manager,
                                               scene_manager,
                                               overlay_scene_state ? &*overlay_scene_state : nullptr,
                                               gizmo_state);
                appendVulkanViewportGizmoOverlay(params,
                                                 *viewer_,
                                                 viewport_layout_,
                                                 *rendering_manager,
                                                 current_ui_scale_,
                                                 gizmo_manager_.isViewportGizmoDragging());
            }
        }

        appendLineRendererCommandOverlays(params);

        if (auto* const rendering_manager = viewer_ ? viewer_->getRenderingManager() : nullptr) {
            appendScreenOverlayCommandOverlays(params, rendering_manager->getScreenOverlayRenderer());

            // Pull GPU mesh / environment frame populated by renderVulkanFrame.
            // vulkan_viewport_pass rasterizes these on the GPU instead of the
            // auxiliary CPU mesh / environment paths.
            auto mesh_frame = rendering_manager->getVulkanMeshFrame();
            params.mesh_view_projection = mesh_frame.view_projection;
            params.mesh_camera_position = mesh_frame.camera_position;
            params.mesh_items = std::move(mesh_frame.items);
            params.mesh_panels = std::move(mesh_frame.panels);
            params.environment = std::move(mesh_frame.environment);
            params.depth_blit = std::move(mesh_frame.depth_blit);
            if (vulkan_depth_blit_external_image_view_ != VK_NULL_HANDLE) {
                params.depth_blit.external_image_view = vulkan_depth_blit_external_image_view_;
                params.depth_blit.external_image_generation = vulkan_depth_blit_external_image_generation_;
            }
            params.split_view = std::move(mesh_frame.split_view);
            // Stitch in CUDA/Vulkan interop views: left reuses the existing scene
            // interop slot; right has its own parallel slot. When set, the split-view
            // pass binds these directly and skips the CPU staging upload.
            if (params.split_view.enabled) {
                if (params.external_scene_image_view != VK_NULL_HANDLE) {
                    params.split_view.left.external_image_view = params.external_scene_image_view;
                    params.split_view.left.external_image_generation = params.external_scene_image_generation;
                }
                if (vulkan_split_right_external_image_view_ != VK_NULL_HANDLE) {
                    params.split_view.right.external_image_view = vulkan_split_right_external_image_view_;
                    params.split_view.right.external_image_generation = vulkan_split_right_external_image_generation_;
                }
            }
        }

        // Sample mouse pos with SDL_GetGlobalMouseState here, after all panel/tool overlay
        // queueing and just before the GPU command buffer is recorded. Global polling hits
        // the OS directly. ImGui::GetMousePos and SDL_GetMouseState both return the cached
        // NewFrame-aligned value, so without this the cursor ring lags an extra event-pump
        // behind the hardware pointer.
        if (viewer_ && !ui_hidden_ && !guiFocusState().want_capture_mouse) {
            if (auto* const sel = viewer_->getSelectionTool(); sel && sel->isEnabled()) {
                SDL_Window* const window = viewer_->getWindow();
                int win_x = 0;
                int win_y = 0;
                if (window) {
                    SDL_GetWindowPosition(window, &win_x, &win_y);
                }
                float gx = 0.0f;
                float gy = 0.0f;
                SDL_GetGlobalMouseState(&gx, &gy);
                const glm::vec2 mp{gx - static_cast<float>(win_x), gy - static_cast<float>(win_y)};
                if (isPositionInViewport(mp.x, mp.y)) {
                    auto* const rm = viewer_->getRenderingManager();
                    const auto mode = rm ? rm->getSelectionPreviewMode()
                                         : lfs::vis::SelectionPreviewMode::Centers;
                    const auto& palette = lfs::vis::theme().palette;
                    const auto& base = (SDL_GetModState() & SDL_KMOD_CTRL) ? palette.error : palette.primary;
                    const glm::vec4 color{base.x * 0.85f, base.y * 0.85f, base.z * 0.85f, 0.85f};
                    auto line = [&](const glm::vec2 a, const glm::vec2 b, const float thickness = 2.0f) {
                        appendShapeOverlayLine(params.ui_shape_overlay_triangles, params, a, b, color, thickness);
                    };
                    auto rect = [&](const glm::vec2 mn, const glm::vec2 mx, const float thickness = 2.0f) {
                        line({mn.x, mn.y}, {mx.x, mn.y}, thickness);
                        line({mx.x, mn.y}, {mx.x, mx.y}, thickness);
                        line({mx.x, mx.y}, {mn.x, mx.y}, thickness);
                        line({mn.x, mx.y}, {mn.x, mn.y}, thickness);
                    };
                    auto polyline = [&](const auto& pts, const bool closed, const float thickness = 2.0f) {
                        for (std::size_t i = 1; i < pts.size(); ++i) {
                            line(pts[i - 1], pts[i], thickness);
                        }
                        if (closed && pts.size() > 1) {
                            line(pts.back(), pts.front(), thickness);
                        }
                    };

                    switch (mode) {
                    case lfs::vis::SelectionPreviewMode::Centers:
                        appendShapeOverlayCircleOutline(params.ui_shape_overlay_triangles, params,
                                                        mp, sel->getBrushRadius(), color, 2.0f);
                        appendShapeOverlayCircle(params.ui_shape_overlay_triangles, params,
                                                 mp, 3.0f, color);
                        break;
                    case lfs::vis::SelectionPreviewMode::Rings:
                        appendShapeOverlayCircleOutline(params.ui_shape_overlay_triangles, params,
                                                        mp, 10.0f, color, 2.0f);
                        line({mp.x - 14.0f, mp.y}, {mp.x - 5.0f, mp.y}, 1.5f);
                        line({mp.x + 5.0f, mp.y}, {mp.x + 14.0f, mp.y}, 1.5f);
                        line({mp.x, mp.y - 14.0f}, {mp.x, mp.y - 5.0f}, 1.5f);
                        line({mp.x, mp.y + 5.0f}, {mp.x, mp.y + 14.0f}, 1.5f);
                        break;
                    case lfs::vis::SelectionPreviewMode::Rectangle:
                        rect({mp.x - 12.0f, mp.y - 9.0f}, {mp.x + 12.0f, mp.y + 9.0f});
                        break;
                    case lfs::vis::SelectionPreviewMode::Box:
                        rect({mp.x - 11.0f, mp.y - 11.0f}, {mp.x + 11.0f, mp.y + 11.0f});
                        line({mp.x - 11.0f, mp.y - 11.0f}, {mp.x - 5.0f, mp.y - 17.0f}, 1.5f);
                        line({mp.x + 11.0f, mp.y - 11.0f}, {mp.x + 17.0f, mp.y - 17.0f}, 1.5f);
                        line({mp.x + 11.0f, mp.y + 11.0f}, {mp.x + 17.0f, mp.y + 5.0f}, 1.5f);
                        line({mp.x - 5.0f, mp.y - 17.0f}, {mp.x + 17.0f, mp.y - 17.0f}, 1.5f);
                        line({mp.x + 17.0f, mp.y - 17.0f}, {mp.x + 17.0f, mp.y + 5.0f}, 1.5f);
                        break;
                    case lfs::vis::SelectionPreviewMode::Sphere:
                        appendShapeOverlayCircleOutline(params.ui_shape_overlay_triangles, params,
                                                        mp, 12.0f, color, 2.0f);
                        appendShapeOverlayCircleOutline(params.ui_shape_overlay_triangles, params,
                                                        mp, 7.0f, color, 1.5f);
                        line({mp.x - 12.0f, mp.y}, {mp.x + 12.0f, mp.y}, 1.5f);
                        break;
                    case lfs::vis::SelectionPreviewMode::Polygon: {
                        const std::array<glm::vec2, 3> pts{{
                            {mp.x, mp.y - 13.0f},
                            {mp.x + 12.0f, mp.y + 8.0f},
                            {mp.x - 12.0f, mp.y + 8.0f},
                        }};
                        polyline(pts, true);
                        break;
                    }
                    case lfs::vis::SelectionPreviewMode::Lasso: {
                        const std::array<glm::vec2, 6> pts{{
                            {mp.x - 13.0f, mp.y - 2.0f},
                            {mp.x - 8.0f, mp.y - 11.0f},
                            {mp.x + 5.0f, mp.y - 12.0f},
                            {mp.x + 13.0f, mp.y - 3.0f},
                            {mp.x + 9.0f, mp.y + 9.0f},
                            {mp.x - 7.0f, mp.y + 11.0f},
                        }};
                        polyline(pts, true);
                        break;
                    }
                    case lfs::vis::SelectionPreviewMode::Color:
                        appendShapeOverlayCircleOutline(params.ui_shape_overlay_triangles, params,
                                                        mp, 8.0f, color, 2.0f);
                        line({mp.x - 10.0f, mp.y + 10.0f}, {mp.x + 10.0f, mp.y - 10.0f});
                        break;
                    }
                }
            }
        }

        const auto& vignette = lfs::vis::theme().vignette;
        params.vignette_enabled = vignette.enabled && vignette.intensity > 0.0f;
        params.vignette_intensity = vignette.intensity;
        params.vignette_radius = vignette.radius;
        params.vignette_softness = vignette.softness;

        if (export_locked) {
            appendViewportDimOverlay(params);
        }

        return params;
    }

    void GuiManager::recordVulkanViewport(VkCommandBuffer command_buffer,
                                          VkExtent2D extent,
                                          const VulkanViewportPassParams& params) {
        if (!vulkan_viewport_pass_ || command_buffer == VK_NULL_HANDLE ||
            extent.width == 0 || extent.height == 0) {
            return;
        }
        vulkan_viewport_pass_->record(command_buffer, extent, params);
    }

    bool GuiManager::drainVulkanFramesForInteractiveTransition(
        WindowManager& window_manager,
        const char* const transition_name) {
        auto* const vulkan_context = window_manager.getVulkanContext();
        if (!vulkan_context) {
            return true;
        }

        const auto drain_start = std::chrono::steady_clock::now();
        if (vulkan_context->waitForSubmittedFrames()) {
            LOG_DEBUG("Vulkan frame drain before {} transition complete: elapsed_ms={:.1f}",
                      transition_name,
                      std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - drain_start)
                          .count());
            return true;
        }

        LOG_WARN("Skipping {} transition because Vulkan frame drain failed after {:.1f} ms: {}",
                 transition_name,
                 std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - drain_start)
                     .count(),
                 vulkan_context->lastError());
        return false;
    }

    void GuiManager::applyInteractiveTransitionCooldown(
        std::chrono::steady_clock::time_point& next_allowed_at,
        const std::chrono::steady_clock::time_point now,
        const bool training_active) {
        next_allowed_at =
            now + (training_active ? kInteractiveTrainingToggleMinInterval
                                   : kInteractiveIdleToggleMinInterval);
        interactive_transition_guard_until_ =
            std::max(interactive_transition_guard_until_,
                     now + kInteractiveTransitionGuardDuration);
    }

    void GuiManager::queueUiVisibilityToggle() {
        if (!viewer_) {
            return;
        }

        if (viewer_->isOnViewerThread()) {
            requestUiVisibilityToggle();
            return;
        }

        const bool posted = viewer_->postWork(VisualizerImpl::WorkItem{
            .run = [this] {
                requestUiVisibilityToggle();
            },
            .cancel = {},
        });
        if (!posted) {
            LOG_DEBUG("Dropping UI visibility transition request because the viewer is shutting down");
        }
    }

    void GuiManager::requestUiVisibilityToggle() {
        auto* const wm = viewer_ ? viewer_->getWindowManager() : nullptr;
        if (!wm) {
            return;
        }

        const auto now = std::chrono::steady_clock::now();
        const auto cooldown_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                ui_toggle_next_allowed_at_ > now ? ui_toggle_next_allowed_at_ - now
                                                 : std::chrono::steady_clock::duration::zero())
                .count();
        LOG_DEBUG("Request UI visibility transition: pending={}, ui_hidden={}, cooldown_remaining_ms={}",
                  ui_toggle_pending_,
                  ui_hidden_,
                  cooldown_ms);
        if (ui_toggle_pending_) {
            wm->wakeEventLoop();
            return;
        }

        ui_toggle_pending_ = true;
        wm->wakeEventLoop();
    }

    void GuiManager::updateUiVisibilityTransition() {
        if (!ui_toggle_pending_) {
            return;
        }

        auto* const wm = viewer_ ? viewer_->getWindowManager() : nullptr;
        if (!wm) {
            ui_toggle_pending_ = false;
            return;
        }

        const auto now = std::chrono::steady_clock::now();
        if (now < ui_toggle_next_allowed_at_) {
            return;
        }

        ui_toggle_pending_ = false;
        beginInteractiveTransitionGuard();
        if (!drainVulkanFramesForInteractiveTransition(*wm, "UI visibility")) {
            ui_toggle_pending_ = true;
            ui_toggle_next_allowed_at_ = now + kInteractiveTrainingToggleMinInterval;
            LOG_WARN("UI visibility transition deferred after Vulkan drain failure: next_retry_ms={}, guard_kept_active=true, guard_remaining_ms={}",
                     kInteractiveTrainingToggleMinInterval.count(),
                     std::chrono::duration_cast<std::chrono::milliseconds>(
                         interactive_transition_guard_until_ > std::chrono::steady_clock::now()
                             ? interactive_transition_guard_until_ - std::chrono::steady_clock::now()
                             : std::chrono::steady_clock::duration::zero())
                         .count());
            return;
        }

        auto* const trainer = viewer_ ? viewer_->getTrainerManager() : nullptr;
        const bool training_active = trainer && trainer->isRunning();
        ui_hidden_ = !ui_hidden_;

        applyInteractiveTransitionCooldown(ui_toggle_next_allowed_at_,
                                           std::chrono::steady_clock::now(),
                                           training_active);
        LOG_DEBUG("UI visibility transition applied: ui_hidden_after={}, training_active={}, next_allowed_in_ms={}",
                  ui_hidden_,
                  training_active,
                  (training_active ? kInteractiveTrainingToggleMinInterval
                                   : kInteractiveIdleToggleMinInterval)
                      .count());
    }

    void GuiManager::queueFullscreenToggle() {
        if (!viewer_) {
            return;
        }

        if (viewer_->isOnViewerThread()) {
            requestFullscreenToggle();
            return;
        }

        const bool posted = viewer_->postWork(VisualizerImpl::WorkItem{
            .run = [this] {
                requestFullscreenToggle();
            },
            .cancel = {},
        });
        if (!posted) {
            LOG_DEBUG("Dropping fullscreen transition request because the viewer is shutting down");
        }
    }

    void GuiManager::requestFullscreenToggle() {
        auto* const wm = viewer_ ? viewer_->getWindowManager() : nullptr;
        if (!wm) {
            return;
        }

        const auto now = std::chrono::steady_clock::now();
        const auto cooldown_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                fullscreen_toggle_next_allowed_at_ > now ? fullscreen_toggle_next_allowed_at_ - now
                                                         : std::chrono::steady_clock::duration::zero())
                .count();
        LOG_DEBUG("Request fullscreen transition: pending={}, current={}, target={}, cooldown_remaining_ms={}",
                  fullscreen_toggle_pending_,
                  wm->isFullscreen(),
                  !wm->isFullscreen(),
                  cooldown_ms);
        if (fullscreen_toggle_pending_) {
            wm->wakeEventLoop();
            return;
        }

        fullscreen_target_state_ = !wm->isFullscreen();
        fullscreen_toggle_pending_ = true;
        wm->wakeEventLoop();
    }

    void GuiManager::updateFullscreenTransition() {
        if (!fullscreen_toggle_pending_) {
            return;
        }

        auto* const wm = viewer_ ? viewer_->getWindowManager() : nullptr;
        if (!wm) {
            fullscreen_toggle_pending_ = false;
            return;
        }

        const auto now = std::chrono::steady_clock::now();
        if (now < fullscreen_toggle_next_allowed_at_) {
            return;
        }

        fullscreen_toggle_pending_ = false;
        if (wm->isFullscreen() == fullscreen_target_state_) {
            LOG_DEBUG("Fullscreen transition already satisfied: target={}", fullscreen_target_state_);
            return;
        }

        beginInteractiveTransitionGuard();
        if (!drainVulkanFramesForInteractiveTransition(*wm, "fullscreen")) {
            fullscreen_toggle_pending_ = true;
            fullscreen_toggle_next_allowed_at_ = now + kInteractiveTrainingToggleMinInterval;
            LOG_WARN("Fullscreen transition deferred after Vulkan drain failure: target={}, next_retry_ms={}, guard_kept_active=true, guard_remaining_ms={}",
                     fullscreen_target_state_,
                     kInteractiveTrainingToggleMinInterval.count(),
                     std::chrono::duration_cast<std::chrono::milliseconds>(
                         interactive_transition_guard_until_ > std::chrono::steady_clock::now()
                             ? interactive_transition_guard_until_ - std::chrono::steady_clock::now()
                             : std::chrono::steady_clock::duration::zero())
                         .count());
            return;
        }

        auto* const trainer = viewer_ ? viewer_->getTrainerManager() : nullptr;
        const bool training_active = trainer && trainer->isRunning();
        wm->setFullscreen(fullscreen_target_state_);

        applyInteractiveTransitionCooldown(fullscreen_toggle_next_allowed_at_,
                                           std::chrono::steady_clock::now(),
                                           training_active);
        LOG_DEBUG("Fullscreen transition applied: target={}, training_active={}, next_allowed_in_ms={}",
                  fullscreen_target_state_,
                  training_active,
                  (training_active ? kInteractiveTrainingToggleMinInterval
                                   : kInteractiveIdleToggleMinInterval)
                      .count());
    }

    void GuiManager::beginInteractiveTransitionGuard() {
        const auto now = std::chrono::steady_clock::now();
        interactive_transition_guard_until_ =
            now + kInteractiveTransitionGuardDuration;

        if (interactive_transition_resume_training_) {
            return;
        }

        auto* const trainer = viewer_ ? viewer_->getTrainerManager() : nullptr;
        if (!trainer || !trainer->isRunning()) {
            return;
        }

        const auto pause_result = trainer->pauseTrainingTemporaryAndWait(kInteractiveTrainingPauseWait);
        interactive_transition_resume_training_ = pause_result.resume_required;
        if (!pause_result.synchronized) {
            LOG_WARN("Vulkan UI transition proceeding without a synchronized training pause");
        }
    }

    void GuiManager::updateInteractiveTransitionGuard() {
        if (!interactive_transition_resume_training_) {
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        if (now < interactive_transition_guard_until_) {
            return;
        }

        endInteractiveTransitionGuard();
    }

    void GuiManager::endInteractiveTransitionGuard() {
        if (!interactive_transition_resume_training_) {
            return;
        }

        interactive_transition_resume_training_ = false;
        auto* const trainer = viewer_ ? viewer_->getTrainerManager() : nullptr;
        if (trainer && trainer->isRunning()) {
            trainer->resumeTrainingTemporary();
            LOG_TRACE("Training resumed after Vulkan UI transition guard");
        }
    }

    void GuiManager::updateInteractiveTransitions() {
        updateFullscreenTransition();
        updateUiVisibilityTransition();
        updateInteractiveTransitionGuard();
    }

    bool GuiManager::isInteractiveTransitionSettling() const {
        return std::chrono::steady_clock::now() < interactive_transition_guard_until_;
    }

    bool GuiManager::shouldUseCachedImGuiResizeFrame(
        const WindowManager* const window_manager,
        const VulkanContext* const vulkan_context) const {
#if defined(__linux__)
        // The cached platform frame is a Windows manual-resize optimization. On
        // X11, SDL's live platform frame keeps ImGui in sync with exposed pixels.
        (void)window_manager;
        (void)vulkan_context;
        return false;
#else
        if (!window_manager || !vulkan_context) {
            return false;
        }

        const glm::ivec2 window_size = window_manager->getWindowSize();
        const glm::ivec2 framebuffer_size = window_manager->getFramebufferSize();
        const bool active_window_resize =
            vulkan_context->hasPendingSwapchainResize() ||
            window_manager->hasRecentWindowSizeChange(kCachedImGuiResizeFrameWindow);
        return active_window_resize &&
               window_size.x > 0 && window_size.y > 0 &&
               framebuffer_size.x > 0 && framebuffer_size.y > 0;
#endif
    }

    void GuiManager::beginImGuiPlatformFrame(WindowManager* const window_manager,
                                             VulkanContext* const vulkan_context) {
        const auto now = std::chrono::steady_clock::now();
        const auto update_delta_time = [&]() -> float {
            if (last_imgui_platform_frame_time_ == std::chrono::steady_clock::time_point{}) {
                last_imgui_platform_frame_time_ = now;
                return 1.0f / 60.0f;
            }

            const auto elapsed = std::chrono::duration<float>(now - last_imgui_platform_frame_time_).count();
            last_imgui_platform_frame_time_ = now;
            return std::clamp(elapsed, 1.0f / 240.0f, 1.0f / 15.0f);
        };

        if (shouldUseCachedImGuiResizeFrame(window_manager, vulkan_context)) {
            ImGuiIO& io = ImGui::GetIO();
            const glm::ivec2 window_size = window_manager->getWindowSize();
            const glm::ivec2 framebuffer_size = window_manager->getFramebufferSize();
            io.DisplaySize = ImVec2(static_cast<float>(window_size.x),
                                    static_cast<float>(window_size.y));
            io.DisplayFramebufferScale = ImVec2(
                static_cast<float>(framebuffer_size.x) / static_cast<float>(window_size.x),
                static_cast<float>(framebuffer_size.y) / static_cast<float>(window_size.y));
            io.DeltaTime = update_delta_time();
            ++cached_imgui_resize_frame_count_;
            used_cached_imgui_resize_frame_ = true;
            return;
        }

        ImGui_ImplSDL3_NewFrame();
        if (used_cached_imgui_resize_frame_) {
            ImGui::GetIO().DeltaTime = update_delta_time();
            used_cached_imgui_resize_frame_ = false;
            LOG_PERF("gui_render.imgui_newFrame_cached_resize frames={}",
                     cached_imgui_resize_frame_count_);
            cached_imgui_resize_frame_count_ = 0;
        } else {
            last_imgui_platform_frame_time_ = now;
        }
    }

    void GuiManager::render() {
        auto* window_manager = viewer_ ? viewer_->getWindowManager() : nullptr;
        auto* vulkan_context = (vulkan_gui_ && window_manager) ? window_manager->getVulkanContext() : nullptr;
        if (vulkan_gui_ && !vulkan_context) {
            updateInteractiveTransitionGuard();
            return;
        }

        std::optional<::lfs::core::ScopedTimer> cpu_ui_before_vulkan_timer;
        if (vulkan_gui_) {
            cpu_ui_before_vulkan_timer.emplace("gui_render.cpu_ui_before_vulkan_begin",
                                               ::lfs::core::LogLevel::Performance,
                                               LFS_SOURCE_SITE_CURRENT());
        }

        if (pending_cuda_warning_) {
            constexpr int MIN_MAJOR = lfs::core::MIN_CUDA_VERSION / 1000;
            constexpr int MIN_MINOR = (lfs::core::MIN_CUDA_VERSION % 1000) / 10;
            lfs::core::events::state::CudaVersionUnsupported{
                .major = pending_cuda_warning_->major,
                .minor = pending_cuda_warning_->minor,
                .min_major = MIN_MAJOR,
                .min_minor = MIN_MINOR}
                .emit();
            pending_cuda_warning_.reset();
        }

        if (!cuda_unavailable_notified_ && lfs::core::cuda_is_unavailable()) {
            cuda_unavailable_notified_ = true;
            lfs::core::events::state::CudaUnavailable{
                .message = "CUDA unavailable — GPU features disabled. A driver restart may be required."}
                .emit();
        }

        promptFileAssociation();

        if (pending_ui_scale_ > 0.0f) {
            applyUiScale(pending_ui_scale_);
            pending_ui_scale_ = 0.0f;
        }

        drag_drop_.pollEvents();
        drag_drop_hovering_ = drag_drop_.isDragHovering();

        if (auto* input_controller = viewer_->getInputController()) {
            input_controller->getBindings().updateCapture();
        }

        // Start frame
        {
            LOG_TIMER_THRESHOLD("gui_render.imgui_newFrame", 0.25);
            beginImGuiPlatformFrame(window_manager, vulkan_context);
            if (auto* input_controller = viewer_->getInputController())
                input_controller->applySplitterCursorOverride();
            rmlui_manager_.clearVulkanQueue();
        }
        const auto& sdl_input = viewer_->getWindowManager()->frameInput();
        if (auto* input_controller = viewer_->getInputController()) {
            capturePressedKeysForRebinding(*input_controller, sdl_input);
        }

        // Check mouse state before ImGui::NewFrame() updates WantCaptureMouse
        const bool mouse_in_viewport = isPositionInViewport(sdl_input.mouse_x, sdl_input.mouse_y);

        {
            LOG_TIMER_THRESHOLD("gui_render.imgui_NewFrame_call", 0.25);
            ImGui::NewFrame();
        }

        std::optional<::lfs::core::ScopedTimer> panel_setup_timer;
        panel_setup_timer.emplace("gui_render.panel_setup",
                                  ::lfs::core::LogLevel::Performance,
                                  LFS_SOURCE_SITE_CURRENT());

        {
            LOG_TIMER_THRESHOLD("gui_render.panel_setup.focus_state", 0.25);
            auto& focus = guiFocusState();
            focus.reset();
            // Seed from ImGui only; RmlUi panels populate their own claims during
            // processInput. Aggregating RmlUi's mouse-capture state here reads stale
            // hover state from the previous frame, which becomes self-perpetuating once
            // a panel sets a hover element — toolbar tools then cannot be activated.
            const ImGuiIO& io = ImGui::GetIO();
            focus.want_capture_mouse = io.WantCaptureMouse;
            focus.want_capture_keyboard = io.WantCaptureKeyboard || rmlui_manager_.wantsCaptureKeyboard();
            focus.want_text_input = io.WantTextInput || rmlui_manager_.wantsTextInput();
        }
        const bool startup_plugin_preload_running = python::is_plugin_preload_running();

        // Run queued Python/UI mutations before panel registries take draw snapshots.
        {
            LOG_TIMER_THRESHOLD("gui_render.panel_setup.python_flush_callbacks", 0.25);
            if (!startup_plugin_preload_running && python::has_pending_graphics_callbacks())
                python::flush_graphics_callbacks();
        }

        bool modal_overlay_open = false;
        bool modal_overlay_pending = false;
        bool context_menu_open = false;
        bool startup_overlay_blocking = startup_overlay_.blocksUnderlayInput();
        bool block_underlay_input = startup_overlay_blocking;
        {
            LOG_TIMER_THRESHOLD("gui_render.panel_setup.frame_state", 0.25);
            rmlui_manager_.beginFrameCursorTracking();
            modal_overlay_open = rml_modal_overlay_->isOpen();
            modal_overlay_pending = rml_modal_overlay_->hasPendingRequest();
            context_menu_open = global_context_menu_ && global_context_menu_->isOpen();
            block_underlay_input = block_underlay_input || modal_overlay_open || modal_overlay_pending || context_menu_open;

            if (ImGui::IsKeyPressed(ImGuiKey_Escape) && !ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId)) {
                auto* console_state = panels::PythonConsoleState::tryGetInstance();
                auto* editor = console_state ? console_state->getEditor() : nullptr;
                const bool editor_owns_escape =
                    editor && (editor->isFocused() || editor->hasActiveCompletion());
                if (!editor_owns_escape) {
                    widgets::RequestActiveEditCancel();
                    ImGui::ClearActiveID();
                    if (editor != nullptr) {
                        editor->unfocus();
                    }
                }
            }
        }

        // Check for async completions that must be applied on the main thread.
        if (async_tasks_.hasPendingMainThreadCompletions()) {
            LOG_TIMER_THRESHOLD("gui_render.panel_setup.async_poll", 0.25);
            async_tasks_.pollImportCompletion();
            async_tasks_.pollMesh2SplatCompletion();
            async_tasks_.pollSplatSimplifyCompletion();
        }

        // Poll UV package manager for async operations
        if (python::PackageManager::instance().has_running_operation()) {
            LOG_TIMER_THRESHOLD("gui_render.panel_setup.package_poll", 0.25);
            python::PackageManager::instance().poll();
        }

        const auto should_poll_dev_resources = [&]() {
            if (!dev_resource_watch_.enabled)
                return false;
            if (dev_resource_watch_.pending_rml_reload ||
                dev_resource_watch_.pending_locale_reload ||
                dev_resource_watch_.scan_future.valid())
                return true;
            return dev_resource_watch_.next_scan == std::chrono::steady_clock::time_point{} ||
                   std::chrono::steady_clock::now() >= dev_resource_watch_.next_scan;
        }();
        if (should_poll_dev_resources) {
            LOG_TIMER_THRESHOLD("gui_render.panel_setup.dev_resource_poll", 0.25);
            pollDevResourceHotReload();
        }

        // Hot-reload themes (check once per second)
        static auto next_theme_check = std::chrono::steady_clock::now() + std::chrono::seconds(1);
        const auto now = std::chrono::steady_clock::now();
        if (now >= next_theme_check) {
            LOG_TIMER_THRESHOLD("gui_render.panel_setup.theme_poll", 0.25);
            if (checkThemeFileChanges()) {
                rml_theme::invalidateThemeMediaCache();
            }
            next_theme_check = now + std::chrono::seconds(1);
        }

        if (menu_bar_) {
            LOG_TIMER_THRESHOLD("gui_render.panel_setup.menu_bar", 0.25);
            menu_bar_->render();

            const auto menu_entries_version = menu_bar_->menuEntriesVersion();
            const auto menu_language_generation = app_store().language_generation.get();
            if (!menu_labels_synced_ ||
                menu_entries_version != synced_menu_entries_version_ ||
                menu_language_generation != synced_menu_language_generation_) {
                if (menu_bar_->hasMenuEntries()) {
                    auto entries = menu_bar_->getMenuEntries();
                    std::vector<std::string> labels;
                    std::vector<std::string> idnames;
                    labels.reserve(entries.size());
                    idnames.reserve(entries.size());
                    for (const auto& entry : entries) {
                        labels.emplace_back(LOC(entry.label.c_str()));
                        idnames.emplace_back(entry.idname);
                    }
                    rml_menu_bar_.updateLabels(labels, idnames);
                } else {
                    rml_menu_bar_.updateLabels({}, {});
                }
                synced_menu_entries_version_ = menu_entries_version;
                synced_menu_language_generation_ = menu_language_generation;
                menu_labels_synced_ = true;
            }

            // Reserve work area for the RML menu bar via ImGui's internal inset mechanism
            {
                auto* vp = static_cast<ImGuiViewportP*>(ImGui::GetMainViewport());
                float bar_h = rml_menu_bar_.barHeight();
                vp->BuildWorkInsetMin.y = ImMax(vp->BuildWorkInsetMin.y, bar_h);
                vp->WorkInsetMin.y = ImMax(vp->WorkInsetMin.y, bar_h);
                vp->UpdateWorkRect();
            }

            PanelInputState menu_input = buildPanelInputFromSDL(sdl_input);
            if (const ImGuiViewport* const main_viewport = ImGui::GetMainViewport()) {
                menu_input.screen_x = main_viewport->Pos.x;
                menu_input.screen_y = main_viewport->Pos.y;
                menu_input.screen_w = static_cast<int>(main_viewport->Size.x);
                menu_input.screen_h = static_cast<int>(main_viewport->Size.y);
            }
            if (block_underlay_input)
                menu_input = maskInputForBlockedUi(std::move(menu_input));

            rml_menu_bar_.setUiHidden(ui_hidden_);
            rml_menu_bar_.processInput(menu_input);

            if (rml_menu_bar_.wantsInput())
                guiFocusState().want_capture_mouse = true;

            if (!vulkan_gui_) {
                rml_menu_bar_.setViewportRightEdge(menu_toolbar_right_edge_ - menu_input.screen_x);
                rml_menu_bar_.draw(menu_input.screen_w, menu_input.screen_h);
            }
        } else {
            LOG_TIMER_THRESHOLD("gui_render.panel_setup.menu_bar_suspend", 0.25);
            rml_menu_bar_.suspend();
        }

        PanelInputState frame_input;
        PanelInputState startup_overlay_input;
        {
            LOG_TIMER_THRESHOLD("gui_render.panel_setup.frame_input", 0.25);
            frame_input = buildPanelInputFromSDL(sdl_input);
            startup_overlay_input = frame_input;
            if (startup_overlay_blocking)
                frame_input = maskInputForBlockedUi(std::move(frame_input));
            updateInputOverrides(frame_input, mouse_in_viewport);
            if (auto* const wm = viewer_->getWindowManager()) {
                frame_input.viewport_keyboard_focus = wm->inputRouter().isViewportKeyboardFocused();
            }
        }
        const bool frame_has_input_activity =
            sdl_input.window_event || hasPointerActivity(sdl_input) || hasKeyboardActivity(sdl_input);

        auto& reg = PanelRegistry::instance();
        const bool has_side_panel_plugins = reg.has_panels(PanelSpace::SidePanel);
        const bool has_floating_panels = reg.has_panels(PanelSpace::Floating);
        const bool has_status_bar_panels = reg.has_panels(PanelSpace::StatusBar);
        const bool has_viewport_overlay_panels = reg.has_panels(PanelSpace::ViewportOverlay);
        const bool right_panel_visible = show_main_panel_ && !ui_hidden_;
        PanelAnimationDemand panel_animation_demand;
        {
            LOG_TIMER_THRESHOLD("gui_render.panel_animation_demand", 0.01);
            panel_animation_demand =
                reg.animationDemandForVisiblePanels({
                    .active_main_tab = panel_layout_.getActiveTab(),
                    .ui_visible = !ui_hidden_,
                    .right_panel_visible = right_panel_visible,
                    .bottom_dock_visible = panel_layout_.isBottomDockVisible(),
                    .left_dock_visible = panel_layout_.isLeftDockVisible(),
                });
        }
        const bool panel_registry_needs_animation = panel_animation_demand.any();
        const bool right_panel_registry_needs_animation = panel_animation_demand.rightPanel();
        const bool bottom_dock_registry_needs_animation = panel_animation_demand.bottom_dock;

        if (!ui_hidden_) {
            LOG_TIMER_THRESHOLD("gui_render.panel_setup.shell_frame", 0.25);
            const auto* mvp = ImGui::GetMainViewport();
            const float status_bar_h = PanelLayoutManager::STATUS_BAR_HEIGHT * current_ui_scale_;
            const float panel_h = mvp->WorkSize.y - status_bar_h;
            panel_layout_.enforceWidthConstraints(show_main_panel_, ui_hidden_,
                                                  {
                                                      .work_pos = {mvp->WorkPos.x, mvp->WorkPos.y},
                                                      .work_size = {mvp->WorkSize.x, mvp->WorkSize.y},
                                                      .any_item_active = ImGui::IsAnyItemActive() ||
                                                                         rmlui_manager_.anyItemActive(),
                                                  });

            ShellRegions shell_regions;
            shell_regions.screen = {mvp->Pos.x, mvp->Pos.y, mvp->Size.x, mvp->Size.y};
            shell_regions.menu = {mvp->Pos.x, mvp->Pos.y,
                                  mvp->Size.x, mvp->WorkPos.y - mvp->Pos.y};

            if (show_main_panel_) {
                const float rpw = panel_layout_.getRightPanelWidth();
                shell_regions.right_panel = {
                    mvp->WorkPos.x + mvp->WorkSize.x - rpw,
                    mvp->WorkPos.y,
                    rpw,
                    panel_h,
                };
            }

            shell_regions.status = {
                mvp->WorkPos.x,
                mvp->WorkPos.y + mvp->WorkSize.y - status_bar_h,
                mvp->WorkSize.x,
                status_bar_h,
            };

            rml_shell_frame_.render(shell_regions);
        }

        // Update editor context state for this frame
        auto& editor_ctx = viewer_->getEditorContext();
        auto* const scene_manager = viewer_->getSceneManager();
        auto* const trainer_manager = viewer_->getTrainerManager();
        EditorContextUpdateStamp editor_context_stamp;
        editor_context_stamp.valid = true;
        editor_context_stamp.has_scene_manager = scene_manager != nullptr;
        editor_context_stamp.has_trainer_manager = trainer_manager != nullptr;
        editor_context_stamp.scene_generation = python::get_scene_generation();
        editor_context_stamp.selection_generation = app_store().selection_generation.get();
        if (scene_manager) {
            editor_context_stamp.has_dataset = scene_manager->hasDataset();
            const auto& scene_ref = scene_manager->getScene();
            editor_context_stamp.has_training_model = scene_ref.getTrainingModel() != nullptr;
            editor_context_stamp.scene_node_count =
                static_cast<std::uint64_t>(scene_ref.getNodeCount());
        }
        if (trainer_manager) {
            editor_context_stamp.trainer_running = trainer_manager->isRunning();
            editor_context_stamp.trainer_paused = trainer_manager->isPaused();
            editor_context_stamp.trainer_finished = trainer_manager->isFinished();
        }
        const bool editor_context_sources_changed =
            !(editor_context_stamp == last_editor_context_update_stamp_);
        const bool update_editor_context =
            editor_context_sources_changed || frame_has_input_activity;
        if (update_editor_context) {
            LOG_TIMER_THRESHOLD("gui_render.panel_setup.editor_context_update", 0.25);
            editor_ctx.update(scene_manager, trainer_manager);
            last_editor_context_update_stamp_ = editor_context_stamp;
        }

        // Create context for this frame
        UIContext ctx;
        {
            LOG_TIMER_THRESHOLD("gui_render.panel_setup.context_build", 0.25);
            ctx = UIContext{
                .viewer = viewer_,
                .window_states = &window_states_,
                .editor = &editor_ctx,
                .sequencer_controller = &sequencer_ui_.controller(),
                .rml_manager = &rmlui_manager_,
                .fonts = buildFontSet()};
        }

        // Build draw context for panel registry
        lfs::core::Scene* scene = nullptr;
        if (auto* sm = ctx.viewer->getSceneManager()) {
            scene = &sm->getScene();
        }
        PanelDrawContext draw_ctx;
        draw_ctx.ui = &ctx;
        draw_ctx.viewport = &viewport_layout_;
        draw_ctx.scene = scene;
        draw_ctx.ui_hidden = ui_hidden_;
        draw_ctx.frame_serial = ++panel_frame_serial_;
        draw_ctx.scene_generation = python::get_scene_generation();
        draw_ctx.suppress_non_native_panels = startup_plugin_preload_running;
        if (auto* sm = ctx.viewer->getSceneManager())
            draw_ctx.has_selection = sm->hasSelectedNode();
        if (auto* cc = lfs::event::command_center())
            draw_ctx.is_training = cc->snapshot().is_running;

        if (has_side_panel_plugins) {
            LOG_TIMER_THRESHOLD("gui_render.panel_setup.legacy_side_panel_preload", 0.25);
            reg.preload_panels(PanelSpace::SidePanel, draw_ctx);
        }

        auto* mvp_input = ImGui::GetMainViewport();
        s_frame_input = &sdl_input;
        PanelInputState panel_input = frame_input;
        panel_input.screen_x = mvp_input->Pos.x;
        panel_input.screen_y = mvp_input->Pos.y;
        panel_input.bg_draw_list = ImGui::GetBackgroundDrawList(mvp_input);
        panel_input.fg_draw_list = ImGui::GetForegroundDrawList(mvp_input);
        PanelInputState raw_panel_input = panel_input;
        if (block_underlay_input)
            panel_input = maskInputForBlockedUi(std::move(panel_input));
        if (!modal_overlay_open && global_context_menu_->isOpen())
            global_context_menu_->processInput(raw_panel_input);

        ScreenState screen;
        {
            LOG_TIMER_THRESHOLD("gui_render.panel_setup.panel_input_state", 0.25);
            screen.work_pos = {mvp_input->WorkPos.x, mvp_input->WorkPos.y};
            screen.work_size = {mvp_input->WorkSize.x, mvp_input->WorkSize.y};
            screen.any_item_active = ImGui::IsAnyItemActive() || rmlui_manager_.anyItemActive();
        }
        panel_layout_.enforceWidthConstraints(show_main_panel_, ui_hidden_, screen);
        viewport_layout_ = panel_layout_.computeViewportLayout(
            show_main_panel_, ui_hidden_, window_states_["python_console"], screen);

        constexpr uint8_t kUiLayoutSettleFrames = 3;
        const bool python_console_visible = window_states_["python_console"];
        const bool ui_layout_changed =
            std::abs(screen.work_pos.x - last_ui_layout_work_pos_.x) > 0.5f ||
            std::abs(screen.work_pos.y - last_ui_layout_work_pos_.y) > 0.5f ||
            std::abs(screen.work_size.x - last_ui_layout_work_size_.x) > 0.5f ||
            std::abs(screen.work_size.y - last_ui_layout_work_size_.y) > 0.5f ||
            std::abs(panel_layout_.getRightPanelWidth() - last_ui_layout_right_panel_w_) > 0.5f ||
            std::abs(panel_layout_.getScenePanelRatio() - last_ui_layout_scene_ratio_) > 0.0001f ||
            std::abs(panel_layout_.getPythonConsoleWidth() - last_ui_layout_python_console_w_) > 0.5f ||
            std::abs(panel_layout_.getBottomDockHeight() - last_ui_layout_bottom_dock_h_) > 0.5f ||
            std::abs(panel_layout_.getLeftDockWidth() - last_ui_layout_left_dock_w_) > 0.5f ||
            show_main_panel_ != last_ui_layout_show_main_panel_ ||
            ui_hidden_ != last_ui_layout_ui_hidden_ ||
            python_console_visible != last_ui_layout_python_console_visible_ ||
            panel_layout_.isBottomDockVisible() != last_ui_layout_bottom_dock_visible_ ||
            panel_layout_.isLeftDockVisible() != last_ui_layout_left_dock_visible_ ||
            panel_layout_.getActiveTab() != last_ui_layout_active_tab_;

        if (ui_layout_changed) {
            LOG_TIMER_THRESHOLD("gui_render.panel_setup.layout_state_update", 0.25);
            ui_layout_settle_frames_ = kUiLayoutSettleFrames;
            last_ui_layout_work_pos_ = screen.work_pos;
            last_ui_layout_work_size_ = screen.work_size;
            last_ui_layout_right_panel_w_ = panel_layout_.getRightPanelWidth();
            last_ui_layout_scene_ratio_ = panel_layout_.getScenePanelRatio();
            last_ui_layout_python_console_w_ = panel_layout_.getPythonConsoleWidth();
            last_ui_layout_bottom_dock_h_ = panel_layout_.getBottomDockHeight();
            last_ui_layout_left_dock_w_ = panel_layout_.getLeftDockWidth();
            last_ui_layout_show_main_panel_ = show_main_panel_;
            last_ui_layout_ui_hidden_ = ui_hidden_;
            last_ui_layout_python_console_visible_ = python_console_visible;
            last_ui_layout_bottom_dock_visible_ = panel_layout_.isBottomDockVisible();
            last_ui_layout_left_dock_visible_ = panel_layout_.isLeftDockVisible();
            last_ui_layout_active_tab_ = panel_layout_.getActiveTab();
        }

        bool right_panel_requires_live_layout = false;
        bool right_panel_active_tab_changed = false;
        bool right_panel_was_dirty = false;
        bool right_panel_needs_animation = false;
        bool right_panel_layout_resize_active = false;
        bool right_panel_pointer_activity = false;
        bool right_panel_pointer_targets_panel = false;
        bool right_panel_pointer_capture_active = false;
        bool right_panel_wants_input = false;
        bool right_panel_keyboard_activity = false;
        bool right_panel_scene_header_live = false;
        bool right_panel_active_tab_live = false;
        bool right_panel_pointer_over_scene_header = false;
        bool right_panel_pointer_over_active_tab = false;
        if (show_main_panel_ && !ui_hidden_) {
            LOG_TIMER_THRESHOLD("gui_render.panel_setup.rml_right_panel", 0.25);
            const float sbh = PanelLayoutManager::STATUS_BAR_HEIGHT * current_ui_scale_;
            const float rpw = panel_layout_.getRightPanelWidth();
            const float ph = screen.work_size.y - sbh;
            const float splitter_h = PanelLayoutManager::SPLITTER_H * current_ui_scale_;
            const float tab_bar_h = PanelLayoutManager::TAB_BAR_H * current_ui_scale_;
            const float avail_h = ph - 16.0f;
            const float scene_h = std::max(80.0f * current_ui_scale_,
                                           avail_h * panel_layout_.getScenePanelRatio() - splitter_h * 0.5f);

            RightPanelLayout rp_layout;
            rp_layout.pos = glm::vec2(screen.work_pos.x + screen.work_size.x - rpw, screen.work_pos.y);
            rp_layout.size = glm::vec2(rpw, ph);
            rp_layout.scene_h = scene_h + 8.0f;
            rp_layout.splitter_h = splitter_h;
            right_panel_was_dirty = rml_right_panel_.needsAnimationFrame();
            const float right_panel_edge_grab_w =
                std::max(PanelLayoutManager::SPLITTER_H * current_ui_scale_,
                         8.0f * current_ui_scale_);
            const bool pointer_over_right_panel =
                pointInRect(panel_input.mouse_x, panel_input.mouse_y,
                            rp_layout.pos, rp_layout.size);
            const bool pointer_over_right_panel_edge =
                panel_input.mouse_x >= rp_layout.pos.x - right_panel_edge_grab_w &&
                panel_input.mouse_x <= rp_layout.pos.x + right_panel_edge_grab_w &&
                panel_input.mouse_y >= rp_layout.pos.y &&
                panel_input.mouse_y < rp_layout.pos.y + rp_layout.size.y;
            constexpr float RIGHT_PANEL_PAD = 8.0f;
            const float content_x = rp_layout.pos.x + RIGHT_PANEL_PAD;
            const float content_top = screen.work_pos.y + RIGHT_PANEL_PAD;
            const float content_w = rpw - 2.0f * RIGHT_PANEL_PAD;
            const float tab_content_y = content_top + scene_h + splitter_h + tab_bar_h;
            const float tab_content_h = std::max(0.0f, content_top + avail_h - tab_content_y);
            right_panel_pointer_over_scene_header =
                !pointer_over_right_panel_edge &&
                pointInRect(panel_input.mouse_x, panel_input.mouse_y,
                            glm::vec2{content_x, content_top},
                            glm::vec2{content_w, scene_h});
            right_panel_pointer_over_active_tab =
                !pointer_over_right_panel_edge &&
                pointInRect(panel_input.mouse_x, panel_input.mouse_y,
                            glm::vec2{content_x, tab_content_y},
                            glm::vec2{content_w, tab_content_h});

            const bool float_blocks_rp = has_floating_panels &&
                                         reg.isPositionOverFloatingPanel(panel_input.mouse_x, panel_input.mouse_y);
            if (float_blocks_rp) {
                PanelInputState masked_input = panel_input;
                masked_input.mouse_x = -1.0e9f;
                masked_input.mouse_y = -1.0e9f;
                for (auto& v : masked_input.mouse_clicked)
                    v = false;
                for (auto& v : masked_input.mouse_released)
                    v = false;
                for (auto& v : masked_input.mouse_down)
                    v = false;
                masked_input.mouse_wheel = 0;
                rml_right_panel_.processInput(rp_layout, masked_input);
            } else {
                rml_right_panel_.processInput(rp_layout, panel_input);
            }

            if (rml_right_panel_.wantsInput() && !float_blocks_rp)
                guiFocusState().want_capture_mouse = true;
            if (rml_right_panel_.wantsKeyboard())
                guiFocusState().want_capture_keyboard = true;

            const auto main_tabs = reg.get_panels_for_space(PanelSpace::MainPanelTab);
            right_panel_active_tab_changed = panel_layout_.syncActiveTab(main_tabs, focus_panel_name_);
            std::vector<TabSnapshot> tab_snaps;
            tab_snaps.reserve(main_tabs.size());
            for (size_t i = 0; i < main_tabs.size(); ++i) {
                const auto& t = main_tabs[i];
                tab_snaps.push_back({
                    .id = t.id,
                    .label = t.label,
                    .dom_id = makeRmlTabDomId(t.id),
                    .closeable = t.tab_closeable,
                });
            }

            const bool pointer_targets_right_panel =
                !float_blocks_rp && (pointer_over_right_panel || pointer_over_right_panel_edge);
            right_panel_pointer_targets_panel = pointer_targets_right_panel;
            if (pointer_targets_right_panel &&
                (hasMouseButtonClicked(sdl_input) || hasMouseButtonDown(sdl_input))) {
                right_panel_pointer_live_capture_ = true;
                if (pointer_over_right_panel_edge ||
                    rml_right_panel_.getCursorRequest() != CursorRequest::None) {
                    right_panel_pointer_capture_region_ = RightPanelPointerRegion::Resize;
                } else if (right_panel_pointer_over_scene_header) {
                    right_panel_pointer_capture_region_ = RightPanelPointerRegion::SceneHeader;
                } else if (right_panel_pointer_over_active_tab) {
                    right_panel_pointer_capture_region_ = RightPanelPointerRegion::ActiveTab;
                } else {
                    right_panel_pointer_capture_region_ = RightPanelPointerRegion::Chrome;
                }
            }
            right_panel_pointer_capture_active = right_panel_pointer_live_capture_;
            right_panel_wants_input = rml_right_panel_.wantsInput();
            right_panel_pointer_activity =
                hasPointerActivity(sdl_input) &&
                (pointer_targets_right_panel || right_panel_wants_input ||
                 right_panel_pointer_live_capture_);
            right_panel_keyboard_activity = hasKeyboardActivity(sdl_input);
            right_panel_needs_animation = rml_right_panel_.needsAnimationFrame();
            right_panel_layout_resize_active = panel_layout_.isResizingPanel();

            const bool force_full_panel_live =
                ui_layout_changed || right_panel_active_tab_changed ||
                right_panel_layout_resize_active ||
                right_panel_pointer_capture_region_ == RightPanelPointerRegion::Resize;
            right_panel_scene_header_live =
                force_full_panel_live || panel_animation_demand.scene_header;
            right_panel_active_tab_live =
                force_full_panel_live || panel_animation_demand.main_panel_tab;

            if (right_panel_pointer_activity) {
                right_panel_scene_header_live =
                    right_panel_scene_header_live || right_panel_pointer_over_scene_header ||
                    right_panel_pointer_capture_region_ == RightPanelPointerRegion::SceneHeader;
                right_panel_active_tab_live =
                    right_panel_active_tab_live || right_panel_pointer_over_active_tab ||
                    right_panel_pointer_capture_region_ == RightPanelPointerRegion::ActiveTab;
            }

            if (right_panel_keyboard_activity) {
                right_panel_scene_header_live = true;
                right_panel_active_tab_live = true;
            }

            right_panel_requires_live_layout =
                right_panel_scene_header_live || right_panel_active_tab_live;

            rml_right_panel_.render(rp_layout, tab_snaps, panel_layout_.getActiveTab(),
                                    panel_input.screen_x, panel_input.screen_y,
                                    panel_input.screen_w, panel_input.screen_h);
        } else {
            right_panel_pointer_live_capture_ = false;
            right_panel_pointer_capture_region_ = RightPanelPointerRegion::None;
        }
        if (!hasMouseButtonDown(sdl_input)) {
            right_panel_pointer_live_capture_ = false;
            right_panel_pointer_capture_region_ = RightPanelPointerRegion::None;
        }
        if (block_underlay_input || !right_panel_requires_live_layout) {
            panel_layout_.renderRightPanelCached(ctx, draw_ctx, show_main_panel_, ui_hidden_,
                                                 window_states_, focus_panel_name_, panel_input, screen);
        } else {
            panel_layout_.renderRightPanel(ctx, draw_ctx, show_main_panel_, ui_hidden_,
                                           window_states_, focus_panel_name_, panel_input, screen,
                                           {
                                               .scene_header_live = right_panel_scene_header_live,
                                               .active_tab_live = right_panel_active_tab_live,
                                           });
        }

        const float status_bar_h = PanelLayoutManager::STATUS_BAR_HEIGHT * current_ui_scale_;
        const float bottom_dock_h = std::max(panel_layout_.getBottomDockHeight(), 0.0f);
        const float bottom_dock_w = show_main_panel_ && !ui_hidden_
                                        ? std::max(0.0f, screen.work_size.x -
                                                             panel_layout_.getRightPanelWidth())
                                        : screen.work_size.x;
        const float bottom_dock_y =
            screen.work_pos.y + screen.work_size.y - status_bar_h - bottom_dock_h;
        const float bottom_dock_edge_grab_h =
            std::max(PanelLayoutManager::SPLITTER_H * current_ui_scale_,
                     8.0f * current_ui_scale_);
        const bool pointer_over_bottom_dock =
            panel_layout_.isBottomDockVisible() &&
            pointInRect(panel_input.mouse_x, panel_input.mouse_y,
                        glm::vec2{screen.work_pos.x, bottom_dock_y},
                        glm::vec2{bottom_dock_w, bottom_dock_h});
        const bool pointer_over_bottom_dock_edge =
            panel_layout_.isBottomDockVisible() &&
            panel_input.mouse_x >= screen.work_pos.x &&
            panel_input.mouse_x < screen.work_pos.x + bottom_dock_w &&
            panel_input.mouse_y >= bottom_dock_y - bottom_dock_edge_grab_h &&
            panel_input.mouse_y <= bottom_dock_y + bottom_dock_edge_grab_h;
        const bool pointer_targets_bottom_dock =
            pointer_over_bottom_dock || pointer_over_bottom_dock_edge;
        if (pointer_targets_bottom_dock &&
            (hasMouseButtonClicked(sdl_input) || hasMouseButtonDown(sdl_input))) {
            bottom_dock_pointer_live_capture_ = true;
        }
        const bool panel_layout_resize_active = panel_layout_.isResizingPanel();
        const bool bottom_dock_pointer_activity =
            hasPointerActivity(sdl_input) &&
            (pointer_targets_bottom_dock || bottom_dock_pointer_live_capture_ ||
             panel_layout_resize_active);
        const bool bottom_dock_input_activity =
            bottom_dock_pointer_activity ||
            hasKeyboardActivity(sdl_input);
        const bool bottom_dock_requires_live_layout =
            ui_layout_changed || panel_layout_resize_active ||
            bottom_dock_registry_needs_animation || sequencer_ui_.needsAnimationFrame() ||
            bottom_dock_input_activity;
        if (block_underlay_input || !bottom_dock_requires_live_layout) {
            panel_layout_.renderBottomDockCached(draw_ctx, show_main_panel_, ui_hidden_,
                                                 panel_input, screen);
        } else {
            panel_layout_.renderBottomDock(draw_ctx, show_main_panel_, ui_hidden_,
                                           panel_input, screen);
        }
        if (!hasMouseButtonDown(sdl_input))
            bottom_dock_pointer_live_capture_ = false;

        // ── Left Dock ─────────────────────────────────────────────
        constexpr float ICON_BAR_WIDTH = 40.0f;
        const float icon_bar_w = ICON_BAR_WIDTH * current_ui_scale_;
        const float left_dock_panel_w = std::max(panel_layout_.getLeftDockWidth(), 0.0f);
        const float left_dock_h = show_main_panel_ && !ui_hidden_
                                      ? std::max(0.0f, screen.work_size.y -
                                                           PanelLayoutManager::STATUS_BAR_HEIGHT * current_ui_scale_)
                                      : screen.work_size.y;
        const float left_dock_edge_grab_w =
            std::max(PanelLayoutManager::SPLITTER_H * current_ui_scale_,
                     8.0f * current_ui_scale_);
        const float left_dock_x = screen.work_pos.x + icon_bar_w;
        const float left_dock_right_x = panel_layout_.isLeftDockVisible() ? left_dock_x + left_dock_panel_w : -1.0f;
        const bool pointer_over_left_dock =
            panel_layout_.isLeftDockVisible() &&
            pointInRect(panel_input.mouse_x, panel_input.mouse_y,
                        glm::vec2{left_dock_x, screen.work_pos.y},
                        glm::vec2{left_dock_panel_w, left_dock_h});
        const bool pointer_over_left_dock_edge =
            panel_layout_.isLeftDockVisible() &&
            panel_input.mouse_x >= left_dock_right_x - left_dock_edge_grab_w &&
            panel_input.mouse_x <= left_dock_right_x + left_dock_edge_grab_w &&
            panel_input.mouse_y >= screen.work_pos.y &&
            panel_input.mouse_y < screen.work_pos.y + left_dock_h;
        const bool pointer_targets_left_dock =
            pointer_over_left_dock || pointer_over_left_dock_edge;
        if (pointer_targets_left_dock &&
            (hasMouseButtonClicked(sdl_input) || hasMouseButtonDown(sdl_input))) {
            left_dock_pointer_live_capture_ = true;
        }
        const bool left_dock_pointer_activity =
            hasPointerActivity(sdl_input) &&
            (pointer_targets_left_dock || left_dock_pointer_live_capture_ ||
             panel_layout_resize_active);
        const bool left_dock_input_activity =
            left_dock_pointer_activity ||
            hasKeyboardActivity(sdl_input);
        const bool left_dock_requires_live_layout =
            ui_layout_changed || panel_layout_resize_active ||
            panel_animation_demand.left_dock ||
            left_dock_input_activity;
        if (block_underlay_input || !left_dock_requires_live_layout) {
            panel_layout_.renderLeftDockCached(draw_ctx, show_main_panel_, ui_hidden_,
                                               panel_input, screen);
        } else {
            panel_layout_.renderLeftDock(draw_ctx, show_main_panel_, ui_hidden_,
                                         panel_input, screen);
        }
        if (!hasMouseButtonDown(sdl_input))
            left_dock_pointer_live_capture_ = false;

        const bool dock_resize_interaction_active = panel_layout_.isResizeInteractionActive();
        if (dock_resize_interaction_active != dock_resize_interaction_active_) {
            viewer_->getRenderingManager()->setViewportResizeActive(dock_resize_interaction_active);
            dock_resize_interaction_active_ = dock_resize_interaction_active;
        }

        if (has_side_panel_plugins || has_floating_panels || has_status_bar_panels ||
            right_panel_requires_live_layout || bottom_dock_requires_live_layout ||
            left_dock_requires_live_layout ||
            ui_layout_changed || panel_registry_needs_animation || block_underlay_input) {
            LOG_PERF("gui_render.router side_panel_plugins={} floating_panels={} status_bar_panels={} viewport_overlay_panels={} editor_update={} right_live={} right_scene_live={} right_tab_live={} bottom_live={} layout_changed={} panel_registry_anim={} right_registry_anim={} bottom_registry_anim={} viewport_registry_anim={} block_underlay={}",
                     has_side_panel_plugins,
                     has_floating_panels,
                     has_status_bar_panels,
                     has_viewport_overlay_panels,
                     update_editor_context,
                     right_panel_requires_live_layout,
                     right_panel_scene_header_live,
                     right_panel_active_tab_live,
                     bottom_dock_requires_live_layout,
                     ui_layout_changed,
                     panel_registry_needs_animation,
                     right_panel_registry_needs_animation,
                     bottom_dock_registry_needs_animation,
                     panel_animation_demand.viewport_overlay,
                     block_underlay_input);
            if (right_panel_requires_live_layout) {
                LOG_PERF("gui_render.router.right_panel_reasons layout={} tab={} dirty={} animation={} registry_anim={} resize={} pointer={} pointer_target={} pointer_scene={} pointer_tab={} pointer_capture={} capture_region={} wants_input={} keyboard={} scene_live={} active_tab_live={}",
                         ui_layout_changed,
                         right_panel_active_tab_changed,
                         right_panel_was_dirty,
                         right_panel_needs_animation,
                         right_panel_registry_needs_animation,
                         right_panel_layout_resize_active,
                         right_panel_pointer_activity,
                         right_panel_pointer_targets_panel,
                         right_panel_pointer_over_scene_header,
                         right_panel_pointer_over_active_tab,
                         right_panel_pointer_capture_active,
                         static_cast<int>(right_panel_pointer_capture_region_),
                         right_panel_wants_input,
                         right_panel_keyboard_activity,
                         right_panel_scene_header_live,
                         right_panel_active_tab_live);
            }
        }

        applyFrameInputCapture(&rml_right_panel_);

        auto apply_cursor = [](CursorRequest req) {
            switch (req) {
            case CursorRequest::ResizeEW: ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW); break;
            case CursorRequest::ResizeNS: ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS); break;
            default: break;
            }
        };
        viewport_layout_ = panel_layout_.computeViewportLayout(
            show_main_panel_, ui_hidden_, window_states_["python_console"], screen);
        python::set_viewport_bounds(viewport_layout_.pos.x, viewport_layout_.pos.y,
                                    viewport_layout_.size.x, viewport_layout_.size.y);

        // The render-mode toolbar anchors to the right panel's edge, so the docked
        // editor console must not drag it left when it shrinks the viewport.
        const ViewportLayout toolbar_layout = panel_layout_.computeViewportLayout(
            show_main_panel_, ui_hidden_, false, screen);
        menu_toolbar_right_edge_ = toolbar_layout.pos.x + toolbar_layout.size.x;

        {
            LOG_TIMER_THRESHOLD("gui_render.gizmo_update", 0.25);
            gizmo_manager_.updateToolState(ctx, ui_hidden_);
            gizmo_manager_.updateCropFlash();
        }

        const float viewport_content_offset = viewport_layout_.pos.x - screen.work_pos.x;
        float primary_toolbar_x = viewport_content_offset;
        float primary_toolbar_width = viewport_layout_.size.x;
        bool show_secondary_toolbar = false;
        float secondary_toolbar_x = 0.0f;
        float secondary_toolbar_width = 0.0f;
        if (auto* const rendering = viewer_ ? viewer_->getRenderingManager() : nullptr;
            rendering && rendering->isIndependentSplitViewActive()) {
            if (const auto primary_panel = rendering->resolveViewerPanel(
                    viewer_->getViewport(),
                    viewport_layout_.pos, viewport_layout_.size, std::nullopt, SplitViewPanelId::Left)) {
                primary_toolbar_x = primary_panel->x - screen.work_pos.x;
                primary_toolbar_width = primary_panel->width;
            }
            if (const auto secondary_panel = rendering->resolveViewerPanel(
                    viewer_->getViewport(),
                    viewport_layout_.pos, viewport_layout_.size, std::nullopt, SplitViewPanelId::Right)) {
                show_secondary_toolbar = secondary_panel->valid();
                secondary_toolbar_x = secondary_panel->x - screen.work_pos.x;
                secondary_toolbar_width = secondary_panel->width;
            }
        }

        rml_viewport_overlay_.setToolbarPanels(primary_toolbar_x,
                                               primary_toolbar_width,
                                               show_secondary_toolbar,
                                               secondary_toolbar_x,
                                               secondary_toolbar_width);
        const float left_dock_w =
            ui_hidden_ ? 0.0f : icon_bar_w + (panel_layout_.isLeftDockVisible() ? left_dock_panel_w : 0.0f);
        const glm::vec2 overlay_pos = {screen.work_pos.x, viewport_layout_.pos.y};
        const glm::vec2 overlay_size = {viewport_layout_.size.x + left_dock_w, viewport_layout_.size.y};
        rml_viewport_overlay_.setViewportBounds(
            overlay_pos, overlay_size,
            {panel_input.screen_x, panel_input.screen_y});
        rml_viewport_overlay_.setViewportContentOffset(viewport_content_offset);
        RmlViewportOverlay::SplitDividerOverlayState split_divider_state;
        if (auto* const rendering = viewer_ ? viewer_->getRenderingManager() : nullptr;
            rendering && rendering->isSplitViewActive() && !rendering->isIndependentSplitViewActive()) {
            const auto divider_x = rendering->getSplitDividerScreenX(viewport_layout_.pos, viewport_layout_.size);
            const auto content_bounds = rendering->getContentBounds(glm::ivec2(
                std::max(static_cast<int>(viewport_layout_.size.x), 0),
                std::max(static_cast<int>(viewport_layout_.size.y), 0)));
            if (divider_x && content_bounds.width > 0.0f && content_bounds.height > 0.0f) {
                const auto& t = theme();
                constexpr float kSplitDividerMinWidthPx = 10.0f;
                const float divider_width =
                    std::max(kSplitDividerMinWidthPx * current_ui_scale_,
                             std::round(t.viewport.border_size * current_ui_scale_ * 4.0f));
                split_divider_state.visible = true;
                split_divider_state.x = std::round((*divider_x - screen.work_pos.x) - divider_width * 0.5f);
                split_divider_state.y = content_bounds.y;
                split_divider_state.width = divider_width;
                split_divider_state.height = content_bounds.height;
            }
        }
        rml_viewport_overlay_.setSplitDividerOverlay(split_divider_state);
        RmlViewportOverlay::LodStatsOverlayState lod_stats_state;
        if (auto* const rendering = viewer_ ? viewer_->getRenderingManager() : nullptr) {
            lod_stats_state = makeLodStatsOverlayState(rendering->getLodStats());
        }
        rml_viewport_overlay_.setLodStatsOverlay(std::move(lod_stats_state));
        AppStore::GTMetricsOverlayConfig gt_metrics_config;
        if (auto* const rendering = viewer_ ? viewer_->getRenderingManager() : nullptr) {
            const auto settings = rendering->getSettings();
            if (rendering->isGTComparisonActive() &&
                settings.camera_metrics_mode != RenderSettings::CameraMetricsMode::Off) {
                gt_metrics_config.visible = true;
                gt_metrics_config.show_ssim =
                    settings.camera_metrics_mode == RenderSettings::CameraMetricsMode::PSNRSSIM;

                const auto content_bounds = rendering->getContentBounds(glm::ivec2(
                    std::max(static_cast<int>(viewport_layout_.size.x), 0),
                    std::max(static_cast<int>(viewport_layout_.size.y), 0)));
                gt_metrics_config.x =
                    content_bounds.x + content_bounds.width * settings.split_position + 18.0f;
                gt_metrics_config.y = content_bounds.y + 18.0f;
                gt_metrics_config.current_camera_id = rendering->getCurrentCameraId();
            }
        }
        if (!published_gt_metrics_overlay_config_ ||
            !(*published_gt_metrics_overlay_config_ == gt_metrics_config)) {
            app_store().gt_metrics_overlay_config.set(gt_metrics_config);
            published_gt_metrics_overlay_config_ = gt_metrics_config;
        }
        const auto publish_vram_hud_overlay_if_due = [&]() {
            const auto now = std::chrono::steady_clock::now();
            if (!isVramHudOverlayVisible()) {
                if (vram_hud_visible_published_) {
                    app_store().vram_hud.set(AppStore::VramHud{});
                    vram_hud_visible_published_ = false;
                }
                next_vram_hud_publish_ = {};
                return;
            }

            if (isVramHudPublishDue(now)) {
                auto& profiler = lfs::diagnostics::VramProfiler::instance();
                {
                    LOG_TIMER("gui_render.vram_hud_sample");
                    profiler.sampleCudaMemory();
                    const auto memory = queryGpuMemory();
                    profiler.updateProcessMemory(memory.process_used,
                                                 memory.total_used,
                                                 memory.total,
                                                 memory.device_name);
                    if (auto* const wm = viewer_ ? viewer_->getWindowManager() : nullptr) {
                        if (auto* const vk = wm->getVulkanContext()) {
                            profiler.setVulkanVmaUsed(vk->queryVmaUsedBytes());
                        }
                    }
                }
                app_store().vram_hud.set(AppStore::VramHud{
                    .visible = true,
                    .snapshot = std::make_shared<const lfs::diagnostics::VramProfilerSnapshot>(
                        profiler.snapshot())});
                vram_hud_visible_published_ = true;
                next_vram_hud_publish_ = now + std::chrono::milliseconds(250);
            }
        };
        if (startup_overlay_.isVisible()) {
            startup_overlay_.setInput(&startup_overlay_input);
            if (startup_overlay_.blocksUnderlayInput()) {
                auto& focus = guiFocusState();
                focus.want_capture_mouse = true;
                focus.want_capture_keyboard = true;
            }
        } else {
            startup_overlay_.setInput(nullptr);
        }
        PanelInputState viewport_overlay_input = panel_input;
        if (has_floating_panels &&
            reg.isPositionOverFloatingPanel(panel_input.mouse_x, panel_input.mouse_y)) {
            viewport_overlay_input = maskInputForBlockedUi(std::move(viewport_overlay_input));
        }
        {
            LOG_TIMER_THRESHOLD("gui_render.rml_viewport_overlay.processInput", 0.25);
            if (!block_underlay_input)
                rml_viewport_overlay_.processInput(viewport_overlay_input);
        }
        if (rml_viewport_overlay_.wantsInput() && viewport_overlay_input.mouse_clicked[0]) {
            if (auto* const rendering = viewer_ ? viewer_->getRenderingManager() : nullptr;
                rendering && rendering->isIndependentSplitViewActive()) {
                if (const auto target_panel = rendering->resolveViewerPanel(
                        viewer_->getViewport(),
                        viewport_layout_.pos,
                        viewport_layout_.size,
                        glm::vec2(viewport_overlay_input.mouse_x,
                                  viewport_overlay_input.mouse_y))) {
                    if (auto* const input_controller = viewer_->getInputController()) {
                        input_controller->setFocusedSplitPanel(target_panel->panel);
                    } else {
                        rendering->setFocusedSplitPanel(target_panel->panel);
                    }
                }
            }
        }
        if (!startup_plugin_preload_running &&
            lfs::python::has_python_hooks("viewport_overlay", "draw")) {
            LOG_TIMER_THRESHOLD("gui_render.viewport_overlay.python_hooks", 0.25);
            lfs::python::invoke_python_hooks("viewport_overlay", "draw", true);
            lfs::python::invoke_python_hooks("viewport_overlay", "draw", false);
        }

        lfs::rendering::ScreenOverlayRenderer* overlay_renderer = nullptr;
        if (auto* const rendering = viewer_ ? viewer_->getRenderingManager() : nullptr) {
            overlay_renderer = rendering->getScreenOverlayRenderer();
        }
        if (overlay_renderer && has_viewport_overlay_panels) {
            LOG_TIMER_THRESHOLD("gui_render.screen_overlay_renderer.beginFrame", 0.25);
            overlay_renderer->beginFrame();
        }

        if (has_viewport_overlay_panels) {
            LOG_TIMER_THRESHOLD("gui_render.draw_panels.ViewportOverlay", 0.25);
            reg.draw_panels(PanelSpace::ViewportOverlay, draw_ctx);
        }

        if (overlay_renderer && has_viewport_overlay_panels) {
            LOG_TIMER_THRESHOLD("gui_render.screen_overlay_renderer.endFrame", 0.25);
            overlay_renderer->endFrame();
        }

        publish_vram_hud_overlay_if_due();
        {
            LOG_TIMER_THRESHOLD("gui_render.rml_viewport_overlay.render", 0.10);
            rml_viewport_overlay_.renderCached();
        }

        PanelInputState floating_input = panel_input;
        floating_input.bg_draw_list = ImGui::GetForegroundDrawList(ImGui::GetMainViewport());
        panel_setup_timer.reset();
        if (has_floating_panels) {
            LOG_TIMER_THRESHOLD("gui_render.draw_panels.Floating", 0.25);
            reg.draw_panels(PanelSpace::Floating, draw_ctx, &floating_input);
        }

        applyFrameInputCapture(&rml_right_panel_);

        if (!ui_hidden_) {
            LOG_TIMER_THRESHOLD("gui_render.status_bar_and_StatusBar", 0.10);
            const float status_bar_height =
                PanelLayoutManager::STATUS_BAR_HEIGHT * lfs::python::get_shared_dpi_scale();
            const float status_bar_x = screen.work_pos.x;
            const float status_bar_y = screen.work_pos.y + screen.work_size.y - status_bar_height;
            const float status_bar_w = screen.work_size.x;
            const bool status_input =
                !block_underlay_input &&
                ((panel_input.mouse_x >= status_bar_x &&
                  panel_input.mouse_x < status_bar_x + status_bar_w &&
                  panel_input.mouse_y >= status_bar_y &&
                  panel_input.mouse_y < status_bar_y + status_bar_height) ||
                 panel_input.mouse_released[0]);
            if (status_input) {
                rml_status_bar_.processInput(panel_input, status_bar_x, status_bar_y,
                                             status_bar_w, status_bar_height);
            }
            if (status_input) {
                rml_status_bar_.render(draw_ctx,
                                       status_bar_x,
                                       status_bar_y,
                                       status_bar_w,
                                       status_bar_height,
                                       panel_input.screen_w,
                                       panel_input.screen_h);
            } else {
                rml_status_bar_.renderCached(draw_ctx,
                                             status_bar_x,
                                             status_bar_y,
                                             status_bar_w,
                                             status_bar_height,
                                             panel_input.screen_w,
                                             panel_input.screen_h);
            }
            if (has_status_bar_panels)
                reg.draw_panels(PanelSpace::StatusBar, draw_ctx, &panel_input);
        }

        if (!startup_plugin_preload_running && python::has_python_modals()) {
            LOG_TIMER("gui_render.python_modals_and_popups");
            python::draw_python_modals(scene);
        }
        if (!startup_plugin_preload_running && python::has_python_popups()) {
            LOG_TIMER("gui_render.python_popups");
            python::draw_python_popups(scene);
        }

        if (rml_modal_overlay_->isOpen()) {
            LOG_TIMER_THRESHOLD("gui_render.rml_modal_processInput", 0.25);
            rml_modal_overlay_->processInput(raw_panel_input);
        }
        const bool window_resize_active =
            viewer_ &&
            viewer_->getWindowManager() &&
            viewer_->getWindowManager()->manualResizeEdgeMask() != 0;
        if (!window_resize_active) {
            if (ImGui::GetMouseCursor() == ImGuiMouseCursor_Arrow)
                applyRmlCursorRequest(rmlui_manager_.consumeCursorRequest());
            apply_cursor(rml_right_panel_.getCursorRequest());
            apply_cursor(panel_layout_.getCursorRequest());
            if (SDL_Cursor* const cursor = systemCursorForImGuiCursor(ImGui::GetMouseCursor()))
                SDL_SetCursor(cursor);
            if (auto* input_controller = viewer_->getInputController())
                input_controller->applySplitterCursorOverride();
        } else if (auto* const wm = viewer_->getWindowManager()) {
            wm->refreshResizeCursor();
        }
        syncWindowTextInput(viewer_->getWindow());

        if (vulkan_gui_) {
            LOG_TIMER_THRESHOLD("gui_render.menu_context_modal_render", 0.25);
            if (menu_bar_) {
                LOG_TIMER_THRESHOLD("gui_render.menu_context_modal_render.menu_bar", 0.25);
                rml_menu_bar_.setUiHidden(ui_hidden_);
                rml_menu_bar_.setViewportRightEdge(menu_toolbar_right_edge_ - panel_input.screen_x);
                rml_menu_bar_.draw(panel_input.screen_w, panel_input.screen_h);
            }
            if (global_context_menu_->hasPendingRenderWork()) {
                LOG_TIMER_THRESHOLD("gui_render.menu_context_modal_render.context_menu", 0.25);
                global_context_menu_->render(panel_input.screen_w, panel_input.screen_h,
                                             panel_input.screen_x, panel_input.screen_y);
            }
            if (rml_toast_overlay_ && rml_toast_overlay_->hasPendingRenderWork()) {
                LOG_TIMER_THRESHOLD("gui_render.menu_context_modal_render.toast_overlay", 0.25);
                rml_toast_overlay_->render(panel_input.screen_w,
                                           panel_input.screen_h,
                                           panel_input.screen_x,
                                           panel_input.screen_y,
                                           viewport_layout_.pos.x,
                                           viewport_layout_.pos.y,
                                           viewport_layout_.size.x,
                                           viewport_layout_.size.y);
            }
            if (rml_modal_overlay_->hasPendingRenderWork()) {
                LOG_TIMER_THRESHOLD("gui_render.menu_context_modal_render.modal_overlay", 0.25);
                rml_modal_overlay_->render(panel_input.screen_w,
                                           panel_input.screen_h,
                                           panel_input.screen_x,
                                           panel_input.screen_y,
                                           viewport_layout_.pos.x,
                                           viewport_layout_.pos.y,
                                           viewport_layout_.size.x,
                                           viewport_layout_.size.y);
            }
        }

        // Was ImGui::Render(): the resulting ImDrawData was never submitted (no
        // ImGui_ImplVulkan_RenderDrawData consumer), so building it was pure CPU waste.
        // We still call EndFrame to keep the per-frame state machine balanced for the
        // panels that exercise ImGui internally (py_ui, ui_widgets, theme, etc.).
        {
            LOG_TIMER_THRESHOLD("gui_render.imgui_EndFrame", 0.25);
            ImGui::EndFrame();
        }

        if (vulkan_gui_) {
            guiFocusState().any_item_active |= ImGui::IsAnyItemActive() || rmlui_manager_.anyItemActive();

            const auto& bg = lfs::vis::theme().menu_background();
            VkClearValue clear_value{};
            clear_value.color = VkClearColorValue{{bg.x, bg.y, bg.z, 1.0f}};

            bool interop_prepare_ok = true;
            if (vulkan_context && !isViewportExportLocked()) {
                LOG_TIMER_THRESHOLD("gui_render.prepareVulkanSceneInterop", 0.25);
                try {
                    prepareVulkanSceneInterop(*vulkan_context);
                    prepareVulkanSplitRightInterop(*vulkan_context);
                    prepareVulkanDepthBlitInterop(*vulkan_context);
                } catch (const std::exception& error) {
                    interop_prepare_ok = false;
                    LOG_ERROR("Skipping Vulkan GUI frame after CUDA/Vulkan interop failure: {}",
                              error.what());
                } catch (...) {
                    interop_prepare_ok = false;
                    LOG_ERROR("Skipping Vulkan GUI frame after unknown CUDA/Vulkan interop failure");
                }
            }

            VulkanContext::Frame frame{};
            bool begin_ok = false;
            {
                cpu_ui_before_vulkan_timer.reset();
                LOG_TIMER("frame_pacing.vulkan_beginFrame");
                begin_ok = interop_prepare_ok && vulkan_context &&
                           vulkan_context->beginFrame(clear_value, frame);
            }
            if (begin_ok) {
                if (vulkan_frame_completion_semaphore_ != VK_NULL_HANDLE &&
                    vulkan_frame_completion_value_ != 0) {
                    LOG_TIMER_THRESHOLD("gui_render.vksplat_completion_wait_submit", 0.25);
                    // VkSplat color/split/depth outputs are first consumed only by
                    // fragment sampling in the viewport pass graph. Earlier graphics
                    // work can proceed while the async compute submission finishes.
                    vulkan_context->addFrameTimelineWait(vulkan_frame_completion_semaphore_,
                                                         vulkan_frame_completion_value_,
                                                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
                }
                VulkanViewportPassParams viewport_params{};
                {
                    LOG_TIMER_THRESHOLD("gui_render.buildVulkanViewportParams", 0.25);
                    viewport_params = buildVulkanViewportParams(frame.extent, frame.frame_slot);
                }
                bool viewport_pass_ready = false;
                if (!vulkan_viewport_pass_) {
                    vulkan_viewport_pass_ = std::make_unique<VulkanViewportPass>();
                }
                viewport_pass_ready = vulkan_viewport_pass_->init(*vulkan_context);
                if (viewport_pass_ready) {
                    LOG_TIMER_THRESHOLD("gui_render.viewport_pass_prepare_record", 0.25);
                    vulkan_viewport_pass_->prepare(*vulkan_context, viewport_params);
                    recordVulkanViewport(frame.command_buffer, frame.extent, viewport_params);
                }
                {
                    LOG_TIMER("gui_render.rmlui_record");
                    if (rmlui_manager_.beginVulkanFrame(frame.command_buffer,
                                                        frame.extent,
                                                        frame.swapchain_image,
                                                        frame.swapchain_image_view,
                                                        frame.depth_stencil_image_view,
                                                        frame.frame_slot)) {
                        {
                            LOG_TIMER_THRESHOLD("gui_render.rmlui_record.background", 0.25);
                            rmlui_manager_.renderQueuedVulkanContexts(false);
                        }
                        {
                            LOG_TIMER_THRESHOLD("gui_render.rmlui_record.foreground", 0.25);
                            rmlui_manager_.renderQueuedVulkanContexts(true);
                        }
                        rmlui_manager_.endVulkanFrame();
                    } else {
                        rmlui_manager_.clearVulkanQueue();
                    }
                }
                if (viewer_) {
                    viewer_->processRenderWorkQueue();
                }
                // Synchronous full-window capture explicitly submits and consumes
                // the active frame before returning its readback.
                if (vulkan_context->hasActiveFrame()) {
                    LOG_TIMER("frame_pacing.vulkan_endFrame_present");
                    if (!vulkan_context->endFrame()) {
                        LOG_WARN("Vulkan GUI frame present failed: {}", vulkan_context->lastError());
                    }
                }
            } else if (vulkan_context) {
                rmlui_manager_.clearVulkanQueue();
                clearLineRendererCommands();
                if (!vulkan_context->lastError().empty()) {
                    LOG_WARN("Vulkan GUI frame begin failed: {} (ui_hidden={}, fullscreen_pending={}, ui_pending={}, settling={}, resume_training_pending={})",
                             vulkan_context->lastError(),
                             ui_hidden_,
                             fullscreen_toggle_pending_,
                             ui_toggle_pending_,
                             isInteractiveTransitionSettling(),
                             interactive_transition_resume_training_);
                }
                if (viewer_) {
                    viewer_->processRenderWorkQueue();
                }
            }

            if (!ui_layout_changed && ui_layout_settle_frames_ > 0)
                --ui_layout_settle_frames_;

            persistImGuiSettingsIfNeeded();
            updateInteractiveTransitionGuard();
            return;
        }

        if (!vulkan_gui_ && viewer_) {
            viewer_->processRenderWorkQueue();
        }
        updateInteractiveTransitionGuard();
    }

    void GuiManager::renderSelectionOverlays(const UIContext& ctx) {
        if (isViewportExportLocked()) {
            return;
        }

        if (auto* const tool = ctx.viewer->getSelectionTool(); tool && tool->isEnabled() && !ui_hidden_) {
            tool->renderUI(ctx, nullptr);
        }

        // Node rectangle-drag outline. Uses the same ScreenOverlayRenderer path
        // as the cursor preview / align tool so it stays consistent with the
        // other native viewport overlays.
        if (auto* const ic = ctx.viewer->getInputController();
            !ui_hidden_ && ic && ic->isNodeRectDragging()) {
            if (auto* const rm = ctx.viewer->getRenderingManager()) {
                if (auto* const overlay = rm->getScreenOverlayRenderer();
                    overlay && overlay->isFrameActive()) {
                    const glm::vec2 s = ic->getNodeRectStart();
                    const glm::vec2 e = ic->getNodeRectEnd();
                    const glm::vec2 tl{std::min(s.x, e.x), std::min(s.y, e.y)};
                    const glm::vec2 br{std::max(s.x, e.x), std::max(s.y, e.y)};
                    if (br.x - tl.x > 0.5f && br.y - tl.y > 0.5f) {
                        const auto& warn = theme().palette.warning;
                        const lfs::rendering::OverlayColor fill{warn.x, warn.y, warn.z, warn.w * 0.15f};
                        const lfs::rendering::OverlayColor stroke{warn.x, warn.y, warn.z, warn.w * 0.85f};
                        overlay->addRectFilled(tl, br, fill);
                        overlay->addRect(tl, br, stroke, 2.0f);
                    }
                }
            }
        }

        const bool mouse_over_ui = guiFocusState().want_capture_mouse;
        if (!ui_hidden_ && !mouse_over_ui && viewport_layout_.size.x > 0 && viewport_layout_.size.y > 0) {
            auto* rm = ctx.viewer->getRenderingManager();
            lfs::rendering::ScreenOverlayRenderer* overlay = nullptr;
            if (rm) {
                overlay = rm->getScreenOverlayRenderer();
            }
            if (!overlay || !overlay->isFrameActive()) {
                return;
            }
            const auto toCol = [](const ImVec4& c, float a) {
                return lfs::rendering::OverlayColor{c.x, c.y, c.z, a};
            };
            const auto toCol4 = [](const ImVec4& c) {
                return lfs::rendering::OverlayColor{c.x, c.y, c.z, c.w};
            };
            const glm::ivec2 rendered_size = rm ? rm->getRenderedSize() : glm::ivec2(0);
            struct PreviewPanelContext {
                float x = 0.0f;
                float y = 0.0f;
                float width = 0.0f;
                float height = 0.0f;
                int render_width = 0;
                int render_height = 0;
                const Viewport* viewport = nullptr;
            };
            const auto resolve_preview_panel = [&](const std::optional<SplitViewPanelId> panel) {
                PreviewPanelContext panel_ctx{
                    .x = viewport_layout_.pos.x,
                    .y = viewport_layout_.pos.y,
                    .width = viewport_layout_.size.x,
                    .height = viewport_layout_.size.y,
                    .render_width =
                        rendered_size.x > 0 ? rendered_size.x : static_cast<int>(ctx.viewer->getViewport().windowSize.x),
                    .render_height =
                        rendered_size.y > 0 ? rendered_size.y : static_cast<int>(ctx.viewer->getViewport().windowSize.y),
                    .viewport = &ctx.viewer->getViewport(),
                };
                if (!rm || !panel || !rm->isIndependentSplitViewActive()) {
                    return panel_ctx;
                }

                const auto info = rm->resolveViewerPanel(
                    ctx.viewer->getViewport(),
                    {viewport_layout_.pos.x, viewport_layout_.pos.y},
                    {viewport_layout_.size.x, viewport_layout_.size.y},
                    std::nullopt,
                    panel);
                if (!info) {
                    return panel_ctx;
                }

                panel_ctx.x = info->x;
                panel_ctx.y = info->y;
                panel_ctx.width = info->width;
                panel_ctx.height = info->height;
                panel_ctx.render_width = info->render_width;
                panel_ctx.render_height = info->render_height;
                panel_ctx.viewport = info->viewport;
                return panel_ctx;
            };
            const auto render_to_screen = [&](const PreviewPanelContext& panel_ctx, const float x, const float y) {
                const float render_to_screen_x =
                    (panel_ctx.render_width > 0)
                        ? (panel_ctx.width / static_cast<float>(panel_ctx.render_width))
                        : (1.0f / std::max(rm ? rm->getSettings().render_scale : 1.0f, 0.001f));
                const float render_to_screen_y =
                    (panel_ctx.render_height > 0)
                        ? (panel_ctx.height / static_cast<float>(panel_ctx.render_height))
                        : (1.0f / std::max(rm ? rm->getSettings().render_scale : 1.0f, 0.001f));
                return glm::vec2(panel_ctx.x + x * render_to_screen_x,
                                 panel_ctx.y + y * render_to_screen_y);
            };
            // Keep preview overlays inside the live viewport region so docked panels stay in front.
            const auto push_preview_clip = [&](const PreviewPanelContext& panel_ctx) {
                const glm::vec2 clip_min(panel_ctx.x, panel_ctx.y);
                float clip_bottom = panel_ctx.y + panel_ctx.height;
                const float bottom_dock_top = panel_layout_.bottomDockTopY();
                if (bottom_dock_top > 0.0f) {
                    clip_bottom = std::min(clip_bottom, bottom_dock_top);
                }

                const glm::vec2 clip_max(panel_ctx.x + panel_ctx.width, clip_bottom);
                if (clip_max.x <= clip_min.x || clip_max.y <= clip_min.y) {
                    return false;
                }

                overlay->pushClipRect(clip_min, clip_max, true);
                return true;
            };

            if (rm && rm->isCursorPreviewActive()) {
                const auto& t = theme();
                float bx, by, br;
                bool add_mode;
                rm->getCursorPreviewState(bx, by, br, add_mode);
                const auto panel_ctx = resolve_preview_panel(rm->getCursorPreviewPanel());

                const glm::vec2 screen_pos = render_to_screen(panel_ctx, bx, by);
                const float screen_radius =
                    (panel_ctx.render_width > 0)
                        ? br * (panel_ctx.width / static_cast<float>(panel_ctx.render_width))
                        : br;

                const auto brush_color = add_mode ? toCol(t.palette.success, 0.8f)
                                                  : toCol(t.palette.error, 0.8f);
                if (push_preview_clip(panel_ctx)) {
                    overlay->addCircle(screen_pos, screen_radius, brush_color, 32, 2.0f);
                    overlay->addCircleFilled(screen_pos, 3.0f, brush_color);
                    overlay->popClipRect();
                }
            }

            if (rm && rm->isRectPreviewActive()) {
                const auto& t = theme();
                float rx0, ry0, rx1, ry1;
                bool add_mode;
                rm->getRectPreview(rx0, ry0, rx1, ry1, add_mode);
                const auto panel_ctx = resolve_preview_panel(rm->getRectPreviewPanel());

                const glm::vec2 p0 = render_to_screen(panel_ctx, rx0, ry0);
                glm::vec2 p1 = render_to_screen(panel_ctx, rx1, ry1);
                if (rm->rectPreviewTracksCursor() && s_frame_input) {
                    p1 = {s_frame_input->mouse_x, s_frame_input->mouse_y};
                }

                const auto fill_color = add_mode ? toCol(t.palette.success, 0.15f)
                                                 : toCol(t.palette.error, 0.15f);
                const auto border_color = add_mode ? toCol(t.palette.success, 0.8f)
                                                   : toCol(t.palette.error, 0.8f);

                if (push_preview_clip(panel_ctx)) {
                    overlay->addRectFilled(p0, p1, fill_color);
                    overlay->addRect(p0, p1, border_color, 2.0f);
                    overlay->popClipRect();
                }
            }

            if (rm && rm->isPolygonPreviewActive()) {
                const auto& t = theme();
                const auto& points = rm->getPolygonPoints();
                const auto& world_points = rm->getPolygonWorldPoints();
                const bool closed = rm->isPolygonClosed();
                const bool add_mode = rm->isPolygonAddMode();
                const auto panel_ctx = resolve_preview_panel(rm->getPolygonPreviewPanel());

                if (!points.empty() || !world_points.empty()) {
                    const auto line_color = add_mode ? toCol(t.palette.success, 0.8f)
                                                     : toCol(t.palette.error, 0.8f);
                    const auto fill_color = add_mode ? toCol(t.palette.success, 0.15f)
                                                     : toCol(t.palette.error, 0.15f);
                    const auto vertex_color = toCol4(t.palette.warning);
                    const auto vertex_hover_color = toCol4(t.palette.success);
                    const auto close_hint_color = toCol(t.palette.success, 0.78f);
                    const auto line_to_mouse_color = add_mode
                                                         ? toCol(t.palette.success, 0.5f)
                                                         : toCol(t.palette.error, 0.5f);

                    std::vector<glm::vec2> screen_points;
                    if (rm->isPolygonPreviewWorldSpace()) {
                        const auto render_settings = rm->getSettings();
                        screen_points.reserve(world_points.size());

                        if (!panel_ctx.viewport) {
                            screen_points.clear();
                        }
                        Viewport projection_viewport = panel_ctx.viewport ? *panel_ctx.viewport : ctx.viewer->getViewport();
                        projection_viewport.windowSize = {std::max(panel_ctx.render_width, 1),
                                                          std::max(panel_ctx.render_height, 1)};

                        bool all_visible = true;
                        for (const auto& world_point : world_points) {
                            const auto projected = lfs::rendering::projectWorldPoint(
                                projection_viewport.camera.R,
                                projection_viewport.camera.t,
                                projection_viewport.windowSize,
                                world_point,
                                render_settings.focal_length_mm,
                                render_settings.orthographic,
                                render_settings.ortho_scale);
                            if (!projected) {
                                all_visible = false;
                                break;
                            }

                            screen_points.emplace_back(
                                panel_ctx.x + projected->x * (panel_ctx.width / static_cast<float>(projection_viewport.windowSize.x)),
                                panel_ctx.y + projected->y * (panel_ctx.height / static_cast<float>(projection_viewport.windowSize.y)));
                        }

                        if (!all_visible) {
                            screen_points.clear();
                        }
                    } else {
                        screen_points.reserve(points.size());
                        for (const auto& [px, py] : points) {
                            screen_points.push_back(render_to_screen(panel_ctx, px, py));
                        }
                    }

                    if (push_preview_clip(panel_ctx)) {
                        if (closed && screen_points.size() >= 3) {
                            overlay->addConvexPolyFilled(screen_points, fill_color);
                        }

                        for (size_t i = 0; i + 1 < screen_points.size(); ++i) {
                            overlay->addLine(screen_points[i], screen_points[i + 1], line_color, 2.0f);
                        }
                        if (closed && screen_points.size() >= 3) {
                            overlay->addLine(screen_points.back(), screen_points.front(), line_color, 2.0f);
                        }

                        const glm::vec2 mouse_pos =
                            s_frame_input
                                ? glm::vec2(s_frame_input->mouse_x, s_frame_input->mouse_y)
                                : glm::vec2(viewport_layout_.pos.x, viewport_layout_.pos.y);
                        constexpr float CLOSE_THRESHOLD = 12.0f;
                        constexpr float VERTEX_RADIUS = 5.0f;
                        const auto distance_sq = [](const glm::vec2 a, const glm::vec2 b) {
                            const float dx = a.x - b.x;
                            const float dy = a.y - b.y;
                            return dx * dx + dy * dy;
                        };
                        const bool can_close = !closed && screen_points.size() >= 3 &&
                                               distance_sq(mouse_pos, screen_points.front()) <
                                                   CLOSE_THRESHOLD * CLOSE_THRESHOLD;
                        int hovered_idx = -1;
                        for (size_t i = 0; i < screen_points.size(); ++i) {
                            if (distance_sq(mouse_pos, screen_points[i]) <= VERTEX_RADIUS * VERTEX_RADIUS) {
                                hovered_idx = static_cast<int>(i);
                                break;
                            }
                        }

                        if (!closed && !screen_points.empty()) {
                            overlay->addLine(screen_points.back(), mouse_pos, line_to_mouse_color, 1.0f);

                            if (can_close) {
                                overlay->addCircle(screen_points.front(), 9.0f, close_hint_color, 16, 2.0f);
                            }
                        }

                        for (size_t i = 0; i < screen_points.size(); ++i) {
                            const auto color = (static_cast<int>(i) == hovered_idx || (can_close && i == 0))
                                                   ? vertex_hover_color
                                                   : vertex_color;
                            overlay->addCircleFilled(screen_points[i], VERTEX_RADIUS, color);
                            overlay->addCircle(screen_points[i], VERTEX_RADIUS, line_color, 16, 1.5f);
                        }

                        if (!screen_points.empty()) {
                            const float initial_ring_radius = can_close ? 9.0f : 8.0f;
                            const float initial_ring_thickness = can_close ? 2.0f : 1.5f;
                            overlay->addCircle(screen_points.front(), initial_ring_radius,
                                               close_hint_color, 24, initial_ring_thickness);
                        }

                        if (closed && screen_points.size() >= 3) {
                            float cx = 0.0f, cy = 0.0f;
                            for (const auto& sp : screen_points) {
                                cx += sp.x;
                                cy += sp.y;
                            }
                            cx /= static_cast<float>(screen_points.size());
                            cy /= static_cast<float>(screen_points.size());

                            constexpr std::string_view hint_lines[] = {
                                "Enter to confirm",
                                "Shift-click edge: add",
                                "Ctrl-click vertex: remove"};
                            const float font_px = theme().fonts.small_size;
                            float max_w = 0.0f;
                            for (const auto& line : hint_lines) {
                                max_w = std::max(max_w, overlay->measureText(line, font_px).x);
                            }
                            const float line_h = font_px * 1.2f;
                            const float total_h = line_h * static_cast<float>(std::size(hint_lines));
                            const float start_x = cx - max_w * 0.5f;
                            const float start_y = cy - total_h * 0.5f;
                            const auto text_color = toCol(t.palette.text, 0.9f);
                            for (size_t i = 0; i < std::size(hint_lines); ++i) {
                                overlay->addText({start_x, start_y + line_h * static_cast<float>(i)},
                                                 hint_lines[i], text_color, font_px);
                            }
                        }

                        overlay->popClipRect();
                    }
                }
            }

            if (rm && rm->isLassoPreviewActive()) {
                const auto& t = theme();
                const auto& points = rm->getLassoPoints();
                const bool add_mode = rm->isLassoAddMode();
                const auto panel_ctx = resolve_preview_panel(rm->getLassoPreviewPanel());

                if (points.size() >= 2) {
                    const auto line_color = add_mode ? toCol(t.palette.success, 0.8f)
                                                     : toCol(t.palette.error, 0.8f);

                    if (push_preview_clip(panel_ctx)) {
                        glm::vec2 prev = render_to_screen(panel_ctx, points[0].first, points[0].second);
                        for (size_t i = 1; i < points.size(); ++i) {
                            const glm::vec2 curr = render_to_screen(panel_ctx, points[i].first, points[i].second);
                            overlay->addLine(prev, curr, line_color, 2.0f);
                            prev = curr;
                        }
                        if (rm->lassoPreviewTracksCursor() && s_frame_input) {
                            overlay->addLine(prev, {s_frame_input->mouse_x, s_frame_input->mouse_y},
                                             line_color, 2.0f);
                        }
                        overlay->popClipRect();
                    }
                }
            }
        }

        auto* align_tool = ctx.viewer->getAlignTool();
        if (align_tool && align_tool->isEnabled() && !ui_hidden_) {
            align_tool->renderUI(ctx, nullptr);
        }
    }

    void GuiManager::renderViewportDecorations() {
        if (!ui_hidden_ && viewport_layout_.size.x > 0 && viewport_layout_.size.y > 0) {
            const auto& t = theme();
            const float r = t.viewport.corner_radius;
            if (r > 0.0f) {
                auto* const dl = ImGui::GetBackgroundDrawList();
                const ImU32 bg = toU32(t.menu_background());
                const float x1 = viewport_layout_.pos.x, y1 = viewport_layout_.pos.y;
                const float x2 = x1 + viewport_layout_.size.x, y2 = y1 + viewport_layout_.size.y;

                constexpr int CORNER_ARC_SEGMENTS = 12;
                const auto maskCorner = [&](const ImVec2 corner, const ImVec2 edge,
                                            const ImVec2 center, const float a0, const float a1) {
                    dl->PathLineTo(corner);
                    dl->PathLineTo(edge);
                    dl->PathArcTo(center, r, a0, a1, CORNER_ARC_SEGMENTS);
                    dl->PathLineTo(corner);
                    dl->PathFillConvex(bg);
                };
                maskCorner({x1, y1}, {x1, y1 + r}, {x1 + r, y1 + r}, IM_PI, IM_PI * 1.5f);
                maskCorner({x2, y1}, {x2 - r, y1}, {x2 - r, y1 + r}, IM_PI * 1.5f, IM_PI * 2.0f);
                maskCorner({x1, y2}, {x1 + r, y2}, {x1 + r, y2 - r}, IM_PI * 0.5f, IM_PI);
                maskCorner({x2, y2}, {x2, y2 - r}, {x2 - r, y2 - r}, 0.0f, IM_PI * 0.5f);

                if (show_main_panel_) {
                    const float rpw = panel_layout_.getRightPanelWidth();
                    auto* mvp = ImGui::GetMainViewport();
                    const float px = mvp->WorkPos.x + mvp->WorkSize.x - rpw;
                    const float py1 = mvp->WorkPos.y;
                    const float py2 = py1 + mvp->WorkSize.y - PanelLayoutManager::STATUS_BAR_HEIGHT * current_ui_scale_;
                    maskCorner({px, py1}, {px, py1 + r}, {px + r, py1 + r}, IM_PI, IM_PI * 1.5f);
                    maskCorner({px, py2}, {px + r, py2}, {px + r, py2 - r}, IM_PI * 0.5f, IM_PI);
                }

                if (t.viewport.border_size > 0.0f) {
                    dl->AddRect({x1, y1}, {x2, y2}, t.viewport_border_u32(), r,
                                ImDrawFlags_RoundCornersAll, t.viewport.border_size);
                }
            }
        }
    }

    void GuiManager::updateInputOverrides(const PanelInputState& input,
                                          bool mouse_in_viewport) {
        if (rml_menu_bar_.wantsInput())
            return;

        auto& focus = guiFocusState();
        const bool any_popup_or_modal_open =
            ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel) ||
            isModalWindowOpen() ||
            (global_context_menu_ && global_context_menu_->isOpen());
        const bool imgui_wants_input = focus.want_text_input || focus.want_capture_keyboard;

        if (isTransformGizmoOverOrUsing() && !any_popup_or_modal_open) {
            focus.want_capture_mouse = false;
            focus.want_capture_keyboard = false;
        }

        if (mouse_in_viewport && !ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow) &&
            !any_popup_or_modal_open && !imgui_wants_input) {
            if (input.mouse_down[1] || input.mouse_down[2]) {
                focus.want_capture_mouse = false;
            }
            if (input.mouse_clicked[0] || input.mouse_clicked[1]) {
                ImGui::ClearActiveID();
                focus.want_capture_keyboard = false;
                auto* console_state = panels::PythonConsoleState::tryGetInstance();
                if (console_state != nullptr) {
                    auto* editor = console_state->getEditor();
                    if (editor != nullptr) {
                        editor->unfocus();
                    }
                }
            }
        }

        auto* rendering_manager = viewer_->getRenderingManager();
        if (rendering_manager) {
            const auto& settings = rendering_manager->getSettings();
            if (settings.point_cloud_mode && mouse_in_viewport &&
                !ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow) &&
                !any_popup_or_modal_open && !imgui_wants_input) {
                focus.want_capture_mouse = false;
                focus.want_capture_keyboard = false;
            }
        }
    }

    glm::vec2 GuiManager::getViewportPos() const {
        return viewport_layout_.pos;
    }

    glm::vec2 GuiManager::getViewportSize() const {
        return viewport_layout_.size;
    }

    bool GuiManager::isViewportFocused() const {
        return viewport_layout_.has_focus;
    }

    bool GuiManager::isPositionInViewport(double x, double y) const {
        return (x >= viewport_layout_.pos.x &&
                x < viewport_layout_.pos.x + viewport_layout_.size.x &&
                y >= viewport_layout_.pos.y &&
                y < viewport_layout_.pos.y + viewport_layout_.size.y);
    }

    bool GuiManager::isPositionOverFloatingPanel(const double x, const double y) const {
        return PanelRegistry::instance().isPositionOverFloatingPanel(x, y);
    }

    GuiHitTestResult GuiManager::hitTestPointer(const double x, const double y) const {
        if (isCapturingInput() || isModalWindowOpen() || startup_overlay_.blocksUnderlayInput() ||
            (global_context_menu_ && global_context_menu_->isOpen())) {
            return {.blocks_pointer = true, .takes_keyboard_focus = true};
        }

        if (rml_menu_bar_.isOpen()) {
            return {.blocks_pointer = true, .takes_keyboard_focus = true};
        }

        if (isViewportExportLocked() && isPositionInViewport(x, y)) {
            return {.blocks_pointer = true, .takes_keyboard_focus = true};
        }

        if (rmlui_manager_.activeOverlayContainsPoint(static_cast<float>(x),
                                                      static_cast<float>(y))) {
            return {.blocks_pointer = true, .takes_keyboard_focus = true};
        }

        if (panel_layout_.isResizingPanel() || isPositionOverFloatingPanel(x, y)) {
            return {.blocks_pointer = true, .takes_keyboard_focus = true};
        }

        if (sequencer_ui_.blocksPointer(x, y) || rml_viewport_overlay_.blocksPointer(x, y)) {
            return {.blocks_pointer = true, .takes_keyboard_focus = true};
        }

        if (ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow)) {
            return {.blocks_pointer = true, .takes_keyboard_focus = true};
        }

        return {};
    }

    GuiInputState GuiManager::inputState() const {
        const auto& focus = guiFocusState();
        const bool modal_open =
            isCapturingInput() ||
            isModalWindowOpen() ||
            startup_overlay_.blocksUnderlayInput() ||
            (global_context_menu_ && global_context_menu_->isOpen()) ||
            isViewportExportLocked() ||
            sequencer_ui_.blocksKeyboard();

        return {
            .has_keyboard_focus = focus.any_item_active || focus.want_capture_keyboard,
            .text_input_active = focus.want_text_input,
            .modal_open = modal_open,
        };
    }

    void GuiManager::setupEventHandlers() {
        using namespace lfs::core::events;

        error_consumer_ = std::make_unique<GuiErrorConsumer>(GuiErrorConsumer::Sinks{
            .modal = [this](lfs::core::ModalRequest req) { enqueueModal(std::move(req)); },
            .toast = [this](ToastRequest req) { enqueueToast(std::move(req)); },
            .status = [this](std::string text, ErrorNoticeLevel level) {
                rml_status_bar_.postStatusMessage(std::move(text), level);
                if (auto* const window_manager = viewer_ ? viewer_->getWindowManager() : nullptr)
                    window_manager->wakeEventLoop(); },
        });
        error_subscription_ = lfs::ErrorBus::instance().subscribe(*error_consumer_);
        registerErrorEventBridge();

        ui::FileDropReceived::when([this](const auto&) {
            if (startup_overlay_.isPluginLoadComplete())
                startup_overlay_.dismiss();
            drag_drop_.resetHovering();
        });

        cmd::ShowWindow::when([this](const auto& e) {
            showWindow(e.window_name, e.show);
        });

        cmd::ShowVideoExtractor::when([this](const auto& e) {
            auto& panels = PanelRegistry::instance();
            panels.set_panel_enabled("native.video_extractor", true);
            panels.bring_panel_to_front("native.video_extractor");
            if (!video_widget_) {
                LOG_ERROR("Video extractor widget is not available");
                return;
            }
            if (!video_widget_->openVideoPath(e.video_path)) {
                LOG_WARN("Failed to open dropped video in extractor: {}",
                         lfs::core::path_to_utf8(e.video_path));
            }
        });

        cmd::GoToCamView::when([this](const auto& e) {
            if (auto* sm = viewer_->getSceneManager()) {
                const auto& scene = sm->getScene();
                for (const auto* node : scene.getNodes()) {
                    if (node->type == core::NodeType::CAMERA && node->camera_uid == e.cam_id) {
                        ui::NodeSelected{.path = node->name, .type = "Camera", .metadata = {}}.emit();
                        break;
                    }
                }
            }
        });

        ui::FocusTrainingPanel::when([this](const auto&) {
            focus_panel_name_ = "Training";
        });

        ui::ToggleUI::when([this](const auto&) {
            queueUiVisibilityToggle();
        });

        ui::ToggleVramHud::when([this](const auto&) {
            show_vram_hud_ = !show_vram_hud_;
            next_vram_hud_publish_ = {};
        });

        ui::ToggleFullscreen::when([this](const auto&) {
            queueFullscreenToggle();
        });

        internal::DisplayScaleChanged::when([this](const auto& e) {
            if (lfs::vis::loadUiScalePreference() <= 0.0f) {
                pending_ui_scale_ = std::clamp(e.scale, 1.0f, 4.0f);
            }
        });

        internal::UiScaleChangeRequested::when([this](const auto& e) {
            if (e.scale <= 0.0f) {
                pending_ui_scale_ = std::clamp(SDL_GetWindowDisplayScale(viewer_->getWindow()), 1.0f, 4.0f);
            } else {
                pending_ui_scale_ = std::clamp(e.scale, 1.0f, 4.0f);
            }
        });

        state::DiskSpaceSaveFailed::when([this](const auto& e) {
            using namespace lichtfeld::Strings;
            if (!e.is_disk_space_error)
                return;

            auto formatBytes = [](size_t bytes) -> std::string {
                constexpr double KB = 1024.0;
                constexpr double MB = KB * 1024.0;
                constexpr double GB = MB * 1024.0;
                if (bytes >= static_cast<size_t>(GB))
                    return std::format("{:.2f} GB", static_cast<double>(bytes) / GB);
                if (bytes >= static_cast<size_t>(MB))
                    return std::format("{:.2f} MB", static_cast<double>(bytes) / MB);
                if (bytes >= static_cast<size_t>(KB))
                    return std::format("{:.2f} KB", static_cast<double>(bytes) / KB);
                return std::format("{} bytes", bytes);
            };

            const std::string subtitle = e.is_checkpoint
                                             ? std::format("{} {})", LOC(DiskSpaceDialog::CHECKPOINT_SAVE_FAILED), e.iteration)
                                             : std::string(LOC(DiskSpaceDialog::EXPORT_FAILED));

            std::string body;
            body += std::format("<div>{}</div>", LOC(DiskSpaceDialog::INSUFFICIENT_SPACE_PREFIX));
            body += std::format("<div class=\"content-row\"><span class=\"dim-text\">{} </span>{}</div>",
                                LOC(DiskSpaceDialog::LOCATION_LABEL), lfs::core::path_to_utf8(e.path.parent_path()));
            body += std::format("<div class=\"content-row\"><span class=\"dim-text\">{} </span>{}</div>",
                                LOC(DiskSpaceDialog::REQUIRED_LABEL), formatBytes(e.required_bytes));
            if (e.available_bytes > 0) {
                body += std::format("<div class=\"content-row\"><span class=\"dim-text\">{} </span>"
                                    "<span class=\"error-text\">{}</span></div>",
                                    LOC(DiskSpaceDialog::AVAILABLE_LABEL), formatBytes(e.available_bytes));
            }
            body += std::format("<div class=\"warning-text\">{}</div>", LOC(DiskSpaceDialog::INSTRUCTION));

            lfs::core::ModalRequest req;
            req.title = std::format("{} | {}", LOC(DiskSpaceDialog::ERROR_LABEL), subtitle);
            req.body_rml = body;
            req.style = lfs::core::ModalStyle::Error;
            req.width_dp = 480;
            req.buttons = {
                {LOC(DiskSpaceDialog::CANCEL), "secondary"},
                {LOC(DiskSpaceDialog::CHANGE_LOCATION), "warning"},
                {LOC(DiskSpaceDialog::RETRY), "primary"}};

            auto path = e.path;
            auto iteration = e.iteration;
            auto is_checkpoint = e.is_checkpoint;

            req.on_result = [this, path, iteration, is_checkpoint](const lfs::core::ModalResult& result) {
                if (result.button_label == LOC(DiskSpaceDialog::RETRY)) {
                    if (is_checkpoint) {
                        if (auto* tm = viewer_->getTrainerManager()) {
                            if (tm->isFinished() || !tm->isTrainingActive()) {
                                if (auto* trainer = tm->getTrainer()) {
                                    LOG_INFO("Retrying save at iteration {}", iteration);
                                    trainer->save_final_ply_and_checkpoint(iteration);
                                }
                            } else {
                                tm->requestSaveCheckpoint();
                            }
                        }
                    }
                } else if (result.button_label == LOC(DiskSpaceDialog::CHANGE_LOCATION)) {
                    std::filesystem::path new_location = PickFolderDialog(path.parent_path());
                    if (!new_location.empty() && is_checkpoint) {
                        if (auto* tm = viewer_->getTrainerManager()) {
                            if (auto* trainer = tm->getTrainer()) {
                                auto params = trainer->getParams();
                                params.dataset.output_path = new_location;
                                trainer->setParams(params);
                                LOG_INFO("Output path changed to: {}", lfs::core::path_to_utf8(new_location));
                                if (tm->isFinished() || !tm->isTrainingActive())
                                    trainer->save_final_ply_and_checkpoint(iteration);
                                else
                                    tm->requestSaveCheckpoint();
                            }
                        }
                    } else if (!new_location.empty()) {
                        LOG_INFO("Re-export manually using File > Export to: {}",
                                 lfs::core::path_to_utf8(new_location));
                    }
                } else {
                    if (is_checkpoint)
                        LOG_WARN("Checkpoint save cancelled by user");
                    else
                        LOG_INFO("Export cancelled by user");
                }
            };
            req.on_cancel = [is_checkpoint]() {
                if (is_checkpoint)
                    LOG_WARN("Checkpoint save cancelled by user");
                else
                    LOG_INFO("Export cancelled by user");
            };

            enqueueModal(std::move(req));
        });

        state::DatasetLoadCompleted::when([this](const auto& e) {
            if (e.success) {
                focus_panel_name_ = "Training";
            }
        });

        state::SplatFileLoadFailed::when([this](const auto& e) {
            lfs::core::ModalRequest req;
            req.title = "Failed to load file";
            req.body_rml = std::format(
                "<div>Could not load <b>{}</b>:</div>"
                "<div class=\"content-row error-text\" style=\"margin-top: 8dp;\">{}</div>",
                lfs::core::path_to_utf8(e.path.filename()),
                e.error);
            req.style = lfs::core::ModalStyle::Error;
            req.width_dp = 520;
            req.buttons = {{"OK", "primary"}};
            enqueueModal(std::move(req));
        });

        internal::TrainerReady::when([this](const auto&) {
            focus_panel_name_ = "Training";
        });
    }

    bool GuiManager::isCapturingInput() const {
        if (auto* input_controller = viewer_->getInputController()) {
            return input_controller->getBindings().isCapturing();
        }
        return false;
    }

    bool GuiManager::isModalWindowOpen() const {
        return ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel) ||
               rml_modal_overlay_->isOpen();
    }

    bool GuiManager::passiveMouseMoveNeedsRender(const float mouse_x, const float mouse_y) const {
        if (rml_menu_bar_.isOpen())
            return true;
        if (ui_hidden_)
            return false;

        const bool imgui_popup_open =
            ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel);
        if (isCapturingInput() || imgui_popup_open || startup_overlay_.blocksUnderlayInput() || drag_drop_hovering_) {
            return true;
        }

        if (!guiFocusState().want_capture_mouse && isPositionInViewport(mouse_x, mouse_y)) {
            if (auto* const sel = viewer_ ? viewer_->getSelectionTool() : nullptr; sel && sel->isEnabled()) {
                return true;
            }
        }

        return rmlui_manager_.passiveMouseMoveNeedsRender(mouse_x, mouse_y);
    }

    std::optional<double> GuiManager::secondsUntilTooltipReveal() const {
        return rmlui_manager_.secondsUntilTooltipReveal();
    }

    void GuiManager::captureKey(int physical_key, int logical_key, int mods) {
        if (auto* input_controller = viewer_->getInputController()) {
            input_controller->getBindings().captureKey(physical_key, logical_key, mods);
        }
    }

    void GuiManager::captureMouseButton(int button, int mods, double x, double y, std::optional<int> chord_key) {
        if (auto* input_controller = viewer_->getInputController()) {
            input_controller->getBindings().captureMouseButton(button, mods, x, y, chord_key);
        }
    }

    void GuiManager::captureMouseButtonRelease(int button) {
        if (auto* input_controller = viewer_->getInputController()) {
            input_controller->getBindings().captureMouseButtonRelease(button);
        }
    }

    void GuiManager::captureMouseMove(double x, double y) {
        if (auto* input_controller = viewer_->getInputController()) {
            input_controller->getBindings().captureMouseMove(x, y);
        }
    }

    void GuiManager::requestThumbnail(const std::string& video_id) {
        if (menu_bar_) {
            menu_bar_->requestThumbnail(video_id);
        }
    }

    void GuiManager::processThumbnails() {
        if (menu_bar_) {
            menu_bar_->processThumbnails();
        }
    }

    bool GuiManager::isThumbnailReady(const std::string& video_id) const {
        return menu_bar_ ? menu_bar_->isThumbnailReady(video_id) : false;
    }

    uint64_t GuiManager::getThumbnailTexture(const std::string& video_id) const {
        return menu_bar_ ? menu_bar_->getThumbnailTexture(video_id) : 0;
    }

    int GuiManager::getHighlightedCameraUid() const {
        if (auto* sm = viewer_->getSceneManager()) {
            return sm->getSelectedCameraUid();
        }
        return -1;
    }

    void GuiManager::applyDefaultStyle() {
        const std::string preferred_theme = loadThemePreferenceName();
        if (!setThemeByName(preferred_theme)) {
            setTheme(darkTheme());
        }
        rmlui_manager_.activateTheme(currentThemeId());
    }

    void GuiManager::showWindow(const std::string& name, bool show) {
        window_states_[name] = show;
    }

    void GuiManager::enqueueModal(lfs::core::ModalRequest request) {
        if (!rml_modal_overlay_)
            return;

        rml_modal_overlay_->enqueue(std::move(request));
        if (auto* const window_manager = viewer_ ? viewer_->getWindowManager() : nullptr)
            window_manager->wakeEventLoop();
    }

    void GuiManager::enqueueToast(ToastRequest request) {
        if (!rml_toast_overlay_)
            return;

        rml_toast_overlay_->enqueue(std::move(request));
        if (auto* const window_manager = viewer_ ? viewer_->getWindowManager() : nullptr)
            window_manager->wakeEventLoop();
    }

    bool GuiManager::isVramHudOverlayVisible() const {
        return show_vram_hud_ && lfs::diagnostics::VramProfiler::instance().enabled();
    }

    bool GuiManager::isVramHudPublishDue(const std::chrono::steady_clock::time_point now) const {
        return isVramHudOverlayVisible() &&
               (next_vram_hud_publish_ == std::chrono::steady_clock::time_point{} ||
                now >= next_vram_hud_publish_);
    }

    void GuiManager::syncVisiblePanelsBeforeSceneRender() {
        const std::uint64_t sync_generation = lfs::python::pre_scene_panel_sync_generation();
        if (sync_generation == 0 ||
            sync_generation == last_pre_scene_panel_sync_generation_)
            return;
        last_pre_scene_panel_sync_generation_ = sync_generation;

        if (!viewer_ || !show_main_panel_ || ui_hidden_) {
            return;
        }

        auto& reg = PanelRegistry::instance();
        const auto main_tabs = reg.get_panels_for_space(PanelSpace::MainPanelTab);
        if (main_tabs.empty()) {
            return;
        }

        panel_layout_.syncActiveTab(main_tabs, focus_panel_name_);
        const std::string& active_tab = panel_layout_.getActiveTab();
        if (active_tab.empty()) {
            return;
        }

        const ImGuiViewport* const mvp = ImGui::GetMainViewport();
        if (!mvp || mvp->WorkSize.x <= 0.0f || mvp->WorkSize.y <= 0.0f) {
            return;
        }

        auto& editor_ctx = viewer_->getEditorContext();
        auto* scene_manager = viewer_->getSceneManager();
        auto* trainer_manager = viewer_->getTrainerManager();
        editor_ctx.update(scene_manager, trainer_manager);

        UIContext ctx{
            .viewer = viewer_,
            .window_states = &window_states_,
            .editor = &editor_ctx,
            .sequencer_controller = &sequencer_ui_.controller(),
            .rml_manager = &rmlui_manager_,
            .fonts = buildFontSet()};

        lfs::core::Scene* scene = nullptr;
        if (scene_manager)
            scene = &scene_manager->getScene();

        PanelDrawContext draw_ctx;
        draw_ctx.ui = &ctx;
        draw_ctx.viewport = &viewport_layout_;
        draw_ctx.scene = scene;
        draw_ctx.ui_hidden = ui_hidden_;
        draw_ctx.scene_generation = python::get_scene_generation();
        draw_ctx.suppress_non_native_panels = python::is_plugin_preload_running();
        if (scene_manager)
            draw_ctx.has_selection = scene_manager->hasSelectedNode();
        if (auto* cc = lfs::event::command_center())
            draw_ctx.is_training = cc->snapshot().is_running;

        PanelInputState input;
        input.mouse_x = -1.0e9f;
        input.mouse_y = -1.0e9f;
        input.screen_x = mvp->Pos.x;
        input.screen_y = mvp->Pos.y;
        input.screen_w = static_cast<int>(mvp->Size.x);
        input.screen_h = static_cast<int>(mvp->Size.y);

        const float dpi = lfs::python::get_shared_dpi_scale();
        const float panel_h = mvp->WorkSize.y - PanelLayoutManager::STATUS_BAR_HEIGHT * dpi;
        if (panel_h <= 0.0f) {
            return;
        }

        constexpr float kPanelPad = 8.0f;
        constexpr float kPreloadMaxHeight = 100000.0f;
        const float content_w = panel_layout_.getRightPanelWidth() - 2.0f * kPanelPad;
        if (content_w <= 0.0f) {
            return;
        }

        const float splitter_h = PanelLayoutManager::SPLITTER_H * dpi;
        const float tab_bar_h = PanelLayoutManager::TAB_BAR_H * dpi;
        const float avail_h = panel_h - 2.0f * kPanelPad;
        const float scene_h =
            std::max(80.0f * dpi,
                     avail_h * panel_layout_.getScenePanelRatio() - splitter_h * 0.5f);
        const float content_top = mvp->WorkPos.y + kPanelPad;
        const float tab_content_y = content_top + scene_h + splitter_h + tab_bar_h;
        const float tab_content_h =
            std::max(0.0f, content_top + avail_h - tab_content_y);
        const float clip_y_min = tab_content_y;
        const float clip_y_max = tab_content_y + tab_content_h;

        reg.preload_single_panel_direct(active_tab, content_w, kPreloadMaxHeight, draw_ctx,
                                        clip_y_min, clip_y_max, &input);
        reg.preload_child_panels_direct(active_tab, content_w, kPreloadMaxHeight, draw_ctx,
                                        clip_y_min, clip_y_max, &input);
    }

    bool GuiManager::needsAnimationFrame() const {
        const auto now = std::chrono::steady_clock::now();
        const bool ui_toggle_due =
            ui_toggle_pending_ && now >= ui_toggle_next_allowed_at_;
        const bool fullscreen_toggle_due =
            fullscreen_toggle_pending_ && now >= fullscreen_toggle_next_allowed_at_;
        if (ui_toggle_due || fullscreen_toggle_due || interactive_transition_resume_training_ ||
            now < interactive_transition_guard_until_) {
            return true;
        }
        if (isViewportExportLocked())
            return true;
        if (startup_overlay_.needsAnimationFrame())
            return true;
        if (rml_modal_overlay_ && rml_modal_overlay_->needsAnimationFrame())
            return true;
        if (rml_toast_overlay_ && rml_toast_overlay_->needsAnimationFrame())
            return true;
        if (global_context_menu_ && global_context_menu_->needsAnimationFrame())
            return true;
        if (video_widget_ && video_widget_->isVideoPlaying())
            return true;
        if (ui_layout_settle_frames_ > 0)
            return true;
        if (isVramHudPublishDue(now))
            return true;
        if (rml_viewport_overlay_.needsAnimationFrame())
            return true;
        if (rml_menu_bar_.needsAnimationFrame())
            return true;
        if (rml_right_panel_.needsAnimationFrame())
            return true;
        if (!python::is_plugin_preload_running() &&
            PanelRegistry::instance().needsAnimationFrameForVisiblePanels({
                .active_main_tab = panel_layout_.getActiveTab(),
                .ui_visible = !ui_hidden_,
                .right_panel_visible = show_main_panel_ && !ui_hidden_,
                .bottom_dock_visible = panel_layout_.isBottomDockVisible(),
            }))
            return true;
        return false;
    }

    bool GuiManager::isViewportExportLocked() const {
        return async_tasks_.isExporting() || async_tasks_.isExportingVideo();
    }

    void GuiManager::dismissStartupOverlay() {
        if (startup_overlay_.isPluginLoadComplete())
            startup_overlay_.dismiss();
    }

    void GuiManager::setStartupPluginLoadState(const bool started,
                                               const bool active,
                                               const float progress,
                                               const std::string& stage) {
        startup_overlay_.setPluginLoadState(started, active, progress, stage);
    }

    void GuiManager::requestExitConfirmation() {
        startup_overlay_.dismiss();
        lfs::core::events::cmd::RequestExit{}.emit();
    }

    bool GuiManager::isExitConfirmationPending() const {
        return lfs::python::is_exit_popup_open();
    }

} // namespace lfs::vis::gui
