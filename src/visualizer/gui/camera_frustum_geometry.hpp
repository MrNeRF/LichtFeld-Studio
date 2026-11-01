/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include "internal/viewport.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "rendering/environment_math.hpp"
#include "rendering/passes/vulkan_viewport_pass.hpp"
#include "rendering/rendering.hpp"
#include "rendering/rendering_types.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <glm/gtc/constants.hpp>
#include <limits>
#include <memory>
#include <span>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <vector>

namespace lfs::vis::gui {
    [[nodiscard]] inline std::optional<glm::mat4> cameraVisualizerTransform(
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

    [[nodiscard]] inline std::optional<glm::mat4> cameraFrustumModelMatrix(
        const lfs::core::Camera& camera,
        const glm::mat4& visualizer_camera_to_world,
        const float scale) {
        // The frustum represents the calibrated camera rather than the
        // resolution selected for loading training images. Undistortion
        // updates these calibration dimensions and FoVy together, while
        // training downscaling only changes operational decode dimensions.
        const int calibration_width = camera.camera_width();
        const int calibration_height = camera.camera_height();
        if (calibration_width <= 0 || calibration_height <= 0 || scale <= 0.0f) {
            return std::nullopt;
        }

        constexpr float kEquirectangularDisplayFov = 1.0472f;
        const bool equirectangular =
            camera.camera_model_type() == lfs::core::CameraModelType::EQUIRECTANGULAR;
        if (!equirectangular && camera.FoVy() <= 0.0f) {
            return std::nullopt;
        }

        const float aspect = static_cast<float>(calibration_width) /
                             static_cast<float>(calibration_height);
        const float fov_y = equirectangular
                                ? kEquirectangularDisplayFov
                                : camera.FoVy();
        const float half_height = std::tan(fov_y * 0.5f);
        const float half_width = half_height * aspect;

        const glm::mat4 fov_scale = glm::scale(
            glm::mat4(1.0f),
            glm::vec3(half_width * 2.0f * scale, half_height * 2.0f * scale, scale));
        return visualizer_camera_to_world * fov_scale;
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

    struct ThumbnailPlacement {
        std::uintptr_t texture_id = 0;
        glm::vec2 uv_min{0.0f};
        glm::vec2 uv_max{1.0f};
    };

    [[nodiscard]] inline float cameraFrustumVisibilityAlpha(const glm::vec3& camera_position,
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

    [[nodiscard]] inline glm::vec4 cachedCameraFrustumColor(
        const bool validation,
        const std::span<const glm::vec3> loss_colors,
        const std::size_t camera_index,
        const RenderSettings& settings,
        const float alpha,
        const bool focused,
        const bool disabled,
        const bool emphasized) {
        glm::vec3 color = validation ? settings.eval_camera_color : settings.train_camera_color;
        if (camera_index < loss_colors.size()) {
            const auto& override_color = loss_colors[camera_index];
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
            color = validation ? glm::vec3(0.9f, 0.75f, 0.0f)
                               : glm::vec3(1.0f, 0.55f, 0.0f);
            final_alpha = std::min(1.0f, final_alpha + 0.3f);
        }
        if (disabled) {
            color = glm::mix(color, glm::vec3(0.5f), 0.5f);
            final_alpha *= 0.5f;
        }
        return glm::vec4(color, std::clamp(final_alpha, 0.0f, 1.0f));
    }

    [[nodiscard]] inline glm::vec2 renderToPanelScreen(const VulkanGuidePanelTarget& panel,
                                                       const glm::vec2& projected) {
        const float sx = panel.size.x / static_cast<float>(std::max(panel.render_size.x, 1));
        const float sy = panel.size.y / static_cast<float>(std::max(panel.render_size.y, 1));
        return glm::vec2(panel.pos.x + projected.x * sx, panel.pos.y + projected.y * sy);
    }

    [[nodiscard]] inline glm::vec2 screenToViewportNdc(const glm::vec2& screen,
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

    [[nodiscard]] inline bool projectedQuadVisible(const std::array<glm::vec2, 4>& points,
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

    inline void appendTexturedOverlayQuad(const VulkanViewportPassParams& params,
                                          std::vector<VulkanViewportTexturedOverlay>& out,
                                          const std::uintptr_t texture_id,
                                          const std::array<glm::vec2, 4>& screen_points,
                                          const glm::vec2& uv_min,
                                          const glm::vec2& uv_max,
                                          const glm::vec4& tint_opacity,
                                          const glm::vec4& effects,
                                          const std::array<float, 4>& view_depths = {0.0f, 0.0f, 0.0f, 0.0f}) {
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
        out.push_back(overlay);
    }

    enum class FrustumEmit : std::uint8_t {
        None,
        Instance,
        Equirectangular,
    };

    struct CameraFrustumGeometry {
        std::vector<int> uids;
        std::vector<glm::mat4> models;
        std::vector<glm::vec3> positions;
        std::vector<std::uint8_t> valid_cameras;
        std::vector<std::uint8_t> equirectangular_cameras;
        std::vector<std::uint8_t> validation_cameras;
        std::vector<std::uint8_t> disabled_cameras;
        std::vector<std::uint8_t> emphasized_cameras;
        std::vector<int> thumbnail_indices;
        std::vector<ThumbnailPlacement> thumbnail_placements;
        std::vector<glm::vec3> loss_colors;

        std::vector<std::array<glm::vec2, 4>> projected_points;
        std::vector<std::array<float, 4>> projected_depths;
        std::vector<std::uint8_t> projected_visible;
        std::vector<glm::vec4> camera_colors;
        std::vector<std::uint32_t> instance_camera_indices;
        std::vector<std::uint32_t> textured_camera_indices;
        std::vector<FrustumEmit> emitted;
        bool has_equirectangular_lines = false;
        std::shared_ptr<VulkanViewportFrustumOverlayData> data =
            std::make_shared<VulkanViewportFrustumOverlayData>();
    };

    template <typename EmitLine>
    void appendEquirectangularCameraFrustum(const glm::mat4& model, const glm::vec4& color,
                                            EmitLine&& emit_line) {
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
                emit_line(previous, current, color);
                previous = current;
            }
        }
        for (int lon = 0; lon < kLonSegments; lon += 4) {
            glm::vec3 previous = point(0, lon);
            for (int lat = 1; lat <= kLatSegments; ++lat) {
                const glm::vec3 current = point(lat, lon);
                emit_line(previous, current, color);
                previous = current;
            }
        }

        const glm::vec3 apex = glm::vec3(model * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
        for (int lon = 0; lon < kLonSegments; lon += kLonSegments / 4) {
            emit_line(apex, point(kLatSegments / 2, lon), color);
        }
    }

    template <typename EmitLine>
    void appendCameraFrustumPanel(VulkanViewportPassParams& params, CameraFrustumGeometry& cache,
                                  const VulkanGuidePanelTarget& panel, const size_t panel_index,
                                  const RenderSettings& settings, const int hovered_camera_id,
                                  EmitLine&& emit_line) {
        const size_t camera_count = cache.models.size();
        const float ortho_scale = panel.viewport->ortho_scale_override.value_or(settings.ortho_scale);
        const glm::mat3 rotation = panel.viewport->getRotationMatrix();
        const glm::vec3 translation = panel.viewport->getTranslation();
        const glm::mat3 world_to_panel_rotation = glm::transpose(rotation);
        const float width = static_cast<float>(std::max(panel.render_size.x, 1));
        const float height = static_cast<float>(std::max(panel.render_size.y, 1));
        const float cx = width * 0.5f;
        const float cy = height * 0.5f;
        const auto [fx, fy] = lfs::rendering::computePixelFocalLengths(
            panel.render_size, settings.focal_length_mm);
        const std::uint32_t first_instance =
            static_cast<std::uint32_t>(cache.data->frustum_instances.size());
        cache.emitted.assign(camera_count, FrustumEmit::None);
        tbb::parallel_for(tbb::blocked_range<size_t>(0, camera_count, 512), [&](const tbb::blocked_range<size_t>& range) {
            for (size_t camera_index = range.begin(); camera_index != range.end(); ++camera_index) {
                const size_t projected_index = panel_index * camera_count + camera_index;
                cache.projected_visible[projected_index] = 0;
                if (!cache.valid_cameras[camera_index])
                    continue;
                const bool disabled = cache.disabled_cameras[camera_index] != 0;
                const float alpha = cameraFrustumVisibilityAlpha(
                    cache.positions[camera_index], translation,
                    settings.camera_frustum_scale, disabled);
                if (alpha <= 0.01f)
                    continue;
                const bool emphasized = cache.emphasized_cameras[camera_index] != 0;
                const bool focused = cache.uids[camera_index] == hovered_camera_id;
                const glm::vec4 color = cachedCameraFrustumColor(
                    cache.validation_cameras[camera_index] != 0,
                    cache.loss_colors,
                    camera_index,
                    settings,
                    alpha,
                    focused,
                    disabled,
                    emphasized);
                if (color.a <= 0.01f)
                    continue;
                cache.camera_colors[projected_index] = color;
                if (cache.equirectangular_cameras[camera_index]) {
                    cache.emitted[camera_index] = FrustumEmit::Equirectangular;
                    continue;
                }

                constexpr std::array image_corners{
                    glm::vec3(-0.5f, -0.5f, -1.0f),
                    glm::vec3(0.5f, -0.5f, -1.0f),
                    glm::vec3(0.5f, 0.5f, -1.0f),
                    glm::vec3(-0.5f, 0.5f, -1.0f),
                };
                std::array<glm::vec2, 4> screen_points{};
                std::array<float, 4> depths{};
                bool quad_visible = true;
                const glm::vec3 center = glm::vec3(
                    cache.models[camera_index] * glm::vec4(0.0f, 0.0f, -0.5f, 1.0f));
                const float radius = 0.5f * std::sqrt(
                                                glm::dot(glm::vec3(cache.models[camera_index][0]), glm::vec3(cache.models[camera_index][0])) +
                                                glm::dot(glm::vec3(cache.models[camera_index][1]), glm::vec3(cache.models[camera_index][1])) +
                                                glm::dot(glm::vec3(cache.models[camera_index][2]), glm::vec3(cache.models[camera_index][2])));
                const glm::vec3 center_view = world_to_panel_rotation * (center - translation);
                if (!settings.equirectangular && center_view.z - radius >= -1e-4f) {
                    continue;
                }
                if (!settings.equirectangular && center_view.z + radius < -1e-4f) {
                    if (settings.orthographic &&
                        std::isfinite(ortho_scale) && ortho_scale > 0.0f) {
                        const float projected_radius = radius * ortho_scale;
                        const float projected_x = cx + center_view.x * ortho_scale;
                        const float projected_y = cy - center_view.y * ortho_scale;
                        if (projected_x + projected_radius < 0.0f ||
                            projected_x - projected_radius > width ||
                            projected_y + projected_radius < 0.0f ||
                            projected_y - projected_radius > height) {
                            continue;
                        }
                    } else if (!settings.orthographic && fx > 0.0f && fy > 0.0f) {
                        const float depth = -center_view.z;
                        const float distance = glm::length(center_view);
                        const float angular_radius = std::asin(std::clamp(radius / distance, 0.0f, 1.0f));
                        const float horizontal_angle = std::atan2(center_view.x, depth);
                        const float vertical_angle = std::atan2(center_view.y, depth);
                        const float horizontal_limit = std::atan2(cx, fx);
                        const float vertical_limit = std::atan2(cy, fy);
                        if (std::abs(horizontal_angle) - angular_radius > horizontal_limit ||
                            std::abs(vertical_angle) - angular_radius > vertical_limit) {
                            continue;
                        }
                    }
                }
                for (size_t corner = 0; corner < image_corners.size(); ++corner) {
                    const glm::vec3 world_point = glm::vec3(
                        cache.models[camera_index] * glm::vec4(image_corners[corner], 1.0f));
                    const glm::vec3 view = world_to_panel_rotation * (world_point - translation);
                    if (settings.equirectangular) {
                        const float len = glm::length(view);
                        if (!std::isfinite(len) || len <= 1e-6f) {
                            quad_visible = false;
                            break;
                        }
                        const glm::vec3 direction = view / len;
                        const auto uv = lfs::rendering::envmath::equirectUvForDirection(
                            {direction.x, direction.y, direction.z});
                        const float ndc_x = uv.u * 2.0f - 1.0f;
                        const float ndc_y = uv.v * 2.0f - 1.0f;
                        screen_points[corner] = panel.pos + glm::vec2(
                                                                (ndc_x * 0.5f + 0.5f) * panel.size.x,
                                                                (ndc_y * 0.5f + 0.5f) * panel.size.y);
                        depths[corner] = len;
                    } else {
                        if (view.z >= -1e-4f) {
                            quad_visible = false;
                            break;
                        }
                        const glm::vec2 projected = settings.orthographic
                                                        ? glm::vec2(cx + view.x * ortho_scale,
                                                                    cy - view.y * ortho_scale)
                                                        : glm::vec2(cx + view.x * fx / -view.z,
                                                                    cy - view.y * fy / -view.z);
                        screen_points[corner] = renderToPanelScreen(panel, projected);
                        depths[corner] = -view.z;
                    }
                }
                quad_visible = quad_visible && projectedQuadVisible(screen_points, panel);
                cache.projected_points[projected_index] = screen_points;
                cache.projected_depths[projected_index] = depths;
                cache.projected_visible[projected_index] = quad_visible;
                cache.emitted[camera_index] = FrustumEmit::Instance;
            }
        });
        for (size_t camera_index = 0; camera_index < camera_count; ++camera_index) {
            if (cache.emitted[camera_index] == FrustumEmit::None)
                continue;
            const size_t projected_index = panel_index * camera_count + camera_index;
            const glm::vec4 color = cache.camera_colors[projected_index];
            if (cache.emitted[camera_index] == FrustumEmit::Equirectangular) {
                appendEquirectangularCameraFrustum(cache.models[camera_index], color, emit_line);
                cache.has_equirectangular_lines = true;
                continue;
            }
            if (cache.emitted[camera_index] != FrustumEmit::Instance)
                continue;
            if (cache.projected_visible[projected_index] && cache.thumbnail_indices[camera_index] >= 0) {
                const auto& placement = cache.thumbnail_placements[camera_index];
                appendTexturedOverlayQuad(
                    params,
                    cache.data->textured_overlays,
                    placement.texture_id,
                    cache.projected_points[projected_index],
                    placement.uv_min,
                    placement.uv_max,
                    {color.r, color.g, color.b, std::clamp(color.a * 0.8f, 0.0f, 0.8f)},
                    {cache.emphasized_cameras[camera_index] != 0 ? 0.18f : 0.0f,
                     cache.disabled_cameras[camera_index] != 0 ? 0.5f : 0.0f,
                     0.0f,
                     0.0f},
                    cache.projected_depths[projected_index]);
                cache.textured_camera_indices.push_back(
                    static_cast<std::uint32_t>(projected_index));
            }
            cache.data->frustum_instances.push_back({.model = cache.models[camera_index], .color = color});
            cache.instance_camera_indices.push_back(static_cast<std::uint32_t>(projected_index));
        }
        const auto instance_count = static_cast<std::uint32_t>(
                                        cache.data->frustum_instances.size()) -
                                    first_instance;
        if (instance_count > 0) {
            const glm::mat4 view = lfs::rendering::makeViewMatrix(rotation, translation);
            cache.data->frustum_batches.push_back({
                .view = view,
                .viewport_pos = panel.pos,
                .viewport_size = panel.size,
                .render_size = glm::vec2(panel.render_size),
                .focal_x = settings.orthographic ? ortho_scale : fx,
                .focal_y = settings.orthographic ? ortho_scale : fy,
                .orthographic = settings.orthographic,
                .equirectangular = settings.equirectangular,
                .first_instance = first_instance,
                .instance_count = instance_count,
            });
        }
    }
} // namespace lfs::vis::gui
