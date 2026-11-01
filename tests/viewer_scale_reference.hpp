/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include "gui/camera_frustum_geometry.hpp"
#include "rendering/coordinate_conventions.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

// Reference picking and sequential overlay loops retained from f4604a1fa.
namespace lfs::vis::viewer_scale_reference {
    using namespace lfs::rendering;
    using namespace lfs::vis::gui;
    [[nodiscard]] inline std::vector<glm::vec3> cameraFrustumWorldPoints(
        const lfs::core::Camera& camera,
        const glm::mat4& visualizer_camera_to_world,
        const float scale) {
        std::vector<glm::vec3> points;
        // Picking must follow the calibrated frustum, not the resolution
        // selected for loading training images. Undistortion updates these
        // calibration dimensions and FoVy together; training downscaling
        // only changes the operational decode dimensions.
        const int calibration_width = camera.camera_width();
        const int calibration_height = camera.camera_height();
        if (calibration_width <= 0 || calibration_height <= 0 || scale <= 0.0f) {
            return points;
        }

        const bool equirectangular =
            camera.camera_model_type() == lfs::core::CameraModelType::EQUIRECTANGULAR;
        if (equirectangular) {
            constexpr int SEGMENTS = 48;
            points.reserve(SEGMENTS * 3);
            for (int circle = 0; circle < 3; ++circle) {
                for (int i = 0; i < SEGMENTS; ++i) {
                    const float a = static_cast<float>(i) / static_cast<float>(SEGMENTS) *
                                    2.0f * glm::pi<float>();
                    glm::vec3 local(0.0f);
                    if (circle == 0) {
                        local = {std::cos(a), std::sin(a), 0.0f};
                    } else if (circle == 1) {
                        local = {std::cos(a), 0.0f, std::sin(a)};
                    } else {
                        local = {0.0f, std::cos(a), std::sin(a)};
                    }
                    points.push_back(glm::vec3(
                        visualizer_camera_to_world * glm::vec4(local * scale, 1.0f)));
                }
            }
            return points;
        }

        if (camera.FoVy() <= 0.0f) {
            return points;
        }

        const float aspect = static_cast<float>(calibration_width) /
                             static_cast<float>(calibration_height);
        const float fov_y = camera.FoVy();
        const float half_height = std::tan(fov_y * 0.5f) * scale;
        const float half_width = half_height * aspect;

        const std::array local_points{
            glm::vec3(0.0f, 0.0f, 0.0f),
            glm::vec3(-half_width, half_height, -scale),
            glm::vec3(half_width, half_height, -scale),
            glm::vec3(half_width, -half_height, -scale),
            glm::vec3(-half_width, -half_height, -scale),
        };
        points.reserve(local_points.size());
        for (const glm::vec3& local : local_points) {
            points.push_back(glm::vec3(visualizer_camera_to_world * glm::vec4(local, 1.0f)));
        }
        return points;
    }

    [[nodiscard]] inline float pointSegmentDistance(
        const glm::vec2& point,
        const glm::vec2& a,
        const glm::vec2& b) {
        const glm::vec2 ab = b - a;
        const float denom = glm::dot(ab, ab);
        if (denom <= 1e-6f) {
            return glm::length(point - a);
        }
        const float t = std::clamp(glm::dot(point - a, ab) / denom, 0.0f, 1.0f);
        return glm::length(point - (a + ab * t));
    }

    [[nodiscard]] inline std::optional<glm::vec2> projectFrustumPoint(
        const glm::vec3& world_point,
        const CameraFrustumPickRequest& request) {
        const auto projected = projectWorldPoint(
            request.viewport.rotation,
            request.viewport.translation,
            request.viewport.size,
            world_point,
            request.viewport.focal_length_mm,
            request.viewport.orthographic,
            request.viewport.ortho_scale);
        if (!projected) {
            return std::nullopt;
        }

        const float scale_x = request.viewport_size.x /
                              static_cast<float>(std::max(request.viewport.size.x, 1));
        const float scale_y = request.viewport_size.y /
                              static_cast<float>(std::max(request.viewport.size.y, 1));
        return glm::vec2(
            request.viewport_pos.x + projected->x * scale_x,
            request.viewport_pos.y + projected->y * scale_y);
    }

    inline int pickCameraFrustum(
        const std::vector<std::shared_ptr<const lfs::core::Camera>>& cameras,
        const CameraFrustumPickRequest& request) {
        if (cameras.empty() || request.viewport_size.x <= 0.0f || request.viewport_size.y <= 0.0f ||
            request.viewport.size.x <= 0 || request.viewport.size.y <= 0) {
            return -1;
        }

        constexpr float HIT_RADIUS_PIXELS = 12.0f;
        int best_uid = -1;
        float best_score = HIT_RADIUS_PIXELS;
        float best_depth = std::numeric_limits<float>::max();
        const glm::vec2 mouse = request.mouse_pos;
        const glm::vec3 viewer_position = request.viewport.translation;

        for (size_t i = 0; i < cameras.size(); ++i) {
            const auto& camera = cameras[i];
            if (!camera) {
                continue;
            }
            glm::mat4 scene_transform = request.scene_transform;
            if (i < request.scene_transforms.size()) {
                scene_transform = request.scene_transforms[i];
            }

            const auto transform = cameraVisualizerTransform(*camera, scene_transform);
            if (!transform) {
                continue;
            }
            const auto points = cameraFrustumWorldPoints(*camera, *transform, request.scale);
            if (points.empty()) {
                continue;
            }

            float camera_best = HIT_RADIUS_PIXELS;
            if (camera->camera_model_type() == lfs::core::CameraModelType::EQUIRECTANGULAR) {
                constexpr int SEGMENTS = 48;
                for (int circle = 0; circle < 3; ++circle) {
                    const int offset = circle * SEGMENTS;
                    if (offset + SEGMENTS > static_cast<int>(points.size())) {
                        break;
                    }
                    for (int segment = 0; segment < SEGMENTS; ++segment) {
                        const auto a = projectFrustumPoint(points[offset + segment], request);
                        const auto b = projectFrustumPoint(points[offset + ((segment + 1) % SEGMENTS)], request);
                        if (!a || !b) {
                            continue;
                        }
                        camera_best = std::min(camera_best, pointSegmentDistance(mouse, *a, *b));
                    }
                }
            } else if (points.size() >= 5) {
                constexpr std::array<std::pair<int, int>, 8> EDGES{{
                    {0, 1},
                    {0, 2},
                    {0, 3},
                    {0, 4},
                    {1, 2},
                    {2, 3},
                    {3, 4},
                    {4, 1},
                }};
                for (const auto& [a_index, b_index] : EDGES) {
                    const auto a = projectFrustumPoint(points[static_cast<size_t>(a_index)], request);
                    const auto b = projectFrustumPoint(points[static_cast<size_t>(b_index)], request);
                    if (!a || !b) {
                        continue;
                    }
                    camera_best = std::min(camera_best, pointSegmentDistance(mouse, *a, *b));
                }
            }

            const glm::vec3 camera_position = glm::vec3((*transform)[3]);
            const float depth = glm::length(camera_position - viewer_position);
            if (camera_best < best_score ||
                (std::abs(camera_best - best_score) <= 1e-3f && depth < best_depth)) {
                best_score = camera_best;
                best_depth = depth;
                best_uid = camera->uid();
            }
        }

        if (best_score < HIT_RADIUS_PIXELS)
            return best_uid;
        return -1;
    }

    template <typename EmitLine>
    void appendCameraFrustumPanelSequential(VulkanViewportPassParams& params, CameraFrustumGeometry& cache,
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
        for (size_t camera_index = 0; camera_index < camera_count; ++camera_index) {
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
            const size_t projected_index = panel_index * camera_count + camera_index;
            cache.camera_colors[projected_index] = color;
            if (cache.equirectangular_cameras[camera_index]) {
                appendEquirectangularCameraFrustum(cache.models[camera_index], color, emit_line);
                cache.has_equirectangular_lines = true;
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
            if (quad_visible && cache.thumbnail_indices[camera_index] >= 0) {
                const auto& placement = cache.thumbnail_placements[camera_index];
                appendTexturedOverlayQuad(
                    params,
                    cache.data->textured_overlays,
                    placement.texture_id,
                    screen_points,
                    placement.uv_min,
                    placement.uv_max,
                    {color.r, color.g, color.b, std::clamp(color.a * 0.8f, 0.0f, 0.8f)},
                    {emphasized ? 0.18f : 0.0f, disabled ? 0.5f : 0.0f, 0.0f, 0.0f},
                    depths);
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
} // namespace lfs::vis::viewer_scale_reference
