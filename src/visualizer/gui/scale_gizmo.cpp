/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/scale_gizmo.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace lfs::vis::gui {
    namespace {
        constexpr float AXIS_LENGTH_PX = 104.0f;
        constexpr float AXIS_START_PX = 14.0f;
        constexpr float AXIS_HIT_THRESHOLD_PX = 10.0f;
        constexpr float AXIS_BOX_HIT_RADIUS_PX = 14.0f;
        constexpr float AXIS_BOX_HALF_PX = 6.5f;
        constexpr float CENTER_HIT_RADIUS_PX = 12.0f;
        constexpr float PLANE_NEAR_PX = 24.0f;
        constexpr float PLANE_FAR_PX = 48.0f;
        constexpr float SCALE_DRAG_PIXELS = 96.0f;
        constexpr float MIN_SCALE_FACTOR = 0.01f;
        constexpr float DELTA_EPSILON = 0.000001f;

        struct AxisVisual {
            ScaleGizmoHandle handle = ScaleGizmoHandle::None;
            glm::vec3 direction{0.0f};
            OverlayColor color = OVERLAY_COL32_WHITE;
        };

        struct PlaneVisual {
            ScaleGizmoHandle handle = ScaleGizmoHandle::None;
            int axis_a = 0;
            int axis_b = 1;
            OverlayColor color = OVERLAY_COL32_WHITE;
        };

        struct ProjectedPlane {
            PlaneVisual plane;
            std::array<glm::vec2, 4> quad{};
            bool valid = false;
        };

        struct ProjectedAxis {
            AxisVisual axis;
            glm::vec2 start{0.0f};
            glm::vec2 end{0.0f};
            bool valid = false;
        };

        struct ActiveState {
            bool active = false;
            int id = 0;
            ScaleGizmoHandle handle = ScaleGizmoHandle::None;
            glm::vec3 pivot_world{0.0f};
            glm::mat3 orientation_world{1.0f};
            glm::vec2 start_mouse{0.0f};
            glm::vec2 screen_axis{1.0f, 0.0f};
            glm::ivec3 scale_axes{1};
            glm::vec3 applied_scale{1.0f};
        };

        ActiveState g_active;
        bool g_hovered = false;

        [[nodiscard]] float lengthSquared(const glm::vec2& v) {
            return v.x * v.x + v.y * v.y;
        }

        [[nodiscard]] float lengthSquared(const glm::vec3& v) {
            return v.x * v.x + v.y * v.y + v.z * v.z;
        }

        [[nodiscard]] glm::vec2 safeNormalize(const glm::vec2& v, const glm::vec2& fallback) {
            const float len2 = lengthSquared(v);
            if (len2 <= std::numeric_limits<float>::epsilon() || !std::isfinite(len2)) {
                return fallback;
            }
            return v / std::sqrt(len2);
        }

        [[nodiscard]] glm::vec3 safeNormalize(const glm::vec3& v, const glm::vec3& fallback) {
            const float len2 = lengthSquared(v);
            if (len2 <= std::numeric_limits<float>::epsilon() || !std::isfinite(len2)) {
                return fallback;
            }
            return v / std::sqrt(len2);
        }

        [[nodiscard]] glm::vec3 axisFromConfig(const glm::mat3& orientation, const int index) {
            constexpr std::array<glm::vec3, 3> fallbacks = {
                glm::vec3(1.0f, 0.0f, 0.0f),
                glm::vec3(0.0f, 1.0f, 0.0f),
                glm::vec3(0.0f, 0.0f, 1.0f),
            };
            return safeNormalize(glm::vec3(orientation[index]), fallbacks[static_cast<size_t>(index)]);
        }

        [[nodiscard]] glm::vec3 cameraRight(const glm::mat4& view) {
            return safeNormalize(
                glm::vec3(view[0][0], view[1][0], view[2][0]),
                glm::vec3(1.0f, 0.0f, 0.0f));
        }

        [[nodiscard]] glm::vec3 cameraUp(const glm::mat4& view) {
            return safeNormalize(
                glm::vec3(view[0][1], view[1][1], view[2][1]),
                glm::vec3(0.0f, 1.0f, 0.0f));
        }

        [[nodiscard]] OverlayColor withAlpha(const OverlayColor color, const float alpha) {
            const int r = static_cast<int>(color & 0xFF);
            const int g = static_cast<int>((color >> 8) & 0xFF);
            const int b = static_cast<int>((color >> 16) & 0xFF);
            const int a = static_cast<int>(std::clamp(alpha, 0.0f, 1.0f) * 255.0f);
            return overlayColor(r, g, b, a);
        }

        [[nodiscard]] bool projectPoint(const glm::mat4& view_projection,
                                        const glm::vec2& viewport_pos,
                                        const glm::vec2& viewport_size,
                                        const glm::vec3& world,
                                        glm::vec2& screen) {
            const glm::vec4 clip = view_projection * glm::vec4(world, 1.0f);
            if (std::abs(clip.w) <= std::numeric_limits<float>::epsilon() || clip.w < 0.0f) {
                return false;
            }

            const glm::vec3 ndc = glm::vec3(clip) / clip.w;
            if (!std::isfinite(ndc.x) || !std::isfinite(ndc.y)) {
                return false;
            }

            screen.x = viewport_pos.x + (ndc.x * 0.5f + 0.5f) * viewport_size.x;
            screen.y = viewport_pos.y + (-ndc.y * 0.5f + 0.5f) * viewport_size.y;
            return true;
        }

        [[nodiscard]] bool isInViewport(const ScaleGizmoConfig& config, const glm::vec2& p) {
            return p.x >= config.viewport_pos.x &&
                   p.x <= config.viewport_pos.x + config.viewport_size.x &&
                   p.y >= config.viewport_pos.y &&
                   p.y <= config.viewport_pos.y + config.viewport_size.y;
        }

        [[nodiscard]] float distanceToSegmentSquared(const glm::vec2& p, const glm::vec2& a, const glm::vec2& b) {
            const glm::vec2 ab = b - a;
            const float denom = lengthSquared(ab);
            if (denom <= std::numeric_limits<float>::epsilon()) {
                return lengthSquared(p - a);
            }
            const float t = std::clamp(glm::dot(p - a, ab) / denom, 0.0f, 1.0f);
            return lengthSquared(p - (a + ab * t));
        }

        [[nodiscard]] bool pointInConvexQuad(const glm::vec2& p, const std::array<glm::vec2, 4>& quad) {
            float sign = 0.0f;
            for (int i = 0; i < 4; ++i) {
                const glm::vec2 a = quad[static_cast<size_t>(i)];
                const glm::vec2 b = quad[static_cast<size_t>((i + 1) % 4)];
                const glm::vec2 edge = b - a;
                const glm::vec2 to_p = p - a;
                const float cross = edge.x * to_p.y - edge.y * to_p.x;
                if (std::abs(cross) <= 0.0001f) {
                    continue;
                }
                if (sign == 0.0f) {
                    sign = cross > 0.0f ? 1.0f : -1.0f;
                } else if ((cross > 0.0f ? 1.0f : -1.0f) != sign) {
                    return false;
                }
            }
            return true;
        }

        [[nodiscard]] float estimatePixelsPerWorld(const ScaleGizmoConfig& config,
                                                   const glm::mat4& view_projection,
                                                   const glm::vec2& pivot_screen) {
            glm::vec2 probe;
            if (projectPoint(view_projection, config.viewport_pos, config.viewport_size,
                             config.pivot_world + cameraRight(config.view), probe)) {
                const float value = glm::length(probe - pivot_screen);
                if (value > 0.0001f && std::isfinite(value)) {
                    return value;
                }
            }
            if (projectPoint(view_projection, config.viewport_pos, config.viewport_size,
                             config.pivot_world + cameraUp(config.view), probe)) {
                const float value = glm::length(probe - pivot_screen);
                if (value > 0.0001f && std::isfinite(value)) {
                    return value;
                }
            }
            return 64.0f;
        }

        [[nodiscard]] glm::ivec3 scaleAxesForHandle(const ScaleGizmoHandle handle) {
            switch (handle) {
            case ScaleGizmoHandle::X: return glm::ivec3(1, 0, 0);
            case ScaleGizmoHandle::Y: return glm::ivec3(0, 1, 0);
            case ScaleGizmoHandle::Z: return glm::ivec3(0, 0, 1);
            case ScaleGizmoHandle::XY: return glm::ivec3(1, 1, 0);
            case ScaleGizmoHandle::YZ: return glm::ivec3(0, 1, 1);
            case ScaleGizmoHandle::ZX: return glm::ivec3(1, 0, 1);
            case ScaleGizmoHandle::Uniform: return glm::ivec3(1, 1, 1);
            case ScaleGizmoHandle::None:
            default: return glm::ivec3(0);
            }
        }

        [[nodiscard]] float snapFactor(const float value, const float snap_ratio) {
            const float step = std::max(0.01f, snap_ratio);
            return std::max(MIN_SCALE_FACTOR, std::round(value / step) * step);
        }

        [[nodiscard]] std::array<glm::vec2, 4> planeQuad(const ScaleGizmoConfig& config,
                                                         const glm::mat4& view_projection,
                                                         const std::array<AxisVisual, 3>& axes,
                                                         const PlaneVisual& plane,
                                                         const float world_per_pixel,
                                                         bool& valid) {
            const glm::vec3 a = axes[static_cast<size_t>(plane.axis_a)].direction;
            const glm::vec3 b = axes[static_cast<size_t>(plane.axis_b)].direction;
            const float near_world = PLANE_NEAR_PX * world_per_pixel;
            const float far_world = PLANE_FAR_PX * world_per_pixel;
            const std::array<glm::vec3, 4> world = {
                config.pivot_world + a * near_world + b * near_world,
                config.pivot_world + a * far_world + b * near_world,
                config.pivot_world + a * far_world + b * far_world,
                config.pivot_world + a * near_world + b * far_world,
            };

            std::array<glm::vec2, 4> screen{};
            valid = true;
            for (size_t i = 0; i < world.size(); ++i) {
                valid = valid &&
                        projectPoint(view_projection, config.viewport_pos, config.viewport_size, world[i], screen[i]);
            }
            return screen;
        }

        [[nodiscard]] std::array<ProjectedPlane, 3> projectPlanes(
            const ScaleGizmoConfig& config,
            const glm::mat4& view_projection,
            const std::array<AxisVisual, 3>& axes,
            const std::array<PlaneVisual, 3>& planes,
            const float world_per_pixel) {
            std::array<ProjectedPlane, 3> projected{};
            for (size_t i = 0; i < planes.size(); ++i) {
                projected[i].plane = planes[i];
                projected[i].quad = planeQuad(config, view_projection, axes, planes[i], world_per_pixel,
                                              projected[i].valid);
            }
            return projected;
        }

        [[nodiscard]] bool axisSegment(const ScaleGizmoConfig& config,
                                       const glm::mat4& view_projection,
                                       const AxisVisual& axis,
                                       const float world_per_pixel,
                                       glm::vec2& start,
                                       glm::vec2& end) {
            const glm::vec3 start_world = config.pivot_world + axis.direction * (AXIS_START_PX * world_per_pixel);
            const glm::vec3 end_world = config.pivot_world + axis.direction * (AXIS_LENGTH_PX * world_per_pixel);
            return projectPoint(view_projection, config.viewport_pos, config.viewport_size, start_world, start) &&
                   projectPoint(view_projection, config.viewport_pos, config.viewport_size, end_world, end);
        }

        [[nodiscard]] std::array<ProjectedAxis, 3> projectAxes(
            const ScaleGizmoConfig& config,
            const glm::mat4& view_projection,
            const std::array<AxisVisual, 3>& axes,
            const float world_per_pixel) {
            std::array<ProjectedAxis, 3> projected{};
            for (size_t i = 0; i < axes.size(); ++i) {
                projected[i].axis = axes[i];
                projected[i].valid = axisSegment(config, view_projection, axes[i], world_per_pixel,
                                                 projected[i].start, projected[i].end);
            }
            return projected;
        }

        void drawPlaneHandle(NativeOverlayDrawList& draw_list,
                             const ProjectedPlane& projected,
                             const bool emphasized,
                             const bool active) {
            if (!projected.valid) {
                return;
            }

            const glm::vec2 points[4] = {
                glm::vec2(projected.quad[0].x, projected.quad[0].y),
                glm::vec2(projected.quad[1].x, projected.quad[1].y),
                glm::vec2(projected.quad[2].x, projected.quad[2].y),
                glm::vec2(projected.quad[3].x, projected.quad[3].y),
            };
            draw_list.AddConvexPolyFilled(points, 4,
                                          withAlpha(projected.plane.color, active ? 0.36f : (emphasized ? 0.25f : 0.12f)));
            draw_list.AddPolyline(points, 4, overlayColor(0, 0, 0, active ? 165 : 95), true, active ? 2.6f : 2.0f);
            draw_list.AddPolyline(points, 4, withAlpha(projected.plane.color, active ? 0.95f : 0.68f),
                                  true, active ? 1.8f : 1.2f);
        }

        void drawAxisHandle(NativeOverlayDrawList& draw_list,
                            const ProjectedAxis& projected,
                            const bool emphasized,
                            const bool active) {
            if (!projected.valid) {
                return;
            }

            const glm::vec2 dir = safeNormalize(projected.end - projected.start, glm::vec2(1.0f, 0.0f));
            const glm::vec2 normal(-dir.y, dir.x);
            const float line_width = active ? 4.0f : (emphasized ? 3.2f : 2.2f);
            const float alpha = active || emphasized ? 1.0f : 0.78f;
            draw_list.AddLine(glm::vec2(projected.start.x, projected.start.y), glm::vec2(projected.end.x, projected.end.y),
                              overlayColor(0, 0, 0, active ? 185 : 115), line_width + 3.0f);
            draw_list.AddLine(glm::vec2(projected.start.x, projected.start.y), glm::vec2(projected.end.x, projected.end.y),
                              withAlpha(projected.axis.color, alpha), line_width);

            const float half = active ? AXIS_BOX_HALF_PX + 1.4f : (emphasized ? AXIS_BOX_HALF_PX + 0.8f : AXIS_BOX_HALF_PX);
            const std::array<glm::vec2, 4> shadow = {
                projected.end + dir * (half + 1.8f) + normal * (half + 1.8f),
                projected.end - dir * (half + 1.8f) + normal * (half + 1.8f),
                projected.end - dir * (half + 1.8f) - normal * (half + 1.8f),
                projected.end + dir * (half + 1.8f) - normal * (half + 1.8f),
            };
            const std::array<glm::vec2, 4> box = {
                projected.end + dir * half + normal * half,
                projected.end - dir * half + normal * half,
                projected.end - dir * half - normal * half,
                projected.end + dir * half - normal * half,
            };
            const glm::vec2 shadow_points[4] = {
                glm::vec2(shadow[0].x, shadow[0].y),
                glm::vec2(shadow[1].x, shadow[1].y),
                glm::vec2(shadow[2].x, shadow[2].y),
                glm::vec2(shadow[3].x, shadow[3].y),
            };
            const glm::vec2 box_points[4] = {
                glm::vec2(box[0].x, box[0].y),
                glm::vec2(box[1].x, box[1].y),
                glm::vec2(box[2].x, box[2].y),
                glm::vec2(box[3].x, box[3].y),
            };
            draw_list.AddConvexPolyFilled(shadow_points, 4, overlayColor(0, 0, 0, active ? 185 : 125));
            draw_list.AddConvexPolyFilled(box_points, 4, withAlpha(projected.axis.color, alpha));
        }

        void drawCenterHandle(NativeOverlayDrawList& draw_list,
                              const glm::vec2& pivot_screen,
                              const bool emphasized,
                              const bool active) {
            const float half = active ? 7.5f : (emphasized ? 7.0f : 6.0f);
            const glm::vec2 min(pivot_screen.x - half, pivot_screen.y - half);
            const glm::vec2 max(pivot_screen.x + half, pivot_screen.y + half);
            draw_list.AddRectFilled(glm::vec2(min.x - 2.0f, min.y - 2.0f),
                                    glm::vec2(max.x + 2.0f, max.y + 2.0f),
                                    overlayColor(0, 0, 0, active ? 185 : 135),
                                    2.0f);
            draw_list.AddRectFilled(min, max,
                                    active ? overlayColor(255, 214, 119, 245)
                                           : (emphasized ? overlayColor(255, 235, 170, 238)
                                                         : overlayColor(245, 248, 255, 220)),
                                    2.0f);
        }

        [[nodiscard]] ScaleGizmoHandle nearestHandle(const std::array<ProjectedAxis, 3>& axes,
                                                     const std::array<ProjectedPlane, 3>& planes,
                                                     const glm::vec2& mouse,
                                                     const glm::vec2& pivot_screen) {
            constexpr float center_hit_radius2 = CENTER_HIT_RADIUS_PX * CENTER_HIT_RADIUS_PX;
            constexpr float axis_hit_threshold2 = AXIS_HIT_THRESHOLD_PX * AXIS_HIT_THRESHOLD_PX;
            constexpr float axis_box_hit_radius2 = AXIS_BOX_HIT_RADIUS_PX * AXIS_BOX_HIT_RADIUS_PX;
            if (lengthSquared(mouse - pivot_screen) <= center_hit_radius2) {
                return ScaleGizmoHandle::Uniform;
            }

            for (const auto& plane : planes) {
                if (plane.valid && pointInConvexQuad(mouse, plane.quad)) {
                    return plane.plane.handle;
                }
            }

            float best_distance2 = std::numeric_limits<float>::max();
            ScaleGizmoHandle best_handle = ScaleGizmoHandle::None;
            for (const auto& axis : axes) {
                if (!axis.valid) {
                    continue;
                }
                const float box_distance2 = lengthSquared(mouse - axis.end);
                const float line_distance2 = distanceToSegmentSquared(mouse, axis.start, axis.end);
                const float distance2 = std::min(box_distance2, line_distance2);
                if (distance2 < best_distance2) {
                    best_distance2 = distance2;
                    best_handle = axis.axis.handle;
                }
            }

            return best_distance2 <= axis_box_hit_radius2 ||
                           best_distance2 <= axis_hit_threshold2
                       ? best_handle
                       : ScaleGizmoHandle::None;
        }

        [[nodiscard]] glm::vec3 targetScaleForMouse(const ScaleGizmoConfig& config,
                                                    const glm::vec2& mouse) {
            float factor = 1.0f + glm::dot(mouse - g_active.start_mouse, g_active.screen_axis) / SCALE_DRAG_PIXELS;
            factor = std::max(MIN_SCALE_FACTOR, factor);
            if (config.snap) {
                factor = snapFactor(factor, config.snap_ratio);
            }

            glm::vec3 scale(1.0f);
            for (int axis = 0; axis < 3; ++axis) {
                if (g_active.scale_axes[axis] != 0) {
                    scale[axis] = factor;
                }
            }
            return scale;
        }

        [[nodiscard]] glm::vec2 projectedAxisForHandle(const ScaleGizmoConfig& config,
                                                       const glm::mat4& view_projection,
                                                       const std::array<AxisVisual, 3>& axes,
                                                       const ScaleGizmoHandle handle,
                                                       const glm::vec2& mouse,
                                                       const glm::vec2& pivot_screen,
                                                       const float world_per_pixel) {
            if (handle == ScaleGizmoHandle::Uniform) {
                const glm::vec2 radial = mouse - pivot_screen;
                if (glm::length(radial) > CENTER_HIT_RADIUS_PX) {
                    return safeNormalize(radial, safeNormalize(glm::vec2(1.0f, -1.0f), glm::vec2(1.0f, 0.0f)));
                }
                return safeNormalize(glm::vec2(1.0f, -1.0f), glm::vec2(1.0f, 0.0f));
            }

            glm::vec3 direction_world(0.0f);
            const glm::ivec3 scale_axes = scaleAxesForHandle(handle);
            for (int axis = 0; axis < 3; ++axis) {
                if (scale_axes[axis] != 0) {
                    direction_world += axes[static_cast<size_t>(axis)].direction;
                }
            }
            direction_world = safeNormalize(direction_world, axes[0].direction);

            glm::vec2 handle_screen;
            if (projectPoint(view_projection, config.viewport_pos, config.viewport_size,
                             config.pivot_world + direction_world * (AXIS_LENGTH_PX * world_per_pixel),
                             handle_screen)) {
                return safeNormalize(handle_screen - pivot_screen, glm::vec2(1.0f, 0.0f));
            }
            return glm::vec2(1.0f, 0.0f);
        }

        void beginDrag(const ScaleGizmoConfig& config,
                       const glm::mat4& view_projection,
                       const std::array<AxisVisual, 3>& axes,
                       const ScaleGizmoHandle handle,
                       const glm::vec2& mouse,
                       const glm::vec2& pivot_screen,
                       const float world_per_pixel) {
            g_active = ActiveState{};
            g_active.active = true;
            g_active.id = config.id;
            g_active.handle = handle;
            g_active.pivot_world = config.pivot_world;
            g_active.orientation_world = config.orientation_world;
            g_active.start_mouse = mouse;
            g_active.scale_axes = scaleAxesForHandle(handle);
            g_active.screen_axis = projectedAxisForHandle(config, view_projection, axes, handle, mouse,
                                                          pivot_screen, world_per_pixel);
        }
    } // namespace

    ScaleGizmoResult drawScaleGizmo(const ScaleGizmoConfig& config) {
        ScaleGizmoResult result;
        g_hovered = false;

        if (!config.draw_list || config.viewport_size.x <= 1.0f || config.viewport_size.y <= 1.0f) {
            return result;
        }

        ScaleGizmoConfig draw_config = config;
        if (g_active.active && g_active.id == config.id) {
            draw_config.pivot_world = g_active.pivot_world;
            draw_config.orientation_world = g_active.orientation_world;
        }

        const glm::mat4 view_projection = draw_config.projection * draw_config.view;
        glm::vec2 pivot_screen;
        if (!projectPoint(view_projection, draw_config.viewport_pos, draw_config.viewport_size,
                          draw_config.pivot_world, pivot_screen)) {
            if (g_active.active && g_active.id == config.id && !config.input.mouse_left_down) {
                g_active = ActiveState{};
            }
            return result;
        }

        const float pixels_per_world = estimatePixelsPerWorld(draw_config, view_projection, pivot_screen);
        const float world_per_pixel = 1.0f / std::max(pixels_per_world, 0.000001f);
        const std::array<AxisVisual, 3> axes = {
            AxisVisual{ScaleGizmoHandle::X, axisFromConfig(draw_config.orientation_world, 0),
                       overlayColor(245, 82, 96, 255)},
            AxisVisual{ScaleGizmoHandle::Y, axisFromConfig(draw_config.orientation_world, 1),
                       overlayColor(74, 208, 119, 255)},
            AxisVisual{ScaleGizmoHandle::Z, axisFromConfig(draw_config.orientation_world, 2),
                       overlayColor(80, 151, 255, 255)},
        };
        const std::array<PlaneVisual, 3> planes = {
            PlaneVisual{ScaleGizmoHandle::XY, 0, 1, overlayColor(245, 215, 84, 255)},
            PlaneVisual{ScaleGizmoHandle::YZ, 1, 2, overlayColor(78, 222, 210, 255)},
            PlaneVisual{ScaleGizmoHandle::ZX, 2, 0, overlayColor(205, 132, 255, 255)},
        };
        const auto projected_planes = projectPlanes(draw_config, view_projection, axes, planes, world_per_pixel);
        const auto projected_axes = projectAxes(draw_config, view_projection, axes, world_per_pixel);

        const glm::vec2 mouse = config.input.mouse_pos;
        ScaleGizmoHandle hovered_handle = ScaleGizmoHandle::None;
        if (isInViewport(draw_config, mouse) && (!g_active.active || g_active.id == config.id)) {
            hovered_handle = nearestHandle(projected_axes, projected_planes, mouse, pivot_screen);
        }

        if (g_active.active && g_active.id == config.id) {
            result.active = config.input.mouse_left_down;
            if (!result.active) {
                g_active = ActiveState{};
            }
        }

        if (!g_active.active && config.input_enabled &&
            hovered_handle != ScaleGizmoHandle::None &&
            config.input.mouse_left_clicked) {
            beginDrag(draw_config, view_projection, axes, hovered_handle, mouse, pivot_screen, world_per_pixel);
            result.active = true;
        }

        if (g_active.active && g_active.id == config.id) {
            const glm::vec3 target_scale = targetScaleForMouse(config, mouse);
            const glm::vec3 delta_scale = target_scale / glm::max(g_active.applied_scale, glm::vec3(MIN_SCALE_FACTOR));
            if (lengthSquared(delta_scale - glm::vec3(1.0f)) > DELTA_EPSILON * DELTA_EPSILON) {
                result.changed = true;
                result.delta_scale = delta_scale;
                g_active.applied_scale = target_scale;
            }
            result.total_scale = g_active.applied_scale;
            result.active = true;
            result.active_handle = g_active.handle;
        }

        result.hovered_handle = hovered_handle;
        result.hovered = hovered_handle != ScaleGizmoHandle::None;
        g_hovered = result.hovered || result.active;

        const ScaleGizmoHandle emphasized = result.active ? g_active.handle : hovered_handle;
        for (const auto& plane : projected_planes) {
            drawPlaneHandle(*draw_config.draw_list, plane,
                            emphasized == plane.plane.handle && !result.active,
                            emphasized == plane.plane.handle && result.active);
        }
        for (const auto& axis : projected_axes) {
            drawAxisHandle(*draw_config.draw_list, axis,
                           emphasized == axis.axis.handle && !result.active,
                           emphasized == axis.axis.handle && result.active);
        }

        drawCenterHandle(*draw_config.draw_list, pivot_screen,
                         emphasized == ScaleGizmoHandle::Uniform && !result.active,
                         emphasized == ScaleGizmoHandle::Uniform && result.active);

        return result;
    }

    bool isScaleGizmoHovered() {
        return g_hovered;
    }

    bool isScaleGizmoActive() {
        return g_active.active;
    }

} // namespace lfs::vis::gui
