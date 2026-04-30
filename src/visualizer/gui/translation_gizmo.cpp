/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/translation_gizmo.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace lfs::vis::gui {
    namespace {
        constexpr float AXIS_LENGTH_PX = 106.0f;
        constexpr float AXIS_START_PX = 13.0f;
        constexpr float AXIS_HIT_THRESHOLD_PX = 10.0f;
        constexpr float CENTER_HIT_RADIUS_PX = 12.0f;
        constexpr float PLANE_NEAR_PX = 24.0f;
        constexpr float PLANE_FAR_PX = 48.0f;
        constexpr float DELTA_EPSILON = 0.000001f;

        struct AxisVisual {
            TranslationGizmoHandle handle = TranslationGizmoHandle::None;
            glm::vec3 direction{0.0f};
            OverlayColor color = OVERLAY_COL32_WHITE;
        };

        struct PlaneVisual {
            TranslationGizmoHandle handle = TranslationGizmoHandle::None;
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
            TranslationGizmoHandle handle = TranslationGizmoHandle::None;
            glm::vec3 pivot_world{0.0f};
            glm::mat3 orientation_world{1.0f};
            glm::mat4 view{1.0f};
            glm::mat4 projection{1.0f};
            glm::vec3 axis_world{1.0f, 0.0f, 0.0f};
            glm::vec3 plane_normal_world{0.0f, 0.0f, 1.0f};
            glm::vec3 snap_axis_a{1.0f, 0.0f, 0.0f};
            glm::vec3 snap_axis_b{0.0f, 1.0f, 0.0f};
            glm::vec3 start_plane_point{0.0f};
            glm::vec2 start_mouse{0.0f};
            glm::vec2 screen_axis{1.0f, 0.0f};
            float axis_pixels_per_world = 64.0f;
            glm::vec3 applied_translation{0.0f};
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

        [[nodiscard]] glm::vec3 cameraForward(const glm::mat4& view) {
            return safeNormalize(
                -glm::vec3(view[0][2], view[1][2], view[2][2]),
                glm::vec3(0.0f, 0.0f, -1.0f));
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

        [[nodiscard]] bool isInViewport(const TranslationGizmoConfig& config, const glm::vec2& p) {
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

        [[nodiscard]] bool unprojectRay(const TranslationGizmoConfig& config,
                                        const glm::vec2& mouse,
                                        glm::vec3& origin,
                                        glm::vec3& direction) {
            if (config.viewport_size.x <= 0.0f || config.viewport_size.y <= 0.0f) {
                return false;
            }

            const float ndc_x = ((mouse.x - config.viewport_pos.x) / config.viewport_size.x) * 2.0f - 1.0f;
            const float ndc_y = 1.0f - ((mouse.y - config.viewport_pos.y) / config.viewport_size.y) * 2.0f;
            const glm::mat4 inv_vp = glm::inverse(config.projection * config.view);
            glm::vec4 near_point = inv_vp * glm::vec4(ndc_x, ndc_y, -1.0f, 1.0f);
            glm::vec4 far_point = inv_vp * glm::vec4(ndc_x, ndc_y, 1.0f, 1.0f);
            if (std::abs(near_point.w) <= std::numeric_limits<float>::epsilon() ||
                std::abs(far_point.w) <= std::numeric_limits<float>::epsilon()) {
                return false;
            }

            near_point /= near_point.w;
            far_point /= far_point.w;
            origin = glm::vec3(near_point);
            direction = safeNormalize(glm::vec3(far_point - near_point), cameraForward(config.view));
            return true;
        }

        [[nodiscard]] bool intersectPlane(const TranslationGizmoConfig& config,
                                          const glm::vec2& mouse,
                                          const glm::vec3& plane_point,
                                          const glm::vec3& plane_normal,
                                          glm::vec3& point) {
            glm::vec3 ray_origin;
            glm::vec3 ray_dir;
            if (!unprojectRay(config, mouse, ray_origin, ray_dir)) {
                return false;
            }

            constexpr float MIN_RAY_PLANE_DOT = 0.05f;
            const float denom = glm::dot(ray_dir, plane_normal);
            if (std::abs(denom) < MIN_RAY_PLANE_DOT) {
                return false;
            }

            const float t = glm::dot(plane_point - ray_origin, plane_normal) / denom;
            point = ray_origin + ray_dir * t;
            return std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z);
        }

        [[nodiscard]] float estimatePixelsPerWorld(const TranslationGizmoConfig& config,
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

        [[nodiscard]] float snapScalar(const float value, const float snap_units) {
            const float step = std::max(0.0001f, snap_units);
            return std::round(value / step) * step;
        }

        [[nodiscard]] glm::vec3 snapDelta(const glm::vec3& delta,
                                          const TranslationGizmoConfig& config) {
            if (!config.snap) {
                return delta;
            }

            switch (g_active.handle) {
            case TranslationGizmoHandle::X:
            case TranslationGizmoHandle::Y:
            case TranslationGizmoHandle::Z:
                return g_active.axis_world * snapScalar(glm::dot(delta, g_active.axis_world), config.snap_units);
            case TranslationGizmoHandle::XY:
            case TranslationGizmoHandle::YZ:
            case TranslationGizmoHandle::ZX:
            case TranslationGizmoHandle::View:
                return g_active.snap_axis_a * snapScalar(glm::dot(delta, g_active.snap_axis_a), config.snap_units) +
                       g_active.snap_axis_b * snapScalar(glm::dot(delta, g_active.snap_axis_b), config.snap_units);
            case TranslationGizmoHandle::None:
            default:
                return delta;
            }
        }

        [[nodiscard]] glm::vec3 planeNormalForHandle(const TranslationGizmoHandle handle,
                                                     const std::array<AxisVisual, 3>& axes,
                                                     const glm::vec3& view_axis) {
            switch (handle) {
            case TranslationGizmoHandle::XY: return axes[2].direction;
            case TranslationGizmoHandle::YZ: return axes[0].direction;
            case TranslationGizmoHandle::ZX: return axes[1].direction;
            case TranslationGizmoHandle::View: return view_axis;
            default: return view_axis;
            }
        }

        void setPlaneSnapAxes(const TranslationGizmoHandle handle,
                              const std::array<AxisVisual, 3>& axes,
                              const TranslationGizmoConfig& config) {
            switch (handle) {
            case TranslationGizmoHandle::XY:
                g_active.snap_axis_a = axes[0].direction;
                g_active.snap_axis_b = axes[1].direction;
                break;
            case TranslationGizmoHandle::YZ:
                g_active.snap_axis_a = axes[1].direction;
                g_active.snap_axis_b = axes[2].direction;
                break;
            case TranslationGizmoHandle::ZX:
                g_active.snap_axis_a = axes[2].direction;
                g_active.snap_axis_b = axes[0].direction;
                break;
            case TranslationGizmoHandle::View:
                g_active.snap_axis_a = cameraRight(config.view);
                g_active.snap_axis_b = cameraUp(config.view);
                break;
            default:
                break;
            }
        }

        [[nodiscard]] std::array<glm::vec2, 4> planeQuad(const TranslationGizmoConfig& config,
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
            const TranslationGizmoConfig& config,
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
                                          withAlpha(projected.plane.color, active ? 0.34f : (emphasized ? 0.24f : 0.12f)));
            draw_list.AddPolyline(points, 4, overlayColor(0, 0, 0, active ? 160 : 90), true, active ? 2.6f : 2.0f);
            draw_list.AddPolyline(points, 4, withAlpha(projected.plane.color, active ? 0.95f : 0.65f),
                                  true, active ? 1.8f : 1.2f);
        }

        [[nodiscard]] bool axisSegment(const TranslationGizmoConfig& config,
                                       const glm::mat4& view_projection,
                                       const AxisVisual& axis,
                                       const float axis_sign,
                                       const float world_per_pixel,
                                       glm::vec2& start,
                                       glm::vec2& end) {
            const glm::vec3 direction = axis.direction * axis_sign;
            const glm::vec3 start_world = config.pivot_world + direction * (AXIS_START_PX * world_per_pixel);
            const glm::vec3 end_world = config.pivot_world + direction * (AXIS_LENGTH_PX * world_per_pixel);
            return projectPoint(view_projection, config.viewport_pos, config.viewport_size, start_world, start) &&
                   projectPoint(view_projection, config.viewport_pos, config.viewport_size, end_world, end);
        }

        [[nodiscard]] std::array<ProjectedAxis, 3> projectAxes(
            const TranslationGizmoConfig& config,
            const glm::mat4& view_projection,
            const std::array<AxisVisual, 3>& axes,
            const float world_per_pixel) {
            std::array<ProjectedAxis, 3> projected{};
            for (size_t i = 0; i < axes.size(); ++i) {
                projected[i].axis = axes[i];
                projected[i].valid = axisSegment(config, view_projection, axes[i], 1.0f, world_per_pixel,
                                                 projected[i].start, projected[i].end);
            }
            return projected;
        }

        void drawAxisSide(NativeOverlayDrawList& draw_list,
                          const glm::vec2& start,
                          const glm::vec2& end,
                          const OverlayColor color,
                          const float alpha,
                          const bool emphasized,
                          const bool active) {
            const glm::vec2 dir = safeNormalize(end - start, glm::vec2(1.0f, 0.0f));
            const glm::vec2 normal(-dir.y, dir.x);
            const float width = active ? 4.0f : (emphasized ? 3.2f : 2.2f);
            draw_list.AddLine(glm::vec2(start.x, start.y), glm::vec2(end.x, end.y), overlayColor(0, 0, 0, active ? 185 : 115), width + 3.0f);
            draw_list.AddLine(glm::vec2(start.x, start.y), glm::vec2(end.x, end.y), withAlpha(color, alpha), width);

            const float head_length = active ? 15.0f : 13.0f;
            const float head_width = active ? 8.0f : 7.0f;
            const glm::vec2 p0 = end;
            const glm::vec2 p1 = end - dir * head_length + normal * head_width;
            const glm::vec2 p2 = end - dir * head_length - normal * head_width;
            draw_list.AddTriangleFilled(glm::vec2(p0.x, p0.y), glm::vec2(p1.x, p1.y), glm::vec2(p2.x, p2.y),
                                        overlayColor(0, 0, 0, active ? 180 : 110));
            const float inset = 1.5f;
            draw_list.AddTriangleFilled(
                glm::vec2(p0.x, p0.y),
                glm::vec2(p1.x + dir.x * inset, p1.y + dir.y * inset),
                glm::vec2(p2.x + dir.x * inset, p2.y + dir.y * inset),
                withAlpha(color, alpha));
        }

        void drawAxisHandle(NativeOverlayDrawList& draw_list,
                            const ProjectedAxis& projected,
                            const bool emphasized,
                            const bool active) {
            if (projected.valid) {
                drawAxisSide(draw_list, projected.start, projected.end, projected.axis.color,
                             active || emphasized ? 1.0f : 0.78f, emphasized, active);
            }
        }

        [[nodiscard]] TranslationGizmoHandle nearestHandle(const std::array<ProjectedAxis, 3>& axes,
                                                           const std::array<ProjectedPlane, 3>& planes,
                                                           const glm::vec2& mouse,
                                                           const glm::vec2& pivot_screen) {
            constexpr float center_hit_radius2 = CENTER_HIT_RADIUS_PX * CENTER_HIT_RADIUS_PX;
            constexpr float axis_hit_threshold2 = AXIS_HIT_THRESHOLD_PX * AXIS_HIT_THRESHOLD_PX;
            if (lengthSquared(mouse - pivot_screen) <= center_hit_radius2) {
                return TranslationGizmoHandle::View;
            }

            for (const auto& plane : planes) {
                if (plane.valid && pointInConvexQuad(mouse, plane.quad)) {
                    return plane.plane.handle;
                }
            }

            float best_distance2 = std::numeric_limits<float>::max();
            TranslationGizmoHandle best_handle = TranslationGizmoHandle::None;
            for (const auto& axis : axes) {
                if (!axis.valid) {
                    continue;
                }
                const float distance2 = distanceToSegmentSquared(mouse, axis.start, axis.end);
                if (distance2 < best_distance2) {
                    best_distance2 = distance2;
                    best_handle = axis.axis.handle;
                }
            }

            return best_distance2 <= axis_hit_threshold2 ? best_handle : TranslationGizmoHandle::None;
        }

        [[nodiscard]] glm::vec3 translationForMouse(const TranslationGizmoConfig& config,
                                                    const glm::vec2& mouse) {
            switch (g_active.handle) {
            case TranslationGizmoHandle::X:
            case TranslationGizmoHandle::Y:
            case TranslationGizmoHandle::Z: {
                const float distance_world =
                    glm::dot(mouse - g_active.start_mouse, g_active.screen_axis) /
                    std::max(0.0001f, g_active.axis_pixels_per_world);
                return g_active.axis_world * distance_world;
            }
            case TranslationGizmoHandle::XY:
            case TranslationGizmoHandle::YZ:
            case TranslationGizmoHandle::ZX:
            case TranslationGizmoHandle::View: {
                glm::vec3 current_point;
                if (!intersectPlane(config, mouse, g_active.pivot_world, g_active.plane_normal_world, current_point)) {
                    return g_active.applied_translation;
                }
                const glm::vec3 raw_delta = current_point - g_active.start_plane_point;
                return raw_delta - g_active.plane_normal_world * glm::dot(raw_delta, g_active.plane_normal_world);
            }
            case TranslationGizmoHandle::None:
            default:
                return g_active.applied_translation;
            }
        }

        void beginDrag(const TranslationGizmoConfig& config,
                       const glm::mat4& view_projection,
                       const std::array<AxisVisual, 3>& axes,
                       const TranslationGizmoHandle handle,
                       const glm::vec2& mouse,
                       const glm::vec2& pivot_screen,
                       const float world_per_pixel) {
            g_active = ActiveState{};
            g_active.active = true;
            g_active.id = config.id;
            g_active.handle = handle;
            g_active.pivot_world = config.pivot_world;
            g_active.orientation_world = config.orientation_world;
            g_active.view = config.view;
            g_active.projection = config.projection;
            g_active.start_mouse = mouse;
            g_active.axis_pixels_per_world = 1.0f / std::max(world_per_pixel, 0.000001f);

            const glm::vec3 view_axis = cameraForward(config.view);
            g_active.plane_normal_world = planeNormalForHandle(handle, axes, view_axis);
            setPlaneSnapAxes(handle, axes, config);

            switch (handle) {
            case TranslationGizmoHandle::X:
            case TranslationGizmoHandle::Y:
            case TranslationGizmoHandle::Z: {
                const int axis_index = static_cast<int>(handle);
                g_active.axis_world = axes[static_cast<size_t>(axis_index)].direction;
                glm::vec2 axis_screen;
                const float axis_world_length = AXIS_LENGTH_PX * world_per_pixel;
                if (projectPoint(view_projection, config.viewport_pos, config.viewport_size,
                                 config.pivot_world + g_active.axis_world * axis_world_length,
                                 axis_screen)) {
                    const glm::vec2 projected_axis = axis_screen - pivot_screen;
                    const float projected_length = glm::length(projected_axis);
                    g_active.screen_axis = safeNormalize(projected_axis, glm::vec2(1.0f, 0.0f));
                    if (projected_length > 1.0f && axis_world_length > 0.000001f) {
                        g_active.axis_pixels_per_world = projected_length / axis_world_length;
                    }
                }
                break;
            }
            case TranslationGizmoHandle::XY:
            case TranslationGizmoHandle::YZ:
            case TranslationGizmoHandle::ZX:
            case TranslationGizmoHandle::View:
                if (!intersectPlane(config, mouse, g_active.pivot_world, g_active.plane_normal_world,
                                    g_active.start_plane_point)) {
                    g_active.start_plane_point = g_active.pivot_world;
                }
                break;
            case TranslationGizmoHandle::None:
            default:
                break;
            }
        }
    } // namespace

    TranslationGizmoResult drawTranslationGizmo(const TranslationGizmoConfig& config) {
        TranslationGizmoResult result;
        g_hovered = false;

        if (!config.draw_list || config.viewport_size.x <= 1.0f || config.viewport_size.y <= 1.0f) {
            return result;
        }

        TranslationGizmoConfig draw_config = config;
        if (g_active.active && g_active.id == config.id) {
            draw_config.pivot_world = g_active.pivot_world + g_active.applied_translation;
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
            AxisVisual{TranslationGizmoHandle::X, axisFromConfig(draw_config.orientation_world, 0),
                       overlayColor(245, 82, 96, 255)},
            AxisVisual{TranslationGizmoHandle::Y, axisFromConfig(draw_config.orientation_world, 1),
                       overlayColor(74, 208, 119, 255)},
            AxisVisual{TranslationGizmoHandle::Z, axisFromConfig(draw_config.orientation_world, 2),
                       overlayColor(80, 151, 255, 255)},
        };
        const std::array<PlaneVisual, 3> planes = {
            PlaneVisual{TranslationGizmoHandle::XY, 0, 1, overlayColor(245, 215, 84, 255)},
            PlaneVisual{TranslationGizmoHandle::YZ, 1, 2, overlayColor(78, 222, 210, 255)},
            PlaneVisual{TranslationGizmoHandle::ZX, 2, 0, overlayColor(205, 132, 255, 255)},
        };
        const auto projected_planes = projectPlanes(draw_config, view_projection, axes, planes, world_per_pixel);
        const auto projected_axes = projectAxes(draw_config, view_projection, axes, world_per_pixel);

        const glm::vec2 mouse = config.input.mouse_pos;
        TranslationGizmoHandle hovered_handle = TranslationGizmoHandle::None;
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
            hovered_handle != TranslationGizmoHandle::None &&
            config.input.mouse_left_clicked) {
            beginDrag(draw_config, view_projection, axes, hovered_handle, mouse, pivot_screen, world_per_pixel);
            result.active = true;
        }

        if (g_active.active && g_active.id == config.id) {
            TranslationGizmoConfig drag_config = config;
            drag_config.pivot_world = g_active.pivot_world;
            drag_config.orientation_world = g_active.orientation_world;
            drag_config.view = g_active.view;
            drag_config.projection = g_active.projection;

            const glm::vec3 target_translation = snapDelta(translationForMouse(drag_config, mouse), config);
            const glm::vec3 delta = target_translation - g_active.applied_translation;
            if (lengthSquared(delta) > DELTA_EPSILON * DELTA_EPSILON) {
                result.changed = true;
                result.delta_translation = delta;
                g_active.applied_translation = target_translation;
            }
            result.total_translation = g_active.applied_translation;
            result.active = true;
            result.active_handle = g_active.handle;
        }

        result.hovered_handle = hovered_handle;
        result.hovered = hovered_handle != TranslationGizmoHandle::None;
        g_hovered = result.hovered || result.active;

        const TranslationGizmoHandle emphasized = result.active ? g_active.handle : hovered_handle;
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

        const bool center_active = emphasized == TranslationGizmoHandle::View && result.active;
        const bool center_hovered = emphasized == TranslationGizmoHandle::View && !result.active;
        draw_config.draw_list->AddCircleFilled(glm::vec2(pivot_screen.x, pivot_screen.y),
                                               center_active ? 7.5f : (center_hovered ? 7.0f : 6.0f),
                                               overlayColor(0, 0, 0, center_active ? 180 : 135), 24);
        draw_config.draw_list->AddCircleFilled(glm::vec2(pivot_screen.x, pivot_screen.y),
                                               center_active ? 5.5f : (center_hovered ? 5.0f : 4.2f),
                                               overlayColor(245, 248, 255, center_active ? 245 : 220), 24);

        return result;
    }

    bool isTranslationGizmoHovered() {
        return g_hovered;
    }

    bool isTranslationGizmoActive() {
        return g_active.active;
    }

} // namespace lfs::vis::gui
