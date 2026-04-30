/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rotation_gizmo.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <limits>

namespace lfs::vis::gui {
    namespace {
        constexpr int RING_SEGMENTS = 96;
        constexpr float HIT_THRESHOLD_PX = 9.0f;
        constexpr float MIN_SCREEN_RADIUS = 54.0f;
        constexpr float MAX_SCREEN_RADIUS = 118.0f;
        constexpr float VIEW_RING_OFFSET_PX = 13.0f;
        constexpr float ANGLE_EPSILON = 0.0005f;
        constexpr float MIN_FULL_INFLUENCE_DISTANCE_PX = 64.0f;
        constexpr float MIN_ZERO_INFLUENCE_DISTANCE_PX = 170.0f;

        struct AxisVisual {
            RotationGizmoAxis axis = RotationGizmoAxis::None;
            glm::vec3 direction{0.0f};
            OverlayColor color = OVERLAY_COL32_WHITE;
        };

        struct RingCache {
            AxisVisual axis;
            std::array<glm::vec2, RING_SEGMENTS + 1> points{};
            std::array<glm::vec3, RING_SEGMENTS + 1> radials{};
            std::array<bool, RING_SEGMENTS + 1> valid{};
        };

        struct ActiveState {
            bool active = false;
            int id = 0;
            RotationGizmoAxis axis = RotationGizmoAxis::None;
            glm::vec3 axis_world{0.0f};
            glm::vec3 pivot_world{0.0f};
            glm::mat3 orientation_world{1.0f};
            glm::vec3 start_vector_world{1.0f, 0.0f, 0.0f};
            glm::vec2 start_mouse{0.0f};
            glm::vec2 screen_tangent{0.0f, 1.0f};
            float screen_pixels_per_radian = 64.0f;
            float applied_angle = 0.0f;
            float angle_offset = 0.0f;
            float previous_raw_angle = 0.0f;
            int rotations = 0;
            bool use_plane_drag = false;
        };

        ActiveState g_active;
        bool g_hovered = false;

        [[nodiscard]] float clampFinite(const float value, const float min_value, const float max_value) {
            if (!std::isfinite(value)) {
                return min_value;
            }
            return std::clamp(value, min_value, max_value);
        }

        [[nodiscard]] float smoothStep(const float edge0, const float edge1, const float x) {
            const float range = std::max(edge1 - edge0, std::numeric_limits<float>::epsilon());
            const float t = std::clamp((x - edge0) / range, 0.0f, 1.0f);
            return t * t * (3.0f - 2.0f * t);
        }

        [[nodiscard]] float lengthSquared(const glm::vec2& v) {
            return v.x * v.x + v.y * v.y;
        }

        [[nodiscard]] float lengthSquared(const glm::vec3& v) {
            return v.x * v.x + v.y * v.y + v.z * v.z;
        }

        [[nodiscard]] glm::vec3 safeNormalize(const glm::vec3& v, const glm::vec3& fallback) {
            const float len2 = lengthSquared(v);
            if (len2 <= std::numeric_limits<float>::epsilon() || !std::isfinite(len2)) {
                return fallback;
            }
            return v / std::sqrt(len2);
        }

        [[nodiscard]] glm::vec2 safeNormalize(const glm::vec2& v, const glm::vec2& fallback) {
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

        [[nodiscard]] bool isInViewport(const RotationGizmoConfig& config, const glm::vec2& p) {
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

        [[nodiscard]] float colorAlpha(const glm::vec3& radial_world,
                                       const glm::vec3& camera_forward,
                                       const bool emphasized) {
            const float front = glm::dot(radial_world, camera_forward);
            const float alpha = front > 0.0f ? 0.38f : 0.82f;
            return emphasized ? 1.0f : alpha;
        }

        [[nodiscard]] OverlayColor withAlpha(const OverlayColor color, const float alpha) {
            const int r = static_cast<int>(color & 0xFF);
            const int g = static_cast<int>((color >> 8) & 0xFF);
            const int b = static_cast<int>((color >> 16) & 0xFF);
            const int a = static_cast<int>(std::clamp(alpha, 0.0f, 1.0f) * 255.0f);
            return overlayColor(r, g, b, a);
        }

        void planeBasis(const glm::vec3& normal, glm::vec3& u, glm::vec3& v) {
            const glm::vec3 reference = std::abs(normal.y) < 0.92f
                                            ? glm::vec3(0.0f, 1.0f, 0.0f)
                                            : glm::vec3(1.0f, 0.0f, 0.0f);
            u = safeNormalize(glm::cross(reference, normal), glm::vec3(1.0f, 0.0f, 0.0f));
            v = safeNormalize(glm::cross(normal, u), glm::vec3(0.0f, 1.0f, 0.0f));
        }

        [[nodiscard]] float estimateWorldRadius(const RotationGizmoConfig& config,
                                                const glm::mat4& view_projection,
                                                const glm::vec2& pivot_screen,
                                                const float target_radius_px) {
            const glm::vec3 right = cameraRight(config.view);
            const glm::vec3 up = cameraUp(config.view);
            glm::vec2 probe_screen;
            float pixels_per_world = 0.0f;
            if (projectPoint(view_projection, config.viewport_pos, config.viewport_size,
                             config.pivot_world + right, probe_screen)) {
                pixels_per_world = glm::length(probe_screen - pivot_screen);
            }
            if (pixels_per_world < 0.0001f &&
                projectPoint(view_projection, config.viewport_pos, config.viewport_size,
                             config.pivot_world + up, probe_screen)) {
                pixels_per_world = glm::length(probe_screen - pivot_screen);
            }
            if (pixels_per_world < 0.0001f || !std::isfinite(pixels_per_world)) {
                return 1.0f;
            }
            return target_radius_px / pixels_per_world;
        }

        [[nodiscard]] RingCache buildRingCache(
            const RotationGizmoConfig& config,
            const glm::mat4& view_projection,
            const AxisVisual& axis,
            const float radius_world) {
            RingCache cache;
            cache.axis = axis;
            glm::vec3 u;
            glm::vec3 v;
            planeBasis(axis.direction, u, v);

            for (int i = 0; i <= RING_SEGMENTS; ++i) {
                const float t = static_cast<float>(i) / static_cast<float>(RING_SEGMENTS);
                const float angle = t * glm::two_pi<float>();
                const glm::vec3 radial = u * std::cos(angle) + v * std::sin(angle);
                const size_t index = static_cast<size_t>(i);
                cache.radials[index] = radial;
                cache.valid[index] = projectPoint(view_projection, config.viewport_pos, config.viewport_size,
                                                  config.pivot_world + radial * radius_world,
                                                  cache.points[index]);
            }
            return cache;
        }

        [[nodiscard]] float distanceToPolylineSquared(const glm::vec2& p, const RingCache& ring) {
            float best_distance2 = std::numeric_limits<float>::max();
            for (int i = 0; i < RING_SEGMENTS; ++i) {
                const size_t a = static_cast<size_t>(i);
                const size_t b = static_cast<size_t>(i + 1);
                if (!ring.valid[a] || !ring.valid[b]) {
                    continue;
                }
                best_distance2 = std::min(best_distance2, distanceToSegmentSquared(
                                                              p,
                                                              ring.points[a],
                                                              ring.points[b]));
            }
            return best_distance2;
        }

        [[nodiscard]] float distanceToPolyline(const glm::vec2& p, const RingCache& ring) {
            return std::sqrt(distanceToPolylineSquared(p, ring));
        }

        void drawRing(NativeOverlayDrawList& draw_list,
                      const RingCache& ring,
                      const glm::vec3& camera_forward,
                      const bool hovered,
                      const bool active) {
            const float line_width = active ? 4.2f : (hovered ? 3.4f : 2.2f);

            for (int i = 0; i < RING_SEGMENTS; ++i) {
                const size_t a = static_cast<size_t>(i);
                const size_t b = static_cast<size_t>(i + 1);
                if (ring.valid[a] && ring.valid[b]) {
                    const auto& prev_screen = ring.points[a];
                    const auto& screen = ring.points[b];
                    const float alpha = colorAlpha((ring.radials[a] + ring.radials[b]) * 0.5f,
                                                   camera_forward, hovered || active);
                    draw_list.AddLine(
                        glm::vec2(prev_screen.x, prev_screen.y),
                        glm::vec2(screen.x, screen.y),
                        overlayColor(0, 0, 0, active ? 180 : 115),
                        line_width + 3.2f);
                    draw_list.AddLine(
                        glm::vec2(prev_screen.x, prev_screen.y),
                        glm::vec2(screen.x, screen.y),
                        withAlpha(ring.axis.color, alpha),
                        line_width);
                }
            }
        }

        void drawScreenRing(NativeOverlayDrawList& draw_list,
                            const glm::vec2& center,
                            const float radius,
                            const bool hovered,
                            const bool active) {
            constexpr OverlayColor VIEW_COLOR = overlayColor(245, 222, 141, 255);
            const float line_width = active ? 3.8f : (hovered ? 3.0f : 1.8f);
            draw_list.AddCircle(glm::vec2(center.x, center.y), radius, overlayColor(0, 0, 0, active ? 175 : 95),
                                RING_SEGMENTS, line_width + 3.0f);
            draw_list.AddCircle(glm::vec2(center.x, center.y), radius,
                                withAlpha(VIEW_COLOR, hovered || active ? 0.95f : 0.52f),
                                RING_SEGMENTS, line_width);
        }

        [[nodiscard]] float screenSignedAngle(const glm::vec2& center,
                                              const glm::vec2& previous,
                                              const glm::vec2& current) {
            const glm::vec2 a = safeNormalize(previous - center, glm::vec2(1.0f, 0.0f));
            const glm::vec2 b = safeNormalize(current - center, glm::vec2(1.0f, 0.0f));
            const float cross = a.x * b.y - a.y * b.x;
            const float dot = glm::dot(a, b);
            return std::atan2(cross, dot);
        }

        [[nodiscard]] bool unprojectRay(const RotationGizmoConfig& config,
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

        [[nodiscard]] bool planeDragVector(const RotationGizmoConfig& config,
                                           const glm::vec2& mouse,
                                           const glm::vec3& axis,
                                           glm::vec3& vector_world) {
            glm::vec3 ray_origin;
            glm::vec3 ray_dir;
            if (!unprojectRay(config, mouse, ray_origin, ray_dir)) {
                return false;
            }

            constexpr float MIN_RAY_PLANE_DOT = 0.15f;
            const float denom = glm::dot(ray_dir, axis);
            if (std::abs(denom) < MIN_RAY_PLANE_DOT) {
                return false;
            }

            const float t = glm::dot(config.pivot_world - ray_origin, axis) / denom;
            const glm::vec3 point = ray_origin + ray_dir * t;
            vector_world = point - config.pivot_world;
            return std::isfinite(vector_world.x) &&
                   std::isfinite(vector_world.y) &&
                   std::isfinite(vector_world.z) &&
                   lengthSquared(vector_world) > 0.000001f;
        }

        [[nodiscard]] float signedAngleOnAxis(const glm::vec3& from,
                                              const glm::vec3& to,
                                              const glm::vec3& axis) {
            const glm::vec3 from_norm = safeNormalize(from, glm::vec3(1.0f, 0.0f, 0.0f));
            const glm::vec3 to_norm = safeNormalize(to, glm::vec3(1.0f, 0.0f, 0.0f));
            return std::atan2(glm::dot(axis, glm::cross(from_norm, to_norm)),
                              glm::dot(from_norm, to_norm));
        }

        [[nodiscard]] bool projectRingAngle(const RotationGizmoConfig& config,
                                            const glm::mat4& view_projection,
                                            const glm::vec3& axis,
                                            const float radius_world,
                                            const float angle,
                                            glm::vec2& screen) {
            glm::vec3 u;
            glm::vec3 v;
            planeBasis(axis, u, v);
            const glm::vec3 radial = u * std::cos(angle) + v * std::sin(angle);
            return projectPoint(view_projection, config.viewport_pos, config.viewport_size,
                                config.pivot_world + radial * radius_world, screen);
        }

        [[nodiscard]] float closestRingAngle(const RotationGizmoConfig& config,
                                             const glm::mat4& view_projection,
                                             const glm::vec3& axis,
                                             const float radius_world,
                                             const glm::vec2& mouse) {
            float best_angle = 0.0f;
            float best_distance2 = std::numeric_limits<float>::max();
            constexpr int SEARCH_SEGMENTS = 192;
            glm::vec3 u;
            glm::vec3 v;
            planeBasis(axis, u, v);

            for (int i = 0; i < SEARCH_SEGMENTS; ++i) {
                const float angle = static_cast<float>(i) * glm::two_pi<float>() /
                                    static_cast<float>(SEARCH_SEGMENTS);
                const glm::vec3 radial = u * std::cos(angle) + v * std::sin(angle);
                glm::vec2 projected;
                if (!projectPoint(view_projection, config.viewport_pos, config.viewport_size,
                                  config.pivot_world + radial * radius_world, projected)) {
                    continue;
                }
                const float distance2 = lengthSquared(projected - mouse);
                if (distance2 < best_distance2) {
                    best_distance2 = distance2;
                    best_angle = angle;
                }
            }

            return best_angle;
        }

        [[nodiscard]] bool screenDragBasis(const RotationGizmoConfig& config,
                                           const glm::mat4& view_projection,
                                           const glm::vec3& axis,
                                           const float radius_world,
                                           const float radius_px,
                                           const glm::vec2& pivot_screen,
                                           const glm::vec2& mouse,
                                           glm::vec2& tangent,
                                           float& pixels_per_radian) {
            const float angle = closestRingAngle(config, view_projection, axis, radius_world, mouse);
            constexpr float STEP = 0.025f;
            glm::vec2 before;
            glm::vec2 after;
            if (projectRingAngle(config, view_projection, axis, radius_world, angle - STEP, before) &&
                projectRingAngle(config, view_projection, axis, radius_world, angle + STEP, after)) {
                const glm::vec2 derivative = after - before;
                const float derivative_length = glm::length(derivative);
                if (std::isfinite(derivative_length) && derivative_length > 0.25f) {
                    tangent = derivative / derivative_length;
                    pixels_per_radian = derivative_length / (STEP * 2.0f);
                    return true;
                }
            }

            const glm::vec2 radial = safeNormalize(mouse - pivot_screen, glm::vec2(1.0f, 0.0f));
            tangent = glm::vec2(-radial.y, radial.x);
            pixels_per_radian = std::max(16.0f, radius_px);
            return std::isfinite(tangent.x) && std::isfinite(tangent.y);
        }

        [[nodiscard]] float unwrapRawAngle(const float raw_angle) {
            // Same thresholding idea as Blender's dial gizmo: detect crossing the
            // +/-pi discontinuity without treating tiny sign changes near zero as turns.
            if ((raw_angle * g_active.previous_raw_angle < 0.0f) &&
                (std::abs(g_active.previous_raw_angle) > glm::half_pi<float>())) {
                if (g_active.previous_raw_angle < 0.0f) {
                    --g_active.rotations;
                } else {
                    ++g_active.rotations;
                }
            }
            g_active.previous_raw_angle = raw_angle;

            return raw_angle + (glm::two_pi<float>() * static_cast<float>(g_active.rotations));
        }

        [[nodiscard]] bool rotationAngle(const RotationGizmoConfig& config,
                                         const glm::vec2& pivot_screen,
                                         const glm::vec2& current,
                                         float& angle) {
            float raw_angle = 0.0f;
            if (g_active.axis == RotationGizmoAxis::View) {
                raw_angle = screenSignedAngle(pivot_screen, g_active.start_mouse, current);
            } else if (g_active.use_plane_drag) {
                glm::vec3 current_vector;
                if (!planeDragVector(config, current, g_active.axis_world, current_vector)) {
                    angle = g_active.previous_raw_angle +
                            (glm::two_pi<float>() * static_cast<float>(g_active.rotations));
                    return true;
                }
                raw_angle = signedAngleOnAxis(g_active.start_vector_world, current_vector, g_active.axis_world);
            } else {
                angle = glm::dot(current - g_active.start_mouse, g_active.screen_tangent) /
                            std::max(1.0f, g_active.screen_pixels_per_radian) +
                        g_active.angle_offset;
                return std::isfinite(angle);
            }

            angle = unwrapRawAngle(raw_angle) + g_active.angle_offset;
            return std::isfinite(angle);
        }

        [[nodiscard]] glm::mat3 rotationMatrix(const glm::vec3& axis, const float angle) {
            return glm::mat3(glm::rotate(glm::mat4(1.0f), angle, axis));
        }

        [[nodiscard]] float snapAngle(const float angle, const float snap_degrees) {
            const float snap_radians = glm::radians(std::max(0.1f, snap_degrees));
            return std::round(angle / snap_radians) * snap_radians;
        }

        [[nodiscard]] RotationGizmoAxis nearestAxis(const glm::vec2& mouse,
                                                    const glm::vec2& pivot_screen,
                                                    const float ring_radius_px,
                                                    const std::array<RingCache, 3>& rings) {
            constexpr float hit_threshold2 = HIT_THRESHOLD_PX * HIT_THRESHOLD_PX;
            float best_distance2 = std::numeric_limits<float>::max();
            RotationGizmoAxis best_axis = RotationGizmoAxis::None;
            for (const auto& ring : rings) {
                const float distance2 = distanceToPolylineSquared(mouse, ring);
                if (distance2 < best_distance2) {
                    best_distance2 = distance2;
                    best_axis = ring.axis.axis;
                }
            }

            const float view_ring_distance = std::abs(glm::length(mouse - pivot_screen) -
                                                      (ring_radius_px + VIEW_RING_OFFSET_PX));
            const float view_ring_distance2 = view_ring_distance * view_ring_distance;
            if (view_ring_distance2 < best_distance2) {
                best_distance2 = view_ring_distance2;
                best_axis = RotationGizmoAxis::View;
            }

            return best_distance2 <= hit_threshold2 ? best_axis : RotationGizmoAxis::None;
        }

        [[nodiscard]] float activeRingDistance(const glm::vec2& mouse,
                                               const glm::vec2& pivot_screen,
                                               const float ring_radius_px,
                                               const std::array<RingCache, 3>& rings) {
            if (g_active.axis == RotationGizmoAxis::View) {
                return std::abs(glm::length(mouse - pivot_screen) -
                                (ring_radius_px + VIEW_RING_OFFSET_PX));
            }

            const int axis_index = static_cast<int>(g_active.axis);
            if (axis_index < 0 || axis_index >= static_cast<int>(rings.size())) {
                return std::numeric_limits<float>::max();
            }
            return distanceToPolyline(mouse, rings[static_cast<size_t>(axis_index)]);
        }

        [[nodiscard]] float activeRingInfluence(const float distance_to_ring, const float ring_radius_px) {
            const float full_distance = std::max(MIN_FULL_INFLUENCE_DISTANCE_PX, ring_radius_px * 0.6f);
            const float zero_distance = std::max(MIN_ZERO_INFLUENCE_DISTANCE_PX, ring_radius_px * 1.75f);
            return 1.0f - smoothStep(full_distance, zero_distance, distance_to_ring);
        }

        [[nodiscard]] glm::vec3 axisDirectionFor(const RotationGizmoAxis axis,
                                                 const std::array<AxisVisual, 3>& axes,
                                                 const glm::vec3& view_axis) {
            switch (axis) {
            case RotationGizmoAxis::X: return axes[0].direction;
            case RotationGizmoAxis::Y: return axes[1].direction;
            case RotationGizmoAxis::Z: return axes[2].direction;
            case RotationGizmoAxis::View: return view_axis;
            case RotationGizmoAxis::None:
            default: return glm::vec3(0.0f);
            }
        }
    } // namespace

    RotationGizmoResult drawRotationGizmo(const RotationGizmoConfig& config) {
        RotationGizmoResult result;
        g_hovered = false;

        if (!config.draw_list || config.viewport_size.x <= 1.0f || config.viewport_size.y <= 1.0f) {
            return result;
        }

        RotationGizmoConfig drag_config = config;
        if (g_active.active && g_active.id == config.id) {
            drag_config.pivot_world = g_active.pivot_world;
            drag_config.orientation_world = g_active.orientation_world;
        }

        const glm::mat4 view_projection = drag_config.projection * drag_config.view;
        glm::vec2 pivot_screen;
        if (!projectPoint(view_projection, drag_config.viewport_pos, drag_config.viewport_size,
                          drag_config.pivot_world, pivot_screen)) {
            if (g_active.active && g_active.id == config.id && !config.input.mouse_left_down) {
                g_active = ActiveState{};
            }
            return result;
        }

        const float target_radius_px = clampFinite(
            std::min(drag_config.viewport_size.x, drag_config.viewport_size.y) * 0.145f,
            MIN_SCREEN_RADIUS,
            MAX_SCREEN_RADIUS);
        const float ring_radius_world = estimateWorldRadius(drag_config, view_projection, pivot_screen, target_radius_px);
        const float ring_radius_px = target_radius_px;

        const std::array<AxisVisual, 3> axes = {
            AxisVisual{RotationGizmoAxis::X, axisFromConfig(drag_config.orientation_world, 0),
                       overlayColor(245, 82, 96, 255)},
            AxisVisual{RotationGizmoAxis::Y, axisFromConfig(drag_config.orientation_world, 1),
                       overlayColor(74, 208, 119, 255)},
            AxisVisual{RotationGizmoAxis::Z, axisFromConfig(drag_config.orientation_world, 2),
                       overlayColor(80, 151, 255, 255)},
        };
        const glm::vec3 view_axis = cameraForward(drag_config.view);
        const std::array<RingCache, 3> rings = {
            buildRingCache(drag_config, view_projection, axes[0], ring_radius_world),
            buildRingCache(drag_config, view_projection, axes[1], ring_radius_world),
            buildRingCache(drag_config, view_projection, axes[2], ring_radius_world),
        };

        const glm::vec2 mouse = config.input.mouse_pos;
        const bool mouse_in_viewport = isInViewport(drag_config, mouse);
        RotationGizmoAxis hovered_axis = RotationGizmoAxis::None;
        if (mouse_in_viewport && (!g_active.active || g_active.id == config.id)) {
            hovered_axis = nearestAxis(mouse, pivot_screen, ring_radius_px, rings);
        }

        if (g_active.active && g_active.id == config.id) {
            result.active = config.input.mouse_left_down;
            if (!result.active) {
                g_active = ActiveState{};
            }
        }

        if (!g_active.active && config.input_enabled &&
            hovered_axis != RotationGizmoAxis::None &&
            config.input.mouse_left_clicked) {
            g_active.active = true;
            g_active.id = config.id;
            g_active.axis = hovered_axis;
            g_active.axis_world = axisDirectionFor(hovered_axis, axes, view_axis);
            g_active.pivot_world = drag_config.pivot_world;
            g_active.orientation_world = drag_config.orientation_world;
            g_active.start_mouse = mouse;
            g_active.use_plane_drag = hovered_axis != RotationGizmoAxis::View &&
                                      planeDragVector(drag_config, mouse, g_active.axis_world,
                                                      g_active.start_vector_world);
            if (hovered_axis != RotationGizmoAxis::View && !g_active.use_plane_drag) {
                const bool has_screen_basis = screenDragBasis(drag_config, view_projection, g_active.axis_world,
                                                              ring_radius_world, ring_radius_px, pivot_screen,
                                                              mouse, g_active.screen_tangent,
                                                              g_active.screen_pixels_per_radian);
                if (!has_screen_basis) {
                    g_active.screen_tangent = glm::vec2(0.0f, 1.0f);
                    g_active.screen_pixels_per_radian = std::max(16.0f, ring_radius_px);
                }
            }
            g_active.applied_angle = 0.0f;
            g_active.angle_offset = 0.0f;
            g_active.previous_raw_angle = 0.0f;
            g_active.rotations = 0;
            result.active = true;
        }

        if (g_active.active && g_active.id == config.id) {
            result.active = true;
            result.active_axis = g_active.axis;
            float absolute_angle = 0.0f;
            if (rotationAngle(drag_config, pivot_screen, mouse, absolute_angle)) {
                const float distance_to_ring = activeRingDistance(mouse, pivot_screen, ring_radius_px, rings);
                const float ring_influence = activeRingInfluence(distance_to_ring, ring_radius_px);
                const float requested_angle = config.snap
                                                  ? snapAngle(absolute_angle, config.snap_degrees)
                                                  : absolute_angle;
                if (ring_influence <= 0.0001f) {
                    g_active.angle_offset += g_active.applied_angle - absolute_angle;
                } else {
                    const float applied_delta = (requested_angle - g_active.applied_angle) * ring_influence;
                    if (std::abs(applied_delta) > ANGLE_EPSILON) {
                        result.changed = true;
                        result.delta_rotation = rotationMatrix(g_active.axis_world, applied_delta);
                        g_active.applied_angle += applied_delta;
                    }
                    if (ring_influence < 0.999f) {
                        g_active.angle_offset += g_active.applied_angle - requested_angle;
                    }
                }
            }
        }

        result.hovered_axis = hovered_axis;
        result.hovered = hovered_axis != RotationGizmoAxis::None;
        g_hovered = result.hovered || result.active;

        const RotationGizmoAxis emphasized_axis = result.active ? g_active.axis : hovered_axis;
        for (const auto& ring : rings) {
            const bool emphasized = emphasized_axis == ring.axis.axis;
            drawRing(*drag_config.draw_list, ring, view_axis,
                     emphasized && !result.active, emphasized && result.active);
        }

        drawScreenRing(*drag_config.draw_list, pivot_screen, ring_radius_px + VIEW_RING_OFFSET_PX,
                       emphasized_axis == RotationGizmoAxis::View && !result.active,
                       emphasized_axis == RotationGizmoAxis::View && result.active);

        drag_config.draw_list->AddCircleFilled(glm::vec2(pivot_screen.x, pivot_screen.y), 5.0f,
                                               overlayColor(0, 0, 0, 155), 24);
        drag_config.draw_list->AddCircleFilled(glm::vec2(pivot_screen.x, pivot_screen.y), 3.5f,
                                               overlayColor(250, 250, 255, 230), 24);

        return result;
    }

    bool isRotationGizmoHovered() {
        return g_hovered;
    }

    bool isRotationGizmoActive() {
        return g_active.active;
    }

} // namespace lfs::vis::gui
