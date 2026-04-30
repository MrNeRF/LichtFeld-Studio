/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/bounds_gizmo.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace lfs::vis::gui {
    namespace {
        constexpr float BIG_ANCHOR_RADIUS_PX = 8.0f;
        constexpr float SMALL_ANCHOR_RADIUS_PX = 6.0f;
        constexpr float FACE_HIT_RADIUS_PX = 13.0f;
        constexpr float CORNER_HIT_RADIUS_PX = 11.0f;
        constexpr float DASH_PERIOD_PX = 12.0f;
        constexpr float DASH_DUTY = 0.55f;
        constexpr int MAX_DASHES_PER_EDGE = 160;
        constexpr float DELTA_EPSILON = 0.000001f;
        constexpr OverlayColor BOUNDS_LINE_COLOR = overlayColor(0xAA, 0xAA, 0xAA, 255);
        constexpr OverlayColor BOUNDS_LINE_SHADOW = overlayColor(0, 0, 0, 130);
        constexpr OverlayColor ANCHOR_COLOR = overlayColor(0xAA, 0xAA, 0xAA, 255);
        constexpr OverlayColor SELECTION_COLOR = overlayColor(255, 128, 16, 210);

        struct AxisVisual {
            glm::vec3 direction{0.0f};
        };

        struct HandleHit {
            BoundsGizmoHandle handle = BoundsGizmoHandle::None;
            int axis = -1;
            float sign = 1.0f;
            int plane_axis = -1;
            glm::ivec3 corner_signs{1};
        };

        struct ActiveState {
            bool active = false;
            int id = 0;
            BoundsGizmoHandle handle = BoundsGizmoHandle::None;
            int axis = -1;
            float sign = 1.0f;
            int plane_axis = -1;
            glm::ivec3 corner_signs{1};
            glm::vec3 start_center_world{0.0f};
            glm::vec3 start_half_extents_world{0.5f};
            glm::vec3 min_half_extents_world{0.0005f};
            glm::mat3 orientation_world{1.0f};
            glm::vec2 start_mouse{0.0f};
            glm::vec2 screen_axis{1.0f, 0.0f};
            float pixels_per_world = 64.0f;
            glm::vec3 applied_center_world{0.0f};
            glm::vec3 applied_half_extents_world{0.5f};
        };

        ActiveState g_active;
        bool g_hovered = false;

        struct PlaneRect {
            int normal_axis = 0;
            int axis_a = 1;
            int axis_b = 2;
        };

        struct ProjectedRect {
            PlaneRect rect;
            std::array<glm::ivec3, 4> signs{};
            std::array<glm::vec2, 4> screen{};
            bool valid = false;
        };

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

        [[nodiscard]] std::array<AxisVisual, 3> axesFromConfig(const glm::mat3& orientation) {
            return {
                AxisVisual{axisFromConfig(orientation, 0)},
                AxisVisual{axisFromConfig(orientation, 1)},
                AxisVisual{axisFromConfig(orientation, 2)},
            };
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

        [[nodiscard]] bool isInViewport(const BoundsGizmoConfig& config, const glm::vec2& p) {
            return p.x >= config.viewport_pos.x &&
                   p.x <= config.viewport_pos.x + config.viewport_size.x &&
                   p.y >= config.viewport_pos.y &&
                   p.y <= config.viewport_pos.y + config.viewport_size.y;
        }

        [[nodiscard]] glm::vec3 cameraPosition(const glm::mat4& view) {
            const glm::vec3 right(view[0][0], view[1][0], view[2][0]);
            const glm::vec3 up(view[0][1], view[1][1], view[2][1]);
            const glm::vec3 back(view[0][2], view[1][2], view[2][2]);
            const glm::vec3 translation(view[3]);
            return -(right * translation.x + up * translation.y + back * translation.z);
        }

        [[nodiscard]] glm::vec3 cornerVector(const std::array<AxisVisual, 3>& axes,
                                             const glm::vec3& half_extents,
                                             const glm::ivec3& signs) {
            return axes[0].direction * (static_cast<float>(signs.x) * half_extents.x) +
                   axes[1].direction * (static_cast<float>(signs.y) * half_extents.y) +
                   axes[2].direction * (static_cast<float>(signs.z) * half_extents.z);
        }

        [[nodiscard]] std::array<PlaneRect, 3> visiblePlaneRects(const BoundsGizmoConfig& config,
                                                                 const std::array<AxisVisual, 3>& axes,
                                                                 int& count) {
            const glm::vec3 view_direction =
                safeNormalize(cameraPosition(config.view) - config.center_world, glm::vec3(0.0f, 0.0f, 1.0f));

            std::array<float, 3> alignment{};
            int best_axis = 0;
            float best_dot = -1.0f;
            for (int axis = 0; axis < 3; ++axis) {
                alignment[static_cast<size_t>(axis)] = std::abs(glm::dot(view_direction, axes[axis].direction));
                if (alignment[static_cast<size_t>(axis)] > best_dot) {
                    best_dot = alignment[static_cast<size_t>(axis)];
                    best_axis = axis;
                }
            }

            std::array<PlaneRect, 3> rects{};
            count = 0;
            if (alignment[static_cast<size_t>(best_axis)] >= 0.1f) {
                rects[static_cast<size_t>(count++)] = {best_axis, (best_axis + 1) % 3, (best_axis + 2) % 3};
            }
            for (int axis = 0; axis < 3; ++axis) {
                if (axis == best_axis) {
                    continue;
                }
                if (alignment[static_cast<size_t>(axis)] >= 0.1f) {
                    rects[static_cast<size_t>(count++)] = {axis, (axis + 1) % 3, (axis + 2) % 3};
                }
            }
            if (count == 0) {
                rects[0] = {best_axis, (best_axis + 1) % 3, (best_axis + 2) % 3};
                count = 1;
            }
            return rects;
        }

        [[nodiscard]] std::array<glm::ivec3, 4> rectCornerSigns(const PlaneRect& rect) {
            std::array<glm::ivec3, 4> signs = {
                glm::ivec3(0),
                glm::ivec3(0),
                glm::ivec3(0),
                glm::ivec3(0),
            };
            signs[0][rect.axis_a] = -1;
            signs[0][rect.axis_b] = -1;
            signs[1][rect.axis_a] = 1;
            signs[1][rect.axis_b] = -1;
            signs[2][rect.axis_a] = 1;
            signs[2][rect.axis_b] = 1;
            signs[3][rect.axis_a] = -1;
            signs[3][rect.axis_b] = 1;
            return signs;
        }

        [[nodiscard]] glm::vec3 rectAnchorWorld(const BoundsGizmoConfig& config,
                                                const std::array<AxisVisual, 3>& axes,
                                                const glm::ivec3& signs) {
            return config.center_world + cornerVector(axes, config.half_extents_world, signs);
        }

        [[nodiscard]] bool projectRect(const BoundsGizmoConfig& config,
                                       const glm::mat4& view_projection,
                                       const std::array<AxisVisual, 3>& axes,
                                       const std::array<glm::ivec3, 4>& signs,
                                       std::array<glm::vec2, 4>& screen) {
            for (size_t i = 0; i < signs.size(); ++i) {
                if (!projectPoint(view_projection, config.viewport_pos, config.viewport_size,
                                  rectAnchorWorld(config, axes, signs[i]), screen[i])) {
                    return false;
                }
            }
            return true;
        }

        [[nodiscard]] std::array<ProjectedRect, 3> projectRects(
            const BoundsGizmoConfig& config,
            const glm::mat4& view_projection,
            const std::array<AxisVisual, 3>& axes,
            const std::array<PlaneRect, 3>& rects,
            const int rect_count) {
            std::array<ProjectedRect, 3> projected{};
            for (int rect_index = 0; rect_index < rect_count; ++rect_index) {
                auto& dst = projected[static_cast<size_t>(rect_index)];
                dst.rect = rects[static_cast<size_t>(rect_index)];
                dst.signs = rectCornerSigns(dst.rect);
                dst.valid = projectRect(config, view_projection, axes, dst.signs, dst.screen);
            }
            return projected;
        }

        void drawDashedLine(NativeOverlayDrawList& draw_list, const glm::vec2& a, const glm::vec2& b) {
            const float distance = glm::length(b - a);
            const int step_count = std::clamp(static_cast<int>(distance / DASH_PERIOD_PX), 1, MAX_DASHES_PER_EDGE);
            for (int step = 0; step < step_count; ++step) {
                const float t0 = static_cast<float>(step) / static_cast<float>(step_count);
                const float t1 = t0 + DASH_DUTY / static_cast<float>(step_count);
                const glm::vec2 p0 = glm::mix(a, b, t0);
                const glm::vec2 p1 = glm::mix(a, b, t1);
                draw_list.AddLine(glm::vec2(p0.x, p0.y), glm::vec2(p1.x, p1.y), BOUNDS_LINE_SHADOW, 3.4f);
                draw_list.AddLine(glm::vec2(p0.x, p0.y), glm::vec2(p1.x, p1.y), BOUNDS_LINE_COLOR, 2.0f);
            }
        }

        void drawBoundsRects(NativeOverlayDrawList& draw_list,
                             const std::array<ProjectedRect, 3>& rects,
                             const int rect_count) {
            for (int rect_index = 0; rect_index < rect_count; ++rect_index) {
                const auto& rect = rects[static_cast<size_t>(rect_index)];
                if (!rect.valid) {
                    continue;
                }
                for (int i = 0; i < 4; ++i) {
                    drawDashedLine(draw_list, rect.screen[static_cast<size_t>(i)],
                                   rect.screen[static_cast<size_t>((i + 1) % 4)]);
                }
            }
        }

        [[nodiscard]] BoundsGizmoHandle faceHandle(const int axis, const float sign) {
            if (axis == 0) {
                return sign < 0.0f ? BoundsGizmoHandle::FaceXNegative : BoundsGizmoHandle::FaceXPositive;
            }
            if (axis == 1) {
                return sign < 0.0f ? BoundsGizmoHandle::FaceYNegative : BoundsGizmoHandle::FaceYPositive;
            }
            return sign < 0.0f ? BoundsGizmoHandle::FaceZNegative : BoundsGizmoHandle::FaceZPositive;
        }

        [[nodiscard]] bool sameSigns(const glm::ivec3& a, const glm::ivec3& b) {
            return a.x == b.x && a.y == b.y && a.z == b.z;
        }

        [[nodiscard]] int faceHandleIndex(const BoundsGizmoHandle handle) {
            const int index = static_cast<int>(handle);
            return index >= static_cast<int>(BoundsGizmoHandle::FaceXNegative) &&
                           index <= static_cast<int>(BoundsGizmoHandle::FaceZPositive)
                       ? index
                       : -1;
        }

        void drawAnchor(NativeOverlayDrawList& draw_list,
                        const glm::vec2& screen,
                        const float radius,
                        const bool hot) {
            draw_list.AddCircleFilled(glm::vec2(screen.x, screen.y), radius, OVERLAY_COL32_BLACK, 24);
            draw_list.AddCircleFilled(glm::vec2(screen.x, screen.y), radius - 1.2f, hot ? SELECTION_COLOR : ANCHOR_COLOR, 24);
        }

        [[nodiscard]] HandleHit edgeHandle(const glm::ivec3& a, const glm::ivec3& b, const int plane_axis) {
            for (int axis = 0; axis < 3; ++axis) {
                if (a[axis] == b[axis] && a[axis] != 0) {
                    return {
                        faceHandle(axis, static_cast<float>(a[axis])),
                        axis,
                        static_cast<float>(a[axis]),
                        plane_axis,
                        glm::ivec3(1),
                    };
                }
            }
            return {};
        }

        void drawBoundsAnchors(NativeOverlayDrawList& draw_list,
                               const std::array<ProjectedRect, 3>& rects,
                               const int rect_count,
                               const HandleHit& highlighted) {
            std::array<bool, 6> drawn_face_handles{};
            for (int rect_index = 0; rect_index < rect_count; ++rect_index) {
                const auto& rect = rects[static_cast<size_t>(rect_index)];
                if (!rect.valid) {
                    continue;
                }

                for (int i = 0; i < 4; ++i) {
                    const bool hot = highlighted.handle == BoundsGizmoHandle::Corner &&
                                     sameSigns(highlighted.corner_signs, rect.signs[static_cast<size_t>(i)]);
                    drawAnchor(draw_list, rect.screen[static_cast<size_t>(i)], BIG_ANCHOR_RADIUS_PX, hot);
                }

                for (int i = 0; i < 4; ++i) {
                    const int next = (i + 1) % 4;
                    const glm::vec2 mid_screen =
                        (rect.screen[static_cast<size_t>(i)] + rect.screen[static_cast<size_t>(next)]) * 0.5f;
                    const HandleHit edge = edgeHandle(rect.signs[static_cast<size_t>(i)],
                                                      rect.signs[static_cast<size_t>(next)],
                                                      rect.rect.normal_axis);
                    const int handle_index = faceHandleIndex(edge.handle);
                    if (handle_index < 0 || drawn_face_handles[static_cast<size_t>(handle_index)]) {
                        continue;
                    }
                    drawn_face_handles[static_cast<size_t>(handle_index)] = true;
                    const bool hot = highlighted.handle == edge.handle && highlighted.handle != BoundsGizmoHandle::None;
                    drawAnchor(draw_list, mid_screen, SMALL_ANCHOR_RADIUS_PX, hot);
                }
            }
        }

        [[nodiscard]] HandleHit nearestHandle(const std::array<ProjectedRect, 3>& rects,
                                              const int rect_count,
                                              const glm::vec2& mouse) {
            constexpr float face_hit_radius2 = FACE_HIT_RADIUS_PX * FACE_HIT_RADIUS_PX;
            constexpr float corner_hit_radius2 = CORNER_HIT_RADIUS_PX * CORNER_HIT_RADIUS_PX;
            float best_distance2 = std::numeric_limits<float>::max();
            HandleHit best;
            std::array<bool, 6> tested_face_handles{};

            for (int rect_index = 0; rect_index < rect_count; ++rect_index) {
                const auto& rect = rects[static_cast<size_t>(rect_index)];
                if (!rect.valid) {
                    continue;
                }

                for (int i = 0; i < 4; ++i) {
                    const int next = (i + 1) % 4;
                    const glm::vec2 mid_screen =
                        (rect.screen[static_cast<size_t>(i)] + rect.screen[static_cast<size_t>(next)]) * 0.5f;
                    const float distance2 = lengthSquared(mouse - mid_screen);
                    if (distance2 < best_distance2 && distance2 <= face_hit_radius2) {
                        const HandleHit edge = edgeHandle(rect.signs[static_cast<size_t>(i)],
                                                          rect.signs[static_cast<size_t>(next)],
                                                          rect.rect.normal_axis);
                        const int handle_index = faceHandleIndex(edge.handle);
                        if (handle_index < 0 || tested_face_handles[static_cast<size_t>(handle_index)]) {
                            continue;
                        }
                        tested_face_handles[static_cast<size_t>(handle_index)] = true;
                        best_distance2 = distance2;
                        best = edge;
                    }
                }

                for (int i = 0; i < 4; ++i) {
                    const float distance2 = lengthSquared(mouse - rect.screen[static_cast<size_t>(i)]);
                    if (distance2 < best_distance2 && distance2 <= corner_hit_radius2) {
                        best_distance2 = distance2;
                        best = {BoundsGizmoHandle::Corner,
                                -1,
                                1.0f,
                                rect.rect.normal_axis,
                                rect.signs[static_cast<size_t>(i)]};
                    }
                }
            }

            return best;
        }

        [[nodiscard]] float snapScale(const float value, const float snap_ratio) {
            const float step = std::max(0.01f, snap_ratio);
            return std::max(step, std::round(value / step) * step);
        }

        [[nodiscard]] BoundsGizmoResult transformForMouse(const glm::vec2& mouse, const BoundsGizmoConfig& config) {
            BoundsGizmoResult result;
            result.active = true;
            result.active_handle = g_active.handle;
            result.center_world = g_active.applied_center_world;
            result.half_extents_world = g_active.applied_half_extents_world;

            const float distance_world =
                glm::dot(mouse - g_active.start_mouse, g_active.screen_axis) /
                std::max(1.0f, g_active.pixels_per_world);

            glm::vec3 next_center = g_active.start_center_world;
            glm::vec3 next_half_extents = g_active.start_half_extents_world;
            const std::array<AxisVisual, 3> axes = axesFromConfig(g_active.orientation_world);

            if (g_active.handle == BoundsGizmoHandle::Corner) {
                const glm::vec3 start_corner =
                    cornerVector(axes, g_active.start_half_extents_world, g_active.corner_signs);
                const float corner_length = std::max(0.0001f, glm::length(start_corner));
                float factor = 1.0f + distance_world / (2.0f * corner_length);
                float min_factor = 0.0f;
                for (int axis = 0; axis < 3; ++axis) {
                    if (g_active.corner_signs[axis] != 0) {
                        min_factor = std::max(
                            min_factor,
                            g_active.min_half_extents_world[axis] /
                                std::max(0.000001f, g_active.start_half_extents_world[axis]));
                    }
                }
                factor = std::max(factor, min_factor);
                if (config.snap) {
                    factor = snapScale(factor, config.snap_ratio);
                }
                for (int axis = 0; axis < 3; ++axis) {
                    if (g_active.corner_signs[axis] != 0) {
                        next_half_extents[axis] =
                            std::max(g_active.min_half_extents_world[axis],
                                     g_active.start_half_extents_world[axis] * factor);
                    }
                }
                next_center = g_active.start_center_world + start_corner * (factor - 1.0f);
            } else if (g_active.axis >= 0) {
                const int axis = g_active.axis;
                const glm::vec3 signed_axis = axes[static_cast<size_t>(axis)].direction * g_active.sign;
                const float start_half = g_active.start_half_extents_world[axis];
                float next_half = std::max(g_active.min_half_extents_world[axis],
                                           start_half + distance_world * 0.5f);
                if (config.snap) {
                    next_half = start_half * snapScale(next_half / std::max(0.000001f, start_half),
                                                       config.snap_ratio);
                    next_half = std::max(g_active.min_half_extents_world[axis], next_half);
                }
                next_half_extents[axis] = next_half;
                next_center = g_active.start_center_world + signed_axis * (next_half - start_half);
            }

            if (lengthSquared(next_center - g_active.applied_center_world) > DELTA_EPSILON * DELTA_EPSILON ||
                lengthSquared(next_half_extents - g_active.applied_half_extents_world) >
                    DELTA_EPSILON * DELTA_EPSILON) {
                result.changed = true;
                g_active.applied_center_world = next_center;
                g_active.applied_half_extents_world = next_half_extents;
            }

            result.center_world = g_active.applied_center_world;
            result.half_extents_world = g_active.applied_half_extents_world;
            return result;
        }

        void beginDrag(const BoundsGizmoConfig& config,
                       const glm::mat4& view_projection,
                       const std::array<AxisVisual, 3>& axes,
                       const HandleHit& hit,
                       const glm::vec2& mouse,
                       const glm::vec2& center_screen) {
            g_active = ActiveState{};
            g_active.active = true;
            g_active.id = config.id;
            g_active.handle = hit.handle;
            g_active.axis = hit.axis;
            g_active.sign = hit.sign;
            g_active.plane_axis = hit.plane_axis;
            g_active.corner_signs = hit.corner_signs;
            g_active.start_center_world = config.center_world;
            g_active.start_half_extents_world = glm::max(config.half_extents_world, config.min_half_extents_world);
            g_active.min_half_extents_world = glm::max(config.min_half_extents_world, glm::vec3(0.000001f));
            g_active.orientation_world = config.orientation_world;
            g_active.start_mouse = mouse;
            g_active.applied_center_world = g_active.start_center_world;
            g_active.applied_half_extents_world = g_active.start_half_extents_world;

            glm::vec3 direction_world{1.0f, 0.0f, 0.0f};
            float world_length = 1.0f;
            if (hit.handle == BoundsGizmoHandle::Corner) {
                direction_world = cornerVector(axes, g_active.start_half_extents_world, hit.corner_signs);
                world_length = std::max(0.0001f, glm::length(direction_world));
                direction_world /= world_length;
            } else if (hit.axis >= 0) {
                direction_world = axes[static_cast<size_t>(hit.axis)].direction * hit.sign;
                world_length = std::max(0.0001f, g_active.start_half_extents_world[hit.axis]);
            }

            glm::vec2 handle_screen;
            const glm::vec3 probe_world = config.center_world + direction_world * world_length;
            if (projectPoint(view_projection, config.viewport_pos, config.viewport_size, probe_world, handle_screen)) {
                const glm::vec2 projected_axis = handle_screen - center_screen;
                const float projected_length = glm::length(projected_axis);
                g_active.screen_axis = safeNormalize(projected_axis, glm::vec2(1.0f, 0.0f));
                if (projected_length > 1.0f) {
                    g_active.pixels_per_world = projected_length / world_length;
                }
            }
        }
    } // namespace

    BoundsGizmoResult drawBoundsGizmo(const BoundsGizmoConfig& config) {
        BoundsGizmoResult result;
        g_hovered = false;

        if (!config.draw_list || config.viewport_size.x <= 1.0f || config.viewport_size.y <= 1.0f) {
            return result;
        }

        BoundsGizmoConfig draw_config = config;
        draw_config.half_extents_world = glm::max(draw_config.half_extents_world,
                                                  glm::max(draw_config.min_half_extents_world, glm::vec3(0.000001f)));
        if (g_active.active && g_active.id == config.id) {
            draw_config.center_world = g_active.applied_center_world;
            draw_config.half_extents_world = g_active.applied_half_extents_world;
            draw_config.orientation_world = g_active.orientation_world;
        }

        const glm::mat4 view_projection = draw_config.projection * draw_config.view;
        glm::vec2 center_screen;
        if (!projectPoint(view_projection, draw_config.viewport_pos, draw_config.viewport_size,
                          draw_config.center_world, center_screen)) {
            if (g_active.active && g_active.id == config.id && !config.input.mouse_left_down) {
                g_active = ActiveState{};
            }
            return result;
        }

        const std::array<AxisVisual, 3> axes = axesFromConfig(draw_config.orientation_world);
        int rect_count = 0;
        auto rects = visiblePlaneRects(draw_config, axes, rect_count);
        if (g_active.active && g_active.id == config.id && g_active.plane_axis >= 0) {
            const int axis = g_active.plane_axis;
            rects[0] = {axis, (axis + 1) % 3, (axis + 2) % 3};
            rect_count = 1;
        }
        const auto projected_rects = projectRects(draw_config, view_projection, axes, rects, rect_count);

        const glm::vec2 mouse = config.input.mouse_pos;
        HandleHit hovered_hit;
        if (isInViewport(draw_config, mouse) && (!g_active.active || g_active.id == config.id)) {
            hovered_hit = nearestHandle(projected_rects, rect_count, mouse);
        }

        if (g_active.active && g_active.id == config.id) {
            result.active = config.input.mouse_left_down;
            if (!result.active) {
                g_active = ActiveState{};
            }
        }

        if (!g_active.active && config.input_enabled &&
            hovered_hit.handle != BoundsGizmoHandle::None &&
            config.input.mouse_left_clicked) {
            beginDrag(draw_config, view_projection, axes, hovered_hit, mouse, center_screen);
            result.active = true;
        }

        if (g_active.active && g_active.id == config.id) {
            result = transformForMouse(mouse, config);
        }

        result.hovered_handle = hovered_hit.handle;
        result.hovered = hovered_hit.handle != BoundsGizmoHandle::None;
        g_hovered = result.hovered || result.active;

        const HandleHit highlighted = result.active
                                          ? HandleHit{g_active.handle,
                                                      g_active.axis,
                                                      g_active.sign,
                                                      g_active.plane_axis,
                                                      g_active.corner_signs}
                                          : hovered_hit;
        draw_config.draw_list->PushClipRect(
            glm::vec2(draw_config.viewport_pos.x, draw_config.viewport_pos.y),
            glm::vec2(draw_config.viewport_pos.x + draw_config.viewport_size.x,
                   draw_config.viewport_pos.y + draw_config.viewport_size.y),
            true);
        drawBoundsRects(*draw_config.draw_list, projected_rects, rect_count);
        drawBoundsAnchors(*draw_config.draw_list, projected_rects, rect_count, highlighted);
        draw_config.draw_list->PopClipRect();

        return result;
    }

    bool isBoundsGizmoHovered() {
        return g_hovered;
    }

    bool isBoundsGizmoActive() {
        return g_active.active;
    }

} // namespace lfs::vis::gui
