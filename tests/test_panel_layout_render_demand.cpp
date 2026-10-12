/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include <algorithm>
#include <array>
#include <gtest/gtest.h>
#include <visualizer/gui/viewport_gizmo_geometry.hpp>
namespace {
    std::array<lfs::vis::gui::ViewportGizmoMarker, 6> homeGizmoMarkers(const float scale = 1.0f) {
        using lfs::vis::gui::ViewportGizmoMarker;
        std::array<ViewportGizmoMarker, 6> markers{{
            {0, {1164.0f, 81.0f}, 8.2f, true},
            {1, {1182.0f, 60.0f}, 8.2f, true},
            {2, {1201.0f, 81.0f}, 8.2f, true},
            {3, {1208.0f, 96.0f}, 8.2f, true},
            {4, {1182.0f, 115.0f}, 8.2f, true},
            {5, {1156.0f, 96.0f}, 8.2f, true},
        }};
        for (auto& marker : markers) {
            marker.screen_pos *= scale;
            marker.radius *= scale;
        }
        return markers;
    }

    TEST(ViewportGizmoGeometryTest, EveryCircleCenterSelectsItsAxisAtEveryUiScale) {
        using namespace lfs::vis::gui;
        for (const float scale : {1.0f, 1.5f, 2.0f}) {
            auto markers = homeGizmoMarkers(scale);
            for (const auto& marker : markers) {
                EXPECT_EQ(hitTestViewportGizmoMarkers(markers, marker.screen_pos), marker.encoded_axis);
                EXPECT_EQ(hitTestViewportGizmoMarkers(markers, marker.screen_pos + glm::vec2(scale)),
                          marker.encoded_axis);
            }
            std::reverse(markers.begin(), markers.end());
            for (const auto& marker : markers) {
                EXPECT_EQ(hitTestViewportGizmoMarkers(markers, marker.screen_pos), marker.encoded_axis);
            }
        }
    }

    TEST(ViewportGizmoGeometryTest, OverlappingPositiveCirclesSelectTheNearestCenter) {
        using namespace lfs::vis::gui;
        std::array<ViewportGizmoMarker, 6> markers{};
        markers[0] = {0, {10.0f, 10.0f}, 8.0f, true};
        markers[1] = {1, {25.0f, 10.0f}, 8.0f, true};
        EXPECT_EQ(hitTestViewportGizmoMarkers(markers, {24.0f, 10.0f}), 1);
        EXPECT_EQ(hitTestViewportGizmoMarkers(markers, {11.0f, 10.0f}), 0);
    }

    TEST(ViewportGizmoGeometryTest, ExistingUnambiguousHitsAndMissesMatchFirstHitReference) {
        using namespace lfs::vis::gui;
        const auto markers = homeGizmoMarkers();
        int checked = 0;
        for (int y = 30; y <= 145; ++y) {
            for (int x = 1120; x <= 1240; ++x) {
                const glm::vec2 mouse(x, y);
                int first = -1;
                int hits = 0;
                for (const auto& marker : markers) {
                    const auto delta = mouse - marker.screen_pos;
                    const float radius = marker.radius * 2.5f;
                    if (marker.visible && glm::dot(delta, delta) <= radius * radius) {
                        if (first == -1)
                            first = marker.encoded_axis;
                        ++hits;
                    }
                }
                if (hits <= 1) {
                    EXPECT_EQ(hitTestViewportGizmoMarkers(markers, mouse), first);
                    ++checked;
                }
            }
        }
        EXPECT_GT(checked, 10000);
    }

    TEST(ViewportGizmoGeometryTest, HitToleranceIncludesBoundaryAndIgnoresHiddenMarkers) {
        using namespace lfs::vis::gui;
        std::array<ViewportGizmoMarker, 6> markers{};
        EXPECT_EQ(hitTestViewportGizmoMarkers(markers, {0.0f, 0.0f}), -1);
        markers[0] = {0, {0.0f, 0.0f}, 8.0f, true};
        markers[1] = {1, {20.0f, 0.0f}, 8.0f, false};
        EXPECT_EQ(hitTestViewportGizmoMarkers(markers, {20.0f, 0.0f}), 0);
        EXPECT_EQ(hitTestViewportGizmoMarkers(markers, {20.01f, 0.0f}), -1);
        markers[1].visible = true;
        EXPECT_EQ(hitTestViewportGizmoMarkers(markers, {20.01f, 0.0f}), 1);
    }

    TEST(ViewportGizmoGeometryTest, EqualDistanceAndCoincidentMarkersKeepFirstHitOrder) {
        using namespace lfs::vis::gui;
        std::array<ViewportGizmoMarker, 6> markers{};
        markers[0] = {2, {0.0f, 0.0f}, 8.0f, true};
        markers[1] = {5, {20.0f, 0.0f}, 8.0f, true};
        EXPECT_EQ(hitTestViewportGizmoMarkers(markers, {10.0f, 0.0f}), 2);
        markers[1].screen_pos = markers[0].screen_pos;
        EXPECT_EQ(hitTestViewportGizmoMarkers(markers, {0.0f, 0.0f}), 2);
    }

} // namespace
