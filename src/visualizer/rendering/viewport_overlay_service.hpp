/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "render_pass.hpp"
#include <utility>
#include <vector>

namespace lfs::vis {

    class ViewportOverlayService {
    public:
        void setCursorPreview(bool active, float x, float y, float radius, bool add_mode,
                              lfs::core::Tensor* selection_tensor,
                              bool saturation_mode, float saturation_amount,
                              std::optional<SplitViewPanelId> panel,
                              int focused_gaussian_id) {
            cursor_preview_.active = active;
            cursor_preview_.x = x;
            cursor_preview_.y = y;
            cursor_preview_.radius = radius;
            cursor_preview_.add_mode = add_mode;
            cursor_preview_.selection_tensor = selection_tensor;
            cursor_preview_.saturation_mode = saturation_mode;
            cursor_preview_.saturation_amount = saturation_amount;
            cursor_preview_.panel = panel;
            cursor_preview_.focused_gaussian_id = focused_gaussian_id;
        }

        void clearCursorPreview() {
            cursor_preview_.active = false;
            cursor_preview_.x = 0.0f;
            cursor_preview_.y = 0.0f;
            cursor_preview_.radius = 0.0f;
            cursor_preview_.selection_tensor = nullptr;
            cursor_preview_.preview_selection = nullptr;
            cursor_preview_.saturation_mode = false;
            cursor_preview_.saturation_amount = 0.0f;
            cursor_preview_.panel.reset();
            cursor_preview_.focused_gaussian_id = -1;
            hovered_gaussian_id_ = -1;
        }

        [[nodiscard]] bool isCursorPreviewActive() const { return cursor_preview_.active; }

        void setPreviewSelection(lfs::core::Tensor* preview, bool add_mode) {
            cursor_preview_.preview_selection = preview;
            cursor_preview_.add_mode = add_mode;
        }

        void clearPreviewSelection() {
            cursor_preview_.preview_selection = nullptr;
        }

        void clearSelectionPreviews() {
            clearPreviewSelection();
            clearCursorPreview();
            clearRect();
            clearPolygon();
            clearLasso();
        }

        void setSelectionPreviewMode(SelectionPreviewMode mode) {
            cursor_preview_.selection_mode = mode;
        }

        [[nodiscard]] SelectionPreviewMode selectionPreviewMode() const {
            return cursor_preview_.selection_mode;
        }

        [[nodiscard]] const CursorPreviewState& cursorPreview() const { return cursor_preview_; }

        void setRect(float x0, float y0, float x1, float y1, bool add_mode,
                     std::optional<SplitViewPanelId> panel) {
            rect_.active = true;
            rect_.x0 = x0;
            rect_.y0 = y0;
            rect_.x1 = x1;
            rect_.y1 = y1;
            rect_.add_mode = add_mode;
            rect_.panel = panel;
        }

        void clearRect() {
            rect_.active = false;
            rect_.panel.reset();
        }
        [[nodiscard]] bool isRectPreviewActive() const { return rect_.active; }
        [[nodiscard]] float rectX0() const { return rect_.x0; }
        [[nodiscard]] float rectY0() const { return rect_.y0; }
        [[nodiscard]] float rectX1() const { return rect_.x1; }
        [[nodiscard]] float rectY1() const { return rect_.y1; }
        [[nodiscard]] bool rectAddMode() const { return rect_.add_mode; }
        [[nodiscard]] std::optional<SplitViewPanelId> rectPanel() const { return rect_.panel; }

        void setPolygon(const std::vector<std::pair<float, float>>& points, bool closed, bool add_mode,
                        std::optional<SplitViewPanelId> panel) {
            polygon_.active = true;
            polygon_.points = points;
            polygon_.world_points.clear();
            polygon_.closed = closed;
            polygon_.add_mode = add_mode;
            polygon_.world_space = false;
            polygon_.panel = panel;
        }

        void setPolygonWorldSpace(const std::vector<glm::vec3>& world_points, bool closed, bool add_mode,
                                  std::optional<SplitViewPanelId> panel) {
            polygon_.active = true;
            polygon_.points.clear();
            polygon_.world_points = world_points;
            polygon_.closed = closed;
            polygon_.add_mode = add_mode;
            polygon_.world_space = true;
            polygon_.panel = panel;
        }

        void clearPolygon() {
            polygon_.active = false;
            polygon_.points.clear();
            polygon_.world_points.clear();
            polygon_.closed = false;
            polygon_.world_space = false;
            polygon_.panel.reset();
        }

        [[nodiscard]] bool isPolygonPreviewActive() const { return polygon_.active; }
        [[nodiscard]] const std::vector<std::pair<float, float>>& polygonPoints() const { return polygon_.points; }
        [[nodiscard]] const std::vector<glm::vec3>& polygonWorldPoints() const { return polygon_.world_points; }
        [[nodiscard]] bool polygonClosed() const { return polygon_.closed; }
        [[nodiscard]] bool polygonAddMode() const { return polygon_.add_mode; }
        [[nodiscard]] bool polygonWorldSpace() const { return polygon_.world_space; }
        [[nodiscard]] std::optional<SplitViewPanelId> polygonPanel() const { return polygon_.panel; }

        void setLasso(const std::vector<std::pair<float, float>>& points, bool add_mode,
                      std::optional<SplitViewPanelId> panel) {
            lasso_.active = true;
            lasso_.points = points;
            lasso_.add_mode = add_mode;
            lasso_.panel = panel;
        }

        void clearLasso() {
            lasso_.active = false;
            lasso_.points.clear();
            lasso_.panel.reset();
        }

        [[nodiscard]] bool isLassoPreviewActive() const { return lasso_.active; }
        [[nodiscard]] const std::vector<std::pair<float, float>>& lassoPoints() const { return lasso_.points; }
        [[nodiscard]] bool lassoAddMode() const { return lasso_.add_mode; }
        [[nodiscard]] std::optional<SplitViewPanelId> lassoPanel() const { return lasso_.panel; }

        [[nodiscard]] int hoveredGaussianId() const { return hovered_gaussian_id_; }
        void setHoveredGaussianId(int hovered_gaussian_id) { hovered_gaussian_id_ = hovered_gaussian_id; }

        void setCropbox(bool active, const glm::vec3& min, const glm::vec3& max,
                        const glm::mat4& world_transform) {
            cropbox_active_ = active;
            if (active) {
                cropbox_min_ = min;
                cropbox_max_ = max;
                cropbox_transform_ = world_transform;
            }
        }

        void setEllipsoid(bool active, const glm::vec3& radii, const glm::mat4& world_transform) {
            ellipsoid_active_ = active;
            if (active) {
                ellipsoid_radii_ = radii;
                ellipsoid_transform_ = world_transform;
            }
        }

        void setCropboxActive(bool active) { cropbox_active_ = active; }
        void setEllipsoidActive(bool active) { ellipsoid_active_ = active; }

        [[nodiscard]] GizmoState makeFrameGizmoState() const {
            return GizmoState{
                .cropbox_active = cropbox_active_,
                .cropbox_min = cropbox_min_,
                .cropbox_max = cropbox_max_,
                .cropbox_transform = cropbox_transform_,
                .ellipsoid_active = ellipsoid_active_,
                .ellipsoid_radii = ellipsoid_radii_,
                .ellipsoid_transform = ellipsoid_transform_,
            };
        }

    private:
        struct RectPreviewState {
            bool active = false;
            float x0 = 0.0f;
            float y0 = 0.0f;
            float x1 = 0.0f;
            float y1 = 0.0f;
            bool add_mode = true;
            std::optional<SplitViewPanelId> panel;
        };

        struct PolygonPreviewState {
            bool active = false;
            std::vector<std::pair<float, float>> points;
            std::vector<glm::vec3> world_points;
            bool closed = false;
            bool add_mode = true;
            bool world_space = false;
            std::optional<SplitViewPanelId> panel;
        };

        struct LassoPreviewState {
            bool active = false;
            std::vector<std::pair<float, float>> points;
            bool add_mode = true;
            std::optional<SplitViewPanelId> panel;
        };

        CursorPreviewState cursor_preview_;
        RectPreviewState rect_;
        PolygonPreviewState polygon_;
        LassoPreviewState lasso_;
        int hovered_gaussian_id_ = -1;

        bool cropbox_active_ = false;
        bool ellipsoid_active_ = false;
        glm::vec3 cropbox_min_{0.0f};
        glm::vec3 cropbox_max_{0.0f};
        glm::mat4 cropbox_transform_{1.0f};
        glm::vec3 ellipsoid_radii_{1.0f};
        glm::mat4 ellipsoid_transform_{1.0f};
    };

} // namespace lfs::vis
