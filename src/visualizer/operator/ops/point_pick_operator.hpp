/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/scene.hpp"
#include "core/services.hpp"
#include "internal/viewport.hpp"
#include "operator/operator.hpp"
#include "rendering/rendering_types.hpp"

namespace lfs::vis::op {

    class LFS_VIS_API PointPickOperator : public Operator {
    public:
        OperatorResult invoke(OperatorContext& ctx, OperatorProperties& props) override;
        OperatorResult modal(OperatorContext& ctx, OperatorProperties& props) override;
        void cancel(OperatorContext& ctx) override;

    protected:
        PointPickOperator(PointPickState& state, size_t max_points, const char* no_surface_key)
            : state_(state), max_points_(max_points), no_surface_key_(no_surface_key) {}

        virtual std::optional<OperatorResult> prepareModal(OperatorContext&) { return std::nullopt; }
        virtual bool pointEditingBlocked() const { return false; }
        virtual void beforePointEdit(OperatorContext&) {}
        virtual void pointSampleApplied() {}
        virtual bool applyPickedPoints(OperatorContext& ctx, float value) = 0;
        virtual void handleToolUiAction(OperatorContext&, PointPickState::UiAction) {}
        virtual OperatorResult handleToolAction(OperatorContext&, input::Action) { return OperatorResult::PASS_THROUGH; }

        std::vector<glm::vec3> picked_points_;
        int pick_button_ = 0;
        bool press_active_ = false;
        int press_button_ = -1;
        glm::dvec2 press_pos_{0.0, 0.0};
        std::optional<int> press_point_index_;
        bool drag_active_ = false;
        std::optional<int> selected_point_;
        std::optional<SplitViewPanelId> pick_panel_;
        struct PendingPick {
            glm::dvec2 screen_position;
            // Empty for a new point; otherwise replace the dragged marker.
            std::optional<int> point_index;
        };
        struct PointSample {
            glm::vec3 world{Viewport::INVALID_WORLD_POS};
            SplitViewPanelId panel = SplitViewPanelId::Left;
            bool pending = false;
        };
        std::optional<PendingPick> pending_pick_;
        void resolvePendingPoint();
        [[nodiscard]] bool requestPoint(const PendingPick& pick);
        [[nodiscard]] bool applyPointSample(const PointSample& sample, std::optional<int> index);
        [[nodiscard]] PointSample sampleScreenPoint(glm::dvec2 position) const;
        [[nodiscard]] std::optional<int> hitTestPoint(double x, double y) const;
        void syncPickedPointsToServices();
        void removeLastPoint();
        void removeSelectedPoint();
        void clearAllPoints();
        [[nodiscard]] bool tryPlacePoint(double x, double y);
        void setStatus(const char* locale_key, double duration_seconds = 1.5) const;

    private:
        OperatorResult handlePendingUiAction(OperatorContext& ctx);
        void markOverlayDirty() const;
        PointPickState& state_;
        size_t max_points_;
        const char* no_surface_key_;
    };

    [[nodiscard]] core::NodeId resolvePointPickTargetId(const core::Scene& scene, const core::SceneNode& node,
                                                        bool include_crop_polygon);
    [[nodiscard]] std::vector<core::NodeId> resolvePointPickTargets(const OperatorContext& ctx, bool include_crop_polygon);

    // Orthographic pixels per world unit for one viewer panel: a split-view panel can carry its own zoom.
    [[nodiscard]] inline float pointPickPanelOrthoScale(const Viewport& panel_viewport, const RenderSettings& settings) {
        return panel_viewport.ortho_scale_override.value_or(settings.ortho_scale);
    }

    inline constexpr float kPointPickMarkerWorldRadius = 0.05f;
    [[nodiscard]] float pointPickMarkerScreenRadius(const glm::vec3& world_pos,
                                                    const glm::mat4& view,
                                                    const glm::mat4& projection,
                                                    float window_height,
                                                    bool orthographic,
                                                    float ortho_scale,
                                                    float screen_scale_x,
                                                    float screen_scale_y);

} // namespace lfs::vis::op
