/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/scene.hpp"
#include "internal/viewport.hpp"
#include "operator/operator.hpp"
#include "rendering/rendering_types.hpp"
#include <glm/glm.hpp>
#include <optional>
#include <vector>

namespace lfs::vis::op {

    class LFS_VIS_API ScaleDistancePickPointOperator : public Operator {
    public:
        static LFS_LOCAL_SYMBOL const OperatorDescriptor DESCRIPTOR;

        [[nodiscard]] const OperatorDescriptor& descriptor() const override { return DESCRIPTOR; }
        [[nodiscard]] bool poll(const OperatorContext& ctx, const OperatorProperties* props = nullptr) const override;
        OperatorResult invoke(OperatorContext& ctx, OperatorProperties& props) override;
        OperatorResult modal(OperatorContext& ctx, OperatorProperties& props) override;
        void cancel(OperatorContext& ctx) override;

    private:
        friend class ScaleDistanceOperatorTest;

        void setPickedPointsForTesting(const std::vector<glm::vec3>& pts) {
            picked_points_ = pts;
            syncPickedPointsToServices();
        }
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
        [[nodiscard]] bool applyScale(OperatorContext& ctx, float target_distance);
        [[nodiscard]] OperatorResult handlePendingUiAction(OperatorContext& ctx);
        void setStatus(const char* locale_key, double duration_seconds = 1.5) const;
    };

    void registerScaleDistanceOperators();
    void unregisterScaleDistanceOperators();

    // Compute scale matrix around the midpoint of p0 and p1 that rescales distance ||p1 - p0|| to target_distance.
    // Returns std::nullopt if points are too close (distance < 1e-6) or target_distance <= 1e-6.
    [[nodiscard]] LFS_VIS_API std::optional<glm::mat4> computeScaleDistanceTransform(
        const glm::vec3& p0, const glm::vec3& p1, float target_distance);

} // namespace lfs::vis::op
