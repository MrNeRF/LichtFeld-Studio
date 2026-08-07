/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "operator/operator.hpp"
#include "rendering/rendering_types.hpp"
#include <glm/glm.hpp>
#include <optional>
#include <vector>

namespace lfs::vis::op {

    class AlignPickPointOperator : public Operator {
    public:
        static LFS_LOCAL_SYMBOL const OperatorDescriptor DESCRIPTOR;

        [[nodiscard]] const OperatorDescriptor& descriptor() const override { return DESCRIPTOR; }
        [[nodiscard]] bool poll(const OperatorContext& ctx, const OperatorProperties* props = nullptr) const override;
        OperatorResult invoke(OperatorContext& ctx, OperatorProperties& props) override;
        OperatorResult modal(OperatorContext& ctx, OperatorProperties& props) override;
        void cancel(OperatorContext& ctx) override;

    private:
        std::vector<glm::vec3> picked_points_;
        int pick_button_ = 0;
        bool press_active_ = false;
        int press_button_ = -1;
        glm::dvec2 press_pos_{0.0, 0.0};
        std::optional<int> press_point_index_;
        bool drag_active_ = false;
        int drag_point_index_ = -1;
        std::optional<int> selected_point_;
        std::optional<SplitViewPanelId> pick_panel_;

        [[nodiscard]] glm::vec3 unprojectScreenPoint(const OperatorContext& ctx, double x, double y,
                                                     SplitViewPanelId* out_panel = nullptr) const;
        [[nodiscard]] std::optional<int> hitTestPoint(double x, double y) const;
        [[nodiscard]] glm::vec3 resolvePickPanelCameraPosition() const;
        void syncPickedPointsToServices();
        void removeLastPoint();
        void removeSelectedPoint();
        void clearAllPoints();
        [[nodiscard]] bool tryPlacePoint(OperatorContext& ctx, double x, double y);
        [[nodiscard]] bool applyAlignment(OperatorContext& ctx);
        [[nodiscard]] OperatorResult handlePendingUiAction(OperatorContext& ctx);
        void setStatus(const char* locale_key, double duration_seconds = 1.5) const;
    };

    void registerAlignOperators();
    void unregisterAlignOperators();

    // Shared by apply path and overlay preview. Returns true if normal was snapped.
    [[nodiscard]] bool snapAlignNormalToNodeAxes(glm::vec3& normal,
                                                 const glm::mat4& node_world,
                                                 float max_degrees = 3.0f);

} // namespace lfs::vis::op
