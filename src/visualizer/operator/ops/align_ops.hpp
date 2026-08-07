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
        std::optional<SplitViewPanelId> pick_panel_;

        [[nodiscard]] glm::vec3 unprojectScreenPoint(const OperatorContext& ctx, double x, double y,
                                                     SplitViewPanelId* out_panel = nullptr) const;
        [[nodiscard]] bool isNearExistingPoint(double x, double y) const;
        [[nodiscard]] glm::vec3 resolvePickPanelCameraPosition() const;
        void syncPickedPointsToServices();
        void removeLastPoint();
        [[nodiscard]] bool tryPlacePoint(OperatorContext& ctx, double x, double y);
        [[nodiscard]] bool applyAlignment(OperatorContext& ctx);
        void setStatus(const char* locale_key, double duration_seconds = 1.5) const;
    };

    void registerAlignOperators();
    void unregisterAlignOperators();

} // namespace lfs::vis::op
