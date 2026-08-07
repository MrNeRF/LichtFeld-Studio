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
        mutable bool logged_masked_depth_fallback_ = false;
        mutable bool logged_exact_depth_fallback_ = false;

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
    inline constexpr float kAlignAxisSnapDegrees = 3.0f;
    [[nodiscard]] bool snapAlignNormalToNodeAxes(glm::vec3& normal,
                                                 const glm::mat4& node_world,
                                                 float max_degrees = kAlignAxisSnapDegrees);

    // Flip face normal so it points toward the camera (shared by apply + overlay preview).
    void faceNormalTowardCamera(glm::vec3& normal, const glm::vec3& center, const glm::vec3& camera_pos);

    // True when exactly 3 points define a non-degenerate triangle (cross length > 1e-6).
    [[nodiscard]] bool pointsAreNonDegenerate(const std::vector<glm::vec3>& points);

    // Optional in-plane yaw after normal→up: aligns projected (p1-p0) with world +X.
    // Returns the yaw rotation to left-multiply onto the up-alignment rotation (identity if skipped).
    [[nodiscard]] glm::mat4 alignEdgeToWorldXRotation(const glm::mat4& up_rotation,
                                                      const glm::vec3& p0,
                                                      const glm::vec3& p1);

} // namespace lfs::vis::op
