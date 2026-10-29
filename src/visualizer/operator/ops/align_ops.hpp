/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "point_pick_operator.hpp"
#include <glm/glm.hpp>
#include <optional>
#include <vector>

namespace lfs::vis::op {

    class LFS_VIS_API AlignPickPointOperator : public PointPickOperator {
    public:
        AlignPickPointOperator();

        static LFS_LOCAL_SYMBOL const OperatorDescriptor DESCRIPTOR;

        [[nodiscard]] const OperatorDescriptor& descriptor() const override { return DESCRIPTOR; }
        [[nodiscard]] bool poll(const OperatorContext& ctx, const OperatorProperties* props = nullptr) const override;

    private:
        friend class AlignPreviewTest;

        struct PreviewTarget {
            core::NodeId id;
            core::Uuid uuid;
            glm::mat4 local;
            glm::mat4 world;
        };
        std::vector<PreviewTarget> preview_targets_;
        std::optional<glm::mat4> preview_snap_world_;
        glm::vec3 preview_camera_{};
        void restorePreview(OperatorContext& ctx);
        [[nodiscard]] bool updatePreview(OperatorContext& ctx);
        [[nodiscard]] bool applyAlignment(OperatorContext& ctx);
        bool applyPickedPoints(OperatorContext& ctx, float value) override;
        std::optional<OperatorResult> prepareModal(OperatorContext& ctx) override;
        bool pointEditingBlocked() const override;
        void beforePointEdit(OperatorContext& ctx) override;
        void pointSampleApplied() override;
        void handleToolUiAction(OperatorContext& ctx, PointPickState::UiAction action) override;
        OperatorResult handleToolAction(OperatorContext& ctx, input::Action action) override;
        [[nodiscard]] glm::vec3 resolvePickPanelCameraPosition() const;
    };

    void registerAlignOperators();
    void unregisterAlignOperators();

    // Shared by apply path and overlay preview. Returns true if normal was snapped.
    inline constexpr float kAlignAxisSnapDegrees = 3.0f;
    [[nodiscard]] LFS_VIS_API bool snapAlignNormalToNodeAxes(glm::vec3& normal,
                                                             const glm::mat4& node_world,
                                                             float max_degrees = kAlignAxisSnapDegrees);

    // Flip face normal so it points toward the camera (shared by apply + overlay preview).
    LFS_VIS_API void faceNormalTowardCamera(glm::vec3& normal, const glm::vec3& center, const glm::vec3& camera_pos);

    // True when exactly 3 points define a non-degenerate triangle (cross length > 1e-6).
    [[nodiscard]] LFS_VIS_API bool pointsAreNonDegenerate(const std::vector<glm::vec3>& points);

    // Optional in-plane yaw after normal→up: aligns projected (p1-p0) with world +X.
    // Returns the yaw rotation to left-multiply onto the up-alignment rotation (identity if skipped).
    [[nodiscard]] LFS_VIS_API glm::mat4 alignEdgeToWorldXRotation(const glm::mat4& up_rotation,
                                                                  const glm::vec3& p0,
                                                                  const glm::vec3& p1);

    struct AlignTransformInputs {
        glm::vec3 p0{};
        glm::vec3 p1{};
        glm::vec3 p2{};
        glm::vec3 camera_pos{};
        std::optional<glm::mat4> snap_node_world;
        bool edge_to_world_x = false;
    };

    // Visualizer-world transform mapping the picked plane normal (camera-facing, optionally
    // snapped) onto +Y with the triangle centroid moved to y = 0, or nullopt if degenerate.
    [[nodiscard]] LFS_VIS_API std::optional<glm::mat4> computeAlignTransform(const AlignTransformInputs& in);

    [[nodiscard]] LFS_VIS_API std::optional<glm::mat4> resolveAlignSnapTargetWorld(const SceneManager& scene);

} // namespace lfs::vis::op
