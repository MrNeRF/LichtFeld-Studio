/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/crop_polygon.hpp"
#include "core/export.hpp"
#include "core/scene.hpp"
#include "operator/operator.hpp"
#include <glm/glm.hpp>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace lfs::vis::op {

    // Draws a crop polygon on the plane through the first click that faces the
    // camera. Vertices stay world points, so the camera may orbit while drawing.
    class LFS_VIS_API CropPolygonDrawOperator : public Operator {
    public:
        static LFS_LOCAL_SYMBOL const OperatorDescriptor DESCRIPTOR;

        [[nodiscard]] const OperatorDescriptor& descriptor() const override { return DESCRIPTOR; }
        [[nodiscard]] bool poll(const OperatorContext& ctx, const OperatorProperties* props = nullptr) const override;
        OperatorResult invoke(OperatorContext& ctx, OperatorProperties& props) override;
        OperatorResult modal(OperatorContext& ctx, OperatorProperties& props) override;
        void cancel(OperatorContext& ctx) override;

    private:
        struct StartState {
            std::vector<core::NodeId> selection;
            std::vector<std::pair<core::NodeId, bool>> volume_visibility;
            std::string active_tool;
            std::string editor_operator;
            std::string editor_gizmo;
        };

        // Cancel, Escape or undoing past the first vertex: leave the scene and tools as they were.
        [[nodiscard]] OperatorResult cancelByUser(OperatorContext& ctx);
        // Returns true when the click closed the outline.
        [[nodiscard]] bool addVertex(glm::vec2 screen_point);
        void updateCursor(glm::vec2 screen_point);
        [[nodiscard]] bool nearFirstVertex(glm::vec2 screen_point) const;
        [[nodiscard]] bool finish(OperatorContext& ctx);
        void publishDraft() const;
        void clearDraft() const;

        core::NodeId target_node_id_ = core::NULL_NODE;
        std::optional<core::crop_polygon::Plane> plane_;
        glm::vec3 camera_right_{1.0f, 0.0f, 0.0f};
        StartState start_;
        std::vector<glm::vec3> points_;
        std::optional<glm::vec3> cursor_;
    };

    // Sets a crop polygon's depth, inverse or enabled flag; node names the polygon or its splat and
    // defaults to the selection.
    class LFS_VIS_API CropPolygonSetOperator : public Operator {
    public:
        static LFS_LOCAL_SYMBOL const OperatorDescriptor DESCRIPTOR;

        [[nodiscard]] const OperatorDescriptor& descriptor() const override { return DESCRIPTOR; }
        [[nodiscard]] bool poll(const OperatorContext& ctx, const OperatorProperties* props = nullptr) const override;
        OperatorResult invoke(OperatorContext& ctx, OperatorProperties& props) override;
    };

    // Node types a crop polygon can be drawn for.
    [[nodiscard]] inline bool isCropPolygonTarget(const core::NodeType type) {
        return core::isSceneNodeParentCompatible(type, core::NodeType::CROP_POLYGON);
    }

    void registerCropPolygonOperators();
    void unregisterCropPolygonOperators();

} // namespace lfs::vis::op
