/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/scene.hpp"

#include <glm/glm.hpp>

namespace lfs::vis::nodes {

    struct NodeViewportTransform {
        glm::vec3 translation{0.0f};
        glm::vec3 rotation_degrees{0.0f};
        glm::vec3 scale{1.0f};
    };

    // Node graph positions are stored in the host object's data-local basis.
    // This is the only boundary used by node viewport tools to cross into the
    // visualizer world basis (Y/Z flipped, with the host world transform).
    class LFS_VIS_API ViewportCoordinates final {
    public:
        ViewportCoordinates(const core::Scene& scene, core::NodeId host);

        [[nodiscard]] bool valid() const noexcept { return valid_; }
        [[nodiscard]] glm::vec3 pointToWorld(const glm::vec3& local) const;
        [[nodiscard]] glm::vec3 pointToLocal(const glm::vec3& world) const;
        [[nodiscard]] float radiusToLocal(float world_radius) const;
        [[nodiscard]] glm::mat4 transformToWorld(const NodeViewportTransform& local) const;
        [[nodiscard]] NodeViewportTransform transformToLocal(const glm::mat4& world) const;

        [[nodiscard]] static glm::mat4 composeLocal(const NodeViewportTransform& transform);
        [[nodiscard]] static NodeViewportTransform decomposeLocal(const glm::mat4& transform);

    private:
        bool valid_ = false;
        glm::mat4 local_to_world_{1.0f};
        glm::mat4 world_to_local_{1.0f};
    };

} // namespace lfs::vis::nodes
