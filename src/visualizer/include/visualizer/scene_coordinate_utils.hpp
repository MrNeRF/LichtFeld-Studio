/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/scene.hpp"
#include "rendering/coordinate_conventions.hpp"
#include <optional>
#include <string>

namespace lfs::vis::scene_coords {

    [[nodiscard]] inline glm::mat4 nodeDataWorldTransform(const core::Scene& scene, const core::NodeId node_id) {
        return scene.getWorldTransform(node_id);
    }

    [[nodiscard]] inline glm::mat4 nodeVisualizerWorldTransform(const core::Scene& scene, const core::NodeId node_id) {
        return rendering::dataWorldTransformToVisualizerWorld(nodeDataWorldTransform(scene, node_id));
    }

    [[nodiscard]] inline std::optional<glm::mat4> nodeVisualizerWorldTransform(
        const core::Scene& scene,
        const std::string& name) {
        const auto* const node = scene.getNode(name);
        if (!node)
            return std::nullopt;
        return nodeVisualizerWorldTransform(scene, node->id);
    }

    [[nodiscard]] inline std::optional<glm::mat4> nodeLocalTransformFromVisualizerWorld(
        const core::Scene& scene,
        const core::NodeId node_id,
        const glm::mat4& visualizer_world_transform) {
        const auto* const node = scene.getNodeById(node_id);
        if (!node)
            return std::nullopt;

        glm::mat4 parent_world_transform(1.0f);
        if (node->parent_id != core::NULL_NODE) {
            parent_world_transform = scene.getWorldTransform(node->parent_id);
        }

        const glm::mat4 data_world_transform =
            rendering::visualizerWorldTransformToDataWorld(visualizer_world_transform);
        return glm::inverse(parent_world_transform) * data_world_transform;
    }

    [[nodiscard]] inline std::optional<glm::mat4> nodeLocalTransformFromVisualizerWorld(
        const core::Scene& scene,
        const std::string& name,
        const glm::mat4& visualizer_world_transform) {
        const auto* const node = scene.getNode(name);
        if (!node)
            return std::nullopt;
        return nodeLocalTransformFromVisualizerWorld(scene, node->id, visualizer_world_transform);
    }

} // namespace lfs::vis::scene_coords
