/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#define GLM_ENABLE_EXPERIMENTAL

#include "visualizer/nodes/viewport_coordinates.hpp"

#include "core/nodes/types.hpp"

#include "visualizer/scene_coordinate_utils.hpp"

#include <cmath>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/euler_angles.hpp>

namespace lfs::vis::nodes {
    namespace {
        glm::vec3 safeScale(const glm::mat4& value) {
            return {glm::length(glm::vec3(value[0])), glm::length(glm::vec3(value[1])),
                    glm::length(glm::vec3(value[2]))};
        }
    } // namespace

    ViewportCoordinates::ViewportCoordinates(const core::Scene& scene, const core::NodeId host) {
        if (!scene.getNodeById(host))
            return;
        local_to_world_ = scene_coords::nodeVisualizerWorldTransform(scene, host);
        const float determinant = glm::determinant(local_to_world_);
        if (!std::isfinite(determinant) || std::abs(determinant) <= 1e-8f)
            return;
        world_to_local_ = glm::inverse(local_to_world_);
        valid_ = true;
    }

    glm::vec3 ViewportCoordinates::pointToWorld(const glm::vec3& local) const {
        return glm::vec3(local_to_world_ * glm::vec4(local, 1.0f));
    }

    glm::vec3 ViewportCoordinates::pointToLocal(const glm::vec3& world) const {
        return glm::vec3(world_to_local_ * glm::vec4(world, 1.0f));
    }

    float ViewportCoordinates::radiusToLocal(const float world_radius) const {
        const float scale = (glm::length(glm::vec3(world_to_local_[0])) +
                             glm::length(glm::vec3(world_to_local_[1])) +
                             glm::length(glm::vec3(world_to_local_[2]))) /
                            3.0f;
        return world_radius * scale;
    }

    glm::mat4 ViewportCoordinates::composeLocal(const NodeViewportTransform& transform) {
        return glm::translate(glm::mat4(1.0f), transform.translation) *
               lfs::nodes::rotation_matrix(transform.rotation_degrees) *
               glm::scale(glm::mat4(1.0f), transform.scale);
    }

    NodeViewportTransform ViewportCoordinates::decomposeLocal(const glm::mat4& transform) {
        NodeViewportTransform result;
        result.translation = glm::vec3(transform[3]);
        result.scale = safeScale(transform);
        glm::mat4 rotation(1.0f);
        for (int axis = 0; axis < 3; ++axis) {
            if (result.scale[axis] > 1e-8f)
                rotation[axis] = glm::vec4(glm::vec3(transform[axis]) / result.scale[axis], 0.0f);
        }
        if (glm::determinant(glm::mat3(rotation)) < 0.0f) {
            result.scale.z = -result.scale.z;
            rotation[2] = -rotation[2];
        }
        glm::vec3 radians(0.0f);
        // Inverse of rotation_matrix: R = Rz * Ry * Rx.
        glm::extractEulerAngleZYX(rotation, radians.z, radians.y, radians.x);
        result.rotation_degrees = glm::degrees(radians);
        return result;
    }

    glm::mat4 ViewportCoordinates::transformToWorld(const NodeViewportTransform& local) const {
        return local_to_world_ * composeLocal(local);
    }

    NodeViewportTransform ViewportCoordinates::transformToLocal(const glm::mat4& world) const {
        return decomposeLocal(world_to_local_ * world);
    }

} // namespace lfs::vis::nodes
