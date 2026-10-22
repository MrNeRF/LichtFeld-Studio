/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/export.hpp"
#include <cstdint>
#include <functional>
#include <glm/glm.hpp>
#include <span>
#include <stop_token>
#include <string>
#include <vector>

namespace lfs::core {

    enum class CollisionPreset { Indoor,
                                 Outdoor,
                                 Object };
    enum class CollisionShape { Smooth,
                                Faces };

    struct LFS_CORE_API CollisionSettings {
        CollisionPreset preset = CollisionPreset::Outdoor;
        bool cluster = false;
        float cluster_resolution = 1.0f;
        float cluster_opacity = 0.999f;
        float cluster_min_contribution = 0.1f;
        float voxel_size = 0.05f;
        float opacity = 0.1f;
        bool external_fill = false;
        float external_size = 1.6f;
        bool floor_fill = true;
        float floor_size = 1.6f;
        float carve_height = 0.0f;
        float carve_radius = 0.2f;
        CollisionShape shape = CollisionShape::Smooth;
        glm::vec3 seed{0};
        void applyPreset(CollisionPreset value);
        void validate() const;
        friend bool operator==(const CollisionSettings&, const CollisionSettings&) = default;
    };

    // axes maps a unit sphere to the Gaussian's rotated/scaled one-sigma ellipsoid.
    struct CollisionGaussian {
        glm::vec3 center;
        glm::mat3 axes;
        float opacity;
    };

    struct CollisionMesh {
        std::vector<float> vertices;
        std::vector<int32_t> indices;
    };

    // Immutable once attached to a scene node. Coordinates are node-local, +Y up.
    struct LFS_CORE_API CollisionVolume {
        static constexpr size_t max_cells = 16'000'000;
        CollisionSettings settings;
        glm::vec3 origin{0};
        glm::ivec3 dimensions{0};
        std::vector<uint8_t> solid;
        // Sparse payload: 0/1 uniform leaves, otherwise child-base + 2.
        std::vector<uint32_t> tree;
        int tree_size = 0;
        bool show_voxels = false;
        std::string warning;
        friend bool operator==(const CollisionVolume&, const CollisionVolume&) = default;

        [[nodiscard]] size_t index(glm::ivec3 p) const;
        [[nodiscard]] bool contains(glm::ivec3 p) const;
        [[nodiscard]] bool occupied(glm::ivec3 p) const;
        [[nodiscard]] glm::ivec3 cell(glm::vec3 p) const;
        [[nodiscard]] bool capsuleIntersects(glm::vec3 feet, glm::vec3 up,
                                             float height, float radius) const;
        void validate() const;
        [[nodiscard]] std::vector<uint32_t> packedOccupancy() const;
        [[nodiscard]] std::vector<uint32_t> octreeOccupancy() const;
        [[nodiscard]] std::vector<uint32_t> gpuOccupancy() const;
        [[nodiscard]] uint64_t occupiedCount() const;
        void forEachSolidBox(const std::function<void(glm::ivec3, glm::ivec3)>& fn) const;
        void forEachBoundaryFace(const std::function<void(glm::ivec3, glm::ivec3)>& fn,
                                 std::stop_token stop = {}) const;
        [[nodiscard]] CollisionMesh mesh(CollisionShape shape, std::stop_token stop = {}) const;
    };

    using CollisionProgress = std::function<void(std::string)>;
    [[nodiscard]] LFS_CORE_API CollisionVolume generateCollision(
        std::span<const CollisionGaussian> gaussians, CollisionSettings settings,
        std::stop_token stop = {}, CollisionProgress progress = {});
    [[nodiscard]] LFS_CORE_API CollisionVolume processCollisionVolume(
        CollisionVolume volume, std::stop_token stop = {}, CollisionProgress progress = {});

    struct LFS_CORE_API WalkSettings {
        float height = 1.8f;
        float radius = 0.2f;
        float speed = 2.5f;
        float step_height = 0.25f;
        float gravity = 9.81f;
        void validate() const;
    };

    struct WalkState {
        glm::vec3 eye{0};
        float vertical_velocity = 0;
        bool grounded = false;
    };

    // The callback tests a world-space capsule against all active collision nodes.
    using CapsuleQuery = std::function<bool(glm::vec3, glm::vec3, float, float)>;
    [[nodiscard]] LFS_CORE_API glm::mat3 walkTangentFrame(glm::vec3 forward, glm::vec3 right, glm::vec3 up);
    [[nodiscard]] LFS_CORE_API bool transformedCapsuleIntersects(
        const CollisionVolume& volume, const glm::mat4& local_to_world,
        glm::vec3 feet, glm::vec3 up, float height, float radius);
    LFS_CORE_API void advanceWalk(WalkState& state, const WalkSettings& settings,
                                  glm::vec3 up, glm::vec3 desired_velocity, float dt,
                                  const CapsuleQuery& blocked);
} // namespace lfs::core
