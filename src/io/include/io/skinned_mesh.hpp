/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "io/error.hpp"
#include <array>
#include <cstdint>
#include <filesystem>
#include <glm/glm.hpp>
#include <string>
#include <vector>

namespace lfs::io {

    // A triangle mesh bound to a skeleton, as a model file holds it: the rest
    // pose, and for every vertex the bones that move it.
    struct SkinnedMesh {
        struct Bone {
            std::string name;
            // Takes a vertex of the rest pose into the frame of the bone.
            glm::mat4 inverse_bind{1.0f};
        };
        struct Vertex {
            glm::vec3 position{0.0f};
            glm::vec3 normal{0.0f};
            // Up to four bones, as indices into `bones`, and how much each
            // moves the vertex: the weights sum to one.
            std::array<std::uint16_t, 4> bones{};
            std::array<float, 4> weights{};
        };
        std::vector<Vertex> vertices;
        std::vector<std::uint32_t> indices; // triangles
        std::vector<Bone> bones;
    };

    // Reads the first mesh that has a skeleton from a model file, through
    // Assimp as MeshLoader does, with the bones kept: every bone of the skin,
    // in the order of the file, whether or not it moves a vertex.
    [[nodiscard]] LFS_IO_API Result<SkinnedMesh> load_skinned_mesh(const std::filesystem::path& path);

} // namespace lfs::io
