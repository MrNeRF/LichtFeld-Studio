/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/collision.hpp"
#include <filesystem>

namespace lfs::io {
    // Both writers use PlayCanvas coordinates: Rz(pi) * the node's data-world
    // transform, matching a world-transformed PLY imported by SuperSplat.
    // Voxel export preserves occupancy exactly; rotated/sheared/nonuniform grids
    // that cannot be represented by the format fail rather than misalign/resample.
    void exportCollisionVoxels(const core::CollisionVolume& volume, const glm::mat4& data_world,
                               const std::filesystem::path& json_path, std::stop_token stop = {},
                               core::CollisionProgress progress = {});
    void exportCollisionGlb(std::span<const float> vertices, std::span<const int32_t> indices, const glm::mat4& data_world,
                            const std::filesystem::path& path, std::stop_token stop = {},
                            core::CollisionProgress progress = {});
} // namespace lfs::io
