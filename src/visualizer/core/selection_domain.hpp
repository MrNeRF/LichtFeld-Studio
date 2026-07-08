/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include <cstdint>

namespace lfs::core {
    class SceneNode;
}

namespace lfs::vis {

    class SceneManager;

    enum class SelectionDomain : uint8_t {
        Gaussians,
        PointCloud,
        Cameras
    };

    // PointCloud: selected POINTCLOUD, or selected CROPBOX/ELLIPSOID parented to POINTCLOUD.
    // Cameras: non-empty selection containing only CAMERA nodes, CAMERA_GROUP
    //   nodes, or a camera-only GROUP container; sticks across an emptied
    //   selection while cameras stay visible (SceneManager::stickySelectionDomain).
    // PointCloud auto-target: empty selection, no gaussians, exactly one visible POINTCLOUD with points.
    // Cameras auto-target: empty selection, no gaussians, no visible point clouds, visible CAMERA nodes.
    // Gaussians: fallback.
    [[nodiscard]] LFS_VIS_API SelectionDomain resolveSelectionDomain(const SceneManager& scene_manager);
    [[nodiscard]] LFS_VIS_API const core::SceneNode* resolveActivePointCloudNode(const SceneManager& scene_manager);

} // namespace lfs::vis
