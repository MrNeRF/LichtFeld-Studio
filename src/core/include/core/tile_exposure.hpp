/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/scene_partitioner.hpp"

#include <cstddef>
#include <functional>
#include <vector>

namespace lfs::core {

    struct TileExposureSample {
        int camera_uid = -1;
        glm::vec3 point{0.0f};
        // Encoded RGB luminance in [0, 1], sampled at an SfM observation.
        float luminance = 0.0f;
    };

    struct TileExposureGain {
        int camera_uid = -1;
        float gain = 1.0f;
        int shared_points = 0;
    };

    struct TileExposureCalibration {
        std::vector<TileExposureGain> cameras;
        size_t shared_points = 0;
        size_t reliable_edges = 0;
        int connected_cameras = 0;
    };

    // Decodes one image at a time, sampling the same SfM point deterministically
    // across cameras. Missing or unreadable images contribute no samples.
    [[nodiscard]] LFS_CORE_API std::vector<TileExposureSample> sample_tile_exposure(
        const std::vector<TileCameraEvidence>& cameras,
        size_t max_tracks_per_camera = 2048,
        const std::function<bool()>& stop_requested = {});

    // Solves one global relative brightness scale from shared SfM tracks.
    // Cameras outside the largest well-constrained view component retain gain 1.
    [[nodiscard]] LFS_CORE_API TileExposureCalibration estimate_tile_exposure(
        const std::vector<TileExposureSample>& samples,
        const std::vector<int>& camera_uids);

} // namespace lfs::core
