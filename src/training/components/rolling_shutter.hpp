/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include <array>
#include <cstddef>
#include <expected>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>

namespace lfs::training {

    // GeosunPre writes rolling_shutter.json beside the COLMAP model when its shutter
    // correction is accepted: the photos are the original rolling-shutter files and the
    // COLMAP poses describe the middle of the readout.
    inline constexpr const char* ROLLING_SHUTTER_SIDECAR = "rolling_shutter.json";

    struct RollingShutterTable {
        struct Image {
            int camera_id = -1; // -1: not recorded, take the camera's own id
            std::array<double, 3> rotation{};
            std::array<double, 3> velocity{};
            std::array<double, 3> translation{}; // must equal readout_s * velocity
        };
        bool readout_rows = true;
        std::unordered_map<int, double> readout_s; // per COLMAP camera, signed by scan order
        std::unordered_map<std::string, Image> images;
    };

    // Parses version 1 of the sidecar. Rejects unknown models, motion outside GeosunPre's
    // operating range and translations that disagree with readout time times velocity.
    std::expected<RollingShutterTable, std::string> parse_rolling_shutter(std::string_view json_text);

    // Returns no table when the dataset has no sidecar.
    std::expected<std::optional<RollingShutterTable>, std::string> read_rolling_shutter(
        const std::filesystem::path& dataset);

    // Attaches each camera's motion, matched by COLMAP image name. Every camera must have
    // an entry whose camera id agrees with its own.
    std::expected<void, std::string> attach_rolling_shutter(
        const RollingShutterTable& table, std::span<const std::shared_ptr<core::Camera>> cameras);

    void clear_rolling_shutter(std::span<const std::shared_ptr<core::Camera>> cameras);

    // Row-major world-to-camera transforms of the first and last readout line of a render
    // window, for gsplat's two-pose rolling shutter (rotation slerp, translation lerp,
    // relative time floor(c) / (extent - 1) inside the window). The window starts at
    // `offset` source pixels along the readout axis of `full_extent` pixels.
    struct RollingShutterWindowPoses {
        std::array<float, 16> first{};
        std::array<float, 16> last{};
    };
    RollingShutterWindowPoses rolling_shutter_window_poses(
        const core::RollingShutterMotion& motion,
        const std::array<float, 9>& rotation, // reference world-to-camera, row-major
        const std::array<float, 3>& translation,
        int full_extent, int offset, int extent);

} // namespace lfs::training
