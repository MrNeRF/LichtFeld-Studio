/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <algorithm>
#include <cmath>

namespace lfs::io::video {

    enum class VideoFrameFit { FIT,
                               CROP,
                               STRETCH };

    struct VideoFrameGeometry {
        int width;
        int height;
    };

    // Bound the intermediate image, including extreme custom aspect ratios.
    // A zero camera size preserves the pre-existing export API behavior.
    [[nodiscard]] inline VideoFrameGeometry videoCameraRenderSize(
        int width, int height, int camera_width, int camera_height, VideoFrameFit fit) {
        if (camera_width <= 0 || camera_height <= 0)
            return {width, height};
        const double sx = static_cast<double>(width) / camera_width;
        const double sy = static_cast<double>(height) / camera_height;
        const double scale = std::min(fit == VideoFrameFit::FIT ? std::min(sx, sy) : std::max(sx, sy),
                                      8192.0 / std::max(camera_width, camera_height));
        return {std::max(1, static_cast<int>(std::lround(camera_width * scale))),
                std::max(1, static_cast<int>(std::lround(camera_height * scale)))};
    }

    struct VideoFrameMapping {
        float scale_x;
        float scale_y;
        float offset_x;
        float offset_y;
    };

    // Destination pixel centers map into source edge coordinates. FIT leaves
    // out-of-image samples black; CROP retains aspect; STRETCH permits distortion.
    [[nodiscard]] inline VideoFrameMapping videoFrameMapping(
        int source_width, int source_height, int width, int height, VideoFrameFit fit) {
        const float sx = static_cast<float>(source_width) / width;
        const float sy = static_cast<float>(source_height) / height;
        const float scale = fit == VideoFrameFit::FIT ? std::max(sx, sy) : std::min(sx, sy);
        const float scale_x = fit == VideoFrameFit::STRETCH ? sx : scale;
        const float scale_y = fit == VideoFrameFit::STRETCH ? sy : scale;
        return {scale_x, scale_y, (source_width - width * scale_x) * 0.5f,
                (source_height - height * scale_y) * 0.5f};
    }
} // namespace lfs::io::video
