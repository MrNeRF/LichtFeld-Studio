/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include "core/error.hpp"
#include <span>

namespace lfs::core {

    inline constexpr int kPanoramaViewCount = 6;
    // Shared centres give overlap no parallax. A small margin gives the loss context across borders.
    inline constexpr float kPanoramaViewFovDegrees = 96.0f;
    inline constexpr float kFisheyeViewThresholdDegrees = 85.0f;

    struct PinholeView {
        UndistortParams params;
        const char* suffix;
    };

    struct CameraViewGroup {
        std::shared_ptr<Camera> photograph;
        std::vector<std::shared_ptr<Camera>> drawn;
    };

    LFS_CORE_API lfs::Result<std::vector<CameraViewGroup>> group_camera_views(
        std::span<const std::shared_ptr<Camera>> drawn, std::span<const std::shared_ptr<Camera>> scene);

    LFS_CORE_API std::vector<PinholeView> plan_camera_views(const Camera& source, float lens_limit);
    LFS_CORE_API std::vector<std::shared_ptr<Camera>> make_camera_views(
        const Camera& source, std::span<const PinholeView> plan, int first_uid);
    LFS_CORE_API std::string photograph_image_name(const std::string& view_image_name);

} // namespace lfs::core
