/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"

namespace lfs::core {

    inline constexpr int kPanoramaViewCount = 6;
    // Shared centres give overlap no parallax. A small margin gives the loss context across borders.
    inline constexpr float kPanoramaViewFovDegrees = 96.0f;

    LFS_CORE_API std::array<UndistortParams, kPanoramaViewCount> panorama_view_params(
        int width, int height);

    LFS_CORE_API std::string panorama_image_name(const std::string& view_image_name);

    LFS_CORE_API std::array<std::shared_ptr<Camera>, kPanoramaViewCount> make_panorama_views(
        const Camera& panorama, int first_uid);

} // namespace lfs::core
