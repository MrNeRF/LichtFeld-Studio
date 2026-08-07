/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>

namespace lfs::core {

    namespace detail {
        constexpr float kCubeFacePi = 3.14159265358979323846f;
        // Clamped before any focal/size conversion so a degenerate field of view
        // cannot produce a zero or infinite focal length.
        constexpr float kMinCubeFaceFovDegrees = 1.0f;
        constexpr float kMaxCubeFaceFovDegrees = 179.0f;

        [[nodiscard]] inline float cube_face_half_fov_tangent(const float fov_degrees) {
            const float fov = std::clamp(fov_degrees, kMinCubeFaceFovDegrees, kMaxCubeFaceFovDegrees);
            return std::tan(0.5f * fov * kCubeFacePi / 180.0f);
        }
    } // namespace detail

    // Sampling of one square perspective face out of an equirectangular panorama.
    // pano_to_face holds the face basis as three rows: right, up, forward.
    struct CubeFaceProjection {
        std::array<float, 9> pano_to_face{};
        int face_size = 0;
        int source_width = 0;
        int source_height = 0;
        float fov_degrees = 90.0f;
    };

    // Pinhole focal length, in pixels, of a square cube face.
    [[nodiscard]] inline float cube_face_focal(const float fov_degrees, const int face_size) {
        return 0.5f * static_cast<float>(face_size) / detail::cube_face_half_fov_tangent(fov_degrees);
    }

    // Face size that matches the panorama's own angular sampling rate at the centre
    // of the face.
    //
    // The source resolves width/(2*pi) px/rad in azimuth and height/pi px/rad in
    // elevation. A gnomonic face with focal f resolves exactly f px/rad at its
    // centre, and more towards its edges because of the cos^3 stretch, so matching
    // the centre means f = source focal and size = 2 * f * tan(fov/2).
    //
    // Sizing by pixel count instead (width * fov / 360) undersamples the face centre
    // by tan(fov/2) / (fov/2) -- 1.27x at 90 degrees -- because a perspective face
    // spaces its columns non-uniformly while equirectangular spacing is uniform in
    // angle.
    [[nodiscard]] inline int cube_face_size_for_panorama(const int source_width,
                                                         const int source_height,
                                                         const float fov_degrees) {
        const float source_focal = std::max(static_cast<float>(source_width) / (2.0f * detail::kCubeFacePi),
                                            static_cast<float>(source_height) / detail::kCubeFacePi);
        const float size = 2.0f * source_focal * detail::cube_face_half_fov_tangent(fov_degrees);
        return std::max(1, static_cast<int>(std::lround(size)));
    }

} // namespace lfs::core
