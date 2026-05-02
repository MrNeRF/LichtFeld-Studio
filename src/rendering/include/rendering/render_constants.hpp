/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <algorithm>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace lfs::rendering {

    enum class GaussianRasterBackend : int {
        FastGs = 0,
        Gut = 1,
        VkSplat = 2,
    };

    constexpr float DEFAULT_NEAR_PLANE = 0.1f;
    constexpr float DEFAULT_FAR_PLANE = 100000.0f;
    constexpr int MAX_VIEWPORT_SIZE = 16384;

    // 35mm full-frame sensor dimensions
    constexpr float SENSOR_WIDTH_35MM = 36.0f;
    constexpr float SENSOR_HEIGHT_35MM = 24.0f;

    constexpr float MIN_FOCAL_LENGTH_MM = 10.0f;
    constexpr float MAX_FOCAL_LENGTH_MM = 200.0f;
    constexpr float DEFAULT_FOCAL_LENGTH_MM = 35.0f;

    constexpr float DEFAULT_ORTHO_SCALE = 100.0f;

    struct CameraIntrinsics {
        float focal_x = 0.0f;
        float focal_y = 0.0f;
        float center_x = 0.0f;
        float center_y = 0.0f;
    };

    inline float focalLengthToVFovRad(const float focal_mm) {
        return 2.0f * std::atan(SENSOR_HEIGHT_35MM / (2.0f * focal_mm));
    }

    inline float focalLengthToVFov(const float focal_mm) {
        return glm::degrees(focalLengthToVFovRad(focal_mm));
    }

    inline float focalLengthToHFov(const float focal_mm) {
        return glm::degrees(2.0f * std::atan(SENSOR_WIDTH_35MM / (2.0f * focal_mm)));
    }

    inline float vFovToFocalLength(const float vfov_degrees) {
        return SENSOR_HEIGHT_35MM / (2.0f * std::tan(glm::radians(vfov_degrees) * 0.5f));
    }

    inline glm::mat4 createProjectionMatrix(const glm::ivec2& viewport_size, const float fov_degrees,
                                            const bool orthographic, const float ortho_scale,
                                            const float near_plane = DEFAULT_NEAR_PLANE,
                                            const float far_plane = DEFAULT_FAR_PLANE) {
        const float aspect = static_cast<float>(viewport_size.x) / viewport_size.y;
        if (orthographic) {
            const float half_width = viewport_size.x / (2.0f * ortho_scale);
            const float half_height = viewport_size.y / (2.0f * ortho_scale);
            return glm::ortho(-half_width, half_width, -half_height, half_height, near_plane, far_plane);
        }
        return glm::perspective(glm::radians(fov_degrees), aspect, near_plane, far_plane);
    }

    inline glm::mat4 createProjectionMatrixFromFocal(const glm::ivec2& viewport_size, const float focal_length_mm,
                                                     const bool orthographic, const float ortho_scale,
                                                     const float near_plane = DEFAULT_NEAR_PLANE,
                                                     const float far_plane = DEFAULT_FAR_PLANE) {
        const float vfov = focalLengthToVFov(focal_length_mm);
        return createProjectionMatrix(viewport_size, vfov, orthographic, ortho_scale, near_plane, far_plane);
    }

    inline glm::mat4 createProjectionMatrixFromIntrinsics(const glm::ivec2& viewport_size,
                                                          const CameraIntrinsics& intrinsics,
                                                          const float near_plane = DEFAULT_NEAR_PLANE,
                                                          const float far_plane = DEFAULT_FAR_PLANE) {
        // This produces an image-space frustum with window-space Y increasing downward.
        const float width = static_cast<float>(viewport_size.x);
        const float height = static_cast<float>(viewport_size.y);
        const float fx = std::max(intrinsics.focal_x, 1e-6f);
        const float fy = std::max(intrinsics.focal_y, 1e-6f);

        const float left = -intrinsics.center_x * near_plane / fx;
        const float right = (width - intrinsics.center_x) * near_plane / fx;
        const float top = (height - intrinsics.center_y) * near_plane / fy;
        const float bottom = -intrinsics.center_y * near_plane / fy;
        return glm::frustum(left, right, bottom, top, near_plane, far_plane);
    }

} // namespace lfs::rendering
