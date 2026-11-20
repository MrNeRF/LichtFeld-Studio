/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/distortion_model.hpp"
#include "undistort.hpp"

namespace lfs::core::detail {
    inline __host__ __device__ float2 source_position(const float pixel_x, const float pixel_y,
                                                      const UndistortParams& params) {
        const float nx = (pixel_x - params.dst_cx) / params.dst_fx;
        const float ny = (pixel_y - params.dst_cy) / params.dst_fy;
        float dnx, dny;
        if (is_view(params)) {
            const auto* r = params.view_rotation;
            const float x = r[0] * nx + r[1] * ny + r[2];
            const float y = r[3] * nx + r[4] * ny + r[5];
            const float z = r[6] * nx + r[7] * ny + r[8];
            if (params.model_type == CameraModelType::EQUIRECTANGULAR) {
                dnx = atan2f(x, z);
                dny = atan2f(y, hypotf(x, z));
            } else {
                project_fisheye_ray(x, y, z, params.model_type, params.distortion,
                                    params.num_distortion, dnx, dny);
            }
        } else {
            apply_distortion(nx, ny, params.model_type, params.distortion,
                             params.num_distortion, dnx, dny);
        }
        return {dnx * params.src_fx + params.src_cx - 0.5f,
                dny * params.src_fy + params.src_cy - 0.5f};
    }

} // namespace lfs::core::detail
