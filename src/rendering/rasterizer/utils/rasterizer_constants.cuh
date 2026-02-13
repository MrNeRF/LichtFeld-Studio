/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

namespace lfs::rendering {

    constexpr float ROT_SCALE_EPS = 1e-8f;

    __device__ inline bool has_non_identity_transform(const float* __restrict__ m) {
        return m[0] != 1.0f || m[5] != 1.0f || m[10] != 1.0f ||
               m[1] != 0.0f || m[2] != 0.0f || m[3] != 0.0f ||
               m[4] != 0.0f || m[6] != 0.0f || m[7] != 0.0f ||
               m[8] != 0.0f || m[9] != 0.0f || m[11] != 0.0f;
    }

    __device__ inline bool extract_rotation_row_major(
        const float* __restrict__ m,
        float* __restrict__ rot_out) {
        const float scale_x = sqrtf(m[0] * m[0] + m[4] * m[4] + m[8] * m[8]);
        const float scale_y = sqrtf(m[1] * m[1] + m[5] * m[5] + m[9] * m[9]);
        const float scale_z = sqrtf(m[2] * m[2] + m[6] * m[6] + m[10] * m[10]);
        if (scale_x <= ROT_SCALE_EPS || scale_y <= ROT_SCALE_EPS || scale_z <= ROT_SCALE_EPS) {
            return false;
        }

        const float inv_scale_x = 1.0f / scale_x;
        const float inv_scale_y = 1.0f / scale_y;
        const float inv_scale_z = 1.0f / scale_z;
        rot_out[0] = m[0] * inv_scale_x;
        rot_out[1] = m[1] * inv_scale_y;
        rot_out[2] = m[2] * inv_scale_z;
        rot_out[3] = m[4] * inv_scale_x;
        rot_out[4] = m[5] * inv_scale_y;
        rot_out[5] = m[6] * inv_scale_z;
        rot_out[6] = m[8] * inv_scale_x;
        rot_out[7] = m[9] * inv_scale_y;
        rot_out[8] = m[10] * inv_scale_z;
        return true;
    }

} // namespace lfs::rendering
