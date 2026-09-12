/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cuda_runtime.h>

namespace lfs::rendering::exportpp {

    struct CompositeParams {
        float rotation[9]; // camera rotation, glm::mat3 memory order (column-major)
        int full_width = 0;
        int full_height = 0;
        int band_width = 0;
        int band_height = 0;
        int y_offset = 0;
        float focal_x = 0.0f;
        float focal_y = 0.0f;
        float center_x = 0.0f;
        float center_y = 0.0f;
        bool equirect_view = false;
        float exposure_factor = 1.0f;
        float env_rotation_radians = 0.0f;
        int env_width = 0;
        int env_height = 0;
    };

    // out u8 HWC RGB = mix(shaded_environment, rgb, alpha) with straight alpha;
    // environment directions use full-image coordinates (band at y_offset).
    cudaError_t launchCompositeEnvironmentBand(const CompositeParams& params, const float* env_pixels,
                                               const float* rgb_chw, const float* alpha, unsigned char* dst,
                                               cudaStream_t stream);

} // namespace lfs::rendering::exportpp
