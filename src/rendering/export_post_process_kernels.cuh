/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cuda_runtime.h>

namespace lfs::rendering::exportpp {

    // u8 HWC [num_pixels, src_channels] -> float CHW rgb planes (*1/255); alpha
    // (from channel 3) written only when non-null.
    cudaError_t launchUnpackU8Hwc(const unsigned char* src, int num_pixels, int src_channels,
                                  float* rgb_chw, float* alpha, cudaStream_t stream);

    // float CHW rgb planes (+ optional alpha) -> u8 HWC, clamped and rounded.
    cudaError_t launchPackChwU8Hwc(const float* rgb_chw, const float* alpha, unsigned char* dst,
                                   int num_pixels, int dst_channels, cudaStream_t stream);

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
