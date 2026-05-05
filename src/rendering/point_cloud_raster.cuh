/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace lfs::rendering::pcraster {

    struct CropBox {
        float to_local[16];
        float min[3];
        float max[3];
        bool inverse;
        bool desaturate;
    };

    struct LaunchParams {
        const float* positions;
        const float* colors;
        const float* transforms; // [n_transforms, 16] column-major or null
        const std::int32_t* transform_indices;
        const std::uint8_t* visibility_mask;
        std::size_t n_points;
        int n_transforms;
        int n_visibility;
        bool has_crop;
        CropBox crop;
        float view[16];
        float view_proj[16];
        int width;
        int height;
        int channels;
        bool equirectangular;
        bool orthographic;
        float ortho_scale;
        float focal_y;
        float voxel_size;
        float scaling_modifier;
        float far_plane;
        float bg_r;
        float bg_g;
        float bg_b;
        float bg_a;
        bool transparent_background;
        float* image;
        float* depth;
        cudaStream_t stream;
    };

    // Returns cudaSuccess on success. The image and depth pointers must point to
    // device memory of size [channels, height, width] and [1, height, width]
    // respectively.
    cudaError_t launchPointCloudRaster(const LaunchParams& params);

} // namespace lfs::rendering::pcraster
