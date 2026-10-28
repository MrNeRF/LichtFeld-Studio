/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include "Common.h"
#include <cuda_runtime.h>

namespace gsplat_lfs {
    struct PopScoreInputs {
        const float* means;
        const float* quats;
        const float* scales;
        const float* opacities;
        const float* colors;
        const float* background;
        const float* background_image;
        const float* viewmat;
        const float* K;
        const float* radial;
        const float* tangential;
        const float* thin_prism;
        const int32_t* offsets;
        const int32_t* ids;
        uint32_t width, height, tile_size, channels;
        CameraModelType camera_model;
    };
    void launch_pop_scores(const PopScoreInputs& inputs, double* scores, cudaStream_t stream);
} // namespace gsplat_lfs
