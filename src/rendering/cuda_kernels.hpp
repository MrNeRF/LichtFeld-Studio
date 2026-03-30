/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cuda_runtime.h>

namespace lfs {

    // Adjust saturation of Gaussians under the cursor overlay.
    // sh0: pointer to SH0 data [N, 3] (or [N, 1, 3] viewed as [N*1, 3])
    // screen_positions: [N, 2] screen positions from last render
    // cursor_x, cursor_y: cursor center in screen coords
    // cursor_radius: radius in pixels
    // saturation_delta: -1 to 1, negative = desaturate, positive = increase saturation
    // num_gaussians: number of Gaussians
    // stream: CUDA stream (0 for default)
    void launchAdjustSaturation(
        float* sh0,
        const float* screen_positions,
        float cursor_x,
        float cursor_y,
        float cursor_radius,
        float saturation_delta,
        int num_gaussians,
        cudaStream_t stream = 0);

} // namespace lfs
