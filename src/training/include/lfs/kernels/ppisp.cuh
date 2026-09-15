/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include <cuda_runtime.h>

namespace lfs::training::kernels {

    // Forward pass (CHW layout, matches rasterizer output)
    void launch_ppisp_forward_chw(const float* exposure_params, const float* vignetting_params,
                                  const float* color_params, const float* crf_params, const float* rgb_in,
                                  float* rgb_out, int height, int width, int num_cameras, int num_frames,
                                  int camera_idx, int frame_idx, cudaStream_t stream = nullptr);

    // Forward pass on a full-width row band [y_offset, y_offset + band_height) of an
    // image with full_height rows; vignetting uses full-image coordinates so banded
    // output is bit-identical to the full-image pass.
    void launch_ppisp_forward_chw_region(const float* exposure_params, const float* vignetting_params,
                                         const float* color_params, const float* crf_params, const float* rgb_in,
                                         float* rgb_out, int band_height, int width, int y_offset, int full_height,
                                         int num_cameras, int num_frames, int camera_idx, int frame_idx,
                                         cudaStream_t stream = nullptr);

    // Backward pass (CHW layout)
    void launch_ppisp_backward_chw(const float* exposure_params, const float* vignetting_params,
                                   const float* color_params, const float* crf_params, const float* rgb_in,
                                   const float* grad_rgb_out, float* grad_exposure_params,
                                   float* grad_vignetting_params, float* grad_color_params, float* grad_crf_params,
                                   float* grad_rgb_in, int height, int width, int num_cameras, int num_frames,
                                   int camera_idx, int frame_idx, cudaStream_t stream = nullptr);

    // Adam optimizer update
    void launch_ppisp_adam_update(float* params, float* exp_avg, float* exp_avg_sq, const float* grad, int num_elements,
                                  float lr, float beta1, float beta2, float bc1_rcp, float bc2_sqrt_rcp, float eps,
                                  cudaStream_t stream = nullptr);

    struct PPISPAdamGroup {
        float* params = nullptr;
        float* exp_avg = nullptr;
        float* exp_avg_sq = nullptr;
        const float* grad = nullptr;
        int n = 0;
    };

    // One launch covering the four PPISP Adam groups. A group with n==0 is skipped.
    void launch_ppisp_adam_update_batched(PPISPAdamGroup exposure, PPISPAdamGroup vignetting,
                                          PPISPAdamGroup color, PPISPAdamGroup crf, float lr, float beta1,
                                          float beta2, float bc1_rcp, float bc2_sqrt_rcp, float eps,
                                          cudaStream_t stream = nullptr);

    // Fused vignetting regulariser. Layout [num_cameras, 3, 5]. Null grad or loss
    // skips that output. Grad is accumulated (+=). Caller zeros *loss if writing it.
    void launch_ppisp_vignetting_reg(const float* vignetting_params, float* vignetting_grad, float* loss,
                                     int num_cameras, float vig_center, float vig_channel, float vig_non_pos,
                                     cudaStream_t stream = nullptr);

    // Subtract per-vector mean of exposure [N] and colour [N, 8] in one launch.
    void launch_ppisp_project_mean(float* exposure_params, float* color_params, int num_frames,
                                   cudaStream_t stream = nullptr);

    // Initialize parameters to identity
    void launch_ppisp_init_identity(float* exposure, float* vignetting, float* color, float* crf, int num_cameras,
                                    int num_frames, cudaStream_t stream = nullptr);

} // namespace lfs::training::kernels
