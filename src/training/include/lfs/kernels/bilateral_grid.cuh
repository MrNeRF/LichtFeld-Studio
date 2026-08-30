/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include <cuda_runtime.h>

namespace lfs::training::kernels {

    // HWC layout kernels

    void launch_bilateral_grid_slice_forward(
        const float* grid, const float* rgb, float* output,
        int L, int H, int W, int h, int w,
        const float* shared_offset,
        cudaStream_t stream = nullptr);

    void launch_bilateral_grid_slice_backward(
        const float* grid, const float* rgb, const float* grad_output,
        float* grad_grid, float* grad_rgb,
        int L, int H, int W, int h, int w,
        const float* shared_offset,
        cudaStream_t stream = nullptr,
        bool warp_aggregate = true);

    // CHW layout kernels (zero-copy for rasterizer output)

    void launch_bilateral_grid_slice_forward_chw(
        const float* grid, const float* rgb, float* output,
        int L, int H, int W, int h, int w,
        const float* shared_offset,
        cudaStream_t stream = nullptr);

    void launch_bilateral_grid_slice_backward_chw(
        const float* grid, const float* rgb, const float* grad_output,
        float* grad_grid, float* grad_rgb,
        int L, int H, int W, int h, int w,
        const float* shared_offset,
        cudaStream_t stream = nullptr,
        bool warp_aggregate = true);

    void launch_bilateral_grid_slice_forward_exposure_chroma(
        const float* grid, const float* rgb, float* output,
        int L, int H, int W, int h, int w,
        const float* shared_offset,
        cudaStream_t stream = nullptr);

    void launch_bilateral_grid_slice_backward_exposure_chroma(
        const float* grid, const float* rgb, const float* grad_output,
        float* grad_grid, float* grad_rgb,
        int L, int H, int W, int h, int w,
        const float* shared_offset,
        cudaStream_t stream = nullptr,
        bool warp_aggregate = true);

    void launch_bilateral_grid_slice_forward_exposure_chroma_chw(
        const float* grid, const float* rgb, float* output,
        int L, int H, int W, int h, int w,
        const float* shared_offset,
        cudaStream_t stream = nullptr);

    void launch_bilateral_grid_slice_backward_exposure_chroma_chw(
        const float* grid, const float* rgb, const float* grad_output,
        float* grad_grid, float* grad_rgb,
        int L, int H, int W, int h, int w,
        const float* shared_offset,
        cudaStream_t stream = nullptr,
        bool warp_aggregate = true);

    // Pre-privatisation scatter kernels, compiled for equivalence tests only.
    void launch_bilateral_grid_slice_backward_reference(
        const float* grid, const float* rgb, const float* grad_output,
        float* grad_grid, float* grad_rgb,
        int L, int H, int W, int h, int w,
        const float* shared_offset,
        cudaStream_t stream = nullptr);

    void launch_bilateral_grid_slice_backward_chw_reference(
        const float* grid, const float* rgb, const float* grad_output,
        float* grad_grid, float* grad_rgb,
        int L, int H, int W, int h, int w,
        const float* shared_offset,
        cudaStream_t stream = nullptr);

    void launch_bilateral_grid_slice_backward_exposure_chroma_reference(
        const float* grid, const float* rgb, const float* grad_output,
        float* grad_grid, float* grad_rgb,
        int L, int H, int W, int h, int w,
        const float* shared_offset,
        cudaStream_t stream = nullptr);

    void launch_bilateral_grid_slice_backward_exposure_chroma_chw_reference(
        const float* grid, const float* rgb, const float* grad_output,
        float* grad_grid, float* grad_rgb,
        int L, int H, int W, int h, int w,
        const float* shared_offset,
        cudaStream_t stream = nullptr);

    // TV loss kernels. norm_n is the divisor for the (C * N) normalisation;
    // pass 0 to use N (the number of images in this launch).

    void launch_bilateral_grid_tv_forward(
        const float* grids, float* tv_loss, float* temp_buffer,
        int N, int C, int L, int H, int W, int norm_n,
        cudaStream_t stream = nullptr);

    void launch_bilateral_grid_tv_backward(
        const float* grids, float grad_output, float* grad_grids,
        int N, int C, int L, int H, int W, int norm_n,
        cudaStream_t stream = nullptr);

    // Utility kernels

    void launch_bilateral_grid_init_identity(
        float* grids, int N, int L, int H, int W,
        cudaStream_t stream = nullptr);

    void launch_bilateral_grid_project_mean(
        float* grids, const float* mean, const float* identity,
        int N, int C, int L, int H, int W, int per_image,
        cudaStream_t stream = nullptr);

    void launch_bilateral_grid_update_shared_offset(
        float* channel_sum, float* shared_offset,
        const float* identity, const float* mean_old, const float* mean_new,
        int C, float spatial, float inv_n_spatial,
        cudaStream_t stream = nullptr);

    void launch_bilateral_grid_accumulate_grad(
        float* dst, const float* src, int num_elements,
        cudaStream_t stream = nullptr);

    void launch_bilateral_grid_adam_update(
        float* grid, float* exp_avg, float* exp_avg_sq, const float* grad_grid,
        int num_elements, float lr, float beta1, float beta2,
        float bias_corr1_rcp, float bias_corr2_sqrt_rcp, float eps,
        cudaStream_t stream = nullptr);

    void launch_bilateral_grid_scale_moments(
        float* exp_avg, float* exp_avg_sq, int num_elements,
        float scale_avg, float scale_avg_sq,
        cudaStream_t stream = nullptr);

} // namespace lfs::training::kernels
