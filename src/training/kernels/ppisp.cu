/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: GPL-3.0-or-later AND Apache-2.0
 *
 * Based on NVIDIA PPISP implementation.
 */

#include "core/cuda_error.hpp"
#include "lfs/kernels/ppisp.cuh"
#include "ppisp_math.cuh"
#include "ppisp_math_bwd.cuh"
#include <cassert>
#include <cub/cub.cuh>

#include "kernel_stream.hpp"

namespace lfs::training::kernels {

    namespace {
        constexpr int PPISP_BLOCK_SIZE = 256;

        inline int divUp(int a, int b) { return (a + b - 1) / b; }
    } // namespace

    // Forward kernel (CHW layout). Processes a full-width row band starting at
    // y_offset within an image of full_height rows; vignetting is evaluated in
    // full-image coordinates so banded application matches the full-image pass.
    __global__ void ppisp_forward_chw_kernel(int height, int width, int y_offset, int full_height, int num_cameras,
                                             int num_frames, const float* __restrict__ exposure_params,
                                             const VignettingChannelParams* __restrict__ vignetting_params,
                                             const ColorPPISPParams* __restrict__ color_params,
                                             const CRFPPISPChannelParams* __restrict__ crf_params,
                                             const float* __restrict__ rgb_in, float* __restrict__ rgb_out,
                                             int camera_idx, int frame_idx) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int num_pixels = height * width;
        if (idx >= num_pixels)
            return;

        int y = idx / width;
        int x = idx % width;

        // Load RGB from CHW layout
        float3 rgb =
            make_float3(rgb_in[0 * num_pixels + idx], rgb_in[1 * num_pixels + idx], rgb_in[2 * num_pixels + idx]);

        // Pixel coordinates (center of pixel, full-image space)
        float2 pixel_coord = make_float2(x + 0.5f, y_offset + y + 0.5f);

        // ISP Pipeline
        if (frame_idx != -1) {
            ppisp_apply_exposure(rgb, exposure_params[frame_idx], rgb);
        }

        if (camera_idx != -1) {
            ppisp_apply_vignetting(rgb, &vignetting_params[camera_idx * 3], pixel_coord, (float)width,
                                   (float)full_height, rgb);
        }

        if (frame_idx != -1) {
            ppisp_apply_color_correction(rgb, &color_params[frame_idx], rgb);
        }

        if (camera_idx != -1) {
            ppisp_apply_crf(rgb, &crf_params[camera_idx * 3], rgb);
        }

        // Store output (CHW layout)
        rgb_out[0 * num_pixels + idx] = rgb.x;
        rgb_out[1 * num_pixels + idx] = rgb.y;
        rgb_out[2 * num_pixels + idx] = rgb.z;
    }

    // Backward kernel (CHW layout)
    template <int BLOCK_SIZE>
    __global__ void ppisp_backward_chw_kernel(int height, int width, int num_cameras, int num_frames,
                                              const float* __restrict__ exposure_params,
                                              const VignettingChannelParams* __restrict__ vignetting_params,
                                              const ColorPPISPParams* __restrict__ color_params,
                                              const CRFPPISPChannelParams* __restrict__ crf_params,
                                              const float* __restrict__ rgb_in, const float* __restrict__ grad_rgb_out,
                                              float* __restrict__ grad_exposure_params,
                                              VignettingChannelParams* __restrict__ grad_vignetting_params,
                                              ColorPPISPParams* __restrict__ grad_color_params,
                                              CRFPPISPChannelParams* __restrict__ grad_crf_params,
                                              float* __restrict__ grad_rgb_in, int camera_idx, int frame_idx) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int num_pixels = height * width;

        // Per-thread gradient accumulators
        float grad_exposure_local = 0.0f;
        VignettingChannelParams grad_vignetting_local[3] = {{0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}};
        ColorPPISPParams grad_color_local = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
        CRFPPISPChannelParams grad_crf_local[3] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};

        if (idx < num_pixels) {
            int y = idx / width;
            int x = idx % width;

            // Load input from CHW
            float3 rgb_input = make_float3(rgb_in[0 * num_pixels + idx], rgb_in[1 * num_pixels + idx],
                                           rgb_in[2 * num_pixels + idx]);
            float2 pixel_coord = make_float2(x + 0.5f, y + 0.5f);

            // Recompute forward pass
            float3 rgb = rgb_input;
            float3 rgb_after_exp = rgb;
            float3 rgb_after_vig = rgb;
            float3 rgb_after_color = rgb;

            if (frame_idx != -1) {
                ppisp_apply_exposure(rgb, exposure_params[frame_idx], rgb_after_exp);
                rgb = rgb_after_exp;
            }

            if (camera_idx != -1) {
                ppisp_apply_vignetting(rgb, &vignetting_params[camera_idx * 3], pixel_coord, (float)width,
                                       (float)height, rgb_after_vig);
                rgb = rgb_after_vig;
            } else {
                rgb_after_vig = rgb;
            }

            if (frame_idx != -1) {
                ppisp_apply_color_correction(rgb, &color_params[frame_idx], rgb_after_color);
                rgb = rgb_after_color;
            } else {
                rgb_after_color = rgb;
            }

            // Backward pass (reverse order)
            float3 grad_rgb = make_float3(grad_rgb_out[0 * num_pixels + idx], grad_rgb_out[1 * num_pixels + idx],
                                          grad_rgb_out[2 * num_pixels + idx]);

            if (camera_idx != -1) {
                ppisp_apply_crf_bwd(rgb_after_color, &crf_params[camera_idx * 3], grad_rgb, grad_rgb, grad_crf_local);
            }

            if (frame_idx != -1) {
                ppisp_apply_color_correction_bwd(rgb_after_vig, &color_params[frame_idx], grad_rgb, grad_rgb,
                                                 &grad_color_local);
            }

            if (camera_idx != -1) {
                ppisp_apply_vignetting_bwd(rgb_after_exp, &vignetting_params[camera_idx * 3], pixel_coord, (float)width,
                                           (float)height, grad_rgb, grad_rgb, grad_vignetting_local);
            }

            if (frame_idx != -1) {
                ppisp_apply_exposure_bwd(rgb_input, exposure_params[frame_idx], grad_rgb, grad_rgb, grad_exposure_local);
            }

            // Store RGB input gradient (CHW)
            grad_rgb_in[0 * num_pixels + idx] = grad_rgb.x;
            grad_rgb_in[1 * num_pixels + idx] = grad_rgb.y;
            grad_rgb_in[2 * num_pixels + idx] = grad_rgb.z;
        }

        // Block-level reduction
        typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduceFloat;
        typedef cub::BlockReduce<float2, BLOCK_SIZE> BlockReduceFloat2;

        if (frame_idx != -1) {
            // Exposure
            {
                __shared__ typename BlockReduceFloat::TempStorage temp;
                float val = BlockReduceFloat(temp).Sum(grad_exposure_local);
                if (threadIdx.x == 0)
                    atomicAdd(&grad_exposure_params[frame_idx], val);
            }

            // Color params
            {
                __shared__ typename BlockReduceFloat2::TempStorage temp;
                ColorPPISPParams* grad_color_out = &grad_color_params[frame_idx];

                float2 val_b = BlockReduceFloat2(temp).Sum(grad_color_local.b);
                __syncthreads();
                if (threadIdx.x == 0) {
                    atomicAdd(&grad_color_out->b.x, val_b.x);
                    atomicAdd(&grad_color_out->b.y, val_b.y);
                }

                float2 val_r = BlockReduceFloat2(temp).Sum(grad_color_local.r);
                __syncthreads();
                if (threadIdx.x == 0) {
                    atomicAdd(&grad_color_out->r.x, val_r.x);
                    atomicAdd(&grad_color_out->r.y, val_r.y);
                }

                float2 val_g = BlockReduceFloat2(temp).Sum(grad_color_local.g);
                __syncthreads();
                if (threadIdx.x == 0) {
                    atomicAdd(&grad_color_out->g.x, val_g.x);
                    atomicAdd(&grad_color_out->g.y, val_g.y);
                }

                float2 val_n = BlockReduceFloat2(temp).Sum(grad_color_local.n);
                __syncthreads();
                if (threadIdx.x == 0) {
                    atomicAdd(&grad_color_out->n.x, val_n.x);
                    atomicAdd(&grad_color_out->n.y, val_n.y);
                }
            }
        }

        if (camera_idx != -1) {
            // Vignetting params
            {
                __shared__ typename BlockReduceFloat::TempStorage temp;
                VignettingChannelParams* grad_vig_out = &grad_vignetting_params[camera_idx * 3];

#pragma unroll
                for (int ch = 0; ch < 3; ch++) {
                    float val_cx = BlockReduceFloat(temp).Sum(grad_vignetting_local[ch].cx);
                    __syncthreads();
                    if (threadIdx.x == 0)
                        atomicAdd(&grad_vig_out[ch].cx, val_cx);

                    float val_cy = BlockReduceFloat(temp).Sum(grad_vignetting_local[ch].cy);
                    __syncthreads();
                    if (threadIdx.x == 0)
                        atomicAdd(&grad_vig_out[ch].cy, val_cy);

                    float val_a0 = BlockReduceFloat(temp).Sum(grad_vignetting_local[ch].alpha0);
                    __syncthreads();
                    if (threadIdx.x == 0)
                        atomicAdd(&grad_vig_out[ch].alpha0, val_a0);

                    float val_a1 = BlockReduceFloat(temp).Sum(grad_vignetting_local[ch].alpha1);
                    __syncthreads();
                    if (threadIdx.x == 0)
                        atomicAdd(&grad_vig_out[ch].alpha1, val_a1);

                    float val_a2 = BlockReduceFloat(temp).Sum(grad_vignetting_local[ch].alpha2);
                    __syncthreads();
                    if (threadIdx.x == 0)
                        atomicAdd(&grad_vig_out[ch].alpha2, val_a2);
                }
            }

            // CRF params
            {
                __shared__ typename BlockReduceFloat::TempStorage temp;
                CRFPPISPChannelParams* grad_crf_out = &grad_crf_params[camera_idx * 3];

#pragma unroll
                for (int ch = 0; ch < 3; ch++) {
                    float val_toe = BlockReduceFloat(temp).Sum(grad_crf_local[ch].toe);
                    __syncthreads();
                    if (threadIdx.x == 0)
                        atomicAdd(&grad_crf_out[ch].toe, val_toe);

                    float val_shoulder = BlockReduceFloat(temp).Sum(grad_crf_local[ch].shoulder);
                    __syncthreads();
                    if (threadIdx.x == 0)
                        atomicAdd(&grad_crf_out[ch].shoulder, val_shoulder);

                    float val_gamma = BlockReduceFloat(temp).Sum(grad_crf_local[ch].gamma);
                    __syncthreads();
                    if (threadIdx.x == 0)
                        atomicAdd(&grad_crf_out[ch].gamma, val_gamma);

                    float val_center = BlockReduceFloat(temp).Sum(grad_crf_local[ch].center);
                    __syncthreads();
                    if (threadIdx.x == 0)
                        atomicAdd(&grad_crf_out[ch].center, val_center);
                }
            }
        }
    }

    __device__ __forceinline__ void ppisp_adam_one(float* params, float* exp_avg, float* exp_avg_sq,
                                                   const float* grad, int idx, float lr, float beta1,
                                                   float beta2, float bc1_rcp, float bc2_sqrt_rcp, float eps) {
        const bool valid_param = isfinite(params[idx]);
        const float parameter = valid_param ? params[idx] : 0.0f;
        const float g = valid_param && isfinite(grad[idx]) ? grad[idx] : 0.0f;
        float m = valid_param && isfinite(exp_avg[idx]) ? exp_avg[idx] : 0.0f;
        float v = valid_param && isfinite(exp_avg_sq[idx]) && exp_avg_sq[idx] >= 0.0f
                      ? exp_avg_sq[idx]
                      : 0.0f;

        m = beta1 * m + (1.0f - beta1) * g;
        v = beta2 * v + (1.0f - beta2) * g * g;

        const float m_hat = m * bc1_rcp;
        const float v_hat_sqrt = sqrtf(v) * bc2_sqrt_rcp;
        const float updated = parameter - lr * m_hat / (v_hat_sqrt + eps);

        if (isfinite(updated) && isfinite(m) && isfinite(v)) {
            params[idx] = updated;
            exp_avg[idx] = m;
            exp_avg_sq[idx] = v;
        } else {
            params[idx] = parameter;
            exp_avg[idx] = 0.0f;
            exp_avg_sq[idx] = 0.0f;
        }
    }

    // Adam update kernel
    __global__ void ppisp_adam_update_kernel(float* params, float* exp_avg, float* exp_avg_sq, const float* grad,
                                             int num_elements, float lr, float beta1, float beta2, float bc1_rcp,
                                             float bc2_sqrt_rcp, float eps) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_elements)
            return;
        ppisp_adam_one(params, exp_avg, exp_avg_sq, grad, idx, lr, beta1, beta2, bc1_rcp, bc2_sqrt_rcp, eps);
    }

    __global__ void ppisp_adam_update_batched_kernel(PPISPAdamGroup g0, PPISPAdamGroup g1, PPISPAdamGroup g2,
                                                     PPISPAdamGroup g3, float lr, float beta1, float beta2,
                                                     float bc1_rcp, float bc2_sqrt_rcp, float eps) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int n0 = g0.n > 0 ? g0.n : 0;
        const int n1 = g1.n > 0 ? g1.n : 0;
        const int n2 = g2.n > 0 ? g2.n : 0;
        const int n3 = g3.n > 0 ? g3.n : 0;
        const int t01 = n0 + n1;
        const int t012 = t01 + n2;
        const int total = t012 + n3;
        if (idx >= total)
            return;
        if (idx < n0) {
            ppisp_adam_one(g0.params, g0.exp_avg, g0.exp_avg_sq, g0.grad, idx, lr, beta1, beta2, bc1_rcp,
                           bc2_sqrt_rcp, eps);
        } else if (idx < t01) {
            ppisp_adam_one(g1.params, g1.exp_avg, g1.exp_avg_sq, g1.grad, idx - n0, lr, beta1, beta2, bc1_rcp,
                           bc2_sqrt_rcp, eps);
        } else if (idx < t012) {
            ppisp_adam_one(g2.params, g2.exp_avg, g2.exp_avg_sq, g2.grad, idx - t01, lr, beta1, beta2, bc1_rcp,
                           bc2_sqrt_rcp, eps);
        } else {
            ppisp_adam_one(g3.params, g3.exp_avg, g3.exp_avg_sq, g3.grad, idx - t012, lr, beta1, beta2, bc1_rcp,
                           bc2_sqrt_rcp, eps);
        }
    }

    // One thread owns one camera's 15 vignetting params so grad += needs no atomics.
    __global__ void ppisp_vignetting_reg_kernel(const float* __restrict__ vig, float* __restrict__ grad,
                                                float* __restrict__ loss, int num_cameras, float w_center,
                                                float w_channel, float w_non_pos) {
        using BlockReduce = cub::BlockReduce<float, PPISP_BLOCK_SIZE>;
        __shared__ typename BlockReduce::TempStorage temp;

        float local_loss = 0.0f;
        const int cam = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        if (cam < num_cameras) {
            const float* p = vig + static_cast<size_t>(cam) * 15;
            float* g = grad ? grad + static_cast<size_t>(cam) * 15 : nullptr;
            const float inv_center = 1.0f / static_cast<float>(num_cameras * 3);
            const float inv_non_pos = 1.0f / static_cast<float>(num_cameras * 3 * 3);
            const float inv_channel = 1.0f / static_cast<float>(num_cameras * 5 * 3);

            if (w_center != 0.0f) {
                const float s = w_center * 2.0f * inv_center;
                for (int ch = 0; ch < 3; ++ch) {
                    const float cx = p[ch * 5 + 0];
                    const float cy = p[ch * 5 + 1];
                    local_loss += w_center * (cx * cx + cy * cy) * inv_center;
                    if (g) {
                        g[ch * 5 + 0] += s * cx;
                        g[ch * 5 + 1] += s * cy;
                    }
                }
            }
            if (w_non_pos != 0.0f) {
                for (int ch = 0; ch < 3; ++ch) {
                    for (int a = 0; a < 3; ++a) {
                        const float alpha = p[ch * 5 + 2 + a];
                        if (alpha > 0.0f) {
                            local_loss += w_non_pos * alpha * inv_non_pos;
                            if (g)
                                g[ch * 5 + 2 + a] += w_non_pos * inv_non_pos;
                        }
                    }
                }
            }
            if (w_channel != 0.0f) {
                const float s = w_channel * 2.0f * inv_channel;
                for (int param = 0; param < 5; ++param) {
                    float vals[3];
                    float mean = 0.0f;
                    for (int ch = 0; ch < 3; ++ch) {
                        vals[ch] = p[ch * 5 + param];
                        mean += vals[ch];
                    }
                    mean /= 3.0f;
                    for (int ch = 0; ch < 3; ++ch) {
                        const float diff = vals[ch] - mean;
                        local_loss += w_channel * diff * diff * inv_channel;
                        if (g)
                            g[ch * 5 + param] += s * diff;
                    }
                }
            }
        }

        const float block_loss = BlockReduce(temp).Sum(local_loss);
        if (loss && threadIdx.x == 0 && block_loss != 0.0f)
            atomicAdd(loss, block_loss);
    }

    __global__ void ppisp_project_mean_kernel(float* __restrict__ exposure, float* __restrict__ color, int n) {
        using BlockReduce = cub::BlockReduce<float, PPISP_BLOCK_SIZE>;
        __shared__ typename BlockReduce::TempStorage temp;
        __shared__ float smean;
        __shared__ float col_mean[8];

        if (blockIdx.x == 0) {
            float local = 0.0f;
            for (int i = static_cast<int>(threadIdx.x); i < n; i += PPISP_BLOCK_SIZE)
                local += exposure[i];
            const float sum = BlockReduce(temp).Sum(local);
            if (threadIdx.x == 0)
                smean = n > 0 ? sum / static_cast<float>(n) : 0.0f;
            __syncthreads();
            const float mean = smean;
            for (int i = static_cast<int>(threadIdx.x); i < n; i += PPISP_BLOCK_SIZE)
                exposure[i] -= mean;
        } else {
            for (int c = 0; c < 8; ++c) {
                float local = 0.0f;
                for (int f = static_cast<int>(threadIdx.x); f < n; f += PPISP_BLOCK_SIZE)
                    local += color[f * 8 + c];
                const float sum = BlockReduce(temp).Sum(local);
                if (threadIdx.x == 0)
                    col_mean[c] = n > 0 ? sum / static_cast<float>(n) : 0.0f;
                __syncthreads();
            }
            for (int i = static_cast<int>(threadIdx.x); i < n * 8; i += PPISP_BLOCK_SIZE)
                color[i] -= col_mean[i % 8];
        }
    }

    // Inverse of bounded_positive_forward: find raw value that produces target
    // bounded_positive_forward(raw, min_value) = min_value + log(1 + exp(raw))
    // Inverse: raw = log(exp(target - min_value) - 1)
    __device__ __forceinline__ float bounded_positive_inverse(float target, float min_value) {
        float delta = target - min_value;
        return __logf(__expf(delta) - 1.0f);
    }

    // Initialize parameters to zero/identity
    __global__ void ppisp_init_identity_kernel(float* exposure, float* vignetting, float* color, float* crf,
                                               int num_cameras, int num_frames) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < num_frames) {
            exposure[idx] = 0.0f;
        }

        int vig_total = num_cameras * 3 * 5;
        if (idx < vig_total) {
            vignetting[idx] = 0.0f;
        }

        int color_total = num_frames * 8;
        if (idx < color_total) {
            color[idx] = 0.0f;
        }

        // CRF: Initialize to identity curve (toe=1, shoulder=1, gamma=1, center=0.5)
        // CRF layout: [num_cameras * 3 channels * 4 params] where params are [toe, shoulder, gamma, center]
        int crf_total = num_cameras * 3 * 4;
        if (idx < crf_total) {
            int param_idx = idx % 4; // Which of the 4 CRF params
            float raw_value;
            switch (param_idx) {
            case 0: // toe: target=1.0, min_value=0.3
                raw_value = bounded_positive_inverse(1.0f, 0.3f);
                break;
            case 1: // shoulder: target=1.0, min_value=0.3
                raw_value = bounded_positive_inverse(1.0f, 0.3f);
                break;
            case 2: // gamma: target=1.0, min_value=0.1
                raw_value = bounded_positive_inverse(1.0f, 0.1f);
                break;
            default: // case 3: center: sigmoid(0) = 0.5 (identity)
                raw_value = 0.0f;
                break;
            }
            crf[idx] = raw_value;
        }
    }

    // Regularization backward kernel
    // Launch functions
    void launch_ppisp_forward_chw_region(const float* exposure_params, const float* vignetting_params,
                                         const float* color_params, const float* crf_params, const float* rgb_in,
                                         float* rgb_out, int band_height, int width, int y_offset, int full_height,
                                         int num_cameras, int num_frames, int camera_idx, int frame_idx,
                                         cudaStream_t stream) {
        assert(band_height > 0 && width > 0);
        assert(y_offset >= 0 && y_offset + band_height <= full_height);
        stream = resolve_stream(stream);
        int num_pixels = band_height * width;
        int threads = PPISP_BLOCK_SIZE;
        int blocks = divUp(num_pixels, threads);

        ppisp_forward_chw_kernel<<<blocks, threads, 0, stream>>>(
            band_height, width, y_offset, full_height, num_cameras, num_frames, exposure_params,
            reinterpret_cast<const VignettingChannelParams*>(vignetting_params),
            reinterpret_cast<const ColorPPISPParams*>(color_params),
            reinterpret_cast<const CRFPPISPChannelParams*>(crf_params), rgb_in, rgb_out, camera_idx, frame_idx);

        LFS_CUDA_CHECK_MSG(cudaGetLastError(), "PPISP forward kernel launch failed");
    }

    void launch_ppisp_forward_chw(const float* exposure_params, const float* vignetting_params,
                                  const float* color_params, const float* crf_params, const float* rgb_in,
                                  float* rgb_out, int height, int width, int num_cameras, int num_frames,
                                  int camera_idx, int frame_idx, cudaStream_t stream) {
        launch_ppisp_forward_chw_region(exposure_params, vignetting_params, color_params, crf_params, rgb_in, rgb_out,
                                        height, width, 0, height, num_cameras, num_frames, camera_idx, frame_idx,
                                        stream);
    }

    void launch_ppisp_backward_chw(const float* exposure_params, const float* vignetting_params,
                                   const float* color_params, const float* crf_params, const float* rgb_in,
                                   const float* grad_rgb_out, float* grad_exposure_params,
                                   float* grad_vignetting_params, float* grad_color_params, float* grad_crf_params,
                                   float* grad_rgb_in, int height, int width, int num_cameras, int num_frames,
                                   int camera_idx, int frame_idx, cudaStream_t stream) {
        stream = resolve_stream(stream);
        int num_pixels = height * width;
        int threads = PPISP_BLOCK_SIZE;
        int blocks = divUp(num_pixels, threads);

        ppisp_backward_chw_kernel<PPISP_BLOCK_SIZE><<<blocks, threads, 0, stream>>>(
            height, width, num_cameras, num_frames, exposure_params,
            reinterpret_cast<const VignettingChannelParams*>(vignetting_params),
            reinterpret_cast<const ColorPPISPParams*>(color_params),
            reinterpret_cast<const CRFPPISPChannelParams*>(crf_params), rgb_in, grad_rgb_out, grad_exposure_params,
            reinterpret_cast<VignettingChannelParams*>(grad_vignetting_params),
            reinterpret_cast<ColorPPISPParams*>(grad_color_params),
            reinterpret_cast<CRFPPISPChannelParams*>(grad_crf_params), grad_rgb_in, camera_idx, frame_idx);

        LFS_CUDA_CHECK_MSG(cudaGetLastError(), "PPISP backward kernel launch failed");
    }

    void launch_ppisp_adam_update(float* params, float* exp_avg, float* exp_avg_sq, const float* grad, int num_elements,
                                  float lr, float beta1, float beta2, float bc1_rcp, float bc2_sqrt_rcp, float eps,
                                  cudaStream_t stream) {
        stream = resolve_stream(stream);
        int threads = PPISP_BLOCK_SIZE;
        int blocks = divUp(num_elements, threads);

        ppisp_adam_update_kernel<<<blocks, threads, 0, stream>>>(params, exp_avg, exp_avg_sq, grad, num_elements, lr,
                                                                 beta1, beta2, bc1_rcp, bc2_sqrt_rcp, eps);

        LFS_CUDA_CHECK_MSG(cudaGetLastError(), "PPISP Adam update kernel launch failed");
    }

    void launch_ppisp_adam_update_batched(PPISPAdamGroup exposure, PPISPAdamGroup vignetting, PPISPAdamGroup color,
                                          PPISPAdamGroup crf, float lr, float beta1, float beta2, float bc1_rcp,
                                          float bc2_sqrt_rcp, float eps, cudaStream_t stream) {
        const int n0 = exposure.n > 0 ? exposure.n : 0;
        const int n1 = vignetting.n > 0 ? vignetting.n : 0;
        const int n2 = color.n > 0 ? color.n : 0;
        const int n3 = crf.n > 0 ? crf.n : 0;
        const int total = n0 + n1 + n2 + n3;
        if (total <= 0)
            return;
        stream = resolve_stream(stream);
        const int threads = PPISP_BLOCK_SIZE;
        const int blocks = divUp(total, threads);
        ppisp_adam_update_batched_kernel<<<blocks, threads, 0, stream>>>(
            exposure, vignetting, color, crf, lr, beta1, beta2, bc1_rcp, bc2_sqrt_rcp, eps);
        LFS_CUDA_CHECK_MSG(cudaGetLastError(), "PPISP batched Adam update kernel launch failed");
    }

    void launch_ppisp_vignetting_reg(const float* vignetting_params, float* vignetting_grad, float* loss,
                                     int num_cameras, float vig_center, float vig_channel, float vig_non_pos,
                                     cudaStream_t stream) {
        if (num_cameras <= 0)
            return;
        stream = resolve_stream(stream);
        const int threads = PPISP_BLOCK_SIZE;
        const int blocks = divUp(num_cameras, threads);
        ppisp_vignetting_reg_kernel<<<blocks, threads, 0, stream>>>(
            vignetting_params, vignetting_grad, loss, num_cameras, vig_center, vig_channel, vig_non_pos);
        LFS_CUDA_CHECK_MSG(cudaGetLastError(), "PPISP vignetting reg kernel launch failed");
    }

    void launch_ppisp_project_mean(float* exposure_params, float* color_params, int num_frames, cudaStream_t stream) {
        if (num_frames <= 0)
            return;
        stream = resolve_stream(stream);
        ppisp_project_mean_kernel<<<2, PPISP_BLOCK_SIZE, 0, stream>>>(exposure_params, color_params, num_frames);
        LFS_CUDA_CHECK_MSG(cudaGetLastError(), "PPISP project_mean kernel launch failed");
    }

    void launch_ppisp_init_identity(float* exposure, float* vignetting, float* color, float* crf, int num_cameras,
                                    int num_frames, cudaStream_t stream) {
        stream = resolve_stream(stream);
        int a = num_frames;
        int b = num_cameras * 3 * 5;
        int c = num_frames * 8;
        int d = num_cameras * 3 * 4;
        int max_elements = std::max(std::max(a, b), std::max(c, d));
        int threads = PPISP_BLOCK_SIZE;
        int blocks = divUp(max_elements, threads);

        ppisp_init_identity_kernel<<<blocks, threads, 0, stream>>>(exposure, vignetting, color, crf, num_cameras,
                                                                   num_frames);

        LFS_CUDA_CHECK_MSG(cudaGetLastError(), "PPISP init kernel launch failed");
    }

} // namespace lfs::training::kernels
