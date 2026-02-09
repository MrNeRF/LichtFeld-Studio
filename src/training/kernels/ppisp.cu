/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: GPL-3.0-or-later AND Apache-2.0
 *
 * Based on NVIDIA PPISP implementation.
 */

#include "lfs/kernels/ppisp.cuh"
#include "ppisp_math.cuh"
#include "ppisp_math_bwd.cuh"
#include <cassert>
#include <cub/cub.cuh>

namespace lfs::training::kernels {

    namespace {
        constexpr int PPISP_BLOCK_SIZE = 256;

        inline int divUp(int a, int b) { return (a + b - 1) / b; }
    } // namespace

    // Forward kernel (CHW layout)
    __global__ void ppisp_forward_chw_kernel(int height, int width, int num_cameras, int num_frames,
                                             const float* __restrict__ exposure_params,
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

        // Pixel coordinates (center of pixel)
        float2 pixel_coord = make_float2(x + 0.5f, y + 0.5f);

        // ISP Pipeline
        if (frame_idx != -1) {
            ppisp_apply_exposure(rgb, exposure_params[frame_idx], rgb);
        }

        if (camera_idx != -1) {
            ppisp_apply_vignetting(rgb, &vignetting_params[camera_idx * 3], pixel_coord, (float)width, (float)height,
                                   rgb);
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

    // Adam update kernel
    __global__ void ppisp_adam_update_kernel(float* params, float* exp_avg, float* exp_avg_sq, const float* grad,
                                             int num_elements, float lr, float beta1, float beta2, float bc1_rcp,
                                             float bc2_sqrt_rcp, float eps) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_elements)
            return;

        float g = grad[idx];
        float m = exp_avg[idx];
        float v = exp_avg_sq[idx];

        m = beta1 * m + (1.0f - beta1) * g;
        v = beta2 * v + (1.0f - beta2) * g * g;

        exp_avg[idx] = m;
        exp_avg_sq[idx] = v;

        float m_hat = m * bc1_rcp;
        float v_hat_sqrt = sqrtf(v) * bc2_sqrt_rcp;

        params[idx] -= lr * m_hat / (v_hat_sqrt + eps);
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
            int param_idx = idx % 4;  // Which of the 4 CRF params
            float raw_value;
            switch (param_idx) {
                case 0:  // toe: target=1.0, min_value=0.3
                    raw_value = bounded_positive_inverse(1.0f, 0.3f);
                    break;
                case 1:  // shoulder: target=1.0, min_value=0.3
                    raw_value = bounded_positive_inverse(1.0f, 0.3f);
                    break;
                case 2:  // gamma: target=1.0, min_value=0.1
                    raw_value = bounded_positive_inverse(1.0f, 0.1f);
                    break;
                default:  // case 3: center: sigmoid(0) = 0.5 (identity)
                    raw_value = 0.0f;
                    break;
            }
            crf[idx] = raw_value;
        }
    }

    __global__ void ppisp_reg_loss_kernel(const float* params, float* loss_out, int num_elements) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        float local_sum = 0.0f;

        if (idx < num_elements) {
            float val = params[idx];
            local_sum = val * val;
        }

        typedef cub::BlockReduce<float, PPISP_BLOCK_SIZE> BlockReduce;
        __shared__ typename BlockReduce::TempStorage temp;
        float sum = BlockReduce(temp).Sum(local_sum);

        if (threadIdx.x == 0) {
            atomicAdd(loss_out, sum);
        }
    }

    // Regularization backward kernel
    __global__ void ppisp_reg_backward_kernel(const float* params, float* grad, float weight, int num_elements) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_elements)
            return;

        grad[idx] += 2.0f * weight * params[idx];
    }

    // Launch functions
    void launch_ppisp_forward_chw(const float* exposure_params, const float* vignetting_params,
                                  const float* color_params, const float* crf_params, const float* rgb_in,
                                  float* rgb_out, int height, int width, int num_cameras, int num_frames,
                                  int camera_idx, int frame_idx, cudaStream_t stream) {
        int num_pixels = height * width;
        int threads = PPISP_BLOCK_SIZE;
        int blocks = divUp(num_pixels, threads);

        ppisp_forward_chw_kernel<<<blocks, threads, 0, stream>>>(
            height, width, num_cameras, num_frames, exposure_params,
            reinterpret_cast<const VignettingChannelParams*>(vignetting_params),
            reinterpret_cast<const ColorPPISPParams*>(color_params),
            reinterpret_cast<const CRFPPISPChannelParams*>(crf_params), rgb_in, rgb_out, camera_idx, frame_idx);

        cudaError_t err = cudaGetLastError();
        assert(err == cudaSuccess && "PPISP forward kernel launch failed");
    }

    void launch_ppisp_backward_chw(const float* exposure_params, const float* vignetting_params,
                                   const float* color_params, const float* crf_params, const float* rgb_in,
                                   const float* grad_rgb_out, float* grad_exposure_params,
                                   float* grad_vignetting_params, float* grad_color_params, float* grad_crf_params,
                                   float* grad_rgb_in, int height, int width, int num_cameras, int num_frames,
                                   int camera_idx, int frame_idx, cudaStream_t stream) {
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

        cudaError_t err = cudaGetLastError();
        assert(err == cudaSuccess && "PPISP backward kernel launch failed");
    }

    void launch_ppisp_adam_update(float* params, float* exp_avg, float* exp_avg_sq, const float* grad, int num_elements,
                                  float lr, float beta1, float beta2, float bc1_rcp, float bc2_sqrt_rcp, float eps,
                                  cudaStream_t stream) {
        int threads = PPISP_BLOCK_SIZE;
        int blocks = divUp(num_elements, threads);

        ppisp_adam_update_kernel<<<blocks, threads, 0, stream>>>(params, exp_avg, exp_avg_sq, grad, num_elements, lr,
                                                                 beta1, beta2, bc1_rcp, bc2_sqrt_rcp, eps);

        cudaError_t err = cudaGetLastError();
        assert(err == cudaSuccess && "PPISP Adam update kernel launch failed");
    }

    void launch_ppisp_init_identity(float* exposure, float* vignetting, float* color, float* crf, int num_cameras,
                                    int num_frames, cudaStream_t stream) {
        int a = num_frames;
        int b = num_cameras * 3 * 5;
        int c = num_frames * 8;
        int d = num_cameras * 3 * 4;
        int max_elements = std::max(std::max(a, b), std::max(c, d));
        int threads = PPISP_BLOCK_SIZE;
        int blocks = divUp(max_elements, threads);

        ppisp_init_identity_kernel<<<blocks, threads, 0, stream>>>(exposure, vignetting, color, crf, num_cameras,
                                                                   num_frames);

        cudaError_t err = cudaGetLastError();
        assert(err == cudaSuccess && "PPISP init kernel launch failed");
    }

    void launch_ppisp_reg_loss(const float* params, float* loss_out, int num_elements, cudaStream_t stream) {
        int threads = PPISP_BLOCK_SIZE;
        int blocks = divUp(num_elements, threads);

        ppisp_reg_loss_kernel<<<blocks, threads, 0, stream>>>(params, loss_out, num_elements);

        cudaError_t err = cudaGetLastError();
        assert(err == cudaSuccess && "PPISP reg loss kernel launch failed");
    }

    void launch_ppisp_reg_backward(const float* params, float* grad, float weight, int num_elements,
                                   cudaStream_t stream) {
        int threads = PPISP_BLOCK_SIZE;
        int blocks = divUp(num_elements, threads);

        ppisp_reg_backward_kernel<<<blocks, threads, 0, stream>>>(params, grad, weight, num_elements);

        cudaError_t err = cudaGetLastError();
        assert(err == cudaSuccess && "PPISP reg backward kernel launch failed");
    }

} // namespace lfs::training::kernels
