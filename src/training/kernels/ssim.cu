/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda_error.hpp"
#include "core/cuda_safe_format.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "lfs/kernels/loss_tensor_contract.hpp"
#include "lfs/kernels/ssim.cuh"
#include "lfs/kernels/ssim_reduction.cuh"
#include <algorithm>
#include <cmath>
#include <cooperative_groups.h>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <type_traits>

namespace cg = cooperative_groups;

namespace {
    // ------------------------------------------
    // Constant Memory for Gaussian Coefficients
    // ------------------------------------------
    __constant__ float cGauss[11] = {
        0.001028380123898387f,
        0.0075987582094967365f,
        0.036000773310661316f,
        0.10936068743467331f,
        0.21300552785396576f,
        0.26601171493530273f,
        0.21300552785396576f,
        0.10936068743467331f,
        0.036000773310661316f,
        0.0075987582094967365f,
        0.001028380123898387f};

// ------------------------------------------
// Block and Shared Memory Dimensions
// ------------------------------------------
#define BLOCK_X 16
#define BLOCK_Y 16
#define HALO    5

#define SHARED_X (BLOCK_X + 2 * HALO)
#define SHARED_Y (BLOCK_Y + 2 * HALO)

// For partial results after horizontal pass
#define CONV_X BLOCK_X
#define CONV_Y SHARED_Y

    constexpr float UINT8_TO_FLOAT = 1.0f / 255.0f;

    template <typename T>
    __device__ __forceinline__ float pixel_value(const T* data, const int index) {
        if constexpr (std::is_same_v<std::remove_cv_t<T>, uint8_t>) {
            return static_cast<float>(data[index]) * UINT8_TO_FLOAT;
        } else {
            return static_cast<float>(data[index]);
        }
    }

    template <typename T>
    __device__ __forceinline__ float mask_value(const T* data, const int index) {
        if constexpr (std::is_same_v<std::remove_cv_t<T>, uint8_t>) {
            return data[index] != 0 ? 1.0f : 0.0f;
        } else {
            return static_cast<float>(data[index]);
        }
    }

    // ------------------------------------------
    // Utility: Safe pixel fetch w/ zero padding
    // ------------------------------------------
    template <typename T>
    __device__ __forceinline__ float get_pix_value(
        const T* img,
        int b, int c, int y, int x,
        int CH, int H, int W) {
        if (x < 0 || x >= W || y < 0 || y >= H) {
            return 0.0f;
        }
        return pixel_value(img, b * CH * H * W + c * H * W + y * W + x);
    }

    // ------------------------------------------
    // Forward Kernel: Fused SSIM
    //  - Two-pass convolution to get mu1, mu2,
    //    sigma1_sq, sigma2_sq, sigma12, etc.
    //  - Writes final SSIM map to ssim_map
    //  - Optionally writes partial derivatives
    //    to dm_dmu1, dm_dsigma1_sq, dm_dsigma12
    // ------------------------------------------
    template <typename TargetT>
    __global__ void fusedssimCUDA(
        int H,
        int W,
        int CH,
        float C1,
        float C2,
        const float* __restrict__ img1,
        const TargetT* __restrict__ img2,
        float* __restrict__ ssim_map,
        float* __restrict__ cs_map,
        __half* __restrict__ dm_dmu1,
        __half* __restrict__ dm_dsigma1_sq,
        __half* __restrict__ dm_dsigma12) {
        auto block = cg::this_thread_block();
        const int bIdx = block.group_index().z; // batch index
        const int pix_y = block.group_index().y * BLOCK_Y + block.thread_index().y;
        const int pix_x = block.group_index().x * BLOCK_X + block.thread_index().x;
        const int pix_id = pix_y * W + pix_x;
        const int num_pix = H * W;

        // Shared memory for the tile (img1, img2)
        __shared__ float sTile[SHARED_Y][SHARED_X][2];
        // After horizontal pass, store partial sums here
        // xconv[y][x] -> (sumX, sumX^2, sumY, sumY^2, sumXY)
        __shared__ float xconv[CONV_Y][CONV_X][5];

        // Each block processes B x C sub-batches. We loop over channels:
        for (int c = 0; c < CH; ++c) {
            // ------------------------------------------------------------
            // 1) Load (img1, img2) tile + halo into shared memory
            // ------------------------------------------------------------
            {
                const int tileSize = SHARED_Y * SHARED_X;
                const int threads = BLOCK_X * BLOCK_Y;
                const int steps = (tileSize + threads - 1) / threads;

                const int tileStartY = block.group_index().y * BLOCK_Y;
                const int tileStartX = block.group_index().x * BLOCK_X;

                for (int s = 0; s < steps; ++s) {
                    int tid = s * threads + block.thread_rank();
                    if (tid < tileSize) {
                        int local_y = tid / SHARED_X;
                        int local_x = tid % SHARED_X;
                        int gy = tileStartY + local_y - HALO;
                        int gx = tileStartX + local_x - HALO;

                        float X = get_pix_value(img1, bIdx, c, gy, gx, CH, H, W);
                        float Y = get_pix_value(img2, bIdx, c, gy, gx, CH, H, W);

                        sTile[local_y][local_x][0] = X;
                        sTile[local_y][local_x][1] = Y;
                    }
                }
            }
            block.sync();

            // ------------------------------------------------------------
            // 2) Horizontal convolution (11x1) in shared memory
            //    We'll accumulate symmetrical pairs around center.
            // ------------------------------------------------------------
            {
                int ly = threadIdx.y;
                int lx = threadIdx.x + HALO; // skip left halo

                float sumX = 0.f;
                float sumX2 = 0.f;
                float sumY = 0.f;
                float sumY2 = 0.f;
                float sumXY = 0.f;

                // #pragma unroll for those 5 pairs
#pragma unroll
                for (int d = 1; d <= HALO; ++d) {
                    float w = cGauss[HALO - d];
                    float Xleft = sTile[ly][lx - d][0];
                    float Yleft = sTile[ly][lx - d][1];
                    float Xright = sTile[ly][lx + d][0];
                    float Yright = sTile[ly][lx + d][1];

                    sumX += (Xleft + Xright) * w;
                    sumX2 += ((Xleft * Xleft) + (Xright * Xright)) * w;
                    sumY += (Yleft + Yright) * w;
                    sumY2 += ((Yleft * Yleft) + (Yright * Yright)) * w;
                    sumXY += ((Xleft * Yleft) + (Xright * Yright)) * w;
                }
                // center
                {
                    float centerX = sTile[ly][lx][0];
                    float centerY = sTile[ly][lx][1];
                    float wc = cGauss[HALO];
                    sumX += centerX * wc;
                    sumX2 += (centerX * centerX) * wc;
                    sumY += centerY * wc;
                    sumY2 += (centerY * centerY) * wc;
                    sumXY += (centerX * centerY) * wc;
                }

                // Write out partial sums
                xconv[ly][threadIdx.x][0] = sumX;
                xconv[ly][threadIdx.x][1] = sumX2;
                xconv[ly][threadIdx.x][2] = sumY;
                xconv[ly][threadIdx.x][3] = sumY2;
                xconv[ly][threadIdx.x][4] = sumXY;

                // Possibly handle second row in same warp
                int ly2 = ly + BLOCK_Y;
                if (ly2 < CONV_Y) {
                    sumX = 0.f;
                    sumX2 = 0.f;
                    sumY = 0.f;
                    sumY2 = 0.f;
                    sumXY = 0.f;

#pragma unroll
                    for (int d = 1; d <= HALO; ++d) {
                        float w = cGauss[HALO - d];
                        float Xleft = sTile[ly2][lx - d][0];
                        float Yleft = sTile[ly2][lx - d][1];
                        float Xright = sTile[ly2][lx + d][0];
                        float Yright = sTile[ly2][lx + d][1];

                        sumX += (Xleft + Xright) * w;
                        sumX2 += ((Xleft * Xleft) + (Xright * Xright)) * w;
                        sumY += (Yleft + Yright) * w;
                        sumY2 += ((Yleft * Yleft) + (Yright * Yright)) * w;
                        sumXY += ((Xleft * Yleft) + (Xright * Yright)) * w;
                    }
                    // center
                    {
                        float cx = sTile[ly2][lx][0];
                        float cy = sTile[ly2][lx][1];
                        float wc = cGauss[HALO];
                        sumX += cx * wc;
                        sumX2 += (cx * cx) * wc;
                        sumY += cy * wc;
                        sumY2 += (cy * cy) * wc;
                        sumXY += (cx * cy) * wc;
                    }
                    xconv[ly2][threadIdx.x][0] = sumX;
                    xconv[ly2][threadIdx.x][1] = sumX2;
                    xconv[ly2][threadIdx.x][2] = sumY;
                    xconv[ly2][threadIdx.x][3] = sumY2;
                    xconv[ly2][threadIdx.x][4] = sumXY;
                }
            }
            block.sync();

            // ------------------------------------------------------------
            // 3) Vertical convolution (1x11) + final SSIM
            // ------------------------------------------------------------
            {
                int ly = threadIdx.y + HALO;
                int lx = threadIdx.x;

                float out0 = 0.f, out1 = 0.f, out2 = 0.f, out3 = 0.f, out4 = 0.f;

#pragma unroll
                for (int d = 1; d <= HALO; ++d) {
                    float w = cGauss[HALO - d];
                    float* top = xconv[ly - d][lx];
                    float* bot = xconv[ly + d][lx];

                    out0 += (top[0] + bot[0]) * w;
                    out1 += (top[1] + bot[1]) * w;
                    out2 += (top[2] + bot[2]) * w;
                    out3 += (top[3] + bot[3]) * w;
                    out4 += (top[4] + bot[4]) * w;
                }
                // center
                {
                    float wC = cGauss[HALO];
                    float* ctr = xconv[ly][lx];
                    out0 += ctr[0] * wC;
                    out1 += ctr[1] * wC;
                    out2 += ctr[2] * wC;
                    out3 += ctr[3] * wC;
                    out4 += ctr[4] * wC;
                }

                if (pix_x < W && pix_y < H) {
                    float mu1 = out0;
                    float mu2 = out2;
                    float mu1_sq = mu1 * mu1;
                    float mu2_sq = mu2 * mu2;

                    float sigma1_sq = out1 - mu1_sq;
                    float sigma2_sq = out3 - mu2_sq;
                    float sigma12 = out4 - mu1 * mu2;

                    float A = mu1_sq + mu2_sq + C1;
                    float B = sigma1_sq + sigma2_sq + C2;
                    float C_ = 2.f * mu1 * mu2 + C1;
                    float D_ = 2.f * sigma12 + C2;

                    float val = (C_ * D_) / (A * B);

                    int global_idx = bIdx * CH * num_pix + c * num_pix + pix_id;
                    ssim_map[global_idx] = val;
                    if (cs_map) {
                        cs_map[global_idx] = D_ / B;
                    }

                    if (dm_dmu1) {
                        // partial derivatives
                        float d_m_dmu1 = ((mu2 * 2.f * D_) / (A * B) - (mu2 * 2.f * C_) / (A * B) - (mu1 * 2.f * C_ * D_) / (A * A * B) + (mu1 * 2.f * C_ * D_) / (A * B * B));
                        float d_m_dsigma1_sq = (-C_ * D_) / (A * B * B);
                        float d_m_dsigma12 = (2.f * C_) / (A * B);

                        dm_dmu1[global_idx] = static_cast<__half>(d_m_dmu1);
                        dm_dsigma1_sq[global_idx] = static_cast<__half>(d_m_dsigma1_sq);
                        dm_dsigma12[global_idx] = static_cast<__half>(d_m_dsigma12);
                    }
                }
            }
        }
    }

    // ------------------------------------------
    // Backward Kernel: Apply chain rule to get
    //    dL/d(img1) from partial derivatives
    //    (dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
    //    and dL/dmap (the gradient from above).
    // ------------------------------------------
    template <typename TargetT>
    __global__ void fusedssim_backwardCUDA(
        int H,
        int W,
        int CH,
        float C1,
        float C2,
        const float* __restrict__ img1,
        const TargetT* __restrict__ img2,
        const float* __restrict__ dL_dmap,
        float* __restrict__ dL_dimg1,
        const __half* __restrict__ dm_dmu1,
        const __half* __restrict__ dm_dsigma1_sq,
        const __half* __restrict__ dm_dsigma12) {
        auto block = cg::this_thread_block();

        const int pix_y = block.group_index().y * BLOCK_Y + block.thread_index().y;
        const int pix_x = block.group_index().x * BLOCK_X + block.thread_index().x;
        const int pix_id = pix_y * W + pix_x;
        const int num_pix = H * W;
        const int bIdx = block.group_index().z;

        // Shared memory for the fused data:
        // [0]: dm_dmu1*dL, [1]: dm_dsigma1_sq*dL, [2]: dm_dsigma12*dL
        __shared__ float sData[3][SHARED_Y][SHARED_X];
        __shared__ float sScratch[CONV_Y][CONV_X][3];

        for (int c = 0; c < CH; ++c) {
            float p1 = 0.f, p2 = 0.f;
            if (pix_x < W && pix_y < H) {
                p1 = get_pix_value(img1, bIdx, c, pix_y, pix_x, CH, H, W);
                p2 = get_pix_value(img2, bIdx, c, pix_y, pix_x, CH, H, W);
            }

            // (1) Load + fuse multiplication
            {
                const int start_y = block.group_index().y * BLOCK_Y;
                const int start_x = block.group_index().x * BLOCK_X;

                int tid = threadIdx.y * blockDim.x + threadIdx.x;
                int warp_id = tid / 32;
                int lane_id = tid % 32;
                int totalThreads = BLOCK_X * BLOCK_Y;
                int num_warps = (totalThreads + 31) / 32;

                for (int row = warp_id; row < SHARED_Y; row += num_warps) {
                    int gy = start_y + row - HALO;
                    for (int col = lane_id; col < SHARED_X; col += 32) {
                        int gx = start_x + col - HALO;

                        float chain = get_pix_value(dL_dmap, bIdx, c, gy, gx, CH, H, W);
                        float vmu = get_pix_value(dm_dmu1, bIdx, c, gy, gx, CH, H, W);
                        float vs1 = get_pix_value(dm_dsigma1_sq, bIdx, c, gy, gx, CH, H, W);
                        float vs12 = get_pix_value(dm_dsigma12, bIdx, c, gy, gx, CH, H, W);

                        sData[0][row][col] = vmu * chain;
                        sData[1][row][col] = vs1 * chain;
                        sData[2][row][col] = vs12 * chain;
                    }
                }
            }
            block.sync();

            // (2) Horizontal pass
            {
                int ly = threadIdx.y;
                int lx = threadIdx.x + HALO;

                for (int pass = 0; pass < 2; ++pass) {
                    int yy = ly + pass * BLOCK_Y;
                    if (yy < CONV_Y) {
                        float accum0 = 0.f, accum1 = 0.f, accum2 = 0.f;

#pragma unroll
                        for (int d = 1; d <= HALO; ++d) {
                            float w = cGauss[HALO - d];
                            float left0 = sData[0][yy][lx - d];
                            float left1 = sData[1][yy][lx - d];
                            float left2 = sData[2][yy][lx - d];

                            float right0 = sData[0][yy][lx + d];
                            float right1 = sData[1][yy][lx + d];
                            float right2 = sData[2][yy][lx + d];

                            accum0 += (left0 + right0) * w;
                            accum1 += (left1 + right1) * w;
                            accum2 += (left2 + right2) * w;
                        }
                        // center
                        {
                            float wc = cGauss[HALO];
                            float c0 = sData[0][yy][lx];
                            float c1 = sData[1][yy][lx];
                            float c2 = sData[2][yy][lx];
                            accum0 += c0 * wc;
                            accum1 += c1 * wc;
                            accum2 += c2 * wc;
                        }

                        sScratch[yy][threadIdx.x][0] = accum0;
                        sScratch[yy][threadIdx.x][1] = accum1;
                        sScratch[yy][threadIdx.x][2] = accum2;
                    }
                }
            }
            block.sync();

            // (3) Vertical pass -> finalize dL/d(img1)
            if (pix_x < W && pix_y < H) {
                int ly = threadIdx.y + HALO;
                int lx = threadIdx.x;

                float sum0 = 0.f, sum1 = 0.f, sum2 = 0.f;

#pragma unroll
                for (int d = 1; d <= HALO; ++d) {
                    float w = cGauss[HALO - d];
                    float* top = sScratch[ly - d][lx];
                    float* bot = sScratch[ly + d][lx];

                    sum0 += (top[0] + bot[0]) * w;
                    sum1 += (top[1] + bot[1]) * w;
                    sum2 += (top[2] + bot[2]) * w;
                }
                // center
                {
                    float wc = cGauss[HALO];
                    float* ctr = sScratch[ly][lx];
                    sum0 += ctr[0] * wc;
                    sum1 += ctr[1] * wc;
                    sum2 += ctr[2] * wc;
                }

                // final accumulation
                float dL_dpix = sum0 + (2.f * p1) * sum1 + (p2)*sum2;

                int out_idx = bIdx * CH * num_pix + c * num_pix + pix_id;
                dL_dimg1[out_idx] = dL_dpix;
            }
            block.sync();
        }
    }

    // Fused L1+SSIM Forward Kernel
    template <typename TargetT>
    __global__ void fusedL1SSIMForwardCUDA(
        int H,
        int W,
        int CH,
        float C1,
        float C2,
        const float* __restrict__ img1,
        const TargetT* __restrict__ img2,
        __half* __restrict__ dm_dmu1,
        __half* __restrict__ dm_dsigma1_sq,
        __half* __restrict__ dm_dsigma12,
        float* __restrict__ ssim_map,
        float* __restrict__ cs_map,
        float* __restrict__ ssim_channels) { // optional [N, C, H, W] per-channel SSIM

        auto block = cg::this_thread_block();
        const int bIdx = block.group_index().z;
        const int pix_y = block.group_index().y * BLOCK_Y + block.thread_index().y;
        const int pix_x = block.group_index().x * BLOCK_X + block.thread_index().x;
        const int pix_id = pix_y * W + pix_x;
        const int num_pix = H * W;

        __shared__ float sTile[SHARED_Y][SHARED_X][2];
        __shared__ float xconv[CONV_Y][CONV_X][5];

        float ssim_sum = 0.0f;
        float cs_sum = 0.0f;
        for (int c = 0; c < CH; ++c) {
            // 1) Load tile + halo into shared memory
            {
                const int tileSize = SHARED_Y * SHARED_X;
                const int threads = BLOCK_X * BLOCK_Y;
                const int steps = (tileSize + threads - 1) / threads;
                const int tileStartY = block.group_index().y * BLOCK_Y;
                const int tileStartX = block.group_index().x * BLOCK_X;

                for (int s = 0; s < steps; ++s) {
                    int tid = s * threads + block.thread_rank();
                    if (tid < tileSize) {
                        int local_y = tid / SHARED_X;
                        int local_x = tid % SHARED_X;
                        int gy = tileStartY + local_y - HALO;
                        int gx = tileStartX + local_x - HALO;

                        float X = get_pix_value(img1, bIdx, c, gy, gx, CH, H, W);
                        float Y = get_pix_value(img2, bIdx, c, gy, gx, CH, H, W);

                        sTile[local_y][local_x][0] = X;
                        sTile[local_y][local_x][1] = Y;
                    }
                }
            }
            block.sync();

            // 2) Horizontal convolution
            {
                int ly = threadIdx.y;
                int lx = threadIdx.x + HALO;

                float sumX = 0.f, sumX2 = 0.f, sumY = 0.f, sumY2 = 0.f, sumXY = 0.f;

#pragma unroll
                for (int d = 1; d <= HALO; ++d) {
                    float w = cGauss[HALO - d];
                    float Xleft = sTile[ly][lx - d][0];
                    float Yleft = sTile[ly][lx - d][1];
                    float Xright = sTile[ly][lx + d][0];
                    float Yright = sTile[ly][lx + d][1];

                    sumX += (Xleft + Xright) * w;
                    sumX2 += ((Xleft * Xleft) + (Xright * Xright)) * w;
                    sumY += (Yleft + Yright) * w;
                    sumY2 += ((Yleft * Yleft) + (Yright * Yright)) * w;
                    sumXY += ((Xleft * Yleft) + (Xright * Yright)) * w;
                }
                // center
                {
                    float centerX = sTile[ly][lx][0];
                    float centerY = sTile[ly][lx][1];
                    float wc = cGauss[HALO];
                    sumX += centerX * wc;
                    sumX2 += (centerX * centerX) * wc;
                    sumY += centerY * wc;
                    sumY2 += (centerY * centerY) * wc;
                    sumXY += (centerX * centerY) * wc;
                }

                xconv[ly][threadIdx.x][0] = sumX;
                xconv[ly][threadIdx.x][1] = sumX2;
                xconv[ly][threadIdx.x][2] = sumY;
                xconv[ly][threadIdx.x][3] = sumY2;
                xconv[ly][threadIdx.x][4] = sumXY;

                // Second row
                int ly2 = ly + BLOCK_Y;
                if (ly2 < CONV_Y) {
                    sumX = 0.f;
                    sumX2 = 0.f;
                    sumY = 0.f;
                    sumY2 = 0.f;
                    sumXY = 0.f;

#pragma unroll
                    for (int d = 1; d <= HALO; ++d) {
                        float w = cGauss[HALO - d];
                        float Xleft = sTile[ly2][lx - d][0];
                        float Yleft = sTile[ly2][lx - d][1];
                        float Xright = sTile[ly2][lx + d][0];
                        float Yright = sTile[ly2][lx + d][1];

                        sumX += (Xleft + Xright) * w;
                        sumX2 += ((Xleft * Xleft) + (Xright * Xright)) * w;
                        sumY += (Yleft + Yright) * w;
                        sumY2 += ((Yleft * Yleft) + (Yright * Yright)) * w;
                        sumXY += ((Xleft * Yleft) + (Xright * Yright)) * w;
                    }
                    {
                        float cx = sTile[ly2][lx][0];
                        float cy = sTile[ly2][lx][1];
                        float wc = cGauss[HALO];
                        sumX += cx * wc;
                        sumX2 += (cx * cx) * wc;
                        sumY += cy * wc;
                        sumY2 += (cy * cy) * wc;
                        sumXY += (cx * cy) * wc;
                    }
                    xconv[ly2][threadIdx.x][0] = sumX;
                    xconv[ly2][threadIdx.x][1] = sumX2;
                    xconv[ly2][threadIdx.x][2] = sumY;
                    xconv[ly2][threadIdx.x][3] = sumY2;
                    xconv[ly2][threadIdx.x][4] = sumXY;
                }
            }
            block.sync();

            // 3) Vertical convolution + SSIM + combined loss
            {
                int ly = threadIdx.y + HALO;
                int lx = threadIdx.x;

                float out0 = 0.f, out1 = 0.f, out2 = 0.f, out3 = 0.f, out4 = 0.f;

#pragma unroll
                for (int d = 1; d <= HALO; ++d) {
                    float w = cGauss[HALO - d];
                    float* top = xconv[ly - d][lx];
                    float* bot = xconv[ly + d][lx];

                    out0 += (top[0] + bot[0]) * w;
                    out1 += (top[1] + bot[1]) * w;
                    out2 += (top[2] + bot[2]) * w;
                    out3 += (top[3] + bot[3]) * w;
                    out4 += (top[4] + bot[4]) * w;
                }
                {
                    float wC = cGauss[HALO];
                    float* ctr = xconv[ly][lx];
                    out0 += ctr[0] * wC;
                    out1 += ctr[1] * wC;
                    out2 += ctr[2] * wC;
                    out3 += ctr[3] * wC;
                    out4 += ctr[4] * wC;
                }

                if (pix_x < W && pix_y < H) {
                    float mu1 = out0;
                    float mu2 = out2;
                    float mu1_sq = mu1 * mu1;
                    float mu2_sq = mu2 * mu2;

                    float sigma1_sq = out1 - mu1_sq;
                    float sigma2_sq = out3 - mu2_sq;
                    float sigma12 = out4 - mu1 * mu2;

                    float A = mu1_sq + mu2_sq + C1;
                    float B = sigma1_sq + sigma2_sq + C2;
                    float C_ = 2.f * mu1 * mu2 + C1;
                    float D_ = 2.f * sigma12 + C2;

                    float ssim_val = (C_ * D_) / (A * B);

                    int global_idx = bIdx * CH * num_pix + c * num_pix + pix_id;

                    ssim_sum += ssim_val;
                    cs_sum += D_ / B;
                    if (ssim_channels)
                        ssim_channels[global_idx] = ssim_val;

                    if (dm_dmu1) {
                        float d_m_dmu1 = ((mu2 * 2.f * D_) / (A * B) - (mu2 * 2.f * C_) / (A * B) - (mu1 * 2.f * C_ * D_) / (A * A * B) + (mu1 * 2.f * C_ * D_) / (A * B * B));
                        float d_m_dsigma1_sq = (-C_ * D_) / (A * B * B);
                        float d_m_dsigma12 = (2.f * C_) / (A * B);

                        dm_dmu1[global_idx] = d_m_dmu1;
                        dm_dsigma1_sq[global_idx] = d_m_dsigma1_sq;
                        dm_dsigma12[global_idx] = d_m_dsigma12;
                    }
                }
            }
        }

        if (pix_x < W && pix_y < H) {
            const float inv_c = 1.0f / static_cast<float>(CH);
            if (ssim_map) {
                ssim_map[bIdx * num_pix + pix_id] = ssim_sum * inv_c;
            }
            if (cs_map) {
                cs_map[bIdx * num_pix + pix_id] = cs_sum * inv_c;
            }
        }
    }

    // Fused L1+SSIM Backward Kernel.
    // HasSigmaPartials=false: sigma partials are identically zero (decoupled
    // appearance branch). Avoids a full-image zero_terms buffer.
    template <typename TargetT, bool HasSigmaPartials = true, bool Accumulate = false>
    __global__ void fusedL1SSIMBackwardCUDA(
        float ssim_weight,
        int H,
        int W,
        int CH,
        float C1,
        float C2,
        float grad_per_pixel,
        bool apply_valid_padding,
        const float* __restrict__ img1,
        const TargetT* __restrict__ img2,
        float* __restrict__ dL_dimg1,
        const __half* __restrict__ dm_dmu1,
        const __half* __restrict__ dm_dsigma1_sq,
        const __half* __restrict__ dm_dsigma12) {

        auto block = cg::this_thread_block();
        const int pix_y = block.group_index().y * BLOCK_Y + block.thread_index().y;
        const int pix_x = block.group_index().x * BLOCK_X + block.thread_index().x;
        const int pix_id = pix_y * W + pix_x;
        const int num_pix = H * W;
        const int bIdx = block.group_index().z;

        const float l1_weight = 1.0f - ssim_weight;

        __shared__ float sData[SHARED_Y][SHARED_X][3];
        __shared__ float sScratch[CONV_Y][CONV_X][3];

        for (int c = 0; c < CH; ++c) {
            float p1 = 0.f, p2 = 0.f;
            if (pix_x < W && pix_y < H) {
                p1 = get_pix_value(img1, bIdx, c, pix_y, pix_x, CH, H, W);
                p2 = get_pix_value(img2, bIdx, c, pix_y, pix_x, CH, H, W);
            }

            // 1) Load + fuse multiplication
            {
                const int start_y = block.group_index().y * BLOCK_Y;
                const int start_x = block.group_index().x * BLOCK_X;

                int tid = threadIdx.y * blockDim.x + threadIdx.x;
                int warp_id = tid / 32;
                int lane_id = tid % 32;
                int totalThreads = BLOCK_X * BLOCK_Y;
                int num_warps = (totalThreads + 31) / 32;

                for (int row = warp_id; row < SHARED_Y; row += num_warps) {
                    int gy = start_y + row - HALO;
                    for (int col = lane_id; col < SHARED_X; col += 32) {
                        int gx = start_x + col - HALO;

                        float chain = 0.0f;
                        if (gx >= 0 && gx < W && gy >= 0 && gy < H) {
                            const bool inside_valid_region =
                                !apply_valid_padding ||
                                ((H <= 10 || (gy >= 5 && gy < H - 5)) &&
                                 (W <= 10 || (gx >= 5 && gx < W - 5)));
                            if (inside_valid_region) {
                                chain = grad_per_pixel;
                            }
                        }
                        float vmu = get_pix_value(dm_dmu1, bIdx, c, gy, gx, CH, H, W);
                        float vs1 = 0.f, vs12 = 0.f;
                        if constexpr (HasSigmaPartials) {
                            vs1 = get_pix_value(dm_dsigma1_sq, bIdx, c, gy, gx, CH, H, W);
                            vs12 = get_pix_value(dm_dsigma12, bIdx, c, gy, gx, CH, H, W);
                        }

                        // SSIM gradient needs -ssim_weight (d(1-ssim)/d(ssim) = -1)
                        sData[row][col][0] = -ssim_weight * vmu * chain;
                        sData[row][col][1] = -ssim_weight * vs1 * chain;
                        sData[row][col][2] = -ssim_weight * vs12 * chain;
                    }
                }
            }
            block.sync();

            // 2) Horizontal pass
            {
                int ly = threadIdx.y;
                int lx = threadIdx.x + HALO;

                for (int pass = 0; pass < 2; ++pass) {
                    int yy = ly + pass * BLOCK_Y;
                    if (yy < CONV_Y) {
                        float accum0 = 0.f, accum1 = 0.f, accum2 = 0.f;

#pragma unroll
                        for (int d = 1; d <= HALO; ++d) {
                            float w = cGauss[HALO - d];
                            float left0 = sData[yy][lx - d][0];
                            float left1 = sData[yy][lx - d][1];
                            float left2 = sData[yy][lx - d][2];

                            float right0 = sData[yy][lx + d][0];
                            float right1 = sData[yy][lx + d][1];
                            float right2 = sData[yy][lx + d][2];

                            accum0 += (left0 + right0) * w;
                            accum1 += (left1 + right1) * w;
                            accum2 += (left2 + right2) * w;
                        }
                        {
                            float wc = cGauss[HALO];
                            float c0 = sData[yy][lx][0];
                            float c1 = sData[yy][lx][1];
                            float c2 = sData[yy][lx][2];
                            accum0 += c0 * wc;
                            accum1 += c1 * wc;
                            accum2 += c2 * wc;
                        }

                        sScratch[yy][threadIdx.x][0] = accum0;
                        sScratch[yy][threadIdx.x][1] = accum1;
                        sScratch[yy][threadIdx.x][2] = accum2;
                    }
                }
            }
            block.sync();

            // 3) Vertical pass + L1 gradient + output
            if (pix_x < W && pix_y < H) {
                int ly = threadIdx.y + HALO;
                int lx = threadIdx.x;

                float sum0 = 0.f, sum1 = 0.f, sum2 = 0.f;

#pragma unroll
                for (int d = 1; d <= HALO; ++d) {
                    float w = cGauss[HALO - d];
                    float* top = sScratch[ly - d][lx];
                    float* bot = sScratch[ly + d][lx];

                    sum0 += (top[0] + bot[0]) * w;
                    sum1 += (top[1] + bot[1]) * w;
                    sum2 += (top[2] + bot[2]) * w;
                }
                {
                    float wc = cGauss[HALO];
                    float* ctr = sScratch[ly][lx];
                    sum0 += ctr[0] * wc;
                    sum1 += ctr[1] * wc;
                    sum2 += ctr[2] * wc;
                }

                // SSIM gradient
                float grad_ssim = sum0 + (2.f * p1) * sum1 + p2 * sum2;

                // L1 gradient: sign(p1 - p2) * l1_weight * chain
                int out_idx = bIdx * CH * num_pix + c * num_pix + pix_id;
                float chain_local = 0.0f;
                const bool inside_valid_region =
                    !apply_valid_padding ||
                    ((H <= 10 || (pix_y >= 5 && pix_y < H - 5)) &&
                     (W <= 10 || (pix_x >= 5 && pix_x < W - 5)));
                if (inside_valid_region) {
                    chain_local = grad_per_pixel;
                }
                float sign_grad = (p1 == p2) ? 0.0f : copysignf(1.0f, p1 - p2);
                float grad_l1 = l1_weight * sign_grad * chain_local;

                // Combined gradient
                if constexpr (Accumulate)
                    dL_dimg1[out_idx] += grad_ssim + grad_l1;
                else
                    dL_dimg1[out_idx] = grad_ssim + grad_l1;
            }
            block.sync();
        }
    }

    // Normalize only the masked samples; retain the existing zero-padding
    // convention outside the image. The scratch storage is reused by the image
    // convolution, so no per-pixel allocation or extra kernel is needed.
    template <typename MaskT, int Moments>
    __device__ float load_mask_window(
        const MaskT* mask, int H, int W,
        float (&tile)[SHARED_Y][SHARED_X],
        float (&scratch)[CONV_Y][CONV_X][Moments], bool& full_support) {
        const auto block = cg::this_thread_block();
        const int start_y = block.group_index().y * BLOCK_Y;
        const int start_x = block.group_index().x * BLOCK_X;
        // A common scale cancels from normalized moments and center/mass.
        // Removing it improves second-moment precision for structure weights.
        const int anchor_y = min(start_y + BLOCK_Y / 2, H - 1);
        const int anchor_x = min(start_x + BLOCK_X / 2, W - 1);
        const float inv_scale = 1.0f / fmaxf(1.0f, mask_value(mask, anchor_y * W + anchor_x));
        bool all_one = true;
        for (int i = block.thread_rank(); i < SHARED_Y * SHARED_X; i += BLOCK_X * BLOCK_Y) {
            const int y = i / SHARED_X;
            const int x = i % SHARED_X;
            const int gy = start_y + y - HALO;
            const int gx = start_x + x - HALO;
            const float value = (gx >= 0 && gx < W && gy >= 0 && gy < H
                                     ? mask_value(mask, gy * W + gx)
                                     : 1.0f) *
                                inv_scale;
            tile[y][x] = value;
            all_one = all_one && value == 1.0f;
        }
        // Most binary-mask tiles have full support; their Gaussian mass is one.
        full_support = __syncthreads_and(all_one);
        if (full_support)
            return 1.0f;
        const int x = threadIdx.x + HALO;
        for (int y = threadIdx.y; y < CONV_Y; y += BLOCK_Y) {
            float sum = tile[y][x] * cGauss[HALO];
#pragma unroll
            for (int d = 1; d <= HALO; ++d)
                sum += (tile[y][x - d] + tile[y][x + d]) * cGauss[HALO - d];
            scratch[y][threadIdx.x][0] = sum;
        }
        block.sync();
        const int y = threadIdx.y + HALO;
        float sum = scratch[y][threadIdx.x][0] * cGauss[HALO];
#pragma unroll
        for (int d = 1; d <= HALO; ++d)
            sum += (scratch[y - d][threadIdx.x][0] + scratch[y + d][threadIdx.x][0]) * cGauss[HALO - d];
        // Below this mass the window is unsupported. This is far below the
        // loss reduction's epsilon, and bounds the reciprocal even without FTZ.
        constexpr float min_window_mass = 1.0e-30f;
        return sum >= min_window_mass ? 1.0f / sum : 0.0f;
    }

    // Full-support tiles retain the cheaper symmetric convolution.
    template <bool Weighted>
    __device__ __forceinline__ void masked_horizontal_moments(
        const float (&sTile)[SHARED_Y][SHARED_X][2],
        const float (&mask_tile)[SHARED_Y][SHARED_X],
        float (&xconv)[CONV_Y][CONV_X][5]) {
        int ly = threadIdx.y;
        int lx = threadIdx.x + HALO;

        float sumX = 0.f, sumX2 = 0.f, sumY = 0.f, sumY2 = 0.f, sumXY = 0.f;

#pragma unroll
        for (int d = 1; d <= HALO; ++d) {
            float w = cGauss[HALO - d];
            float Xleft = sTile[ly][lx - d][0];
            float Yleft = sTile[ly][lx - d][1];
            float Xright = sTile[ly][lx + d][0];
            float Yright = sTile[ly][lx + d][1];
            if constexpr (Weighted) {
                const float ml = mask_tile[ly][lx - d] * w;
                const float mr = mask_tile[ly][lx + d] * w;

                sumX += Xleft * ml + Xright * mr;
                sumX2 += (Xleft * Xleft) * ml + (Xright * Xright) * mr;
                sumY += Yleft * ml + Yright * mr;
                sumY2 += (Yleft * Yleft) * ml + (Yright * Yright) * mr;
                sumXY += (Xleft * Yleft) * ml + (Xright * Yright) * mr;
            } else {
                sumX += (Xleft + Xright) * w;
                sumX2 += (Xleft * Xleft + Xright * Xright) * w;
                sumY += (Yleft + Yright) * w;
                sumY2 += (Yleft * Yleft + Yright * Yright) * w;
                sumXY += (Xleft * Yleft + Xright * Yright) * w;
            }
        }
        {
            float centerX = sTile[ly][lx][0];
            float centerY = sTile[ly][lx][1];
            float wc = cGauss[HALO];
            if constexpr (Weighted)
                wc *= mask_tile[ly][lx];
            sumX += centerX * wc;
            sumX2 += (centerX * centerX) * wc;
            sumY += centerY * wc;
            sumY2 += (centerY * centerY) * wc;
            sumXY += (centerX * centerY) * wc;
        }

        xconv[ly][threadIdx.x][0] = sumX;
        xconv[ly][threadIdx.x][1] = sumX2;
        xconv[ly][threadIdx.x][2] = sumY;
        xconv[ly][threadIdx.x][3] = sumY2;
        xconv[ly][threadIdx.x][4] = sumXY;

        int ly2 = ly + BLOCK_Y;
        if (ly2 < CONV_Y) {
            sumX = 0.f;
            sumX2 = 0.f;
            sumY = 0.f;
            sumY2 = 0.f;
            sumXY = 0.f;

#pragma unroll
            for (int d = 1; d <= HALO; ++d) {
                float w = cGauss[HALO - d];
                float Xleft = sTile[ly2][lx - d][0];
                float Yleft = sTile[ly2][lx - d][1];
                float Xright = sTile[ly2][lx + d][0];
                float Yright = sTile[ly2][lx + d][1];
                if constexpr (Weighted) {
                    const float ml = mask_tile[ly2][lx - d] * w;
                    const float mr = mask_tile[ly2][lx + d] * w;

                    sumX += Xleft * ml + Xright * mr;
                    sumX2 += (Xleft * Xleft) * ml + (Xright * Xright) * mr;
                    sumY += Yleft * ml + Yright * mr;
                    sumY2 += (Yleft * Yleft) * ml + (Yright * Yright) * mr;
                    sumXY += (Xleft * Yleft) * ml + (Xright * Yright) * mr;
                } else {
                    sumX += (Xleft + Xright) * w;
                    sumX2 += (Xleft * Xleft + Xright * Xright) * w;
                    sumY += (Yleft + Yright) * w;
                    sumY2 += (Yleft * Yleft + Yright * Yright) * w;
                    sumXY += (Xleft * Yleft + Xright * Yright) * w;
                }
            }
            {
                float cx = sTile[ly2][lx][0];
                float cy = sTile[ly2][lx][1];
                float wc = cGauss[HALO];
                if constexpr (Weighted)
                    wc *= mask_tile[ly2][lx];
                sumX += cx * wc;
                sumX2 += (cx * cx) * wc;
                sumY += cy * wc;
                sumY2 += (cy * cy) * wc;
                sumXY += (cx * cy) * wc;
            }
            xconv[ly2][threadIdx.x][0] = sumX;
            xconv[ly2][threadIdx.x][1] = sumX2;
            xconv[ly2][threadIdx.x][2] = sumY;
            xconv[ly2][threadIdx.x][3] = sumY2;
            xconv[ly2][threadIdx.x][4] = sumXY;
        }
    }

    // Masked Fused L1+SSIM Forward Kernel.
    template <typename TargetT, typename MaskT>
    __global__ void maskedFusedL1SSIMForwardCUDA(
        int H,
        int W,
        int CH,
        float C1,
        float C2,
        const float* __restrict__ img1,
        const TargetT* __restrict__ img2,
        const MaskT* __restrict__ mask,
        __half* __restrict__ dm_dmu1,
        __half* __restrict__ dm_dsigma1_sq,
        __half* __restrict__ dm_dsigma12,
        float* __restrict__ ssim_map,
        float* __restrict__ cs_map,
        float* __restrict__ ssim_channels) { // optional [N, C, H, W] per-channel SSIM

        auto block = cg::this_thread_block();
        const int bIdx = block.group_index().z;
        const int pix_y = block.group_index().y * BLOCK_Y + block.thread_index().y;
        const int pix_x = block.group_index().x * BLOCK_X + block.thread_index().x;
        const int pix_id = pix_y * W + pix_x;
        const int num_pix = H * W;

        __shared__ float sTile[SHARED_Y][SHARED_X][2];
        __shared__ float xconv[CONV_Y][CONV_X][5];

        __shared__ float mask_tile[SHARED_Y][SHARED_X];
        bool full_support;
        const float inv_window = load_mask_window(mask, H, W, mask_tile, xconv, full_support);
        const float center_weight = mask_tile[threadIdx.y + HALO][threadIdx.x + HALO];
        const float derivative_weight = center_weight * inv_window;
        // Ignored centers contribute no loss or partials, even near a kept tile.
        if (!__syncthreads_or(pix_x < W && pix_y < H && center_weight != 0.0f)) {
            if (pix_x < W && pix_y < H) {
                for (int c = 0; c < CH; ++c) {
                    const int index = bIdx * CH * num_pix + c * num_pix + pix_id;
                    if (ssim_channels)
                        ssim_channels[index] = 1.0f;
                    if (dm_dmu1)
                        dm_dmu1[index] = __float2half(0.0f);
                    if (dm_dsigma1_sq)
                        dm_dsigma1_sq[index] = __float2half(0.0f);
                    if (dm_dsigma12)
                        dm_dsigma12[index] = __float2half(0.0f);
                }
                if (ssim_map)
                    ssim_map[bIdx * num_pix + pix_id] = 1.0f;
                if (cs_map)
                    cs_map[bIdx * num_pix + pix_id] = 1.0f;
            }
            return;
        }

        float ssim_sum = 0.0f;
        float cs_sum = 0.0f;
        for (int c = 0; c < CH; ++c) {
            // 1) Load tile
            {
                const int tileSize = SHARED_Y * SHARED_X;
                const int threads = BLOCK_X * BLOCK_Y;
                const int steps = (tileSize + threads - 1) / threads;
                const int tileStartY = block.group_index().y * BLOCK_Y;
                const int tileStartX = block.group_index().x * BLOCK_X;

                for (int s = 0; s < steps; ++s) {
                    int tid = s * threads + block.thread_rank();
                    if (tid < tileSize) {
                        int local_y = tid / SHARED_X;
                        int local_x = tid % SHARED_X;
                        int gy = tileStartY + local_y - HALO;
                        int gx = tileStartX + local_x - HALO;

                        float X = get_pix_value(img1, bIdx, c, gy, gx, CH, H, W);
                        float Y = get_pix_value(img2, bIdx, c, gy, gx, CH, H, W);

                        sTile[local_y][local_x][0] = X;
                        sTile[local_y][local_x][1] = Y;
                    }
                }
            }
            block.sync();

            if (full_support)
                masked_horizontal_moments<false>(sTile, mask_tile, xconv);
            else
                masked_horizontal_moments<true>(sTile, mask_tile, xconv);
            block.sync();

            // 3) Vertical convolution + SSIM
            {
                int ly = threadIdx.y + HALO;
                int lx = threadIdx.x;

                float out0 = 0.f, out1 = 0.f, out2 = 0.f, out3 = 0.f, out4 = 0.f;

#pragma unroll
                for (int d = 1; d <= HALO; ++d) {
                    float w = cGauss[HALO - d];
                    float* top = xconv[ly - d][lx];
                    float* bot = xconv[ly + d][lx];

                    out0 += (top[0] + bot[0]) * w;
                    out1 += (top[1] + bot[1]) * w;
                    out2 += (top[2] + bot[2]) * w;
                    out3 += (top[3] + bot[3]) * w;
                    out4 += (top[4] + bot[4]) * w;
                }
                {
                    float wC = cGauss[HALO];
                    float* ctr = xconv[ly][lx];
                    out0 += ctr[0] * wC;
                    out1 += ctr[1] * wC;
                    out2 += ctr[2] * wC;
                    out3 += ctr[3] * wC;
                    out4 += ctr[4] * wC;
                }

                if (pix_x < W && pix_y < H) {
                    float mu1 = out0 * inv_window;
                    float mu2 = out2 * inv_window;
                    float mu1_sq = mu1 * mu1;
                    float mu2_sq = mu2 * mu2;

                    float sigma1_sq = out1 * inv_window - mu1_sq;
                    float sigma2_sq = out3 * inv_window - mu2_sq;
                    float sigma12 = out4 * inv_window - mu1 * mu2;

                    float A = mu1_sq + mu2_sq + C1;
                    float B = sigma1_sq + sigma2_sq + C2;
                    float C_ = 2.f * mu1 * mu2 + C1;
                    float D_ = 2.f * sigma12 + C2;

                    float ssim_val = (C_ * D_) / (A * B);

                    int global_idx = bIdx * CH * num_pix + c * num_pix + pix_id;

                    ssim_sum += ssim_val;
                    cs_sum += D_ / B;
                    if (ssim_channels)
                        ssim_channels[global_idx] = ssim_val;

                    if (dm_dmu1) {
                        float d_m_dmu1 = ((mu2 * 2.f * D_) / (A * B) - (mu2 * 2.f * C_) / (A * B) - (mu1 * 2.f * C_ * D_) / (A * A * B) + (mu1 * 2.f * C_ * D_) / (A * B * B));
                        float d_m_dsigma1_sq = (-C_ * D_) / (A * B * B);
                        float d_m_dsigma12 = (2.f * C_) / (A * B);

                        dm_dmu1[global_idx] = static_cast<__half>(d_m_dmu1 * derivative_weight);
                        dm_dsigma1_sq[global_idx] = static_cast<__half>(d_m_dsigma1_sq * derivative_weight);
                        dm_dsigma12[global_idx] = static_cast<__half>(d_m_dsigma12 * derivative_weight);
                    }
                }
            }
        }

        if (pix_x < W && pix_y < H) {
            const float inv_c = 1.0f / static_cast<float>(CH);
            if (ssim_map) {
                ssim_map[bIdx * num_pix + pix_id] = center_weight > 0.0f ? ssim_sum * inv_c : 1.0f;
            }
            if (cs_map) {
                cs_map[bIdx * num_pix + pix_id] = center_weight > 0.0f ? cs_sum * inv_c : 1.0f;
            }
        }
    }

    template <typename TargetT, typename MaskT>
    __global__ void maskedDecoupledFusedL1SSIMForwardCUDA(
        int H,
        int W,
        int CH,
        float C1,
        float C2,
        float ssim_weight,
        const float* __restrict__ corrected_img,
        const float* __restrict__ raw_img,
        const TargetT* __restrict__ gt_img,
        const MaskT* __restrict__ mask,
        __half* __restrict__ app_dm_dmu1,
        __half* __restrict__ raw_dm_dmu1,
        __half* __restrict__ raw_dm_dsigma1_sq,
        __half* __restrict__ raw_dm_dsigma12,
        float* __restrict__ ssim_map,
        float* __restrict__ cs_map,
        float* __restrict__ ssim_channels) { // optional [N, C, H, W] per-channel SSIM

        auto block = cg::this_thread_block();
        const int bIdx = block.group_index().z;
        const int pix_y = block.group_index().y * BLOCK_Y + block.thread_index().y;
        const int pix_x = block.group_index().x * BLOCK_X + block.thread_index().x;
        const int pix_id = pix_y * W + pix_x;
        const int num_pix = H * W;

        __shared__ float sTile[SHARED_Y][SHARED_X][3];
        // [0]=corrected, [1]=raw, [2]=raw^2, [3]=gt, [4]=gt^2, [5]=raw*gt
        __shared__ float xconv[CONV_Y][CONV_X][6];

        __shared__ float mask_tile[SHARED_Y][SHARED_X];
        bool full_support;
        const float inv_window = load_mask_window(mask, H, W, mask_tile, xconv, full_support);
        const float center_weight = mask_tile[threadIdx.y + HALO][threadIdx.x + HALO];
        const float derivative_weight = center_weight * inv_window;
        // Ignored centers contribute no loss or partials, even near a kept tile.
        if (!__syncthreads_or(pix_x < W && pix_y < H && center_weight != 0.0f)) {
            if (pix_x < W && pix_y < H) {
                for (int c = 0; c < CH; ++c) {
                    const int index = bIdx * CH * num_pix + c * num_pix + pix_id;
                    if (app_dm_dmu1)
                        app_dm_dmu1[index] = __float2half(0.0f);
                    if (raw_dm_dmu1)
                        raw_dm_dmu1[index] = __float2half(0.0f);
                    if (raw_dm_dsigma1_sq)
                        raw_dm_dsigma1_sq[index] = __float2half(0.0f);
                    if (raw_dm_dsigma12)
                        raw_dm_dsigma12[index] = __float2half(0.0f);
                    if (ssim_channels)
                        ssim_channels[index] = 1.0f;
                }
                if (ssim_map)
                    ssim_map[bIdx * num_pix + pix_id] = 1.0f;
                if (cs_map)
                    cs_map[bIdx * num_pix + pix_id] = 1.0f;
            }
            return;
        }

        float ssim_sum = 0.0f;
        float cs_sum = 0.0f;
        for (int c = 0; c < CH; ++c) {
            {
                const int tileSize = SHARED_Y * SHARED_X;
                const int threads = BLOCK_X * BLOCK_Y;
                const int steps = (tileSize + threads - 1) / threads;
                const int tileStartY = block.group_index().y * BLOCK_Y;
                const int tileStartX = block.group_index().x * BLOCK_X;

                for (int s = 0; s < steps; ++s) {
                    int tid = s * threads + block.thread_rank();
                    if (tid < tileSize) {
                        int local_y = tid / SHARED_X;
                        int local_x = tid % SHARED_X;
                        int gy = tileStartY + local_y - HALO;
                        int gx = tileStartX + local_x - HALO;

                        sTile[local_y][local_x][0] = get_pix_value(corrected_img, bIdx, c, gy, gx, CH, H, W);
                        sTile[local_y][local_x][1] = get_pix_value(raw_img, bIdx, c, gy, gx, CH, H, W);
                        sTile[local_y][local_x][2] = get_pix_value(gt_img, bIdx, c, gy, gx, CH, H, W);
                    }
                }
            }
            block.sync();

            {
                int ly = threadIdx.y;
                int lx = threadIdx.x + HALO;

                float sum_corrected = 0.f;
                float sum_raw = 0.f;
                float sum_raw_sq = 0.f;
                float sum_gt = 0.f;
                float sum_gt_sq = 0.f;
                float sum_raw_gt = 0.f;

#pragma unroll
                for (int d = 1; d <= HALO; ++d) {
                    const float w = cGauss[HALO - d];

                    const float corrected_left = sTile[ly][lx - d][0];
                    const float corrected_right = sTile[ly][lx + d][0];
                    const float raw_left = sTile[ly][lx - d][1];
                    const float raw_right = sTile[ly][lx + d][1];
                    const float gt_left = sTile[ly][lx - d][2];
                    const float gt_right = sTile[ly][lx + d][2];
                    const float ml = mask_tile[ly][lx - d] * w;
                    const float mr = mask_tile[ly][lx + d] * w;

                    sum_corrected += corrected_left * ml + corrected_right * mr;
                    sum_raw += raw_left * ml + raw_right * mr;
                    sum_raw_sq += (raw_left * raw_left) * ml + (raw_right * raw_right) * mr;
                    sum_gt += gt_left * ml + gt_right * mr;
                    sum_gt_sq += (gt_left * gt_left) * ml + (gt_right * gt_right) * mr;
                    sum_raw_gt += (raw_left * gt_left) * ml + (raw_right * gt_right) * mr;
                }

                {
                    const float corrected_center = sTile[ly][lx][0];
                    const float raw_center = sTile[ly][lx][1];
                    const float gt_center = sTile[ly][lx][2];
                    const float wc = cGauss[HALO] * mask_tile[ly][lx];

                    sum_corrected += corrected_center * wc;
                    sum_raw += raw_center * wc;
                    sum_raw_sq += (raw_center * raw_center) * wc;
                    sum_gt += gt_center * wc;
                    sum_gt_sq += (gt_center * gt_center) * wc;
                    sum_raw_gt += (raw_center * gt_center) * wc;
                }

                xconv[ly][threadIdx.x][0] = sum_corrected;
                xconv[ly][threadIdx.x][1] = sum_raw;
                xconv[ly][threadIdx.x][2] = sum_raw_sq;
                xconv[ly][threadIdx.x][3] = sum_gt;
                xconv[ly][threadIdx.x][4] = sum_gt_sq;
                xconv[ly][threadIdx.x][5] = sum_raw_gt;

                const int ly2 = ly + BLOCK_Y;
                if (ly2 < CONV_Y) {
                    sum_corrected = 0.f;
                    sum_raw = 0.f;
                    sum_raw_sq = 0.f;
                    sum_gt = 0.f;
                    sum_gt_sq = 0.f;
                    sum_raw_gt = 0.f;

#pragma unroll
                    for (int d = 1; d <= HALO; ++d) {
                        const float w = cGauss[HALO - d];

                        const float corrected_left = sTile[ly2][lx - d][0];
                        const float corrected_right = sTile[ly2][lx + d][0];
                        const float raw_left = sTile[ly2][lx - d][1];
                        const float raw_right = sTile[ly2][lx + d][1];
                        const float gt_left = sTile[ly2][lx - d][2];
                        const float gt_right = sTile[ly2][lx + d][2];
                        const float ml = mask_tile[ly2][lx - d] * w;
                        const float mr = mask_tile[ly2][lx + d] * w;

                        sum_corrected += corrected_left * ml + corrected_right * mr;
                        sum_raw += raw_left * ml + raw_right * mr;
                        sum_raw_sq += (raw_left * raw_left) * ml + (raw_right * raw_right) * mr;
                        sum_gt += gt_left * ml + gt_right * mr;
                        sum_gt_sq += (gt_left * gt_left) * ml + (gt_right * gt_right) * mr;
                        sum_raw_gt += (raw_left * gt_left) * ml + (raw_right * gt_right) * mr;
                    }

                    {
                        const float corrected_center = sTile[ly2][lx][0];
                        const float raw_center = sTile[ly2][lx][1];
                        const float gt_center = sTile[ly2][lx][2];
                        const float wc = cGauss[HALO] * mask_tile[ly2][lx];

                        sum_corrected += corrected_center * wc;
                        sum_raw += raw_center * wc;
                        sum_raw_sq += (raw_center * raw_center) * wc;
                        sum_gt += gt_center * wc;
                        sum_gt_sq += (gt_center * gt_center) * wc;
                        sum_raw_gt += (raw_center * gt_center) * wc;
                    }

                    xconv[ly2][threadIdx.x][0] = sum_corrected;
                    xconv[ly2][threadIdx.x][1] = sum_raw;
                    xconv[ly2][threadIdx.x][2] = sum_raw_sq;
                    xconv[ly2][threadIdx.x][3] = sum_gt;
                    xconv[ly2][threadIdx.x][4] = sum_gt_sq;
                    xconv[ly2][threadIdx.x][5] = sum_raw_gt;
                }
            }
            block.sync();

            {
                int ly = threadIdx.y + HALO;
                int lx = threadIdx.x;

                float corrected_conv = 0.f;
                float raw_conv = 0.f;
                float raw_sq_conv = 0.f;
                float gt_conv = 0.f;
                float gt_sq_conv = 0.f;
                float raw_gt_conv = 0.f;

#pragma unroll
                for (int d = 1; d <= HALO; ++d) {
                    const float w = cGauss[HALO - d];
                    float* top = xconv[ly - d][lx];
                    float* bot = xconv[ly + d][lx];

                    corrected_conv += (top[0] + bot[0]) * w;
                    raw_conv += (top[1] + bot[1]) * w;
                    raw_sq_conv += (top[2] + bot[2]) * w;
                    gt_conv += (top[3] + bot[3]) * w;
                    gt_sq_conv += (top[4] + bot[4]) * w;
                    raw_gt_conv += (top[5] + bot[5]) * w;
                }

                {
                    const float wc = cGauss[HALO];
                    float* ctr = xconv[ly][lx];
                    corrected_conv += ctr[0] * wc;
                    raw_conv += ctr[1] * wc;
                    raw_sq_conv += ctr[2] * wc;
                    gt_conv += ctr[3] * wc;
                    gt_sq_conv += ctr[4] * wc;
                    raw_gt_conv += ctr[5] * wc;
                }

                if (pix_x < W && pix_y < H) {
                    const float mu_corrected = corrected_conv * inv_window;
                    const float mu_raw = raw_conv * inv_window;
                    const float mu_gt = gt_conv * inv_window;

                    const float mu_corrected_sq = mu_corrected * mu_corrected;
                    const float mu_raw_sq = mu_raw * mu_raw;
                    const float mu_gt_sq = mu_gt * mu_gt;

                    const float sigma_raw_sq = raw_sq_conv * inv_window - mu_raw_sq;
                    const float sigma_gt_sq = gt_sq_conv * inv_window - mu_gt_sq;
                    const float sigma12 = raw_gt_conv * inv_window - mu_raw * mu_gt;

                    const float A_app = mu_corrected_sq + mu_gt_sq + C1;
                    const float C_app = 2.f * mu_corrected * mu_gt + C1;
                    const float luminance = C_app / A_app;

                    const float B_raw = sigma_raw_sq + sigma_gt_sq + C2;
                    const float D_raw = 2.f * sigma12 + C2;
                    const float contrast_structure = D_raw / B_raw;

                    const int global_idx = bIdx * CH * num_pix + c * num_pix + pix_id;
                    const float ssim_val = luminance * contrast_structure;
                    ssim_sum += ssim_val;
                    if (ssim_channels)
                        ssim_channels[global_idx] = ssim_val;
                    cs_sum += contrast_structure;

                    if (app_dm_dmu1) {
                        const float dl_dmu1 =
                            2.f * (mu_gt * A_app - mu_corrected * C_app) / (A_app * A_app);
                        app_dm_dmu1[global_idx] = static_cast<__half>(contrast_structure * dl_dmu1 * derivative_weight);

                        const float ds1sq = ssim_weight * (-(luminance * D_raw) / (B_raw * B_raw));
                        const float ds12 = ssim_weight * ((2.f * luminance) / B_raw);
                        raw_dm_dsigma1_sq[global_idx] = static_cast<__half>(ds1sq * derivative_weight);
                        raw_dm_dsigma12[global_idx] = static_cast<__half>(ds12 * derivative_weight);
                        // center / mass < 16 can overflow the raw mean partial
                        // in half even for raw RGB = 3. Remove that amplification
                        // with an exact power of two; backward restores it in float.
                        raw_dm_dmu1[global_idx] =
                            static_cast<__half>((ds1sq * (-2.f * mu_raw) + ds12 * (-mu_gt)) * (derivative_weight * (1.0f / 16.0f)));
                    }
                }
            }
        }

        if (pix_x < W && pix_y < H) {
            const float inv_c = 1.0f / static_cast<float>(CH);
            if (ssim_map) {
                ssim_map[bIdx * num_pix + pix_id] = center_weight > 0.0f ? ssim_sum * inv_c : 1.0f;
            }
            if (cs_map) {
                cs_map[bIdx * num_pix + pix_id] = center_weight > 0.0f ? cs_sum * inv_c : 1.0f;
            }
        }
    }

    // Masked Fused L1+SSIM Backward Kernel.
    // HasSigmaPartials=false: appearance branch skips unused sigma partials.
    template <typename TargetT, typename MaskT, bool HasSigmaPartials = true, bool Accumulate = false, bool ScaledMeanPartial = false>
    __global__ void maskedFusedL1SSIMBackwardCUDA(
        float ssim_weight,
        float inv_mask_sum, // 1.0 / mask_sum for normalization
        int H,
        int W,
        int CH,
        float C1,
        float C2,
        const float* __restrict__ img1,
        const TargetT* __restrict__ img2,
        const MaskT* __restrict__ mask,
        float* __restrict__ dL_dimg1,
        const __half* __restrict__ dm_dmu1,
        const __half* __restrict__ dm_dsigma1_sq,
        const __half* __restrict__ dm_dsigma12) {

        auto block = cg::this_thread_block();
        const int pix_y = block.group_index().y * BLOCK_Y + block.thread_index().y;
        const int pix_x = block.group_index().x * BLOCK_X + block.thread_index().x;
        const int pix_id = pix_y * W + pix_x;
        const int num_pix = H * W;
        const int bIdx = block.group_index().z;

        const float l1_weight = 1.0f - ssim_weight;

        // Get mask value
        float mask_val = 0.0f;
        if (pix_x < W && pix_y < H) {
            mask_val = mask_value(mask, pix_y * W + pix_x);
        }

        // No sample gradient can cross into an ignored output tile.
        if (!__syncthreads_or(mask_val != 0.0f)) {
            if constexpr (!Accumulate) {
                if (pix_x < W && pix_y < H)
                    for (int c = 0; c < CH; ++c)
                        dL_dimg1[bIdx * CH * num_pix + c * num_pix + pix_id] = 0.0f;
            }
            return;
        }

        __shared__ float sData[SHARED_Y][SHARED_X][3];
        __shared__ float sScratch[CONV_Y][CONV_X][3];

        for (int c = 0; c < CH; ++c) {
            float p1 = 0.f, p2 = 0.f;
            if (pix_x < W && pix_y < H) {
                p1 = get_pix_value(img1, bIdx, c, pix_y, pix_x, CH, H, W);
                p2 = get_pix_value(img2, bIdx, c, pix_y, pix_x, CH, H, W);
            }

            // 1) Load SSIM derivatives (weighted by mask and inv_mask_sum)
            {
                const int start_y = block.group_index().y * BLOCK_Y;
                const int start_x = block.group_index().x * BLOCK_X;

                int tid = threadIdx.y * blockDim.x + threadIdx.x;
                int warp_id = tid / 32;
                int lane_id = tid % 32;
                int totalThreads = BLOCK_X * BLOCK_Y;
                int num_warps = (totalThreads + 31) / 32;

                for (int row = warp_id; row < SHARED_Y; row += num_warps) {
                    int gy = start_y + row - HALO;
                    for (int col = lane_id; col < SHARED_X; col += 32) {
                        int gx = start_x + col - HALO;

                        // Forward partials include center weight / window mass.
                        const float chain = inv_mask_sum;

                        float vmu = get_pix_value(dm_dmu1, bIdx, c, gy, gx, CH, H, W);
                        if constexpr (ScaledMeanPartial)
                            vmu *= 16.0f;
                        float vs1 = 0.f, vs12 = 0.f;
                        if constexpr (HasSigmaPartials) {
                            vs1 = get_pix_value(dm_dsigma1_sq, bIdx, c, gy, gx, CH, H, W);
                            vs12 = get_pix_value(dm_dsigma12, bIdx, c, gy, gx, CH, H, W);
                        }

                        sData[row][col][0] = -ssim_weight * vmu * chain;
                        sData[row][col][1] = -ssim_weight * vs1 * chain;
                        sData[row][col][2] = -ssim_weight * vs12 * chain;
                    }
                }
            }
            block.sync();

            // 2) Horizontal pass
            {
                int ly = threadIdx.y;
                int lx = threadIdx.x + HALO;

                for (int pass = 0; pass < 2; ++pass) {
                    int yy = ly + pass * BLOCK_Y;
                    if (yy < CONV_Y) {
                        float accum0 = 0.f, accum1 = 0.f, accum2 = 0.f;

#pragma unroll
                        for (int d = 1; d <= HALO; ++d) {
                            float w = cGauss[HALO - d];
                            accum0 += (sData[yy][lx - d][0] + sData[yy][lx + d][0]) * w;
                            accum1 += (sData[yy][lx - d][1] + sData[yy][lx + d][1]) * w;
                            accum2 += (sData[yy][lx - d][2] + sData[yy][lx + d][2]) * w;
                        }
                        {
                            float wc = cGauss[HALO];
                            accum0 += sData[yy][lx][0] * wc;
                            accum1 += sData[yy][lx][1] * wc;
                            accum2 += sData[yy][lx][2] * wc;
                        }

                        sScratch[yy][threadIdx.x][0] = accum0;
                        sScratch[yy][threadIdx.x][1] = accum1;
                        sScratch[yy][threadIdx.x][2] = accum2;
                    }
                }
            }
            block.sync();

            // 3) Vertical pass + L1 gradient
            if (pix_x < W && pix_y < H) {
                int ly = threadIdx.y + HALO;
                int lx = threadIdx.x;

                float sum0 = 0.f, sum1 = 0.f, sum2 = 0.f;

#pragma unroll
                for (int d = 1; d <= HALO; ++d) {
                    float w = cGauss[HALO - d];
                    float* top = sScratch[ly - d][lx];
                    float* bot = sScratch[ly + d][lx];

                    sum0 += (top[0] + bot[0]) * w;
                    sum1 += (top[1] + bot[1]) * w;
                    sum2 += (top[2] + bot[2]) * w;
                }
                {
                    float wc = cGauss[HALO];
                    float* ctr = sScratch[ly][lx];
                    sum0 += ctr[0] * wc;
                    sum1 += ctr[1] * wc;
                    sum2 += ctr[2] * wc;
                }

                float grad_ssim = mask_val * (sum0 + (2.f * p1) * sum1 + p2 * sum2);

                // L1 gradient with mask
                float sign_grad = (p1 == p2) ? 0.0f : copysignf(1.0f, p1 - p2);
                float grad_l1 = l1_weight * sign_grad * mask_val * inv_mask_sum;

                int out_idx = bIdx * CH * num_pix + c * num_pix + pix_id;
                if constexpr (Accumulate)
                    dL_dimg1[out_idx] += grad_ssim + grad_l1;
                else
                    dL_dimg1[out_idx] = grad_ssim + grad_l1;
            }
            block.sync();
        }
    }

    template <typename TargetT>
    __global__ void decoupledFusedL1SSIMForwardCUDA(
        int H,
        int W,
        int CH,
        float C1,
        float C2,
        float ssim_weight,
        const float* __restrict__ corrected_img,
        const float* __restrict__ raw_img,
        const TargetT* __restrict__ gt_img,
        __half* __restrict__ app_dm_dmu1,
        __half* __restrict__ raw_dm_dmu1,
        __half* __restrict__ raw_dm_dsigma1_sq,
        __half* __restrict__ raw_dm_dsigma12,
        float* __restrict__ ssim_map,
        float* __restrict__ cs_map,
        float* __restrict__ ssim_channels) { // optional [N, C, H, W] per-channel SSIM

        auto block = cg::this_thread_block();
        const int bIdx = block.group_index().z;
        const int pix_y = block.group_index().y * BLOCK_Y + block.thread_index().y;
        const int pix_x = block.group_index().x * BLOCK_X + block.thread_index().x;
        const int pix_id = pix_y * W + pix_x;
        const int num_pix = H * W;

        __shared__ float sTile[SHARED_Y][SHARED_X][3];
        // [0]=corrected, [1]=raw, [2]=raw^2, [3]=gt, [4]=gt^2, [5]=raw*gt
        __shared__ float xconv[CONV_Y][CONV_X][6];

        float ssim_sum = 0.0f;
        float cs_sum = 0.0f;
        for (int c = 0; c < CH; ++c) {
            {
                const int tileSize = SHARED_Y * SHARED_X;
                const int threads = BLOCK_X * BLOCK_Y;
                const int steps = (tileSize + threads - 1) / threads;
                const int tileStartY = block.group_index().y * BLOCK_Y;
                const int tileStartX = block.group_index().x * BLOCK_X;

                for (int s = 0; s < steps; ++s) {
                    int tid = s * threads + block.thread_rank();
                    if (tid < tileSize) {
                        int local_y = tid / SHARED_X;
                        int local_x = tid % SHARED_X;
                        int gy = tileStartY + local_y - HALO;
                        int gx = tileStartX + local_x - HALO;

                        sTile[local_y][local_x][0] = get_pix_value(corrected_img, bIdx, c, gy, gx, CH, H, W);
                        sTile[local_y][local_x][1] = get_pix_value(raw_img, bIdx, c, gy, gx, CH, H, W);
                        sTile[local_y][local_x][2] = get_pix_value(gt_img, bIdx, c, gy, gx, CH, H, W);
                    }
                }
            }
            block.sync();

            {
                int ly = threadIdx.y;
                int lx = threadIdx.x + HALO;

                float sum_corrected = 0.f;
                float sum_raw = 0.f;
                float sum_raw_sq = 0.f;
                float sum_gt = 0.f;
                float sum_gt_sq = 0.f;
                float sum_raw_gt = 0.f;

#pragma unroll
                for (int d = 1; d <= HALO; ++d) {
                    const float w = cGauss[HALO - d];

                    const float corrected_left = sTile[ly][lx - d][0];
                    const float corrected_right = sTile[ly][lx + d][0];
                    const float raw_left = sTile[ly][lx - d][1];
                    const float raw_right = sTile[ly][lx + d][1];
                    const float gt_left = sTile[ly][lx - d][2];
                    const float gt_right = sTile[ly][lx + d][2];

                    sum_corrected += (corrected_left + corrected_right) * w;
                    sum_raw += (raw_left + raw_right) * w;
                    sum_raw_sq += ((raw_left * raw_left) + (raw_right * raw_right)) * w;
                    sum_gt += (gt_left + gt_right) * w;
                    sum_gt_sq += ((gt_left * gt_left) + (gt_right * gt_right)) * w;
                    sum_raw_gt += ((raw_left * gt_left) + (raw_right * gt_right)) * w;
                }

                {
                    const float corrected_center = sTile[ly][lx][0];
                    const float raw_center = sTile[ly][lx][1];
                    const float gt_center = sTile[ly][lx][2];
                    const float wc = cGauss[HALO];

                    sum_corrected += corrected_center * wc;
                    sum_raw += raw_center * wc;
                    sum_raw_sq += (raw_center * raw_center) * wc;
                    sum_gt += gt_center * wc;
                    sum_gt_sq += (gt_center * gt_center) * wc;
                    sum_raw_gt += (raw_center * gt_center) * wc;
                }

                xconv[ly][threadIdx.x][0] = sum_corrected;
                xconv[ly][threadIdx.x][1] = sum_raw;
                xconv[ly][threadIdx.x][2] = sum_raw_sq;
                xconv[ly][threadIdx.x][3] = sum_gt;
                xconv[ly][threadIdx.x][4] = sum_gt_sq;
                xconv[ly][threadIdx.x][5] = sum_raw_gt;

                const int ly2 = ly + BLOCK_Y;
                if (ly2 < CONV_Y) {
                    sum_corrected = 0.f;
                    sum_raw = 0.f;
                    sum_raw_sq = 0.f;
                    sum_gt = 0.f;
                    sum_gt_sq = 0.f;
                    sum_raw_gt = 0.f;

#pragma unroll
                    for (int d = 1; d <= HALO; ++d) {
                        const float w = cGauss[HALO - d];

                        const float corrected_left = sTile[ly2][lx - d][0];
                        const float corrected_right = sTile[ly2][lx + d][0];
                        const float raw_left = sTile[ly2][lx - d][1];
                        const float raw_right = sTile[ly2][lx + d][1];
                        const float gt_left = sTile[ly2][lx - d][2];
                        const float gt_right = sTile[ly2][lx + d][2];

                        sum_corrected += (corrected_left + corrected_right) * w;
                        sum_raw += (raw_left + raw_right) * w;
                        sum_raw_sq += ((raw_left * raw_left) + (raw_right * raw_right)) * w;
                        sum_gt += (gt_left + gt_right) * w;
                        sum_gt_sq += ((gt_left * gt_left) + (gt_right * gt_right)) * w;
                        sum_raw_gt += ((raw_left * gt_left) + (raw_right * gt_right)) * w;
                    }

                    {
                        const float corrected_center = sTile[ly2][lx][0];
                        const float raw_center = sTile[ly2][lx][1];
                        const float gt_center = sTile[ly2][lx][2];
                        const float wc = cGauss[HALO];

                        sum_corrected += corrected_center * wc;
                        sum_raw += raw_center * wc;
                        sum_raw_sq += (raw_center * raw_center) * wc;
                        sum_gt += gt_center * wc;
                        sum_gt_sq += (gt_center * gt_center) * wc;
                        sum_raw_gt += (raw_center * gt_center) * wc;
                    }

                    xconv[ly2][threadIdx.x][0] = sum_corrected;
                    xconv[ly2][threadIdx.x][1] = sum_raw;
                    xconv[ly2][threadIdx.x][2] = sum_raw_sq;
                    xconv[ly2][threadIdx.x][3] = sum_gt;
                    xconv[ly2][threadIdx.x][4] = sum_gt_sq;
                    xconv[ly2][threadIdx.x][5] = sum_raw_gt;
                }
            }
            block.sync();

            {
                int ly = threadIdx.y + HALO;
                int lx = threadIdx.x;

                float corrected_conv = 0.f;
                float raw_conv = 0.f;
                float raw_sq_conv = 0.f;
                float gt_conv = 0.f;
                float gt_sq_conv = 0.f;
                float raw_gt_conv = 0.f;

#pragma unroll
                for (int d = 1; d <= HALO; ++d) {
                    const float w = cGauss[HALO - d];
                    float* top = xconv[ly - d][lx];
                    float* bot = xconv[ly + d][lx];

                    corrected_conv += (top[0] + bot[0]) * w;
                    raw_conv += (top[1] + bot[1]) * w;
                    raw_sq_conv += (top[2] + bot[2]) * w;
                    gt_conv += (top[3] + bot[3]) * w;
                    gt_sq_conv += (top[4] + bot[4]) * w;
                    raw_gt_conv += (top[5] + bot[5]) * w;
                }

                {
                    const float wc = cGauss[HALO];
                    float* ctr = xconv[ly][lx];
                    corrected_conv += ctr[0] * wc;
                    raw_conv += ctr[1] * wc;
                    raw_sq_conv += ctr[2] * wc;
                    gt_conv += ctr[3] * wc;
                    gt_sq_conv += ctr[4] * wc;
                    raw_gt_conv += ctr[5] * wc;
                }

                if (pix_x < W && pix_y < H) {
                    const float mu_corrected = corrected_conv;
                    const float mu_raw = raw_conv;
                    const float mu_gt = gt_conv;

                    const float mu_corrected_sq = mu_corrected * mu_corrected;
                    const float mu_raw_sq = mu_raw * mu_raw;
                    const float mu_gt_sq = mu_gt * mu_gt;

                    const float sigma_raw_sq = raw_sq_conv - mu_raw_sq;
                    const float sigma_gt_sq = gt_sq_conv - mu_gt_sq;
                    const float sigma12 = raw_gt_conv - mu_raw * mu_gt;

                    const float A_app = mu_corrected_sq + mu_gt_sq + C1;
                    const float C_app = 2.f * mu_corrected * mu_gt + C1;
                    const float luminance = C_app / A_app;

                    const float B_raw = sigma_raw_sq + sigma_gt_sq + C2;
                    const float D_raw = 2.f * sigma12 + C2;
                    const float contrast_structure = D_raw / B_raw;

                    const int global_idx = bIdx * CH * num_pix + c * num_pix + pix_id;
                    const float ssim_val = luminance * contrast_structure;
                    ssim_sum += ssim_val;
                    if (ssim_channels)
                        ssim_channels[global_idx] = ssim_val;
                    cs_sum += contrast_structure;

                    if (app_dm_dmu1) {
                        const float dl_dmu1 =
                            2.f * (mu_gt * A_app - mu_corrected * C_app) / (A_app * A_app);
                        app_dm_dmu1[global_idx] = static_cast<__half>(contrast_structure * dl_dmu1);

                        const float ds1sq = ssim_weight * (-(luminance * D_raw) / (B_raw * B_raw));
                        const float ds12 = ssim_weight * ((2.f * luminance) / B_raw);
                        raw_dm_dsigma1_sq[global_idx] = static_cast<__half>(ds1sq);
                        raw_dm_dsigma12[global_idx] = static_cast<__half>(ds12);
                        raw_dm_dmu1[global_idx] =
                            static_cast<__half>(ds1sq * (-2.f * mu_raw) + ds12 * (-mu_gt));
                    }
                }
            }
        }

        if (pix_x < W && pix_y < H) {
            const float inv_c = 1.0f / static_cast<float>(CH);
            if (ssim_map) {
                ssim_map[bIdx * num_pix + pix_id] = ssim_sum * inv_c;
            }
            if (cs_map) {
                cs_map[bIdx * num_pix + pix_id] = cs_sum * inv_c;
            }
        }
    }

    // Evaluation needs only the map: normalized masked moments, without backward
    // partials or contrast buffers. Keep the unmasked kernels independent.
    template <typename TargetT, typename MaskT>
    __global__ void maskedSSIMMapCUDA(
        int H, int W, int CH, const float* __restrict__ pred,
        const TargetT* __restrict__ target, const MaskT* __restrict__ mask,
        float* __restrict__ map) {
        const auto block = cg::this_thread_block();
        const int start_y = block.group_index().y * BLOCK_Y;
        const int start_x = block.group_index().x * BLOCK_X;
        const int py = start_y + threadIdx.y, px = start_x + threadIdx.x;
        const int b = block.group_index().z;
        __shared__ float samples[SHARED_Y][SHARED_X][2];
        __shared__ float weights[SHARED_Y][SHARED_X];
        __shared__ float horizontal[CONV_Y][CONV_X][5];
        bool all_one = true;
        for (int i = block.thread_rank(); i < SHARED_Y * SHARED_X; i += BLOCK_X * BLOCK_Y) {
            const int y = i / SHARED_X, x = i % SHARED_X;
            const int gy = start_y + y - HALO, gx = start_x + x - HALO;
            // Byte masks represent binary support, as in the training loss.
            // Preserve zero-valued image padding outside the image with unit support.
            const float value = gx >= 0 && gx < W && gy >= 0 && gy < H
                                    ? mask_value(mask, gy * W + gx)
                                    : 1.0f;
            weights[y][x] = value;
            all_one = all_one && value == 1.0f;
        }
        const bool full_support = __syncthreads_and(all_one);
        const bool kept = px < W && py < H && weights[threadIdx.y + HALO][threadIdx.x + HALO] > 0.0f;
        if (!__syncthreads_or(kept)) {
            if (px < W && py < H)
                for (int c = 0; c < CH; ++c)
                    map[(b * CH + c) * H * W + py * W + px] = 1.0f;
            return;
        }
        float inv_mass = 1.0f;
        if (!full_support) {
            const int x = threadIdx.x + HALO;
            for (int y = threadIdx.y; y < CONV_Y; y += BLOCK_Y) {
                float mass = weights[y][x] * cGauss[HALO];
#pragma unroll
                for (int d = 1; d <= HALO; ++d)
                    mass += (weights[y][x - d] + weights[y][x + d]) * cGauss[HALO - d];
                horizontal[y][threadIdx.x][0] = mass;
            }
            block.sync();
            const int y = threadIdx.y + HALO;
            float mass = horizontal[y][threadIdx.x][0] * cGauss[HALO];
#pragma unroll
            for (int d = 1; d <= HALO; ++d)
                mass += (horizontal[y - d][threadIdx.x][0] + horizontal[y + d][threadIdx.x][0]) * cGauss[HALO - d];
            // Tiny support is neutral, including when subnormals are retained.
            inv_mass = mass >= 1.0e-30f ? 1.0f / mass : 0.0f;
            block.sync();
        }
        for (int c = 0; c < CH; ++c) {
            for (int i = block.thread_rank(); i < SHARED_Y * SHARED_X; i += BLOCK_X * BLOCK_Y) {
                const int y = i / SHARED_X, x = i % SHARED_X;
                const int gy = start_y + y - HALO, gx = start_x + x - HALO;
                const bool supported = weights[y][x] > 0.0f;
                samples[y][x][0] = supported ? get_pix_value(pred, b, c, gy, gx, CH, H, W) : 0.0f;
                samples[y][x][1] = supported ? get_pix_value(target, b, c, gy, gx, CH, H, W) : 0.0f;
            }
            block.sync();
            const int x = threadIdx.x + HALO;
            for (int y = threadIdx.y; y < CONV_Y; y += BLOCK_Y) {
                float sx = 0, sxx = 0, sy = 0, syy = 0, sxy = 0;
#pragma unroll
                for (int d = 1; d <= HALO; ++d) {
                    const float w = cGauss[HALO - d];
                    const float xl = samples[y][x - d][0], xr = samples[y][x + d][0];
                    const float yl = samples[y][x - d][1], yr = samples[y][x + d][1];
                    if (full_support) {
                        sx = __fmaf_rn(xl + xr, w, sx);
                        sxx = __fmaf_rn(xl * xl + xr * xr, w, sxx);
                        sy = __fmaf_rn(yl + yr, w, sy);
                        syy = __fmaf_rn(yl * yl + yr * yr, w, syy);
                        sxy = __fmaf_rn(xl * yl + xr * yr, w, sxy);
                    } else {
                        const float ml = weights[y][x - d] * w, mr = weights[y][x + d] * w;
                        sx += xl * ml + xr * mr;
                        sxx += xl * xl * ml + xr * xr * mr;
                        sy += yl * ml + yr * mr;
                        syy += yl * yl * ml + yr * yr * mr;
                        sxy += xl * yl * ml + xr * yr * mr;
                    }
                }
                const float cx = samples[y][x][0], cy = samples[y][x][1];
                const float wc = full_support ? cGauss[HALO] : cGauss[HALO] * weights[y][x];
                horizontal[y][threadIdx.x][0] = sx + cx * wc;
                horizontal[y][threadIdx.x][1] = sxx + cx * cx * wc;
                horizontal[y][threadIdx.x][2] = sy + cy * wc;
                horizontal[y][threadIdx.x][3] = syy + cy * cy * wc;
                horizontal[y][threadIdx.x][4] = sxy + cx * cy * wc;
            }
            block.sync();
            const int y = threadIdx.y + HALO;
            float moments[5] = {};
#pragma unroll
            for (int d = 1; d <= HALO; ++d)
#pragma unroll
                for (int k = 0; k < 5; ++k)
                    moments[k] += (horizontal[y - d][threadIdx.x][k] + horizontal[y + d][threadIdx.x][k]) * cGauss[HALO - d];
#pragma unroll
            for (int k = 0; k < 5; ++k)
                moments[k] += horizontal[y][threadIdx.x][k] * cGauss[HALO];
            // Preserve the unmasked kernel's FMA rounding with full support.
            if (!full_support) {
#pragma unroll
                for (int k = 0; k < 5; ++k)
                    moments[k] *= inv_mass;
            }
            if (px < W && py < H) {
                const float mx = moments[0], my = moments[2];
                const float xx = mx * mx, yy = my * my;
                const float vx = moments[1] - xx, vy = moments[3] - yy;
                const float xy = moments[4] - mx * my;
                constexpr float C1 = 0.01f * 0.01f, C2 = 0.03f * 0.03f;
                const float A = xx + yy + C1, B = vx + vy + C2;
                const float C = __fmaf_rn(2.0f * mx, my, C1), D = 2.0f * xy + C2;
                map[(b * CH + c) * H * W + py * W + px] = kept ? (C * D) / (A * B) : 1.0f;
            }
            block.sync();
        }
    }

    template <typename Fn>
    void dispatch_target_ptr(const lfs::core::Tensor& target, Fn&& fn) {
        if (target.dtype() == lfs::core::DataType::UInt8) {
            fn(target.ptr<uint8_t>());
        } else {
            fn(target.ptr<float>());
        }
    }

    template <typename Fn>
    void dispatch_mask_ptr(const lfs::core::Tensor& mask, Fn&& fn) {
        if (mask.dtype() == lfs::core::DataType::UInt8 || mask.dtype() == lfs::core::DataType::Bool) {
            fn(mask.ptr<uint8_t>());
        } else {
            fn(mask.ptr<float>());
        }
    }

} // anonymous namespace

// LibTorch-Free API
namespace lfs::training::kernels {

    namespace {
        void validate_ssim_context(const SSIMContext& ctx) {
            validate_loss_context_images(ctx.img1, ctx.img2, ctx.original_h, ctx.original_w);
            const auto validate_derivative = [&](const lfs::core::Tensor& tensor, const std::string_view name) {
                LFS_ASSERT_MSG(tensor.is_valid() && tensor.device() == lfs::core::Device::CUDA &&
                                   tensor.dtype() == lfs::core::DataType::Float16 &&
                                   tensor.is_contiguous() && tensor.shape() == ctx.img1.shape(),
                               lfs::core::detail::format_cuda_safe(
                                   "{} must be a contiguous Float16 CUDA tensor matching {} (shape={})",
                                   name, ctx.img1.shape().str(), tensor.shape().str()));
            };
            validate_derivative(ctx.dm_dmu1, "SSIM dmu derivative");
            validate_derivative(ctx.dm_dsigma1_sq, "SSIM variance derivative");
            validate_derivative(ctx.dm_dsigma12, "SSIM covariance derivative");
        }
    } // namespace

    std::pair<lfs::core::Tensor, SSIMContext> ssim_forward(
        const lfs::core::Tensor& img1_input,
        const lfs::core::Tensor& img2_input,
        bool apply_valid_padding) {

        const float C1 = 0.01f * 0.01f;
        const float C2 = 0.03f * 0.03f;

        auto prepared_images = prepare_loss_images(img1_input, img2_input);
        auto& img1 = prepared_images.prediction;
        auto& img2 = prepared_images.target;

        int N = static_cast<int>(img1.shape()[0]);
        int C = static_cast<int>(img1.shape()[1]);
        int H = static_cast<int>(img1.shape()[2]);
        int W = static_cast<int>(img1.shape()[3]);

        // Launch config
        dim3 grid((W + BLOCK_X - 1) / BLOCK_X,
                  (H + BLOCK_Y - 1) / BLOCK_Y,
                  N);
        dim3 block(BLOCK_X, BLOCK_Y);

        // Output SSIM map
        auto ssim_map = lfs::core::Tensor::zeros(img1.shape(), lfs::core::Device::CUDA);

        auto dm_dmu1 = lfs::core::Tensor::zeros(img1.shape(), lfs::core::Device::CUDA,
                                                lfs::core::DataType::Float16);
        auto dm_dsigma1_sq = lfs::core::Tensor::zeros(img1.shape(), lfs::core::Device::CUDA,
                                                      lfs::core::DataType::Float16);
        auto dm_dsigma12 = lfs::core::Tensor::zeros(img1.shape(), lfs::core::Device::CUDA,
                                                    lfs::core::DataType::Float16);

        dispatch_target_ptr(img2, [&](auto* img2_ptr) {
            using TargetT = std::remove_cv_t<std::remove_pointer_t<decltype(img2_ptr)>>;
            fusedssimCUDA<TargetT><<<grid, block, 0, lfs::core::getCurrentCUDAStream()>>>(
                H, W, C, C1, C2,
                img1.ptr<float>(),
                img2_ptr,
                ssim_map.ptr<float>(),
                /*cs_map=*/nullptr,
                dm_dmu1.ptr<__half>(),
                dm_dsigma1_sq.ptr<__half>(),
                dm_dsigma12.ptr<__half>());
            LFS_CUDA_LAUNCH_CHECK(lfs::core::getCurrentCUDAStream(), "training.ssim.forward");
        });

        // Store original dimensions
        int h = H;
        int w = W;

        // Apply valid padding (crop 5 pixels from each side) using efficient view slicing
        // Then compute mean using optimized tensor reduction (matches PyTorch speed!)
        lfs::core::Tensor ssim_map_cropped = ssim_map;
        if (apply_valid_padding) {
            if (H > 10) {
                ssim_map_cropped = ssim_map_cropped.slice(2, 5, H - 5);
            }
            if (W > 10) {
                ssim_map_cropped = ssim_map_cropped.slice(3, 5, W - 5);
            }
        }

        // Use tensor library's optimized mean (warp reductions + vectorized loads)
        // CRITICAL FIX: Return Tensor (on GPU) instead of syncing to CPU with .item<float>()!
        lfs::core::Tensor ssim_value_tensor = ssim_map_cropped.mean(); // Keep on GPU!

        // Save context for backward
        SSIMContext ctx;
        ctx.img1 = img1;
        ctx.img2 = img2;
        ctx.dm_dmu1 = dm_dmu1;
        ctx.dm_dsigma1_sq = dm_dsigma1_sq;
        ctx.dm_dsigma12 = dm_dsigma12;
        ctx.original_h = h;
        ctx.original_w = w;
        ctx.apply_valid_padding = apply_valid_padding;

        return {ssim_value_tensor, ctx};
    }

    SSIMMapResult ssim_forward_map(
        const lfs::core::Tensor& img1_input,
        const lfs::core::Tensor& img2_input,
        const bool apply_valid_padding) {

        constexpr float C1 = 0.01f * 0.01f;
        constexpr float C2 = 0.03f * 0.03f;

        auto prepared_images = prepare_loss_images(img1_input, img2_input);
        auto& img1 = prepared_images.prediction;
        auto& img2 = prepared_images.target;

        const int N = static_cast<int>(img1.shape()[0]);
        const int C = static_cast<int>(img1.shape()[1]);
        const int H = static_cast<int>(img1.shape()[2]);
        const int W = static_cast<int>(img1.shape()[3]);

        const dim3 grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, N);
        const dim3 block(BLOCK_X, BLOCK_Y);

        auto ssim_map = lfs::core::Tensor::zeros(img1.shape(), lfs::core::Device::CUDA);
        auto cs_map = lfs::core::Tensor::zeros(img1.shape(), lfs::core::Device::CUDA);
        // dm_* fp16 (map path still materializes them for SSIMContext).
        auto dm_dmu1 = lfs::core::Tensor::zeros(img1.shape(), lfs::core::Device::CUDA,
                                                lfs::core::DataType::Float16);
        auto dm_dsigma1_sq = lfs::core::Tensor::zeros(img1.shape(), lfs::core::Device::CUDA,
                                                      lfs::core::DataType::Float16);
        auto dm_dsigma12 = lfs::core::Tensor::zeros(img1.shape(), lfs::core::Device::CUDA,
                                                    lfs::core::DataType::Float16);

        dispatch_target_ptr(img2, [&](auto* img2_ptr) {
            using TargetT = std::remove_cv_t<std::remove_pointer_t<decltype(img2_ptr)>>;
            fusedssimCUDA<TargetT><<<grid, block, 0, lfs::core::getCurrentCUDAStream()>>>(
                H, W, C, C1, C2,
                img1.ptr<float>(), img2_ptr,
                ssim_map.ptr<float>(), cs_map.ptr<float>(),
                dm_dmu1.ptr<__half>(),
                dm_dsigma1_sq.ptr<__half>(), dm_dsigma12.ptr<__half>());
            LFS_CUDA_LAUNCH_CHECK(lfs::core::getCurrentCUDAStream(), "training.ssim.forward_map");
        });

        lfs::core::Tensor ssim_map_for_mean = ssim_map;
        if (apply_valid_padding) {
            if (H > 10) {
                ssim_map_for_mean = ssim_map_for_mean.slice(2, 5, H - 5);
            }
            if (W > 10) {
                ssim_map_for_mean = ssim_map_for_mean.slice(3, 5, W - 5);
            }
        }

        return SSIMMapResult{
            .ssim_map = ssim_map,
            .cs_map = cs_map,
            .ssim_value = ssim_map_for_mean.mean(),
            .ctx = SSIMContext{
                .img1 = img1,
                .img2 = img2,
                .dm_dmu1 = dm_dmu1,
                .dm_dsigma1_sq = dm_dsigma1_sq,
                .dm_dsigma12 = dm_dsigma12,
                .original_h = H,
                .original_w = W,
                .apply_valid_padding = apply_valid_padding}};
    }

    lfs::core::Tensor masked_ssim_map_forward(
        const lfs::core::Tensor& pred_input, const lfs::core::Tensor& target_input,
        const lfs::core::Tensor& mask_input) {
        auto images = prepare_loss_images(pred_input, target_input);
        auto& pred = images.prediction;
        auto& target = images.target;
        const auto dtype = mask_input.dtype();
        const auto mask = prepare_loss_mask(
            mask_input.to(lfs::core::Device::CUDA).to(dtype == lfs::core::DataType::UInt8 || dtype == lfs::core::DataType::Bool ? dtype : lfs::core::DataType::Float32), pred);
        auto map = lfs::core::Tensor::empty(pred.shape(), lfs::core::Device::CUDA);
        const int N = pred.shape()[0], C = pred.shape()[1], H = pred.shape()[2], W = pred.shape()[3];
        const dim3 grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, N);
        const dim3 block(BLOCK_X, BLOCK_Y);
        dispatch_target_ptr(target, [&](auto* target_ptr) {
            using TargetT = std::remove_cv_t<std::remove_pointer_t<decltype(target_ptr)>>;
            dispatch_mask_ptr(mask, [&](auto* mask_ptr) {
                using MaskT = std::remove_cv_t<std::remove_pointer_t<decltype(mask_ptr)>>;
                maskedSSIMMapCUDA<TargetT, MaskT><<<grid, block, 0, lfs::core::getCurrentCUDAStream()>>>(
                    H, W, C, pred.ptr<float>(), target_ptr, mask_ptr, map.ptr<float>());
                LFS_CUDA_LAUNCH_CHECK(lfs::core::getCurrentCUDAStream(), "training.ssim.masked_metric_map");
            });
        });
        return map;
    }

    void ssim_error_map_forward(
        const lfs::core::Tensor& img1_input,
        const lfs::core::Tensor& img2_input,
        SSIMMapWorkspace& workspace,
        lfs::core::Tensor& error_map,
        bool contrast_structure_only) {

        constexpr float C1 = 0.01f * 0.01f;
        constexpr float C2 = 0.03f * 0.03f;

        auto prepared_images = prepare_loss_images(img1_input, img2_input);
        auto& img1 = prepared_images.prediction;
        auto& img2 = prepared_images.target;
        LFS_ASSERT_MSG(img1.shape()[0] == 1,
                       lfs::core::detail::format_cuda_safe(
                           "SSIM error maps require a single-image batch (shape={})",
                           img1.shape().str()));

        const int N = static_cast<int>(img1.shape()[0]);
        const int C = static_cast<int>(img1.shape()[1]);
        const int H = static_cast<int>(img1.shape()[2]);
        const int W = static_cast<int>(img1.shape()[3]);

        workspace.ensure_size(img1.shape().dims());

        const dim3 grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, N);
        const dim3 block(BLOCK_X, BLOCK_Y);

        dispatch_target_ptr(img2, [&](auto* img2_ptr) {
            using TargetT = std::remove_cv_t<std::remove_pointer_t<decltype(img2_ptr)>>;
            fusedssimCUDA<TargetT><<<grid, block, 0, lfs::core::getCurrentCUDAStream()>>>(
                H, W, C, C1, C2,
                img1.ptr<float>(), img2_ptr,
                workspace.ssim_map.ptr<float>(),
                workspace.cs_map.ptr<float>(),
                static_cast<__half*>(nullptr), static_cast<__half*>(nullptr),
                static_cast<__half*>(nullptr));
            LFS_CUDA_LAUNCH_CHECK(lfs::core::getCurrentCUDAStream(), "training.ssim.error_map_forward");
        });

        if (!error_map.is_valid() ||
            error_map.device() != lfs::core::Device::CUDA ||
            error_map.dtype() != lfs::core::DataType::Float32 ||
            !error_map.is_contiguous() ||
            error_map.ndim() != 2 ||
            error_map.shape()[0] != static_cast<size_t>(H) ||
            error_map.shape()[1] != static_cast<size_t>(W)) {
            error_map = lfs::core::Tensor::empty({static_cast<size_t>(H), static_cast<size_t>(W)},
                                                 lfs::core::Device::CUDA);
        }

        launch_ssim_to_error_map(
            contrast_structure_only ? workspace.cs_map : workspace.ssim_map, error_map);
    }

    lfs::core::Tensor ssim_backward(
        const SSIMContext& ctx,
        float grad_loss) {

        validate_ssim_context(ctx);
        LFS_ASSERT_MSG(std::isfinite(grad_loss), "SSIM loss gradient must be finite");

        const float C1 = 0.01f * 0.01f;
        const float C2 = 0.03f * 0.03f;

        // Compute gradient map size (after cropping if applicable)
        int grad_h = ctx.original_h;
        int grad_w = ctx.original_w;
        size_t N = ctx.img1.shape()[0];
        size_t C = ctx.img1.shape()[1];
        size_t numel = N * C * grad_h * grad_w;

        if (ctx.apply_valid_padding) {
            grad_h = grad_h > 10 ? grad_h - 10 : grad_h;
            grad_w = grad_w > 10 ? grad_w - 10 : grad_w;
            numel = N * C * grad_h * grad_w;
        }

        // Create gradient map: d(loss)/d(ssim_scalar) = grad_loss
        // d(ssim_scalar)/d(ssim_map[i]) = 1/numel
        // So: d(loss)/d(ssim_map[i]) = grad_loss / numel
        float grad_per_pixel = grad_loss / static_cast<float>(numel);

        // Create gradient tensor for cropped region
        auto dL_dmap = lfs::core::Tensor::zeros(ctx.img1.shape(), lfs::core::Device::CUDA);

        auto cropped_view = dL_dmap;
        if (ctx.apply_valid_padding && ctx.original_h > 10) {
            cropped_view = cropped_view.slice(2, 5, ctx.original_h - 5);
        }
        if (ctx.apply_valid_padding && ctx.original_w > 10) {
            cropped_view = cropped_view.slice(3, 5, ctx.original_w - 5);
        }
        cropped_view.fill_(grad_per_pixel, nullptr);

        // Allocate output gradient
        auto dL_dimg1 = lfs::core::Tensor::zeros(ctx.img1.shape(), lfs::core::Device::CUDA);

        // Launch backward kernel
        dim3 grid((ctx.original_w + BLOCK_X - 1) / BLOCK_X,
                  (ctx.original_h + BLOCK_Y - 1) / BLOCK_Y,
                  N);
        dim3 block(BLOCK_X, BLOCK_Y);

        dispatch_target_ptr(ctx.img2, [&](auto* img2_ptr) {
            using TargetT = std::remove_cv_t<std::remove_pointer_t<decltype(img2_ptr)>>;
            fusedssim_backwardCUDA<TargetT><<<grid, block, 0, lfs::core::getCurrentCUDAStream()>>>(
                ctx.original_h, ctx.original_w, static_cast<int>(C), C1, C2,
                ctx.img1.ptr<float>(),
                img2_ptr,
                dL_dmap.ptr<float>(),
                dL_dimg1.ptr<float>(),
                ctx.dm_dmu1.ptr<__half>(),
                ctx.dm_dsigma1_sq.ptr<__half>(),
                ctx.dm_dsigma12.ptr<__half>());
            LFS_CUDA_LAUNCH_CHECK(lfs::core::getCurrentCUDAStream(), "training.ssim.backward");
        });

        return dL_dimg1;
    }

    lfs::core::Tensor ssim_backward_with_grad_map(
        const SSIMContext& ctx,
        const lfs::core::Tensor& dL_dmap) {

        validate_ssim_context(ctx);
        auto gradient_map = prepare_loss_prediction(dL_dmap, "SSIM gradient map");
        LFS_ASSERT_MSG(gradient_map.shape() == ctx.img1.shape(),
                       lfs::core::detail::format_cuda_safe(
                           "SSIM gradient map must match the context image (gradient={}, image={})",
                           gradient_map.shape().str(), ctx.img1.shape().str()));

        constexpr float C1 = 0.01f * 0.01f;
        constexpr float C2 = 0.03f * 0.03f;
        const size_t N = ctx.img1.shape()[0];
        const size_t C = ctx.img1.shape()[1];

        auto dL_dimg1 = lfs::core::Tensor::zeros(ctx.img1.shape(), lfs::core::Device::CUDA);
        const dim3 grid((ctx.original_w + BLOCK_X - 1) / BLOCK_X,
                        (ctx.original_h + BLOCK_Y - 1) / BLOCK_Y, N);
        const dim3 block(BLOCK_X, BLOCK_Y);

        dispatch_target_ptr(ctx.img2, [&](auto* img2_ptr) {
            using TargetT = std::remove_cv_t<std::remove_pointer_t<decltype(img2_ptr)>>;
            fusedssim_backwardCUDA<TargetT><<<grid, block, 0, lfs::core::getCurrentCUDAStream()>>>(
                ctx.original_h, ctx.original_w, static_cast<int>(C), C1, C2,
                ctx.img1.ptr<float>(), img2_ptr, gradient_map.ptr<float>(),
                dL_dimg1.ptr<float>(), ctx.dm_dmu1.ptr<__half>(),
                ctx.dm_dsigma1_sq.ptr<__half>(), ctx.dm_dsigma12.ptr<__half>());
            LFS_CUDA_LAUNCH_CHECK(lfs::core::getCurrentCUDAStream(), "training.ssim.backward_grad_map");
        });

        return dL_dimg1;
    }

    // Version with pre-allocated workspace
    std::pair<lfs::core::Tensor, SSIMContext> ssim_forward(
        const lfs::core::Tensor& img1_input,
        const lfs::core::Tensor& img2_input,
        SSIMWorkspace& workspace,
        bool apply_valid_padding) {

        const float C1 = 0.01f * 0.01f;
        const float C2 = 0.03f * 0.03f;

        auto prepared_images = prepare_loss_images(img1_input, img2_input);
        auto& img1 = prepared_images.prediction;
        auto& img2 = prepared_images.target;

        int N = static_cast<int>(img1.shape()[0]);
        int C = static_cast<int>(img1.shape()[1]);
        int H = static_cast<int>(img1.shape()[2]);
        int W = static_cast<int>(img1.shape()[3]);

        // Ensure workspace is sized correctly (only reallocates if shape changed)
        workspace.ensure_size(img1.shape().dims());

        // Launch config
        dim3 grid((W + BLOCK_X - 1) / BLOCK_X,
                  (H + BLOCK_Y - 1) / BLOCK_Y,
                  N);
        dim3 block(BLOCK_X, BLOCK_Y);

        // Use pre-allocated workspace buffers (zero them out)
        workspace.ssim_map.zero_();
        workspace.dm_dmu1.zero_();
        workspace.dm_dsigma1_sq.zero_();
        workspace.dm_dsigma12.zero_();

        dispatch_target_ptr(img2, [&](auto* img2_ptr) {
            using TargetT = std::remove_cv_t<std::remove_pointer_t<decltype(img2_ptr)>>;
            fusedssimCUDA<TargetT><<<grid, block, 0, lfs::core::getCurrentCUDAStream()>>>(
                H, W, C, C1, C2,
                img1.ptr<float>(),
                img2_ptr,
                workspace.ssim_map.ptr<float>(),
                workspace.cs_map.ptr<float>(),
                workspace.dm_dmu1.ptr<__half>(),
                workspace.dm_dsigma1_sq.ptr<__half>(),
                workspace.dm_dsigma12.ptr<__half>());
            LFS_CUDA_LAUNCH_CHECK(lfs::core::getCurrentCUDAStream(), "training.ssim.forward");
        });

        // Store original dimensions
        int h = H;
        int w = W;

        launch_fused_ssim_mean_device(
            workspace.ssim_map.ptr<float>(),
            workspace.reduction_temp.ptr<float>(),
            workspace.reduction_result.ptr<float>(),
            N, C, H, W,
            apply_valid_padding,
            workspace.ssim_map.stream());
        // The returned scalar aliases the workspace and must be consumed before
        // the next forward using this workspace.
        lfs::core::Tensor ssim_value_tensor = workspace.reduction_result;

        // Save context for backward (reference workspace buffers, not copies!)
        SSIMContext ctx;
        ctx.img1 = img1;
        ctx.img2 = img2;
        ctx.dm_dmu1 = workspace.dm_dmu1; // Reference to workspace
        ctx.dm_dsigma1_sq = workspace.dm_dsigma1_sq;
        ctx.dm_dsigma12 = workspace.dm_dsigma12;
        ctx.original_h = h;
        ctx.original_w = w;
        ctx.apply_valid_padding = apply_valid_padding;

        return {ssim_value_tensor, ctx};
    }

    // Optimized version with pre-allocated workspace
    lfs::core::Tensor ssim_backward(
        const SSIMContext& ctx,
        SSIMWorkspace& workspace,
        float grad_loss) {

        validate_ssim_context(ctx);
        LFS_ASSERT_MSG(std::isfinite(grad_loss), "SSIM loss gradient must be finite");
        workspace.ensure_size(ctx.img1.shape().dims());

        const float C1 = 0.01f * 0.01f;
        const float C2 = 0.03f * 0.03f;

        // Compute gradient map size (after cropping if applicable)
        int grad_h = ctx.original_h;
        int grad_w = ctx.original_w;
        size_t N = ctx.img1.shape()[0];
        size_t C = ctx.img1.shape()[1];
        size_t numel = N * C * grad_h * grad_w;

        if (ctx.apply_valid_padding) {
            grad_h = grad_h > 10 ? grad_h - 10 : grad_h;
            grad_w = grad_w > 10 ? grad_w - 10 : grad_w;
            numel = N * C * grad_h * grad_w;
        }

        float grad_per_pixel = grad_loss / static_cast<float>(numel);

        // Use pre-allocated workspace buffer
        workspace.dL_dmap.zero_();

        auto cropped_view = workspace.dL_dmap;
        if (ctx.apply_valid_padding && ctx.original_h > 10) {
            cropped_view = cropped_view.slice(2, 5, ctx.original_h - 5);
        }
        if (ctx.apply_valid_padding && ctx.original_w > 10) {
            cropped_view = cropped_view.slice(3, 5, ctx.original_w - 5);
        }
        cropped_view.fill_(grad_per_pixel, nullptr);

        // Use pre-allocated output buffer
        workspace.dL_dimg1.zero_();

        // Launch backward kernel
        dim3 grid((ctx.original_w + BLOCK_X - 1) / BLOCK_X,
                  (ctx.original_h + BLOCK_Y - 1) / BLOCK_Y,
                  N);
        dim3 block(BLOCK_X, BLOCK_Y);

        dispatch_target_ptr(ctx.img2, [&](auto* img2_ptr) {
            using TargetT = std::remove_cv_t<std::remove_pointer_t<decltype(img2_ptr)>>;
            fusedssim_backwardCUDA<TargetT><<<grid, block, 0, lfs::core::getCurrentCUDAStream()>>>(
                ctx.original_h, ctx.original_w, static_cast<int>(C), C1, C2,
                ctx.img1.ptr<float>(),
                img2_ptr,
                workspace.dL_dmap.ptr<float>(),
                workspace.dL_dimg1.ptr<float>(),
                ctx.dm_dmu1.ptr<__half>(),
                ctx.dm_dsigma1_sq.ptr<__half>(),
                ctx.dm_dsigma12.ptr<__half>());
            LFS_CUDA_LAUNCH_CHECK(lfs::core::getCurrentCUDAStream(), "training.ssim.backward");
        });

        return workspace.dL_dimg1;
    }

    // ============================================================================
    // Fused L1+SSIM Implementation
    // ============================================================================

    std::pair<lfs::core::Tensor, FusedL1SSIMContext> fused_l1_ssim_forward(
        const lfs::core::Tensor& img1_input,
        const lfs::core::Tensor& img2_input,
        float ssim_weight,
        FusedL1SSIMWorkspace& workspace,
        bool apply_valid_padding) {

        constexpr float C1 = 0.01f * 0.01f;
        constexpr float C2 = 0.03f * 0.03f;

        validate_loss_weight(ssim_weight);
        auto prepared_images = prepare_loss_images(img1_input, img2_input);
        auto& img1 = prepared_images.prediction;
        auto& img2 = prepared_images.target;

        const int N = static_cast<int>(img1.shape()[0]);
        const int C = static_cast<int>(img1.shape()[1]);
        const int H = static_cast<int>(img1.shape()[2]);
        const int W = static_cast<int>(img1.shape()[3]);

        workspace.ensure_size(img1.shape().dims());

        const dim3 grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, N);
        const dim3 block(BLOCK_X, BLOCK_Y);

        dispatch_target_ptr(img2, [&](auto* img2_ptr) {
            using TargetT = std::remove_cv_t<std::remove_pointer_t<decltype(img2_ptr)>>;
            fusedL1SSIMForwardCUDA<TargetT><<<grid, block, 0, lfs::core::getCurrentCUDAStream()>>>(
                H, W, C, C1, C2,
                img1.ptr<float>(), img2_ptr,
                workspace.dm_dmu1.ptr<__half>(),
                workspace.dm_dsigma1_sq.ptr<__half>(),
                workspace.dm_dsigma12.ptr<__half>(),
                workspace.ssim_map.ptr<float>(),
                workspace.cs_map.ptr<float>(),
                /*ssim_channels=*/nullptr);
            LFS_CUDA_LAUNCH_CHECK(lfs::core::getCurrentCUDAStream(), "training.ssim.fused_l1_forward");

            launch_fused_l1_ssim_mean_device(
                img1.ptr<float>(),
                img2_ptr,
                workspace.ssim_map.ptr<float>(),
                ssim_weight,
                workspace.reduction_temp.ptr<float>(),
                workspace.reduction_result.ptr<float>(),
                N, C, H, W,
                apply_valid_padding,
                workspace.ssim_map.stream());
        });
        lfs::core::Tensor loss_scalar = workspace.reduction_result;

        FusedL1SSIMContext ctx{
            .img1 = img1,
            .img2 = img2,
            .dm_dmu1 = workspace.dm_dmu1,
            .dm_dsigma1_sq = workspace.dm_dsigma1_sq,
            .dm_dsigma12 = workspace.dm_dsigma12,
            .ssim_weight = ssim_weight,
            .H = H,
            .W = W,
            .apply_valid_padding = apply_valid_padding};

        return {loss_scalar, ctx};
    }

    lfs::core::Tensor fused_l1_ssim_backward(
        const FusedL1SSIMContext& ctx,
        FusedL1SSIMWorkspace& workspace) {

        validate_loss_context_images(ctx.img1, ctx.img2, ctx.H, ctx.W);
        validate_loss_weight(ctx.ssim_weight);
        workspace.ensure_size(ctx.img1.shape().dims());

        constexpr float C1 = 0.01f * 0.01f;
        constexpr float C2 = 0.03f * 0.03f;

        const size_t N = ctx.img1.shape()[0];
        const size_t C = ctx.img1.shape()[1];

        // Compute gradient normalization factor
        int grad_h = ctx.H;
        int grad_w = ctx.W;
        if (ctx.apply_valid_padding) {
            grad_h = grad_h > 10 ? grad_h - 10 : grad_h;
            grad_w = grad_w > 10 ? grad_w - 10 : grad_w;
        }
        const size_t numel = N * C * grad_h * grad_w;
        const float grad_per_pixel = 1.0f / static_cast<float>(numel);

        const dim3 grid((ctx.W + BLOCK_X - 1) / BLOCK_X, (ctx.H + BLOCK_Y - 1) / BLOCK_Y, N);
        const dim3 block(BLOCK_X, BLOCK_Y);

        dispatch_target_ptr(ctx.img2, [&](auto* img2_ptr) {
            using TargetT = std::remove_cv_t<std::remove_pointer_t<decltype(img2_ptr)>>;
            fusedL1SSIMBackwardCUDA<TargetT><<<grid, block, 0, lfs::core::getCurrentCUDAStream()>>>(
                ctx.ssim_weight, ctx.H, ctx.W, static_cast<int>(C), C1, C2,
                grad_per_pixel, ctx.apply_valid_padding,
                ctx.img1.ptr<float>(), img2_ptr,
                workspace.grad_img.ptr<float>(),
                ctx.dm_dmu1.ptr<__half>(), ctx.dm_dsigma1_sq.ptr<__half>(),
                ctx.dm_dsigma12.ptr<__half>());
            LFS_CUDA_LAUNCH_CHECK(lfs::core::getCurrentCUDAStream(), "training.ssim.fused_l1_backward");
        });

        return workspace.grad_img;
    }

    // ============================================================================
    // Decoupled Fused L1+SSIM Implementation
    // ============================================================================

    std::pair<lfs::core::Tensor, DecoupledFusedL1SSIMContext> decoupled_fused_l1_ssim_forward(
        const lfs::core::Tensor& corrected_input,
        const lfs::core::Tensor& raw_input,
        const lfs::core::Tensor& gt_input,
        float ssim_weight,
        DecoupledFusedL1SSIMWorkspace& workspace,
        bool apply_valid_padding) {

        constexpr float C1 = 0.01f * 0.01f;
        constexpr float C2 = 0.03f * 0.03f;

        validate_loss_weight(ssim_weight);
        auto corrected = prepare_loss_prediction(corrected_input, "Corrected loss image");
        auto raw = prepare_loss_prediction(raw_input, "Raw loss image");
        auto gt = prepare_loss_target(gt_input, "Loss target");
        LFS_ASSERT_MSG(corrected.shape() == raw.shape() && corrected.shape() == gt.shape(),
                       lfs::core::detail::format_cuda_safe(
                           "Decoupled loss image shapes must match (corrected={}, raw={}, target={})",
                           corrected.shape().str(), raw.shape().str(), gt.shape().str()));

        const int N = static_cast<int>(corrected.shape()[0]);
        const int C = static_cast<int>(corrected.shape()[1]);
        const int H = static_cast<int>(corrected.shape()[2]);
        const int W = static_cast<int>(corrected.shape()[3]);

        workspace.ensure_size(corrected.shape().dims());

        const dim3 grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, N);
        const dim3 block(BLOCK_X, BLOCK_Y);

        dispatch_target_ptr(gt, [&](auto* gt_ptr) {
            using TargetT = std::remove_cv_t<std::remove_pointer_t<decltype(gt_ptr)>>;
            decoupledFusedL1SSIMForwardCUDA<TargetT><<<grid, block, 0, lfs::core::getCurrentCUDAStream()>>>(
                H, W, C, C1, C2, ssim_weight,
                corrected.ptr<float>(), raw.ptr<float>(), gt_ptr,
                workspace.app_dm_dmu1.ptr<__half>(),
                workspace.raw_dm_dmu1.ptr<__half>(),
                workspace.raw_dm_dsigma1_sq.ptr<__half>(),
                workspace.raw_dm_dsigma12.ptr<__half>(),
                workspace.ssim_map.ptr<float>(),
                workspace.cs_map.ptr<float>(), nullptr);
            LFS_CUDA_LAUNCH_CHECK(lfs::core::getCurrentCUDAStream(), "training.ssim.decoupled_fused_l1_forward");

            launch_fused_l1_ssim_mean_device(
                corrected.ptr<float>(),
                gt_ptr,
                workspace.ssim_map.ptr<float>(),
                ssim_weight,
                workspace.reduction_temp.ptr<float>(),
                workspace.reduction_result.ptr<float>(),
                N, C, H, W,
                apply_valid_padding,
                workspace.ssim_map.stream());
        });
        lfs::core::Tensor loss_scalar = workspace.reduction_result;

        DecoupledFusedL1SSIMContext ctx{
            .corrected_img = corrected,
            .raw_img = raw,
            .gt_img = gt,
            .app_dm_dmu1 = workspace.app_dm_dmu1,
            .raw_dm_dmu1 = workspace.raw_dm_dmu1,
            .raw_dm_dsigma1_sq = workspace.raw_dm_dsigma1_sq,
            .raw_dm_dsigma12 = workspace.raw_dm_dsigma12,
            .ssim_weight = ssim_weight,
            .H = H,
            .W = W,
            .apply_valid_padding = apply_valid_padding};

        return {loss_scalar, ctx};
    }

    DecoupledGradients decoupled_fused_l1_ssim_backward(
        const DecoupledFusedL1SSIMContext& ctx,
        DecoupledFusedL1SSIMWorkspace& workspace) {

        validate_loss_context_images(ctx.corrected_img, ctx.gt_img, ctx.H, ctx.W);
        validate_loss_context_images(ctx.raw_img, ctx.gt_img, ctx.H, ctx.W);
        validate_loss_weight(ctx.ssim_weight);
        workspace.ensure_size(ctx.corrected_img.shape().dims());

        constexpr float C1 = 0.01f * 0.01f;
        constexpr float C2 = 0.03f * 0.03f;

        const size_t N = ctx.corrected_img.shape()[0];
        const size_t C = ctx.corrected_img.shape()[1];

        int grad_h = ctx.H;
        int grad_w = ctx.W;
        if (ctx.apply_valid_padding) {
            grad_h = grad_h > 10 ? grad_h - 10 : grad_h;
            grad_w = grad_w > 10 ? grad_w - 10 : grad_w;
        }
        const size_t numel = N * C * grad_h * grad_w;
        const float grad_per_pixel = 1.0f / static_cast<float>(numel);

        const dim3 grid((ctx.W + BLOCK_X - 1) / BLOCK_X, (ctx.H + BLOCK_Y - 1) / BLOCK_Y, N);
        const dim3 block(BLOCK_X, BLOCK_Y);

        dispatch_target_ptr(ctx.gt_img, [&](auto* gt_ptr) {
            using TargetT = std::remove_cv_t<std::remove_pointer_t<decltype(gt_ptr)>>;
            fusedL1SSIMBackwardCUDA<TargetT, /*HasSigmaPartials=*/false>
                <<<grid, block, 0, lfs::core::getCurrentCUDAStream()>>>(
                    ctx.ssim_weight, ctx.H, ctx.W, static_cast<int>(C), C1, C2,
                    grad_per_pixel, ctx.apply_valid_padding,
                    ctx.corrected_img.ptr<float>(), gt_ptr,
                    workspace.grad_corrected.ptr<float>(),
                    ctx.app_dm_dmu1.ptr<__half>(),
                    /*dm_dsigma1_sq=*/nullptr,
                    /*dm_dsigma12=*/nullptr);
            LFS_CUDA_LAUNCH_CHECK(lfs::core::getCurrentCUDAStream(), "training.ssim.decoupled_fused_l1_backward");
        });

        return DecoupledGradients{.grad_corrected = workspace.grad_corrected};
    }

    DecoupledRawGradient decoupled_raw_gradient(const DecoupledFusedL1SSIMContext& ctx) {
        return DecoupledRawGradient{
            .raw_img = ctx.raw_img,
            .gt_img = ctx.gt_img,
            .raw_dm_dmu1 = ctx.raw_dm_dmu1,
            .raw_dm_dsigma1_sq = ctx.raw_dm_dsigma1_sq,
            .raw_dm_dsigma12 = ctx.raw_dm_dsigma12,
            .H = ctx.H,
            .W = ctx.W,
            .apply_valid_padding = ctx.apply_valid_padding};
    }

    DecoupledRawGradient decoupled_raw_gradient(const MaskedDecoupledFusedL1SSIMContext& ctx) {
        return DecoupledRawGradient{
            .raw_img = ctx.raw_img,
            .gt_img = ctx.gt_img,
            .mask = ctx.mask,
            .raw_dm_dmu1 = ctx.raw_dm_dmu1,
            .raw_dm_dsigma1_sq = ctx.raw_dm_dsigma1_sq,
            .raw_dm_dsigma12 = ctx.raw_dm_dsigma12,
            .mask_sum_value = ctx.mask_sum_value,
            .H = ctx.H,
            .W = ctx.W};
    }

    namespace {
        __global__ void add_gradient_kernel(float* __restrict__ dst, const float* __restrict__ src, const size_t n) {
            const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (idx < n)
                dst[idx] += src[idx];
        }
    } // namespace

    void accumulate_decoupled_raw_gradient(const DecoupledRawGradient& raw, lfs::core::Tensor& grad_image) {
        LFS_ASSERT(grad_image.is_contiguous() && grad_image.dtype() == lfs::core::DataType::Float32 &&
                   grad_image.numel() == raw.raw_img.numel());
        if (raw.precomputed.is_valid()) {
            LFS_ASSERT(raw.precomputed.is_contiguous() && raw.precomputed.dtype() == lfs::core::DataType::Float32 &&
                       raw.precomputed.numel() == grad_image.numel());
            const size_t n = grad_image.numel();
            const auto stream = lfs::core::getCurrentCUDAStream();
            add_gradient_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(
                grad_image.ptr<float>(), raw.precomputed.ptr<float>(), n);
            LFS_CUDA_LAUNCH_CHECK(stream, "training.ssim.decoupled_raw_precomputed");
            return;
        }
        validate_loss_context_images(raw.raw_img, raw.gt_img, raw.H, raw.W);

        constexpr float C1 = 0.01f * 0.01f;
        constexpr float C2 = 0.03f * 0.03f;

        const size_t N = raw.raw_img.shape()[0];
        const size_t C = raw.raw_img.shape()[1];
        const dim3 grid((raw.W + BLOCK_X - 1) / BLOCK_X, (raw.H + BLOCK_Y - 1) / BLOCK_Y, N);
        const dim3 block(BLOCK_X, BLOCK_Y);

        if (raw.mask.is_valid()) {
            LFS_ASSERT_MSG(std::isfinite(raw.mask_sum_value) && raw.mask_sum_value > 0.0f,
                           "Masked loss normalization must be positive and finite");
            const auto mask = prepare_loss_mask(raw.mask, raw.raw_img);
            const float inv_mask_sum = 1.0f / raw.mask_sum_value;
            dispatch_target_ptr(raw.gt_img, [&](auto* gt_ptr) {
                using TargetT = std::remove_cv_t<std::remove_pointer_t<decltype(gt_ptr)>>;
                dispatch_mask_ptr(mask, [&](auto* mask_ptr) {
                    using MaskT = std::remove_cv_t<std::remove_pointer_t<decltype(mask_ptr)>>;
                    maskedFusedL1SSIMBackwardCUDA<TargetT, MaskT, /*HasSigmaPartials=*/true, /*Accumulate=*/true, /*ScaledMeanPartial=*/true>
                        <<<grid, block, 0, lfs::core::getCurrentCUDAStream()>>>(
                            1.0f, inv_mask_sum, raw.H, raw.W, static_cast<int>(C), C1, C2,
                            raw.raw_img.ptr<float>(), gt_ptr, mask_ptr,
                            grad_image.ptr<float>(),
                            raw.raw_dm_dmu1.ptr<__half>(),
                            raw.raw_dm_dsigma1_sq.ptr<__half>(),
                            raw.raw_dm_dsigma12.ptr<__half>());
                    LFS_CUDA_LAUNCH_CHECK(lfs::core::getCurrentCUDAStream(), "training.ssim.masked_decoupled_raw_backward");
                });
            });
            return;
        }

        int grad_h = raw.H;
        int grad_w = raw.W;
        if (raw.apply_valid_padding) {
            grad_h = grad_h > 10 ? grad_h - 10 : grad_h;
            grad_w = grad_w > 10 ? grad_w - 10 : grad_w;
        }
        const size_t numel = N * C * grad_h * grad_w;
        const float grad_per_pixel = 1.0f / static_cast<float>(numel);

        dispatch_target_ptr(raw.gt_img, [&](auto* gt_ptr) {
            using TargetT = std::remove_cv_t<std::remove_pointer_t<decltype(gt_ptr)>>;
            fusedL1SSIMBackwardCUDA<TargetT, /*HasSigmaPartials=*/true, /*Accumulate=*/true>
                <<<grid, block, 0, lfs::core::getCurrentCUDAStream()>>>(
                    1.0f, raw.H, raw.W, static_cast<int>(C), C1, C2,
                    grad_per_pixel, raw.apply_valid_padding,
                    raw.raw_img.ptr<float>(), gt_ptr,
                    grad_image.ptr<float>(),
                    raw.raw_dm_dmu1.ptr<__half>(),
                    raw.raw_dm_dsigma1_sq.ptr<__half>(),
                    raw.raw_dm_dsigma12.ptr<__half>());
            LFS_CUDA_LAUNCH_CHECK(lfs::core::getCurrentCUDAStream(), "training.ssim.decoupled_raw_backward");
        });
    }

    // ============================================================================
    // Masked Fused L1+SSIM Implementation
    // ============================================================================

    std::pair<lfs::core::Tensor, MaskedFusedL1SSIMContext> masked_fused_l1_ssim_forward(
        const lfs::core::Tensor& img1_input,
        const lfs::core::Tensor& img2_input,
        const lfs::core::Tensor& mask_input,
        float ssim_weight,
        MaskedFusedL1SSIMWorkspace& workspace, float denominator) {
        LFS_ASSERT(std::isfinite(denominator) && denominator >= 0.0f);

        constexpr float C1 = 0.01f * 0.01f;
        constexpr float C2 = 0.03f * 0.03f;

        validate_loss_weight(ssim_weight);
        auto prepared_images = prepare_loss_images(img1_input, img2_input);
        auto& img1 = prepared_images.prediction;
        auto& img2 = prepared_images.target;
        auto mask_2d = prepare_loss_mask(mask_input, img1);

        const int N = static_cast<int>(img1.shape()[0]);
        const int C = static_cast<int>(img1.shape()[1]);
        const int H = static_cast<int>(img1.shape()[2]);
        const int W = static_cast<int>(img1.shape()[3]);

        workspace.ensure_size(img1.shape().dims());

        const dim3 grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, N);
        const dim3 block(BLOCK_X, BLOCK_Y);

        const auto stream = workspace.ssim_map.stream();
        dispatch_target_ptr(img2, [&](auto* img2_ptr) {
            using TargetT = std::remove_cv_t<std::remove_pointer_t<decltype(img2_ptr)>>;
            dispatch_mask_ptr(mask_2d, [&](auto* mask_ptr) {
                using MaskT = std::remove_cv_t<std::remove_pointer_t<decltype(mask_ptr)>>;
                maskedFusedL1SSIMForwardCUDA<TargetT, MaskT><<<grid, block, 0, lfs::core::getCurrentCUDAStream()>>>(
                    H, W, C, C1, C2,
                    img1.ptr<float>(), img2_ptr, mask_ptr,
                    workspace.dm_dmu1.ptr<__half>(),
                    workspace.dm_dsigma1_sq.ptr<__half>(),
                    workspace.dm_dsigma12.ptr<__half>(),
                    workspace.ssim_map.ptr<float>(),
                    workspace.cs_map.ptr<float>(),
                    /*ssim_channels=*/nullptr);
                LFS_CUDA_LAUNCH_CHECK(lfs::core::getCurrentCUDAStream(), "training.ssim.masked_fused_l1_forward");

                launch_masked_fused_l1_ssim_mean_device(
                    img1.ptr<float>(),
                    img2_ptr,
                    workspace.ssim_map.ptr<float>(),
                    mask_ptr,
                    ssim_weight,
                    workspace.reduction_temp.ptr<float>(),
                    workspace.masked_loss.ptr<float>(),
                    workspace.mask_sum.ptr<float>(),
                    N, C, H, W,
                    stream, denominator);
            });
        });

        auto loss_scalar = workspace.masked_loss;
        const float mask_sum = denominator > 0.0f ? denominator : workspace.mask_sum.item<float>();

        MaskedFusedL1SSIMContext ctx{
            .img1 = img1,
            .img2 = img2,
            .mask = mask_2d,
            .dm_dmu1 = workspace.dm_dmu1,
            .dm_dsigma1_sq = workspace.dm_dsigma1_sq,
            .dm_dsigma12 = workspace.dm_dsigma12,
            .ssim_weight = ssim_weight,
            .mask_sum_value = mask_sum,
            .H = H,
            .W = W};

        return {loss_scalar, ctx};
    }

    void masked_fused_l1_ssim_value(
        const lfs::core::Tensor& corrected_input,
        const lfs::core::Tensor& raw_input,
        const lfs::core::Tensor& target_input,
        const lfs::core::Tensor& mask_input,
        float ssim_weight,
        MaskedL1SSIMValueWorkspace& workspace,
        const float denominator) {
        LFS_ASSERT(std::isfinite(denominator) && denominator >= 0.0f);
        constexpr float C1 = 0.01f * 0.01f;
        constexpr float C2 = 0.03f * 0.03f;

        validate_loss_weight(ssim_weight);
        auto prepared_images = prepare_loss_images(corrected_input, target_input);
        auto& corrected = prepared_images.prediction;
        auto& target = prepared_images.target;
        const bool decoupled = raw_input.is_valid() && raw_input.numel() > 0;
        lfs::core::Tensor raw;
        if (decoupled) {
            raw = prepare_loss_prediction(raw_input, "Raw loss image");
            LFS_ASSERT_MSG(raw.shape() == corrected.shape(), "Raw and corrected loss images must match");
        }
        auto mask_2d = prepare_loss_mask(mask_input, corrected);

        const int N = static_cast<int>(corrected.shape()[0]);
        const int C = static_cast<int>(corrected.shape()[1]);
        const int H = static_cast<int>(corrected.shape()[2]);
        const int W = static_cast<int>(corrected.shape()[3]);

        const lfs::core::TensorShape map_shape{static_cast<size_t>(N), 1, static_cast<size_t>(H),
                                               static_cast<size_t>(W)};
        if (!workspace.ssim_map.is_valid() || workspace.ssim_map.shape() != map_shape) {
            workspace.ssim_map = lfs::core::Tensor::empty(map_shape, lfs::core::Device::CUDA);
            workspace.reduction_temp = lfs::core::Tensor::empty({2048}, lfs::core::Device::CUDA);
            workspace.loss = lfs::core::Tensor::empty({1}, lfs::core::Device::CUDA);
            workspace.mask_sum = lfs::core::Tensor::empty({1}, lfs::core::Device::CUDA);
        }

        const dim3 grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, N);
        const dim3 block(BLOCK_X, BLOCK_Y);
        const auto stream = lfs::core::getCurrentCUDAStream();
        dispatch_target_ptr(target, [&](auto* target_ptr) {
            using TargetT = std::remove_cv_t<std::remove_pointer_t<decltype(target_ptr)>>;
            dispatch_mask_ptr(mask_2d, [&](auto* mask_ptr) {
                using MaskT = std::remove_cv_t<std::remove_pointer_t<decltype(mask_ptr)>>;
                // Null partials and cs_map: the kernels write only the SSIM map.
                if (decoupled) {
                    maskedDecoupledFusedL1SSIMForwardCUDA<TargetT, MaskT><<<grid, block, 0, stream>>>(
                        H, W, C, C1, C2, ssim_weight,
                        corrected.ptr<float>(), raw.ptr<float>(), target_ptr, mask_ptr,
                        nullptr, nullptr, nullptr, nullptr,
                        workspace.ssim_map.ptr<float>(),
                        nullptr, nullptr);
                    LFS_CUDA_LAUNCH_CHECK(stream, "training.ssim.masked_decoupled_fused_l1_value");
                } else {
                    maskedFusedL1SSIMForwardCUDA<TargetT, MaskT><<<grid, block, 0, stream>>>(
                        H, W, C, C1, C2,
                        corrected.ptr<float>(), target_ptr, mask_ptr,
                        nullptr, nullptr, nullptr,
                        workspace.ssim_map.ptr<float>(),
                        nullptr, nullptr);
                    LFS_CUDA_LAUNCH_CHECK(stream, "training.ssim.masked_fused_l1_value");
                }

                launch_masked_fused_l1_ssim_mean_device(
                    corrected.ptr<float>(),
                    target_ptr,
                    workspace.ssim_map.ptr<float>(),
                    mask_ptr,
                    ssim_weight,
                    workspace.reduction_temp.ptr<float>(),
                    workspace.loss.ptr<float>(),
                    workspace.mask_sum.ptr<float>(),
                    N, C, H, W,
                    stream, denominator);
            });
        });
        // A fixed denominator skips the mask sum; it is the normalization then.
        if (denominator > 0.0f)
            workspace.mask_sum.fill_(denominator, stream);
    }

    lfs::core::Tensor masked_fused_l1_ssim_backward(
        const MaskedFusedL1SSIMContext& ctx,
        MaskedFusedL1SSIMWorkspace& workspace) {

        validate_loss_context_images(ctx.img1, ctx.img2, ctx.H, ctx.W);
        validate_loss_weight(ctx.ssim_weight);
        LFS_ASSERT_MSG(std::isfinite(ctx.mask_sum_value) && ctx.mask_sum_value > 0.0f,
                       "Masked loss normalization must be positive and finite");
        const auto mask = prepare_loss_mask(ctx.mask, ctx.img1);
        workspace.ensure_size(ctx.img1.shape().dims());

        constexpr float C1 = 0.01f * 0.01f;
        constexpr float C2 = 0.03f * 0.03f;

        const size_t N = ctx.img1.shape()[0];
        const size_t C = ctx.img1.shape()[1];
        const float inv_mask_sum = 1.0f / ctx.mask_sum_value;

        const dim3 grid((ctx.W + BLOCK_X - 1) / BLOCK_X, (ctx.H + BLOCK_Y - 1) / BLOCK_Y, N);
        const dim3 block(BLOCK_X, BLOCK_Y);

        dispatch_target_ptr(ctx.img2, [&](auto* img2_ptr) {
            using TargetT = std::remove_cv_t<std::remove_pointer_t<decltype(img2_ptr)>>;
            dispatch_mask_ptr(mask, [&](auto* mask_ptr) {
                using MaskT = std::remove_cv_t<std::remove_pointer_t<decltype(mask_ptr)>>;
                maskedFusedL1SSIMBackwardCUDA<TargetT, MaskT, /*HasSigmaPartials=*/true>
                    <<<grid, block, 0, lfs::core::getCurrentCUDAStream()>>>(
                        ctx.ssim_weight, inv_mask_sum, ctx.H, ctx.W, static_cast<int>(C), C1, C2,
                        ctx.img1.ptr<float>(), img2_ptr, mask_ptr,
                        workspace.grad_img.ptr<float>(),
                        ctx.dm_dmu1.ptr<__half>(), ctx.dm_dsigma1_sq.ptr<__half>(),
                        ctx.dm_dsigma12.ptr<__half>());
                LFS_CUDA_LAUNCH_CHECK(lfs::core::getCurrentCUDAStream(), "training.ssim.masked_fused_l1_backward");
            });
        });

        return workspace.grad_img;
    }

    std::pair<lfs::core::Tensor, MaskedDecoupledFusedL1SSIMContext> masked_decoupled_fused_l1_ssim_forward(
        const lfs::core::Tensor& corrected_input,
        const lfs::core::Tensor& raw_input,
        const lfs::core::Tensor& gt_input,
        const lfs::core::Tensor& mask_input,
        float ssim_weight,
        MaskedDecoupledFusedL1SSIMWorkspace& workspace, float denominator) {
        LFS_ASSERT(std::isfinite(denominator) && denominator >= 0.0f);

        constexpr float C1 = 0.01f * 0.01f;
        constexpr float C2 = 0.03f * 0.03f;

        validate_loss_weight(ssim_weight);
        auto corrected = prepare_loss_prediction(corrected_input, "Corrected loss image");
        auto raw = prepare_loss_prediction(raw_input, "Raw loss image");
        auto gt = prepare_loss_target(gt_input, "Loss target");
        LFS_ASSERT_MSG(corrected.shape() == raw.shape() && corrected.shape() == gt.shape(),
                       lfs::core::detail::format_cuda_safe(
                           "Masked decoupled loss image shapes must match "
                           "(corrected={}, raw={}, target={})",
                           corrected.shape().str(), raw.shape().str(), gt.shape().str()));
        auto mask_2d = prepare_loss_mask(mask_input, corrected);

        const int N = static_cast<int>(corrected.shape()[0]);
        const int C = static_cast<int>(corrected.shape()[1]);
        const int H = static_cast<int>(corrected.shape()[2]);
        const int W = static_cast<int>(corrected.shape()[3]);

        workspace.ensure_size(corrected.shape().dims());

        const dim3 grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, N);
        const dim3 block(BLOCK_X, BLOCK_Y);

        const auto stream = workspace.ssim_map.stream();
        dispatch_target_ptr(gt, [&](auto* gt_ptr) {
            using TargetT = std::remove_cv_t<std::remove_pointer_t<decltype(gt_ptr)>>;
            dispatch_mask_ptr(mask_2d, [&](auto* mask_ptr) {
                using MaskT = std::remove_cv_t<std::remove_pointer_t<decltype(mask_ptr)>>;
                maskedDecoupledFusedL1SSIMForwardCUDA<TargetT, MaskT><<<grid, block, 0, lfs::core::getCurrentCUDAStream()>>>(
                    H, W, C, C1, C2, ssim_weight,
                    corrected.ptr<float>(), raw.ptr<float>(), gt_ptr, mask_ptr,
                    workspace.app_dm_dmu1.ptr<__half>(),
                    workspace.raw_dm_dmu1.ptr<__half>(),
                    workspace.raw_dm_dsigma1_sq.ptr<__half>(),
                    workspace.raw_dm_dsigma12.ptr<__half>(),
                    workspace.ssim_map.ptr<float>(),
                    workspace.cs_map.ptr<float>(), nullptr);
                LFS_CUDA_LAUNCH_CHECK(lfs::core::getCurrentCUDAStream(), "training.ssim.masked_decoupled_fused_l1_forward");

                launch_masked_fused_l1_ssim_mean_device(
                    corrected.ptr<float>(),
                    gt_ptr,
                    workspace.ssim_map.ptr<float>(),
                    mask_ptr,
                    ssim_weight,
                    workspace.reduction_temp.ptr<float>(),
                    workspace.masked_loss.ptr<float>(),
                    workspace.mask_sum.ptr<float>(),
                    N, C, H, W,
                    stream, denominator);
            });
        });

        auto loss_scalar = workspace.masked_loss;
        const float mask_sum = denominator > 0.0f ? denominator : workspace.mask_sum.item<float>();

        MaskedDecoupledFusedL1SSIMContext ctx{
            .corrected_img = corrected,
            .raw_img = raw,
            .gt_img = gt,
            .mask = mask_2d,
            .app_dm_dmu1 = workspace.app_dm_dmu1,
            .raw_dm_dmu1 = workspace.raw_dm_dmu1,
            .raw_dm_dsigma1_sq = workspace.raw_dm_dsigma1_sq,
            .raw_dm_dsigma12 = workspace.raw_dm_dsigma12,
            .ssim_weight = ssim_weight,
            .mask_sum_value = mask_sum,
            .H = H,
            .W = W};

        return {loss_scalar, ctx};
    }

    DecoupledGradients masked_decoupled_fused_l1_ssim_backward(
        const MaskedDecoupledFusedL1SSIMContext& ctx,
        MaskedDecoupledFusedL1SSIMWorkspace& workspace) {

        validate_loss_context_images(ctx.corrected_img, ctx.gt_img, ctx.H, ctx.W);
        validate_loss_context_images(ctx.raw_img, ctx.gt_img, ctx.H, ctx.W);
        validate_loss_weight(ctx.ssim_weight);
        LFS_ASSERT_MSG(std::isfinite(ctx.mask_sum_value) && ctx.mask_sum_value > 0.0f,
                       "Masked loss normalization must be positive and finite");
        const auto mask = prepare_loss_mask(ctx.mask, ctx.corrected_img);
        workspace.ensure_size(ctx.corrected_img.shape().dims());

        constexpr float C1 = 0.01f * 0.01f;
        constexpr float C2 = 0.03f * 0.03f;

        const size_t N = ctx.corrected_img.shape()[0];
        const size_t C = ctx.corrected_img.shape()[1];
        const float inv_mask_sum = 1.0f / ctx.mask_sum_value;

        const dim3 grid((ctx.W + BLOCK_X - 1) / BLOCK_X, (ctx.H + BLOCK_Y - 1) / BLOCK_Y, N);
        const dim3 block(BLOCK_X, BLOCK_Y);

        dispatch_target_ptr(ctx.gt_img, [&](auto* gt_ptr) {
            using TargetT = std::remove_cv_t<std::remove_pointer_t<decltype(gt_ptr)>>;
            dispatch_mask_ptr(mask, [&](auto* mask_ptr) {
                using MaskT = std::remove_cv_t<std::remove_pointer_t<decltype(mask_ptr)>>;
                maskedFusedL1SSIMBackwardCUDA<TargetT, MaskT, /*HasSigmaPartials=*/false>
                    <<<grid, block, 0, lfs::core::getCurrentCUDAStream()>>>(
                        ctx.ssim_weight, inv_mask_sum, ctx.H, ctx.W, static_cast<int>(C), C1, C2,
                        ctx.corrected_img.ptr<float>(), gt_ptr, mask_ptr,
                        workspace.grad_corrected.ptr<float>(),
                        ctx.app_dm_dmu1.ptr<__half>(),
                        /*dm_dsigma1_sq=*/nullptr,
                        /*dm_dsigma12=*/nullptr);
                LFS_CUDA_LAUNCH_CHECK(lfs::core::getCurrentCUDAStream(), "training.ssim.masked_decoupled_fused_l1_backward");
            });
        });

        return DecoupledGradients{.grad_corrected = workspace.grad_corrected};
    }

    // Fused SSIM map [1, C, H, W] → error map [H, W]
    // error_map[i] = max(0, 1 - mean_c(ssim_map[c, i]))
    namespace {
        __global__ void ssim_to_error_map_kernel(
            const float* __restrict__ ssim_map,
            float* __restrict__ error_map,
            int C, int H, int W) {

            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int HW = H * W;
            if (idx >= HW)
                return;

            float sum = 0.0f;
            for (int c = 0; c < C; ++c) {
                sum += ssim_map[c * HW + idx];
            }
            float inv_c = 1.0f / static_cast<float>(C);
            float err = 1.0f - sum * inv_c;
            error_map[idx] = fmaxf(err, 0.0f);
        }
    } // namespace

    void launch_ssim_to_error_map(
        const lfs::core::Tensor& ssim_map,
        lfs::core::Tensor& error_map) {

        LFS_ASSERT_MSG(ssim_map.is_valid() && ssim_map.device() == lfs::core::Device::CUDA &&
                           ssim_map.dtype() == lfs::core::DataType::Float32 &&
                           ssim_map.is_contiguous() && ssim_map.ndim() == 4 &&
                           ssim_map.shape()[0] == 1 && ssim_map.shape()[1] > 0 &&
                           ssim_map.shape()[2] > 0 && ssim_map.shape()[3] > 0,
                       lfs::core::detail::format_cuda_safe(
                           "SSIM map must be contiguous Float32 CUDA [1,C,H,W] (shape={})",
                           ssim_map.shape().str()));

        const int C = static_cast<int>(ssim_map.shape()[1]);
        const int H = static_cast<int>(ssim_map.shape()[2]);
        const int W = static_cast<int>(ssim_map.shape()[3]);
        const int HW = H * W;

        LFS_ASSERT_MSG(error_map.is_valid() && error_map.device() == lfs::core::Device::CUDA &&
                           error_map.dtype() == lfs::core::DataType::Float32 &&
                           error_map.is_contiguous() && error_map.ndim() == 2 &&
                           error_map.shape()[0] == static_cast<size_t>(H) &&
                           error_map.shape()[1] == static_cast<size_t>(W),
                       lfs::core::detail::format_cuda_safe(
                           "SSIM error map must be contiguous Float32 CUDA [H,W] "
                           "(map={}, expected=[{}, {}])",
                           error_map.shape().str(), H, W));

        constexpr int THREADS = 256;
        dim3 grid((HW + THREADS - 1) / THREADS);
        dim3 block(THREADS);

        ssim_to_error_map_kernel<<<grid, block, 0, lfs::core::getCurrentCUDAStream()>>>(
            ssim_map.ptr<float>(),
            error_map.ptr<float>(),
            C, H, W);
        LFS_CUDA_LAUNCH_CHECK(lfs::core::getCurrentCUDAStream(), "training.ssim.to_error_map");
    }

    // =========================================================================
    // Blur-tolerant SSIM and multi-scale photometric loss
    // =========================================================================
    namespace {
        constexpr int PYRAMID_THREADS = 256;
        constexpr int MAX_BLUR_RADIUS = 6;
        constexpr int MAX_PYRAMID_LEVELS = 4;
        constexpr int PYRAMID_BASE_SIZE = 1920;
        constexpr float LN2 = 0.69314718055994531f;

        struct BlurTaps {
            float weight[MAX_BLUR_RADIUS + 1];
            int radius;
        };

        BlurTaps make_blur_taps(const float sigma) {
            LFS_ASSERT_MSG(std::isfinite(sigma) && sigma > 0.0f, "Blur sigma must be finite and positive");
            BlurTaps taps{};
            taps.radius = std::min(MAX_BLUR_RADIUS, static_cast<int>(std::ceil(3.0 * sigma)));
            double raw[MAX_BLUR_RADIUS + 1] = {};
            double sum = 0.0;
            for (int d = 0; d <= taps.radius; ++d) {
                raw[d] = std::exp(-0.5 * d * d / (static_cast<double>(sigma) * sigma));
                sum += d == 0 ? raw[d] : 2.0 * raw[d];
            }
            for (int d = 0; d <= taps.radius; ++d)
                taps.weight[d] = static_cast<float>(raw[d] / sum);
            return taps;
        }

        unsigned pyramid_blocks(const size_t items) {
            return static_cast<unsigned>((items + PYRAMID_THREADS - 1) / PYRAMID_THREADS);
        }

        // A pixel enters the blur when it is inside the image and, with a weight map, weighted.
        template <typename MaskT>
        __device__ __forceinline__ float keep_value(const MaskT* mask, const int index) {
            return mask == nullptr || mask_value(mask, index) > 0.0f ? 1.0f : 0.0f;
        }

        template <typename MaskT>
        __global__ void blur_support_rows_kernel(const MaskT* __restrict__ mask, float* __restrict__ rows,
                                                 const int H, const int W, const BlurTaps taps) {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= H * W)
                return;
            const int y = idx / W;
            const int x = idx - y * W;
            float sum = 0.0f;
            for (int d = -taps.radius; d <= taps.radius; ++d) {
                const int xx = x + d;
                if (xx >= 0 && xx < W)
                    sum += taps.weight[d < 0 ? -d : d] * keep_value(mask, y * W + xx);
            }
            rows[idx] = sum;
        }

        __global__ void blur_support_cols_kernel(const float* __restrict__ rows, float* __restrict__ inv_support,
                                                 const int H, const int W, const BlurTaps taps) {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= H * W)
                return;
            const int y = idx / W;
            const int x = idx - y * W;
            float sum = 0.0f;
            for (int d = -taps.radius; d <= taps.radius; ++d) {
                const int yy = y + d;
                if (yy >= 0 && yy < H)
                    sum += taps.weight[d < 0 ? -d : d] * rows[yy * W + x];
            }
            inv_support[idx] = sum > 1.0e-30f ? 1.0f / sum : 0.0f;
        }

        // Horizontal taps over the kept pixels of each plane.
        template <typename MaskT>
        __global__ void blur_rows_kernel(const float* __restrict__ in, const MaskT* __restrict__ mask,
                                         float* __restrict__ out, const int planes, const int H, const int W,
                                         const BlurTaps taps) {
            const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            const size_t plane_pixels = static_cast<size_t>(H) * W;
            if (idx >= plane_pixels * planes)
                return;
            const size_t plane = idx / plane_pixels;
            const int p = static_cast<int>(idx - plane * plane_pixels);
            const int y = p / W;
            const int x = p - y * W;
            const float* row = in + plane * plane_pixels + static_cast<size_t>(y) * W;
            float sum = 0.0f;
            for (int d = -taps.radius; d <= taps.radius; ++d) {
                const int xx = x + d;
                if (xx >= 0 && xx < W)
                    sum += taps.weight[d < 0 ? -d : d] * keep_value(mask, y * W + xx) * row[xx];
            }
            out[idx] = sum;
        }

        // Vertical taps, then the division by the blurred keep mask.
        __global__ void blur_cols_kernel(const float* __restrict__ in, const float* __restrict__ inv_support,
                                         float* __restrict__ out, const int planes, const int H, const int W,
                                         const BlurTaps taps) {
            const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            const size_t plane_pixels = static_cast<size_t>(H) * W;
            if (idx >= plane_pixels * planes)
                return;
            const size_t plane = idx / plane_pixels;
            const int p = static_cast<int>(idx - plane * plane_pixels);
            const int y = p / W;
            const int x = p - y * W;
            const float* column = in + plane * plane_pixels + x;
            float sum = 0.0f;
            for (int d = -taps.radius; d <= taps.radius; ++d) {
                const int yy = y + d;
                if (yy >= 0 && yy < H)
                    sum += taps.weight[d < 0 ? -d : d] * column[static_cast<size_t>(yy) * W];
            }
            out[idx] = sum * inv_support[p];
        }

        // Adjoint, first half: divide by the blurred keep mask, then the vertical taps.
        __global__ void blur_adjoint_cols_kernel(const float* __restrict__ grad, const float* __restrict__ inv_support,
                                                 float* __restrict__ out, const int planes, const int H, const int W,
                                                 const BlurTaps taps) {
            const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            const size_t plane_pixels = static_cast<size_t>(H) * W;
            if (idx >= plane_pixels * planes)
                return;
            const size_t plane = idx / plane_pixels;
            const int p = static_cast<int>(idx - plane * plane_pixels);
            const int y = p / W;
            const int x = p - y * W;
            const float* column = grad + plane * plane_pixels + x;
            float sum = 0.0f;
            for (int d = -taps.radius; d <= taps.radius; ++d) {
                const int yy = y + d;
                if (yy >= 0 && yy < H)
                    sum += taps.weight[d < 0 ? -d : d] * column[static_cast<size_t>(yy) * W] * inv_support[yy * W + x];
            }
            out[idx] = sum;
        }

        // Adjoint, second half: horizontal taps; only kept pixels receive gradient.
        template <typename MaskT>
        __global__ void blur_adjoint_rows_kernel(const float* __restrict__ in, const MaskT* __restrict__ mask,
                                                 float* __restrict__ out, const bool accumulate, const int planes,
                                                 const int H, const int W, const BlurTaps taps) {
            const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            const size_t plane_pixels = static_cast<size_t>(H) * W;
            if (idx >= plane_pixels * planes)
                return;
            const size_t plane = idx / plane_pixels;
            const int p = static_cast<int>(idx - plane * plane_pixels);
            const int y = p / W;
            const int x = p - y * W;
            const float* row = in + plane * plane_pixels + static_cast<size_t>(y) * W;
            float sum = 0.0f;
            for (int d = -taps.radius; d <= taps.radius; ++d) {
                const int xx = x + d;
                if (xx >= 0 && xx < W)
                    sum += taps.weight[d < 0 ? -d : d] * row[xx];
            }
            const float value = keep_value(mask, p) * sum;
            out[idx] = accumulate ? out[idx] + value : value;
        }

        // 2x2 average; an odd last row or column averages the pixels it has.
        template <typename T, bool IsMask>
        __global__ void pool2x2_kernel(const T* __restrict__ in, float* __restrict__ out, const int planes,
                                       const int H, const int W, const int h, const int w) {
            const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            const size_t coarse_pixels = static_cast<size_t>(h) * w;
            if (idx >= coarse_pixels * planes)
                return;
            const size_t plane = idx / coarse_pixels;
            const int p = static_cast<int>(idx - plane * coarse_pixels);
            const int Y = p / w;
            const int X = p - Y * w;
            const size_t base = plane * static_cast<size_t>(H) * W;
            float sum = 0.0f;
            int count = 0;
            for (int a = 0; a < 2; ++a) {
                const int y = 2 * Y + a;
                if (y >= H)
                    continue;
                for (int b = 0; b < 2; ++b) {
                    const int x = 2 * X + b;
                    if (x >= W)
                        continue;
                    const int index = static_cast<int>(base + static_cast<size_t>(y) * W + x);
                    if constexpr (IsMask)
                        sum += mask_value(in, index);
                    else
                        sum += pixel_value(in, index);
                    ++count;
                }
            }
            out[idx] = sum / static_cast<float>(count);
        }

        __global__ void unpool2x2_adjoint_accumulate_kernel(const float* __restrict__ coarse, float* __restrict__ fine,
                                                            const int planes, const int H, const int W,
                                                            const int h, const int w) {
            const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            const size_t fine_pixels = static_cast<size_t>(H) * W;
            if (idx >= fine_pixels * planes)
                return;
            const size_t plane = idx / fine_pixels;
            const int p = static_cast<int>(idx - plane * fine_pixels);
            const int Y = (p / W) >> 1;
            const int X = (p % W) >> 1;
            const int count = min(2, H - 2 * Y) * min(2, W - 2 * X);
            fine[idx] += coarse[plane * static_cast<size_t>(h) * w + static_cast<size_t>(Y) * w + X] /
                         static_cast<float>(count);
        }

        // SSIM partials of one branch: mu, sigma1^2 and sigma12 of the render (decoupled loss: of the
        // raw render), and the corrected image's mu partial of the decoupled loss (null otherwise).
        struct BranchPartials {
            __half* p[4] = {nullptr, nullptr, nullptr, nullptr};
        };

        // Soft maximum of the two SSIM branches per channel and pixel. The partials of each branch are
        // scaled in place by its softmax weight, the blurred branch's also by blurred_scale[k], so that
        // its backward runs with SSIM weight 1 (no L1 term) and the SSIM weight inside its partials;
        // the channel mean of the blend goes to the loss.
        __global__ void blend_blur_tolerant_kernel(const float* __restrict__ ssim_sharp,
                                                   const float* __restrict__ ssim_blurred,
                                                   const BranchPartials sharp,
                                                   const BranchPartials blurred,
                                                   const float4 blurred_scale,
                                                   float* __restrict__ blend_map,
                                                   const int N, const int C, const int HW,
                                                   const float temperature) {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= N * HW)
                return;
            const int n = idx / HW;
            const int p = idx - n * HW;
            const float scales[4] = {blurred_scale.x, blurred_scale.y, blurred_scale.z, blurred_scale.w};
            float sum = 0.0f;
            for (int c = 0; c < C; ++c) {
                const size_t i = (static_cast<size_t>(n) * C + c) * HW + p;
                const float s0 = ssim_sharp[i];
                const float s1 = ssim_blurred[i];
                const float d = temperature * (s0 - s1);
                sum += fmaxf(s0, s1) + (log1pf(expf(-fabsf(d))) - LN2) / temperature;
                const float w0 = 1.0f / (1.0f + expf(-d));
                const float w1 = 1.0f / (1.0f + expf(d));
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    if (sharp.p[k])
                        sharp.p[k][i] = __float2half(__half2float(sharp.p[k][i]) * w0);
                    if (blurred.p[k])
                        blurred.p[k][i] = __float2half(__half2float(blurred.p[k][i]) * (w1 * scales[k]));
                }
            }
            blend_map[idx] = sum / static_cast<float>(C);
        }

        // ---------------------------------------------------------------------------------------
        // Fused blur-tolerant level: one pass blurs the render (and the raw render), writes the blurred
        // keep mask's reciprocal and pools the next scale's render; one forward computes both SSIM
        // branches from shared photo statistics and folds the soft-maximum blend into its partials; one
        // backward convolves the partials of both branches; one pass applies the blur adjoint, takes the
        // coarser scale's gradient through the pooling and, at scale 0, the 1 / K of the pyramid. Each step keeps the
        // expressions and summation order of the per-branch kernels above, so the fused level gives
        // their result up to float contraction.
        // ---------------------------------------------------------------------------------------
        constexpr int FUSED_BLUR_TX = 32;
        constexpr int FUSED_BLUR_TY = 16;

        // The taps of radius R in registers, so the unrolled loops below index them at compile time.
        template <int R>
        struct RegisterTaps {
            float w[R + 1];
            __device__ __forceinline__ explicit RegisterTaps(const BlurTaps& taps) {
#pragma unroll
                for (int k = 0; k <= R; ++k)
                    w[k] = taps.weight[k];
            }
            __device__ __forceinline__ float operator[](const int d) const { return w[d < 0 ? -d : d]; }
        };

        // Blur of in0 (and in1 when given) over kept pixels, as blur_support_* + blur_rows/cols_kernel,
        // and, when pool0 is given, the 2x2 average of each input for the next scale (as
        // pool2x2_kernel) from the same tile.
        template <typename MaskT, int R>
        __global__ void __launch_bounds__(FUSED_BLUR_TX* FUSED_BLUR_TY)
            fused_blur_kernel(const float* __restrict__ in0, const float* __restrict__ in1,
                              const MaskT* __restrict__ mask, float* __restrict__ out0, float* __restrict__ out1,
                              float* __restrict__ inv_support, float* __restrict__ pool0, float* __restrict__ pool1,
                              const int planes, const int H, const int W, const int h, const int w,
                              const BlurTaps blur_taps) {
            static_assert(FUSED_BLUR_TX % 2 == 0 && FUSED_BLUR_TY % 2 == 0, "Pooling needs even tiles");
            constexpr int SX = FUSED_BLUR_TX + 2 * R;
            constexpr int SY = FUSED_BLUR_TY + 2 * R;
            __shared__ float keep[SY][SX];
            __shared__ float tile[SY][SX];
            __shared__ float rows[SY][FUSED_BLUR_TX];
            const RegisterTaps<R> taps(blur_taps);
            const int x0 = blockIdx.x * FUSED_BLUR_TX, y0 = blockIdx.y * FUSED_BLUR_TY;
            const int tx = threadIdx.x, ty = threadIdx.y;
            const int x = x0 + tx, y = y0 + ty;
            const bool inside = x < W && y < H;
            const size_t plane_pixels = static_cast<size_t>(H) * W;

            for (int r = ty; r < SY; r += FUSED_BLUR_TY)
                for (int c = tx; c < SX; c += FUSED_BLUR_TX) {
                    const int gy = y0 - R + r, gx = x0 - R + c;
                    keep[r][c] = gy >= 0 && gy < H && gx >= 0 && gx < W ? keep_value(mask, gy * W + gx) : 0.0f;
                }
            __syncthreads();
            for (int r = ty; r < SY; r += FUSED_BLUR_TY) {
                float sum = 0.0f;
#pragma unroll
                for (int d = -R; d <= R; ++d) {
                    const int xx = x + d;
                    if (xx >= 0 && xx < W)
                        sum += taps[d] * keep[r][R + tx + d];
                }
                rows[r][tx] = sum;
            }
            __syncthreads();
            float inv = 0.0f;
            if (inside) {
                float sum = 0.0f;
#pragma unroll
                for (int d = -R; d <= R; ++d) {
                    const int yy = y + d;
                    if (yy >= 0 && yy < H)
                        sum += taps[d] * rows[R + ty + d][tx];
                }
                inv = sum > 1.0e-30f ? 1.0f / sum : 0.0f;
                inv_support[static_cast<size_t>(y) * W + x] = inv;
            }

            for (int image = 0; image < 2; ++image) {
                const float* in = image == 0 ? in0 : in1;
                float* out = image == 0 ? out0 : out1;
                float* pool = image == 0 ? pool0 : pool1;
                if (in == nullptr)
                    break;
                for (int plane = 0; plane < planes; ++plane) {
                    const float* src = in + plane * plane_pixels;
                    __syncthreads();
                    for (int r = ty; r < SY; r += FUSED_BLUR_TY)
                        for (int c = tx; c < SX; c += FUSED_BLUR_TX) {
                            const int gy = y0 - R + r, gx = x0 - R + c;
                            tile[r][c] = gy >= 0 && gy < H && gx >= 0 && gx < W ? src[static_cast<size_t>(gy) * W + gx]
                                                                                : 0.0f;
                        }
                    __syncthreads();
                    if (pool != nullptr) {
                        constexpr int PX = FUSED_BLUR_TX / 2;
                        const int tid = ty * FUSED_BLUR_TX + tx;
                        const int X = x0 / 2 + tid % PX, Y = y0 / 2 + tid / PX;
                        if (tid < PX * (FUSED_BLUR_TY / 2) && X < w && Y < h) {
                            float sum = 0.0f;
                            int count = 0;
                            for (int a = 0; a < 2; ++a) {
                                const int yy = 2 * Y + a;
                                if (yy >= H)
                                    continue;
                                for (int b = 0; b < 2; ++b) {
                                    const int xx = 2 * X + b;
                                    if (xx >= W)
                                        continue;
                                    sum += tile[R + yy - y0][R + xx - x0];
                                    ++count;
                                }
                            }
                            pool[plane * static_cast<size_t>(h) * w + static_cast<size_t>(Y) * w + X] =
                                sum / static_cast<float>(count);
                        }
                    }
                    for (int r = ty; r < SY; r += FUSED_BLUR_TY) {
                        float sum = 0.0f;
#pragma unroll
                        for (int d = -R; d <= R; ++d) {
                            const int xx = x + d;
                            if (xx >= 0 && xx < W)
                                sum += taps[d] * keep[r][R + tx + d] * tile[r][R + tx + d];
                        }
                        rows[r][tx] = sum;
                    }
                    __syncthreads();
                    if (inside) {
                        float sum = 0.0f;
#pragma unroll
                        for (int d = -R; d <= R; ++d) {
                            const int yy = y + d;
                            if (yy >= 0 && yy < H)
                                sum += taps[d] * rows[R + ty + d][tx];
                        }
                        out[plane * plane_pixels + static_cast<size_t>(y) * W + x] = sum * inv;
                    }
                }
            }
        }

        // out_k = out_k + blur adjoint of g_k (as blur_adjoint_cols + blur_adjoint_rows_kernel with
        // accumulate), then + the coarser gradient through the 2x2 pooling (as
        // unpool2x2_adjoint_accumulate_kernel) when coarse_k is given, then x scale when apply_scale.
        template <typename MaskT, int R>
        __global__ void __launch_bounds__(FUSED_BLUR_TX* FUSED_BLUR_TY)
            fused_blur_adjoint_kernel(const float* __restrict__ g0, const float* __restrict__ g1,
                                      const float* __restrict__ inv_support, const MaskT* __restrict__ mask,
                                      float* __restrict__ out0, float* __restrict__ out1,
                                      const float* __restrict__ coarse0, const float* __restrict__ coarse1,
                                      const int planes, const int H, const int W, const int h, const int w,
                                      const BlurTaps blur_taps, const float scale, const bool apply_scale) {
            constexpr int SX = FUSED_BLUR_TX + 2 * R;
            constexpr int SY = FUSED_BLUR_TY + 2 * R;
            __shared__ float inv_tile[SY][SX];
            __shared__ float tile[SY][SX];
            __shared__ float cols[FUSED_BLUR_TY][SX];
            const RegisterTaps<R> taps(blur_taps);
            const int x0 = blockIdx.x * FUSED_BLUR_TX, y0 = blockIdx.y * FUSED_BLUR_TY;
            const int tx = threadIdx.x, ty = threadIdx.y;
            const int x = x0 + tx, y = y0 + ty;
            const bool inside = x < W && y < H;
            const size_t plane_pixels = static_cast<size_t>(H) * W;

            for (int r = ty; r < SY; r += FUSED_BLUR_TY)
                for (int c = tx; c < SX; c += FUSED_BLUR_TX) {
                    const int gy = y0 - R + r, gx = x0 - R + c;
                    inv_tile[r][c] =
                        gy >= 0 && gy < H && gx >= 0 && gx < W ? inv_support[static_cast<size_t>(gy) * W + gx] : 0.0f;
                }
            const float keep_center = inside ? keep_value(mask, y * W + x) : 0.0f;
            const int Y = y >> 1, X = x >> 1;
            const float count = inside ? static_cast<float>(min(2, H - 2 * Y) * min(2, W - 2 * X)) : 1.0f;

            for (int image = 0; image < 2; ++image) {
                const float* g = image == 0 ? g0 : g1;
                float* out = image == 0 ? out0 : out1;
                const float* coarse = image == 0 ? coarse0 : coarse1;
                if (g == nullptr)
                    break;
                for (int plane = 0; plane < planes; ++plane) {
                    const float* src = g + plane * plane_pixels;
                    __syncthreads();
                    for (int r = ty; r < SY; r += FUSED_BLUR_TY)
                        for (int c = tx; c < SX; c += FUSED_BLUR_TX) {
                            const int gy = y0 - R + r, gx = x0 - R + c;
                            tile[r][c] = gy >= 0 && gy < H && gx >= 0 && gx < W ? src[static_cast<size_t>(gy) * W + gx]
                                                                                : 0.0f;
                        }
                    __syncthreads();
                    for (int c = tx; c < SX; c += FUSED_BLUR_TX) {
                        float sum = 0.0f;
#pragma unroll
                        for (int d = -R; d <= R; ++d) {
                            const int yy = y + d;
                            if (yy >= 0 && yy < H)
                                sum += taps[d] * tile[R + ty + d][c] * inv_tile[R + ty + d][c];
                        }
                        cols[ty][c] = sum;
                    }
                    __syncthreads();
                    if (inside) {
                        float sum = 0.0f;
#pragma unroll
                        for (int d = -R; d <= R; ++d) {
                            const int xx = x + d;
                            if (xx >= 0 && xx < W)
                                sum += taps[d] * cols[ty][R + tx + d];
                        }
                        const size_t idx = plane * plane_pixels + static_cast<size_t>(y) * W + x;
                        const float value = keep_center * sum;
                        float result = out[idx] + value;
                        if (coarse != nullptr)
                            result += coarse[plane * static_cast<size_t>(h) * w + static_cast<size_t>(Y) * w + X] / count;
                        if (apply_scale)
                            result *= scale;
                        out[idx] = result;
                    }
                }
            }
        }

        // Runs fn with the blur radius as a compile-time constant.
        template <typename Fn>
        void dispatch_blur_radius(const int radius, Fn&& fn) {
            switch (radius) {
            case 1: fn(std::integral_constant<int, 1>{}); break;
            case 2: fn(std::integral_constant<int, 2>{}); break;
            case 3: fn(std::integral_constant<int, 3>{}); break;
            case 4: fn(std::integral_constant<int, 4>{}); break;
            case 5: fn(std::integral_constant<int, 5>{}); break;
            case 6: fn(std::integral_constant<int, 6>{}); break;
            default: LFS_ASSERT_MSG(false, "Blur radius must be within [1, 6]");
            }
        }

        template <bool Weighted>
        __device__ __forceinline__ void horizontal_tap(float& sum, const float left, const float right, const float w,
                                                       const float ml, const float mr) {
            if constexpr (Weighted)
                sum += left * ml + right * mr;
            else
                sum += (left + right) * w;
        }

        // Tile channels: x, y, Gx (plain loss) or corrected, raw, y, G corrected, G raw (decoupled).
        // Moments: x, x^2, y, y^2, xy, b, b^2, by (plain) or corrected, raw, raw^2, y, y^2, raw y,
        // b corrected, b raw, b raw^2, b raw y (decoupled).
        template <bool Weighted, bool Decoupled, int T, int Stride>
        __device__ __forceinline__ void dual_horizontal_moments(const float (&tile)[SHARED_Y][SHARED_X][T],
                                                                const float (&mask_tile)[SHARED_Y][SHARED_X],
                                                                float (&xconv)[CONV_Y][CONV_X][Stride], const int yy) {
            constexpr int Mo = Decoupled ? 10 : 8;
            const int lx = threadIdx.x + HALO;
            float s[Mo];
#pragma unroll
            for (int k = 0; k < Mo; ++k)
                s[k] = 0.0f;
#pragma unroll
            for (int d = 1; d <= HALO; ++d) {
                const float w = cGauss[HALO - d];
                float ml = 0.0f, mr = 0.0f;
                if constexpr (Weighted) {
                    ml = mask_tile[yy][lx - d] * w;
                    mr = mask_tile[yy][lx + d] * w;
                }
                const auto* L = tile[yy][lx - d];
                const auto* Rt = tile[yy][lx + d];
                if constexpr (Decoupled) {
                    horizontal_tap<Weighted>(s[0], L[0], Rt[0], w, ml, mr);
                    horizontal_tap<Weighted>(s[1], L[1], Rt[1], w, ml, mr);
                    horizontal_tap<Weighted>(s[2], L[1] * L[1], Rt[1] * Rt[1], w, ml, mr);
                    horizontal_tap<Weighted>(s[3], L[2], Rt[2], w, ml, mr);
                    horizontal_tap<Weighted>(s[4], L[2] * L[2], Rt[2] * Rt[2], w, ml, mr);
                    horizontal_tap<Weighted>(s[5], L[1] * L[2], Rt[1] * Rt[2], w, ml, mr);
                    horizontal_tap<Weighted>(s[6], L[3], Rt[3], w, ml, mr);
                    horizontal_tap<Weighted>(s[7], L[4], Rt[4], w, ml, mr);
                    horizontal_tap<Weighted>(s[8], L[4] * L[4], Rt[4] * Rt[4], w, ml, mr);
                    horizontal_tap<Weighted>(s[9], L[4] * L[2], Rt[4] * Rt[2], w, ml, mr);
                } else {
                    horizontal_tap<Weighted>(s[0], L[0], Rt[0], w, ml, mr);
                    horizontal_tap<Weighted>(s[1], L[0] * L[0], Rt[0] * Rt[0], w, ml, mr);
                    horizontal_tap<Weighted>(s[2], L[1], Rt[1], w, ml, mr);
                    horizontal_tap<Weighted>(s[3], L[1] * L[1], Rt[1] * Rt[1], w, ml, mr);
                    horizontal_tap<Weighted>(s[4], L[0] * L[1], Rt[0] * Rt[1], w, ml, mr);
                    horizontal_tap<Weighted>(s[5], L[2], Rt[2], w, ml, mr);
                    horizontal_tap<Weighted>(s[6], L[2] * L[2], Rt[2] * Rt[2], w, ml, mr);
                    horizontal_tap<Weighted>(s[7], L[2] * L[1], Rt[2] * Rt[1], w, ml, mr);
                }
            }
            {
                float wc = cGauss[HALO];
                if constexpr (Weighted)
                    wc *= mask_tile[yy][lx];
                const auto* Ct = tile[yy][lx];
                if constexpr (Decoupled) {
                    s[0] += Ct[0] * wc;
                    s[1] += Ct[1] * wc;
                    s[2] += (Ct[1] * Ct[1]) * wc;
                    s[3] += Ct[2] * wc;
                    s[4] += (Ct[2] * Ct[2]) * wc;
                    s[5] += (Ct[1] * Ct[2]) * wc;
                    s[6] += Ct[3] * wc;
                    s[7] += Ct[4] * wc;
                    s[8] += (Ct[4] * Ct[4]) * wc;
                    s[9] += (Ct[4] * Ct[2]) * wc;
                } else {
                    s[0] += Ct[0] * wc;
                    s[1] += (Ct[0] * Ct[0]) * wc;
                    s[2] += Ct[1] * wc;
                    s[3] += (Ct[1] * Ct[1]) * wc;
                    s[4] += (Ct[0] * Ct[1]) * wc;
                    s[5] += Ct[2] * wc;
                    s[6] += (Ct[2] * Ct[2]) * wc;
                    s[7] += (Ct[2] * Ct[1]) * wc;
                }
            }
#pragma unroll
            for (int k = 0; k < Mo; ++k)
                xconv[yy][threadIdx.x][k] = s[k];
        }

        // SSIM of one branch of the plain loss and its partials before the half rounding, as in
        // fusedL1SSIMForwardCUDA (Masked = false) or maskedFusedL1SSIMForwardCUDA.
        template <bool Masked>
        __device__ __forceinline__ float plain_branch(const float m1, const float m11, const float m2, const float m22,
                                                      const float m12, const float inv_window, const float C1,
                                                      const float C2, float& cs, float (&p)[4]) {
            float mu1, mu2, sigma1_sq, sigma2_sq, sigma12;
            float mu1_sq, mu2_sq;
            if constexpr (Masked) {
                mu1 = m1 * inv_window;
                mu2 = m2 * inv_window;
                mu1_sq = mu1 * mu1;
                mu2_sq = mu2 * mu2;
                sigma1_sq = m11 * inv_window - mu1_sq;
                sigma2_sq = m22 * inv_window - mu2_sq;
                sigma12 = m12 * inv_window - mu1 * mu2;
            } else {
                mu1 = m1;
                mu2 = m2;
                mu1_sq = mu1 * mu1;
                mu2_sq = mu2 * mu2;
                sigma1_sq = m11 - mu1_sq;
                sigma2_sq = m22 - mu2_sq;
                sigma12 = m12 - mu1 * mu2;
            }
            const float A = mu1_sq + mu2_sq + C1;
            const float B = sigma1_sq + sigma2_sq + C2;
            const float C_ = 2.f * mu1 * mu2 + C1;
            const float D_ = 2.f * sigma12 + C2;
            const float ssim_val = (C_ * D_) / (A * B);
            cs = D_ / B;
            p[0] = ((mu2 * 2.f * D_) / (A * B) - (mu2 * 2.f * C_) / (A * B) - (mu1 * 2.f * C_ * D_) / (A * A * B) + (mu1 * 2.f * C_ * D_) / (A * B * B));
            p[1] = (-C_ * D_) / (A * B * B);
            p[2] = (2.f * C_) / (A * B);
            p[3] = 0.0f;
            return ssim_val;
        }

        // The same for the decoupled loss (decoupledFusedL1SSIMForwardCUDA or the masked one): p[0..2]
        // the raw-render partials, p[3] the corrected-image partial, each with its final scaling except
        // the half rounding.
        template <bool Masked>
        __device__ __forceinline__ float decoupled_branch(const float corrected_conv, const float raw_conv,
                                                          const float raw_sq_conv, const float gt_conv,
                                                          const float gt_sq_conv, const float raw_gt_conv,
                                                          const float inv_window, const float derivative_weight,
                                                          const float ssim_weight, const float C1, const float C2,
                                                          float& cs, float (&p)[4]) {
            float mu_corrected, mu_raw, mu_gt, sigma_raw_sq, sigma_gt_sq, sigma12;
            float mu_corrected_sq, mu_raw_sq, mu_gt_sq;
            if constexpr (Masked) {
                mu_corrected = corrected_conv * inv_window;
                mu_raw = raw_conv * inv_window;
                mu_gt = gt_conv * inv_window;
                mu_corrected_sq = mu_corrected * mu_corrected;
                mu_raw_sq = mu_raw * mu_raw;
                mu_gt_sq = mu_gt * mu_gt;
                sigma_raw_sq = raw_sq_conv * inv_window - mu_raw_sq;
                sigma_gt_sq = gt_sq_conv * inv_window - mu_gt_sq;
                sigma12 = raw_gt_conv * inv_window - mu_raw * mu_gt;
            } else {
                mu_corrected = corrected_conv;
                mu_raw = raw_conv;
                mu_gt = gt_conv;
                mu_corrected_sq = mu_corrected * mu_corrected;
                mu_raw_sq = mu_raw * mu_raw;
                mu_gt_sq = mu_gt * mu_gt;
                sigma_raw_sq = raw_sq_conv - mu_raw_sq;
                sigma_gt_sq = gt_sq_conv - mu_gt_sq;
                sigma12 = raw_gt_conv - mu_raw * mu_gt;
            }
            const float A_app = mu_corrected_sq + mu_gt_sq + C1;
            const float C_app = 2.f * mu_corrected * mu_gt + C1;
            const float luminance = C_app / A_app;
            const float B_raw = sigma_raw_sq + sigma_gt_sq + C2;
            const float D_raw = 2.f * sigma12 + C2;
            const float contrast_structure = D_raw / B_raw;
            const float ssim_val = luminance * contrast_structure;
            cs = contrast_structure;
            const float dl_dmu1 = 2.f * (mu_gt * A_app - mu_corrected * C_app) / (A_app * A_app);
            const float ds1sq = ssim_weight * (-(luminance * D_raw) / (B_raw * B_raw));
            const float ds12 = ssim_weight * ((2.f * luminance) / B_raw);
            if constexpr (Masked) {
                p[3] = contrast_structure * dl_dmu1 * derivative_weight;
                p[1] = ds1sq * derivative_weight;
                p[2] = ds12 * derivative_weight;
                p[0] = (ds1sq * (-2.f * mu_raw) + ds12 * (-mu_gt)) * (derivative_weight * (1.0f / 16.0f));
            } else {
                p[3] = contrast_structure * dl_dmu1;
                p[1] = ds1sq;
                p[2] = ds12;
                p[0] = ds1sq * (-2.f * mu_raw) + ds12 * (-mu_gt);
            }
            return ssim_val;
        }

        // Both SSIM branches of one scale in one pass, the photo's moments shared, and the blend of
        // blend_blur_tolerant_kernel applied to the half partials as they are written. ssim_map and
        // cs_map are the render's branch, as the per-branch forward writes them.
        template <bool Masked, bool Decoupled, typename TargetT, typename MaskT>
        __global__ void blurTolerantForwardCUDA(const int H, const int W, const int CH, const float C1, const float C2,
                                                const float ssim_weight, const float* __restrict__ x,
                                                const float* __restrict__ raw, const TargetT* __restrict__ y,
                                                const float* __restrict__ bx, const float* __restrict__ braw,
                                                const MaskT* __restrict__ mask, const BranchPartials sharp,
                                                const BranchPartials blurred, const float4 blurred_scale,
                                                float* __restrict__ ssim_map, float* __restrict__ cs_map,
                                                float* __restrict__ blend_map, const float temperature) {
            constexpr int T = Decoupled ? 5 : 3;
            constexpr int Mo = Decoupled ? 10 : 8;
            constexpr int KP = Decoupled ? 4 : 3; // partials per branch
            // Odd per-pixel strides keep the 16-wide rows of a warp on distinct shared-memory banks.
            constexpr int MoS = Mo | 1;
            auto block = cg::this_thread_block();
            const int bIdx = block.group_index().z;
            const int pix_y = block.group_index().y * BLOCK_Y + block.thread_index().y;
            const int pix_x = block.group_index().x * BLOCK_X + block.thread_index().x;
            const int pix_id = pix_y * W + pix_x;
            const int num_pix = H * W;
            const bool in_image = pix_x < W && pix_y < H;
            const float scales[4] = {blurred_scale.x, blurred_scale.y, blurred_scale.z, blurred_scale.w};

            __shared__ float sTile[SHARED_Y][SHARED_X][T];
            __shared__ float xconv[CONV_Y][CONV_X][MoS];
            __shared__ float mask_tile[SHARED_Y][SHARED_X];

            float inv_window = 1.0f;
            float center_weight = 1.0f;
            float derivative_weight = 1.0f;
            bool full_support = true;
            if constexpr (Masked) {
                inv_window = load_mask_window(mask, H, W, mask_tile, xconv, full_support);
                center_weight = mask_tile[threadIdx.y + HALO][threadIdx.x + HALO];
                derivative_weight = center_weight * inv_window;
                // Ignored centers contribute no loss or partials, even near a kept tile.
                if (!__syncthreads_or(in_image && center_weight != 0.0f)) {
                    if (in_image) {
                        float sum = 0.0f;
                        for (int c = 0; c < CH; ++c) {
                            const int index = bIdx * CH * num_pix + c * num_pix + pix_id;
#pragma unroll
                            for (int k = 0; k < 4; ++k) {
                                if (sharp.p[k])
                                    sharp.p[k][index] = __float2half(0.0f);
                                if (blurred.p[k])
                                    blurred.p[k][index] = __float2half(0.0f);
                            }
                            const float s0 = 1.0f, s1 = 1.0f;
                            const float d = temperature * (s0 - s1);
                            sum += fmaxf(s0, s1) + (log1pf(expf(-fabsf(d))) - LN2) / temperature;
                        }
                        if (ssim_map)
                            ssim_map[bIdx * num_pix + pix_id] = 1.0f;
                        if (cs_map)
                            cs_map[bIdx * num_pix + pix_id] = 1.0f;
                        blend_map[bIdx * num_pix + pix_id] = sum / static_cast<float>(CH);
                    }
                    return;
                }
            }

            float ssim_sum = 0.0f;
            float cs_sum = 0.0f;
            float blend_sum = 0.0f;
            for (int c = 0; c < CH; ++c) {
                {
                    const int tileStartY = block.group_index().y * BLOCK_Y;
                    const int tileStartX = block.group_index().x * BLOCK_X;
                    for (int tid = block.thread_rank(); tid < SHARED_Y * SHARED_X; tid += BLOCK_X * BLOCK_Y) {
                        const int local_y = tid / SHARED_X;
                        const int local_x = tid % SHARED_X;
                        const int gy = tileStartY + local_y - HALO;
                        const int gx = tileStartX + local_x - HALO;
                        float* t = sTile[local_y][local_x];
                        if constexpr (Decoupled) {
                            t[0] = get_pix_value(x, bIdx, c, gy, gx, CH, H, W);
                            t[1] = get_pix_value(raw, bIdx, c, gy, gx, CH, H, W);
                            t[2] = get_pix_value(y, bIdx, c, gy, gx, CH, H, W);
                            t[3] = get_pix_value(bx, bIdx, c, gy, gx, CH, H, W);
                            t[4] = get_pix_value(braw, bIdx, c, gy, gx, CH, H, W);
                        } else {
                            t[0] = get_pix_value(x, bIdx, c, gy, gx, CH, H, W);
                            t[1] = get_pix_value(y, bIdx, c, gy, gx, CH, H, W);
                            t[2] = get_pix_value(bx, bIdx, c, gy, gx, CH, H, W);
                        }
                    }
                }
                block.sync();

                // The masked decoupled forward always weights its taps; the masked plain one only on
                // tiles without full support.
                const auto horizontal = [&](const int yy) {
                    if constexpr (!Masked)
                        dual_horizontal_moments<false, Decoupled>(sTile, mask_tile, xconv, yy);
                    else if constexpr (Decoupled)
                        dual_horizontal_moments<true, Decoupled>(sTile, mask_tile, xconv, yy);
                    else if (full_support)
                        dual_horizontal_moments<false, Decoupled>(sTile, mask_tile, xconv, yy);
                    else
                        dual_horizontal_moments<true, Decoupled>(sTile, mask_tile, xconv, yy);
                };
                horizontal(threadIdx.y);
                if (threadIdx.y + BLOCK_Y < CONV_Y)
                    horizontal(threadIdx.y + BLOCK_Y);
                block.sync();

                {
                    const int ly = threadIdx.y + HALO;
                    const int lx = threadIdx.x;
                    float o[Mo];
#pragma unroll
                    for (int k = 0; k < Mo; ++k)
                        o[k] = 0.0f;
#pragma unroll
                    for (int d = 1; d <= HALO; ++d) {
                        const float w = cGauss[HALO - d];
                        const float* top = xconv[ly - d][lx];
                        const float* bot = xconv[ly + d][lx];
#pragma unroll
                        for (int k = 0; k < Mo; ++k)
                            o[k] += (top[k] + bot[k]) * w;
                    }
                    {
                        const float wC = cGauss[HALO];
                        const float* ctr = xconv[ly][lx];
#pragma unroll
                        for (int k = 0; k < Mo; ++k)
                            o[k] += ctr[k] * wC;
                    }

                    if (in_image) {
                        float p0[4], p1[4];
                        float cs0, cs1;
                        float s0, s1;
                        if constexpr (Decoupled) {
                            s0 = decoupled_branch<Masked>(o[0], o[1], o[2], o[3], o[4], o[5], inv_window,
                                                          derivative_weight, ssim_weight, C1, C2, cs0, p0);
                            s1 = decoupled_branch<Masked>(o[6], o[7], o[8], o[3], o[4], o[9], inv_window,
                                                          derivative_weight, ssim_weight, C1, C2, cs1, p1);
                        } else {
                            s0 = plain_branch<Masked>(o[0], o[1], o[2], o[3], o[4], inv_window, C1, C2, cs0, p0);
                            s1 = plain_branch<Masked>(o[5], o[6], o[2], o[3], o[7], inv_window, C1, C2, cs1, p1);
                            if constexpr (Masked) {
#pragma unroll
                                for (int k = 0; k < 3; ++k) {
                                    p0[k] *= derivative_weight;
                                    p1[k] *= derivative_weight;
                                }
                            }
                        }
                        ssim_sum += s0;
                        cs_sum += cs0;

                        const float d = temperature * (s0 - s1);
                        blend_sum += fmaxf(s0, s1) + (log1pf(expf(-fabsf(d))) - LN2) / temperature;
                        const float w0 = 1.0f / (1.0f + expf(-d));
                        const float w1 = 1.0f / (1.0f + expf(d));
                        const int global_idx = bIdx * CH * num_pix + c * num_pix + pix_id;
#pragma unroll
                        for (int k = 0; k < KP; ++k) {
                            sharp.p[k][global_idx] = __float2half(__half2float(__float2half(p0[k])) * w0);
                            blurred.p[k][global_idx] =
                                __float2half(__half2float(__float2half(p1[k])) * (w1 * scales[k]));
                        }
                    }
                }
                block.sync();
            }

            if (in_image) {
                const float inv_c = 1.0f / static_cast<float>(CH);
                const bool kept = !Masked || center_weight > 0.0f;
                if (ssim_map)
                    ssim_map[bIdx * num_pix + pix_id] = kept ? ssim_sum * inv_c : 1.0f;
                if (cs_map)
                    cs_map[bIdx * num_pix + pix_id] = kept ? cs_sum * inv_c : 1.0f;
                blend_map[bIdx * num_pix + pix_id] = blend_sum / static_cast<float>(CH);
            }
        }

        // Backward of both branches of one scale: the render's gradient (with its L1 term) into grad,
        // the blurred branch's gradient with respect to the blurred render into bgrad; for the
        // decoupled loss also the raw render's into grad_raw and bgrad_raw. Each output is the one of
        // fusedL1SSIMBackwardCUDA / maskedFusedL1SSIMBackwardCUDA as the per-branch level calls it; per
        // channel the sharp and the blurred pass run in turn in the per-branch shared-memory layout.
        template <bool Masked, bool Decoupled, typename TargetT, typename MaskT>
        __global__ void blurTolerantBackwardCUDA(const float ssim_weight, const float inv_mask_sum,
                                                 const float grad_per_pixel, const int H, const int W, const int CH,
                                                 const float* __restrict__ x, const float* __restrict__ raw,
                                                 const TargetT* __restrict__ y, const float* __restrict__ bx,
                                                 const float* __restrict__ braw, const MaskT* __restrict__ mask,
                                                 const BranchPartials sharp, const BranchPartials blurred,
                                                 float* __restrict__ grad, float* __restrict__ grad_raw,
                                                 float* __restrict__ bgrad, float* __restrict__ bgrad_raw) {
            // Per pass (sharp, then blurred) the channels are mu, sigma^2, sigma12 (plain loss) or
            // corrected mu, raw mu, raw sigma^2, raw sigma12 (decoupled), in the per-branch layout.
            constexpr int K = Decoupled ? 4 : 3;
            auto block = cg::this_thread_block();
            const int pix_y = block.group_index().y * BLOCK_Y + block.thread_index().y;
            const int pix_x = block.group_index().x * BLOCK_X + block.thread_index().x;
            const int pix_id = pix_y * W + pix_x;
            const int num_pix = H * W;
            const int bIdx = block.group_index().z;
            const bool in_image = pix_x < W && pix_y < H;
            const float l1_weight = 1.0f - ssim_weight;
            const float blurred_l1_weight = 1.0f - 1.0f;

            float mask_val = 0.0f;
            if constexpr (Masked) {
                if (in_image)
                    mask_val = mask_value(mask, pix_y * W + pix_x);
                // No sample gradient can cross into an ignored output tile.
                if (!__syncthreads_or(mask_val != 0.0f)) {
                    if (in_image) {
                        for (int c = 0; c < CH; ++c) {
                            const int index = bIdx * CH * num_pix + c * num_pix + pix_id;
                            grad[index] = 0.0f;
                            bgrad[index] = 0.0f;
                            if constexpr (Decoupled) {
                                grad_raw[index] = 0.0f;
                                bgrad_raw[index] = 0.0f;
                            }
                        }
                    }
                    return;
                }
            }

            __shared__ float sData[SHARED_Y][SHARED_X][K];
            __shared__ float sScratch[CONV_Y][CONV_X][K];

            for (int c = 0; c < CH; ++c) {
                for (int pass = 0; pass < 2; ++pass) {
                    const BranchPartials& partials = pass == 0 ? sharp : blurred;
                    // The sharp pass carries the SSIM weight (on the corrected-image partial only for
                    // the decoupled loss), the blurred pass the weight one.
                    const float lead = pass == 0 ? ssim_weight : 1.0f;
                    constexpr float one = 1.0f;
                    float pa = 0.f, py = 0.f, pr = 0.f;
                    if (in_image) {
                        pa = get_pix_value(pass == 0 ? x : bx, bIdx, c, pix_y, pix_x, CH, H, W);
                        py = get_pix_value(y, bIdx, c, pix_y, pix_x, CH, H, W);
                        if constexpr (Decoupled)
                            pr = get_pix_value(pass == 0 ? raw : braw, bIdx, c, pix_y, pix_x, CH, H, W);
                    }

                    {
                        const int start_y = block.group_index().y * BLOCK_Y;
                        const int start_x = block.group_index().x * BLOCK_X;
                        const int tid = threadIdx.y * blockDim.x + threadIdx.x;
                        const int warp_id = tid / 32;
                        const int lane_id = tid % 32;
                        constexpr int num_warps = (BLOCK_X * BLOCK_Y + 31) / 32;
                        for (int row = warp_id; row < SHARED_Y; row += num_warps) {
                            const int gy = start_y + row - HALO;
                            for (int col = lane_id; col < SHARED_X; col += 32) {
                                const int gx = start_x + col - HALO;
                                float chain;
                                if constexpr (Masked) {
                                    chain = inv_mask_sum;
                                } else {
                                    chain = 0.0f;
                                    if (gx >= 0 && gx < W && gy >= 0 && gy < H) {
                                        const bool inside_valid_region = (H <= 10 || (gy >= 5 && gy < H - 5)) &&
                                                                         (W <= 10 || (gx >= 5 && gx < W - 5));
                                        if (inside_valid_region)
                                            chain = grad_per_pixel;
                                    }
                                }
                                const auto load = [&](const __half* p) {
                                    return get_pix_value(p, bIdx, c, gy, gx, CH, H, W);
                                };
                                float* out = sData[row][col];
                                if constexpr (Decoupled) {
                                    float raw_mu = load(partials.p[0]);
                                    if constexpr (Masked)
                                        raw_mu *= 16.0f;
                                    out[0] = -lead * load(partials.p[3]) * chain;
                                    out[1] = -one * raw_mu * chain;
                                    out[2] = -one * load(partials.p[1]) * chain;
                                    out[3] = -one * load(partials.p[2]) * chain;
                                } else {
                                    out[0] = -lead * load(partials.p[0]) * chain;
                                    out[1] = -lead * load(partials.p[1]) * chain;
                                    out[2] = -lead * load(partials.p[2]) * chain;
                                }
                            }
                        }
                    }
                    block.sync();

                    {
                        const int lx = threadIdx.x + HALO;
                        for (int half_pass = 0; half_pass < 2; ++half_pass) {
                            const int yy = threadIdx.y + half_pass * BLOCK_Y;
                            if (yy < CONV_Y) {
                                float accum[K];
#pragma unroll
                                for (int k = 0; k < K; ++k)
                                    accum[k] = 0.0f;
#pragma unroll
                                for (int d = 1; d <= HALO; ++d) {
                                    const float w = cGauss[HALO - d];
#pragma unroll
                                    for (int k = 0; k < K; ++k)
                                        accum[k] += (sData[yy][lx - d][k] + sData[yy][lx + d][k]) * w;
                                }
                                {
                                    const float wc = cGauss[HALO];
#pragma unroll
                                    for (int k = 0; k < K; ++k)
                                        accum[k] += sData[yy][lx][k] * wc;
                                }
#pragma unroll
                                for (int k = 0; k < K; ++k)
                                    sScratch[yy][threadIdx.x][k] = accum[k];
                            }
                        }
                    }
                    block.sync();

                    if (in_image) {
                        const int ly = threadIdx.y + HALO;
                        const int lx = threadIdx.x;
                        float sum[K];
#pragma unroll
                        for (int k = 0; k < K; ++k)
                            sum[k] = 0.0f;
#pragma unroll
                        for (int d = 1; d <= HALO; ++d) {
                            const float w = cGauss[HALO - d];
                            const float* top = sScratch[ly - d][lx];
                            const float* bot = sScratch[ly + d][lx];
#pragma unroll
                            for (int k = 0; k < K; ++k)
                                sum[k] += (top[k] + bot[k]) * w;
                        }
                        {
                            const float wc = cGauss[HALO];
                            const float* ctr = sScratch[ly][lx];
#pragma unroll
                            for (int k = 0; k < K; ++k)
                                sum[k] += ctr[k] * wc;
                        }

                        // One output of the per-branch backward: SSIM part from the convolved
                        // partials, L1 part with the branch's L1 weight.
                        const auto finish = [&](const float p1, const float ssim_part, const float weight_l1) {
                            const float sign_grad = (p1 == py) ? 0.0f : copysignf(1.0f, p1 - py);
                            if constexpr (Masked) {
                                const float grad_ssim = mask_val * ssim_part;
                                const float grad_l1 = weight_l1 * sign_grad * mask_val * inv_mask_sum;
                                return grad_ssim + grad_l1;
                            } else {
                                const bool inside_valid_region = (H <= 10 || (pix_y >= 5 && pix_y < H - 5)) &&
                                                                 (W <= 10 || (pix_x >= 5 && pix_x < W - 5));
                                const float chain_local = inside_valid_region ? grad_per_pixel : 0.0f;
                                const float grad_l1 = weight_l1 * sign_grad * chain_local;
                                return ssim_part + grad_l1;
                            }
                        };
                        const int out_idx = bIdx * CH * num_pix + c * num_pix + pix_id;
                        if constexpr (Decoupled) {
                            const float corrected = finish(pa, sum[0], pass == 0 ? l1_weight : blurred_l1_weight);
                            const float raw_grad =
                                finish(pr, sum[1] + (2.f * pr) * sum[2] + py * sum[3], blurred_l1_weight);
                            (pass == 0 ? grad : bgrad)[out_idx] = corrected;
                            (pass == 0 ? grad_raw : bgrad_raw)[out_idx] = raw_grad;
                        } else {
                            (pass == 0 ? grad : bgrad)[out_idx] =
                                finish(pa, sum[0] + (2.f * pa) * sum[1] + py * sum[2],
                                       pass == 0 ? l1_weight : blurred_l1_weight);
                        }
                    }
                    block.sync();
                }
            }
        }

        __global__ void accumulate_scaled_scalar_kernel(float* __restrict__ dst, const float* __restrict__ src,
                                                        const float scale, const bool overwrite) {
            if (blockIdx.x == 0 && threadIdx.x == 0)
                dst[0] = (overwrite ? 0.0f : dst[0]) + scale * src[0];
        }

        __global__ void scale_kernel(float* __restrict__ data, const size_t n, const float scale) {
            const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (idx < n)
                data[idx] *= scale;
        }

        // Contrast-structure error of one scale (one image): per pixel 1 - mean_c(w0 cs0 + w1 cs1),
        // cs0 = cs(x, y), cs1 = cs(G x, y) and w0, w1 the soft-maximum weights of the two SSIM branches
        // (the loss's blend); without the blurred render (blurred == nullptr) it is 1 - mean_c cs0.
        // Written as scale * value, or added to out when accumulate is set. Same windows and zero
        // padding as the loss kernels.
        template <typename TargetT>
        __global__ void blend_cs_error_kernel(const int H, const int W, const int CH, const float* __restrict__ x,
                                              const float* __restrict__ blurred, const TargetT* __restrict__ y,
                                              float* __restrict__ out, const float scale, const float temperature) {
            constexpr float C1 = 0.01f * 0.01f;
            constexpr float C2 = 0.03f * 0.03f;
            constexpr int SUMS = 8; // x, x^2, xy, b, b^2, by, y, y^2
            auto block = cg::this_thread_block();
            const int pix_y = block.group_index().y * BLOCK_Y + block.thread_index().y;
            const int pix_x = block.group_index().x * BLOCK_X + block.thread_index().x;
            __shared__ float tile[SHARED_Y][SHARED_X][3];
            __shared__ float xconv[CONV_Y][CONV_X][SUMS];
            const bool blur = blurred != nullptr;
            float cs_sum = 0.0f;

            for (int c = 0; c < CH; ++c) {
                const int tile_size = SHARED_Y * SHARED_X;
                const int threads = BLOCK_X * BLOCK_Y;
                const int tile_y0 = block.group_index().y * BLOCK_Y;
                const int tile_x0 = block.group_index().x * BLOCK_X;
                for (int t = block.thread_rank(); t < tile_size; t += threads) {
                    const int ly = t / SHARED_X, lx = t % SHARED_X;
                    const int gy = tile_y0 + ly - HALO, gx = tile_x0 + lx - HALO;
                    tile[ly][lx][0] = get_pix_value(x, 0, c, gy, gx, CH, H, W);
                    tile[ly][lx][1] = blur ? get_pix_value(blurred, 0, c, gy, gx, CH, H, W) : 0.0f;
                    tile[ly][lx][2] = get_pix_value(y, 0, c, gy, gx, CH, H, W);
                }
                block.sync();

                for (int ly = threadIdx.y; ly < CONV_Y; ly += BLOCK_Y) {
                    const int lx = threadIdx.x + HALO;
                    float s[SUMS] = {};
#pragma unroll
                    for (int d = -HALO; d <= HALO; ++d) {
                        const float w = cGauss[HALO + d];
                        const float X = tile[ly][lx + d][0], B = tile[ly][lx + d][1], Y = tile[ly][lx + d][2];
                        s[0] += X * w;
                        s[1] += X * X * w;
                        s[2] += X * Y * w;
                        s[3] += B * w;
                        s[4] += B * B * w;
                        s[5] += B * Y * w;
                        s[6] += Y * w;
                        s[7] += Y * Y * w;
                    }
#pragma unroll
                    for (int k = 0; k < SUMS; ++k)
                        xconv[ly][threadIdx.x][k] = s[k];
                }
                block.sync();

                {
                    const int ly = threadIdx.y + HALO;
                    const int lx = threadIdx.x;
                    float s[SUMS] = {};
#pragma unroll
                    for (int d = -HALO; d <= HALO; ++d) {
                        const float w = cGauss[HALO + d];
#pragma unroll
                        for (int k = 0; k < SUMS; ++k)
                            s[k] += xconv[ly + d][lx][k] * w;
                    }
                    const float mu_y = s[6];
                    const float var_y = s[7] - mu_y * mu_y;
                    const auto branch = [&](const float mu, const float sq, const float cross, float& cs) {
                        const float var = sq - mu * mu;
                        const float cov = cross - mu * mu_y;
                        const float A = mu * mu + mu_y * mu_y + C1;
                        const float B = var + var_y + C2;
                        cs = (2.0f * cov + C2) / B;
                        return (2.0f * mu * mu_y + C1) / A * cs;
                    };
                    float cs0, cs1 = 0.0f;
                    const float S0 = branch(s[0], s[1], s[2], cs0);
                    float blend = cs0;
                    if (blur) {
                        const float S1 = branch(s[3], s[4], s[5], cs1);
                        const float w0 = 1.0f / (1.0f + expf(-temperature * (S0 - S1)));
                        blend = w0 * cs0 + (1.0f - w0) * cs1;
                    }
                    cs_sum += blend;
                }
                block.sync();
            }

            if (pix_x < W && pix_y < H) {
                const int idx = pix_y * W + pix_x;
                const float value = scale * fmaxf(1.0f - cs_sum / static_cast<float>(CH), 0.0f);
                out[idx] = value;
            }
        }

        // fine[p] += scale * coarse[pixel pooling p], the coarse map `shift` halvings below the fine one.
        __global__ void add_upsampled_error_kernel(const float* __restrict__ coarse, float* __restrict__ fine,
                                                   const int H, const int W, const int w, const int shift,
                                                   const float scale) {
            const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (idx >= static_cast<size_t>(H) * W)
                return;
            const int py = static_cast<int>(idx / W), px = static_cast<int>(idx % W);
            fine[idx] += scale * coarse[static_cast<size_t>(py >> shift) * w + (px >> shift)];
        }

        void ensure_tensor(lfs::core::Tensor& tensor, const lfs::core::TensorShape& shape,
                           const lfs::core::DataType dtype = lfs::core::DataType::Float32) {
            if (!tensor.is_valid() || tensor.shape() != shape || tensor.dtype() != dtype)
                tensor = lfs::core::Tensor::empty(shape, lfs::core::Device::CUDA, dtype);
        }

        // ensure_tensor for a pyramid level's tensors: a view into the level's largest allocation so far
        // for that tensor, so photos of different sizes do not reallocate the pyramid every step.
        void ensure_level_tensor(PhotometricPyramidWorkspace::Level& level, lfs::core::Tensor& tensor,
                                 const lfs::core::TensorShape& shape,
                                 const lfs::core::DataType dtype = lfs::core::DataType::Float32) {
            if (tensor.is_valid() && tensor.shape() == shape && tensor.dtype() == dtype)
                return;
            const std::ptrdiff_t key = reinterpret_cast<const char*>(&tensor) - reinterpret_cast<const char*>(&level);
            LFS_ASSERT_MSG(key >= 0 && static_cast<size_t>(key) < sizeof(PhotometricPyramidWorkspace::Level),
                           "Pyramid level tensor must be a field of its level");
            const size_t bytes = shape.elements() * lfs::core::dtype_size(dtype);
            auto& blob = level.storage[key];
            if (!blob.is_valid() || blob.numel() < bytes)
                blob = lfs::core::Tensor::empty({bytes + bytes / 32}, lfs::core::Device::CUDA,
                                                lfs::core::DataType::UInt8);
            tensor = lfs::core::Tensor::from_blob(blob.data_ptr(), shape, lfs::core::Device::CUDA, dtype,
                                                  lfs::core::getCurrentCUDAStream());
        }

        template <typename MaskT>
        void launch_blur_support(const MaskT* mask, float* rows, float* inv_support, const int H, const int W,
                                 const BlurTaps& taps, const cudaStream_t stream) {
            const unsigned blocks = pyramid_blocks(static_cast<size_t>(H) * W);
            blur_support_rows_kernel<MaskT><<<blocks, PYRAMID_THREADS, 0, stream>>>(mask, rows, H, W, taps);
            LFS_CUDA_LAUNCH_CHECK(stream, "training.ssim.blur_support_rows");
            blur_support_cols_kernel<<<blocks, PYRAMID_THREADS, 0, stream>>>(rows, inv_support, H, W, taps);
            LFS_CUDA_LAUNCH_CHECK(stream, "training.ssim.blur_support_cols");
        }

        template <typename MaskT>
        void launch_blur(const float* in, const MaskT* mask, const float* inv_support, float* scratch, float* out,
                         const int planes, const int H, const int W, const BlurTaps& taps,
                         const cudaStream_t stream) {
            const unsigned blocks = pyramid_blocks(static_cast<size_t>(planes) * H * W);
            blur_rows_kernel<MaskT><<<blocks, PYRAMID_THREADS, 0, stream>>>(in, mask, scratch, planes, H, W, taps);
            LFS_CUDA_LAUNCH_CHECK(stream, "training.ssim.blur_rows");
            blur_cols_kernel<<<blocks, PYRAMID_THREADS, 0, stream>>>(scratch, inv_support, out, planes, H, W, taps);
            LFS_CUDA_LAUNCH_CHECK(stream, "training.ssim.blur_cols");
        }

        template <typename MaskT>
        void launch_blur_adjoint(const float* grad, const MaskT* mask, const float* inv_support, float* scratch,
                                 float* out, const bool accumulate, const int planes, const int H, const int W,
                                 const BlurTaps& taps, const cudaStream_t stream) {
            const unsigned blocks = pyramid_blocks(static_cast<size_t>(planes) * H * W);
            blur_adjoint_cols_kernel<<<blocks, PYRAMID_THREADS, 0, stream>>>(grad, inv_support, scratch, planes, H, W,
                                                                             taps);
            LFS_CUDA_LAUNCH_CHECK(stream, "training.ssim.blur_adjoint_cols");
            blur_adjoint_rows_kernel<MaskT><<<blocks, PYRAMID_THREADS, 0, stream>>>(scratch, mask, out, accumulate,
                                                                                    planes, H, W, taps);
            LFS_CUDA_LAUNCH_CHECK(stream, "training.ssim.blur_adjoint_rows");
        }

        struct PyramidLevelResult {
            lfs::core::Tensor loss;
            float mask_sum = 0.0f;
        };

        template <typename Workspace>
        inline constexpr bool is_masked_workspace = std::is_same_v<Workspace, MaskedFusedL1SSIMWorkspace> ||
                                                    std::is_same_v<Workspace, MaskedDecoupledFusedL1SSIMWorkspace>;
        template <typename Workspace>
        inline constexpr bool is_decoupled_workspace =
            std::is_same_v<Workspace, DecoupledFusedL1SSIMWorkspace> ||
            std::is_same_v<Workspace, MaskedDecoupledFusedL1SSIMWorkspace>;

        template <typename Workspace>
        lfs::core::Tensor& level_gradient(Workspace& ws) {
            if constexpr (is_decoupled_workspace<Workspace>)
                return ws.grad_corrected;
            else
                return ws.grad_img;
        }

        // One scale: today's loss forward and backward on (x, y), x the render (decoupled loss: the
        // corrected image, raw the raw render); with the blur-tolerant term also the blurred branch, the
        // blend and the blur adjoint. The render gradient lands in level_gradient(ws), the raw-render
        // gradient of the decoupled loss in grad_raw.
        template <typename TargetT, typename MaskT, typename Workspace>
        PyramidLevelResult run_pyramid_level(const float* x, const float* raw, const TargetT* y, const MaskT* mask,
                                             float* grad_raw, const int N, const int C, const int H, const int W,
                                             const float ssim_weight, const bool blur_tolerant, const float sigma,
                                             const float denominator, Workspace& ws,
                                             PhotometricPyramidWorkspace::Level& level,
                                             const cudaStream_t stream) {
            constexpr float C1 = 0.01f * 0.01f;
            constexpr float C2 = 0.03f * 0.03f;
            constexpr bool masked = is_masked_workspace<Workspace>;
            constexpr bool decoupled = is_decoupled_workspace<Workspace>;
            const dim3 grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, N);
            const dim3 block(BLOCK_X, BLOCK_Y);
            const lfs::core::TensorShape image_shape{static_cast<size_t>(N), static_cast<size_t>(C),
                                                     static_cast<size_t>(H), static_cast<size_t>(W)};
            const lfs::core::TensorShape plane_shape{static_cast<size_t>(H), static_cast<size_t>(W)};
            const int planes = N * C;

            BranchPartials sharp;
            if constexpr (decoupled) {
                sharp.p[0] = ws.raw_dm_dmu1.template ptr<__half>();
                sharp.p[1] = ws.raw_dm_dsigma1_sq.template ptr<__half>();
                sharp.p[2] = ws.raw_dm_dsigma12.template ptr<__half>();
                sharp.p[3] = ws.app_dm_dmu1.template ptr<__half>();
            } else {
                sharp.p[0] = ws.dm_dmu1.template ptr<__half>();
                sharp.p[1] = ws.dm_dsigma1_sq.template ptr<__half>();
                sharp.p[2] = ws.dm_dsigma12.template ptr<__half>();
            }
            BranchPartials blurred;
            float* ssim_sharp = nullptr;
            if (blur_tolerant) {
                ensure_level_tensor(level, level.blurred, image_shape);
                ensure_level_tensor(level, level.ssim_sharp, image_shape);
                ensure_level_tensor(level, level.ssim_blurred, image_shape);
                ensure_level_tensor(level, level.inv_support, plane_shape);
                ensure_level_tensor(level, level.support_scratch, plane_shape);
                ensure_level_tensor(level, level.blend_map, {static_cast<size_t>(N), 1, static_cast<size_t>(H), static_cast<size_t>(W)});
                ensure_level_tensor(level, level.dm_dmu1, image_shape, lfs::core::DataType::Float16);
                ensure_level_tensor(level, level.dm_dsigma1_sq, image_shape, lfs::core::DataType::Float16);
                ensure_level_tensor(level, level.dm_dsigma12, image_shape, lfs::core::DataType::Float16);
                blurred.p[0] = level.dm_dmu1.template ptr<__half>();
                blurred.p[1] = level.dm_dsigma1_sq.template ptr<__half>();
                blurred.p[2] = level.dm_dsigma12.template ptr<__half>();
                if constexpr (decoupled) {
                    ensure_level_tensor(level, level.blurred_raw, image_shape);
                    ensure_level_tensor(level, level.app_dm_dmu1, image_shape, lfs::core::DataType::Float16);
                    blurred.p[3] = level.app_dm_dmu1.template ptr<__half>();
                }
                ssim_sharp = level.ssim_sharp.template ptr<float>();
            }

            const auto forward = [&](const float* image, const float* raw_image, const BranchPartials& partials,
                                     float* ssim_map, float* cs_map, float* ssim_channels) {
                const auto& p = partials.p;
                if constexpr (decoupled && masked) {
                    maskedDecoupledFusedL1SSIMForwardCUDA<TargetT, MaskT><<<grid, block, 0, stream>>>(
                        H, W, C, C1, C2, ssim_weight, image, raw_image, y, mask, p[3], p[0], p[1], p[2], ssim_map,
                        cs_map, ssim_channels);
                } else if constexpr (decoupled) {
                    decoupledFusedL1SSIMForwardCUDA<TargetT><<<grid, block, 0, stream>>>(
                        H, W, C, C1, C2, ssim_weight, image, raw_image, y, p[3], p[0], p[1], p[2], ssim_map, cs_map,
                        ssim_channels);
                } else if constexpr (masked) {
                    maskedFusedL1SSIMForwardCUDA<TargetT, MaskT><<<grid, block, 0, stream>>>(
                        H, W, C, C1, C2, image, y, mask, p[0], p[1], p[2], ssim_map, cs_map, ssim_channels);
                } else {
                    fusedL1SSIMForwardCUDA<TargetT><<<grid, block, 0, stream>>>(
                        H, W, C, C1, C2, image, y, p[0], p[1], p[2], ssim_map, cs_map, ssim_channels);
                }
                LFS_CUDA_LAUNCH_CHECK(stream, "training.ssim.pyramid_forward");
            };

            forward(x, raw, sharp, ws.ssim_map.template ptr<float>(), ws.cs_map.template ptr<float>(), ssim_sharp);

            const float* loss_map = ws.ssim_map.template ptr<float>();
            BlurTaps taps{};
            if (blur_tolerant) {
                taps = make_blur_taps(sigma);
                launch_blur_support(mask, level.support_scratch.template ptr<float>(),
                                    level.inv_support.template ptr<float>(), H, W, taps, stream);
                launch_blur(x, mask, level.inv_support.template ptr<float>(), level.ssim_blurred.template ptr<float>(),
                            level.blurred.template ptr<float>(), planes, H, W, taps, stream);
                const float* blurred_raw = nullptr;
                if constexpr (decoupled) {
                    launch_blur(raw, mask, level.inv_support.template ptr<float>(),
                                level.ssim_blurred.template ptr<float>(), level.blurred_raw.template ptr<float>(),
                                planes, H, W, taps, stream);
                    blurred_raw = level.blurred_raw.template ptr<float>();
                }
                forward(level.blurred.template ptr<float>(), blurred_raw, blurred, nullptr, nullptr,
                        level.ssim_blurred.template ptr<float>());
                // The fused partials carry no SSIM weight; the decoupled raw-render partials do, its
                // corrected-image partial does not.
                const float4 blurred_scale = decoupled ? make_float4(1.0f, 1.0f, 1.0f, ssim_weight)
                                                       : make_float4(ssim_weight, ssim_weight, ssim_weight, 0.0f);
                blend_blur_tolerant_kernel<<<pyramid_blocks(static_cast<size_t>(N) * H * W), PYRAMID_THREADS, 0,
                                             stream>>>(ssim_sharp, level.ssim_blurred.template ptr<float>(), sharp,
                                                       blurred, blurred_scale, level.blend_map.template ptr<float>(),
                                                       N, C, H * W, BLUR_TOLERANT_SSIM_TEMPERATURE);
                LFS_CUDA_LAUNCH_CHECK(stream, "training.ssim.blur_tolerant_blend");
                loss_map = level.blend_map.template ptr<float>();
            }

            PyramidLevelResult result;
            float inv_mask_sum = 0.0f;
            float grad_per_pixel = 0.0f;
            if constexpr (masked) {
                launch_masked_fused_l1_ssim_mean_device(
                    x, y, loss_map, mask, ssim_weight, ws.reduction_temp.template ptr<float>(),
                    ws.masked_loss.template ptr<float>(), ws.mask_sum.template ptr<float>(), N, C, H, W, stream,
                    denominator);
                result.loss = ws.masked_loss;
                result.mask_sum = denominator > 0.0f ? denominator : ws.mask_sum.template item<float>();
                LFS_ASSERT_MSG(std::isfinite(result.mask_sum) && result.mask_sum > 0.0f,
                               "Masked loss normalization must be positive and finite");
                inv_mask_sum = 1.0f / result.mask_sum;
            } else {
                launch_fused_l1_ssim_mean_device(
                    x, y, loss_map, ssim_weight, ws.reduction_temp.template ptr<float>(),
                    ws.reduction_result.template ptr<float>(), N, C, H, W, /*apply_valid_padding=*/true, stream);
                result.loss = ws.reduction_result;
                const int valid_h = H > 10 ? H - 10 : H;
                const int valid_w = W > 10 ? W - 10 : W;
                grad_per_pixel = 1.0f / static_cast<float>(static_cast<size_t>(N) * C * valid_h * valid_w);
            }

            // Backward. The blurred branch runs with SSIM weight 1, which drops its L1 term; its
            // softmax weight and the SSIM weight are already in its partials.
            const auto render_backward = [&](const float* image, const float weight, float* out,
                                             const BranchPartials& partials) {
                const auto& p = partials.p;
                if constexpr (decoupled && masked) {
                    maskedFusedL1SSIMBackwardCUDA<TargetT, MaskT, /*HasSigmaPartials=*/false>
                        <<<grid, block, 0, stream>>>(weight, inv_mask_sum, H, W, C, C1, C2, image, y, mask, out, p[3],
                                                     nullptr, nullptr);
                } else if constexpr (decoupled) {
                    fusedL1SSIMBackwardCUDA<TargetT, /*HasSigmaPartials=*/false><<<grid, block, 0, stream>>>(
                        weight, H, W, C, C1, C2, grad_per_pixel, /*apply_valid_padding=*/true, image, y, out, p[3],
                        nullptr, nullptr);
                } else if constexpr (masked) {
                    maskedFusedL1SSIMBackwardCUDA<TargetT, MaskT, /*HasSigmaPartials=*/true>
                        <<<grid, block, 0, stream>>>(weight, inv_mask_sum, H, W, C, C1, C2, image, y, mask, out, p[0],
                                                     p[1], p[2]);
                } else {
                    fusedL1SSIMBackwardCUDA<TargetT><<<grid, block, 0, stream>>>(
                        weight, H, W, C, C1, C2, grad_per_pixel, /*apply_valid_padding=*/true, image, y, out, p[0],
                        p[1], p[2]);
                }
                LFS_CUDA_LAUNCH_CHECK(stream, "training.ssim.pyramid_backward");
            };
            // Decoupled loss: contrast and structure of the raw render (its partials carry the SSIM weight).
            const auto raw_backward = [&](const float* image, float* out, const BranchPartials& partials) {
                const auto& p = partials.p;
                if constexpr (masked) {
                    maskedFusedL1SSIMBackwardCUDA<TargetT, MaskT, /*HasSigmaPartials=*/true, /*Accumulate=*/false,
                                                  /*ScaledMeanPartial=*/true>
                        <<<grid, block, 0, stream>>>(1.0f, inv_mask_sum, H, W, C, C1, C2, image, y, mask, out, p[0],
                                                     p[1], p[2]);
                } else {
                    fusedL1SSIMBackwardCUDA<TargetT><<<grid, block, 0, stream>>>(
                        1.0f, H, W, C, C1, C2, grad_per_pixel, /*apply_valid_padding=*/true, image, y, out, p[0], p[1],
                        p[2]);
                }
                LFS_CUDA_LAUNCH_CHECK(stream, "training.ssim.pyramid_raw_backward");
            };

            float* grad = level_gradient(ws).template ptr<float>();
            render_backward(x, ssim_weight, grad, sharp);
            if constexpr (decoupled)
                raw_backward(raw, grad_raw, sharp);
            if (blur_tolerant) {
                // The render's SSIM is consumed by the blend; its buffer takes the blurred-branch gradient.
                float* blurred_grad = level.ssim_sharp.template ptr<float>();
                render_backward(level.blurred.template ptr<float>(), 1.0f, blurred_grad, blurred);
                launch_blur_adjoint(blurred_grad, mask, level.inv_support.template ptr<float>(),
                                    level.ssim_blurred.template ptr<float>(), grad, /*accumulate=*/true, planes, H, W,
                                    taps, stream);
                if constexpr (decoupled) {
                    raw_backward(level.blurred_raw.template ptr<float>(), blurred_grad, blurred);
                    launch_blur_adjoint(blurred_grad, mask, level.inv_support.template ptr<float>(),
                                        level.ssim_blurred.template ptr<float>(), grad_raw, /*accumulate=*/true,
                                        planes, H, W, taps, stream);
                }
            }
            return result;
        }

        // The blur of one scale's render (and raw render) over kept pixels, as run_pyramid_level
        // computes it, and the 2x2 pooled render (and raw render) of the next scale when coarser is
        // given, as the pyramid's pooling computes it.
        template <typename MaskT>
        void run_fused_blur(const float* x, const float* raw, const MaskT* mask, const int N, const int C, const int H,
                            const int W, const float sigma, PhotometricPyramidWorkspace::Level& level,
                            PhotometricPyramidWorkspace::Level* coarser, const int h, const int w,
                            const cudaStream_t stream) {
            const lfs::core::TensorShape image_shape{static_cast<size_t>(N), static_cast<size_t>(C),
                                                     static_cast<size_t>(H), static_cast<size_t>(W)};
            ensure_level_tensor(level, level.blurred, image_shape);
            ensure_level_tensor(level, level.inv_support, {static_cast<size_t>(H), static_cast<size_t>(W)});
            if (raw != nullptr)
                ensure_level_tensor(level, level.blurred_raw, image_shape);
            float* pool = nullptr;
            float* pool_raw = nullptr;
            if (coarser != nullptr) {
                const lfs::core::TensorShape coarse_shape{static_cast<size_t>(N), static_cast<size_t>(C),
                                                          static_cast<size_t>(h), static_cast<size_t>(w)};
                ensure_level_tensor(*coarser, coarser->prediction, coarse_shape);
                pool = coarser->prediction.template ptr<float>();
                if (raw != nullptr) {
                    ensure_level_tensor(*coarser, coarser->raw, coarse_shape);
                    pool_raw = coarser->raw.template ptr<float>();
                }
            }
            const BlurTaps taps = make_blur_taps(sigma);
            const dim3 grid((W + FUSED_BLUR_TX - 1) / FUSED_BLUR_TX, (H + FUSED_BLUR_TY - 1) / FUSED_BLUR_TY);
            const dim3 block(FUSED_BLUR_TX, FUSED_BLUR_TY);
            dispatch_blur_radius(taps.radius, [&](auto radius) {
                fused_blur_kernel<MaskT, decltype(radius)::value><<<grid, block, 0, stream>>>(
                    x, raw, mask, level.blurred.template ptr<float>(),
                    raw != nullptr ? level.blurred_raw.template ptr<float>() : nullptr,
                    level.inv_support.template ptr<float>(), pool, pool_raw, N * C, H, W, h, w, taps);
            });
            LFS_CUDA_LAUNCH_CHECK(stream, "training.ssim.fused_blur");
        }

        // One blur-tolerant scale with the fused kernels after run_fused_blur: the result of
        // run_pyramid_level with blur_tolerant set, followed by the coarser scale's gradient through
        // the pooling (when given) and the gradient scale (when apply_scale). bgrad buffers reuse the
        // per-branch SSIM maps.
        template <typename TargetT, typename MaskT, typename Workspace>
        PyramidLevelResult run_fused_blur_tolerant_level(const float* x, const float* raw, const TargetT* y,
                                                         const MaskT* mask, float* grad_raw, const int N, const int C,
                                                         const int H, const int W, const float ssim_weight,
                                                         const float sigma, const float denominator, Workspace& ws,
                                                         PhotometricPyramidWorkspace::Level& level,
                                                         const float* coarser_grad, const float* coarser_grad_raw,
                                                         const int coarse_h, const int coarse_w,
                                                         const float gradient_scale, const bool apply_scale,
                                                         const cudaStream_t stream) {
            constexpr float C1 = 0.01f * 0.01f;
            constexpr float C2 = 0.03f * 0.03f;
            constexpr bool masked = is_masked_workspace<Workspace>;
            constexpr bool decoupled = is_decoupled_workspace<Workspace>;
            const dim3 grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, N);
            const dim3 block(BLOCK_X, BLOCK_Y);
            const dim3 blur_grid((W + FUSED_BLUR_TX - 1) / FUSED_BLUR_TX, (H + FUSED_BLUR_TY - 1) / FUSED_BLUR_TY);
            const dim3 blur_block(FUSED_BLUR_TX, FUSED_BLUR_TY);
            const lfs::core::TensorShape image_shape{static_cast<size_t>(N), static_cast<size_t>(C),
                                                     static_cast<size_t>(H), static_cast<size_t>(W)};
            const int planes = N * C;

            BranchPartials sharp;
            if constexpr (decoupled) {
                sharp.p[0] = ws.raw_dm_dmu1.template ptr<__half>();
                sharp.p[1] = ws.raw_dm_dsigma1_sq.template ptr<__half>();
                sharp.p[2] = ws.raw_dm_dsigma12.template ptr<__half>();
                sharp.p[3] = ws.app_dm_dmu1.template ptr<__half>();
            } else {
                sharp.p[0] = ws.dm_dmu1.template ptr<__half>();
                sharp.p[1] = ws.dm_dsigma1_sq.template ptr<__half>();
                sharp.p[2] = ws.dm_dsigma12.template ptr<__half>();
            }
            ensure_level_tensor(level, level.blurred, image_shape);
            ensure_level_tensor(level, level.ssim_sharp, image_shape);
            ensure_level_tensor(level, level.inv_support, {static_cast<size_t>(H), static_cast<size_t>(W)});
            ensure_level_tensor(level, level.blend_map, {static_cast<size_t>(N), 1, static_cast<size_t>(H), static_cast<size_t>(W)});
            ensure_level_tensor(level, level.dm_dmu1, image_shape, lfs::core::DataType::Float16);
            ensure_level_tensor(level, level.dm_dsigma1_sq, image_shape, lfs::core::DataType::Float16);
            ensure_level_tensor(level, level.dm_dsigma12, image_shape, lfs::core::DataType::Float16);
            BranchPartials blurred;
            blurred.p[0] = level.dm_dmu1.template ptr<__half>();
            blurred.p[1] = level.dm_dsigma1_sq.template ptr<__half>();
            blurred.p[2] = level.dm_dsigma12.template ptr<__half>();
            const float* braw = nullptr;
            float* bgrad_raw = nullptr;
            if constexpr (decoupled) {
                ensure_level_tensor(level, level.blurred_raw, image_shape);
                ensure_level_tensor(level, level.ssim_blurred, image_shape);
                ensure_level_tensor(level, level.app_dm_dmu1, image_shape, lfs::core::DataType::Float16);
                blurred.p[3] = level.app_dm_dmu1.template ptr<__half>();
                braw = level.blurred_raw.template ptr<float>();
                bgrad_raw = level.ssim_blurred.template ptr<float>();
            }
            const float* bx = level.blurred.template ptr<float>();
            float* bgrad = level.ssim_sharp.template ptr<float>();
            const float* inv_support = level.inv_support.template ptr<float>();
            const BlurTaps taps = make_blur_taps(sigma);

            // The fused partials carry no SSIM weight; the decoupled raw-render partials do, its
            // corrected-image partial does not.
            const float4 blurred_scale = decoupled ? make_float4(1.0f, 1.0f, 1.0f, ssim_weight)
                                                   : make_float4(ssim_weight, ssim_weight, ssim_weight, 0.0f);
            blurTolerantForwardCUDA<masked, decoupled, TargetT, MaskT><<<grid, block, 0, stream>>>(
                H, W, C, C1, C2, ssim_weight, x, raw, y, bx, braw, mask, sharp, blurred, blurred_scale,
                ws.ssim_map.template ptr<float>(), ws.cs_map.template ptr<float>(),
                level.blend_map.template ptr<float>(), BLUR_TOLERANT_SSIM_TEMPERATURE);
            LFS_CUDA_LAUNCH_CHECK(stream, "training.ssim.blur_tolerant_forward");

            const float* loss_map = level.blend_map.template ptr<float>();
            PyramidLevelResult result;
            float inv_mask_sum = 0.0f;
            float grad_per_pixel = 0.0f;
            if constexpr (masked) {
                launch_masked_fused_l1_ssim_mean_device(
                    x, y, loss_map, mask, ssim_weight, ws.reduction_temp.template ptr<float>(),
                    ws.masked_loss.template ptr<float>(), ws.mask_sum.template ptr<float>(), N, C, H, W, stream,
                    denominator);
                result.loss = ws.masked_loss;
                result.mask_sum = denominator > 0.0f ? denominator : ws.mask_sum.template item<float>();
                LFS_ASSERT_MSG(std::isfinite(result.mask_sum) && result.mask_sum > 0.0f,
                               "Masked loss normalization must be positive and finite");
                inv_mask_sum = 1.0f / result.mask_sum;
            } else {
                launch_fused_l1_ssim_mean_device(
                    x, y, loss_map, ssim_weight, ws.reduction_temp.template ptr<float>(),
                    ws.reduction_result.template ptr<float>(), N, C, H, W, /*apply_valid_padding=*/true, stream);
                result.loss = ws.reduction_result;
                const int valid_h = H > 10 ? H - 10 : H;
                const int valid_w = W > 10 ? W - 10 : W;
                grad_per_pixel = 1.0f / static_cast<float>(static_cast<size_t>(N) * C * valid_h * valid_w);
            }

            float* grad = level_gradient(ws).template ptr<float>();
            blurTolerantBackwardCUDA<masked, decoupled, TargetT, MaskT><<<grid, block, 0, stream>>>(
                ssim_weight, inv_mask_sum, grad_per_pixel, H, W, C, x, raw, y, bx, braw, mask, sharp, blurred, grad,
                grad_raw, bgrad, bgrad_raw);
            LFS_CUDA_LAUNCH_CHECK(stream, "training.ssim.blur_tolerant_backward");

            dispatch_blur_radius(taps.radius, [&](auto radius) {
                fused_blur_adjoint_kernel<MaskT, decltype(radius)::value><<<blur_grid, blur_block, 0, stream>>>(
                    bgrad, bgrad_raw, inv_support, mask, grad, grad_raw, coarser_grad, coarser_grad_raw, planes, H,
                    W, coarse_h, coarse_w, taps, gradient_scale, apply_scale);
            });
            LFS_CUDA_LAUNCH_CHECK(stream, "training.ssim.fused_blur_adjoint");
            return result;
        }

        PhotometricPyramidResult pyramid_loss(const lfs::core::Tensor& prediction, const lfs::core::Tensor& raw_input,
                                              const lfs::core::Tensor& target, const lfs::core::Tensor& weight,
                                              const float ssim_weight, const int levels, const bool blur_tolerant,
                                              LossWorkspaceArena& arena, PhotometricPyramidWorkspace& workspace,
                                              const float denominator) {
            using lfs::core::Tensor;
            using lfs::core::TensorShape;
            LFS_ASSERT_MSG(std::isfinite(denominator) && denominator >= 0.0f,
                           "Loss denominator must be finite and nonnegative");
            LFS_ASSERT_MSG(levels >= 1 && levels <= MAX_PYRAMID_LEVELS,
                           lfs::core::detail::format_cuda_safe("Loss pyramid levels must be within [1, {}] (got {})",
                                                               MAX_PYRAMID_LEVELS, levels));
            validate_loss_weight(ssim_weight);
            const bool weighted = weight.is_valid() && weight.numel() > 0;
            const bool decoupled = raw_input.is_valid();

            auto prepared = prepare_loss_images(prediction, target);
            const Tensor& x0 = prepared.prediction;
            const Tensor& y0 = prepared.target;
            Tensor r0;
            if (decoupled) {
                r0 = prepare_loss_prediction(raw_input, "Raw loss image");
                LFS_ASSERT_MSG(r0.shape() == x0.shape(),
                               lfs::core::detail::format_cuda_safe(
                                   "Decoupled loss image shapes must match (corrected={}, raw={})", x0.shape().str(),
                                   r0.shape().str()));
            }
            const Tensor m0 = weighted ? prepare_loss_mask(weight, x0) : Tensor{};
            const int N = static_cast<int>(x0.shape()[0]);
            const int C = static_cast<int>(x0.shape()[1]);
            const auto stream = lfs::core::getCurrentCUDAStream();

            if (workspace.levels.size() < static_cast<size_t>(levels))
                workspace.levels.resize(static_cast<size_t>(levels));
            std::vector<int> heights(levels), widths(levels);
            heights[0] = static_cast<int>(x0.shape()[2]);
            widths[0] = static_cast<int>(x0.shape()[3]);

            // The fused blur-tolerant levels pool the render (and raw render) in their blur pass below.
            const bool fused = blur_tolerant && !workspace.reference_kernels;

            // Pool render (and raw render), photo and weights, scale by scale.
            for (int s = 1; s < levels; ++s) {
                const int H = heights[s - 1], W = widths[s - 1];
                const int h = (H + 1) / 2, w = (W + 1) / 2;
                heights[s] = h;
                widths[s] = w;
                auto& level = workspace.levels[s];
                const auto& finer = workspace.levels[s - 1];
                const TensorShape shape{static_cast<size_t>(N), static_cast<size_t>(C), static_cast<size_t>(h),
                                        static_cast<size_t>(w)};
                const unsigned blocks = pyramid_blocks(static_cast<size_t>(N) * C * h * w);
                const auto pool_image = [&](const float* in, Tensor& out) {
                    ensure_level_tensor(level, out, shape);
                    pool2x2_kernel<float, false><<<blocks, PYRAMID_THREADS, 0, stream>>>(in, out.ptr<float>(), N * C,
                                                                                         H, W, h, w);
                    LFS_CUDA_LAUNCH_CHECK(stream, "training.ssim.pyramid_pool_image");
                };
                if (!fused) {
                    pool_image(s == 1 ? x0.ptr<float>() : finer.prediction.ptr<float>(), level.prediction);
                    if (decoupled)
                        pool_image(s == 1 ? r0.ptr<float>() : finer.raw.ptr<float>(), level.raw);
                }
                ensure_level_tensor(level, level.target, shape);
                const auto pool_target = [&](const auto* in) {
                    using T = std::remove_cv_t<std::remove_pointer_t<decltype(in)>>;
                    pool2x2_kernel<T, false><<<blocks, PYRAMID_THREADS, 0, stream>>>(
                        in, level.target.ptr<float>(), N * C, H, W, h, w);
                    LFS_CUDA_LAUNCH_CHECK(stream, "training.ssim.pyramid_pool_target");
                };
                if (s == 1)
                    dispatch_target_ptr(y0, pool_target);
                else
                    pool_target(finer.target.ptr<float>());
                if (weighted) {
                    ensure_level_tensor(level, level.mask, {static_cast<size_t>(h), static_cast<size_t>(w)});
                    const unsigned mask_blocks = pyramid_blocks(static_cast<size_t>(h) * w);
                    const auto pool_mask = [&](const auto* in) {
                        using T = std::remove_cv_t<std::remove_pointer_t<decltype(in)>>;
                        pool2x2_kernel<T, true><<<mask_blocks, PYRAMID_THREADS, 0, stream>>>(
                            in, level.mask.ptr<float>(), 1, H, W, h, w);
                        LFS_CUDA_LAUNCH_CHECK(stream, "training.ssim.pyramid_pool_weight");
                    };
                    if (s == 1)
                        dispatch_mask_ptr(m0, pool_mask);
                    else
                        pool_mask(finer.mask.ptr<float>());
                }
            }

            // Fine to coarse: each scale's blur, which also pools the next scale's render.
            if (fused) {
                for (int s = 0; s < levels; ++s) {
                    const int H = heights[s], W = widths[s];
                    auto& level = workspace.levels[s];
                    auto* coarser = s + 1 < levels ? &workspace.levels[s + 1] : nullptr;
                    const int h = coarser ? heights[s + 1] : 0, w = coarser ? widths[s + 1] : 0;
                    const float* x = s == 0 ? x0.ptr<float>() : level.prediction.ptr<float>();
                    const float* r = decoupled ? (s == 0 ? r0.ptr<float>() : level.raw.ptr<float>()) : nullptr;
                    const float sigma = BLUR_TOLERANT_SSIM_SIGMA / static_cast<float>(1 << s);
                    const auto blur = [&](const auto* mask) {
                        run_fused_blur(x, r, mask, N, C, H, W, sigma, level, coarser, h, w, stream);
                    };
                    if (!weighted)
                        blur(static_cast<const float*>(nullptr));
                    else if (s == 0)
                        dispatch_mask_ptr(m0, blur);
                    else
                        blur(level.mask.ptr<float>());
                }
            }

            // Coarse to fine, so the finer scale can take the coarser gradient through the pooling and
            // scale 0 (in the arena) is computed last.
            const float level_scale = 1.0f / static_cast<float>(levels);
            const double full_pixels = static_cast<double>(heights[0]) * widths[0];
            Tensor coarser_grad, coarser_grad_raw;
            Tensor loss;
            float mask_sum_value = 0.0f;
            // The fused level also takes the coarser gradient through the pooling and the final 1 / K.
            for (int s = levels - 1; s >= 0; --s) {
                const int H = heights[s], W = widths[s];
                auto& level = workspace.levels[s];
                const std::vector<size_t> dims{static_cast<size_t>(N), static_cast<size_t>(C), static_cast<size_t>(H),
                                               static_cast<size_t>(W)};
                const float sigma = BLUR_TOLERANT_SSIM_SIGMA / static_cast<float>(1 << s);
                const float level_denominator =
                    denominator > 0.0f ? static_cast<float>(denominator * (static_cast<double>(H) * W / full_pixels))
                                       : 0.0f;
                const float* x = s == 0 ? x0.ptr<float>() : level.prediction.ptr<float>();
                const float* r = nullptr;
                float* grad_raw = nullptr;
                if (decoupled) {
                    r = s == 0 ? r0.ptr<float>() : level.raw.ptr<float>();
                    ensure_level_tensor(level, level.grad_raw, TensorShape(dims));
                    grad_raw = level.grad_raw.ptr<float>();
                }
                Tensor grad;
                PyramidLevelResult level_result;
                const bool coarser = coarser_grad.is_valid();
                const float* coarse = coarser ? coarser_grad.ptr<float>() : nullptr;
                const float* coarse_raw = coarser && decoupled ? coarser_grad_raw.ptr<float>() : nullptr;
                const int coarse_h = coarser ? heights[s + 1] : 0;
                const int coarse_w = coarser ? widths[s + 1] : 0;
                const auto run = [&](auto& ws, const auto* y, const auto* mask) {
                    ws.ensure_size(dims);
                    if (fused)
                        level_result = run_fused_blur_tolerant_level(
                            x, r, y, mask, grad_raw, N, C, H, W, ssim_weight, sigma, level_denominator, ws, level,
                            coarse, coarse_raw, coarse_h, coarse_w, level_scale, s == 0 && levels > 1, stream);
                    else
                        level_result = run_pyramid_level(x, r, y, mask, grad_raw, N, C, H, W, ssim_weight,
                                                         blur_tolerant, sigma, level_denominator, ws, level, stream);
                    grad = level_gradient(ws);
                };
                const auto run_on = [&](auto& ws) {
                    using Ws = std::remove_reference_t<decltype(ws)>;
                    if constexpr (is_masked_workspace<Ws>) {
                        if (s == 0)
                            dispatch_target_ptr(y0, [&](auto* y) { dispatch_mask_ptr(m0, [&](auto* m) { run(ws, y, m); }); });
                        else
                            run(ws, level.target.ptr<float>(), level.mask.ptr<float>());
                    } else {
                        if (s == 0)
                            dispatch_target_ptr(y0, [&](auto* y) { run(ws, y, static_cast<const float*>(nullptr)); });
                        else
                            run(ws, level.target.ptr<float>(), static_cast<const float*>(nullptr));
                    }
                };
                if (decoupled && weighted)
                    run_on(s == 0 ? arena.masked_decoupled() : level.arena.masked_decoupled());
                else if (decoupled)
                    run_on(s == 0 ? arena.decoupled() : level.arena.decoupled());
                else if (weighted)
                    run_on(s == 0 ? arena.masked_fused() : level.arena.masked_fused());
                else
                    run_on(s == 0 ? arena.fused() : level.arena.fused());
                if (s == 0)
                    mask_sum_value = level_result.mask_sum;

                if (coarser && !fused) {
                    const int h = heights[s + 1], w = widths[s + 1];
                    const unsigned blocks = pyramid_blocks(static_cast<size_t>(N) * C * H * W);
                    unpool2x2_adjoint_accumulate_kernel<<<blocks, PYRAMID_THREADS, 0, stream>>>(
                        coarser_grad.ptr<float>(), grad.ptr<float>(), N * C, H, W, h, w);
                    LFS_CUDA_LAUNCH_CHECK(stream, "training.ssim.pyramid_unpool_adjoint");
                    if (decoupled) {
                        unpool2x2_adjoint_accumulate_kernel<<<blocks, PYRAMID_THREADS, 0, stream>>>(
                            coarser_grad_raw.ptr<float>(), level.grad_raw.ptr<float>(), N * C, H, W, h, w);
                        LFS_CUDA_LAUNCH_CHECK(stream, "training.ssim.pyramid_unpool_adjoint_raw");
                    }
                }
                coarser_grad = grad;
                if (decoupled)
                    coarser_grad_raw = level.grad_raw;

                if (levels == 1) {
                    loss = level_result.loss;
                } else {
                    ensure_tensor(workspace.loss, {1});
                    accumulate_scaled_scalar_kernel<<<1, 1, 0, stream>>>(
                        workspace.loss.ptr<float>(), level_result.loss.ptr<float>(), level_scale, s == levels - 1);
                    LFS_CUDA_LAUNCH_CHECK(stream, "training.ssim.pyramid_loss_sum");
                    loss = workspace.loss;
                }
            }

            // The densification error map of the same loss, scale 0 written, coarser scales added.
            if (workspace.want_cs_error_map && N == 1) {
                const dim3 block(BLOCK_X, BLOCK_Y);
                ensure_tensor(workspace.cs_error_map,
                              {static_cast<size_t>(heights[0]), static_cast<size_t>(widths[0])});
                for (int s = 0; s < levels; ++s) {
                    const int H = heights[s], W = widths[s];
                    auto& level = workspace.levels[s];
                    const float* x = s == 0 ? x0.ptr<float>() : level.prediction.ptr<float>();
                    const float* blurred = blur_tolerant ? level.blurred.ptr<float>() : nullptr;
                    float* out = workspace.cs_error_map.ptr<float>();
                    if (s > 0) {
                        ensure_level_tensor(level, level.cs_error, {static_cast<size_t>(H), static_cast<size_t>(W)});
                        out = level.cs_error.ptr<float>();
                    }
                    const dim3 grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);
                    const auto launch = [&](const auto* y) {
                        blend_cs_error_kernel<<<grid, block, 0, stream>>>(H, W, C, x, blurred, y, out,
                                                                          s == 0 ? level_scale : 1.0f,
                                                                          BLUR_TOLERANT_SSIM_TEMPERATURE);
                    };
                    if (s == 0)
                        dispatch_target_ptr(y0, launch);
                    else
                        launch(level.target.ptr<float>());
                    LFS_CUDA_LAUNCH_CHECK(stream, "training.ssim.pyramid_cs_error");
                    if (s > 0) {
                        add_upsampled_error_kernel<<<pyramid_blocks(static_cast<size_t>(heights[0]) * widths[0]),
                                                     PYRAMID_THREADS, 0, stream>>>(
                            out, workspace.cs_error_map.ptr<float>(), heights[0], widths[0], W, s, level_scale);
                        LFS_CUDA_LAUNCH_CHECK(stream, "training.ssim.pyramid_cs_error_upsample");
                    }
                }
                workspace.cs_error_map_ready = true;
            }

            if (levels > 1 && !fused) {
                for (Tensor* g : {&coarser_grad, &coarser_grad_raw}) {
                    if (!g->is_valid())
                        continue;
                    scale_kernel<<<pyramid_blocks(g->numel()), PYRAMID_THREADS, 0, stream>>>(g->ptr<float>(),
                                                                                             g->numel(), level_scale);
                    LFS_CUDA_LAUNCH_CHECK(stream, "training.ssim.pyramid_scale_gradient");
                }
            }
            return {.loss = loss, .grad = coarser_grad, .grad_raw = coarser_grad_raw, .mask_sum_value = mask_sum_value};
        }
    } // namespace

    int loss_pyramid_levels(const int height, const int width) {
        LFS_ASSERT_MSG(height > 0 && width > 0, "Loss pyramid dimensions must be positive");
        const int short_side = std::min(height, width);
        if (short_side < PYRAMID_BASE_SIZE)
            return 1;
        const int octaves = static_cast<int>(
            std::floor(std::log2(static_cast<double>(short_side) / static_cast<double>(PYRAMID_BASE_SIZE))));
        return std::min(MAX_PYRAMID_LEVELS, octaves + 2);
    }

    PhotometricPyramidResult photometric_pyramid_loss(
        const lfs::core::Tensor& prediction,
        const lfs::core::Tensor& target,
        const lfs::core::Tensor& weight,
        const float ssim_weight,
        const int levels,
        const bool blur_tolerant,
        LossWorkspaceArena& arena,
        PhotometricPyramidWorkspace& workspace,
        const float denominator) {
        workspace.cs_error_map_ready = false;
        const bool weighted = weight.is_valid() && weight.numel() > 0;
        if (levels == 1 && !blur_tolerant) {
            validate_loss_weight(ssim_weight);
            if (weighted) {
                auto& ws = arena.masked_fused();
                auto [loss, ctx] = masked_fused_l1_ssim_forward(prediction, target, weight, ssim_weight, ws,
                                                                denominator);
                auto grad = masked_fused_l1_ssim_backward(ctx, ws);
                return {.loss = loss, .grad = grad, .mask_sum_value = ctx.mask_sum_value};
            }
            auto& ws = arena.fused();
            auto [loss, ctx] = fused_l1_ssim_forward(prediction, target, ssim_weight, ws, /*apply_valid_padding=*/true);
            auto grad = fused_l1_ssim_backward(ctx, ws);
            return {.loss = loss, .grad = grad};
        }
        return pyramid_loss(prediction, lfs::core::Tensor{}, target, weight, ssim_weight, levels, blur_tolerant, arena,
                            workspace, denominator);
    }

    PhotometricPyramidResult decoupled_photometric_pyramid_loss(
        const lfs::core::Tensor& corrected,
        const lfs::core::Tensor& raw,
        const lfs::core::Tensor& target,
        const lfs::core::Tensor& weight,
        const float ssim_weight,
        const int levels,
        const bool blur_tolerant,
        LossWorkspaceArena& arena,
        PhotometricPyramidWorkspace& workspace,
        const float denominator) {
        LFS_ASSERT_MSG(raw.is_valid() && raw.numel() > 0, "Decoupled pyramid loss needs the raw render");
        workspace.cs_error_map_ready = false;
        return pyramid_loss(corrected, raw, target, weight, ssim_weight, levels, blur_tolerant, arena, workspace,
                            denominator);
    }

    DecoupledRawGradient decoupled_raw_gradient(const PhotometricPyramidResult& result, const lfs::core::Tensor& raw) {
        LFS_ASSERT_MSG(result.grad_raw.is_valid(), "Pyramid loss result has no raw-render gradient");
        return DecoupledRawGradient{.raw_img = raw, .precomputed = result.grad_raw};
    }

    namespace {
        template <typename Fn>
        void dispatch_optional_mask(const lfs::core::Tensor& mask, Fn&& fn) {
            if (mask.is_valid())
                dispatch_mask_ptr(mask, fn);
            else
                fn(static_cast<const float*>(nullptr));
        }

        lfs::core::Tensor blur_or_adjoint(const lfs::core::Tensor& input, const lfs::core::Tensor& weight,
                                          const float sigma, const bool adjoint) {
            auto image = prepare_loss_prediction(input, adjoint ? "Blur adjoint gradient" : "Blur image");
            const lfs::core::Tensor mask =
                weight.is_valid() && weight.numel() > 0 ? prepare_loss_mask(weight, image) : lfs::core::Tensor{};
            const int planes = static_cast<int>(image.shape()[0] * image.shape()[1]);
            const int H = static_cast<int>(image.shape()[2]);
            const int W = static_cast<int>(image.shape()[3]);
            const auto taps = make_blur_taps(sigma);
            const auto stream = lfs::core::getCurrentCUDAStream();
            auto rows = lfs::core::Tensor::empty({static_cast<size_t>(H), static_cast<size_t>(W)}, lfs::core::Device::CUDA);
            auto inv_support = lfs::core::Tensor::empty({static_cast<size_t>(H), static_cast<size_t>(W)}, lfs::core::Device::CUDA);
            auto scratch = lfs::core::Tensor::empty(image.shape(), lfs::core::Device::CUDA);
            auto out = lfs::core::Tensor::empty(image.shape(), lfs::core::Device::CUDA);
            dispatch_optional_mask(mask, [&](const auto* mask_ptr) {
                launch_blur_support(mask_ptr, rows.ptr<float>(), inv_support.ptr<float>(), H, W, taps, stream);
                if (adjoint)
                    launch_blur_adjoint(image.ptr<float>(), mask_ptr, inv_support.ptr<float>(), scratch.ptr<float>(),
                                        out.ptr<float>(), /*accumulate=*/false, planes, H, W, taps, stream);
                else
                    launch_blur(image.ptr<float>(), mask_ptr, inv_support.ptr<float>(), scratch.ptr<float>(),
                                out.ptr<float>(), planes, H, W, taps, stream);
            });
            return out;
        }
    } // namespace

    lfs::core::Tensor normalized_gaussian_blur(const lfs::core::Tensor& image, const lfs::core::Tensor& weight,
                                               const float sigma) {
        return blur_or_adjoint(image, weight, sigma, /*adjoint=*/false);
    }

    lfs::core::Tensor normalized_gaussian_blur_adjoint(const lfs::core::Tensor& grad, const lfs::core::Tensor& weight,
                                                       const float sigma) {
        return blur_or_adjoint(grad, weight, sigma, /*adjoint=*/true);
    }

    // =========================================================================
    // LossWorkspaceArena + independent ensure_size fallbacks
    // =========================================================================
    namespace {
        constexpr size_t kArenaAlign = 256;

        size_t align_up_bytes(size_t x) {
            return (x + kArenaAlign - 1) & ~(kArenaAlign - 1);
        }

        size_t numel_of(const std::vector<size_t>& shape) {
            size_t n = 1;
            for (size_t d : shape) {
                n *= d;
            }
            return n;
        }

        size_t field_bytes(const std::vector<size_t>& shape, lfs::core::DataType dtype) {
            return numel_of(shape) * lfs::core::dtype_size(dtype);
        }

        size_t pack_fields(std::initializer_list<size_t> field_sizes) {
            size_t total = 0;
            for (size_t s : field_sizes) {
                total = align_up_bytes(total) + s;
            }
            return align_up_bytes(total);
        }

        void ensure_independent_cs_map(lfs::core::Tensor& cs_map, const lfs::core::Tensor& ssim_map) {
            if (!ssim_map.is_valid())
                return;
            if (cs_map.is_valid() && cs_map.shape() == ssim_map.shape() &&
                cs_map.dtype() == ssim_map.dtype() && cs_map.device() == ssim_map.device() &&
                cs_map.is_contiguous()) {
                return;
            }
            cs_map = ssim_map.device() == lfs::core::Device::CUDA
                         ? lfs::core::Tensor::empty_exact(ssim_map.shape(), ssim_map.dtype())
                         : lfs::core::Tensor::empty(ssim_map.shape(), ssim_map.device(), ssim_map.dtype());
        }
    } // namespace

    void SSIMWorkspace::ensure_size(const std::vector<size_t>& shape) {
        if (arena) {
            arena->ensure_pure_ssim(shape);
            ensure_independent_cs_map(cs_map, ssim_map);
            return;
        }
        if (allocated_shape != shape) {
            lfs::core::TensorShape tshape(shape);
            ssim_map = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA);
            dm_dmu1 = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA, lfs::core::DataType::Float16);
            dm_dsigma1_sq = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA, lfs::core::DataType::Float16);
            dm_dsigma12 = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA, lfs::core::DataType::Float16);
            dL_dmap = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA);
            dL_dimg1 = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA);
            reduction_temp = lfs::core::Tensor::empty({1024}, lfs::core::Device::CUDA);
            reduction_result = lfs::core::Tensor::empty({1}, lfs::core::Device::CUDA);
            allocated_shape = shape;
        }
        ensure_independent_cs_map(cs_map, ssim_map);
    }

    void FusedL1SSIMWorkspace::ensure_size(const std::vector<size_t>& shape) {
        if (arena) {
            arena->ensure_fused(shape);
            ensure_independent_cs_map(cs_map, ssim_map);
            return;
        }
        if (allocated_shape != shape) {
            lfs::core::TensorShape tshape(shape);
            std::vector<size_t> map_shape = shape;
            map_shape[1] = 1;
            ssim_map = lfs::core::Tensor::empty(lfs::core::TensorShape(map_shape), lfs::core::Device::CUDA);
            dm_dmu1 = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA, lfs::core::DataType::Float16);
            dm_dsigma1_sq = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA, lfs::core::DataType::Float16);
            dm_dsigma12 = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA, lfs::core::DataType::Float16);
            grad_img = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA);
            reduction_temp = lfs::core::Tensor::empty({1024}, lfs::core::Device::CUDA);
            reduction_result = lfs::core::Tensor::empty({1}, lfs::core::Device::CUDA);
            allocated_shape = shape;
        }
        ensure_independent_cs_map(cs_map, ssim_map);
    }

    void DecoupledFusedL1SSIMWorkspace::ensure_size(const std::vector<size_t>& shape) {
        if (arena) {
            arena->ensure_decoupled(shape);
            ensure_independent_cs_map(cs_map, ssim_map);
            return;
        }
        if (allocated_shape != shape) {
            lfs::core::TensorShape tshape(shape);
            std::vector<size_t> map_shape = shape;
            map_shape[1] = 1;
            ssim_map = lfs::core::Tensor::empty(lfs::core::TensorShape(map_shape), lfs::core::Device::CUDA);
            app_dm_dmu1 = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA, lfs::core::DataType::Float16);
            raw_dm_dmu1 = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA, lfs::core::DataType::Float16);
            raw_dm_dsigma1_sq = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA, lfs::core::DataType::Float16);
            raw_dm_dsigma12 = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA, lfs::core::DataType::Float16);
            grad_corrected = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA);
            reduction_temp = lfs::core::Tensor::empty({1024}, lfs::core::Device::CUDA);
            reduction_result = lfs::core::Tensor::empty({1}, lfs::core::Device::CUDA);
            allocated_shape = shape;
        }
        ensure_independent_cs_map(cs_map, ssim_map);
    }

    void MaskedFusedL1SSIMWorkspace::ensure_size(const std::vector<size_t>& shape) {
        if (arena) {
            arena->ensure_masked_fused(shape);
            ensure_independent_cs_map(cs_map, ssim_map);
            return;
        }
        if (allocated_shape != shape) {
            lfs::core::TensorShape tshape(shape);
            std::vector<size_t> map_shape = shape;
            map_shape[1] = 1;
            ssim_map = lfs::core::Tensor::empty(lfs::core::TensorShape(map_shape), lfs::core::Device::CUDA);
            dm_dmu1 = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA, lfs::core::DataType::Float16);
            dm_dsigma1_sq = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA, lfs::core::DataType::Float16);
            dm_dsigma12 = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA, lfs::core::DataType::Float16);
            grad_img = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA);
            reduction_temp = lfs::core::Tensor::empty({2048}, lfs::core::Device::CUDA);
            masked_loss = lfs::core::Tensor::empty({1}, lfs::core::Device::CUDA);
            mask_sum = lfs::core::Tensor::empty({1}, lfs::core::Device::CUDA);
            allocated_shape = shape;
        }
        ensure_independent_cs_map(cs_map, ssim_map);
    }

    void MaskedDecoupledFusedL1SSIMWorkspace::ensure_size(const std::vector<size_t>& shape) {
        if (arena) {
            arena->ensure_masked_decoupled(shape);
            ensure_independent_cs_map(cs_map, ssim_map);
            return;
        }
        if (allocated_shape != shape) {
            lfs::core::TensorShape tshape(shape);
            std::vector<size_t> map_shape = shape;
            map_shape[1] = 1;
            ssim_map = lfs::core::Tensor::empty(lfs::core::TensorShape(map_shape), lfs::core::Device::CUDA);
            app_dm_dmu1 = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA, lfs::core::DataType::Float16);
            raw_dm_dmu1 = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA, lfs::core::DataType::Float16);
            raw_dm_dsigma1_sq = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA, lfs::core::DataType::Float16);
            raw_dm_dsigma12 = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA, lfs::core::DataType::Float16);
            grad_corrected = lfs::core::Tensor::empty(tshape, lfs::core::Device::CUDA);
            reduction_temp = lfs::core::Tensor::empty({2048}, lfs::core::Device::CUDA);
            masked_loss = lfs::core::Tensor::empty({1}, lfs::core::Device::CUDA);
            mask_sum = lfs::core::Tensor::empty({1}, lfs::core::Device::CUDA);
            allocated_shape = shape;
        }
        ensure_independent_cs_map(cs_map, ssim_map);
    }

    size_t LossWorkspaceArena::fused_layout_bytes(const std::vector<size_t>& shape) {
        std::vector<size_t> map_shape = shape;
        map_shape[1] = 1;
        return pack_fields({
            field_bytes(map_shape, lfs::core::DataType::Float32),
            field_bytes(shape, lfs::core::DataType::Float16),
            field_bytes(shape, lfs::core::DataType::Float16),
            field_bytes(shape, lfs::core::DataType::Float16),
            field_bytes(shape, lfs::core::DataType::Float32),
            1024 * sizeof(float),
            sizeof(float),
        });
    }

    size_t LossWorkspaceArena::pure_ssim_layout_bytes(const std::vector<size_t>& shape) {
        // 3× dm_* fp16; ssim_map / dL_dmap / dL_dimg1 stay fp32.
        return pack_fields({
            field_bytes(shape, lfs::core::DataType::Float32),
            field_bytes(shape, lfs::core::DataType::Float16),
            field_bytes(shape, lfs::core::DataType::Float16),
            field_bytes(shape, lfs::core::DataType::Float16),
            field_bytes(shape, lfs::core::DataType::Float32),
            field_bytes(shape, lfs::core::DataType::Float32),
            1024 * sizeof(float),
            sizeof(float),
        });
    }

    size_t LossWorkspaceArena::decoupled_layout_bytes(const std::vector<size_t>& shape) {
        std::vector<size_t> map_shape = shape;
        map_shape[1] = 1;
        // Four dm_* fields use fp16; the corrected gradient remains fp32. The raw-render
        // gradient accumulates into the caller's buffer. There is no zero_terms field.
        return pack_fields({
            field_bytes(map_shape, lfs::core::DataType::Float32),
            field_bytes(shape, lfs::core::DataType::Float16),
            field_bytes(shape, lfs::core::DataType::Float16),
            field_bytes(shape, lfs::core::DataType::Float16),
            field_bytes(shape, lfs::core::DataType::Float16),
            field_bytes(shape, lfs::core::DataType::Float32),
            1024 * sizeof(float),
            sizeof(float),
        });
    }

    size_t LossWorkspaceArena::masked_fused_layout_bytes(const std::vector<size_t>& shape) {
        std::vector<size_t> map_shape = shape;
        map_shape[1] = 1;
        return pack_fields({
            field_bytes(map_shape, lfs::core::DataType::Float32),
            field_bytes(shape, lfs::core::DataType::Float16),
            field_bytes(shape, lfs::core::DataType::Float16),
            field_bytes(shape, lfs::core::DataType::Float16),
            field_bytes(shape, lfs::core::DataType::Float32),
            2048 * sizeof(float),
            sizeof(float),
            sizeof(float),
        });
    }

    size_t LossWorkspaceArena::masked_decoupled_layout_bytes(const std::vector<size_t>& shape) {
        std::vector<size_t> map_shape = shape;
        map_shape[1] = 1;
        // Four dm_* fields use fp16, with no zero_terms field.
        return pack_fields({
            field_bytes(map_shape, lfs::core::DataType::Float32),
            field_bytes(shape, lfs::core::DataType::Float16),
            field_bytes(shape, lfs::core::DataType::Float16),
            field_bytes(shape, lfs::core::DataType::Float16),
            field_bytes(shape, lfs::core::DataType::Float16),
            field_bytes(shape, lfs::core::DataType::Float32),
            2048 * sizeof(float),
            sizeof(float),
            sizeof(float),
        });
    }

    size_t LossWorkspaceArena::max_variant_bytes(const std::vector<size_t>& shape) {
        return std::max({fused_layout_bytes(shape), pure_ssim_layout_bytes(shape),
                         decoupled_layout_bytes(shape), masked_fused_layout_bytes(shape),
                         masked_decoupled_layout_bytes(shape)});
    }

    size_t LossWorkspaceArena::required_bytes() const {
        if (active_shape_.empty()) {
            return 0;
        }
        switch (active_kind_) {
        case Kind::Fused:
            return fused_layout_bytes(active_shape_);
        case Kind::PureSSIM:
            return pure_ssim_layout_bytes(active_shape_);
        case Kind::Decoupled:
            return decoupled_layout_bytes(active_shape_);
        case Kind::MaskedFused:
            return masked_fused_layout_bytes(active_shape_);
        case Kind::MaskedDecoupled:
            return masked_decoupled_layout_bytes(active_shape_);
        case Kind::None:
            return 0;
        }
        return 0;
    }

    void LossWorkspaceArena::replace_storage_exact(const size_t bytes) {
        const size_t required_capacity = align_up_bytes(bytes);
        LFS_ASSERT_MSG(required_capacity > 0,
                       "LossWorkspaceArena cannot allocate an empty active layout");

        auto new_storage = lfs::core::Tensor::empty_exact({required_capacity}, lfs::core::DataType::UInt8);
        // Stamp the home stream so from_blob views inherit a valid producer stream.
        if (new_storage.stream() != lfs::core::getCurrentCUDAStream()) {
            new_storage.set_stream(lfs::core::getCurrentCUDAStream());
        }

        clear_all_views();
        storage_ = std::move(new_storage);
        capacity_bytes_ = required_capacity;
        active_kind_ = Kind::None;
        active_shape_.clear();
    }

    void LossWorkspaceArena::ensure_capacity_for(const Kind kind, const size_t bytes) {
        const size_t required_capacity = align_up_bytes(bytes);
        const bool variant_changed = active_kind_ != Kind::None && active_kind_ != kind;
        const bool must_allocate = !storage_.is_valid() || variant_changed ||
                                   required_capacity > capacity_bytes_;
        if (!must_allocate) {
            return;
        }
        // Variant switches are exact in either direction. Within one variant,
        // shape changes only grow; smaller shapes rebind into the retained block.
        replace_storage_exact(required_capacity);
    }

    void LossWorkspaceArena::clear_all_views() {
        auto clear_fused = [](FusedL1SSIMWorkspace& w) {
            w.ssim_map = {};
            w.dm_dmu1 = {};
            w.dm_dsigma1_sq = {};
            w.dm_dsigma12 = {};
            w.grad_img = {};
            w.reduction_temp = {};
            w.reduction_result = {};
            w.allocated_shape.clear();
        };
        auto clear_ssim = [](SSIMWorkspace& w) {
            w.ssim_map = {};
            w.dm_dmu1 = {};
            w.dm_dsigma1_sq = {};
            w.dm_dsigma12 = {};
            w.dL_dmap = {};
            w.dL_dimg1 = {};
            w.reduction_temp = {};
            w.reduction_result = {};
            w.allocated_shape.clear();
        };
        auto clear_decoupled = [](DecoupledFusedL1SSIMWorkspace& w) {
            w.ssim_map = {};
            w.app_dm_dmu1 = {};
            w.raw_dm_dmu1 = {};
            w.raw_dm_dsigma1_sq = {};
            w.raw_dm_dsigma12 = {};
            w.grad_corrected = {};
            w.reduction_temp = {};
            w.reduction_result = {};
            w.allocated_shape.clear();
        };
        auto clear_masked = [](MaskedFusedL1SSIMWorkspace& w) {
            w.ssim_map = {};
            w.dm_dmu1 = {};
            w.dm_dsigma1_sq = {};
            w.dm_dsigma12 = {};
            w.grad_img = {};
            w.reduction_temp = {};
            w.masked_loss = {};
            w.mask_sum = {};
            w.allocated_shape.clear();
        };
        auto clear_masked_dec = [](MaskedDecoupledFusedL1SSIMWorkspace& w) {
            w.ssim_map = {};
            w.app_dm_dmu1 = {};
            w.raw_dm_dmu1 = {};
            w.raw_dm_dsigma1_sq = {};
            w.raw_dm_dsigma12 = {};
            w.grad_corrected = {};
            w.reduction_temp = {};
            w.masked_loss = {};
            w.mask_sum = {};
            w.allocated_shape.clear();
        };
        clear_fused(fused_);
        clear_ssim(pure_ssim_);
        clear_decoupled(decoupled_);
        clear_masked(masked_fused_);
        clear_masked_dec(masked_decoupled_);
        fused_.arena = this;
        pure_ssim_.arena = this;
        decoupled_.arena = this;
        masked_fused_.arena = this;
        masked_decoupled_.arena = this;
    }

    lfs::core::Tensor LossWorkspaceArena::make_view(size_t& offset,
                                                    const std::vector<size_t>& shape,
                                                    lfs::core::DataType dtype) {
        const size_t nbytes = field_bytes(shape, dtype);
        offset = align_up_bytes(offset);
        LFS_ASSERT_MSG(offset + nbytes <= capacity_bytes_,
                       lfs::core::detail::format_cuda_safe(
                           "LossWorkspaceArena view exceeds capacity (offset={}, need={}, cap={})",
                           offset, nbytes, capacity_bytes_));
        void* ptr = static_cast<char*>(storage_.data_ptr()) + offset;
        offset += nbytes;
        // Prefer current stream (matches training step); fall back to storage home.
        cudaStream_t home = lfs::core::getCurrentCUDAStream();
        if (home == nullptr) {
            home = storage_.stream();
        }
        return lfs::core::Tensor::from_blob(ptr, lfs::core::TensorShape(shape),
                                            lfs::core::Device::CUDA, dtype, home);
    }

    void LossWorkspaceArena::bind_fused(const std::vector<size_t>& shape) {
        std::vector<size_t> map_shape = shape;
        map_shape[1] = 1;
        size_t off = 0;
        fused_.ssim_map = make_view(off, map_shape, lfs::core::DataType::Float32);
        fused_.dm_dmu1 = make_view(off, shape, lfs::core::DataType::Float16);
        fused_.dm_dsigma1_sq = make_view(off, shape, lfs::core::DataType::Float16);
        fused_.dm_dsigma12 = make_view(off, shape, lfs::core::DataType::Float16);
        fused_.grad_img = make_view(off, shape, lfs::core::DataType::Float32);
        fused_.reduction_temp = make_view(off, {1024}, lfs::core::DataType::Float32);
        fused_.reduction_result = make_view(off, {1}, lfs::core::DataType::Float32);
        fused_.allocated_shape = shape;
        fused_.arena = this;
    }

    void LossWorkspaceArena::bind_pure_ssim(const std::vector<size_t>& shape) {
        size_t off = 0;
        pure_ssim_.ssim_map = make_view(off, shape, lfs::core::DataType::Float32);
        pure_ssim_.dm_dmu1 = make_view(off, shape, lfs::core::DataType::Float16);
        pure_ssim_.dm_dsigma1_sq = make_view(off, shape, lfs::core::DataType::Float16);
        pure_ssim_.dm_dsigma12 = make_view(off, shape, lfs::core::DataType::Float16);
        pure_ssim_.dL_dmap = make_view(off, shape, lfs::core::DataType::Float32);
        pure_ssim_.dL_dimg1 = make_view(off, shape, lfs::core::DataType::Float32);
        pure_ssim_.reduction_temp = make_view(off, {1024}, lfs::core::DataType::Float32);
        pure_ssim_.reduction_result = make_view(off, {1}, lfs::core::DataType::Float32);
        pure_ssim_.allocated_shape = shape;
        pure_ssim_.arena = this;
    }

    void LossWorkspaceArena::bind_decoupled(const std::vector<size_t>& shape) {
        std::vector<size_t> map_shape = shape;
        map_shape[1] = 1;
        size_t off = 0;
        decoupled_.ssim_map = make_view(off, map_shape, lfs::core::DataType::Float32);
        decoupled_.app_dm_dmu1 = make_view(off, shape, lfs::core::DataType::Float16);
        decoupled_.raw_dm_dmu1 = make_view(off, shape, lfs::core::DataType::Float16);
        decoupled_.raw_dm_dsigma1_sq = make_view(off, shape, lfs::core::DataType::Float16);
        decoupled_.raw_dm_dsigma12 = make_view(off, shape, lfs::core::DataType::Float16);
        decoupled_.grad_corrected = make_view(off, shape, lfs::core::DataType::Float32);
        decoupled_.reduction_temp = make_view(off, {1024}, lfs::core::DataType::Float32);
        decoupled_.reduction_result = make_view(off, {1}, lfs::core::DataType::Float32);
        decoupled_.allocated_shape = shape;
        decoupled_.arena = this;
    }

    void LossWorkspaceArena::bind_masked_fused(const std::vector<size_t>& shape) {
        std::vector<size_t> map_shape = shape;
        map_shape[1] = 1;
        size_t off = 0;
        masked_fused_.ssim_map = make_view(off, map_shape, lfs::core::DataType::Float32);
        masked_fused_.dm_dmu1 = make_view(off, shape, lfs::core::DataType::Float16);
        masked_fused_.dm_dsigma1_sq = make_view(off, shape, lfs::core::DataType::Float16);
        masked_fused_.dm_dsigma12 = make_view(off, shape, lfs::core::DataType::Float16);
        masked_fused_.grad_img = make_view(off, shape, lfs::core::DataType::Float32);
        masked_fused_.reduction_temp = make_view(off, {2048}, lfs::core::DataType::Float32);
        masked_fused_.masked_loss = make_view(off, {1}, lfs::core::DataType::Float32);
        masked_fused_.mask_sum = make_view(off, {1}, lfs::core::DataType::Float32);
        masked_fused_.allocated_shape = shape;
        masked_fused_.arena = this;
    }

    void LossWorkspaceArena::bind_masked_decoupled(const std::vector<size_t>& shape) {
        std::vector<size_t> map_shape = shape;
        map_shape[1] = 1;
        size_t off = 0;
        masked_decoupled_.ssim_map = make_view(off, map_shape, lfs::core::DataType::Float32);
        masked_decoupled_.app_dm_dmu1 = make_view(off, shape, lfs::core::DataType::Float16);
        masked_decoupled_.raw_dm_dmu1 = make_view(off, shape, lfs::core::DataType::Float16);
        masked_decoupled_.raw_dm_dsigma1_sq = make_view(off, shape, lfs::core::DataType::Float16);
        masked_decoupled_.raw_dm_dsigma12 = make_view(off, shape, lfs::core::DataType::Float16);
        masked_decoupled_.grad_corrected = make_view(off, shape, lfs::core::DataType::Float32);
        masked_decoupled_.reduction_temp = make_view(off, {2048}, lfs::core::DataType::Float32);
        masked_decoupled_.masked_loss = make_view(off, {1}, lfs::core::DataType::Float32);
        masked_decoupled_.mask_sum = make_view(off, {1}, lfs::core::DataType::Float32);
        masked_decoupled_.allocated_shape = shape;
        masked_decoupled_.arena = this;
    }

    FusedL1SSIMWorkspace& LossWorkspaceArena::ensure_fused(const std::vector<size_t>& shape) {
        ensure_capacity_for(Kind::Fused, fused_layout_bytes(shape));
        if (active_kind_ == Kind::Fused && active_shape_ == shape && fused_.ssim_map.is_valid()) {
            return fused_;
        }
        clear_all_views();
        bind_fused(shape);
        active_kind_ = Kind::Fused;
        active_shape_ = shape;
        return fused_;
    }

    SSIMWorkspace& LossWorkspaceArena::ensure_pure_ssim(const std::vector<size_t>& shape) {
        ensure_capacity_for(Kind::PureSSIM, pure_ssim_layout_bytes(shape));
        if (active_kind_ == Kind::PureSSIM && active_shape_ == shape && pure_ssim_.ssim_map.is_valid()) {
            return pure_ssim_;
        }
        clear_all_views();
        bind_pure_ssim(shape);
        active_kind_ = Kind::PureSSIM;
        active_shape_ = shape;
        return pure_ssim_;
    }

    DecoupledFusedL1SSIMWorkspace& LossWorkspaceArena::ensure_decoupled(const std::vector<size_t>& shape) {
        ensure_capacity_for(Kind::Decoupled, decoupled_layout_bytes(shape));
        if (active_kind_ == Kind::Decoupled && active_shape_ == shape && decoupled_.ssim_map.is_valid()) {
            return decoupled_;
        }
        clear_all_views();
        bind_decoupled(shape);
        active_kind_ = Kind::Decoupled;
        active_shape_ = shape;
        return decoupled_;
    }

    MaskedFusedL1SSIMWorkspace& LossWorkspaceArena::ensure_masked_fused(const std::vector<size_t>& shape) {
        ensure_capacity_for(Kind::MaskedFused, masked_fused_layout_bytes(shape));
        if (active_kind_ == Kind::MaskedFused && active_shape_ == shape &&
            masked_fused_.ssim_map.is_valid()) {
            return masked_fused_;
        }
        clear_all_views();
        bind_masked_fused(shape);
        active_kind_ = Kind::MaskedFused;
        active_shape_ = shape;
        return masked_fused_;
    }

    MaskedDecoupledFusedL1SSIMWorkspace&
    LossWorkspaceArena::ensure_masked_decoupled(const std::vector<size_t>& shape) {
        ensure_capacity_for(Kind::MaskedDecoupled, masked_decoupled_layout_bytes(shape));
        if (active_kind_ == Kind::MaskedDecoupled && active_shape_ == shape &&
            masked_decoupled_.ssim_map.is_valid()) {
            return masked_decoupled_;
        }
        clear_all_views();
        bind_masked_decoupled(shape);
        active_kind_ = Kind::MaskedDecoupled;
        active_shape_ = shape;
        return masked_decoupled_;
    }

    void LossWorkspaceArena::shrink_to_required() {
        if (active_kind_ == Kind::None || active_shape_.empty()) {
            reset();
            return;
        }

        const Kind kind = active_kind_;
        const std::vector<size_t> shape = active_shape_;
        const size_t required_capacity = align_up_bytes(required_bytes());
        if (storage_.is_valid() && capacity_bytes_ == required_capacity) {
            return;
        }

        replace_storage_exact(required_capacity);
        switch (kind) {
        case Kind::Fused:
            bind_fused(shape);
            break;
        case Kind::PureSSIM:
            bind_pure_ssim(shape);
            break;
        case Kind::Decoupled:
            bind_decoupled(shape);
            break;
        case Kind::MaskedFused:
            bind_masked_fused(shape);
            break;
        case Kind::MaskedDecoupled:
            bind_masked_decoupled(shape);
            break;
        case Kind::None:
            return;
        }
        active_kind_ = kind;
        active_shape_ = shape;
    }

    void LossWorkspaceArena::reset() {
        clear_all_views();
        storage_ = {};
        capacity_bytes_ = 0;
        active_kind_ = Kind::None;
        active_shape_.clear();
    }

} // namespace lfs::training::kernels
