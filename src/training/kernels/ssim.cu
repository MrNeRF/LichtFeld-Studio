/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/kernels/ssim.cuh"
#include "lfs/kernels/ssim_reduction.cuh"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include <algorithm>
#include <cooperative_groups.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace {

    // ------------------------------------------
    // Utility: Copy rectangular crop from src to dst
    // ------------------------------------------
    __global__ void copy_crop_kernel(
        const float* __restrict__ src,
        float* __restrict__ dst,
        int N, int C, int H, int W,
        int crop_h, int crop_w,
        int start_h, int start_w) {

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = N * C * crop_h * crop_w;

        if (idx < total) {
            int w = idx % crop_w;
            int h = (idx / crop_w) % crop_h;
            int c = (idx / (crop_w * crop_h)) % C;
            int n = idx / (crop_w * crop_h * C);

            int src_h = h + start_h;
            int src_w = w + start_w;

            int src_idx = n * (C * H * W) + c * (H * W) + src_h * W + src_w;
            dst[idx] = src[src_idx];
        }
    }

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

    // ------------------------------------------
    // Utility: Safe pixel fetch w/ zero padding
    // ------------------------------------------
    __device__ __forceinline__ float get_pix_value(
        const float* img,
        int b, int c, int y, int x,
        int CH, int H, int W) {
        if (x < 0 || x >= W || y < 0 || y >= H) {
            return 0.0f;
        }
        return img[b * CH * H * W + c * H * W + y * W + x];
    }

    // ------------------------------------------
    // Forward Kernel: Fused SSIM
    //  - Two-pass convolution to get mu1, mu2,
    //    sigma1_sq, sigma2_sq, sigma12, etc.
    //  - Writes final SSIM map to ssim_map
    //  - Optionally writes partial derivatives
    //    to dm_dmu1, dm_dsigma1_sq, dm_dsigma12
    // ------------------------------------------
    __global__ void fusedssimCUDA(
        int H,
        int W,
        int CH,
        float C1,
        float C2,
        const float* __restrict__ img1,
        const float* __restrict__ img2,
        float* __restrict__ ssim_map,
        float* __restrict__ dm_dmu1,
        float* __restrict__ dm_dsigma1_sq,
        float* __restrict__ dm_dsigma12) {
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

                    if (dm_dmu1) {
                        // partial derivatives
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
    }

    // ------------------------------------------
    // Backward Kernel: Apply chain rule to get
    //    dL/d(img1) from partial derivatives
    //    (dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
    //    and dL/dmap (the gradient from above).
    // ------------------------------------------
    __global__ void fusedssim_backwardCUDA(
        int H,
        int W,
        int CH,
        float C1,
        float C2,
        const float* __restrict__ img1,
        const float* __restrict__ img2,
        const float* __restrict__ dL_dmap,
        float* __restrict__ dL_dimg1,
        const float* __restrict__ dm_dmu1,
        const float* __restrict__ dm_dsigma1_sq,
        const float* __restrict__ dm_dsigma12) {
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
    // loss = (1-ssim_weight)*|img1-img2| + ssim_weight*(1-SSIM)
    __global__ void fusedL1SSIMForwardCUDA(
        float ssim_weight,
        int H,
        int W,
        int CH,
        float C1,
        float C2,
        const float* __restrict__ img1,
        const float* __restrict__ img2,
        float* __restrict__ loss_map,
        float* __restrict__ dm_dmu1,
        float* __restrict__ dm_dsigma1_sq,
        float* __restrict__ dm_dsigma12) {

        auto block = cg::this_thread_block();
        const int bIdx = block.group_index().z;
        const int pix_y = block.group_index().y * BLOCK_Y + block.thread_index().y;
        const int pix_x = block.group_index().x * BLOCK_X + block.thread_index().x;
        const int pix_id = pix_y * W + pix_x;
        const int num_pix = H * W;

        __shared__ float sTile[SHARED_Y][SHARED_X][2];
        __shared__ float xconv[CONV_Y][CONV_X][5];

        const float l1_weight = 1.0f - ssim_weight;

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

            // L1 loss from shared memory
            float l1_loss = fabsf(
                sTile[block.thread_index().y + HALO][block.thread_index().x + HALO][0] -
                sTile[block.thread_index().y + HALO][block.thread_index().x + HALO][1]);

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

                    // Combined loss: (1-w)*L1 + w*(1-SSIM)
                    loss_map[global_idx] = l1_weight * l1_loss + ssim_weight * (1.0f - ssim_val);

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
    }

    // Fused L1+SSIM Backward Kernel
    __global__ void fusedL1SSIMBackwardCUDA(
        float ssim_weight,
        int H,
        int W,
        int CH,
        float C1,
        float C2,
        const float* __restrict__ img1,
        const float* __restrict__ img2,
        const float* __restrict__ dL_dmap,
        float* __restrict__ dL_dimg1,
        const float* __restrict__ dm_dmu1,
        const float* __restrict__ dm_dsigma1_sq,
        const float* __restrict__ dm_dsigma12) {

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

                        float chain = get_pix_value(dL_dmap, bIdx, c, gy, gx, CH, H, W);
                        float vmu = get_pix_value(dm_dmu1, bIdx, c, gy, gx, CH, H, W);
                        float vs1 = get_pix_value(dm_dsigma1_sq, bIdx, c, gy, gx, CH, H, W);
                        float vs12 = get_pix_value(dm_dsigma12, bIdx, c, gy, gx, CH, H, W);

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
                float chain_local = get_pix_value(dL_dmap, bIdx, c, pix_y, pix_x, CH, H, W);
                float sign_grad = (p1 == p2) ? 0.0f : copysignf(1.0f, p1 - p2);
                float grad_l1 = l1_weight * sign_grad * chain_local;

                // Combined gradient
                dL_dimg1[out_idx] = grad_ssim + grad_l1;
            }
            block.sync();
        }
    }

    // Masked Fused L1+SSIM Forward Kernel
    __global__ void maskedFusedL1SSIMForwardCUDA(
        float ssim_weight,
        int H,
        int W,
        int CH,
        float C1,
        float C2,
        const float* __restrict__ img1,
        const float* __restrict__ img2,
        const float* __restrict__ mask, // [H, W] single channel
        float* __restrict__ loss_map,
        float* __restrict__ dm_dmu1,
        float* __restrict__ dm_dsigma1_sq,
        float* __restrict__ dm_dsigma12) {

        auto block = cg::this_thread_block();
        const int bIdx = block.group_index().z;
        const int pix_y = block.group_index().y * BLOCK_Y + block.thread_index().y;
        const int pix_x = block.group_index().x * BLOCK_X + block.thread_index().x;
        const int pix_id = pix_y * W + pix_x;
        const int num_pix = H * W;

        __shared__ float sTile[SHARED_Y][SHARED_X][2];
        __shared__ float xconv[CONV_Y][CONV_X][5];

        const float l1_weight = 1.0f - ssim_weight;

        // Get mask value for this pixel
        float mask_val = 0.0f;
        if (pix_x < W && pix_y < H) {
            mask_val = mask[pix_y * W + pix_x];
        }

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

            float l1_loss = fabsf(
                sTile[block.thread_index().y + HALO][block.thread_index().x + HALO][0] -
                sTile[block.thread_index().y + HALO][block.thread_index().x + HALO][1]);

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

                    // Masked combined loss: multiply by mask
                    float combined = l1_weight * l1_loss + ssim_weight * (1.0f - ssim_val);
                    loss_map[global_idx] = combined * mask_val;

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
    }

    // Masked Fused L1+SSIM Backward Kernel
    __global__ void maskedFusedL1SSIMBackwardCUDA(
        float ssim_weight,
        float inv_mask_sum, // 1.0 / mask_sum for normalization
        int H,
        int W,
        int CH,
        float C1,
        float C2,
        const float* __restrict__ img1,
        const float* __restrict__ img2,
        const float* __restrict__ mask,
        float* __restrict__ dL_dimg1,
        const float* __restrict__ dm_dmu1,
        const float* __restrict__ dm_dsigma1_sq,
        const float* __restrict__ dm_dsigma12) {

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
            mask_val = mask[pix_y * W + pix_x];
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

                        float local_mask = (gx >= 0 && gx < W && gy >= 0 && gy < H) ? mask[gy * W + gx] : 0.0f;
                        float chain = local_mask * inv_mask_sum;

                        float vmu = get_pix_value(dm_dmu1, bIdx, c, gy, gx, CH, H, W);
                        float vs1 = get_pix_value(dm_dsigma1_sq, bIdx, c, gy, gx, CH, H, W);
                        float vs12 = get_pix_value(dm_dsigma12, bIdx, c, gy, gx, CH, H, W);

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

                float grad_ssim = sum0 + (2.f * p1) * sum1 + p2 * sum2;

                // L1 gradient with mask
                float sign_grad = (p1 == p2) ? 0.0f : copysignf(1.0f, p1 - p2);
                float grad_l1 = l1_weight * sign_grad * mask_val * inv_mask_sum;

                int out_idx = bIdx * CH * num_pix + c * num_pix + pix_id;
                dL_dimg1[out_idx] = grad_ssim + grad_l1;
            }
            block.sync();
        }
    }

} // anonymous namespace

// LibTorch-Free API
namespace lfs::training::kernels {

    std::pair<lfs::core::Tensor, SSIMContext> ssim_forward(
        const lfs::core::Tensor& img1_input,
        const lfs::core::Tensor& img2_input,
        bool apply_valid_padding) {

        const float C1 = 0.01f * 0.01f;
        const float C2 = 0.03f * 0.03f;

        // Make tensors contiguous and ensure 4D [N, C, H, W]
        auto img1 = img1_input.contiguous();
        auto img2 = img2_input.contiguous();

        if (img1.ndim() == 3) {
            img1 = img1.unsqueeze(0);
        }
        if (img2.ndim() == 3) {
            img2 = img2.unsqueeze(0);
        }

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

        // Allocate derivative Tensors
        auto dm_dmu1 = lfs::core::Tensor::zeros(img1.shape(), lfs::core::Device::CUDA);
        auto dm_dsigma1_sq = lfs::core::Tensor::zeros(img1.shape(), lfs::core::Device::CUDA);
        auto dm_dsigma12 = lfs::core::Tensor::zeros(img1.shape(), lfs::core::Device::CUDA);

        fusedssimCUDA<<<grid, block>>>(
            H, W, C, C1, C2,
            img1.ptr<float>(),
            img2.ptr<float>(),
            ssim_map.ptr<float>(),
            dm_dmu1.ptr<float>(),
            dm_dsigma1_sq.ptr<float>(),
            dm_dsigma12.ptr<float>());

        // Store original dimensions
        int h = H;
        int w = W;

        // Apply valid padding (crop 5 pixels from each side) using efficient view slicing
        // Then compute mean using optimized tensor reduction (matches PyTorch speed!)
        lfs::core::Tensor ssim_map_cropped = ssim_map;
        if (apply_valid_padding && H > 10 && W > 10) {
            ssim_map_cropped = ssim_map.slice(2, 5, H - 5).slice(3, 5, W - 5);
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

        auto img1 = img1_input.contiguous();
        auto img2 = img2_input.contiguous();
        if (img1.ndim() == 3)
            img1 = img1.unsqueeze(0);
        if (img2.ndim() == 3)
            img2 = img2.unsqueeze(0);

        const int N = static_cast<int>(img1.shape()[0]);
        const int C = static_cast<int>(img1.shape()[1]);
        const int H = static_cast<int>(img1.shape()[2]);
        const int W = static_cast<int>(img1.shape()[3]);

        const dim3 grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, N);
        const dim3 block(BLOCK_X, BLOCK_Y);

        auto ssim_map = lfs::core::Tensor::zeros(img1.shape(), lfs::core::Device::CUDA);
        auto dm_dmu1 = lfs::core::Tensor::zeros(img1.shape(), lfs::core::Device::CUDA);
        auto dm_dsigma1_sq = lfs::core::Tensor::zeros(img1.shape(), lfs::core::Device::CUDA);
        auto dm_dsigma12 = lfs::core::Tensor::zeros(img1.shape(), lfs::core::Device::CUDA);

        fusedssimCUDA<<<grid, block>>>(
            H, W, C, C1, C2,
            img1.ptr<float>(), img2.ptr<float>(),
            ssim_map.ptr<float>(), dm_dmu1.ptr<float>(),
            dm_dsigma1_sq.ptr<float>(), dm_dsigma12.ptr<float>());

        lfs::core::Tensor ssim_map_for_mean = ssim_map;
        if (apply_valid_padding && H > 10 && W > 10) {
            ssim_map_for_mean = ssim_map.slice(2, 5, H - 5).slice(3, 5, W - 5);
        }

        return SSIMMapResult{
            .ssim_map = ssim_map,
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

    lfs::core::Tensor ssim_backward(
        const SSIMContext& ctx,
        float grad_loss) {

        const float C1 = 0.01f * 0.01f;
        const float C2 = 0.03f * 0.03f;

        // Compute gradient map size (after cropping if applicable)
        int grad_h = ctx.original_h;
        int grad_w = ctx.original_w;
        size_t N = ctx.img1.shape()[0];
        size_t C = ctx.img1.shape()[1];
        size_t numel = N * C * grad_h * grad_w;

        if (ctx.apply_valid_padding && grad_h > 10 && grad_w > 10) {
            grad_h -= 10; // Remove 5 pixels from each side
            grad_w -= 10;
            numel = N * C * grad_h * grad_w;
        }

        // Create gradient map: d(loss)/d(ssim_scalar) = grad_loss
        // d(ssim_scalar)/d(ssim_map[i]) = 1/numel
        // So: d(loss)/d(ssim_map[i]) = grad_loss / numel
        float grad_per_pixel = grad_loss / static_cast<float>(numel);

        // Create gradient tensor for cropped region
        auto dL_dmap = lfs::core::Tensor::zeros(ctx.img1.shape(), lfs::core::Device::CUDA);

        if (ctx.apply_valid_padding && ctx.original_h > 10 && ctx.original_w > 10) {
            auto cropped_view = dL_dmap.slice(2, 5, ctx.original_h - 5).slice(3, 5, ctx.original_w - 5);
            cropped_view.fill_(grad_per_pixel, lfs::core::getCurrentCUDAStream());
        } else {
            dL_dmap.fill_(grad_per_pixel, lfs::core::getCurrentCUDAStream());
        }

        // Allocate output gradient
        auto dL_dimg1 = lfs::core::Tensor::zeros(ctx.img1.shape(), lfs::core::Device::CUDA);

        // Launch backward kernel
        dim3 grid((ctx.original_w + BLOCK_X - 1) / BLOCK_X,
                  (ctx.original_h + BLOCK_Y - 1) / BLOCK_Y,
                  N);
        dim3 block(BLOCK_X, BLOCK_Y);

        fusedssim_backwardCUDA<<<grid, block>>>(
            ctx.original_h, ctx.original_w, static_cast<int>(C), C1, C2,
            ctx.img1.ptr<float>(),
            ctx.img2.ptr<float>(),
            dL_dmap.ptr<float>(),
            dL_dimg1.ptr<float>(),
            ctx.dm_dmu1.ptr<float>(),
            ctx.dm_dsigma1_sq.ptr<float>(),
            ctx.dm_dsigma12.ptr<float>());

        return dL_dimg1;
    }

    lfs::core::Tensor ssim_backward_with_grad_map(
        const SSIMContext& ctx,
        const lfs::core::Tensor& dL_dmap) {

        constexpr float C1 = 0.01f * 0.01f;
        constexpr float C2 = 0.03f * 0.03f;
        const size_t N = ctx.img1.shape()[0];
        const size_t C = ctx.img1.shape()[1];

        auto dL_dimg1 = lfs::core::Tensor::zeros(ctx.img1.shape(), lfs::core::Device::CUDA);
        const dim3 grid((ctx.original_w + BLOCK_X - 1) / BLOCK_X,
                        (ctx.original_h + BLOCK_Y - 1) / BLOCK_Y, N);
        const dim3 block(BLOCK_X, BLOCK_Y);

        fusedssim_backwardCUDA<<<grid, block>>>(
            ctx.original_h, ctx.original_w, static_cast<int>(C), C1, C2,
            ctx.img1.ptr<float>(), ctx.img2.ptr<float>(), dL_dmap.ptr<float>(),
            dL_dimg1.ptr<float>(), ctx.dm_dmu1.ptr<float>(),
            ctx.dm_dsigma1_sq.ptr<float>(), ctx.dm_dsigma12.ptr<float>());

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

        // Make tensors contiguous and ensure 4D [N, C, H, W]
        auto img1 = img1_input.contiguous();
        auto img2 = img2_input.contiguous();

        if (img1.ndim() == 3) {
            img1 = img1.unsqueeze(0);
        }
        if (img2.ndim() == 3) {
            img2 = img2.unsqueeze(0);
        }

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

        fusedssimCUDA<<<grid, block>>>(
            H, W, C, C1, C2,
            img1.ptr<float>(),
            img2.ptr<float>(),
            workspace.ssim_map.ptr<float>(),
            workspace.dm_dmu1.ptr<float>(),
            workspace.dm_dsigma1_sq.ptr<float>(),
            workspace.dm_dsigma12.ptr<float>());

        // Store original dimensions
        int h = H;
        int w = W;

        // Compute mean efficiently without .contiguous() allocation
        lfs::core::Tensor ssim_value_tensor;
        if (apply_valid_padding && H > 10 && W > 10) {
            // Use custom kernel to copy cropped region directly to pre-allocated buffer
            // This avoids the 8.6GB .contiguous() allocation that .slice().mean() causes!
            int crop_h = H - 10;
            int crop_w = W - 10;
            int total = N * C * crop_h * crop_w;
            int threads = 256;
            int blocks = (total + threads - 1) / threads;

            copy_crop_kernel<<<blocks, threads>>>(
                workspace.ssim_map.ptr<float>(),
                workspace.ssim_map_cropped.ptr<float>(),
                N, C, H, W,
                crop_h, crop_w,
                5, 5); // start_h=5, start_w=5

            ssim_value_tensor = workspace.ssim_map_cropped.mean();
        } else {
            // No cropping needed
            ssim_value_tensor = workspace.ssim_map.mean();
        }

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

        const float C1 = 0.01f * 0.01f;
        const float C2 = 0.03f * 0.03f;

        // Compute gradient map size (after cropping if applicable)
        int grad_h = ctx.original_h;
        int grad_w = ctx.original_w;
        size_t N = ctx.img1.shape()[0];
        size_t C = ctx.img1.shape()[1];
        size_t numel = N * C * grad_h * grad_w;

        if (ctx.apply_valid_padding && grad_h > 10 && grad_w > 10) {
            grad_h -= 10;
            grad_w -= 10;
            numel = N * C * grad_h * grad_w;
        }

        float grad_per_pixel = grad_loss / static_cast<float>(numel);

        // Use pre-allocated workspace buffer
        workspace.dL_dmap.zero_();

        if (ctx.apply_valid_padding && ctx.original_h > 10 && ctx.original_w > 10) {
            auto cropped_view = workspace.dL_dmap.slice(2, 5, ctx.original_h - 5).slice(3, 5, ctx.original_w - 5);
            cropped_view.fill_(grad_per_pixel, lfs::core::getCurrentCUDAStream());
        } else {
            workspace.dL_dmap.fill_(grad_per_pixel, lfs::core::getCurrentCUDAStream());
        }

        // Use pre-allocated output buffer
        workspace.dL_dimg1.zero_();

        // Launch backward kernel
        dim3 grid((ctx.original_w + BLOCK_X - 1) / BLOCK_X,
                  (ctx.original_h + BLOCK_Y - 1) / BLOCK_Y,
                  N);
        dim3 block(BLOCK_X, BLOCK_Y);

        fusedssim_backwardCUDA<<<grid, block>>>(
            ctx.original_h, ctx.original_w, static_cast<int>(C), C1, C2,
            ctx.img1.ptr<float>(),
            ctx.img2.ptr<float>(),
            workspace.dL_dmap.ptr<float>(),
            workspace.dL_dimg1.ptr<float>(),
            ctx.dm_dmu1.ptr<float>(),
            ctx.dm_dsigma1_sq.ptr<float>(),
            ctx.dm_dsigma12.ptr<float>());

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

        auto img1 = img1_input.contiguous();
        auto img2 = img2_input.contiguous();

        if (img1.ndim() == 3)
            img1 = img1.unsqueeze(0);
        if (img2.ndim() == 3)
            img2 = img2.unsqueeze(0);

        const int N = static_cast<int>(img1.shape()[0]);
        const int C = static_cast<int>(img1.shape()[1]);
        const int H = static_cast<int>(img1.shape()[2]);
        const int W = static_cast<int>(img1.shape()[3]);

        workspace.ensure_size(img1.shape().dims());

        const dim3 grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, N);
        const dim3 block(BLOCK_X, BLOCK_Y);

        fusedL1SSIMForwardCUDA<<<grid, block>>>(
            ssim_weight, H, W, C, C1, C2,
            img1.ptr<float>(), img2.ptr<float>(),
            workspace.loss_map.ptr<float>(),
            workspace.dm_dmu1.ptr<float>(),
            workspace.dm_dsigma1_sq.ptr<float>(),
            workspace.dm_dsigma12.ptr<float>());

        // Compute mean loss (with valid padding if requested)
        lfs::core::Tensor loss_scalar;
        if (apply_valid_padding && H > 10 && W > 10) {
            const int crop_h = H - 10;
            const int crop_w = W - 10;
            const int total = N * C * crop_h * crop_w;
            const int threads = 256;
            const int blocks = (total + threads - 1) / threads;

            copy_crop_kernel<<<blocks, threads>>>(
                workspace.loss_map.ptr<float>(),
                workspace.loss_map_cropped.ptr<float>(),
                N, C, H, W, crop_h, crop_w, 5, 5);

            loss_scalar = workspace.loss_map_cropped.mean();
        } else {
            loss_scalar = workspace.loss_map.mean();
        }

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

        constexpr float C1 = 0.01f * 0.01f;
        constexpr float C2 = 0.03f * 0.03f;

        const size_t N = ctx.img1.shape()[0];
        const size_t C = ctx.img1.shape()[1];

        // Compute gradient normalization factor
        int grad_h = ctx.H;
        int grad_w = ctx.W;
        if (ctx.apply_valid_padding && grad_h > 10 && grad_w > 10) {
            grad_h -= 10;
            grad_w -= 10;
        }
        const size_t numel = N * C * grad_h * grad_w;
        const float grad_per_pixel = 1.0f / static_cast<float>(numel);

        // Create gradient map (dL/d(loss_map))
        auto dL_dmap = lfs::core::Tensor::zeros(ctx.img1.shape(), lfs::core::Device::CUDA);
        if (ctx.apply_valid_padding && ctx.H > 10 && ctx.W > 10) {
            auto cropped = dL_dmap.slice(2, 5, ctx.H - 5).slice(3, 5, ctx.W - 5);
            cropped.fill_(grad_per_pixel, lfs::core::getCurrentCUDAStream());
        } else {
            dL_dmap.fill_(grad_per_pixel, lfs::core::getCurrentCUDAStream());
        }

        workspace.grad_img.zero_();

        const dim3 grid((ctx.W + BLOCK_X - 1) / BLOCK_X, (ctx.H + BLOCK_Y - 1) / BLOCK_Y, N);
        const dim3 block(BLOCK_X, BLOCK_Y);

        fusedL1SSIMBackwardCUDA<<<grid, block>>>(
            ctx.ssim_weight, ctx.H, ctx.W, static_cast<int>(C), C1, C2,
            ctx.img1.ptr<float>(), ctx.img2.ptr<float>(),
            dL_dmap.ptr<float>(), workspace.grad_img.ptr<float>(),
            ctx.dm_dmu1.ptr<float>(), ctx.dm_dsigma1_sq.ptr<float>(),
            ctx.dm_dsigma12.ptr<float>());

        return workspace.grad_img;
    }

    // ============================================================================
    // Masked Fused L1+SSIM Implementation
    // ============================================================================

    std::pair<lfs::core::Tensor, MaskedFusedL1SSIMContext> masked_fused_l1_ssim_forward(
        const lfs::core::Tensor& img1_input,
        const lfs::core::Tensor& img2_input,
        const lfs::core::Tensor& mask_input,
        float ssim_weight,
        MaskedFusedL1SSIMWorkspace& workspace) {

        constexpr float C1 = 0.01f * 0.01f;
        constexpr float C2 = 0.03f * 0.03f;

        auto img1 = img1_input.contiguous();
        auto img2 = img2_input.contiguous();
        auto mask = mask_input.contiguous();

        if (img1.ndim() == 3)
            img1 = img1.unsqueeze(0);
        if (img2.ndim() == 3)
            img2 = img2.unsqueeze(0);

        // Ensure mask is 2D [H, W]
        auto mask_2d = mask.ndim() == 3 ? mask.squeeze(0) : mask;

        const int N = static_cast<int>(img1.shape()[0]);
        const int C = static_cast<int>(img1.shape()[1]);
        const int H = static_cast<int>(img1.shape()[2]);
        const int W = static_cast<int>(img1.shape()[3]);

        workspace.ensure_size(img1.shape().dims());

        const dim3 grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, N);
        const dim3 block(BLOCK_X, BLOCK_Y);

        maskedFusedL1SSIMForwardCUDA<<<grid, block>>>(
            ssim_weight, H, W, C, C1, C2,
            img1.ptr<float>(), img2.ptr<float>(), mask_2d.ptr<float>(),
            workspace.loss_map.ptr<float>(),
            workspace.dm_dmu1.ptr<float>(),
            workspace.dm_dsigma1_sq.ptr<float>(),
            workspace.dm_dsigma12.ptr<float>());

        // Compute masked mean: sum(loss_map) / (mask_sum * C)
        // Note: loss_map already has mask applied per-pixel
        const float loss_sum = workspace.loss_map.sum().item<float>();
        const float mask_sum = mask_2d.sum().item<float>() * static_cast<float>(C) + SSIM_EPSILON;
        const float loss_value = loss_sum / mask_sum;

        auto loss_scalar = lfs::core::Tensor::full({1}, loss_value, lfs::core::Device::CUDA);

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

    lfs::core::Tensor masked_fused_l1_ssim_backward(
        const MaskedFusedL1SSIMContext& ctx,
        MaskedFusedL1SSIMWorkspace& workspace) {

        constexpr float C1 = 0.01f * 0.01f;
        constexpr float C2 = 0.03f * 0.03f;

        const size_t N = ctx.img1.shape()[0];
        const size_t C = ctx.img1.shape()[1];
        const float inv_mask_sum = 1.0f / ctx.mask_sum_value;

        workspace.grad_img.zero_();

        const dim3 grid((ctx.W + BLOCK_X - 1) / BLOCK_X, (ctx.H + BLOCK_Y - 1) / BLOCK_Y, N);
        const dim3 block(BLOCK_X, BLOCK_Y);

        maskedFusedL1SSIMBackwardCUDA<<<grid, block>>>(
            ctx.ssim_weight, inv_mask_sum, ctx.H, ctx.W, static_cast<int>(C), C1, C2,
            ctx.img1.ptr<float>(), ctx.img2.ptr<float>(), ctx.mask.ptr<float>(),
            workspace.grad_img.ptr<float>(),
            ctx.dm_dmu1.ptr<float>(), ctx.dm_dsigma1_sq.ptr<float>(),
            ctx.dm_dsigma12.ptr<float>());

        return workspace.grad_img;
    }

} // namespace lfs::training::kernels
