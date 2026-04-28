/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/kernels/ssim.cuh"
#include "lfs/kernels/ssim_reduction.cuh"
#include <algorithm>
#include <cassert>
#include <cooperative_groups.h>
#include <cuda_runtime.h>

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
    __global__ void fusedL1SSIMForwardCUDA(
        int H,
        int W,
        int CH,
        float C1,
        float C2,
        const float* __restrict__ img1,
        const float* __restrict__ img2,
        float* __restrict__ dm_dmu1,
        float* __restrict__ dm_dsigma1_sq,
        float* __restrict__ dm_dsigma12,
        float* __restrict__ ssim_map) {

        auto block = cg::this_thread_block();
        const int bIdx = block.group_index().z;
        const int pix_y = block.group_index().y * BLOCK_Y + block.thread_index().y;
        const int pix_x = block.group_index().x * BLOCK_X + block.thread_index().x;
        const int pix_id = pix_y * W + pix_x;
        const int num_pix = H * W;

        __shared__ float sTile[SHARED_Y][SHARED_X][2];
        __shared__ float xconv[CONV_Y][CONV_X][5];

        float ssim_sum = 0.0f;
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

        if (ssim_map && pix_x < W && pix_y < H) {
            ssim_map[bIdx * num_pix + pix_id] = ssim_sum / static_cast<float>(CH);
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
        float grad_per_pixel,
        bool apply_valid_padding,
        const float* __restrict__ img1,
        const float* __restrict__ img2,
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

                        float chain = 0.0f;
                        if (gx >= 0 && gx < W && gy >= 0 && gy < H) {
                            const bool inside_valid_region =
                                !apply_valid_padding ||
                                (H <= 10 || W <= 10) ||
                                (gx >= 5 && gx < W - 5 && gy >= 5 && gy < H - 5);
                            if (inside_valid_region) {
                                chain = grad_per_pixel;
                            }
                        }
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
                float chain_local = 0.0f;
                const bool inside_valid_region =
                    !apply_valid_padding ||
                    (H <= 10 || W <= 10) ||
                    (pix_x >= 5 && pix_x < W - 5 && pix_y >= 5 && pix_y < H - 5);
                if (inside_valid_region) {
                    chain_local = grad_per_pixel;
                }
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
        int H,
        int W,
        int CH,
        float C1,
        float C2,
        const float* __restrict__ img1,
        const float* __restrict__ img2,
        float* __restrict__ dm_dmu1,
        float* __restrict__ dm_dsigma1_sq,
        float* __restrict__ dm_dsigma12,
        float* __restrict__ ssim_map) {

        auto block = cg::this_thread_block();
        const int bIdx = block.group_index().z;
        const int pix_y = block.group_index().y * BLOCK_Y + block.thread_index().y;
        const int pix_x = block.group_index().x * BLOCK_X + block.thread_index().x;
        const int pix_id = pix_y * W + pix_x;
        const int num_pix = H * W;

        __shared__ float sTile[SHARED_Y][SHARED_X][2];
        __shared__ float xconv[CONV_Y][CONV_X][5];

        float ssim_sum = 0.0f;
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

                    ssim_sum += ssim_val;

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

        if (ssim_map && pix_x < W && pix_y < H) {
            ssim_map[bIdx * num_pix + pix_id] = ssim_sum / static_cast<float>(CH);
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

    __global__ void decoupledFusedL1SSIMForwardCUDA(
        int H,
        int W,
        int CH,
        float C1,
        float C2,
        float ssim_weight,
        const float* __restrict__ corrected_img,
        const float* __restrict__ raw_img,
        const float* __restrict__ gt_img,
        float* __restrict__ app_dm_dmu1,
        float* __restrict__ raw_dm_dmu1,
        float* __restrict__ raw_dm_dsigma1_sq,
        float* __restrict__ raw_dm_dsigma12,
        float* __restrict__ ssim_map) {

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

                    if (app_dm_dmu1) {
                        const float dl_dmu1 =
                            2.f * (mu_gt * A_app - mu_corrected * C_app) / (A_app * A_app);
                        app_dm_dmu1[global_idx] = contrast_structure * dl_dmu1;

                        const float ds1sq = ssim_weight * (-(luminance * D_raw) / (B_raw * B_raw));
                        const float ds12 = ssim_weight * ((2.f * luminance) / B_raw);
                        raw_dm_dsigma1_sq[global_idx] = ds1sq;
                        raw_dm_dsigma12[global_idx] = ds12;
                        raw_dm_dmu1[global_idx] = ds1sq * (-2.f * mu_raw) + ds12 * (-mu_gt);
                    }
                }
            }
        }

        if (ssim_map && pix_x < W && pix_y < H) {
            ssim_map[bIdx * num_pix + pix_id] = ssim_sum / static_cast<float>(CH);
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

    void ssim_error_map_forward(
        const lfs::core::Tensor& img1_input,
        const lfs::core::Tensor& img2_input,
        SSIMMapWorkspace& workspace,
        lfs::core::Tensor& error_map) {

        constexpr float C1 = 0.01f * 0.01f;
        constexpr float C2 = 0.03f * 0.03f;

        auto img1 = img1_input.contiguous();
        auto img2 = img2_input.contiguous();
        if (img1.ndim() == 3)
            img1 = img1.unsqueeze(0);
        if (img2.ndim() == 3)
            img2 = img2.unsqueeze(0);

        assert(img1.shape() == img2.shape());

        const int N = static_cast<int>(img1.shape()[0]);
        const int C = static_cast<int>(img1.shape()[1]);
        const int H = static_cast<int>(img1.shape()[2]);
        const int W = static_cast<int>(img1.shape()[3]);

        workspace.ensure_size(img1.shape().dims());

        const dim3 grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, N);
        const dim3 block(BLOCK_X, BLOCK_Y);

        fusedssimCUDA<<<grid, block>>>(
            H, W, C, C1, C2,
            img1.ptr<float>(), img2.ptr<float>(),
            workspace.ssim_map.ptr<float>(),
            nullptr, nullptr, nullptr);

        if (!error_map.is_valid() ||
            error_map.ndim() != 2 ||
            error_map.shape()[0] != static_cast<size_t>(H) ||
            error_map.shape()[1] != static_cast<size_t>(W)) {
            error_map = lfs::core::Tensor::empty({static_cast<size_t>(H), static_cast<size_t>(W)},
                                                 lfs::core::Device::CUDA);
        }

        launch_ssim_to_error_map(workspace.ssim_map, error_map);
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
            // Fill cropped region with gradient (use stream-aware version to avoid sync)
            auto cropped_view = dL_dmap.slice(2, 5, ctx.original_h - 5).slice(3, 5, ctx.original_w - 5);
            cropped_view.fill_(grad_per_pixel, nullptr); // stream-aware version, no sync
        } else {
            // No cropping - fill entire map (use stream-aware version to avoid sync)
            dL_dmap.fill_(grad_per_pixel, nullptr);
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

        launch_fused_ssim_mean_device(
            workspace.ssim_map.ptr<float>(),
            workspace.reduction_temp.ptr<float>(),
            workspace.reduction_result.ptr<float>(),
            N, C, H, W,
            apply_valid_padding,
            workspace.ssim_map.stream());
        lfs::core::Tensor ssim_value_tensor = workspace.reduction_result.clone();

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
            cropped_view.fill_(grad_per_pixel, nullptr); // stream-aware version, no sync
        } else {
            workspace.dL_dmap.fill_(grad_per_pixel, nullptr);
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
            H, W, C, C1, C2,
            img1.ptr<float>(), img2.ptr<float>(),
            workspace.dm_dmu1.ptr<float>(),
            workspace.dm_dsigma1_sq.ptr<float>(),
            workspace.dm_dsigma12.ptr<float>(),
            workspace.ssim_map.ptr<float>());

        launch_fused_l1_ssim_mean_device(
            img1.ptr<float>(),
            img2.ptr<float>(),
            workspace.ssim_map.ptr<float>(),
            ssim_weight,
            workspace.reduction_temp.ptr<float>(),
            workspace.reduction_result.ptr<float>(),
            N, C, H, W,
            apply_valid_padding,
            workspace.ssim_map.stream());
        lfs::core::Tensor loss_scalar = workspace.reduction_result.clone();

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

        workspace.grad_img.zero_();

        const dim3 grid((ctx.W + BLOCK_X - 1) / BLOCK_X, (ctx.H + BLOCK_Y - 1) / BLOCK_Y, N);
        const dim3 block(BLOCK_X, BLOCK_Y);

        fusedL1SSIMBackwardCUDA<<<grid, block>>>(
            ctx.ssim_weight, ctx.H, ctx.W, static_cast<int>(C), C1, C2,
            grad_per_pixel, ctx.apply_valid_padding,
            ctx.img1.ptr<float>(), ctx.img2.ptr<float>(),
            workspace.grad_img.ptr<float>(),
            ctx.dm_dmu1.ptr<float>(), ctx.dm_dsigma1_sq.ptr<float>(),
            ctx.dm_dsigma12.ptr<float>());

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

        auto corrected = corrected_input.contiguous();
        auto raw = raw_input.contiguous();
        auto gt = gt_input.contiguous();

        if (corrected.ndim() == 3)
            corrected = corrected.unsqueeze(0);
        if (raw.ndim() == 3)
            raw = raw.unsqueeze(0);
        if (gt.ndim() == 3)
            gt = gt.unsqueeze(0);

        assert(corrected.shape() == raw.shape());
        assert(corrected.shape() == gt.shape());

        const int N = static_cast<int>(corrected.shape()[0]);
        const int C = static_cast<int>(corrected.shape()[1]);
        const int H = static_cast<int>(corrected.shape()[2]);
        const int W = static_cast<int>(corrected.shape()[3]);

        workspace.ensure_size(corrected.shape().dims());

        const dim3 grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, N);
        const dim3 block(BLOCK_X, BLOCK_Y);

        decoupledFusedL1SSIMForwardCUDA<<<grid, block>>>(
            H, W, C, C1, C2, ssim_weight,
            corrected.ptr<float>(), raw.ptr<float>(), gt.ptr<float>(),
            workspace.app_dm_dmu1.ptr<float>(),
            workspace.raw_dm_dmu1.ptr<float>(),
            workspace.raw_dm_dsigma1_sq.ptr<float>(),
            workspace.raw_dm_dsigma12.ptr<float>(),
            workspace.ssim_map.ptr<float>());

        launch_fused_l1_ssim_mean_device(
            corrected.ptr<float>(),
            gt.ptr<float>(),
            workspace.ssim_map.ptr<float>(),
            ssim_weight,
            workspace.reduction_temp.ptr<float>(),
            workspace.reduction_result.ptr<float>(),
            N, C, H, W,
            apply_valid_padding,
            workspace.ssim_map.stream());
        lfs::core::Tensor loss_scalar = workspace.reduction_result.clone();

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

        constexpr float C1 = 0.01f * 0.01f;
        constexpr float C2 = 0.03f * 0.03f;

        const size_t N = ctx.corrected_img.shape()[0];
        const size_t C = ctx.corrected_img.shape()[1];

        int grad_h = ctx.H;
        int grad_w = ctx.W;
        if (ctx.apply_valid_padding && grad_h > 10 && grad_w > 10) {
            grad_h -= 10;
            grad_w -= 10;
        }
        const size_t numel = N * C * grad_h * grad_w;
        const float grad_per_pixel = 1.0f / static_cast<float>(numel);

        const dim3 grid((ctx.W + BLOCK_X - 1) / BLOCK_X, (ctx.H + BLOCK_Y - 1) / BLOCK_Y, N);
        const dim3 block(BLOCK_X, BLOCK_Y);

        workspace.grad_corrected.zero_();
        fusedL1SSIMBackwardCUDA<<<grid, block>>>(
            ctx.ssim_weight, ctx.H, ctx.W, static_cast<int>(C), C1, C2,
            grad_per_pixel, ctx.apply_valid_padding,
            ctx.corrected_img.ptr<float>(), ctx.gt_img.ptr<float>(),
            workspace.grad_corrected.ptr<float>(),
            ctx.app_dm_dmu1.ptr<float>(),
            workspace.zero_terms.ptr<float>(),
            workspace.zero_terms.ptr<float>());

        workspace.grad_raw.zero_();
        fusedL1SSIMBackwardCUDA<<<grid, block>>>(
            1.0f, ctx.H, ctx.W, static_cast<int>(C), C1, C2,
            grad_per_pixel, ctx.apply_valid_padding,
            ctx.raw_img.ptr<float>(), ctx.gt_img.ptr<float>(),
            workspace.grad_raw.ptr<float>(),
            ctx.raw_dm_dmu1.ptr<float>(),
            ctx.raw_dm_dsigma1_sq.ptr<float>(),
            ctx.raw_dm_dsigma12.ptr<float>());

        return DecoupledGradients{
            .grad_corrected = workspace.grad_corrected,
            .grad_raw = workspace.grad_raw};
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
            H, W, C, C1, C2,
            img1.ptr<float>(), img2.ptr<float>(),
            workspace.dm_dmu1.ptr<float>(),
            workspace.dm_dsigma1_sq.ptr<float>(),
            workspace.dm_dsigma12.ptr<float>(),
            workspace.ssim_map.ptr<float>());

        const auto stream = workspace.ssim_map.stream();
        launch_masked_fused_l1_ssim_mean_device(
            img1.ptr<float>(),
            img2.ptr<float>(),
            workspace.ssim_map.ptr<float>(),
            mask_2d.ptr<float>(),
            ssim_weight,
            workspace.reduction_temp.ptr<float>(),
            workspace.masked_loss.ptr<float>(),
            workspace.mask_sum.ptr<float>(),
            N, C, H, W,
            stream);

        auto loss_scalar = workspace.masked_loss.clone();
        const float mask_sum = workspace.mask_sum.item<float>();

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

    std::pair<lfs::core::Tensor, MaskedDecoupledFusedL1SSIMContext> masked_decoupled_fused_l1_ssim_forward(
        const lfs::core::Tensor& corrected_input,
        const lfs::core::Tensor& raw_input,
        const lfs::core::Tensor& gt_input,
        const lfs::core::Tensor& mask_input,
        float ssim_weight,
        MaskedDecoupledFusedL1SSIMWorkspace& workspace) {

        constexpr float C1 = 0.01f * 0.01f;
        constexpr float C2 = 0.03f * 0.03f;

        auto corrected = corrected_input.contiguous();
        auto raw = raw_input.contiguous();
        auto gt = gt_input.contiguous();
        auto mask = mask_input.contiguous();

        if (corrected.ndim() == 3)
            corrected = corrected.unsqueeze(0);
        if (raw.ndim() == 3)
            raw = raw.unsqueeze(0);
        if (gt.ndim() == 3)
            gt = gt.unsqueeze(0);

        auto mask_2d = mask.ndim() == 3 ? mask.squeeze(0) : mask;

        assert(corrected.shape() == raw.shape());
        assert(corrected.shape() == gt.shape());

        const int N = static_cast<int>(corrected.shape()[0]);
        const int C = static_cast<int>(corrected.shape()[1]);
        const int H = static_cast<int>(corrected.shape()[2]);
        const int W = static_cast<int>(corrected.shape()[3]);

        workspace.ensure_size(corrected.shape().dims());

        const dim3 grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, N);
        const dim3 block(BLOCK_X, BLOCK_Y);

        decoupledFusedL1SSIMForwardCUDA<<<grid, block>>>(
            H, W, C, C1, C2, ssim_weight,
            corrected.ptr<float>(), raw.ptr<float>(), gt.ptr<float>(),
            workspace.app_dm_dmu1.ptr<float>(),
            workspace.raw_dm_dmu1.ptr<float>(),
            workspace.raw_dm_dsigma1_sq.ptr<float>(),
            workspace.raw_dm_dsigma12.ptr<float>(),
            workspace.ssim_map.ptr<float>());

        const auto stream = workspace.ssim_map.stream();
        launch_masked_fused_l1_ssim_mean_device(
            corrected.ptr<float>(),
            gt.ptr<float>(),
            workspace.ssim_map.ptr<float>(),
            mask_2d.ptr<float>(),
            ssim_weight,
            workspace.reduction_temp.ptr<float>(),
            workspace.masked_loss.ptr<float>(),
            workspace.mask_sum.ptr<float>(),
            N, C, H, W,
            stream);

        auto loss_scalar = workspace.masked_loss.clone();
        const float mask_sum = workspace.mask_sum.item<float>();

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

        constexpr float C1 = 0.01f * 0.01f;
        constexpr float C2 = 0.03f * 0.03f;

        const size_t N = ctx.corrected_img.shape()[0];
        const size_t C = ctx.corrected_img.shape()[1];
        const float inv_mask_sum = 1.0f / ctx.mask_sum_value;

        const dim3 grid((ctx.W + BLOCK_X - 1) / BLOCK_X, (ctx.H + BLOCK_Y - 1) / BLOCK_Y, N);
        const dim3 block(BLOCK_X, BLOCK_Y);

        workspace.grad_corrected.zero_();
        maskedFusedL1SSIMBackwardCUDA<<<grid, block>>>(
            ctx.ssim_weight, inv_mask_sum, ctx.H, ctx.W, static_cast<int>(C), C1, C2,
            ctx.corrected_img.ptr<float>(), ctx.gt_img.ptr<float>(), ctx.mask.ptr<float>(),
            workspace.grad_corrected.ptr<float>(),
            ctx.app_dm_dmu1.ptr<float>(),
            workspace.zero_terms.ptr<float>(),
            workspace.zero_terms.ptr<float>());

        workspace.grad_raw.zero_();
        maskedFusedL1SSIMBackwardCUDA<<<grid, block>>>(
            1.0f, inv_mask_sum, ctx.H, ctx.W, static_cast<int>(C), C1, C2,
            ctx.raw_img.ptr<float>(), ctx.gt_img.ptr<float>(), ctx.mask.ptr<float>(),
            workspace.grad_raw.ptr<float>(),
            ctx.raw_dm_dmu1.ptr<float>(),
            ctx.raw_dm_dsigma1_sq.ptr<float>(),
            ctx.raw_dm_dsigma12.ptr<float>());

        return DecoupledGradients{
            .grad_corrected = workspace.grad_corrected,
            .grad_raw = workspace.grad_raw};
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

        assert(ssim_map.ndim() == 4);
        assert(ssim_map.shape()[0] == 1);

        const int C = static_cast<int>(ssim_map.shape()[1]);
        const int H = static_cast<int>(ssim_map.shape()[2]);
        const int W = static_cast<int>(ssim_map.shape()[3]);
        const int HW = H * W;

        assert(error_map.ndim() == 2);
        assert(error_map.shape()[0] == static_cast<size_t>(H));
        assert(error_map.shape()[1] == static_cast<size_t>(W));

        constexpr int THREADS = 256;
        dim3 grid((HW + THREADS - 1) / THREADS);
        dim3 block(THREADS);

        ssim_to_error_map_kernel<<<grid, block>>>(
            ssim_map.ptr<float>(),
            error_map.ptr<float>(),
            C, H, W);
    }

} // namespace lfs::training::kernels
