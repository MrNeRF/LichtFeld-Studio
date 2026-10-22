/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
// Adapted from AMD-Ecosystem/LichtFeld-Studio PR #2,
// a0b02ebf3810d3a478d7b60f6ba2308013e93c29 (GPL-3.0-or-later).
// Inserted into the generated HIP photometric namespace after canonical SSIM
// constants/helpers. CUDA sources and FP16 partial storage remain unchanged.
    constexpr int HIP_SSIM_BLOCK_X = 32;
    constexpr int HIP_SSIM_BLOCK_Y = 8;
    constexpr int HIP_SSIM_SHARED_X = HIP_SSIM_BLOCK_X + 2 * HALO;
    constexpr int HIP_SSIM_SHARED_Y = HIP_SSIM_BLOCK_Y + 2 * HALO;

    // Give each block one channel plane so the image and derivative loads stay
    // contiguous across x. The CUDA kernel above processes every channel in a
    // block and stores interleaved derivative components in shared memory,
    // which is poorly matched to the LDS access pattern on AMD GPUs.
    template <typename TargetT, bool HasSigmaPartials, bool Accumulate>
    __global__ void fusedL1SSIMBackwardHIP(
        float ssim_weight,
        int H,
        int W,
        int CH,
        float grad_per_pixel,
        bool apply_valid_padding,
        const float* __restrict__ img1,
        const TargetT* __restrict__ img2,
        float* __restrict__ dL_dimg1,
        const __half* __restrict__ dm_dmu1,
        const __half* __restrict__ dm_dsigma1_sq,
        const __half* __restrict__ dm_dsigma12) {
        constexpr int COMPONENTS = HasSigmaPartials ? 3 : 1;
        __shared__ float derivatives[COMPONENTS][HIP_SSIM_SHARED_Y][HIP_SSIM_SHARED_X];
        __shared__ float horizontal[COMPONENTS][HIP_SSIM_SHARED_Y][HIP_SSIM_BLOCK_X];

        const int plane = static_cast<int>(blockIdx.z);
        const int batch = plane / CH;
        const int channel = plane % CH;
        const int start_x = static_cast<int>(blockIdx.x) * HIP_SSIM_BLOCK_X;
        const int start_y = static_cast<int>(blockIdx.y) * HIP_SSIM_BLOCK_Y;
        const int pix_x = start_x + threadIdx.x;
        const int pix_y = start_y + threadIdx.y;
        const bool crop_y = apply_valid_padding && H > 10;
        const bool crop_x = apply_valid_padding && W > 10;

        for (int row = threadIdx.y; row < HIP_SSIM_SHARED_Y; row += HIP_SSIM_BLOCK_Y) {
            const int gy = start_y + row - HALO;
            for (int col = threadIdx.x; col < HIP_SSIM_SHARED_X; col += HIP_SSIM_BLOCK_X) {
                const int gx = start_x + col - HALO;
                float chain = 0.0f;
                float vmu = 0.0f;
                float vs1 = 0.0f;
                float vs12 = 0.0f;
                if (gx >= 0 && gx < W && gy >= 0 && gy < H) {
                    if ((!crop_x || (gx >= HALO && gx < W - HALO)) && (!crop_y || (gy >= HALO && gy < H - HALO)))
                        chain = grad_per_pixel;
                    vmu = get_pix_value(dm_dmu1, batch, channel, gy, gx, CH, H, W);
                    if constexpr (HasSigmaPartials) {
                        vs1 = get_pix_value(dm_dsigma1_sq, batch, channel, gy, gx, CH, H, W);
                        vs12 = get_pix_value(dm_dsigma12, batch, channel, gy, gx, CH, H, W);
                    }
                }
                derivatives[0][row][col] = -ssim_weight * vmu * chain;
                if constexpr (HasSigmaPartials) {
                    derivatives[1][row][col] = -ssim_weight * vs1 * chain;
                    derivatives[2][row][col] = -ssim_weight * vs12 * chain;
                }
            }
        }
        __syncthreads();

        const int center_x = threadIdx.x + HALO;
        for (int row = threadIdx.y; row < HIP_SSIM_SHARED_Y; row += HIP_SSIM_BLOCK_Y) {
#pragma unroll
            for (int component = 0; component < COMPONENTS; ++component) {
                float sum = 0.0f;
#pragma unroll
                for (int d = 1; d <= HALO; ++d) {
                    const float weight = cGauss[HALO - d];
                    sum += (derivatives[component][row][center_x - d] +
                            derivatives[component][row][center_x + d]) * weight;
                }
                sum += derivatives[component][row][center_x] * cGauss[HALO];
                horizontal[component][row][threadIdx.x] = sum;
            }
        }
        __syncthreads();

        if (pix_x < W && pix_y < H) {
            const int center_y = threadIdx.y + HALO;
            float convolved[3]{};
#pragma unroll
            for (int component = 0; component < COMPONENTS; ++component) {
#pragma unroll
                for (int d = 1; d <= HALO; ++d) {
                    const float weight = cGauss[HALO - d];
                    convolved[component] +=
                        (horizontal[component][center_y - d][threadIdx.x] +
                         horizontal[component][center_y + d][threadIdx.x]) * weight;
                }
                convolved[component] += horizontal[component][center_y][threadIdx.x] * cGauss[HALO];
            }

            const int index = batch * CH * H * W + channel * H * W + pix_y * W + pix_x;
            const float p1 = pixel_value(img1, index);
            const float p2 = pixel_value(img2, index);
            const float grad_ssim = convolved[0] + (2.0f * p1) * convolved[1] + p2 * convolved[2];
            const bool inside_valid_region =
                (!crop_x || (pix_x >= HALO && pix_x < W - HALO)) && (!crop_y || (pix_y >= HALO && pix_y < H - HALO));
            const float chain = inside_valid_region ? grad_per_pixel : 0.0f;
            const float sign = p1 == p2 ? 0.0f : copysignf(1.0f, p1 - p2);
            const float grad_l1 = (1.0f - ssim_weight) * sign * chain;
            if constexpr (Accumulate) {
                dL_dimg1[index] += grad_ssim + grad_l1;
            } else {
                dL_dimg1[index] = grad_ssim + grad_l1;
            }
        }
    }

