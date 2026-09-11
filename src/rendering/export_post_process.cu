/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "environment_math.hpp"
#include "export_post_process_kernels.cuh"

namespace lfs::rendering::exportpp {

    namespace {

        constexpr int kBlockSize = 256;

        [[nodiscard]] int divUp(const int a, const int b) { return (a + b - 1) / b; }

        __device__ __forceinline__ unsigned char floatToU8(const float value) {
            const float clamped = fminf(fmaxf(value, 0.0f), 1.0f);
            return static_cast<unsigned char>(clamped * 255.0f + 0.5f);
        }

        __global__ void unpack_u8_hwc_kernel(const unsigned char* __restrict__ src, const int num_pixels,
                                             const int src_channels, float* __restrict__ rgb_chw,
                                             float* __restrict__ alpha) {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= num_pixels)
                return;

            constexpr float kInv255 = 1.0f / 255.0f;
            const unsigned char* const px = src + static_cast<size_t>(idx) * static_cast<size_t>(src_channels);
            const size_t np = static_cast<size_t>(num_pixels);
            rgb_chw[idx] = static_cast<float>(px[0]) * kInv255;
            rgb_chw[np + idx] = static_cast<float>(px[1]) * kInv255;
            rgb_chw[2 * np + idx] = static_cast<float>(px[2]) * kInv255;
            if (alpha != nullptr) {
                alpha[idx] = static_cast<float>(px[3]) * kInv255;
            }
        }

        __global__ void pack_chw_u8_hwc_kernel(const float* __restrict__ rgb_chw, const float* __restrict__ alpha,
                                               unsigned char* __restrict__ dst, const int num_pixels,
                                               const int dst_channels) {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= num_pixels)
                return;

            const size_t np = static_cast<size_t>(num_pixels);
            unsigned char* const px = dst + static_cast<size_t>(idx) * static_cast<size_t>(dst_channels);
            px[0] = floatToU8(rgb_chw[idx]);
            px[1] = floatToU8(rgb_chw[np + idx]);
            px[2] = floatToU8(rgb_chw[2 * np + idx]);
            if (dst_channels == 4) {
                px[3] = floatToU8(alpha[idx]);
            }
        }

        __global__ void composite_environment_band_kernel(const CompositeParams p,
                                                          const float* __restrict__ env_pixels,
                                                          const float* __restrict__ rgb_chw,
                                                          const float* __restrict__ alpha,
                                                          unsigned char* __restrict__ dst) {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            const int num_pixels = p.band_width * p.band_height;
            if (idx >= num_pixels)
                return;

            const int x = idx % p.band_width;
            const int y = idx / p.band_width;

            envmath::Vec3 dir = envmath::environmentWorldDirection(
                static_cast<float>(x), static_cast<float>(p.y_offset + y),
                static_cast<float>(p.full_width), static_cast<float>(p.full_height),
                p.equirect_view, p.focal_x, p.focal_y, p.center_x, p.center_y, p.rotation);
            dir = envmath::normalized(envmath::rotateAroundY(dir, p.env_rotation_radians));
            const envmath::EquirectUv uv = envmath::equirectUvForDirection(dir);

            const auto fetch = [&](const int px, const int py) -> envmath::Vec3 {
                const size_t index =
                    (static_cast<size_t>(py) * static_cast<size_t>(p.env_width) + static_cast<size_t>(px)) * 3u;
                return {env_pixels[index], env_pixels[index + 1], env_pixels[index + 2]};
            };
            const envmath::Vec3 hdr = envmath::sampleEnvironmentBilinear(fetch, uv.u, uv.v, p.env_width, p.env_height);
            const envmath::Vec3 background = envmath::shadeEnvironmentRadiance(hdr, p.exposure_factor);

            const size_t np = static_cast<size_t>(num_pixels);
            const envmath::Vec3 rgb{rgb_chw[idx], rgb_chw[np + idx], rgb_chw[2 * np + idx]};
            const envmath::Vec3 out = envmath::mix(background, rgb, alpha[idx]);

            unsigned char* const px_out = dst + static_cast<size_t>(idx) * 3u;
            px_out[0] = floatToU8(out.x);
            px_out[1] = floatToU8(out.y);
            px_out[2] = floatToU8(out.z);
        }

    } // namespace

    cudaError_t launchUnpackU8Hwc(const unsigned char* const src, const int num_pixels, const int src_channels,
                                  float* const rgb_chw, float* const alpha, const cudaStream_t stream) {
        unpack_u8_hwc_kernel<<<divUp(num_pixels, kBlockSize), kBlockSize, 0, stream>>>(
            src, num_pixels, src_channels, rgb_chw, alpha);
        return cudaPeekAtLastError();
    }

    cudaError_t launchPackChwU8Hwc(const float* const rgb_chw, const float* const alpha, unsigned char* const dst,
                                   const int num_pixels, const int dst_channels, const cudaStream_t stream) {
        pack_chw_u8_hwc_kernel<<<divUp(num_pixels, kBlockSize), kBlockSize, 0, stream>>>(
            rgb_chw, alpha, dst, num_pixels, dst_channels);
        return cudaPeekAtLastError();
    }

    cudaError_t launchCompositeEnvironmentBand(const CompositeParams& params, const float* const env_pixels,
                                               const float* const rgb_chw, const float* const alpha,
                                               unsigned char* const dst, const cudaStream_t stream) {
        const int num_pixels = params.band_width * params.band_height;
        composite_environment_band_kernel<<<divUp(num_pixels, kBlockSize), kBlockSize, 0, stream>>>(
            params, env_pixels, rgb_chw, alpha, dst);
        return cudaPeekAtLastError();
    }

} // namespace lfs::rendering::exportpp
