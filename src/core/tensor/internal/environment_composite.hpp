/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/environment_math.hpp"
#include "core/tensor/internal/private_access.hpp"
#include "core/tensor_environment.hpp"
namespace lfs::core::internal {
#if defined(__CUDACC__)
#define LFS_IMAGE_HD __host__ __device__ __forceinline__
#else
#define LFS_IMAGE_HD inline
#endif
    LFS_IMAGE_HD unsigned char floatToU8(float value) {
        return static_cast<unsigned char>(fminf(fmaxf(value, 0.0f), 1.0f) * 255.0f + 0.5f);
    }
    LFS_IMAGE_HD void composite_environment_pixel(const EnvironmentCompositeParams& p, const float* env_pixels,
                                                  const float* rgb_chw, const float* alpha, unsigned char* dst, int idx) {
        const int num_pixels = p.band_width * p.band_height;
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
#undef LFS_IMAGE_HD
} // namespace lfs::core::internal
