/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Environment-background sampling math shared by the CPU software composite
// (raster_rendering_engine.cpp) and the CUDA export composite
// (export_post_process.cu). Header-only, host/device compatible, glm-free so
// it compiles in device code.

#pragma once

#include <cmath>

#if defined(__CUDACC__)
#define LFS_ENV_HD __host__ __device__ __forceinline__
#else
#define LFS_ENV_HD inline
#endif

namespace lfs::rendering::envmath {

    inline constexpr float kPi = 3.14159265358979323846f;

    struct Vec3 {
        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;
    };

    LFS_ENV_HD float clampf(const float value, const float lo, const float hi) {
        return value < lo ? lo : (value > hi ? hi : value);
    }

    LFS_ENV_HD Vec3 normalized(const Vec3& v) {
        const float len = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
        if (len <= 0.0f) {
            return v;
        }
        const float inv = 1.0f / len;
        return {v.x * inv, v.y * inv, v.z * inv};
    }

    LFS_ENV_HD Vec3 mix(const Vec3& a, const Vec3& b, const float t) {
        return {a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t};
    }

    LFS_ENV_HD Vec3 rotateAroundY(const Vec3& value, const float radians) {
        const float c = std::cos(radians);
        const float s = std::sin(radians);
        return {
            c * value.x + s * value.z,
            value.y,
            -s * value.x + c * value.z,
        };
    }

    LFS_ENV_HD float acesTonemapChannel(const float v) {
        constexpr float a = 2.51f;
        constexpr float b = 0.03f;
        constexpr float c = 2.43f;
        constexpr float d = 0.59f;
        constexpr float e = 0.14f;
        return clampf((v * (a * v + b)) / (v * (c * v + d) + e), 0.0f, 1.0f);
    }

    LFS_ENV_HD Vec3 acesTonemap(const Vec3& value) {
        return {acesTonemapChannel(value.x), acesTonemapChannel(value.y), acesTonemapChannel(value.z)};
    }

    // exposure_factor = exp2(exposure EV); ACES + gamma 1/2.2 + clamp, matching the
    // CPU composite pipeline order.
    LFS_ENV_HD Vec3 shadeEnvironmentRadiance(const Vec3& hdr, const float exposure_factor) {
        Vec3 color = acesTonemap({hdr.x * exposure_factor, hdr.y * exposure_factor, hdr.z * exposure_factor});
        constexpr float kInvGamma = 1.0f / 2.2f;
        color = {std::pow(color.x, kInvGamma), std::pow(color.y, kInvGamma), std::pow(color.z, kInvGamma)};
        return {clampf(color.x, 0.0f, 1.0f), clampf(color.y, 0.0f, 1.0f), clampf(color.z, 0.0f, 1.0f)};
    }

    // View ray for a pixel in full-image coordinates. rotation_col_major is the
    // camera rotation laid out like glm::mat3 memory (&m[0][0]); focal/center are
    // full-frame pixel intrinsics resolved by the caller.
    LFS_ENV_HD Vec3 environmentWorldDirection(const float pixel_x, const float pixel_y,
                                              const float full_width, const float full_height,
                                              const bool equirectangular_view,
                                              const float focal_x, const float focal_y,
                                              const float center_x, const float center_y,
                                              const float* rotation_col_major) {
        const float width = full_width < 1.0f ? 1.0f : full_width;
        const float height = full_height < 1.0f ? 1.0f : full_height;
        const float tex_u = (pixel_x + 0.5f) / width;
        const float tex_v = 1.0f - (pixel_y + 0.5f) / height;

        Vec3 local_dir;
        if (equirectangular_view) {
            const float lon = (tex_u - 0.5f) * (2.0f * kPi);
            const float lat = (tex_v - 0.5f) * kPi;
            const float cos_lat = std::cos(lat);
            local_dir = normalized({std::sin(lon) * cos_lat,
                                    std::sin(lat),
                                    -std::cos(lon) * cos_lat});
        } else {
            const float px = tex_u * width;
            const float py = tex_v * height;
            local_dir = normalized({(px - center_x) / (focal_x > 1e-6f ? focal_x : 1e-6f),
                                    (py - center_y) / (focal_y > 1e-6f ? focal_y : 1e-6f),
                                    -1.0f});
        }

        const float* m = rotation_col_major;
        return normalized({m[0] * local_dir.x + m[3] * local_dir.y + m[6] * local_dir.z,
                           m[1] * local_dir.x + m[4] * local_dir.y + m[7] * local_dir.z,
                           m[2] * local_dir.x + m[5] * local_dir.y + m[8] * local_dir.z});
    }

    struct EquirectUv {
        float u = 0.0f;
        float v = 0.0f;
    };

    LFS_ENV_HD EquirectUv equirectUvForDirection(const Vec3& world_dir) {
        const float longitude = std::atan2(world_dir.x, -world_dir.z);
        const float latitude = std::asin(clampf(world_dir.y, -1.0f, 1.0f));
        return {longitude / (2.0f * kPi) + 0.5f, 0.5f - latitude / kPi};
    }

    // Bilinear tap positions matching the CPU sampler: u wraps, v clamps.
    struct BilinearTap {
        int x0 = 0;
        int y0 = 0;
        int x1 = 0;
        int y1 = 0;
        float tx = 0.0f;
        float ty = 0.0f;
    };

    LFS_ENV_HD BilinearTap environmentBilinearTap(float u, float v, const int width, const int height) {
        u = u - std::floor(u);
        v = clampf(v, 0.0f, 1.0f);

        const float x = u * static_cast<float>(width - 1);
        const float y = v * static_cast<float>(height - 1);
        BilinearTap tap;
        tap.x0 = static_cast<int>(std::floor(x));
        tap.x0 = tap.x0 < 0 ? 0 : (tap.x0 > width - 1 ? width - 1 : tap.x0);
        tap.y0 = static_cast<int>(std::floor(y));
        tap.y0 = tap.y0 < 0 ? 0 : (tap.y0 > height - 1 ? height - 1 : tap.y0);
        tap.x1 = (tap.x0 + 1) % width;
        tap.y1 = tap.y0 + 1 > height - 1 ? height - 1 : tap.y0 + 1;
        tap.tx = x - static_cast<float>(tap.x0);
        tap.ty = y - static_cast<float>(tap.y0);
        return tap;
    }

    // fetch(x, y) -> Vec3 texel of the [height, width, 3] environment image.
    template <typename FetchFn>
    LFS_ENV_HD Vec3 sampleEnvironmentBilinear(FetchFn&& fetch, const float u, const float v,
                                              const int width, const int height) {
        const BilinearTap tap = environmentBilinearTap(u, v, width, height);
        const Vec3 top = mix(fetch(tap.x0, tap.y0), fetch(tap.x1, tap.y0), tap.tx);
        const Vec3 bottom = mix(fetch(tap.x0, tap.y1), fetch(tap.x1, tap.y1), tap.tx);
        return mix(top, bottom, tap.ty);
    }

} // namespace lfs::rendering::envmath

#undef LFS_ENV_HD
