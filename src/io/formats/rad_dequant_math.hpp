/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Per-element RAD dequantization formulas shared between the CPU chunk
// decoders (rad.cpp) and the CUDA page-dequant kernel. The CPU/GPU outputs
// must agree bit-exactly for pure-arithmetic paths (f16, r8, s8); paths
// through libm (exp/log/trig) agree within a few ULP and are covered by
// tolerance in the parity tests. The kernel translation unit is compiled
// with --fmad=false so a*b+c sequences round identically to the host.

#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__)
#define LFS_RAD_HD __host__ __device__ inline
#else
#define LFS_RAD_HD inline
#endif

namespace lfs::io::radmath {

    inline constexpr float kShC0 = 0.28209479177387814f;
    inline constexpr float kPi = 3.14159265358979323846f;

    LFS_RAD_HD float bitsToFloat(const std::uint32_t bits) {
#if defined(__CUDA_ARCH__)
        return __uint_as_float(bits);
#else
        float v;
        std::memcpy(&v, &bits, sizeof(v));
        return v;
#endif
    }

    LFS_RAD_HD float clampf(const float v, const float lo, const float hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    }

    // IEEE binary16 -> binary32. Exact for every f16 value (all are
    // representable), so any correct implementation is bit-identical; the
    // subnormal branch normalizes by shifting to the implicit-one position.
    LFS_RAD_HD float halfToFloat(const std::uint16_t value) {
        const std::uint32_t sign = (value >> 15) & 0x1u;
        std::uint32_t exponent = (value >> 10) & 0x1Fu;
        std::uint32_t mantissa = value & 0x3FFu;

        std::uint32_t f32;
        if (exponent == 0u) {
            if (mantissa == 0u) {
                f32 = sign << 31;
            } else {
                int shift = 0;
                while ((mantissa & 0x400u) == 0u) {
                    mantissa <<= 1;
                    ++shift;
                }
                exponent = static_cast<std::uint32_t>(1 - shift);
                f32 = (sign << 31) | ((exponent + 127u - 15u) << 23) | ((mantissa & 0x3FFu) << 13);
            }
        } else if (exponent == 0x1Fu) {
            f32 = (sign << 31) | (0xFFu << 23) | (mantissa << 13);
        } else {
            f32 = (sign << 31) | ((exponent + 127u - 15u) << 23) | (mantissa << 13);
        }
        return bitsToFloat(f32);
    }

    LFS_RAD_HD std::uint32_t floatToBits(const float v) {
#if defined(__CUDA_ARCH__)
        return __float_as_uint(v);
#else
        std::uint32_t bits;
        std::memcpy(&bits, &v, sizeof(bits));
        return bits;
#endif
    }

    // IEEE binary32 -> binary16, round-to-nearest-even with flush-to-zero on
    // underflow — the same convention as rad.cpp's file encoder. One bit
    // algorithm on host and device (NOT __float2half_rn, which emits
    // subnormals the CPU side flushes): the canonical pool stores f16 and
    // parity tests compare the bits.
    LFS_RAD_HD std::uint16_t floatToHalf(const float value) {
        const std::uint32_t f32 = floatToBits(value);
        const std::uint32_t sign = (f32 >> 31) & 0x1u;
        const std::uint32_t exponent = (f32 >> 23) & 0xFFu;
        const std::uint32_t mantissa = f32 & 0x7FFFFFu;
        if (exponent == 0u) {
            return static_cast<std::uint16_t>(sign << 15);
        }
        if (exponent == 0xFFu) {
            return static_cast<std::uint16_t>((sign << 15) | 0x7C00u | (mantissa >> 13));
        }
        const std::int32_t new_exp = static_cast<std::int32_t>(exponent) - 127 + 15;
        if (new_exp >= 31) {
            return static_cast<std::uint16_t>((sign << 15) | 0x7C00u);
        }
        if (new_exp <= 0) {
            return static_cast<std::uint16_t>(sign << 15);
        }
        std::uint32_t new_mantissa = mantissa >> 13;
        if ((mantissa & 0x1FFFu) > 0x1000u ||
            ((mantissa & 0x1FFFu) == 0x1000u && (new_mantissa & 1u))) {
            ++new_mantissa;
        }
        // ADD, not OR: a rounding carry (mantissa 0x400) must propagate into
        // the exponent or values crossing a power of two get halved.
        return static_cast<std::uint16_t>((sign << 15) +
                                          (static_cast<std::uint32_t>(new_exp) << 10) +
                                          new_mantissa);
    }

    // Explicit fused multiply-add: host GCC contracts a*b+c into fma while
    // the kernel (--fmad=false) would not, and the 1-ULP difference flips
    // values sitting on an f16 rounding boundary. Forcing the fused form on
    // both sides keeps r8-derived pool values bit-exact.
    LFS_RAD_HD float fmaExact(const float a, const float b, const float c) {
#if defined(__CUDA_ARCH__)
        return __fmaf_rn(a, b, c);
#else
        return std::fma(a, b, c);
#endif
    }

    LFS_RAD_HD float dequantR8(const std::uint8_t v, const float min_val, const float range) {
        return fmaExact(static_cast<float>(v) / 255.0f, range, min_val);
    }

    LFS_RAD_HD float dequantS8(const std::int8_t v, const float max_abs) {
        return (static_cast<float>(v) / 127.0f) * max_abs;
    }

    LFS_RAD_HD float shMaxAbs(const float min_val, const float max_val, const float scale) {
        float m = std::fabs(min_val);
        const float a = std::fabs(max_val);
        const float s = std::fabs(scale);
        if (a > m) {
            m = a;
        }
        if (s > m) {
            m = s;
        }
        return m > 1e-6f ? m : 1e-6f;
    }

    // Value 0 is reserved for zero scales; 1-255 span [ln_min, ln_max].
    LFS_RAD_HD float dequantLn0R8(const std::uint8_t v, const float ln_min, const float ln_max) {
        if (v == 0u) {
            return 0.0f;
        }
        const float ln_scale = ln_min + static_cast<float>(v - 1) * (ln_max - ln_min) / 254.0f;
        return std::exp(ln_scale);
    }

    // Octahedral axis (2 bytes) + angle (1 byte) -> normalized [x,y,z,w].
    LFS_RAD_HD void dequantQuatOct88R8(const std::uint8_t b0,
                                       const std::uint8_t b1,
                                       const std::uint8_t b2,
                                       float out_xyzw[4]) {
        float oct_x = (static_cast<float>(b0) / 255.0f) * 2.0f - 1.0f;
        float oct_y = (static_cast<float>(b1) / 255.0f) * 2.0f - 1.0f;

        const float oct_z = 1.0f - std::fabs(oct_x) - std::fabs(oct_y);
        if (oct_z < 0.0f) {
            const float temp_x = oct_x;
            oct_x = (1.0f - std::fabs(oct_y)) * (oct_x >= 0.0f ? 1.0f : -1.0f);
            oct_y = (1.0f - std::fabs(temp_x)) * (oct_y >= 0.0f ? 1.0f : -1.0f);
        }

        float axis_x = oct_x;
        float axis_y = oct_y;
        float axis_z = oct_z;
        const float len = std::sqrt(axis_x * axis_x + axis_y * axis_y + axis_z * axis_z);
        if (len > 0.0f) {
            axis_x /= len;
            axis_y /= len;
            axis_z /= len;
        }

        const float theta = (static_cast<float>(b2) / 255.0f) * kPi;
        const float half_theta = theta * 0.5f;
        const float sin_half_theta = std::sin(half_theta);
        const float cos_half_theta = std::cos(half_theta);

        float x = axis_x * sin_half_theta;
        float y = axis_y * sin_half_theta;
        float z = axis_z * sin_half_theta;
        float w = cos_half_theta;

        const float q_len = std::sqrt(x * x + y * y + z * z + w * w);
        if (q_len > 0.0f) {
            x /= q_len;
            y /= q_len;
            z /= q_len;
            w /= q_len;
        }
        out_xyzw[0] = x;
        out_xyzw[1] = y;
        out_xyzw[2] = z;
        out_xyzw[3] = w;
    }

    // Normalized [x,y,z,w] -> octahedral axis (2 bytes) + angle (1 byte).
    // Same math as the RAD file encoder, so scratch-quantized rotations land
    // on the grid the final encode would pick anyway.
    LFS_RAD_HD void quantQuatOct88R8(float x, float y, float z, float w,
                                     std::uint8_t out[3]) {
        const float len = std::sqrt(x * x + y * y + z * z + w * w);
        if (len > 0.0f) {
            x /= len;
            y /= len;
            z /= len;
            w /= len;
        }
        if (w < 0.0f) {
            x = -x;
            y = -y;
            z = -z;
            w = -w;
        }

        const float theta = 2.0f * std::acos(clampf(w, -1.0f, 1.0f));
        const float sin_half_theta = std::sin(theta * 0.5f);
        float axis_x = 1.0f;
        float axis_y = 0.0f;
        float axis_z = 0.0f;
        if (sin_half_theta > 1e-6f) {
            axis_x = x / sin_half_theta;
            axis_y = y / sin_half_theta;
            axis_z = z / sin_half_theta;
        }
        const float axis_len =
            std::sqrt(axis_x * axis_x + axis_y * axis_y + axis_z * axis_z);
        if (axis_len > 0.0f) {
            axis_x /= axis_len;
            axis_y /= axis_len;
            axis_z /= axis_len;
        }

        const float abs_sum = std::fabs(axis_x) + std::fabs(axis_y) + std::fabs(axis_z);
        float oct_x = axis_x;
        float oct_y = axis_y;
        float oct_z = axis_z;
        if (abs_sum > 0.0f) {
            const float inv_sum = 1.0f / abs_sum;
            oct_x *= inv_sum;
            oct_y *= inv_sum;
            oct_z *= inv_sum;
        }
        if (oct_z < 0.0f) {
            const float temp_x = oct_x;
            oct_x = (1.0f - std::fabs(oct_y)) * (oct_x >= 0.0f ? 1.0f : -1.0f);
            oct_y = (1.0f - std::fabs(temp_x)) * (oct_y >= 0.0f ? 1.0f : -1.0f);
        }

        out[0] = static_cast<std::uint8_t>(clampf((oct_x + 1.0f) * 0.5f * 255.0f, 0.0f, 255.0f));
        out[1] = static_cast<std::uint8_t>(clampf((oct_y + 1.0f) * 0.5f * 255.0f, 0.0f, 255.0f));
        out[2] = static_cast<std::uint8_t>(clampf(theta / kPi * 255.0f, 0.0f, 255.0f));
    }

    // f32/f16 orientation planes store xyz; w is reconstructed.
    LFS_RAD_HD float quatWFromXyz(const float x, const float y, const float z) {
        const float t = 1.0f - x * x - y * y - z * z;
        return std::sqrt(t > 0.0f ? t : 0.0f);
    }

    // Post-decode transforms applied by decode_rad_chunk_into, in order.
    LFS_RAD_HD float sh0Transform(const float v) {
        return (v - 0.5f) / kShC0;
    }

    LFS_RAD_HD float opacityLogit(const float v) {
        const float a = clampf(v, 1.0e-6f, 1.0f - 1.0e-6f);
        return std::log(a / (1.0f - a));
    }

    LFS_RAD_HD float opacityLodEncoded(const float v) {
        return v < 0.0f ? 0.0f : v;
    }

    LFS_RAD_HD float scaleLog(const float v) {
        return std::log(v < 1.0e-8f ? 1.0e-8f : v);
    }

} // namespace lfs::io::radmath

#undef LFS_RAD_HD
