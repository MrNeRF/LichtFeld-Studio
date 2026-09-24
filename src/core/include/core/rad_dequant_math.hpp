/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// RAD scalar and rotation formulas shared by the file codec and page backends.

#pragma once

#if defined(__SLANG__)
#include "rounded_math.slangh"
#define RAD_ADD       roundedAdd
#define RAD_SUB       roundedSub
#define RAD_MUL       roundedMul
#define RAD_DIV(a, b) roundedMul(a, rotationReciprocal(b))
#define LFS_RAD_HD
#define RAD_CONST static const
#define RAD_U32   uint
#define RAD_I32   int
#define RAD_U16   uint
#define RAD_U8    uint
#define RAD_I8    int
#define RAD_OUT   out
#define RAD_ABS   abs
#define RAD_SQRT  sqrt
#define RAD_EXP   exp
#define RAD_LOG   log
#define RAD_SIN   sin
#define RAD_COS   cos
#define RAD_ACOS  acos
#else
#define RAD_ADD(a, b) ((a) + (b))
#define RAD_SUB(a, b) ((a) - (b))
#define RAD_MUL(a, b) ((a) * (b))
#define RAD_DIV(a, b) ((a) / (b))
#define RAD_CONST     inline constexpr
#define RAD_U32       std::uint32_t
#define RAD_I32       std::int32_t
#define RAD_U16       std::uint16_t
#define RAD_U8        std::uint8_t
#define RAD_I8        std::int8_t
#define RAD_OUT
#define RAD_ABS  std::fabs
#define RAD_SQRT std::sqrt
#define RAD_EXP  std::exp
#define RAD_LOG  std::log
#define RAD_SIN  std::sin
#define RAD_COS  std::cos
#define RAD_ACOS std::acos
#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__)
#define LFS_RAD_HD __host__ __device__ inline
#else
#define LFS_RAD_HD inline
#endif

#endif

namespace lfs::core::radmath {

    RAD_CONST float kShC0 = 0.28209479177387814f;
    RAD_CONST float kPi = 3.14159265358979323846f;

    LFS_RAD_HD float bitsToFloat(const RAD_U32 bits) {
#if defined(__SLANG__)
        return asfloat(bits);
#elif defined(__CUDA_ARCH__)
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
    LFS_RAD_HD float halfToFloat(const RAD_U16 value) {
        const RAD_U32 sign = (value >> 15) & 0x1u;
        RAD_U32 exponent = (value >> 10) & 0x1Fu;
        RAD_U32 mantissa = value & 0x3FFu;

        RAD_U32 f32;
        if (exponent == 0u) {
            if (mantissa == 0u) {
                f32 = sign << 31;
            } else {
                int shift = 0;
                while ((mantissa & 0x400u) == 0u) {
                    mantissa <<= 1;
                    ++shift;
                }
                exponent = RAD_U32(1 - shift);
                f32 = (sign << 31) | ((exponent + 127u - 15u) << 23) | ((mantissa & 0x3FFu) << 13);
            }
        } else if (exponent == 0x1Fu) {
            f32 = (sign << 31) | (0xFFu << 23) | (mantissa << 13);
        } else {
            f32 = (sign << 31) | ((exponent + 127u - 15u) << 23) | (mantissa << 13);
        }
        return bitsToFloat(f32);
    }

    LFS_RAD_HD RAD_U32 floatToBits(const float v) {
#if defined(__SLANG__)
        return asuint(v);
#elif defined(__CUDA_ARCH__)
        return __float_as_uint(v);
#else
        RAD_U32 bits;
        std::memcpy(&bits, &v, sizeof(bits));
        return bits;
#endif
    }

    // IEEE binary32 -> binary16, round-to-nearest-even with flush-to-zero on
    // underflow — the same convention as rad.cpp's file encoder. One bit
    // algorithm on host and device (NOT __float2half_rn, which emits
    // subnormals the CPU side flushes): the canonical pool stores f16 and
    // parity tests compare the bits.
    LFS_RAD_HD RAD_U16 floatToHalf(const float value) {
        const RAD_U32 f32 = floatToBits(value);
        const RAD_U32 sign = (f32 >> 31) & 0x1u;
        const RAD_U32 exponent = (f32 >> 23) & 0xFFu;
        const RAD_U32 mantissa = f32 & 0x7FFFFFu;
        if (exponent == 0u) {
            return RAD_U16(sign << 15);
        }
        if (exponent == 0xFFu) {
            return RAD_U16((sign << 15) | 0x7C00u | (mantissa >> 13));
        }
        const RAD_I32 new_exp = RAD_I32(exponent) - 127 + 15;
        if (new_exp >= 31) {
            return RAD_U16((sign << 15) | 0x7C00u);
        }
        if (new_exp <= 0) {
            return RAD_U16(sign << 15);
        }
        RAD_U32 new_mantissa = mantissa >> 13;
        if ((mantissa & 0x1FFFu) > 0x1000u ||
            ((mantissa & 0x1FFFu) == 0x1000u && (new_mantissa & 1u) != 0u)) {
            ++new_mantissa;
        }
        // ADD, not OR: a rounding carry (mantissa 0x400) must propagate into
        // the exponent or values crossing a power of two get halved.
        return RAD_U16((sign << 15) +
                       (RAD_U32(new_exp) << 10) +
                       new_mantissa);
    }

    // Explicit fused multiply-add: host GCC contracts a*b+c into fma while
    // the kernel (--fmad=false) would not, and the 1-ULP difference flips
    // values sitting on an f16 rounding boundary. Forcing the fused form on
    // both sides keeps r8-derived pool values bit-exact.
    LFS_RAD_HD float fmaExact(const float a, const float b, const float c) {
#if defined(__SLANG__)
        return fma(a, b, c);
#elif defined(__CUDA_ARCH__)
        return __fmaf_rn(a, b, c);
#else
        return std::fma(a, b, c);
#endif
    }

    // CUDA fast math multiplies by a rounded reciprocal for these fixed byte
    // denominators. The Slang path uses the same float operation.
    LFS_RAD_HD float byteFraction(const RAD_U32 value, const RAD_U32 denominator) {
#if defined(__SLANG__)
        const float reciprocal = denominator == 255u ? bitsToFloat(0x3b808081u) : bitsToFloat(0x3c010204u);
        return RAD_MUL(float(value), reciprocal);
#else
        return float(value) / float(denominator);
#endif
    }

    LFS_RAD_HD float dequantR8(const RAD_U8 v, const float min_val, const float range) {
        return fmaExact(byteFraction(v, 255u), range, min_val);
    }

    LFS_RAD_HD float dequantS8(const RAD_I8 v, const float max_abs) {
#if defined(__SLANG__)
        return (v < 0 ? -byteFraction(RAD_U32(-int(v)), 127u) : byteFraction(RAD_U32(v), 127u)) * max_abs;
#else
        return (float(v) / 127.0f) * max_abs;
#endif
    }

    LFS_RAD_HD float shMaxAbs(const float min_val, const float max_val, const float scale) {
        float m = RAD_ABS(min_val);
        const float a = RAD_ABS(max_val);
        const float s = RAD_ABS(scale);
        if (a > m) {
            m = a;
        }
        if (s > m) {
            m = s;
        }
        return m > 1e-6f ? m : 1e-6f;
    }

    // Value 0 is reserved for zero scales; 1-255 span [ln_min, ln_max].
    LFS_RAD_HD float dequantLn0R8(const RAD_U8 v, const float ln_min, const float ln_max) {
        if (v == 0u) {
            return 0.0f;
        }
        const float ln_scale = ln_min + float(v - 1) * (ln_max - ln_min) / 254.0f;
        return RAD_EXP(ln_scale);
    }

#if defined(__SLANG__)
    // Normalization reciprocals are positive normal floats, rounded to nearest even.
    float rotationReciprocal(const float value) {
        float reciprocal = roundedDiv(1.0f, value);
        reciprocal = fma(fma(-value, reciprocal, 1.0f), reciprocal, reciprocal);
        const float error = fma(-value, reciprocal, 1.0f);
        if (error == 0.0f)
            return reciprocal;
        const uint bits = asuint(reciprocal);
        const float adjacent = asfloat(error > 0.0f ? bits + 1u : bits - 1u);
        const float midpoint = roundedMul(value, roundedMul(abs(roundedSub(adjacent, reciprocal)), 0.5f));
        return abs(error) > midpoint || (abs(error) == midpoint && (bits & 1u) != 0u)
                   ? adjacent
                   : reciprocal;
    }
#endif

    // Octahedral axis (2 bytes) + angle (1 byte) -> normalized [x,y,z,w].
    LFS_RAD_HD void dequantQuatOct88R8(const RAD_U8 b0,
                                       const RAD_U8 b1,
                                       const RAD_U8 b2,
                                       RAD_OUT float out_xyzw[4]) {
        float oct_x = RAD_SUB(RAD_MUL(byteFraction(b0, 255u), 2.0f), 1.0f);
        float oct_y = RAD_SUB(RAD_MUL(byteFraction(b1, 255u), 2.0f), 1.0f);

        const float oct_z = RAD_SUB(RAD_SUB(1.0f, RAD_ABS(oct_x)), RAD_ABS(oct_y));
        if (oct_z < 0.0f) {
            const float temp_x = oct_x;
            oct_x = RAD_MUL(RAD_SUB(1.0f, RAD_ABS(oct_y)), (oct_x >= 0.0f ? 1.0f : -1.0f));
            oct_y = RAD_MUL(RAD_SUB(1.0f, RAD_ABS(temp_x)), (oct_y >= 0.0f ? 1.0f : -1.0f));
        }

        float axis_x = oct_x;
        float axis_y = oct_y;
        float axis_z = oct_z;
        const float len = RAD_SQRT(RAD_ADD(RAD_ADD(RAD_MUL(axis_x, axis_x), RAD_MUL(axis_y, axis_y)), RAD_MUL(axis_z, axis_z)));
        if (len > 0.0f) {
            axis_x = RAD_DIV(axis_x, len);
            axis_y = RAD_DIV(axis_y, len);
            axis_z = RAD_DIV(axis_z, len);
        }

        const float theta = RAD_MUL(byteFraction(b2, 255u), kPi);
        const float half_theta = RAD_MUL(theta, 0.5f);
        const float sin_half_theta = RAD_SIN(half_theta);
        const float cos_half_theta = RAD_COS(half_theta);

        float x = RAD_MUL(axis_x, sin_half_theta);
        float y = RAD_MUL(axis_y, sin_half_theta);
        float z = RAD_MUL(axis_z, sin_half_theta);
        float w = cos_half_theta;

        const float q_len = RAD_SQRT(RAD_ADD(RAD_ADD(RAD_ADD(RAD_MUL(x, x), RAD_MUL(y, y)), RAD_MUL(z, z)), RAD_MUL(w, w)));
        if (q_len > 0.0f) {
            x = RAD_DIV(x, q_len);
            y = RAD_DIV(y, q_len);
            z = RAD_DIV(z, q_len);
            w = RAD_DIV(w, q_len);
        }
#if defined(__SLANG__)
        if ((floatToHalf(x) & 0x7fffu) == 0u)
            x = bitsToFloat(b0 < 128u ? 0x80000000u : 0u);
        if ((floatToHalf(y) & 0x7fffu) == 0u)
            y = bitsToFloat(b1 < 128u ? 0x80000000u : 0u);
#endif
        out_xyzw[0] = x;
        out_xyzw[1] = y;
        out_xyzw[2] = z;
        out_xyzw[3] = w;
    }

    // Normalized [x,y,z,w] -> octahedral axis (2 bytes) + angle (1 byte).
    // Same math as the RAD file encoder, so scratch-quantized rotations land
    // on the grid the final encode would pick anyway.
    LFS_RAD_HD void quantQuatOct88R8(float x, float y, float z, float w,
                                     RAD_OUT RAD_U8 out[3]) {
        const float len = RAD_SQRT(x * x + y * y + z * z + w * w);
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

        const float theta = 2.0f * RAD_ACOS(clampf(w, -1.0f, 1.0f));
        const float sin_half_theta = RAD_SIN(theta * 0.5f);
        float axis_x = 1.0f;
        float axis_y = 0.0f;
        float axis_z = 0.0f;
        if (sin_half_theta > 1e-6f) {
            axis_x = x / sin_half_theta;
            axis_y = y / sin_half_theta;
            axis_z = z / sin_half_theta;
        }
        const float axis_len =
            RAD_SQRT(axis_x * axis_x + axis_y * axis_y + axis_z * axis_z);
        if (axis_len > 0.0f) {
            axis_x /= axis_len;
            axis_y /= axis_len;
            axis_z /= axis_len;
        }

        const float abs_sum = RAD_ABS(axis_x) + RAD_ABS(axis_y) + RAD_ABS(axis_z);
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
            oct_x = (1.0f - RAD_ABS(oct_y)) * (oct_x >= 0.0f ? 1.0f : -1.0f);
            oct_y = (1.0f - RAD_ABS(temp_x)) * (oct_y >= 0.0f ? 1.0f : -1.0f);
        }

        out[0] = RAD_U8(clampf((oct_x + 1.0f) * 0.5f * 255.0f, 0.0f, 255.0f));
        out[1] = RAD_U8(clampf((oct_y + 1.0f) * 0.5f * 255.0f, 0.0f, 255.0f));
        out[2] = RAD_U8(clampf(theta / kPi * 255.0f, 0.0f, 255.0f));
    }

    // f32/f16 orientation planes store xyz; w is reconstructed.
    LFS_RAD_HD float quatWFromXyz(const float x, const float y, const float z) {
        const float t = 1.0f - x * x - y * y - z * z;
        return RAD_SQRT(t > 0.0f ? t : 0.0f);
    }

    // Post-decode transforms applied by decode_rad_chunk_into, in order.
    LFS_RAD_HD float sh0Transform(const float v) {
        return (v - 0.5f) / kShC0;
    }

    LFS_RAD_HD float opacityLogit(const float v) {
        const float a = clampf(v, 1.0e-6f, 1.0f - 1.0e-6f);
        return RAD_LOG(a / (1.0f - a));
    }

    LFS_RAD_HD float opacityLodEncoded(const float v) {
        return v < 0.0f ? 0.0f : v;
    }

    LFS_RAD_HD float scaleLog(const float v) {
        return RAD_LOG(v < 1.0e-8f ? 1.0e-8f : v);
    }

} // namespace lfs::core::radmath

#undef LFS_RAD_HD
#undef RAD_ADD
#undef RAD_SUB
#undef RAD_MUL
#undef RAD_DIV

#undef RAD_CONST
#undef RAD_U32
#undef RAD_I32
#undef RAD_U16
#undef RAD_U8
#undef RAD_I8
#undef RAD_OUT
#undef RAD_ABS
#undef RAD_SQRT
#undef RAD_EXP
#undef RAD_LOG
#undef RAD_SIN
#undef RAD_COS
#undef RAD_ACOS
