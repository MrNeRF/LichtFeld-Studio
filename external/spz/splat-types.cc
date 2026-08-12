/*
MIT License

Copyright (c) 2024 Niantic Labs

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "splat-types.h"

#include <cmath>
#include <limits>

namespace spz {

float halfToFloat(Half h) {
  auto sgn = ((h >> 15) & 0x1);
  auto exponent = ((h >> 10) & 0x1f);
  auto mantissa = h & 0x3ff;

  float signMul = sgn == 1 ? -1.0 : 1.0;
  if (exponent == 0) {
    // Subnormal numbers (no exponent, 0 in the mantissa decimal).
    return signMul * std::pow(2.0f, -14.0f) * static_cast<float>(mantissa) / 1024.0f;
  }

  if (exponent == 31) {
    // Infinity or NaN.
    return mantissa != 0 ? std::numeric_limits<float>::quiet_NaN() : signMul * std::numeric_limits<float>::infinity();
  }

  // non-zero exponent implies 1 in the mantissa decimal.
  return signMul * std::pow(2.0f, static_cast<float>(exponent) - 15.0f)
    * (1.0f + static_cast<float>(mantissa) / 1024.0f);
}

Half floatToHalf(float f) {
  uint32_t f32 = *reinterpret_cast<uint32_t *>(&f);
  int32_t sign = (f32 >> 31) & 0x01;        // 1 bit   -> 1 bit
  int32_t exponent = ((f32 >> 23) & 0xff);  // 8 bits  -> 5 bits
  int32_t mantissa = f32 & 0x7fffff;        // 23 bits -> 10 bits

  // Handle inf and nan from float.
  if (exponent == 0xFF) {
    if (mantissa == 0) {
      return (sign << 15) | 0x7C00; // Inf
    }

    return (sign << 15) | 0x7C01;  // Nan
  }

  // If the exponent is greater than the range of half, return +/- Inf.
  int32_t centeredExp = exponent - 127;
  if (centeredExp > 15) {
    return (sign << 15) | 0x7C00;
  }

  // Normal numbers. centeredExp = [-15, 15]
  if (centeredExp > -15) {
    return (sign << 15) | ((centeredExp + 15) << 10) | (mantissa >> 13);
  }

  // Subnormal numbers.
  int32_t fullMantissa = 0x800000 | mantissa;
  int32_t shift = -(centeredExp + 14);  // Shift is in [-1 to -113]
  int32_t newMantissa = fullMantissa >> shift;
  return (sign << 15) | (newMantissa >> 13);
}

float norm(const Vec3f &a) {
  return std::sqrt(squaredNorm(a));
}

Vec3f normalized(const Vec3f &v) {
  float n = norm(v);
  return {v[0] / n, v[1] / n, v[2] / n};
}

float norm(const Quat4f &q) {
  return std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
}

Quat4f axisAngleQuat(const Vec3f &scaledAxis) {
  const float &a0 = scaledAxis[0];
  const float &a1 = scaledAxis[1];
  const float &a2 = scaledAxis[2];
  const float thetaSquared = a0 * a0 + a1 * a1 + a2 * a2;
  // For points not at the origin, the full conversion is numerically stable.
  if (thetaSquared > 0.0f) {
    const float theta = std::sqrt(thetaSquared);
    const float halfTheta = theta * 0.5f;
    const float k = std::sin(halfTheta) / theta;
    return normalized(std::array<float, 4>{std::cos(halfTheta), a0 * k, a1 * k, a2 * k});
  }
  // If thetaSquared is 0, then we will get NaNs when dividing by theta.  By approximating with a
  // Taylor series, and truncating at one term, the value will be computed correctly.
  const float k = 0.5f;
  return normalized(Quat4f{1.0f, a0 * k, a1 * k, a2 * k});
}

}  // namespace spz
