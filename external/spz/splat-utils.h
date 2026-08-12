#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "load-spz.h"
#include "splat-types.h"

namespace spz {

// Scale factor for DC color components. To convert to RGB, we should multiply by 0.282, but it can
// be useful to represent base colors that are out of range if the higher spherical harmonics bands
// bring them back into range so we multiply by a smaller value.
constexpr float colorScale = 0.15f;
constexpr float sqrt1_2 = 0.707106781186547524401f;

constexpr int32_t dimForDegree(int32_t degree) {
  switch (degree) {
    case 0: return 0;
    case 1: return 3;
    case 2: return 8;
    case 3: return 15;
    case 4: return 24;
    default: return 0;
  }
}

constexpr int32_t degreeForDim(int32_t dim) {
  if (dim < 3)
    return 0;
  if (dim < 8)
    return 1;
  if (dim < 15)
    return 2;
  if (dim < 24)
    return 3;
  return 4;
}

constexpr size_t floatsPerPoint(SplatAttribute attr, int32_t shDegree) {
  switch (attr) {
    case SplatAttribute::Positions: return 3;
    case SplatAttribute::Alphas:    return 1;
    case SplatAttribute::Colors:    return 3;
    case SplatAttribute::Scales:    return 3;
    case SplatAttribute::Rotations: return 4;
    case SplatAttribute::Sh:        return static_cast<size_t>(dimForDegree(shDegree)) * 3;
  }
  return 0;
}

inline float unquantizeSH(uint8_t x) {
  return (static_cast<float>(x) - 128.0f) / 128.0f;
}

inline float invSigmoid(float x) {
  return std::log(x / (1.0f - x));
}

// Decode a quaternion stored as its first three components in 8-bit fixed point;
// w is reconstructed from the constraint that the quaternion is unit-length and
// non-negative. Within-family: `c.flipQ` scales xyz (fused into the decode).
// Cross-family: coordinateConverter sets `c.flipQ` to identity (so the fused
// multiply is a no-op) and `c.rotFlipQFunc` rotates the full xyzw afterwards.
inline void unpackQuaternionFirstThree(
    float rotation[4], const uint8_t r[3],
    const CoordinateConverter &c = CoordinateConverter()) {
  Vec3f xyz = times(
    plus(
      times(
        Vec3f{static_cast<float>(r[0]), static_cast<float>(r[1]), static_cast<float>(r[2])},
        1.0f / 127.5f),
      Vec3f{-1, -1, -1}),
    c.flipQ);
  std::copy(xyz.data(), xyz.data() + 3, &rotation[0]);
  rotation[3] = std::sqrt(std::max(0.0f, 1.0f - squaredNorm(xyz)));
  if (c.rotFlipQFunc) {
    c.rotFlipQFunc(rotation);
  }
}

// Decode a quaternion stored in the "smallest three" encoding: 2 bits identify
// the largest component (reconstructed from the unit-length constraint) and the
// other three are 10-bit signed magnitudes scaled by 1/sqrt(2).
inline void unpackQuaternionSmallestThree(
    float rotation[4], const uint8_t r[4],
    const CoordinateConverter &c = CoordinateConverter()) {
  uint32_t comp =
    r[0] + (r[1] << 8) + (r[2] << 16) + (r[3] << 24);

  constexpr uint32_t c_mask = (1u << 9u) - 1u;

  const int i_largest = comp >> 30;
  float sum_squares = 0;
  for (int i = 3; i >= 0; --i) {
    if (i != i_largest) {
      uint32_t mag    = comp & c_mask;
      uint32_t negbit = (comp >> 9u) & 0x1u;
      comp            = comp >> 10u;
      rotation[i]     = sqrt1_2 * static_cast<float>(mag) / static_cast<float>(c_mask);
      if (negbit == 1) {
        rotation[i] = -rotation[i];
      }
      sum_squares += rotation[i] * rotation[i];
    }
  }
  rotation[i_largest] = std::sqrt(1.0f - sum_squares);

  // Within-family flip via c.flipQ (identity in the cross-family case, where
  // c.rotFlipQFunc carries the full flip+rotation on the xyzw quaternion).
  for (int i = 0; i < 3; i++) {
    rotation[i] *= c.flipQ[i];
  }
  if (c.rotFlipQFunc) {
    c.rotFlipQFunc(rotation);
  }
}

// Access a PackedGaussians attribute buffer by enum tag. Lets callers iterate
// attributes (see kAllSplatAttributes) instead of hard-coding the six member
// names in every loop.
inline std::vector<uint8_t> &packedBuffer(PackedGaussians &p, SplatAttribute attr) {
  switch (attr) {
    case SplatAttribute::Positions: return p.positions;
    case SplatAttribute::Alphas:    return p.alphas;
    case SplatAttribute::Colors:    return p.colors;
    case SplatAttribute::Scales:    return p.scales;
    case SplatAttribute::Rotations: return p.rotations;
    case SplatAttribute::Sh:        return p.sh;
  }
  return p.positions;  // unreachable
}

inline constexpr size_t floatCount(const PackedGaussians &p, SplatAttribute attr) {
  return floatsPerPoint(attr, p.shDegree) * static_cast<size_t>(p.numPoints);
}

inline void releasePacked(PackedGaussians &p, SplatAttribute attr) {
  std::vector<uint8_t>().swap(packedBuffer(p, attr));
}

}  // namespace spz
