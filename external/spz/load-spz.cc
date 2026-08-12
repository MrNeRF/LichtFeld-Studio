/*
MIT License

Copyright (c) 2025 Niantic Labs
Copyright (c) 2025 Adobe Inc.

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

#include "load-spz.h"
#include "splat-utils.h"
#include "splat-types.h"
#ifdef SPZ_BUILD_EXTENSIONS
#include "splat-extensions.h"
#endif

#include <zlib.h>
#include <zstd.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace spz {

namespace {

uint8_t toUint8(float x) { return static_cast<uint8_t>(std::clamp(std::round(x), 0.0f, 255.0f)); }

// Quantizes to 8 bits, then rounds to nearest bucket center. 0 always maps to a bucket center.
uint8_t quantizeSH(float x, int32_t bucketSize) {
  int32_t q = static_cast<int>(std::round(x * 128.0f) + 128.0f);
  q = (q + bucketSize / 2) / bucketSize * bucketSize;
  return static_cast<uint8_t>(std::clamp(q, 0, 255));
}

float sigmoid(float x) { return 1 / (1 + std::exp(-x)); }

template <typename T>
size_t countBytes(const std::vector<T> &vec) {
  return vec.size() * sizeof(vec[0]);
}

bool checkedMultiply(size_t lhs, size_t rhs, size_t *result) {
  if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
    return false;
  }
  *result = lhs * rhs;
  return true;
}

bool checkedAdd(size_t lhs, size_t rhs, size_t *result) {
  if (rhs > std::numeric_limits<size_t>::max() - lhs) {
    return false;
  }
  *result = lhs + rhs;
  return true;
}

#define CHECK(x)                                                              \
  {                                                                           \
    if (!(x)) {                                                               \
      SpzLog("[SPZ: ERROR] Check failed: %s:%d: %s", __FILE__, __LINE__, #x); \
      return false;                                                           \
    }                                                                         \
  }

#define CHECK_GE(x, y) CHECK((x) >= (y))
#define CHECK_LE(x, y) CHECK((x) <= (y))
#define CHECK_EQ(x, y) CHECK((x) == (y))

bool checkSizes(const GaussianCloud &g) {
  CHECK_GE(g.numPoints, 0);
  CHECK_LE(static_cast<uint32_t>(g.numPoints), kMaxSpzPoints);
  CHECK_GE(g.shDegree, 0);
  CHECK_LE(g.shDegree, SH_MAX_DEGREE);
  const size_t count = static_cast<size_t>(g.numPoints);
  CHECK_EQ(g.positions.size(), count * 3);
  CHECK_EQ(g.scales.size(), count * 3);
  CHECK_EQ(g.rotations.size(), count * 4);
  CHECK_EQ(g.alphas.size(), count);
  CHECK_EQ(g.colors.size(), count * 3);
  CHECK_EQ(g.sh.size(), count * static_cast<size_t>(dimForDegree(g.shDegree)) * 3);
  return true;
}

bool checkSizes(const PackedGaussians &packed, int32_t numPoints, int32_t shDim, bool usesFloat16) {
  CHECK_GE(numPoints, 0);
  CHECK_LE(static_cast<uint32_t>(numPoints), kMaxSpzPoints);
  CHECK_GE(shDim, 0);
  CHECK_GE(packed.fractionalBits, 0);
  CHECK_LE(packed.fractionalBits, kMaxSpzFractionalBits);
  const size_t count = static_cast<size_t>(numPoints);
  CHECK_EQ(packed.positions.size(), count * 3 * (usesFloat16 ? 2 : 3));
  CHECK_EQ(packed.scales.size(), count * 3);
  CHECK_EQ(packed.rotations.size(), count * (packed.usesQuaternionSmallestThree ? 4 : 3));
  CHECK_EQ(packed.alphas.size(), count);
  CHECK_EQ(packed.colors.size(), count * 3);
  CHECK_EQ(packed.sh.size(), count * static_cast<size_t>(shDim) * 3);
  return true;
}

constexpr uint8_t FlagAntialiased = 0x1;
constexpr uint8_t FlagHasExtensions = 0x2;

// Generous upper bound on the zstd compression ratio (uncompressed / compressed) used to
// sanity-check numPoints against the actual file size. Real packed-gaussian streams compress
// well below this; the bound only exists to reject headers that claim vastly more points than
// any legitimate file of the given size could contain. Effective ceiling is
//   maxPoints = (fileSize * kMaxCompressionRatio) / kMinBytesPerPoint
// Examples (kMaxCompressionRatio=1024, kMinBytesPerPoint=9):
//   file size      max numPoints accepted
//      1 KB        ~116 K
//      1 MB        ~119 M
//    100 MB        ~11.9 B   (far above any realistic scene)
//      1 GB        ~119 B    (effectively unlimited)
// A 32-byte file is capped at ~3.6 K points, so a header claiming billions of points in a
// tiny file is rejected immediately.
// 64-bit so the `size * kMaxCompressionRatio` product below can't overflow on 32-bit targets
// (e.g. wasm32, where size_t is 32-bit and the multiply wraps for files >= 4 MiB).
constexpr uint64_t kMaxCompressionRatio = 1024;
constexpr uint64_t kMinBytesPerPoint = 9;  // positions stream alone: 3 components * 3 bytes

struct NgspFileHeader {
  uint32_t magic          = NGSP_MAGIC;
  uint32_t version        = LATEST_SPZ_HEADER_VERSION;
  uint32_t numPoints      = 0;
  uint8_t  shDegree       = 0;
  uint8_t  fractionalBits = 0;
  uint8_t  flags          = 0;
  uint8_t  numStreams     = 0;       // number of ZSTD-compressed attribute streams
  uint32_t tocByteOffset  = 0;       // byte offset from file start to the TOC (table of contents)
  uint8_t  reserved[12]   = {};
};
static_assert(sizeof(NgspFileHeader) == 32, "NgspFileHeader must be 32 bytes");

// TODO: After v4 is released, move legacy logic to separate files.
// Legacy 16-byte header used in gzip single-stream files (pre-v4). Read-only path.
struct LegacyPackedGaussiansHeader {
  uint32_t magic = NGSP_MAGIC;
  uint32_t version = 0;
  uint32_t numPoints = 0;
  uint8_t shDegree = 0;
  uint8_t fractionalBits = 0;
  uint8_t flags = 0;
  uint8_t reserved = 0;
};

void appendInflated(std::vector<uint8_t> *output, const uint8_t *data, const size_t size) {
  output->insert(output->end(), data, data + size);
}

bool decompressGzippedImpl(
  const uint8_t *compressed, size_t size, int32_t windowSize, std::vector<uint8_t> *out) {
  if (!compressed || !out || size == 0 || size > kMaxSpzCompressedBytes ||
      size > std::numeric_limits<uInt>::max()) {
    return false;
  }
  std::vector<uint8_t> buffer(8192);
  z_stream stream = {};
  stream.next_in = const_cast<Bytef *>(compressed);
  stream.avail_in = static_cast<uInt>(size);
  if (inflateInit2(&stream, windowSize) != Z_OK) {
    return false;
  }
  out->clear();
  bool success = false;
  try {
    while (true) {
      stream.next_out = buffer.data();
      stream.avail_out = static_cast<uInt>(buffer.size());
      const int32_t res = inflate(&stream, Z_NO_FLUSH);
      if (res != Z_OK && res != Z_STREAM_END) {
        break;
      }
      const size_t produced = buffer.size() - stream.avail_out;
      if (out->size() > kMaxSpzDecompressedBytes - produced) {
        break;
      }
      appendInflated(out, buffer.data(), produced);
      if (res == Z_STREAM_END) {
        success = stream.avail_in == 0;
        break;
      }
    }
  } catch (...) {
    inflateEnd(&stream);
    out->clear();
    throw;
  }
  inflateEnd(&stream);
  if (!success) {
    out->clear();
  }
  return success;
}

bool decompressGzipped(const uint8_t *compressed, size_t size, std::vector<uint8_t> *out) {
  // Here 16 means enable automatic gzip header detection; consider switching this to 32 to enable
  // both automated gzip and zlib header detection.
  return decompressGzippedImpl(compressed, size, 16 | MAX_WBITS, out);
}

// A read-only streambuf over a contiguous byte range, avoiding any copy.
struct membuf : std::streambuf {
  membuf(const uint8_t *data, size_t size) {
    auto *p = reinterpret_cast<char *>(const_cast<uint8_t *>(data));
    setg(p, p, p + size);
  }
};

}  // namespace

bool compressGzipped(const uint8_t *data, size_t size, std::vector<uint8_t> *out) {
  if (!data || !out || size == 0 || size > kMaxSpzDecompressedBytes ||
      size > std::numeric_limits<uInt>::max()) {
    return false;
  }
  std::vector<uint8_t> buffer(8192);
  z_stream stream = {};
  if (deflateInit2(&stream, Z_DEFAULT_COMPRESSION, Z_DEFLATED, 16 + MAX_WBITS, 9, Z_DEFAULT_STRATEGY)
      != Z_OK) {
    return false;
  }
  bool success = false;
  try {
    out->clear();
    out->reserve(size / 4);
    stream.next_in = const_cast<Bytef *>(reinterpret_cast<const Bytef *>(data));
    stream.avail_in = static_cast<uInt>(size);
    while (true) {
      stream.next_out = buffer.data();
      stream.avail_out = static_cast<uInt>(buffer.size());
      const int32_t res = deflate(&stream, Z_FINISH);
      if (res != Z_OK && res != Z_STREAM_END) {
        break;
      }
      const size_t produced = buffer.size() - stream.avail_out;
      if (out->size() > kMaxSpzCompressedBytes - produced) {
        break;
      }
      out->insert(out->end(), buffer.data(), buffer.data() + produced);
      if (res == Z_STREAM_END) {
        success = true;
        break;
      }
    }
  } catch (...) {
    deflateEnd(&stream);
    out->clear();
    throw;
  }
  deflateEnd(&stream);
  if (!success) {
    out->clear();
  }
  return success;
}

bool compressZstd(const uint8_t *data, size_t size, std::vector<uint8_t> *out,
                  int compressionLevel = 12) {
  size_t const bound = ZSTD_compressBound(size);
  out->resize(bound);

  ZSTD_CCtx *cctx = ZSTD_createCCtx();
  if (!cctx) return false;
  ZSTD_CCtx_setParameter(cctx, ZSTD_c_compressionLevel, compressionLevel);

  size_t const compressedSize = ZSTD_compress2(cctx, out->data(), bound, data, size);
  ZSTD_freeCCtx(cctx);

  if (ZSTD_isError(compressedSize)) return false;
  out->resize(compressedSize);
  return true;
}

// Backward compatibility function for version 2. In version 2, rotations are represented as the
// (x, y, z) components of the normalized rotation quaternion, with each component encoded as an
// 8-bit signed integer. Version 3+ uses packQuaternionSmallestThree for better accuracy.
void packQuaternionFirstThree(uint8_t r[3], const float rotation[4], const CoordinateConverter& c) {
    // Normalize the quaternion, make w positive, then store xyz. w can be derived from xyz.
    // NOTE: These are already in xyzw order.
    Quat4f q = normalized(quat4f(rotation));
    if (c.rotFlipQFunc) {
      c.rotFlipQFunc(q.data());
    } else {
      q[0] *= c.flipQ[0];
      q[1] *= c.flipQ[1];
      q[2] *= c.flipQ[2];
    }
    q = times(q, (q[3] < 0 ? -127.5f : 127.5f));
    q = plus(q, Quat4f{127.5f, 127.5f, 127.5f, 127.5f});
    r[0] = toUint8(q[0]);
    r[1] = toUint8(q[1]);
    r[2] = toUint8(q[2]);
}

void packQuaternionSmallestThree(uint8_t r[4], const float rotation[4], const CoordinateConverter& c) {
  // Normalize the quaternion
  Quat4f q = normalized(quat4f(&rotation[0]));
  if (c.rotFlipQFunc) { c.rotFlipQFunc(q.data()); }
  else { q[0] *= c.flipQ[0]; q[1] *= c.flipQ[1]; q[2] *= c.flipQ[2]; }
  // Find largest component
  unsigned iLargest = 0;
  for (unsigned i = 1; i < 4; ++i)
  {
    if (std::abs(q[i]) > std::abs(q[iLargest]))
    {
      iLargest = i;
    }
  }

  // since -q represents the same rotation as q, transform the quaternion so the largest element
  // is positive. This avoids having to send its sign bit.
  unsigned negate = q[iLargest] < 0;

  // Do compression using sign bit and 9-bit precision per element.
  uint32_t comp = iLargest;
  for (unsigned i = 0; i < 4; ++i)
  {
      if (i != iLargest)
      {
          uint32_t negbit = (q[i] < 0) ^ negate;
          uint32_t mag =
              (uint32_t)(float((1u << 9u) - 1u) * (std::fabs(q[i]) / sqrt1_2) + 0.5f);
          comp = (comp << 10u) | (negbit << 9u) | mag;
      }
  }

  // Ensure little-endianness on all platforms
  r[0] = comp & 0xff;
  r[1] = (comp >> 8) & 0xff;
  r[2] = (comp >> 16) & 0xff;
  r[3] = (comp >> 24) & 0xff;
}

PackedGaussians packGaussians(const GaussianCloud &g, const PackOptions &o) {
  if (!checkSizes(g)) {
    return {};
  }

  // Validate SH quantization bit parameters
  if (o.sh1Bits > 8 || o.shRestBits > 8 || o.sh1Bits < 1 || o.shRestBits < 1) {
    SpzLog("[SPZ ERROR] SH quantization bits cannot exceed 8 or be less than 1 (sh1Bits=%d, shRestBits=%d)",
           o.sh1Bits, o.shRestBits);
    return {};
  }

  const int32_t numPoints = g.numPoints;
  const int32_t shDim = dimForDegree(g.shDegree);
  if (o.version < 3 && g.shDegree > 3) {
    SpzLog("[SPZ WARNING] SPZ with SH degrees %d will not be loadable in a legacy loader of version %d",
        g.shDegree, o.version);
  }
#ifdef SPZ_BUILD_EXTENSIONS
  // If the cloud carries a coordinate-system extension, use its value as the storage target
  // instead of RUB so that callers can persist data in any coordinate system they choose.
  CoordinateSystem packToCoord = getPackedCoordinateSystem(g.extensions);
  CoordinateConverter c = coordinateConverter(o.from, packToCoord, g.shDegree);
#else
  CoordinateConverter c = coordinateConverter(o.from, CoordinateSystem::RUB, g.shDegree);
#endif

  // Use 12 bits for the fractional part of coordinates (~0.25 millimeter resolution). In the future
  // we can use different values on a per-splat basis and still be compatible with the decoder.
  PackedGaussians packed;
  packed.version = o.version;
  packed.numPoints = g.numPoints;
  packed.shDegree = g.shDegree;
  packed.fractionalBits = 12;
  packed.antialiased = g.antialiased;
  // Turn off quaternion-smallest-three for backward compatibility, since version 2 does not
  // support it.
  packed.usesQuaternionSmallestThree = o.version >= MIN_SMALLEST_THREE_QUATERNIONS_VERSION;

  packed.rotations.resize(numPoints * (packed.usesQuaternionSmallestThree ? 4 : 3));
  packed.positions.resize(numPoints * 3 * 3);
  packed.scales.resize(numPoints * 3);
  packed.alphas.resize(numPoints);
  packed.colors.resize(numPoints * 3);
  packed.sh.resize(numPoints * shDim * 3);

#ifdef SPZ_BUILD_EXTENSIONS
  packed.extensions = g.extensions;
#endif

  // Store coordinates as 24-bit fixed point values.
  const float scale = (1 << packed.fractionalBits);
  std::array<float, 3> bufPos = {};
  for (int32_t pi = 0; pi < numPoints; ++pi) {
    const size_t base = static_cast<size_t>(pi) * 3;
    bufPos[0] = g.positions[base + 0] * scale;
    bufPos[1] = g.positions[base + 1] * scale;
    bufPos[2] = g.positions[base + 2] * scale;
    if (c.rotFlipPFunc) { c.rotFlipPFunc(bufPos.data()); }
    else { bufPos[0] *= c.flipP[0]; bufPos[1] *= c.flipP[1]; bufPos[2] *= c.flipP[2]; }
    for (size_t j = 0; j < 3; ++j) {
      const int32_t fixed32 =
          static_cast<int32_t>(std::round(bufPos[j]));
      packed.positions[(base + j) * 3 + 0] = fixed32 & 0xff;
      packed.positions[(base + j) * 3 + 1] = (fixed32 >> 8) & 0xff;
      packed.positions[(base + j) * 3 + 2] = (fixed32 >> 16) & 0xff;
    }
  }

  for (size_t i = 0; i < numPoints * 3; i++) {
    packed.scales[i] = toUint8((g.scales[i] + 10.0f) * 16.0f);
  }

  if (packed.usesQuaternionSmallestThree) {
    for (size_t i = 0; i < numPoints; i++)
    {
      packQuaternionSmallestThree(&packed.rotations[4 * i], &g.rotations[4 * i], c);
    }
  } else {
    for (size_t i = 0; i < numPoints; i++)
    {
      packQuaternionFirstThree(&packed.rotations[3 * i], &g.rotations[4 * i], c);
    }
  }

  for (size_t i = 0; i < numPoints; i++) {
    // Apply sigmoid activation to alpha
    packed.alphas[i] = toUint8(sigmoid(g.alphas[i]) * 255.0f);
  }

  for (size_t i = 0; i < numPoints * 3; i++) {
    // Convert SH DC component to wide RGB (allowing values that are a bit above 1 and below 0).
    packed.colors[i] = toUint8(g.colors[i] * (colorScale * 255.0f) + (0.5f * 255.0f));
  }

  if (g.shDegree > 0) {
    // Use configurable spherical harmonics quantization parameters from PackOptions.
    // Quantization reduces information entropy for better g-zipping compression.
    // Note: Unpacking doesn't need these bits since g-unzipping fills zero bits automatically.
    const uint8_t sh1Bits = o.sh1Bits;
    const uint8_t shRestBits = o.shRestBits;
    const int32_t shPerPoint = dimForDegree(g.shDegree) * 3;
    std::array<float, 24> bufSh = {};
    for (size_t i = 0; i < numPoints * shPerPoint; i += shPerPoint) {
      for (size_t channel = 0; channel < 3; channel++) {
        for (size_t k = 0; k < static_cast<size_t>(shDim); ++k) {
          bufSh[k] = g.sh[i + k * 3 + channel];
        }
        for (size_t band = 0; band < static_cast<size_t>(g.shDegree) && band < static_cast<size_t>(SH_MAX_DEGREE); ++band) {
          if (c.rotFlipShFuncs[band]) { c.rotFlipShFuncs[band](bufSh.data() + band * (band + 2)); }
        }
        for (size_t k = 0; k < static_cast<size_t>(shDim); ++k) { bufSh[k] *= c.flipSh[k]; }
        size_t j = 0, k = 0;
        for (; j < 9; j += 3, k++) {  // degree-1: 3 coefficients × 3 RGB channels = 9 slots
          packed.sh[i + j + channel] = quantizeSH(bufSh[k], 1 << (8 - sh1Bits));
        }
        for (; j < shPerPoint; j += 3, k++) {
          packed.sh[i + j + channel] = quantizeSH(bufSh[k], 1 << (8 - shRestBits));
        }
      }
    }
  }

  return packed;
}

UnpackedGaussian PackedGaussian::unpack(
  bool usesFloat16, bool usesQuaternionSmallestThree, int32_t fractionalBits, const CoordinateConverter &c) const {
  UnpackedGaussian result{};
  if (fractionalBits < 0 || fractionalBits > kMaxSpzFractionalBits) {
    return result;
  }
  if (usesFloat16) {
    // Decode legacy float16 format. We can remove this at some point as it was never released.
    const auto *halfData = reinterpret_cast<const Half *>(position.data());
    for (size_t i = 0; i < 3; i++) {
      result.position[i] = halfToFloat(halfData[i]);
    }
  } else {
    // Decode 24-bit fixed point coordinates
    float scale = 1.0 / (1 << fractionalBits);
    for (size_t i = 0; i < 3; i++) {
      int32_t fixed32 = position[i * 3 + 0];
      fixed32 |= position[i * 3 + 1] << 8;
      fixed32 |= position[i * 3 + 2] << 16;
      fixed32 |= (fixed32 & 0x800000) ? 0xff000000 : 0;  // sign extension
      result.position[i] = static_cast<float>(fixed32) * scale;
    }
  }
  if (c.rotFlipPFunc) { c.rotFlipPFunc(result.position.data()); }
  else { for (size_t i = 0; i < 3; i++) result.position[i] *= c.flipP[i]; }

  for (size_t i = 0; i < 3; i++) {
    result.scale[i] = (scale[i] / 16.0f - 10.0f);
  }

  if (usesQuaternionSmallestThree)
  {
      unpackQuaternionSmallestThree(&result.rotation[0], &rotation[0], c);
  }
  else
  {
      unpackQuaternionFirstThree(&result.rotation[0], &rotation[0], c);
  }

  const float normalized_alpha = std::clamp(alpha / 255.0f, 1.0f / 255.0f, 254.0f / 255.0f);
  result.alpha = invSigmoid(normalized_alpha);

  for (size_t i = 0; i < 3; i++) {
    result.color[i] = ((color[i] / 255.0f) - 0.5f) / colorScale;
  }

  for (size_t i = 0; i < SH_MAX_COEFFS; i++) {
    result.shR[i] = unquantizeSH(shR[i]);
    result.shG[i] = unquantizeSH(shG[i]);
    result.shB[i] = unquantizeSH(shB[i]);
  }

  if (c.rotFlipShFuncs[0]) {
    for (size_t i = 0; i < SH_MAX_DEGREE; i++) {
      if (!c.rotFlipShFuncs[i]) { break; }
      const size_t baseIndex = i * (i + 2);
      c.rotFlipShFuncs[i](result.shR.data() + baseIndex);
      c.rotFlipShFuncs[i](result.shG.data() + baseIndex);
      c.rotFlipShFuncs[i](result.shB.data() + baseIndex);
    }
  } else {
    for (size_t i = 0; i < SH_MAX_COEFFS; i++) {
      result.shR[i] *= c.flipSh[i];
      result.shG[i] *= c.flipSh[i];
      result.shB[i] *= c.flipSh[i];
    }
  }

  return result;
}

PackedGaussian PackedGaussians::at(int32_t i) const {
  PackedGaussian result;
  int32_t positionBytes = usesFloat16() ? 6 : 9;
  int32_t start3 = i * 3;
  const auto *p = &positions[i * positionBytes];
  std::copy(p, p + positionBytes, result.position.data());
  std::copy(&scales[start3], &scales[start3] + 3, result.scale.data());
  int32_t rotationBytes = usesQuaternionSmallestThree ? 4 : 3;
  const auto& r = &rotations[i * rotationBytes];
  std::copy(r, r + rotationBytes, result.rotation.data());
  std::copy(&colors[start3], &colors[start3] + 3, result.color.data());
  result.alpha = alphas[i];

  int32_t shDim = dimForDegree(shDegree);
  const auto *sh = &this->sh[i * shDim * 3];
  for (int32_t j = 0; j < shDim; ++j, sh += 3) {
    result.shR[j] = sh[0];
    result.shG[j] = sh[1];
    result.shB[j] = sh[2];
  }
  for (int32_t j = shDim; j < SH_MAX_COEFFS; ++j) {
    result.shR[j] = 128;
    result.shG[j] = 128;
    result.shB[j] = 128;
  }

  return result;
}

UnpackedGaussian PackedGaussians::unpack(int32_t i, const CoordinateConverter &c) const {
  return at(i).unpack(usesFloat16(), usesQuaternionSmallestThree, fractionalBits, c);
}

bool PackedGaussians::usesFloat16() const {
  return numPoints >= 0 &&
         positions.size() == static_cast<size_t>(numPoints) * 3 * 2;
}

GaussianCloud unpackGaussians(const PackedGaussians &packed, const UnpackOptions &o) {
  const int32_t numPoints = packed.numPoints;
  const int32_t shDim = dimForDegree(packed.shDegree);
  const bool usesFloat16 = packed.usesFloat16();
  const bool usesQuaternionSmallestThree = packed.usesQuaternionSmallestThree;
  if (!checkSizes(packed, numPoints, shDim, usesFloat16)) {
    return {};
  }

  GaussianCloud result;
  result.numPoints = packed.numPoints;
  result.shDegree = packed.shDegree;
  result.antialiased = packed.antialiased;

#ifdef SPZ_BUILD_EXTENSIONS
  // Copy all extensions from PackedGaussians to GaussianCloud.
  // Note: Some extensions (like SH quantization) are only used during packing and may not
  // be needed in the unpacked cloud, but we preserve them for metadata completeness
  // and future extensibility.
  result.extensions = packed.extensions;
#endif

  const size_t count = static_cast<size_t>(numPoints);
  result.positions.resize(count * 3);
  result.scales.resize(count * 3);
  result.rotations.resize(count * 4);
  result.alphas.resize(count);
  result.colors.resize(count * 3);
  result.sh.resize(count * static_cast<size_t>(shDim) * 3);

  if (usesFloat16) {
    // Decode legacy float16 format. We can remove this at some point as it was never released.
    const auto *halfData = reinterpret_cast<const Half *>(packed.positions.data());
    for (size_t i = 0; i < count * 3; i++) {
      result.positions[i] = halfToFloat(halfData[i]);
    }
  } else {
    // Decode 24-bit fixed point coordinates
    float scale = 1.0 / (1 << packed.fractionalBits);
    for (size_t i = 0; i < count * 3; i++) {
      int32_t fixed32 = packed.positions[i * 3 + 0];
      fixed32 |= packed.positions[i * 3 + 1] << 8;
      fixed32 |= packed.positions[i * 3 + 2] << 16;
      fixed32 |= (fixed32 & 0x800000) ? 0xff000000 : 0;  // sign extension
      result.positions[i] = static_cast<float>(fixed32) * scale;
    }
  }

  for (size_t i = 0; i < numPoints * 3; i++) {
    result.scales[i] = packed.scales[i] / 16.0f - 10.0f;
  }

  for (size_t i = 0; i < numPoints; i++) {
    if (usesQuaternionSmallestThree) {
      unpackQuaternionSmallestThree(&result.rotations[4 * i], &packed.rotations[4 * i]);
    } else {
      unpackQuaternionFirstThree(&result.rotations[4 * i], &packed.rotations[3 * i]);
    }
  }

  for (size_t i = 0; i < numPoints; i++) {
    const float normalized_alpha = std::clamp(packed.alphas[i] / 255.0f, 1.0f / 255.0f, 254.0f / 255.0f);
    result.alphas[i] = invSigmoid(normalized_alpha);
  }

  for (size_t i = 0; i < numPoints * 3; i++) {
    result.colors[i] = ((packed.colors[i] / 255.0f) - 0.5f) / colorScale;
  }

  for (size_t i = 0; i < packed.sh.size(); i++) {
    result.sh[i] = unquantizeSH(packed.sh[i]);
  }

#ifdef SPZ_BUILD_EXTENSIONS
  {
    CoordinateSystem fromCoord = getPackedCoordinateSystem(result.extensions);
    result.convertCoordinates(fromCoord, o.to);
  }
#else
  if (packed.hadSkippedExtensions) {
    SpzLog("[SPZ WARNING] unpackGaussians: extensions were skipped at load time — "
           "unpacked data may be incorrect due to unknown packing behavior; "
           "build with SPZ_BUILD_EXTENSIONS to ensure correct results");
  }
  result.convertCoordinates(CoordinateSystem::RUB, o.to);
#endif
  return result;
}

void serializePackedGaussians(const PackedGaussians &packed, std::ostream *out) {
  LegacyPackedGaussiansHeader header;
  header.version = packed.version;
  header.numPoints = static_cast<uint32_t>(packed.numPoints);
  header.shDegree = static_cast<uint8_t>(packed.shDegree);
  header.fractionalBits = static_cast<uint8_t>(packed.fractionalBits);
  header.flags = static_cast<uint8_t>(packed.antialiased ? FlagAntialiased : 0)
#ifdef SPZ_BUILD_EXTENSIONS
    | static_cast<uint8_t>(packed.extensions.empty() ? 0 : FlagHasExtensions)
#endif
    ;
  out->write(reinterpret_cast<const char *>(&header), sizeof(header));

  out->write(reinterpret_cast<const char *>(packed.positions.data()), countBytes(packed.positions));
  out->write(reinterpret_cast<const char *>(packed.alphas.data()), countBytes(packed.alphas));
  out->write(reinterpret_cast<const char *>(packed.colors.data()), countBytes(packed.colors));
  out->write(reinterpret_cast<const char *>(packed.scales.data()), countBytes(packed.scales));
  out->write(reinterpret_cast<const char *>(packed.rotations.data()), countBytes(packed.rotations));
  out->write(reinterpret_cast<const char *>(packed.sh.data()), countBytes(packed.sh));

  // Write extensions at the end
#ifdef SPZ_BUILD_EXTENSIONS
  writeAllExtensions(packed.extensions, *out);
#endif
}

// Decompresses the NGSP attribute streams directly into the caller-provided destination buffers,
// avoiding any intermediate combined-buffer allocation.  dests must have exactly header.numStreams
// entries, each pre-sized to the expected uncompressed byte count for that stream.
bool decompressNgspStreams(const uint8_t *data, size_t size,
                           const NgspFileHeader &header,
                           const std::vector<std::pair<uint8_t *, size_t>> &dests) {
  if (header.tocByteOffset < sizeof(NgspFileHeader)) {
    SpzLog("[SPZ ERROR] decompressNgspStreams: TOC byte offset is less than the size of the header");
    return false;
  }
  const size_t tocSize = header.numStreams * 2 * sizeof(uint64_t);
  const size_t tocEnd = header.tocByteOffset + tocSize;
  if (tocEnd > size) {
    SpzLog("[SPZ ERROR] decompressNgspStreams: TOC end is greater than the size of the data");
    return false;
  }
  if (dests.size() != header.numStreams) {
    SpzLog("[SPZ ERROR] decompressNgspStreams: stream count mismatch");
    return false;
  }

  struct StreamInfo {
    uint64_t compressedSize;
    uint64_t uncompressedSize;
    size_t compressedOffset;
  };
  std::vector<StreamInfo> infos(header.numStreams);
  size_t compressedOffset = tocEnd;
  for (uint8_t i = 0; i < header.numStreams; i++) {
    const size_t e = header.tocByteOffset + i * 16;
    std::memcpy(&infos[i].compressedSize, data + e, sizeof(uint64_t));
    std::memcpy(&infos[i].uncompressedSize, data + e + sizeof(uint64_t), sizeof(uint64_t));
    infos[i].compressedOffset = compressedOffset;
    if (infos[i].compressedSize > size - compressedOffset) {
      SpzLog("[SPZ ERROR] decompressNgspStreams: stream extends past end of data");
      return false;
    }
    compressedOffset += infos[i].compressedSize;
    if (infos[i].uncompressedSize != dests[i].second) {
      SpzLog("[SPZ ERROR] decompressNgspStreams: stream size mismatch");
      return false;
    }
  }
  if (compressedOffset != size) {
    SpzLog("[SPZ ERROR] decompressNgspStreams: compressed data size mismatch");
    return false;
  }

#if defined(__EMSCRIPTEN__)
  // TODO: Add support for parallel decompression on WASM.
  for (size_t i = 0; i < infos.size(); i++) {
    const size_t ret = ZSTD_decompress(
      dests[i].first, dests[i].second,
      data + infos[i].compressedOffset, infos[i].compressedSize);
    if (ZSTD_isError(ret) || ret != dests[i].second) {
      SpzLog("[SPZ ERROR] decompressNgspStreams: ZSTD decompression failed");
      return false;
    }
  }
#else
  std::vector<std::future<bool>> futures;
  for (size_t i = 0; i < infos.size(); i++) {
    const auto info = infos[i];
    const auto dest = dests[i];
    futures.push_back(std::async(std::launch::async, [data, info, dest]() -> bool {
      const size_t ret = ZSTD_decompress(
        dest.first, dest.second,
        data + info.compressedOffset, info.compressedSize);
      if (ZSTD_isError(ret) || ret != dest.second) {
        SpzLog("[SPZ ERROR] decompressNgspStreams: ZSTD decompression failed");
        return false;
      }
      return true;
    }));
  }
  for (auto &f : futures) {
    if (!f.get()) return false;
  }
#endif
  return true;
}

bool compressNgspStreams(const std::vector<std::pair<const uint8_t *, size_t>> &srcs,
                         std::vector<std::vector<uint8_t>> *chunks,
                         std::vector<uint64_t> *uncompressedSizes) {
#if defined(__EMSCRIPTEN__)
  // TODO: Add support for parallel compression on WASM.
  for (const auto &s : srcs) {
    if (s.second == 0) continue;
    uncompressedSizes->push_back(s.second);
    std::vector<uint8_t> chunk;
    if (!compressZstd(s.first, s.second, &chunk)) {
      SpzLog("[SPZ ERROR] compressNgspStreams: ZSTD compression failed");
      return false;
    }
    chunks->push_back(std::move(chunk));
  }
#else
  std::vector<std::future<std::vector<uint8_t>>> futures;
  for (const auto &s : srcs) {
    if (s.second == 0) continue;
    uncompressedSizes->push_back(s.second);
    futures.push_back(std::async(std::launch::async, [s]() -> std::vector<uint8_t> {
      std::vector<uint8_t> chunk;
      if (!compressZstd(s.first, s.second, &chunk)) {
        SpzLog("[SPZ ERROR] compressNgspStreams: ZSTD compression failed");
        return {};
      }
      return chunk;
    }));
  }
  for (auto &f : futures) {
    chunks->push_back(f.get());
    if (chunks->back().empty()) {
      SpzLog("[SPZ ERROR] compressNgspStreams: compression failed");
      return false;
    }
  }
#endif
  return true;
}

PackedGaussians loadPackedGaussiansFromNgsp(const uint8_t *data, size_t size,
                                            const NgspFileHeader &header) {
  if (header.numPoints == 0 || header.numPoints > kMaxSpzPoints ||
      header.fractionalBits > kMaxSpzFractionalBits ||
      header.shDegree > SH_MAX_DEGREE) {
    SpzLog("[SPZ ERROR] loadPackedGaussiansFromNgsp: header exceeds safety limits "
           "(numPoints=%u fractionalBits=%u shDegree=%u)",
           header.numPoints, header.fractionalBits, header.shDegree);
    return {};
  }

  const int32_t numPoints = static_cast<int32_t>(header.numPoints);
  const int32_t shDim = dimForDegree(header.shDegree);
  const bool usesQuaternionSmallestThree = header.version >= MIN_SMALLEST_THREE_QUATERNIONS_VERSION;
  const size_t count = static_cast<size_t>(numPoints);
  const size_t rotBytes = usesQuaternionSmallestThree ? 4u : 3u;

  // Bound total attribute allocation before any resize.
  size_t total_bytes = 0;
  for (const size_t field_size :
       {size_t{9}, size_t{1}, size_t{3}, size_t{3}, rotBytes,
        static_cast<size_t>(shDim) * 3}) {
    size_t field_total = 0;
    if (!checkedMultiply(count, field_size, &field_total) ||
        !checkedAdd(total_bytes, field_total, &total_bytes)) {
      SpzLog("[SPZ ERROR] loadPackedGaussiansFromNgsp: attribute size overflow");
      return {};
    }
  }
  if (total_bytes > kMaxSpzDecompressedBytes) {
    SpzLog("[SPZ ERROR] loadPackedGaussiansFromNgsp: decompressed payload exceeds limit");
    return {};
  }

  PackedGaussians result;
  result.version = header.version;
  result.numPoints = numPoints;
  result.shDegree = header.shDegree;
  result.fractionalBits = header.fractionalBits;
  result.antialiased = (header.flags & FlagAntialiased) != 0;
  result.usesQuaternionSmallestThree = usesQuaternionSmallestThree;

  // Pre-size attribute vectors so decompressNgspStreams can write directly into them,
  // avoiding both the intermediate combined buffer and the readChunk copies.
  result.positions.resize(count * 9);
  result.alphas.resize(count);
  result.colors.resize(count * 3);
  result.scales.resize(count * 3);
  result.rotations.resize(count * rotBytes);
  result.sh.resize(count * static_cast<size_t>(shDim) * 3);

  // Build destination list in the same order saveSpz writes streams, skipping zero-size buffers.
  std::vector<std::pair<uint8_t *, size_t>> dests;
  for (auto attr : kAllSplatAttributes) {
    auto &v = packedBuffer(result, attr);
    if (!v.empty()) dests.push_back({v.data(), v.size()});
  }

  if (!decompressNgspStreams(data, size, header, dests)) {
    SpzLog("[SPZ ERROR] loadSpzPacked: NGSP stream decompression failed");
    return {};
  }

  if ((header.flags & FlagHasExtensions) != 0) {
    const size_t extStart = sizeof(NgspFileHeader);
    const size_t extEnd = header.tocByteOffset;
    if (extStart < extEnd && extEnd <= size) {
#ifdef SPZ_BUILD_EXTENSIONS
      std::string extStr(reinterpret_cast<const char *>(data + extStart), extEnd - extStart);
      std::istringstream extStream(std::move(extStr));
      readAllExtensions(extStream, result.extensions);
#else
      SpzLog("[SPZ WARNING] loadSpzPacked: file has extensions but extension support is disabled — "
             "skipped extensions may affect how data was packed or will be unpacked; "
             "build with SPZ_BUILD_EXTENSIONS to ensure correct results");
      result.hadSkippedExtensions = true;
#endif
    }
  }

  return result;
}

PackedGaussians deserializePackedGaussians(std::istream &in) {
  LegacyPackedGaussiansHeader header;
  in.read(reinterpret_cast<char *>(&header), sizeof(header));
  if (!in || header.magic != NGSP_MAGIC) {
    SpzLog("[SPZ ERROR] deserializePackedGaussians: header not found");
    return {};
  }
  if (header.version < 1 || header.version > LATEST_SPZ_HEADER_VERSION) {
    SpzLog("[SPZ ERROR] deserializePackedGaussians: version not supported: %d", header.version);
    return {};
  }
  // Bound numPoints against the bytes actually remaining in the decompressed legacy stream.
  // Legacy files are already gzip-decompressed at this point, so the ratio of header-claimed
  // points to remaining bytes can't exceed ~1 point per kMinBytesPerPoint bytes.
  size_t remaining = 0;
  {
    const std::streampos cur = in.tellg();
    if (cur != std::streampos(-1)) {
      in.seekg(0, std::ios::end);
      const std::streampos end = in.tellg();
      in.seekg(cur);
      if (end != std::streampos(-1)) remaining = static_cast<size_t>(end - cur);
    }
  }
  // Same INT32_MAX / kMaxSpzPoints caps as the NGSP path: numPoints is consumed as int32_t
  // below, so values above those limits are rejected. This also bounds numPoints when the
  // size probe above fails (remaining == 0), in which case the ratio check is skipped.
  if (header.numPoints == 0 ||
      header.numPoints > kMaxSpzPoints ||
      header.numPoints > static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) ||
      (remaining > 0 &&
       static_cast<size_t>(header.numPoints) > remaining / kMinBytesPerPoint)) {
    SpzLog("[SPZ ERROR] deserializePackedGaussians: invalid point count: %u", header.numPoints);
    return {};
  }
  if (header.fractionalBits > kMaxSpzFractionalBits) {
    SpzLog("[SPZ ERROR] deserializePackedGaussians: invalid fractional bits: %u",
           header.fractionalBits);
    return {};
  }
  if (header.shDegree > SH_MAX_DEGREE) {
    SpzLog("[SPZ ERROR] deserializePackedGaussians: Unsupported SH degree: %d", header.shDegree);
    return {};
  }
  SpzLogDebug(
    "[SPZ] deserializePackedGaussians: version=%d, numPoints=%d, shDegree=%d, fractionalBits=%d, antialiased=%d, hasExtensions=%d",
    header.version,
    header.numPoints,
    header.shDegree,
    header.fractionalBits,
    int((header.flags & FlagAntialiased) != 0),
    int((header.flags & FlagHasExtensions) != 0)
  );

  const int32_t numPoints = static_cast<int32_t>(header.numPoints);
  const int32_t shDim = dimForDegree(header.shDegree);
  const bool usesFloat16 = header.version == 1;
  const bool usesQuaternionSmallestThree = header.version >= MIN_SMALLEST_THREE_QUATERNIONS_VERSION;
  const bool hasExtensions = (header.flags & FlagHasExtensions) != 0;
  const size_t count = static_cast<size_t>(numPoints);

  // Bound expected attribute payload before allocating.
  size_t bytes_per_point = 0;
  for (const size_t field_size : {
           size_t{usesFloat16 ? 6U : 9U},
           size_t{1},
           size_t{3},
           size_t{3},
           size_t{usesQuaternionSmallestThree ? 4U : 3U},
           static_cast<size_t>(shDim) * 3}) {
    if (!checkedAdd(bytes_per_point, field_size, &bytes_per_point)) {
      return {};
    }
  }
  size_t expected_payload_size = 0;
  if (!checkedMultiply(count, bytes_per_point, &expected_payload_size) ||
      expected_payload_size > kMaxSpzDecompressedBytes - sizeof(header)) {
    SpzLog("[SPZ ERROR] deserializePackedGaussians: payload exceeds limit");
    return {};
  }

  PackedGaussians result;
  result.version = header.version;
  result.numPoints = numPoints;
  result.shDegree = header.shDegree;
  result.fractionalBits = header.fractionalBits;
  result.antialiased = (header.flags & FlagAntialiased) != 0;
  result.positions.resize(count * 3 * (usesFloat16 ? 2 : 3));
  result.scales.resize(count * 3);
  result.usesQuaternionSmallestThree = usesQuaternionSmallestThree;
  result.rotations.resize(count * (usesQuaternionSmallestThree ? 4 : 3));
  result.alphas.resize(count);
  result.colors.resize(count * 3);
  result.sh.resize(count * static_cast<size_t>(shDim) * 3);
  in.read(reinterpret_cast<char *>(result.positions.data()), countBytes(result.positions));
  in.read(reinterpret_cast<char *>(result.alphas.data()), countBytes(result.alphas));
  in.read(reinterpret_cast<char *>(result.colors.data()), countBytes(result.colors));
  in.read(reinterpret_cast<char *>(result.scales.data()), countBytes(result.scales));
  in.read(reinterpret_cast<char *>(result.rotations.data()), countBytes(result.rotations));
  in.read(reinterpret_cast<char *>(result.sh.data()), countBytes(result.sh));

  // Read extensions at the end
  if (hasExtensions) {
#ifdef SPZ_BUILD_EXTENSIONS
    readAllExtensions(in, result.extensions);
#else
    SpzLog("[SPZ WARNING] deserializePackedGaussians: stream has extensions but extension support is disabled — "
           "skipped extensions may affect how data was packed or will be unpacked; "
           "build with SPZ_BUILD_EXTENSIONS to ensure correct results");
    result.hadSkippedExtensions = true;
#endif
  }

  if (!in) {
    SpzLog("[SPZ ERROR] deserializePackedGaussians: read error");
    return {};
  }

  return result;
}

bool saveSpz(const GaussianCloud &g, const PackOptions &o, std::vector<uint8_t> *out) {
  PackedGaussians packed = packGaussians(g, o);

  if (g.numPoints > 0 && packed.numPoints == 0) {
    return false;
  }

  if (o.version < MIN_ZSTD_SPZ_HEADER_VERSION) {
    // Legacy gzip path for versions 1–3.
    std::stringstream ss;
    serializePackedGaussians(packed, &ss);
    const std::string data = ss.str();
    return compressGzipped(reinterpret_cast<const uint8_t *>(data.data()), data.size(), out);
  }

  std::vector<uint8_t> extensionData;
#ifdef SPZ_BUILD_EXTENSIONS
  if (!packed.extensions.empty()) {
    std::ostringstream extStream;
    writeAllExtensions(packed.extensions, extStream);
    const std::string &s = extStream.str();
    extensionData.assign(s.begin(), s.end());
  }
#endif

  const std::vector<std::pair<const uint8_t *, size_t>> srcs = {
    {packed.positions.data(), packed.positions.size()},
    {packed.alphas.data(),    packed.alphas.size()},
    {packed.colors.data(),    packed.colors.size()},
    {packed.scales.data(),    packed.scales.size()},
    {packed.rotations.data(), packed.rotations.size()},
    {packed.sh.data(),        packed.sh.size()},
  };

  std::vector<std::vector<uint8_t>> chunks;
  std::vector<uint64_t> uncompressedSizes;
  if (!compressNgspStreams(srcs, &chunks, &uncompressedSizes)) {
    return false;
  }

  const uint8_t numStreams = static_cast<uint8_t>(chunks.size());
  const uint32_t tocByteOffset = static_cast<uint32_t>(sizeof(NgspFileHeader) + extensionData.size());
  NgspFileHeader header;
  header.version        = o.version;
  header.numPoints      = packed.numPoints;
  header.shDegree       = packed.shDegree;
  header.fractionalBits = packed.fractionalBits;
  header.flags          = static_cast<uint8_t>(packed.antialiased ? FlagAntialiased : 0)
#ifdef SPZ_BUILD_EXTENSIONS
      | static_cast<uint8_t>(extensionData.empty() ? 0 : FlagHasExtensions)
#endif
      ;
  header.numStreams     = numStreams;
  header.tocByteOffset  = tocByteOffset;

  // Write plaintext zone: [header][extensions][TOC]
  out->resize(tocByteOffset + numStreams * 2 * sizeof(uint64_t));
  uint8_t *buf = out->data();
  std::memcpy(buf, &header, sizeof(header));
  if (!extensionData.empty()) {
    std::memcpy(buf + sizeof(NgspFileHeader), extensionData.data(), extensionData.size());
  }
  for (uint8_t i = 0; i < numStreams; i++) {
    const uint64_t cs = static_cast<uint64_t>(chunks[i].size());
    const uint64_t us = uncompressedSizes[i];
    const size_t e = tocByteOffset + i * 2 * sizeof(uint64_t);
    std::memcpy(buf + e,     &cs, sizeof(cs));
    std::memcpy(buf + e + 8, &us, sizeof(us));
  }
  for (const auto &chunk : chunks) {
    out->insert(out->end(), chunk.begin(), chunk.end());
  }
  return true;
}

PackedGaussians loadSpzPacked(const uint8_t *data, size_t size) {
  if (!data || size == 0 || size > kMaxSpzCompressedBytes) {
    return {};
  }

  uint32_t magic = 0;
  if (size >= sizeof(magic)) std::memcpy(&magic, data, sizeof(magic));

  if (magic == NGSP_MAGIC) {
    if (size < sizeof(NgspFileHeader)) {
      SpzLog("[SPZ ERROR] loadSpzPacked: NGSP file too short");
      return {};
    }

    NgspFileHeader header;
    std::memcpy(&header, data, sizeof(header));

    if (header.version < MIN_ZSTD_SPZ_HEADER_VERSION || header.version > LATEST_SPZ_HEADER_VERSION) {
      SpzLog("[SPZ ERROR] loadSpzPacked: unsupported version: %d", header.version);
      return {};
    }
    // numPoints is stored unsigned but consumed as int32_t downstream (loadPackedGaussiansFromNgsp),
    // so reject anything above INT32_MAX / kMaxSpzPoints up front — otherwise the cast wraps
    // negative. The file-size ratio check below is evaluated in 64-bit to avoid overflowing the
    // multiply on 32-bit targets.
    if (header.numPoints == 0 ||
        header.numPoints > kMaxSpzPoints ||
        header.numPoints > static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) ||
        header.fractionalBits > kMaxSpzFractionalBits ||
        static_cast<uint64_t>(header.numPoints) >
          (static_cast<uint64_t>(size) * kMaxCompressionRatio) / kMinBytesPerPoint) {
      SpzLog("[SPZ ERROR] loadSpzPacked: invalid point count: %u (file size %zu)",
             header.numPoints, size);
      return {};
    }
    SpzLogDebug(
      "[SPZ] loadSpzPacked (NGSP): version=%d, numPoints=%d, shDegree=%d, numStreams=%d, tocByteOffset=%d",
      header.version, header.numPoints, header.shDegree, header.numStreams, header.tocByteOffset);

    return loadPackedGaussiansFromNgsp(data, size, header);

  } else if (size >= 2 && data[0] == 0x1f && data[1] == 0x8b) {
    // Legacy single-stream GZip format (pre-v4).
    std::vector<uint8_t> decompressed;
    if (!decompressGzipped(data, size, &decompressed)) return {};
    if (decompressed.size() > kMaxSpzDecompressedBytes) return {};
    membuf buf(decompressed.data(), decompressed.size());
    std::istream stream(&buf);
    return deserializePackedGaussians(stream);

  } else {
    SpzLog("[SPZ ERROR] loadSpzPacked: unrecognized format");
    return {};
  }
}

PackedGaussians loadSpzPacked(const std::vector<uint8_t> &data) {
  if (data.empty() || data.size() > kMaxSpzCompressedBytes) {
    return {};
  }
  return loadSpzPacked(data.data(), data.size());
}

PackedGaussians loadSpzPacked(const std::string &filename) {
  std::ifstream in(filename, std::ios::binary | std::ios::ate);
  if (!in.good())
    return {};
  const std::streampos end = in.tellg();
  if (end <= 0 || static_cast<uint64_t>(end) > kMaxSpzCompressedBytes) {
    return {};
  }
  std::vector<uint8_t> data(static_cast<size_t>(end));
  in.seekg(0, std::ios::beg);
  in.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(data.size()));
  if (!in.good()) {
    return {};
  }
  return loadSpzPacked(data);
}

GaussianCloud loadSpz(const std::vector<uint8_t> &data, const UnpackOptions &o) {
  return unpackGaussians(loadSpzPacked(data), o);
}

GaussianCloud loadSpz(const uint8_t *data, size_t size, const UnpackOptions &o) {
  return unpackGaussians(loadSpzPacked(data, size), o);
}

bool saveSpz(const GaussianCloud &g, const PackOptions &o, const std::string &filename) {
  std::vector<uint8_t> data;
  if (!saveSpz(g, o, &data)) {
    return false;
  }
  std::ofstream out(filename, std::ios::binary | std::ios::out);
  out.write(reinterpret_cast<const char *>(data.data()), data.size());
  out.close();
  return out.good();
}

GaussianCloud loadSpz(const std::string &filename, const UnpackOptions &o) {
  std::ifstream in(filename, std::ios::binary | std::ios::ate);
  if (!in.good()) {
    SpzLog("[SPZ ERROR] Unable to open: %s", filename.c_str());
    return {};
  }
  std::vector<uint8_t> data(in.tellg());
  in.seekg(0, std::ios::beg);
  in.read(reinterpret_cast<char *>(data.data()), data.size());
  in.close();
  if (!in.good()) {
    SpzLog("[SPZ ERROR] Unable to load data from: %s", filename.c_str());
    return {};
  }
  return loadSpz(data, o);
}

bool getNextHeaderLine(std::ifstream &in, std::string &line) {
  while (std::getline(in, line)) {
    // Find the first non-whitespace character
    size_t start = line.find_first_not_of(" \t\n\r\f\v");
    // If line is empty or whitespace-only, skip it and continue reading.
    if (std::string::npos == start) {
      continue;
    }
    // Trim leading whitespace and check for 'comment'
    std::string trimmed_line = line.substr(start);
    if (trimmed_line.rfind("comment", 0) == 0) {
      continue; // Skip comment line
    }
    // Found a valid non-comment, non-empty line
    line = trimmed_line; // Update the reference string with the trimmed line
    return true;
  }
  // Failed to read a line (EOF or error)
  return false;
}

GaussianCloud loadSplatFromPly(const std::string &filename, const UnpackOptions &o) {
  SpzLogDebug("[SPZ] Loading: %s", filename.c_str());
  std::ifstream in(filename, std::ios::binary);
  if (!in.good()) {
    SpzLog("[SPZ ERROR] Unable to open: %s", filename.c_str());
    in.close();
    return {};
  }
  std::string line;
  std::getline(in, line);
  if (line != "ply") {
    SpzLog("[SPZ ERROR] %s: not a .ply file", filename.c_str());
    in.close();
    return {};
  }
  if (!getNextHeaderLine(in, line) || line != "format binary_little_endian 1.0") {
    SpzLog("[SPZ ERROR] %s: unsupported .ply format", filename.c_str());
    in.close();
    return {};
  }
  if (!getNextHeaderLine(in, line) || line.find("element vertex ") != 0) {
    SpzLog("[SPZ ERROR] %s: missing vertex count", filename.c_str());
    in.close();
    return {};
  }
  int32_t numPoints = std::stoi(line.substr(std::strlen("element vertex ")));
  if (numPoints <= 0) {
    SpzLog("[SPZ ERROR] %s: invalid vertex count: %d", filename.c_str(), numPoints);
    in.close();
    return {};
  }

  SpzLogDebug("[SPZ] Loading %d points", numPoints);
  std::unordered_map<std::string, int> fields;  // name -> index

  // Helper function to get property size from PLY type string
  auto getPropertySize = [](const std::string& line) -> size_t {
    if (line.find("property float ") == 0 || line.find("property int ") == 0 ||
        line.find("property uint ") == 0) {
      return 4;
    } else if (line.find("property double ") == 0) {
      return 8;
    } else if (line.find("property char ") == 0 || line.find("property uchar ") == 0) {
      return 1;
    } else if (line.find("property short ") == 0 || line.find("property ushort ") == 0) {
      return 2;
    }
    return 4;  // Default assumption
  };

  // Track extra elements (non-vertex) to handle their data
  std::vector<PlyExtraElement> extraElements;

  // State machine for parsing header
  enum class ParseState { IN_VERTEX, IN_EXTRA_ELEMENT };
  ParseState state = ParseState::IN_VERTEX;
  std::string currentElementName;
  int32_t currentElementCount = 0;
  size_t currentElementBytes = 0;
  bool currentElementIsKnown = false;

  for (int32_t i = 0;; i++) {
    if (!getNextHeaderLine(in, line)) {
      SpzLog("[SPZ ERROR] %s: unexpected EOF while reading header properties.", filename.c_str());
      in.close();
      return {};
    }
    if (line == "end_header") {
      // Finalize any pending extra element
      if (state == ParseState::IN_EXTRA_ELEMENT && currentElementCount > 0) {
        extraElements.push_back({currentElementName, currentElementCount, currentElementBytes, currentElementIsKnown});
      }
      break;
    }

    // Check for new element definitions (non-vertex)
    if (line.find("element ") == 0 && line.find("element vertex ") != 0) {
      // Finalize previous extra element if any
      if (state == ParseState::IN_EXTRA_ELEMENT && currentElementCount > 0) {
        extraElements.push_back({currentElementName, currentElementCount, currentElementBytes, currentElementIsKnown});
      }

      // Parse element name and count
      size_t spacePos = line.find(' ', 8);  // After "element "
      if (spacePos != std::string::npos) {
        currentElementName = line.substr(8, spacePos - 8);
        currentElementCount = std::stoi(line.substr(spacePos + 1));
        currentElementBytes = 0;

        // Check if this is a known element we handle specially (via extensions)
#ifdef SPZ_BUILD_EXTENSIONS
        currentElementIsKnown = isKnownPlyExtensionElement(currentElementName);
#else
        currentElementIsKnown = false;
#endif

        state = ParseState::IN_EXTRA_ELEMENT;
        if (!currentElementIsKnown) {
          SpzLogDebug("[SPZ] Found extra element: %s (%d items)", currentElementName.c_str(), currentElementCount);
        }
      }
      continue;
    }

    // Handle properties based on current state
    if (state == ParseState::IN_EXTRA_ELEMENT) {
      if (line.find("property ") == 0) {
        currentElementBytes += getPropertySize(line);
      }
      continue;
    }

    // We're in vertex element - only accept float properties
    if (line.find("property float ") != 0) {
      SpzLog("[SPZ ERROR] %s: unsupported vertex property type: %s", filename.c_str(), line.c_str());
      in.close();
      return {};
    }
    std::string name = line.substr(std::strlen("property float "));
    fields[name] = i;
  }

  // Returns the index for a given field name, ensuring the name exists.
  const auto index = [&fields](const std::string &name) {
    const auto &itr = fields.find(name);
    if (itr == fields.end()) {
      SpzLog("[SPZ ERROR] Missing field: %s", name.c_str());
      return -1;
    }
    return itr->second;
  };

  const std::vector<int> positionIdx = {index("x"), index("y"), index("z")};
  const std::vector<int> scaleIdx = {index("scale_0"), index("scale_1"), index("scale_2")};
  const std::vector<int> rotIdx = {index("rot_1"), index("rot_2"), index("rot_3"), index("rot_0")};
  const std::vector<int> alphaIdx = {index("opacity")};
  const std::vector<int> colorIdx = {index("f_dc_0"), index("f_dc_1"), index("f_dc_2")};

  // Check that only valid indices were returned.
  for (auto idx : positionIdx) {
    if (idx < 0) {
      in.close();
      return {};
    }
  }
  for (auto idx : scaleIdx) {
    if (idx < 0) {
      in.close();
      return {};
    }
  }
  for (auto idx : rotIdx) {
    if (idx < 0) {
      in.close();
      return {};
    }
  }
  for (auto idx : alphaIdx) {
    if (idx < 0) {
      in.close();
      return {};
    }
  }
  for (auto idx : colorIdx) {
    if (idx < 0) {
      in.close();
      return {};
    }
  }

  // Spherical harmonics are optional and variable in size (depending on degree)
  std::vector<int> shIdx;
  const int32_t shMaxCoeffsRGB = SH_MAX_COEFFS * 3;
  for (int32_t i = 0; i < shMaxCoeffsRGB; i++) {
    const auto &itr = fields.find("f_rest_" + std::to_string(i));
    if (itr == fields.end())
      break;
    shIdx.push_back(itr->second);
  }
  const int32_t shDim = static_cast<int>(shIdx.size() / 3);

  std::vector<float> values(numPoints * fields.size());
  in.read(reinterpret_cast<char *>(values.data()), values.size() * sizeof(float));
  if (!in.good()) {
    SpzLog("[SPZ ERROR] Unable to load data from: %s", filename.c_str());
    in.close();
    return {};
  }

  GaussianCloud result;
#ifdef SPZ_BUILD_EXTENSIONS
  readExtensionsFromPly(in, extraElements, result.extensions);
#endif

  // Skip data for extra elements (they appear after vertex and safe orbit data in the file)
  for (const auto& elem : extraElements) {
    if (elem.isKnown) continue;  // Already handled above
    size_t bytesToSkip = elem.count * elem.bytesPerElement;
    if (bytesToSkip > 0) {
      in.seekg(bytesToSkip, std::ios::cur);
      SpzLogDebug("[SPZ] Skipped %zu bytes for element '%s'", bytesToSkip, elem.name.c_str());
    }
  }

  in.close();

  result.numPoints = numPoints;
  result.shDegree = degreeForDim(shDim);
  result.positions.reserve(numPoints * 3);
  result.scales.reserve(numPoints * 3);
  result.rotations.reserve(numPoints * 4);
  result.alphas.reserve(numPoints * 1);
  result.colors.reserve(numPoints * 3);
  for (size_t i = 0; i < values.size(); i += fields.size()) {
    for (int32_t j = 0; j < positionIdx.size(); j++) {
      result.positions.push_back(values[i + positionIdx[j]]);
    }
    for (int32_t j = 0; j < scaleIdx.size(); j++) {
      result.scales.push_back(values[i + scaleIdx[j]]);
    }
    for (int32_t j = 0; j < rotIdx.size(); j++) {
      result.rotations.push_back(values[i + rotIdx[j]]);
    }
    for (int32_t j = 0; j < alphaIdx.size(); j++) {
      result.alphas.push_back(values[i + alphaIdx[j]]);
    }
    for (int32_t j = 0; j < colorIdx.size(); j++) {
      result.colors.push_back(values[i + colorIdx[j]]);
    }
    // Convert from [N,C,S] to [N,S,C] (where C is color channel, S is SH coeff).
    for (int32_t j = 0; j < shDim; j++) {
      result.sh.push_back(values[i + shIdx[j]]);
      result.sh.push_back(values[i + shIdx[j + shDim]]);
      result.sh.push_back(values[i + shIdx[j + 2 * shDim]]);
    }
  }

  result.convertCoordinates(CoordinateSystem::RDF, o.to);
  return result;
}

bool saveSplatToPly(const GaussianCloud &data, const PackOptions &o, const std::string &filename) {
  // Use int64_t for N so that N*D never overflows int32_t for clouds with >35M gaussians.
  const int64_t N = data.numPoints;
  CHECK_EQ(data.positions.size(), N * 3);
  CHECK_EQ(data.scales.size(), N * 3);
  CHECK_EQ(data.rotations.size(), N * 4);
  CHECK_EQ(data.alphas.size(), N);
  CHECK_EQ(data.colors.size(), N * 3);
  const int64_t shDim = (N > 0) ? static_cast<int64_t>(data.sh.size() / N / 3) : 0;
  const int64_t D = 17 + shDim * 3;

  CoordinateConverter c = coordinateConverter(o.from, CoordinateSystem::RDF, data.shDegree);

  std::ofstream out(filename, std::ios::binary);
  if (!out.good()) {
    SpzLog("[SPZ ERROR] Unable to open for writing: %s", filename.c_str());
    return false;
  }
  out << "ply\n";
  out << "format binary_little_endian 1.0\n";
  out << "element vertex " << N << "\n";
  out << "property float x\n";
  out << "property float y\n";
  out << "property float z\n";
  out << "property float nx\n";
  out << "property float ny\n";
  out << "property float nz\n";
  out << "property float f_dc_0\n";
  out << "property float f_dc_1\n";
  out << "property float f_dc_2\n";
  for (int64_t i = 0; i < shDim * 3; i++) {
    out << "property float f_rest_" << i << "\n";
  }
  out << "property float opacity\n";
  out << "property float scale_0\n";
  out << "property float scale_1\n";
  out << "property float scale_2\n";
  out << "property float rot_0\n";
  out << "property float rot_1\n";
  out << "property float rot_2\n";
  out << "property float rot_3\n";

#ifdef SPZ_BUILD_EXTENSIONS
  writeExtensionsToPlyHeader(data.extensions, out);
#endif

  out << "end_header\n";

  // Write in chunks of 1M points to bound peak memory regardless of cloud size.
  const int64_t kChunkSize = 1'000'000;
  std::vector<float> values(std::min(kChunkSize, N) * D);
  std::array<float, 4> bufQuat = {};
  for (int64_t start = 0; start < N; start += kChunkSize) {
    const int64_t end = std::min(start + kChunkSize, N);
    const int64_t rowCount = end - start;
    if (rowCount * D != static_cast<int64_t>(values.size())) {
      values.resize(rowCount * D);
    }
    int64_t outIdx = 0;
    for (int64_t i = start; i < end; i++) {
      const int64_t i3 = i * 3;
      const int64_t i4 = i * 4;
      // Position (x, y, z)
      for (size_t j = 0; j < 3; j++) { values[outIdx + j] = data.positions[i3 + j]; }
      if (c.rotFlipPFunc) {
        c.rotFlipPFunc(values.data() + outIdx);
      } else {
        for (size_t j = 0; j < 3; j++) { values[outIdx + j] *= c.flipP[j]; }
      }
      outIdx += 3;
      // Normals (nx, ny, nz): always zero, but some viewers expect them present
      values[outIdx] = 0.0f; values[outIdx + 1] = 0.0f; values[outIdx + 2] = 0.0f;
      outIdx += 3;
      // Color (r, g, b): DC component for spherical harmonics
      values[outIdx++] = data.colors[i3 + 0];
      values[outIdx++] = data.colors[i3 + 1];
      values[outIdx++] = data.colors[i3 + 2];
      // Spherical harmonics: Interleave so the coefficients are the fastest-changing axis and
      // the channel (r, g, b) is slower-changing axis.
      for (int64_t k = 0; k < 3; k++) {
        for (int64_t j = 0; j < shDim; j++) {
          values[outIdx + j] = data.sh[(i * shDim + j) * 3 + k];
        }
        for (int32_t band = 0; band < data.shDegree && band < SH_MAX_DEGREE; ++band) {
          if (c.rotFlipShFuncs[static_cast<size_t>(band)]) {
            c.rotFlipShFuncs[static_cast<size_t>(band)](values.data() + outIdx + static_cast<size_t>(band * (band + 2)));
          }
        }
        for (int64_t j = 0; j < shDim; j++) { values[outIdx + j] *= c.flipSh[j]; }
        outIdx += shDim;
      }
      // Alpha
      values[outIdx++] = data.alphas[i];
      // Scale (sx, sy, sz)
      values[outIdx++] = data.scales[i3 + 0];
      values[outIdx++] = data.scales[i3 + 1];
      values[outIdx++] = data.scales[i3 + 2];
      // Rotation (qw, qx, qy, qz)
      for (int32_t j = 0; j < 4; j++) { bufQuat[j] = data.rotations[i4 + j]; }
      if (c.rotFlipQFunc) {
        c.rotFlipQFunc(bufQuat.data());
      } else {
        for (int32_t j = 0; j < 3; j++) { bufQuat[j] *= c.flipQ[j]; }
      }
      // data.rotations are x,y,z,w per point; PLY expects w then x,y,z (see property order below).
      values[outIdx++] = bufQuat[3];
      values[outIdx++] = bufQuat[0];
      values[outIdx++] = bufQuat[1];
      values[outIdx++] = bufQuat[2];
    }
    CHECK_EQ(outIdx, rowCount * D);
    out.write(reinterpret_cast<char *>(values.data()), rowCount * D * sizeof(float));
  }

#ifdef SPZ_BUILD_EXTENSIONS
  writeExtensionsToPlyData(data.extensions, out);
#endif

  out.close();
  if (!out.good()) {
    SpzLog("[SPZ ERROR] Failed to write to: %s", filename.c_str());
    return false;
  }
  return true;
}

bool hasExtensionSupport() {
#ifdef SPZ_BUILD_EXTENSIONS
  return true;
#else
  return false;
#endif
}

}  // namespace spz
