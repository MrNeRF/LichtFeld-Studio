/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rad.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/splat_data_transform.hpp"
#include "core/splat_simplify.hpp"
#include "core/tensor.hpp"
#include "io/atomic_output.hpp"
#include "io/error.hpp"
#include "nanoflann.hpp"

#include <nlohmann/json.hpp>
#include <zlib.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace lfs::io {

    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::SplatData;
    using lfs::core::Tensor;
    using lfs::core::TensorShape;

    namespace {

        // ============================================================================
        // Constants
        // ============================================================================

        constexpr uint32_t RAD_MAGIC = 0x30444152;       // "RAD0" in little-endian
        constexpr uint32_t RAD_CHUNK_MAGIC = 0x43444152; // "RADC" in little-endian
        constexpr uint32_t CHUNK_SIZE = 65536;           // Splats per chunk
        constexpr int GZ_LEVEL = 6;                      // Default gzip compression level
        constexpr float SH_C0 = 0.28209479177387814f;    // Degree-0 SH basis constant

        // SH coefficient count per degree: 0->0, 1->3, 2->8, 3->15
        constexpr int SH_COEFFS_FOR_DEGREE[] = {0, 3, 8, 15};

        // ============================================================================
        // Encoding Type Enums
        // ============================================================================

        enum class RadCenterEncoding : uint8_t {
            Auto = 0,
            F32 = 1,
            F32LeBytes = 2,
            F16 = 3,
            F16LeBytes = 4
        };

        enum class RadAlphaEncoding : uint8_t {
            Auto = 0,
            F32 = 1,
            F16 = 2,
            R8 = 3
        };

        enum class RadRgbEncoding : uint8_t {
            Auto = 0,
            F32 = 1,
            F16 = 2,
            R8 = 3,
            R8Delta = 4
        };

        enum class RadScalesEncoding : uint8_t {
            Auto = 0,
            F32 = 1,
            Ln0R8 = 2,
            LnF16 = 3
        };

        enum class RadOrientationEncoding : uint8_t {
            Auto = 0,
            F32 = 1,
            F16 = 2,
            Oct88R8 = 3
        };

        enum class RadShEncoding : uint8_t {
            Auto = 0,
            F32 = 1,
            F16 = 2,
            S8 = 3,
            S8Delta = 4
        };

        // ============================================================================
        // Property Names
        // ============================================================================

        constexpr const char* PROP_CENTER = "center";
        constexpr const char* PROP_ALPHA = "alpha";
        constexpr const char* PROP_RGB = "rgb";
        constexpr const char* PROP_SCALES = "scales";
        constexpr const char* PROP_ORIENTATION = "orientation";
        constexpr const char* PROP_SH1 = "sh1";
        constexpr const char* PROP_SH2 = "sh2";
        constexpr const char* PROP_SH3 = "sh3";
        constexpr const char* PROP_CHILD_COUNT = "child_count";
        constexpr const char* PROP_CHILD_START = "child_start";
        constexpr const char* PROP_SH1_CODE = "sh1_code";
        constexpr const char* PROP_SH2_CODE = "sh2_code";
        constexpr const char* PROP_SH3_CODE = "sh3_code";
        constexpr const char* PROP_SH_LABEL = "sh_label";

        // ============================================================================
        // Utility Functions
        // ============================================================================

        // Write little-endian uint16_t
        inline void encode_u16(uint8_t* dst, uint16_t value) {
            dst[0] = static_cast<uint8_t>(value & 0xFF);
            dst[1] = static_cast<uint8_t>((value >> 8) & 0xFF);
        }

        // Write little-endian uint32_t
        inline void encode_u32(uint8_t* dst, uint32_t value) {
            dst[0] = static_cast<uint8_t>(value & 0xFF);
            dst[1] = static_cast<uint8_t>((value >> 8) & 0xFF);
            dst[2] = static_cast<uint8_t>((value >> 16) & 0xFF);
            dst[3] = static_cast<uint8_t>((value >> 24) & 0xFF);
        }

        // Write little-endian uint64_t
        inline void encode_u64(uint8_t* dst, uint64_t value) {
            dst[0] = static_cast<uint8_t>(value & 0xFF);
            dst[1] = static_cast<uint8_t>((value >> 8) & 0xFF);
            dst[2] = static_cast<uint8_t>((value >> 16) & 0xFF);
            dst[3] = static_cast<uint8_t>((value >> 24) & 0xFF);
            dst[4] = static_cast<uint8_t>((value >> 32) & 0xFF);
            dst[5] = static_cast<uint8_t>((value >> 40) & 0xFF);
            dst[6] = static_cast<uint8_t>((value >> 48) & 0xFF);
            dst[7] = static_cast<uint8_t>((value >> 56) & 0xFF);
        }

        // Read little-endian uint16_t
        inline uint16_t decode_u16(const uint8_t* src) {
            return static_cast<uint16_t>(src[0]) |
                   (static_cast<uint16_t>(src[1]) << 8);
        }

        // Read little-endian uint32_t
        inline uint32_t decode_u32(const uint8_t* src) {
            return static_cast<uint32_t>(src[0]) |
                   (static_cast<uint32_t>(src[1]) << 8) |
                   (static_cast<uint32_t>(src[2]) << 16) |
                   (static_cast<uint32_t>(src[3]) << 24);
        }

        // Read little-endian uint64_t
        inline uint64_t decode_u64(const uint8_t* src) {
            return static_cast<uint64_t>(src[0]) |
                   (static_cast<uint64_t>(src[1]) << 8) |
                   (static_cast<uint64_t>(src[2]) << 16) |
                   (static_cast<uint64_t>(src[3]) << 24) |
                   (static_cast<uint64_t>(src[4]) << 32) |
                   (static_cast<uint64_t>(src[5]) << 40) |
                   (static_cast<uint64_t>(src[6]) << 48) |
                   (static_cast<uint64_t>(src[7]) << 56);
        }

        // Pad size to 8-byte alignment
        inline size_t pad8(size_t size) {
            return (size + 7) & ~7;
        }

        // Padding bytes required to reach 8-byte alignment
        inline size_t pad8_len(size_t size) {
            return (8 - (size & 7)) & 7;
        }

        // ============================================================================
        // Half-Precision Float Conversion
        // ============================================================================

        // Convert float32 to float16 (IEEE 754)
        inline uint16_t float32_to_float16(float value) {
            uint32_t f32;
            std::memcpy(&f32, &value, sizeof(float));

            uint32_t sign = (f32 >> 31) & 0x1;
            uint32_t exponent = (f32 >> 23) & 0xFF;
            uint32_t mantissa = f32 & 0x7FFFFF;

            uint16_t f16;

            if (exponent == 0) {
                // Zero or subnormal - flush to zero
                f16 = static_cast<uint16_t>(sign << 15);
            } else if (exponent == 0xFF) {
                // Infinity or NaN
                f16 = static_cast<uint16_t>((sign << 15) | 0x7C00 | (mantissa >> 13));
            } else {
                // Normal number
                int32_t new_exp = static_cast<int32_t>(exponent) - 127 + 15;
                if (new_exp >= 31) {
                    // Overflow to infinity
                    f16 = static_cast<uint16_t>((sign << 15) | 0x7C00);
                } else if (new_exp <= 0) {
                    // Underflow to zero
                    f16 = static_cast<uint16_t>(sign << 15);
                } else {
                    uint32_t new_mantissa = mantissa >> 13;
                    // Round to nearest even
                    if ((mantissa & 0x1FFF) > 0x1000 ||
                        ((mantissa & 0x1FFF) == 0x1000 && (new_mantissa & 1))) {
                        new_mantissa++;
                    }
                    f16 = static_cast<uint16_t>((sign << 15) | (static_cast<uint32_t>(new_exp) << 10) | new_mantissa);
                }
            }

            return f16;
        }

        // Convert float16 to float32
        inline float float16_to_float32(uint16_t value) {
            uint32_t sign = (value >> 15) & 0x1;
            uint32_t exponent = (value >> 10) & 0x1F;
            uint32_t mantissa = value & 0x3FF;

            uint32_t f32;

            if (exponent == 0) {
                // Zero or subnormal
                if (mantissa == 0) {
                    f32 = sign << 31;
                } else {
                    // Subnormal - normalize it
                    int shift = 10 - static_cast<int>(std::log2(mantissa));
                    exponent = 1 - shift;
                    mantissa <<= shift;
                    f32 = (sign << 31) | ((exponent + 127 - 15) << 23) | ((mantissa & 0x3FF) << 13);
                }
            } else if (exponent == 0x1F) {
                // Infinity or NaN
                f32 = (sign << 31) | (0xFF << 23) | (mantissa << 13);
            } else {
                // Normal number
                f32 = (sign << 31) | ((exponent + 127 - 15) << 23) | (mantissa << 13);
            }

            float result;
            std::memcpy(&result, &f32, sizeof(float));
            return result;
        }

        // ============================================================================
        // RAD Compression/Decompression
        // ============================================================================

        // Reference RAD writers mark compression as "gz" but emit raw DEFLATE streams
        // (without gzip/zlib wrapper bytes).
        std::vector<uint8_t> rad_compress(const uint8_t* data, size_t size, int level = GZ_LEVEL) {
            if (size == 0) {
                return {};
            }

            z_stream strm{};
            const int init_ret = deflateInit2(&strm, level, Z_DEFLATED, -15, 8, Z_DEFAULT_STRATEGY);
            if (init_ret != Z_OK) {
                LOG_ERROR("rad_compress: deflateInit2 failed (ret={}, level={})", init_ret, level);
                return {};
            }

            strm.next_in = const_cast<Bytef*>(data);
            strm.avail_in = static_cast<uInt>(size);

            std::vector<uint8_t> output;
            output.reserve(deflateBound(&strm, static_cast<uLong>(size)));

            const size_t chunk_size = 65536;
            std::vector<uint8_t> chunk(chunk_size);
            bool success = false;

            while (true) {
                strm.next_out = chunk.data();
                strm.avail_out = static_cast<uInt>(chunk.size());

                int ret = deflate(&strm, Z_FINISH);
                if (ret != Z_OK && ret != Z_STREAM_END) {
                    LOG_ERROR("rad_compress: deflate failed with error {}", ret);
                    break;
                }

                size_t have = chunk.size() - strm.avail_out;
                output.insert(output.end(), chunk.begin(), chunk.begin() + have);

                if (ret == Z_STREAM_END) {
                    success = true;
                    break;
                }
            }

            deflateEnd(&strm);

            if (!success) {
                LOG_ERROR("rad_compress: compression failed");
                return {};
            }

            return output;
        }

        std::optional<std::vector<uint8_t>> inflate_with_window_bits(const uint8_t* data, size_t size, int window_bits) {
            z_stream strm = {};
            strm.zalloc = Z_NULL;
            strm.zfree = Z_NULL;
            strm.opaque = Z_NULL;
            strm.avail_in = static_cast<uInt>(size);
            strm.next_in = const_cast<Bytef*>(data);

            int ret = inflateInit2(&strm, window_bits);
            if (ret != Z_OK) {
                return std::nullopt;
            }

            std::vector<uint8_t> output;
            const size_t chunk_size = 65536;
            std::vector<uint8_t> chunk(chunk_size);
            bool success = false;

            do {
                strm.avail_out = static_cast<uInt>(chunk.size());
                strm.next_out = chunk.data();

                ret = inflate(&strm, Z_NO_FLUSH);
                if (ret == Z_STREAM_ERROR || ret == Z_DATA_ERROR || ret == Z_MEM_ERROR) {
                    inflateEnd(&strm);
                    return std::nullopt;
                }

                size_t have = chunk.size() - strm.avail_out;
                output.insert(output.end(), chunk.begin(), chunk.begin() + have);
                if (ret == Z_STREAM_END) {
                    success = true;
                }
            } while (ret != Z_STREAM_END);

            inflateEnd(&strm);
            if (!success) {
                return std::nullopt;
            }
            return output;
        }

        std::vector<uint8_t> rad_decompress(const uint8_t* data, size_t size) {
            // Match reference first.
            if (auto raw = inflate_with_window_bits(data, size, -15)) {
                return std::move(*raw);
            }
            // Backward compatibility for older local files.
            if (auto gzip = inflate_with_window_bits(data, size, 15 + 16)) {
                return std::move(*gzip);
            }
            // Accept zlib wrapper as a permissive fallback.
            if (auto zlib = inflate_with_window_bits(data, size, 15)) {
                return std::move(*zlib);
            }
            return {};
        }

        // ============================================================================
        // Encoding Functions
        // ============================================================================

        // Encode interleaved [count, dims] floats into dimension-major f32 bytes.
        std::vector<uint8_t> encode_f32(const float* data, size_t dims, size_t count) {
            std::vector<uint8_t> result(count * dims * 4);
            size_t out_idx = 0;

            for (size_t d = 0; d < dims; ++d) {
                size_t index = d;
                for (size_t i = 0; i < count; ++i) {
                    float v = data[index];
                    const auto* bytes = reinterpret_cast<const uint8_t*>(&v);
                    result[out_idx++] = bytes[0];
                    result[out_idx++] = bytes[1];
                    result[out_idx++] = bytes[2];
                    result[out_idx++] = bytes[3];
                    index += dims;
                }
            }
            return result;
        }

        void decode_f32(const uint8_t* encoded, float* output, size_t dims, size_t count) {
            for (size_t i = 0; i < count; ++i) {
                for (size_t d = 0; d < dims; ++d) {
                    const size_t src = (d * count + i) * 4;
                    uint32_t bits = decode_u32(&encoded[src]);
                    std::memcpy(&output[i * dims + d], &bits, sizeof(float));
                }
            }
        }

        // Encode interleaved [count, dims] floats into byte-interleaved little-endian blocks.
        std::vector<uint8_t> encode_f32_lebytes(const float* data, size_t dims, size_t count) {
            std::vector<uint8_t> result(count * dims * 4);
            const size_t stride = count * dims;
            for (size_t b = 0; b < 4; ++b) {
                for (size_t d = 0; d < dims; ++d) {
                    size_t index = d;
                    for (size_t i = 0; i < count; ++i) {
                        const auto* bytes = reinterpret_cast<const uint8_t*>(&data[index]);
                        result[b * stride + d * count + i] = bytes[b];
                        index += dims;
                    }
                }
            }
            return result;
        }

        void decode_f32_lebytes(const uint8_t* encoded, float* output, size_t dims, size_t count) {
            const size_t stride = count * dims;
            for (size_t i = 0; i < count; ++i) {
                for (size_t d = 0; d < dims; ++d) {
                    uint8_t bytes[4];
                    for (size_t b = 0; b < 4; ++b) {
                        bytes[b] = encoded[b * stride + d * count + i];
                    }
                    float v;
                    std::memcpy(&v, bytes, sizeof(float));
                    output[i * dims + d] = v;
                }
            }
        }

        std::vector<uint8_t> encode_f16(const float* data, size_t dims, size_t count) {
            std::vector<uint8_t> result(count * dims * 2);
            size_t out_idx = 0;

            for (size_t d = 0; d < dims; ++d) {
                size_t index = d;
                for (size_t i = 0; i < count; ++i) {
                    uint16_t v = float32_to_float16(data[index]);
                    result[out_idx++] = static_cast<uint8_t>(v & 0xFF);
                    result[out_idx++] = static_cast<uint8_t>((v >> 8) & 0xFF);
                    index += dims;
                }
            }
            return result;
        }

        void decode_f16(const uint8_t* encoded, float* output, size_t dims, size_t count) {
            for (size_t i = 0; i < count; ++i) {
                for (size_t d = 0; d < dims; ++d) {
                    const size_t src = (d * count + i) * 2;
                    uint16_t f16 = decode_u16(&encoded[src]);
                    output[i * dims + d] = float16_to_float32(f16);
                }
            }
        }

        std::vector<uint8_t> encode_f16_lebytes(const float* data, size_t dims, size_t count) {
            std::vector<uint8_t> result(count * dims * 2);
            const size_t stride = count * dims;
            for (size_t b = 0; b < 2; ++b) {
                for (size_t d = 0; d < dims; ++d) {
                    size_t index = d;
                    for (size_t i = 0; i < count; ++i) {
                        uint16_t f16 = float32_to_float16(data[index]);
                        result[b * stride + d * count + i] = (b == 0)
                                                                 ? static_cast<uint8_t>(f16 & 0xFF)
                                                                 : static_cast<uint8_t>((f16 >> 8) & 0xFF);
                        index += dims;
                    }
                }
            }
            return result;
        }

        void decode_f16_lebytes(const uint8_t* encoded, float* output, size_t dims, size_t count) {
            const size_t stride = count * dims;
            for (size_t i = 0; i < count; ++i) {
                for (size_t d = 0; d < dims; ++d) {
                    uint8_t b0 = encoded[d * count + i];
                    uint8_t b1 = encoded[stride + d * count + i];
                    output[i * dims + d] = float16_to_float32(static_cast<uint16_t>(b0 | (b1 << 8)));
                }
            }
        }

        // Encode f32 to R8 (8-bit quantized with min/max)
        struct R8Result {
            std::vector<uint8_t> data;
            float min_val;
            float max_val;
        };

        R8Result encode_r8(const float* data, size_t dims, size_t count, std::optional<float> forced_min = std::nullopt,
                           std::optional<float> forced_max = std::nullopt) {
            if (count == 0 || dims == 0) {
                return {{}, 0.0f, 0.0f};
            }

            float min_val = forced_min.value_or(data[0]);
            float max_val = forced_max.value_or(data[0]);
            if (!forced_min.has_value() || !forced_max.has_value()) {
                for (size_t i = 0; i < count * dims; ++i) {
                    min_val = std::min(min_val, data[i]);
                    max_val = std::max(max_val, data[i]);
                }
            }

            float range = max_val - min_val;
            if (range < 1e-7f) {
                range = 1e-7f;
            }

            std::vector<uint8_t> result(count * dims);
            size_t out_idx = 0;
            for (size_t d = 0; d < dims; ++d) {
                size_t index = d;
                for (size_t i = 0; i < count; ++i) {
                    float normalized = (data[index] - min_val) / range;
                    result[out_idx++] = static_cast<uint8_t>(std::clamp(std::round(normalized * 255.0f), 0.0f, 255.0f));
                    index += dims;
                }
            }

            return {result, min_val, max_val};
        }

        // Decode R8 to f32
        void decode_r8(const uint8_t* encoded, float* output, size_t dims, size_t count, float min_val, float max_val) {
            float range = max_val - min_val;
            for (size_t i = 0; i < count; ++i) {
                for (size_t d = 0; d < dims; ++d) {
                    const size_t src = d * count + i;
                    output[i * dims + d] = min_val + (static_cast<float>(encoded[src]) / 255.0f) * range;
                }
            }
        }

        // Encode f32 to R8 with delta encoding
        struct R8DeltaResult {
            std::vector<uint8_t> data;
            float min_val;
            float max_val;
        };

        R8DeltaResult encode_r8_delta(const float* data, size_t dims, size_t count,
                                      std::optional<float> forced_min = std::nullopt,
                                      std::optional<float> forced_max = std::nullopt) {
            if (count == 0 || dims == 0) {
                return {{}, 0.0f, 0.0f};
            }

            auto base_quant = encode_r8(data, dims, count, forced_min, forced_max);
            std::vector<uint8_t> result;
            result.reserve(base_quant.data.size());
            for (size_t d = 0; d < dims; ++d) {
                uint8_t last = 0;
                for (size_t i = 0; i < count; ++i) {
                    const uint8_t value = base_quant.data[d * count + i];
                    result.push_back(static_cast<uint8_t>(value - last));
                    last = value;
                }
            }
            return {result, base_quant.min_val, base_quant.max_val};
        }

        // Decode R8 delta to f32
        void decode_r8_delta(const uint8_t* encoded, float* output, size_t dims, size_t count, float min_val, float max_val) {
            std::vector<uint8_t> quantized(count * dims, 0);
            for (size_t d = 0; d < dims; ++d) {
                uint8_t last = 0;
                for (size_t i = 0; i < count; ++i) {
                    const size_t idx = d * count + i;
                    last = static_cast<uint8_t>(last + encoded[idx]);
                    quantized[idx] = last;
                }
            }
            decode_r8(quantized.data(), output, dims, count, min_val, max_val);
        }

        // Encode f32 to S8 (signed 8-bit for SH coefficients)
        struct S8Result {
            std::vector<int8_t> data;
            float max_abs;
        };

        S8Result encode_s8(const float* data, size_t dims, size_t count, std::optional<float> forced_max = std::nullopt) {
            float max_val = 0.0f;
            if (forced_max.has_value()) {
                max_val = std::max(1e-6f, std::abs(forced_max.value()));
            } else {
                for (size_t i = 0; i < count * dims; ++i) {
                    max_val = std::max(max_val, std::abs(data[i]));
                }
                max_val = std::max(max_val, 1e-6f);
            }

            std::vector<int8_t> result(count * dims);
            size_t out_idx = 0;
            for (size_t d = 0; d < dims; ++d) {
                size_t index = d;
                for (size_t i = 0; i < count; ++i) {
                    float scaled = data[index] / max_val * 127.0f;
                    result[out_idx++] = static_cast<int8_t>(std::clamp(std::round(scaled), -127.0f, 127.0f));
                    index += dims;
                }
            }

            return {result, max_val};
        }

        // Decode S8 to f32
        void decode_s8(const int8_t* encoded, float* output, size_t dims, size_t count, float max_abs) {
            for (size_t i = 0; i < count; ++i) {
                for (size_t d = 0; d < dims; ++d) {
                    const size_t src = d * count + i;
                    output[i * dims + d] = (static_cast<float>(encoded[src]) / 127.0f) * max_abs;
                }
            }
        }

        // Encode f32 to S8 with delta encoding
        struct S8DeltaResult {
            std::vector<int8_t> data;
            float max_abs;
        };

        S8DeltaResult encode_s8_delta(const float* data, size_t dims, size_t count, std::optional<float> forced_max = std::nullopt) {
            auto base_quant = encode_s8(data, dims, count, forced_max);
            std::vector<int8_t> result;
            result.reserve(base_quant.data.size());
            for (size_t d = 0; d < dims; ++d) {
                uint8_t last = 0;
                for (size_t i = 0; i < count; ++i) {
                    const uint8_t value = static_cast<uint8_t>(base_quant.data[d * count + i]);
                    result.push_back(static_cast<int8_t>(value - last));
                    last = value;
                }
            }
            return {result, base_quant.max_abs};
        }

        // Decode S8 delta to f32
        void decode_s8_delta(const int8_t* encoded, float* output, size_t dims, size_t count, float max_abs) {
            std::vector<int8_t> quantized(count * dims, 0);
            for (size_t d = 0; d < dims; ++d) {
                uint8_t last = 0;
                for (size_t i = 0; i < count; ++i) {
                    const size_t idx = d * count + i;
                    last = static_cast<uint8_t>(last + static_cast<uint8_t>(encoded[idx]));
                    quantized[idx] = static_cast<int8_t>(last);
                }
            }
            decode_s8(quantized.data(), output, dims, count, max_abs);
        }

        // Encode scales to log-space 8-bit with zero handling
        // Algorithm: scale -> ln(scale) -> quantize with zero handling
        // Value 0 is reserved for zero scales, values 1-255 encode ln(scale) in [ln_min, ln_max]
        struct Ln0R8Result {
            std::vector<uint8_t> data;
            float min_val;
            float max_val;
        };

        Ln0R8Result encode_ln_0r8(const float* data, size_t dims, size_t count) {
            // First pass: compute ln of all positive scales and find range
            std::vector<float> log_values;
            log_values.reserve(count * dims);
            float ln_min = std::numeric_limits<float>::infinity();
            float ln_max = -std::numeric_limits<float>::infinity();

            for (size_t i = 0; i < count * dims; ++i) {
                if (data[i] > 0.0f) {
                    float ln_val = std::log(data[i]);
                    log_values.push_back(ln_val);
                    ln_min = std::min(ln_min, ln_val);
                    ln_max = std::max(ln_max, ln_val);
                } else {
                    log_values.push_back(-std::numeric_limits<float>::infinity()); // Marker for zero
                }
            }

            // Handle edge case: all zeros or single value
            if (!std::isfinite(ln_min) || !std::isfinite(ln_max) || ln_max - ln_min < 1e-7f) {
                ln_min = -10.0f; // Default ~exp(-10) = 4.5e-5
                ln_max = 2.0f;   // Default exp(2) = 7.4
            }

            // Compute ln_zero threshold (scales below this encode to 0)
            float ln_zero = ln_min - 1.0f;

            std::vector<uint8_t> result;
            result.reserve(count * dims);
            for (size_t d = 0; d < dims; ++d) {
                size_t index = d;
                for (size_t i = 0; i < count; ++i) {
                    float scale = data[index];
                    uint8_t encoded;
                    if (scale <= 0.0f) {
                        encoded = 0;
                    } else {
                        float ln_scale = std::log(scale);
                        if (ln_scale <= ln_zero) {
                            encoded = 0;
                        } else {
                            float normalized = (ln_scale - ln_min) / (ln_max - ln_min) * 254.0f;
                            encoded = static_cast<uint8_t>(std::clamp(std::round(normalized), 0.0f, 254.0f)) + 1;
                        }
                    }
                    result.push_back(encoded);
                    index += dims;
                }
            }

            return {result, ln_min, ln_max};
        }

        // Decode log-space 8-bit to scales
        // Value 0 decodes to 0, values 1-255 decode to exp(ln) in [ln_min, ln_max]
        void decode_ln_0r8(const uint8_t* encoded, float* output, size_t dims, size_t count, float ln_min, float ln_max) {
            for (size_t i = 0; i < count; ++i) {
                for (size_t d = 0; d < dims; ++d) {
                    const size_t src = d * count + i;
                    uint8_t value = encoded[src];
                    if (value == 0) {
                        output[i * dims + d] = 0.0f;
                    } else {
                        float ln_scale = ln_min + (value - 1) * (ln_max - ln_min) / 254.0f;
                        output[i * dims + d] = std::exp(ln_scale);
                    }
                }
            }
        }

        // Encode scales to log-space f16
        std::vector<uint8_t> encode_ln_f16(const float* data, size_t dims, size_t count) {
            std::vector<float> log_data(count * dims);
            for (size_t i = 0; i < count * dims; ++i) {
                log_data[i] = std::log(data[i]);
            }
            return encode_f16(log_data.data(), dims, count);
        }

        // Decode log-space f16 to scales
        void decode_ln_f16(const uint8_t* encoded, float* output, size_t dims, size_t count) {
            decode_f16(encoded, output, dims, count);
            for (size_t i = 0; i < count * dims; ++i) {
                output[i] = std::exp(output[i]);
            }
        }

        // Encode quaternion to octahedral 3-byte representation using axis-angle encoding
        // Returns 3 bytes per quaternion:
        //   - 2 bytes: rotation axis encoded with octahedral projection
        //   - 1 byte: rotation angle (theta) quantized to 8 bits
        //
        // Algorithm:
        //   1. Ensure positive w (if w < 0, negate all components)
        //   2. Extract angle: theta = 2 * acos(w)
        //   3. Extract axis: if sin(theta/2) > epsilon, axis = (x,y,z) / sin(theta/2), else use (1,0,0)
        //   4. Encode axis to octahedral (2 bytes)
        //   5. Encode angle to 1 byte: theta / PI * 255
        std::vector<uint8_t> encode_quat_oct88r8(const float* quats, size_t count) {
            std::vector<uint8_t> result(count * 3);
            constexpr float PI = 3.14159265358979323846f;
            constexpr float EPSILON = 1e-6f;

            for (size_t i = 0; i < count; ++i) {
                float x = quats[i * 4 + 0];
                float y = quats[i * 4 + 1];
                float z = quats[i * 4 + 2];
                float w = quats[i * 4 + 3];

                // Normalize
                float len = std::sqrt(x * x + y * y + z * z + w * w);
                if (len > 0.0f) {
                    x /= len;
                    y /= len;
                    z /= len;
                    w /= len;
                }

                // Ensure positive w for consistency
                if (w < 0.0f) {
                    x = -x;
                    y = -y;
                    z = -z;
                    w = -w;
                }

                // Extract angle: theta = 2 * acos(w)
                float theta = 2.0f * std::acos(std::clamp(w, -1.0f, 1.0f));

                // Extract axis
                float sin_half_theta = std::sin(theta * 0.5f);
                float axis_x, axis_y, axis_z;
                if (sin_half_theta > EPSILON) {
                    axis_x = x / sin_half_theta;
                    axis_y = y / sin_half_theta;
                    axis_z = z / sin_half_theta;
                } else {
                    // Near identity rotation, use default axis
                    axis_x = 1.0f;
                    axis_y = 0.0f;
                    axis_z = 0.0f;
                }

                // Normalize axis
                float axis_len = std::sqrt(axis_x * axis_x + axis_y * axis_y + axis_z * axis_z);
                if (axis_len > 0.0f) {
                    axis_x /= axis_len;
                    axis_y /= axis_len;
                    axis_z /= axis_len;
                }

                // Octahedral encoding of axis (project to octahedron then to square)
                float abs_x = std::abs(axis_x);
                float abs_y = std::abs(axis_y);
                float abs_z = std::abs(axis_z);

                float oct_x, oct_y, oct_z;
                if (abs_x + abs_y + abs_z > 0.0f) {
                    float inv_sum = 1.0f / (abs_x + abs_y + abs_z);
                    oct_x = axis_x * inv_sum;
                    oct_y = axis_y * inv_sum;
                    oct_z = axis_z * inv_sum;
                } else {
                    oct_x = axis_x;
                    oct_y = axis_y;
                    oct_z = axis_z;
                }

                // Fold to upper hemisphere if needed
                if (oct_z < 0.0f) {
                    float temp_x = oct_x;
                    oct_x = (1.0f - std::abs(oct_y)) * (oct_x >= 0.0f ? 1.0f : -1.0f);
                    oct_y = (1.0f - std::abs(temp_x)) * (oct_y >= 0.0f ? 1.0f : -1.0f);
                }

                // Map from [-1, 1] to [0, 255] for axis
                result[i * 3 + 0] = static_cast<uint8_t>(std::clamp((oct_x + 1.0f) * 0.5f * 255.0f, 0.0f, 255.0f));
                result[i * 3 + 1] = static_cast<uint8_t>(std::clamp((oct_y + 1.0f) * 0.5f * 255.0f, 0.0f, 255.0f));

                // Encode angle: theta / PI * 255
                result[i * 3 + 2] = static_cast<uint8_t>(std::clamp(theta / PI * 255.0f, 0.0f, 255.0f));
            }

            return result;
        }

        // Decode octahedral 3-byte to quaternion using axis-angle decoding
        // Input: 3 bytes per quaternion (2 bytes axis + 1 byte angle)
        // Algorithm:
        //   1. Decode axis from octahedral (2 bytes)
        //   2. Decode angle from 1 byte: theta = value / 255.0 * PI
        //   3. Reconstruct quaternion: w = cos(theta/2), (x,y,z) = axis * sin(theta/2)
        void decode_quat_oct88r8(const uint8_t* encoded, float* quats, size_t count) {
            constexpr float PI = 3.14159265358979323846f;

            for (size_t i = 0; i < count; ++i) {
                // Map from [0, 255] to [-1, 1] for octahedral coordinates
                float oct_x = (static_cast<float>(encoded[i * 3 + 0]) / 255.0f) * 2.0f - 1.0f;
                float oct_y = (static_cast<float>(encoded[i * 3 + 1]) / 255.0f) * 2.0f - 1.0f;

                // Unfold from lower hemisphere when the stored oct_z was negative
                float oct_z = 1.0f - std::abs(oct_x) - std::abs(oct_y);
                if (oct_z < 0.0f) {
                    float temp_x = oct_x;
                    oct_x = (1.0f - std::abs(oct_y)) * (oct_x >= 0.0f ? 1.0f : -1.0f);
                    oct_y = (1.0f - std::abs(temp_x)) * (oct_y >= 0.0f ? 1.0f : -1.0f);
                    oct_z = -oct_z;
                }

                // Project from octahedron to sphere (normalize)
                float axis_x = oct_x;
                float axis_y = oct_y;
                float axis_z = oct_z;
                float len = std::sqrt(axis_x * axis_x + axis_y * axis_y + axis_z * axis_z);
                if (len > 0.0f) {
                    axis_x /= len;
                    axis_y /= len;
                    axis_z /= len;
                }

                // Decode angle: theta = value / 255.0 * PI
                float theta = (static_cast<float>(encoded[i * 3 + 2]) / 255.0f) * PI;

                // Reconstruct quaternion from axis-angle
                float half_theta = theta * 0.5f;
                float sin_half_theta = std::sin(half_theta);
                float cos_half_theta = std::cos(half_theta);

                float x = axis_x * sin_half_theta;
                float y = axis_y * sin_half_theta;
                float z = axis_z * sin_half_theta;
                float w = cos_half_theta;

                // Normalize
                float q_len = std::sqrt(x * x + y * y + z * z + w * w);
                if (q_len > 0.0f) {
                    x /= q_len;
                    y /= q_len;
                    z /= q_len;
                    w /= q_len;
                }

                quats[i * 4 + 0] = x;
                quats[i * 4 + 1] = y;
                quats[i * 4 + 2] = z;
                quats[i * 4 + 3] = w;
            }
        }

        // ============================================================================
        // Metadata Structures
        // ============================================================================

        struct RadChunkProperty {
            uint64_t offset = 0;
            uint64_t bytes = 0;
            std::string property;
            std::string encoding;
            std::optional<std::string> compression;
            std::optional<float> min_val;
            std::optional<float> max_val;
            std::optional<float> base;
            std::optional<float> scale;

            nlohmann::json to_json() const {
                nlohmann::json j;
                j["offset"] = offset;
                j["bytes"] = bytes;
                j["property"] = property;
                j["encoding"] = encoding;
                if (compression.has_value())
                    j["compression"] = compression.value();
                if (min_val.has_value())
                    j["min"] = min_val.value();
                if (max_val.has_value())
                    j["max"] = max_val.value();
                if (base.has_value())
                    j["base"] = base.value();
                if (scale.has_value())
                    j["scale"] = scale.value();
                return j;
            }

            static RadChunkProperty from_json(const nlohmann::json& j) {
                RadChunkProperty prop;
                prop.offset = j.at("offset").get<uint64_t>();
                prop.bytes = j.at("bytes").get<uint64_t>();
                prop.property = j.at("property").get<std::string>();
                prop.encoding = j.at("encoding").get<std::string>();
                if (j.contains("compression"))
                    prop.compression = j.at("compression").get<std::string>();
                if (j.contains("min"))
                    prop.min_val = j.at("min").get<float>();
                if (j.contains("max"))
                    prop.max_val = j.at("max").get<float>();
                if (j.contains("base"))
                    prop.base = j.at("base").get<float>();
                if (j.contains("scale"))
                    prop.scale = j.at("scale").get<float>();
                return prop;
            }
        };

        struct RadChunkMeta {
            uint32_t version = 1;
            uint64_t base = 0;
            uint64_t count = 0;
            uint64_t payload_bytes = 0;
            int max_sh = 0;
            bool lod_tree = false;
            std::optional<nlohmann::json> splat_encoding;
            std::vector<RadChunkProperty> properties;

            nlohmann::json to_json() const {
                nlohmann::json j;
                j["version"] = version;
                j["base"] = base;
                j["count"] = count;
                j["payloadBytes"] = payload_bytes; // camelCase
                if (max_sh > 0)
                    j["maxSh"] = max_sh; // camelCase, optional
                if (lod_tree)
                    j["lodTree"] = lod_tree; // camelCase, optional
                if (splat_encoding.has_value())
                    j["splatEncoding"] = splat_encoding.value(); // camelCase

                nlohmann::json props = nlohmann::json::array();
                for (const auto& prop : properties) {
                    props.push_back(prop.to_json());
                }
                j["properties"] = props;
                return j;
            }

            static RadChunkMeta from_json(const nlohmann::json& j) {
                RadChunkMeta meta;
                meta.version = j.at("version").get<uint32_t>();
                meta.base = j.at("base").get<uint64_t>();
                meta.count = j.at("count").get<uint64_t>();
                meta.payload_bytes = j.at("payloadBytes").get<uint64_t>(); // camelCase
                if (j.contains("maxSh"))
                    meta.max_sh = j.at("maxSh").get<int>();
                if (j.contains("lodTree"))
                    meta.lod_tree = j.at("lodTree").get<bool>();
                if (j.contains("splatEncoding")) {
                    meta.splat_encoding = j.at("splatEncoding");
                }
                for (const auto& prop_json : j.at("properties")) {
                    meta.properties.push_back(RadChunkProperty::from_json(prop_json));
                }
                return meta;
            }
        };

        struct RadChunkRange {
            uint64_t offset = 0;
            uint64_t bytes = 0;
            std::optional<uint64_t> base;
            std::optional<uint64_t> count;
            std::optional<std::string> filename;

            nlohmann::json to_json() const {
                nlohmann::json j;
                j["offset"] = offset;
                j["bytes"] = bytes;
                if (base.has_value())
                    j["base"] = base.value();
                if (count.has_value())
                    j["count"] = count.value();
                if (filename.has_value())
                    j["filename"] = filename.value();
                return j;
            }

            static RadChunkRange from_json(const nlohmann::json& j) {
                RadChunkRange range;
                range.offset = j.at("offset").get<uint64_t>();
                range.bytes = j.at("bytes").get<uint64_t>();
                if (j.contains("base"))
                    range.base = j.at("base").get<uint64_t>();
                if (j.contains("count"))
                    range.count = j.at("count").get<uint64_t>();
                if (j.contains("filename"))
                    range.filename = j.at("filename").get<std::string>();
                return range;
            }
        };

        struct RadMeta {
            uint32_t version = 1;
            std::string type = "gsplat";
            uint64_t count = 0;                 // Changed from uint32_t
            std::optional<int> max_sh;          // Changed to optional
            std::optional<bool> lod_tree;       // Changed to optional
            std::optional<uint32_t> chunk_size; // Changed to optional
            uint64_t all_chunk_bytes = 0;
            std::vector<RadChunkRange> chunks; // Changed from RadChunkMeta
            std::optional<nlohmann::json> splat_encoding;
            std::optional<uint32_t> sh_code_count; // Added missing field
            std::optional<std::string> comment;

            nlohmann::json to_json() const {
                nlohmann::json j;
                j["version"] = version;
                j["type"] = type;
                j["count"] = count;
                if (max_sh.has_value())
                    j["maxSh"] = max_sh.value(); // camelCase
                if (lod_tree.has_value())
                    j["lodTree"] = lod_tree.value(); // camelCase
                if (chunk_size.has_value())
                    j["chunkSize"] = chunk_size.value(); // camelCase
                j["allChunkBytes"] = all_chunk_bytes;    // camelCase

                nlohmann::json chunks_json = nlohmann::json::array();
                for (const auto& chunk : chunks) {
                    chunks_json.push_back(chunk.to_json());
                }
                j["chunks"] = chunks_json;

                if (splat_encoding.has_value()) {
                    j["splatEncoding"] = splat_encoding.value(); // camelCase
                }
                if (sh_code_count.has_value()) {
                    j["shCodeCount"] = sh_code_count.value(); // camelCase
                }
                if (comment.has_value()) {
                    j["comment"] = comment.value();
                }
                return j;
            }

            static RadMeta from_json(const nlohmann::json& j) {
                RadMeta meta;
                meta.version = j.at("version").get<uint32_t>();
                meta.type = j.at("type").get<std::string>();
                meta.count = j.at("count").get<uint64_t>();
                if (j.contains("maxSh"))
                    meta.max_sh = j.at("maxSh").get<int>();
                if (j.contains("lodTree"))
                    meta.lod_tree = j.at("lodTree").get<bool>();
                if (j.contains("chunkSize"))
                    meta.chunk_size = j.at("chunkSize").get<uint32_t>();
                meta.all_chunk_bytes = j.at("allChunkBytes").get<uint64_t>();

                for (const auto& chunk_json : j.at("chunks")) {
                    meta.chunks.push_back(RadChunkRange::from_json(chunk_json));
                }

                if (j.contains("splatEncoding")) {
                    meta.splat_encoding = j.at("splatEncoding");
                }
                if (j.contains("shCodeCount")) {
                    meta.sh_code_count = j.at("shCodeCount").get<uint32_t>();
                }
                if (j.contains("comment")) {
                    meta.comment = j.at("comment").get<std::string>();
                }
                return meta;
            }
        };

        // ============================================================================
        // Property Encoding/Decoding
        // ============================================================================

        struct EncodedProperty {
            std::vector<uint8_t> data;
            std::string encoding;
            std::string compression;
            std::optional<float> min_val;
            std::optional<float> max_val;
            std::optional<float> base;
            std::optional<float> scale;
        };

        class PropertyEncoder {
        public:
            static EncodedProperty encode_center(const float* data, size_t dims, size_t count, RadCenterEncoding encoding) {
                EncodedProperty result;

                switch (encoding) {
                case RadCenterEncoding::F32:
                    result.data = encode_f32(data, dims, count);
                    result.encoding = "f32";
                    result.compression = "none";
                    break;

                case RadCenterEncoding::F32LeBytes:
                    result.data = encode_f32_lebytes(data, dims, count);
                    result.encoding = "f32_lebytes";
                    result.compression = "none";
                    break;

                case RadCenterEncoding::F16:
                    result.data = encode_f16(data, dims, count);
                    result.encoding = "f16";
                    result.compression = "none";
                    break;

                case RadCenterEncoding::F16LeBytes:
                    result.data = encode_f16_lebytes(data, dims, count);
                    result.encoding = "f16_lebytes";
                    result.compression = "none";
                    break;

                default:
                    // Match reference default behavior.
                    result.data = encode_f32_lebytes(data, dims, count);
                    result.encoding = "f32_lebytes";
                    result.compression = "none";
                    break;
                }

                return result;
            }

            static EncodedProperty encode_alpha(const float* data, size_t count, RadAlphaEncoding encoding) {
                EncodedProperty result;

                switch (encoding) {
                case RadAlphaEncoding::F32:
                    result.data = encode_f32(data, 1, count);
                    result.encoding = "f32";
                    result.compression = "none";
                    break;

                case RadAlphaEncoding::F16:
                    result.data = encode_f16(data, 1, count);
                    result.encoding = "f16";
                    result.compression = "none";
                    break;

                case RadAlphaEncoding::R8: {
                    auto r8_result = encode_r8(data, 1, count, 0.0f, 1.0f);
                    result.data = std::move(r8_result.data);
                    result.min_val = r8_result.min_val;
                    result.max_val = r8_result.max_val;
                }
                    result.encoding = "r8";
                    result.compression = "none";
                    break;

                default: {
                    float max_alpha = 0.0f;
                    for (size_t i = 0; i < count; ++i)
                        max_alpha = std::max(max_alpha, data[i]);
                    if (max_alpha > 1.0f) {
                        result.data = encode_f16(data, 1, count);
                        result.encoding = "f16";
                    } else {
                        auto r8_result = encode_r8(data, 1, count, 0.0f, 1.0f);
                        result.data = std::move(r8_result.data);
                        result.min_val = r8_result.min_val;
                        result.max_val = r8_result.max_val;
                        result.encoding = "r8";
                    }
                }
                    result.compression = "none";
                    break;
                }

                return result;
            }

            static EncodedProperty encode_rgb(const float* data, size_t dims, size_t count, RadRgbEncoding encoding) {
                EncodedProperty result;

                switch (encoding) {
                case RadRgbEncoding::F32:
                    result.data = encode_f32(data, dims, count);
                    result.encoding = "f32";
                    result.compression = "none";
                    break;

                case RadRgbEncoding::F16:
                    result.data = encode_f16(data, dims, count);
                    result.encoding = "f16";
                    result.compression = "none";
                    break;

                case RadRgbEncoding::R8: {
                    auto r8_result = encode_r8(data, dims, count);
                    result.data = std::move(r8_result.data);
                    result.min_val = r8_result.min_val;
                    result.max_val = r8_result.max_val;
                }
                    result.encoding = "r8";
                    result.compression = "none";
                    break;

                case RadRgbEncoding::R8Delta: {
                    auto r8d_result = encode_r8_delta(data, dims, count);
                    result.data = std::move(r8d_result.data);
                    result.min_val = r8d_result.min_val;
                    result.max_val = r8d_result.max_val;
                }
                    result.encoding = "r8_delta";
                    result.compression = "none";
                    break;

                default: {
                    auto r8d_result = encode_r8_delta(data, dims, count);
                    result.data = std::move(r8d_result.data);
                    result.min_val = r8d_result.min_val;
                    result.max_val = r8d_result.max_val;
                }
                    result.encoding = "r8_delta";
                    result.compression = "none";
                    break;
                }

                return result;
            }

            static EncodedProperty encode_scales(const float* data, size_t dims, size_t count, RadScalesEncoding encoding) {
                EncodedProperty result;

                switch (encoding) {
                case RadScalesEncoding::F32:
                    result.data = encode_f32(data, dims, count);
                    result.encoding = "f32";
                    result.compression = "none";
                    break;

                case RadScalesEncoding::Ln0R8: {
                    auto ln_result = encode_ln_0r8(data, dims, count);
                    result.data = std::move(ln_result.data);
                    result.min_val = ln_result.min_val;
                    result.max_val = ln_result.max_val;
                }
                    result.encoding = "ln_0r8";
                    result.compression = "none";
                    break;

                case RadScalesEncoding::LnF16:
                    result.data = encode_ln_f16(data, dims, count);
                    result.encoding = "ln_f16";
                    result.compression = "none";
                    break;

                default:
                    result.data = encode_ln_f16(data, dims, count);
                    result.encoding = "ln_f16";
                    result.compression = "none";
                    break;
                }

                return result;
            }

            static EncodedProperty encode_orientation(const float* data, size_t count, RadOrientationEncoding encoding) {
                EncodedProperty result;
                std::vector<float> xyz(count * 3);
                for (size_t i = 0; i < count; ++i) {
                    xyz[i * 3 + 0] = data[i * 4 + 0];
                    xyz[i * 3 + 1] = data[i * 4 + 1];
                    xyz[i * 3 + 2] = data[i * 4 + 2];
                }

                switch (encoding) {
                case RadOrientationEncoding::F32:
                    result.data = encode_f32(xyz.data(), 3, count);
                    result.encoding = "f32";
                    result.compression = "none";
                    break;

                case RadOrientationEncoding::F16:
                    result.data = encode_f16(xyz.data(), 3, count);
                    result.encoding = "f16";
                    result.compression = "none";
                    break;

                case RadOrientationEncoding::Oct88R8:
                    result.data = encode_quat_oct88r8(data, count);
                    result.encoding = "oct88r8";
                    result.compression = "none";
                    break;

                default:
                    // Auto-detect: use oct88r8 for compact storage
                    result.data = encode_quat_oct88r8(data, count);
                    result.encoding = "oct88r8";
                    result.compression = "none";
                    break;
                }

                return result;
            }

            static EncodedProperty encode_sh(const float* data, size_t dims, size_t count, RadShEncoding encoding) {
                EncodedProperty result;

                switch (encoding) {
                case RadShEncoding::F32:
                    result.data = encode_f32(data, dims, count);
                    result.encoding = "f32";
                    result.compression = "none";
                    break;

                case RadShEncoding::F16:
                    result.data = encode_f16(data, dims, count);
                    result.encoding = "f16";
                    result.compression = "none";
                    break;

                case RadShEncoding::S8: {
                    auto s8_result = encode_s8(data, dims, count);
                    result.data.assign(reinterpret_cast<const uint8_t*>(s8_result.data.data()),
                                       reinterpret_cast<const uint8_t*>(s8_result.data.data() + s8_result.data.size()));
                    result.min_val = -s8_result.max_abs;
                    result.max_val = s8_result.max_abs;
                }
                    result.encoding = "s8";
                    result.compression = "none";
                    break;

                case RadShEncoding::S8Delta: {
                    auto s8d_result = encode_s8_delta(data, dims, count);
                    result.data.assign(reinterpret_cast<const uint8_t*>(s8d_result.data.data()),
                                       reinterpret_cast<const uint8_t*>(s8d_result.data.data() + s8d_result.data.size()));
                    result.min_val = -s8d_result.max_abs;
                    result.max_val = s8d_result.max_abs;
                }
                    result.encoding = "s8_delta";
                    result.compression = "none";
                    break;

                default: {
                    auto s8_result = encode_s8(data, dims, count);
                    result.data.assign(reinterpret_cast<const uint8_t*>(s8_result.data.data()),
                                       reinterpret_cast<const uint8_t*>(s8_result.data.data() + s8_result.data.size()));
                    result.min_val = -s8_result.max_abs;
                    result.max_val = s8_result.max_abs;
                }
                    result.encoding = "s8";
                    result.compression = "none";
                    break;
                }

                return result;
            }
        };

        class PropertyDecoder {
        public:
            static void decode_center(const uint8_t* data, float* output, size_t dims, size_t count,
                                      const std::string& encoding) {
                if (encoding == "f32") {
                    decode_f32(data, output, dims, count);
                } else if (encoding == "f32_lebytes") {
                    decode_f32_lebytes(data, output, dims, count);
                } else if (encoding == "f16") {
                    decode_f16(data, output, dims, count);
                } else if (encoding == "f16_lebytes") {
                    decode_f16_lebytes(data, output, dims, count);
                } else {
                    throw std::runtime_error("Unknown center encoding: " + encoding);
                }
            }

            static void decode_alpha(const uint8_t* data, float* output, size_t count,
                                     const std::string& encoding,
                                     float min_val, float max_val) {
                if (encoding == "f32") {
                    decode_f32(data, output, 1, count);
                } else if (encoding == "f16") {
                    decode_f16(data, output, 1, count);
                } else if (encoding == "r8") {
                    decode_r8(data, output, 1, count, min_val, max_val);
                } else {
                    throw std::runtime_error("Unknown alpha encoding: " + encoding);
                }
            }

            static void decode_rgb(const uint8_t* data, float* output, size_t dims, size_t count,
                                   const std::string& encoding,
                                   float min_val, float max_val,
                                   float, float) {
                if (encoding == "f32") {
                    decode_f32(data, output, dims, count);
                } else if (encoding == "f16") {
                    decode_f16(data, output, dims, count);
                } else if (encoding == "r8") {
                    decode_r8(data, output, dims, count, min_val, max_val);
                } else if (encoding == "r8_delta") {
                    decode_r8_delta(data, output, dims, count, min_val, max_val);
                } else {
                    throw std::runtime_error("Unknown RGB encoding: " + encoding);
                }
            }

            static void decode_scales(const uint8_t* data, float* output, size_t dims, size_t count,
                                      const std::string& encoding,
                                      float min_val, float max_val) {
                if (encoding == "f32") {
                    decode_f32(data, output, dims, count);
                } else if (encoding == "ln_0r8") {
                    decode_ln_0r8(data, output, dims, count, min_val, max_val);
                } else if (encoding == "ln_f16") {
                    decode_ln_f16(data, output, dims, count);
                } else {
                    throw std::runtime_error("Unknown scales encoding: " + encoding);
                }
            }

            static void decode_orientation(const uint8_t* data, float* output, size_t count,
                                           const std::string& encoding) {
                if (encoding == "f32") {
                    std::vector<float> xyz(count * 3);
                    decode_f32(data, xyz.data(), 3, count);
                    for (size_t i = 0; i < count; ++i) {
                        const float x = xyz[i * 3 + 0];
                        const float y = xyz[i * 3 + 1];
                        const float z = xyz[i * 3 + 2];
                        const float w = std::sqrt(std::max(0.0f, 1.0f - x * x - y * y - z * z));
                        output[i * 4 + 0] = x;
                        output[i * 4 + 1] = y;
                        output[i * 4 + 2] = z;
                        output[i * 4 + 3] = w;
                    }
                } else if (encoding == "f16") {
                    std::vector<float> xyz(count * 3);
                    decode_f16(data, xyz.data(), 3, count);
                    for (size_t i = 0; i < count; ++i) {
                        const float x = xyz[i * 3 + 0];
                        const float y = xyz[i * 3 + 1];
                        const float z = xyz[i * 3 + 2];
                        const float w = std::sqrt(std::max(0.0f, 1.0f - x * x - y * y - z * z));
                        output[i * 4 + 0] = x;
                        output[i * 4 + 1] = y;
                        output[i * 4 + 2] = z;
                        output[i * 4 + 3] = w;
                    }
                } else if (encoding == "oct88r8") {
                    decode_quat_oct88r8(data, output, count);
                } else {
                    throw std::runtime_error("Unknown orientation encoding: " + encoding);
                }
            }

            static void decode_sh(const uint8_t* data, float* output, size_t dims, size_t count,
                                  const std::string& encoding,
                                  float min_val, float max_val,
                                  float, float scale) {
                const float sh_max = std::max({std::abs(min_val), std::abs(max_val), std::abs(scale), 1e-6f});
                if (encoding == "f32") {
                    decode_f32(data, output, dims, count);
                } else if (encoding == "f16") {
                    decode_f16(data, output, dims, count);
                } else if (encoding == "s8") {
                    decode_s8(reinterpret_cast<const int8_t*>(data), output, dims, count, sh_max);
                } else if (encoding == "s8_delta") {
                    decode_s8_delta(reinterpret_cast<const int8_t*>(data), output, dims, count, sh_max);
                } else {
                    throw std::runtime_error("Unknown SH encoding: " + encoding);
                }
            }
        };

        // ============================================================================
        // RAD Encoder
        // ============================================================================

        class RadEncoder {
        public:
            explicit RadEncoder(int compression_level = GZ_LEVEL,
                                const std::vector<float>& lod_ratios = {},
                                bool flip_y = false,
                                ExportProgressCallback progress_callback = nullptr)
                : compression_level_(compression_level),
                  lod_ratios_(lod_ratios),
                  flip_y_(flip_y),
                  progress_callback_(std::move(progress_callback)) {}

            std::vector<uint8_t> encode(const SplatData& splat_data) {
                // 0.0: Preparing data
                if (!report_progress(0.0f, "Preparing data...")) {
                    throw std::runtime_error("CANCELLED");
                }

                std::optional<SplatData> visible_splat_data;
                const SplatData* export_source = &splat_data;
                if (splat_data.has_deleted_mask() && splat_data.deleted().count_nonzero() > 0) {
                    const Tensor keep_mask = splat_data.deleted().logical_not();
                    auto extracted = lfs::core::extract_by_mask(splat_data, keep_mask);
                    if (extracted.size() > 0) {
                        visible_splat_data = std::move(extracted);
                        export_source = &visible_splat_data.value();
                    }
                }

                // 0.1: Packing splat data
                if (!report_progress(0.1f, "Packing splat data...")) {
                    throw std::runtime_error("CANCELLED");
                }

                PackedSplatData packed = pack_splat_data(*export_source, flip_y_);

                // 0.2: Data packed
                if (!report_progress(0.2f, "Data packed")) {
                    throw std::runtime_error("CANCELLED");
                }

                // 0.3: Building LOD (always report, even if using defaults)
                if (!report_progress(0.3f, "Building LOD...")) {
                    throw std::runtime_error("CANCELLED");
                }

                if (auto lod_packed = build_lod(*export_source, lod_ratios_)) {
                    packed = std::move(*lod_packed);
                }

                // 0.4: Preparing chunks
                if (!report_progress(0.4f, "Preparing chunks...")) {
                    throw std::runtime_error("CANCELLED");
                }

                if (packed.count > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
                    throw std::runtime_error("RAD export exceeds maximum supported splat count");
                }

                const uint32_t num_splats = static_cast<uint32_t>(packed.count);
                const int sh_degree = packed.sh_degree;
                const int sh_coeffs = packed.sh_coeffs;
                const bool lod_tree = packed.lod_tree;

                const uint32_t num_chunks = (num_splats + CHUNK_SIZE - 1) / CHUNK_SIZE;

                // Build metadata
                RadMeta meta;
                meta.count = num_splats;
                meta.max_sh = sh_degree;
                meta.lod_tree = lod_tree ? std::optional<bool>(true) : std::nullopt;
                meta.chunk_size = CHUNK_SIZE;
                if (lod_tree) {
                    meta.splat_encoding = nlohmann::json{{"lodOpacity", true}};
                }

                // Encode chunks in parallel with dynamic scheduling
                std::vector<std::vector<uint8_t>> chunk_payloads(num_chunks);
                std::vector<RadChunkRange> chunk_ranges(num_chunks);
                std::atomic<uint32_t> completed_chunks{0};

                // Report initial progress
                if (!report_progress(0.5f, "Encoding chunks...")) {
                    throw std::runtime_error("CANCELLED");
                }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
                for (int32_t chunk_idx = 0; chunk_idx < static_cast<int32_t>(num_chunks); ++chunk_idx) {
                    const uint32_t base = static_cast<uint32_t>(chunk_idx) * CHUNK_SIZE;
                    const uint32_t count = std::min(CHUNK_SIZE, num_splats - base);

                    // Thread-local progress callback (only report every 10% to reduce contention)
                    auto chunk_progress_cb = [&](float /*progress*/) -> bool {
                        // Check for cancellation periodically
                        uint32_t completed = completed_chunks.load(std::memory_order_relaxed);
                        if (completed % 16 == 0) {
                            // Approximate overall progress
                            float overall = 0.5f + (0.4f * static_cast<float>(completed) / static_cast<float>(num_chunks));
                            return report_progress(overall, "Encoding chunks...");
                        }
                        return true;
                    };

                    auto chunk_result = encode_chunk(
                        base, count, sh_degree, sh_coeffs,
                        packed.means.data(),
                        packed.opacity.data(),
                        packed.sh0.data(),
                        packed.scales.data(),
                        packed.rotation.data(),
                        packed.shN.empty() ? nullptr : packed.shN.data(),
                        lod_tree ? packed.child_count.data() : nullptr,
                        lod_tree ? packed.child_start.data() : nullptr,
                        lod_tree,
                        chunk_progress_cb);

                    // Store results (each thread writes to its own index)
                    chunk_ranges[chunk_idx].base = base;
                    chunk_ranges[chunk_idx].count = count;
                    chunk_ranges[chunk_idx].bytes = chunk_result.second.size();
                    chunk_payloads[chunk_idx] = std::move(chunk_result.second);

                    // Atomically increment completed count for progress tracking
                    completed_chunks.fetch_add(1, std::memory_order_relaxed);
                }

                // Build metadata in order (sequential - must preserve chunk order)
                uint64_t current_chunk_offset = 0;
                for (uint32_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
                    chunk_ranges[chunk_idx].offset = current_chunk_offset;
                    meta.chunks.push_back(chunk_ranges[chunk_idx]);
                    current_chunk_offset += chunk_ranges[chunk_idx].bytes;
                }

                // Calculate total chunk bytes
                for (const auto& payload : chunk_payloads) {
                    meta.all_chunk_bytes += payload.size();
                }

                // Serialize metadata to JSON
                std::string meta_json = meta.to_json().dump();

                const size_t meta_size = meta_json.size();
                const size_t meta_padded_size = pad8(meta_size);

                // Build header: RAD_MAGIC (4 bytes) + metadata_length (4 bytes) = 8 bytes total
                std::vector<uint8_t> header(8);
                encode_u32(&header[0], RAD_MAGIC);
                encode_u32(&header[4], static_cast<uint32_t>(meta_size));

                // 0.9: Assembling file data
                if (!report_progress(0.9f, "Assembling RAD data...")) {
                    throw std::runtime_error("CANCELLED");
                }

                // Combine all data
                std::vector<uint8_t> result;
                result.reserve(header.size() + meta_padded_size + meta.all_chunk_bytes);

                result.insert(result.end(), header.begin(), header.end());
                result.insert(result.end(), meta_json.begin(), meta_json.end());
                if (meta_padded_size > meta_size) {
                    result.insert(result.end(), meta_padded_size - meta_size, 0);
                }

                for (const auto& payload : chunk_payloads) {
                    result.insert(result.end(), payload.begin(), payload.end());
                }

                // 1.0: Encoding complete
                if (!report_progress(1.0f, "RAD data prepared")) {
                    throw std::runtime_error("CANCELLED");
                }

                return result;
            }

        private:
            struct PackedSplatData {
                size_t count = 0;
                int sh_degree = 0;
                int sh_coeffs = 0;
                bool lod_tree = false;
                std::vector<float> means;
                std::vector<float> opacity;
                std::vector<float> sh0;
                std::vector<float> scales;
                std::vector<float> rotation;
                std::vector<float> shN;
                std::vector<uint16_t> child_count;
                std::vector<uint32_t> child_start;
            };

            struct PointCloudAdaptor {
                const float* points = nullptr;
                size_t num_points = 0;

                [[nodiscard]] inline size_t kdtree_get_point_count() const { return num_points; }
                [[nodiscard]] inline float kdtree_get_pt(const size_t idx, const size_t dim) const {
                    return points[idx * 3 + dim];
                }
                template <class BBOX>
                bool kdtree_get_bbox(BBOX&) const { return false; }
            };

            using LodKDTree = nanoflann::KDTreeSingleIndexAdaptor<
                nanoflann::L2_Simple_Adaptor<float, PointCloudAdaptor>,
                PointCloudAdaptor,
                3>;

            static void reorder_rows(std::vector<float>& rows, size_t row_width, const std::vector<size_t>& order) {
                if (rows.empty()) {
                    return;
                }
                if (row_width == 0 || rows.size() % row_width != 0) {
                    rows.clear();
                    return;
                }

                const size_t old_count = rows.size() / row_width;
                if (order.size() != old_count) {
                    return;
                }

                std::vector<float> reordered(order.size() * row_width);
                for (size_t new_idx = 0; new_idx < order.size(); ++new_idx) {
                    const size_t old_idx = order[new_idx];
                    if (old_idx >= old_count) {
                        continue;
                    }
                    std::memcpy(
                        reordered.data() + new_idx * row_width,
                        rows.data() + old_idx * row_width,
                        row_width * sizeof(float));
                }
                rows = std::move(reordered);
            }

            static void reorder_level(PackedSplatData& level, const std::vector<size_t>& order) {
                reorder_rows(level.means, 3, order);
                reorder_rows(level.opacity, 1, order);
                reorder_rows(level.sh0, 3, order);
                reorder_rows(level.scales, 3, order);
                reorder_rows(level.rotation, 4, order);
                if (level.sh_coeffs > 0 && !level.shN.empty()) {
                    reorder_rows(level.shN, static_cast<size_t>(level.sh_coeffs) * 3, order);
                }
                level.count = order.size();
            }

            static uint64_t expand_morton_21(uint32_t value) {
                uint64_t x = static_cast<uint64_t>(value) & 0x1fffffULL;
                x = (x | (x << 32)) & 0x1f00000000ffffULL;
                x = (x | (x << 16)) & 0x1f0000ff0000ffULL;
                x = (x | (x << 8)) & 0x100f00f00f00f00fULL;
                x = (x | (x << 4)) & 0x10c30c30c30c30c3ULL;
                x = (x | (x << 2)) & 0x1249249249249249ULL;
                return x;
            }

            static uint64_t morton_code_3d(uint32_t x, uint32_t y, uint32_t z) {
                return expand_morton_21(x) |
                       (expand_morton_21(y) << 1) |
                       (expand_morton_21(z) << 2);
            }

            static void sort_level_spatially(PackedSplatData& level) {
                if (level.count <= 1 || level.means.size() < level.count * 3) {
                    return;
                }

                std::array<float, 3> min_corner{
                    std::numeric_limits<float>::infinity(),
                    std::numeric_limits<float>::infinity(),
                    std::numeric_limits<float>::infinity()};
                std::array<float, 3> max_corner{
                    -std::numeric_limits<float>::infinity(),
                    -std::numeric_limits<float>::infinity(),
                    -std::numeric_limits<float>::infinity()};

                for (size_t i = 0; i < level.count; ++i) {
                    for (size_t d = 0; d < 3; ++d) {
                        const float value = level.means[i * 3 + d];
                        min_corner[d] = std::min(min_corner[d], value);
                        max_corner[d] = std::max(max_corner[d], value);
                    }
                }

                std::array<float, 3> inv_extent{};
                for (size_t d = 0; d < 3; ++d) {
                    const float extent = max_corner[d] - min_corner[d];
                    inv_extent[d] = extent > 1.0e-20f ? 1.0f / extent : 0.0f;
                }

                std::vector<std::pair<uint64_t, size_t>> keyed_order;
                keyed_order.reserve(level.count);
                for (size_t i = 0; i < level.count; ++i) {
                    std::array<uint32_t, 3> q{};
                    for (size_t d = 0; d < 3; ++d) {
                        const float normalized = std::clamp(
                            (level.means[i * 3 + d] - min_corner[d]) * inv_extent[d],
                            0.0f,
                            1.0f);
                        q[d] = static_cast<uint32_t>(std::round(normalized * 2097151.0f));
                    }
                    keyed_order.emplace_back(morton_code_3d(q[0], q[1], q[2]), i);
                }

                std::sort(keyed_order.begin(), keyed_order.end(), [](const auto& a, const auto& b) {
                    if (a.first != b.first) {
                        return a.first < b.first;
                    }
                    return a.second < b.second;
                });

                std::vector<size_t> order;
                order.reserve(level.count);
                for (const auto& [_, index] : keyed_order) {
                    order.push_back(index);
                }
                reorder_level(level, order);
            }

            static std::vector<size_t> assign_to_nearest_parent(const std::vector<float>& child_means,
                                                                size_t child_count,
                                                                const std::vector<float>& parent_means,
                                                                size_t parent_count) {
                std::vector<size_t> assignments(child_count, 0);
                if (child_count == 0 || parent_count == 0) {
                    return assignments;
                }
                if (parent_count == 1) {
                    return assignments;
                }

                PointCloudAdaptor adaptor{parent_means.data(), parent_count};
                LodKDTree index(3, adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(10));
                index.buildIndex();

                for (size_t i = 0; i < child_count; ++i) {
                    size_t nearest = 0;
                    float out_dist_sqr = 0.0f;
                    nanoflann::KNNResultSet<float> result_set(1);
                    result_set.init(&nearest, &out_dist_sqr);
                    index.findNeighbors(
                        result_set,
                        child_means.data() + i * 3,
                        nanoflann::SearchParameters(10));
                    assignments[i] = nearest;
                }

                return assignments;
            }

            static std::vector<size_t> build_grouped_child_order(const std::vector<size_t>& child_to_parent,
                                                                 size_t parent_count,
                                                                 uint32_t child_global_offset,
                                                                 std::vector<uint16_t>& out_child_count,
                                                                 std::vector<uint32_t>& out_child_start) {
                out_child_count.assign(parent_count, 0);
                out_child_start.assign(parent_count, 0);

                std::vector<std::vector<size_t>> groups(parent_count);
                for (size_t child = 0; child < child_to_parent.size(); ++child) {
                    const size_t parent = std::min(child_to_parent[child], parent_count - 1);
                    groups[parent].push_back(child);
                }

                std::vector<size_t> order;
                order.reserve(child_to_parent.size());
                for (size_t parent = 0; parent < parent_count; ++parent) {
                    const auto& children = groups[parent];
                    if (children.empty()) {
                        continue;
                    }

                    if (children.size() > static_cast<size_t>(std::numeric_limits<uint16_t>::max())) {
                        return {};
                    }

                    const uint64_t start = static_cast<uint64_t>(child_global_offset) + order.size();
                    if (start > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
                        return {};
                    }

                    out_child_count[parent] = static_cast<uint16_t>(children.size());
                    out_child_start[parent] = static_cast<uint32_t>(start);
                    order.insert(order.end(), children.begin(), children.end());
                }

                return order;
            }

            static PackedSplatData pack_splat_data(const SplatData& splat_data, bool flip_y = false) {
                PackedSplatData packed;
                packed.count = static_cast<size_t>(splat_data.size());
                packed.sh_degree = std::clamp(splat_data.get_max_sh_degree(), 0, 3);
                packed.sh_coeffs = packed.sh_degree > 0 ? SH_COEFFS_FOR_DEGREE[packed.sh_degree] : 0;

                auto copy_f32 = [](const Tensor& tensor, size_t expected_values) {
                    std::vector<float> result;
                    if (expected_values == 0 || !tensor.is_valid()) {
                        return result;
                    }
                    const auto cpu = tensor.contiguous().to(Device::CPU);
                    const auto* ptr = static_cast<const float*>(cpu.data_ptr());
                    result.assign(ptr, ptr + expected_values);
                    return result;
                };

                // Spark RAD stores render-space values, not optimizer-domain tensors.
                packed.means = copy_f32(splat_data.get_means(), packed.count * 3);

                // Apply Y-flip if requested (negate Y coordinate of positions)
                if (flip_y) {
                    for (size_t i = 0; i < packed.count; ++i) {
                        packed.means[i * 3 + 1] = -packed.means[i * 3 + 1];
                    }
                }

                packed.opacity = copy_f32(splat_data.get_opacity(), packed.count);
                packed.sh0 = copy_f32(splat_data.sh0_raw(), packed.count * 3);
                for (float& value : packed.sh0) {
                    value = 0.5f + SH_C0 * value;
                }
                packed.scales = copy_f32(splat_data.get_scaling(), packed.count * 3);
                packed.rotation = copy_f32(splat_data.get_rotation(), packed.count * 4);
                if (packed.sh_coeffs > 0) {
                    packed.shN = copy_f32(splat_data.shN_raw(), packed.count * static_cast<size_t>(packed.sh_coeffs) * 3);
                }

                return packed;
            }

            // Build LOD with custom ratios. Ratios should be in ascending order (e.g., {0.2, 0.5, 1.0})
            static std::optional<PackedSplatData> build_lod(const SplatData& source,
                                                            const std::vector<float>& custom_ratios) {
                if (source.size() == 0) {
                    return std::nullopt;
                }

                // Use default ratios if not provided
                std::vector<float> ratios = custom_ratios;
                if (ratios.empty()) {
                    ratios = {0.2f, 0.5f, 1.0f}; // Default: 20%, 50%, 100%
                }

                // Ensure ratios are sorted and within valid range
                std::sort(ratios.begin(), ratios.end());
                for (float& r : ratios) {
                    r = std::clamp(r, 0.01f, 1.0f);
                }

                // Remove duplicates
                ratios.erase(std::unique(ratios.begin(), ratios.end()), ratios.end());

                if (ratios.empty()) {
                    ratios = {1.0f}; // At least 100%
                }

                // Ensure 100% (full detail) is included
                if (ratios.back() < 1.0f) {
                    ratios.push_back(1.0f);
                }

                constexpr size_t kUpperLodFanout = 64;

                // Build all LOD levels
                std::vector<PackedSplatData> levels;
                std::vector<std::vector<uint16_t>> child_counts;
                std::vector<std::vector<uint32_t>> child_starts;

                for (size_t i = 0; i < ratios.size(); ++i) {
                    float ratio = ratios[i];
                    PackedSplatData level;

                    if (ratio >= 1.0f) {
                        // Full detail - use source directly
                        level = pack_splat_data(source);
                    } else {
                        // Simplified level
                        lfs::core::SplatSimplifyOptions options;
                        options.ratio = ratio;
                        auto result = lfs::core::simplify_splats(source, options, {});
                        if (!result || !result.value()) {
                            LOG_WARN("RAD export: failed to build LOD at ratio {}, falling back to non-LOD", ratio);
                            return std::nullopt;
                        }
                        level = pack_splat_data(*result.value());
                    }

                    if (level.count == 0) {
                        LOG_WARN("RAD export: LOD at ratio {} produced empty level, falling back to non-LOD", ratio);
                        return std::nullopt;
                    }

                    levels.push_back(std::move(level));
                    child_counts.emplace_back();
                    child_starts.emplace_back();
                }

                // Verify all levels have same SH degree
                int sh_degree = levels.back().sh_degree;
                for (const auto& level : levels) {
                    if (level.sh_degree != sh_degree) {
                        LOG_WARN("RAD export: LOD levels disagree on SH degree, falling back to non-LOD");
                        return std::nullopt;
                    }
                }

                sort_level_spatially(levels.front());

                // Build parent-child relationships and reorder levels
                // Process from coarsest to finest (reverse order of ratios)
                for (size_t i = 0; i < levels.size() - 1; ++i) {
                    size_t coarse_idx = i;
                    size_t fine_idx = i + 1;

                    const auto& coarse = levels[coarse_idx];
                    auto& fine = levels[fine_idx];

                    // Calculate global offset for this fine level
                    uint64_t global_offset = 0;
                    for (size_t j = 0; j <= coarse_idx; ++j) {
                        global_offset += levels[j].count;
                    }

                    if (global_offset > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
                        LOG_WARN("RAD export: too many splats for LOD index range, falling back to non-LOD");
                        return std::nullopt;
                    }

                    // Assign children
                    const auto fine_to_coarse = assign_to_nearest_parent(
                        fine.means, fine.count, coarse.means, coarse.count);

                    auto fine_order = build_grouped_child_order(
                        fine_to_coarse,
                        coarse.count,
                        static_cast<uint32_t>(global_offset),
                        child_counts[coarse_idx],
                        child_starts[coarse_idx]);

                    if (fine_order.size() != fine.count) {
                        LOG_WARN("RAD export: failed to order LOD children, falling back to non-LOD");
                        return std::nullopt;
                    }

                    reorder_level(fine, fine_order);
                }

                // Aggregate node helper
                struct AggregateNode {
                    std::array<float, 3> center{0.0f, 0.0f, 0.0f};
                    float opacity = 1.0f;
                    std::array<float, 3> sh0{0.0f, 0.0f, 0.0f};
                    std::array<float, 3> scale{1.0e-3f, 1.0e-3f, 1.0e-3f};
                    std::array<float, 4> rotation{1.0f, 0.0f, 0.0f, 0.0f};
                    std::vector<float> shN;
                };

                auto aggregate_range = [&](const PackedSplatData& level, size_t start, size_t count) -> AggregateNode {
                    AggregateNode node;
                    if (count == 0) {
                        return node;
                    }

                    std::array<float, 3> min_corner{
                        std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::infinity()};
                    std::array<float, 3> max_corner{
                        -std::numeric_limits<float>::infinity(),
                        -std::numeric_limits<float>::infinity(),
                        -std::numeric_limits<float>::infinity()};
                    std::array<double, 3> sh0_sum{0.0, 0.0, 0.0};
                    double opacity_sum = 0.0;

                    const size_t shn_row_width = static_cast<size_t>(std::max(level.sh_coeffs, 0)) * 3;
                    if (shn_row_width > 0) {
                        node.shN.assign(shn_row_width, 0.0f);
                    }

                    for (size_t i = 0; i < count; ++i) {
                        const size_t idx = start + i;
                        for (size_t d = 0; d < 3; ++d) {
                            const float c = level.means[idx * 3 + d];
                            const float s = std::max(std::abs(level.scales[idx * 3 + d]), 1.0e-6f);
                            min_corner[d] = std::min(min_corner[d], c - s);
                            max_corner[d] = std::max(max_corner[d], c + s);
                            sh0_sum[d] += static_cast<double>(level.sh0[idx * 3 + d]);
                        }
                        opacity_sum += static_cast<double>(level.opacity[idx]);

                        if (shn_row_width > 0 && !level.shN.empty()) {
                            const size_t base = idx * shn_row_width;
                            for (size_t k = 0; k < shn_row_width; ++k) {
                                node.shN[k] += level.shN[base + k];
                            }
                        }
                    }

                    for (size_t d = 0; d < 3; ++d) {
                        node.center[d] = 0.5f * (min_corner[d] + max_corner[d]);
                        node.scale[d] = std::max(0.5f * (max_corner[d] - min_corner[d]), 1.0e-6f);
                        node.sh0[d] = static_cast<float>(sh0_sum[d] / static_cast<double>(count));
                    }
                    node.opacity = std::clamp(static_cast<float>(opacity_sum / static_cast<double>(count)), 0.0f, 1.0f);

                    if (!node.shN.empty()) {
                        const float inv_count = 1.0f / static_cast<float>(count);
                        for (float& v : node.shN) {
                            v *= inv_count;
                        }
                    }

                    return node;
                };

                auto append_node_to_level = [&](PackedSplatData& dst, const AggregateNode& node) {
                    dst.means.insert(dst.means.end(), node.center.begin(), node.center.end());
                    dst.opacity.push_back(node.opacity);
                    dst.sh0.insert(dst.sh0.end(), node.sh0.begin(), node.sh0.end());
                    dst.scales.insert(dst.scales.end(), node.scale.begin(), node.scale.end());
                    dst.rotation.insert(dst.rotation.end(), node.rotation.begin(), node.rotation.end());
                    if (dst.sh_coeffs > 0) {
                        if (!node.shN.empty()) {
                            dst.shN.insert(dst.shN.end(), node.shN.begin(), node.shN.end());
                        } else {
                            dst.shN.insert(dst.shN.end(), static_cast<size_t>(dst.sh_coeffs) * 3, 0.0f);
                        }
                    }
                };

                auto make_upper_level = [&](const PackedSplatData& child_level) {
                    PackedSplatData parent_level;
                    parent_level.count = (child_level.count + kUpperLodFanout - 1) / kUpperLodFanout;
                    parent_level.sh_degree = sh_degree;
                    parent_level.sh_coeffs = child_level.sh_coeffs;
                    parent_level.lod_tree = true;
                    parent_level.means.reserve(parent_level.count * 3);
                    parent_level.opacity.reserve(parent_level.count);
                    parent_level.sh0.reserve(parent_level.count * 3);
                    parent_level.scales.reserve(parent_level.count * 3);
                    parent_level.rotation.reserve(parent_level.count * 4);
                    if (parent_level.sh_coeffs > 0) {
                        parent_level.shN.reserve(parent_level.count * static_cast<size_t>(parent_level.sh_coeffs) * 3);
                    }

                    for (size_t g = 0; g < parent_level.count; ++g) {
                        const size_t group_start = g * kUpperLodFanout;
                        const size_t group_count = std::min(kUpperLodFanout, child_level.count - group_start);
                        append_node_to_level(parent_level, aggregate_range(child_level, group_start, group_count));
                    }
                    return parent_level;
                };

                // Build a bounded-fanout hierarchy above the coarsest simplified level.
                // The previous implementation connected the root to arbitrary 65k-wide
                // row groups, which creates scene-sized blobs and weak paging boundaries.
                std::vector<PackedSplatData> upper_levels;
                const PackedSplatData* child_level = &levels.front();
                while (true) {
                    upper_levels.push_back(make_upper_level(*child_level));
                    if (upper_levels.back().count == 1) {
                        break;
                    }
                    child_level = &upper_levels.back();
                }

                // Build final packed data with tree structure. Upper levels are stored
                // root-first, followed by original LOD levels from coarsest to finest.
                const size_t finest_idx = levels.size() - 1;
                size_t current_base = 0;
                std::vector<size_t> upper_bases(upper_levels.size());
                for (size_t out = 0; out < upper_levels.size(); ++out) {
                    const size_t upper_idx = upper_levels.size() - 1 - out;
                    upper_bases[upper_idx] = current_base;
                    current_base += upper_levels[upper_idx].count;
                }

                std::vector<size_t> level_bases(levels.size());
                for (size_t i = 0; i < levels.size(); ++i) {
                    level_bases[i] = current_base;
                    current_base += levels[i].count;
                }
                const size_t total_count = current_base;

                if (total_count > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
                    LOG_WARN("RAD export: packed LOD exceeds u32 index range, falling back to non-LOD");
                    return std::nullopt;
                }

                // Build final packed data
                PackedSplatData packed;
                packed.count = total_count;
                packed.sh_degree = sh_degree;
                packed.sh_coeffs = levels[finest_idx].sh_coeffs;
                packed.lod_tree = true;
                packed.child_count.assign(packed.count, 0);
                packed.child_start.assign(packed.count, 0);

                auto append_rows = [](std::vector<float>& dst, const std::vector<float>& src) {
                    dst.insert(dst.end(), src.begin(), src.end());
                };

                // Reserve space
                packed.means.reserve(packed.count * 3);
                packed.opacity.reserve(packed.count);
                packed.sh0.reserve(packed.count * 3);
                packed.scales.reserve(packed.count * 3);
                packed.rotation.reserve(packed.count * 4);
                if (packed.sh_coeffs > 0) {
                    packed.shN.reserve(packed.count * static_cast<size_t>(packed.sh_coeffs) * 3);
                }

                // Add upper hierarchy root-first.
                for (size_t out = 0; out < upper_levels.size(); ++out) {
                    const size_t upper_idx = upper_levels.size() - 1 - out;
                    append_rows(packed.means, upper_levels[upper_idx].means);
                    packed.opacity.insert(packed.opacity.end(), upper_levels[upper_idx].opacity.begin(), upper_levels[upper_idx].opacity.end());
                    append_rows(packed.sh0, upper_levels[upper_idx].sh0);
                    append_rows(packed.scales, upper_levels[upper_idx].scales);
                    append_rows(packed.rotation, upper_levels[upper_idx].rotation);
                    if (packed.sh_coeffs > 0) {
                        append_rows(packed.shN, upper_levels[upper_idx].shN);
                    }
                }

                // Add all LOD levels (coarsest to finest)
                for (size_t i = 0; i < levels.size(); ++i) {
                    append_rows(packed.means, levels[i].means);
                    packed.opacity.insert(packed.opacity.end(), levels[i].opacity.begin(), levels[i].opacity.end());
                    append_rows(packed.sh0, levels[i].sh0);
                    append_rows(packed.scales, levels[i].scales);
                    append_rows(packed.rotation, levels[i].rotation);
                    if (packed.sh_coeffs > 0) {
                        append_rows(packed.shN, levels[i].shN);
                    }
                }

                // Set up child links
                for (size_t upper_idx = 0; upper_idx < upper_levels.size(); ++upper_idx) {
                    const bool points_to_coarsest = upper_idx == 0;
                    const size_t child_base = points_to_coarsest ? level_bases[0] : upper_bases[upper_idx - 1];
                    const size_t child_total = points_to_coarsest ? levels[0].count : upper_levels[upper_idx - 1].count;

                    for (size_t j = 0; j < upper_levels[upper_idx].count; ++j) {
                        const size_t group_start = j * kUpperLodFanout;
                        const size_t group_count = std::min(kUpperLodFanout, child_total - group_start);
                        const size_t idx = upper_bases[upper_idx] + j;
                        packed.child_count[idx] = static_cast<uint16_t>(group_count);
                        packed.child_start[idx] = static_cast<uint32_t>(child_base + group_start);
                    }
                }

                // Level-to-level links
                const uint32_t index_shift = static_cast<uint32_t>(level_bases[0]);

                for (size_t i = 0; i < levels.size() - 1; ++i) {
                    size_t coarse_level_idx = i;

                    for (size_t j = 0; j < levels[coarse_level_idx].count; ++j) {
                        const size_t idx = level_bases[coarse_level_idx] + j;
                        packed.child_count[idx] = child_counts[coarse_level_idx][j];
                        if (child_counts[coarse_level_idx][j] > 0) {
                            packed.child_start[idx] = child_starts[coarse_level_idx][j] + index_shift;
                        }
                    }
                }

                return packed;
            }

            // Backward compatibility: use default three-level LOD
            static std::optional<PackedSplatData> build_three_level_lod(const SplatData& source) {
                return build_lod(source, {}); // Empty ratios = use defaults
            }

            std::pair<RadChunkMeta, std::vector<uint8_t>> encode_chunk(
                uint32_t base, uint32_t count, int sh_degree, int sh_coeffs,
                const float* means_ptr,
                const float* opacity_ptr,
                const float* sh0_ptr,
                const float* scales_ptr,
                const float* rotation_ptr,
                const float* shN_ptr,
                const uint16_t* child_count_ptr,
                const uint32_t* child_start_ptr,
                bool lod_tree,
                const std::function<bool(float)>& progress_callback = nullptr) {

                RadChunkMeta chunk_meta;
                chunk_meta.version = 1;
                chunk_meta.base = base;
                chunk_meta.count = count;
                chunk_meta.max_sh = sh_degree;
                chunk_meta.lod_tree = lod_tree;
                if (lod_tree) {
                    chunk_meta.splat_encoding = nlohmann::json{{"lodOpacity", true}};
                }

                std::vector<EncodedProperty> encoded_props;
                encoded_props.reserve(lod_tree ? 12 : 10);

                // Thread-local buffers for temporary data to avoid allocation contention
                thread_local std::vector<float> tl_center_data;
                thread_local std::vector<float> tl_alpha_data;
                thread_local std::vector<float> tl_rgb_data;
                thread_local std::vector<float> tl_scales_data;
                thread_local std::vector<float> tl_quat_data;
                thread_local std::vector<float> tl_sh_data;

                // Encode center (3 components together as single property)
                {
                    tl_center_data.resize(count * 3);
                    for (uint32_t i = 0; i < count; ++i) {
                        tl_center_data[i * 3 + 0] = means_ptr[(base + i) * 3 + 0];
                        tl_center_data[i * 3 + 1] = means_ptr[(base + i) * 3 + 1];
                        tl_center_data[i * 3 + 2] = means_ptr[(base + i) * 3 + 2];
                    }

                    // Encode all 3 components together as "center" property
                    auto encoded = PropertyEncoder::encode_center(tl_center_data.data(), 3, count, RadCenterEncoding::Auto);
                    auto compressed = rad_compress(encoded.data.data(), encoded.data.size(), compression_level_);

                    RadChunkProperty prop;
                    prop.property = PROP_CENTER;
                    prop.encoding = encoded.encoding;
                    prop.compression = "gz";
                    prop.bytes = compressed.size();

                    encoded_props.push_back({std::move(compressed), encoded.encoding, "gz",
                                             encoded.min_val, encoded.max_val, encoded.base, encoded.scale});
                    chunk_meta.properties.push_back(prop);

                    // Report progress after encoding center: 0.1f
                    if (progress_callback && !progress_callback(0.1f)) {
                        throw std::runtime_error("CANCELLED");
                    }
                }

                // Encode alpha
                {
                    tl_alpha_data.resize(count);
                    for (uint32_t i = 0; i < count; ++i) {
                        tl_alpha_data[i] = opacity_ptr[base + i];
                    }

                    auto encoded = PropertyEncoder::encode_alpha(tl_alpha_data.data(), count, RadAlphaEncoding::Auto);
                    auto compressed = rad_compress(encoded.data.data(), encoded.data.size(), compression_level_);

                    RadChunkProperty prop;
                    prop.property = PROP_ALPHA;
                    prop.encoding = encoded.encoding;
                    prop.compression = "gz";
                    prop.bytes = compressed.size();
                    if (encoded.min_val.has_value())
                        prop.min_val = encoded.min_val.value();
                    if (encoded.max_val.has_value())
                        prop.max_val = encoded.max_val.value();

                    encoded_props.push_back({std::move(compressed), encoded.encoding, "gz",
                                             encoded.min_val, encoded.max_val, encoded.base, encoded.scale});
                    chunk_meta.properties.push_back(prop);

                    // Report progress after encoding alpha: 0.2f
                    if (progress_callback && !progress_callback(0.2f)) {
                        throw std::runtime_error("CANCELLED");
                    }
                }

                // Encode RGB (sh0) - all 3 components together as single property
                {
                    tl_rgb_data.resize(count * 3);
                    for (uint32_t i = 0; i < count; ++i) {
                        tl_rgb_data[i * 3 + 0] = sh0_ptr[(base + i) * 3 + 0];
                        tl_rgb_data[i * 3 + 1] = sh0_ptr[(base + i) * 3 + 1];
                        tl_rgb_data[i * 3 + 2] = sh0_ptr[(base + i) * 3 + 2];
                    }

                    // Encode all 3 components together as "rgb" property
                    auto encoded = PropertyEncoder::encode_rgb(tl_rgb_data.data(), 3, count, RadRgbEncoding::Auto);
                    auto compressed = rad_compress(encoded.data.data(), encoded.data.size(), compression_level_);

                    RadChunkProperty prop;
                    prop.property = PROP_RGB;
                    prop.encoding = encoded.encoding;
                    prop.compression = "gz";
                    prop.bytes = compressed.size();
                    if (encoded.min_val.has_value())
                        prop.min_val = encoded.min_val.value();
                    if (encoded.max_val.has_value())
                        prop.max_val = encoded.max_val.value();
                    if (encoded.base.has_value())
                        prop.base = encoded.base.value();
                    if (encoded.scale.has_value())
                        prop.scale = encoded.scale.value();

                    encoded_props.push_back({std::move(compressed), encoded.encoding, "gz",
                                             encoded.min_val, encoded.max_val, encoded.base, encoded.scale});
                    chunk_meta.properties.push_back(prop);

                    // Report progress after encoding RGB: 0.4f
                    if (progress_callback && !progress_callback(0.4f)) {
                        throw std::runtime_error("CANCELLED");
                    }
                }

                // Encode scales - all 3 components together as single property
                {
                    tl_scales_data.resize(count * 3);
                    for (uint32_t i = 0; i < count; ++i) {
                        tl_scales_data[i * 3 + 0] = scales_ptr[(base + i) * 3 + 0];
                        tl_scales_data[i * 3 + 1] = scales_ptr[(base + i) * 3 + 1];
                        tl_scales_data[i * 3 + 2] = scales_ptr[(base + i) * 3 + 2];
                    }

                    // Encode all 3 components together as "scales" property
                    auto encoded = PropertyEncoder::encode_scales(tl_scales_data.data(), 3, count, RadScalesEncoding::Auto);
                    auto compressed = rad_compress(encoded.data.data(), encoded.data.size(), compression_level_);

                    RadChunkProperty prop;
                    prop.property = PROP_SCALES;
                    prop.encoding = encoded.encoding;
                    prop.compression = "gz";
                    prop.bytes = compressed.size();
                    if (encoded.min_val.has_value())
                        prop.min_val = encoded.min_val.value();
                    if (encoded.max_val.has_value())
                        prop.max_val = encoded.max_val.value();
                    if (encoded.scale.has_value())
                        prop.scale = encoded.scale.value();

                    encoded_props.push_back({std::move(compressed), encoded.encoding, "gz",
                                             encoded.min_val, encoded.max_val, encoded.base, encoded.scale});
                    chunk_meta.properties.push_back(prop);

                    // Report progress after encoding scales: 0.6f
                    if (progress_callback && !progress_callback(0.6f)) {
                        throw std::runtime_error("CANCELLED");
                    }
                }

                // Encode orientation
                {
                    tl_quat_data.resize(count * 4);
                    for (uint32_t i = 0; i < count; ++i) {
                        // SplatData stores as [w, x, y, z], we need [x, y, z, w] for encoding
                        tl_quat_data[i * 4 + 0] = rotation_ptr[(base + i) * 4 + 1]; // x
                        tl_quat_data[i * 4 + 1] = rotation_ptr[(base + i) * 4 + 2]; // y
                        tl_quat_data[i * 4 + 2] = rotation_ptr[(base + i) * 4 + 3]; // z
                        tl_quat_data[i * 4 + 3] = rotation_ptr[(base + i) * 4 + 0]; // w
                    }

                    auto encoded = PropertyEncoder::encode_orientation(tl_quat_data.data(), count, RadOrientationEncoding::Auto);
                    auto compressed = rad_compress(encoded.data.data(), encoded.data.size(), compression_level_);

                    RadChunkProperty prop;
                    prop.property = PROP_ORIENTATION;
                    prop.encoding = encoded.encoding;
                    prop.compression = "gz";
                    prop.bytes = compressed.size();

                    encoded_props.push_back({std::move(compressed), encoded.encoding, "gz",
                                             encoded.min_val, encoded.max_val, encoded.base, encoded.scale});
                    chunk_meta.properties.push_back(prop);

                    // Report progress after encoding orientation: 0.8f
                    if (progress_callback && !progress_callback(0.8f)) {
                        throw std::runtime_error("CANCELLED");
                    }
                }

                // Encode SH if present
                if (sh_coeffs > 0 && shN_ptr != nullptr) {
                    auto encode_sh_band = [&](const char* prop_name, int coeff_start, int coeff_count) {
                        if (sh_coeffs < coeff_start + coeff_count) {
                            return;
                        }
                        const size_t dims = static_cast<size_t>(coeff_count) * 3;
                        tl_sh_data.resize(static_cast<size_t>(count) * dims);
                        for (uint32_t i = 0; i < count; ++i) {
                            for (int c = 0; c < coeff_count; ++c) {
                                for (int ch = 0; ch < 3; ++ch) {
                                    tl_sh_data[i * dims + c * 3 + ch] =
                                        shN_ptr[(base + i) * sh_coeffs * 3 + (coeff_start + c) * 3 + ch];
                                }
                            }
                        }

                        auto encoded = PropertyEncoder::encode_sh(tl_sh_data.data(), dims, count, RadShEncoding::Auto);
                        auto compressed = rad_compress(encoded.data.data(), encoded.data.size(), compression_level_);

                        RadChunkProperty prop;
                        prop.property = prop_name;
                        prop.encoding = encoded.encoding;
                        prop.compression = "gz";
                        prop.bytes = compressed.size();
                        if (encoded.min_val.has_value())
                            prop.min_val = encoded.min_val.value();
                        if (encoded.max_val.has_value())
                            prop.max_val = encoded.max_val.value();
                        if (encoded.base.has_value())
                            prop.base = encoded.base.value();
                        if (encoded.scale.has_value())
                            prop.scale = encoded.scale.value();

                        encoded_props.push_back({std::move(compressed), encoded.encoding, "gz",
                                                 encoded.min_val, encoded.max_val, encoded.base, encoded.scale});
                        chunk_meta.properties.push_back(prop);
                    };

                    encode_sh_band(PROP_SH1, 0, 3);
                    encode_sh_band(PROP_SH2, 3, 5);
                    encode_sh_band(PROP_SH3, 8, 7);

                    // Report progress after encoding SH: 0.9f
                    if (progress_callback && !progress_callback(0.9f)) {
                        throw std::runtime_error("CANCELLED");
                    }
                }

                if (lod_tree && child_count_ptr != nullptr && child_start_ptr != nullptr) {
                    // Encode child_count
                    std::vector<uint8_t> child_count_data(static_cast<size_t>(count) * 2);
                    for (uint32_t i = 0; i < count; ++i) {
                        encode_u16(child_count_data.data() + static_cast<size_t>(i) * 2, child_count_ptr[base + i]);
                    }
                    auto child_count_compressed = rad_compress(child_count_data.data(), child_count_data.size(), compression_level_);

                    RadChunkProperty count_prop;
                    count_prop.property = PROP_CHILD_COUNT;
                    count_prop.encoding = "u16";
                    count_prop.compression = "gz";
                    count_prop.bytes = child_count_compressed.size();

                    encoded_props.push_back({std::move(child_count_compressed), "u16", "gz",
                                             std::nullopt, std::nullopt, std::nullopt, std::nullopt});
                    chunk_meta.properties.push_back(count_prop);

                    // Encode child_start
                    std::vector<uint8_t> child_start_data(static_cast<size_t>(count) * 4);
                    for (uint32_t i = 0; i < count; ++i) {
                        encode_u32(child_start_data.data() + static_cast<size_t>(i) * 4, child_start_ptr[base + i]);
                    }
                    auto child_start_compressed = rad_compress(child_start_data.data(), child_start_data.size(), compression_level_);

                    RadChunkProperty start_prop;
                    start_prop.property = PROP_CHILD_START;
                    start_prop.encoding = "u32";
                    start_prop.compression = "gz";
                    start_prop.bytes = child_start_compressed.size();

                    encoded_props.push_back({std::move(child_start_compressed), "u32", "gz",
                                             std::nullopt, std::nullopt, std::nullopt, std::nullopt});
                    chunk_meta.properties.push_back(start_prop);

                    // Report progress after encoding LOD data: 0.95f
                    if (progress_callback && !progress_callback(0.95f)) {
                        throw std::runtime_error("CANCELLED");
                    }
                }

                std::vector<uint8_t> payload;
                // Property offsets are payload-relative (start at first property byte),
                // not chunk-relative. This matches Spark's RAD decoder semantics where
                // absolute_offset = payload_start + prop.offset.
                uint64_t payload_bytes = 0;
                for (size_t i = 0; i < encoded_props.size(); ++i) {
                    chunk_meta.properties[i].offset = payload_bytes;
                    size_t prop_size = encoded_props[i].data.size();
                    size_t padded_size = pad8(prop_size);
                    payload_bytes += padded_size;
                }

                chunk_meta.payload_bytes = payload_bytes;

                std::string chunk_json = chunk_meta.to_json().dump();
                const size_t chunk_json_size = chunk_json.size();
                const size_t chunk_json_padded = pad8(chunk_json_size);

                payload.reserve(8 + chunk_json_padded + 8 + static_cast<size_t>(payload_bytes));
                payload.resize(8);
                encode_u32(payload.data(), RAD_CHUNK_MAGIC);
                encode_u32(payload.data() + 4, static_cast<uint32_t>(chunk_json_size));

                payload.insert(payload.end(), chunk_json.begin(), chunk_json.end());
                if (chunk_json_padded > chunk_json_size) {
                    payload.insert(payload.end(), chunk_json_padded - chunk_json_size, 0);
                }

                uint8_t payload_bytes_buf[8];
                encode_u64(payload_bytes_buf, payload_bytes);
                payload.insert(payload.end(), payload_bytes_buf, payload_bytes_buf + 8);

                for (size_t i = 0; i < encoded_props.size(); ++i) {
                    size_t prop_size = encoded_props[i].data.size();
                    size_t padded_size = pad8(prop_size);

                    payload.insert(payload.end(), encoded_props[i].data.begin(), encoded_props[i].data.end());
                    if (padded_size > prop_size) {
                        payload.insert(payload.end(), padded_size - prop_size, 0);
                    }
                }

                return {chunk_meta, payload};
            }

            int compression_level_;
            std::vector<float> lod_ratios_;
            bool flip_y_;
            ExportProgressCallback progress_callback_;

            bool report_progress(float progress, const std::string& stage) const {
                if (progress_callback_) {
                    return progress_callback_(progress, stage);
                }
                return true;
            }
        };

        // ============================================================================
        // RAD Decoder
        // ============================================================================

        class RadDecoder {
        public:
            std::expected<SplatData, std::string> decode(const std::vector<uint8_t>& data) {
                if (data.size() < 8) {
                    return std::unexpected("RAD file too small");
                }

                // Read header: 8 bytes (magic + metadata length)
                uint32_t magic = decode_u32(&data[0]);
                if (magic != RAD_MAGIC) {
                    return std::unexpected("Invalid RAD magic number");
                }

                uint32_t meta_size = decode_u32(&data[4]);

                // Read and parse metadata
                if (8 + meta_size > data.size()) {
                    return std::unexpected("RAD metadata size exceeds file size");
                }

                std::string meta_json(reinterpret_cast<const char*>(&data[8]), meta_size);
                // Trim padding spaces
                size_t actual_size = meta_json.find_last_not_of(' ');
                if (actual_size != std::string::npos) {
                    meta_json.resize(actual_size + 1);
                }

                RadMeta meta;
                try {
                    meta = RadMeta::from_json(nlohmann::json::parse(meta_json));
                } catch (const std::exception& e) {
                    return std::unexpected(std::string("Failed to parse RAD metadata: ") + e.what());
                }

                // Decode chunks
                size_t offset = 8 + pad8(meta_size);

                std::vector<float> all_means;
                std::vector<float> all_opacity;
                std::vector<float> all_sh0;
                std::vector<float> all_scales;
                std::vector<float> all_rotation;
                std::vector<float> all_shN;

                const int max_sh = meta.max_sh.value_or(0);
                const int sh_coeffs = max_sh > 0 ? SH_COEFFS_FOR_DEGREE[max_sh] : 0;

                for (size_t chunk_idx = 0; chunk_idx < meta.chunks.size(); ++chunk_idx) {
                    if (offset + 8 > data.size()) {
                        return std::unexpected("Unexpected end of RAD file (chunk header)");
                    }

                    uint32_t chunk_magic = decode_u32(&data[offset]);
                    if (chunk_magic != RAD_CHUNK_MAGIC) {
                        return std::unexpected("Invalid RAD chunk magic");
                    }

                    uint32_t chunk_meta_size = decode_u32(&data[offset + 4]);
                    const size_t chunk_meta_padded = pad8(chunk_meta_size);

                    if (offset + 8 + chunk_meta_padded + 8 > data.size()) {
                        return std::unexpected("Unexpected end of RAD file (chunk metadata)");
                    }

                    std::string chunk_json(reinterpret_cast<const char*>(&data[offset + 8]), chunk_meta_size);

                    RadChunkMeta chunk;
                    try {
                        chunk = RadChunkMeta::from_json(nlohmann::json::parse(chunk_json));
                    } catch (const std::exception& e) {
                        return std::unexpected(std::string("Failed to parse chunk metadata: ") + e.what());
                    }

                    const size_t payload_size_offset = offset + 8 + chunk_meta_padded;
                    bool has_payload_prefix = false;
                    size_t payload_start = payload_size_offset;
                    size_t chunk_end = 0;
                    if (payload_size_offset + 8 <= data.size()) {
                        const uint64_t payload_bytes = decode_u64(&data[payload_size_offset]);
                        payload_start = payload_size_offset + 8;
                        chunk_end = payload_start + static_cast<size_t>(payload_bytes);
                        has_payload_prefix = (chunk_end <= data.size()) && (chunk.payload_bytes == payload_bytes);
                    }

                    // Legacy fallback: old C++ layout did not include payload_bytes after chunk metadata.
                    if (!has_payload_prefix) {
                        payload_start = offset;
                        chunk_end = offset + pad8(static_cast<size_t>(chunk.payload_bytes));
                        if (chunk_end > data.size()) {
                            return std::unexpected("Chunk payload exceeds file bounds");
                        }
                    }

                    const size_t chunk_count = static_cast<size_t>(chunk.count);
                    std::vector<float> chunk_means(chunk_count * 3);
                    std::vector<float> chunk_opacity(chunk_count);
                    std::vector<float> chunk_sh0(chunk_count * 3);
                    std::vector<float> chunk_scales(chunk_count * 3);
                    std::vector<float> chunk_rotation(chunk_count * 4);
                    std::vector<float> chunk_shN(chunk_count * sh_coeffs * 3, 0.0f);

                    // Temporary buffers for component data
                    std::vector<float> comp_data(chunk_count);

                    for (const auto& prop : chunk.properties) {
                        // New format uses payload-relative offsets, legacy uses chunk-relative offsets.
                        const size_t prop_offset = static_cast<size_t>(prop.offset);
                        const size_t prop_bytes = static_cast<size_t>(prop.bytes);
                        size_t absolute_offset = has_payload_prefix ? (payload_start + prop_offset) : (offset + prop_offset);
                        if (absolute_offset + prop_bytes > chunk_end) {
                            return std::unexpected("Property data exceeds file bounds");
                        }

                        // Decompress if needed (handle both "gz" and legacy "gzip")
                        std::vector<uint8_t> prop_data;
                        if (prop.compression.has_value() &&
                            (prop.compression.value() == "gz" || prop.compression.value() == "gzip")) {
                            prop_data = rad_decompress(&data[absolute_offset], prop_bytes);
                            if (prop_data.empty()) {
                                return std::unexpected("Failed to decompress property: " + prop.property);
                            }
                        } else {
                            prop_data.assign(&data[absolute_offset], &data[absolute_offset + prop_bytes]);
                        }

                        // Decode based on property name
                        if (prop.property == PROP_CENTER) {
                            PropertyDecoder::decode_center(prop_data.data(), chunk_means.data(), 3, chunk_count, prop.encoding);
                        } else if (prop.property.find(PROP_CENTER) == 0 && prop.property != PROP_CENTER) {
                            // Legacy format: center_{0,1,2} per-component
                            int comp = prop.property.back() - '0';
                            PropertyDecoder::decode_center(prop_data.data(), comp_data.data(), 1, chunk_count, prop.encoding);
                            for (size_t i = 0; i < chunk_count; ++i) {
                                chunk_means[i * 3 + comp] = comp_data[i];
                            }
                        } else if (prop.property == PROP_ALPHA) {
                            float min_val = prop.min_val.value_or(0.0f);
                            float max_val = prop.max_val.value_or(1.0f);
                            PropertyDecoder::decode_alpha(prop_data.data(), chunk_opacity.data(), chunk_count,
                                                          prop.encoding, min_val, max_val);
                        } else if (prop.property == PROP_RGB) {
                            PropertyDecoder::decode_rgb(prop_data.data(), chunk_sh0.data(), 3, chunk_count,
                                                        prop.encoding,
                                                        prop.min_val.value_or(0.0f),
                                                        prop.max_val.value_or(1.0f),
                                                        prop.base.value_or(0.0f),
                                                        prop.scale.value_or(1.0f));
                        } else if (prop.property.find(PROP_RGB) == 0 && prop.property != PROP_RGB) {
                            // Legacy format: rgb_{0,1,2} per-component
                            int comp = prop.property.back() - '0';
                            float min_val = prop.min_val.value_or(0.0f);
                            float max_val = prop.max_val.value_or(1.0f);
                            float base = prop.base.value_or(0.0f);
                            float scale = prop.scale.value_or(1.0f);
                            PropertyDecoder::decode_rgb(prop_data.data(), comp_data.data(), 1, chunk_count,
                                                        prop.encoding, min_val, max_val, base, scale);
                            for (size_t i = 0; i < chunk_count; ++i) {
                                chunk_sh0[i * 3 + comp] = comp_data[i];
                            }
                        } else if (prop.property == PROP_SCALES) {
                            PropertyDecoder::decode_scales(prop_data.data(), chunk_scales.data(), 3, chunk_count,
                                                           prop.encoding,
                                                           prop.min_val.value_or(0.0f),
                                                           prop.max_val.value_or(prop.scale.value_or(1.0f)));
                        } else if (prop.property.find(PROP_SCALES) == 0 && prop.property != PROP_SCALES) {
                            // Legacy format: scales_{0,1,2} per-component
                            int comp = prop.property.back() - '0';
                            float min_val = prop.min_val.value_or(0.0f);
                            float max_val = prop.max_val.value_or(prop.scale.value_or(1.0f));
                            PropertyDecoder::decode_scales(prop_data.data(), comp_data.data(), 1, chunk_count,
                                                           prop.encoding, min_val, max_val);
                            for (size_t i = 0; i < chunk_count; ++i) {
                                chunk_scales[i * 3 + comp] = comp_data[i];
                            }
                        } else if (prop.property == PROP_ORIENTATION) {
                            std::vector<float> quat_data(chunk_count * 4);
                            PropertyDecoder::decode_orientation(prop_data.data(), quat_data.data(), chunk_count, prop.encoding);
                            // Convert from [x, y, z, w] to [w, x, y, z]
                            for (size_t i = 0; i < chunk_count; ++i) {
                                chunk_rotation[i * 4 + 0] = quat_data[i * 4 + 3]; // w
                                chunk_rotation[i * 4 + 1] = quat_data[i * 4 + 0]; // x
                                chunk_rotation[i * 4 + 2] = quat_data[i * 4 + 1]; // y
                                chunk_rotation[i * 4 + 3] = quat_data[i * 4 + 2]; // z
                            }
                        } else if (prop.property == PROP_SH1 || prop.property == PROP_SH2 || prop.property == PROP_SH3) {
                            int coeff_start = 0;
                            int coeff_count = 0;
                            if (prop.property == PROP_SH1) {
                                coeff_start = 0;
                                coeff_count = 3;
                            } else if (prop.property == PROP_SH2) {
                                coeff_start = 3;
                                coeff_count = 5;
                            } else {
                                coeff_start = 8;
                                coeff_count = 7;
                            }

                            const size_t dims = static_cast<size_t>(coeff_count) * 3;
                            std::vector<float> sh_block(chunk_count * dims, 0.0f);
                            PropertyDecoder::decode_sh(prop_data.data(), sh_block.data(), dims, chunk_count,
                                                       prop.encoding,
                                                       prop.min_val.value_or(0.0f),
                                                       prop.max_val.value_or(1.0f),
                                                       prop.base.value_or(0.0f),
                                                       prop.scale.value_or(1.0f));

                            for (size_t i = 0; i < chunk_count; ++i) {
                                for (int c = 0; c < coeff_count; ++c) {
                                    for (int ch = 0; ch < 3; ++ch) {
                                        const int coeff = coeff_start + c;
                                        if (coeff < sh_coeffs) {
                                            chunk_shN[i * sh_coeffs * 3 + coeff * 3 + ch] =
                                                sh_block[i * dims + c * 3 + ch];
                                        }
                                    }
                                }
                            }
                        } else if (prop.property.find("sh") == 0) {
                            // Legacy format: sh{1,2,3}_{coeff}_{channel}
                            size_t first_underscore = prop.property.find('_');
                            size_t second_underscore = prop.property.find('_', first_underscore + 1);
                            if (first_underscore != std::string::npos && second_underscore != std::string::npos) {
                                int coeff = std::stoi(prop.property.substr(first_underscore + 1, second_underscore - first_underscore - 1));
                                int ch = prop.property.back() - '0';
                                PropertyDecoder::decode_sh(prop_data.data(), comp_data.data(), 1, chunk_count,
                                                           prop.encoding,
                                                           prop.min_val.value_or(0.0f),
                                                           prop.max_val.value_or(1.0f),
                                                           prop.base.value_or(0.0f),
                                                           prop.scale.value_or(1.0f));
                                for (size_t i = 0; i < chunk_count; ++i) {
                                    chunk_shN[i * sh_coeffs * 3 + coeff * 3 + ch] = comp_data[i];
                                }
                            }
                        }
                    }

                    // Append chunk data to global arrays
                    all_means.insert(all_means.end(), chunk_means.begin(), chunk_means.end());
                    all_opacity.insert(all_opacity.end(), chunk_opacity.begin(), chunk_opacity.end());
                    all_sh0.insert(all_sh0.end(), chunk_sh0.begin(), chunk_sh0.end());
                    all_scales.insert(all_scales.end(), chunk_scales.begin(), chunk_scales.end());
                    all_rotation.insert(all_rotation.end(), chunk_rotation.begin(), chunk_rotation.end());
                    all_shN.insert(all_shN.end(), chunk_shN.begin(), chunk_shN.end());

                    // Move to next chunk
                    offset = chunk_end;
                }

                // Create tensors
                const size_t N = meta.count;

                Tensor means_tensor = Tensor::from_vector(all_means, {N, 3}, Device::CPU);
                Tensor opacity_tensor = Tensor::from_vector(all_opacity, {N, 1}, Device::CPU);
                Tensor sh0_tensor = Tensor::from_vector(all_sh0, {N, 1, 3}, Device::CPU);
                Tensor scales_tensor = Tensor::from_vector(all_scales, {N, 3}, Device::CPU);
                Tensor rotation_tensor = Tensor::from_vector(all_rotation, {N, 4}, Device::CPU);

                Tensor shN_tensor;
                if (sh_coeffs > 0) {
                    shN_tensor = Tensor::from_vector(all_shN, {N, static_cast<size_t>(sh_coeffs), 3}, Device::CPU);
                }

                // Create SplatData
                SplatData splat_data(
                    max_sh,
                    std::move(means_tensor),
                    std::move(sh0_tensor),
                    std::move(shN_tensor),
                    std::move(scales_tensor),
                    std::move(rotation_tensor),
                    std::move(opacity_tensor),
                    1.0f // scene_scale
                );

                return std::expected<SplatData, std::string>(std::move(splat_data));
            }
        };

    } // namespace

    // ============================================================================
    // Public API Implementation
    // ============================================================================

    std::expected<SplatData, std::string> load_rad(const std::filesystem::path& filepath) {
        auto start = std::chrono::high_resolution_clock::now();

        LOG_INFO("Loading RAD file: {}", lfs::core::path_to_utf8(filepath));

        // Read file
        std::ifstream in;
        if (!lfs::core::open_file_for_read(filepath, std::ios::binary | std::ios::ate, in)) {
            return std::unexpected(std::format("Failed to open RAD file: {}", lfs::core::path_to_utf8(filepath)));
        }

        const auto size = in.tellg();
        if (size < 0) {
            return std::unexpected(std::format("Failed to read RAD file size: {}", lfs::core::path_to_utf8(filepath)));
        }

        std::vector<uint8_t> data(static_cast<size_t>(size));
        in.seekg(0, std::ios::beg);
        in.read(reinterpret_cast<char*>(data.data()), size);
        in.close();

        if (!in.good()) {
            return std::unexpected(std::format("Failed to read RAD file: {}", lfs::core::path_to_utf8(filepath)));
        }

        // Decode
        RadDecoder decoder;
        auto result = decoder.decode(data);

        if (!result) {
            return result;
        }

        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start);

        LOG_INFO("RAD loaded: {} gaussians with SH degree {} in {}ms",
                 result->size(), result->get_max_sh_degree(), elapsed.count());

        return result;
    }

    Result<void> save_rad(const SplatData& splat_data, const RadSaveOptions& options) {
        auto start = std::chrono::high_resolution_clock::now();

        LOG_INFO("Saving RAD file: {}", lfs::core::path_to_utf8(options.output_path));

        int compression_level = options.compression_level;
        if (compression_level != Z_DEFAULT_COMPRESSION &&
            (compression_level < Z_NO_COMPRESSION || compression_level > Z_BEST_COMPRESSION)) {
            LOG_WARN("save_rad: invalid compression_level={} (expected 0..9 or -1), falling back to {}",
                     compression_level, GZ_LEVEL);
            compression_level = GZ_LEVEL;
        }

        // Encode
        RadEncoder encoder(compression_level,
                           options.lod_ratios,
                           options.flip_y,
                           scale_export_progress(options.progress_callback, 0.0f, 0.95f));
        std::vector<uint8_t> data;
        try {
            data = encoder.encode(splat_data);
        } catch (const std::runtime_error& e) {
            if (std::string(e.what()) == "CANCELLED") {
                return make_error(ErrorCode::CANCELLED, "Export cancelled by user");
            }
            throw;
        }

        if (!report_export_progress(options.progress_callback, 0.95f, "Writing RAD")) {
            return make_error(ErrorCode::CANCELLED, "RAD export cancelled", options.output_path);
        }

        if (auto dir_result = ensure_output_parent_directory(options.output_path); !dir_result) {
            return std::unexpected(dir_result.error());
        }

        ScopedAtomicOutputFile atomic_output(options.output_path);
        std::ofstream out;
        if (!lfs::core::open_file_for_write(atomic_output.temp_path(), std::ios::binary | std::ios::out, out)) {
            return make_error(ErrorCode::WRITE_FAILURE,
                              "Failed to open temporary RAD file for writing",
                              atomic_output.temp_path());
        }

        out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
        out.close();

        if (!out.good()) {
            return make_error(ErrorCode::WRITE_FAILURE,
                              "Failed to write RAD file", atomic_output.temp_path());
        }

        if (!report_export_progress(options.progress_callback, 1.0f, "RAD export complete")) {
            return make_error(ErrorCode::CANCELLED, "RAD export cancelled", options.output_path);
        }

        if (auto commit_result = atomic_output.commit(); !commit_result) {
            return std::unexpected(commit_result.error());
        }

        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start);

        // Get file size
        auto file_size = std::filesystem::file_size(options.output_path);
        LOG_INFO("RAD saved: {} gaussians, {:.1f} MB in {}ms",
                 splat_data.size(),
                 static_cast<double>(file_size) / (1024.0 * 1024.0),
                 elapsed.count());

        return {};
    }

} // namespace lfs::io
