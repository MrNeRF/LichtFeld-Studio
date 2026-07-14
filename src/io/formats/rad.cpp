/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rad.hpp"
#include "rad_dequant_math.hpp"

#include "io/cuda/rad_encode_quant.hpp"

#include "core/bhatt_lod.hpp"
#include "core/logger.hpp"
#include "core/mapped_file.hpp"
#include "core/path_utils.hpp"
#include "core/splat_data_transform.hpp"
#include "core/tensor.hpp"
#include "io/atomic_output.hpp"
#include "io/error.hpp"

#include <cuda_runtime.h>
#include <libdeflate.h>
#include <nlohmann/json.hpp>
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <zlib.h>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace lfs::io {

    using lfs::core::Device;
    using lfs::core::SplatData;
    using lfs::core::Tensor;
    using lfs::core::TensorShape;

    namespace {

        // ============================================================================
        // Constants
        // ============================================================================

        // OS-level detail (e.g. "No space left on device") for failed stream writes.
        std::string io_error_detail() {
            return errno != 0 ? std::strerror(errno) : "unknown I/O error";
        }

        constexpr uint32_t RAD_MAGIC = 0x30444152;       // "RAD0" in little-endian
        constexpr uint32_t RAD_CHUNK_MAGIC = 0x43444152; // "RADC" in little-endian
        constexpr uint32_t NATIVE_CHUNK_SIZE = kRadNativeChunkSplats;
        constexpr uint32_t SPARK_CHUNK_SIZE = kRadStreamableChunkSplats;
        constexpr uint32_t DEFAULT_RAD_FILE_CHUNK_SIZE = SPARK_CHUNK_SIZE;
        static_assert(SPARK_CHUNK_SIZE % NATIVE_CHUNK_SIZE == 0,
                      "Spark RAD file chunks must split evenly into native pages");
        constexpr int GZ_LEVEL = 6;                   // Default gzip compression level
        constexpr float SH_C0 = 0.28209479177387814f; // Degree-0 SH basis constant

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

        // Convert float32 to float16 (IEEE 754). Delegates to the shared
        // radmath implementation (ADD-composed rounding carry); the legacy
        // OR-composed encoder halved values whose mantissa carry crossed a
        // power of two, so files written by this build differ in ~0.02% of
        // f16 bytes from pre-8K-chunk files.
        inline uint16_t float32_to_float16(float value) {
            return radmath::floatToHalf(value);
        }

        // Convert float16 to float32
        inline float float16_to_float32(uint16_t value) {
            return radmath::halfToFloat(value);
        }

        // ============================================================================
        // RAD Compression/Decompression
        // ============================================================================

        struct TlsDeflateCompressor {
            libdeflate_compressor* handle = nullptr;
            int level = -1;

            ~TlsDeflateCompressor() {
                if (handle != nullptr) {
                    libdeflate_free_compressor(handle);
                }
            }

            libdeflate_compressor* get(int requested_level) {
                if (handle == nullptr || level != requested_level) {
                    if (handle != nullptr) {
                        libdeflate_free_compressor(handle);
                    }
                    handle = libdeflate_alloc_compressor(requested_level);
                    level = requested_level;
                }
                return handle;
            }
        };

        struct TlsDeflateDecompressor {
            libdeflate_decompressor* handle = nullptr;

            ~TlsDeflateDecompressor() {
                if (handle != nullptr) {
                    libdeflate_free_decompressor(handle);
                }
            }

            libdeflate_decompressor* get() {
                if (handle == nullptr) {
                    handle = libdeflate_alloc_decompressor();
                }
                return handle;
            }
        };

        thread_local TlsDeflateCompressor tls_compressor;
        thread_local TlsDeflateDecompressor tls_decompressor;

        // Reference RAD writers mark compression as "gz" but emit raw DEFLATE streams
        // (without gzip/zlib wrapper bytes).
        std::vector<uint8_t> rad_compress_zlib(const uint8_t* data, size_t size, int level = GZ_LEVEL) {
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

        std::vector<uint8_t> rad_compress(const uint8_t* data, size_t size, int level = GZ_LEVEL) {
            if (size == 0) {
                return {};
            }

            const int effective_level = std::clamp(level == Z_DEFAULT_COMPRESSION ? GZ_LEVEL : level, 0, 9);
            libdeflate_compressor* compressor = tls_compressor.get(effective_level);
            if (compressor == nullptr) {
                return rad_compress_zlib(data, size, level);
            }

            std::vector<uint8_t> output(libdeflate_deflate_compress_bound(compressor, size));
            const size_t written = libdeflate_deflate_compress(compressor, data, size, output.data(), output.size());
            if (written == 0) {
                return rad_compress_zlib(data, size, level);
            }
            output.resize(written);
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

        // Fast one-shot decompression when the decoded size is known from the
        // property layout. Falls back to streaming zlib for size mismatches and
        // wrapped/legacy streams.
        std::vector<uint8_t> rad_decompress_sized(const uint8_t* data, size_t size, size_t expected_bytes) {
            if (expected_bytes > 0 && size > 0) {
                if (libdeflate_decompressor* decompressor = tls_decompressor.get()) {
                    std::vector<uint8_t> output(expected_bytes);
                    size_t actual = 0;
                    if (libdeflate_deflate_decompress(decompressor, data, size,
                                                      output.data(), output.size(), &actual) == LIBDEFLATE_SUCCESS &&
                        actual == expected_bytes) {
                        return output;
                    }
                    if (libdeflate_gzip_decompress(decompressor, data, size,
                                                   output.data(), output.size(), &actual) == LIBDEFLATE_SUCCESS &&
                        actual == expected_bytes) {
                        return output;
                    }
                    if (libdeflate_zlib_decompress(decompressor, data, size,
                                                   output.data(), output.size(), &actual) == LIBDEFLATE_SUCCESS &&
                        actual == expected_bytes) {
                        return output;
                    }
                }
            }
            return rad_decompress(data, size);
        }

        // Decompress a property stream directly into a caller buffer of the
        // exact expected size. Streaming-profile planes always have a known
        // decoded size; any mismatch is a hard error at the caller.
        bool rad_decompress_into(const uint8_t* data, size_t size, uint8_t* dst, size_t expected_bytes) {
            if (expected_bytes == 0 || size == 0) {
                return false;
            }
            libdeflate_decompressor* const decompressor = tls_decompressor.get();
            if (decompressor == nullptr) {
                return false;
            }
            size_t actual = 0;
            if (libdeflate_deflate_decompress(decompressor, data, size, dst, expected_bytes, &actual) ==
                    LIBDEFLATE_SUCCESS &&
                actual == expected_bytes) {
                return true;
            }
            if (libdeflate_gzip_decompress(decompressor, data, size, dst, expected_bytes, &actual) ==
                    LIBDEFLATE_SUCCESS &&
                actual == expected_bytes) {
                return true;
            }
            if (libdeflate_zlib_decompress(decompressor, data, size, dst, expected_bytes, &actual) ==
                    LIBDEFLATE_SUCCESS &&
                actual == expected_bytes) {
                return true;
            }
            return false;
        }

        // In-place dimension-wise prefix sum reversing r8_delta/s8_delta; the
        // s8 variant deltas the same uint8 reinterpretation, so one byte-wise
        // pass serves both (matches decode_r8_delta/decode_s8_delta exactly).
        void undelta_planes_u8(uint8_t* plane, size_t dims, size_t count) {
            for (size_t d = 0; d < dims; ++d) {
                uint8_t* const p = plane + d * count;
                uint8_t last = 0;
                for (size_t i = 0; i < count; ++i) {
                    last = static_cast<uint8_t>(last + p[i]);
                    p[i] = last;
                }
            }
        }

        // Decoded byte count implied by a property's name + encoding, or 0 when unknown.
        size_t rad_property_decoded_bytes(const std::string& property, const std::string& encoding, size_t count) {
            size_t dims = 0;
            if (property == PROP_CENTER || property == PROP_RGB || property == PROP_SCALES) {
                dims = 3;
            } else if (property == PROP_ALPHA || property == PROP_CHILD_COUNT || property == PROP_CHILD_START) {
                dims = 1;
            } else if (property == PROP_ORIENTATION) {
                return encoding == "oct88r8" ? count * 3 : count * 3 * (encoding == "f16" ? 2 : 4);
            } else if (property == PROP_SH1) {
                dims = 9;
            } else if (property == PROP_SH2) {
                dims = 15;
            } else if (property == PROP_SH3) {
                dims = 21;
            } else if (property.find(PROP_CENTER) == 0 || property.find(PROP_RGB) == 0 ||
                       property.find(PROP_SCALES) == 0 || property.find("sh") == 0) {
                dims = 1;
            } else {
                return 0;
            }

            size_t element_bytes = 0;
            if (encoding == "f32" || encoding == "f32_lebytes" || encoding == "u32") {
                element_bytes = 4;
            } else if (encoding == "f16" || encoding == "f16_lebytes" || encoding == "ln_f16" || encoding == "u16") {
                element_bytes = 2;
            } else if (encoding == "r8" || encoding == "r8_delta" || encoding == "s8" ||
                       encoding == "s8_delta" || encoding == "ln_0r8") {
                element_bytes = 1;
            } else {
                return 0;
            }
            return count * dims * element_bytes;
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
                    output[i * dims + d] = radmath::dequantR8(encoded[src], min_val, range);
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
                    output[i * dims + d] = radmath::dequantS8(encoded[src], max_abs);
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
                    output[i * dims + d] = radmath::dequantLn0R8(encoded[src], ln_min, ln_max);
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
        std::vector<uint8_t> encode_quat_oct88r8(const float* quats, size_t count, bool input_wxyz = false) {
            std::vector<uint8_t> result(count * 3);
            constexpr float PI = 3.14159265358979323846f;
            constexpr float EPSILON = 1e-6f;

            for (size_t i = 0; i < count; ++i) {
                float x = input_wxyz ? quats[i * 4 + 1] : quats[i * 4 + 0];
                float y = input_wxyz ? quats[i * 4 + 2] : quats[i * 4 + 1];
                float z = input_wxyz ? quats[i * 4 + 3] : quats[i * 4 + 2];
                float w = input_wxyz ? quats[i * 4 + 0] : quats[i * 4 + 3];

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
            for (size_t i = 0; i < count; ++i) {
                radmath::dequantQuatOct88R8(encoded[i * 3 + 0],
                                            encoded[i * 3 + 1],
                                            encoded[i * 3 + 2],
                                            &quats[i * 4]);
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
            uint64_t count = 0;
            std::optional<int> max_sh;
            std::optional<bool> lod_tree;
            std::optional<uint32_t> chunk_size;
            uint64_t all_chunk_bytes = 0;
            std::vector<RadChunkRange> chunks;
            std::optional<nlohmann::json> splat_encoding;
            std::optional<uint32_t> sh_code_count;
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

        std::uint32_t normalized_lod_file_chunk_size(const RadMeta& meta) {
            return meta.chunk_size.value_or(NATIVE_CHUNK_SIZE);
        }

        bool supported_lod_file_chunk_size(const std::uint32_t chunk_size) {
            return chunk_size >= NATIVE_CHUNK_SIZE &&
                   chunk_size % NATIVE_CHUNK_SIZE == 0;
        }

        std::uint32_t normalized_export_chunk_size(const std::uint32_t chunk_size) {
            if (!supported_lod_file_chunk_size(chunk_size)) {
                LOG_WARN("RAD export chunk_size={} is not a positive multiple of {}; using {}",
                         chunk_size, NATIVE_CHUNK_SIZE, DEFAULT_RAD_FILE_CHUNK_SIZE);
                return DEFAULT_RAD_FILE_CHUNK_SIZE;
            }
            return chunk_size;
        }

        std::size_t native_lod_chunk_count(const std::uint64_t node_count) {
            return static_cast<std::size_t>(
                (node_count + NATIVE_CHUNK_SIZE - 1) / NATIVE_CHUNK_SIZE);
        }

        void assign_native_chunk_ranges(
            std::vector<lfs::core::SplatLodTree::ChunkFileRange>& out,
            const std::uint64_t file_offset,
            const std::uint64_t file_bytes,
            const std::uint64_t payload_offset,
            const std::uint64_t payload_bytes,
            const std::uint64_t file_base,
            const std::uint64_t file_count) {
            if (file_count == 0) {
                return;
            }
            const std::uint64_t file_end = file_base + file_count;
            std::uint64_t logical_base =
                (file_base / NATIVE_CHUNK_SIZE) * static_cast<std::uint64_t>(NATIVE_CHUNK_SIZE);
            if (logical_base < file_base) {
                logical_base += NATIVE_CHUNK_SIZE;
            }
            for (; logical_base < file_end; logical_base += NATIVE_CHUNK_SIZE) {
                const std::uint64_t logical_end =
                    std::min<std::uint64_t>(logical_base + NATIVE_CHUNK_SIZE, file_end);
                const std::uint64_t logical_count = logical_end - logical_base;
                const std::size_t logical_chunk =
                    static_cast<std::size_t>(logical_base / NATIVE_CHUNK_SIZE);
                if (logical_chunk >= out.size()) {
                    continue;
                }
                out[logical_chunk] = {
                    .file_offset = file_offset,
                    .file_bytes = file_bytes,
                    .payload_offset = payload_offset,
                    .payload_bytes = payload_bytes,
                    .file_base = file_base,
                    .file_count = file_count,
                    .base = logical_base,
                    .count = logical_count,
                };
            }
        }

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

            static EncodedProperty encode_alpha(const float* data, size_t count, RadAlphaEncoding encoding, bool lod_tree) {
                EncodedProperty result;
                const float max_encoded_alpha = lod_tree ? 2.0f : 1.0f;

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
                    auto r8_result = encode_r8(data, 1, count, 0.0f, max_encoded_alpha);
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
                        auto r8_result = encode_r8(data, 1, count, 0.0f, max_encoded_alpha);
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
                        output[i * 4 + 0] = x;
                        output[i * 4 + 1] = y;
                        output[i * 4 + 2] = z;
                        output[i * 4 + 3] = radmath::quatWFromXyz(x, y, z);
                    }
                } else if (encoding == "f16") {
                    std::vector<float> xyz(count * 3);
                    decode_f16(data, xyz.data(), 3, count);
                    for (size_t i = 0; i < count; ++i) {
                        const float x = xyz[i * 3 + 0];
                        const float y = xyz[i * 3 + 1];
                        const float z = xyz[i * 3 + 2];
                        output[i * 4 + 0] = x;
                        output[i * 4 + 1] = y;
                        output[i * 4 + 2] = z;
                        output[i * 4 + 3] = radmath::quatWFromXyz(x, y, z);
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
                const float sh_max = radmath::shMaxAbs(min_val, max_val, scale);
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

        // Decode one chunk's properties into caller-provided buffers holding
        // display-space values. Offsets are absolute positions in `data`;
        // `chunk_origin` anchors legacy chunk-relative property offsets.
        std::optional<std::string> decode_chunk_properties(
            const uint8_t* data,
            const RadChunkMeta& chunk,
            const size_t chunk_origin,
            const size_t payload_start,
            const bool has_payload_prefix,
            const size_t chunk_end,
            const int sh_coeffs,
            float* const means,
            float* const opacity,
            float* const sh0,
            float* const scales,
            float* const rotation,
            float* const shN,
            uint16_t* const child_count,
            uint32_t* const child_start) {
            const std::size_t chunk_count = static_cast<std::size_t>(chunk.count);
            std::vector<float> comp_data(chunk_count);

            // Skip decompression for properties whose destination is null:
            // tree-metadata-only decodes otherwise pay full shN/rgb/orientation
            // DEFLATE for data they discard.
            const auto property_wanted = [&](const std::string& name) -> bool {
                if (name.find(PROP_CENTER) == 0) {
                    return means != nullptr;
                }
                if (name == PROP_ALPHA) {
                    return opacity != nullptr;
                }
                if (name.find(PROP_RGB) == 0) {
                    return sh0 != nullptr;
                }
                if (name.find(PROP_SCALES) == 0) {
                    return scales != nullptr;
                }
                if (name == PROP_ORIENTATION) {
                    return rotation != nullptr;
                }
                if (name == PROP_SH1 || name == PROP_SH2 || name == PROP_SH3 ||
                    name.find("sh") == 0) {
                    return sh_coeffs > 0 && shN != nullptr;
                }
                if (name == PROP_CHILD_COUNT) {
                    return child_count != nullptr;
                }
                if (name == PROP_CHILD_START) {
                    return child_start != nullptr;
                }
                return false;
            };

            try {
                for (const auto& prop : chunk.properties) {
                    if (!property_wanted(prop.property)) {
                        continue;
                    }
                    const std::size_t prop_offset = static_cast<std::size_t>(prop.offset);
                    const std::size_t prop_bytes = static_cast<std::size_t>(prop.bytes);
                    const std::size_t absolute_offset =
                        has_payload_prefix ? (payload_start + prop_offset) : (chunk_origin + prop_offset);
                    if (absolute_offset + prop_bytes > chunk_end) {
                        return "RAD chunk property data exceeds file bounds";
                    }

                    std::vector<uint8_t> prop_data;
                    if (prop.compression.has_value() &&
                        (prop.compression.value() == "gz" || prop.compression.value() == "gzip")) {
                        const size_t decoded_bytes = rad_property_decoded_bytes(prop.property, prop.encoding, chunk_count);
                        prop_data = rad_decompress_sized(&data[absolute_offset], prop_bytes, decoded_bytes);
                        if (prop_data.empty()) {
                            return "Failed to decompress RAD chunk property: " + prop.property;
                        }
                    } else {
                        prop_data.assign(&data[absolute_offset], &data[absolute_offset + prop_bytes]);
                    }

                    if (prop.property == PROP_CENTER) {
                        if (means != nullptr) {
                            PropertyDecoder::decode_center(prop_data.data(), means, 3, chunk_count, prop.encoding);
                        }
                    } else if (prop.property.find(PROP_CENTER) == 0 && prop.property != PROP_CENTER && means != nullptr) {
                        const int comp = prop.property.back() - '0';
                        PropertyDecoder::decode_center(prop_data.data(), comp_data.data(), 1, chunk_count, prop.encoding);
                        for (std::size_t i = 0; i < chunk_count; ++i) {
                            means[i * 3u + static_cast<std::size_t>(comp)] = comp_data[i];
                        }
                    } else if (prop.property == PROP_ALPHA) {
                        if (opacity != nullptr)
                            PropertyDecoder::decode_alpha(prop_data.data(),
                                                          opacity,
                                                          chunk_count,
                                                          prop.encoding,
                                                          prop.min_val.value_or(0.0f),
                                                          prop.max_val.value_or(1.0f));
                    } else if (prop.property == PROP_RGB) {
                        if (sh0 != nullptr)
                            PropertyDecoder::decode_rgb(prop_data.data(),
                                                        sh0,
                                                        3,
                                                        chunk_count,
                                                        prop.encoding,
                                                        prop.min_val.value_or(0.0f),
                                                        prop.max_val.value_or(1.0f),
                                                        prop.base.value_or(0.0f),
                                                        prop.scale.value_or(1.0f));
                    } else if (prop.property.find(PROP_RGB) == 0 && prop.property != PROP_RGB && sh0 != nullptr) {
                        const int comp = prop.property.back() - '0';
                        PropertyDecoder::decode_rgb(prop_data.data(),
                                                    comp_data.data(),
                                                    1,
                                                    chunk_count,
                                                    prop.encoding,
                                                    prop.min_val.value_or(0.0f),
                                                    prop.max_val.value_or(1.0f),
                                                    prop.base.value_or(0.0f),
                                                    prop.scale.value_or(1.0f));
                        for (std::size_t i = 0; i < chunk_count; ++i) {
                            sh0[i * 3u + static_cast<std::size_t>(comp)] = comp_data[i];
                        }
                    } else if (prop.property == PROP_SCALES) {
                        if (scales != nullptr)
                            PropertyDecoder::decode_scales(prop_data.data(),
                                                           scales,
                                                           3,
                                                           chunk_count,
                                                           prop.encoding,
                                                           prop.min_val.value_or(0.0f),
                                                           prop.max_val.value_or(prop.scale.value_or(1.0f)));
                    } else if (prop.property.find(PROP_SCALES) == 0 && prop.property != PROP_SCALES && scales != nullptr) {
                        const int comp = prop.property.back() - '0';
                        PropertyDecoder::decode_scales(prop_data.data(),
                                                       comp_data.data(),
                                                       1,
                                                       chunk_count,
                                                       prop.encoding,
                                                       prop.min_val.value_or(0.0f),
                                                       prop.max_val.value_or(prop.scale.value_or(1.0f)));
                        for (std::size_t i = 0; i < chunk_count; ++i) {
                            scales[i * 3u + static_cast<std::size_t>(comp)] = comp_data[i];
                        }
                    } else if (prop.property == PROP_ORIENTATION && rotation != nullptr) {
                        std::vector<float> quat_data(chunk_count * 4u);
                        PropertyDecoder::decode_orientation(prop_data.data(), quat_data.data(), chunk_count, prop.encoding);
                        for (std::size_t i = 0; i < chunk_count; ++i) {
                            rotation[i * 4u + 0u] = quat_data[i * 4u + 3u];
                            rotation[i * 4u + 1u] = quat_data[i * 4u + 0u];
                            rotation[i * 4u + 2u] = quat_data[i * 4u + 1u];
                            rotation[i * 4u + 3u] = quat_data[i * 4u + 2u];
                        }
                    } else if ((prop.property == PROP_SH1 || prop.property == PROP_SH2 || prop.property == PROP_SH3) &&
                               sh_coeffs > 0 && shN != nullptr) {
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

                        const std::size_t dims = static_cast<std::size_t>(coeff_count) * 3u;
                        std::vector<float> sh_block(chunk_count * dims, 0.0f);
                        PropertyDecoder::decode_sh(prop_data.data(),
                                                   sh_block.data(),
                                                   dims,
                                                   chunk_count,
                                                   prop.encoding,
                                                   prop.min_val.value_or(0.0f),
                                                   prop.max_val.value_or(1.0f),
                                                   prop.base.value_or(0.0f),
                                                   prop.scale.value_or(1.0f));

                        for (std::size_t i = 0; i < chunk_count; ++i) {
                            for (int c = 0; c < coeff_count; ++c) {
                                const int coeff = coeff_start + c;
                                if (coeff >= sh_coeffs) {
                                    continue;
                                }
                                for (int ch = 0; ch < 3; ++ch) {
                                    shN[i * static_cast<std::size_t>(sh_coeffs) * 3u +
                                        static_cast<std::size_t>(coeff) * 3u +
                                        static_cast<std::size_t>(ch)] =
                                        sh_block[i * dims + static_cast<std::size_t>(c) * 3u +
                                                 static_cast<std::size_t>(ch)];
                                }
                            }
                        }
                    } else if (prop.property.find("sh") == 0 && sh_coeffs > 0 && shN != nullptr) {
                        const std::size_t first_underscore = prop.property.find('_');
                        const std::size_t second_underscore = prop.property.find('_', first_underscore + 1);
                        if (first_underscore != std::string::npos && second_underscore != std::string::npos) {
                            const int coeff = std::stoi(prop.property.substr(first_underscore + 1,
                                                                             second_underscore - first_underscore - 1));
                            const int ch = prop.property.back() - '0';
                            if (coeff >= 0 && coeff < sh_coeffs && ch >= 0 && ch < 3) {
                                PropertyDecoder::decode_sh(prop_data.data(),
                                                           comp_data.data(),
                                                           1,
                                                           chunk_count,
                                                           prop.encoding,
                                                           prop.min_val.value_or(0.0f),
                                                           prop.max_val.value_or(1.0f),
                                                           prop.base.value_or(0.0f),
                                                           prop.scale.value_or(1.0f));
                                for (std::size_t i = 0; i < chunk_count; ++i) {
                                    shN[i * static_cast<std::size_t>(sh_coeffs) * 3u +
                                        static_cast<std::size_t>(coeff) * 3u +
                                        static_cast<std::size_t>(ch)] = comp_data[i];
                                }
                            }
                        }
                    } else if (prop.property == PROP_CHILD_COUNT && child_count != nullptr) {
                        if (prop_data.size() >= chunk_count * 2u) {
                            for (std::size_t i = 0; i < chunk_count; ++i) {
                                child_count[i] = decode_u16(&prop_data[i * 2u]);
                            }
                        }
                    } else if (prop.property == PROP_CHILD_START && child_start != nullptr) {
                        if (prop_data.size() >= chunk_count * 4u) {
                            for (std::size_t i = 0; i < chunk_count; ++i) {
                                child_start[i] = decode_u32(&prop_data[i * 4u]);
                            }
                        }
                    }
                }
            } catch (const std::exception& e) {
                return std::string("Failed to decode RAD chunk: ") + e.what();
            }
            return std::nullopt;
        }

        std::expected<RadDecodedChunk, std::string> decode_rad_chunk_buffer(
            const std::vector<uint8_t>& data,
            int fallback_max_sh,
            const bool has_lod_tree,
            bool lod_opacity_encoded) {
            if (data.size() < 8) {
                return std::unexpected("RAD chunk too small");
            }

            std::size_t offset = 0;
            const uint32_t chunk_magic = decode_u32(&data[offset]);
            if (chunk_magic != RAD_CHUNK_MAGIC) {
                return std::unexpected("Invalid RAD chunk magic");
            }

            const uint32_t chunk_meta_size = decode_u32(&data[offset + 4]);
            const std::size_t chunk_meta_padded = pad8(chunk_meta_size);
            if (offset + 8 + chunk_meta_padded + 8 > data.size()) {
                return std::unexpected("Unexpected end of RAD chunk metadata");
            }

            RadChunkMeta chunk;
            try {
                std::string chunk_json(reinterpret_cast<const char*>(&data[offset + 8]), chunk_meta_size);
                chunk = RadChunkMeta::from_json(nlohmann::json::parse(chunk_json));
            } catch (const std::exception& e) {
                return std::unexpected(std::string("Failed to parse RAD chunk metadata: ") + e.what());
            }

            const std::size_t payload_size_offset = offset + 8 + chunk_meta_padded;
            bool has_payload_prefix = false;
            std::size_t payload_start = payload_size_offset;
            std::size_t chunk_end = 0;
            if (payload_size_offset + 8 <= data.size()) {
                const uint64_t payload_bytes = decode_u64(&data[payload_size_offset]);
                payload_start = payload_size_offset + 8;
                chunk_end = payload_start + static_cast<std::size_t>(payload_bytes);
                has_payload_prefix = (chunk_end <= data.size()) && (chunk.payload_bytes == payload_bytes);
            }

            if (!has_payload_prefix) {
                payload_start = offset;
                chunk_end = offset + pad8(static_cast<std::size_t>(chunk.payload_bytes));
                if (chunk_end > data.size()) {
                    return std::unexpected("RAD chunk payload exceeds file bounds");
                }
            }

            if (chunk.splat_encoding.has_value()) {
                const auto& enc = chunk.splat_encoding.value();
                if (enc.is_object()) {
                    const auto it = enc.find("lodOpacity");
                    if (it != enc.end() && it->is_boolean()) {
                        lod_opacity_encoded = it->get<bool>();
                    }
                }
            }

            int max_sh = chunk.max_sh > 0 ? chunk.max_sh : fallback_max_sh;
            max_sh = std::clamp(max_sh, 0, 3);
            const int sh_coeffs = max_sh > 0 ? SH_COEFFS_FOR_DEGREE[max_sh] : 0;
            const std::size_t chunk_count = static_cast<std::size_t>(chunk.count);
            const bool decode_tree = has_lod_tree || chunk.lod_tree;

            std::vector<float> chunk_means(chunk_count * 3u);
            std::vector<float> chunk_opacity(chunk_count);
            std::vector<float> chunk_sh0(chunk_count * 3u);
            std::vector<float> chunk_scales(chunk_count * 3u);
            std::vector<float> chunk_rotation(chunk_count * 4u);
            std::vector<float> chunk_shN(chunk_count * static_cast<std::size_t>(sh_coeffs) * 3u, 0.0f);
            std::vector<uint16_t> chunk_child_count;
            std::vector<uint32_t> chunk_child_start;
            if (decode_tree) {
                chunk_child_count.resize(chunk_count);
                chunk_child_start.resize(chunk_count);
            }

            if (auto err = decode_chunk_properties(data.data(),
                                                   chunk,
                                                   offset,
                                                   payload_start,
                                                   has_payload_prefix,
                                                   chunk_end,
                                                   sh_coeffs,
                                                   chunk_means.data(),
                                                   chunk_opacity.data(),
                                                   chunk_sh0.data(),
                                                   chunk_scales.data(),
                                                   chunk_rotation.data(),
                                                   chunk_shN.empty() ? nullptr : chunk_shN.data(),
                                                   decode_tree ? chunk_child_count.data() : nullptr,
                                                   decode_tree ? chunk_child_start.data() : nullptr);
                err.has_value()) {
                return std::unexpected(std::move(*err));
            }

            for (float& v : chunk_sh0) {
                v = (v - 0.5f) / SH_C0;
            }
            if (!lod_opacity_encoded) {
                for (float& v : chunk_opacity) {
                    const float a = std::clamp(v, 1.0e-6f, 1.0f - 1.0e-6f);
                    v = std::log(a / (1.0f - a));
                }
            } else {
                for (float& v : chunk_opacity) {
                    v = std::max(v, 0.0f);
                }
            }
            for (float& v : chunk_scales) {
                v = std::log(std::max(v, 1.0e-8f));
            }

            return RadDecodedChunk{
                .base = chunk.base,
                .count = chunk.count,
                .max_sh_degree = max_sh,
                .sh_coeffs_rest = static_cast<std::uint32_t>(sh_coeffs),
                .lod_opacity_encoded = lod_opacity_encoded,
                .means = std::move(chunk_means),
                .opacity_raw = std::move(chunk_opacity),
                .sh0_raw = std::move(chunk_sh0),
                .scaling_raw = std::move(chunk_scales),
                .rotation_raw = std::move(chunk_rotation),
                .shN_canonical = std::move(chunk_shN),
                .child_count = std::move(chunk_child_count),
                .child_start = std::move(chunk_child_start),
            };
        }

        // ============================================================================
        // RAD Chunk Encoding
        // ============================================================================

        std::pair<RadChunkMeta, std::vector<uint8_t>> encode_rad_chunk(
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
            int compression_level,
            const cuda::RadEncodeQuantChunkOut* gpu_planes = nullptr,
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
            thread_local std::vector<float> tl_sh_data;

            // Encode center (3 components together as single property)
            {
                EncodedProperty encoded;
                const uint8_t* enc_data = nullptr;
                size_t enc_size = 0;
                if (gpu_planes != nullptr) {
                    encoded.encoding = "f32_lebytes";
                    enc_data = gpu_planes->center;
                    enc_size = static_cast<size_t>(count) * 12;
                } else {
                    encoded = PropertyEncoder::encode_center(means_ptr, 3, count, RadCenterEncoding::Auto);
                    enc_data = encoded.data.data();
                    enc_size = encoded.data.size();
                }
                auto compressed = rad_compress(enc_data, enc_size, compression_level);

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
                EncodedProperty encoded;
                const uint8_t* enc_data = nullptr;
                size_t enc_size = 0;
                if (gpu_planes != nullptr) {
                    enc_data = gpu_planes->alpha;
                    if (gpu_planes->alpha_f16) {
                        encoded.encoding = "f16";
                        enc_size = static_cast<size_t>(count) * 2;
                    } else {
                        encoded.encoding = "r8";
                        encoded.min_val = gpu_planes->alpha_min;
                        encoded.max_val = gpu_planes->alpha_max;
                        enc_size = count;
                    }
                } else {
                    encoded = PropertyEncoder::encode_alpha(opacity_ptr, count, RadAlphaEncoding::Auto, lod_tree);
                    enc_data = encoded.data.data();
                    enc_size = encoded.data.size();
                }
                auto compressed = rad_compress(enc_data, enc_size, compression_level);

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
                EncodedProperty encoded;
                const uint8_t* enc_data = nullptr;
                size_t enc_size = 0;
                if (gpu_planes != nullptr) {
                    encoded.encoding = "r8_delta";
                    encoded.min_val = gpu_planes->rgb_min;
                    encoded.max_val = gpu_planes->rgb_max;
                    enc_data = gpu_planes->rgb;
                    enc_size = static_cast<size_t>(count) * 3;
                } else {
                    encoded = PropertyEncoder::encode_rgb(sh0_ptr, 3, count, RadRgbEncoding::Auto);
                    enc_data = encoded.data.data();
                    enc_size = encoded.data.size();
                }
                auto compressed = rad_compress(enc_data, enc_size, compression_level);

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
                // Encode all 3 components together as "scales" property
                auto encoded = PropertyEncoder::encode_scales(scales_ptr, 3, count, RadScalesEncoding::Auto);
                auto compressed = rad_compress(encoded.data.data(), encoded.data.size(), compression_level);

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
                EncodedProperty encoded;
                encoded.data = encode_quat_oct88r8(rotation_ptr, count, true);
                encoded.encoding = "oct88r8";
                encoded.compression = "none";
                auto compressed = rad_compress(encoded.data.data(), encoded.data.size(), compression_level);

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
                auto encode_sh_band = [&](const char* prop_name, int band, int coeff_start, int coeff_count) {
                    if (sh_coeffs < coeff_start + coeff_count) {
                        return;
                    }
                    const size_t dims = static_cast<size_t>(coeff_count) * 3;
                    EncodedProperty encoded;
                    const uint8_t* enc_data = nullptr;
                    size_t enc_size = 0;
                    if (gpu_planes != nullptr && gpu_planes->sh[band] != nullptr) {
                        encoded.encoding = "s8";
                        encoded.min_val = -gpu_planes->sh_max_abs[band];
                        encoded.max_val = gpu_planes->sh_max_abs[band];
                        enc_data = gpu_planes->sh[band];
                        enc_size = static_cast<size_t>(count) * dims;
                    } else {
                        tl_sh_data.resize(static_cast<size_t>(count) * dims);
                        for (uint32_t i = 0; i < count; ++i) {
                            for (int c = 0; c < coeff_count; ++c) {
                                for (int ch = 0; ch < 3; ++ch) {
                                    tl_sh_data[i * dims + c * 3 + ch] =
                                        shN_ptr[static_cast<size_t>(i) * sh_coeffs * 3 + (coeff_start + c) * 3 + ch];
                                }
                            }
                        }

                        encoded = PropertyEncoder::encode_sh(tl_sh_data.data(), dims, count, RadShEncoding::Auto);
                        enc_data = encoded.data.data();
                        enc_size = encoded.data.size();
                    }
                    auto compressed = rad_compress(enc_data, enc_size, compression_level);

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

                encode_sh_band(PROP_SH1, 0, 0, 3);
                encode_sh_band(PROP_SH2, 1, 3, 5);
                encode_sh_band(PROP_SH3, 2, 8, 7);

                // Report progress after encoding SH: 0.9f
                if (progress_callback && !progress_callback(0.9f)) {
                    throw std::runtime_error("CANCELLED");
                }
            }

            if (lod_tree && child_count_ptr != nullptr && child_start_ptr != nullptr) {
                // Encode child_count
                std::vector<uint8_t> child_count_data(static_cast<size_t>(count) * 2);
                for (uint32_t i = 0; i < count; ++i) {
                    encode_u16(child_count_data.data() + static_cast<size_t>(i) * 2, child_count_ptr[i]);
                }
                auto child_count_compressed = rad_compress(child_count_data.data(), child_count_data.size(), compression_level);

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
                    encode_u32(child_start_data.data() + static_cast<size_t>(i) * 4, child_start_ptr[i]);
                }
                auto child_start_compressed = rad_compress(child_start_data.data(), child_start_data.size(), compression_level);

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

        // ============================================================================
        // RAD Encoder
        // ============================================================================

        class RadEncoder {
        public:
            explicit RadEncoder(int compression_level = GZ_LEVEL,
                                bool flip_y = false,
                                std::uint32_t file_chunk_size = DEFAULT_RAD_FILE_CHUNK_SIZE,
                                ExportProgressCallback progress_callback = nullptr)
                : compression_level_(compression_level),
                  flip_y_(flip_y),
                  file_chunk_size_(normalized_export_chunk_size(file_chunk_size)),
                  progress_callback_(std::move(progress_callback)) {}

            std::vector<uint8_t> encode(const SplatData& splat_data) {
                // 0.0: Preparing data
                if (!report_progress(0.0f, "Preparing data...")) {
                    throw std::runtime_error("CANCELLED");
                }

                std::optional<SplatData> visible_splat_data;
                std::optional<SplatData> lod_splat_data;
                const SplatData* export_source = &splat_data;

                const bool has_deleted = splat_data.has_deleted_mask() && splat_data.deleted().count_nonzero() > 0;

                if (has_deleted) {
                    const Tensor keep_mask = splat_data.deleted().logical_not();
                    auto extracted = lfs::core::extract_by_mask(splat_data, keep_mask);
                    if (extracted.size() > 0) {
                        visible_splat_data = std::move(extracted);
                        export_source = &visible_splat_data.value();
                    }
                }

                // Build LOD tree if the source doesn't have one.
                if (!export_source->lod_tree || !export_source->lod_tree->has_tree()) {
                    auto lod_progress = [&](float p, const std::string& stage) -> bool {
                        return report_progress(p * 0.1f, stage);
                    };
                    auto lod_result = lfs::core::build_bhatt_lod(*export_source, 1.25f, lod_progress);
                    if (lod_result && (*lod_result)->lod_tree && (*lod_result)->lod_tree->has_tree()) {
                        lod_splat_data = std::move(**lod_result);
                        export_source = &lod_splat_data.value();
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

                // 0.3: Preparing chunks
                if (!report_progress(0.3f, "Preparing chunks...")) {
                    throw std::runtime_error("CANCELLED");
                }

                if (packed.count > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
                    throw std::runtime_error("RAD export exceeds maximum supported splat count");
                }

                const uint32_t num_splats = static_cast<uint32_t>(packed.count);
                const int sh_degree = packed.sh_degree;
                const int sh_coeffs = packed.sh_coeffs;
                const bool lod_tree = packed.lod_tree;

                const uint32_t num_chunks =
                    (num_splats + file_chunk_size_ - 1) / file_chunk_size_;

                // Build metadata
                RadMeta meta;
                meta.count = num_splats;
                meta.max_sh = sh_degree;
                meta.lod_tree = lod_tree ? std::optional<bool>(true) : std::nullopt;
                meta.chunk_size = file_chunk_size_;
                if (lod_tree) {
                    meta.splat_encoding = nlohmann::json{{"lodOpacity", true}};
                }

                // Encode chunks in parallel. lfs_io already links TBB, so keep
                // RAD export on the same threading runtime as the rest of IO.
                std::vector<std::vector<uint8_t>> chunk_payloads(num_chunks);
                std::vector<RadChunkRange> chunk_ranges(num_chunks);
                std::atomic<uint32_t> completed_chunks{0};

                // Report initial progress
                if (!report_progress(0.5f, "Encoding chunks...")) {
                    throw std::runtime_error("CANCELLED");
                }

                tbb::parallel_for(
                    tbb::blocked_range<uint32_t>(0, num_chunks, 1),
                    [&](const tbb::blocked_range<uint32_t>& range) {
                        for (uint32_t chunk_idx = range.begin(); chunk_idx != range.end(); ++chunk_idx) {
                            const uint32_t base = chunk_idx * file_chunk_size_;
                            const uint32_t count = std::min(file_chunk_size_, num_splats - base);

                            auto chunk_progress_cb = [&](float /*progress*/) -> bool {
                                const uint32_t completed = completed_chunks.load(std::memory_order_relaxed);
                                if (completed % 16 == 0) {
                                    const float overall = 0.5f + (0.4f * static_cast<float>(completed) / static_cast<float>(num_chunks));
                                    return report_progress(overall, "Encoding chunks...");
                                }
                                return true;
                            };

                            auto chunk_result = encode_rad_chunk(
                                base, count, sh_degree, sh_coeffs,
                                packed.means + static_cast<size_t>(base) * 3,
                                packed.opacity + base,
                                packed.sh0 + static_cast<size_t>(base) * 3,
                                packed.scales + static_cast<size_t>(base) * 3,
                                packed.rotation + static_cast<size_t>(base) * 4,
                                packed.shN != nullptr ? packed.shN + static_cast<size_t>(base) * sh_coeffs * 3 : nullptr,
                                lod_tree ? packed.child_count.data() + base : nullptr,
                                lod_tree ? packed.child_start.data() + base : nullptr,
                                lod_tree,
                                compression_level_,
                                nullptr,
                                chunk_progress_cb);

                            chunk_ranges[chunk_idx].base = base;
                            chunk_ranges[chunk_idx].count = count;
                            chunk_ranges[chunk_idx].bytes = chunk_result.second.size();
                            chunk_payloads[chunk_idx] = std::move(chunk_result.second);

                            completed_chunks.fetch_add(1, std::memory_order_relaxed);
                        }
                    });

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
            // Holds CPU-resident tensors (or transformed copies where the RAD
            // domain differs) and exposes raw pointers for chunk encoding.
            struct PackedSplatData {
                size_t count = 0;
                int sh_degree = 0;
                int sh_coeffs = 0;
                bool lod_tree = false;
                Tensor means_storage;
                Tensor opacity_storage;
                Tensor scales_storage;
                Tensor shN_storage;
                std::vector<float> means_flipped;
                std::vector<float> sh0_display;
                std::vector<float> rotation_normalized;
                const float* means = nullptr;
                const float* opacity = nullptr;
                const float* sh0 = nullptr;
                const float* scales = nullptr;
                const float* rotation = nullptr;
                const float* shN = nullptr;
                std::vector<uint16_t> child_count;
                std::vector<uint32_t> child_start;
            };

            static PackedSplatData pack_splat_data(const SplatData& splat_data, bool flip_y = false) {
                PackedSplatData packed;
                packed.count = static_cast<size_t>(splat_data.size());
                packed.sh_degree = std::clamp(splat_data.get_max_sh_degree(), 0, 3);
                packed.sh_coeffs = packed.sh_degree > 0 ? SH_COEFFS_FOR_DEGREE[packed.sh_degree] : 0;
                if (packed.count == 0) {
                    return packed;
                }

                auto cpu_contiguous = [](Tensor tensor) {
                    return tensor.contiguous().to(Device::CPU);
                };

                // Spark RAD stores render-space values, not optimizer-domain tensors.
                packed.means_storage = cpu_contiguous(splat_data.get_means());
                packed.means = packed.means_storage.ptr<float>();
                if (flip_y) {
                    const float* const src = packed.means;
                    packed.means_flipped.resize(packed.count * 3);
                    tbb::parallel_for(
                        tbb::blocked_range<size_t>(0, packed.count),
                        [&](const tbb::blocked_range<size_t>& range) {
                            for (size_t i = range.begin(); i != range.end(); ++i) {
                                packed.means_flipped[i * 3 + 0] = src[i * 3 + 0];
                                packed.means_flipped[i * 3 + 1] = -src[i * 3 + 1];
                                packed.means_flipped[i * 3 + 2] = src[i * 3 + 2];
                            }
                        });
                    packed.means = packed.means_flipped.data();
                }

                if (splat_data.lod_tree && splat_data.lod_tree->lod_opacity_encoded) {
                    // For LOD-encoded opacity, read raw values directly (display-space, can exceed 1.0)
                    packed.opacity_storage = cpu_contiguous(splat_data.opacity_raw());
                } else {
                    packed.opacity_storage = cpu_contiguous(splat_data.get_opacity());
                }
                packed.opacity = packed.opacity_storage.ptr<float>();

                const Tensor sh0_cpu = cpu_contiguous(splat_data.sh0_raw());
                const float* const sh0_src = sh0_cpu.ptr<float>();
                packed.sh0_display.resize(packed.count * 3);
                tbb::parallel_for(
                    tbb::blocked_range<size_t>(0, packed.sh0_display.size()),
                    [&](const tbb::blocked_range<size_t>& range) {
                        for (size_t i = range.begin(); i != range.end(); ++i) {
                            packed.sh0_display[i] = 0.5f + SH_C0 * sh0_src[i];
                        }
                    });
                packed.sh0 = packed.sh0_display.data();

                packed.scales_storage = cpu_contiguous(splat_data.get_scaling());
                packed.scales = packed.scales_storage.ptr<float>();

                // Normalize quaternions directly: the tensor-op chain
                // (square/sum/sqrt/div) materializes four temporaries and
                // dominates pack time on CPU.
                const Tensor rotation_cpu = cpu_contiguous(splat_data.rotation_raw());
                const float* const rot_src = rotation_cpu.ptr<float>();
                packed.rotation_normalized.resize(packed.count * 4);
                tbb::parallel_for(
                    tbb::blocked_range<size_t>(0, packed.count),
                    [&](const tbb::blocked_range<size_t>& range) {
                        for (size_t i = range.begin(); i != range.end(); ++i) {
                            const float x = rot_src[i * 4 + 0];
                            const float y = rot_src[i * 4 + 1];
                            const float z = rot_src[i * 4 + 2];
                            const float w = rot_src[i * 4 + 3];
                            // Match the tensor-op reference bit-for-bit: float
                            // squares, double-accumulated sum, float sqrt.
                            double sum_squared = 0.0;
                            sum_squared += x * x;
                            sum_squared += y * y;
                            sum_squared += z * z;
                            sum_squared += w * w;
                            const float norm = std::max(std::sqrt(static_cast<float>(sum_squared)), 1e-12f);
                            packed.rotation_normalized[i * 4 + 0] = x / norm;
                            packed.rotation_normalized[i * 4 + 1] = y / norm;
                            packed.rotation_normalized[i * 4 + 2] = z / norm;
                            packed.rotation_normalized[i * 4 + 3] = w / norm;
                        }
                    });
                packed.rotation = packed.rotation_normalized.data();
                if (packed.sh_coeffs > 0) {
                    // shN is stored swizzled; unpack on CPU to avoid a canonical CUDA copy.
                    packed.shN_storage = splat_data.shN_canonical_cpu();
                    packed.shN = packed.shN_storage.ptr<float>();
                }
                if (splat_data.lod_tree && splat_data.lod_tree->has_tree()) {
                    packed.lod_tree = true;
                    packed.child_count = splat_data.lod_tree->child_count;
                    packed.child_start = splat_data.lod_tree->child_start;
                }

                return packed;
            }

            int compression_level_;
            bool flip_y_;
            std::uint32_t file_chunk_size_;
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
            std::expected<SplatData, std::string> decode(
                const std::vector<uint8_t>& data,
                const std::filesystem::path* source_path = nullptr) {
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

                const int max_sh = meta.max_sh.value_or(0);
                const int sh_coeffs = max_sh > 0 ? SH_COEFFS_FOR_DEGREE[max_sh] : 0;
                const bool has_lod_tree = meta.lod_tree.value_or(false);
                const size_t N = meta.count;

                bool lod_opacity_encoded = has_lod_tree;
                if (meta.splat_encoding.has_value()) {
                    const auto& enc = meta.splat_encoding.value();
                    if (enc.is_object()) {
                        auto it = enc.find("lodOpacity");
                        if (it != enc.end() && it->is_boolean()) {
                            lod_opacity_encoded = it->get<bool>();
                        }
                    }
                }

                struct ChunkSlice {
                    RadChunkMeta meta;
                    size_t origin = 0;
                    size_t payload_start = 0;
                    size_t chunk_end = 0;
                    bool has_payload_prefix = false;
                };

                std::vector<ChunkSlice> slices;
                slices.reserve(meta.chunks.size());
                std::vector<lfs::core::SplatLodTree::ChunkFileRange> chunk_file_ranges;
                if (has_lod_tree) {
                    chunk_file_ranges.resize(native_lod_chunk_count(N));
                }

                uint64_t scanned_count = 0;
                for (size_t chunk_idx = 0; chunk_idx < meta.chunks.size(); ++chunk_idx) {
                    const size_t chunk_file_offset = offset;
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
                    if (has_lod_tree) {
                        assign_native_chunk_ranges(
                            chunk_file_ranges,
                            static_cast<uint64_t>(chunk_file_offset),
                            static_cast<uint64_t>(chunk_end - chunk_file_offset),
                            static_cast<uint64_t>(payload_start),
                            chunk.payload_bytes,
                            chunk.base,
                            chunk.count);
                    }

                    // Chunks must tile [0, N) in file order for the parallel
                    // slice writes below to be disjoint and complete.
                    if (chunk.base != scanned_count || chunk.base + chunk.count > N) {
                        return std::unexpected("RAD chunk base/count layout is inconsistent");
                    }
                    scanned_count += chunk.count;

                    slices.push_back({std::move(chunk), chunk_file_offset, payload_start, chunk_end, has_payload_prefix});
                    offset = chunk_end;
                }
                if (scanned_count != N) {
                    return std::unexpected("RAD chunk counts do not sum to splat count");
                }

                // Decode straight into the output tensors, one chunk per task.
                Tensor means_tensor = Tensor::empty({N, 3}, Device::CPU, lfs::core::DataType::Float32);
                Tensor opacity_tensor = Tensor::empty({N, 1}, Device::CPU, lfs::core::DataType::Float32);
                Tensor sh0_tensor = Tensor::empty({N, 1, 3}, Device::CPU, lfs::core::DataType::Float32);
                Tensor scales_tensor = Tensor::empty({N, 3}, Device::CPU, lfs::core::DataType::Float32);
                Tensor rotation_tensor = Tensor::empty({N, 4}, Device::CPU, lfs::core::DataType::Float32);
                Tensor shN_tensor;
                if (sh_coeffs > 0) {
                    shN_tensor = Tensor::empty({N, static_cast<size_t>(sh_coeffs), 3}, Device::CPU,
                                               lfs::core::DataType::Float32);
                }

                float* const all_means = means_tensor.ptr<float>();
                float* const all_opacity = opacity_tensor.ptr<float>();
                float* const all_sh0 = sh0_tensor.ptr<float>();
                float* const all_scales = scales_tensor.ptr<float>();
                float* const all_rotation = rotation_tensor.ptr<float>();
                float* const all_shN = sh_coeffs > 0 ? shN_tensor.ptr<float>() : nullptr;

                std::vector<uint16_t> all_child_count;
                std::vector<uint32_t> all_child_start;
                auto tree = has_lod_tree ? std::make_unique<lfs::core::SplatLodTree>() : nullptr;
                if (has_lod_tree) {
                    all_child_count.resize(N);
                    all_child_start.resize(N);
                    tree->centers.resize(N);
                    tree->sizes.resize(N);
                }

                std::atomic<bool> failed{false};
                std::mutex error_mutex;
                std::string decode_error;
                auto record_error = [&](std::string msg) {
                    failed.store(true, std::memory_order_relaxed);
                    std::lock_guard<std::mutex> lock(error_mutex);
                    if (decode_error.empty()) {
                        decode_error = std::move(msg);
                    }
                };

                tbb::parallel_for(
                    tbb::blocked_range<size_t>(0, slices.size(), 1),
                    [&](const tbb::blocked_range<size_t>& range) {
                        for (size_t chunk_idx = range.begin(); chunk_idx != range.end(); ++chunk_idx) {
                            if (failed.load(std::memory_order_relaxed)) {
                                return;
                            }
                            const ChunkSlice& slice = slices[chunk_idx];
                            const size_t base = static_cast<size_t>(slice.meta.base);
                            const size_t chunk_count = static_cast<size_t>(slice.meta.count);

                            float* const means = all_means + base * 3;
                            float* const opacity = all_opacity + base;
                            float* const sh0 = all_sh0 + base * 3;
                            float* const scales = all_scales + base * 3;
                            float* const rotation = all_rotation + base * 4;
                            float* const shN =
                                all_shN != nullptr ? all_shN + base * static_cast<size_t>(sh_coeffs) * 3 : nullptr;

                            std::memset(means, 0, chunk_count * 3 * sizeof(float));
                            std::memset(opacity, 0, chunk_count * sizeof(float));
                            std::memset(sh0, 0, chunk_count * 3 * sizeof(float));
                            std::memset(scales, 0, chunk_count * 3 * sizeof(float));
                            std::memset(rotation, 0, chunk_count * 4 * sizeof(float));
                            if (shN != nullptr) {
                                std::memset(shN, 0, chunk_count * static_cast<size_t>(sh_coeffs) * 3 * sizeof(float));
                            }

                            auto err = decode_chunk_properties(
                                data.data(), slice.meta, slice.origin, slice.payload_start,
                                slice.has_payload_prefix, slice.chunk_end, sh_coeffs,
                                means, opacity, sh0, scales, rotation, shN,
                                has_lod_tree ? all_child_count.data() + base : nullptr,
                                has_lod_tree ? all_child_start.data() + base : nullptr);
                            if (err.has_value()) {
                                record_error(std::move(*err));
                                return;
                            }

                            // RAD stores display RGB in the SH0 slot and activated
                            // alpha/scale values; convert back to optimizer domain.
                            for (size_t i = 0; i < chunk_count * 3; ++i) {
                                sh0[i] = (sh0[i] - 0.5f) / SH_C0;
                            }
                            if (!lod_opacity_encoded) {
                                for (size_t i = 0; i < chunk_count; ++i) {
                                    const float a = std::clamp(opacity[i], 1.0e-6f, 1.0f - 1.0e-6f);
                                    opacity[i] = std::log(a / (1.0f - a));
                                }
                            } else {
                                // Spark LOD opacity encoding stores display-space alpha
                                // directly and can exceed 1.0 for dense merged nodes.
                                for (size_t i = 0; i < chunk_count; ++i) {
                                    opacity[i] = std::max(opacity[i], 0.0f);
                                }
                            }
                            if (tree) {
                                for (size_t i = 0; i < chunk_count; ++i) {
                                    tree->centers[base + i] =
                                        glm::vec3(means[i * 3 + 0], means[i * 3 + 1], means[i * 3 + 2]);
                                    const float max_scale =
                                        std::max({scales[i * 3 + 0], scales[i * 3 + 1], scales[i * 3 + 2]});
                                    float expansion = 1.0f;
                                    if (lod_opacity_encoded) {
                                        const float lod_alpha = std::max(opacity[i], 0.0f);
                                        if (lod_alpha > 1.0f) {
                                            const float spark_lod_opacity = std::min(lod_alpha * 4.0f - 3.0f, 5.0f);
                                            expansion = 1.0f + 0.7f * (spark_lod_opacity - 1.0f);
                                        }
                                    }
                                    tree->sizes[base + i] = 2.0f * expansion * max_scale;
                                }
                            }
                            for (size_t i = 0; i < chunk_count * 3; ++i) {
                                scales[i] = std::log(std::max(scales[i], 1.0e-8f));
                            }
                        }
                    });

                if (failed.load()) {
                    return std::unexpected(decode_error);
                }

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

                // Attach LOD tree if present
                if (tree && N > 0) {
                    if (const std::uint32_t file_chunk = normalized_lod_file_chunk_size(meta);
                        !supported_lod_file_chunk_size(file_chunk)) {
                        return std::unexpected(std::format(
                            "RAD LOD file uses unsupported {}-splat chunks; expected a multiple of {}.",
                            file_chunk, NATIVE_CHUNK_SIZE));
                    }
                    tree->child_count = std::move(all_child_count);
                    tree->child_start = std::move(all_child_start);
                    // RAD files don't store node depths; derive them in one
                    // forward pass (children always follow their parent in the
                    // BFS-ordered layout).
                    tree->lod_level.assign(N, 0);
                    for (size_t i = 0; i < N; ++i) {
                        const std::uint32_t count = tree->child_count[i];
                        if (count == 0) {
                            continue;
                        }
                        const std::uint32_t start = tree->child_start[i];
                        const auto child_level = static_cast<std::uint8_t>(
                            std::min<std::uint32_t>(tree->lod_level[i] + 1u, 255u));
                        for (std::uint32_t c = 0; c < count; ++c) {
                            const size_t child = static_cast<size_t>(start) + c;
                            if (child > i && child < N) {
                                tree->lod_level[child] = child_level;
                            }
                        }
                    }
                    const size_t chunk_count =
                        (N + lfs::core::SplatLodTree::kChunkSplats - 1) /
                        lfs::core::SplatLodTree::kChunkSplats;
                    tree->chunk_to_page.resize(chunk_count);
                    tree->page_to_chunk.resize(chunk_count);
                    std::iota(tree->chunk_to_page.begin(), tree->chunk_to_page.end(), 0u);
                    std::iota(tree->page_to_chunk.begin(), tree->page_to_chunk.end(), 0u);
                    if (source_path != nullptr && !source_path->empty()) {
                        tree->rad_source.path = *source_path;
                        tree->rad_source.chunk_size = normalized_lod_file_chunk_size(meta);
                        tree->rad_source.metadata_bytes = 8 + pad8(meta_size);
                        tree->rad_source.chunks = std::move(chunk_file_ranges);
                    }
                    tree->lod_opacity_encoded = lod_opacity_encoded;
                    splat_data.lod_tree = std::move(tree);
                }

                return std::expected<SplatData, std::string>(std::move(splat_data));
            }
        };

        // ====================================================================
        // Chunked (range-based) RAD loading
        // ====================================================================

        struct RadFileInfo {
            RadMeta meta;
            std::uint32_t meta_size = 0;
            std::size_t chunk_area_start = 0;
        };

        std::expected<RadFileInfo, std::string> read_rad_file_info(const std::filesystem::path& filepath) {
            std::ifstream in;
            if (!lfs::core::open_file_for_read(filepath, std::ios::binary, in)) {
                return std::unexpected("Failed to open RAD file");
            }
            std::array<std::uint8_t, 8> header{};
            in.read(reinterpret_cast<char*>(header.data()), header.size());
            if (!in.good()) {
                return std::unexpected("Failed to read RAD header");
            }
            if (decode_u32(header.data()) != RAD_MAGIC) {
                return std::unexpected("Invalid RAD magic number");
            }
            const std::uint32_t meta_size = decode_u32(header.data() + 4);
            std::string meta_json(meta_size, '\0');
            in.read(meta_json.data(), meta_size);
            if (!in.good()) {
                return std::unexpected("Failed to read RAD metadata");
            }
            const std::size_t actual_size = meta_json.find_last_not_of(' ');
            if (actual_size != std::string::npos) {
                meta_json.resize(actual_size + 1);
            }

            RadFileInfo info;
            try {
                info.meta = RadMeta::from_json(nlohmann::json::parse(meta_json));
            } catch (const std::exception& e) {
                return std::unexpected(std::string("Failed to parse RAD metadata: ") + e.what());
            }
            info.meta_size = meta_size;
            info.chunk_area_start = 8 + pad8(meta_size);
            return info;
        }

        // True when the chunk index covers [0, count) contiguously, which the
        // parallel range-based reader requires.
        bool rad_ranges_usable(const RadMeta& meta) {
            std::uint64_t cursor = 0;
            for (const auto& range : meta.chunks) {
                if (!range.base.has_value() || !range.count.has_value() ||
                    range.bytes == 0 || range.filename.has_value()) {
                    return false;
                }
                if (*range.base != cursor || *range.count == 0) {
                    return false;
                }
                cursor += *range.count;
            }
            return cursor == meta.count && cursor > 0;
        }

        struct ParsedChunkHeader {
            RadChunkMeta meta;
            std::size_t payload_start = 0; // relative to chunk start
            std::size_t chunk_end = 0;     // relative to chunk start
            bool has_payload_prefix = false;
        };

        std::expected<ParsedChunkHeader, std::string> parse_rad_chunk_header(
            const std::uint8_t* data, const std::size_t size) {
            if (size < 8) {
                return std::unexpected("RAD chunk too small");
            }
            if (decode_u32(data) != RAD_CHUNK_MAGIC) {
                return std::unexpected("Invalid RAD chunk magic");
            }
            const std::uint32_t meta_size = decode_u32(data + 4);
            const std::size_t meta_padded = pad8(meta_size);
            if (8 + meta_padded + 8 > size) {
                return std::unexpected("Unexpected end of RAD chunk metadata");
            }

            ParsedChunkHeader parsed;
            try {
                const std::string chunk_json(reinterpret_cast<const char*>(data + 8), meta_size);
                parsed.meta = RadChunkMeta::from_json(nlohmann::json::parse(chunk_json));
            } catch (const std::exception& e) {
                return std::unexpected(std::string("Failed to parse RAD chunk metadata: ") + e.what());
            }

            const std::size_t payload_size_offset = 8 + meta_padded;
            bool has_prefix = false;
            std::size_t payload_start = payload_size_offset;
            std::size_t chunk_end = 0;
            if (payload_size_offset + 8 <= size) {
                const std::uint64_t payload_bytes = decode_u64(data + payload_size_offset);
                payload_start = payload_size_offset + 8;
                chunk_end = payload_start + static_cast<std::size_t>(payload_bytes);
                has_prefix = (chunk_end <= size) && (parsed.meta.payload_bytes == payload_bytes);
            }
            if (!has_prefix) {
                payload_start = 0;
                chunk_end = pad8(static_cast<std::size_t>(parsed.meta.payload_bytes));
                if (chunk_end > size) {
                    return std::unexpected("RAD chunk payload exceeds chunk bounds");
                }
            }
            parsed.payload_start = payload_start;
            parsed.chunk_end = chunk_end;
            parsed.has_payload_prefix = has_prefix;
            return parsed;
        }

        // --------------------------------------------------------------------
        // Node-metadata sidecar (<file>.rad.meta) internals
        // --------------------------------------------------------------------

        constexpr std::uint32_t RAD_META_MAGIC = 0x4D52464Cu; // 'LFRM'
        // v2: quantized 20 B/node planes (RadMetaBoundsQ + RadMetaLinksQ with
        // per-chunk dequant frames). v1 sidecars are invalid and rebuilt.
        constexpr std::uint32_t RAD_META_VERSION = 2;
        constexpr std::uint32_t RAD_META_ENDIAN_SENTINEL = 0x01020304u;
        constexpr std::size_t RAD_META_HEADER_BYTES = 4096;
        constexpr std::size_t RAD_META_PLANE_ALIGN = 4096;

        // All fields naturally aligned; layout is identical across MSVC/GCC.
        struct RadMetaHeader {
            std::uint32_t magic = 0;
            std::uint32_t version = 0;
            std::uint32_t endian_sentinel = 0;
            std::uint32_t chunk_size = 0;
            std::uint64_t node_count = 0;
            std::uint64_t chunk_count = 0;
            std::uint64_t leaf_count = 0;
            std::uint64_t source_file_size = 0;
            std::uint64_t source_mtime = 0;
            std::uint64_t source_header_hash = 0;
            std::uint64_t bounds_offset = 0;
            std::uint64_t links_offset = 0;
            std::uint64_t chunk_table_offset = 0;
            std::uint8_t lod_opacity_encoded = 0;
            std::uint8_t complete = 0;
        };
        static_assert(sizeof(RadMetaHeader) == 96);
        static_assert(sizeof(RadMetaHeader) <= RAD_META_HEADER_BYTES);

        using RadMetaChunkEntry = lfs::core::RadMetaChunkRecord;

        std::size_t alignPlane(const std::size_t bytes) {
            return (bytes + RAD_META_PLANE_ALIGN - 1) / RAD_META_PLANE_ALIGN * RAD_META_PLANE_ALIGN;
        }

        std::uint64_t fnv1a(const std::uint8_t* const data, const std::size_t size) {
            std::uint64_t hash = 0xcbf29ce484222325ull;
            for (std::size_t i = 0; i < size; ++i) {
                hash ^= data[i];
                hash *= 0x100000001b3ull;
            }
            return hash;
        }

        struct RadSourceStamp {
            std::uint64_t file_size = 0;
            std::uint64_t mtime = 0;
            std::uint64_t header_hash = 0;
        };

        std::expected<RadSourceStamp, std::string> radSourceStamp(
            const std::filesystem::path& rad_path,
            const std::size_t chunk_area_start) {
            RadSourceStamp stamp;
            std::error_code ec;
            stamp.file_size = std::filesystem::file_size(rad_path, ec);
            if (ec) {
                return std::unexpected(std::format("Failed to stat RAD file: {}", ec.message()));
            }
            const auto mtime = std::filesystem::last_write_time(rad_path, ec);
            if (ec) {
                return std::unexpected(std::format("Failed to read RAD mtime: {}", ec.message()));
            }
            stamp.mtime = static_cast<std::uint64_t>(mtime.time_since_epoch().count());

            std::ifstream in;
            if (!lfs::core::open_file_for_read(rad_path, std::ios::binary, in)) {
                return std::unexpected("Failed to open RAD file for header hash");
            }
            std::vector<std::uint8_t> header(chunk_area_start);
            in.read(reinterpret_cast<char*>(header.data()),
                    static_cast<std::streamsize>(header.size()));
            if (!in.good()) {
                return std::unexpected("Failed to read RAD header for hash");
            }
            stamp.header_hash = fnv1a(header.data(), header.size());
            return stamp;
        }

        // Per-chunk sidecar quantization shared by the standalone builder and
        // the inline converter path. Both feed decoded chunk values, so the
        // resulting planes are bit-identical regardless of which path ran.
        void quantizeRadMetaChunk(const std::size_t base,
                                  const std::size_t count,
                                  const float* const means,
                                  const float* const scales,
                                  const float* const alpha,
                                  const std::uint16_t* const child_count,
                                  const std::uint32_t* const child_start,
                                  const bool lod_opacity_encoded,
                                  std::vector<float>& sizes_scratch,
                                  lfs::core::RadMetaBoundsQ* const out_bounds,
                                  lfs::core::RadMetaLinksQ* const out_links,
                                  RadMetaChunkEntry& out_entry) {
            // Pass 1: float centers/sizes + the chunk's quantization frame.
            constexpr float kSizeFloor = 1e-20f;
            sizes_scratch.resize(count);
            glm::vec3 bbox_min{std::numeric_limits<float>::max()};
            glm::vec3 bbox_max{std::numeric_limits<float>::lowest()};
            float log_min = std::numeric_limits<float>::max();
            float log_max = std::numeric_limits<float>::lowest();
            for (std::size_t i = 0; i < count; ++i) {
                const float max_scale = std::max({scales[i * 3 + 0],
                                                  scales[i * 3 + 1],
                                                  scales[i * 3 + 2]});
                float expansion = 1.0f;
                if (lod_opacity_encoded) {
                    const float lod_alpha = std::max(alpha[i], 0.0f);
                    if (lod_alpha > 1.0f) {
                        const float spark_lod_opacity =
                            std::min(lod_alpha * 4.0f - 3.0f, 5.0f);
                        expansion = 1.0f + 0.7f * (spark_lod_opacity - 1.0f);
                    }
                }
                const float size = std::max(2.0f * expansion * max_scale, kSizeFloor);
                sizes_scratch[i] = size;
                const glm::vec3 center{means[i * 3 + 0],
                                       means[i * 3 + 1],
                                       means[i * 3 + 2]};
                bbox_min = glm::min(bbox_min, center);
                bbox_max = glm::max(bbox_max, center);
                log_min = std::min(log_min, std::log(size));
                log_max = std::max(log_max, std::log(size));
            }
            const glm::vec3 extent = bbox_max - bbox_min;
            const float log_range = log_max - log_min;

            // Pass 2: quantize against the frame.
            const auto quant = [](const float v, const float lo, const float range) {
                if (!(range > 0.0f)) {
                    return std::uint16_t{0};
                }
                const float t = std::clamp((v - lo) / range, 0.0f, 1.0f);
                return static_cast<std::uint16_t>(std::lround(t * 65535.0f));
            };
            for (std::size_t i = 0; i < count; ++i) {
                out_bounds[i] = {
                    .qx = quant(means[i * 3 + 0], bbox_min.x, extent.x),
                    .qy = quant(means[i * 3 + 1], bbox_min.y, extent.y),
                    .qz = quant(means[i * 3 + 2], bbox_min.z, extent.z),
                    .qsize = quant(std::log(sizes_scratch[i]), log_min, log_range),
                };

                const std::uint32_t cc = child_count[i];
                const std::uint32_t logical = static_cast<std::uint32_t>(base + i);
                std::uint32_t flags = 0u;
                if (cc == 0u) {
                    flags |= 1u;
                }
                if (logical == 0u) {
                    flags |= 2u;
                }
                if (lod_opacity_encoded) {
                    flags |= 4u;
                }
                out_links[i] = {
                    .child_start = child_start[i],
                    .packed = (cc & 0xffffu) | ((flags & 0xffu) << 24u),
                    .parent = lfs::core::SplatLodTree::kInvalidPage,
                };
            }

            out_entry.bbox_min[0] = bbox_min.x;
            out_entry.bbox_min[1] = bbox_min.y;
            out_entry.bbox_min[2] = bbox_min.z;
            out_entry.bbox_extent[0] = extent.x;
            out_entry.bbox_extent[1] = extent.y;
            out_entry.bbox_extent[2] = extent.z;
            out_entry.log_size_min = log_min;
            out_entry.log_size_range = log_range;
        }

        // Phase 2 of sidecar builds: block-stream the links plane patching
        // parent and level via forward scatter. Children always follow their
        // parent in the BFS layout, so each node's entry is final when
        // reached. Returns the leaf count.
        Result<std::uint64_t> patchRadMetaLinksPlane(
            std::fstream& rmw,
            const std::uint64_t links_offset,
            const std::uint64_t n,
            const std::filesystem::path& rad_path,
            const std::filesystem::path& tmp_path,
            const std::function<bool(float)>& progress) {
            std::vector<std::uint32_t> parent;
            std::vector<std::uint8_t> level;
            try {
                parent.assign(n, lfs::core::SplatLodTree::kInvalidPage);
                level.assign(n, 0);
            } catch (const std::bad_alloc&) {
                return make_error(ErrorCode::WRITE_FAILURE,
                                  std::format("Not enough RAM for sidecar parent scatter ({} nodes)", n),
                                  rad_path);
            }
            std::uint64_t leaf_count = 0;
            std::uint64_t assigned = 0;
            constexpr std::size_t kBlockRecords = std::size_t{4} << 20;
            std::vector<lfs::core::RadMetaLinksQ> block(std::min<std::size_t>(kBlockRecords, n));
            for (std::uint64_t first = 0; first < n; first += kBlockRecords) {
                const std::size_t count = static_cast<std::size_t>(
                    std::min<std::uint64_t>(kBlockRecords, n - first));
                const std::streamoff offset = static_cast<std::streamoff>(
                    links_offset + first * sizeof(lfs::core::RadMetaLinksQ));
                rmw.clear();
                rmw.seekg(offset, std::ios::beg);
                rmw.read(reinterpret_cast<char*>(block.data()),
                         static_cast<std::streamsize>(count * sizeof(lfs::core::RadMetaLinksQ)));
                if (!rmw.good()) {
                    return make_error(ErrorCode::READ_FAILURE,
                                      std::format("Failed to read sidecar links block: {}", io_error_detail()),
                                      tmp_path);
                }
                for (std::size_t k = 0; k < count; ++k) {
                    const std::uint64_t i = first + k;
                    auto& rec = block[k];
                    rec.parent = parent[i];
                    rec.packed = (rec.packed & 0xff00ffffu) |
                                 ((static_cast<std::uint32_t>(level[i]) & 0xffu) << 16u);
                    const std::uint32_t cc = rec.packed & 0xffffu;
                    if (cc == 0) {
                        ++leaf_count;
                        continue;
                    }
                    const std::uint64_t cs = rec.child_start;
                    if (cs <= i || cs + cc > n) {
                        return make_error(ErrorCode::CORRUPTED_DATA,
                                          std::format("corrupt LOD layout: node {} children [{}, {})",
                                                      i, cs, cs + cc),
                                          rad_path);
                    }
                    const std::uint8_t child_level =
                        static_cast<std::uint8_t>(std::min<std::uint32_t>(level[i] + 1u, 255u));
                    for (std::uint32_t c = 0; c < cc; ++c) {
                        parent[cs + c] = static_cast<std::uint32_t>(i);
                        level[cs + c] = child_level;
                    }
                    assigned += cc;
                }
                rmw.clear();
                rmw.seekp(offset, std::ios::beg);
                rmw.write(reinterpret_cast<const char*>(block.data()),
                          static_cast<std::streamsize>(count * sizeof(lfs::core::RadMetaLinksQ)));
                if (!rmw.good()) {
                    return make_error(ErrorCode::WRITE_FAILURE,
                                      std::format("Failed to write sidecar links block: {}", io_error_detail()),
                                      tmp_path);
                }
                if (progress != nullptr &&
                    !progress(static_cast<float>(first + count) / static_cast<float>(n))) {
                    return make_error(ErrorCode::CANCELLED, "Sidecar build cancelled", rad_path);
                }
            }
            if (assigned != n - 1) {
                return make_error(ErrorCode::CORRUPTED_DATA,
                                  std::format("corrupt LOD layout: {} of {} nodes have a parent",
                                              assigned, n - 1),
                                  rad_path);
            }
            return leaf_count;
        }

        std::size_t available_host_memory_bytes() {
#ifdef _WIN32
            MEMORYSTATUSEX status{};
            status.dwLength = sizeof(status);
            if (GlobalMemoryStatusEx(&status)) {
                return static_cast<std::size_t>(status.ullAvailPhys);
            }
#else
            std::ifstream meminfo("/proc/meminfo");
            std::string line;
            while (std::getline(meminfo, line)) {
                if (line.rfind("MemAvailable:", 0) == 0) {
                    std::uint64_t kb = 0;
                    std::sscanf(line.c_str(), "MemAvailable: %lu kB", &kb);
                    return static_cast<std::size_t>(kb) * 1024;
                }
            }
#endif
            return std::size_t{8} * 1024 * 1024 * 1024;
        }

        // Range-based parallel decoder. Reads chunks individually (no whole-file
        // buffer) and materializes leaf payload tensors only for the first
        // `payload_count` nodes; the LOD tree and chunk ranges always cover all
        // nodes so the renderer can stream the rest from disk.
        std::expected<SplatData, std::string> decode_rad_chunked(
            const std::filesystem::path& filepath,
            const RadFileInfo& info,
            const std::size_t payload_count,
            const lfs::core::SplatLodTree::NodeMetaView* const meta_view = nullptr) {
            const RadMeta& meta = info.meta;
            const int max_sh = meta.max_sh.value_or(0);
            const int sh_coeffs = max_sh > 0 ? SH_COEFFS_FOR_DEGREE[max_sh] : 0;
            const bool has_lod_tree = meta.lod_tree.value_or(false);
            const std::size_t N = meta.count;
            if (payload_count < N && !has_lod_tree) {
                return std::unexpected("Partial RAD payload requires a LOD tree");
            }
            const bool use_view = meta_view != nullptr && meta_view->valid() && has_lod_tree;
            if (use_view && meta_view->node_count != N) {
                return std::unexpected("RAD metadata sidecar node count mismatch");
            }

            bool lod_opacity_encoded = has_lod_tree;
            if (meta.splat_encoding.has_value()) {
                const auto& enc = meta.splat_encoding.value();
                if (enc.is_object()) {
                    auto it = enc.find("lodOpacity");
                    if (it != enc.end() && it->is_boolean()) {
                        lod_opacity_encoded = it->get<bool>();
                    }
                }
            }

            Tensor means_tensor = Tensor::empty({payload_count, 3}, Device::CPU, lfs::core::DataType::Float32);
            Tensor opacity_tensor = Tensor::empty({payload_count, 1}, Device::CPU, lfs::core::DataType::Float32);
            Tensor sh0_tensor = Tensor::empty({payload_count, 1, 3}, Device::CPU, lfs::core::DataType::Float32);
            Tensor scales_tensor = Tensor::empty({payload_count, 3}, Device::CPU, lfs::core::DataType::Float32);
            Tensor rotation_tensor = Tensor::empty({payload_count, 4}, Device::CPU, lfs::core::DataType::Float32);
            Tensor shN_tensor;
            if (sh_coeffs > 0) {
                shN_tensor = Tensor::empty({payload_count, static_cast<std::size_t>(sh_coeffs), 3},
                                           Device::CPU, lfs::core::DataType::Float32);
            }

            float* const all_means = means_tensor.ptr<float>();
            float* const all_opacity = opacity_tensor.ptr<float>();
            float* const all_sh0 = sh0_tensor.ptr<float>();
            float* const all_scales = scales_tensor.ptr<float>();
            float* const all_rotation = rotation_tensor.ptr<float>();
            float* const all_shN = sh_coeffs > 0 ? shN_tensor.ptr<float>() : nullptr;

            std::vector<uint16_t> all_child_count;
            std::vector<uint32_t> all_child_start;
            auto tree = has_lod_tree ? std::make_unique<lfs::core::SplatLodTree>() : nullptr;
            std::vector<lfs::core::SplatLodTree::ChunkFileRange> chunk_file_ranges;
            if (has_lod_tree) {
                if (!use_view) {
                    all_child_count.resize(N);
                    all_child_start.resize(N);
                    tree->centers.resize(N);
                    tree->sizes.resize(N);
                }
                chunk_file_ranges.resize(native_lod_chunk_count(N));
            }

            // With a sidecar the per-node metadata stays on disk; chunk payload
            // splits come from its table and non-resident chunks need no reads.
            const RadMetaChunkEntry* sidecar_chunk_table = nullptr;
            if (use_view) {
                RadMetaHeader sidecar_header{};
                std::memcpy(&sidecar_header, meta_view->file->data(), sizeof(sidecar_header));
                if (sidecar_header.chunk_count != native_lod_chunk_count(N)) {
                    return std::unexpected("RAD metadata sidecar chunk count mismatch");
                }
                sidecar_chunk_table = reinterpret_cast<const RadMetaChunkEntry*>(
                    meta_view->file->data() + sidecar_header.chunk_table_offset);
                for (std::size_t ci = 0; ci < meta.chunks.size(); ++ci) {
                    const auto& range = meta.chunks[ci];
                    if (!range.base.has_value() || !range.count.has_value()) {
                        return std::unexpected("RAD chunk index lacks base/count");
                    }
                    const std::uint64_t file_base = *range.base;
                    const std::uint64_t file_count = *range.count;
                    const std::uint64_t file_end = file_base + file_count;
                    std::uint64_t logical_base =
                        (file_base / NATIVE_CHUNK_SIZE) * static_cast<std::uint64_t>(NATIVE_CHUNK_SIZE);
                    if (logical_base < file_base) {
                        logical_base += NATIVE_CHUNK_SIZE;
                    }
                    for (; logical_base < file_end; logical_base += NATIVE_CHUNK_SIZE) {
                        const std::size_t logical_chunk =
                            static_cast<std::size_t>(logical_base / NATIVE_CHUNK_SIZE);
                        if (logical_chunk >= chunk_file_ranges.size()) {
                            break;
                        }
                        const auto& page_record = sidecar_chunk_table[logical_chunk];
                        chunk_file_ranges[logical_chunk] = {
                            .file_offset = info.chunk_area_start + static_cast<std::size_t>(range.offset),
                            .file_bytes = range.bytes,
                            .payload_offset = page_record.payload_offset,
                            .payload_bytes = page_record.payload_bytes,
                            .file_base = file_base,
                            .file_count = file_count,
                            .base = logical_base,
                            .count = std::min<std::uint64_t>(
                                NATIVE_CHUNK_SIZE, file_end - logical_base),
                        };
                    }
                }
            }

            std::atomic<bool> failed{false};
            std::mutex error_mutex;
            std::string decode_error;
            auto record_error = [&](std::string msg) {
                failed.store(true, std::memory_order_relaxed);
                std::lock_guard<std::mutex> lock(error_mutex);
                if (decode_error.empty()) {
                    decode_error = std::move(msg);
                }
            };

            struct ChunkScratch {
                std::ifstream stream;
                bool stream_ok = false;
                std::vector<std::uint8_t> raw;
                std::vector<float> means;
                std::vector<float> scales;
                std::vector<float> alpha;
            };
            tbb::enumerable_thread_specific<ChunkScratch> scratch_tls;

            tbb::parallel_for(
                tbb::blocked_range<std::size_t>(0, meta.chunks.size(), 1),
                [&](const tbb::blocked_range<std::size_t>& chunk_range) {
                    auto& scratch = scratch_tls.local();
                    if (!scratch.stream_ok) {
                        if (!lfs::core::open_file_for_read(filepath, std::ios::binary, scratch.stream)) {
                            record_error("Failed to open RAD file for chunk read");
                            return;
                        }
                        scratch.stream_ok = true;
                    }

                    for (std::size_t ci = chunk_range.begin(); ci != chunk_range.end(); ++ci) {
                        if (failed.load(std::memory_order_relaxed)) {
                            return;
                        }
                        const auto& range = meta.chunks[ci];
                        const std::size_t base = static_cast<std::size_t>(*range.base);
                        const std::size_t count = static_cast<std::size_t>(*range.count);
                        const std::size_t bytes = static_cast<std::size_t>(range.bytes);
                        const std::uint64_t file_offset =
                            info.chunk_area_start + static_cast<std::size_t>(range.offset);

                        // Sidecar-backed loads need nothing from non-resident
                        // chunks: tree metadata is in the view and chunk ranges
                        // were prebuilt from its table.
                        if (use_view && base + count > payload_count) {
                            continue;
                        }

                        scratch.raw.resize(bytes);
                        scratch.stream.clear();
                        scratch.stream.seekg(static_cast<std::streamoff>(file_offset), std::ios::beg);
                        scratch.stream.read(reinterpret_cast<char*>(scratch.raw.data()),
                                            static_cast<std::streamsize>(bytes));
                        if (!scratch.stream.good()) {
                            record_error("Failed to read RAD chunk from file");
                            return;
                        }

                        auto parsed = parse_rad_chunk_header(scratch.raw.data(), scratch.raw.size());
                        if (!parsed) {
                            record_error(parsed.error());
                            return;
                        }
                        const RadChunkMeta& chunk = parsed->meta;
                        if (chunk.base != base || chunk.count != count) {
                            record_error("RAD chunk header disagrees with chunk index");
                            return;
                        }

                        const bool resident = base + count <= payload_count;
                        float* means_dst = nullptr;
                        float* opacity_dst = nullptr;
                        float* sh0_dst = nullptr;
                        float* scales_dst = nullptr;
                        float* rotation_dst = nullptr;
                        float* shN_dst = nullptr;
                        if (resident) {
                            means_dst = all_means + base * 3;
                            opacity_dst = all_opacity + base;
                            sh0_dst = all_sh0 + base * 3;
                            scales_dst = all_scales + base * 3;
                            rotation_dst = all_rotation + base * 4;
                            shN_dst = all_shN != nullptr
                                          ? all_shN + base * static_cast<std::size_t>(sh_coeffs) * 3
                                          : nullptr;
                            std::memset(means_dst, 0, count * 3 * sizeof(float));
                            std::memset(opacity_dst, 0, count * sizeof(float));
                            std::memset(sh0_dst, 0, count * 3 * sizeof(float));
                            std::memset(scales_dst, 0, count * 3 * sizeof(float));
                            std::memset(rotation_dst, 0, count * 4 * sizeof(float));
                            if (shN_dst != nullptr) {
                                std::memset(shN_dst, 0,
                                            count * static_cast<std::size_t>(sh_coeffs) * 3 * sizeof(float));
                            }
                        } else {
                            // Non-resident chunks only feed the LOD tree:
                            // centers/sizes need means, scales, and alpha.
                            scratch.means.assign(count * 3, 0.0f);
                            scratch.scales.assign(count * 3, 0.0f);
                            scratch.alpha.assign(count, 0.0f);
                            means_dst = scratch.means.data();
                            scales_dst = scratch.scales.data();
                            opacity_dst = scratch.alpha.data();
                        }

                        auto err = decode_chunk_properties(
                            scratch.raw.data(), chunk, 0, parsed->payload_start,
                            parsed->has_payload_prefix, parsed->chunk_end, sh_coeffs,
                            means_dst, opacity_dst, sh0_dst, scales_dst, rotation_dst, shN_dst,
                            has_lod_tree && !use_view ? all_child_count.data() + base : nullptr,
                            has_lod_tree && !use_view ? all_child_start.data() + base : nullptr);
                        if (err.has_value()) {
                            record_error(std::move(*err));
                            return;
                        }

                        // RAD stores display-space values; convert what we keep.
                        if (resident) {
                            for (std::size_t i = 0; i < count * 3; ++i) {
                                sh0_dst[i] = (sh0_dst[i] - 0.5f) / SH_C0;
                            }
                        }
                        if (!lod_opacity_encoded) {
                            for (std::size_t i = 0; i < count; ++i) {
                                const float a = std::clamp(opacity_dst[i], 1.0e-6f, 1.0f - 1.0e-6f);
                                opacity_dst[i] = std::log(a / (1.0f - a));
                            }
                        } else {
                            for (std::size_t i = 0; i < count; ++i) {
                                opacity_dst[i] = std::max(opacity_dst[i], 0.0f);
                            }
                        }
                        if (tree && !use_view) {
                            for (std::size_t i = 0; i < count; ++i) {
                                tree->centers[base + i] = glm::vec3(means_dst[i * 3 + 0],
                                                                    means_dst[i * 3 + 1],
                                                                    means_dst[i * 3 + 2]);
                                const float max_scale = std::max({scales_dst[i * 3 + 0],
                                                                  scales_dst[i * 3 + 1],
                                                                  scales_dst[i * 3 + 2]});
                                float expansion = 1.0f;
                                if (lod_opacity_encoded) {
                                    const float lod_alpha = std::max(opacity_dst[i], 0.0f);
                                    if (lod_alpha > 1.0f) {
                                        const float spark_lod_opacity =
                                            std::min(lod_alpha * 4.0f - 3.0f, 5.0f);
                                        expansion = 1.0f + 0.7f * (spark_lod_opacity - 1.0f);
                                    }
                                }
                                tree->sizes[base + i] = 2.0f * expansion * max_scale;
                            }
                            assign_native_chunk_ranges(
                                chunk_file_ranges,
                                file_offset,
                                static_cast<std::uint64_t>(parsed->chunk_end),
                                file_offset + parsed->payload_start,
                                chunk.payload_bytes,
                                chunk.base,
                                chunk.count);
                        }
                        if (resident) {
                            for (std::size_t i = 0; i < count * 3; ++i) {
                                scales_dst[i] = std::log(std::max(scales_dst[i], 1.0e-8f));
                            }
                        }
                    }
                });

            if (failed.load()) {
                return std::unexpected(decode_error);
            }

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

            if (tree && N > 0) {
                if (const std::uint32_t file_chunk = normalized_lod_file_chunk_size(meta);
                    !supported_lod_file_chunk_size(file_chunk)) {
                    return std::unexpected(std::format(
                        "RAD LOD file uses unsupported {}-splat chunks; expected a multiple of {}.",
                        file_chunk, NATIVE_CHUNK_SIZE));
                }
                if (use_view) {
                    tree->meta_view = *meta_view;
                } else {
                    tree->child_count = std::move(all_child_count);
                    tree->child_start = std::move(all_child_start);
                    // Derive node depths in one forward pass (children always
                    // follow their parent in the BFS-ordered layout).
                    tree->lod_level.assign(N, 0);
                    for (std::size_t i = 0; i < N; ++i) {
                        const std::uint32_t count = tree->child_count[i];
                        if (count == 0) {
                            continue;
                        }
                        const std::uint32_t start = tree->child_start[i];
                        const auto child_level = static_cast<std::uint8_t>(
                            std::min<std::uint32_t>(tree->lod_level[i] + 1u, 255u));
                        for (std::uint32_t c = 0; c < count; ++c) {
                            const std::size_t child = static_cast<std::size_t>(start) + c;
                            if (child > i && child < N) {
                                tree->lod_level[child] = child_level;
                            }
                        }
                    }
                }
                const std::size_t chunk_count =
                    (N + lfs::core::SplatLodTree::kChunkSplats - 1) /
                    lfs::core::SplatLodTree::kChunkSplats;
                tree->chunk_to_page.resize(chunk_count);
                tree->page_to_chunk.resize(chunk_count);
                std::iota(tree->chunk_to_page.begin(), tree->chunk_to_page.end(), 0u);
                std::iota(tree->page_to_chunk.begin(), tree->page_to_chunk.end(), 0u);
                tree->rad_source.path = filepath;
                tree->rad_source.chunk_size = normalized_lod_file_chunk_size(meta);
                tree->rad_source.metadata_bytes = info.chunk_area_start;
                tree->rad_source.chunks = std::move(chunk_file_ranges);
                tree->lod_opacity_encoded = lod_opacity_encoded;
                splat_data.lod_tree = std::move(tree);
            }

            return std::expected<SplatData, std::string>(std::move(splat_data));
        }

    } // namespace

    // ============================================================================
    // Public API Implementation
    // ============================================================================

    std::expected<SplatData, std::string> load_rad(const std::filesystem::path& filepath) {
        return load_rad(filepath, RadLoadOverrides{});
    }

    std::expected<SplatData, std::string> load_rad(const std::filesystem::path& filepath,
                                                   const RadLoadOverrides& overrides) {
        auto start = std::chrono::high_resolution_clock::now();

        LOG_INFO("Loading RAD file: {}", lfs::core::path_to_utf8(filepath));

        // Preferred path: range-based parallel chunk reads without a whole-file
        // buffer, with optional out-of-core payload for huge LOD models.
        if (auto info = read_rad_file_info(filepath); info && rad_ranges_usable(info->meta)) {
            const RadMeta& meta = info->meta;
            const std::size_t N = meta.count;
            const int max_sh = meta.max_sh.value_or(0);
            const int sh_coeffs = max_sh > 0 ? SH_COEFFS_FOR_DEGREE[max_sh] : 0;
            const bool has_lod_tree = meta.lod_tree.value_or(false);

            std::size_t payload_count = N;
            if (has_lod_tree && meta.chunks.size() > 1) {
                // Per node: 14 base floats (+3 per SH coeff) of tensor payload,
                // 23 bytes of host tree metadata (child links, bounds, ranges).
                const std::size_t payload_bytes =
                    N * (56 + static_cast<std::size_t>(sh_coeffs) * 12);
                const std::size_t tree_bytes = N * 23;
                const bool out_of_core = overrides.out_of_core.value_or(
                    payload_bytes + tree_bytes > available_host_memory_bytes() / 2);
                if (out_of_core) {
                    // Chunk count keeps the preview a clean prefix; every
                    // chunk is either fully resident or tree-only.
                    const std::size_t file_chunk =
                        normalized_lod_file_chunk_size(meta);
                    std::size_t preview =
                        overrides.preview_splats.value_or(2048 * NATIVE_CHUNK_SIZE);
                    preview = ((preview + file_chunk - 1) / file_chunk) * file_chunk;
                    payload_count = std::min<std::size_t>(N, preview);
                    LOG_INFO("RAD out-of-core load: {} nodes, keeping {} resident "
                             "(coarsest LOD prefix); leaves stream from disk",
                             N, payload_count);
                }
            }

            // Out-of-core models keep tree metadata on disk via the sidecar:
            // host RAM stays O(chunks) and cached re-opens skip decoding every
            // chunk. The sidecar is required — in-RAM tree metadata does not
            // scale to the model sizes that stream out of core.
            lfs::core::SplatLodTree::NodeMetaView meta_view;
            if (payload_count < N) {
                auto view = open_rad_meta_sidecar(filepath);
                if (!view) {
                    LOG_INFO("RAD metadata sidecar unavailable ({}); building", view.error());
                    if (auto built = build_rad_meta_sidecar(filepath); built) {
                        view = open_rad_meta_sidecar(filepath);
                    } else {
                        return std::unexpected(std::format(
                            "RAD metadata sidecar build failed: {}", built.error().message));
                    }
                }
                if (!view) {
                    return std::unexpected(std::format(
                        "RAD metadata sidecar unusable: {}", view.error()));
                }
                meta_view = std::move(*view);
            }

            auto result = decode_rad_chunked(filepath, *info, payload_count,
                                             meta_view.valid() ? &meta_view : nullptr);
            if (result) {
                const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - start);
                LOG_INFO("RAD loaded: {} of {} nodes resident with SH degree {} in {}ms",
                         result->size(), N, result->get_max_sh_degree(), elapsed.count());
                return result;
            }
            LOG_WARN("RAD chunked load failed ({}); falling back to buffered decode",
                     result.error());
        }

        // Fallback: whole-file buffered decode for legacy or non-indexed files.
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

        RadDecoder decoder;
        auto result = decoder.decode(data, &filepath);

        if (!result) {
            return result;
        }

        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start);

        LOG_INFO("RAD loaded: {} gaussians with SH degree {} in {}ms",
                 result->size(), result->get_max_sh_degree(), elapsed.count());

        return result;
    }

    std::expected<RadChunkInfo, std::string> decode_rad_chunk_into(
        const std::span<const std::uint8_t> data,
        const int fallback_max_sh,
        bool lod_opacity_encoded,
        const std::size_t dst_capacity,
        const RadChunkDsts& dsts) {
        auto parsed = parse_rad_chunk_header(data.data(), data.size());
        if (!parsed) {
            return std::unexpected(parsed.error());
        }
        const RadChunkMeta& chunk = parsed->meta;
        if (chunk.splat_encoding.has_value() && chunk.splat_encoding->is_object()) {
            const auto it = chunk.splat_encoding->find("lodOpacity");
            if (it != chunk.splat_encoding->end() && it->is_boolean()) {
                lod_opacity_encoded = it->get<bool>();
            }
        }
        const int max_sh = std::clamp(chunk.max_sh > 0 ? chunk.max_sh : fallback_max_sh, 0, 3);
        const int sh_coeffs = max_sh > 0 ? SH_COEFFS_FOR_DEGREE[max_sh] : 0;
        const std::size_t count = static_cast<std::size_t>(chunk.count);
        if (count > dst_capacity) {
            return std::unexpected(std::format(
                "RAD chunk holds {} splats, destination capacity is {}", count, dst_capacity));
        }

        float* shN = nullptr;
        if (sh_coeffs > 0 && dsts.shN_canonical != nullptr) {
            dsts.shN_canonical->assign(count * static_cast<std::size_t>(sh_coeffs) * 3u, 0.0f);
            shN = dsts.shN_canonical->data();
        }
        if (auto err = decode_chunk_properties(data.data(),
                                               chunk,
                                               0,
                                               parsed->payload_start,
                                               parsed->has_payload_prefix,
                                               parsed->chunk_end,
                                               sh_coeffs,
                                               dsts.means,
                                               dsts.opacity_raw,
                                               dsts.sh0_raw,
                                               dsts.scaling_raw,
                                               dsts.rotation_raw,
                                               shN,
                                               nullptr,
                                               nullptr);
            err.has_value()) {
            return std::unexpected(std::move(*err));
        }

        // Same post-decode transforms as decode_rad_chunk_buffer, in place.
        if (dsts.sh0_raw != nullptr) {
            for (std::size_t i = 0; i < count * 3u; ++i) {
                dsts.sh0_raw[i] = radmath::sh0Transform(dsts.sh0_raw[i]);
            }
        }
        if (dsts.opacity_raw != nullptr) {
            if (!lod_opacity_encoded) {
                for (std::size_t i = 0; i < count; ++i) {
                    dsts.opacity_raw[i] = radmath::opacityLogit(dsts.opacity_raw[i]);
                }
            } else {
                for (std::size_t i = 0; i < count; ++i) {
                    dsts.opacity_raw[i] = radmath::opacityLodEncoded(dsts.opacity_raw[i]);
                }
            }
        }
        if (dsts.scaling_raw != nullptr) {
            for (std::size_t i = 0; i < count * 3u; ++i) {
                dsts.scaling_raw[i] = radmath::scaleLog(dsts.scaling_raw[i]);
            }
        }

        return RadChunkInfo{
            .base = chunk.base,
            .count = chunk.count,
            .max_sh_degree = max_sh,
            .sh_coeffs_rest = static_cast<std::uint32_t>(sh_coeffs),
            .lod_opacity_encoded = lod_opacity_encoded,
        };
    }

    namespace {

        struct PackedPropInfo {
            RadPackedKind kind;
            std::size_t dims;
        };

        std::optional<PackedPropInfo> packed_prop_info(const std::string& name) {
            if (name == PROP_CENTER) {
                return PackedPropInfo{RadPackedKind::Means, 3};
            }
            if (name == PROP_ALPHA) {
                return PackedPropInfo{RadPackedKind::Alpha, 1};
            }
            if (name == PROP_RGB) {
                return PackedPropInfo{RadPackedKind::Sh0, 3};
            }
            if (name == PROP_SCALES) {
                return PackedPropInfo{RadPackedKind::Scales, 3};
            }
            if (name == PROP_ORIENTATION) {
                return PackedPropInfo{RadPackedKind::Rotation, 3};
            }
            if (name == PROP_SH1) {
                return PackedPropInfo{RadPackedKind::Sh1, 9};
            }
            if (name == PROP_SH2) {
                return PackedPropInfo{RadPackedKind::Sh2, 15};
            }
            if (name == PROP_SH3) {
                return PackedPropInfo{RadPackedKind::Sh3, 21};
            }
            return std::nullopt;
        }

        struct PackedEncodingInfo {
            RadPackedEncoding encoding;
            bool delta;
        };

        std::optional<PackedEncodingInfo> packed_encoding_info(const std::string& encoding) {
            if (encoding == "f32") {
                return PackedEncodingInfo{RadPackedEncoding::F32, false};
            }
            if (encoding == "f32_lebytes") {
                return PackedEncodingInfo{RadPackedEncoding::F32LeBytes, false};
            }
            if (encoding == "f16") {
                return PackedEncodingInfo{RadPackedEncoding::F16, false};
            }
            if (encoding == "f16_lebytes") {
                return PackedEncodingInfo{RadPackedEncoding::F16LeBytes, false};
            }
            if (encoding == "r8") {
                return PackedEncodingInfo{RadPackedEncoding::R8, false};
            }
            if (encoding == "r8_delta") {
                return PackedEncodingInfo{RadPackedEncoding::R8, true};
            }
            if (encoding == "s8") {
                return PackedEncodingInfo{RadPackedEncoding::S8, false};
            }
            if (encoding == "s8_delta") {
                return PackedEncodingInfo{RadPackedEncoding::S8, true};
            }
            if (encoding == "ln_0r8") {
                return PackedEncodingInfo{RadPackedEncoding::Ln0R8, false};
            }
            if (encoding == "ln_f16") {
                return PackedEncodingInfo{RadPackedEncoding::LnF16, false};
            }
            if (encoding == "oct88r8") {
                return PackedEncodingInfo{RadPackedEncoding::Oct88R8, false};
            }
            return std::nullopt;
        }

        std::size_t packed_encoding_element_bytes(const RadPackedEncoding encoding) {
            switch (encoding) {
            case RadPackedEncoding::F32:
            case RadPackedEncoding::F32LeBytes:
                return 4;
            case RadPackedEncoding::F16:
            case RadPackedEncoding::F16LeBytes:
            case RadPackedEncoding::LnF16:
                return 2;
            case RadPackedEncoding::R8:
            case RadPackedEncoding::S8:
            case RadPackedEncoding::Ln0R8:
                return 1;
            case RadPackedEncoding::Oct88R8:
                return 1;
            }
            return 0;
        }

        void copy_packed_plane_window(const std::uint8_t* const src,
                                      const std::size_t source_count,
                                      const std::size_t source_offset,
                                      const std::size_t count,
                                      const PackedPropInfo& info,
                                      const RadPackedEncoding encoding,
                                      std::uint8_t* const dst) {
            if (encoding == RadPackedEncoding::Oct88R8) {
                std::memcpy(dst, src + source_offset * 3, count * 3);
                return;
            }
            const std::size_t elem = packed_encoding_element_bytes(encoding);
            if (encoding == RadPackedEncoding::F32LeBytes ||
                encoding == RadPackedEncoding::F16LeBytes) {
                const std::size_t source_stride = source_count * info.dims;
                const std::size_t dst_stride = count * info.dims;
                for (std::size_t b = 0; b < elem; ++b) {
                    for (std::size_t d = 0; d < info.dims; ++d) {
                        const std::size_t src_off =
                            b * source_stride + d * source_count + source_offset;
                        const std::size_t dst_off = b * dst_stride + d * count;
                        std::memcpy(dst + dst_off, src + src_off, count);
                    }
                }
                return;
            }
            for (std::size_t d = 0; d < info.dims; ++d) {
                const std::size_t src_off = (d * source_count + source_offset) * elem;
                const std::size_t dst_off = d * count * elem;
                std::memcpy(dst + dst_off, src + src_off, count * elem);
            }
        }

    } // namespace

    std::expected<RadPagePackedDesc, std::string> decode_rad_chunk_packed(
        const std::span<const std::uint8_t> data,
        const int fallback_max_sh,
        bool lod_opacity_encoded,
        const std::size_t dst_capacity,
        const lfs::core::SplatLodTree::NodeMetaView& meta_view,
        const std::uint32_t chunk,
        const std::span<std::uint8_t> dst) {
        static_assert(std::is_trivially_copyable_v<RadPagePackedDesc>);

        auto parsed = parse_rad_chunk_header(data.data(), data.size());
        if (!parsed) {
            return std::unexpected(parsed.error());
        }
        const RadChunkMeta& chunk_meta = parsed->meta;
        if (chunk_meta.splat_encoding.has_value() && chunk_meta.splat_encoding->is_object()) {
            const auto it = chunk_meta.splat_encoding->find("lodOpacity");
            if (it != chunk_meta.splat_encoding->end() && it->is_boolean()) {
                lod_opacity_encoded = it->get<bool>();
            }
        }
        const int max_sh = std::clamp(chunk_meta.max_sh > 0 ? chunk_meta.max_sh : fallback_max_sh, 0, 3);
        const int sh_coeffs = max_sh > 0 ? SH_COEFFS_FOR_DEGREE[max_sh] : 0;
        const std::size_t source_count = static_cast<std::size_t>(chunk_meta.count);
        const std::uint64_t logical_base =
            static_cast<std::uint64_t>(chunk) * NATIVE_CHUNK_SIZE;
        const std::uint64_t file_base = chunk_meta.base;
        const std::uint64_t file_end = file_base + source_count;
        if (logical_base < file_base || logical_base >= file_end) {
            return std::unexpected(std::format(
                "RAD logical chunk {} (base {}) is outside file chunk [{}, {})",
                chunk, logical_base, file_base, file_end));
        }
        const std::size_t source_offset = static_cast<std::size_t>(logical_base - file_base);
        std::size_t count = std::min<std::size_t>(
            NATIVE_CHUNK_SIZE, source_count - source_offset);
        if (meta_view.valid()) {
            if (logical_base >= meta_view.node_count) {
                return std::unexpected("RAD logical chunk exceeds sidecar node count");
            }
            count = std::min<std::size_t>(
                count, meta_view.node_count - static_cast<std::size_t>(logical_base));
        }
        if (count == 0 || count > dst_capacity) {
            return std::unexpected(std::format(
                "RAD chunk holds {} splats, destination capacity is {}", count, dst_capacity));
        }

        RadPagePackedDesc desc{};
        desc.count = static_cast<std::uint32_t>(count);
        desc.sh_coeffs_rest = static_cast<std::uint32_t>(sh_coeffs);
        desc.lod_opacity = lod_opacity_encoded ? 1u : 0u;
        desc.chunk = chunk;

        std::size_t cursor = 0;
        const auto alloc_plane = [&](const std::size_t bytes) -> std::optional<std::size_t> {
            const std::size_t at = (cursor + 15u) & ~std::size_t{15u};
            if (at + bytes > dst.size()) {
                return std::nullopt;
            }
            cursor = at + bytes;
            return at;
        };

        std::uint32_t seen_kinds = 0;
        for (const auto& prop : chunk_meta.properties) {
            if (prop.property == PROP_CHILD_COUNT || prop.property == PROP_CHILD_START) {
                continue; // tree links come from the sidecar planes instead
            }
            const auto info = packed_prop_info(prop.property);
            if (!info) {
                return std::unexpected(std::format(
                    "RAD property '{}' is not part of the packed streaming profile", prop.property));
            }
            const auto enc = packed_encoding_info(prop.encoding);
            if (!enc) {
                return std::unexpected(std::format(
                    "RAD encoding '{}' for property '{}' is not part of the packed streaming profile",
                    prop.encoding, prop.property));
            }
            const std::uint32_t kind_bit = 1u << static_cast<std::uint32_t>(info->kind);
            if ((seen_kinds & kind_bit) != 0u) {
                return std::unexpected(std::format("Duplicate RAD property '{}'", prop.property));
            }
            seen_kinds |= kind_bit;
            if (desc.property_count >= kRadPackedMaxProps) {
                return std::unexpected("RAD chunk exceeds packed property limit");
            }

            const std::size_t source_plane_bytes =
                rad_property_decoded_bytes(prop.property, prop.encoding, source_count);
            const std::size_t plane_bytes =
                rad_property_decoded_bytes(prop.property, prop.encoding, count);
            if (source_plane_bytes == 0 || plane_bytes == 0) {
                return std::unexpected(std::format(
                    "Cannot size RAD property '{}' ({})", prop.property, prop.encoding));
            }
            const auto plane_offset = alloc_plane(plane_bytes);
            if (!plane_offset) {
                return std::unexpected("RAD packed staging slot too small for chunk planes");
            }

            const std::size_t prop_bytes = static_cast<std::size_t>(prop.bytes);
            const std::size_t absolute_offset =
                parsed->has_payload_prefix
                    ? parsed->payload_start + static_cast<std::size_t>(prop.offset)
                    : static_cast<std::size_t>(prop.offset);
            if (absolute_offset + prop_bytes > parsed->chunk_end || absolute_offset + prop_bytes > data.size()) {
                return std::unexpected("RAD chunk property data exceeds file bounds");
            }

            std::uint8_t* const plane = dst.data() + *plane_offset;
            const bool compressed =
                prop.compression.has_value() &&
                (prop.compression.value() == "gz" || prop.compression.value() == "gzip");
            const bool windowed =
                source_offset != 0 || count != source_count;
            if (!windowed) {
                if (compressed) {
                    if (!rad_decompress_into(&data[absolute_offset], prop_bytes, plane, plane_bytes)) {
                        return std::unexpected(std::format(
                            "Failed to decompress RAD chunk property: {}", prop.property));
                    }
                } else {
                    if (prop_bytes != plane_bytes) {
                        return std::unexpected(std::format(
                            "RAD property '{}' size mismatch: {} bytes on disk, {} expected",
                            prop.property, prop_bytes, plane_bytes));
                    }
                    std::memcpy(plane, &data[absolute_offset], plane_bytes);
                }
                if (enc->delta) {
                    undelta_planes_u8(plane, info->dims, count);
                }
            } else {
                std::vector<std::uint8_t> full_plane(source_plane_bytes);
                if (compressed) {
                    if (!rad_decompress_into(&data[absolute_offset], prop_bytes,
                                             full_plane.data(), source_plane_bytes)) {
                        return std::unexpected(std::format(
                            "Failed to decompress RAD chunk property: {}", prop.property));
                    }
                } else {
                    if (prop_bytes != source_plane_bytes) {
                        return std::unexpected(std::format(
                            "RAD property '{}' size mismatch: {} bytes on disk, {} expected",
                            prop.property, prop_bytes, source_plane_bytes));
                    }
                    std::memcpy(full_plane.data(), &data[absolute_offset], source_plane_bytes);
                }
                if (enc->delta) {
                    undelta_planes_u8(full_plane.data(), info->dims, source_count);
                }
                copy_packed_plane_window(full_plane.data(), source_count, source_offset,
                                         count, *info, enc->encoding, plane);
            }

            auto& out = desc.props[desc.property_count++];
            out.kind = static_cast<std::uint32_t>(info->kind);
            out.encoding = static_cast<std::uint32_t>(enc->encoding);
            out.plane_offset = static_cast<std::uint32_t>(*plane_offset);
            out.plane_bytes = static_cast<std::uint32_t>(plane_bytes);
            out.min_val = prop.min_val.value_or(0.0f);
            out.max_val = info->kind == RadPackedKind::Scales
                              ? prop.max_val.value_or(prop.scale.value_or(1.0f))
                              : prop.max_val.value_or(1.0f);
            out.base = prop.base.value_or(0.0f);
            out.scale = prop.scale.value_or(1.0f);
        }

        if (meta_view.valid() && chunk < meta_view.chunk_count) {
            const std::size_t logical_start = static_cast<std::size_t>(logical_base);
            if (logical_start < meta_view.node_count) {
                const std::size_t run = std::min(lfs::core::SplatLodTree::kChunkSplats,
                                                 meta_view.node_count - logical_start);
                const auto bounds_offset = alloc_plane(run * sizeof(lfs::core::RadMetaBoundsQ));
                const auto links_offset = alloc_plane(run * sizeof(lfs::core::RadMetaLinksQ));
                if (!bounds_offset || !links_offset) {
                    return std::unexpected("RAD packed staging slot too small for sidecar planes");
                }
                std::memcpy(dst.data() + *bounds_offset, meta_view.bounds + logical_start,
                            run * sizeof(lfs::core::RadMetaBoundsQ));
                std::memcpy(dst.data() + *links_offset, meta_view.links + logical_start,
                            run * sizeof(lfs::core::RadMetaLinksQ));
                desc.meta_bounds_offset = static_cast<std::uint32_t>(*bounds_offset);
                desc.meta_links_offset = static_cast<std::uint32_t>(*links_offset);
                desc.meta_node_count = static_cast<std::uint32_t>(run);

                const auto& record = meta_view.chunks[chunk];
                desc.frame.bbox_min[0] = record.bbox_min[0];
                desc.frame.bbox_min[1] = record.bbox_min[1];
                desc.frame.bbox_min[2] = record.bbox_min[2];
                desc.frame.bbox_extent[0] = record.bbox_extent[0];
                desc.frame.bbox_extent[1] = record.bbox_extent[1];
                desc.frame.bbox_extent[2] = record.bbox_extent[2];
                desc.frame.log_size_min = record.log_size_min;
                desc.frame.log_size_range = record.log_size_range;
            }
        }

        desc.used_bytes = static_cast<std::uint32_t>(cursor);
        return desc;
    }

    std::expected<RadDecodedChunk, std::string> load_rad_chunk(
        const std::filesystem::path& filepath,
        const lfs::core::SplatLodTree::ChunkFileRange& range,
        const int max_sh_degree,
        const bool lod_opacity_encoded) {
        std::ifstream in;
        if (!lfs::core::open_file_for_read(filepath, std::ios::binary, in)) {
            return std::unexpected(std::format("Failed to open RAD file for chunk read: {}",
                                               lfs::core::path_to_utf8(filepath)));
        }
        return load_rad_chunk(in, filepath, range, max_sh_degree, lod_opacity_encoded);
    }

    std::expected<RadDecodedChunk, std::string> load_rad_chunk(
        std::istream& in,
        const std::filesystem::path& filepath_for_errors,
        const lfs::core::SplatLodTree::ChunkFileRange& range,
        const int max_sh_degree,
        const bool lod_opacity_encoded) {
        const auto& filepath = filepath_for_errors;
        if (range.file_bytes == 0) {
            return std::unexpected("RAD chunk range has zero bytes");
        }
        if (range.file_bytes > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
            return std::unexpected("RAD chunk range is too large for this platform");
        }

        in.clear();
        in.seekg(static_cast<std::streamoff>(range.file_offset), std::ios::beg);
        if (!in.good()) {
            return std::unexpected(std::format("Failed to seek RAD chunk at offset {} in {}",
                                               range.file_offset,
                                               lfs::core::path_to_utf8(filepath)));
        }

        std::vector<uint8_t> data(static_cast<std::size_t>(range.file_bytes));
        in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
        if (!in.good()) {
            return std::unexpected(std::format("Failed to read RAD chunk at offset {} in {}",
                                               range.file_offset,
                                               lfs::core::path_to_utf8(filepath)));
        }

        auto decoded = decode_rad_chunk_buffer(data,
                                               max_sh_degree,
                                               true,
                                               lod_opacity_encoded);
        if (!decoded) {
            return decoded;
        }
        const std::uint64_t expected_file_base =
            range.file_count != 0 ? range.file_base : range.base;
        const std::uint64_t expected_file_count =
            range.file_count != 0 ? range.file_count : range.count;
        if (expected_file_base != 0 || expected_file_count != 0) {
            if (decoded->base != expected_file_base || decoded->count != expected_file_count) {
                return std::unexpected(std::format(
                    "RAD chunk range mismatch: range base/count={}/{}, decoded base/count={}/{}",
                    expected_file_base,
                    expected_file_count,
                    decoded->base,
                    decoded->count));
            }
        }
        if (range.count == 0 ||
            (range.base == decoded->base && range.count == decoded->count)) {
            return decoded;
        }
        if (range.base < decoded->base || range.base + range.count > decoded->base + decoded->count) {
            return std::unexpected(std::format(
                "RAD logical chunk range {}/{} is outside decoded file chunk {}/{}",
                range.base, range.count, decoded->base, decoded->count));
        }
        const std::size_t first = static_cast<std::size_t>(range.base - decoded->base);
        const std::size_t count = static_cast<std::size_t>(range.count);
        RadDecodedChunk sliced;
        sliced.base = range.base;
        sliced.count = range.count;
        sliced.max_sh_degree = decoded->max_sh_degree;
        sliced.sh_coeffs_rest = decoded->sh_coeffs_rest;
        sliced.lod_opacity_encoded = decoded->lod_opacity_encoded;
        const auto copy_floats = [&](const std::vector<float>& src,
                                     std::vector<float>& dst,
                                     const std::size_t dims) {
            if (src.empty()) {
                return;
            }
            dst.resize(count * dims);
            std::memcpy(dst.data(), src.data() + first * dims,
                        count * dims * sizeof(float));
        };
        copy_floats(decoded->means, sliced.means, 3);
        copy_floats(decoded->opacity_raw, sliced.opacity_raw, 1);
        copy_floats(decoded->sh0_raw, sliced.sh0_raw, 3);
        copy_floats(decoded->scaling_raw, sliced.scaling_raw, 3);
        copy_floats(decoded->rotation_raw, sliced.rotation_raw, 4);
        copy_floats(decoded->shN_canonical, sliced.shN_canonical,
                    static_cast<std::size_t>(decoded->sh_coeffs_rest) * 3);
        if (!decoded->child_count.empty()) {
            sliced.child_count.resize(count);
            std::memcpy(sliced.child_count.data(), decoded->child_count.data() + first,
                        count * sizeof(std::uint16_t));
        }
        if (!decoded->child_start.empty()) {
            sliced.child_start.resize(count);
            std::memcpy(sliced.child_start.data(), decoded->child_start.data() + first,
                        count * sizeof(std::uint32_t));
        }
        return sliced;
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
                           options.flip_y,
                           options.chunk_size,
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
                              std::format("Failed to write RAD file: {}", io_error_detail()),
                              atomic_output.temp_path());
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

    // ============================================================================
    // RadStreamWriter
    // ============================================================================

    namespace {

        // Streams the .rad.meta sidecar while the converter emits chunks, so
        // converter-produced files skip the standalone full-file rebuild pass.
        // Inputs are the freshly encoded chunk's decoded values, making the
        // planes bit-identical to a build_rad_meta_sidecar() run over the
        // finished file. Emission is best-effort: any failure abandons the
        // temp file and the loader rebuilds on demand.
        class RadMetaInlineWriter {
        public:
            ~RadMetaInlineWriter() { abandon(); }

            [[nodiscard]] std::expected<void, std::string> open(
                const std::filesystem::path& rad_path,
                const std::uint64_t node_count,
                const std::uint64_t chunk_count) {
                header_ = {};
                header_.magic = RAD_META_MAGIC;
                header_.version = RAD_META_VERSION;
                header_.endian_sentinel = RAD_META_ENDIAN_SENTINEL;
                header_.chunk_size = static_cast<std::uint32_t>(lfs::core::SplatLodTree::kChunkSplats);
                header_.node_count = node_count;
                header_.chunk_count = chunk_count;
                header_.bounds_offset = RAD_META_HEADER_BYTES;
                header_.links_offset =
                    header_.bounds_offset + alignPlane(node_count * sizeof(lfs::core::RadMetaBoundsQ));
                header_.chunk_table_offset =
                    header_.links_offset + alignPlane(node_count * sizeof(lfs::core::RadMetaLinksQ));
                header_.lod_opacity_encoded = 1;
                header_.complete = 0;
                const std::uint64_t total_bytes =
                    header_.chunk_table_offset + chunk_count * sizeof(RadMetaChunkEntry);

                meta_path_ = rad_meta_sidecar_path(rad_path);
                {
                    std::error_code ec;
                    const auto space = std::filesystem::space(meta_path_.parent_path(), ec);
                    if (!ec && space.available < total_bytes + (std::uint64_t{1} << 28)) {
                        return std::unexpected(
                            std::format("not enough disk space for sidecar: need {} GB, {} GB free",
                                        total_bytes >> 30, space.available >> 30));
                    }
                }

#ifdef _WIN32
                const auto pid = static_cast<std::uint64_t>(GetCurrentProcessId());
#else
                const auto pid = static_cast<std::uint64_t>(getpid());
#endif
                tmp_path_ = meta_path_;
                tmp_path_ += std::format(".tmp.{}", pid);
                {
                    std::ofstream create;
                    if (!lfs::core::open_file_for_write(tmp_path_, std::ios::binary | std::ios::trunc, create)) {
                        return std::unexpected(std::format(
                            "failed to create sidecar temp file: {}", io_error_detail()));
                    }
                    create.seekp(static_cast<std::streamoff>(total_bytes - 1), std::ios::beg);
                    create.put('\0');
                    if (!create.good()) {
                        create.close();
                        removeTmp();
                        return std::unexpected(std::format(
                            "failed to size sidecar temp file: {}", io_error_detail()));
                    }
                }
                out_.open(tmp_path_, std::ios::binary | std::ios::in | std::ios::out);
                if (!out_.is_open()) {
                    removeTmp();
                    return std::unexpected(std::format(
                        "failed to open sidecar temp file: {}", io_error_detail()));
                }
                chunk_table_.assign(static_cast<std::size_t>(chunk_count), {});
                nodes_written_ = 0;
                open_ = true;
                return {};
            }

            [[nodiscard]] bool isOpen() const { return open_; }

            [[nodiscard]] std::expected<void, std::string> writeChunk(
                const std::uint64_t chunk_idx,
                const std::uint64_t base,
                const std::span<const lfs::core::RadMetaBoundsQ> bounds,
                const std::span<const lfs::core::RadMetaLinksQ> links,
                const RadMetaChunkEntry& entry) {
                if (chunk_idx >= chunk_table_.size() ||
                    base != chunk_idx * lfs::core::SplatLodTree::kChunkSplats) {
                    return std::unexpected(std::format(
                        "sidecar chunk {} misaligned (base {})", chunk_idx, base));
                }
                out_.clear();
                out_.seekp(static_cast<std::streamoff>(
                               header_.bounds_offset + base * sizeof(lfs::core::RadMetaBoundsQ)),
                           std::ios::beg);
                out_.write(reinterpret_cast<const char*>(bounds.data()),
                           static_cast<std::streamsize>(bounds.size_bytes()));
                out_.seekp(static_cast<std::streamoff>(
                               header_.links_offset + base * sizeof(lfs::core::RadMetaLinksQ)),
                           std::ios::beg);
                out_.write(reinterpret_cast<const char*>(links.data()),
                           static_cast<std::streamsize>(links.size_bytes()));
                if (!out_.good()) {
                    return std::unexpected(std::format(
                        "failed to write sidecar planes: {}", io_error_detail()));
                }
                chunk_table_[static_cast<std::size_t>(chunk_idx)] = entry;
                nodes_written_ += links.size();
                return {};
            }

            // Must run after the RAD file is committed to its final path: the
            // staleness stamp hashes the published header bytes.
            [[nodiscard]] std::expected<void, std::string> finalize(
                const std::filesystem::path& rad_path,
                const std::size_t chunk_area_start) {
                if (!open_) {
                    return std::unexpected("sidecar writer not open");
                }
                if (nodes_written_ != header_.node_count) {
                    return std::unexpected(std::format(
                        "sidecar holds {} of {} nodes", nodes_written_, header_.node_count));
                }
                auto leaf_count = patchRadMetaLinksPlane(
                    out_, header_.links_offset, header_.node_count, rad_path, tmp_path_, nullptr);
                if (!leaf_count) {
                    return std::unexpected(leaf_count.error().message);
                }
                header_.leaf_count = *leaf_count;

                auto stamp = radSourceStamp(rad_path, chunk_area_start);
                if (!stamp) {
                    return std::unexpected(stamp.error());
                }
                header_.source_file_size = stamp->file_size;
                header_.source_mtime = stamp->mtime;
                header_.source_header_hash = stamp->header_hash;
                header_.complete = 1;

                out_.clear();
                out_.seekp(static_cast<std::streamoff>(header_.chunk_table_offset), std::ios::beg);
                out_.write(reinterpret_cast<const char*>(chunk_table_.data()),
                           static_cast<std::streamsize>(chunk_table_.size() * sizeof(RadMetaChunkEntry)));
                out_.seekp(0, std::ios::beg);
                out_.write(reinterpret_cast<const char*>(&header_), sizeof(header_));
                out_.flush();
                if (!out_.good()) {
                    return std::unexpected(std::format(
                        "failed to finalize sidecar: {}", io_error_detail()));
                }
                out_.close();

                std::error_code rename_ec;
                std::filesystem::rename(tmp_path_, meta_path_, rename_ec);
                if (rename_ec) {
                    removeTmp();
                    open_ = false;
                    if (open_rad_meta_sidecar(rad_path)) {
                        return {};
                    }
                    return std::unexpected(std::format(
                        "failed to publish sidecar: {}", rename_ec.message()));
                }
                open_ = false;
                LOG_INFO("RAD metadata sidecar emitted inline: {} ({} nodes, {} leaves)",
                         lfs::core::path_to_utf8(meta_path_), header_.node_count,
                         header_.leaf_count);
                return {};
            }

            void abandon() {
                if (!open_) {
                    return;
                }
                out_.close();
                removeTmp();
                open_ = false;
            }

        private:
            void removeTmp() {
                std::error_code ec;
                std::filesystem::remove(tmp_path_, ec);
            }

            RadMetaHeader header_{};
            std::vector<RadMetaChunkEntry> chunk_table_;
            std::filesystem::path meta_path_;
            std::filesystem::path tmp_path_;
            std::fstream out_;
            std::uint64_t nodes_written_ = 0;
            bool open_ = false;
        };

    } // namespace

    struct RadStreamWriter::Impl {
        std::filesystem::path output_path;
        std::uint64_t total_count = 0;
        int sh_degree = 0;
        int sh_coeffs = 0;
        bool lod_tree = false;
        int compression_level = GZ_LEVEL;
        std::uint32_t file_chunk_size = DEFAULT_RAD_FILE_CHUNK_SIZE;

        std::optional<ScopedAtomicOutputFile> atomic_output;
        std::ofstream out;
        std::size_t meta_reserved = 0;
        std::uint64_t written_count = 0;
        std::uint64_t chunk_area_bytes = 0;
        std::vector<RadChunkRange> ranges;
        bool opened = false;
        bool finished = false;

        bool emit_meta_sidecar = false;
        RadMetaInlineWriter meta_writer;

        // GPU chunk quantization (bit-identical planes; DEFLATE stays on the
        // CPU). Any CUDA failure falls back permanently for this writer.
        RadGpuQuantization gpu_quantization = RadGpuQuantization::Auto;
        std::unique_ptr<cuda::RadEncodeGpuQuantizer> gpu_quant;
        bool gpu_quant_resolved = false;

        bool gpuQuantEnabled() {
            if (!gpu_quant_resolved) {
                gpu_quant_resolved = true;
                if (gpu_quantization == RadGpuQuantization::Auto &&
                    cuda::rad_encode_gpu_available()) {
                    gpu_quant = std::make_unique<cuda::RadEncodeGpuQuantizer>();
                }
            }
            return gpu_quant != nullptr;
        }

        void dropMetaWriter(const std::string& reason) {
            LOG_WARN("RAD sidecar inline emission disabled: {}; loader rebuilds on demand", reason);
            meta_writer.abandon();
        }
    };

    RadStreamWriter::RadStreamWriter(std::filesystem::path output_path,
                                     const std::uint64_t total_count,
                                     const int sh_degree,
                                     const bool lod_tree,
                                     const int compression_level,
                                     const bool emit_meta_sidecar,
                                     const std::uint32_t chunk_size,
                                     const RadGpuQuantization gpu_quantization)
        : impl_(std::make_unique<Impl>()) {
        impl_->output_path = std::move(output_path);
        impl_->total_count = total_count;
        impl_->sh_degree = std::clamp(sh_degree, 0, 3);
        impl_->sh_coeffs = impl_->sh_degree > 0 ? SH_COEFFS_FOR_DEGREE[impl_->sh_degree] : 0;
        impl_->lod_tree = lod_tree;
        impl_->file_chunk_size = normalized_export_chunk_size(chunk_size);
        impl_->compression_level =
            (compression_level >= Z_NO_COMPRESSION && compression_level <= Z_BEST_COMPRESSION)
                ? compression_level
                : GZ_LEVEL;
        impl_->emit_meta_sidecar = emit_meta_sidecar && lod_tree;
        impl_->gpu_quantization = gpu_quantization;
    }

    RadStreamWriter::~RadStreamWriter() = default;

    std::expected<void, std::string> RadStreamWriter::open() {
        auto& s = *impl_;
        if (s.opened) {
            return std::unexpected("RadStreamWriter: already open");
        }
        if (s.total_count == 0) {
            return std::unexpected("RadStreamWriter: total count must be non-zero");
        }
        if (s.total_count > std::numeric_limits<std::uint32_t>::max()) {
            return std::unexpected("RadStreamWriter: node count exceeds RAD format limit");
        }

        if (auto dir_result = ensure_output_parent_directory(s.output_path); !dir_result) {
            return std::unexpected(dir_result.error().message);
        }

        const std::uint64_t expected_chunks =
            (s.total_count + s.file_chunk_size - 1) / s.file_chunk_size;
        s.meta_reserved = pad8(static_cast<std::size_t>(512 + expected_chunks * 120));
        s.ranges.reserve(static_cast<std::size_t>(expected_chunks));

        s.atomic_output.emplace(s.output_path);
        if (!lfs::core::open_file_for_write(s.atomic_output->temp_path(),
                                            std::ios::binary | std::ios::out,
                                            s.out)) {
            return std::unexpected("RadStreamWriter: failed to open temporary output file");
        }

        std::array<std::uint8_t, 8> header{};
        encode_u32(header.data(), RAD_MAGIC);
        encode_u32(header.data() + 4, static_cast<std::uint32_t>(s.meta_reserved));
        s.out.write(reinterpret_cast<const char*>(header.data()), header.size());

        const std::string padding(s.meta_reserved, ' ');
        s.out.write(padding.data(), static_cast<std::streamsize>(padding.size()));
        if (!s.out.good()) {
            return std::unexpected(std::format(
                "RadStreamWriter: failed to write RAD header: {}", io_error_detail()));
        }
        s.opened = true;

        if (s.emit_meta_sidecar) {
            if (auto meta_opened = s.meta_writer.open(
                    s.output_path, s.total_count, native_lod_chunk_count(s.total_count));
                !meta_opened) {
                s.dropMetaWriter(meta_opened.error());
            }
        }
        return {};
    }

    std::expected<void, std::string> RadStreamWriter::append(const RadStreamChunkSource& chunk) {
        return append_batch(std::span(&chunk, 1));
    }

    std::expected<void, std::string> RadStreamWriter::append_batch(
        const std::span<const RadStreamChunkSource> chunks) {
        auto& s = *impl_;
        if (!s.opened || s.finished) {
            return std::unexpected("RadStreamWriter: append on unopened or finished writer");
        }
        if (chunks.empty()) {
            return {};
        }

        std::vector<std::uint64_t> bases(chunks.size());
        std::uint64_t cursor = s.written_count;
        for (std::size_t i = 0; i < chunks.size(); ++i) {
            const auto& chunk = chunks[i];
            if (chunk.count == 0 || chunk.count > s.file_chunk_size) {
                return std::unexpected("RadStreamWriter: invalid chunk splat count");
            }
            if (cursor + chunk.count < s.total_count &&
                chunk.count != s.file_chunk_size) {
                return std::unexpected("RadStreamWriter: non-final chunk is smaller than file chunk size");
            }
            if (chunk.means == nullptr || chunk.alpha == nullptr || chunk.rgb == nullptr ||
                chunk.scales == nullptr || chunk.rotation == nullptr) {
                return std::unexpected("RadStreamWriter: missing chunk property arrays");
            }
            if (s.lod_tree && (chunk.child_count == nullptr || chunk.child_start == nullptr)) {
                return std::unexpected("RadStreamWriter: LOD tree chunks require child arrays");
            }
            bases[i] = cursor;
            cursor += chunk.count;
        }
        if (cursor > s.total_count) {
            return std::unexpected("RadStreamWriter: chunk exceeds declared total count");
        }

        // Sidecar planes quantize from the chunk's encoded-then-decoded values
        // so they match a standalone rebuild over the finished file bit for
        // bit (lossy property encodings shift means/scales/alpha slightly).
        struct MetaPageSlice {
            std::uint64_t base = 0;
            std::vector<lfs::core::RadMetaBoundsQ> bounds;
            std::vector<lfs::core::RadMetaLinksQ> links;
            RadMetaChunkEntry entry{};
        };
        struct MetaSlice {
            std::vector<MetaPageSlice> pages;
            std::size_t payload_start = 0;
            std::uint64_t payload_bytes = 0;
            std::string error;
        };
        const bool emit_meta = s.meta_writer.isOpen();
        std::vector<MetaSlice> meta(emit_meta ? chunks.size() : 0);

        // Batch-quantize the pure-arithmetic planes on the GPU while the TBB
        // workers keep the libm encoders (scales, orientation) and DEFLATE.
        std::vector<cuda::RadEncodeQuantChunkOut> gpu_planes;
        if (s.gpuQuantEnabled()) {
            std::vector<cuda::RadEncodeQuantChunkIn> gpu_in(chunks.size());
            for (std::size_t i = 0; i < chunks.size(); ++i) {
                const auto& chunk = chunks[i];
                gpu_in[i] = {chunk.count, chunk.means, chunk.alpha, chunk.rgb,
                             s.sh_coeffs > 0 ? chunk.shN : nullptr};
            }
            gpu_planes.resize(chunks.size());
            if (!s.gpu_quant->quantize_batch(gpu_in, s.sh_coeffs, s.lod_tree, gpu_planes)) {
                gpu_planes.clear();
                s.gpu_quant.reset();
                LOG_WARN("RAD GPU encode quantization failed; using CPU encoders");
            }
        }

        std::vector<std::pair<RadChunkMeta, std::vector<uint8_t>>> encoded(chunks.size());
        tbb::parallel_for(std::size_t{0}, chunks.size(), [&](const std::size_t i) {
            const auto& chunk = chunks[i];
            encoded[i] = encode_rad_chunk(
                static_cast<std::uint32_t>(bases[i]),
                chunk.count,
                s.sh_degree,
                s.sh_coeffs,
                chunk.means,
                chunk.alpha,
                chunk.rgb,
                chunk.scales,
                chunk.rotation,
                s.sh_coeffs > 0 ? chunk.shN : nullptr,
                s.lod_tree ? chunk.child_count : nullptr,
                s.lod_tree ? chunk.child_start : nullptr,
                s.lod_tree,
                s.compression_level,
                gpu_planes.empty() ? nullptr : &gpu_planes[i]);

            if (!emit_meta) {
                return;
            }
            auto& slice = meta[i];
            const auto& bytes = encoded[i].second;
            auto parsed = parse_rad_chunk_header(bytes.data(), bytes.size());
            if (!parsed) {
                slice.error = parsed.error();
                return;
            }
            const std::size_t count = chunk.count;
            std::vector<float> means(count * 3);
            std::vector<float> scales(count * 3);
            std::vector<float> alpha(count);
            auto err = decode_chunk_properties(
                bytes.data(), parsed->meta, 0, parsed->payload_start,
                parsed->has_payload_prefix, parsed->chunk_end, 0,
                means.data(), alpha.data(), nullptr,
                scales.data(), nullptr, nullptr, nullptr, nullptr);
            if (err.has_value()) {
                slice.error = std::move(*err);
                return;
            }
            slice.pages.clear();
            for (std::size_t offset = 0; offset < count; offset += NATIVE_CHUNK_SIZE) {
                const std::size_t run = std::min<std::size_t>(NATIVE_CHUNK_SIZE, count - offset);
                MetaPageSlice page;
                page.base = bases[i] + offset;
                page.bounds.resize(run);
                page.links.resize(run);
                std::vector<float> sizes;
                quantizeRadMetaChunk(page.base, run,
                                     means.data() + offset * 3,
                                     scales.data() + offset * 3,
                                     alpha.data() + offset,
                                     chunk.child_count + offset,
                                     chunk.child_start + offset,
                                     /*lod_opacity_encoded=*/true, sizes,
                                     page.bounds.data(), page.links.data(), page.entry);
                slice.pages.push_back(std::move(page));
            }
            slice.payload_start = parsed->payload_start;
            slice.payload_bytes = parsed->meta.payload_bytes;
        });

        for (std::size_t i = 0; i < chunks.size(); ++i) {
            const auto& payload = encoded[i].second;
            s.out.write(reinterpret_cast<const char*>(payload.data()),
                        static_cast<std::streamsize>(payload.size()));
            if (!s.out.good()) {
                return std::unexpected(std::format(
                    "RadStreamWriter: failed to write chunk payload: {}", io_error_detail()));
            }

            RadChunkRange range;
            range.offset = s.chunk_area_bytes;
            range.bytes = payload.size();
            range.base = s.written_count;
            range.count = chunks[i].count;
            s.ranges.push_back(range);

            if (s.meta_writer.isOpen()) {
                auto& slice = meta[i];
                if (!slice.error.empty()) {
                    s.dropMetaWriter(slice.error);
                } else {
                    const std::uint64_t payload_offset =
                        8 + s.meta_reserved + range.offset + slice.payload_start;
                    for (auto& page : slice.pages) {
                        page.entry.payload_offset = payload_offset;
                        page.entry.payload_bytes = slice.payload_bytes;
                        const std::uint64_t page_idx = page.base / NATIVE_CHUNK_SIZE;
                        if (auto written = s.meta_writer.writeChunk(
                                page_idx, page.base, page.bounds, page.links, page.entry);
                            !written) {
                            s.dropMetaWriter(written.error());
                            break;
                        }
                    }
                }
            }

            s.chunk_area_bytes += payload.size();
            s.written_count += chunks[i].count;
        }
        return {};
    }

    std::expected<void, std::string> RadStreamWriter::finish() {
        auto& s = *impl_;
        if (!s.opened || s.finished) {
            return std::unexpected("RadStreamWriter: finish on unopened or finished writer");
        }
        if (s.written_count != s.total_count) {
            return std::unexpected(std::format(
                "RadStreamWriter: wrote {} splats but {} were declared",
                s.written_count, s.total_count));
        }

        RadMeta meta;
        meta.count = s.written_count;
        meta.max_sh = s.sh_degree;
        meta.lod_tree = s.lod_tree ? std::optional<bool>(true) : std::nullopt;
        meta.chunk_size = s.file_chunk_size;
        meta.all_chunk_bytes = s.chunk_area_bytes;
        meta.chunks = std::move(s.ranges);
        if (s.lod_tree) {
            meta.splat_encoding = nlohmann::json{{"lodOpacity", true}};
        }

        const std::string meta_json = meta.to_json().dump();
        if (meta_json.size() > s.meta_reserved) {
            return std::unexpected(std::format(
                "RadStreamWriter: metadata ({} bytes) exceeds reserved space ({} bytes)",
                meta_json.size(), s.meta_reserved));
        }

        s.out.seekp(8, std::ios::beg);
        s.out.write(meta_json.data(), static_cast<std::streamsize>(meta_json.size()));
        s.out.flush();
        s.out.close();
        if (!s.out.good()) {
            return std::unexpected(std::format(
                "RadStreamWriter: failed to finalize RAD file: {}", io_error_detail()));
        }

        if (auto commit_result = s.atomic_output->commit(); !commit_result) {
            return std::unexpected(commit_result.error().message);
        }
        s.finished = true;

        if (s.meta_writer.isOpen()) {
            // The stamp hashes the committed file's header, so this must come
            // after the atomic rename above.
            if (auto finalized = s.meta_writer.finalize(s.output_path, 8 + s.meta_reserved);
                !finalized) {
                s.dropMetaWriter(finalized.error());
            }
        }
        return {};
    }

    // ========================================================================
    // Node-metadata sidecar (<file>.rad.meta)
    // ========================================================================

    std::filesystem::path rad_meta_sidecar_path(const std::filesystem::path& rad_path) {
        auto path = rad_path;
        path += ".meta";
        return path;
    }

    std::expected<std::uint64_t, std::string> derive_rad_meta_parents_levels(
        const std::span<lfs::core::RadMetaLinksQ> links) {
        const std::uint64_t n = links.size();
        if (n == 0) {
            return std::unexpected("empty links plane");
        }
        std::vector<std::uint32_t> parent(n, lfs::core::SplatLodTree::kInvalidPage);
        std::vector<std::uint8_t> level(n, 0);
        std::uint64_t leaf_count = 0;
        std::uint64_t assigned = 0;
        for (std::uint64_t i = 0; i < n; ++i) {
            auto& rec = links[i];
            rec.parent = parent[i];
            rec.packed = (rec.packed & 0xff00ffffu) |
                         ((static_cast<std::uint32_t>(level[i]) & 0xffu) << 16u);
            const std::uint32_t cc = rec.packed & 0xffffu;
            if (cc == 0) {
                ++leaf_count;
                continue;
            }
            const std::uint64_t cs = rec.child_start;
            if (cs <= i || cs + cc > n) {
                return std::unexpected(std::format(
                    "corrupt LOD layout: node {} children [{}, {}) out of order", i, cs, cs + cc));
            }
            const std::uint8_t child_level =
                static_cast<std::uint8_t>(std::min<std::uint32_t>(level[i] + 1u, 255u));
            for (std::uint32_t c = 0; c < cc; ++c) {
                parent[cs + c] = static_cast<std::uint32_t>(i);
                level[cs + c] = child_level;
            }
            assigned += cc;
        }
        if (assigned != n - 1) {
            return std::unexpected(std::format(
                "corrupt LOD layout: {} of {} nodes have a parent", assigned, n - 1));
        }
        return leaf_count;
    }

    std::expected<lfs::core::SplatLodTree::NodeMetaView, std::string> open_rad_meta_sidecar(
        const std::filesystem::path& rad_path) {
        const auto meta_path = rad_meta_sidecar_path(rad_path);
        auto file = std::make_shared<lfs::core::MappedFile>();
        if (!file->open(meta_path, lfs::core::MappedFile::Advice::Random)) {
            return std::unexpected("sidecar not present");
        }
        if (file->size() < RAD_META_HEADER_BYTES) {
            return std::unexpected("sidecar truncated");
        }
        RadMetaHeader header{};
        std::memcpy(&header, file->data(), sizeof(header));
        if (header.magic != RAD_META_MAGIC) {
            return std::unexpected("sidecar magic mismatch");
        }
        if (header.version != RAD_META_VERSION) {
            return std::unexpected("sidecar version mismatch");
        }
        if (header.endian_sentinel != RAD_META_ENDIAN_SENTINEL) {
            return std::unexpected("sidecar endianness mismatch");
        }
        if (header.complete != 1) {
            return std::unexpected("sidecar incomplete (interrupted build)");
        }
        if (header.chunk_size != lfs::core::SplatLodTree::kChunkSplats) {
            return std::unexpected("sidecar chunk size mismatch");
        }

        auto info = read_rad_file_info(rad_path);
        if (!info) {
            return std::unexpected(info.error());
        }
        auto stamp = radSourceStamp(rad_path, info->chunk_area_start);
        if (!stamp) {
            return std::unexpected(stamp.error());
        }
        if (header.source_file_size != stamp->file_size ||
            header.source_mtime != stamp->mtime ||
            header.source_header_hash != stamp->header_hash) {
            return std::unexpected("sidecar stale (RAD file changed)");
        }
        if (header.node_count != info->meta.count) {
            return std::unexpected("sidecar node count disagrees with RAD metadata");
        }

        const std::size_t bounds_end =
            header.bounds_offset + header.node_count * sizeof(lfs::core::RadMetaBoundsQ);
        const std::size_t links_end =
            header.links_offset + header.node_count * sizeof(lfs::core::RadMetaLinksQ);
        const std::size_t chunks_end =
            header.chunk_table_offset + header.chunk_count * sizeof(lfs::core::RadMetaChunkRecord);
        if (bounds_end > file->size() || links_end > file->size() || chunks_end > file->size()) {
            return std::unexpected("sidecar planes exceed file size");
        }

        lfs::core::SplatLodTree::NodeMetaView view;
        view.bounds = reinterpret_cast<const lfs::core::RadMetaBoundsQ*>(
            file->data() + header.bounds_offset);
        view.links = reinterpret_cast<const lfs::core::RadMetaLinksQ*>(
            file->data() + header.links_offset);
        view.chunks = reinterpret_cast<const lfs::core::RadMetaChunkRecord*>(
            file->data() + header.chunk_table_offset);
        view.node_count = header.node_count;
        view.chunk_count = header.chunk_count;
        view.leaf_count = header.leaf_count;
        view.file = std::move(file);
        return view;
    }

    void expand_rad_meta_page(const lfs::core::SplatLodTree::NodeMetaView& view,
                              const std::uint32_t chunk,
                              const std::size_t node_count,
                              lfs::core::NodeBoundsRecord* const out_bounds,
                              lfs::core::NodeLinksRecord* const out_links) {
        const std::size_t logical_start =
            static_cast<std::size_t>(chunk) * lfs::core::SplatLodTree::kChunkSplats;
        const auto& frame = view.chunks[chunk];
        for (std::size_t i = 0; i < node_count; ++i) {
            const auto& q = view.bounds[logical_start + i];
            const glm::vec3 center = frame.dequantCenter(q);
            out_bounds[i] = {
                .x = center.x,
                .y = center.y,
                .z = center.z,
                .size = frame.dequantSize(q),
            };
            const auto& l = view.links[logical_start + i];
            out_links[i] = {
                .child_start = l.child_start,
                .packed = l.packed,
                .parent = l.parent,
                .logical = static_cast<std::uint32_t>(logical_start + i),
            };
        }
    }

    Result<void> build_rad_meta_sidecar(
        const std::filesystem::path& rad_path,
        const ExportProgressCallback& progress) {
        const auto t_start = std::chrono::high_resolution_clock::now();
        const auto report = [&](const float p, const std::string& stage) -> bool {
            return progress == nullptr || progress(p, stage);
        };

        auto info_result = read_rad_file_info(rad_path);
        if (!info_result) {
            return make_error(ErrorCode::INVALID_HEADER, info_result.error(), rad_path);
        }
        const RadFileInfo& info = *info_result;
        const RadMeta& meta = info.meta;
        if (!meta.lod_tree.value_or(false) || !rad_ranges_usable(meta)) {
            return make_error(ErrorCode::CORRUPTED_DATA,
                              "RAD file has no usable LOD chunk index", rad_path);
        }
        if (const std::uint32_t file_chunk = normalized_lod_file_chunk_size(meta);
            !supported_lod_file_chunk_size(file_chunk)) {
            return make_error(ErrorCode::CORRUPTED_DATA,
                              std::format("RAD LOD file uses unsupported {}-splat chunks; expected a multiple of {}.",
                                          file_chunk, NATIVE_CHUNK_SIZE),
                              rad_path);
        }
        const std::uint64_t n = meta.count;
        if (n > std::numeric_limits<std::uint32_t>::max()) {
            return make_error(ErrorCode::CORRUPTED_DATA,
                              "RAD node count exceeds 32-bit sidecar limit", rad_path);
        }

        bool lod_opacity_encoded = true;
        if (meta.splat_encoding.has_value() && meta.splat_encoding->is_object()) {
            auto it = meta.splat_encoding->find("lodOpacity");
            if (it != meta.splat_encoding->end() && it->is_boolean()) {
                lod_opacity_encoded = it->get<bool>();
            }
        }

        auto stamp = radSourceStamp(rad_path, info.chunk_area_start);
        if (!stamp) {
            return make_error(ErrorCode::READ_FAILURE, stamp.error(), rad_path);
        }

        RadMetaHeader header{};
        header.magic = RAD_META_MAGIC;
        header.version = RAD_META_VERSION;
        header.endian_sentinel = RAD_META_ENDIAN_SENTINEL;
        header.chunk_size = static_cast<std::uint32_t>(lfs::core::SplatLodTree::kChunkSplats);
        header.node_count = n;
        header.chunk_count = native_lod_chunk_count(n);
        header.source_file_size = stamp->file_size;
        header.source_mtime = stamp->mtime;
        header.source_header_hash = stamp->header_hash;
        header.bounds_offset = RAD_META_HEADER_BYTES;
        header.links_offset =
            header.bounds_offset + alignPlane(n * sizeof(lfs::core::RadMetaBoundsQ));
        header.chunk_table_offset =
            header.links_offset + alignPlane(n * sizeof(lfs::core::RadMetaLinksQ));
        header.lod_opacity_encoded = lod_opacity_encoded ? 1 : 0;
        header.complete = 0;
        const std::uint64_t total_bytes =
            header.chunk_table_offset + header.chunk_count * sizeof(RadMetaChunkEntry);

        const auto meta_path = rad_meta_sidecar_path(rad_path);
        {
            std::error_code ec;
            const auto space = std::filesystem::space(meta_path.parent_path(), ec);
            if (!ec && space.available < total_bytes + (std::uint64_t{1} << 28)) {
                return make_error(ErrorCode::WRITE_FAILURE,
                                  std::format("Not enough disk space for sidecar: need {} GB, {} GB free",
                                              total_bytes >> 30, space.available >> 30),
                                  meta_path);
            }
        }

#ifdef _WIN32
        const auto pid = static_cast<std::uint64_t>(GetCurrentProcessId());
#else
        const auto pid = static_cast<std::uint64_t>(getpid());
#endif
        auto tmp_path = meta_path;
        tmp_path += std::format(".tmp.{}", pid);
        struct TmpGuard {
            std::filesystem::path path;
            bool keep = false;
            ~TmpGuard() {
                if (!keep) {
                    std::error_code ec;
                    std::filesystem::remove(path, ec);
                }
            }
        } tmp_guard{.path = tmp_path};

        // Create and pre-extend so parallel writers can seek anywhere.
        {
            std::ofstream create;
            if (!lfs::core::open_file_for_write(tmp_path, std::ios::binary | std::ios::trunc, create)) {
                return make_error(ErrorCode::WRITE_FAILURE,
                                  std::format("Failed to create sidecar temp file: {}", io_error_detail()),
                                  tmp_path);
            }
            create.seekp(static_cast<std::streamoff>(total_bytes - 1), std::ios::beg);
            create.put('\0');
            if (!create.good()) {
                return make_error(ErrorCode::WRITE_FAILURE,
                                  std::format("Failed to size sidecar temp file: {}", io_error_detail()),
                                  tmp_path);
            }
        }

        if (!report(0.0f, "Building RAD metadata sidecar")) {
            return make_error(ErrorCode::CANCELLED, "Sidecar build cancelled", rad_path);
        }

        // Phase 1 (parallel): decode each chunk's tree-relevant properties and
        // write bounds + partial links (level/parent patched in phase 2).
        std::vector<RadMetaChunkEntry> chunk_table(static_cast<std::size_t>(header.chunk_count));
        std::atomic<bool> failed{false};
        std::mutex error_mutex;
        std::string build_error;
        const auto record_error = [&](std::string msg) {
            failed.store(true, std::memory_order_relaxed);
            std::lock_guard<std::mutex> lock(error_mutex);
            if (build_error.empty()) {
                build_error = std::move(msg);
            }
        };
        std::atomic<std::size_t> chunks_done{0};
        std::mutex progress_mutex;
        std::atomic<bool> cancelled{false};

        struct BuildScratch {
            std::ifstream rad;
            std::fstream out;
            bool open_ok = false;
            std::vector<std::uint8_t> raw;
            std::vector<float> means;
            std::vector<float> scales;
            std::vector<float> alpha;
            std::vector<float> sizes;
            std::vector<std::uint16_t> child_count;
            std::vector<std::uint32_t> child_start;
            std::vector<lfs::core::RadMetaBoundsQ> bounds;
            std::vector<lfs::core::RadMetaLinksQ> links;
        };
        tbb::enumerable_thread_specific<BuildScratch> scratch_tls;

        tbb::parallel_for(
            tbb::blocked_range<std::size_t>(0, meta.chunks.size(), 1),
            [&](const tbb::blocked_range<std::size_t>& chunk_range) {
                auto& scratch = scratch_tls.local();
                if (!scratch.open_ok) {
                    if (!lfs::core::open_file_for_read(rad_path, std::ios::binary, scratch.rad)) {
                        record_error("Failed to open RAD file for sidecar build");
                        return;
                    }
                    scratch.out.open(tmp_path, std::ios::binary | std::ios::in | std::ios::out);
                    if (!scratch.out.is_open()) {
                        record_error(std::format("Failed to open sidecar temp file: {}", io_error_detail()));
                        return;
                    }
                    scratch.open_ok = true;
                }

                for (std::size_t ci = chunk_range.begin(); ci != chunk_range.end(); ++ci) {
                    if (failed.load(std::memory_order_relaxed) ||
                        cancelled.load(std::memory_order_relaxed)) {
                        return;
                    }
                    const auto& range = meta.chunks[ci];
                    const std::size_t base = static_cast<std::size_t>(*range.base);
                    const std::size_t count = static_cast<std::size_t>(*range.count);
                    const std::size_t bytes = static_cast<std::size_t>(range.bytes);
                    const std::uint64_t file_offset =
                        info.chunk_area_start + static_cast<std::size_t>(range.offset);

                    scratch.raw.resize(bytes);
                    scratch.rad.clear();
                    scratch.rad.seekg(static_cast<std::streamoff>(file_offset), std::ios::beg);
                    scratch.rad.read(reinterpret_cast<char*>(scratch.raw.data()),
                                     static_cast<std::streamsize>(bytes));
                    if (!scratch.rad.good()) {
                        record_error("Failed to read RAD chunk for sidecar build");
                        return;
                    }

                    auto parsed = parse_rad_chunk_header(scratch.raw.data(), scratch.raw.size());
                    if (!parsed) {
                        record_error(parsed.error());
                        return;
                    }
                    const RadChunkMeta& chunk = parsed->meta;
                    if (chunk.base != base || chunk.count != count) {
                        record_error("RAD chunk header disagrees with chunk index");
                        return;
                    }

                    scratch.means.assign(count * 3, 0.0f);
                    scratch.scales.assign(count * 3, 0.0f);
                    scratch.alpha.assign(count, 0.0f);
                    scratch.child_count.assign(count, 0);
                    scratch.child_start.assign(count, 0);
                    auto err = decode_chunk_properties(
                        scratch.raw.data(), chunk, 0, parsed->payload_start,
                        parsed->has_payload_prefix, parsed->chunk_end, 0,
                        scratch.means.data(), scratch.alpha.data(), nullptr,
                        scratch.scales.data(), nullptr, nullptr,
                        scratch.child_count.data(), scratch.child_start.data());
                    if (err.has_value()) {
                        record_error(std::move(*err));
                        return;
                    }

                    for (std::size_t s = 0; s < count; s += NATIVE_CHUNK_SIZE) {
                        const std::size_t run = std::min<std::size_t>(NATIVE_CHUNK_SIZE, count - s);
                        const std::size_t logical_base = base + s;
                        const std::size_t logical_chunk = logical_base / NATIVE_CHUNK_SIZE;
                        if (logical_chunk >= chunk_table.size()) {
                            record_error("RAD sidecar logical chunk index exceeds table");
                            return;
                        }
                        scratch.bounds.resize(run);
                        scratch.links.resize(run);
                        RadMetaChunkEntry entry{};
                        quantizeRadMetaChunk(logical_base, run,
                                             scratch.means.data() + s * 3,
                                             scratch.scales.data() + s * 3,
                                             scratch.alpha.data() + s,
                                             scratch.child_count.data() + s,
                                             scratch.child_start.data() + s,
                                             lod_opacity_encoded,
                                             scratch.sizes,
                                             scratch.bounds.data(), scratch.links.data(), entry);

                        scratch.out.clear();
                        scratch.out.seekp(static_cast<std::streamoff>(
                                              header.bounds_offset +
                                              logical_base * sizeof(lfs::core::RadMetaBoundsQ)),
                                          std::ios::beg);
                        scratch.out.write(reinterpret_cast<const char*>(scratch.bounds.data()),
                                          static_cast<std::streamsize>(run * sizeof(lfs::core::RadMetaBoundsQ)));
                        scratch.out.seekp(static_cast<std::streamoff>(
                                              header.links_offset +
                                              logical_base * sizeof(lfs::core::RadMetaLinksQ)),
                                          std::ios::beg);
                        scratch.out.write(reinterpret_cast<const char*>(scratch.links.data()),
                                          static_cast<std::streamsize>(run * sizeof(lfs::core::RadMetaLinksQ)));
                        if (!scratch.out.good()) {
                            record_error(std::format("Failed to write sidecar planes: {}", io_error_detail()));
                            return;
                        }

                        entry.payload_offset = file_offset + parsed->payload_start;
                        entry.payload_bytes = chunk.payload_bytes;
                        chunk_table[logical_chunk] = entry;
                    }

                    const std::size_t done = chunks_done.fetch_add(1) + 1;
                    if ((done & 0x3f) == 0 || done == meta.chunks.size()) {
                        std::lock_guard<std::mutex> lock(progress_mutex);
                        const float p = 0.8f * static_cast<float>(done) /
                                        static_cast<float>(meta.chunks.size());
                        if (!report(p, "Building RAD metadata sidecar")) {
                            cancelled.store(true, std::memory_order_relaxed);
                        }
                    }
                }
            });

        for (auto& scratch : scratch_tls) {
            if (scratch.open_ok) {
                scratch.out.flush();
            }
        }
        if (cancelled.load()) {
            return make_error(ErrorCode::CANCELLED, "Sidecar build cancelled", rad_path);
        }
        if (failed.load()) {
            return make_error(ErrorCode::CORRUPTED_DATA, build_error, rad_path);
        }

        {
            std::fstream rmw(tmp_path, std::ios::binary | std::ios::in | std::ios::out);
            if (!rmw.is_open()) {
                return make_error(ErrorCode::WRITE_FAILURE,
                                  std::format("Failed to reopen sidecar temp file: {}", io_error_detail()),
                                  tmp_path);
            }
            auto leaf_count = patchRadMetaLinksPlane(
                rmw, header.links_offset, n, rad_path, tmp_path,
                [&](const float f) {
                    return report(0.8f + 0.2f * f, "Linking RAD metadata sidecar");
                });
            if (!leaf_count) {
                return std::unexpected(leaf_count.error());
            }
            header.leaf_count = *leaf_count;

            // Chunk table + finalized header (complete=1) last, then rename.
            rmw.clear();
            rmw.seekp(static_cast<std::streamoff>(header.chunk_table_offset), std::ios::beg);
            rmw.write(reinterpret_cast<const char*>(chunk_table.data()),
                      static_cast<std::streamsize>(chunk_table.size() * sizeof(RadMetaChunkEntry)));
            header.complete = 1;
            rmw.seekp(0, std::ios::beg);
            rmw.write(reinterpret_cast<const char*>(&header), sizeof(header));
            rmw.flush();
            if (!rmw.good()) {
                return make_error(ErrorCode::WRITE_FAILURE,
                                  std::format("Failed to finalize sidecar: {}", io_error_detail()),
                                  tmp_path);
            }
        }

        std::error_code rename_ec;
        std::filesystem::rename(tmp_path, meta_path, rename_ec);
        if (rename_ec) {
            // A concurrent builder may have won the race; adopt its result if
            // it validates, otherwise surface the rename failure.
            if (open_rad_meta_sidecar(rad_path)) {
                return {};
            }
            return make_error(ErrorCode::WRITE_FAILURE,
                              std::format("Failed to publish sidecar: {}", rename_ec.message()),
                              meta_path);
        }
        tmp_guard.keep = true;

        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::high_resolution_clock::now() - t_start);
        LOG_INFO("RAD metadata sidecar built: {} ({} nodes, {} leaves, {:.1f} GB) in {}s",
                 lfs::core::path_to_utf8(meta_path), n, header.leaf_count,
                 static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0),
                 elapsed.count());
        return {};
    }

    std::expected<std::optional<std::uint32_t>, std::string> rad_lod_file_chunk_size(
        const std::filesystem::path& input) {
        auto info = read_rad_file_info(input);
        if (!info) {
            return std::unexpected(info.error());
        }
        if (!info->meta.lod_tree.value_or(false)) {
            return std::optional<std::uint32_t>{};
        }
        return std::optional<std::uint32_t>{normalized_lod_file_chunk_size(info->meta)};
    }

    Result<void> rechunk_rad_lod(const std::filesystem::path& input,
                                 const std::filesystem::path& output,
                                 const std::uint32_t target_chunk_size,
                                 const RechunkProgressCallback& progress) {
        auto info = read_rad_file_info(input);
        if (!info) {
            return make_error(ErrorCode::INVALID_HEADER, info.error(), input);
        }
        const RadMeta& meta = info->meta;
        if (!meta.lod_tree.value_or(false)) {
            return make_error(ErrorCode::CORRUPTED_DATA,
                              "Not a RAD LOD file; flat RAD files load unchanged", input);
        }
        if (!rad_ranges_usable(meta)) {
            return make_error(ErrorCode::CORRUPTED_DATA,
                              "RAD file has no usable LOD chunk index", input);
        }
        const std::uint32_t source_chunk = normalized_lod_file_chunk_size(meta);
        if (!supported_lod_file_chunk_size(source_chunk)) {
            return make_error(ErrorCode::CORRUPTED_DATA,
                              std::format("Unsupported source chunk size {}", source_chunk),
                              input);
        }
        const std::uint32_t output_chunk_size = normalized_export_chunk_size(target_chunk_size);
        const int max_sh = meta.max_sh.value_or(0);
        const int sh_coeffs = max_sh > 0 ? SH_COEFFS_FOR_DEGREE[max_sh] : 0;

        std::ifstream in;
        if (!lfs::core::open_file_for_read(input, std::ios::binary, in)) {
            return make_error(ErrorCode::PATH_NOT_FOUND, "Failed to open RAD file", input);
        }

        RadStreamWriter writer(output, meta.count, max_sh, /*lod_tree=*/true,
                               GZ_LEVEL, /*emit_meta_sidecar=*/false, output_chunk_size);
        if (auto opened = writer.open(); !opened) {
            return make_error(ErrorCode::WRITE_FAILURE, opened.error(), output);
        }

        const std::size_t old_count = source_chunk;
        const std::size_t out_capacity = output_chunk_size;
        std::vector<float> means(old_count * 3), opacity(old_count), rgb(old_count * 3);
        std::vector<float> scales(old_count * 3), rotation(old_count * 4);
        std::vector<float> shN(sh_coeffs > 0
                                   ? old_count * static_cast<std::size_t>(sh_coeffs) * 3
                                   : 0);
        std::vector<std::uint16_t> child_count(old_count);
        std::vector<std::uint32_t> child_start(old_count);
        std::vector<float> out_means(out_capacity * 3), out_opacity(out_capacity), out_rgb(out_capacity * 3);
        std::vector<float> out_scales(out_capacity * 3), out_rotation(out_capacity * 4);
        std::vector<float> out_shN(sh_coeffs > 0
                                       ? out_capacity * static_cast<std::size_t>(sh_coeffs) * 3
                                       : 0);
        std::vector<std::uint16_t> out_child_count(out_capacity);
        std::vector<std::uint32_t> out_child_start(out_capacity);
        std::vector<std::uint8_t> chunk_bytes;
        std::uint32_t out_fill = 0;

        auto flush_output_chunk = [&]() -> Result<void> {
            if (out_fill == 0) {
                return {};
            }
            const RadStreamChunkSource chunk{
                .count = out_fill,
                .means = out_means.data(),
                .alpha = out_opacity.data(),
                .rgb = out_rgb.data(),
                .scales = out_scales.data(),
                .rotation = out_rotation.data(),
                .shN = sh_coeffs > 0 ? out_shN.data() : nullptr,
                .child_count = out_child_count.data(),
                .child_start = out_child_start.data(),
            };
            if (auto appended = writer.append(chunk); !appended) {
                return make_error(ErrorCode::WRITE_FAILURE, appended.error(), output);
            }
            out_fill = 0;
            return {};
        };

        auto append_decoded_range = [&](std::uint32_t src_offset,
                                        std::uint32_t count) -> Result<void> {
            while (count > 0) {
                const std::uint32_t take = std::min<std::uint32_t>(
                    output_chunk_size - out_fill, count);
                const std::size_t dst = out_fill;
                const std::size_t src = src_offset;
                std::memcpy(out_means.data() + dst * 3,
                            means.data() + src * 3,
                            static_cast<std::size_t>(take) * 3 * sizeof(float));
                std::memcpy(out_opacity.data() + dst,
                            opacity.data() + src,
                            static_cast<std::size_t>(take) * sizeof(float));
                std::memcpy(out_rgb.data() + dst * 3,
                            rgb.data() + src * 3,
                            static_cast<std::size_t>(take) * 3 * sizeof(float));
                std::memcpy(out_scales.data() + dst * 3,
                            scales.data() + src * 3,
                            static_cast<std::size_t>(take) * 3 * sizeof(float));
                std::memcpy(out_rotation.data() + dst * 4,
                            rotation.data() + src * 4,
                            static_cast<std::size_t>(take) * 4 * sizeof(float));
                if (sh_coeffs > 0) {
                    const std::size_t dims = static_cast<std::size_t>(sh_coeffs) * 3;
                    std::memcpy(out_shN.data() + dst * dims,
                                shN.data() + src * dims,
                                static_cast<std::size_t>(take) * dims * sizeof(float));
                }
                std::memcpy(out_child_count.data() + dst,
                            child_count.data() + src,
                            static_cast<std::size_t>(take) * sizeof(std::uint16_t));
                std::memcpy(out_child_start.data() + dst,
                            child_start.data() + src,
                            static_cast<std::size_t>(take) * sizeof(std::uint32_t));

                out_fill += take;
                src_offset += take;
                count -= take;
                if (out_fill == output_chunk_size) {
                    if (auto flushed = flush_output_chunk(); !flushed) {
                        return flushed;
                    }
                }
            }
            return {};
        };

        for (std::size_t c = 0; c < meta.chunks.size(); ++c) {
            const auto& range = meta.chunks[c];
            if (range.bytes > std::numeric_limits<std::size_t>::max()) {
                return make_error(ErrorCode::CORRUPTED_DATA,
                                  std::format("Chunk {} is too large for this platform", c),
                                  input);
            }
            chunk_bytes.resize(static_cast<std::size_t>(range.bytes));
            in.seekg(static_cast<std::streamoff>(info->chunk_area_start + range.offset),
                     std::ios::beg);
            in.read(reinterpret_cast<char*>(chunk_bytes.data()),
                    static_cast<std::streamsize>(chunk_bytes.size()));
            if (!in.good()) {
                return make_error(ErrorCode::CORRUPTED_DATA,
                                  std::format("Failed to read chunk {}", c), input);
            }
            auto parsed = parse_rad_chunk_header(chunk_bytes.data(), chunk_bytes.size());
            if (!parsed) {
                return make_error(ErrorCode::CORRUPTED_DATA, parsed.error(), input);
            }
            const std::uint32_t count = parsed->meta.count;
            if (parsed->meta.base != *range.base || count != *range.count) {
                return make_error(ErrorCode::CORRUPTED_DATA,
                                  std::format("Chunk {} range mismatch", c), input);
            }
            if (count == 0 || count > old_count) {
                return make_error(ErrorCode::CORRUPTED_DATA,
                                  std::format("Chunk {} has invalid count {}", c, count), input);
            }
            if (sh_coeffs > 0) {
                std::fill(shN.begin(), shN.end(), 0.0f);
            }
            if (auto err = decode_chunk_properties(chunk_bytes.data(),
                                                   parsed->meta,
                                                   0,
                                                   parsed->payload_start,
                                                   parsed->has_payload_prefix,
                                                   parsed->chunk_end,
                                                   sh_coeffs,
                                                   means.data(),
                                                   opacity.data(),
                                                   rgb.data(),
                                                   scales.data(),
                                                   rotation.data(),
                                                   sh_coeffs > 0 ? shN.data() : nullptr,
                                                   child_count.data(),
                                                   child_start.data());
                err.has_value()) {
                return make_error(ErrorCode::CORRUPTED_DATA, std::move(*err), input);
            }

            if (auto appended = append_decoded_range(0, count); !appended) {
                return appended;
            }
            if (progress &&
                !progress(static_cast<float>(c + 1) / static_cast<float>(meta.chunks.size()))) {
                return make_error(ErrorCode::CANCELLED, "Re-chunking cancelled", input);
            }
        }
        in.close();
        if (auto flushed = flush_output_chunk(); !flushed) {
            return flushed;
        }
        if (auto finished = writer.finish(); !finished) {
            return make_error(ErrorCode::WRITE_FAILURE, finished.error(), output);
        }
        LOG_INFO("RAD re-chunked: {} -> {} ({} nodes, {}-splat chunks -> {})",
                 lfs::core::path_to_utf8(input), lfs::core::path_to_utf8(output),
                 meta.count, source_chunk, output_chunk_size);
        return {};
    }

    bool rad_paged_load_recommended(const SplatData& data) {
        if (!data.lod_tree || !data.lod_tree->rad_source.valid()) {
            return false;
        }
        // Out-of-core load: only a coarse LOD prefix is resident, so streaming
        // is the only way to reach the remaining nodes.
        if (data.lod_tree->total_nodes() > static_cast<std::size_t>(data.size())) {
            return true;
        }
        const std::size_t logical_chunks = data.lod_tree->chunk_count();
        if (logical_chunks <= 1) {
            return false;
        }
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess || free_bytes == 0) {
            return false;
        }
        const auto tensor_bytes = [](const lfs::core::Tensor& t) -> std::size_t {
            return t.is_valid() ? t.bytes() : 0;
        };
        const std::size_t model_bytes =
            tensor_bytes(data.means_raw()) +
            tensor_bytes(data.sh0_raw()) +
            tensor_bytes(data.shN_raw()) +
            tensor_bytes(data.scaling_raw()) +
            tensor_bytes(data.rotation_raw()) +
            tensor_bytes(data.opacity_raw());
        // Stream when full residency would crowd the GPU: the renderer still
        // needs sort scratch, tile buffers, and framebuffers on top.
        const bool paged = model_bytes > free_bytes / 2;
        if (paged) {
            LOG_INFO("RAD paged load recommended: model={:.1f} MB, free VRAM={:.1f} MB",
                     static_cast<double>(model_bytes) / (1024.0 * 1024.0),
                     static_cast<double>(free_bytes) / (1024.0 * 1024.0));
        }
        return paged;
    }

} // namespace lfs::io
