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

#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <streambuf>
#include <string>
#include <vector>

#include "splat-types.h"
#ifdef SPZ_BUILD_EXTENSIONS
#include "splat-extensions.h"
#endif

#ifdef ANDROID
#include <android/log.h>
#endif

namespace spz {

    // These limits bound all allocations driven by an untrusted SPZ stream. They
    // remain comfortably above practical scenes while keeping the gzip/zstd and
    // expanded-float peaks finite.
    inline constexpr size_t kMaxSpzCompressedBytes = 1ULL * 1024 * 1024 * 1024;
    inline constexpr size_t kMaxSpzDecompressedBytes = 2ULL * 1024 * 1024 * 1024;
    inline constexpr uint32_t kMaxSpzPoints = 30'000'000;
    inline constexpr uint8_t kMaxSpzFractionalBits = 24;

#ifdef ANDROID
    static constexpr char LOG_TAG[] = "SPZ";
    template <class... Args>
    static void SpzLog(const char* fmt, Args&&... args) {
        __android_log_print(ANDROID_LOG_INFO, LOG_TAG, fmt, std::forward<Args>(args)...);
    }
#else
    template <class... Args>
    static void SpzLog(const char* fmt, Args&&... args) {
        printf(fmt, std::forward<Args>(args)...);
        printf("\n");
        fflush(stdout);
    }
#endif // ANDROID

    template <class... Args>
    static void SpzLog(const char* fmt) {
        SpzLog("%s", fmt);
    }

// Routine per-operation status (load banners, extension found/writing notices, etc.).
// Compiles to a no-op unless SPZ_VERBOSE_LOGGING is defined; errors/warnings stay on SpzLog.
#ifdef SPZ_VERBOSE_LOGGING
    template <class... Args>
    static void SpzLogDebug(const char* fmt, Args&&... args) {
        SpzLog(fmt, std::forward<Args>(args)...);
    }
    template <class... Args>
    static void SpzLogDebug(const char* fmt) {
        SpzLog("%s", fmt);
    }
#else
    template <class... Args>
    static void SpzLogDebug(const char*, Args&&...) {}
    template <class... Args>
    static void SpzLogDebug(const char*) {}
#endif // SPZ_VERBOSE_LOGGING

    constexpr int SH_MAX_COEFFS = 24; // Maximum number of SH coefficients (degree 0..4)
    constexpr int DEFAULT_SH1_BITS = 5;
    constexpr int DEFAULT_SH_REST_BITS = 4;

    // Latest version of the packed format, update this when changing the format.
    constexpr int LATEST_SPZ_HEADER_VERSION = 4;

    // Minimum version of the ZSTD-compressed SPZ format.
    constexpr int MIN_ZSTD_SPZ_HEADER_VERSION = 4;

    // Minimum version of that uses SmallestThree quaternions.
    constexpr int MIN_SMALLEST_THREE_QUATERNIONS_VERSION = 3;

    // NGSP Magic Number used for spz file identification.
    constexpr uint32_t NGSP_MAGIC = 0x5053474e;

    // Represents a single inflated gaussian. Each gaussian has 344 bytes (position, rotation, scale,
    // color, alpha, and 24 SH coeffs x 3 channels). Although the data is easier to interpret in this
    // format, it is not more precise than the packed format, since it was inflated.
    struct UnpackedGaussian {
        std::array<float, 3> position; // x, y, z
        std::array<float, 4> rotation; // x, y, z, w
        std::array<float, 3> scale;    // std::log(scale)
        std::array<float, 3> color;    // rgb sh0 encoding
        float alpha;                   // inverse logistic
        std::array<float, SH_MAX_COEFFS> shR;
        std::array<float, SH_MAX_COEFFS> shG;
        std::array<float, SH_MAX_COEFFS> shB;
    };

    // Represents a single low precision gaussian. Each gaussian has exactly 92 bytes (for degree 4
    // spherical harmonics); the struct layout is fixed so that at() can index into non-interleaved buffers.
    struct PackedGaussian {
        std::array<uint8_t, 9> position{};
        std::array<uint8_t, 4> rotation{};
        std::array<uint8_t, 3> scale{};
        std::array<uint8_t, 3> color{};
        uint8_t alpha = 0;
        std::array<uint8_t, SH_MAX_COEFFS> shR{};
        std::array<uint8_t, SH_MAX_COEFFS> shG{};
        std::array<uint8_t, SH_MAX_COEFFS> shB{};

        UnpackedGaussian unpack(
            bool usesFloat16, bool usesQuaternionSmallestThree, int32_t fractionalBits, const CoordinateConverter& c) const;
    };

    // Represents a full splat with lower precision. Each splat has at most 92 bytes (for degree 4
    // spherical harmonics), although splats with fewer spherical harmonics degrees will have less.
    // The data is stored non-interleaved.
    struct PackedGaussians {
        uint32_t version = LATEST_SPZ_HEADER_VERSION; // Version of the packed format
        int32_t numPoints = 0;                        // Total number of points (gaussians)
        int32_t shDegree = 0;                         // Degree of spherical harmonics
        int32_t fractionalBits = 0;                   // Number of bits used for fractional part of fixed-point coords
        bool antialiased = false;                     // Whether gaussians should be rendered with mip-splat antialiasing
        bool usesQuaternionSmallestThree = true;      // Whether gaussians use the smallest three method to store quaternions
        bool hadSkippedExtensions = false;            // True when extensions were present in the file but ignored at load time

        std::vector<uint8_t> positions;
        std::vector<uint8_t> scales;
        std::vector<uint8_t> rotations;
        std::vector<uint8_t> alphas;
        std::vector<uint8_t> colors;
        std::vector<uint8_t> sh;

#ifdef SPZ_BUILD_EXTENSIONS
        std::vector<SpzExtensionBasePtr> extensions; // List of extensions, if any
#endif

        bool usesFloat16() const;
        PackedGaussian at(int32_t i) const;
        UnpackedGaussian unpack(int32_t i, const CoordinateConverter& c) const;
    };

    struct PackOptions {
        uint32_t version = LATEST_SPZ_HEADER_VERSION; // Version of the packed format

        CoordinateSystem from = CoordinateSystem::UNSPECIFIED;

        // Quantization bits are only used during packing to reduce information entropy for g-zipping.
        // Unpacking doesn't need these values since g-unzipping already fills zero bits for quantized data.
        uint8_t sh1Bits = DEFAULT_SH1_BITS;        // Bits for SH degree 1 coefficients
        uint8_t shRestBits = DEFAULT_SH_REST_BITS; // Bits for SH degree 2+ coefficients
    };

    struct UnpackOptions {
        CoordinateSystem to = CoordinateSystem::UNSPECIFIED;
    };

    // Structure for PLY extra elements (non-vertex elements)
    struct PlyExtraElement {
        std::string name;
        int32_t count;
        size_t bytesPerElement;
        bool isKnown; // true for elements we explicitly handle (like safe_orbit)
    };

    // Saves Gaussian splat in packed format, returning a vector of bytes.
    bool saveSpz(
        const GaussianCloud& gaussians, const PackOptions& options, std::vector<uint8_t>* output);

    // Loads Gaussian splat from a vector of bytes in packed format.
    GaussianCloud loadSpz(const std::vector<uint8_t>& data, const UnpackOptions& options);

    // Loads Gaussian splat from a file / byte pointer / vector in packed format.
    PackedGaussians loadSpzPacked(const std::string& filename);
    PackedGaussians loadSpzPacked(const uint8_t* data, size_t size);
    PackedGaussians loadSpzPacked(const std::vector<uint8_t>& data);

    // Saves Gaussian splat in packed format to a file
    bool saveSpz(
        const GaussianCloud& gaussians, const PackOptions& options, const std::string& filename);

    // Loads Gaussian splat from a file in packed format
    GaussianCloud loadSpz(const std::string& filename, const UnpackOptions& o);

    // Loads Gaussian splat from a byte pointer in packed format.
    GaussianCloud loadSpz(const uint8_t* data, size_t size, const UnpackOptions& options);

    // Saves Gaussian splat data in .ply format
    bool saveSplatToPly(
        const spz::GaussianCloud& gaussians, const PackOptions& options, const std::string& filename);

    // Loads Gaussian splat data in .ply format
    GaussianCloud loadSplatFromPly(const std::string& filename, const UnpackOptions& options);

    // LichtFeld patch: windowed read-only streambuf over a contiguous byte
    // range. Get-area windows stay at most INT_MAX bytes because MSVC stores
    // the get-area length as int (LichtFeld #1697). deserializePackedGaussians
    // seeks (tellg / seekg-end / restore), so seeking is implemented with
    // 64-bit math. window_bytes is a defaulted constructor argument so tests
    // can install a tiny window.
    class membuf : public std::streambuf {
    public:
        membuf(const uint8_t* data, size_t size,
               size_t window_bytes = static_cast<size_t>(std::numeric_limits<int>::max()))
            : data_(data == nullptr
                        ? nullptr
                        : reinterpret_cast<char*>(const_cast<uint8_t*>(data))),
              size_(data == nullptr ? 0 : static_cast<uint64_t>(size)),
              window_bytes_(std::clamp(
                  window_bytes, size_t{1},
                  static_cast<size_t>(std::numeric_limits<int>::max()))) {
            set_window(0);
        }

    protected:
        int_type underflow() override {
            if (gptr() < egptr()) {
                return traits_type::to_int_type(*gptr());
            }
            const uint64_t pos = logical_position();
            if (pos >= size_) {
                return traits_type::eof();
            }
            set_window(pos);
            if (gptr() < egptr()) {
                return traits_type::to_int_type(*gptr());
            }
            return traits_type::eof();
        }

        std::streamsize xsgetn(char* s, std::streamsize count) override {
            if (s == nullptr || count <= 0) {
                return 0;
            }
            const uint64_t pos = logical_position();
            if (pos >= size_) {
                return 0;
            }
            const uint64_t want = static_cast<uint64_t>(count);
            const uint64_t take = std::min(want, size_ - pos);
            std::memcpy(s, data_ + pos, static_cast<size_t>(take));
            set_window(pos + take);
            return static_cast<std::streamsize>(take);
        }

        pos_type seekoff(off_type offset, std::ios_base::seekdir direction,
                         std::ios_base::openmode mode) override {
            if ((mode & std::ios_base::in) == 0) {
                return pos_type(off_type(-1));
            }
            uint64_t base = 0;
            if (direction == std::ios_base::beg) {
                base = 0;
            } else if (direction == std::ios_base::cur) {
                base = logical_position();
            } else if (direction == std::ios_base::end) {
                base = size_;
            } else {
                return pos_type(off_type(-1));
            }

            uint64_t target = 0;
            if (offset >= 0) {
                const uint64_t add = static_cast<uint64_t>(offset);
                if (add > size_ - base) {
                    return pos_type(off_type(-1));
                }
                target = base + add;
            } else {
                uint64_t mag = 0;
                if (offset == std::numeric_limits<off_type>::min()) {
                    mag = static_cast<uint64_t>(std::numeric_limits<off_type>::max()) + 1;
                } else {
                    mag = static_cast<uint64_t>(-offset);
                }
                if (mag > base) {
                    return pos_type(off_type(-1));
                }
                target = base - mag;
            }

            const uint64_t window_len =
                (eback() != nullptr && egptr() != nullptr)
                    ? static_cast<uint64_t>(egptr() - eback())
                    : 0;
            if (eback() != nullptr && target >= window_start_ &&
                target <= window_start_ + window_len) {
                setg(eback(),
                     eback() + static_cast<std::ptrdiff_t>(target - window_start_),
                     egptr());
            } else {
                set_window(target);
            }
            return pos_type(static_cast<off_type>(target));
        }

        pos_type seekpos(pos_type position, std::ios_base::openmode mode) override {
            return seekoff(static_cast<off_type>(position), std::ios_base::beg, mode);
        }

    private:
        uint64_t logical_position() const {
            if (eback() == nullptr || gptr() == nullptr) {
                return window_start_;
            }
            return window_start_ + static_cast<uint64_t>(gptr() - eback());
        }

        void set_window(uint64_t pos) {
            window_start_ = pos > size_ ? size_ : pos;
            if (data_ == nullptr || window_start_ >= size_) {
                if (data_ == nullptr) {
                    setg(nullptr, nullptr, nullptr);
                    return;
                }
                char* const first = data_ + static_cast<size_t>(window_start_);
                setg(first, first, first);
                return;
            }
            const uint64_t remaining = size_ - window_start_;
            const uint64_t count =
                std::min(remaining, static_cast<uint64_t>(window_bytes_));
            char* const first = data_ + static_cast<size_t>(window_start_);
            setg(first, first, first + static_cast<std::ptrdiff_t>(count));
        }

        char* data_ = nullptr;
        uint64_t size_ = 0;
        uint64_t window_start_ = 0;
        size_t window_bytes_ = 1;
    };

    void serializePackedGaussians(const PackedGaussians& packed, std::ostream* out);

    bool compressGzipped(const uint8_t* data, size_t size, std::vector<uint8_t>* out);

    // Returns true if the build has extension support enabled, false otherwise
    bool hasExtensionSupport();
} // namespace spz
