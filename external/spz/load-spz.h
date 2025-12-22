#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "splat-types.h"

namespace spz {

    // Represents a single inflated gaussian. Each gaussian has 236 bytes. Although the data is easier
    // to intepret in this format, it is not more precise than the packed format, since it was inflated.
    struct UnpackedGaussian {
        std::array<float, 3> position; // x, y, z
        std::array<float, 4> rotation; // x, y, z, w
        std::array<float, 3> scale;    // std::log(scale)
        std::array<float, 3> color;    // rgb sh0 encoding
        float alpha;                   // inverse logistic
        std::array<float, 15> shR;
        std::array<float, 15> shG;
        std::array<float, 15> shB;
    };

    // Represents a single low precision gaussian. Each gaussian has exactly 65 bytes, even if it does
    // not have full spherical harmonics.
    struct PackedGaussian {
        std::array<uint8_t, 9> position{};
        std::array<uint8_t, 4> rotation{};
        std::array<uint8_t, 3> scale{};
        std::array<uint8_t, 3> color{};
        uint8_t alpha = 0;
        std::array<uint8_t, 15> shR{};
        std::array<uint8_t, 15> shG{};
        std::array<uint8_t, 15> shB{};

        UnpackedGaussian unpack(
            bool usesFloat16, bool usesQuaternionSmallestThree, int32_t fractionalBits, const CoordinateConverter& c) const;
    };

    // Represents a full splat with lower precision. Each splat has at most 64 bytes, although splats
    // with fewer spherical harmonics degrees will have less. The data is stored non-interleaved.
    struct PackedGaussians {
        int32_t numPoints = 0;                   // Total number of points (gaussians)
        int32_t shDegree = 0;                    // Degree of spherical harmonics
        int32_t fractionalBits = 0;              // Number of bits used for fractional part of fixed-point coords
        bool antialiased = false;                // Whether gaussians should be rendered with mip-splat antialiasing
        bool usesQuaternionSmallestThree = true; // Whether gaussians use the smallest three method to store quaternions

        std::vector<uint8_t> positions;
        std::vector<uint8_t> scales;
        std::vector<uint8_t> rotations;
        std::vector<uint8_t> alphas;
        std::vector<uint8_t> colors;
        std::vector<uint8_t> sh;

        bool usesFloat16() const;
        PackedGaussian at(int32_t i) const;
        UnpackedGaussian unpack(int32_t i, const CoordinateConverter& c) const;
    };

    struct PackOptions {
        CoordinateSystem from = CoordinateSystem::UNSPECIFIED;
    };

    struct UnpackOptions {
        CoordinateSystem to = CoordinateSystem::UNSPECIFIED;
    };

    // Saves Gaussian splat in packed format, returning a vector of bytes.
    bool saveSpz(
        const GaussianCloud& gaussians, const PackOptions& options, std::vector<uint8_t>* output);

    // Loads Gaussian splat from a vector of bytes in packed format.
    GaussianCloud loadSpz(const std::vector<uint8_t>& data, const UnpackOptions& options);

    // Loads Gaussian splat from a vector of bytes in packed format.
    PackedGaussians loadSpzPacked(const std::string& filename);
    PackedGaussians loadSpzPacked(const uint8_t* data, int32_t size);
    PackedGaussians loadSpzPacked(const std::vector<uint8_t>& data);

    // Saves Gaussian splat in packed format to a file
    bool saveSpz(
        const GaussianCloud& gaussians, const PackOptions& options, const std::string& filename);

    // Loads Gaussian splat from a file in packed format
    GaussianCloud loadSpz(const std::string& filename, const UnpackOptions& o);

    // Loads Gaussian splat from a byte pointer in packed format.
    GaussianCloud loadSpz(const uint8_t* data, int32_t size, const UnpackOptions& options);

    // Saves Gaussian splat data in .ply format
    bool saveSplatToPly(
        const spz::GaussianCloud& gaussians, const PackOptions& options, const std::string& filename);

    // Loads Gaussian splat data in .ply format
    GaussianCloud loadSplatFromPly(const std::string& filename, const UnpackOptions& options);

    void serializePackedGaussians(const PackedGaussians& packed, std::ostream* out);

    bool compressGzipped(const uint8_t* data, size_t size, std::vector<uint8_t>* out);
} // namespace spz
