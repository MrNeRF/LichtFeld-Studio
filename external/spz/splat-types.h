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
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <vector>

#include "splat-c-types.h"
#ifdef SPZ_BUILD_EXTENSIONS
#include "splat-extensions.h"
#endif

namespace spz {
    constexpr int SH_MAX_DEGREE = 4;

    inline SpzFloatBuffer copyFloatBuffer(const std::vector<float>& vector) {
        SpzFloatBuffer buffer = {0, nullptr};
        if (!vector.empty()) {
            buffer.count = vector.size();
            buffer.data = new float[buffer.count];
            std::memcpy(buffer.data, vector.data(), buffer.count * sizeof(float));
        }
        return buffer;
    }

    enum class CoordinateSystem : uint32_t {
        UNSPECIFIED = 0,
        LDB = 1,  // Left Down Back
        RDB = 2,  // Right Down Back
        LUB = 3,  // Left Up Back
        RUB = 4,  // Right Up Back, Three.js coordinate system
        LDF = 5,  // Left Down Front
        RDF = 6,  // Right Down Front, PLY coordinate system
        LUF = 7,  // Left Up Front, GLB coordinate system
        RUF = 8,  // Right Up Front, Unity coordinate system
        LFD = 9,  // Left Front Down
        RFD = 10, // Right Front Down
        LFU = 11, // Left Front Up
        RFU = 12, // Right Front Up
        LBD = 13, // Left Back Down
        RBD = 14, // Right Back Down
        LBU = 15, // Left Back Up
        RBU = 16, // Right Back Up
    };

    using AnalyticRotateShFn = void (*)(float*);

    enum class SplatAttribute : int32_t {
        Positions,
        Alphas,
        Colors,
        Scales,
        Rotations,
        Sh,
    };

    // Iteration order for the six per-point attributes. The emscripten bindings
    // (`slots[]` in spz-bindings.cc) index a parallel array in lockstep — keep the
    // two ordered identically.
    inline constexpr std::array<SplatAttribute, 6> kAllSplatAttributes = {
        SplatAttribute::Positions,
        SplatAttribute::Alphas,
        SplatAttribute::Colors,
        SplatAttribute::Scales,
        SplatAttribute::Rotations,
        SplatAttribute::Sh,
    };

    struct CoordinateConverter {
        std::array<float, 3> flipP = {1.0f, 1.0f, 1.0f}; // x, y, z flips.
        std::array<float, 3> flipQ = {1.0f, 1.0f, 1.0f}; // x, y, z flips, w is never flipped.
        std::array<float, 24> flipSh =                   // Flips for the 24 spherical harmonics coefficients.
            {1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
             1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
             1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
             1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
             1.0f, 1.0f, 1.0f, 1.0f};
        // Per-band SH transform; for cross-family, rotation and flip are baked in together.
        // For within-family, these are empty and flipSh above is used instead.
        std::array<std::function<void(float*)>, SH_MAX_DEGREE> rotFlipShFuncs = {};
        std::function<void(float*)> rotFlipPFunc = nullptr;
        std::function<void(float*)> rotFlipQFunc = nullptr;
    };

    constexpr std::array<bool, 3> axesMatch(CoordinateSystem a, CoordinateSystem b) {
        auto aNum = static_cast<int>(a) - 1;
        auto bNum = static_cast<int>(b) - 1;
        if (aNum < 0 || bNum < 0) {
            return {true, true, true};
        }
        return {
            ((aNum >> 0) & 1) == ((bNum >> 0) & 1),
            ((aNum >> 1) & 1) == ((bNum >> 1) & 1),
            ((aNum >> 2) & 1) == ((bNum >> 2) & 1)};
    }

    constexpr bool needRotation(CoordinateSystem a, CoordinateSystem b) {
        auto aNum = static_cast<int>(a) - 1;
        auto bNum = static_cast<int>(b) - 1;
        if (aNum < 0 || bNum < 0) {
            return false;
        }
        return ((aNum >> 3) & 1) != ((bNum >> 3) & 1);
    }

    // SH rotation matrices for R_x(+π/2) and R_x(-π/2). Each entry operates on the 2l+1
    // coefficients of band l (l = 1..4). The minus table is the transpose (= inverse) of the plus
    // table since both are orthogonal matrices.
    inline constexpr std::array<AnalyticRotateShFn, 4> kAnalyticRotatePlusPiHalfAboutXTable = {
        [](float* p) {
            const float t0 = p[0], t1 = p[1], t2 = p[2];
            p[0] = t1;
            p[1] = -t0;
            p[2] = t2;
        },
        [](float* p) {
            std::array<float, 5> s{};
            for (int i = 0; i < 5; ++i) {
                s[static_cast<size_t>(i)] = p[i];
            }
            const float s3 = std::sqrt(3.f);
            p[0] = s[3];
            p[1] = -s[1];
            p[2] = -0.5f * s[2] - (s3 / 2.f) * s[4];
            p[3] = -s[0];
            p[4] = -(s3 / 2.f) * s[2] + 0.5f * s[4];
        },
        [](float* p) {
            std::array<float, 7> s{};
            for (int i = 0; i < 7; ++i) {
                s[static_cast<size_t>(i)] = p[i];
            }
            const float s15 = std::sqrt(15.f);
            p[0] = -std::sqrt(5.f / 8.f) * s[3] + std::sqrt(3.f / 8.f) * s[5];
            p[1] = -s[1];
            p[2] = -std::sqrt(3.f / 8.f) * s[3] - std::sqrt(5.f / 8.f) * s[5];
            p[3] = std::sqrt(5.f / 8.f) * s[0] + std::sqrt(3.f / 8.f) * s[2];
            p[4] = -0.25f * s[4] - (s15 / 4.f) * s[6];
            p[5] = -std::sqrt(3.f / 8.f) * s[0] + std::sqrt(5.f / 8.f) * s[2];
            p[6] = -(s15 / 4.f) * s[4] + 0.25f * s[6];
        },
        [](float* p) {
            std::array<float, 9> s{};
            for (int i = 0; i < 9; ++i) {
                s[static_cast<size_t>(i)] = p[i];
            }
            const float s2 = std::sqrt(2.f);
            const float s5 = std::sqrt(5.f);
            const float s7 = std::sqrt(7.f);
            const float s14 = std::sqrt(14.f);
            const float s35 = std::sqrt(35.f);
            p[0] = -(s14 / 4.f) * s[5] + (s2 / 4.f) * s[7];
            p[1] = -0.75f * s[1] + (s7 / 4.f) * s[3];
            p[2] = -(s2 / 4.f) * s[5] - (s14 / 4.f) * s[7];
            p[3] = (s7 / 4.f) * s[1] + 0.75f * s[3];
            p[4] = (3.f / 8.f) * s[4] + (s5 / 4.f) * s[6] + (s35 / 8.f) * s[8];
            p[5] = (s14 / 4.f) * s[0] + (s2 / 4.f) * s[2];
            p[6] = (s5 / 4.f) * s[4] + 0.5f * s[6] - (s7 / 4.f) * s[8];
            p[7] = -(s2 / 4.f) * s[0] + (s14 / 4.f) * s[2];
            p[8] = (s35 / 8.f) * s[4] - (s7 / 4.f) * s[6] + 0.125f * s[8];
        },
    };

    inline constexpr std::array<AnalyticRotateShFn, 4> kAnalyticRotateMinusPiHalfAboutXTable = {
        [](float* p) {
            const float t0 = p[0], t1 = p[1], t2 = p[2];
            p[0] = -t1;
            p[1] = t0;
            p[2] = t2;
        },
        [](float* p) {
            std::array<float, 5> s{};
            for (int i = 0; i < 5; ++i)
                s[static_cast<size_t>(i)] = p[i];
            const float s3 = std::sqrt(3.f);
            p[0] = -s[3];
            p[1] = -s[1];
            p[2] = -0.5f * s[2] - (s3 / 2.f) * s[4];
            p[3] = s[0];
            p[4] = -(s3 / 2.f) * s[2] + 0.5f * s[4];
        },
        [](float* p) {
            std::array<float, 7> s{};
            for (int i = 0; i < 7; ++i)
                s[static_cast<size_t>(i)] = p[i];
            const float s15 = std::sqrt(15.f);
            p[0] = std::sqrt(5.f / 8.f) * s[3] - std::sqrt(3.f / 8.f) * s[5];
            p[1] = -s[1];
            p[2] = std::sqrt(3.f / 8.f) * s[3] + std::sqrt(5.f / 8.f) * s[5];
            p[3] = -std::sqrt(5.f / 8.f) * s[0] - std::sqrt(3.f / 8.f) * s[2];
            p[4] = -0.25f * s[4] - (s15 / 4.f) * s[6];
            p[5] = std::sqrt(3.f / 8.f) * s[0] - std::sqrt(5.f / 8.f) * s[2];
            p[6] = -(s15 / 4.f) * s[4] + 0.25f * s[6];
        },
        [](float* p) {
            std::array<float, 9> s{};
            for (int i = 0; i < 9; ++i)
                s[static_cast<size_t>(i)] = p[i];
            const float s2 = std::sqrt(2.f);
            const float s5 = std::sqrt(5.f);
            const float s7 = std::sqrt(7.f);
            const float s14 = std::sqrt(14.f);
            const float s35 = std::sqrt(35.f);
            p[0] = (s14 / 4.f) * s[5] - (s2 / 4.f) * s[7];
            p[1] = -0.75f * s[1] + (s7 / 4.f) * s[3];
            p[2] = (s2 / 4.f) * s[5] + (s14 / 4.f) * s[7];
            p[3] = (s7 / 4.f) * s[1] + 0.75f * s[3];
            p[4] = (3.f / 8.f) * s[4] + (s5 / 4.f) * s[6] + (s35 / 8.f) * s[8];
            p[5] = -(s14 / 4.f) * s[0] - (s2 / 4.f) * s[2];
            p[6] = (s5 / 4.f) * s[4] + 0.5f * s[6] - (s7 / 4.f) * s[8];
            p[7] = (s2 / 4.f) * s[0] - (s14 / 4.f) * s[2];
            p[8] = (s35 / 8.f) * s[4] - (s7 / 4.f) * s[6] + 0.125f * s[8];
        },
    };

    inline CoordinateConverter coordinateConverter(CoordinateSystem from, CoordinateSystem to, int shDegree) {
        CoordinateConverter result;

        if (needRotation(from, to)) {
            // A cross-family conversion decomposes into R_x(±π/2) plus a within-family flip.
            // Shift `from` into the same family as `to` so the recursive call resolves to a flip only.
            const bool backward = ((static_cast<int>(from) - 1) >> 3) & 1;
            const CoordinateSystem innerFrom = backward
                                                   ? static_cast<CoordinateSystem>(static_cast<int>(from) - 8)  // rotated→standard: shift from into standard family
                                                   : static_cast<CoordinateSystem>(static_cast<int>(from) + 8); // standard→rotated: shift from into rotated family
            result = coordinateConverter(innerFrom, to, shDegree);

            // Bake the inner flip into each transform function so callers never need to know the order.
            // Forward (standard→rotated): rotate then flip.
            // Backward (rotated→standard): flip then rotate.
            const auto fp = result.flipP;
            const auto fq = result.flipQ;
            if (backward) {
                result.rotFlipPFunc = [fp](float* p) {
                    p[0] *= fp[0];
                    p[1] *= fp[1];
                    p[2] *= fp[2];
                    const float y = p[1], z = p[2];
                    p[1] = z;
                    p[2] = -y; // R_x(-π/2)
                };
                result.rotFlipQFunc = [fq](float* p) {
                    p[0] *= fq[0];
                    p[1] *= fq[1];
                    p[2] *= fq[2];
                    const float s = std::sqrt(2.f) / 2.f;
                    const float x = p[0], y = p[1], z = p[2], w = p[3];
                    p[0] = s * (-w + x);
                    p[1] = s * (y + z);
                    p[2] = s * (-y + z);
                    p[3] = s * (w + x);
                };
            } else {
                result.rotFlipPFunc = [fp](float* p) {
                    const float y = p[1], z = p[2];
                    p[1] = -z;
                    p[2] = y; // R_x(+π/2)
                    p[0] *= fp[0];
                    p[1] *= fp[1];
                    p[2] *= fp[2];
                };
                result.rotFlipQFunc = [fq](float* p) {
                    const float s = std::sqrt(2.f) / 2.f;
                    const float x = p[0], y = p[1], z = p[2], w = p[3];
                    p[0] = s * (w + x);
                    p[1] = s * (y - z);
                    p[2] = s * (y + z);
                    p[3] = s * (w - x);
                    p[0] *= fq[0];
                    p[1] *= fq[1];
                    p[2] *= fq[2];
                };
            }
            result.flipP = {1.0f, 1.0f, 1.0f};
            result.flipQ = {1.0f, 1.0f, 1.0f};

            const AnalyticRotateShFn* shTable = backward
                                                    ? kAnalyticRotateMinusPiHalfAboutXTable.data()
                                                    : kAnalyticRotatePlusPiHalfAboutXTable.data();
            for (int b = 0; b < shDegree && b < SH_MAX_DEGREE; ++b) {
                const AnalyticRotateShFn rotFn = shTable[b];
                const size_t bandStart = static_cast<size_t>(b * (b + 2));
                const size_t bandSize = static_cast<size_t>(2 * b + 3);
                std::array<float, 9> bandFlip{};
                for (size_t k = 0; k < bandSize; ++k)
                    bandFlip[k] = result.flipSh[bandStart + k];
                if (backward) {
                    result.rotFlipShFuncs[static_cast<size_t>(b)] = [rotFn, bandFlip, bandSize](float* p) {
                        for (size_t k = 0; k < bandSize; ++k)
                            p[k] *= bandFlip[k];
                        rotFn(p);
                    };
                } else {
                    result.rotFlipShFuncs[static_cast<size_t>(b)] = [rotFn, bandFlip, bandSize](float* p) {
                        rotFn(p);
                        for (size_t k = 0; k < bandSize; ++k)
                            p[k] *= bandFlip[k];
                    };
                }
            }
            result.flipSh.fill(1.0f);
            return result;
        }

        auto [xMatch, yMatch, zMatch] = axesMatch(from, to);
        float x = xMatch ? 1.0f : -1.0f;
        float y = yMatch ? 1.0f : -1.0f;
        float z = zMatch ? 1.0f : -1.0f;

        result.flipP = {x, y, z};
        result.flipQ = {y * z, x * z, x * y};
        result.flipSh = {
            y,         // 0
            z,         // 1
            x,         // 2
            x * y,     // 3
            y * z,     // 4
            1.0f,      // 5
            x * z,     // 6
            1.0f,      // 7
            y,         // 8
            x * y * z, // 9
            y,         // 10
            z,         // 11
            x,         // 12
            z,         // 13
            x,         // 14
            // Used https://github.com/nerfstudio-project/gsplat/blob/main/gsplat/cuda/csrc/SphericalHarmonicsCUDA.cu
            // to compute these values.
            x * y, // 15
            y * z, // 16
            x * y, // 17
            y * z, // 18
            1.0f,  // 19
            x * z, // 20
            1.0f,  // 21
            x * z, // 22
            y,     // 23
        };
        return result;
    }

    // A point cloud composed of Gaussians. Each gaussian is represented by:
    //   - xyz position
    //   - xyz scales (on log scale, compute exp(x) to get scale factor)
    //   - xyzw quaternion
    //   - alpha (before sigmoid activation, compute sigmoid(a) to get alpha value between 0 and 1)
    //   - rgb color (as SH DC component, compute 0.5 + 0.282095 * x to get color value between 0 and 1)
    //   - 0 to 71 spherical harmonics coefficients (see comment below)
    struct GaussianCloud {
        // Total number of points (gaussians) in this splat.
        int32_t numPoints = 0;

        // Degree of spherical harmonics for this splat.
        int32_t shDegree = 0;

        // Whether the gaussians should be rendered in antialiased mode (mip splatting)
        bool antialiased = false;

        // See block comment above for details
        std::vector<float> positions;
        std::vector<float> scales;
        std::vector<float> rotations;
        std::vector<float> alphas;
        std::vector<float> colors;

        // Spherical harmonics coefficients. The number of coefficients per point depends on shDegree:
        //   0 -> 0
        //   1 -> 9   (3 coeffs x 3 channels)
        //   2 -> 24  (8 coeffs x 3 channels)
        //   3 -> 45  (15 coeffs x 3 channels)
        //   4 -> 72  (24 coeffs x 3 channels)
        // The color channel is the inner (fastest varying) axis, and the coefficient is the outer
        // (slower varying) axis, i.e. for degree 1, the order of the 9 values is:
        //   sh1n1_r, sh1n1_g, sh1n1_b, sh10_r, sh10_g, sh10_b, sh1p1_r, sh1p1_g, sh1p1_b
        std::vector<float> sh;

#ifdef SPZ_BUILD_EXTENSIONS
        std::vector<SpzExtensionBasePtr> extensions; // List of extensions, if any
#endif

        // The caller is responsible for freeing the pointers in the returned GaussianCloudData
        GaussianCloudData data() const {
            GaussianCloudData data;
            data.numPoints = numPoints;
            data.shDegree = shDegree;
            data.antialiased = antialiased;
            data.positions = copyFloatBuffer(positions);
            data.scales = copyFloatBuffer(scales);
            data.rotations = copyFloatBuffer(rotations);
            data.alphas = copyFloatBuffer(alphas);
            data.colors = copyFloatBuffer(colors);
            data.sh = copyFloatBuffer(sh);
#ifdef SPZ_BUILD_EXTENSIONS
            data.extensions = copyExtensions(extensions);
#else
            data.extensions = nullptr;
#endif
            return data;
        }

        // Convert between two coordinate systems, for example from RDF (ply format) to RUB (used by spz).
        // This is performed in-place.
        void convertCoordinates(CoordinateSystem from, CoordinateSystem to) {
            if (numPoints == 0) {
                // There is nothing to convert.
                return;
            }
            CoordinateConverter c = coordinateConverter(from, to, shDegree);
            // Positions: call transform (which includes flip for cross-family), else apply flipP.
            if (c.rotFlipPFunc) {
                for (size_t i = 0; i < positions.size(); i += 3) {
                    c.rotFlipPFunc(positions.data() + i);
                }
            } else {
                for (size_t i = 0; i < positions.size(); i += 3) {
                    positions[i + 0] *= c.flipP[0];
                    positions[i + 1] *= c.flipP[1];
                    positions[i + 2] *= c.flipP[2];
                }
            }
            // Rotations: same pattern. Within-family only flips x,y,z; w is untouched. Cross-family rotates all four.
            if (c.rotFlipQFunc) {
                for (size_t i = 0; i < rotations.size(); i += 4) {
                    c.rotFlipQFunc(rotations.data() + i);
                }
            } else {
                for (size_t i = 0; i < rotations.size(); i += 4) {
                    rotations[i + 0] *= c.flipQ[0];
                    rotations[i + 1] *= c.flipQ[1];
                    rotations[i + 2] *= c.flipQ[2];
                    // Don't modify rotations[i + 3] (w component)
                }
            }
            const size_t numCoeffs = sh.size() / 3;
            const size_t numCoeffsPerPoint = numCoeffs / numPoints;
            if (c.rotFlipShFuncs[0]) {
                // Cross-family: rotation+flip baked into rotFlipShFuncs per band.
                for (size_t coeffBase = 0; coeffBase < numCoeffs; coeffBase += numCoeffsPerPoint) {
                    for (int band = 0; band < shDegree && band < SH_MAX_DEGREE; ++band) {
                        const size_t bandStart = static_cast<size_t>(band * (band + 2));
                        const size_t bandSize = static_cast<size_t>(2 * band + 3);
                        if (bandStart + bandSize > numCoeffsPerPoint) {
                            break;
                        }
                        std::array<float, 9> tmp{};
                        for (int channel = 0; channel < 3; ++channel) {
                            for (size_t k = 0; k < bandSize; ++k) {
                                tmp[k] = sh[(coeffBase + bandStart + k) * 3 + static_cast<size_t>(channel)];
                            }
                            c.rotFlipShFuncs[static_cast<size_t>(band)](tmp.data());
                            for (size_t k = 0; k < bandSize; ++k) {
                                sh[(coeffBase + bandStart + k) * 3 + static_cast<size_t>(channel)] = tmp[k];
                            }
                        }
                    }
                }
            } else {
                // Within-family: rotate spherical harmonics by inverting coefficients that reference the
                // y and z axes, for each RGB channel. See spherical_harmonics_kernel_impl.h for spherical
                // harmonics formulas.
                for (size_t coeffBase = 0; coeffBase < numCoeffs; coeffBase += numCoeffsPerPoint) {
                    for (size_t j = 0; j < numCoeffsPerPoint; ++j) {
                        const size_t base = (coeffBase + j) * 3;
                        sh[base + 0] *= c.flipSh[j];
                        sh[base + 1] *= c.flipSh[j];
                        sh[base + 2] *= c.flipSh[j];
                    }
                }
            }
        }

        // Rotates the GaussianCloud by 180 degrees about the x axis (converts from RUB to RDF coordinates
        // and vice versa. This is performed in-place.
        void rotate180DegAboutX() { convertCoordinates(CoordinateSystem::RUB, CoordinateSystem::RDF); }

        float medianVolume() const {
            if (numPoints == 0) {
                return 0.01f;
            }
            // The volume of an ellipsoid is 4/3 * pi * x * y * z, where x, y, and z are the radii on each
            // axis. Scales are stored on a log scale, and exp(x) * exp(y) * exp(z) = exp(x + y + z). So we
            // can sort by value = (x + y + z) and compute volume = 4/3 * pi * exp(value) later.
            std::vector<float> scaleSums;
            scaleSums.reserve(scales.size() / 3);
            for (size_t i = 0; i < scales.size(); i += 3) {
                scaleSums.push_back(scales[i] + scales[i + 1] + scales[i + 2]);
            }
            const auto mid = scaleSums.begin() + scaleSums.size() / 2;
            std::nth_element(scaleSums.begin(), mid, scaleSums.end());
            return static_cast<float>((M_PI * 4 / 3) * exp(*mid));
        }
    };

    // SPZ Splat math helpers, lightweight implementations of vector and quaternion math.
    using Vec3f = std::array<float, 3>;  // x, y, z
    using Quat4f = std::array<float, 4>; // w, x, y, z
    using Half = uint16_t;

    // Half-precision helpers.
    float halfToFloat(Half h);
    Half floatToHalf(float f);

    // Vector helpers.
    Vec3f normalized(const Vec3f& v);
    float norm(const Vec3f& a);

    // Quaternion helpers.
    float norm(const Quat4f& q);
    inline Quat4f normalized(const Quat4f& v) {
        float norm = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]);
        return {v[0] / norm, v[1] / norm, v[2] / norm, v[3] / norm};
    }
    Quat4f axisAngleQuat(const Vec3f& scaledAxis);

    // Constexpr helpers.
    constexpr Vec3f vec3f(const float* data) { return {data[0], data[1], data[2]}; }

    constexpr float dot(const Vec3f& a, const Vec3f& b) {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    }

    constexpr float squaredNorm(const Vec3f& v) { return dot(v, v); }

    constexpr Quat4f quat4f(const float* data) { return {data[0], data[1], data[2], data[3]}; }

    constexpr Vec3f times(const Quat4f& q, const Vec3f& p) {
        auto [w, x, y, z] = q;
        auto [vx, vy, vz] = p;
        auto x2 = x + x;
        auto y2 = y + y;
        auto z2 = z + z;
        auto wx2 = w * x2;
        auto wy2 = w * y2;
        auto wz2 = w * z2;
        auto xx2 = x * x2;
        auto xy2 = x * y2;
        auto xz2 = x * z2;
        auto yy2 = y * y2;
        auto yz2 = y * z2;
        auto zz2 = z * z2;
        return {
            vx * (1.0f - (yy2 + zz2)) + vy * (xy2 - wz2) + vz * (xz2 + wy2),
            vx * (xy2 + wz2) + vy * (1.0f - (xx2 + zz2)) + vz * (yz2 - wx2),
            vx * (xz2 - wy2) + vy * (yz2 + wx2) + vz * (1.0f - (xx2 + yy2))};
    }

    inline Quat4f times(const Quat4f& a, const Quat4f& b) {
        auto [w, x, y, z] = a;
        auto [qw, qx, qy, qz] = b;
        return normalized(std::array<float, 4>{
            w * qw - x * qx - y * qy - z * qz,
            w * qx + x * qw + y * qz - z * qy,
            w * qy - x * qz + y * qw + z * qx,
            w * qz + x * qy - y * qx + z * qw});
    }

    constexpr Quat4f times(const Quat4f& a, float s) {
        return {a[0] * s, a[1] * s, a[2] * s, a[3] * s};
    }

    constexpr Quat4f plus(const Quat4f& a, const Quat4f& b) {
        return {a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]};
    }

    constexpr Vec3f times(const Vec3f& v, float s) { return {v[0] * s, v[1] * s, v[2] * s}; }

    constexpr Vec3f plus(const Vec3f& a, const Vec3f& b) {
        return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
    }

    constexpr Vec3f times(const Vec3f& a, const Vec3f& b) {
        return {a[0] * b[0], a[1] * b[1], a[2] * b[2]};
    }

} // namespace spz
