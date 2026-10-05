/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/assert.hpp"
#include "core/cuda_error.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "flip.cuh"
#include "training/kernels/kernel_stream.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <vector>

namespace lfs::training {

    namespace {
        constexpr int ROW_THREADS = 256;
        constexpr int MAX_RADIUS = 64;
        constexpr int HORIZONTAL_PLANES = 7;
        constexpr int FEATURE_PLANES = 5;

        // linear sRGB to XYZ and back, and the XYZ of linear white used as the reference illuminant.
        constexpr float RGB_TO_XYZ[9] = {
            10135552.0f / 24577794.0f, 8788810.0f / 24577794.0f, 4435075.0f / 24577794.0f,
            2613072.0f / 12288897.0f, 8788810.0f / 12288897.0f, 887015.0f / 12288897.0f,
            1425312.0f / 73733382.0f, 8788810.0f / 73733382.0f, 70074185.0f / 73733382.0f};
        constexpr float XYZ_TO_RGB[9] = {
            3.241003275f, -1.537398934f, -0.498615861f,
            -0.969224334f, 1.875930071f, 0.041554224f,
            0.055639423f, -0.204011202f, 1.057148933f};
        constexpr float WHITE_X = RGB_TO_XYZ[0] + RGB_TO_XYZ[1] + RGB_TO_XYZ[2];
        constexpr float WHITE_Z = RGB_TO_XYZ[6] + RGB_TO_XYZ[7] + RGB_TO_XYZ[8];

        constexpr float COLOR_EXPONENT = 0.7f;
        constexpr float FEATURE_EXPONENT = 0.5f;
        constexpr float COLOR_CUTOFF = 0.4f;
        constexpr float COLOR_CUTOFF_ERROR = 0.95f;
        constexpr float FEATURE_WIDTH_DEGREES = 0.082f;

        __constant__ uint8_t MAGMA[256 * 3] = {
            0, 0, 4, 1, 0, 5, 1, 1, 6, 1, 1, 8, 2, 1, 9, 2, 2, 11, 2, 2, 13, 3, 3, 15,
            3, 3, 18, 4, 4, 20, 5, 4, 22, 6, 5, 24, 6, 5, 26, 7, 6, 28, 8, 7, 30, 9, 7, 32,
            10, 8, 34, 11, 9, 36, 12, 9, 38, 13, 10, 41, 14, 11, 43, 16, 11, 45, 17, 12, 47, 18, 13, 49,
            19, 13, 52, 20, 14, 54, 21, 14, 56, 22, 15, 59, 24, 15, 61, 25, 16, 63, 26, 16, 66, 28, 16, 68,
            29, 17, 71, 30, 17, 73, 32, 17, 75, 33, 17, 78, 34, 17, 80, 36, 18, 83, 37, 18, 85, 39, 18, 88,
            41, 17, 90, 42, 17, 92, 44, 17, 95, 45, 17, 97, 47, 17, 99, 49, 17, 101, 51, 16, 103, 52, 16, 105,
            54, 16, 107, 56, 16, 108, 57, 15, 110, 59, 15, 112, 61, 15, 113, 63, 15, 114, 64, 15, 116, 66, 15, 117,
            68, 15, 118, 69, 16, 119, 71, 16, 120, 73, 16, 120, 74, 16, 121, 76, 17, 122, 78, 17, 123, 79, 18, 123,
            81, 18, 124, 82, 19, 124, 84, 19, 125, 86, 20, 125, 87, 21, 126, 89, 21, 126, 90, 22, 126, 92, 22, 127,
            93, 23, 127, 95, 24, 127, 96, 24, 128, 98, 25, 128, 100, 26, 128, 101, 26, 128, 103, 27, 128, 104, 28, 129,
            106, 28, 129, 107, 29, 129, 109, 29, 129, 110, 30, 129, 112, 31, 129, 114, 31, 129, 115, 32, 129, 117, 33, 129,
            118, 33, 129, 120, 34, 129, 121, 34, 130, 123, 35, 130, 124, 35, 130, 126, 36, 130, 128, 37, 130, 129, 37, 129,
            131, 38, 129, 132, 38, 129, 134, 39, 129, 136, 39, 129, 137, 40, 129, 139, 41, 129, 140, 41, 129, 142, 42, 129,
            144, 42, 129, 145, 43, 129, 147, 43, 128, 148, 44, 128, 150, 44, 128, 152, 45, 128, 153, 45, 128, 155, 46, 127,
            156, 46, 127, 158, 47, 127, 160, 47, 127, 161, 48, 126, 163, 48, 126, 165, 49, 126, 166, 49, 125, 168, 50, 125,
            170, 51, 125, 171, 51, 124, 173, 52, 124, 174, 52, 123, 176, 53, 123, 178, 53, 123, 179, 54, 122, 181, 54, 122,
            183, 55, 121, 184, 55, 121, 186, 56, 120, 188, 57, 120, 189, 57, 119, 191, 58, 119, 192, 58, 118, 194, 59, 117,
            196, 60, 117, 197, 60, 116, 199, 61, 115, 200, 62, 115, 202, 62, 114, 204, 63, 113, 205, 64, 113, 207, 64, 112,
            208, 65, 111, 210, 66, 111, 211, 67, 110, 213, 68, 109, 214, 69, 108, 216, 69, 108, 217, 70, 107, 219, 71, 106,
            220, 72, 105, 222, 73, 104, 223, 74, 104, 224, 76, 103, 226, 77, 102, 227, 78, 101, 228, 79, 100, 229, 80, 100,
            231, 82, 99, 232, 83, 98, 233, 84, 98, 234, 86, 97, 235, 87, 96, 236, 88, 96, 237, 90, 95, 238, 91, 94,
            239, 93, 94, 240, 95, 94, 241, 96, 93, 242, 98, 93, 242, 100, 92, 243, 101, 92, 244, 103, 92, 244, 105, 92,
            245, 107, 92, 246, 108, 92, 246, 110, 92, 247, 112, 92, 247, 114, 92, 248, 116, 92, 248, 118, 92, 249, 120, 93,
            249, 121, 93, 249, 123, 93, 250, 125, 94, 250, 127, 94, 250, 129, 95, 251, 131, 95, 251, 133, 96, 251, 135, 97,
            252, 137, 97, 252, 138, 98, 252, 140, 99, 252, 142, 100, 252, 144, 101, 253, 146, 102, 253, 148, 103, 253, 150, 104,
            253, 152, 105, 253, 154, 106, 253, 155, 107, 254, 157, 108, 254, 159, 109, 254, 161, 110, 254, 163, 111, 254, 165, 113,
            254, 167, 114, 254, 169, 115, 254, 170, 116, 254, 172, 118, 254, 174, 119, 254, 176, 120, 254, 178, 122, 254, 180, 123,
            254, 182, 124, 254, 183, 126, 254, 185, 127, 254, 187, 129, 254, 189, 130, 254, 191, 132, 254, 193, 133, 254, 194, 135,
            254, 196, 136, 254, 198, 138, 254, 200, 140, 254, 202, 141, 254, 204, 143, 254, 205, 144, 254, 207, 146, 254, 209, 148,
            254, 211, 149, 254, 213, 151, 254, 215, 153, 254, 216, 154, 253, 218, 156, 253, 220, 158, 253, 222, 160, 253, 224, 161,
            253, 226, 163, 253, 227, 165, 253, 229, 167, 253, 231, 169, 253, 233, 170, 253, 235, 172, 252, 236, 174, 252, 238, 176,
            252, 240, 178, 252, 242, 180, 252, 244, 182, 252, 246, 184, 252, 247, 185, 252, 249, 187, 252, 251, 189, 252, 253, 191};

        struct Lab {
            float l;
            float a;
            float b;
        };

        __host__ __device__ inline float srgb_to_linear(const float c) {
            return c <= 0.04045f ? c / 12.92f : powf((c + 0.055f) / 1.055f, 2.4f);
        }

        __host__ __device__ inline float lab_f(const float t) {
            constexpr float delta = 6.0f / 29.0f;
            return t > delta * delta * delta ? cbrtf(t) : t / (3.0f * delta * delta) + 4.0f / 29.0f;
        }

        __host__ __device__ inline Lab hunt_adjusted_lab(const float r, const float g, const float b) {
            const float x = (RGB_TO_XYZ[0] * r + RGB_TO_XYZ[1] * g + RGB_TO_XYZ[2] * b) / WHITE_X;
            const float y = RGB_TO_XYZ[3] * r + RGB_TO_XYZ[4] * g + RGB_TO_XYZ[5] * b;
            const float z = (RGB_TO_XYZ[6] * r + RGB_TO_XYZ[7] * g + RGB_TO_XYZ[8] * b) / WHITE_Z;
            const float fx = lab_f(x);
            const float fy = lab_f(y);
            const float fz = lab_f(z);
            const float l = 116.0f * fy - 16.0f;
            return {l, 0.01f * l * 500.0f * (fx - fy), 0.01f * l * 200.0f * (fy - fz)};
        }

        __host__ __device__ inline float hyab(const Lab& p, const Lab& q) {
            const float da = p.a - q.a;
            const float db = p.b - q.b;
            return fabsf(p.l - q.l) + sqrtf(da * da + db * db);
        }

        __device__ inline int clamp_to_edge(const int index, const int size) {
            return min(max(index, 0), size - 1);
        }

        struct FlipTaps {
            int csf_radius;
            int feature_radius;
            const float* csf_horizontal; // 4 rows: achromatic, red-green, blue-yellow narrow, blue-yellow wide
            const float* csf_vertical;   // the same Gaussians scaled so each channel's 2D filter sums to one
            const float* feature;        // 3 rows: edge (first derivative), point (second derivative), Gaussian
        };

        // Each block converts one row segment plus halo to YCxCz and luminance once, then filters it.
        __global__ void flip_horizontal_kernel(const float* __restrict__ image, const int width, const int height,
                                               const FlipTaps taps, float* __restrict__ planes) {
            extern __shared__ float row[];
            const int radius = max(taps.csf_radius, taps.feature_radius);
            const int span = ROW_THREADS + 2 * radius;
            const int y = blockIdx.y;
            const int x0 = blockIdx.x * ROW_THREADS;
            const size_t pixels = static_cast<size_t>(width) * height;
            for (int i = threadIdx.x; i < span; i += ROW_THREADS) {
                const size_t source = static_cast<size_t>(y) * width + clamp_to_edge(x0 + i - radius, width);
                const float r = srgb_to_linear(image[source]);
                const float g = srgb_to_linear(image[pixels + source]);
                const float b = srgb_to_linear(image[2 * pixels + source]);
                const float x = (RGB_TO_XYZ[0] * r + RGB_TO_XYZ[1] * g + RGB_TO_XYZ[2] * b) / WHITE_X;
                const float lum = RGB_TO_XYZ[3] * r + RGB_TO_XYZ[4] * g + RGB_TO_XYZ[5] * b;
                const float z = (RGB_TO_XYZ[6] * r + RGB_TO_XYZ[7] * g + RGB_TO_XYZ[8] * b) / WHITE_Z;
                row[i] = 116.0f * lum - 16.0f;
                row[span + i] = 500.0f * (x - lum);
                row[2 * span + i] = 200.0f * (lum - z);
                row[3 * span + i] = lum;
            }
            __syncthreads();
            const int x = x0 + threadIdx.x;
            if (x >= width)
                return;
            const int csf_taps = 2 * taps.csf_radius + 1;
            float sums[HORIZONTAL_PLANES] = {};
            for (int k = 0; k < csf_taps; ++k) {
                const int i = threadIdx.x + radius - taps.csf_radius + k;
                sums[0] += taps.csf_horizontal[k] * row[i];
                sums[1] += taps.csf_horizontal[csf_taps + k] * row[span + i];
                sums[2] += taps.csf_horizontal[2 * csf_taps + k] * row[2 * span + i];
                sums[3] += taps.csf_horizontal[3 * csf_taps + k] * row[2 * span + i];
            }
            const int feature_taps = 2 * taps.feature_radius + 1;
            for (int k = 0; k < feature_taps; ++k) {
                const float lum = row[3 * span + threadIdx.x + radius - taps.feature_radius + k];
                sums[4] += taps.feature[k] * lum;
                sums[5] += taps.feature[feature_taps + k] * lum;
                sums[6] += taps.feature[2 * feature_taps + k] * lum;
            }
            const size_t out = static_cast<size_t>(y) * width + x;
            for (int p = 0; p < HORIZONTAL_PLANES; ++p)
                planes[p * pixels + out] = sums[p];
        }

        // Finishes the separable filters. Without a reference it stores Hunt-adjusted Lab and the edge and point
        // magnitudes; with one it writes the FLIP error against it.
        __global__ void flip_vertical_kernel(const float* __restrict__ planes, const int width, const int height,
                                             const FlipTaps taps, const float* __restrict__ reference,
                                             float* __restrict__ features, float* __restrict__ error,
                                             const float max_color_error) {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y;
            if (x >= width)
                return;
            const size_t pixels = static_cast<size_t>(width) * height;
            const int csf_taps = 2 * taps.csf_radius + 1;
            float opponent[3] = {};
            for (int k = 0; k < csf_taps; ++k) {
                const size_t source = static_cast<size_t>(clamp_to_edge(y + k - taps.csf_radius, height)) * width + x;
                opponent[0] += taps.csf_vertical[k] * planes[source];
                opponent[1] += taps.csf_vertical[csf_taps + k] * planes[pixels + source];
                opponent[2] += taps.csf_vertical[2 * csf_taps + k] * planes[2 * pixels + source] +
                               taps.csf_vertical[3 * csf_taps + k] * planes[3 * pixels + source];
            }
            const int feature_taps = 2 * taps.feature_radius + 1;
            float edge_x = 0.0f, edge_y = 0.0f, point_x = 0.0f, point_y = 0.0f;
            for (int k = 0; k < feature_taps; ++k) {
                const size_t source = static_cast<size_t>(clamp_to_edge(y + k - taps.feature_radius, height)) * width + x;
                const float gaussian = taps.feature[2 * feature_taps + k];
                edge_x += gaussian * planes[4 * pixels + source];
                point_x += gaussian * planes[5 * pixels + source];
                edge_y += taps.feature[k] * planes[6 * pixels + source];
                point_y += taps.feature[feature_taps + k] * planes[6 * pixels + source];
            }

            const float lum = (opponent[0] + 16.0f) / 116.0f;
            const float cx = (opponent[1] / 500.0f + lum) * WHITE_X;
            const float cz = (lum - opponent[2] / 200.0f) * WHITE_Z;
            const float r = fminf(fmaxf(XYZ_TO_RGB[0] * cx + XYZ_TO_RGB[1] * lum + XYZ_TO_RGB[2] * cz, 0.0f), 1.0f);
            const float g = fminf(fmaxf(XYZ_TO_RGB[3] * cx + XYZ_TO_RGB[4] * lum + XYZ_TO_RGB[5] * cz, 0.0f), 1.0f);
            const float b = fminf(fmaxf(XYZ_TO_RGB[6] * cx + XYZ_TO_RGB[7] * lum + XYZ_TO_RGB[8] * cz, 0.0f), 1.0f);
            const Lab lab = hunt_adjusted_lab(r, g, b);
            const float edge = sqrtf(edge_x * edge_x + edge_y * edge_y);
            const float point = sqrtf(point_x * point_x + point_y * point_y);
            const size_t out = static_cast<size_t>(y) * width + x;
            if (!reference) {
                features[out] = lab.l;
                features[pixels + out] = lab.a;
                features[2 * pixels + out] = lab.b;
                features[3 * pixels + out] = edge;
                features[4 * pixels + out] = point;
                return;
            }

            const Lab reference_lab{reference[out], reference[pixels + out], reference[2 * pixels + out]};
            const float color = powf(hyab(reference_lab, lab), COLOR_EXPONENT);
            const float cutoff = COLOR_CUTOFF * max_color_error;
            const float color_error =
                color < cutoff ? color * (COLOR_CUTOFF_ERROR / cutoff)
                               : COLOR_CUTOFF_ERROR + (color - cutoff) / (max_color_error - cutoff) * (1.0f - COLOR_CUTOFF_ERROR);
            const float feature_difference = fmaxf(fabsf(reference[3 * pixels + out] - edge),
                                                   fabsf(reference[4 * pixels + out] - point));
            const float feature_error = powf(feature_difference * 0.70710678f, FEATURE_EXPONENT);
            error[out] = powf(color_error, 1.0f - feature_error);
        }

        __global__ void flip_magma_kernel(const float* __restrict__ error, const size_t pixels, uint8_t* __restrict__ image) {
            const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= pixels)
                return;
            const int index = static_cast<int>(fminf(fmaxf(error[i], 0.0f), 1.0f) * 255.0f + 0.5f);
            for (int c = 0; c < 3; ++c)
                image[c * pixels + i] = MAGMA[3 * index + c];
        }

        // Spatial contrast sensitivity as a sum of Gaussians, a * sqrt(pi / b) * exp(-pi^2 d^2 / b), d in degrees.
        struct GaussianTerm {
            float a;
            float b;
        };

        std::vector<float> flip_taps(const float pixels_per_degree, int& csf_radius, int& feature_radius) {
            constexpr double pi = 3.14159265358979323846;
            const std::array<std::array<GaussianTerm, 2>, 3> channels{{{{{1.0f, 0.0047f}, {0.0f, 1e-5f}}},
                                                                       {{{1.0f, 0.0053f}, {0.0f, 1e-5f}}},
                                                                       {{{34.1f, 0.04f}, {13.5f, 0.025f}}}}};
            csf_radius = static_cast<int>(std::ceil(3.0 * std::sqrt(0.04 / (2.0 * pi * pi)) * pixels_per_degree));
            const double sigma = 0.5 * FEATURE_WIDTH_DEGREES * pixels_per_degree;
            feature_radius = static_cast<int>(std::ceil(3.0 * sigma));
            LFS_ASSERT_MSG(csf_radius <= MAX_RADIUS && feature_radius <= MAX_RADIUS,
                           "FLIP pixels per degree is too large for the filter radius limit");

            const int csf_taps = 2 * csf_radius + 1;
            const auto gaussian = [&](const GaussianTerm term, const int offset) {
                const double d = offset / static_cast<double>(pixels_per_degree);
                return std::exp(-pi * pi * d * d / term.b);
            };
            const std::array<std::pair<int, int>, 4> terms{{{0, 0}, {1, 0}, {2, 0}, {2, 1}}};
            std::vector<float> taps(static_cast<size_t>(8 * csf_taps + 3 * (2 * feature_radius + 1)));
            for (int c = 0; c < 3; ++c) {
                double total = 0.0;
                for (const auto& term : channels[c]) {
                    double line = 0.0;
                    for (int k = -csf_radius; k <= csf_radius; ++k)
                        line += gaussian(term, k);
                    total += term.a * std::sqrt(pi / term.b) * line * line;
                }
                for (int t = 0; t < 4; ++t) {
                    if (terms[t].first != c)
                        continue;
                    const auto term = channels[c][terms[t].second];
                    const double weight = term.a * std::sqrt(pi / term.b) / total;
                    for (int k = -csf_radius; k <= csf_radius; ++k) {
                        taps[t * csf_taps + k + csf_radius] = static_cast<float>(gaussian(term, k));
                        taps[(4 + t) * csf_taps + k + csf_radius] = static_cast<float>(weight * gaussian(term, k));
                    }
                }
            }

            const int feature_taps = 2 * feature_radius + 1;
            std::vector<double> edge(feature_taps), point(feature_taps), smooth(feature_taps);
            double smooth_sum = 0.0;
            for (int k = -feature_radius; k <= feature_radius; ++k) {
                const double g = std::exp(-(k * k) / (2.0 * sigma * sigma));
                edge[k + feature_radius] = -k * g;
                point[k + feature_radius] = (k * k / (sigma * sigma) - 1.0) * g;
                smooth[k + feature_radius] = g;
                smooth_sum += g;
            }
            // Positive and negative lobes each sum to one, as in the 2D detectors.
            for (auto* kernel : {&edge, &point}) {
                double positive = 0.0, negative = 0.0;
                for (const double v : *kernel)
                    (v > 0.0 ? positive : negative) += v;
                for (double& v : *kernel)
                    v = v > 0.0 ? v / positive : (v < 0.0 ? v / -negative : 0.0);
            }
            float* feature = taps.data() + 8 * csf_taps;
            for (int k = 0; k < feature_taps; ++k) {
                feature[k] = static_cast<float>(edge[k]);
                feature[feature_taps + k] = static_cast<float>(point[k]);
                feature[2 * feature_taps + k] = static_cast<float>(smooth[k] / smooth_sum);
            }
            return taps;
        }
    } // namespace

    lfs::core::Tensor flip_error_map(const lfs::core::Tensor& reference, const lfs::core::Tensor& test,
                                     const float pixels_per_degree, cudaStream_t stream) {
        LFS_ASSERT_MSG(reference.device() == lfs::core::Device::CUDA && reference.dtype() == lfs::core::DataType::Float32 &&
                           reference.ndim() == 3 && reference.shape()[0] == 3,
                       "FLIP reference must be a CUDA Float32 [3,H,W] tensor");
        LFS_ASSERT_MSG(test.shape() == reference.shape() && test.device() == lfs::core::Device::CUDA &&
                           test.dtype() == lfs::core::DataType::Float32,
                       "FLIP test image must match the reference");
        LFS_ASSERT_MSG(pixels_per_degree > 0.0f, "FLIP pixels per degree must be positive");
        stream = resolve_stream(stream);
        nvtxRangePush("flip_error_map");
        const lfs::core::CUDAStreamGuard stream_guard(stream);
        reference.sync_to_stream(stream);
        test.sync_to_stream(stream);

        int csf_radius = 0;
        int feature_radius = 0;
        const auto host_taps = flip_taps(pixels_per_degree, csf_radius, feature_radius);
        const auto device_taps = lfs::core::Tensor::from_vector(host_taps, {host_taps.size()}, lfs::core::Device::CUDA);
        const int csf_taps = 2 * csf_radius + 1;
        const FlipTaps taps{csf_radius, feature_radius, device_taps.ptr<float>(),
                            device_taps.ptr<float>() + 4 * csf_taps, device_taps.ptr<float>() + 8 * csf_taps};

        const Lab green = hunt_adjusted_lab(0.0f, 1.0f, 0.0f);
        const Lab blue = hunt_adjusted_lab(0.0f, 0.0f, 1.0f);
        const float max_color_error = std::pow(hyab(green, blue), COLOR_EXPONENT);

        const size_t height = reference.shape()[1];
        const size_t width = reference.shape()[2];
        const int w = static_cast<int>(width);
        const int h = static_cast<int>(height);
        auto planes = lfs::core::Tensor::empty({HORIZONTAL_PLANES, height, width}, lfs::core::Device::CUDA);
        auto features = lfs::core::Tensor::empty({FEATURE_PLANES, height, width}, lfs::core::Device::CUDA);
        auto error = lfs::core::Tensor::empty({height, width}, lfs::core::Device::CUDA);
        const dim3 row_grid((w + ROW_THREADS - 1) / ROW_THREADS, h);
        const size_t shared = 4 * (ROW_THREADS + 2 * std::max(csf_radius, feature_radius)) * sizeof(float);
        const auto source_reference = reference.contiguous();
        const auto source_test = test.contiguous();

        flip_horizontal_kernel<<<row_grid, ROW_THREADS, shared, stream>>>(source_reference.ptr<float>(), w, h, taps,
                                                                          planes.ptr<float>());
        LFS_CUDA_LAUNCH_CHECK(stream, "training.flip.reference_rows");
        flip_vertical_kernel<<<row_grid, ROW_THREADS, 0, stream>>>(planes.ptr<float>(), w, h, taps, nullptr,
                                                                   features.ptr<float>(), nullptr, max_color_error);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.flip.reference_columns");
        flip_horizontal_kernel<<<row_grid, ROW_THREADS, shared, stream>>>(source_test.ptr<float>(), w, h, taps,
                                                                          planes.ptr<float>());
        LFS_CUDA_LAUNCH_CHECK(stream, "training.flip.test_rows");
        flip_vertical_kernel<<<row_grid, ROW_THREADS, 0, stream>>>(planes.ptr<float>(), w, h, taps,
                                                                   features.ptr<float>(), nullptr, error.ptr<float>(),
                                                                   max_color_error);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.flip.test_columns");
        nvtxRangePop();
        return error;
    }

    lfs::core::Tensor flip_error_image(const lfs::core::Tensor& error_map, cudaStream_t stream) {
        LFS_ASSERT_MSG(error_map.device() == lfs::core::Device::CUDA && error_map.dtype() == lfs::core::DataType::Float32 &&
                           error_map.ndim() == 2,
                       "FLIP error image needs a CUDA Float32 [H,W] error map");
        stream = resolve_stream(stream);
        const lfs::core::CUDAStreamGuard stream_guard(stream);
        error_map.sync_to_stream(stream);
        const auto source = error_map.contiguous();
        const size_t pixels = source.numel();
        auto image = lfs::core::Tensor::empty({3, source.shape()[0], source.shape()[1]}, lfs::core::Device::CUDA,
                                              lfs::core::DataType::UInt8);
        if (pixels == 0)
            return image;
        flip_magma_kernel<<<static_cast<unsigned>((pixels + ROW_THREADS - 1) / ROW_THREADS), ROW_THREADS, 0, stream>>>(
            source.ptr<float>(), pixels, image.ptr<uint8_t>());
        LFS_CUDA_LAUNCH_CHECK(stream, "training.flip.magma");
        return image;
    }

} // namespace lfs::training
