/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/cube_face_projection.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace lfs::io {

    namespace {
        constexpr float kPi = 3.14159265358979323846f;

        // Anisotropic (EWA) resampling footprint, after Heckbert 1989.
        //
        // The source-space filter covariance is
        //     V = sigma_dst^2 * J * J^T + sigma_src^2 * I
        // where J = d(source pixel) / d(face pixel) is the analytic Jacobian of the
        // gnomonic-to-equirectangular map. The first term is the destination pixel's
        // footprint pushed into source space; it is what prefilters the panorama
        // where the face minifies it, and it is what a plain bilinear fetch is
        // missing. The second term is the source reconstruction kernel, which keeps
        // magnified regions smooth instead of blocky.
        //
        // This matters because the map is minifying at the centre of every face and
        // magnifying towards the corners, so no fixed kernel is correct across a
        // single face.
        //
        // Both sigmas are variance-matched to a one-pixel box: a box of width w has
        // variance w^2/12. Matching variance rather than picking a sigma by eye is
        // what makes the filter reduce to bilinear sharpness at unity magnification
        // (a tent has variance 1/6, and 1/12 + 1/12 = 1/6), so this only ever blurs
        // more than the old bilinear path where the map is genuinely minifying.
        constexpr float kBoxSigma = 0.28867513f; // 1 / sqrt(12)
        constexpr float kDstSigma = kBoxSigma;   // destination pixel footprint
        constexpr float kSrcSigma = kBoxSigma;   // source pixel reconstruction
        constexpr float kFilterRadiusSigmas = 3.0f;

        // A face pixel near a pole spans a huge azimuth range in the source. Rather
        // than clamp the footprint (which would alias) or walk every pixel in it
        // (which would be unbounded work), stride through the footprint so the
        // estimate stays representative and the tap count stays fixed.
        constexpr int kMaxTapsPerAxis = 16;
        constexpr float kMinHalfExtent = 1.0f;

        // Guard for looking exactly along +/-Y, where azimuth is undefined.
        constexpr float kMinHorizontalNorm = 1e-6f;

        [[nodiscard]] float wrap_source_x(float x, const int width) {
            const float w = static_cast<float>(width);
            x = std::fmod(x, w);
            if (x < 0.0f)
                x += w;
            return x;
        }

        [[nodiscard]] int wrap_index(int x, const int width) {
            x %= width;
            if (x < 0)
                x += width;
            return x;
        }

        // Covariance of the source-space resampling filter for one output pixel.
        struct FilterCovariance {
            float v00, v01, v11;
        };

        // p is the (unnormalised) view direction for this output pixel. The face
        // basis rows give dp/d(face_x) = right = basis[0..2] and
        // dp/d(face_y) = up = basis[3..5].
        [[nodiscard]] FilterCovariance filter_covariance(
            const float px, const float py, const float pz,
            const std::array<float, 9>& basis,
            const float inv_focal,
            const int width, const int height) {
            const float* const right = basis.data();
            const float* const up = basis.data() + 3;
            const float h2 = std::max(px * px + pz * pz, kMinHorizontalNorm);
            const float h = std::sqrt(h2);
            const float n2 = h2 + py * py;

            // d(azimuth)/dp with azimuth = atan2(px, pz)
            const float gaz_x = pz / h2;
            const float gaz_z = -px / h2;

            // d(elevation)/dp with elevation = asin(py / |p|)
            const float gel_x = -py * px / (h * n2);
            const float gel_y = h / n2;
            const float gel_z = -py * pz / (h * n2);

            // Angular derivatives to source pixel derivatives.
            const float az_to_px = static_cast<float>(width) / (2.0f * kPi);
            const float el_to_py = static_cast<float>(height) / kPi;

            const float j00 = az_to_px * (gaz_x * right[0] + gaz_z * right[2]) * inv_focal;
            const float j01 = az_to_px * (gaz_x * up[0] + gaz_z * up[2]) * inv_focal;
            const float j10 = el_to_py * (gel_x * right[0] + gel_y * right[1] + gel_z * right[2]) * inv_focal;
            const float j11 = el_to_py * (gel_x * up[0] + gel_y * up[1] + gel_z * up[2]) * inv_focal;

            constexpr float dst_var = kDstSigma * kDstSigma;
            constexpr float src_var = kSrcSigma * kSrcSigma;

            return {
                dst_var * (j00 * j00 + j01 * j01) + src_var,
                dst_var * (j00 * j10 + j01 * j11),
                dst_var * (j10 * j10 + j11 * j11) + src_var};
        }
    } // namespace

    int projected_face_output_size(
        const lfs::core::CubeFaceProjection& projection,
        const int resize_factor,
        const int max_width) {
        int size = std::max(1, projection.face_size);
        if (resize_factor > 1) {
            size = std::max(1, size / resize_factor);
        }
        if (max_width > 0 && size > max_width) {
            size = max_width;
        }
        return size;
    }

    std::vector<uint8_t> project_cube_face_hwc(
        const uint8_t* source,
        const int width,
        const int height,
        const int channels,
        const lfs::core::CubeFaceProjection& projection,
        const int output_size) {
        if (!source || width <= 0 || height <= 0 || channels <= 0 || output_size <= 0) {
            throw std::runtime_error("Invalid spherical projection input");
        }

        std::vector<uint8_t> projected(static_cast<size_t>(output_size) * output_size * channels);

        const float focal = lfs::core::cube_face_focal(projection.fov_degrees, output_size);
        const float inv_focal = 1.0f / focal;
        const float center = 0.5f * static_cast<float>(output_size);
        const auto& m = projection.pano_to_face;

        constexpr float radius2 = kFilterRadiusSigmas * kFilterRadiusSigmas;
        std::vector<float> accum(static_cast<size_t>(channels));

        for (int y = 0; y < output_size; ++y) {
            const float face_y = (static_cast<float>(y) + 0.5f - center) * inv_focal;
            for (int x = 0; x < output_size; ++x) {
                const float face_x = (static_cast<float>(x) + 0.5f - center) * inv_focal;
                constexpr float face_z = 1.0f;

                const float pano_x = m[0] * face_x + m[3] * face_y + m[6] * face_z;
                const float pano_y = m[1] * face_x + m[4] * face_y + m[7] * face_z;
                const float pano_z = m[2] * face_x + m[5] * face_y + m[8] * face_z;

                const float inv_len = 1.0f / std::sqrt(pano_x * pano_x + pano_y * pano_y + pano_z * pano_z);
                const float dir_x = pano_x * inv_len;
                const float dir_y = pano_y * inv_len;
                const float dir_z = pano_z * inv_len;

                const float azimuth = std::atan2(dir_x, dir_z);
                const float elevation = std::asin(std::clamp(dir_y, -1.0f, 1.0f));
                const float src_x = wrap_source_x(
                    (azimuth / (2.0f * kPi) + 0.5f) * static_cast<float>(width) - 0.5f, width);
                const float src_y = std::clamp(
                    (elevation / kPi + 0.5f) * static_cast<float>(height) - 0.5f,
                    0.0f,
                    static_cast<float>(height - 1));

                const auto [v00, v01, v11] = filter_covariance(
                    pano_x, pano_y, pano_z, m, inv_focal, width, height);

                // Inverse covariance defines the quadratic form of the elliptical
                // Gaussian; sqrt of the diagonal gives the axis-aligned extent.
                const float det = std::max(v00 * v11 - v01 * v01, 1e-12f);
                const float vi00 = v11 / det;
                const float vi01 = -v01 / det;
                const float vi11 = v00 / det;

                const float half_x = std::max(kMinHalfExtent, kFilterRadiusSigmas * std::sqrt(v00));
                const float half_y = std::max(kMinHalfExtent, kFilterRadiusSigmas * std::sqrt(v11));

                const int stride_x = std::max(1, static_cast<int>(std::ceil(2.0f * half_x / kMaxTapsPerAxis)));
                const int stride_y = std::max(1, static_cast<int>(std::ceil(2.0f * half_y / kMaxTapsPerAxis)));

                const int y_begin = static_cast<int>(std::floor(src_y - half_y));
                const int y_end = static_cast<int>(std::ceil(src_y + half_y));
                const int x_begin = static_cast<int>(std::floor(src_x - half_x));
                const int x_end = static_cast<int>(std::ceil(src_x + half_x));

                std::fill(accum.begin(), accum.end(), 0.0f);
                float weight_sum = 0.0f;

                for (int sy = y_begin; sy <= y_end; sy += stride_y) {
                    const float dy = static_cast<float>(sy) - src_y;
                    // Clamping past the poles replicates the top/bottom row, which is
                    // the same convention the forward mapping uses.
                    const int row = std::clamp(sy, 0, height - 1);
                    const size_t row_base = static_cast<size_t>(row) * width;

                    for (int sx = x_begin; sx <= x_end; sx += stride_x) {
                        const float dx = static_cast<float>(sx) - src_x;
                        const float q = vi00 * dx * dx + 2.0f * vi01 * dx * dy + vi11 * dy * dy;
                        if (q > radius2)
                            continue;

                        const float w = std::exp(-0.5f * q);
                        const size_t base = (row_base + static_cast<size_t>(wrap_index(sx, width))) *
                                            static_cast<size_t>(channels);
                        for (int c = 0; c < channels; ++c) {
                            accum[static_cast<size_t>(c)] += w * static_cast<float>(source[base + c]);
                        }
                        weight_sum += w;
                    }
                }

                const size_t dst_base = (static_cast<size_t>(y) * output_size + x) * channels;
                if (weight_sum > 0.0f) {
                    const float inv_weight = 1.0f / weight_sum;
                    for (int c = 0; c < channels; ++c) {
                        projected[dst_base + c] = static_cast<uint8_t>(std::clamp(
                            static_cast<int>(std::lround(accum[static_cast<size_t>(c)] * inv_weight)), 0, 255));
                    }
                } else {
                    // Degenerate footprint: fall back to the nearest source texel.
                    const size_t base = (static_cast<size_t>(std::clamp(static_cast<int>(std::lround(src_y)), 0, height - 1)) *
                                             width +
                                         static_cast<size_t>(wrap_index(static_cast<int>(std::lround(src_x)), width))) *
                                        static_cast<size_t>(channels);
                    for (int c = 0; c < channels; ++c) {
                        projected[dst_base + c] = source[base + c];
                    }
                }
            }
        }

        return projected;
    }

} // namespace lfs::io
