/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Host-side undistortion parameters: the COLMAP-style undistorted camera for a
// distorted one, and its rescaling to the image size actually loaded. The lens
// inverse itself runs only on the GPU (inverse_distortion_sample_map).

#include "core/camera_distortion_layout.hpp"
#include "core/logger.hpp"
#include "core/tensor.hpp"
#include "core/tensor_image.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace lfs::core {

    namespace {

        constexpr float PIXEL_CENTER_OFFSET = 0.5f;
        constexpr float COLMAP_MIN_SCALE = 0.2f;
        constexpr float COLMAP_MAX_SCALE = 2.0f;

        // Host twin of the radial branch scan in undistort.cu. image_warp.slang takes the branch in
        // its camera block, and the crop gives border pixels without a ray the fold boundary.
        // Pixels past the first turning point of the lens map, or past the horizon, have no ray.
        constexpr float BRANCH_MAX_THETA = 1.570795f;
        constexpr int BRANCH_SAMPLES = 1024;
        constexpr float NO_BOUND = std::numeric_limits<float>::infinity();

        using RadialBranch = UndistortRadialBranch;

        void radial_mapping(const float r, const UndistortParams& params,
                            float& value, float& derivative, float& denominator) {
            const auto* k = params.distortion;
            const bool fisheye = params.model_type != CameraModelType::PINHOLE;
            const float s = r * r;
            const float k1 = params.num_distortion > 0 ? k[0] : 0.0f;
            const float k2 = params.num_distortion > 1 ? k[1] : 0.0f;
            const float k3 = params.num_distortion > 2 ? k[2] : 0.0f;
            const float k4 = fisheye && params.num_distortion > 3 ? k[3] : 0.0f;
            const float numerator = 1.0f + s * (k1 + s * (k2 + s * (k3 + s * k4)));
            const float dn = k1 + s * (2.0f * k2 + s * (3.0f * k3 + s * 4.0f * k4));
            const bool rational = !fisheye && params.num_distortion >= 6;
            denominator = rational ? 1.0f + s * (k[3] + s * (k[4] + s * k[5])) : 1.0f;
            const float dd = rational ? k[3] + s * (2.0f * k[4] + s * 3.0f * k[5]) : 0.0f;
            const float radial = numerator / denominator;
            value = r * radial;
            derivative = radial + 2.0f * s * (dn - radial * dd) / denominator;
        }

        RadialBranch radial_branch(const UndistortParams& params) {
            const bool fisheye = params.model_type != CameraModelType::PINHOLE;
            RadialBranch branch{fisheye ? BRANCH_MAX_THETA : NO_BOUND, NO_BOUND};
            // Scanning atan(r) covers pinhole radii without a lens-specific cutoff.
            int first = BRANCH_SAMPLES + 1;
            for (int i = 1; i <= BRANCH_SAMPLES; ++i) {
                const float angle = BRANCH_MAX_THETA * (static_cast<float>(i) / BRANCH_SAMPLES);
                float value, derivative, denominator;
                radial_mapping(fisheye ? angle : std::tan(angle), params, value, derivative, denominator);
                if (derivative <= 0.0f || denominator <= 0.0f) {
                    first = i;
                    break;
                }
            }
            if (first <= BRANCH_SAMPLES) {
                float lo = BRANCH_MAX_THETA * (static_cast<float>(first - 1) / BRANCH_SAMPLES);
                float hi = BRANCH_MAX_THETA * (static_cast<float>(first) / BRANCH_SAMPLES);
                for (int i = 0; i < 24; ++i) {
                    const float mid = 0.5f * (lo + hi);
                    float value, derivative, denominator;
                    radial_mapping(fisheye ? mid : std::tan(mid), params, value, derivative, denominator);
                    if (derivative > 0.0f && denominator > 0.0f)
                        lo = mid;
                    else
                        hi = mid;
                }
                branch.bound = fisheye ? lo : std::tan(lo);
            }
            if (std::isfinite(branch.bound)) {
                float derivative, denominator;
                radial_mapping(branch.bound, params, branch.maximum, derivative, denominator);
            }
            return branch;
        }

        // Rays of one source frame edge from the GPU lens inverse, NaN where a pixel has none. A
        // one-pixel-wide sample map along the edge has the edge's pixel centres.
        std::vector<float> edge_rays(UndistortParams params, const int edge) {
            if (edge < 2) {
                if (edge == 1)
                    params.src_cx -= static_cast<float>(params.src_width - 1);
                params.src_width = 1;
            } else {
                if (edge == 3)
                    params.src_cy -= static_cast<float>(params.src_height - 1);
                params.src_height = 1;
            }
            return inverse_distortion_sample_map(params, nullptr).cpu().to_vector();
        }

        // Traces the frame border through the GPU lens inverse; border samples without a ray
        // take the fold boundary. Keeps COLMAP's scale rules and fails below the minimum scale.
        bool solve_undistort_crop(const UndistortParams& params, const float blank_pixels,
                                  int& crop_width, int& crop_height) {
            const auto branch = radial_branch(params);
            const float radius = params.model_type == CameraModelType::PINHOLE ? branch.bound : std::tan(branch.bound);
            std::array<float, 4> low;
            std::array<float, 4> high;
            low.fill(NO_BOUND);
            high.fill(-NO_BOUND);
            for (int edge = 0; edge < 4; ++edge) {
                const int length = edge < 2 ? params.src_height : params.src_width;
                const auto rays = edge_rays(params, edge);
                for (int i = 0; i < length; ++i) {
                    const float px = edge < 2 ? (edge == 0 ? PIXEL_CENTER_OFFSET : params.src_width - PIXEL_CENTER_OFFSET)
                                              : i + PIXEL_CENTER_OFFSET;
                    const float py = edge >= 2 ? (edge == 2 ? PIXEL_CENTER_OFFSET : params.src_height - PIXEL_CENTER_OFFSET)
                                               : i + PIXEL_CENTER_OFFSET;
                    const float xd = (px - params.src_cx) / params.src_fx;
                    const float yd = (py - params.src_cy) / params.src_fy;
                    float ux = rays[2 * i];
                    float uy = rays[2 * i + 1];
                    if (!std::isfinite(ux) || !std::isfinite(uy)) {
                        const float rd = std::hypot(xd, yd);
                        if (!std::isfinite(radius) || rd <= 0.0f)
                            continue;
                        ux = xd * radius / rd;
                        uy = yd * radius / rd;
                    }
                    const float value = edge < 2 ? params.src_fx * ux : params.src_fy * uy;
                    low[edge] = std::fmin(low[edge], value);
                    high[edge] = std::fmax(high[edge], value);
                }
            }
            float scale[2];
            for (int axis = 0; axis < 2; ++axis) {
                const float center = axis == 0 ? params.src_cx : params.src_cy;
                const float size = static_cast<float>(axis == 0 ? params.src_width : params.src_height);
                const int near_edge = 2 * axis;
                const int far_edge = near_edge + 1;
                const float outer = std::fmin(center / -low[near_edge], (size - PIXEL_CENTER_OFFSET - center) / high[far_edge]);
                const float inner = std::fmax(center / -high[near_edge], (size - PIXEL_CENTER_OFFSET - center) / low[far_edge]);
                const float blended = outer * blank_pixels + inner * (1.0f - blank_pixels);
                if (!std::isfinite(blended) || blended <= 0.0f)
                    return false;
                scale[axis] = std::fmin(COLMAP_MAX_SCALE, std::fmax(COLMAP_MIN_SCALE, 1.0f / blended));
            }
            const float corner = std::hypot(
                std::fmax(params.src_cx, params.src_width - params.src_cx) * scale[0] / params.src_fx,
                std::fmax(params.src_cy, params.src_height - params.src_cy) * scale[1] / params.src_fy);
            if (corner >= radius) {
                const float shrink = std::nextafter(radius, 0.0f) / corner;
                scale[0] *= shrink;
                scale[1] *= shrink;
            }
            if (scale[0] < COLMAP_MIN_SCALE || scale[1] < COLMAP_MIN_SCALE)
                return false;
            crop_width = std::max(1, static_cast<int>(params.src_width * scale[0]));
            crop_height = std::max(1, static_cast<int>(params.src_height * scale[1]));
            return true;
        }

    } // namespace

    UndistortRadialBranch undistort_radial_branch(const UndistortParams& params) {
        return radial_branch(params);
    }

    UndistortParams detail::initialize_undistort_params(
        float fx, float fy, float cx, float cy,
        int width, int height,
        const Tensor& radial, const Tensor& tangential,
        CameraModelType model) {

        UndistortParams params{};
        params.src_fx = fx;
        params.src_fy = fy;
        params.src_cx = cx;
        params.src_cy = cy;
        params.src_width = width;
        params.src_height = height;
        params.model_type = model;

        const auto coefficients = [](const Tensor& tensor) {
            std::vector<float> values;
            if (tensor.is_valid() && tensor.numel() > 0) {
                assert(tensor.ndim() == 1);
                auto cpu = tensor.cpu();
                auto acc = cpu.accessor<float, 1>();
                for (size_t i = 0; i < cpu.numel(); ++i)
                    values.push_back(acc(i));
            }
            return values;
        };
        const auto rad_vec = coefficients(radial);
        const auto tan_vec = coefficients(tangential);

        switch (model) {
        case CameraModelType::PINHOLE: {
            if (rad_vec.size() > 3 && rad_vec.size() != 6) {
                throw std::invalid_argument(
                    "Pinhole distortion requires at most three polynomial radial coefficients or six rational radial coefficients");
            }
            if (!tan_vec.empty() && tan_vec.size() != 2) {
                throw std::invalid_argument(
                    "Pinhole distortion supports exactly two tangential coefficients");
            }
            break;
        }

        case CameraModelType::FISHEYE:
            if (rad_vec.size() > 4) {
                throw std::invalid_argument(
                    "Fisheye distortion supports at most four radial coefficients");
            }
            if (std::any_of(tan_vec.begin(), tan_vec.end(), [](const float value) {
                    return value != 0.0f;
                })) {
                throw std::invalid_argument(
                    "Fisheye distortion does not support tangential coefficients");
            }
            break;

        case CameraModelType::THIN_PRISM_FISHEYE: {
            if (rad_vec.size() > 4 || tan_vec.size() > 6) {
                throw std::invalid_argument(
                    "Thin prism fisheye distortion supports four radial and six tangential or prism coefficients");
            }
            break;
        }

        default:
            if (!rad_vec.empty() || !tan_vec.empty()) {
                throw std::invalid_argument(
                    "Distortion coefficients are unsupported by this camera model");
            }
            break;
        }

        params.num_distortion = packCameraDistortion(model, rad_vec, tan_vec, std::span(params.distortion));

        params.dst_fx = fx;
        params.dst_fy = fy;
        params.dst_cx = cx;
        params.dst_cy = cy;
        params.dst_width = width;
        params.dst_height = height;

        return params;
    }

    UndistortParams compute_undistort_params(
        float fx, float fy, float cx, float cy,
        int width, int height,
        const Tensor& radial, const Tensor& tangential,
        CameraModelType model, float blank_pixels) {

        UndistortParams params = detail::initialize_undistort_params(
            fx, fy, cx, cy, width, height, radial, tangential, model);

        const bool distorted =
            model != CameraModelType::PINHOLE ||
            std::any_of(params.distortion, params.distortion + params.num_distortion,
                        [](const float value) { return value != 0.0f; });
        if (!distorted) {
            return params;
        }

        int crop_width = 0;
        int crop_height = 0;
        if (width <= 0 || height <= 0 || fx <= 0.0f || fy <= 0.0f ||
            !solve_undistort_crop(params, blank_pixels, crop_width, crop_height)) {
            params.crop_solve_failed = true;
            return params;
        }
        params.dst_width = crop_width;
        params.dst_height = crop_height;
        params.dst_cx = cx * static_cast<float>(params.dst_width) / static_cast<float>(width);
        params.dst_cy = cy * static_cast<float>(params.dst_height) / static_cast<float>(height);

        LOG_DEBUG("Undistort: {}x{} -> {}x{}, fx={:.1f}->{:.1f}, fy={:.1f}->{:.1f}",
                  width, height, params.dst_width, params.dst_height,
                  fx, params.dst_fx, fy, params.dst_fy);

        return params;
    }

    UndistortGrid compute_undistort_grid(
        const UndistortParams& params, const int resize_factor, const int max_width) {
        assert(params.dst_width > 0 && params.dst_height > 0);
        const float resize_scale = 1.0f / static_cast<float>(std::max(1, resize_factor));
        const int largest_crop_dimension = std::max(params.dst_width, params.dst_height);
        const float width_scale = max_width > 0
                                      ? static_cast<float>(max_width) /
                                            static_cast<float>(largest_crop_dimension)
                                      : 1.0f;
        const float scale = std::min({1.0f, resize_scale, width_scale});
        const int width = std::max(
            1, static_cast<int>(std::lround(static_cast<double>(params.dst_width) * scale)));
        const int height = std::max(
            1, static_cast<int>(std::lround(static_cast<double>(params.dst_height) * scale)));
        return {
            .width = width,
            .height = height,
            .scale_x = static_cast<float>(width) / static_cast<float>(params.dst_width),
            .scale_y = static_cast<float>(height) / static_cast<float>(params.dst_height)};
    }

    UndistortParams prepare_undistort_params(
        const UndistortParams& params,
        const int actual_src_width,
        const int actual_src_height,
        const int resize_factor,
        const int max_width) {
        const auto grid = compute_undistort_grid(params, resize_factor, max_width);
        assert(actual_src_width > 0 && actual_src_height > 0);
        assert(grid.width > 0 && grid.height > 0);
        UndistortParams scaled = params;
        const float src_scale_x = static_cast<float>(actual_src_width) /
                                  static_cast<float>(params.src_width);
        const float src_scale_y = static_cast<float>(actual_src_height) /
                                  static_cast<float>(params.src_height);
        scaled.src_fx = params.src_fx * src_scale_x;
        scaled.src_fy = params.src_fy * src_scale_y;
        scaled.src_cx = params.src_cx * src_scale_x;
        scaled.src_cy = params.src_cy * src_scale_y;
        scaled.src_width = actual_src_width;
        scaled.src_height = actual_src_height;
        scaled.dst_fx = params.dst_fx * grid.scale_x;
        scaled.dst_fy = params.dst_fy * grid.scale_y;
        scaled.dst_cx = params.dst_cx * grid.scale_x;
        scaled.dst_cy = params.dst_cy * grid.scale_y;
        scaled.dst_width = grid.width;
        scaled.dst_height = grid.height;
        return scaled;
    }

    UndistortParams scale_undistort_params(
        const UndistortParams& params, const int actual_src_width, const int actual_src_height,
        const int max_width) {
        UndistortParams scaled = params;

        if (actual_src_width != params.src_width || actual_src_height != params.src_height) {
            assert(actual_src_width > 0 && actual_src_height > 0);

            const float sx = static_cast<float>(actual_src_width) / static_cast<float>(params.src_width);
            const float sy = static_cast<float>(actual_src_height) / static_cast<float>(params.src_height);

            scaled.src_fx = params.src_fx * sx;
            scaled.src_fy = params.src_fy * sy;
            scaled.src_cx = params.src_cx * sx;
            scaled.src_cy = params.src_cy * sy;
            scaled.src_width = actual_src_width;
            scaled.src_height = actual_src_height;

            scaled.dst_fx = params.dst_fx * sx;
            scaled.dst_fy = params.dst_fy * sy;
            scaled.dst_width = std::max(1, static_cast<int>(std::lroundf(params.dst_width * sx)));
            scaled.dst_height = std::max(1, static_cast<int>(std::lroundf(params.dst_height * sy)));
            scaled.dst_cx = params.dst_cx * sx;
            scaled.dst_cy = params.dst_cy * sy;
        }

        if (max_width > 0 && (scaled.dst_width > max_width || scaled.dst_height > max_width)) {
            const int old_width = scaled.dst_width;
            const int old_height = scaled.dst_height;
            int new_width;
            int new_height;
            if (old_width > old_height) {
                new_width = max_width;
                new_height = std::max(
                    1, static_cast<int>(static_cast<std::int64_t>(old_height) * max_width / old_width));
            } else {
                new_height = max_width;
                new_width = std::max(
                    1, static_cast<int>(static_cast<std::int64_t>(old_width) * max_width / old_height));
            }

            const float dst_sx = static_cast<float>(new_width) / static_cast<float>(old_width);
            const float dst_sy = static_cast<float>(new_height) / static_cast<float>(old_height);
            scaled.dst_fx *= dst_sx;
            scaled.dst_fy *= dst_sy;
            scaled.dst_cx *= dst_sx;
            scaled.dst_cy *= dst_sy;
            scaled.dst_width = new_width;
            scaled.dst_height = new_height;
        }

        return scaled;
    }

} // namespace lfs::core
