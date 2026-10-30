/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
constexpr int BLOCK_DIM = 16;
constexpr float PIXEL_CENTER_OFFSET = 0.5f;
constexpr float NEWTON_EPSILON = 1e-6f;
// CUDA fast tan rounds the nearest float below pi/2 to infinity.
constexpr float MAX_FISHEYE_THETA = 1.570795f;
constexpr int MAX_NEWTON_ITERATIONS = 20;
constexpr float INVERSE_RESIDUAL_PIXELS = 5.0e-4f;
// Float32 Newton stalls near 6e-4 px at 8k image scales; accepting up to 1e-2 px keeps
// those pixels valid while the geometric error stays far below sampling resolution.
constexpr float INVERSE_ACCEPT_PIXELS = 1.0e-2f;
constexpr float INVERSE_JACOBIAN_STEP = 1.0e-4f;
constexpr float INVERSE_MAX_STEP = 2.0f;
constexpr int AREA_QUADRATURE = 8;
constexpr int LANCZOS_RADIUS = 3;
constexpr int OUTPUT_TILE_ROWS = 256;
constexpr float MIN_SIGNED_WEIGHT_RATIO = 1.0e-4f;
constexpr float EVALUATION_MIN_COVERAGE = 0.999f;

using detail::apply_distortion;
using detail::thin_prism_fisheye_from_theta_point;

template <typename T>
__device__ float normalized_sample(T value) {
    if constexpr (std::is_same_v<T, std::uint8_t>)
        return static_cast<float>(value) / 255.0f;
    return static_cast<float>(value);
}

template <typename T>
__device__ float bilinear_sample_renormalized(
    const T* __restrict__ src,
    const int width, const int height, const int stride,
    const float sx, const float sy,
    const bool positive_only = false) {
    const int x0 = static_cast<int>(floorf(sx));
    const int y0 = static_cast<int>(floorf(sy));
    const float fx = sx - static_cast<float>(x0);
    const float fy = sy - static_cast<float>(y0);
    float value = 0.0f;
    float weight_sum = 0.0f;
    for (int dy = 0; dy < 2; ++dy) {
        const int y = y0 + dy;
        const float wy = dy == 0 ? 1.0f - fy : fy;
        for (int dx = 0; dx < 2; ++dx) {
            const int x = x0 + dx;
            if (x < 0 || x >= width || y < 0 || y >= height)
                continue;
            const float sample = normalized_sample(src[y * stride + x]);
            if (!isfinite(sample) || (positive_only && sample <= 0.0f))
                continue;
            const float wx = dx == 0 ? 1.0f - fx : fx;
            const float weight = wx * wy;
            value += sample * weight;
            weight_sum += weight;
        }
    }
    if (weight_sum > 1.0e-8f)
        return value / weight_sum;
    if (positive_only)
        return 0.0f;
    const int nearest_x = min(max(static_cast<int>(floorf(sx + 0.5f)), 0), width - 1);
    const int nearest_y = min(max(static_cast<int>(floorf(sy + 0.5f)), 0), height - 1);
    return normalized_sample(src[nearest_y * stride + nearest_x]);
}

__device__ float lanczos3_weight(const float value) {
    const float absolute_value = fabsf(value);
    if (absolute_value >= static_cast<float>(LANCZOS_RADIUS))
        return 0.0f;
    if (absolute_value < 1.0e-6f)
        return 1.0f;
    constexpr float PI = 3.14159265358979323846f;
    const float pi_value = PI * value;
    // Preserve sinc zeros near integer taps even with CUDA fast math.
    // Border renormalization amplifies sine argument-reduction error.
    return sinpif(value) / pi_value *
           (sinpif(value / static_cast<float>(LANCZOS_RADIUS)) /
            (pi_value / static_cast<float>(LANCZOS_RADIUS)));
}

template <typename T>
__device__ bool lanczos3_sample(
    const T* __restrict__ src,
    const int width, const int height, const int channels,
    const float sx, const float sy,
    float* values, float& absolute_inside, float& absolute_full) {
    const int base_x = static_cast<int>(floorf(sx));
    const int base_y = static_cast<int>(floorf(sy));
    float weights_x[6];
    float weights_y[6];
    for (int i = 0; i < 6; ++i) {
        weights_x[i] = lanczos3_weight(sx - static_cast<float>(base_x + i - 2));
        weights_y[i] = lanczos3_weight(sy - static_cast<float>(base_y + i - 2));
    }

    float weighted[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float signed_inside = 0.0f;
    absolute_inside = 0.0f;
    absolute_full = 0.0f;
    const int plane = width * height;
    for (int j = 0; j < 6; ++j) {
        const int y = base_y + j - 2;
        for (int i = 0; i < 6; ++i) {
            const int x = base_x + i - 2;
            const float weight = weights_x[i] * weights_y[j];
            absolute_full += fabsf(weight);
            if (x < 0 || x >= width || y < 0 || y >= height)
                continue;
            signed_inside += weight;
            absolute_inside += fabsf(weight);
            const int index = y * width + x;
            for (int channel = 0; channel < channels; ++channel)
                weighted[channel] += normalized_sample(src[channel * plane + index]) * weight;
        }
    }

    if (fabsf(signed_inside) > MIN_SIGNED_WEIGHT_RATIO * absolute_inside) {
        for (int channel = 0; channel < channels; ++channel)
            values[channel] = weighted[channel] / signed_inside;
        return true;
    }

    for (int channel = 0; channel < channels; ++channel) {
        values[channel] = bilinear_sample_renormalized(
            src + channel * plane, width, height, width, sx, sy);
    }
    return false;
}

template <bool Bounded, int Channels = 0, typename T>
__device__ __forceinline__ void
undistort_image_kernel_impl(
    const T* __restrict__ src,
    float* __restrict__ dst,
    const int runtime_channels,
    const int output_y_offset,
    const int quadrature,
    const UndistortParams& params, const int output_y_end) {
    const int channels = Channels ? Channels : runtime_channels;
    const int ox = blockIdx.x * BLOCK_DIM + threadIdx.x;
    const int oy = output_y_offset + blockIdx.y * BLOCK_DIM + threadIdx.y;
    if constexpr (Bounded) {
        if (oy >= output_y_end)
            return;
    }
    if (ox >= params.dst_width || oy >= params.dst_height)
        return;

    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int qy = 0; qy < quadrature; ++qy) {
        for (int qx = 0; qx < quadrature; ++qx) {
            const float pixel_x = static_cast<float>(ox) +
                                  (static_cast<float>(qx) + 0.5f) / quadrature;
            const float pixel_y = static_cast<float>(oy) +
                                  (static_cast<float>(qy) + 0.5f) / quadrature;
            const float nx = (pixel_x - params.dst_cx) / params.dst_fx;
            const float ny = (pixel_y - params.dst_cy) / params.dst_fy;
            float dnx, dny;
            apply_distortion(nx, ny, static_cast<CameraModelType>(params.model_type), params.distortion,
                             params.num_distortion, dnx, dny);
            const float sx = (Bounded ? fmaf(dnx, params.src_fx, params.src_cx) : dnx * params.src_fx + params.src_cx) - PIXEL_CENTER_OFFSET;
            const float sy = (Bounded ? fmaf(dny, params.src_fy, params.src_cy) : dny * params.src_fy + params.src_cy) - PIXEL_CENTER_OFFSET;
            float sample[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float absolute_inside = 0.0f;
            float absolute_full = 0.0f;
            if constexpr (Bounded) {
                if (!isfinite(sx) || !isfinite(sy) || fabsf(sx) > 1073741824.f || fabsf(sy) > 1073741824.f)
                    continue;
            }
            (void)lanczos3_sample(
                src, params.src_width, params.src_height, channels,
                sx, sy, sample, absolute_inside, absolute_full);
            for (int channel = 0; channel < channels; ++channel)
                result[channel] += sample[channel];
        }
    }

    const int output_index = oy * params.dst_width + ox;
    const int output_plane = params.dst_width * params.dst_height;
    const float inverse_samples = 1.0f / static_cast<float>(quadrature * quadrature);
    for (int channel = 0; channel < channels; ++channel)
        dst[channel * output_plane + output_index] = result[channel] * inverse_samples;
}

struct RadialBranch {
    float bound;
    float maximum;
};

__device__ void radial_mapping(const float r, const UndistortParams& params,
                               float& value, float& derivative, float& denominator) {
    const auto* k = params.distortion;
    const bool fisheye = static_cast<CameraModelType>(params.model_type) != CameraModelType::PINHOLE;
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

__device__ RadialBranch radial_branch(const UndistortParams& params) {
    constexpr int SAMPLES = 1024;
    constexpr int THREADS = BLOCK_DIM * BLOCK_DIM;
    __shared__ int first[THREADS];
    __shared__ RadialBranch branch;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const bool fisheye = static_cast<CameraModelType>(params.model_type) != CameraModelType::PINHOLE;
    int crossing = SAMPLES + 1;
    // Scanning atan(r) covers pinhole radii without a lens-specific cutoff.
    for (int i = tid + 1; i <= SAMPLES; i += THREADS) {
        const float angle = MAX_FISHEYE_THETA * (static_cast<float>(i) / SAMPLES);
        const float r = fisheye ? angle : tanf(angle);
        float value, derivative, denominator;
        radial_mapping(r, params, value, derivative, denominator);
        if (derivative <= 0.0f || denominator <= 0.0f)
            crossing = min(crossing, i);
    }
    first[tid] = crossing;
    __syncthreads();
    for (int stride = THREADS / 2; stride > 0; stride /= 2) {
        if (tid < stride)
            first[tid] = min(first[tid], first[tid + stride]);
        __syncthreads();
    }
    if (tid == 0) {
        branch.bound = fisheye ? MAX_FISHEYE_THETA : INFINITY;
        branch.maximum = INFINITY;
        if (first[0] <= SAMPLES) {
            float lo = MAX_FISHEYE_THETA * (static_cast<float>(first[0] - 1) / SAMPLES);
            float hi = MAX_FISHEYE_THETA * (static_cast<float>(first[0]) / SAMPLES);
            for (int i = 0; i < 24; ++i) {
                const float mid = 0.5f * (lo + hi);
                float value, derivative, denominator;
                radial_mapping(fisheye ? mid : tanf(mid), params, value, derivative, denominator);
                if (derivative > 0.0f && denominator > 0.0f)
                    lo = mid;
                else
                    hi = mid;
            }
            branch.bound = fisheye ? lo : tanf(lo);
        }
        if (isfinite(branch.bound)) {
            float derivative, denominator;
            radial_mapping(branch.bound, params, branch.maximum, derivative, denominator);
        }
    }
    __syncthreads();
    return branch;
}

__device__ bool inverse_distortion(
    const float xd, const float yd, const UndistortParams& params, const RadialBranch branch,
    float& ux, float& uy) {
    const bool fisheye = static_cast<CameraModelType>(params.model_type) != CameraModelType::PINHOLE;
    const bool prism = static_cast<CameraModelType>(params.model_type) == CameraModelType::THIN_PRISM_FISHEYE;
    const int offset = params.num_distortion >= 6 ? 6 : 3;
    bool coupled = !fisheye &&
                   ((params.num_distortion > offset && params.distortion[offset] != 0.0f) ||
                    (params.num_distortion > offset + 1 && params.distortion[offset + 1] != 0.0f));
    if (prism) {
        for (int i = 4; i < params.num_distortion; ++i)
            coupled |= params.distortion[i] != 0.0f;
    }
    const float rd = sqrtf(xd * xd + yd * yd);
    if (!isfinite(rd) || (!coupled && rd >= branch.maximum))
        return false;
    if (rd < 1e-8f) {
        ux = xd;
        uy = yd;
        return true;
    }
    const float px = params.src_fx * xd;
    const float py = params.src_fy * yd;
    const float pixel_scale = sqrtf(px * px + py * py) / rd;
    float lo = 0.0f;
    float hi = branch.bound;
    if (!isfinite(hi)) {
        hi = fmaxf(1.0f, 2.0f * rd);
        for (int i = 0; i < 32; ++i) {
            float value, derivative, denominator;
            radial_mapping(hi, params, value, derivative, denominator);
            if (value >= rd)
                break;
            hi *= 2.0f;
        }
    }
    float r = isfinite(branch.maximum) ? rd * (branch.bound / branch.maximum) : rd;
    r = fminf(r, hi);
    if (rd >= branch.maximum) {
        r = nextafterf(branch.bound, 0.0f);
    } else {
        float previous_error_px = INFINITY;
        for (int i = 0; i < 40; ++i) {
            float value, derivative, denominator;
            radial_mapping(r, params, value, derivative, denominator);
            const float residual = value - rd;
            const float error_px = fabsf(residual) * pixel_scale;
            if (error_px <= INVERSE_RESIDUAL_PIXELS ||
                (error_px <= INVERSE_ACCEPT_PIXELS && error_px >= previous_error_px))
                break;
            previous_error_px = error_px;
            if (residual < 0.0f)
                lo = r;
            else
                hi = r;
            const float next = r - residual / derivative;
            r = next > lo && next < hi ? next : 0.5f * (lo + hi);
        }
    }
    if (!coupled) {
        float value, derivative, denominator;
        radial_mapping(r, params, value, derivative, denominator);
        if (!(r < branch.bound) || !isfinite(value) ||
            fabsf(value - rd) * pixel_scale > INVERSE_ACCEPT_PIXELS)
            return false;
        const float scale = (fisheye ? tanf(r) : r) / rd;
        ux = xd * scale;
        uy = yd * scale;
        return isfinite(ux) && isfinite(uy);
    }
    ux = xd * r / rd;
    uy = yd * r / rd;

    // Coupled roots must converge in pixels inside the first radial branch with det J > 0.
    // Prism refinement stays in theta coordinates so a horizon seed does not overflow tan(theta).
    const auto forward = [&](const float x, const float y, float& dx, float& dy) {
        if (prism)
            thin_prism_fisheye_from_theta_point(x, y, params.distortion, params.num_distortion, dx, dy);
        else
            apply_distortion(x, y, static_cast<CameraModelType>(params.model_type), params.distortion, params.num_distortion, dx, dy);
    };
    float previous_error_px = INFINITY;
    for (int iter = 0; iter <= MAX_NEWTON_ITERATIONS; ++iter) {
        float eval_x, eval_y;
        forward(ux, uy, eval_x, eval_y);
        const float rx = eval_x - xd;
        const float ry = eval_y - yd;
        const float error_px = hypotf(rx * params.src_fx, ry * params.src_fy);
        if (!isfinite(error_px))
            return false;
        float xp_x, xp_y, xm_x, xm_y;
        float yp_x, yp_y, ym_x, ym_y;
        forward(ux + INVERSE_JACOBIAN_STEP, uy, xp_x, xp_y);
        forward(ux - INVERSE_JACOBIAN_STEP, uy, xm_x, xm_y);
        forward(ux, uy + INVERSE_JACOBIAN_STEP, yp_x, yp_y);
        forward(ux, uy - INVERSE_JACOBIAN_STEP, ym_x, ym_y);
        const float inverse_step = 0.5f / INVERSE_JACOBIAN_STEP;
        const float j00 = (xp_x - xm_x) * inverse_step;
        const float j10 = (xp_y - xm_y) * inverse_step;
        const float j01 = (yp_x - ym_x) * inverse_step;
        const float j11 = (yp_y - ym_y) * inverse_step;
        const float det = j00 * j11 - j01 * j10;
        if (!isfinite(det) || fabsf(det) < NEWTON_EPSILON)
            return false;
        const bool stalled = error_px <= INVERSE_ACCEPT_PIXELS && error_px >= previous_error_px;
        if (error_px <= INVERSE_RESIDUAL_PIXELS || stalled || iter == MAX_NEWTON_ITERATIONS) {
            const float radius = hypotf(ux, uy);
            if (error_px > INVERSE_ACCEPT_PIXELS || det <= 0.0f || !(radius < branch.bound))
                return false;
            if (prism && radius > 1e-8f) {
                const float scale = tanf(radius) / radius;
                ux *= scale;
                uy *= scale;
            }
            return isfinite(ux) && isfinite(uy);
        }
        previous_error_px = error_px;
        float step_x = (j11 * rx - j01 * ry) / det;
        float step_y = (-j10 * rx + j00 * ry) / det;
        if (!isfinite(step_x) || !isfinite(step_y))
            return false;
        const float step_length = fmaxf(fabsf(step_x), fabsf(step_y));
        if (step_length > INVERSE_MAX_STEP) {
            step_x *= INVERSE_MAX_STEP / step_length;
            step_y *= INVERSE_MAX_STEP / step_length;
        }
        ux -= step_x;
        uy -= step_y;
    }
    return false;
}

template <bool Bounded>
__device__ __forceinline__ void
distort_image_to_source_kernel_impl(
    const float* __restrict__ src,
    float* __restrict__ dst,
    uint8_t* __restrict__ validity,
    const int channels,
    const int output_y_offset,
    const UndistortParams& params, const int output_y_end) {
    const auto branch = radial_branch(params);
    const int ox = blockIdx.x * BLOCK_DIM + threadIdx.x;
    const int oy = output_y_offset + blockIdx.y * BLOCK_DIM + threadIdx.y;
    if constexpr (Bounded) {
        if (oy >= output_y_end)
            return;
    }
    if (ox >= params.src_width || oy >= params.src_height)
        return;

    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float absolute_inside_sum = 0.0f;
    float absolute_full_sum = 0.0f;
    bool converged = true;
    for (int qy = 0; qy < AREA_QUADRATURE && converged; ++qy) {
        for (int qx = 0; qx < AREA_QUADRATURE; ++qx) {
            const float pixel_x = static_cast<float>(ox) +
                                  (static_cast<float>(qx) + 0.5f) / AREA_QUADRATURE;
            const float pixel_y = static_cast<float>(oy) +
                                  (static_cast<float>(qy) + 0.5f) / AREA_QUADRATURE;
            const float xd = (pixel_x - params.src_cx) / params.src_fx;
            const float yd = (pixel_y - params.src_cy) / params.src_fy;
            float ux, uy;
            if (!inverse_distortion(xd, yd, params, branch, ux, uy)) {
                converged = false;
                break;
            }
            const float sx = ux * params.dst_fx + params.dst_cx - PIXEL_CENTER_OFFSET;
            const float sy = uy * params.dst_fy + params.dst_cy - PIXEL_CENTER_OFFSET;
            if (!isfinite(sx) || !isfinite(sy) || (Bounded && (fabsf(sx) > 1073741824.f || fabsf(sy) > 1073741824.f))) {
                converged = false;
                break;
            }
            float sample[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float absolute_inside = 0.0f;
            float absolute_full = 0.0f;
            if (!lanczos3_sample(
                    src, params.dst_width, params.dst_height, channels,
                    sx, sy, sample, absolute_inside, absolute_full)) {
                converged = false;
                break;
            }
            absolute_inside_sum += absolute_inside;
            absolute_full_sum += absolute_full;
            for (int channel = 0; channel < channels; ++channel)
                result[channel] += sample[channel];
        }
    }

    const int output_index = oy * params.src_width + ox;
    const int output_plane = params.src_width * params.src_height;
    const float coverage = absolute_full_sum > 0.0f
                               ? absolute_inside_sum / absolute_full_sum
                               : 0.0f;
    if (!converged || coverage < EVALUATION_MIN_COVERAGE) {
        validity[output_index] = 0;
        for (int channel = 0; channel < channels; ++channel)
            dst[channel * output_plane + output_index] = 0.0f;
        return;
    }

    validity[output_index] = 1;
    constexpr float inverse_samples = 1.0f / (AREA_QUADRATURE * AREA_QUADRATURE);
    for (int channel = 0; channel < channels; ++channel)
        dst[channel * output_plane + output_index] = result[channel] * inverse_samples;
}

enum class AreaFilterMode : int {
    NONNEGATIVE,
    DEPTH,
    NORMAL
};

template <bool Bounded>
__device__ __forceinline__ void
undistort_area_kernel_impl(
    const float* __restrict__ src,
    float* __restrict__ dst,
    const int channels,
    const AreaFilterMode mode,
    const int output_y_offset,
    const int quadrature,
    const UndistortParams params, const int output_y_end) {
    const int ox = blockIdx.x * BLOCK_DIM + threadIdx.x;
    const int oy = output_y_offset + blockIdx.y * BLOCK_DIM + threadIdx.y;
    if constexpr (Bounded) {
        if (oy >= output_y_end)
            return;
    }
    if (ox >= params.dst_width || oy >= params.dst_height)
        return;

    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float valid_samples = 0.0f;
    const int input_plane = params.src_width * params.src_height;
    for (int qy = 0; qy < quadrature; ++qy) {
        for (int qx = 0; qx < quadrature; ++qx) {
            const float pixel_x = static_cast<float>(ox) +
                                  (static_cast<float>(qx) + 0.5f) / quadrature;
            const float pixel_y = static_cast<float>(oy) +
                                  (static_cast<float>(qy) + 0.5f) / quadrature;
            const float nx = (pixel_x - params.dst_cx) / params.dst_fx;
            const float ny = (pixel_y - params.dst_cy) / params.dst_fy;
            float dnx, dny;
            apply_distortion(nx, ny, static_cast<CameraModelType>(params.model_type), params.distortion,
                             params.num_distortion, dnx, dny);
            const float sx = (Bounded ? fmaf(dnx, params.src_fx, params.src_cx) : dnx * params.src_fx + params.src_cx) - PIXEL_CENTER_OFFSET;
            const float sy = (Bounded ? fmaf(dny, params.src_fy, params.src_cy) : dny * params.src_fy + params.src_cy) - PIXEL_CENTER_OFFSET;
            if constexpr (Bounded) {
                if (!isfinite(sx) || !isfinite(sy) || fabsf(sx) > 1073741824.f || fabsf(sy) > 1073741824.f)
                    continue;
            }
            if (mode == AreaFilterMode::DEPTH) {
                const float sample = bilinear_sample_renormalized(
                    src, params.src_width, params.src_height, params.src_width,
                    sx, sy, true);
                if (sample > 0.0f && isfinite(sample)) {
                    result[0] += sample;
                    valid_samples += 1.0f;
                }
            } else {
                for (int channel = 0; channel < channels; ++channel) {
                    result[channel] += bilinear_sample_renormalized(
                        src + channel * input_plane,
                        params.src_width, params.src_height, params.src_width,
                        sx, sy);
                }
                valid_samples += 1.0f;
            }
        }
    }

    const int output_index = oy * params.dst_width + ox;
    const int output_plane = params.dst_width * params.dst_height;
    if (valid_samples <= 0.0f) {
        for (int channel = 0; channel < channels; ++channel)
            dst[channel * output_plane + output_index] = 0.0f;
        return;
    }
    for (int channel = 0; channel < channels; ++channel)
        result[channel] /= valid_samples;
    if (mode == AreaFilterMode::NORMAL) {
        const float norm = sqrtf(result[0] * result[0] + result[1] * result[1] +
                                 result[2] * result[2]);
        if (norm > 1.0e-8f && isfinite(norm)) {
            result[0] /= norm;
            result[1] /= norm;
            result[2] /= norm;
        } else {
            result[0] = result[1] = result[2] = 0.0f;
        }
    }
    for (int channel = 0; channel < channels; ++channel)
        dst[channel * output_plane + output_index] = result[channel];
}

template <bool Bounded>
__device__ __forceinline__ void
distort_area_to_source_kernel_impl(
    const float* __restrict__ src,
    float* __restrict__ dst,
    const int channels,
    const AreaFilterMode mode,
    const int output_y_offset,
    const UndistortParams& params, const int output_y_end) {
    const auto branch = radial_branch(params);
    const int ox = blockIdx.x * BLOCK_DIM + threadIdx.x;
    const int oy = output_y_offset + blockIdx.y * BLOCK_DIM + threadIdx.y;
    if constexpr (Bounded) {
        if (oy >= output_y_end)
            return;
    }
    if (ox >= params.src_width || oy >= params.src_height)
        return;

    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float valid_samples = 0.0f;
    const int input_plane = params.dst_width * params.dst_height;
    bool converged = true;
    for (int qy = 0; qy < AREA_QUADRATURE && converged; ++qy) {
        for (int qx = 0; qx < AREA_QUADRATURE; ++qx) {
            const float pixel_x = static_cast<float>(ox) +
                                  (static_cast<float>(qx) + 0.5f) / AREA_QUADRATURE;
            const float pixel_y = static_cast<float>(oy) +
                                  (static_cast<float>(qy) + 0.5f) / AREA_QUADRATURE;
            const float xd = (pixel_x - params.src_cx) / params.src_fx;
            const float yd = (pixel_y - params.src_cy) / params.src_fy;
            float ux, uy;
            if (!inverse_distortion(xd, yd, params, branch, ux, uy)) {
                converged = false;
                break;
            }
            const float sx = ux * params.dst_fx + params.dst_cx - PIXEL_CENTER_OFFSET;
            const float sy = uy * params.dst_fy + params.dst_cy - PIXEL_CENTER_OFFSET;
            if constexpr (Bounded) {
                if (!isfinite(sx) || !isfinite(sy) || fabsf(sx) > 1073741824.f || fabsf(sy) > 1073741824.f) {
                    converged = false;
                    break;
                }
            }
            if (mode == AreaFilterMode::DEPTH) {
                const float sample = bilinear_sample_renormalized(
                    src, params.dst_width, params.dst_height, params.dst_width,
                    sx, sy, true);
                if (sample > 0.0f && isfinite(sample)) {
                    result[0] += sample;
                    valid_samples += 1.0f;
                }
            } else {
                for (int channel = 0; channel < channels; ++channel) {
                    result[channel] += bilinear_sample_renormalized(
                        src + channel * input_plane,
                        params.dst_width, params.dst_height, params.dst_width,
                        sx, sy);
                }
                valid_samples += 1.0f;
            }
        }
    }

    const int output_index = oy * params.src_width + ox;
    const int output_plane = params.src_width * params.src_height;
    if (!converged || valid_samples <= 0.0f) {
        for (int channel = 0; channel < channels; ++channel)
            dst[channel * output_plane + output_index] = 0.0f;
        return;
    }
    for (int channel = 0; channel < channels; ++channel)
        result[channel] /= valid_samples;
    if (mode == AreaFilterMode::NORMAL) {
        const float norm = sqrtf(result[0] * result[0] + result[1] * result[1] +
                                 result[2] * result[2]);
        if (norm > 1.0e-8f && isfinite(norm)) {
            result[0] /= norm;
            result[1] /= norm;
            result[2] /= norm;
        } else {
            result[0] = result[1] = result[2] = 0.0f;
        }
    }
    for (int channel = 0; channel < channels; ++channel)
        dst[channel * output_plane + output_index] = result[channel];
}

// At least one sample per source pixel along each axis, so strong minification still
// integrates the whole footprint instead of aliasing on a fixed sample lattice.
int area_quadrature(const UndistortParams& params) {
    const float minification = std::max(params.src_fx / params.dst_fx,
                                        params.src_fy / params.dst_fy);
    return std::max(AREA_QUADRATURE, static_cast<int>(std::ceil(minification)));
}

template <bool Bounded>
__device__ __forceinline__ void
inverse_distortion_sample_map_kernel_impl(
    float2* __restrict__ samples,
    const UndistortParams& params) {
    const auto branch = radial_branch(params);
    const int ox = blockIdx.x * BLOCK_DIM + threadIdx.x;
    const int oy = blockIdx.y * BLOCK_DIM + threadIdx.y;
    if (ox >= params.src_width || oy >= params.src_height)
        return;
    const float xd = (static_cast<float>(ox) + PIXEL_CENTER_OFFSET - params.src_cx) /
                     params.src_fx;
    const float yd = (static_cast<float>(oy) + PIXEL_CENTER_OFFSET - params.src_cy) /
                     params.src_fy;
    float ux, uy;
    if (!inverse_distortion(xd, yd, params, branch, ux, uy)) {
        ux = nanf("");
        uy = nanf("");
    }
    samples[oy * params.src_width + ox] = make_float2(ux, uy);
}
