// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
static const uint LFS_LOCAL_SIZE_X = 256;
static const uint LFS_PUSH_CONSTANT_SIZE = 56;
struct UndistortParams {
    float src_fx, src_fy, src_cx, src_cy, dst_fx, dst_fy, dst_cx, dst_cy;
    int src_width, src_height, dst_width, dst_height, model_type;
    float distortion[12];
    int num_distortion;
    // undistort_radial_branch() on the host: rays from branch_maximum on fold back.
    float branch_bound, branch_maximum;
};
struct WarpPush {
#ifdef LFS_WARP_GPU_PROGRAM
    Ptr<uint8_t, Access.Read> input;
    Ptr<float, Access.ReadWrite> output;
    Ptr<int, Access.ReadWrite> validity;
    Ptr<UndistortParams, Access.Read> camera;
#else
    // The existing Vulkan finalizer consumes scalar device addresses. Metal's
    // module reflection needs typed pointers; both layouts remain 56 bytes.
    uint64_t input, output, validity, camera;
#endif
    int width, height, channels, mode, inverse, quadrature;
};
[[vk::push_constant]] ConstantBuffer<WarpPush> pc;
float hypot(float x, float y) { return sqrt(x * x + y * y); }
static const float PIXEL_CENTER_OFFSET = 0.5f;
static const float NEWTON_EPSILON = 1e-6f;
static const int MAX_NEWTON_ITERATIONS = 20;
static const float INVERSE_RESIDUAL_PIXELS = 5.0e-4f;
// Float32 Newton stalls near 6e-4 px at 8k image scales; accepting up to 1e-2 px keeps
// those pixels valid while the geometric error stays far below sampling resolution.
static const float INVERSE_ACCEPT_PIXELS = 1.0e-2f;
static const float INVERSE_JACOBIAN_STEP = 1.0e-4f;
static const float INVERSE_MAX_STEP = 2.0f;
static const float COLMAP_MIN_SCALE = 0.2f;
static const float COLMAP_MAX_SCALE = 2.0f;
static const int AREA_QUADRATURE = 8;
static const int LANCZOS_RADIUS = 3;
static const int OUTPUT_TILE_ROWS = 256;
static const float MIN_SIGNED_WEIGHT_RATIO = 1.0e-4f;
static const float EVALUATION_MIN_COVERAGE = 0.999f;

#include "distortion_model.slangh"

// Mode bits 0..7 select the operation; bit 8 identifies RGB8 storage.
// Keep WarpPush at 56 bytes on Vulkan and Metal. Vulkan byte storage uses
// aligned word reads so it does not require optional shaderInt8 arithmetic.
#ifdef LFS_WARP_GPU_PROGRAM
typedef Ptr<uint8_t, Access.Read> ImageSource;
#else
#include "byte_read.slangh"
typedef uint64_t ImageSource;
#endif
#ifndef LFS_WARP_GPU_PROGRAM
[SpecializationConstant] const uint warpMode = 0;
[SpecializationConstant] const uint warpInverse = 0;
[SpecializationConstant] const uint warpChannels = 3;
[SpecializationConstant] const uint warpQuadrature = 8;
[SpecializationConstant] const uint warpModel = 0;
#define WARP_MODE int(warpMode)
#define WARP_INVERSE int(warpInverse)
#define WARP_CHANNELS int(warpChannels)
#define WARP_QUADRATURE int(warpQuadrature)
#else
#define WARP_MODE pc.mode
#define WARP_INVERSE pc.inverse
#define WARP_CHANNELS pc.channels
#define WARP_QUADRATURE pc.quadrature
#endif
float read_image_sample(ImageSource source, int index) {
    if ((WARP_MODE & 256) != 0) {
#ifdef LFS_WARP_GPU_PROGRAM
        return float(uint(source[index])) / 255.0f;
#else
        return float(loadByte(source, uint64_t(index))) / 255.0f;
#endif
    }
    return ((Ptr<float, Access.Read>)source)[index];
}
ImageSource offset_image_source(ImageSource source, int elements) {
    return source + elements * ((WARP_MODE & 256) != 0 ? 1u : 4u);
}

float bilinear_sample_renormalized(
    ImageSource src,
    const int width, const int height, const int stride,
    const float sx, const float sy,
    const bool positive_only = false) {
    const int x0 = int(floor(sx));
    const int y0 = int(floor(sy));
    const float fx = sx - float(x0);
    const float fy = sy - float(y0);
    float value = 0.0f;
    float weight_sum = 0.0f;
    for (int dy = 0; dy < 2; ++dy) {
        const int y = y0 + dy;
        const float wy = dy == 0 ? 1.0f - fy : fy;
        for (int dx = 0; dx < 2; ++dx) {
            const int x = x0 + dx;
            if (x < 0 || x >= width || y < 0 || y >= height)
                continue;
            const float sample = read_image_sample(src, y * stride + x);
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
    const int nearest_x = min(max(int(floor(sx + 0.5f)), 0), width - 1);
    const int nearest_y = min(max(int(floor(sy + 0.5f)), 0), height - 1);
    return read_image_sample(src, nearest_y * stride + nearest_x);
}

float lanczos3_weight(const float value) {
    const float absolute_value = abs(value);
    if (absolute_value >= float(LANCZOS_RADIUS))
        return 0.0f;
    if (absolute_value < 1.0e-6f)
        return 1.0f;
    static const float PI = 3.14159265358979323846f;
    const float pi_value = PI * value;
    return sin(pi_value) / pi_value *
           (sin(pi_value / float(LANCZOS_RADIUS)) /
            (pi_value / float(LANCZOS_RADIUS)));
}

bool lanczos3_sample(
    ImageSource src,
    const int width, const int height, const int channels,
    const float sx, const float sy,
    inout float values[4], out float absolute_inside, out float absolute_full) {
    const int base_x = int(floor(sx));
    const int base_y = int(floor(sy));
    float weights_x[6];
    float weights_y[6];
    for (int i = 0; i < 6; ++i) {
        weights_x[i] = lanczos3_weight(sx - float(base_x + i - 2));
        weights_y[i] = lanczos3_weight(sy - float(base_y + i - 2));
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
            absolute_full += abs(weight);
            if (x < 0 || x >= width || y < 0 || y >= height)
                continue;
            signed_inside += weight;
            absolute_inside += abs(weight);
            const int index = y * width + x;
            for (int channel = 0; channel < channels; ++channel)
                weighted[channel] += read_image_sample(src, channel * plane + index) * weight;
        }
    }

    if (abs(signed_inside) > MIN_SIGNED_WEIGHT_RATIO * absolute_inside) {
        for (int channel = 0; channel < channels; ++channel)
            values[channel] = weighted[channel] / signed_inside;
        return true;
    }

    for (int channel = 0; channel < channels; ++channel) {
        values[channel] = bilinear_sample_renormalized(
            offset_image_source(src, channel * plane), width, height, width, sx, sy);
    }
    return false;
}

void undistort_image_kernel(
    ImageSource src,
    Ptr<float, Access.ReadWrite> dst,
    const int channels,
    const int ox, const int oy,
    const int quadrature,
    const UndistortParams params) {
    if (ox >= params.dst_width || oy >= params.dst_height)
        return;

    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int qy = 0; qy < quadrature; ++qy) {
        for (int qx = 0; qx < quadrature; ++qx) {
            const float pixel_x = float(ox) +
                                  (float(qx) + 0.5f) / quadrature;
            const float pixel_y = float(oy) +
                                  (float(qy) + 0.5f) / quadrature;
            const float nx = (pixel_x - params.dst_cx) / params.dst_fx;
            const float ny = (pixel_y - params.dst_cy) / params.dst_fy;
            float dnx, dny;
            apply_distortion(nx, ny, params.model_type, params.distortion,
                             params.num_distortion, dnx, dny);
            const float sx = dnx * params.src_fx + params.src_cx - PIXEL_CENTER_OFFSET;
            const float sy = dny * params.src_fy + params.src_cy - PIXEL_CENTER_OFFSET;
            float sample[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float absolute_inside = 0.0f;
            float absolute_full = 0.0f;
            lanczos3_sample(
                src, params.src_width, params.src_height, channels,
                sx, sy, sample, absolute_inside, absolute_full);
            for (int channel = 0; channel < channels; ++channel)
                result[channel] += sample[channel];
        }
    }

    const int output_index = oy * params.dst_width + ox;
    const int output_plane = params.dst_width * params.dst_height;
    const float inverse_samples = 1.0f / float(quadrature * quadrature);
    for (int channel = 0; channel < channels; ++channel)
        dst[channel * output_plane + output_index] = result[channel] * inverse_samples;
}

void radial_mapping(const float r, const UndistortParams params,
                    out float value, out float derivative, out float denominator) {
    const bool fisheye = params.model_type != 0;
    const float s = r * r;
    const float k1 = params.num_distortion > 0 ? params.distortion[0] : 0.0f;
    const float k2 = params.num_distortion > 1 ? params.distortion[1] : 0.0f;
    const float k3 = params.num_distortion > 2 ? params.distortion[2] : 0.0f;
    const float k4 = fisheye && params.num_distortion > 3 ? params.distortion[3] : 0.0f;
    const float numerator = 1.0f + s * (k1 + s * (k2 + s * (k3 + s * k4)));
    const float dn = k1 + s * (2.0f * k2 + s * (3.0f * k3 + s * 4.0f * k4));
    const bool rational = !fisheye && params.num_distortion >= 6;
    denominator = rational ? 1.0f + s * (params.distortion[3] + s * (params.distortion[4] + s * params.distortion[5])) : 1.0f;
    const float dd = rational ? params.distortion[3] + s * (2.0f * params.distortion[4] + s * 3.0f * params.distortion[5]) : 0.0f;
    const float radial = numerator / denominator;
    value = r * radial;
    derivative = radial + 2.0f * s * (dn - radial * dd) / denominator;
}

void inverse_forward(const float x, const float y, const UndistortParams params, const bool prism,
                     out float dx, out float dy) {
    if (prism)
        thin_prism_fisheye_from_theta_point(x, y, params.distortion, params.num_distortion, dx, dy);
    else
        apply_distortion(x, y, params.model_type, params.distortion, params.num_distortion, dx, dy);
}

// The branch-aware inverse of undistort.cu. Rays past the first turning point of the radial map,
// or past the horizon, have no inverse; the host computes that branch once per warp.
bool inverse_distortion(
    const float xd, const float yd, const UndistortParams params,
    out float ux, out float uy) {
    ux = xd;
    uy = yd;
    const float infinity = asfloat(0x7f800000u);
    const bool fisheye = params.model_type != 0;
    const bool prism = params.model_type == 4;
    const int offset = params.num_distortion >= 6 ? 6 : 3;
    bool coupled = !fisheye &&
                   ((params.num_distortion > offset && params.distortion[offset] != 0.0f) ||
                    (params.num_distortion > offset + 1 && params.distortion[offset + 1] != 0.0f));
    if (prism) {
        for (int i = 4; i < params.num_distortion; ++i)
            coupled = coupled || params.distortion[i] != 0.0f;
    }
    const float bound = params.branch_bound;
    const float maximum = params.branch_maximum;
    const float rd = sqrt(xd * xd + yd * yd);
    if (!isfinite(rd) || (!coupled && rd >= maximum))
        return false;
    if (rd < 1e-8f)
        return true;
    const float px = params.src_fx * xd;
    const float py = params.src_fy * yd;
    const float pixel_scale = sqrt(px * px + py * py) / rd;
    float lo = 0.0f;
    float hi = bound;
    if (!isfinite(hi)) {
        hi = max(1.0f, 2.0f * rd);
        for (int i = 0; i < 32; ++i) {
            float value, derivative, denominator;
            radial_mapping(hi, params, value, derivative, denominator);
            if (value >= rd)
                break;
            hi *= 2.0f;
        }
    }
    float r = isfinite(maximum) ? rd * (bound / maximum) : rd;
    r = min(r, hi);
    if (rd >= maximum) {
        // nextafter(bound, 0): a finite fold bound is positive.
        r = asfloat(asuint(bound) - 1u);
    } else {
        float previous_error_px = infinity;
        for (int i = 0; i < 40; ++i) {
            float value, derivative, denominator;
            radial_mapping(r, params, value, derivative, denominator);
            const float residual = value - rd;
            const float error_px = abs(residual) * pixel_scale;
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
        if (!(r < bound) || !isfinite(value) ||
            abs(value - rd) * pixel_scale > INVERSE_ACCEPT_PIXELS)
            return false;
        const float scale = (fisheye ? tan(r) : r) / rd;
        ux = xd * scale;
        uy = yd * scale;
        return isfinite(ux) && isfinite(uy);
    }
    ux = xd * r / rd;
    uy = yd * r / rd;

    // Coupled roots must converge in pixels inside the first radial branch with det J > 0.
    // Prism refinement stays in theta coordinates so a horizon seed does not overflow tan(theta).
    float previous_error_px = infinity;
    for (int iter = 0; iter <= MAX_NEWTON_ITERATIONS; ++iter) {
        float eval_x, eval_y;
        inverse_forward(ux, uy, params, prism, eval_x, eval_y);
        const float rx = eval_x - xd;
        const float ry = eval_y - yd;
        const float error_px = hypot(rx * params.src_fx, ry * params.src_fy);
        if (!isfinite(error_px))
            return false;
        float xp_x, xp_y, xm_x, xm_y;
        float yp_x, yp_y, ym_x, ym_y;
        inverse_forward(ux + INVERSE_JACOBIAN_STEP, uy, params, prism, xp_x, xp_y);
        inverse_forward(ux - INVERSE_JACOBIAN_STEP, uy, params, prism, xm_x, xm_y);
        inverse_forward(ux, uy + INVERSE_JACOBIAN_STEP, params, prism, yp_x, yp_y);
        inverse_forward(ux, uy - INVERSE_JACOBIAN_STEP, params, prism, ym_x, ym_y);
        const float inverse_step = 0.5f / INVERSE_JACOBIAN_STEP;
        const float j00 = (xp_x - xm_x) * inverse_step;
        const float j10 = (xp_y - xm_y) * inverse_step;
        const float j01 = (yp_x - ym_x) * inverse_step;
        const float j11 = (yp_y - ym_y) * inverse_step;
        const float det = j00 * j11 - j01 * j10;
        if (!isfinite(det) || abs(det) < NEWTON_EPSILON)
            return false;
        const bool stalled = error_px <= INVERSE_ACCEPT_PIXELS && error_px >= previous_error_px;
        if (error_px <= INVERSE_RESIDUAL_PIXELS || stalled || iter == MAX_NEWTON_ITERATIONS) {
            const float radius = hypot(ux, uy);
            if (error_px > INVERSE_ACCEPT_PIXELS || det <= 0.0f || !(radius < bound))
                return false;
            if (prism && radius > 1e-8f) {
                const float scale = tan(radius) / radius;
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
        const float step_length = max(abs(step_x), abs(step_y));
        if (step_length > INVERSE_MAX_STEP) {
            step_x *= INVERSE_MAX_STEP / step_length;
            step_y *= INVERSE_MAX_STEP / step_length;
        }
        ux -= step_x;
        uy -= step_y;
    }
    return false;
}

void distort_image_to_source_kernel(
    ImageSource src,
    Ptr<float, Access.ReadWrite> dst,
    Ptr<int, Access.ReadWrite> validity,
    const int channels,
    const int ox, const int oy,
    const UndistortParams params) {
    if (ox >= params.src_width || oy >= params.src_height)
        return;

    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float absolute_inside_sum = 0.0f;
    float absolute_full_sum = 0.0f;
    bool converged = true;
    for (int qy = 0; qy < AREA_QUADRATURE && converged; ++qy) {
        for (int qx = 0; qx < AREA_QUADRATURE; ++qx) {
            const float pixel_x = float(ox) +
                                  (float(qx) + 0.5f) / AREA_QUADRATURE;
            const float pixel_y = float(oy) +
                                  (float(qy) + 0.5f) / AREA_QUADRATURE;
            const float xd = (pixel_x - params.src_cx) / params.src_fx;
            const float yd = (pixel_y - params.src_cy) / params.src_fy;
            float ux, uy;
            if (!inverse_distortion(xd, yd, params, ux, uy)) {
                converged = false;
                break;
            }
            const float sx = ux * params.dst_fx + params.dst_cx - PIXEL_CENTER_OFFSET;
            const float sy = uy * params.dst_fy + params.dst_cy - PIXEL_CENTER_OFFSET;
            if (!isfinite(sx) || !isfinite(sy)) {
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
    static const float inverse_samples = 1.0f / (AREA_QUADRATURE * AREA_QUADRATURE);
    for (int channel = 0; channel < channels; ++channel)
        dst[channel * output_plane + output_index] = result[channel] * inverse_samples;
}

void undistort_area_kernel(
    ImageSource src,
    Ptr<float, Access.ReadWrite> dst,
    const int channels,
    const int mode,
    const int ox, const int oy,
    const int quadrature,
    const UndistortParams params) {
    if (ox >= params.dst_width || oy >= params.dst_height)
        return;

    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float valid_samples = 0.0f;
    const int input_plane = params.src_width * params.src_height;
    for (int qy = 0; qy < quadrature; ++qy) {
        for (int qx = 0; qx < quadrature; ++qx) {
            const float pixel_x = float(ox) +
                                  (float(qx) + 0.5f) / quadrature;
            const float pixel_y = float(oy) +
                                  (float(qy) + 0.5f) / quadrature;
            const float nx = (pixel_x - params.dst_cx) / params.dst_fx;
            const float ny = (pixel_y - params.dst_cy) / params.dst_fy;
            float dnx, dny;
            apply_distortion(nx, ny, params.model_type, params.distortion,
                             params.num_distortion, dnx, dny);
            const float sx = dnx * params.src_fx + params.src_cx - PIXEL_CENTER_OFFSET;
            const float sy = dny * params.src_fy + params.src_cy - PIXEL_CENTER_OFFSET;
            if (mode == 1) {
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
                        offset_image_source(src, channel * input_plane),
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
    if (mode == 2) {
        const float norm = sqrt(result[0] * result[0] + result[1] * result[1] +
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

void distort_area_to_source_kernel(
    ImageSource src,
    Ptr<float, Access.ReadWrite> dst,
    const int channels,
    const int mode,
    const int ox, const int oy,
    const UndistortParams params) {
    if (ox >= params.src_width || oy >= params.src_height)
        return;

    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float valid_samples = 0.0f;
    const int input_plane = params.dst_width * params.dst_height;
    bool converged = true;
    for (int qy = 0; qy < AREA_QUADRATURE && converged; ++qy) {
        for (int qx = 0; qx < AREA_QUADRATURE; ++qx) {
            const float pixel_x = float(ox) +
                                  (float(qx) + 0.5f) / AREA_QUADRATURE;
            const float pixel_y = float(oy) +
                                  (float(qy) + 0.5f) / AREA_QUADRATURE;
            const float xd = (pixel_x - params.src_cx) / params.src_fx;
            const float yd = (pixel_y - params.src_cy) / params.src_fy;
            float ux, uy;
            if (!inverse_distortion(xd, yd, params, ux, uy)) {
                converged = false;
                break;
            }
            const float sx = ux * params.dst_fx + params.dst_cx - PIXEL_CENTER_OFFSET;
            const float sy = uy * params.dst_fy + params.dst_cy - PIXEL_CENTER_OFFSET;
            if (mode == 1) {
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
                        offset_image_source(src, channel * input_plane),
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
    if (mode == 2) {
        const float norm = sqrt(result[0] * result[0] + result[1] * result[1] +
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

#ifdef LFS_WARP_GPU_PROGRAM
#define main imageWarp
#endif
[numthreads(256, 1, 1)] void main(uint3 tid : SV_DispatchThreadID, uint3 groups : SV_GroupID) {
    UndistortParams params = ((Ptr<UndistortParams, Access.Read>)pc.camera)[0];
#ifndef LFS_WARP_GPU_PROGRAM
    params.model_type = int(warpModel);
#endif
    ImageSource src = (ImageSource)pc.input;
    Ptr<float, Access.ReadWrite> dst = (Ptr<float, Access.ReadWrite>)pc.output;
    const uint count = uint(pc.width * pc.height);
    const uint stride = min(65535u, (count + 255u) / 256u) * 256u;
    for (uint index = tid.x; index < count; index += stride) {
        int x = int(index) % pc.width, y = int(index) / pc.width;
        if ((WARP_MODE & 255) == 4) {
            float ux, uy;
            const bool valid = inverse_distortion((float(x) + 0.5 - params.src_cx) / params.src_fx,
                                                  (float(y) + 0.5 - params.src_cy) / params.src_fy, params, ux, uy);
            dst[2 * index] = valid ? ux : asfloat(0x7fc00000u);
            dst[2 * index + 1] = valid ? uy : asfloat(0x7fc00000u);
        } else if ((WARP_MODE & 255) == 0) {
            if (WARP_INVERSE != 0)
                distort_image_to_source_kernel(src, dst, (Ptr<int, Access.ReadWrite>)pc.validity, WARP_CHANNELS, x, y, params);
            else
                undistort_image_kernel(src, dst, WARP_CHANNELS, x, y, WARP_QUADRATURE, params);
        } else {
            if (WARP_INVERSE != 0)
                distort_area_to_source_kernel(src, dst, WARP_CHANNELS, (WARP_MODE & 255) - 1, x, y, params);
            else
                undistort_area_kernel(src, dst, WARP_CHANNELS, (WARP_MODE & 255) - 1, x, y, WARP_QUADRATURE, params);
        }
    }
}
