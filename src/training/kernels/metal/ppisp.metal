// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: GPL-3.0-or-later AND Apache-2.0

// PPISPOps and ControllerOps kernels: ports of ppisp.cu, ppisp_math.cuh,
// ppisp_math_bwd.cuh and ppisp_controller.cu. bilateral.metal reuses the
// exposure and color stages. Parameters stay flat: vignetting [cam][3][5]
// (cx, cy, alpha0..2), color [frame][8] (b, r, g, n), CRF [cam][3][4]
// (toe, shoulder, gamma, center).

// Fast math may fold isfinite; test the exponent bits instead.
static bool ppisp_isfinite(const float v) { return (as_type<uint>(v) & 0x7f800000u) != 0x7f800000u; }
static float ppisp_finite_or_zero(const float v) { return ppisp_isfinite(v) ? v : 0.0f; }
static float2 ppisp_finite_or_zero(const float2 v) { return float2(ppisp_finite_or_zero(v.x), ppisp_finite_or_zero(v.y)); }

// __powf(x, y) = exp2(y * log2(x)); every caller has x >= 0 and y > 0 at x == 0.
static float ppisp_pow(const float x, const float y) { return x > 0.0f ? exp2(y * log2(x)) : 0.0f; }

constant constexpr float PPISP_MAX_SHAPE_RAW = 32.0f;
constant constexpr float PPISP_CENTER_EPSILON = 1.0e-4f;
constant constexpr float PPISP_MIN_EXPOSURE_EV = -16.0f;
constant constexpr float PPISP_MAX_EXPOSURE_EV = 16.0f;

constant float PPISP_COLOR_PINV_BLOCKS[16] = {
    0.0480542f, -0.0043631f, -0.0043631f, 0.0481283f,
    0.0580570f, -0.0179872f, -0.0179872f, 0.0431061f,
    0.0433336f, -0.0180537f, -0.0180537f, 0.0580500f,
    0.0128369f, -0.0034654f, -0.0034654f, 0.0128158f};

// Row-major 3x3, as float3x3 of ppisp_math.cuh.
struct PpispMat3 {
    float m[9];
};

struct PpispColor {
    float2 b, r, g, n;
};

static PpispMat3 ppisp_mat3(const float m00, const float m01, const float m02, const float m10, const float m11,
                            const float m12, const float m20, const float m21, const float m22) {
    PpispMat3 a;
    a.m[0] = m00;
    a.m[1] = m01;
    a.m[2] = m02;
    a.m[3] = m10;
    a.m[4] = m11;
    a.m[5] = m12;
    a.m[6] = m20;
    a.m[7] = m21;
    a.m[8] = m22;
    return a;
}

static float ppisp_dot(const float2 a, const float2 b) { return fma(a.x, b.x, a.y * b.y); }
static float ppisp_dot(const float3 a, const float3 b) { return fma(a.x, b.x, fma(a.y, b.y, a.z * b.z)); }

static float3 ppisp_cross(const float3 a, const float3 b) {
    return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

static float3 ppisp_mul(const PpispMat3 a, const float3 v) {
    return float3(fma(a.m[0], v.x, fma(a.m[1], v.y, a.m[2] * v.z)),
                  fma(a.m[3], v.x, fma(a.m[4], v.y, a.m[5] * v.z)),
                  fma(a.m[6], v.x, fma(a.m[7], v.y, a.m[8] * v.z)));
}

static PpispMat3 ppisp_mul(const PpispMat3 a, const PpispMat3 b) {
    PpispMat3 c;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            c.m[i * 3 + j] = fma(a.m[i * 3], b.m[j], fma(a.m[i * 3 + 1], b.m[3 + j], a.m[i * 3 + 2] * b.m[6 + j]));
    return c;
}

static PpispMat3 ppisp_transpose(const PpispMat3 a) {
    return ppisp_mat3(a.m[0], a.m[3], a.m[6], a.m[1], a.m[4], a.m[7], a.m[2], a.m[5], a.m[8]);
}

// PPISP_COLOR_PINV_BLOCKS[block] times v.
static float2 ppisp_zca(const int block, const float2 v) {
    constant float* z = PPISP_COLOR_PINV_BLOCKS + block * 4;
    return float2(fma(z[0], v.x, z[1] * v.y), fma(z[2], v.x, z[3] * v.y));
}

static float2 ppisp_zca_transposed(const int block, const float2 v) {
    constant float* z = PPISP_COLOR_PINV_BLOCKS + block * 4;
    return float2(fma(z[0], v.x, z[2] * v.y), fma(z[1], v.x, z[3] * v.y));
}

static PpispColor ppisp_load_color(device const float* color) {
    return {float2(color[0], color[1]), float2(color[2], color[3]), float2(color[4], color[5]),
            float2(color[6], color[7])};
}

static float ppisp_sigmoid(const float raw) {
    const float value = ppisp_finite_or_zero(raw);
    if (value >= 0.0f)
        return 1.0f / (1.0f + exp(-value));
    const float e = exp(value);
    return e / (1.0f + e);
}

static float ppisp_exposure_value(const float raw) {
    return fmin(PPISP_MAX_EXPOSURE_EV, fmax(PPISP_MIN_EXPOSURE_EV, ppisp_finite_or_zero(raw)));
}

static float ppisp_bounded_positive_forward(const float raw, const float min_value) {
    const float value = fmin(PPISP_MAX_SHAPE_RAW, ppisp_finite_or_zero(raw));
    return min_value + fmax(value, 0.0f) + log(1.0f + exp(-fabs(value)));
}

static float ppisp_clamped_forward(const float raw) {
    return fmin(1.0f - PPISP_CENTER_EPSILON, fmax(PPISP_CENTER_EPSILON, ppisp_sigmoid(raw)));
}

// The homography's parts, kept for its backward pass.
struct PpispHomography {
    PpispMat3 t, skew, m, d, td, h_unnorm, h;
    float3 lambda;
};

static PpispHomography ppisp_homography_parts(const PpispColor c) {
    const float2 bd = ppisp_zca(0, ppisp_finite_or_zero(c.b));
    const float2 rd = ppisp_zca(1, ppisp_finite_or_zero(c.r));
    const float2 gd = ppisp_zca(2, ppisp_finite_or_zero(c.g));
    const float2 nd = ppisp_zca(3, ppisp_finite_or_zero(c.n));
    const float3 t_b = float3(0.0f + bd.x, 0.0f + bd.y, 1.0f);
    const float3 t_r = float3(1.0f + rd.x, 0.0f + rd.y, 1.0f);
    const float3 t_g = float3(0.0f + gd.x, 1.0f + gd.y, 1.0f);
    const float3 t_gray = float3(1.0f / 3.0f + nd.x, 1.0f / 3.0f + nd.y, 1.0f);

    PpispHomography p;
    p.t = ppisp_mat3(t_b.x, t_r.x, t_g.x, t_b.y, t_r.y, t_g.y, t_b.z, t_r.z, t_g.z);
    p.skew = ppisp_mat3(0.0f, -t_gray.z, t_gray.y, t_gray.z, 0.0f, -t_gray.x, -t_gray.y, t_gray.x, 0.0f);
    p.m = ppisp_mul(p.skew, p.t);
    const float3 r0 = float3(p.m.m[0], p.m.m[1], p.m.m[2]);
    const float3 r1 = float3(p.m.m[3], p.m.m[4], p.m.m[5]);
    const float3 r2 = float3(p.m.m[6], p.m.m[7], p.m.m[8]);
    float3 lambda = ppisp_cross(r0, r1);
    float n2 = ppisp_dot(lambda, lambda);
    if (n2 < 1.0e-20f) {
        lambda = ppisp_cross(r0, r2);
        n2 = ppisp_dot(lambda, lambda);
        if (n2 < 1.0e-20f)
            lambda = ppisp_cross(r1, r2);
    }
    p.lambda = lambda;
    const PpispMat3 s_inv = ppisp_mat3(-1.0f, -1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
    p.d = ppisp_mat3(lambda.x, 0.0f, 0.0f, 0.0f, lambda.y, 0.0f, 0.0f, 0.0f, lambda.z);
    p.td = ppisp_mul(p.t, p.d);
    p.h_unnorm = ppisp_mul(p.td, s_inv);
    p.h = p.h_unnorm;
    const float s = p.h.m[8];
    if (fabs(s) > 1.0e-20f) {
        const float inv_s = 1.0f / s;
        for (int i = 0; i < 9; ++i)
            p.h.m[i] *= inv_s;
    }
    return p;
}

static float3 ppisp_apply_exposure(const float3 rgb, const float exposure) {
    return rgb * exp2(ppisp_exposure_value(exposure));
}

static float3 ppisp_apply_vignetting(const float3 rgb, device const float* vignetting, const float2 pixel,
                                     const float resolution_x, const float resolution_y) {
    const float max_res = fmax(resolution_x, resolution_y);
    const float2 uv = float2((pixel.x - resolution_x * 0.5f) / max_res, (pixel.y - resolution_y * 0.5f) / max_res);
    float3 out = rgb;
    for (int i = 0; i < 3; ++i) {
        device const float* v = vignetting + i * 5;
        const float dx = uv.x - ppisp_finite_or_zero(v[0]);
        const float dy = uv.y - ppisp_finite_or_zero(v[1]);
        const float r2 = fma(dx, dx, dy * dy);
        const float r4 = r2 * r2;
        const float r6 = r4 * r2;
        float falloff = fma(ppisp_finite_or_zero(v[4]), r6,
                            fma(ppisp_finite_or_zero(v[3]), r4, fma(ppisp_finite_or_zero(v[2]), r2, 1.0f)));
        falloff = fmax(0.0f, fmin(1.0f, falloff));
        out[i] *= falloff;
    }
    return out;
}

static float3 ppisp_apply_color_correction(const float3 rgb_in, const PpispColor color) {
    const PpispMat3 h = ppisp_homography_parts(color).h;
    // Chromaticity normalization needs nonnegative channels.
    const float3 rgb = fmax(rgb_in, 0.0f);
    const float intensity = rgb.x + rgb.y + rgb.z;
    float3 rgi = ppisp_mul(h, float3(rgb.x, rgb.y, intensity));
    rgi = rgi * (intensity / (fmax(rgi.z, 0.0f) + 1.0e-5f));
    return float3(rgi.x, rgi.y, rgi.z - rgi.x - rgi.y);
}

struct PpispCrfChannel {
    float toe, shoulder, gamma, center;
};

static PpispCrfChannel ppisp_crf_channel(device const float* crf) {
    return {ppisp_bounded_positive_forward(crf[0], 0.3f), ppisp_bounded_positive_forward(crf[1], 0.3f),
            ppisp_bounded_positive_forward(crf[2], 0.1f), ppisp_clamped_forward(crf[3])};
}

static float3 ppisp_apply_crf(const float3 rgb_in, device const float* crf) {
    const float3 rgb = clamp(rgb_in, 0.0f, 1.0f);
    float3 out;
    for (int i = 0; i < 3; ++i) {
        const PpispCrfChannel c = ppisp_crf_channel(crf + i * 4);
        const float lerp_val = fma(c.shoulder - c.toe, c.center, c.toe);
        const float a = (c.shoulder * c.center) / lerp_val;
        const float b = 1.0f - a;
        const float x = rgb[i];
        const float y = x <= c.center ? a * ppisp_pow(x / c.center, c.toe)
                                      : 1.0f - b * ppisp_pow((1.0f - x) / (1.0f - c.center), c.shoulder);
        out[i] = ppisp_pow(fmax(0.0f, y), c.gamma);
    }
    return out;
}

// Backward stages. Parameter gradients accumulate into the caller's arrays.

static float ppisp_bounded_positive_backward(const float raw, const float grad) {
    if (!ppisp_isfinite(raw) || !ppisp_isfinite(grad) || raw >= PPISP_MAX_SHAPE_RAW)
        return 0.0f;
    return grad * ppisp_sigmoid(raw);
}

static float ppisp_clamped_backward(const float raw, const float grad) {
    if (!ppisp_isfinite(raw) || !ppisp_isfinite(grad))
        return 0.0f;
    const float s = ppisp_sigmoid(raw);
    if (s <= PPISP_CENTER_EPSILON || s >= 1.0f - PPISP_CENTER_EPSILON)
        return 0.0f;
    return grad * s * (1.0f - s);
}

static float3 ppisp_apply_exposure_bwd(const float3 rgb_in, const float exposure, const float3 grad_out,
                                       thread float& grad_exposure) {
    const float factor = exp2(ppisp_exposure_value(exposure));
    const float3 rgb_out = rgb_in * factor;
    const bool has_gradient = ppisp_isfinite(exposure) && exposure > PPISP_MIN_EXPOSURE_EV &&
                              exposure < PPISP_MAX_EXPOSURE_EV;
    grad_exposure = has_gradient ? ppisp_dot(grad_out, rgb_out) * 0.69314718f : 0.0f;
    if (!ppisp_isfinite(grad_exposure))
        grad_exposure = 0.0f;
    return grad_out * factor;
}

static float3 ppisp_apply_vignetting_bwd(const float3 rgb_in, device const float* vignetting, const float2 pixel,
                                         const float resolution_x, const float resolution_y, const float3 grad_out,
                                         thread float* grad_vignetting) {
    const float max_res = fmax(resolution_x, resolution_y);
    const float2 uv = float2((pixel.x - resolution_x * 0.5f) / max_res, (pixel.y - resolution_y * 0.5f) / max_res);
    float3 grad_in;
    for (int i = 0; i < 3; ++i) {
        device const float* v = vignetting + i * 5;
        const float cx = ppisp_finite_or_zero(v[0]);
        const float cy = ppisp_finite_or_zero(v[1]);
        const float alpha0 = ppisp_finite_or_zero(v[2]);
        const float alpha1 = ppisp_finite_or_zero(v[3]);
        const float alpha2 = ppisp_finite_or_zero(v[4]);
        const float dx = uv.x - cx;
        const float dy = uv.y - cy;
        const float r2 = fma(dx, dx, dy * dy);
        const float r4 = r2 * r2;
        const float r6 = r4 * r2;
        const float falloff = fma(alpha2, r6, fma(alpha1, r4, fma(alpha0, r2, 1.0f)));
        grad_in[i] = grad_out[i] * fmax(0.0f, fmin(1.0f, falloff));
        const float grad_falloff = grad_out[i] * rgb_in[i];
        if (falloff >= 0.0f && falloff <= 1.0f) {
            thread float* g = grad_vignetting + i * 5;
            g[2] += grad_falloff * r2;
            g[3] += grad_falloff * r4;
            g[4] += grad_falloff * r6;
            const float grad_r2 = grad_falloff * fma(3.0f * alpha2, r4, fma(2.0f * alpha1, r2, alpha0));
            g[0] += -grad_r2 * 2.0f * dx;
            g[1] += -grad_r2 * 2.0f * dy;
        }
    }
    return grad_in;
}

// grad_a = grad_c * transpose(b), grad_b = transpose(a) * grad_c.
static void ppisp_mul_mat_bwd(const PpispMat3 a, const PpispMat3 b, const PpispMat3 grad_c, thread PpispMat3& grad_a,
                              thread PpispMat3& grad_b) {
    grad_a = ppisp_mul(grad_c, ppisp_transpose(b));
    grad_b = ppisp_mul(ppisp_transpose(a), grad_c);
}

static void ppisp_homography_bwd(const PpispColor color, const PpispMat3 grad_h, thread float* grad_color) {
    const PpispHomography p = ppisp_homography_parts(color);

    PpispMat3 grad_h_unnorm;
    const float s = p.h_unnorm.m[8];
    if (fabs(s) > 1.0e-20f) {
        const float inv_s = 1.0f / s;
        const float inv_s2 = inv_s * inv_s;
        float grad_s = 0.0f;
        for (int i = 0; i < 9; ++i) {
            grad_h_unnorm.m[i] = grad_h.m[i] * inv_s;
            grad_s += -grad_h.m[i] * p.h_unnorm.m[i] * inv_s2;
        }
        grad_h_unnorm.m[8] += grad_s;
    } else {
        grad_h_unnorm = grad_h;
    }

    const PpispMat3 s_inv = ppisp_mat3(-1.0f, -1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
    PpispMat3 grad_td, unused, grad_t, grad_d;
    ppisp_mul_mat_bwd(p.td, s_inv, grad_h_unnorm, grad_td, unused);
    ppisp_mul_mat_bwd(p.t, p.d, grad_td, grad_t, grad_d);
    const float3 grad_lambda = float3(grad_d.m[0], grad_d.m[4], grad_d.m[8]);

    // The null vector's gradient flows into the two rows its cross product used.
    const float3 r0 = float3(p.m.m[0], p.m.m[1], p.m.m[2]);
    const float3 r1 = float3(p.m.m[3], p.m.m[4], p.m.m[5]);
    const float3 r2 = float3(p.m.m[6], p.m.m[7], p.m.m[8]);
    float3 grad_r0 = 0.0f, grad_r1 = 0.0f, grad_r2 = 0.0f;
    float3 test = ppisp_cross(r0, r1);
    float n2 = ppisp_dot(test, test);
    if (n2 < 1.0e-20f) {
        test = ppisp_cross(r0, r2);
        n2 = ppisp_dot(test, test);
        if (n2 < 1.0e-20f) {
            grad_r1 = ppisp_cross(r2, grad_lambda);
            grad_r2 = ppisp_cross(grad_lambda, r1);
        } else {
            grad_r0 = ppisp_cross(r2, grad_lambda);
            grad_r2 = ppisp_cross(grad_lambda, r0);
        }
    } else {
        grad_r0 = ppisp_cross(r1, grad_lambda);
        grad_r1 = ppisp_cross(grad_lambda, r0);
    }
    const PpispMat3 grad_m = ppisp_mat3(grad_r0.x, grad_r0.y, grad_r0.z, grad_r1.x, grad_r1.y, grad_r1.z,
                                        grad_r2.x, grad_r2.y, grad_r2.z);

    PpispMat3 grad_skew, grad_t_from_m;
    ppisp_mul_mat_bwd(p.skew, p.t, grad_m, grad_skew, grad_t_from_m);
    for (int i = 0; i < 9; ++i)
        grad_t.m[i] += grad_t_from_m.m[i];

    float3 grad_t_gray = 0.0f;
    grad_t_gray.z += -grad_skew.m[1];
    grad_t_gray.y += grad_skew.m[2];
    grad_t_gray.z += grad_skew.m[3];
    grad_t_gray.x += -grad_skew.m[5];
    grad_t_gray.y += -grad_skew.m[6];
    grad_t_gray.x += grad_skew.m[7];

    const float2 grad_bd = float2(grad_t.m[0], grad_t.m[3]);
    const float2 grad_rd = float2(grad_t.m[1], grad_t.m[4]);
    const float2 grad_gd = float2(grad_t.m[2], grad_t.m[5]);
    const float2 grad_nd = grad_t_gray.xy;
    const float2 grads[4] = {ppisp_zca_transposed(0, grad_bd), ppisp_zca_transposed(1, grad_rd),
                             ppisp_zca_transposed(2, grad_gd), ppisp_zca_transposed(3, grad_nd)};
    for (int i = 0; i < 4; ++i) {
        grad_color[i * 2] += grads[i].x;
        grad_color[i * 2 + 1] += grads[i].y;
    }
}

static float3 ppisp_apply_color_correction_bwd(const float3 rgb_in, const PpispColor color, const float3 grad_out,
                                               thread float* grad_color) {
    const PpispMat3 h = ppisp_homography_parts(color).h;
    const float3 rgb = fmax(rgb_in, 0.0f);
    const float intensity = rgb.x + rgb.y + rgb.z;
    const float3 rgi_in = float3(rgb.x, rgb.y, intensity);
    const float3 rgi_out = ppisp_mul(h, rgi_in);
    const float z_safe = fmax(rgi_out.z, 0.0f);
    const float norm_factor = intensity / (z_safe + 1.0e-5f);

    const float3 grad_rgi_out_norm = float3(grad_out.x - grad_out.z, grad_out.y - grad_out.z, grad_out.z);
    float3 grad_rgi_out = grad_rgi_out_norm * norm_factor;
    const float grad_norm_factor = ppisp_dot(grad_rgi_out_norm, rgi_out);
    // d(norm_factor)/d(rgi_out.z) is zero on the clamped (z <= 0) side.
    if (rgi_out.z > 0.0f)
        grad_rgi_out.z = grad_rgi_out.z + (-grad_norm_factor * norm_factor / (rgi_out.z + 1.0e-5f));

    PpispMat3 grad_h;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            grad_h.m[i * 3 + j] = grad_rgi_out[i] * rgi_in[j];
    const float3 grad_rgi_in = ppisp_mul(ppisp_transpose(h), grad_rgi_out);

    const float grad_intensity = grad_norm_factor / (z_safe + 1.0e-5f);
    float3 grad_in = float3(grad_rgi_in.x + grad_rgi_in.z, grad_rgi_in.y + grad_rgi_in.z, grad_rgi_in.z);
    grad_in = grad_in + grad_intensity;
    // Match the lower clamp of the forward stage.
    grad_in = select(grad_in, float3(0.0f), rgb_in < 0.0f);

    ppisp_homography_bwd(color, grad_h, grad_color);
    return grad_in;
}

static float3 ppisp_apply_crf_bwd(const float3 rgb_in, device const float* crf, const float3 grad_out,
                                  thread float* grad_crf) {
    const float3 rgb = clamp(rgb_in, 0.0f, 1.0f);
    float3 grad_in;
    for (int i = 0; i < 3; ++i) {
        device const float* raw = crf + i * 4;
        const PpispCrfChannel c = ppisp_crf_channel(raw);
        const float toe = c.toe, shoulder = c.shoulder, gamma = c.gamma, center = c.center;
        const float lerp_val = fma(shoulder - toe, center, toe);
        const float a = (shoulder * center) / lerp_val;
        const float b = 1.0f - a;
        const float x = rgb[i];
        const float y = x <= center ? a * ppisp_pow(x / center, toe)
                                    : 1.0f - b * ppisp_pow((1.0f - x) / (1.0f - center), shoulder);
        const float output = ppisp_pow(fmax(0.0f, y), gamma);
        const float y_clamped = fmax(0.0f, y);

        const float grad_y = y_clamped > 0.0f ? grad_out[i] * gamma * ppisp_pow(y_clamped, gamma - 1.0f) : 0.0f;
        float grad_x = 0.0f;
        if (x <= center && center > 0.0f) {
            const float base = x / center;
            if (base > 0.0f)
                grad_x = grad_y * a * toe * ppisp_pow(base, toe - 1.0f) / center;
        } else if (x > center && center < 1.0f) {
            const float base = (1.0f - x) / (1.0f - center);
            if (base > 0.0f)
                grad_x = grad_y * b * shoulder * ppisp_pow(base, shoulder - 1.0f) / (1.0f - center);
        }
        grad_in[i] = grad_x;

        float grad_toe = 0.0f, grad_shoulder = 0.0f, grad_center = 0.0f;
        const float grad_gamma = y_clamped > 0.0f ? grad_out[i] * output * log(y_clamped + 1e-8f) : 0.0f;
        float grad_a = 0.0f, grad_b = 0.0f;
        if (x <= center && center > 0.0f) {
            const float base = x / center;
            if (base > 0.0f) {
                const float powered = ppisp_pow(base, toe);
                grad_a += grad_y * powered;
                grad_toe += grad_y * a * powered * log(base + 1e-8f);
                const float grad_base = grad_y * a * toe * ppisp_pow(base, toe - 1.0f);
                grad_center += grad_base * (-x / (center * center));
            }
        } else if (x > center && center < 1.0f) {
            const float base = (1.0f - x) / (1.0f - center);
            if (base > 0.0f) {
                const float powered = ppisp_pow(base, shoulder);
                grad_b += -grad_y * powered;
                grad_shoulder += -grad_y * b * powered * log(base + 1e-8f);
                const float grad_base = grad_y * (-b * shoulder * ppisp_pow(base, shoulder - 1.0f));
                grad_center += grad_base * ((1.0f - x) / ((1.0f - center) * (1.0f - center)));
            }
        }
        grad_a += -grad_b;
        float grad_lerp_val = 0.0f;
        if (fabs(lerp_val) > 1e-8f) {
            const float a_over_lerp = (shoulder * center) / lerp_val;
            grad_shoulder += grad_a * center / lerp_val;
            grad_center += grad_a * shoulder / lerp_val;
            grad_lerp_val += -grad_a * a_over_lerp / lerp_val;
        }
        grad_shoulder += grad_lerp_val * center;
        grad_toe += grad_lerp_val * (1.0f - center);
        grad_center += grad_lerp_val * (shoulder - toe);

        thread float* g = grad_crf + i * 4;
        g[0] += ppisp_bounded_positive_backward(raw[0], grad_toe);
        g[1] += ppisp_bounded_positive_backward(raw[1], grad_shoulder);
        g[2] += ppisp_bounded_positive_backward(raw[2], grad_gamma);
        g[3] += ppisp_clamped_backward(raw[3], grad_center);
    }
    return grad_in;
}

// Sum of a threadgroup of kThreadgroupWidth threads, as cub::BlockReduce adds:
// ascending shuffle offsets within a simdgroup, then the simdgroup sums in order.
static float ppisp_group_sum(const float value, threadgroup float* partials, const uint lane, const uint simd,
                             const uint simds) {
    float v = value;
    for (ushort offset = 1; offset < 32; offset <<= 1) {
        const float other = simd_shuffle_down(v, offset);
        v += lane + offset < 32 ? other : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0)
        partials[simd] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = partials[0];
    for (uint i = 1; i < simds; ++i)
        total += partials[i];
    return total;
}

static void ppisp_atomic_add(device float* address, const float value) {
    atomic_fetch_add_explicit((device atomic_float*)address, value, memory_order_relaxed);
}

struct PpispForwardParams {
    device const float* exposure;
    device const float* vignetting;
    device const float* color;
    device const float* crf;
    device const float* rgb_in;
    device float* rgb_out;
    int height, width, y_offset, full_height;
    int camera_index, frame_index, x_offset, full_width;
};

kernel void ppisp_forward(constant PpispForwardParams& p [[buffer(0)]], uint index [[thread_position_in_grid]]) {
    const int pixels = p.height * p.width;
    if (int(index) >= pixels)
        return;
    const int y = int(index) / p.width;
    const int x = int(index) % p.width;
    float3 rgb = float3(p.rgb_in[index], p.rgb_in[pixels + index], p.rgb_in[2 * pixels + index]);
    const float2 pixel = float2(float(p.x_offset + x) + 0.5f, float(p.y_offset + y) + 0.5f);
    if (p.frame_index != -1)
        rgb = ppisp_apply_exposure(rgb, p.exposure[p.frame_index]);
    if (p.camera_index != -1)
        rgb = ppisp_apply_vignetting(rgb, p.vignetting + p.camera_index * 15, pixel, float(p.full_width),
                                     float(p.full_height));
    if (p.frame_index != -1)
        rgb = ppisp_apply_color_correction(rgb, ppisp_load_color(p.color + p.frame_index * 8));
    if (p.camera_index != -1)
        rgb = ppisp_apply_crf(rgb, p.crf + p.camera_index * 12);
    p.rgb_out[index] = rgb.x;
    p.rgb_out[pixels + index] = rgb.y;
    p.rgb_out[2 * pixels + index] = rgb.z;
}

struct PpispBackwardParams {
    device const float* exposure;
    device const float* vignetting;
    device const float* color;
    device const float* crf;
    device const float* rgb_in;
    device const float* grad_rgb_out;
    device float* grad_exposure;
    device float* grad_vignetting;
    device float* grad_color;
    device float* grad_crf;
    device float* grad_rgb_in;
    int height, width, camera_index, frame_index;
};

// Gradient slots per threadgroup: exposure, color[8], vignetting[15], CRF[12].
constant constexpr int kPpispGradients = 36;

kernel void ppisp_backward(constant PpispBackwardParams& p [[buffer(0)]], uint index [[thread_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]],
                           uint simds [[simdgroups_per_threadgroup]], uint local [[thread_index_in_threadgroup]]) {
    threadgroup float partials[32];
    threadgroup float totals[kPpispGradients];
    float g[kPpispGradients];
    for (int i = 0; i < kPpispGradients; ++i)
        g[i] = 0.0f;
    const int pixels = p.height * p.width;
    if (int(index) < pixels) {
        const int y = int(index) / p.width;
        const int x = int(index) % p.width;
        const float3 rgb_input = float3(p.rgb_in[index], p.rgb_in[pixels + index], p.rgb_in[2 * pixels + index]);
        const float2 pixel = float2(float(x) + 0.5f, float(y) + 0.5f);
        device const float* vignetting = p.vignetting + p.camera_index * 15;
        device const float* crf = p.crf + p.camera_index * 12;

        float3 rgb = rgb_input;
        float3 after_exposure = rgb, after_vignetting = rgb, after_color = rgb;
        if (p.frame_index != -1) {
            after_exposure = ppisp_apply_exposure(rgb, p.exposure[p.frame_index]);
            rgb = after_exposure;
        }
        if (p.camera_index != -1) {
            after_vignetting = ppisp_apply_vignetting(rgb, vignetting, pixel, float(p.width), float(p.height));
            rgb = after_vignetting;
        } else {
            after_vignetting = rgb;
        }
        if (p.frame_index != -1) {
            after_color = ppisp_apply_color_correction(rgb, ppisp_load_color(p.color + p.frame_index * 8));
        } else {
            after_color = rgb;
        }

        float3 grad = float3(p.grad_rgb_out[index], p.grad_rgb_out[pixels + index], p.grad_rgb_out[2 * pixels + index]);
        if (p.camera_index != -1)
            grad = ppisp_apply_crf_bwd(after_color, crf, grad, g + 24);
        if (p.frame_index != -1)
            grad = ppisp_apply_color_correction_bwd(after_vignetting, ppisp_load_color(p.color + p.frame_index * 8),
                                                    grad, g + 1);
        if (p.camera_index != -1)
            grad = ppisp_apply_vignetting_bwd(after_exposure, vignetting, pixel, float(p.width), float(p.height),
                                              grad, g + 9);
        if (p.frame_index != -1)
            grad = ppisp_apply_exposure_bwd(rgb_input, p.exposure[p.frame_index], grad, g[0]);
        p.grad_rgb_in[index] = grad.x;
        p.grad_rgb_in[pixels + index] = grad.y;
        p.grad_rgb_in[2 * pixels + index] = grad.z;
    }

    for (int i = 0; i < kPpispGradients; ++i) {
        const float total = ppisp_group_sum(g[i], partials, lane, simd, simds);
        if (local == 0)
            totals[i] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (local >= uint(kPpispGradients))
        return;
    const int i = int(local);
    if (i < 9) {
        if (p.frame_index != -1)
            ppisp_atomic_add(i == 0 ? p.grad_exposure + p.frame_index : p.grad_color + p.frame_index * 8 + (i - 1),
                             totals[i]);
    } else if (p.camera_index != -1) {
        ppisp_atomic_add(i < 24 ? p.grad_vignetting + p.camera_index * 15 + (i - 9)
                                : p.grad_crf + p.camera_index * 12 + (i - 24),
                         totals[i]);
    }
}

struct PpispAdamGroup {
    device float* parameter;
    device float* moment1;
    device float* moment2;
    device const float* gradient;
    uint count;
    uint padding;
};

struct PpispAdamParams {
    PpispAdamGroup groups[4];
    float lr, beta1, beta2, bc1_rcp, bc2_sqrt_rcp, eps;
};

kernel void ppisp_adam(constant PpispAdamParams& p [[buffer(0)]], uint index [[thread_position_in_grid]]) {
    uint i = index;
    int group = 0;
    while (group < 4 && i >= p.groups[group].count) {
        i -= p.groups[group].count;
        ++group;
    }
    if (group == 4)
        return;
    constant PpispAdamGroup& g = p.groups[group];
    const bool valid = ppisp_isfinite(g.parameter[i]);
    const float parameter = valid ? g.parameter[i] : 0.0f;
    const float grad = valid && ppisp_isfinite(g.gradient[i]) ? g.gradient[i] : 0.0f;
    float m = valid && ppisp_isfinite(g.moment1[i]) ? g.moment1[i] : 0.0f;
    float v = valid && ppisp_isfinite(g.moment2[i]) && g.moment2[i] >= 0.0f ? g.moment2[i] : 0.0f;
    m = p.beta1 * m + (1.0f - p.beta1) * grad;
    v = p.beta2 * v + (1.0f - p.beta2) * grad * grad;
    const float m_hat = m * p.bc1_rcp;
    const float v_hat_sqrt = sqrt(v) * p.bc2_sqrt_rcp;
    const float updated = parameter - p.lr * m_hat / (v_hat_sqrt + p.eps);
    const bool ok = ppisp_isfinite(updated) && ppisp_isfinite(m) && ppisp_isfinite(v);
    g.parameter[i] = ok ? updated : parameter;
    g.moment1[i] = ok ? m : 0.0f;
    g.moment2[i] = ok ? v : 0.0f;
}

struct PpispVignettingRegParams {
    device const float* vignetting;
    device float* gradient;
    device float* loss;
    int cameras;
    float center, channel, non_positive;
};

// One thread owns one camera's 15 parameters, so gradients need no atomics.
kernel void ppisp_vignetting_reg(constant PpispVignettingRegParams& p [[buffer(0)]],
                                 uint index [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                                 uint simd [[simdgroup_index_in_threadgroup]],
                                 uint simds [[simdgroups_per_threadgroup]], uint local [[thread_index_in_threadgroup]]) {
    threadgroup float partials[32];
    float local_loss = 0.0f;
    const int cam = int(index);
    if (cam < p.cameras) {
        device const float* v = p.vignetting + cam * 15;
        device float* g = p.gradient ? p.gradient + cam * 15 : nullptr;
        const float inv_center = 1.0f / float(p.cameras * 3);
        const float inv_non_pos = 1.0f / float(p.cameras * 3 * 3);
        const float inv_channel = 1.0f / float(p.cameras * 5 * 3);
        if (p.center != 0.0f) {
            const float s = p.center * 2.0f * inv_center;
            for (int ch = 0; ch < 3; ++ch) {
                const float cx = v[ch * 5], cy = v[ch * 5 + 1];
                local_loss += p.center * (cx * cx + cy * cy) * inv_center;
                if (g) {
                    g[ch * 5] += s * cx;
                    g[ch * 5 + 1] += s * cy;
                }
            }
        }
        if (p.non_positive != 0.0f) {
            for (int ch = 0; ch < 3; ++ch) {
                for (int a = 0; a < 3; ++a) {
                    const float alpha = v[ch * 5 + 2 + a];
                    if (alpha > 0.0f) {
                        local_loss += p.non_positive * alpha * inv_non_pos;
                        if (g)
                            g[ch * 5 + 2 + a] += p.non_positive * inv_non_pos;
                    }
                }
            }
        }
        if (p.channel != 0.0f) {
            const float s = p.channel * 2.0f * inv_channel;
            for (int k = 0; k < 5; ++k) {
                const float vals[3] = {v[k], v[5 + k], v[10 + k]};
                const float mean = (vals[0] + vals[1] + vals[2]) / 3.0f;
                for (int ch = 0; ch < 3; ++ch) {
                    const float diff = vals[ch] - mean;
                    local_loss += p.channel * diff * diff * inv_channel;
                    if (g)
                        g[ch * 5 + k] += s * diff;
                }
            }
        }
    }
    const float total = ppisp_group_sum(local_loss, partials, lane, simd, simds);
    if (p.loss && local == 0 && total != 0.0f)
        ppisp_atomic_add(p.loss, total);
}

struct PpispProjectMeanParams {
    device float* exposure;
    device float* color;
    int frames;
};

// Threadgroup 0 centers the exposures, threadgroup 1 each color channel.
kernel void ppisp_project_mean(constant PpispProjectMeanParams& p [[buffer(0)]],
                               uint group [[threadgroup_position_in_grid]], uint local [[thread_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]],
                               uint simds [[simdgroups_per_threadgroup]]) {
    threadgroup float partials[32];
    threadgroup float means[8];
    const int n = p.frames;
    const float inv_n = 1.0f / float(n);
    if (group == 0) {
        float sum = 0.0f;
        for (int i = int(local); i < n; i += int(kThreadgroupWidth))
            sum += p.exposure[i];
        const float mean = ppisp_group_sum(sum, partials, lane, simd, simds) * inv_n;
        for (int i = int(local); i < n; i += int(kThreadgroupWidth))
            p.exposure[i] -= mean;
        return;
    }
    for (int c = 0; c < 8; ++c) {
        float sum = 0.0f;
        for (int f = int(local); f < n; f += int(kThreadgroupWidth))
            sum += p.color[f * 8 + c];
        const float total = ppisp_group_sum(sum, partials, lane, simd, simds);
        if (local == 0)
            means[c] = total * inv_n;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = int(local); i < n * 8; i += int(kThreadgroupWidth))
        p.color[i] -= means[i % 8];
}

struct PpispInitializeParams {
    device float* exposure;
    device float* vignetting;
    device float* color;
    device float* crf;
    int cameras, frames;
    float toe, shoulder, gamma;
};

kernel void ppisp_initialize(constant PpispInitializeParams& p [[buffer(0)]], uint index [[thread_position_in_grid]]) {
    const int i = int(index);
    if (i < p.frames)
        p.exposure[i] = 0.0f;
    if (i < p.cameras * 15)
        p.vignetting[i] = 0.0f;
    if (i < p.frames * 8)
        p.color[i] = 0.0f;
    if (i < p.cameras * 12) {
        const int k = i % 4;
        p.crf[i] = k == 0 ? p.toe : k == 1 ? p.shoulder : k == 2 ? p.gamma : 0.0f;
    }
}

// ControllerOps.

struct ControllerPrepareParams {
    device const float* features;
    device float* fc_input;
    float exposure_prior;
    uint write_prior;
};

constant constexpr int kControllerFeatures = 1600;

kernel void controller_prepare_input(constant ControllerPrepareParams& p [[buffer(0)]],
                                     uint index [[thread_position_in_grid]]) {
    if (int(index) < kControllerFeatures && p.features)
        p.fc_input[index] = p.features[index];
    if (int(index) == kControllerFeatures && p.write_prior != 0)
        p.fc_input[index] = p.exposure_prior;
}

struct ControllerOuterParams {
    device const float* grad;
    device const float* activation;
    device float* weight_gradient;
    device float* bias_gradient;
    int m, n;
};

kernel void controller_outer_product(constant ControllerOuterParams& p [[buffer(0)]],
                                     uint index [[thread_position_in_grid]]) {
    if (int(index) >= p.m * p.n)
        return;
    const int i = int(index) / p.n;
    const int j = int(index) % p.n;
    p.weight_gradient[index] += 1.0f * p.grad[i] * p.activation[j];
    if (j == 0)
        p.bias_gradient[i] += p.grad[i];
}

struct ControllerInputGradParams {
    device const float* grad;
    device const float* activation;
    device const float* weight;
    device float* grad_input;
    int m, n;
};

// grad_input = (grad[1 x m] * weight[m x n]) masked by activation > 0.
kernel void controller_input_grad(constant ControllerInputGradParams& p [[buffer(0)]],
                                  uint index [[thread_position_in_grid]]) {
    if (int(index) >= p.n)
        return;
    float sum = 0.0f;
    for (int k = 0; k < p.m; ++k)
        sum += p.grad[k] * p.weight[k * p.n + int(index)];
    p.grad_input[index] = p.activation[index] > 0.0f ? sum : 0.0f;
}
