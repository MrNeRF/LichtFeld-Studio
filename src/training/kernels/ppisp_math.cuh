/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: GPL-3.0-or-later AND Apache-2.0
 *
 * Based on NVIDIA PPISP implementation.
 */

#pragma once

#include <cuda_runtime.h>

// Matrix Types (row-major storage)
struct float2x2 {
    float m[4];

    __device__ __forceinline__ float& operator()(int row, int col) { return m[row * 2 + col]; }
    __device__ __forceinline__ const float& operator()(int row, int col) const { return m[row * 2 + col]; }
};

struct float3x3 {
    float m[9];

    __device__ __forceinline__ float& operator()(int row, int col) { return m[row * 3 + col]; }
    __device__ __forceinline__ const float& operator()(int row, int col) const { return m[row * 3 + col]; }
};

__device__ __forceinline__ float2x2 make_float2x2(float m00, float m01, float m10, float m11) {
    float2x2 mat;
    mat.m[0] = m00;
    mat.m[1] = m01;
    mat.m[2] = m10;
    mat.m[3] = m11;
    return mat;
}

__device__ __forceinline__ float3x3 make_float3x3(float m00, float m01, float m02, float m10, float m11, float m12,
                                                  float m20, float m21, float m22) {
    float3x3 mat;
    mat.m[0] = m00;
    mat.m[1] = m01;
    mat.m[2] = m02;
    mat.m[3] = m10;
    mat.m[4] = m11;
    mat.m[5] = m12;
    mat.m[6] = m20;
    mat.m[7] = m21;
    mat.m[8] = m22;
    return mat;
}

// Vector operators
__device__ __forceinline__ float2 operator+(const float2& a, const float2& b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

__device__ __forceinline__ float2 operator-(const float2& a, const float2& b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

__device__ __forceinline__ float2 operator*(const float2& a, float s) { return make_float2(a.x * s, a.y * s); }

__device__ __forceinline__ float2 operator*(float s, const float2& a) { return make_float2(a.x * s, a.y * s); }

__device__ __forceinline__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float3 operator*(const float3& a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ __forceinline__ float3 operator*(float s, const float3& a) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

// Vector functions
__device__ __forceinline__ float ppisp_dot(const float2& a, const float2& b) {
    return __fmaf_rn(a.x, b.x, a.y * b.y);
}

__device__ __forceinline__ float ppisp_dot(const float3& a, const float3& b) {
    return __fmaf_rn(a.x, b.x, __fmaf_rn(a.y, b.y, a.z * b.z));
}

__device__ __forceinline__ float3 ppisp_cross(const float3& a, const float3& b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

__device__ __forceinline__ float3 ppisp_clamp(const float3& v, float min_val, float max_val) {
    return make_float3(fminf(fmaxf(v.x, min_val), max_val), fminf(fmaxf(v.y, min_val), max_val),
                       fminf(fmaxf(v.z, min_val), max_val));
}

// Matrix-vector multiplication
__device__ __forceinline__ float2 operator*(const float2x2& mat, const float2& v) {
    return make_float2(__fmaf_rn(mat.m[0], v.x, mat.m[1] * v.y), __fmaf_rn(mat.m[2], v.x, mat.m[3] * v.y));
}

__device__ __forceinline__ float3 operator*(const float3x3& mat, const float3& v) {
    return make_float3(__fmaf_rn(mat.m[0], v.x, __fmaf_rn(mat.m[1], v.y, mat.m[2] * v.z)),
                       __fmaf_rn(mat.m[3], v.x, __fmaf_rn(mat.m[4], v.y, mat.m[5] * v.z)),
                       __fmaf_rn(mat.m[6], v.x, __fmaf_rn(mat.m[7], v.y, mat.m[8] * v.z)));
}

// Matrix-matrix multiplication
__device__ __forceinline__ float3x3 operator*(const float3x3& A, const float3x3& B) {
    float3x3 C;
#pragma unroll
    for (int i = 0; i < 3; i++) {
#pragma unroll
        for (int j = 0; j < 3; j++) {
            C(i, j) = __fmaf_rn(A(i, 0), B(0, j), __fmaf_rn(A(i, 1), B(1, j), A(i, 2) * B(2, j)));
        }
    }
    return C;
}

__device__ __forceinline__ float3x3 ppisp_transpose(const float3x3& A) {
    float3x3 AT;
#pragma unroll
    for (int i = 0; i < 3; i++) {
#pragma unroll
        for (int j = 0; j < 3; j++) {
            AT(i, j) = A(j, i);
        }
    }
    return AT;
}

// ISP Parameter structures
struct VignettingChannelParams {
    float cx;
    float cy;
    float alpha0;
    float alpha1;
    float alpha2;
};

struct CRFPPISPChannelParams {
    float toe;
    float shoulder;
    float gamma;
    float center;
};

struct ColorPPISPParams {
    float2 b;
    float2 r;
    float2 g;
    float2 n;
};

constexpr float PPISP_MAX_SHAPE_RAW = 32.0f;
constexpr float PPISP_CENTER_EPSILON = 1.0e-4f;
constexpr float PPISP_MIN_EXPOSURE_EV = -16.0f;
constexpr float PPISP_MAX_EXPOSURE_EV = 16.0f;

__device__ __forceinline__ float ppisp_finite_or_zero(const float value) {
    return isfinite(value) ? value : 0.0f;
}

__device__ __forceinline__ float2 ppisp_finite_or_zero(const float2 value) {
    return make_float2(ppisp_finite_or_zero(value.x), ppisp_finite_or_zero(value.y));
}

__device__ __forceinline__ float ppisp_sigmoid(const float raw) {
    const float value = ppisp_finite_or_zero(raw);
    if (value >= 0.0f) {
        return __fdividef(1.0f, 1.0f + __expf(-value));
    }
    const float exp_value = __expf(value);
    return __fdividef(exp_value, 1.0f + exp_value);
}

__device__ __forceinline__ float ppisp_exposure_value(const float raw) {
    return fminf(PPISP_MAX_EXPOSURE_EV, fmaxf(PPISP_MIN_EXPOSURE_EV, ppisp_finite_or_zero(raw)));
}

// Color correction pinv blocks (constant memory)
__constant__ float PPISP_COLOR_PINV_BLOCKS[4][4] = {
    {0.0480542f, -0.0043631f, -0.0043631f, 0.0481283f},
    {0.0580570f, -0.0179872f, -0.0179872f, 0.0431061f},
    {0.0433336f, -0.0180537f, -0.0180537f, 0.0580500f},
    {0.0128369f, -0.0034654f, -0.0034654f, 0.0128158f}};

__device__ __forceinline__ float ppisp_bounded_positive_forward(float raw, float min_value = 0.1f) {
    const float value = fminf(PPISP_MAX_SHAPE_RAW, ppisp_finite_or_zero(raw));
    return min_value + fmaxf(value, 0.0f) + __logf(1.0f + __expf(-fabsf(value)));
}

__device__ __forceinline__ float ppisp_clamped_forward(float raw) {
    return fminf(1.0f - PPISP_CENTER_EPSILON,
                 fmaxf(PPISP_CENTER_EPSILON, ppisp_sigmoid(raw)));
}

__device__ __forceinline__ float3x3 ppisp_compute_homography(const ColorPPISPParams* params) {
    const float2 b_lat = ppisp_finite_or_zero(params->b);
    const float2 r_lat = ppisp_finite_or_zero(params->r);
    const float2 g_lat = ppisp_finite_or_zero(params->g);
    const float2 n_lat = ppisp_finite_or_zero(params->n);

    float2x2 zca_b, zca_r, zca_g, zca_n;
    zca_b.m[0] = PPISP_COLOR_PINV_BLOCKS[0][0];
    zca_b.m[1] = PPISP_COLOR_PINV_BLOCKS[0][1];
    zca_b.m[2] = PPISP_COLOR_PINV_BLOCKS[0][2];
    zca_b.m[3] = PPISP_COLOR_PINV_BLOCKS[0][3];

    zca_r.m[0] = PPISP_COLOR_PINV_BLOCKS[1][0];
    zca_r.m[1] = PPISP_COLOR_PINV_BLOCKS[1][1];
    zca_r.m[2] = PPISP_COLOR_PINV_BLOCKS[1][2];
    zca_r.m[3] = PPISP_COLOR_PINV_BLOCKS[1][3];

    zca_g.m[0] = PPISP_COLOR_PINV_BLOCKS[2][0];
    zca_g.m[1] = PPISP_COLOR_PINV_BLOCKS[2][1];
    zca_g.m[2] = PPISP_COLOR_PINV_BLOCKS[2][2];
    zca_g.m[3] = PPISP_COLOR_PINV_BLOCKS[2][3];

    zca_n.m[0] = PPISP_COLOR_PINV_BLOCKS[3][0];
    zca_n.m[1] = PPISP_COLOR_PINV_BLOCKS[3][1];
    zca_n.m[2] = PPISP_COLOR_PINV_BLOCKS[3][2];
    zca_n.m[3] = PPISP_COLOR_PINV_BLOCKS[3][3];

    float2 bd = zca_b * b_lat;
    float2 rd = zca_r * r_lat;
    float2 gd = zca_g * g_lat;
    float2 nd = zca_n * n_lat;

    float3 t_b = make_float3(0.0f + bd.x, 0.0f + bd.y, 1.0f);
    float3 t_r = make_float3(1.0f + rd.x, 0.0f + rd.y, 1.0f);
    float3 t_g = make_float3(0.0f + gd.x, 1.0f + gd.y, 1.0f);
    float3 t_gray = make_float3(1.0f / 3.0f + nd.x, 1.0f / 3.0f + nd.y, 1.0f);

    float3x3 T = make_float3x3(t_b.x, t_r.x, t_g.x, t_b.y, t_r.y, t_g.y, t_b.z, t_r.z, t_g.z);

    float3x3 skew =
        make_float3x3(0.0f, -t_gray.z, t_gray.y, t_gray.z, 0.0f, -t_gray.x, -t_gray.y, t_gray.x, 0.0f);

    float3x3 M = skew * T;

    float3 r0 = make_float3(M(0, 0), M(0, 1), M(0, 2));
    float3 r1 = make_float3(M(1, 0), M(1, 1), M(1, 2));
    float3 r2 = make_float3(M(2, 0), M(2, 1), M(2, 2));

    float3 lambda_v = ppisp_cross(r0, r1);
    float n2 = ppisp_dot(lambda_v, lambda_v);

    if (n2 < 1.0e-20f) {
        lambda_v = ppisp_cross(r0, r2);
        n2 = ppisp_dot(lambda_v, lambda_v);
        if (n2 < 1.0e-20f) {
            lambda_v = ppisp_cross(r1, r2);
        }
    }

    float3x3 S_inv = make_float3x3(-1.0f, -1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);

    float3x3 D = make_float3x3(lambda_v.x, 0.0f, 0.0f, 0.0f, lambda_v.y, 0.0f, 0.0f, 0.0f, lambda_v.z);

    float3x3 H = T * D * S_inv;

    float s = H(2, 2);
    if (fabsf(s) > 1.0e-20f) {
        float inv_s = 1.0f / s;
#pragma unroll
        for (int i = 0; i < 9; i++) {
            H.m[i] *= inv_s;
        }
    }

    return H;
}

// ISP component functions
__device__ __forceinline__ void ppisp_apply_exposure(const float3& rgb_in, float exposure_param, float3& rgb_out) {
    const float exposure_factor = exp2f(ppisp_exposure_value(exposure_param));
    rgb_out = rgb_in * exposure_factor;
}

__device__ __forceinline__ void ppisp_apply_vignetting(const float3& rgb_in,
                                                       const VignettingChannelParams* vignetting_params,
                                                       const float2& pixel_coords, float resolution_x,
                                                       float resolution_y, float3& rgb_out) {
    float max_res = fmaxf(resolution_x, resolution_y);
    float2 uv = make_float2(__fdividef(pixel_coords.x - resolution_x * 0.5f, max_res),
                            __fdividef(pixel_coords.y - resolution_y * 0.5f, max_res));

    float rgb_arr[3] = {rgb_in.x, rgb_in.y, rgb_in.z};

#pragma unroll
    for (int i = 0; i < 3; i++) {
        const VignettingChannelParams& params = vignetting_params[i];

        float dx = uv.x - ppisp_finite_or_zero(params.cx);
        float dy = uv.y - ppisp_finite_or_zero(params.cy);
        float r2 = __fmaf_rn(dx, dx, dy * dy);
        float r4 = r2 * r2;
        float r6 = r4 * r2;

        float falloff = __fmaf_rn(
            ppisp_finite_or_zero(params.alpha2), r6,
            __fmaf_rn(ppisp_finite_or_zero(params.alpha1), r4,
                      __fmaf_rn(ppisp_finite_or_zero(params.alpha0), r2, 1.0f)));
        falloff = fmaxf(0.0f, fminf(1.0f, falloff));

        rgb_arr[i] *= falloff;
    }

    rgb_out = make_float3(rgb_arr[0], rgb_arr[1], rgb_arr[2]);
}

__device__ __forceinline__ void ppisp_apply_color_correction(const float3& rgb_in,
                                                             const ColorPPISPParams* color_params, float3& rgb_out) {
    float3x3 H = ppisp_compute_homography(color_params);

    float intensity = rgb_in.x + rgb_in.y + rgb_in.z;
    float3 rgi_in = make_float3(rgb_in.x, rgb_in.y, intensity);

    float3 rgi_out = H * rgi_in;

    float norm_factor = __fdividef(intensity, rgi_out.z + 1.0e-5f);
    rgi_out = rgi_out * norm_factor;

    rgb_out = make_float3(rgi_out.x, rgi_out.y, rgi_out.z - rgi_out.x - rgi_out.y);
}

__device__ __forceinline__ void ppisp_apply_crf(const float3& rgb_in, const CRFPPISPChannelParams* crf_params,
                                                float3& rgb_out) {
    float3 rgb_clamped = ppisp_clamp(rgb_in, 0.0f, 1.0f);
    float out_arr[3];

#pragma unroll
    for (int i = 0; i < 3; i++) {
        float x = (i == 0) ? rgb_clamped.x : (i == 1) ? rgb_clamped.y
                                                      : rgb_clamped.z;

        const CRFPPISPChannelParams& params = crf_params[i];

        float toe = ppisp_bounded_positive_forward(params.toe, 0.3f);
        float shoulder = ppisp_bounded_positive_forward(params.shoulder, 0.3f);
        float gamma = ppisp_bounded_positive_forward(params.gamma, 0.1f);
        float center = ppisp_clamped_forward(params.center);

        float lerp_val = __fmaf_rn(shoulder - toe, center, toe);
        float a = __fdividef(shoulder * center, lerp_val);
        float b = 1.0f - a;

        float y;

        if (x <= center) {
            y = a * __powf(__fdividef(x, center), toe);
        } else {
            y = 1.0f - b * __powf(__fdividef(1.0f - x, 1.0f - center), shoulder);
        }

        out_arr[i] = __powf(fmaxf(0.0f, y), gamma);
    }

    rgb_out = make_float3(out_arr[0], out_arr[1], out_arr[2]);
}
