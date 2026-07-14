/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: GPL-3.0-or-later AND Apache-2.0
 *
 * Based on NVIDIA PPISP implementation.
 */

#pragma once

#include "ppisp_math.cuh"

// Matrix backward helpers
__device__ __forceinline__ float2x2 ppisp_transpose_2x2(const float2x2& A) {
    float2x2 AT;
    AT(0, 0) = A(0, 0);
    AT(0, 1) = A(1, 0);
    AT(1, 0) = A(0, 1);
    AT(1, 1) = A(1, 1);
    return AT;
}

__device__ __forceinline__ void ppisp_cross_bwd(const float3& grad_output, const float3& a, const float3& b,
                                                float3& grad_a, float3& grad_b) {
    grad_a = ppisp_cross(b, grad_output);
    grad_b = ppisp_cross(grad_output, a);
}

__device__ __forceinline__ void ppisp_mul_bwd(const float2x2& A, const float2& x, const float2& grad_y,
                                              float2x2& grad_A, float2& grad_x) {
    grad_x = ppisp_transpose_2x2(A) * grad_y;

#pragma unroll
    for (int i = 0; i < 2; i++) {
        float grad_y_i = (i == 0) ? grad_y.x : grad_y.y;
#pragma unroll
        for (int j = 0; j < 2; j++) {
            float x_j = (j == 0) ? x.x : x.y;
            grad_A(i, j) = grad_y_i * x_j;
        }
    }
}

__device__ __forceinline__ void ppisp_mul_bwd(const float3x3& A, const float3& x, const float3& grad_y,
                                              float3x3& grad_A, float3& grad_x) {
    grad_x = ppisp_transpose(A) * grad_y;

#pragma unroll
    for (int i = 0; i < 3; i++) {
        float grad_y_i = (i == 0) ? grad_y.x : (i == 1) ? grad_y.y
                                                        : grad_y.z;
#pragma unroll
        for (int j = 0; j < 3; j++) {
            float x_j = (j == 0) ? x.x : (j == 1) ? x.y
                                                  : x.z;
            grad_A(i, j) = grad_y_i * x_j;
        }
    }
}

__device__ __forceinline__ void ppisp_mul_mat_bwd(const float3x3& A, const float3x3& B, const float3x3& grad_C,
                                                  float3x3& grad_A, float3x3& grad_B) {
    grad_A = grad_C * ppisp_transpose(B);
    grad_B = ppisp_transpose(A) * grad_C;
}

// Parameter transformation gradients
__device__ __forceinline__ float ppisp_bounded_positive_backward(float raw, float /*min_value*/, float grad_output) {
    if (!isfinite(raw) || !isfinite(grad_output) || raw >= PPISP_MAX_SHAPE_RAW)
        return 0.0f;
    return grad_output * ppisp_sigmoid(raw);
}

__device__ __forceinline__ float ppisp_clamped_backward(float raw, float grad_output) {
    if (!isfinite(raw) || !isfinite(grad_output))
        return 0.0f;
    const float sigmoid = ppisp_sigmoid(raw);
    if (sigmoid <= PPISP_CENTER_EPSILON || sigmoid >= 1.0f - PPISP_CENTER_EPSILON)
        return 0.0f;
    return grad_output * sigmoid * (1.0f - sigmoid);
}

// Exposure backward
__device__ __forceinline__ void ppisp_apply_exposure_bwd(const float3& rgb_in, float exposure_param,
                                                         const float3& grad_rgb_out, float3& grad_rgb_in,
                                                         float& grad_exposure_param) {
    const float exposure_factor = exp2f(ppisp_exposure_value(exposure_param));
    const float3 rgb_out = rgb_in * exposure_factor;
    const bool exposure_has_gradient = isfinite(exposure_param) &&
                                       exposure_param > PPISP_MIN_EXPOSURE_EV &&
                                       exposure_param < PPISP_MAX_EXPOSURE_EV;
    grad_exposure_param = exposure_has_gradient
                              ? ppisp_dot(grad_rgb_out, rgb_out) * 0.69314718f
                              : 0.0f;
    if (!isfinite(grad_exposure_param))
        grad_exposure_param = 0.0f;
    grad_rgb_in = grad_rgb_out * exposure_factor;
}

// Vignetting backward
__device__ __forceinline__ void ppisp_apply_vignetting_bwd(const float3& rgb_in,
                                                           const VignettingChannelParams* vignetting_params,
                                                           const float2& pixel_coords, float resolution_x,
                                                           float resolution_y, const float3& grad_rgb_out,
                                                           float3& grad_rgb_in,
                                                           VignettingChannelParams* grad_vignetting_params) {
    float max_res = fmaxf(resolution_x, resolution_y);
    float2 uv = make_float2(__fdividef(pixel_coords.x - resolution_x * 0.5f, max_res),
                            __fdividef(pixel_coords.y - resolution_y * 0.5f, max_res));

    float rgb_arr[3] = {rgb_in.x, rgb_in.y, rgb_in.z};
    float grad_rgb_out_arr[3] = {grad_rgb_out.x, grad_rgb_out.y, grad_rgb_out.z};
    float grad_rgb_in_arr[3] = {0, 0, 0};

#pragma unroll
    for (int i = 0; i < 3; i++) {
        const VignettingChannelParams& params = vignetting_params[i];

        const float cx = ppisp_finite_or_zero(params.cx);
        const float cy = ppisp_finite_or_zero(params.cy);
        const float alpha0 = ppisp_finite_or_zero(params.alpha0);
        const float alpha1 = ppisp_finite_or_zero(params.alpha1);
        const float alpha2 = ppisp_finite_or_zero(params.alpha2);
        float dx = uv.x - cx;
        float dy = uv.y - cy;
        float r2 = __fmaf_rn(dx, dx, dy * dy);
        float r4 = r2 * r2;
        float r6 = r4 * r2;

        float falloff = __fmaf_rn(alpha2, r6, __fmaf_rn(alpha1, r4, __fmaf_rn(alpha0, r2, 1.0f)));
        float falloff_clamped = fmaxf(0.0f, fminf(1.0f, falloff));

        grad_rgb_in_arr[i] = grad_rgb_out_arr[i] * falloff_clamped;
        float grad_falloff = grad_rgb_out_arr[i] * rgb_arr[i];

        if (falloff >= 0.0f && falloff <= 1.0f) {
            grad_vignetting_params[i].alpha0 += grad_falloff * r2;
            grad_vignetting_params[i].alpha1 += grad_falloff * r4;
            grad_vignetting_params[i].alpha2 += grad_falloff * r6;

            float grad_r2 = grad_falloff *
                            __fmaf_rn(3.0f * alpha2, r4,
                                      __fmaf_rn(2.0f * alpha1, r2, alpha0));
            grad_vignetting_params[i].cx += -grad_r2 * 2.0f * dx;
            grad_vignetting_params[i].cy += -grad_r2 * 2.0f * dy;
        }
    }

    grad_rgb_in = make_float3(grad_rgb_in_arr[0], grad_rgb_in_arr[1], grad_rgb_in_arr[2]);
}

// Homography backward helpers
__device__ __forceinline__ void ppisp_compute_homography_normalization_bwd(const float3x3& H_unnorm,
                                                                           const float3x3& grad_H_norm,
                                                                           float3x3& grad_H_unnorm) {
    float s = H_unnorm(2, 2);

    if (fabsf(s) > 1.0e-20f) {
        float inv_s = 1.0f / s;
        float inv_s2 = inv_s * inv_s;
        float grad_s = 0.0f;

#pragma unroll
        for (int i = 0; i < 9; i++) {
            grad_H_unnorm.m[i] = grad_H_norm.m[i] * inv_s;
            grad_s += -grad_H_norm.m[i] * H_unnorm.m[i] * inv_s2;
        }
        grad_H_unnorm(2, 2) += grad_s;
    } else {
#pragma unroll
        for (int i = 0; i < 9; i++) {
            grad_H_unnorm.m[i] = grad_H_norm.m[i];
        }
    }
}

__device__ __forceinline__ void ppisp_compute_homography_diagonal_matrix_bwd(const float3x3& grad_D,
                                                                             float3& grad_lambda) {
    grad_lambda.x = grad_D(0, 0);
    grad_lambda.y = grad_D(1, 1);
    grad_lambda.z = grad_D(2, 2);
}

__device__ __forceinline__ void ppisp_compute_homography_skew_matrix_construction_bwd(const float3x3& grad_skew,
                                                                                      float3& grad_t_gray) {
    grad_t_gray.x = 0.0f;
    grad_t_gray.y = 0.0f;
    grad_t_gray.z = 0.0f;

    grad_t_gray.z += -grad_skew(0, 1);
    grad_t_gray.y += grad_skew(0, 2);
    grad_t_gray.z += grad_skew(1, 0);
    grad_t_gray.x += -grad_skew(1, 2);
    grad_t_gray.y += -grad_skew(2, 0);
    grad_t_gray.x += grad_skew(2, 1);
}

__device__ __forceinline__ void ppisp_compute_homography_matrix_T_construction_bwd(const float3x3& grad_T,
                                                                                   float3& grad_t_b, float3& grad_t_r,
                                                                                   float3& grad_t_g) {
    grad_t_b.x = grad_T(0, 0);
    grad_t_b.y = grad_T(1, 0);
    grad_t_b.z = grad_T(2, 0);

    grad_t_r.x = grad_T(0, 1);
    grad_t_r.y = grad_T(1, 1);
    grad_t_r.z = grad_T(2, 1);

    grad_t_g.x = grad_T(0, 2);
    grad_t_g.y = grad_T(1, 2);
    grad_t_g.z = grad_T(2, 2);
}

__device__ __forceinline__ void ppisp_compute_homography_target_point_bwd(const float3& grad_t, float2& grad_offset) {
    grad_offset.x = grad_t.x;
    grad_offset.y = grad_t.y;
}

__device__ __forceinline__ void ppisp_compute_homography_nullspace_computation_bwd(const float3x3& M,
                                                                                   const float3& lambda_v,
                                                                                   const float3x3& grad_M_in,
                                                                                   float3x3& grad_M) {
    float3 r0 = make_float3(M(0, 0), M(0, 1), M(0, 2));
    float3 r1 = make_float3(M(1, 0), M(1, 1), M(1, 2));
    float3 r2 = make_float3(M(2, 0), M(2, 1), M(2, 2));

    float3 lambda_test = ppisp_cross(r0, r1);
    float n2 = ppisp_dot(lambda_test, lambda_test);

    float3 grad_lambda = make_float3(0.0f, 0.0f, 0.0f);
    grad_lambda.x = grad_M_in(0, 0);
    grad_lambda.y = grad_M_in(1, 1);
    grad_lambda.z = grad_M_in(2, 2);

    float3 grad_r0 = make_float3(0.0f, 0.0f, 0.0f);
    float3 grad_r1 = make_float3(0.0f, 0.0f, 0.0f);
    float3 grad_r2 = make_float3(0.0f, 0.0f, 0.0f);

    if (n2 < 1.0e-20f) {
        lambda_test = ppisp_cross(r0, r2);
        n2 = ppisp_dot(lambda_test, lambda_test);
        if (n2 < 1.0e-20f) {
            ppisp_cross_bwd(grad_lambda, r1, r2, grad_r1, grad_r2);
        } else {
            ppisp_cross_bwd(grad_lambda, r0, r2, grad_r0, grad_r2);
        }
    } else {
        ppisp_cross_bwd(grad_lambda, r0, r1, grad_r0, grad_r1);
    }

    grad_M = make_float3x3(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);

    grad_M(0, 0) = grad_r0.x;
    grad_M(0, 1) = grad_r0.y;
    grad_M(0, 2) = grad_r0.z;
    grad_M(1, 0) = grad_r1.x;
    grad_M(1, 1) = grad_r1.y;
    grad_M(1, 2) = grad_r1.z;
    grad_M(2, 0) = grad_r2.x;
    grad_M(2, 1) = grad_r2.y;
    grad_M(2, 2) = grad_r2.z;
}

// Main homography backward
__device__ __forceinline__ void ppisp_compute_homography_bwd(const ColorPPISPParams* color_params,
                                                             const float3x3& grad_H,
                                                             ColorPPISPParams* grad_color_params) {
    const float2 b_lat = ppisp_finite_or_zero(color_params->b);
    const float2 r_lat = ppisp_finite_or_zero(color_params->r);
    const float2 g_lat = ppisp_finite_or_zero(color_params->g);
    const float2 n_lat = ppisp_finite_or_zero(color_params->n);

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
    float3x3 TD = T * D;
    float3x3 H_unnorm = TD * S_inv;

    // Backward pass
    float3x3 grad_H_unnorm = make_float3x3(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    ppisp_compute_homography_normalization_bwd(H_unnorm, grad_H, grad_H_unnorm);

    float3x3 grad_TD, grad_S_inv_unused;
    ppisp_mul_mat_bwd(TD, S_inv, grad_H_unnorm, grad_TD, grad_S_inv_unused);

    float3x3 grad_T, grad_D;
    ppisp_mul_mat_bwd(T, D, grad_TD, grad_T, grad_D);

    float3 grad_lambda;
    ppisp_compute_homography_diagonal_matrix_bwd(grad_D, grad_lambda);

    float3x3 grad_D_for_nullspace =
        make_float3x3(grad_lambda.x, 0.0f, 0.0f, 0.0f, grad_lambda.y, 0.0f, 0.0f, 0.0f, grad_lambda.z);
    float3x3 grad_M;
    ppisp_compute_homography_nullspace_computation_bwd(M, lambda_v, grad_D_for_nullspace, grad_M);

    float3x3 grad_skew, grad_T_from_M;
    ppisp_mul_mat_bwd(skew, T, grad_M, grad_skew, grad_T_from_M);

#pragma unroll
    for (int i = 0; i < 9; i++) {
        grad_T.m[i] += grad_T_from_M.m[i];
    }

    float3 grad_t_gray;
    ppisp_compute_homography_skew_matrix_construction_bwd(grad_skew, grad_t_gray);

    float3 grad_t_b, grad_t_r, grad_t_g;
    ppisp_compute_homography_matrix_T_construction_bwd(grad_T, grad_t_b, grad_t_r, grad_t_g);

    float2 grad_bd, grad_rd, grad_gd, grad_nd;
    ppisp_compute_homography_target_point_bwd(grad_t_b, grad_bd);
    ppisp_compute_homography_target_point_bwd(grad_t_r, grad_rd);
    ppisp_compute_homography_target_point_bwd(grad_t_g, grad_gd);
    ppisp_compute_homography_target_point_bwd(grad_t_gray, grad_nd);

    float2x2 dummy_grad_zca;
    float2 grad_b_lat, grad_r_lat, grad_g_lat, grad_n_lat;

    ppisp_mul_bwd(zca_b, b_lat, grad_bd, dummy_grad_zca, grad_b_lat);
    ppisp_mul_bwd(zca_r, r_lat, grad_rd, dummy_grad_zca, grad_r_lat);
    ppisp_mul_bwd(zca_g, g_lat, grad_gd, dummy_grad_zca, grad_g_lat);
    ppisp_mul_bwd(zca_n, n_lat, grad_nd, dummy_grad_zca, grad_n_lat);

    grad_color_params->b.x += grad_b_lat.x;
    grad_color_params->b.y += grad_b_lat.y;
    grad_color_params->r.x += grad_r_lat.x;
    grad_color_params->r.y += grad_r_lat.y;
    grad_color_params->g.x += grad_g_lat.x;
    grad_color_params->g.y += grad_g_lat.y;
    grad_color_params->n.x += grad_n_lat.x;
    grad_color_params->n.y += grad_n_lat.y;
}

// Color correction backward
__device__ __forceinline__ void ppisp_apply_color_correction_bwd(const float3& rgb_in,
                                                                 const ColorPPISPParams* color_params,
                                                                 const float3& grad_rgb_out, float3& grad_rgb_in,
                                                                 ColorPPISPParams* grad_color_params) {
    float3x3 H = ppisp_compute_homography(color_params);

    float intensity = rgb_in.x + rgb_in.y + rgb_in.z;
    float3 rgi_in = make_float3(rgb_in.x, rgb_in.y, intensity);
    float3 rgi_out = H * rgi_in;

    float norm_factor = __fdividef(intensity, rgi_out.z + 1.0e-5f);

    float3 grad_rgi_out_norm;
    grad_rgi_out_norm.x = grad_rgb_out.x - grad_rgb_out.z;
    grad_rgi_out_norm.y = grad_rgb_out.y - grad_rgb_out.z;
    grad_rgi_out_norm.z = grad_rgb_out.z;

    float3 grad_rgi_out = grad_rgi_out_norm * norm_factor;
    float grad_norm_factor = ppisp_dot(grad_rgi_out_norm, rgi_out);
    float grad_rgi_out_z_norm = -grad_norm_factor * norm_factor / (rgi_out.z + 1.0e-5f);
    grad_rgi_out.z = grad_rgi_out.z + grad_rgi_out_z_norm;

    float3x3 grad_H;
    float3 grad_rgi_in;
    ppisp_mul_bwd(H, rgi_in, grad_rgi_out, grad_H, grad_rgi_in);

    grad_rgb_in.x = grad_rgi_in.x + grad_rgi_in.z;
    grad_rgb_in.y = grad_rgi_in.y + grad_rgi_in.z;
    grad_rgb_in.z = grad_rgi_in.z;

    float grad_intensity = 0.0f;
    if (intensity > 1e-8f) {
        grad_intensity = grad_norm_factor * norm_factor / intensity;
    }

    grad_rgb_in.x = grad_rgb_in.x + grad_intensity;
    grad_rgb_in.y = grad_rgb_in.y + grad_intensity;
    grad_rgb_in.z = grad_rgb_in.z + grad_intensity;

    ppisp_compute_homography_bwd(color_params, grad_H, grad_color_params);
}

// CRF backward
__device__ __forceinline__ void ppisp_apply_crf_bwd(const float3& rgb_in, const CRFPPISPChannelParams* crf_params,
                                                    const float3& grad_rgb_out, float3& grad_rgb_in,
                                                    CRFPPISPChannelParams* grad_crf_params) {
    float3 rgb_clamped = ppisp_clamp(rgb_in, 0.0f, 1.0f);
    float rgb_arr[3] = {rgb_clamped.x, rgb_clamped.y, rgb_clamped.z};
    float grad_out_arr[3] = {grad_rgb_out.x, grad_rgb_out.y, grad_rgb_out.z};
    float grad_in_arr[3] = {0, 0, 0};

#pragma unroll
    for (int i = 0; i < 3; i++) {
        const CRFPPISPChannelParams& params = crf_params[i];

        float toe = ppisp_bounded_positive_forward(params.toe, 0.3f);
        float shoulder = ppisp_bounded_positive_forward(params.shoulder, 0.3f);
        float gamma = ppisp_bounded_positive_forward(params.gamma, 0.1f);
        float center = ppisp_clamped_forward(params.center);

        float lerp_val = __fmaf_rn(shoulder - toe, center, toe);
        float a = __fdividef(shoulder * center, lerp_val);
        float b = 1.0f - a;

        float x = rgb_arr[i];
        float y;

        if (x <= center) {
            y = a * __powf(__fdividef(x, center), toe);
        } else {
            y = 1.0f - b * __powf(__fdividef(1.0f - x, 1.0f - center), shoulder);
        }

        float output = __powf(fmaxf(0.0f, y), gamma);
        float y_clamped = fmaxf(0.0f, y);

        float grad_y = 0.0f;
        if (y_clamped > 0.0f) {
            grad_y = grad_out_arr[i] * gamma * __powf(y_clamped, gamma - 1.0f);
        }

        float grad_x = 0.0f;
        if (x <= center && center > 0.0f) {
            float base = __fdividef(x, center);
            if (base > 0.0f) {
                grad_x = grad_y * a * toe * __powf(base, toe - 1.0f) / center;
            }
        } else if (x > center && center < 1.0f) {
            float base = __fdividef(1.0f - x, 1.0f - center);
            if (base > 0.0f) {
                grad_x = grad_y * b * shoulder * __powf(base, shoulder - 1.0f) / (1.0f - center);
            }
        }

        grad_in_arr[i] = grad_x;

        float grad_toe = 0.0f;
        float grad_shoulder = 0.0f;
        float grad_gamma = 0.0f;
        float grad_center = 0.0f;

        if (y_clamped > 0.0f) {
            grad_gamma = grad_out_arr[i] * output * __logf(y_clamped + 1e-8f);
        }

        float grad_a = 0.0f;
        float grad_b = 0.0f;

        if (x <= center && center > 0.0f) {
            float base = __fdividef(x, center);
            if (base > 0.0f) {
                float powered = __powf(base, toe);
                grad_a += grad_y * powered;
                grad_toe += grad_y * a * powered * __logf(base + 1e-8f);
                float grad_base = grad_y * a * toe * __powf(base, toe - 1.0f);
                grad_center += grad_base * (-x / (center * center));
            }
        } else if (x > center && center < 1.0f) {
            float base = __fdividef(1.0f - x, 1.0f - center);
            if (base > 0.0f) {
                float powered = __powf(base, shoulder);
                grad_b += -grad_y * powered;
                grad_shoulder += -grad_y * b * powered * __logf(base + 1e-8f);
                float grad_base = grad_y * (-b * shoulder * __powf(base, shoulder - 1.0f));
                float dbase_dcenter = (1.0f - x) / ((1.0f - center) * (1.0f - center));
                grad_center += grad_base * dbase_dcenter;
            }
        }

        grad_a += -grad_b;

        float grad_lerp_val = 0.0f;
        if (fabsf(lerp_val) > 1e-8f) {
            float a_over_lerp = __fdividef(shoulder * center, lerp_val);
            grad_shoulder += grad_a * center / lerp_val;
            grad_center += grad_a * shoulder / lerp_val;
            grad_lerp_val += -grad_a * a_over_lerp / lerp_val;
        }

        grad_shoulder += grad_lerp_val * center;
        grad_toe += grad_lerp_val * (1.0f - center);
        grad_center += grad_lerp_val * (shoulder - toe);

        grad_crf_params[i].toe += ppisp_bounded_positive_backward(params.toe, 0.3f, grad_toe);
        grad_crf_params[i].shoulder += ppisp_bounded_positive_backward(params.shoulder, 0.3f, grad_shoulder);
        grad_crf_params[i].gamma += ppisp_bounded_positive_backward(params.gamma, 0.1f, grad_gamma);
        grad_crf_params[i].center += ppisp_clamped_backward(params.center, grad_center);
    }

    grad_rgb_in = make_float3(grad_in_arr[0], grad_in_arr[1], grad_in_arr[2]);
}
