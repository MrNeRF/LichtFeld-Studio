// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "config.slang"

static const float eps = 1e-7;

struct rectangle {
    uint16_t min_x;
    uint16_t min_y;
    uint16_t max_x;
    uint16_t max_y;
};

enum CameraModel {
    PINHOLE = 0,
    OPENCV = 1,
    FISHEYE = 2,
};

[ForceInline]
[Differentiable]
float read_t1_float(uint32_t idx, StructuredBuffer<float> t1) {
    return t1[idx];
}

[ForceInline]
[Differentiable]
float2 read_t2_float2(uint32_t idx, StructuredBuffer<float> t2) {
    return float2(t2[2*idx+0], t2[2*idx+1]);
}

[ForceInline]
[Differentiable]
float3 read_t3_float3(uint32_t idx, StructuredBuffer<float> t3) {
    return float3(t3[3*idx+0], t3[3*idx+1], t3[3*idx+2]);
}

[ForceInline]
[Differentiable]
half3 read_t3_float3(uint32_t idx, StructuredBuffer<half> t3) {
    return half3(t3[3*idx+0], t3[3*idx+1], t3[3*idx+2]);
}

[ForceInline]
[Differentiable]
float4 read_t4_float4(uint32_t idx, StructuredBuffer<float> t4) {
    return float4(t4[4*idx+0], t4[4*idx+1], t4[4*idx+2], t4[4*idx+3]);
}

[ForceInline]
void write_t1_float(float val, uint32_t idx, RWStructuredBuffer<float> t1) {
    t1[idx] = val;
}

[ForceInline]
void write_t2_float2(float2 val, uint32_t idx, RWStructuredBuffer<float> t2) {
    t2[2*idx+0] = val.x;
    t2[2*idx+1] = val.y;
}

[ForceInline]
void write_t3_float3(float3 val, uint32_t idx, RWStructuredBuffer<float> t3) {
    t3[3*idx+0] = val.x;
    t3[3*idx+1] = val.y;
    t3[3*idx+2] = val.z;
}

[ForceInline]
void write_t3_float3(half3 val, uint32_t idx, RWStructuredBuffer<half> t3) {
    t3[3*idx+0] = val.x;
    t3[3*idx+1] = val.y;
    t3[3*idx+2] = val.z;
}

[ForceInline]
void write_t4_float4(float4 val, uint32_t idx, RWStructuredBuffer<float> t4) {
    t4[4*idx+0] = val.x;
    t4[4*idx+1] = val.y;
    t4[4*idx+2] = val.z;
    t4[4*idx+3] = val.w;
}

[ForceInline]
[Differentiable]
float3x3 quat_to_rotmat(float4 q) {
    float r = q[0], x = q[1], y = q[2], z = q[3];
    return float3x3(
        1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y),
        2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
        2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y));
}


struct Camera : IDifferentiable
{
    float3x4 world_view_transform;
    float fx, fy, cx, cy;
    float4 dist_coeffs;
    int H, W, model;
}

[ForceInline]
Camera load_camera(
    float4x4 world_view_transform_t,
    float fx, float fy, float cx, float cy, int model, float4 dist_coeffs,
    uint H, uint W
) {
    return { float3x4(world_view_transform_t), fx, fy, cx, cy, dist_coeffs, H, W, model};
}


[ForceInline]
[Differentiable]
float3 geom_transform_points(float3 point, float3x4 transf_matrix)
{
    return mul(transf_matrix, float4(point, 1.0));
}

[Differentiable]
float3 project_point_pinhole(float3 point, Camera cam) {
    float3 view_point = geom_transform_points(point, cam.world_view_transform);
    float3 proj_point = float3(
        view_point.x / view_point.z,
        view_point.y / view_point.z,
        view_point.z
    );
    return {
        cam.fx * proj_point.x + cam.cx,
        cam.fy * proj_point.y + cam.cy,
        proj_point.z
    };
}

[Differentiable]
float3x3 compute_jacobian_pinhole(float3 xyz_ws, Camera cam) {
    float tan_fovx = 0.5f * cam.W / cam.fx;
    float tan_fovy = 0.5f * cam.H / cam.fy;

    float3 t = geom_transform_points(xyz_ws, cam.world_view_transform);

    const float limx_pos = (cam.W-cam.cx)/cam.fx + 0.3*tan_fovx;
    const float limx_neg = cam.cx/cam.fx + 0.3*tan_fovx;
    const float limy_pos = (cam.H-cam.cy)/cam.fy + 0.3*tan_fovy;
    const float limy_neg = cam.cy/cam.fy + 0.3*tan_fovy;
    const float txtz = t.x / t.z;
    const float tytz = t.y / t.z;
    t.x = min(limx_pos, max(-limx_neg, txtz)) * t.z;
    t.y = min(limy_pos, max(-limy_neg, tytz)) * t.z;

    float3x3 J = float3x3(cam.fx / t.z, 0.0, -(cam.fx * t.x) / (t.z * t.z),
                          0.0, cam.fy / t.z, -(cam.fy * t.y) / (t.z * t.z),
                          0.0, 0.0, 0.0);

    return J;
}

[ForceInline]
bool is_inside_opencv(float x, float y, float z, float4 dist_coeffs) {
    float k1 = dist_coeffs.x,
        k2 = dist_coeffs.y,
        p1 = dist_coeffs.z,
        p2 = dist_coeffs.w;
    float u = x/z, v = y/z;
    
    float dxdu = 5*k2*u*u*u*u + (6*k2*v*v + 3*k1)*u*u + 6*p2*u + k2*v*v*v*v + k1*v*v + 2*p1*v + 1;
    float dxdv = 2 * ( 2*k2*u*v*v*v + (2*k2*u*u*u + k1*u + p2)*v + p1*u );
    float dydu = 2 * ( 2*k2*v*u*u*u + (2*k2*v*v*v + k1*v + p1)*u + p2*v );
    float dydv = 5*k2*v*v*v*v + (6*k2*u*u + 3*k1)*v*v + 6*p1*v + k2*u*u*u*u + k1*u*u + 2*p2*u + 1;

    float detJ = dxdu * dydv - dxdv * dydu;
    return fmin(fmin(detJ, z), fmin(dxdu, dydv)) > 0.0;
}

[ForceInline]
bool is_inside_fisheye(float x, float y, float z, float4 dist_coeffs) {
    float k1 = dist_coeffs.x,
        k2 = dist_coeffs.y,
        k3 = dist_coeffs.z,
        k4 = dist_coeffs.w;

    float eps = 0.0000001f;
    float xy_len = length(float2(x, y)) + eps;
    float theta = atan2(xy_len, z + eps);
    float r2 = theta * theta;
    
    return 1.0 + r2*(3.0*k1 + r2*(5.0+k2 + r2*(7.0*k3 + 9.0*k4))) > 0.0;
}

[Differentiable]
[ForceInline]
float2 distort_point_kkpp(float2 uv, no_diff float4 dist_coeffs) {
    float k1 = dist_coeffs.x,
        k2 = dist_coeffs.y,
        p1 = dist_coeffs.z,
        p2 = dist_coeffs.w;
    float u = uv.x, v = uv.y;
    float r2 = dot(uv, uv);
    return uv * (1.0f + r2*(k1 + r2*k2)) + float2(
        2.0 * p1 * u*v + p2 * (r2+2.0*u*u),
        2.0 * p2 * u*v + p1 * (r2+2.0*v*v)
    );
}

[ForceInline]
float2x2 distort_point_kkpp_jac(float2 uv, no_diff float4 dist_coeffs) {

    DifferentialPair<float2> dx = fwd_diff(distort_point_kkpp)
        (diffPair(uv, float2(1, 0)), dist_coeffs);
    DifferentialPair<float2> dy = fwd_diff(distort_point_kkpp)
        (diffPair(uv, float2(0, 1)), dist_coeffs);

    return float2x2(
        dx.d.x, dy.d.x,
        dx.d.y, dy.d.y);
}

float2 undistort_point_kkpp(float2 uv, no_diff float4 dist_coeffs) {
    // TODO: can probably optimize to 1D for radial only case
    float2 uv_undist = uv;
    for (int i = 0; i < 12; i++) {
        float2 uv_dist = distort_point_kkpp(uv_undist, dist_coeffs);
        float2x2 jac = distort_point_kkpp_jac(uv_undist, dist_coeffs);
        float2 delta = solve2x2(jac, uv_dist-uv);
        uv_undist -= delta;
    }
    return uv_undist;
}

[Differentiable]
[ForceInline]
float2 _project_point_opencv(float x, float y, float z, no_diff float4 dist_coeffs) {
    return distort_point_kkpp(float2(x/z, y/z), dist_coeffs);
}

[ForceInline]
float2 solve2x2(float2x2 A, float2 b) {
    return float2(
        b.x*A[1][1] - b.y*A[0][1],
        -b.x*A[1][0] + b.y*A[0][0]
    ) / (A[0][0]*A[1][1] - A[0][1]*A[1][0]);
}

[Differentiable]
[ForceInline]
float2 distort_point_kkkk(float2 uv, no_diff float4 dist_coeffs) {
    float k1 = dist_coeffs.x,
        k2 = dist_coeffs.y,
        k3 = dist_coeffs.z,
        k4 = dist_coeffs.w;
    float r2 = dot(uv, uv);
    return uv * (1.0f + r2*(k1 + r2*(k2 + r2*(k3 + r2*k4))));
}

[ForceInline]
float2x2 distort_point_kkkk_jac(float2 uv, no_diff float4 dist_coeffs) {

    DifferentialPair<float2> dx = fwd_diff(distort_point_kkkk)
        (diffPair(uv, float2(1, 0)), dist_coeffs);
    DifferentialPair<float2> dy = fwd_diff(distort_point_kkkk)
        (diffPair(uv, float2(0, 1)), dist_coeffs);

    return float2x2(
        dx.d.x, dy.d.x,
        dx.d.y, dy.d.y);
}

float2 undistort_point_kkkk(float2 uv, no_diff float4 dist_coeffs) {
    // TODO: can probably optimize to 1D for radial only case
    float2 uv_undist = uv;
    for (int i = 0; i < 12; i++) {
        float2 uv_dist = distort_point_kkkk(uv_undist, dist_coeffs);
        float2x2 jac = distort_point_kkkk_jac(uv_undist, dist_coeffs);
        float2 delta = solve2x2(jac, uv_dist-uv);
        uv_undist -= delta;
    }
    return uv_undist;
}

[Differentiable]
[ForceInline]
float2 _project_point_fisheye(float x, float y, float z, no_diff float4 dist_coeffs) {
    float eps = 0.0000001f;
    float xy_len = length(float2(x, y)) + eps;
    float theta = atan2(xy_len, z + eps);

    float2 uv = float2(x, y) * (theta / xy_len);
    return distort_point_kkkk(uv, dist_coeffs);
}

[Differentiable]
float3 project_point_opencv(float3 point, Camera cam) {
    float3 view_point = geom_transform_points(point, cam.world_view_transform);

    if (!is_inside_opencv(view_point.x, view_point.y, view_point.z, cam.dist_coeffs))
        return {0, 0, -1};

    float2 uv = _project_point_opencv(view_point.x, view_point.y, view_point.z, cam.dist_coeffs);
    return {
        cam.fx * uv.x + cam.cx,
        cam.fy * uv.y + cam.cy,
        view_point.z  // TODO: support >180deg fisheye
    };
}

[Differentiable]
float3 project_point_fisheye(float3 point, Camera cam) {
    float3 view_point = geom_transform_points(point, cam.world_view_transform);

    if (!is_inside_fisheye(view_point.x, view_point.y, view_point.z, cam.dist_coeffs))
        return {0, 0, -1};

    float2 uv = _project_point_fisheye(view_point.x, view_point.y, view_point.z, cam.dist_coeffs);
    return {
        cam.fx * uv.x + cam.cx,
        cam.fy * uv.y + cam.cy,
        view_point.z  // TODO: support >180deg fisheye
    };
}

[Differentiable]
float3x3 compute_jacobian_opencv(float3 xyz_ws, Camera cam) {
    float3 view_point = geom_transform_points(xyz_ws, cam.world_view_transform);
    float x = view_point.x, y = view_point.y, z = view_point.z;
    float fx = cam.fx, fy = cam.fy, cx = cam.cx, cy = cam.cy;

    DifferentialPair<float2> dx = fwd_diff(_project_point_opencv)
        (diffPair(x, 1.0), diffPair(y, 0.0), diffPair(z, 0.0), cam.dist_coeffs);
    DifferentialPair<float2> dy = fwd_diff(_project_point_opencv)
        (diffPair(x, 0.0), diffPair(y, 1.0), diffPair(z, 0.0), cam.dist_coeffs);
    DifferentialPair<float2> dz = fwd_diff(_project_point_opencv)
        (diffPair(x, 0.0), diffPair(y, 0.0), diffPair(z, 1.0), cam.dist_coeffs);

    return float3x3(
        fx * dx.d.x, fx * dy.d.x, fx * dz.d.x,
        fy * dx.d.y, fy * dy.d.y, fy * dz.d.y,
        0.0, 0.0, 0.0);
}

[Differentiable]
float3x3 compute_jacobian_fisheye(float3 xyz_ws, Camera cam) {
    float3 view_point = geom_transform_points(xyz_ws, cam.world_view_transform);
    float x = view_point.x, y = view_point.y, z = view_point.z;
    float fx = cam.fx, fy = cam.fy, cx = cam.cx, cy = cam.cy;

    DifferentialPair<float2> dx = fwd_diff(_project_point_fisheye)
        (diffPair(x, 1.0), diffPair(y, 0.0), diffPair(z, 0.0), cam.dist_coeffs);
    DifferentialPair<float2> dy = fwd_diff(_project_point_fisheye)
        (diffPair(x, 0.0), diffPair(y, 1.0), diffPair(z, 0.0), cam.dist_coeffs);
    DifferentialPair<float2> dz = fwd_diff(_project_point_fisheye)
        (diffPair(x, 0.0), diffPair(y, 0.0), diffPair(z, 1.0), cam.dist_coeffs);

    return float3x3(
        fx * dx.d.x, fx * dy.d.x, fx * dz.d.x,
        fy * dx.d.y, fy * dy.d.y, fy * dz.d.y,
        0.0, 0.0, 0.0);
}


[Differentiable]
[ForceInline]
float3 project_point(float3 point, Camera cam) {
    if (cam.model == int(CameraModel::FISHEYE))
        return project_point_fisheye(point, cam);
    else if (cam.model == int(CameraModel::OPENCV))
        return project_point_opencv(point, cam);
    else
        return project_point_pinhole(point, cam);
}

[Differentiable]
float3x3 compute_jacobian(float3 xyz_ws, Camera cam) {
    if (cam.model == int(CameraModel::FISHEYE))
        return compute_jacobian_fisheye(xyz_ws, cam);
    else if (cam.model == int(CameraModel::OPENCV))
        // return compute_jacobian_opencv(xyz_ws, cam);
        return compute_jacobian_pinhole(xyz_ws, cam);  // TODO
    else
        return compute_jacobian_pinhole(xyz_ws, cam);
}

[ForceInline]
float2 undistort_fisheye(float2 uv) {
    float theta = length(uv);
    float r = tan(theta);
    const float eps = 1e-7f;
    return uv * (r + eps) / (theta + eps);
}

float2 undistort_fisheye(float2 uv, float4 dist_coeffs) {
    uv = undistort_point_kkkk(uv, dist_coeffs);
    float theta = length(uv);
    float r = tan(theta);
    const float eps = 1e-7f;
    return uv * (r + eps) / (theta + eps);
}


// returns (xx, xy, yy, compensation)
[Differentiable]
float4 covariance_3d_to_2d(Camera cam, float3 xyz_ws, float3x3 cov_ws) {
    float3x3 R = (float3x3)cam.world_view_transform;
    float3x3 J = compute_jacobian(xyz_ws, cam);
    float3x3 cov_vs = mul(J, mul(R, mul(cov_ws, mul(transpose(R), transpose(J)))));

    float4 result = float4(cov_vs[0][0], cov_vs[0][1], cov_vs[1][1], 1.0);

    #if ANTIALIASE_MODE == 0
        result.x += AA_EPS2D;
        result.z += AA_EPS2D;
    #elif ANTIALIASE_MODE == 1
        float old_det = compute_det(result.xyz);
        result.x += AA_EPS2D;
        result.z += AA_EPS2D;
        float new_det = compute_det(result.xyz);
        result.w = sqrt(max(0.f, old_det / new_det));
    #endif

    return result;
}


struct Splat_2D_Vertex : IDifferentiable
{
    float3 xyz_vs;
    float3 rgb;
    float3 cov_vs;
    float4 inv_cov_vs_opac;
    float det;
};

float splat_radius(float3 cov_vs, float det) {
    float mid = 0.5f * (cov_vs.x + cov_vs.z);
    float eigen_val_1 = mid + sqrt(max(0.1f, mid * mid - det));
    float eigen_val_2 = mid - sqrt(max(0.1f, mid * mid - det));
    float radius = ceil(3.f * sqrt(max(eigen_val_1, eigen_val_2)));

    return radius;
}

[ForceInline]
[Differentiable]
float compute_det(float3 M) {
    return M.x * M.z - M.y * M.y;
}

[ForceInline]
[Differentiable]
float3 compute_inverse(float3 M) {
    return float3(M.z, -M.y, M.x) * (1.0 / compute_det(M));
}


struct Splat_2D_AlphaBlend : IDifferentiable
{
    float4 inv_cov_vs_opacity;
    float2 xy_vs;
    float3 rgb;
};

[ForceInline]
[Differentiable]
Splat_2D_AlphaBlend load_splat_alphablend(
    int32_t g_idx,
    StructuredBuffer<float2> xy_vs,
    StructuredBuffer<float4> inv_cov_vs_opacity,
    StructuredBuffer<float> rgb
) {
    float2 g_xy_vs = xy_vs[g_idx];
    float3 g_rgb = read_t3_float3(g_idx, rgb);
    float4 g_inv_cov_opac = inv_cov_vs_opacity[g_idx];

    return { g_inv_cov_opac, g_xy_vs, g_rgb };
}

[ForceInline]
[Differentiable]
float3x3 rotation_scale_to_iscl_rot(float4 g_rotation, float3 g_scale) {
    float3x3 R = quat_to_rotmat(normalize(g_rotation));
    float3x3 S = float3x3(
        1.0f/g_scale.x, 0.f, 0.f,
        0.f, 1.0f/g_scale.y, 0.f,
        0.f, 0.f, 1.0f/g_scale.z
    );
    return mul(S, transpose(R));
}


[Differentiable]
[ForceInline]
[ForwardDerivative(normalCdfDiffFwd)]
[BackwardDerivative(normalCdfDiffBwd)]
float normalCdf(float x) {
    // return 0.5+0.5*tanh(x*(0.8+0.035*x*x));
    return 0.5+0.5*tanh(clamp(x*(0.8+0.035*x*x),-7.0,7.0));
}

[ForceInline]
DifferentialPair<float> normalCdfDiffFwd(DifferentialPair<float> vx) {
    float x = vx.p;
    float y = 0.5+0.5*tanh(x*(0.8+0.035*x*x));
    float dydx = sqrt(0.5/M_PI) * exp(-0.5*x*x);
    return DifferentialPair<float>(y, dydx*vx.d);
}

[ForceInline]
void normalCdfDiffBwd(inout DifferentialPair<float> vx, float.Differential v) {
    float x = vx.p;
    float dx = sqrt(0.5/M_PI) * exp(-0.5*x*x);
    vx = DifferentialPair<float>(x, v * dx);
}

[Differentiable]
[ForceInline]
[ForwardDerivative(erfDiffFwd)]
[BackwardDerivative(erfDiffBwd)]
float erf(float x) {
    float t = abs(x);
    float d = (((0.078108*t+0.000972)*t+0.230389)*t+0.278393)*t+1.0;
    return sign(x) * (1.0 - 1.0/(d*d*d*d));
}

[ForceInline]
DifferentialPair<float> erfDiffFwd(DifferentialPair<float> vx) {
    float x = vx.p;
    float y = erf(x);
    float dydx = sqrt(4.0/M_PI) * exp(-x*x);
    return DifferentialPair<float>(y, dydx*vx.d);
}

[ForceInline]
void erfDiffBwd(inout DifferentialPair<float> vx, float.Differential v) {
    float x = vx.p;
    float dx = sqrt(4.0/M_PI) * exp(-x*x);
    vx = DifferentialPair<float>(x, v * dx);
}

[ForceInline]
[Differentiable]
[BackwardDerivative(clipOpacityDiffBwd)]
float clipOpacity(float x) {
    return min(ALPHA_BLEND_MAX_OPACITY, x);
}

[ForceInline]
void clipOpacityDiffBwd(inout DifferentialPair<float> vx, float.Differential v) {
    float x = vx.p;
    float dx = x < ALPHA_BLEND_MAX_OPACITY ? v : 0.0;
    vx = DifferentialPair<float>(x, dx);
}

[ForceInline]
half clipOpacity(half x) {
    return min(half(ALPHA_BLEND_MAX_OPACITY), x);
}

[ForceInline]
[Differentiable]
float4 evaluate_splat(Splat_2D_AlphaBlend g, 
                      no_diff float2 pix_coord)
{

    float2 g_xy = g.xy_vs;
    float3 g_rgb = g.rgb;
    float3 g_inv_cov_vs = g.inv_cov_vs_opacity.xyz;
    float g_opacity = g.inv_cov_vs_opacity.w;

    float2 d = pix_coord - g_xy;

    // evaluate at center
    float power = -0.5f * (g_inv_cov_vs.x * d.x*d.x + g_inv_cov_vs.z * d.y*d.y + 2.0*g_inv_cov_vs.y * d.x*d.y);
    float val = exp(power);

    float alpha = float(clipOpacity(g_opacity * val));
    float3 premult_rgb = g_rgb * alpha;
    
    return float4(premult_rgb, alpha);
}

[ForceInline]
[Differentiable]
float evaluate_splat_alpha(
    float2 g_xy,
    float g_opacity,
    float3 g_inv_cov_vs,
    no_diff float2 pix_coord)
{

    float2 d = pix_coord - g_xy;

    float power = -0.5f * (g_inv_cov_vs.x * d.x*d.x + g_inv_cov_vs.z * d.y*d.y + 2.0*g_inv_cov_vs.y * d.x*d.y);

    float alpha = clipOpacity(g_opacity * exp(power));
    
    return float(alpha);
}

[Differentiable]
float3x3 get_covariance_from_quat_scales(float4 q, float3 s) {

    float3x3 rotation_matrix = quat_to_rotmat(q);

    float3x3 scales_matrix = float3x3(s[0], 0, 0,
                                      0, s[1], 0,
                                      0, 0, s[2]);

    float3x3 L = mul(rotation_matrix, scales_matrix);

    return mul(L, transpose(L));
}


float extend_from_opacity(float opac) {
    // return min(3.329f, sqrt(2.0*log(opac/ALPHA_THRESHOLD)));
    // return sqrt(2.0*log(min(opac,1.0f)/ALPHA_THRESHOLD));
    return sqrt(2.0*log(opac/ALPHA_THRESHOLD));
}


[ForceInline]
bool ellipse_box_overlap_test_fast(
    float3 inv_cov, float x0, float x1, float y0, float y1
) {
    // test if ellipse and box overlap
    // ellipse: centered at origin, defined by inv_cov and r=1
    // box: axis-aligned, defined by x0, x1, y0, y1

    float det = inv_cov.x*inv_cov.z - inv_cov.y*inv_cov.y;
    float2 wh = sqrt(inv_cov.zx / det);

    bool x_overlap = max(x0, -wh.x) < min(x1, wh.x);
    bool y_overlap = max(y0, -wh.y) < min(y1, wh.y);
    return x_overlap && y_overlap;
}

[ForceInline]
bool ellipse_box_overlap_test(
    float3 inv_cov, float x0, float x1, float y0, float y1
) {
    // test if ellipse and box overlap
    // ellipse: centered at origin, defined by inv_cov and r=1
    // box: axis-aligned, defined by x0, x1, y0, y1

    // basic idea:
    // if overlap, then box contains center of ellipse, or box boundary intersects ellipse boundary
    // center is easy to check
    // ellipse is a quadratic form, value on each edge is a parabola; check values at parabola vertices

    // to make branchless:
    // x > 0 => -x < 0
    // x < a && y < a => max(x,y) < a
    // x < a || y < a => min(x,y) < a
    // x < a && y < 0 => max(x-a, y) < 0

    float a = inv_cov.x, b = inv_cov.y, c = inv_cov.z;

    // parabola vertex on 4 edges
    float wx = -b/c, wy = -b/a;
    float u0 = clamp(x0 * wx, y0, y1);
    float u1 = clamp(x1 * wx, y0, y1);
    float v0 = clamp(y0 * wy, x0, x1);
    float v1 = clamp(y1 * wy, x0, x1);
    b *= 2.0f;
    float mx = min(
        a*x0*x0 + b*x0*u0 + c*u0*u0,
        a*x1*x1 + b*x1*u1 + c*u1*u1
    );
    float my = min(
        a*v0*v0 + b*v0*y0 + c*y0*y0,
        a*v1*v1 + b*v1*y1 + c*y1*y1
    );

    // center
    float mc = max(max(x0, -x1), max(y0, -y1));
    return min(mc, min(mx, my)-1.0f) < 0.f;
}


float2 ellipse_range_bound(
    float3 inv_cov, float y0, float y1
) {
    // find x_min and x_max of an ellipse clipped to y0 < y <= y1 
    // ellipse: centered at origin, defined by inv_cov and r=1

    float a = inv_cov.x, b = inv_cov.y, c = inv_cov.z;
    float ym = -b/c * sqrt(c/(a*c-b*b));  // can be pre-computed

    float v0 = clamp(-ym, y0, y1);
    float v1 = clamp(ym, y0, y1);

    float bv0 = -b*v0, bv1 = -b*v1;

    float inv_a = 1.0f / a;
    float x0 = inv_a * (bv0 - sqrt(bv0*bv0 - a*(c*v0*v0-1.f)));
    float x1 = inv_a * (bv1 + sqrt(bv1*bv1 - a*(c*v1*v1-1.f)));

    return float2(x0, x1);

    // https://www.desmos.com/calculator/sjdw2xmohr
}


int count_ellipse_grid_overlaps(
    float2 pixelspace_xy,
    float3 inv_cov,
    rectangle grid
) {
    // count the number grid cells that overlap with an ellipse

    int n_tiles = 0;

    if (grid.max_y-grid.min_y <= grid.max_x-grid.min_x) {
        for (uint16_t y = grid.min_y; y < grid.max_y; y++) {
            float y0 = y * TILE_HEIGHT - pixelspace_xy.y;
            float y1 = y0 + TILE_HEIGHT;
            float2 bound = ellipse_range_bound(inv_cov, y0, y1);
            int x0 = int(floor((bound.x + pixelspace_xy.x) / TILE_WIDTH));
            int x1 = int(ceil((bound.y + pixelspace_xy.x) / TILE_WIDTH));
            x0 = clamp(x0, int(grid.min_x), int(grid.max_x));
            x1 = clamp(x1, int(grid.min_x), int(grid.max_x));
            n_tiles += x1-x0;
        }
    } else {
        inv_cov = inv_cov.zyx;
        for (uint16_t x = grid.min_x; x < grid.max_x; x++) {
            float x0 = x * TILE_WIDTH - pixelspace_xy.x;
            float x1 = x0 + TILE_WIDTH;
            float2 bound = ellipse_range_bound(inv_cov, x0, x1);
            int y0 = int(floor((bound.x + pixelspace_xy.y) / TILE_HEIGHT));
            int y1 = int(ceil((bound.y + pixelspace_xy.y) / TILE_HEIGHT));
            y0 = clamp(y0, int(grid.min_y), int(grid.max_y));
            y1 = clamp(y1, int(grid.min_y), int(grid.max_y));
            n_tiles += y1-y0;
        }
    }

    return n_tiles;
}




struct Uniforms {
    uint image_height;
    uint image_width;
    uint grid_height;
    uint grid_width;
    uint num_splats;
    uint active_sh;
    uint step;
    uint camera_model;
    float fx;
    float fy;
    float cx;
    float cy;
    float4 dist_coeffs;
    float4x4 world_view_transform;
}
