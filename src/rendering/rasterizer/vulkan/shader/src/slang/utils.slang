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
    ORTHO = 1,
    FISHEYE = 2,
    EQUIRECTANGULAR = 3,
    OPENCV = 5,
};

static const uint CAMERA_MODEL_MASK = 0xffu;
static const uint PROJECTION_MODE_SHIFT = 8u;
static const uint PROJECTION_MODE_GUT = 1u;

[ForceInline]
[Differentiable]
float sigmoid_activation(float x) {
    return 1.0f / (1.0f + exp(-x));
}

[ForceInline]
uint base_camera_model(uint packed_camera_model) {
    return packed_camera_model & CAMERA_MODEL_MASK;
}

[ForceInline]
bool is_gut_projection_mode(uint packed_camera_model) {
    return ((packed_camera_model >> PROJECTION_MODE_SHIFT) & CAMERA_MODEL_MASK) == PROJECTION_MODE_GUT;
}

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
float3 project_point_ortho(float3 point, Camera cam) {
    float3 view_point = geom_transform_points(point, cam.world_view_transform);
    return {
        cam.fx * view_point.x + cam.cx,
        cam.fy * view_point.y + cam.cy,
        view_point.z
    };
}

[Differentiable]
float3 project_point_equirect(float3 point, Camera cam) {
    float3 view_point = geom_transform_points(point, cam.world_view_transform);
    float len = length(view_point);
    if (len <= eps || !isfinite(len))
        return {0, 0, -1};

    float3 dir = view_point / len;
    float azimuth = atan2(dir.x, dir.z);
    float elevation = asin(clamp(dir.y, -1.0f, 1.0f));
    return {
        (azimuth / (2.0f * M_PI) + 0.5f) * float(cam.W),
        (elevation / M_PI + 0.5f) * float(cam.H),
        len
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

[Differentiable]
float3x3 compute_jacobian_ortho(float3 xyz_ws, Camera cam) {
    return float3x3(cam.fx, 0.0, 0.0,
                    0.0, cam.fy, 0.0,
                    0.0, 0.0, 0.0);
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
    else if (cam.model == int(CameraModel::EQUIRECTANGULAR))
        return project_point_equirect(point, cam);
    else if (cam.model == int(CameraModel::OPENCV))
        return project_point_opencv(point, cam);
    else if (cam.model == int(CameraModel::ORTHO))
        return project_point_ortho(point, cam);
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
    else if (cam.model == int(CameraModel::ORTHO))
        return compute_jacobian_ortho(xyz_ws, cam);
    else
        return compute_jacobian_pinhole(xyz_ws, cam);
}

[ForceInline]
bool image_point_in_image_bounds_margin(float2 image_point, Camera cam, float margin_factor) {
    float margin_x = float(cam.W) * margin_factor;
    float margin_y = float(cam.H) * margin_factor;
    return (-margin_x <= image_point.x && image_point.x < float(cam.W) + margin_x) &&
           (-margin_y <= image_point.y && image_point.y < float(cam.H) + margin_y);
}

[ForceInline]
bool is_equirectangular_camera(Camera cam) {
    return cam.model == int(CameraModel::EQUIRECTANGULAR);
}

[ForceInline]
float camera_wrap_period_x(Camera cam) {
    return is_equirectangular_camera(cam) ? float(cam.W) : 0.0f;
}

[ForceInline]
float wrap_image_x(float x, Camera cam) {
    float period = camera_wrap_period_x(cam);
    if (period <= 0.0f)
        return x;
    float wrapped = x - period * floor(x / period);
    return wrapped >= period ? wrapped - period : wrapped;
}

[ForceInline]
void unwrap_image_points_x(inout float2 image_points[7], Camera cam) {
    float period = camera_wrap_period_x(cam);
    if (period <= 0.0f)
        return;

    float inv_period = 1.0f / period;
    float ref_x = image_points[0].x;
    for (uint i = 0u; i < 7u; ++i) {
        float dx = image_points[i].x - ref_x;
        dx -= period * round(dx * inv_period);
        image_points[i].x = ref_x + dx;
    }
}

[ForceInline]
float3 safe_normalize(float3 v) {
    float l2 = dot(v, v);
    return l2 > 0.0f ? v * rsqrt(l2) : v;
}

// Matches CUDA reference: reject near-zero quaternions (would produce NaN
// after normalize) by falling back to the identity rotation. Identity quat
// composed with the (already clamped) scale yields an axis-aligned covariance
// that the downstream extent clamp can bound, instead of NaN-poisoning the
// whole pixel tile.
[ForceInline]
[Differentiable]
float4 safe_quat_normalize(float4 q) {
    float l2 = dot(q, q);
    return l2 > 1e-8f ? q * rsqrt(l2) : float4(1.0f, 0.0f, 0.0f, 0.0f);
}

struct WorldRay
{
    float3 origin;
    float3 direction;
    bool valid;
};

[ForceInline]
WorldRay pixel_to_world_ray(float2 pixel, Camera cam) {
    float2 uv = (pixel - float2(cam.cx, cam.cy)) / float2(cam.fx, cam.fy);
    if (cam.model == int(CameraModel::FISHEYE)) {
        uv = undistort_fisheye(uv, cam.dist_coeffs);
    } else if (cam.model == int(CameraModel::OPENCV)) {
        uv = undistort_point_kkpp(uv, cam.dist_coeffs);
    }

    float4x3 world_view_t = transpose(cam.world_view_transform);
    float3x3 view_to_world_rotation = float3x3(world_view_t);
    float3 camera_origin = -mul(view_to_world_rotation, world_view_t[3]);
    if (cam.model == int(CameraModel::EQUIRECTANGULAR)) {
        float azimuth = 2.0f * M_PI * (pixel.x / float(cam.W) - 0.5f);
        float elevation = M_PI * (pixel.y / float(cam.H) - 0.5f);
        float cos_elevation = cos(elevation);
        float3 ray_camera = safe_normalize(float3(
            sin(azimuth) * cos_elevation,
            sin(elevation),
            cos(azimuth) * cos_elevation));
        float3 direction = safe_normalize(mul(view_to_world_rotation, ray_camera));
        return { camera_origin, direction, isfinite(dot(camera_origin, camera_origin)) && isfinite(dot(direction, direction)) };
    }
    float3 ray_camera = cam.model == int(CameraModel::ORTHO)
                            ? float3(0.0f, 0.0f, 1.0f)
                            : safe_normalize(float3(uv, 1.0f));
    float3 origin = cam.model == int(CameraModel::ORTHO)
                        ? camera_origin + mul(view_to_world_rotation, float3(uv, 0.0f))
                        : camera_origin;
    float3 direction = safe_normalize(mul(view_to_world_rotation, ray_camera));
    return { origin, direction, isfinite(dot(origin, origin)) && isfinite(dot(direction, direction)) };
}

[ForceInline]
float3 camera_origin_ws(Camera cam) {
    float4x3 world_view_t = transpose(cam.world_view_transform);
    float3x3 view_to_world_rotation = float3x3(world_view_t);
    return -mul(view_to_world_rotation, world_view_t[3]);
}

[ForceInline]
float3 view_direction_to_point(float3 world_point, Camera cam) {
    float4x3 world_view_t = transpose(cam.world_view_transform);
    float3x3 view_to_world_rotation = float3x3(world_view_t);
    if (cam.model == int(CameraModel::ORTHO)) {
        return safe_normalize(mul(view_to_world_rotation, float3(0.0f, 0.0f, 1.0f)));
    }
    float3 camera_origin = -mul(view_to_world_rotation, world_view_t[3]);
    return safe_normalize(world_point - camera_origin);
}

// Matches CUDA reference forward kernel: clamp the projected 2D covariance
// eigenvalues so the splat extent never exceeds MAX_PIXEL_RADIUS. Without
// this, splats viewed at oblique angles can collapse into needle-like spikes
// that streak across many tiles.
// Forward-only (not [Differentiable]); the backward path reconstructs cov
// from buffers and uses analytic gradients.
static const float MAX_PIXEL_RADIUS = 512.0f;

float3 clamp_covariance_2d_extent(float3 cov_vs, float opacity_after_compensation) {
    const float opac_safe = max(opacity_after_compensation, ALPHA_THRESHOLD + 1e-8f);
    const float power_threshold = max(MAX_POWER_THRESHOLD, log(opac_safe / ALPHA_THRESHOLD));
    const float stddev = sqrt(2.0f * power_threshold);
    const float max_sigma = (MAX_PIXEL_RADIUS + 0.5f) / max(stddev, 1e-6f);
    const float max_variance = max_sigma * max_sigma;

    const float eigen_avg = 0.5f * (cov_vs.x + cov_vs.z);
    const float det_cov = cov_vs.x * cov_vs.z - cov_vs.y * cov_vs.y;
    const float eigen_delta = sqrt(max(0.0f, eigen_avg * eigen_avg - det_cov));
    const float eigen1 = eigen_avg + eigen_delta;
    const float eigen2 = max(eigen_avg - eigen_delta, 0.0f);
    const float eigen1_clamped = min(eigen1, max_variance);
    const float eigen2_clamped = min(eigen2, max_variance);

    if (eigen1_clamped >= eigen1 && eigen2_clamped >= eigen2)
        return cov_vs;

    const float2 eigen_vec1 = abs(cov_vs.y) > 1e-6f
        ? normalize(float2(cov_vs.y, eigen1 - cov_vs.x))
        : (cov_vs.x >= cov_vs.z ? float2(1.0f, 0.0f) : float2(0.0f, 1.0f));
    const float2 eigen_vec2 = float2(eigen_vec1.y, -eigen_vec1.x);
    return float3(
        eigen1_clamped * eigen_vec1.x * eigen_vec1.x + eigen2_clamped * eigen_vec2.x * eigen_vec2.x,
        eigen1_clamped * eigen_vec1.x * eigen_vec1.y + eigen2_clamped * eigen_vec2.x * eigen_vec2.y,
        eigen1_clamped * eigen_vec1.y * eigen_vec1.y + eigen2_clamped * eigen_vec2.y * eigen_vec2.y);
}

[ForceInline]
[Differentiable]
float4 add_blur_to_covariance2d(float3 cov_vs, no_diff bool mip_filter, no_diff bool compensate_non_mip) {
    float4 result = float4(cov_vs, 1.0);

    if (mip_filter) {
        float old_det = compute_det(result.xyz);
        result.x += DILATION_MIP_FILTER;
        result.z += DILATION_MIP_FILTER;
        float new_det = compute_det(result.xyz);
        result.w = sqrt(max(0.f, old_det / new_det));
    } else {
        float old_det = compute_det(result.xyz);
        result.x += DILATION_3DGS;
        result.z += DILATION_3DGS;
        if (compensate_non_mip) {
            float new_det = compute_det(result.xyz);
            result.w = sqrt(max(0.f, old_det / new_det));
        }
    }

    return result;
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
float4 covariance_3d_to_2d(
    Camera cam,
    float3 xyz_ws,
    float3x3 cov_ws,
    no_diff bool mip_filter,
    no_diff bool compensate_non_mip) {
    float3x3 R = (float3x3)cam.world_view_transform;
    float3x3 J = compute_jacobian(xyz_ws, cam);
    float3x3 cov_vs = mul(J, mul(R, mul(cov_ws, mul(transpose(R), transpose(J)))));

    return add_blur_to_covariance2d(
        float3(cov_vs[0][0], cov_vs[0][1], cov_vs[1][1]),
        mip_filter,
        compensate_non_mip);
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

struct Splat_3D_GUT_AlphaBlend
{
    float3 xyz_ws;
    float3x3 iscl_rot;
    float opacity;
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
Splat_3D_GUT_AlphaBlend load_splat_gut_alphablend(
    int32_t g_idx,
    StructuredBuffer<float> xyz_ws,
    StructuredBuffer<float4> rotations,
    StructuredBuffer<float> scaling_raw,
    StructuredBuffer<float> opacity_raw,
    StructuredBuffer<float> rgb
) {
    float3 g_xyz_ws = read_t3_float3(g_idx, xyz_ws);
    float4 g_rotation = rotations[g_idx];
    float3 g_scale = exp(read_t3_float3(g_idx, scaling_raw));
    float g_opacity = sigmoid_activation(opacity_raw[g_idx]);
    float3 g_rgb = read_t3_float3(g_idx, rgb);
    float3x3 rotation = quat_to_rotmat(normalize(g_rotation));
    float3 safe_scale = max(g_scale, float3(1e-8f));
    float3x3 inv_scale = float3x3(
        1.0f / safe_scale.x, 0.0f, 0.0f,
        0.0f, 1.0f / safe_scale.y, 0.0f,
        0.0f, 0.0f, 1.0f / safe_scale.z
    );
    float3x3 iscl_rot = mul(inv_scale, transpose(rotation));
    return { g_xyz_ws, iscl_rot, g_opacity, g_rgb };
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

    // Matches CUDA reference forward kernel: skip fragments whose alpha
    // would fall below ALPHA_THRESHOLD before computing exp(). sigma_over_2
    // < 0 corresponds to an indefinite projected covariance.
    float sigma_over_2 = 0.5f * (g_inv_cov_vs.x * d.x*d.x + g_inv_cov_vs.z * d.y*d.y) + g_inv_cov_vs.y * d.x*d.y;
    float val = (sigma_over_2 < 0.0f || sigma_over_2 > power_threshold_for_opacity(g_opacity))
        ? 0.0f
        : exp(-sigma_over_2);

    float alpha = float(clipOpacity(g_opacity * val));
    float3 premult_rgb = clamp(g_rgb, float3(0.0f), float3(MAX_BLEND_COLOR)) * alpha;

    return float4(premult_rgb, alpha);
}

[ForceInline]
float spark_rad_alpha_from_value(float opacity, float val)
{
    if (opacity <= 1.0f) {
        return clipOpacity(opacity * val);
    }

    const float lod_density = exp((opacity * opacity - 1.0f) / 2.718281828459045f);
    const float lod_alpha = 1.0f - pow(max(0.0f, 1.0f - val), lod_density);
    return clipOpacity(lod_alpha);
}

[ForceInline]
float4 evaluate_splat_spark_rad(Splat_2D_AlphaBlend g,
                                no_diff float2 pix_coord)
{
    float2 g_xy = g.xy_vs;
    float3 g_rgb = g.rgb;
    float3 g_inv_cov_vs = g.inv_cov_vs_opacity.xyz;
    float g_opacity = g.inv_cov_vs_opacity.w;

    float2 d = pix_coord - g_xy;

    float sigma_over_2 = 0.5f * (g_inv_cov_vs.x * d.x*d.x + g_inv_cov_vs.z * d.y*d.y) + g_inv_cov_vs.y * d.x*d.y;
    float val = (sigma_over_2 < 0.0f || sigma_over_2 > power_threshold_for_spark_rad_opacity(g_opacity))
        ? 0.0f
        : exp(-sigma_over_2);

    float alpha = float(spark_rad_alpha_from_value(g_opacity, val));
    float3 premult_rgb = clamp(g_rgb, float3(0.0f), float3(MAX_BLEND_COLOR)) * alpha;

    return float4(premult_rgb, alpha);
}

[ForceInline]
float4 evaluate_gut_splat(Splat_3D_GUT_AlphaBlend g, float3 ray_origin, float3 ray_direction)
{
    // Same world-space 3DGUT alpha test as the CUDA path: transform the pixel
    // ray into the Gaussian's normalized local frame and measure ray distance.
    float3 gro = mul(g.iscl_rot, ray_origin - g.xyz_ws);
    float3 grd = safe_normalize(mul(g.iscl_rot, ray_direction));
    float3 gcrod = cross(grd, gro);
    float power = -0.5f * dot(gcrod, gcrod);
    float alpha = float(clipOpacity(g.opacity * exp(power)));
    return float4(clamp(g.rgb, float3(0.0f), float3(MAX_BLEND_COLOR)) * alpha, alpha);
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

    float sigma_over_2 = 0.5f * (g_inv_cov_vs.x * d.x*d.x + g_inv_cov_vs.z * d.y*d.y) + g_inv_cov_vs.y * d.x*d.y;
    float val = (sigma_over_2 < 0.0f || sigma_over_2 > power_threshold_for_opacity(g_opacity))
        ? 0.0f
        : exp(-sigma_over_2);

    float alpha = clipOpacity(g_opacity * val);
    return float(alpha);
}

[ForceInline]
float evaluate_splat_alpha_spark_rad(
    float2 g_xy,
    float g_opacity,
    float3 g_inv_cov_vs,
    no_diff float2 pix_coord)
{

    float2 d = pix_coord - g_xy;

    float sigma_over_2 = 0.5f * (g_inv_cov_vs.x * d.x*d.x + g_inv_cov_vs.z * d.y*d.y) + g_inv_cov_vs.y * d.x*d.y;
    float val = (sigma_over_2 < 0.0f || sigma_over_2 > power_threshold_for_spark_rad_opacity(g_opacity))
        ? 0.0f
        : exp(-sigma_over_2);

    return float(spark_rad_alpha_from_value(g_opacity, val));
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


// Matches CUDA reference power_threshold_for_opacity: floored at
// MAX_POWER_THRESHOLD so low-opacity splats still get a finite influence
// radius (the alpha-cutoff distance never collapses to zero).
float power_threshold_for_opacity(float opacity) {
    const float safe_opacity = max(opacity, ALPHA_THRESHOLD);
    return max(MAX_POWER_THRESHOLD, log(safe_opacity / ALPHA_THRESHOLD));
}

float power_threshold_for_spark_rad_opacity(float opacity) {
    const float safe_opacity = max(opacity, ALPHA_THRESHOLD);
    if (safe_opacity > 1.0f) {
        const float spark_lod_opacity = min(safe_opacity, 5.0f);
        const float adjusted_stddev = sqrt(8.0f) + 0.7f * (spark_lod_opacity - 1.0f);
        return 0.5f * adjusted_stddev * adjusted_stddev;
    }
    return power_threshold_for_opacity(safe_opacity);
}

float extend_from_opacity(float opac) {
    return sqrt(2.0f * power_threshold_for_opacity(opac));
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
    float x0 = inv_a * (bv0 - sqrt(max(bv0*bv0 - a*(c*v0*v0-1.f), 0.0f)));
    float x1 = inv_a * (bv1 + sqrt(max(bv1*bv1 - a*(c*v1*v1-1.f), 0.0f)));

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
    uint sort_capacity;
    uint shN_layout_slots;
    uint lod_enabled;
    uint lod_count;
    uint mip_filter;
    uint render_origin_x;
    uint render_origin_y;
    uint camera_width;
    uint camera_height;
    uint model_num_splats;
    float fx;
    float fy;
    float cx;
    float cy;
    // HiGS raster/compose wave window start (batch index); occupies what was
    // implicit alignment padding before dist_coeffs, so the layout is unchanged.
    uint wave_base;
    // Splats per LOD pool page (quant-pool projection: page = g_idx / this).
    uint lod_page_splats;
    // 0 = median depth; > 0 = alpha-weighted (expected) depth, far bound = value.
    float expected_far;
    // Explicit padding to keep dist_coeffs 16-byte aligned; mirrors the C++
    // struct exactly (192 bytes total).
    uint depth_wave;
    uint uniforms_pad1;
    uint uniforms_pad2;
    float4 dist_coeffs;
    float4x4 world_view_transform;
}

// HiGS macro-tile grid (64x32 px macro tiles), derived from the legacy
// 16px-tile grid dims carried in the uniforms: ceil(ceil(w/16)/4) == ceil(w/64).
[ForceInline]
uint higs_macro_grid_width(uint grid_width) {
    return (grid_width + HIGS_MACRO_T16_W - 1u) / HIGS_MACRO_T16_W;
}

[ForceInline]
uint higs_macro_grid_height(uint grid_height) {
    return (grid_height + HIGS_MACRO_T16_H - 1u) / HIGS_MACRO_T16_H;
}

static const uint OVERLAY_FLAG_DIM = 1;
static const uint OVERLAY_FLAG_NONSELECTABLE = 2;
static const uint OVERLAY_FLAG_FLASH = 4;

static const uint SELECTION_PREVIEW_FLAG = 0x80;
static const uint SELECTION_GROUP_MASK = 0x7F;
static const uint SELECTION_PREVIEW_COLOR_INDEX = 256;
static const uint SELECTION_SELECTED_HOVER_COLOR_INDEX = 257;
static const float SELECTION_PREVIEW_BLEND = 0.9f;
static const float SELECTION_COMMITTED_BLEND = 0.8f;

static const uint OVERLAY_PARAM_CROP_FLAGS = 0;
static const uint OVERLAY_PARAM_CROP_MIN = 1;
static const uint OVERLAY_PARAM_CROP_MAX = 2;
static const uint OVERLAY_PARAM_CROP_TRANSFORM = 3;
static const uint OVERLAY_PARAM_ELLIPSOID_FLAGS = 7;
static const uint OVERLAY_PARAM_ELLIPSOID_RADII = 8;
static const uint OVERLAY_PARAM_ELLIPSOID_TRANSFORM = 9;
static const uint OVERLAY_PARAM_VIEW_FLAGS = 13;
static const uint OVERLAY_PARAM_VIEW_MIN = 14;
static const uint OVERLAY_PARAM_VIEW_MAX = 15;
static const uint OVERLAY_PARAM_VIEW_TRANSFORM = 16;
static const uint OVERLAY_PARAM_EMPHASIS_FLAGS = 20;
static const uint OVERLAY_PARAM_CURSOR_FLAGS = 21;
static const uint OVERLAY_PARAM_MARKER_FLAGS = 22;
static const uint OVERLAY_PARAM_SELECTION_CURSOR = 23;
static const uint OVERLAY_PARAM_SELECTION_FLAGS = 24;
static const uint OVERLAY_PARAM_VISIBILITY_FLAGS = 25;
static const uint OVERLAY_PARAM_CROP_EXTRA_BASE = 26;
static const uint OVERLAY_PARAM_CROP_STRIDE = 7;
static const uint OVERLAY_PARAM_CROP_EXTRA_COUNT = 15;
static const uint OVERLAY_PARAM_ELLIPSOID_EXTRA_BASE = OVERLAY_PARAM_CROP_EXTRA_BASE + OVERLAY_PARAM_CROP_STRIDE * OVERLAY_PARAM_CROP_EXTRA_COUNT;
static const uint OVERLAY_PARAM_ELLIPSOID_STRIDE = 5;
static const uint OVERLAY_PARAM_ELLIPSOID_EXTRA_COUNT = 15;
static const uint OVERLAY_PARAM_COUNT = OVERLAY_PARAM_ELLIPSOID_EXTRA_BASE + OVERLAY_PARAM_ELLIPSOID_STRIDE * OVERLAY_PARAM_ELLIPSOID_EXTRA_COUNT;

[ForceInline]
bool overlay_enabled(float value)
{
    return value > 0.5f;
}

[ForceInline]
int overlay_int(float value)
{
    return int(round(value));
}

[ForceInline]
uint read_byte(ByteAddressBuffer buffer, uint idx)
{
    uint aligned_idx = idx & ~3u;
    uint shift = (idx & 3u) * 8u;
    return (buffer.Load(aligned_idx) >> shift) & 0xffu;
}

[ForceInline]
int read_int32(ByteAddressBuffer buffer, uint idx)
{
    return int(buffer.Load(idx * 4u));
}

[ForceInline]
bool overlay_cursor_hit(float2 gaussian_xy, StructuredBuffer<float4> overlay_params)
{
    float4 cursor = overlay_params[OVERLAY_PARAM_SELECTION_CURSOR];
    float2 delta = gaussian_xy - cursor.xy;
    return dot(delta, delta) <= cursor.z * cursor.z;
}

[ForceInline]
float4 replace_unpremultiplied_color(float4 gauss_rgba, float3 color)
{
    return float4(color * gauss_rgba.a, gauss_rgba.a);
}

[ForceInline]
float3 unpremultiplied_color(float4 gauss_rgba)
{
    return gauss_rgba.a > eps ? gauss_rgba.rgb / gauss_rgba.a : float3(0.0f);
}

[ForceInline]
uint selection_status(
    float2 gaussian_xy,
    uint splat_id,
    uint flags,
    ByteAddressBuffer selection_mask,
    ByteAddressBuffer preview_mask,
    StructuredBuffer<float4> overlay_params)
{
    if (overlay_enabled(overlay_params[OVERLAY_PARAM_CURSOR_FLAGS].y)) {
        return 0u;
    }
    bool selectable = (flags & OVERLAY_FLAG_NONSELECTABLE) == 0u;
    float4 selection_flags = overlay_params[OVERLAY_PARAM_SELECTION_FLAGS];
    uint group_id = 0;
    if (overlay_enabled(selection_flags.x)) {
        group_id = read_byte(selection_mask, splat_id) & SELECTION_GROUP_MASK;
    }
    bool is_committed = group_id > 0;
    bool is_in_preview = false;
    if (overlay_enabled(selection_flags.y)) {
        is_in_preview = read_byte(preview_mask, splat_id) != 0;
    }
    bool under_brush = false;
    if (overlay_enabled(overlay_params[OVERLAY_PARAM_SELECTION_CURSOR].w) && selectable) {
        under_brush = overlay_cursor_hit(gaussian_xy, overlay_params);
    }
    bool add_mode = overlay_enabled(selection_flags.z);
    bool is_ring_highlight = selectable && overlay_int(selection_flags.w) == int(splat_id);
    bool is_preview = (is_in_preview && !is_committed && add_mode) ||
                      (is_in_preview && is_committed && !add_mode) ||
                      (under_brush && selectable && add_mode && !is_committed) ||
                      (under_brush && selectable && !add_mode && is_committed) ||
                      is_ring_highlight;

    return (group_id & SELECTION_GROUP_MASK) | (is_preview ? SELECTION_PREVIEW_FLAG : 0u);
}

[ForceInline]
float4 apply_projection_overlay_color(
    float4 gauss_rgba,
    float2 gaussian_xy,
    uint flags,
    StructuredBuffer<float4> overlay_params)
{
    float3 color = unpremultiplied_color(gauss_rgba);
    bool selectable = (flags & OVERLAY_FLAG_NONSELECTABLE) == 0u;
    float4 cursor_flags = overlay_params[OVERLAY_PARAM_CURSOR_FLAGS];
    if (overlay_enabled(cursor_flags.x) && overlay_enabled(cursor_flags.y) && selectable &&
        overlay_cursor_hit(gaussian_xy, overlay_params)) {
        float lum = 0.2126f * color.x + 0.7152f * color.y + 0.0722f * color.z;
        float sat = 1.0f + cursor_flags.z;
        color = clamp(lum + sat * (color - lum), 0.0f, 1.0f);
    }
    if ((flags & OVERLAY_FLAG_DIM) != 0u) {
        const float DEPTH_FILTER_BRIGHTNESS = 0.25f;
        float lum = 0.2126f * color.x + 0.7152f * color.y + 0.0722f * color.z;
        color = float3(lum * DEPTH_FILTER_BRIGHTNESS);
    }
    return replace_unpremultiplied_color(gauss_rgba, color);
}

[ForceInline]
float3 selection_overlay_target_color(uint sel_status, StructuredBuffer<float4> selection_colors)
{
    uint group_id = sel_status & SELECTION_GROUP_MASK;
    bool is_preview = (sel_status & SELECTION_PREVIEW_FLAG) != 0u;
    if (is_preview && group_id > 0u) {
        return selection_colors[SELECTION_SELECTED_HOVER_COLOR_INDEX].xyz;
    }
    if (is_preview) {
        return selection_colors[SELECTION_PREVIEW_COLOR_INDEX].xyz;
    }
    if (group_id > 0u) {
        return selection_colors[group_id].xyz;
    }
    return selection_colors[0].xyz;
}

[ForceInline]
float4 apply_ring_overlay(
    float4 gauss_rgba,
    Splat_2D_AlphaBlend g,
    float2 pix_coord,
    uint sel_status,
    uint flags,
    StructuredBuffer<float4> overlay_params,
    StructuredBuffer<float4> selection_colors)
{
    bool show_rings = overlay_enabled(overlay_params[OVERLAY_PARAM_MARKER_FLAGS].x);
    bool selectable = (flags & OVERLAY_FLAG_NONSELECTABLE) == 0u;
    if (!show_rings || !selectable) {
        return gauss_rgba;
    }
    float ring_width = overlay_params[OVERLAY_PARAM_CURSOR_FLAGS].w;
    float2 d = pix_coord - g.xy_vs;
    float3 inv_cov = g.inv_cov_vs_opacity.xyz;
    float sigma_over_2 = 0.5f * (inv_cov.x * d.x * d.x + inv_cov.z * d.y * d.y) + inv_cov.y * d.x * d.y;
    float gaussian = exp(-sigma_over_2);
    float boundary_gaussian = ALPHA_THRESHOLD / max(g.inv_cov_vs_opacity.w, eps);
    float ring_outer = boundary_gaussian * (1.0f + ring_width * 10.0f);
    float ring_inner = boundary_gaussian * (1.0f - ring_width * 10.0f);
    if (gaussian < ring_outer && gaussian > ring_inner) {
        float3 color = unpremultiplied_color(gauss_rgba);
        uint group_id = sel_status & SELECTION_GROUP_MASK;
        bool is_preview = (sel_status & SELECTION_PREVIEW_FLAG) != 0u;
        if (is_preview || group_id > 0u) {
            color = selection_overlay_target_color(sel_status, selection_colors);
        }
        float alpha = min(0.8f, ALPHA_BLEND_MAX_OPACITY);
        return float4(color * alpha, alpha);
    }
    return gauss_rgba;
}

[ForceInline]
float4 apply_selection_overlay(
    float4 gauss_rgba,
    uint sel_status,
    StructuredBuffer<float4> selection_colors)
{
    uint group_id = sel_status & SELECTION_GROUP_MASK;
    bool is_preview = (sel_status & SELECTION_PREVIEW_FLAG) != 0u;
    if (!is_preview && group_id == 0u) {
        return gauss_rgba;
    }

    float3 color = unpremultiplied_color(gauss_rgba);
    if (is_preview) {
        color = lerp(color, selection_overlay_target_color(sel_status, selection_colors), SELECTION_PREVIEW_BLEND);
    } else if (group_id > 0u) {
        color = lerp(color, selection_colors[group_id].xyz, SELECTION_COMMITTED_BLEND);
    }
    return replace_unpremultiplied_color(gauss_rgba, color);
}

[ForceInline]
float4 apply_flash_overlay(
    float4 gauss_rgba,
    uint flags,
    StructuredBuffer<float4> overlay_params)
{
    if ((flags & OVERLAY_FLAG_FLASH) == 0u) {
        return gauss_rgba;
    }
    float intensity = overlay_params[OVERLAY_PARAM_EMPHASIS_FLAGS].w;
    if (intensity <= 0.0f) {
        return gauss_rgba;
    }
    float3 color = unpremultiplied_color(gauss_rgba);
    const float3 flash_color = float3(1.0f, 0.95f, 0.6f);
    color = lerp(color, flash_color, intensity * 0.5f);
    return replace_unpremultiplied_color(gauss_rgba, color);
}

[ForceInline]
float3 transform_point_rows(StructuredBuffer<float4> params, uint base_idx, float3 p)
{
    float4 r0 = params[base_idx + 0];
    float4 r1 = params[base_idx + 1];
    float4 r2 = params[base_idx + 2];
    return float3(dot(r0.xyz, p) + r0.w,
                  dot(r1.xyz, p) + r1.w,
                  dot(r2.xyz, p) + r2.w);
}

[ForceInline]
float3x3 linear_transform_rows(StructuredBuffer<float4> params, uint base_idx)
{
    float4 r0 = params[base_idx + 0];
    float4 r1 = params[base_idx + 1];
    float4 r2 = params[base_idx + 2];
    return float3x3(r0.x, r0.y, r0.z,
                    r1.x, r1.y, r1.z,
                    r2.x, r2.y, r2.z);
}

[ForceInline]
float3x3 transform_covariance_rows(StructuredBuffer<float4> params, uint base_idx, float3x3 cov)
{
    float3x3 m = linear_transform_rows(params, base_idx);
    return mul(m, mul(cov, transpose(m)));
}

[ForceInline]
bool extract_rotation_rows(StructuredBuffer<float4> params, uint base_idx, out float3 row0, out float3 row1, out float3 row2)
{
    row0 = float3(1.0f, 0.0f, 0.0f);
    row1 = float3(0.0f, 1.0f, 0.0f);
    row2 = float3(0.0f, 0.0f, 1.0f);
    float4 m0 = params[base_idx + 0];
    float4 m1 = params[base_idx + 1];
    float4 m2 = params[base_idx + 2];
    float scale_x = length(float3(m0.x, m1.x, m2.x));
    float scale_y = length(float3(m0.y, m1.y, m2.y));
    float scale_z = length(float3(m0.z, m1.z, m2.z));
    if (scale_x <= 1.0e-8f || scale_y <= 1.0e-8f || scale_z <= 1.0e-8f)
        return false;

    row0 = float3(m0.x / scale_x, m0.y / scale_y, m0.z / scale_z);
    row1 = float3(m1.x / scale_x, m1.y / scale_y, m1.z / scale_z);
    row2 = float3(m2.x / scale_x, m2.y / scale_y, m2.z / scale_z);
    return true;
}

[ForceInline]
float3 transpose_mul_rows(float3 row0, float3 row1, float3 row2, float3 v)
{
    return float3(row0.x * v.x + row1.x * v.y + row2.x * v.z,
                  row0.y * v.x + row1.y * v.y + row2.y * v.z,
                  row0.z * v.x + row1.z * v.y + row2.z * v.z);
}
