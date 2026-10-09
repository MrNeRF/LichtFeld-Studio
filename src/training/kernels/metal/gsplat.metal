// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

// GsplatRasterOps: ports of rasterization/gsplat (camera models, 3DGUT
// unscented projection, tile binning, world-space ray rasterization, SH) and
// the gradient accumulation of gsplat_rasterizer_cuda.cpp. One camera, global
// shutter. Binning sorts gaussians by depth bits, emits their tiles in that
// order and stable-sorts the tile ids, which yields CUDA's (tile, depth, id)
// order with a sort over the tile bits only.

constant constexpr uint kGsplatTile = 16u;
constant constexpr uint kGsplatBatch = 64u;
constant constexpr uint kGsplatRadixItems = 8u;
constant constexpr uint kGsplatRadixBlock = kThreadgroupWidth * kGsplatRadixItems;
constant constexpr uint kGsplatGradStride = 14u; // mean 3, scale 3, quat 4, opacity 1, rgb 3
constant constexpr float kGsplatAlphaMin = 1.0f / 255.0f;
constant constexpr float kGsplatPi = 3.14159265358979323846f;
constant constexpr float kGsplatNear = 0.01f;
constant constexpr float kGsplatFar = 10000.0f;
constant constexpr float kGsplatEps2d = 0.3f;
constant constexpr float kGsplatMargin = 0.1f;

constant constexpr uint kGsplatPinhole = 0u;
constant constexpr uint kGsplatFisheye = 2u;
constant constexpr uint kGsplatEquirect = 3u;
constant constexpr uint kGsplatThinPrism = 4u;

// Written once per forward by gsplat_camera_setup; read by every other kernel.
struct GsplatCamera {
    float3x3 R; // world to camera rotation
    float3 t;
    float3 origin; // camera centre, -R^T t
    float2 focal;
    float2 principal;
    float2 offset;    // equirectangular tile offset
    uint2 resolution; // camera-model resolution
    uint model;
    uint distorted; // pinhole with any distortion tensor: OpenCV model
    float max_angle;
    float backward_slope;
    float radial[6];
    float tangential[2];
    float prism[4];
};
static_assert(sizeof(GsplatCamera) <= 256, "gsplat camera block exceeds its 64-float buffer");

struct GsplatRay {
    float3 origin;
    float3 dir;
    bool valid;
};

// ---------------------------------------------------------------------------
// Camera models (Cameras.cuh)

static float gsplat_cbrt(const float x) {
    return copysign(precise::pow(fabs(x), 1.0f / 3.0f), x);
}

// Smallest positive root of 1 + a x + b x^2 + c x^3.
static float gsplat_fisheye_max_angle(const float a, const float b, const float c) {
    if (c == 0.0f) {
        if (b == 0.0f)
            return a >= 0.0f ? FLT_MAX : -1.0f / a;
        float delta = a * a - 4.0f * b;
        if (delta >= 0.0f) {
            delta = precise::sqrt(delta) - a;
            if (delta > 0.0f)
                return 2.0f / delta;
        }
        return FLT_MAX;
    }
    const float boc = b / c;
    const float boc2 = boc * boc;
    const float t1 = (9.0f * a * boc - 2.0f * b * boc2 - 27.0f) / c;
    const float t2 = 3.0f * a / c - boc2;
    const float delta = t1 * t1 + 4.0f * t2 * t2 * t2;
    if (delta >= 0.0f) {
        const float cube_root = gsplat_cbrt((precise::sqrt(delta) + t1) / 2.0f);
        if (cube_root != 0.0f) {
            const float soln = (cube_root - (t2 / cube_root) - boc) / 3.0f;
            if (soln > 0.0f)
                return soln;
        }
        return FLT_MAX;
    }
    const float theta = precise::atan2(precise::sqrt(-delta), t1) / 3.0f;
    const float t3 = 2.0f * precise::sqrt(-t2);
    float soln = FLT_MAX;
    for (int i = -1; i <= 1; ++i) {
        const float s = (t3 * precise::cos(theta + float(i) * (2.0f * kGsplatPi / 3.0f)) - boc) / 3.0f;
        if (s > 0.0f)
            soln = min(soln, s);
    }
    return soln;
}

// theta + k1 theta^3 + ... + k4 theta^9 and its derivative.
static float gsplat_fisheye_forward(device const GsplatCamera& cam, const float theta) {
    const float t2 = theta * theta;
    return theta * (1.0f + t2 * (cam.radial[0] + t2 * (cam.radial[1] + t2 * (cam.radial[2] + t2 * cam.radial[3]))));
}

static float gsplat_fisheye_dforward(device const GsplatCamera& cam, const float theta) {
    const float t2 = theta * theta;
    return 1.0f + t2 * (3.0f * cam.radial[0] +
                        t2 * (5.0f * cam.radial[1] + t2 * (7.0f * cam.radial[2] + t2 * (9.0f * cam.radial[3]))));
}

// Newton inverse of the forward polynomial, 20 iterations.
static float gsplat_fisheye_theta(device const GsplatCamera& cam, const float delta, thread bool& converged) {
    float x = delta * cam.backward_slope;
    converged = false;
    for (int j = 0; j < 20; ++j) {
        const float dx = (gsplat_fisheye_forward(cam, x) - delta) / gsplat_fisheye_dforward(cam, x);
        x -= dx;
        if (fabs(dx) < 1e-6f) {
            converged = true;
            break;
        }
    }
    return x;
}

// Tangential and prism terms act on the theta-scaled point.
static float2 gsplat_thin_prism_terms(device const GsplatCamera& cam, const float2 point) {
    const float p1 = cam.prism[0], p2 = cam.prism[1], sx1 = cam.prism[2], sy1 = cam.prism[3];
    const float r2 = point.x * point.x + point.y * point.y;
    return float2(2.0f * p1 * point.x * point.y + p2 * (r2 + 2.0f * point.x * point.x) + sx1 * r2,
                  p1 * (r2 + 2.0f * point.y * point.y) + 2.0f * p2 * point.x * point.y + sy1 * r2);
}

static bool gsplat_in_bounds(const float2 p, const uint2 resolution) {
    const float mx = float(resolution.x) * kGsplatMargin;
    const float my = float(resolution.y) * kGsplatMargin;
    return -mx <= p.x && p.x < float(resolution.x) + mx && -my <= p.y && p.y < float(resolution.y) + my;
}

// OpenCV radial / tangential / thin-prism distortion; returns icD.
static float gsplat_distortion(device const GsplatCamera& cam, const float2 uv, thread float2& delta) {
    const float u2 = uv.x * uv.x, v2 = uv.y * uv.y;
    const float r2 = u2 + v2;
    const float a1 = 2.0f * uv.x * uv.y;
    const float a2 = r2 + 2.0f * u2;
    const float a3 = r2 + 2.0f * v2;
    const float numerator = 1.0f + r2 * (cam.radial[0] + r2 * (cam.radial[1] + r2 * cam.radial[2]));
    const float denominator = 1.0f + r2 * (cam.radial[3] + r2 * (cam.radial[4] + r2 * cam.radial[5]));
    delta = float2(cam.tangential[0] * a1 + cam.tangential[1] * a2 + r2 * (cam.prism[0] + r2 * cam.prism[1]),
                   cam.tangential[0] * a3 + cam.tangential[1] * a1 + r2 * (cam.prism[2] + r2 * cam.prism[3]));
    return numerator / denominator;
}

// Camera point to image point; false when the projection is invalid.
static bool gsplat_camera_to_image(device const GsplatCamera& cam, const float3 p, thread float2& image) {
    image = float2(0.0f);
    if (cam.model == kGsplatEquirect) {
        const float azimuth = atan2(p.x, p.z);
        const float elevation = asin(p.y / length(p));
        image = float2((azimuth / (2.0f * kGsplatPi) + 0.5f) * float(cam.resolution.x),
                       (elevation / kGsplatPi + 0.5f) * float(cam.resolution.y));
        return true;
    }
    if (p.z <= 0.0f && (cam.model == kGsplatPinhole || (p.x == 0.0f && p.y == 0.0f)))
        return false;
    if (cam.model == kGsplatPinhole) {
        const float2 uv = p.xy / p.z;
        if (cam.distorted == 0u) {
            image = uv * cam.focal + cam.principal;
            return gsplat_in_bounds(image, cam.resolution);
        }
        float2 delta;
        const float icD = gsplat_distortion(cam, uv, delta);
        image = (icD * uv + delta) * cam.focal + cam.principal;
        return icD > 0.8f && gsplat_in_bounds(image, cam.resolution);
    }
    // Fisheye and thin-prism fisheye.
    const float ax = fabs(p.x), ay = fabs(p.y);
    const float lo = fmin(ax, ay), hi = fmax(ax, ay);
    float norm = hi <= 0.0f ? 0.0f : hi * sqrt(1.0f + (lo / hi) * (lo / hi));
    if (norm <= 0.0f)
        norm = FLT_EPSILON;
    const float theta_full = atan2(norm, p.z);
    const float theta = theta_full < cam.max_angle ? theta_full : cam.max_angle;
    const float thetad = gsplat_fisheye_forward(cam, theta);
    if (cam.model == kGsplatFisheye) {
        const float delta = thetad / norm;
        if (delta <= 0.0f)
            return false;
        image = cam.focal * delta * p.xy + cam.principal;
    } else {
        if (thetad <= 0.0f)
            return false;
        const float2 theta_point = p.xy * (theta / norm);
        const float2 prism = gsplat_thin_prism_terms(cam, theta_point);
        const float radial_scale = thetad / norm;
        const float xd = radial_scale * p.x + prism.x;
        const float yd = radial_scale * p.y + prism.y;
        image = cam.focal * float2(xd, yd) + cam.principal;
    }
    return gsplat_in_bounds(image, cam.resolution) && theta_full < cam.max_angle;
}

// Newton undistortion of the OpenCV pinhole model, at most 5 iterations.
static float2 gsplat_undistort(device const GsplatCamera& cam, const float2 uv0, thread bool& converged) {
    const float k1 = cam.radial[0], k2 = cam.radial[1], k3 = cam.radial[2];
    const float k4 = cam.radial[3], k5 = cam.radial[4], k6 = cam.radial[5];
    const float p1 = cam.tangential[0], p2 = cam.tangential[1];
    const float s1 = cam.prism[0], s2 = cam.prism[1], s3 = cam.prism[2], s4 = cam.prism[3];
    const float xd = uv0.x, yd = uv0.y;
    float x = xd, y = yd;
    converged = false;
    for (int iter = 0; iter < 5; ++iter) {
        const float r = x * x + y * y;
        const float r2 = r * r;
        const float alpha = 1.0f + r * (k1 + r * (k2 + r * k3));
        const float beta = 1.0f + r * (k4 + r * (k5 + r * k6));
        const float d = alpha / beta;
        if (d <= 0.0f)
            break;
        const float fx = d * x + 2 * p1 * x * y + p2 * (r + 2 * x * x) + s1 * r + s2 * r2 - xd;
        const float fy = d * y + 2 * p2 * x * y + p1 * (r + 2 * y * y) + s3 * r + s4 * r2 - yd;
        const float alpha_r = k1 + r * (2.0f * k2 + r * (3.0f * k3));
        const float beta_r = k4 + r * (2.0f * k5 + r * (3.0f * k6));
        const float d_r = (alpha_r * beta - alpha * beta_r) / (beta * beta);
        const float d_x = 2.0f * x * d_r;
        const float d_y = 2.0f * y * d_r;
        const float fx_x = d + d_x * x + 2.0f * p1 * y + 6.0f * p2 * x + 2.0f * x * (s1 + 2.0f * s2 * r);
        const float fx_y = d_y * x + 2.0f * p1 * x + 2.0f * p2 * y + 2.0f * y * (s1 + 2.0f * s2 * r);
        const float fy_x = d_x * y + 2.0f * p2 * y + 2.0f * p1 * x + 2.0f * x * (s3 + 2.0f * s4 * r);
        const float fy_y = d + d_y * y + 2.0f * p2 * x + 6.0f * p1 * y + 2.0f * y * (s3 + 2.0f * s4 * r);
        const float det = fx_y * fy_x - fx_x * fy_y;
        if (fabs(det) < 1e-6f)
            break;
        const float dx = (fx * fy_y - fy * fx_y) / det;
        const float dy = (fy * fx_x - fx * fy_x) / det;
        x += dx;
        y += dy;
        if (fabs(dx) < 1e-6f && fabs(dy) < 1e-6f) {
            converged = true;
            break;
        }
    }
    return float2(x, y);
}

// World ray through a render-space pixel position.
static GsplatRay gsplat_pixel_ray(device const GsplatCamera& cam, const float2 pixel) {
    float3 ray = float3(0.0f, 0.0f, 1.0f);
    bool valid = true;
    if (cam.model == kGsplatEquirect) {
        const float2 full = pixel + cam.offset;
        const float azimuth = 2.0f * kGsplatPi * (full.x / float(cam.resolution.x) - 0.5f);
        const float elevation = kGsplatPi * (full.y / float(cam.resolution.y) - 0.5f);
        ray = float3(cos(elevation) * sin(azimuth), sin(elevation), cos(azimuth) * cos(elevation));
        ray /= length(ray);
    } else {
        float2 uv = (pixel - cam.principal) / cam.focal;
        if (cam.model == kGsplatPinhole) {
            if (cam.distorted != 0u)
                uv = gsplat_undistort(cam, uv, valid);
            ray = float3(uv, 1.0f);
            ray /= length(ray);
        } else {
            if (cam.model == kGsplatThinPrism) {
                float2 radial_point = uv, theta_point = float2(0.0f);
                float theta = 0.0f;
                for (int i = 0; i <= 5; ++i) {
                    if (i > 0)
                        radial_point = uv - gsplat_thin_prism_terms(cam, theta_point);
                    const float thetad = length(radial_point);
                    bool converged;
                    theta = gsplat_fisheye_theta(cam, thetad, converged);
                    valid = !(theta < 0.0f || theta >= cam.max_angle || !converged);
                    if (!valid)
                        break;
                    theta_point = thetad >= 1e-6f ? radial_point * (theta / thetad) : float2(0.0f);
                }
                if (valid && theta >= 1e-6f) {
                    const float scale = sin(theta) / theta;
                    ray = float3(scale * theta_point.x, scale * theta_point.y, cos(theta));
                }
            } else {
                const float delta = length(uv);
                bool converged;
                const float theta = gsplat_fisheye_theta(cam, delta, converged);
                valid = !(theta < 0.0f || theta >= cam.max_angle || !converged);
                if (valid && delta >= 1e-6f) {
                    const float s = sin(theta) / delta;
                    ray = float3(s * uv.x, s * uv.y, cos(theta));
                }
            }
        }
    }
    if (!valid)
        return {float3(0.0f), float3(0.0f), false};
    const float3x3 R = cam.R;
    return {cam.origin, transpose(R) * ray, true};
}

struct GsplatSetupParams {
    device const float* view;
    device const float* radial;
    device const float* tangential;
    device const float* prism;
    device GsplatCamera* camera;
    float fx, fy, cx, cy;
    uint width, height;
    uint model;
    uint radial_count, tangential_count, prism_count;
};

kernel void gsplat_camera_setup(constant GsplatSetupParams& p [[buffer(0)]]) {
    GsplatCamera cam;
    device const float* v = p.view; // row-major [R | t]
    cam.R = float3x3(float3(v[0], v[4], v[8]), float3(v[1], v[5], v[9]), float3(v[2], v[6], v[10]));
    cam.t = float3(v[3], v[7], v[11]);
    cam.origin = -(transpose(cam.R) * cam.t);
    cam.focal = float2(p.fx, p.fy);
    cam.principal = float2(p.cx, p.cy);
    cam.model = p.model;
    cam.distorted = p.radial_count + p.tangential_count > 0u ? 1u : 0u;
    for (uint i = 0; i < 6u; ++i)
        cam.radial[i] = i < p.radial_count ? p.radial[i] : 0.0f;
    for (uint i = 0; i < 2u; ++i)
        cam.tangential[i] = i < p.tangential_count ? p.tangential[i] : 0.0f;
    for (uint i = 0; i < 4u; ++i)
        cam.prism[i] = i < p.prism_count ? p.prism[i] : 0.0f;
    cam.offset = float2(0.0f);
    cam.resolution = uint2(p.width, p.height);
    cam.max_angle = FLT_MAX;
    cam.backward_slope = 0.0f;
    if (p.model == kGsplatEquirect) {
        // K carries the full image size and the tile offset.
        cam.resolution = uint2(uint(p.fx), uint(p.fy));
        cam.offset = float2(p.cx, p.cy);
    } else if (p.model == kGsplatFisheye || p.model == kGsplatThinPrism) {
        const float k1 = cam.radial[0], k2 = cam.radial[1], k3 = cam.radial[2], k4 = cam.radial[3];
        const float2 res = float2(cam.resolution);
        const float max_x = fmax(res.x - p.cx, p.cx);
        const float max_y = fmax(res.y - p.cy, p.cy);
        const float max_radius = precise::sqrt(max_x * max_x + max_y * max_y);
        float max_angle;
        if (k4 == 0.0f) {
            max_angle = precise::sqrt(gsplat_fisheye_max_angle(3.0f * k1, 5.0f * k2, 7.0f * k3));
        } else {
            // Newton on the derivative polynomial from 1.57.
            float x = 1.57f;
            bool converged = false;
            for (int j = 0; j < 20; ++j) {
                const float x2 = x * x;
                const float f = 1.0f + x2 * (3.0f * k1 + x2 * (5.0f * k2 + x2 * (7.0f * k3 + x2 * (9.0f * k4))));
                const float df = x * (6.0f * k1 + x2 * (20.0f * k2 + x2 * (42.0f * k3 + x2 * (72.0f * k4))));
                const float dx = f / df;
                x -= dx;
                if (fabs(dx) < 1e-6f) {
                    converged = true;
                    break;
                }
            }
            max_angle = (!converged || x <= 0.0f) ? FLT_MAX : x;
        }
        cam.max_angle = fmin(fmin(max_angle,kGsplatPi), fmax(max_radius / p.fx, max_radius / p.fy));
        cam.backward_slope = cam.max_angle / fmax(res.x / 2.0f / p.fx, res.y / 2.0f / p.fy);
    }
    *p.camera = cam;
}

// ---------------------------------------------------------------------------
// Gaussian math (Utils.cuh)

static float3 gsplat_load3(device const float* p, const uint i) {
    return float3(p[i * 3u], p[i * 3u + 1u], p[i * 3u + 2u]);
}

static float4 gsplat_load4(device const float* p, const uint i) {
    return float4(p[i * 4u], p[i * 4u + 1u], p[i * 4u + 2u], p[i * 4u + 3u]);
}

static float gsplat_sigmoid(const float raw) {
    return 1.0f / (1.0f + exp(-raw));
}

// Rotation of a unit (w, x, y, z) quaternion, column-major as glm.
static float3x3 gsplat_rotmat(const float4 q) {
    const float w = q.x, x = q.y, y = q.z, z = q.w;
    const float x2 = x * x, y2 = y * y, z2 = z * z;
    const float xy = x * y, xz = x * z, yz = y * z;
    const float wx = w * x, wy = w * y, wz = w * z;
    return float3x3(float3(1.0f - 2.0f * (y2 + z2), 2.0f * (xy + wz), 2.0f * (xz - wy)),
                    float3(2.0f * (xy - wz), 1.0f - 2.0f * (x2 + z2), 2.0f * (yz + wx)),
                    float3(2.0f * (xz + wy), 2.0f * (yz - wx), 1.0f - 2.0f * (x2 + y2)));
}

static float3x3 gsplat_quat_to_rotmat(const float4 q) {
    return gsplat_rotmat(q * rsqrt(dot(q, q)));
}

// S^-1 R^T: world offsets to the gaussian's unit frame.
static float3x3 gsplat_world_to_unit(const float4 quat, const float3 scale) {
    const float3x3 R = gsplat_quat_to_rotmat(quat);
    const float3x3 S_inv = float3x3(float3(1.0f / scale.x, 0.0f, 0.0f), float3(0.0f, 1.0f / scale.y, 0.0f),
                                    float3(0.0f, 0.0f, 1.0f / scale.z));
    return transpose(R * S_inv);
}

static float3 gsplat_safe_normalize(const float3 v) {
    const float l = dot(v, v);
    return l > 0.0f ? v * rsqrt(l) : v;
}

static float3 gsplat_safe_normalize_bw(const float3 v, const float3 d_out) {
    const float l = dot(v, v);
    if (l > 0.0f) {
        const float il = rsqrt(l);
        return il * d_out - il * il * il * dot(d_out, v) * v;
    }
    return d_out;
}

static float3x3 gsplat_outer(const float3 c, const float3 r) {
    return float3x3(c * r.x, c * r.y, c * r.z);
}

static void gsplat_quat_to_rotmat_vjp(const float4 quat, const float3x3 v_R, thread float4& v_quat) {
    const float inv_norm = rsqrt(dot(quat, quat));
    const float w = quat.x * inv_norm, x = quat.y * inv_norm, y = quat.z * inv_norm, z = quat.w * inv_norm;
    const float4 v_quat_n = float4(
        2.0f * (x * (v_R[1][2] - v_R[2][1]) + y * (v_R[2][0] - v_R[0][2]) + z * (v_R[0][1] - v_R[1][0])),
        2.0f * (-2.0f * x * (v_R[1][1] + v_R[2][2]) + y * (v_R[0][1] + v_R[1][0]) + z * (v_R[0][2] + v_R[2][0]) +
                w * (v_R[1][2] - v_R[2][1])),
        2.0f * (x * (v_R[0][1] + v_R[1][0]) - 2.0f * y * (v_R[0][0] + v_R[2][2]) + z * (v_R[1][2] + v_R[2][1]) +
                w * (v_R[2][0] - v_R[0][2])),
        2.0f * (x * (v_R[0][2] + v_R[2][0]) + y * (v_R[1][2] + v_R[2][1]) - 2.0f * z * (v_R[0][0] + v_R[1][1]) +
                w * (v_R[0][1] - v_R[1][0])));
    const float4 quat_n = float4(w, x, y, z);
    v_quat += (v_quat_n - dot(v_quat_n, quat_n) * quat_n) * inv_norm;
}

// Gradient of M = R S^-1 into the quaternion and the activated scale.
static void gsplat_preci_half_vjp(const float4 quat, const float3 scale, const float3x3 R, const float3x3 v_M,
                                  thread float4& v_quat, thread float3& v_scale) {
    const float sx = 1.0f / scale.x, sy = 1.0f / scale.y, sz = 1.0f / scale.z;
    const float3x3 S = float3x3(float3(sx, 0.0f, 0.0f), float3(0.0f, sy, 0.0f), float3(0.0f, 0.0f, sz));
    gsplat_quat_to_rotmat_vjp(quat, v_M * S, v_quat);
    v_scale.x += -sx * sx * (R[0][0] * v_M[0][0] + R[0][1] * v_M[0][1] + R[0][2] * v_M[0][2]);
    v_scale.y += -sy * sy * (R[1][0] * v_M[1][0] + R[1][1] * v_M[1][1] + R[1][2] * v_M[1][2]);
    v_scale.z += -sz * sz * (R[2][0] * v_M[2][0] + R[2][1] * v_M[2][1] + R[2][2] * v_M[2][2]);
}

// ---------------------------------------------------------------------------
// Spherical harmonics (SphericalHarmonicsCUDA.cu), basis values up to degree 3.

static void gsplat_sh_basis(const float3 dir, const uint degree, thread float* b) {
    b[0] = 0.2820947917738781f;
    if (degree < 1u)
        return;
    const float3 n = dir * rsqrt(dot(dir, dir));
    const float x = n.x, y = n.y, z = n.z;
    constexpr float C1 = 0.48860251190292f;
    b[1] = -C1 * y;
    b[2] = C1 * z;
    b[3] = -C1 * x;
    if (degree < 2u)
        return;
    const float z2 = z * z;
    const float fTmp0B = -1.092548430592079f * z;
    const float fC1 = x * x - y * y;
    const float fS1 = 2.0f * x * y;
    b[4] = 0.5462742152960395f * fS1;
    b[5] = fTmp0B * y;
    b[6] = 0.9461746957575601f * z2 - 0.3153915652525201f;
    b[7] = fTmp0B * x;
    b[8] = 0.5462742152960395f * fC1;
    if (degree < 3u)
        return;
    const float fTmp0C = -2.285228997322329f * z2 + 0.4570457994644658f;
    const float fTmp1B = 1.445305721320277f * z;
    const float fC2 = x * fC1 - y * fS1;
    const float fS2 = x * fS1 + y * fC1;
    b[9] = -0.5900435899266435f * fS2;
    b[10] = fTmp1B * fS1;
    b[11] = fTmp0C * y;
    b[12] = z * (1.865881662950577f * z2 - 1.119528997770346f);
    b[13] = fTmp0C * x;
    b[14] = fTmp1B * fC1;
    b[15] = -0.5900435899266435f * fC2;
}

// Float index of SH-rest value (coeff, channel) in swizzled float storage.
static uint gsplat_sh_float_index(const uint g, const uint coeff, const uint channel, const uint layout_rest) {
    const uint offset = coeff * 3u + channel;
    return sh_swizzled_index(g, offset / 4u, layout_rest) * 4u + offset % 4u;
}

// ---------------------------------------------------------------------------
// Projection, SH colour and tile count per gaussian (ProjectionUT3DGSFused.cu,
// IntersectTile.cu first pass, spherical_harmonics_swizzled_fwd).

struct GsplatProjectParams {
    device const GsplatCamera* camera;
    device const float* means;
    device const float* scales;
    device const float* quats;
    device const float* opacities;
    device const float* sh0;
    device const float* sh_rest;       // swizzled float storage
    device const ushort* sh_codes;     // swizzled Q16 storage
    device const float* sh_bounds;     // Q16 bounds, float2 per 256 primitives
    device int* radii;                 // [N, 2]
    device float* means2d;             // [N, 2]
    device float* colors;              // [N, 3]
    device uint* depth_keys;           // [N]
    device int* tile_counts;           // [N]
    uint count;
    uint width, height;
    uint tiles_x, tiles_y;
    uint degree;
    uint layout_rest;
};

struct GsplatTileRect {
    uint x0, y0, x1, y1;
};

static GsplatTileRect gsplat_tile_rect(const float2 mean, const int2 radius, const uint tiles_x, const uint tiles_y) {
    const float tile = float(kGsplatTile);
    const float rx = float(radius.x) / tile, ry = float(radius.y) / tile;
    const float tx = mean.x / tile, ty = mean.y / tile;
    return {uint(clamp(floor(tx - rx), 0.0f, float(tiles_x))), uint(clamp(floor(ty - ry), 0.0f, float(tiles_y))),
            uint(clamp(ceil(tx + rx), 0.0f, float(tiles_x))), uint(clamp(ceil(ty + ry), 0.0f, float(tiles_y)))};
}

kernel void gsplat_project(constant GsplatProjectParams& p [[buffer(0)]], uint g [[thread_position_in_grid]]) {
    if (g >= p.count)
        return;
    device const GsplatCamera& cam = *p.camera;
    p.radii[g * 2u] = 0;
    p.radii[g * 2u + 1u] = 0;
    p.tile_counts[g] = 0;
    p.depth_keys[g] = 0xffffffffu;

    const float3 mean = gsplat_load3(p.means, g);
    const float3x3 view_R = cam.R;
    const float3 mean_c = view_R * mean + cam.t;
    if ((cam.model==kGsplatPinhole?mean_c.z:length(mean_c)) < kGsplatNear || (cam.model==kGsplatPinhole?mean_c.z:length(mean_c)) > kGsplatFar)
        return;

    const float3 scale = exp(gsplat_load3(p.scales, g));
    float4 quat = gsplat_load4(p.quats, g);
    const float quat_length = length(quat);
    quat = quat_length > 0.0f ? quat / quat_length : float4(1.0f, 0.0f, 0.0f, 0.0f);
    const float3x3 rotation = gsplat_rotmat(quat);

    // Unscented transform: alpha 0.1, beta 2, kappa 0, all sigma points valid.
    constexpr float D = 3.0f, ut_alpha = 0.1f, ut_beta = 2.0f;
    constexpr float lambda = ut_alpha * ut_alpha * D - D;
    const float spread = sqrt(D + lambda);
    const float w_mean0 = lambda / (D + lambda);
    const float w_cov0 = lambda / (D + lambda) + (1.0f - ut_alpha * ut_alpha + ut_beta);
    const float w_i = 1.0f / (2.0f * (D + lambda));

    float2 image_points[7];
    float2 image_mean = float2(0.0f);
    for (uint i = 0; i < 7u; ++i) {
        float3 point = mean;
        if (i > 0u) {
            const uint axis = (i - 1u) % 3u;
            const float3 delta = spread * scale[axis] * rotation[axis];
            point = i <= 3u ? mean + delta : mean - delta;
        }
        if (!gsplat_camera_to_image(cam, view_R * point + cam.t, image_points[i]))
            return;
        image_mean += (i == 0u ? w_mean0 : w_i) * image_points[i];
    }
    float c00 = 0.0f, c01 = 0.0f, c11 = 0.0f;
    for (uint i = 0; i < 7u; ++i) {
        const float2 d = image_points[i] - image_mean;
        const float w = i == 0u ? w_cov0 : w_i;
        c00 += w * d.x * d.x;
        c01 += w * d.x * d.y;
        c11 += w * d.y * d.y;
    }
    if (cam.model == kGsplatEquirect)
        image_mean -= cam.offset;

    const float det_orig = c00 * c11 - c01 * c01;
    c00 += kGsplatEps2d;
    c11 += kGsplatEps2d;
    const float det = c00 * c11 - c01 * c01;
    const float compensation = sqrt(max(0.0f, det_orig / det));
    if (det <= 0.0f)
        return;
    const float opacity = gsplat_sigmoid(p.opacities[g]) * compensation;
    if (opacity < kGsplatAlphaMin)
        return;
    const float extend = min(3.33f, sqrt(2.0f * log(opacity / kGsplatAlphaMin)));
    const float b = 0.5f * (c00 + c11);
    const float r1 = extend * sqrt(b + sqrt(max(0.01f, b * b - det)));
    const float radius_x = ceil(min(extend * sqrt(c00), r1));
    const float radius_y = ceil(min(extend * sqrt(c11), r1));
    if (radius_x <= 0.0f && radius_y <= 0.0f)
        return;
    if (image_mean.x + radius_x <= 0.0f || image_mean.x - radius_x >= float(p.width) ||
        image_mean.y + radius_y <= 0.0f || image_mean.y - radius_y >= float(p.height))
        return;

    const int2 radius = int2(int(radius_x), int(radius_y));
    p.radii[g * 2u] = radius.x;
    p.radii[g * 2u + 1u] = radius.y;
    p.means2d[g * 2u] = image_mean.x;
    p.means2d[g * 2u + 1u] = image_mean.y;
    p.depth_keys[g] = as_type<uint>(cam.model==kGsplatPinhole?mean_c.z:length(mean_c));
    if (radius.x > 0 && radius.y > 0) {
        const GsplatTileRect rect = gsplat_tile_rect(image_mean, radius, p.tiles_x, p.tiles_y);
        p.tile_counts[g] = int((rect.y1 - rect.y0) * (rect.x1 - rect.x0));
    }

    float basis[16];
    gsplat_sh_basis(mean - cam.origin, p.degree, basis);
    const uint bases = (p.degree + 1u) * (p.degree + 1u);
    const bool q16 = p.sh_codes != nullptr;
    const bool has_rest = q16 || p.sh_rest != nullptr;
    const float2 bounds = q16 ? float2(p.sh_bounds[(g / 256u) * 2u], p.sh_bounds[(g / 256u) * 2u + 1u]) : float2(0.0f);
    for (uint c = 0; c < 3u; ++c) {
        float color = basis[0] * p.sh0[g * 3u + c];
        for (uint k = 1; k < bases && has_rest; ++k) {
            const float value = q16 ? sh_q16_decode(p.sh_codes[sh_q16_index(g, (k - 1u) * 3u + c, p.layout_rest * 3u)],
                                                    bounds.x, bounds.y)
                                    : p.sh_rest[gsplat_sh_float_index(g, k - 1u, c, p.layout_rest)];
            color += basis[k] * value;
        }
        p.colors[g * 3u + c] = color + 0.5f;
    }
}

// ---------------------------------------------------------------------------
// Tile binning.

struct GsplatBinParams {
    device const uint* order;       // gaussians by depth
    device const int* tile_counts;  // [N]
    device int* ranked_counts;      // [N] tile counts in depth order
    device const int* ranked_ends;  // inclusive scan of ranked_counts
    device const int* radii;
    device const float* means2d;
    device uint* tile_keys;
    device uint* gaussian_ids;
    device int* tile_offsets;       // [tiles + 1]
    uint count;
    uint tiles_x, tiles_y;
};

kernel void gsplat_rank_counts(constant GsplatBinParams& p [[buffer(0)]], uint r [[thread_position_in_grid]]) {
    if (r < p.count)
        p.ranked_counts[r] = p.tile_counts[p.order[r]];
}

kernel void gsplat_fill_intersections(constant GsplatBinParams& p [[buffer(0)]], uint r [[thread_position_in_grid]]) {
    if (r >= p.count || p.ranked_counts[r] == 0)
        return;
    const uint g = p.order[r];
    const GsplatTileRect rect = gsplat_tile_rect(float2(p.means2d[g * 2u], p.means2d[g * 2u + 1u]),
                                                 int2(p.radii[g * 2u], p.radii[g * 2u + 1u]), p.tiles_x, p.tiles_y);
    uint cursor = uint(p.ranked_ends[r] - p.ranked_counts[r]);
    for (uint y = rect.y0; y < rect.y1; ++y) {
        for (uint x = rect.x0; x < rect.x1; ++x) {
            p.tile_keys[cursor] = y * p.tiles_x + x;
            p.gaussian_ids[cursor] = g;
            ++cursor;
        }
    }
}

// Tile ranges from sorted tile ids, as intersect_offset_kernel; `count` is
// the intersection count here.
kernel void gsplat_tile_offsets(constant GsplatBinParams& p [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.count)
        return;
    const uint tiles = p.tiles_x * p.tiles_y;
    const uint current = p.tile_keys[i];
    if (i == 0u) {
        for (uint t = 0; t <= current; ++t)
            p.tile_offsets[t] = 0;
    }
    if (i == p.count - 1u) {
        for (uint t = current + 1u; t <= tiles; ++t)
            p.tile_offsets[t] = int(p.count);
    }
    if (i > 0u) {
        const uint previous = p.tile_keys[i - 1u];
        for (uint t = previous + 1u; t <= current; ++t)
            p.tile_offsets[t] = int(i);
    }
}

// Stable LSD radix sort pass over 8 bits of uint keys with uint values.
// Blocks of kGsplatRadixBlock keys; counts are digit-major [256, blocks].
struct GsplatRadixParams {
    device const uint* keys_in;
    device const uint* values_in; // null: the key's index
    device uint* keys_out;
    device uint* values_out;
    device uint* counts;
    device const uint* ends; // inclusive scan of counts
    uint count;
    uint shift;
    uint blocks;
};

kernel void gsplat_radix_histogram(constant GsplatRadixParams& p [[buffer(0)]],
                                   uint tid [[thread_index_in_threadgroup]],
                                   uint block [[threadgroup_position_in_grid]]) {
    threadgroup atomic_uint histogram[kThreadgroupWidth];
    atomic_store_explicit(&histogram[tid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint begin = block * kGsplatRadixBlock;
    for (uint i = 0; i < kGsplatRadixItems; ++i) {
        const uint index = begin + i * kThreadgroupWidth + tid;
        if (index < p.count)
            atomic_fetch_add_explicit(&histogram[(p.keys_in[index] >> p.shift) & 255u], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    p.counts[tid * p.blocks + block] = atomic_load_explicit(&histogram[tid], memory_order_relaxed);
}

// Ranks keys with equal digits inside each simdgroup by ballots over the
// digit bits, then across simdgroups and chunks through threadgroup counts.
// Assumes 32-wide simdgroups, as on every Apple GPU.
kernel void gsplat_radix_scatter(constant GsplatRadixParams& p [[buffer(0)]],
                                 uint tid [[thread_index_in_threadgroup]],
                                 uint block [[threadgroup_position_in_grid]],
                                 uint lane [[thread_index_in_simdgroup]],
                                 uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    constexpr uint kSimdgroups = kThreadgroupWidth / 32u;
    threadgroup uint base[kThreadgroupWidth];
    threadgroup uint simd_counts[kSimdgroups][kThreadgroupWidth];
    const uint slot = tid * p.blocks + block;
    base[tid] = p.ends[slot] - p.counts[slot];
    for (uint s = 0; s < kSimdgroups; ++s)
        simd_counts[s][tid] = 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint lower_lanes = (1u << lane) - 1u;
    const uint begin = block * kGsplatRadixBlock;
    for (uint chunk = 0; chunk < kGsplatRadixItems; ++chunk) {
        const uint index = begin + chunk * kThreadgroupWidth + tid;
        const bool present = index < p.count;
        const uint key = present ? p.keys_in[index] : 0u;
        const uint digit = (key >> p.shift) & 255u;
        uint peers = static_cast<uint>(static_cast<simd_vote::vote_t>(simd_ballot(present)));
        for (uint bit = 0; bit < 8u; ++bit) {
            const bool set = ((digit >> bit) & 1u) != 0u;
            const uint vote = static_cast<uint>(static_cast<simd_vote::vote_t>(simd_ballot(set)));
            peers &= set ? vote : ~vote;
        }
        const uint rank = popcount(peers & lower_lanes);
        if (present && rank == 0u)
            simd_counts[simdgroup][digit] = popcount(peers);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (present) {
            uint position = base[digit] + rank;
            for (uint s = 0; s < simdgroup; ++s)
                position += simd_counts[s][digit];
            p.keys_out[position] = key;
            p.values_out[position] = p.values_in != nullptr ? p.values_in[index] : index;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint total = 0u;
        for (uint s = 0; s < kSimdgroups; ++s) {
            total += simd_counts[s][tid];
            simd_counts[s][tid] = 0u;
        }
        base[tid] += total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ---------------------------------------------------------------------------
// Rasterization (RasterizeToPixelsFromWorld3DGS{Fwd,Bwd}.cu), RGB, 16x16 tiles.

struct GsplatRasterParams {
    device const GsplatCamera* camera;
    device const float* means;
    device const float* scales;
    device const float* quats;
    device const float* opacities;
    device const float* colors;
    device const float* bg_color; // [3] or null
    device const float* bg_image; // [3, H, W] or null
    device const int* tile_offsets;
    device const uint* gaussian_ids;
    device float* image;       // [3, H, W]
    device float* alpha;       // [1, H, W]
    device int* last_ids;      // [H, W]
    device const float* v_image;
    device const float* v_alpha;
    device float* grads;       // [N, kGsplatGradStride]
    device float* densification;       // [2, N] or null
    device const float* error_map;     // [H, W] or null
    device const float* edge_map;      // [H, W] or null
    device float* edge_scores;         // [N] or null
    uint count;
    uint width, height;
    uint tiles_x;
    device float* depth;
    device float* normal;
    device float* rays;
    device const float* v_depth;
    device const float* v_normal;
    uint mode;
};

static float gsplat_background(constant GsplatRasterParams& p, const uint channel, const uint pixel) {
    if (p.bg_image != nullptr)
        return p.bg_image[channel * p.width * p.height + pixel];
    return p.bg_color != nullptr ? p.bg_color[channel] : 0.0f;
}

// Stages one gaussian: xyz and opacity, the rows of S^-1 R^T with the
// activated scale in w, the raw quaternion, and rgb with the id in w.
static void gsplat_stage(constant GsplatRasterParams& p, const uint g, const uint slot, threadgroup float4* xyzo,
                         threadgroup float4* m0, threadgroup float4* m1, threadgroup float4* m2,
                         threadgroup float4* quats, threadgroup float4* rgb) {
    const float3 scale = exp(gsplat_load3(p.scales, g));
    const float4 quat = gsplat_load4(p.quats, g);
    const float3x3 M = gsplat_world_to_unit(quat, scale);
    xyzo[slot] = float4(gsplat_load3(p.means, g), gsplat_sigmoid(p.opacities[g]));
    m0[slot] = float4(M[0], scale.x);
    m1[slot] = float4(M[1], scale.y);
    m2[slot] = float4(M[2], scale.z);
    quats[slot] = quat;
    rgb[slot] = float4(gsplat_load3(p.colors, g), as_type<float>(g));
}

static uint gsplat_normal_axis(float3 scales) { return scales.x<=scales.y && scales.x<=scales.z?0u:scales.y<=scales.z?1u:2u; }
static float3 gsplat_geometry_normal(GsplatCamera cam,float3 mean,float3 scales,float4 quat) {
    float3 world=gsplat_quat_to_rotmat(quat)[gsplat_normal_axis(scales)];
    float3 normal=cam.R*world, position=cam.R*mean+cam.t;
    return normal*(dot(normal,position)>0?-1.f:1.f);
}
kernel void gsplat_geometry_rays(constant GsplatRasterParams& p [[buffer(0)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.width*p.height) return;
    const GsplatCamera cam=*p.camera;
    GsplatRay ray=gsplat_pixel_ray(cam,float2(i%p.width,i/p.width)+0.5f);
    float3 r=cam.R*ray.dir;
    r=cam.model==kGsplatPinhole?r/r.z:normalize(r);
    for(uint c=0;c<3;++c) p.rays[3*i+c]=ray.valid?r[c]:as_type<float>(0x7fc00000u);
}
kernel void gsplat_rasterize_forward(constant GsplatRasterParams& p [[buffer(0)]],
                                     uint2 group [[threadgroup_position_in_grid]],
                                     uint2 local [[thread_position_in_threadgroup]],
                                     uint tid [[thread_index_in_threadgroup]],
                                     uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    threadgroup float4 s_xyzo[kGsplatBatch], s_m0[kGsplatBatch], s_m1[kGsplatBatch], s_m2[kGsplatBatch];
    threadgroup float4 s_quat[kGsplatBatch], s_rgb[kGsplatBatch];
    threadgroup bool s_done[kThreadgroupWidth / 32u];

    const uint tile_id = group.y * p.tiles_x + group.x;
    const uint i = group.y * kGsplatTile + local.y;
    const uint j = group.x * kGsplatTile + local.x;
    const GsplatRay ray = gsplat_pixel_ray(*p.camera, float2(float(j) + 0.5f, float(i) + 0.5f));
    const bool inside = i < p.height && j < p.width;
    bool done = !inside || !ray.valid;

    const int range_start = p.tile_offsets[tile_id];
    const int range_end = p.tile_offsets[tile_id + 1u];
    float T = 1.0f;
    uint cur_idx = 0u;
    float3 pix = float3(0.0f), normal=0;
    float depth=0;
    for (int batch_start = range_start; batch_start < range_end; batch_start += int(kGsplatBatch)) {
        const bool simd_done = simd_all(done);
        if (simd_is_first())
            s_done[simdgroup] = simd_done;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bool all_done = true;
        for (uint s = 0; s < kThreadgroupWidth / 32u; ++s)
            all_done = all_done && s_done[s];
        if (all_done)
            break;
        const int idx = batch_start + int(tid);
        if (tid < kGsplatBatch && idx < range_end)
            gsplat_stage(p, p.gaussian_ids[idx], tid, s_xyzo, s_m0, s_m1, s_m2, s_quat, s_rgb);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint batch_size = uint(min(int(kGsplatBatch), range_end - batch_start));
        for (uint t = 0; t < batch_size && !done; ++t) {
            const float4 xyzo = s_xyzo[t];
            const float3x3 M = float3x3(s_m0[t].xyz, s_m1[t].xyz, s_m2[t].xyz);
            const float3 gro = M * (ray.origin - xyzo.xyz);
            const float3 grd = gsplat_safe_normalize(M * ray.dir);
            const float3 gcrod = cross(grd, gro);
            const float alpha = min(0.999f, xyzo.w * exp(-0.5f * dot(gcrod, gcrod)));
            if (alpha < kGsplatAlphaMin)
                continue;
            const float next_T = T * (1.0f - alpha);
            if (next_T <= 1e-4f) {
                done = true;
                break;
            }
            pix += s_rgb[t].xyz * (alpha * T);
            float3 mc=p.camera->R*xyzo.xyz+p.camera->t;
            depth+=(p.camera->model==kGsplatPinhole?mc.z:length(mc))*(alpha*T);
            if(p.normal) normal+=gsplat_geometry_normal(*p.camera,xyzo.xyz,float3(s_m0[t].w,s_m1[t].w,s_m2[t].w),s_quat[t])*(alpha*T);
            cur_idx = uint(batch_start) + t;
            T = next_T;
        }
    }
    if (!inside)
        return;
    const uint pixel = i * p.width + j;
    const uint plane = p.width * p.height;
    if(p.depth) p.depth[pixel]=(p.mode==2 || p.mode==4)?depth/max(1.f-T,1e-10f):depth;
    if(p.normal) for(uint c=0;c<3;++c) p.normal[c*plane+pixel]=normal[c];
    p.alpha[pixel] = 1.0f - T;
    if(p.image) for (uint c = 0; c < 3u; ++c)
        p.image[c * plane + pixel] = pix[c] + T * gsplat_background(p, c, pixel);
    p.last_ids[pixel] = int(cur_idx);
}

static void gsplat_atomic_add(device float* address, const float value) {
    atomic_fetch_add_explicit(reinterpret_cast<device atomic_float*>(address), value, memory_order_relaxed);
}

// One pixel of the backward walk.
struct GsplatPixel {
    float3 origin, dir, v_c, buffer, depth_axis, v_normal, normal_buffer;
    float3x3 camera_R;
    bool radial_depth;
    float v_depth, depth_buffer;
    float T, T_final, v_a, bg_accum, error, edge_weight;
    int bin_final;
    bool active;
};

static GsplatPixel gsplat_pixel(constant GsplatRasterParams& p, const uint i, const uint j) {
    const uint plane = p.width * p.height;
    const uint pixel = min(i * p.width + j, plane - 1u);
    const GsplatRay ray = gsplat_pixel_ray(*p.camera, float2(float(j) + 0.5f, float(i) + 0.5f));
    GsplatPixel px;
    px.camera_R=p.camera->R;
    px.radial_depth=p.camera->model!=kGsplatPinhole;
    px.depth_axis=float3(p.camera->R[0][2],p.camera->R[1][2],p.camera->R[2][2]);
    px.v_depth=p.v_depth?p.v_depth[pixel]:0;
    px.depth_buffer=0;
    px.v_normal=p.v_normal?float3(p.v_normal[pixel],p.v_normal[plane+pixel],p.v_normal[2*plane+pixel]):float3(0.f);
    px.normal_buffer=0;
    px.origin = ray.origin;
    px.dir = ray.dir;
    px.active = i < p.height && j < p.width && ray.valid;
    px.T_final = 1.0f - p.alpha[pixel];
    px.T = px.T_final;
    px.bin_final = px.active ? p.last_ids[pixel] : 0;
    px.v_c = float3(p.v_image[pixel], p.v_image[plane + pixel], p.v_image[2u * plane + pixel]);
    px.v_a = p.v_alpha[pixel];
    if(p.v_depth && (p.mode==2 || p.mode==4)) {
        px.v_depth/=max(p.alpha[pixel],1e-10f);
        if(p.alpha[pixel]>1e-10f) px.v_a-=px.v_depth*p.depth[pixel];
    }
    px.bg_accum = 0.0f;
    if (p.bg_image != nullptr || p.bg_color != nullptr) {
        for (uint c = 0; c < 3u; ++c)
            px.bg_accum += gsplat_background(p, c, pixel) * px.v_c[c];
    }
    px.error = p.error_map != nullptr ? p.error_map[pixel] : 0.0f;
    px.edge_weight = p.edge_map != nullptr ? p.edge_map[pixel] : 0.0f;
    px.buffer = float3(0.0f);
    return px;
}

struct GsplatGrad {
    float3 rgb, mean, scale;
    float4 quat;
    float opacity, dens_w, dens_e, edge;
};

// Un-blends one gaussian from a pixel and adds its gradient; false when the
// gaussian did not contribute to the pixel.
static bool gsplat_contribute(thread GsplatPixel& px, const int isect, const float4 xyzo, const float4 m0,
                              const float4 m1, const float4 m2, const float4 quat, const float3 rgb, const bool have_bg,
                              thread GsplatGrad& grad) {
    if (!px.active || isect > px.bin_final)
        return false;
    const float3x3 Mt = float3x3(m0.xyz, m1.xyz, m2.xyz);
    const float opac = xyzo.w;
    const float3 o_minus_mu = px.origin - xyzo.xyz;
    const float3 gro = Mt * o_minus_mu;
    const float3 grd = Mt * px.dir;
    const float3 grd_n = gsplat_safe_normalize(grd);
    const float3 gcrod = cross(grd_n, gro);
    const float power = -0.5f * dot(gcrod, gcrod);
    const float vis = exp(power);
    const float alpha = min(0.999f, opac * vis);
    if (power > 0.0f || alpha < kGsplatAlphaMin)
        return false;
    const float ra = 1.0f / (1.0f - alpha);
    px.T *= ra;
    const float fac = alpha * px.T;
    grad.rgb += fac * px.v_c;
    const float3 delta=xyzo.xyz-px.origin;
    const float depth=px.radial_depth?length(delta):dot(delta,px.depth_axis);
    grad.mean+=(fac*px.v_depth)*(px.radial_depth?normalize(delta):px.depth_axis);
    const uint axis=gsplat_normal_axis(float3(m0.w,m1.w,m2.w));
    const float3 world_normal=gsplat_quat_to_rotmat(quat)[axis];
    const float facing=dot(world_normal,delta)>0?-1.f:1.f;
    const float3 normal=px.camera_R*world_normal*facing;
    float3x3 v_R=float3x3(0.f);
    v_R[axis]=transpose(px.camera_R)*px.v_normal*(fac*facing);
    gsplat_quat_to_rotmat_vjp(quat,v_R,grad.quat);
    grad.dens_w += fac;
    grad.dens_e += fac * px.error;
    const float edge_contribution = fac * px.edge_weight;
    if (px.edge_weight > 0.0f && isfinite(edge_contribution))
        grad.edge += edge_contribution;
    float v_alpha = dot(rgb * px.T - px.buffer * ra, px.v_c) + px.T_final * ra * px.v_a;
    v_alpha+=(depth*px.T-px.depth_buffer*ra)*px.v_depth+dot(normal*px.T-px.normal_buffer*ra,px.v_normal);
    if (have_bg)
        v_alpha += -px.T_final * ra * px.bg_accum;
    if (opac * vis <= 0.999f) {
        const float v_gradDist = -0.5f * vis * (opac * v_alpha);
        const float3 v_gcrod = 2.0f * v_gradDist * gcrod;
        const float3 v_grd_n = -cross(v_gcrod, gro);
        const float3 v_gro = cross(v_gcrod, grd_n);
        const float3 v_grd = gsplat_safe_normalize_bw(grd, v_grd_n);
        const float3x3 v_Mt = gsplat_outer(v_grd, px.dir) + gsplat_outer(v_gro, o_minus_mu);
        grad.mean -= transpose(Mt) * v_gro;
        gsplat_preci_half_vjp(quat, float3(m0.w, m1.w, m2.w), gsplat_quat_to_rotmat(quat), transpose(v_Mt), grad.quat,
                              grad.scale);
        grad.opacity += vis * v_alpha;
    }
    px.buffer += rgb * fac;
    px.depth_buffer+=depth*fac;
    px.normal_buffer+=normal*fac;
    return true;
}

// Back to front over each tile, two horizontally adjacent pixels per thread
// as CUDA's dual kernel. All lanes of a simdgroup walk the same gaussian, so
// its gradient reduces in registers and lane 0 adds it once.
kernel void gsplat_rasterize_backward(constant GsplatRasterParams& p [[buffer(0)]],
                                      uint2 group [[threadgroup_position_in_grid]],
                                      uint2 local [[thread_position_in_threadgroup]],
                                      uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float4 s_xyzo[kGsplatBatch], s_m0[kGsplatBatch], s_m1[kGsplatBatch], s_m2[kGsplatBatch];
    threadgroup float4 s_quat[kGsplatBatch], s_rgb[kGsplatBatch];

    const uint tile_id = group.y * p.tiles_x + group.x;
    const uint i = group.y * kGsplatTile + local.y;
    const uint j = group.x * kGsplatTile + local.x * 2u;
    GsplatPixel px0 = gsplat_pixel(p, i, j);
    GsplatPixel px1 = gsplat_pixel(p, i, j + 1u);
    const bool have_bg = p.bg_image != nullptr || p.bg_color != nullptr;
    const bool do_dens = p.densification != nullptr && p.error_map != nullptr;
    const bool edge_scoring = p.edge_map != nullptr && p.edge_scores != nullptr;

    const int range_start = p.tile_offsets[tile_id];
    const int range_end = p.tile_offsets[tile_id + 1u];
    const int simd_bin_final = simd_max(max(px0.bin_final, px1.bin_final));
    for (int batch_end = range_end - 1; batch_end >= range_start; batch_end -= int(kGsplatBatch)) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int idx = batch_end - int(tid);
        if (tid < kGsplatBatch && idx >= range_start)
            gsplat_stage(p, p.gaussian_ids[idx], tid, s_xyzo, s_m0, s_m1, s_m2, s_quat, s_rgb);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const int batch_size = min(int(kGsplatBatch), batch_end + 1 - range_start);
        for (int t = max(0, batch_end - simd_bin_final); t < batch_size; ++t) {
            const float4 xyzo = s_xyzo[t], m0 = s_m0[t], m1 = s_m1[t], m2 = s_m2[t], quat = s_quat[t];
            const float3 rgb = s_rgb[t].xyz;
            GsplatGrad grad = {};
            const bool hit0 = gsplat_contribute(px0, batch_end - t, xyzo, m0, m1, m2, quat, rgb, have_bg, grad);
            const bool hit1 = gsplat_contribute(px1, batch_end - t, xyzo, m0, m1, m2, quat, rgb, have_bg, grad);
            if (!simd_any(hit0 || hit1))
                continue;
            const float values[kGsplatGradStride] = {
                simd_sum(grad.mean.x),  simd_sum(grad.mean.y),  simd_sum(grad.mean.z),  simd_sum(grad.scale.x),
                simd_sum(grad.scale.y), simd_sum(grad.scale.z), simd_sum(grad.quat.x),  simd_sum(grad.quat.y),
                simd_sum(grad.quat.z),  simd_sum(grad.quat.w),  simd_sum(grad.opacity), simd_sum(grad.rgb.x),
                simd_sum(grad.rgb.y),   simd_sum(grad.rgb.z)};
            const float dens_w = do_dens ? simd_sum(grad.dens_w) : 0.0f;
            const float dens_e = do_dens ? simd_sum(grad.dens_e) : 0.0f;
            const float edge = edge_scoring ? simd_sum(grad.edge) : 0.0f;
            if (simd_is_first()) {
                const uint g = as_type<uint>(s_rgb[t].w);
                device float* out = p.grads + g * kGsplatGradStride;
                for (uint k = 0; k < kGsplatGradStride; ++k)
                    gsplat_atomic_add(out + k, values[k]);
                if (do_dens) {
                    gsplat_atomic_add(p.densification + g, dens_w);
                    gsplat_atomic_add(p.densification + p.count + g, dens_e);
                }
                if (edge_scoring)
                    gsplat_atomic_add(p.edge_scores + g, edge);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Per-gaussian accumulation into the optimizer gradients: activation chain
// rules, SH backward into sh0 and swizzled SH-rest, the gradient-norm
// densification fallback and the projected screen share.

struct GsplatAccumulateParams {
    device const GsplatCamera* camera;
    device const float* means;
    device const float* scales;
    device const float* opacities;
    device const float* grads;
    device const int* radii;
    device const float* means2d;
    device float* means_grad;
    device float* scaling_grad;
    device float* rotation_grad;
    device float* opacity_grad;
    device float* sh0_grad;
    device float* sh_rest_grad; // swizzled float, or null
    device float* grad_norms;   // densification row 0, or null
    device float* shares;       // [N] or null
    uint count;
    uint degree;
    uint layout_rest;
    uint width, height;
    float flatten_weight;
};

kernel void gsplat_accumulate(constant GsplatAccumulateParams& p [[buffer(0)]], uint g [[thread_position_in_grid]]) {
    if (g >= p.count)
        return;
    device const float* v = p.grads + g * kGsplatGradStride;
    const float3 scale = exp(gsplat_load3(p.scales, g));
    const float opacity = gsplat_sigmoid(p.opacities[g]);
    for (uint k = 0; k < 3u; ++k) {
        p.means_grad[g * 3u + k] += v[k];
        p.scaling_grad[g * 3u + k] += v[3u + k] * scale[k];
    }
    uint axis=gsplat_normal_axis(scale);
    p.scaling_grad[g*3+axis]+=p.flatten_weight*scale[axis]/float(p.count);
    for (uint k = 0; k < 4u; ++k)
        p.rotation_grad[g * 4u + k] += v[6u + k];
    p.opacity_grad[g] += v[10] * opacity * (1.0f - opacity);

    float basis[16];
    gsplat_sh_basis(gsplat_load3(p.means, g) - p.camera->origin, p.degree, basis);
    const uint bases = (p.degree + 1u) * (p.degree + 1u);
    for (uint c = 0; c < 3u; ++c) {
        const float v_color = v[11u + c];
        p.sh0_grad[g * 3u + c] += basis[0] * v_color;
        for (uint k = 1; k < bases && p.sh_rest_grad != nullptr; ++k)
            p.sh_rest_grad[gsplat_sh_float_index(g, k - 1u, c, p.layout_rest)] += basis[k] * v_color;
    }

    if (p.grad_norms != nullptr)
        p.grad_norms[g] += sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    const int rx = p.radii[g * 2u], ry = p.radii[g * 2u + 1u];
    if (p.shares != nullptr && rx > 0 && ry > 0) {
        const float x = p.means2d[g * 2u], y = p.means2d[g * 2u + 1u];
        const float w = float(p.width), h = float(p.height);
        const float dx = fmax(0.0f, fmin(w, x + float(rx)) - fmax(0.0f, x - float(rx)));
        const float dy = fmax(0.0f, fmin(h, y + float(ry)) - fmax(0.0f, y - float(ry)));
        p.shares[g] = fmax(p.shares[g], (dx / w) * (dy / h));
    }
}
