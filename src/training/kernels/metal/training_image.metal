// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

// TrainingImageOps: ports of camera_loss_heatmap.cu, roi_weight_map.cu,
// grad_alpha.cu (resize, random background) and image_kernels.cu.

struct HeatmapParams {
    device const float* loss;
    device float* latest;
    device float* ema;
    int slot;
    uint slot_count;
    float ema_alpha;
};

kernel void training_image_heatmap(constant HeatmapParams& p [[buffer(0)]]) {
    if (p.slot < 0 || uint(p.slot) >= p.slot_count)
        return;
    const float loss = p.loss[0];
    p.latest[p.slot] = loss;
    const float previous = p.ema[p.slot];
    p.ema[p.slot] = previous >= 0.0f ? previous + p.ema_alpha * (loss - previous) : loss;
}

struct RoiParams {
    device const float* world_to_camera;
    device const float* camera_position;
    device float* output;
    device const float* camera_rays;   // [H,W,3] or null: pinhole rays from the intrinsics
    device const float* surface_depth; // [H,W] or null
    device const float* surface_alpha; // [H,W] or null
    float4 world_to_cropbox[3];
    float3 crop_min;
    float3 crop_max;
    float fx, fy, cx, cy;
    int width, height;
    float outside_weight;
    uint inverse;
    device const float* polygon;
    uint polygon_count;
};

static float3 affine_point(constant float4* rows, const float3 point) {
    return float3(dot(rows[0].xyz, point) + rows[0].w, dot(rows[1].xyz, point) + rows[1].w,
                  dot(rows[2].xyz, point) + rows[2].w);
}

static float3 affine_direction(constant float4* rows, const float3 direction) {
    return float3(dot(rows[0].xyz, direction), dot(rows[1].xyz, direction), dot(rows[2].xyz, direction));
}

static bool ray_hits_box(const float3 origin, const float3 direction, const float3 lo, const float3 hi) {
    float t_enter = -FLT_MAX;
    float t_exit = FLT_MAX;
    for (int axis = 0; axis < 3; ++axis) {
        if (fabs(direction[axis]) < 1.0e-8f) {
            if (origin[axis] < lo[axis] || origin[axis] > hi[axis])
                return false;
            continue;
        }
        const float inv = 1.0f / direction[axis];
        float t0 = (lo[axis] - origin[axis]) * inv;
        float t1 = (hi[axis] - origin[axis]) * inv;
        if (t0 > t1) {
            const float swap = t0;
            t0 = t1;
            t1 = swap;
        }
        t_enter = fmax(t_enter, t0);
        t_exit = fmin(t_exit, t1);
        if (t_exit < t_enter)
            return false;
    }
    return t_exit >= 0.0f;
}

static float2 polygonVertex(constant RoiParams& p, uint i) {
    device const float* v = p.polygon;
    return float2(v[2u * i], v[2u * i + 1u]);
}
static bool outlineContains(constant RoiParams& p, float2 q) {
    bool inside = false;
    float2 b = polygonVertex(p, p.polygon_count - 1u);
    for (uint i = 0; i < p.polygon_count; ++i) {
        float2 a = polygonVertex(p, i);
        if ((a.y > q.y) != (b.y > q.y) && q.x < (b.x - a.x) * (q.y - a.y) / (b.y - a.y) + a.x) inside = !inside;
        b = a;
    }
    return inside;
}
static float cross2(float2 a, float2 b) { return a.x * b.y - a.y * b.x; }
// The ray (t >= 0) clipped to the slab and the outline bounds must leave a segment that
// touches the outline: an endpoint inside it, or a crossing with one of its edges.
static bool rayHitsPrism(constant RoiParams& p, float3 origin, float3 direction) {
    float enter = 0.0f, leave = 3.402823466e+38f;
    for (uint axis = 0; axis < 3; ++axis) {
        if (fabs(direction[axis]) < 1.0e-8f) {
            if (origin[axis] < p.crop_min[axis] || origin[axis] > p.crop_max[axis]) return false;
        } else {
            float inverseDirection = 1.0f / direction[axis];
            float a = (p.crop_min[axis] - origin[axis]) * inverseDirection;
            float b = (p.crop_max[axis] - origin[axis]) * inverseDirection;
            if (a > b) { float t = a; a = b; b = t; }
            enter = max(enter, a); leave = min(leave, b);
            if (leave < enter) return false;
        }
    }
    if (leave == 3.402823466e+38f) leave = enter;
    float2 p0 = origin.xy + enter * direction.xy, p1 = origin.xy + leave * direction.xy;
    if (outlineContains(p, p0) || outlineContains(p, p1)) return true;
    float2 d = p1 - p0;
    float2 a = polygonVertex(p, p.polygon_count - 1u);
    for (uint i = 0; i < p.polygon_count; ++i) {
        float2 b = polygonVertex(p, i), e = b - a;
        if (((cross2(d, a - p0) > 0.0f) != (cross2(d, b - p0) > 0.0f)) &&
            ((cross2(e, p0 - a) > 0.0f) != (cross2(e, p1 - a) > 0.0f))) return true;
        a = b;
    }
    return false;
}

kernel void training_image_roi(constant RoiParams& p [[buffer(0)]], uint index [[thread_position_in_grid]]) {
    if (index >= uint(p.width) * uint(p.height))
        return;
    const int x = int(index % uint(p.width));
    const int y = int(index / uint(p.width));
    float3 camera = float3((float(x) + 0.5f - p.cx) / p.fx, (float(y) + 0.5f - p.cy) / p.fy, 1.0f);
    if (p.camera_rays != nullptr) {
        camera = float3(p.camera_rays[3u * index], p.camera_rays[3u * index + 1u], p.camera_rays[3u * index + 2u]);
        // The lens has no ray for this pixel.
        if (!all(isfinite(camera))) {
            p.output[index] = p.inverse != 0 ? 1.0f : p.outside_weight;
            return;
        }
    }
    device const float* m = p.world_to_camera;
    const float3 world = float3(m[0] * camera.x + m[4] * camera.y + m[8] * camera.z,
                                m[1] * camera.x + m[5] * camera.y + m[9] * camera.z,
                                m[2] * camera.x + m[6] * camera.y + m[10] * camera.z);
    const float3 origin = float3(p.camera_position[0], p.camera_position[1], p.camera_position[2]);
    const float3 origin_in_box = affine_point(p.world_to_cropbox, origin);
    const float3 direction_in_box = affine_direction(p.world_to_cropbox, world);
    bool hit;
    if (p.surface_depth != nullptr && p.surface_alpha != nullptr) {
        // Below half coverage a pixel shows mostly background, so no surface in the box.
        const float alpha = p.surface_alpha[index];
        const float3 point = origin_in_box + (p.surface_depth[index] / alpha) * direction_in_box;
        hit = alpha >= 0.5f && all(point >= p.crop_min) && all(point <= p.crop_max) &&
              (p.polygon_count == 0u || outlineContains(p, point.xy));
    } else {
        hit = p.polygon_count != 0u ? rayHitsPrism(p, origin_in_box, direction_in_box)
                                   : ray_hits_box(origin_in_box, direction_in_box, p.crop_min, p.crop_max);
    }
    p.output[index] = (p.inverse != 0 ? !hit : hit) ? 1.0f : p.outside_weight;
}

struct ResizeParams {
    device const float* source;
    device float* destination;
    int channels;
    int source_h, source_w;
    int destination_h, destination_w;
};

kernel void training_image_resize(constant ResizeParams& p [[buffer(0)]], uint index [[thread_position_in_grid]]) {
    const int total = p.destination_h * p.destination_w;
    if (int(index) >= total)
        return;
    const int dy = int(index) / p.destination_w;
    const int dx = int(index) % p.destination_w;
    float sy = (float(dy) + 0.5f) * (float(p.source_h) / float(p.destination_h)) - 0.5f;
    float sx = (float(dx) + 0.5f) * (float(p.source_w) / float(p.destination_w)) - 0.5f;
    sy = fmax(0.0f, fmin(sy, float(p.source_h - 1)));
    sx = fmax(0.0f, fmin(sx, float(p.source_w - 1)));
    const int y0 = int(sy);
    const int x0 = int(sx);
    const int y1 = min(y0 + 1, p.source_h - 1);
    const int x1 = min(x0 + 1, p.source_w - 1);
    const float wy = sy - float(y0);
    const float wx = sx - float(x0);
    const float w00 = (1.0f - wy) * (1.0f - wx);
    const float w01 = (1.0f - wy) * wx;
    const float w10 = wy * (1.0f - wx);
    const float w11 = wy * wx;
    const int source_plane = p.source_h * p.source_w;
    for (int c = 0; c < p.channels; ++c) {
        device const float* s = p.source + c * source_plane;
        p.destination[c * total + index] = w00 * s[y0 * p.source_w + x0] + w01 * s[y0 * p.source_w + x1] +
                                           w10 * s[y1 * p.source_w + x0] + w11 * s[y1 * p.source_w + x1];
    }
}

static uint pcg_hash(const uint v) {
    const uint state = v * 747796405u + 2891336453u;
    const uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

static float unit_float(const uint v) { return float(v) * (1.0f / 4294967296.0f); }

struct RandomBackgroundParams {
    device float* output;
    int plane;
    uint seed;
};

kernel void training_image_random_background(constant RandomBackgroundParams& p [[buffer(0)]],
                                             uint index [[thread_position_in_grid]]) {
    if (int(index) >= p.plane)
        return;
    const uint base = p.seed ^ index;
    p.output[index] = unit_float(pcg_hash(base));
    p.output[p.plane + index] = unit_float(pcg_hash(base + 0x9E3779B9u));
    p.output[2 * p.plane + index] = unit_float(pcg_hash(base + 0x6C8E9CF9u));
}

struct CannyParams {
    device const void* input;
    device float* output;
    device const float* mask;
    int height, width;
};

constant uint kCannyBytes [[function_constant(0)]];

constant float kCannyBlur[25] = {
    2.f / 159.f, 4.f / 159.f, 5.f / 159.f, 4.f / 159.f, 2.f / 159.f,
    4.f / 159.f, 9.f / 159.f, 12.f / 159.f, 9.f / 159.f, 4.f / 159.f,
    5.f / 159.f, 12.f / 159.f, 15.f / 159.f, 12.f / 159.f, 5.f / 159.f,
    4.f / 159.f, 9.f / 159.f, 12.f / 159.f, 9.f / 159.f, 4.f / 159.f,
    2.f / 159.f, 4.f / 159.f, 5.f / 159.f, 4.f / 159.f, 2.f / 159.f};
constant float kCannySobel[9] = {-1.0f, 0.0f, 1.0f, -2.0f, 0.0f, 2.0f, -1.0f, 0.0f, 1.0f};

static float canny_value(constant CannyParams& p, const int index) {
    if (kCannyBytes != 0)
        return float(static_cast<device const uchar*>(p.input)[index]) * (1.0f / 255.0f);
    return static_cast<device const float*>(p.input)[index];
}

// 32x32 threadgroups with a 4-pixel halo, as the CUDA kernel.
kernel void training_image_canny(constant CannyParams& p [[buffer(0)]],
                                 uint2 group [[threadgroup_position_in_grid]],
                                 uint2 local [[thread_position_in_threadgroup]]) {
    constexpr int kBlock = 32;
    constexpr int kHalo = 4;
    constexpr int kHalo1 = 2;
    constexpr int kHalo2 = 1;
    constexpr int kPixels = kBlock + 2 * kHalo;
    constexpr int kBlurred = kBlock + 2 * kHalo1;
    constexpr int kFiltered = kBlock + 2 * kHalo2;
    const int x_id = int(group.x) * kBlock + int(local.x);
    const int y_id = int(group.y) * kBlock + int(local.y);
    const int plane = p.height * p.width;
    const int lane = int(local.y) * kBlock + int(local.x);

    threadgroup float pixels[kPixels][kPixels];
    for (int batch = 0; batch < kPixels * kPixels; batch += kBlock * kBlock) {
        const int t = batch + lane;
        if (t < kPixels * kPixels) {
            const int y = t / kPixels;
            const int x = t % kPixels;
            const int yi = min(max(int(group.y) * kBlock + y - kHalo, 0), p.height - 1);
            const int xi = min(max(int(group.x) * kBlock + x - kHalo, 0), p.width - 1);
            const int i = yi * p.width + xi;
            const bool keep = p.mask == nullptr || p.mask[i] > 0.0f;
            pixels[y][x] = keep ? 0.299f * canny_value(p, i) + 0.587f * canny_value(p, i + plane) +
                                      0.114f * canny_value(p, i + 2 * plane)
                                : 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float blurred[kBlurred][kBlurred];
    for (int batch = 0; batch < kBlurred * kBlurred; batch += kBlock * kBlock) {
        const int t = batch + lane;
        if (t >= kBlurred * kBlurred)
            continue;
        const int y = t / kBlurred;
        const int x = t % kBlurred;
        float total = 0.0f;
        for (int cy = -2; cy <= 2; ++cy)
            for (int cx = -2; cx <= 2; ++cx)
                total += kCannyBlur[(cy + 2) * 5 + (cx + 2)] * pixels[y - kHalo1 + cy + kHalo][x - kHalo1 + cx + kHalo];
        blurred[y][x] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float2 filtered[kFiltered][kFiltered];
    for (int batch = 0; batch < kFiltered * kFiltered; batch += kBlock * kBlock) {
        const int t = batch + lane;
        if (t >= kFiltered * kFiltered)
            continue;
        const int y = t / kFiltered;
        const int x = t % kFiltered;
        float total1 = 0.0f;
        float total2 = 0.0f;
        for (int cy = -1; cy <= 1; ++cy) {
            for (int cx = -1; cx <= 1; ++cx) {
                const float value = blurred[y - kHalo2 + cy + kHalo1][x - kHalo2 + cx + kHalo1];
                total1 += kCannySobel[(cy + 1) * 3 + (cx + 1)] * value;
                total2 += kCannySobel[(cx + 1) * 3 + (cy + 1)] * value;
            }
        }
        filtered[y][x] = float2(total1, total2);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float2 gradient = filtered[local.y + kHalo2][local.x + kHalo2];
    float magnitude = length(gradient);
    if (magnitude > 0.0f) {
        const int dx = min(max(int(round(gradient.x / magnitude)), -kHalo2), kHalo2);
        const int dy = min(max(int(round(gradient.y / magnitude)), -kHalo2), kHalo2);
        const float2 forward = filtered[int(local.y) + dy + kHalo2][int(local.x) + dx + kHalo2];
        const float2 backward = filtered[int(local.y) - dy + kHalo2][int(local.x) - dx + kHalo2];
        if (magnitude < length(forward) || magnitude < length(backward))
            magnitude = 0.0f;
    }
    if (y_id < p.height && x_id < p.width) {
        if (p.mask != nullptr && p.mask[y_id * p.width + x_id] <= 0.0f) {
            p.output[y_id * p.width + x_id] = 0.0f;
            return;
        }
        p.output[y_id * p.width + x_id] = magnitude;
    }
}

struct NormalizeScalarParams {
    device float* values;
    device const float* scalar;
    uint count;
    float skip_below;
};

kernel void training_image_normalize_scalar(constant NormalizeScalarParams& p [[buffer(0)]],
                                            uint index [[thread_position_in_grid]]) {
    const float value = p.scalar[0];
    if (value <= p.skip_below || index >= p.count)
        return;
    p.values[index] /= fmax(value, 1e-9f);
}
