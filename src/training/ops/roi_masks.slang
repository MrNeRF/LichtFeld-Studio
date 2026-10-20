// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
// Per-splat crop-box ROI masks: a splat is relevant when the ray from some listed camera
// through its centre meets the box, i.e. it lies inside, in front of or behind the box.
struct Parameters {
    Ptr<float, Access.Read> means, scales, model, cameras;
    Ptr<uint8_t, Access.Read> deleted;
    Ptr<uint8_t, Access.ReadWrite> rejected, in_roi;
    float lo_x, lo_y, lo_z, hi_x, hi_y, hi_z, margin_scale;
    uint count, camera_count, inverse;
};
[[vk::push_constant]] ConstantBuffer<Parameters> p;

// model holds model_to_cropbox as three row-major rows of four.
float3 model_to_box(float3 v) {
    return float3(p.model[0] * v.x + p.model[1] * v.y + p.model[2] * v.z + p.model[3],
                  p.model[4] * v.x + p.model[5] * v.y + p.model[6] * v.z + p.model[7],
                  p.model[8] * v.x + p.model[9] * v.y + p.model[10] * v.z + p.model[11]);
}

bool ray_meets_box(float3 origin, float3 direction, float3 lo, float3 hi) {
    float enter = -3.402823466e38f, exit = 3.402823466e38f;
    [unroll] for (uint axis = 0; axis < 3; ++axis) {
        if (abs(direction[axis]) < 1.0e-8f) {
            if (origin[axis] < lo[axis] || origin[axis] > hi[axis])
                return false;
            continue;
        }
        float inverse_direction = 1.0f / direction[axis];
        float t0 = (lo[axis] - origin[axis]) * inverse_direction;
        float t1 = (hi[axis] - origin[axis]) * inverse_direction;
        enter = max(enter, min(t0, t1));
        exit = min(exit, max(t0, t1));
        if (exit < enter)
            return false;
    }
    return exit >= 0.0f;
}

// margin grows the box by the splat footprint (shrinks it when negative).
bool sees_box(float3 camera, float3 centre, float margin) {
    float3 lo = float3(p.lo_x, p.lo_y, p.lo_z) - margin;
    float3 hi = float3(p.hi_x, p.hi_y, p.hi_z) + margin;
    if (any(lo > hi))
        return false;
    return ray_meets_box(camera, centre - camera, lo, hi);
}

[shader("compute")][numthreads(256, 1, 1)]
void reject(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    if (i >= p.count)
        return;
    if ((uint64_t)p.deleted != 0 && p.deleted[i] != 0) {
        p.rejected[i] = 0;
        if ((uint64_t)p.in_roi != 0)
            p.in_roi[i] = 0;
        return;
    }
    float3 centre = model_to_box(float3(p.means[3 * i], p.means[3 * i + 1], p.means[3 * i + 2]));
    bool inverse = p.inverse != 0;
    if ((uint64_t)p.in_roi != 0) {
        bool inside = all(centre >= float3(p.lo_x, p.lo_y, p.lo_z)) && all(centre <= float3(p.hi_x, p.hi_y, p.hi_z));
        p.in_roi[i] = inside != inverse ? uint8_t(1) : uint8_t(0);
    }
    // Three standard deviations of the largest axis, in box units. Inverse boxes shrink
    // instead, so the splats on their rim train with the outside.
    float margin = 0.0f;
    if ((uint64_t)p.scales != 0)
        margin = 3.0f * exp(max(p.scales[3 * i], max(p.scales[3 * i + 1], p.scales[3 * i + 2]))) * p.margin_scale;
    if (inverse)
        margin = -margin;
    bool relevant = false;
    for (uint c = 0; c < p.camera_count && !relevant; ++c) {
        float3 camera = float3(p.cameras[3 * c], p.cameras[3 * c + 1], p.cameras[3 * c + 2]);
        relevant = sees_box(camera, centre, margin) != inverse;
    }
    p.rejected[i] = relevant ? uint8_t(0) : uint8_t(1);
}
