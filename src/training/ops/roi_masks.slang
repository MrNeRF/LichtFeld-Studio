// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
// Per-splat crop-box ROI masks: a splat is relevant when the ray from some listed camera
// through its centre meets the box, i.e. it lies inside, in front of or behind the box.
// With polygon_count > 0 the region is a crop polygon prism instead: the outline (polygon,
// x and y pairs) in the local XY plane over |z| <= hi_z, which lo and hi bound.
struct Parameters {
    Ptr<float, Access.Read> means, scales, model, cameras;
    Ptr<uint8_t, Access.Read> deleted;
    Ptr<uint8_t, Access.ReadWrite> rejected, in_roi;
    Ptr<float, Access.Read> polygon;
    float lo_x, lo_y, lo_z, hi_x, hi_y, hi_z, margin_scale;
    uint count, camera_count, inverse, polygon_count, padding;
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

float2 polygon_vertex(uint i) {
    return float2(p.polygon[2 * i], p.polygon[2 * i + 1]);
}

// The even-odd rule of crop_polygon::containsPoint.
bool outline_contains(float2 q) {
    bool inside = false;
    float2 b = polygon_vertex(p.polygon_count - 1);
    for (uint i = 0; i < p.polygon_count; ++i) {
        float2 a = polygon_vertex(i);
        if ((a.y > q.y) != (b.y > q.y) && q.x < (b.x - a.x) * (q.y - a.y) / (b.y - a.y) + a.x)
            inside = !inside;
        b = a;
    }
    return inside;
}

bool prism_contains(float3 q) {
    return abs(q.z) <= p.hi_z && outline_contains(q.xy);
}

float cross2(float2 a, float2 b) {
    return a.x * b.y - a.y * b.x;
}

float point_segment_distance_sq(float2 q, float2 a, float2 b) {
    float2 ab = b - a;
    float2 aq = q - a;
    float length_sq = dot(ab, ab);
    float t = length_sq > 0.0f ? clamp(dot(aq, ab) / length_sq, 0.0f, 1.0f) : 0.0f;
    float2 d = aq - t * ab;
    return dot(d, d);
}

// Whether the segment p0-p1 touches the outline region: an endpoint lies inside it by the
// even-odd rule, or the segment crosses an edge. With a positive margin, passing within margin
// of an edge counts too. Mirrors segment_meets_outline in kernels/roi_weight_map.cu.
bool segment_meets_outline(float2 p0, float2 p1, float margin) {
    if (outline_contains(p0) || outline_contains(p1))
        return true;
    float2 d = p1 - p0;
    float margin_sq = margin * margin;
    float2 a = polygon_vertex(p.polygon_count - 1);
    for (uint i = 0; i < p.polygon_count; ++i) {
        float2 b = polygon_vertex(i);
        float2 e = b - a;
        bool edge_straddles = (cross2(d, a - p0) > 0.0f) != (cross2(d, b - p0) > 0.0f);
        bool segment_straddles = (cross2(e, p0 - a) > 0.0f) != (cross2(e, p1 - a) > 0.0f);
        if (edge_straddles && segment_straddles)
            return true;
        if (margin > 0.0f &&
            min(min(point_segment_distance_sq(p0, a, b), point_segment_distance_sq(p1, a, b)),
                min(point_segment_distance_sq(a, p0, p1), point_segment_distance_sq(b, p0, p1))) <= margin_sq)
            return true;
        a = b;
    }
    return false;
}

// Whether the ray (t >= 0) passes through the prism grown by margin_xy around the outline and
// by margin_z along the slab (shrunk when negative): clipped to the slab and the outline bounds,
// the segment left must meet the outline. Mirrors ray_meets_prism in kernels/roi_weight_map.cu.
bool ray_meets_prism(float3 origin, float3 direction, float margin_xy, float margin_z) {
    float3 lo = float3(p.lo_x - margin_xy, p.lo_y - margin_xy, -p.hi_z - margin_z);
    float3 hi = float3(p.hi_x + margin_xy, p.hi_y + margin_xy, p.hi_z + margin_z);
    float enter = 0.0f, exit = 3.402823466e38f;
    [unroll] for (uint axis = 0; axis < 3; ++axis) {
        if (lo[axis] > hi[axis])
            return false;
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
    if (exit == 3.402823466e38f)
        exit = enter;
    return segment_meets_outline(origin.xy + enter * direction.xy, origin.xy + exit * direction.xy, margin_xy);
}

// sees_box for the prism: the outline and slab grow by the margin; a negative margin (inverse
// polygons) shrinks only the slab, as eroding a concave outline per splat costs far more.
bool sees_prism(float3 camera, float3 centre, float margin) {
    return ray_meets_prism(camera, centre - camera, max(margin, 0.0f), margin);
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
    bool prism = p.polygon_count != 0;
    if ((uint64_t)p.in_roi != 0) {
        bool inside = prism ? prism_contains(centre)
                            : all(centre >= float3(p.lo_x, p.lo_y, p.lo_z)) && all(centre <= float3(p.hi_x, p.hi_y, p.hi_z));
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
        relevant = (prism ? sees_prism(camera, centre, margin) : sees_box(camera, centre, margin)) != inverse;
    }
    p.rejected[i] = relevant ? uint8_t(0) : uint8_t(1);
}
