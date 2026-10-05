// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

static const float kPi = 3.14159265358979323846f;

// Column-major 4x4 matrices, as the native host writes them.
struct Matrix4 {
    float4 c0, c1, c2, c3;
};
float4 mul4(Matrix4 m, float4 v) { return m.c0 * v.x + m.c1 * v.y + m.c2 * v.z + m.c3 * v.w; }
Matrix4 mul44(Matrix4 a, Matrix4 b) {
    Matrix4 r;
    r.c0 = mul4(a, b.c0);
    r.c1 = mul4(a, b.c1);
    r.c2 = mul4(a, b.c2);
    r.c3 = mul4(a, b.c3);
    return r;
}
// Upper-left 3x3 of a column-major matrix applied to v.
float3 mul3(Matrix4 m, float3 v) { return m.c0.xyz * v.x + m.c1.xyz * v.y + m.c2.xyz * v.z; }

struct SelectionParameters {
    Matrix4 world_to_camera;
    float4 intrinsics;
    uint4 image, source, scene, payload, aabb;
    float4 ring;
};
struct Parameters {
    Ptr<SelectionParameters, Access.Read> parameters;
    Ptr<float, Access.Read> means;
    Ptr<float, Access.Read> scales;
    Ptr<float4, Access.Read> rotations;
    Ptr<float, Access.Read> opacity;
    Ptr<uint8_t, Access.Read> deleted;
    Ptr<Matrix4, Access.Read> transforms;
    Ptr<int, Access.Read> indices;
    Ptr<uint8_t, Access.Read> visibility;
    Ptr<float4, Access.Read> primitives;
    Ptr<float2, Access.Read> polygon_vertices;
    Ptr<uint8_t, Access.ReadWrite> polygon_mask;
    Ptr<uint8_t, Access.ReadWrite> output;
    Ptr<uint, Access.ReadWrite> pick;
};
[[vk::push_constant]] ConstantBuffer<Parameters> p;
groupshared float2 polygon_cached[2048];

float3 rotate(float4 q, float3 v) {
    return v + 2.f * cross(q.yzw, cross(q.yzw, v) + q.x * v);
}
float3 camera_project(float3 view, SelectionParameters parameters) {
    if (parameters.image.z == 2u) {
        const float length_view = length(view);
        if (!isfinite(length_view) || length_view <= 1e-7f)
            return float3(0);
        const float3 direction = view / length_view;
        return float3((float2(atan2(direction.x, direction.z) / (2.f * kPi), asin(clamp(direction.y, -1.f, 1.f)) / kPi) + .5f) * float2(parameters.image.xy), length_view);
    }
    return float3(parameters.intrinsics.xy * view.xy / (parameters.image.z == 1u ? 1.f : view.z) + parameters.intrinsics.zw, view.z);
}
bool in_margin(float2 point, SelectionParameters parameters, float margin) {
    const float2 extent = float2(parameters.image.xy);
    return all(point >= -margin * extent) && all(point < (1.f + margin) * extent);
}
float3 cap_covariance(float3 covariance, float opacity) {
    const float power = max(4.f, log(max(opacity, .5f / 255.f + 1e-8f) * 510.f));
    const float maximum = 512.5f * 512.5f / (2.f * power);
    const float average = .5f * (covariance.x + covariance.z);
    const float delta = sqrt(max(0.f, average * average - (covariance.x * covariance.z - covariance.y * covariance.y)));
    const float e1 = average + delta, e2 = max(average - delta, 0.f);
    const float c1 = min(e1, maximum), c2 = min(e2, maximum);
    if (c1 >= e1 && c2 >= e2)
        return covariance;
    const float2 a = abs(covariance.y) > 1e-6f ? normalize(float2(covariance.y, e1 - covariance.x)) : (covariance.x >= covariance.z ? float2(1, 0) : float2(0, 1));
    const float2 b = float2(a.y, -a.x);
    return float3(c1 * a.x * a.x + c2 * b.x * b.x, c1 * a.x * a.y + c2 * b.x * b.y, c1 * a.y * a.y + c2 * b.y * b.y);
}
float3 load3(Ptr<float, Access.Read> data, uint index) {
    return float3(data[index * 3u], data[index * 3u + 1u], data[index * 3u + 2u]);
}
float3 load_half3(Ptr<float, Access.Read> data, uint index) {
    const Ptr<half, Access.Read> h = (Ptr<half, Access.Read>)data;
    return float3(float(h[index * 3u]), float(h[index * 3u + 1u]), float(h[index * 3u + 2u]));
}
float4 load_rotation(Ptr<float4, Access.Read> data, uint index, bool half_attributes) {
    if (!half_attributes)
        return data[index];
    const Ptr<half, Access.Read> h = (Ptr<half, Access.Read>)data;
    return float4(float(h[index * 4u]), float(h[index * 4u + 1u]), float(h[index * 4u + 2u]), float(h[index * 4u + 3u]));
}
// Queries deliberately use center geometry, including zero-opacity splats for
// brush/rectangle/polygon, as in the shared editor's selection_mask contract.
bool project_query(uint id, SelectionParameters parameters, bool ring,
                   inout float2 center, inout float3 conic, inout float opacity, inout float depth) {
    if (id < parameters.scene.w && p.deleted[id] != 0u)
        return false;
    int object = parameters.scene.y != 0u ? p.indices[id] : 0;
    if (parameters.scene.y != 0u && parameters.scene.z != 0u && (object < 0 || uint(object) >= parameters.scene.z || p.visibility[object] == 0u))
        return false;
    Matrix4 model;
    model.c0 = float4(1, 0, 0, 0);
    model.c1 = float4(0, 1, 0, 0);
    model.c2 = float4(0, 0, 1, 0);
    model.c3 = float4(0, 0, 0, 1);
    if (parameters.scene.x != 0u)
        model = p.transforms[uint(clamp(object, 0, int(parameters.scene.x) - 1))];
    const float3 mean = load3(p.means, id);
    const Matrix4 matrix = mul44(parameters.world_to_camera, model);
    const float3 world = mul4(model, float4(mean, 1)).xyz;
    const float3 view = mul4(parameters.world_to_camera, float4(world, 1)).xyz;
    const float3 projected = camera_project(view, parameters);
    if (projected.z <= parameters.ring.y || !all(isfinite(projected)))
        return false;
    center = projected.xy - .5f;
    depth = projected.z;
    if (parameters.image.w == 0u && !ring)
        return true;
    float4 q = load_rotation(p.rotations, id, parameters.payload.x != 0u);
    const float norm2 = dot(q, q);
    if (!isfinite(norm2) || (!ring && norm2 <= 0))
        return false;
    q = norm2 > (ring ? 1e-8f : 0.f) ? q * rsqrt(norm2) : float4(1, 0, 0, 0);
    const float3 raw = parameters.payload.x != 0u ? load_half3(p.scales, id) : load3(p.scales, id);
    const float3 scale = exp(ring ? min(raw, float3(20.f)) : raw);
    if (!all(isfinite(scale)))
        return false;
    const float3 a = mul3(matrix, rotate(q, float3(scale.x, 0, 0)));
    const float3 bb = mul3(matrix, rotate(q, float3(0, scale.y, 0)));
    const float3 c = mul3(matrix, rotate(q, float3(0, 0, scale.z)));
    float3 covariance = 0;
    if (parameters.image.w != 0u) {
        const float lambda = .1f * .1f * 3.f - 3.f, denominator = 3.f + lambda;
        const float sigma_scale = sqrt(denominator), mean_weight = lambda / denominator, sigma_weight = 1.f / (2.f * denominator);
        const float covariance_weight = mean_weight + (1.f - .1f * .1f + 2.f);
        // Keep the shared editor's source -> object -> camera evaluation order.
        // Combining matrices before constructing sigma points changes narrow
        // ring hits after the large signed UT weights amplify FP32 rounding.
        const float3 axes[3] = {rotate(q, float3(scale.x, 0, 0)), rotate(q, float3(0, scale.y, 0)), rotate(q, float3(0, 0, scale.z))};
        float2 points[7];
        for (uint n = 0; n < 7; ++n) {
            const float3 source = n != 0u ? mean + (n <= 3u ? 1.f : -1.f) * sigma_scale * axes[(n - 1u) % 3u] : mean;
            const float3 sigma_world = mul4(model, float4(source, 1)).xyz;
            const float3 sample = camera_project(mul4(parameters.world_to_camera, float4(sigma_world, 1)).xyz, parameters);
            if (sample.z <= 0 || !all(isfinite(sample)) || !in_margin(sample.xy, parameters, .1f))
                return false;
            points[n] = sample.xy;
            if (parameters.image.z == 2u && n != 0u)
                points[n].x -= float(parameters.image.x) * round((points[n].x - points[0].x) / float(parameters.image.x));
        }
        // Accumulate offsets to avoid subtracting ~100 times the image center
        // from six large terms. The UT weights sum to one analytically.
        float2 offset = 0;
        for (uint n = 1; n < 7; ++n)
            offset += points[n] - points[0];
        const float2 mean2d_unwrapped = fma(float2(sigma_weight), offset, points[0]);
        float2 mean2d = mean2d_unwrapped;
        if (parameters.image.z == 1u) {
            // A linear orthographic projection preserves the Gaussian mean.
            mean2d = projected.xy;
        } else if (parameters.image.z == 0u) {
            // Pair the +/- rational projections analytically. Subtracting two
            // large pixel coordinates loses subpixel curvature for tiny splats.
            // sum(pair-base*2) = 2*dz*(xy*dz-z*dxy)/(z*(z*z-dz*dz)).
            float2 correction = 0;
            const float3 view_axes[3] = {sigma_scale * a, sigma_scale * bb, sigma_scale * c};
            for (uint axis = 0; axis < 3; ++axis) {
                const float3 d = view_axes[axis];
                correction += 2.f * d.z * (view.xy * d.z - view.z * d.xy) / (view.z * (view.z * view.z - d.z * d.z));
            }
            mean2d = fma(parameters.intrinsics.xy * sigma_weight, correction, projected.xy);
        }
        if (ring)
            for (uint n = 0; n < 7; ++n) {
                const float2 delta = points[n] - mean2d;
                covariance += (n != 0u ? sigma_weight : covariance_weight) * float3(delta.x * delta.x, delta.x * delta.y, delta.y * delta.y);
            }
        if (parameters.image.z == 2u)
            mean2d.x -= float(parameters.image.x) * floor(mean2d.x / float(parameters.image.x));
        center = mean2d - .5f;
        if (!ring)
            return all(isfinite(center));
    } else {
        if (!in_margin(projected.xy, parameters, .2f))
            return false;
        const float2 margin = .3f * .5f * float2(parameters.image.xy) / parameters.intrinsics.xy;
        const float2 positive = (float2(parameters.image.xy) - parameters.intrinsics.zw) / parameters.intrinsics.xy + margin;
        const float2 negative = parameters.intrinsics.zw / parameters.intrinsics.xy + margin;
        const float2 ratio = clamp(view.xy / view.z, -negative, positive);
        float3 jx = parameters.image.z == 1u ? float3(parameters.intrinsics.x, 0, 0) : float3(parameters.intrinsics.x / view.z, 0, -parameters.intrinsics.x * ratio.x / view.z);
        float3 jy = parameters.image.z == 1u ? float3(0, parameters.intrinsics.y, 0) : float3(0, parameters.intrinsics.y / view.z, -parameters.intrinsics.y * ratio.y / view.z);
        if (parameters.image.z == 2u) {
            const float h2 = max(dot(view.xz, view.xz), 1e-16f), h = sqrt(h2), r2 = max(dot(view, view), 1e-16f);
            jx = float3(view.z, 0, -view.x) * (float(parameters.image.x) / (2.f * kPi * h2));
            jy = float3(-view.y * view.x, h2, -view.y * view.z) * (float(parameters.image.y) / (kPi * r2 * h));
        }
        const float3 u = float3(dot(jx, a), dot(jx, bb), dot(jx, c)), v = float3(dot(jy, a), dot(jy, bb), dot(jy, c));
        covariance = float3(dot(u, u), dot(u, v), dot(v, v));
        if (parameters.image.z == 2u && dot(view.xz, view.xz) <= 1e-16f) {
            const float latitude_scale = float(parameters.image.y) / (kPi * max(length(view), 1e-8f));
            covariance = float3(float(parameters.image.x) * float(parameters.image.x), 0,
                                latitude_scale * latitude_scale * max(dot(a, a), max(dot(bb, bb), dot(c, c))));
        }
    }
    const float before = covariance.x * covariance.z - covariance.y * covariance.y;
    const float dilation = parameters.payload.y != 0u ? .1f : .3f;
    covariance.x += dilation;
    covariance.z += dilation;
    const float det = covariance.x * covariance.z - covariance.y * covariance.y;
    if (!isfinite(det) || det <= 0)
        return false;
    const float raw_opacity = parameters.payload.x != 0u ? float(((Ptr<half, Access.Read>)p.opacity)[id]) : p.opacity[id];
    opacity = 1.f / (1.f + exp(-raw_opacity)) * (parameters.payload.y != 0u ? sqrt(max(0.f, before / det)) : 1.f);
    if (opacity <= .5f / 255.f || !isfinite(opacity))
        return false;
    if (parameters.image.w == 0u)
        covariance = cap_covariance(covariance, opacity);
    const float final_det = covariance.x * covariance.z - covariance.y * covariance.y;
    conic = float3(covariance.z, -covariance.y, covariance.x) / final_det;
    return all(isfinite(conic));
}
bool primitive_hit(float2 point, float4 primitive, uint shape) {
    if (shape == 1u)
        return all(point >= primitive.xy) && all(point <= primitive.zw);
    const float2 delta = point - primitive.xy;
    return dot(delta, delta) <= primitive.z;
}
bool ring_hit(float2 point, float3 conic, float opacity, float4 primitive, float width, float period) {
    const float boundary = (.5f / 255.f) / max(opacity, 1e-7f), band = max(width, 0.f) * 10.f;
    const float outer = boundary * (1.f + band), inner = boundary * max(0.f, 1.f - band), padding = max(primitive.z, 0.f);
    const float2 samples[5] = {primitive.xy, primitive.xy + float2(padding, 0), primitive.xy - float2(padding, 0), primitive.xy + float2(0, padding), primitive.xy - float2(0, padding)};
    for (uint n = 0; n < (padding > 0 ? 5u : 1u); ++n) {
        float2 d = samples[n] - point;
        if (period > 0)
            d.x -= period * round(d.x / period);
        const float power = .5f * (conic.x * d.x * d.x + conic.z * d.y * d.y) + conic.y * d.x * d.y;
        if (power < 0 || !isfinite(power))
            continue;
        const float value = exp(-power);
        if (value < outer && value > inner)
            return true;
    }
    return false;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void polygon_coverage(uint3 tid : SV_DispatchThreadID, uint lane : SV_GroupIndex) {
    const SelectionParameters parameters = p.parameters[0];
    const uint cached_count = min(parameters.source.w, 2048u);
    for (uint n = lane; n < cached_count; n += 64u)
        polygon_cached[n] = p.polygon_vertices[n];
    GroupMemoryBarrierWithGroupSync();
    const uint2 pixel = tid.xy;
    if (pixel.x >= parameters.aabb.z || pixel.y >= parameters.aabb.w)
        return;
    const float2 point = float2(parameters.aabb.xy + pixel) + .5f;
    bool inside = false;
    uint previous = parameters.source.w - 1u;
    for (uint n = 0; n < parameters.source.w; ++n) {
        const float2 a = n < cached_count ? polygon_cached[n] : p.polygon_vertices[n];
        const float2 b = previous < cached_count ? polygon_cached[previous] : p.polygon_vertices[previous];
        if ((a.y > point.y) != (b.y > point.y)) {
            const float edge = (b.x - a.x) * (point.y - a.y) / (b.y - a.y) + a.x;
            if (point.x < edge)
                inside = !inside;
        }
        previous = n;
    }
    p.polygon_mask[pixel.y * parameters.aabb.z + pixel.x] = uint8_t(inside ? 255u : 0u);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void selection_query(uint3 tid : SV_DispatchThreadID) {
    const SelectionParameters parameters = p.parameters[0];
    const uint id = tid.x;
    if (id >= parameters.source.x)
        return;
    if (parameters.payload.z == 3u) {
        p.output[id] = uint8_t(p.pick[0] != 0xffffffffu && p.pick[1] == id ? 1u : 0u);
        return;
    }
    float2 point = 0;
    float3 conic = 0;
    float alpha = 0, depth = 0;
    bool selected = false;
    if (project_query(id, parameters, parameters.source.y == 3u, point, conic, alpha, depth)) {
        if (parameters.source.y == 2u) {
            const int2 at = int2(floor(point)) - int2(parameters.aabb.xy);
            selected = all(at >= 0) && all(uint2(at) < parameters.aabb.zw) && p.polygon_mask[uint(at.y) * parameters.aabb.z + uint(at.x)] != 0u;
        } else
            for (uint n = 0; n < parameters.source.z; ++n)
                if (parameters.source.y == 3u ? ring_hit(point, conic, alpha, p.primitives[n], parameters.ring.x, parameters.image.z == 2u ? float(parameters.image.x) : 0.f)
                                             : primitive_hit(point, p.primitives[n], parameters.source.y)) {
                    selected = true;
                    break;
                }
    }
    if (parameters.source.y == 3u) {
        if (selected && isfinite(depth) && depth > 0) {
            const uint key = asuint(depth);
            uint ignored;
            if (parameters.payload.z == 1u)
                InterlockedMin(p.pick[0], key, ignored);
            else if (p.pick[0] == key)
                InterlockedMin(p.pick[1], id, ignored);
        }
    } else
        p.output[id] = uint8_t(selected ? 1u : 0u);
}
