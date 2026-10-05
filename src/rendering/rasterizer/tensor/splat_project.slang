// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
// Per-splat projection: culling, overlay filters, 3D->2D covariance (pinhole,
// orthographic, equirectangular, 3DGUT unscented transform), mip compensation,
// spherical harmonics, and the tile-aligned pixel bounds. Ported from the
// native Metal viewer (splat_preprocessor.metal); outputs share its layout.
#include "splat_overlay.slang"
#include "splat_types.slang"

#define LFS_PORTAL_INLINE
#define LFS_PORTAL_VEC3 float3
#define LFS_PORTAL_VEC4 float4
#include "../slang/portal_compact.inc"
#define LFS_COLOR_INLINE
#define LFS_COLOR_VEC3 float3
#define LFS_COLOR_UINT uint
#define LFS_COLOR_MIX lerp
#include "../slang/display_color.inc"

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

struct Projection {
    Matrix4 model_to_world, world_to_camera;
    float4 camera_local, intrinsics, clip_scale;
    uint4 extent;
    float4 rasterization, display, panorama;
};
struct InputLayout {
    uint count, rest, has_deleted, objects, half_attrs, overlay, object_indexed, draw_count;
    uint lod, page_splats, logical_count, deleted_count;
};
struct SceneObject {
    Matrix4 model_to_world;
    float4 camera_local;
    uint4 flags; // visible, maximum active SH degree
};
struct GutSplat {
    float4 inverse0, inverse1, inverse2, mean_opacity;
};

struct Parameters {
    Ptr<float, Access.Read> means;     // packed float3
    Ptr<float, Access.Read> scales;    // packed float3 / half3 / RAD half4
    Ptr<float4, Access.Read> rotations;
    Ptr<float, Access.Read> opacity;
    Ptr<float, Access.Read> sh0;       // packed float3 / RAD half4
    Ptr<uint8_t, Access.Read> rest;
    Ptr<float2, Access.Read> bounds;
    Ptr<uint8_t, Access.Read> deleted;
    Ptr<ProjectedSplat, Access.ReadWrite> output;
    Ptr<Projection, Access.Read> frame;
    Ptr<InputLayout, Access.Read> layout;
    Ptr<uint, Access.Read> object_indices;
    Ptr<SceneObject, Access.Read> objects;
    Ptr<float4, Access.Read> overlay;
    Ptr<uint, Access.ReadWrite> overlay_flags;
    Ptr<uint8_t, Access.Read> node_mask;
    Ptr<GutSplat, Access.ReadWrite> gut_output;
    Ptr<uint, Access.Read> lod_indices;
    Ptr<uint, Access.Read> logical_indices;
    Ptr<uint, Access.Read> lod_levels;
    Ptr<float, Access.Read> lod_weights;
    Ptr<uint, Access.Read> lod_count;
    uint sh_storage;     // ShStorage
    uint sh_degree;      // maximum SH degree of the pipeline
    uint primitive_mode; // 0 Gaussian, 1 points, 2 discs, 3 GUT
    uint tight_bounds;
};
[[vk::push_constant]] ConstantBuffer<Parameters> p;

float2 panorama_project(float3 view, float2 extent) {
    const float distance = length(view);
    const float3 direction = view / max(distance, 1e-8f);
    return float2(atan2(direction.x, direction.z) / (2.f * kPi) + .5f,
                  asin(clamp(direction.y, -1.f, 1.f)) / kPi + .5f) * extent;
}

// Periodic support intersected with the pixel viewport before tiling.
uint2 panorama_tile_span(float center, float radius, float period, uint width) {
    const uint columns = (width + 15u) / 16u;
    if (2.f * radius >= period)
        return uint2(0, columns * 16u);
    center += period * round((.5f * float(width) - center) / period);
    uint first = columns, end = 0, second = columns, second_end = 0;
    for (int shift = -1; shift <= 1; ++shift) {
        const float lo = max(0.f, center + float(shift) * period - radius);
        const float hi = min(float(width), center + float(shift) * period + radius);
        if (lo >= hi)
            continue;
        const uint x0 = uint(floor(lo / 16.f)), x1 = min(columns, uint(ceil(hi / 16.f)));
        if (first == columns) {
            first = x0;
            end = x1;
        } else {
            second = x0;
            second_end = x1;
        }
    }
    if (first == columns)
        return uint2(0);
    if (second == columns)
        return uint2(first, end) * 16u;
    if (second < first) {
        const uint a = first, b = end;
        first = second;
        end = second_end;
        second = a;
        second_end = b;
    }
    if (end >= second)
        return uint2(0, columns * 16u);
    return uint2(second, columns + end) * 16u;
}

float opacity_power(float opacity, bool spark) {
    if (spark && opacity > 1.f) {
        const float extent = sqrt(8.f) + .7f * (min(opacity, 5.f) - 1.f);
        return .5f * extent * extent;
    }
    return max(4.f, log(max(opacity, .5f / 255.f + 1e-8f) * 510.f));
}

float3 clamp_covariance_extent(float3 covariance, float opacity, bool spark) {
    const float power = opacity_power(opacity, spark);
    const float maximum = 512.5f * 512.5f / (2.f * power);
    const float average = .5f * (covariance.x + covariance.z);
    const float delta = sqrt(max(0.f, average * average - (covariance.x * covariance.z - covariance.y * covariance.y)));
    const float eigen1 = average + delta, eigen2 = max(average - delta, 0.f);
    const float capped1 = min(eigen1, maximum), capped2 = min(eigen2, maximum);
    if (capped1 >= eigen1 && capped2 >= eigen2)
        return covariance;
    const float2 axis = abs(covariance.y) > 1e-6f ? normalize(float2(covariance.y, eigen1 - covariance.x))
                                                  : (covariance.x >= covariance.z ? float2(1, 0) : float2(0, 1));
    const float2 other = float2(axis.y, -axis.x);
    return float3(capped1 * axis.x * axis.x + capped2 * other.x * other.x,
                  capped1 * axis.x * axis.y + capped2 * other.x * other.y,
                  capped1 * axis.y * axis.y + capped2 * other.y * other.y);
}

float3 portal_covariance(float3 covariance, float2 extent) {
    const float limit = min(1024.f, min(extent.x, extent.y));
    const float cap = max(1.f, sqrt(8.f * max(covariance.x, covariance.z)) / limit);
    covariance /= cap * cap;
    const float mid = .5f * (covariance.x + covariance.z);
    const float radius = length(float2(.5f * (covariance.x - covariance.z), covariance.y));
    const float first = mid + radius, second = max(mid - radius, .025f);
    const float maximum = limit * limit / 8.f;
    const float a = min(first, maximum), b = min(second, maximum);
    const float2 axis = abs(covariance.y) > 1e-6f ? normalize(float2(covariance.y, first - covariance.x))
                                                  : (covariance.x >= covariance.z ? float2(1, 0) : float2(0, 1));
    const float2 other = float2(axis.y, -axis.x);
    return float3(a * axis.x * axis.x + b * other.x * other.x,
                  a * axis.x * axis.y + b * other.x * other.y,
                  a * axis.y * axis.y + b * other.y * other.y);
}

// Same cell swizzle and per-256 bounds as core/sh_value_codec.cuh. Q16 holds
// integer codes, never IEEE half.
float sh_component(uint primitive, uint component, uint rest, uint page_splats) {
    const uint block = primitive / 32u, lane = primitive % 32u;
    if (p.sh_storage == 3u) {
        const uint64_t index = uint64_t(block) * (rest * 3u * 32u) + component * 32u + lane;
        const float2 mm = p.bounds[primitive / 256u];
        const uint16_t q = ((Ptr<uint16_t, Access.Read>)p.rest)[index];
        return fma(mm.y - mm.x, float(q) * (1.0f / 65535.0f), mm.x);
    }
    const uint slots = (rest * 3u + 3u) / 4u;
    const uint64_t index = p.sh_storage == 0u ? uint64_t(primitive) * rest * 3u + component
                                              : (uint64_t(block) * slots * 32u + (component / 4u) * 32u + lane) * 4u + component % 4u;
    if (p.sh_storage == 4u) {
        // RAD: signed bytes in the float4 cell swizzle, one scale per SH band.
        const float4 maxima = ((Ptr<float4, Access.Read>)p.bounds)[(primitive / page_splats) * 4u];
        const uint band = component < 9u ? 0u : component < 24u ? 1u : 2u;
        return (float(((Ptr<int8_t, Access.Read>)p.rest)[index]) / 127.f) * maxima[band];
    }
    if (p.sh_storage == 2u)
        return float(((Ptr<half, Access.Read>)p.rest)[index]);
    return ((Ptr<float, Access.Read>)p.rest)[index];
}

float3 evaluate_sh(float3 dc, float3 direction, uint primitive, uint layout_rest, uint active_degree, uint page_splats) {
    float3 color = 0.5f + 0.28209479177387814f * dc;
    if (p.sh_degree == 0u || active_degree == 0u)
        return max(color, 0.0f);
    const float norm2 = dot(direction, direction);
    const float3 d = norm2 > 1e-20f ? direction * rsqrt(norm2) : float3(0, 0, 1);
    const float x = d.x, y = d.y, z = d.z, xx = x * x, yy = y * y, zz = z * z;
    const float basis[15] = {-0.4886025119029199f * y, 0.4886025119029199f * z, -0.4886025119029199f * x,
                             1.0925484305920792f * x * y, -1.0925484305920792f * y * z, 0.31539156525252005f * (2 * zz - xx - yy),
                             -1.0925484305920792f * x * z, 0.5462742152960396f * (xx - yy),
                             0.5900435899266435f * y * (-3 * xx + yy), 2.890611442640554f * x * y * z,
                             0.4570457994644658f * y * (xx + yy - 4 * zz), 0.3731763325901154f * z * (2 * zz - 3 * xx - 3 * yy),
                             0.4570457994644658f * x * (xx + yy - 4 * zz), 1.445305721320277f * z * (xx - yy),
                             0.5900435899266435f * x * (-xx + 3 * yy)};
    const uint count = (active_degree + 1u) * (active_degree + 1u) - 1u;
    for (uint k = 0; k < count; ++k) {
        const uint c = k * 3u;
        color += basis[k] * float3(sh_component(primitive, c, layout_rest, page_splats),
                                   sh_component(primitive, c + 1u, layout_rest, page_splats),
                                   sh_component(primitive, c + 2u, layout_rest, page_splats));
    }
    return max(color, 0.0f);
}

float3 rotate_axis(float4 q, float3 v) { return v + 2.0f * cross(q.yzw, cross(q.yzw, v) + q.x * v); }

float3 load3(Ptr<float, Access.Read> data, uint index) {
    return float3(data[index * 3u], data[index * 3u + 1u], data[index * 3u + 2u]);
}
float3 load_half3(Ptr<float, Access.Read> data, uint index, uint stride) {
    const Ptr<half, Access.Read> h = (Ptr<half, Access.Read>)data;
    return float3(float(h[index * stride]), float(h[index * stride + 1u]), float(h[index * stride + 2u]));
}

void store_output(uint i, ProjectedSplat value) { p.output[i] = value; }

[shader("compute")]
[numthreads(256, 1, 1)]
void project_splats(uint3 tid : SV_DispatchThreadID) {
    const uint i = tid.x;
    const InputLayout layout = p.layout[0];
    if (i >= layout.draw_count)
        return;
    store_output(i, (ProjectedSplat)0);
    if (layout.overlay != 0u)
        p.overlay_flags[i] = 0u;
    // Clear unused slots before reading GPU-private indirection.
    if ((layout.lod & 32u) != 0u && (p.lod_count[0] > layout.draw_count || i >= p.lod_count[0]))
        return;
    const uint source = (layout.lod & 1u) != 0u ? p.lod_indices[i] : i;
    const uint logical = (layout.lod & 2u) != 0u ? p.logical_indices[i] : source;
    if (source >= layout.count || logical >= layout.logical_count)
        return;
    if (layout.has_deleted != 0u) {
        const uint deleted_id = p.sh_storage == 4u ? logical : source;
        if (deleted_id < layout.deleted_count && p.deleted[deleted_id] != 0u)
            return;
    }
    const Projection frame = p.frame[0];
    const float3 position = load3(p.means, source);
    Matrix4 model_to_world = frame.model_to_world;
    float3 camera_local = frame.camera_local.xyz;
    uint active_degree = p.sh_degree;
    int node = 0;
    if (layout.objects != 0u) {
        const uint index = layout.object_indexed != 0u ? p.object_indices[logical] : 0u;
        node = int(index);
        if (index >= layout.objects)
            return;
        const SceneObject object = p.objects[index];
        if (object.flags.x == 0u)
            return;
        model_to_world = object.model_to_world;
        camera_local = object.camera_local.xyz;
        active_degree = min(active_degree, object.flags.y);
    }
    const Matrix4 matrix = mul44(frame.world_to_camera, model_to_world);
    const float3 world = mul4(model_to_world, float4(position, 1)).xyz;
    const float3 view = mul4(matrix, float4(position, 1)).xyz;
    const bool spark = frame.display.z == 1.f;
    const bool portal = frame.rasterization.w == 1.f && !spark;
    const bool equirectangular = frame.extent.z == 2u;
    const bool orthographic = frame.extent.z == 1u;
    const float projection_depth = equirectangular ? length(view) : view.z;
    if (!all(isfinite(view)) || projection_depth <= frame.clip_scale.x || projection_depth >= frame.clip_scale.y)
        return;
    if (frame.display.w == 1.f && p.primitive_mode == 0u && !equirectangular) {
        // Desktop GS survivor admission: reject off-frustum means early.
        const float2 camera_extent = all(frame.panorama.xy > 0) ? frame.panorama.xy : float2(frame.extent.xy);
        const float2 image = frame.intrinsics.xy * view.xy / (orthographic ? 1.f : view.z) + frame.intrinsics.zw + frame.panorama.zw;
        const float2 margin = .2f * camera_extent;
        if (any(image < -margin) || any(image >= camera_extent + margin))
            return;
    }
    uint flags = 0;
    if (layout.overlay != 0u) {
        bool active = true;
        overlay_filter(p.overlay, 0, false, node, world, active, flags);
        for (uint n = 0; n < 15 && overlay_enabled(p.overlay[26 + n * 7].x); ++n)
            overlay_filter(p.overlay, 26 + n * 7, false, node, world, active, flags);
        overlay_filter(p.overlay, 7, true, node, world, active, flags);
        for (uint n = 0; n < 15 && overlay_enabled(p.overlay[131 + n * 5].x); ++n)
            overlay_filter(p.overlay, 131 + n * 5, true, node, world, active, flags);
        if (active && overlay_enabled(p.overlay[13].x)) {
            float4 k = p.overlay[12];
            if (k.x <= 0)
                k = float4(frame.intrinsics.xy, float2(frame.extent.xy) * .5f);
            if (orthographic)
                k = float4(frame.intrinsics.xy, float2(frame.extent.xy) * .5f);
            const float2 full_extent = equirectangular ? frame.panorama.xy : float2(frame.extent.xy);
            const float2 center = equirectangular ? panorama_project(view, full_extent)
                                                  : k.xy * view.xy / (orthographic ? 1.f : view.z) + k.zw;
            const float2 half_extent = .5f * full_extent * p.overlay[206].xy;
            const float2 window = .5f * full_extent + float2(p.overlay[14].w, p.overlay[15].w) * (.5f * full_extent - half_extent);
            const bool inside = all(abs(center - window) <= half_extent) && projection_depth >= -p.overlay[15].z &&
                                projection_depth <= -p.overlay[14].z;
            if (!inside) {
                flags |= 2u;
                if (overlay_enabled(p.overlay[13].z))
                    flags |= 1u;
                if (overlay_enabled(p.overlay[13].y))
                    active = false;
            }
        }
        const float4 emphasis = p.overlay[20];
        if (active && overlay_enabled(emphasis.y) && node >= 0 && node < int(round(emphasis.z))) {
            if (overlay_enabled(emphasis.x) && p.node_mask[node] == 0u)
                flags |= 3u;
            if (emphasis.w > 0 && p.node_mask[node] != 0u)
                flags |= 4u;
        }
        p.overlay_flags[i] = flags;
        if (!active)
            return;
    }
    const float logit = layout.half_attrs != 0u ? float(((Ptr<half, Access.Read>)p.opacity)[source]) : p.opacity[source];
    float alpha = spark ? max(logit, 0.f) : 1.0f / (1.0f + exp(-logit));
    if (spark && alpha > 1.f)
        alpha = min(alpha * 4.f - 3.f, 5.f);
    if (!isfinite(alpha) || alpha < 0.5f / 255.0f)
        return;
    if ((layout.lod & 8u) != 0u)
        alpha *= clamp(p.lod_weights[i], 0.f, 1.f);
    if (!isfinite(alpha))
        return;
    if (portal)
        alpha = lfsPortalCompactOpacity(alpha);
    if (portal && alpha <= 1.f / 255.f)
        return;
    const float source_alpha = alpha;
    // Divide the focal directly by Z, as the reference does.
    float2 center = equirectangular ? panorama_project(view, frame.panorama.xy) - frame.panorama.zw - .5f
                                    : fma(frame.intrinsics.xy / (orthographic ? 1.f : view.z), view.xy, frame.intrinsics.zw) - .5f;
    float3 conic;
    float radius;
    float2 panorama_radius = 0, portal_axis = 0, portal_extent = 0, gaussian_support = 0;
    float2 gut_lo = 0, gut_hi = float2(frame.extent.xy);
    bool gut_rect_bounds = false;
    if (p.primitive_mode == 1u) {
        radius = 2.0f;
        conic = float3(1, 0, 1);
    } else {
        float4 q = layout.half_attrs != 0u ? float4(float(((Ptr<half, Access.Read>)p.rotations)[source * 4u]),
                                                    float(((Ptr<half, Access.Read>)p.rotations)[source * 4u + 1u]),
                                                    float(((Ptr<half, Access.Read>)p.rotations)[source * 4u + 2u]),
                                                    float(((Ptr<half, Access.Read>)p.rotations)[source * 4u + 3u]))
                                             : p.rotations[source];
        const float norm2 = dot(q, q);
        if (!isfinite(norm2))
            return;
        q = norm2 > 1e-8f ? q * rsqrt(norm2) : float4(1, 0, 0, 0);
        if (portal) {
            q = lfsPortalCompactRotation(q);
            q *= rsqrt(max(dot(q, q), 1e-8f));
        }
        const float3 log_scale = p.sh_storage == 4u ? load_half3(p.scales, source, 4u)
                                 : layout.half_attrs != 0u ? load_half3(p.scales, source, 3u)
                                                           : load3(p.scales, source);
        float3 s = exp(min(log_scale, float3(20.f))) * frame.clip_scale.z;
        if (portal)
            s = lfsPortalCompactScales(s);
        if (!all(isfinite(s)))
            return;
        const float3 a = mul3(matrix, rotate_axis(q, float3(s.x, 0, 0)));
        const float3 b = mul3(matrix, rotate_axis(q, float3(0, s.y, 0)));
        const float3 c = mul3(matrix, rotate_axis(q, float3(0, 0, s.z)));
        float3 raw_covariance = 0;
        if (p.primitive_mode != 3u) {
            // Bound the covariance Jacobian near the frustum, as in 3DGS.
            const float2 margin = .3f * .5f * float2(frame.extent.xy) / frame.intrinsics.xy;
            const float2 positive = (float2(frame.extent.xy) - frame.intrinsics.zw) / frame.intrinsics.xy + margin;
            const float2 negative = frame.intrinsics.zw / frame.intrinsics.xy + margin;
            const float2 ratio = portal ? view.xy / view.z : clamp(view.xy / view.z, -negative, positive);
            float3 jx = orthographic ? float3(frame.intrinsics.x, 0, 0) : float3(frame.intrinsics.x / view.z, 0, -frame.intrinsics.x * ratio.x / view.z);
            float3 jy = orthographic ? float3(0, frame.intrinsics.y, 0) : float3(0, frame.intrinsics.y / view.z, -frame.intrinsics.y * ratio.y / view.z);
            if (equirectangular) {
                const float horizontal2 = max(dot(view.xz, view.xz), 1e-16f);
                const float horizontal = sqrt(horizontal2), distance2 = max(dot(view, view), 1e-16f);
                jx = float3(view.z, 0, -view.x) * (frame.panorama.x / (2.f * kPi * horizontal2));
                jy = float3(-view.y * view.x, horizontal2, -view.y * view.z) * (frame.panorama.y / (kPi * distance2 * horizontal));
            }
            const float3 u = float3(dot(jx, a), dot(jx, b), dot(jx, c));
            const float3 v = float3(dot(jy, a), dot(jy, b), dot(jy, c));
            raw_covariance = float3(dot(u, u), dot(u, v), dot(v, v));
            if (equirectangular && dot(view.xz, view.xz) <= 1e-16f) {
                // Longitude is undefined at a pole: reserve the full width.
                const float latitude_scale = frame.panorama.y / (kPi * max(length(view), 1e-8f));
                raw_covariance = float3(frame.panorama.x * frame.panorama.x, 0,
                                        latitude_scale * latitude_scale * max(dot(a, a), max(dot(b, b), dot(c, c))));
            }
        }
        if (p.primitive_mode == 3u) {
            // project_gaussian_to_camera_gut, including FP32 UT weights.
            const float lambda = .1f * .1f * 3.f - 3.f;
            const float denominator = 3.f + lambda;
            const float sigma_scale = sqrt(denominator);
            const float mean_weight = lambda / denominator;
            const float covariance_weight = mean_weight + (1.f - .1f * .1f + 2.f);
            const float sigma_weight = 1.f / (2.f * denominator);
            const float3 sigma[7] = {view, view + sigma_scale * a, view + sigma_scale * b, view + sigma_scale * c,
                                     view - sigma_scale * a, view - sigma_scale * b, view - sigma_scale * c};
            float2 image[7];
            const float raster_scale = frame.rasterization.x > 0 ? frame.rasterization.x : 1.f;
            const float2 panorama_extent = max(float2(1), round(frame.panorama.xy / raster_scale)) * raster_scale;
            for (uint n = 0; n < 7; ++n) {
                if (!all(isfinite(sigma[n])) || (equirectangular ? length(sigma[n]) <= 1e-8f : sigma[n].z <= frame.clip_scale.x))
                    return;
                image[n] = equirectangular ? panorama_project(sigma[n], panorama_extent)
                                           : frame.intrinsics.xy * sigma[n].xy / (orthographic ? 1.f : sigma[n].z) + frame.intrinsics.zw;
                if (equirectangular && n != 0u) {
                    float dx = image[n].x - image[0].x;
                    dx -= panorama_extent.x * round(dx / panorama_extent.x);
                    image[n].x = image[0].x + dx;
                }
            }
            float2 mean = 0;
            if (equirectangular) {
                float2 offset = 0;
                for (uint n = 1; n < 7; ++n)
                    offset += image[n] - image[0];
                mean = fma(float2(sigma_weight), offset, image[0]);
            } else {
                for (uint n = 0; n < 7; ++n)
                    mean += (n != 0u ? sigma_weight : mean_weight) * image[n];
            }
            raw_covariance = 0;
            for (uint n = 0; n < 7; ++n) {
                const float2 d = image[n] - mean;
                raw_covariance += (n != 0u ? sigma_weight : covariance_weight) * float3(d.x * d.x, d.x * d.y, d.y * d.y);
            }
            if (equirectangular)
                mean.x -= panorama_extent.x * floor(mean.x / panorama_extent.x);
            center = mean - .5f - (equirectangular ? frame.panorama.zw : float2(0));
            // 3D alpha uses the source scale, independent of binning scale.
            const float3 source_scale = portal ? lfsPortalCompactScales(exp(log_scale)) : exp(log_scale);
            const float3 base_scale = max(source_scale, float3(1e-8f));
            const float3 ga = mul3(matrix, rotate_axis(q, float3(base_scale.x, 0, 0)));
            const float3 gb = mul3(matrix, rotate_axis(q, float3(0, base_scale.y, 0)));
            const float3 gc = mul3(matrix, rotate_axis(q, float3(0, 0, base_scale.z)));
            if (!portal && !spark && !equirectangular) {
                // Conservative projected ellipsoid bound, unioned below.
                gut_rect_bounds = true;
                const float3 rx = float3(ga.x, gb.x, gc.x), ry = float3(ga.y, gb.y, gc.y), rz = float3(ga.z, gb.z, gc.z);
                const float r2 = 2.f * (opacity_power(source_alpha, false) + 1e-4f);
                const float bound_denominator = view.z * view.z - r2 * dot(rz, rz);
                float2 bound_center, bound_extent;
                const bool finite_bound = orthographic || (view.z > sqrt(r2 * dot(rz, rz)) * 1.001f + frame.clip_scale.x && bound_denominator > 0.f);
                if (finite_bound) {
                    if (orthographic) {
                        bound_center = view.xy;
                        bound_extent = sqrt(r2 * float2(dot(rx, rx), dot(ry, ry)));
                    } else {
                        bound_center = (view.xy * view.z - r2 * float2(dot(rx, rz), dot(ry, rz))) / bound_denominator;
                        const float3 tx = view.z * rx - view.x * rz, ty = view.z * ry - view.y * rz;
                        bound_extent = sqrt(r2 * float2(dot(tx, tx), dot(ty, ty))) / bound_denominator;
                    }
                    bound_center = bound_center * frame.intrinsics.xy + frame.intrinsics.zw - .5f;
                    bound_extent = bound_extent * frame.intrinsics.xy * 1.001f + 1.f;
                    if (all(isfinite(bound_center)) && all(isfinite(bound_extent))) {
                        gut_lo = bound_center - bound_extent;
                        gut_hi = bound_center + bound_extent;
                    }
                }
            }
            const float determinant = dot(ga, cross(gb, gc));
            if (!isfinite(determinant) || abs(determinant) <= 1e-30f)
                return;
            const float norm_product = length(ga) * length(gb) * length(gc);
            const float sphere = (!portal && abs(determinant) > norm_product * 1e-5f)
                                     ? sqrt(2.f * (opacity_power(alpha, spark) + 1e-4f) * (dot(ga, ga) + dot(gb, gb) + dot(gc, gc))) * 1.001f
                                     : 0.f;
            GutSplat gut;
            gut.inverse0 = float4(cross(gb, gc) / determinant, isfinite(sphere) ? sphere : 0.f);
            gut.inverse1 = float4(cross(gc, ga) / determinant, 0);
            gut.inverse2 = float4(cross(ga, gb) / determinant, 0);
            gut.mean_opacity = float4(view, alpha);
            p.gut_output[i] = gut;
        }
        // Dilation and the covariance cap use source viewport pixels.
        const float raster_scale = isfinite(frame.rasterization.x) && frame.rasterization.x > 0 ? frame.rasterization.x : 1.f;
        const float variance_scale = raster_scale * raster_scale;
        const float raw_xx = raw_covariance.x / variance_scale, raw_yy = raw_covariance.z / variance_scale;
        float3 covariance = float3(raw_xx + frame.clip_scale.w, raw_covariance.y / variance_scale, raw_yy + frame.clip_scale.w);
        float det = covariance.x * covariance.z - covariance.y * covariance.y;
        if (!isfinite(det) || det <= 1e-12f)
            return;
        if (frame.extent.w != 0u || (spark && p.primitive_mode != 3u))
            alpha *= sqrt(max((raw_xx * raw_yy - covariance.y * covariance.y) / det, 0.f));
        if (alpha < 0.5f / 255.0f)
            return;
        const float2 camera_extent = all(frame.panorama.xy > 0) ? frame.panorama.xy : float2(frame.extent.xy);
        const float2 source_extent = max(float2(1), round(camera_extent / raster_scale));
        if (portal && p.primitive_mode != 3u)
            covariance = portal_covariance(covariance, source_extent);
        else if (!portal && p.primitive_mode != 3u)
            covariance = clamp_covariance_extent(covariance, alpha, spark);
        covariance *= variance_scale;
        const float xx = covariance.x, xy = covariance.y, yy = covariance.z;
        det = xx * yy - xy * xy;
        conic = float3(yy, -xy, xx) / det;
        float3 bin_covariance = covariance;
        if (portal && p.primitive_mode == 3u) {
            const float3 billboard = portal_covariance(covariance, camera_extent);
            const float mid = .5f * (billboard.x + billboard.z);
            const float delta = length(float2(.5f * (billboard.x - billboard.z), billboard.y));
            const float first = mid + delta, second = max(mid - delta, .025f);
            portal_axis = abs(billboard.y) > 1e-6f ? normalize(float2(billboard.y, first - billboard.x))
                                                   : (billboard.x >= billboard.z ? float2(1, 0) : float2(0, 1));
            const float clip = min(1.f, .5f * sqrt(max(0.f, log(source_alpha * 255.f))));
            portal_extent = sqrt(8.f * float2(first, second)) * clip;
            GutSplat gut = p.gut_output[i];
            gut.inverse0.w = portal_axis.x;
            gut.inverse1.w = portal_axis.y;
            gut.inverse2.w = portal_extent.y;
            p.gut_output[i] = gut;
            bin_covariance = 2.f * billboard;
        }
        const float eigen = .5f * (bin_covariance.x + bin_covariance.z) +
                            sqrt(max(0.0f, .25f * (bin_covariance.x - bin_covariance.z) * (bin_covariance.x - bin_covariance.z) + bin_covariance.y * bin_covariance.y));
        const float support_alpha = spark && p.primitive_mode == 3u ? source_alpha : alpha;
        radius = p.primitive_mode == 2u ? 3.0f * sqrt(eigen) : sqrt(2.0f * opacity_power(support_alpha, spark) * eigen);
        gaussian_support = float2(radius);
        if (p.tight_bounds != 0u && p.primitive_mode == 0u && !portal && !spark && layout.overlay == 0u && det > xx * yy * 1e-4f) {
            const float power = opacity_power(alpha, false);
            gaussian_support = min(gaussian_support, sqrt(2.f * (power + 1e-4f) * bin_covariance.xz) * 1.001f + 1.f);
        }
        if (equirectangular) {
            const float extent_factor = sqrt(2.f * opacity_power(source_alpha, spark));
            panorama_radius = min(sqrt(bin_covariance.xz) * extent_factor, frame.panorama.xy * float2(1.f, .49f));
            radius = max(panorama_radius.x, panorama_radius.y);
        }
    }
    if (!all(isfinite(center)) || !isfinite(radius))
        return;
    const float2 extent = float2(frame.extent.xy);
    const float2 support = equirectangular ? panorama_radius
                                           : (p.tight_bounds != 0u && p.primitive_mode == 0u ? gaussian_support : float2(radius));
    float2 lo = clamp(floor(center - support), 0.0f, extent), hi = clamp(ceil(center + support), 0.0f, extent);
    if (gut_rect_bounds) {
        lo = min(lo, clamp(floor(gut_lo), 0.f, extent));
        hi = max(hi, clamp(ceil(gut_hi), 0.f, extent));
    }
    if (equirectangular) {
        const uint2 span = panorama_tile_span(center.x, support.x, frame.panorama.x, frame.extent.x);
        lo.x = float(span.x);
        hi.x = float(span.y);
    }
    if (any(lo >= hi))
        return;
    // SH follows the object's rotation; nonuniform scale must not distort it.
    float3 direction = orthographic ? float3(frame.world_to_camera.c0.z, frame.world_to_camera.c1.z, frame.world_to_camera.c2.z)
                                    : mul3(model_to_world, position - camera_local);
    const float3 axis_lengths = float3(length(model_to_world.c0.xyz), length(model_to_world.c1.xyz), length(model_to_world.c2.xyz));
    if (all(axis_lengths > 1e-8f))
        direction = float3(dot(model_to_world.c0.xyz / axis_lengths.x, direction),
                           dot(model_to_world.c1.xyz / axis_lengths.y, direction),
                           dot(model_to_world.c2.xyz / axis_lengths.z, direction));
    const float3 dc = p.sh_storage == 4u ? load_half3(p.sh0, source, 4u) : load3(p.sh0, source);
    float3 color = evaluate_sh(dc, direction, source, layout.rest, active_degree, layout.page_splats);
    if ((layout.lod & 16u) != 0u) {
        const uint level = (layout.lod & 4u) != 0u ? p.lod_levels[i] % 5u : 0u;
        const float3 palette[5] = {float3(1, 0, 0), float3(0, 1, 0), float3(0, 0, 1), float3(1, 1, 0), float3(1, 0, 1)};
        color *= palette[level];
    }
    if (portal)
        color = lfsPortalCompactColor(color);
    if (frame.rasterization.w == 1.f && frame.display.x > 0)
        color = lfsDisplaySplat(color, uint(frame.display.x), frame.display.y);
    if (layout.overlay != 0u)
        color = overlay_projection_color(color, center + (equirectangular ? frame.panorama.zw : float2(0)), flags, p.overlay);
    if (!all(isfinite(color)))
        return;
    // Radial sort metric in world space, as the reference computes it.
    const float3 translation = frame.world_to_camera.c3.xyz;
    const float3 sort_delta = world + float3(dot(translation, frame.world_to_camera.c0.xyz),
                                             dot(translation, frame.world_to_camera.c1.xyz),
                                             dot(translation, frame.world_to_camera.c2.xyz));
    ProjectedSplat result;
    result.mean_depth = float4(center, projection_depth, portal && p.primitive_mode == 3u ? portal_extent.x : radius);
    result.conic_opacity = float4(conic, alpha);
    result.color = float4(color, dot(sort_delta, sort_delta));
    result.bounds = uint4(uint2(lo), uint2(hi));
    store_output(i, result);
}
