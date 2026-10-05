// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

// Point-cloud rendering shared by the Metal and Vulkan tensor backends. Native
// points are expanded to pixel-sized camera-facing quads because tensor raster
// programs deliberately expose triangle lists only.
struct PointObject {
    column_major float4x4 transform;
    float4 camera;
    uint4 flags;
};

struct Parameters {
    Ptr<float, Access.ReadWrite> positions;
    Ptr<float, Access.ReadWrite> colors;
    Ptr<PointObject, Access.ReadWrite> objects;
    Ptr<int, Access.ReadWrite> transformIndices;
    Ptr<uint8_t, Access.ReadWrite> selection;
    Ptr<uint8_t, Access.ReadWrite> preview;
    Ptr<float4, Access.ReadWrite> palette;
    Ptr<uint8_t, Access.ReadWrite> deleted;
    Ptr<float4, Access.ReadWrite> depthRgba;
    Ptr<float, Access.ReadWrite> linearDepth;
    // Flattened PointParameters: the tensor binding ABI intentionally permits
    // no nested parameter structs. This remains the same 256-byte host layout.
    column_major float4x4 viewProjection;
    column_major float4x4 view;
    column_major float4x4 cropToLocal;
    float4 cropMin;
    float4 cropMax;
    float4 voxelFocalOrtho;
    uint4 counts;
    uint2 extent;
    uint2 padding;
};

[[vk::push_constant]] ConstantBuffer<Parameters> p;

static const float2 QUAD[6] = {
    float2(-1.0, -1.0), float2(1.0, -1.0), float2(1.0, 1.0),
    float2(-1.0, -1.0), float2(1.0, 1.0), float2(-1.0, 1.0)
};

struct PointVertex {
    float4 position : SV_Position;
    float2 pointDelta : TEXCOORD0;
    float3 color : COLOR0;
    float viewDepth : TEXCOORD1;
};

PointVertex rejectedPoint() {
    PointVertex result;
    // All six vertices coincide, so even a very large requested point size
    // cannot pull a rejected primitive back into the viewport.
    result.position = float4(2.0, 2.0, 2.0, 1.0);
    result.pointDelta = float2(0.0);
    result.color = float3(0.0);
    result.viewDepth = 0.0;
    return result;
}

[shader("vertex")]
PointVertex pointVertex(uint vertexId : SV_VertexID, uint instanceId : SV_InstanceID) {
    const uint source = instanceId;
    const float2 corner = QUAD[vertexId];
    const uint flags = p.counts.z;
    if ((flags & 512u) != 0u && p.deleted[source] != 0u)
        return rejectedPoint();

    float3 world = float3(p.positions[source * 3u], p.positions[source * 3u + 1u], p.positions[source * 3u + 2u]);
    if (p.counts.x != 0u) {
        int node = (flags & 16u) != 0u ? p.transformIndices[source] : 0;
        node = clamp(node, 0, int(p.counts.x) - 1);
        PointObject object = p.objects[uint(node)];
        if (object.flags.x == 0u)
            return rejectedPoint();
        const float4 transformed = mul(object.transform, float4(world, 1.0));
        world = abs(transformed.w) > 1.0e-6 ? transformed.xyz / transformed.w : transformed.xyz;
    }

    bool desaturate = false;
    if ((flags & 1u) != 0u) {
        const float3 local = mul(p.cropToLocal, float4(world, 1.0)).xyz;
        const float3 radius = max(abs(p.cropMin.xyz), float3(1.0e-8));
        const bool inside = (flags & 1024u) != 0u
                                ? dot(local / radius, local / radius) <= 1.0
                                : all(local >= p.cropMin.xyz) && all(local <= p.cropMax.xyz);
        const bool visible = (flags & 2u) != 0u ? !inside : inside;
        if (!visible) {
            if ((flags & 4u) == 0u)
                return rejectedPoint();
            desaturate = true;
        }
    }

    const float4 viewPosition = mul(p.view, float4(world, 1.0));
    float4 clip = mul(p.viewProjection, float4(world, 1.0));
    if (abs(clip.w) <= 1.0e-6 || ((flags & 8u) == 0u && -viewPosition.z < 1.0e-4))
        return rejectedPoint();

    const float voxel = max(p.voxelFocalOrtho.x, 1.0e-5);
    const float radiusPixels = max(1.0, ceil((flags & 8u) != 0u
        ? voxel * max(p.voxelFocalOrtho.z, 1.0e-5) * 0.5
        : voxel * max(p.voxelFocalOrtho.y, 1.0) / max(-viewPosition.z, 1.0e-4)));
    const float maxSize = float(p.counts.w);
    float diameter = clamp(2.0 * radiusPixels, 1.0, maxSize);

    float3 color = saturate(float3(p.colors[source * 3u], p.colors[source * 3u + 1u], p.colors[source * 3u + 2u]));
    if (desaturate) {
        const float gray = dot(color, float3(0.299, 0.587, 0.114));
        color = lerp(color, float3(gray), 0.75);
    }
    const uint group = (flags & 32u) != 0u ? uint(p.selection[source]) : 0u;
    const bool inPreview = (flags & 64u) != 0u && p.preview[source] != 0u;
    const bool highlight = (inPreview && group == 0u && (flags & 128u) != 0u) ||
                           (inPreview && group != 0u && (flags & 128u) == 0u);
    if (highlight) {
        color = lerp(color, p.palette[256].xyz, 0.9);
        diameter = clamp(max(diameter + 2.0, diameter * 1.6), 1.0, maxSize);
    } else if (group != 0u) {
        color = lerp(color, p.palette[group].xyz, 0.75);
        diameter = clamp(max(diameter + 2.0, diameter * 1.35), 1.0, maxSize);
    }

    // Requests use Vulkan Y-down clip and OpenGL depth. GpuKernelModule uses
    // NDC +Y up and a zero-to-one depth attachment on both backends.
    clip.y = -clip.y;
    clip.z = 0.5 * (clip.z + clip.w);
    clip.xy += corner * float2(diameter / float(p.extent.x), diameter / float(p.extent.y)) * clip.w;

    PointVertex result;
    result.position = clip;
    result.pointDelta = corner;
    result.color = color;
    result.viewDepth = -viewPosition.z;
    return result;
}

float normalizedDepth(float depth, float lo, float hi) {
    lo = max(lo, 1.0e-4);
    hi = max(hi, lo + 1.0e-4);
    depth = clamp(depth, lo, hi);
    const float linear = saturate((depth - lo) / max(hi - lo, 1.0e-5));
    const float logarithmic = saturate(log2(depth / lo) / max(log2(hi / lo), 1.0e-4));
    return smoothstep(0.0, 1.0, lerp(linear, logarithmic, smoothstep(1.75, 24.0, hi / lo)));
}

float3 depthPalette(float t) {
    t = saturate(t);
    const float3 far0 = float3(0.050, 0.040, 0.150), far1 = float3(0.060, 0.195, 0.500);
    const float3 mid0 = float3(0.000, 0.500, 0.650), mid1 = float3(0.360, 0.735, 0.410);
    const float3 near0 = float3(0.965, 0.820, 0.300), near1 = float3(0.985, 0.430, 0.125);
    if (t < 0.20) return lerp(far0, far1, smoothstep(0.00, 0.20, t));
    if (t < 0.43) return lerp(far1, mid0, smoothstep(0.20, 0.43, t));
    if (t < 0.67) return lerp(mid0, mid1, smoothstep(0.43, 0.67, t));
    if (t < 0.86) return lerp(mid1, near0, smoothstep(0.67, 0.86, t));
    return lerp(near0, near1, smoothstep(0.86, 1.00, t));
}

void discardOutsidePoint(PointVertex input) {
    if (dot(input.pointDelta, input.pointDelta) > 1.0)
        discard;
}

[shader("fragment")]
float4 pointColorFragment(PointVertex input) : SV_Target0 {
    discardOutsidePoint(input);
    float3 color = input.color;
    if (p.voxelFocalOrtho.w != 0.0) {
        const float hi = p.cropMax.w <= p.cropMin.w + 1.0e-5
                             ? p.cropMin.w + 1.0
                             : p.cropMax.w;
        const float t = 1.0 - normalizedDepth(input.viewDepth, p.cropMin.w, hi);
        color = (p.counts.z & 256u) != 0u ? float3(t) : depthPalette(t);
    }
    return float4(color, 1.0);
}

[shader("fragment")]
float4 pointDepthFragment(PointVertex input) : SV_Target0 {
    discardOutsidePoint(input);
    return float4(input.viewDepth, 0.0, 0.0, 0.0);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void extractPointDepth(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= p.extent.x || tid.y >= p.extent.y)
        return;
    const uint index = tid.y * p.extent.x + tid.x;
    p.linearDepth[index] = p.depthRgba[index].x;
}
