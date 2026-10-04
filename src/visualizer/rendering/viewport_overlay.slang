// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

// World and UI overlays of the viewport compositor, ported from
// resources/viewport/{overlay,shape_overlay,textured_overlay,pivot,frustum}.*.
// One parameter block serves every entry point so a frame's overlays share
// one raster batch. Overlay positions are Vulkan-convention NDC (+Y down)
// relative to the viewport panel; GPU programs use +Y up, so vertex shaders
// negate Y.
struct Parameters {
    // Per-vertex float4 records; `vertexStride` records per vertex.
    Ptr<float4, Access.Read> vertices;
    // Frustum instances: four model columns and a color, 5 records each.
    Ptr<float4, Access.Read> instances;
    Ptr<uint8_t, Access.Read> texture;
    // Linear view depth of the splats, [depthSize.y, depthSize.x].
    Ptr<float, Access.Read> depth;
    column_major float4x4 view;
    // Framebuffer pixel rectangle of the viewport panel.
    float4 viewportRect;
    // x: splat depth available, y: flip depth V, z: line thickness, w: projection mode.
    float4 depthParams;
    // Valid depth UV region: xy scale, zw clamp max.
    float4 uvRegion;
    // Frustum panel position and size, in the batch's render coordinates.
    float4 panel;
    // Frustum render size and focal lengths.
    float4 projection;
    // Textured tint/opacity, or pivot color/opacity.
    float4 tint;
    // Textured effects, or pivot center/size in NDC.
    float4 effects;
    // xy: texture size, zw: depth size.
    uint4 sizes;
    // x: float texture, y: CHW texture, z: first instance, w: vertex stride.
    uint4 flags;
};
[[vk::push_constant]] ConstantBuffer<Parameters> p;

static const float PI = 3.14159265358979323846;

float4 record(uint vertexId, uint index) {
    return p.vertices[vertexId * p.flags.w + index];
}

float4 ndcPosition(float2 position) {
    return float4(position.x, -position.y, 0.0, 1.0);
}

// Splat depth at a framebuffer pixel; 0 when none covers it.
float splatDepth(float2 fragCoord) {
    if (p.depthParams.x < 0.5 || p.sizes.z == 0 || p.sizes.w == 0)
        return 0.0;
    float2 viewport = max(p.viewportRect.zw, float2(1.0));
    float2 uv = (fragCoord - p.viewportRect.xy) / viewport;
    if (p.depthParams.y > 0.5)
        uv.y = 1.0 - uv.y;
    uv = min(uv * p.uvRegion.xy, p.uvRegion.zw);
    uint2 q = uint2(clamp(uv, 0.0, 1.0) * float2(p.sizes.zw - 1u) + 0.5);
    float value = p.depth[q.y * p.sizes.z + q.x];
    return value > 0.0 && value < 1.0e9 ? value : 0.0;
}

// --- Flat triangles -----------------------------------------------------------

struct OverlayOutput {
    float4 position : SV_Position;
    float4 color : COLOR0;
};

[shader("vertex")]
OverlayOutput overlayVertex(uint vertexId : SV_VertexID) {
    OverlayOutput result;
    result.position = ndcPosition(record(vertexId, 0).xy);
    result.color = record(vertexId, 1);
    return result;
}

[shader("fragment")]
float4 overlayFragment(OverlayOutput input) : SV_Target {
    return input.color;
}

// --- Anti-aliased segments, discs and rings -------------------------------------

struct ShapeOutput {
    float4 position : SV_Position;
    float2 screen : TEXCOORD0;
    float2 p0 : TEXCOORD1;
    float2 p1 : TEXCOORD2;
    float4 color : COLOR0;
    float4 shape : TEXCOORD3;
    float viewDepth : TEXCOORD4;
};

[shader("vertex")]
ShapeOutput shapeVertex(uint vertexId : SV_VertexID) {
    float4 a = record(vertexId, 0), b = record(vertexId, 1);
    ShapeOutput result;
    result.position = ndcPosition(a.xy);
    result.screen = a.zw;
    result.p0 = b.xy;
    result.p1 = b.zw;
    result.color = record(vertexId, 2);
    result.shape = record(vertexId, 3);
    result.viewDepth = record(vertexId, 4).x;
    return result;
}

float sdSegment(float2 point, float2 a, float2 b) {
    float2 pa = point - a, ba = b - a;
    float h = saturate(dot(pa, ba) / max(dot(ba, ba), 1e-6));
    return length(pa - ba * h);
}

[shader("fragment")]
float4 shapeFragment(ShapeOutput input) : SV_Target {
    float thickness = max(input.shape.y, 1.0);
    float radius = max(input.shape.z, 0.0);
    float aa = max(input.shape.w, 0.75);
    float distance = -aa;
    if (input.shape.x < 0.5)
        distance = sdSegment(input.screen, input.p0, input.p1) - thickness * 0.5;
    else if (input.shape.x < 1.5)
        distance = length(input.screen - input.p0) - radius;
    else if (input.shape.x < 2.5)
        distance = abs(length(input.screen - input.p0) - radius) - thickness * 0.5;
    float alpha = input.color.a * smoothstep(aa, -aa, distance);
    // Lines behind dense splats fade to a quarter so outlines stay readable.
    // View depth 0 marks UI overlays, which always draw in front.
    if (input.viewDepth > 0.0) {
        float splat = splatDepth(input.position.xy);
        if (splat > 0.0)
            alpha *= lerp(1.0, 0.25, smoothstep(0.01, 0.15, input.viewDepth - splat));
    }
    if (alpha <= 0.001)
        discard;
    return float4(input.color.rgb, alpha);
}

// --- Camera frusta: thick edge quads per instance --------------------------------

static const float3 FRUSTUM_POINTS[5] = {
    float3(-0.5, -0.5, -1.0), float3(0.5, -0.5, -1.0), float3(0.5, 0.5, -1.0),
    float3(-0.5, 0.5, -1.0), float3(0.0, 0.0, 0.0)
};
static const int2 FRUSTUM_EDGES[8] = {
    int2(0, 1), int2(1, 2), int2(2, 3), int2(3, 0),
    int2(0, 4), int2(1, 4), int2(2, 4), int2(3, 4)
};
static const float2 EDGE_QUAD[6] = {
    float2(0.0, 1.0), float2(1.0, 1.0), float2(1.0, -1.0),
    float2(0.0, 1.0), float2(1.0, -1.0), float2(0.0, -1.0)
};

ShapeOutput degenerateShape() {
    ShapeOutput result;
    result.position = float4(0.0, 0.0, 2.0, 1.0);
    result.screen = result.p0 = result.p1 = float2(0.0);
    result.color = float4(0.0);
    result.shape = float4(0.0, 1.0, 0.0, 1.0);
    result.viewDepth = 0.0;
    return result;
}

bool projectEquirect(float3 view, float2 panelPos, float2 panelSize, out float2 screen, out float depth) {
    depth = length(view);
    screen = float2(0.0);
    if (depth <= 1e-6)
        return false;
    float3 direction = view / depth;
    float ndcX = atan2(direction.x, -direction.z) / PI;
    float ndcY = -asin(clamp(direction.y, -1.0, 1.0)) / (PI * 0.5);
    screen = panelPos + float2((ndcX * 0.5 + 0.5) * panelSize.x, (ndcY * 0.5 + 0.5) * panelSize.y);
    return true;
}

bool segmentOutsidePanel(float2 a, float2 b, float2 panelPos, float2 panelSize, float extent) {
    float2 panelMin = panelPos - float2(extent), panelMax = panelPos + panelSize + float2(extent);
    float2 segmentMin = min(a, b), segmentMax = max(a, b);
    return segmentMax.x < panelMin.x || segmentMax.y < panelMin.y ||
           segmentMin.x > panelMax.x || segmentMin.y > panelMax.y;
}

[shader("vertex")]
ShapeOutput frustumVertex(uint vertexId : SV_VertexID, uint instanceId : SV_InstanceID) {
    uint edge = vertexId / 6u, corner = vertexId % 6u;
    uint base = (instanceId + p.flags.z) * 5u;
    float4x4 model = float4x4(p.instances[base], p.instances[base + 1u], p.instances[base + 2u], p.instances[base + 3u]);
    // Records are columns; float4x4(rows) needs a transpose.
    model = transpose(model);
    float4 color = p.instances[base + 4u];
    int2 e = FRUSTUM_EDGES[edge];
    float3 viewA = mul(p.view, mul(model, float4(FRUSTUM_POINTS[e.x], 1.0))).xyz;
    float3 viewB = mul(p.view, mul(model, float4(FRUSTUM_POINTS[e.y], 1.0))).xyz;

    float2 panelPos = p.panel.xy;
    float2 panelSize = max(p.panel.zw, float2(1.0));
    float2 renderSize = max(p.projection.xy, float2(1.0));
    float2 center = renderSize * 0.5;
    float2 scale = panelSize / renderSize;
    bool equirectangular = p.depthParams.w > 1.5;
    bool ortho = p.depthParams.w > 0.5 && p.depthParams.w < 1.5;

    if (!equirectangular) {
        const float minViewZ = -1e-4;
        bool aBehind = viewA.z >= minViewZ, bBehind = viewB.z >= minViewZ;
        if (aBehind && bBehind)
            return degenerateShape();
        if (aBehind) {
            float t = (minViewZ - viewA.z) / (viewB.z - viewA.z);
            viewA = lerp(viewA, viewB, saturate(t));
            viewA.z = minViewZ;
        } else if (bBehind) {
            float t = (minViewZ - viewB.z) / (viewA.z - viewB.z);
            viewB = lerp(viewB, viewA, saturate(t));
            viewB.z = minViewZ;
        }
    }

    float2 screenA, screenB;
    float depthA = 0.0, depthB = 0.0;
    if (equirectangular) {
        if (!projectEquirect(viewA, panelPos, panelSize, screenA, depthA) ||
            !projectEquirect(viewB, panelPos, panelSize, screenB, depthB))
            return degenerateShape();
        if (abs((screenA.x - screenB.x) / panelSize.x) > 0.5)
            return degenerateShape();
    } else if (ortho) {
        float orthoScale = max(p.projection.z, 1.0e-6);
        screenA = panelPos + float2(center.x + viewA.x * orthoScale, center.y - viewA.y * orthoScale) * scale;
        screenB = panelPos + float2(center.x + viewB.x * orthoScale, center.y - viewB.y * orthoScale) * scale;
        depthA = max(-viewA.z, 0.0);
        depthB = max(-viewB.z, 0.0);
    } else {
        float perspectiveA = max(-viewA.z, 1.0e-6), perspectiveB = max(-viewB.z, 1.0e-6);
        screenA = panelPos + float2(center.x + viewA.x * p.projection.z / perspectiveA,
                                    center.y - viewA.y * p.projection.w / perspectiveA) * scale;
        screenB = panelPos + float2(center.x + viewB.x * p.projection.z / perspectiveB,
                                    center.y - viewB.y * p.projection.w / perspectiveB) * scale;
        depthA = max(-viewA.z, 0.0);
        depthB = max(-viewB.z, 0.0);
    }

    float thickness = max(p.depthParams.z, 1.0);
    float extent = thickness * 0.5 + 2.0;
    if (segmentOutsidePanel(screenA, screenB, panelPos, panelSize, extent))
        return degenerateShape();
    float2 delta = screenB - screenA;
    float len = length(delta);
    float2 direction = len > 1e-4 ? delta / len : float2(1.0, 0.0);
    float2 normal = float2(-direction.y, direction.x);
    float2 quad = EDGE_QUAD[corner];
    float2 endpoint = quad.x < 0.5 ? screenA - direction * extent : screenB + direction * extent;
    float2 screen = endpoint + normal * (quad.y * extent);

    ShapeOutput result;
    result.screen = screen;
    result.p0 = screenA;
    result.p1 = screenB;
    result.color = color;
    result.shape = float4(0.0, thickness, 0.0, 1.0);
    result.viewDepth = quad.x < 0.5 ? depthA : depthB;
    float2 ndc = (screen - panelPos) / panelSize * 2.0 - 1.0;
    result.position = ndcPosition(ndc);
    return result;
}

// --- Textured quads: camera thumbnails, text and icons ----------------------------

struct TexturedOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
    float viewDepth : TEXCOORD1;
};

[shader("vertex")]
TexturedOutput texturedVertex(uint vertexId : SV_VertexID) {
    float4 a = record(vertexId, 0);
    TexturedOutput result;
    result.position = ndcPosition(a.xy);
    result.uv = a.zw;
    result.viewDepth = record(vertexId, 1).x;
    return result;
}

float texel(uint2 q, uint component) {
    uint2 size = p.sizes.xy;
    uint index = p.flags.y != 0 ? component * size.x * size.y + q.y * size.x + q.x
                                : (q.y * size.x + q.x) * 4u + component;
    return p.flags.x != 0 ? ((Ptr<float, Access.Read>)p.texture)[index]
                          : float(p.texture[index]) * (1.0 / 255.0);
}

float4 texel4(uint2 q) {
    return float4(texel(q, 0), texel(q, 1), texel(q, 2), texel(q, 3));
}

// Bilinear, clamp-to-edge, like the Vulkan overlay sampler.
float4 sampleBilinear(float2 uv) {
    float2 size = float2(max(p.sizes.xy, uint2(1u)));
    float2 position = clamp(uv, 0.0, 1.0) * size - 0.5;
    float2 base = floor(position);
    float2 f = position - base;
    int2 maxIndex = int2(size) - 1;
    uint2 q0 = uint2(clamp(int2(base), int2(0), maxIndex));
    uint2 q1 = uint2(clamp(int2(base) + 1, int2(0), maxIndex));
    float4 top = lerp(texel4(q0), texel4(uint2(q1.x, q0.y)), f.x);
    float4 bottom = lerp(texel4(uint2(q0.x, q1.y)), texel4(q1), f.x);
    return lerp(top, bottom, f.y);
}

[shader("fragment")]
float4 texturedFragment(TexturedOutput input) : SV_Target {
    // Thumbnails hide behind dense splats; view depth 0 is always in front.
    if (input.viewDepth > 0.0) {
        float splat = splatDepth(input.position.xy);
        if (splat > 0.0 && input.viewDepth > splat + 0.01)
            discard;
    }
    float4 sampled = sampleBilinear(input.uv);
    float3 rgb = lerp(sampled.rgb, p.tint.rgb, saturate(p.effects.x));
    rgb = lerp(rgb, float3(0.5), saturate(p.effects.y));
    return float4(rgb, sampled.a * saturate(p.tint.a));
}

// --- Pivot marker -----------------------------------------------------------------

static const float2 PIVOT_CORNERS[6] = {
    float2(-1.0, -1.0), float2(1.0, -1.0), float2(1.0, 1.0),
    float2(-1.0, -1.0), float2(1.0, 1.0), float2(-1.0, 1.0)
};

struct PivotOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
PivotOutput pivotVertex(uint vertexId : SV_VertexID) {
    float2 corner = PIVOT_CORNERS[vertexId];
    PivotOutput result;
    result.uv = corner;
    result.position = ndcPosition(p.effects.xy + corner * p.effects.zw);
    return result;
}

float pivotRing(float distance, float radius, float width, float edge) {
    float inner = radius - width * 0.5, outer = radius + width * 0.5;
    return smoothstep(inner - edge, inner + edge, distance) * (1.0 - smoothstep(outer - edge, outer + edge, distance));
}

[shader("fragment")]
float4 pivotFragment(PivotOutput input) : SV_Target {
    const float dotRadius = 0.06, ringWidth = 0.045, maxRadius = 0.92, edge = 0.015;
    float distance = length(input.uv);
    float opacity = p.tint.w;
    float progress = 1.0 - opacity;
    float flashIntensity = pow(max(0.0, 1.0 - progress * 4.0), 2.0);
    float flash = (1.0 - smoothstep(0.0, 0.25, distance)) * flashIntensity;
    float dotScale = 1.0 + 0.3 * sin(progress * 6.28) * (1.0 - progress);
    float radius = dotRadius * dotScale;
    float dotAlpha = (1.0 - smoothstep(radius - edge, radius + edge, distance)) * pow(opacity, 0.5);
    float ring1Progress = saturate(progress * 1.5);
    float ring1Radius = dotRadius + ring1Progress * (maxRadius - dotRadius);
    float ring1 = pivotRing(distance, ring1Radius, ringWidth, edge) * pow(1.0 - ring1Progress, 1.5);
    float ring2Progress = saturate((progress - 0.15) * 1.5);
    float ring2Radius = dotRadius + ring2Progress * (maxRadius * 0.85 - dotRadius);
    float ring2 = pivotRing(distance, ring2Radius, ringWidth * 0.7, edge) * pow(1.0 - ring2Progress, 1.5) * 0.6;
    float glow = exp(-pow((distance - ring1Radius * 0.95) * 4.0, 2.0)) * (1.0 - ring1Progress) * 0.3;
    float alpha = max(max(dotAlpha, flash), max(ring1 + ring2, glow));
    if (alpha < 0.01)
        discard;
    return float4(p.tint.rgb + float3(0.4) * (flash + dotAlpha * 0.3), alpha);
}
