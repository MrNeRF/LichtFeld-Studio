// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

// Adaptive ground grid, ported from resources/viewport/grid.{vert,frag}.
// Drawn into the panel's viewport; each pixel casts a ray and intersects the
// grid plane. The grid is a floor around the orbit pivot that fades outward.
struct Parameters {
    float4 planeOpacityOrtho; // x plane index (0 YZ, 1 XZ, 2 XY), y opacity, z 1 for orthographic
    float4 pivotRadius;       // xyz orbit pivot, w floor radius
    // Pixel rays across the screen from the bottom-left corner. Directions are kept apart from
    // the origin: two far-away points subtracted in float lose the direction when zoomed out.
    float4 rayOrigin;
    float4 rayOriginX;
    float4 rayOriginY;
    float4 rayDir;
    float4 rayDirX;
    float4 rayDirY;
};
[[vk::push_constant]] ConstantBuffer<Parameters> p;

static const float2 QUAD[6] = {
    float2(-1.0, -1.0), float2(1.0, -1.0), float2(1.0, 1.0),
    float2(-1.0, -1.0), float2(1.0, 1.0), float2(-1.0, 1.0)
};
// Axis colours match the navigation gizmo.
static const float3 AXIS_COLORS[3] = { float3(0.89, 0.15, 0.21), float3(0.54, 0.86, 0.20), float3(0.17, 0.48, 0.87) };
static const int AXIS0[3] = { 1, 0, 0 };
static const int AXIS1[3] = { 2, 2, 1 };

static const float3 LINE_COLOR = float3(0.58, 0.64, 0.74);
static const float MINOR_ALPHA = 0.32;
static const float MAJOR_ALPHA = 0.7;
static const float AXIS_ALPHA = 1.0;
// Lines closer than this many pixels are not drawn; each decade fades in above it.
static const float MIN_SPACING_PX = 8.0;

struct GridOutput {
    float4 position : SV_Position;
    float3 rayOrigin : TEXCOORD0;
    float3 rayDir : TEXCOORD1;
};

[shader("vertex")]
GridOutput gridVertex(uint vertexId : SV_VertexID) {
    float2 ndc = QUAD[vertexId];
    // NDC +Y is up here; (0,0) is the bottom-left screen corner.
    float2 t = ndc * 0.5 + 0.5;
    GridOutput result;
    result.position = float4(ndc, 0.0, 1.0);
    result.rayOrigin = (p.rayOrigin + p.rayOriginX * t.x + p.rayOriginY * t.y).xyz;
    result.rayDir = (p.rayDir + p.rayDirX * t.x + p.rayDirY * t.y).xyz;
    return result;
}

// Coverage of lines `widthPx` wide every `spacing` world units, filtered over the pixel footprint
// along each plane axis: lines thinner than a pixel keep their average brightness instead of
// breaking up, and lines closer than two pixels fade to their average coverage instead of moire.
float lineCoverage(float2 pos, float2 worldPerPx, float spacing, float widthPx) {
    float2 footprint = worldPerPx / spacing;
    float2 target = min(widthPx * footprint, float2(1.0));
    float2 drawn = clamp(target, footprint, float2(0.5));
    float2 edge = footprint * 1.5;
    float2 dist = abs(frac(pos / spacing + 0.5) - 0.5) * 2.0;
    float2 cov = (1.0 - smoothstep(drawn - edge, drawn + edge, dist)) * saturate(target / drawn);
    cov = lerp(cov, target, saturate(footprint * 2.0 - 1.0));
    return lerp(cov.x, 1.0, cov.y);
}

// The same filter for the single line through the origin along each plane axis.
float2 axisCoverage(float2 pos, float2 worldPerPx, float widthPx) {
    float2 distPx = abs(pos) / worldPerPx;
    float drawn = max(widthPx, 1.0);
    return (1.0 - smoothstep(float2(drawn * 0.5 - 0.75), float2(drawn * 0.5 + 0.75), distPx)) * min(widthPx, 1.0);
}

// x = log10(spacing / MIN_SPACING_PX): a decade fades in as a minor level, then becomes major.
float levelAlpha(float x) {
    return lerp(MINOR_ALPHA * smoothstep(0.0, 1.0, x), MAJOR_ALPHA, smoothstep(1.0, 2.0, x));
}

float4 over(float4 src, float4 dst) {
    float a = src.a + dst.a * (1.0 - src.a);
    float3 rgb = (src.rgb * src.a + dst.rgb * dst.a * (1.0 - src.a)) / max(a, 1e-6);
    return float4(rgb, a);
}

[shader("fragment")]
float4 gridFragment(GridOutput input) : SV_Target {
    int plane = clamp(int(p.planeOpacityOrtho.x + 0.5), 0, 2);
    float3 origin = input.rayOrigin;
    float3 direction = normalize(input.rayDir);
    float3 normal = plane == 0 ? float3(1, 0, 0) : (plane == 1 ? float3(0, 1, 0) : float3(0, 0, 1));
    float d = dot(direction, normal);
    if (abs(d) < 1e-6)
        discard;
    float t = -dot(origin, normal) / d;
    if (t < 0.0)
        discard;

    float3 world = origin + direction * t;
    float2 pos = plane == 0 ? world.yz : (plane == 1 ? world.xz : world.xy);
    float2 dx = ddx(pos);
    float2 dy = ddy(pos);
    float2 worldPerPx = max(float2(length(float2(dx.x, dy.x)), length(float2(dx.y, dy.y))), float2(1e-20));

    // Soft horizon: fade out where the view ray grazes the plane. Orthographic rays are all
    // parallel, so there is no horizon and the floor fade alone bounds the grid.
    float horizon = p.planeOpacityOrtho.z > 0.5 ? 1.0 : smoothstep(0.0, 0.18, abs(d));
    float fade = horizon * saturate(p.planeOpacityOrtho.y);
    // The grid is a floor under the orbit pivot that dissolves outward.
    float3 pivot = p.pivotRadius.xyz;
    float3 pivotOnPlane = pivot - normal * dot(pivot, normal);
    float radius = max(p.pivotRadius.w, 1e-6);
    float radial = 1.0 - smoothstep(0.1 * radius, radius, length(world - pivotOnPlane));

    float lod = log2(max(worldPerPx.x, worldPerPx.y) * MIN_SPACING_PX) * 0.30103;
    float firstLevel = floor(lod) + 1.0;
    float gridAlpha = 0.0;
    for (int k = 0; k < 3; ++k) {
        float level = firstLevel + float(k);
        gridAlpha = max(gridAlpha, lineCoverage(pos, worldPerPx, pow(10.0, level), 1.0) * levelAlpha(level - lod));
    }

    // Axes reach further out than the grid lines.
    float axisRadial = sqrt(radial);
    float2 axisCov = axisCoverage(pos, worldPerPx, 2.0);
    float4 color = float4(LINE_COLOR, gridAlpha * radial);
    color = over(float4(AXIS_COLORS[AXIS0[plane]], axisCov.y * AXIS_ALPHA * axisRadial), color);
    color = over(float4(AXIS_COLORS[AXIS1[plane]], axisCov.x * AXIS_ALPHA * axisRadial), color);
    color.a *= fade;
    if (color.a < 1.0 / 255.0)
        discard;
    return color;
}
