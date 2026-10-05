// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

// Infinite ground grid, ported from resources/viewport/grid.{vert,frag}.
// Drawn into the panel's viewport; each pixel casts a ray between the near
// and far frustum planes and intersects the grid plane.
struct Parameters {
    float4 viewPositionPlane; // xyz camera position, w plane index (0 YZ, 1 XZ, 2 XY)
    float4 opacity;           // x
    float4 nearOrigin;        // bottom-left corner, xyz
    float4 nearX;
    float4 nearY;
    float4 farOrigin;
    float4 farX;
    float4 farY;
};
[[vk::push_constant]] ConstantBuffer<Parameters> p;

static const float2 QUAD[6] = {
    float2(-1.0, -1.0), float2(1.0, -1.0), float2(1.0, 1.0),
    float2(-1.0, -1.0), float2(1.0, 1.0), float2(-1.0, 1.0)
};
static const float3 AXIS_COLORS[3] = { float3(1.0, 0.2, 0.2), float3(0.2, 1.0, 0.2), float3(0.2, 0.2, 1.0) };
static const int AXIS0[3] = { 1, 0, 0 };
static const int AXIS1[3] = { 2, 2, 1 };

struct GridOutput {
    float4 position : SV_Position;
    float3 worldNear : TEXCOORD0;
    float3 worldFar : TEXCOORD1;
};

[shader("vertex")]
GridOutput gridVertex(uint vertexId : SV_VertexID) {
    float2 ndc = QUAD[vertexId];
    // NDC +Y is up here; (0,0) is the bottom-left frustum corner.
    float2 t = ndc * 0.5 + 0.5;
    GridOutput result;
    result.position = float4(ndc, 0.0, 1.0);
    result.worldNear = (p.nearOrigin + p.nearX * t.x + p.nearY * t.y).xyz;
    result.worldFar = (p.farOrigin + p.farX * t.x + p.farY * t.y).xyz;
    return result;
}

float pristineGrid(float2 uv, float2 dx, float2 dy, float2 lineWidth) {
    float2 uvDeriv = float2(length(float2(dx.x, dy.x)), length(float2(dx.y, dy.y)));
    bool2 invertLine = bool2(lineWidth.x > 0.5, lineWidth.y > 0.5);
    float2 targetWidth = float2(invertLine.x ? 1.0 - lineWidth.x : lineWidth.x,
                                invertLine.y ? 1.0 - lineWidth.y : lineWidth.y);
    float2 drawWidth = clamp(targetWidth, uvDeriv, float2(0.5));
    float2 lineAA = uvDeriv * 1.5;
    float2 gridUV = abs(frac(uv) * 2.0 - 1.0);
    gridUV.x = invertLine.x ? gridUV.x : 1.0 - gridUV.x;
    gridUV.y = invertLine.y ? gridUV.y : 1.0 - gridUV.y;
    float2 grid2 = smoothstep(drawWidth + lineAA, drawWidth - lineAA, gridUV);
    grid2 *= saturate(targetWidth / drawWidth);
    grid2 = lerp(grid2, targetWidth, saturate(uvDeriv * 2.0 - 1.0));
    grid2.x = invertLine.x ? 1.0 - grid2.x : grid2.x;
    grid2.y = invertLine.y ? 1.0 - grid2.y : grid2.y;
    return lerp(grid2.x, 1.0, grid2.y);
}

[shader("fragment")]
float4 gridFragment(GridOutput input) : SV_Target {
    int plane = clamp(int(p.viewPositionPlane.w), 0, 2);
    float3 origin = input.worldNear;
    float3 direction = normalize(input.worldFar - input.worldNear);
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
    float fade = (1.0 - smoothstep(400.0, 1000.0, length(world - p.viewPositionPlane.xyz))) *
                 saturate(p.opacity.x);
    const float epsilon = 1.0 / 255.0;
    if (fade < epsilon)
        discard;

    float2 levelPos = pos * 0.1;
    float levelSize = 2.0 / 1000.0;
    float levelAlpha = pristineGrid(levelPos, dx * 0.1, dy * 0.1, float2(levelSize)) * fade;
    if (levelAlpha > epsilon) {
        float2 loc = abs(levelPos);
        float2 axisDeriv = float2(length(float2(dx.x, dy.x)), length(float2(dx.y, dy.y))) * 0.1;
        float axisWidth = levelSize * 1.5;
        float axisX = 1.0 - smoothstep(axisWidth - axisDeriv.x, axisWidth + axisDeriv.x, loc.x);
        float axisY = 1.0 - smoothstep(axisWidth - axisDeriv.y, axisWidth + axisDeriv.y, loc.y);
        bool isAxisX = axisX > 0.01, isAxisY = axisY > 0.01;
        float3 color = float3(0.4);
        if (isAxisX && isAxisY)
            color = float3(1.0);
        else if (isAxisX)
            color = AXIS_COLORS[AXIS1[plane]];
        else if (isAxisY)
            color = AXIS_COLORS[AXIS0[plane]];
        float alpha = (isAxisX || isAxisY) ? max(axisX, axisY) * fade : levelAlpha;
        return float4(color, alpha);
    }

    levelPos = pos;
    levelSize = 1.0 / 100.0;
    levelAlpha = pristineGrid(levelPos, dx, dy, float2(levelSize)) * fade;
    if (levelAlpha > epsilon)
        return float4(float3(0.3), levelAlpha);

    levelPos = pos * 10.0;
    levelAlpha = pristineGrid(levelPos, dx * 10.0, dy * 10.0, float2(levelSize)) * fade;
    if (levelAlpha > epsilon)
        return float4(float3(0.3), levelAlpha);
    discard;
    return float4(0.0);
}
