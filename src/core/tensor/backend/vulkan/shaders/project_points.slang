// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

static const uint LFS_LOCAL_SIZE_X = 256;
static const uint LFS_PUSH_CONSTANT_SIZE = 128;
[SpecializationConstant] const uint kModel = 0;
[SpecializationConstant] const uint kTransforms = 0;
[SpecializationConstant] const uint kIndices = 0;
[SpecializationConstant] const uint kVisibility = 0;
static const float kPi = 3.14159265358979323846f;

#include "../../../../include/core/rounded_math.slangh"

struct Parameters
{
    float4 row0;
    float4 row1;
    float4 row2;
    uint64_t points;
    uint64_t output;
    uint64_t transforms;
    uint64_t indices;
    uint64_t visibility;
    float2 scale;
    float2 center;
    float invalidValue;
    float nearDistance;
    uint count;
    uint transformCount;
    uint visibilityCount;
    uint padding;
};

#include "byte_read.slangh"

float4 viewPosition(Parameters p, uint i)
{
    const float4 invalid = float4(0, 0, 0, 0);
    int node = kIndices != 0 ? ((Ptr<int, Access.Read>)p.indices)[i] : 0;
    if (kVisibility != 0 && kIndices != 0)
    {
        if (node < 0 || uint(node) >= p.visibilityCount) return invalid;
        if (loadByte(p.visibility, uint(node)) == 0u) return invalid;
    }

    Ptr<float, Access.Read> points = (Ptr<float, Access.Read>)p.points;
    const uint64_t base = uint64_t(i) * 3;
    float3 position = float3(points[base], points[base + 1], points[base + 2]);
    if (kTransforms != 0 && p.transformCount > 0)
    {
        node = clamp(node, 0, int(p.transformCount - 1));
        Ptr<float, Access.Read> m = (Ptr<float, Access.Read>)(p.transforms + uint64_t(node) * 64);
        position = float3(
            roundedAdd(dotRounded(float3(m[0], m[1], m[2]), position), m[3]),
            roundedAdd(dotRounded(float3(m[4], m[5], m[6]), position), m[7]),
            roundedAdd(dotRounded(float3(m[8], m[9], m[10]), position), m[11]));
    }
    const float3 d = float3(roundedSub(position.x, p.row0.w), roundedSub(position.y, p.row1.w), roundedSub(position.z, p.row2.w));
    const float3 view = float3(dotRounded(p.row0.xyz, d), dotRounded(p.row1.xyz, d), dotRounded(p.row2.xyz, d));
    return all(isfinite(view)) ? float4(view, 1.0f) : invalid;
}

float2 projectFlat(Parameters p, float4 view)
{
    const float2 invalid = float2(p.invalidValue, p.invalidValue);
    if (view.w == 0.0f || view.z >= -p.nearDistance) return invalid;
    if (kModel == 1u)
    {
        if (!isfinite(p.scale.x) || p.scale.x <= 0.0f) return invalid;
        const float2 pixel = float2(roundedAdd(roundedMul(view.x, p.scale.x), p.center.x), roundedAdd(roundedMul(-view.y, p.scale.y), p.center.y));
        return pixel;
    }
    const float depth = -view.z;
    const float2 pixel = float2(
        roundedAdd(roundedDiv(roundedMul(view.x, p.scale.x), depth), p.center.x),
        roundedAdd(roundedDiv(roundedMul(-view.y, p.scale.y), depth), p.center.y));
    return pixel;
}

float2 projectPanorama(Parameters p, float4 direction)
{
    const float length = direction.w;
    if (!isfinite(length) || length <= p.nearDistance) return float2(p.invalidValue, p.invalidValue);
    const float3 dir = float3(roundedDiv(direction.x, length), roundedDiv(direction.y, length), roundedDiv(direction.z, length));
    const float bounded = clamp(dir.y, -1.0f, 1.0f);
    const float latitude = asin(bounded);
    return float2(
        roundedMul(roundedAdd(roundedDiv(atan2(dir.x, dir.z), 2.0f * kPi), 0.5f), p.scale.x),
        roundedMul(roundedAdd(roundedDiv(latitude, kPi), 0.5f), p.scale.y));
}

[numthreads(LFS_LOCAL_SIZE_X, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID, uniform Parameters p)
{
    const uint3 numWorkgroups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
    const uint stride = numWorkgroups.x * LFS_LOCAL_SIZE_X;
    Ptr<float2, Access.ReadWrite> output = (Ptr<float2, Access.ReadWrite>)p.output;
    for (uint i = dispatchThreadId.x; i < p.count; i += stride)
    {
        const float4 view = viewPosition(p, i);
        const float3 eq = float3(view.x, -view.y, -view.z);
        const float2 xy = kModel == 3u
            ? projectPanorama(p, float4(eq, view.w == 0.0f ? 0.0f : sqrt(dotRounded(eq, eq))))
            : projectFlat(p, view);
        output[i] = xy;
    }
}
