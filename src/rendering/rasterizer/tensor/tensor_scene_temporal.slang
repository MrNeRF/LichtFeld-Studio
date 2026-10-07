// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

float4 mul4(float4 c0, float4 c1, float4 c2, float4 c3, float4 v) {
    return c0 * v.x + c1 * v.y + c2 * v.z + c3 * v.w;
}

struct Parameters {
    Ptr<float, Access.Read> depth;
    Ptr<float2, Access.ReadWrite> motionOutput;
    Ptr<uint8_t, Access.Read> current;
    Ptr<float4, Access.Read> history;
    Ptr<float2, Access.Read> motionInput;
    Ptr<float, Access.Read> currentDepth;
    Ptr<float, Access.Read> historyDepth;
    Ptr<float4, Access.ReadWrite> output;
    Ptr<uint8_t, Access.Read> samples;
    Ptr<uint8_t, Access.ReadWrite> oracleOutput;
#if LFS_TEMPORAL_PASS == 0
    float4 inverseCurrentViewProjection0;
    float4 inverseCurrentViewProjection1;
    float4 inverseCurrentViewProjection2;
    float4 inverseCurrentViewProjection3;
    float4 previousViewProjection0;
    float4 previousViewProjection1;
    float4 previousViewProjection2;
    float4 previousViewProjection3;
    uint4 renderInfo;
    float4 depthInfo;
#elif LFS_TEMPORAL_PASS == 1
    uint4 extents;
    uint4 currentLayout;
    float4 control;
    float4 currentUv;
    float4 depthControl;
    float4 jitterPixels;
    float4 reconstruction;
#else
    uint count;
    uint3 padding;
#endif
};
[[vk::push_constant]] ConstantBuffer<Parameters> p;

#if LFS_TEMPORAL_PASS == 0
float viewDepthToNdc(float viewDepth) {
    const float nearPlane = p.depthInfo.x;
    const float farPlane = p.depthInfo.y;
    if (!(viewDepth > 0.0f) || !(nearPlane > 0.0f) || !(farPlane > nearPlane))
        return 1.0f;
    if (p.renderInfo.w == 2u)
        return (viewDepth - nearPlane) / (farPlane - nearPlane);
    return farPlane / (farPlane - nearPlane) *
           (1.0f - nearPlane / max(viewDepth, nearPlane));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void scene_motion(uint3 tid : SV_DispatchThreadID) {
    const uint2 pixel = tid.xy;
    const uint2 extent = p.renderInfo.xy;
    if (any(pixel >= extent))
        return;
    uint2 depthPixel = pixel;
    if (p.depthInfo.z > 0.5f)
        depthPixel.y = extent.y - 1u - depthPixel.y;
    uint depthStride = p.depthInfo.w > 0.0f ? uint(p.depthInfo.w) : extent.x;
    float depth = p.depth[depthPixel.y * depthStride + depthPixel.x];
    if (p.renderInfo.w != 0u)
        depth = viewDepthToNdc(depth);
    if (!isfinite(depth) || depth < 0.0f || depth >= 1.0f) {
        p.motionOutput[pixel.y * extent.x + pixel.x] = float2(0.0f);
        return;
    }
    const float2 size = float2(extent);
    const float2 currentNdc = float2(
        2.0f * (float(pixel.x) + 0.5f) / size.x - 1.0f,
        1.0f - 2.0f * (float(pixel.y) + 0.5f) / size.y);
    float4 world = mul4(p.inverseCurrentViewProjection0,
                        p.inverseCurrentViewProjection1,
                        p.inverseCurrentViewProjection2,
                        p.inverseCurrentViewProjection3,
                        float4(currentNdc, depth, 1.0f));
    if (!isfinite(world.w) || abs(world.w) <= 1e-7f) {
        p.motionOutput[pixel.y * extent.x + pixel.x] = float2(0.0f);
        return;
    }
    world /= world.w;
    const float4 previousClip = mul4(p.previousViewProjection0,
                                     p.previousViewProjection1,
                                     p.previousViewProjection2,
                                     p.previousViewProjection3, world);
    if (!all(isfinite(previousClip.xyw)) || abs(previousClip.w) <= 1e-7f) {
        p.motionOutput[pixel.y * extent.x + pixel.x] = float2(0.0f);
        return;
    }
    const float2 previousNdc = previousClip.xy / previousClip.w;
    float2 motion = float2(
        (previousNdc.x - currentNdc.x) * 0.5f * size.x,
        (currentNdc.y - previousNdc.y) * 0.5f * size.y);
    if (p.renderInfo.z != 0u)
        motion.y = -motion.y;
    p.motionOutput[pixel.y * extent.x + pixel.x] = motion;
}

#endif

struct ResolveCoreInput {
    float4 current, history;
    float3 neighborhoodMin, neighborhoodMax, neighborhoodCrossSum;
    float2 currentPixelCenter, motion, currentJitter, previousJitter;
    uint2 motionExtent, outputExtent;
    float currentDepth, historyDepth, farPlane;
    float historyWeight, depthRelative, depthAbsolute, motionRejection;
    float motionConfidence, currentSharpness;
    uint historyValid, depthAvailable;
};
struct ResolveCoreResult {
    float4 color;
    float2 currentRenderUv, previousRenderUv, previousUv;
    float weight;
    uint rejection;
};

ResolveCoreResult resolveCore(ResolveCoreInput s) {
    ResolveCoreResult r = (ResolveCoreResult)0;
    r.color = s.current;
    if (!all(isfinite(s.current)) || !all(isfinite(s.currentPixelCenter)) ||
        any(s.motionExtent == uint2(0u)) || any(s.outputExtent == uint2(0u)) ||
        !all(isfinite(s.neighborhoodMin)) || !all(isfinite(s.neighborhoodMax)) ||
        !all(isfinite(s.neighborhoodCrossSum))) {
        r.color = float4(0.0f);
        r.rejection = 2u;
        return r;
    }
    const float3 lower = min(s.neighborhoodMin, s.neighborhoodMax);
    const float3 upper = max(s.neighborhoodMin, s.neighborhoodMax);
    const float sharpness = clamp(s.currentSharpness, 0.0f, 0.25f);
    const float3 reconstructed = clamp(
        s.current.rgb * (1.0f + 4.0f * sharpness) - s.neighborhoodCrossSum * sharpness,
        lower, upper);
    r.color = float4(reconstructed, s.current.a);
    if (s.historyValid == 0u) {
        r.rejection = 1u;
        return r;
    }
    const float motionLimit = max(0.0f, s.motionRejection);
    if (!all(isfinite(s.motion))) {
        r.rejection = 4u;
        return r;
    }
    const float motionLength = length(s.motion);
    if (!isfinite(motionLength) || motionLength > motionLimit) {
        r.rejection = 4u;
        return r;
    }
    const float2 renderExtent = float2(s.motionExtent);
    r.currentRenderUv = s.currentPixelCenter / float2(s.outputExtent) +
                        s.currentJitter / renderExtent;
    r.previousUv = s.currentPixelCenter / float2(s.outputExtent) + s.motion / renderExtent;
    r.previousRenderUv = r.previousUv + s.previousJitter / renderExtent;
    const float2 minimumUv = 0.5f / float2(s.outputExtent);
    const float2 maximumUv = 1.0f - minimumUv;
    if (!all(isfinite(r.previousUv)) || any(r.previousUv < minimumUv) ||
        any(r.previousUv > maximumUv)) {
        r.rejection = 5u;
        return r;
    }
    if (s.depthAvailable != 0u) {
        const float2 renderMinimumUv = 0.5f / renderExtent;
        const float2 renderMaximumUv = 1.0f - renderMinimumUv;
        if (any(r.previousRenderUv < renderMinimumUv) ||
            any(r.previousRenderUv > renderMaximumUv) ||
            !isfinite(s.currentDepth) || !isfinite(s.historyDepth) ||
            !isfinite(s.farPlane) || !(s.currentDepth > 0.0f) ||
            !(s.historyDepth > 0.0f) || !(s.farPlane > 0.0f) ||
            s.currentDepth >= s.farPlane || s.historyDepth >= s.farPlane) {
            r.rejection = 6u;
            return r;
        }
        const float threshold = max(max(0.0f, s.depthAbsolute),
                                    max(0.0f, s.depthRelative) * s.currentDepth);
        if (abs(s.currentDepth - s.historyDepth) > threshold) {
            r.rejection = 6u;
            return r;
        }
    }
    if (!all(isfinite(s.history))) {
        r.rejection = 3u;
        return r;
    }
    const float3 clampedHistory = clamp(s.history.rgb, lower, upper);
    const float confidenceSpan = max(0.0f, s.motionConfidence);
    const float confidence = motionLimit > 0.0f && confidenceSpan > 0.0f &&
                                     isfinite(motionLength)
                                 ? clamp(1.0f - motionLength / confidenceSpan, 0.0f, 1.0f)
                                 : 0.0f;
    r.weight = clamp(s.historyWeight, 0.0f, 1.0f) * confidence;
    r.color = float4(lerp(reconstructed, clampedHistory, r.weight), s.current.a);
    return r;
}

#if LFS_TEMPORAL_PASS == 1
float4 loadCurrent(uint2 pixel) {
    const uint index = (pixel.y * p.currentLayout.x + pixel.x) * p.currentLayout.z;
    if (p.currentLayout.w != 0u)
        return ((Ptr<float4, Access.Read>)p.current)[index / 4u];
    return float4(p.current[index], p.current[index + 1u],
                  p.current[index + 2u], p.current[index + 3u]) *
           (1.0f / 255.0f);
}

float4 sampleCurrent(float2 uv) {
    const uint2 extent = p.currentLayout.xy;
    const float2 pixel = clamp(uv * float2(extent) - 0.5f, 0.0f, float2(extent - 1u));
    const uint2 a = uint2(floor(pixel));
    const uint2 b = min(a + 1u, extent - 1u);
    const float2 f = frac(pixel);
    return lerp(lerp(loadCurrent(a), loadCurrent(uint2(b.x, a.y)), f.x),
                lerp(loadCurrent(uint2(a.x, b.y)), loadCurrent(b), f.x), f.y);
}

float4 sampleFloat4(Ptr<float4, Access.Read> source, uint2 extent, float2 uv) {
    const float2 pixel = clamp(uv * float2(extent) - 0.5f, 0.0f, float2(extent - 1u));
    const uint2 a = uint2(floor(pixel));
    const uint2 b = min(a + 1u, extent - 1u);
    const float2 f = frac(pixel);
    const float4 c00 = source[a.y * extent.x + a.x];
    const float4 c10 = source[a.y * extent.x + b.x];
    const float4 c01 = source[b.y * extent.x + a.x];
    const float4 c11 = source[b.y * extent.x + b.x];
    return lerp(lerp(c00, c10, f.x), lerp(c01, c11, f.x), f.y);
}

float2 sampleFloat2(Ptr<float2, Access.Read> source, uint2 extent, float2 uv) {
    const float2 pixel = clamp(uv * float2(extent) - 0.5f, 0.0f, float2(extent - 1u));
    const uint2 a = uint2(floor(pixel));
    const uint2 b = min(a + 1u, extent - 1u);
    const float2 f = frac(pixel);
    return lerp(lerp(source[a.y * extent.x + a.x], source[a.y * extent.x + b.x], f.x),
                lerp(source[b.y * extent.x + a.x], source[b.y * extent.x + b.x], f.x), f.y);
}

float sampleFloat(Ptr<float, Access.Read> source, uint2 extent, float2 uv) {
    const float2 pixel = clamp(uv * float2(extent) - 0.5f, 0.0f, float2(extent - 1u));
    const uint2 a = uint2(floor(pixel));
    const uint2 b = min(a + 1u, extent - 1u);
    const float2 f = frac(pixel);
    return lerp(lerp(source[a.y * extent.x + a.x], source[a.y * extent.x + b.x], f.x),
                lerp(source[b.y * extent.x + a.x], source[b.y * extent.x + b.x], f.x), f.y);
}

float2 sampleFloat2Nearest(Ptr<float2, Access.Read> source, uint2 extent, float2 uv) {
    const uint2 pixel = min(uint2(clamp(uv, 0.0f, 1.0f) * float2(extent)), extent - 1u);
    return source[pixel.y * extent.x + pixel.x];
}

float sampleFloatNearest(Ptr<float, Access.Read> source, uint2 extent, float2 uv) {
    const uint2 pixel = min(uint2(clamp(uv, 0.0f, 1.0f) * float2(extent)), extent - 1u);
    return source[pixel.y * extent.x + pixel.x];
}

[shader("compute")]
[numthreads(8, 8, 1)]
void scene_spatial(uint3 tid : SV_DispatchThreadID) {
    const uint2 pixel = tid.xy;
    const uint2 renderExtent = p.extents.xy;
    const uint2 outputExtent = p.extents.zw;
    if (any(pixel >= outputExtent))
        return;
    const float2 outputUv = (float2(pixel) + 0.5f) / float2(outputExtent);
    const float2 uv = min(outputUv * p.currentUv.xy, p.currentUv.zw);
    const float2 texel = 1.0f / float2(p.currentLayout.xy);
    const float4 center = sampleCurrent(clamp(uv, 0.0f, p.currentUv.zw));
    const float4 left = sampleCurrent(clamp(uv - float2(texel.x, 0.0f), 0.0f, p.currentUv.zw));
    const float4 right = sampleCurrent(clamp(uv + float2(texel.x, 0.0f), 0.0f, p.currentUv.zw));
    const float4 up = sampleCurrent(clamp(uv - float2(0.0f, texel.y), 0.0f, p.currentUv.zw));
    const float4 down = sampleCurrent(clamp(uv + float2(0.0f, texel.y), 0.0f, p.currentUv.zw));
    const float strength = 0.18f;
    const float3 sharpened = center.rgb * (1.0f + 4.0f * strength) -
                             (left.rgb + right.rgb + up.rgb + down.rgb) * strength;
    const float3 neighborhoodMin = min(center.rgb, min(min(left.rgb, right.rgb), min(up.rgb, down.rgb)));
    const float3 neighborhoodMax = max(center.rgb, max(max(left.rgb, right.rgb), max(up.rgb, down.rgb)));
    p.output[pixel.y * outputExtent.x + pixel.x] =
        float4(clamp(sharpened, neighborhoodMin, neighborhoodMax), center.a);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void scene_temporal_resolve(uint3 tid : SV_DispatchThreadID) {
    const uint2 pixel = tid.xy;
    const uint2 renderExtent = p.extents.xy;
    const uint2 outputExtent = p.extents.zw;
    if (any(pixel >= outputExtent))
        return;
    const float2 outputUv = (float2(pixel) + 0.5f) / float2(outputExtent);
    const float2 renderHalfTexel = 0.5f / float2(renderExtent);
    const float2 currentRenderUv = outputUv + p.jitterPixels.xy / float2(renderExtent);
    const float2 motionUv = clamp(currentRenderUv, renderHalfTexel, 1.0f - renderHalfTexel);
    const float2 currentColorUv = clamp(currentRenderUv * p.currentUv.xy,
                                        0.0f, p.currentUv.zw);
    const float4 current = sampleCurrent(currentColorUv);
    float3 neighborhoodMin = current.rgb;
    float3 neighborhoodMax = current.rgb;
    float3 crossSum = float3(0.0f);
    const float2 colorTexel = p.currentUv.xy / float2(renderExtent);
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            const float2 sampleUv = clamp(currentColorUv + float2(x, y) * colorTexel,
                                          0.0f, p.currentUv.zw);
            const float3 value = sampleCurrent(sampleUv).rgb;
            neighborhoodMin = min(neighborhoodMin, value);
            neighborhoodMax = max(neighborhoodMax, value);
            if (abs(x) + abs(y) == 1)
                crossSum += value;
        }
    }
    const float2 motion = sampleFloat2Nearest(p.motionInput, renderExtent, motionUv);
    const float2 previousUv = outputUv + motion / float2(renderExtent);
    const float2 previousRenderUv = previousUv + p.jitterPixels.zw / float2(renderExtent);
    ResolveCoreInput input = (ResolveCoreInput)0;
    input.current = current;
    input.history = p.control.x >= 0.5f
                        ? sampleFloat4(p.history, outputExtent, previousUv)
                        : float4(0.0f);
    input.neighborhoodMin = neighborhoodMin;
    input.neighborhoodMax = neighborhoodMax;
    input.neighborhoodCrossSum = crossSum;
    input.currentPixelCenter = float2(pixel) + 0.5f;
    input.motion = motion;
    input.currentJitter = p.jitterPixels.xy;
    input.previousJitter = p.jitterPixels.zw;
    input.motionExtent = renderExtent;
    input.outputExtent = outputExtent;
    input.currentDepth = p.depthControl.x > 0.5f
                             ? sampleFloatNearest(p.currentDepth, renderExtent, motionUv)
                             : 0.0f;
    input.historyDepth = p.depthControl.x > 0.5f
                             ? sampleFloatNearest(p.historyDepth, renderExtent, previousRenderUv)
                             : 0.0f;
    input.farPlane = p.depthControl.w;
    input.historyWeight = p.control.y;
    input.depthRelative = p.depthControl.y;
    input.depthAbsolute = p.depthControl.z;
    input.motionRejection = p.control.z;
    input.motionConfidence = p.reconstruction.y;
    input.currentSharpness = p.reconstruction.x;
    input.historyValid = p.control.x >= 0.5f ? 1u : 0u;
    input.depthAvailable = p.depthControl.x > 0.5f ? 1u : 0u;
    p.output[pixel.y * outputExtent.x + pixel.x] = resolveCore(input).color;
}

#endif

#if LFS_TEMPORAL_PASS == 2
struct OracleSample {
    float4 current, history;
    float4 neighborhoodMin, neighborhoodMax, neighborhoodCrossSum;
    float4 pixelMotion;
    float4 jitter;
    uint4 extents;
    float4 depth;
    float4 settings0, settings1;
};
struct OracleResult { float4 color, uv, status; };
[shader("compute")]
[numthreads(64, 1, 1)]
void scene_temporal_resolve_samples(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= p.count)
        return;
    const OracleSample s = ((Ptr<OracleSample, Access.Read>)p.samples)[tid.x];
    ResolveCoreInput input = (ResolveCoreInput)0;
    input.current = s.current;
    input.history = s.history;
    input.neighborhoodMin = s.neighborhoodMin.xyz;
    input.neighborhoodMax = s.neighborhoodMax.xyz;
    input.neighborhoodCrossSum = s.neighborhoodCrossSum.xyz;
    input.currentPixelCenter = s.pixelMotion.xy;
    input.motion = s.pixelMotion.zw;
    input.currentJitter = s.jitter.xy;
    input.previousJitter = s.jitter.zw;
    input.motionExtent = s.extents.xy;
    input.outputExtent = s.extents.zw;
    input.currentDepth = s.depth.x;
    input.historyDepth = s.depth.y;
    input.farPlane = s.depth.z;
    input.historyValid = s.depth.w > 0.5f ? 1u : 0u;
    input.historyWeight = s.settings0.x;
    input.depthRelative = s.settings0.y;
    input.depthAbsolute = s.settings0.z;
    input.motionRejection = s.settings0.w;
    input.motionConfidence = s.settings1.x;
    input.currentSharpness = s.settings1.y;
    input.depthAvailable = s.settings1.z > 0.5f ? 1u : 0u;
    const ResolveCoreResult r = resolveCore(input);
    OracleResult result;
    result.color = r.color;
    result.uv = float4(r.currentRenderUv, r.previousRenderUv);
    result.status = float4(r.previousUv, r.weight, float(r.rejection));
    ((Ptr<OracleResult, Access.ReadWrite>)p.oracleOutput)[tid.x] = result;
}

#endif
