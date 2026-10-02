/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "fast_common.slangh"
[SpecializationConstant] const uint fastStage = 0;
struct Push {
    uint64_t parameters;
};
[[vk::push_constant]] Push push;
// First fault wins. Synchronization reports it through the tensor runtime;
// every invalid access below is still suppressed on the device.
void tileFault(FastPush p, uint value, uint bound, uint operation) {
    Ptr<uint, Access.ReadWrite> words = (Ptr<uint, Access.ReadWrite>)p.status;
    uint observed;
    InterlockedCompareExchange(words[0], 0u, 3u, observed);
    if (observed == 0) {
        words[1] = value;
        words[2] = bound;
        words[3] = operation;
    }
}
Projected project(FastPush p, uint i, out uint touched) {
    Projected o = (Projected)0;
    touched = 0;
    float3 m = load3(p.means, i), s = load3(p.scales, i);
    float4 q = ((Ptr<float4, Access.Read>)p.rotations)[i];
    float4 v0 = ((Ptr<float4, Access.Read>)p.view)[0], v1 = ((Ptr<float4, Access.Read>)p.view)[1], v2 = ((Ptr<float4, Access.Read>)p.view)[2];
    float z = dot(v2.xyz, m) + v2.w;
    float rawOpacity = loadFloat(p.opacities, i);
    // Primitives with NaN/Inf geometry or opacity are culled on every backend.
    if (!finiteValue(m.x) || !finiteValue(m.y) || !finiteValue(m.z) || !finiteValue(s.x) || !finiteValue(s.y) ||
        !finiteValue(s.z) || !finiteValue(q.x) || !finiteValue(q.y) || !finiteValue(q.z) || !finiteValue(q.w) ||
        !finiteValue(rawOpacity))
        return o;
    float opacity = 1 / (1 + exp(-rawOpacity));
    if (z < 0.01f || z > 1e10f || opacity < 1.0f / 255 || dot(q, q) < 1e-8f)
        return o;
    float3 variance = exp(2 * min(s, float3(20)));
    float3x3 R = rotation(q);
    float3x3 scaled = float3x3(R[0] * variance, R[1] * variance, R[2] * variance);
    float3x3 cov = mul(scaled, transpose(R));
    float x = (dot(v0.xyz, m) + v0.w) / z, y = (dot(v1.xyz, m) + v1.w) / z;
    float jx = p.fx / z, jy = p.fy / z;
    float3 jw0 = jx * v0.xyz - jx * clamp(x, p.clipLeft, p.clipRight) * v2.xyz;
    float3 jw1 = jy * v1.xyz - jy * clamp(y, p.clipTop, p.clipBottom) * v2.xyz;
    float3 c0 = mul(jw0, cov), c1 = mul(jw1, cov);
    float a = dot(c0, jw0), b = dot(c0, jw1), c = dot(c1, jw1);
    float rawDet = p.mip != 0 ? max(a * c - b * b, 0.0f) : 0;
    float blur = p.mip != 0 ? 0.1f : 0.3f;
    a += blur;
    c += blur;
    float det = a * c - b * b;
    if (det < 1e-6f)
        return o;
    if (p.mip != 0)
        opacity *= sqrt(rawDet / det);
    if (opacity < 1.0f / 255)
        return o;
    o.mx = x * p.fx + p.cx;
    o.my = y * p.fy + p.cy;
    o.a = c / det;
    o.b = -b / det;
    o.c = a / det;
    o.opacity = opacity;
    o.depth = z;
    float r = sqrt(2 * log(opacity * 255));
    float ex = max(r * sqrt(a) - 0.5f, 0.0f), ey = max(r * sqrt(c) - 0.5f, 0.0f);
    o.xmin = min(p.gridWidth, uint(max(0, int(ceil((o.mx - ex) / 16)) - 1)));
    o.xmax = min(p.gridWidth, uint(max(0, int(floor((o.mx + ex) / 16)) + 1)));
    o.ymin = min(p.gridHeight, uint(max(0, int(ceil((o.my - ey) / 16)) - 1)));
    o.ymax = min(p.gridHeight, uint(max(0, int(floor((o.my + ey) / 16)) + 1)));
    bool alongX = (o.ymax - o.ymin) <= (o.xmax - o.xmin);
    uint start = alongX ? o.ymin : o.xmin, end = alongX ? o.ymax : o.xmax;
    for (uint scan = start; scan < end; ++scan) {
        uint2 span = tileSpan(o, alongX, scan);
        touched += span.y - span.x;
    }
    o.depthKey = touched;
    return o;
}
void shade(FastPush p, uint i, inout Projected o) {
    float3 m = load3(p.means, i);
    float basis[16];
    shBasis(direction(m - load3(p.camera, 0)), basis);
    float3 color = 0.5f + basis[0] * load3(p.sh0, i);
    [unroll] for (uint bidx = 1; bidx < 16; ++bidx) if (bidx < p.activeBases) color += basis[bidx] * shColor(p, i, bidx - 1);
    o.red = color.x;
    o.green = color.y;
    o.blue = color.z;
    if (p.renderNormal != 0) {
        float3 variance = exp(2 * min(load3(p.scales, i), float3(20)));
        float3x3 R = rotation(((Ptr<float4, Access.Read>)p.rotations)[i]);
        uint axis = variance.x <= variance.y && variance.x <= variance.z ? 0 : variance.y <= variance.z ? 1
                                                                                                        : 2;
        float3 normal = float3(R[0][axis], R[1][axis], R[2][axis]);
        if (dot(normal, m - load3(p.camera, 0)) > 0)
            normal = -normal;
        o.nx = dot(((Ptr<float4, Access.Read>)p.view)[0].xyz, normal);
        o.ny = dot(((Ptr<float4, Access.Read>)p.view)[1].xyz, normal);
        o.nz = dot(((Ptr<float4, Access.Read>)p.view)[2].xyz, normal);
    }
    float z = o.depth;
    float u = clamp((2 * z + 1) / (z + 1), 1.0f, asfloat(0x3fffffffu));
    o.depthKey = p.depthBits == 0 ? 0 : (asuint(u) & 0x7fffffu) >> (23 - p.depthBits);
}
void rasterSubgroup(FastPush p, uint tile, uint rank) {
    uint pixel = pixelIndex(p, tile, rank), begin = loadUint(p.ranges, tile * 2), end = loadUint(p.ranges, tile * 2 + 1);
    bool done = pixel == 0xffffffffu;
    float2 center = float2(pixel % p.width, pixel / p.width) + 0.5f;
    float2 origin = float2((tile % p.gridWidth) * 16 + (rank / 32 % 2) * 8, (tile / p.gridWidth) * 16 + (rank / 32 / 2) * 4);
    float T = 1, depth = 0;
    float3 rgb = 0, norm = 0;
    uint last = 0;
    for (uint batch = begin; batch < end; batch += 32) {
        if (WaveActiveAllTrue(done))
            break;
        uint at = batch + WaveGetLaneIndex();
        Projected fetched = (Projected)0;
        bool hit = false;
        if (at < end) {
            fetched = projectedAt(p, loadUint(p.values, at));
            hit = overlapsSubtile(fetched, origin);
        }
        uint live = WaveActiveBallot(hit).x;
        while (live != 0) {
            uint selected = firstbitlow(live);
            live &= live - 1;
            Projected o = broadcastProjected(fetched, selected, p.renderDepth != 0, p.renderNormal != 0);
            if (done)
                continue;
            float2 d = float2(o.mx, o.my) - center;
            float h = 0.5f * (o.a * d.x * d.x + o.c * d.y * d.y) + o.b * d.x * d.y;
            if (!(h >= 0))
                continue;
            float alpha = min(o.opacity * exp(-h), 0.999f);
            if (alpha < 1.0f / 255)
                continue;
            float weight = T * alpha;
            rgb += weight * clamp(float3(o.red, o.green, o.blue), float3(0), float3(4));
            if (p.renderDepth != 0)
                depth += weight * o.depth;
            if (p.renderNormal != 0)
                norm += weight * float3(o.nx, o.ny, o.nz);
            T *= 1 - alpha;
            last = batch + selected - begin + 1;
            done = T < 1e-4f;
        }
    }
    if (pixel != 0xffffffffu) {
        rgb += T * background(p, pixel);
        uint pixels = p.width * p.height;
        for (uint c = 0; c < 3; ++c) {
            storeFloat(p.image, c * pixels + pixel, rgb[c]);
            if (p.renderNormal != 0)
                storeFloat(p.normal, c * pixels + pixel, norm[c]);
        }
        storeFloat(p.alpha, pixel, 1 - T);
        if (p.renderDepth != 0)
            storeFloat(p.depth, pixel, depth);
        storeUint(p.last, pixel, last);
    }
    storeFloat(p.transmittance, tile * 256 + rank, T);
}
groupshared uint64_t scanValues[256];
[numthreads(256, 1, 1)] void main(uint3 gid : SV_GroupID, uint lane : SV_GroupIndex) {
    uint3 groups = spirv_asm {
result:
        $$uint3 = OpLoad builtin(NumWorkgroups : uint3);
    };
    FastPush p = ((Ptr<FastPush, Access.Read>)push.parameters)[0];
    specializeFast(p, fastStage);
    if (p.stage == 8) {
        if (gid.x == 0 && lane == 0) {
            uint64_t count = ((Ptr<uint64_t, Access.Read>)p.offsets)[p.visible - 1];
            uint overflow = count > uint64_t(0x7fffffffu) ? 2 : count > p.instances ? 1 : 0;
            Ptr<uint, Access.ReadWrite> args = (Ptr<uint, Access.ReadWrite>)p.visibility;
            args[0] = overflow == 0 ? uint(count) : 0;
            args[1] = overflow; args[2] = count != 0 ? 1 : 0;
            args[24] = uint(count); args[25] = uint(count >> 32);
            uint active = overflow == 0 && count != 0 ? 1 : 0;
            args[4] = active * min((p.visible + 255) / 256, p.padding); args[5] = 1; args[6] = 1;
            args[8] = active * ((uint(count) + 2047) / 2048); args[9] = 1; args[10] = 1;
            args[12] = active * min((uint(count) + 255) / 256, p.padding); args[13] = 1; args[14] = 1;
            args[16] = active * 256; args[17] = 1; args[18] = 1;
            args[20] = overflow == 0 ? min(p.gridWidth * p.gridHeight, p.padding) : 0; args[21] = 1; args[22] = 1;
            args[28] = active; args[29] = 1; args[30] = 1;
        }
        return;
    }
    if (p.padding != 0 && p.stage >= 3 && p.stage <= 5)
        p.instances = loadUint(p.visibility, 0);
    if (p.stage == 6) {
        uint blocks = (p.count + 255) / 256;
        for (uint block = gid.x; block < blocks; block += groups.x) {
            uint index = block * 256 + lane;
            if (WaveGetLaneCount() == 32) {
                uint64_t value = index < p.count ? ((Ptr<uint64_t, Access.Read>)p.counts)[index] : uint64_t(0);
                uint wl = WaveGetLaneIndex();
                [unroll] for (uint stride = 1; stride < 32; stride <<= 1) {
                    uint2 other = WaveReadLaneAt(uint2(uint(value), uint(value >> 32)), wl >= stride ? wl - stride : 0);
                    if (wl >= stride)
                        value += uint64_t(other.x) | (uint64_t(other.y) << 32);
                }
                if (wl == 31)
                    scanValues[lane / 32] = value;
                GroupMemoryBarrierWithGroupSync();
                for (uint warp = 0; warp < lane / 32; ++warp)
                    value += scanValues[warp];
                if (index < p.count)
                    ((Ptr<uint64_t, Access.ReadWrite>)p.offsets)[index] = value;
                if (lane == 255)
                    ((Ptr<uint64_t, Access.ReadWrite>)p.visibility)[block] = value;
                GroupMemoryBarrierWithGroupSync();
                continue;
            }
            scanValues[lane] = index < p.count ? ((Ptr<uint64_t, Access.Read>)p.counts)[index] : uint64_t(0);
            GroupMemoryBarrierWithGroupSync();
            for (uint stride = 1; stride < 256; stride <<= 1) {
                uint64_t value = scanValues[lane] + (lane >= stride ? scanValues[lane - stride] : uint64_t(0));
                GroupMemoryBarrierWithGroupSync();
                scanValues[lane] = value;
                GroupMemoryBarrierWithGroupSync();
            }
            if (index < p.count)
                ((Ptr<uint64_t, Access.ReadWrite>)p.offsets)[index] = scanValues[lane];
            if (lane == 255)
                ((Ptr<uint64_t, Access.ReadWrite>)p.visibility)[block] = scanValues[255];
            GroupMemoryBarrierWithGroupSync();
        }
        return;
    }
    if (p.stage == 7) {
        for (uint i = gid.x * 256 + lane; i < p.count; i += groups.x * 256)
            if (i >= 256)
                ((Ptr<uint64_t, Access.ReadWrite>)p.offsets)[i] += ((Ptr<uint64_t, Access.Read>)p.visibility)[i / 256 - 1];
        return;
    }
    if (p.stage == 5 && WaveGetLaneCount() == 32) {
        for (uint tile = gid.x; tile < p.gridWidth * p.gridHeight; tile += groups.x)
            rasterSubgroup(p, tile, lane);
        return;
    }
    uint work = p.stage == 4 ? p.instances : p.stage == 5 ? p.gridWidth * p.gridHeight * 256
                                         : p.stage == 1   ? p.count
                                         : p.stage == 0   ? p.count
                                                          : p.visible;
    for (uint i = gid.x * 256 + lane; i < work; i += groups.x * 256) {
        if (p.stage == 0 || p.stage == 2) {
            uint original = i;
            uint touched;
            Projected o;
            o = project(p, original, touched);
            if (p.stage == 2 && touched != 0)
                shade(p, original, o);
            if (p.stage == 0) {
                ((Ptr<uint64_t, Access.ReadWrite>)p.visibility)[i] = touched != 0 ? 1 : 0;
            } else {
                storeUint(p.originalToWork, i, touched != 0 ? i : 0xffffffffu);
                storeUint(p.workToOriginal, i, i);
                if (touched != 0)
                    ((Ptr<Projected, Access.ReadWrite>)p.projected)[i] = o;
                ((Ptr<uint64_t, Access.ReadWrite>)p.counts)[i] = touched;
                if (touched != 0 && p.screenShare != 0) {
                    float3 s = load3(p.scales, original);
                    float radius = exp(max(s.x, max(s.y, s.z))) * sqrt(2 * log(max(255 / (1 + exp(-loadFloat(p.opacities, original))), 1.0f)));
                    float d = length(load3(p.means, original) - load3(p.camera, 0));
                    float denominator = max(d, radius) + sqrt(max(d * d - radius * radius, 0.0f));
                    float share = radius > 0 && denominator > 0 ? clamp(radius / denominator, 0.0f, 1.0f) : 0;
                    storeFloat(p.screenShare, original, max(loadFloat(p.screenShare, original), share));
                }
            }
        } else if (p.stage == 1) {
            bool visible = ((Ptr<uint64_t, Access.Read>)p.visibility)[i] != 0;
            uint rank = visible ? uint(((Ptr<uint64_t, Access.Read>)p.offsets)[i] - 1) : 0xffffffffu;
            storeUint(p.originalToWork, i, rank);
            if (visible)
                storeUint(p.workToOriginal, rank, i);
        } else if (p.stage == 3) {
            uint64_t end = ((Ptr<uint64_t, Access.Read>)p.offsets)[i];
            uint64_t at = i == 0 ? 0 : ((Ptr<uint64_t, Access.Read>)p.offsets)[i - 1];
            if (end == at)
                continue;
            Projected o = projectedAt(p, i);
            bool alongX = (o.ymax - o.ymin) <= (o.xmax - o.xmin);
            uint first = alongX ? o.ymin : o.xmin, last = alongX ? o.ymax : o.xmax;
            for (uint scan = first; scan < last; ++scan) {
                uint2 span = tileSpan(o, alongX, scan);
                for (uint cross = span.x; cross < span.y; ++cross) {
                    uint tile = alongX ? scan * p.gridWidth + cross : cross * p.gridWidth + scan;
                    if (at >= end || at >= p.instances || tile >= p.gridWidth * p.gridHeight) {
                        tileFault(p, uint(at), p.instances, 1u);
                        continue;
                    }
                    storeUint(p.keys, uint(at), (tile << p.depthBits) | o.depthKey);
                    storeUint(p.values, uint(at), i);
                    ++at;
                }
            }
            if (at != end)
                tileFault(p, uint(at), uint(end), 2u);
        } else if (p.stage == 4) {
            uint tile = loadUint(p.keys, i) >> p.depthBits;
            if (tile >= p.gridWidth * p.gridHeight || loadUint(p.values, i) >= p.visible) {
                tileFault(p, tile >= p.gridWidth * p.gridHeight ? tile : loadUint(p.values, i),
                          tile >= p.gridWidth * p.gridHeight ? p.gridWidth * p.gridHeight : p.visible, 4u);
                continue;
            }
            if (i == 0 || (loadUint(p.keys, i - 1) >> p.depthBits) != tile)
                storeUint(p.ranges, 2 * tile, i);
            if (i + 1 == p.instances || (loadUint(p.keys, i + 1) >> p.depthBits) != tile)
                storeUint(p.ranges, 2 * tile + 1, i + 1);
        } else if (p.stage == 5) {
            uint tile = i / 256, rank = i % 256, pixel = pixelIndex(p, tile, rank);
            float T = 1;
            uint last = 0;
            float3 rgb = 0, norm = 0;
            float depth = 0;
            uint begin = loadUint(p.ranges, tile * 2), end = loadUint(p.ranges, tile * 2 + 1);
            if (pixel != 0xffffffffu) {
                float2 center = float2(pixel % p.width, pixel / p.width) + 0.5f;
                for (uint at = begin; at < end; ++at) {
                    Projected o = projectedAt(p, loadUint(p.values, at));
                    float2 d = float2(o.mx, o.my) - center;
                    float h = 0.5f * (o.a * d.x * d.x + o.c * d.y * d.y) + o.b * d.x * d.y;
                    if (!(h >= 0))
                        continue;
                    float alpha = min(o.opacity * exp(-h), 0.999f);
                    if (alpha < 1.0f / 255)
                        continue;
                    float weight = T * alpha;
                    rgb += weight * clamp(float3(o.red, o.green, o.blue), float3(0), float3(4));
                    if (p.renderDepth != 0)
                        depth += weight * o.depth;
                    if (p.renderNormal != 0)
                        norm += weight * float3(o.nx, o.ny, o.nz);
                    T *= 1 - alpha;
                    last = at - begin + 1;
                    if (T < 1e-4f)
                        break;
                }
                rgb += T * background(p, pixel);
                uint pixels = p.width * p.height;
                for (uint c = 0; c < 3; ++c) {
                    storeFloat(p.image, c * pixels + pixel, rgb[c]);
                    if (p.renderNormal != 0)
                        storeFloat(p.normal, c * pixels + pixel, norm[c]);
                }
                storeFloat(p.alpha, pixel, 1 - T);
                if (p.renderDepth != 0)
                    storeFloat(p.depth, pixel, depth);
                storeUint(p.last, pixel, last);
            }
            storeFloat(p.transmittance, i, T);
        }
    }
}
