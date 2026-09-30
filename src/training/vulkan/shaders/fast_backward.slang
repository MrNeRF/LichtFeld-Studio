/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "fast_common.slangh"
[SpecializationConstant] const uint fastStage = 0;
struct BackPush {
    FastPush forward;
    uint64_t imageGradient, alphaGradient, depthGradient, normalGradient, screenGradient;
    uint64_t densification, error, edge, scores;
    uint64_t meanGradient, scaleGradient, rotationGradient, opacityGradient, dcGradient, shGradient;
    uint64_t sigmoid, z, u;
    uint densify, stage, slots, sparsityCount;
    float scaleWeight, flattenWeight, opacityWeight, rho, gradLoss;
    uint scaleElements, opacityElements;
};
struct Push {
    uint64_t parameters;
};
[[vk::push_constant]] Push push;
void atomicFloat(uint64_t address, uint index, float value) {
#if FAST_NATIVE_ATOMIC
    float previous;
    InterlockedAdd(((Ptr<float, Access.ReadWrite>)address)[index], value, previous);
#else
    Ptr<uint, Access.ReadWrite> words = (Ptr<uint, Access.ReadWrite>)address;
    uint observed = words[index];
    for (;;) {
        uint assumed = observed;
        InterlockedCompareExchange(words[index], assumed, asuint(asfloat(assumed) + value), observed);
        if (observed == assumed)
            break;
    }
#endif
}
float3 image3(uint64_t address, uint pixel, uint pixels) {
    return address == 0 ? float3(0) : float3(loadFloat(address, pixel), loadFloat(address, pixels + pixel), loadFloat(address, 2 * pixels + pixel));
}
float screen(BackPush b, uint work, uint field) { return loadFloat(b.screenGradient, work * 16 + field); }
float3 screen3(BackPush b, uint work, uint field) { return float3(screen(b, work, field), screen(b, work, field + 1), screen(b, work, field + 2)); }
#include "fast_projection_backward.slangh"
groupshared float reduction[16][128];
groupshared uint maxLast[128];
[numthreads(128, 1, 1)] void main(uint3 gid : SV_GroupID, uint lane : SV_GroupIndex) {
    BackPush b = ((Ptr<BackPush, Access.Read>)push.parameters)[0];
    b.stage = fastStage & 15;
    specializeFast(b.forward, fastStage);
    if ((fastStage & (1u << 18)) == 0)
        b.depthGradient = 0;
    if ((fastStage & (1u << 19)) == 0)
        b.normalGradient = 0;
    if ((fastStage & ((1u << 20) | (1u << 22))) == 0)
        b.densification = 0;
    if ((fastStage & (1u << 21)) == 0)
        b.scores = 0;
    FastPush p = b.forward;
    uint3 groups = spirv_asm {
result:
        $$uint3 = OpLoad builtin(NumWorkgroups : uint3);
    };
    if (b.stage == 2) {
        for (uint row = gid.x * 32 + lane / 4; row < p.count; row += groups.x * 32) {
            uint work = loadUint(p.originalToWork, row);
            if (work != 0xffffffffu)
                ((Ptr<float4, Access.ReadWrite>)b.screenGradient)[uint64_t(work) * 4 + lane % 4] = float4(0);
        }
        return;
    }
    if (b.stage == 1) {
        for (uint i = gid.x * 128 + lane; i < p.count; i += groups.x * 128)
            projectionBackward(b, i);
        return;
    }
    uint pixels = p.width * p.height;
    for (uint tile = gid.x; tile < p.gridWidth * p.gridHeight; tile += groups.x) {
        uint pixel[2], last[2];
        float2 center[2];
        float T[2], gT[2], gd[2];
        float3 gc[2], gn[2];
        [unroll] for (uint k = 0; k < 2; ++k) {
            uint rank = lane + k * 128;
            pixel[k] = pixelIndex(p, tile, rank);
            center[k] = float2(pixel[k] % p.width, pixel[k] / p.width) + 0.5f;
            bool valid = pixel[k] != 0xffffffffu;
            T[k] = loadFloat(p.transmittance, tile * 256 + rank);
            last[k] = valid ? loadUint(p.last, pixel[k]) : 0;
            gc[k] = valid ? image3(b.imageGradient, pixel[k], pixels) : float3(0);
            gn[k] = valid && p.renderNormal != 0 ? image3(b.normalGradient, pixel[k], pixels) : float3(0);
            gd[k] = valid && b.depthGradient != 0 ? loadFloat(b.depthGradient, pixel[k]) : 0;
            gT[k] = valid ? dot(gc[k], background(p, pixel[k])) - (b.alphaGradient != 0 ? loadFloat(b.alphaGradient, pixel[k]) : 0) : 0;
        }
        bool nativeWarp = WaveGetLaneCount() == 32;
        uint length;
        if (nativeWarp)
            length = WaveActiveMax(max(last[0], last[1]));
        else {
            maxLast[lane] = max(last[0], last[1]);
            GroupMemoryBarrierWithGroupSync();
            for (uint stride = 64; stride > 0; stride >>= 1) {
                if (lane < stride)
                    maxLast[lane] = max(maxLast[lane], maxLast[lane + stride]);
                GroupMemoryBarrierWithGroupSync();
            }
            length = maxLast[0];
        }
        uint begin = loadUint(p.ranges, tile * 2);
        uint warp = lane / 32;
        float2 subOrigin = float2((tile % p.gridWidth) * 16 + (warp % 2) * 8,
                                  (tile / p.gridWidth) * 16 + (warp / 2) * 4);
        for (uint batch = 0; batch < length; batch += nativeWarp ? 32 : 1) {
            uint reverse = batch + (nativeWarp ? WaveGetLaneIndex() : 0);
            uint fetchedWork = 0;
            Projected fetched = (Projected)0;
            bool hit0 = false, hit1 = false;
            if (reverse < length) {
                fetchedWork = loadUint(p.values, begin + length - reverse - 1);
                fetched = projectedAt(p, fetchedWork);
                hit0 = !nativeWarp || overlapsSubtile(fetched, subOrigin);
                hit1 = !nativeWarp || overlapsSubtile(fetched, subOrigin + float2(0, 8));
            }
            uint mask0 = nativeWarp ? WaveActiveBallot(hit0).x : 1;
            uint mask1 = nativeWarp ? WaveActiveBallot(hit1).x : 1;
            uint live = mask0 | mask1;
            while (live != 0) {
                uint selected = firstbitlow(live);
                live &= live - 1;
                uint ordinal = length - batch - selected - 1;
                uint work = nativeWarp ? WaveReadLaneAt(fetchedWork, selected) : fetchedWork;
                Projected o = nativeWarp ? broadcastProjected(fetched, selected, b.depthGradient != 0, p.renderNormal != 0 && b.normalGradient != 0) : fetched;
                uint original = loadUint(p.workToOriginal, work);
                float3 color = clamp(float3(o.red, o.green, o.blue), float3(0), float3(4));
                float3 colorGate = float3(o.red <= 4 ? 1 : 0, o.green <= 4 ? 1 : 0, o.blue <= 4 ? 1 : 0);
                float accum[16];
                [unroll] for (uint f = 0; f < 16; ++f)
                    accum[f] = 0;
                [unroll] for (uint k = 0; k < 2; ++k) {
                    if (((k == 0 ? mask0 : mask1) & (1u << selected)) == 0 || pixel[k] == 0xffffffffu || ordinal >= last[k])
                        continue;
                    float2 d = float2(o.mx, o.my) - center[k];
                    float h = 0.5f * (o.a * d.x * d.x + o.c * d.y * d.y) + o.b * d.x * d.y;
                    if (!(h >= 0))
                        continue;
                    float gaussian = exp(-h), unclamped = o.opacity * gaussian, alpha = minNumber(unclamped, 0.999f);
                    if (!(alpha >= 1.0f / 255))
                        continue;
                    float oneMinus = 1 - alpha, Tbefore = T[k] / max(oneMinus, 1e-4f), weight = Tbefore * alpha;
                    float normalDot = dot(float3(o.nx, o.ny, o.nz), gn[k]);
                    float ga = dot(Tbefore * color, gc[k]) - gT[k] * Tbefore + Tbefore * o.depth * gd[k] + Tbefore * normalDot;
                    float gh = unclamped >= 0.999f ? 0 : -alpha * ga;
                    accum[0] += gh * (o.a * d.x + o.b * d.y);
                    accum[1] += gh * (o.b * d.x + o.c * d.y);
                    accum[2] += 0.5f * gh * d.x * d.x;
                    accum[3] += 0.5f * gh * d.x * d.y;
                    accum[4] += 0.5f * gh * d.y * d.y;
                    accum[5] += unclamped >= 0.999f ? 0 : gaussian * ga;
                    float3 gcolor = weight * gc[k] * colorGate;
                    accum[6] += gcolor.x;
                    accum[7] += gcolor.y;
                    accum[8] += gcolor.z;
                    accum[9] += weight * gd[k];
                    accum[10] += weight * gn[k].x;
                    accum[11] += weight * gn[k].y;
                    accum[12] += weight * gn[k].z;
                    accum[13] += weight;
                    accum[14] += weight * (b.error != 0 ? loadFloat(b.error, pixel[k]) : 1);
                    accum[15] += weight * (b.edge != 0 ? loadFloat(b.edge, pixel[k]) : 0);
                    T[k] = Tbefore;
                    gT[k] = dot(gc[k], alpha * color) + alpha * o.depth * gd[k] + alpha * normalDot + gT[k] * oneMinus;
                }
                if (nativeWarp && !WaveActiveAnyTrue(accum[13] != 0))
                    continue;
                if (b.densify == 0 || b.densification == 0) {
                    accum[13] = 0;
                    accum[14] = 0;
                }
                if (b.scores == 0)
                    accum[15] = 0;
                if (nativeWarp) {
                    // Reduce and scatter fields to lanes in the same XOR tree.
                    uint wl = WaveGetLaneIndex();
                    [unroll] for (uint f = 0; f < 16; ++f)
                        accum[f] += WaveReadLaneAt(accum[f], wl ^ 16);
                    [unroll] for (uint stride = 8; stride > 0; stride >>= 1) {
                        [unroll] for (uint f = 0; f < stride; ++f) {
                            float own = (wl & stride) != 0 ? accum[f + stride] : accum[f];
                            float send = (wl & stride) != 0 ? accum[f] : accum[f + stride];
                            accum[f] = own + WaveReadLaneAt(send, wl ^ stride);
                        }
                    }
                    uint field = wl & 15;
                    float value = accum[0];
                    if (wl < 16 && value != 0) {
                        if (field < 13)
                            atomicFloat(b.screenGradient, work * 16 + field, clampGradient(value));
                        else if (field < 15 && b.densify != 0 && b.densification != 0)
                            atomicFloat(b.densification, (field - 13) * p.count + original, value);
                        else if (field == 15 && b.scores != 0)
                            atomicFloat(b.scores, original, value);
                    }
                } else {
                    // Logical 32-lane warps on other subgroup widths.
                    [unroll] for (uint f = 0; f < 16; ++f)
                        reduction[f][lane] = accum[f];
                    GroupMemoryBarrierWithGroupSync();
                    for (uint stride = 16; stride > 0; stride >>= 1) {
                        [unroll] for (uint f = 0; f < 16; ++f)
                            accum[f] = reduction[f][lane] + reduction[f][lane ^ stride];
                        GroupMemoryBarrierWithGroupSync();
                        [unroll] for (uint f = 0; f < 16; ++f)
                            reduction[f][lane] = accum[f];
                        GroupMemoryBarrierWithGroupSync();
                    }
                }
                if (!nativeWarp && lane % 32 == 0) {
                    [unroll] for (uint f = 0; f < 13; ++f) if (accum[f] != 0)
                        atomicFloat(b.screenGradient, work * 16 + f, clampGradient(accum[f]));
                    if (b.densify != 0 && b.densification != 0) {
                        atomicFloat(b.densification, original, accum[13]);
                        atomicFloat(b.densification, p.count + original, accum[14]);
                    }
                    if (b.scores != 0)
                        atomicFloat(b.scores, original, accum[15]);
                }
                if (!nativeWarp)
                    GroupMemoryBarrierWithGroupSync();
            }
        }
    }
}
