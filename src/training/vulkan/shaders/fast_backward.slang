/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "fast_common.slangh"
[SpecializationConstant] const uint fastStage = 0;
struct BackPush {
    FastPush forward;
    uint64_t imageGradient, alphaGradient, depthGradient, normalGradient, screenGradient;
    uint64_t densification, error, edge, scores;
    uint64_t meanGradient, scaleGradient, rotationGradient, opacityGradient, dcGradient, shGradient;
    uint densify, stage, slots;
    float scaleWeight, flattenWeight, opacityWeight;
    uint scaleElements, opacityElements;
    uint64_t renderedCount, erankLoss, dcLoss, shLoss, scaleLoss;
    uint logScale;
    float scaleNormalizer, erankWeight, dcWeight, shWeight;
    uint64_t impactSum, impactPeak;
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
groupshared Projected stagedBackward[128];
groupshared uint backwardWork[128], backwardPosition[128];
groupshared float backwardOpacityInverse[128];
struct PixelWalk {
    uint pixel, last;
    float T, gT, gd, error, edge;
    float3 gc, gn, behind;
};
struct SplatSums {
    float h, hDy, u[12], peak;
    bool contributed;
};
SplatSums walkSplat(BackPush b, uint slot, float2 center, inout PixelWalk state[2]) {
    SplatSums sums = (SplatSums)0;
    Projected o = stagedBackward[slot];
    uint position = backwardPosition[slot];
    float3 color = clamp(float3(o.red, o.green, o.blue), float3(0), float3(4));
    float3 gate = float3(o.red <= 4 ? 1 : 0, o.green <= 4 ? 1 : 0, o.blue <= 4 ? 1 : 0);
    AlphaColumn column = alphaColumn(o, o.mx - center.x);
    float dy0 = o.my - center.y;
    float h[2] = {0, 0};
    [unroll] for (uint k = 0; k < 2; ++k) {
        if (state[k].pixel == 0xffffffffu || position >= state[k].last)
            continue;
        float unclamped = columnAlpha(column, dy0 - float(k));
        if (!(unclamped >= 1.0f / 255))
            continue;
        float alpha = min(unclamped, 0.999f), oneMinus = 1 - alpha;
        float before = state[k].T * rcp(oneMinus), weight = before * alpha;
        float normalDot = dot(float3(o.nx, o.ny, o.nz), state[k].gn);
        float3 colorGradient = weight * state[k].gc * gate;
        float ga = before * (dot(color, state[k].gc) + o.depth * state[k].gd + normalDot - state[k].gT);
        h[k] = unclamped >= 0.999f ? 0 : -alpha * ga;
        sums.u[1] += colorGradient.x;
        sums.u[2] += colorGradient.y;
        sums.u[3] += colorGradient.z;
        sums.u[4] += weight * state[k].gd;
        sums.u[5] += weight * state[k].gn.x;
        sums.u[6] += weight * state[k].gn.y;
        sums.u[7] += weight * state[k].gn.z;
        if (b.densify != 0 && b.densification != 0) {
            sums.u[8] += weight;
            sums.u[9] += weight * state[k].error;
        }
        if (b.scores != 0)
            sums.u[10] += weight * state[k].edge;
        if ((fastStage & (1u << 23)) != 0) {
            float3 difference = abs(color - state[k].behind);
            float removal = (fastStage & (1u << 24)) != 0 ? weight * weight * dot(difference, difference) :
                                                                          weight * (difference.x + difference.y + difference.z);
            sums.u[11] += removal;
            sums.peak = max(sums.peak, removal);
            state[k].behind = alpha * color + oneMinus * state[k].behind;
        }
        state[k].T = before;
        state[k].gT = alpha * (dot(color, state[k].gc) + o.depth * state[k].gd + normalDot) + state[k].gT * oneMinus;
        sums.contributed = true;
    }
    sums.h = h[0] + h[1];
    sums.hDy = fma(dy0, sums.h, -h[1]);
    sums.u[0] = 0.5f * fma(dy0, sums.hDy - h[1], h[1]);
    return sums;
}
int halvingIndex(uint n, uint firstBit, uint steps, uint lane, uint element) {
    uint sizes[3];
    [unroll] for (uint s = 0; s < steps; ++s) {
        sizes[s] = n;
        n = (n + 1) / 2;
    }
    uint index = element;
    for (int s = int(steps) - 1; s >= 0; --s) {
        index += (lane & (firstBit >> s)) != 0 ? (sizes[s] + 1) / 2 : 0;
        if (index >= sizes[s])
            return -1;
    }
    return int(index);
}
int sumField(int u) {
    return u < 0 ? -1 : u == 0 ? 4 : u + 5;
}
void storeField(BackPush b, uint slot, int field, float value) {
    if (field < 0 || value == 0)
        return;
    uint work = backwardWork[slot], original = loadUint(b.forward.workToOriginal, work);
    if (field == 16) {
        if ((fastStage & (1u << 23)) != 0)
            atomicFloat(b.impactSum, original, value);
    } else if ((fastStage & (1u << 24)) == 0) {
        if (field < 13)
            atomicFloat(b.screenGradient, work * 16 + uint(field), clampGradient(value));
        else if (field < 15 && b.densify != 0 && b.densification != 0)
            atomicFloat(b.densification, (uint(field) - 13) * b.forward.count + original, value);
        else if (field == 15 && b.scores != 0)
            atomicFloat(b.scores, original, value);
    }
}
void storePeak(BackPush b, uint slot, float peak) {
    if ((fastStage & (1u << 23)) != 0 && (fastStage & (1u << 24)) == 0 && peak > 0) {
        uint original = loadUint(b.forward.workToOriginal, backwardWork[slot]), previous;
        InterlockedMax(((Ptr<uint, Access.ReadWrite>)b.impactPeak)[original], asuint(peak), previous);
    }
}
// Screen field (0-12), densification (13, 14), edge (15) or impact (16) total
// that reduceSingle leaves in this lane, or -1.
int reducedField(uint lane) {
    uint row = lane / 8;
    int index = halvingIndex(5, 4, 3, lane, 0);
    return index == 0 ? int(row) : index == 1 ? (row == 0 ? 5 : -1) :
           index >= 2 ? sumField(halvingIndex(12, 16, 2, lane, uint(index - 2))) : -1;
}
void reduceSingle(BackPush b, uint slot, SplatSums sums, float centerX, int field) {
    uint lane = WaveGetLaneIndex(), row = lane / 8;
    float h = sums.h, hDy = sums.hDy, u[12];
    [unroll] for (uint f = 0; f < 12; ++f)
        u[f] = sums.u[f];
    uint n = 12;
    [unroll] for (uint bit = 16; bit >= 8; bit >>= 1) {
        uint half = (n + 1) / 2;
        [unroll] for (uint f = 0; f < half; ++f) {
            float low = u[f], high = f + half < n ? u[f + half] : 0;
            u[f] = ((lane & bit) != 0 ? high : low) + WaveReadLaneAt((lane & bit) != 0 ? low : high, lane ^ bit);
        }
        n = half;
        h += WaveReadLaneAt(h, lane ^ bit);
        hDy += WaveReadLaneAt(hDy, lane ^ bit);
    }
    Projected o = stagedBackward[slot];
    float dx = o.mx - centerX, hdx = dx * h;
    float values[5];
    values[0] = row == 0 ? o.a * hdx + o.b * hDy : row == 1 ? o.b * hdx + o.c * hDy :
                row == 2 ? 0.5f * dx * hdx : 0.5f * dx * hDy;
    values[1] = row == 0 ? -h * backwardOpacityInverse[slot] : 0;
    [unroll] for (uint f = 0; f < 3; ++f)
        values[2 + f] = u[f];
    n = 5;
    [unroll] for (uint bit = 4; bit > 0; bit >>= 1) {
        uint half = (n + 1) / 2;
        [unroll] for (uint f = 0; f < half; ++f) {
            float low = values[f], high = f + half < n ? values[f + half] : 0;
            values[f] = ((lane & bit) != 0 ? high : low) + WaveReadLaneAt((lane & bit) != 0 ? low : high, lane ^ bit);
        }
        n = half;
    }
    storeField(b, slot, field, values[0]);
    if ((fastStage & (1u << 23)) != 0) {
        float peak = WaveActiveMax(sums.peak);
        if (lane == 0)
            storePeak(b, slot, peak);
    }
}
// Back-to-front walk of one tile by four independent 32-lane subgroups
// (kFastQuadrants). Each subgroup stages, 32 list entries at a time, only the
// splats that the forward recorded as blended in its two 8x4 regions, into its
// own slice of shared memory, so no workgroup barrier is needed. Per splat the
// dx-independent sums are reduced over each pixel column before the column
// terms of the mean and conic gradients are formed.
void backwardQuadrants(BackPush b, uint tile, uint thread) {
    FastPush p = b.forward;
    uint lane = WaveGetLaneIndex(), warpBase = thread & ~31u, warp = thread / 32;
    uint rank0 = quadrantRank(thread, 0), pixels = p.width * p.height;
    float2 center = float2((tile % p.gridWidth) * 16 + rank0 % 16,
                          (tile / p.gridWidth) * 16 + rank0 / 16) + 0.5f;
    PixelWalk state[2];
    [unroll] for (uint k = 0; k < 2; ++k) {
        state[k] = (PixelWalk)0;
        uint rank = quadrantRank(thread, k), pixel = quadrantPixel(p, tile, rank);
        bool valid = pixel != 0xffffffffu;
        state[k].pixel = pixel;
        state[k].last = valid ? loadUint(p.last, pixel) : 0;
        state[k].T = loadFloat(p.transmittance, tile * 256 + rank);
        if (valid) {
            state[k].gc = image3(b.imageGradient, pixel, pixels);
            state[k].gn = p.renderNormal != 0 ? image3(b.normalGradient, pixel, pixels) : float3(0);
            state[k].gd = b.depthGradient != 0 ? loadFloat(b.depthGradient, pixel) : 0;
            state[k].gT = dot(state[k].gc, background(p, pixel)) - (b.alphaGradient != 0 ? loadFloat(b.alphaGradient, pixel) : 0);
            state[k].error = b.error != 0 ? loadFloat(b.error, pixel) : 1;
            state[k].edge = b.edge != 0 ? loadFloat(b.edge, pixel) : 0;
            if ((fastStage & (1u << 23)) != 0)
                state[k].behind = background(p, pixel);
        }
    }
    uint topRegion = (warp % 2) + 4 * (warp / 2), regions = (1u << topRegion) | (1u << (topRegion + 2));
    uint begin = loadUint(p.ranges, tile * 2), end = loadUint(p.ranges, tile * 2 + 1);
    uint last = WaveActiveMax(max(state[0].last, state[1].last));
    int field = reducedField(lane);
    // Each subgroup stages only its covered entries into its own 32 slots.
    for (uint remaining = min(end - begin, last); remaining > 0; remaining = remaining > 32 ? remaining - 32 : 0) {
        bool staged = false;
        uint slot = warpBase + lane;
        if (lane < remaining) {
            uint at = begin + remaining - lane - 1;
            // The forward stores (regions reached, original list position)
            // for each kept entry.
            uint2 contributor = ((Ptr<uint2, Access.Read>)p.contributors)[at];
            if ((contributor.x & regions) != 0) {
                uint position = contributor.y;
                if (position < last) {
                    uint work = loadUint(p.values, at);
                    backwardWork[slot] = work;
                    backwardPosition[slot] = position;
                    Projected o = projectedAt(p, work);
                    stagedBackward[slot] = o;
                    backwardOpacityInverse[slot] = rcp(o.opacity);
                    staged = true;
                }
            }
        }
        uint mask = WaveActiveBallot(staged).x;
        subgroupBarrier();
        // Back to front over the staged splats, one reduction each. Sharing the
        // first exchange between two splats made this pass 7-9% slower, as the
        // disabled pairing in the CUDA kernel suggests.
        while (mask != 0) {
            uint selected = warpBase + firstbitlow(mask);
            mask &= mask - 1;
            SplatSums sums = walkSplat(b, selected, center, state);
            if (WaveActiveAnyTrue(sums.contributed))
                reduceSingle(b, selected, sums, center.x, field);
        }
        subgroupBarrier();
    }
}

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
    // Per-splat removal impact: how much each pixel changes without the splat.
    // POP reuses the impact walk, with squared RGB energy and no gradient writes.
    const bool pop = (fastStage & (1u << 24)) != 0;
    const bool impact = (fastStage & (1u << 23)) != 0;
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
    if ((fastStage & kFastQuadrants) != 0) {
        for (uint tile = gid.x; tile < p.gridWidth * p.gridHeight; tile += groups.x)
            backwardQuadrants(b, tile, lane);
        return;
    }
    uint pixels = p.width * p.height;
    for (uint tile = gid.x; tile < p.gridWidth * p.gridHeight; tile += groups.x) {
        uint pixel[2], last[2];
        float2 center[2];
        float T[2], gT[2], gd[2];
        float3 gc[2], gn[2], behind[2];
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
            behind[k] = valid && impact ? background(p, pixel[k]) : float3(0);
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
                float impactSum = 0, impactPeak = 0;
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
                    if (impact) {
                        float3 difference = abs(color - behind[k]);
                        float removal = pop ? weight * weight * dot(difference, difference) : weight * (difference.x + difference.y + difference.z);
                        impactSum += removal;
                        impactPeak = max(impactPeak, removal);
                        behind[k] = alpha * color + oneMinus * behind[k];
                    }
                    T[k] = Tbefore;
                    gT[k] = dot(gc[k], alpha * color) + alpha * o.depth * gd[k] + alpha * normalDot + gT[k] * oneMinus;
                }
                if (nativeWarp && !WaveActiveAnyTrue(accum[13] != 0))
                    continue;
                if (impact) {
                    if (nativeWarp || pop) {
                        impactSum = WaveActiveSum(impactSum);
                        impactPeak = WaveActiveMax(impactPeak);
                    }
                    if ((!nativeWarp && !pop) || WaveGetLaneIndex() == 0) {
                        if (impactSum != 0)
                            atomicFloat(b.impactSum, original, impactSum);
                        if (!pop && impactPeak > 0) {
                            uint previous;
                            InterlockedMax(((Ptr<uint, Access.ReadWrite>)b.impactPeak)[original], asuint(impactPeak), previous);
                        }
                    }
                }
                if (pop)
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
