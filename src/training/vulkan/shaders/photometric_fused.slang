/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
[SpecializationConstant] const uint operation = 0;
struct Params {
    uint64_t corrected, raw, target, mask, horizontal, partials;
    uint64_t ssim, cs, grad, gradRaw, losses, normalizer;
    uint count, channels, height, width, batch, stage, path, targetByte;
    uint maskByte, validPadding, partialStride, reserved;
    float weight, denominator;
};
struct Push {
    uint64_t parameters;
};
[[vk::push_constant]] Push push;
static const float weights[6] = {0.26601171493530273f, 0.21300552785396576f,
                                 0.10936068743467331f, 0.036000773310661316f, 0.0075987582094967365f, 0.001028380123898387f};
float readFloat(uint64_t address, uint i) { return ((Ptr<float, Access.Read>)address)[i]; }
void writeFloat(uint64_t address, uint i, float v) { ((Ptr<float, Access.ReadWrite>)address)[i] = v; }
bool decoupled(Params p) { return (operation >> 3u) == 3 || (operation >> 3u) == 5; }
bool masked(Params p) { return (operation >> 3u) >= 4; }
float targetAt(Params p, uint i) {
    return p.targetByte != 0 ? float(((Ptr<uint8_t, Access.Read>)p.target)[i]) * (1.0f / 255.0f) : readFloat(p.target, i);
}
float maskAt(Params p, uint i) {
    if (!masked(p))
        return 1.0f;
    return p.maskByte != 0 ? (((Ptr<uint8_t, Access.Read>)p.mask)[i] != 0 ? 1.0f : 0.0f) : readFloat(p.mask, i);
}
float moment(Params p, uint base, int x, int y, uint field) {
    if (x < 0 || y < 0 || x >= int(p.width) || y >= int(p.height))
        return 0;
    uint i = base + uint(y) * p.width + uint(x);
    float a = readFloat(p.corrected, i), t = targetAt(p, i);
    float r = decoupled(p) ? readFloat(p.raw, i) : a;
    if (field == 0)
        return a;
    if (field == 1)
        return r;
    if (field == 2)
        return r * r;
    if (field == 3)
        return t;
    if (field == 4)
        return t * t;
    return r * t;
}
float chain(Params p, int x, int y) {
    if (x < 0 || y < 0 || x >= int(p.width) || y >= int(p.height))
        return 0;
    if (masked(p))
        return 1.0f / (p.denominator > 0.0f ? p.denominator : readFloat(p.normalizer, 0) * float(p.batch) * float(p.channels) + 1e-8f);
    bool valid = (operation >> 3u) != 0 && p.validPadding != 0;
    bool cropX = valid && p.width > 10, cropY = valid && p.height > 10;
    if ((cropX && (x < 5 || x >= int(p.width) - 5)) ||
        (cropY && (y < 5 || y >= int(p.height) - 5)))
        return 0;
    uint width = cropX ? p.width - 10 : p.width;
    uint height = cropY ? p.height - 10 : p.height;
    return 1.0f / float(p.batch * p.channels * width * height);
}
void storePartial(Params p, uint i, uint field, float value) {
    if (p.partials == 0)
        return;
    ((Ptr<uint16_t, Access.ReadWrite>)p.partials)[field * p.partialStride + i] = uint16_t(f32tof16(value));
}
float partial(Params p, uint base, int x, int y, uint field) {
    if (x < 0 || y < 0 || x >= int(p.width) || y >= int(p.height))
        return 0;
    uint i = base + uint(y) * p.width + uint(x);
    float value = f16tof32(((Ptr<uint16_t, Access.Read>)p.partials)[field * p.partialStride + i]);
    float weight = decoupled(p) && field > 0 ? 1.0f : p.weight;
    return -weight * value * chain(p, x, y);
}
// Match CUDA's weighted pair: round the right product and the pair before
// adding it to the running sum. Explicit fma/precise preserve those boundaries.
float accumulateWeightedPair(float sum, float left, float right, float wl, float wr) {
    precise float rightProduct = right * wr;
    precise float result = sum + fma(left, wl, rightProduct);
    return result;
}
void finishMoments(Params p, uint i, float m[7], bool weighted, float invMass, float derivativeWeight) {
    float a = m[0], r = m[1], t = m[3];
    float A, B, covariance;
    if (weighted) {
        a *= invMass;
        r *= invMass;
        t *= invMass;
        // CUDA rounds mean products, then fuses normalization with subtraction.
        // Rounding normalized second moments first loses precision near flat windows.
        precise float aa = a * a, rr = r * r, tt = t * t, rt = r * t;
        precise float luminanceDenominator = (aa + tt) + 0.01f * 0.01f;
        A = luminanceDenominator;
        B = fma(m[2], invMass, -rr) + fma(m[4], invMass, -tt) + 0.03f * 0.03f;
        covariance = fma(m[5], invMass, -rt);
    } else {
        A = fma(a, a, t * t) + 0.01f * 0.01f;
        B = fma(-r, r, m[2]) + fma(-t, t, m[4]) + 0.03f * 0.03f;
        covariance = fma(-r, t, m[5]);
    }
    float C = fma(2 * a, t, 0.01f * 0.01f), D = fma(2.0f, covariance, 0.03f * 0.03f);
    float invAB = 1.0f / (A * B), invAAB = 1.0f / (A * A * B), invABB = 1.0f / (A * B * B);
    float similarity;
    if (decoupled(p)) {
        float lum = C / A, cs = D / B;
        similarity = lum * cs;
        storePartial(p, i, 0, cs * (2 * (t * A - a * C) / (A * A)) * derivativeWeight);
        float dv = p.weight * (-(lum * D) / (B * B));
        float dc = p.weight * ((2 * lum) / B);
        storePartial(p, i, 1, (dv * (-2 * r) + dc * (-t)) * derivativeWeight);
        storePartial(p, i, 2, dv * derivativeWeight);
        storePartial(p, i, 3, dc * derivativeWeight);
    } else {
        similarity = (C * D) * invAB;
        if (masked(p)) {
            float dmu = (t * 2 * D) / (A * B) - (t * 2 * C) / (A * B) -
                        (a * 2 * C * D) / (A * A * B) + (a * 2 * C * D) / (A * B * B);
            storePartial(p, i, 0, dmu * derivativeWeight);
            storePartial(p, i, 1, ((-C * D) / (A * B * B)) * derivativeWeight);
            storePartial(p, i, 2, ((2 * C) / (A * B)) * derivativeWeight);
        } else {
            storePartial(p, i, 0, ((t * 2 * D) * invAB - (t * 2 * C) * invAB - (a * 2 * C * D) * invAAB + (a * 2 * C * D) * invABB) * derivativeWeight);
            storePartial(p, i, 1, (-C * D) * invABB * derivativeWeight);
            storePartial(p, i, 2, (2 * C) * invAB * derivativeWeight);
        }
    }
    float centerWeight = maskAt(p, i % (p.width * p.height));
    writeFloat(p.ssim, i, masked(p) && centerWeight <= 0.0f ? 1.0f : similarity);
    if (p.cs != 0)
        writeFloat(p.cs, i, masked(p) && centerWeight <= 0.0f ? 1.0f : D / B);

    if (p.losses != 0) {
        uint pixels = p.width * p.height;
        float a = readFloat(p.corrected, i), t = targetAt(p, i);
        float loss = (1 - p.weight) * abs(a - t) + p.weight * (1 - similarity);
        writeFloat(p.losses, i, loss * maskAt(p, i % pixels));
    }
}
void finishGradient(Params p, uint i, int x, int y, float m[4]) {
    float a = readFloat(p.corrected, i), t = targetAt(p, i);
    float centerWeight = maskAt(p, uint(y) * p.width + uint(x));
    float l1 = (1 - p.weight) * sign(a - t) * chain(p, x, y) * centerWeight;
    writeFloat(p.grad, i, l1 + centerWeight * (decoupled(p) ? m[0] : fma(t, m[2], fma(2 * a, m[1], m[0]))));
    if (decoupled(p))
        writeFloat(p.gradRaw, i, centerWeight * fma(t, m[3], fma(2 * readFloat(p.raw, i), m[2], m[1])));
}
// Both separable passes stay in one workgroup. Match CUDA's weighted moment
// rounding and retain half-rounded partials and zero padding at image edges.
// Keep the existing device workspace sizes to preserve allocation placement.
// The horizontal pass writes 16 columns across 26 halo rows (416 entries).
// Specialize scratch fields to avoid reserving masked/decoupled workspace
// for paths that never access it; arithmetic and sample order stay unchanged.
static const uint tileFields = PHOTO_MOMENTS != 0 ? ((operation >> 3u) >= 4 ? 7 : 6) : (((operation >> 3u) == 3 || (operation >> 3u) == 5) ? 4 : 3);
groupshared float verticalTile[tileFields][416];
static const uint inputFields = PHOTO_MOMENTS != 0 && (operation >> 3u) < 4 ? 3 : 4;
groupshared float fusedInput[inputFields][676];
// One past the last tile whose mask window held a value other than one.
groupshared uint weightedTileEnd;
void fusedFilter(Params p, uint group, uint lane, uint groupCount) {
    uint columns = (p.width + 15) / 16, rows = (p.height + 15) / 16;
    uint tiles = columns * rows * p.batch * p.channels, pixels = p.width * p.height;
    static const bool moments = PHOTO_MOMENTS != 0;
    uint fields = moments ? (masked(p) ? 7 : 6) : decoupled(p) ? 4
                                             : 3;
    if (lane == 0)
        weightedTileEnd = 0;
    GroupMemoryBarrierWithGroupSync();
    for (uint tile = group; tile < tiles; tile += groupCount) {
        uint base = (tile / (columns * rows)) * pixels;
        int ox = int((tile % columns) * 16), oy = int(((tile / columns) % rows) * 16);
        float invMaskScale = 1.0f;
        if (moments && masked(p)) {
            uint anchorX = min(uint(ox + 8), p.width - 1), anchorY = min(uint(oy + 8), p.height - 1);
            invMaskScale = 1.0f / max(1.0f, maskAt(p, anchorY * p.width + anchorX));
        }
        bool partialSupport = false;
        for (uint j = lane; j < 676; j += 256) {
            int x = ox + int(j % 26) - 5, y = oy + int(j / 26) - 5;
            if (moments) {
                fusedInput[0][j] = moment(p, base, x, y, 0);
                fusedInput[1][j] = decoupled(p) ? moment(p, base, x, y, 1) : fusedInput[0][j];
                fusedInput[2][j] = moment(p, base, x, y, 3);
                if (masked(p)) {
                    float weight = (x >= 0 && y >= 0 && x < int(p.width) && y < int(p.height)
                                        ? maskAt(p, uint(y) * p.width + uint(x)) : 1.0f) * invMaskScale;
                    fusedInput[3][j] = weight;
                    partialSupport = partialSupport || weight != 1.0f;
                }
            } else {
                [loop] for (uint f = 0; f < 4; ++f) if (f < fields) fusedInput[f][j] = partial(p, base, x, y, f);
            }
        }
        if (moments && masked(p) && WaveActiveAnyTrue(partialSupport) && WaveIsFirstLane())
            InterlockedMax(weightedTileEnd, tile + 1);
        GroupMemoryBarrierWithGroupSync();
        // A tile whose whole window has mask one keeps the unmasked rounding and mass one.
        const bool weighted = moments && masked(p) && weightedTileEnd == tile + 1;
        for (uint j = lane; j < 416; j += 256) {
            uint center = (j / 16) * 26 + (j % 16) + 5;
            float sums[7] = {0, 0, 0, 0, 0, 0, 0};
            if (weighted)
                sums[6] = fusedInput[3][center] * weights[0];
            [loop] for (uint d = 1; d <= 5; ++d) {
                if (moments) {
                    float al = fusedInput[0][center - d], ar = fusedInput[0][center + d];
                    float rl = fusedInput[1][center - d], rr = fusedInput[1][center + d];
                    float tl = fusedInput[2][center - d], tr = fusedInput[2][center + d];
                    if (weighted) {
                        float ml = fusedInput[3][center - d], mr = fusedInput[3][center + d];
                        float wl = ml * weights[d], wr = mr * weights[d];
                        sums[6] = fma(ml + mr, weights[d], sums[6]);
                        sums[0] = accumulateWeightedPair(sums[0], al, ar, wl, wr);
                        sums[1] = accumulateWeightedPair(sums[1], rl, rr, wl, wr);
                        sums[2] = accumulateWeightedPair(sums[2], rl * rl, rr * rr, wl, wr);
                        sums[3] = accumulateWeightedPair(sums[3], tl, tr, wl, wr);
                        sums[4] = accumulateWeightedPair(sums[4], tl * tl, tr * tr, wl, wr);
                        sums[5] = accumulateWeightedPair(sums[5], rl * tl, rr * tr, wl, wr);
                    } else {
                        sums[0] = fma(al + ar, weights[d], sums[0]);
                        sums[1] = fma(rl + rr, weights[d], sums[1]);
                        sums[2] = fma(fma(rl, rl, rr * rr), weights[d], sums[2]);
                        sums[3] = fma(tl + tr, weights[d], sums[3]);
                        sums[4] = fma(fma(tl, tl, tr * tr), weights[d], sums[4]);
                        sums[5] = fma(fma(rl, tl, rr * tr), weights[d], sums[5]);
                    }
                } else {
                    [loop] for (uint f = 0; f < 4; ++f) if (f < fields) sums[f] = fma(fusedInput[f][center - d] + fusedInput[f][center + d], weights[d], sums[f]);
                }
            }
            if (moments) {
                float a = fusedInput[0][center], r = fusedInput[1][center], t = fusedInput[2][center];
                float mc = masked(p) ? fusedInput[3][center] : 1.0f;
                // CUDA weights the center coefficient before multiplying a moment.
                float wc = weighted ? mc * weights[0] : weights[0];
                float v[7] = {a, r, r * r, t, t * t, r * t, mc};
                [loop] for (uint f = 0; f < 7; ++f)
                    if (f < fields)
                        verticalTile[f][j] = f == 6 ? sums[f] : fma(v[f], wc, sums[f]);
            } else {
                [loop] for (uint f = 0; f < 4; ++f) if (f < fields) verticalTile[f][j] = fma(fusedInput[f][center], weights[0], sums[f]);
            }
        }
        GroupMemoryBarrierWithGroupSync();
        int x = ox + int(lane % 16), y = oy + int(lane / 16);
        if (x < int(p.width) && y < int(p.height)) {
            uint center = lane + 80;
            float m[7] = {0, 0, 0, 0, 0, 0, 0};
            [loop] for (uint f = 0; f < 7; ++f) {
                if (f >= fields)
                    continue;
                if (moments && !decoupled(p) && f == 1) {
                    m[1] = m[0];
                    continue;
                }
                float sum = weighted && f == 6
                                ? verticalTile[f][center] * weights[0] : 0;
                [loop] for (uint d = 1; d <= 5; ++d) sum = fma(verticalTile[f][center - d * 16] + verticalTile[f][center + d * 16], weights[d], sum);
                m[f] = weighted && f == 6
                           ? sum : fma(verticalTile[f][center], weights[0], sum);
            }
            uint i = base + uint(y) * p.width + uint(x);
            if (moments) {
                float invMass = weighted ? (m[6] >= 1.0e-30f ? 1.0f / m[6] : 0.0f) : 1.0f;
                float centerWeight = masked(p) ? fusedInput[3][lane / 16 * 26 + lane % 16 + 135] : 1.0f;
                finishMoments(p, i, m, weighted, invMass, centerWeight * invMass);
            }
            else {
                float dm[4];
                [loop] for (uint f = 0; f < 4; ++f) dm[f] = m[f];
                finishGradient(p, i, x, y, dm);
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint lane : SV_GroupIndex) {
    uint3 groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
    Params p = ((Ptr<Params, Access.Read>)push.parameters)[0];
    fusedFilter(p, gid.x, lane, groups.x);
}
