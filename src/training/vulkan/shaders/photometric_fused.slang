/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
[SpecializationConstant] const uint operation = 0;
struct Params {
    uint64_t corrected, raw, target, mask, horizontal, partials;
    uint64_t ssim, cs, grad, gradRaw, losses, normalizer;
    uint count, channels, height, width, batch, stage, path, targetByte;
    uint maskByte, validPadding, partialStride, reserved;
    float weight;
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
        return maskAt(p, uint(y) * p.width + uint(x)) / (readFloat(p.normalizer, 0) * float(p.channels) + 1e-8f);
    bool crop = (operation >> 3u) != 0 && p.validPadding != 0 && p.width > 10 && p.height > 10;
    if (crop && (x < 5 || y < 5 || x >= int(p.width) - 5 || y >= int(p.height) - 5))
        return 0;
    return 1.0f / float(crop ? p.batch * p.channels * (p.width - 10) * (p.height - 10) : p.count);
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
void finishMoments(Params p, uint i, float m[6]) {
    float a = m[0], r = m[1], t = m[3];
    float A = fma(a, a, t * t) + 0.01f * 0.01f;
    float B = fma(-r, r, m[2]) + fma(-t, t, m[4]) + 0.03f * 0.03f;
    float C = fma(2 * a, t, 0.01f * 0.01f), D = fma(2.0f, fma(-r, t, m[5]), 0.03f * 0.03f);
    float invAB = 1.0f / (A * B), invAAB = 1.0f / (A * A * B), invABB = 1.0f / (A * B * B);
    float similarity;
    if (decoupled(p)) {
        float lum = C / A, cs = D / B;
        similarity = lum * cs;
        storePartial(p, i, 0, cs * (2 * (t * A - a * C) / (A * A)));
        float dv = p.weight * (-(lum * D) / (B * B));
        float dc = p.weight * ((2 * lum) / B);
        storePartial(p, i, 1, dv * (-2 * r) + dc * (-t));
        storePartial(p, i, 2, dv);
        storePartial(p, i, 3, dc);
    } else {
        similarity = (C * D) * invAB;
        storePartial(p, i, 0, (t * 2 * D) * invAB - (t * 2 * C) * invAB - (a * 2 * C * D) * invAAB + (a * 2 * C * D) * invABB);
        storePartial(p, i, 1, (-C * D) * invABB);
        storePartial(p, i, 2, (2 * C) * invAB);
    }
    writeFloat(p.ssim, i, similarity);
    writeFloat(p.cs, i, D / B);

    {
        uint pixels = p.width * p.height;
        float a = readFloat(p.corrected, i), t = targetAt(p, i);
        float loss = (1 - p.weight) * abs(a - t) + p.weight * (1 - similarity);
        writeFloat(p.losses, i, loss * maskAt(p, i % pixels));
    }
}
void finishGradient(Params p, uint i, int x, int y, float m[4]) {
    float a = readFloat(p.corrected, i), t = targetAt(p, i);
    float l1 = (1 - p.weight) * sign(a - t) * chain(p, x, y);
    writeFloat(p.grad, i, l1 + (decoupled(p) ? m[0] : fma(t, m[2], fma(2 * a, m[1], m[0]))));
    if (decoupled(p))
        writeFloat(p.gradRaw, i, fma(t, m[3], fma(2 * readFloat(p.raw, i), m[2], m[1])));
}
// Both separable passes stay in one workgroup. Preserve the original FMA
// order and half-rounded partials, including zero padding at image edges.
// Keep the existing device workspace sizes to preserve allocation placement.
groupshared float verticalTile[6][576];
groupshared float fusedInput[4][676];
void fusedFilter(Params p, uint group, uint lane, uint groupCount) {
    uint columns = (p.width + 15) / 16, rows = (p.height + 15) / 16;
    uint tiles = columns * rows * p.batch * p.channels, pixels = p.width * p.height;
    static const bool moments = PHOTO_MOMENTS != 0;
    uint fields = moments ? 6 : decoupled(p) ? 4
                                             : 3;
    for (uint tile = group; tile < tiles; tile += groupCount) {
        uint base = (tile / (columns * rows)) * pixels;
        int ox = int((tile % columns) * 16), oy = int(((tile / columns) % rows) * 16);
        for (uint j = lane; j < 676; j += 256) {
            int x = ox + int(j % 26) - 5, y = oy + int(j / 26) - 5;
            if (moments) {
                fusedInput[0][j] = moment(p, base, x, y, 0);
                fusedInput[1][j] = decoupled(p) ? moment(p, base, x, y, 1) : fusedInput[0][j];
                fusedInput[2][j] = moment(p, base, x, y, 3);
            } else {
                [loop] for (uint f = 0; f < 4; ++f) if (f < fields) fusedInput[f][j] = partial(p, base, x, y, f);
            }
        }
        GroupMemoryBarrierWithGroupSync();
        for (uint j = lane; j < 416; j += 256) {
            uint center = (j / 16) * 26 + (j % 16) + 5;
            float sums[6] = {0, 0, 0, 0, 0, 0};
            [loop] for (uint d = 1; d <= 5; ++d) {
                if (moments) {
                    float al = fusedInput[0][center - d], ar = fusedInput[0][center + d];
                    float rl = fusedInput[1][center - d], rr = fusedInput[1][center + d];
                    float tl = fusedInput[2][center - d], tr = fusedInput[2][center + d];
                    sums[0] = fma(al + ar, weights[d], sums[0]);
                    sums[1] = fma(rl + rr, weights[d], sums[1]);
                    sums[2] = fma(fma(rl, rl, rr * rr), weights[d], sums[2]);
                    sums[3] = fma(tl + tr, weights[d], sums[3]);
                    sums[4] = fma(fma(tl, tl, tr * tr), weights[d], sums[4]);
                    sums[5] = fma(fma(rl, tl, rr * tr), weights[d], sums[5]);
                } else {
                    [loop] for (uint f = 0; f < 4; ++f) if (f < fields) sums[f] = fma(fusedInput[f][center - d] + fusedInput[f][center + d], weights[d], sums[f]);
                }
            }
            if (moments) {
                float a = fusedInput[0][center], r = fusedInput[1][center], t = fusedInput[2][center];
                float v[6] = {a, r, r * r, t, t * t, r * t};
                [loop] for (uint f = 0; f < 6; ++f) verticalTile[f][j] = fma(v[f], weights[0], sums[f]);
            } else {
                [loop] for (uint f = 0; f < 4; ++f) if (f < fields) verticalTile[f][j] = fma(fusedInput[f][center], weights[0], sums[f]);
            }
        }
        GroupMemoryBarrierWithGroupSync();
        int x = ox + int(lane % 16), y = oy + int(lane / 16);
        if (x < int(p.width) && y < int(p.height)) {
            uint center = lane + 80;
            float m[6] = {0, 0, 0, 0, 0, 0};
            [loop] for (uint f = 0; f < 6; ++f) {
                if (f >= fields)
                    continue;
                if (moments && !decoupled(p) && f == 1) {
                    m[1] = m[0];
                    continue;
                }
                float sum = 0;
                [loop] for (uint d = 1; d <= 5; ++d) sum = fma(verticalTile[f][center - d * 16] + verticalTile[f][center + d * 16], weights[d], sum);
                m[f] = fma(verticalTile[f][center], weights[0], sum);
            }
            uint i = base + uint(y) * p.width + uint(x);
            if (moments)
                finishMoments(p, i, m);
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
