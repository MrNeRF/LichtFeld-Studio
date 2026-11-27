// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
// MRNF's blur-tolerant SSIM (ssim.cu blurTolerantForwardCUDA / blurTolerantBackwardCUDA), for every backend that runs
// Slang programs. G is a normalised Gaussian blur over the kept pixels (weight above zero): G v = n * T(keep v),
// n = 1 / T(keep), T the separable taps of the photo's blur sigma, radius R = LFS_BLUR_RADIUS (one program per radius).
// The forward scores each channel and window with two decoupled SSIM branches from the same photo statistics, against
// the render and against G of the render, and takes their soft maximum (temperature 50); a flat photo window keeps the
// sharp branch alone. Three passes over 16x16 tiles and a reduction:
//   blur_forward:  G inside the tile, both branches' partials as fp16, the loss summed per group;
//   blur_reduce:   the loss over the groups and, masked, the loss denominator;
//   blur_branch:   the blurred branch's backward at G raw, as fp16 (v, u);
//   blur_gradient: the sharp backward plus the adjoint blur T(v), T(u).
// The *_masked entries weight the windows and the loss by a per-pixel mask. Byte photos and masks come through the
// *_u8 pointers and an absent input is a null pointer. fp16 values are stored as packed uint, so no backend needs
// 16-bit storage.
#ifndef LFS_BLUR_RADIUS
#define LFS_BLUR_RADIUS 5
#endif

struct Parameters {
    Ptr<float, Access.Read> corrected;
    Ptr<float, Access.Read> raw; // the corrected image again when not decoupled
    Ptr<float, Access.Read> photo;
    Ptr<uint8_t, Access.Read> photo_u8;
    Ptr<float, Access.Read> mask;
    Ptr<uint8_t, Access.Read> mask_u8;
    Ptr<float, Access.ReadWrite> ssim_map; // [N,1,H,W]: the sharp SSIM
    Ptr<float, Access.ReadWrite> cs_map;   // [N,1,H,W]: the blend's contrast-structure
    Ptr<uint2, Access.ReadWrite> sharp;    // [N,C,H,W] x 4 fp16: raw mean, raw sigma^2, raw sigma12, corrected mean partials
    Ptr<uint2, Access.ReadWrite> blurred;  // the blurred branch's partials in the same order
    Ptr<uint, Access.ReadWrite> vu;        // [N,C,H,W] x 2 fp16: written by the branch pass, read by the gradient pass
    Ptr<float, Access.ReadWrite> partials; // per group loss sums, then mask sums at kMaxGroups
    Ptr<float, Access.ReadWrite> loss;
    Ptr<float, Access.ReadWrite> mask_sum; // masked: the loss denominator, formed by blur_reduce
    Ptr<float, Access.ReadWrite> grad;     // the corrected image's, or both images' sum without grad_raw
    Ptr<float, Access.ReadWrite> grad_raw; // decoupled only
    int width, height, channels, batches;
    uint tiles_x, tiles_y, groups, count; // count: blur_reduce's partial sums
    float ssim_weight;
    float chain;       // unmasked: 1 / the cropped element count
    float divisor;     // unmasked: the cropped element count
    float denominator; // masked: the caller's loss denominator, or 0 for the mask sum
    float tap0, tap1, tap2, tap3, tap4, tap5; // centre-first normalised taps, scalars for the tensor binding ABI
};
[[vk::push_constant]] ConstantBuffer<Parameters> p;

static const int R = LFS_BLUR_RADIUS;
static const int kTile = 16;
static const int kHalo = 5;
static const int kSpan = kTile + 2 * kHalo;
static const uint kThreads = kTile * kTile;
static const uint kSpanCells = kSpan * kSpan;
static const uint kConvCells = kSpan * kTile;
static const uint kMaxGroups = 1024;
// The blur's reach around the SSIM span, and the adjoint blur's around a tile; pitches are odd row lengths.
static const int kReach = kSpan + 2 * R;
static const int kReachPitch = kReach + 1;
static const int kFoot = kTile + 2 * R;
static const int kFootPitch = kFoot + 1;
static const int kDataPitch = kSpan + 1;
static const float kC1 = 0.01f * 0.01f;
static const float kC2 = 0.03f * 0.03f;
static const float kMaskEpsilon = 1e-8f;
static const float kTemperature = 50.0f;
static const float kFlatVariance = 0.0005f;
static const float kLn2 = 0.69314718055994531f;
static const float kGauss[11] = {0.001028380123898387f, 0.0075987582094967365f, 0.036000773310661316f,
                                 0.10936068743467331f, 0.21300552785396576f, 0.26601171493530273f,
                                 0.21300552785396576f, 0.10936068743467331f, 0.036000773310661316f,
                                 0.0075987582094967365f, 0.001028380123898387f};
// The adjoint blur's footprint marks a dropped pixel -0; a kept -0 is stored as +0, the same value in every sum.
static const uint kDropped = 0x80000000u;

// The forward's tile storage.
groupshared float2 src[kReach * kReachPitch]; // (corrected, raw) or (keep, 0) over the blur's reach
groupshared float photoTile[kSpanCells];
groupshared float weights[kSpanCells];
groupshared float support[kSpanCells];
groupshared float rows[5 * kConvCells]; // five planes of floats, or the blur's float2 rows interleaved
groupshared float scratch[32];          // one sum per wave of at least eight lanes
groupshared uint flags[3];
// The backward's.
groupshared uint2 data[kSpan * kDataPitch]; // fp16 x 4 partials over the span
groupshared float4 conv[kConvCells];
groupshared uint source[kFoot * kFootPitch]; // gradient: fp16 (v, u); branch: keep raw, kDropped where not kept
groupshared float2 footRows[kFoot * kTile];
groupshared uint weighted;

float blurTap(const int k) {
    switch (k) {
    case 0: return p.tap0;
    case 1: return p.tap1;
    case 2: return p.tap2;
    case 3: return p.tap3;
    case 4: return p.tap4;
    default: return p.tap5;
    }
}

uint2 pack4(const float4 v) {
    return uint2(f32tof16(v.x) | (f32tof16(v.y) << 16), f32tof16(v.z) | (f32tof16(v.w) << 16));
}
float4 unpack4(const uint2 v) { return float4(f16tof32(v.x), f16tof32(v.x >> 16), f16tof32(v.y), f16tof32(v.y >> 16)); }
uint pack2(const float2 v) { return f32tof16(v.x) | (f32tof16(v.y) << 16); }
float2 unpack2(const uint v) { return float2(f16tof32(v), f16tof32(v >> 16)); }

bool present(Ptr<uint8_t, Access.Read> x) { return (uint64_t)x != 0; }
float photoAt(const uint i) { return present(p.photo_u8) ? float(p.photo_u8[i]) * (1.0f / 255.0f) : p.photo[i]; }
float maskAt(const uint i) { return present(p.mask_u8) ? (p.mask_u8[i] != 0 ? 1.0f : 0.0f) : p.mask[i]; }
// Four consecutive values from index.
float4 photo4(const uint i) {
    if (present(p.photo_u8))
        return float4(float(p.photo_u8[i]), float(p.photo_u8[i + 1]), float(p.photo_u8[i + 2]), float(p.photo_u8[i + 3])) *
               (1.0f / 255.0f);
    return float4(p.photo[i], p.photo[i + 1], p.photo[i + 2], p.photo[i + 3]);
}
float4 mask4(const uint i) {
    if (present(p.mask_u8))
        return float4(p.mask_u8[i] != 0 ? 1.0f : 0.0f, p.mask_u8[i + 1] != 0 ? 1.0f : 0.0f,
                      p.mask_u8[i + 2] != 0 ? 1.0f : 0.0f, p.mask_u8[i + 3] != 0 ? 1.0f : 0.0f);
    return float4(p.mask[i], p.mask[i + 1], p.mask[i + 2], p.mask[i + 3]);
}

// The loss weight a pixel gives the blur: its mask (the masked loss), or 1 inside the image.
float blurWeight<let MASKED : int>(const int y, const int x) {
    if (x < 0 || y < 0 || x >= p.width || y >= p.height)
        return 0.0f;
    return MASKED != 0 ? maskAt(uint(y) * uint(p.width) + uint(x)) : 1.0f;
}

// Window centres with partials: masked partials carry their own window weights; unmasked, the valid padding counts
// only centres 5 pixels or more inside the border.
bool validCentre<let MASKED : int>(const int y, const int x) {
    const int H = p.height, W = p.width;
    return x >= 0 && x < W && y >= 0 && y < H &&
           (MASKED != 0 || ((H <= 10 || (y >= 5 && y < H - 5)) && (W <= 10 || (x >= 5 && x < W - 5))));
}

// Sum over the group; the result is valid in thread 0.
float groupSum(float value, const uint rank) {
    value = WaveActiveSum(value);
    const uint lanes = WaveGetLaneCount();
    if (WaveIsFirstLane())
        scratch[rank / lanes] = value;
    GroupMemoryBarrierWithGroupSync();
    float total = 0.0f;
    if (rank == 0) {
        for (uint i = 0; i < kThreads / lanes; ++i)
            total += scratch[i];
    }
    GroupMemoryBarrierWithGroupSync();
    return total;
}

// One channel's loads: keep corrected and keep raw over the blur's reach into src, the photo over the SSIM span into
// photoTile. Four columns per item; items that leave the image or the row load element by element.
void loadChannel<let MASKED : int>(const uint base, const int y0, const int x0, const uint rank) {
    const int H = p.height, W = p.width;
    const int reachQuads = (kReach + 3) / 4;
    for (uint i = rank; i < uint(kReach * reachQuads); i += kThreads) {
        const int yy = int(i) / reachQuads, xx = 4 * (int(i) % reachQuads);
        const int gy = y0 + yy - kHalo - R, gx = x0 + xx - kHalo - R;
        const int out = yy * kReachPitch + xx;
        if (xx + 4 <= kReach && gy >= 0 && gy < H && gx >= 0 && gx + 4 <= W) {
            const uint g = base + uint(gy) * uint(W) + uint(gx);
            const float4 c = float4(p.corrected[g], p.corrected[g + 1], p.corrected[g + 2], p.corrected[g + 3]);
            const float4 r = float4(p.raw[g], p.raw[g + 1], p.raw[g + 2], p.raw[g + 3]);
            const bool4 keep = MASKED != 0 ? mask4(uint(gy) * uint(W) + uint(gx)) > 0.0f : bool4(true);
            src[out] = keep.x ? float2(c.x, r.x) : float2(0.0f);
            src[out + 1] = keep.y ? float2(c.y, r.y) : float2(0.0f);
            src[out + 2] = keep.z ? float2(c.z, r.z) : float2(0.0f);
            src[out + 3] = keep.w ? float2(c.w, r.w) : float2(0.0f);
        } else {
            for (int k = 0; k < 4 && xx + k < kReach; ++k) {
                float2 value = float2(0.0f);
                if (blurWeight<MASKED>(gy, gx + k) > 0.0f) {
                    const uint g = base + uint(gy) * uint(W) + uint(gx + k);
                    value = float2(p.corrected[g], p.raw[g]);
                }
                src[out + k] = value;
            }
        }
    }
    const int spanQuads = (kSpan + 3) / 4;
    for (uint i = rank; i < uint(kSpan * spanQuads); i += kThreads) {
        const int yy = int(i) / spanQuads, xx = 4 * (int(i) % spanQuads);
        const int gy = y0 + yy - kHalo, gx = x0 + xx - kHalo;
        const int out = yy * kSpan + xx;
        if (xx + 4 <= kSpan && gy >= 0 && gy < H && gx >= 0 && gx + 4 <= W) {
            const float4 y = photo4(base + uint(gy) * uint(W) + uint(gx));
            photoTile[out] = y.x;
            photoTile[out + 1] = y.y;
            photoTile[out + 2] = y.z;
            photoTile[out + 3] = y.w;
        } else {
            for (int k = 0; k < 4 && xx + k < kSpan; ++k) {
                const bool in = gx + k >= 0 && gx + k < W && gy >= 0 && gy < H;
                photoTile[out + k] = in ? photoAt(base + uint(gy) * uint(W) + uint(gx + k)) : 0.0f;
            }
        }
    }
}

// One decoupled SSIM branch from window means: returns S, writes cs and the partials in the order of the stores.
float branchScore(const float mc, const float mr, const float mrr, const float my, const float myy, const float mry,
                  const float ssimWeight, out float cs, out float4 partial) {
    const float A = mc * mc + my * my + kC1;
    const float Cn = 2.0f * mc * my + kC1;
    const float lum = Cn / A;
    const float B = (mrr - mr * mr) + (myy - my * my) + kC2;
    const float D = 2.0f * (mry - mr * my) + kC2;
    cs = D / B;
    const float ds1 = ssimWeight * (-(lum * D) / (B * B));
    const float ds12 = ssimWeight * ((2.0f * lum) / B);
    partial = float4(ds1 * (-2.0f * mr) + ds12 * (-my), ds1, ds12, cs * (2.0f * (my * A - mc * Cn) / (A * A)));
    return lum * cs;
}

// Five window moments at this thread's pixel. Horizontal sums of the SSIM span's 26 rows go to rows (five planes of
// 26x16), then the vertical sums at the pixel, times invWindow. QUANTITY selects the samples: 0 the sharp moments
// (corrected, raw, raw^2, raw photo, photo), 1 the blurred ones (photo^2, G corrected, G raw, (G raw)^2, G raw photo);
// src holds the values at reach position (span + R).
void moments<let QUANTITY : int, let MASKED : int>(const uint2 t, const uint rank, const float invWindow, out float m[5]) {
    for (uint o = rank; o < kConvCells; o += kThreads) {
        const int r = int(o) / kTile, q0 = int(o) % kTile;
        float a[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        [ForceUnroll]
        for (int d = -kHalo; d <= kHalo; ++d) {
            const int q = q0 + kHalo + d;
            const float2 x = src[(r + R) * kReachPitch + q + R];
            const float py = photoTile[r * kSpan + q];
            const float w = MASKED != 0 ? weights[r * kSpan + q] : 1.0f;
            const float tap = kGauss[kHalo + d];
            float v[5];
            if (QUANTITY == 0) {
                v[0] = w * x.x;
                v[1] = w * x.y;
                v[2] = w * (x.y * x.y);
                v[3] = w * (x.y * py);
                v[4] = w * py;
            } else {
                v[0] = w * (py * py);
                v[1] = w * x.x;
                v[2] = w * x.y;
                v[3] = w * (x.y * x.y);
                v[4] = w * (x.y * py);
            }
            [ForceUnroll]
            for (int k = 0; k < 5; ++k)
                a[k] += tap * v[k];
        }
        [ForceUnroll]
        for (int k = 0; k < 5; ++k)
            rows[uint(k) * kConvCells + o] = a[k];
    }
    GroupMemoryBarrierWithGroupSync();
    float a[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    [ForceUnroll]
    for (int d = -kHalo; d <= kHalo; ++d) {
        const uint cell = uint(int(t.y) + kHalo + d) * uint(kTile) + t.x;
        const float tap = kGauss[kHalo + d];
        [ForceUnroll]
        for (int k = 0; k < 5; ++k)
            a[k] += tap * rows[uint(k) * kConvCells + cell];
    }
    [ForceUnroll]
    for (int k = 0; k < 5; ++k)
        m[k] = a[k] * invWindow;
    GroupMemoryBarrierWithGroupSync();
}

// The blur T of src's two planes over the span: horizontal sums into rows (reach x 26 float2), then vertical ones,
// times the support n, written back over src at the span's reach position; allKept: n is 1. With SUPPORT the planes
// are (keep, 0) and the result 1 / T(keep) goes to support.
void blurSpan<let SUPPORT : int>(const uint rank, const int y0, const int x0, const bool allKept) {
    for (uint o = rank; o < uint(kReach * kSpan); o += kThreads) {
        const int r = int(o) / kSpan, q = int(o) % kSpan;
        float2 a = float2(0.0f);
        [ForceUnroll]
        for (int d = -R; d <= R; ++d)
            a += blurTap(abs(d)) * src[r * kReachPitch + q + R + d];
        rows[2 * o] = a.x;
        rows[2 * o + 1] = a.y;
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint o = rank; o < kSpanCells; o += kThreads) {
        const int r = int(o) / kSpan, q = int(o) % kSpan;
        float2 a = float2(0.0f);
        [ForceUnroll]
        for (int d = -R; d <= R; ++d) {
            const uint cell = uint((r + R + d) * kSpan + q);
            a += blurTap(abs(d)) * float2(rows[2 * cell], rows[2 * cell + 1]);
        }
        if (SUPPORT != 0) {
            const int gy = y0 + r - kHalo, gx = x0 + q - kHalo;
            const bool inside = gx >= 0 && gx < p.width && gy >= 0 && gy < p.height;
            support[o] = inside && a.x > 1.0e-30f ? 1.0f / a.x : 0.0f;
        } else {
            const float n = allKept ? 1.0f : support[o];
            src[(r + R) * kReachPitch + q + R] = float2(a.x * n, a.y * n);
        }
    }
    GroupMemoryBarrierWithGroupSync();
}

// One 16x16 output tile of the forward. Sums the tile's loss terms into lossSum (masked: times the pixel weight, and
// the weights into maskSum).
void forwardTile<let MASKED : int>(const uint tile, const uint2 t, const uint rank, inout float lossSum,
                                   inout float maskSum) {
    const int H = p.height, W = p.width, C = p.channels;
    const uint perImage = p.tiles_x * p.tiles_y;
    const uint b = tile / perImage;
    const uint within = tile % perImage;
    const int y0 = int(within / p.tiles_x) * kTile;
    const int x0 = int(within % p.tiles_x) * kTile;
    const int px = x0 + int(t.x), py = y0 + int(t.y);
    const bool inside = px < W && py < H;
    const uint plane = uint(H) * uint(W);
    const uint pixel = inside ? uint(py) * uint(W) + uint(px) : 0;
    const uint centreCell = (t.y + kHalo) * kSpan + t.x + kHalo;

    // flags: 0 a window weight other than one, 1 a pixel of the reach not kept, 2 a weighted pixel in the tile.
    GroupMemoryBarrierWithGroupSync();
    if (rank < 3)
        flags[rank] = 0;
    GroupMemoryBarrierWithGroupSync();
    bool partialSupport = false, dropped = false;
    if (MASKED != 0) {
        // A common scale cancels from normalised moments and centre/mass (load_mask_window).
        const int anchorX = min(x0 + kTile / 2, W - 1), anchorY = min(y0 + kTile / 2, H - 1);
        const float scale = 1.0f / max(1.0f, maskAt(uint(anchorY) * uint(W) + uint(anchorX)));
        const int spanQuads = (kSpan + 3) / 4;
        if (rank < uint(kSpan * spanQuads)) {
            const int yy = int(rank) / spanQuads, xx = 4 * (int(rank) % spanQuads);
            const int gy = y0 + yy - kHalo, gx = x0 + xx - kHalo;
            const int out = yy * kSpan + xx;
            if (xx + 4 <= kSpan && gy >= 0 && gy < H && gx >= 0 && gx + 4 <= W) {
                const float4 value = mask4(uint(gy) * uint(W) + uint(gx)) * scale;
                weights[out] = value.x;
                weights[out + 1] = value.y;
                weights[out + 2] = value.z;
                weights[out + 3] = value.w;
                partialSupport = partialSupport || any(value != 1.0f);
            } else {
                for (int k = 0; k < 4 && xx + k < kSpan; ++k) {
                    const bool in = gx + k >= 0 && gx + k < W && gy >= 0 && gy < H;
                    const float value = (in ? maskAt(uint(gy) * uint(W) + uint(gx + k)) : 1.0f) * scale;
                    weights[out + k] = value;
                    partialSupport = partialSupport || value != 1.0f;
                }
            }
        }
        if (WaveActiveAnyTrue(inside && maskAt(pixel) != 0.0f) && WaveIsFirstLane())
            InterlockedOr(flags[2], 1u);
        if (WaveActiveAnyTrue(partialSupport) && WaveIsFirstLane())
            InterlockedOr(flags[0], 1u);
    }
    const int reachQuads = (kReach + 3) / 4;
    for (uint i = rank; i < uint(kReach * reachQuads); i += kThreads) {
        const int yy = int(i) / reachQuads, xx = 4 * (int(i) % reachQuads);
        const int gy = y0 + yy - kHalo - R, gx = x0 + xx - kHalo - R;
        const int out = yy * kReachPitch + xx;
        if (xx + 4 <= kReach && gy >= 0 && gy < H && gx >= 0 && gx + 4 <= W) {
            const bool4 keep = MASKED != 0 ? mask4(uint(gy) * uint(W) + uint(gx)) > 0.0f : bool4(true);
            src[out] = float2(keep.x ? 1.0f : 0.0f, 0.0f);
            src[out + 1] = float2(keep.y ? 1.0f : 0.0f, 0.0f);
            src[out + 2] = float2(keep.z ? 1.0f : 0.0f, 0.0f);
            src[out + 3] = float2(keep.w ? 1.0f : 0.0f, 0.0f);
            dropped = dropped || !all(keep);
        } else {
            for (int k = 0; k < 4 && xx + k < kReach; ++k) {
                const bool keep = blurWeight<MASKED>(gy, gx + k) > 0.0f;
                src[out + k] = float2(keep ? 1.0f : 0.0f, 0.0f);
                dropped = dropped || !keep;
            }
        }
    }
    if (WaveActiveAnyTrue(dropped) && WaveIsFirstLane())
        InterlockedOr(flags[1], 1u);
    GroupMemoryBarrierWithGroupSync();
    const bool weightedWindow = MASKED != 0 && flags[0] != 0;
    const bool allKept = flags[1] == 0;
    if (MASKED != 0 && flags[2] == 0) {
        // No loss term in the tile: zero partials and unit maps, as the CUDA forward.
        if (!inside)
            return;
        for (int c = 0; c < C; ++c) {
            const uint index = (b * uint(C) + uint(c)) * plane + pixel;
            p.sharp[index] = uint2(0);
            p.blurred[index] = uint2(0);
        }
        p.ssim_map[b * plane + pixel] = 1.0f;
        p.cs_map[b * plane + pixel] = 1.0f;
        return;
    }

    float invWindow = 1.0f, centre = 1.0f;
    if (MASKED != 0) {
        centre = weights[centreCell];
        if (weightedWindow) {
            // The window mass (load_mask_window's order: the centre tap, then the pairs outwards).
            for (uint o = rank; o < kConvCells; o += kThreads) {
                const uint r = o / uint(kTile), x = o % uint(kTile) + uint(kHalo);
                float sum = weights[r * kSpan + x] * kGauss[kHalo];
                [ForceUnroll]
                for (int d = 1; d <= kHalo; ++d)
                    sum += (weights[r * kSpan + x - uint(d)] + weights[r * kSpan + x + uint(d)]) * kGauss[kHalo - d];
                rows[o] = sum;
            }
            GroupMemoryBarrierWithGroupSync();
            const uint y = t.y + kHalo;
            float sum = rows[y * kTile + t.x] * kGauss[kHalo];
            [ForceUnroll]
            for (int d = 1; d <= kHalo; ++d)
                sum += (rows[(y - uint(d)) * kTile + t.x] + rows[(y + uint(d)) * kTile + t.x]) * kGauss[kHalo - d];
            invWindow = sum >= 1.0e-30f ? 1.0f / sum : 0.0f;
            GroupMemoryBarrierWithGroupSync();
        }
    }
    if (!allKept)
        blurSpan<1>(rank, y0, x0, false);

    float ssimSum = 0.0f, blendSum = 0.0f, csSum = 0.0f, l1Sum = 0.0f;
    for (int c = 0; c < C; ++c) {
        const uint base = (b * uint(C) + uint(c)) * plane;
        GroupMemoryBarrierWithGroupSync();
        loadChannel<MASKED>(base, y0, x0, rank);
        GroupMemoryBarrierWithGroupSync();
        float ms[5], mb[5];
        moments<0, MASKED>(t, rank, invWindow, ms);
        blurSpan<0>(rank, y0, x0, allKept);
        moments<1, MASKED>(t, rank, invWindow, mb);
        if (!inside)
            continue;
        float cs0, cs1;
        float4 p0, p1;
        const float s0 = branchScore(ms[0], ms[1], ms[2], ms[4], mb[0], ms[3], p.ssim_weight, cs0, p0);
        const float s1 = branchScore(mb[1], mb[2], mb[3], ms[4], mb[0], mb[4], p.ssim_weight, cs1, p1);
        const float d = kTemperature * (s0 - s1), e = exp(-abs(d));
        const float high = 1.0f / (1.0f + e), low = e * high; // the soft-max weights of the larger, smaller branch
        const bool flat = mb[0] - ms[4] * ms[4] < kFlatVariance;
        const float w0 = flat ? 1.0f : (d >= 0.0f ? high : low), w1 = flat ? 0.0f : (d >= 0.0f ? low : high);
        ssimSum += s0;
        blendSum += flat ? s0 : max(s0, s1) + (log(1.0f + e) - kLn2) / kTemperature;
        csSum += w0 * cs0 + w1 * cs1;
        // Partial scaling as the masked decoupled kernels: the masked raw mean partial is stored / 16.
        const float dw = centre * invWindow;
        const float rawMeanScale = MASKED != 0 ? dw * (1.0f / 16.0f) : dw;
        const uint index = base + pixel;
        p.sharp[index] = pack4(float4(p0.x * (w0 * rawMeanScale), p0.y * (w0 * dw), p0.z * (w0 * dw), p0.w * (w0 * dw)));
        p.blurred[index] = pack4(float4(p1.x * (w1 * rawMeanScale), p1.y * (w1 * dw), p1.z * (w1 * dw),
                                        p1.w * (w1 * dw * p.ssim_weight)));
        l1Sum += abs(p.corrected[index] - photoTile[centreCell]);
    }
    if (!inside)
        return;
    const float invC = 1.0f / float(C);
    const bool kept = MASKED == 0 || centre > 0.0f;
    p.ssim_map[b * plane + pixel] = kept ? ssimSum * invC : 1.0f;
    p.cs_map[b * plane + pixel] = kept ? csSum * invC : 1.0f;
    const float blend = kept ? blendSum * invC : 1.0f;
    const float perPixel = (1.0f - p.ssim_weight) * l1Sum + float(C) * p.ssim_weight * (1.0f - blend);
    if (MASKED != 0) {
        const float m = maskAt(pixel);
        lossSum += perPixel * m;
        if (b == 0)
            maskSum += m;
        return;
    }
    // The valid padding crops each axis on its own.
    if ((H > 10 && (py < 5 || py >= H - 5)) || (W > 10 && (px < 5 || px >= W - 5)))
        return;
    lossSum += perPixel;
}

// Groups loop over tiles, so at most kMaxGroups partial sums come out.
void forwardGroup<let MASKED : int>(const uint group, const uint2 t, const uint rank) {
    float lossSum = 0.0f, maskSum = 0.0f;
    const uint tiles = p.tiles_x * p.tiles_y * uint(p.batches);
    for (uint tile = group; tile < tiles; tile += p.groups)
        forwardTile<MASKED>(tile, t, rank, lossSum, maskSum);
    lossSum = groupSum(lossSum, rank);
    if (rank == 0)
        p.partials[group] = lossSum;
    if (MASKED != 0) {
        maskSum = groupSum(maskSum, rank);
        if (rank == 0)
            p.partials[kMaxGroups + group] = maskSum;
    }
}

// One 16x16 tile per group. Per channel: the window adjoint of the partials (window centres with partials only), T of
// two planes, then per pixel
//   branch (GRADIENT 0): the blurred branch's backward at G raw = T(keep raw) / T(keep) and the photo, times
//     weight * n: v, u;
//   gradient (GRADIENT 1): the sharp backward (SSIM weight sw, L1 1 - sw) plus keep * T(v) and keep * T(u).
void backwardTile<let MASKED : int, let GRADIENT : int>(const uint2 g, const uint2 t, const uint rank) {
    const int H = p.height, W = p.width, C = p.channels;
    const uint b = g.y / p.tiles_y;
    const int y0 = int(g.y % p.tiles_y) * kTile;
    const int x0 = int(g.x) * kTile;
    const int px = x0 + int(t.x), py = y0 + int(t.y);
    const bool inside = px < W && py < H;
    const uint plane = uint(H) * uint(W);
    const uint pixel = inside ? uint(py) * uint(W) + uint(px) : 0;
    const float centre = blurWeight<MASKED>(py, px);
    const bool decoupled = (uint64_t)p.grad_raw != 0;
    if (MASKED != 0) {
        // The masked rule: no gradient crosses into a tile without a weighted pixel.
        if (rank == 0)
            weighted = 0;
        GroupMemoryBarrierWithGroupSync();
        if (WaveActiveAnyTrue(centre != 0.0f) && WaveIsFirstLane())
            InterlockedOr(weighted, 1u);
        GroupMemoryBarrierWithGroupSync();
        if (weighted == 0) {
            if (!inside)
                return;
            for (int c = 0; c < C; ++c) {
                const uint q = (b * uint(C) + uint(c)) * plane + pixel;
                if (GRADIENT == 0) {
                    p.vu[q] = 0;
                } else {
                    p.grad[q] = 0.0f;
                    if (decoupled)
                        p.grad_raw[q] = 0.0f;
                }
            }
            return;
        }
    }
    const float chain = MASKED != 0 ? 1.0f / p.mask_sum[0] : p.chain;
    for (int c = 0; c < C; ++c) {
        const uint base = (b * uint(C) + uint(c)) * plane;
        for (uint i = rank; i < kSpanCells; i += kThreads) {
            const int yy = int(i) / kSpan, xx = int(i) % kSpan;
            const int gy = y0 + yy - kHalo, gx = x0 + xx - kHalo;
            uint2 value = uint2(0);
            if (validCentre<MASKED>(gy, gx))
                value = GRADIENT != 0 ? p.sharp[base + uint(gy) * uint(W) + uint(gx)]
                                      : p.blurred[base + uint(gy) * uint(W) + uint(gx)];
            data[yy * kDataPitch + xx] = value;
        }
        for (uint i = rank; i < uint(kFoot * kFoot); i += kThreads) {
            const int yy = int(i) / kFoot, xx = int(i) % kFoot;
            const int gy = y0 + yy - R, gx = x0 + xx - R;
            const int cell = yy * kFootPitch + xx;
            if (GRADIENT != 0) {
                uint value = 0;
                if (gx >= 0 && gx < W && gy >= 0 && gy < H)
                    value = p.vu[base + uint(gy) * uint(W) + uint(gx)];
                source[cell] = value;
            } else {
                uint value = kDropped;
                if (blurWeight<MASKED>(gy, gx) > 0.0f) {
                    const uint bits = asuint(p.raw[base + uint(gy) * uint(W) + uint(gx)]);
                    value = bits == kDropped ? 0u : bits;
                }
                source[cell] = value;
            }
        }
        GroupMemoryBarrierWithGroupSync();
        for (uint o = rank; o < kConvCells; o += kThreads) {
            const int r = int(o) / kTile, q = int(o) % kTile;
            float4 a = float4(0.0f);
            [ForceUnroll]
            for (int d = -kHalo; d <= kHalo; ++d)
                a += kGauss[kHalo + d] * unpack4(data[r * kDataPitch + q + kHalo + d]);
            conv[o] = a;
        }
        for (uint o = rank; o < uint(kFoot * kTile); o += kThreads) {
            const int r = int(o) / kTile, q = int(o) % kTile;
            float2 a = float2(0.0f);
            [ForceUnroll]
            for (int d = -R; d <= R; ++d) {
                const uint v = source[r * kFootPitch + q + R + d];
                float2 value;
                if (GRADIENT != 0)
                    value = unpack2(v);
                else
                    value = v == kDropped ? float2(0.0f) : float2(asfloat(v), 1.0f);
                a += blurTap(abs(d)) * value;
            }
            footRows[o] = a;
        }
        GroupMemoryBarrierWithGroupSync();
        if (inside) {
            float4 m = float4(0.0f);
            [ForceUnroll]
            for (int d = -kHalo; d <= kHalo; ++d)
                m += kGauss[kHalo + d] * conv[uint(int(t.y) + kHalo + d) * uint(kTile) + t.x];
            float2 s = float2(0.0f);
            [ForceUnroll]
            for (int d = -R; d <= R; ++d)
                s += blurTap(abs(d)) * footRows[uint(int(t.y) + R + d) * uint(kTile) + t.x];
            const uint q = base + pixel;
            const float p2 = photoAt(q);
            const float s0 = (MASKED != 0 ? -16.0f : -1.0f) * m.x; // the masked mean partial is / 16
            if (GRADIENT != 0) {
                const float pc = p.corrected[q], pr = p.raw[q];
                const float keep = centre > 0.0f ? 1.0f : 0.0f; // the adjoint lands on kept pixels
                const float sign = pc == p2 ? 0.0f : (pc - p2 < 0.0f ? -1.0f : 1.0f);
                const float l1 = (1.0f - p.ssim_weight) * sign *
                                 (MASKED != 0 ? centre : (validCentre<0>(py, px) ? 1.0f : 0.0f));
                const float gc = chain * (centre * -p.ssim_weight * m.w + l1 + keep * s.x);
                const float gr = chain * (centre * (s0 - 2.0f * pr * m.y - p2 * m.z) + keep * s.y);
                if (decoupled) {
                    p.grad[q] = gc;
                    p.grad_raw[q] = gr;
                } else {
                    p.grad[q] = gc + gr;
                }
            } else {
                const float n = s.y > 1.0e-30f ? 1.0f / s.y : 0.0f, kn = centre * n;
                p.vu[q] = pack2(float2(-kn * m.w, kn * (s0 - 2.0f * (s.x * n) * m.y - p2 * m.z)));
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
}

[shader("compute")]
[numthreads(16, 16, 1)]
void blur_forward(uint3 group : SV_GroupID, uint3 t : SV_GroupThreadID, uint rank : SV_GroupIndex) {
    forwardGroup<0>(group.x, t.xy, rank);
}
[shader("compute")]
[numthreads(16, 16, 1)]
void blur_forward_masked(uint3 group : SV_GroupID, uint3 t : SV_GroupThreadID, uint rank : SV_GroupIndex) {
    forwardGroup<1>(group.x, t.xy, rank);
}
[shader("compute")]
[numthreads(16, 16, 1)]
void blur_branch(uint3 group : SV_GroupID, uint3 t : SV_GroupThreadID, uint rank : SV_GroupIndex) {
    backwardTile<0, 0>(group.xy, t.xy, rank);
}
[shader("compute")]
[numthreads(16, 16, 1)]
void blur_branch_masked(uint3 group : SV_GroupID, uint3 t : SV_GroupThreadID, uint rank : SV_GroupIndex) {
    backwardTile<1, 0>(group.xy, t.xy, rank);
}
[shader("compute")]
[numthreads(16, 16, 1)]
void blur_gradient(uint3 group : SV_GroupID, uint3 t : SV_GroupThreadID, uint rank : SV_GroupIndex) {
    backwardTile<0, 1>(group.xy, t.xy, rank);
}
[shader("compute")]
[numthreads(16, 16, 1)]
void blur_gradient_masked(uint3 group : SV_GroupID, uint3 t : SV_GroupThreadID, uint rank : SV_GroupIndex) {
    backwardTile<1, 1>(group.xy, t.xy, rank);
}

// The loss over the forward's partial sums. Masked: divided by the caller's denominator, or by the mask sum times
// the images' channels (ssim_reduction.cu), which also goes to mask_sum for the backward; unmasked: by the cropped
// element count.
[shader("compute")]
[numthreads(256, 1, 1)]
void blur_reduce(uint rank : SV_GroupIndex) {
    const bool masked = (uint64_t)p.mask_sum != 0;
    float sum = 0.0f, mask = 0.0f;
    for (uint i = rank; i < p.count; i += kThreads) {
        sum += p.partials[i];
        if (masked)
            mask += p.partials[kMaxGroups + i];
    }
    sum = groupSum(sum, rank);
    if (masked)
        mask = groupSum(mask, rank);
    if (rank != 0)
        return;
    if (masked) {
        const float normalized = p.denominator > 0.0f ? p.denominator : mask * float(p.batches * p.channels) + kMaskEpsilon;
        p.loss[0] = sum / normalized;
        p.mask_sum[0] = normalized;
    } else {
        p.loss[0] = sum / p.divisor;
    }
}
