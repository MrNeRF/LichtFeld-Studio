// SPDX-License-Identifier: GPL-3.0-or-later
[SpecializationConstant] const uint stage = 0;
struct Parameters {
    uint64_t corrected, raw, photo, mask, blurred, sharp_partials, blur_partials, branch_gradient;
    uint64_t sharp_map, cs_map, loss, grad, raw_grad;
    int width, height, channels, batches, byte_photo, byte_mask, radius;
    float sw, denominator, taps[6];
};
struct Push {
    uint64_t parameters;
};
[[vk::push_constant]] Push push;
static const float window[6] = {0.26601171493530273f, 0.21300552785396576f, 0.10936068743467331f,
                                0.036000773310661316f, 0.0075987582094967365f, 0.001028380123898387f};
float readF(uint64_t p, int i) { return ((Ptr<float, Access.Read>)p)[i]; }
void writeF(uint64_t p, int i, float v) { ((Ptr<float, Access.ReadWrite>)p)[i] = v; }
float target(Parameters p, int i) {
    return p.byte_photo ? float(((Ptr<uint8_t, Access.Read>)p.photo)[i]) * (1.f / 255.f) : readF(p.photo, i);
}
float weight(Parameters p, int i) {
    if (!p.mask)
        return 1;
    return p.byte_mask ? float(((Ptr<uint8_t, Access.Read>)p.mask)[i] != 0) : readF(p.mask, i);
}
bool valid(Parameters p, int x, int y) {
    return p.mask || ((p.width <= 10 || (x >= 5 && x < p.width - 5)) && (p.height <= 10 || (y >= 5 && y < p.height - 5)));
}
bool inside(Parameters p, int x, int y) { return x >= 0 && y >= 0 && x < p.width && y < p.height; }
float4 partials(uint64_t p, int i) {
    let v = ((Ptr<uint2, Access.Read>)p)[i];
    return float4(f16tof32(v.x), f16tof32(v.x >> 16), f16tof32(v.y), f16tof32(v.y >> 16));
}
void putPartials(uint64_t p, int i, float4 v) {
    ((Ptr<uint2, Access.ReadWrite>)p)[i] = uint2(f32tof16(v.x) | (f32tof16(v.y) << 16), f32tof16(v.z) | (f32tof16(v.w) << 16));
}
float2 pairAt(uint64_t p, int i) {
    uint v = ((Ptr<uint, Access.Read>)p)[i];
    return float2(f16tof32(v), f16tof32(v >> 16));
}
float branch(float mc, float mr, float mrr, float my, float myy, float mry, float sw, out float cs, out float4 deriv) {
    float A = mc * mc + my * my + 0.0001f, C = 2 * mc * my + 0.0001f, lum = C / A;
    float B = (mrr - mr * mr) + (myy - my * my) + 0.0009f, D = 2 * (mry - mr * my) + 0.0009f;
    cs = D / B;
    float ds1 = sw * (-(lum * D) / (B * B)), ds12 = sw * ((2 * lum) / B);
    deriv = float4(ds1 * (-2 * mr) + ds12 * (-my), ds1, ds12, cs * (2 * (my * A - mc * C) / (A * A)));
    return lum * cs;
}
// Horizontal sums for a 16x16 output tile and its five-pixel vertical halo.
groupshared float4 firstRows[26][16], secondRows[26][16], thirdRows[26][16];
[numthreads(16, 16, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    Parameters p = ((Ptr<Parameters, Access.Read>)push.parameters)[0];
    int nx = (p.width + 15) / 16, ny = (p.height + 15) / 16, pixels = p.width * p.height;
    uint3 groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
    for (int tile = group.x; tile < nx * ny * p.batches; tile += groups.x) {
        int b = tile / (nx * ny), ox = (tile % nx) * 16, oy = ((tile / nx) % ny) * 16;
        int lx = local.x, ly = local.y, x = ox + lx, y = oy + ly, i = y * p.width + x;
        bool live = inside(p, x, y);
        float anchor = p.mask ? 1.f / max(1.f, weight(p, min(oy + 8, p.height - 1) * p.width + min(ox + 8, p.width - 1))) : 1;
        float ss = 0, cs = 0, loss = 0;
        for (int c = 0; c < p.channels; ++c) {
            int base = (b * p.channels + c) * pixels;
            for (int sy = ly; sy < 26; sy += 16) {
                int yy = oy + sy - 5;
                float4 A = 0, B = 0, C = 0;
                for (int d = -5; d <= 5; ++d) {
                    int xx = x + d, j = yy * p.width + xx, g = base + j;
                    bool ok = inside(p, xx, yy);
                    float w = (ok ? weight(p, j) : 1.f) * anchor;
                    float t = abs(d) <= p.radius ? p.taps[abs(d)] : 0;
                    if (stage == 0) {
                        if (ok && w > 0)
                            A.xyz += t * float3(readF(p.corrected, g), readF(p.raw, g), 1);
                    } else if (stage == 1) {
                        float xc = ok ? readF(p.corrected, g) : 0, xr = ok ? readF(p.raw, g) : 0, photo = ok ? target(p, g) : 0;
                        float2 blurred = ok ? ((Ptr<float2, Access.Read>)p.blurred)[g] : float2(0);
                        float scale = window[abs(d)] * w;
                        A += scale * float4(xc, xr, xr * xr, xr * photo);
                        B += scale * float4(photo, photo * photo, blurred.x, blurred.y);
                        C.xyz += scale * float3(blurred.y * blurred.y, blurred.y * photo, 1);
                    } else if (ok) {
                        if (valid(p, xx, yy))
                            A += window[abs(d)] * partials(stage == 2 ? p.blur_partials : p.sharp_partials, g);
                        if (stage == 2 && w > 0)
                            B.xy += t * float2(readF(p.raw, g), 1);
                        if (stage == 3)
                            B.xy += t * pairAt(p.branch_gradient, g);
                    }
                }
                firstRows[sy][lx] = A;
                secondRows[sy][lx] = B;
                thirdRows[sy][lx] = C;
            }
            GroupMemoryBarrierWithGroupSync();
            if (live) {
                float4 A = 0, B = 0, C = 0;
                for (int d = -5; d <= 5; ++d) {
                    float t = abs(d) <= p.radius ? p.taps[abs(d)] : 0, w = window[abs(d)];
                    A += (stage == 0 ? t : w) * firstRows[ly + 5 + d][lx];
                    B += (stage == 1 ? w : t) * secondRows[ly + 5 + d][lx];
                    if (stage == 1)
                        C += w * thirdRows[ly + 5 + d][lx];
                }
                int g = base + i;
                float center = weight(p, i);
                if (stage == 0) {
                    ((Ptr<float2, Access.ReadWrite>)p.blurred)[g] = A.z > 1e-30f ? A.xy / A.z : float2(0);
                } else if (stage == 1) {
                    float inv = p.mask ? (C.z >= 1e-30f ? 1.f / C.z : 0) : 1;
                    A *= inv;
                    B *= inv;
                    C *= inv;
                    float cs0, cs1;
                    float4 p0, p1;
                    float s0 = branch(A.x, A.y, A.z, B.x, B.y, A.w, p.sw, cs0, p0);
                    float s1 = branch(B.z, B.w, C.x, B.x, B.y, C.y, p.sw, cs1, p1);
                    float diff = 50.f * (s0 - s1), e = exp(-abs(diff)), high = 1.f / (1.f + e), low = e * high;
                    bool flat = B.y - B.x * B.x < .0005f;
                    float w0 = flat ? 1 : (diff >= 0 ? high : low), w1 = flat ? 0 : (diff >= 0 ? low : high);
                    float blend = flat ? s0 : max(s0, s1) + (log(1.f + e) - 0.69314718055994531f) / 50.f;
                    float dw = center * anchor * inv;
                    float4 scale = dw * float4(p.mask ? 0.0625f : 1.f, 1, 1, 1);
                    putPartials(p.sharp_partials, g, p0 * (w0 * scale));
                    putPartials(p.blur_partials, g, p1 * (w1 * scale) * float4(1, 1, 1, p.sw));
                    ss += s0;
                    cs += w0 * cs0 + w1 * cs1;
                    loss += valid(p, x, y) ? center * ((1 - p.sw) * abs(readF(p.corrected, g) - target(p, g)) + p.sw * (1 - blend)) : 0;
                } else if (stage == 2) {
                    float n = B.y > 1e-30f ? 1.f / B.y : 0;
                    float v = -center * n * A.w, u = center * n * ((p.mask ? -16.f : -1.f) * A.x - 2 * (B.x * n) * A.y - target(p, g) * A.z);
                    ((Ptr<uint, Access.ReadWrite>)p.branch_gradient)[g] = f32tof16(v) | (f32tof16(u) << 16);
                } else {
                    float photo = target(p, g), xc = readF(p.corrected, g), xr = readF(p.raw, g);
                    float l1 = (1 - p.sw) * sign(xc - photo) * (p.mask ? center : float(valid(p, x, y)));
                    float gc = (center * (-p.sw) * A.w + l1 + (center > 0 ? B.x : 0)) / p.denominator;
                    float gr = (center * ((p.mask ? -16.f : -1.f) * A.x - 2 * xr * A.y - photo * A.z) + (center > 0 ? B.y : 0)) / p.denominator;
                    writeF(p.grad, g, p.raw_grad ? gc : gc + gr);
                    if (p.raw_grad)
                        writeF(p.raw_grad, g, gr);
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
        if (stage == 1 && live) {
            writeF(p.sharp_map, b * pixels + i, weight(p, i) > 0 ? ss / p.channels : 1);
            writeF(p.cs_map, b * pixels + i, weight(p, i) > 0 ? cs / p.channels : 1);
            writeF(p.loss, b * pixels + i, loss);
        }
    }
}
