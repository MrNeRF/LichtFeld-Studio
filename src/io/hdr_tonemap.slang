// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
// HDR10, HLG and Dolby Vision to 8-bit sRGB, ported from libplacebo v7.360
// pl_render_image with pl_render_default_params: lanczos chroma upsampling and
// upscaling, hermite downscaling, peak detection, spline tone mapping in IPT,
// the perceptual gamut mapping 3D LUT and blue noise dithering. Inputs read
// through fp16() are the ones libplacebo keeps in an RGBA16F FBO.

struct Constants {
    uint4 component[3];  // byte offset, row pitch, pixel step, bit shift
    uint4 layout;        // two-byte samples, sample mask, chroma width, chroma height
    uint4 flags;         // chroma subsampled in x, in y, Dolby Vision, HLG
    float4 chroma;       // luma pixel to chroma texel: x scale, x offset, y scale, y offset
    float4 sample;       // x: texel value per code, y: bit depth normalization
    float4 decode[3];    // YCbCr or Dolby Vision nonlinear matrix rows, offset in w
    float4 dovi_lms[3];  // Dolby Vision LMS to BT.2020 RGB
    float4 hlg;          // OETF^-1 black lift scale and offset, peak, OOTF gamma - 1
    float4 luma;         // source RGB to Y
    float4 rgb_to_lms[3];
    float4 lms_to_rgb[3];
    float4 tone[3];      // input min, max, output min, max; pivots, P; Q
    float4 target;       // gamut LUT I scale, offset; sRGB black scale, offset
    float4 gamut;        // min/max luma (PQ), min/max RGB
    float4 gamut_in[3];  // input gamut RGB to LMS
    float4 gamut_in_inverse[3];
    float4 gamut_out[3];
    float4 gamut_out_inverse[3];
};

struct Parameters {
    Ptr<uint8_t, Access.Read> planes;
    Ptr<float4, Access.Read> source;
    Ptr<float4, Access.ReadWrite> image;
    Ptr<Constants, Access.Read> c;
    Ptr<float, Access.Read> dovi;
    Ptr<float, Access.ReadWrite> lut;
    Ptr<float, Access.Read> dither;
    Ptr<uint8_t, Access.ReadWrite> output;
    Ptr<uint, Access.ReadWrite> tiles;
    Ptr<float, Access.Read> pq_table;  // libplacebo's 1024-step PQ EOTF table, padded
    uint source_width;
    uint source_height;
    uint target_width;
    uint target_height;
    uint mode;        // resample: vertical; render: rotation; render and resample: + 4 FP16 input
    uint taps;        // resample: 0 lanczos, else hermite taps; render: output channels
    float blur;
    uint gamut_map;
};
[[vk::push_constant]] ConstantBuffer<Parameters> p;

static const float PQ_M1 = 2610.0 / 4096.0 / 4.0;
static const float PQ_M2 = 2523.0 / 4096.0 * 128.0;
static const float PQ_C1 = 3424.0 / 4096.0;
static const float PQ_C2 = 2413.0 / 4096.0 * 32.0;
static const float PQ_C3 = 2392.0 / 4096.0 * 32.0;
static const float HLG_A = 0.17883277;
static const float HLG_B = 0.28466892;
static const float HLG_C = 0.55991073;
static const float PI = 3.14159265358979323846;
static const uint DOVI_COMPONENT = 226;  // pivot count, 9 pivots, 8 pieces of 27
static const uint GAMUT_I = 48;
static const uint GAMUT_C = 32;
static const uint GAMUT_H = 256;

float3 transform(float4 rows[3], float3 v) {
    return float3(dot(rows[0].xyz, v), dot(rows[1].xyz, v), dot(rows[2].xyz, v));
}

float mix(float a, float b, float x) {
    return x * b + (1.0 - x) * a;
}

// Apple GPUs round render target stores to FP16 toward zero.
float3 fp16(float3 v) {
    uint3 h = f32tof16(v);
    h = select(abs(f16tof32(h)) > abs(v), h - 1u, h);
    return f16tof32(h);
}

float pq_eotf(float x) {
    x = pow(max(x, 0.0), 1.0 / PQ_M2);
    x = max(x - PQ_C1, 0.0) / (PQ_C2 - PQ_C3 * x);
    return pow(x, 1.0 / PQ_M1);
}

float3 pq_eotf(float3 v) {
    return float3(pq_eotf(v.x), pq_eotf(v.y), pq_eotf(v.z));
}

float pq_oetf(float x) {
    x = pow(max(x, 0.0), PQ_M1);
    x = (PQ_C1 + PQ_C2 * x) / (1.0 + PQ_C3 * x);
    return pow(x, PQ_M2);
}

float3 pq_oetf(float3 v) {
    return float3(pq_oetf(v.x), pq_oetf(v.y), pq_oetf(v.z));
}

// ---- decode: planes to nonlinear RGB (pass_read_image, pl_shader_decode_color)

uint load(Constants q, uint c, int x, int y) {
    const uint4 component = q.component[c];
    const uint at = component.x + uint(y) * component.y + uint(x) * component.z;
    uint value = p.planes[at];
    if (q.layout.x != 0u)
        value |= uint(p.planes[at + 1u]) << 8;
    return (value >> component.w) & q.layout.y;
}

float lanczos(float x) {
    x = abs(x);
    if (x > 3.0)
        return 0.0;
    if (x < 1e-8)
        return 1.0;
    const float a = x * PI;
    const float b = a / 3.0;
    return sin(a) / a * (sin(b) / b);
}

void lanczos_row(float offset, out float w[6]) {
    float sum = 0.0;
    for (int n = 0; n < 6; ++n) {
        w[n] = lanczos(float(n) - 2.0 - offset);
        sum += w[n];
    }
    for (int n = 0; n < 6; ++n)
        w[n] /= sum;
}

// pl_shader_sample_ortho2 taps around a texel-space position. libplacebo reads
// the normalized weights from 256 subpixel rows, linearly interpolated.
int lanczos_weights(float position, out float w[6]) {
    const float start = floor(position);
    const float row = (position - start) * 255.0;
    const float first = floor(row);
    float a[6];
    float b[6];
    lanczos_row(first / 255.0, a);
    lanczos_row(min(first + 1.0, 255.0) / 255.0, b);
    for (int n = 0; n < 6; ++n)
        w[n] = lerp(a[n], b[n], row - first);
    return int(start) - 2;
}

float2 load_chroma(Constants q, int x, int y) {
    return float2(load(q, 1u, x, y), load(q, 2u, x, y)) * q.sample.x;
}

// Vertical pass into an FP16 FBO, then horizontal, as libplacebo upsamples.
float2 chroma(Constants q, int2 pixel) {
    const int2 size = int2(q.layout.zw);
    const float2 position = float2(pixel) * q.chroma.xz + q.chroma.yw;
    float wy[6];
    float wx[6];
    const int y0 = q.flags.y != 0u ? lanczos_weights(position.y, wy) : pixel.y;
    const int x0 = q.flags.x != 0u ? lanczos_weights(position.x, wx) : pixel.x;
    const int columns = q.flags.x != 0u ? 6 : 1;
    float2 result = float2(0.0);
    for (int i = 0; i < columns; ++i) {
        const int x = clamp(x0 + i, 0, size.x - 1);
        float2 column = float2(0.0);
        if (q.flags.y != 0u) {
            for (int j = 0; j < 6; ++j)
                column += wy[j] * load_chroma(q, x, clamp(y0 + j, 0, size.y - 1));
            column *= q.sample.y;
            if (q.flags.x != 0u)
                column = fp16(float3(column, 0.0)).xy;
        } else {
            column = q.sample.y * load_chroma(q, x, y0);
        }
        result += q.flags.x != 0u ? wx[i] * column : column;
    }
    return result;
}

// pl_shader_dovi_reshape on one component.
float reshape(uint c, float value, float3 signal) {
    const uint base = c * DOVI_COMPONENT;
    const uint pivots = uint(p.dovi[base]);
    if (pivots == 0u)
        return value;
    const float s = signal[c];
    uint piece = 0u;
    for (uint i = 1u; i + 1u < pivots; ++i)
        piece += s >= p.dovi[base + 1u + i] ? 1u : 0u;
    const uint k = base + 10u + piece * 27u;
    float result;
    if (p.dovi[k] == 0.0) {
        result = (p.dovi[k + 3u] * s + p.dovi[k + 2u]) * s + p.dovi[k + 1u];
    } else {
        const uint order = uint(p.dovi[k + 4u]);
        const float4 cross = float4(signal.xxy * signal.yzz, signal.x * signal.y * signal.z);
        float3 power = signal;
        float4 cross_power = cross;
        result = p.dovi[k + 5u];
        for (uint j = 0u; j < order; ++j) {
            const uint m = k + 6u + j * 7u;
            result += dot(float3(p.dovi[m], p.dovi[m + 1u], p.dovi[m + 2u]), power);
            result += dot(float4(p.dovi[m + 3u], p.dovi[m + 4u], p.dovi[m + 5u], p.dovi[m + 6u]), cross_power);
            power *= signal;
            cross_power *= cross;
        }
    }
    return clamp(result, p.dovi[base + 1u], p.dovi[base + pivots]);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void decode(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= p.source_width || tid.y >= p.source_height)
        return;
    const Constants q = p.c[0];
    const int2 pixel = int2(tid.xy);
    float3 color = float3(q.sample.y * (float(load(q, 0u, pixel.x, pixel.y)) * q.sample.x), chroma(q, pixel));
    if (q.flags.z != 0u) {
        const float3 signal = clamp(color, 0.0, 1.0);
        color = float3(reshape(0u, color.x, signal), reshape(1u, color.y, signal), reshape(2u, color.z, signal));
    }
    color = transform(q.decode, color) + float3(q.decode[0].w, q.decode[1].w, q.decode[2].w);
    if (q.flags.z != 0u)
        color = pq_oetf(transform(q.dovi_lms, pq_eotf(color)));
    p.image[tid.y * p.source_width + tid.x] = float4(color, 1.0);
}

// ---- peak detection (pl_shader_detect_peak, percentile 100, black cutoff 1%)

float3 linearize(Constants q, float3 color) {
    color = max(color, 0.0);
    if (q.flags.w == 0u)
        return pq_eotf(color) * (10000.0 / 203.0);
    color = q.hlg.x * color + q.hlg.y;
    color = select(color > 0.5, exp((color - HLG_C) * (1.0 / HLG_A)) + HLG_B, 4.0 * color * color);
    color *= 1.0 / 12.0;
    return color * (q.hlg.z * pow(max(dot(q.luma.xyz, color), 0.0), q.hlg.w));
}

// One thread per libplacebo 16x16 work group: PQ sum, maximum, black count.
[shader("compute")]
[numthreads(64, 1, 1)]
void measure(uint3 tid : SV_DispatchThreadID) {
    const uint columns = (p.source_width + 15u) / 16u;
    if (tid.x >= columns * ((p.source_height + 15u) / 16u))
        return;
    const Constants q = p.c[0];
    const uint2 origin = uint2(tid.x % columns, tid.x / columns) * 16u;
    const uint2 last = uint2(p.source_width, p.source_height) - 1u;
    uint sum = 0u;
    uint peak = 0u;
    uint black = 0u;
    for (uint i = 0u; i < 256u; ++i) {
        const uint2 pixel = min(origin + uint2(i % 16u, i / 16u), last);
        const float3 color = linearize(q, p.source[pixel.y * p.source_width + pixel.x].xyz);
        float luma = dot(q.luma.xyz, color) * (203.0 / 10000.0);
        luma = pow(clamp(luma, 0.0, 1.0), PQ_M1);
        luma = (PQ_C1 + PQ_C2 * luma) / (1.0 + PQ_C3 * luma);
        luma = pow(luma, PQ_M2);
        luma *= smoothstep(0.0, 0.01, luma);
        const uint y = uint(16383.0 * luma);
        sum += y;
        peak = max(peak, y);
        black += y == 0u ? 1u : 0u;
    }
    p.tiles[tid.x * 3u] = sum;
    p.tiles[tid.x * 3u + 1u] = peak;
    p.tiles[tid.x * 3u + 2u] = black;
}

// ---- main scaler (pl_shader_sample_ortho2), one axis per dispatch

float hermite(float x) {
    x = abs(x);
    if (x > p.blur)
        return 0.0;
    x /= p.blur;
    return x < 1.0 ? (6.0 + x * x * (-18.0 + x * 12.0)) / 6.0 : 0.0;
}

float4 read_source(uint index) {
    const float4 color = p.source[index];
    return (p.mode & 4u) != 0u ? float4(fp16(color.xyz), 1.0) : color;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void resample(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= p.target_width || tid.y >= p.target_height)
        return;
    const bool vertical = (p.mode & 1u) != 0u;
    const uint from = vertical ? p.source_height : p.source_width;
    const uint to = vertical ? p.target_height : p.target_width;
    const float position = (float(vertical ? tid.y : tid.x) + 0.5) * float(from) / float(to) - 0.5;
    const float start = floor(position);
    const int taps = p.taps == 0u ? 6 : int(p.taps);
    float w[16];
    int base = int(start) - (taps / 2 - 1);
    if (p.taps == 0u) {
        float lanczos_w[6];
        base = lanczos_weights(position, lanczos_w);
        for (int n = 0; n < 6; ++n)
            w[n] = lanczos_w[n];
    } else {
        float sum = 0.0;
        for (int n = 0; n < taps; ++n) {
            w[n] = hermite(float(n) - float(taps / 2 - 1) - (position - start));
            sum += w[n];
        }
        for (int n = 0; n < taps; ++n)
            w[n] /= sum;
    }
    float3 color = float3(0.0);
    for (int n = 0; n < taps; ++n) {
        const uint at = uint(clamp(base + n, 0, int(from) - 1));
        color += w[n] * read_source(vertical ? at * p.source_width + tid.x : tid.y * p.source_width + at).xyz;
    }
    p.image[tid.y * p.target_width + tid.x] = float4(color, 1.0);
}

// ---- perceptual gamut mapping LUT (pl_gamut_map_perceptual)

static const float GAMUT_DELTA = 5e-5;

// gamut_mapping.c evaluates the PQ EOTF by linear interpolation of a table.
float pq_eotf_table(float x) {
    const float index = clamp(x, 0.0, 1.0) * 1023.0;
    const uint i = uint(floor(index));
    return mix(p.pq_table[i], p.pq_table[i + 1u], index - floor(index));
}

float3 ipt_to_lms_pq(float3 ipt) {
    return float3(ipt.x + 0.0975689 * ipt.y + 0.205226 * ipt.z,
                  ipt.x - 0.1138760 * ipt.y + 0.133217 * ipt.z,
                  ipt.x + 0.0326151 * ipt.y - 0.676887 * ipt.z);
}

float3 rgb_to_ipt(float3 rgb, bool output) {
    const Constants q = p.c[0];
    const float3 lms = pq_oetf(output ? transform(q.gamut_out, rgb) : transform(q.gamut_in, rgb));
    return float3(0.4000 * lms.x + 0.4000 * lms.y + 0.2000 * lms.z,
                  4.4550 * lms.x - 4.8510 * lms.y + 0.3960 * lms.z,
                  0.8056 * lms.x + 0.3572 * lms.y - 1.1628 * lms.z);
}

float3 ipt_to_rgb(float3 ipt, bool output) {
    const Constants q = p.c[0];
    const float3 lms_pq = ipt_to_lms_pq(ipt);
    const float3 lms = float3(pq_eotf_table(lms_pq.x), pq_eotf_table(lms_pq.y), pq_eotf_table(lms_pq.z));
    return output ? transform(q.gamut_out_inverse, lms) : transform(q.gamut_in_inverse, lms);
}

float3 ich_to_ipt(float3 ich) {
    return float3(ich.x, ich.y * cos(ich.z), ich.y * sin(ich.z));
}

bool in_gamut(float3 ipt, bool output) {
    const float4 g = p.c[0].gamut;
    const float3 lms = ipt_to_lms_pq(ipt);
    if (any(lms < g.x) || any(lms > g.y))
        return false;
    const float3 rgb = ipt_to_rgb(ipt, output);
    return all(rgb >= g.z) && all(rgb <= g.w);
}

float3 desaturate_bounded(float i, float h, float c_min, float c_max, bool output) {
    const float4 g = p.c[0].gamut;
    if (i <= g.x)
        return float3(g.x, 0.0, h);
    if (i >= g.y)
        return float3(g.y, 0.0, h);
    const float max_di = i * GAMUT_DELTA;
    float3 result = float3(i, (c_min + c_max) / 2.0, h);
    do {
        if (in_gamut(ich_to_ipt(result), output))
            c_min = result.y;
        else
            c_max = result.y;
        result.y = (c_min + c_max) / 2.0;
    } while (c_max - c_min > max_di);
    return result;
}

// Maximally saturated in-gamut color for a hue, by golden section search.
float3 most_saturated(float hue, bool output) {
    const float invphi = 0.6180339887498948;
    const float invphi2 = 0.38196601125010515;
    const float4 g = p.c[0].gamut;
    float3 lo = float3(g.x, 0.0, hue);
    float3 hi = float3(g.y, 0.0, hue);
    float de = hi.x - lo.x;
    float3 a = desaturate_bounded(lo.x + invphi2 * de, hue, 0.0, 0.5, output);
    float3 b = desaturate_bounded(lo.x + invphi * de, hue, 0.0, 0.5, output);
    while (de > GAMUT_DELTA) {
        de *= invphi;
        if (a.y > b.y) {
            hi = b;
            b = a;
            a = desaturate_bounded(lo.x + invphi2 * de, hue, lo.y - GAMUT_DELTA, 0.5, output);
        } else {
            lo = a;
            a = b;
            b = desaturate_bounded(lo.x + invphi * de, hue, hi.y - GAMUT_DELTA, 0.5, output);
        }
    }
    return a.y > b.y ? a : b;
}

float softclip(float value, float source, float target) {
    const float j = 0.70;
    const float peak = source / target;
    const float x = min(value / target, peak);
    if (x <= j || peak <= 1.0)
        return value;
    const float a = -j * j * (peak - 1.0) / (j * j - 2.0 * j + peak);
    const float b = (j * j - 2.0 * j * peak + peak) / max(1e-6, peak - 1.0);
    const float scale = (b * b + 2.0 * b * j + j * j) / (b - a);
    return scale * (x + a) / (x + b) * target;
}

float smoothstep_placebo(float edge0, float edge1, float x) {
    x = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return x * x * (3.0 - 2.0 * x);
}

// One thread per (C, h) column of the I x C x h LUT, stored as RGBA16 UNORM.
[shader("compute")]
[numthreads(64, 1, 1)]
void gamut(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= GAMUT_C * GAMUT_H)
        return;
    const float4 g = p.c[0].gamut;
    const uint column = tid.x % GAMUT_C;
    const uint slice = tid.x / GAMUT_C;
    const float hue = mix(-PI, PI, float(slice) / float(GAMUT_H - 1u));
    const float3 source_peak = most_saturated(hue, false);
    const float3 target_peak = most_saturated(hue, true);
    const float max_c = max(source_peak.y, target_peak.y);
    for (uint i = 0u; i < GAMUT_I; ++i) {
        float3 ipt = ich_to_ipt(float3(mix(g.x, g.y, float(i) / float(GAMUT_I - 1u)),
                                       mix(0.0, 0.5, float(column) / float(GAMUT_C - 1u)), hue));
        const float3 mapped = rgb_to_ipt(ipt_to_rgb(ipt, false), true);
        const float k = smoothstep_placebo(0.30, 1.0, sqrt(ipt.y * ipt.y + ipt.z * ipt.z) / max_c) * 0.80;
        ipt = float3(mix(ipt.x, mapped.x, k), mix(ipt.y, mapped.y, k), mix(ipt.z, mapped.z, k));
        float3 rgb = ipt_to_rgb(ipt, true);
        const float max_rgb = max(rgb.x, max(rgb.y, rgb.z));
        rgb = float3(max(softclip(rgb.x, max_rgb, g.w), g.z), max(softclip(rgb.y, max_rgb, g.w), g.z),
                     max(softclip(rgb.z, max_rgb, g.w), g.z));
        ipt = rgb_to_ipt(rgb, true);
        const uint at = ((slice * GAMUT_C + column) * GAMUT_I + i) * 3u;
        p.lut[at] = round(ipt.x * 65535.0) / 65535.0;
        p.lut[at + 1u] = round(ipt.y * 65535.0 + 32767.0) / 65535.0;
        p.lut[at + 2u] = round(ipt.z * 65535.0 + 32767.0) / 65535.0;
    }
}

// ---- color map, dither and output (pl_shader_color_map_ex, pl_shader_dither)

float3 lut_texel(uint3 index) {
    const uint at = ((index.z * GAMUT_C + index.y) * GAMUT_I + index.x) * 3u;
    return float3(p.lut[at], p.lut[at + 1u], p.lut[at + 2u]);
}

float3 gamut_lookup(Constants q, float3 ipt) {
    float3 index = float3(q.target.x * ipt.x + q.target.y, 2.0 * length(ipt.yz),
                          0.15915494309189535 * atan2(ipt.z, ipt.y) + 0.5);
    index = clamp(index, 0.0, 1.0) * float3(GAMUT_I - 1u, GAMUT_C - 1u, GAMUT_H - 1u);
    const uint3 i0 = uint3(floor(index));
    const uint3 i1 = min(i0 + 1u, uint3(GAMUT_I - 1u, GAMUT_C - 1u, GAMUT_H - 1u));
    const float3 f = index - floor(index);
    const float3 c00 = lerp(lut_texel(i0), lut_texel(uint3(i1.x, i0.y, i0.z)), f.x);
    const float3 c10 = lerp(lut_texel(uint3(i0.x, i1.y, i0.z)), lut_texel(uint3(i1.x, i1.y, i0.z)), f.x);
    const float3 c01 = lerp(lut_texel(uint3(i0.x, i0.y, i1.z)), lut_texel(uint3(i1.x, i0.y, i1.z)), f.x);
    const float3 c11 = lerp(lut_texel(uint3(i0.x, i1.y, i1.z)), lut_texel(i1), f.x);
    const float3 result = lerp(lerp(c00, c10, f.y), lerp(c01, c11, f.y), f.z);
    return float3(result.x, result.yz - 32768.0 / 65535.0);
}

float tone_map(Constants q, float x) {
    x = clamp(x, q.tone[0].x, q.tone[0].y) - q.tone[1].x;
    x = x > 0.0 ? ((q.tone[2].x * x + q.tone[2].y) * x + q.tone[2].z) * x : (q.tone[1].z * x + q.tone[1].w) * x;
    return clamp(x + q.tone[1].y, q.tone[0].z, q.tone[0].w);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void render(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= p.target_width || tid.y >= p.target_height)
        return;
    const Constants q = p.c[0];
    const uint rotation = p.mode & 3u;
    uint2 pixel = tid.xy;
    if (rotation == 1u)
        pixel = uint2(tid.y, p.source_height - 1u - tid.x);
    else if (rotation == 2u)
        pixel = uint2(p.source_width - 1u - tid.x, p.source_height - 1u - tid.y);
    else if (rotation == 3u)
        pixel = uint2(p.source_width - 1u - tid.y, tid.x);
    float3 color = linearize(q, read_source(pixel.y * p.source_width + pixel.x).xyz);

    float3 lms = transform(q.rgb_to_lms, color);
    float3 ipt = pq_oetf(0.0203000009059906 * lms);
    ipt = float3(0.4000 * ipt.x + 0.4000 * ipt.y + 0.2000 * ipt.z,
                 4.4550 * ipt.x - 4.8510 * ipt.y + 0.3960 * ipt.z,
                 0.8056 * ipt.x + 0.3572 * ipt.y - 1.1628 * ipt.z);
    const float i_orig = ipt.x;
    ipt.x = tone_map(q, ipt.x);
    float2 hull = float2(i_orig, ipt.x);
    hull = ((hull - 6.0) * hull + 9.0) * hull;
    ipt.yz *= min(i_orig / ipt.x, hull.y / hull.x);
    if (p.gamut_map != 0u)
        ipt = gamut_lookup(q, ipt);
    lms = pq_eotf(ipt_to_lms_pq(ipt)) * 49.261085510253906;
    color = transform(q.lms_to_rgb, lms);

    color = max(q.target.z * color + q.target.w, 0.0);
    color = select(color >= 0.0031308, 1.055 * pow(color, 1.0 / 2.4) - 0.055, color * 12.92);
    const float bias = p.dither[(tid.y % 64u) * 64u + tid.x % 64u];
    const uint3 value = uint3(clamp(floor(255.0 * color + bias), 0.0, 255.0));
    const uint at = (tid.y * p.target_width + tid.x) * p.taps;
    p.output[at] = uint8_t(value.x);
    p.output[at + 1u] = uint8_t(value.y);
    p.output[at + 2u] = uint8_t(value.z);
    if (p.taps == 4u)
        p.output[at + 3u] = uint8_t(255u);
}
