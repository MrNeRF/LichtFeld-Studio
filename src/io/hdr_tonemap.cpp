/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "hdr_tonemap.hpp"

#define _USE_MATH_DEFINES
#include <cmath>
#include <algorithm>
#include <cstring>

namespace lfs::io {

// ── HLG (ARIB STD-B67 / Rec. 2100) ────────────────────────────────────────
static constexpr float HLG_A = 0.17883277f;
static constexpr float HLG_B = 0.28466892f;
static constexpr float HLG_C = 0.55991073f;

// ── PQ (SMPTE ST 2084) ────────────────────────────────────────────────────
static constexpr float PQ_M1 = 0.1593017578125f;
static constexpr float PQ_M2 = 78.84375f;
static constexpr float PQ_C1 = 0.8359375f;
static constexpr float PQ_C2 = 18.8515625f;
static constexpr float PQ_C3 = 18.6875f;

// ── BT.2020 luminance coeffs (for desaturation clamp) ─────────────────────
static constexpr float LUMA_R = 0.2627f;
static constexpr float LUMA_G = 0.6780f;
static constexpr float LUMA_B = 0.0593f;

// ── Helpers ───────────────────────────────────────────────────────────────

static inline float inv_hlg(float e) {
    if (e <= 0.5f)
        return (e * e) / 3.0f;
    return (std::exp((e - HLG_C) / HLG_A) + HLG_B) / 12.0f;
}

static inline float inv_pq(float e) {
    float e_pow = std::pow(e, 1.0f / PQ_M2);
    float num = std::max(e_pow - PQ_C1, 0.0f);
    float den = PQ_C2 - PQ_C3 * e_pow;
    return std::pow(num / den, 1.0f / PQ_M1);
}

static inline float bt709_oetf(float v) {
    v = std::clamp(v, 0.0f, 1.0f);
    if (v < 0.018f)
        return 4.5f * v;
    return 1.099f * std::pow(v, 0.45f) - 0.099f;
}

static inline float hable(float x) {
    const float A = 0.15f, B = 0.50f, C = 0.10f, D = 0.20f, E = 0.02f, F = 0.30f;
    return ((x * (A * x + C * B) + D * E) /
            (x * (A * x + B) + D * F)) - (E / F);
}

/// Safe clamp with desaturation — avoids hue shift from hard clamp
static inline void clamp_rgb(float& r, float& g, float& b) {
    float max_c = std::max({r, g, b});
    if (max_c > 1.0f) {
        float luma = LUMA_R * r + LUMA_G * g + LUMA_B * b;
        float scale = 1.0f / max_c;
        // Blend toward uniformly scaled to desaturate clipped colors
        float t = (max_c - 1.0f) / (max_c - luma * scale + 1e-6f);
        t = std::clamp(t, 0.0f, 1.0f);
        float rs = r * scale, gs = g * scale, bs = b * scale;
        r = rs * (1.0f - t) + luma * scale * t;
        g = gs * (1.0f - t) + luma * scale * t;
        b = bs * (1.0f - t) + luma * scale * t;
    }
    r = std::max(r, 0.0f);
    g = std::max(g, 0.0f);
    b = std::max(b, 0.0f);
}

// ── Public API ─────────────────────────────────────────────────────────────

HdrFormat detectHdrFormat(int color_trc, int) {
    switch (color_trc) {
    case 18:  return HdrFormat::HLG;
    case 16:  return HdrFormat::HDR10;
    default:  return HdrFormat::SDR;
    }
}

const char* hdrFormatLabel(HdrFormat fmt) {
    switch (fmt) {
    case HdrFormat::HLG:   return "HLG";
    case HdrFormat::HDR10: return "HDR10";
    default:               return "SDR";
    }
}

const char* hdrFormatType(HdrFormat fmt) {
    switch (fmt) {
    case HdrFormat::HLG:   return "HDR";
    case HdrFormat::HDR10: return "HDR";
    default:               return "SDR";
    }
}

void tonemapHdrToSdr(unsigned char* data, int width, int height, int stride,
                     HdrFormat format, float) {
    if (!data || width <= 0 || height <= 0)
        return;

    // Per-channel look-up table approach for HLG
    // Pre-compute HLG LUT for performance
    float hlg_lut[256];
    for (int i = 0; i < 256; ++i) {
        float e = i / 255.0f;
        float l = inv_hlg(e);
        // Gain 12: ref white (0.0833) → 1.0
        l *= 12.0f;
        // Simple Reinhard: x/(1+x) maps ref white 1.0 → 0.5 → BT.709 pixel 180
        l = l / (1.0f + l);
        hlg_lut[i] = l;
    }

    // Pre-compute PQ LUT
    float pq_lut[256];
    for (int i = 0; i < 256; ++i) {
        float e = i / 255.0f;
        float cd = inv_pq(e);
        // Normalize to 100 nit SDR peak + Hable tonemap
        float peak = 10000.0f / 100.0f;
        cd = hable(cd / 100.0f) / hable(peak);
        pq_lut[i] = cd;
    }

    for (int y = 0; y < height; ++y) {
        unsigned char* row = data + y * stride;
        for (int x = 0; x < width; ++x) {
            unsigned char* p = row + x * 3;

            // Look up tone-mapped linear values from LUT
            float lr, lg, lb;
            switch (format) {
            case HdrFormat::HLG:
                lr = hlg_lut[p[0]];
                lg = hlg_lut[p[1]];
                lb = hlg_lut[p[2]];
                break;
            case HdrFormat::HDR10:
                lr = pq_lut[p[0]];
                lg = pq_lut[p[1]];
                lb = pq_lut[p[2]];
                break;
            default:
                continue;
            }

            // Primaries BT.2020 → BT.709 → SKIPPED: sws_scale YUV→RGB
            // already outputs in BT.709/BT.601 primaries (default matrix).
            // Applying extra primaries would double-convert → oversaturation.
            // Gamut clamp (catches any out-of-range values from LUT rounding)
            clamp_rgb(lr, lg, lb);

            // BT.709 OETF + denormalize
            p[0] = static_cast<unsigned char>(
                std::clamp(std::round(bt709_oetf(lr) * 255.0f), 0.0f, 255.0f));
            p[1] = static_cast<unsigned char>(
                std::clamp(std::round(bt709_oetf(lg) * 255.0f), 0.0f, 255.0f));
            p[2] = static_cast<unsigned char>(
                std::clamp(std::round(bt709_oetf(lb) * 255.0f), 0.0f, 255.0f));
        }
    }
}

} // namespace lfs::io
