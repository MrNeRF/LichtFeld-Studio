// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

// Deterministic helpers for user element effects, re-exported by `import lfs`.

#include "reveal_noise.inc"

// Hash of a 3D point to three values in [0, 1), the one built-in reveals use.
public float3 lfs_hash(float3 p) {
    return lfs_reveal_hash(p);
}

// Smooth value noise in [0, 1) per channel, the one built-in reveals use.
public float3 lfs_noise(float3 p) {
    return lfs_reveal_noise(p);
}

// Smoothstep easing of t clamped to [0, 1].
public float lfs_ease(float t) {
    t = saturate(t);
    return t * t * (3.0f - 2.0f * t);
}

// Hue wraps; saturation and value are clamped to [0, 1]. Values are used as
// given, so linear RGB in gives linear RGB out.
public float3 lfs_hsv_to_rgb(float3 hsv) {
    float3 rgb = saturate(abs(frac(hsv.xxx + float3(1.0f, 2.0f / 3.0f, 1.0f / 3.0f)) * 6.0f - 3.0f) - 1.0f);
    return saturate(hsv.z) * lerp(float3(1.0f), rgb, saturate(hsv.y));
}

// Inverse of lfs_hsv_to_rgb for non-negative RGB; hue in [0, 1).
public float3 lfs_rgb_to_hsv(float3 rgb) {
    float4 k = float4(0.0f, -1.0f / 3.0f, 2.0f / 3.0f, -1.0f);
    float4 p = rgb.g < rgb.b ? float4(rgb.bg, k.wz) : float4(rgb.gb, k.xy);
    float4 q = rgb.r < p.x ? float4(p.xyw, rgb.r) : float4(rgb.r, p.yzx);
    float d = q.x - min(q.w, q.y);
    return float3(abs(q.z + (q.w - q.y) / (6.0f * d + 1e-10f)), d / (q.x + 1e-10f), q.x);
}
