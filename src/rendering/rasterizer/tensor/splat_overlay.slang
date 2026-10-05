// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
// Desktop overlay parameter ABI (buildOverlayParamsCpuFloats): crop boxes and
// ellipsoids, view window, emphasis and brush. Ported from viewer_overlay.metal.
#pragma once

static const uint kOverlayParameterCount = 207;

bool overlay_enabled(float x) { return x > .5f; }

float3 overlay_transform(Ptr<float4, Access.Read> p, uint base, float3 v) {
    const float4 h = float4(v, 1);
    return float3(dot(p[base], h), dot(p[base + 1], h), dot(p[base + 2], h));
}

void overlay_filter(Ptr<float4, Access.Read> p, uint base, bool ellipsoid, int node, float3 world,
                    inout bool active, inout uint flags) {
    const float4 f = p[base];
    if (!active || !overlay_enabled(f.x) || (int(round(f.w)) >= 0 && int(round(f.w)) != node))
        return;
    const float3 local = overlay_transform(p, base + (ellipsoid ? 2u : 3u), world);
    bool inside;
    if (ellipsoid) {
        const float3 radius = max(abs(p[base + 1].xyz), 1e-8f);
        const float3 normalized = local / radius;
        inside = dot(normalized, normalized) <= 1;
    } else {
        inside = all(local >= p[base + 1].xyz) && all(local <= p[base + 2].xyz);
    }
    if (inside == overlay_enabled(f.y)) {
        flags |= 2u;
        if (overlay_enabled(f.z))
            flags |= 1u;
        else
            active = false;
    }
}

float3 overlay_projection_color(float3 color, float2 center, uint flags, Ptr<float4, Access.Read> p) {
    const float2 delta = center - p[23].xy;
    if (overlay_enabled(p[21].x) && overlay_enabled(p[21].y) && (flags & 2u) == 0u && dot(delta, delta) <= p[23].z * p[23].z) {
        const float lum = dot(color, float3(.2126f, .7152f, .0722f));
        color = clamp(lum + (1 + p[21].z) * (color - lum), 0.f, 1.f);
    }
    if ((flags & 1u) != 0u)
        color = float3(dot(color, float3(.2126f, .7152f, .0722f)) * .25f);
    return color;
}
