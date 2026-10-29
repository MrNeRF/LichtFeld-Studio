// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

// Minimal compute host for the compiler contract tests and install smoke
// checks. Real hosts bind LfsBoundEffect the same way. It declares the
// subgroup size its manifest entry pins, which the compiler checks.

#include "config.slang"
import lfs;

extern struct LfsBoundEffect : _LfsEffects;

[[vk::binding(0, 0)]] RWStructuredBuffer<float4> output;

[shader("compute")]
[WaveSize(SUBGROUP_SIZE)]
[numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    LfsElement e = {};
    LfsContext ctx = {};
    e.position = float3(tid);
    e.opacity = 1.0f;
    ctx.time = 1.0f;
    ctx.values[0] = float4(1.0f);
    LfsBoundEffect bound;
    e = bound.apply(tid.y, e, ctx);
    output[tid.x] = float4(e.position + e.color, e.opacity);
}
