/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "config.slang"
import utils;

// SteadySplats Algorithm 2, in gather form. A thread owns one source pixel's
// deduplicated FIFO. It publishes four outgoing weights; the resolve pass
// gathers them, avoiding floating-point atomics and cross-workgroup races.
static const uint HISTORY_SIZE = 4u;
static const uint EMPTY_ID = 0xfffffffeu;
static const uint BACKGROUND_ID = 0xffffffffu;
static const int2 OFFSETS[4] = {int2(-1, 0), int2(1, 0), int2(0, -1), int2(0, 1)};
// An occluder takes 8 bytes, as in the authors' viewer: the splat ID, then a
// 16-bit depth key (low half) and four 4-bit occlusion ratios (high half).
// Empty entries are also farthest and occlude nothing.
static const uint2 EMPTY_ENTRY = uint2(EMPTY_ID, 0xffffffffu);

layout(binding=0) StructuredBuffer<uint> sample_ids;
layout(binding=1) StructuredBuffer<float> sample_depths;
layout(binding=2) StructuredBuffer<float2> positions;
layout(binding=3) StructuredBuffer<float4> conics;
layout(binding=4) RWStructuredBuffer<uint4> history; // Two entries per element.
layout(binding=5) RWStructuredBuffer<float4> outgoing_weights;

// Monotone key of the squared radial distance: its float exponent within
// [2^-28, 2^36) and the top 10 mantissa bits. The distance keeps 11 bits of
// relative precision up to 2^18; farther splats and the background saturate.
uint depth_key(float metric) {
    const uint low = (127u - 28u) << 23;
    uint bits = asuint(metric);
    return bits < low ? 0u : min((bits - low) >> 13, 0xffffu);
}

// Steps of 1/15. Code 15 is exactly 1; larger ratios clamp to it.
uint pack_ratios(float4 ratios) {
    uint4 q = uint4(saturate(ratios) * 15.0f + 0.5f);
    return q.x | (q.y << 4) | (q.z << 8) | (q.w << 12);
}

float4 unpack_ratios(uint bits) {
    return float4((uint4(bits, bits >> 4, bits >> 8, bits >> 12) & 15u)) / 15.0f;
}

float fragment_alpha(uint id, float2 pixel, Uniforms uniforms) {
    if (id == BACKGROUND_ID)
        return 1.0f;
    float4 conic = conics[id];
    float alpha = (uniforms.lod_enabled & 4u) != 0u
        ? evaluate_splat_alpha_spark_rad(positions[id], conic.w, conic.xyz, pixel)
        : evaluate_splat_alpha_for_profile(positions[id], conic.w, conic.xyz, pixel,
                                           uniforms.splat_render_profile);
    return alpha >= ALPHA_THRESHOLD ? alpha : 0.0f;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uniform Uniforms uniforms) {
    uint p = dispatchThreadID.x;
    uint count = uniforms.image_width * uniforms.image_height;
    if (p >= count)
        return;
    uint2 pixel = uint2(p % uniforms.image_width, p / uniforms.image_width);
    float2 origin = full_frame_binning(uniforms)
        ? float2(uniforms.render_origin_x, uniforms.render_origin_y) : float2(0.0f);
    float2 xy = float2(pixel) + origin;
    uint id = sample_ids[p];
    float depth = sample_depths[p];
    float source_alpha = fragment_alpha(id, xy, uniforms);
    float4 target_alpha;
    for (uint d = 0; d < 4; ++d)
        target_alpha[d] = fragment_alpha(id, xy + float2(OFFSETS[d]), uniforms);
    float4 weights = source_alpha > 0.0f ? target_alpha / source_alpha : float4(0.0f);

    uint key = depth_key(depth);
    uint2 entries[HISTORY_SIZE];
    for (uint i = 0; i < HISTORY_SIZE / 2u; ++i) {
        uint4 pair = uniforms.stochastic_history_valid != 0u
            ? history[p * (HISTORY_SIZE / 2u) + i] : uint4(EMPTY_ENTRY, EMPTY_ENTRY);
        entries[2u * i] = pair.xy;
        entries[2u * i + 1u] = pair.zw;
    }
    bool unique = id != BACKGROUND_ID;
    for (uint k = 0; k < HISTORY_SIZE; ++k) {
        uint2 entry = entries[k];
        if (entry.x == id)
            unique = false;
        // Empty slots and the sample itself never contribute an occlusion
        // factor. Repeated samples do not refresh or evict the rare entries.
        // Equal keys fall back to the sampler's lower-ID tie break.
        uint stored_key = entry.y & 0xffffu;
        if (entry.x != EMPTY_ID && entry.x != id &&
            (stored_key < key || (stored_key == key && entry.x < id)))
            weights *= unpack_ratios(entry.y >> 16);
    }
    // With K=4 and alpha capped below one, these products fit in float32.
    outgoing_weights[p] = weights;

    if (unique) {
        for (uint k = HISTORY_SIZE - 1u; k > 0u; --k)
            entries[k] = entries[k - 1u];
        entries[0] = uint2(id, key | (pack_ratios((1.0f - target_alpha) / (1.0f - source_alpha)) << 16));
    }
    if (unique || uniforms.stochastic_history_valid == 0u)
        for (uint i = 0; i < HISTORY_SIZE / 2u; ++i)
            history[p * (HISTORY_SIZE / 2u) + i] = uint4(entries[2u * i], entries[2u * i + 1u]);
}
