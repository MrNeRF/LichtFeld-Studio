// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

#include "selection_outline_common.slang"
layout(binding=0) StructuredBuffer<uint> outline_ids;
layout(binding=1) RWStructuredBuffer<float4> pixel_state;
layout(binding=2) StructuredBuffer<uint> background_parents;
layout(binding=3) StructuredBuffer<uint> nearest_seeds;

[numthreads(8, 8, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID, uniform OutlineSize size) {
    uint2 p = dispatch_id.xy;
    if (p.x >= size.width || p.y >= size.height || size.opacity <= 0.0f) return;
    uint index = p.y * size.width + p.x;
    int2 normal = int2(1, 0);
    float edge_alpha = 1.0f;
    float coverage = 0.0f;
    if (size.line_width > 2.0f) {
        // Separable nearest-edge search: O(radius) reads in each direction,
        // rather than a square O(radius^2) neighborhood for every pixel.
        int radius = int(ceil(size.line_width * 0.5f + 0.5f));
        float best_squared = 1.0e10f;
        uint best = OUTLINE_INVALID_SEED;
        uint2 best_position = uint2(0, 0);
        for (int y = -radius; y <= radius; ++y) {
            int qy = int(p.y) + y;
            if (qy < 0 || qy >= int(size.height)) continue;
            uint candidate = nearest_seeds[uint(qy) * size.width + p.x];
            if (candidate == OUTLINE_INVALID_SEED) continue;
            uint qx = candidate - uint(qy) * size.width;
            float2 delta = float2(qx, qy) - float2(p);
            float squared = dot(delta, delta);
            if (squared < best_squared) {
                best_squared = squared;
                best = candidate;
                best_position = uint2(qx, qy);
            }
        }
        if (best == OUTLINE_INVALID_SEED) return;
        float distance_to_seed = sqrt(best_squared);
        if (distance_to_seed > size.line_width * 0.5f) return;
        // Keep occluded alpha attached to the nearest boundary seed.
        float seed_distance;
        outlineBoundary(best_position, size, outline_ids, background_parents, seed_distance, normal, edge_alpha);
        coverage = 1.0f - smoothstep(size.line_width * 0.5f, size.line_width * 0.5f + 1.0f, distance_to_seed + seed_distance);
    } else {
        float distance_to_edge;
        if (!outlineBoundary(p, size, outline_ids, background_parents, distance_to_edge, normal, edge_alpha)) return;
        coverage = 1.0f - smoothstep(size.line_width * 0.5f, size.line_width * 0.5f + 1.0f, distance_to_edge);
    }

    coverage *= edge_alpha * size.opacity * (254.0f / 255.0f);
    if (coverage > 0.0f) {
        float4 pixel = pixel_state[index];
        pixel_state[index] = float4(lerp(pixel.rgb, float3(size.red, size.green, size.blue), coverage),
                                    pixel.a * (1.0f - coverage));
    }
}
