// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

struct OutlineSize {
    uint width;
    uint height;
};
layout(binding=0) StructuredBuffer<uint> outline_ids;
layout(binding=1) RWStructuredBuffer<float4> pixel_state;
layout(binding=2) StructuredBuffer<float4> selection_colors;

[numthreads(8, 8, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID, uniform OutlineSize size)
{
    uint2 p = dispatch_id.xy;
    if (p.x >= size.width || p.y >= size.height)
        return;
    uint index = p.y * size.width + p.x;
    uint center_sample = outline_ids[index];
    uint center = center_sample & 0x7fffffffu;
    // The mask contains selected objects only. The high bit records scene
    // occlusion without cutting holes in their silhouettes.
    center = center > 1u ? center : 0u;
    float edge_alpha = (center_sample & 0x80000000u) != 0u ? 0.35f : 1.0f;
    float distance_to_edge = 3.0f;
    // Only the 3x3 neighborhood can contribute to this two-sided 2px line.
    // Samples at distance >= 2 have zero coverage.
    [unroll]
    for (int y = -1; y <= 1; ++y) {
        [unroll]
        for (int x = -1; x <= 1; ++x) {
            if (x == 0 && y == 0)
                continue;
            int2 q = int2(p) + int2(x, y);
            if (q.x < 0 || q.y < 0 || q.x >= int(size.width) || q.y >= int(size.height))
                continue; // Clipping at the viewport edge is not an object contour.
            uint neighbor_sample = outline_ids[uint(q.y) * size.width + uint(q.x)];
            uint neighbor = neighbor_sample & 0x7fffffffu;
            neighbor = neighbor > 1u ? neighbor : 0u;
            if (neighbor != center && (neighbor != 0u || center != 0u)) {
                float distance = length(float2(x, y));
                if (distance < distance_to_edge) {
                    distance_to_edge = distance;
                    if (center == 0u)
                        edge_alpha = (neighbor_sample & 0x80000000u) != 0u ? 0.35f : 1.0f;
                }
            }
        }
    }
    float coverage = (1.0f - smoothstep(1.0f, 2.0f, distance_to_edge)) * edge_alpha * (254.0f / 255.0f);
    if (coverage > 0.0f) {
        // The raster buffer contains premultiplied RGB and transmittance.
        // Match Blender's normal-view occluded alpha, without changing depth.
        float4 pixel = pixel_state[index];
        pixel_state[index] = float4(lerp(pixel.rgb, selection_colors[0].rgb, coverage),
                                    pixel.a * (1.0f - coverage));
    }
}
