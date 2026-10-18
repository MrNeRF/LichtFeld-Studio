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
    uint center = outline_ids[index];
    // IDs 0/1 both represent non-selected coverage. Keep the selection's
    // exterior solid and distinguish visible object boundaries with dashes.
    center = center > 1u ? center : 0u;
    float exterior_distance = 3.0f;
    float interior_distance = 3.0f;
    float2 interior_normal = float2(0.0f);
    float2 nearest_interior_offset = float2(0.0f);
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
            uint neighbor = outline_ids[uint(q.y) * size.width + uint(q.x)];
            neighbor = neighbor > 1u ? neighbor : 0u;
            float2 offset = float2(x, y);
            float distance = length(offset);
            if ((neighbor == 0u) != (center == 0u)) {
                exterior_distance = min(exterior_distance, distance);
            } else if (center != 0u && neighbor != center) {
                interior_normal += offset;
                if (distance < interior_distance) {
                    interior_distance = distance;
                    nearest_interior_offset = offset;
                }
            }
        }
    }
    float coverage = 1.0f - smoothstep(1.0f, 2.0f, exterior_distance);
    if (interior_distance < 2.0f) {
        // Advance along the dominant tangent, so horizontal, vertical and
        // diagonal boundaries all have gaps. Absolute normals make the two
        // sides of a boundary use the same axis, independent of object IDs.
        if (dot(interior_normal, interior_normal) == 0.0f)
            interior_normal = nearest_interior_offset;
        float coordinate = abs(interior_normal.x) >= abs(interior_normal.y) ? float(p.y) : float(p.x);
        // Six-pixel dashes and gaps, with one-pixel antialiased end caps.
        float phase = frac((coordinate + 0.5f) / 12.0f) * 12.0f;
        float dash = smoothstep(0.0f, 1.0f, phase) * (1.0f - smoothstep(5.0f, 6.0f, phase));
        float interior_coverage = (1.0f - smoothstep(1.0f, 2.0f, interior_distance)) * dash;
        coverage = max(coverage, interior_coverage);
    }
    if (coverage > 0.0f) {
        // The raster buffer contains premultiplied RGB and transmittance.
        // Blend an opaque UI line without touching the scene/depth buffers.
        float4 pixel = pixel_state[index];
        pixel_state[index] = float4(lerp(pixel.rgb, selection_colors[0].rgb, coverage),
                                    pixel.a * (1.0f - coverage));
    }
}
