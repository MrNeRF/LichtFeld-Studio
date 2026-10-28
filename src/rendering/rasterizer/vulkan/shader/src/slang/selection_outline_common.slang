// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

struct OutlineSize {
    uint width;
    uint height;
    float red;
    float green;
    float blue;
    float opacity;
    float line_width;
    float threshold;
    uint holes;
};
static const uint OUTLINE_INVALID_SEED = 0xffffffffu;
uint outlineNode(uint sample) {
    uint node = sample & 0x3fffffffu;
    return node > 1u && (sample & 0x40000000u) == 0u ? node : 0u;
}
bool outlineExterior(uint index, StructuredBuffer<uint> parents) {
    uint label = index + 1u;
    for (;;) {
        uint parent = parents[label];
        if (parent == label) return label == 0u;
        label = parent;
    }
}

// The original 3x3 test is shared by thin lines and thick-line seed creation.
// Clipping at the viewport edge is not an object contour.
bool outlineBoundary(uint2 p, OutlineSize size, StructuredBuffer<uint> ids,
                     StructuredBuffer<uint> parents, out float distance_to_edge,
                     out int2 normal, out float edge_alpha) {
    uint index = p.y * size.width + p.x;
    uint sample = ids[index];
    uint center = outlineNode(sample);
    normal = int2(1, 0);
    edge_alpha = (sample & 0x80000000u) != 0u ? 0.35f : 1.0f;
    distance_to_edge = 3.0f;
    if (size.holes == 0u && center == 0u && !outlineExterior(index, parents)) return false;
    [unroll]
    for (int y = -1; y <= 1; ++y) {
        [unroll]
        for (int x = -1; x <= 1; ++x) {
            if (x == 0 && y == 0) continue;
            int2 q = int2(p) + int2(x, y);
            if (q.x < 0 || q.y < 0 || q.x >= int(size.width) || q.y >= int(size.height)) continue;
            uint neighbor_index = uint(q.y) * size.width + uint(q.x);
            uint neighbor_sample = ids[neighbor_index];
            uint neighbor = outlineNode(neighbor_sample);
            if (size.holes == 0u && neighbor == 0u && !outlineExterior(neighbor_index, parents)) continue;
            if (neighbor != center && (neighbor != 0u || center != 0u)) {
                float distance = length(float2(x, y));
                if (distance < distance_to_edge) {
                    distance_to_edge = distance;
                    normal = int2(x, y);
                    if (center == 0u) edge_alpha = (neighbor_sample & 0x80000000u) != 0u ? 0.35f : 1.0f;
                }
            }
        }
    }
    return distance_to_edge < 3.0f;
}
