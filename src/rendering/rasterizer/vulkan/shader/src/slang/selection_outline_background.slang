// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef OUTLINE_BACKGROUND_INIT
#define OUTLINE_BACKGROUND_INIT 0
#endif

// Four-connected background. Boundary pixels share sentinel root 0. The
// strictly decreasing parent labels prevent cycles; concurrent unions retry
// stale roots with compare/exchange, so no fixed iteration limit is needed.
struct OutlineSize { uint width; uint height; };
layout(binding=0) StructuredBuffer<uint> outline_ids;
layout(binding=1) globallycoherent RWStructuredBuffer<uint> parents;

bool background(uint index) {
    uint value = outline_ids[index];
    return (value & 0x3fffffffu) <= 1u || (value & 0x40000000u) != 0u;
}
uint root(uint label) {
    for (;;) {
        if (label == 0u) return 0u;
        uint parent;
        InterlockedAdd(parents[label], 0u, parent);
        if (parent == label) return label;
        // Path halving keeps long, narrow background corridors from producing
        // a linear parent walk for every subsequent pixel. All updates lower
        // labels, including when another invocation links this root meanwhile.
        uint grandparent = 0u;
        if (parent != 0u) InterlockedAdd(parents[parent], 0u, grandparent);
        InterlockedMin(parents[label], grandparent);
        label = grandparent;
    }
}
void unite(uint a, uint b) {
    for (;;) {
        a = root(a);
        b = root(b);
        if (a == b) return;
        uint high = max(a, b);
        uint low = min(a, b);
        uint previous;
        InterlockedCompareExchange(parents[high], high, low, previous);
        if (previous == high) return;
    }
}
[numthreads(8, 8, 1)]
void main(uint3 pixel : SV_DispatchThreadID, uniform OutlineSize size) {
    if (pixel.x >= size.width || pixel.y >= size.height) return;
    uint index = pixel.y * size.width + pixel.x;
#if OUTLINE_BACKGROUND_INIT
    if (index == 0u) parents[0] = 0u;
    bool boundary = pixel.x == 0u || pixel.y == 0u || pixel.x + 1u == size.width || pixel.y + 1u == size.height;
    parents[index + 1u] = background(index) && boundary ? 0u : index + 1u;
#else
    if (!background(index)) return;
    if (pixel.x > 0u && background(index - 1u)) unite(index + 1u, index);
    if (pixel.y > 0u && background(index - size.width)) unite(index + 1u, index - size.width + 1u);
#endif
}
