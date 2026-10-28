// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

#include "selection_outline_common.slang"
layout(binding=0) StructuredBuffer<uint> edge_seeds;
layout(binding=1) RWStructuredBuffer<uint> nearest_seeds;
[numthreads(8, 8, 1)]
void main(uint3 pixel : SV_DispatchThreadID, uniform OutlineSize size) {
    if (pixel.x >= size.width || pixel.y >= size.height) return;
    uint index = pixel.y * size.width + pixel.x;
    int radius = int(ceil(size.line_width * 0.5f + 0.5f));
    int best_distance = radius + 1;
    uint best = OUTLINE_INVALID_SEED;
    for (int x = -radius; x <= radius; ++x) {
        int qx = int(pixel.x) + x;
        if (qx < 0 || qx >= int(size.width)) continue;
        uint candidate = edge_seeds[pixel.y * size.width + uint(qx)];
        if (candidate != OUTLINE_INVALID_SEED && abs(x) < best_distance) {
            best = candidate;
            best_distance = abs(x);
        }
    }
    nearest_seeds[index] = best;
}
