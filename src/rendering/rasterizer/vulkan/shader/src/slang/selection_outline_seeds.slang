// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

#include "selection_outline_common.slang"
layout(binding=0) StructuredBuffer<uint> outline_ids;
layout(binding=1) RWStructuredBuffer<uint> edge_seeds;
layout(binding=2) StructuredBuffer<uint> background_parents;
[numthreads(8, 8, 1)]
void main(uint3 pixel : SV_DispatchThreadID, uniform OutlineSize size) {
    if (pixel.x >= size.width || pixel.y >= size.height) return;
    uint index = pixel.y * size.width + pixel.x;
    float distance, alpha;
    int2 normal;
    edge_seeds[index] = outlineBoundary(pixel.xy, size, outline_ids, background_parents, distance, normal, alpha)
                           ? index : OUTLINE_INVALID_SEED;
}
