// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
// A single selected scene node without selection filters or raster overlays:
// rendered transmittance is already the selected coverage, with no occluder.
struct OutlineSize { uint width; uint height; float threshold; };
layout(binding=0) StructuredBuffer<float4> pixel_state;
layout(binding=1) RWStructuredBuffer<uint> outline_ids;
[numthreads(8, 8, 1)]
void main(uint3 pixel : SV_DispatchThreadID, uniform OutlineSize size) {
    if (pixel.x >= size.width || pixel.y >= size.height) return;
    uint index = pixel.y * size.width + pixel.x;
    outline_ids[index] = pixel_state[index].a <= (1.0f - size.threshold) ? 2u : 0u;
}
