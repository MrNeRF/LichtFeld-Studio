/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// One-thread compute pass that reads the cumsum tail (num_indices) from
// index_buffer_offset[num_splats - 1] and writes a VkDispatchIndirectCommand
// for the subsequent compute_tile_ranges pass. Replaces the synchronous
// CPU readback that previously stalled the queue every frame.

#include "config.slang"

layout(binding = 0) StructuredBuffer<int32_t> index_buffer_offset;
layout(binding = 1) RWStructuredBuffer<uint32_t> dispatch_indirect_out;

struct Uniforms {
    uint32_t num_splats;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uniform Uniforms uniforms) {
    uint32_t num_indices =
        uniforms.num_splats == 0u
            ? 0u
            : (uint32_t)index_buffer_offset[uniforms.num_splats - 1u];

    // compute_tile_ranges thread layout is [numthreads(256, 1, 1)] dispatched
    // over (num_isects + 1) threads.
    uint32_t group_count_x = (num_indices + 1u + 255u) / 256u;

    dispatch_indirect_out[0] = group_count_x;
    dispatch_indirect_out[1] = 1u;
    dispatch_indirect_out[2] = 1u;
}
