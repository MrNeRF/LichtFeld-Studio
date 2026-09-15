/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cooperative_groups.h>
#include <cuda_runtime.h>

namespace lfs::rasterization::visibility {

    // Keep the block size aligned with FastGS' joint-Adam and visibility
    // bookkeeping block. EDGE uses the same stable ordering contract.
    constexpr unsigned int kBlockSize = 256u;

    __global__ inline void count_blocks(
        const unsigned int* __restrict__ visibility_mask,
        unsigned int* __restrict__ block_counts,
        const unsigned int n_primitives) {
        const unsigned int block_idx = blockIdx.x;
        const unsigned int n_blocks = (n_primitives + kBlockSize - 1u) / kBlockSize;
        if (block_idx >= n_blocks || threadIdx.x != 0)
            return;

        const unsigned int first_word = (block_idx * kBlockSize) >> 5;
        const unsigned int last_primitive = min(
            n_primitives, (block_idx + 1u) * kBlockSize);
        const unsigned int n_words = (last_primitive + 31u) / 32u - first_word;
        unsigned int count = 0;
        for (unsigned int word = 0; word < n_words; ++word)
            count += __popc(visibility_mask[first_word + word]);
        block_counts[block_idx] = count;
    }

    // Stable compaction: block_offsets supplies the original-order base and
    // the mask popcount supplies the in-block rank.
    __global__ inline void compact_indices(
        const unsigned int* __restrict__ visibility_mask,
        const unsigned int* __restrict__ block_offsets,
        unsigned int* __restrict__ visible_indices,
        unsigned int* __restrict__ primitive_work_indices,
        const unsigned int n_primitives) {
        const unsigned int primitive_idx =
            static_cast<unsigned int>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (primitive_idx >= n_primitives)
            return;

        const unsigned int word_idx = primitive_idx >> 5;
        const unsigned int bit_idx = primitive_idx & 31u;
        const unsigned int visible_word = visibility_mask[word_idx];
        if ((visible_word & (1u << bit_idx)) == 0u) {
            if (primitive_work_indices != nullptr)
                primitive_work_indices[primitive_idx] = 0xffffffffu;
            return;
        }

        const unsigned int block_idx = primitive_idx / kBlockSize;
        const unsigned int block_first_word = (block_idx * kBlockSize) >> 5;
        unsigned int rank = 0;
        for (unsigned int word = block_first_word; word < word_idx; ++word)
            rank += __popc(visibility_mask[word]);
        const unsigned int prior_bits = bit_idx == 0 ? 0u : ((1u << bit_idx) - 1u);
        rank += __popc(visible_word & prior_bits);
        const unsigned int block_base = block_idx == 0 ? 0u : block_offsets[block_idx - 1u];
        const unsigned int work_idx = block_base + rank;
        visible_indices[work_idx] = primitive_idx;
        if (primitive_work_indices != nullptr)
            primitive_work_indices[primitive_idx] = work_idx;
    }

} // namespace lfs::rasterization::visibility
