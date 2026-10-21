/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <cstdlib>
#include <cstdio>
#include <string_view>

// Depth-ordered emission lets the stable instance sort visit only tile bits.
// Without that prerequisite all original depth and tile bits are sorted.
template<class Key, class Value, class Count>
hipError_t lfs_fast_sort_pairs(void* temporary, size_t& bytes,
                              hipcub::DoubleBuffer<Key>& keys,
                              hipcub::DoubleBuffer<Value>& values,
                              Count count, int begin_bit, int end_bit,
                              hipStream_t stream, bool depth_ordered = false) {
    if (depth_ordered && end_bit <= 32) {
        if (!temporary) bytes = 1; // Nonempty query contract, no work for one tile.
        return hipSuccess;
    }
    return lfs_fast_full_sort_pairs(temporary, bytes, keys, values, count,
                                   depth_ordered ? 32 : begin_bit, end_bit, stream);
}
