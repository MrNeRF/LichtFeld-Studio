/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/tensor.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace lfs::training {
    struct PairSortBuffers {
        core::Tensor* keys_a;
        core::Tensor* keys_b;
        core::Tensor* values_a;
        core::Tensor* values_b;
    };

    struct PairSortPassTimings {
        float histogram_ms = 0.0f;
        float partition_scan_ms = 0.0f;
        float histogram_reduce_ms = 0.0f;
        float digit_base_scan_ms = 0.0f;
        float scatter_ms = 0.0f;
    };

    // Returns true when the sorted pairs occupy the A buffers. Keys are packed
    // as uint32 or uint64; values are uint32 and equal keys retain input order.
    [[nodiscard]] bool vulkan_pair_sort(PairSortBuffers buffers, uint32_t element_count,
                                        uint32_t begin_bit, uint32_t end_bit,
                                        bool wide_keys,
                                        std::vector<PairSortPassTimings>* pass_timings = nullptr);
} // namespace lfs::training
