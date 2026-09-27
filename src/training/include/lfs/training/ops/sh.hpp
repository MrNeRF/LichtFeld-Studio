/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "lfs/training/ops/types.hpp"

#include <cstddef>
#include <cstdint>

namespace lfs::gpu_ops {

    struct ShRowsParams {
        size_t source_rows = 0;
        size_t count = 0;
        size_t destination_offset = 0;
        uint32_t source_rest = 0;
        uint32_t destination_rest = 0;
    };

    struct ShRangeParams {
        uint64_t canonical_float_offset = 0;
        uint64_t float_count = 0;
        size_t primitives = 0;
        uint32_t destination_rest = 0;
        uint32_t layout_rest = 0;
        ShStorage storage = ShStorage::Float32;
    };

    struct Q16TouchParams {
        size_t sorted_count = 0;
        size_t primitives = 0;
        size_t decode_source_rows = 0;
        uint32_t rest = 0;
    };

    struct ShOps {
        State (*create_run_scratch)();

        // code_offset / bounds_offset are elements past the exportable region
        // base. Chunk encode writes one quant block into the middle of storage.
        void (*encode_q16)(
            In swizzled, Out codes, Out bounds, size_t primitives, uint32_t rest,
            size_t code_offset, size_t bounds_offset);
        void (*decode_q16)(
            In codes, In bounds, Out swizzled, size_t primitives, uint32_t rest);

        void (*block_ids)(In destination_indices, Out block_ids);
        void (*block_runs)(
            BackendState&, In sorted_block_ids,
            Out unique_blocks, Out offsets, Out run_count);

        // canonical_order, when valid, maps sorted row i to canonical row
        // canonical_order[i]; otherwise canonical is already in sorted order.
        void (*reencode_touched)(
            Out codes, Out bounds, In canonical, In destinations,
            In unique_blocks, In offsets, In run_count, In canonical_order,
            const Q16TouchParams&);

        void (*decode_range)(
            In values, In bounds, Out canonical, const ShRangeParams&);

        void (*zero_rows)(Out values, In indices, uint32_t rest);
        void (*gather_swizzled)(In source, In indices, Out destination, const ShRowsParams&);
        void (*gather_canonical)(In source, In indices, Out canonical, const ShRowsParams&);
        void (*append_canonical)(In canonical, Out destination, const ShRowsParams&);
        void (*scatter_canonical)(In canonical, In indices, Out destination, const ShRowsParams&);

        void (*fill_bytes)(Out storage, size_t byte_count, uint8_t value);
    };

} // namespace lfs::gpu_ops
