/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include "buffer_utils.h"

namespace fast_lfs::rasterization::kernels::forward {
    template <typename Key>
    __device__ __forceinline__ Key make_instance_key(const uint tile_key, const uint depth_key) {
        if constexpr (sizeof(Key) == sizeof(InstanceKey))
            return (static_cast<Key>(tile_key) << kInstanceDepthBits) | static_cast<Key>(depth_key);
        else
            return static_cast<Key>(tile_key);
    }

    template <typename Key>
    __device__ __forceinline__ uint instance_tile(const Key key) {
        if constexpr (sizeof(Key) == sizeof(InstanceKey))
            return static_cast<uint>(key >> kInstanceDepthBits);
        else
            return static_cast<uint>(key);
    }

    // n_instances_upper sizes the grid; real count is read from d_n_instances
    template <typename Key>
    __global__ void extract_instance_ranges_cu(
        const Key* instance_keys,
        uint2* tile_instance_ranges,
        FastGSForwardStatus* __restrict__ status,
        const uint n_tiles,
        const uint n_instances_upper,
        const std::uint64_t* __restrict__ d_n_instances) {
        const uint instance_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (instance_idx >= n_instances_upper)
            return;
        const uint n_instances = d_n_instances != nullptr
                                     ? static_cast<uint>(*d_n_instances)
                                     : n_instances_upper;
        if (instance_idx >= n_instances)
            return;
        const uint instance_tile_idx = instance_tile(instance_keys[instance_idx]);
        if (instance_tile_idx >= n_tiles) {
            report_fastgs_status(
                status,
                kFastGSForwardStatusTileIndexOutOfRange,
                instance_idx,
                instance_tile_idx,
                instance_tile_idx,
                make_uint4(0, 0, 0, 0),
                n_tiles,
                0);
            return;
        }
        if (instance_idx == 0)
            tile_instance_ranges[instance_tile_idx].x = 0;
        else {
            const uint previous_instance_tile_idx = instance_tile(instance_keys[instance_idx - 1]);
            if (previous_instance_tile_idx >= n_tiles) {
                report_fastgs_status(
                    status,
                    kFastGSForwardStatusTileIndexOutOfRange,
                    instance_idx - 1,
                    previous_instance_tile_idx,
                    previous_instance_tile_idx,
                    make_uint4(0, 0, 0, 0),
                    n_tiles,
                    0);
                return;
            }
            if (instance_tile_idx != previous_instance_tile_idx) {
                tile_instance_ranges[previous_instance_tile_idx].y = instance_idx;
                tile_instance_ranges[instance_tile_idx].x = instance_idx;
            }
        }
        if (instance_idx == n_instances - 1)
            tile_instance_ranges[instance_tile_idx].y = n_instances;
    }

} // namespace fast_lfs::rasterization::kernels::forward
