/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "training_api.h"
#include <algorithm>
#include <array>
#include <core/splat_exportable_storage.hpp>
#include <stdexcept>

namespace lfs::training::hip {
    // Convert the application's live storage control, never a duplicate layout
    // formula. This does not export a handle or assert that arbitrary CUDA VMM
    // chunks can be imported as one dedicated Vulkan allocation. The caller must
    // supply the actual matching Vulkan allocation and retain its ownership.
    // Higher SH requires an explicit FP32 view at the TensorModel boundary.
    inline LfsHipTrainModelLayout splat_storage_layout(
        const core::SplatExportableStorage::Control& storage, size_t live_count, bool allow_fp32_sh = false) {
        using Storage = core::SplatExportableStorage;
        if (storage.poisoned || !storage.block || !storage.block->device_ptr ||
            !live_count || live_count > 10000000 || live_count > storage.capacity ||
            std::any_of(storage.block->chunks.begin(), storage.block->chunks.end(),
                        [](const auto& chunk) { return chunk.handle.valid(); }) ||
            (storage.region_bytes[Storage::ShN] && !allow_fp32_sh) ||
            storage.region_bytes[Storage::ShNBounds])
            throw std::invalid_argument("HIP requires a healthy fixed SH0 contiguous storage");
        constexpr std::array<size_t, 5> regions{Storage::Means, Storage::Rotation, Storage::Scaling, Storage::Sh0, Storage::Opacity};
        constexpr std::array<size_t, 5> attributes{3, 4, 3, 3, 1};
        LfsHipTrainModelLayout result{};
        result.struct_bytes = sizeof(result);
        result.required_bytes = storage.block->reserved_bytes;
        for (size_t family = 0; family < regions.size(); ++family) {
            const size_t region = regions[family];
            const uint64_t offset = storage.region_offsets[region];
            const uint64_t bytes = storage.region_bytes[region];
            const uint64_t live_bytes = live_count * attributes[family] * sizeof(float);
            if (offset % 256 || offset > result.required_bytes || bytes > result.required_bytes - offset ||
                live_bytes > bytes)
                throw std::invalid_argument("HIP splat storage region is too small, misaligned or out of range");
            for (size_t previous = 0; previous < family; ++previous) {
                const auto previous_region = regions[previous];
                const uint64_t previous_offset = storage.region_offsets[previous_region];
                const uint64_t previous_end = previous_offset + storage.region_bytes[previous_region];
                if (offset < previous_end && offset + bytes > previous_offset)
                    throw std::invalid_argument("HIP splat storage reserved regions overlap");
            }
            result.offsets_bytes[family] = offset;
            result.float_counts[family] = live_count * attributes[family];
        }
        if (allow_fp32_sh) {
            const uint64_t offset = storage.region_offsets[Storage::ShN];
            const uint64_t bytes = storage.region_bytes[Storage::ShN];
            if (!bytes || offset % 256 || offset > result.required_bytes || bytes > result.required_bytes - offset)
                throw std::invalid_argument("HIP FP32 SH region is empty, misaligned or out of range");
            for (const auto region : regions) {
                const auto begin = storage.region_offsets[region];
                if (offset < begin + storage.region_bytes[region] && offset + bytes > begin)
                    throw std::invalid_argument("HIP FP32 SH reserved region overlaps base parameters");
            }
        }
        return result;
    }
} // namespace lfs::training::hip
