/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/exportable_storage.hpp"
#include "core/splat_data.hpp"

#include <array>
#include <cstddef>
#include <expected>
#include <memory>
#include <string>

namespace lfs::core {

    // Coalesced exportable storage for the six per-primitive splat tensors. One
    // CUDA VMM allocation backs all six; each tensor is a view at a fixed offset
    // into the same physical memory. The Vulkan viewer imports this single block
    // and reads the trainer's writes directly — no per-frame copy.
    struct SplatExportableStorage {
        enum Region : std::size_t {
            Means = 0,
            Scaling = 1,
            Rotation = 2,
            Opacity = 3,
            Sh0 = 4,
            ShN = 5,
            Count = 6,
        };

        std::shared_ptr<ExportableBlock> block;
        std::array<std::size_t, Count> region_offsets{};
        std::array<std::size_t, Count> region_bytes{};

        // Build the layout, allocate the backing block, return the storage.
        // capacity = max gaussian count the trainer will reach.
        // sh_degree = SH degree the run uses; determines shN region size.
        [[nodiscard]] LFS_CORE_API static std::expected<SplatExportableStorage, std::string>
        create(std::size_t capacity, int sh_degree, int device = 0);

        // Returns a SplatTensorAllocator that hands out Tensor views into the
        // backing block. Matches on the name passed by SplatData
        // ("SplatData.means", "SplatData.scaling", "SplatData.rotation",
        //  "SplatData.opacity", "SplatData.sh0", "SplatData.shN").
        [[nodiscard]] LFS_CORE_API SplatTensorAllocator make_allocator() const;

        [[nodiscard]] bool valid() const noexcept { return static_cast<bool>(block); }
    };

} // namespace lfs::core
