/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/exportable_storage.hpp"
#include "core/splat_data.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <string>

namespace lfs::core {

    // Coalesced exportable storage for the six per-primitive splat tensors. One
    // CUDA VMM allocation backs all six; each tensor is a view at a fixed offset
    // into the same physical memory. The Vulkan viewer imports this single block
    // and reads the trainer's writes directly — no per-frame copy.
    //
    // Capacity is the *committed* gaussian-row budget (live N + headroom), not
    // max_cap. Virtual address space can be reserved up to reserve_capacity so
    // grow() commits more physical under a stable device_ptr. Growth relocates
    // region packing (later regions move) and bumps generation so Vulkan can
    // re-import the new export handle.
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
        // capacity = committed gaussian count (live N + headroom).
        // reserve_capacity = virtual-reserve gaussian count (typically max_cap);
        //                    0 means reserve exactly `capacity`.
        // sh_degree = SH degree the run uses; determines shN region size.
        [[nodiscard]] LFS_CORE_API static std::expected<SplatExportableStorage, std::string>
        create(std::size_t capacity, int sh_degree, int device = 0,
               std::size_t reserve_capacity = 0);

        // Byte size of the packed six-region layout for `capacity` gaussians.
        [[nodiscard]] LFS_CORE_API static std::size_t layoutBytes(std::size_t capacity, int sh_degree);

        // 1.5× growth helper, clamped to max_capacity when max_capacity > 0.
        // Used for initial headroom and densification growth steps.
        [[nodiscard]] LFS_CORE_API static std::size_t growthCapacity(
            std::size_t live_or_needed, std::size_t max_capacity = 0);

        // Grow committed capacity to at least new_capacity. Relocates regions so
        // each SoA slice expands; preserves existing primitive data. device_ptr is
        // stable; export handle changes when physical size grows (Vulkan must
        // re-import). Returns true if capacity increased, false if already large
        // enough. Must not run while the GPU is using the block.
        [[nodiscard]] LFS_CORE_API std::expected<bool, std::string>
        grow(std::size_t new_capacity);

        // Returns a SplatTensorAllocator that hands out Tensor views into the
        // backing block. Matches on the name passed by SplatData
        // ("SplatData.means", "SplatData.scaling", "SplatData.rotation",
        //  "SplatData.opacity", "SplatData.sh0", "SplatData.shN").
        // Tensor capacity is clamped to this storage's committed capacity so
        // callers that still pass max_cap cannot overflow the packed regions.
        // Captures a shared control block so post-grow offset updates are visible
        // to *new* tensors from this allocator; existing tensors must be rebuilt
        // (see rebindSplatDataToStorage).
        [[nodiscard]] LFS_CORE_API SplatTensorAllocator make_allocator() const;

        // Rebuild SplatData parameter tensors as views into this storage at the
        // current capacity. When a source tensor already aliases this block
        // (same ExportableBlock VA range — including CUDA-only and Vulkan-interop
        // views), installs views WITHOUT copying. grow() has already relocated
        // every region to the new offsets; a copy_from of the stale pre-grow
        // views would overwrite correct data (ISS-025). Genuine cross-allocator
        // migrations (cuda.direct / other blocks → this storage) still copy.
        // When `allocator` is empty, uses make_allocator(); pass the Vulkan
        // interop allocator for GUI zero-copy rebind after growth.
        [[nodiscard]] LFS_CORE_API std::expected<void, std::string>
        rebindSplatData(SplatData& model, SplatTensorAllocator allocator = {}) const;

        [[nodiscard]] bool valid() const noexcept { return static_cast<bool>(block); }
        [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
        [[nodiscard]] std::size_t reservedCapacity() const noexcept { return reserved_capacity_; }
        [[nodiscard]] int shDegree() const noexcept { return sh_degree_; }
        [[nodiscard]] std::uint64_t generation() const noexcept { return generation_; }

    private:
        std::size_t capacity_ = 0;
        std::size_t reserved_capacity_ = 0;
        int sh_degree_ = 0;
        std::uint64_t generation_ = 0;

        // Shared so make_allocator captures live offsets/capacity across grow().
        struct Control {
            std::shared_ptr<ExportableBlock> block;
            std::array<std::size_t, Count> region_offsets{};
            std::array<std::size_t, Count> region_bytes{};
            std::size_t capacity = 0;
            int sh_degree = 0;
            std::uint64_t generation = 0;
        };
        std::shared_ptr<Control> control_;

        void syncControl() const;
    };

} // namespace lfs::core
