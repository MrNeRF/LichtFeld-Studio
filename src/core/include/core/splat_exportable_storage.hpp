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
#include <optional>
#include <string>

namespace lfs::core {

    // Coalesced exportable storage for the per-primitive splat tensors. One
    // CUDA VMM allocation backs every region; each tensor is a view at a fixed
    // offset into the same physical memory. The Vulkan viewer imports this
    // single block and reads the trainer's writes directly — no per-frame copy.
    //
    // ShN is pad-dropped q16 (uint16 cells, [ceil(N/32), n_cells, 32]) with
    // per-256-splat float2 bounds in ShNBounds. The training viewport dequants
    // in-shader (zero-copy). Standalone PLY/SOG viewing (no exportable block)
    // keeps the separate IEEE f16 resident path.
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
            ShNBounds = 6,
            Count = 7,
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
        //  "SplatData.opacity", "SplatData.sh0", "SplatData.shN",
        //  "SplatData.shN_value_bounds").
        // Tensor capacity is clamped to this storage's committed capacity so
        // callers that still pass max_cap cannot overflow the packed regions.
        // Captures a shared control block so post-grow offset updates are visible
        // to *new* tensors from this allocator; existing tensors must be rebuilt
        // (see rebindSplatDataToStorage).
        // ShN is always Float16 (q16 codes); ShNBounds is Float32 float2s.
        [[nodiscard]] LFS_CORE_API SplatTensorAllocator make_allocator() const;

        // Rebuild SplatData parameter tensors as views into this storage at the
        // current capacity. When a source tensor already aliases this block
        // (same ExportableBlock VA range — including CUDA-only and Vulkan-interop
        // views), installs views WITHOUT copying. grow() has already relocated
        // every region to the new offsets; a copy_from of the stale pre-grow
        // views would overwrite correct data. Genuine cross-allocator
        // migrations (cuda.direct / other blocks → this storage) still copy.
        // When `allocator` is empty, uses make_allocator(); pass the Vulkan
        // interop allocator for GUI zero-copy rebind after growth.
        [[nodiscard]] LFS_CORE_API std::expected<void, std::string>
        rebindSplatData(SplatData& model, SplatTensorAllocator allocator = {}) const;

        [[nodiscard]] bool valid() const noexcept {
            return static_cast<bool>(block) && !poisoned_;
        }
        [[nodiscard]] bool poisoned() const noexcept { return poisoned_; }
        [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
        [[nodiscard]] std::size_t reservedCapacity() const noexcept { return reserved_capacity_; }
        [[nodiscard]] int shDegree() const noexcept { return sh_degree_; }
        [[nodiscard]] std::uint64_t generation() const noexcept { return generation_; }

        // Live control block shared with every allocator lambda. Offsets/bytes/
        // generation update on grow(); consumers must resolve pointers through
        // this block (or via resolve_exportable_device_ptr) — never trust a raw
        // pointer baked into a Tensor that outlived a generation bump.
        struct Control {
            std::shared_ptr<ExportableBlock> block;
            std::array<std::size_t, Count> region_offsets{};
            std::array<std::size_t, Count> region_bytes{};
            std::size_t capacity = 0;
            std::uint64_t generation = 0;
            bool poisoned = false;

            [[nodiscard]] void* region_ptr(Region region) const {
                if (poisoned || !block || !block->device_ptr) {
                    return nullptr;
                }
                return static_cast<char*>(block->device_ptr) + region_offsets[region];
            }
        };

        [[nodiscard]] std::shared_ptr<Control> control() const noexcept { return control_; }

        // Live region base for the current generation (null if invalid).
        [[nodiscard]] LFS_CORE_API void* live_region_ptr(Region region) const;

    private:
        std::size_t capacity_ = 0;
        std::size_t reserved_capacity_ = 0;
        int sh_degree_ = 0;
        std::uint64_t generation_ = 0;
        bool poisoned_ = false;

        std::shared_ptr<Control> control_;

        void syncControl() const;
    };

    // One-shot failure seam for the destructive grow transaction. Passing a
    // region injects a failure after that region has been relocated; nullopt
    // clears the seam. Production code never enables it.
    LFS_CORE_API void set_splat_exportable_relocate_failure_for_testing(
        std::optional<std::size_t> region) noexcept;

    // Stamp live-control provenance onto an exportable view Tensor so bind sites
    // can re-resolve the device pointer after a grow without holding storage.
    LFS_CORE_API void stamp_exportable_provenance(
        Tensor& tensor,
        std::shared_ptr<SplatExportableStorage::Control> control,
        SplatExportableStorage::Region region);

    // Resolve the device pointer for an exportable-backed Tensor through the
    // live control block. On generation mismatch: logs + returns the live
    // pointer (release) / throws (debug asserts). Non-exportable tensors fall
    // through to data_ptr().
    [[nodiscard]] LFS_CORE_API void* resolve_exportable_device_ptr(const Tensor& tensor);

    // q16 codes + bounds pair fetched through live control when exportable-
    // tracked; null bounds when not q16. Generation mismatch fails loud.
    struct Q16BindPtrs {
        const float* codes = nullptr;  // u16 bit-patterns as float*
        const float* bounds = nullptr; // float2 as float*
        unsigned n_cells_per_prim = 0;
        std::uint64_t generation = 0;
        bool generation_checked = false;
    };
    [[nodiscard]] LFS_CORE_API Q16BindPtrs resolve_q16_bind_ptrs(const SplatData& model);

} // namespace lfs::core
