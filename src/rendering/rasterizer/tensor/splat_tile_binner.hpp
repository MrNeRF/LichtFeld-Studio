/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/error.hpp"
#include "core/gpu_kernel_module.hpp"
#include "core/tensor.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace lfs::rendering {
    // Bins projected splats into 16x16 pixel tiles and sorts every (tile,
    // depth) instance on the GPU through tensor programs, with indirect
    // dispatches sized by the GPU-computed instance count: the host never
    // waits. Layouts and results match the native Metal viewer rasterizer.
    class SplatTileBinner {
    public:
        // Byte sizes of the shared GPU structs (splat_types.slang).
        static constexpr size_t kProjectedSplatBytes = 64;
        static constexpr size_t kRasterParametersBytes = 144;
        static constexpr size_t kRasterStatusBytes = 24;

        explicit SplatTileBinner(core::GpuBackend backend);
        ~SplatTileBinner();
        SplatTileBinner(const SplatTileBinner&) = delete;
        SplatTileBinner& operator=(const SplatTileBinner&) = delete;

        // Reserves scratch for `splats` sources, `tiles` tiles and `capacity`
        // instances. Grows only.
        [[nodiscard]] lfs::Result<void> reserve(uint32_t splats, uint32_t tiles, uint32_t capacity);

        // `splats` holds `count` ProjectedSplat records; `raster` one
        // RasterParameters whose count, tiles, columns and capacity match.
        // Overflowing the capacity sets status.error and bins nothing.
        // `source_sorted` (raster flag 256) sorts the visible sources by depth
        // first and then only the tile bits of each instance: cheaper when
        // instances outnumber sources. Requires count <= capacity.
        [[nodiscard]] lfs::Result<void> bin(const core::Tensor& splats, const core::Tensor& raster,
                                            uint32_t count, uint32_t tiles, bool source_sorted = false);

        // Writes the depth-batch job of every chunk slot of the last bin() into
        // `jobs` (Int32 pairs); slots of tiles below the split stay -1.
        [[nodiscard]] lfs::Result<void> depth_batches(const core::Tensor& raster, uint32_t tiles, core::Tensor& jobs);

        // Diagnostics: called after each binning stage is recorded ("source",
        // "counts", "instances", "sort"), e.g. to place GPU timestamps.
        using StageMarker = std::function<void(const char* stage)>;
        void set_stage_marker(StageMarker marker);

        // Results of the last bin(), valid on the tensor timeline.
        [[nodiscard]] const core::Tensor& status() const;  // RasterStatus bytes
        [[nodiscard]] const core::Tensor& keys() const;    // Int64 (tile << 32 | depth bits); uint32 tiles when source-sorted
        [[nodiscard]] const core::Tensor& indices() const; // UInt32 source per sorted key
        [[nodiscard]] const core::Tensor& ranges() const;  // UInt32 [tiles][begin, end)
        // UInt32[21]: sort and range dispatches; [18..20] the depth-batch blend.
        [[nodiscard]] const core::Tensor& dispatch_args() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace lfs::rendering
