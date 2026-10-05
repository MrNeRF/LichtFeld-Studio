/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/error.hpp"
#include "core/tensor.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <memory>

namespace lfs::rendering {
    // RasterParameters of splat_types.slang, field for field.
    struct SplatRasterParameters {
        uint32_t count = 0, width = 0, height = 0, columns = 0;
        uint32_t tiles = 0, capacity = 0, mode = 0, flags = 0; // flags: blend feature bits (Slang `unused`)
        std::array<float, 4> background{}, render_origin{}, intrinsics{}, clip{};
        std::array<uint32_t, 4> camera{}; // width, height, camera model, mip
        std::array<float, 4> panorama{};
        std::array<uint32_t, 4> mask_limits{}; // selection count, preview count, depth-batch instances
    };
    static_assert(sizeof(SplatRasterParameters) == 144);

    // PresentParameters of splat_present.slang.
    struct SplatPresentParameters {
        float exposure = 1;
        uint32_t tone = 0, transparent = 0, has_previous = 0;
        float depth_min = 0, depth_max = 0;
        uint32_t depth_view = 0, depth_mode = 0;
        std::array<float, 4> background{};
        std::array<uint32_t, 4> capture{};
        std::array<uint32_t, 4> extent{}; // set by present(): width, height, previous width, previous height
    };
    static_assert(sizeof(SplatPresentParameters) == 80);

    enum class SplatRasterMode : uint32_t { Gaussian,
                                            Points,
                                            Discs,
                                            Gut };

    // The single-source splat rasterizer: tile binning, sorting, blending and
    // presentation of projected splats as tensor programs on Metal and Vulkan.
    // Every pass is ordered on the tensor timeline; the host never waits.
    class SplatRasterizer {
    public:
        explicit SplatRasterizer(core::GpuBackend backend);
        ~SplatRasterizer();
        SplatRasterizer(const SplatRasterizer&) = delete;
        SplatRasterizer& operator=(const SplatRasterizer&) = delete;

        // Grows the scratch for `splats` sources, a width x height frame and
        // `capacity` tile instances.
        [[nodiscard]] lfs::Result<void> reserve(uint32_t splats, uint32_t width, uint32_t height, uint32_t capacity);

        // Blends `count` ProjectedSplat records (and 3DGUT geometry) into
        // color(), depth() and pick(). An instance overflow sets status().error.
        [[nodiscard]] lfs::Result<void> rasterize(const core::Tensor& projected, const core::Tensor* gut, uint32_t count,
                                             SplatRasterMode mode, const SplatRasterParameters& parameters);

        // Writes the display image (packed RGBA8) and linear view depth
        // (Float32) of the last rasterize() into rgba() and linear_depth(). An
        // overflowing frame keeps the previous image of the same extent; the
        // rasterizer sets has_previous and extent.
        [[nodiscard]] lfs::Result<void> present(const SplatPresentParameters& parameters);

        // Diagnostics: called after each recorded stage of rasterize() and
        // present() (binning stages, "ranges", "jobs", "prefix", "chunks",
        // "compose" or "blend", "present"), e.g. to place GPU timestamps.
        void set_stage_marker(std::function<void(const char* stage)> marker);

        // Byte views carved from one per-extent allocation.
        [[nodiscard]] const core::Tensor& status() const;       // RasterStatus bytes
        [[nodiscard]] const core::Tensor& color() const;        // Float16 [H,W,4], premultiplied
        [[nodiscard]] const core::Tensor& depth() const;        // Float32 [H,W,4]
        [[nodiscard]] const core::Tensor& pick() const;         // UInt32 [H,W]
        [[nodiscard]] const core::Tensor& rgba() const;         // RGBA8 [H,W]
        [[nodiscard]] const core::Tensor& linear_depth() const; // Float32 [H,W]

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace lfs::rendering
