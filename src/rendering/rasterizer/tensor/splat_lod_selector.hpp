/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/error.hpp"
#include "core/tensor.hpp"

#include <array>
#include <cstdint>
#include <memory>

namespace lfs::rendering {
    // Uniforms of splat_lod.slang, field for field.
    struct alignas(16) SplatLodParameters {
        uint32_t node_count = 0, output_capacity = 0, chunk_splats = 0, invalid_page = 0xffffffffu;
        float pixel_scale_limit = 0, object_scale = 1, behind_camera_penalty = .2f, cone_foveation = .4f;
        float cone_dot0 = 0, cone_dot = 0, cone_blend_denominator = 0, cone_tail_valid = 0;
        std::array<float, 4> view_row0{}, view_row1{}, view_row2{};
        float outside_view_foveation = .05f, viewport_half_tan_x = 0, viewport_half_tan_y = 0, ortho_half_width = 0;
        float ortho_half_height = 0;
        uint32_t viewport_foveation = 1, orthographic = 0, physical_node_count = 0;
        uint32_t logical_chunk_count = 0, current_frame = 0, fade_frames = 0, budget_pass = 0;
    };
    static_assert(sizeof(SplatLodParameters) == 144);

    // Resident quantized LOD sidecars, read in place as byte tensors.
    struct SplatLodTree {
        const core::Tensor *bounds = nullptr, *links = nullptr, *chunk_to_page = nullptr;
        const core::Tensor *page_age = nullptr, *page_frames = nullptr, *page_to_chunk = nullptr;
    };

    class SplatLodSelector {
    public:
        explicit SplatLodSelector(core::GpuBackend backend);
        ~SplatLodSelector();
        SplatLodSelector(const SplatLodSelector&) = delete;
        SplatLodSelector& operator=(const SplatLodSelector&) = delete;

        // Reserves a cut for `capacity` selected nodes from `source_count`
        // physical nodes and `chunks` logical chunks. Grows only.
        [[nodiscard]] lfs::Result<void> reserve(uint32_t capacity, uint32_t source_count, uint32_t chunks);
        // The cut's output capacity and logical chunks must fit the reservation.
        [[nodiscard]] lfs::Result<void> select(const SplatLodTree& tree, const SplatLodParameters& parameters);

        // UInt32[8]: selected, overflow, threshold-multiplier bits, and the
        // indirect retry dispatch in elements [3..5].
        [[nodiscard]] const core::Tensor& counts() const;
        [[nodiscard]] const core::Tensor& indices() const;
        [[nodiscard]] const core::Tensor& logical_indices() const;
        [[nodiscard]] const core::Tensor& weights() const;
        [[nodiscard]] const core::Tensor& levels() const;
        [[nodiscard]] const core::Tensor& touches() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace lfs::rendering
