/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "splat_preprocessor.hpp"
#include <memory>

namespace lfs::rendering::metal {
    struct alignas(16) LodParameters {
        uint32_t node_count = 0, output_capacity = 0, chunk_splats = 65536, invalid_page = 0xffffffffu;
        float pixel_scale_limit = 0, object_scale = 1, behind_camera_penalty = .2f, cone_foveation = .4f;
        float cone_dot0 = 0, cone_dot = 0, cone_blend_denominator = 0, cone_tail_valid = 0;
        simd_float4 view_row0{}, view_row1{}, view_row2{};
        float outside_view_foveation = .05f, viewport_half_tan_x = 0, viewport_half_tan_y = 0, ortho_half_width = 0;
        float ortho_half_height = 0;
        uint32_t viewport_foveation = 1, orthographic = 0, physical_node_count = 0;
        uint32_t logical_chunk_count = 0, current_frame = 0, fade_frames = 0, budget_pass = 0;
    };
    static_assert(sizeof(LodParameters) == 144);
    struct LodTreeBuffers {
        BufferSlice bounds, links, chunk_to_page, page_age, page_frames, page_to_chunk;
    };
    struct LodCutStatus {
        uint32_t selected = 0, overflow = 0;
        float threshold_multiplier = 1;
    };
    class LodCutFrame {
    public:
        LodCutFrame(id<MTLDevice> device, uint32_t capacity, uint32_t source_count, uint32_t chunks);
        ~LodCutFrame();
        LodCutFrame(const LodCutFrame&) = delete;
        LodCutFrame& operator=(const LodCutFrame&) = delete;
        LodSelection selection(bool debug = false) const;
        bool busy() const;
        // Read only after successful completion; an abandoned reservation must be discarded.
        LodCutStatus status() const;
        BufferSlice touches() const;

    private:
        struct Impl;
        std::shared_ptr<Impl> impl_;
        friend class LodSelector;
    };
    class LodSelector {
    public:
        explicit LodSelector(id<MTLDevice> device);
        ~LodSelector();
        void encode(id<MTLCommandBuffer> command, const LodTreeBuffers& tree,
                    const LodParameters& parameters, LodCutFrame& frame);

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace lfs::rendering::metal
