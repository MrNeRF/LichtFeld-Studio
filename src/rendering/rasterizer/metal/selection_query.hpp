/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "splat_preprocessor.hpp"

namespace lfs::rendering::metal {
    enum class SelectionShape : uint32_t { Brush,
                                           Rectangle,
                                           Polygon,
                                           Ring };
    struct alignas(16) SelectionParameters {
        simd_float4x4 world_to_camera = matrix_identity_float4x4;
        simd_float4 intrinsics{};
        simd_uint4 image{};   // width, height, camera model, GUT
        simd_uint4 source{};  // count, shape, primitive count, polygon vertices
        simd_uint4 scene{};   // transforms, indexed, visibility count, deleted count
        simd_uint4 payload{}; // half geometry, mip, ring phase, reserved
        simd_uint4 aabb{};
        simd_float4 ring{.01f, kViewerNearClip, 0, 0};
    };
    static_assert(sizeof(SelectionParameters) == 176);
    struct SelectionBuffers {
        BufferSlice means, log_scales, rotations, opacity, deleted;
        BufferSlice transforms, transform_indices, visibility;
        BufferSlice primitives, polygon_vertices, polygon_mask, output, ring_pick;
    };
    class SelectionQuery {
    public:
        explicit SelectionQuery(id<MTLDevice>);
        ~SelectionQuery();
        // No host readback. The output is one Boolean byte per source primitive;
        // ring picking is deterministic by positive depth and then source ID.
        void encode(id<MTLCommandBuffer>, const SelectionBuffers&, SelectionParameters);

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace lfs::rendering::metal
