/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/error.hpp"
#include "core/tensor.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace lfs::rendering {
    // SelectionParameters of the native Metal selection query, byte for byte;
    // matrices are column-major.
    struct alignas(16) SplatSelectionParameters {
        std::array<float, 16> world_to_camera{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
        std::array<float, 4> intrinsics{};
        std::array<uint32_t, 4> image{}, source{}, scene{}, payload{}, aabb{};
        std::array<float, 4> ring{.01f, .01f, 0, 0};
    };
    static_assert(sizeof(SplatSelectionParameters) == 176);

    struct SplatSelectionInputs {
        const core::Tensor *means = nullptr, *log_scales = nullptr, *rotations = nullptr, *opacity = nullptr;
        const core::Tensor *deleted = nullptr, *transform_indices = nullptr;
        std::span<const std::byte> transforms;       // column-major float matrices, 64 bytes each
        std::span<const std::byte> visibility;       // one byte per visible scene entry
        std::span<const std::byte> primitives;       // float4 records
        std::span<const std::byte> polygon_vertices; // float2 records
    };

    class SplatSelectionQuery {
    public:
        explicit SplatSelectionQuery(core::GpuBackend backend);
        ~SplatSelectionQuery();
        SplatSelectionQuery(const SplatSelectionQuery&) = delete;
        SplatSelectionQuery& operator=(const SplatSelectionQuery&) = delete;

        // No host wait or readback. Output holds one byte per source; ring_pick
        // is two UInt32 words and is required only by Ring queries.
        [[nodiscard]] lfs::Result<void> query(const SplatSelectionInputs& inputs,
                                              const SplatSelectionParameters& parameters,
                                              core::Tensor& output,
                                              core::Tensor* ring_pick = nullptr);

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace lfs::rendering
