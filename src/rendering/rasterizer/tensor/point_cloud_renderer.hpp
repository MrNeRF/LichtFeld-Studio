/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/error.hpp"
#include "core/gpu_backend_fwd.hpp"
#include "core/tensor.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <type_traits>

namespace lfs::rendering {
    // PointParameters of point_cloud.slang, field for field. Matrices are
    // column-major, matching glm and the desktop point renderers.
    struct alignas(16) PointParameters {
        std::array<float, 16> view_projection{}, view{}, crop_to_local{};
        std::array<float, 4> crop_min{}, crop_max{}, voxel_focal_ortho{};
        std::array<uint32_t, 4> counts{};
    };
    static_assert(sizeof(PointParameters) == 256);
    static_assert(std::is_standard_layout_v<PointParameters>);
    static_assert(std::is_trivially_copyable_v<PointParameters>);

    struct SplatPointInputs {
        const core::Tensor* positions = nullptr; // Float32 [N,3]
        const core::Tensor* colors = nullptr;    // Float32 [N,3]
        const core::Tensor* transform_indices = nullptr;
        const core::Tensor* selection = nullptr;
        const core::Tensor* preview = nullptr;
        const core::Tensor* deleted = nullptr;
        // SceneObject records are 96 bytes: column-major transform, float4
        // camera, uint4 flags. The palette is packed float4 records.
        std::span<const std::byte> objects;
        std::span<const std::byte> selection_palette;
    };

    // Rasterizes point-cloud discs through GpuKernelModule on Metal or Vulkan.
    // Work stays asynchronous on the tensor timeline.
    class SplatPointRenderer {
    public:
        explicit SplatPointRenderer(core::GpuBackend backend);
        ~SplatPointRenderer();
        SplatPointRenderer(const SplatPointRenderer&) = delete;
        SplatPointRenderer& operator=(const SplatPointRenderer&) = delete;

        [[nodiscard]] lfs::Result<void> render(const SplatPointInputs& inputs,
                                               const PointParameters& parameters,
                                               uint32_t width, uint32_t height,
                                               std::array<float, 4> background);

        [[nodiscard]] const core::Tensor& color() const;        // UInt8 [H,W,4]
        [[nodiscard]] const core::Tensor& linear_depth() const; // Float32 [H,W]

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace lfs::rendering
