/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/error.hpp"
#include "core/tensor.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <span>

namespace lfs::rendering {
    // Projection of splat_project.slang (the native viewer's Projection),
    // column-major matrices.
    struct SplatProjection {
        std::array<float, 16> model_to_world{}, world_to_camera{};
        std::array<float, 4> camera_local{}, intrinsics{}, clip_scale{};
        std::array<uint32_t, 4> extent{};      // width, height, camera model, mip
        std::array<float, 4> rasterization{};  // pixel scale, expected depth, far, render profile
        std::array<float, 4> display{};        // tone, exposure, Spark opacity, GS center admission
        std::array<float, 4> panorama{};
    };
    static_assert(sizeof(SplatProjection) == 240);

    enum class SplatShStorage : uint32_t { CanonicalFloat32,
                                           SwizzledFloat32,
                                           SwizzledFloat16,
                                           Q16,
                                           RadSigned8 };

    // Resident SplatData attributes, read in place.
    struct SplatSources {
        const core::Tensor *means = nullptr, *scales = nullptr, *rotations = nullptr, *opacity = nullptr;
        const core::Tensor *sh0 = nullptr, *sh_rest = nullptr, *sh_bounds = nullptr;
        const core::Tensor* deleted = nullptr; // optional byte per source
        uint32_t count = 0, layout_rest = 0;   // layout_rest: resident SH rest coefficients
        SplatShStorage storage = SplatShStorage::CanonicalFloat32;
        bool half_attributes = false;
        uint32_t deleted_count = 0;            // zero uses count
        // Optional scene objects: host SceneObject records (96 bytes each),
        // uploaded per frame, and per-source object indices.
        std::span<const std::byte> objects;
        const core::Tensor* object_indices = nullptr;
    };

    // Overlay filters (crop boxes, ellipsoids, view volume, node emphasis) of
    // the desktop overlay parameter ABI: 207 float4, host bytes uploaded per frame.
    struct SplatOverlayInputs {
        std::span<const std::byte> parameters;
        std::span<const std::byte> node_mask; // one byte per scene node
    };

    // A resident LOD cut selected on the host: physical indices address the
    // source attributes; logical indices address scene and editor masks.
    // Optional arrays are empty. Uploaded per frame.
    struct SplatLodInputs {
        std::span<const uint32_t> indices;
        std::span<const uint32_t> logical_indices;
        std::span<const uint32_t> levels;  // debug colors
        std::span<const float> weights;    // transition fade
        bool debug = false;
        uint32_t logical_count = 0;        // zero uses the source count
    };

    enum class SplatPrimitive : uint32_t { Gaussian,
                                           Points,
                                           Discs,
                                           Gut };

    // Projects splats to ProjectedSplat records (and 3DGUT geometry) with the
    // single-source projection pass, on the tensor timeline.
    class SplatProjector {
    public:
        explicit SplatProjector(core::GpuBackend backend);
        ~SplatProjector();
        SplatProjector(const SplatProjector&) = delete;
        SplatProjector& operator=(const SplatProjector&) = delete;

        // Projects `count` sources, or the `lod` cut. `projected` holds one
        // 64-byte record per drawn splat; `gut` as many for Gut.
        [[nodiscard]] lfs::Result<void> project(const SplatSources& sources, const SplatProjection& projection, uint32_t degree,
                                           SplatPrimitive primitive, bool tight_bounds, core::Tensor& projected,
                                           core::Tensor* gut = nullptr, const SplatOverlayInputs* overlay = nullptr,
                                           const SplatLodInputs* lod = nullptr);

        // With overlay inputs, the uploaded parameters and the per-splat overlay
        // flags of the last project(), for the blend.
        [[nodiscard]] const core::Tensor& overlay_parameters() const;
        [[nodiscard]] const core::Tensor& overlay_flags() const;
        // With a LOD cut, the logical ID of each drawn splat (the physical index
        // when the cut has no logical indices), for the blend's masks.
        [[nodiscard]] const core::Tensor& logical_ids() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace lfs::rendering
