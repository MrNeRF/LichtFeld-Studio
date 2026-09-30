/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#import <Metal/Metal.h>
#include <cstdint>
#include <memory>
#include <simd/simd.h>
#include <string>

namespace lfs::rendering::metal {

    // Matches SplatData storage, not the training backend or a UIKit packed copy.
    enum class ShStorage : uint32_t { CanonicalFloat32,
                                      SwizzledFloat32,
                                      SwizzledFloat16,
                                      Q16 };
    enum class PrimitiveMode : uint32_t { Gaussian,
                                          Points,
                                          Discs,
                                          Gut };

    inline constexpr float kViewerNearClip = LFS_METAL_VIEWER_NEAR_CLIP;

    enum class CameraModel : uint32_t { Perspective,
                                        Orthographic,
                                        Equirectangular };

    struct BufferSlice {
        id<MTLBuffer> buffer = nil;
        NSUInteger offset = 0;
    };

    struct SplatBuffers {
        BufferSlice means, log_scales, rotations, opacity_logits, sh0, sh_rest, sh_bounds;
        BufferSlice deleted; // optional byte per source primitive
        uint32_t count = 0;
        uint32_t layout_rest = 0; // maximum/resident degree, independent of active degree
        ShStorage storage = ShStorage::Q16;
        bool non_sh_attrs_f16 = false;
    };

    struct alignas(16) SceneObject {
        simd_float4x4 model_to_world;
        simd_float4 camera_local;
        simd_uint4 flags; // visible, maximum active SH degree, reserved, reserved
    };
    static_assert(sizeof(SceneObject) == 96);
    struct SceneBuffers {
        BufferSlice object_indices; // uint32 per primitive; invalid indices are culled
        BufferSlice objects;        // SceneObject[count]
        uint32_t count = 0;         // zero uses the single object in Projection
    };
    // Shares the desktop overlay parameter ABI; masks stay resident and retain
    // their original logical primitive IDs through sorting.
    struct OverlayBuffers {
        BufferSlice parameters, flags, selection, preview, node_mask, colors;
        uint32_t parameter_count = 0;
        uint32_t node_count = 0;
        simd_float4 render_origin{};
    };

    // Draw a compact resident cut without copying/repacking SplatData. Physical
    // indices address source attributes; logical IDs address scene/editor masks.
    struct LodSelection {
        BufferSlice indices, logical_indices, levels, weights;
        uint32_t count = 0;
        uint32_t source_count = 0;
        bool enabled = false;
        bool debug = false;
        BufferSlice counter; // optional GPU-produced selected count, without host readback
    };

    struct alignas(16) Projection {
        simd_float4x4 model_to_world;
        simd_float4x4 world_to_camera;         // positive view Z, as in the desktop rasterizer
        simd_float4 camera_local;              // SH direction uses the source/model coordinate frame
        simd_float4 intrinsics;                // fx, fy, cx, cy in render pixels
        simd_float4 clip_scale;                // near, far, scale modifier, pixel dilation variance
        simd_uint4 extent;                     // width, height, CameraModel, mip antialiasing (0/1)
        simd_float4 rasterization{1, 0, 0, 0}; // output/source pixel scale, expected-depth flag/far, render profile (0 Studio, 1 portal)
        simd_float4 display{0, 1, 0, 0};       // tone operator, exposure, Spark opacity (0/1), reserved
        simd_float4 panorama{};                // full camera width/height and subregion origin in output pixels
    };
    static_assert(sizeof(Projection) == 240);

    // Normalized local-frame inverse rows in camera coordinates. Independent of
    // projected covariance and mip compensation: 3DGUT evaluates the pixel ray.
    // Portal only: inverse0/1.w hold the billboard axis; inverse2.w its minor extent.
    struct alignas(16) GutSplat {
        simd_float4 inverse0, inverse1, inverse2, mean_opacity;
    };
    static_assert(sizeof(GutSplat) == 64);

    struct alignas(16) ProjectedSplat {
        simd_float4 mean_depth;    // x, y, linear view depth, radius (portal GUT: billboard major extent)
        simd_float4 conic_opacity; // inverse covariance xx,xy,yy and activated opacity
        simd_float4 color;         // RGB radiance; w is radial sort distance squared, not alpha
        simd_uint4 bounds;         // exclusive pixel AABB; panorama X is a wrapped tile span; empty means culled
    };
    static_assert(sizeof(ProjectedSplat) == 64);

    // Encodes into a caller-owned command buffer. No CPU sort, readback, global
    // float32 SH expansion, queue wait, or per-frame allocation. Callers retain
    // buffer ownership until completion and synchronize external producers.
    class SplatPreprocessor {
    public:
        explicit SplatPreprocessor(id<MTLDevice> device);
        ~SplatPreprocessor();
        SplatPreprocessor(const SplatPreprocessor&) = delete;
        SplatPreprocessor& operator=(const SplatPreprocessor&) = delete;

        void prepare(ShStorage storage, uint32_t active_degree, PrimitiveMode mode);
        void encode(id<MTLCommandBuffer> command, const SplatBuffers& inputs,
                    const Projection& projection, uint32_t active_degree,
                    PrimitiveMode mode, BufferSlice output, const SceneBuffers& scene = {}, const OverlayBuffers& overlay = {}, BufferSlice gut_output = {}, const LodSelection& lod = {});

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace lfs::rendering::metal
