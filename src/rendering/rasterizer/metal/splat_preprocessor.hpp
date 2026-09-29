/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#import <Metal/Metal.h>
#include <simd/simd.h>
#include <cstdint>
#include <memory>
#include <string>

namespace lfs::rendering::metal {

// Matches SplatData storage, not the training backend or a UIKit packed copy.
enum class ShStorage : uint32_t { CanonicalFloat32, SwizzledFloat32, SwizzledFloat16, Q16 };
enum class PrimitiveMode : uint32_t { Gaussian, Points, Discs };

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
};

struct alignas(16) SceneObject {
    simd_float4x4 model_to_world;
    simd_float4 camera_local;
    simd_uint4 flags; // visible, maximum active SH degree, reserved, reserved
};
static_assert(sizeof(SceneObject) == 96);
struct SceneBuffers {
    BufferSlice object_indices; // uint32 per primitive; invalid indices are culled
    BufferSlice objects; // SceneObject[count]
    uint32_t count = 0; // zero uses the single object in Projection
};

struct alignas(16) Projection {
    simd_float4x4 model_to_world;
    simd_float4x4 world_to_camera; // positive view Z, as in the desktop rasterizer
    simd_float4 camera_local; // SH direction uses the source/model coordinate frame
    simd_float4 intrinsics; // fx, fy, cx, cy in render pixels
    simd_float4 clip_scale; // near, far, scale modifier, pixel dilation variance
    simd_uint4 extent; // width, height, orthographic (0/1), mip antialiasing (0/1)
};
static_assert(sizeof(Projection) == 192);

struct alignas(16) ProjectedSplat {
    simd_float4 mean_depth; // x, y, linear view depth, contribution radius in pixels
    simd_float4 conic_opacity; // inverse covariance xx,xy,yy and activated opacity
    simd_float4 color;
    simd_uint4 bounds; // exclusive pixel AABB; empty means culled
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
                PrimitiveMode mode, BufferSlice output, const SceneBuffers& scene = {});

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace lfs::rendering::metal
