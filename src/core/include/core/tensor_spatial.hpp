/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/export.hpp"
#include "core/tensor_fwd.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <utility>

namespace lfs::core {
    enum class PointRegion2DKind : uint32_t { Disk,
                                              Rectangle,
                                              Polygon,
                                              Disks };

    struct PointRegion2D {
        PointRegion2DKind kind = PointRegion2DKind::Disk;
        float x0 = 0, y0 = 0, x1 = 0, y1 = 0;
        float radius = 1;
        float minimum_coordinate = -std::numeric_limits<float>::infinity();
    };

    // Set mask[i] to 1 when Float32 [N,2] points[i] is inside the region;
    // preserve every other Bool/UInt8 mask value. Disk uses (x0,y0), rectangle
    // includes its edges, and polygon uses even-odd inclusion. Polygon/Disks
    // require Float32 [K,2] vertices/centers. Empty geometry selects nothing.
    // Coordinates below minimum_coordinate are ignored. Tensors share a
    // device/backend; offset and strided views are supported, with no [N,K]
    // expansion. The mask must not share storage with points or geometry.
    LFS_CORE_API void mark_points_2d(Tensor& mask, const Tensor& points,
                                     const PointRegion2D& region,
                                     const Tensor* geometry = nullptr);

    enum class PointProjectionModel : uint32_t {
        Pinhole = 0,
        Orthographic = 1,
        Equirectangular = 3,
    };

    struct PointProjection {
        std::array<float, 9> rotation{1, 0, 0, 0, 1, 0, 0, 0, 1};
        std::array<float, 3> translation{};
        float focal_x = 1;
        float focal_y = 1;
        float center_x = 0;
        float center_y = 0;
        float ortho_scale = 1;
        int32_t width = 1;
        int32_t height = 1;
        PointProjectionModel model = PointProjectionModel::Pinhole;
        static constexpr float near_distance = 1e-6f;
        float invalid_value = -1e8f;
    };

    // Project Float32 [N,3] points to Float32 [N,2], preserving device/backend.
    // View coordinates are rotation * (world - translation), with +X right,
    // +Y up and -Z forward. Equirectangular projection also sees behind the
    // camera; invalid points receive invalid_value in both output components.
    // Optional Float32 row-major [T,4,4] transforms apply before the view.
    // Int32 [N] indices choose a transform (clamped to [0,T-1]); without them,
    // transform zero applies to every point. A Bool/UInt8 visibility table is
    // applied only with indices; out-of-range or hidden entries are invalid.
    // All supplied tensors share a device/backend; strided inputs are supported.
    LFS_CORE_API Tensor project_points(const Tensor& points, const PointProjection& projection,
                                       const Tensor* transforms = nullptr,
                                       const Tensor* indices = nullptr,
                                       const Tensor* visibility = nullptr);

    enum class PointRasterCrop : uint32_t {
        None,
        Box,
        Ellipsoid,
    };

    struct PointRaster {
        // Column-major matrices, as glm stores them.
        std::array<float, 16> view{};
        std::array<float, 16> view_projection{};
        int32_t width = 1;
        int32_t height = 1;
        bool orthographic = false;
        bool equirectangular = false;
        bool transparent_background = false;
        float ortho_scale = 1;
        float focal_y = 1;
        // World-space splat size, already scaled.
        float voxel_size = 0;
        float far_plane = 0;
        std::array<float, 3> background{};
        // Points outside the crop (inside with crop_inverse) are dropped, or
        // with crop_desaturate drawn mostly gray. crop_to_local is column-major;
        // an ellipsoid takes its radii from crop_min.
        PointRasterCrop crop = PointRasterCrop::None;
        std::array<float, 16> crop_to_local{};
        std::array<float, 3> crop_min{};
        std::array<float, 3> crop_max{};
        bool crop_inverse = false;
        bool crop_desaturate = false;
    };

    // Splat Float32 [N,3] points with Float32 [N,3] colors in [0,1] as
    // depth-tested disks whose pixel radius follows voxel_size and depth, into
    // a Float32 [3 or 4 with transparent_background, H, W] image and a
    // [1, H, W] depth, which is far_plane where nothing lands. Of the points
    // at a pixel's nearest depth, the one with the smallest 8-bit color wins.
    // Optional column-major Float32 [T,16] transforms apply per Int32 [N]
    // index (clamped to [0,T-1], zero without indices); a Bool/UInt8 [T]
    // visibility table hides points by that index, and a Bool/UInt8 [N]
    // deleted mask hides points. Runs on the
    // Vulkan and Metal backends; CUDA builds rasterize in the renderer.
    LFS_CORE_API std::pair<Tensor, Tensor> rasterize_points(const Tensor& points, const Tensor& colors,
                                                            const PointRaster& raster,
                                                            const Tensor* transforms = nullptr,
                                                            const Tensor* indices = nullptr,
                                                            const Tensor* visibility = nullptr,
                                                            const Tensor* deleted = nullptr);

    // For each Float32 [N,3] point, return whether a reference point is within
    // the inclusive radius. references is a Bool or UInt8 [N] mask; nonzero
    // entries participate, including the point itself. Nonfinite points never
    // match. Inputs share a device/backend; the Bool [N] result preserves it.
    // Radius must be positive, finite and normal (GPU subnormal arithmetic can
    // flush to zero). Scratch space is O(N), independent
    // of scene extent. Dense neighborhoods can still require quadratic work.
    LFS_CORE_API Tensor radius_neighbors(const Tensor& points, const Tensor& references, float radius);
} // namespace lfs::core
