/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/device/nearest_point.hpp"
#include "core/device/point_filter.hpp"
#include "core/device/point_labels.hpp"
#include "core/device/point_projection.hpp"
#include "core/device/point_region.hpp"
#include "core/device/point_tree.hpp"
#include "core/device/portability.cuh"
#include "core/device/ppisp.hpp"
#include "core/device/triangle_tree.hpp"
#include "core/morton.hpp"
#include "core/splat_transform_math.hpp"
#include <cfloat>
namespace lfs::core::internal {
    using lfs::core::PointProjection;
    using lfs::core::PointProjectionModel;
    using lfs::core::PointRegion2D;
    using lfs::core::PointRegion2DKind;
} // namespace lfs::core::internal
namespace hip_ppisp {
    using lfs::core::PpispParams;
    using lfs::core::internal::ppisp_pixel;
} // namespace hip_ppisp
namespace hip_affine {
    namespace splat_transform = lfs::core::splat_transform;
    using splat_transform::LinearTransform;
#include "core/device/affine_splat.cuh"
} // namespace hip_affine
namespace hip_simplify {
#include "core/device/simplify_merge.cuh"
}
namespace hip_spatial_components {
    using namespace lfs::core::internal;
    using lfs::core::device::componentRoot;
    using lfs::core::device::finalComponentRoot;
#include "core/device/spatial_components.cuh"
} // namespace hip_spatial_components
