/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor_spatial.hpp"
#include "internal/point_region.hpp"
#include "internal/tensor_impl.hpp"

#include <limits>
#include <optional>

namespace lfs::core {
    void mark_points_2d(Tensor& mask, const Tensor& points, const PointRegion2D& region,
                        const Tensor* geometry) {
        LFS_ASSERT_MSG(points.is_valid() && points.dtype() == DataType::Float32 &&
                           points.ndim() == 2 && points.size(1) == 2,
                       "mark_points_2d requires Float32 [N,2] points");
        LFS_ASSERT_MSG(mask.is_valid() && mask.ndim() == 1 && mask.numel() == points.size(0) &&
                           (mask.dtype() == DataType::Bool || mask.dtype() == DataType::UInt8),
                       "mark_points_2d requires a Bool or UInt8 [N] mask");
        const auto count = mask.numel();
        LFS_ASSERT_MSG(!mask.has_zero_stride(), "mark_points_2d cannot write a broadcast mask");
        constexpr auto max_count = static_cast<size_t>(std::numeric_limits<int32_t>::max());
        LFS_ASSERT_MSG(count <= max_count, "mark_points_2d count exceeds int32");
        LFS_ASSERT_MSG(region.kind == PointRegion2DKind::Disk || region.kind == PointRegion2DKind::Rectangle ||
                           region.kind == PointRegion2DKind::Polygon || region.kind == PointRegion2DKind::Disks,
                       "mark_points_2d received an unknown region kind");
        const bool uses_geometry = region.kind == PointRegion2DKind::Polygon || region.kind == PointRegion2DKind::Disks;
        if (!uses_geometry)
            geometry = nullptr;
        if (uses_geometry) {
            LFS_ASSERT_MSG(geometry && geometry->is_valid() && geometry->dtype() == DataType::Float32 &&
                               geometry->ndim() == 2 && geometry->size(1) == 2 && geometry->size(0) <= max_count,
                           "mark_points_2d requires Float32 [K,2] geometry");
        }
        for (const auto* input : {&points, geometry}) {
            if (!input)
                continue;
            LFS_ASSERT_MSG(input->device() == mask.device(), "mark_points_2d requires the same device");
            internal::require_same_gpu_backend(mask, *input, "mark_points_2d");
            LFS_ASSERT_MSG(!internal::shares_storage(mask, *input), "mark_points_2d mask must not alias its inputs");
        }
        const auto geometry_count = geometry ? geometry->size(0) : 0;
        if (count == 0 || (uses_geometry && geometry_count == 0) ||
            (region.kind == PointRegion2DKind::Polygon && geometry_count < 3))
            return;

        internal::preserve_lazy_snapshots_before_write(mask);
        auto output = mask.contiguous();
        const auto positions = points.contiguous();
        const auto vertices = geometry ? geometry->contiguous() : Tensor{};
        if (mask.device() == Device::GPU) {
            pin_operands({&output, &positions, geometry ? &vertices : nullptr});
            const auto stream = prepare_inputs_for_stream({&output, &positions}, output.stream());
            std::optional<internal::StorageRef> vertex_storage;
            if (geometry) {
                (void)prepare_inputs_for_stream({&vertices}, stream);
                vertex_storage = internal::storage_ref(vertices);
            }
            internal::backend_ops_for(output).mark_points_2d(
                internal::storage_ref(output), internal::storage_ref(positions), count, region,
                vertex_storage ? &*vertex_storage : nullptr, geometry_count, internal::ExecContext{stream});
        } else {
            const auto* xy = positions.ptr<float>();
            const auto* poly = geometry ? vertices.ptr<float>() : nullptr;
            auto* bytes = static_cast<uint8_t*>(output.data_ptr());
            const auto mark = [&]<PointRegion2DKind Kind>() {
                for (size_t i = 0; i < count; ++i)
                    if (internal::pointInRegion<Kind>(xy[2 * i], xy[2 * i + 1], region, poly, geometry_count))
                        bytes[i] = 1;
            };
            switch (region.kind) {
            case PointRegion2DKind::Disk: mark.template operator()<PointRegion2DKind::Disk>(); break;
            case PointRegion2DKind::Rectangle: mark.template operator()<PointRegion2DKind::Rectangle>(); break;
            case PointRegion2DKind::Polygon: mark.template operator()<PointRegion2DKind::Polygon>(); break;
            case PointRegion2DKind::Disks: mark.template operator()<PointRegion2DKind::Disks>(); break;
            }
        }
        if (!mask.is_contiguous())
            mask.copy_from(output);
    }
} // namespace lfs::core
