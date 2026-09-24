/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/tensor_filters.hpp"
#include "internal/point_filter.hpp"
#include "internal/tensor_impl.hpp"

namespace lfs::core {
    void filter_points(Tensor& mask, const Tensor* points, const PointFilter& filter) {
        const auto byte_vector = [](const Tensor& t) {
            return t.is_valid() && t.ndim() == 1 && (t.dtype() == DataType::Bool || t.dtype() == DataType::UInt8);
        };
        LFS_ASSERT_MSG(byte_vector(mask) && !mask.has_zero_stride(),
                       "filter_points requires writable Bool/UInt8 [N] mask");
        constexpr auto maximum = static_cast<size_t>(std::numeric_limits<int32_t>::max());
        const auto count = mask.numel();
        LFS_ASSERT_MSG(count <= maximum, "filter_points count exceeds int32");
        LFS_ASSERT_MSG(bool(filter.box_transform) == bool(filter.box_min) &&
                           bool(filter.box_min) == bool(filter.box_max),
                       "filter_points requires all three box tensors");
        LFS_ASSERT_MSG(bool(filter.ellipsoid_transform) == bool(filter.ellipsoid_radii),
                       "filter_points requires both ellipsoid tensors");
        LFS_ASSERT_MSG(!filter.allowed || filter.indices, "filter_points allowed table requires indices");
        const bool geometry = filter.box_transform || filter.ellipsoid_transform || filter.window;
        LFS_ASSERT_MSG(!geometry || points, "filter_points geometry requires points");
        const std::array<const Tensor*, internal::FilterInputCount> sources{
            points, filter.transforms, filter.indices,
            filter.allowed, filter.box_transform, filter.box_min,
            filter.box_max, filter.ellipsoid_transform, filter.ellipsoid_radii};
        for (const auto* input : sources) {
            if (!input)
                continue;
            LFS_ASSERT_MSG(input->is_valid() && input->device() == mask.device(),
                           "filter_points requires valid tensors on the same device");
            internal::require_same_gpu_backend(mask, *input, "filter_points");
        }
        if (points)
            LFS_ASSERT_MSG(points->dtype() == DataType::Float32 && points->ndim() == 2 && points->size(0) == count &&
                               points->size(1) == 3,
                           "filter_points points must be Float32 [N,3]");
        if (filter.transforms)
            LFS_ASSERT_MSG(filter.transforms->dtype() == DataType::Float32 && filter.transforms->numel() % 16 == 0 &&
                               filter.transforms->numel() / 16 <= maximum,
                           "filter_points transforms must be Float32 4x4 matrices");
        if (filter.indices)
            LFS_ASSERT_MSG(filter.indices->dtype() == DataType::Int32 && filter.indices->ndim() == 1 &&
                               filter.indices->numel() == count,
                           "filter_points indices must be Int32 [N]");
        if (filter.allowed)
            LFS_ASSERT_MSG(byte_vector(*filter.allowed) && filter.allowed->numel() <= maximum,
                           "filter_points allowed must be Bool/UInt8 [K]");
        for (const auto* matrix : {filter.box_transform, filter.ellipsoid_transform})
            if (matrix)
                LFS_ASSERT_MSG(matrix->dtype() == DataType::Float32 && matrix->numel() == 16,
                               "filter_points shape transforms require Float32 4x4 matrices");
        for (const auto* vector : {filter.box_min, filter.box_max, filter.ellipsoid_radii})
            if (vector)
                LFS_ASSERT_MSG(vector->dtype() == DataType::Float32 && vector->numel() == 3,
                               "filter_points shape vectors require three Float32 values");
        if (filter.window) {
            const auto& p = filter.window->projection;
            LFS_ASSERT_MSG(p.width > 0 && p.height > 0, "filter_points requires positive image dimensions");
            LFS_ASSERT_MSG(p.model == PointProjectionModel::Pinhole || p.model == PointProjectionModel::Orthographic ||
                               p.model == PointProjectionModel::Equirectangular,
                           "filter_points received an unknown projection model");
        }
        if (count == 0 || (!geometry && !filter.allowed))
            return;
        internal::preserve_lazy_snapshots_before_write(mask);
        std::array<Tensor, internal::FilterInputCount> inputs;
        for (size_t i = 0; i < sources.size(); ++i)
            if (sources[i])
                inputs[i] =
                    internal::shares_storage(mask, *sources[i]) ? sources[i]->clone() : sources[i]->contiguous();
        auto output = mask.contiguous();
        internal::PointFilterProgram p;
        p.count = static_cast<uint32_t>(count);
        p.transform_count = filter.transforms ? filter.transforms->numel() / 16 : 0;
        p.allowed_count = filter.allowed ? filter.allowed->numel() : 0;
        p.flags = (filter.allowed ? LFS_FILTER_NODES : 0u) | (filter.box_transform ? LFS_FILTER_BOX : 0u) |
                  (filter.ellipsoid_transform ? LFS_FILTER_ELLIPSOID : 0u) | (filter.window ? LFS_FILTER_WINDOW : 0u) |
                  (filter.box_inverse ? LFS_FILTER_INVERSE_BOX : 0u) |
                  (filter.ellipsoid_inverse ? LFS_FILTER_INVERSE_ELLIPSOID : 0u);
        if (filter.window)
            p.window = *filter.window;
        if (mask.device() == Device::GPU) {
            pin_operands({&output});
            const auto stream = prepare_inputs_for_stream({&output}, output.stream());
            output.set_stream(stream);
            mask.set_stream(stream);
            for (size_t i = 0; i < inputs.size(); ++i)
                if (sources[i] && inputs[i].numel() != 0) {
                    pin_operands({&inputs[i]});
                    (void)prepare_inputs_for_stream({&inputs[i]}, stream);
                    p.inputs[i] = internal::storage_ref(inputs[i]);
                }
            internal::backend_ops_for(output).filter_points(internal::storage_ref(output), p,
                                                            internal::ExecContext{stream});
        } else {
            auto a = internal::pointFilterArgs(p.window);
            const auto data = [&]<class T>(size_t i) -> const T* {
                return sources[i] && inputs[i].numel() ? static_cast<const T*>(inputs[i].data_ptr()) : nullptr;
            };
            internal::loadPointFilterInputs(a, static_cast<uint8_t*>(output.data_ptr()), data, p.count, p.transform_count,
                                            p.allowed_count, p.flags);
            for (uint32_t i = 0; i < p.count; ++i)
                if (a.mask[i] && !internal::pointPassesFilter(a, i))
                    a.mask[i] = 0;
        }
        if (!mask.is_contiguous())
            mask.copy_from(output);
    }
} // namespace lfs::core
