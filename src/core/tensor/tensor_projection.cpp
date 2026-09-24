/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor_spatial.hpp"
#include "internal/point_projection.hpp"
#include "internal/tensor_impl.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

namespace lfs::core {
    Tensor project_points(const Tensor& points, const PointProjection& projection,
                          const Tensor* transforms, const Tensor* indices, const Tensor* visibility) {
        LFS_ASSERT_MSG(points.is_valid() && points.dtype() == DataType::Float32 &&
                           points.ndim() == 2 && points.size(1) == 3,
                       "project_points requires Float32 [N,3] points");
        const auto count = points.size(0);
        constexpr auto max_count = static_cast<size_t>(std::numeric_limits<int32_t>::max());
        LFS_ASSERT_MSG(count <= max_count, "project_points point count exceeds int32");
        LFS_ASSERT_MSG(projection.width > 0 && projection.height > 0,
                       "project_points requires positive image dimensions");
        LFS_ASSERT_MSG(projection.model == PointProjectionModel::Pinhole ||
                           projection.model == PointProjectionModel::Orthographic ||
                           projection.model == PointProjectionModel::Equirectangular,
                       "project_points received an unknown camera model");

        const auto present = [](const Tensor* tensor) {
            return tensor && tensor->is_valid() && tensor->numel() != 0;
        };
        if (!present(transforms))
            transforms = nullptr;
        if (!present(indices))
            indices = nullptr;
        if (!indices || !present(visibility))
            visibility = nullptr;

        for (const auto* tensor : {transforms, indices, visibility}) {
            if (!tensor)
                continue;
            LFS_ASSERT_MSG(tensor->device() == points.device(), "project_points requires the same device");
            internal::require_same_gpu_backend(points, *tensor, "project_points");
        }
        const size_t transform_count = transforms ? transforms->numel() / 16 : 0;
        const size_t visibility_count = visibility ? visibility->numel() : 0;
        if (transforms) {
            LFS_ASSERT_MSG(transforms->dtype() == DataType::Float32 && transforms->numel() % 16 == 0 &&
                               transform_count <= max_count,
                           "project_points transforms must contain Float32 row-major 4x4 matrices");
        }
        if (indices) {
            LFS_ASSERT_MSG(indices->dtype() == DataType::Int32 && indices->ndim() == 1 && indices->numel() == count,
                           "project_points indices must be Int32 [N]");
        }
        if (visibility) {
            LFS_ASSERT_MSG((visibility->dtype() == DataType::Bool || visibility->dtype() == DataType::UInt8) &&
                               visibility->ndim() == 1 && visibility_count <= max_count,
                           "project_points visibility must be a Bool or UInt8 vector");
        }

        auto output = internal::allocate_like(points, TensorShape{count, 2}, DataType::Float32);
        if (count == 0)
            return output;
        const auto positions = points.contiguous();
        const auto matrices = transforms ? transforms->contiguous() : Tensor{};
        const auto node_indices = indices ? indices->contiguous() : Tensor{};
        const auto visible = visibility ? visibility->contiguous() : Tensor{};

        if (points.device() == Device::GPU) {
            pin_operands({&positions, transforms ? &matrices : nullptr,
                          indices ? &node_indices : nullptr, visibility ? &visible : nullptr});
            const auto stream = prepare_inputs_for_stream({&positions}, output.stream());
            std::optional<internal::StorageRef> matrix_storage, index_storage, visibility_storage;
            if (transforms) {
                (void)prepare_inputs_for_stream({&matrices}, stream);
                matrix_storage = internal::storage_ref(matrices);
            }
            if (indices) {
                (void)prepare_inputs_for_stream({&node_indices}, stream);
                index_storage = internal::storage_ref(node_indices);
            }
            if (visibility) {
                (void)prepare_inputs_for_stream({&visible}, stream);
                visibility_storage = internal::storage_ref(visible);
            }
            internal::backend_ops_for(positions).project_points(
                internal::storage_ref(positions), internal::storage_ref(output), count, projection,
                matrix_storage ? &*matrix_storage : nullptr, transform_count,
                index_storage ? &*index_storage : nullptr,
                visibility_storage ? &*visibility_storage : nullptr, visibility_count,
                internal::ExecContext{stream});
            return output;
        }

        const float* xyz = positions.ptr<float>();
        const float* matrices_data = transforms ? matrices.ptr<float>() : nullptr;
        const int32_t* ids = indices ? node_indices.ptr<int32_t>() : nullptr;
        const auto* visibility_data = visibility ? static_cast<const uint8_t*>(visible.data_ptr()) : nullptr;
        float* result = output.ptr<float>();
        for (size_t i = 0; i < count; ++i)
            internal::projectPoint(xyz, result, i, projection, matrices_data, transform_count, ids, visibility_data, visibility_count);
        return output;
    }
} // namespace lfs::core
