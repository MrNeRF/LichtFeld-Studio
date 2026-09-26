/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor_spatial.hpp"
#include "core/tensor_backend.hpp"
#include "internal/point_projection.hpp"
#include "internal/tensor_impl.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <vector>

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

    std::pair<Tensor, Tensor> rasterize_points(const Tensor& points, const Tensor& colors, const PointRaster& raster,
                                               const Tensor* transforms, const Tensor* indices,
                                               const Tensor* visibility, const Tensor* deleted) {
        LFS_ASSERT_MSG(points.is_valid() && points.dtype() == DataType::Float32 && points.ndim() == 2 &&
                           points.size(1) == 3 && points.device() == Device::GPU,
                       "rasterize_points requires Float32 [N,3] GPU points");
        const auto count = points.size(0);
        LFS_ASSERT_MSG(colors.is_valid() && colors.dtype() == DataType::Float32 && colors.ndim() == 2 &&
                           colors.size(0) == count && colors.size(1) == 3,
                       "rasterize_points requires Float32 [N,3] colors");
        LFS_ASSERT_MSG(gpu_backend_of(points) != GpuBackend::CUDA,
                       "rasterize_points runs on Vulkan and Metal; CUDA builds rasterize in the renderer");
        LFS_ASSERT_MSG(raster.width > 0 && raster.height > 0, "rasterize_points requires positive image dimensions");
        const size_t pixels = static_cast<size_t>(raster.width) * static_cast<size_t>(raster.height);
        constexpr auto max_count = static_cast<size_t>(std::numeric_limits<int32_t>::max());
        LFS_ASSERT_MSG(count <= max_count && 2 * pixels <= max_count, "rasterize_points size exceeds int32");

        const auto present = [](const Tensor* tensor) { return tensor && tensor->is_valid() && tensor->numel() != 0; };
        // As in the renderer's CUDA kernel, points without indices take
        // transform and visibility entry zero.
        for (const Tensor** tensor : {&transforms, &indices, &visibility, &deleted}) {
            if (!present(*tensor))
                *tensor = nullptr;
        }
        internal::require_same_gpu_backend(points, colors, "rasterize_points");
        for (const auto* tensor : {transforms, indices, visibility, deleted}) {
            if (tensor)
                internal::require_same_gpu_backend(points, *tensor, "rasterize_points");
        }
        LFS_ASSERT_MSG(!transforms || (transforms->dtype() == DataType::Float32 && transforms->numel() % 16 == 0),
                       "rasterize_points transforms must be Float32 [T,16]");
        LFS_ASSERT_MSG(!indices || (indices->dtype() == DataType::Int32 && indices->numel() == count),
                       "rasterize_points indices must be Int32 [N]");
        const auto is_byte_mask = [](const Tensor& tensor) {
            return tensor.dtype() == DataType::Bool || tensor.dtype() == DataType::UInt8;
        };
        LFS_ASSERT_MSG(!visibility || is_byte_mask(*visibility), "rasterize_points visibility must be Bool or UInt8");
        LFS_ASSERT_MSG(!deleted || (is_byte_mask(*deleted) && deleted->numel() == count),
                       "rasterize_points deleted mask must be Bool or UInt8 [N]");

        const uint32_t channels = raster.transparent_background ? 4u : 3u;
        auto image = internal::allocate_like(
            points, TensorShape{channels, static_cast<size_t>(raster.height), static_cast<size_t>(raster.width)},
            DataType::Float32);
        auto depth = internal::allocate_like(
            points, TensorShape{1, static_cast<size_t>(raster.height), static_cast<size_t>(raster.width)},
            DataType::Float32);
        auto scratch = internal::allocate_like(points, TensorShape{2 * pixels}, DataType::Int32);

        // Layout of internal::kPointRasterParameters, as the kernels read it.
        std::vector<float> values;
        values.reserve(internal::kPointRasterParameters);
        for (const auto* block : {&raster.view, &raster.view_projection, &raster.crop_to_local})
            values.insert(values.end(), block->begin(), block->end());
        for (const auto* block : {&raster.crop_min, &raster.crop_max, &raster.background})
            values.insert(values.end(), block->begin(), block->end());
        LFS_ASSERT_MSG(values.size() == internal::kPointRasterParameters, "rasterize_points parameter layout");
        Tensor parameters;
        {
            GpuBackendScope scope(*gpu_backend_of(points));
            parameters = Tensor::from_vector(values, {values.size()}, Device::CPU).to(Device::GPU);
        }

        const auto positions = points.contiguous();
        const auto rgb = colors.contiguous();
        const auto matrices = transforms ? transforms->contiguous() : Tensor{};
        const auto node_indices = indices ? indices->contiguous() : Tensor{};
        const auto visible = visibility ? visibility->contiguous() : Tensor{};
        const auto removed = deleted ? deleted->contiguous() : Tensor{};
        pin_operands({&positions, &rgb, &parameters, transforms ? &matrices : nullptr,
                      indices ? &node_indices : nullptr, visibility ? &visible : nullptr,
                      deleted ? &removed : nullptr});
        const auto stream = prepare_inputs_for_stream({&positions, &rgb, &parameters}, image.stream());
        const auto optional_ref = [&](const bool used, const Tensor& tensor) -> std::optional<internal::StorageRef> {
            if (!used)
                return std::nullopt;
            (void)prepare_inputs_for_stream({&tensor}, stream);
            return internal::storage_ref(tensor);
        };
        uint32_t flags = 0;
        if (raster.crop == PointRasterCrop::Box)
            flags |= internal::kPointRasterCropBox;
        if (raster.crop == PointRasterCrop::Ellipsoid)
            flags |= internal::kPointRasterCropEllipsoid;
        if (raster.crop_inverse)
            flags |= internal::kPointRasterCropInverse;
        if (raster.crop_desaturate)
            flags |= internal::kPointRasterCropDesaturate;
        if (raster.equirectangular)
            flags |= internal::kPointRasterEquirectangular;
        if (raster.orthographic)
            flags |= internal::kPointRasterOrthographic;
        if (raster.transparent_background)
            flags |= internal::kPointRasterTransparent;
        internal::backend_ops_for(positions).rasterize_points(
            {.positions = internal::storage_ref(positions),
             .colors = internal::storage_ref(rgb),
             .parameters = internal::storage_ref(parameters),
             .scratch = internal::storage_ref(scratch),
             .image = internal::storage_ref(image),
             .depth = internal::storage_ref(depth),
             .transforms = optional_ref(transforms != nullptr, matrices),
             .indices = optional_ref(indices != nullptr, node_indices),
             .visibility = optional_ref(visibility != nullptr, visible),
             .deleted = optional_ref(deleted != nullptr, removed),
             .count = count,
             .width = static_cast<uint32_t>(raster.width),
             .height = static_cast<uint32_t>(raster.height),
             .channels = channels,
             .transform_count = static_cast<uint32_t>(transforms ? transforms->numel() / 16 : 0),
             .visibility_count = static_cast<uint32_t>(visibility ? visibility->numel() : 0),
             .flags = flags,
             .ortho_scale = raster.ortho_scale,
             .focal_y = raster.focal_y,
             .voxel_size = raster.voxel_size,
             .far_plane = raster.far_plane},
            internal::ExecContext{stream});
        return {std::move(image), std::move(depth)};
    }
} // namespace lfs::core
