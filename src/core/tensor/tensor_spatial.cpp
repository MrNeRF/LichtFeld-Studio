/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor_spatial.hpp"
#include "internal/point_spatial.hpp"
#include "internal/tensor_impl.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <format>
#include <limits>
#include <vector>

namespace lfs::core {
    using namespace internal;

    static Tensor radius_query(const Tensor& points, const Tensor& references, const float radius,
                               const bool exclude_self, const Tensor* queries, const int32_t max_count, const bool spacing = false) {
        LFS_ASSERT_MSG(points.is_valid() && references.is_valid(), "radius_neighbors requires valid tensors");
        LFS_ASSERT_MSG(points.ndim() == 2 && points.size(1) == 3 && points.dtype() == DataType::Float32,
                       "radius_neighbors requires Float32 [N,3] points");
        LFS_ASSERT_MSG(references.ndim() == 1 && references.numel() == points.size(0) &&
                           (references.dtype() == DataType::Bool || references.dtype() == DataType::UInt8),
                       "radius_neighbors requires a Bool or UInt8 [N] reference mask");
        LFS_ASSERT_MSG(std::isnormal(radius) && radius > 0.0f, "radius_neighbors radius must be positive, finite and normal");
        LFS_ASSERT_MSG(points.device() == references.device(), "radius_neighbors requires the same device");
        internal::require_same_gpu_backend(points, references, "radius_neighbors");
        if (queries) {
            LFS_ASSERT_MSG(queries->is_valid() && queries->ndim() == 1 && queries->numel() == points.size(0) &&
                               (queries->dtype() == DataType::Bool || queries->dtype() == DataType::UInt8) &&
                               queries->device() == points.device(),
                           "radius_neighbors requires a Bool or UInt8 [N] query mask on the same device");
            internal::require_same_gpu_backend(points, *queries, "radius_neighbors");
        }
        const size_t count = points.size(0);
        LFS_ASSERT_MSG(count <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
                       "radius_neighbors point count exceeds int32");
        auto output = internal::allocate_like(points, TensorShape{count}, spacing ? DataType::Float32 : max_count ? DataType::Int32
                                                                                                                  : DataType::Bool);
        if (count == 0) {
            return output;
        }
        const auto positions = points.contiguous();
        const auto mask = references.contiguous();
        const auto query_mask = queries ? queries->contiguous() : Tensor{};
        const size_t buckets = std::bit_ceil(count);
        const auto bucket_mask = static_cast<uint32_t>(buckets - 1);
        if (points.device() == Device::GPU) {
            auto heads = internal::allocate_like(points, TensorShape{buckets}, DataType::Int32, -1.0f);
            auto next = internal::allocate_like(points, TensorShape{count}, DataType::Int32);
            pin_operands({&positions, &mask, &heads, &next});
            if (queries) {
                pin_operands({&query_mask});
            }
            const auto stream = queries
                                    ? prepare_inputs_for_stream({&positions, &mask, &heads, &next, &query_mask}, output.stream())
                                    : prepare_inputs_for_stream({&positions, &mask, &heads, &next}, output.stream());
            if (spacing) {
                internal::backend_ops_for(positions).point_neighbor_spacing(
                    internal::storage_ref(positions), internal::storage_ref(mask),
                    internal::storage_ref(heads), internal::storage_ref(next), internal::storage_ref(output),
                    count, buckets, radius, internal::ExecContext{stream});
            } else if (max_count) {
                internal::backend_ops_for(positions).radius_neighbor_counts(
                    internal::storage_ref(positions), internal::storage_ref(mask),
                    internal::storage_ref(heads), internal::storage_ref(next), internal::storage_ref(output),
                    count, buckets, radius, max_count,
                    queries ? std::optional{internal::storage_ref(query_mask)} : std::nullopt,
                    internal::ExecContext{stream});
            } else {
                internal::backend_ops_for(positions).radius_neighbors(
                    internal::storage_ref(positions), internal::storage_ref(mask),
                    internal::storage_ref(heads), internal::storage_ref(next), internal::storage_ref(output),
                    count, buckets, radius, exclude_self,
                    queries ? std::optional{internal::storage_ref(query_mask)} : std::nullopt,
                    internal::ExecContext{stream});
            }
            return output;
        }

        const auto* xyz = positions.ptr<float>();
        const auto* selected = static_cast<const uint8_t*>(mask.data_ptr());
        const auto* queried = queries ? static_cast<const uint8_t*>(query_mask.data_ptr()) : nullptr;
        std::vector<int32_t> heads(buckets, -1), next(count, -1);
        for (size_t i = 0; i < count; ++i) {
            const auto* p = xyz + i * 3;
            if (!selected[i] || !finite_point(p)) {
                continue;
            }
            const auto bucket = hash_cell(cell(p[0], radius), cell(p[1], radius), cell(p[2], radius), bucket_mask);
            next[i] = heads[bucket];
            heads[bucket] = static_cast<int32_t>(i);
        }
        if (spacing) {
            auto* result = output.ptr<float>();
            for (size_t i = 0; i < count; ++i)
                result[i] = pointNeighborSpacing(xyz, heads.data(), next.data(), i, bucket_mask, radius);
        } else if (max_count) {
            auto* result = output.ptr<int32_t>();
            for (size_t i = 0; i < count; ++i)
                result[i] = (!queried || queried[i]) ? pointNeighborCount(xyz, heads.data(), next.data(), i, bucket_mask, radius, max_count) : 0;
        } else {
            auto* result = output.ptr<bool>();
            for (size_t i = 0; i < count; ++i)
                result[i] = (!queried || queried[i]) && pointHasNeighbor(xyz, selected, heads.data(), next.data(), i, bucket_mask, radius, exclude_self);
        }
        return output;
    }

    Tensor radius_neighbors(const Tensor& points, const Tensor& references, const float radius,
                            const bool exclude_self, const Tensor* queries) {
        return radius_query(points, references, radius, exclude_self, queries, 0);
    }

    Tensor radius_neighbor_counts(const Tensor& points, const Tensor& references, const float radius,
                                  const int32_t max_count, const Tensor* queries) {
        LFS_ASSERT_MSG(max_count > 0, std::format("radius_neighbor_counts requires a positive max_count (max_count={})", max_count));
        return radius_query(points, references, radius, true, queries, max_count);
    }

    Tensor radius_neighbor_min(const Tensor& points, const Tensor& values, const float radius) {
        LFS_ASSERT_MSG(points.is_valid() && points.ndim() == 2 && points.size(1) == 3 &&
                           points.dtype() == DataType::Float32,
                       std::format("radius_neighbor_min requires Float32 [N,3] points (valid={}, rank={}, columns={}, dtype={})",
                                   points.is_valid(), points.ndim(), points.ndim() == 2 ? points.size(1) : 0,
                                   points.is_valid() ? static_cast<int>(points.dtype()) : -1));
        LFS_ASSERT_MSG(values.is_valid() && values.ndim() == 1 && values.numel() == points.size(0) &&
                           (values.dtype() == DataType::Int32 || values.dtype() == DataType::Float32),
                       std::format("radius_neighbor_min requires Int32 or Float32 [N] values (valid={}, rank={}, count={}, dtype={})",
                                   values.is_valid(), values.ndim(), values.numel(),
                                   values.is_valid() ? static_cast<int>(values.dtype()) : -1));
        LFS_ASSERT_MSG(std::isnormal(radius) && radius > 0.0f,
                       std::format("radius_neighbor_min radius must be positive, finite and normal (radius={})", radius));
        LFS_ASSERT_MSG(points.device() == values.device(),
                       std::format("radius_neighbor_min requires the same device (points={}, values={})",
                                   static_cast<int>(points.device()), static_cast<int>(values.device())));
        internal::require_same_gpu_backend(points, values, "radius_neighbor_min");
        const size_t count = points.size(0);
        LFS_ASSERT_MSG(count <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
                       std::format("radius_neighbor_min point count exceeds int32 (count={})", count));
        auto output = internal::allocate_like(values, TensorShape{count}, values.dtype());
        if (!count)
            return output;
        const auto positions = points.contiguous();
        const auto source = values.contiguous();
        const auto references = internal::allocate_like(points, TensorShape{count}, DataType::Bool, 1.0f);
        const size_t buckets = std::bit_ceil(count);
        const auto bucket_mask = static_cast<uint32_t>(buckets - 1);
        if (points.device() == Device::GPU) {
            auto heads = internal::allocate_like(points, TensorShape{buckets}, DataType::Int32, -1.0f);
            auto next = internal::allocate_like(points, TensorShape{count}, DataType::Int32);
            pin_operands({&positions, &source, &references, &heads, &next});
            const auto stream = prepare_inputs_for_stream({&positions, &source, &references, &heads, &next}, output.stream());
            internal::backend_ops_for(positions).radius_neighbor_min(
                internal::storage_ref(positions), internal::storage_ref(source), internal::storage_ref(references),
                internal::storage_ref(heads), internal::storage_ref(next), internal::storage_ref(output),
                count, buckets, radius, internal::ExecContext{stream});
            return output;
        }
        const auto* xyz = positions.ptr<float>();
        std::vector<int32_t> heads(buckets, -1), next(count, -1);
        for (size_t i = 0; i < count; ++i) {
            const auto* p = xyz + i * 3;
            if (!finite_point(p))
                continue;
            const auto bucket = hash_cell(cell(p[0], radius), cell(p[1], radius), cell(p[2], radius), bucket_mask);
            next[i] = heads[bucket];
            heads[bucket] = static_cast<int32_t>(i);
        }
        if (values.dtype() == DataType::Int32) {
            const auto* input = source.ptr<int32_t>();
            auto* result = output.ptr<int32_t>();
            for (size_t i = 0; i < count; ++i)
                result[i] = pointNeighborMin(xyz, input, heads.data(), next.data(), i, bucket_mask, radius);
        } else {
            const auto* input = source.ptr<float>();
            auto* result = output.ptr<float>();
            for (size_t i = 0; i < count; ++i)
                result[i] = pointNeighborMin(xyz, input, heads.data(), next.data(), i, bucket_mask, radius);
        }
        return output;
    }

    Tensor point_neighbor_spacing(const Tensor& points, const float cell_width) {
        LFS_ASSERT_MSG(points.is_valid() && points.ndim() == 2,
                       std::format("point_neighbor_spacing requires rank-2 points (valid={}, rank={})", points.is_valid(), points.ndim()));
        return radius_query(points, internal::allocate_like(points, {points.size(0)}, DataType::Bool, 1.0f), cell_width, true, nullptr, 0, true);
    }
} // namespace lfs::core
