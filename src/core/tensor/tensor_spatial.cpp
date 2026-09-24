/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor_spatial.hpp"
#include "internal/point_spatial.hpp"
#include "internal/tensor_impl.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <vector>

namespace lfs::core {
    using namespace internal;

    Tensor radius_neighbors(const Tensor& points, const Tensor& references, const float radius) {
        LFS_ASSERT_MSG(points.is_valid() && references.is_valid(), "radius_neighbors requires valid tensors");
        LFS_ASSERT_MSG(points.ndim() == 2 && points.size(1) == 3 && points.dtype() == DataType::Float32,
                       "radius_neighbors requires Float32 [N,3] points");
        LFS_ASSERT_MSG(references.ndim() == 1 && references.numel() == points.size(0) &&
                           (references.dtype() == DataType::Bool || references.dtype() == DataType::UInt8),
                       "radius_neighbors requires a Bool or UInt8 [N] reference mask");
        LFS_ASSERT_MSG(std::isnormal(radius) && radius > 0.0f, "radius_neighbors radius must be positive, finite and normal");
        LFS_ASSERT_MSG(points.device() == references.device(), "radius_neighbors requires the same device");
        internal::require_same_gpu_backend(points, references, "radius_neighbors");
        const size_t count = points.size(0);
        LFS_ASSERT_MSG(count <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
                       "radius_neighbors point count exceeds int32");
        auto output = internal::allocate_like(points, TensorShape{count}, DataType::Bool);
        if (count == 0) {
            return output;
        }
        const auto positions = points.contiguous();
        const auto mask = references.contiguous();
        const size_t buckets = std::bit_ceil(count);
        const auto bucket_mask = static_cast<uint32_t>(buckets - 1);
        if (points.device() == Device::GPU) {
            auto heads = internal::allocate_like(points, TensorShape{buckets}, DataType::Int32, -1.0f);
            auto next = internal::allocate_like(points, TensorShape{count}, DataType::Int32);
            pin_operands({&positions, &mask, &heads, &next});
            const auto stream = prepare_inputs_for_stream({&positions, &mask, &heads, &next}, output.stream());
            internal::backend_ops_for(positions).radius_neighbors(
                internal::storage_ref(positions), internal::storage_ref(mask),
                internal::storage_ref(heads), internal::storage_ref(next), internal::storage_ref(output),
                count, buckets, radius, internal::ExecContext{stream});
            return output;
        }

        const auto* xyz = positions.ptr<float>();
        const auto* selected = static_cast<const uint8_t*>(mask.data_ptr());
        auto* result = output.ptr<bool>();
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
        for (size_t i = 0; i < count; ++i)
            result[i] = internal::pointHasNeighbor(xyz, selected, heads.data(), next.data(), i, bucket_mask, radius);
        return output;
    }
} // namespace lfs::core
