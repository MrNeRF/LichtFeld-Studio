/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/tensor_spatial.hpp"
#include "internal/nearest_point.hpp"
#include "internal/tensor_impl.hpp"
#include <bit>
#include <format>
#include <tbb/parallel_for.h>
namespace lfs::core {
    namespace {
        void check_points(const Tensor& value, size_t columns, const char* name) {
            LFS_ASSERT_MSG(value.is_valid() && value.ndim() == 2 && value.size(1) == columns && value.dtype() == DataType::Float32,
                           std::format("{} requires Float32 [N,{}] (valid={}, rank={}, columns={}, dtype={})", name, columns,
                                       value.is_valid(), value.ndim(), value.ndim() == 2 ? value.size(1) : 0, int(value.dtype())));
            LFS_ASSERT_MSG(value.size(0) <= INT32_MAX, std::format("{} count exceeds int32 (count={})", name, value.size(0)));
        }
    } // namespace
    Tensor nearest_point_indices(const Tensor& queries, const Tensor& targets) {
        check_points(queries, 3, "nearest_point_indices queries");
        check_points(targets, 3, "nearest_point_indices targets");
        LFS_ASSERT_MSG(queries.device() == targets.device(), std::format("nearest_point_indices devices differ (queries={}, targets={})", int(queries.device()), int(targets.device())));
        internal::require_same_gpu_backend(queries, targets, "nearest_point_indices");
        auto output = internal::allocate_like(queries, {queries.size(0)}, DataType::Int32, -1.0f);
        if (!queries.size(0) || !targets.size(0))
            return output;
        const auto q = queries.contiguous(), t = targets.contiguous();
        // Six reductions determine grid scale; no element coordinates leave the device.
        const auto extent = (t.max(0) - t.min(0)).abs();
        const float largest = extent.max().item<float>();
        const float width = std::isfinite(largest) ? std::max(1e-6f, largest / std::cbrt(float(t.size(0)))) : 1.0f;
        const size_t buckets = std::bit_ceil(t.size(0));
        auto heads = internal::allocate_like(t, {buckets}, DataType::Int32, -1.0f);
        auto next = internal::allocate_like(t, {t.size(0)}, DataType::Int32, -1.0f);
        if (q.device() == Device::GPU) {
            pin_operands({&q, &t, &heads, &next});
            const auto stream = prepare_inputs_for_stream({&q, &t, &heads, &next}, output.stream());
            internal::backend_ops_for(q).nearest_point_indices(internal::storage_ref(q), internal::storage_ref(t), internal::storage_ref(heads), internal::storage_ref(next), internal::storage_ref(output), q.size(0), t.size(0), buckets, width, {stream});
        } else {
            auto* h = heads.ptr<int>();
            auto* n = next.ptr<int>();
            const auto* xyz = t.ptr<float>();
            for (size_t j = 0; j < t.size(0); ++j) {
                const auto* p = xyz + 3 * j;
                if (!internal::finite_point(p))
                    continue;
                const auto bucket = internal::hash_cell(internal::cell(p[0], width), internal::cell(p[1], width), internal::cell(p[2], width), buckets - 1);
                n[j] = h[bucket];
                h[bucket] = int(j);
            }
            const auto* points = q.ptr<float>();
            auto* result = output.ptr<int>();
            tbb::parallel_for(size_t{0}, q.size(0), [&](size_t i) { result[i] = internal::nearestPoint(points + 3 * i, xyz, h, n, int(t.size(0)), unsigned(buckets - 1), width); });
        }
        return output;
    }
    Tensor camera_frustum_counts(const Tensor& points, const Tensor& cameras, float maximum) {
        check_points(points, 3, "camera_frustum_counts points");
        check_points(cameras, 16, "camera_frustum_counts cameras");
        LFS_ASSERT_MSG(std::isfinite(maximum) && maximum >= 0, std::format("camera_frustum_counts invalid maximum (maximum={})", maximum));
        LFS_ASSERT_MSG(points.device() == cameras.device(), std::format("camera_frustum_counts devices differ (points={}, cameras={})", int(points.device()), int(cameras.device())));
        internal::require_same_gpu_backend(points, cameras, "camera_frustum_counts");
        auto output = internal::allocate_like(points, {points.size(0)}, DataType::Int32, 0.0f);
        if (!points.size(0) || !cameras.size(0))
            return output;
        const auto p = points.contiguous(), c = cameras.contiguous();
        if (p.device() == Device::GPU) {
            pin_operands({&p, &c});
            const auto stream = prepare_inputs_for_stream({&p, &c}, output.stream());
            internal::backend_ops_for(p).camera_frustum_counts(internal::storage_ref(p), internal::storage_ref(c), internal::storage_ref(output), p.size(0), c.size(0), maximum, {stream});
        } else {
            const auto* xyz = p.ptr<float>();
            const auto* views = c.ptr<float>();
            auto* result = output.ptr<int>();
            tbb::parallel_for(size_t{0}, p.size(0), [&](size_t i) { result[i] = internal::cameraFrustumCount(xyz + 3 * i, views, int(c.size(0)), maximum); });
        }
        return output;
    }
} // namespace lfs::core
