/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor_spatial.hpp"
#include "core/tensor_export.hpp"
#include "core/tensor_fused.hpp"
#include "internal/point_spatial.hpp"
#include "internal/point_tree.hpp"
#include "internal/tensor_impl.hpp"
#include "internal/triangle_tree.hpp"

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

    namespace {
        struct PointTree {
            Tensor sorted;
            Tensor boxes;
            Tensor visit;
            PointTreeProgram program;
        };

        // Order of finite [N,3] points along a 30-bit Morton curve, which keeps runs of the order spatially
        // compact. The curve spans the interquartile box of a sample widened by twice its size, so sparse
        // outliers clamp to the border cells instead of coarsening every cell; one 32-bit key sorts as a
        // float in place of 64-bit keys.
        Tensor morton30_order(const Tensor& points) {
            static const auto kernel = [] {
                namespace f = fused;
                f::Builder builder(1);
                const auto xyz = builder.input(DataType::Float32, 2);
                const auto low = builder.input(DataType::Float32, 1);
                const auto scale = builder.input(DataType::Float32, 1);
                const auto row = builder.iota(0);
                const auto spread = [](f::Expr v) {
                    v = (v | (v << 16u)) & 0x030000FFu;
                    v = (v | (v << 8u)) & 0x0300F00Fu;
                    v = (v | (v << 4u)) & 0x030C30C3u;
                    return (v | (v << 2u)) & 0x09249249u;
                };
                const auto axis = [&](const int32_t a) {
                    const auto offset = xyz.gather({row, builder.constant(a)}) - low.at({uint32_t(a)});
                    return spread(f::clamp(f::floor(offset * scale.at({uint32_t(a)})), 0.0f, 1023.0f).cast(DataType::UInt32));
                };
                // Offset into the normal floats, whose order as Float32 is the order of their bits.
                builder.output((((axis(0) << 2u) | (axis(1) << 1u) | axis(2)) + 0x00800000u).cast(DataType::Int32),
                               DataType::Int32);
                return f::Kernel(builder);
            }();
            const size_t count = points.size(0), sampled = std::min<size_t>(count, 4096);
            const auto indices = (Tensor::linspace(0, static_cast<float>(sampled - 1), sampled, points.device()) *
                                  (static_cast<float>(count) / static_cast<float>(sampled)))
                                     .to(DataType::Int32);
            const auto sorted = points.index_select(0, indices).sort(0).first;
            const auto lower = sorted.slice(0, sampled / 4, sampled / 4 + 1).squeeze(0);
            const auto upper = sorted.slice(0, sampled * 3 / 4, sampled * 3 / 4 + 1).squeeze(0);
            const auto spread = upper - lower;
            const auto low = lower - spread * 2.0f;
            const auto extent = spread * 5.0f;
            const auto scale = Tensor::where(extent.gt(0.0f), extent.reciprocal() * 1024.0f, Tensor::zeros_like(extent));
            const auto keys = kernel({count}, {points, low.contiguous(), scale.contiguous()})[0];
            return keys.view_as(DataType::Float32).sort(0).second.to(DataType::Int32);
        }

        // Rows padded to a multiple of the fanout by repeating the last, which leaves every bound unchanged.
        Tensor fanout_groups(const Tensor& rows) {
            const size_t count = rows.size(0), width = rows.size(1);
            const size_t groups = (count + kPointTreeFanout - 1) / kPointTreeFanout;
            const auto padded = groups * kPointTreeFanout == count
                                    ? rows
                                    : Tensor::cat({rows, rows.slice(0, count - 1, count).expand({int(groups * kPointTreeFanout - count), int(width)})}, 0);
            return padded.contiguous().reshape({int(groups), int(kPointTreeFanout), int(width)});
        }

        PointTree build_point_tree(const Tensor& points, const Tensor& references) {
            PointTree tree;
            const size_t count = points.size(0);
            tree.program.points = static_cast<uint32_t>(count);
            const auto keep = references.to(DataType::Bool).logical_and(points.isfinite().all(1));
            const auto ids = keep.nonzero().reshape({-1}).to(DataType::Int32);
            tree.program.references = static_cast<uint32_t>(ids.numel());
            if (tree.program.references == 0)
                return tree;
            const auto reference_points = points.index_select(0, ids);
            const auto order = reference_points.device() == Device::GPU ? morton30_order(reference_points)
                                                                        : morton_sort_indices(reference_points).to(DataType::Int32);
            tree.sorted = reference_points.index_select(0, order).contiguous();
            const auto sorted_ids = ids.index_select(0, order);
            const auto others = keep.logical_not().nonzero().reshape({-1}).to(DataType::Int32);
            tree.visit = (others.numel() ? Tensor::cat({sorted_ids, others}, 0) : sorted_ids).contiguous();
            std::vector<Tensor> levels;
            auto grouped = fanout_groups(tree.sorted);
            levels.push_back(Tensor::cat({grouped.min(1), grouped.max(1)}, 1));
            while (levels.back().size(0) > kPointTreeFanout) {
                grouped = fanout_groups(levels.back());
                levels.push_back(Tensor::cat({grouped.slice(2, 0, 3).min(1), grouped.slice(2, 3, 6).max(1)}, 1));
            }
            LFS_ASSERT_MSG(levels.size() <= kPointTreeMaxLevels, "point tree exceeds its level limit");
            tree.program.levels = static_cast<uint32_t>(levels.size());
            uint32_t offset = 0;
            for (size_t level = 0; level < levels.size(); ++level) {
                tree.program.level_offset[level] = offset;
                tree.program.level_count[level] = static_cast<uint32_t>(levels[level].size(0));
                offset += tree.program.level_count[level];
            }
            tree.boxes = Tensor::cat(levels, 0).contiguous();
            return tree;
        }

        // The largest of values (in tree order) under each box, stored like the boxes.
        Tensor box_maxima(const PointTree& tree, const Tensor& sorted_values) {
            std::vector<Tensor> levels;
            auto groups = fanout_groups(sorted_values.unsqueeze(1));
            levels.push_back(groups.max(1).squeeze(1));
            for (uint32_t level = 1; level < tree.program.levels; ++level) {
                groups = fanout_groups(levels.back().unsqueeze(1));
                levels.push_back(groups.max(1).squeeze(1));
            }
            return Tensor::cat(levels, 0).contiguous();
        }

        void point_tree_counts_cpu(const Tensor& points, const PointTree& tree, const Tensor& radii,
                                   const Tensor& queries, Tensor& output) {
            const auto xyz = points.contiguous();
            const auto* p = xyz.ptr<float>();
            const auto* sorted = tree.sorted.ptr<float>();
            const auto* boxes = tree.boxes.ptr<float>();
            const auto* visit = tree.visit.ptr<int32_t>();
            const auto query_bytes = queries.is_valid() ? queries.contiguous() : Tensor{};
            const auto* use = query_bytes.is_valid() ? static_cast<const uint8_t*>(query_bytes.data_ptr()) : nullptr;
            const auto radius_values = radii.is_valid() ? radii.contiguous() : Tensor{};
            const auto* radius = radius_values.is_valid() ? radius_values.ptr<float>() : nullptr;
            auto* result = output.ptr<int32_t>();
            const int64_t count = tree.program.points;
#pragma omp parallel for schedule(dynamic, 1024)
            for (int64_t t = 0; t < count; ++t) {
                const auto i = static_cast<size_t>(visit[t]);
                const float* q = p + i * 3;
                result[i] = (use && !use[i]) || !finite_point(q)
                                ? 0
                                : pointTreeCount(sorted, boxes, tree.program, q, t < int64_t(tree.program.references) ? t : -1,
                                                 radius ? radius[i] : tree.program.radius, tree.program.max_count);
            }
        }
    } // namespace

    Tensor radius_neighbor_counts(const Tensor& points, const Tensor& references, const Tensor& radii,
                                  const int32_t max_count, const Tensor* queries) {
        LFS_ASSERT_MSG(points.is_valid() && references.is_valid() && radii.is_valid(),
                       "radius_neighbor_counts requires valid tensors");
        LFS_ASSERT_MSG(points.ndim() == 2 && points.size(1) == 3 && points.dtype() == DataType::Float32,
                       "radius_neighbor_counts requires Float32 [N,3] points");
        const size_t count = points.size(0);
        LFS_ASSERT_MSG(count <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
                       "radius_neighbor_counts point count exceeds int32");
        LFS_ASSERT_MSG(references.ndim() == 1 && references.numel() == count &&
                           (references.dtype() == DataType::Bool || references.dtype() == DataType::UInt8),
                       "radius_neighbor_counts requires a Bool or UInt8 [N] reference mask");
        LFS_ASSERT_MSG(radii.ndim() == 1 && radii.numel() == count && radii.dtype() == DataType::Float32,
                       "radius_neighbor_counts requires Float32 [N] radii");
        LFS_ASSERT_MSG(max_count > 0, std::format("radius_neighbor_counts requires a positive max_count (max_count={})", max_count));
        LFS_ASSERT_MSG(points.device() == references.device() && points.device() == radii.device(),
                       "radius_neighbor_counts requires the same device");
        internal::require_same_gpu_backend(points, references, "radius_neighbor_counts");
        internal::require_same_gpu_backend(points, radii, "radius_neighbor_counts");
        if (queries) {
            LFS_ASSERT_MSG(queries->is_valid() && queries->ndim() == 1 && queries->numel() == count &&
                               (queries->dtype() == DataType::Bool || queries->dtype() == DataType::UInt8) &&
                               queries->device() == points.device(),
                           "radius_neighbor_counts requires a Bool or UInt8 [N] query mask on the same device");
            internal::require_same_gpu_backend(points, *queries, "radius_neighbor_counts");
        }
        auto output = internal::allocate_like(points, TensorShape{count}, DataType::Int32, 0.0f);
        if (count == 0)
            return output;
        auto tree = build_point_tree(points, references);
        if (tree.program.references == 0)
            return output;
        tree.program.max_count = max_count;
        const auto positions = points.contiguous();
        const auto radius_values = radii.contiguous();
        const auto query_mask = queries ? queries->contiguous() : Tensor{};
        if (points.device() == Device::GPU) {
            pin_operands({&positions, &tree.sorted, &tree.boxes, &tree.visit, &radius_values});
            if (queries)
                pin_operands({&query_mask});
            const auto stream = queries ? prepare_inputs_for_stream({&positions, &tree.sorted, &tree.boxes, &tree.visit,
                                                                     &radius_values, &query_mask},
                                                                    output.stream())
                                        : prepare_inputs_for_stream({&positions, &tree.sorted, &tree.boxes, &tree.visit,
                                                                     &radius_values},
                                                                    output.stream());
            if (internal::backend_ops_for(positions).point_tree_counts(
                    internal::storage_ref(positions), internal::storage_ref(tree.sorted), internal::storage_ref(tree.boxes),
                    internal::storage_ref(tree.visit), internal::storage_ref(radius_values),
                    queries ? std::optional{internal::storage_ref(query_mask)} : std::nullopt,
                    internal::storage_ref(output), tree.program, internal::ExecContext{stream}))
                return output;
            // No kernel on this backend: traverse on the host.
            PointTree host{tree.sorted.cpu(), tree.boxes.cpu(), tree.visit.cpu(), tree.program};
            auto result = Tensor::zeros({count}, Device::CPU, DataType::Int32);
            point_tree_counts_cpu(positions.cpu(), host, radius_values.cpu(), queries ? query_mask.cpu() : Tensor{}, result);
            GpuBackendScope scope(gpu_backend_of(positions).value());
            return result.to(Device::GPU);
        }
        point_tree_counts_cpu(positions, tree, radius_values, query_mask, output);
        return output;
    }

    Tensor radius_neighbor_min(const Tensor& points, const Tensor& values, const float radius, const Tensor* radii) {
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
        Tensor local_radii;
        if (radii) {
            LFS_ASSERT_MSG(radii->is_valid() && radii->ndim() == 1 && radii->numel() == points.size(0) &&
                               radii->dtype() == DataType::Float32 && radii->device() == points.device(),
                           std::format("radius_neighbor_min radii require Float32 [N] on the points device (valid={}, rank={}, count={}, dtype={}, device={}, N={}, points device={})",
                                       radii->is_valid(), radii->ndim(), radii->numel(), static_cast<int>(radii->dtype()),
                                       static_cast<int>(radii->device()), points.size(0), static_cast<int>(points.device())));
            internal::require_same_gpu_backend(points, *radii, "radius_neighbor_min radii");
            local_radii = radii->contiguous();
        }
        const size_t count = points.size(0);
        LFS_ASSERT_MSG(count <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
                       std::format("radius_neighbor_min point count exceeds int32 (count={})", count));
        auto output = internal::allocate_like(values, TensorShape{count}, values.dtype());
        if (!count)
            return output;
        const auto positions = points.contiguous();
        const auto source = values.contiguous();
        const auto references = radii ? local_radii.gt(radius * 0.5f)
                                      : internal::allocate_like(points, TensorShape{count}, DataType::Bool, 1.0f);
        const size_t buckets = std::bit_ceil(count);
        const auto bucket_mask = static_cast<uint32_t>(buckets - 1);
        if (points.device() == Device::GPU) {
            auto heads = internal::allocate_like(points, TensorShape{buckets}, DataType::Int32, -1.0f);
            auto next = internal::allocate_like(points, TensorShape{count}, DataType::Int32);
            pin_operands({&positions, &source, &references, &heads, &next, radii ? &local_radii : &source});
            const auto stream = prepare_inputs_for_stream({&positions, &source, &references, &heads, &next, radii ? &local_radii : &source}, output.stream());
            internal::backend_ops_for(positions).radius_neighbor_min(
                internal::storage_ref(positions), internal::storage_ref(source), internal::storage_ref(references),
                internal::storage_ref(heads), internal::storage_ref(next), internal::storage_ref(output),
                count, buckets, radius, radii ? std::optional(internal::storage_ref(local_radii)) : std::nullopt,
                internal::ExecContext{stream});
            return output;
        }
        const auto* xyz = positions.ptr<float>();
        std::vector<int32_t> heads(buckets, -1), next(count, -1);
        for (size_t i = 0; i < count; ++i) {
            const auto* p = xyz + i * 3;
            if (!finite_point(p) || (radii && !(local_radii.ptr<float>()[i] > radius * 0.5f)))
                continue;
            const auto bucket = hash_cell(cell(p[0], radius), cell(p[1], radius), cell(p[2], radius), bucket_mask);
            next[i] = heads[bucket];
            heads[bucket] = static_cast<int32_t>(i);
        }
        if (values.dtype() == DataType::Int32) {
            const auto* input = source.ptr<int32_t>();
            auto* result = output.ptr<int32_t>();
            for (size_t i = 0; i < count; ++i)
                result[i] = pointNeighborMin(xyz, input, heads.data(), next.data(), i, bucket_mask, radius, radii ? local_radii.ptr<float>() : nullptr);
        } else {
            const auto* input = source.ptr<float>();
            auto* result = output.ptr<float>();
            for (size_t i = 0; i < count; ++i)
                result[i] = pointNeighborMin(xyz, input, heads.data(), next.data(), i, bucket_mask, radius, radii ? local_radii.ptr<float>() : nullptr);
        }
        return output;
    }

    namespace {
        // Minimum-label propagation, shortcut to a fixed point after every round, for backends without a
        // connected-components kernel. Labels only move within a component and every round lowers or keeps
        // them, so a round that changes nothing leaves each component on its smallest index.
        Tensor propagate_components(const Tensor& points, const Tensor& own, const Tensor& selected, const float radius) {
            // Unselected points carry a label above every index, so they never lower a selected one.
            const auto sentinel = Tensor::full({own.numel()}, 1'000'000'000.0f, own.device(), DataType::Int32);
            auto labels = selected.is_valid() ? Tensor::where(selected, own, sentinel) : own;
            for (;;) {
                auto next = radius_neighbor_min(points, labels, radius);
                if (selected.is_valid())
                    next = Tensor::where(selected, next, sentinel);
                for (;;) {
                    const auto index = selected.is_valid() ? Tensor::where(selected, next, own) : next;
                    auto jumped = next.index_select(0, index);
                    if (jumped.ne(next).count_nonzero() == 0)
                        break;
                    next = std::move(jumped);
                }
                if (next.ne(labels).count_nonzero() == 0)
                    return selected.is_valid() ? Tensor::where(selected, next, own) : next;
                labels = std::move(next);
            }
        }
    } // namespace

    Tensor radius_connected_components(const Tensor& points, const float radius, const Tensor& selected) {
        LFS_ASSERT_MSG(points.is_valid() && points.ndim() == 2 && points.size(1) == 3 &&
                           points.dtype() == DataType::Float32,
                       std::format("radius_connected_components requires Float32 [N,3] points (valid={}, rank={}, columns={}, dtype={})",
                                   points.is_valid(), points.ndim(), points.ndim() == 2 ? points.size(1) : 0,
                                   points.is_valid() ? static_cast<int>(points.dtype()) : -1));
        LFS_ASSERT_MSG(std::isnormal(radius) && radius > 0.0f,
                       std::format("radius_connected_components radius must be positive, finite and normal (radius={})", radius));
        const size_t count = points.size(0);
        LFS_ASSERT_MSG(count <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
                       std::format("radius_connected_components point count exceeds int32 (count={})", count));
        if (selected.is_valid()) {
            LFS_ASSERT_MSG(selected.ndim() == 1 && selected.numel() == count && selected.dtype() == DataType::Bool &&
                               selected.device() == points.device(),
                           "radius_connected_components selection must be a Bool [N] tensor on the points' device");
            internal::require_same_gpu_backend(points, selected, "radius_connected_components");
        }
        auto labels = (internal::allocate_like(points, TensorShape{count}, DataType::Int32, 1.0f).cumsum(0) - 1)
                          .to(DataType::Int32)
                          .contiguous();
        if (!count)
            return labels;
        const auto positions = points.contiguous();
        const auto references = selected.is_valid()
                                    ? selected.contiguous()
                                    : internal::allocate_like(points, TensorShape{count}, DataType::Bool, 1.0f);
        const size_t buckets = std::bit_ceil(count);
        const auto bucket_mask = static_cast<uint32_t>(buckets - 1);
        if (points.device() == Device::GPU) {
            auto heads = internal::allocate_like(points, TensorShape{buckets}, DataType::Int32, -1.0f);
            auto next = internal::allocate_like(points, TensorShape{count}, DataType::Int32);
            pin_operands({&positions, &references, &heads, &next, &labels});
            const auto stream = prepare_inputs_for_stream({&positions, &references, &heads, &next}, labels.stream());
            if (internal::backend_ops_for(positions).radius_connected_components(
                    internal::storage_ref(positions), internal::storage_ref(references), internal::storage_ref(heads),
                    internal::storage_ref(next), internal::storage_ref(labels), count, buckets, radius,
                    internal::ExecContext{stream}))
                return labels;
            return propagate_components(positions, labels, selected, radius);
        }
        const auto* xyz = positions.ptr<float>();
        const auto* use = references.ptr<bool>();
        std::vector<int32_t> heads(buckets, -1), next(count, -1);
        for (size_t i = 0; i < count; ++i) {
            const auto* p = xyz + i * 3;
            if (!use[i] || !finite_point(p))
                continue;
            const auto bucket = hash_cell(cell(p[0], radius), cell(p[1], radius), cell(p[2], radius), bucket_mask);
            next[i] = heads[bucket];
            heads[bucket] = static_cast<int32_t>(i);
        }
        auto* parent = labels.ptr<int32_t>();
        for (size_t i = 0; i < count; ++i) {
            if (!use[i])
                continue;
            forEachRadiusNeighbor(xyz, heads.data(), next.data(), i, bucket_mask, radius, [&](const int32_t j) {
                if (static_cast<size_t>(j) <= i)
                    return;
                const int32_t a = componentRoot(parent, static_cast<int32_t>(i));
                const int32_t b = componentRoot(parent, j);
                if (a < b)
                    parent[b] = a;
                else if (b < a)
                    parent[a] = b;
            });
        }
        for (size_t i = 0; i < count; ++i)
            parent[i] = componentRoot(parent, static_cast<int32_t>(i));
        return labels;
    }

    Tensor radius_connected_components(const Tensor& points, const float radius) {
        return radius_connected_components(points, radius, Tensor{});
    }

    Tensor mutual_radius_components(const Tensor& points, const Tensor& radii) {
        LFS_ASSERT_MSG(points.is_valid() && points.ndim() == 2 && points.size(1) == 3 && points.dtype() == DataType::Float32,
                       "mutual_radius_components requires Float32 [N,3] points");
        const size_t count = points.size(0);
        LFS_ASSERT_MSG(count <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
                       "mutual_radius_components point count exceeds int32");
        LFS_ASSERT_MSG(radii.is_valid() && radii.ndim() == 1 && radii.numel() == count && radii.dtype() == DataType::Float32 &&
                           radii.device() == points.device(),
                       "mutual_radius_components requires Float32 [N] radii on the points' device");
        internal::require_same_gpu_backend(points, radii, "mutual_radius_components");
        auto labels = (internal::allocate_like(points, TensorShape{count}, DataType::Int32, 1.0f).cumsum(0) - 1)
                          .to(DataType::Int32)
                          .contiguous();
        if (!count)
            return labels;
        const auto usable = radii.isfinite().logical_and(radii.gt(0.0f));
        const auto tree = build_point_tree(points, usable);
        if (tree.program.references == 0)
            return labels;
        const auto positions = points.contiguous();
        const auto own_radii = Tensor::where(usable, radii, Tensor::zeros_like(radii)).contiguous();
        const auto sorted_radii = own_radii.index_select(0, tree.visit.slice(0, 0, tree.program.references)).contiguous();
        const auto box_radii = box_maxima(tree, sorted_radii);
        if (points.device() == Device::GPU) {
            pin_operands({&positions, &tree.sorted, &tree.boxes, &box_radii, &tree.visit, &sorted_radii, &own_radii, &labels});
            const auto stream = prepare_inputs_for_stream({&positions, &tree.sorted, &tree.boxes, &box_radii, &tree.visit,
                                                           &sorted_radii, &own_radii},
                                                          labels.stream());
            if (internal::backend_ops_for(positions).point_tree_components(
                    internal::storage_ref(positions), internal::storage_ref(tree.sorted), internal::storage_ref(tree.boxes),
                    internal::storage_ref(box_radii), internal::storage_ref(tree.visit), internal::storage_ref(sorted_radii),
                    internal::storage_ref(own_radii), internal::storage_ref(labels), tree.program, internal::ExecContext{stream}))
                return labels;
            // No kernel on this backend: join on the host.
            const auto host = mutual_radius_components(positions.cpu(), radii.cpu());
            GpuBackendScope scope(gpu_backend_of(positions).value());
            return host.to(Device::GPU);
        }
        const auto* xyz = positions.ptr<float>();
        const auto* sorted = tree.sorted.ptr<float>();
        const auto* boxes = tree.boxes.ptr<float>();
        const auto* reach = box_radii.ptr<float>();
        const auto* sorted_reach = sorted_radii.ptr<float>();
        const auto* visit = tree.visit.ptr<int32_t>();
        const auto* radius = own_radii.ptr<float>();
        auto* parent = labels.ptr<int32_t>();
        for (uint32_t t = 0; t < tree.program.references; ++t) {
            const auto self = visit[t];
            pointTreeMutualNeighbors(sorted, boxes, reach, sorted_reach, tree.program, xyz + size_t(self) * 3, t,
                                     radius[self], [&](const uint64_t j) {
                                         const int32_t a = componentRoot(parent, self);
                                         const int32_t b = componentRoot(parent, visit[j]);
                                         if (a < b)
                                             parent[b] = a;
                                         else if (b < a)
                                             parent[a] = b;
                                     });
        }
        for (size_t i = 0; i < count; ++i)
            parent[i] = componentRoot(parent, static_cast<int32_t>(i));
        return labels;
    }

    struct TriangleRayIndex::Tree {
        Tensor triangles; // Float32 [F,9]: a, b - a, c - a in tree order
        Tensor boxes;     // Float32 [nodes,6]: low and high in the ray frame
        PointTreeProgram program;
    };

    TriangleRayIndex::TriangleRayIndex(const Tensor& vertices, const Tensor& indices) {
        LFS_ASSERT_MSG(vertices.is_valid() && vertices.ndim() == 2 && vertices.size(1) == 3 &&
                           vertices.dtype() == DataType::Float32,
                       "TriangleRayIndex requires Float32 [V,3] vertices");
        LFS_ASSERT_MSG(indices.is_valid() && indices.ndim() == 2 && indices.size(1) == 3 &&
                           indices.dtype() == DataType::Int32 && indices.device() == vertices.device(),
                       "TriangleRayIndex requires Int32 [F,3] indices on the vertices' device");
        internal::require_same_gpu_backend(vertices, indices, "TriangleRayIndex");
        LFS_ASSERT_MSG(indices.size(0) <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
                       "TriangleRayIndex triangle count exceeds int32");
        auto tree = std::make_shared<Tree>();
        const size_t faces = indices.size(0);
        if (faces == 0 || vertices.size(0) == 0) {
            tree_ = std::move(tree);
            return;
        }
        auto corners = vertices.index_select(0, indices.flatten()).reshape({-1, 3, 3});
        const auto kept = corners.isfinite().reshape({-1, 9}).all(1).nonzero().reshape({-1}).to(DataType::Int32);
        tree->program.references = static_cast<uint32_t>(kept.numel());
        if (tree->program.references == 0) {
            tree_ = std::move(tree);
            return;
        }
        corners = corners.index_select(0, kept);
        // Row vectors times the transposed frame give (U, V, D) coordinates.
        std::vector<float> frame(9);
        for (int row = 0; row < 3; ++row)
            for (int column = 0; column < 3; ++column)
                frame[column * 3 + row] = kRayFrame[row * 3 + column];
        const auto projected =
            corners.reshape({-1, 3})
                .matmul(Tensor::from_vector(frame, {3, 3}, Device::CPU).to(vertices.device()))
                .reshape({-1, 3, 3});
        // Widened by more than the rounding of the projection and of the crossing test.
        const auto margin = (projected.abs().max() + 1.0f) * 1e-5f;
        const auto low = projected.min(1) - margin;
        const auto high = projected.max(1) + margin;
        // Tree order follows the box centres across the ray, the plane the queries search.
        const auto centres = (low + high) * Tensor::from_vector({0.5f, 0.5f, 0.0f}, {1, 3}, Device::CPU).to(vertices.device());
        const auto order = vertices.device() == Device::GPU ? morton30_order(centres)
                                                            : morton_sort_indices(centres).to(DataType::Int32);
        const auto a = corners.slice(1, 0, 1).squeeze(1);
        tree->triangles = Tensor::cat({a, corners.slice(1, 1, 2).squeeze(1) - a, corners.slice(1, 2, 3).squeeze(1) - a}, 1)
                              .index_select(0, order)
                              .contiguous();
        std::vector<Tensor> levels;
        auto grouped = fanout_groups(Tensor::cat({low, high}, 1).index_select(0, order));
        do {
            levels.push_back(Tensor::cat({grouped.slice(2, 0, 3).min(1), grouped.slice(2, 3, 6).max(1)}, 1));
            if (levels.back().size(0) <= kPointTreeFanout)
                break;
            grouped = fanout_groups(levels.back());
        } while (true);
        LFS_ASSERT_MSG(levels.size() <= kPointTreeMaxLevels, "triangle tree exceeds its level limit");
        tree->program.levels = static_cast<uint32_t>(levels.size());
        uint32_t offset = 0;
        for (size_t level = 0; level < levels.size(); ++level) {
            tree->program.level_offset[level] = offset;
            tree->program.level_count[level] = static_cast<uint32_t>(levels[level].size(0));
            offset += tree->program.level_count[level];
        }
        tree->boxes = Tensor::cat(levels, 0).contiguous();
        tree_ = std::move(tree);
    }

    Tensor TriangleRayIndex::odd_crossings(const Tensor& points) const {
        LFS_ASSERT_MSG(points.is_valid() && points.ndim() == 2 && points.size(1) == 3 && points.dtype() == DataType::Float32,
                       "odd_crossings requires Float32 [N,3] points");
        const size_t count = points.size(0);
        LFS_ASSERT_MSG(count <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
                       "odd_crossings point count exceeds int32");
        auto output = internal::allocate_like(points, TensorShape{count}, DataType::Int32, 0.0f);
        const auto& tree = *tree_;
        if (count == 0 || tree.program.references == 0)
            return output.ne(0);
        LFS_ASSERT_MSG(points.device() == tree.triangles.device(), "odd_crossings requires points on the index's device");
        internal::require_same_gpu_backend(points, tree.triangles, "odd_crossings");
        auto program = tree.program;
        program.points = static_cast<uint32_t>(count);
        const auto positions = points.contiguous();
        if (points.device() == Device::GPU) {
            // Neighbouring queries walk the same branches.
            const auto visit = count >= 4096 ? morton30_order(positions)
                                             : (Tensor::ones({count}, points.device(), DataType::Int32).cumsum(0) - 1)
                                                   .to(DataType::Int32)
                                                   .contiguous();
            pin_operands({&positions, &visit, &tree.triangles, &tree.boxes, &output});
            const auto stream = prepare_inputs_for_stream({&positions, &visit, &tree.triangles, &tree.boxes},
                                                          output.stream());
            if (internal::backend_ops_for(positions).triangle_tree_parity(
                    internal::storage_ref(positions), internal::storage_ref(visit), internal::storage_ref(tree.triangles),
                    internal::storage_ref(tree.boxes), internal::storage_ref(output), program,
                    internal::ExecContext{stream}))
                return output.ne(0);
        }
        // CPU, or no kernel on this backend: traverse on the host.
        const auto host_points = positions.cpu();
        const auto host_triangles = tree.triangles.cpu();
        const auto host_boxes = tree.boxes.cpu();
        auto parity = Tensor::zeros({count}, Device::CPU, DataType::Int32);
        const auto* p = host_points.ptr<float>();
        const auto* triangles = host_triangles.ptr<float>();
        const auto* boxes = host_boxes.ptr<float>();
        auto* result = parity.ptr<int32_t>();
        const auto total = static_cast<int64_t>(count);
#pragma omp parallel for schedule(dynamic, 256)
        for (int64_t i = 0; i < total; ++i)
            result[i] = triangleTreeParity(triangles, boxes, program, p + i * 3);
        if (points.device() == Device::CPU)
            return parity.ne(0);
        GpuBackendScope scope(gpu_backend_of(positions).value());
        return parity.to(Device::GPU).ne(0);
    }

    Tensor point_neighbor_spacing(const Tensor& points, const float cell_width) {
        LFS_ASSERT_MSG(points.is_valid() && points.ndim() == 2,
                       std::format("point_neighbor_spacing requires rank-2 points (valid={}, rank={})", points.is_valid(), points.ndim()));
        const auto everything = internal::allocate_like(points, {points.size(0)}, DataType::Bool, 1.0f);
        const auto grid = [&] { return radius_query(points, everything, cell_width, true, nullptr, 0, true); };
        LFS_ASSERT_MSG(points.size(1) == 3 && points.dtype() == DataType::Float32 && std::isnormal(cell_width) && cell_width > 0.0f,
                       std::format("point_neighbor_spacing requires Float32 [N,3] points and a positive normal cell width (width={})",
                                   cell_width));
        const size_t count = points.size(0);
        LFS_ASSERT_MSG(count <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
                       "point_neighbor_spacing point count exceeds int32");
        auto output = internal::allocate_like(points, TensorShape{count}, DataType::Float32, 0.0f);
        if (count == 0)
            return output;
        auto tree = build_point_tree(points, everything);
        if (tree.program.references == 0)
            return output;
        tree.program.radius = cell_width;
        const auto positions = points.contiguous();
        if (points.device() == Device::GPU) {
            pin_operands({&positions, &tree.sorted, &tree.boxes, &tree.visit, &output});
            const auto stream = prepare_inputs_for_stream({&positions, &tree.sorted, &tree.boxes, &tree.visit}, output.stream());
            if (internal::backend_ops_for(positions).point_tree_spacing(
                    internal::storage_ref(positions), internal::storage_ref(tree.sorted), internal::storage_ref(tree.boxes),
                    internal::storage_ref(tree.visit), internal::storage_ref(output), tree.program,
                    internal::ExecContext{stream}))
                return output;
            return grid();
        }
        const auto* xyz = positions.ptr<float>();
        const auto* sorted = tree.sorted.ptr<float>();
        const auto* boxes = tree.boxes.ptr<float>();
        const auto* visit = tree.visit.ptr<int32_t>();
        auto* result = output.ptr<float>();
        const auto references = static_cast<int64_t>(tree.program.references);
        const auto total = static_cast<int64_t>(count);
#pragma omp parallel for schedule(dynamic, 1024)
        for (int64_t t = 0; t < total; ++t) {
            const auto i = static_cast<size_t>(visit[t]);
            result[i] = pointTreeSpacing(sorted, boxes, tree.program, xyz + i * 3, t < references ? t : -1, cell_width);
        }
        return output;
    }
} // namespace lfs::core
