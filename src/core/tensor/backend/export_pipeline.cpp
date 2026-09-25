/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "export_pipeline.hpp"

#include "core/assert.hpp"
#include "core/decimate/math.hpp"
#include "core/tensor_backend.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <unordered_map>

namespace lfs::core::internal {
    namespace {
        struct MortPush {
            uint64_t positions, partials, keys;
            uint32_t n, blocks, padding;
            float min_x, min_y, min_z, mul_x, mul_y, mul_z;
            uint32_t tail;
        };
        static_assert(sizeof(MortPush) == 64);
        struct RadixPush {
            uint64_t keys_in, keys_out, index_in, index_out, counts, scratch;
            uint32_t n, blocks, shift, padding;
        };
        static_assert(sizeof(RadixPush) == 64);
        struct KPush {
            uint64_t sh, centroids, indices, labels, sums, counts, norms;
            uint32_t n, k, dims, slots, seed, padding;
        };
        static_assert(sizeof(KPush) == 80);
        struct ScreenPush {
            uint64_t sh, centroids, half_centroids, norms, labels, point_order, centroid_order;
            uint32_t n, k, slots, have_labels;
        };
        static_assert(sizeof(ScreenPush) == 72);
        struct DecPush {
            uint64_t pos, order, boxes, neighbors;
            uint32_t n, leaves, begin, padding;
        };
        static_assert(sizeof(DecPush) == 48);
        struct CostPush {
            uint64_t pos, rot, scale, opacity, dc, sh, neighbors, cache, idx, cost;
            uint32_t n, rest, pad0, pad1;
        };
        static_assert(sizeof(CostPush) == 96);
        struct MergePush {
            uint64_t table;
            uint32_t n, rest, pad0, pad1;
        };
        static_assert(sizeof(MergePush) == 24);
        struct GroupPush {
            uint64_t sh, centroids, norms, labels, sorted, group_offsets, task_offsets, members, member_offsets, nearest;
            uint32_t n, slots;
        };
        static_assert(sizeof(GroupPush) == 88);

        // Workgroups of the per-block Morton bounds; every backend dispatches at least this many.
        constexpr size_t kMaxBoundBlocks = 65535;

        template <class Push>
        void launch(ExportKernels& kernels, const char* module, const uint32_t phase, const Push& push,
                    const std::vector<StorageRef>& reads, const std::vector<StorageRef>& writes, const size_t work) {
            kernels.launch(module, phase, std::as_bytes(std::span(&push, 1)), reads, writes, work);
        }

        std::vector<int> sample_unique(int n, int count, unsigned seed) {
            std::vector<int> indices(static_cast<size_t>(count));
            std::unordered_map<int, int> swaps;
            auto value_for = [&](int index) {
                const auto it = swaps.find(index);
                return it == swaps.end() ? index : it->second;
            };
            std::mt19937 rng(seed);
            for (int i = 0; i < count; ++i) {
                std::uniform_int_distribution<int> dist(i, n - 1);
                const int pick = dist(rng);
                indices[static_cast<size_t>(i)] = value_for(pick);
                swaps[pick] = value_for(i);
            }
            return indices;
        }

        void kmeans_pass(ExportKernels& kernels, const Tensor& sh, const Tensor& centroids, const Tensor& indices,
                         Tensor& labels, Tensor& sums, Tensor& counts, const Tensor& norms, uint32_t n, uint32_t k,
                         uint32_t dims, uint32_t slots, uint32_t phase, uint32_t seed) {
            if (phase == 2 && dims == 45)
                phase = 11;
            if (phase == 7 && dims == 45)
                phase = 16;
            const auto address = [&](const Tensor& t) { return t.is_valid() ? kernels.address(t) : uint64_t{0}; };
            KPush push{kernels.address(sh), kernels.address(centroids), address(indices), address(labels),
                       address(sums), address(counts), address(norms), n, k, dims, slots, seed, 0};
            const bool over_points = phase == 2 || phase == 3 || phase == 7 || phase == 8 || phase == 11 || phase == 16 || phase == 13 || phase == 17;
            const size_t work = over_points ? size_t(n) : size_t(k);
            std::vector<StorageRef> reads{storage_ref(sh), storage_ref(centroids)};
            std::vector<StorageRef> writes;
            if (phase == 0 || phase == 9 || phase == 15 || phase == 17)
                reads.push_back(storage_ref(indices));
            if (phase == 2 || phase == 7 || phase == 11 || phase == 16 || phase == 14 || phase == 20)
                reads.push_back(storage_ref(norms));
            if (phase == 3 || phase == 8)
                reads.push_back(storage_ref(labels));
            if (phase == 19)
                reads.push_back(storage_ref(indices));
            if (phase == 21) {
                reads.push_back(storage_ref(indices));
                reads.push_back(storage_ref(labels));
            }
            if (phase == 4 || phase == 10) {
                reads.push_back(storage_ref(sums));
                reads.push_back(storage_ref(counts));
            }
            if (phase == 0 || phase == 4 || phase == 15 || phase == 9 || phase == 10)
                writes.push_back(storage_ref(phase == 15 ? sums : centroids));
            if (phase == 1)
                writes.push_back(storage_ref(norms));
            if (phase == 2 || phase == 7 || phase == 11 || phase == 16 || phase == 18 || phase == 19 || phase == 20)
                writes.push_back(storage_ref(labels));
            if (phase == 3 || phase == 8 || phase == 13 || phase == 14 || phase == 17 || phase == 21) {
                writes.push_back(storage_ref(sums));
                writes.push_back(storage_ref(counts));
            }
            const char* module = phase == 3 || phase == 8 ? "export_kmeans_accumulate"
                                 : phase == 15            ? "export_kmeans_pack"
                                                          : "export_kmeans";
            launch(kernels, module, phase, push, reads, writes, std::max<size_t>(work, 1));
        }

        // An even number of 4-bit passes lands sorted pairs in the caller's buffers.
        void radix_sort_pairs(ExportKernels& kernels, Tensor& keys, Tensor& indices, uint32_t bits = 64) {
            LFS_ASSERT(bits > 0 && bits <= 64 && bits % 8 == 0);
            const uint32_t n = static_cast<uint32_t>(keys.size(0));
            if (n <= 1)
                return;
            const uint32_t radix_blocks = (n + 2047u) / 2048u;
            const uint32_t ncounts = radix_blocks * 16u;
            auto counts = Tensor::empty({ncounts}, Device::GPU, DataType::UInt32);
            auto keys_b = Tensor::empty({n}, Device::GPU, DataType::Int64);
            auto index_b = Tensor::empty({n}, Device::GPU, DataType::Int32);
            Tensor* src_k = &keys;
            Tensor* dst_k = &keys_b;
            Tensor* src_i = &indices;
            Tensor* dst_i = &index_b;
            const uint64_t counts_addr = kernels.address(counts);
            for (uint32_t shift = 0; shift < bits; shift += 4) {
                const RadixPush radix{kernels.address(*src_k), kernels.address(*dst_k), kernels.address(*src_i),
                                      kernels.address(*dst_i), counts_addr, 0, n, radix_blocks, shift, 0};
                launch(kernels, "export_radix", 0, radix, {storage_ref(*src_k)}, {storage_ref(counts)},
                       size_t(radix_blocks) * 256u);
                launch(kernels, "export_radix", 2, radix, {storage_ref(counts)}, {storage_ref(counts)}, 256);
                launch(kernels, "export_radix", 1, radix, {storage_ref(*src_k), storage_ref(*src_i), storage_ref(counts)},
                       {storage_ref(*dst_k), storage_ref(*dst_i)}, size_t(radix_blocks) * 256u);
                std::swap(src_k, dst_k);
                std::swap(src_i, dst_i);
            }
        }

        // Per-label sums and counts. Without float atomics the points are sorted by label
        // and each run is summed in order, which needs no optional shader capability.
        void accumulate(ExportKernels& kernels, const Tensor& points, const Tensor& centroids, Tensor& labels,
                        Tensor& sums, Tensor& counts, uint32_t n, uint32_t k, uint32_t dims, uint32_t slots, bool dense) {
            auto dummy = Tensor::empty({1}, Device::GPU, DataType::Float32);
            if (kernels.float_atomics()) {
                kmeans_pass(kernels, points, centroids, dummy, labels, sums, counts, dummy, n, k, dims, slots, dense ? 8 : 3, 0);
                return;
            }
            auto keys = Tensor::empty({std::max<uint32_t>(n, 1)}, Device::GPU, DataType::Int64);
            auto order = Tensor::empty({std::max<uint32_t>(n, 1)}, Device::GPU, DataType::Int32);
            kmeans_pass(kernels, points, centroids, labels, dummy, keys, order, dummy, n, k, dims, slots, 17, 0);
            radix_sort_pairs(kernels, keys, order, 32);
            auto offsets = Tensor::empty({size_t(k) + 1u}, Device::GPU, DataType::Int32);
            kmeans_pass(kernels, keys, centroids, dummy, offsets, dummy, dummy, dummy, n, k + 1u, dims, slots, 18, 0);
            kmeans_pass(kernels, points, centroids, order, offsets, sums, counts, dummy, n, k, dims, slots, 21, dense ? 1u : 0u);
        }

        void super_step(ExportKernels& kernels, const Tensor& points, Tensor& supers, Tensor& membership, Tensor& norms,
                        uint32_t n, uint32_t dims, uint32_t seed) {
            constexpr uint32_t k = 256;
            auto dummy = Tensor::empty({1}, Device::GPU, DataType::Float32);
            kmeans_pass(kernels, points, supers, dummy, membership, dummy, dummy, norms, n, k, dims, (dims + 3) / 4, 1, 0);
            kmeans_pass(kernels, points, supers, dummy, membership, dummy, dummy, norms, n, k, dims, (dims + 3) / 4, 7, 0);
            auto sums = Tensor::zeros({k, dims}, Device::GPU, DataType::Float32);
            auto counts = Tensor::zeros({k}, Device::GPU, DataType::Int32);
            accumulate(kernels, points, supers, membership, sums, counts, n, k, dims, (dims + 3) / 4, true);
            kmeans_pass(kernels, points, supers, dummy, membership, sums, counts, norms, n, k, dims, (dims + 3) / 4, 10, seed);
        }

        // Points that share a super are scored against that super and its three nearest.
        void assign_grouped(ExportKernels& kernels, const Tensor& sh, const Tensor& centroids, const Tensor& norms,
                            Tensor& labels, const Tensor& supers, const Tensor& centroid_supers,
                            const Tensor& point_supers, uint32_t n, uint32_t slots) {
            constexpr uint32_t super_count = 256;
            const auto palette = static_cast<uint32_t>(centroids.size(0));
            auto point_keys = Tensor::empty({n}, Device::GPU, DataType::Int64);
            auto point_order = Tensor::empty({n}, Device::GPU, DataType::Int32);
            auto member_keys = Tensor::empty({palette}, Device::GPU, DataType::Int64);
            auto members = Tensor::empty({palette}, Device::GPU, DataType::Int32);
            auto group_offsets = Tensor::empty({super_count + 1}, Device::GPU, DataType::Int32);
            auto member_offsets = Tensor::empty({super_count + 1}, Device::GPU, DataType::Int32);
            auto task_offsets = Tensor::empty({super_count + 1}, Device::GPU, DataType::Int32);
            auto nearest = Tensor::empty({super_count, 4}, Device::GPU, DataType::Int32);
            auto super_norms = Tensor::empty({super_count}, Device::GPU, DataType::Float32);
            auto dummy = Tensor::empty({1}, Device::GPU, DataType::Float32);
            kmeans_pass(kernels, sh, centroids, point_supers, dummy, point_keys, point_order, norms, n, palette, 45, slots, 17, 0);
            radix_sort_pairs(kernels, point_keys, point_order, 8);
            kmeans_pass(kernels, sh, centroids, centroid_supers, dummy, member_keys, members, norms, palette, palette, 45, slots, 17, 0);
            radix_sort_pairs(kernels, member_keys, members, 8);
            kmeans_pass(kernels, point_keys, centroids, dummy, group_offsets, dummy, dummy, norms, n, super_count + 1, 45, slots, 18, 0);
            kmeans_pass(kernels, member_keys, centroids, dummy, member_offsets, dummy, dummy, norms, palette, super_count + 1, 45, slots, 18, 0);
            kmeans_pass(kernels, sh, centroids, group_offsets, task_offsets, dummy, dummy, norms, n, 1, 45, slots, 19, 0);
            kmeans_pass(kernels, supers, supers, dummy, nearest, dummy, dummy, super_norms, super_count, super_count, 45, slots, 1, 0);
            kmeans_pass(kernels, supers, supers, dummy, nearest, dummy, dummy, super_norms, super_count, super_count, 45, slots, 20, 0);
            const GroupPush push{kernels.address(sh), kernels.address(centroids), kernels.address(norms),
                                 kernels.address(labels), kernels.address(point_order), kernels.address(group_offsets),
                                 kernels.address(task_offsets), kernels.address(members),
                                 kernels.address(member_offsets), kernels.address(nearest), n, slots};
            launch(kernels, "export_kmeans_group", 0, push,
                   {storage_ref(sh), storage_ref(centroids), storage_ref(norms), storage_ref(point_order), storage_ref(group_offsets),
                    storage_ref(task_offsets), storage_ref(members), storage_ref(member_offsets), storage_ref(nearest)},
                   {storage_ref(labels)}, (size_t(n) / 256u + super_count) * 256u);
        }
    } // namespace

    Tensor export_morton_sort(ExportKernels& kernels, const Tensor& positions, Tensor* sorted_keys) {
        const GpuBackendScope scope(kernels.backend());
        const uint32_t n = static_cast<uint32_t>(positions.size(0));
        auto keys = Tensor::empty({n}, Device::GPU, DataType::Int64);
        auto indices = Tensor::empty({n}, Device::GPU, DataType::Int32);
        if (n == 0) {
            if (sorted_keys)
                *sorted_keys = keys;
            return indices;
        }
        const auto blocks = static_cast<uint32_t>(std::min<size_t>((size_t(n) + 255u) / 256u, kMaxBoundBlocks));
        auto partials = Tensor::empty({size_t(blocks) * 6u}, Device::GPU, DataType::Float32);
        auto bounds = Tensor::empty({6}, Device::GPU, DataType::Float32);
        MortPush push{kernels.address(positions), kernels.address(partials), kernels.address(bounds),
                      n, blocks, 0, 0, 0, 0, 0, 0, 0, 0};
        launch(kernels, "export_morton", 0, push, {storage_ref(positions)}, {storage_ref(partials)}, size_t(blocks) * 256u);
        launch(kernels, "export_morton", 1, push, {storage_ref(partials)}, {storage_ref(bounds)}, 256);
        const auto host_bounds = bounds.cpu();
        const float* b = host_bounds.ptr<float>();
        const float lens[3] = {b[3] - b[0], b[4] - b[1], b[5] - b[2]};
        push.min_x = b[0];
        push.min_y = b[1];
        push.min_z = b[2];
        push.mul_x = lens[0] == 0 ? 0 : float(1u << 21) / lens[0];
        push.mul_y = lens[1] == 0 ? 0 : float(1u << 21) / lens[1];
        push.mul_z = lens[2] == 0 ? 0 : float(1u << 21) / lens[2];
        push.keys = kernels.address(keys);
        launch(kernels, "export_morton", 2, push, {storage_ref(positions)}, {storage_ref(keys)}, n);
        std::vector<int> identity(n);
        std::iota(identity.begin(), identity.end(), 0);
        indices = Tensor::from_vector(identity, {n}, Device::GPU);
        radix_sort_pairs(kernels, keys, indices);
        if (sorted_keys)
            *sorted_keys = keys;
        return indices;
    }

    std::tuple<Tensor, Tensor> export_kmeans_sh(ExportKernels& kernels, const Tensor& sh, const int n_points,
                                                const int sh_coeffs, const int k, const int iterations) {
        const GpuBackendScope scope(kernels.backend());
        const uint32_t n = static_cast<uint32_t>(n_points);
        const uint32_t kk = static_cast<uint32_t>(k);
        const uint32_t dims = static_cast<uint32_t>(sh_coeffs * 3);
        const uint32_t slots = (dims + 3u) / 4u;
        auto labels = Tensor::empty({n}, Device::GPU, DataType::Int32);
        if (n <= kk) {
            // n <= k keeps one centroid per point. The returned palette is [n, dims].
            auto centroids = Tensor::zeros({n, dims}, Device::GPU, DataType::Float32);
            std::vector<int> id(n);
            std::iota(id.begin(), id.end(), 0);
            auto index = Tensor::from_vector(id, {n}, Device::GPU);
            kmeans_pass(kernels, sh, centroids, index, labels, centroids, centroids, centroids, n, n, dims, slots, 0, 0);
            return {centroids, index};
        }
        auto centroids = Tensor::zeros({kk, dims}, Device::GPU, DataType::Float32);
        auto perm = Tensor::from_vector(sample_unique(int(n), int(kk), std::random_device{}()), {kk}, Device::GPU);
        auto dummy = Tensor::empty({1}, Device::GPU, DataType::Float32);
        kmeans_pass(kernels, sh, centroids, perm, labels, dummy, dummy, dummy, n, kk, dims, slots, 0, 0);
        auto norms = Tensor::empty({kk}, Device::GPU, DataType::Float32);
        // Screening rejects centroids the FP32 argmin cannot pick. It is safe for the
        // exact final iteration, including when the caller did not ask for the fast path.
        const bool screen = dims == 45 && kernels.screened_assignment() && kk >= 32;
        const bool hierarchical = dims == 45 && kk >= 4096;
        Tensor supers;
        Tensor centroid_supers;
        Tensor super_norms;
        Tensor point_supers;
        if (hierarchical) {
            supers = Tensor::empty({256, dims}, Device::GPU, DataType::Float32);
            centroid_supers = Tensor::empty({kk}, Device::GPU, DataType::Int32);
            super_norms = Tensor::empty({256}, Device::GPU, DataType::Float32);
            point_supers = Tensor::empty({n}, Device::GPU, DataType::Int32);
            auto seeded = Tensor::from_vector(sample_unique(int(kk), 256, std::random_device{}()), {256}, Device::GPU);
            kmeans_pass(kernels, centroids, supers, seeded, centroid_supers, dummy, dummy, super_norms, kk, 256, dims, slots, 9, 0);
            for (int warm = 0; warm < 5; ++warm)
                super_step(kernels, centroids, supers, centroid_supers, super_norms, kk, dims, static_cast<uint32_t>(warm * 12345 + 67890));
        }
        for (int iter = 0; iter < iterations; ++iter) {
            const bool last = iter + 1 == iterations;
            if (hierarchical && !last) {
                if (iter > 0)
                    super_step(kernels, centroids, supers, centroid_supers, super_norms, kk, dims, static_cast<uint32_t>(iter * 111));
                kmeans_pass(kernels, sh, supers, dummy, point_supers, dummy, dummy, super_norms, n, 256, dims, slots, 1, 0);
                kmeans_pass(kernels, sh, supers, dummy, point_supers, dummy, dummy, super_norms, n, 256, dims, slots, 2, 0);
                kmeans_pass(kernels, sh, centroids, dummy, labels, dummy, dummy, norms, n, kk, dims, slots, 1, 0);
                assign_grouped(kernels, sh, centroids, norms, labels, supers, centroid_supers, point_supers, n, slots);
            } else if (screen && last) {
                kmeans_pass(kernels, sh, centroids, dummy, labels, dummy, dummy, norms, n, kk, dims, slots, 1, 0);
                export_assign_sh3(kernels, sh, centroids, norms, labels, true, iter > 0);
            } else {
                kmeans_pass(kernels, sh, centroids, dummy, labels, dummy, dummy, norms, n, kk, dims, slots, 1, 0);
                kmeans_pass(kernels, sh, centroids, dummy, labels, dummy, dummy, norms, n, kk, dims, slots, 2, 0);
            }
            auto sums = Tensor::zeros({kk, dims}, Device::GPU, DataType::Float32);
            auto counts = Tensor::zeros({kk}, Device::GPU, DataType::Int32);
            accumulate(kernels, sh, centroids, labels, sums, counts, n, kk, dims, slots, false);
            kmeans_pass(kernels, sh, centroids, dummy, labels, sums, counts, norms, n, kk, dims, slots, 4,
                        static_cast<uint32_t>(iter * 12345 + 67890));
        }
        return {centroids, labels};
    }

    void export_assign_sh3(ExportKernels& kernels, const Tensor& sh, const Tensor& centroids, const Tensor& norms,
                           Tensor& labels, const bool fast, const bool have_labels) {
        const GpuBackendScope scope(kernels.backend());
        const uint32_t n = static_cast<uint32_t>(labels.numel());
        const uint32_t k = static_cast<uint32_t>(centroids.size(0));
        if (!n)
            return;
        // Both modes are the FP32 argmin. Screening is an acceleration of the same winners.
        auto dummy = Tensor::empty({1}, Device::GPU, DataType::Float32);
        if (fast && k > 0 && kernels.screened_assignment()) {
            auto point_keys = Tensor::empty({n}, Device::GPU, DataType::Int64);
            auto point_order = Tensor::empty({n}, Device::GPU, DataType::Int32);
            kmeans_pass(kernels, sh, centroids, dummy, labels, point_keys, point_order, norms, n, k, 45, 12, 13, 0);
            radix_sort_pairs(kernels, point_keys, point_order, 16);
            auto centroid_keys = Tensor::empty({k}, Device::GPU, DataType::Int64);
            auto centroid_order = Tensor::empty({k}, Device::GPU, DataType::Int32);
            kmeans_pass(kernels, sh, centroids, dummy, labels, centroid_keys, centroid_order, norms, n, k, 45, 12, 14, 0);
            radix_sort_pairs(kernels, centroid_keys, centroid_order, 32);
            auto half = Tensor::zeros({((size_t(k) + 15u) / 16u) * 16u * 48u}, Device::GPU, DataType::Float16);
            kmeans_pass(kernels, sh, centroids, centroid_order, labels, half, dummy, norms, n, k, 45, 12, 15, 0);
            const ScreenPush sp{kernels.address(sh), kernels.address(centroids), kernels.address(half),
                                kernels.address(norms), kernels.address(labels), kernels.address(point_order),
                                kernels.address(centroid_order), n, k, 12, have_labels ? 1u : 0u};
            launch(kernels, "export_kmeans_screen", 0, sp,
                   {storage_ref(sh), storage_ref(centroids), storage_ref(half), storage_ref(norms), storage_ref(labels),
                    storage_ref(point_order), storage_ref(centroid_order)},
                   {storage_ref(labels)}, size_t(n) * 2u);
            return;
        }
        kmeans_pass(kernels, sh, centroids, dummy, labels, dummy, dummy, norms, n, k, 45, 12, 2, 0);
    }

    void export_decimate_candidates(ExportKernels& kernels, const Tensor& position, const Tensor& rotation,
                                    const Tensor& scale, const Tensor& opacity, const Tensor& dc, const Tensor& sh,
                                    const int rest, std::vector<uint32_t>& idx, std::vector<float>& cost) {
        const GpuBackendScope scope(kernels.backend());
        Tensor keys;
        auto order = export_morton_sort(kernels, position, &keys);
        const uint32_t n = static_cast<uint32_t>(position.size(0));
        uint32_t leaves = 1;
        while (leaves < (n + 7u) / 8u)
            leaves *= 2u;
        auto boxes = Tensor::empty({size_t(leaves) * 2u * 6u}, Device::GPU, DataType::Float32);
        auto neighbors = Tensor::empty({size_t(n) * 16u}, Device::GPU, DataType::UInt32);
        DecPush push{kernels.address(position), kernels.address(order), kernels.address(boxes),
                     kernels.address(neighbors), n, leaves, 0, 0};
        launch(kernels, "export_decimate", 0, push, {storage_ref(position), storage_ref(order)}, {storage_ref(boxes)}, leaves);
        for (uint32_t begin = leaves / 2u; begin > 0; begin /= 2u) {
            push.begin = begin;
            launch(kernels, "export_decimate", 1, push, {storage_ref(boxes)}, {storage_ref(boxes)}, begin);
        }
        launch(kernels, "export_decimate", 2, push, {storage_ref(position), storage_ref(order), storage_ref(boxes)},
               {storage_ref(neighbors)}, n);
        const auto bounds = boxes.slice(0, 6, 12).cpu();
        bool wide_coordinates = false;
        for (size_t a = 0; a < 6; ++a)
            wide_coordinates |= std::abs(bounds.ptr<float>()[a]) > 0x1p20f;
        // A determinant contains six coordinate factors. Beyond this bound its
        // intermediates can exceed float32 range even when the final cost is finite.
        if (wide_coordinates) {
            namespace dec = lfs::core::decimate;
            const dec::Data host{position.cpu(), rotation.cpu(), scale.cpu(), opacity.cpu(), dc.cpu(),
                                 rest ? sh.cpu() : Tensor{}, n, rest};
            const auto nearest = neighbors.cpu();
            std::vector<dec::Cache> reference(n);
            for (uint32_t i = 0; i < n; ++i)
                reference[i] = dec::cache_one(host.view(), i);
            idx.resize(size_t(n) * 4);
            cost.resize(size_t(n) * 4);
            for (uint32_t i = 0; i < n; ++i) {
                std::array<std::pair<float, uint32_t>, 16> edges;
                edges.fill({INFINITY, dec::invalid});
                for (uint32_t a = 0; a < 16; ++a) {
                    const uint32_t j = nearest.ptr<uint32_t>()[size_t(i) * 16 + a];
                    if (j == dec::invalid)
                        continue;
                    const float value = dec::edge(host.view(), reference.data(), i, j);
                    if (std::isfinite(value))
                        edges[a] = {value, j};
                }
                std::ranges::sort(edges);
                for (uint32_t a = 0; a < 4; ++a) {
                    idx[size_t(i) * 4 + a] = edges[a].second;
                    cost[size_t(i) * 4 + a] = edges[a].first;
                }
            }
            return;
        }
        auto chosen = Tensor::empty({size_t(n) * 4u}, Device::GPU, DataType::UInt32);
        auto scored = Tensor::empty({size_t(n) * 4u}, Device::GPU, DataType::Float32);
        auto cache = Tensor::empty({size_t(n) * 32u}, Device::GPU, DataType::Float32);
        const CostPush cost_push{kernels.address(position), kernels.address(rotation), kernels.address(scale),
                                 kernels.address(opacity), kernels.address(dc), rest ? kernels.address(sh) : 0,
                                 kernels.address(neighbors), kernels.address(cache), kernels.address(chosen),
                                 kernels.address(scored), n, static_cast<uint32_t>(std::max(rest, 0)), 0, 0};
        launch(kernels, "export_decimate_cost", 0, cost_push,
               {storage_ref(position), storage_ref(rotation), storage_ref(scale), storage_ref(opacity)},
               {storage_ref(cache)}, n);
        std::vector<StorageRef> reads{storage_ref(position), storage_ref(dc), storage_ref(neighbors), storage_ref(cache)};
        if (rest)
            reads.push_back(storage_ref(sh));
        launch(kernels, "export_decimate_cost", 1, cost_push, reads, {storage_ref(chosen), storage_ref(scored)}, n);
        idx.resize(size_t(n) * 4u);
        cost.resize(size_t(n) * 4u);
        auto& ops = backend_ops(kernels.backend());
        ops.copy_device_to_host(CopyRequest{
            .src = storage_ref(chosen),
            .dst = raw_storage_ref(idx.data()),
            .bytes = idx.size() * sizeof(uint32_t),
            .synchronous = true,
            .operation = "tensor.decimate.indices",
        });
        ops.copy_device_to_host(CopyRequest{
            .src = storage_ref(scored),
            .dst = raw_storage_ref(cost.data()),
            .bytes = cost.size() * sizeof(float),
            .synchronous = true,
            .operation = "tensor.decimate.costs",
        });
    }

    DecimateMerge export_decimate_merge(ExportKernels& kernels, const Tensor& position, const Tensor& rotation,
                                        const Tensor& scale, const Tensor& opacity, const Tensor& dc, const Tensor& sh,
                                        const int rest, const std::vector<int>& member_group,
                                        const std::vector<uint32_t>& minimum, const std::vector<uint32_t>& members,
                                        const std::vector<uint32_t>& offsets, const size_t removed) {
        const GpuBackendScope scope(kernels.backend());
        const uint32_t n = static_cast<uint32_t>(position.size(0));
        const size_t kept = n - removed;
        auto upload_u32 = [](const std::vector<uint32_t>& values) {
            const std::vector<uint32_t> dummy{0};
            const auto& src = values.empty() ? dummy : values;
            auto host = Tensor::empty({src.size()}, Device::CPU, DataType::UInt32);
            std::memcpy(host.ptr<uint32_t>(), src.data(), src.size() * sizeof(uint32_t));
            return host.to(Device::GPU);
        };
        auto upload_i32 = [](const std::vector<int>& values) {
            const std::vector<int> dummy{0};
            const auto& src = values.empty() ? dummy : values;
            auto host = Tensor::empty({src.size()}, Device::CPU, DataType::Int32);
            std::memcpy(host.ptr<int>(), src.data(), src.size() * sizeof(int));
            return host.to(Device::GPU);
        };
        // Exclusive output rows. A kept splat is ungrouped or the group's minimum index.
        std::vector<uint32_t> rows(n);
        uint32_t running = 0;
        for (uint32_t i = 0; i < n; ++i) {
            rows[i] = running;
            const int group = i < member_group.size() ? member_group[i] : -1;
            if (group < 0 || (size_t(group) < minimum.size() && i == minimum[size_t(group)]))
                ++running;
        }
        if (running != kept)
            throw std::runtime_error("decimation merge row count does not match the selection");
        auto out_pos = Tensor::empty({kept, 3}, Device::GPU);
        auto out_rot = Tensor::empty({kept, 4}, Device::GPU);
        auto out_scale = Tensor::empty({kept, 3}, Device::GPU);
        auto out_opacity = Tensor::empty({kept, 1}, Device::GPU);
        auto out_dc = Tensor::empty({kept, 1, 3}, Device::GPU);
        auto out_sh = Tensor::empty({kept, size_t(std::max(rest, 0)), 3}, Device::GPU);
        if (!n)
            return {out_pos, out_rot, out_scale, out_opacity, out_dc, out_sh};
        // Covariance products of coordinate differences leave the float-float range past the
        // candidate bound; the double reference merge keeps those intermediates finite.
        if (position.abs().max_scalar() > 0x1p20f) {
            namespace dec = lfs::core::decimate;
            const dec::Data host{position.cpu(), rotation.cpu(), scale.cpu(), opacity.cpu(), dc.cpu(),
                                 rest ? sh.cpu() : Tensor{}, n, rest};
            const auto merged = dec::allocate(kept, rest, Device::CPU);
            for (uint32_t i = 0; i < n; ++i) {
                const int group = i < member_group.size() ? member_group[i] : -1;
                if (group < 0)
                    dec::copy_one(host.view(), i, merged.view(), rows[i]);
                else if (i == minimum[size_t(group)])
                    dec::merge_one(host.view(), members.data() + offsets[size_t(group)],
                                   int(offsets[size_t(group) + 1] - offsets[size_t(group)]), merged.view(), rows[i]);
            }
            return {merged.pos.to(Device::GPU), merged.rot.to(Device::GPU), merged.scale.to(Device::GPU),
                    merged.opacity.to(Device::GPU), merged.dc.to(Device::GPU), merged.sh.to(Device::GPU)};
        }
        auto groups = upload_i32(member_group);
        auto minimum_gpu = upload_u32(minimum);
        auto members_gpu = upload_u32(members);
        auto offsets_gpu = upload_u32(offsets);
        auto rows_gpu = upload_u32(rows);
        uint64_t words[17] = {
            kernels.address(position), kernels.address(rotation), kernels.address(scale),
            kernels.address(opacity), kernels.address(dc),
            rest && sh.is_valid() ? kernels.address(sh) : 0,
            kernels.address(groups), kernels.address(minimum_gpu), kernels.address(members_gpu),
            kernels.address(offsets_gpu), kernels.address(rows_gpu), kernels.address(out_pos),
            kernels.address(out_rot), kernels.address(out_scale), kernels.address(out_opacity),
            kernels.address(out_dc), rest ? kernels.address(out_sh) : 0};
        auto table_host = Tensor::empty({17}, Device::CPU, DataType::Int64);
        std::memcpy(table_host.ptr<int64_t>(), words, sizeof(words));
        auto table = table_host.to(Device::GPU);
        const MergePush push{kernels.address(table), n, static_cast<uint32_t>(std::max(rest, 0)), 0, 0};
        std::vector<StorageRef> reads{storage_ref(table), storage_ref(position), storage_ref(rotation), storage_ref(scale),
                                      storage_ref(opacity), storage_ref(dc), storage_ref(groups), storage_ref(minimum_gpu),
                                      storage_ref(members_gpu), storage_ref(offsets_gpu), storage_ref(rows_gpu)};
        if (rest && sh.is_valid())
            reads.push_back(storage_ref(sh));
        std::vector<StorageRef> writes{storage_ref(out_pos), storage_ref(out_rot), storage_ref(out_scale),
                                       storage_ref(out_opacity), storage_ref(out_dc)};
        if (rest)
            writes.push_back(storage_ref(out_sh));
        launch(kernels, "export_decimate_merge", 0, push, reads, writes, n);
        // The table holds addresses of the temporaries above, which die with this frame.
        backend_ops(kernels.backend()).synchronize_device();
        return {out_pos, out_rot, out_scale, out_opacity, out_dc, out_sh};
    }
} // namespace lfs::core::internal
