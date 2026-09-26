/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/tensor_export.hpp"
#include "core/logger.hpp"

#include "core/morton.hpp"
#include "internal/tensor_impl.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <tbb/parallel_for.h>
#include <thread>

namespace lfs::core {
    namespace {
        Tensor morton_cpu(const Tensor& positions, Tensor* sorted_keys) {
            const auto host = positions.device() == Device::CPU ? positions.contiguous() : positions.cpu().contiguous();
            const int n = static_cast<int>(host.size(0));
            const float* p = host.ptr<float>();
            float low[3] = {INFINITY, INFINITY, INFINITY};
            float high[3] = {-INFINITY, -INFINITY, -INFINITY};
            for (int i = 0; i < n; ++i)
                for (int a = 0; a < 3; ++a) {
                    low[a] = std::min(low[a], p[i * 3 + a]);
                    high[a] = std::max(high[a], p[i * 3 + a]);
                }
            float mul[3];
            for (int a = 0; a < 3; ++a)
                mul[a] = morton_multiplier(high[a] - low[a]);
            std::vector<int32_t> order(n);
            std::vector<int64_t> keys(n);
            std::iota(order.begin(), order.end(), 0);
            for (int i = 0; i < n; ++i) {
                keys[i] = static_cast<int64_t>(morton_encode(
                    morton_coordinate(p[i * 3], low[0], mul[0]),
                    morton_coordinate(p[i * 3 + 1], low[1], mul[1]),
                    morton_coordinate(p[i * 3 + 2], low[2], mul[2])));
            }
            std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return keys[a] < keys[b]; });
            std::vector<int64_t> sorted(n);
            std::vector<int32_t> indices(n);
            for (int i = 0; i < n; ++i) {
                indices[i] = order[i];
                sorted[i] = keys[order[i]];
            }
            auto idx = Tensor::from_vector(std::vector<int>(indices.begin(), indices.end()), {static_cast<size_t>(n)}, Device::CPU);
            if (sorted_keys) {
                auto keys_tensor = Tensor::empty({static_cast<size_t>(n)}, Device::CPU, DataType::Int64);
                if (n > 0)
                    std::memcpy(keys_tensor.ptr<int64_t>(), sorted.data(), static_cast<size_t>(n) * sizeof(int64_t));
                *sorted_keys = std::move(keys_tensor);
            }
            return idx;
        }

        int nearest_centroid_1d(const std::vector<float>& centroids, float value, int hint) {
            auto it = centroids.begin();
            if (hint < 0 || !std::isfinite(value)) {
                it = std::lower_bound(centroids.begin(), centroids.end(), value);
            } else {
                it += hint;
                while (it != centroids.begin() && *(it - 1) >= value)
                    --it;
                while (it != centroids.end() && *it < value)
                    ++it;
            }
            int best = static_cast<int>(std::distance(centroids.begin(), it));
            if (best >= static_cast<int>(centroids.size()))
                best = static_cast<int>(centroids.size()) - 1;
            float best_dist = std::abs(value - centroids[best]);
            if (best > 0) {
                const float prev = std::abs(value - centroids[best - 1]);
                if (prev < best_dist) {
                    best -= 1;
                    best_dist = prev;
                }
            }
            if (best + 1 < static_cast<int>(centroids.size()) && std::abs(value - centroids[best + 1]) < best_dist)
                best += 1;
            return best;
        }

        // Floats in the swizzled SH layout: 32-point blocks of float4 slots.
        size_t swizzled_float_count(size_t points, int coeffs) {
            const size_t slots = (size_t(coeffs) * 3u + 3u) / 4u;
            return (points + 31u) / 32u * slots * 32u * 4u;
        }
        void require(const bool condition, const char* what) {
            if (!condition)
                throw std::invalid_argument(what);
        }
        void require_rows(const Tensor& t, size_t rows, size_t width, const char* what) {
            require(t.is_valid() && t.dtype() == DataType::Float32 && t.numel() == rows * width, what);
        }
        void validate_splats(const Tensor& position, const Tensor& rotation, const Tensor& scale, const Tensor& opacity,
                             const Tensor& dc, const Tensor& sh, int rest) {
            require(position.is_valid() && position.dtype() == DataType::Float32 && position.ndim() == 2 &&
                        position.size(1) == 3 && position.size(0) <= size_t(std::numeric_limits<int32_t>::max() / 4),
                    "decimation positions must be float32 [N,3] with N below 2^29");
            require(rest == 0 || rest == 3 || rest == 8 || rest == 15, "decimation SH rest count must be 0, 3, 8 or 15");
            const size_t n = position.size(0);
            require_rows(rotation, n, 4, "decimation rotation must be float32 [N,4]");
            require_rows(scale, n, 3, "decimation scale must be float32 [N,3]");
            require_rows(opacity, n, 1, "decimation opacity must be float32 [N]");
            require_rows(dc, n, 3, "decimation DC must be float32 [N,3]");
            if (rest)
                require_rows(sh, n, size_t(rest) * 3u, "decimation SH must be float32 [N,rest,3]");
            for (const Tensor* t : {&rotation, &scale, &opacity, &dc, &sh})
                require(!t->is_valid() || t->device() == position.device(), "decimation attributes must share the position device");
        }
    } // namespace

    Tensor morton_sort_indices(const Tensor& positions, Tensor* sorted_keys) {
        require(!positions.is_valid() || (positions.dtype() == DataType::Float32 && positions.ndim() == 2 &&
                                          positions.size(1) == 3 &&
                                          positions.size(0) <= size_t(std::numeric_limits<int32_t>::max())),
                "Morton positions must be float32 [N,3]");
        if (!positions.is_valid() || positions.device() == Device::CPU)
            return morton_cpu(positions, sorted_keys);
        return internal::backend_ops_for(positions).morton_sort(positions, sorted_keys, {});
    }

    std::tuple<Tensor, Tensor> kmeans_sh(const Tensor& shN_swizzled, int n_points, int sh_coeffs, int k, int iterations,
                                         bool fast_assignment) {
        if (!shN_swizzled.is_valid() || shN_swizzled.device() != Device::GPU || shN_swizzled.ndim() != 1 ||
            shN_swizzled.dtype() != DataType::Float32 || n_points <= 0 || k <= 0 ||
            (sh_coeffs != 3 && sh_coeffs != 8 && sh_coeffs != 15) ||
            shN_swizzled.numel() < swizzled_float_count(size_t(n_points), sh_coeffs)) {
            LOG_ERROR("kmeans_sh expects a GPU float32 swizzled SH tensor for {} points with 3, 8 or 15 coefficients "
                      "and a positive palette size",
                      n_points);
            return {Tensor(), Tensor()};
        }
        return internal::backend_ops_for(shN_swizzled).kmeans_sh(shN_swizzled, n_points, sh_coeffs, k, iterations, fast_assignment, {});
    }

    void assign_sh3(const Tensor& shN_swizzled, const Tensor& centroids, const Tensor& centroid_norms, Tensor& labels,
                    bool fast, bool have_labels) {
        require(labels.is_valid() && labels.dtype() == DataType::Int32 && labels.ndim() == 1,
                "assign_sh3 labels must be int32 [N]");
        require(shN_swizzled.is_valid() && shN_swizzled.device() == Device::GPU &&
                    shN_swizzled.dtype() == DataType::Float32 &&
                    shN_swizzled.numel() >= swizzled_float_count(labels.numel(), 15),
                "assign_sh3 needs a GPU float32 swizzled SH3 tensor covering every label");
        require(centroids.is_valid() && centroids.dtype() == DataType::Float32 && centroids.ndim() == 2 &&
                    centroids.size(1) == 45 && centroids.size(0) <= size_t(std::numeric_limits<int32_t>::max()),
                "assign_sh3 centroids must be float32 [K,45]");
        require(centroid_norms.is_valid() && centroid_norms.dtype() == DataType::Float32 &&
                    centroid_norms.numel() == centroids.size(0),
                "assign_sh3 norms must be float32 [K]");
        for (const Tensor* t : std::initializer_list<const Tensor*>{&centroids, &centroid_norms, &labels})
            require(t->device() == Device::GPU && internal::gpu_backend_tag(*t) == internal::gpu_backend_tag(shN_swizzled),
                    "assign_sh3 tensors must share one GPU backend");
        internal::backend_ops_for(shN_swizzled).assign_sh3(shN_swizzled, centroids, centroid_norms, labels, fast, have_labels, {});
    }

    ScalarCodebook cluster_scalar(const float* data, int num_rows, int num_columns, int iterations, bool pooled) {
        constexpr int K = 256;
        const size_t total_points = static_cast<size_t>(num_rows) * static_cast<size_t>(num_columns);
        float min_val = std::numeric_limits<float>::infinity();
        float max_val = -std::numeric_limits<float>::infinity();
        for (int col = 0; col < num_columns; ++col)
            for (int row = 0; row < num_rows; ++row) {
                const float value = data[row * num_columns + col];
                min_val = std::min(min_val, value);
                max_val = std::max(max_val, value);
            }
        std::vector<float> centroid_vals(K);
        const float step = (K > 1) ? (max_val - min_val) / (K - 1) : 0.0f;
        for (int i = 0; i < K; ++i)
            centroid_vals[i] = min_val + i * step;
        ScalarCodebook result;
        result.labels.assign(total_points, 0);
        struct LocalAccum {
            std::array<double, K> sums{};
            std::array<int64_t, K> counts{};
        };
        const unsigned hw_threads = std::max(1u, std::thread::hardware_concurrency());
        const size_t worker_count = std::max<size_t>(1, std::min<size_t>(hw_threads, (total_points + 65535) / 65536));
        bool use_hints = false;
        auto accumulate_range = [&](size_t begin, size_t end, bool write_labels, LocalAccum& accum) {
            for (size_t linear = begin; linear < end; ++linear) {
                const int col = static_cast<int>(linear / static_cast<size_t>(num_rows));
                const int row = static_cast<int>(linear - static_cast<size_t>(col) * static_cast<size_t>(num_rows));
                const float value = data[row * num_columns + col];
                const int label = nearest_centroid_1d(centroid_vals, value, use_hints ? result.labels[linear] : -1);
                if (write_labels || pooled)
                    result.labels[linear] = static_cast<uint8_t>(label);
                accum.sums[label] += static_cast<double>(value);
                accum.counts[label]++;
            }
        };
        const int effective = std::max(0, iterations);
        for (int iter = 0; iter < effective; ++iter) {
            std::vector<LocalAccum> accums(worker_count);
            const bool write_labels = iter == effective - 1;
            if (worker_count == 1) {
                accumulate_range(0, total_points, write_labels, accums[0]);
            } else if (pooled) {
                tbb::parallel_for(size_t{0}, worker_count, [&](size_t worker) {
                    accumulate_range(total_points * worker / worker_count, total_points * (worker + 1) / worker_count,
                                     write_labels, accums[worker]);
                });
            } else {
                std::vector<std::thread> workers;
                workers.reserve(worker_count);
                for (size_t worker = 0; worker < worker_count; ++worker) {
                    workers.emplace_back(accumulate_range, total_points * worker / worker_count,
                                         total_points * (worker + 1) / worker_count, write_labels, std::ref(accums[worker]));
                }
                for (auto& worker : workers)
                    worker.join();
            }
            for (int c = 0; c < K; ++c) {
                double sum = 0;
                int64_t count = 0;
                for (const auto& accum : accums) {
                    sum += accum.sums[c];
                    count += accum.counts[c];
                }
                if (count > 0)
                    centroid_vals[c] = static_cast<float>(sum / static_cast<double>(count));
            }
            use_hints = pooled && std::is_sorted(centroid_vals.begin(), centroid_vals.end()) &&
                        std::all_of(centroid_vals.begin(), centroid_vals.end(), [](float v) { return std::isfinite(v); });
        }
        std::vector<int> order(K);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) { return centroid_vals[a] < centroid_vals[b]; });
        result.centroids.resize(K);
        std::vector<int> inverse(K);
        for (int i = 0; i < K; ++i) {
            result.centroids[i] = centroid_vals[order[i]];
            inverse[order[i]] = i;
        }
        for (uint8_t& label : result.labels)
            label = static_cast<uint8_t>(inverse[label]);
        return result;
    }

    void decimate_candidates(const Tensor& position, const Tensor& rotation, const Tensor& scale, const Tensor& opacity,
                             const Tensor& dc, const Tensor& sh, int rest, std::vector<uint32_t>& idx, std::vector<float>& cost) {
        validate_splats(position, rotation, scale, opacity, dc, sh, rest);
        require(position.device() == Device::GPU, "decimation candidates need GPU tensors");
        internal::backend_ops_for(position).decimate_candidates(position.contiguous(), rotation.contiguous(), scale.contiguous(),
                                                                opacity.contiguous(), dc.contiguous(),
                                                                rest ? sh.contiguous() : sh, rest, idx, cost, {});
    }

    DecimateMerge decimate_merge(const Tensor& position, const Tensor& rotation, const Tensor& scale, const Tensor& opacity,
                                 const Tensor& dc, const Tensor& sh, int rest, const std::vector<int>& member_group,
                                 const std::vector<uint32_t>& minimum, const std::vector<uint32_t>& members,
                                 const std::vector<uint32_t>& offsets, size_t removed) {
        validate_splats(position, rotation, scale, opacity, dc, sh, rest);
        require(position.device() == Device::GPU, "decimation merge needs GPU tensors");
        const size_t n = position.size(0);
        const size_t groups = minimum.size();
        require(member_group.size() == n, "decimation merge needs one group entry per splat");
        require(offsets.size() == groups + 1 && offsets.front() == 0 && offsets.back() == members.size(),
                "decimation merge offsets must frame the member list");
        size_t merged = 0;
        for (size_t g = 0; g < groups; ++g) {
            const uint32_t begin = offsets[g], end = offsets[g + 1];
            // The merge kernels weigh at most four members.
            require(end > begin && end - begin <= 4, "decimation merge groups hold one to four splats");
            uint32_t lowest = std::numeric_limits<uint32_t>::max();
            for (uint32_t m = begin; m < end; ++m) {
                require(members[m] < n && member_group[members[m]] == int(g), "decimation merge member outside its group");
                lowest = std::min(lowest, members[m]);
            }
            require(minimum[g] == lowest, "decimation merge minimum must be the lowest member");
            merged += end - begin - 1;
        }
        size_t grouped = 0;
        for (const int g : member_group) {
            require(g >= -1 && g < int(groups), "decimation merge group index out of range");
            grouped += g >= 0;
        }
        require(grouped == members.size() && merged == removed, "decimation merge counts do not match the selection");
        return internal::backend_ops_for(position).decimate_merge(position.contiguous(), rotation.contiguous(), scale.contiguous(),
                                                                  opacity.contiguous(), dc.contiguous(),
                                                                  rest ? sh.contiguous() : sh, rest, member_group,
                                                                  minimum, members, offsets, removed, {});
    }
} // namespace lfs::core
