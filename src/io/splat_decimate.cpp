/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/tensor_backend.hpp"
#include "cuda/splat_decimate_math.hpp"
#include <algorithm>
#include <array>
#include <cuda_runtime.h>
#include <io/splat_decimate.hpp>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace lfs::io::decimate {
    using core::Device;
    using core::Tensor;
    Data allocate(size_t n, int rest, Device device) {
        return {Tensor::empty({n, 3}, device), Tensor::empty({n, 4}, device), Tensor::empty({n, 3}, device),
                Tensor::empty({n, 1}, device), Tensor::empty({n, 1, 3}, device), Tensor::empty({n, size_t(rest), 3}, device), n, rest};
    }
    Selection select(const Candidates& c, size_t n, int k, size_t needed) {
        constexpr int buckets = 1024;
        double low = INFINITY, high = -INFINITY;
        for (float cost : c.cost)
            if (std::isfinite(cost)) {
                low = std::min(low, double(cost));
                high = std::max(high, double(cost));
            }
        double span = high > low ? high - low : 1;
        auto bucket = [&](float cost) { return std::min(buckets - 1, int(std::floor(((cost - low) / span) * buckets))); };
        std::array<uint32_t, buckets + 1> counts{};
        for (float cost : c.cost)
            if (std::isfinite(cost))
                ++counts[bucket(cost) + 1];
        for (int b = 0; b < buckets; ++b)
            counts[b + 1] += counts[b];
        auto cursor = counts;
        std::vector<uint32_t> order(counts[buckets]);
        for (size_t e = 0; e < c.cost.size(); ++e)
            if (std::isfinite(c.cost[e]))
                order[cursor[bucket(c.cost[e])]++] = e;
        Selection s;
        s.member_group.assign(n, -1);
        std::vector<std::array<uint32_t, 4>> groups;
        std::vector<uint32_t> sizes;
        groups.reserve(needed);
        sizes.reserve(needed);
        for (uint32_t e : order) {
            if (s.removed == needed)
                break;
            uint32_t i = e / k, j = c.idx[e];
            if (j == invalid || s.member_group[i] != -1 || s.member_group[j] != -1)
                continue;
            s.member_group[i] = s.member_group[j] = int(groups.size());
            groups.push_back({i, j, 0, 0});
            sizes.push_back(2);
            ++s.removed;
        }
        for (uint32_t cap : {3u, 4u}) {
            if (s.removed == needed)
                break;
            for (uint32_t e : order) {
                if (s.removed == needed)
                    break;
                uint32_t i = e / k, j = c.idx[e];
                if (j == invalid || s.member_group[i] != -1)
                    continue;
                int g = s.member_group[j];
                if (g < 0 || sizes[g] >= cap)
                    continue;
                s.member_group[i] = g;
                groups[g][sizes[g]++] = i;
                ++s.removed;
            }
        }
        s.offsets.push_back(0);
        for (size_t g = 0; g < groups.size(); ++g) {
            s.minimum.push_back(*std::min_element(groups[g].begin(), groups[g].begin() + sizes[g]));
            s.members.insert(s.members.end(), groups[g].begin(), groups[g].begin() + sizes[g]);
            s.offsets.push_back(s.members.size());
        }
        return s;
    }
    // Independent CPU exact-kNN oracle: balanced median kd-tree, double distances.
    Candidates cpu_candidates(const Data& data) {
        constexpr int knn = knn_k, k = candidates_k;
        auto v = data.view();
        size_t n = data.n;
        std::vector<Cache> cache(n);
        for (size_t i = 0; i < n; ++i)
            cache[i] = cache_one(v, i);
        std::vector<uint32_t> ids(n);
        std::iota(ids.begin(), ids.end(), 0);
        auto build = [&](auto&& self, size_t begin, size_t end, int axis) -> void {
            if (begin >= end)
                return;
            size_t mid = (begin + end) / 2;
            std::nth_element(ids.begin() + begin, ids.begin() + mid, ids.begin() + end, [&](uint32_t a, uint32_t b) {
                float x = v.pos[size_t(a) * 3 + axis], y = v.pos[size_t(b) * 3 + axis];
                return x == y ? a < b : x < y;
            });
            self(self, begin, mid, (axis + 1) % 3);
            self(self, mid + 1, end, (axis + 1) % 3);
        };
        build(build, 0, n, 0);
        Candidates out{std::vector<uint32_t>(n * k, invalid), std::vector<float>(n * k, INFINITY)};
        for (uint32_t i = 0; i < n; ++i) {
            std::array<std::pair<double, uint32_t>, knn_k> best;
            best.fill({INFINITY, invalid});
            auto visit = [&](auto&& self, size_t begin, size_t end, int axis) -> void {
                if (begin >= end)
                    return;
                size_t mid = (begin + end) / 2;
                uint32_t j = ids[mid];
                double delta = double(v.pos[size_t(i) * 3 + axis]) - v.pos[size_t(j) * 3 + axis];
                if (i != j) {
                    double d2 = 0;
                    for (int a = 0; a < 3; ++a) {
                        double d = double(v.pos[size_t(i) * 3 + a]) - v.pos[size_t(j) * 3 + a];
                        d2 += d * d;
                    }
                    std::pair<double, uint32_t> p{d2, j};
                    if (p < best[knn - 1]) {
                        int a = knn - 1;
                        while (a > 0 && p < best[a - 1]) {
                            best[a] = best[a - 1];
                            --a;
                        }
                        best[a] = p;
                    }
                }
                if (delta < 0) {
                    self(self, begin, mid, (axis + 1) % 3);
                    if (delta * delta <= best[knn - 1].first)
                        self(self, mid + 1, end, (axis + 1) % 3);
                } else {
                    self(self, mid + 1, end, (axis + 1) % 3);
                    if (delta * delta <= best[knn - 1].first)
                        self(self, begin, mid, (axis + 1) % 3);
                }
            };
            visit(visit, 0, n, 0);
            std::array<std::pair<float, uint32_t>, knn_k> candidates;
            candidates.fill({INFINITY, invalid});
            for (int a = 0; a < knn; ++a)
                if (best[a].second != invalid) {
                    auto j = best[a].second;
                    float cost = edge(v, cache.data(), i, j);
                    if (std::isfinite(cost))
                        candidates[a] = {cost, j};
                }
            std::sort(candidates.begin(), candidates.begin() + knn);
            for (int a = 0; a < k; ++a) {
                out.cost[size_t(i) * k + a] = candidates[a].first;
                out.idx[size_t(i) * k + a] = candidates[a].second;
            }
        }
        return out;
    }
    Data cpu_merge(const Data& data, const Selection& s) {
        auto out = allocate(data.n - s.removed, data.rest, Device::CPU);
        uint32_t row = 0;
        for (uint32_t i = 0; i < data.n; ++i) {
            int g = s.member_group[i];
            if (g < 0)
                copy_one(data.view(), i, out.view(), row++);
            else if (i == s.minimum[g])
                merge_one(data.view(), s.members.data() + s.offsets[g], s.offsets[g + 1] - s.offsets[g], out.view(), row++);
        }
        return out;
    }
} // namespace lfs::io::decimate

namespace lfs::io {
    namespace {
        struct Cancelled {};
    } // namespace

    Result<core::SplatData> decimate_splats(const core::SplatData& input, const DecimateOptions& o) {
        using namespace decimate;
        using core::Device;
        try {
            // The decimator's kernels and scratch allocations are CUDA-specific.
            const core::GpuBackendScope cuda_scope(core::GpuBackend::CUDA);
            if (!o.target_count)
                return make_error(ErrorCode::INVALID_DATASET, "decimation target must be at least 1");
            auto progress = [&](float p, const std::string& stage) { if(o.progress && !o.progress(p,stage)) throw Cancelled{}; };
            progress(0, "Preparing decimation");
            if (!input.means().is_valid() || input.means().ndim() != 2 || input.means().size(1) != 3)
                return make_error(ErrorCode::INVALID_DATASET, "decimation requires means with shape [N,3]");
            const size_t input_n = input.size();
            auto valid_attribute = [&](const core::Tensor& t, size_t width) {
                return t.is_valid() && t.dtype() == core::DataType::Float32 && t.numel() == input_n * width;
            };
            if (!valid_attribute(input.means(), 3) || !valid_attribute(input.rotation_raw(), 4) ||
                !valid_attribute(input.scaling_raw(), 3) || !valid_attribute(input.opacity_raw(), 1) || !valid_attribute(input.sh0(), 3))
                return make_error(ErrorCode::INVALID_DATASET, "decimation requires float32 position, rotation, scale, opacity and DC attributes");
            const core::SplatData* source = &input;
            core::SplatData visible;
            if (input.has_deleted_mask()) {
                const auto& mask = input.deleted();
                if (mask.dtype() != core::DataType::Bool || mask.ndim() != 1 || mask.numel() != input_n)
                    return make_error(ErrorCode::INVALID_DATASET, "decimation deleted mask must be bool [N]");
                // apply_deleted deliberately refuses to remove every row. The IO
                // contract must still exclude an entirely soft-deleted scene.
                if (size_t(mask.sum_scalar()) == input_n) {
                    auto empty = allocate(0, int(input.max_sh_coeffs_rest()), Device::CUDA);
                    visible = core::SplatData(input.get_max_sh_degree(), empty.pos, empty.dc, empty.sh,
                                              empty.scale, empty.rot, empty.opacity, input.get_scene_scale(), core::SplatData::ShNLayout::Canonical);
                    visible.set_active_sh_degree(input.get_active_sh_degree());
                } else {
                    visible = input.clone();
                    visible.apply_deleted();
                }
                source = &visible;
            }
            size_t n = source->size();
            if (n > size_t(std::numeric_limits<int>::max()) || n * candidates_k > std::numeric_limits<uint32_t>::max())
                return make_error(ErrorCode::INVALID_DATASET, "decimation input exceeds index capacity");
            Device device = o.use_gpu ? Device::CUDA : Device::CPU;
            auto materialize = [&](const core::Tensor& t) {
                if (o.use_gpu && core::gpu_backend_of(t) == core::GpuBackend::Vulkan)
                    return t.cpu().to(device).contiguous();
                return t.device() == device ? t.contiguous() : t.to(device).contiguous();
            };
            Data data{materialize(source->means()), materialize(source->rotation_raw()), materialize(source->scaling_raw()),
                      materialize(source->opacity_raw()), materialize(source->sh0()), materialize(source->shN_canonical()), n, int(source->max_sh_coeffs_rest())};
            // Establish ordering with caller-owned streams before default-stream kernels.
            if (o.use_gpu) {
                auto err = cudaDeviceSynchronize();
                if (err != cudaSuccess)
                    throw std::runtime_error(cudaGetErrorString(err));
            }
            size_t initial = n;
            int generation = 0;
            while (data.n > o.target_count) {
                float p = float(initial - data.n) / float(initial - o.target_count);
                progress(p, "Decimation generation " + std::to_string(++generation) + ": neighbours and costs");
                auto c = o.use_gpu ? gpu_candidates(data) : cpu_candidates(data);
                progress(p, "Selecting merges");
                size_t needed = data.n - std::max(o.target_count, data.n - data.n / 2);
                auto s = select(c, data.n, candidates_k, needed);
                if (!s.removed)
                    throw std::runtime_error("decimation found no finite merge candidates");
                if (s.removed < needed && double(s.removed) < 0.05 * data.n)
                    throw std::runtime_error("decimation stalled: fewer than 5% of splats merged");
                c = {};
                progress(p, "Merging splats");
                data = o.use_gpu ? gpu_merge(data, s) : cpu_merge(data, s);
            }
            // Clone unchanged inputs too: the API promises independently owned output.
            auto output = [&](core::Tensor& t) { return t.device() == Device::CUDA ? (generation ? t : t.clone()) : t.to(Device::CUDA); };
            core::SplatData result(source->get_max_sh_degree(), output(data.pos), output(data.dc), output(data.sh),
                                   output(data.scale), output(data.rot), output(data.opacity), source->get_scene_scale(), core::SplatData::ShNLayout::Canonical);
            result.set_active_sh_degree(source->get_active_sh_degree());
            progress(1, "Decimation complete");
            return result;
        } catch (const Cancelled&) { return make_error(ErrorCode::CANCELLED, "Decimation cancelled by user"); } catch (const std::exception& e) {
            return make_error(ErrorCode::INTERNAL_ERROR, e.what());
        }
    }
} // namespace lfs::io
