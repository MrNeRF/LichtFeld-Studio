/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Metal MrnfOps; see ops/mrnf_cuda.cpp for the reference behaviour.

#include "metal_families.hpp"
#include "metal_kernels.hpp"
#include "metal_select.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <format>
#include <limits>
#include <stdexcept>

namespace lfs::training {
    namespace metal {

        struct SelectParams {
            uint64_t values, workspace;
            uint32_t count, stride, positive_only, pass, selections;
            uint32_t offsets[8], ranks[8];
        };

        core::Tensor radix_select(const core::Tensor& values, const size_t count, const uint32_t stride,
                                  const bool positive_only, const std::span<const uint32_t> offsets,
                                  const std::span<const uint32_t> ranks) {
            constexpr size_t kBins = 2048;
            const size_t selections = offsets.size();
            if (selections == 0 || selections > 8 || ranks.size() != selections)
                throw std::invalid_argument(std::format(
                    "radix select takes 1 to 8 selections with one rank each, got {} offsets and {} ranks",
                    selections, ranks.size()));
            if (count > std::numeric_limits<uint32_t>::max())
                throw std::invalid_argument(std::format("radix select count {} exceeds uint32", count));
            auto workspace = core::Tensor::zeros({selections * (4 + kBins)}, core::Device::GPU, core::DataType::Int32);
            SelectParams params{.values = address(values),
                                .workspace = address(workspace),
                                .count = static_cast<uint32_t>(count),
                                .stride = stride,
                                .positive_only = positive_only ? 1u : 0u,
                                .selections = static_cast<uint32_t>(selections)};
            std::ranges::copy(offsets, params.offsets);
            std::ranges::copy(ranks, params.ranks);
            const uint32_t groups = static_cast<uint32_t>(
                std::clamp<size_t>((count + kGroupWidth * 16 - 1) / (kGroupWidth * 16), 1, 512));
            for (uint32_t pass = 0; pass < 3; ++pass) {
                params.pass = pass;
                launch_2d("mrnf_select_histogram", params, {&values, &workspace}, groups,
                          static_cast<uint32_t>(selections), kGroupWidth, 1);
                launch("mrnf_select_pick", params, {&workspace}, 1, static_cast<uint32_t>(selections));
            }
            return workspace;
        }

        std::vector<float> selected_values(const core::Tensor& workspace, const size_t selections) {
            const auto states = workspace.slice(0, 0, 4 * selections).cpu();
            std::vector<float> result(selections);
            for (size_t s = 0; s < selections; ++s)
                result[s] = std::bit_cast<float>(states.ptr<int32_t>()[4 * s + 3]);
            return result;
        }

    } // namespace metal

    namespace {
        using lfs::gpu_ops::Bounds;
        using lfs::gpu_ops::DecayParams;
        using lfs::gpu_ops::GumbelParams;
        using lfs::gpu_ops::MrnfNoiseParams;
        using lfs::gpu_ops::ProjectParams;
        using lfs::gpu_ops::ScalarValidity;
        using lfs::gpu_ops::Tensor;
        namespace mk = metal;

        uint32_t count32(const size_t count, const char* what) {
            if (count > std::numeric_limits<uint32_t>::max())
                throw std::invalid_argument(std::format("{} count {} exceeds uint32", what, count));
            return static_cast<uint32_t>(count);
        }

        uint32_t optional_count(const Tensor& tensor) {
            return tensor.is_valid() ? count32(tensor.numel(), "optional mask") : 0u;
        }

        struct NoiseParams {
            uint64_t means, raw_opacity, visibility, frozen;
            uint64_t seed;
            uint32_t frozen_count, count;
            float lr_mean, noise_weight, median_scale;
        };

        void noise(Tensor& means, const Tensor& raw_opacity, const Tensor& visibility, const Tensor& frozen,
                   const MrnfNoiseParams& p) {
            const size_t n = means.shape()[0];
            if (n == 0)
                return;
            const NoiseParams params{mk::address(means), mk::address(raw_opacity), mk::address(visibility),
                                     mk::address(frozen), p.seed, optional_count(frozen), count32(n, "MRNF noise"),
                                     p.lr_mean, p.noise_weight, p.median_scale};
            mk::launch_items("mrnf_noise", params, {&means, &raw_opacity, &visibility, &frozen}, n);
        }

        struct DecayKernelParams {
            uint64_t raw_opacity, log_scales, frozen, far_mask;
            uint32_t frozen_count, far_count, count;
            float opacity_decay, scale_decay, far_decay_scale, train_t;
        };

        void decay(Tensor& raw_opacity, Tensor& log_scales, const Tensor& frozen, const Tensor& far_mask,
                   const DecayParams& p) {
            const size_t n = log_scales.shape()[0];
            if (n == 0)
                return;
            const DecayKernelParams params{mk::address(raw_opacity), mk::address(log_scales), mk::address(frozen),
                                           mk::address(far_mask), optional_count(frozen), optional_count(far_mask),
                                           count32(n, "MRNF decay"), p.opacity_decay, p.scale_decay,
                                           p.far_decay_scale, p.train_t};
            mk::launch_items("mrnf_decay", params, {&raw_opacity, &log_scales, &frozen, &far_mask}, n);
        }

        Bounds percentile_bounds(const Tensor& means, const float percentile) {
            const size_t n = means.shape()[0];
            if (n == 0 || n > static_cast<size_t>(std::numeric_limits<int>::max()))
                throw std::invalid_argument(std::format("MRNF bounds need 1 to INT_MAX means, got {}", n));
            if (!std::isfinite(percentile) || percentile < 0.0f || percentile > 1.0f)
                throw std::invalid_argument(
                    std::format("MRNF bounds percentile must be finite and within [0, 1], got {}", percentile));
            const float low_pct = (1.0f - percentile) / 2.0f;
            const float high_pct = 1.0f - low_pct;
            const auto low = static_cast<uint32_t>(low_pct * static_cast<float>(n - 1));
            const auto high = static_cast<uint32_t>(high_pct * static_cast<float>(n - 1));
            constexpr std::array<uint32_t, 6> offsets{0, 0, 1, 1, 2, 2};
            const std::array<uint32_t, 6> ranks{low, high, low, high, low, high};
            const auto values = mk::selected_values(mk::radix_select(means, n, 3, false, offsets, ranks), 6);

            Bounds bounds{};
            for (int axis = 0; axis < 3; ++axis) {
                bounds.center[axis] = (values[2 * axis] + values[2 * axis + 1]) * 0.5f;
                bounds.extent[axis] = (values[2 * axis + 1] - values[2 * axis]) * 0.5f;
            }
            float sorted[3] = {bounds.extent[0], bounds.extent[1], bounds.extent[2]};
            std::sort(sorted, sorted + 3);
            bounds.median_size = sorted[1] * 2.0f;
            bounds.max_extent = sorted[2];
            return bounds;
        }

        struct GeomeanParams {
            uint64_t raw_scales, extents;
            uint32_t count;
        };

        ScalarValidity median_extent(const Tensor& raw_scales) {
            if (!raw_scales.is_valid() || raw_scales.numel() == 0)
                return {};
            const size_t n = raw_scales.shape()[0];
            auto extents = Tensor::empty({n}, core::Device::GPU, core::DataType::Float32);
            const GeomeanParams params{mk::address(raw_scales), mk::address(extents), count32(n, "MRNF median extent")};
            mk::launch_items("mrnf_geomean_extent", params, {&raw_scales, &extents}, n);
            constexpr std::array<uint32_t, 1> offsets{0};
            constexpr std::array<uint32_t, 1> ranks{mk::kSelectMedian};
            const float median = mk::selected_values(mk::radix_select(extents, n, 1, true, offsets, ranks), 1)[0];
            ScalarValidity result;
            result.value = std::isfinite(median) && median > 0.0f ? median : 0.0f;
            result.valid = result.value > 0.0f;
            return result;
        }

        struct GumbelKeyParams {
            uint64_t weights, sources, keys;
            uint64_t seed;
            uint32_t count;
        };

        struct GatherIndicesParams {
            uint64_t order, sources, output;
            uint32_t count;
        };

        // Stable descending sort of the Gumbel keys, as CUB's SortPairsDescending.
        // The scratch buffers stay CUDA's; tensor sorts allocate their own.
        void gumbel(GumbelTopKScratch*, const Tensor& weights, Tensor& indices, const GumbelParams& p) {
            const size_t n = weights.numel();
            const size_t k = indices.numel();
            if (k > n)
                throw std::invalid_argument(std::format("MRNF Gumbel top-k needs k <= n, got k={} n={}", k, n));
            if (k == 0)
                return;
            if (n > static_cast<size_t>(std::numeric_limits<int>::max()))
                throw std::invalid_argument(std::format("MRNF Gumbel input {} exceeds INT_MAX", n));
            if (k == n) {
                const GatherIndicesParams params{0, 0, mk::address(indices), static_cast<uint32_t>(k)};
                mk::launch_items("mrnf_gather_indices", params, {&indices}, k);
                return;
            }
            size_t active = n;
            Tensor positive;
            if (p.compact_sparse) {
                if (p.known_nnz > n)
                    throw std::invalid_argument(
                        std::format("Gumbel known_nnz must be <= N (known_nnz={}, N={})", p.known_nnz, n));
                positive = weights > 0.0f;
                active = p.known_nnz > 0 ? p.known_nnz : positive.count_nonzero();
            }
            const bool compact = p.compact_sparse && active >= k && active < n;
            const size_t sort_count = compact ? active : n;
            Tensor sources;
            if (compact) {
                sources = positive.nonzero();
                if (sources.shape()[0] != active)
                    throw std::invalid_argument(std::format(
                        "Gumbel known_nnz {} does not match the {} positive weights", active, sources.shape()[0]));
                sources = sources.reshape({static_cast<int>(active)});
            }
            auto keys = Tensor::empty({sort_count}, core::Device::GPU, core::DataType::Float32);
            const GumbelKeyParams key_params{mk::address(weights), mk::address(sources), mk::address(keys), p.seed,
                                             static_cast<uint32_t>(sort_count)};
            mk::launch_items("mrnf_gumbel_keys", key_params, {&weights, &sources, &keys}, sort_count);
            const auto order = keys.sort(0, true).second;
            const GatherIndicesParams params{mk::address(order), mk::address(sources), mk::address(indices),
                                             static_cast<uint32_t>(k)};
            mk::launch_items("mrnf_gather_indices", params, {&order, &sources, &indices}, k);
        }

        struct FoldParams {
            uint64_t visibility, weight_max, densification, ratio_max;
            uint32_t count;
            float ratio_power;
        };

        void fold(Tensor& visibility, Tensor& weight_max, Tensor& densification, Tensor& ratio_max,
                  const float ratio_power) {
            const size_t n = visibility.numel();
            if (n == 0)
                return;
            const FoldParams params{mk::address(visibility), mk::address(weight_max), mk::address(densification),
                                    mk::address(ratio_max), count32(n, "MRNF fold"), ratio_power};
            mk::launch_items("mrnf_fold", params, {&visibility, &weight_max, &densification, &ratio_max}, n);
        }

        void fold_error(Tensor& weight_max, Tensor& densification) {
            const size_t n = weight_max.numel();
            if (n == 0)
                return;
            const FoldParams params{0, mk::address(weight_max), mk::address(densification), 0,
                                    count32(n, "MRNF fold error"), 0.0f};
            mk::launch_items("mrnf_fold_error", params, {&weight_max, &densification}, n);
        }

        struct ProjectKernelParams {
            uint64_t means, view, means2d, radii;
            uint32_t count;
            int32_t width, height;
            float fx, fy, cx, cy, near_plane;
        };

        void project_centers(const Tensor& means, const Tensor& view, Tensor& means2d, Tensor& radii,
                             const ProjectParams& p) {
            const size_t n = means.shape()[0];
            if (n == 0)
                return;
            if (p.image.w <= 0 || p.image.h <= 0)
                throw std::invalid_argument(
                    std::format("MRNF projection needs a positive image, got {}x{}", p.image.w, p.image.h));
            const ProjectKernelParams params{mk::address(means), mk::address(view), mk::address(means2d),
                                             mk::address(radii), count32(n, "MRNF projection"), p.image.w,
                                             p.image.h, p.intrinsics.fx, p.intrinsics.fy, p.intrinsics.cx,
                                             p.intrinsics.cy, p.near_plane};
            mk::launch_items("mrnf_project_centers", params, {&means, &view, &means2d, &radii}, n);
        }

        struct CenterErrorParams {
            uint64_t means2d, radii, error, scores;
            uint32_t count;
            int32_t width, height;
        };

        void gather_center_error(const Tensor& means2d, const Tensor& radii, const Tensor& error, Tensor& scores) {
            const size_t n = means2d.shape()[0];
            if (n == 0)
                return;
            const int height = static_cast<int>(error.shape()[0]);
            const int width = static_cast<int>(error.shape()[1]);
            if (width <= 0 || height <= 0)
                throw std::invalid_argument(std::format("MRNF center error needs a positive image, got {}x{}", width, height));
            const CenterErrorParams params{mk::address(means2d), mk::address(radii), mk::address(error),
                                           mk::address(scores), count32(n, "MRNF center error"), width, height};
            mk::launch_items("mrnf_gather_center_error", params, {&means2d, &radii, &error, &scores}, n);
        }

        struct FarMaskParams {
            uint64_t means, mask;
            mk::Float3 center;
            uint32_t count;
            float radius_sq;
        };

        void far_mask(const Tensor& means, Tensor& mask, const std::array<float, 3> center, const float radius) {
            const size_t n = means.shape()[0];
            if (n == 0)
                return;
            const FarMaskParams params{mk::address(means), mk::address(mask), {center[0], center[1], center[2]},
                                       count32(n, "MRNF far mask"), radius * radius};
            mk::launch_items("mrnf_far_mask", params, {&means, &mask}, n);
        }

        struct MeanAbsErrorParams {
            uint64_t predicted, target, error;
            uint32_t pixels, channels;
        };

        void mean_abs_error(const Tensor& predicted, const Tensor& target, Tensor& error) {
            const size_t channels = predicted.shape()[0];
            const size_t pixels = predicted.shape()[1] * predicted.shape()[2];
            if (channels == 0 || pixels == 0)
                throw std::invalid_argument(std::format("MRNF mean abs error needs a non-empty [C,H,W] image, got {}x{}",
                                                        channels, pixels));
            const MeanAbsErrorParams params{mk::address(predicted), mk::address(target), mk::address(error),
                                            count32(pixels, "MRNF mean abs error"), static_cast<uint32_t>(channels)};
            mk::launch_items("mrnf_mean_abs_error", params, {&predicted, &target, &error}, pixels);
        }

        struct SeedWeightsParams {
            uint64_t error, alpha, weights;
            uint32_t count;
        };

        void seed_weights(const Tensor& error, const Tensor& alpha, Tensor& weights) {
            const size_t n = weights.numel();
            if (n == 0)
                return;
            const SeedWeightsParams params{mk::address(error), mk::address(alpha), mk::address(weights),
                                           count32(n, "MRNF seed weights")};
            mk::launch_items("mrnf_seed_weights", params, {&error, &alpha, &weights}, n);
        }

        struct GatherSeedsParams {
            uint64_t indices, target, alpha, depth, rgb, sampled_alpha, sampled_depth;
            uint32_t count, pixels;
            int32_t channels;
        };

        void gather_seeds(const Tensor& indices, const Tensor& target, const Tensor& alpha, const Tensor& depth,
                          Tensor& rgb, Tensor& sampled_alpha, Tensor& sampled_depth) {
            const size_t k = indices.numel();
            if (k == 0)
                return;
            const size_t pixels = alpha.numel();
            const int channels = static_cast<int>(target.shape()[0]);
            if (pixels == 0 || channels <= 0)
                throw std::invalid_argument(
                    std::format("MRNF seed gather needs pixels and channels, got {} pixels and {} channels", pixels, channels));
            const GatherSeedsParams params{mk::address(indices), mk::address(target), mk::address(alpha),
                                           mk::address(depth), mk::address(rgb), mk::address(sampled_alpha),
                                           mk::address(sampled_depth), count32(k, "MRNF seed gather"),
                                           count32(pixels, "MRNF seed gather pixels"), channels};
            mk::launch_items("mrnf_gather_seeds", params,
                             {&indices, &target, &alpha, &depth, &rgb, &sampled_alpha, &sampled_depth}, k);
        }

        float sorted_median(const Tensor& values) {
            if (!values.is_valid() || values.numel() == 0)
                return 0.0f;
            constexpr std::array<uint32_t, 1> offsets{0};
            constexpr std::array<uint32_t, 1> ranks{mk::kSelectMedian};
            const float median =
                mk::selected_values(mk::radix_select(values, values.numel(), 1, false, offsets, ranks), 1)[0];
            return std::isfinite(median) ? median : 0.0f;
        }

        struct StarvationParams {
            uint64_t weights, visibility;
            uint32_t count;
            float median;
        };

        void starvation_weights(Tensor& weights, const Tensor& visibility, const float median) {
            const size_t n = weights.numel();
            if (n == 0)
                return;
            const StarvationParams params{mk::address(weights), mk::address(visibility),
                                          count32(n, "MRNF starvation"), median};
            mk::launch_items("mrnf_starvation_weights", params, {&weights, &visibility}, n);
        }

        // Ascending positions of the set bytes, as thrust::copy_if; returns how many are set.
        size_t compact_bool_indices(const Tensor& mask, Tensor& indices, const size_t count) {
            if (mask.numel() == 0 || count == 0)
                return 0;
            const Tensor found = mask.nonzero();
            const size_t total = found.shape()[0];
            const size_t written = std::min({total, count, indices.numel()});
            if (written > 0)
                indices.slice(0, 0, written).copy_from(found.reshape({static_cast<int>(total)}).slice(0, 0, written));
            return total;
        }

        struct PruneBoundsParams {
            uint64_t means, scale_max, mask;
            mk::Float3 center;
            uint32_t count;
            float maximum, log_maximum;
        };

        void prune_bounds(const Tensor& means, const Tensor& scale_max, Tensor& mask, const std::array<float, 3> center,
                          const float maximum, const float log_maximum) {
            const size_t n = means.shape()[0];
            if (n == 0)
                return;
            const PruneBoundsParams params{mk::address(means), mk::address(scale_max), mk::address(mask),
                                           {center[0], center[1], center[2]}, count32(n, "MRNF prune bounds"),
                                           maximum, log_maximum};
            mk::launch_items("mrnf_prune_bounds", params, {&means, &scale_max, &mask}, n);
        }

        struct ParentWeightsParams {
            uint64_t opacity, visibility, active, trainable, edge, weights;
            uint32_t count;
        };

        void replace_parent_weights(const Tensor& opacity, const Tensor& visibility, const Tensor& active,
                                    const Tensor& trainable, const Tensor& edge, Tensor& weights) {
            const size_t n = opacity.numel();
            if (n == 0)
                return;
            const ParentWeightsParams params{mk::address(opacity), mk::address(visibility), mk::address(active),
                                             mk::address(trainable), mk::address(edge), mk::address(weights),
                                             count32(n, "MRNF parent weights")};
            mk::launch_items("mrnf_replace_parent_weights", params,
                             {&opacity, &visibility, &active, &trainable, &edge, &weights}, n);
        }
    } // namespace

    const lfs::gpu_ops::MrnfOps& metal_mrnf_ops() {
        static const lfs::gpu_ops::MrnfOps ops{
            .noise = noise,
            .decay = decay,
            .percentile_bounds = percentile_bounds,
            .median_extent = median_extent,
            .gumbel = gumbel,
            .fold = fold,
            .fold_error = fold_error,
            .project_centers = project_centers,
            .gather_center_error = gather_center_error,
            .far_mask = far_mask,
            .mean_abs_error = mean_abs_error,
            .seed_weights = seed_weights,
            .gather_seeds = gather_seeds,
            .sorted_median = sorted_median,
            .starvation_weights = starvation_weights,
            .compact_bool_indices = compact_bool_indices,
            .prune_bounds = prune_bounds,
            .replace_parent_weights = replace_parent_weights,
        };
        return ops;
    }
} // namespace lfs::training
