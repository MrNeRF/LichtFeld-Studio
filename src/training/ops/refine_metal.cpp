/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Metal RefineOps; see ops/refine_cuda.cpp for the reference behaviour.

#include "metal_families.hpp"
#include "metal_kernels.hpp"
#include "metal_select.hpp"

#include <array>
#include <format>
#include <limits>
#include <stdexcept>

namespace lfs::training {
    namespace {
        using lfs::gpu_ops::RefineInputs;
        using lfs::gpu_ops::RefineOutputs;
        using lfs::gpu_ops::Tensor;
        namespace mk = metal;

        using metal::count32;

        struct SplitParams {
            uint64_t means, rotations, scales, sh0, opacity;
            uint64_t child_means, child_rotations, child_scales, child_sh0, child_opacity;
            uint64_t indices;
            uint32_t count;
        };

        void split(const RefineOutputs& parents, const RefineOutputs& children, const Tensor& indices) {
            const size_t n = indices.numel();
            if (n == 0)
                return;
            if (n > static_cast<size_t>(std::numeric_limits<int>::max()))
                throw std::invalid_argument(std::format("refine split count {} exceeds INT_MAX", n));
            const SplitParams params{mk::address(parents.means), mk::address(parents.rotations),
                                     mk::address(parents.scales), mk::address(parents.sh0),
                                     mk::address(parents.opacity), mk::address(children.means),
                                     mk::address(children.rotations), mk::address(children.scales),
                                     mk::address(children.sh0), mk::address(children.opacity),
                                     mk::address(indices), static_cast<uint32_t>(n)};
            mk::launch_items("refine_split", params,
                             {&parents.means, &parents.rotations, &parents.scales, &parents.sh0, &parents.opacity,
                              &children.means, &children.rotations, &children.scales, &children.sh0,
                              &children.opacity, &indices},
                             n);
        }

        struct FillParams {
            uint64_t indices, means, rotations, scales, sh0, opacity;
            uint64_t dst_means, dst_rotations, dst_scales, dst_sh0, dst_opacity, free_mask;
            uint32_t count, rows;
        };

        void fill_slots(const Tensor& indices, const RefineInputs& source, const RefineOutputs& destination,
                        Tensor& free_mask) {
            const size_t n = indices.numel();
            if (n == 0)
                return;
            const FillParams params{mk::address(indices),
                                    mk::address(source.means),
                                    mk::address(source.rotations),
                                    mk::address(source.scales),
                                    mk::address(source.sh0),
                                    mk::address(source.opacity),
                                    mk::address(destination.means),
                                    mk::address(destination.rotations),
                                    mk::address(destination.scales),
                                    mk::address(destination.sh0),
                                    mk::address(destination.opacity),
                                    mk::address(free_mask),
                                    count32(n, "refine fill"),
                                    count32(destination.means.shape()[0], "refine fill rows")};
            mk::launch_items("refine_fill_slots", params,
                             {&indices, &source.means, &source.rotations, &source.scales, &source.sh0,
                              &source.opacity, &destination.means, &destination.rotations, &destination.scales,
                              &destination.sh0, &destination.opacity, &free_mask},
                             n);
        }

        struct CountsParams {
            uint64_t bool0, bool1, float0, float1, counts;
            uint32_t count_bool0, count_bool1, count_float0, count_float1;
        };

        uint32_t optional_count(const Tensor& tensor) {
            return tensor.is_valid() ? count32(tensor.numel(), "refine counts") : 0u;
        }

        void counts(const Tensor& bool0, const Tensor& bool1, const Tensor& float0, const Tensor& float1,
                    Tensor& counts4) {
            const CountsParams params{mk::address(bool0), mk::address(bool1), mk::address(float0),
                                      mk::address(float1), mk::address(counts4), optional_count(bool0),
                                      optional_count(bool1), optional_count(float0), optional_count(float1)};
            mk::launch("refine_counts", params, {&bool0, &bool1, &float0, &float1, &counts4}, 1, 1024);
        }

        struct ValuesParams {
            uint64_t values, select_state;
            uint32_t count;
        };

        void normalize_positive_median(Tensor& values) {
            const size_t n = values.numel();
            if (n == 0)
                return;
            const ValuesParams zero{mk::address(values), 0, count32(n, "positive median")};
            mk::launch_items("refine_zero_nan", zero, {&values}, n);
            constexpr std::array<uint32_t, 1> offsets{0};
            constexpr std::array<uint32_t, 1> ranks{mk::kSelectMedian};
            const Tensor workspace = mk::radix_select(values, n, 1, true, offsets, ranks);
            const ValuesParams divide{mk::address(values), mk::address(workspace), static_cast<uint32_t>(n)};
            mk::launch_items("refine_divide_by_median", divide, {&values, &workspace}, n);
        }

        struct ShareParams {
            uint64_t log_scales, error, shares, frozen, scores;
            uint32_t frozen_count, count;
            float limit;
        };

        void clip_scales(Tensor& log_scales, const Tensor& shares, const Tensor& frozen, const float limit) {
            const size_t n = log_scales.shape()[0];
            if (n == 0 || !(limit > 0.0f) || !(limit < 1.0f))
                return;
            const ShareParams params{mk::address(log_scales), 0, mk::address(shares), mk::address(frozen), 0,
                                     optional_count(frozen), count32(n, "screen-share clip"), limit};
            mk::launch_items("refine_clip_scales", params, {&log_scales, &shares, &frozen}, n);
        }

        void oversize_scores(const Tensor& error, const Tensor& shares, const Tensor& frozen, Tensor& scores,
                             const float limit) {
            const size_t n = error.numel();
            if (n == 0)
                return;
            const ShareParams params{0, mk::address(error), mk::address(shares), mk::address(frozen),
                                     mk::address(scores), optional_count(frozen), count32(n, "oversize scores"),
                                     limit};
            mk::launch_items("refine_oversize_scores", params, {&error, &shares, &frozen, &scores}, n);
        }

        struct MaskParams {
            uint64_t opacity, rotations, mask;
            uint32_t count;
            float minimum_opacity;
        };

        void dead_mask(const Tensor& opacity, const Tensor& rotations, Tensor& mask, const float minimum_opacity) {
            const size_t n = opacity.numel();
            if (n == 0)
                return;
            const MaskParams params{mk::address(opacity), mk::address(rotations), mk::address(mask),
                                    count32(n, "dead mask"), minimum_opacity};
            mk::launch_items("refine_dead_mask", params, {&opacity, &rotations, &mask}, n);
        }

        void rotation_mask(const Tensor& rotations, Tensor& mask) {
            const size_t n = rotations.shape()[0];
            if (n == 0)
                return;
            const MaskParams params{0, mk::address(rotations), mk::address(mask), count32(n, "rotation mask"), 0.0f};
            mk::launch_items("refine_rotation_mask", params, {&rotations, &mask}, n);
        }
    } // namespace

    const lfs::gpu_ops::RefineOps& metal_refine_ops() {
        static const lfs::gpu_ops::RefineOps ops{
            .split = split,
            .fill_slots = fill_slots,
            .counts = counts,
            .normalize_positive_median = normalize_positive_median,
            .clip_scales = clip_scales,
            .oversize_scores = oversize_scores,
            .dead_mask = dead_mask,
            .rotation_mask = rotation_mask,
        };
        return ops;
    }
} // namespace lfs::training
