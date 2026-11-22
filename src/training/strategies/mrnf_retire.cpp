/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// MRNF retire quota and keep covered, both read from the peak rows: the largest blend weight each splat
// reached in a recent view and its renders since birth. At the cap, each refine frees the splats that
// never reach a photo or show the least of their opacity, and large splats hidden behind others are
// spared the decay they cannot undo.

#include "kernels/mrnf_kernels.hpp"
#include "mrnf.hpp"
#include "strategy_utils.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace lfs::training {

    namespace {

        // Retirees per refine as a share of the live splats, from the first refine at the cap,
        // tapering linearly to 0 at grow_until_iter.
        constexpr float MRNF_RETIRE_SHARE = 0.015f;
        // Renders [steps] since birth before a splat can retire or count as covered.
        constexpr float MRNF_RETIRE_GRACE = 30.0f;
        // Half-life [steps] of the running peak blend weight.
        constexpr float MRNF_PEAK_HALF_LIFE_STEPS = 800.0f;
        // Reach [0, 1] below which a splat is covered: its best recent view saw less than this share
        // of its opacity.
        constexpr float MRNF_COVERED_REACH = 0.5f;
        // Keep covered applies to splats whose largest axis is at least this many times the live median.
        constexpr float MRNF_KEEP_COVERED_SIZE_FACTOR = 6.0f;
        // Peak blend weight below which a splat is dead: the retire quota takes dead splats first.
        constexpr float MRNF_RETIRE_DEAD_PEAK = 0.01f;

    } // namespace

    // The peak rows need the FastGS backward, which writes each step's peak blend weight.
    bool MRNF::peak_rows_on() const {
        return _params && !_params->gut && !_topology_frozen;
    }

    void MRNF::ensure_peak_rows() {
        using namespace lfs::core;
        const size_t n = _splat_data ? static_cast<size_t>(_splat_data->size()) : 0;
        for (auto* row : {&_blend_peak, &_step_blend_peak, &_renders}) {
            if (!peak_rows_on()) {
                *row = Tensor();
            } else if (!row->is_valid() || row->ndim() != 1 || row->numel() > n) {
                *row = zero_splat_vector(n, _splat_data->means().device());
            } else if (const size_t rows = row->numel(); rows < n) {
                *row = Tensor::cat(std::vector<Tensor>{*row, zero_splat_vector(n - rows, row->device())}, 0);
            }
        }
    }

    lfs::core::Tensor MRNF::blend_peak_scratch(const int /*iter*/) {
        if (!_params || !_splat_data || _splat_data->size() == 0) {
            return {};
        }
        ensure_peak_rows();
        return _step_blend_peak;
    }

    // After the backward: folds the step's peak into the running peak, which halves every
    // MRNF_PEAK_HALF_LIFE_STEPS, and counts the step as a render where the splat was seen.
    void MRNF::fold_blend_peak() {
        const size_t n = static_cast<size_t>(_splat_data->size());
        if (n == 0 || _blend_peak.numel() != n || _step_blend_peak.numel() != n || _renders.numel() != n) {
            return;
        }
        mrnf_strategy::launch_blend_peak_update(_step_blend_peak.ptr<float>(), _blend_peak.ptr<float>(),
                                                _renders.ptr<float>(),
                                                std::pow(0.5f, 1.0f / MRNF_PEAK_HALF_LIFE_STEPS), n);
    }

    // Retire quota: at the cap, each refine also frees the dead splats (peak blend weight below
    // MRNF_RETIRE_DEAD_PEAK) and then those with the lowest reach. A faint splat that shows all of its
    // opacity stays; one that never reaches a photo, or only a small share of its opacity, goes. Their
    // slots go to growth where the error is. Adds them to prune_mask.
    void MRNF::retire_lowest_reach(const int iter, lfs::core::Tensor& prune_mask) {
        _last_retired = 0;
        if (!peak_rows_on()) {
            return;
        }
        update_covered_size_floor(prune_mask);
        _last_retired = mark_retirees(iter, prune_mask);
    }

    size_t MRNF::retire_count(const int iter, const int first_iter, const int stop, const size_t live) {
        if (first_iter < 0 || iter < first_iter || iter >= stop)
            return 0;
        const float taper = static_cast<float>(stop - iter) / static_cast<float>(stop - first_iter);
        return static_cast<size_t>(MRNF_RETIRE_SHARE * taper * static_cast<float>(live));
    }

    size_t MRNF::mark_retirees(const int iter, lfs::core::Tensor& prune_mask) {
        using namespace lfs::core;
        const size_t n = static_cast<size_t>(_splat_data->size());
        const size_t active = active_count();
        if (_retire_first_iter < 0 && at_cap(active)) {
            _retire_first_iter = iter;
        }
        const size_t natural = prune_mask.count_nonzero();
        const size_t count = retire_count(iter, _retire_first_iter, static_cast<int>(_params->grow_until_iter),
                                          active > natural ? active - natural : 0);
        ensure_peak_rows();
        if (count == 0 || n == 0 || _blend_peak.numel() != n || _renders.numel() != n) {
            return 0;
        }
        // Rows that can retire: rendered past the grace, not pruned, free or frozen.
        Tensor eligible = (_renders >= MRNF_RETIRE_GRACE).logical_and(prune_mask.logical_not());
        if (_free_mask.is_valid()) {
            eligible = eligible.logical_and(_free_mask.slice(0, 0, n).logical_not());
        }
        if (auto trainable = make_trainable_mask(*_splat_data, n, _splat_data->means().device()); trainable.is_valid()) {
            eligible = eligible.logical_and(trainable);
        }
        if (const Tensor factor = covered_factor(); factor.is_valid()) {
            // Kept rows leave the pool before the pick; the quota is taken from the others.
            eligible = eligible.logical_and((factor < MRNF_COVERED_REACH).logical_not());
        }
        eligible = eligible.contiguous();
        const size_t marked = std::min(count, static_cast<size_t>(eligible.count_nonzero()));
        if (marked == 0) {
            return 0;
        }
        Tensor key = Tensor::empty({n}, Device::CUDA, DataType::Float32);
        mrnf_strategy::launch_mrnf_retire_key(_blend_peak.ptr<float>(), _splat_data->opacity_raw().ptr<float>(),
                                              MRNF_RETIRE_DEAD_PEAK, n, key.ptr<float>());
        Tensor rows = Tensor::empty({marked}, Device::CUDA, DataType::Int64);
        mrnf_strategy::launch_lowest_eligible_rows(key.ptr<float>(), eligible.ptr<bool>(), n, marked,
                                                   rows.ptr<int64_t>());
        Tensor retired = Tensor::zeros_bool({n}, Device::CUDA);
        mrnf_strategy::launch_fill_index_mask(rows.ptr<int64_t>(), marked, retired.ptr<bool>());
        prune_mask = prune_mask.logical_or(retired).contiguous();
        return marked;
    }

    // Keep covered, size floor of this refine: MRNF_KEEP_COVERED_SIZE_FACTOR times the median largest
    // axis of the splats that stay live.
    void MRNF::update_covered_size_floor(const lfs::core::Tensor& prune_mask) {
        using namespace lfs::core;
        _covered_log_size_floor = std::numeric_limits<float>::infinity();
        const size_t n = _splat_data ? static_cast<size_t>(_splat_data->size()) : 0;
        if (n == 0) {
            return;
        }
        Tensor live = prune_mask.is_valid() && prune_mask.numel() == n ? prune_mask.logical_not()
                                                                       : Tensor::ones_bool({n}, Device::CUDA);
        if (_free_mask.is_valid()) {
            live = live.logical_and(_free_mask.slice(0, 0, n).logical_not());
        }
        live = live.contiguous();
        const size_t count = live.count_nonzero();
        if (count == 0) {
            return;
        }
        // The median of the largest log-scale is the log of the median largest axis (exp is monotone).
        const Tensor largest = _splat_data->scaling_raw().max(1).contiguous();
        const size_t k = (count + 1) / 2;
        const float median_log = mrnf_strategy::launch_kth_largest(largest.ptr<float>(), live.ptr<bool>(), n, k);
        _covered_log_size_floor = median_log + std::log(MRNF_KEEP_COVERED_SIZE_FACTOR);
    }

    // Keep covered: a large splat whose reach (peak blend weight over opacity) is below
    // MRNF_COVERED_REACH sits behind others, so it cannot earn its opacity back. Its decays scale by
    // its reach and it is never retired. Returns that factor per row, 1 for every other row.
    lfs::core::Tensor MRNF::covered_factor() const {
        using namespace lfs::core;
        const size_t n = _splat_data ? static_cast<size_t>(_splat_data->size()) : 0;
        if (!peak_rows_on() || n == 0 || _blend_peak.numel() != n || _renders.numel() != n) {
            return {};
        }
        Tensor factor = Tensor::empty({n}, Device::CUDA, DataType::Float32);
        mrnf_strategy::launch_mrnf_covered_factor(
            _blend_peak.ptr<float>(), _renders.ptr<float>(), _splat_data->opacity_raw().ptr<float>(),
            _splat_data->scaling_raw().ptr<float>(), MRNF_RETIRE_GRACE, MRNF_COVERED_REACH,
            _covered_log_size_floor, n, factor.ptr<float>());
        return factor;
    }

} // namespace lfs::training
