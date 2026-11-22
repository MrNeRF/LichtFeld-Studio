/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// MRNF growth at the splat cap. Growth draws by NEED, the per-view refine error relative to its mean.
// Once the model is full, each refine also frees the splats that contribute least per view (growth exchange) so
// growth keeps working until grow_until_iter, and every split keeps the parent's opacity where its two halves
// overlap.

#include "core/assert.hpp"
#include "kernels/mrnf_kernels.hpp"
#include "mrnf.hpp"
#include "strategy_utils.hpp"
#include "training/dataset.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace lfs::training {

    namespace {

        // Share of max_cap [0, 1] the model must fill before the exchange starts; below it, growth first
        // refills the slots freed earlier.
        constexpr double MRNF_EXCHANGE_MIN_CAP_FILL = 0.98;
        // Exchange victims per refine as a share of the active splats; tapers on a cosine to 0 at grow_until_iter.
        constexpr double MRNF_EXCHANGE_SHARE = 0.05;
        // Exponent of NEED: growth and refill draw in proportion to (error / mean error)^3.
        constexpr float MRNF_REFINE_NEED_POWER = 3.0f;
        // Upper clamp of error / mean error before the power, so a few outliers cannot take all growth.
        constexpr float MRNF_NEED_ERROR_MAX = 20.0f;
        // Floor of a victim candidate's view count as a share of the training views (at least one view).
        constexpr double MRNF_EXCHANGE_VIEW_FLOOR_SHARE = 0.05;
        // Share of the rows a refine already freed (pruned and retired) that it takes off its exchange quota.
        constexpr double MRNF_EXCHANGE_FREED_CREDIT = 0.5;
        // Split at the cap: factor on the longest and the smallest axis of both halves (the middle axis is kept).
        constexpr float MRNF_CAP_SPLIT_AXIS_SCALE = 0.7f;

    } // namespace

    // Adds this step's per-view error to the window. Each view's error sum is compressed before it is
    // added, so a large footprint cannot outweigh a high error per pixel; the views that saw each
    // splat are counted for the exchange ranking.
    void MRNF::fold_step_error(const int iter) {
        using namespace lfs::core;
        const size_t n = static_cast<size_t>(_splat_data->size());
        float* coverage_views = nullptr;
        if (growth_exchange_active(iter)) {
            if (!_exchange_window_views.is_valid() || _exchange_window_views.numel() != n)
                _exchange_window_views = Tensor::zeros({n}, _splat_data->means().device());
            coverage_views = _exchange_window_views.ptr<float>();
        }
        mrnf_strategy::launch_fold_densification_error_view_sum_and_zero(
            _refine_weight_max.ptr<float>(),
            _splat_data->_densification_info.ptr<float>(),
            n, coverage_views);
    }

    // Turns the window's error sum into the mean over the views that saw an error, so a splat seen
    // from many views does not outrank one seen from few for that alone.
    void MRNF::average_refine_error_over_views() {
        const size_t n = static_cast<size_t>(_splat_data->size());
        if (_refine_weight_max.numel() == n) {
            mrnf_strategy::launch_refine_error_view_mean(
                _refine_weight_max.ptr<float>(), _splat_data->_densification_info.ptr<float>(), n);
        }
    }

    bool MRNF::growth_exchange_active(const int iter) const {
        // Without a finite cap growth always has budget, so there is nothing to exchange.
        return _params && _params->max_cap > 0 && !_topology_frozen &&
               iter < static_cast<int>(_params->grow_until_iter) &&
               iter < static_cast<int>(_params->stop_refine);
    }

    bool MRNF::at_cap(const size_t active) const {
        return _params && _params->max_cap > 0 &&
               static_cast<double>(active) >= MRNF_EXCHANGE_MIN_CAP_FILL * static_cast<double>(_params->max_cap);
    }

    // NEED: the window error over its mean among the active splats, clamped and raised to
    // MRNF_REFINE_NEED_POWER. Growth and refill draw by it, so splats go where the image is most wrong.
    void MRNF::fold_need_into_refine_weights() {
        using namespace lfs::core;
        const size_t n = static_cast<size_t>(_splat_data->size());
        if (n == 0 || !_refine_weight_max.is_valid() || _refine_weight_max.numel() != n) {
            return;
        }
        if ((_refine_weight_max > 0.0f).count_nonzero() == 0) {
            return;
        }
        const size_t active = active_count();
        const float error_scale = std::max(
            (_free_mask.is_valid() ? _refine_weight_max.masked_fill(_free_mask.slice(0, 0, n), 0.0f)
                                   : _refine_weight_max)
                    .sum_scalar() /
                static_cast<float>(std::max<size_t>(active, 1)),
            std::numeric_limits<float>::min());
        mrnf_strategy::launch_fold_need(_refine_weight_max.ptr<float>(), error_scale, MRNF_NEED_ERROR_MAX,
                                        MRNF_REFINE_NEED_POWER, n);
    }

    int MRNF::exchange_warmup_windows() const {
        // Rank only after the memory has seen every training view once.
        constexpr int MIN_WINDOWS = 2;
        const size_t views = _views ? _views->size() : 0;
        const size_t every = std::max<size_t>(1, _params->refine_every);
        return std::max(MIN_WINDOWS, static_cast<int>((views + every - 1) / every));
    }

    // Exchange victims per refine: MRNF_EXCHANGE_SHARE of the active splats, tapering on a cosine to 0 at
    // grow_until_iter so the last children can settle, less MRNF_EXCHANGE_FREED_CREDIT of the rows this
    // refine freed already (pruned and retired), never below 0. Derived from the iteration and this
    // refine alone, so a resume keeps the schedule.
    size_t MRNF::exchange_quota(const int iter, const size_t active, const size_t freed) const {
        constexpr double pi = 3.14159265358979323846;
        const double end = static_cast<double>(_params->grow_until_iter);
        const double u = end > 0.0 ? std::clamp(static_cast<double>(iter) / end, 0.0, 1.0) : 1.0;
        const auto quota = static_cast<size_t>(std::floor(MRNF_EXCHANGE_SHARE * 0.5 * (1.0 + std::cos(pi * u)) *
                                                          static_cast<double>(active)));
        const auto credit = static_cast<size_t>(std::floor(MRNF_EXCHANGE_FREED_CREDIT * static_cast<double>(freed)));
        return quota > credit ? quota - credit : 0;
    }

    // A splat's contribution: its summed blend weight over about one pass over the training views,
    // with the views that saw it counted the same way. Each window adds its sums and older windows
    // fade by 1 - refine_every / views.
    void MRNF::update_exchange_importance() {
        using namespace lfs::core;
        const size_t n = static_cast<size_t>(_splat_data->size());
        const Tensor window_coverage = visibility_accumulator();
        LFS_ASSERT_MSG(window_coverage.is_valid() && window_coverage.numel() == n,
                       "MRNF growth exchange: coverage does not match the splat count");
        // This window's count of views that covered each splat; read before the window outputs reset.
        const Tensor window_views = _exchange_window_views.is_valid() && _exchange_window_views.numel() == n
                                        ? _exchange_window_views
                                        : Tensor::zeros({n}, Device::CUDA);
        reset_exchange_window_outputs();
        if (!_exchange_importance_sum.is_valid()) {
            _exchange_importance_sum = window_coverage.clone();
            _exchange_view_count = window_views;
            _exchange_birth = Tensor::zeros({n}, Device::CUDA);
            _exchange_windows = 1;
            return;
        }
        LFS_ASSERT_MSG(_exchange_importance_sum.numel() == n && _exchange_birth.numel() == n,
                       "MRNF growth exchange: importance does not match the splat count");
        const size_t views = _views ? _views->size() : 0;
        const float keep = views == 0 ? 0.0f
                                      : std::max(0.0f, 1.0f - static_cast<float>(_params->refine_every) /
                                                                  static_cast<float>(views));
        _exchange_importance_sum = _exchange_importance_sum.mul(keep).add(window_coverage);
        if (_exchange_view_count.is_valid() && _exchange_view_count.numel() == n) {
            _exchange_view_count = _exchange_view_count.mul(keep).add(window_views);
        } else {
            _exchange_view_count = window_views;
        }
        ++_exchange_windows;
    }

    float MRNF::exchange_view_floor() const {
        const size_t views = _views ? _views->size() : 0;
        return static_cast<float>(
            std::max(1.0, std::round(MRNF_EXCHANGE_VIEW_FLOOR_SHARE * static_cast<double>(views))));
    }

    // Exchange victims: the eligible splats with the least contribution per view that saw them. Structure seen
    // by few views is not drained for that alone, and the view floor keeps a splat seen once from
    // ranking on noise.
    lfs::core::Tensor MRNF::select_exchange_victims(
        size_t count,
        const lfs::core::Tensor& excluded) {
        using namespace lfs::core;
        const size_t n = static_cast<size_t>(_splat_data->size());
        LFS_ASSERT_MSG(_exchange_importance_sum.numel() == n && _exchange_birth.numel() == n,
                       "MRNF growth exchange: importance does not match the splat count");
        if (count == 0) {
            return {};
        }
        Tensor eligible = _free_mask.is_valid() ? _free_mask.slice(0, 0, n).logical_not()
                                                : Tensor::ones_bool({n}, Device::CUDA);
        if (auto trainable = make_trainable_mask(*_splat_data, n, _splat_data->means().device());
            trainable.is_valid()) {
            eligible = eligible.logical_and(trainable);
        }
        if (excluded.is_valid()) {
            LFS_ASSERT_MSG(excluded.numel() == n && excluded.dtype() == DataType::Bool,
                           "MRNF growth exchange: exclusion mask does not match the splat count");
            eligible = eligible.logical_and(excluded.logical_not());
        }
        // A splat is ranked only after a full pass over the training views since its birth.
        eligible = eligible.logical_and(
            _exchange_birth < static_cast<float>(_exchange_windows - exchange_warmup_windows()));
        count = std::min(count, static_cast<size_t>(eligible.count_nonzero()));
        if (count == 0) {
            return {};
        }
        Tensor victims = Tensor::empty({count}, Device::CUDA, DataType::Int64);
        const Tensor views = _exchange_view_count.is_valid() && _exchange_view_count.numel() == n
                                 ? _exchange_view_count
                                 : Tensor::zeros({n}, Device::CUDA);
        const Tensor ranked = _exchange_importance_sum.div(views.clamp_min(exchange_view_floor())).contiguous();
        mrnf_strategy::launch_lowest_eligible_rows(
            ranked.ptr<float>(), eligible.ptr<bool>(), n, count, victims.ptr<int64_t>());
        return victims;
    }

    // Split at the cap: both halves keep the middle axis, so the split does not narrow the surface it
    // refines, shrink the longest and the smallest axis by MRNF_CAP_SPLIT_AXIS_SCALE, and take opacity
    // 1 - sqrt(1 - o) each: where they overlap they composite back to the parent's opacity o, so a faint
    // parent leaves faint halves. Returns the split kernel's conserve_axis_scale: that factor at the cap,
    // 0 (the ordinary split) below it.
    float MRNF::cap_split_axis_scale(const bool at_cap) {
        return at_cap ? MRNF_CAP_SPLIT_AXIS_SCALE : 0.0f;
    }

    void MRNF::record_exchange_births(const lfs::core::Tensor& split_parents) {
        using namespace lfs::core;
        const size_t before = _exchange_free_before_growth.numel();
        const size_t n = static_cast<size_t>(_splat_data->size());
        LFS_ASSERT_MSG(_exchange_free_before_growth.is_valid() && n >= before &&
                           _exchange_importance_sum.numel() == before && _exchange_birth.numel() == before &&
                           (!_exchange_view_count.is_valid() || _exchange_view_count.numel() == before),
                       "MRNF growth exchange: importance does not match the splat count");
        // Children landed in previously free slots or in appended rows. They start without
        // importance and are ranked once a full pass has scored them.
        Tensor born = _free_mask.is_valid()
                          ? _exchange_free_before_growth.logical_and(_free_mask.slice(0, 0, before).logical_not())
                          : Tensor::zeros_bool({before}, Device::CUDA);
        const auto extend = [&](Tensor& t) {
            if (t.is_valid() && n > before) {
                t = Tensor::cat(std::vector<Tensor>{t, Tensor::zeros({n - before}, Device::CUDA)}, 0);
            }
        };
        if (n > before) {
            born = Tensor::cat(std::vector<Tensor>{born, Tensor::ones_bool({n - before}, Device::CUDA)}, 0);
        }
        if (split_parents.is_valid() && split_parents.numel() > 0) {
            mrnf_strategy::launch_fill_index_mask(
                split_parents.ptr<int64_t>(), split_parents.numel(), born.ptr<bool>());
        }
        extend(_exchange_importance_sum);
        extend(_exchange_view_count);
        extend(_exchange_birth);
        _exchange_importance_sum = _exchange_importance_sum.masked_fill(born, 0.0f);
        if (_exchange_view_count.is_valid()) {
            _exchange_view_count = _exchange_view_count.masked_fill(born, 0.0f);
        }
        _exchange_birth = _exchange_birth.masked_fill(born, static_cast<float>(_exchange_windows));
        _exchange_free_before_growth = Tensor();
    }

    void MRNF::reset_exchange_window_outputs() {
        _exchange_window_views = lfs::core::Tensor();
    }

    void MRNF::reset_exchange_state() {
        _exchange_importance_sum = lfs::core::Tensor();
        _exchange_view_count = lfs::core::Tensor();
        _exchange_birth = lfs::core::Tensor();
        reset_exchange_window_outputs();
        _exchange_free_before_growth = lfs::core::Tensor();
        _exchange_windows = 0;
    }

} // namespace lfs::training
