/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "match.hpp"
#include "core/cuda_error.hpp"
#include "core/logger.hpp"
#include "core/sh_value_quant.hpp"
#include "dataset.hpp"
#include "kernels/match_kernels.hpp"
#include "lfs/training/sh_value_storage.hpp"
#include "strategy_utils.hpp"
#include <algorithm>
#include <cmath>
#include <format>
#include <stdexcept>
#include <string>
#include <utility>

namespace lfs::training {

    namespace {
        constexpr uint32_t MATCH_MAGIC = 0x4C464D41; // "LFMA"
        constexpr uint32_t MATCH_VERSION = 1;
        // Steps between size clips, for the whole run.
        constexpr int kClipEvery = 100;
        // The position walk's learning-rate share: amplitude = 0.05 x walk scale.
        constexpr float kWalkShare = 0.05f;

        constexpr std::array<ParamType, 6> kParamTypes = {ParamType::Means, ParamType::Sh0, ParamType::ShN,
                                                          ParamType::Scaling, ParamType::Rotation,
                                                          ParamType::Opacity};

        [[nodiscard]] double exp_schedule(const double start, const double end, const double t) {
            if (!(start > 0.0) || !(end > 0.0))
                return start;
            return start * std::pow(end / start, std::clamp(t, 0.0, 1.0));
        }
    } // namespace

    Match::Match(lfs::core::SplatData& splat_data) : _splat_data(&splat_data) {}

    uint64_t Match::seed(const int step, const uint64_t stream) const {
        uint64_t value = static_cast<uint64_t>(std::max(step, 0));
        value ^= 0x9e3779b97f4a7c15ULL + stream + (value << 6) + (value >> 2);
        value += 0x9e3779b97f4a7c15ULL;
        value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
        return value ^ (value >> 31);
    }

    lfs::core::Tensor Match::rows_view(const lfs::core::Tensor& row) const {
        const size_t n = _splat_data ? static_cast<size_t>(_splat_data->size()) : 0;
        if (!row.is_valid() || row.numel() < n)
            return {};
        return row.slice(0, 0, n);
    }

    void Match::ensure_rows() {
        using namespace lfs::core;
        const size_t n = static_cast<size_t>(_splat_data->size());
        const size_t capacity =
            std::max(n, _params && _params->max_cap > 0 ? static_cast<size_t>(_params->max_cap) : n);
        for (auto* row : {&_score, &_score_count, &_oversize, &_peak, &_streak, &_visible, &_step_peak, &_radius, &_renders}) {
            if (row->is_valid() && row->numel() >= n)
                continue;
            auto grown = Tensor::zeros({capacity}, _splat_data->means().device());
            if (row->is_valid() && row->numel() > 0)
                grown.slice(0, 0, row->numel()).copy_from(*row);
            *row = std::move(grown);
        }
    }

    void Match::ensure_error_row() {
        const size_t n = static_cast<size_t>(_splat_data->size());
        auto& info = _splat_data->_densification_info;
        if (info.is_valid() && info.ndim() == 1 && info.numel() == n)
            return;
        ensure_densification_info_shape_inplace(info, n, _splat_data->means().device(), 1);
        info.zero_();
    }

    int Match::steps_per_epoch() const {
        const size_t views = _views ? _views->size() : 0;
        return static_cast<int>(std::max<size_t>(views, 1));
    }

    bool Match::refine_step(const int step) const {
        return _params && _params->refine_every > 0 && step > static_cast<int>(_params->start_refine) &&
               step % static_cast<int>(_params->refine_every) == 0 && step < static_cast<int>(_params->stop_refine);
    }

    bool Match::is_refining(const int iter) const {
        return refine_step(completed_step(iter));
    }

    float Match::split_opacity_share(const int step) const {
        const int steps = _params->match_split_opacity_steps;
        const float t = steps > 0 ? std::clamp(static_cast<float>(step) / static_cast<float>(steps), 0.0f, 1.0f)
                                  : 1.0f;
        return _params->match_split_opacity + (_params->match_split_opacity_end - _params->match_split_opacity) * t;
    }

    float Match::walk_amplitude(const int step) const {
        if (!(_params->match_walk > 0.0f))
            return 0.0f;
        const double t = (static_cast<double>(step) + 0.5) /
                         static_cast<double>(std::max<size_t>(_params->iterations, 1));
        return static_cast<float>(kWalkShare * exp_schedule(_params->match_walk, _params->match_walk_end, t));
    }

    void Match::apply_learning_rates(const int step) {
        const double t = static_cast<double>(std::max(step, 0)) /
                         static_cast<double>(std::max<size_t>(_params->iterations, 1));
        double means_lr = exp_schedule(_params->means_lr, _params->means_lr_end, t);
        if (!_params->match_angular_mean_step)
            means_lr *= _splat_data->get_scene_scale();
        _optimizer->set_param_lr(ParamType::Means, means_lr);
        _optimizer->set_lr(static_cast<float>(means_lr));
        _optimizer->set_param_lr(ParamType::Scaling, exp_schedule(_params->scaling_lr, _params->scaling_lr_end, t));
    }

    void Match::set_training_dataset(std::shared_ptr<CameraDataset> views) {
        _views = std::move(views);
    }

    void Match::set_optimization_params(const lfs::core::param::OptimizationParameters& params) {
        _params = std::make_unique<const lfs::core::param::OptimizationParameters>(params);
    }

    void Match::initialize(const lfs::core::param::OptimizationParameters& optimParams) {
        using namespace lfs::core;
        _params = std::make_unique<const lfs::core::param::OptimizationParameters>(optimParams);
        if (_params->gut)
            throw std::runtime_error("The match strategy needs the FastGS rasterizer (not --gut)");

        if (_params->max_cap > 0) {
            const size_t capacity = static_cast<size_t>(_params->max_cap);
            try {
                auto ensure_capacity_direct = [capacity](Tensor& param) {
                    if (param.capacity() >= capacity || param.is_external_storage())
                        return;
                    auto new_param = Tensor::zeros_direct(param.shape(), capacity);
                    cudaMemcpy(new_param.ptr<float>(), param.ptr<float>(), param.numel() * sizeof(float),
                               cudaMemcpyDeviceToDevice);
                    param = std::move(new_param);
                };
                const auto layout_rest = static_cast<uint32_t>(_splat_data->max_sh_coeffs_rest());
                const bool shN_quantized = _splat_data->shN_value_quantized();
                auto ensure_shN_capacity_direct = [capacity, layout_rest, shN_quantized](Tensor& param) {
                    const size_t required =
                        shN_quantized ? lfs::core::sh_value_quant::sh_value_u16_count(capacity, layout_rest)
                                      : lfs::core::sh_swizzled_float_count(capacity, layout_rest);
                    if (param.capacity() >= required || param.is_external_storage())
                        return;
                    auto new_param = Tensor::zeros_direct(param.shape(), required, Device::CUDA, param.dtype());
                    cudaMemcpy(new_param.data_ptr(), param.data_ptr(), param.numel() * dtype_size(param.dtype()),
                               cudaMemcpyDeviceToDevice);
                    param = std::move(new_param);
                };
                ensure_capacity_direct(_splat_data->means());
                ensure_capacity_direct(_splat_data->sh0());
                if (layout_rest > 0 && _splat_data->shN().is_valid() && _splat_data->shN().numel() > 0)
                    ensure_shN_capacity_direct(_splat_data->shN());
                ensure_capacity_direct(_splat_data->scaling_raw());
                ensure_capacity_direct(_splat_data->rotation_raw());
                ensure_capacity_direct(_splat_data->opacity_raw());
            } catch (const std::exception& e) {
                LOG_WARN("Match: failed to pre-allocate capacity: {}", e.what());
            }
        }
        lfs::training::sh_value::apply_shN_value_quant(*_splat_data);

        _optimizer = create_optimizer(*_splat_data, *_params);
        _optimizer->allocate_gradients(_params->max_cap > 0 ? static_cast<size_t>(_params->max_cap) : 0);
        _optimizer->set_per_splat_bias_correction(_params->per_splat_bias_correction);
        // Full SH degree from step 3: the SH rest is optimized from the start.
        _optimizer->set_sh_warmup_iterations(0);
        apply_learning_rates(0);

        ensure_rows();
        ensure_error_row();
        const size_t n = static_cast<size_t>(_splat_data->size());
        if (n > 0) {
            // The peak blend weight starts at the opacity.
            rows_view(_peak).copy_from(_splat_data->get_opacity().reshape({static_cast<int>(n)}));
        }
        LOG_INFO("Match strategy: {} splats, cap {}, growth x{} per refine every {} steps in ({}, {})", n,
                 _params->max_cap, _params->match_growth, _params->refine_every, _params->start_refine,
                 _params->stop_refine);
    }

    std::pair<lfs::core::Tensor, lfs::core::Tensor> Match::explore_scratch(const int /*iter*/) {
        if (!_params || !_splat_data || _splat_data->size() == 0)
            return {};
        ensure_rows();
        return {rows_view(_visible), rows_view(_step_peak)};
    }

    IStrategy::FusedGeometryHooks Match::fused_geometry_hooks(const int /*iter*/) {
        FusedGeometryHooks hooks;
        if (!_params || !_splat_data || _splat_data->size() == 0)
            return hooks;
        ensure_rows();
        hooks.view_radius = rows_view(_radius);
        hooks.angular_mean_step = _params->match_angular_mean_step;
        hooks.screen_hinge_limit = _params->match_screen_limit;
        hooks.screen_hinge_weight = _params->match_screen_hinge;
        hooks.reg_rendered_only = _params->match_reg_rendered_only;
        return hooks;
    }

    void Match::copy_shN_rows(const lfs::core::Tensor& src, const lfs::core::Tensor& dst) {
        using namespace lfs::core;
        if (!_splat_data->shN().is_valid() || _splat_data->shN().numel() == 0 ||
            _splat_data->max_sh_coeffs_rest() == 0 || dst.numel() == 0)
            return;
        constexpr size_t kStageBytes = 64ull << 20;
        const auto layout_rest = static_cast<size_t>(_splat_data->max_sh_coeffs_rest());
        const size_t pairs = dst.numel();
        const size_t chunk = std::max<size_t>(1, kStageBytes / (layout_rest * 3 * sizeof(float)));
        Tensor staged = Tensor::empty({std::min(pairs, chunk), layout_rest, 3}, _splat_data->shN().device());
        for (size_t first = 0; first < pairs; first += chunk) {
            const size_t count = std::min(chunk, pairs - first);
            Tensor rows = staged.slice(0, 0, count);
            lfs::training::sh_value::gather_shN_to_canonical(*_splat_data, src.slice(0, first, first + count), rows);
            lfs::training::sh_value::scatter_canonical_into_shN(*_splat_data, dst.slice(0, first, first + count),
                                                                rows);
        }
    }

    void Match::split(const lfs::core::Tensor& src, const lfs::core::Tensor& dst, const float share) {
        match::SplitRows rows;
        rows.means = _splat_data->means().ptr<float>();
        rows.log_scales = _splat_data->scaling_raw().ptr<float>();
        rows.raw_rotations = _splat_data->rotation_raw().ptr<float>();
        rows.raw_opacities = _splat_data->opacity_raw().ptr<float>();
        rows.sh0 = _splat_data->sh0().ptr<float>();
        rows.score = _score.ptr<float>();
        rows.score_count = _score_count.ptr<float>();
        rows.oversize = _oversize.ptr<float>();
        rows.peak = _peak.ptr<float>();
        rows.streak = _streak.ptr<float>();
        rows.renders = _renders.ptr<float>();
        match::launch_long_axis_split(rows, src.ptr<int64_t>(), dst.ptr<int64_t>(), src.numel(), share);
    }

    size_t Match::append_split_children(const lfs::core::Tensor& donors, const float share) {
        using namespace lfs::core;
        const size_t count = donors.numel();
        if (count == 0)
            return 0;
        if (!_optimizer->preflight_grow_capacity(count)) {
            LOG_ERROR("Match: growth by {} rows aborted, capacity could not be ensured", count);
            return 0;
        }
        const auto first = static_cast<int64_t>(_splat_data->size());
        for (const auto type : kParamTypes)
            _optimizer->add_new_params_gather(type, donors);
        ensure_rows();
        // Row numbers stay exact in float below 2^24.
        auto children = Tensor::arange(static_cast<float>(first), static_cast<float>(first + static_cast<int64_t>(count)),
                                       1.0f, _splat_data->means().device())
                            .to(DataType::Int64);
        split(donors, children, share);
        return count;
    }

    // Counts in float arithmetic, as in the configuration they reproduce: (int)(0.1f x (float)N) etc.
    size_t Match::split_budget(const lfs::core::param::OptimizationParameters& params, const size_t n) {
        return std::max<size_t>(1, static_cast<size_t>(params.match_split_budget * static_cast<float>(n)));
    }

    size_t Match::growth_births(const lfs::core::param::OptimizationParameters& params, const size_t n,
                                const size_t budget, const size_t revived) {
        const auto grown = static_cast<size_t>(params.match_growth * static_cast<float>(n));
        const size_t cap = params.max_cap > 0 ? static_cast<size_t>(params.max_cap) : grown;
        const size_t target = std::max(n, std::min(cap, grown));
        return std::min(target - n, budget > revived ? budget - revived : 0);
    }

    size_t Match::oversize_births(const lfs::core::param::OptimizationParameters& params, const size_t added) {
        const long drawn = std::lround(params.match_oversize_share * static_cast<float>(added));
        return std::min(added, static_cast<size_t>(std::max(0L, drawn)));
    }

    size_t Match::extra_deaths(const lfs::core::param::OptimizationParameters& params, const int step,
                               const int cap_step, const size_t live) {
        const int stop = static_cast<int>(params.stop_refine);
        if (!(params.match_extra_death_share > 0.0f) || cap_step < 0 || step < cap_step || step >= stop)
            return 0;
        const float taper = static_cast<float>(stop - step) / static_cast<float>(stop - cap_step);
        return static_cast<size_t>(params.match_extra_death_share * taper * static_cast<float>(live));
    }

    Match::RefineCounts Match::refine(const int step) {
        using namespace lfs::core;
        RefineCounts counts;
        const bool shN_expanded = !_splat_data->shN_value_quantized() &&
                                  lfs::training::sh_value::ensure_shN_fp32_for_mutation(*_splat_data);
        lfs::training::sh_value::ShNCommitGuard shn_guard(*_splat_data, shN_expanded, "Match::refine");
        ensure_rows();
        const size_t n = static_cast<size_t>(_splat_data->size());
        if (n == 0)
            return counts;
        const auto device = _splat_data->means().device();
        const float share = split_opacity_share(step);
        const size_t budget = split_budget(*_params, n);

        // Dead pool.
        auto dead = Tensor::empty({n}, device, DataType::UInt8);
        auto eligible = Tensor::empty({n}, device, DataType::UInt8);
        match::DeadSettings dead_settings;
        dead_settings.min_opacity = _params->min_opacity;
        const bool own_peak_threshold = _params->match_peak_death_threshold >= 0.0f;
        dead_settings.dead_peak = own_peak_threshold ? _params->match_peak_death_threshold : _params->match_dead_peak;
        dead_settings.peak_grace = own_peak_threshold ? static_cast<float>(_params->match_death_grace_renders) : 0.0f;
        dead_settings.dead_streak = _params->match_dead_epochs > 0
                                        ? static_cast<float>(_params->match_dead_epochs * steps_per_epoch())
                                        : 0.0f;
        match::launch_dead_mask(_splat_data->means().ptr<float>(), _splat_data->scaling_raw().ptr<float>(),
                                _splat_data->rotation_raw().ptr<float>(), _splat_data->opacity_raw().ptr<float>(),
                                _peak.ptr<float>(), _streak.ptr<float>(), _score.ptr<float>(), dead_settings,
                                dead.ptr<uint8_t>(), eligible.ptr<uint8_t>(), n, _renders.ptr<float>());
        // Extra deaths once at the cap: the live splats the picture needs least (lowest peak blend weight).
        if (_params->max_cap > 0 && n >= static_cast<size_t>(_params->max_cap) && _cap_step < 0)
            _cap_step = step;
        if (_params->match_extra_death_share > 0.0f && _cap_step >= 0) {
            const size_t natural = static_cast<size_t>(dead.to(DataType::Int32).sum().item<int>());
            const size_t extra = extra_deaths(*_params, step, _cap_step, n - natural);
            counts.extra_dead = match::mark_lowest_dead(_peak.ptr<float>(), _renders.ptr<float>(),
                                                        static_cast<float>(_params->match_death_grace_renders),
                                                        dead.ptr<uint8_t>(), eligible.ptr<uint8_t>(), n, extra);
        }
        const auto dead_rows = dead.to(DataType::Bool).nonzero().squeeze(-1);
        counts.dead = dead_rows.numel();
        const size_t live_scored = static_cast<size_t>(eligible.to(DataType::Int32).sum().item<int>());
        const size_t revive = std::min({counts.dead, budget, live_scored});
        if (revive > 0) {
            auto donors = Tensor::empty({revive}, device, DataType::Int64);
            const size_t drawn = match::draw_without_replacement(_score.ptr<float>(), nullptr,
                                                                 eligible.ptr<uint8_t>(), n, revive,
                                                                 seed(step, 0x52455649ULL), donors.ptr<int64_t>());
            if (drawn > 0) {
                donors = donors.slice(0, 0, drawn).contiguous();
                const auto slots = dead_rows.slice(0, 0, drawn).contiguous();
                copy_shN_rows(donors, slots);
                split(donors, slots, share);
                for (const auto type : kParamTypes)
                    _optimizer->relocate_params_at_indices_gpu(type, slots.ptr<int64_t>(), slots.numel());
                counts.revived = drawn;
            }
        }

        // Growth.
        size_t added = growth_births(*_params, n, budget, counts.revived);
        added = clamp_growth_to_vram(added, *_splat_data, _optimizer.get());
        if (added > 0) {
            const size_t oversize_count = oversize_births(*_params, added);
            if (oversize_count > 0) {
                auto donors = Tensor::empty({oversize_count}, device, DataType::Int64);
                const size_t drawn = match::draw_without_replacement(
                    _score.ptr<float>(), _oversize.ptr<float>(), nullptr, n, oversize_count,
                    seed(step, 0x4f564552ULL), donors.ptr<int64_t>());
                counts.oversize_born = append_split_children(donors.slice(0, 0, drawn).contiguous(), share);
            }
            const size_t rest = added - counts.oversize_born;
            const size_t candidates = static_cast<size_t>(_splat_data->size());
            if (rest > 0) {
                auto donors = Tensor::empty({rest}, device, DataType::Int64);
                const size_t drawn = match::draw_without_replacement(_score.ptr<float>(), nullptr, nullptr,
                                                                     candidates, rest, seed(step, 0x47524f57ULL),
                                                                     donors.ptr<int64_t>());
                counts.born = append_split_children(donors.slice(0, 0, drawn).contiguous(), share);
            }
        }

        const size_t n_after = static_cast<size_t>(_splat_data->size());
        match::launch_zero_rows(_score.ptr<float>(), _score_count.ptr<float>(), _oversize.ptr<float>(), nullptr,
                                n_after);
        _last_refine = counts;
        return counts;
    }

    void Match::post_backward(const int iter, RenderOutput& /*render_output*/) {
        LOG_TIMER("Match::post_backward");
        // Full SH degree from step 3 (degree = step).
        while (_splat_data->get_active_sh_degree() < _splat_data->get_max_sh_degree() &&
               _splat_data->get_active_sh_degree() < std::max(iter - 1, 0))
            _splat_data->increment_sh_degree();

        const int step = completed_step(iter);
        // The learning rates and SH degree are for the step this iteration renders.
        apply_learning_rates(std::max(iter - 1, 0));
        ensure_rows();
        ensure_error_row();
        size_t n = static_cast<size_t>(_splat_data->size());
        if (step < 0 || n == 0)
            return;

        // Size clip every 100 steps, from the radii of the step's view.
        if (step % kClipEvery == 0 && _params->match_clip_factor > 1.0f) {
            match::launch_size_clip(_splat_data->scaling_raw().ptr<float>(), _radius.ptr<float>(),
                                    _params->match_screen_limit, _params->match_clip_factor, n);
        }

        const bool scoring = step < static_cast<int>(_params->stop_refine);
        match::StepRows rows;
        rows.error = _splat_data->_densification_info.ptr<float>();
        rows.step_peak = _step_peak.ptr<float>();
        rows.visible = _visible.ptr<float>();
        rows.radius = _radius.ptr<float>();
        rows.score = _score.ptr<float>();
        rows.score_count = _score_count.ptr<float>();
        rows.oversize = _oversize.ptr<float>();
        rows.peak = _peak.ptr<float>();
        rows.streak = _streak.ptr<float>();
        rows.renders = _renders.ptr<float>();
        match::StepSettings settings;
        settings.scoring = scoring;
        settings.screen_limit = _params->match_screen_limit;
        settings.score_power = _params->match_score_power;
        settings.peak_decay = std::pow(0.5f, 1.0f / _params->match_peak_half_life);
        match::launch_step_update(rows, settings, n);

        if (refine_step(step)) {
            const RefineVramWindow vram_window(*_splat_data);
            const auto counts = refine(step);
            n = static_cast<size_t>(_splat_data->size());
            ensure_error_row();
            lfs::core::Tensor::trim_memory_pool();
            LOG_INFO("Match refine step {}: {} splats, {} dead, {} revived, {} born ({} oversize){}", step, n,
                     counts.dead, counts.revived, counts.born + counts.oversize_born, counts.oversize_born,
                     _params->match_extra_death_share > 0.0f ? std::format(", {} extra dead", counts.extra_dead)
                                                             : std::string{});
        }

        const bool gate_peak = _params->match_walk_gate_peak && scoring;
        match::launch_walk(_splat_data->means().ptr<float>(), _splat_data->scaling_raw().ptr<float>(),
                           _splat_data->rotation_raw().ptr<float>(), _splat_data->opacity_raw().ptr<float>(),
                           _visible.ptr<float>(), gate_peak ? _peak.ptr<float>() : nullptr, walk_amplitude(step), n,
                           seed(step, 0x57414c4bULL));
        match::launch_zero_rows(_radius.ptr<float>(), nullptr, nullptr, nullptr, n);
    }

    void Match::step(const int iter) {
        LOG_TIMER("Match::step");
        if (iter < static_cast<int>(_params->iterations)) {
            _optimizer->step(iter);
            _optimizer->zero_grad(iter);
        }
    }

    void Match::remove_gaussians(const lfs::core::Tensor& /*mask*/) {
        LOG_WARN("Match strategy: removing splats is not supported; the dead pool recycles them");
    }

    void Match::serialize(std::ostream& os) const {
        os.write(reinterpret_cast<const char*>(&MATCH_MAGIC), sizeof(MATCH_MAGIC));
        os.write(reinterpret_cast<const char*>(&MATCH_VERSION), sizeof(MATCH_VERSION));
        const uint8_t has_optimizer = _optimizer ? 1 : 0;
        os.write(reinterpret_cast<const char*>(&has_optimizer), sizeof(has_optimizer));
        if (_optimizer)
            _optimizer->serialize(os);
        const uint8_t has_scheduler = 0; // learning rates follow the step; nothing to save
        os.write(reinterpret_cast<const char*>(&has_scheduler), sizeof(has_scheduler));
    }

    void Match::deserialize(std::istream& is) {
        uint32_t magic = 0, version = 0;
        lfs::core::serialization_detail::read_exact(is, &magic, sizeof(magic), "match magic");
        lfs::core::serialization_detail::read_exact(is, &version, sizeof(version), "match version");
        if (magic != MATCH_MAGIC || version != MATCH_VERSION)
            throw std::runtime_error("Invalid match strategy checkpoint");
        uint8_t has_optimizer = 0;
        lfs::core::serialization_detail::read_exact(is, &has_optimizer, sizeof(has_optimizer), "match optimizer flag");
        if (has_optimizer > 1 || (has_optimizer && !_optimizer))
            throw std::runtime_error("Invalid match checkpoint: optimizer flag/state mismatch");
        if (has_optimizer)
            _optimizer->deserialize(is);
        uint8_t has_scheduler = 0;
        lfs::core::serialization_detail::read_exact(is, &has_scheduler, sizeof(has_scheduler), "match scheduler flag");
        if (has_scheduler != 0)
            throw std::runtime_error("Invalid match checkpoint: unexpected scheduler state");
        // The strategy rows start afresh; the peak restarts at the opacity.
        ensure_rows();
        const size_t n = static_cast<size_t>(_splat_data->size());
        if (n > 0)
            rows_view(_peak).copy_from(_splat_data->get_opacity().reshape({static_cast<int>(n)}));
    }

    bool Match::can_adopt_checkpoint_state(const IStrategy& loaded) const noexcept {
        const auto* source = dynamic_cast<const Match*>(&loaded);
        return source && static_cast<bool>(_optimizer) == static_cast<bool>(source->_optimizer);
    }

    void Match::adopt_checkpoint_state(IStrategy& loaded) noexcept {
        auto& source = checked_checkpoint_source<Match>(loaded);
        if (_optimizer)
            _optimizer->adopt_checkpoint_state(*source._optimizer);
        _params.swap(source._params);
        for (auto [mine, theirs] : {std::pair{&_score, &source._score}, std::pair{&_score_count, &source._score_count},
                                    std::pair{&_oversize, &source._oversize}, std::pair{&_peak, &source._peak},
                                    std::pair{&_streak, &source._streak}, std::pair{&_visible, &source._visible},
                                    std::pair{&_step_peak, &source._step_peak}, std::pair{&_radius, &source._radius},
                                    std::pair{&_renders, &source._renders}})
            std::swap(*mine, *theirs);
        std::swap(_cap_step, source._cap_step);
    }

    void Match::reserve_optimizer_capacity(const size_t capacity) {
        if (_optimizer)
            _optimizer->reserve_capacity(capacity);
    }

} // namespace lfs::training
