/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "popspa_controller.hpp"
#include "core/tensor/internal/tensor_serialization.hpp"
#include "kernels/kernel_stream.hpp"
#include "lfs/kernels/popspa.cuh"
#include "losses/effective_rank_regularization.hpp"
#include <algorithm>
#include <cmath>
#include <format>
#include <istream>
#include <limits>
#include <numeric>
#include <ostream>
#include <stdexcept>
#include <utility>

namespace lfs::training {
    namespace {
        using core::DataType;
        using core::Device;
        using core::Tensor;
        constexpr uint32_t magic = 0x5053504c; // LPSP
        constexpr uint32_t version = 1;
        // Fixed-width header: magic/version/phase/steps, then eleven uint64s and four floats.
        constexpr size_t header_bytes = 6 * sizeof(uint32_t) + 11 * sizeof(uint64_t) + 4 * sizeof(float);
        lfs::Error failure(const std::exception& e, lfs::ErrorCode code = lfs::ErrorCode::InvalidArgument) {
            return lfs::make_error({.code = code, .domain = lfs::ErrorDomain::Training, .user_message = "POPSpa could not complete the requested operation.", .detail = e.what(), .detection = LFS_SOURCE_SITE_CURRENT()});
        }
        void require(bool condition, const std::string& detail) {
            LFS_ASSERT_MSG(condition, detail);
        }
        void validate_opacity(const Tensor& opacity, size_t n) {
            require(opacity.is_valid() && opacity.device() == Device::CUDA && opacity.dtype() == DataType::Float32 &&
                        opacity.is_contiguous() && (opacity.ndim() == 1 || (opacity.ndim() == 2 && opacity.size(1) == 1)) && opacity.numel() == n,
                    std::format("POPSpa opacity must be contiguous CUDA float32 [N] or [N,1] (shape={}, dtype={}, device={}, N={})",
                                opacity.shape().str(), static_cast<int>(opacity.dtype()), static_cast<int>(opacity.device()), n));
        }
        void validate_config(const POPSpaController::Config& c, size_t n) {
            require(c.target_count > 0 && c.target_count <= c.first_prune_count && c.first_prune_count <= n,
                    std::format("POPSpa budgets must satisfy 0 < K <= K1 <= N (K={}, K1={}, N={})", c.target_count, c.first_prune_count, n));
            require(c.projection_interval > 0 && std::isfinite(c.rho) && c.rho >= 0 &&
                        std::isfinite(c.erank_weight) && c.erank_weight >= 0 && std::isfinite(c.thin_scale_weight) && c.thin_scale_weight >= 0 &&
                        std::isfinite(c.erank_epsilon) && c.erank_epsilon > 0,
                    std::format("POPSpa projection interval must be positive and weights finite/nonnegative (interval={}, rho={}, rank={}, thin={}, epsilon={})",
                                c.projection_interval, c.rho, c.erank_weight, c.thin_scale_weight, c.erank_epsilon));
        }
        template <class T>
        void write(std::ostream& os, T value) {
            os.write(reinterpret_cast<const char*>(&value), sizeof(value));
        }
        template <class T>
        T read(std::istream& is) {
            T value{};
            core::serialization_detail::read_exact(is, &value, sizeof(value), "POPSpa component header");
            return value;
        }
        size_t tensor_bytes(const Tensor& tensor) {
            const auto overhead = sizeof(core::TensorFileHeader) + tensor.ndim() * sizeof(uint64_t);
            require(tensor.bytes() <= std::numeric_limits<size_t>::max() - overhead,
                    std::format("POPSpa serialized tensor size overflows (bytes={}, overhead={})", tensor.bytes(), overhead));
            return overhead + tensor.bytes();
        }
    } // namespace

    lfs::Result<std::vector<int64_t>> POPSpaController::select_keep_indices(
        std::span<const double> scores, size_t count, std::span<const uint8_t> active, std::span<const uint8_t> frozen) {
        try {
            const size_t n = scores.size();
            require((active.empty() || active.size() == n) && (frozen.empty() || frozen.size() == n),
                    std::format("POPSpa selection mask sizes must match scores (scores={}, active={}, frozen={})", n, active.size(), frozen.size()));
            std::vector<int64_t> candidates, keep;
            for (size_t i = 0; i < n; ++i) {
                const bool a = active.empty() || active[i];
                const bool f = !frozen.empty() && frozen[i];
                require(!f || a, std::format("POPSpa frozen row must be active (row={})", i));
                require(std::isfinite(scores[i]) && scores[i] >= 0,
                        std::format("POPSpa score must be finite and nonnegative (row={}, score={})", i, scores[i]));
                if (f)
                    keep.push_back(static_cast<int64_t>(i));
                else if (a)
                    candidates.push_back(static_cast<int64_t>(i));
            }
            require(count > 0 && count >= keep.size() && count <= keep.size() + candidates.size(),
                    std::format("POPSpa exact retention budget must include frozen rows and fit active rows (K={}, frozen={}, active={})",
                                count, keep.size(), keep.size() + candidates.size()));
            std::sort(candidates.begin(), candidates.end(), [&](int64_t a, int64_t b) {
                return scores[a] == scores[b] ? a < b : scores[a] > scores[b];
            });
            keep.insert(keep.end(), candidates.begin(), candidates.begin() + (count - keep.size()));
            std::sort(keep.begin(), keep.end());
            return keep;
        } catch (const std::exception& e) {
            // LFS-CENSUS-OK(empty-catch): failure() returns a typed error carrying the exception detail.
            return failure(e);
        }
    }

    void POPSpaController::set_masks(const Tensor& active, const Tensor& frozen) {
        for (const auto* mask : {&active, &frozen}) {
            require(!mask->is_valid() || (mask->device() == Device::CUDA && mask->dtype() == DataType::Bool && mask->ndim() == 1 &&
                                          mask->numel() == rows_ && mask->is_contiguous()),
                    std::format("POPSpa mask must be contiguous CUDA bool [N] (shape={}, N={})", mask->shape().str(), rows_));
        }
        active_ = active.is_valid() ? active.clone() : Tensor::from_vector(std::vector<bool>(rows_, true), {rows_}, Device::CUDA);
        frozen_ = frozen.is_valid() ? frozen.clone() : Tensor::zeros_bool({rows_}, Device::CUDA);
        const auto a = active_.to_vector_bool(), f = frozen_.to_vector_bool();
        active_count_ = frozen_count_ = 0;
        for (size_t i = 0; i < rows_; ++i) {
            require(!f[i] || a[i], std::format("POPSpa frozen row must be active (row={})", i));
            active_count_ += a[i];
            frozen_count_ += f[i];
        }
        require(config_.target_count >= frozen_count_ && config_.target_count <= active_count_,
                std::format("POPSpa target must include frozen rows and fit active rows (K={}, frozen={}, active={})",
                            config_.target_count, frozen_count_, active_count_));
    }

    lfs::Status POPSpaController::initialize(const Config& config, const Tensor& opacities,
                                             size_t training_view_count, const Tensor& active, const Tensor& frozen) {
        try {
            POPSpaController next;
            next.config_ = config;
            next.rows_ = next.input_rows_ = opacities.numel();
            validate_opacity(opacities, next.rows_);
            for (const float value : opacities.to_vector())
                require(std::isfinite(value), std::format("POPSpa initial opacity must be finite (value={})", value));
            next.set_masks(active, frozen);
            next.input_active_count_ = next.active_count_;
            // N is the number of active Gaussians, not unused strategy capacity.
            if (next.config_.first_prune_count == 0)
                next.config_.first_prune_count = std::max(config.target_count, static_cast<size_t>(
                                                                                   std::round(std::sqrt(static_cast<long double>(next.active_count_) * config.target_count))));
            validate_config(next.config_, next.active_count_);
            require(training_view_count > 0, std::format("POPSpa requires training views (count={})", training_view_count));
            next.view_count_ = training_view_count;
            next.scores_ = Tensor::zeros({next.rows_, sizeof(double)}, Device::CUDA, DataType::UInt8);
            next.z_ = Tensor::zeros(opacities.shape(), Device::CUDA);
            next.u_ = Tensor::zeros(opacities.shape(), Device::CUDA);
            next.initialized_ = true;
            adopt_checkpoint_state(next);
            return {};
        } catch (const std::exception& e) {
            // LFS-CENSUS-OK(empty-catch): failure() returns a typed error carrying the exception detail.
            return lfs::Status::failure(failure(e));
        }
    }

    uint64_t POPSpaController::additional_iterations() const noexcept {
        return static_cast<uint64_t>(config_.sparsify_steps) + config_.refine_steps;
    }
    bool POPSpaController::projection_due() const noexcept {
        return initialized_ && phase_ == POPSpaPhase::Sparsify && (phase_step_ + 1) % config_.projection_interval == 0;
    }
    lfs::Status POPSpaController::finish_score_view() {
        try {
            require(initialized_ && (phase_ == POPSpaPhase::FirstScore || phase_ == POPSpaPhase::SecondScore) && score_view_ < view_count_,
                    std::format("POPSpa score completion requires an unfinished score phase (phase={}, view={}, count={})", static_cast<int>(phase_), score_view_, view_count_));
            if (++score_view_ == view_count_)
                phase_ = phase_ == POPSpaPhase::FirstScore ? POPSpaPhase::FirstPrune : POPSpaPhase::FinalPrune;
            return {};
        } catch (const std::exception& e) {
            // LFS-CENSUS-OK(empty-catch): failure() returns a typed error carrying the exception detail.
            return lfs::Status::failure(failure(e));
        }
    }
    lfs::Result<Tensor> POPSpaController::prune_indices() const {
        try {
            validate_state(rows_);
            require(initialized_ && (phase_ == POPSpaPhase::FirstPrune || phase_ == POPSpaPhase::FinalPrune),
                    std::format("POPSpa prune indices require a prune phase (phase={})", static_cast<int>(phase_)));
            const size_t k = phase_ == POPSpaPhase::FirstPrune ? config_.first_prune_count : config_.target_count;
            // A cold, twice-per-run validation prevents NaNs from violating the GPU
            // sort's strict weak ordering. Projection never performs a CPU readback.
            auto cpu = scores_.cpu();
            const auto* values = reinterpret_cast<const double*>(cpu.data_ptr());
            for (size_t i = 0; i < rows_; ++i)
                require(std::isfinite(values[i]) && values[i] >= 0,
                        std::format("POPSpa score must be finite/nonnegative (row={}, score={})", i, values[i]));
            auto scratch = Tensor::empty({rows_}, Device::CUDA, DataType::Int64);
            auto keep = Tensor::empty({k}, Device::CUDA, DataType::Int64);
            const auto stream = resolve_stream(scores_.stream());
            scores_.sync_to_stream(stream);
            active_.sync_to_stream(stream);
            frozen_.sync_to_stream(stream);
            scratch.sync_to_stream(stream);
            keep.sync_to_stream(stream);
            kernels::launch_popspa_keep_indices(reinterpret_cast<const double*>(scores_.data_ptr()), active_.ptr<bool>(), frozen_.ptr<bool>(), rows_, k,
                                                scratch.ptr<int64_t>(), keep.ptr<int64_t>(), stream);
            scratch.set_stream(stream);
            keep.set_stream(stream);
            return keep;
        } catch (const std::exception& e) {
            // LFS-CENSUS-OK(empty-catch): failure() returns a typed error carrying the exception detail.
            return failure(e);
        }
    }
    void POPSpaController::project(const Tensor& opacities, bool update_dual) {
        validate_opacity(opacities, rows_);
        if (!projection_values_.is_valid() || projection_values_.size(0) != rows_)
            projection_values_ = Tensor::empty({rows_, sizeof(double)}, Device::CUDA, DataType::UInt8);
        if (!projection_indices_.is_valid() || projection_indices_.numel() != rows_)
            projection_indices_ = Tensor::empty({rows_}, Device::CUDA, DataType::Int64);
        if (!projection_keep_.is_valid() || projection_keep_.numel() != config_.target_count)
            projection_keep_ = Tensor::empty({config_.target_count}, Device::CUDA, DataType::Int64);
        const auto stream = resolve_stream(opacities.stream());
        opacities.sync_to_stream(stream);
        for (const auto* tensor : {&z_, &u_, &active_, &frozen_, &projection_values_, &projection_indices_, &projection_keep_})
            tensor->sync_to_stream(stream);
        kernels::launch_popspa_project(opacities.ptr<float>(), z_.ptr<float>(), u_.ptr<float>(), active_.ptr<bool>(), frozen_.ptr<bool>(),
                                       rows_, config_.target_count, reinterpret_cast<double*>(projection_values_.data_ptr()), projection_indices_.ptr<int64_t>(), projection_keep_.ptr<int64_t>(), update_dual, stream);
        for (const auto* tensor : {&z_, &u_, &active_, &frozen_, &projection_values_, &projection_indices_, &projection_keep_})
            tensor->record_stream(stream);
        for (auto* tensor : {&z_, &u_, &projection_values_, &projection_indices_, &projection_keep_})
            tensor->set_stream(stream);
    }
    lfs::Status POPSpaController::accept_prune(const Tensor& opacities, const Tensor& active, const Tensor& frozen) {
        try {
            require(initialized_ && (phase_ == POPSpaPhase::FirstPrune || phase_ == POPSpaPhase::FinalPrune),
                    std::format("POPSpa accept_prune requires prune phase (phase={})", static_cast<int>(phase_)));
            const bool first = phase_ == POPSpaPhase::FirstPrune;
            const size_t expected = first ? config_.first_prune_count : config_.target_count;
            validate_opacity(opacities, expected);
            POPSpaController next = *this;
            next.projection_values_ = {};
            next.projection_indices_ = {};
            next.projection_keep_ = {};
            next.rows_ = expected;
            next.set_masks(active, frozen);
            next.scores_ = Tensor::zeros({expected, sizeof(double)}, Device::CUDA, DataType::UInt8);
            next.z_ = Tensor::zeros(opacities.shape(), Device::CUDA);
            next.u_ = Tensor::zeros(opacities.shape(), Device::CUDA);
            next.phase_step_ = 0;
            next.score_view_ = 0;
            next.phase_ = first ? (config_.sparsify_steps ? POPSpaPhase::Sparsify : POPSpaPhase::SecondScore)
                                : (config_.refine_steps ? POPSpaPhase::Recover : POPSpaPhase::Complete);
            if (first)
                next.project(opacities, false);
            adopt_checkpoint_state(next);
            return {};
        } catch (const std::exception& e) {
            // LFS-CENSUS-OK(empty-catch): failure() returns a typed error carrying the exception detail.
            return lfs::Status::failure(failure(e));
        }
    }
    lfs::Result<Tensor> POPSpaController::regularization(const Tensor& opacities, const Tensor& log_scales,
                                                         Tensor& opacity_grad, Tensor& scales_grad) {
        try {
            require(initialized_ && (phase_ == POPSpaPhase::Sparsify || phase_ == POPSpaPhase::Recover),
                    std::format("POPSpa regularization requires optimization phase (phase={})", static_cast<int>(phase_)));
            validate_opacity(opacities, rows_);
            if (phase_ == POPSpaPhase::Recover)
                return Tensor::zeros({1}, Device::CUDA);
            require(opacity_grad.shape() == opacities.shape(),
                    std::format("POPSpa opacity gradient shape must match opacity (gradient={}, opacity={})", opacity_grad.shape().str(), opacities.shape().str()));
            validate_opacity(opacity_grad, rows_);
            require(log_scales.is_valid() && log_scales.ndim() == 2 && log_scales.size(0) == rows_,
                    std::format("POPSpa scales must match model rows (shape={}, N={})", log_scales.shape().str(), rows_));
            auto result = losses::EffectiveRankRegularization::forward(log_scales, scales_grad,
                                                                       {.erank_weight = config_.erank_weight, .thin_scale_weight = config_.thin_scale_weight, .epsilon = config_.erank_epsilon},
                                                                       active_, frozen_, active_count_);
            if (!result)
                return result.error();
            auto loss = std::move(result.value());
            if (projection_due()) {
                const auto stream = resolve_stream(opacities.stream());
                opacities.sync_to_stream(stream);
                for (const auto* tensor : {&z_, &u_, &active_, &frozen_, &opacity_grad, &loss})
                    tensor->sync_to_stream(stream);
                kernels::launch_popspa_penalty(opacities.ptr<float>(), z_.ptr<float>(), u_.ptr<float>(), active_.ptr<bool>(), frozen_.ptr<bool>(),
                                               opacity_grad.ptr<float>(), loss.ptr<float>(), rows_, config_.rho, stream);
                for (const auto* tensor : {&z_, &u_, &active_, &frozen_, &opacity_grad, &loss})
                    tensor->record_stream(stream);
                opacity_grad.set_stream(stream);
                loss.set_stream(stream);
            }
            return loss;
        } catch (const std::exception& e) {
            // LFS-CENSUS-OK(empty-catch): failure() returns a typed error carrying the exception detail.
            return failure(e);
        }
    }
    lfs::Status POPSpaController::after_backward(const Tensor& opacities) {
        try {
            require(initialized_ && (phase_ == POPSpaPhase::Sparsify || phase_ == POPSpaPhase::Recover),
                    std::format("POPSpa ADMM update requires optimization phase (phase={})", static_cast<int>(phase_)));
            if (projection_due())
                project(opacities, true);
            return {};
        } catch (const std::exception& e) {
            // LFS-CENSUS-OK(empty-catch): failure() returns a typed error carrying the exception detail.
            return lfs::Status::failure(failure(e));
        }
    }
    lfs::Status POPSpaController::finish_optimization_step() {
        try {
            require(initialized_ && (phase_ == POPSpaPhase::Sparsify || phase_ == POPSpaPhase::Recover),
                    std::format("POPSpa step completion requires optimization phase (phase={})", static_cast<int>(phase_)));
            const auto limit = phase_ == POPSpaPhase::Sparsify ? config_.sparsify_steps : config_.refine_steps;
            require(phase_step_ < limit, std::format("POPSpa step must precede phase limit (step={}, limit={})", phase_step_, limit));
            if (++phase_step_ == limit) {
                phase_ = phase_ == POPSpaPhase::Sparsify ? POPSpaPhase::SecondScore : POPSpaPhase::Complete;
                phase_step_ = 0;
                score_view_ = 0;
                scores_.zero_();
            }
            return {};
        } catch (const std::exception& e) {
            // LFS-CENSUS-OK(empty-catch): failure() returns a typed error carrying the exception detail.
            return lfs::Status::failure(failure(e));
        }
    }
    void POPSpaController::reset() noexcept {
        POPSpaController empty;
        adopt_checkpoint_state(empty);
    }
    void POPSpaController::adopt_checkpoint_state(POPSpaController& loaded) noexcept {
        using std::swap;
        swap(config_, loaded.config_);
        swap(initialized_, loaded.initialized_);
        swap(phase_, loaded.phase_);
        swap(phase_step_, loaded.phase_step_);
        swap(score_view_, loaded.score_view_);
        swap(view_count_, loaded.view_count_);
        swap(rows_, loaded.rows_);
        swap(input_rows_, loaded.input_rows_);
        swap(input_active_count_, loaded.input_active_count_);
        swap(active_count_, loaded.active_count_);
        swap(frozen_count_, loaded.frozen_count_);
        swap(scores_, loaded.scores_);
        swap(z_, loaded.z_);
        swap(u_, loaded.u_);
        swap(active_, loaded.active_);
        swap(frozen_, loaded.frozen_);
        // Scratch is derived and may have a different topology or stream. Never
        // adopt shared scratch from temporary copied/checkpoint state.
        projection_values_ = {};
        projection_indices_ = {};
        projection_keep_ = {};
        loaded.projection_values_ = {};
        loaded.projection_indices_ = {};
        loaded.projection_keep_ = {};
    }

    void POPSpaController::validate_state(size_t expected_rows, bool inspect_contents) const {
        require(initialized_ && rows_ > 0 && rows_ == expected_rows && view_count_ > 0,
                std::format("POPSpa checkpoint must be initialized and match model/views (initialized={}, rows={}, expected={}, views={})", initialized_, rows_, expected_rows, view_count_));
        validate_config(config_, input_active_count_);
        require(input_active_count_ <= input_rows_, std::format("POPSpa initial active count exceeds original storage rows (active={}, rows={})", input_active_count_, input_rows_));
        require(static_cast<uint32_t>(phase_) <= static_cast<uint32_t>(POPSpaPhase::Complete),
                std::format("POPSpa checkpoint phase is unknown (phase={})", static_cast<uint32_t>(phase_)));
        const bool scoring = phase_ == POPSpaPhase::FirstScore || phase_ == POPSpaPhase::SecondScore;
        const bool pruning = phase_ == POPSpaPhase::FirstPrune || phase_ == POPSpaPhase::FinalPrune;
        require((scoring && score_view_ < view_count_) || (pruning && score_view_ == view_count_) || (!scoring && !pruning && score_view_ == 0),
                std::format("POPSpa checkpoint score cursor disagrees with phase (phase={}, view={}, views={})", static_cast<int>(phase_), score_view_, view_count_));
        const uint32_t limit = phase_ == POPSpaPhase::Sparsify ? config_.sparsify_steps : config_.refine_steps;
        const bool optimization = phase_ == POPSpaPhase::Sparsify || phase_ == POPSpaPhase::Recover;
        require(optimization ? phase_step_ < limit : phase_step_ == 0,
                std::format("POPSpa checkpoint step disagrees with phase (phase={}, step={}, limit={})", static_cast<int>(phase_), phase_step_, limit));
        const size_t expected_phase_rows = (phase_ == POPSpaPhase::FirstScore || phase_ == POPSpaPhase::FirstPrune) ? input_rows_
                                           : (phase_ == POPSpaPhase::Recover || phase_ == POPSpaPhase::Complete)    ? config_.target_count
                                                                                                                    : config_.first_prune_count;
        require(rows_ == expected_phase_rows,
                std::format("POPSpa checkpoint topology disagrees with phase (phase={}, rows={}, expected={})", static_cast<int>(phase_), rows_, expected_phase_rows));
        require(scores_.is_valid() && scores_.dtype() == DataType::UInt8 && scores_.ndim() == 2 && scores_.size(0) == rows_ && scores_.size(1) == sizeof(double) && scores_.is_contiguous(),
                std::format("POPSpa checkpoint scores must be packed FP64 UInt8 [N,8] (shape={}, dtype={}, N={})", scores_.shape().str(), static_cast<int>(scores_.dtype()), rows_));
        for (const auto* tensor : {&z_, &u_}) {
            require(tensor->is_valid() && tensor->dtype() == DataType::Float32 && tensor->numel() == rows_ &&
                        (tensor->ndim() == 1 || (tensor->ndim() == 2 && tensor->size(1) == 1)) && tensor->is_contiguous(),
                    std::format("POPSpa checkpoint ADMM tensor must be float32 [N] or [N,1] (shape={}, dtype={}, N={})", tensor->shape().str(), static_cast<int>(tensor->dtype()), rows_));
            if (inspect_contents)
                for (float value : tensor->to_vector())
                    require(std::isfinite(value), std::format("POPSpa checkpoint ADMM value must be finite (value={})", value));
        }
        require(z_.shape() == u_.shape(), std::format("POPSpa checkpoint ADMM shapes differ (z={}, u={})", z_.shape().str(), u_.shape().str()));
        for (const auto* tensor : {&active_, &frozen_})
            require(tensor->is_valid() && tensor->dtype() == DataType::Bool && tensor->ndim() == 1 && tensor->numel() == rows_,
                    std::format("POPSpa checkpoint mask must be bool [N] (shape={}, dtype={}, N={})", tensor->shape().str(), static_cast<int>(tensor->dtype()), rows_));
        require(frozen_count_ <= config_.target_count && config_.target_count <= active_count_ && active_count_ <= rows_ &&
                    ((phase_ != POPSpaPhase::FirstScore && phase_ != POPSpaPhase::FirstPrune) || config_.first_prune_count <= active_count_),
                std::format("POPSpa checkpoint count metadata violates budgets (active={}, frozen={}, rows={}, K={}, K1={})",
                            active_count_, frozen_count_, rows_, config_.target_count, config_.first_prune_count));
        // Writer/sizer trust values already validated at mutation boundaries. Only
        // an untrusted checkpoint reader inspects tensor contents on the host.
        if (!inspect_contents)
            return;
        const auto a = active_.to_vector_bool(), f = frozen_.to_vector_bool();
        size_t active = 0, frozen = 0;
        auto score_cpu = scores_.cpu();
        for (size_t i = 0; i < rows_; ++i) {
            require(!f[i] || a[i], std::format("POPSpa checkpoint frozen row must be active (row={})", i));
            active += a[i];
            frozen += f[i];
            const double value = reinterpret_cast<const double*>(score_cpu.data_ptr())[i];
            require(std::isfinite(value) && value >= 0, std::format("POPSpa checkpoint score must be finite/nonnegative (row={}, value={})", i, value));
        }
        require(active == active_count_ && frozen == frozen_count_ && frozen <= config_.target_count && config_.target_count <= active &&
                    ((phase_ != POPSpaPhase::FirstScore && phase_ != POPSpaPhase::FirstPrune) || config_.first_prune_count <= active),
                std::format("POPSpa checkpoint counts/budgets disagree (active={}/{}, frozen={}/{}, K={}, K1={})", active, active_count_, frozen, frozen_count_, config_.target_count, config_.first_prune_count));
    }
    lfs::Status POPSpaController::serialize(std::ostream& os) const {
        try {
            validate_state(rows_);
            write(os, magic);
            write(os, version);
            write(os, static_cast<uint32_t>(phase_));
            write(os, phase_step_);
            write(os, config_.sparsify_steps);
            write(os, config_.refine_steps);
            for (uint64_t value : {uint64_t(config_.target_count), uint64_t(config_.first_prune_count), uint64_t(config_.projection_interval),
                                   config_.camera_fingerprint, uint64_t(rows_), uint64_t(input_rows_), uint64_t(view_count_),
                                   uint64_t(score_view_), uint64_t(active_count_), uint64_t(frozen_count_), uint64_t(input_active_count_)})
                write(os, value);
            write(os, config_.rho);
            write(os, config_.erank_weight);
            write(os, config_.thin_scale_weight);
            write(os, config_.erank_epsilon);
            os << scores_ << z_ << u_ << active_ << frozen_;
            require(os.good(), "POPSpa checkpoint write failed (stream.good=false)");
            return {};
        } catch (const std::exception& e) {
            // LFS-CENSUS-OK(empty-catch): failure() returns a typed error carrying the exception detail.
            return lfs::Status::failure(failure(e, lfs::ErrorCode::DataLoss));
        }
    }
    POPSpaController POPSpaController::read_serialized(std::istream& is) {
        const auto observed_magic = read<uint32_t>(is), observed_version = read<uint32_t>(is);
        require(observed_magic == magic && observed_version == version,
                std::format("POPSpa checkpoint magic/version mismatch (magic={}, version={}, expected_magic={}, expected_version={})", observed_magic, observed_version, magic, version));
        POPSpaController next;
        next.phase_ = static_cast<POPSpaPhase>(read<uint32_t>(is));
        next.phase_step_ = read<uint32_t>(is);
        next.config_.sparsify_steps = read<uint32_t>(is);
        next.config_.refine_steps = read<uint32_t>(is);
        next.config_.target_count = read<uint64_t>(is);
        next.config_.first_prune_count = read<uint64_t>(is);
        const auto interval = read<uint64_t>(is);
        require(interval <= std::numeric_limits<uint32_t>::max(), std::format("POPSpa checkpoint projection interval overflows (value={})", interval));
        next.config_.projection_interval = static_cast<uint32_t>(interval);
        next.config_.camera_fingerprint = read<uint64_t>(is);
        next.rows_ = read<uint64_t>(is);
        next.input_rows_ = read<uint64_t>(is);
        next.view_count_ = read<uint64_t>(is);
        next.score_view_ = read<uint64_t>(is);
        next.active_count_ = read<uint64_t>(is);
        next.frozen_count_ = read<uint64_t>(is);
        next.input_active_count_ = read<uint64_t>(is);
        next.config_.rho = read<float>(is);
        next.config_.erank_weight = read<float>(is);
        next.config_.thin_scale_weight = read<float>(is);
        next.config_.erank_epsilon = read<float>(is);
        is >> next.scores_ >> next.z_ >> next.u_ >> next.active_ >> next.frozen_;
        next.initialized_ = true;
        next.validate_state(next.rows_, true);
        return next;
    }
    lfs::Status POPSpaController::deserialize(std::istream& is) {
        try {
            auto next = read_serialized(is);
            next.scores_ = next.scores_.cuda();
            next.z_ = next.z_.cuda();
            next.u_ = next.u_.cuda();
            next.active_ = next.active_.cuda();
            next.frozen_ = next.frozen_.cuda();
            adopt_checkpoint_state(next);
            return {};
        } catch (const std::exception& e) {
            // LFS-CENSUS-OK(empty-catch): failure() returns a typed error carrying the exception detail.
            return lfs::Status::failure(failure(e, lfs::ErrorCode::DataLoss));
        }
    }
    lfs::Result<size_t> POPSpaController::checkpoint_size_bytes(size_t expected_rows) const {
        try {
            validate_state(expected_rows);
            size_t bytes = header_bytes;
            for (const auto* tensor : {&scores_, &z_, &u_, &active_, &frozen_}) {
                const auto extra = tensor_bytes(*tensor);
                require(bytes <= std::numeric_limits<size_t>::max() - extra,
                        std::format("POPSpa checkpoint size overflows (bytes={}, extra={})", bytes, extra));
                bytes += extra;
            }
            return bytes;
        } catch (const std::exception& e) {
            // LFS-CENSUS-OK(empty-catch): failure() returns a typed error carrying the exception detail.
            return failure(e, lfs::ErrorCode::DataLoss);
        }
    }
    lfs::Result<size_t> POPSpaController::consume_checkpoint(std::istream& is) {
        try {
            return read_serialized(is).rows_;
        } catch (const std::exception& e) {
            // LFS-CENSUS-OK(empty-catch): failure() returns a typed error carrying the exception detail.
            return failure(e, lfs::ErrorCode::DataLoss);
        }
    }
} // namespace lfs::training
