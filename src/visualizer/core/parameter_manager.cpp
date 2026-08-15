/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "parameter_manager.hpp"
#include "core/logger.hpp"
#include "io/project_chapters.hpp"

#include <cassert>
#include <cmath>

namespace lfs::vis {

    namespace {
        constexpr size_t BASE_IMAGE_COUNT = 300;

        lfs::Error parameter_project_error(const lfs::ErrorCode code,
                                           std::string message) {
            return lfs::make_error(lfs::ErrorInit{
                .code = code,
                .domain = lfs::ErrorDomain::App,
                .severity = lfs::Severity::Error,
                .retryability = lfs::Retryability::NotRetryable,
                .operation_id = {},
                .user_message = message,
                .detail = std::move(message),
                .detection = LFS_SOURCE_SITE_CURRENT(),
                .fields = {},
                .native = std::nullopt,
            });
        }

        void apply_scaler_to_params(lfs::core::param::OptimizationParameters& p, const float new_scaler) {
            const float prev = p.steps_scaler;
            p.steps_scaler = new_scaler;
            if (new_scaler <= 0.0f)
                return;
            const float ratio = (prev > 0.0f) ? (new_scaler / prev) : new_scaler;
            if (std::abs(ratio - 1.0f) < 0.001f)
                return;
            p.scale_steps(ratio);
        }
    } // namespace

    std::expected<void, std::string> ParameterManager::ensureLoaded() {
        if (loaded_)
            return {};

        mcmc_session_ = lfs::core::param::OptimizationParameters::mcmc_defaults();
        mcmc_current_ = mcmc_session_;
        mrnf_session_ = lfs::core::param::OptimizationParameters::mrnf_defaults();
        mrnf_current_ = mrnf_session_;
        igs_session_ = lfs::core::param::OptimizationParameters::igs_plus_defaults();
        igs_current_ = igs_session_;
        dataset_config_.loading_params = lfs::core::param::LoadingParams{};

        loaded_ = true;
        return {};
    }

    lfs::Result<lfs::io::project::ParameterManagerSnapshot>
    ParameterManager::capturePendingProjectState() const {
        std::lock_guard lock(params_mutex_);
        if (!loaded_) {
            return parameter_project_error(
                lfs::ErrorCode::FailedPrecondition,
                "ParameterManager is not loaded");
        }
        return lfs::io::project::ParameterManagerSnapshot{
            .active_strategy = active_strategy_,
            .mcmc_session = mcmc_session_,
            .mrnf_session = mrnf_session_,
            .igs_session = igs_session_,
            .mcmc_current = mcmc_current_,
            .mrnf_current = mrnf_current_,
            .igs_current = igs_current_,
            .mcmc_session_references =
                mcmc_session_references_,
            .mrnf_session_references =
                mrnf_session_references_,
            .igs_session_references =
                igs_session_references_,
            .mcmc_current_references =
                mcmc_current_references_,
            .mrnf_current_references =
                mrnf_current_references_,
            .igs_current_references =
                igs_current_references_,
            .dataset = dataset_config_,
        };
    }

    lfs::Result<void>
    ParameterManager::validatePendingProjectState(
        const lfs::io::project::ParameterManagerSnapshot& snapshot) {
        if (!lfs::core::param::is_valid_strategy_name(snapshot.active_strategy)) {
            return lfs::Result<void>::failure(parameter_project_error(
                lfs::ErrorCode::InvalidArgument,
                "Pending project strategy is invalid"));
        }
        const std::array params{
            &snapshot.mcmc_session,
            &snapshot.mrnf_session,
            &snapshot.igs_session,
            &snapshot.mcmc_current,
            &snapshot.mrnf_current,
            &snapshot.igs_current,
        };
        for (const auto* value : params) {
            if (const std::string error = value->validate(); !error.empty()) {
                return lfs::Result<void>::failure(parameter_project_error(
                    lfs::ErrorCode::InvalidArgument,
                    "Pending project parameters are invalid: " + error));
            }
        }
        const bool role_mismatch =
            snapshot.mcmc_session.strategy !=
                lfs::core::param::kStrategyMCMC ||
            snapshot.mcmc_current.strategy !=
                lfs::core::param::kStrategyMCMC ||
            !lfs::core::param::is_mrnf_strategy(
                snapshot.mrnf_session.strategy) ||
            !lfs::core::param::is_mrnf_strategy(
                snapshot.mrnf_current.strategy) ||
            snapshot.igs_session.strategy !=
                lfs::core::param::kStrategyIGSPlus ||
            snapshot.igs_current.strategy !=
                lfs::core::param::kStrategyIGSPlus;
        if (role_mismatch) {
            return lfs::Result<void>::failure(parameter_project_error(
                lfs::ErrorCode::InvalidArgument,
                "Pending project preset is stored under the wrong strategy role"));
        }
        if (const std::string error = snapshot.dataset.validate(); !error.empty()) {
            return lfs::Result<void>::failure(parameter_project_error(
                lfs::ErrorCode::InvalidArgument,
                "Pending project dataset parameters are invalid: " + error));
        }
        return {};
    }

    void ParameterManager::installValidatedPendingProjectState(
        const lfs::io::project::ParameterManagerSnapshot& snapshot) {
        std::lock_guard lock(params_mutex_);
        active_strategy_ = std::string(
            lfs::core::param::canonical_strategy_name(snapshot.active_strategy));
        mcmc_session_ = snapshot.mcmc_session;
        mrnf_session_ = snapshot.mrnf_session;
        igs_session_ = snapshot.igs_session;
        mcmc_current_ = snapshot.mcmc_current;
        mrnf_current_ = snapshot.mrnf_current;
        igs_current_ = snapshot.igs_current;
        mcmc_session_references_ =
            snapshot.mcmc_session_references;
        mrnf_session_references_ =
            snapshot.mrnf_session_references;
        igs_session_references_ =
            snapshot.igs_session_references;
        mcmc_current_references_ =
            snapshot.mcmc_current_references;
        mrnf_current_references_ =
            snapshot.mrnf_current_references;
        igs_current_references_ =
            snapshot.igs_current_references;
        dataset_config_ = snapshot.dataset;
        loaded_ = true;
        dirty_.store(false, std::memory_order_release);
    }

    lfs::Result<void> ParameterManager::restorePendingProjectState(
        const lfs::io::project::ParameterManagerSnapshot& snapshot) {
        if (auto valid =
                validatePendingProjectState(snapshot);
            !valid) {
            return valid;
        }
        installValidatedPendingProjectState(snapshot);
        return {};
    }

    lfs::core::param::OptimizationParameters& ParameterManager::getCurrentParams(const std::string_view strategy) {
        if (strategy == "mcmc")
            return mcmc_current_;
        if (lfs::core::param::is_mrnf_strategy(strategy))
            return mrnf_current_;
        if (strategy == "igs+")
            return igs_current_;
        return mrnf_current_;
    }

    const lfs::core::param::OptimizationParameters& ParameterManager::getCurrentParams(const std::string_view strategy) const {
        if (strategy == "mcmc")
            return mcmc_current_;
        if (lfs::core::param::is_mrnf_strategy(strategy))
            return mrnf_current_;
        if (strategy == "igs+")
            return igs_current_;
        return mrnf_current_;
    }

    lfs::core::param::OptimizationParameters ParameterManager::copySessionParams(const std::string_view strategy) {
        ensureLoaded().value();

        std::lock_guard lock(params_mutex_);
        const std::string_view resolved_strategy = strategy.empty() ? std::string_view(active_strategy_) : strategy;
        if (resolved_strategy == "mcmc")
            return mcmc_session_;
        if (lfs::core::param::is_mrnf_strategy(resolved_strategy))
            return mrnf_session_;
        if (resolved_strategy == "igs+")
            return igs_session_;
        return mrnf_session_;
    }

    void ParameterManager::resetToDefaults(const std::string_view strategy) {
        std::lock_guard lock(params_mutex_);
        if (strategy.empty() || strategy == "mcmc") {
            mcmc_current_ = mcmc_session_;
            mcmc_current_references_ =
                mcmc_session_references_;
        }
        if (strategy.empty() || lfs::core::param::is_mrnf_strategy(strategy)) {
            mrnf_current_ = mrnf_session_;
            mrnf_current_references_ =
                mrnf_session_references_;
        }
        if (strategy.empty() || strategy == "igs+") {
            igs_current_ = igs_session_;
            igs_current_references_ =
                igs_session_references_;
        }
    }

    void ParameterManager::clearSession() {
        if (const auto result = ensureLoaded(); !result) {
            LOG_ERROR("Failed to load params: {}", result.error());
            return;
        }

        std::lock_guard lock(params_mutex_);
        active_strategy_ = std::string(lfs::core::param::kStrategyMRNF);
        mcmc_session_ = lfs::core::param::OptimizationParameters::mcmc_defaults();
        mcmc_current_ = mcmc_session_;
        mrnf_session_ = lfs::core::param::OptimizationParameters::mrnf_defaults();
        mrnf_current_ = mrnf_session_;
        igs_session_ = lfs::core::param::OptimizationParameters::igs_plus_defaults();
        igs_current_ = igs_session_;
        mcmc_session_references_ = {};
        mrnf_session_references_ = {};
        igs_session_references_ = {};
        mcmc_current_references_ = {};
        mrnf_current_references_ = {};
        igs_current_references_ = {};
        dataset_config_ = lfs::core::param::DatasetConfig{};
        dataset_config_.centralize_dataset = "off";
        dataset_config_.loading_params = lfs::core::param::LoadingParams{};
        dirty_.store(false, std::memory_order_release);
    }

    void ParameterManager::setSessionDefaults(const lfs::core::param::TrainingParameters& params) {
        if (const auto result = ensureLoaded(); !result) {
            LOG_ERROR("Failed to load params: {}", result.error());
            return;
        }
        std::lock_guard lock(params_mutex_);
        const auto& opt = params.optimization;
        if (!opt.strategy.empty())
            setActiveStrategy(opt.strategy);

        auto* session = &mrnf_session_;
        auto* current = &mrnf_current_;
        auto* session_references =
            &mrnf_session_references_;
        auto* current_references =
            &mrnf_current_references_;
        if (active_strategy_ == "mcmc") {
            session = &mcmc_session_;
            current = &mcmc_current_;
            session_references =
                &mcmc_session_references_;
            current_references =
                &mcmc_current_references_;
        } else if (active_strategy_ == "igs+") {
            session = &igs_session_;
            current = &igs_current_;
            session_references =
                &igs_session_references_;
            current_references =
                &igs_current_references_;
        }
        *session = opt;
        *current = opt;
        *session_references = {};
        *current_references = {};

        // Apply CLI overrides to dataset config
        const auto& ds = params.dataset;
        if (ds.resize_factor > 0)
            dataset_config_.resize_factor = ds.resize_factor;
        if (ds.max_width >= 0)
            dataset_config_.max_width = ds.max_width;
        if (!ds.images.empty())
            dataset_config_.images = ds.images;
        if (ds.test_every > 0)
            dataset_config_.test_every = ds.test_every;
        dataset_config_.loading_params = ds.loading_params;
        dataset_config_.timelapse_images = ds.timelapse_images;
        dataset_config_.timelapse_every = ds.timelapse_every;
        dataset_config_.invert_masks = ds.invert_masks;
        dataset_config_.mask_threshold = ds.mask_threshold;

        LOG_INFO("Session: strategy={}, iter={}, resize={}", opt.strategy, opt.iterations, dataset_config_.resize_factor);
    }

    void ParameterManager::importParams(const lfs::core::param::OptimizationParameters& params) {
        std::lock_guard lock(params_mutex_);
        if (!params.strategy.empty()) {
            setActiveStrategy(params.strategy);
        }
        if (active_strategy_ == "mcmc") {
            mcmc_session_ = params;
            mcmc_current_ = params;
            mcmc_session_references_ = {};
            mcmc_current_references_ = {};
        } else if (lfs::core::param::is_mrnf_strategy(active_strategy_)) {
            mrnf_session_ = params;
            mrnf_current_ = params;
            mrnf_session_references_ = {};
            mrnf_current_references_ = {};
        } else if (active_strategy_ == "igs+") {
            igs_session_ = params;
            igs_current_ = params;
            igs_session_references_ = {};
            igs_current_references_ = {};
        }
        LOG_INFO("Imported params: strategy={}, iter={}, sh={}", params.strategy, params.iterations, params.sh_degree);
    }

    void ParameterManager::importTrainingParams(const lfs::core::param::TrainingParameters& params) {
        if (const auto result = ensureLoaded(); !result) {
            LOG_ERROR("Failed to load params: {}", result.error());
            return;
        }

        std::lock_guard lock(params_mutex_);
        if (!params.optimization.strategy.empty()) {
            setActiveStrategy(params.optimization.strategy);
        }

        if (active_strategy_ == "mcmc") {
            mcmc_session_ = params.optimization;
            mcmc_current_ = params.optimization;
            mcmc_session_references_ = {};
            mcmc_current_references_ = {};
        } else if (lfs::core::param::is_mrnf_strategy(active_strategy_)) {
            mrnf_session_ = params.optimization;
            mrnf_current_ = params.optimization;
            mrnf_session_references_ = {};
            mrnf_current_references_ = {};
        } else if (active_strategy_ == "igs+") {
            igs_session_ = params.optimization;
            igs_current_ = params.optimization;
            igs_session_references_ = {};
            igs_current_references_ = {};
        }

        dataset_config_ = params.dataset;
        dirty_.store(false, std::memory_order_release);

        LOG_INFO("Imported training params: strategy={}, iter={}, images={}, resize={}",
                 params.optimization.strategy,
                 params.optimization.iterations,
                 dataset_config_.images,
                 dataset_config_.resize_factor);
    }

    void ParameterManager::setActiveStrategy(const std::string_view strategy) {
        if (const auto canonical_strategy = lfs::core::param::canonical_strategy_name(strategy);
            !canonical_strategy.empty()) {
            active_strategy_ = std::string(canonical_strategy);
        }
    }

    lfs::core::param::OptimizationParameters& ParameterManager::getActiveParams() {
        return getCurrentParams(active_strategy_);
    }

    const lfs::core::param::OptimizationParameters& ParameterManager::getActiveParams() const {
        return getCurrentParams(active_strategy_);
    }

    void ParameterManager::autoScaleSteps(const size_t image_count) {
        assert(image_count > 0);
        const float new_scaler = (image_count <= BASE_IMAGE_COUNT)
                                     ? 1.0f
                                     : static_cast<float>(image_count) / static_cast<float>(BASE_IMAGE_COUNT);

        std::lock_guard lock(params_mutex_);
        apply_scaler_to_params(mcmc_current_, new_scaler);
        apply_scaler_to_params(mrnf_current_, new_scaler);
        apply_scaler_to_params(igs_current_, new_scaler);
        markDirty();
        LOG_INFO("Auto-scaled steps for {} images: scaler={:.2f}", image_count, new_scaler);
    }

    lfs::core::param::TrainingParameters ParameterManager::createForDataset(
        const std::filesystem::path& data_path,
        const std::filesystem::path& output_path) const {

        std::lock_guard lock(params_mutex_);
        lfs::core::param::TrainingParameters params;
        params.optimization = getActiveParams();
        params.dataset = dataset_config_;
        params.dataset.data_path = data_path;
        params.dataset.output_path = output_path;
        return params;
    }

} // namespace lfs::vis
