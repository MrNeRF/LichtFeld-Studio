/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "parameter_manager.hpp"
#include "core/logger.hpp"

#include <cassert>
#include <cmath>

namespace lfs::vis {

    namespace {
        constexpr size_t BASE_IMAGE_COUNT = 300;

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
        adc_session_ = lfs::core::param::OptimizationParameters::adc_defaults();
        adc_current_ = adc_session_;
        igs_session_ = lfs::core::param::OptimizationParameters::igs_plus_defaults();
        igs_current_ = igs_session_;
        dataset_config_.loading_params = lfs::core::param::LoadingParams{};

        loaded_ = true;
        return {};
    }

    lfs::core::param::OptimizationParameters& ParameterManager::getCurrentParams(const std::string_view strategy) {
        if (strategy == "mcmc")
            return mcmc_current_;
        if (strategy == "igs+")
            return igs_current_;
        return adc_current_;
    }

    const lfs::core::param::OptimizationParameters& ParameterManager::getCurrentParams(const std::string_view strategy) const {
        if (strategy == "mcmc")
            return mcmc_current_;
        if (strategy == "igs+")
            return igs_current_;
        return adc_current_;
    }

    void ParameterManager::resetToDefaults(const std::string_view strategy) {
        std::lock_guard lock(params_mutex_);
        if (strategy.empty() || strategy == "mcmc") {
            mcmc_current_ = mcmc_session_;
        }
        if (strategy.empty() || strategy == "adc") {
            adc_current_ = adc_session_;
        }
        if (strategy.empty() || strategy == "igs+") {
            igs_current_ = igs_session_;
        }
    }

    void ParameterManager::setSessionDefaults(const lfs::core::param::TrainingParameters& params) {
        if (const auto result = ensureLoaded(); !result) {
            LOG_ERROR("Failed to load params: {}", result.error());
            return;
        }
        if (session_defaults_set_)
            return;

        const auto& opt = params.optimization;
        if (!opt.strategy.empty())
            setActiveStrategy(opt.strategy);

        auto& session = (active_strategy_ == "mcmc")   ? mcmc_session_
                        : (active_strategy_ == "igs+") ? igs_session_
                                                       : adc_session_;
        auto& current = (active_strategy_ == "mcmc")   ? mcmc_current_
                        : (active_strategy_ == "igs+") ? igs_current_
                                                       : adc_current_;
        session = opt;
        current = opt;

        // Apply CLI overrides to dataset config
        const auto& ds = params.dataset;
        if (ds.resize_factor > 0)
            dataset_config_.resize_factor = ds.resize_factor;
        if (ds.max_width > 0)
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

        session_defaults_set_ = true;
        LOG_INFO("Session: strategy={}, iter={}, resize={}", opt.strategy, opt.iterations, dataset_config_.resize_factor);
    }

    void ParameterManager::setCurrentParams(const lfs::core::param::OptimizationParameters& params) {
        std::lock_guard lock(params_mutex_);
        if (!params.strategy.empty()) {
            setActiveStrategy(params.strategy);
        }
        if (active_strategy_ == "mcmc") {
            mcmc_current_ = params;
        } else if (active_strategy_ == "igs+") {
            igs_current_ = params;
        } else {
            adc_current_ = params;
        }
        LOG_DEBUG("Current params updated: strategy={}, iter={}, sh={}", params.strategy, params.iterations, params.sh_degree);
    }

    void ParameterManager::importParams(const lfs::core::param::OptimizationParameters& params) {
        std::lock_guard lock(params_mutex_);
        if (!params.strategy.empty()) {
            setActiveStrategy(params.strategy);
        }
        if (active_strategy_ == "mcmc") {
            mcmc_session_ = params;
            mcmc_current_ = params;
        } else if (active_strategy_ == "igs+") {
            igs_session_ = params;
            igs_current_ = params;
        } else {
            adc_session_ = params;
            adc_current_ = params;
        }
        LOG_INFO("Imported params: strategy={}, iter={}, sh={}", params.strategy, params.iterations, params.sh_degree);
    }

    void ParameterManager::setActiveStrategy(const std::string_view strategy) {
        if (strategy == "mcmc" || strategy == "adc" || strategy == "igs+") {
            active_strategy_ = std::string(strategy);
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
        apply_scaler_to_params(adc_current_, new_scaler);
        apply_scaler_to_params(igs_current_, new_scaler);
        dirty_.store(true, std::memory_order_release);
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
