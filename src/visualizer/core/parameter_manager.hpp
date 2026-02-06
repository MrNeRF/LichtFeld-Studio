/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/parameters.hpp"
#include <atomic>
#include <expected>
#include <mutex>
#include <string>
#include <string_view>

namespace lfs::vis {

    // Session defaults set once at startup (CLI > --config > JSON), current params are user-editable.
    class LFS_VIS_API ParameterManager {
    public:
        std::expected<void, std::string> ensureLoaded();

        [[nodiscard]] lfs::core::param::OptimizationParameters& getCurrentParams(std::string_view strategy);
        [[nodiscard]] const lfs::core::param::OptimizationParameters& getCurrentParams(std::string_view strategy) const;

        [[nodiscard]] lfs::core::param::DatasetConfig& getDatasetConfig() { return dataset_config_; }
        [[nodiscard]] const lfs::core::param::DatasetConfig& getDatasetConfig() const { return dataset_config_; }

        // Reset current to session defaults
        void resetToDefaults(std::string_view strategy = "");

        // Set session defaults from CLI params (called once at startup)
        void setSessionDefaults(const lfs::core::param::TrainingParameters& params);

        // Set current params (e.g., from loaded checkpoint)
        void setCurrentParams(const lfs::core::param::OptimizationParameters& params);

        // Import params: overwrites both session and current for active strategy
        void importParams(const lfs::core::param::OptimizationParameters& params);

        [[nodiscard]] const std::string& getActiveStrategy() const { return active_strategy_; }
        void setActiveStrategy(std::string_view strategy);

        [[nodiscard]] lfs::core::param::OptimizationParameters& getActiveParams();
        [[nodiscard]] const lfs::core::param::OptimizationParameters& getActiveParams() const;

        [[nodiscard]] lfs::core::param::TrainingParameters createForDataset(
            const std::filesystem::path& data_path,
            const std::filesystem::path& output_path) const;

        [[nodiscard]] bool isLoaded() const { return loaded_; }

        void markDirty() { dirty_.store(true, std::memory_order_release); }
        bool consumeDirty() { return dirty_.exchange(false, std::memory_order_acq_rel); }

        [[nodiscard]] lfs::core::param::OptimizationParameters copyActiveParams() const {
            std::lock_guard lock(params_mutex_);
            return getActiveParams();
        }

        template <typename F>
        void modifyActiveParams(F&& fn) {
            std::lock_guard lock(params_mutex_);
            fn(getActiveParams());
            dirty_.store(true, std::memory_order_release);
        }

    private:
        bool loaded_ = false;
        bool session_defaults_set_ = false;
        std::string active_strategy_ = "mcmc";

        // Session defaults (immutable after startup)
        lfs::core::param::OptimizationParameters mcmc_session_;
        lfs::core::param::OptimizationParameters adc_session_;

        // Current params (user-editable)
        lfs::core::param::OptimizationParameters mcmc_current_;
        lfs::core::param::OptimizationParameters adc_current_;

        // Dataset config (CLI overrides JSON defaults)
        lfs::core::param::DatasetConfig dataset_config_;

        mutable std::mutex params_mutex_;
        std::atomic<bool> dirty_{false};
    };

} // namespace lfs::vis
