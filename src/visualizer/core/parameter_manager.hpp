/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/error.hpp"
#include "core/export.hpp"
#include "core/parameters.hpp"
#include "io/project_chapters.hpp"
#include <atomic>
#include <expected>
#include <mutex>
#include <string>
#include <string_view>

namespace lfs::vis {

    // Session defaults come from the most recent explicit source (CLI, JSON, checkpoint/import); current params are user-editable.
    class LFS_VIS_API ParameterManager {
    public:
        std::expected<void, std::string> ensureLoaded();

        [[nodiscard]] lfs::core::param::OptimizationParameters& getCurrentParams(std::string_view strategy);
        [[nodiscard]] const lfs::core::param::OptimizationParameters& getCurrentParams(std::string_view strategy) const;
        [[nodiscard]] lfs::core::param::OptimizationParameters copySessionParams(std::string_view strategy = {});

        [[nodiscard]] lfs::core::param::DatasetConfig& getDatasetConfig() { return dataset_config_; }
        [[nodiscard]] const lfs::core::param::DatasetConfig& getDatasetConfig() const { return dataset_config_; }

        // Reset current to session defaults
        void resetToDefaults(std::string_view strategy = "");

        // Restore built-in defaults and clear cached dataset configuration.
        void clearSession();

        // Set or replace session defaults from explicit params.
        void setSessionDefaults(const lfs::core::param::TrainingParameters& params);

        // Set current params (e.g., from loaded checkpoint)

        // Import params: overwrites both session and current for active strategy
        void importParams(const lfs::core::param::OptimizationParameters& params);

        // Import a fully resolved training configuration (e.g., checkpoint restore).
        void importTrainingParams(const lfs::core::param::TrainingParameters& params);

        [[nodiscard]] const std::string& getActiveStrategy() const { return active_strategy_; }
        void setActiveStrategy(std::string_view strategy);

        [[nodiscard]] lfs::core::param::OptimizationParameters& getActiveParams();
        [[nodiscard]] const lfs::core::param::OptimizationParameters& getActiveParams() const;

        void autoScaleSteps(size_t image_count);

        [[nodiscard]] lfs::core::param::TrainingParameters createForDataset(
            const std::filesystem::path& data_path,
            const std::filesystem::path& output_path) const;

        [[nodiscard]] bool isLoaded() const { return loaded_; }

        [[nodiscard]] lfs::Result<lfs::io::project::ParameterManagerSnapshot>
        capturePendingProjectState() const;
        [[nodiscard]] static lfs::Result<void>
        validatePendingProjectState(
            const lfs::io::project::ParameterManagerSnapshot& snapshot);
        // Restores only ParameterManager's role-qualified next-run/session
        // values. It deliberately has no TrainerManager dependency and cannot
        // alter an active trainer.
        lfs::Result<void> restorePendingProjectState(
            const lfs::io::project::ParameterManagerSnapshot& snapshot);
        // Phase-B install for a snapshot already accepted by
        // validatePendingProjectState during transactional project-open Phase A.
        void installValidatedPendingProjectState(
            const lfs::io::project::ParameterManagerSnapshot& snapshot);

        void markDirty() {
            dirty_serial_.fetch_add(1, std::memory_order_acq_rel);
            dirty_.store(true, std::memory_order_release);
        }
        [[nodiscard]] bool isDirty() const { return dirty_.load(std::memory_order_acquire); }
        bool consumeDirty() { return dirty_.exchange(false, std::memory_order_acq_rel); }
        [[nodiscard]] std::uint64_t dirtySerial() const {
            return dirty_serial_.load(std::memory_order_acquire);
        }
        void clearDirtyIfUnchanged(const std::uint64_t serial) {
            if (dirty_serial_.load(std::memory_order_acquire) == serial)
                dirty_.store(false, std::memory_order_release);
        }

        [[nodiscard]] lfs::core::param::OptimizationParameters copyActiveParams() const {
            std::lock_guard lock(params_mutex_);
            return getActiveParams();
        }

        template <typename F>
        void modifyActiveParams(F&& fn) {
            std::lock_guard lock(params_mutex_);
            fn(getActiveParams());
            markDirty();
        }

    private:
        bool loaded_ = false;
        std::string active_strategy_ = std::string(lfs::core::param::kStrategyMRNF);

        // Session defaults
        lfs::core::param::OptimizationParameters mcmc_session_;
        lfs::core::param::OptimizationParameters mrnf_session_;
        lfs::core::param::OptimizationParameters igs_session_;

        // Current params (user-editable)
        lfs::core::param::OptimizationParameters mcmc_current_;
        lfs::core::param::OptimizationParameters mrnf_current_;
        lfs::core::param::OptimizationParameters igs_current_;

        // Logical REFS bindings are part of each role, not derivable from
        // the path-bearing runtime parameter structs. Keep them alongside
        // the live slots so a training safe-point can preserve inactive
        // sessions without consulting serialized PRMS.
        lfs::io::project::ParameterManagerSnapshot::
            ReferenceBindings mcmc_session_references_;
        lfs::io::project::ParameterManagerSnapshot::
            ReferenceBindings mrnf_session_references_;
        lfs::io::project::ParameterManagerSnapshot::
            ReferenceBindings igs_session_references_;
        lfs::io::project::ParameterManagerSnapshot::
            ReferenceBindings mcmc_current_references_;
        lfs::io::project::ParameterManagerSnapshot::
            ReferenceBindings mrnf_current_references_;
        lfs::io::project::ParameterManagerSnapshot::
            ReferenceBindings igs_current_references_;

        // Dataset config (CLI overrides JSON defaults)
        lfs::core::param::DatasetConfig dataset_config_;

        mutable std::mutex params_mutex_;
        std::atomic<bool> dirty_{false};
        std::atomic<std::uint64_t> dirty_serial_{0};
    };

} // namespace lfs::vis
