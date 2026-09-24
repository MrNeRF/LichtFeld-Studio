#pragma once
/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/error_envelope.hpp"
#include "core/event_bridge/command_api.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "core/reactive/store.hpp"
#include "core/scene.hpp"
#include "core/training_manager.hpp"
#include "python/gil.hpp"
#include "python/python_runtime.hpp"
#include "visualizer/app_store.hpp"
#include "visualizer/visualizer_impl.hpp"
#include <algorithm>

namespace lfs::vis::training_session_detail {
    [[nodiscard]] inline std::vector<size_t> normalize_save_steps(std::vector<size_t> steps) {
        steps.erase(std::remove(steps.begin(), steps.end(), 0), steps.end());
        std::sort(steps.begin(), steps.end());
        steps.erase(std::unique(steps.begin(), steps.end()), steps.end());
        return steps;
    }

    inline void apply_save_steps(lfs::core::param::OptimizationParameters& params,
                                 const std::vector<size_t>& steps) {
        params.save_steps = steps;
        if (params.enable_eval)
            params.eval_steps = steps;
    }

    [[nodiscard]] inline lfs::Error training_initialization_error(std::string message) {
        return lfs::make_legacy_error(std::move(message), lfs::LegacyErrorContext{
                                                              .code = lfs::ErrorCode::FailedPrecondition,
                                                              .domain = lfs::ErrorDomain::Training,
                                                              .operation = "training.start",
                                                              .source = LFS_SOURCE_SITE_CURRENT(),
                                                              .operation_id = lfs::OperationId::generate(),
                                                          });
    }

    inline void refreshCameraEvaluationSplit(
        lfs::core::Scene& scene,
        const bool enable_eval,
        const int test_every) {
        auto cameras = scene.getActiveCameras();
        std::erase_if(cameras, [](const auto& camera) {
            return !camera || !camera->has_image();
        });
        std::sort(
            cameras.begin(), cameras.end(),
            [](const auto& lhs, const auto& rhs) {
                return lhs->uid() < rhs->uid();
            });

        const auto split_interval = static_cast<size_t>(std::max(1, test_every));
        size_t eval_count = 0;
        for (size_t i = 0; i < cameras.size(); ++i) {
            const bool is_eval = enable_eval && (i % split_interval) == 0;
            cameras[i]->set_split(
                is_eval ? lfs::core::CameraSplit::Eval
                        : lfs::core::CameraSplit::Train);
            eval_count += is_eval;
        }

        LOG_INFO(
            "Refreshed camera evaluation split: {} train, {} val images",
            cameras.size() - eval_count,
            eval_count);
    }

    [[nodiscard]] inline lfs::io::project::TrainingFinishReason
    toIoFinishReason(const FinishReason reason) {
        switch (reason) {
        case FinishReason::Completed:
            return lfs::io::project::TrainingFinishReason::Completed;
        case FinishReason::UserStopped:
            return lfs::io::project::TrainingFinishReason::UserStopped;
        case FinishReason::Error:
            return lfs::io::project::TrainingFinishReason::Error;
        case FinishReason::None:
            break;
        }
        return lfs::io::project::TrainingFinishReason::None;
    }

    struct SignalGilBatch {
        SignalGilBatch() {
            if (lfs::python::can_acquire_gil()) {
                gil_ = std::make_unique<lfs::python::GilAcquire>();
            }
        }
        std::unique_ptr<lfs::python::GilAcquire> gil_;
    };

    struct LastStoredSessionPublish {
        const TrainerManager* owner = nullptr;
        bool valid = false;
        bool available = false;
        bool completed = false;
        bool stopped = false;
        bool hydrated = false;
        int iteration = 0;
        int max_iterations = 0;
        int num_gaussians = 0;
        std::string strategy;
    };

    inline LastStoredSessionPublish g_last_stored_session_publish;

    [[nodiscard]] inline FinishReason
    fromIoFinishReason(
        const lfs::io::project::TrainingFinishReason reason) {
        switch (reason) {
        case lfs::io::project::TrainingFinishReason::Completed:
            return FinishReason::Completed;
        case lfs::io::project::TrainingFinishReason::UserStopped:
            return FinishReason::UserStopped;
        case lfs::io::project::TrainingFinishReason::Error:
            return FinishReason::Error;
        case lfs::io::project::TrainingFinishReason::None:
            break;
        }
        return FinishReason::None;
    }

} // namespace lfs::vis::training_session_detail
