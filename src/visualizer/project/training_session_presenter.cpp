/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/event_bridge/command_api.hpp"
#include "core/logger.hpp"
#include "core/parameter_manager.hpp"
#include "core/scene.hpp"
#include "core/services.hpp"
#include "core/training_manager.hpp"
#include "project/training_session_internal.hpp"
#include "visualizer/visualizer_impl.hpp"

#include <utility>

namespace lfs::vis {
    using namespace training_session_detail;

    using namespace lfs::core::events;

    bool TrainerManager::canPerform(const TrainingAction action) const {
        return action == TrainingAction::LoadDataset || action == TrainingAction::LoadCheckpoint ||
               action == TrainingAction::Reset || action == TrainingAction::ClearScene ||
               action == TrainingAction::DeleteTrainingNode;
    }

    int TrainerManager::getCurrentIteration() const {
        if (stored_session_presentation_active_) {
            return stored_session_presentation_iteration_;
        }
        return 0;
    }

    float TrainerManager::getCurrentLoss() const {
        return 0.0f;
    }

    int TrainerManager::getTotalIterations() const {
        if (stored_session_presentation_active_) {
            return stored_session_presentation_max_iterations_;
        }
        return 0;
    }

    int TrainerManager::getNumSplats() const {
        if (!trainer_) {
            if (stored_session_presentation_active_) {
                const core::Scene* scene = scene_;
                if (!scene && viewer_) {
                    scene = &viewer_->getScene();
                }
                if (scene) {
                    return static_cast<int>(
                        scene->getTrainingModelGaussianCount());
                }
            }
            return 0;
        }
        return 0;
    }

    int TrainerManager::getMaxGaussians() const {
        return 0;
    }

    std::vector<size_t> TrainerManager::getSaveSteps() const {
        if (auto* const param_mgr = services().paramsOrNull(); param_mgr && param_mgr->isLoaded())
            return param_mgr->copyActiveParams().save_steps;
        return pending_opt_params_.save_steps;
    }

    void TrainerManager::setSaveSteps(std::vector<size_t> save_steps) {
        save_steps = normalize_save_steps(std::move(save_steps));
        apply_save_steps(pending_opt_params_, save_steps);

        if (auto* const param_mgr = services().paramsOrNull()) {
            if (const auto loaded = param_mgr->ensureLoaded(); loaded) {
                param_mgr->modifyActiveParams([&save_steps](auto& params) {
                    apply_save_steps(params, save_steps);
                });
            } else {
                LOG_WARN("Could not update save steps: {}", loaded.error());
            }
        }
    }

    const char* TrainerManager::getStrategyType() const {
        if (stored_session_presentation_active_ &&
            !stored_session_presentation_strategy_.empty()) {
            return stored_session_presentation_strategy_.c_str();
        }
        return "unknown";
    }

    bool TrainerManager::isGutEnabled() const {
        return false;
    }

    lfs::io::project::MetricsChapter
    TrainerManager::captureProjectMetrics() const {
        using lfs::io::project::LastEvaluationMetrics;
        using lfs::io::project::MetricHistorySample;

        lfs::io::project::MetricsChapter result;
        const auto loss =
            lfs::training::CommandCenter::instance()
                .loss_history();
        result.loss_history.reserve(loss.size());
        for (const auto& sample : loss) {
            result.loss_history.push_back(
                MetricHistorySample{
                    .iteration = sample.iteration,
                    .value = sample.loss,
                });
        }
        {
            std::lock_guard<std::mutex> lock(
                eval_metrics_mutex_);
            result.psnr_history.reserve(
                evaluation_history_.size());
            for (const auto& sample :
                 evaluation_history_) {
                result.psnr_history.push_back(
                    MetricHistorySample{
                        .iteration =
                            sample.iteration,
                        .value = sample.psnr,
                    });
            }
            if (last_eval_metrics_) {
                result.last_evaluation =
                    LastEvaluationMetrics{
                        .iteration =
                            last_eval_metrics_
                                ->iteration,
                        .psnr =
                            last_eval_metrics_->psnr,
                        .ssim =
                            last_eval_metrics_->ssim,
                    };
            }
        }
        result.accumulated_training_seconds =
            getElapsedSeconds();
        result.finish_reason =
            toIoFinishReason(state_machine_.getFinishReason());
        // Viewing and saving a project must retain the saved training result.
        result.finish_reason = restored_finish_reason_.value_or(result.finish_reason);
        return result;
    }

    std::expected<lfs::training::CameraMetricsSnapshot, std::string>
    TrainerManager::computeCameraMetricsForCameraId(
        const int camera_id,
        const bool include_ssim,
        const lfs::training::CameraMetricsAppearanceConfig& appearance) const {
        return std::unexpected("Training is not included in this build");
    }

    TrainerManager::TrainerManager() = default;
    TrainerManager::~TrainerManager() {
        if (g_last_stored_session_publish.owner == this)
            g_last_stored_session_publish = {};
    }
    bool TrainerManager::clearTrainer() {
        clearStoredSessionPresentation();
        clearRestoredProjectMetrics();
        clearEvaluationMetrics();
        lfs::training::CommandCenter::instance().reset_snapshot();
        return true;
    }
    bool TrainerManager::startTraining() {
        (void)rejectStart("Training is not included in this build", lfs::ErrorCode::Unavailable);
        return false;
    }
    lfs::Status TrainerManager::preflightStartParameters() {
        return lfs::Status::failure(rejectStart("Training is not included in this build", lfs::ErrorCode::Unavailable));
    }
    lfs::Status TrainerManager::resumeTraining() { return preflightStartParameters(); }
    lfs::Status TrainerManager::applyPendingParams() { return preflightStartParameters(); }
    lfs::Result<void> TrainerManager::waitForInitialization() {
        return lfs::Result<void>::failure(rejectStart("Training is not included in this build", lfs::ErrorCode::Unavailable));
    }
    void TrainerManager::pauseTraining() {}
    void TrainerManager::stopTraining() {}
    void TrainerManager::pauseTrainingTemporary() {}
    void TrainerManager::resumeTrainingTemporary() {}
    bool TrainerManager::requestSaveProject() { return false; }
    bool TrainerManager::waitForCompletion() { return true; }
    bool TrainerManager::hasLiveTrainingThread() const { return false; }
    bool TrainerManager::isPausedAtCheckpointBaseline() const { return false; }
    void TrainerManager::applyRestoredCheckpointPresentation() {}

} // namespace lfs::vis
