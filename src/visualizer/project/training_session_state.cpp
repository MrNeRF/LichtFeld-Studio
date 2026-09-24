#include "project/training_session_internal.hpp"
namespace lfs::vis {
    using namespace training_session_detail;
    using namespace lfs::core::events;

    void TrainerManager::clearEvaluationMetrics() {
        {
            std::lock_guard<std::mutex> lock(psnr_buffer_mutex_);
            psnr_buffer_.clear();
        }
        {
            std::lock_guard<std::mutex> lock(eval_metrics_mutex_);
            last_eval_metrics_.reset();
            evaluation_history_.clear();
        }
    }

    void TrainerManager::clearRestoredProjectMetrics() {
        restored_accumulated_training_time_.reset();
        restored_finish_reason_.reset();
        restored_finish_published_ = false;
    }

    void TrainerManager::clearStoredSessionPresentation() {
        stored_session_presentation_active_ = false;
        stored_session_presentation_completed_ = false;
        stored_session_presentation_iteration_ = 0;
        stored_session_presentation_max_iterations_ = 0;
        stored_session_presentation_strategy_.clear();
    }

    std::vector<std::shared_ptr<lfs::core::Camera>> TrainerManager::getAllCamList() const {
        if (scene_) {
            return scene_->getAllCameras();
        }
        return {};
    }

    std::shared_ptr<const lfs::core::Camera> TrainerManager::getCamById(int camId) const {
        if (scene_) {
            return scene_->getCameraByUid(camId);
        }
        LOG_ERROR("getCamById called but scene is not set");
        return nullptr;
    }

    float TrainerManager::getElapsedSeconds() const {
        const auto state = getState();
        if (state == TrainingState::Running) {
            const auto current = std::chrono::steady_clock::now() - training_start_time_;
            return std::chrono::duration<float>(accumulated_training_time_ + current).count();
        }
        return std::chrono::duration<float>(
                   accumulated_training_time_)
            .count();
    }

    float TrainerManager::getEstimatedRemainingSeconds() const {
        const float elapsed = getElapsedSeconds();
        const int current_iter = getCurrentIteration();
        const int total_iter = getTotalIterations();

        if (current_iter <= 0 || elapsed <= 0.0f || total_iter <= current_iter)
            return 0.0f;

        const float secs_per_iter = elapsed / static_cast<float>(current_iter);
        return secs_per_iter * static_cast<float>(total_iter - current_iter);
    }

    std::optional<TrainerManager::EvaluationMetricsSnapshot> TrainerManager::getLastEvaluationMetrics() const {
        std::lock_guard<std::mutex> lock(eval_metrics_mutex_);
        return last_eval_metrics_;
    }

    std::deque<float> TrainerManager::getLossBuffer() const {
        std::lock_guard<std::mutex> lock(loss_buffer_mutex_);
        return loss_buffer_;
    }

    std::deque<float> TrainerManager::getPSNRBuffer() const {
        std::lock_guard<std::mutex> lock(psnr_buffer_mutex_);
        return psnr_buffer_;
    }

    TrainingState TrainerManager::getState() const {
        const auto state = state_machine_.getState();
        if (trainer_ || state != TrainingState::Idle) {
            return state;
        }
        if (stored_session_presentation_active_) {
            return stored_session_presentation_completed_
                       ? TrainingState::Finished
                       : TrainingState::Paused;
        }
        return state;
    }

    bool TrainerManager::hasTrainer() const {
        return trainer_ != nullptr;
    }

    void TrainerManager::publishStoredSessionPresentation() {
        if (trainer_) {
            clearStoredSessionPresentation();
            return;
        }
        Visualizer::ProjectTrainingSessionState session;
        if (viewer_) {
            session = viewer_->projectTrainingSessionState();
        }

        LastStoredSessionPublish desired;
        desired.owner = this;
        desired.valid = true;
        desired.available = session.available;
        if (session.available) {
            const core::Scene* scene = scene_;
            if (!scene && viewer_) {
                scene = &viewer_->getScene();
            }
            desired.completed = session.completed;
            desired.stopped = isFinished() && state_machine_.getFinishReason() == FinishReason::UserStopped;
            desired.hydrated = session.hydrated;
            desired.iteration = session.iteration;
            desired.max_iterations = session.max_iterations;
            if (scene) {
                desired.num_gaussians = static_cast<int>(
                    scene->getTrainingModelGaussianCount());
            }
            desired.strategy =
                session.strategy.empty() ? "unknown" : session.strategy;
        }

        const bool stored_matches = desired.available
                                        ? (stored_session_presentation_active_ &&
                                           stored_session_presentation_completed_ == desired.completed &&
                                           stored_session_presentation_iteration_ == desired.iteration &&
                                           stored_session_presentation_max_iterations_ ==
                                               desired.max_iterations &&
                                           stored_session_presentation_strategy_ == desired.strategy)
                                        : !stored_session_presentation_active_;
        const auto& last = g_last_stored_session_publish;
        if (last.owner == this && last.valid && stored_matches &&
            last.available == desired.available &&
            last.completed == desired.completed &&
            last.stopped == desired.stopped &&
            last.hydrated == desired.hydrated &&
            last.iteration == desired.iteration &&
            last.max_iterations == desired.max_iterations &&
            last.num_gaussians == desired.num_gaussians &&
            last.strategy == desired.strategy) {
            return;
        }

        if (!session.available) {
            clearStoredSessionPresentation();
            auto& store = app_store();
            lfs::core::reactive::BatchUpdate batch(store.store());
            store.trainer_loaded.set(false);
            store.training_running.set(false);
            store.training_state.set("idle");
            store.iteration.set(0);
            store.total_iterations.set(0);
            {
                SignalGilBatch gil_batch;
                python::update_training_state(false, "idle");
                python::update_trainer_loaded(false, 0, 0);
                python::flush_signals();
            }
            g_last_stored_session_publish = desired;
            return;
        }

        stored_session_presentation_active_ = true;
        stored_session_presentation_completed_ = session.completed;
        stored_session_presentation_iteration_ = session.iteration;
        stored_session_presentation_max_iterations_ =
            session.max_iterations;
        stored_session_presentation_strategy_ = desired.strategy;
        const char* const presented_state =
            desired.stopped ? "stopped" : session.completed ? "completed"
                                                            : "paused";

        auto& store = app_store();
        {
            lfs::core::reactive::BatchUpdate batch(store.store());
            store.trainer_loaded.set(false);
            store.training_running.set(false);
            store.training_state.set(presented_state);
            store.iteration.set(session.iteration);
            store.total_iterations.set(session.max_iterations);
            store.num_gaussians.set(
                static_cast<std::int64_t>(desired.num_gaussians));
        }

        {
            SignalGilBatch gil_batch;
            python::update_trainer_loaded(
                false, session.max_iterations, session.iteration);
            python::update_training_state(false, presented_state);
            python::update_training_progress(
                session.iteration, 0.0f,
                static_cast<std::size_t>(std::max(0, desired.num_gaussians)));
            python::flush_signals();
        }

        lfs::training::CommandCenter::instance().update_snapshot(
            lfs::training::HookContext{
                .iteration = session.iteration,
                .num_gaussians = static_cast<std::size_t>(
                    std::max(0, desired.num_gaussians)),
                .trainer = nullptr},
            session.max_iterations,
            !session.completed && !desired.stopped,
            false,
            false,
            lfs::training::TrainingPhase::Idle);
        lfs::training::CommandCenter::instance().overlay_stored_session(
            stored_session_presentation_strategy_, session.hydrated);
        g_last_stored_session_publish = desired;
    }

    lfs::Error TrainerManager::rejectStart(
        std::string message, const lfs::ErrorCode code) {
        LOG_ERROR("Cannot start training: {}", message);
        lfs::Error typed = lfs::make_error(lfs::ErrorInit{
            .code = code,
            .domain = lfs::ErrorDomain::Training,
            .operation_id = lfs::OperationId::generate(),
            .user_message = message,
            .detail = "Training command rejected: " + message,
            .detection = LFS_SOURCE_SITE_CURRENT(),
        });
        // A rejected command does not own the accepted run's initialization
        // result, candidate or waiters. Return its error to that caller only.
        if (!isRunning() && getState() != TrainingState::Starting) {
            last_error_ = message;
            last_training_error_.set(typed);
        }
        state::TrainingStartRejected{
            .error = message,
            .error_info = core::to_wire_error(typed)}
            .emit();
        return typed;
    }

    FinishReason TrainerManager::resolvedRestoredFinishReason() const {
        // UserStopped and Error are saved pauses: resume unless the run
        // already hit total. An error terminal save is a valid safe-point
        // snapshot; the persisted Error value is provenance, not a restore
        // directive. Only Completed still restores as Finished.
        if (restored_finish_reason_ &&
            *restored_finish_reason_ !=
                lfs::io::project::TrainingFinishReason::None &&
            *restored_finish_reason_ !=
                lfs::io::project::TrainingFinishReason::UserStopped &&
            *restored_finish_reason_ !=
                lfs::io::project::TrainingFinishReason::Error) {
            return fromIoFinishReason(*restored_finish_reason_);
        }
        int iteration = getCurrentIteration();
        if (checkpoint_baseline_iteration_ &&
            *checkpoint_baseline_iteration_ > iteration) {
            iteration = *checkpoint_baseline_iteration_;
        }
        const int total = getTotalIterations();
        if (total > 0 && iteration >= total) {
            return FinishReason::Completed;
        }
        return FinishReason::None;
    }

    void TrainerManager::restoreProjectMetrics(
        const lfs::io::project::MetricsChapter&
            metrics) {
        std::vector<
            lfs::training::LossHistoryPoint>
            loss;
        loss.reserve(metrics.loss_history.size());
        for (const auto& sample :
             metrics.loss_history) {
            loss.push_back({
                .iteration = sample.iteration,
                .loss = sample.value,
            });
        }
        lfs::training::CommandCenter::instance()
            .replace_loss_history(std::move(loss));

        {
            std::lock_guard<std::mutex> lock(
                loss_buffer_mutex_);
            loss_buffer_.clear();
            const std::size_t begin =
                metrics.loss_history.size() >
                        static_cast<std::size_t>(
                            MAX_LOSS_POINTS)
                    ? metrics.loss_history.size() -
                          MAX_LOSS_POINTS
                    : 0;
            for (std::size_t index = begin;
                 index < metrics.loss_history.size();
                 ++index) {
                loss_buffer_.push_back(
                    metrics.loss_history[index]
                        .value);
            }
        }
        {
            std::scoped_lock lock(
                psnr_buffer_mutex_,
                eval_metrics_mutex_);
            psnr_buffer_.clear();
            evaluation_history_.clear();
            evaluation_history_.reserve(
                metrics.psnr_history.size());
            const std::size_t begin =
                metrics.psnr_history.size() >
                        static_cast<std::size_t>(
                            MAX_PSNR_POINTS)
                    ? metrics.psnr_history.size() -
                          MAX_PSNR_POINTS
                    : 0;
            for (std::size_t index = 0;
                 index < metrics.psnr_history.size();
                 ++index) {
                const auto& sample =
                    metrics.psnr_history[index];
                evaluation_history_.push_back({
                    .iteration = sample.iteration,
                    .psnr = sample.value,
                    .ssim = 0.0f,
                    .lpips = std::nullopt,
                });
                if (index >= begin)
                    psnr_buffer_.push_back(
                        sample.value);
            }
            if (metrics.last_evaluation) {
                last_eval_metrics_ = {
                    .iteration =
                        metrics.last_evaluation
                            ->iteration,
                    .psnr =
                        metrics.last_evaluation
                            ->psnr,
                    .ssim =
                        metrics.last_evaluation
                            ->ssim,
                    .lpips = std::nullopt,
                };
                if (!evaluation_history_.empty() &&
                    evaluation_history_.back()
                            .iteration ==
                        last_eval_metrics_
                            ->iteration) {
                    evaluation_history_.back()
                        .ssim =
                        last_eval_metrics_->ssim;
                }
            } else {
                last_eval_metrics_.reset();
            }
        }
        accumulated_training_time_ =
            std::chrono::duration_cast<
                std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(
                    metrics
                        .accumulated_training_seconds));
        restored_accumulated_training_time_ =
            accumulated_training_time_;
        restored_finish_reason_ = metrics.finish_reason;
        restored_finish_published_ = false;
        if (trainer_) {
            applyRestoredCheckpointPresentation();
        }
    }

    void TrainerManager::updateEvaluationMetrics(int iteration, float psnr, float ssim,
                                                 std::optional<float> lpips) {
        updatePSNR(psnr);
        std::lock_guard<std::mutex> lock(eval_metrics_mutex_);
        last_eval_metrics_ = EvaluationMetricsSnapshot{
            .iteration = iteration,
            .psnr = psnr,
            .ssim = ssim,
            .lpips = lpips};
        const auto position = std::lower_bound(
            evaluation_history_.begin(),
            evaluation_history_.end(), iteration,
            [](const EvaluationMetricsSnapshot& sample,
               const int target_iteration) {
                return sample.iteration <
                       target_iteration;
            });
        if (position != evaluation_history_.end() &&
            position->iteration == iteration) {
            *position = *last_eval_metrics_;
        } else {
            evaluation_history_.insert(
                position, *last_eval_metrics_);
        }
    }

    void TrainerManager::updateLoss(float loss) {
        std::lock_guard<std::mutex> lock(loss_buffer_mutex_);
        loss_buffer_.push_back(loss);
        while (loss_buffer_.size() > static_cast<size_t>(MAX_LOSS_POINTS)) {
            loss_buffer_.pop_front();
        }
        LOG_TRACE("Loss updated: {:.6f} (buffer size: {})", loss, loss_buffer_.size());
    }

    void TrainerManager::updatePSNR(float psnr) {
        std::lock_guard<std::mutex> lock(psnr_buffer_mutex_);
        psnr_buffer_.push_back(psnr);
        while (psnr_buffer_.size() > static_cast<size_t>(MAX_PSNR_POINTS)) {
            psnr_buffer_.pop_front();
        }
    }
} // namespace lfs::vis
