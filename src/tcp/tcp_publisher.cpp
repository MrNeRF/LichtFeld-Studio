/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "tcp_publisher.hpp"
#include "core/include/core/events.hpp"
#include "core/include/core/json_utils.hpp"
#include "core/include/core/logger.hpp"

#include <algorithm>
#include <string_view>

// Define a transform to JSON for each event structure
#define ENABLE_TO_JSON(event, ...) NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_ONLY_SERIALIZE(event, __VA_ARGS__)
namespace lfs::core::events::state {
    ENABLE_TO_JSON(TrainingStarted, total_iterations);
    ENABLE_TO_JSON(TrainingProgress, iteration, loss, num_gaussians, is_refining);
    ENABLE_TO_JSON(TrainingPaused, iteration);
    ENABLE_TO_JSON(TrainingResumed, iteration);
    ENABLE_TO_JSON(TrainingCompleted, iteration, final_loss, elapsed_seconds, success, user_stopped, error, error_info);
    ENABLE_TO_JSON(TrainingStopped, iteration, user_requested);

    ENABLE_TO_JSON(ModelUpdated, iteration, num_gaussians);
    ENABLE_TO_JSON(PLYAdded, name, node_gaussians, total_gaussians, is_visible, parent_name, is_group, node_type);
    ENABLE_TO_JSON(PLYRemoved, name, children_kept, parent_of_removed);
    ENABLE_TO_JSON(NodeReparented, name, old_parent, new_parent);

    // Data loading
    ENABLE_TO_JSON(DatasetLoadStarted, path);
    ENABLE_TO_JSON(DatasetLoadProgress, path, progress, step);
    ENABLE_TO_JSON(DatasetLoadCompleted, path, success, error, num_images, num_points);
    ENABLE_TO_JSON(ConfigLoadFailed, path, error);
    ENABLE_TO_JSON(FileDropFailed, files, error);

    // Evaluation
    ENABLE_TO_JSON(EvaluationStarted, iteration, num_images);
    ENABLE_TO_JSON(EvaluationProgress, iteration, current, total);
    ENABLE_TO_JSON(EvaluationCompleted, iteration, psnr, ssim, lpips, elapsed_time, num_gaussians);

    // System state
    ENABLE_TO_JSON(CheckpointSaved, iteration, path);
    ENABLE_TO_JSON(DiskSpaceSaveFailed, iteration, path, error, required_bytes, available_bytes, is_disk_space_error, is_checkpoint);
    ENABLE_TO_JSON(MemoryUsage, gpu_used, gpu_total, gpu_percent, ram_used, ram_total, ram_percent);
    ENABLE_TO_JSON(FrameRendered, render_ms, fps, num_gaussians);
    ENABLE_TO_JSON(KeyframeListChanged, count);
    ENABLE_TO_JSON(ExportCompleted, path, format);
    ENABLE_TO_JSON(ExportFailed, error, error_info);
    ENABLE_TO_JSON(VideoExportCompleted, path, total_frames);
    ENABLE_TO_JSON(VideoExportFailed, error);

    // CUDA version check
    ENABLE_TO_JSON(CudaVersionUnsupported, major, minor, min_major, min_minor);
    ENABLE_TO_JSON(CudaUnavailable, message);
} // namespace lfs::core::events::state
#undef ENABLE_TO_JSON

namespace {
    constexpr size_t MAX_PUBLISH_QUEUE_EVENTS = 1024;
    constexpr size_t MAX_PUBLISH_EVENT_BYTES = 256 * 1024;
    constexpr size_t MAX_PUBLISH_QUEUE_BYTES = 4 * 1024 * 1024;

    [[nodiscard]] bool is_lossy_event(const std::string_view event_type) {
        return event_type == "log" ||
               event_type == "TrainingProgress" ||
               event_type == "DatasetLoadProgress" ||
               event_type == "EvaluationProgress" ||
               event_type == "MemoryUsage" ||
               event_type == "FrameRendered";
    }

    [[nodiscard]] bool is_coalescible_event(const std::string_view event_type) {
        return event_type != "log" && is_lossy_event(event_type);
    }
} // namespace

// EventBridge callbacks can still be in flight after unsubscribe. Capture only a weak
// queue generation so a late callback never dereferences the PublisherServer itself.
#define SUBSCRIBE_EVENT(Type)                                                        \
    do {                                                                             \
        const std::weak_ptr<QueueState> weak_state = queue_state_;                   \
        subscriptions_.emplace_back(                                                 \
            [id = lfs::core::events::state::Type::when([weak_state](const auto& e) { \
                 try {                                                               \
                     if (auto state = weak_state.lock())                             \
                         enqueueEvent(state, nlohmann::json(e), #Type);              \
                 } catch (...) {                                                     \
                     if (auto state = weak_state.lock())                             \
                         state->dropped.fetch_add(1, std::memory_order_relaxed);     \
                 }                                                                   \
             })]() {                                                                 \
                ::lfs::event::EventBridge::instance().unsubscribe(                   \
                    typeid(lfs::core::events::state::Type), id);                     \
            });                                                                      \
    } while (false)

namespace lfs::tcp {

    PublisherServer::PublisherServer(int port, std::shared_ptr<lfs::vis::TrainerManager> trainer_manager, core::LogLevel level, bool warm_up)
        : TCPServer(port, std::move(trainer_manager), zmq::socket_type::pub),
          level_(level),
          log_handler_token_(std::nullopt) {
        socket_.set(zmq::sockopt::sndhwm, static_cast<int>(MAX_PUBLISH_QUEUE_EVENTS));
        socket_.set(zmq::sockopt::sndtimeo, 1000);
        // Wait for subs to connect
        if (warm_up) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
    }

    PublisherServer::~PublisherServer() {
        PublisherServer::stop();
    }

    void PublisherServer::start() {
        if (queue_state_ || publisher_thread_.joinable() || !subscriptions_.empty()) {
            stop();
        }

        auto state = std::make_shared<QueueState>();
        state->accepting.store(true, std::memory_order_release);
        queue_state_ = state;
        try {
            publisher_thread_ = std::thread([this, state] {
                runPublisher(state);
            });
        } catch (...) {
            state->accepting.store(false, std::memory_order_release);
            queue_state_.reset();
            throw;
        }

        try {
            const std::weak_ptr<QueueState> weak_state = state;
            log_handler_token_ = core::Logger::get().add_log_handler(
                [weak_state, level = level_](core::LogLevel in_level,
                                             const core::SourceSite&,
                                             std::string_view msg) {
                    if (in_level < level) {
                        return;
                    }
                    try {
                        if (auto queue = weak_state.lock()) {
                            nlohmann::json data{
                                {"message", msg},
                                {"level", core::Logger::to_string(in_level)}};
                            enqueueEvent(queue, std::move(data), "log");
                        }
                    } catch (...) {
                        if (auto queue = weak_state.lock())
                            queue->dropped.fetch_add(1, std::memory_order_relaxed);
                    }
                });

            SUBSCRIBE_EVENT(TrainingStarted);
            SUBSCRIBE_EVENT(TrainingProgress);
            SUBSCRIBE_EVENT(TrainingPaused);
            SUBSCRIBE_EVENT(TrainingResumed);
            SUBSCRIBE_EVENT(TrainingCompleted);
            SUBSCRIBE_EVENT(TrainingStopped);
            SUBSCRIBE_EVENT(ModelUpdated);
            SUBSCRIBE_EVENT(PLYAdded);
            SUBSCRIBE_EVENT(PLYRemoved);
            SUBSCRIBE_EVENT(NodeReparented);
            SUBSCRIBE_EVENT(DatasetLoadStarted);
            SUBSCRIBE_EVENT(DatasetLoadProgress);
            SUBSCRIBE_EVENT(DatasetLoadCompleted);
            SUBSCRIBE_EVENT(ConfigLoadFailed);
            SUBSCRIBE_EVENT(FileDropFailed);
            SUBSCRIBE_EVENT(EvaluationStarted);
            SUBSCRIBE_EVENT(EvaluationProgress);
            SUBSCRIBE_EVENT(EvaluationCompleted);
            SUBSCRIBE_EVENT(CheckpointSaved);
            SUBSCRIBE_EVENT(DiskSpaceSaveFailed);
            SUBSCRIBE_EVENT(MemoryUsage);
            SUBSCRIBE_EVENT(FrameRendered);
            SUBSCRIBE_EVENT(CudaVersionUnsupported);
            SUBSCRIBE_EVENT(CudaUnavailable);
            SUBSCRIBE_EVENT(KeyframeListChanged);
            SUBSCRIBE_EVENT(ExportCompleted);
            SUBSCRIBE_EVENT(ExportFailed);
            SUBSCRIBE_EVENT(VideoExportCompleted);
            SUBSCRIBE_EVENT(VideoExportFailed);
        } catch (...) {
            stop();
            throw;
        }
    }

    void PublisherServer::stop() {
        auto state = queue_state_;
        if (state)
            state->accepting.store(false, std::memory_order_release);

        for (auto& unsubscribe : subscriptions_) {
            unsubscribe();
        }
        subscriptions_.clear();
        if (log_handler_token_.has_value()) {
            core::Logger::get().remove_log_handler(log_handler_token_.value());
            log_handler_token_ = std::nullopt; // prevent double-removal on a second stop()/dtor call
        }

        if (state)
            state->cv.notify_all();
        join();

        if (state) {
            const size_t dropped = state->dropped.load(std::memory_order_relaxed);
            if (dropped > 0)
                LOG_WARN("TCP publisher dropped {} events due to queue or payload limits", dropped);
        }
        queue_state_.reset();
    }

    void PublisherServer::join() {
        if (publisher_thread_.joinable()) {
            publisher_thread_.join();
        }
    }

    void PublisherServer::enqueueEvent(const std::shared_ptr<QueueState>& state,
                                       nlohmann::json data,
                                       std::string event_type) noexcept {
        try {
            if (!state || !state->accepting.load(std::memory_order_acquire))
                return;

            QueuedEvent event{
                .message = makeEventMessage(data, event_type),
                .event_type = std::move(event_type),
            };
            event.lossy = is_lossy_event(event.event_type);
            if (!lfs::core::add_bounded_json_cost(
                    event.message, event.estimated_bytes, MAX_PUBLISH_EVENT_BYTES)) {
                state->dropped.fetch_add(1, std::memory_order_relaxed);
                return;
            }

            std::unique_lock lock(state->mutex);
            if (!state->accepting.load(std::memory_order_acquire))
                return;

            if (is_coalescible_event(event.event_type)) {
                const auto existing = std::find_if(
                    state->queue.rbegin(), state->queue.rend(), [&](const QueuedEvent& queued) {
                        return queued.event_type == event.event_type;
                    });
                if (existing != state->queue.rend()) {
                    const size_t bytes_without_existing = state->queued_bytes - existing->estimated_bytes;
                    if (event.estimated_bytes <= MAX_PUBLISH_QUEUE_BYTES - bytes_without_existing) {
                        state->queued_bytes = bytes_without_existing + event.estimated_bytes;
                        *existing = std::move(event);
                        lock.unlock();
                        state->cv.notify_one();
                        return;
                    }
                }
            }

            const auto fits = [&] {
                return state->queue.size() < MAX_PUBLISH_QUEUE_EVENTS &&
                       event.estimated_bytes <= MAX_PUBLISH_QUEUE_BYTES - state->queued_bytes;
            };
            while (!fits() && !state->queue.empty()) {
                auto evict = std::find_if(state->queue.begin(), state->queue.end(), [](const QueuedEvent& queued) {
                    return queued.lossy;
                });
                if (evict == state->queue.end()) {
                    if (event.lossy) {
                        state->dropped.fetch_add(1, std::memory_order_relaxed);
                        return;
                    }
                    evict = state->queue.begin();
                }
                state->queued_bytes -= evict->estimated_bytes;
                state->queue.erase(evict);
                state->dropped.fetch_add(1, std::memory_order_relaxed);
            }
            if (!fits()) {
                state->dropped.fetch_add(1, std::memory_order_relaxed);
                return;
            }

            state->queued_bytes += event.estimated_bytes;
            state->queue.push_back(std::move(event));
            lock.unlock();
            state->cv.notify_one();
        } catch (...) {
            if (state)
                state->dropped.fetch_add(1, std::memory_order_relaxed);
        }
    }

    void PublisherServer::runPublisher(std::shared_ptr<QueueState> state) noexcept {
        std::string failure;
        while (true) {
            QueuedEvent event;
            {
                std::unique_lock lock(state->mutex);
                state->cv.wait(lock, [&] {
                    return !state->queue.empty() || !state->accepting.load(std::memory_order_acquire);
                });
                if (state->queue.empty())
                    break;

                event = std::move(state->queue.front());
                state->queue.pop_front();
                state->queued_bytes -= event.estimated_bytes;
            }

            if (const lfs::Status send_status = send(event.message); !send_status) {
                // A retryable send failure is transient back-pressure (HWM /
                // sndtimeo): count the drop and keep publishing. Anything else
                // is terminal for the socket, so stop and clear the queue.
                if (send_status.error().retryability() == lfs::Retryability::NotRetryable) {
                    failure = std::string(send_status.error().user_message());
                } else {
                    state->dropped.fetch_add(1, std::memory_order_relaxed);
                }
            }
            if (!failure.empty()) {
                state->accepting.store(false, std::memory_order_release);
                std::lock_guard lock(state->mutex);
                state->dropped.fetch_add(state->queue.size(), std::memory_order_relaxed);
                state->queue.clear();
                state->queued_bytes = 0;
                break;
            }
        }

        if (!failure.empty()) {
            LOG_ERROR("TCP publisher stopped after send failure: {}", failure);
        }
    }

    nlohmann::json PublisherServer::makeEventMessage(const nlohmann::json& data, const std::string& event_type) {
        return {
            {"command", "event"},
            {"event_type", event_type},
            {"data", data}};
    }

    nlohmann::json PublisherServer::wireMessageFor(const core::events::state::TrainingCompleted& event) {
        return makeEventMessage(nlohmann::json(event), "TrainingCompleted");
    }
} // namespace lfs::tcp

#undef SUBSCRIBE_EVENT
