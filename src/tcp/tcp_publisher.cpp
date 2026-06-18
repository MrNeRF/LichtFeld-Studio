/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "tcp_publisher.hpp"
#include "core/include/core/events.hpp"
#include "core/include/core/logger.hpp"

// Define a transform to JSON for each event structure
#define ENABLE_TO_JSON(event, ...) NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_ONLY_SERIALIZE(event, __VA_ARGS__)
namespace lfs::core::events::state {
    ENABLE_TO_JSON(TrainingStarted, total_iterations);
    ENABLE_TO_JSON(TrainingProgress, iteration, loss, num_gaussians, is_refining);
    ENABLE_TO_JSON(TrainingPaused, iteration);
    ENABLE_TO_JSON(TrainingResumed, iteration);
    ENABLE_TO_JSON(TrainingCompleted, iteration, final_loss, elapsed_seconds, success, user_stopped, error);
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
    ENABLE_TO_JSON(ExportFailed, error);
    ENABLE_TO_JSON(VideoExportCompleted, path, total_frames);
    ENABLE_TO_JSON(VideoExportFailed, error);

    // CUDA version check
    ENABLE_TO_JSON(CudaVersionUnsupported, major, minor, min_major, min_minor);
} // namespace lfs::core::events::state
#undef ENABLE_TO_JSON

// Create a subscription to an event that sends a broadcast with its corresponding to unsubscribe function
#define SUBSCRIBE_EVENT(Type)                                                                      \
    subscriptions_.emplace_back([id = lfs::core::events::state::Type::when([this](const auto& e) { \
                                     std::lock_guard lock(send_mutex_);                            \
                                     if (stopped_)                                                 \
                                         return;                                                   \
                                     send(makeEventMessage(e, #Type));                             \
                                 })]() {                                                           \
        ::lfs::event::EventBridge::instance().unsubscribe(                                         \
            typeid(lfs::core::events::state::Type), id);                                           \
    })

namespace lfs::tcp {

    PublisherServer::PublisherServer(int port, std::shared_ptr<lfs::vis::TrainerManager> trainer_manager, core::LogLevel level, bool warm_up)
        : TCPServer(port, std::move(trainer_manager), zmq::socket_type::pub), stopped_(false), level_(level), log_handler_token_(std::nullopt) {
        // Wait for subs to connect
        if (warm_up) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
    }

    PublisherServer::~PublisherServer() {
        PublisherServer::stop();
    }

    void PublisherServer::start() {
        if (!subscriptions_.empty()) {
            // start() called twice without stop() — stop first to avoid duplicate subscriptions.
            stop();
        }
        stopped_ = false;

        log_handler_token_ = core::Logger::get().add_log_handler(
            [this](core::LogLevel in_level, const std::source_location& loc, std::string_view msg) {
                if (in_level < level_) {
                    return;
                }
                std::lock_guard lock(send_mutex_);
                if (stopped_) {
                    return;
                }
                nlohmann::json data{
                    {"message", msg},
                    {"level", core::Logger::to_string(in_level)}};
                send(makeEventMessage(data, "log"));
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
        SUBSCRIBE_EVENT(KeyframeListChanged);
        SUBSCRIBE_EVENT(ExportCompleted);
        SUBSCRIBE_EVENT(ExportFailed);
        SUBSCRIBE_EVENT(VideoExportCompleted);
        SUBSCRIBE_EVENT(VideoExportFailed);
    }

    void PublisherServer::stop() {
        {
            std::lock_guard lock(send_mutex_);
            stopped_ = true;
        }
        for (auto& unsubscribe : subscriptions_) {
            unsubscribe();
        }
        subscriptions_.clear();
        if (log_handler_token_.has_value()) {
            core::Logger::get().remove_log_handler(log_handler_token_.value());
            log_handler_token_ = std::nullopt; // prevent double-removal on a second stop()/dtor call
        }
    }

    nlohmann::json PublisherServer::makeEventMessage(const nlohmann::json& data, const std::string& event_type) {
        return {
            {"command", "event"},
            {"event_type", event_type},
            {"data", data}};
    }
} // namespace lfs::tcp

#undef SUBSCRIBE_EVENT
