#include "framerate_controller.hpp"

namespace gs::visualizer {

    FramerateController::FramerateController() {
        auto now = std::chrono::high_resolution_clock::now();
        frame_start_time_ = now;
        last_frame_time_ = now;
        last_non_dropped_training_frame_time_ = now;
    }

    void FramerateController::beginFrame() {
        frame_start_time_ = std::chrono::high_resolution_clock::now();

        // Calculate time since last frame
        auto frame_duration = std::chrono::duration<float>(frame_start_time_ - last_frame_time_).count();

        // Store frame time with timestamp
        frame_times_.push_back({frame_duration, frame_start_time_});

        // Clean up old frames based on time window and max samples
        cleanupOldFrames();

        updateFPSStats();
        updatePerformanceState();

        last_frame_time_ = frame_start_time_;
    }

    void FramerateController::endFrame() {
        // This can be used for additional timing if needed in the future
    }

    bool FramerateController::shouldSkipSceneRender(bool is_training, bool scene_changed) {

        // Don't skip if scene changed - user interaction requires immediate response
        if (scene_changed) {
            return false;
        }

        // when not training - skip if we're in static mode and scene hasn't changed
        if (settings_.skip_when_static && !is_training) {
            return true;
        }

        // Skip if performance is critical and we haven't been skipping too many frames
        if (settings_.adaptive_quality && is_performance_critical_ &&
            consecutive_skips_ < max_consecutive_skips_) {
            consecutive_skips_++;
            // reset consecutive_skips_ counter
            if (consecutive_skips_ > max_consecutive_skips_) {
                consecutive_skips_ = 0;
            } else {
                return true;
            }
        }
        // if training - refresh rate should correspond to training_frame_refresh_time_sec_
        if (is_training) {
            using seconds_f = std::chrono::duration<float>;

            auto now = std::chrono::high_resolution_clock::now();
            float time_diff_sec = seconds_f(now - last_non_dropped_training_frame_time_).count();
            if (time_diff_sec < settings_.training_frame_refresh_time_sec) {
                return true;
            }
            last_non_dropped_training_frame_time_ = now;
            return false;
        }

        return false;
    }

    void FramerateController::cleanupOldFrames() {
        auto now = std::chrono::high_resolution_clock::now();

        // Remove frames older than time window
        while (!frame_times_.empty()) {
            auto age = std::chrono::duration<float>(now - frame_times_.front().timestamp).count();
            if (age > settings_.time_window_seconds) {
                frame_times_.pop_front();
            } else {
                break; // Since deque is ordered by time, we can break here
            }
        }

        // Limit maximum number of samples
        while (frame_times_.size() > settings_.max_frame_samples) {
            frame_times_.pop_front();
        }
    }

    void FramerateController::updateFPSStats() {
        if (frame_times_.empty()) {
            current_fps_ = 0.0f;
            average_fps_ = 0.0f;
            return;
        }

        // Current FPS from last frame
        if (frame_times_.back().duration > 0.0f) {
            current_fps_ = 1.0f / frame_times_.back().duration;
        }

        // Average FPS from recent frames
        if (frame_times_.size() >= 5) { // Need at least 5 samples for meaningful average
            float sum = 0.0f;
            for (const auto& frame_data : frame_times_) {
                sum += frame_data.duration;
            }
            float avg_frame_time = sum / frame_times_.size();
            if (avg_frame_time > 0.0f) {
                average_fps_ = 1.0f / avg_frame_time;
            }
        } else {
            average_fps_ = current_fps_;
        }
    }

    void FramerateController::updatePerformanceState() {
        // Consider performance critical if average FPS is below threshold
        bool was_critical = is_performance_critical_;
        is_performance_critical_ = (average_fps_ < settings_.min_fps_threshold && average_fps_ > 0.0f);

        // Reset skip counter if performance improved
        if (was_critical && !is_performance_critical_) {
            consecutive_skips_ = 0;
        }
    }

    void FramerateController::reset() {
        frame_times_.clear();
        current_fps_ = 0.0f;
        average_fps_ = 0.0f;
        is_performance_critical_ = false;
        consecutive_skips_ = 0;

        auto now = std::chrono::high_resolution_clock::now();
        frame_start_time_ = now;
        last_frame_time_ = now;
    }
} // namespace gs::visualizer
