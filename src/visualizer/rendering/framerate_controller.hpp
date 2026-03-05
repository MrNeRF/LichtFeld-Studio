/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <chrono>
#include <deque>

namespace lfs::vis {

    struct FramerateSettings {
        float target_fps = 30.0f;
        float min_fps_threshold = 10.0f;
        bool adaptive_quality = true;
        bool skip_when_static = true;
        float time_window_seconds = 5.0f; // Time window to keep frame samples (seconds)
        size_t max_frame_samples = 1000;  // Maximum number of frame samples to keep
        float training_frame_refresh_time_sec = 1;
    };

    class FramerateController {
    public:
        FramerateController();

        // Update settings
        void updateSettings(const FramerateSettings& settings) { settings_ = settings; }
        const FramerateSettings& getSettings() const { return settings_; }

        // Call at the beginning of each frame
        void beginFrame();

        // Call after rendering is complete
        void endFrame();

        // Check if we should skip rendering the scene this frame
        bool shouldSkipSceneRender(bool is_training, bool scene_changed);

        // Get current FPS statistics
        float getCurrentFPS() {
            cleanupOldFrames();
            updateFPSStats();
            return current_fps_;
        }
        float getAverageFPS() {
            cleanupOldFrames();
            updateFPSStats();
            return average_fps_;
        }
        bool isPerformanceCritical() const { return is_performance_critical_; }

        // Reset state (useful when scene changes significantly)
        void reset();

    private:
        void updateFPSStats();
        void updatePerformanceState();
        void cleanupOldFrames(); // Remove old frames based on time and size limits

        FramerateSettings settings_;

        // Timing with timestamps
        std::chrono::high_resolution_clock::time_point frame_start_time_;
        std::chrono::high_resolution_clock::time_point last_frame_time_;

        // Frame timing data with timestamps
        struct FrameData {
            float duration; // Frame time in seconds
            std::chrono::high_resolution_clock::time_point timestamp;
        };
        std::deque<FrameData> frame_times_; // Store recent frame data with timestamps

        // FPS tracking
        float current_fps_ = 0.0f; // very noisy right now
        float average_fps_ = 0.0f; // average is over time_window_seconds
        bool is_performance_critical_ = false;

        // Skip logic state
        int consecutive_skips_ = 0;
        // in the worst case - we drop max_consecutive_skips_ of of max_consecutive_skips_+1 frames
        static constexpr int max_consecutive_skips_ = 10;

        std::chrono::high_resolution_clock::time_point last_non_dropped_training_frame_time_;
    };

} // namespace lfs::vis
