/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "sequencer/keyframe.hpp"
#include "sequencer/timeline.hpp"
#include <algorithm>
#include <optional>

namespace lfs::vis {

    inline constexpr float MIN_PLAYBACK_SPEED = 0.1f;
    inline constexpr float MAX_PLAYBACK_SPEED = 4.0f;

    enum class PlaybackState : uint8_t {
        STOPPED,
        PLAYING,
        PAUSED,
        SCRUBBING
    };

    enum class LoopMode : uint8_t {
        ONCE,
        LOOP,
        PING_PONG
    };

    class SequencerController {
    public:
        [[nodiscard]] sequencer::Timeline& timeline() { return timeline_; }
        [[nodiscard]] const sequencer::Timeline& timeline() const { return timeline_; }

        void play();
        void pause();
        void stop();
        void togglePlayPause();

        void seek(float time);
        void seekToFirstKeyframe();
        void seekToLastKeyframe();
        void beginScrub();
        void scrub(float time);
        void endScrub();

        // Returns true if camera should update
        bool update(float delta_seconds);

        void addKeyframe(const sequencer::Keyframe& keyframe);
        void updateSelectedKeyframe(const glm::vec3& position, const glm::quat& rotation, float focal_length_mm);
        void removeSelectedKeyframe();

        void selectKeyframe(size_t index);
        void deselectKeyframe();
        [[nodiscard]] std::optional<size_t> selectedKeyframe() const { return selected_keyframe_; }
        [[nodiscard]] bool hasSelection() const { return selected_keyframe_.has_value(); }

        [[nodiscard]] PlaybackState state() const { return state_; }
        [[nodiscard]] bool isPlaying() const { return state_ == PlaybackState::PLAYING; }
        [[nodiscard]] bool isStopped() const { return state_ == PlaybackState::STOPPED; }

        [[nodiscard]] LoopMode loopMode() const { return loop_mode_; }
        void setLoopMode(LoopMode mode) { loop_mode_ = mode; }
        void toggleLoop();
        void updateLoopKeyframe(); // Sync loop keyframe with first keyframe

        [[nodiscard]] float playbackSpeed() const { return playback_speed_; }
        void setPlaybackSpeed(const float speed) { playback_speed_ = std::clamp(speed, MIN_PLAYBACK_SPEED, MAX_PLAYBACK_SPEED); }

        [[nodiscard]] float playhead() const { return playhead_; }
        [[nodiscard]] sequencer::CameraState currentCameraState() const;

    private:
        void addLoopKeyframe();
        void removeLoopKeyframe();

        sequencer::Timeline timeline_;
        PlaybackState state_ = PlaybackState::STOPPED;
        LoopMode loop_mode_ = LoopMode::ONCE;

        float playhead_ = 0.0f;
        float playback_speed_ = 1.0f;
        bool reverse_direction_ = false;

        std::optional<size_t> selected_keyframe_;
    };

} // namespace lfs::vis
