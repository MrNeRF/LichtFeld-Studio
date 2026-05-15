/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "sequencer/keyframe.hpp"
#include "sequencer/timeline.hpp"
#include <algorithm>
#include <optional>
#include <string>

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

    class LFS_VIS_API SequencerController {
    public:
        [[nodiscard]] sequencer::Timeline& timeline() { return timeline_; }
        [[nodiscard]] const sequencer::Timeline& timeline() const { return timeline_; }
        [[nodiscard]] uint64_t timelineRevision() const { return timeline_revision_; }
        [[nodiscard]] uint64_t selectionRevision() const { return selection_revision_; }

        void play();
        void pause();
        void stop();
        void togglePlayPause();

        void seek(float time);
        void seekToFirstKeyframe();
        void seekToPreviousKeyframe();
        void seekToNextKeyframe();
        void seekToLastKeyframe();
        void beginScrub();
        void scrub(float time);
        void endScrub();

        // Returns true if camera should update
        bool update(float delta_seconds);

        sequencer::KeyframeId addKeyframe(const sequencer::Keyframe& keyframe);
        sequencer::KeyframeId addKeyframeAtTime(const sequencer::Keyframe& keyframe, float time);
        bool setKeyframeTime(size_t index, float new_time);
        bool setKeyframeTimeById(sequencer::KeyframeId id, float new_time);
        bool previewKeyframeTimeById(sequencer::KeyframeId id, float new_time);
        bool commitKeyframeTimeById(sequencer::KeyframeId id);
        bool updateKeyframe(size_t index, const glm::vec3& position, const glm::quat& rotation, float focal_length_mm);
        bool updateKeyframeById(sequencer::KeyframeId id, const glm::vec3& position, const glm::quat& rotation, float focal_length_mm);
        bool updateSelectedKeyframe(const glm::vec3& position, const glm::quat& rotation, float focal_length_mm);
        bool setKeyframeFocalLength(size_t index, float focal_length_mm);
        bool setKeyframeFocalLengthById(sequencer::KeyframeId id, float focal_length_mm);
        bool setKeyframeEasing(size_t index, sequencer::EasingType easing);
        bool setKeyframeEasingById(sequencer::KeyframeId id, sequencer::EasingType easing);
        bool removeKeyframeById(sequencer::KeyframeId id);
        bool removeSelectedKeyframe();
        void clear();
        bool saveToJson(const std::string& path) const;
        bool loadFromJson(const std::string& path);

        bool selectKeyframe(size_t index);
        bool selectKeyframeById(sequencer::KeyframeId id);
        void deselectKeyframe();
        [[nodiscard]] std::optional<size_t> selectedKeyframe() const;
        [[nodiscard]] std::optional<sequencer::KeyframeId> selectedKeyframeId() const { return selected_keyframe_id_; }
        [[nodiscard]] bool hasSelection() const { return selected_keyframe_id_.has_value(); }

        [[nodiscard]] PlaybackState state() const { return state_; }
        [[nodiscard]] bool isPlaying() const { return state_ == PlaybackState::PLAYING; }
        [[nodiscard]] bool isStopped() const { return state_ == PlaybackState::STOPPED; }

        [[nodiscard]] float clipDuration() const { return timeline_.clipDuration(); }
        void setClipDuration(float duration);

        [[nodiscard]] LoopMode loopMode() const { return loop_mode_; }
        void setLoopMode(LoopMode mode);
        void toggleLoop();
        [[nodiscard]] bool isLoopKeyframe(size_t index) const;
        [[nodiscard]] bool isEditableKeyframe(size_t index) const;

        [[nodiscard]] float playbackSpeed() const { return playback_speed_; }
        void setPlaybackSpeed(const float speed) { playback_speed_ = std::clamp(speed, MIN_PLAYBACK_SPEED, MAX_PLAYBACK_SPEED); }

        [[nodiscard]] float playhead() const { return playhead_; }
        [[nodiscard]] sequencer::CameraState currentCameraState() const;

    private:
        [[nodiscard]] bool isFirstRealKeyframe(sequencer::KeyframeId id) const;
        void rebuildLoopKeyframe();
        void removeLoopKeyframe();
        void markTimelineChanged();
        void markSelectionChanged();

        sequencer::Timeline timeline_;
        PlaybackState state_ = PlaybackState::STOPPED;
        LoopMode loop_mode_ = LoopMode::ONCE;

        float playhead_ = 0.0f;
        float playback_speed_ = 1.0f;
        bool reverse_direction_ = false;

        std::optional<sequencer::KeyframeId> selected_keyframe_id_;
        uint64_t timeline_revision_ = 0;
        uint64_t selection_revision_ = 0;
    };

} // namespace lfs::vis
