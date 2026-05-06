/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "sequencer_controller.hpp"

#include <algorithm>

namespace lfs::vis {

    namespace {
        constexpr float KEYFRAME_SEEK_EPS = 1e-4f;
    } // namespace

    void SequencerController::play() {
        if (timeline_.empty())
            return;
        if (state_ == PlaybackState::STOPPED) {
            playhead_ = timeline_.startTime();
            reverse_direction_ = false;
        }
        state_ = PlaybackState::PLAYING;
    }

    void SequencerController::pause() {
        if (state_ == PlaybackState::PLAYING) {
            state_ = PlaybackState::PAUSED;
        }
    }

    void SequencerController::stop() {
        state_ = PlaybackState::STOPPED;
        playhead_ = timeline_.startTime();
        reverse_direction_ = false;
    }

    void SequencerController::togglePlayPause() {
        isPlaying() ? pause() : play();
    }

    void SequencerController::seek(const float time) {
        playhead_ = std::clamp(time, 0.0f, timeline_.clipDuration());
    }

    void SequencerController::seekToFirstKeyframe() {
        if (!timeline_.empty()) {
            playhead_ = timeline_.startTime();
            if (state_ == PlaybackState::PLAYING) {
                state_ = PlaybackState::PAUSED;
            }
        }
    }

    void SequencerController::seekToPreviousKeyframe() {
        if (timeline_.empty())
            return;

        float target_time = timeline_.startTime();
        bool found_previous = false;
        for (const auto& keyframe : timeline_.keyframes()) {
            if (keyframe.is_loop_point)
                continue;
            if (keyframe.time < playhead_ - KEYFRAME_SEEK_EPS) {
                target_time = keyframe.time;
                found_previous = true;
            }
        }

        if (!found_previous && timeline_.realKeyframeCount() > 0) {
            target_time = timeline_.startTime();
        }

        playhead_ = target_time;
        if (state_ == PlaybackState::PLAYING) {
            state_ = PlaybackState::PAUSED;
        }
    }

    void SequencerController::seekToNextKeyframe() {
        if (timeline_.empty())
            return;

        float target_time = timeline_.endTime();
        bool found_next = false;
        for (const auto& keyframe : timeline_.keyframes()) {
            if (keyframe.is_loop_point)
                continue;
            if (keyframe.time > playhead_ + KEYFRAME_SEEK_EPS) {
                target_time = keyframe.time;
                found_next = true;
                break;
            }
        }

        if (!found_next && timeline_.realKeyframeCount() > 0) {
            target_time = timeline_.realEndTime();
        }

        playhead_ = target_time;
        if (state_ == PlaybackState::PLAYING) {
            state_ = PlaybackState::PAUSED;
        }
    }

    void SequencerController::seekToLastKeyframe() {
        if (!timeline_.empty()) {
            playhead_ = timeline_.realKeyframeCount() > 0 ? timeline_.realEndTime() : timeline_.endTime();
            if (state_ == PlaybackState::PLAYING) {
                state_ = PlaybackState::PAUSED;
            }
        }
    }

    void SequencerController::setClipDuration(const float duration) {
        const float previous = timeline_.clipDuration();
        timeline_.setClipDuration(duration);
        if (timeline_.clipDuration() == previous)
            return;
        if (playhead_ > timeline_.clipDuration())
            playhead_ = timeline_.clipDuration();
        rebuildLoopKeyframe();
        markTimelineChanged();
    }

    void SequencerController::setLoopMode(const LoopMode mode) {
        if (loop_mode_ == mode)
            return;
        loop_mode_ = mode;
        rebuildLoopKeyframe();
        markTimelineChanged();
    }

    void SequencerController::toggleLoop() {
        if (loop_mode_ == LoopMode::ONCE) {
            loop_mode_ = LoopMode::LOOP;
        } else {
            loop_mode_ = LoopMode::ONCE;
        }
        rebuildLoopKeyframe();
        markTimelineChanged();
    }

    bool SequencerController::isFirstRealKeyframe(const sequencer::KeyframeId id) const {
        const auto& keyframes = timeline_.keyframes();
        const auto it = std::find_if(keyframes.begin(), keyframes.end(),
                                     [](const sequencer::Keyframe& kf) { return !kf.is_loop_point; });
        return it != keyframes.end() && it->id == id;
    }

    bool SequencerController::isLoopKeyframe(const size_t index) const {
        const auto* const keyframe = timeline_.getKeyframe(index);
        return keyframe && keyframe->is_loop_point;
    }

    bool SequencerController::isEditableKeyframe(const size_t index) const {
        const auto* const keyframe = timeline_.getKeyframe(index);
        return keyframe && !keyframe->is_loop_point;
    }

    void SequencerController::removeLoopKeyframe() {
        for (size_t index = timeline_.size(); index-- > 0;) {
            const auto* const keyframe = timeline_.getKeyframe(index);
            if (keyframe && keyframe->is_loop_point)
                timeline_.removeKeyframe(index);
        }
    }

    void SequencerController::rebuildLoopKeyframe() {
        removeLoopKeyframe();

        if (loop_mode_ != LoopMode::LOOP || timeline_.realKeyframeCount() < 2)
            return;

        const auto& keyframes = timeline_.keyframes();
        const auto first_it = std::find_if(keyframes.begin(), keyframes.end(),
                                           [](const sequencer::Keyframe& keyframe) {
                                               return !keyframe.is_loop_point;
                                           });
        if (first_it == keyframes.end())
            return;

        sequencer::Keyframe loop_kf = *first_it;
        loop_kf.id = sequencer::INVALID_KEYFRAME_ID;
        loop_kf.time = timeline_.clipDuration();
        loop_kf.is_loop_point = true;
        timeline_.addKeyframe(loop_kf);
    }

    void SequencerController::beginScrub() {
        state_ = PlaybackState::SCRUBBING;
    }

    void SequencerController::scrub(const float time) {
        playhead_ = std::clamp(time, 0.0f, timeline_.clipDuration());
    }

    void SequencerController::endScrub() {
        state_ = PlaybackState::PAUSED;
    }

    bool SequencerController::update(const float delta_seconds) {
        if (state_ != PlaybackState::PLAYING || timeline_.empty()) {
            return false;
        }

        // Playback runs across the full clip [0, clipDuration]. interpolateSpline clamps to
        // the keyframe range, so non-loop modes naturally hold the last pose past the final
        // real keyframe until the clip end.
        const float end = timeline_.clipDuration();
        const float delta = delta_seconds * playback_speed_ * (reverse_direction_ ? -1.0f : 1.0f);

        playhead_ += delta;

        switch (loop_mode_) {
        case LoopMode::ONCE:
            if (playhead_ >= end) {
                playhead_ = end;
                state_ = PlaybackState::STOPPED;
            } else if (playhead_ < 0.0f) {
                playhead_ = 0.0f;
                state_ = PlaybackState::STOPPED;
            }
            break;

        case LoopMode::LOOP:
            if (playhead_ >= end) {
                playhead_ -= end;
            } else if (playhead_ < 0.0f) {
                playhead_ += end;
            }
            break;

        case LoopMode::PING_PONG:
            if (playhead_ >= end) {
                playhead_ = end - (playhead_ - end);
                reverse_direction_ = true;
            } else if (playhead_ < 0.0f) {
                playhead_ = -playhead_;
                reverse_direction_ = false;
            }
            break;
        }
        return true;
    }

    sequencer::KeyframeId SequencerController::addKeyframe(const sequencer::Keyframe& keyframe) {
        sequencer::Keyframe inserted = keyframe;
        inserted.is_loop_point = false;

        removeLoopKeyframe();
        const auto id = timeline_.addKeyframe(inserted);
        rebuildLoopKeyframe();
        markTimelineChanged();
        return id;
    }

    sequencer::KeyframeId SequencerController::addKeyframeAtTime(const sequencer::Keyframe& keyframe, const float time) {
        auto inserted = keyframe;
        inserted.time = time;
        return addKeyframe(inserted);
    }

    bool SequencerController::setKeyframeTime(const size_t index, const float new_time) {
        const auto* const keyframe = timeline_.getKeyframe(index);
        if (!keyframe || keyframe->is_loop_point)
            return false;
        return setKeyframeTimeById(keyframe->id, new_time);
    }

    bool SequencerController::setKeyframeTimeById(const sequencer::KeyframeId id, const float new_time) {
        const auto* const keyframe = timeline_.getKeyframeById(id);
        if (!keyframe || keyframe->is_loop_point)
            return false;

        removeLoopKeyframe();
        const bool changed = timeline_.setKeyframeTimeById(id, new_time, true);
        rebuildLoopKeyframe();
        if (changed)
            markTimelineChanged();
        return changed;
    }

    bool SequencerController::previewKeyframeTimeById(const sequencer::KeyframeId id, const float new_time) {
        const auto* const keyframe = timeline_.getKeyframeById(id);
        if (!keyframe || keyframe->is_loop_point || keyframe->time == new_time)
            return false;

        removeLoopKeyframe();
        const bool changed = timeline_.setKeyframeTimeById(id, new_time, false);
        if (changed)
            markTimelineChanged();
        return changed;
    }

    bool SequencerController::commitKeyframeTimeById(const sequencer::KeyframeId id) {
        const auto* const keyframe = timeline_.getKeyframeById(id);
        if (!keyframe || keyframe->is_loop_point)
            return false;

        timeline_.sortKeyframes();
        rebuildLoopKeyframe();
        markTimelineChanged();
        return true;
    }

    bool SequencerController::updateKeyframe(const size_t index, const glm::vec3& position,
                                             const glm::quat& rotation, const float focal_length_mm) {
        const auto* const keyframe = timeline_.getKeyframe(index);
        if (!keyframe || keyframe->is_loop_point)
            return false;
        return updateKeyframeById(keyframe->id, position, rotation, focal_length_mm);
    }

    bool SequencerController::updateKeyframeById(const sequencer::KeyframeId id, const glm::vec3& position,
                                                 const glm::quat& rotation, const float focal_length_mm) {
        const auto* const keyframe = timeline_.getKeyframeById(id);
        if (!keyframe || keyframe->is_loop_point)
            return false;

        const bool changed = timeline_.updateKeyframeById(id, position, rotation, focal_length_mm);
        if (changed && isFirstRealKeyframe(id))
            rebuildLoopKeyframe();
        if (changed)
            markTimelineChanged();
        return changed;
    }

    bool SequencerController::updateSelectedKeyframe(const glm::vec3& position, const glm::quat& rotation,
                                                     const float focal_length_mm) {
        return selected_keyframe_id_.has_value() &&
               updateKeyframeById(*selected_keyframe_id_, position, rotation, focal_length_mm);
    }

    bool SequencerController::setKeyframeFocalLength(const size_t index, const float focal_length_mm) {
        const auto* const keyframe = timeline_.getKeyframe(index);
        if (!keyframe || keyframe->is_loop_point)
            return false;
        return setKeyframeFocalLengthById(keyframe->id, focal_length_mm);
    }

    bool SequencerController::setKeyframeFocalLengthById(const sequencer::KeyframeId id, const float focal_length_mm) {
        const auto* const keyframe = timeline_.getKeyframeById(id);
        if (!keyframe || keyframe->is_loop_point)
            return false;

        const bool changed = timeline_.setKeyframeFocalLengthById(id, focal_length_mm);
        if (changed && isFirstRealKeyframe(id))
            rebuildLoopKeyframe();
        if (changed)
            markTimelineChanged();
        return changed;
    }

    bool SequencerController::setKeyframeEasing(const size_t index, const sequencer::EasingType easing) {
        const auto* const keyframe = timeline_.getKeyframe(index);
        if (!keyframe || keyframe->is_loop_point)
            return false;
        return setKeyframeEasingById(keyframe->id, easing);
    }

    bool SequencerController::setKeyframeEasingById(const sequencer::KeyframeId id, const sequencer::EasingType easing) {
        const auto* const keyframe = timeline_.getKeyframeById(id);
        if (!keyframe || keyframe->is_loop_point)
            return false;

        const bool changed = timeline_.setKeyframeEasingById(id, easing);
        if (changed)
            markTimelineChanged();
        return changed;
    }

    bool SequencerController::removeKeyframeById(const sequencer::KeyframeId id) {
        const auto* const keyframe = timeline_.getKeyframeById(id);
        if (!keyframe || keyframe->is_loop_point)
            return false;

        removeLoopKeyframe();
        const bool removed = timeline_.removeKeyframeById(id);
        rebuildLoopKeyframe();
        if (removed) {
            if (selected_keyframe_id_ == id)
                deselectKeyframe();
            markTimelineChanged();
        }
        return removed;
    }

    bool SequencerController::removeSelectedKeyframe() {
        return selected_keyframe_id_.has_value() &&
               removeKeyframeById(*selected_keyframe_id_);
    }

    void SequencerController::clear() {
        stop();
        deselectKeyframe();
        timeline_.clear();
        markTimelineChanged();
    }

    bool SequencerController::saveToJson(const std::string& path) const {
        return timeline_.saveToJson(path);
    }

    bool SequencerController::loadFromJson(const std::string& path) {
        stop();
        deselectKeyframe();
        const bool loaded = timeline_.loadFromJson(path);
        if (!loaded)
            return false;
        rebuildLoopKeyframe();
        markTimelineChanged();
        return true;
    }

    bool SequencerController::selectKeyframe(const size_t index) {
        const auto* const keyframe = timeline_.getKeyframe(index);
        if (!keyframe || keyframe->is_loop_point)
            return false;
        return selectKeyframeById(keyframe->id);
    }

    bool SequencerController::selectKeyframeById(const sequencer::KeyframeId id) {
        const auto* const keyframe = timeline_.getKeyframeById(id);
        if (!keyframe || keyframe->is_loop_point)
            return false;
        if (selected_keyframe_id_ == id)
            return true;
        selected_keyframe_id_ = id;
        markSelectionChanged();
        return true;
    }

    void SequencerController::deselectKeyframe() {
        if (!selected_keyframe_id_.has_value())
            return;
        selected_keyframe_id_ = std::nullopt;
        markSelectionChanged();
    }

    std::optional<size_t> SequencerController::selectedKeyframe() const {
        if (!selected_keyframe_id_.has_value())
            return std::nullopt;

        const auto index = timeline_.findKeyframeIndex(*selected_keyframe_id_);
        if (!index.has_value())
            return std::nullopt;

        const auto* const keyframe = timeline_.getKeyframe(*index);
        if (!keyframe || keyframe->is_loop_point)
            return std::nullopt;
        return index;
    }

    sequencer::CameraState SequencerController::currentCameraState() const {
        return timeline_.evaluate(playhead_);
    }

    void SequencerController::markTimelineChanged() {
        ++timeline_revision_;
    }

    void SequencerController::markSelectionChanged() {
        ++selection_revision_;
    }

} // namespace lfs::vis
