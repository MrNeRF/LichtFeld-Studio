/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "sequencer_controller.hpp"

#include <algorithm>
#include <cmath>

namespace lfs::vis {

    namespace {
        constexpr float KEYFRAME_SEEK_EPS = 1e-4f;

        [[nodiscard]] float clampSequenceFps(const float fps) {
            return std::clamp(std::isfinite(fps) ? fps : DEFAULT_SEQUENCE_FPS,
                              MIN_SEQUENCE_FPS,
                              MAX_SEQUENCE_FPS);
        }

        [[nodiscard]] bool hasTimelineCameraContent(const sequencer::Timeline& timeline) {
            return timeline.realKeyframeCount() > 0 || timeline.hasAnimationClip();
        }
    } // namespace

    bool SequencerController::hasPlayableContent() const {
        return hasTimelineCameraContent(timeline_) || hasPlySequence();
    }

    float SequencerController::playbackStartTime() const {
        return (hasPlySequence() || timeline_.hasAnimationClip()) ? 0.0f : timeline_.startTime();
    }

    void SequencerController::play() {
        if (!hasPlayableContent())
            return;
        const float end = timeline_.clipDuration();
        if (state_ == PlaybackState::PAUSED && loop_mode_ == LoopMode::ONCE &&
            end > 0.0f && playhead_ >= end - KEYFRAME_SEEK_EPS) {
            playhead_ = playbackStartTime();
            reverse_direction_ = false;
        }
        if (state_ == PlaybackState::STOPPED) {
            playhead_ = playbackStartTime();
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
        playhead_ = playbackStartTime();
        reverse_direction_ = false;
    }

    void SequencerController::togglePlayPause() {
        isPlaying() ? pause() : play();
    }

    void SequencerController::seek(const float time) {
        playhead_ = std::clamp(time, 0.0f, timeline_.clipDuration());
    }

    void SequencerController::seekToFirstKeyframe() {
        if (hasPlayableContent()) {
            playhead_ = playbackStartTime();
            if (state_ == PlaybackState::PLAYING) {
                state_ = PlaybackState::PAUSED;
            }
        }
    }

    void SequencerController::seekToPreviousKeyframe() {
        if (!hasPlayableContent())
            return;

        if (timeline_.realKeyframeCount() == 0 && hasPlySequence()) {
            const auto current_frame = plySequenceFrameIndex(playhead_).value_or(0);
            const size_t target_frame = current_frame > 0 ? current_frame - 1 : 0;
            playhead_ = static_cast<float>(target_frame) / plySequenceFps();
            if (state_ == PlaybackState::PLAYING) {
                state_ = PlaybackState::PAUSED;
            }
            return;
        }

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
        if (!hasPlayableContent())
            return;

        if (timeline_.realKeyframeCount() == 0 && hasPlySequence()) {
            const auto* const sequence = plySequence();
            const auto current_frame = plySequenceFrameIndex(playhead_).value_or(0);
            const size_t target_frame = std::min(current_frame + 1, sequence->frames.size() - 1);
            playhead_ = static_cast<float>(target_frame) / plySequenceFps();
            if (state_ == PlaybackState::PLAYING) {
                state_ = PlaybackState::PAUSED;
            }
            return;
        }

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
        if (hasPlayableContent()) {
            if (timeline_.realKeyframeCount() == 0 && hasPlySequence()) {
                const auto* const sequence = plySequence();
                playhead_ = static_cast<float>(sequence->frames.size() - 1) / plySequenceFps();
            } else {
                playhead_ = timeline_.realKeyframeCount() > 0 ? timeline_.realEndTime() : timeline_.endTime();
            }
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
        if (state_ != PlaybackState::PLAYING || !hasPlayableContent()) {
            return false;
        }

        // Playback runs across the full clip [0, clipDuration]. interpolateSpline clamps to
        // the keyframe range, so non-loop modes naturally hold the last pose past the final
        // real keyframe until the clip end.
        const float end = timeline_.clipDuration();
        if (end <= 0.0f) {
            state_ = PlaybackState::STOPPED;
            playhead_ = 0.0f;
            return false;
        }
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
            playhead_ = std::fmod(playhead_, end);
            if (playhead_ < 0.0f)
                playhead_ += end;
            break;

        case LoopMode::PING_PONG:
            if (const float period = end * 2.0f; period > 0.0f) {
                float phase = std::fmod(playhead_, period);
                if (phase < 0.0f)
                    phase += period;
                if (phase <= end) {
                    playhead_ = phase;
                    reverse_direction_ = false;
                } else {
                    playhead_ = period - phase;
                    reverse_direction_ = true;
                }
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
        ply_sequence_.reset();
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

    void SequencerController::setPlySequence(std::filesystem::path directory,
                                             std::string node_name,
                                             std::vector<std::filesystem::path> paths,
                                             std::vector<std::string> node_names,
                                             const float fps) {
        const size_t frame_count = std::min(paths.size(), node_names.size());
        if (frame_count == 0) {
            clearPlySequence();
            return;
        }

        PlySequenceClip sequence;
        sequence.directory = std::move(directory);
        sequence.node_name = std::move(node_name);
        sequence.fps = clampSequenceFps(fps);
        sequence.frames.reserve(frame_count);
        for (size_t i = 0; i < frame_count; ++i) {
            sequence.frames.push_back({
                .path = std::move(paths[i]),
                .node_name = std::move(node_names[i]),
            });
        }

        const bool preserve_timeline_duration = hasTimelineCameraContent(timeline_);
        const float sequence_duration = sequence.duration();
        ply_sequence_ = std::move(sequence);
        timeline_.setClipDuration(preserve_timeline_duration
                                      ? std::max(timeline_.clipDuration(), sequence_duration)
                                      : sequence_duration);
        state_ = PlaybackState::STOPPED;
        playhead_ = 0.0f;
        reverse_direction_ = false;
        markTimelineChanged();
    }

    void SequencerController::clearPlySequence() {
        if (!ply_sequence_)
            return;

        ply_sequence_.reset();
        if (!hasTimelineCameraContent(timeline_))
            timeline_.setClipDuration(sequencer::DEFAULT_CLIP_DURATION_SECONDS);
        if (playhead_ > timeline_.clipDuration())
            playhead_ = timeline_.clipDuration();
        markTimelineChanged();
    }

    void SequencerController::setPlySequenceFps(const float fps) {
        if (!ply_sequence_)
            return;

        const float clamped = clampSequenceFps(fps);
        if (ply_sequence_->fps == clamped)
            return;

        ply_sequence_->fps = clamped;
        const float sequence_duration = ply_sequence_->duration();
        timeline_.setClipDuration(hasTimelineCameraContent(timeline_)
                                      ? std::max(timeline_.clipDuration(), sequence_duration)
                                      : sequence_duration);
        if (playhead_ > timeline_.clipDuration())
            playhead_ = timeline_.clipDuration();
        markTimelineChanged();
    }

    bool SequencerController::setPlySequenceFrameNodeName(const size_t frame_index, std::string node_name) {
        if (!ply_sequence_ || frame_index >= ply_sequence_->frames.size())
            return false;
        if (ply_sequence_->frames[frame_index].node_name == node_name)
            return true;

        ply_sequence_->frames[frame_index].node_name = std::move(node_name);
        markTimelineChanged();
        return true;
    }

    std::optional<size_t> SequencerController::plySequenceFrameIndex(const float time) const {
        const auto* const sequence = plySequence();
        if (!sequence)
            return std::nullopt;

        const float frame_time = std::max(time, 0.0f) * sequence->fps;
        const auto frame = static_cast<size_t>(std::floor(frame_time + 1e-4f));
        return std::min(frame, sequence->frames.size() - 1);
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
