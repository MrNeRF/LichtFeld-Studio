/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "animation_clip.hpp"
#include "keyframe.hpp"

#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace lfs::sequencer {

    inline constexpr int DEFAULT_PATH_SAMPLES = 20;
    inline constexpr float DEFAULT_CLIP_DURATION_SECONDS = 30.0f;
    inline constexpr float MIN_CLIP_DURATION_SECONDS = 0.1f;

    class Timeline {
    public:
        // ========== Legacy Camera Keyframes ==========
        KeyframeId addKeyframe(const Keyframe& keyframe);
        void removeKeyframe(size_t index);
        bool removeKeyframeById(KeyframeId id);
        bool setKeyframeTimeById(KeyframeId id, float new_time, bool sort = true);
        bool updateKeyframeById(KeyframeId id, const glm::vec3& position, const glm::quat& rotation, float focal_length_mm);
        bool setKeyframeFocalLengthById(KeyframeId id, float focal_length_mm);
        bool setKeyframeEasingById(KeyframeId id, EasingType easing);
        void sortKeyframes();
        void clear();

        [[nodiscard]] const Keyframe* getKeyframe(size_t index) const;
        [[nodiscard]] Keyframe* getKeyframe(size_t index);
        [[nodiscard]] const Keyframe* getKeyframeById(KeyframeId id) const;
        [[nodiscard]] Keyframe* getKeyframeById(KeyframeId id);
        [[nodiscard]] std::optional<size_t> findKeyframeIndex(KeyframeId id) const;

        [[nodiscard]] bool empty() const { return keyframes_.empty(); }
        [[nodiscard]] size_t size() const { return keyframes_.size(); }
        [[nodiscard]] std::span<const Keyframe> keyframes() const { return keyframes_; }
        [[nodiscard]] size_t realKeyframeCount() const;
        [[nodiscard]] float realEndTime() const;

        [[nodiscard]] float duration() const;
        [[nodiscard]] float startTime() const;
        [[nodiscard]] float endTime() const;

        // User-editable clip length. The setter floors to max(MIN_CLIP_DURATION_SECONDS,
        // realEndTime()) so it cannot truncate existing keyframes.
        [[nodiscard]] float clipDuration() const { return clip_duration_; }
        void setClipDuration(float duration);

        [[nodiscard]] CameraState evaluate(float time) const;
        [[nodiscard]] std::vector<glm::vec3> generatePath(int samples_per_segment = DEFAULT_PATH_SAMPLES) const;
        [[nodiscard]] std::vector<glm::vec3> generatePathAtTimeStep(float sample_step_seconds) const;

        [[nodiscard]] bool saveToJson(const std::string& path) const;
        [[nodiscard]] bool loadFromJson(const std::string& path);

        // ========== Multi-Track Animation Clip ==========
        void setAnimationClip(std::unique_ptr<AnimationClip> clip);
        [[nodiscard]] AnimationClip* animationClip() { return clip_.get(); }
        [[nodiscard]] const AnimationClip* animationClip() const { return clip_.get(); }
        [[nodiscard]] bool hasAnimationClip() const { return clip_ != nullptr; }

        AnimationClip& ensureAnimationClip();

        [[nodiscard]] std::unordered_map<std::string, AnimationValue> evaluateClip(float time) const;

        [[nodiscard]] float totalDuration() const;

    private:
        std::vector<Keyframe> keyframes_;
        std::unique_ptr<AnimationClip> clip_;
        KeyframeId next_keyframe_id_ = 1;
        float clip_duration_ = DEFAULT_CLIP_DURATION_SECONDS;
    };

} // namespace lfs::sequencer
