/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "animation_clip.hpp"
#include "keyframe.hpp"

#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace lfs::sequencer {

    inline constexpr int DEFAULT_PATH_SAMPLES = 20;

    class Timeline {
    public:
        // ========== Legacy Camera Keyframes ==========
        void addKeyframe(const Keyframe& keyframe);
        void removeKeyframe(size_t index);
        void setKeyframeTime(size_t index, float new_time, bool sort = true);
        void updateKeyframe(size_t index, const glm::vec3& position, const glm::quat& rotation, float focal_length_mm);
        void setKeyframeFocalLength(size_t index, float focal_length_mm);
        void setKeyframeEasing(size_t index, EasingType easing);
        void sortKeyframes();
        void clear();

        [[nodiscard]] const Keyframe* getKeyframe(size_t index) const;

        [[nodiscard]] bool empty() const { return keyframes_.empty(); }
        [[nodiscard]] size_t size() const { return keyframes_.size(); }
        [[nodiscard]] std::span<const Keyframe> keyframes() const { return keyframes_; }

        [[nodiscard]] float duration() const;
        [[nodiscard]] float startTime() const;
        [[nodiscard]] float endTime() const;

        [[nodiscard]] CameraState evaluate(float time) const;
        [[nodiscard]] std::vector<glm::vec3> generatePath(int samples_per_segment = DEFAULT_PATH_SAMPLES) const;

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
    };

} // namespace lfs::sequencer
