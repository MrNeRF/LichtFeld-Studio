/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "animation_clip.hpp"
#include "core/error.hpp"
#include "core/uuid.hpp"
#include "keyframe.hpp"

#include <memory>
#include <nlohmann/json_fwd.hpp>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace lfs::sequencer {

    inline constexpr int DEFAULT_PATH_SAMPLES = 20;
    inline constexpr float DEFAULT_CLIP_DURATION_SECONDS = 30.0f;
    inline constexpr float MIN_CLIP_DURATION_SECONDS = 0.1f;

    // A reveal is evaluated independently of camera keyframes. Its progress is
    // shared by interactive playback and offline rendering.
    enum class RevealEffect : int {
        Magic = 1,
        Spread = 2,
        Unroll = 3,
        Twister = 4,
        Rain = 5,
        Unveil = 6,
        LightEdge = 7,
        Rainbow = 8,
        DarkToLight = 9,
        Distortion = 10,
        CameraLight = 11,
        Custom = 12
    };
    // Whether the effect moves splats away from their stored positions while it plays.
    [[nodiscard]] constexpr bool revealMovesSplats(const RevealEffect effect) {
        return effect <= RevealEffect::Rainbow || effect == RevealEffect::Distortion || effect == RevealEffect::Custom;
    }
    struct RevealClip {
        float start = 0.0f;
        float end = 3.0f;
        float softness = 0.1f;
        RevealEffect effect = RevealEffect::Spread;
        std::optional<core::Uuid> shader_id;

        [[nodiscard]] float progress(float time) const;
        [[nodiscard]] Status validate() const;
        [[nodiscard]] nlohmann::json toJson() const;
        [[nodiscard]] static Result<RevealClip> fromJson(const nlohmann::json& value, bool allow_extensions = false);
        bool operator==(const RevealClip&) const = default;
    };

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
        [[nodiscard]] std::vector<glm::vec3> generatePathAtTimeStep(float sample_step_seconds) const;

        [[nodiscard]] bool saveToJson(const std::string& path) const;
        [[nodiscard]] bool loadFromJson(const std::string& path);
        [[nodiscard]] nlohmann::json saveToJson() const;
        [[nodiscard]] bool loadFromJson(const nlohmann::json& json);

        // ========== Multi-Track Animation Clip ==========
        void setAnimationClip(std::unique_ptr<AnimationClip> clip);
        [[nodiscard]] AnimationClip* animationClip() { return clip_.get(); }
        [[nodiscard]] const AnimationClip* animationClip() const { return clip_.get(); }
        [[nodiscard]] bool hasAnimationClip() const { return clip_ != nullptr; }

        AnimationClip& ensureAnimationClip();

        [[nodiscard]] std::unordered_map<std::string, AnimationValue> evaluateClip(float time) const;

        [[nodiscard]] float totalDuration() const;

        [[nodiscard]] const std::optional<RevealClip>& revealClip() const { return reveal_clip_; }
        void setRevealClip(std::optional<RevealClip> clip);

    private:
        std::vector<Keyframe> keyframes_;
        std::unique_ptr<AnimationClip> clip_;
        std::optional<RevealClip> reveal_clip_;
        KeyframeId next_keyframe_id_ = 1;
        float clip_duration_ = DEFAULT_CLIP_DURATION_SECONDS;
    };

} // namespace lfs::sequencer
