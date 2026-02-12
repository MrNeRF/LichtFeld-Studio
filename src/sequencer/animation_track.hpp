/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "animation_value.hpp"
#include "keyframe.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace lfs::sequencer {

    using TrackId = uint64_t;

    struct GenericKeyframe {
        float time = 0.0f;
        AnimationValue value;
        EasingType easing = EasingType::LINEAR;

        [[nodiscard]] bool operator<(const GenericKeyframe& other) const { return time < other.time; }
    };

    class AnimationTrack {
    public:
        AnimationTrack(TrackId id, ValueType type, std::string target_path);

        [[nodiscard]] TrackId id() const { return id_; }
        [[nodiscard]] ValueType valueType() const { return type_; }
        [[nodiscard]] const std::string& targetPath() const { return target_path_; }

        void addKeyframe(float time, const AnimationValue& value, EasingType easing = EasingType::LINEAR);
        void removeKeyframe(size_t index);
        void updateKeyframe(size_t index, float time, const AnimationValue& value);

        [[nodiscard]] size_t keyframeCount() const { return keyframes_.size(); }
        [[nodiscard]] const GenericKeyframe& keyframe(size_t index) const { return keyframes_[index]; }
        [[nodiscard]] const std::vector<GenericKeyframe>& keyframes() const { return keyframes_; }

        [[nodiscard]] std::optional<AnimationValue> evaluate(float time) const;

        [[nodiscard]] float startTime() const;
        [[nodiscard]] float endTime() const;

    private:
        void sortKeyframes();

        TrackId id_;
        ValueType type_;
        std::string target_path_;
        std::vector<GenericKeyframe> keyframes_;
    };

} // namespace lfs::sequencer
