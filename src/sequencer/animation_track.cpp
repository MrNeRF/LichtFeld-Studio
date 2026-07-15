/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "animation_track.hpp"
#include "core/assert.hpp"
#include "interpolation.hpp"

#include <algorithm>
#include <cmath>
#include <type_traits>

namespace lfs::sequencer {

    namespace {
        constexpr float TIME_EPSILON = 0.0001f;

        [[nodiscard]] bool finiteAnimationValue(const AnimationValue& value) {
            return std::visit(
                [](const auto& typed_value) {
                    using T = std::decay_t<decltype(typed_value)>;
                    if constexpr (std::is_same_v<T, bool> || std::is_same_v<T, int>) {
                        return true;
                    } else if constexpr (std::is_same_v<T, float>) {
                        return std::isfinite(typed_value) &&
                               std::abs(typed_value) <= MAX_ANIMATION_VALUE_MAGNITUDE;
                    } else if constexpr (std::is_same_v<T, glm::quat>) {
                        const float norm_squared = glm::dot(typed_value, typed_value);
                        return std::isfinite(typed_value.w) && std::isfinite(typed_value.x) &&
                               std::isfinite(typed_value.y) && std::isfinite(typed_value.z) &&
                               std::isfinite(norm_squared) &&
                               norm_squared >= MIN_ANIMATION_QUATERNION_NORM_SQUARED;
                    } else {
                        for (glm::length_t column = 0; column < typed_value.length(); ++column) {
                            if constexpr (std::is_same_v<T, glm::mat4>) {
                                for (glm::length_t row = 0; row < typed_value[column].length(); ++row) {
                                    const float component = typed_value[column][row];
                                    if (!std::isfinite(component) ||
                                        std::abs(component) > MAX_ANIMATION_VALUE_MAGNITUDE)
                                        return false;
                                }
                            } else {
                                const float component = typed_value[column];
                                if (!std::isfinite(component) ||
                                    std::abs(component) > MAX_ANIMATION_VALUE_MAGNITUDE)
                                    return false;
                            }
                        }
                        return true;
                    }
                },
                value);
        }

        [[nodiscard]] AnimationValue normalizedAnimationValue(const AnimationValue& value) {
            if (const auto* quaternion = std::get_if<glm::quat>(&value))
                return glm::normalize(*quaternion);
            return value;
        }
    } // namespace

    AnimationTrack::AnimationTrack(TrackId id, ValueType type, std::string target_path)
        : id_(id),
          type_(type),
          target_path_(std::move(target_path)) {}

    void AnimationTrack::addKeyframe(float time, const AnimationValue& value, EasingType easing) {
        LFS_ASSERT_MSG(getValueType(value) == type_, "Keyframe value type must match track type");
        LFS_ASSERT_MSG(std::isfinite(time) && std::abs(time) <= MAX_SEQUENCER_TIME_SECONDS,
                       "Animation keyframe time is outside the supported range");
        LFS_ASSERT_MSG(isValidEasingType(easing), "Animation keyframe easing is invalid");
        LFS_ASSERT_MSG(finiteAnimationValue(value), "Animation keyframe value must be bounded and finite");
        const AnimationValue stored_value = normalizedAnimationValue(value);

        for (auto& kf : keyframes_) {
            if (std::abs(kf.time - time) < TIME_EPSILON) {
                kf.value = stored_value;
                kf.easing = easing;
                return;
            }
        }

        keyframes_.push_back({time, stored_value, easing});
        sortKeyframes();
    }

    void AnimationTrack::removeKeyframe(size_t index) {
        LFS_ASSERT_MSG(index < keyframes_.size(), "Animation keyframe index is out of range");
        keyframes_.erase(keyframes_.begin() + static_cast<ptrdiff_t>(index));
    }

    void AnimationTrack::updateKeyframe(size_t index, float time, const AnimationValue& value) {
        LFS_ASSERT_MSG(index < keyframes_.size(), "Animation keyframe index is out of range");
        LFS_ASSERT_MSG(getValueType(value) == type_, "Keyframe value type must match track type");
        LFS_ASSERT_MSG(std::isfinite(time) && std::abs(time) <= MAX_SEQUENCER_TIME_SECONDS,
                       "Animation keyframe time is outside the supported range");
        LFS_ASSERT_MSG(finiteAnimationValue(value), "Animation keyframe value must be bounded and finite");

        keyframes_[index].time = time;
        keyframes_[index].value = normalizedAnimationValue(value);
        sortKeyframes();
    }

    std::optional<AnimationValue> AnimationTrack::evaluate(float time) const {
        LFS_ASSERT_MSG(std::isfinite(time), "Animation evaluation time must be finite");
        if (keyframes_.empty()) {
            return std::nullopt;
        }

        if (keyframes_.size() == 1 || time <= keyframes_.front().time) {
            return keyframes_.front().value;
        }

        if (time >= keyframes_.back().time) {
            return keyframes_.back().value;
        }

        for (size_t i = 0; i < keyframes_.size() - 1; ++i) {
            if (time >= keyframes_[i].time && time < keyframes_[i + 1].time) {
                const float segment_duration = keyframes_[i + 1].time - keyframes_[i].time;
                const float local_t = (time - keyframes_[i].time) / segment_duration;
                const float eased_t = applyEasing(local_t, keyframes_[i].easing);

                return interpolateValue(keyframes_[i].value, keyframes_[i + 1].value, eased_t);
            }
        }

        return keyframes_.back().value;
    }

    float AnimationTrack::startTime() const {
        if (keyframes_.empty()) {
            return 0.0f;
        }
        return keyframes_.front().time;
    }

    float AnimationTrack::endTime() const {
        if (keyframes_.empty()) {
            return 0.0f;
        }
        return keyframes_.back().time;
    }

    void AnimationTrack::replaceKeyframes(std::vector<GenericKeyframe> keyframes) {
        keyframes_ = std::move(keyframes);
        sortKeyframes();
    }

    void AnimationTrack::sortKeyframes() { std::sort(keyframes_.begin(), keyframes_.end()); }

} // namespace lfs::sequencer
