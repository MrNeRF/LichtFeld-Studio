/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "animation_clip.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string_view>

namespace lfs::sequencer {

    AnimationClip::AnimationClip(std::string name) : name_(std::move(name)) {}

    TrackId AnimationClip::addTrack(ValueType type, const std::string& target_path) {
        if (const auto it = path_to_track_.find(target_path); it != path_to_track_.end()) {
            return it->second;
        }

        const TrackId id = next_track_id_++;
        tracks_[id] = std::make_unique<AnimationTrack>(id, type, target_path);
        path_to_track_[target_path] = id;
        return id;
    }

    void AnimationClip::removeTrack(TrackId id) {
        const auto it = tracks_.find(id);
        if (it == tracks_.end()) {
            return;
        }

        path_to_track_.erase(it->second->targetPath());
        tracks_.erase(it);
    }

    AnimationTrack* AnimationClip::getTrack(TrackId id) {
        const auto it = tracks_.find(id);
        return it != tracks_.end() ? it->second.get() : nullptr;
    }

    const AnimationTrack* AnimationClip::getTrack(TrackId id) const {
        const auto it = tracks_.find(id);
        return it != tracks_.end() ? it->second.get() : nullptr;
    }

    AnimationTrack* AnimationClip::getTrackByPath(const std::string& target_path) {
        const auto it = path_to_track_.find(target_path);
        if (it == path_to_track_.end()) {
            return nullptr;
        }
        return getTrack(it->second);
    }

    const AnimationTrack* AnimationClip::getTrackByPath(const std::string& target_path) const {
        const auto it = path_to_track_.find(target_path);
        if (it == path_to_track_.end()) {
            return nullptr;
        }
        return getTrack(it->second);
    }

    std::vector<TrackId> AnimationClip::trackIds() const {
        std::vector<TrackId> ids;
        ids.reserve(tracks_.size());
        for (const auto& [id, _] : tracks_) {
            ids.push_back(id);
        }
        return ids;
    }

    std::unordered_map<std::string, AnimationValue> AnimationClip::evaluate(float time) const {
        std::unordered_map<std::string, AnimationValue> result;
        for (const auto& [_, track] : tracks_) {
            if (auto value = track->evaluate(time)) {
                result[track->targetPath()] = *value;
            }
        }
        return result;
    }

    float AnimationClip::duration() const {
        float max_time = 0.0f;
        for (const auto& [_, track] : tracks_) {
            max_time = std::max(max_time, track->endTime());
        }
        return max_time;
    }

    namespace {
        constexpr size_t MAX_ANIMATION_TRACKS = 4096;
        constexpr size_t MAX_ANIMATION_KEYFRAMES_PER_TRACK = 100'000;
        constexpr size_t MAX_ANIMATION_KEYFRAMES_TOTAL = 1'000'000;
        constexpr size_t MAX_ANIMATION_STRING_BYTES = 4096;
        constexpr float KEYFRAME_TIME_EPSILON = 0.0001f;

        [[nodiscard]] float finiteJsonFloat(const nlohmann::json& value, const std::string_view name) {
            if (!value.is_number())
                throw std::runtime_error(std::string(name) + " must be numeric");
            const double parsed = value.get<double>();
            if (!std::isfinite(parsed) || std::abs(parsed) > MAX_ANIMATION_VALUE_MAGNITUDE)
                throw std::runtime_error(std::string(name) + " must be a bounded finite value");
            return static_cast<float>(parsed);
        }

        template <size_t N>
        [[nodiscard]] std::array<float, N> finiteJsonArray(
            const nlohmann::json& value,
            const std::string_view name) {
            if (!value.is_array() || value.size() != N)
                throw std::runtime_error(std::string(name) + " has the wrong dimensions");
            std::array<float, N> result{};
            for (size_t i = 0; i < N; ++i)
                result[i] = finiteJsonFloat(value[i], name);
            return result;
        }

        std::string valueTypeToString(ValueType type) {
            switch (type) {
            case ValueType::Bool:
                return "bool";
            case ValueType::Int:
                return "int";
            case ValueType::Float:
                return "float";
            case ValueType::Vec2:
                return "vec2";
            case ValueType::Vec3:
                return "vec3";
            case ValueType::Vec4:
                return "vec4";
            case ValueType::Quat:
                return "quat";
            case ValueType::Mat4:
                return "mat4";
            }
            return "float";
        }

        ValueType stringToValueType(const std::string& str) {
            if (str == "bool")
                return ValueType::Bool;
            if (str == "int")
                return ValueType::Int;
            if (str == "float")
                return ValueType::Float;
            if (str == "vec2")
                return ValueType::Vec2;
            if (str == "vec3")
                return ValueType::Vec3;
            if (str == "vec4")
                return ValueType::Vec4;
            if (str == "quat")
                return ValueType::Quat;
            if (str == "mat4")
                return ValueType::Mat4;
            throw std::runtime_error("Unsupported animation value type: " + str);
        }

        std::string easingTypeToString(EasingType easing) {
            switch (easing) {
            case EasingType::LINEAR:
                return "linear";
            case EasingType::EASE_IN:
                return "ease_in";
            case EasingType::EASE_OUT:
                return "ease_out";
            case EasingType::EASE_IN_OUT:
                return "ease_in_out";
            }
            return "linear";
        }

        EasingType stringToEasingType(const std::string& str) {
            if (str == "linear")
                return EasingType::LINEAR;
            if (str == "ease_in")
                return EasingType::EASE_IN;
            if (str == "ease_out")
                return EasingType::EASE_OUT;
            if (str == "ease_in_out")
                return EasingType::EASE_IN_OUT;
            throw std::runtime_error("Unsupported animation easing type: " + str);
        }

        nlohmann::json valueToJson(const AnimationValue& value) {
            return std::visit(
                [](auto&& v) -> nlohmann::json {
                    using T = std::decay_t<decltype(v)>;
                    if constexpr (std::is_same_v<T, bool> || std::is_same_v<T, int> || std::is_same_v<T, float>) {
                        return v;
                    } else if constexpr (std::is_same_v<T, glm::vec2>) {
                        return nlohmann::json::array({v.x, v.y});
                    } else if constexpr (std::is_same_v<T, glm::vec3>) {
                        return nlohmann::json::array({v.x, v.y, v.z});
                    } else if constexpr (std::is_same_v<T, glm::vec4>) {
                        return nlohmann::json::array({v.x, v.y, v.z, v.w});
                    } else if constexpr (std::is_same_v<T, glm::quat>) {
                        return nlohmann::json::array({v.w, v.x, v.y, v.z});
                    } else if constexpr (std::is_same_v<T, glm::mat4>) {
                        nlohmann::json arr = nlohmann::json::array();
                        for (int i = 0; i < 4; ++i) {
                            for (int j = 0; j < 4; ++j) {
                                arr.push_back(v[i][j]);
                            }
                        }
                        return arr;
                    }
                    return nullptr;
                },
                value);
        }

        AnimationValue jsonToValue(const nlohmann::json& j, ValueType type) {
            switch (type) {
            case ValueType::Bool:
                if (!j.is_boolean())
                    throw std::runtime_error("Animation bool value must be boolean");
                return j.get<bool>();
            case ValueType::Int:
                if (!j.is_number_integer())
                    throw std::runtime_error("Animation int value must be an integer");
                return j.get<int>();
            case ValueType::Float:
                return finiteJsonFloat(j, "Animation float value");
            case ValueType::Vec2: {
                const auto values = finiteJsonArray<2>(j, "Animation vec2 value");
                return glm::vec2(values[0], values[1]);
            }
            case ValueType::Vec3: {
                const auto values = finiteJsonArray<3>(j, "Animation vec3 value");
                return glm::vec3(values[0], values[1], values[2]);
            }
            case ValueType::Vec4: {
                const auto values = finiteJsonArray<4>(j, "Animation vec4 value");
                return glm::vec4(values[0], values[1], values[2], values[3]);
            }
            case ValueType::Quat: {
                const auto values = finiteJsonArray<4>(j, "Animation quaternion value");
                const glm::quat quaternion(values[0], values[1], values[2], values[3]);
                const float norm_squared = glm::dot(quaternion, quaternion);
                if (!std::isfinite(norm_squared) || norm_squared < MIN_ANIMATION_QUATERNION_NORM_SQUARED)
                    throw std::runtime_error("Animation quaternion value must be non-zero");
                return glm::normalize(quaternion);
            }
            case ValueType::Mat4: {
                const auto values = finiteJsonArray<16>(j, "Animation mat4 value");
                glm::mat4 m;
                for (int i = 0; i < 4; ++i) {
                    for (int k = 0; k < 4; ++k) {
                        m[i][k] = values[static_cast<size_t>(i) * 4 + static_cast<size_t>(k)];
                    }
                }
                return m;
            }
            }
            return 0.0f;
        }
    } // namespace

    nlohmann::json AnimationClip::toJson() const {
        nlohmann::json j;
        j["name"] = name_;
        j["tracks"] = nlohmann::json::array();

        for (const auto& [id, track] : tracks_) {
            nlohmann::json track_json;
            track_json["id"] = id;
            track_json["type"] = valueTypeToString(track->valueType());
            track_json["target"] = track->targetPath();
            track_json["keyframes"] = nlohmann::json::array();

            for (const auto& kf : track->keyframes()) {
                nlohmann::json kf_json;
                kf_json["time"] = kf.time;
                kf_json["value"] = valueToJson(kf.value);
                kf_json["easing"] = easingTypeToString(kf.easing);
                track_json["keyframes"].push_back(kf_json);
            }

            j["tracks"].push_back(track_json);
        }

        return j;
    }

    AnimationClip AnimationClip::fromJson(const nlohmann::json& j) {
        if (!j.is_object())
            throw std::runtime_error("Animation clip must be an object");
        if (j.contains("name") && !j["name"].is_string())
            throw std::runtime_error("Animation clip name must be a string");
        const std::string name = j.value("name", "");
        if (name.size() > MAX_ANIMATION_STRING_BYTES)
            throw std::runtime_error("Animation clip name exceeds the string budget");
        AnimationClip clip(name);

        if (!j.contains("tracks")) {
            return clip;
        }
        if (!j["tracks"].is_array())
            throw std::runtime_error("Animation clip tracks must be an array");
        if (j["tracks"].size() > MAX_ANIMATION_TRACKS)
            throw std::runtime_error("Animation clip exceeds the track-count budget");

        size_t total_keyframes = 0;
        for (const auto& track_json : j["tracks"]) {
            if (!track_json.is_object() || !track_json.contains("type") || !track_json["type"].is_string() ||
                !track_json.contains("target") || !track_json["target"].is_string()) {
                throw std::runtime_error("Animation track requires string type and target fields");
            }
            const ValueType type = stringToValueType(track_json["type"].get<std::string>());
            const std::string target = track_json["target"].get<std::string>();
            if (target.empty() || target.size() > MAX_ANIMATION_STRING_BYTES)
                throw std::runtime_error("Animation track target is empty or exceeds the string budget");
            if (clip.path_to_track_.contains(target))
                throw std::runtime_error("Animation clip contains duplicate track targets");

            TrackId id = clip.next_track_id_;
            if (track_json.contains("id")) {
                if (!track_json["id"].is_number_unsigned() && !track_json["id"].is_number_integer())
                    throw std::runtime_error("Animation track id must be an integer");
                if (track_json["id"].is_number_unsigned()) {
                    id = track_json["id"].get<TrackId>();
                } else {
                    const int64_t signed_id = track_json["id"].get<int64_t>();
                    if (signed_id <= 0)
                        throw std::runtime_error("Animation track id must be positive");
                    id = static_cast<TrackId>(signed_id);
                }
            }
            if (id == 0 || id == std::numeric_limits<TrackId>::max() || clip.tracks_.contains(id))
                throw std::runtime_error("Animation clip contains an invalid or duplicate track id");

            clip.tracks_[id] = std::make_unique<AnimationTrack>(id, type, target);
            clip.path_to_track_[target] = id;
            clip.next_track_id_ = std::max(clip.next_track_id_, id + 1);
            AnimationTrack* const track = clip.getTrack(id);

            if (track_json.contains("keyframes") && !track_json["keyframes"].is_array())
                throw std::runtime_error("Animation track keyframes must be an array");
            const nlohmann::json empty_keyframes = nlohmann::json::array();
            const auto& keyframes = track_json.contains("keyframes") ? track_json["keyframes"] : empty_keyframes;
            if (keyframes.size() > MAX_ANIMATION_KEYFRAMES_PER_TRACK ||
                total_keyframes > MAX_ANIMATION_KEYFRAMES_TOTAL - keyframes.size()) {
                throw std::runtime_error("Animation clip exceeds the keyframe-count budget");
            }
            total_keyframes += keyframes.size();

            std::vector<GenericKeyframe> parsed_keyframes;
            parsed_keyframes.reserve(keyframes.size());
            for (const auto& kf_json : keyframes) {
                if (!kf_json.is_object() || !kf_json.contains("time") || !kf_json.contains("value"))
                    throw std::runtime_error("Animation keyframe requires time and value fields");
                const float time = finiteJsonFloat(kf_json["time"], "Animation keyframe time");
                if (std::abs(time) > MAX_SEQUENCER_TIME_SECONDS)
                    throw std::runtime_error("Animation keyframe time is outside the supported range");
                if (kf_json.contains("easing") && !kf_json["easing"].is_string())
                    throw std::runtime_error("Animation easing must be a string");
                const std::string easing_name = kf_json.value("easing", "linear");
                parsed_keyframes.push_back({
                    .time = time,
                    .value = jsonToValue(kf_json["value"], type),
                    .easing = stringToEasingType(easing_name),
                });
            }
            std::stable_sort(parsed_keyframes.begin(), parsed_keyframes.end());
            for (size_t i = 1; i < parsed_keyframes.size(); ++i) {
                if (std::abs(parsed_keyframes[i].time - parsed_keyframes[i - 1].time) < KEYFRAME_TIME_EPSILON)
                    throw std::runtime_error("Animation track contains duplicate keyframe times");
            }
            track->replaceKeyframes(std::move(parsed_keyframes));
        }

        return clip;
    }

} // namespace lfs::sequencer
