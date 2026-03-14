/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "animation_clip.hpp"

#include <algorithm>
#include <nlohmann/json.hpp>

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
            return ValueType::Float;
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
            return EasingType::LINEAR;
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
                return j.get<bool>();
            case ValueType::Int:
                return j.get<int>();
            case ValueType::Float:
                return j.get<float>();
            case ValueType::Vec2:
                return glm::vec2(j[0].get<float>(), j[1].get<float>());
            case ValueType::Vec3:
                return glm::vec3(j[0].get<float>(), j[1].get<float>(), j[2].get<float>());
            case ValueType::Vec4:
                return glm::vec4(j[0].get<float>(), j[1].get<float>(), j[2].get<float>(), j[3].get<float>());
            case ValueType::Quat:
                return glm::quat(j[0].get<float>(), j[1].get<float>(), j[2].get<float>(), j[3].get<float>());
            case ValueType::Mat4: {
                glm::mat4 m;
                for (int i = 0; i < 4; ++i) {
                    for (int k = 0; k < 4; ++k) {
                        m[i][k] = j[i * 4 + k].get<float>();
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
        AnimationClip clip(j.value("name", ""));

        if (!j.contains("tracks")) {
            return clip;
        }

        for (const auto& track_json : j["tracks"]) {
            const ValueType type = stringToValueType(track_json.value("type", "float"));
            const std::string target = track_json.value("target", "");
            if (target.empty() || clip.path_to_track_.contains(target))
                continue;

            TrackId id = track_json.value("id", clip.next_track_id_);
            if (id == 0 || clip.tracks_.contains(id))
                id = clip.next_track_id_;

            clip.tracks_[id] = std::make_unique<AnimationTrack>(id, type, target);
            clip.path_to_track_[target] = id;
            clip.next_track_id_ = std::max(clip.next_track_id_, id + 1);
            AnimationTrack* const track = clip.getTrack(id);

            if (track && track_json.contains("keyframes")) {
                for (const auto& kf_json : track_json["keyframes"]) {
                    const float time = kf_json.value("time", 0.0f);
                    const EasingType easing = stringToEasingType(kf_json.value("easing", "linear"));
                    const AnimationValue value = jsonToValue(kf_json["value"], type);
                    track->addKeyframe(time, value, easing);
                }
            }
        }

        return clip;
    }

} // namespace lfs::sequencer
