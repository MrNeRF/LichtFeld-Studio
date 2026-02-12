/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "timeline.hpp"
#include "core/logger.hpp"
#include "interpolation.hpp"
#include "rendering/render_constants.hpp"
#include <algorithm>
#include <fstream>
#include <nlohmann/json.hpp>

namespace lfs::sequencer {

    namespace {
        constexpr int JSON_VERSION = 2;
    } // namespace

    void Timeline::addKeyframe(const Keyframe& keyframe) {
        keyframes_.push_back(keyframe);
        sortKeyframes();
    }

    void Timeline::removeKeyframe(const size_t index) {
        if (index >= keyframes_.size())
            return;
        keyframes_.erase(keyframes_.begin() + static_cast<ptrdiff_t>(index));
    }

    void Timeline::setKeyframeTime(const size_t index, const float new_time, const bool sort) {
        if (index >= keyframes_.size())
            return;
        keyframes_[index].time = new_time;
        if (sort)
            sortKeyframes();
    }

    void Timeline::updateKeyframe(const size_t index, const glm::vec3& position,
                                  const glm::quat& rotation, const float focal_length_mm) {
        if (index >= keyframes_.size())
            return;
        keyframes_[index].position = position;
        keyframes_[index].rotation = rotation;
        keyframes_[index].focal_length_mm = focal_length_mm;
    }

    void Timeline::setKeyframeFocalLength(const size_t index, const float focal_length_mm) {
        if (index >= keyframes_.size())
            return;
        keyframes_[index].focal_length_mm = std::clamp(focal_length_mm,
                                                       lfs::rendering::MIN_FOCAL_LENGTH_MM,
                                                       lfs::rendering::MAX_FOCAL_LENGTH_MM);
    }

    void Timeline::setKeyframeEasing(const size_t index, const EasingType easing) {
        if (index >= keyframes_.size())
            return;
        keyframes_[index].easing = easing;
    }

    const Keyframe* Timeline::getKeyframe(const size_t index) const {
        return index < keyframes_.size() ? &keyframes_[index] : nullptr;
    }

    void Timeline::clear() {
        keyframes_.clear();
    }

    float Timeline::duration() const {
        return keyframes_.size() < 2 ? 0.0f : keyframes_.back().time - keyframes_.front().time;
    }

    float Timeline::startTime() const {
        return keyframes_.empty() ? 0.0f : keyframes_.front().time;
    }

    float Timeline::endTime() const {
        return keyframes_.empty() ? 0.0f : keyframes_.back().time;
    }

    CameraState Timeline::evaluate(const float time) const {
        return interpolateSpline(keyframes_, time);
    }

    std::vector<glm::vec3> Timeline::generatePath(const int samples_per_segment) const {
        return generatePathPoints(keyframes_, samples_per_segment);
    }

    void Timeline::sortKeyframes() {
        std::sort(keyframes_.begin(), keyframes_.end());
    }

    bool Timeline::saveToJson(const std::string& path) const {
        try {
            nlohmann::json j;
            j["version"] = JSON_VERSION;
            j["keyframes"] = nlohmann::json::array();

            for (const auto& kf : keyframes_) {
                j["keyframes"].push_back({{"time", kf.time},
                                          {"position", {kf.position.x, kf.position.y, kf.position.z}},
                                          {"rotation", {kf.rotation.w, kf.rotation.x, kf.rotation.y, kf.rotation.z}},
                                          {"focal_length_mm", kf.focal_length_mm},
                                          {"easing", static_cast<int>(kf.easing)}});
            }

            // Save animation clip if present
            if (clip_) {
                j["animation_clip"] = clip_->toJson();
            }

            std::ofstream file(path);
            if (!file.is_open()) {
                LOG_ERROR("Failed to open timeline file: {}", path);
                return false;
            }
            file << j.dump(2);
            LOG_INFO("Saved {} keyframes to {}", keyframes_.size(), path);
            return true;
        } catch (const std::exception& e) {
            LOG_ERROR("Timeline save failed: {}", e.what());
            return false;
        }
    }

    bool Timeline::loadFromJson(const std::string& path) {
        try {
            std::ifstream file(path);
            if (!file.is_open()) {
                LOG_ERROR("Failed to open timeline file: {}", path);
                return false;
            }

            const auto j = nlohmann::json::parse(file);
            keyframes_.clear();

            const int version = j.value("version", 1);
            for (const auto& jkf : j["keyframes"]) {
                Keyframe kf;
                kf.time = jkf["time"];
                kf.position = {jkf["position"][0], jkf["position"][1], jkf["position"][2]};
                kf.rotation = {jkf["rotation"][0], jkf["rotation"][1], jkf["rotation"][2], jkf["rotation"][3]};
                if (jkf.contains("focal_length_mm")) {
                    kf.focal_length_mm = std::clamp(jkf["focal_length_mm"].get<float>(),
                                                    lfs::rendering::MIN_FOCAL_LENGTH_MM,
                                                    lfs::rendering::MAX_FOCAL_LENGTH_MM);
                } else if (version <= 1 && jkf.contains("fov")) {
                    kf.focal_length_mm = lfs::rendering::vFovToFocalLength(jkf["fov"].get<float>());
                } else {
                    LOG_DEBUG("Keyframe at t={} missing focal length, using default", kf.time);
                }
                kf.easing = static_cast<EasingType>(jkf["easing"].get<int>());
                keyframes_.push_back(kf);
            }

            // Load animation clip if present
            if (j.contains("animation_clip")) {
                clip_ = std::make_unique<AnimationClip>(AnimationClip::fromJson(j["animation_clip"]));
            }

            sortKeyframes();
            LOG_INFO("Loaded {} keyframes from {}", keyframes_.size(), path);
            return true;
        } catch (const std::exception& e) {
            LOG_ERROR("Timeline load failed: {}", e.what());
            return false;
        }
    }

    void Timeline::setAnimationClip(std::unique_ptr<AnimationClip> clip) { clip_ = std::move(clip); }

    AnimationClip& Timeline::ensureAnimationClip() {
        if (!clip_) {
            clip_ = std::make_unique<AnimationClip>("default");
        }
        return *clip_;
    }

    std::unordered_map<std::string, AnimationValue> Timeline::evaluateClip(float time) const {
        if (!clip_) {
            return {};
        }
        return clip_->evaluate(time);
    }

    float Timeline::totalDuration() const {
        float camera_duration = duration();
        float clip_duration = clip_ ? clip_->duration() : 0.0f;
        return std::max(camera_duration, clip_duration);
    }

} // namespace lfs::sequencer
