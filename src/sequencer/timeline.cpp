/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "timeline.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "interpolation.hpp"
#include "rendering/render_constants.hpp"
#include <algorithm>
#include <fstream>
#include <nlohmann/json.hpp>

namespace lfs::sequencer {

    namespace {
        constexpr int JSON_VERSION = 4;

        [[nodiscard]] float clampFocalLength(const float focal_length_mm) {
            return std::clamp(focal_length_mm,
                              lfs::rendering::MIN_FOCAL_LENGTH_MM,
                              lfs::rendering::MAX_FOCAL_LENGTH_MM);
        }
    } // namespace

    KeyframeId Timeline::addKeyframe(const Keyframe& keyframe) {
        Keyframe inserted = keyframe;
        if (inserted.id == INVALID_KEYFRAME_ID) {
            inserted.id = next_keyframe_id_++;
        } else {
            next_keyframe_id_ = std::max(next_keyframe_id_, inserted.id + 1);
        }

        keyframes_.push_back(inserted);
        sortKeyframes();
        if (!inserted.is_loop_point && inserted.time > clip_duration_)
            clip_duration_ = inserted.time;
        return inserted.id;
    }

    void Timeline::removeKeyframe(const size_t index) {
        if (index >= keyframes_.size())
            return;
        keyframes_.erase(keyframes_.begin() + static_cast<ptrdiff_t>(index));
    }

    bool Timeline::removeKeyframeById(const KeyframeId id) {
        const auto index = findKeyframeIndex(id);
        if (!index.has_value())
            return false;
        removeKeyframe(*index);
        return true;
    }

    bool Timeline::setKeyframeTimeById(const KeyframeId id, const float new_time, const bool sort) {
        auto* const keyframe = getKeyframeById(id);
        if (!keyframe)
            return false;
        keyframe->time = new_time;
        if (sort)
            sortKeyframes();
        return true;
    }

    bool Timeline::updateKeyframeById(const KeyframeId id, const glm::vec3& position,
                                      const glm::quat& rotation, const float focal_length_mm) {
        auto* const keyframe = getKeyframeById(id);
        if (!keyframe)
            return false;
        keyframe->position = position;
        keyframe->rotation = rotation;
        keyframe->focal_length_mm = clampFocalLength(focal_length_mm);
        return true;
    }

    bool Timeline::setKeyframeFocalLengthById(const KeyframeId id, const float focal_length_mm) {
        auto* const keyframe = getKeyframeById(id);
        if (!keyframe)
            return false;
        keyframe->focal_length_mm = clampFocalLength(focal_length_mm);
        return true;
    }

    bool Timeline::setKeyframeEasingById(const KeyframeId id, const EasingType easing) {
        auto* const keyframe = getKeyframeById(id);
        if (!keyframe)
            return false;
        keyframe->easing = easing;
        return true;
    }

    const Keyframe* Timeline::getKeyframe(const size_t index) const {
        return index < keyframes_.size() ? &keyframes_[index] : nullptr;
    }

    Keyframe* Timeline::getKeyframe(const size_t index) {
        return index < keyframes_.size() ? &keyframes_[index] : nullptr;
    }

    const Keyframe* Timeline::getKeyframeById(const KeyframeId id) const {
        if (const auto index = findKeyframeIndex(id); index.has_value()) {
            return &keyframes_[*index];
        }
        return nullptr;
    }

    Keyframe* Timeline::getKeyframeById(const KeyframeId id) {
        if (const auto index = findKeyframeIndex(id); index.has_value()) {
            return &keyframes_[*index];
        }
        return nullptr;
    }

    std::optional<size_t> Timeline::findKeyframeIndex(const KeyframeId id) const {
        if (id == INVALID_KEYFRAME_ID)
            return std::nullopt;

        for (size_t i = 0; i < keyframes_.size(); ++i) {
            if (keyframes_[i].id == id)
                return i;
        }
        return std::nullopt;
    }

    void Timeline::clear() {
        keyframes_.clear();
        clip_.reset();
        next_keyframe_id_ = 1;
        clip_duration_ = DEFAULT_CLIP_DURATION_SECONDS;
    }

    size_t Timeline::realKeyframeCount() const {
        return static_cast<size_t>(std::count_if(
            keyframes_.begin(), keyframes_.end(),
            [](const Keyframe& keyframe) { return !keyframe.is_loop_point; }));
    }

    float Timeline::realEndTime() const {
        for (auto it = keyframes_.rbegin(); it != keyframes_.rend(); ++it) {
            if (!it->is_loop_point)
                return it->time;
        }
        return 0.0f;
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

    void Timeline::setClipDuration(const float duration) {
        clip_duration_ = std::max({MIN_CLIP_DURATION_SECONDS, duration, realEndTime()});
    }

    CameraState Timeline::evaluate(const float time) const {
        return interpolateSpline(keyframes_, time);
    }

    std::vector<glm::vec3> Timeline::generatePath(const int samples_per_segment) const {
        return generatePathPoints(keyframes_, samples_per_segment);
    }

    std::vector<glm::vec3> Timeline::generatePathAtTimeStep(const float sample_step_seconds) const {
        if (keyframes_.size() < 2) {
            return keyframes_.empty() ? std::vector<glm::vec3>{}
                                      : std::vector<glm::vec3>{keyframes_.front().position};
        }

        const float start = startTime();
        const float end = endTime();
        if (end <= start)
            return {evaluate(start).position};

        const float step = sample_step_seconds > 0.0f ? sample_step_seconds : 1.0f / 30.0f;

        std::vector<glm::vec3> points;
        const auto reserve_count =
            static_cast<size_t>(std::ceil((end - start) / step)) + 1;
        points.reserve(std::max<size_t>(reserve_count, 2));

        for (float time = start; time < end; time += step)
            points.push_back(evaluate(time).position);
        points.push_back(evaluate(end).position);
        return points;
    }

    void Timeline::sortKeyframes() {
        std::sort(keyframes_.begin(), keyframes_.end());
    }

    bool Timeline::saveToJson(const std::string& path) const {
        try {
            const std::filesystem::path path_fs = lfs::core::utf8_to_path(path);
            nlohmann::json j;
            j["version"] = JSON_VERSION;
            j["clip_duration"] = clip_duration_;
            j["keyframes"] = nlohmann::json::array();

            for (const auto& kf : keyframes_) {
                if (kf.is_loop_point)
                    continue;
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

            std::ofstream file;
            if (!lfs::core::open_file_for_write(path_fs, file)) {
                LOG_ERROR("Failed to open timeline file: {}", path);
                return false;
            }
            file << j.dump(2);
            LOG_INFO("Saved {} keyframes to {}", realKeyframeCount(), path);
            return true;
        } catch (const std::exception& e) {
            LOG_ERROR("Timeline save failed: {}", e.what());
            return false;
        }
    }

    bool Timeline::loadFromJson(const std::string& path) {
        try {
            const std::filesystem::path path_fs = lfs::core::utf8_to_path(path);
            std::ifstream file;
            if (!lfs::core::open_file_for_read(path_fs, file)) {
                LOG_ERROR("Failed to open timeline file: {}", path);
                return false;
            }

            const auto j = nlohmann::json::parse(file);

            std::vector<Keyframe> loaded_keyframes;
            loaded_keyframes.reserve(j.at("keyframes").size());
            KeyframeId next_keyframe_id = 1;

            for (const auto& jkf : j.at("keyframes")) {
                Keyframe kf;
                kf.id = next_keyframe_id++;
                kf.time = jkf.at("time").get<float>();
                kf.position = {jkf.at("position").at(0).get<float>(),
                               jkf.at("position").at(1).get<float>(),
                               jkf.at("position").at(2).get<float>()};
                kf.rotation = {jkf.at("rotation").at(0).get<float>(),
                               jkf.at("rotation").at(1).get<float>(),
                               jkf.at("rotation").at(2).get<float>(),
                               jkf.at("rotation").at(3).get<float>()};
                kf.focal_length_mm = clampFocalLength(jkf.at("focal_length_mm").get<float>());
                kf.easing = static_cast<EasingType>(jkf.at("easing").get<int>());
                loaded_keyframes.push_back(kf);
            }

            std::unique_ptr<AnimationClip> loaded_clip;
            if (j.contains("animation_clip")) {
                loaded_clip = std::make_unique<AnimationClip>(AnimationClip::fromJson(j["animation_clip"]));
            }

            keyframes_ = std::move(loaded_keyframes);
            clip_ = std::move(loaded_clip);
            next_keyframe_id_ = next_keyframe_id;
            sortKeyframes();
            // Pre-v4 fallback: setClipDuration floors to realEndTime() so loaded keyframes
            // outside the default 30s clip remain visible.
            const float loaded_duration = j.value("clip_duration", 0.0f);
            setClipDuration(loaded_duration > 0.0f ? loaded_duration : DEFAULT_CLIP_DURATION_SECONDS);
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
