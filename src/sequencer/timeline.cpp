/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "timeline.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "interpolation.hpp"
#include "io/atomic_output.hpp"
#include "rendering/render_constants.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string_view>

namespace lfs::sequencer {

    namespace {
        constexpr int JSON_VERSION = 4;
        constexpr uintmax_t MAX_TIMELINE_JSON_BYTES = 16ULL * 1024ULL * 1024ULL;
        constexpr size_t MAX_TIMELINE_KEYFRAMES = 100'000;
        constexpr float MIN_QUATERNION_NORM_SQUARED = 1.0e-12f;
        constexpr float MAX_KEYFRAME_POSITION_MAGNITUDE = 1.0e12f;

        [[nodiscard]] bool finiteVec3(const glm::vec3& value) {
            return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z) &&
                   std::abs(value.x) <= MAX_KEYFRAME_POSITION_MAGNITUDE &&
                   std::abs(value.y) <= MAX_KEYFRAME_POSITION_MAGNITUDE &&
                   std::abs(value.z) <= MAX_KEYFRAME_POSITION_MAGNITUDE;
        }

        [[nodiscard]] bool finiteQuat(const glm::quat& value) {
            return std::isfinite(value.w) && std::isfinite(value.x) &&
                   std::isfinite(value.y) && std::isfinite(value.z);
        }

        [[nodiscard]] bool validTime(const float time) {
            return std::isfinite(time) && std::abs(time) <= MAX_SEQUENCER_TIME_SECONDS;
        }

        [[nodiscard]] glm::quat normalizedRotation(const glm::quat& rotation) {
            if (!finiteQuat(rotation))
                throw std::invalid_argument("Sequencer rotation must be finite");
            const float norm_squared = glm::dot(rotation, rotation);
            if (!std::isfinite(norm_squared) || norm_squared < MIN_QUATERNION_NORM_SQUARED)
                throw std::invalid_argument("Sequencer rotation quaternion must be non-zero");
            return glm::normalize(rotation);
        }

        void validateKeyframe(Keyframe& keyframe) {
            if (!validTime(keyframe.time))
                throw std::invalid_argument("Sequencer keyframe time is outside the supported range");
            if (!finiteVec3(keyframe.position))
                throw std::invalid_argument("Sequencer keyframe position must be bounded and finite");
            keyframe.rotation = normalizedRotation(keyframe.rotation);
            if (!std::isfinite(keyframe.focal_length_mm))
                throw std::invalid_argument("Sequencer focal length must be finite");
            if (!isValidEasingType(keyframe.easing))
                throw std::invalid_argument("Sequencer easing value is invalid");
        }

        [[nodiscard]] float jsonFloat(const nlohmann::json& value, const std::string_view name) {
            if (!value.is_number())
                throw std::runtime_error(std::string(name) + " must be numeric");
            const double parsed = value.get<double>();
            if (!std::isfinite(parsed) || std::abs(parsed) > std::numeric_limits<float>::max())
                throw std::runtime_error(std::string(name) + " must be a finite float32 value");
            return static_cast<float>(parsed);
        }

        template <size_t N>
        [[nodiscard]] std::array<float, N> jsonFloatArray(
            const nlohmann::json& value,
            const std::string_view name) {
            if (!value.is_array() || value.size() != N)
                throw std::runtime_error(std::string(name) + " has the wrong dimensions");
            std::array<float, N> result{};
            for (size_t i = 0; i < N; ++i)
                result[i] = jsonFloat(value[i], name);
            return result;
        }

        [[nodiscard]] float clampFocalLength(const float focal_length_mm) {
            return std::clamp(focal_length_mm,
                              lfs::rendering::MIN_FOCAL_LENGTH_MM,
                              lfs::rendering::MAX_FOCAL_LENGTH_MM);
        }
    } // namespace

    KeyframeId Timeline::addKeyframe(const Keyframe& keyframe) {
        Keyframe inserted = keyframe;
        validateKeyframe(inserted);
        if (keyframes_.size() >= MAX_TIMELINE_KEYFRAMES)
            throw std::length_error("Timeline exceeds the keyframe-count budget");
        if (inserted.id == INVALID_KEYFRAME_ID) {
            inserted.id = next_keyframe_id_++;
        } else {
            if (inserted.id == std::numeric_limits<KeyframeId>::max())
                throw std::invalid_argument("Sequencer keyframe id is outside the supported range");
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
        if (!validTime(new_time))
            return false;
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
        const float rotation_norm_squared = glm::dot(rotation, rotation);
        if (!finiteVec3(position) || !finiteQuat(rotation) || !std::isfinite(rotation_norm_squared) ||
            rotation_norm_squared < MIN_QUATERNION_NORM_SQUARED ||
            !std::isfinite(focal_length_mm)) {
            return false;
        }
        auto* const keyframe = getKeyframeById(id);
        if (!keyframe)
            return false;
        keyframe->position = position;
        keyframe->rotation = glm::normalize(rotation);
        keyframe->focal_length_mm = clampFocalLength(focal_length_mm);
        return true;
    }

    bool Timeline::setKeyframeFocalLengthById(const KeyframeId id, const float focal_length_mm) {
        if (!std::isfinite(focal_length_mm))
            return false;
        auto* const keyframe = getKeyframeById(id);
        if (!keyframe)
            return false;
        keyframe->focal_length_mm = clampFocalLength(focal_length_mm);
        return true;
    }

    bool Timeline::setKeyframeEasingById(const KeyframeId id, const EasingType easing) {
        if (!isValidEasingType(easing))
            return false;
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
        if (!std::isfinite(duration) || duration < 0.0f || duration > MAX_SEQUENCER_TIME_SECONDS)
            throw std::invalid_argument("Timeline duration is outside the supported range");
        clip_duration_ = std::max({MIN_CLIP_DURATION_SECONDS, duration, realEndTime()});
    }

    CameraState Timeline::evaluate(const float time) const {
        if (!std::isfinite(time))
            throw std::invalid_argument("Timeline evaluation time must be finite");
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

        if (!std::isfinite(sample_step_seconds))
            throw std::invalid_argument("Timeline path sample step must be finite");
        const float step = sample_step_seconds > 0.0f ? sample_step_seconds : 1.0f / 30.0f;

        std::vector<glm::vec3> points;
        const double intervals_double = std::ceil(
            static_cast<double>(end - start) / static_cast<double>(step));
        if (!std::isfinite(intervals_double) ||
            intervals_double >= static_cast<double>(MAX_GENERATED_PATH_SAMPLES)) {
            throw std::length_error("Timeline path exceeds the sample budget");
        }
        const size_t intervals = std::max<size_t>(static_cast<size_t>(intervals_double), 1);
        points.reserve(intervals + 1);

        for (size_t sample = 0; sample < intervals; ++sample) {
            const float time = start + static_cast<float>(sample) * step;
            points.push_back(evaluate(std::min(time, end)).position);
        }
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

            if (auto result = lfs::io::ensure_output_parent_directory(path_fs); !result) {
                LOG_ERROR("Failed to prepare timeline output '{}': {}", path, result.error().format());
                return false;
            }
            lfs::io::ScopedAtomicOutputFile atomic_output(
                path_fs,
                lfs::io::AtomicOutputTempName::AppendSuffix,
                lfs::io::AtomicOutputDurability::Durable);
            std::ofstream file;
            if (!lfs::core::open_file_for_write(atomic_output.temp_path(), file)) {
                LOG_ERROR("Failed to open timeline file: {}", path);
                return false;
            }
            file << j.dump(2);
            file.close();
            if (!file) {
                LOG_ERROR("Failed to write complete timeline file: {}", path);
                return false;
            }
            if (auto result = atomic_output.commit(); !result) {
                LOG_ERROR("Failed to publish timeline file '{}': {}", path, result.error().format());
                return false;
            }
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

            std::error_code size_error;
            const uintmax_t file_size = std::filesystem::file_size(path_fs, size_error);
            if (size_error || file_size == 0 || file_size > MAX_TIMELINE_JSON_BYTES)
                throw std::runtime_error("Timeline JSON is empty or exceeds the 16 MiB input budget");
            std::vector<char> json_bytes(static_cast<size_t>(file_size) + 1);
            file.read(json_bytes.data(), static_cast<std::streamsize>(json_bytes.size()));
            const size_t bytes_read = static_cast<size_t>(file.gcount());
            if (bytes_read != file_size)
                throw std::runtime_error("Timeline JSON changed size while it was being read");

            const auto j = nlohmann::json::parse(
                json_bytes.begin(), json_bytes.begin() + static_cast<ptrdiff_t>(bytes_read));
            if (!j.is_object())
                throw std::runtime_error("Timeline JSON root must be an object");
            if (j.contains("version")) {
                if (!j["version"].is_number_integer())
                    throw std::runtime_error("Timeline version must be an integer");
                const int version = j["version"].get<int>();
                if (version <= 0 || version > JSON_VERSION)
                    throw std::runtime_error("Unsupported timeline JSON version");
            }
            if (!j.contains("keyframes") || !j["keyframes"].is_array())
                throw std::runtime_error("Timeline JSON must contain a keyframes array");
            if (j["keyframes"].size() > MAX_TIMELINE_KEYFRAMES)
                throw std::runtime_error("Timeline JSON exceeds the keyframe-count budget");

            std::vector<Keyframe> loaded_keyframes;
            loaded_keyframes.reserve(j["keyframes"].size());
            KeyframeId next_keyframe_id = 1;

            for (const auto& jkf : j["keyframes"]) {
                if (!jkf.is_object())
                    throw std::runtime_error("Timeline keyframe must be an object");
                const auto position = jsonFloatArray<3>(jkf.at("position"), "Timeline position");
                const auto rotation = jsonFloatArray<4>(jkf.at("rotation"), "Timeline rotation");
                if (!jkf.at("easing").is_number_integer())
                    throw std::runtime_error("Timeline easing must be an integer");
                Keyframe kf;
                kf.id = next_keyframe_id++;
                kf.time = jsonFloat(jkf.at("time"), "Timeline keyframe time");
                kf.position = {position[0], position[1], position[2]};
                kf.rotation = {rotation[0], rotation[1], rotation[2], rotation[3]};
                kf.focal_length_mm = clampFocalLength(
                    jsonFloat(jkf.at("focal_length_mm"), "Timeline focal length"));
                kf.easing = static_cast<EasingType>(jkf.at("easing").get<int>());
                validateKeyframe(kf);
                loaded_keyframes.push_back(kf);
            }

            std::unique_ptr<AnimationClip> loaded_clip;
            if (j.contains("animation_clip")) {
                loaded_clip = std::make_unique<AnimationClip>(AnimationClip::fromJson(j["animation_clip"]));
            }

            std::sort(loaded_keyframes.begin(), loaded_keyframes.end());
            // Pre-v4 fallback: setClipDuration floors to realEndTime() so loaded keyframes
            // outside the default 30s clip remain visible.
            float loaded_duration = DEFAULT_CLIP_DURATION_SECONDS;
            if (j.contains("clip_duration")) {
                loaded_duration = jsonFloat(j["clip_duration"], "Timeline clip_duration");
                if (loaded_duration < 0.0f || loaded_duration > MAX_SEQUENCER_TIME_SECONDS)
                    throw std::runtime_error("Timeline clip_duration is outside the supported range");
                if (loaded_duration == 0.0f)
                    loaded_duration = DEFAULT_CLIP_DURATION_SECONDS;
            }
            float real_end_time = 0.0f;
            for (auto it = loaded_keyframes.rbegin(); it != loaded_keyframes.rend(); ++it) {
                if (!it->is_loop_point) {
                    real_end_time = it->time;
                    break;
                }
            }
            const float final_duration = std::max({MIN_CLIP_DURATION_SECONDS, loaded_duration, real_end_time});

            keyframes_ = std::move(loaded_keyframes);
            clip_ = std::move(loaded_clip);
            next_keyframe_id_ = next_keyframe_id;
            clip_duration_ = final_duration;
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
