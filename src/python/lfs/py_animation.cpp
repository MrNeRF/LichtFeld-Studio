/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_animation.hpp"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

namespace lfs::python {

    namespace {
        sequencer::EasingType string_to_easing(const std::string& str) {
            if (str == "linear")
                return sequencer::EasingType::LINEAR;
            if (str == "ease_in")
                return sequencer::EasingType::EASE_IN;
            if (str == "ease_out")
                return sequencer::EasingType::EASE_OUT;
            return sequencer::EasingType::EASE_IN_OUT;
        }

        std::string easing_to_string(sequencer::EasingType easing) {
            switch (easing) {
            case sequencer::EasingType::LINEAR:
                return "linear";
            case sequencer::EasingType::EASE_IN:
                return "ease_in";
            case sequencer::EasingType::EASE_OUT:
                return "ease_out";
            case sequencer::EasingType::EASE_IN_OUT:
                return "ease_in_out";
            }
            return "linear";
        }

        sequencer::ValueType string_to_value_type(const std::string& str) {
            if (str == "bool")
                return sequencer::ValueType::Bool;
            if (str == "int")
                return sequencer::ValueType::Int;
            if (str == "float")
                return sequencer::ValueType::Float;
            if (str == "vec2")
                return sequencer::ValueType::Vec2;
            if (str == "vec3")
                return sequencer::ValueType::Vec3;
            if (str == "vec4")
                return sequencer::ValueType::Vec4;
            if (str == "quat")
                return sequencer::ValueType::Quat;
            if (str == "mat4")
                return sequencer::ValueType::Mat4;
            return sequencer::ValueType::Float;
        }

        sequencer::AnimationValue python_to_animation_value(nb::object obj, sequencer::ValueType type) {
            switch (type) {
            case sequencer::ValueType::Bool:
                return nb::cast<bool>(obj);
            case sequencer::ValueType::Int:
                return nb::cast<int>(obj);
            case sequencer::ValueType::Float:
                return nb::cast<float>(obj);
            case sequencer::ValueType::Vec2: {
                const auto list = nb::cast<nb::list>(obj);
                return glm::vec2(nb::cast<float>(list[0]), nb::cast<float>(list[1]));
            }
            case sequencer::ValueType::Vec3: {
                const auto list = nb::cast<nb::list>(obj);
                return glm::vec3(nb::cast<float>(list[0]), nb::cast<float>(list[1]), nb::cast<float>(list[2]));
            }
            case sequencer::ValueType::Vec4: {
                const auto list = nb::cast<nb::list>(obj);
                return glm::vec4(nb::cast<float>(list[0]), nb::cast<float>(list[1]), nb::cast<float>(list[2]),
                                 nb::cast<float>(list[3]));
            }
            case sequencer::ValueType::Quat: {
                const auto list = nb::cast<nb::list>(obj);
                return glm::quat(nb::cast<float>(list[0]), nb::cast<float>(list[1]), nb::cast<float>(list[2]),
                                 nb::cast<float>(list[3]));
            }
            case sequencer::ValueType::Mat4: {
                const auto list = nb::cast<nb::list>(obj);
                glm::mat4 m;
                for (int i = 0; i < 4; ++i) {
                    const auto row = nb::cast<nb::list>(list[i]);
                    for (int j = 0; j < 4; ++j) {
                        m[j][i] = nb::cast<float>(row[j]);
                    }
                }
                return m;
            }
            }
            return 0.0f;
        }

        nb::object animation_value_to_python(const sequencer::AnimationValue& value) {
            return std::visit(
                [](auto&& v) -> nb::object {
                    using T = std::decay_t<decltype(v)>;
                    if constexpr (std::is_same_v<T, bool> || std::is_same_v<T, int> || std::is_same_v<T, float>) {
                        return nb::cast(v);
                    } else if constexpr (std::is_same_v<T, glm::vec2>) {
                        nb::list l;
                        l.append(v.x);
                        l.append(v.y);
                        return l;
                    } else if constexpr (std::is_same_v<T, glm::vec3>) {
                        nb::list l;
                        l.append(v.x);
                        l.append(v.y);
                        l.append(v.z);
                        return l;
                    } else if constexpr (std::is_same_v<T, glm::vec4>) {
                        nb::list l;
                        l.append(v.x);
                        l.append(v.y);
                        l.append(v.z);
                        l.append(v.w);
                        return l;
                    } else if constexpr (std::is_same_v<T, glm::quat>) {
                        nb::list l;
                        l.append(v.w);
                        l.append(v.x);
                        l.append(v.y);
                        l.append(v.z);
                        return l;
                    } else if constexpr (std::is_same_v<T, glm::mat4>) {
                        nb::list rows;
                        for (int i = 0; i < 4; ++i) {
                            nb::list row;
                            for (int j = 0; j < 4; ++j) {
                                row.append(v[j][i]);
                            }
                            rows.append(row);
                        }
                        return rows;
                    }
                    return nb::none();
                },
                value);
        }
    } // namespace

    void PyAnimationTrack::add_keyframe(float time, nb::object value, const std::string& easing) {
        auto& track = resolve();
        const auto anim_value = python_to_animation_value(value, track.valueType());
        track.addKeyframe(time, anim_value, string_to_easing(easing));
    }

    void PyAnimationTrack::remove_keyframe(size_t index) { resolve().removeKeyframe(index); }

    nb::object PyAnimationTrack::evaluate(float time) const {
        const auto result = resolve().evaluate(time);
        if (!result) {
            return nb::none();
        }
        return animation_value_to_python(*result);
    }

    nb::list PyAnimationTrack::keyframes() const {
        nb::list result;
        for (const auto& kf : resolve().keyframes()) {
            nb::dict d;
            d["time"] = kf.time;
            d["value"] = animation_value_to_python(kf.value);
            d["easing"] = easing_to_string(kf.easing);
            result.append(d);
        }
        return result;
    }

    PyAnimationTrack PyAnimationClip::add_track(const std::string& value_type, const std::string& target_path) {
        const auto type = string_to_value_type(value_type);
        const auto id = clip_->addTrack(type, target_path);
        return PyAnimationTrack(clip_, id, owner_);
    }

    void PyAnimationClip::remove_track(uint64_t id) { clip_->removeTrack(id); }

    std::optional<PyAnimationTrack> PyAnimationClip::get_track(uint64_t id) {
        auto* const track = clip_->getTrack(id);
        if (!track) {
            return std::nullopt;
        }
        return PyAnimationTrack(clip_, track->id(), owner_);
    }

    std::optional<PyAnimationTrack> PyAnimationClip::get_track_by_path(const std::string& path) {
        auto* const track = clip_->getTrackByPath(path);
        if (!track) {
            return std::nullopt;
        }
        return PyAnimationTrack(clip_, track->id(), owner_);
    }

    nb::list PyAnimationClip::tracks() const {
        nb::list result;
        for (const auto id : clip_->trackIds()) {
            result.append(PyAnimationTrack(clip_, id, owner_));
        }
        return result;
    }

    nb::dict PyAnimationClip::evaluate(float time) const {
        nb::dict result;
        const auto values = clip_->evaluate(time);
        for (const auto& [path, value] : values) {
            result[nb::cast(path)] = animation_value_to_python(value);
        }
        return result;
    }

    PyAnimationClip PyTimeline::animation_clip() {
        return PyAnimationClip(&timeline_->ensureAnimationClip(), owned_timeline_);
    }

    nb::dict PyTimeline::evaluate_clip(float time) const {
        nb::dict result;
        const auto values = timeline_->evaluateClip(time);
        for (const auto& [path, value] : values) {
            result[nb::cast(path)] = animation_value_to_python(value);
        }
        return result;
    }

    void register_animation(nb::module_& m) {
        nb::class_<PyAnimationTrack>(m, "AnimationTrack", "A single property animation track with keyframes")
            .def_prop_ro("id", &PyAnimationTrack::id, "Track ID")
            .def_prop_ro("target_path", &PyAnimationTrack::target_path, "Target property path (e.g., 'node:Model.transform')")
            .def_prop_ro("keyframe_count", &PyAnimationTrack::keyframe_count, "Number of keyframes")
            .def("add_keyframe", &PyAnimationTrack::add_keyframe, nb::arg("time"), nb::arg("value"),
                 nb::arg("easing") = "ease_in_out", "Add a keyframe at the specified time")
            .def("remove_keyframe", &PyAnimationTrack::remove_keyframe, nb::arg("index"), "Remove keyframe at index")
            .def("evaluate", &PyAnimationTrack::evaluate, nb::arg("time"), "Evaluate the track at the given time")
            .def("keyframes", &PyAnimationTrack::keyframes, "Get all keyframes as a list of dicts");

        nb::class_<PyAnimationClip>(m, "AnimationClip", "Multi-track animation container")
            .def_prop_rw("name", &PyAnimationClip::name, &PyAnimationClip::set_name, "Clip name")
            .def_prop_ro("track_count", &PyAnimationClip::track_count, "Number of tracks")
            .def_prop_ro("duration", &PyAnimationClip::duration, "Total duration of the clip")
            .def("add_track", &PyAnimationClip::add_track, nb::arg("value_type"), nb::arg("target_path"),
                 "Add a new track. value_type: 'bool', 'int', 'float', 'vec2', 'vec3', 'vec4', 'quat', 'mat4'")
            .def("remove_track", &PyAnimationClip::remove_track, nb::arg("id"), "Remove track by ID")
            .def("get_track", &PyAnimationClip::get_track, nb::arg("id"), "Get track by ID")
            .def("get_track_by_path", &PyAnimationClip::get_track_by_path, nb::arg("path"), "Get track by target path")
            .def("tracks", &PyAnimationClip::tracks, "Get all tracks")
            .def("evaluate", &PyAnimationClip::evaluate, nb::arg("time"),
                 "Evaluate all tracks at the given time, returns dict of path -> value");

        nb::class_<PyTimeline>(m, "Timeline", "Animation timeline with camera keyframes and multi-track clips")
            .def(nb::init<>())
            .def_prop_ro("has_animation_clip", &PyTimeline::has_animation_clip, "True if an animation clip exists")
            .def_prop_ro("keyframe_count", &PyTimeline::keyframe_count, "Number of camera keyframes")
            .def_prop_ro("camera_duration", &PyTimeline::camera_duration, "Duration of camera animation")
            .def_prop_ro("total_duration", &PyTimeline::total_duration, "Total duration including all clips")
            .def("animation_clip", &PyTimeline::animation_clip, "Get or create the animation clip")
            .def("evaluate_clip", &PyTimeline::evaluate_clip, nb::arg("time"),
                 "Evaluate the animation clip at the given time");
    }

} // namespace lfs::python
