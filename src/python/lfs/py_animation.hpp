/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "sequencer/animation_clip.hpp"
#include "sequencer/animation_track.hpp"
#include "sequencer/animation_value.hpp"
#include "sequencer/timeline.hpp"

#include <memory>
#include <nanobind/nanobind.h>
#include <optional>
#include <stdexcept>

namespace nb = nanobind;

namespace lfs::python {

    class PyAnimationTrack {
    public:
        PyAnimationTrack(sequencer::AnimationClip* clip, sequencer::TrackId id,
                         std::shared_ptr<sequencer::Timeline> owner)
            : clip_(clip), id_(id), owner_(std::move(owner)) {}

        [[nodiscard]] uint64_t id() const { return id_; }
        [[nodiscard]] std::string target_path() const { return resolve().targetPath(); }
        [[nodiscard]] size_t keyframe_count() const { return resolve().keyframeCount(); }

        void add_keyframe(float time, nb::object value, const std::string& easing = "ease_in_out");
        void remove_keyframe(size_t index);

        [[nodiscard]] nb::object evaluate(float time) const;
        [[nodiscard]] nb::list keyframes() const;

    private:
        [[nodiscard]] sequencer::AnimationTrack& resolve() const {
            if (auto* const track = clip_->getTrack(id_)) {
                return *track;
            }
            throw std::runtime_error("animation track " + std::to_string(id_) + " was removed");
        }

        sequencer::AnimationClip* clip_;
        sequencer::TrackId id_;
        std::shared_ptr<sequencer::Timeline> owner_;
    };

    class PyAnimationClip {
    public:
        PyAnimationClip(sequencer::AnimationClip* clip, std::shared_ptr<sequencer::Timeline> owner)
            : clip_(clip), owner_(std::move(owner)) {}

        [[nodiscard]] std::string name() const { return clip_->name(); }
        void set_name(const std::string& name) { clip_->setName(name); }

        PyAnimationTrack add_track(const std::string& value_type, const std::string& target_path);
        void remove_track(uint64_t id);

        [[nodiscard]] std::optional<PyAnimationTrack> get_track(uint64_t id);
        [[nodiscard]] std::optional<PyAnimationTrack> get_track_by_path(const std::string& path);

        [[nodiscard]] size_t track_count() const { return clip_->trackCount(); }
        [[nodiscard]] nb::list tracks() const;

        [[nodiscard]] nb::dict evaluate(float time) const;
        [[nodiscard]] float duration() const { return clip_->duration(); }

    private:
        sequencer::AnimationClip* clip_;
        std::shared_ptr<sequencer::Timeline> owner_;
    };

    class PyTimeline {
    public:
        PyTimeline()
            : owned_timeline_(std::make_shared<sequencer::Timeline>()),
              timeline_(owned_timeline_.get()) {}
        explicit PyTimeline(sequencer::Timeline* timeline) : timeline_(timeline) {}

        [[nodiscard]] PyAnimationClip animation_clip();
        [[nodiscard]] bool has_animation_clip() const { return timeline_->hasAnimationClip(); }

        [[nodiscard]] nb::dict evaluate_clip(float time) const;
        [[nodiscard]] float total_duration() const { return timeline_->totalDuration(); }

        [[nodiscard]] size_t keyframe_count() const { return timeline_->size(); }
        [[nodiscard]] float camera_duration() const { return timeline_->duration(); }

    private:
        std::shared_ptr<sequencer::Timeline> owned_timeline_;
        sequencer::Timeline* timeline_;
    };

    void register_animation(nb::module_& m);

} // namespace lfs::python
