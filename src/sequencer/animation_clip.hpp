/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "animation_track.hpp"

#include <memory>
#include <nlohmann/json_fwd.hpp>
#include <string>
#include <unordered_map>

namespace lfs::sequencer {

    class AnimationClip {
    public:
        AnimationClip() = default;
        explicit AnimationClip(std::string name);

        [[nodiscard]] const std::string& name() const { return name_; }
        void setName(const std::string& name) { name_ = name; }

        TrackId addTrack(ValueType type, const std::string& target_path);
        void removeTrack(TrackId id);
        bool renameTrack(TrackId id, const std::string& target_path);

        [[nodiscard]] AnimationTrack* getTrack(TrackId id);
        [[nodiscard]] const AnimationTrack* getTrack(TrackId id) const;
        [[nodiscard]] AnimationTrack* getTrackByPath(const std::string& target_path);
        [[nodiscard]] const AnimationTrack* getTrackByPath(const std::string& target_path) const;

        [[nodiscard]] size_t trackCount() const { return tracks_.size(); }
        [[nodiscard]] std::vector<TrackId> trackIds() const;

        [[nodiscard]] std::unordered_map<std::string, AnimationValue> evaluate(float time) const;

        [[nodiscard]] float duration() const;

        [[nodiscard]] nlohmann::json toJson() const;
        static AnimationClip fromJson(const nlohmann::json& j);

    private:
        std::string name_;
        std::unordered_map<TrackId, std::unique_ptr<AnimationTrack>> tracks_;
        std::unordered_map<std::string, TrackId> path_to_track_;
        TrackId next_track_id_ = 1;
    };

} // namespace lfs::sequencer
