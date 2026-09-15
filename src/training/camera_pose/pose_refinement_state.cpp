/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/checkpoint_format.hpp"
#include "pose_refinement_session.hpp"
#include <cmath>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <utility>

namespace lfs::training::camera_pose {
    namespace {
        using Json = nlohmann::json;

        void validate_format(const Json& state) {
            // The discriminator separates the public schema from unpublished
            // experiments. Never reinterpret an old objective on resume.
            if (auto schema = core::validate_camera_pose_state_schema(state); !schema)
                throw std::invalid_argument(schema.error());
        }

        Json settings(const PoseSessionConfig& config) {
            const auto& opt = config.optimizer;
            return {
                {"total_iterations", config.total_iterations},
                {"warmup_iterations", config.warmup_iterations},
                {"freeze_fraction", config.freeze_fraction},
                {"visits_between_updates", config.visits_between_updates},
                {"steps_per_visit", config.steps_per_visit},
                {"choose_anchors", config.choose_anchors},
                {"joint_reprojection_weight", config.joint_reprojection_weight},
                {"joint_update_rule", config.joint_reprojection_weight > 0 ? "simultaneous_adam" : "bounded_photometric"},
                {"optimizer", {
                                  {"scene_scale", opt.scene_scale},
                                  {"max_center_fraction", opt.max_center_fraction},
                                  {"max_rotation_radians", opt.max_rotation_radians},
                                  {"step_center_fraction", opt.step_center_fraction},
                                  {"step_rotation_radians", opt.step_rotation_radians},
                                  {"center_prior", opt.center_prior},
                                  {"rotation_prior", opt.rotation_prior},
                                  {"min_relative_improvement", opt.min_relative_improvement},
                                  {"max_backtracks", opt.max_backtracks},
                              }},
            };
        }

        std::uint64_t counter(const Json& value) {
            if (!value.is_number_integer() ||
                (!value.is_number_unsigned() && value.get<std::int64_t>() < 0))
                throw std::invalid_argument("Camera pose state requires nonnegative integer counters");
            const auto result = value.get<std::uint64_t>();
            if (result == std::numeric_limits<std::uint64_t>::max())
                throw std::invalid_argument("Camera pose counter has no increment headroom");
            return result;
        }

        int integer(const Json& value) {
            const auto result = counter(value);
            if (result > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
                throw std::invalid_argument("Camera pose integer out of range");
            return static_cast<int>(result);
        }

        Matrix4 matrix(const Json& value) {
            if (!value.is_array() || value.size() != 16)
                throw std::invalid_argument("Camera pose matrix requires exactly 16 numbers");
            Matrix4 result{};
            for (size_t i = 0; i < result.size(); ++i) {
                if (!value[i].is_number())
                    throw std::invalid_argument("Camera pose matrix contains a non-number");
                result[i] = value[i].get<float>();
            }
            return result;
        }
    } // namespace

    PoseSessionConfig pose_session_config_from_state(const nlohmann::json& state) {
        validate_format(state);
        const auto& saved = state.at("settings");
        const auto& opt = saved.at("optimizer");
        PoseSessionConfig config;
        config.joint_reprojection_weight = saved.at("joint_reprojection_weight").get<double>();
        config.total_iterations = integer(saved.at("total_iterations"));
        config.warmup_iterations = integer(saved.at("warmup_iterations"));
        config.freeze_fraction = saved.at("freeze_fraction").get<double>();
        config.visits_between_updates = integer(saved.at("visits_between_updates"));
        config.steps_per_visit = integer(saved.at("steps_per_visit"));
        config.choose_anchors = saved.at("choose_anchors").get<bool>();
        config.optimizer.scene_scale = opt.at("scene_scale").get<double>();
        config.optimizer.max_center_fraction = opt.at("max_center_fraction").get<double>();
        config.optimizer.max_rotation_radians = opt.at("max_rotation_radians").get<double>();
        config.optimizer.step_center_fraction = opt.at("step_center_fraction").get<double>();
        config.optimizer.step_rotation_radians = opt.at("step_rotation_radians").get<double>();
        config.optimizer.center_prior = opt.at("center_prior").get<double>();
        config.optimizer.rotation_prior = opt.at("rotation_prior").get<double>();
        config.optimizer.min_relative_improvement = opt.at("min_relative_improvement").get<double>();
        config.optimizer.max_backtracks = integer(opt.at("max_backtracks"));
        if (saved != settings(config))
            throw std::invalid_argument("Unknown or inconsistent camera pose settings");
        return config; // Constructor and restore_state validate values and membership.
    }

    nlohmann::json PoseRefinementSession::save_state() const {
        auto cameras = Json::array();
        for (const auto& entry : entries_) {
            const auto pose = entry.optimizer.snapshot();
            cameras.push_back({{"uid", pose.uid}, {"role", static_cast<int>(entry.role)}, {"source", pose.source}, {"current", pose.current}, {"revision", pose.revision}, {"accepted_steps", pose.accepted_steps}, {"rejected_steps", pose.rejected_steps}, {"display_state", static_cast<int>(entry.state)}, {"eligible_visits", entry.visits}, {"candidate_renders", entry.renders}});
            cameras.back()["adam"] = {{"steps", pose.adaptive_steps}, {"first", pose.first_moment}, {"second", pose.second_moment}};
        }
        Json result = {{"format", core::CAMERA_POSE_STATE_FORMAT},
                       {"version", core::CAMERA_POSE_STATE_VERSION},
                       {"settings", settings(config_)},
                       {"iteration", iteration_},
                       {"paused", paused_},
                       {"cameras", std::move(cameras)}};
        auto points = Json::array();
        for (size_t i = 0; i < sparse_tracks_.size(); ++i) {
            const auto& track = sparse_tracks_[i];
            points.push_back({{"id", track.point_id}, {"source", track.source}, {"current", sparse_positions_[i]}, {"fingerprint", sparse_track_fingerprint(track)}});
            points.back()["adam"] = {{"steps", sparse_adam_[i].steps}, {"first", sparse_adam_[i].first}, {"second", sparse_adam_[i].second}};
        }
        result["points"] = std::move(points);
        return result;
    }

    void PoseRefinementSession::restore_state(const nlohmann::json& state) {
        // Parse and validate on copies; no live pose, cadence, generation or
        // snapshot changes until every camera and the new snapshot are ready.
        validate_format(state);
        if (state.size() != 7u ||
            state.at("settings") != settings(config_) || !state.at("paused").is_boolean())
            throw std::invalid_argument("Camera pose state version or configuration mismatch");
        const int iteration = integer(state.at("iteration"));
        if (iteration > config_.total_iterations)
            throw std::invalid_argument("Saved pose iteration exceeds the training schedule");
        const bool paused = state.at("paused").get<bool>();
        const auto& cameras = state.at("cameras");
        if (!cameras.is_array() || cameras.size() != entries_.size())
            throw std::invalid_argument("Camera pose state membership mismatch");
        auto working = entries_;
        std::vector<bool> seen(working.size(), false);
        for (const auto& camera : cameras) {
            if (!camera.is_object() || camera.size() != 11u)
                throw std::invalid_argument("Invalid saved camera pose record");
            const int uid = integer(camera.at("uid"));
            const auto found = index_.find(uid);
            if (found == index_.end() || seen[found->second])
                throw std::invalid_argument("Saved camera pose UID missing or duplicated");
            seen[found->second] = true;
            auto& entry = working[found->second];
            if (integer(camera.at("role")) != static_cast<int>(entry.role))
                throw std::invalid_argument("Camera pose split or reference role changed");
            PoseSnapshot pose;
            pose.uid = uid;
            pose.source = matrix(camera.at("source"));
            pose.current = matrix(camera.at("current"));
            pose.revision = counter(camera.at("revision"));
            pose.accepted_steps = counter(camera.at("accepted_steps"));
            pose.rejected_steps = counter(camera.at("rejected_steps"));
            {
                const auto& adam = camera.at("adam");
                if (!adam.is_object() || adam.size() != 3 || !adam.at("first").is_array() || adam.at("first").size() != 6 ||
                    !adam.at("second").is_array() || adam.at("second").size() != 6)
                    throw std::invalid_argument("Invalid adaptive pose state");
                pose.adaptive_steps = counter(adam.at("steps"));
                pose.first_moment = adam.at("first").get<std::array<double, 6>>();
                pose.second_moment = adam.at("second").get<std::array<double, 6>>();
            }
            entry.optimizer.restore(pose);
            const int display = integer(camera.at("display_state"));
            // Only internal last-attempt states are durable; role/freeze states
            // are derived from role and schedule when snapshots are published.
            if (display > static_cast<int>(PoseDisplayState::Rejected))
                throw std::invalid_argument("Invalid durable camera pose display state");
            entry.state = static_cast<PoseDisplayState>(display);
            entry.visits = counter(camera.at("eligible_visits"));
            entry.renders = counter(camera.at("candidate_renders"));
            if (config_.joint_reprojection_weight == 0 && entry.renders < pose.accepted_steps)
                throw std::invalid_argument("Saved pose accepted steps exceed candidate renders");
        }
        auto positions = sparse_positions_;
        auto point_adam = sparse_adam_;
        {
            const auto& points = state.at("points");
            if (!points.is_array() || points.size() != sparse_tracks_.size())
                throw std::invalid_argument("Shared point state membership mismatch");
            const double limit = config_.optimizer.scene_scale * config_.optimizer.max_center_fraction;
            for (size_t i = 0; i < points.size(); ++i) {
                const auto& point = points[i];
                const auto& track = sparse_tracks_[i];
                if (!point.is_object() || point.size() != 5u ||
                    point.at("id") != Json(track.point_id) || point.at("source") != Json(track.source) ||
                    point.at("fingerprint") != Json(sparse_track_fingerprint(track)))
                    throw std::invalid_argument("Shared point source graph changed");
                {
                    const auto& adam = point.at("adam");
                    if (!adam.is_object() || adam.size() != 3 || !adam.at("first").is_array() || adam.at("first").size() != 3 ||
                        !adam.at("second").is_array() || adam.at("second").size() != 3)
                        throw std::invalid_argument("Invalid shared point Adam state");
                    auto& saved = point_adam[i];
                    saved.steps = counter(adam.at("steps"));
                    saved.first = adam.at("first").get<SparsePointPosition>();
                    saved.second = adam.at("second").get<SparsePointPosition>();
                    for (size_t a = 0; a < 3; ++a)
                        if (!std::isfinite(saved.first[a]) || !std::isfinite(saved.second[a]) || saved.second[a] < 0 ||
                            (saved.steps == 0 && (saved.first[a] != 0 || saved.second[a] != 0)))
                            throw std::invalid_argument("Invalid shared point Adam moments");
                }
                const auto& current = point.at("current");
                if (!current.is_array() || current.size() != 3)
                    throw std::invalid_argument("Shared point requires three coordinates");
                double distance = 0;
                for (size_t axis = 0; axis < 3; ++axis) {
                    if (!current[axis].is_number())
                        throw std::invalid_argument("Shared point contains a non-number");
                    const double x = current[axis].get<double>();
                    if (!std::isfinite(x))
                        throw std::invalid_argument("Shared point contains a nonfinite coordinate");
                    positions[i][axis] = x;
                    distance = std::hypot(distance, x - track.source[axis]);
                }
                if (distance > limit * (1 + 1e-12))
                    throw std::invalid_argument("Shared point exceeds source-relative displacement limit");
                auto restored_track = track;
                for (auto& m : restored_track.measurements)
                    m.pose = working[index_.at(m.camera_uid)].optimizer.snapshot().current;
                if (!std::isfinite(sparse_point_cost(restored_track, positions[i])))
                    throw std::invalid_argument("Shared point cannot be projected with restored poses");
            }
        }
        if (sequence_ == std::numeric_limits<std::uint64_t>::max())
            throw std::invalid_argument("Camera pose snapshot sequence exhausted");
        const auto snapshot = make_snapshot(working, iteration, paused, sequence_ + 1);
        entries_ = std::move(working);
        sparse_positions_ = std::move(positions);
        sparse_adam_ = std::move(point_adam);
        diagnostics_ = {};
        iteration_ = iteration;
        paused_ = paused;
        ++sequence_;
        published_.store(snapshot, std::memory_order_release);
        dirty_ = false;
        next_publish_ = std::chrono::steady_clock::now() + std::chrono::milliseconds(250);
    }
} // namespace lfs::training::camera_pose
