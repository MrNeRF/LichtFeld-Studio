/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "pose_refinement_session.hpp"
#include "core/logger.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace lfs::training::camera_pose {
    namespace {
        std::array<double, 3> camera_center(const Matrix4& pose) {
            std::array<double, 3> result{};
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    result[i] -= static_cast<double>(pose[4 * j + i]) * pose[4 * j + 3];
            return result;
        }
        double squared_distance(const Matrix4& a, const Matrix4& b) {
            const auto ca = camera_center(a), cb = camera_center(b);
            double result = 0;
            for (int i = 0; i < 3; ++i)
                result += (ca[i] - cb[i]) * (ca[i] - cb[i]);
            return result;
        }
    } // namespace

    std::string_view pose_display_state_name(PoseDisplayState state) noexcept {
        switch (state) {
        case PoseDisplayState::Waiting: return "waiting";
        case PoseDisplayState::Ready: return "ready";
        case PoseDisplayState::Updated: return "updated";
        case PoseDisplayState::Rejected: return "rejected";
        case PoseDisplayState::Frozen: return "frozen";
        case PoseDisplayState::Anchor: return "anchor";
        case PoseDisplayState::Evaluation: return "evaluation";
        }
        return "unknown";
    }

    PoseRefinementSession::PoseRefinementSession(std::uint64_t generation,
                                                 std::vector<PoseCameraInput> cameras, PoseSessionConfig config)
        : config_(config), generation_(generation) {
        if (generation == 0 || config.total_iterations <= 0 || config.warmup_iterations < 0 ||
            !std::isfinite(config.freeze_fraction) || config.freeze_fraction <= 0 || config.freeze_fraction > 1 ||
            config.warmup_iterations >= std::floor(config.total_iterations * config.freeze_fraction) ||
            config.visits_between_updates < 1 || config.steps_per_visit < 1 || config.steps_per_visit > 8)
            throw std::invalid_argument("Invalid camera pose session schedule");
        std::sort(cameras.begin(), cameras.end(), [](const auto& a, const auto& b) { return a.uid < b.uid; });
        std::vector<size_t> training, anchors;
        for (size_t i = 0; i < cameras.size(); ++i) {
            if (i && cameras[i - 1].uid == cameras[i].uid)
                throw std::invalid_argument("Duplicate camera UID in pose session");
            // Validate every source, including excluded cameras, before policy.
            (void)BoundedPoseOptimizer(cameras[i].uid, cameras[i].source, config.optimizer, cameras[i].role);
            if (cameras[i].role != PoseRole::Evaluation)
                training.push_back(i);
            if (cameras[i].role == PoseRole::Anchor)
                anchors.push_back(i);
        }
        if (training.size() < 3)
            throw std::invalid_argument("Pose refinement requires at least three training cameras");
        if (anchors.empty() && config.choose_anchors) {
            // Deterministic baseline from source poses, independent of input
            // order and evaluation views. This is a conservative gauge policy,
            // NOT a claim that the selected SfM observations are more accurate.
            const size_t first = training.front();
            size_t second = first;
            double farthest = 0;
            for (const auto i : training) {
                const double distance = squared_distance(cameras[first].source, cameras[i].source);
                if (distance > farthest) {
                    farthest = distance;
                    second = i;
                }
            }
            cameras[first].role = cameras[second].role = PoseRole::Anchor;
            anchors = {first, second};
        }
        if (anchors.size() < 2 || anchors.size() >= training.size())
            throw std::invalid_argument("Pose refinement requires two reference cameras and at least one movable camera");
        double baseline = 0;
        for (const auto i : anchors)
            baseline = std::max(baseline, squared_distance(cameras[anchors.front()].source, cameras[i].source));
        if (baseline <= std::pow(config.optimizer.scene_scale * 1e-6, 2))
            throw std::invalid_argument("Reference camera baseline is degenerate; scene scale is not constrained");
        for (const auto& camera : cameras) {
            index_.emplace(camera.uid, entries_.size());
            entries_.push_back({BoundedPoseOptimizer(camera.uid, camera.source, config.optimizer, camera.role), camera.role});
        }
        publish(true);
    }

    bool PoseRefinementSession::in_window() const noexcept {
        return iteration_ >= config_.warmup_iterations &&
               iteration_ < static_cast<int>(std::floor(config_.total_iterations * config_.freeze_fraction));
    }

    Matrix4 PoseRefinementSession::current_pose(int uid) const {
        return entries_.at(index_.at(uid)).optimizer.snapshot().current;
    }

    PoseVisitResult PoseRefinementSession::visit(int uid, int iteration, std::uint64_t model_revision,
                                                 const std::function<PoseImageEvaluation(const Matrix4&)>& evaluate,
                                                 const std::function<double(const Matrix4&)>& candidate_loss, std::stop_token stop,
                                                 const std::function<bool(const Matrix4&)>& candidate_allowed) {
        if (iteration < iteration_ || iteration > config_.total_iterations)
            throw std::invalid_argument("Camera pose iteration must be monotonic and within the training schedule");
        auto& entry = entries_.at(index_.at(uid));
        const int freeze_iteration = static_cast<int>(std::floor(config_.total_iterations * config_.freeze_fraction));
        const bool reached_freeze = iteration_ < freeze_iteration && iteration >= freeze_iteration;
        iteration_ = iteration;
        dirty_ = true;
        if (reached_freeze) {
            LOG_INFO("Camera pose refinement frozen at iteration {} (scheduled stop {}); retaining accepted poses for subsequent Gaussian training",
                     iteration, freeze_iteration);
        }
        PoseVisitResult result;
        if (paused_ || stop.stop_requested() || !in_window() || entry.role != PoseRole::Train) {
            result.cancelled = stop.stop_requested();
            publish(reached_freeze);
            return result;
        }
        if (!evaluate || !candidate_loss)
            throw std::invalid_argument("Pose refinement requires image evaluation callbacks");
        const auto next_visit = entry.visits + 1;
        // First eligible visit runs immediately, then cadence is PER CAMERA.
        // Sparse/random sampling must not starve a camera via a global modulo.
        if ((next_visit - 1) % static_cast<std::uint64_t>(config_.visits_between_updates) != 0) {
            entry.visits = next_visit;
            publish();
            return result;
        }
        result.scheduled = true;
        // Pose/history/cadence commit together. Iteration records the observed
        // training clock even on failure; cancelled work still counts renders.
        auto working = entry.optimizer;
        auto display = PoseDisplayState::Ready;
        for (int step = 0; step < config_.steps_per_visit; ++step) {
            if (stop.stop_requested()) {
                result.cancelled = true;
                break;
            }
            const auto pose = working.snapshot();
            const auto image = evaluate(pose.current);
            if (stop.stop_requested()) {
                result.cancelled = true;
                break;
            }
            const PoseEvaluation baseline{uid, model_revision, pose.revision, image.loss, image.gradient, image.geometric_proposal};
            const auto update = working.step(baseline, [&](const Matrix4& candidate) {
                if (stop.stop_requested())
                    return std::numeric_limits<double>::quiet_NaN();
                if (candidate_allowed && !candidate_allowed(candidate))
                    return std::numeric_limits<double>::infinity();
                ++result.candidate_renders;
                return candidate_loss(candidate);
            });
            if (stop.stop_requested()) {
                result.cancelled = true;
                break;
            }
            if (update.status == PoseStepStatus::Accepted) {
                ++result.accepted_steps;
                display = PoseDisplayState::Updated;
            } else {
                // A zero/tiny gradient is not enough to claim convergence.
                display = update.status == PoseStepStatus::NoDescent ? PoseDisplayState::Ready : PoseDisplayState::Rejected;
                break;
            }
        }
        if (result.cancelled) {
            result.accepted_steps = 0;
        } else {
            entry.optimizer = std::move(working);
            entry.state = display;
            entry.visits = next_visit;
        }
        entry.renders += result.candidate_renders;
        publish();
        return result;
    }

    void PoseRefinementSession::set_paused(bool paused) {
        paused_ = paused;
        dirty_ = true;
        publish(true);
    }

    void PoseRefinementSession::reset() {
        iteration_ = 0;
        paused_ = false;
        for (auto& entry : entries_) {
            entry.optimizer.reset();
            entry.visits = entry.renders = 0;
            entry.state = PoseDisplayState::Waiting;
        }
        dirty_ = true;
        publish(true);
    }

    std::shared_ptr<const PoseSessionSnapshot> PoseRefinementSession::make_snapshot(
        const std::vector<Entry>& entries, int iteration, bool paused, std::uint64_t sequence) const {
        auto snapshot = std::make_shared<PoseSessionSnapshot>();
        snapshot->generation = generation_;
        snapshot->sequence = sequence;
        snapshot->iteration = iteration;
        snapshot->stop_iteration = static_cast<int>(std::floor(config_.total_iterations * config_.freeze_fraction));
        snapshot->paused = paused;
        snapshot->refinement_finished = iteration >= static_cast<int>(std::floor(config_.total_iterations * config_.freeze_fraction));
        snapshot->cameras.reserve(entries.size());
        for (const auto& entry : entries) {
            auto state = entry.state;
            if (entry.role == PoseRole::Anchor)
                state = PoseDisplayState::Anchor;
            else if (entry.role == PoseRole::Evaluation)
                state = PoseDisplayState::Evaluation;
            else if (snapshot->refinement_finished)
                state = PoseDisplayState::Frozen;
            else if (iteration < config_.warmup_iterations)
                state = PoseDisplayState::Waiting;
            else if (state == PoseDisplayState::Waiting)
                state = PoseDisplayState::Ready;
            snapshot->cameras.push_back({entry.optimizer.snapshot(), state, entry.visits, entry.renders});
        }
        return snapshot;
    }

    void PoseRefinementSession::publish(bool force) {
        const auto now = std::chrono::steady_clock::now();
        if (!force && (!dirty_ || now < next_publish_))
            return;
        const auto snapshot = make_snapshot(entries_, iteration_, paused_, sequence_ + 1);
        ++sequence_;
        published_.store(snapshot, std::memory_order_release);
        dirty_ = false;
        next_publish_ = now + std::chrono::milliseconds(250);
    }

    std::shared_ptr<const PoseSessionSnapshot> PoseRefinementSession::published_snapshot() const noexcept {
        return published_.load(std::memory_order_acquire);
    }
} // namespace lfs::training::camera_pose
