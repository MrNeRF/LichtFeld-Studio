/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "pose_refinement_session.hpp"
#include "core/logger.hpp"
#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <stdexcept>
#include <utility>

namespace lfs::training::camera_pose {
    namespace {
        struct WallTimer {
            double& milliseconds;
            std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
            ~WallTimer() noexcept {
                milliseconds += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            }
        };
        struct VisitAudit {
            PoseDiagnostics& diagnostics;
            std::stop_token stop;
            int exceptions = std::uncaught_exceptions();
            ~VisitAudit() noexcept {
                diagnostics.exceptions += std::uncaught_exceptions() > exceptions;
                diagnostics.cancellations += stop.stop_requested();
            }
        };
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

    void PoseRefinementSession::configure_sparse_points(std::vector<SparseTrackMeasurement> measurements) {
        if (iteration_ != 0 || !sparse_tracks_.empty())
            throw std::invalid_argument("Configure shared points before starting or restoring the session");
        std::erase_if(measurements, [&](const auto& m) {
            const auto found = index_.find(m.camera_uid);
            return !m.training || found == index_.end() || entries_[found->second].role == PoseRole::Evaluation;
        });
        for (const auto& m : measurements)
            if (m.pose != entries_[index_.at(m.camera_uid)].optimizer.snapshot().source)
                throw std::invalid_argument("Shared point source pose differs from session source");
        auto tracks = build_sparse_point_tracks(measurements);
        std::erase_if(tracks, [](const auto& track) {
            if (!std::isfinite(sparse_point_cost(track, track.source)))
                return true;
            return std::any_of(track.measurements.begin(), track.measurements.end(), [](const auto& m) {
                return !std::isfinite(m.u) || !std::isfinite(m.v) || m.u < 0 || m.v < 0 ||
                       m.u >= m.calibration.width || m.v >= m.calibration.height;
            });
        });
        std::unordered_map<int, std::vector<size_t>> by_camera;
        std::vector<SparsePointPosition> positions;
        for (size_t i = 0; i < tracks.size(); ++i) {
            positions.push_back(tracks[i].source);
            for (const auto& m : tracks[i].measurements)
                by_camera[m.camera_uid].push_back(i);
        }
        std::erase_if(by_camera, [&](const auto& item) {
            std::vector<ReprojectionObservation> observations;
            ReprojectionCalibration k{};
            for (const auto i : item.second) {
                const auto& track = tracks[i];
                const auto m = std::find_if(track.measurements.begin(), track.measurements.end(),
                                            [&](const auto& m) { return m.camera_uid == item.first; });
                k = m->calibration;
                observations.push_back({m->u, m->v, track.source[0], track.source[1], track.source[2]});
            }
            return !SparseReprojectionGuard(current_pose(item.first), k, observations).active();
        });
        sparse_tracks_ = std::move(tracks);
        sparse_positions_ = std::move(positions);
        tracks_by_camera_ = std::move(by_camera);
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
            log_diagnostics();
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
        if (diagnostics_.visits && diagnostics_.visits % 64 == 0)
            log_diagnostics();
        ++diagnostics_.visits;
        const WallTimer visit_timer{diagnostics_.visit_ms};
        const VisitAudit visit_audit{diagnostics_, stop};
        // Pose/history/cadence commit together. Iteration records the observed
        // training clock even on failure; cancelled work still counts renders.
        auto working = entry.optimizer;
        auto display = PoseDisplayState::Ready;
        const auto joint = tracks_by_camera_.find(uid);
        std::vector<SparsePointTrack> joint_tracks;
        std::vector<SparsePointPosition> points;
        if (joint != tracks_by_camera_.end()) {
            for (const auto i : joint->second) {
                joint_tracks.push_back(sparse_tracks_[i]);
                points.push_back(sparse_positions_[i]);
                for (auto& m : joint_tracks.back().measurements)
                    m.pose = current_pose(m.camera_uid);
            }
        }
        const double point_limit = config_.optimizer.scene_scale * config_.optimizer.max_center_fraction;
        auto relax_points = [&](const Matrix4& pose, std::vector<SparsePointPosition>& positions) {
            const WallTimer timer{diagnostics_.point_ms};
            double cost = 0;
            for (size_t i = 0; i < joint_tracks.size(); ++i) {
                if (stop.stop_requested())
                    return std::numeric_limits<double>::infinity();
                auto& track = joint_tracks[i];
                for (auto& m : track.measurements)
                    if (m.camera_uid == uid)
                        m.pose = pose;
                ++diagnostics_.point_solves;
                if (const auto proposal = propose_sparse_point(track, point_limit, 2.0, positions[i])) {
                    ++diagnostics_.point_proposals;
                    positions[i] = proposal->position;
                }
                cost += sparse_point_cost(track, positions[i]);
            }
            return cost;
        };
        for (int step = 0; step < config_.steps_per_visit; ++step) {
            if (stop.stop_requested()) {
                result.cancelled = true;
                break;
            }
            const auto pose = working.snapshot();
            // Baseline and candidates start from identical points and receive
            // the same solve budget. Extra point iterations must not buy an
            // otherwise geometrically worse camera update.
            const auto initial_points = points;
            const double geometry_baseline = relax_points(pose.current, points);
            if (stop.stop_requested()) {
                result.cancelled = true;
                break;
            }
            if (!std::isfinite(geometry_baseline))
                throw std::runtime_error("Shared point baseline is not projectable");
            auto image = [&] {
                const WallTimer timer{diagnostics_.baseline_ms};
                ++diagnostics_.baseline_evaluations;
                return evaluate(pose.current);
            }();
            if (!joint_tracks.empty()) {
                const WallTimer timer{diagnostics_.proposal_ms};
                std::vector<ReprojectionObservation> observations;
                ReprojectionCalibration k{};
                for (size_t i = 0; i < joint_tracks.size(); ++i) {
                    const auto& track = joint_tracks[i];
                    const auto m = std::find_if(track.measurements.begin(), track.measurements.end(),
                                                [&](const auto& m) { return m.camera_uid == uid; });
                    k = m->calibration;
                    observations.push_back({m->u, m->v, points[i][0], points[i][1], points[i][2]});
                }
                image.geometric_proposal = SparseReprojectionGuard(pose.current, k, observations).proposal(pose.current, config_.optimizer.scene_scale);
            }
            if (stop.stop_requested()) {
                result.cancelled = true;
                break;
            }
            const PoseEvaluation baseline{uid, model_revision, pose.revision, image.loss, image.gradient, image.geometric_proposal};
            auto candidate_points = points;
            std::uint64_t passed_image = 0;
            const auto update = working.step(baseline, [&](const Matrix4& candidate) {
                if (stop.stop_requested())
                    return std::numeric_limits<double>::quiet_NaN();
                ++diagnostics_.candidate_checks;
                if (candidate_allowed && !candidate_allowed(candidate)) {
                    ++diagnostics_.fixed_rejections;
                    return std::numeric_limits<double>::infinity();
                }
                if (!joint_tracks.empty()) {
                    candidate_points = initial_points;
                    const double cost = relax_points(candidate, candidate_points);
                    if (!std::isfinite(cost) || !std::isfinite(geometry_baseline) ||
                        cost > geometry_baseline + 1e-12 * joint_tracks.size()) {
                        if (!stop.stop_requested())
                            ++diagnostics_.joint_rejections;
                        return std::numeric_limits<double>::infinity();
                    }
                }
                ++result.candidate_renders;
                ++diagnostics_.candidate_renders;
                const double loss = [&] {
                    const WallTimer timer{diagnostics_.candidate_ms};
                    return candidate_loss(candidate);
                }();
                if (!std::isfinite(loss) || loss < 0)
                    ++diagnostics_.invalid_losses;
                else if (!(loss < baseline.image_loss * (1 - config_.optimizer.min_relative_improvement)))
                    ++diagnostics_.image_rejections;
                else
                    ++passed_image;
                return loss;
            });
            const bool accepted = update.status == PoseStepStatus::Accepted;
            diagnostics_.accepted_candidates += accepted;
            diagnostics_.objective_rejections += passed_image - static_cast<std::uint64_t>(accepted);
            diagnostics_.no_descent += update.status == PoseStepStatus::NoDescent;
            if (stop.stop_requested()) {
                result.cancelled = true;
                break;
            }
            if (update.status == PoseStepStatus::Accepted) {
                points = std::move(candidate_points);
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
            // No allocations after this point: pose and shared positions commit
            // together. Exceptions/cancellation above leave both live states intact.
            if (joint != tracks_by_camera_.end())
                for (size_t i = 0; i < points.size(); ++i)
                    sparse_positions_[joint->second[i]] = points[i];
            entry.optimizer = std::move(working);
            entry.state = display;
            entry.visits = next_visit;
            diagnostics_.committed_steps += result.accepted_steps;
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
        diagnostics_ = {};
        iteration_ = 0;
        paused_ = false;
        for (size_t i = 0; i < sparse_tracks_.size(); ++i)
            sparse_positions_[i] = sparse_tracks_[i].source;
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

    void PoseRefinementSession::log_diagnostics() const {
        const auto& d = diagnostics_;
        LOG_INFO("Camera pose diagnostics (since start/restore): visits={} baselines={} checks={} renders={} fixed_reject={} joint_reject={} invalid_loss={} image_reject={} objective_reject={} accepted_candidates={} committed_steps={} no_descent={} exceptions={} cancellations={} point_solves={} point_proposals={}",
                 d.visits, d.baseline_evaluations, d.candidate_checks, d.candidate_renders, d.fixed_rejections, d.joint_rejections,
                 d.invalid_losses, d.image_rejections, d.objective_rejections, d.accepted_candidates, d.committed_steps,
                 d.no_descent, d.exceptions, d.cancellations, d.point_solves, d.point_proposals);
        LOG_INFO("Camera pose wall time (ms, since start/restore): visits={} points={} baseline={} candidates={} proposal={} other={}; host elapsed including callback waits, not GPU event timing",
                 d.visit_ms, d.point_ms, d.baseline_ms, d.candidate_ms, d.proposal_ms,
                 std::max(0.0, d.visit_ms - d.point_ms - d.baseline_ms - d.candidate_ms - d.proposal_ms));
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
