/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "bounded_pose_optimizer.hpp"
#include "sparse_point_refinement.hpp"
#include <atomic>
#include <chrono>
#include <memory>
#include <nlohmann/json_fwd.hpp>
#include <stop_token>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace lfs::training::camera_pose {
    // Status is an observation of the optimizer, never a confidence score.
    enum class PoseDisplayState { Waiting,
                                  Ready,
                                  Updated,
                                  Rejected,
                                  Frozen,
                                  Anchor,
                                  Evaluation };
    [[nodiscard]] std::string_view pose_display_state_name(PoseDisplayState state) noexcept;

    struct PoseCameraInput {
        int uid = -1;
        Matrix4 source{};
        PoseRole role = PoseRole::Train;
    };

    struct PoseSessionConfig {
        int total_iterations = 30000;
        int warmup_iterations = 500;
        double freeze_fraction = 0.8;
        int visits_between_updates = 8;
        int steps_per_visit = 2;
        bool choose_anchors = true;
        // Zero selects the bounded photometric solver with geometric guards.
        // Production Trainer sessions use the combined objective below. Both
        // use the same state schema; resume never silently changes objectives.
        double joint_reprojection_weight = 0.0;
        static constexpr double DEFAULT_JOINT_REPROJECTION_WEIGHT = 1.0e-4;
        // scene_scale is explicit in scene units; caller must estimate it once
        // from source geometry, not from evolving optimized poses.
        BoundedPoseConfig optimizer{};
    };

    struct PoseCameraDisplay {
        PoseSnapshot pose;
        PoseDisplayState state = PoseDisplayState::Waiting;
        std::uint64_t eligible_visits = 0;
        std::uint64_t candidate_renders = 0;
    };

    [[nodiscard]] PoseSessionConfig pose_session_config_from_state(const nlohmann::json& state);

    struct PoseSessionSnapshot {
        std::uint64_t generation = 0; // Assigned by the owning training session.
        std::uint64_t sequence = 0;
        int iteration = 0;
        int stop_iteration = 0;
        bool paused = false;
        bool refinement_finished = false;
        std::vector<PoseCameraDisplay> cameras; // UID-sorted, immutable once published.
    };

    struct PoseImageEvaluation {
        double loss = 0;
        Twist gradient{};
        std::optional<Twist> geometric_proposal{};
    };

    struct PoseVisitResult {
        bool scheduled = false;
        bool cancelled = false;
        int accepted_steps = 0;
        int candidate_renders = 0;
    };

    // Training-thread wall-clock diagnostics, not GPU event timings. Counts
    // include attempted work discarded on cancellation; not checkpoint state.
    struct PoseDiagnostics {
        std::uint64_t visits = 0, baseline_evaluations = 0, candidate_checks = 0;
        std::uint64_t point_solves = 0, point_proposals = 0;
        std::uint64_t fixed_rejections = 0, joint_rejections = 0;
        std::uint64_t invalid_losses = 0, image_rejections = 0, objective_rejections = 0;
        std::uint64_t combined_rejections = 0;
        std::uint64_t candidate_renders = 0, accepted_candidates = 0, committed_steps = 0;
        std::uint64_t no_descent = 0, exceptions = 0, cancellations = 0;
        double visit_ms = 0, point_ms = 0, baseline_ms = 0, candidate_ms = 0, proposal_ms = 0;
    };

    // Multi-camera owner, intended to be called at a Trainer safe point.
    // Mutating methods have one training-thread owner; published_snapshot()
    // is the only cross-thread API. It performs no GPU work and acquires no
    // training/render mutex. Viewport and Scene Graph consume the SAME snapshot.
    // This component does not mutate Camera, select backends or save projects.
    class PoseRefinementSession {
    public:
        PoseRefinementSession(std::uint64_t generation, std::vector<PoseCameraInput> cameras,
                              PoseSessionConfig config = {});
        [[nodiscard]] Matrix4 current_pose(int uid) const;
        void configure_sparse_points(std::vector<SparseTrackMeasurement> measurements);
        [[nodiscard]] bool joint_geometry_enabled(int uid) const noexcept { return tracks_by_camera_.contains(uid); }
        [[nodiscard]] size_t shared_point_count() const noexcept { return sparse_tracks_.size(); }
        [[nodiscard]] PoseDiagnostics diagnostics() const noexcept { return diagnostics_; }
        // Optional fixed-geometry predicate runs before candidate rendering.
        // A false result rejects the proposal without consuming a render count.
        [[nodiscard]] PoseVisitResult visit(
            int uid, int iteration, std::uint64_t model_revision,
            const std::function<PoseImageEvaluation(const Matrix4&)>& evaluate,
            const std::function<double(const Matrix4&)>& candidate_loss,
            std::stop_token stop = {},
            const std::function<bool(const Matrix4&)>& candidate_allowed = {});
        void set_paused(bool paused);
        void reset();
        // Training-thread only, independent of throttled display publication.
        // Restore requires identical settings, source poses, UIDs and roles.
        [[nodiscard]] nlohmann::json save_state() const;
        void restore_state(const nlohmann::json& state);
        void publish(bool force = false);
        [[nodiscard]] std::shared_ptr<const PoseSessionSnapshot> published_snapshot() const noexcept;

    private:
        struct Entry {
            BoundedPoseOptimizer optimizer;
            PoseRole role;
            PoseDisplayState state = PoseDisplayState::Waiting;
            std::uint64_t visits = 0;
            std::uint64_t renders = 0;
        };
        [[nodiscard]] bool in_window() const noexcept;
        PoseVisitResult visit_joint(int uid, const std::function<PoseImageEvaluation(const Matrix4&)>& evaluate,
                                    std::stop_token stop, const std::function<bool(const Matrix4&)>& candidate_allowed);
        void log_diagnostics() const;
        [[nodiscard]] std::shared_ptr<const PoseSessionSnapshot> make_snapshot(
            const std::vector<Entry>& entries, int iteration, bool paused, std::uint64_t sequence) const;
        PoseSessionConfig config_;
        std::uint64_t generation_;
        std::uint64_t sequence_ = 0;
        int iteration_ = 0;
        bool paused_ = false;
        bool dirty_ = true;
        std::vector<Entry> entries_;
        std::unordered_map<int, size_t> index_;
        std::vector<SparsePointTrack> sparse_tracks_;
        std::vector<SparsePointPosition> sparse_positions_;
        std::vector<SparsePointAdam> sparse_adam_;
        std::unordered_map<int, std::vector<size_t>> tracks_by_camera_;
        PoseDiagnostics diagnostics_;
        std::chrono::steady_clock::time_point next_publish_{};
        std::atomic<std::shared_ptr<const PoseSessionSnapshot>> published_{};
    };
} // namespace lfs::training::camera_pose
