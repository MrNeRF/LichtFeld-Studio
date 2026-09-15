/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "se3.hpp"
#include <cstdint>
#include <functional>
#include <optional>

namespace lfs::training::camera_pose {

    enum class PoseRole { Train,
                          Anchor,
                          Evaluation };
    enum class PoseStepStatus { Accepted,
                                Frozen,
                                Stale,
                                InvalidInput,
                                NoDescent,
                                Rejected };

    struct BoundedPoseConfig {
        double scene_scale = 1.0;
        double max_center_fraction = 0.03;
        double max_rotation_radians = 0.0523598776; // 3 degrees, cumulative
        double step_center_fraction = 0.003;
        double step_rotation_radians = 0.00436332313; // 0.25 degrees per step
        double center_prior = 1.0e-6;
        double rotation_prior = 1.0e-6;
        double min_relative_improvement = 1.0e-6;
        int max_backtracks = 8;
    };

    struct PoseSnapshot {
        int uid = -1;
        Matrix4 source{};
        Matrix4 current{};
        std::uint64_t revision = 0;
        std::uint64_t accepted_steps = 0;
        std::uint64_t rejected_steps = 0;
        double center_displacement = 0.0;
        double rotation_displacement = 0.0;
        std::array<double, 6> first_moment{};
        std::array<double, 6> second_moment{};
        std::uint64_t adaptive_steps = 0;
    };

    struct PoseEvaluation {
        int uid = -1;
        // Caller must invalidate model_revision whenever Gaussian geometry,
        // appearance, mask, resolution, background or the loss objective changes.
        // Baseline gradient and every candidate loss use the SAME frozen model.
        std::uint64_t model_revision = 0;
        std::uint64_t pose_revision = 0;
        // Scalar data objective and its matching gradient. Normally photometric;
        // combined sessions supply image + reprojection (source prior is internal).
        double image_loss = 0.0;
        Twist image_gradient{};                    // Fresh left-tangent gradient, not matrix gradient
        std::optional<Twist> geometric_proposal{}; // World-unit left increment; never an acceptance override.
        bool adaptive = false;                     // Small persistent Adam direction; same nonlinear acceptance.
    };

    struct PoseStepResult {
        PoseStepStatus status = PoseStepStatus::Rejected;
        int evaluations = 0;
        double image_loss = 0.0;
    };

    // Single-training-thread, per-UID controller. No CUDA pointers or Camera
    // mutation. A copy of snapshot() can be published to readers; this class
    // itself is NOT thread safe. Callers own anchoring of world-frame/scale,
    // dataset membership, persistence and generation-correct snapshot delivery.
    class BoundedPoseOptimizer {
    public:
        BoundedPoseOptimizer(int uid, const Matrix4& source, BoundedPoseConfig config,
                             PoseRole role = PoseRole::Train);
        [[nodiscard]] PoseSnapshot snapshot() const noexcept;
        void set_frozen(bool frozen) noexcept;
        void reset() noexcept;
        // Restore durable pose/counters, invalidate old evaluations and restart
        // curvature history. Source and role belong to the current dataset.
        void restore(const PoseSnapshot& saved);

        // Joint stochastic update: no per-image monotonic line search. The
        // session commits all involved cameras/points as one transaction after
        // checking finite values, projection and source-relative safety bounds.
        [[nodiscard]] bool joint_step(const std::array<double, 6>& gradient);

        // Candidate callback must only render/evaluate; it must not update
        // Gaussian/appearance optimizers. At most max_backtracks callbacks.
        // Exceptions propagate with the entire controller state unchanged.
        [[nodiscard]] PoseStepResult step(
            const PoseEvaluation& baseline,
            const std::function<double(const Matrix4&)>& candidate_loss);

    private:
        using Vector = std::array<double, 6>;
        using Hessian = std::array<Vector, 6>;
        PoseStepResult step_impl(const PoseEvaluation&, const std::function<double(const Matrix4&)>&);
        double prior(const Matrix4&, Vector* tangent_gradient = nullptr) const noexcept;
        void clear_history() noexcept;
        BoundedPoseConfig config_;
        PoseRole role_;
        PoseSnapshot state_;
        bool frozen_ = false;
        bool history_valid_ = false;
        std::uint64_t model_revision_ = 0;
        Hessian inverse_hessian_{};
        Vector previous_gradient_{};
        Vector previous_step_{};
    };
} // namespace lfs::training::camera_pose
