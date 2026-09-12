/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "pose_refinement_session.hpp"
#include "rasterization/fast_rasterizer.hpp"
#include <functional>

namespace lfs::training::camera_pose {
    struct PoseObjectiveResult {
        double loss = 0;
        lfs::core::Tensor grad_image;
        lfs::core::Tensor grad_alpha;
    };

    // The callback must evaluate the SAME fixed image objective with/without
    // gradients. Gradients refer to raw raster RGB/alpha, not a postprocessed
    // image. It must not update appearance parameters, model or camera.
    using PoseObjective = std::function<PoseObjectiveResult(const RenderOutput&, bool gradients)>;

    // Training-thread adapter for full-image FastGS, no Mip/normal/depth loss.
    // References and objective captures must outlive the adapter. Owner holds
    // the normal training safe point and keeps geometry/SH/background/target
    // immutable for a whole visit. This object does not acquire model locks.
    class FastGSPoseEvaluator {
    public:
        FastGSPoseEvaluator(lfs::core::Camera& camera, lfs::core::SplatData& model,
                            AdamOptimizer& optimizer, lfs::core::Tensor& background,
                            PoseObjective objective, lfs::core::Tensor background_image = {});
        [[nodiscard]] PoseImageEvaluation evaluate(const Matrix4& pose);
        [[nodiscard]] double loss(const Matrix4& pose);
        [[nodiscard]] PoseVisitResult visit(PoseRefinementSession& session, int iteration,
                                            std::uint64_t model_revision, std::stop_token stop = {});

    private:
        [[nodiscard]] std::pair<RenderOutput, FastRasterizeContext> forward(const Matrix4& pose);
        lfs::core::Camera& camera_;
        lfs::core::SplatData& model_;
        AdamOptimizer& optimizer_;
        lfs::core::Tensor& background_;
        PoseObjective objective_;
        lfs::core::Tensor background_image_;
    };

    // RGB MSE reference objective for the fixed-geometry quality gate.
    // GPU reduction, scalar readback only; clone target once per visit owner,
    // not per candidate. Does not replace the normal training loss implicitly.
    [[nodiscard]] PoseObjective make_pose_mse_objective(const lfs::core::Tensor& target);
} // namespace lfs::training::camera_pose
