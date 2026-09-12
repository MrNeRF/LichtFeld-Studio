/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "fastgs_pose_evaluator.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace lfs::training::camera_pose {
    using namespace lfs::core;

    FastGSPoseEvaluator::FastGSPoseEvaluator(Camera& camera, SplatData& model,
                                             AdamOptimizer& optimizer, Tensor& background, PoseObjective objective, Tensor background_image)
        : camera_(camera), model_(model), optimizer_(optimizer), background_(background), objective_(std::move(objective)), background_image_(std::move(background_image)) {
        if (!objective_)
            throw std::invalid_argument("Missing camera pose image objective");
    }

    std::pair<RenderOutput, FastRasterizeContext> FastGSPoseEvaluator::forward(const Matrix4& pose) {
        // Reuse the same rigid-source validation as the controller. Do not
        // silently repair arbitrary matrices or mutate source Camera tensors.
        (void)BoundedPoseOptimizer(camera_.uid(), pose, BoundedPoseConfig{});
        std::vector<float> center(3, 0);
        for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 3; ++i)
                center[j] -= pose[4 * i + j] * pose[4 * i + 3];
        FastGSCameraPoseOverride tensors{
            Tensor::from_vector(std::vector<float>(pose.begin(), pose.end()), {1, 4, 4}, Device::CUDA),
            Tensor::from_vector(center, {3}, Device::CUDA)};
        auto result = fast_rasterize_forward(camera_, model_, background_, 0, 0, 0, 0,
                                             false, background_image_, false, &tensors);
        if (!result)
            throw lfs::Exception(std::move(result.error()));
        return std::move(*result);
    }

    PoseImageEvaluation FastGSPoseEvaluator::evaluate(const Matrix4& pose) {
        auto rendered = forward(pose);
        const auto objective = objective_(rendered.first, true);
        if (!std::isfinite(objective.loss) || objective.loss < 0)
            throw std::invalid_argument("Nonfinite or negative camera pose baseline loss");
        const auto valid_gradient = [&](const Tensor& gradient, const Tensor& reference) {
            return gradient.is_valid() && gradient.device() == Device::CUDA &&
                   gradient.dtype() == DataType::Float32 && gradient.is_contiguous() &&
                   gradient.shape() == reference.shape() && gradient.stream() == reference.stream();
        };
        if (!valid_gradient(objective.grad_image, rendered.first.image) ||
            (objective.grad_alpha.is_valid() && !valid_gradient(objective.grad_alpha, rendered.first.alpha)))
            throw std::invalid_argument("Pose objective gradients must match raw raster output shape, dtype and stream");
        // Allocate disjoint caller-owned storage. Never alias Gaussian/Adam
        // scratch, even when forward uses arena-backed temporary buffers.
        auto matrix_gradient = Tensor::zeros({4, 4}, Device::CUDA);
        fast_rasterize_backward(rendered.second, objective.grad_image, model_, optimizer_,
                                objective.grad_alpha, {}, DensificationType::None, 0,
                                {}, {}, {}, &matrix_gradient, FastGSBackwardMode::CameraOnly);
        const auto cpu = matrix_gradient.to(Device::CPU).contiguous();
        Matrix4 gradient{};
        std::copy_n(cpu.ptr<float>(), gradient.size(), gradient.begin());
        return {objective.loss, left_increment_gradient(pose, gradient)};
    }

    double FastGSPoseEvaluator::loss(const Matrix4& pose) {
        auto rendered = forward(pose);
        // Context RAII releases forward scratch on success and exceptions.
        return objective_(rendered.first, false).loss;
    }

    PoseVisitResult FastGSPoseEvaluator::visit(PoseRefinementSession& session, int iteration,
                                               std::uint64_t model_revision, std::stop_token stop) {
        return session.visit(camera_.uid(), iteration, model_revision, [this](const Matrix4& pose) { return evaluate(pose); }, [this](const Matrix4& pose) { return loss(pose); }, stop);
    }

    PoseObjective make_pose_mse_objective(const Tensor& target) {
        if (!target.is_valid() || target.device() != Device::CUDA || target.dtype() != DataType::Float32 ||
            target.ndim() != 3 || target.shape()[0] != 3 || target.numel() == 0 || !target.is_contiguous())
            throw std::invalid_argument("Pose MSE target must be contiguous CUDA float32 [3,H,W]");
        return [fixed_target = target.clone()](const RenderOutput& output, bool gradients) {
            if (output.image.shape() != fixed_target.shape() || output.image.stream() != fixed_target.stream())
                throw std::invalid_argument("Pose MSE target shape or CUDA stream differs from render");
            const Tensor residual = output.image - fixed_target;
            const Tensor squared = residual * residual;
            PoseObjectiveResult result;
            result.loss = squared.mean().item<float>();
            if (gradients)
                result.grad_image = residual * (2.0f / static_cast<float>(fixed_target.numel()));
            return result;
        };
    }
} // namespace lfs::training::camera_pose
