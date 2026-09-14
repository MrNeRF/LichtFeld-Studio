/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "fastgs_pose_evaluator.hpp"
#include "core/gpu_backend_fwd.hpp"
#include "losses/photometric_loss.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace lfs::training::camera_pose {
    using namespace lfs::core;

    FastGSPoseEvaluator::FastGSPoseEvaluator(Camera& camera, SplatData& model,
                                             AdamOptimizer& optimizer, Tensor& background, PoseObjective objective, Tensor background_image, bool mip_filter,
                                             const PoseRefinementSession* session)
        : camera_(camera), model_(model), optimizer_(optimizer), background_(background), objective_(std::move(objective)), background_image_(std::move(background_image)), mip_filter_(mip_filter), joint_geometry_(session && session->joint_geometry_enabled(camera.uid())) {
        if (!objective_)
            throw std::invalid_argument("Missing camera pose image objective");
        if (!joint_geometry_)
            reprojection_guard_ = make_sparse_reprojection_guard(camera);
    }

    SparseReprojectionGuard make_sparse_reprojection_guard(const Camera& camera) {
        const bool rectified = camera.is_undistort_prepared();
        if ((!rectified && (camera.camera_model_type() != CameraModelType::PINHOLE || camera.has_distortion())) ||
            camera.sfm_observations().empty())
            return {};
        const auto source_tensor = camera.world_view_transform().to(Device::CPU).contiguous();
        if (source_tensor.dtype() != DataType::Float32 || source_tensor.numel() != 16)
            return {};
        Matrix4 source;
        std::copy_n(source_tensor.ptr<float>(), source.size(), source.begin());
        std::vector<ReprojectionObservation> observations;
        observations.reserve(camera.sfm_observations().size());
        for (const auto& point : camera.sfm_observations()) {
            float u = point.u, v = point.v;
            // Keep original measurements intact, including for checkpoint reload
            // and repeated evaluator construction. No pose-derived pseudo-targets.
            if (rectified && !undistort_observation(camera.undistort_params(), point.u, point.v, u, v))
                continue;
            observations.push_back({u, v, point.x, point.y, point.z});
        }
        // Native calibration matches images_N-scaled SfM pixels and does not
        // depend on lazy image loading, training resize or a tile offset.
        return SparseReprojectionGuard(source,
                                       {camera.focal_x(), camera.focal_y(), camera.center_x(), camera.center_y(),
                                        camera.camera_width(), camera.camera_height()},
                                       observations);
    }

    std::vector<SparseTrackMeasurement> make_sparse_track_measurements(const Camera& camera, bool training_member) {
        if (!training_member || camera.split() == CameraSplit::Eval)
            return {};
        const bool rectified = camera.is_undistort_prepared();
        if (!rectified && (camera.camera_model_type() != CameraModelType::PINHOLE || camera.has_distortion()))
            return {};
        const auto tensor = camera.world_view_transform().to(Device::CPU).contiguous();
        if (tensor.dtype() != DataType::Float32 || tensor.numel() != 16)
            return {};
        Matrix4 pose;
        std::copy_n(tensor.ptr<float>(), pose.size(), pose.begin());
        const ReprojectionCalibration k{camera.focal_x(), camera.focal_y(), camera.center_x(), camera.center_y(),
                                        camera.camera_width(), camera.camera_height()};
        std::vector<SparseTrackMeasurement> result;
        result.reserve(camera.sfm_observations().size());
        for (const auto& observation : camera.sfm_observations()) {
            if (observation.point3d_id == std::numeric_limits<std::uint64_t>::max())
                continue;
            float u = observation.u, v = observation.v;
            if (rectified && !undistort_observation(camera.undistort_params(), u, v, u, v))
                continue;
            result.push_back({observation.point3d_id, {observation.x, observation.y, observation.z}, camera.uid(), true, pose, k, u, v});
        }
        return result;
    }

    FastGSCameraPoseOverride make_fastgs_pose_override(int uid, const Matrix4& pose) {
        const GpuBackendScope backend_scope(GpuBackend::CUDA);
        // Reuse the same rigid-source validation as the controller. Do not
        // silently repair arbitrary matrices or mutate source Camera tensors.
        (void)BoundedPoseOptimizer(uid, pose, BoundedPoseConfig{});
        std::vector<float> center(3, 0);
        for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 3; ++i)
                center[j] -= pose[4 * i + j] * pose[4 * i + 3];
        return {
            Tensor::from_vector(std::vector<float>(pose.begin(), pose.end()), {1, 4, 4}, Device::CUDA),
            Tensor::from_vector(center, {3}, Device::CUDA)};
    }

    std::pair<RenderOutput, FastRasterizeContext> FastGSPoseEvaluator::forward(const Matrix4& pose) {
        const GpuBackendScope backend_scope(GpuBackend::CUDA);
        auto tensors = make_fastgs_pose_override(camera_.uid(), pose);
        auto result = fast_rasterize_forward(camera_, model_, background_, 0, 0, 0, 0,
                                             mip_filter_, background_image_, false, &tensors);
        if (!result)
            throw lfs::Exception(std::move(result.error()));
        return std::move(*result);
    }

    PoseImageEvaluation FastGSPoseEvaluator::evaluate(const Matrix4& pose) {
        const GpuBackendScope backend_scope(GpuBackend::CUDA);
        auto rendered = forward(pose);
        const auto objective = objective_(rendered.first, true);
        if (!std::isfinite(objective.loss) || objective.loss < 0)
            throw std::invalid_argument("Nonfinite or negative camera pose baseline loss");
        const auto valid_gradient = [&](const Tensor& gradient, const Tensor& reference) {
            return gradient.is_valid() && gpu_backend_of(gradient) == GpuBackend::CUDA &&
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
        return {objective.loss, left_increment_gradient(pose, gradient),
                joint_geometry_ ? std::nullopt : reprojection_guard_.proposal(pose, model_.get_scene_scale())};
    }

    double FastGSPoseEvaluator::loss(const Matrix4& pose) {
        const GpuBackendScope backend_scope(GpuBackend::CUDA);
        if (!allows(pose))
            return std::numeric_limits<double>::infinity();
        auto rendered = forward(pose);
        // Context RAII releases forward scratch on success and exceptions.
        return objective_(rendered.first, false).loss;
    }

    PoseVisitResult FastGSPoseEvaluator::visit(PoseRefinementSession& session, int iteration,
                                               std::uint64_t model_revision, std::stop_token stop) {
        // Also support adapters constructed before the session was supplied.
        joint_geometry_ = session.joint_geometry_enabled(camera_.uid());
        if (!joint_geometry_)
            reprojection_guard_ = make_sparse_reprojection_guard(camera_);
        return session.visit(camera_.uid(), iteration, model_revision, [this](const Matrix4& pose) { return evaluate(pose); }, [this](const Matrix4& pose) { return loss(pose); }, stop, [this](const Matrix4& pose) { return allows(pose); });
    }

    PoseObjective make_pose_photometric_objective(const Tensor& target, float lambda_dssim) {
        if (!std::isfinite(lambda_dssim) || lambda_dssim < 0 || lambda_dssim > 1 ||
            !target.is_valid() || gpu_backend_of(target) != GpuBackend::CUDA || target.ndim() != 3 ||
            target.shape()[0] != 3 || target.numel() == 0 ||
            (target.dtype() != DataType::Float32 && target.dtype() != DataType::UInt8))
            throw std::invalid_argument("Invalid pose photometric target or SSIM weight");
        return [fixed_target = target.clone(), lambda_dssim,
                loss = std::make_shared<losses::PhotometricLoss>()](const RenderOutput& output, bool gradients) {
            if (output.image.stream() != fixed_target.stream())
                throw std::invalid_argument("Pose photometric target stream differs from render");
            auto result = loss->forward(output.image, fixed_target, {lambda_dssim});
            if (!result)
                throw std::runtime_error(result.error());
            const float value = result->first.item<float>();
            // SSIM can exceed one by float roundoff for identical images.
            // Rectify only that narrow numerical range, with the matching zero
            // derivative. Nonfinite/substantially negative objectives still fail
            // evaluator validation; ordinary Gaussian training is unaffected.
            const float tolerance = 32 * std::numeric_limits<float>::epsilon() * lambda_dssim;
            if (std::isfinite(value) && value < 0 && value >= -tolerance)
                return PoseObjectiveResult{0.0, gradients ? Tensor::zeros_like(result->second.grad_image) : Tensor{}, {}};
            // The existing loss API also computes its gradient for candidates;
            // camera/Gaussian backward is still omitted for candidate scoring.
            return PoseObjectiveResult{value,
                                       gradients ? result->second.grad_image : Tensor{},
                                       {}};
        };
    }

    PoseObjective make_pose_mse_objective(const Tensor& target) {
        if (!target.is_valid() || gpu_backend_of(target) != GpuBackend::CUDA || target.dtype() != DataType::Float32 ||
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
