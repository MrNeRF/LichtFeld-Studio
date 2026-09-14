/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "trainer_pose_integration.hpp"
#include "core/logger.hpp"
#include "fastgs_pose_evaluator.hpp"
#include "trainer.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <nlohmann/json.hpp>
#include <unordered_set>

namespace lfs::training::camera_pose {
    std::string trainer_pose_incompatibility(const lfs::core::param::OptimizationParameters& params) {
        return params.camera_pose_incompatibility();
    }
} // namespace lfs::training::camera_pose

namespace lfs::training {
    namespace {
        lfs::Error pose_initialization_error(std::string detail, lfs::core::SourceSite source) {
            return lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::FailedPrecondition,
                .domain = lfs::ErrorDomain::Training,
                .user_message = "Camera pose refinement could not be initialized.",
                .detail = std::move(detail),
                .detection = source,
            });
        }
    } // namespace

    lfs::Status Trainer::configureCameraPoseRefinement(
        std::optional<camera_pose::PoseSessionConfig> config) {
        std::lock_guard lock(init_mutex_);
        if (initialized_.load() || is_running_.load())
            return lfs::Status::failure(pose_initialization_error(
                "Configure camera pose refinement before Trainer initialization", LFS_SOURCE_SITE_CURRENT()));
        camera_pose_config_ = std::move(config);
        return {};
    }

    std::shared_ptr<const camera_pose::PoseSessionSnapshot> Trainer::cameraPoseSnapshot() const {
        const auto session = camera_pose_session_.load(std::memory_order_acquire);
        return session ? session->published_snapshot() : nullptr;
    }

    lfs::Status Trainer::initialize_camera_pose_refinement(
        const std::vector<std::shared_ptr<lfs::core::Camera>>& cameras) {
        camera_pose_sources_ = cameras;
        auto session = make_camera_pose_session(params_, strategy_->get_model());
        if (!session)
            return lfs::Status::failure(std::move(session).error());
        camera_pose_last_visit_iteration_ = session.value() ? session.value()->published_snapshot()->iteration : -1;
        camera_pose_total_iterations_ = get_total_iterations();
        camera_pose_session_.store(std::move(session).value(), std::memory_order_release);
        return {};
    }

    lfs::Result<std::shared_ptr<camera_pose::PoseRefinementSession>> Trainer::make_camera_pose_session(
        const lfs::core::param::TrainingParameters& params, const lfs::core::SplatData& model) const try {
        using namespace camera_pose;
        using namespace lfs::core;
        if (!camera_pose_config_ && !params.optimization.refine_camera_poses && params.camera_pose_state_json.empty())
            return std::shared_ptr<PoseRefinementSession>{};
        if (auto error = trainer_pose_incompatibility(params.optimization); !error.empty())
            return pose_initialization_error(std::move(error), LFS_SOURCE_SITE_CURRENT());
        if (!train_dataset_ || camera_pose_sources_.empty())
            return pose_initialization_error("Camera pose refinement requires an initialized dataset", LFS_SOURCE_SITE_CURRENT());
        const auto saved = params.camera_pose_state_json.empty() ? nlohmann::json{} : nlohmann::json::parse(params.camera_pose_state_json);
        auto config = saved.is_null() ? camera_pose_config_.value_or(PoseSessionConfig{}) : pose_session_config_from_state(saved);
        if (saved.is_null() && !camera_pose_config_) {
            config.warmup_iterations = params.optimization.camera_pose_start_step;
        }
        config.total_iterations = params.optimization.resolved_total_iterations();
        if (config.total_iterations <= 0)
            return pose_initialization_error("Camera pose training duration must be positive", LFS_SOURCE_SITE_CURRENT());
        if (saved.is_null() && !camera_pose_config_) {
            const int stop = params.optimization.resolved_camera_pose_stop_step();
            // Preserve the exact integer cutoff through floor(total*fraction),
            // including percentages whose binary representation rounds down.
            config.freeze_fraction = std::nextafter(static_cast<double>(stop) / config.total_iterations, 1.0);
        }
        if (saved.is_null() && params.optimization.ppisp_use_controller) {
            const int controller_start = params.optimization.resolved_ppisp_controller_activation_step(config.total_iterations);
            // Controller distillation is a separate phase. Keep camera poses
            // fixed before it starts, including when Gaussians are not frozen.
            const double controller_fraction = std::nextafter(
                static_cast<double>(controller_start) / config.total_iterations, 1.0);
            config.freeze_fraction = std::min(config.freeze_fraction, controller_fraction);
        }
        config.optimizer.scene_scale = model.get_scene_scale();
        if (params.optimization.ppisp_use_controller &&
            std::floor(config.total_iterations * config.freeze_fraction) >
                params.optimization.resolved_ppisp_controller_activation_step(config.total_iterations))
            return pose_initialization_error("Saved camera poses must freeze before controller distillation; reinitialize to change the schedule", LFS_SOURCE_SITE_CURRENT());
        if (config.warmup_iterations >= std::floor(config.total_iterations * config.freeze_fraction))
            return pose_initialization_error("Camera pose warmup must end before the pose-freeze phase; increase training iterations", LFS_SOURCE_SITE_CURRENT());
        std::unordered_set<int> training;
        for (size_t i = 0; i < train_dataset_->size(); ++i) {
            const auto& camera = train_dataset_->get_cameras().at(train_dataset_->local_to_source(i));
            if (camera && camera->split() != CameraSplit::Eval)
                training.insert(camera->uid());
        }
        std::vector<PoseCameraInput> inputs;
        inputs.reserve(camera_pose_sources_.size());
        for (const auto& camera : camera_pose_sources_) {
            if (!camera)
                return pose_initialization_error("Missing source camera for pose refinement", LFS_SOURCE_SITE_CURRENT());
            const auto source = camera->world_view_transform().to(Device::CPU).contiguous();
            if (source.dtype() != DataType::Float32 || source.numel() != 16)
                return pose_initialization_error("Camera pose refinement requires float32 source transforms", LFS_SOURCE_SITE_CURRENT());
            PoseCameraInput input;
            input.uid = camera->uid();
            input.role = training.contains(input.uid) ? PoseRole::Train : PoseRole::Evaluation;
            std::copy_n(source.ptr<float>(), 16, input.source.begin());
            inputs.push_back(input);
        }
        static std::atomic<std::uint64_t> next_generation{1};
        auto session = std::make_shared<PoseRefinementSession>(next_generation.fetch_add(1), std::move(inputs), config);
        // Legacy checkpoints retain their original fixed-structure objective.
        // New sessions and shared-geometry checkpoints use the active training
        // membership and the already-prepared undistortion calibration.
        if (saved.is_null() || saved.at("version") == 2) {
            std::vector<SparseTrackMeasurement> measurements;
            for (const auto& camera : camera_pose_sources_) {
                auto observations = make_sparse_track_measurements(*camera, training.contains(camera->uid()));
                measurements.insert(measurements.end(), observations.begin(), observations.end());
            }
            session->configure_sparse_points(std::move(measurements));
        }
        if (!saved.is_null()) {
            session->restore_state(saved);
        }
        size_t guarded = 0, movable = 0, joint = 0;
        const auto snapshot = session->published_snapshot();
        std::unordered_set<int> anchors;
        for (const auto& pose : snapshot->cameras)
            if (pose.state == PoseDisplayState::Anchor)
                anchors.insert(pose.pose.uid);
        for (const auto& camera : camera_pose_sources_) {
            if (!training.contains(camera->uid()) || anchors.contains(camera->uid()))
                continue;
            ++movable;
            if (session->joint_geometry_enabled(camera->uid()))
                ++joint;
            else
                guarded += make_sparse_reprojection_guard(*camera).active();
        }
        LOG_INFO("Camera pose SfM geometry: {} shared points; {}/{} movable cameras use joint reprojection, {} use fixed source reprojection, {} use photometric-only acceptance",
                 session->shared_point_count(), joint, movable, guarded, movable - guarded - joint);
        LOG_INFO("Camera pose refinement: {} cameras, warmup={}, freeze at={}, steps/visit={}, visits between updates={}, restored={}",
                 session->published_snapshot()->cameras.size(), config.warmup_iterations, static_cast<int>(std::floor(config.total_iterations * config.freeze_fraction)),
                 config.steps_per_visit, config.visits_between_updates, !saved.is_null());
        return session;
    } catch (const std::exception& error) {
        return lfs::make_error(lfs::ErrorInit{
            .code = lfs::ErrorCode::FailedPrecondition,
            .domain = lfs::ErrorDomain::Training,
            .user_message = "Camera pose refinement could not be initialized.",
            .detail = std::string("Camera pose state could not be initialized: ") + error.what(),
            .detection = LFS_SOURCE_SITE_CURRENT(),
        });
    }
} // namespace lfs::training
