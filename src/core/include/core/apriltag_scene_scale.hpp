/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include "core/export.hpp"
#include "core/scene.hpp"
#include "geometry/apriltag_detector.hpp"
#include "geometry/apriltag_scaling.hpp"

#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace lfs::core {

    using AprilTagProgressCallback = std::function<bool(std::size_t completed, std::size_t total, const std::string& image_name)>;

    [[nodiscard]] LFS_CORE_API geometry::apriltag::CameraObservationView buildCameraObservationView(
        const Camera& camera,
        const glm::mat4& scene_world_transform = glm::mat4(1.0f));

    struct AprilTagCameraInput {
        geometry::apriltag::CameraObservationView view;
        std::filesystem::path image_path;
    };

    struct AprilTagSceneSnapshot {
        std::vector<AprilTagCameraInput> cameras;
        std::vector<Uuid> node_uuids;
        uint64_t render_generation = 0;
        uint64_t camera_list_generation = 0;

        [[nodiscard]] LFS_CORE_API bool matches(const Scene& scene) const;
    };

    // Capture and validate on the scene owner thread; workers only use copied inputs.
    [[nodiscard]] LFS_CORE_API AprilTagSceneSnapshot snapshotAprilTagScene(
        const Scene& scene, bool active_cameras_only = false);

    [[nodiscard]] LFS_CORE_API std::vector<geometry::apriltag::CameraObservationView> detectAprilTagsInSnapshot(
        std::span<const AprilTagCameraInput> cameras,
        const geometry::apriltag::AprilTagDetectorConfig& detector_config = {},
        int max_image_width = 0,
        const AprilTagProgressCallback& progress_callback = {});

    [[nodiscard]] LFS_CORE_API std::vector<geometry::apriltag::CameraObservationView> detectAprilTagsInCameras(
        std::span<const std::shared_ptr<Camera>> cameras,
        const geometry::apriltag::AprilTagDetectorConfig& detector_config = {},
        std::span<const glm::mat4> camera_scene_transforms = {},
        int max_image_width = 0,
        const AprilTagProgressCallback& progress_callback = {});

    [[nodiscard]] LFS_CORE_API std::vector<geometry::apriltag::CameraObservationView> detectAprilTagsInScene(
        const Scene& scene,
        const geometry::apriltag::AprilTagDetectorConfig& detector_config = {},
        bool active_cameras_only = false,
        int max_image_width = 0,
        const AprilTagProgressCallback& progress_callback = {});

    struct AprilTagSceneScaleOptions {
        geometry::apriltag::AprilTagDetectorConfig detector_config{};
        geometry::apriltag::TriangulationConfig triangulation_config{};
        geometry::apriltag::ScaleSolverConfig solver_config{};
        bool active_cameras_only = false;
        int max_image_width = 0;
        std::filesystem::path manifest_output_path{};
    };

    struct AprilTagSceneScaleReport {
        bool success = false;
        std::string error_message;
        std::vector<geometry::apriltag::CameraObservationView> views;
        geometry::apriltag::TriangulationSummary triangulation;
        geometry::apriltag::ScaleSolveResult scale_result;
        std::string manifest_json;
        std::filesystem::path manifest_saved_path;
    };

    [[nodiscard]] LFS_CORE_API bool saveCalibrationManifest(
        const std::filesystem::path& path,
        const geometry::apriltag::ScaleCalibrationManifest& manifest,
        std::string* out_error = nullptr);

    [[nodiscard]] LFS_CORE_API AprilTagSceneScaleReport computeAprilTagSceneScale(
        const Scene& scene,
        std::span<const geometry::apriltag::TagDistanceConstraint> constraints,
        const AprilTagSceneScaleOptions& options = {},
        const AprilTagProgressCallback& progress_callback = {});

    [[nodiscard]] LFS_CORE_API AprilTagSceneScaleReport solveAprilTagScaleFromViews(
        std::span<const geometry::apriltag::CameraObservationView> views,
        std::span<const geometry::apriltag::TagDistanceConstraint> constraints,
        const AprilTagSceneScaleOptions& options = {});

} // namespace lfs::core
