/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera_types.h"
#include "geometry/apriltag_detector.hpp"

#include <array>
#include <glm/glm.hpp>
#include <nlohmann/json_fwd.hpp>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace lfs::geometry::apriltag {

    struct CameraIntrinsics {
        double fx = 0.0;
        double fy = 0.0;
        double cx = 0.0;
        double cy = 0.0;
        int width = 0;
        int height = 0;
        core::CameraModelType lens_model = core::CameraModelType::PINHOLE;
        std::array<double, 12> distortion{};
        int num_distortion = 0;
    };

    struct CameraObservationView {
        int camera_uid = -1;
        std::string image_name{};
        glm::dmat3 world_to_camera_rotation{1.0};
        glm::dvec3 world_to_camera_translation{0.0};
        glm::dmat4 scene_world_transform{1.0};
        CameraIntrinsics intrinsics{};
        int detected_image_width = 0;
        int detected_image_height = 0;
        std::vector<TagDetection2D> detections{};
    };

    struct Ray3D {
        glm::dvec3 origin{0.0};
        glm::dvec3 direction{0.0, 0.0, 1.0};
    };

    [[nodiscard]] glm::dvec2 applyForwardDistortion(
        const glm::dvec2& normalized_xy,
        core::CameraModelType model,
        std::span<const double> coeffs);

    [[nodiscard]] std::optional<glm::dvec2> invertDistortion(
        const glm::dvec2& distorted_xy,
        core::CameraModelType model,
        std::span<const double> coeffs);

    [[nodiscard]] std::optional<Ray3D> unprojectPixelToWorldRay(
        const CameraObservationView& view,
        const glm::dvec2& detected_pixel);

    [[nodiscard]] std::optional<glm::dvec2> projectWorldPointToDetectedPixel(
        const CameraObservationView& view,
        const glm::dvec3& world_point);

    struct TriangulationConfig {
        int min_views = 2;
        double min_parallax_angle_deg = 1.5;
        double max_reprojection_error_px = 3.0;
        int max_hamming = 1;
        float min_decision_margin = 25.0f;
    };

    struct TriangulatedTag3D {
        int tag_id = -1;
        glm::dvec3 position{0.0};
        int total_observations = 0;
        int inlier_observations = 0;
        double mean_reprojection_error_px = 0.0;
        double max_reprojection_error_px = 0.0;
        double max_parallax_angle_deg = 0.0;
        std::vector<int> camera_uids{};
        std::vector<std::string> image_names{};
    };

    struct RejectedTagDiagnostic {
        int tag_id = -1;
        int total_observations = 0;
        double max_parallax_angle_deg = 0.0;
        std::string reason{};
    };

    struct TriangulationSummary {
        std::vector<TriangulatedTag3D> tags{};
        std::vector<RejectedTagDiagnostic> rejected_tags{};
    };

    [[nodiscard]] TriangulationSummary triangulateAprilTags(
        std::span<const CameraObservationView> views,
        const TriangulationConfig& config = {});

    struct TagDistanceConstraint {
        int tag_a = -1;
        int tag_b = -1;
        double target_distance = 0.0;
    };

    struct ScaleSolverConfig {
        double inlier_relative_threshold = 0.02;
        double inlier_absolute_threshold = 0.0;
        int min_inliers = 1;
        double min_inlier_ratio = 0.5;
    };

    struct EvaluatedDistanceConstraint {
        int tag_a = -1;
        int tag_b = -1;
        double target_distance = 0.0;
        double unscaled_distance = 0.0;
        double scaled_distance = 0.0;
        double absolute_error = 0.0;
        double relative_error = 0.0;
        double candidate_scale = 0.0;
        bool valid = false;
        bool inlier = false;
        std::string status{};
    };

    struct ScaleSolveResult {
        bool success = false;
        std::string error_message{};
        double scale_factor = 1.0;
        glm::dvec3 pivot{0.0};
        glm::mat4 scale_transform{1.0f};
        int total_constraints = 0;
        int valid_constraints = 0;
        int inlier_count = 0;
        double inlier_ratio = 0.0;
        double rmse = 0.0;
        double mae = 0.0;
        double max_error = 0.0;
        double mean_relative_error = 0.0;
        std::vector<EvaluatedDistanceConstraint> evaluated_constraints{};
    };

    [[nodiscard]] glm::mat4 buildUniformScaleTransform(
        const glm::vec3& pivot,
        float scale_factor);

    [[nodiscard]] ScaleSolveResult solveGlobalScale(
        std::span<const TriangulatedTag3D> tags,
        std::span<const TagDistanceConstraint> constraints,
        const ScaleSolverConfig& config = {});

    [[nodiscard]] nlohmann::ordered_json triangulationToJson(const TriangulationSummary& summary);
    [[nodiscard]] nlohmann::ordered_json scaleResultToJson(const ScaleSolveResult& result);

    struct ScaleCalibrationManifest {
        AprilTagFamily family = AprilTagFamily::Tag36h11;
        TriangulationConfig triangulation_config{};
        ScaleSolverConfig solver_config{};
        TriangulationSummary triangulation{};
        ScaleSolveResult scale_result{};
        std::string timestamp_utc;
    };

    [[nodiscard]] std::string serializeManifestToJson(
        const ScaleCalibrationManifest& manifest,
        int indent = 2);

} // namespace lfs::geometry::apriltag
