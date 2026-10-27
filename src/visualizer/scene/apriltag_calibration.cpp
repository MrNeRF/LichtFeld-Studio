/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "visualizer/apriltag_calibration.hpp"

#include "core/camera_distortion_layout.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/native_fs.hpp"
#include "core/path_utils.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "scene/scene_manager.hpp"
#include "visualizer/gui_capabilities.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <fstream>
#include <glm/gtc/type_ptr.hpp>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <unordered_map>

namespace lfs::vis {

    using namespace core;

    geometry::apriltag::CameraObservationView buildCameraObservationView(
        const Camera& camera,
        const glm::mat4& scene_world_transform) {
        geometry::apriltag::CameraObservationView view;
        view.camera_uid = camera.uid();
        view.image_name = camera.image_name();
        view.scene_world_transform = glm::dmat4(scene_world_transform);

        if (const auto& R_tensor = camera.R(); R_tensor.is_valid() && R_tensor.numel() >= 9) {
            const auto R_cpu = R_tensor.cpu().contiguous();
            const float* const ptr = R_cpu.ptr<float>();
            if (ptr)
                view.world_to_camera_rotation = glm::dmat3(glm::transpose(glm::make_mat3(ptr)));
        }

        if (const auto& T_tensor = camera.T(); T_tensor.is_valid() && T_tensor.numel() >= 3) {
            const auto T_cpu = T_tensor.cpu().contiguous();
            const float* const ptr = T_cpu.ptr<float>();
            if (ptr)
                view.world_to_camera_translation = glm::dvec3(glm::make_vec3(ptr));
        }

        view.intrinsics.fx = static_cast<double>(camera.focal_x());
        view.intrinsics.fy = static_cast<double>(camera.focal_y());
        view.intrinsics.cx = static_cast<double>(camera.center_x());
        view.intrinsics.cy = static_cast<double>(camera.center_y());
        view.intrinsics.width = camera.camera_width();
        view.intrinsics.height = camera.camera_height();
        view.intrinsics.lens_model = camera.camera_model_type();
        const auto coefficients = [](const Tensor& tensor) {
            return tensor.is_valid() ? tensor.to_vector() : std::vector<float>{};
        };
        const auto rad_vec = coefficients(camera.radial_distortion());
        const auto tan_vec = coefficients(camera.tangential_distortion());
        const auto model = camera.camera_model_type();
        const bool rational = rad_vec.size() >= 6 &&
                              (std::abs(rad_vec[3]) > 0.0 || std::abs(rad_vec[4]) > 0.0 || std::abs(rad_vec[5]) > 0.0);
        const std::size_t radial_limit = model == CameraModelType::PINHOLE ? (rational ? 6 : 3) : 4;
        const std::size_t tangential_limit = model == CameraModelType::THIN_PRISM_FISHEYE ? 6 : 2;
        view.intrinsics.num_distortion = packCameraDistortion(model,
                                                              std::span(rad_vec).first(std::min(rad_vec.size(), radial_limit)),
                                                              std::span(tan_vec).first(std::min(tan_vec.size(), tangential_limit)), std::span(view.intrinsics.distortion));

        // Detection reads the original file, even after training prepares an undistorted grid.
        if (camera.is_undistort_precomputed()) {
            const auto& source = camera.undistort_params();
            view.intrinsics.fx = source.src_fx;
            view.intrinsics.fy = source.src_fy;
            view.intrinsics.cx = source.src_cx;
            view.intrinsics.cy = source.src_cy;
            view.intrinsics.width = source.src_width;
            view.intrinsics.height = source.src_height;
            view.intrinsics.lens_model = source.model_type;
            view.intrinsics.num_distortion = std::clamp(source.num_distortion, 0, 12);
            std::copy_n(source.distortion, view.intrinsics.num_distortion, view.intrinsics.distortion.begin());
        }
        view.detected_image_width = view.intrinsics.width;
        view.detected_image_height = view.intrinsics.height;
        return view;
    }

    bool AprilTagSceneSnapshot::matches(const Scene& scene) const {
        if (render_generation != scene.renderGeneration() ||
            camera_list_generation != scene.cameraListGeneration())
            return false;
        const auto nodes = scene.getNodes();
        if (nodes.size() != node_uuids.size())
            return false;
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            if (nodes[i]->uuid != node_uuids[i])
                return false;
        }
        return true;
    }

    AprilTagSceneSnapshot snapshotAprilTagScene(const Scene& scene, const bool active_cameras_only) {
        AprilTagSceneSnapshot snapshot;
        snapshot.render_generation = scene.renderGeneration();
        snapshot.camera_list_generation = scene.cameraListGeneration();
        std::unordered_map<int, glm::mat4> transforms;
        for (const auto* node : scene.getNodes()) {
            snapshot.node_uuids.push_back(node->uuid);
            if (node->camera)
                transforms.try_emplace(node->camera->uid(), scene.getWorldTransform(node->id));
        }
        const auto cameras = active_cameras_only ? scene.getActiveCameras() : scene.getAllCameras();
        for (const auto& camera : cameras) {
            if (camera)
                snapshot.cameras.push_back({buildCameraObservationView(*camera, transforms.at(camera->uid())), camera->image_path()});
        }
        return snapshot;
    }

    std::vector<geometry::apriltag::CameraObservationView> detectAprilTagsInSnapshot(
        const std::span<const AprilTagCameraInput> cameras,
        const geometry::apriltag::AprilTagDetectorConfig& detector_config,
        const int max_image_width,
        const AprilTagProgressCallback& progress_callback) {
        std::vector<geometry::apriltag::CameraObservationView> views(cameras.size());
        geometry::apriltag::AprilTagDetectorConfig per_worker_config = detector_config;
        per_worker_config.num_threads = 1;
        tbb::enumerable_thread_specific<geometry::apriltag::AprilTagDetector> detectors(
            [&]() { return geometry::apriltag::AprilTagDetector(per_worker_config); });

        std::atomic<std::size_t> completed_count{0};
        std::atomic<bool> cancelled{false};
        std::mutex progress_mutex;

        tbb::parallel_for(std::size_t{0}, cameras.size(), [&](const std::size_t idx) {
            if (cancelled.load(std::memory_order_relaxed))
                return;

            auto view = cameras[idx].view;
            const auto& img_path = cameras[idx].image_path;
            if (progress_callback) {
                std::lock_guard<std::mutex> lock(progress_mutex);
                if (cancelled.load(std::memory_order_relaxed) ||
                    !progress_callback(completed_count.load(), cameras.size(), view.image_name)) {
                    cancelled.store(true, std::memory_order_relaxed);
                    return;
                }
            }
            if (!img_path.empty() && native_fs::exists(img_path)) {
                auto [pixels, w, h, c] = load_image(img_path, -1, max_image_width);
                const std::unique_ptr<unsigned char, decltype(&free_image)> image(pixels, &free_image);
                if (image && w > 0 && h > 0 && c > 0) {
                    view.detected_image_width = w;
                    view.detected_image_height = h;
                    auto& detector = detectors.local();
                    view.detections = detector.detectInterleaved(image.get(), w, h, c);
                }
            }

            views[idx] = std::move(view);

            completed_count.fetch_add(1, std::memory_order_relaxed);
            if (progress_callback) {
                std::lock_guard<std::mutex> lock(progress_mutex);
                if (!cancelled.load(std::memory_order_relaxed) &&
                    !progress_callback(completed_count.load(std::memory_order_relaxed), cameras.size(), cameras[idx].view.image_name))
                    cancelled.store(true, std::memory_order_relaxed);
            }
        });

        if (cancelled.load(std::memory_order_relaxed))
            return {};
        return views;
    }

    AprilTagSceneScaleOptions aprilTagSceneScaleOptions(const nlohmann::json& args) {
        AprilTagSceneScaleOptions options;
        const auto family_name = args.value("family", std::string(geometry::apriltag::aprilTagFamilyName(options.detector_config.family)));
        const auto family = geometry::apriltag::parseAprilTagFamily(family_name);
        if (!family)
            throw std::invalid_argument("Unsupported AprilTag family: " + family_name);
        options.detector_config.family = *family;
        options.detector_config.max_hamming = args.value("max_hamming", options.detector_config.max_hamming);
        options.detector_config.min_decision_margin = args.value("min_decision_margin", options.detector_config.min_decision_margin);
        options.triangulation_config.min_views = args.value("min_views", options.triangulation_config.min_views);
        options.triangulation_config.min_parallax_angle_deg = args.value("min_parallax_deg", options.triangulation_config.min_parallax_angle_deg);
        options.triangulation_config.max_reprojection_error_px = args.value("max_reprojection_error_px", options.triangulation_config.max_reprojection_error_px);
        options.triangulation_config.max_hamming = options.detector_config.max_hamming;
        options.triangulation_config.min_decision_margin = options.detector_config.min_decision_margin;
        options.solver_config.inlier_relative_threshold = args.value("inlier_relative_threshold", options.solver_config.inlier_relative_threshold);
        options.solver_config.inlier_absolute_threshold = args.value("inlier_absolute_threshold", options.solver_config.inlier_absolute_threshold);
        options.solver_config.min_inliers = args.value("min_inliers", options.solver_config.min_inliers);
        options.solver_config.min_inlier_ratio = args.value("min_inlier_ratio", options.solver_config.min_inlier_ratio);
        options.active_cameras_only = args.value("active_cameras_only", options.active_cameras_only);
        options.max_image_width = args.value("max_image_width", options.max_image_width);
        options.manifest_output_path = native_fs::display_path(utf8_to_path(args.value("manifest_path", std::string{})));
        return options;
    }

    AprilTagSceneScaleReport computeAprilTagSceneScale(
        const AprilTagSceneSnapshot& snapshot,
        const std::span<const geometry::apriltag::TagDistanceConstraint> constraints,
        const AprilTagSceneScaleOptions& options,
        const bool solve_scale,
        const AprilTagProgressCallback& progress_callback) {
        AprilTagSceneScaleReport report;
        auto& views = report.views;
        views = detectAprilTagsInSnapshot(snapshot.cameras, options.detector_config, options.max_image_width, progress_callback);
        if ((!snapshot.cameras.empty() && views.empty()) ||
            (progress_callback && !progress_callback(views.size(), snapshot.cameras.size(), {}))) {
            views.clear();
            report.error_message = "AprilTag detection was cancelled";
            return report;
        }
        if (solve_scale && views.empty()) {
            report.error_message = "No camera views available for AprilTag triangulation";
            return report;
        }
        report.triangulation = geometry::apriltag::triangulateAprilTags(views, options.triangulation_config);
        if (!solve_scale) {
            report.success = true;
            return report;
        }
        if (report.triangulation.tags.empty())
            report.error_message = "No AprilTags could be triangulated in 3D from camera views";
        else {
            report.scale_result = geometry::apriltag::solveGlobalScale(report.triangulation.tags, constraints, options.solver_config);
            report.success = report.scale_result.success;
            if (!report.success)
                report.error_message = report.scale_result.error_message;
        }

        geometry::apriltag::ScaleCalibrationManifest manifest;
        manifest.family = options.detector_config.family;
        manifest.triangulation_config = options.triangulation_config;
        manifest.solver_config = options.solver_config;
        manifest.triangulation = report.triangulation;
        manifest.scale_result = report.scale_result;
        report.manifest_json = geometry::apriltag::serializeManifestToJson(manifest, 2);

        if (!options.manifest_output_path.empty()) {
            const auto& path = options.manifest_output_path;
            std::error_code ec;
            if (const auto parent = path.parent_path(); !parent.empty())
                native_fs::create_directories(parent, ec);
            std::ofstream out;
            if (!ec && open_file_for_write(path, std::ios::binary | std::ios::trunc, out) &&
                (out << report.manifest_json).flush().good()) {
                report.manifest_saved_path = path;
            } else {
                report.success = false;
                const auto error = "Failed to save AprilTag calibration manifest: " + path_to_utf8(path);
                LOG_WARN("{}", error);
                if (report.error_message.empty())
                    report.error_message = error;
            }
        }

        return report;
    }

    AprilTagSceneApplyResult applyAprilTagScaleReport(
        SceneManager& scene_manager,
        const AprilTagSceneSnapshot& snapshot,
        AprilTagSceneScaleReport report,
        const std::vector<std::string>& requested_targets,
        const bool apply_to_scene,
        const std::string_view undo_label) {
        AprilTagSceneApplyResult result;
        if (!snapshot.matches(scene_manager.getScene())) {
            report.success = false;
            report.error_message = LOC("runtime.apriltag_scene_changed");
        }
        result.scale_report = std::move(report);
        result.error_message = result.scale_report.error_message;
        if (!result.scale_report.success || !apply_to_scene)
            return result;
        const auto scale_factor = static_cast<float>(result.scale_report.scale_result.scale_factor);
        if (!std::isfinite(scale_factor) || scale_factor <= 1e-6f) {
            result.error_message = "Scale factor must be finite and positive";
            return result;
        }

        const auto& scene = scene_manager.getScene();
        auto candidate_names = requested_targets;
        if (candidate_names.empty()) {
            for (const auto* const node : scene.getNodes()) {
                if (node && (cap::isAlignTransformTargetType(node->type) ||
                             node->type == core::NodeType::COLLISION ||
                             node->type == core::NodeType::CAMERA_GROUP || node->type == core::NodeType::CAMERA)) {
                    candidate_names.push_back(node->name);
                }
            }
        }

        const auto vis_pivot = rendering::visualizerWorldPointFromDataWorld(glm::vec3(result.scale_report.scale_result.pivot));
        const auto delta = geometry::apriltag::buildUniformScaleTransform(vis_pivot, scale_factor);
        auto transformed = cap::applyVisualizerWorldDelta(scene_manager, candidate_names, delta, undo_label);
        result.applied = transformed.has_value();
        if (transformed)
            result.transformed_nodes = std::move(*transformed);
        else
            result.error_message = transformed.error();
        return result;
    }

    nlohmann::json aprilTagResultJson(const AprilTagSceneApplyResult& result) {
        const auto& report = result.scale_report;
        auto output = nlohmann::json::parse(report.manifest_json, nullptr, false);
        if (output.is_discarded())
            output = nlohmann::json::object();
        output.update(geometry::apriltag::scaleResultToJson(report.scale_result));
        output.update(geometry::apriltag::triangulationToJson(report.triangulation));
        for (auto& tag : output["triangulated_tags"]) {
            const auto& p = tag["position"];
            const auto vis = rendering::visualizerWorldPointFromDataWorld(
                glm::vec3(p[0].get<float>(), p[1].get<float>(), p[2].get<float>()));
            tag["visualizer_position"] = {vis.x, vis.y, vis.z};
        }
        output["success"] = report.success;
        output["error_message"] = result.error_message;
        output["applied"] = result.applied;
        output["camera_count"] = report.views.size();
        output["transformed_nodes"] = result.transformed_nodes;
        output["manifest_json"] = report.manifest_json;
        output["manifest_saved_path"] = core::path_to_utf8(report.manifest_saved_path);
        return output;
    }

} // namespace lfs::vis
