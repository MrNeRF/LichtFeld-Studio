/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/apriltag_scene_scale.hpp"

#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/native_fs.hpp"
#include "core/path_utils.hpp"

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

namespace lfs::core {

    namespace {

        void populateCameraDistortion(
            const Camera& camera,
            geometry::apriltag::CameraIntrinsics& intrinsics) {
            intrinsics.distortion.fill(0.0);
            intrinsics.num_distortion = 0;

            const auto coefficients = [](const Tensor& tensor) {
                return tensor.is_valid() ? tensor.to_vector() : std::vector<float>{};
            };
            const auto rad_vec = coefficients(camera.radial_distortion());
            const auto tan_vec = coefficients(camera.tangential_distortion());
            const auto place = [&](const int idx, const double val) {
                intrinsics.distortion[idx] = val;
                intrinsics.num_distortion = std::max(intrinsics.num_distortion, idx + 1);
            };

            switch (camera.camera_model_type()) {
            case CameraModelType::PINHOLE: {
                const bool rational =
                    rad_vec.size() >= 6 &&
                    (std::abs(rad_vec[3]) > 0.0 || std::abs(rad_vec[4]) > 0.0 || std::abs(rad_vec[5]) > 0.0);
                const std::size_t radial_count =
                    rational ? std::min<std::size_t>(rad_vec.size(), 6) : std::min<std::size_t>(rad_vec.size(), 3);
                for (std::size_t i = 0; i < radial_count; ++i) {
                    place(static_cast<int>(i), rad_vec[i]);
                }
                const int tangential_offset = rational ? 6 : 3;
                const std::size_t tan_count = std::min<std::size_t>(tan_vec.size(), 2);
                for (std::size_t i = 0; i < tan_count; ++i) {
                    place(tangential_offset + static_cast<int>(i), tan_vec[i]);
                }
                break;
            }
            case CameraModelType::FISHEYE:
            case CameraModelType::THIN_PRISM_FISHEYE: {
                const std::size_t radial_count = std::min<std::size_t>(rad_vec.size(), 4);
                for (std::size_t i = 0; i < radial_count; ++i) {
                    place(static_cast<int>(i), rad_vec[i]);
                }
                if (camera.camera_model_type() == CameraModelType::FISHEYE)
                    break;
                constexpr std::array<int, 6> kThinPrismSlots = {4, 5, 6, 8, 7, 9};
                const std::size_t tan_count = std::min<std::size_t>(tan_vec.size(), kThinPrismSlots.size());
                for (std::size_t i = 0; i < tan_count; ++i) {
                    place(kThinPrismSlots[i], tan_vec[i]);
                }
                break;
            }
            default:
                break;
            }
        }

    } // namespace

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
            if (ptr) {
                view.world_to_camera_rotation = glm::dmat3(glm::transpose(glm::make_mat3(ptr)));
            }
        }

        if (const auto& T_tensor = camera.T(); T_tensor.is_valid() && T_tensor.numel() >= 3) {
            const auto T_cpu = T_tensor.cpu().contiguous();
            const float* const ptr = T_cpu.ptr<float>();
            if (ptr) {
                view.world_to_camera_translation = glm::dvec3(glm::make_vec3(ptr));
            }
        }

        view.intrinsics.fx = static_cast<double>(camera.focal_x());
        view.intrinsics.fy = static_cast<double>(camera.focal_y());
        view.intrinsics.cx = static_cast<double>(camera.center_x());
        view.intrinsics.cy = static_cast<double>(camera.center_y());
        view.intrinsics.width = camera.camera_width();
        view.intrinsics.height = camera.camera_height();
        view.intrinsics.lens_model = camera.camera_model_type();
        populateCameraDistortion(camera, view.intrinsics);

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
            camera_list_generation != scene.cameraListGeneration()) {
            return false;
        }
        const auto nodes = scene.getNodes();
        if (nodes.size() != node_uuids.size()) {
            return false;
        }
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            if (nodes[i]->uuid != node_uuids[i]) {
                return false;
            }
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
            if (node->camera) {
                transforms.try_emplace(node->camera->uid(), scene.getWorldTransform(node->id));
            }
        }
        const auto cameras = active_cameras_only ? scene.getActiveCameras() : scene.getAllCameras();
        for (const auto& camera : cameras) {
            if (camera) {
                snapshot.cameras.push_back({buildCameraObservationView(*camera,
                                                                       transforms.at(camera->uid())),
                                            camera->image_path()});
            }
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
            if (cancelled.load(std::memory_order_relaxed)) {
                return;
            }

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
                    !progress_callback(completed_count.load(std::memory_order_relaxed), cameras.size(), cameras[idx].view.image_name)) {
                    cancelled.store(true, std::memory_order_relaxed);
                }
            }
        });

        if (cancelled.load(std::memory_order_relaxed)) {
            return {};
        }
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

    namespace {

        AprilTagSceneScaleReport solveAprilTagScaleFromViews(
            std::vector<geometry::apriltag::CameraObservationView> views,
            const std::span<const geometry::apriltag::TagDistanceConstraint> constraints,
            const AprilTagSceneScaleOptions& options) {
            AprilTagSceneScaleReport report;

            if (views.empty()) {
                report.error_message = "No camera views available for AprilTag triangulation";
                return report;
            }

            report.triangulation = geometry::apriltag::triangulateAprilTags(
                views,
                options.triangulation_config);

            if (report.triangulation.tags.empty()) {
                report.error_message = "No AprilTags could be triangulated in 3D from camera views";
            } else {
                report.scale_result = geometry::apriltag::solveGlobalScale(
                    report.triangulation.tags,
                    constraints,
                    options.solver_config);
                report.success = report.scale_result.success;
                if (!report.success) {
                    report.error_message = report.scale_result.error_message;
                }
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
                    const auto error = "Failed to save AprilTag calibration manifest: " + path_to_utf8(path);
                    LOG_WARN("{}", error);
                    if (report.error_message.empty())
                        report.error_message = error;
                }
            }

            report.views = std::move(views);
            return report;
        }

    } // namespace

    AprilTagSceneScaleReport computeAprilTagSceneScale(
        const AprilTagSceneSnapshot& snapshot,
        const std::span<const geometry::apriltag::TagDistanceConstraint> constraints,
        const AprilTagSceneScaleOptions& options,
        const bool solve_scale,
        const AprilTagProgressCallback& progress_callback) {
        auto views = detectAprilTagsInSnapshot(snapshot.cameras, options.detector_config,
                                               options.max_image_width, progress_callback);
        if ((!snapshot.cameras.empty() && views.empty()) ||
            (progress_callback && !progress_callback(views.size(), snapshot.cameras.size(), {}))) {
            AprilTagSceneScaleReport report;
            report.error_message = "AprilTag detection was cancelled";
            return report;
        }
        if (solve_scale)
            return solveAprilTagScaleFromViews(std::move(views), constraints, options);
        AprilTagSceneScaleReport report;
        report.triangulation = geometry::apriltag::triangulateAprilTags(views, options.triangulation_config);
        report.views = std::move(views);
        report.success = true;
        return report;
    }

} // namespace lfs::core
