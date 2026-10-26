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
#include <mutex>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <unordered_map>

namespace lfs::core {

    namespace {

        [[nodiscard]] geometry::apriltag::LensModel toGeometryLensModel(const CameraModelType model) noexcept {
            switch (model) {
            case CameraModelType::PINHOLE:
                return geometry::apriltag::LensModel::Pinhole;
            case CameraModelType::ORTHO:
                return geometry::apriltag::LensModel::Ortho;
            case CameraModelType::FISHEYE:
                return geometry::apriltag::LensModel::Fisheye;
            case CameraModelType::EQUIRECTANGULAR:
                return geometry::apriltag::LensModel::Equirectangular;
            case CameraModelType::THIN_PRISM_FISHEYE:
                return geometry::apriltag::LensModel::ThinPrismFisheye;
            }
            return geometry::apriltag::LensModel::Pinhole;
        }

        void populateCameraDistortion(
            const Camera& camera,
            geometry::apriltag::CameraIntrinsics& intrinsics) {
            intrinsics.distortion.fill(0.0);
            intrinsics.num_distortion = 0;

            std::vector<double> rad_vec;
            if (const auto& radial = camera.radial_distortion(); radial.is_valid() && radial.numel() > 0) {
                const auto rad_cpu = radial.cpu().contiguous();
                const float* const ptr = rad_cpu.ptr<float>();
                if (ptr) {
                    rad_vec.reserve(rad_cpu.numel());
                    for (std::size_t i = 0; i < rad_cpu.numel(); ++i) {
                        rad_vec.push_back(static_cast<double>(ptr[i]));
                    }
                }
            }

            std::vector<double> tan_vec;
            if (const auto& tangential = camera.tangential_distortion(); tangential.is_valid() && tangential.numel() > 0) {
                const auto tan_cpu = tangential.cpu().contiguous();
                const float* const ptr = tan_cpu.ptr<float>();
                if (ptr) {
                    tan_vec.reserve(tan_cpu.numel());
                    for (std::size_t i = 0; i < tan_cpu.numel(); ++i) {
                        tan_vec.push_back(static_cast<double>(ptr[i]));
                    }
                }
            }

            const auto place = [&](const int idx, const double val) {
                if (idx >= 0 && idx < static_cast<int>(intrinsics.distortion.size())) {
                    intrinsics.distortion[static_cast<std::size_t>(idx)] = val;
                    intrinsics.num_distortion = std::max(intrinsics.num_distortion, idx + 1);
                }
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
            case CameraModelType::FISHEYE: {
                const std::size_t radial_count = std::min<std::size_t>(rad_vec.size(), 4);
                for (std::size_t i = 0; i < radial_count; ++i) {
                    place(static_cast<int>(i), rad_vec[i]);
                }
                break;
            }
            case CameraModelType::THIN_PRISM_FISHEYE: {
                constexpr std::array<int, 6> kThinPrismSlots = {4, 5, 6, 8, 7, 9};
                const std::size_t radial_count = std::min<std::size_t>(rad_vec.size(), 4);
                for (std::size_t i = 0; i < radial_count; ++i) {
                    place(static_cast<int>(i), rad_vec[i]);
                }
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
                glm::dmat3 R_mat(1.0);
                for (int row = 0; row < 3; ++row) {
                    for (int col = 0; col < 3; ++col) {
                        R_mat[col][row] = static_cast<double>(ptr[row * 3 + col]);
                    }
                }
                view.world_to_camera_rotation = R_mat;
            }
        }

        if (const auto& T_tensor = camera.T(); T_tensor.is_valid() && T_tensor.numel() >= 3) {
            const auto T_cpu = T_tensor.cpu().contiguous();
            const float* const ptr = T_cpu.ptr<float>();
            if (ptr) {
                view.world_to_camera_translation = glm::dvec3(
                    static_cast<double>(ptr[0]),
                    static_cast<double>(ptr[1]),
                    static_cast<double>(ptr[2]));
            }
        }

        view.intrinsics.fx = static_cast<double>(camera.focal_x());
        view.intrinsics.fy = static_cast<double>(camera.focal_y());
        view.intrinsics.cx = static_cast<double>(camera.center_x());
        view.intrinsics.cy = static_cast<double>(camera.center_y());
        view.intrinsics.width = camera.camera_width();
        view.intrinsics.height = camera.camera_height();
        view.intrinsics.lens_model = toGeometryLensModel(camera.camera_model_type());
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
            view.intrinsics.lens_model = toGeometryLensModel(source.model_type);
            view.intrinsics.num_distortion = std::clamp(source.num_distortion, 0, 12);
            std::copy_n(source.distortion, view.intrinsics.num_distortion, view.intrinsics.distortion.begin());
        }
        view.detected_image_width = view.intrinsics.width;
        view.detected_image_height = view.intrinsics.height;
        return view;
    }

    std::vector<geometry::apriltag::CameraObservationView> detectAprilTagsInCameras(
        const std::span<const std::shared_ptr<Camera>> cameras,
        const geometry::apriltag::AprilTagDetectorConfig& detector_config,
        const std::span<const glm::mat4> camera_scene_transforms,
        const int max_image_width,
        const AprilTagProgressCallback& progress_callback) {
        std::vector<AprilTagCameraInput> inputs;
        inputs.reserve(cameras.size());
        for (std::size_t i = 0; i < cameras.size(); ++i) {
            if (cameras[i]) {
                inputs.push_back({buildCameraObservationView(*cameras[i],
                                                             i < camera_scene_transforms.size() ? camera_scene_transforms[i] : glm::mat4(1.0f)),
                                  cameras[i]->image_path()});
            }
        }
        return detectAprilTagsInSnapshot(inputs, detector_config, max_image_width, progress_callback);
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
        if (cameras.empty()) {
            return views;
        }

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
                if (pixels && w > 0 && h > 0 && c > 0) {
                    view.detected_image_width = w;
                    view.detected_image_height = h;
                    const std::unique_ptr<unsigned char, decltype(&free_image)> image(pixels, &free_image);
                    auto& detector = detectors.local();
                    view.detections = detector.detectInterleaved(image.get(), w, h, c);
                } else if (pixels) {
                    free_image(pixels);
                }
            }

            views[idx] = std::move(view);

            completed_count.fetch_add(1, std::memory_order_relaxed);
            if (progress_callback) {
                std::lock_guard<std::mutex> lock(progress_mutex);
                if (!cancelled.load(std::memory_order_relaxed)) {
                    if (!progress_callback(completed_count.load(std::memory_order_relaxed), cameras.size(), cameras[idx].view.image_name)) {
                        cancelled.store(true, std::memory_order_relaxed);
                    }
                }
            }
        });

        if (cancelled.load(std::memory_order_relaxed)) {
            return {};
        }
        return views;
    }

    std::vector<geometry::apriltag::CameraObservationView> detectAprilTagsInScene(
        const Scene& scene,
        const geometry::apriltag::AprilTagDetectorConfig& detector_config,
        const bool active_cameras_only,
        const int max_image_width,
        const AprilTagProgressCallback& progress_callback) {
        const auto snapshot = snapshotAprilTagScene(scene, active_cameras_only);
        return detectAprilTagsInSnapshot(snapshot.cameras, detector_config, max_image_width, progress_callback);
    }

    bool saveCalibrationManifest(
        const std::filesystem::path& path,
        const geometry::apriltag::ScaleCalibrationManifest& manifest,
        std::string* const out_error) {
        if (path.empty()) {
            if (out_error) {
                *out_error = "Manifest path is empty";
            }
            return false;
        }

        if (const auto parent = path.parent_path(); !parent.empty()) {
            std::error_code ec;
            native_fs::create_directories(parent, ec);
            if (ec) {
                if (out_error) {
                    *out_error = "Failed to create manifest directory: " + ec.message();
                }
                return false;
            }
        }

        std::ofstream out;
        if (!open_file_for_write(path, std::ios::binary | std::ios::trunc, out)) {
            if (out_error) {
                *out_error = "Failed to open manifest file for writing: " + path_to_utf8(path);
            }
            return false;
        }

        const std::string json_text = geometry::apriltag::serializeManifestToJson(manifest, 2);
        out.write(json_text.data(), static_cast<std::streamsize>(json_text.size()));
        out.flush();
        if (!out.good()) {
            if (out_error) {
                *out_error = "Failed to write manifest contents to: " + path_to_utf8(path);
            }
            return false;
        }
        return true;
    }

    AprilTagSceneScaleReport solveAprilTagScaleFromViews(
        const std::span<const geometry::apriltag::CameraObservationView> views,
        const std::span<const geometry::apriltag::TagDistanceConstraint> constraints,
        const AprilTagSceneScaleOptions& options) {
        AprilTagSceneScaleReport report;
        report.views.assign(views.begin(), views.end());

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
            std::string save_err;
            if (saveCalibrationManifest(options.manifest_output_path, manifest, &save_err)) {
                report.manifest_saved_path = options.manifest_output_path;
            } else {
                LOG_WARN("Failed to save AprilTag calibration manifest: {}", save_err);
                if (report.error_message.empty()) {
                    report.error_message = save_err;
                }
            }
        }

        return report;
    }

    AprilTagSceneScaleReport computeAprilTagSceneScale(
        const Scene& scene,
        const std::span<const geometry::apriltag::TagDistanceConstraint> constraints,
        const AprilTagSceneScaleOptions& options,
        const AprilTagProgressCallback& progress_callback) {
        const auto views = detectAprilTagsInScene(
            scene,
            options.detector_config,
            options.active_cameras_only,
            options.max_image_width,
            progress_callback);
        if (views.empty()) {
            AprilTagSceneScaleReport report;
            report.error_message = "No camera views processed or detection was cancelled";
            return report;
        }
        return solveAprilTagScaleFromViews(views, constraints, options);
    }

} // namespace lfs::core
