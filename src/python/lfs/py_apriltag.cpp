/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_apriltag.hpp"
#include "py_error.hpp"

#include "core/event_bridge/localization_manager.hpp"
#include "python/python_runtime.hpp"
#include "visualizer/apriltag_calibration.hpp"
#include "visualizer/gui/gui_manager.hpp"
#include "visualizer/post_work_utils.hpp"

#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <nlohmann/json.hpp>

namespace lfs::python {

    namespace {

        vis::Visualizer& april_tag_worker_viewer() {
            auto* viewer = get_visualizer();
            if (!viewer)
                throw std::runtime_error("Viewer not available");
            if (viewer->isOnViewerThread())
                throw std::runtime_error(LOC("runtime.apriltag_worker_required"));
            return *viewer;
        }

        vis::AprilTagSceneApplyResult run_april_tag_scan(
            vis::Visualizer& viewer,
            const vis::AprilTagSceneScaleOptions& options,
            const std::vector<geometry::apriltag::TagDistanceConstraint>& constraints = {},
            const bool solve_scale = false, const bool apply_to_scene = false,
            const std::vector<std::string>& targets = {}) {
            nb::gil_scoped_release release;
            const core::TaskContext context{
                .name = "python.apriltag",
                .domain = ErrorDomain::Python,
                .operation_id = OperationId::generate(),
                .site = LFS_SOURCE_SITE_CURRENT(),
            };
            using Request = vis::gui::AsyncTaskManager::AprilTagScanRequest;
            auto started = vis::post_guarded_and_wait<Request>(
                viewer, context, [&]() -> Result<Request> {
                    auto* gui = get_gui_manager();
                    if (!gui)
                        throw std::runtime_error("GUI manager not initialized");
                    auto request = gui->asyncTasks().startAprilTagScan(options, constraints, targets, solve_scale, apply_to_scene);
                    if (!request)
                        return core::failure_result<Request>(make_error(ErrorInit{
                            .code = ErrorCode::Unavailable,
                            .domain = ErrorDomain::Python,
                            .user_message = std::move(request.error()),
                            .detection = LFS_SOURCE_SITE_CURRENT(),
                        }));
                    return std::move(*request);
                },
                make_error(ErrorInit{
                    .code = ErrorCode::Cancelled,
                    .domain = ErrorDomain::Python,
                    .severity = Severity::Warning,
                    .detail = "Viewer is shutting down",
                    .detection = LFS_SOURCE_SITE_CURRENT(),
                }));
            return unwrap(std::move(started)).completion.get();
        }

    } // namespace

    void register_apriltag(nb::module_& m) {
        m.def(
            "detect_scene_apriltags",
            [](const std::string& family,
               const int min_views,
               const double min_parallax_deg,
               const double max_reprojection_error_px,
               const int max_hamming,
               const float min_decision_margin,
               const bool active_cameras_only,
               const int max_image_width) -> nb::dict {
                auto& viewer = april_tag_worker_viewer();

                const auto options = lfs::vis::aprilTagSceneScaleOptions({
                    {"family", family},
                    {"min_views", min_views},
                    {"min_parallax_deg", min_parallax_deg},
                    {"max_reprojection_error_px", max_reprojection_error_px},
                    {"max_hamming", max_hamming},
                    {"min_decision_margin", min_decision_margin},
                    {"active_cameras_only", active_cameras_only},
                    {"max_image_width", max_image_width},
                });
                const auto result = run_april_tag_scan(viewer, options);
                if (!result.scale_report.success)
                    throw std::runtime_error(result.error_message);
                const auto output = lfs::vis::aprilTagResultJson(result);
                const nlohmann::json detection = {
                    {"camera_count", output["camera_count"]},
                    {"triangulated_tags", output["triangulated_tags"]},
                    {"rejected_tags", output["rejected_tags"]},
                };
                auto out = nb::cast<nb::dict>(nb::module_::import_("json").attr("loads")(detection.dump()));
                const auto tuple = nb::module_::import_("builtins").attr("tuple");
                for (auto item : nb::cast<nb::list>(out["triangulated_tags"])) {
                    auto tag = nb::cast<nb::dict>(item);
                    tag["position"] = tuple(tag["position"]);
                    tag["visualizer_position"] = tuple(tag["visualizer_position"]);
                }
                return out;
            },
            nb::arg("family") = "tag36h11",
            nb::arg("min_views") = 2,
            nb::arg("min_parallax_deg") = 1.5,
            nb::arg("max_reprojection_error_px") = 3.0,
            nb::arg("max_hamming") = 1,
            nb::arg("min_decision_margin") = 25.0f,
            nb::arg("active_cameras_only") = false,
            nb::arg("max_image_width") = 0,
            "Detect and triangulate AprilTags from a background editor script; viewer-thread calls are rejected");

        m.def(
            "scale_scene_by_apriltags",
            [](const std::vector<std::tuple<int, int, double>>& constraints_in,
               const std::string& family,
               const double inlier_relative_threshold,
               const double inlier_absolute_threshold,
               const int min_inliers,
               const double min_inlier_ratio,
               const int min_views,
               const double min_parallax_deg,
               const double max_reprojection_error_px,
               const int max_hamming,
               const float min_decision_margin,
               const bool active_cameras_only,
               const int max_image_width,
               const std::string& manifest_path,
               const bool apply_to_scene,
               const std::vector<std::string>& targets) -> nb::dict {
                auto& viewer = april_tag_worker_viewer();

                std::vector<lfs::geometry::apriltag::TagDistanceConstraint> constraints;
                constraints.reserve(constraints_in.size());
                for (const auto& [tag_a, tag_b, distance] : constraints_in)
                    constraints.push_back({tag_a, tag_b, distance});
                const auto options = lfs::vis::aprilTagSceneScaleOptions({
                    {"family", family},
                    {"min_views", min_views},
                    {"min_parallax_deg", min_parallax_deg},
                    {"max_reprojection_error_px", max_reprojection_error_px},
                    {"max_hamming", max_hamming},
                    {"min_decision_margin", min_decision_margin},
                    {"active_cameras_only", active_cameras_only},
                    {"max_image_width", max_image_width},
                    {"manifest_path", manifest_path},
                    {"inlier_relative_threshold", inlier_relative_threshold},
                    {"inlier_absolute_threshold", inlier_absolute_threshold},
                    {"min_inliers", min_inliers},
                    {"min_inlier_ratio", min_inlier_ratio},
                });
                const auto result = run_april_tag_scan(viewer, options, constraints, true, apply_to_scene, targets);
                const auto output = lfs::vis::aprilTagResultJson(result);
                nlohmann::json scaling;
                for (const auto* key : {"success", "applied", "error_message", "scale_factor", "pivot",
                                        "total_constraints", "valid_constraints", "inlier_count", "inlier_ratio",
                                        "rmse", "mae", "max_error", "mean_relative_error", "transformed_nodes",
                                        "manifest_json", "manifest_saved_path"})
                    scaling[key] = output[key];
                scaling["success"] = apply_to_scene ? result.applied : result.scale_report.success;
                auto out = nb::cast<nb::dict>(nb::module_::import_("json").attr("loads")(scaling.dump()));
                const auto& pivot = result.scale_report.scale_result.pivot;
                out["pivot"] = nb::make_tuple(pivot.x, pivot.y, pivot.z);
                return out;
            },
            nb::arg("constraints"),
            nb::arg("family") = "tag36h11",
            nb::arg("inlier_relative_threshold") = 0.02,
            nb::arg("inlier_absolute_threshold") = 0.0,
            nb::arg("min_inliers") = 1,
            nb::arg("min_inlier_ratio") = 0.5,
            nb::arg("min_views") = 2,
            nb::arg("min_parallax_deg") = 1.5,
            nb::arg("max_reprojection_error_px") = 3.0,
            nb::arg("max_hamming") = 1,
            nb::arg("min_decision_margin") = 25.0f,
            nb::arg("active_cameras_only") = false,
            nb::arg("max_image_width") = 0,
            nb::arg("manifest_path") = "",
            nb::arg("apply_to_scene") = true,
            nb::arg("targets") = std::vector<std::string>{},
            "Calibrate the complete scene from a background editor script. Explicit targets allow partial scaling; viewer-thread calls are rejected");
    }

} // namespace lfs::python
