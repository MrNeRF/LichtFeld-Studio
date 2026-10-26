/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "geometry/apriltag_detector.hpp"
#include "geometry/apriltag_scaling.hpp"

#include <algorithm>
#include <cmath>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

namespace {

    using namespace lfs::geometry::apriltag;

    CameraObservationView makeSyntheticCamera(
        const int uid,
        const std::string& name,
        const glm::dvec3& eye,
        const glm::dvec3& target,
        const glm::dvec3& up_hint = glm::dvec3(0.0, -1.0, 0.0),
        const int width = 1280,
        const int height = 960,
        const double fx = 900.0,
        const double fy = 900.0) {
        const glm::dvec3 forward = glm::normalize(target - eye);
        const glm::dvec3 right = glm::normalize(glm::cross(forward, up_hint));
        const glm::dvec3 down = glm::cross(forward, right);

        glm::dmat3 R_c2w(1.0);
        R_c2w[0] = right;
        R_c2w[1] = down;
        R_c2w[2] = forward;

        const glm::dmat3 R_w2c = glm::transpose(R_c2w);
        const glm::dvec3 T_w2c = -R_w2c * eye;

        CameraObservationView view;
        view.camera_uid = uid;
        view.image_name = name;
        view.world_to_camera_rotation = R_w2c;
        view.world_to_camera_translation = T_w2c;
        view.intrinsics.fx = fx;
        view.intrinsics.fy = fy;
        view.intrinsics.cx = static_cast<double>(width) * 0.5;
        view.intrinsics.cy = static_cast<double>(height) * 0.5;
        view.intrinsics.width = width;
        view.intrinsics.height = height;
        view.detected_image_width = width;
        view.detected_image_height = height;
        return view;
    }

    std::vector<std::uint8_t> renderSyntheticCameraImageWithTags(
        const CameraObservationView& cam,
        const AprilTagFamily family,
        const std::vector<std::pair<int, glm::dvec3>>& tag_world_centers,
        const double tag_world_half_size) {
        const int w = cam.intrinsics.width;
        const int h = cam.intrinsics.height;
        std::vector<std::uint8_t> canvas(static_cast<std::size_t>(w) * static_cast<std::size_t>(h), 200U);

        const glm::dmat3 R_c2w = glm::transpose(cam.world_to_camera_rotation);
        const glm::dvec3 cam_right = R_c2w[0];
        const glm::dvec3 cam_down = R_c2w[1];

        for (const auto& [tag_id, center_3d] : tag_world_centers) {
            const auto tag_bmp = renderAprilTagBitmap(family, tag_id, 1, 2);
            if (!tag_bmp || tag_bmp->width <= 0 || tag_bmp->height <= 0) {
                continue;
            }

            const auto center_px = projectWorldPointToDetectedPixel(cam, center_3d);
            if (!center_px) {
                continue;
            }

            const auto corner_tl = projectWorldPointToDetectedPixel(
                cam, center_3d - cam_right * tag_world_half_size - cam_down * tag_world_half_size);
            const auto corner_br = projectWorldPointToDetectedPixel(
                cam, center_3d + cam_right * tag_world_half_size + cam_down * tag_world_half_size);
            if (!corner_tl || !corner_br) {
                continue;
            }

            const int min_x = std::max(0, static_cast<int>(std::floor(std::min(corner_tl->x, corner_br->x))) - 2);
            const int max_x = std::min(w - 1, static_cast<int>(std::ceil(std::max(corner_tl->x, corner_br->x))) + 2);
            const int min_y = std::max(0, static_cast<int>(std::floor(std::min(corner_tl->y, corner_br->y))) - 2);
            const int max_y = std::min(h - 1, static_cast<int>(std::ceil(std::max(corner_tl->y, corner_br->y))) + 2);

            const glm::dvec3 plane_normal = R_c2w[2];
            for (int py = min_y; py <= max_y; ++py) {
                for (int px = min_x; px <= max_x; ++px) {
                    const auto ray = unprojectPixelToWorldRay(cam, glm::dvec2(px, py));
                    if (!ray) {
                        continue;
                    }
                    const double denom = glm::dot(ray->direction, plane_normal);
                    if (std::abs(denom) <= 1e-8) {
                        continue;
                    }
                    const double t = glm::dot(center_3d - ray->origin, plane_normal) / denom;
                    if (t <= 0.0) {
                        continue;
                    }
                    const glm::dvec3 hit = ray->origin + t * ray->direction;
                    const glm::dvec3 offset = hit - center_3d;
                    const double u_norm = (glm::dot(offset, cam_right) / (2.0 * tag_world_half_size)) + 0.5;
                    const double v_norm = (glm::dot(offset, cam_down) / (2.0 * tag_world_half_size)) + 0.5;
                    if (u_norm < 0.0 || u_norm >= 1.0 || v_norm < 0.0 || v_norm >= 1.0) {
                        continue;
                    }
                    const int tx = std::clamp(
                        static_cast<int>(std::floor(u_norm * static_cast<double>(tag_bmp->width))),
                        0,
                        tag_bmp->width - 1);
                    const int ty = std::clamp(
                        static_cast<int>(std::floor(v_norm * static_cast<double>(tag_bmp->height))),
                        0,
                        tag_bmp->height - 1);
                    canvas[static_cast<std::size_t>(py) * static_cast<std::size_t>(w) + static_cast<std::size_t>(px)] =
                        tag_bmp->pixels[static_cast<std::size_t>(ty) * static_cast<std::size_t>(tag_bmp->width) +
                                        static_cast<std::size_t>(tx)];
                }
            }
        }

        return canvas;
    }

} // namespace

TEST(AprilTagScalingTest, FamilyNameAndParsingRoundTrip) {
    EXPECT_EQ(parseAprilTagFamily("tag36h11"), AprilTagFamily::Tag36h11);
    EXPECT_EQ(parseAprilTagFamily("36h11"), AprilTagFamily::Tag36h11);
    EXPECT_EQ(parseAprilTagFamily("TAG_25H9"), AprilTagFamily::Tag25h9);
    EXPECT_EQ(parseAprilTagFamily("tag16h5"), AprilTagFamily::Tag16h5);
    EXPECT_EQ(parseAprilTagFamily("tag36h10"), AprilTagFamily::Tag36h10);
    EXPECT_EQ(parseAprilTagFamily("tagStandard41h12"), AprilTagFamily::TagStandard41h12);
    EXPECT_EQ(parseAprilTagFamily("tagCircle21h7"), AprilTagFamily::TagCircle21h7);
    EXPECT_EQ(parseAprilTagFamily("unknown_family"), std::nullopt);
}

TEST(AprilTagScalingTest, DetectorFindsRenderedTagInGrayscaleAndRgb) {
    const auto bmp = renderAprilTagBitmap(AprilTagFamily::Tag36h11, 7, 10, 3);
    ASSERT_TRUE(bmp.has_value());

    AprilTagDetectorConfig cfg;
    cfg.family = AprilTagFamily::Tag36h11;
    cfg.quad_decimate = 1.0f;
    AprilTagDetector detector(cfg);

    const auto gray_dets = detector.detectGrayscale(bmp->pixels.data(), bmp->width, bmp->height);
    ASSERT_EQ(gray_dets.size(), 1U);
    EXPECT_EQ(gray_dets[0].id, 7);
    EXPECT_EQ(gray_dets[0].hamming, 0);
    const double expected_center = static_cast<double>(bmp->width - 1) * 0.5;
    EXPECT_NEAR(gray_dets[0].center.x, expected_center, 0.75);
    EXPECT_NEAR(gray_dets[0].center.y, expected_center, 0.75);

    std::vector<std::uint8_t> rgb(bmp->pixels.size() * 3);
    for (std::size_t i = 0; i < bmp->pixels.size(); ++i) {
        rgb[i * 3 + 0] = bmp->pixels[i];
        rgb[i * 3 + 1] = bmp->pixels[i];
        rgb[i * 3 + 2] = bmp->pixels[i];
    }
    const auto rgb_dets = detector.detectInterleaved(rgb.data(), bmp->width, bmp->height, 3);
    ASSERT_EQ(rgb_dets.size(), 1U);
    EXPECT_EQ(rgb_dets[0].id, 7);
    EXPECT_NEAR(rgb_dets[0].center.x, gray_dets[0].center.x, 1e-4);
    EXPECT_NEAR(rgb_dets[0].center.y, gray_dets[0].center.y, 1e-4);
}

TEST(AprilTagScalingTest, LensDistortionRoundTripMatchesModels) {
    const glm::dvec2 pt{0.23, -0.17};

    const std::array<double, 5> pinhole_coeffs{-0.12, 0.03, -0.005, 0.001, -0.002};
    const glm::dvec2 pinhole_dist = applyForwardDistortion(pt, lfs::core::CameraModelType::PINHOLE, pinhole_coeffs);
    const auto pinhole_inv = invertDistortion(pinhole_dist, lfs::core::CameraModelType::PINHOLE, pinhole_coeffs);
    ASSERT_TRUE(pinhole_inv.has_value());
    EXPECT_NEAR(pinhole_inv->x, pt.x, 1e-7);
    EXPECT_NEAR(pinhole_inv->y, pt.y, 1e-7);

    const std::array<double, 4> fisheye_coeffs{0.05, -0.02, 0.004, -0.001};
    const glm::dvec2 fisheye_dist = applyForwardDistortion(pt, lfs::core::CameraModelType::FISHEYE, fisheye_coeffs);
    const auto fisheye_inv = invertDistortion(fisheye_dist, lfs::core::CameraModelType::FISHEYE, fisheye_coeffs);
    ASSERT_TRUE(fisheye_inv.has_value());
    EXPECT_NEAR(fisheye_inv->x, pt.x, 1e-7);
    EXPECT_NEAR(fisheye_inv->y, pt.y, 1e-7);

    const std::array<double, 10> thin_prism_coeffs{0.03, -0.01, 0.002, 0.0, 0.001, -0.001, 0.002, 0.0, -0.001, 0.0};
    const glm::dvec2 tp_dist = applyForwardDistortion(pt, lfs::core::CameraModelType::THIN_PRISM_FISHEYE, thin_prism_coeffs);
    const auto tp_inv = invertDistortion(tp_dist, lfs::core::CameraModelType::THIN_PRISM_FISHEYE, thin_prism_coeffs);
    ASSERT_TRUE(tp_inv.has_value());
    EXPECT_NEAR(tp_inv->x, pt.x, 1e-7);
    EXPECT_NEAR(tp_inv->y, pt.y, 1e-7);
}

TEST(AprilTagScalingTest, TriangulatesMultiViewAndRejectsOutlierObservation) {
    const std::vector<std::pair<int, glm::dvec3>> ground_truth_tags = {
        {1, {-1.0, 0.2, 4.0}},
        {2, {1.0, 0.2, 4.0}},
        {3, {0.0, 1.5, 4.5}},
    };

    std::vector<CameraObservationView> views = {
        makeSyntheticCamera(10, "cam_0.png", {-2.5, 0.0, 0.0}, {0.0, 0.5, 4.0}),
        makeSyntheticCamera(11, "cam_1.png", {0.0, -2.0, 0.0}, {0.0, 0.5, 4.0}),
        makeSyntheticCamera(12, "cam_2.png", {2.5, 0.0, 0.0}, {0.0, 0.5, 4.0}),
        makeSyntheticCamera(13, "cam_3.png", {0.0, 2.0, 0.5}, {0.0, 0.5, 4.0}),
    };

    for (auto& view : views) {
        for (const auto& [tag_id, pos] : ground_truth_tags) {
            const auto px = projectWorldPointToDetectedPixel(view, pos);
            ASSERT_TRUE(px.has_value());
            TagDetection2D det;
            det.id = tag_id;
            det.hamming = 0;
            det.decision_margin = 80.0f;
            det.center = *px;
            view.detections.push_back(det);
        }
    }

    // Corrupt tag 1 observation in cam_3 by +120 pixels so it acts as a gross outlier.
    views[3].detections[0].center += glm::dvec2(120.0, -95.0);

    // Add tag 99 seen in only 1 camera to verify rejection diagnostic.
    views[0].detections.push_back(TagDetection2D{
        .id = 99,
        .hamming = 0,
        .decision_margin = 90.0f,
        .center = {640.0, 480.0},
    });

    const auto summary = triangulateAprilTags(views);
    ASSERT_EQ(summary.tags.size(), 3U);
    ASSERT_EQ(summary.rejected_tags.size(), 1U);
    EXPECT_EQ(summary.rejected_tags[0].tag_id, 99);

    for (std::size_t i = 0; i < ground_truth_tags.size(); ++i) {
        EXPECT_EQ(summary.tags[i].tag_id, ground_truth_tags[i].first);
        EXPECT_NEAR(glm::length(summary.tags[i].position - ground_truth_tags[i].second), 0.0, 1e-6);
    }

    // Tag 1 had 4 total observations, 1 outlier rejected -> 3 inlier observations.
    EXPECT_EQ(summary.tags[0].total_observations, 4);
    EXPECT_EQ(summary.tags[0].inlier_observations, 3);
}

TEST(AprilTagScalingTest, RansacScaleSolverRejectsOutlierDistanceConstraints) {
    // True scale factor is 2.5 (unscaled positions are 1 / 2.5 = 0.4 of real world).
    constexpr double kTrueScale = 2.5;
    const std::vector<TriangulatedTag3D> tags = {
        {.tag_id = 1, .position = {0.0, 0.0, 0.0}},
        {.tag_id = 2, .position = {5.0 / kTrueScale, 0.0, 0.0}},
        {.tag_id = 3, .position = {0.0, 8.0 / kTrueScale, 0.0}},
        {.tag_id = 4, .position = {0.0, 0.0, 3.5 / kTrueScale}},
        {.tag_id = 5, .position = {6.0 / kTrueScale, 8.0 / kTrueScale, 0.0}},
    };

    // 4 accurate constraints + 2 gross user errors.
    const std::vector<TagDistanceConstraint> constraints = {
        {.tag_a = 1, .tag_b = 2, .target_distance = 5.0},
        {.tag_a = 1, .tag_b = 3, .target_distance = 8.0},
        {.tag_a = 1, .tag_b = 4, .target_distance = 3.5},
        {.tag_a = 1, .tag_b = 5, .target_distance = 10.0},
        {.tag_a = 2, .tag_b = 3, .target_distance = 50.0}, // Typo outlier (50m instead of ~9.43m)
        {.tag_a = 3, .tag_b = 4, .target_distance = 1.1},  // Typo outlier
    };

    ScaleSolverConfig cfg;
    cfg.inlier_relative_threshold = 0.02;
    cfg.min_inliers = 3;
    cfg.min_inlier_ratio = 0.5;

    const auto res = solveGlobalScale(tags, constraints, cfg);
    ASSERT_TRUE(res.success) << res.error_message;
    EXPECT_NEAR(res.scale_factor, kTrueScale, 1e-6);
    EXPECT_EQ(res.total_constraints, 6);
    EXPECT_EQ(res.valid_constraints, 6);
    EXPECT_EQ(res.inlier_count, 4);
    EXPECT_NEAR(res.inlier_ratio, 4.0 / 6.0, 1e-6);
    EXPECT_NEAR(res.rmse, 0.0, 1e-6);
    EXPECT_NEAR(res.mae, 0.0, 1e-6);

    EXPECT_TRUE(res.evaluated_constraints[0].inlier);
    EXPECT_TRUE(res.evaluated_constraints[1].inlier);
    EXPECT_TRUE(res.evaluated_constraints[2].inlier);
    EXPECT_TRUE(res.evaluated_constraints[3].inlier);
    EXPECT_FALSE(res.evaluated_constraints[4].inlier);
    EXPECT_FALSE(res.evaluated_constraints[5].inlier);
}

TEST(AprilTagScalingTest, EndToEndPhotogrammetricDetectionTriangulationAndRescaling) {
    // Place 4 AprilTags in unscaled reconstruction coordinates where 1 unit = 2.0 meters.
    // Tag 1 -> Tag 2: unscaled distance = 2.5 -> real target = 5.0 m
    // Tag 3 -> Tag 4: unscaled distance = 4.0 -> real target = 8.0 m
    // Tag 1 -> Tag 3: unscaled distance = 1.5 -> real target = 3.0 m
    const std::vector<std::pair<int, glm::dvec3>> world_tags = {
        {1, {-1.25, -0.75, 5.0}},
        {2, {1.25, -0.75, 5.0}},
        {3, {-1.25, 0.75, 5.0}},
        {4, {2.75, 0.75, 5.0}},
    };

    std::vector<CameraObservationView> cameras = {
        makeSyntheticCamera(1, "view_left.png", {-1.8, 0.0, 0.0}, {0.2, 0.0, 5.0}, {0.0, -1.0, 0.0}, 1280, 960, 1000.0, 1000.0),
        makeSyntheticCamera(2, "view_center.png", {0.2, -1.2, 0.0}, {0.2, 0.0, 5.0}, {0.0, -1.0, 0.0}, 1280, 960, 1000.0, 1000.0),
        makeSyntheticCamera(3, "view_right.png", {2.2, 0.0, 0.0}, {0.2, 0.0, 5.0}, {0.0, -1.0, 0.0}, 1280, 960, 1000.0, 1000.0),
        makeSyntheticCamera(4, "view_top.png", {0.2, 1.2, 0.2}, {0.2, 0.0, 5.0}, {0.0, -1.0, 0.0}, 1280, 960, 1000.0, 1000.0),
    };

    AprilTagDetectorConfig det_cfg;
    det_cfg.family = AprilTagFamily::Tag36h11;
    det_cfg.quad_decimate = 1.0f;
    AprilTagDetector detector(det_cfg);

    for (auto& cam : cameras) {
        const auto img = renderSyntheticCameraImageWithTags(
            cam,
            AprilTagFamily::Tag36h11,
            world_tags,
            0.35);
        cam.detections = detector.detectGrayscale(img.data(), cam.intrinsics.width, cam.intrinsics.height);
        EXPECT_EQ(cam.detections.size(), 4U) << "Camera " << cam.image_name << " missed tags";
    }

    const auto triangulation = triangulateAprilTags(cameras);
    ASSERT_EQ(triangulation.tags.size(), 4U);
    for (std::size_t i = 0; i < world_tags.size(); ++i) {
        EXPECT_EQ(triangulation.tags[i].tag_id, world_tags[i].first);
        EXPECT_LT(glm::length(triangulation.tags[i].position - world_tags[i].second), 0.01);
        EXPECT_LT(triangulation.tags[i].mean_reprojection_error_px, 1.0);
    }

    const std::vector<TagDistanceConstraint> constraints = {
        {.tag_a = 1, .tag_b = 2, .target_distance = 5.0},
        {.tag_a = 3, .tag_b = 4, .target_distance = 8.0},
        {.tag_a = 1, .tag_b = 3, .target_distance = 3.0},
        {.tag_a = 2, .tag_b = 4, .target_distance = 25.0}, // Outlier constraint
    };

    const auto solve_res = solveGlobalScale(triangulation.tags, constraints);
    ASSERT_TRUE(solve_res.success) << solve_res.error_message;
    EXPECT_NEAR(solve_res.scale_factor, 2.0, 0.01);
    EXPECT_EQ(solve_res.inlier_count, 3);
    EXPECT_NEAR(solve_res.inlier_ratio, 0.75, 1e-6);
    EXPECT_LT(solve_res.rmse, 0.02);

    // Verify applying scale_transform rescales the 3D distance between Tag 1 and Tag 2 to 5.0m
    // and between Tag 3 and Tag 4 to 8.0m.
    const glm::vec3 p1_scaled = glm::vec3(solve_res.scale_transform * glm::vec4(glm::vec3(triangulation.tags[0].position), 1.0f));
    const glm::vec3 p2_scaled = glm::vec3(solve_res.scale_transform * glm::vec4(glm::vec3(triangulation.tags[1].position), 1.0f));
    const glm::vec3 p3_scaled = glm::vec3(solve_res.scale_transform * glm::vec4(glm::vec3(triangulation.tags[2].position), 1.0f));
    const glm::vec3 p4_scaled = glm::vec3(solve_res.scale_transform * glm::vec4(glm::vec3(triangulation.tags[3].position), 1.0f));

    EXPECT_NEAR(glm::length(p2_scaled - p1_scaled), 5.0f, 0.02f);
    EXPECT_NEAR(glm::length(p4_scaled - p3_scaled), 8.0f, 0.02f);
    EXPECT_NEAR(glm::length(p3_scaled - p1_scaled), 3.0f, 0.02f);

    // Verify manifest JSON serialization.
    ScaleCalibrationManifest manifest;
    manifest.family = AprilTagFamily::Tag36h11;
    manifest.triangulation = triangulation;
    manifest.scale_result = solve_res;
    manifest.timestamp_utc = "2026-10-26T02:00:00Z";

    const std::string json_str = serializeManifestToJson(manifest);
    const auto parsed = nlohmann::json::parse(json_str);
    EXPECT_EQ(parsed["schema_version"], 1);
    EXPECT_EQ(parsed["tag_family"], "tag36h11");
    EXPECT_TRUE(parsed["scaling_summary"]["success"].get<bool>());
    EXPECT_EQ(parsed["scaling_summary"]["inlier_count"], 3);
    EXPECT_NEAR(parsed["scaling_summary"]["inlier_ratio"].get<double>(), 0.75, 1e-6);
    EXPECT_EQ(parsed["triangulated_tags"].size(), 4U);
    EXPECT_EQ(parsed["evaluated_constraints"].size(), 4U);
}

TEST(AprilTagScalingTest, RobustInitializationSurvivesSixtyFiveViews) {
    std::vector<CameraObservationView> views;
    const glm::dvec3 point(0.0, 0.0, 5.0);
    for (int i = 0; i < 63; ++i) {
        const double x = -0.1 + 0.2 * i / 62.0;
        auto view = makeSyntheticCamera(i, "camera", {x, 0.0, 0.0}, {x, 0.0, 5.0},
                                        {0.0, -1.0, 0.0}, 1280, 960, 1000.0, 1000.0);
        view.detections.push_back({.id = 1, .decision_margin = 80.0f, .center = *projectWorldPointToDetectedPixel(view, point)});
        views.push_back(std::move(view));
    }
    auto outlier = makeSyntheticCamera(63, "outlier", {100.0, 0.0, 0.0}, {0.0, 0.0, 10.0});
    outlier.detections.push_back({.id = 1, .decision_margin = 80.0f, .center = *projectWorldPointToDetectedPixel(outlier, {0.0, 0.0, 10.0})});
    views.push_back(std::move(outlier));
    TriangulationConfig config;
    config.min_parallax_angle_deg = 0.1;
    for (int count : {64, 65}) {
        if (count == 65) {
            auto extra = makeSyntheticCamera(64, "extra", {0.11, 0.0, 0.0}, {0.11, 0.0, 5.0});
            extra.detections.push_back({.id = 1, .decision_margin = 80.0f, .center = *projectWorldPointToDetectedPixel(extra, point)});
            views.push_back(std::move(extra));
        }
        for (bool reverse : {false, true}) {
            SCOPED_TRACE(::testing::Message() << count << " views, reversed=" << reverse);
            if (reverse)
                std::reverse(views.begin(), views.end());
            const auto result = triangulateAprilTags(views, config);
            ASSERT_EQ(result.tags.size(), 1u);
            EXPECT_NEAR(result.tags[0].position.z, 5.0, 1e-6);
            EXPECT_EQ(result.tags[0].inlier_observations, count - 1);
            if (reverse)
                std::reverse(views.begin(), views.end());
        }
    }
}

TEST(AprilTagScalingTest, FinalFitReclassifiesInliersAndRechecksThresholds) {
    const std::vector<TriangulatedTag3D> tags{
        {.tag_id = 0, .position = {0, 0, 0}},
        {.tag_id = 1, .position = {1, 0, 0}},
        {.tag_id = 2, .position = {2, 0, 0}},
        {.tag_id = 3, .position = {3, 0, 0}}};
    const std::vector<TagDistanceConstraint> constraints{
        {.tag_a = 0, .tag_b = 1, .target_distance = 1.0},
        {.tag_a = 0, .tag_b = 2, .target_distance = 2.078},
        {.tag_a = 0, .tag_b = 3, .target_distance = 3.117}};
    ScaleSolverConfig config;
    config.inlier_relative_threshold = 0.02;
    for (const auto [minimum, ratio] : {std::pair{3, 0.5}, std::pair{1, 0.9}, std::pair{2, 0.5}}) {
        config.min_inliers = minimum;
        config.min_inlier_ratio = ratio;
        const auto result = solveGlobalScale(tags, constraints, config);
        EXPECT_EQ(result.success, minimum == 2);
        EXPECT_EQ(result.inlier_count, 2);
        EXPECT_FALSE(result.evaluated_constraints[0].inlier);
        for (const auto& constraint : result.evaluated_constraints) {
            EXPECT_EQ(constraint.inlier, constraint.relative_error <= 0.02);
        }
    }
}

TEST(AprilTagScalingTest, OrthographicProjectionAndParallelRays) {
    auto view = makeSyntheticCamera(1, "ortho", {0, 0, 0}, {0, 0, 1},
                                    {0, -1, 0}, 1280, 960, 1000, 1000);
    view.intrinsics.lens_model = lfs::core::CameraModelType::ORTHO;
    for (const double depth : {2.0, 4.0}) {
        const auto pixel = projectWorldPointToDetectedPixel(view, {1, 0, depth});
        ASSERT_TRUE(pixel);
        EXPECT_DOUBLE_EQ(pixel->x, 1640.0);
        EXPECT_DOUBLE_EQ(pixel->y, 480.0);
        const auto ray = unprojectPixelToWorldRay(view, *pixel);
        ASSERT_TRUE(ray);
        EXPECT_NEAR(glm::length(ray->origin - glm::dvec3(1, 0, 0)), 0.0, 1e-12);
        EXPECT_NEAR(glm::length(ray->direction - glm::dvec3(0, 0, 1)), 0.0, 1e-12);
    }
    const auto left = unprojectPixelToWorldRay(view, {640, 480});
    const auto right = unprojectPixelToWorldRay(view, {1640, 480});
    ASSERT_TRUE(left && right);
    EXPECT_EQ(left->direction, right->direction);
    EXPECT_NE(left->origin, right->origin);

    view.detected_image_width = 640;
    view.detected_image_height = 480;
    view.scene_world_transform = glm::translate(glm::dmat4(1), glm::dvec3(2, 3, 4)) *
                                 glm::rotate(glm::dmat4(1), 0.4, glm::dvec3(0, 1, 0));
    const auto world = glm::dvec3(view.scene_world_transform * glm::dvec4(1, 0, 4, 1));
    const auto pixel = projectWorldPointToDetectedPixel(view, world);
    ASSERT_TRUE(pixel);
    EXPECT_NEAR(pixel->x, 820.0, 1e-9);
    const auto ray = unprojectPixelToWorldRay(view, *pixel);
    ASSERT_TRUE(ray);
    EXPECT_NEAR(glm::length(glm::cross(world - ray->origin, ray->direction)), 0.0, 1e-9);
}
