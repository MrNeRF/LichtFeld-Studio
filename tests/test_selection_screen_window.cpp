/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering/render_constants.hpp"
#include "rendering/selection_ops.hpp"

#include "core/tensor.hpp"

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <tuple>
#include <vector>

namespace {

    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::Tensor;
    using lfs::rendering::ScreenWindowCameraModel;

    constexpr int kWidth = 1280;
    constexpr int kHeight = 720;
    constexpr float kPixelFocalX = 910.0f;
    constexpr float kPixelFocalY = 900.0f;
    constexpr float kOrthoScale = 42.0f;
    constexpr std::size_t kRandomPointCount = 10'000;

    struct ScreenWindowConfig {
        ScreenWindowCameraModel camera_model = ScreenWindowCameraModel::Pinhole;
        int width = kWidth;
        int height = kHeight;
        float pixel_focal_x = kPixelFocalX;
        float pixel_focal_y = kPixelFocalY;
        float center_x = 0.5f * static_cast<float>(kWidth);
        float center_y = 0.5f * static_cast<float>(kHeight);
        float ortho_scale = kOrthoScale;
        float near_depth = 0.25f;
        float far_depth = 20.0f;
        float scale = 0.35f;
        float offset_x = 0.0f;
        float offset_y = 0.0f;
    };

    struct ProjectedPoint {
        float px = 0.0f;
        float py = 0.0f;
        float depth = 0.0f;
    };

    constexpr std::array<float, 9> kViewRotationRows{
        0.9393727f,
        0.0f,
        -0.3428978f,
        0.0f,
        1.0f,
        0.0f,
        0.3428978f,
        0.0f,
        0.9393727f,
    };
    constexpr std::array<float, 3> kTranslation{1.0f, -2.0f, 0.5f};

    std::array<float, 3> transformPoint(
        const std::array<float, 3>& point,
        const std::vector<float>* const model_transforms,
        const int transform_index) {
        if (!model_transforms || model_transforms->empty()) {
            return point;
        }
        const int transform_count = static_cast<int>(model_transforms->size() / 16);
        const int clamped_index = std::clamp(transform_index, 0, transform_count - 1);
        const float* const m = model_transforms->data() + clamped_index * 16;
        return {
            m[0] * point[0] + m[1] * point[1] + m[2] * point[2] + m[3],
            m[4] * point[0] + m[5] * point[1] + m[6] * point[2] + m[7],
            m[8] * point[0] + m[9] * point[1] + m[10] * point[2] + m[11],
        };
    }

    ProjectedPoint projectCpu(
        const std::array<float, 3>& input_point,
        const ScreenWindowConfig& config,
        const std::array<float, 9>& view_rotation_rows,
        const std::array<float, 3>& translation,
        const std::vector<float>* const model_transforms = nullptr,
        const int transform_index = 0) {
        const auto point = transformPoint(input_point, model_transforms, transform_index);
        const float dx = point[0] - translation[0];
        const float dy = point[1] - translation[1];
        const float dz = point[2] - translation[2];
        const float visualizer_x =
            view_rotation_rows[0] * dx + view_rotation_rows[1] * dy + view_rotation_rows[2] * dz;
        const float visualizer_y =
            view_rotation_rows[3] * dx + view_rotation_rows[4] * dy + view_rotation_rows[5] * dz;
        const float visualizer_z =
            view_rotation_rows[6] * dx + view_rotation_rows[7] * dy + view_rotation_rows[8] * dz;

        const float view_x = visualizer_x;
        const float view_y = -visualizer_y;
        const float view_z = -visualizer_z;
        const float width = static_cast<float>(config.width);
        const float height = static_cast<float>(config.height);

        ProjectedPoint projected{.depth = view_z};
        if (config.camera_model == ScreenWindowCameraModel::Pinhole) {
            projected.px = config.center_x + config.pixel_focal_x * view_x / view_z;
            projected.py = config.center_y + config.pixel_focal_y * view_y / view_z;
        } else if (config.camera_model == ScreenWindowCameraModel::Orthographic) {
            projected.px = 0.5f * width + config.ortho_scale * view_x;
            projected.py = 0.5f * height + config.ortho_scale * view_y;
        } else {
            const float len = std::sqrt(view_x * view_x + view_y * view_y + view_z * view_z);
            if (len <= 1.0e-6f || !std::isfinite(len)) {
                projected.depth = -1.0f;
            } else {
                const float dir_x = view_x / len;
                const float dir_y = view_y / len;
                const float dir_z = view_z / len;
                constexpr float pi = 3.14159265358979323846f;
                projected.px = (std::atan2(dir_x, dir_z) / (2.0f * pi) + 0.5f) * width;
                projected.py =
                    (std::asin(std::clamp(dir_y, -1.0f, 1.0f)) / pi + 0.5f) * height;
                projected.depth = len;
            }
        }
        return projected;
    }

    bool cpuReferenceInside(const ProjectedPoint& projected, const ScreenWindowConfig& config) {
        const float width = static_cast<float>(config.width);
        const float height = static_cast<float>(config.height);

        // KEEP IN SYNC: the screen-window formula lives in four places — this CPU
        // reference, vertex_shader.slang compute_splat_active_state,
        // filterSelectionByScreenWindowKernel (selection_ops.cu), and (rect only,
        // no depth test) the 2D overlay in gui_manager.cpp
        // appendScreenWindowOverlay.
        // Contract: the window RECTANGLE is framebuffer-centred in all four copies;
        // the splat projection uses the displayed camera's real unjittered intrinsics;
        // jitter moves the draw sample, not the containment boundary.
        // Coverage warning: the slang copy is still uncompared, but the host packing
        // that feeds it is covered by the overlay-slot test.
        const float half_w = 0.5f * config.scale * width;
        const float half_h = 0.5f * config.scale * height;
        const float cx = 0.5f * width + config.offset_x * (0.5f * width - half_w);
        const float cy = 0.5f * height + config.offset_y * (0.5f * height - half_h);
        const bool inside_rect = std::abs(projected.px - cx) <= half_w &&
                                 std::abs(projected.py - cy) <= half_h;
        return inside_rect && projected.depth >= config.near_depth &&
               projected.depth <= config.far_depth && projected.depth > 0.0f;
    }

    float boundaryTolerance(const float scale_of_quantity) {
        return std::max(1.0e-4f, 32.0f * FLT_EPSILON * std::max(std::abs(scale_of_quantity), 1.0f));
    }

    bool nearDecisionBoundary(const ProjectedPoint& projected, const ScreenWindowConfig& config) {
        const float width = static_cast<float>(config.width);
        const float height = static_cast<float>(config.height);
        const float half_w = 0.5f * config.scale * width;
        const float half_h = 0.5f * config.scale * height;
        const float cx = 0.5f * width + config.offset_x * (0.5f * width - half_w);
        const float cy = 0.5f * height + config.offset_y * (0.5f * height - half_h);
        const float x_distance = std::abs(std::abs(projected.px - cx) - half_w);
        const float y_distance = std::abs(std::abs(projected.py - cy) - half_h);
        const float depth_scale = std::max(
            {std::abs(projected.depth), std::abs(config.near_depth), std::abs(config.far_depth), 1.0f});

        if (x_distance < boundaryTolerance(width) || y_distance < boundaryTolerance(height) ||
            std::abs(projected.depth - config.near_depth) < boundaryTolerance(depth_scale) ||
            std::abs(projected.depth - config.far_depth) < boundaryTolerance(depth_scale) ||
            std::abs(projected.depth) < boundaryTolerance(depth_scale)) {
            return true;
        }
        if (config.camera_model == ScreenWindowCameraModel::Equirectangular) {
            const float seam_distance = std::min(std::abs(projected.px), std::abs(width - projected.px));
            return seam_distance < boundaryTolerance(width);
        }
        return false;
    }

    Tensor cudaMeans(const std::vector<float>& points) {
        return Tensor::from_vector(points, {points.size() / 3, std::size_t{3}}, Device::CUDA);
    }

    std::vector<bool> runCuda(
        const std::vector<float>& points,
        const ScreenWindowConfig& config,
        const std::array<float, 9>& view_rotation_rows,
        const std::array<float, 3>& translation,
        const Tensor* const model_transforms = nullptr,
        const Tensor* const transform_indices = nullptr) {
        const auto means = cudaMeans(points);
        auto selection = Tensor::ones({points.size() / 3}, Device::CUDA, DataType::Bool);
        lfs::rendering::filter_selection_by_screen_window(
            selection,
            means,
            view_rotation_rows,
            translation,
            config.camera_model,
            config.width,
            config.height,
            config.pixel_focal_x,
            config.pixel_focal_y,
            config.center_x,
            config.center_y,
            config.ortho_scale,
            config.near_depth,
            config.far_depth,
            config.scale,
            config.offset_x,
            config.offset_y,
            model_transforms,
            transform_indices);
        return selection.cpu().to_vector_bool();
    }

    class SelectionScreenWindow : public ::testing::Test {
    protected:
        void SetUp() override {
            int device = -1;
            if (cudaGetDevice(&device) != cudaSuccess) {
                (void)cudaGetLastError();
                GTEST_SKIP() << "a live CUDA device is required";
            }
        }
    };

    TEST_F(SelectionScreenWindow, RandomCpuReferenceMatchesCuda) {
        std::mt19937 rng(0x51EC710u);
        std::uniform_real_distribution<float> coordinate(-25.0f, 25.0f);
        std::vector<float> points(kRandomPointCount * 3);
        std::vector<std::array<float, 3>> point_arrays(kRandomPointCount);
        std::vector<std::int32_t> transform_indices(kRandomPointCount);
        for (std::size_t i = 0; i < kRandomPointCount; ++i) {
            point_arrays[i] = {coordinate(rng), coordinate(rng), coordinate(rng)};
            points[i * 3] = point_arrays[i][0];
            points[i * 3 + 1] = point_arrays[i][1];
            points[i * 3 + 2] = point_arrays[i][2];
            transform_indices[i] = static_cast<std::int32_t>(i % 2);
        }

        const std::vector<float> model_transform_values{
            1.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            1.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            1.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            1.0f,
            0.9f,
            0.0f,
            0.0f,
            1.25f,
            0.0f,
            1.1f,
            0.0f,
            -0.75f,
            0.0f,
            0.0f,
            1.0f,
            -2.0f,
            0.0f,
            0.0f,
            0.0f,
            1.0f,
        };
        const auto model_transforms = Tensor::from_vector(
            model_transform_values, {std::size_t{2}, std::size_t{4}, std::size_t{4}}, Device::CUDA);
        const auto indices = Tensor::from_vector(
            transform_indices, {transform_indices.size()}, Device::CUDA);

        constexpr std::array models{
            ScreenWindowCameraModel::Pinhole,
            ScreenWindowCameraModel::Orthographic,
            ScreenWindowCameraModel::Equirectangular,
        };
        constexpr std::array scales{0.05f, 0.35f, 1.0f};
        constexpr std::array offsets{-1.0f, 0.0f, 1.0f};
        constexpr std::array near_far_pairs{
            std::pair{0.0f, 5.0f},
            std::pair{0.25f, 20.0f},
            std::pair{5.0f, 40.0f},
        };

        for (const auto model : models) {
            for (const float scale : scales) {
                for (const float offset_x : offsets) {
                    for (const float offset_y : offsets) {
                        for (const auto [near_depth, far_depth] : near_far_pairs) {
                            const ScreenWindowConfig config{
                                .camera_model = model,
                                .near_depth = near_depth,
                                .far_depth = far_depth,
                                .scale = scale,
                                .offset_x = offset_x,
                                .offset_y = offset_y,
                            };
                            const auto actual = runCuda(
                                points, config, kViewRotationRows, kTranslation,
                                &model_transforms, &indices);
                            ASSERT_EQ(actual.size(), kRandomPointCount);

                            std::size_t compared = 0;
                            for (std::size_t i = 0; i < kRandomPointCount; ++i) {
                                const auto projected = projectCpu(
                                    point_arrays[i], config, kViewRotationRows, kTranslation,
                                    &model_transform_values, transform_indices[i]);
                                if (nearDecisionBoundary(projected, config)) {
                                    continue;
                                }
                                ++compared;
                                ASSERT_EQ(actual[i], cpuReferenceInside(projected, config))
                                    << "point=" << i << " model=" << static_cast<std::uint32_t>(model)
                                    << " scale=" << scale << " offsets=(" << offset_x << ',' << offset_y
                                    << ") near=" << near_depth << " far=" << far_depth;
                            }
                            EXPECT_GT(compared, kRandomPointCount / 2);
                        }
                    }
                }
            }
        }
    }

    TEST_F(SelectionScreenWindow, InclusiveBoundariesAndPinnedCases) {
        constexpr std::array<float, 9> identity_rows{
            1.0f,
            0.0f,
            0.0f,
            0.0f,
            1.0f,
            0.0f,
            0.0f,
            0.0f,
            1.0f,
        };
        constexpr std::array<float, 3> origin{0.0f, 0.0f, 0.0f};
        const ScreenWindowConfig pinhole{
            .camera_model = ScreenWindowCameraModel::Pinhole,
            .width = 1024,
            .height = 512,
            .pixel_focal_x = 256.0f,
            .pixel_focal_y = 256.0f,
            // The struct defaults derive the centre from kWidth/kHeight; this config
            // overrides the size, so the centre must follow it or the projection is
            // computed against the wrong principal point.
            .center_x = 512.0f,
            .center_y = 256.0f,
            .near_depth = 2.0f,
            .far_depth = 8.0f,
            .scale = 0.5f,
        };
        const std::vector<float> boundary_points{
            -4.0f,
            0.0f,
            -4.0f,
            4.0f,
            0.0f,
            -4.0f,
            0.0f,
            2.0f,
            -4.0f,
            0.0f,
            -2.0f,
            -4.0f,
            0.0f,
            0.0f,
            -2.0f,
            0.0f,
            0.0f,
            -8.0f,
            0.0f,
            0.0f,
            1.0f,
        };
        EXPECT_EQ(runCuda(boundary_points, pinhole, identity_rows, origin),
                  (std::vector<bool>{true, true, true, true, true, true, false}));

        ScreenWindowConfig orthographic = pinhole;
        orthographic.camera_model = ScreenWindowCameraModel::Orthographic;
        orthographic.ortho_scale = 64.0f;
        const std::vector<float> ortho_points{
            -4.0f,
            0.0f,
            -2.0f,
            4.0f,
            0.0f,
            -8.0f,
            0.0f,
            0.0f,
            1.0f,
        };
        EXPECT_EQ(runCuda(ortho_points, orthographic, identity_rows, origin),
                  (std::vector<bool>{true, true, false}));

        const ScreenWindowConfig equirectangular{
            .camera_model = ScreenWindowCameraModel::Equirectangular,
            .width = 1024,
            .height = 512,
            .near_depth = 0.0f,
            .far_depth = 8.0f,
            .scale = 1.0f,
        };
        const std::vector<float> seam_and_sentinel_points{
            1.0e-7f,
            0.0f,
            4.0f,
            -1.0e-7f,
            0.0f,
            4.0f,
            0.0f,
            0.0f,
            0.0f,
        };
        EXPECT_EQ(runCuda(seam_and_sentinel_points, equirectangular, identity_rows, origin),
                  (std::vector<bool>{true, true, false}));

        ScreenWindowConfig garbage_range = equirectangular;
        garbage_range.near_depth = -100.0f;
        garbage_range.far_depth = 0.0f;
        EXPECT_EQ(runCuda({0.0f, 0.0f, 0.0f}, garbage_range, identity_rows, origin),
                  (std::vector<bool>{false}));
    }

    // The cases above only ever pin valid orthographic scales. The Vulkan lane
    // replaces any non-finite or sub-threshold scale with DEFAULT_ORTHO_SCALE
    // before the shader sees it (vksplat_viewport_renderer.cpp), so the CUDA lane
    // has to make the same substitution or the two disagree about which splats
    // are inside the window.
    TEST_F(SelectionScreenWindow, OrthographicSubstitutesDefaultScaleOverInvalidDomain) {
        const std::array<float, 9> identity_rows{
            1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
        const std::array<float, 3> origin{0.0f, 0.0f, 0.0f};

        ScreenWindowConfig config;
        config.camera_model = ScreenWindowCameraModel::Orthographic;
        config.ortho_scale = lfs::rendering::DEFAULT_ORTHO_SCALE;

        // Inside, outside on X, and dead centre. These are only distinguishable
        // at the substituted scale: a scale that collapsed to zero would land
        // every point on the principal point and read all three true, and a NaN
        // scale would poison every comparison and read all three false.
        const std::vector<float> points{
            1.0f, 0.0f, -5.0f,
            3.0f, 0.0f, -5.0f,
            0.0f, 0.0f, -5.0f};
        const std::vector<bool> expected{true, false, true};
        ASSERT_EQ(runCuda(points, config, identity_rows, origin), expected);

        // 1.0e-5f is the boundary itself: the predicate admits only scales
        // strictly greater, so it belongs on the invalid side and pins a
        // future '>' -> '>=' slip.
        for (const float invalid : {0.0f,
                                    -0.0f,
                                    -5.0f,
                                    1.0e-9f,
                                    1.0e-6f,
                                    1.0e-5f,
                                    std::numeric_limits<float>::quiet_NaN(),
                                    std::numeric_limits<float>::infinity()}) {
            ScreenWindowConfig degenerate = config;
            degenerate.ortho_scale = invalid;
            EXPECT_EQ(runCuda(points, degenerate, identity_rows, origin), expected)
                << "ortho_scale " << invalid << " did not fall back to DEFAULT_ORTHO_SCALE";
        }

        config.center_x = 9999.0f;
        config.center_y = -9999.0f;
        EXPECT_EQ(runCuda(points, config, identity_rows, origin), expected);
    }

    TEST_F(SelectionScreenWindow, PrincipalPointDiscriminatesOpticalAxisPoint) {
        constexpr std::array<float, 9> identity_rows{
            1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
        constexpr std::array<float, 3> origin{0.0f, 0.0f, 0.0f};
        const std::vector<float> optical_axis_point{0.0f, 0.0f, -5.0f};
        ScreenWindowConfig off_centre{
            .width = 640,
            .height = 480,
            .center_x = 100.0f,
            .center_y = 240.0f,
            .near_depth = 2.0f,
            .far_depth = 8.0f,
            .scale = 0.35f,
            .offset_x = -1.0f};
        ScreenWindowConfig image_centre = off_centre;
        image_centre.center_x = 320.0f;
        image_centre.center_y = 240.0f;
        const std::array<float, 3> optical_axis_point_arr{
            optical_axis_point[0], optical_axis_point[1], optical_axis_point[2]};
        const auto projected_off_centre = projectCpu(optical_axis_point_arr, off_centre, identity_rows, origin);
        const auto projected_image_centre = projectCpu(optical_axis_point_arr, image_centre, identity_rows, origin);
        ASSERT_FALSE(nearDecisionBoundary(projected_off_centre, off_centre));
        ASSERT_FALSE(nearDecisionBoundary(projected_image_centre, image_centre));
        const bool cpu_inside_off_centre = cpuReferenceInside(projected_off_centre, off_centre);
        const bool cpu_outside_image_centre = cpuReferenceInside(projected_image_centre, image_centre);
        ASSERT_TRUE(cpu_inside_off_centre);
        ASSERT_FALSE(cpu_outside_image_centre);
        const auto cuda_off_centre = runCuda(optical_axis_point, off_centre, identity_rows, origin);
        const auto cuda_image_centre = runCuda(optical_axis_point, image_centre, identity_rows, origin);
        ASSERT_EQ(cuda_off_centre.size(), 1u);
        ASSERT_EQ(cuda_image_centre.size(), 1u);
        EXPECT_EQ(cuda_off_centre.front(), cpu_inside_off_centre);
        EXPECT_EQ(cuda_image_centre.front(), cpu_outside_image_centre);
    }

    TEST_F(SelectionScreenWindow, AsymmetricFocalsAreNotReconstructedFromAspect) {
        constexpr std::array<float, 9> identity_rows{
            1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
        constexpr std::array<float, 3> origin{0.0f, 0.0f, 0.0f};
        // view_x/view_z = 0.36 puts fx=500 at px=500 (inside the [280.8, 504.8] window)
        // and fx=600 at px=536 (outside) — the pair that actually discriminates.
        const std::vector<float> point{1.44f, 0.0f, -4.0f};
        ScreenWindowConfig carried_focals{
            .width = 640,
            .height = 480,
            .pixel_focal_x = 500.0f,
            .pixel_focal_y = 600.0f,
            .center_x = 320.0f,
            .center_y = 240.0f,
            .near_depth = 2.0f,
            .far_depth = 8.0f,
            .scale = 0.35f,
            .offset_x = 0.35f};
        ScreenWindowConfig aspect_reconstruction = carried_focals;
        aspect_reconstruction.pixel_focal_x = 600.0f;
        const std::array<float, 3> point_arr{point[0], point[1], point[2]};
        const auto projected_carried = projectCpu(point_arr, carried_focals, identity_rows, origin);
        const auto projected_aspect = projectCpu(point_arr, aspect_reconstruction, identity_rows, origin);
        ASSERT_FALSE(nearDecisionBoundary(projected_carried, carried_focals));
        ASSERT_FALSE(nearDecisionBoundary(projected_aspect, aspect_reconstruction));
        const bool inside_with_fx500 = cpuReferenceInside(projected_carried, carried_focals);
        const bool inside_with_fx600 = cpuReferenceInside(projected_aspect, aspect_reconstruction);
        ASSERT_TRUE(inside_with_fx500);
        ASSERT_FALSE(inside_with_fx600);
        const auto cuda_carried = runCuda(point, carried_focals, identity_rows, origin);
        const auto cuda_aspect = runCuda(point, aspect_reconstruction, identity_rows, origin);
        EXPECT_TRUE(cuda_carried.front());
        EXPECT_FALSE(cuda_aspect.front());
    }

    TEST_F(SelectionScreenWindow, RandomOffCentreAsymmetricCpuReferenceMatchesCuda) {
        std::mt19937 rng(0xA5F1234u);
        std::uniform_real_distribution<float> coordinate(-25.0f, 25.0f);
        std::vector<float> points(kRandomPointCount * 3);
        std::vector<std::array<float, 3>> point_arrays(kRandomPointCount);
        for (std::size_t i = 0; i < kRandomPointCount; ++i) {
            point_arrays[i] = {coordinate(rng), coordinate(rng), coordinate(rng)};
            points[i * 3] = point_arrays[i][0];
            points[i * 3 + 1] = point_arrays[i][1];
            points[i * 3 + 2] = point_arrays[i][2];
        }
        const ScreenWindowConfig config{
            .width = 640,
            .height = 480,
            .pixel_focal_x = 500.0f,
            .pixel_focal_y = 600.0f,
            .center_x = 120.0f,
            .center_y = 210.0f,
            .near_depth = 0.25f,
            .far_depth = 20.0f,
            .scale = 0.35f,
            .offset_x = -0.5f,
            .offset_y = 0.25f};
        const auto actual = runCuda(points, config, kViewRotationRows, kTranslation);
        ASSERT_EQ(actual.size(), kRandomPointCount);
        std::size_t compared = 0;
        for (std::size_t i = 0; i < kRandomPointCount; ++i) {
            const auto projected = projectCpu(point_arrays[i], config, kViewRotationRows, kTranslation);
            if (nearDecisionBoundary(projected, config))
                continue;
            ++compared;
            ASSERT_EQ(actual[i], cpuReferenceInside(projected, config)) << "point=" << i;
        }
        EXPECT_GT(compared, kRandomPointCount / 2);
    }

    TEST_F(SelectionScreenWindow, OrthographicAndEquirectIgnorePrincipalPoint) {
        constexpr std::array<float, 9> identity_rows{
            1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
        constexpr std::array<float, 3> origin{0.0f, 0.0f, 0.0f};
        ScreenWindowConfig orthographic;
        orthographic.camera_model = ScreenWindowCameraModel::Orthographic;
        orthographic.ortho_scale = lfs::rendering::DEFAULT_ORTHO_SCALE;
        const std::vector<float> ortho_points{1.0f, 0.0f, -5.0f, 3.0f, 0.0f, -5.0f, 0.0f, 0.0f, -5.0f};
        const std::vector<bool> ortho_expected{true, false, true};
        const auto centred_ortho = runCuda(ortho_points, orthographic, identity_rows, origin);
        orthographic.center_x = 9999.0f;
        orthographic.center_y = -9999.0f;
        EXPECT_EQ(runCuda(ortho_points, orthographic, identity_rows, origin), centred_ortho);
        EXPECT_EQ(centred_ortho, ortho_expected);
        const ScreenWindowConfig equirectangular{
            .camera_model = ScreenWindowCameraModel::Equirectangular,
            .width = 1024,
            .height = 512,
            .near_depth = 0.0f,
            .far_depth = 8.0f,
            .scale = 1.0f};
        const std::vector<float> equirect_points{1.0e-7f, 0.0f, 4.0f, -1.0e-7f, 0.0f, 4.0f, 0.0f, 0.0f, 0.0f};
        const std::vector<bool> equirect_expected{true, true, false};
        const auto centred_equirect = runCuda(equirect_points, equirectangular, identity_rows, origin);
        ScreenWindowConfig bogus_equirect = equirectangular;
        bogus_equirect.center_x = 9999.0f;
        bogus_equirect.center_y = -9999.0f;
        EXPECT_EQ(runCuda(equirect_points, bogus_equirect, identity_rows, origin), centred_equirect);
        EXPECT_EQ(centred_equirect, equirect_expected);
    }

} // namespace
