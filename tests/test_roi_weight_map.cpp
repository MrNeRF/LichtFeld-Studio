/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "training/kernels/roi_weight_map.hpp"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace {

    bool ray_intersects_box(
        const glm::vec3& world_origin,
        const glm::vec3& world_direction,
        const glm::mat4& world_to_cropbox,
        const glm::vec3& crop_min,
        const glm::vec3& crop_max) {
        const glm::vec3 origin =
            glm::vec3(world_to_cropbox * glm::vec4(world_origin, 1.0f));
        const glm::vec3 direction =
            glm::vec3(world_to_cropbox * glm::vec4(world_direction, 0.0f));

        float t_enter = -std::numeric_limits<float>::infinity();
        float t_exit = std::numeric_limits<float>::infinity();
        for (int axis = 0; axis < 3; ++axis) {
            if (std::abs(direction[axis]) < 1.0e-8f) {
                if (origin[axis] < crop_min[axis] || origin[axis] > crop_max[axis]) {
                    return false;
                }
                continue;
            }
            float t0 = (crop_min[axis] - origin[axis]) / direction[axis];
            float t1 = (crop_max[axis] - origin[axis]) / direction[axis];
            if (t0 > t1) {
                std::swap(t0, t1);
            }
            t_enter = std::max(t_enter, t0);
            t_exit = std::min(t_exit, t1);
            if (t_exit < t_enter) {
                return false;
            }
        }
        return t_exit >= 0.0f;
    }

    void expect_gpu_matches_reference(
        const glm::mat4& world_to_cropbox,
        const glm::vec3& crop_min,
        const glm::vec3& crop_max,
        const bool inverse,
        const glm::mat4& world_to_camera = glm::mat4(1.0f),
        const glm::vec3& camera_position = glm::vec3(0.0f)) {
        constexpr int width = 9;
        constexpr int height = 7;
        constexpr float fx = 6.0f;
        constexpr float fy = 6.0f;
        constexpr float cx = 4.5f;
        constexpr float cy = 3.5f;
        constexpr float outside_weight = 0.2f;

        std::vector<float> world_to_camera_values(16);
        for (size_t row = 0; row < 4; ++row) {
            for (size_t column = 0; column < 4; ++column) {
                world_to_camera_values[row * 4 + column] =
                    world_to_camera[column][row];
            }
        }
        const std::vector<float> camera_position_values{
            camera_position.x, camera_position.y, camera_position.z};
        const auto w2c = lfs::core::Tensor::from_vector(
            world_to_camera_values, {size_t{4}, size_t{4}}, lfs::core::Device::CUDA);
        const auto camera_pos = lfs::core::Tensor::from_vector(
            camera_position_values, {size_t{3}}, lfs::core::Device::CUDA);
        auto output = lfs::core::Tensor::empty(
            {size_t{height}, size_t{width}},
            lfs::core::Device::CUDA,
            lfs::core::DataType::Float32);

        lfs::training::kernels::launch_roi_weight_map(
            w2c.ptr<float>(),
            camera_pos.ptr<float>(),
            fx,
            fy,
            cx,
            cy,
            width,
            height,
            world_to_cropbox,
            crop_min,
            crop_max,
            inverse,
            outside_weight,
            nullptr,
            nullptr,
            nullptr,
            output.ptr<float>(),
            output.stream());

        std::vector<float> expected;
        expected.reserve(static_cast<size_t>(width) * height);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const glm::vec3 camera_ray{
                    (static_cast<float>(x) + 0.5f - cx) / fx,
                    (static_cast<float>(y) + 0.5f - cy) / fy,
                    1.0f};
                const glm::vec3 world_ray =
                    glm::transpose(glm::mat3(world_to_camera)) * camera_ray;
                const bool hit = ray_intersects_box(
                    camera_position, world_ray, world_to_cropbox, crop_min, crop_max);
                const bool full_weight = inverse ? !hit : hit;
                expected.push_back(full_weight ? 1.0f : outside_weight);
            }
        }

        const auto actual = output.to(lfs::core::Device::CPU).to_vector();
        ASSERT_EQ(actual.size(), expected.size());
        EXPECT_GT(std::count(expected.begin(), expected.end(), 1.0f), 0);
        EXPECT_GT(std::count(expected.begin(), expected.end(), outside_weight), 0);
        for (size_t i = 0; i < actual.size(); ++i) {
            EXPECT_FLOAT_EQ(actual[i], expected[i]) << "pixel " << i;
        }
    }

} // namespace

TEST(RoiWeightMapTest, AxisAlignedBoxMatchesCpuSlabReference) {
    expect_gpu_matches_reference(
        glm::mat4(1.0f),
        {-0.6f, -0.4f, 2.0f},
        {0.6f, 0.4f, 3.0f},
        false);
}

TEST(RoiWeightMapTest, RotatedObbMatchesCpuSlabReference) {
    const glm::mat4 cropbox_to_world =
        glm::rotate(
            glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 3.0f)),
            glm::radians(45.0f),
            glm::vec3(0.0f, 0.0f, 1.0f));
    expect_gpu_matches_reference(
        glm::inverse(cropbox_to_world),
        {-0.8f, -0.15f, -0.4f},
        {0.8f, 0.15f, 0.4f},
        false);
}

TEST(RoiWeightMapTest, InverseBoxFlipsHitAndMissWeights) {
    expect_gpu_matches_reference(
        glm::mat4(1.0f),
        {-0.6f, -0.4f, 2.0f},
        {0.6f, 0.4f, 3.0f},
        true);
}

TEST(RoiWeightMapTest, RotatedTranslatedCameraUsesWorldSpaceRays) {
    const glm::vec3 camera_position{-1.0f, 0.0f, 0.0f};
    const glm::mat4 camera_to_world =
        glm::translate(glm::mat4(1.0f), camera_position) *
        glm::rotate(
            glm::mat4(1.0f),
            glm::radians(90.0f),
            glm::vec3(0.0f, 1.0f, 0.0f));
    expect_gpu_matches_reference(
        glm::mat4(1.0f),
        {1.0f, -0.5f, -0.4f},
        {2.0f, 0.5f, 0.4f},
        false,
        glm::inverse(camera_to_world),
        camera_position);
}

// Catches a lens-distorted GUT camera weighting pixels through pinhole rays instead of its own rays
// (fisheye pixels near the rim see a different part of the scene than the pinhole formula says).
TEST(RoiWeightMapTest, CameraRaysReplaceThePinholeRays) {
    constexpr int width = 9;
    constexpr int height = 7;
    constexpr float outside_weight = 0.2f;
    const glm::vec3 crop_min{-0.6f, -0.4f, 2.0f};
    const glm::vec3 crop_max{0.6f, 0.4f, 3.0f};
    // Rays of a narrower lens than the pinhole intrinsics passed below, with two rayless pixels.
    std::vector<float> rays;
    std::vector<float> expected;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const glm::vec3 ray{(x + 0.5f - 4.5f) / 12.0f, (y + 0.5f - 3.5f) / 12.0f, 1.0f};
            const bool rayless = (x == 4 && y == 3) || (x == 0 && y == 0);
            const float nan = std::numeric_limits<float>::quiet_NaN();
            rays.insert(rays.end(), {rayless ? nan : ray.x, rayless ? nan : ray.y, rayless ? nan : ray.z});
            const bool hit = !rayless && ray_intersects_box(glm::vec3(0.0f), ray, glm::mat4(1.0f), crop_min, crop_max);
            expected.push_back(hit ? 1.0f : outside_weight);
        }
    }
    const std::vector<float> identity{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    const auto w2c = lfs::core::Tensor::from_vector(identity, {size_t{4}, size_t{4}}, lfs::core::Device::CUDA);
    const auto camera_pos = lfs::core::Tensor::zeros({size_t{3}}, lfs::core::Device::CUDA);
    const auto rays_tensor = lfs::core::Tensor::from_vector(
        rays, {size_t{height}, size_t{width}, size_t{3}}, lfs::core::Device::CUDA);
    auto output = lfs::core::Tensor::empty({size_t{height}, size_t{width}}, lfs::core::Device::CUDA);
    lfs::training::kernels::launch_roi_weight_map(
        w2c.ptr<float>(), camera_pos.ptr<float>(), 6.0f, 6.0f, 4.5f, 3.5f, width, height,
        glm::mat4(1.0f), crop_min, crop_max, false, outside_weight, rays_tensor.ptr<float>(), nullptr, nullptr,
        output.ptr<float>(), output.stream());
    const auto actual = output.cpu().to_vector();
    ASSERT_EQ(actual.size(), expected.size());
    EXPECT_EQ(actual[4 + 3 * width], outside_weight) << "a pixel without a ray must not count as seen";
    EXPECT_GT(std::count(expected.begin(), expected.end(), 1.0f), 0);
    for (size_t i = 0; i < actual.size(); ++i)
        EXPECT_FLOAT_EQ(actual[i], expected[i]) << "pixel " << i;
}

// Catches in-box floaters seen from cameras standing inside the box: every ray of such a camera
// starts in the box, so the ray test would count the far background as ROI. The surface the pixel
// shows decides instead; pixels without a surface count as outside.
TEST(RoiWeightMapTest, CameraInsideTheBoxWeighsPixelsByTheSurfaceTheyShow) {
    constexpr int width = 4;
    constexpr int height = 1;
    constexpr float outside_weight = 0.1f;
    const glm::vec3 crop_min{-2.0f, -2.0f, -2.0f};
    const glm::vec3 crop_max{2.0f, 2.0f, 2.0f};
    // Pinhole rays with z = 1 and z-depth: surfaces at z 1.5 (in the box), 10 (beyond it), a pixel with
    // too little alpha and a surface at z 1.0 half covered (alpha 0.6, accumulated depth 0.6).
    const std::vector<float> depth{1.5f, 10.0f, 1.0f, 0.6f};
    const std::vector<float> alpha{1.0f, 1.0f, 0.2f, 0.6f};
    const std::vector<float> identity{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    const auto w2c = lfs::core::Tensor::from_vector(identity, {size_t{4}, size_t{4}}, lfs::core::Device::CUDA);
    const auto camera_pos = lfs::core::Tensor::zeros({size_t{3}}, lfs::core::Device::CUDA);
    const auto depth_t = lfs::core::Tensor::from_vector(depth, {size_t{height}, size_t{width}}, lfs::core::Device::CUDA);
    const auto alpha_t = lfs::core::Tensor::from_vector(alpha, {size_t{height}, size_t{width}}, lfs::core::Device::CUDA);
    auto output = lfs::core::Tensor::empty({size_t{height}, size_t{width}}, lfs::core::Device::CUDA);
    const auto run = [&](const bool inverse) {
        lfs::training::kernels::launch_roi_weight_map(
            w2c.ptr<float>(), camera_pos.ptr<float>(), 100.0f, 100.0f, 2.0f, 0.5f, width, height,
            glm::mat4(1.0f), crop_min, crop_max, inverse, outside_weight, nullptr, depth_t.ptr<float>(),
            alpha_t.ptr<float>(), output.ptr<float>(), output.stream());
        return output.cpu().to_vector();
    };
    EXPECT_EQ(run(false), (std::vector<float>{1.0f, outside_weight, outside_weight, 1.0f}));
    EXPECT_EQ(run(true), (std::vector<float>{outside_weight, 1.0f, 1.0f, outside_weight}));
}

namespace {
    // Splat centres: inside the box, behind it, in front of it, beside it, behind but off-axis.
    const std::vector<float> kCentres{0, 0, 10, 0, 0, 20, 0, 0, 5, 5, 0, 10, 0, 5, 20};
    const glm::vec3 kBoxMin{-1.0f, -1.0f, 9.0f};
    const glm::vec3 kBoxMax{1.0f, 1.0f, 11.0f};

    std::vector<bool> run_view_mask(const glm::vec3& camera, const bool inverse, const std::vector<bool>& deleted = {}) {
        const auto means = lfs::core::Tensor::from_vector(kCentres, {size_t{5}, size_t{3}}, lfs::core::Device::CUDA);
        const auto position = lfs::core::Tensor::from_vector({camera.x, camera.y, camera.z}, {size_t{3}}, lfs::core::Device::CUDA);
        auto rejected = lfs::core::Tensor::empty({size_t{5}}, lfs::core::Device::CUDA, lfs::core::DataType::Bool);
        lfs::core::Tensor deleted_tensor;
        if (!deleted.empty()) {
            std::vector<float> values(deleted.begin(), deleted.end());
            deleted_tensor = lfs::core::Tensor::from_vector(values, {size_t{5}}, lfs::core::Device::CUDA).to(lfs::core::DataType::Bool);
        }
        lfs::training::kernels::launch_roi_view_reject_mask(
            means.ptr<float>(), nullptr, 5, glm::mat4(1.0f), glm::mat4(1.0f), position.ptr<float>(), kBoxMin, kBoxMax, inverse,
            deleted_tensor.is_valid() ? deleted_tensor.ptr<bool>() : nullptr, rejected.ptr<bool>(), nullptr);
        const auto values = rejected.to(lfs::core::DataType::Float32).cpu().to_vector();
        return {values.begin(), values.end()};
    }
} // namespace

// Catches crop damping that slows the background behind (or the occluders in front of) the crop box:
// those splats shade full-weight pixels, so they must train at full rate or in-box splats paint them.
TEST(RoiSplatMaskTest, ViewMaskKeepsSplatsInsideInFrontAndBehindTheBox) {
    EXPECT_EQ(run_view_mask({0, 0, 0}, false), (std::vector<bool>{false, false, false, true, true}));
    EXPECT_EQ(run_view_mask({0, 0, 0}, true), (std::vector<bool>{true, true, true, false, false}));
    // From the side the box now hides splat 3 instead of splats 1 and 2.
    EXPECT_EQ(run_view_mask({20, 0, 10}, false), (std::vector<bool>{false, true, true, false, true}));
    // Deleted rows are never damped.
    EXPECT_EQ(run_view_mask({0, 0, 0}, false, {false, false, false, true, false}),
              (std::vector<bool>{false, false, false, false, true}));
}

// Catches the halo at the box rim: ground beside the box that a splat's footprint reaches from the
// box must train at full rate, or in-box splats spill over it. Inverse boxes shrink instead.
TEST(RoiSplatMaskTest, FootprintMarginReachesSplatsBesideTheBox) {
    const auto means = lfs::core::Tensor::from_vector(kCentres, {size_t{5}, size_t{3}}, lfs::core::Device::CUDA);
    const auto position = lfs::core::Tensor::from_vector({0.0f, 0.0f, 0.0f}, {size_t{3}}, lfs::core::Device::CUDA);
    auto rejected = lfs::core::Tensor::empty({size_t{5}}, lfs::core::Device::CUDA, lfs::core::DataType::Bool);
    const auto run = [&](const float sigma, const bool inverse) {
        const std::vector<float> scales(15, std::log(sigma));
        const auto scaling = lfs::core::Tensor::from_vector(scales, {size_t{5}, size_t{3}}, lfs::core::Device::CUDA);
        lfs::training::kernels::launch_roi_view_reject_mask(
            means.ptr<float>(), scaling.ptr<float>(), 5, glm::mat4(1.0f), glm::mat4(1.0f), position.ptr<float>(),
            kBoxMin, kBoxMax, inverse, nullptr, rejected.ptr<bool>(), nullptr);
        const auto values = rejected.to(lfs::core::DataType::Float32).cpu().to_vector();
        return std::vector<bool>(values.begin(), values.end());
    };
    // Small splats: the centre ray decides, as without a margin.
    EXPECT_EQ(run(0.01f, false), (std::vector<bool>{false, false, false, true, true}));
    // Wide splats beside the box reach it.
    EXPECT_EQ(run(2.0f, false), (std::vector<bool>{false, false, false, false, false}));
    // Wide splats empty an inverse box, so nothing is damped.
    EXPECT_EQ(run(2.0f, true), (std::vector<bool>{false, false, false, false, false}));

    EXPECT_NEAR(lfs::training::kernels::box_units_per_model_unit(glm::scale(glm::mat4(1.0f), glm::vec3(1.0f, 3.0f, 2.0f))),
                3.0f, 1e-4f);
}

// Catches refine scores damped for splats that some training camera sees through the box.
TEST(RoiSplatMaskTest, RelevanceMaskKeepsSplatsAnyCameraSeesThroughTheBox) {
    const auto means = lfs::core::Tensor::from_vector(kCentres, {size_t{5}, size_t{3}}, lfs::core::Device::CUDA);
    auto rejected = lfs::core::Tensor::empty({size_t{5}}, lfs::core::Device::CUDA, lfs::core::DataType::Bool);
    const auto run = [&](const std::vector<float>& cameras) {
        const auto centres = lfs::core::Tensor::from_vector(cameras, {cameras.size() / 3, size_t{3}}, lfs::core::Device::CUDA);
        lfs::training::kernels::launch_roi_relevance_reject_mask(
            means.ptr<float>(), nullptr, 5, glm::mat4(1.0f), centres.ptr<float>(), static_cast<int>(cameras.size() / 3),
            kBoxMin, kBoxMax, false, nullptr, rejected.ptr<bool>(), nullptr, nullptr);
        const auto values = rejected.to(lfs::core::DataType::Float32).cpu().to_vector();
        return std::vector<bool>(values.begin(), values.end());
    };
    EXPECT_EQ(run({0, 0, 0}), (std::vector<bool>{false, false, false, true, true}));
    // A second camera behind splat 4 sees it through the box: only splat 3 stays irrelevant.
    EXPECT_EQ(run({0, 0, 0, 0, -5, 0}), (std::vector<bool>{false, false, false, true, false}));
}

namespace {
    // An L outline: the square [-1, 1]^2 without its (0, 1) x (0, 1) quarter, the notch, over |z| <= 0.5.
    const std::vector<glm::vec2> kLOutline{{-1.0f, -1.0f}, {1.0f, -1.0f}, {1.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 1.0f}, {-1.0f, 1.0f}};
    const lfs::training::kernels::RoiPolygon kLPrism{
        .vertices = kLOutline.data(),
        .count = static_cast<int>(kLOutline.size()),
        .half_depth = 0.5f};

    // Weights of a 1-row image whose pixels look along the given rays from the world origin, with the polygon
    // frame placed so that the camera sits at camera_in_polygon.
    std::vector<float> run_polygon_weight_map(const glm::vec3& camera_in_polygon, const std::vector<glm::vec3>& rays,
                                              const bool inverse, const std::vector<float>& depth = {},
                                              const std::vector<float>& alpha = {}) {
        const int width = static_cast<int>(rays.size());
        std::vector<float> values;
        for (const auto& ray : rays)
            values.insert(values.end(), {ray.x, ray.y, ray.z});
        const std::vector<float> identity{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
        const auto w2c = lfs::core::Tensor::from_vector(identity, {size_t{4}, size_t{4}}, lfs::core::Device::CUDA);
        const auto camera_pos = lfs::core::Tensor::zeros({size_t{3}}, lfs::core::Device::CUDA);
        const auto rays_tensor = lfs::core::Tensor::from_vector(
            values, {size_t{1}, static_cast<size_t>(width), size_t{3}}, lfs::core::Device::CUDA);
        lfs::core::Tensor depth_tensor, alpha_tensor;
        if (!depth.empty()) {
            depth_tensor = lfs::core::Tensor::from_vector(depth, {size_t{1}, static_cast<size_t>(width)}, lfs::core::Device::CUDA);
            alpha_tensor = lfs::core::Tensor::from_vector(alpha, {size_t{1}, static_cast<size_t>(width)}, lfs::core::Device::CUDA);
        }
        auto output = lfs::core::Tensor::empty({size_t{1}, static_cast<size_t>(width)}, lfs::core::Device::CUDA);
        lfs::training::kernels::launch_roi_weight_map(
            w2c.ptr<float>(), camera_pos.ptr<float>(), 1.0f, 1.0f, 0.0f, 0.0f, width, 1,
            glm::translate(glm::mat4(1.0f), camera_in_polygon), kLPrism, inverse, 0.25f, rays_tensor.ptr<float>(),
            depth.empty() ? nullptr : depth_tensor.ptr<float>(), alpha.empty() ? nullptr : alpha_tensor.ptr<float>(),
            output.ptr<float>(), output.stream());
        return output.cpu().to_vector();
    }
} // namespace

// Catches a polygon ROI taken from the convex hull or bounds of the outline instead of the prism itself: from
// above, the ray through the notch meets neither the L nor its slab, though the hull of the projected corners
// covers it; from the side, a ray just over the cap passes above the L.
TEST(RoiWeightMapTest, PolygonRegionIsExactForAConcaveOutline) {
    // Above the outline, looking down at the points (x, y, 0).
    const std::vector<glm::vec3> down{{-0.5f, -0.5f, -10.0f}, {0.5f, 0.5f, -10.0f}, {0.5f, -0.5f, -10.0f}, {-0.5f, 0.5f, -10.0f}, {1.5f, 1.5f, -10.0f}};
    EXPECT_EQ(run_polygon_weight_map({0.0f, 0.0f, 10.0f}, down, false),
              (std::vector<float>{1.0f, 0.25f, 1.0f, 1.0f, 0.25f}));
    EXPECT_EQ(run_polygon_weight_map({0.0f, 0.0f, 10.0f}, down, true),
              (std::vector<float>{0.25f, 1.0f, 0.25f, 0.25f, 1.0f}));

    // Beside the outline at y = -0.5, 0.1 above the cap: level rays pass over the L, rays tilted into the slab cross it.
    const std::vector<glm::vec3> side{{1.0f, 0.0f, 0.0f}, {1.0f, 0.0f, -0.02f}, {1.0f, 0.0f, 0.02f}};
    EXPECT_EQ(run_polygon_weight_map({-10.0f, -0.5f, 0.6f}, side, false), (std::vector<float>{0.25f, 1.0f, 0.25f}));
}

// Catches a camera inside a crop polygon weighing pixels by their rays instead of the surface they show, which
// counts the notch of the outline as ROI.
TEST(RoiWeightMapTest, CameraInsideThePolygonWeighsPixelsByTheSurfaceTheyShow) {
    // From inside the bottom arm: surfaces in the bottom arm, in the notch, and one pixel with too little alpha.
    const std::vector<glm::vec3> rays{{1.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 0.0f}};
    EXPECT_EQ(run_polygon_weight_map({-0.5f, -0.5f, 0.0f}, rays, false, {1.0f, 1.0f, 0.1f}, {1.0f, 1.0f, 0.2f}),
              (std::vector<float>{1.0f, 0.25f, 0.25f}));
}

namespace {
    // Splat centres in polygon space: in the bottom arm, in the notch, below the bottom arm, beside the outline.
    const std::vector<float> kPolygonCentres{0.5f, -0.5f, 0.0f, 0.5f, 0.5f, 0.0f, 0.5f, -0.5f, -5.0f, 5.0f, 0.0f, 0.0f};

    std::vector<bool> to_bools(const lfs::core::Tensor& mask) {
        const auto values = mask.to(lfs::core::DataType::Float32).cpu().to_vector();
        return {values.begin(), values.end()};
    }
} // namespace

// Catches polygon damping that tests the outline's bounds: seen from above, the notch splat is neither in nor
// in front of or behind the L, so it trains damped, while the splat under the bottom arm trains at full rate.
TEST(RoiSplatMaskTest, PolygonViewMaskFollowsTheConcaveOutline) {
    const auto means = lfs::core::Tensor::from_vector(kPolygonCentres, {size_t{4}, size_t{3}}, lfs::core::Device::CUDA);
    const auto position = lfs::core::Tensor::from_vector({0.0f, 0.0f, 10.0f}, {size_t{3}}, lfs::core::Device::CUDA);
    auto rejected = lfs::core::Tensor::empty({size_t{4}}, lfs::core::Device::CUDA, lfs::core::DataType::Bool);
    const auto run = [&](const float sigma, const bool inverse) {
        const auto scaling = lfs::core::Tensor::from_vector(std::vector<float>(12, std::log(sigma)), {size_t{4}, size_t{3}},
                                                            lfs::core::Device::CUDA);
        lfs::training::kernels::launch_roi_view_reject_mask(
            means.ptr<float>(), scaling.ptr<float>(), 4, glm::mat4(1.0f), glm::mat4(1.0f), position.ptr<float>(),
            kLPrism, inverse, nullptr, rejected.ptr<bool>(), nullptr);
        return to_bools(rejected);
    };
    EXPECT_EQ(run(0.01f, false), (std::vector<bool>{false, true, false, true}));
    EXPECT_EQ(run(0.01f, true), (std::vector<bool>{true, false, true, false}));
    // A footprint reaching over the notch edges keeps the notch splat at full rate.
    EXPECT_EQ(run(0.2f, false), (std::vector<bool>{false, false, false, true}));
}

// Catches polygon refine scores or ROI rows that use the outline's bounds instead of the outline.
TEST(RoiSplatMaskTest, PolygonRelevanceMaskAndRoiRowsFollowTheConcaveOutline) {
    const auto means = lfs::core::Tensor::from_vector(kPolygonCentres, {size_t{4}, size_t{3}}, lfs::core::Device::CUDA);
    auto rejected = lfs::core::Tensor::empty({size_t{4}}, lfs::core::Device::CUDA, lfs::core::DataType::Bool);
    auto in_roi = lfs::core::Tensor::empty({size_t{4}}, lfs::core::Device::CUDA, lfs::core::DataType::Bool);
    const auto cameras = lfs::core::Tensor::from_vector({0.0f, 0.0f, 10.0f}, {size_t{1}, size_t{3}}, lfs::core::Device::CUDA);
    lfs::training::kernels::launch_roi_relevance_reject_mask(
        means.ptr<float>(), nullptr, 4, glm::mat4(1.0f), cameras.ptr<float>(), 1, kLPrism, false, nullptr,
        rejected.ptr<bool>(), in_roi.ptr<bool>(), nullptr);
    EXPECT_EQ(to_bools(rejected), (std::vector<bool>{false, true, false, true}));
    EXPECT_EQ(to_bools(in_roi), (std::vector<bool>{true, false, false, false}));
}
