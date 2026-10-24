/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/crop_polygon.hpp"
#include "core/scene.hpp"
#include "core/splat_data.hpp"
#include "core/splat_data_transform.hpp"
#include "rendering/coordinate_conventions.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <glm/gtc/matrix_transform.hpp>
#include <limits>
#include <vector>

namespace {
    namespace cp = lfs::core::crop_polygon;
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::Tensor;

    // Concave L: the notch (1..2, 1..2) is outside.
    const std::vector<glm::vec2> kLShape{{-2, -2}, {2, -2}, {2, 1}, {1, 1}, {1, 2}, {-2, 2}};

    void expectNear(const glm::vec3& actual, const glm::vec3& expected, const float tolerance = 1e-4f) {
        EXPECT_NEAR(actual.x, expected.x, tolerance);
        EXPECT_NEAR(actual.y, expected.y, tolerance);
        EXPECT_NEAR(actual.z, expected.z, tolerance);
    }

    // Visualizer camera looking along `forward` from `eye`, as the viewport stores it.
    glm::mat3 lookRotation(const glm::vec3& forward, const glm::vec3& up) {
        const glm::vec3 back = -glm::normalize(forward);
        const glm::vec3 right = glm::normalize(glm::cross(up, back));
        return glm::mat3(right, glm::cross(back, right), back);
    }

    TEST(CropPolygonGeometryTest, OutlineValidationRejectsDegenerateInput) {
        EXPECT_TRUE(cp::isValidOutline(kLShape));
        EXPECT_FALSE(cp::isValidOutline(std::vector<glm::vec2>{{0, 0}, {1, 0}}));
        EXPECT_FALSE(cp::isValidOutline(std::vector<glm::vec2>{{0, 0}, {1, 1}, {2, 2}}));
        EXPECT_FALSE(cp::isValidOutline(std::vector<glm::vec2>{{0, 0}, {1, 0}, {1, 0}, {0, 0}}));
        EXPECT_FALSE(cp::isValidOutline(
            std::vector<glm::vec2>{{0, 0}, {1, 0}, {0, std::numeric_limits<float>::quiet_NaN()}}));
        std::vector<glm::vec2> too_many(cp::kMaxVertices + 1);
        for (size_t i = 0; i < too_many.size(); ++i) {
            const float angle = static_cast<float>(i) * 6.2831853f / static_cast<float>(too_many.size());
            too_many[i] = {std::cos(angle), std::sin(angle)};
        }
        EXPECT_FALSE(cp::isValidOutline(too_many));
        too_many.pop_back();
        EXPECT_TRUE(cp::isValidOutline(too_many));

        EXPECT_TRUE(cp::containsPoint(kLShape, {0.0f, 0.0f}));
        EXPECT_FALSE(cp::containsPoint(kLShape, {1.5f, 1.5f}));
        EXPECT_FALSE(cp::containsPoint(kLShape, {3.0f, 0.0f}));
    }

    TEST(CropPolygonGeometryTest, PlaneFacesTheCameraThroughTheFirstPoint) {
        const auto plane = cp::makePlane({0.0f, 0.0f, -4.0f}, {1.0f, 2.0f, 3.0f});
        ASSERT_TRUE(plane);
        expectNear(plane->normal, {0.0f, 0.0f, -1.0f});
        expectNear(plane->point, {1.0f, 2.0f, 3.0f});
        EXPECT_FALSE(cp::makePlane(glm::vec3(0.0f), {0.0f, 0.0f, 0.0f}));
        EXPECT_FALSE(cp::makePlane({0.0f, 0.0f, 1.0f}, glm::vec3(std::numeric_limits<float>::infinity())));
    }

    // The draw operator builds each ray from two unprojections of the clicked
    // pixel; both projections must hit the plane where the pixel shows it.
    TEST(CropPolygonGeometryTest, PixelRaysHitThePlaneInPerspectiveAndOrthographic) {
        const glm::ivec2 size(640, 480);
        const glm::vec3 eye(0.5f, -1.0f, 6.0f);
        const glm::mat3 rotation = lookRotation(glm::vec3(0.2f, 0.1f, -1.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        const glm::vec3 forward = lfs::rendering::cameraForward(rotation);
        const auto plane = cp::makePlane(forward, eye + forward * 5.0f);
        ASSERT_TRUE(plane);

        for (const bool orthographic : {false, true}) {
            SCOPED_TRACE(orthographic ? "orthographic" : "perspective");
            for (const glm::vec2 pixel : {glm::vec2(320.0f, 240.0f), glm::vec2(17.0f, 401.0f), glm::vec2(600.0f, 3.0f)}) {
                const auto unproject = [&](const float depth) {
                    return lfs::rendering::unprojectScreenPoint(rotation, eye, size, pixel.x, pixel.y, depth, 30.0f,
                                                                orthographic, 80.0f);
                };
                const auto ray = cp::rayThroughPoints(unproject(1e-3f), unproject(1.0f));
                ASSERT_TRUE(ray);
                const auto hit = cp::intersect(*ray, *plane);
                ASSERT_TRUE(hit);
                // The plane faces the camera, so every pixel meets it at view depth 5.
                expectNear(*hit, unproject(5.0f), 1e-3f);
                EXPECT_NEAR(glm::dot(*hit - plane->point, plane->normal), 0.0f, 1e-4f);
            }
        }

        const cp::Ray parallel{.origin = eye, .direction = lfs::rendering::cameraRight(rotation)};
        EXPECT_FALSE(cp::intersect(parallel, *plane));
        const cp::Ray away{.origin = eye, .direction = -forward};
        EXPECT_FALSE(cp::intersect(away, *plane));
    }

    TEST(CropPolygonGeometryTest, FrameIsOrthonormalFacesForwardAndCentresTheOutline) {
        const glm::vec3 forward = glm::normalize(glm::vec3(1.0f, -2.0f, 0.5f));
        const glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0.0f, 0.0f, 1.0f)));
        const glm::vec3 down = glm::cross(forward, right);
        const glm::vec3 origin(3.0f, 1.0f, -2.0f);
        const auto plane = cp::makePlane(forward, origin);
        ASSERT_TRUE(plane);
        std::vector<glm::vec3> world;
        for (const glm::vec2& v : kLShape)
            world.push_back(origin + right * (v.x + 7.0f) + down * (v.y - 3.0f));

        const auto frame = cp::buildFrame(world, *plane, right + forward * 0.25f, {}, 2.5f);
        ASSERT_TRUE(frame);
        const glm::mat3 axes(frame->world_from_local);
        for (int i = 0; i < 3; ++i) {
            EXPECT_NEAR(glm::length(axes[i]), 1.0f, 1e-5f);
            for (int j = i + 1; j < 3; ++j)
                EXPECT_NEAR(glm::dot(axes[i], axes[j]), 0.0f, 1e-5f);
        }
        EXPECT_NEAR(glm::determinant(axes), 1.0f, 1e-5f);
        expectNear(axes[2], forward);
        expectNear(axes[0], right);

        ASSERT_EQ(frame->vertices.size(), kLShape.size());
        glm::vec2 lo(std::numeric_limits<float>::max());
        glm::vec2 hi(std::numeric_limits<float>::lowest());
        for (size_t i = 0; i < world.size(); ++i) {
            lo = glm::min(lo, frame->vertices[i]);
            hi = glm::max(hi, frame->vertices[i]);
            expectNear(glm::vec3(frame->world_from_local * glm::vec4(frame->vertices[i], 0.0f, 1.0f)), world[i]);
        }
        EXPECT_NEAR(lo.x + hi.x, 0.0f, 1e-4f);
        EXPECT_NEAR(lo.y + hi.y, 0.0f, 1e-4f);
        // Without bounds the prism is centred on the drawing plane.
        EXPECT_FLOAT_EQ(frame->half_depth, 2.5f);
        EXPECT_NEAR(glm::dot(glm::vec3(frame->world_from_local[3]) - origin, forward), 0.0f, 1e-4f);
    }

    TEST(CropPolygonGeometryTest, DefaultDepthCoversTheBoundsSymmetrically) {
        const glm::vec3 forward(0.0f, 0.0f, -1.0f);
        const auto plane = cp::makePlane(forward, glm::vec3(0.0f, 0.0f, 10.0f));
        ASSERT_TRUE(plane);
        const std::vector<glm::vec3> world{{-1, -1, 10}, {1, -1, 10}, {1, 1, 10}, {-1, 1, 10}};
        std::array<glm::vec3, 8> corners{};
        for (int i = 0; i < 8; ++i)
            corners[i] = {(i & 1) ? 4.0f : -4.0f, (i & 2) ? 3.0f : -3.0f, (i & 4) ? 2.0f : -6.0f};

        const auto frame = cp::buildFrame(world, *plane, {1.0f, 0.0f, 0.0f}, corners, 1.0f);
        ASSERT_TRUE(frame);
        EXPECT_NEAR(frame->half_depth, 4.0f * (1.0f + cp::kDefaultDepthPadding), 1e-5f);
        expectNear(glm::vec3(frame->world_from_local[3]), {0.0f, 0.0f, -2.0f});
        const glm::mat4 local_from_world = glm::inverse(frame->world_from_local);
        for (const auto& corner : corners)
            EXPECT_LE(std::abs((local_from_world * glm::vec4(corner, 1.0f)).z), frame->half_depth);
    }

    bool cudaAvailable() {
        int count = 0;
        return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
    }

    lfs::core::SplatData makeSplat(const std::vector<glm::vec3>& points) {
        std::vector<float> means;
        std::vector<float> rotation(points.size() * 4, 0.0f);
        for (size_t i = 0; i < points.size(); ++i) {
            means.insert(means.end(), {points[i].x, points[i].y, points[i].z});
            rotation[i * 4] = 1.0f;
        }
        const size_t n = points.size();
        return lfs::core::SplatData(1, Tensor::from_vector(means, {n, size_t{3}}, Device::CUDA),
                                    Tensor::zeros({n, size_t{1}, size_t{3}}, Device::CUDA, DataType::Float32),
                                    Tensor::zeros({n, size_t{3}, size_t{3}}, Device::CUDA, DataType::Float32),
                                    Tensor::zeros({n, size_t{3}}, Device::CUDA, DataType::Float32),
                                    Tensor::from_vector(rotation, {n, size_t{4}}, Device::CUDA),
                                    Tensor::zeros({n, size_t{1}}, Device::CUDA, DataType::Float32), 1.0f);
    }

    std::vector<bool> keptMask(const lfs::core::SplatData& splat) {
        std::vector<bool> kept;
        const auto deleted = splat.has_deleted_mask() ? splat.deleted().cpu().to_vector_bool()
                                                      : std::vector<bool>(static_cast<size_t>(splat.size()), false);
        for (const bool removed : deleted)
            kept.push_back(!removed);
        return kept;
    }

    TEST(CropPolygonSoftCropTest, ConcaveOutlineAndDepthUnderRotatedScaledTransform) {
        if (!cudaAvailable())
            GTEST_SKIP() << "CUDA device unavailable";

        // Polygon frame in world space: rotated, translated and scaled.
        const glm::mat4 polygon_to_world =
            glm::scale(glm::rotate(glm::translate(glm::mat4(1.0f), glm::vec3(5.0f, -1.0f, 2.0f)),
                                   glm::radians(35.0f), glm::normalize(glm::vec3(1.0f, 2.0f, 0.5f))),
                       glm::vec3(2.0f, 0.5f, 1.5f));
        const std::vector<glm::vec3> local_points{
            {0.0f, 0.0f, 0.0f},  // inside
            {-1.5f, 1.5f, 0.9f}, // inside, near a cap
            {1.5f, 1.5f, 0.0f},  // in the notch: outside in xy
            {3.0f, 0.0f, 0.0f},  // outside in xy
            {0.0f, 0.0f, 1.2f},  // outside in z
            {0.0f, 0.0f, -1.2f}, // outside in z
        };
        std::vector<glm::vec3> world_points;
        for (const auto& p : local_points)
            world_points.emplace_back(polygon_to_world * glm::vec4(p, 1.0f));

        lfs::core::CropPolygonData polygon;
        polygon.vertices = kLShape;
        polygon.half_depth = 1.0f;
        const glm::mat4 world_to_local = glm::inverse(polygon_to_world);

        auto kept_inside = makeSplat(world_points);
        EXPECT_TRUE(lfs::core::soft_crop_by_polygon(kept_inside, world_to_local, polygon).is_valid());
        EXPECT_EQ(keptMask(kept_inside), (std::vector<bool>{true, true, false, false, false, false}));

        polygon.inverse = true;
        auto kept_outside = makeSplat(world_points);
        EXPECT_TRUE(lfs::core::soft_crop_by_polygon(kept_outside, world_to_local, polygon).is_valid());
        EXPECT_EQ(keptMask(kept_outside), (std::vector<bool>{false, false, true, true, true, true}));

        polygon.vertices = {{0, 0}, {1, 1}, {2, 2}};
        auto untouched = makeSplat(world_points);
        EXPECT_FALSE(lfs::core::soft_crop_by_polygon(untouched, world_to_local, polygon).is_valid());
        EXPECT_EQ(untouched.visible_count(), world_points.size());
    }

} // namespace
