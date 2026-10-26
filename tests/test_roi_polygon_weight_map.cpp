/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "lfs/training/ops/registry.hpp"
#include "lfs/training/ops/training_image.hpp"

#include <algorithm>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace {
    // An L outline: the square [-1, 1]^2 without its (0, 1) x (0, 1) quarter, the notch, over |z| <= 0.5.
    const std::vector<float> kLOutline{-1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, -1.0f, 1.0f};

    // The polygon weight map runs through each backend's training-image op: CUDA's polygon kernel
    // and the Vulkan and Metal shader ports.
    class RoiPolygonWeightMapTest : public testing::TestWithParam<lfs::core::GpuBackend> {
    protected:
        void SetUp() override {
            if (!lfs::core::gpu_backend_available(GetParam()))
                GTEST_SKIP() << "Backend unavailable";
            scope_.emplace(GetParam());
        }

        // Weights of a 1-row image whose pixels look along the given rays from the world origin, with the polygon
        // frame placed so that the camera sits at camera_in_polygon.
        std::vector<float> run(const glm::vec3& camera_in_polygon, const std::vector<glm::vec3>& rays,
                               const bool inverse, const std::vector<float>& depth = {},
                               const std::vector<float>& alpha = {}) const {
            using lfs::core::Device;
            using lfs::core::Tensor;
            const auto& ops = *lfs::training::training_ops(GetParam()).training_image;
            EXPECT_TRUE(ops.roi_polygon);
            const int width = static_cast<int>(rays.size());
            std::vector<float> values;
            for (const auto& ray : rays)
                values.insert(values.end(), {ray.x, ray.y, ray.z});
            const std::vector<float> identity{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
            const auto w2c = Tensor::from_vector(identity, {size_t{4}, size_t{4}}, Device::GPU);
            const auto camera_pos = Tensor::zeros({size_t{3}}, Device::GPU);
            const auto rays_tensor =
                Tensor::from_vector(values, {size_t{1}, static_cast<size_t>(width), size_t{3}}, Device::GPU);
            Tensor depth_tensor, alpha_tensor;
            if (!depth.empty()) {
                depth_tensor = Tensor::from_vector(depth, {size_t{1}, static_cast<size_t>(width)}, Device::GPU);
                alpha_tensor = Tensor::from_vector(alpha, {size_t{1}, static_cast<size_t>(width)}, Device::GPU);
            }
            lfs::gpu_ops::RoiParams params{
                .image = {1, width},
                .intrinsics = {1.0f, 1.0f, 0.0f, 0.0f},
                .minimum = {-1.0f, -1.0f, -0.5f},
                .maximum = {1.0f, 1.0f, 0.5f},
                .outside_weight = 0.25f,
                .inverse = inverse,
                .polygon = kLOutline,
                .polygon_device = Tensor::from_vector(kLOutline, {kLOutline.size() / 2, size_t{2}}, Device::GPU)};
            const glm::mat4 world_to_polygon = glm::translate(glm::mat4(1.0f), camera_in_polygon);
            std::copy_n(&world_to_polygon[0][0], 16, params.world_to_cropbox.begin());
            auto output = Tensor::full({size_t{1}, static_cast<size_t>(width)}, -1.0f, Device::GPU);
            ops.roi(w2c, camera_pos, rays_tensor, depth_tensor, alpha_tensor, output, params);
            return output.cpu().to_vector();
        }

        std::optional<lfs::core::GpuBackendScope> scope_;
    };
} // namespace

// Catches a polygon ROI taken from the convex hull or bounds of the outline instead of the prism itself: from
// above, the ray through the notch meets neither the L nor its slab, though the hull of the projected corners
// covers it; from the side, a ray just over the cap passes above the L.
TEST_P(RoiPolygonWeightMapTest, PolygonRegionIsExactForAConcaveOutline) {
    // Above the outline, looking down at the points (x, y, 0).
    const std::vector<glm::vec3> down{{-0.5f, -0.5f, -10.0f}, {0.5f, 0.5f, -10.0f}, {0.5f, -0.5f, -10.0f}, {-0.5f, 0.5f, -10.0f}, {1.5f, 1.5f, -10.0f}};
    EXPECT_EQ(run({0.0f, 0.0f, 10.0f}, down, false), (std::vector<float>{1.0f, 0.25f, 1.0f, 1.0f, 0.25f}));
    EXPECT_EQ(run({0.0f, 0.0f, 10.0f}, down, true), (std::vector<float>{0.25f, 1.0f, 0.25f, 0.25f, 1.0f}));

    // Beside the outline at y = -0.5, 0.1 above the cap: level rays pass over the L, rays tilted into the slab cross it.
    const std::vector<glm::vec3> side{{1.0f, 0.0f, 0.0f}, {1.0f, 0.0f, -0.02f}, {1.0f, 0.0f, 0.02f}};
    EXPECT_EQ(run({-10.0f, -0.5f, 0.6f}, side, false), (std::vector<float>{0.25f, 1.0f, 0.25f}));
}

// Catches a camera inside a crop polygon weighing pixels by their rays instead of the surface they show, which
// counts the notch of the outline as ROI.
TEST_P(RoiPolygonWeightMapTest, CameraInsideThePolygonWeighsPixelsByTheSurfaceTheyShow) {
    // From inside the bottom arm: surfaces in the bottom arm, in the notch, and one pixel with too little alpha.
    const std::vector<glm::vec3> rays{{1.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 0.0f}};
    EXPECT_EQ(run({-0.5f, -0.5f, 0.0f}, rays, false, {1.0f, 1.0f, 0.1f}, {1.0f, 1.0f, 0.2f}),
              (std::vector<float>{1.0f, 0.25f, 0.25f}));
}

TEST_P(RoiPolygonWeightMapTest, InvalidLensRaysAndLowCoverageRespectInverse) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::vector<glm::vec3> rays{{nan, 0.0f, 1.0f}, {0.0f, 0.0f, 1.0f}};
    EXPECT_EQ(run({-0.5f, -0.5f, 0.0f}, rays, false, {0.0f, 0.1f}, {0.0f, 0.2f}),
              (std::vector<float>{0.25f, 0.25f}));
    EXPECT_EQ(run({-0.5f, -0.5f, 0.0f}, rays, true, {0.0f, 0.1f}, {0.0f, 0.2f}),
              (std::vector<float>{1.0f, 1.0f}));
}

INSTANTIATE_TEST_SUITE_P(Backends, RoiPolygonWeightMapTest,
                         testing::Values(lfs::core::GpuBackend::CUDA, lfs::core::GpuBackend::Vulkan, lfs::core::GpuBackend::Metal),
                         [](const testing::TestParamInfo<lfs::core::GpuBackend>& info) {
                             return std::string(lfs::core::gpu_backend_name(info.param));
                         });
