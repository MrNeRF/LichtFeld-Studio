/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/point_cloud.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "point_cloud_style.hpp"
#include "rendering/rendering.hpp"
#include <array>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>
#include <optional>
#include <regex>
#include <string>

TEST(PointCloudSelectionOverlay, CommittedAndTransientColorsReachRasterImage) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
        GTEST_SKIP() << "requires CUDA";
    using namespace lfs::core;
    using namespace lfs::rendering;
    auto engine = RenderingEngine::create();
    lfs::core::PointCloud cloud;
    cloud.means = Tensor::from_vector(std::vector<float>{0.f, 0.f, -2.f}, {size_t{1}, size_t{3}}, Device::CPU);
    cloud.colors = Tensor::zeros({size_t{1}, size_t{3}}, Device::CPU);
    PointCloudRenderRequest request;
    request.frame_view.size = {64, 64};
    request.overlay.selection_colors[3] = {1.f, 0.f, 0.f, 1.f};
    request.overlay.selection_colors[kSelectionPreviewColorIndex] = {0.f, 1.f, 0.f, 1.f};
    request.overlay.selection_mask = std::make_shared<Tensor>(
        Tensor::from_vector(std::vector<int>{3}, {size_t{1}}, Device::CPU).to(DataType::UInt8));
    const auto image_max = [](const auto& result, size_t channel) {
        return result.image->slice(0, channel, channel + 1).max_scalar();
    };
    auto committed = engine->renderPointCloudImage(cloud, request);
    ASSERT_TRUE(committed) << committed.error();
    EXPECT_NEAR(image_max(*committed, 0), .75f, .01f);
    EXPECT_LT(image_max(*committed, 1), .01f);
    Tensor preview = Tensor::ones({size_t{1}}, Device::CPU, DataType::UInt8);
    request.overlay.transient_mask.mask = &preview;
    request.overlay.transient_mask.additive = false;
    auto subtract = engine->renderPointCloudImage(cloud, request);
    ASSERT_TRUE(subtract) << subtract.error();
    EXPECT_NEAR(image_max(*subtract, 1), .9f, .01f);
    request.overlay.selection_mask.reset();
    request.overlay.transient_mask.additive = true;
    auto add = engine->renderPointCloudImage(cloud, request);
    ASSERT_TRUE(add) << add.error();
    EXPECT_NEAR(image_max(*add, 1), .9f, .01f);
    request.overlay.transient_mask = {};
    auto plain = engine->renderPointCloudImage(cloud, request);
    ASSERT_TRUE(plain) << plain.error();
    EXPECT_LT(image_max(*plain, 0), .01f);
    EXPECT_LT(image_max(*plain, 1), .01f);
    // A mask sized for another point count is skipped, not an error.
    request.overlay.selection_mask = std::make_shared<Tensor>(
        Tensor::from_vector(std::vector<int>{3, 3}, {size_t{2}}, Device::CPU).to(DataType::UInt8));
    auto stale = engine->renderPointCloudImage(cloud, request);
    ASSERT_TRUE(stale) << stale.error();
    EXPECT_LT(image_max(*stale, 0), .01f);
}

TEST(PointCloudSelectionOverlay, SoftwareStyleMatchesViewportShader) {
    std::ifstream file(std::filesystem::path(PROJECT_ROOT_PATH) /
                       "src/visualizer/rendering/resources/viewport/point_cloud.vert");
    ASSERT_TRUE(file);
    const std::string shader(std::istreambuf_iterator<char>(file), {});
    const auto shader_value = [&](const std::string& name) {
        std::smatch match;
        const std::regex constant("const (?:float|uint) " + name + " = ([0-9.]+)u?;");
        return std::regex_search(shader, match, constant) ? std::stod(match[1]) : -1.0;
    };
    using namespace lfs::rendering::pcraster;
    EXPECT_EQ(shader_value("SELECTION_PREVIEW_COLOR_INDEX"), SELECTION_PREVIEW_COLOR_INDEX);
    EXPECT_EQ(lfs::rendering::kSelectionPreviewColorIndex, SELECTION_PREVIEW_COLOR_INDEX);
    EXPECT_FLOAT_EQ(shader_value("DESATURATE_BLEND"), DESATURATE_BLEND);
    EXPECT_FLOAT_EQ(shader_value("SELECTION_COMMITTED_BLEND"), SELECTION_COMMITTED_BLEND);
    EXPECT_FLOAT_EQ(shader_value("SELECTION_PREVIEW_BLEND"), SELECTION_PREVIEW_BLEND);
    EXPECT_FLOAT_EQ(shader_value("SELECTION_COMMITTED_SCALE"), SELECTION_COMMITTED_SCALE);
    EXPECT_FLOAT_EQ(shader_value("SELECTION_PREVIEW_SCALE"), SELECTION_PREVIEW_SCALE);
}

namespace {
    class PointCloudCropPolygonRender : public testing::TestWithParam<lfs::core::GpuBackend> {
    protected:
        void SetUp() override {
            if (!lfs::core::gpu_backend_available(GetParam()))
                GTEST_SKIP() << "Backend unavailable";
            scope_.emplace(GetParam());
        }
        std::optional<lfs::core::GpuBackendScope> scope_;
    };
} // namespace

// Catches a point-cloud render (video export, Python rendering) that ignores a crop polygon or crops by its bounds:
// of an L outline only the point in its arm stays; the points in its notch and past its slab go. Vulkan renders
// through the tensor rasterizer, which hides them through the deleted mask.
TEST_P(PointCloudCropPolygonRender, KeepsOnlyPointsInsideThePrism) {
    using namespace lfs::core;
    using namespace lfs::rendering;
    auto engine = RenderingEngine::create();
    PointCloud cloud;
    // Red in the bottom arm, green in the notch, blue over the upper arm but past the slab.
    cloud.means = Tensor::from_vector(std::vector<float>{0.5f, -0.5f, -5.0f, 0.5f, 0.5f, -5.0f, -0.5f, 0.5f, -8.0f},
                                      {size_t{3}, size_t{3}}, Device::CPU);
    cloud.colors = Tensor::from_vector(std::vector<float>{1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f},
                                       {size_t{3}, size_t{3}}, Device::CPU);
    PointCloudRenderRequest request;
    request.frame_view.size = {64, 64};
    // The prism sits at z = -5 with |z| <= 1.
    request.filters.crop_polygon = CropPolygon{
        .vertices = {{-1.0f, -1.0f}, {1.0f, -1.0f}, {1.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 1.0f}, {-1.0f, 1.0f}},
        .half_depth = 1.0f,
        .transform = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 5.0f))};
    const auto seen = [&](const bool inverse) {
        request.filters.crop_inverse = inverse;
        auto result = engine->renderPointCloudImage(cloud, request);
        EXPECT_TRUE(result) << (result ? "" : result.error());
        std::array<bool, 3> visible{};
        if (!result)
            return visible;
        for (size_t channel = 0; channel < 3; ++channel)
            visible[channel] = result->image->slice(0, channel, channel + 1).max_scalar() > 0.5f;
        return visible;
    };
    EXPECT_EQ(seen(false), (std::array<bool, 3>{true, false, false}));
    EXPECT_EQ(seen(true), (std::array<bool, 3>{false, true, true}));
    if (GetParam() != GpuBackend::CUDA) {
        // Without a polygon crop of its own, the tensor rasterizer refuses to dim outside points.
        request.filters.crop_desaturate = true;
        EXPECT_FALSE(engine->renderPointCloudImage(cloud, request));
    }
}

INSTANTIATE_TEST_SUITE_P(Backends, PointCloudCropPolygonRender,
                         testing::Values(lfs::core::GpuBackend::CUDA, lfs::core::GpuBackend::Vulkan),
                         [](const testing::TestParamInfo<lfs::core::GpuBackend>& info) {
                             return std::string(lfs::core::gpu_backend_name(info.param));
                         });
