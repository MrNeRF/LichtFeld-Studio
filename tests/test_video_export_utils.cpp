/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/mesh_data.hpp"
#include "core/path_utils.hpp"
#include "core/point_cloud.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "gui/video_export_utils.hpp"
#include "io/loader.hpp"
#include "io/video/video_encoder.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "rendering/cuda_vulkan_interop.hpp"
#include "rendering/rendering_manager.hpp"
#include "rendering/vksplat_viewport_renderer.hpp"
#include "rendering/vulkan_external_tensor.hpp"
#include "scene/scene_manager.hpp"
#include "scene/scene_render_state.hpp"
#include "visualizer/gui_capabilities.hpp"
#include "window/vulkan_context.hpp"
#include <SDL3/SDL.h>
#include <algorithm>

#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

using lfs::core::Device;
using lfs::core::Tensor;

namespace {

    std::unique_ptr<lfs::core::SplatData> make_test_splat(const std::vector<float>& xyz,
                                                          const float log_scale = 0.0f) {
        const size_t count = xyz.size() / 3;
        auto means = Tensor::from_vector(xyz, {count, size_t{3}}, Device::CUDA).to(lfs::core::DataType::Float32);
        auto sh0 = Tensor::zeros({count, size_t{1}, size_t{3}}, Device::CUDA, lfs::core::DataType::Float32);
        auto shN = Tensor::zeros({count, size_t{3}, size_t{3}}, Device::CUDA, lfs::core::DataType::Float32);
        auto scaling = Tensor::full({count, size_t{3}}, log_scale, Device::CUDA,
                                    lfs::core::DataType::Float32);

        std::vector<float> rotation_data(count * 4, 0.0f);
        for (size_t i = 0; i < count; ++i) {
            rotation_data[i * 4] = 1.0f;
        }
        auto rotation = Tensor::from_vector(rotation_data, {count, size_t{4}}, Device::CUDA).to(lfs::core::DataType::Float32);
        auto opacity = Tensor::zeros({count, size_t{1}}, Device::CUDA, lfs::core::DataType::Float32);

        return std::make_unique<lfs::core::SplatData>(
            1,
            std::move(means),
            std::move(sh0),
            std::move(shN),
            std::move(scaling),
            std::move(rotation),
            std::move(opacity),
            1.0f);
    }

    std::shared_ptr<lfs::core::PointCloud> make_test_point_cloud() {
        auto means = Tensor::from_vector(
                         std::vector<float>{0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f},
                         {size_t{2}, size_t{3}},
                         Device::CUDA)
                         .to(lfs::core::DataType::Float32);
        auto colors = Tensor::from_vector(
                          std::vector<float>{1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f},
                          {size_t{2}, size_t{3}},
                          Device::CUDA)
                          .to(lfs::core::DataType::Float32);
        return std::make_shared<lfs::core::PointCloud>(std::move(means), std::move(colors));
    }

    std::shared_ptr<lfs::core::MeshData> make_test_mesh() {
        auto mesh = std::make_shared<lfs::core::MeshData>();
        mesh->vertices = Tensor::from_vector(
                             std::vector<float>{
                                 0.0f,
                                 0.0f,
                                 0.0f,
                                 1.0f,
                                 0.0f,
                                 0.0f,
                                 0.0f,
                                 1.0f,
                                 0.0f,
                             },
                             {size_t{3}, size_t{3}},
                             Device::CPU)
                             .to(lfs::core::DataType::Float32);
        mesh->indices = Tensor::from_vector(
                            std::vector<int>{0, 1, 2},
                            {size_t{1}, size_t{3}},
                            Device::CPU)
                            .to(lfs::core::DataType::Int32);
        return mesh;
    }

    void expect_translation(const glm::mat4& transform, const glm::vec3& expected) {
        EXPECT_FLOAT_EQ(transform[3][0], expected.x);
        EXPECT_FLOAT_EQ(transform[3][1], expected.y);
        EXPECT_FLOAT_EQ(transform[3][2], expected.z);
    }

    void expect_visualizer_translation_from_data(const glm::mat4& transform, const glm::vec3& data_translation) {
        expect_translation(transform, lfs::rendering::visualizerWorldPointFromDataWorld(data_translation));
    }

} // namespace

TEST(VideoExportUtilsTest, ImmediateSequenceCaptureUsesCurrentGeometryWithoutCopying) {
    lfs::vis::SceneManager manager;
    auto& scene = manager.getScene();
    const auto first = scene.addSplat("frame_0", make_test_splat({1.0f, 2.0f, 3.0f}));
    const auto second = scene.addSplat("frame_1", make_test_splat({4.0f, 5.0f, 6.0f}));
    scene.setNodeVisibility(second, false);
    auto owned = lfs::vis::gui::captureVideoExportSceneSnapshot(manager);
    ASSERT_TRUE(owned);
    ASSERT_TRUE(owned->combined_model);
    EXPECT_EQ(owned->borrowed_model, nullptr);
    EXPECT_NE(owned->gaussianModel()->means().data_ptr(), manager.getModelForRendering()->means().data_ptr());
    for (const bool show_first : {true, false, true}) {
        scene.setNodeVisibility(first, show_first);
        scene.setNodeVisibility(second, !show_first);
        auto current = lfs::vis::gui::captureVideoExportSceneSnapshot(
            manager, lfs::vis::gui::VideoExportCapture::ImmediateRender);
        ASSERT_TRUE(current);
        EXPECT_FALSE(current->combined_model);
        EXPECT_EQ(current->gaussianModel(), manager.getModelForRendering());
        EXPECT_EQ(current->gaussianModel()->means().data_ptr(), manager.getModelForRendering()->means().data_ptr());
        EXPECT_FLOAT_EQ(current->gaussianModel()->means().to(lfs::core::Device::CPU).ptr<float>()[0], show_first ? 1.0f : 4.0f);
    }
    // The owned path remains immutable across sequence switches.
    EXPECT_FLOAT_EQ(owned->gaussianModel()->means().to(lfs::core::Device::CPU).ptr<float>()[0], 1.0f);
}

TEST(VideoExportUtilsTest, ImmediateCaptureBorrowsPointCloudAndMeshWithTheirTransforms) {
    lfs::vis::SceneManager manager;
    auto& scene = manager.getScene();
    const auto cloud = make_test_point_cloud();
    const auto mesh = make_test_mesh();
    scene.addPointCloud("cloud", cloud);
    scene.addMesh("mesh", mesh);
    scene.setNodeTransform("cloud", glm::translate(glm::mat4(1.0f), glm::vec3(1.0f, 2.0f, 3.0f)));
    auto owned = lfs::vis::gui::captureVideoExportSceneSnapshot(manager);
    auto borrowed = lfs::vis::gui::captureVideoExportSceneSnapshot(manager, lfs::vis::gui::VideoExportCapture::ImmediateRender);
    ASSERT_TRUE(owned);
    ASSERT_TRUE(borrowed);
    ASSERT_TRUE(borrowed->pointCloud());
    EXPECT_FALSE(borrowed->point_cloud);
    EXPECT_EQ(borrowed->pointCloud()->means.data_ptr(), cloud->means.data_ptr());
    EXPECT_NE(owned->pointCloud()->means.data_ptr(), cloud->means.data_ptr());
    EXPECT_EQ(borrowed->point_cloud_transform, owned->point_cloud_transform);
    ASSERT_EQ(borrowed->meshes.size(), 1u);
    EXPECT_EQ(borrowed->meshes[0].meshData(), mesh.get());
    EXPECT_FALSE(borrowed->meshes[0].mesh);
    EXPECT_NE(owned->meshes[0].meshData(), mesh.get());
}

TEST(VideoExportUtilsTest, CaptureSnapshotUsesRenderableModelAndTransforms) {
    lfs::vis::SceneManager scene_manager;
    auto& scene = scene_manager.getScene();

    scene.addSplat("left", make_test_splat({0.0f, 0.0f, 0.0f}));
    scene.addSplat("right", make_test_splat({0.0f, 0.0f, 0.0f}));
    scene.setNodeTransform("left", glm::translate(glm::mat4(1.0f), glm::vec3(1.0f, 2.0f, 3.0f)));
    scene.setNodeTransform("right", glm::translate(glm::mat4(1.0f), glm::vec3(-4.0f, 0.5f, 2.0f)));

    auto snapshot_result = lfs::vis::gui::captureVideoExportSceneSnapshot(scene_manager);
    ASSERT_TRUE(snapshot_result.has_value()) << snapshot_result.error();

    const auto& snapshot = *snapshot_result;
    ASSERT_TRUE(snapshot.combined_model);
    EXPECT_EQ(snapshot.combined_model->size(), 2u);
    EXPECT_FALSE(snapshot.point_cloud);
    ASSERT_EQ(snapshot.model_transforms.size(), 2u);
    expect_visualizer_translation_from_data(snapshot.model_transforms[0], {1.0f, 2.0f, 3.0f});
    expect_visualizer_translation_from_data(snapshot.model_transforms[1], {-4.0f, 0.5f, 2.0f});

    ASSERT_TRUE(snapshot.transform_indices);
    EXPECT_EQ(snapshot.transform_indices->cpu().to_vector_int(), (std::vector<int>{0, 1}));
}

TEST(VideoExportUtilsTest, UnconsolidatedAllNodeCacheVisibilityReachesCudaAndVulkan) {
    if (!SDL_InitSubSystem(SDL_INIT_VIDEO))
        GTEST_SKIP() << SDL_GetError();
    struct VideoGuard {
        ~VideoGuard() { SDL_QuitSubSystem(SDL_INIT_VIDEO); }
    } video_guard;
    std::unique_ptr<SDL_Window, decltype(&SDL_DestroyWindow)> window(
        SDL_CreateWindow("Cache visibility test", 64, 64, SDL_WINDOW_VULKAN | SDL_WINDOW_HIDDEN),
        SDL_DestroyWindow);
    ASSERT_TRUE(window) << SDL_GetError();
    lfs::vis::VulkanContext context;
    ASSERT_TRUE(context.init(window.get(), 64, 64)) << context.lastError();
    lfs::rendering::setExpectedVulkanDeviceUuid(context.deviceUUID());
    lfs::vis::SceneManager manager;
    auto& scene = manager.getScene();
    const auto first = scene.addSplat("first", make_test_splat({-0.5f, 0.0f, 3.0f}));
    const auto second = scene.addSplat("second", make_test_splat({0.5f, 0.0f, 3.0f}));
    scene.setNodeVisibility(first, false);
    scene.setCombinedModelAllocator([&context](lfs::core::TensorShape shape, size_t capacity,
                                               lfs::core::DataType dtype, std::string_view name) {
        auto tensor = lfs::vis::makeVulkanExternalTensor(context, std::move(shape), dtype, capacity,
                                                         std::string(name).c_str());
        if (!tensor)
            throw std::runtime_error(tensor.error());
        return std::move(*tensor);
    });
    scene.requestCombinedModelBuild(true);
    while (scene.combinedModelBuildPending()) {
        (void)scene.combinedModelBuildError();
        std::this_thread::yield();
    }
    const auto* model = scene.getCombinedModel();
    ASSERT_TRUE(model);
    ASSERT_EQ(model->size(), 2u);
    ASSERT_FALSE(scene.isConsolidated());
    auto transforms = scene.getVisibleNodeTransforms();
    for (auto& transform : transforms)
        transform = lfs::rendering::dataWorldTransformToVisualizerWorld(transform);
    const auto indices = scene.getTransformIndices();
    ASSERT_EQ(transforms.size(), 2u);
    lfs::vis::VksplatViewportRenderer renderer;
    auto cuda_renderer = lfs::rendering::RenderingEngine::create();
    ASSERT_TRUE(cuda_renderer->initialize());
    const auto slot = lfs::vis::VksplatViewportRenderer::OutputSlot::Preview;
    for (int backend = 0; backend < 3; ++backend) {
        SCOPED_TRACE(backend); // CUDA point rendering, Vulkan 3DGS and 3DGUT.
        const auto render = [&]() -> std::vector<float> {
            EXPECT_EQ(scene.getCombinedModel(), model);
            const auto mask = scene.getNodeVisibilityMask();
            if (backend == 0) {
                lfs::rendering::PointCloudRenderRequest request;
                request.frame_view.size = {64, 64};
                request.render.voxel_size = 0.1f;
                request.scene = {&transforms, indices, mask};
                auto image = cuda_renderer->renderPointCloudImage(*model, request);
                EXPECT_TRUE(image) << (image ? "" : image.error());
                return image ? image->image->cpu().to_vector() : std::vector<float>{};
            }
            lfs::rendering::ViewportRenderRequest request;
            request.deterministic_export = true;
            request.frame_view.size = {64, 64};
            request.sh_degree = 0;
            request.scene = {&transforms, indices, mask};
            request.raster_backend = backend == 1 ? lfs::rendering::GaussianRasterBackend::ThreeDgs
                                                  : lfs::rendering::GaussianRasterBackend::ThreeDgut;
            request.gut = backend == 2;
            auto result = renderer.render(context, *model, request, false, slot, false);
            EXPECT_TRUE(result) << (result ? "" : result.error());
            if (!result)
                return {};
            auto image = renderer.readOutputImageRgb8(context, slot);
            EXPECT_TRUE(image) << (image ? "" : image.error());
            if (!image)
                return {};
            const auto* bytes = (*image)->ptr<uint8_t>();
            return {bytes, bytes + (*image)->numel()};
        };
        manager.setNodeVisibility(first, true);
        ASSERT_EQ(scene.getCombinedModel(), model);
        const auto shown = render();
        ASSERT_FALSE(shown.empty());
        EXPECT_GT(*std::max_element(shown.begin(), shown.end()), 0);
        manager.setNodeVisibility(first, false);
        ASSERT_EQ(scene.getCombinedModel(), model);
        const auto hidden = render();
        EXPECT_NE(hidden, shown);
        manager.setNodeVisibilityTransient(second, false);
        ASSERT_EQ(scene.getCombinedModel(), model);
        const auto empty = render();
        EXPECT_TRUE(std::all_of(empty.begin(), empty.end(), [](float v) { return v == 0; }));
        manager.setNodeVisibilityTransient(second, true);
        ASSERT_EQ(scene.getCombinedModel(), model);
        EXPECT_EQ(render(), hidden);
        manager.setNodeVisibility(first, true);
        ASSERT_EQ(scene.getCombinedModel(), model);
        EXPECT_EQ(render(), shown);
    }
    cuda_renderer->shutdown();
}

TEST(VideoExportUtilsTest, EquirectangularMarkerPixelsFollowViewportCamera) {
    ASSERT_TRUE(SDL_InitSubSystem(SDL_INIT_VIDEO)) << SDL_GetError();
    struct VideoGuard {
        ~VideoGuard() { SDL_QuitSubSystem(SDL_INIT_VIDEO); }
    } video_guard;
    std::unique_ptr<SDL_Window, decltype(&SDL_DestroyWindow)> window(
        SDL_CreateWindow("Equirectangular markers", 64, 64, SDL_WINDOW_VULKAN | SDL_WINDOW_HIDDEN), SDL_DestroyWindow);
    ASSERT_TRUE(window) << SDL_GetError();
    lfs::vis::VulkanContext context;
    ASSERT_TRUE(context.init(window.get(), 64, 64)) << context.lastError();
    lfs::rendering::setExpectedVulkanDeviceUuid(context.deviceUUID());

    constexpr std::array points{std::array{0.0f, 0.0f, -5.0f}, std::array{0.0f, 0.0f, 5.0f},
                                std::array{0.0f, -5.0f, 0.0f}, std::array{0.0f, 5.0f, 0.0f},
                                std::array{0.0f, 0.5f, 5.0f}};
    lfs::vis::SceneManager manager;
    auto& scene = manager.getScene();
    std::array<lfs::core::NodeId, points.size()> ids;
    for (size_t i = 0; i < points.size(); ++i) {
        const auto p = lfs::rendering::visualizerWorldPointFromDataWorld(
            glm::vec3{points[i][0], points[i][1], points[i][2]});
        ids[i] = scene.addSplat(std::format("marker_{}", i), make_test_splat({p.x, p.y, p.z}, -2.0f));
        ASSERT_NE(ids[i], lfs::core::NULL_NODE);
    }
    scene.setCombinedModelAllocator([&context](lfs::core::TensorShape shape, size_t capacity,
                                               lfs::core::DataType dtype, std::string_view name) {
        auto tensor = lfs::vis::makeVulkanExternalTensor(context, std::move(shape), dtype, capacity,
                                                         std::string(name).c_str());
        if (!tensor)
            throw std::runtime_error(tensor.error());
        return std::move(*tensor);
    });
    scene.requestCombinedModelBuild(true);
    while (scene.combinedModelBuildPending()) {
        (void)scene.combinedModelBuildError();
        std::this_thread::yield();
    }
    const auto* model = scene.getCombinedModel();
    ASSERT_TRUE(model);
    auto transforms = scene.getVisibleNodeTransforms();
    for (auto& transform : transforms)
        transform = lfs::rendering::dataWorldTransformToVisualizerWorld(transform);
    const auto indices = scene.getTransformIndices();
    lfs::vis::VksplatViewportRenderer renderer;

    const auto check_marker = [&](size_t marker, bool panorama) {
        for (size_t i = 0; i < ids.size(); ++i)
            scene.setNodeVisibility(ids[i], i == marker);
        lfs::rendering::ViewportRenderRequest request;
        request.deterministic_export = true;
        request.frame_view.rotation = glm::mat3(-1.0f, 0.0f, 0.0f,
                                                0.0f, 1.0f, 0.0f,
                                                0.0f, 0.0f, -1.0f);
        request.frame_view.size = {1024, 512};
        request.frame_view.focal_length_mm = 50.0f;
        request.raster_backend = lfs::rendering::GaussianRasterBackend::ThreeDgut;
        request.gut = true;
        request.equirectangular = panorama;
        request.sh_degree = 0;
        request.scene = {&transforms, indices, scene.getNodeVisibilityMask()};
        auto result = renderer.render(context, *model, request, false,
                                      lfs::vis::VksplatViewportRenderer::OutputSlot::Preview, false);
        ASSERT_TRUE(result) << (result ? "" : result.error());
        auto image = renderer.readOutputImageRgb8(context, lfs::vis::VksplatViewportRenderer::OutputSlot::Preview);
        ASSERT_TRUE(image) << (image ? "" : image.error());
        const auto* bytes = (*image)->ptr<uint8_t>();
        if (marker == 4) {
            // Up is up in both projections: a point above the horizon draws above the centre row.
            int lit_rows = 0;
            for (int y = 0; y < 512; ++y)
                for (int x = 512 - 32; x <= 512 + 32; ++x) {
                    const size_t offset = (static_cast<size_t>(y) * 1024 + x) * 3;
                    if (bytes[offset] + bytes[offset + 1] + bytes[offset + 2] > 24) {
                        EXPECT_LT(y, 256) << "panorama=" << panorama;
                        ++lit_rows;
                        break;
                    }
                }
            EXPECT_GT(lit_rows, 0) << "panorama=" << panorama;
            return;
        }
        const int expected_y = panorama ? std::array{256, 256, 511, 0}[marker] : 256;
        const bool pole = panorama && (marker == 2 || marker == 3);
        const bool seam = panorama && marker == 0;
        constexpr int expected_x = 512;
        bool found = false;
        for (int y = std::max(0, expected_y - 16); y <= std::min(511, expected_y + 16) && !found; ++y)
            for (int x = 0; x < 1024; ++x) {
                const bool x_matches = pole || (seam ? x < 16 || x >= 1008 : std::abs(x - expected_x) <= 16);
                const size_t offset = (static_cast<size_t>(y) * 1024 + x) * 3;
                if (x_matches && bytes[offset] + bytes[offset + 1] + bytes[offset + 2] > 24) {
                    found = true;
                    break;
                }
            }
        EXPECT_TRUE(found) << "marker=" << marker << " panorama=" << panorama;
    };
    for (bool panorama : {false, true})
        for (size_t marker = 0; marker < points.size(); ++marker)
            if (panorama || marker == 1 || marker == 4)
                check_marker(marker, panorama);

    for (const auto id : ids)
        scene.setNodeVisibility(id, true);
    lfs::vis::VksplatViewportRenderer::SelectionMaskRequest selection;
    selection.frame_view.rotation = glm::mat3(-1.0f, 0.0f, 0.0f,
                                              0.0f, 1.0f, 0.0f,
                                              0.0f, 0.0f, -1.0f);
    selection.frame_view.size = {1024, 512};
    selection.frame_view.focal_length_mm = 50.0f;
    selection.scene.model_transforms = &transforms;
    selection.scene.transform_indices = indices;
    selection.scene.node_visibility_mask = scene.getNodeVisibilityMask();
    selection.gut = true;
    selection.equirectangular = true;
    for (const int x : {0, 1023}) {
        selection.primitives = {{static_cast<float>(x), 256.0f, 64.0f, 0.0f}};
        auto mask = renderer.buildSelectionMask(context, *model, selection, false);
        ASSERT_TRUE(mask) << mask.error();
        EXPECT_EQ(mask->cpu().ptr<uint8_t>()[0], 1u) << "seam click x=" << x;
    }

    // Rectangle and lasso must select the seam marker from either wrapped copy,
    // and the pole markers on the top and bottom edges.
    const auto selected_at = [&](const auto shape, const glm::vec4 primitive, const int index) {
        selection.shape = shape;
        selection.primitives = {primitive};
        selection.polygon_vertices = {{primitive.x, primitive.y},
                                      {primitive.z, primitive.y},
                                      {primitive.z, primitive.w},
                                      {primitive.x, primitive.w}};
        auto mask = renderer.buildSelectionMask(context, *model, selection, false);
        EXPECT_TRUE(mask) << mask.error();
        return mask ? mask->cpu().ptr<uint8_t>()[index] : static_cast<uint8_t>(255);
    };
    using Shape = lfs::vis::VksplatViewportRenderer::SelectionMaskShape;
    for (const float x0 : {0.0f, 960.0f}) {
        const glm::vec4 strip{x0, 232.0f, x0 + 64.0f, 280.0f};
        EXPECT_EQ(selected_at(Shape::Rectangle, strip, 0), 1u) << "rect x0=" << x0;
        EXPECT_EQ(selected_at(Shape::Polygon, strip, 0), 1u) << "lasso x0=" << x0;
    }
    const glm::vec4 middle{480.0f, 232.0f, 544.0f, 280.0f};
    EXPECT_EQ(selected_at(Shape::Rectangle, middle, 0), 0u);
    EXPECT_EQ(selected_at(Shape::Rectangle, middle, 1), 1u);
    EXPECT_EQ(selected_at(Shape::Polygon, middle, 1), 1u);
    EXPECT_EQ(selected_at(Shape::Brush, {512.0f, 0.0f, 90.0f * 90.0f, 0.0f}, 3), 1u);
    EXPECT_EQ(selected_at(Shape::Brush, {512.0f, 511.0f, 90.0f * 90.0f, 0.0f}, 2), 1u);
    EXPECT_EQ(selected_at(Shape::Rectangle, {0.0f, 0.0f, 1024.0f, 48.0f}, 3), 1u);
    EXPECT_EQ(selected_at(Shape::Rectangle, {0.0f, 464.0f, 1024.0f, 512.0f}, 2), 1u);
    EXPECT_EQ(selected_at(Shape::Polygon, {0.0f, 0.0f, 1024.0f, 48.0f}, 3), 1u);
    EXPECT_EQ(selected_at(Shape::Polygon, {0.0f, 464.0f, 1024.0f, 512.0f}, 2), 1u);
}

TEST(VideoExportUtilsTest, HiddenDatasetSnapshotCullsImplicitNodeInPreview) {
    if (!SDL_InitSubSystem(SDL_INIT_VIDEO))
        GTEST_SKIP() << "SDL video unavailable: " << SDL_GetError();
    struct VideoGuard {
        ~VideoGuard() { SDL_QuitSubSystem(SDL_INIT_VIDEO); }
    } video_guard;
    std::unique_ptr<SDL_Window, decltype(&SDL_DestroyWindow)> window(
        SDL_CreateWindow("Video visibility test", 64, 64, SDL_WINDOW_VULKAN | SDL_WINDOW_HIDDEN),
        SDL_DestroyWindow);
    if (!window)
        GTEST_SKIP() << "Vulkan window unavailable: " << SDL_GetError();
    lfs::vis::VulkanContext context;
    if (!context.init(window.get(), 64, 64))
        GTEST_SKIP() << "Vulkan unavailable: " << context.lastError();

    lfs::rendering::setExpectedVulkanDeviceUuid(context.deviceUUID());

    lfs::vis::SceneManager manager;
    manager.changeContentType(lfs::vis::SceneManager::ContentType::Dataset);
    auto& scene = manager.getScene();
    const auto parent = scene.addGroup("Dataset");
    scene.addSplat("Model", make_test_splat({0.0f, 0.0f, 3.0f}), parent);
    scene.setTrainingModelNode("Model");
    scene.setNodeVisibility(parent, false);
    auto snapshot = lfs::vis::gui::captureVideoExportSceneSnapshot(manager);
    ASSERT_TRUE(snapshot) << snapshot.error();
    ASSERT_EQ(snapshot->node_visibility_mask, (std::vector<bool>{false}));
    ASSERT_FALSE(snapshot->transform_indices);
    const auto allocator = [&context](lfs::core::TensorShape shape, size_t capacity,
                                      lfs::core::DataType dtype, std::string_view name) {
        auto tensor = lfs::vis::makeVulkanExternalTensor(context, std::move(shape), dtype, capacity,
                                                         std::string(name).c_str());
        if (!tensor)
            throw std::runtime_error(tensor.error());
        return std::move(*tensor);
    };
    auto migrated = lfs::io::migrateSplatTensorsToAllocator(*snapshot->combined_model, allocator);
    ASSERT_TRUE(migrated) << migrated.error().format();

    lfs::vis::VksplatViewportRenderer renderer;
    const auto slot = lfs::vis::VksplatViewportRenderer::OutputSlot::Preview;
    lfs::rendering::ViewportRenderRequest request;
    request.deterministic_export = true;
    request.frame_view.size = {64, 64};
    request.sh_degree = 0;
    request.scene.model_transforms = &snapshot->model_transforms;
    auto explicit_indices = std::make_shared<Tensor>(
        Tensor::zeros({size_t{1}}, Device::CUDA, lfs::core::DataType::Int32));
    const auto render = [&](const std::vector<bool>& mask, const bool indexed) {
        request.scene.node_visibility_mask = mask;
        request.scene.transform_indices = indexed ? explicit_indices : nullptr;
        auto rendered = renderer.render(context, *snapshot->combined_model, request, false, slot, false);
        EXPECT_TRUE(rendered) << (rendered ? "" : rendered.error());
        if (!rendered)
            return std::vector<uint8_t>{};
        auto image = renderer.readOutputImageRgb8(context, slot);
        EXPECT_TRUE(image) << (image ? "" : image.error());
        if (!image)
            return std::vector<uint8_t>{};
        const auto* bytes = (*image)->ptr<uint8_t>();
        return std::vector<uint8_t>(bytes, bytes + (*image)->numel());
    };
    for (const auto backend : {lfs::rendering::GaussianRasterBackend::ThreeDgs,
                               lfs::rendering::GaussianRasterBackend::ThreeDgut}) {
        request.raster_backend = backend;
        request.gut = lfs::rendering::isGutBackend(backend);
        SCOPED_TRACE(static_cast<int>(backend));
        const auto visible = render({}, false);
        ASSERT_EQ(visible.size(), 64u * 64u * 3u);
        ASSERT_GT(*std::max_element(visible.begin(), visible.end()), 0);
        EXPECT_EQ(render({true}, false), visible);
        EXPECT_EQ(render({true}, true), visible);
        const auto hidden_reference = render({false}, true);
        ASSERT_EQ(hidden_reference.size(), visible.size());
        EXPECT_TRUE(std::all_of(hidden_reference.begin(), hidden_reference.end(), [](uint8_t v) { return v == 0; }));
        EXPECT_EQ(render(snapshot->node_visibility_mask, false), hidden_reference);
        EXPECT_EQ(render({}, false), visible);

        lfs::vis::VksplatViewportRenderer::SelectionMaskRequest selection;
        selection.frame_view = request.frame_view;
        selection.scene.model_transforms = &snapshot->model_transforms;
        selection.gut = request.gut;
        selection.primitives = {{32.0f, 32.0f, 64.0f, 0.0f}};
        for (const bool indexed : {true, false}) {
            selection.scene.transform_indices = indexed ? explicit_indices : nullptr;
            for (const bool is_visible : {true, false}) {
                selection.scene.node_visibility_mask = {is_visible};
                auto mask = renderer.buildSelectionMask(context, *snapshot->combined_model, selection, false);
                ASSERT_TRUE(mask) << mask.error();
                EXPECT_EQ(mask->cpu().ptr<uint8_t>()[0], is_visible ? 1 : 0);
            }
        }

        std::vector<double> visible_times;
        for (int batch = 0; batch < 7; ++batch) {
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < 20; ++i)
                (void)render({}, false);
            visible_times.push_back(std::chrono::duration<double, std::micro>(
                                        std::chrono::steady_clock::now() - start)
                                        .count() /
                                    20.0);
        }
        std::sort(visible_times.begin(), visible_times.end());
        RecordProperty(request.gut ? "gut_visible_render_readback_us" : "gs_visible_render_readback_us",
                       std::to_string(visible_times[3]));
    }
}

namespace {
    class VideoExportBandTest : public ::testing::TestWithParam<std::tuple<bool, bool>> {
    protected:
        void SetUp() override {
            ASSERT_TRUE(SDL_InitSubSystem(SDL_INIT_VIDEO)) << SDL_GetError();
            window_.reset(SDL_CreateWindow("Export regression", 64, 64,
                                           SDL_WINDOW_VULKAN | SDL_WINDOW_HIDDEN));
            ASSERT_TRUE(window_) << SDL_GetError();
            ASSERT_TRUE(context_.init(window_.get(), 64, 64)) << context_.lastError();
            lfs::rendering::setExpectedVulkanDeviceUuid(context_.deviceUUID());
            // Include both poles, the longitude seam, and splats crossing band edges.
            model_ = make_test_splat({0, -8, -3, 0, -3, -3, 0, 0, -3,
                                      0, 3, -3, 0, 8, -3, 0.1f, 0, 3});
            const auto allocator = [this](lfs::core::TensorShape shape, size_t capacity,
                                          lfs::core::DataType dtype, std::string_view name) {
                auto tensor = lfs::vis::makeVulkanExternalTensor(
                    context_, std::move(shape), dtype, capacity, std::string(name).c_str());
                if (!tensor)
                    throw std::runtime_error(tensor.error());
                return std::move(*tensor);
            };
            auto migrated = lfs::io::migrateSplatTensorsToAllocator(*model_, allocator);
            ASSERT_TRUE(migrated) << migrated.error().format();
        }

        void TearDown() override {
            renderer_.reset();
            model_.reset();
            context_.shutdown();
            window_.reset();
            SDL_QuitSubSystem(SDL_INIT_VIDEO);
        }

        std::vector<uint8_t> render(int y, int rows, bool banded, bool full_frame) {
            lfs::rendering::ViewportRenderRequest request;
            request.deterministic_export = true;
            request.frame_view.size = {render_width_, rows};
            request.frame_view.subregion_origin = {0, y};
            if (banded)
                request.frame_view.subregion_full_size = {render_width_, render_height_};
            request.sh_degree = 0;
            request.gut = true;
            request.raster_backend = lfs::rendering::GaussianRasterBackend::ThreeDgut;
            request.equirectangular = std::get<0>(GetParam());
            const auto slot = full_frame ? lfs::vis::VksplatViewportRenderer::OutputSlot::Preview
                                         : lfs::vis::VksplatViewportRenderer::OutputSlot::Main;
            auto result = renderer_->render(context_, *model_, request, false, slot, false);
            EXPECT_TRUE(result) << (result ? "" : result.error());
            if (!result)
                return {};
            auto image = renderer_->readOutputImageRgb8(context_, slot);
            EXPECT_TRUE(image) << (image ? "" : image.error());
            if (!image)
                return {};
            const auto* data = (*image)->ptr<uint8_t>();
            return {data, data + (*image)->numel()};
        }

        int render_width_ = 1024;
        int render_height_ = 512;
        std::unique_ptr<SDL_Window, decltype(&SDL_DestroyWindow)> window_{nullptr, SDL_DestroyWindow};
        lfs::vis::VulkanContext context_;
        std::unique_ptr<lfs::core::SplatData> model_;
        std::unique_ptr<lfs::vis::VksplatViewportRenderer> renderer_ =
            std::make_unique<lfs::vis::VksplatViewportRenderer>();
    };

    TEST_P(VideoExportBandTest, BandedEqualsFullImage) {
        const bool full_frame = std::get<1>(GetParam());
        for (const int width : {1024, 2048}) {
            render_width_ = width;
            render_height_ = width / 2;
            SCOPED_TRACE(::testing::Message() << "extent=" << width << "x" << render_height_);
            const auto full = render(0, render_height_, false, full_frame);
            ASSERT_EQ(full.size(), size_t(width) * render_height_ * 3u);
            for (const int band_rows : {256, 64}) {
                for (int y = 0; y < render_height_; y += band_rows) {
                    SCOPED_TRACE(::testing::Message() << "rows=" << band_rows << " y=" << y);
                    const auto band = render(y, band_rows, true, full_frame);
                    ASSERT_EQ(band.size(), size_t(width) * band_rows * 3u);
                    const auto first = full.begin() + size_t(y) * width * 3;
                    ASSERT_GT(*std::max_element(first, first + band.size()), 0);
                    int max_diff = 0;
                    size_t changed = 0;
                    for (size_t i = 0; i < band.size(); ++i) {
                        max_diff = std::max(max_diff, std::abs(int(band[i]) - int(first[i])));
                        changed += band[i] != first[i];
                    }
                    EXPECT_LE(max_diff, 1) << "changed channels=" << changed;
                }
            }
        }
    }

    INSTANTIATE_TEST_SUITE_P(Projection, VideoExportBandTest,
                             ::testing::Combine(::testing::Bool(), ::testing::Bool()),
                             [](const auto& info) {
                                 return std::string(std::get<0>(info.param) ? "Equirectangular" : "Perspective") +
                                        (std::get<1>(info.param) ? "FullFrame" : "BandRelative");
                             });
} // namespace

TEST(VideoExportUtilsTest, CaptureSnapshotPrefersSplatsOverPointCloudAndKeepsMeshes) {
    lfs::vis::SceneManager scene_manager;
    auto& scene = scene_manager.getScene();

    scene.addSplat("splat", make_test_splat({0.0f, 0.0f, 0.0f}));
    scene.addPointCloud("points", make_test_point_cloud());
    scene.addMesh("mesh", make_test_mesh());

    auto snapshot_result = lfs::vis::gui::captureVideoExportSceneSnapshot(scene_manager);
    ASSERT_TRUE(snapshot_result.has_value()) << snapshot_result.error();

    const auto& snapshot = *snapshot_result;
    ASSERT_TRUE(snapshot.combined_model);
    EXPECT_FALSE(snapshot.point_cloud);
    ASSERT_EQ(snapshot.meshes.size(), 1u);
    ASSERT_TRUE(snapshot.meshes[0].mesh);
}

TEST(VideoExportUtilsTest, CaptureSnapshotKeepsPointCloudTransformWhenNoModelExists) {
    lfs::vis::SceneManager scene_manager;
    auto& scene = scene_manager.getScene();

    scene.addPointCloud("points", make_test_point_cloud());
    scene.setNodeTransform("points", glm::translate(glm::mat4(1.0f), glm::vec3(3.0f, -2.0f, 5.0f)));

    auto snapshot_result = lfs::vis::gui::captureVideoExportSceneSnapshot(scene_manager);
    ASSERT_TRUE(snapshot_result.has_value()) << snapshot_result.error();

    const auto& snapshot = *snapshot_result;
    EXPECT_FALSE(snapshot.combined_model);
    ASSERT_TRUE(snapshot.point_cloud);
    EXPECT_EQ(snapshot.point_cloud->size(), 2);
    expect_visualizer_translation_from_data(snapshot.point_cloud_transform, {3.0f, -2.0f, 5.0f});
}

TEST(VideoExportUtilsTest, CaptureSnapshotKeepsPointCloudCropBoxWithoutSplatParent) {
    lfs::vis::SceneManager scene_manager;
    auto& scene = scene_manager.getScene();

    const auto parent_id = scene.addPointCloud("points", make_test_point_cloud());
    ASSERT_NE(parent_id, lfs::core::NULL_NODE);
    auto cropbox_result = lfs::vis::cap::ensureCropBox(scene_manager, nullptr, parent_id);
    ASSERT_TRUE(cropbox_result) << cropbox_result.error();
    auto* cropbox_node = scene.getNodeById(*cropbox_result);
    ASSERT_NE(cropbox_node, nullptr);
    ASSERT_TRUE(cropbox_node->cropbox);
    cropbox_node->cropbox->enabled = true;
    cropbox_node->cropbox->inverse = true;
    cropbox_node->cropbox->min = {-1.0f, -2.0f, -3.0f};
    cropbox_node->cropbox->max = {1.0f, 2.0f, 3.0f};
    scene_manager.selectNode(cropbox_node->name);

    auto snapshot_result = lfs::vis::gui::captureVideoExportSceneSnapshot(scene_manager);
    ASSERT_TRUE(snapshot_result.has_value()) << snapshot_result.error();

    const auto& snapshot = *snapshot_result;
    ASSERT_TRUE(snapshot.point_cloud);
    ASSERT_EQ(snapshot.cropboxes.size(), 1u);
    EXPECT_EQ(snapshot.selected_cropbox_index, 0);
    EXPECT_LT(snapshot.cropboxes.front().parent_node_index, 0);
    EXPECT_TRUE(snapshot.cropboxes.front().has_data);
    EXPECT_TRUE(snapshot.cropboxes.front().data.enabled);
    EXPECT_TRUE(snapshot.cropboxes.front().data.inverse);
    EXPECT_EQ(snapshot.cropboxes.front().data.min, glm::vec3(-1.0f, -2.0f, -3.0f));
    EXPECT_EQ(snapshot.cropboxes.front().data.max, glm::vec3(1.0f, 2.0f, 3.0f));
}

TEST(VideoExportUtilsTest, CaptureSnapshotKeepsPointCloudActiveEllipsoidWithoutSplatParent) {
    lfs::vis::SceneManager scene_manager;
    auto& scene = scene_manager.getScene();

    const auto parent_id = scene.addPointCloud("points", make_test_point_cloud());
    ASSERT_NE(parent_id, lfs::core::NULL_NODE);
    auto ellipsoid_result = lfs::vis::cap::ensureEllipsoid(scene_manager, nullptr, parent_id);
    ASSERT_TRUE(ellipsoid_result) << ellipsoid_result.error();
    auto* ellipsoid_node = scene.getNodeById(*ellipsoid_result);
    ASSERT_NE(ellipsoid_node, nullptr);
    ASSERT_TRUE(ellipsoid_node->ellipsoid);
    ellipsoid_node->ellipsoid->enabled = true;
    ellipsoid_node->ellipsoid->inverse = true;
    ellipsoid_node->ellipsoid->radii = {2.0f, 3.0f, 4.0f};
    scene_manager.selectNode(ellipsoid_node->name);

    auto snapshot_result = lfs::vis::gui::captureVideoExportSceneSnapshot(scene_manager);
    ASSERT_TRUE(snapshot_result.has_value()) << snapshot_result.error();

    const auto& snapshot = *snapshot_result;
    ASSERT_TRUE(snapshot.point_cloud);
    ASSERT_TRUE(snapshot.active_ellipsoid.has_value());
    EXPECT_LT(snapshot.active_ellipsoid->parent_node_index, 0);
    EXPECT_TRUE(snapshot.active_ellipsoid->data.enabled);
    EXPECT_TRUE(snapshot.active_ellipsoid->data.inverse);
    EXPECT_EQ(snapshot.active_ellipsoid->data.radii, glm::vec3(2.0f, 3.0f, 4.0f));
}

// Catches video export frames that drop a crop polygon the viewport shows: on a point cloud it must reach the
// point-cloud filter, and on a splat the scene state the splat frames render with.
TEST(VideoExportUtilsTest, CropPolygonsReachPointCloudAndSplatFrames) {
    const std::vector<glm::vec2> outline{{-1.0f, -1.0f}, {1.0f, -1.0f}, {1.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 1.0f}, {-1.0f, 1.0f}};
    lfs::vis::RenderSettings settings;
    settings.desaturate_cropping = true;
    {
        lfs::vis::SceneManager scene_manager;
        auto& scene = scene_manager.getScene();
        const auto cloud = scene.addPointCloud("points", make_test_point_cloud());
        ASSERT_NE(cloud, lfs::core::NULL_NODE);
        const auto polygon = lfs::vis::cap::createCropPolygon(
            scene_manager, nullptr, cloud,
            lfs::core::CropPolygonData{.vertices = outline, .half_depth = 0.5f, .inverse = true},
            glm::translate(glm::mat4(1.0f), glm::vec3(1.0f, 2.0f, 3.0f)));
        ASSERT_TRUE(polygon) << polygon.error().user_message();
        scene_manager.selectNode(*polygon);

        const auto snapshot = lfs::vis::gui::captureVideoExportSceneSnapshot(scene_manager);
        ASSERT_TRUE(snapshot.has_value()) << snapshot.error();
        ASSERT_TRUE(snapshot->point_cloud);
        ASSERT_EQ(snapshot->crop_polygons.size(), 1u);
        EXPECT_EQ(snapshot->selected_crop_polygon_index, 0);
        EXPECT_LT(snapshot->crop_polygons.front().parent_node_index, 0);

        lfs::rendering::PointCloudFilterState filters;
        lfs::vis::gui::applyVideoExportPointCloudFilters(filters, *snapshot, settings);
        ASSERT_TRUE(filters.crop_polygon.has_value());
        EXPECT_FALSE(filters.crop_box.has_value());
        EXPECT_FALSE(filters.crop_ellipsoid.has_value());
        EXPECT_EQ(filters.crop_polygon->vertices, outline);
        EXPECT_FLOAT_EQ(filters.crop_polygon->half_depth, 0.5f);
        EXPECT_EQ(filters.crop_polygon->transform, glm::inverse(snapshot->crop_polygons.front().world_transform));
        EXPECT_TRUE(filters.crop_inverse);
        EXPECT_TRUE(filters.crop_desaturate);
    }
    {
        lfs::vis::SceneManager scene_manager;
        auto& scene = scene_manager.getScene();
        const auto splat = scene.addSplat("splat", make_test_splat({0.0f, 0.0f, 0.0f}));
        ASSERT_TRUE(lfs::vis::cap::createCropPolygon(scene_manager, nullptr, splat,
                                                     lfs::core::CropPolygonData{.vertices = outline, .half_depth = 0.5f},
                                                     glm::mat4(1.0f)));

        const auto snapshot = lfs::vis::gui::captureVideoExportSceneSnapshot(scene_manager);
        ASSERT_TRUE(snapshot.has_value()) << snapshot.error();
        ASSERT_EQ(snapshot->crop_polygons.size(), 1u);
        EXPECT_EQ(snapshot->crop_polygons.front().parent_node_index, 0);
        const auto state = lfs::vis::gui::makeVideoExportGaussianSceneState(*snapshot, std::nullopt, nullptr, 0.0f);
        ASSERT_EQ(state.crop_polygons.size(), 1u);
        EXPECT_EQ(state.crop_polygons.front().parent_node_index, 0);
        ASSERT_NE(state.crop_polygons.front().data, nullptr);
        EXPECT_EQ(state.crop_polygons.front().data->vertices, outline);
        EXPECT_TRUE(state.crop_polygons.front().data->enabled);
        EXPECT_EQ(state.crop_polygons.front().world_transform, snapshot->crop_polygons.front().world_transform);
    }
}

TEST(VideoExportUtilsTest, CaptureSnapshotSupportsMeshOnlyScenes) {
    lfs::vis::SceneManager scene_manager;
    auto& scene = scene_manager.getScene();

    scene.addMesh("mesh", make_test_mesh());
    scene.setNodeTransform("mesh", glm::translate(glm::mat4(1.0f), glm::vec3(-1.5f, 0.0f, 4.0f)));

    auto snapshot_result = lfs::vis::gui::captureVideoExportSceneSnapshot(scene_manager);
    ASSERT_TRUE(snapshot_result.has_value()) << snapshot_result.error();

    const auto& snapshot = *snapshot_result;
    EXPECT_FALSE(snapshot.combined_model);
    EXPECT_FALSE(snapshot.point_cloud);
    ASSERT_EQ(snapshot.meshes.size(), 1u);
    ASSERT_TRUE(snapshot.meshes[0].mesh);
    expect_visualizer_translation_from_data(snapshot.meshes[0].transform, {-1.5f, 0.0f, 4.0f});
}

TEST(VideoExportUtilsTest, ValidateVideoExportOptionsRejectsInvalidValues) {
    EXPECT_FALSE(lfs::vis::gui::validateVideoExportOptions({.width = 0,
                                                            .height = 1080,
                                                            .framerate = 30,
                                                            .crf = 18}));
    EXPECT_FALSE(lfs::vis::gui::validateVideoExportOptions({.width = 1920,
                                                            .height = -1,
                                                            .framerate = 30,
                                                            .crf = 18}));
    EXPECT_FALSE(lfs::vis::gui::validateVideoExportOptions({.width = 1920,
                                                            .height = 1080,
                                                            .framerate = 0,
                                                            .crf = 18}));
    EXPECT_FALSE(lfs::vis::gui::validateVideoExportOptions({.width = 1920,
                                                            .height = 1080,
                                                            .framerate = 30,
                                                            .crf = 99}));
    EXPECT_FALSE(lfs::vis::gui::validateVideoExportOptions({.width = 1919,
                                                            .height = 1080,
                                                            .framerate = 30,
                                                            .crf = 18}));
}

TEST(VideoExportUtilsTest, ValidateVideoExportOptionsAcceptsNativeResolution) {
    auto result = lfs::vis::gui::validateVideoExportOptions({.width = 32768,
                                                             .height = 17280,
                                                             .framerate = 30,
                                                             .crf = 18});

    ASSERT_TRUE(result.has_value()) << result.error();
    EXPECT_EQ(result->width, 32768);
    EXPECT_EQ(result->height, 17280);
    EXPECT_EQ(result->framerate, 30);
    EXPECT_EQ(result->crf, 18);
}

TEST(VideoEncoderValidationTest, RejectsUnsafeOptionsBeforeCodecInitialization) {
    lfs::io::video::VideoEncoder encoder;
    const std::filesystem::path unused_path = "/tmp/lfs-invalid-video-options.mp4";

    auto options = lfs::io::video::VideoExportOptions{
        .preset = lfs::io::video::VideoPreset::CUSTOM,
        .width = 3,
        .height = 2,
        .framerate = 30,
        .crf = 18,
    };
    auto result = encoder.open(unused_path, options);
    ASSERT_FALSE(result.has_value());
    EXPECT_NE(result.error().find("even"), std::string::npos);

    options.width = std::numeric_limits<int>::max() - 1;
    result = encoder.open(unused_path, options);
    ASSERT_FALSE(result.has_value());
    EXPECT_NE(result.error().find("pixel budget"), std::string::npos);
    EXPECT_FALSE(encoder.isOpen());
}

namespace {
    struct VideoOutputDirectory {
        std::filesystem::path path = std::filesystem::temp_directory_path() /
                                     ("lfs-video-extension-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        VideoOutputDirectory() { std::filesystem::create_directories(path); }
        ~VideoOutputDirectory() {
            std::error_code ec;
            std::filesystem::remove_all(path, ec);
        }
    };
} // namespace

TEST(VideoEncoderOpenTest, RejectsUnsupportedDimensionsWithoutCreatingOutput) {
    VideoOutputDirectory directory;
    const auto path = directory.path / "unsupported.mp4";
    const lfs::io::video::VideoExportOptions options{
        .preset = lfs::io::video::VideoPreset::CUSTOM,
        .width = 8192,
        .height = 4096,
        .framerate = 2,
        .crf = 20,
    };

    lfs::io::video::VideoEncoder encoder;
    const auto result = encoder.open(path, options);
    if (!result && result.error().find("h264_nvenc: encoder unavailable") != std::string::npos &&
        result.error().find("H.264 fallback: H.264 encoder not found") != std::string::npos) {
        GTEST_SKIP() << "No H.264 encoder is available in this build";
    }

    ASSERT_FALSE(result.has_value());
    EXPECT_NE(result.error().find("8192x4096"), std::string::npos);
    EXPECT_NE(result.error().find("h264_nvenc"), std::string::npos);
    EXPECT_NE(result.error().find("H.264 fallback"), std::string::npos);
    EXPECT_NE(result.error().find("codec open failed"), std::string::npos);
    EXPECT_FALSE(std::filesystem::exists(path));
}

TEST(VideoEncoderValidationTest, RejectsConflictingExtensionsWithoutTouchingDestination) {
    VideoOutputDirectory directory;
    const auto options = lfs::io::video::VideoExportOptions{
        .preset = lfs::io::video::VideoPreset::CUSTOM,
        .width = 160,
        .height = 90,
        .framerate = 2,
        .crf = 20,
    };
    for (const auto* suffix : {".mkv", ".mov", ".webm", ".avi", ".mp4.webm"}) {
        for (const bool exists : {false, true}) {
            SCOPED_TRACE(::testing::Message() << suffix << " exists=" << exists);
            const auto path = directory.path / (std::string(exists ? "existing" : "new") + suffix);
            if (exists)
                std::ofstream(path, std::ios::binary) << "original contents";
            lfs::io::video::VideoEncoder encoder;
            const auto result = encoder.open(path, options);
            EXPECT_FALSE(result.has_value());
            EXPECT_FALSE(encoder.isOpen());
            if (!result)
                EXPECT_NE(result.error().find("MP4"), std::string::npos);
            else
                EXPECT_TRUE(encoder.close());
            if (exists) {
                std::ifstream file(path, std::ios::binary);
                const std::string contents{std::istreambuf_iterator<char>(file), {}};
                EXPECT_EQ(contents, "original contents");
            } else {
                EXPECT_FALSE(std::filesystem::exists(path));
            }
        }
    }
}

TEST(VideoEncoderValidationTest, AcceptsMp4CaseVariantsAndExtensionlessNames) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
        GTEST_SKIP() << "CUDA device required to initialize the video encoder";

    VideoOutputDirectory directory;
    const auto options = lfs::io::video::VideoExportOptions{
        .preset = lfs::io::video::VideoPreset::CUSTOM,
        .width = 160,
        .height = 90,
        .framerate = 2,
        .crf = 20,
    };
    for (const auto* name : {"video.mp4", "video.MP4", "video.mP4", "video"}) {
        SCOPED_TRACE(name);
        lfs::io::video::VideoEncoder encoder;
        const auto result = encoder.open(directory.path / name, options);
        ASSERT_TRUE(result.has_value()) << result.error();
        EXPECT_TRUE(encoder.isOpen());
        EXPECT_TRUE(encoder.close());
        EXPECT_TRUE(std::filesystem::is_regular_file(directory.path / name));
    }
}

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/spherical.h>
}

namespace {
    class VideoEncoderColorTest : public ::testing::TestWithParam<bool> {};

    TEST_P(VideoEncoderColorTest, TagsLimitedRangeBt601) {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
            GTEST_SKIP() << "CUDA is required by the RGB conversion";
        // NVENC rejects a 16x16 frame; exercise the actual software fallback.
        const bool hardware = GetParam();
        if (hardware && !avcodec_find_encoder_by_name("h264_nvenc"))
            GTEST_SKIP() << "NVENC is not compiled in";
        const int width = hardware ? 160 : 16;
        const int height = hardware ? 96 : 16;
        VideoOutputDirectory directory;
        const auto path = directory.path / "color.mp4";
        lfs::io::video::VideoEncoder encoder;
        auto opened = encoder.open(path, {.preset = lfs::io::video::VideoPreset::CUSTOM,
                                          .width = width,
                                          .height = height,
                                          .framerate = 4});
        ASSERT_TRUE(opened) << opened.error();
        std::vector<float> pixels(width * height * 3, 0.0f);
        for (size_t i = 0; i < pixels.size(); i += 3)
            pixels[i] = 1.0f;
        auto frame = Tensor::from_vector(pixels, {size_t(height), size_t(width), size_t{3}}, Device::CUDA);
        for (int i = 0; i < 6; ++i) {
            const auto written = encoder.writeFrameGpu(frame.ptr<float>(), width, height);
            ASSERT_TRUE(written) << written.error();
        }
        const auto closed = encoder.close();
        ASSERT_TRUE(closed) << closed.error();
        AVFormatContext* raw_format = nullptr;
        ASSERT_EQ(avformat_open_input(&raw_format, lfs::core::path_to_utf8(path).c_str(), nullptr, nullptr), 0);
        const auto close_format = [](AVFormatContext* format) { avformat_close_input(&format); };
        std::unique_ptr<AVFormatContext, decltype(close_format)> format(raw_format, close_format);
        ASSERT_GE(avformat_find_stream_info(format.get(), nullptr), 0);
        const int stream = av_find_best_stream(format.get(), AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
        ASSERT_GE(stream, 0);
        const auto* params = format->streams[stream]->codecpar;
        EXPECT_EQ(params->color_space, AVCOL_SPC_SMPTE170M);
        EXPECT_EQ(params->color_range, AVCOL_RANGE_MPEG);
        EXPECT_EQ(params->color_primaries, AVCOL_PRI_UNSPECIFIED);
        EXPECT_EQ(params->color_trc, AVCOL_TRC_UNSPECIFIED);
    }

    INSTANTIATE_TEST_SUITE_P(Backend, VideoEncoderColorTest, ::testing::Bool(),
                             [](const auto& info) { return info.param ? "Hardware" : "Software"; });
} // namespace

namespace {
    struct EncodedVideo {
        std::optional<AVSphericalProjection> projection;
        std::vector<std::string> packets;
    };

    std::optional<EncodedVideo> inspectEncodedVideo(const std::filesystem::path& path) {
        AVFormatContext* raw_format = nullptr;
        if (avformat_open_input(&raw_format, lfs::core::path_to_utf8(path).c_str(), nullptr, nullptr) < 0)
            return std::nullopt;
        const auto close_format = [](AVFormatContext* format) { avformat_close_input(&format); };
        std::unique_ptr<AVFormatContext, decltype(close_format)> format(raw_format, close_format);
        if (avformat_find_stream_info(format.get(), nullptr) < 0)
            return std::nullopt;
        const int stream = av_find_best_stream(format.get(), AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
        if (stream < 0)
            return std::nullopt;

        EncodedVideo result;
        const auto* params = format->streams[stream]->codecpar;
        for (int i = 0; i < params->nb_coded_side_data; ++i) {
            const auto& side_data = params->coded_side_data[i];
            if (side_data.type == AV_PKT_DATA_SPHERICAL && side_data.size >= sizeof(AVSphericalMapping))
                result.projection = reinterpret_cast<const AVSphericalMapping*>(side_data.data)->projection;
        }
        AVPacket* packet = av_packet_alloc();
        if (!packet)
            return std::nullopt;
        int read_result = 0;
        while ((read_result = av_read_frame(format.get(), packet)) >= 0) {
            if (packet->stream_index == stream)
                result.packets.emplace_back(reinterpret_cast<const char*>(packet->data),
                                            static_cast<size_t>(packet->size));
            av_packet_unref(packet);
        }
        av_packet_free(&packet);
        return read_result == AVERROR_EOF ? std::optional<EncodedVideo>(std::move(result)) : std::nullopt;
    }
} // namespace

TEST(VideoEncoderSphericalMetadataTest, TagsEquirectangularOnlyAndPreservesEncodedFrames) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
        GTEST_SKIP() << "CUDA device required to encode video frames";

    constexpr int width = 160;
    constexpr int height = 96;
    VideoOutputDirectory directory;
    const auto frame = Tensor::zeros({size_t(height), size_t(width), size_t{3}}, Device::CUDA,
                                     lfs::core::DataType::Float32);
    const auto encode = [&](const std::filesystem::path& path, const bool spherical) {
        lfs::io::video::VideoEncoder encoder;
        const auto opened = encoder.open(path, {.preset = lfs::io::video::VideoPreset::CUSTOM, .width = width, .height = height, .framerate = 4, .crf = 20, .spherical_video = spherical});
        if (!opened) {
            ADD_FAILURE() << opened.error();
            return false;
        }
        for (int i = 0; i < 4; ++i) {
            const auto written = encoder.writeFrameGpu(frame.ptr<float>(), width, height);
            if (!written) {
                ADD_FAILURE() << written.error();
                return false;
            }
        }
        const auto closed = encoder.close();
        if (!closed) {
            ADD_FAILURE() << closed.error();
            return false;
        }
        return true;
    };
    ASSERT_TRUE(encode(directory.path / "spherical.mp4", true));
    ASSERT_TRUE(encode(directory.path / "perspective.mp4", false));
    const auto spherical = inspectEncodedVideo(directory.path / "spherical.mp4");
    const auto perspective = inspectEncodedVideo(directory.path / "perspective.mp4");
    ASSERT_TRUE(spherical);
    ASSERT_TRUE(perspective);
    ASSERT_TRUE(spherical->projection);
    EXPECT_EQ(*spherical->projection, AV_SPHERICAL_EQUIRECTANGULAR);
    EXPECT_FALSE(perspective->projection);
    ASSERT_FALSE(spherical->packets.empty());
    EXPECT_EQ(spherical->packets, perspective->packets);
}

namespace {
    // Keep the regression runnable against the old API to prove the image failure.
    template <typename Manager>
    std::shared_ptr<Tensor> renderExportPreview(Manager& manager, const lfs::core::SplatData& model,
                                                int format, int width, int height, bool panorama) {
        const glm::mat3 rotation(1.0f);
        const glm::vec3 position(0.0f);
        if (format == 0) {
            if constexpr (requires { manager.renderPreviewImage(model, {}, rotation, position, 35.0f, width, height,
                                                                std::nullopt, std::nullopt, std::nullopt, panorama); })
                return manager.renderPreviewImage(model, {}, rotation, position, 35.0f, width, height,
                                                  std::nullopt, std::nullopt, std::nullopt, panorama);
            else
                return manager.renderPreviewImage(model, {}, rotation, position, 35.0f, width, height);
        }
        if (format == 1) {
            if constexpr (requires { manager.renderPreviewImageRgb8(model, {}, rotation, position, 35.0f, width, height,
                                                                    std::nullopt, std::nullopt, std::nullopt, panorama); })
                return manager.renderPreviewImageRgb8(model, {}, rotation, position, 35.0f, width, height,
                                                      std::nullopt, std::nullopt, std::nullopt, panorama);
            else
                return manager.renderPreviewImageRgb8(model, {}, rotation, position, 35.0f, width, height);
        }
        if constexpr (requires { manager.renderPreviewImageRgba8(model, {}, rotation, position, 35.0f, width, height,
                                                                 std::nullopt, std::nullopt, panorama); })
            return manager.renderPreviewImageRgba8(model, {}, rotation, position, 35.0f, width, height,
                                                   std::nullopt, std::nullopt, panorama);
        else
            return manager.renderPreviewImageRgba8(model, {}, rotation, position, 35.0f, width, height);
    }

    TEST_P(VideoExportBandTest, PreviewProjectionMatchesRenderer) {
        const bool panorama = std::get<0>(GetParam());
        const bool tiled = std::get<1>(GetParam());
        const int width = tiled ? 6144 : 1024;
        const int height = width / 2;
        lfs::vis::RenderingManager manager;
        auto settings = manager.getSettings();
        settings.raster_backend = lfs::rendering::GaussianRasterBackend::ThreeDgut;
        settings.gut = true;
        settings.equirectangular = panorama;
        settings.sh_degree = 0;
        settings.mip_filter = false;
        settings.antialiasing = false;
        settings.background_color = glm::vec3(0.0f);
        manager.updateSettings(settings);
        const Viewport viewport(64, 64);
        // Establish the same context that the viewer passes to the export manager.
        manager.renderVulkanFrame({.viewport = viewport, .settings = settings, .vulkan_context = &context_});
        lfs::rendering::ViewportRenderRequest request;
        request.deterministic_export = true;
        request.frame_view.size = {width, height};
        request.frame_view.focal_length_mm = 35.0f;
        request.sh_degree = 0;
        request.raster_backend = settings.raster_backend;
        request.gut = true;
        request.equirectangular = panorama;
        const auto slot = lfs::vis::VksplatViewportRenderer::OutputSlot::Preview;
        for (int format = 0; format < 3; ++format) {
            SCOPED_TRACE(::testing::Message() << "format=" << format << " tiled=" << tiled);
            request.transparent_background = format == 2;
            auto rendered = renderer_->render(context_, *model_, request, false, slot, false);
            ASSERT_TRUE(rendered) << rendered.error();
            auto reference = format == 2 ? renderer_->readOutputImageRgba8(context_, slot)
                                         : renderer_->readOutputImageRgb8(context_, slot);
            ASSERT_TRUE(reference) << reference.error();
            auto actual = renderExportPreview(manager, *model_, format, width, height, panorama);
            ASSERT_TRUE(actual);
            ASSERT_EQ(actual->numel(), (*reference)->numel());
            int max_diff = 0;
            size_t changed = 0;
            const auto* expected = (*reference)->ptr<uint8_t>();
            for (size_t i = 0; i < actual->numel(); ++i) {
                const int value = format == 0 ? int(std::clamp(actual->ptr<float>()[i], 0.0f, 1.0f) * 255.0f + 0.5f)
                                              : actual->ptr<uint8_t>()[i];
                const int diff = std::abs(value - int(expected[i]));
                max_diff = std::max(max_diff, diff);
                changed += diff > 1;
            }
            EXPECT_LE(max_diff, 1) << "channels differing by more than one=" << changed;
        }
    }
} // namespace
