/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/mesh_data.hpp"
#include "core/point_cloud.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "gui/video_export_utils.hpp"
#include "io/loader.hpp"
#include "io/video/video_encoder.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "rendering/cuda_vulkan_interop.hpp"
#include "rendering/vksplat_viewport_renderer.hpp"
#include "rendering/vulkan_external_tensor.hpp"
#include "scene/scene_manager.hpp"
#include "visualizer/gui_capabilities.hpp"
#include "window/vulkan_context.hpp"
#include <SDL3/SDL.h>
#include <algorithm>

#include <cuda_runtime.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using lfs::core::Device;
using lfs::core::Tensor;

namespace {

    std::unique_ptr<lfs::core::SplatData> make_test_splat(const std::vector<float>& xyz) {
        const size_t count = xyz.size() / 3;
        auto means = Tensor::from_vector(xyz, {count, size_t{3}}, Device::CUDA).to(lfs::core::DataType::Float32);
        auto sh0 = Tensor::zeros({count, size_t{1}, size_t{3}}, Device::CUDA, lfs::core::DataType::Float32);
        auto shN = Tensor::zeros({count, size_t{3}, size_t{3}}, Device::CUDA, lfs::core::DataType::Float32);
        auto scaling = Tensor::zeros({count, size_t{3}}, Device::CUDA, lfs::core::DataType::Float32);

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
    request.frame_view.size = {64, 64};
    request.sh_degree = 0;
    request.scene.model_transforms = &snapshot->model_transforms;
    auto explicit_indices = std::make_shared<Tensor>(
        Tensor::zeros({size_t{1}}, Device::CUDA, lfs::core::DataType::Int32));
    const auto render = [&](const std::vector<bool>& mask, const bool indexed) {
        request.scene.node_visibility_mask = mask;
        request.scene.transform_indices = indexed ? explicit_indices : nullptr;
        auto rendered = renderer.render(context, *snapshot->combined_model, request, false, slot, false, true);
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
