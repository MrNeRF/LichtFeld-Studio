/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/mesh_data.hpp"
#include "core/point_cloud.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "gui/video_export_utils.hpp"
#include "io/exporter.hpp"
#include "io/video/video_encoder.hpp"
#include "lfs/training/sh_value_storage.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "scene/scene_manager.hpp"
#include "visualizer/gui_capabilities.hpp"

#include <chrono>
#include <filesystem>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using lfs::core::Device;
using lfs::core::Tensor;

namespace {

    int optional_video_adapter_factory_calls = 0;

    class ReadyVideoOptionalAdapter final : public lfs::vis::SceneUpscalerAdapter {
    public:
        lfs::vis::SceneUpscalerAvailability probe(
            const lfs::vis::SceneUpscalerProbeContext&) const noexcept override {
            return {.reason = lfs::vis::SceneUpscalerAvailabilityReason::Ready};
        }
    };

    lfs::vis::SceneUpscalerAdapterFactoryResult makeReadyVideoOptionalAdapter() noexcept {
        ++optional_video_adapter_factory_calls;
        return std::unique_ptr<lfs::vis::SceneUpscalerAdapter>(
            std::make_unique<ReadyVideoOptionalAdapter>());
    }

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

    std::unique_ptr<lfs::core::SplatData> make_test_fp32_splat_256() {
        constexpr size_t count = 256;
        std::vector<float> rotations(count * 4, 0.0f);
        for (size_t i = 0; i < count; ++i)
            rotations[i * 4] = 1.0f;
        return std::make_unique<lfs::core::SplatData>(
            3,
            Tensor::zeros({count, size_t{3}}, Device::CUDA, lfs::core::DataType::Float32),
            Tensor::zeros({count, size_t{1}, size_t{3}}, Device::CUDA, lfs::core::DataType::Float32),
            Tensor::zeros({count, size_t{15}, size_t{3}}, Device::CUDA, lfs::core::DataType::Float32),
            Tensor::zeros({count, size_t{3}}, Device::CUDA, lfs::core::DataType::Float32),
            Tensor::from_vector(
                rotations, lfs::core::TensorShape({count, size_t{4}}), Device::CUDA),
            Tensor::zeros({count, size_t{1}}, Device::CUDA, lfs::core::DataType::Float32),
            1.0f,
            lfs::core::SplatData::ShNLayout::Canonical);
    }

    std::unique_ptr<lfs::core::SplatData> make_test_q16_splat() {
        auto model = make_test_fp32_splat_256();
        if (!lfs::training::sh_value::apply_shN_value_quant(*model))
            throw std::runtime_error("Failed to create q16 test fixture");
        return model;
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

TEST(VideoExportUtilsTest, Float16SnapshotBuildsAValidDedicatedQuantizedRepresentation) {
    lfs::vis::SceneManager scene_manager;
    auto resident = make_test_q16_splat();
    const int expected_active_sh_degree = resident->get_active_sh_degree();
    scene_manager.getScene().addSplat("q16", std::move(resident));

    const auto snapshot_result =
        lfs::vis::gui::captureVideoExportSceneSnapshot(
            scene_manager, lfs::io::video::VideoSplatPrecision::Float16);
    ASSERT_TRUE(snapshot_result.has_value()) << snapshot_result.error();
    ASSERT_TRUE(snapshot_result->combined_model);
    EXPECT_TRUE(snapshot_result->combined_model->shN_value_quantized());
    EXPECT_TRUE(snapshot_result->combined_model->shN_value_bounds().is_valid());
    EXPECT_EQ(snapshot_result->combined_model->get_active_sh_degree(),
              expected_active_sh_degree);
}

TEST(VideoExportUtilsTest, Float16SnapshotQuantizesAFloat32ResidentModel) {
    lfs::vis::SceneManager scene_manager;
    scene_manager.getScene().addSplat("fp32", make_test_fp32_splat_256());

    const auto snapshot_result = lfs::vis::gui::captureVideoExportSceneSnapshot(
        scene_manager, lfs::io::video::VideoSplatPrecision::Float16);
    ASSERT_TRUE(snapshot_result.has_value()) << snapshot_result.error();
    ASSERT_TRUE(snapshot_result->combined_model);
    EXPECT_TRUE(snapshot_result->combined_model->shN_value_quantized());
    EXPECT_EQ(snapshot_result->combined_model->shN_raw().dtype(),
              lfs::core::DataType::Float16);
}

TEST(VideoExportUtilsTest, Float32SnapshotClonesAFloat32ResidentModelWithoutSourceReload) {
    lfs::vis::SceneManager scene_manager;
    scene_manager.getScene().addSplat("fp32", make_test_fp32_splat_256());

    const auto snapshot_result = lfs::vis::gui::captureVideoExportSceneSnapshot(
        scene_manager, lfs::io::video::VideoSplatPrecision::Float32);
    ASSERT_TRUE(snapshot_result.has_value()) << snapshot_result.error();
    ASSERT_TRUE(snapshot_result->combined_model);
    EXPECT_EQ(snapshot_result->combined_model->shN_raw().dtype(),
              lfs::core::DataType::Float32);
    EXPECT_FALSE(snapshot_result->combined_model->shN_value_quantized());
}

TEST(VideoExportUtilsTest, Float32SnapshotRejectsGeneratedReducedResidentData) {
    lfs::vis::SceneManager scene_manager;
    scene_manager.getScene().addSplat("q16", make_test_q16_splat());

    const auto snapshot_result = lfs::vis::gui::captureVideoExportSceneSnapshot(
        scene_manager, lfs::io::video::VideoSplatPrecision::Float32);
    ASSERT_FALSE(snapshot_result.has_value());
    EXPECT_NE(snapshot_result.error().find("generated node"), std::string::npos);
}

TEST(VideoExportUtilsTest, Float32SnapshotReloadsFileInsteadOfResidentQuantizedStorage) {
    const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto source_path = std::filesystem::temp_directory_path() /
                             ("lfs_video_source_fp32_" + std::to_string(unique) + ".ply");
    struct RemoveSourceFile {
        std::filesystem::path path;
        ~RemoveSourceFile() {
            std::error_code ec;
            std::filesystem::remove(path, ec);
        }
    } cleanup{source_path};

    auto resident = make_test_fp32_splat_256();
    const auto save_result = lfs::io::save_ply(
        *resident, {.output_path = source_path, .binary = true});
    ASSERT_TRUE(save_result.has_value()) << save_result.error().format();
    ASSERT_TRUE(lfs::training::sh_value::apply_shN_value_quant(*resident));

    lfs::vis::SceneManager scene_manager;
    const auto node_id = scene_manager.getScene().addSplat("q16", std::move(resident));
    ASSERT_NE(node_id, lfs::core::NULL_NODE);
    scene_manager.setPlyPath(node_id, source_path);

    const auto snapshot_result = lfs::vis::gui::captureVideoExportSceneSnapshot(
        scene_manager, lfs::io::video::VideoSplatPrecision::Float32);
    ASSERT_TRUE(snapshot_result.has_value()) << snapshot_result.error();
    ASSERT_TRUE(snapshot_result->combined_model);
    EXPECT_EQ(snapshot_result->combined_model->shN_raw().dtype(), lfs::core::DataType::Float32);
    EXPECT_FALSE(snapshot_result->combined_model->shN_value_quantized());
    EXPECT_FALSE(snapshot_result->combined_model->shN_ieee_f16());
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

TEST(VideoExportUtilsTest, NativeRenderPlanPreservesExactOutput) {
    const lfs::io::video::VideoExportOptions options{
        .width = 1920,
        .height = 1080,
        .framerate = 30,
        .crf = 18,
    };

    const auto plan = lfs::vis::gui::makeVideoExportRenderPlan(options);
    ASSERT_TRUE(plan) << plan.error();
    EXPECT_EQ(plan->input_width, 1920);
    EXPECT_EQ(plan->input_height, 1080);
    EXPECT_EQ(plan->output_width, 1920);
    EXPECT_EQ(plan->output_height, 1080);
    EXPECT_FALSE(plan->requires_upscale);
    EXPECT_EQ(plan->backend, "native");
}

TEST(VideoExportUtilsTest, UpscaledRenderPlanUsesEvenInternalExtent) {
    lfs::io::video::VideoExportOptions options{
        .preset = lfs::io::video::VideoPreset::CUSTOM,
        .width = 1920,
        .height = 1080,
        .framerate = 30,
        .crf = 18,
    };
    options.upscaler = {
        .backend = "spatial",
        .input_scale = 0.67f,
        .quality = 1,
        .fallback = lfs::io::video::VideoUpscalerFallback::Native,
    };

    const auto plan = lfs::vis::gui::makeVideoExportRenderPlan(options);
    ASSERT_TRUE(plan) << plan.error();
    EXPECT_EQ(plan->input_width % 2, 0);
    EXPECT_EQ(plan->input_height % 2, 0);
    EXPECT_LT(plan->input_width, plan->output_width);
    EXPECT_LT(plan->input_height, plan->output_height);
    EXPECT_TRUE(plan->requires_upscale);
}

TEST(VideoExportUtilsTest, RejectsInvalidOfflineUpscalerContract) {
    lfs::io::video::VideoExportOptions options{};
    options.upscaler.input_scale = 0.5f;
    EXPECT_FALSE(lfs::vis::gui::makeVideoExportRenderPlan(options));

    options.upscaler.backend = "temporal";
    options.upscaler.input_scale = 0.1f;
    EXPECT_FALSE(lfs::vis::gui::makeVideoExportRenderPlan(options));

    options.upscaler.input_scale = 0.5f;
    options.upscaler.quality = 4;
    EXPECT_FALSE(lfs::vis::gui::makeVideoExportRenderPlan(options));

    options.upscaler.quality = 1;
    options.upscaler.backend = "temporal";
    EXPECT_TRUE(lfs::vis::gui::makeVideoExportRenderPlan(options));
}

TEST(VideoExportUtilsTest, TemporalSampleTimesCoverOneCenteredFrameInterval) {
    lfs::io::video::VideoExportOptions options{};
    options.upscaler.backend = "temporal";
    options.upscaler.input_scale = 0.5f;
    options.upscaler.quality = 1;

    const auto times = lfs::vis::gui::videoExportSampleTimes(1.0f, 0.1f, 0.0f, 2.0f, options);
    ASSERT_EQ(times.size(), 4u);
    EXPECT_FLOAT_EQ(times[0], 0.9625f);
    EXPECT_FLOAT_EQ(times[1], 0.9875f);
    EXPECT_FLOAT_EQ(times[2], 1.0125f);
    EXPECT_FLOAT_EQ(times[3], 1.0375f);
}

TEST(VideoExportUtilsTest, NativeAndSpatialUseOneExactSample) {
    lfs::io::video::VideoExportOptions options{};
    for (const std::string backend : {"native", "spatial"}) {
        options.upscaler.backend = backend;
        const auto times = lfs::vis::gui::videoExportSampleTimes(1.0f, 0.1f, 0.0f, 2.0f, options);
        ASSERT_EQ(times.size(), 1u);
        EXPECT_FLOAT_EQ(times.front(), 1.0f);
    }
}

TEST(VideoExportUtilsTest, OptionalVendorContractIsResolvedWithoutCreatingItsAdapter) {
    lfs::vis::OptionalSceneUpscalerRegistry registry;
    ASSERT_TRUE(registry.registerAdapter(
        {.id = "nvidia-dlss",
         .label_key = "preferences.scene_upscaler_nvidia_dlss",
         .requirements = {
             .depth = true,
             .motion_vectors = true,
             .jitter = true,
             .history = true,
             .exposure = true,
         }},
        &makeReadyVideoOptionalAdapter));
    ASSERT_TRUE(registry.registerAdapter(
        {.id = "amd-fsr3",
         .label_key = "preferences.scene_upscaler_amd_fsr3",
         .requirements = {
             .depth = true,
             .motion_vectors = true,
             .jitter = true,
             .history = true,
         }},
        &makeReadyVideoOptionalAdapter));
    optional_video_adapter_factory_calls = 0;

    const auto contract = lfs::vis::gui::videoExportUpscalerContract("nvidia-dlss", registry);
    ASSERT_TRUE(contract);
    EXPECT_EQ(contract->execution,
              lfs::vis::gui::VideoExportUpscalerExecution::VulkanAdapter);
    EXPECT_TRUE(contract->optional);
    EXPECT_TRUE(contract->lazy_adapter);
    EXPECT_TRUE(contract->requirements.depth);
    EXPECT_TRUE(contract->requirements.motion_vectors);
    EXPECT_TRUE(contract->requirements.jitter);
    EXPECT_TRUE(contract->requirements.history);
    EXPECT_EQ(optional_video_adapter_factory_calls, 0);

    const auto fsr_contract = lfs::vis::gui::videoExportUpscalerContract("amd-fsr3", registry);
    ASSERT_TRUE(fsr_contract);
    EXPECT_EQ(fsr_contract->execution,
              lfs::vis::gui::VideoExportUpscalerExecution::VulkanAdapter);
    EXPECT_TRUE(fsr_contract->lazy_adapter);
    EXPECT_TRUE(fsr_contract->requirements.temporal());
    EXPECT_EQ(optional_video_adapter_factory_calls, 0);

    EXPECT_EQ(lfs::vis::gui::validateVideoExportUpscalerResources(
                  *contract, {.vulkan_color = true, .depth = true}),
              lfs::vis::gui::VideoExportUpscalerResourceIssue::MotionVectors);
    EXPECT_EQ(lfs::vis::gui::validateVideoExportUpscalerResources(
                  *contract,
                  {.vulkan_color = true,
                   .depth = true,
                   .motion_vectors = true,
                   .jitter = true,
                   .history = true,
                   .exposure = true}),
              lfs::vis::gui::VideoExportUpscalerResourceIssue::None);
    EXPECT_EQ(optional_video_adapter_factory_calls, 0);
}

TEST(VideoExportUtilsTest, UnknownOptionalBackendIsRejectedBeforeExport) {
    lfs::vis::OptionalSceneUpscalerRegistry registry;
    const auto contract = lfs::vis::gui::videoExportUpscalerContract("unknown-vendor", registry);
    EXPECT_FALSE(contract);
}

TEST(VideoExportUtilsTest, VulkanFrameResourcesRequireValidImagesAndHistory) {
    const glm::ivec2 extent{960, 540};
    const auto resource = [extent](const std::uintptr_t base, const VkFormat format) {
        return lfs::vis::VulkanSceneUpscalerResource{
            .image = reinterpret_cast<VkImage>(base),
            .view = reinterpret_cast<VkImageView>(base + 1),
            .format = format,
            .layout = VK_IMAGE_LAYOUT_GENERAL,
            .valid_extent = extent,
            .allocation_extent = extent,
        };
    };
    const lfs::vis::gui::VideoExportVulkanFrameInputs inputs{
        .color = resource(0x1000, VK_FORMAT_R8G8B8A8_UNORM),
        .depth = resource(0x2000, VK_FORMAT_R32_SFLOAT),
        .motion = resource(0x3000, VK_FORMAT_R16G16_SFLOAT),
        .output_extent = {1920, 1080},
        .jitter_pixels = {0.25f, -0.25f},
        .previous_jitter_pixels = {-0.25f, 0.25f},
        .history_valid = true,
        .exposure = 1.0f,
    };

    const auto resources = lfs::vis::gui::videoExportUpscalerResources(inputs);
    EXPECT_TRUE(resources.vulkan_color);
    EXPECT_TRUE(resources.depth);
    EXPECT_TRUE(resources.motion_vectors);
    EXPECT_TRUE(resources.jitter);
    EXPECT_TRUE(resources.history);
    EXPECT_TRUE(resources.exposure);
    EXPECT_FALSE(resources.reactive_mask);
}

TEST(VideoExportUtilsTest, KeepsReconstructionWhenTheSceneSupportsIt) {
    lfs::io::video::VideoExportOptions options{};
    options.upscaler.backend = "spatial";
    options.upscaler.input_scale = 0.5f;

    const auto plan = lfs::vis::gui::resolveVideoExportRenderPlan(options, true);
    ASSERT_TRUE(plan) << plan.error();
    EXPECT_EQ(plan->backend, "spatial");
    EXPECT_TRUE(plan->requires_upscale);
}

TEST(VideoExportUtilsTest, AbortsUnsupportedCompositeReconstructionByDefault) {
    lfs::io::video::VideoExportOptions options{};
    options.upscaler.backend = "spatial";
    options.upscaler.input_scale = 0.5f;

    const auto plan = lfs::vis::gui::resolveVideoExportRenderPlan(options, false);
    ASSERT_FALSE(plan);
    EXPECT_NE(plan.error().find("mesh or environment"), std::string::npos);
}

TEST(VideoExportUtilsTest, ExplicitFallbackResolvesUnsupportedCompositeToNative) {
    lfs::io::video::VideoExportOptions options{};
    options.upscaler.backend = "spatial";
    options.upscaler.input_scale = 0.5f;
    options.upscaler.fallback = lfs::io::video::VideoUpscalerFallback::Native;

    const auto plan = lfs::vis::gui::resolveVideoExportRenderPlan(options, false);
    ASSERT_TRUE(plan) << plan.error();
    EXPECT_EQ(plan->backend, "native");
    EXPECT_FALSE(plan->requires_upscale);
    EXPECT_EQ(plan->input_width, options.width);
    EXPECT_EQ(plan->input_height, options.height);
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
