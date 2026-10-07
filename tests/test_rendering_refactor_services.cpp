/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/editor_context.hpp"
#include "core/event_bridge/event_bridge.hpp"
#include "core/event_bus.hpp"
#include "core/events.hpp"
#include "core/image_io.hpp"
#include "core/image_loader.hpp"
#include "core/memory_pressure.hpp"
#include "core/point_cloud.hpp"
#include "core/scene.hpp"
#include "core/services.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "input/key_codes.hpp"
#include "io/cache_image_loader.hpp"
#include "operation/undo_history.hpp"
#include "operator/operator_registry.hpp"
#include "operator/ops/depth_window_ops.hpp"
#include "operator/ops/selection_ops.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "rendering/render_constants.hpp"
#include "rendering/rendering.hpp"
#include "rendering/vksplat_viewport_renderer.hpp"
#include "screen/screen_service.hpp"
#include "selection/selection_service.hpp"
#include "tools/selection_tool.hpp"
#include "visualizer/gui_capabilities.hpp"
#include "visualizer/rendering/gt_comparison_cache_utils.hpp"
#include "visualizer/rendering/gt_comparison_geometry.hpp"
#include "visualizer/rendering/render_pass.hpp"
#include "visualizer/rendering/rendering_manager.hpp"
#include "visualizer/rendering/rendering_manager_split_view.hpp"
#include "visualizer/rendering/split_capture_compositor.hpp"
#include "visualizer/rendering/split_view_service.hpp"
#include "visualizer/rendering/stale_frame_guard.hpp"
#include "visualizer/rendering/viewport_artifact_service.hpp"
#include "visualizer/rendering/viewport_frame_lifecycle_service.hpp"
#include "visualizer/rendering/viewport_interop_service.hpp"
#include "visualizer/rendering/viewport_request_builder.hpp"
#include "visualizer/scene/scene_manager.hpp"
#include "visualizer/scene_coordinate_utils.hpp"
#include "visualizer_impl.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace lfs::vis {

    TEST(StaleFrameGuardTest, DeferralsBelowBoundKeepCachedThenEscalateOnce) {
        StaleFrameGuard guard;
        EXPECT_TRUE(guard.canUseCachedFrame());
        EXPECT_FALSE(guard.takeRecoveryRequest());
        for (std::uint32_t attempt = 1; attempt < StaleFrameGuard::kMaxCachedDeferrals; ++attempt) {
            EXPECT_FALSE(guard.onDeferral()) << attempt;
            EXPECT_TRUE(guard.canUseCachedFrame()) << attempt;
            EXPECT_FALSE(guard.takeRecoveryRequest()) << attempt;
        }
        EXPECT_TRUE(guard.onDeferral());
        EXPECT_FALSE(guard.canUseCachedFrame());
        EXPECT_TRUE(guard.takeRecoveryRequest());
        EXPECT_FALSE(guard.takeRecoveryRequest());
        // Reimport/reset is not a successful publication. Failure after reset
        // must not restore the stale image or trigger another reset/WARN.
        for (int attempt = 0; attempt < 100; ++attempt) {
            EXPECT_FALSE(guard.onDeferral());
            EXPECT_FALSE(guard.canUseCachedFrame());
            EXPECT_FALSE(guard.takeRecoveryRequest());
        }
    }

    TEST(StaleFrameGuardTest, SuccessfulPublicationResetsTheWholeEpisode) {
        StaleFrameGuard guard;
        for (int episode = 0; episode < 2; ++episode) {
            for (std::uint32_t attempt = 1; attempt < StaleFrameGuard::kMaxCachedDeferrals; ++attempt) {
                EXPECT_FALSE(guard.onDeferral());
            }
            EXPECT_TRUE(guard.onDeferral());
            guard.onSuccess();
            EXPECT_TRUE(guard.canUseCachedFrame());
            EXPECT_FALSE(guard.takeRecoveryRequest());
        }
        // Success also clears a partially consumed budget.
        EXPECT_FALSE(guard.onDeferral());
        guard.onSuccess();
        for (std::uint32_t attempt = 1; attempt < StaleFrameGuard::kMaxCachedDeferrals; ++attempt) {
            EXPECT_FALSE(guard.onDeferral());
        }
        EXPECT_TRUE(guard.onDeferral());
    }

    namespace {
        std::unique_ptr<lfs::core::SplatData> makeTestSplat(const float x, const int sh_degree = 0) {
            using lfs::core::DataType;
            using lfs::core::Device;
            using lfs::core::Tensor;

            return std::make_unique<lfs::core::SplatData>(
                sh_degree,
                Tensor::from_vector({x, 0.0f, 2.0f}, {size_t{1}, size_t{3}}, Device::CPU),
                Tensor::from_vector({1.0f, 1.0f, 1.0f}, {size_t{1}, size_t{1}, size_t{3}}, Device::CPU),
                Tensor::zeros({size_t{1}, static_cast<size_t>((sh_degree + 1) * (sh_degree + 1) - 1), size_t{3}}, Device::CPU, DataType::Float32),
                Tensor::from_vector({0.0f, 0.0f, 0.0f}, {size_t{1}, size_t{3}}, Device::CPU),
                Tensor::from_vector({1.0f, 0.0f, 0.0f, 0.0f}, {size_t{1}, size_t{4}}, Device::CPU),
                Tensor::from_vector({8.0f}, {size_t{1}, size_t{1}}, Device::CPU),
                1.0f);
        }

        std::unique_ptr<lfs::core::SplatData> makeTwoPointTestSplat(const float x0, const float x1) {
            using lfs::core::DataType;
            using lfs::core::Device;
            using lfs::core::Tensor;

            return std::make_unique<lfs::core::SplatData>(
                0,
                Tensor::from_vector({x0, 0.0f, 2.0f,
                                     x1, 0.0f, 2.0f},
                                    {size_t{2}, size_t{3}}, Device::CPU),
                Tensor::from_vector({1.0f, 1.0f, 1.0f,
                                     1.0f, 1.0f, 1.0f},
                                    {size_t{2}, size_t{1}, size_t{3}}, Device::CPU),
                Tensor::zeros({size_t{2}, size_t{0}, size_t{3}}, Device::CPU, DataType::Float32),
                Tensor::from_vector({0.0f, 0.0f, 0.0f,
                                     0.0f, 0.0f, 0.0f},
                                    {size_t{2}, size_t{3}}, Device::CPU),
                Tensor::from_vector({1.0f, 0.0f, 0.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 0.0f},
                                    {size_t{2}, size_t{4}}, Device::CPU),
                Tensor::from_vector({8.0f,
                                     8.0f},
                                    {size_t{2}, size_t{1}}, Device::CPU),
                1.0f);
        }
        std::shared_ptr<lfs::core::PointCloud> makeTestPointCloud() {
            using lfs::core::Device;
            using lfs::core::Tensor;

            auto means = Tensor::from_vector(
                {0.0f, 0.0f, 0.0f,
                 1.0f, 0.0f, 0.0f},
                {size_t{2}, size_t{3}},
                Device::CPU);
            auto colors = Tensor::from_vector(
                {1.0f, 0.0f, 0.0f,
                 0.0f, 1.0f, 0.0f},
                {size_t{2}, size_t{3}},
                Device::CPU);
            return std::make_shared<lfs::core::PointCloud>(std::move(means), std::move(colors));
        }

        void expectVisualizerTranslationFromData(const glm::mat4& transform, const glm::vec3& data_translation) {
            const glm::vec3 expected =
                lfs::rendering::visualizerWorldPointFromDataWorld(data_translation);
            EXPECT_FLOAT_EQ(transform[3][0], expected.x);
            EXPECT_FLOAT_EQ(transform[3][1], expected.y);
            EXPECT_FLOAT_EQ(transform[3][2], expected.z);
        }

        void expectMat3Near(const glm::mat3& actual, const glm::mat3& expected, const float epsilon = 1e-5f) {
            for (int col = 0; col < 3; ++col) {
                for (int row = 0; row < 3; ++row) {
                    EXPECT_NEAR(actual[col][row], expected[col][row], epsilon);
                }
            }
        }

        void waitUntilResizeSettleReady(ViewportFrameLifecycleService& service) {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
            while (!service.resizeSettleReady() && std::chrono::steady_clock::now() < deadline) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            ASSERT_TRUE(service.resizeSettleReady());
        }

        void ensureCameraImageLoader() {
            static bool initialized = false;
            if (initialized) {
                return;
            }

            lfs::io::CacheLoader::getInstance(false);
            lfs::core::set_image_loader([](const lfs::core::ImageLoadParams& p) {
                return lfs::io::CacheLoader::getInstance().load_cached_image(
                    p.path,
                    {.resize_factor = p.resize_factor,
                     .max_width = p.max_width,
                     .cuda_stream = p.stream,
                     .output_uint8 = p.output_uint8});
            });
            initialized = true;
        }

        bool has_cuda_device() {
            return lfs::core::gpu_backend_available(lfs::core::GpuBackend::CUDA);
        }

        void writeU8Image(const std::filesystem::path& path,
                          const int width,
                          const int height,
                          const int channels,
                          std::vector<std::uint8_t> pixels,
                          const int jpeg_quality = 100) {
            if (pixels.size() != static_cast<std::size_t>(width) * height * channels) {
                throw std::invalid_argument("test image byte count does not match its shape");
            }
            auto image = lfs::core::Tensor::from_blob(
                             pixels.data(),
                             {static_cast<std::size_t>(height),
                              static_cast<std::size_t>(width),
                              static_cast<std::size_t>(channels)},
                             lfs::core::Device::CPU,
                             lfs::core::DataType::UInt8)
                             .clone();
            lfs::core::save_image_u8(path, std::move(image), jpeg_quality);
        }
    } // namespace

    class RenderingManagerEventsTest : public ::testing::Test {
    protected:
        void SetUp() override {
            lfs::event::EventBridge::instance().clear_all();
            lfs::core::event::bus().clear_all();
            services().clear();
        }

        void TearDown() override {
            services().clear();
            lfs::event::EventBridge::instance().clear_all();
            lfs::core::event::bus().clear_all();
        }
    };

    namespace {
        void addDatasetCameraWithImage(SceneManager& manager) {
            manager.changeContentType(SceneManager::ContentType::Dataset);
            auto& scene = manager.getScene();
            const auto group = scene.addCameraGroup("Cameras", scene.addGroup("Dataset"), 1);
            auto camera = std::make_shared<lfs::core::Camera>(
                lfs::core::Tensor::eye(3, lfs::core::Device::CPU),
                lfs::core::Tensor::zeros({size_t{3}}, lfs::core::Device::CPU),
                100.0f, 100.0f, 32.0f, 32.0f,
                lfs::core::Tensor(), lfs::core::Tensor(), lfs::core::CameraModelType::PINHOLE,
                "source", std::filesystem::path("source.png"), std::filesystem::path{},
                64, 64, 1);
            scene.addCamera("source", group, std::move(camera));
        }
    } // namespace

    class SceneManagerRenderStateTest : public ::testing::Test {
    protected:
        void SetUp() override {
            lfs::event::EventBridge::instance().clear_all();
            lfs::core::event::bus().clear_all();
            services().clear();
            op::undoHistory().clear();
        }

        void TearDown() override {
            op::undoHistory().clear();
            services().clear();
            lfs::event::EventBridge::instance().clear_all();
            lfs::core::event::bus().clear_all();
        }
    };

    TEST(SplitViewServiceTest, ToggleGtComparisonRestoresPreviousProjectionMode) {
        SplitViewService service;
        RenderSettings settings;
        settings.equirectangular = true;

        const auto enable = service.toggleMode(settings, SplitViewMode::GTComparison);
        EXPECT_TRUE(enable.mode_changed);
        EXPECT_EQ(enable.previous_mode, SplitViewMode::Disabled);
        EXPECT_EQ(enable.current_mode, SplitViewMode::GTComparison);
        EXPECT_EQ(settings.split_view_mode, SplitViewMode::GTComparison);

        settings.equirectangular = false;

        const auto disable = service.toggleMode(settings, SplitViewMode::GTComparison);
        EXPECT_TRUE(disable.mode_changed);
        EXPECT_EQ(disable.previous_mode, SplitViewMode::GTComparison);
        EXPECT_EQ(disable.current_mode, SplitViewMode::Disabled);
        ASSERT_TRUE(disable.restore_equirectangular.has_value());
        EXPECT_TRUE(*disable.restore_equirectangular);
        EXPECT_TRUE(settings.equirectangular);
        EXPECT_EQ(settings.split_view_mode, SplitViewMode::Disabled);
    }

    TEST(GTComparisonCache, UInt8PreviewAccountingKeepsCurrentAndNeighbor) {
        constexpr std::size_t budget = 128ULL * 1024ULL * 1024ULL;
        const auto current = gt_comparison_detail::previewBytes({3840, 2160});
        const auto neighbor = gt_comparison_detail::previewBytes({3840, 2160});
        EXPECT_EQ(current, 3840ULL * 2160ULL * 3ULL);
        EXPECT_TRUE(gt_comparison_detail::prefetchFits(0, current, neighbor, budget));
        EXPECT_FALSE(gt_comparison_detail::prefetchFits(0, 80ULL, 40ULL, 100ULL));
    }

    TEST(GTComparisonCache, DisplayConversionMatchesFloatChwAndUInt8Hwc) {
        using lfs::core::DataType;
        using lfs::core::Device;
        using lfs::core::Tensor;

        constexpr std::size_t plane = 2 * 2;
        const std::array<std::uint8_t, 12> hwc_bytes{
            0, 255, 191,
            128, 64, 0,
            255, 128, 64,
            64, 0, 255};
        std::vector<float> float_chw_values(3 * plane);
        for (std::size_t pixel = 0; pixel < plane; ++pixel) {
            for (std::size_t channel = 0; channel < 3; ++channel) {
                float_chw_values[channel * plane + pixel] =
                    static_cast<float>(hwc_bytes[pixel * 3 + channel]) / 255.0f;
            }
        }
        const auto float_chw = std::make_shared<Tensor>(Tensor::from_vector(
            float_chw_values,
            {size_t{3}, size_t{2}, size_t{2}}, Device::CPU));
        const auto uint8_hwc = std::make_shared<Tensor>(Tensor::empty(
            {size_t{2}, size_t{2}, size_t{3}}, Device::CPU, DataType::UInt8));
        std::memcpy(uint8_hwc->ptr<std::uint8_t>(), hwc_bytes.data(), hwc_bytes.size());

        const auto float_preview = gt_comparison_detail::convertDisplayTensorToUInt8(float_chw);
        const auto uint8_preview = gt_comparison_detail::convertDisplayTensorToUInt8(uint8_hwc);
        ASSERT_TRUE(float_preview);
        ASSERT_TRUE(uint8_preview);
        ASSERT_EQ(float_preview->shape(), uint8_preview->shape());
        const auto* const float_preview_bytes = float_preview->ptr<std::uint8_t>();
        const auto* const uint8_preview_bytes = uint8_preview->ptr<std::uint8_t>();
        std::size_t first_mismatch = float_preview->bytes();
        for (std::size_t index = 0; index < float_preview->bytes(); ++index) {
            if (float_preview_bytes[index] != uint8_preview_bytes[index]) {
                first_mismatch = index;
                break;
            }
        }
        const auto float_value = first_mismatch < float_preview->bytes()
                                     ? float_preview_bytes[first_mismatch]
                                     : std::uint8_t{0};
        const auto uint8_value = first_mismatch < uint8_preview->bytes()
                                     ? uint8_preview_bytes[first_mismatch]
                                     : std::uint8_t{0};
        EXPECT_EQ(first_mismatch, float_preview->bytes())
            << "first mismatching index=" << first_mismatch
            << ", float byte=" << static_cast<unsigned int>(float_value)
            << ", uint8 byte=" << static_cast<unsigned int>(uint8_value);

        const auto uint8_chw = std::make_shared<Tensor>(Tensor::empty(
            {size_t{3}, size_t{2}, size_t{2}}, Device::CPU, DataType::UInt8));
        const std::array<std::uint8_t, 12> chw_bytes{
            0, 128, 255, 64,
            255, 64, 128, 0,
            191, 0, 64, 255};
        std::memcpy(uint8_chw->ptr<std::uint8_t>(), chw_bytes.data(), chw_bytes.size());
        const auto uint8_chw_preview = gt_comparison_detail::convertDisplayTensorToUInt8(uint8_chw);
        ASSERT_EQ(uint8_chw_preview.get(), uint8_chw.get());
        EXPECT_EQ(std::memcmp(uint8_chw_preview->ptr<std::uint8_t>(),
                              chw_bytes.data(),
                              chw_bytes.size()),
                  0);
    }

    TEST(GTComparisonCache, DisplayConversionCopiesCudaUInt8ChwToCpu) {
        using lfs::core::DataType;
        using lfs::core::Device;
        using lfs::core::Tensor;

        const std::array<std::uint8_t, 12> chw_bytes{
            0, 128, 255, 64,
            255, 64, 128, 0,
            191, 0, 64, 255};
        auto cpu_uint8_chw = std::make_shared<Tensor>(Tensor::empty(
            {size_t{3}, size_t{2}, size_t{2}}, Device::CPU, DataType::UInt8));
        std::memcpy(cpu_uint8_chw->ptr<std::uint8_t>(), chw_bytes.data(), chw_bytes.size());
        const auto cuda_uint8_chw = std::make_shared<Tensor>(cpu_uint8_chw->cuda());

        const auto preview = gt_comparison_detail::convertDisplayTensorToUInt8(cuda_uint8_chw);
        ASSERT_TRUE(preview);
        EXPECT_EQ(preview->device(), Device::CPU);
        EXPECT_NE(preview.get(), cuda_uint8_chw.get());
        EXPECT_EQ(std::memcmp(preview->ptr<std::uint8_t>(), chw_bytes.data(), chw_bytes.size()), 0);
    }

    TEST(GTComparisonCache, RightImageGenerationUsesFrameGenerationForRecycledTarget) {
        constexpr glm::ivec2 size{640, 480};
        constexpr auto stable_bit = gt_comparison_detail::SPLIT_RIGHT_GENERATION_BIT;
        const int recycled_address = 1;
        const int held_display = 2;
        std::uint64_t generation = 7;

        gt_comparison_detail::updateSplitImageGeneration(
            &recycled_address, size, nullptr, size, 11, generation);
        EXPECT_EQ(generation, 11U);
        // A new ordinary tensor may reuse the same address; it still carries
        // the current frame generation and must not inherit the old upload key.
        gt_comparison_detail::updateSplitImageGeneration(
            &recycled_address, size, nullptr, size, 12, generation);
        EXPECT_EQ(generation, 12U);
        gt_comparison_detail::updateSplitImageGeneration(
            &held_display, size, &held_display, size, 13, generation);
        EXPECT_EQ(generation, 12U | stable_bit);
        gt_comparison_detail::updateSplitImageGeneration(
            &held_display, {800, 600}, &held_display, size, 14, generation);
        EXPECT_EQ(generation, 14U);
    }

    TEST(SceneCameraTraining, UnknownUidIsEnabled) {
        const lfs::core::Scene scene;
        EXPECT_TRUE(scene.isCameraTrainingEnabled(123456));
    }

    TEST(SplitViewServiceTest, UpdateInfoClearsStaleSplitViewLabels) {
        SplitViewService service;

        FrameResources active_resources;
        active_resources.split_view_executed = true;
        active_resources.split_info = {.enabled = true, .left_name = "Left", .right_name = "Right"};
        service.updateInfo(active_resources);

        const auto active_info = service.getInfo();
        EXPECT_TRUE(active_info.enabled);
        EXPECT_EQ(active_info.left_name, "Left");
        EXPECT_EQ(active_info.right_name, "Right");

        FrameResources idle_resources;
        service.updateInfo(idle_resources);

        const auto idle_info = service.getInfo();
        EXPECT_FALSE(idle_info.enabled);
        EXPECT_TRUE(idle_info.left_name.empty());
        EXPECT_TRUE(idle_info.right_name.empty());
    }

    TEST(GTComparisonTest, LossHeatmapIsBlackForMatchingPixelsAndHighlightsLargerErrors) {
        const glm::vec3 ground_truth(0.4f, 0.5f, 0.6f);
        const glm::vec3 matching = gtLossHeatmapColor(ground_truth, ground_truth);
        const glm::vec3 small_error = gtLossHeatmapColor(
            ground_truth, ground_truth + glm::vec3(0.01f));
        const glm::vec3 large_error = gtLossHeatmapColor(
            ground_truth, ground_truth + glm::vec3(0.5f));

        EXPECT_EQ(matching, glm::vec3(0.0f));
        EXPECT_GT(small_error.r + small_error.g + small_error.b, 0.0f);
        EXPECT_GT(large_error.r + large_error.g + large_error.b,
                  small_error.r + small_error.g + small_error.b);
    }

    TEST(GTComparisonTest, LossModeSurvivesSettingsSanitization) {
        RenderSettings settings;
        settings.gt_comparison_mode = GTComparisonMode::Loss;

        sanitizeGTComparisonSettings(settings);

        EXPECT_EQ(settings.gt_comparison_mode, GTComparisonMode::Loss);
    }

    TEST_F(RenderingManagerEventsTest, LossModeDoesNotExposeDraggableDivider) {
        lfs::vis::screen::ScreenService manager_views;
        RenderingManager manager{manager_views};
        auto settings = manager.getSettings();
        settings.split_view_mode = SplitViewMode::GTComparison;
        settings.gt_comparison_mode = GTComparisonMode::RGB;
        manager.updateSettings(settings);
        ASSERT_TRUE(manager.getSplitDividerScreenX(manager.activeViewId(), {10.0f, 20.0f}, {800.0f, 600.0f}).has_value());

        settings.gt_comparison_mode = GTComparisonMode::Loss;
        manager.updateSettings(settings);

        EXPECT_FALSE(manager.getSplitDividerScreenX(manager.activeViewId(), {10.0f, 20.0f}, {800.0f, 600.0f}).has_value());
    }

    TEST(SplitViewServiceTest, SceneClearedDisablesSplitViewAndResetsOffset) {
        SplitViewService service;
        RenderSettings settings;
        settings.split_view_mode = SplitViewMode::PLYComparison;
        settings.split_view_offset = 3;

        const auto result = service.handleSceneCleared(settings);

        EXPECT_TRUE(result.mode_changed);
        EXPECT_EQ(settings.split_view_mode, SplitViewMode::Disabled);
        EXPECT_EQ(settings.split_view_offset, 0);
    }

    TEST(SplitViewServiceTest, GtRenderCameraUsesVisualizerCameraAxesAndNormalizedSceneRotation) {
        using lfs::core::Camera;
        using lfs::core::CameraModelType;
        using lfs::core::Device;
        using lfs::core::Tensor;

        Camera camera(
            Tensor::from_vector(
                {1.0f, 0.0f, 0.0f,
                 0.0f, 1.0f, 0.0f,
                 0.0f, 0.0f, 1.0f},
                {size_t{3}, size_t{3}},
                Device::CPU),
            Tensor::from_vector({0.0f, 0.0f, 0.0f}, {size_t{3}}, Device::CPU),
            500.0f,
            600.0f,
            320.0f,
            240.0f,
            Tensor(),
            Tensor(),
            CameraModelType::PINHOLE,
            "test.png",
            {},
            {},
            640,
            480,
            7);

        glm::mat4 scene_transform(1.0f);
        scene_transform = glm::translate(scene_transform, glm::vec3(1.0f, 2.0f, 3.0f));
        scene_transform = glm::scale(scene_transform, glm::vec3(2.0f, 3.0f, 4.0f));

        const auto render_camera =
            detail::buildGTRenderCamera(camera, {1280, 960}, scene_transform);
        ASSERT_TRUE(render_camera.has_value());

        expectMat3Near(
            render_camera->rotation,
            lfs::rendering::DATA_TO_VISUALIZER_CAMERA_AXES);
        EXPECT_EQ(render_camera->translation, glm::vec3(1.0f, 2.0f, 3.0f));
        ASSERT_TRUE(render_camera->intrinsics.has_value());
        EXPECT_FLOAT_EQ(render_camera->intrinsics->focal_x, 1000.0f);
        EXPECT_FLOAT_EQ(render_camera->intrinsics->focal_y, 1200.0f);
        EXPECT_FLOAT_EQ(render_camera->intrinsics->center_x, 640.0f);
        EXPECT_FLOAT_EQ(render_camera->intrinsics->center_y, 480.0f);
        EXPECT_FALSE(render_camera->equirectangular);
    }

    TEST(SplitCaptureCompositorTest, LossCaptureKeepsHeatmapWithoutDivider) {
        using lfs::core::Device;
        using lfs::core::Tensor;
        VulkanSplitViewParams params;
        params.enabled = true;
        params.loss_visualization = true;
        params.content_rect = {0, 0, 16, 12};
        params.left.image = std::make_shared<Tensor>(Tensor::zeros({3, 12, 16}, Device::CPU));
        params.right.image = std::make_shared<Tensor>(Tensor::full({3, 12, 16}, 0.25f, Device::CPU));
        const auto expected = gtLossHeatmapColor(glm::vec3(0.0f), glm::vec3(0.25f));
        for (const bool exact : {false, true}) {
            params.exact_texel_sampling = exact;
            const auto capture = composeSplitCaptureCpu(params, {16, 12});
            ASSERT_TRUE(capture && capture->is_valid());
            const auto pixels = capture->to_vector();
            for (int channel = 0; channel < 3; ++channel)
                for (int pixel = 0; pixel < 16 * 12; ++pixel)
                    EXPECT_NEAR(pixels[channel * 16 * 12 + pixel], expected[channel], 1e-6f);
        }
    }

    TEST(SplitViewServiceTest, ActualSizeCropCentersAndClampsInIntegerPixels) {
        const auto centered =
            detail::centerGTComparisonCrop({8192, 6144}, {3840, 2160});
        EXPECT_EQ(centered.origin, glm::ivec2(2176, 1992));
        EXPECT_EQ(centered.extent, glm::ivec2(3840, 2160));

        const auto clamped =
            detail::clampGTComparisonCrop({8192, 6144}, {3840, 2160}, {-20, 9000});
        EXPECT_EQ(clamped.origin, glm::ivec2(0, 3984));
        EXPECT_EQ(clamped.extent, glm::ivec2(3840, 2160));

        const auto letterboxed =
            detail::centerGTComparisonCrop({1000, 3000}, {2000, 1000});
        EXPECT_EQ(letterboxed.origin, glm::ivec2(0, 1000));
        EXPECT_EQ(letterboxed.extent, glm::ivec2(1000, 1000));

        const auto resized =
            detail::centerGTComparisonCrop({8192, 6144}, {2560, 1440});
        EXPECT_EQ(resized.origin, glm::ivec2(2816, 2352));
        EXPECT_EQ(resized.extent, glm::ivec2(2560, 1440));

        const auto smaller_than_viewport =
            detail::centerGTComparisonCrop({1000, 800}, {2000, 1000});
        EXPECT_EQ(smaller_than_viewport.origin, glm::ivec2(0, 0));
        EXPECT_EQ(smaller_than_viewport.extent, glm::ivec2(1000, 800));
    }

    TEST(SplitViewServiceTest, ActualSizeCropFromCenterGrowsShrinksAndClamps) {
        const auto grown = detail::cropGTComparisonFromCenter(
            {8192, 6144}, {2560, 1440}, {3960.0, 2540.0});
        EXPECT_EQ(grown.origin, glm::ivec2(2680, 1820));
        EXPECT_EQ(grown.extent, glm::ivec2(2560, 1440));

        const auto shrunk = detail::cropGTComparisonFromCenter(
            {8192, 6144}, {1280, 720}, {3960.0, 2540.0});
        EXPECT_EQ(shrunk.origin, glm::ivec2(3320, 2180));
        EXPECT_EQ(shrunk.extent, glm::ivec2(1280, 720));

        const auto odd = detail::cropGTComparisonFromCenter(
            {101, 101}, {4, 6}, {12.5, 22.5});
        EXPECT_EQ(odd.origin, glm::ivec2(11, 20));
        EXPECT_EQ(odd.extent, glm::ivec2(4, 6));

        const auto one_axis_letterboxed =
            detail::cropGTComparisonFromCenter(
                {1000, 3000}, {2000, 1500}, {500.0, 1500.0});
        EXPECT_EQ(one_axis_letterboxed.origin, glm::ivec2(0, 750));
        EXPECT_EQ(one_axis_letterboxed.extent, glm::ivec2(1000, 1500));

        const auto edge_clamped = detail::cropGTComparisonFromCenter(
            {8192, 6144}, {3840, 2160}, {7232.0, 5604.0});
        EXPECT_EQ(edge_clamped.origin, glm::ivec2(4352, 3984));
        EXPECT_EQ(edge_clamped.extent, glm::ivec2(3840, 2160));

        const auto hidpi = detail::cropGTComparisonFromCenter(
            {8192, 6144}, {3001, 1501}, {3500.0, 2400.0});
        EXPECT_EQ(hidpi.origin, glm::ivec2(2000, 1650));
        EXPECT_EQ(hidpi.extent, glm::ivec2(3001, 1501));

        const auto centered = detail::cropGTComparisonFromCenter(
            {8192, 6144}, {2560, 1440}, {4096.0, 3072.0});
        EXPECT_EQ(centered.origin, glm::ivec2(2816, 2352));
        EXPECT_EQ(centered.extent, glm::ivec2(2560, 1440));
    }

    TEST(SplitViewServiceTest, ActualSizeGeometryUsesPhysicalEdgesAndOneRoundedDrag) {
        const glm::ivec2 physical_extent{2001, 1001};
        const glm::ivec2 logical_extent{1000, 500};
        const auto physical_rect =
            detail::centeredGTComparisonContentRect(physical_extent, {1000, 800});
        EXPECT_EQ(physical_rect, glm::ivec4(500, 100, 1000, 800));

        const auto logical_rect = detail::physicalToLogicalContentRect(
            physical_rect, physical_extent, logical_extent);
        EXPECT_NEAR(logical_rect.x, 500.0 * 1000.0 / 2001.0, 1.0e-4);
        EXPECT_NEAR(logical_rect.y, 100.0 * 500.0 / 1001.0, 1.0e-4);
        EXPECT_NEAR(logical_rect.z, 1000.0 * 1000.0 / 2001.0, 1.0e-4);
        EXPECT_NEAR(logical_rect.w, 800.0 * 500.0 / 1001.0, 1.0e-4);

        EXPECT_EQ(splitViewDividerPixel(200, 0.25f), 50);
        EXPECT_EQ(splitViewDividerPixel(200, 0.5f), 100);
        EXPECT_EQ(splitViewDividerPixel(200, 0.75f), 150);

        for (const double ratio : {1.0, 1.25, 1.5, 2.0}) {
            // A physical drag of 20 pixels arrives identically at all ratios.
            const glm::dvec2 window_drag = glm::dvec2(20.0, -12.0) / ratio;
            EXPECT_EQ(detail::roundedPhysicalDrag(window_drag * ratio), glm::ivec2(20, -12));
        }
    }

    TEST(SplitViewServiceTest, ActualSizeSupportsPerspectiveCameraModelsOnly) {
        using lfs::core::CameraModelType;

        EXPECT_TRUE(detail::isGTComparisonActualSizeCameraModelSupported(
            CameraModelType::PINHOLE));
        EXPECT_TRUE(detail::isGTComparisonActualSizeCameraModelSupported(
            CameraModelType::FISHEYE));
        EXPECT_TRUE(detail::isGTComparisonActualSizeCameraModelSupported(
            CameraModelType::THIN_PRISM_FISHEYE));
        EXPECT_FALSE(detail::isGTComparisonActualSizeCameraModelSupported(
            CameraModelType::ORTHO));
        EXPECT_FALSE(detail::isGTComparisonActualSizeCameraModelSupported(
            CameraModelType::EQUIRECTANGULAR));
    }

    TEST(GTComparisonCropTest, OddEvenResizeCyclesPreserveUnroundedCenterAndPanReanchors) {
        const glm::ivec2 full{8193, 4321};
        const glm::dvec2 center{4021.5, 2140.5};
        const auto original = detail::cropGTComparisonFromCenter(full, {1001, 701}, center);
        for (int i = 0; i < 1000; ++i) {
            for (const auto size : {glm::ivec2(1000, 700), glm::ivec2(1002, 702), glm::ivec2(9000, 5000), glm::ivec2(1001, 701)}) {
                const auto crop = detail::cropGTComparisonFromCenter(full, size, center);
                EXPECT_TRUE(detail::isGTComparisonCropValidForViewport(full, size, crop));
            }
            EXPECT_EQ(detail::cropGTComparisonFromCenter(full, {1001, 701}, center), original);
        }
        const auto displayed = detail::cropGTComparisonFromCenter(full, {1000, 700}, center);
        const glm::dvec2 panned_center = glm::dvec2(displayed.origin - glm::ivec2(15, -9)) + glm::dvec2(displayed.extent) * 0.5;
        const auto panned = detail::cropGTComparisonFromCenter(full, displayed.extent, panned_center);
        EXPECT_EQ(panned.origin, displayed.origin - glm::ivec2(15, -9));
        EXPECT_EQ(detail::cropGTComparisonFromCenter(full, {1000, 700}, panned_center), panned);
    }

    TEST(GTComparisonSettingsTest, UnsupportedViewProjectionsClearNativeRequestPermanently) {
        for (const bool orthographic : {false, true}) {
            screen::ScreenService views;
            const auto id = views.activeView();
            ASSERT_TRUE(views.editViewSettings(id, [&](auto& s) {
                s.gt_comparison_actual_size = true;
                s.orthographic = orthographic;
                s.equirectangular = !orthographic;
            }));
            EXPECT_FALSE(views.viewSettings(id)->gt_comparison_actual_size);
            EXPECT_FALSE(gtComparisonActualSizeEligible(*views.viewSettings(id)));
            ASSERT_TRUE(views.editViewSettings(id, [](auto& s) { s.orthographic = false; s.equirectangular = false; }));
            EXPECT_TRUE(gtComparisonActualSizeEligible(*views.viewSettings(id)));
            EXPECT_FALSE(views.viewSettings(id)->gt_comparison_actual_size);
        }
    }

    TEST(ViewportRequestBuilderTest, PointCloudProjectionUsesCropIntrinsicsAndPreservesClipOrientation) {
        lfs::rendering::PointCloudRenderRequest frame;
        frame.frame_view.size = {319, 241};
        frame.frame_view.intrinsics_override = lfs::rendering::CameraIntrinsics{
            .focal_x = 501.0f,
            .focal_y = 603.0f,
            .center_x = 117.25f,
            .center_y = 151.75f};
        const auto request = buildPointSceneRequest(frame, RenderSettings{});
        EXPECT_FLOAT_EQ(request.focal_y, 603.0f);
        for (const auto point : {glm::vec3(0.0f, 0.0f, -2.0f), glm::vec3(0.1f, -0.15f, -2.0f)}) {
            const auto world = glm::inverse(frame.frame_view.getViewMatrix()) * glm::vec4(point, 1.0f);
            const auto clip = request.view_projection * world;
            const glm::vec2 pixel = (glm::vec2(clip) / clip.w + 1.0f) * glm::vec2(frame.frame_view.size) * 0.5f;
            EXPECT_NEAR(pixel.x, 501.0f * point.x / -point.z + 117.25f, 1.0e-4f);
            EXPECT_NEAR(pixel.y, -603.0f * point.y / -point.z + 151.75f, 1.0e-4f);
        }
    }

    TEST(SplitViewServiceTest, ActualSizeCropPreservesFocalLengthAndOffsetsPrincipalPoint) {
        using lfs::core::Camera;
        using lfs::core::CameraModelType;
        using lfs::core::Device;
        using lfs::core::Tensor;

        Camera camera(
            Tensor::from_vector(
                {1.0f, 0.0f, 0.0f,
                 0.0f, 1.0f, 0.0f,
                 0.0f, 0.0f, 1.0f},
                {size_t{3}, size_t{3}},
                Device::CPU),
            Tensor::from_vector({0.0f, 0.0f, 0.0f}, {size_t{3}}, Device::CPU),
            500.0f,
            600.0f,
            320.0f,
            240.0f,
            Tensor(),
            Tensor(),
            CameraModelType::PINHOLE,
            "test.png",
            {},
            {},
            640,
            480,
            8);

        const auto render_camera = detail::buildGTRenderCamera(
            camera,
            {320, 240},
            glm::mat4(1.0f),
            detail::GTComparisonPixelRegion{
                .origin = {100, 50},
                .full_extent = {640, 480}});
        ASSERT_TRUE(render_camera);
        ASSERT_TRUE(render_camera->intrinsics);
        EXPECT_FLOAT_EQ(render_camera->intrinsics->focal_x, 500.0f);
        EXPECT_FLOAT_EQ(render_camera->intrinsics->focal_y, 600.0f);
        EXPECT_FLOAT_EQ(render_camera->intrinsics->center_x, 220.0f);
        EXPECT_FLOAT_EQ(render_camera->intrinsics->center_y, 190.0f);
    }

    TEST(SplitViewServiceTest, ActualSizePointCloudRenderMatchesFitCropAndPanPixels) {
        if (!has_cuda_device()) {
            GTEST_SKIP() << "CUDA device required";
        }
        using lfs::core::Device;
        using lfs::core::Tensor;

        lfs::core::Camera camera(
            Tensor::from_vector({1.0f, 0.0f, 0.0f,
                                 0.0f, 1.0f, 0.0f,
                                 0.0f, 0.0f, 1.0f},
                                {size_t{3}, size_t{3}}, Device::CPU),
            Tensor::from_vector({0.0f, 0.0f, 0.0f}, {size_t{3}}, Device::CPU),
            500.0f, 600.0f, 320.0f, 240.0f,
            Tensor(), Tensor(), lfs::core::CameraModelType::PINHOLE,
            "test.png", {}, {}, 640, 480, 8);
        lfs::core::PointCloud points(
            Tensor::from_vector({0.0f, 0.0f, 2.0f,
                                 0.12f, -0.08f, 2.0f,
                                 -0.08f, 0.10f, 2.0f},
                                {size_t{3}, size_t{3}}, Device::CPU),
            Tensor::from_vector({1.0f, 0.0f, 0.0f,
                                 0.0f, 1.0f, 0.0f,
                                 0.0f, 0.0f, 1.0f},
                                {size_t{3}, size_t{3}}, Device::CPU));
        auto engine = lfs::rendering::RenderingEngine::create();
        ASSERT_TRUE(engine->initialize());

        const auto expect_pixels = [&](const std::optional<detail::GTComparisonPixelRegion> region,
                                       const std::array<glm::ivec2, 3>& centers,
                                       const int radius) {
            const auto render_camera = detail::buildGTRenderCamera(
                camera, {320, 240}, glm::mat4(1.0f), region);
            ASSERT_TRUE(render_camera);
            lfs::rendering::PointCloudRenderRequest request;
            request.frame_view.size = {320, 240};
            request.frame_view.rotation = render_camera->rotation;
            request.frame_view.translation = render_camera->translation;
            request.frame_view.intrinsics_override = render_camera->intrinsics;
            request.render.voxel_size = 0.02f;
            auto rendered = engine->renderPointCloudImage(points, request);
            ASSERT_TRUE(rendered) << rendered.error();
            ASSERT_TRUE(rendered->image);
            const auto image = rendered->image->cpu().contiguous();
            const float* pixels = image.ptr<float>();
            for (int channel = 0; channel < 3; ++channel) {
                SCOPED_TRACE(channel);
                glm::ivec2 minimum{320, 240};
                glm::ivec2 maximum{-1, -1};
                for (int y = 0; y < 240; ++y) {
                    for (int x = 0; x < 320; ++x) {
                        // Point-cloud images and GT pixel coordinates both use top-down rows.
                        if (pixels[channel * 320 * 240 + y * 320 + x] > 0.5f) {
                            minimum = glm::min(minimum, glm::ivec2(x, y));
                            maximum = glm::max(maximum, glm::ivec2(x, y));
                        }
                    }
                }
                EXPECT_EQ(minimum, centers[channel] - glm::ivec2(radius));
                EXPECT_EQ(maximum, centers[channel] + glm::ivec2(radius));
            }
        };

        // The same three projected points in a half-size Fit preview and native crops.
        expect_pixels(std::nullopt, {{{160, 120}, {175, 108}, {150, 135}}}, 3);
        expect_pixels(detail::GTComparisonPixelRegion{.origin = {100, 50}, .full_extent = {640, 480}},
                      {{{220, 190}, {250, 166}, {200, 220}}}, 6);
        expect_pixels(detail::GTComparisonPixelRegion{.origin = {117, 37}, .full_extent = {640, 480}},
                      {{{203, 203}, {233, 179}, {183, 233}}}, 6);
    }

    TEST(SplitViewServiceTest, ActualSizeCropScalesUndistortedIntrinsicsBeforeOffset) {
        using lfs::core::Camera;
        using lfs::core::CameraModelType;
        using lfs::core::Device;
        using lfs::core::Tensor;

        Camera camera(
            Tensor::from_vector(
                {1.0f, 0.0f, 0.0f,
                 0.0f, 1.0f, 0.0f,
                 0.0f, 0.0f, 1.0f},
                {size_t{3}, size_t{3}},
                Device::CPU),
            Tensor::from_vector({0.0f, 0.0f, 0.0f}, {size_t{3}}, Device::CPU),
            500.0f,
            600.0f,
            320.0f,
            240.0f,
            Tensor::from_vector({-0.08f, 0.01f}, {size_t{2}}, Device::CPU),
            Tensor(),
            CameraModelType::PINHOLE,
            "test.png",
            {},
            {},
            640,
            480,
            9);
        camera.precompute_undistortion();
        ASSERT_TRUE(camera.is_undistort_precomputed());
        const auto& undistort = camera.undistort_params();
        const auto scaled = lfs::core::scale_undistort_params(
            undistort, 1001, 751);
        const glm::ivec2 full_extent{
            scaled.dst_width,
            scaled.dst_height};
        const glm::ivec2 crop_origin{37, 29};

        const auto render_camera = detail::buildGTRenderCamera(
            camera,
            {320, 240},
            glm::mat4(1.0f),
            detail::GTComparisonPixelRegion{
                .origin = crop_origin,
                .full_extent = full_extent,
                .full_intrinsics = lfs::rendering::CameraIntrinsics{
                    .focal_x = scaled.dst_fx,
                    .focal_y = scaled.dst_fy,
                    .center_x = scaled.dst_cx,
                    .center_y = scaled.dst_cy}});
        ASSERT_TRUE(render_camera);
        ASSERT_TRUE(render_camera->intrinsics);
        EXPECT_FLOAT_EQ(render_camera->intrinsics->focal_x, scaled.dst_fx);
        EXPECT_FLOAT_EQ(render_camera->intrinsics->focal_y, scaled.dst_fy);
        EXPECT_FLOAT_EQ(
            render_camera->intrinsics->center_x,
            scaled.dst_cx - static_cast<float>(crop_origin.x));
        EXPECT_FLOAT_EQ(
            render_camera->intrinsics->center_y,
            scaled.dst_cy - static_cast<float>(crop_origin.y));
    }

    TEST(SplitViewServiceTest, RestoredUndistortionCalibrationSurvivesTransformCopies) {
        using lfs::core::Camera;
        using lfs::core::CameraCalibration;
        using lfs::core::CameraModelType;
        using lfs::core::Device;
        using lfs::core::Tensor;

        const auto rotation = Tensor::from_vector(
            {1.0f, 0.0f, 0.0f,
             0.0f, 1.0f, 0.0f,
             0.0f, 0.0f, 1.0f},
            {size_t{3}, size_t{3}}, Device::CPU);
        const auto translation =
            Tensor::from_vector({0.0f, 0.0f, 0.0f}, {size_t{3}}, Device::CPU);
        const CameraCalibration source{
            .fx = 801.25f,
            .fy = 799.75f,
            .cx = 639.5f,
            .cy = 359.25f,
            .width = 1280,
            .height = 720};
        const CameraCalibration destination{
            .fx = 733.125f,
            .fy = 731.875f,
            .cx = 602.75f,
            .cy = 341.5f,
            .width = 1207,
            .height = 683};

        const std::array models{
            CameraModelType::PINHOLE,
            CameraModelType::FISHEYE,
            CameraModelType::THIN_PRISM_FISHEYE};
        for (const auto model : models) {
            const auto radial = model == CameraModelType::PINHOLE
                                    ? Tensor::from_vector({-0.08f, 0.01f}, {size_t{2}}, Device::CPU)
                                    : Tensor::from_vector(
                                          {0.04f, -0.005f, 0.001f, -0.0002f},
                                          {size_t{4}}, Device::CPU);
            const auto tangential = model == CameraModelType::THIN_PRISM_FISHEYE
                                        ? Tensor::from_vector(
                                              {0.001f, -0.0015f, 0.0005f, -0.0004f},
                                              {size_t{4}}, Device::CPU)
                                        : Tensor();
            for (const bool prepared : {false, true}) {
                SCOPED_TRACE(static_cast<int>(model));
                SCOPED_TRACE(prepared);
                Camera camera(
                    rotation, translation,
                    source.fx, source.fy, source.cx, source.cy,
                    radial, tangential, model,
                    "calibration.png", "calibration.png", {},
                    source.width, source.height, 101);
                camera.restore_undistortion_state(
                    source, destination, prepared, true);

                ASSERT_TRUE(camera.is_undistort_precomputed());
                EXPECT_EQ(camera.is_undistort_prepared(), prepared);
                const auto& params = camera.undistort_params();
                EXPECT_FLOAT_EQ(params.src_fx, source.fx);
                EXPECT_FLOAT_EQ(params.src_fy, source.fy);
                EXPECT_FLOAT_EQ(params.src_cx, source.cx);
                EXPECT_FLOAT_EQ(params.src_cy, source.cy);
                EXPECT_EQ(params.src_width, source.width);
                EXPECT_EQ(params.src_height, source.height);
                EXPECT_FLOAT_EQ(params.dst_fx, destination.fx);
                EXPECT_FLOAT_EQ(params.dst_fy, destination.fy);
                EXPECT_FLOAT_EQ(params.dst_cx, destination.cx);
                EXPECT_FLOAT_EQ(params.dst_cy, destination.cy);
                EXPECT_EQ(params.dst_width, destination.width);
                EXPECT_EQ(params.dst_height, destination.height);
                EXPECT_EQ(params.model_type, model);
                EXPECT_TRUE(params.crop_solve_failed);

                Camera transformed(camera, camera.world_view_transform());
                EXPECT_TRUE(transformed.is_undistort_precomputed());
                EXPECT_EQ(transformed.is_undistort_prepared(), prepared);
                EXPECT_EQ(transformed.undistort_params().src_width, source.width);
                EXPECT_EQ(transformed.undistort_params().dst_width, destination.width);
                EXPECT_FLOAT_EQ(transformed.undistort_params().dst_fx, destination.fx);
                EXPECT_EQ(transformed.undistort_params().model_type, model);
                EXPECT_EQ(
                    transformed.undistort_params().num_distortion,
                    params.num_distortion);
                EXPECT_TRUE(transformed.undistort_params().crop_solve_failed);
                for (int i = 0; i < 12; ++i) {
                    EXPECT_FLOAT_EQ(
                        transformed.undistort_params().distortion[i],
                        params.distortion[i]);
                }
                const auto& current = prepared ? destination : source;
                EXPECT_FLOAT_EQ(transformed.focal_x(), current.fx);
                EXPECT_FLOAT_EQ(transformed.focal_y(), current.fy);
                EXPECT_FLOAT_EQ(transformed.center_x(), current.cx);
                EXPECT_FLOAT_EQ(transformed.center_y(), current.cy);
                EXPECT_EQ(transformed.camera_width(), current.width);
                EXPECT_EQ(transformed.camera_height(), current.height);
            }
        }
    }

    TEST(RenderingManagerActualSizeStateTest,
         PublishesOnlyAtCommitAndRetainsAcrossResourceInvalidation) {
        lfs::vis::screen::ScreenService manager_views;
        RenderingManager manager{manager_views};
        const detail::GTComparisonSourceKey source_key{
            .camera_uid = 17,
            .image_path = "frame.png"};
        const RenderingManager::GTComparisonActualFrame::Snapshot prepared{
            .source_key = source_key,
            .source_generation = 9,
            .full_extent = {400, 300},
            .framebuffer_extent = {200, 120},
            .crop = {
                .origin = {30, 15},
                .extent = {100, 80}}};

        manager.state().gt_comparison_actual_size_state_.source_key = source_key;
        manager.state().gt_comparison_actual_size_state_.source_generation = 9;
        manager.state().gt_comparison_actual_size_state_.full_extent = {400, 300};
        manager.state().gt_comparison_actual_size_state_.framebuffer_extent = {200, 120};
        manager.state().gt_comparison_actual_size_state_.crop = prepared.crop;
        manager.state().gt_comparison_actual_size_state_.fit_fallback =
            std::make_shared<lfs::core::Tensor>();
        manager.gt_comparison_image_cache_.emplace_back();

        RenderingManager::GTComparisonActualFrame candidate;
        candidate.snapshot = prepared;
        EXPECT_FALSE(manager.isGTComparisonActualSizeActive());
        EXPECT_EQ(manager.getGTComparisonCropOrigin(), glm::ivec2(0, 0));
        EXPECT_TRUE(manager.state().gt_comparison_actual_size_state_.fit_fallback);

        manager.state().gt_comparison_actual_size_state_.error = "previous native failure";
        EXPECT_FALSE(manager.getGTComparisonActualSizeError().empty());
        manager.publishGTComparisonActualFrame(manager.state(), *candidate.snapshot);
        EXPECT_TRUE(manager.getGTComparisonActualSizeError().empty());
        EXPECT_TRUE(manager.isGTComparisonActualSizeActive());
        EXPECT_EQ(manager.getGTComparisonCropOrigin(), prepared.crop.origin);
        EXPECT_FALSE(manager.state().gt_comparison_actual_size_state_.fit_fallback);
        EXPECT_TRUE(manager.gt_comparison_image_cache_.empty());

        const auto bounds = manager.getContentBounds(manager.activeViewId(), {100, 60});
        EXPECT_FLOAT_EQ(bounds.x, 25.0f);
        EXPECT_FLOAT_EQ(bounds.y, 10.0f);
        EXPECT_FLOAT_EQ(bounds.width, 50.0f);
        EXPECT_FLOAT_EQ(bounds.height, 40.0f);

        manager.invalidateGTComparisonActualSizeResources(manager.state());
        EXPECT_TRUE(manager.isGTComparisonActualSizeActive());
        EXPECT_EQ(manager.getGTComparisonCropOrigin(), prepared.crop.origin);

        manager.state().gt_comparison_actual_size_state_.source_key = source_key;
        manager.state().gt_comparison_actual_size_state_.source_generation = 9;
        manager.state().gt_comparison_actual_size_state_.full_extent = prepared.full_extent;
        manager.state().gt_comparison_actual_size_state_.framebuffer_extent =
            prepared.framebuffer_extent;
        manager.state().gt_comparison_actual_size_state_.crop = prepared.crop;
        manager.setGTComparisonCropOrigin({60, 30});
        EXPECT_EQ(
            manager.state().gt_comparison_actual_size_state_.crop.origin,
            glm::ivec2(60, 30));
        EXPECT_EQ(manager.getGTComparisonCropOrigin(), prepared.crop.origin);

        auto replacement = prepared;
        replacement.crop = manager.state().gt_comparison_actual_size_state_.crop;
        manager.publishGTComparisonActualFrame(manager.state(), replacement);
        EXPECT_EQ(manager.getGTComparisonCropOrigin(), glm::ivec2(60, 30));

        // The production presentation path clears the snapshot when a newly
        // assembled Fit fallback replaces the native frame.
        manager.clearPublishedGTComparisonActualFrame(manager.state());
        EXPECT_FALSE(manager.isGTComparisonActualSizeActive());
        EXPECT_EQ(manager.getGTComparisonCropOrigin(), glm::ivec2(0, 0));
    }

    // The worker is stopped at the queue/decode/publication boundary so these
    // tests control completion order without filesystem timing or sleeps.
    class RenderingManagerGTComparisonReviewTest : public ::testing::Test {
    protected:
        screen::ScreenService views;
        RenderingManager manager{views};
        ViewId owner = views.activeView();
        ViewId other = views.screen().split(screen::AreaId{owner}, screen::SplitAxis::Columns, 0.5f).value;
        using Request = RenderingManager::GTComparisonPreviewRequest;
        using Full = RenderingManager::GTComparisonFullSourceRequest;
        using Status = RenderingManager::GTComparisonImageStatus;

        void SetUp() override {
            manager.gt_comparison_image_worker_.request_stop();
            manager.gt_comparison_image_cv_.notify_all();
            manager.gt_comparison_image_worker_.join();
            manager.retainVisibleViews({owner, other});
            manager.editViewSettings(owner, [](auto& s) { s.split_view_mode = SplitViewMode::GTComparison; });
        }
        auto image(glm::ivec2 size = {32, 24}) {
            return std::make_shared<lfs::core::Tensor>(lfs::core::Tensor::empty(
                {3, static_cast<std::size_t>(size.y), static_cast<std::size_t>(size.x)},
                lfs::core::Device::CPU, lfs::core::DataType::UInt8));
        }
        Request request(int uid, glm::ivec2 size = {32, 24}) {
            return {.owner = owner, .camera_uid = uid, .image_path = std::to_string(uid) + ".png", .image_size = size};
        }
        Request beginForeground(Request request) {
            (void)manager.getOrQueueGTComparisonImage(request);
            auto result = *manager.pending_gt_comparison_image_request_;
            manager.pending_gt_comparison_image_request_.reset();
            manager.active_gt_comparison_worker_request_ = result;
            manager.active_gt_comparison_image_is_prefetch_ = false;
            return result;
        }
        Request beginPrefetch(Request request) {
            manager.queueGTComparisonImagePrefetch(request);
            EXPECT_FALSE(manager.prefetch_gt_comparison_image_requests_.empty());
            if (manager.prefetch_gt_comparison_image_requests_.empty())
                return {};
            auto result = manager.prefetch_gt_comparison_image_requests_.back();
            manager.prefetch_gt_comparison_image_requests_.pop_back();
            manager.active_gt_comparison_worker_request_ = result;
            manager.active_gt_comparison_image_is_prefetch_ = true;
            return result;
        }
        void ownership() {
            auto& a = manager.viewState(owner);
            a.gt_comparison_actual_size_state_.crop = {{11, 7}, {32, 24}};
            const auto source = image();
            manager.gt_comparison_full_source_slot_ = RenderingManager::GTComparisonFullSourceSlot{
                .owner = owner,
                .status = Status::Ready,
                .source_key = {.camera_uid = 0},
                .cpu_source = source};
            manager.gt_comparison_cuda_owner_ = owner;
            manager.gt_comparison_cuda_image_ = source;
            manager.invalidateGTComparisonActualSizeResources(manager.viewState(other));
            ASSERT_TRUE(manager.gt_comparison_full_source_slot_);
            EXPECT_EQ(manager.gt_comparison_full_source_slot_->cpu_source, source);
            EXPECT_EQ(manager.gt_comparison_cuda_image_, source);
            EXPECT_EQ(a.gt_comparison_actual_size_state_.crop.origin, glm::ivec2(11, 7));
            manager.invalidateGTComparisonActualSizeTile(manager.viewState(other));
            EXPECT_EQ(manager.gt_comparison_cuda_image_, source);
            manager.dropViewStates();
            EXPECT_FALSE(manager.gt_comparison_full_source_slot_);
            EXPECT_FALSE(manager.gt_comparison_cuda_image_);
        }
        void cropResizeAndPendingPan() {
            using namespace lfs::core;
            if (!gpu_backend_available(GpuBackend::CUDA))
                GTEST_SKIP() << "CUDA device required for native display upload";
            const Camera camera(Tensor::eye(3, Device::CPU), Tensor::zeros({3}, Device::CPU),
                                70.0f, 72.0f, 50.5f, 40.5f, Tensor(), Tensor(), CameraModelType::PINHOLE,
                                "0.png", "0.png", {}, 101, 81, 0);
            manager.gt_comparison_full_source_slot_ = RenderingManager::GTComparisonFullSourceSlot{
                .owner = owner,
                .status = Status::Ready,
                .source_key = {.camera_uid = 0, .image_path = "0.png"},
                .generation = 1,
                .cpu_source = image({101, 81})};
            auto& view = manager.viewState(owner);
            auto frame = manager.prepareGTActualFrame(view, camera, {20, 18});
            ASSERT_TRUE(frame.snapshot);
            manager.publishGTComparisonActualFrame(view, *frame.snapshot);
            manager.setGTComparisonCropOrigin({11, 17}, owner);
            const glm::dvec2 center{21, 26};
            for (int i = 0; i < 20; ++i) {
                const glm::ivec2 extent = i % 2 ? glm::ivec2{20, 18} : glm::ivec2{21, 19};
                frame = manager.prepareGTActualFrame(view, camera, extent);
                ASSERT_TRUE(frame.snapshot);
                manager.publishGTComparisonActualFrame(view, *frame.snapshot);
                EXPECT_EQ(view.gt_comparison_actual_size_state_.desired_crop_center, center);
            }
            EXPECT_EQ(frame.snapshot->crop.origin, glm::ivec2(11, 17));
            // A resize has prepared a new tile but the user still sees the old
            // committed crop. Panning must anchor to that displayed extent.
            frame = manager.prepareGTActualFrame(view, camera, {21, 19});
            ASSERT_TRUE(frame.snapshot);
            manager.setGTComparisonCropOrigin({9, 14}, owner);
            const glm::dvec2 panned_center{19, 23};
            EXPECT_EQ(view.gt_comparison_actual_size_state_.desired_crop_center, panned_center);
            frame = manager.prepareGTActualFrame(view, camera, {21, 19});
            ASSERT_TRUE(frame.snapshot);
            EXPECT_EQ(view.gt_comparison_actual_size_state_.desired_crop_center, panned_center);
            EXPECT_EQ(frame.snapshot->crop, detail::cropGTComparisonFromCenter({101, 81}, {21, 19}, panned_center));
        }
        void sameSourceRedecodeKeepsCrop() {
            using namespace lfs::core;
            if (!gpu_backend_available(GpuBackend::CUDA))
                GTEST_SKIP() << "CUDA device required for native display upload";
            const Camera camera(Tensor::eye(3, Device::CPU), Tensor::zeros({3}, Device::CPU),
                                70.0f, 72.0f, 50.5f, 40.5f, Tensor(), Tensor(), CameraModelType::PINHOLE,
                                "0.png", "0.png", {}, 101, 81, 0);
            const auto ready_slot = [&](const std::uint64_t generation) {
                manager.gt_comparison_full_source_slot_ = RenderingManager::GTComparisonFullSourceSlot{
                    .owner = owner,
                    .status = Status::Ready,
                    .source_key = {.camera_uid = 0, .image_path = "0.png"},
                    .generation = generation,
                    .cpu_source = image({101, 81})};
            };
            ready_slot(1);
            auto& view = manager.viewState(owner);
            auto frame = manager.prepareGTActualFrame(view, camera, {20, 18});
            ASSERT_TRUE(frame.snapshot);
            manager.publishGTComparisonActualFrame(view, *frame.snapshot);
            manager.setGTComparisonCropOrigin({11, 17}, owner);
            const auto panned = view.gt_comparison_actual_size_state_.desired_crop_center;

            // A late decode of the same image must not recentre the user's crop.
            ready_slot(2);
            frame = manager.prepareGTActualFrame(view, camera, {20, 18});
            ASSERT_TRUE(frame.snapshot);
            EXPECT_EQ(view.gt_comparison_actual_size_state_.desired_crop_center, panned);
            EXPECT_EQ(frame.snapshot->crop.origin, glm::ivec2(11, 17));
        }
        void pendingPanWhileLoading() {
            using namespace lfs::core;
            if (!gpu_backend_available(GpuBackend::CUDA))
                GTEST_SKIP() << "CUDA device required for native display upload";
            auto settings = manager.settingsForView(owner);
            settings.gt_comparison_actual_size = true;
            ASSERT_TRUE(manager.updateSettingsForView(
                owner, settings, DirtyFlag::SPLIT_VIEW));
            manager.setCurrentCameraId(0);
            EXPECT_TRUE(manager.isGTComparisonActive());
            EXPECT_TRUE(manager.isGTComparisonActualSizeRequested(owner));
            EXPECT_FALSE(manager.isGTComparisonActualSizeActive(owner));

            manager.setGTComparisonCropOffsetFromCenter({-12, 7}, owner);
            EXPECT_TRUE(manager.isGTComparisonActive());
            EXPECT_TRUE(manager.isGTComparisonActualSizeRequested(owner));
            EXPECT_EQ(manager.getCurrentCameraId(), 0);
            auto& view = manager.viewState(owner);
            ASSERT_EQ(view.gt_comparison_actual_size_state_.pending_pan_camera_uid,
                      std::optional{0});
            EXPECT_EQ(view.gt_comparison_actual_size_state_.pending_pan_offset,
                      glm::ivec2(-12, 7));

            const Camera camera(Tensor::eye(3, Device::CPU), Tensor::zeros({3}, Device::CPU),
                                70.0f, 72.0f, 50.5f, 40.5f, Tensor(), Tensor(), CameraModelType::PINHOLE,
                                "0.png", "0.png", {}, 101, 81, 0);
            manager.gt_comparison_full_source_slot_ = RenderingManager::GTComparisonFullSourceSlot{
                .owner = owner,
                .status = Status::Ready,
                .source_key = {.camera_uid = 0, .image_path = "0.png"},
                .generation = 1,
                .cpu_source = image({101, 81})};
            const auto frame = manager.prepareGTActualFrame(view, camera, {20, 18});
            ASSERT_TRUE(frame.snapshot);
            EXPECT_FALSE(view.gt_comparison_actual_size_state_.pending_pan_camera_uid);
            const glm::dvec2 desired_center = glm::dvec2(101, 81) * 0.5 +
                                              glm::dvec2(-12, 7);
            EXPECT_EQ(view.gt_comparison_actual_size_state_.desired_crop_center,
                      desired_center);
            EXPECT_EQ(frame.snapshot->crop,
                      detail::cropGTComparisonFromCenter(
                          {101, 81}, {20, 18}, desired_center));
        }
        void globalInvalidation() {
            auto& a = manager.viewState(owner);
            const auto source = image();
            a.gt_comparison_actual_size_state_.cpu_source = source;
            a.gt_comparison_actual_size_state_.cuda_source = source;
            a.gt_comparison_actual_size_state_.crop = {{11, 7}, {32, 24}};
            manager.gt_comparison_full_source_slot_ = RenderingManager::GTComparisonFullSourceSlot{
                .owner = owner,
                .status = Status::Ready,
                .cpu_source = source};
            manager.invalidateGTComparisonImageCache(manager.viewState(other));
            EXPECT_FALSE(a.gt_comparison_actual_size_state_.cpu_source);
            EXPECT_FALSE(a.gt_comparison_actual_size_state_.cuda_source);
            EXPECT_FALSE(manager.gt_comparison_full_source_slot_);
            EXPECT_EQ(a.gt_comparison_actual_size_state_.crop.origin, glm::ivec2(0));
        }
        void cameraSwitch() {
            manager.setCurrentCameraId(0);
            (void)manager.getOrQueueGTComparisonFullSource({.owner = owner, .source_key = {.camera_uid = 0, .image_path = "0.png"}});
            const auto old = *manager.pending_gt_comparison_full_source_request_;
            manager.pending_gt_comparison_full_source_request_.reset();
            manager.active_gt_comparison_worker_request_ = old;
            manager.setCurrentCameraId(1);
            EXPECT_EQ(manager.viewState(owner).gt_comparison_camera_uid_, 1);
            EXPECT_EQ(manager.viewState(other).gt_comparison_camera_uid_, -1);
            (void)manager.getOrQueueGTComparisonFullSource({.owner = owner, .source_key = {.camera_uid = 1, .image_path = "1.png"}});
            EXPECT_EQ(manager.completeGTComparisonImage(old, image(), {}, false), kNoView);
            ASSERT_TRUE(manager.gt_comparison_full_source_slot_);
            EXPECT_EQ(manager.gt_comparison_full_source_slot_->status, Status::Loading);
            EXPECT_EQ(manager.gt_comparison_full_source_slot_->source_key.camera_uid, 1);
            const auto current = *manager.pending_gt_comparison_full_source_request_;
            manager.pending_gt_comparison_full_source_request_.reset();
            manager.active_gt_comparison_worker_request_ = current;
            const auto pixels = image();
            EXPECT_EQ(manager.completeGTComparisonImage(current, pixels, {}, false), owner);
            EXPECT_EQ(manager.gt_comparison_full_source_slot_->cpu_source, pixels);
            EXPECT_EQ(manager.gt_comparison_full_source_slot_->source_key.camera_uid, 1);
            auto first = beginForeground(request(0));
            auto second = request(1);
            (void)manager.getOrQueueGTComparisonImage(second);
            EXPECT_EQ(manager.completeGTComparisonImage(first, image(), {}, false), kNoView);
            EXPECT_TRUE(manager.gt_comparison_image_cache_.empty());
        }
        void budget() {
            auto foreground = beginForeground(request(0, {8193, 4321}));
            auto displayed = image(foreground.image_size);
            EXPECT_EQ(manager.completeGTComparisonImage(foreground, displayed, {}, false), owner);
            manager.queueGTComparisonImagePrefetch(request(1, {8193, 4321}));
            EXPECT_TRUE(manager.prefetch_gt_comparison_image_requests_.empty());
            auto speculative = beginPrefetch(request(1, {1024, 1024}));
            manager.queueGTComparisonImagePrefetch(request(2, {2048, 2048}));
            manager.queueGTComparisonImagePrefetch(request(3, {2048, 2048}));
            EXPECT_EQ(manager.prefetch_gt_comparison_image_requests_.size(), 1u);
            // Admission changes while decode is in flight: the newly displayed
            // image alone exceeds the budget. Completion must not evict it.
            auto replacement = request(4, {8193, 8193});
            (void)manager.getOrQueueGTComparisonImage(replacement);
            replacement = *manager.pending_gt_comparison_image_request_;
            auto replacement_image = image(replacement.image_size);
            manager.insertGTComparisonImageCacheEntry(replacement, replacement_image, {}, std::chrono::steady_clock::now());
            EXPECT_EQ(manager.completeGTComparisonImage(speculative, image(speculative.image_size), {}, false), kNoView);
            ASSERT_EQ(manager.gt_comparison_image_cache_.size(), 1u);
            EXPECT_EQ(manager.gt_comparison_image_cache_.front().image, replacement_image);
            EXPECT_FALSE(manager.active_gt_comparison_worker_request_);
        }
        void immutableEpoch(bool promote) {
            auto decoded = beginPrefetch(request(1));
            if (promote) {
                (void)manager.getOrQueueGTComparisonImage(request(1));
                EXPECT_FALSE(manager.active_gt_comparison_image_is_prefetch_);
            }
            manager.invalidateGTComparisonImageCache(manager.viewState(owner));
            (void)manager.getOrQueueGTComparisonImage(request(1));
            ASSERT_TRUE(manager.pending_gt_comparison_image_request_);
            EXPECT_NE(decoded.cache_epoch, manager.pending_gt_comparison_image_request_->cache_epoch);
            EXPECT_EQ(manager.completeGTComparisonImage(decoded, image(), {}, false), kNoView);
            EXPECT_TRUE(manager.gt_comparison_image_cache_.empty());
            EXPECT_TRUE(manager.pending_gt_comparison_image_request_);
        }
        void calibration(bool prefetch = false, bool promote = false) {
            manager.invalidateGTComparisonImageCache(manager.viewState(owner));
            using namespace lfs::core;
            auto camera = std::make_shared<Camera>(Tensor::eye(3, Device::CPU),
                                                   Tensor::zeros({3}, Device::CPU), 24.0f, 23.0f, 16.0f, 12.0f,
                                                   Tensor(), Tensor(), CameraModelType::PINHOLE, "0.png", "0.png", std::filesystem::path{}, 32, 24, 0);
            auto r = request(0);
            r.camera = camera;
            auto old = prefetch ? beginPrefetch(r) : beginForeground(r);
            if (promote)
                (void)manager.getOrQueueGTComparisonImage(r);
            camera->adopt_undistortion({});
            EXPECT_EQ(manager.completeGTComparisonImage(old, image(), {}, false), kNoView);
            EXPECT_TRUE(manager.gt_comparison_image_cache_.empty());
            auto next = beginForeground(r);
            EXPECT_NE(old.calibration_revision, next.calibration_revision);
            EXPECT_EQ(manager.completeGTComparisonImage(next, image(), {}, false), owner);
        }
        void prefetchOnlyAndPromotion() {
            auto speculative = beginPrefetch(request(1));
            EXPECT_EQ(manager.completeGTComparisonImage(speculative, image(), {}, false), kNoView);
            ASSERT_EQ(manager.gt_comparison_image_cache_.size(), 1u);
            auto promoted = beginPrefetch(request(2));
            (void)manager.getOrQueueGTComparisonImage(request(2));
            EXPECT_EQ(manager.completeGTComparisonImage(promoted, image(), {}, false), owner);
            ASSERT_EQ(manager.gt_comparison_image_cache_.size(), 2u);
            auto cancelled = beginPrefetch(request(3));
            EXPECT_EQ(manager.completeGTComparisonImage(cancelled, {}, "cancelled", true), kNoView);
            EXPECT_FALSE(manager.active_gt_comparison_worker_request_);
        }
    };

    TEST_F(RenderingManagerGTComparisonReviewTest, PreparedResizeAndPanPreserveDesiredCenter) { cropResizeAndPendingPan(); }
    TEST_F(RenderingManagerGTComparisonReviewTest, PanWhileNativeSourceLoadsKeepsRequestAndAppliesIntent) { pendingPanWhileLoading(); }
    TEST_F(RenderingManagerGTComparisonReviewTest, SameSourceRedecodeKeepsPannedCrop) { sameSourceRedecodeKeepsCrop(); }
    TEST_F(RenderingManagerGTComparisonReviewTest, UnrelatedViewResetPreservesOwnerAndTeardownReleases) { ownership(); }
    TEST_F(RenderingManagerGTComparisonReviewTest, SameSizedCameraSwitchRejectsPreviousFitAndNativeCompletion) { cameraSwitch(); }
    TEST_F(RenderingManagerGTComparisonReviewTest, EightKAdmissionReservationsAndCompletionProtectDisplayedImage) { budget(); }
    TEST_F(RenderingManagerGTComparisonReviewTest, ObsoletePrefetchCannotBePromotedAcrossInvalidation) { immutableEpoch(false); }
    TEST_F(RenderingManagerGTComparisonReviewTest, PromotedPrefetchRetainsItsDecodeEpoch) { immutableEpoch(true); }
    TEST_F(RenderingManagerGTComparisonReviewTest, CalibrationRevisionRejectsDecodedOldPixels) { calibration(); }
    TEST_F(RenderingManagerGTComparisonReviewTest, CalibrationRevisionRejectsPrefetchIncludingPromotion) {
        calibration(true, false);
        calibration(true, true);
    }
    TEST_F(RenderingManagerGTComparisonReviewTest, GlobalInvalidationFromOtherViewReleasesOwner) { globalInvalidation(); }

    TEST_F(RenderingManagerGTComparisonReviewTest, CacheOnlyCompletionHasNoRedrawConsumer) { prefetchOnlyAndPromotion(); }

    TEST(RenderingManagerActualSizeCacheTest,
         RetainsReadySourceAcrossTogglesAndClearsOnContextChanges) {
        lfs::vis::screen::ScreenService manager_views;
        RenderingManager manager{manager_views};
        manager.gt_comparison_image_worker_.request_stop();
        manager.gt_comparison_image_cv_.notify_all();
        manager.gt_comparison_image_worker_.join();
        auto settings = manager.getSettings();
        settings.split_view_mode = SplitViewMode::GTComparison;
        settings.gt_comparison_actual_size = true;
        manager.updateSettings(settings);
        const detail::GTComparisonSourceKey key{.camera_uid = 17, .image_path = "frame.png"};
        const auto source = std::make_shared<lfs::core::Tensor>(lfs::core::Tensor::empty(
            {3, 8, 12}, lfs::core::Device::CPU, lfs::core::DataType::UInt8));
        const auto seed_source = [&] {
            manager.gt_comparison_full_source_slot_ = RenderingManager::GTComparisonFullSourceSlot{
                .owner = manager.activeViewId(),
                .status = RenderingManager::GTComparisonImageStatus::Ready,
                .source_key = key,
                .generation = manager.gt_comparison_full_source_generation_,
                .cpu_source = source};
        };
        seed_source();
        const auto generation = manager.gt_comparison_full_source_generation_;
        manager.state().gt_comparison_actual_size_state_.cpu_source = source;
        manager.state().gt_comparison_actual_size_state_.visible_tile = source;
        manager.state().gt_comparison_actual_size_state_.error = "old error";
        settings.gt_comparison_actual_size = false;
        manager.updateSettings(settings);
        ASSERT_TRUE(manager.gt_comparison_full_source_slot_);
        EXPECT_EQ(manager.gt_comparison_full_source_slot_->cpu_source, source);
        EXPECT_EQ(manager.gt_comparison_full_source_generation_, generation);
        EXPECT_FALSE(manager.state().gt_comparison_actual_size_state_.cpu_source);
        EXPECT_FALSE(manager.state().gt_comparison_actual_size_state_.visible_tile);
        EXPECT_TRUE(manager.getGTComparisonActualSizeError().empty());
        settings.gt_comparison_actual_size = true;
        manager.updateSettings(settings);
        const auto hit = manager.getOrQueueGTComparisonFullSource({.source_key = key});
        EXPECT_EQ(hit.source, source);
        EXPECT_EQ(hit.generation, generation);
        EXPECT_FALSE(manager.pending_gt_comparison_full_source_request_);

        settings.gt_comparison_actual_size = false;
        manager.updateSettings(settings);
        settings.gt_comparison_mode = GTComparisonMode::Depth;
        manager.updateSettings(settings);
        EXPECT_FALSE(manager.gt_comparison_full_source_slot_);
        settings.gt_comparison_mode = GTComparisonMode::RGB;
        manager.updateSettings(settings);
        seed_source();
        manager.setCurrentCameraId(18);
        EXPECT_FALSE(manager.gt_comparison_full_source_slot_);
        seed_source();
        settings.split_view_mode = SplitViewMode::Disabled;
        manager.updateSettings(settings);
        EXPECT_FALSE(manager.gt_comparison_full_source_slot_);
        seed_source();
        manager.invalidateGTComparisonImageCache(manager.state());
        EXPECT_FALSE(manager.gt_comparison_full_source_slot_);
        seed_source();
        const auto miss = manager.getOrQueueGTComparisonFullSource(
            {.source_key = {.camera_uid = 17, .image_path = "replacement.png"}});
        EXPECT_EQ(miss.status, RenderingManager::GTComparisonImageStatus::Loading);
        EXPECT_FALSE(miss.source);
    }

    TEST(RenderingManagerActualSizeCacheTest,
         RetryAndCancellationPreserveOnlyCurrentWork) {
        lfs::vis::screen::ScreenService manager_views;
        RenderingManager manager{manager_views};
        manager.retainVisibleViews({manager.activeViewId()});
        manager.gt_comparison_image_worker_.request_stop();
        manager.gt_comparison_image_cv_.notify_all();
        manager.gt_comparison_image_worker_.join();
        auto settings = manager.getSettings();
        settings.split_view_mode = SplitViewMode::GTComparison;
        settings.gt_comparison_actual_size = true;
        manager.updateSettings(settings);
        const detail::GTComparisonSourceKey key{.camera_uid = 17, .image_path = "frame.png"};
        manager.gt_comparison_full_source_slot_ = RenderingManager::GTComparisonFullSourceSlot{
            .owner = manager.activeViewId(),
            .status = RenderingManager::GTComparisonImageStatus::Failed,
            .source_key = key,
            .generation = manager.gt_comparison_full_source_generation_,
            .error = "decode failed",
            .failure_time = std::chrono::steady_clock::now()};
        manager.state().gt_comparison_actual_size_state_.error = "decode failed";
        manager.state().dirty_mask_.store(0);
        EXPECT_FALSE(manager.pollDirtyState());
        manager.gt_comparison_full_source_slot_->failure_time -= std::chrono::seconds(3);
        EXPECT_TRUE(manager.pollDirtyState());
        EXPECT_NE(manager.state().dirty_mask_.load() & DirtyFlag::SPLIT_VIEW, 0u);
        manager.gt_comparison_full_source_slot_->failure_time = std::chrono::steady_clock::now();
        EXPECT_EQ(manager.getOrQueueGTComparisonFullSource({.source_key = key}).status,
                  RenderingManager::GTComparisonImageStatus::Failed);
        EXPECT_FALSE(manager.pending_gt_comparison_full_source_request_);
        manager.retryGTComparisonActualSize();
        EXPECT_FALSE(manager.gt_comparison_full_source_slot_);
        EXPECT_TRUE(manager.getGTComparisonActualSizeError().empty());
        const auto pending = manager.getOrQueueGTComparisonFullSource({.source_key = key});
        manager.active_gt_comparison_worker_request_ = *manager.pending_gt_comparison_full_source_request_;
        manager.pending_gt_comparison_full_source_request_.reset();
        manager.retryGTComparisonActualSize();
        const auto same = manager.getOrQueueGTComparisonFullSource({.source_key = key});
        EXPECT_EQ(same.generation, pending.generation);
        EXPECT_TRUE(manager.active_gt_comparison_worker_request_);
        EXPECT_FALSE(manager.pending_gt_comparison_full_source_request_);

        settings.gt_comparison_actual_size = false;
        manager.updateSettings(settings);
        EXPECT_FALSE(manager.gt_comparison_full_source_slot_);
        EXPECT_FALSE(manager.pending_gt_comparison_full_source_request_);
        EXPECT_GT(manager.gt_comparison_full_source_generation_, pending.generation);
        const auto& stale = std::get<RenderingManager::GTComparisonFullSourceRequest>(
            *manager.active_gt_comparison_worker_request_);
        EXPECT_NE(stale.generation, manager.gt_comparison_full_source_generation_);

        settings.gt_comparison_actual_size = true;
        manager.updateSettings(settings);
        const auto source = std::make_shared<lfs::core::Tensor>(lfs::core::Tensor::empty(
            {3, 8, 12}, lfs::core::Device::CPU, lfs::core::DataType::UInt8));
        manager.gt_comparison_full_source_slot_ = RenderingManager::GTComparisonFullSourceSlot{
            .owner = manager.activeViewId(),
            .status = RenderingManager::GTComparisonImageStatus::Ready,
            .source_key = key,
            .generation = manager.gt_comparison_full_source_generation_,
            .cpu_source = source};
        manager.state().gt_comparison_actual_size_state_.tile_failure =
            RenderingManager::GTComparisonActualSizeState::TileFailure{.error = "tile failed"};
        manager.state().gt_comparison_actual_size_state_.error = "tile failed";
        manager.state().dirty_mask_.store(0);
        EXPECT_TRUE(manager.pollDirtyState());
        manager.retryGTComparisonActualSize();
        EXPECT_FALSE(manager.state().gt_comparison_actual_size_state_.tile_failure);
        EXPECT_TRUE(manager.getGTComparisonActualSizeError().empty());
        EXPECT_EQ(manager.getOrQueueGTComparisonFullSource({.source_key = key}).source, source);
        EXPECT_FALSE(manager.pending_gt_comparison_full_source_request_);
    }

    TEST(RenderingManagerGTComparisonGenerationTest,
         SplitLeftGenerationRemainsMonotonicAcrossInvalidations) {
        lfs::vis::screen::ScreenService manager_views;
        RenderingManager manager{manager_views};
        auto source_a = lfs::core::Tensor::empty(
            {3, 4, 6}, lfs::core::Device::CPU, lfs::core::DataType::UInt8);
        auto source_b = lfs::core::Tensor::empty(
            {3, 4, 6}, lfs::core::Device::CPU, lfs::core::DataType::UInt8);
        auto source_c = lfs::core::Tensor::empty(
            {3, 4, 6}, lfs::core::Device::CPU, lfs::core::DataType::UInt8);
        constexpr glm::ivec2 source_size{6, 4};

        manager.updateSplitLeftCpuSourceIdentity(manager.state(),
                                                 &source_a, source_size, 11, false);
        const auto generation_a = manager.state().split_left_image_generation_;
        EXPECT_NE(generation_a, 0u);
        EXPECT_NE(generation_a & RenderingManager::SPLIT_LEFT_GENERATION_BIT, 0u);

        manager.invalidateGTComparisonImageCache(manager.state());
        EXPECT_EQ(manager.state().split_left_image_generation_, generation_a);
        EXPECT_EQ(manager.state().split_left_source_, nullptr);

        manager.updateSplitLeftCpuSourceIdentity(manager.state(),
                                                 &source_b, source_size, 12, false);
        const auto generation_b = manager.state().split_left_image_generation_;
        EXPECT_GT(generation_b, generation_a);

        lfs::vis::ViewportInteropSlotInputs interop{
            .source_ok = true,
            .frame_slot_in_range = true,
            .target_present = true,
            .target_size_matches = true,
            .target_valid_size_matches = true,
            .target_interop_valid = true,
            .target_layout_read_only = true,
            .source_generation = generation_b,
            .uploaded_source_generation = generation_a};
        EXPECT_NE(
            lfs::vis::decideViewportInteropEarly(interop).action,
            lfs::vis::ViewportInteropAction::CacheHit);

        interop.uploaded_source_generation = generation_b;
        EXPECT_EQ(
            lfs::vis::decideViewportInteropEarly(interop).action,
            lfs::vis::ViewportInteropAction::CacheHit);
        manager.updateSplitLeftCpuSourceIdentity(manager.state(),
                                                 &source_b, source_size, 12, false);
        EXPECT_EQ(manager.state().split_left_image_generation_, generation_b);

        manager.invalidateGTComparisonActualSizeResources(manager.state());
        EXPECT_EQ(manager.state().split_left_image_generation_, generation_b);
        EXPECT_EQ(manager.state().split_left_source_, nullptr);

        manager.updateSplitLeftCpuSourceIdentity(manager.state(),
                                                 &source_c, source_size, 13, true);
        EXPECT_GT(manager.state().split_left_image_generation_, generation_b);
    }

    TEST(RenderingManagerActualSizeFailureTest,
         KeyedCooldownRetainsFallbackAndAllowsRelevantChanges) {
        using State = RenderingManager::GTComparisonActualSizeState;
        State state;
        state.fit_fallback = std::make_shared<lfs::core::Tensor>();
        state.visible_tile = std::make_shared<lfs::core::Tensor>();
        state.tile_key = detail::GTComparisonTileKey{};

        const detail::GTComparisonTileKey failed_key{
            .source_generation = 4,
            .full_extent = {8192, 6144},
            .framebuffer_extent = {1920, 1080},
            .crop = {
                .origin = {3136, 2532},
                .extent = {1920, 1080}},
            .distorted = true};
        const auto failed_at =
            std::chrono::steady_clock::time_point(std::chrono::seconds(10));
        state.tile_failure = State::TileFailure{
            .key = failed_key,
            .time = failed_at,
            .error = "tile allocation failed"};

        EXPECT_TRUE(state.tile_failure->suppresses(
            failed_key,
            failed_at + std::chrono::seconds(1),
            RenderingManager::GT_COMPARISON_IMAGE_RETRY_COOLDOWN));
        EXPECT_FALSE(state.tile_failure->suppresses(
            failed_key,
            failed_at + RenderingManager::GT_COMPARISON_IMAGE_RETRY_COOLDOWN,
            RenderingManager::GT_COMPARISON_IMAGE_RETRY_COOLDOWN));

        for (int changed_field = 0; changed_field < 5; ++changed_field) {
            auto changed = failed_key;
            switch (changed_field) {
            case 0:
                ++changed.source_generation;
                break;
            case 1:
                ++changed.full_extent.x;
                break;
            case 2:
                ++changed.framebuffer_extent.y;
                break;
            case 3:
                ++changed.crop.origin.x;
                break;
            case 4:
                changed.distorted = false;
                break;
            }
            EXPECT_FALSE(state.tile_failure->suppresses(
                changed,
                failed_at + std::chrono::milliseconds(1),
                RenderingManager::GT_COMPARISON_IMAGE_RETRY_COOLDOWN));
        }

        state.invalidateTile();
        EXPECT_TRUE(state.fit_fallback);
        EXPECT_TRUE(state.tile_failure);
        EXPECT_FALSE(state.visible_tile);
        EXPECT_FALSE(state.tile_key);

        state.tile_failure.reset();
        EXPECT_FALSE(state.tile_failure);
        EXPECT_TRUE(state.fit_fallback);
    }

    TEST(RenderingManagerActualSizeFailureTest,
         PinholeUploadFailureRetriesAndReusesCache) {
        using lfs::core::Camera;
        using lfs::core::CameraModelType;
        using lfs::core::DataType;
        using lfs::core::Device;
        using lfs::core::MemoryDomain;
        using lfs::core::MemoryPressureCoordinator;
        using lfs::core::Tensor;

        if (!has_cuda_device()) {
            GTEST_SKIP() << "CUDA device required";
        }

        lfs::vis::screen::ScreenService manager_views;
        RenderingManager manager{manager_views};
        manager.gt_comparison_image_worker_.request_stop();
        manager.gt_comparison_image_cv_.notify_all();
        manager.gt_comparison_image_worker_.join();
        auto settings = manager.getSettings();
        settings.split_view_mode = SplitViewMode::GTComparison;
        settings.gt_comparison_mode = GTComparisonMode::RGB;
        settings.gt_comparison_actual_size = true;
        manager.updateSettings(settings);

        Camera camera(
            Tensor::from_vector(
                {1.0f, 0.0f, 0.0f,
                 0.0f, 1.0f, 0.0f,
                 0.0f, 0.0f, 1.0f},
                {3, 3}, Device::CPU),
            Tensor::from_vector({0.0f, 0.0f, 0.0f}, {3}, Device::CPU),
            24.0f, 23.0f, 16.0f, 12.0f,
            Tensor(), Tensor(), CameraModelType::PINHOLE,
            "frame.png", "frame.png", {}, 32, 24, 17);
        const detail::GTComparisonSourceKey source_key{
            .camera_uid = camera.uid(),
            .image_path = camera.image_path()};
        const auto source = std::make_shared<Tensor>(
            Tensor::zeros({3, 24, 32}, Device::CPU, DataType::UInt8));
        const auto fallback = std::make_shared<Tensor>(
            Tensor::zeros({3, 6, 8}, Device::CPU, DataType::UInt8));
        manager.gt_comparison_full_source_slot_ = RenderingManager::GTComparisonFullSourceSlot{
            .owner = manager.activeViewId(),
            .status = RenderingManager::GTComparisonImageStatus::Ready,
            .source_key = source_key,
            .generation = manager.gt_comparison_full_source_generation_,
            .cpu_source = source};
        auto& state = manager.state().gt_comparison_actual_size_state_;
        state.source_key = source_key;
        state.fit_fallback = fallback;

        std::atomic<bool> fail_upload{true};
        std::atomic<std::size_t> allocation_attempts{0};
        struct AllocationProbeReset {
            ~AllocationProbeReset() {
                MemoryPressureCoordinator::instance().set_allocation_probe(nullptr);
            }
        } reset_probe;
        MemoryPressureCoordinator::instance().set_allocation_probe(
            [&](const MemoryDomain domain, std::size_t) {
                if (domain != MemoryDomain::CudaDevice) {
                    return false;
                }
                ++allocation_attempts;
                return fail_upload.load();
            });

        constexpr glm::ivec2 viewport{16, 12};
        const auto failed = manager.prepareGTActualFrame(manager.state(), camera, viewport);
        ASSERT_EQ(failed.status, RenderingManager::GTComparisonImageStatus::Failed);
        EXPECT_GT(allocation_attempts.load(), 0u);
        EXPECT_FALSE(failed.tile);
        EXPECT_FALSE(failed.error.empty());
        EXPECT_EQ(manager.getGTComparisonActualSizeError(), failed.error);
        EXPECT_EQ(failed.fallback, fallback);
        EXPECT_EQ(state.fit_fallback, fallback);
        EXPECT_EQ(state.cpu_source, source);
        EXPECT_FALSE(state.visible_tile);
        EXPECT_FALSE(state.tile_key);
        ASSERT_TRUE(state.tile_failure);
        EXPECT_FALSE(state.tile_failure->key.distorted);

        const auto failed_attempts = allocation_attempts.load();
        const auto suppressed = manager.prepareGTActualFrame(manager.state(), camera, viewport);
        EXPECT_EQ(suppressed.status, RenderingManager::GTComparisonImageStatus::Failed);
        EXPECT_EQ(suppressed.error, failed.error);
        EXPECT_EQ(suppressed.fallback, fallback);
        EXPECT_EQ(allocation_attempts.load(), failed_attempts);

        fail_upload = false;
        manager.retryGTComparisonActualSize();
        EXPECT_FALSE(state.tile_failure);
        EXPECT_TRUE(manager.getGTComparisonActualSizeError().empty());
        ASSERT_TRUE(manager.gt_comparison_full_source_slot_);
        EXPECT_EQ(manager.gt_comparison_full_source_slot_->cpu_source, source);
        const auto ready = manager.prepareGTActualFrame(manager.state(), camera, viewport);
        ASSERT_EQ(ready.status, RenderingManager::GTComparisonImageStatus::Ready);
        ASSERT_TRUE(ready.tile && ready.tile->is_valid());
        EXPECT_EQ(ready.tile->device(), Device::CUDA);
        EXPECT_EQ(ready.tile, manager.gt_comparison_cuda_image_);
        ASSERT_TRUE(state.visible_tile);
        EXPECT_EQ(state.visible_tile->device(), Device::CPU);
        EXPECT_EQ(manager.gt_comparison_cuda_source_, state.visible_tile.get());
        EXPECT_EQ(state.fit_fallback, fallback);
        EXPECT_FALSE(manager.isGTComparisonActualSizeActive());

        const auto cpu_tile = state.visible_tile;
        const auto generation = manager.state().split_left_image_generation_;
        EXPECT_NE(generation & RenderingManager::SPLIT_LEFT_GENERATION_BIT, 0u);
        const auto ready_attempts = allocation_attempts.load();
        fail_upload = true;
        const auto cached = manager.prepareGTActualFrame(manager.state(), camera, viewport);
        EXPECT_EQ(cached.status, RenderingManager::GTComparisonImageStatus::Ready);
        EXPECT_EQ(cached.tile, ready.tile);
        EXPECT_EQ(state.visible_tile, cpu_tile);
        EXPECT_EQ(manager.state().split_left_image_generation_, generation);
        EXPECT_EQ(allocation_attempts.load(), ready_attempts);

        ASSERT_TRUE(ready.snapshot);
        manager.publishGTComparisonActualFrame(manager.state(), *ready.snapshot);
        EXPECT_TRUE(manager.isGTComparisonActualSizeActive());
        EXPECT_FALSE(state.fit_fallback);
        EXPECT_TRUE(manager.getGTComparisonActualSizeError().empty());
    }

    TEST(SplitViewServiceTest, ActualSizeCaptureUsesExactTexelsFlipAndLetterbox) {
        constexpr int panel_width = 8;
        constexpr int panel_height = 6;
        constexpr std::size_t panel_pixels = panel_width * panel_height;
        std::vector<float> left_values(3 * panel_pixels);
        std::vector<float> right_values(3 * panel_pixels);
        for (int channel = 0; channel < 3; ++channel) {
            for (int y = 0; y < panel_height; ++y) {
                for (int x = 0; x < panel_width; ++x) {
                    const auto index = static_cast<std::size_t>(channel) * panel_pixels +
                                       static_cast<std::size_t>(y) * panel_width + x;
                    left_values[index] = 0.1f + channel * 0.2f + y * 0.01f + x * 0.001f;
                    right_values[index] = 0.5f + channel * 0.1f + y * 0.01f + x * 0.001f;
                }
            }
        }
        auto left = std::make_shared<lfs::core::Tensor>(lfs::core::Tensor::from_vector(
            left_values, {3, panel_height, panel_width}, lfs::core::Device::CPU));
        auto right = std::make_shared<lfs::core::Tensor>(lfs::core::Tensor::from_vector(
            right_values, {3, panel_height, panel_width}, lfs::core::Device::CPU));

        VulkanSplitViewParams params{
            .enabled = true,
            .left = {.image = left},
            .right = {.image = right, .flip_y = true},
            .split_position = 0.5f,
            .content_rect = {2, 1, panel_width, panel_height},
            .coordinate_extent = {12, 8},
            .background = {0.01f, 0.02f, 0.03f},
            .exact_texel_sampling = true};
        const auto output = composeSplitCaptureCpu(params, {12, 8});
        ASSERT_TRUE(output && output->is_valid());
        const SplitViewCpuDesc shared_params{
            .exact_texel_sampling = true,
            .left = {.image = left},
            .right = {.image = right, .flip_y = true},
            .split_position = params.split_position,
            .content_rect = params.content_rect,
            .coordinate_extent = params.coordinate_extent,
            .background = params.background};
        const auto shared_output = composeSplitViewCpuImage(shared_params, {12, 8});
        ASSERT_TRUE(shared_output && shared_output->is_valid());
        ASSERT_EQ(shared_output->numel(), output->numel());
        // Upstream captures publish upright HWC; the legacy test compositor is CHW.
        const auto shared_chw = shared_output->permute({2, 0, 1}).contiguous();
        for (std::size_t i = 0; i < output->numel(); ++i)
            EXPECT_FLOAT_EQ(shared_chw.ptr<float>()[i], output->ptr<float>()[i]) << i;
        const float* data = output->ptr<float>();
        constexpr std::size_t output_pixels = 12 * 8;
        const auto at = [&](const int channel, const int x, const int y) {
            return data[static_cast<std::size_t>(channel) * output_pixels +
                        static_cast<std::size_t>(y) * 12 + x];
        };

        EXPECT_FLOAT_EQ(at(0, 0, 0), 0.01f);
        EXPECT_FLOAT_EQ(at(1, 0, 0), 0.02f);
        EXPECT_FLOAT_EQ(at(0, 2, 1), left_values[0]);
        EXPECT_FLOAT_EQ(at(0, 3, 6), left_values[5 * panel_width + 1]);
        EXPECT_FLOAT_EQ(at(0, 8, 1), right_values[5 * panel_width + 6]);
        EXPECT_FLOAT_EQ(at(2, 9, 6), right_values[2 * panel_pixels + 7]);
        EXPECT_NEAR(at(0, 5, 1), 0.29f * 0.8f, 1.0e-6f);
    }

    TEST(SplitViewServiceTest, FitCaptureRetainsLinearSampling) {
        const std::vector<float> values{
            0.0f, 1.0f, 2.0f, 3.0f,
            0.0f, 1.0f, 2.0f, 3.0f,
            0.0f, 1.0f, 2.0f, 3.0f};
        auto image = std::make_shared<lfs::core::Tensor>(lfs::core::Tensor::from_vector(
            values, {3, 2, 2}, lfs::core::Device::CPU));
        VulkanSplitViewParams params{
            .enabled = true,
            .left = {.image = image},
            .right = {.image = image},
            .split_position = 0.75f,
            .content_rect = {0, 0, 10, 6},
            .coordinate_extent = {10, 6},
            .exact_texel_sampling = false};
        const auto output = composeSplitCaptureCpu(params, {10, 6});
        ASSERT_TRUE(output && output->is_valid());
        const float* data = output->ptr<float>();
        // Pixel-center UVs map (1, 1) to source coordinates (-0.2, 0),
        // where linear filtering clamps to the first texel.
        EXPECT_FLOAT_EQ(data[1 * 10 + 1], 0.0f);

        // Away from the edge and divider, (3, 2) maps to (0.2, 1/3).
        // Bilinear interpolation of {0, 1; 2, 3} gives 0.2 + 2/3 = 13/15,
        // rather than the first texel that nearest sampling would select.
        EXPECT_NEAR(data[2 * 10 + 3], 13.0f / 15.0f, 1.0e-6f);
    }

    TEST(CameraImageLoadTest, PreviewLoadsCanAvoidMutatingCameraImageDimensions) {
        using lfs::core::Camera;
        using lfs::core::CameraModelType;
        using lfs::core::DataType;
        using lfs::core::Device;
        using lfs::core::Tensor;

        ensureCameraImageLoader();

        const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
        const auto image_path = std::filesystem::temp_directory_path() /
                                ("lfs_camera_preview_" + std::to_string(now) + ".png");
        auto image = Tensor::zeros({size_t{6}, size_t{8}, size_t{3}}, Device::CPU, DataType::UInt8);
        ASSERT_NO_THROW(lfs::core::save_image(image_path, image));

        Camera camera(
            Tensor::from_vector(
                {1.0f, 0.0f, 0.0f,
                 0.0f, 1.0f, 0.0f,
                 0.0f, 0.0f, 1.0f},
                {size_t{3}, size_t{3}},
                Device::CPU),
            Tensor::from_vector({0.0f, 0.0f, 0.0f}, {size_t{3}}, Device::CPU),
            500.0f,
            500.0f,
            4.0f,
            3.0f,
            Tensor(),
            Tensor(),
            CameraModelType::PINHOLE,
            "preview.png",
            image_path,
            {},
            8,
            6,
            42);

        auto preview = camera.load_and_get_image(-1, 4, false, false);
        ASSERT_TRUE(preview.is_valid());
        ASSERT_EQ(preview.ndim(), 3);
        EXPECT_EQ(static_cast<int>(preview.shape()[1]), 3);
        EXPECT_EQ(static_cast<int>(preview.shape()[2]), 4);
        EXPECT_EQ(camera.image_width(), 8);
        EXPECT_EQ(camera.image_height(), 6);

        auto published = camera.load_and_get_image(-1, 4, false, true);
        ASSERT_TRUE(published.is_valid());
        EXPECT_EQ(camera.image_width(), 4);
        EXPECT_EQ(camera.image_height(), 3);

        std::filesystem::remove(image_path);
    }

    TEST(CameraImageLoadTest, NativeResolutionRgb8LoaderPreservesEightBitBytesAndChannelPolicy) {
        using lfs::core::DataType;
        using lfs::core::Device;
        using lfs::core::TensorShape;

        for (const auto [width, height] : {
                 std::pair{1, 1}, std::pair{7, 1}, std::pair{1, 7},
                 std::pair{5, 2}, std::pair{7, 5}}) {
            SCOPED_TRACE(std::format("{}x{}", width, height));
            const std::size_t pixel_count = static_cast<std::size_t>(width) * height;
            const auto stamp = std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count());
            const auto root = std::filesystem::temp_directory_path() /
                              ("lfs_gt_rgb8_native_" + stamp);
            std::filesystem::create_directories(root);

            std::vector<std::uint8_t> gray(pixel_count);
            std::vector<std::uint8_t> two_channel(pixel_count * 2);
            std::vector<std::uint8_t> rgb(pixel_count * 3);
            std::vector<std::uint8_t> rgba(pixel_count * 4);
            for (std::size_t index = 0; index < pixel_count; ++index) {
                gray[index] = static_cast<std::uint8_t>(11 + index);
                two_channel[index * 2] = static_cast<std::uint8_t>(21 + index);
                two_channel[index * 2 + 1] = static_cast<std::uint8_t>(101 + index);
                rgb[index * 3] = static_cast<std::uint8_t>(31 + index);
                rgb[index * 3 + 1] = static_cast<std::uint8_t>(111 + index);
                rgb[index * 3 + 2] = static_cast<std::uint8_t>(211 - index);
                rgba[index * 4] = static_cast<std::uint8_t>(41 + index);
                rgba[index * 4 + 1] = static_cast<std::uint8_t>(121 + index);
                rgba[index * 4 + 2] = static_cast<std::uint8_t>(221 - index);
                rgba[index * 4 + 3] = static_cast<std::uint8_t>(7 + index);
            }

            const auto gray_path = root / "gray.png";
            const auto two_path = root / "two.png";
            const auto rgb_path = root / "rgb.png";
            const auto rgba_path = root / "rgba.png";
            const auto jpeg_path = root / "rgb.jpg";
            ASSERT_NO_THROW(writeU8Image(gray_path, width, height, 1, gray));
            ASSERT_NO_THROW(writeU8Image(two_path, width, height, 2, two_channel));
            ASSERT_NO_THROW(writeU8Image(rgb_path, width, height, 3, rgb));
            ASSERT_NO_THROW(writeU8Image(rgba_path, width, height, 4, rgba));
            ASSERT_NO_THROW(writeU8Image(jpeg_path, width, height, 3, rgb));

            const auto loaded_rgb = lfs::core::load_image_rgb8_chw_native_resolution(rgb_path);
            const auto repeated_rgb = lfs::core::load_image_rgb8_chw_native_resolution(rgb_path);
            ASSERT_TRUE(loaded_rgb.is_valid());
            EXPECT_EQ(loaded_rgb.device(), Device::CPU);
            EXPECT_EQ(loaded_rgb.dtype(), DataType::UInt8);
            EXPECT_EQ(loaded_rgb.shape(), TensorShape({size_t{3}, static_cast<size_t>(height), static_cast<size_t>(width)}));
            EXPECT_TRUE(loaded_rgb.is_contiguous());
            EXPECT_TRUE(loaded_rgb.owns_memory());
            EXPECT_FALSE(loaded_rgb.is_view());
            EXPECT_EQ(loaded_rgb.stream(), nullptr);
            EXPECT_EQ(loaded_rgb.to_vector_uint8(), repeated_rgb.to_vector_uint8());

            const auto expect_channels = [&](const std::filesystem::path& path,
                                             const std::vector<std::uint8_t>& expected_r,
                                             const std::vector<std::uint8_t>& expected_g,
                                             const std::vector<std::uint8_t>& expected_b) {
                const auto loaded = lfs::core::load_image_rgb8_chw_native_resolution(path);
                const auto values = loaded.to_vector_uint8();
                ASSERT_EQ(values.size(), 3 * pixel_count);
                for (std::size_t index = 0; index < pixel_count; ++index) {
                    EXPECT_EQ(values[index], expected_r[index]);
                    EXPECT_EQ(values[pixel_count + index], expected_g[index]);
                    EXPECT_EQ(values[2 * pixel_count + index], expected_b[index]);
                }
            };

            expect_channels(gray_path, gray, gray, gray);
            std::vector<std::uint8_t> two_r(pixel_count);
            std::vector<std::uint8_t> two_g(pixel_count);
            std::vector<std::uint8_t> two_b(pixel_count);
            std::vector<std::uint8_t> rgb_r(pixel_count);
            std::vector<std::uint8_t> rgb_g(pixel_count);
            std::vector<std::uint8_t> rgb_b(pixel_count);
            std::vector<std::uint8_t> rgba_r(pixel_count);
            std::vector<std::uint8_t> rgba_g(pixel_count);
            std::vector<std::uint8_t> rgba_b(pixel_count);
            for (std::size_t index = 0; index < pixel_count; ++index) {
                two_r[index] = two_channel[index * 2];
                two_g[index] = two_channel[index * 2 + 1];
                two_b[index] = static_cast<std::uint8_t>(
                    (static_cast<std::uint16_t>(two_r[index]) + two_g[index]) / 2u);
                rgb_r[index] = rgb[index * 3];
                rgb_g[index] = rgb[index * 3 + 1];
                rgb_b[index] = rgb[index * 3 + 2];
                rgba_r[index] = rgba[index * 4];
                rgba_g[index] = rgba[index * 4 + 1];
                rgba_b[index] = rgba[index * 4 + 2];
            }
            expect_channels(two_path, two_r, two_g, two_b);
            expect_channels(rgb_path, rgb_r, rgb_g, rgb_b);
            expect_channels(rgba_path, rgba_r, rgba_g, rgba_b);

            const auto jpeg_first =
                lfs::core::load_image_rgb8_chw_native_resolution(jpeg_path).to_vector_uint8();
            const auto jpeg_second =
                lfs::core::load_image_rgb8_chw_native_resolution(jpeg_path).to_vector_uint8();
            EXPECT_EQ(jpeg_first, jpeg_second);
            EXPECT_THROW(
                (void)lfs::core::load_image_rgb8_chw_native_resolution(root / "missing.png"),
                std::runtime_error);

            std::filesystem::remove_all(root);
        }
    }

    TEST(SplitViewServiceTest, SharedCameraPoseHelperNormalizesSceneRotationAndAppliesVisualizerAxes) {
        const glm::mat3 world_to_camera = glm::mat3(1.0f);
        const glm::vec3 world_to_camera_translation(0.0f, 0.0f, 0.0f);

        glm::mat4 scene_transform(1.0f);
        scene_transform = glm::translate(scene_transform, glm::vec3(1.0f, 2.0f, 3.0f));
        scene_transform = glm::scale(scene_transform, glm::vec3(2.0f, 3.0f, 4.0f));

        const auto pose = lfs::rendering::visualizerCameraPoseFromDataWorldToCamera(
            world_to_camera,
            world_to_camera_translation,
            scene_transform);

        expectMat3Near(pose.rotation, lfs::rendering::DATA_TO_VISUALIZER_CAMERA_AXES);
        EXPECT_EQ(pose.translation, glm::vec3(1.0f, 2.0f, 3.0f));
    }

    TEST_F(RenderingManagerEventsTest, OrthographicEnterSetsScaleFromCurrentFocal) {
        lfs::vis::screen::ScreenService manager_views;
        RenderingManager manager{manager_views};
        auto settings = manager.getSettings();
        settings.focal_length_mm = 50.0f;
        manager.updateSettings(settings);

        constexpr float viewport_height = 900.0f;
        constexpr float distance_to_pivot = 7.5f;

        manager.setOrthographic(true, viewport_height, distance_to_pivot);
        const auto ortho_settings = manager.getSettings();
        ASSERT_TRUE(ortho_settings.orthographic);

        const float expected_scale = viewport_height /
                                     (2.0f * distance_to_pivot *
                                      std::tan(glm::radians(lfs::rendering::focalLengthToVFov(50.0f)) * 0.5f));
        EXPECT_NEAR(ortho_settings.ortho_scale, expected_scale, 1e-4f);
    }

    TEST_F(RenderingManagerEventsTest, OrthographicLeaveKeepsFocalLength) {
        lfs::vis::screen::ScreenService manager_views;
        RenderingManager manager{manager_views};
        auto settings = manager.getSettings();
        settings.focal_length_mm = 35.0f;
        manager.updateSettings(settings);

        constexpr float viewport_height = 900.0f;
        constexpr float distance_to_pivot = 7.5f;

        manager.setOrthographic(true, viewport_height, distance_to_pivot);
        manager.setOrthographic(false, viewport_height, distance_to_pivot);
        const auto after_round_trip = manager.getSettings();
        ASSERT_FALSE(after_round_trip.orthographic);
        EXPECT_FLOAT_EQ(after_round_trip.focal_length_mm, 35.0f);

        manager.setOrthographic(true, viewport_height, distance_to_pivot);
        settings = manager.getSettings();
        ASSERT_TRUE(settings.orthographic);
        settings.ortho_scale *= std::pow(1.1f, 20.0f);
        manager.updateSettings(settings);

        manager.setOrthographic(false, viewport_height, distance_to_pivot);
        const auto after_zoom = manager.getSettings();
        ASSERT_FALSE(after_zoom.orthographic);
        EXPECT_FLOAT_EQ(after_zoom.focal_length_mm, 35.0f);
    }

    TEST(SplitViewServiceTest, CurrentSceneTransformUsesIdentityForMultipleVisiblePointClouds) {
        SceneManager manager;
        auto& scene = manager.getScene();

        const auto left_parent = scene.addGroup("LeftParent");
        const auto right_parent = scene.addGroup("RightParent");

        scene.setNodeTransform(
            "LeftParent",
            glm::translate(glm::mat4(1.0f), glm::vec3(1.0f, 2.0f, 3.0f)));
        scene.setNodeTransform(
            "RightParent",
            glm::translate(glm::mat4(1.0f), glm::vec3(-4.0f, 5.0f, 6.0f)));

        scene.addPointCloud("LeftCloud", makeTestPointCloud(), left_parent);
        scene.addPointCloud("RightCloud", makeTestPointCloud(), right_parent);

        EXPECT_EQ(
            detail::currentSceneTransform(&manager, -1),
            lfs::rendering::dataWorldTransformToVisualizerWorld(glm::mat4(1.0f)));
    }

    TEST_F(SceneManagerRenderStateTest, DatasetReadyStateKeepsVisiblePointCloudWhenTrainingModelIsEmpty) {
        SceneManager manager;
        manager.changeContentType(SceneManager::ContentType::Dataset);

        auto& scene = manager.getScene();
        const auto dataset_id = scene.addGroup("Dataset");

        auto means_empty = lfs::core::Tensor::zeros({size_t{0}, size_t{3}}, lfs::core::Device::CPU, lfs::core::DataType::Float32);
        auto sh0_empty = lfs::core::Tensor::zeros({size_t{0}, size_t{1}, size_t{3}}, lfs::core::Device::CPU, lfs::core::DataType::Float32);
        auto shN_empty = lfs::core::Tensor::zeros({size_t{0}, size_t{3}, size_t{3}}, lfs::core::Device::CPU, lfs::core::DataType::Float32);
        auto scaling_empty = lfs::core::Tensor::zeros({size_t{0}, size_t{3}}, lfs::core::Device::CPU, lfs::core::DataType::Float32);
        auto rotation_empty = lfs::core::Tensor::zeros({size_t{0}, size_t{4}}, lfs::core::Device::CPU, lfs::core::DataType::Float32);
        auto opacity_empty = lfs::core::Tensor::zeros({size_t{0}, size_t{1}}, lfs::core::Device::CPU, lfs::core::DataType::Float32);
        scene.addSplat(
            "Model",
            std::make_unique<lfs::core::SplatData>(
                1,
                std::move(means_empty),
                std::move(sh0_empty),
                std::move(shN_empty),
                std::move(scaling_empty),
                std::move(rotation_empty),
                std::move(opacity_empty),
                1.0f),
            dataset_id);
        scene.setTrainingModelNode("Model");

        auto means = lfs::core::Tensor::from_vector({0.0f, 0.0f, 0.0f}, {size_t{1}, size_t{3}}, lfs::core::Device::CPU);
        auto colors = lfs::core::Tensor::from_vector({1.0f, 0.0f, 0.0f}, {size_t{1}, size_t{3}}, lfs::core::Device::CPU);
        scene.addPointCloud("PointCloud", std::make_shared<lfs::core::PointCloud>(std::move(means), std::move(colors)), dataset_id);

        const auto state = manager.buildRenderState();
        ASSERT_NE(state.combined_model, nullptr);
        EXPECT_TRUE(state.combined_model->means_raw().is_valid());
        EXPECT_EQ(state.combined_model->size(), 0u);
        ASSERT_NE(state.point_cloud, nullptr);
        EXPECT_EQ(state.point_cloud->size(), 1);
        EXPECT_EQ(state.point_cloud_transform,
                  lfs::rendering::dataWorldTransformToVisualizerWorld(glm::mat4(1.0f)));
    }

    TEST_F(SceneManagerRenderStateTest, HiddenDatasetTrainingModelStaysResidentAndIsCulledByMask) {
        SceneManager manager;
        manager.changeContentType(SceneManager::ContentType::Dataset);

        auto& scene = manager.getScene();
        scene.addSplat("Model", makeTestSplat(0.0f));
        scene.setTrainingModelNode("Model");

        scene.setNodeVisibility("Model", false);

        const auto state = manager.buildRenderState();
        ASSERT_NE(state.combined_model, nullptr);
        EXPECT_EQ(state.combined_model->size(), 1u);
        EXPECT_EQ(state.visible_splat_count, 0u);
        ASSERT_EQ(state.node_visibility_mask.size(), 1u);
        EXPECT_FALSE(state.node_visibility_mask[0]);
        EXPECT_EQ(manager.getModelForRendering(), state.combined_model);
    }

    TEST_F(SceneManagerRenderStateTest, VisibleSelectionMaskIsCachedForUnchangedGenerations) {
        SceneManager manager;
        auto& scene = manager.getScene();

        scene.addSplat("Visible", makeTwoPointTestSplat(0.0f, 1.0f));
        scene.addSplat("Hidden", makeTestSplat(2.0f));
        scene.setNodeVisibility("Hidden", false);
        scene.setSelection({0});

        const auto first = scene.getVisibleSelectionMask();
        const auto second = scene.getVisibleSelectionMask();

        ASSERT_NE(first, nullptr);
        ASSERT_NE(second, nullptr);
        EXPECT_EQ(first.get(), second.get());
    }

    TEST_F(SceneManagerRenderStateTest, PointCloudTransformIsTrackedSeparatelyFromModelTransforms) {
        SceneManager manager;
        auto& scene = manager.getScene();

        scene.addPointCloud("PointCloud", makeTestPointCloud());
        scene.setNodeTransform(
            "PointCloud",
            glm::translate(glm::mat4(1.0f), glm::vec3(3.0f, -2.0f, 5.0f)));

        const auto state = manager.buildRenderState();
        ASSERT_NE(state.point_cloud, nullptr);
        EXPECT_TRUE(state.model_transforms.empty());
        expectVisualizerTranslationFromData(state.point_cloud_transform, {3.0f, -2.0f, 5.0f});
    }

    TEST_F(SceneManagerRenderStateTest, VisiblePointCloudDoesNotPolluteModelTransformArray) {
        SceneManager manager;
        auto& scene = manager.getScene();

        scene.addPointCloud("PointCloud", makeTestPointCloud());
        scene.setNodeTransform(
            "PointCloud",
            glm::translate(glm::mat4(1.0f), glm::vec3(9.0f, 8.0f, 7.0f)));
        scene.addSplat("Model", makeTestSplat(0.0f));
        scene.setNodeTransform(
            "Model",
            glm::translate(glm::mat4(1.0f), glm::vec3(1.0f, 2.0f, 3.0f)));

        const auto state = manager.buildRenderState();
        ASSERT_EQ(state.model_transforms.size(), 1u);
        expectVisualizerTranslationFromData(state.model_transforms[0], {1.0f, 2.0f, 3.0f});
    }

    TEST_F(SceneManagerRenderStateTest, MultipleVisiblePointCloudsAreMergedAcrossParentTransforms) {
        SceneManager manager;
        auto& scene = manager.getScene();

        const auto left_parent = scene.addGroup("LeftParent");
        const auto right_parent = scene.addGroup("RightParent");

        scene.setNodeTransform(
            "LeftParent",
            glm::translate(glm::mat4(1.0f), glm::vec3(1.0f, 2.0f, 3.0f)));
        scene.setNodeTransform(
            "RightParent",
            glm::translate(glm::mat4(1.0f), glm::vec3(-4.0f, 5.0f, 6.0f)));

        auto left_means = lfs::core::Tensor::from_vector(
            {0.0f, 0.0f, 0.0f,
             1.0f, 0.0f, 0.0f},
            {size_t{2}, size_t{3}},
            lfs::core::Device::CPU);
        auto left_colors = lfs::core::Tensor::from_vector(
            {1.0f, 0.0f, 0.0f,
             0.0f, 1.0f, 0.0f},
            {size_t{2}, size_t{3}},
            lfs::core::Device::CPU);
        scene.addPointCloud(
            "LeftCloud",
            std::make_shared<lfs::core::PointCloud>(std::move(left_means), std::move(left_colors)),
            left_parent);

        auto right_means = lfs::core::Tensor::from_vector(
            {0.0f, 1.0f, 0.0f},
            {size_t{1}, size_t{3}},
            lfs::core::Device::CPU);
        auto right_colors = lfs::core::Tensor::from_vector(
            {0.0f, 0.0f, 1.0f},
            {size_t{1}, size_t{3}},
            lfs::core::Device::CPU);
        scene.addPointCloud(
            "RightCloud",
            std::make_shared<lfs::core::PointCloud>(std::move(right_means), std::move(right_colors)),
            right_parent);

        const auto state = manager.buildRenderState();
        ASSERT_NE(state.point_cloud, nullptr);
        EXPECT_EQ(state.point_cloud->size(), 3);
        EXPECT_TRUE(state.model_transforms.empty());
        EXPECT_EQ(state.point_cloud_transform,
                  lfs::rendering::dataWorldTransformToVisualizerWorld(glm::mat4(1.0f)));

        auto means_cpu = state.point_cloud->means.cpu();
        auto acc = means_cpu.accessor<float, 2>();
        EXPECT_FLOAT_EQ(acc(0, 0), 1.0f);
        EXPECT_FLOAT_EQ(acc(0, 1), 2.0f);
        EXPECT_FLOAT_EQ(acc(0, 2), 3.0f);
        EXPECT_FLOAT_EQ(acc(1, 0), 2.0f);
        EXPECT_FLOAT_EQ(acc(1, 1), 2.0f);
        EXPECT_FLOAT_EQ(acc(1, 2), 3.0f);
        EXPECT_FLOAT_EQ(acc(2, 0), -4.0f);
        EXPECT_FLOAT_EQ(acc(2, 1), 6.0f);
        EXPECT_FLOAT_EQ(acc(2, 2), 6.0f);
    }

    TEST_F(SceneManagerRenderStateTest, MultipleVisiblePointCloudMergeRefreshesWhenSourceDataChanges) {
        SceneManager manager;
        auto& scene = manager.getScene();

        const auto left_parent = scene.addGroup("LeftParent");
        const auto right_parent = scene.addGroup("RightParent");

        auto left_point_cloud = std::make_shared<lfs::core::PointCloud>(
            lfs::core::Tensor::from_vector(
                {0.0f, 0.0f, 0.0f},
                {size_t{1}, size_t{3}},
                lfs::core::Device::CPU),
            lfs::core::Tensor::from_vector(
                {1.0f, 0.0f, 0.0f},
                {size_t{1}, size_t{3}},
                lfs::core::Device::CPU));
        auto right_point_cloud = std::make_shared<lfs::core::PointCloud>(
            lfs::core::Tensor::from_vector(
                {1.0f, 1.0f, 1.0f},
                {size_t{1}, size_t{3}},
                lfs::core::Device::CPU),
            lfs::core::Tensor::from_vector(
                {0.0f, 1.0f, 0.0f},
                {size_t{1}, size_t{3}},
                lfs::core::Device::CPU));

        scene.addPointCloud("LeftCloud", left_point_cloud, left_parent);
        scene.addPointCloud("RightCloud", right_point_cloud, right_parent);

        const auto initial_state = manager.buildRenderState();
        ASSERT_NE(initial_state.point_cloud, nullptr);
        ASSERT_EQ(initial_state.point_cloud->size(), 2);

        right_point_cloud->means = lfs::core::Tensor::from_vector(
            {10.0f, 20.0f, 30.0f},
            {size_t{1}, size_t{3}},
            lfs::core::Device::CPU);

        const auto updated_state = manager.buildRenderState();
        ASSERT_NE(updated_state.point_cloud, nullptr);
        ASSERT_EQ(updated_state.point_cloud->size(), 2);

        auto means_cpu = updated_state.point_cloud->means.cpu();
        auto acc = means_cpu.accessor<float, 2>();
        EXPECT_FLOAT_EQ(acc(0, 0), 0.0f);
        EXPECT_FLOAT_EQ(acc(0, 1), 0.0f);
        EXPECT_FLOAT_EQ(acc(0, 2), 0.0f);
        EXPECT_FLOAT_EQ(acc(1, 0), 10.0f);
        EXPECT_FLOAT_EQ(acc(1, 1), 20.0f);
        EXPECT_FLOAT_EQ(acc(1, 2), 30.0f);
    }

    TEST_F(SceneManagerRenderStateTest, MultipleVisiblePointCloudMergeRefreshesWhenSourceTensorChangesInPlace) {
        SceneManager manager;
        auto& scene = manager.getScene();

        const auto left_parent = scene.addGroup("LeftParent");
        const auto right_parent = scene.addGroup("RightParent");

        auto left_point_cloud = std::make_shared<lfs::core::PointCloud>(
            lfs::core::Tensor::from_vector(
                {0.0f, 0.0f, 0.0f},
                {size_t{1}, size_t{3}},
                lfs::core::Device::CPU),
            lfs::core::Tensor::from_vector(
                {1.0f, 0.0f, 0.0f},
                {size_t{1}, size_t{3}},
                lfs::core::Device::CPU));
        auto right_point_cloud = std::make_shared<lfs::core::PointCloud>(
            lfs::core::Tensor::from_vector(
                {1.0f, 1.0f, 1.0f},
                {size_t{1}, size_t{3}},
                lfs::core::Device::CPU),
            lfs::core::Tensor::from_vector(
                {0.0f, 1.0f, 0.0f},
                {size_t{1}, size_t{3}},
                lfs::core::Device::CPU));

        scene.addPointCloud("LeftCloud", left_point_cloud, left_parent);
        scene.addPointCloud("RightCloud", right_point_cloud, right_parent);

        const auto initial_state = manager.buildRenderState();
        ASSERT_NE(initial_state.point_cloud, nullptr);
        ASSERT_EQ(initial_state.point_cloud->size(), 2);

        right_point_cloud->means.copy_(lfs::core::Tensor::from_vector(
            {7.0f, 8.0f, 9.0f},
            {size_t{1}, size_t{3}},
            lfs::core::Device::CPU));

        const auto updated_state = manager.buildRenderState();
        ASSERT_NE(updated_state.point_cloud, nullptr);
        ASSERT_EQ(updated_state.point_cloud->size(), 2);

        auto means_cpu = updated_state.point_cloud->means.cpu();
        auto acc = means_cpu.accessor<float, 2>();
        EXPECT_FLOAT_EQ(acc(0, 0), 0.0f);
        EXPECT_FLOAT_EQ(acc(0, 1), 0.0f);
        EXPECT_FLOAT_EQ(acc(0, 2), 0.0f);
        EXPECT_FLOAT_EQ(acc(1, 0), 7.0f);
        EXPECT_FLOAT_EQ(acc(1, 1), 8.0f);
        EXPECT_FLOAT_EQ(acc(1, 2), 9.0f);
    }

    TEST_F(SceneManagerRenderStateTest, PlyComparisonRendersFromOwnedNodeModelsWithoutCombinedPrep) {
        SceneManager manager;
        manager.changeContentType(SceneManager::ContentType::SplatFiles);

        auto& scene = manager.getScene();
        const auto left_id = scene.addSplat("left", makeTestSplat(0.0f, 3));
        const auto right_id = scene.addSplat("right", makeTestSplat(1.0f, 3));
        scene.getNodeById(left_id)->model->set_active_sh_degree(1);
        scene.getNodeById(right_id)->model->set_active_sh_degree(2);
        scene.setNodeTransform(
            left_id, glm::translate(glm::mat4(1.0f), glm::vec3(1.0f, 2.0f, 3.0f)));
        scene.setNodeTransform(
            right_id, glm::translate(glm::mat4(1.0f), glm::vec3(-4.0f, 5.0f, 6.0f)));

        EXPECT_FALSE(scene.hasPreparedCombinedModel());
        EXPECT_FALSE(scene.combinedModelBuildPending());

        RenderSettings settings;
        settings.split_view_mode = SplitViewMode::PLYComparison;
        settings.split_position = 0.35f;
        settings.show_rings = true;
        settings.depth_filter_enabled = true;
        settings.depth_filter_min = {-1.0f, -1.0f, -1.0f};
        settings.depth_filter_max = {1.0f, 1.0f, 1.0f};

        Viewport viewport(640, 480);
        const auto scene_state = manager.buildRenderState({.metadata_only = true});
        EXPECT_EQ(scene_state.combined_model, nullptr);
        EXPECT_EQ(scene_state.transform_indices, nullptr);
        EXPECT_EQ(scene_state.selection_mask, nullptr);
        EXPECT_FALSE(scene.hasPreparedCombinedModel());
        EXPECT_FALSE(scene.combinedModelBuildPending());

        const FrameContext ctx{
            .viewport = viewport,
            .scene_manager = &manager,
            .model = nullptr,
            .scene_state = scene_state,
            .settings = settings,
            .render_size = {640, 480},
            .viewport_pos = {0, 0},
        };

        const auto* const left_node = scene.getNodeById(left_id);
        const auto* const right_node = scene.getNodeById(right_id);
        ASSERT_NE(left_node, nullptr);
        ASSERT_NE(right_node, nullptr);
        ASSERT_NE(left_node->model, nullptr);
        ASSERT_NE(right_node->model, nullptr);

        for (size_t i = 0; i < 2; ++i) {

            auto scoped_state = scene_state;
            const auto& node = i == 0 ? *left_node : *right_node;
            // The Vulkan path starts with the full scene request, then replaces
            // its transform array with the owned node's single world transform.
            // Aggregate SH limits must not survive that replacement.
            auto request = buildViewportRenderRequest(
                ctx, {320, 480}, &viewport,
                i == 0 ? SplitViewPanelId::Left : SplitViewPanelId::Right,
                {static_cast<int>(i) * 320, 0}, ctx.render_size);
            ASSERT_EQ(request.scene.node_active_sh_degrees, (std::vector<int>{1, 2}));
            ASSERT_EQ(request.scene.model_transforms->size(), 2u);
            applyPlyComparisonNodeScope(
                request.scene, request.filters, request.overlay, ctx, node, static_cast<int>(i));
            const std::vector<glm::mat4> transforms{scene_coords::nodeVisualizerWorldTransform(scene, node.id)};
            request.scene.model_transforms = &transforms;
            EXPECT_EQ(request.scene.model_transforms->size(), 1u);
            EXPECT_EQ(request.scene.transform_indices, nullptr);
            EXPECT_TRUE(request.scene.node_visibility_mask.empty());
            EXPECT_TRUE(request.scene.node_active_sh_degrees.empty());
            EXPECT_EQ(node.model->get_active_sh_degree(), static_cast<int>(i) + 1);
            EXPECT_EQ(scene_state.node_active_sh_degrees, (std::vector<int>{1, 2}));
            scopeSceneRenderStateToVisibleSplatNode(
                scoped_state, scene, node, static_cast<int>(i), scene_coords::nodeVisualizerWorldTransform(scene, node.id));
            EXPECT_EQ(scoped_state.node_active_sh_degrees,
                      std::vector<int>{static_cast<int>(i) + 1});
        }

        EXPECT_FALSE(scene.hasPreparedCombinedModel());
        EXPECT_EQ(scene.getVisibleNodeIndex(left_id), 0);
        EXPECT_EQ(scene.getVisibleNodeIndex(right_id), 1);
    }

    TEST_F(SceneManagerRenderStateTest, EditHandoffInvalidatesViewportDespitePreservingModelAddress) {
        SceneManager manager;
        manager.changeContentType(SceneManager::ContentType::Dataset);
        auto& scene = manager.getScene();
        scene.addSplat("Model", makeTestSplat(0.0f));
        scene.setTrainingModelNode("Model");

        ViewportFrameLifecycleService lifecycle;
        ViewportArtifactService artifacts;
        const auto observe = [&] {
            return lifecycle.handleModelChange(
                reinterpret_cast<size_t>(manager.getModelForRendering()), artifacts,
                manager.hasDataset() ? ViewportFrameLifecycleService::ModelSource::Training
                                     : ViewportFrameLifecycleService::ModelSource::Scene);
        };
        const auto* training_model = manager.getModelForRendering();
        ASSERT_NE(training_model, nullptr);
        EXPECT_TRUE(observe().changed);
        EXPECT_FALSE(observe().changed);
        const auto generation = artifacts.artifactGeneration();

        manager.switchToEditMode();

        ASSERT_FALSE(manager.hasDataset());
        ASSERT_EQ(manager.getModelForRendering(), training_model);
        const auto handoff = observe();
        EXPECT_TRUE(handoff.changed);
        EXPECT_EQ(handoff.previous_model_ptr, reinterpret_cast<size_t>(training_model));
        EXPECT_GT(artifacts.artifactGeneration(), generation);
        EXPECT_FALSE(observe().changed);
    }

    TEST_F(SceneManagerRenderStateTest, SwitchToEditModePlyComparisonScopesCropAndSelectionToOwnedNodes) {
        using lfs::core::DataType;
        using lfs::core::Device;
        using lfs::core::Tensor;

        SceneManager manager;
        manager.changeContentType(SceneManager::ContentType::Dataset);

        auto& scene = manager.getScene();
        scene.addSplat("Model", makeTestSplat(0.0f));
        scene.setTrainingModelNode("Model");

        manager.switchToEditMode();
        const auto trained_id = scene.getNodeIdByName("Trained Model");
        const auto bike_id = scene.addSplat("bike", makeTwoPointTestSplat(1.0f, 1.5f));

        const auto cropbox_id = scene.getOrCreateCropBoxForSplat(trained_id);
        auto* cropbox = scene.getCropBoxData(cropbox_id);
        ASSERT_NE(cropbox, nullptr);
        cropbox->min = {-1.0f, -1.0f, -1.0f};
        cropbox->max = {1.0f, 1.0f, 1.0f};
        cropbox->enabled = true;

        scene.setSelection({0});

        auto scene_state = manager.buildRenderState({.metadata_only = true});
        EXPECT_EQ(scene_state.combined_model, nullptr);
        EXPECT_EQ(scene_state.transform_indices, nullptr);
        EXPECT_TRUE(scene_state.has_selection);
        scene_state.selected_node_mask = {true, false};

        Tensor transient_selection =
            Tensor::zeros({size_t{3}}, Device::CPU, DataType::Bool);

        RenderSettings settings;
        settings.split_view_mode = SplitViewMode::PLYComparison;
        settings.split_position = 0.4f;
        settings.show_rings = true;
        settings.depth_filter_enabled = true;
        settings.depth_filter_min = {-2.0f, -2.0f, -2.0f};
        settings.depth_filter_max = {2.0f, 2.0f, 2.0f};
        settings.desaturate_unselected = true;

        Viewport viewport(640, 480);
        const FrameContext ctx{
            .viewport = viewport,
            .scene_manager = &manager,
            .model = nullptr,
            .scene_state = std::move(scene_state),
            .settings = settings,
            .render_size = {640, 480},
            .viewport_pos = {0, 0},
            .cursor_preview =
                {.active = true,
                 .x = 32.0f,
                 .y = 24.0f,
                 .radius = 10.0f,
                 .add_mode = true,
                 .selection_tensor = &transient_selection,
                 .preview_selection = &transient_selection,
                 .focused_gaussian_id = 0,
                 .selection_mode = SelectionPreviewMode::Rings},
        };

        const auto* const trained = scene.getNodeById(trained_id);
        const auto* const bike = scene.getNodeById(bike_id);
        ASSERT_NE(trained, nullptr);
        ASSERT_NE(bike, nullptr);

        auto left = buildViewportRenderRequest(ctx, ctx.render_size, &viewport);
        auto right = buildViewportRenderRequest(ctx, ctx.render_size, &viewport);
        left.scene = {};
        right.scene = {};
        applyPlyComparisonNodeScope(left.scene, left.filters, left.overlay, ctx, *trained, 0);
        applyPlyComparisonNodeScope(right.scene, right.filters, right.overlay, ctx, *bike, 1);

        ASSERT_TRUE(left.filters.crop_region.has_value());
        EXPECT_EQ(left.filters.crop_region->parent_node_index, 0);
        EXPECT_FALSE(right.filters.crop_region.has_value());

        ASSERT_NE(left.overlay.emphasis.mask, nullptr);
        EXPECT_EQ(left.overlay.emphasis.mask->numel(), 1u);
        ASSERT_NE(right.overlay.emphasis.mask, nullptr);
        EXPECT_EQ(right.overlay.emphasis.mask->numel(), 2u);

        ASSERT_EQ(left.overlay.emphasis.emphasized_node_mask.size(), 1u);
        EXPECT_TRUE(left.overlay.emphasis.emphasized_node_mask[0]);
        ASSERT_EQ(right.overlay.emphasis.emphasized_node_mask.size(), 1u);
        EXPECT_FALSE(right.overlay.emphasis.emphasized_node_mask[0]);
        EXPECT_TRUE(left.overlay.emphasis.dim_non_emphasized);
        EXPECT_TRUE(right.overlay.emphasis.dim_non_emphasized);
        EXPECT_TRUE(left.overlay.cursor.enabled);
        ASSERT_NE(left.overlay.emphasis.transient_mask.mask, nullptr);
        EXPECT_EQ(left.overlay.emphasis.transient_mask.mask->numel(), 1u);
        ASSERT_NE(right.overlay.emphasis.transient_mask.mask, nullptr);
        EXPECT_EQ(right.overlay.emphasis.transient_mask.mask->numel(), 2u);
        EXPECT_EQ(left.overlay.emphasis.focused_gaussian_id, 0);
        EXPECT_EQ(right.overlay.emphasis.focused_gaussian_id, -1);

        EXPECT_EQ(scene.getVisibleNodeIndex(trained_id), 0);
        EXPECT_EQ(scene.getVisibleNodeIndex(bike_id), 1);
    }

    TEST_F(SceneManagerRenderStateTest, VisibleCountDoesNotBuildAnAggregate) {
        SceneManager manager;
        manager.changeContentType(SceneManager::ContentType::SplatFiles);
        auto& scene = manager.getScene();
        const auto left = scene.addSplat("left", makeTestSplat(0.0f));
        const auto right = scene.addSplat("right", makeTwoPointTestSplat(1.0f, 2.0f));
        scene.getNodeById(right)->model->deleted() =
            lfs::core::Tensor::from_vector({1.0f, 0.0f}, {size_t{2}}, lfs::core::Device::GPU)
                .to(lfs::core::DataType::Bool);
        EXPECT_EQ(scene.getVisibleGaussianCount(), 2u);
        EXPECT_FALSE(scene.hasPreparedCombinedModel());
        EXPECT_FALSE(scene.combinedModelBuildPending());
        scene.setNodeVisibility(left, false);
        EXPECT_EQ(scene.getVisibleGaussianCount(), 1u);
        EXPECT_FALSE(scene.hasPreparedCombinedModel());
    }

    TEST_F(SceneManagerRenderStateTest, ComparisonReleasesRedundantAggregateAndPreservesOwnedModels) {
        SceneManager manager;
        manager.changeContentType(SceneManager::ContentType::SplatFiles);
        auto& scene = manager.getScene();
        const auto left = scene.addSplat("left", makeTestSplat(0.0f));
        const auto right = scene.addSplat("right", makeTestSplat(1.0f));
        scene.setSelection({1});
        ASSERT_NE(scene.getCombinedModel(), nullptr);
        ASSERT_NE(scene.peekTransformIndices(), nullptr);
        scene.discardUnconsolidatedModelCache();
        EXPECT_EQ(scene.peekCombinedModel(), nullptr);
        EXPECT_EQ(scene.peekTransformIndices(), nullptr);
        ASSERT_NE(scene.getNodeById(left)->model, nullptr);
        ASSERT_NE(scene.getNodeById(right)->model, nullptr);
        ASSERT_NE(scene.selectionMaskSliceForNode(right), nullptr);
        EXPECT_TRUE(scene.hasSelection());
        // Returning to a mode that needs the aggregate can reconstruct it.
        ASSERT_NE(scene.getCombinedModel(), nullptr);
        EXPECT_EQ(scene.getCombinedModel()->size(), 2u);
        scene.consolidateNodeModels();
        const auto* consolidated = scene.peekCombinedModel();
        ASSERT_NE(consolidated, nullptr);
        scene.discardUnconsolidatedModelCache();
        EXPECT_EQ(scene.peekCombinedModel(), consolidated);
    }

    TEST_F(SceneManagerRenderStateTest, PlyComparisonMetadataCacheIsDistinctFromFullCombinedState) {
        SceneManager manager;
        manager.changeContentType(SceneManager::ContentType::SplatFiles);
        auto& scene = manager.getScene();
        scene.addSplat("left", makeTestSplat(0.0f));
        scene.addSplat("right", makeTestSplat(1.0f));

        const auto metadata = manager.buildRenderState({.metadata_only = true});
        EXPECT_EQ(metadata.combined_model, nullptr);
        EXPECT_EQ(metadata.transform_indices, nullptr);
        EXPECT_FALSE(scene.hasPreparedCombinedModel());

        const auto full = manager.buildRenderState();
        ASSERT_NE(full.combined_model, nullptr);
        EXPECT_NE(full.transform_indices, nullptr);
        EXPECT_TRUE(scene.hasPreparedCombinedModel());

        const auto metadata_again = manager.buildRenderState({.metadata_only = true});
        EXPECT_EQ(metadata_again.combined_model, nullptr);
        EXPECT_EQ(metadata_again.transform_indices, nullptr);
        EXPECT_EQ(metadata_again.selection_mask, nullptr);
        EXPECT_TRUE(scene.hasPreparedCombinedModel());
        EXPECT_FALSE(scene.combinedModelBuildPending());
    }

    TEST_F(SceneManagerRenderStateTest, ActiveShChangesRefreshFullAndMetadataSnapshots) {
        SceneManager manager;
        manager.changeContentType(SceneManager::ContentType::SplatFiles);
        auto& scene = manager.getScene();
        const auto id = scene.addSplat("model", makeTestSplat(0.0f, 3));
        auto* model = scene.getNodeById(id)->model.get();
        for (const bool metadata_only : {false, true}) {
            model->set_active_sh_degree(1);
            const auto before = manager.buildRenderState({.metadata_only = metadata_only});
            EXPECT_EQ(before.node_active_sh_degrees, std::vector<int>{1});
            model->set_active_sh_degree(2);
            const auto after = manager.buildRenderState({.metadata_only = metadata_only});
            EXPECT_EQ(after.node_active_sh_degrees, std::vector<int>{2});
            EXPECT_EQ(after.combined_model, metadata_only ? nullptr : model);
        }
    }

    TEST_F(SceneManagerRenderStateTest, PlyComparisonPivotSamplesClickedOwnedNode) {
        SceneManager manager;
        manager.changeContentType(SceneManager::ContentType::SplatFiles);
        auto& scene = manager.getScene();
        const auto left_id = scene.addSplat("left", makeTestSplat(0.0f));
        const auto right_id = scene.addSplat("right", makeTestSplat(1.0f));

        const auto left = resolvePlyComparisonDepthSample(scene, 0, SplitViewPanelId::Left);
        const auto right = resolvePlyComparisonDepthSample(scene, 0, SplitViewPanelId::Right);
        ASSERT_NE(left.node, nullptr);
        ASSERT_NE(right.node, nullptr);
        EXPECT_EQ(left.node->id, left_id);
        EXPECT_EQ(right.node->id, right_id);
        EXPECT_TRUE(left.uses_owned_node_model);
        EXPECT_TRUE(right.uses_owned_node_model);
        EXPECT_EQ(left.model, scene.getNodeById(left_id)->model.get());
        EXPECT_EQ(right.model, scene.getNodeById(right_id)->model.get());
        EXPECT_EQ(left.visible_index, 0);
        EXPECT_EQ(right.visible_index, 1);

        auto state = manager.buildRenderState({.metadata_only = true});
        const auto cropbox_id = scene.getOrCreateCropBoxForSplat(right_id);
        auto* cropbox = scene.getCropBoxData(cropbox_id);
        ASSERT_NE(cropbox, nullptr);
        cropbox->enabled = true;
        state = manager.buildRenderState({.metadata_only = true});
        ASSERT_FALSE(state.cropboxes.empty());

        scopeSceneRenderStateToVisibleSplatNode(
            state,
            scene,
            *right.node,
            right.visible_index,
            scene_coords::nodeVisualizerWorldTransform(scene, right_id));
        EXPECT_EQ(state.combined_model, right.model);
        ASSERT_EQ(state.model_transforms.size(), 1u);
        EXPECT_EQ(
            state.model_transforms.front(),
            scene_coords::nodeVisualizerWorldTransform(scene, right_id));
        EXPECT_EQ(state.transform_indices, nullptr);
        ASSERT_EQ(state.cropboxes.size(), 1u);
        EXPECT_EQ(state.cropboxes.front().parent_node_index, 0);
        EXPECT_FALSE(scene.hasPreparedCombinedModel());
    }

    TEST_F(SceneManagerRenderStateTest, PlyComparisonConsolidatedUsesPreparedCombinedWithoutRebuild) {
        SceneManager manager;
        manager.changeContentType(SceneManager::ContentType::SplatFiles);
        auto& scene = manager.getScene();
        const auto left_id = scene.addSplat("left", makeTestSplat(0.0f));
        const auto right_id = scene.addSplat("right", makeTestSplat(1.0f));

        ASSERT_EQ(scene.consolidateNodeModels(), 2u);
        ASSERT_TRUE(scene.isConsolidated());
        ASSERT_TRUE(scene.hasPreparedCombinedModel());
        EXPECT_EQ(scene.getNodeById(left_id)->model, nullptr);
        EXPECT_EQ(scene.getNodeById(right_id)->model, nullptr);
        const auto* const prepared = scene.peekCombinedModel();
        ASSERT_NE(prepared, nullptr);
        EXPECT_NE(scene.peekTransformIndices(), nullptr);

        const auto metadata = manager.buildRenderState({.metadata_only = true});
        EXPECT_EQ(metadata.combined_model, nullptr);
        EXPECT_EQ(metadata.transform_indices, nullptr);
        EXPECT_EQ(scene.peekCombinedModel(), prepared);
        EXPECT_FALSE(scene.combinedModelBuildPending());

        const auto left = resolvePlyComparisonDepthSample(scene, 0, SplitViewPanelId::Left);
        EXPECT_FALSE(left.uses_owned_node_model);
        EXPECT_EQ(left.model, prepared);
        EXPECT_EQ(left.node->id, left_id);
    }

    TEST(SplitViewServiceTest, PlyComparisonPairOffsetWalksUniquePairs) {
        EXPECT_FALSE(plyComparisonPairForOffset(0, 0).has_value());
        EXPECT_FALSE(plyComparisonPairForOffset(1, 0).has_value());
        ASSERT_TRUE(plyComparisonPairForOffset(2, 0).has_value());
        EXPECT_EQ(*plyComparisonPairForOffset(2, 0), (std::pair<size_t, size_t>{0, 1}));
        EXPECT_EQ(*plyComparisonPairForOffset(3, 0), (std::pair<size_t, size_t>{0, 1}));
        EXPECT_EQ(*plyComparisonPairForOffset(3, 1), (std::pair<size_t, size_t>{0, 2}));
        EXPECT_EQ(*plyComparisonPairForOffset(3, 2), (std::pair<size_t, size_t>{1, 2}));
        EXPECT_EQ(*plyComparisonPairForOffset(3, 3), (std::pair<size_t, size_t>{0, 1}));
    }

    TEST_F(SceneManagerRenderStateTest, HiddenEnabledCropBoxStillFiltersRender) {
        SceneManager manager;
        auto& scene = manager.getScene();
        const auto model_id = scene.addSplat("Model", makeTestSplat(0.0f));
        const auto cropbox_id = scene.getOrCreateCropBoxForSplat(model_id);
        ASSERT_NE(cropbox_id, lfs::core::NULL_NODE);

        auto* cropbox = scene.getCropBoxData(cropbox_id);
        ASSERT_NE(cropbox, nullptr);
        cropbox->min = {-0.25f, -0.25f, -0.25f};
        cropbox->max = {0.25f, 0.25f, 0.25f};
        cropbox->enabled = true;
        scene.setNodeVisibility(cropbox_id, false);

        Viewport viewport(640, 480);
        RenderSettings settings;
        const auto scene_state = manager.buildRenderState();
        const FrameContext ctx{
            .viewport = viewport,
            .scene_manager = &manager,
            .model = manager.getModelForRendering(),
            .scene_state = scene_state,
            .settings = settings,
            .render_size = {640, 480},
        };

        const auto request = buildViewportRenderRequest(ctx, {640, 480});
        ASSERT_TRUE(request.filters.crop_region.has_value());
        EXPECT_EQ(request.filters.crop_region->bounds.min, cropbox->min);
        EXPECT_EQ(request.filters.crop_region->bounds.max, cropbox->max);
    }

    TEST_F(SceneManagerRenderStateTest, CropBoxWireframeSelectionHonorsParentSelectionAndVisibility) {
        SceneManager manager;
        auto& scene = manager.getScene();
        const auto model_a_id = scene.addSplat("ModelA", makeTestSplat(0.0f));
        const auto model_b_id = scene.addSplat("ModelB", makeTestSplat(2.0f));
        ASSERT_NE(model_a_id, lfs::core::NULL_NODE);
        ASSERT_NE(model_b_id, lfs::core::NULL_NODE);

        const auto cropbox_id = scene.getOrCreateCropBoxForSplat(model_a_id);
        ASSERT_NE(cropbox_id, lfs::core::NULL_NODE);
        auto* cropbox = scene.getCropBoxData(cropbox_id);
        ASSERT_NE(cropbox, nullptr);
        cropbox->enabled = true;

        manager.selectNode("ModelB");
        auto state = manager.buildRenderState();
        EXPECT_EQ(manager.getSelectedNodeCropBoxId(), lfs::core::NULL_NODE);
        EXPECT_EQ(manager.getActiveSelectionCropBoxId(), lfs::core::NULL_NODE);
        EXPECT_EQ(state.selected_cropbox_index, -1);
        ASSERT_EQ(state.cropboxes.size(), 1u);
        EXPECT_TRUE(state.cropboxes.front().effectively_visible);

        Viewport viewport(640, 480);
        FrameContext ctx{
            .viewport = viewport,
            .scene_manager = &manager,
            .model = manager.getModelForRendering(),
            .scene_state = state,
            .settings = RenderSettings{},
            .render_size = {640, 480},
        };
        auto request = buildViewportRenderRequest(ctx, {640, 480});
        ASSERT_TRUE(request.filters.crop_region.has_value());
        ASSERT_EQ(request.filters.crop_regions.size(), 1u);
        EXPECT_EQ(request.filters.crop_regions.front().parent_node_index, scene.getVisibleNodeIndex(model_a_id));

        manager.selectNode("ModelA");
        state = manager.buildRenderState();
        EXPECT_EQ(manager.getSelectedNodeCropBoxId(), cropbox_id);
        EXPECT_EQ(manager.getActiveSelectionCropBoxId(), cropbox_id);
        ASSERT_GE(state.selected_cropbox_index, 0);
        ASSERT_LT(static_cast<size_t>(state.selected_cropbox_index), state.cropboxes.size());
        EXPECT_EQ(state.cropboxes[static_cast<size_t>(state.selected_cropbox_index)].node_id, cropbox_id);
        EXPECT_TRUE(state.cropboxes[static_cast<size_t>(state.selected_cropbox_index)].effectively_visible);

        scene.setNodeVisibility(cropbox_id, false);
        state = manager.buildRenderState();
        EXPECT_EQ(manager.getSelectedNodeCropBoxId(), cropbox_id);
        EXPECT_EQ(manager.getActiveSelectionCropBoxId(), cropbox_id);
        ASSERT_EQ(state.cropboxes.size(), 1u);
        EXPECT_EQ(state.cropboxes.front().node_id, cropbox_id);
        EXPECT_FALSE(state.cropboxes.front().effectively_visible);

        ctx.scene_state = state;
        request = buildViewportRenderRequest(ctx, {640, 480});
        ASSERT_TRUE(request.filters.crop_region.has_value());
    }

    TEST_F(SceneManagerRenderStateTest, EnabledCropBoxesRemainScopedToTheirParents) {
        SceneManager manager;
        auto& scene = manager.getScene();
        const auto model_a_id = scene.addSplat("ModelA", makeTestSplat(0.0f));
        const auto model_b_id = scene.addSplat("ModelB", makeTestSplat(2.0f));
        ASSERT_NE(model_a_id, lfs::core::NULL_NODE);
        ASSERT_NE(model_b_id, lfs::core::NULL_NODE);

        const auto cropbox_a_id = scene.getOrCreateCropBoxForSplat(model_a_id);
        ASSERT_NE(cropbox_a_id, lfs::core::NULL_NODE);
        auto* cropbox_a = scene.getCropBoxData(cropbox_a_id);
        ASSERT_NE(cropbox_a, nullptr);
        cropbox_a->min = {-2.0f, -2.0f, -2.0f};
        cropbox_a->max = {-1.0f, -1.0f, -1.0f};
        cropbox_a->enabled = true;

        Viewport viewport(640, 480);
        RenderSettings settings;
        settings.desaturate_cropping = true;

        manager.selectNode("ModelB");
        FrameContext ctx{
            .viewport = viewport,
            .scene_manager = &manager,
            .model = manager.getModelForRendering(),
            .scene_state = manager.buildRenderState(),
            .settings = settings,
            .render_size = {640, 480},
        };
        auto request = buildViewportRenderRequest(ctx, {640, 480});
        ASSERT_TRUE(request.filters.crop_region.has_value());
        ASSERT_EQ(request.filters.crop_regions.size(), 1u);
        EXPECT_EQ(request.filters.crop_regions.front().bounds.min, cropbox_a->min);
        EXPECT_EQ(request.filters.crop_regions.front().bounds.max, cropbox_a->max);
        EXPECT_EQ(request.filters.crop_regions.front().parent_node_index, scene.getVisibleNodeIndex(model_a_id));

        const auto cropbox_b_id = scene.getOrCreateCropBoxForSplat(model_b_id);
        ASSERT_NE(cropbox_b_id, lfs::core::NULL_NODE);
        auto* cropbox_b = scene.getCropBoxData(cropbox_b_id);
        ASSERT_NE(cropbox_b, nullptr);
        cropbox_b->min = {1.0f, 1.0f, 1.0f};
        cropbox_b->max = {2.0f, 2.0f, 2.0f};
        cropbox_b->enabled = true;

        ctx.scene_state = manager.buildRenderState();
        request = buildViewportRenderRequest(ctx, {640, 480});
        ASSERT_TRUE(request.filters.crop_region.has_value());
        ASSERT_EQ(request.filters.crop_regions.size(), 2u);
        const auto* filter_a = request.filters.crop_regions[0].parent_node_index == scene.getVisibleNodeIndex(model_a_id)
                                   ? &request.filters.crop_regions[0]
                                   : &request.filters.crop_regions[1];
        const auto* filter_b = request.filters.crop_regions[0].parent_node_index == scene.getVisibleNodeIndex(model_b_id)
                                   ? &request.filters.crop_regions[0]
                                   : &request.filters.crop_regions[1];
        EXPECT_EQ(filter_a->bounds.min, cropbox_a->min);
        EXPECT_EQ(filter_a->bounds.max, cropbox_a->max);
        EXPECT_EQ(filter_b->bounds.min, cropbox_b->min);
        EXPECT_EQ(filter_b->bounds.max, cropbox_b->max);
        EXPECT_TRUE(filter_a->desaturate);
        EXPECT_TRUE(filter_b->desaturate);

        ctx.settings.desaturate_cropping = false;
        request = buildViewportRenderRequest(ctx, {640, 480});
        ASSERT_EQ(request.filters.crop_regions.size(), 2u);
        EXPECT_FALSE(request.filters.crop_regions[0].desaturate);
        EXPECT_FALSE(request.filters.crop_regions[1].desaturate);

        scene.setNodeVisibility(model_a_id, false);
        ctx.scene_state = manager.buildRenderState();
        request = buildViewportRenderRequest(ctx, {640, 480});
        ASSERT_TRUE(request.filters.crop_region.has_value());
        ASSERT_EQ(request.filters.crop_regions.size(), 1u);
        EXPECT_EQ(request.filters.crop_regions.front().parent_node_index, scene.getVisibleNodeIndex(model_b_id));
        EXPECT_EQ(request.filters.crop_regions.front().bounds.min, cropbox_b->min);
        EXPECT_EQ(request.filters.crop_regions.front().bounds.max, cropbox_b->max);
    }

    TEST_F(SceneManagerRenderStateTest, MultipleEnabledCropBoxesWithoutSelectionRemainParentScoped) {
        SceneManager manager;
        auto& scene = manager.getScene();
        const auto model_a_id = scene.addSplat("ModelA", makeTestSplat(0.0f));
        const auto model_b_id = scene.addSplat("ModelB", makeTestSplat(2.0f));
        ASSERT_NE(model_a_id, lfs::core::NULL_NODE);
        ASSERT_NE(model_b_id, lfs::core::NULL_NODE);

        const auto cropbox_a_id = scene.getOrCreateCropBoxForSplat(model_a_id);
        const auto cropbox_b_id = scene.getOrCreateCropBoxForSplat(model_b_id);
        ASSERT_NE(cropbox_a_id, lfs::core::NULL_NODE);
        ASSERT_NE(cropbox_b_id, lfs::core::NULL_NODE);
        ASSERT_NE(scene.getCropBoxData(cropbox_a_id), nullptr);
        ASSERT_NE(scene.getCropBoxData(cropbox_b_id), nullptr);
        scene.getCropBoxData(cropbox_a_id)->enabled = true;
        scene.getCropBoxData(cropbox_b_id)->enabled = true;
        manager.clearSelection();

        Viewport viewport(640, 480);
        const FrameContext ctx{
            .viewport = viewport,
            .scene_manager = &manager,
            .model = manager.getModelForRendering(),
            .scene_state = manager.buildRenderState(),
            .settings = RenderSettings{},
            .render_size = {640, 480},
        };

        const auto request = buildViewportRenderRequest(ctx, {640, 480});
        ASSERT_TRUE(request.filters.crop_region.has_value());
        ASSERT_EQ(request.filters.crop_regions.size(), 2u);
        EXPECT_NE(request.filters.crop_regions[0].parent_node_index, request.filters.crop_regions[1].parent_node_index);
        EXPECT_FALSE(request.filters.crop_regions[0].desaturate);
        EXPECT_FALSE(request.filters.crop_regions[1].desaturate);
    }

    TEST_F(SceneManagerRenderStateTest, LiveCropboxPreviewOverridesOnlyEditedParent) {
        SceneManager manager;
        auto& scene = manager.getScene();
        const auto model_a_id = scene.addSplat("ModelA", makeTestSplat(0.0f));
        const auto model_b_id = scene.addSplat("ModelB", makeTestSplat(2.0f));
        ASSERT_NE(model_a_id, lfs::core::NULL_NODE);
        ASSERT_NE(model_b_id, lfs::core::NULL_NODE);

        const auto cropbox_a_id = scene.getOrCreateCropBoxForSplat(model_a_id);
        const auto cropbox_b_id = scene.getOrCreateCropBoxForSplat(model_b_id);
        ASSERT_NE(cropbox_a_id, lfs::core::NULL_NODE);
        ASSERT_NE(cropbox_b_id, lfs::core::NULL_NODE);
        auto* cropbox_a = scene.getCropBoxData(cropbox_a_id);
        auto* cropbox_b = scene.getCropBoxData(cropbox_b_id);
        ASSERT_NE(cropbox_a, nullptr);
        ASSERT_NE(cropbox_b, nullptr);
        cropbox_a->min = {-2.0f, -2.0f, -2.0f};
        cropbox_a->max = {-1.0f, -1.0f, -1.0f};
        cropbox_a->enabled = true;
        cropbox_b->min = {1.0f, 1.0f, 1.0f};
        cropbox_b->max = {2.0f, 2.0f, 2.0f};
        cropbox_b->enabled = true;

        Viewport viewport(640, 480);
        const FrameContext ctx{
            .viewport = viewport,
            .scene_manager = &manager,
            .model = manager.getModelForRendering(),
            .scene_state = manager.buildRenderState(),
            .settings = RenderSettings{},
            .render_size = {640, 480},
            .gizmo = {
                .cropbox_active = true,
                .cropbox_min = {-0.5f, -0.5f, -0.5f},
                .cropbox_max = {0.5f, 0.5f, 0.5f},
                .cropbox_transform = glm::mat4(1.0f),
                .cropbox_affects_render = true,
                .cropbox_parent_node_index = scene.getVisibleNodeIndex(model_a_id),
            },
        };

        const auto request = buildViewportRenderRequest(ctx, {640, 480});
        ASSERT_EQ(request.filters.crop_regions.size(), 2u);
        const auto* filter_a = request.filters.crop_regions[0].parent_node_index == scene.getVisibleNodeIndex(model_a_id)
                                   ? &request.filters.crop_regions[0]
                                   : &request.filters.crop_regions[1];
        const auto* filter_b = request.filters.crop_regions[0].parent_node_index == scene.getVisibleNodeIndex(model_b_id)
                                   ? &request.filters.crop_regions[0]
                                   : &request.filters.crop_regions[1];
        EXPECT_EQ(filter_a->bounds.min, glm::vec3(-0.5f));
        EXPECT_EQ(filter_a->bounds.max, glm::vec3(0.5f));
        EXPECT_EQ(filter_b->bounds.min, cropbox_b->min);
        EXPECT_EQ(filter_b->bounds.max, cropbox_b->max);
    }
    TEST_F(SceneManagerRenderStateTest, ApplyCropBoxTargetsOnlyParentSplat) {
        SceneManager manager;
        services().set(&manager);
        auto& scene = manager.getScene();
        const auto model_a_id = scene.addSplat("ModelA", makeTwoPointTestSplat(0.0f, 2.0f));
        const auto model_b_id = scene.addSplat("ModelB", makeTwoPointTestSplat(0.0f, 2.0f));
        ASSERT_NE(model_a_id, lfs::core::NULL_NODE);
        ASSERT_NE(model_b_id, lfs::core::NULL_NODE);
        auto* model_a = scene.getMutableNode("ModelA");
        auto* model_b = scene.getMutableNode("ModelB");
        ASSERT_NE(model_a, nullptr);
        ASSERT_NE(model_b, nullptr);
        ASSERT_TRUE(model_a->model);
        ASSERT_TRUE(model_b->model);
        ASSERT_EQ(model_a->model->visible_count(), 2u);
        ASSERT_EQ(model_b->model->visible_count(), 2u);
        ASSERT_FALSE(model_a->model->has_deleted_mask());
        ASSERT_FALSE(model_b->model->has_deleted_mask());

        lfs::geometry::BoundingBox crop_box;
        crop_box.setBounds(glm::vec3(-0.5f, -0.5f, -0.5f), glm::vec3(0.5f, 0.5f, 3.0f));
        lfs::core::events::cmd::CropPLY{
            .crop_box = crop_box,
            .inverse = false,
            .target_node_id = static_cast<int32_t>(model_a_id),
        }
            .emit();

        model_a = scene.getMutableNode("ModelA");
        model_b = scene.getMutableNode("ModelB");
        ASSERT_NE(model_a, nullptr);
        ASSERT_NE(model_b, nullptr);
        ASSERT_TRUE(model_a->model);
        ASSERT_TRUE(model_b->model);
        EXPECT_TRUE(model_a->model->has_deleted_mask());
        EXPECT_FALSE(model_b->model->has_deleted_mask());
        EXPECT_EQ(model_b->model->visible_count(), 2u);
    }
    TEST_F(SceneManagerRenderStateTest, EnsureEllipsoidConvertsExistingCropBoxInPlace) {
        SceneManager manager;
        lfs::vis::screen::ScreenService rendering_manager_views;
        RenderingManager rendering_manager{rendering_manager_views};
        auto& scene = manager.getScene();
        const auto parent_id = scene.addPointCloud("Model", makeTestPointCloud());
        ASSERT_NE(parent_id, lfs::core::NULL_NODE);

        const auto cropbox_id = scene.addCropBox("Model_cropbox", parent_id);
        ASSERT_NE(cropbox_id, lfs::core::NULL_NODE);
        auto* cropbox_node = scene.getNodeById(cropbox_id);
        ASSERT_NE(cropbox_node, nullptr);
        ASSERT_NE(cropbox_node->cropbox, nullptr);
        cropbox_node->cropbox->min = {-3.0f, -5.0f, -7.0f};
        cropbox_node->cropbox->max = {3.0f, 5.0f, 7.0f};
        cropbox_node->cropbox->enabled = true;
        cropbox_node->cropbox->inverse = true;
        cropbox_node->cropbox->color = {0.25f, 0.5f, 0.75f};
        cropbox_node->cropbox->line_width = 5.0f;
        cropbox_node->cropbox->flash_intensity = 0.4f;
        const glm::mat4 transform =
            glm::rotate(glm::translate(glm::mat4(1.0f), glm::vec3(1.0f, 2.0f, 3.0f)),
                        glm::radians(30.0f),
                        glm::vec3(0.0f, 1.0f, 0.0f));
        scene.setNodeTransform(cropbox_node->name, transform);
        scene.setNodeVisibility(cropbox_id, false);
        manager.selectNode(cropbox_node->name);

        auto result = cap::ensureEllipsoid(manager, &rendering_manager, parent_id);
        ASSERT_TRUE(result) << result.error();
        EXPECT_EQ(*result, cropbox_id);
        EXPECT_EQ(scene.getCropBoxForSplat(parent_id), lfs::core::NULL_NODE);
        EXPECT_EQ(scene.getEllipsoidForSplat(parent_id), cropbox_id);

        const auto* converted = scene.getNodeById(cropbox_id);
        ASSERT_NE(converted, nullptr);
        EXPECT_EQ(converted->type, lfs::core::NodeType::ELLIPSOID);
        EXPECT_EQ(converted->name, "Model_cropbox");
        EXPECT_FALSE(converted->cropbox);
        ASSERT_TRUE(converted->ellipsoid);
        EXPECT_EQ(converted->parent_id, parent_id);
        EXPECT_FALSE(converted->visible.get());
        EXPECT_TRUE(converted->ellipsoid->enabled);
        EXPECT_TRUE(converted->ellipsoid->inverse);
        EXPECT_EQ(converted->ellipsoid->color, glm::vec3(0.25f, 0.5f, 0.75f));
        EXPECT_FLOAT_EQ(converted->ellipsoid->line_width, 5.0f);
        EXPECT_FLOAT_EQ(converted->ellipsoid->flash_intensity, 0.4f);
        EXPECT_EQ(converted->ellipsoid->radii, glm::vec3(3.0f, 5.0f, 7.0f));
        EXPECT_EQ(scene.getNodeTransform(converted->name), transform);
        EXPECT_EQ(manager.getSelectedNodeName(), converted->name);

        const auto settings = rendering_manager.getSettings();
        EXPECT_FALSE(settings.show_crop_box);
        EXPECT_FALSE(settings.use_crop_box);
        EXPECT_FALSE(settings.show_ellipsoid);
        EXPECT_TRUE(settings.use_ellipsoid);

        const auto state = manager.buildRenderState();
        EXPECT_TRUE(state.cropboxes.empty());
        ASSERT_EQ(state.ellipsoids.size(), 1u);
        EXPECT_EQ(state.ellipsoids.front().node_id, cropbox_id);
    }

    TEST_F(SceneManagerRenderStateTest, DefaultCropBoxConvertsToEllipsoidAtCropCenter) {
        SceneManager manager;
        lfs::vis::screen::ScreenService rendering_manager_views;
        RenderingManager rendering_manager{rendering_manager_views};
        auto& scene = manager.getScene();
        const auto parent_id = scene.addPointCloud("Model", makeTestPointCloud());
        ASSERT_NE(parent_id, lfs::core::NULL_NODE);

        auto cropbox_result = cap::ensureCropBox(manager, &rendering_manager, parent_id);
        ASSERT_TRUE(cropbox_result) << cropbox_result.error();
        const auto cropbox_id = *cropbox_result;
        const auto* cropbox_node = scene.getNodeById(cropbox_id);
        ASSERT_NE(cropbox_node, nullptr);
        ASSERT_TRUE(cropbox_node->cropbox);
        EXPECT_EQ(cropbox_node->cropbox->min, glm::vec3(-0.5f, -1e-4f, -1e-4f));
        EXPECT_EQ(cropbox_node->cropbox->max, glm::vec3(0.5f, 1e-4f, 1e-4f));
        const glm::mat4 cropbox_transform = scene.getNodeTransform(cropbox_node->name);
        EXPECT_EQ(glm::vec3(cropbox_transform[3]), glm::vec3(0.5f, 0.0f, 0.0f));

        auto ellipsoid_result = cap::ensureEllipsoid(manager, &rendering_manager, parent_id);
        ASSERT_TRUE(ellipsoid_result) << ellipsoid_result.error();
        EXPECT_EQ(*ellipsoid_result, cropbox_id);

        const auto* ellipsoid_node = scene.getNodeById(cropbox_id);
        ASSERT_NE(ellipsoid_node, nullptr);
        EXPECT_EQ(ellipsoid_node->type, lfs::core::NodeType::ELLIPSOID);
        ASSERT_TRUE(ellipsoid_node->ellipsoid);
        EXPECT_EQ(ellipsoid_node->ellipsoid->radii, glm::vec3(0.5f, 1e-4f, 1e-4f));
        EXPECT_EQ(scene.getNodeTransform(ellipsoid_node->name), cropbox_transform);
    }

    TEST_F(SceneManagerRenderStateTest, AddCropCommandsConvertSelectedCropVolumeViaParent) {
        SceneManager manager;
        lfs::vis::screen::ScreenService rendering_manager_views;
        RenderingManager rendering_manager{rendering_manager_views};
        services().set(&rendering_manager);
        EditorContext editor;
        services().set(&editor);
        auto& scene = manager.getScene();
        const auto parent_id = scene.addPointCloud("Model", makeTestPointCloud());
        ASSERT_NE(parent_id, lfs::core::NULL_NODE);

        auto cropbox_result = cap::ensureCropBox(manager, &rendering_manager, parent_id);
        ASSERT_TRUE(cropbox_result) << cropbox_result.error();
        const auto cropbox_id = *cropbox_result;
        const auto* cropbox_node = scene.getNodeById(cropbox_id);
        ASSERT_NE(cropbox_node, nullptr);
        ASSERT_TRUE(cropbox_node->cropbox);

        lfs::core::events::cmd::AddCropEllipsoid{.node_name = cropbox_node->name}.emit();

        const auto* ellipsoid_node = scene.getNodeById(cropbox_id);
        ASSERT_NE(ellipsoid_node, nullptr);
        EXPECT_EQ(ellipsoid_node->type, lfs::core::NodeType::ELLIPSOID);
        EXPECT_EQ(scene.getEllipsoidForSplat(parent_id), cropbox_id);
        EXPECT_EQ(scene.getCropBoxForSplat(parent_id), lfs::core::NULL_NODE);
        EXPECT_EQ(manager.getSelectedNodeName(), ellipsoid_node->name);
        EXPECT_EQ(editor.getActiveOperator(), "builtin.cropbox");

        lfs::core::events::cmd::AddCropBox{.node_name = ellipsoid_node->name}.emit();

        const auto* converted_node = scene.getNodeById(cropbox_id);
        ASSERT_NE(converted_node, nullptr);
        EXPECT_EQ(converted_node->type, lfs::core::NodeType::CROPBOX);
        EXPECT_EQ(scene.getCropBoxForSplat(parent_id), cropbox_id);
        EXPECT_EQ(scene.getEllipsoidForSplat(parent_id), lfs::core::NULL_NODE);
        EXPECT_EQ(manager.getSelectedNodeName(), converted_node->name);
        EXPECT_EQ(editor.getActiveOperator(), "builtin.cropbox");
    }

    TEST_F(SceneManagerRenderStateTest, AddCropCommandsRevealExistingHiddenCropVolume) {
        SceneManager manager;
        lfs::vis::screen::ScreenService rendering_manager_views;
        RenderingManager rendering_manager{rendering_manager_views};
        services().set(&rendering_manager);
        auto& scene = manager.getScene();
        const auto parent_id = scene.addPointCloud("Model", makeTestPointCloud());
        ASSERT_NE(parent_id, lfs::core::NULL_NODE);

        auto cropbox_result = cap::ensureCropBox(manager, &rendering_manager, parent_id);
        ASSERT_TRUE(cropbox_result) << cropbox_result.error();
        const auto cropbox_id = *cropbox_result;
        const auto* cropbox_node = scene.getNodeById(cropbox_id);
        ASSERT_NE(cropbox_node, nullptr);

        manager.setNodeVisibility(cropbox_id, false);
        lfs::core::events::cmd::AddCropBox{.node_name = "Model"}.emit();

        cropbox_node = scene.getNodeById(cropbox_id);
        ASSERT_NE(cropbox_node, nullptr);
        EXPECT_TRUE(cropbox_node->visible);
        EXPECT_EQ(cropbox_node->type, lfs::core::NodeType::CROPBOX);
        EXPECT_EQ(manager.getSelectedNodeName(), cropbox_node->name);

        manager.setNodeVisibility(cropbox_id, false);
        lfs::core::events::cmd::AddCropEllipsoid{.node_name = cropbox_node->name}.emit();

        const auto* ellipsoid_node = scene.getNodeById(cropbox_id);
        ASSERT_NE(ellipsoid_node, nullptr);
        EXPECT_TRUE(ellipsoid_node->visible);
        EXPECT_EQ(ellipsoid_node->type, lfs::core::NodeType::ELLIPSOID);
        EXPECT_EQ(scene.getEllipsoidForSplat(parent_id), cropbox_id);
        EXPECT_EQ(manager.getSelectedNodeName(), ellipsoid_node->name);
    }

    TEST_F(SceneManagerRenderStateTest, ResetAndFitPreserveEnabledCropEffects) {
        SceneManager manager;
        lfs::vis::screen::ScreenService rendering_manager_views;
        RenderingManager rendering_manager{rendering_manager_views};
        services().set(&rendering_manager);
        auto& scene = manager.getScene();
        const auto parent_id = scene.addSplat("Model", makeTestSplat(0.0f));
        ASSERT_NE(parent_id, lfs::core::NULL_NODE);

        auto cropbox_result = cap::ensureCropBox(manager, &rendering_manager, parent_id);
        ASSERT_TRUE(cropbox_result) << cropbox_result.error();
        const auto cropbox_id = *cropbox_result;
        auto* cropbox_node = scene.getNodeById(cropbox_id);
        ASSERT_NE(cropbox_node, nullptr);
        ASSERT_TRUE(cropbox_node->cropbox);
        cropbox_node->cropbox->enabled = true;
        manager.selectNode(cropbox_node->name);

        auto settings = rendering_manager.getSettings();
        settings.use_crop_box = false;
        rendering_manager.updateSettings(settings);
        lfs::core::events::cmd::ResetCropBox{}.emit();

        cropbox_node = scene.getNodeById(cropbox_id);
        ASSERT_NE(cropbox_node, nullptr);
        ASSERT_TRUE(cropbox_node->cropbox);
        EXPECT_TRUE(cropbox_node->cropbox->enabled);
        EXPECT_TRUE(rendering_manager.getSettings().use_crop_box);

        Viewport viewport(640, 480);
        auto scene_state = manager.buildRenderState();
        FrameContext ctx{
            .viewport = viewport,
            .scene_manager = &manager,
            .model = manager.getModelForRendering(),
            .scene_state = scene_state,
            .settings = rendering_manager.getSettings(),
            .render_size = {640, 480},
        };
        auto request = buildViewportRenderRequest(ctx, {640, 480});
        EXPECT_TRUE(request.filters.crop_region.has_value());

        settings = rendering_manager.getSettings();
        settings.use_crop_box = false;
        rendering_manager.updateSettings(settings);
        lfs::core::events::cmd::FitCropBoxToScene{.use_percentile = false}.emit();
        cropbox_node = scene.getNodeById(cropbox_id);
        ASSERT_NE(cropbox_node, nullptr);
        ASSERT_TRUE(cropbox_node->cropbox);
        EXPECT_TRUE(cropbox_node->cropbox->enabled);
        EXPECT_TRUE(rendering_manager.getSettings().use_crop_box);
        scene_state = manager.buildRenderState();
        ctx.scene_state = scene_state;
        ctx.settings = rendering_manager.getSettings();
        request = buildViewportRenderRequest(ctx, {640, 480});
        EXPECT_TRUE(request.filters.crop_region.has_value());

        lfs::core::events::cmd::AddCropEllipsoid{.node_name = cropbox_node->name}.emit();
        auto* ellipsoid_node = scene.getNodeById(cropbox_id);
        ASSERT_NE(ellipsoid_node, nullptr);
        ASSERT_TRUE(ellipsoid_node->ellipsoid);
        ellipsoid_node->ellipsoid->enabled = true;
        settings = rendering_manager.getSettings();
        settings.use_ellipsoid = false;
        rendering_manager.updateSettings(settings);
        lfs::core::events::cmd::ResetEllipsoid{}.emit();

        ellipsoid_node = scene.getNodeById(cropbox_id);
        ASSERT_NE(ellipsoid_node, nullptr);
        ASSERT_TRUE(ellipsoid_node->ellipsoid);
        EXPECT_TRUE(ellipsoid_node->ellipsoid->enabled);
        EXPECT_TRUE(rendering_manager.getSettings().use_ellipsoid);
        scene_state = manager.buildRenderState();
        ctx.scene_state = scene_state;
        ctx.settings = rendering_manager.getSettings();
        request = buildViewportRenderRequest(ctx, {640, 480});
        EXPECT_TRUE(request.filters.ellipsoid_region.has_value());

        settings = rendering_manager.getSettings();
        settings.use_ellipsoid = false;
        rendering_manager.updateSettings(settings);
        lfs::core::events::cmd::FitEllipsoidToScene{.use_percentile = false}.emit();
        ellipsoid_node = scene.getNodeById(cropbox_id);
        ASSERT_NE(ellipsoid_node, nullptr);
        ASSERT_TRUE(ellipsoid_node->ellipsoid);
        EXPECT_TRUE(ellipsoid_node->ellipsoid->enabled);
        EXPECT_TRUE(rendering_manager.getSettings().use_ellipsoid);
        scene_state = manager.buildRenderState();
        ctx.scene_state = scene_state;
        ctx.settings = rendering_manager.getSettings();
        request = buildViewportRenderRequest(ctx, {640, 480});
        EXPECT_TRUE(request.filters.ellipsoid_region.has_value());
    }

    TEST_F(SceneManagerRenderStateTest, DeletingSelectedCropVolumeSelectsParentAndClearsRenderState) {
        SceneManager manager;
        lfs::vis::screen::ScreenService rendering_manager_views;
        RenderingManager rendering_manager{rendering_manager_views};
        services().set(&manager);
        services().set(&rendering_manager);
        auto& scene = manager.getScene();
        const auto parent_id = scene.addPointCloud("Model", makeTestPointCloud());
        ASSERT_NE(parent_id, lfs::core::NULL_NODE);

        auto cropbox_result = cap::ensureCropBox(manager, &rendering_manager, parent_id);
        ASSERT_TRUE(cropbox_result) << cropbox_result.error();
        const auto cropbox_id = *cropbox_result;
        const auto* cropbox_node = scene.getNodeById(cropbox_id);
        ASSERT_NE(cropbox_node, nullptr);
        manager.selectNode(cropbox_node->name);

        auto settings = rendering_manager.getSettings();
        settings.show_crop_box = true;
        settings.use_crop_box = true;
        rendering_manager.updateSettings(settings);

        manager.removePLY(cropbox_node->name);

        EXPECT_EQ(scene.getCropBoxForSplat(parent_id), lfs::core::NULL_NODE);
        EXPECT_EQ(manager.getSelectedNodeName(), "Model");
        settings = rendering_manager.getSettings();
        EXPECT_FALSE(settings.show_crop_box);
        EXPECT_FALSE(settings.use_crop_box);
        EXPECT_TRUE(manager.buildRenderState().cropboxes.empty());
    }

    TEST_F(SceneManagerRenderStateTest, EnsureCropBoxConvertsExistingEllipsoidAndUndoRedoRestoresShape) {
        SceneManager manager;
        lfs::vis::screen::ScreenService rendering_manager_views;
        RenderingManager rendering_manager{rendering_manager_views};
        auto& scene = manager.getScene();
        const auto parent_id = scene.addPointCloud("Model", makeTestPointCloud());
        ASSERT_NE(parent_id, lfs::core::NULL_NODE);

        const auto ellipsoid_id = scene.addEllipsoid("Model_ellipsoid", parent_id);
        ASSERT_NE(ellipsoid_id, lfs::core::NULL_NODE);
        auto* ellipsoid_node = scene.getNodeById(ellipsoid_id);
        ASSERT_NE(ellipsoid_node, nullptr);
        ASSERT_NE(ellipsoid_node->ellipsoid, nullptr);
        ellipsoid_node->ellipsoid->radii = {2.0f, 3.0f, 4.0f};
        ellipsoid_node->ellipsoid->enabled = true;
        ellipsoid_node->ellipsoid->inverse = true;
        ellipsoid_node->ellipsoid->color = {0.8f, 0.6f, 0.4f};
        ellipsoid_node->ellipsoid->line_width = 6.0f;
        ellipsoid_node->ellipsoid->flash_intensity = 0.3f;
        const glm::mat4 transform = glm::scale(
            glm::translate(glm::mat4(1.0f), glm::vec3(-1.0f, 2.0f, -3.0f)),
            glm::vec3(1.0f, 2.0f, 3.0f));
        scene.setNodeTransform(ellipsoid_node->name, transform);
        scene.setNodeVisibility(ellipsoid_id, false);
        manager.selectNode(ellipsoid_node->name);

        auto result = cap::ensureCropBox(manager, &rendering_manager, parent_id);
        ASSERT_TRUE(result) << result.error();
        EXPECT_EQ(*result, ellipsoid_id);
        EXPECT_EQ(scene.getCropBoxForSplat(parent_id), ellipsoid_id);
        EXPECT_EQ(scene.getEllipsoidForSplat(parent_id), lfs::core::NULL_NODE);

        auto* converted = scene.getNodeById(ellipsoid_id);
        ASSERT_NE(converted, nullptr);
        EXPECT_EQ(converted->type, lfs::core::NodeType::CROPBOX);
        EXPECT_EQ(converted->name, "Model_ellipsoid");
        ASSERT_TRUE(converted->cropbox);
        EXPECT_FALSE(converted->ellipsoid);
        EXPECT_EQ(converted->cropbox->min, glm::vec3(-2.0f, -3.0f, -4.0f));
        EXPECT_EQ(converted->cropbox->max, glm::vec3(2.0f, 3.0f, 4.0f));
        EXPECT_TRUE(converted->cropbox->enabled);
        EXPECT_TRUE(converted->cropbox->inverse);
        EXPECT_EQ(converted->cropbox->color, glm::vec3(0.8f, 0.6f, 0.4f));
        EXPECT_FLOAT_EQ(converted->cropbox->line_width, 6.0f);
        EXPECT_FLOAT_EQ(converted->cropbox->flash_intensity, 0.3f);
        EXPECT_EQ(scene.getNodeTransform(converted->name), transform);
        EXPECT_EQ(manager.getSelectedNodeName(), converted->name);

        auto settings = rendering_manager.getSettings();
        EXPECT_FALSE(settings.show_ellipsoid);
        EXPECT_FALSE(settings.use_ellipsoid);
        EXPECT_FALSE(settings.show_crop_box);
        EXPECT_TRUE(settings.use_crop_box);

        auto state = manager.buildRenderState();
        ASSERT_EQ(state.cropboxes.size(), 1u);
        EXPECT_TRUE(state.ellipsoids.empty());

        auto undo_result = op::undoHistory().undo();
        ASSERT_TRUE(undo_result.success);
        const auto* undone = scene.getNode("Model_ellipsoid");
        ASSERT_NE(undone, nullptr);
        EXPECT_EQ(undone->type, lfs::core::NodeType::ELLIPSOID);
        ASSERT_TRUE(undone->ellipsoid);
        EXPECT_FALSE(undone->cropbox);
        EXPECT_EQ(undone->ellipsoid->radii, glm::vec3(2.0f, 3.0f, 4.0f));
        EXPECT_TRUE(undone->ellipsoid->enabled);
        EXPECT_TRUE(undone->ellipsoid->inverse);
        EXPECT_FALSE(undone->visible.get());
        EXPECT_EQ(manager.getSelectedNodeName(), "Model_ellipsoid");

        auto redo_result = op::undoHistory().redo();
        ASSERT_TRUE(redo_result.success);
        const auto* redone = scene.getNode("Model_ellipsoid");
        ASSERT_NE(redone, nullptr);
        EXPECT_EQ(redone->type, lfs::core::NodeType::CROPBOX);
        ASSERT_TRUE(redone->cropbox);
        EXPECT_FALSE(redone->ellipsoid);
        EXPECT_EQ(redone->cropbox->min, glm::vec3(-2.0f, -3.0f, -4.0f));
        EXPECT_EQ(redone->cropbox->max, glm::vec3(2.0f, 3.0f, 4.0f));
        EXPECT_TRUE(redone->cropbox->enabled);
        EXPECT_TRUE(redone->cropbox->inverse);
    }

    TEST_F(SceneManagerRenderStateTest, PointCloudRequestUsesSelectedEnabledEllipsoidCrop) {
        SceneManager manager;
        auto& scene = manager.getScene();
        const auto parent_id = scene.addPointCloud("Model", makeTestPointCloud());
        ASSERT_NE(parent_id, lfs::core::NULL_NODE);

        auto ellipsoid_result = cap::ensureEllipsoid(manager, nullptr, parent_id);
        ASSERT_TRUE(ellipsoid_result) << ellipsoid_result.error();
        auto* ellipsoid_node = scene.getNodeById(*ellipsoid_result);
        ASSERT_NE(ellipsoid_node, nullptr);
        ASSERT_TRUE(ellipsoid_node->ellipsoid);
        ellipsoid_node->ellipsoid->enabled = true;
        ellipsoid_node->ellipsoid->inverse = true;
        ellipsoid_node->ellipsoid->radii = {2.0f, 3.0f, 4.0f};
        const auto transform = glm::translate(glm::mat4(1.0f), glm::vec3(1.0f, 2.0f, 3.0f));
        scene.setNodeTransform(ellipsoid_node->name, transform);
        manager.selectNode(ellipsoid_node->name);

        Viewport viewport(640, 480);
        auto scene_state = manager.buildRenderState();
        ASSERT_EQ(scene_state.ellipsoids.size(), 1u);
        RenderSettings settings;
        const FrameContext ctx{
            .viewport = viewport,
            .scene_manager = &manager,
            .scene_state = scene_state,
            .settings = settings,
            .render_size = {640, 480},
        };
        const std::vector<glm::mat4> transforms = scene_state.model_transforms;

        const auto request = buildPointCloudRenderRequest(ctx, {640, 480}, transforms);

        ASSERT_TRUE(request.filters.crop_ellipsoid.has_value());
        EXPECT_FALSE(request.filters.crop_box.has_value());
        EXPECT_EQ(request.filters.crop_ellipsoid->radii, glm::vec3(2.0f, 3.0f, 4.0f));
        EXPECT_EQ(request.filters.crop_ellipsoid->transform,
                  glm::inverse(scene_state.ellipsoids.front().world_transform));
        EXPECT_TRUE(request.filters.crop_inverse);
        EXPECT_FALSE(request.filters.crop_desaturate);
    }

    TEST_F(SceneManagerRenderStateTest, PointCloudRequestIgnoresDisabledSelectedEllipsoidCrop) {
        SceneManager manager;
        auto& scene = manager.getScene();
        const auto parent_id = scene.addPointCloud("Model", makeTestPointCloud());
        ASSERT_NE(parent_id, lfs::core::NULL_NODE);

        auto ellipsoid_result = cap::ensureEllipsoid(manager, nullptr, parent_id);
        ASSERT_TRUE(ellipsoid_result) << ellipsoid_result.error();
        auto* ellipsoid_node = scene.getNodeById(*ellipsoid_result);
        ASSERT_NE(ellipsoid_node, nullptr);
        ASSERT_TRUE(ellipsoid_node->ellipsoid);
        ellipsoid_node->ellipsoid->enabled = false;
        manager.selectNode(ellipsoid_node->name);

        Viewport viewport(640, 480);
        auto scene_state = manager.buildRenderState();
        RenderSettings settings;
        const FrameContext ctx{
            .viewport = viewport,
            .scene_manager = &manager,
            .scene_state = scene_state,
            .settings = settings,
            .render_size = {640, 480},
        };
        const std::vector<glm::mat4> transforms = scene_state.model_transforms;

        const auto request = buildPointCloudRenderRequest(ctx, {640, 480}, transforms);

        EXPECT_FALSE(request.filters.crop_ellipsoid.has_value());
        EXPECT_FALSE(request.filters.crop_box.has_value());
    }

    TEST_F(SceneManagerRenderStateTest, DeleteSelectedGaussiansRejectsSelectedCropVolumeWithoutRemovingIt) {
        SceneManager manager;
        auto& scene = manager.getScene();
        const auto parent_id = scene.addPointCloud("Model", makeTestPointCloud());
        ASSERT_NE(parent_id, lfs::core::NULL_NODE);

        auto cropbox_result = cap::ensureCropBox(manager, nullptr, parent_id);
        ASSERT_TRUE(cropbox_result) << cropbox_result.error();
        const auto cropbox_id = *cropbox_result;
        auto* cropbox_node = scene.getNodeById(cropbox_id);
        ASSERT_NE(cropbox_node, nullptr);
        const auto cropbox_name = cropbox_node->name;
        manager.selectNode(cropbox_name);

        const auto result = manager.deleteSelectedGaussiansWithHistory();

        ASSERT_FALSE(result.has_value());
        EXPECT_EQ(result.error(), "Use the Crop toolbar Delete action to remove selected crop volumes");
        EXPECT_NE(scene.getNodeById(cropbox_id), nullptr);
        EXPECT_EQ(manager.getSelectedNodeName(), cropbox_name);
    }

    TEST_F(SceneManagerRenderStateTest, PointCloudRequestKeepsSingleEnabledCropBoxAfterDeselection) {
        SceneManager manager;
        auto& scene = manager.getScene();
        const auto parent_id = scene.addPointCloud("Model", makeTestPointCloud());
        ASSERT_NE(parent_id, lfs::core::NULL_NODE);

        auto cropbox_result = cap::ensureCropBox(manager, nullptr, parent_id);
        ASSERT_TRUE(cropbox_result) << cropbox_result.error();
        auto* cropbox_node = scene.getNodeById(*cropbox_result);
        ASSERT_NE(cropbox_node, nullptr);
        ASSERT_TRUE(cropbox_node->cropbox);
        cropbox_node->cropbox->enabled = true;
        cropbox_node->cropbox->inverse = true;
        cropbox_node->cropbox->min = {-1.0f, -2.0f, -3.0f};
        cropbox_node->cropbox->max = {1.0f, 2.0f, 3.0f};
        scene.setNodeTransform(cropbox_node->name, glm::translate(glm::mat4(1.0f), glm::vec3(2.0f, 3.0f, 4.0f)));
        scene.setNodeVisibility(*cropbox_result, false);
        manager.clearSelection();

        Viewport viewport(640, 480);
        auto scene_state = manager.buildRenderState();
        ASSERT_EQ(scene_state.cropboxes.size(), 1u);
        EXPECT_LT(scene_state.cropboxes.front().parent_node_index, 0);
        EXPECT_FALSE(scene_state.cropboxes.front().effectively_visible);
        EXPECT_TRUE(scene_state.cropboxes.front().parent_effectively_visible);
        RenderSettings settings;
        settings.desaturate_cropping = false;
        const FrameContext ctx{
            .viewport = viewport,
            .scene_manager = &manager,
            .scene_state = scene_state,
            .settings = settings,
            .render_size = {640, 480},
        };
        const std::vector<glm::mat4> transforms = scene_state.model_transforms;

        const auto request = buildPointCloudRenderRequest(ctx, {640, 480}, transforms);

        ASSERT_TRUE(request.filters.crop_box.has_value());
        EXPECT_FALSE(request.filters.crop_ellipsoid.has_value());
        EXPECT_EQ(request.filters.crop_box->min, glm::vec3(-1.0f, -2.0f, -3.0f));
        EXPECT_EQ(request.filters.crop_box->max, glm::vec3(1.0f, 2.0f, 3.0f));
        EXPECT_EQ(request.filters.crop_box->transform, glm::inverse(scene_state.cropboxes.front().world_transform));
        EXPECT_TRUE(request.filters.crop_inverse);
        EXPECT_FALSE(request.filters.crop_desaturate);
    }

    TEST_F(SceneManagerRenderStateTest, PointCloudRequestIgnoresHiddenSplatCropBoxFallback) {
        SceneManager manager;
        auto& scene = manager.getScene();
        const auto hidden_splat_id = scene.addSplat("HiddenSplat", makeTestSplat(0.0f));
        ASSERT_NE(hidden_splat_id, lfs::core::NULL_NODE);
        const auto point_cloud_id = scene.addPointCloud("VisiblePoints", makeTestPointCloud());
        ASSERT_NE(point_cloud_id, lfs::core::NULL_NODE);

        auto cropbox_result = cap::ensureCropBox(manager, nullptr, hidden_splat_id);
        ASSERT_TRUE(cropbox_result) << cropbox_result.error();
        auto* cropbox_node = scene.getNodeById(*cropbox_result);
        ASSERT_NE(cropbox_node, nullptr);
        ASSERT_TRUE(cropbox_node->cropbox);
        cropbox_node->cropbox->enabled = true;
        scene.setNodeVisibility("HiddenSplat", false);
        manager.clearSelection();

        Viewport viewport(640, 480);
        auto scene_state = manager.buildRenderState();
        ASSERT_EQ(scene_state.cropboxes.size(), 1u);
        EXPECT_FALSE(scene_state.cropboxes.front().effectively_visible);
        EXPECT_FALSE(scene_state.cropboxes.front().parent_effectively_visible);
        RenderSettings settings;
        const FrameContext ctx{
            .viewport = viewport,
            .scene_manager = &manager,
            .scene_state = scene_state,
            .settings = settings,
            .render_size = {640, 480},
        };
        const std::vector<glm::mat4> transforms = scene_state.model_transforms;

        const auto request = buildPointCloudRenderRequest(ctx, {640, 480}, transforms);

        EXPECT_FALSE(request.filters.crop_box.has_value());
        EXPECT_FALSE(request.filters.crop_ellipsoid.has_value());
    }

    TEST_F(SceneManagerRenderStateTest, PointCloudRequestKeepsSingleEnabledEllipsoidAfterDeselection) {
        SceneManager manager;
        auto& scene = manager.getScene();
        const auto parent_id = scene.addPointCloud("Model", makeTestPointCloud());
        ASSERT_NE(parent_id, lfs::core::NULL_NODE);

        auto ellipsoid_result = cap::ensureEllipsoid(manager, nullptr, parent_id);
        ASSERT_TRUE(ellipsoid_result) << ellipsoid_result.error();
        auto* ellipsoid_node = scene.getNodeById(*ellipsoid_result);
        ASSERT_NE(ellipsoid_node, nullptr);
        ASSERT_TRUE(ellipsoid_node->ellipsoid);
        ellipsoid_node->ellipsoid->enabled = true;
        ellipsoid_node->ellipsoid->inverse = true;
        ellipsoid_node->ellipsoid->radii = {2.0f, 3.0f, 4.0f};
        scene.setNodeTransform(ellipsoid_node->name, glm::translate(glm::mat4(1.0f), glm::vec3(3.0f, 4.0f, 5.0f)));
        scene.setNodeVisibility(*ellipsoid_result, false);
        manager.clearSelection();

        Viewport viewport(640, 480);
        auto scene_state = manager.buildRenderState();
        ASSERT_EQ(scene_state.ellipsoids.size(), 1u);
        EXPECT_LT(scene_state.ellipsoids.front().parent_node_index, 0);
        EXPECT_FALSE(scene_state.ellipsoids.front().effectively_visible);
        EXPECT_TRUE(scene_state.ellipsoids.front().parent_effectively_visible);
        RenderSettings settings;
        settings.desaturate_cropping = false;
        const FrameContext ctx{
            .viewport = viewport,
            .scene_manager = &manager,
            .scene_state = scene_state,
            .settings = settings,
            .render_size = {640, 480},
        };
        const std::vector<glm::mat4> transforms = scene_state.model_transforms;

        const auto request = buildPointCloudRenderRequest(ctx, {640, 480}, transforms);

        ASSERT_TRUE(request.filters.crop_ellipsoid.has_value());
        EXPECT_FALSE(request.filters.crop_box.has_value());
        EXPECT_EQ(request.filters.crop_ellipsoid->radii, glm::vec3(2.0f, 3.0f, 4.0f));
        EXPECT_EQ(request.filters.crop_ellipsoid->transform, glm::inverse(scene_state.ellipsoids.front().world_transform));
        EXPECT_TRUE(request.filters.crop_inverse);
        EXPECT_FALSE(request.filters.crop_desaturate);
    }

    TEST_F(SceneManagerRenderStateTest, PointCloudRequestIgnoresHiddenSplatEllipsoidFallback) {
        SceneManager manager;
        auto& scene = manager.getScene();
        const auto hidden_splat_id = scene.addSplat("HiddenSplat", makeTestSplat(0.0f));
        ASSERT_NE(hidden_splat_id, lfs::core::NULL_NODE);
        const auto point_cloud_id = scene.addPointCloud("VisiblePoints", makeTestPointCloud());
        ASSERT_NE(point_cloud_id, lfs::core::NULL_NODE);

        auto ellipsoid_result = cap::ensureEllipsoid(manager, nullptr, hidden_splat_id);
        ASSERT_TRUE(ellipsoid_result) << ellipsoid_result.error();
        auto* ellipsoid_node = scene.getNodeById(*ellipsoid_result);
        ASSERT_NE(ellipsoid_node, nullptr);
        ASSERT_TRUE(ellipsoid_node->ellipsoid);
        ellipsoid_node->ellipsoid->enabled = true;
        scene.setNodeVisibility("HiddenSplat", false);
        manager.clearSelection();

        Viewport viewport(640, 480);
        auto scene_state = manager.buildRenderState();
        ASSERT_EQ(scene_state.ellipsoids.size(), 1u);
        EXPECT_FALSE(scene_state.ellipsoids.front().effectively_visible);
        EXPECT_FALSE(scene_state.ellipsoids.front().parent_effectively_visible);
        RenderSettings settings;
        const FrameContext ctx{
            .viewport = viewport,
            .scene_manager = &manager,
            .scene_state = scene_state,
            .settings = settings,
            .render_size = {640, 480},
        };
        const std::vector<glm::mat4> transforms = scene_state.model_transforms;

        const auto request = buildPointCloudRenderRequest(ctx, {640, 480}, transforms);

        EXPECT_FALSE(request.filters.crop_box.has_value());
        EXPECT_FALSE(request.filters.crop_ellipsoid.has_value());
    }

    TEST(ViewportRequestBuilderTest, PointCloudRequestUsesCropBoxWhenBothPointCloudGizmosAffectRender) {
        Viewport viewport(640, 480);
        SceneRenderState scene_state;
        RenderSettings settings;
        settings.desaturate_cropping = false;
        const auto ellipsoid_transform = glm::translate(glm::mat4(1.0f), glm::vec3(4.0f, 5.0f, 6.0f));
        const FrameContext ctx{
            .viewport = viewport,
            .scene_state = scene_state,
            .settings = settings,
            .render_size = {640, 480},
            .gizmo =
                {.cropbox_active = true,
                 .cropbox_min = {-1.0f, -1.0f, -1.0f},
                 .cropbox_max = {1.0f, 1.0f, 1.0f},
                 .cropbox_transform = glm::mat4(1.0f),
                 .cropbox_affects_render = true,
                 .cropbox_parent_node_index = 0,
                 .ellipsoid_active = true,
                 .ellipsoid_radii = {3.0f, 4.0f, 5.0f},
                 .ellipsoid_transform = ellipsoid_transform,
                 .ellipsoid_affects_render = true,
                 .ellipsoid_parent_node_index = 0},
        };
        const std::vector<glm::mat4> transforms{glm::mat4(1.0f)};

        const auto request = buildPointCloudRenderRequest(ctx, {640, 480}, transforms);

        ASSERT_TRUE(request.filters.crop_box.has_value());
        EXPECT_FALSE(request.filters.crop_ellipsoid.has_value());
        EXPECT_EQ(request.filters.crop_box->min, glm::vec3(-1.0f, -1.0f, -1.0f));
        EXPECT_EQ(request.filters.crop_box->max, glm::vec3(1.0f, 1.0f, 1.0f));
        EXPECT_EQ(request.filters.crop_box->transform, glm::mat4(1.0f));
        EXPECT_FALSE(request.filters.crop_inverse);
        EXPECT_FALSE(request.filters.crop_desaturate);
    }

    TEST(ViewportRequestBuilderTest, PointCloudRequestKeepsActiveCropBoxBehavior) {
        Viewport viewport(640, 480);
        SceneRenderState scene_state;
        RenderSettings settings;
        settings.desaturate_cropping = true;
        const auto cropbox_transform = glm::translate(glm::mat4(1.0f), glm::vec3(1.0f, 0.0f, 0.0f));
        const FrameContext ctx{
            .viewport = viewport,
            .scene_state = scene_state,
            .settings = settings,
            .render_size = {640, 480},
            .gizmo =
                {.cropbox_active = true,
                 .cropbox_min = {-2.0f, -3.0f, -4.0f},
                 .cropbox_max = {2.0f, 3.0f, 4.0f},
                 .cropbox_transform = cropbox_transform,
                 .cropbox_affects_render = true,
                 .cropbox_parent_node_index = -1},
        };
        const std::vector<glm::mat4> transforms{glm::mat4(1.0f)};

        const auto request = buildPointCloudRenderRequest(ctx, {640, 480}, transforms);

        ASSERT_TRUE(request.filters.crop_box.has_value());
        EXPECT_FALSE(request.filters.crop_ellipsoid.has_value());
        EXPECT_EQ(request.filters.crop_box->min, glm::vec3(-2.0f, -3.0f, -4.0f));
        EXPECT_EQ(request.filters.crop_box->max, glm::vec3(2.0f, 3.0f, 4.0f));
        EXPECT_EQ(request.filters.crop_box->transform, glm::inverse(cropbox_transform));
        EXPECT_FALSE(request.filters.crop_inverse);
        EXPECT_TRUE(request.filters.crop_desaturate);
    }

    TEST(ViewportRequestBuilderTest, PointCloudRequestCarriesSelectionOverlay) {
        using lfs::core::DataType;
        using lfs::core::Device;
        using lfs::core::Tensor;

        Viewport viewport(640, 480);
        SceneRenderState scene_state;
        scene_state.selection_mask = std::make_shared<Tensor>(
            Tensor::zeros({size_t{2}}, Device::CPU, DataType::UInt8));
        scene_state.selection_mask->ptr<std::uint8_t>()[0] = 1;

        Tensor preview_selection =
            Tensor::zeros({size_t{2}}, Device::CPU, DataType::UInt8);
        preview_selection.ptr<std::uint8_t>()[1] = 1;

        RenderSettings settings;
        settings.selection_color_committed = {0.25f, 0.5f, 0.75f};
        settings.selection_color_preview = {0.1f, 0.9f, 0.2f};
        settings.voxel_size = 0.02f;

        const FrameContext ctx{
            .viewport = viewport,
            .scene_state = scene_state,
            .settings = settings,
            .render_size = {640, 480},
            .viewport_pos = {0, 0},
            .cursor_preview =
                {.add_mode = false,
                 .preview_selection = &preview_selection},
        };
        const std::vector<glm::mat4> transforms{glm::mat4(1.0f)};

        const auto request = buildPointCloudRenderRequest(ctx, {640, 480}, transforms);

        EXPECT_EQ(request.overlay.selection_mask, scene_state.selection_mask);
        EXPECT_EQ(request.overlay.transient_mask.mask, &preview_selection);
        EXPECT_FALSE(request.overlay.transient_mask.additive);
        EXPECT_EQ(request.overlay.selection_colors[1], glm::vec4(settings.selection_color_committed, 1.0f));
        EXPECT_EQ(request.overlay.selection_colors[lfs::rendering::kSelectionPreviewColorIndex],
                  glm::vec4(settings.selection_color_preview, 1.0f));
        EXPECT_EQ(request.render.voxel_size, settings.voxel_size);
    }

    // Catches a preview timer that never reports the next refresh (the idle loop
    // then refreshes only on unrelated redraws), reports it as due right away
    // (the loop spins), or leaves the last training state unshown after a pause.
    TEST(ViewportFrameLifecycleServiceTest, TrainingRefreshRunsOnItsOwnIntervalAndOnceAfterStopping) {
        ViewportFrameLifecycleService service;
        constexpr float interval = 0.05f;
        EXPECT_EQ(service.handleTrainingRefresh(true, interval), DirtyFlag::SPLATS);
        EXPECT_EQ(service.handleTrainingRefresh(true, interval), 0u);
        const double wait = service.secondsUntilTrainingRefresh(interval);
        EXPECT_GT(wait, 0.0);
        EXPECT_LE(wait, static_cast<double>(interval));
        std::this_thread::sleep_for(std::chrono::duration<double>(wait + 0.005));
        EXPECT_EQ(service.secondsUntilTrainingRefresh(interval), 0.0);
        EXPECT_EQ(service.handleTrainingRefresh(true, interval), DirtyFlag::SPLATS);

        EXPECT_EQ(service.handleTrainingRefresh(false, interval), DirtyFlag::SPLATS);
        EXPECT_EQ(service.handleTrainingRefresh(false, interval), 0u);
    }

    TEST(ViewportFrameLifecycleServiceTest, RestRenderRestartsTheIdlePreviewBudget) {
        ViewportFrameLifecycleService service;
        constexpr float interval = 0.05f;
        EXPECT_EQ(service.handleTrainingRefresh(true, interval), DirtyFlag::SPLATS);
        // Motion deferred the requested preview until the camera came to rest.
        std::this_thread::sleep_for(std::chrono::milliseconds(60));
        service.restartTrainingRefresh();
        EXPECT_EQ(service.handleTrainingRefresh(true, interval), 0u);
        EXPECT_GT(service.secondsUntilTrainingRefresh(interval), 0.0);
        // A pause still publishes the final training state immediately.
        EXPECT_EQ(service.handleTrainingRefresh(false, interval), DirtyFlag::SPLATS);
    }

    TEST(ViewportFrameLifecycleServiceTest, ResizeActiveDefersFullRefreshUntilDebounceCompletes) {
        ViewportFrameLifecycleService service;

        const auto initial_resize = service.handleViewportResize({640, 480});
        EXPECT_EQ(initial_resize.dirty, DirtyFlag::VIEWPORT | DirtyFlag::CAMERA | DirtyFlag::OVERLAY);
        EXPECT_FALSE(initial_resize.completed);

        EXPECT_EQ(service.setViewportResizeActive(true), 0u);

        const auto active_resize = service.handleViewportResize({800, 600});
        EXPECT_EQ(active_resize.dirty, DirtyFlag::OVERLAY);
        EXPECT_FALSE(active_resize.completed);
        EXPECT_TRUE(active_resize.render_resized_frame);
        EXPECT_TRUE(active_resize.use_interactive_render_scale);
        EXPECT_FALSE(active_resize.require_immediate_output_resize);
        EXPECT_TRUE(service.isResizeDeferring());

        EXPECT_EQ(service.setViewportResizeActive(false),
                  DirtyFlag::VIEWPORT | DirtyFlag::CAMERA | DirtyFlag::OVERLAY);

        const auto debounce_step_1 = service.handleViewportResize({800, 600});
        EXPECT_EQ(debounce_step_1.dirty, DirtyFlag::OVERLAY);
        EXPECT_FALSE(debounce_step_1.completed);

        const auto debounce_step_2 = service.handleViewportResize({800, 600});
        EXPECT_EQ(debounce_step_2.dirty, DirtyFlag::OVERLAY);
        EXPECT_FALSE(debounce_step_2.completed);

        waitUntilResizeSettleReady(service);
        const auto debounce_step_3 = service.handleViewportResize({800, 600});
        EXPECT_EQ(debounce_step_3.dirty, DirtyFlag::VIEWPORT | DirtyFlag::CAMERA);
        EXPECT_TRUE(debounce_step_3.completed);
        EXPECT_FALSE(service.isResizeDeferring());
    }

    TEST(ViewportFrameLifecycleServiceTest, PassiveWindowResizeDefersFullRefreshUntilDebounceCompletes) {
        ViewportFrameLifecycleService service;

        EXPECT_EQ(service.handleViewportResize({640, 480}).dirty,
                  DirtyFlag::VIEWPORT | DirtyFlag::CAMERA | DirtyFlag::OVERLAY);

        const auto passive_resize = service.handleViewportResize({800, 600});
        EXPECT_EQ(passive_resize.dirty, DirtyFlag::OVERLAY);
        EXPECT_FALSE(passive_resize.completed);
        EXPECT_TRUE(passive_resize.render_resized_frame);
        EXPECT_TRUE(passive_resize.use_interactive_render_scale);
        EXPECT_FALSE(passive_resize.require_immediate_output_resize);
        EXPECT_TRUE(service.isResizeDeferring());

        EXPECT_EQ(service.handleViewportResize({800, 600}).dirty, DirtyFlag::OVERLAY);
        EXPECT_EQ(service.handleViewportResize({800, 600}).dirty, DirtyFlag::OVERLAY);

        waitUntilResizeSettleReady(service);
        const auto completed = service.handleViewportResize({800, 600});
        EXPECT_EQ(completed.dirty, DirtyFlag::VIEWPORT | DirtyFlag::CAMERA);
        EXPECT_TRUE(completed.completed);
        EXPECT_FALSE(service.isResizeDeferring());
    }

    TEST(ViewportFrameLifecycleServiceTest, DiscreteLayoutResizeRendersAtFullResolution) {
        ViewportFrameLifecycleService service;

        EXPECT_EQ(service.handleViewportResize({640, 480}).dirty,
                  DirtyFlag::VIEWPORT | DirtyFlag::CAMERA | DirtyFlag::OVERLAY);
        EXPECT_EQ(service.setViewportResizeActive(
                      true, ViewportResizeRenderPolicy::FullResolution),
                  0u);

        const auto layout_resize = service.handleViewportResize({960, 480});
        EXPECT_EQ(layout_resize.dirty, DirtyFlag::OVERLAY);
        EXPECT_TRUE(layout_resize.render_resized_frame);
        EXPECT_FALSE(layout_resize.use_interactive_render_scale);
        EXPECT_TRUE(layout_resize.require_immediate_output_resize);
        EXPECT_TRUE(service.isResizeDeferring());

        const auto guarded_frame = service.handleViewportResize({960, 480});
        EXPECT_FALSE(guarded_frame.render_resized_frame);
        EXPECT_TRUE(guarded_frame.require_immediate_output_resize);

        EXPECT_EQ(service.setViewportResizeActive(false),
                  DirtyFlag::VIEWPORT | DirtyFlag::CAMERA | DirtyFlag::OVERLAY);
    }

    TEST(ViewportFrameLifecycleServiceTest, ExplicitRefreshDeferralCompletesAfterStableFrames) {
        ViewportFrameLifecycleService service;

        EXPECT_EQ(service.handleViewportResize({640, 480}).dirty,
                  DirtyFlag::VIEWPORT | DirtyFlag::CAMERA | DirtyFlag::OVERLAY);
        EXPECT_EQ(service.deferViewportRefresh(), DirtyFlag::OVERLAY);
        EXPECT_TRUE(service.isResizeDeferring());

        EXPECT_EQ(service.handleViewportResize({640, 480}).dirty, DirtyFlag::OVERLAY);
        EXPECT_EQ(service.handleViewportResize({640, 480}).dirty, DirtyFlag::OVERLAY);

        waitUntilResizeSettleReady(service);
        const auto completed = service.handleViewportResize({640, 480});
        EXPECT_EQ(completed.dirty, DirtyFlag::VIEWPORT | DirtyFlag::CAMERA);
        EXPECT_TRUE(completed.completed);
        EXPECT_FALSE(service.isResizeDeferring());
    }

    TEST(ViewportFrameLifecycleServiceTest, ModelChangeClearsCachedViewportArtifactsOncePerModelPointer) {
        ViewportFrameLifecycleService service;
        ViewportArtifactService artifacts;

        const auto generation_before = artifacts.artifactGeneration();
        const auto first_change = service.handleModelChange(0x1234, artifacts);
        EXPECT_TRUE(first_change.changed);
        EXPECT_EQ(first_change.previous_model_ptr, 0u);
        EXPECT_GT(artifacts.artifactGeneration(), generation_before);

        const auto generation_after_first_change = artifacts.artifactGeneration();
        const auto repeated_change = service.handleModelChange(0x1234, artifacts);
        EXPECT_FALSE(repeated_change.changed);
        EXPECT_EQ(artifacts.artifactGeneration(), generation_after_first_change);
    }

    TEST(ViewportFrameLifecycleServiceTest, ModelSourceChangesInvalidateOnceInBothDirections) {
        using Source = ViewportFrameLifecycleService::ModelSource;
        ViewportFrameLifecycleService service;
        ViewportArtifactService artifacts;

        EXPECT_TRUE(service.handleModelChange(0x1234, artifacts, Source::Scene).changed);
        auto generation = artifacts.artifactGeneration();
        EXPECT_TRUE(service.handleModelChange(0x1234, artifacts, Source::Training).changed);
        EXPECT_GT(artifacts.artifactGeneration(), generation);
        generation = artifacts.artifactGeneration();
        EXPECT_FALSE(service.handleModelChange(0x1234, artifacts, Source::Training).changed);
        EXPECT_EQ(artifacts.artifactGeneration(), generation);
        EXPECT_TRUE(service.handleModelChange(0x1234, artifacts, Source::Scene).changed);
        EXPECT_GT(artifacts.artifactGeneration(), generation);
        generation = artifacts.artifactGeneration();
        EXPECT_FALSE(service.handleModelChange(0x1234, artifacts, Source::Scene).changed);
        EXPECT_EQ(artifacts.artifactGeneration(), generation);

        service.resetModelTracking();
        EXPECT_TRUE(service.handleModelChange(0x1234, artifacts, Source::Training).changed);
    }

    enum class DepthStorage { CPU,
                              CUDA,
                              Vulkan };
    class ViewportDepthBackendTest : public ::testing::TestWithParam<DepthStorage> {};

    TEST_P(ViewportDepthBackendTest, SamplesStridedDepthOnTheStorageBackend) {
        const bool on_cpu = GetParam() == DepthStorage::CPU;
        const auto backend = GetParam() == DepthStorage::Vulkan ? core::GpuBackend::Vulkan
                                                                : core::GpuBackend::CUDA;
        if (!on_cpu && !core::gpu_backend_available(backend)) {
            GTEST_SKIP() << "Requested GPU backend unavailable";
        }
        core::GpuBackendScope scope(backend);
        auto depth = core::Tensor::from_vector(
            {99.0f, 2.0f, 3.0f, 99.0f, 99.0f, 4.0f, 5.0f, 99.0f},
            {1, 2, 4}, core::Device::CPU);
        if (!on_cpu) {
            depth = depth.gpu();
        }
        depth = depth.slice(2, 1, 3);
        ASSERT_FALSE(depth.is_contiguous());
        const auto original_backend = core::gpu_backend_of(depth);

        lfs::rendering::FrameMetadata metadata{};
        metadata.valid = true;
        metadata.depth_panel_count = 1;
        metadata.depth_panels[0].depth = std::make_shared<core::Tensor>(std::move(depth));
        ViewportArtifactService artifacts;
        artifacts.updateFromImageOutput({}, metadata, {4, 4}, true);

        // The selected default is independent of the storage being sampled.
        core::GpuBackendScope other(backend == core::GpuBackend::CUDA ? core::GpuBackend::Vulkan
                                                                      : core::GpuBackend::CUDA);
        EXPECT_FLOAT_EQ(artifacts.sampleLinearDepthAt(0, 0, {4, 4}), 2.0f);
        EXPECT_FLOAT_EQ(artifacts.sampleLinearDepthAt(3, 0, {4, 4}), 3.0f);
        EXPECT_FLOAT_EQ(artifacts.sampleLinearDepthAt(0, 3, {4, 4}), 4.0f);
        EXPECT_FLOAT_EQ(artifacts.sampleLinearDepthAt(3, 3, {4, 4}), 5.0f);
        EXPECT_EQ(core::gpu_backend_of(*metadata.depth_panels[0].depth), original_backend);
    }

    INSTANTIATE_TEST_SUITE_P(StorageBackends, ViewportDepthBackendTest,
                             ::testing::Values(DepthStorage::CPU, DepthStorage::CUDA, DepthStorage::Vulkan));

    TEST(ViewportArtifactServiceTest, ExplicitSplitPanelSamplingUsesPanelLocalCoordinates) {
        ViewportArtifactService artifacts;

        auto left_depth = lfs::core::Tensor::from_vector(
                              std::vector<float>(512, 1.0f),
                              {size_t{1}, size_t{1}, size_t{512}},
                              lfs::core::Device::CPU)
                              .gpu();
        auto right_values = std::vector<float>(512, 2.0f);
        right_values[256] = 42.0f;
        auto right_depth = lfs::core::Tensor::from_vector(
                               right_values,
                               {size_t{1}, size_t{1}, size_t{512}},
                               lfs::core::Device::CPU)
                               .gpu();

        const lfs::rendering::FrameMetadata metadata{
            .depth_panels =
                {lfs::rendering::FramePanelMetadata{
                     .depth = std::make_shared<lfs::core::Tensor>(std::move(left_depth)),
                     .start_position = 0.0f,
                     .end_position = 0.5f,
                 },
                 lfs::rendering::FramePanelMetadata{
                     .depth = std::make_shared<lfs::core::Tensor>(std::move(right_depth)),
                     .start_position = 0.5f,
                     .end_position = 1.0f,
                 }},
            .depth_panel_count = 2,
            .valid = true,
            .depth_is_ndc = false,
        };
        artifacts.updateFromImageOutput({}, metadata, {1024, 1}, false);

        EXPECT_FLOAT_EQ(
            artifacts.sampleLinearDepthAt(256, 0, {1024, 1}, SplitViewPanelId::Right),
            42.0f);
    }

    TEST(ViewportFrameLifecycleServiceTest, MissingViewportOutputForcesFreshRedraw) {
        ViewportFrameLifecycleService service;

        EXPECT_EQ(
            service.requiredDirtyMask(false, true, SplitViewMode::Disabled),
            DirtyFlag::ALL);
        EXPECT_EQ(
            service.requiredDirtyMask(false, false, SplitViewMode::PLYComparison),
            DirtyFlag::ALL | DirtyFlag::SPLIT_VIEW);
        EXPECT_EQ(
            service.requiredDirtyMask(true, true, SplitViewMode::PLYComparison),
            0u);
    }

    TEST(ViewportRequestBuilderTest, CursorPreviewTargetsOnlyItsSplitPanel) {
        Viewport viewport;
        RenderSettings settings;
        FrameContext ctx{
            .viewport = viewport,
            .settings = settings,
            .render_size = {800, 600},
            .cursor_preview =
                {.active = true,
                 .x = 120.0f,
                 .y = 80.0f,
                 .radius = 24.0f,
                 .add_mode = true,
                 .panel = SplitViewPanelId::Right},
        };

        const auto left_request = buildViewportRenderRequest(
            ctx, {400, 600}, &ctx.viewport, SplitViewPanelId::Left);
        const auto right_request = buildViewportRenderRequest(
            ctx, {400, 600}, &ctx.viewport, SplitViewPanelId::Right);

        EXPECT_FALSE(left_request.overlay.cursor.enabled);
        EXPECT_TRUE(right_request.overlay.cursor.enabled);
    }

    TEST(ViewportRequestBuilderTest, TrainingSuppressesInteractiveSelectionOverlayButKeepsRenderMarkers) {
        using lfs::core::DataType;
        using lfs::core::Device;
        using lfs::core::Tensor;

        Viewport viewport(640, 480);
        SceneRenderState scene_state;
        scene_state.selection_mask = std::make_shared<Tensor>(
            Tensor::zeros({size_t{2}}, Device::CPU, DataType::UInt8));
        scene_state.selected_node_mask = {true, false};

        Tensor preview_selection =
            Tensor::zeros({size_t{2}}, Device::CPU, DataType::UInt8);

        RenderSettings settings;
        settings.show_rings = true;
        settings.show_center_markers = true;
        settings.desaturate_unselected = true;
        settings.selection_color_center_marker = glm::vec3(0.25f, 0.5f, 0.75f);

        const FrameContext ctx{
            .viewport = viewport,
            .scene_state = scene_state,
            .settings = settings,
            .render_size = {640, 480},
            .viewport_pos = {0, 0},
            .training_active = true,
            .cursor_preview =
                {.active = true,
                 .x = 120.0f,
                 .y = 80.0f,
                 .radius = 24.0f,
                 .add_mode = true,
                 .preview_selection = &preview_selection,
                 .focused_gaussian_id = 1,
                 .selection_mode = SelectionPreviewMode::Rings},
            .selection_flash_intensity = 0.75f,
        };

        const auto request = buildViewportRenderRequest(ctx, {640, 480});

        EXPECT_TRUE(request.overlay.markers.show_rings);
        EXPECT_TRUE(request.overlay.markers.show_center_markers);
        EXPECT_EQ(request.overlay.selection_colors[0], glm::vec4(settings.selection_color_center_marker, 1.0f));
        EXPECT_FALSE(request.overlay.cursor.enabled);
        EXPECT_EQ(request.overlay.emphasis.mask, nullptr);
        EXPECT_EQ(request.overlay.emphasis.transient_mask.mask, nullptr);
        EXPECT_FALSE(request.overlay.emphasis.transient_mask.additive);
        EXPECT_TRUE(request.overlay.emphasis.emphasized_node_mask.empty());
        EXPECT_FALSE(request.overlay.emphasis.dim_non_emphasized);
        EXPECT_FLOAT_EQ(request.overlay.emphasis.flash_intensity, 0.0f);
        EXPECT_EQ(request.overlay.emphasis.focused_gaussian_id, -1);

        const std::vector<glm::mat4> transforms{glm::mat4(1.0f)};
        const auto point_cloud_request = buildPointCloudRenderRequest(ctx, {640, 480}, transforms);
        EXPECT_EQ(point_cloud_request.overlay.selection_mask, nullptr);
        EXPECT_EQ(point_cloud_request.overlay.transient_mask.mask, nullptr);
    }

    TEST_F(RenderingManagerEventsTest, SceneLoadedDisablesGtComparison) {
        SceneManager scene_manager;
        addDatasetCameraWithImage(scene_manager);
        services().set(&scene_manager);
        lfs::vis::screen::ScreenService manager_views;
        RenderingManager manager{manager_views};
        lfs::core::events::cmd::ToggleGTComparison{}.emit();
        EXPECT_EQ(manager.getSettings().split_view_mode, SplitViewMode::GTComparison);

        lfs::core::events::state::SceneLoaded{
            .scene = nullptr,
            .path = std::filesystem::path{},
            .type = lfs::core::events::state::SceneLoaded::Type::PLY,
            .num_gaussians = 0}
            .emit();

        EXPECT_EQ(manager.getSettings().split_view_mode, SplitViewMode::Disabled);
    }

    TEST_F(RenderingManagerEventsTest, ViewerCannotEnterGtComparisonThroughCommand) {
        services().clear();
        SceneManager scene_manager;
        lfs::vis::screen::ScreenService manager_views;
        RenderingManager manager{manager_views};
        services().set(&scene_manager);
        services().set(&manager);

        lfs::core::events::cmd::ToggleGTComparison{}.emit();

        EXPECT_FALSE(manager.isGTComparisonActive());
        EXPECT_EQ(manager.getSettings().split_view_mode, SplitViewMode::Disabled);
    }

    TEST_F(RenderingManagerEventsTest, SceneClearedDisablesGtComparison) {
        SceneManager scene_manager;
        addDatasetCameraWithImage(scene_manager);
        services().set(&scene_manager);
        lfs::vis::screen::ScreenService manager_views;
        RenderingManager manager{manager_views};
        lfs::core::events::cmd::ToggleGTComparison{}.emit();
        EXPECT_EQ(manager.getSettings().split_view_mode, SplitViewMode::GTComparison);

        lfs::core::events::state::SceneCleared{}.emit();

        EXPECT_EQ(manager.getSettings().split_view_mode, SplitViewMode::Disabled);
    }

    TEST_F(RenderingManagerEventsTest,
           RenderSettingsChangedEquirectangularPreservesSceneBackend) {
        using Backend = lfs::rendering::GaussianRasterBackend;

        lfs::vis::screen::ScreenService manager_views;

        RenderingManager manager{manager_views};
        auto settings = manager.getSettings();
        settings.raster_backend = Backend::ThreeDgs;
        settings.gut = false;
        settings.equirectangular = false;
        manager.updateSettings(settings);

        auto event = lfs::core::events::ui::RenderSettingsChanged{};
        event.equirectangular = true;
        event.emit();

        settings = manager.getSettings();
        EXPECT_TRUE(settings.equirectangular);
        EXPECT_EQ(settings.raster_backend, Backend::ThreeDgs);
        EXPECT_FALSE(settings.gut);
        lfs::vis::enforceProjectionBackend(settings);
        EXPECT_EQ(settings.raster_backend, Backend::ThreeDgut);
        EXPECT_TRUE(settings.gut);
    }

    TEST_F(RenderingManagerEventsTest,
           EnablingDepthFilterMigratesConstructorDefaultPositiveZBox) {
        lfs::vis::screen::ScreenService manager_views;
        RenderingManager manager{manager_views};
        auto settings = manager.getSettings();
        ASSERT_FALSE(settings.depth_filter_enabled);
        ASSERT_EQ(settings.depth_filter_min.z, 0.0f);
        ASSERT_EQ(settings.depth_filter_max.z, 100.0f);

        settings.depth_filter_enabled = true;
        manager.updateSettings(settings);

        settings = manager.getSettings();
        EXPECT_EQ(settings.depth_filter_min.z, -100.0f);
        EXPECT_EQ(settings.depth_filter_max.z, 0.0f);

        manager.updateSettings(settings);
        settings = manager.getSettings();
        EXPECT_EQ(settings.depth_filter_min.z, -100.0f);
        EXPECT_EQ(settings.depth_filter_max.z, 0.0f);
    }

    TEST(OverlayParamPackingTest, GTComparisonRenderCameraReplacesDrawAndContainmentIntrinsics) {
        using lfs::core::Camera;
        using lfs::core::CameraModelType;
        using lfs::core::Device;
        using lfs::core::Tensor;

        Camera camera(
            Tensor::from_vector({1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}, {size_t{3}, size_t{3}}, Device::CPU),
            Tensor::from_vector({0.0f, 0.0f, 0.0f}, {size_t{3}}, Device::CPU),
            500.0f, 600.0f, 320.0f, 240.0f,
            Tensor(), Tensor(), CameraModelType::PINHOLE,
            "test.png", {}, {}, 640, 480, 7);
        glm::mat4 scene_transform(1.0f);
        scene_transform = glm::translate(scene_transform, glm::vec3(1.0f, 2.0f, 3.0f));
        scene_transform = glm::scale(scene_transform, glm::vec3(2.0f, 3.0f, 4.0f));
        const auto render_camera = detail::buildGTRenderCamera(camera, {1280, 960}, scene_transform);
        ASSERT_TRUE(render_camera.has_value());

        lfs::rendering::FrameView frame_view{};
        frame_view.intrinsics_override = lfs::rendering::CameraIntrinsics{
            .focal_x = 111.0f,
            .focal_y = 122.0f,
            .center_x = 133.0f,
            .center_y = 144.0f};
        frame_view.containment_intrinsics = lfs::rendering::CameraIntrinsics{
            .focal_x = 211.0f,
            .focal_y = 222.0f,
            .center_x = 233.0f,
            .center_y = 244.0f};
        bool equirectangular = true;
        applyGTComparisonRenderCamera(frame_view, equirectangular, *render_camera);

        ASSERT_TRUE(frame_view.intrinsics_override.has_value());
        ASSERT_TRUE(frame_view.containment_intrinsics.has_value());
        EXPECT_FLOAT_EQ(frame_view.intrinsics_override->focal_x, 1000.0f);
        EXPECT_FLOAT_EQ(frame_view.intrinsics_override->focal_y, 1200.0f);
        EXPECT_FLOAT_EQ(frame_view.intrinsics_override->center_x, 640.0f);
        EXPECT_FLOAT_EQ(frame_view.intrinsics_override->center_y, 480.0f);
        EXPECT_FLOAT_EQ(frame_view.containment_intrinsics->focal_x, 1000.0f);
        EXPECT_FLOAT_EQ(frame_view.containment_intrinsics->focal_y, 1200.0f);
        EXPECT_FLOAT_EQ(frame_view.containment_intrinsics->center_x, 640.0f);
        EXPECT_FLOAT_EQ(frame_view.containment_intrinsics->center_y, 480.0f);
        EXPECT_FALSE(frame_view.orthographic);
        EXPECT_FLOAT_EQ(frame_view.ortho_scale, lfs::rendering::DEFAULT_ORTHO_SCALE);
        EXPECT_FALSE(equirectangular);
    }

    TEST(OverlayParamPackingTest, OverlayParamsCarryUnjitteredContainmentIntrinsics) {
        lfs::rendering::ViewportRenderRequest request{};
        request.frame_view.size = {640, 480};
        request.frame_view.intrinsics_override = lfs::rendering::CameraIntrinsics{
            .focal_x = 111.0f,
            .focal_y = 122.0f,
            .center_x = 133.25f,
            .center_y = 144.125f};
        request.frame_view.containment_intrinsics = lfs::rendering::CameraIntrinsics{
            .focal_x = 211.0f,
            .focal_y = 222.0f,
            .center_x = 233.0f,
            .center_y = 244.0f};
        request.filters.view_volume = lfs::rendering::BoundingBox{
            .min = glm::vec3(-1.0f),
            .max = glm::vec3(1.0f),
            .transform = glm::mat4(1.0f)};
        request.filters.screen_window = lfs::rendering::SelectionScreenWindow{};

        const auto packed = detail::buildOverlayParamsCpuFloats(request, false, false, false, 0, false);
        ASSERT_TRUE(packed.has_value());
        const std::size_t base = static_cast<std::size_t>(detail::ViewIntrinsics) * 4u;
        EXPECT_FLOAT_EQ((*packed)[base + 0], 211.0f);
        EXPECT_FLOAT_EQ((*packed)[base + 1], 222.0f);
        EXPECT_FLOAT_EQ((*packed)[base + 2], 233.0f);
        EXPECT_FLOAT_EQ((*packed)[base + 3], 244.0f);
    }

    TEST(OverlayParamPackingTest, OverlayParamsSlotTwelveStaysZeroWithoutContainmentIntrinsics) {
        lfs::rendering::ViewportRenderRequest request{};
        request.frame_view.size = {640, 480};
        request.frame_view.intrinsics_override = lfs::rendering::CameraIntrinsics{
            .focal_x = 111.0f,
            .focal_y = 122.0f,
            .center_x = 133.0f,
            .center_y = 144.0f};
        request.filters.view_volume = lfs::rendering::BoundingBox{
            .min = glm::vec3(-1.0f),
            .max = glm::vec3(1.0f),
            .transform = glm::mat4(1.0f)};
        request.filters.screen_window = lfs::rendering::SelectionScreenWindow{};

        const auto packed = detail::buildOverlayParamsCpuFloats(request, false, false, false, 0, false);
        ASSERT_TRUE(packed.has_value());
        const std::size_t base = static_cast<std::size_t>(detail::ViewIntrinsics) * 4u;
        EXPECT_FLOAT_EQ((*packed)[base + 0], 0.0f);
        EXPECT_FLOAT_EQ((*packed)[base + 1], 0.0f);
        EXPECT_FLOAT_EQ((*packed)[base + 2], 0.0f);
        EXPECT_FLOAT_EQ((*packed)[base + 3], 0.0f);
    }

    TEST(OverlayParamPackingTest, OverlayParamsSlotTwelveStaysZeroForOrthographic) {
        lfs::rendering::ViewportRenderRequest request{};
        request.frame_view.size = {640, 480};
        request.frame_view.orthographic = true;
        request.frame_view.containment_intrinsics = lfs::rendering::CameraIntrinsics{
            .focal_x = 211.0f,
            .focal_y = 222.0f,
            .center_x = 233.0f,
            .center_y = 244.0f};
        request.filters.view_volume = lfs::rendering::BoundingBox{
            .min = glm::vec3(-1.0f),
            .max = glm::vec3(1.0f),
            .transform = glm::mat4(1.0f)};
        request.filters.screen_window = lfs::rendering::SelectionScreenWindow{};

        const auto packed = detail::buildOverlayParamsCpuFloats(request, false, false, false, 0, false);
        ASSERT_TRUE(packed.has_value());
        const std::size_t base = static_cast<std::size_t>(detail::ViewIntrinsics) * 4u;
        EXPECT_FLOAT_EQ((*packed)[base + 0], 0.0f);
        EXPECT_FLOAT_EQ((*packed)[base + 1], 0.0f);
        EXPECT_FLOAT_EQ((*packed)[base + 2], 0.0f);
        EXPECT_FLOAT_EQ((*packed)[base + 3], 0.0f);
    }

    TEST(OverlayParamPackingTest, OverlayParamLayoutAndAnisotropicScreenWindowPacking) {
        static_assert(detail::ViewIntrinsics == 12);
        static_assert(detail::ViewWindow == 206);
        static_assert(detail::ParamCount == 207);

        lfs::rendering::ViewportRenderRequest request{};
        request.frame_view.size = {640, 480};
        request.filters.view_volume = lfs::rendering::BoundingBox{
            .min = glm::vec3(-1.0f),
            .max = glm::vec3(1.0f),
            .transform = glm::mat4(1.0f)};
        request.filters.screen_window = lfs::rendering::SelectionScreenWindow{
            .scale_x = 0.25f,
            .scale_y = 0.75f,
            .offset_x = 0.15f,
            .offset_y = -0.35f,
        };

        const auto packed = detail::buildOverlayParamsCpuFloats(request, false, false, false, 0, false);
        ASSERT_TRUE(packed.has_value());

        const std::size_t view_flags_base = static_cast<std::size_t>(detail::ViewFlags) * 4u;
        EXPECT_FLOAT_EQ((*packed)[view_flags_base + 3], 0.0f);

        const std::size_t view_window_base = static_cast<std::size_t>(detail::ViewWindow) * 4u;
        EXPECT_FLOAT_EQ((*packed)[view_window_base + 0], 0.25f);
        EXPECT_FLOAT_EQ((*packed)[view_window_base + 1], 0.75f);
        EXPECT_FLOAT_EQ((*packed)[view_window_base + 2], 0.0f);
        EXPECT_FLOAT_EQ((*packed)[view_window_base + 3], 0.0f);
    }

    TEST(OverlayParamPackingTest, ViewWindowZPacksDepthWindowDragPreviewFlag) {
        const std::size_t view_window_base = static_cast<std::size_t>(detail::ViewWindow) * 4u;

        lfs::rendering::ViewportRenderRequest preview_on{};
        preview_on.frame_view.size = {640, 480};
        preview_on.filters.view_volume = lfs::rendering::BoundingBox{
            .min = glm::vec3(-1.0f),
            .max = glm::vec3(1.0f),
            .transform = glm::mat4(1.0f)};
        preview_on.filters.screen_window = lfs::rendering::SelectionScreenWindow{
            .scale_x = 0.25f,
            .scale_y = 0.75f,
            .drag_preview = true,
        };

        const auto packed_on =
            detail::buildOverlayParamsCpuFloats(preview_on, false, false, false, 0, false);
        ASSERT_TRUE(packed_on.has_value());
        EXPECT_FLOAT_EQ((*packed_on)[view_window_base + 2], 1.0f);
        EXPECT_FLOAT_EQ((*packed_on)[view_window_base + 3], 0.0f);

        lfs::rendering::ViewportRenderRequest preview_off = preview_on;
        preview_off.filters.screen_window->drag_preview = false;

        const auto packed_off =
            detail::buildOverlayParamsCpuFloats(preview_off, false, false, false, 0, false);
        ASSERT_TRUE(packed_off.has_value());
        EXPECT_FLOAT_EQ((*packed_off)[view_window_base + 2], 0.0f);
        EXPECT_FLOAT_EQ((*packed_off)[view_window_base + 3], 0.0f);
    }

    class DepthWindowGtHookTest : public ::testing::Test {
    protected:
        void SetUp() override {
            lfs::event::EventBridge::instance().clear_all();
            lfs::core::event::bus().clear_all();
            lfs::vis::services().clear();
            lfs::vis::op::undoHistory().clear();

            options_.show_startup_overlay = false;
            options_.width = 200;
            options_.height = 200;
            viewer_ = std::make_unique<lfs::vis::VisualizerImpl>(options_);
            viewer_->initializeTools();

            rendering_manager_ = viewer_->getRenderingManager();
            selection_tool_ = viewer_->getSelectionTool();
            ASSERT_NE(rendering_manager_, nullptr);
            ASSERT_NE(selection_tool_, nullptr);

            const std::vector<float> means_data{0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
            const std::vector<float> rotation_data{1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f};
            auto means = lfs::core::Tensor::from_vector(means_data, {size_t{2}, size_t{3}}, lfs::core::Device::GPU).to(lfs::core::DataType::Float32);
            auto sh0 = lfs::core::Tensor::zeros({size_t{2}, size_t{1}, size_t{3}}, lfs::core::Device::GPU, lfs::core::DataType::Float32);
            auto shN = lfs::core::Tensor::zeros({size_t{2}, size_t{3}, size_t{3}}, lfs::core::Device::GPU, lfs::core::DataType::Float32);
            auto scaling = lfs::core::Tensor::zeros({size_t{2}, size_t{3}}, lfs::core::Device::GPU, lfs::core::DataType::Float32);
            auto rotation = lfs::core::Tensor::from_vector(rotation_data, {size_t{2}, size_t{4}}, lfs::core::Device::GPU).to(lfs::core::DataType::Float32);
            auto opacity = lfs::core::Tensor::zeros({size_t{2}, size_t{1}}, lfs::core::Device::GPU, lfs::core::DataType::Float32);
            viewer_->getSceneManager()->getScene().addSplat(
                "depth_window_test",
                std::make_unique<lfs::core::SplatData>(1, std::move(means), std::move(sh0), std::move(shN), std::move(scaling), std::move(rotation), std::move(opacity), 1.0f));
            viewer_->getSceneManager()->initSelectionService();
            if (auto* const service = viewer_->getSceneManager()->getSelectionService()) {
                service->setTestingViewport({
                    .x = 0.0f,
                    .y = 0.0f,
                    .width = static_cast<float>(options_.width),
                    .height = static_cast<float>(options_.height),
                    .render_width = options_.width,
                    .render_height = options_.height,
                });
            }
            selection_tool_->setEnabled(true);

            auto settings = rendering_manager_->getSettings();
            settings.depth_filter_enabled = true;
            settings.depth_filter_scale_x = 0.5f;
            settings.depth_filter_scale_y = 0.5f;
            rendering_manager_->updateSettings(settings);
            selection_tool_->setDepthFilterEnabled(true);
        }

        void TearDown() override {
            lfs::vis::op::operators().cancelModalOperator();
            lfs::event::EventBridge::instance().clear_all();
            lfs::core::event::bus().clear_all();
            lfs::vis::services().clear();
            viewer_.reset();
            lfs::vis::op::undoHistory().clear();
        }

        bool startDepthDrag() {
            lfs::vis::op::OperatorProperties props;
            props.set("x", 10.0);
            props.set("y", 10.0);
            props.set("viewport_x", 0.0f);
            props.set("viewport_y", 0.0f);
            props.set("viewport_width", static_cast<float>(options_.width));
            props.set("viewport_height", static_cast<float>(options_.height));
            props.set("modifiers", lfs::vis::input::KEYMOD_SHIFT | lfs::vis::input::KEYMOD_ALT);
            const auto result = lfs::vis::op::operators().invoke(lfs::vis::op::BuiltinOp::DepthWindowDrag, &props);
            return result.status == lfs::vis::op::OperatorResult::RUNNING_MODAL;
        }

        lfs::vis::ViewerOptions options_{};
        std::unique_ptr<lfs::vis::VisualizerImpl> viewer_;
        lfs::vis::RenderingManager* rendering_manager_ = nullptr;
        lfs::vis::tools::SelectionTool* selection_tool_ = nullptr;
    };

    TEST_F(DepthWindowGtHookTest, GtToggleCancelsDepthWindowDragAndClearsHover) {
        addDatasetCameraWithImage(*viewer_->getSceneManager());
        ASSERT_TRUE(startDepthDrag());
        const op::ModalEvent move{
            .type = op::ModalEvent::Type::MOUSE_MOVE,
            .data = MouseMoveEvent{
                .position = {60.0, 60.0},
                .delta = {50.0, 50.0},
            },
        };
        ASSERT_EQ(op::operators().dispatchModalEvent(move), op::OperatorResult::RUNNING_MODAL);
        ASSERT_TRUE(rendering_manager_->depthWindowDragPreview());
        (void)lfs::vis::op::updateDepthWindowHover(rendering_manager_->activeViewId(),
                                                   glm::vec2(50.0f, 50.0f),
                                                   glm::vec4(0.0f, 0.0f, static_cast<float>(options_.width), static_cast<float>(options_.height)),
                                                   true);

        lfs::core::events::cmd::ToggleGTComparison{}.emit();

        EXPECT_FALSE(rendering_manager_->depthWindowDragPreview());
        EXPECT_FALSE(lfs::vis::op::operators().hasModalOperator());
        EXPECT_FALSE(lfs::vis::op::depthWindowOverlayState().visible);
        EXPECT_EQ(lfs::vis::op::depthWindowOverlayState().hovered_handle, lfs::vis::op::DepthWindowHandle::None);
    }

    TEST_F(DepthWindowGtHookTest, GtToggleLeavesUnrelatedSelectionStrokeModalActive) {
        lfs::vis::op::OperatorProperties stroke_props;
        stroke_props.set("mode", 0);
        stroke_props.set("op", 0);
        stroke_props.set("x", 30.0);
        stroke_props.set("y", 30.0);
        const auto stroke = lfs::vis::op::operators().invoke(lfs::vis::op::BuiltinOp::SelectionStroke, &stroke_props);
        ASSERT_EQ(stroke.status, lfs::vis::op::OperatorResult::RUNNING_MODAL);

        lfs::core::events::cmd::ToggleGTComparison{}.emit();

        EXPECT_TRUE(lfs::vis::op::operators().hasModalOperator());
        EXPECT_EQ(lfs::vis::op::operators().activeModalId(), "selection.stroke");
        lfs::vis::op::operators().cancelModalOperator();
    }

} // namespace lfs::vis
