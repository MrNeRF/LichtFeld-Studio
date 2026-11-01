/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/environment.hpp"
#include "core/event_bridge/event_bridge.hpp"
#include "core/point_cloud.hpp"
#include "core/services.hpp"
#include "io/formats/colmap.hpp"
#include "rendering/camera_interaction_service.hpp"
#include "scene/scene_manager.hpp"
#include "viewer_scale_reference.hpp"
#include <filesystem>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>
#include <memory>
#include <tbb/global_control.h>
#include <thread>
#include <tuple>
#include <vector>

namespace lfs::vis {
    namespace ref = viewer_scale_reference;
    using lfs::core::Camera;
    using lfs::core::Device;
    using lfs::core::Tensor;
    using lfs::rendering::CameraFrustumPickRequest;
    using lfs::rendering::RenderingEngine;

    class ViewerScaleTest : public ::testing::Test {
    protected:
        static inline std::vector<std::shared_ptr<const Camera>> cameras;

        static void SetUpTestSuite() {
            const auto model = lfs::core::environment::value("LFS_TEST_COLMAP_MODEL");
            if (!model)
                return;
            auto loaded = lfs::io::read_colmap_cameras_only(std::filesystem::path(*model));
            ASSERT_TRUE(loaded.has_value()) << loaded.error().format();
            const auto& source = std::get<0>(*loaded);
            ASSERT_GE(source.size(), 32u);
            RecordProperty("source_camera_count", static_cast<int>(source.size()));
            const size_t count = std::min(size_t{4096}, source.size());
            RecordProperty("sample_camera_count", static_cast<int>(count));
            for (size_t i = 0; i < count; ++i)
                cameras.push_back(source[i * source.size() / count]);
        }

        void SetUp() override {
            if (cameras.empty())
                GTEST_SKIP() << "Set LFS_TEST_COLMAP_MODEL to a real COLMAP sparse model";
            lfs::event::EventBridge::instance().clear_all();
            services().clear();
        }

        void TearDown() override {
            services().clear();
            lfs::event::EventBridge::instance().clear_all();
        }

        static void TearDownTestSuite() { cameras.clear(); }

        static std::shared_ptr<Camera> copyCamera(const Camera& camera, const int uid,
                                                  const lfs::core::CameraModelType model = lfs::core::CameraModelType::PINHOLE) {
            return std::make_shared<Camera>(
                camera.R().clone(), camera.T().clone(), camera.focal_x(), camera.focal_y(),
                camera.center_x(), camera.center_y(), camera.radial_distortion(), camera.tangential_distortion(),
                model, "camera", std::filesystem::path{}, std::filesystem::path{},
                camera.camera_width(), camera.camera_height(), uid);
        }

        static CameraFrustumPickRequest requestForFirstCamera() {
            CameraFrustumPickRequest request;
            request.viewport_pos = {31.0f, 47.0f};
            request.viewport_size = {1000.0f, 800.0f};
            request.viewport.size = {1600, 1000};
            request.viewport.focal_length_mm = 35.0f;
            request.viewport.rotation = glm::mat3(1.0f);
            request.viewport.translation = {0.0f, 0.0f, 3.0f};
            request.scale = 0.4f;
            request.scene_transform = glm::inverse(*ref::cameraVisualizerTransform(*cameras.front(), glm::mat4(1.0f)));
            request.camera_generation = 1;
            request.mouse_pos = request.viewport_pos + request.viewport_size * 0.5f;
            return request;
        }

        static std::shared_ptr<const SceneRenderState> cachedState(const SceneManager& manager) {
            return manager.cached_render_state_;
        }

        static std::shared_ptr<lfs::core::PointCloud> pointCloud() {
            std::vector<float> positions;
            for (size_t i = 0; i < 4; ++i) {
                const glm::vec3 position((*ref::cameraVisualizerTransform(*cameras[i], glm::mat4(1.0f)))[3]);
                positions.insert(positions.end(), {position.x, position.y, position.z});
            }
            return std::make_shared<lfs::core::PointCloud>(
                Tensor::from_vector(positions, {size_t{4}, size_t{3}}, Device::CPU),
                Tensor::ones({size_t{4}, size_t{3}}, Device::CPU));
        }
    };

    TEST_F(ViewerScaleTest, PickingMatchesBruteForceAcrossViewsAndCursors) {
        tbb::global_control control(tbb::global_control::max_allowed_parallelism, 4);
        auto engine = RenderingEngine::create();
        std::vector<std::shared_ptr<const Camera>> sample(cameras.begin(), cameras.begin() + std::min(size_t{512}, cameras.size()));
        sample.insert(sample.begin() + 3, nullptr);
        sample.push_back(copyCamera(*cameras[7], 1000000, lfs::core::CameraModelType::EQUIRECTANGULAR));
        RecordProperty("pick_camera_count", static_cast<int>(sample.size()));
        RecordProperty("parallelism_limit", 4);
        auto request = requestForFirstCamera();
        std::vector<glm::mat4> transforms(sample.size(), request.scene_transform);
        for (size_t i = 0; i < transforms.size(); ++i)
            transforms[i] = glm::translate(glm::mat4(1.0f), glm::vec3(0.01f * (i % 4), 0.0f, 0.0f)) * transforms[i];
        request.scene_transforms = transforms;
        size_t hits = 0, misses = 0;
        const auto check = [&] {
            const int expected = ref::pickCameraFrustum(sample, request);
            const auto actual = engine->pickCameraFrustum(sample, request);
            ASSERT_TRUE(actual.has_value());
            ASSERT_EQ(*actual, expected) << "cursor " << request.mouse_pos.x << ',' << request.mouse_pos.y;
            expected >= 0 ? ++hits : ++misses;
        };
        for (int mode = 0; mode < 2; ++mode) {
            request.viewport.orthographic = mode != 0;
            request.viewport.ortho_scale = 140.0f;
            for (int view = 0; view < 6; ++view) {
                request.viewport.rotation = glm::mat3(glm::rotate(glm::mat4(1.0f), 0.21f * view, glm::vec3(0.0f, 1.0f, 0.0f)));
                request.viewport.translation = {0.3f * view, 0.07f * view, view == 5 ? 0.05f : 3.0f};
                for (int y = -1; y <= 10; ++y)
                    for (int x = -1; x <= 16; ++x) {
                        request.mouse_pos = request.viewport_pos + glm::vec2(x * 67.0f, y * 83.0f);
                        ASSERT_NO_FATAL_FAILURE(check());
                    }
                for (size_t i = 0; i < 16; ++i) {
                    if (!sample[i])
                        continue;
                    const auto transform = ref::cameraVisualizerTransform(*sample[i], transforms[i]);
                    ASSERT_TRUE(transform);
                    for (const auto& point : ref::cameraFrustumWorldPoints(*sample[i], *transform, request.scale)) {
                        const auto projected = ref::projectFrustumPoint(point, request);
                        if (!projected)
                            continue;
                        for (const float offset : {0.0f, 4.0f, 11.9995f, 12.001f}) {
                            request.mouse_pos = *projected + glm::vec2(offset, 0.0f);
                            ASSERT_NO_FATAL_FAILURE(check());
                        }
                    }
                }
            }
        }
        RecordProperty("comparisons", static_cast<int>(hits + misses));
        RecordProperty("hits", static_cast<int>(hits));
        RecordProperty("misses", static_cast<int>(misses));
        EXPECT_GT(hits, 100u);
        EXPECT_GT(misses, 100u);
        request.viewport_size.x = 0;
        EXPECT_EQ(*engine->pickCameraFrustum(sample, request), -1);
        EXPECT_EQ(*engine->pickCameraFrustum({}, request), -1);
    }

    TEST_F(ViewerScaleTest, OrderedTieAtHitRadiusMatchesBruteForce) {
        auto engine = RenderingEngine::create();
        std::vector<std::shared_ptr<const Camera>> sample{copyCamera(*cameras.front(), 10), copyCamera(*cameras.front(), 11)};
        auto request = requestForFirstCamera();
        std::vector<glm::mat4> transforms{
            request.scene_transform,
            glm::translate(glm::mat4(1.0f), glm::vec3(1.0f, 0.0f, 1.0f)) * request.scene_transform};
        request.scene_transforms = transforms;
        const auto transform = ref::cameraVisualizerTransform(*sample[0], transforms[0]);
        const auto points = ref::cameraFrustumWorldPoints(*sample[0], *transform, request.scale);
        const auto top = ref::projectFrustumPoint(points[2], request);
        const auto bottom = ref::projectFrustumPoint(points[3], request);
        ASSERT_TRUE(top && bottom);
        request.mouse_pos = (*top + *bottom) * 0.5f + glm::vec2(11.9995f, 0.0f);
        ASSERT_EQ(ref::pickCameraFrustum({sample[0]}, request), 10);
        ASSERT_EQ(ref::pickCameraFrustum(sample, request), -1);
        EXPECT_EQ(*engine->pickCameraFrustum(sample, request), -1);
    }

    TEST_F(ViewerScaleTest, EqualGenerationReusesFrustumsAndNewGenerationReadsMovedPose) {
        auto engine = RenderingEngine::create();
        auto camera = copyCamera(*cameras.front(), 10);
        std::vector<std::shared_ptr<const Camera>> sample{camera};
        auto request = requestForFirstCamera();
        ASSERT_EQ(*engine->pickCameraFrustum(sample, request), 10);
        auto translation = camera->T();
        translation.add_(10000.0f);
        ASSERT_EQ(ref::pickCameraFrustum(sample, request), -1);
        EXPECT_EQ(*engine->pickCameraFrustum(sample, request), 10);
        ++request.camera_generation;
        EXPECT_EQ(*engine->pickCameraFrustum(sample, request), -1);
        translation.copy_(cameras.front()->T());
        request.camera_generation = 0;
        EXPECT_EQ(*engine->pickCameraFrustum(sample, request), 10);
        translation.add_(10000.0f);
        EXPECT_EQ(*engine->pickCameraFrustum(sample, request), -1);
    }

    TEST_F(ViewerScaleTest, ScaleAndBothKindsOfSceneTransformInvalidateCache) {
        auto engine = RenderingEngine::create();
        const std::vector<std::shared_ptr<const Camera>> sample{cameras.front()};
        auto request = requestForFirstCamera();
        const auto transform = ref::cameraVisualizerTransform(*sample.front(), request.scene_transform);
        const auto points = ref::cameraFrustumWorldPoints(*sample.front(), *transform, request.scale);
        request.mouse_pos = *ref::projectFrustumPoint(points[2], request);
        ASSERT_EQ(*engine->pickCameraFrustum(sample, request), sample.front()->uid());
        request.scale = 0.01f;
        ASSERT_EQ(ref::pickCameraFrustum(sample, request), -1);
        EXPECT_EQ(*engine->pickCameraFrustum(sample, request), -1);
        request = requestForFirstCamera();
        ASSERT_EQ(*engine->pickCameraFrustum(sample, request), sample.front()->uid());
        const auto original = request.scene_transform;
        request.scene_transform = glm::translate(glm::mat4(1.0f), glm::vec3(100.0f, 0.0f, 0.0f)) * original;
        EXPECT_EQ(*engine->pickCameraFrustum(sample, request), -1);
        std::vector<glm::mat4> transforms{original};
        request.scene_transforms = transforms;
        EXPECT_EQ(*engine->pickCameraFrustum(sample, request), sample.front()->uid());
        transforms.front() = request.scene_transform;
        EXPECT_EQ(*engine->pickCameraFrustum(sample, request), -1);
        transforms.front() = original;
        EXPECT_EQ(*engine->pickCameraFrustum(sample, request), sample.front()->uid());
        request.scene_transforms = {};
        EXPECT_EQ(*engine->pickCameraFrustum(sample, request), -1);
    }

    TEST_F(ViewerScaleTest, EqualGenerationDoesNotAliasDifferentCameraSetsOrOrder) {
        auto engine = RenderingEngine::create();
        const auto first = copyCamera(*cameras.front(), 10);
        const auto second = copyCamera(*cameras.front(), 11);
        auto request = requestForFirstCamera();
        std::vector<std::shared_ptr<const Camera>> sample{first};
        ASSERT_EQ(*engine->pickCameraFrustum(sample, request), 10);
        sample[0] = second;
        EXPECT_EQ(*engine->pickCameraFrustum(sample, request), 11);
        sample.push_back(first);
        EXPECT_EQ(*engine->pickCameraFrustum(sample, request), 11);
        std::reverse(sample.begin(), sample.end());
        EXPECT_EQ(*engine->pickCameraFrustum(sample, request), 10);
        sample[0] = nullptr;
        EXPECT_EQ(*engine->pickCameraFrustum(sample, request), 11);
    }

    TEST_F(ViewerScaleTest, PickCacheDoesNotRetainCameraStorage) {
        auto engine = RenderingEngine::create();
        std::weak_ptr<const Camera> lifetime;
        {
            const auto camera = copyCamera(*cameras.front(), 10);
            lifetime = camera;
            EXPECT_EQ(*engine->pickCameraFrustum({camera}, requestForFirstCamera()), 10);
        }
        EXPECT_TRUE(lifetime.expired());
    }

    TEST_F(ViewerScaleTest, InteractionRefreshesTransformsWithTheSceneCameraSnapshot) {
        SceneManager manager;
        auto& scene = manager.getScene();
        const auto camera = copyCamera(*cameras.front(), 10);
        const auto group = scene.addCameraGroup("cameras", lfs::core::NULL_NODE, 1);
        scene.addCamera("camera", group, camera);
        const auto request = requestForFirstCamera();
        const auto transform = lfs::rendering::visualizerWorldTransformToDataWorld(request.scene_transform);
        scene.setNodeTransform(group, transform);
        const auto snapshot = scene.getVisibleCamerasCached();
        EXPECT_EQ(snapshot, scene.getVisibleCamerasCached());
        ViewportInteractionContext context;
        const std::array panels{ViewportInteractionPanel{
            .viewport_data = request.viewport,
            .viewport_pos = request.viewport_pos,
            .viewport_size = request.viewport_size}};
        context.updatePickContext(panels);
        RenderSettings settings;
        settings.show_camera_frustums = true;
        settings.camera_frustum_scale = request.scale;
        auto engine = RenderingEngine::create();
        CameraInteractionService service;
        bool changed = false;
        EXPECT_EQ(service.pickCameraFrustum(engine.get(), &manager, context, settings, request.mouse_pos, changed), 10);
        EXPECT_TRUE(changed);
        scene.setNodeTransform(group, glm::translate(glm::mat4(1.0f), glm::vec3(10000.0f, 0.0f, 0.0f)) * transform);
        EXPECT_NE(snapshot, scene.getVisibleCamerasCached());
        std::this_thread::sleep_for(CameraInteractionService::PICK_THROTTLE_INTERVAL + std::chrono::milliseconds(1));
        EXPECT_EQ(service.pickCameraFrustum(engine.get(), &manager, context, settings, request.mouse_pos, changed), -1);
        EXPECT_TRUE(changed);
        scene.setNodeTransform(group, transform);
        std::this_thread::sleep_for(CameraInteractionService::PICK_THROTTLE_INTERVAL + std::chrono::milliseconds(1));
        EXPECT_EQ(service.pickCameraFrustum(engine.get(), &manager, context, settings, request.mouse_pos, changed), 10);
    }

    namespace {
        void expectEqualGeometry(const gui::CameraFrustumGeometry& actual, const gui::CameraFrustumGeometry& expected) {
            EXPECT_EQ(actual.instance_camera_indices, expected.instance_camera_indices);
            EXPECT_EQ(actual.textured_camera_indices, expected.textured_camera_indices);
            EXPECT_EQ(actual.projected_visible, expected.projected_visible);
            ASSERT_EQ(actual.data->frustum_instances.size(), expected.data->frustum_instances.size());
            for (size_t i = 0; i < actual.data->frustum_instances.size(); ++i) {
                EXPECT_EQ(actual.data->frustum_instances[i].model, expected.data->frustum_instances[i].model);
                EXPECT_EQ(actual.data->frustum_instances[i].color, expected.data->frustum_instances[i].color);
            }
            ASSERT_EQ(actual.data->textured_overlays.size(), expected.data->textured_overlays.size());
            for (size_t i = 0; i < actual.data->textured_overlays.size(); ++i) {
                const auto& a = actual.data->textured_overlays[i];
                const auto& b = expected.data->textured_overlays[i];
                EXPECT_EQ(a.texture_id, b.texture_id);
                EXPECT_EQ(a.tint_opacity, b.tint_opacity);
                EXPECT_EQ(a.effects, b.effects);
                for (size_t j = 0; j < a.vertices.size(); ++j) {
                    EXPECT_EQ(a.vertices[j].position, b.vertices[j].position);
                    EXPECT_EQ(a.vertices[j].uv, b.vertices[j].uv);
                    EXPECT_EQ(a.vertices[j].view_depth, b.vertices[j].view_depth);
                }
            }
            ASSERT_EQ(actual.data->frustum_batches.size(), expected.data->frustum_batches.size());
            for (size_t i = 0; i < actual.data->frustum_batches.size(); ++i) {
                const auto& a = actual.data->frustum_batches[i];
                const auto& b = expected.data->frustum_batches[i];
                EXPECT_EQ(a.view, b.view);
                EXPECT_EQ(a.viewport_pos, b.viewport_pos);
                EXPECT_EQ(a.viewport_size, b.viewport_size);
                EXPECT_EQ(a.render_size, b.render_size);
                EXPECT_EQ(a.focal_x, b.focal_x);
                EXPECT_EQ(a.focal_y, b.focal_y);
                EXPECT_EQ(a.orthographic, b.orthographic);
                EXPECT_EQ(a.equirectangular, b.equirectangular);
                EXPECT_EQ(a.first_instance, b.first_instance);
                EXPECT_EQ(a.instance_count, b.instance_count);
            }
        }
    } // namespace

    TEST_F(ViewerScaleTest, ParallelOverlayMatchesSequentialInstancesThumbnailsAndLineOrder) {
        gui::CameraFrustumGeometry inputs;
        const auto request = requestForFirstCamera();
        for (size_t i = 0; i < cameras.size(); ++i) {
            const auto& camera = *cameras[i];
            const auto transform = *ref::cameraVisualizerTransform(camera, request.scene_transform);
            const float half_height = std::tan(camera.FoVy() * 0.5f) * request.scale;
            const float half_width = half_height * camera.camera_width() / camera.camera_height();
            inputs.models.push_back(transform * glm::scale(glm::mat4(1.0f), glm::vec3(2.0f * half_width, 2.0f * half_height, request.scale)));
            inputs.positions.push_back(glm::vec3(transform[3]));
            inputs.uids.push_back(camera.uid());
            inputs.valid_cameras.push_back(i % 13 != 1);
            inputs.equirectangular_cameras.push_back(i % 31 == 0);
            inputs.validation_cameras.push_back(i % 3 == 0);
            inputs.disabled_cameras.push_back(i % 5 == 0);
            inputs.emphasized_cameras.push_back(i % 7 == 0);
            inputs.thumbnail_indices.push_back(i % 4 == 0 ? -1 : static_cast<int>(i));
            inputs.thumbnail_placements.push_back({.texture_id = i + 1, .uv_min = {0.1f, 0.2f}, .uv_max = {0.8f, 0.9f}});
            inputs.loss_colors.push_back(i % 2 ? glm::vec3(0.2f, 0.3f, 0.4f) : glm::vec3(std::numeric_limits<float>::quiet_NaN()));
        }
        inputs.projected_points.resize(2 * cameras.size());
        inputs.projected_depths.resize(2 * cameras.size());
        inputs.projected_visible.resize(2 * cameras.size(), 0);
        inputs.camera_colors.resize(2 * cameras.size());
        size_t instances = 0, thumbnails = 0, lines = 0;
        for (const size_t workers : {size_t{1}, size_t{4}}) {
            tbb::global_control control(tbb::global_control::max_allowed_parallelism, workers);
            for (int mode = 0; mode < 3; ++mode) {
                auto actual = inputs;
                auto expected = inputs;
                actual.data = std::make_shared<VulkanViewportFrustumOverlayData>();
                expected.data = std::make_shared<VulkanViewportFrustumOverlayData>();
                RenderSettings settings;
                settings.camera_frustum_scale = request.scale;
                settings.orthographic = mode == 1;
                settings.equirectangular = mode == 2;
                settings.ortho_scale = 160.0f;
                VulkanViewportPassParams params;
                params.viewport_pos = {20, 30};
                params.viewport_size = {1600, 900};
                std::vector<std::tuple<glm::vec3, glm::vec3, glm::vec4>> actual_lines, expected_lines;
                for (size_t panel_index = 0; panel_index < 2; ++panel_index) {
                    Viewport viewport(1000, 800);
                    viewport.setViewMatrix(glm::mat3(glm::rotate(glm::mat4(1.0f), 0.1f * panel_index, glm::vec3(0, 1, 0))),
                                           {0.2f * panel_index, 0.0f, 3.0f});
                    const gui::VulkanGuidePanelTarget panel{
                        .viewport = &viewport,
                        .pos = {20.0f + panel_index * 800.0f, 30.0f},
                        .size = {800.0f, 900.0f},
                        .render_size = {1000, 800}};
                    gui::appendCameraFrustumPanel(params, actual, panel, panel_index, settings, cameras[2]->uid(),
                                                  [&](const auto& a, const auto& b, const auto& color) { actual_lines.emplace_back(a, b, color); });
                    ref::appendCameraFrustumPanelSequential(params, expected, panel, panel_index, settings, cameras[2]->uid(),
                                                            [&](const auto& a, const auto& b, const auto& color) { expected_lines.emplace_back(a, b, color); });
                }
                expectEqualGeometry(actual, expected);
                EXPECT_EQ(actual_lines, expected_lines);
                EXPECT_EQ(actual.has_equirectangular_lines, expected.has_equirectangular_lines);
                instances += actual.data->frustum_instances.size();
                thumbnails += actual.data->textured_overlays.size();
                lines += actual_lines.size();
            }
        }
        RecordProperty("instances_compared", static_cast<int>(instances));
        RecordProperty("thumbnails_compared", static_cast<int>(thumbnails));
        RecordProperty("line_segments_compared", static_cast<int>(lines));
        EXPECT_GT(instances, 0u);
        EXPECT_GT(thumbnails, 0u);
        EXPECT_GT(lines, 0u);
    }

    TEST_F(ViewerScaleTest, SinglePointCloudReusesStateAndReflectsInPlaceEdits) {
        SceneManager manager;
        auto cloud = pointCloud();
        const auto id = manager.getScene().addPointCloud("cloud", cloud);
        const auto first = manager.buildRenderState();
        const auto cached = cachedState(manager);
        ASSERT_EQ(first.point_cloud, cloud.get());
        ASSERT_FALSE(first.owned_point_cloud);
        cloud->means.add_(1.0f);
        const auto second = manager.buildRenderState();
        EXPECT_EQ(cachedState(manager), cached);
        EXPECT_EQ(second.point_cloud, cloud.get());
        EXPECT_EQ(second.point_cloud->means.to_vector(), cloud->means.to_vector());
        manager.getScene().setNodeTransform(id, glm::translate(glm::mat4(1.0f), glm::vec3(1.0f, 2.0f, 3.0f)));
        const auto moved = manager.buildRenderState();
        EXPECT_NE(cachedState(manager), cached);
        EXPECT_NE(moved.point_cloud_transform, first.point_cloud_transform);
        const auto before_replace = cachedState(manager);
        auto replacement = pointCloud();
        manager.getScene().getNodeById(id)->point_cloud = replacement;
        EXPECT_EQ(manager.buildRenderState().point_cloud, replacement.get());
        EXPECT_NE(cachedState(manager), before_replace);
    }

    TEST_F(ViewerScaleTest, TwoPointCloudsRebuildMergeAndReturnToSharedStateWhenHidden) {
        SceneManager manager;
        const auto left = pointCloud();
        const auto right = pointCloud();
        manager.getScene().addPointCloud("left", left);
        const auto right_id = manager.getScene().addPointCloud("right", right);
        const auto first = manager.buildRenderState();
        const auto cached = cachedState(manager);
        ASSERT_NE(first.owned_point_cloud, nullptr);
        EXPECT_EQ(first.point_cloud->size(), 8u);
        const auto before = first.point_cloud->means.to_vector();
        right->means.add_(2.0f);
        const auto second = manager.buildRenderState();
        EXPECT_NE(cachedState(manager), cached);
        EXPECT_NE(second.owned_point_cloud, first.owned_point_cloud);
        EXPECT_NE(second.point_cloud->means.to_vector(), before);
        EXPECT_EQ(first.point_cloud->means.to_vector(), before);
        manager.getScene().setNodeVisibility(right_id, false);
        const auto single = manager.buildRenderState();
        const auto shared = cachedState(manager);
        ASSERT_EQ(single.point_cloud, left.get());
        EXPECT_FALSE(single.owned_point_cloud);
        manager.buildRenderState();
        EXPECT_EQ(cachedState(manager), shared);
    }
} // namespace lfs::vis
