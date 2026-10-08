/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/event_bridge/event_bridge.hpp"
#include "core/event_bus.hpp"
#include "core/point_cloud.hpp"
#include "core/services.hpp"
#include "operation/undo_history.hpp"
#include "operator/ops/align_ops.hpp"
#if LFS_GRAPHICS_VULKAN && !defined(LFS_TENSOR_METAL)
#include "rendering/point_cloud_vulkan_renderer.hpp"
#include "window/vulkan_graphics_context.hpp"
#endif
#include "rendering/rendering_manager.hpp"
#include "rendering/viewport_artifact_service.hpp"
#include "scene/scene_manager.hpp"
#include "screen/screen_service.hpp"
#include "visualizer/app_store.hpp"

#include <SDL3/SDL.h>
#include <cmath>
#include <cstdlib>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>
#include <limits>

namespace {

    [[nodiscard]] glm::vec3 applyRot(const glm::mat4& m, const glm::vec3& v) {
        return glm::mat3(m) * v;
    }

    TEST(AlignDepth, PointDepthSamplerFollowsPublishedOutput) {
        lfs::vis::ViewportArtifactService artifacts;
        artifacts.setDepthSampler([](int x, int y, auto, bool) { return x == 2 && y == 3 ? 4.0f : -1.0f; });
        EXPECT_FLOAT_EQ(artifacts.sampleLinearDepthAt(2, 3, {10, 10}), 4.0f);
        EXPECT_FLOAT_EQ(artifacts.sampleLinearDepthAt(0, 0, {10, 10}), -1.0f);
        artifacts.updateFromImageOutput({}, {}, {10, 10}, true);
        EXPECT_FALSE(artifacts.hasDepthSampler());
        artifacts.setDepthSampler([](int, int, auto, bool) { return 5.0f; });
        artifacts.clearViewportOutput();
        EXPECT_FALSE(artifacts.hasDepthSampler());
    }

#if LFS_GRAPHICS_VULKAN && !defined(LFS_TENSOR_METAL)
    void verifyPointCloudDepthWithoutGaussians() {
        lfs::vis::VulkanGraphicsContext graphics;
        ASSERT_TRUE(graphics.initializeHeadless());
        auto& context = graphics.vulkanContext();
        lfs::vis::PointCloudVulkanRenderer renderer;
        auto positions = lfs::core::Tensor::from_vector(std::vector<float>{0, 0, -5},
                                                        {std::size_t{1}, std::size_t{3}}, lfs::core::Device::CPU);
        auto colors = lfs::core::Tensor::from_vector(std::vector<float>{1, 1, 1},
                                                     {std::size_t{1}, std::size_t{3}}, lfs::core::Device::CPU);
        const lfs::vis::RenderTargetId target{1};
        for (const bool orthographic : {false, true}) {
            lfs::vis::PointCloudVulkanRenderer::RenderRequest request;
            request.positions = &positions;
            request.colors = &colors;
            request.size = {64, 64};
            request.orthographic = orthographic;
            request.ortho_scale = 10.0f;
            request.focal_y = 32.0f;
            request.voxel_size = 0.5f;
            request.view_projection = orthographic
                                          ? glm::ortho(-5.0f, 5.0f, -5.0f, 5.0f, 0.1f, 8.0f)
                                          : glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 100.0f);
            ASSERT_TRUE(renderer.render(context, request, target));
            const auto depth = renderer.sampleDepthAtPixel(context, {.pixel = {32, 32}, .source_size = {64, 64}, .target = target});
            ASSERT_TRUE(depth) << lfs::format_for_developer(depth.error());
            EXPECT_NEAR(*depth, 5.0f, 1e-3f);
            lfs::vis::PointCloudVulkanRenderer::DepthSampleRequest async_request{
                .pixel = {32, 32},
                .source_size = {64, 64},
                .target = target,
                .nonblocking = true};
            const auto pending = renderer.sampleDepthAtPixel(context, async_request);
            ASSERT_TRUE(pending);
            EXPECT_EQ(*pending, lfs::vis::PointCloudVulkanRenderer::kDepthSamplePending);
            EXPECT_TRUE(renderer.takeRefinementRequest());
            ASSERT_TRUE(context.deviceWaitIdle());
            const auto completed = renderer.sampleDepthAtPixel(context, async_request);
            ASSERT_TRUE(completed);
            EXPECT_NEAR(*completed, 5.0f, 1e-3f);
            for (int i = 0; i < 1000; ++i) {
                async_request.pixel = i % 2 ? glm::ivec2(0, 0) : glm::ivec2(32, 32);
                const auto sample = renderer.sampleDepthAtPixel(context, async_request);
                ASSERT_TRUE(sample);
                if (i % 2)
                    EXPECT_LT(*sample, 0.0f);
                else
                    EXPECT_NEAR(*sample, 5.0f, 1e-3f);
            }
            EXPECT_FALSE(renderer.takeRefinementRequest());
            // Overlay-only redraws must keep the completed snapshot usable.
            ASSERT_TRUE(renderer.render(context, request, target));
            async_request.pixel = {32, 32};
            EXPECT_NEAR(renderer.sampleDepthAtPixel(context, async_request).value(), 5.0f, 1e-3f);
            EXPECT_FALSE(renderer.takeRefinementRequest());
            // A changed camera must never reuse the old surface snapshot.
            request.view = glm::translate(glm::mat4(1.0f), glm::vec3(0, 0, -1));
            request.view_projection *= request.view;
            ASSERT_TRUE(renderer.render(context, request, target));
            EXPECT_EQ(renderer.sampleDepthAtPixel(context, async_request).value(), lfs::vis::PointCloudVulkanRenderer::kDepthSamplePending);
            ASSERT_TRUE(context.deviceWaitIdle());
            EXPECT_NEAR(renderer.sampleDepthAtPixel(context, async_request).value(), 6.0f, 1e-3f);
            (void)renderer.takeRefinementRequest();
            const auto background = renderer.sampleDepthAtPixel(context, {.pixel = {0, 0}, .source_size = {64, 64}, .target = target});
            ASSERT_TRUE(background);
            EXPECT_LT(*background, 0.0f);
            EXPECT_FALSE(renderer.sampleDepthAtPixel(context, {.pixel = {64, 0}, .source_size = {64, 64}, .target = target}));
            // Resize/replacement must not reuse the old mapped image or coordinates.
            request.size = {32, 32};
            request.focal_y = 16.0f;
            ASSERT_TRUE(renderer.render(context, request, target));
            async_request.pixel = {16, 16};
            async_request.source_size = {32, 32};
            EXPECT_EQ(renderer.sampleDepthAtPixel(context, async_request).value(), lfs::vis::PointCloudVulkanRenderer::kDepthSamplePending);
            ASSERT_TRUE(context.deviceWaitIdle());
            EXPECT_NEAR(renderer.sampleDepthAtPixel(context, async_request).value(), 6.0f, 1e-3f);
            (void)renderer.takeRefinementRequest();

            // Resetting with a queued copy must retire its images and staging storage.
            const auto temporary_target = lfs::vis::RenderTargetId{2};
            ASSERT_TRUE(renderer.render(context, request, temporary_target));
            async_request.target = temporary_target;
            EXPECT_EQ(renderer.sampleDepthAtPixel(context, async_request).value(), lfs::vis::PointCloudVulkanRenderer::kDepthSamplePending);
            renderer.reset();
            EXPECT_FALSE(renderer.sampleDepthAtPixel(context, async_request));
            ASSERT_TRUE(context.deviceWaitIdle());
            (void)renderer.takeRefinementRequest();
        }
    }

    TEST(AlignDepth, VulkanPointCloudDepthMatchesSurfaceWithoutGaussians) {
        if (!std::getenv("LFS_ASYNC_GPU_TESTS"))
            GTEST_SKIP() << "Set LFS_ASYNC_GPU_TESTS=1 on a GPU host";
        // Device adoption must precede the first Vulkan tensor in the process.
        GTEST_FLAG_SET(death_test_style, "threadsafe");
        EXPECT_EXIT({
            verifyPointCloudDepthWithoutGaussians();
            std::_Exit(::testing::Test::HasFailure() ? 1 : 0); }, ::testing::ExitedWithCode(0), "");
    }
#endif

    TEST(AlignPanelOrthoScale, PanelZoomUsesEachViewsOwnSettings) {
        using namespace lfs::vis;
        screen::ScreenService source;
        const auto first = source.screen().activeView();
        const auto second = source.screen().split(first, screen::SplitAxis::Columns, 0.5f);
        ASSERT_TRUE(second.valid());
        RenderingManager renderer(source);
        source.editViewSettings(first.value, [](ViewSettings& settings) {
            settings.orthographic = true;
            settings.ortho_scale = 100.0f;
        });
        source.editViewSettings(second.value, [](ViewSettings& settings) {
            settings.orthographic = true;
            settings.ortho_scale = 400.0f;
        });
        EXPECT_FLOAT_EQ(op::alignPanelOrthoScale(renderer, first.value), 100.0f);
        EXPECT_FLOAT_EQ(op::alignPanelOrthoScale(renderer, second.value), 400.0f);
        source.screen().setActiveView(second);
        EXPECT_FLOAT_EQ(op::alignPanelOrthoScale(renderer, first.value), 100.0f);
        EXPECT_FLOAT_EQ(op::alignPanelOrthoScale(renderer, renderer.activeViewId()), 400.0f);
    }

    TEST(AlignEdgeToWorldX, IdentityUpMapsPositiveZEdgeToPlusX) {
        const glm::mat4 up(1.0f);
        const glm::vec3 p0(0.0f, 0.0f, 0.0f);
        const glm::vec3 p1(0.0f, 0.0f, 2.0f); // +Z edge

        const glm::mat4 yaw = lfs::vis::op::alignEdgeToWorldXRotation(up, p0, p1);
        const glm::mat4 total = yaw * up;
        const glm::vec3 rotated = applyRot(total, p1 - p0);

        EXPECT_NEAR(rotated.x, 2.0f, 1e-5f);
        EXPECT_NEAR(rotated.y, 0.0f, 1e-5f);
        EXPECT_NEAR(rotated.z, 0.0f, 1e-5f);
    }

    TEST(AlignEdgeToWorldX, IdentityUpMapsNegativeXEdgeToPlusX) {
        const glm::mat4 up(1.0f);
        const glm::vec3 p0(1.0f, 0.0f, 0.0f);
        const glm::vec3 p1(-1.0f, 0.0f, 0.0f); // -X edge length 2

        const glm::mat4 yaw = lfs::vis::op::alignEdgeToWorldXRotation(up, p0, p1);
        const glm::vec3 rotated = applyRot(yaw * up, p1 - p0);

        EXPECT_NEAR(rotated.x, 2.0f, 1e-5f);
        EXPECT_NEAR(rotated.y, 0.0f, 1e-5f);
        EXPECT_NEAR(rotated.z, 0.0f, 1e-5f);
    }

    TEST(AlignEdgeToWorldX, DegenerateVerticalEdgeYieldsIdentity) {
        const glm::mat4 up(1.0f);
        const glm::vec3 p0(0.0f, 0.0f, 0.0f);
        const glm::vec3 p1(0.0f, 3.0f, 0.0f); // pure +Y — zero ground projection

        const glm::mat4 yaw = lfs::vis::op::alignEdgeToWorldXRotation(up, p0, p1);
        for (int col = 0; col < 4; ++col) {
            for (int row = 0; row < 4; ++row) {
                const float expected = (row == col) ? 1.0f : 0.0f;
                EXPECT_NEAR(yaw[col][row], expected, 1e-6f);
            }
        }
    }

    TEST(AlignEdgeToWorldX, ComposesWithUpRotation) {
        // Plane originally facing +Z; rotate normal to +Y then yaw edge to +X.
        const glm::vec3 p0(0.0f, 0.0f, 0.0f);
        const glm::vec3 p1(1.0f, 0.0f, 0.0f);
        const glm::vec3 p2(0.0f, 1.0f, 0.0f);
        const glm::vec3 normal = glm::normalize(glm::cross(p1 - p0, p2 - p0)); // +Z
        const glm::vec3 target_up(0.0f, 1.0f, 0.0f);
        const glm::vec3 axis = glm::normalize(glm::cross(normal, target_up));
        const float angle = std::acos(glm::clamp(glm::dot(normal, target_up), -1.0f, 1.0f));
        const glm::mat4 up_rot = glm::rotate(glm::mat4(1.0f), angle, axis);

        const glm::mat4 yaw = lfs::vis::op::alignEdgeToWorldXRotation(up_rot, p0, p1);
        const glm::mat4 total = yaw * up_rot;
        const glm::vec3 rotated_edge = applyRot(total, p1 - p0);
        const glm::vec3 rotated_normal = applyRot(total, normal);

        EXPECT_NEAR(rotated_normal.x, 0.0f, 1e-5f);
        EXPECT_NEAR(rotated_normal.y, 1.0f, 1e-5f);
        EXPECT_NEAR(rotated_normal.z, 0.0f, 1e-5f);
        EXPECT_GT(rotated_edge.x, 0.5f);
        EXPECT_NEAR(rotated_edge.z, 0.0f, 1e-5f);
        EXPECT_NEAR(rotated_edge.y, 0.0f, 1e-4f);
    }

    TEST(AlignTransform, NormalMapsToPlusY) {
        lfs::vis::op::AlignTransformInputs in;
        in.p0 = glm::vec3(0.0f, 1.0f, 0.0f);
        in.p1 = glm::vec3(2.0f, 1.4f, 0.3f);
        in.p2 = glm::vec3(0.4f, 0.2f, 1.7f);
        in.camera_pos = glm::vec3(0.0f, 8.0f, 4.0f);

        const auto xform = lfs::vis::op::computeAlignTransform(in);
        ASSERT_TRUE(xform.has_value());

        glm::vec3 normal = glm::normalize(glm::cross(in.p1 - in.p0, in.p2 - in.p0));
        lfs::vis::op::faceNormalTowardCamera(normal, (in.p0 + in.p1 + in.p2) / 3.0f, in.camera_pos);
        const glm::vec3 transformed = glm::mat3(*xform) * normal;
        EXPECT_NEAR(transformed.x, 0.0f, 1e-5f);
        EXPECT_NEAR(transformed.y, 1.0f, 1e-5f);
        EXPECT_NEAR(transformed.z, 0.0f, 1e-5f);
    }

    TEST(AlignTransform, CentroidLandsOnGround) {
        lfs::vis::op::AlignTransformInputs in;
        in.p0 = glm::vec3(1.0f, 2.0f, 3.0f);
        in.p1 = glm::vec3(2.0f, 2.5f, 3.4f);
        in.p2 = glm::vec3(1.4f, 1.2f, 4.1f);
        in.camera_pos = glm::vec3(1.5f, 9.0f, 3.5f);

        const auto xform = lfs::vis::op::computeAlignTransform(in);
        ASSERT_TRUE(xform.has_value());

        const glm::vec3 center = (in.p0 + in.p1 + in.p2) / 3.0f;
        const glm::vec3 transformed = glm::vec3((*xform) * glm::vec4(center, 1.0f));
        EXPECT_NEAR(transformed.y, 0.0f, 1e-5f);
        EXPECT_NEAR(transformed.x, center.x, 1e-5f);
        EXPECT_NEAR(transformed.z, center.z, 1e-5f);
    }

    TEST(AlignTransform, AntiParallelNormalRotates180) {
        lfs::vis::op::AlignTransformInputs in;
        in.p0 = glm::vec3(0.0f, 1.0f, 0.0f);
        in.p1 = glm::vec3(1.0f, 1.0f, 0.0f);
        in.p2 = glm::vec3(0.0f, 1.0f, 1.0f);
        in.camera_pos = glm::vec3(0.0f, -5.0f, 0.0f);

        const auto xform = lfs::vis::op::computeAlignTransform(in);
        ASSERT_TRUE(xform.has_value());

        glm::vec3 normal = glm::normalize(glm::cross(in.p1 - in.p0, in.p2 - in.p0));
        lfs::vis::op::faceNormalTowardCamera(normal, (in.p0 + in.p1 + in.p2) / 3.0f, in.camera_pos);
        EXPECT_LT(normal.y, 0.0f);

        const glm::vec3 transformed = glm::normalize(glm::mat3(*xform) * normal);
        EXPECT_NEAR(transformed.x, 0.0f, 1e-5f);
        EXPECT_NEAR(transformed.y, 1.0f, 1e-5f);
        EXPECT_NEAR(transformed.z, 0.0f, 1e-5f);
    }

    TEST(AlignTransform, AxisSnapWithinTolerance) {
        glm::vec3 normal = glm::normalize(glm::vec3(1.0f, std::tan(glm::radians(2.0f)), 0.0f));
        EXPECT_TRUE(lfs::vis::op::snapAlignNormalToNodeAxes(normal, glm::mat4(1.0f)));
        EXPECT_NEAR(normal.x, 1.0f, 1e-5f);
        EXPECT_NEAR(normal.y, 0.0f, 1e-5f);
        EXPECT_NEAR(normal.z, 0.0f, 1e-5f);
    }

    TEST(AlignTransform, AxisSnapOutsideTolerance) {
        glm::vec3 normal = glm::normalize(glm::vec3(1.0f, std::tan(glm::radians(5.0f)), 0.0f));
        const glm::vec3 original = normal;
        EXPECT_FALSE(lfs::vis::op::snapAlignNormalToNodeAxes(normal, glm::mat4(1.0f)));
        EXPECT_NEAR(normal.x, original.x, 1e-6f);
        EXPECT_NEAR(normal.y, original.y, 1e-6f);
        EXPECT_NEAR(normal.z, original.z, 1e-6f);
    }

    TEST(AlignTransform, EdgeToWorldXComposes) {
        lfs::vis::op::AlignTransformInputs in;
        in.p0 = glm::vec3(0.0f, 0.0f, 0.0f);
        in.p1 = glm::vec3(0.0f, 0.0f, 2.0f);
        in.p2 = glm::vec3(1.0f, 0.0f, 0.0f);
        in.camera_pos = glm::vec3(0.0f, 5.0f, 0.0f);
        in.edge_to_world_x = true;

        const auto xform = lfs::vis::op::computeAlignTransform(in);
        ASSERT_TRUE(xform.has_value());

        const glm::vec3 edge = glm::mat3(*xform) * (in.p1 - in.p0);
        EXPECT_NEAR(edge.z, 0.0f, 1e-5f);
        EXPECT_GT(edge.x, 0.0f);
    }

    TEST(AlignTransform, DegenerateReturnsNullopt) {
        lfs::vis::op::AlignTransformInputs in;
        in.p0 = glm::vec3(0.0f, 0.0f, 0.0f);
        in.p1 = glm::vec3(1.0f, 0.0f, 0.0f);
        in.p2 = glm::vec3(2.0f, 0.0f, 0.0f);
        in.camera_pos = glm::vec3(0.0f, 5.0f, 0.0f);
        EXPECT_FALSE(lfs::vis::op::computeAlignTransform(in).has_value());
    }

    TEST(AlignTransform, NearlyVerticalNormalsStillLevelThePlane) {
        for (const float slope : {0.0001f, 0.001f, -0.0001f, -0.001f}) {
            for (const float camera_y : {-5.0f, 5.0f}) {
                lfs::vis::op::AlignTransformInputs in;
                in.p0 = {0.0f, 1.0f, 0.0f};
                in.p1 = {1.0f, 1.0f + slope, 0.0f};
                in.p2 = {0.0f, 1.0f, 1.0f};
                in.camera_pos = {0.0f, camera_y, 0.0f};
                const auto transform = lfs::vis::op::computeAlignTransform(in);
                ASSERT_TRUE(transform);
                for (const auto point : {in.p0, in.p1, in.p2}) {
                    EXPECT_NEAR((*transform * glm::vec4(point, 1.0f)).y, 0.0f, 1e-6f)
                        << "slope=" << slope << " camera_y=" << camera_y;
                }
            }
        }
    }

    TEST(AlignTransform, NonFinitePointsCannotProduceATransform) {
        for (const float invalid : {std::numeric_limits<float>::quiet_NaN(),
                                    std::numeric_limits<float>::infinity()}) {
            lfs::vis::op::AlignTransformInputs in;
            in.p0 = {invalid, 0.0f, 0.0f};
            in.p1 = {1.0f, 0.0f, 0.0f};
            in.p2 = {0.0f, 0.0f, 1.0f};
            EXPECT_FALSE(lfs::vis::op::computeAlignTransform(in));
            EXPECT_FALSE(lfs::vis::op::pointsAreNonDegenerate({in.p0, in.p1, in.p2}));
        }
    }

} // namespace

namespace lfs::vis::op {

    class AlignPreviewTest : public ::testing::Test {
    protected:
        void SetUp() override {
            services().clear();
            undoHistory().clear();
            manager_ = std::make_unique<SceneManager>();
            services().set(manager_.get());
            manager_->getScene().addGroup("target");
            manager_->selectNode("target");
            context_ = std::make_unique<OperatorContext>(*manager_);
            operation_.picked_points_ = {{0, 2, 0}, {1, 2, 1}, {0, 3, 1}};
            operation_.syncPickedPointsToServices();
            services().setAlignCameraPosition({0, 5, 5});
            services().setAlignAxisSnapEnabled(false);
            services().setAlignEdgeToAxisEnabled(false);
        }

        void TearDown() override {
            operation_.cancel(*context_);
            undoHistory().clear();
            services().clear();
            context_.reset();
            manager_.reset();
            lfs::event::EventBridge::instance().clear_all();
            lfs::core::event::bus().clear_all();
        }

        OperatorResult action(Services::AlignUiAction action) {
            services().requestAlignUiAction(action);
            OperatorProperties properties;
            return operation_.modal(*context_, properties);
        }

        glm::mat4 transform() const { return manager_->getScene().getNode("target")->transform(); }

        std::unique_ptr<SceneManager> manager_;
        std::unique_ptr<OperatorContext> context_;
        AlignPickPointOperator operation_;
    };

    TEST_F(AlignPreviewTest, AlignmentChangesNotifyUiOncePerFrame) {
        auto& store = app_store();
        (void)store.store().drain_dirty_into_frame();
        int notifications = 0;
        auto token = store.align_state_generation.subscribe([&](const auto&) { ++notifications; });
        const auto generation = store.align_state_generation.get();

        // Repeated synchronization of the same triangle must not keep the UI awake.
        services().setAlignPickedPoints(services().getAlignPickedPoints());
        services().setAlignPreviewEnabled(services().getAlignPreviewEnabled());
        services().setAlignAxisSnapEnabled(services().getAlignAxisSnapEnabled());
        services().setAlignEdgeToAxisEnabled(services().getAlignEdgeToAxisEnabled());
        EXPECT_EQ(store.align_state_generation.get(), generation);

        services().setAlignPickedPoints({{0, 0, 0}, {1, 0, 0}, {0, 0, 1}});
        services().setAlignPreviewEnabled(true);
        services().setAlignAxisSnapEnabled(true);
        services().setAlignEdgeToAxisEnabled(true);
        EXPECT_EQ(notifications, 0);
        EXPECT_TRUE(store.store().drain_dirty_into_frame());
        EXPECT_EQ(notifications, 1);

        services().clearAlignPickedPoints();
        EXPECT_TRUE(store.store().drain_dirty_into_frame());
        EXPECT_EQ(notifications, 2);
        services().clearAlignPickedPoints();
        EXPECT_FALSE(store.store().drain_dirty_into_frame());
        EXPECT_EQ(notifications, 2);
    }

    TEST_F(AlignPreviewTest, ClearingStatusWithoutPickedPointsNotifiesUi) {
        services().clearAlignPickedPoints();
        services().setAlignStatusMessage("No surface at the selected point", 5.0);
        auto& store = app_store();
        (void)store.store().drain_dirty_into_frame();
        const auto generation = store.align_state_generation.get();
        ASSERT_NE(services().getAlignStatusMessage(), nullptr);

        services().clearAlignPickedPoints();
        EXPECT_EQ(services().getAlignStatusMessage(), nullptr);
        EXPECT_EQ(store.align_state_generation.get(), generation + 1);
        EXPECT_TRUE(store.store().drain_dirty_into_frame());

        services().clearAlignPickedPoints();
        EXPECT_EQ(store.align_state_generation.get(), generation + 1);
        EXPECT_FALSE(store.store().drain_dirty_into_frame());
    }

    TEST_F(AlignPreviewTest, PointCloudCanAlignBeforeTraining) {
        auto& scene = manager_->getScene();
        const auto dataset = scene.addDataset("dataset");
        auto means = core::Tensor::from_vector(std::vector<float>{0, 2, 0, 1, 2, 1, 0, 3, 1},
                                               {std::size_t{3}, std::size_t{3}}, core::Device::CPU);
        auto colors = core::Tensor::from_vector(std::vector<float>(9, 1.0f),
                                                {std::size_t{3}, std::size_t{3}}, core::Device::CPU);
        scene.addPointCloud("points", std::make_shared<core::PointCloud>(std::move(means), std::move(colors)), dataset);
        manager_->selectNode("points");
        ASSERT_EQ(scene.getTotalGaussianCount(), 0u);
        ASSERT_TRUE(operation_.poll(*context_));
        const auto original = scene.getNode("points")->transform();
        EXPECT_EQ(action(Services::AlignUiAction::Apply), OperatorResult::FINISHED);
        EXPECT_NE(scene.getNode("points")->transform(), original);
        ASSERT_TRUE(undoHistory().undo().success);
        EXPECT_EQ(scene.getNode("points")->transform(), original);
        manager_->selectNode("dataset");
        EXPECT_TRUE(operation_.poll(*context_));
        scene.setNodeLocked("dataset", true);
        EXPECT_FALSE(operation_.poll(*context_));
        manager_->clearSelection();
        EXPECT_FALSE(operation_.poll(*context_));
    }

    TEST_F(AlignPreviewTest, CancelRestoresOriginalTransformAfterTargetIsLocked) {
        const auto original = transform();
        action(Services::AlignUiAction::TogglePreview);
        ASSERT_NE(transform(), original);
        manager_->getScene().setNodeLocked("target", true);
        operation_.cancel(*context_);
        EXPECT_EQ(transform(), original);
        EXPECT_TRUE(static_cast<bool>(manager_->getScene().getNode("target")->locked));
        EXPECT_FALSE(services().getAlignPreviewEnabled());
    }

    TEST_F(AlignPreviewTest, OptionTogglesUseOriginalTransformWithoutDrift) {
        const auto original = transform();
        action(Services::AlignUiAction::TogglePreview);
        const auto without_axis = transform();
        services().setAlignEdgeToAxisEnabled(true);
        action(Services::AlignUiAction::RefreshPreview);
        EXPECT_NE(transform(), without_axis);
        services().setAlignEdgeToAxisEnabled(false);
        action(Services::AlignUiAction::RefreshPreview);
        EXPECT_EQ(transform(), without_axis);
        action(Services::AlignUiAction::TogglePreview);
        EXPECT_EQ(transform(), original);
        EXPECT_EQ(undoHistory().undoCount(), 0u);
    }

    TEST_F(AlignPreviewTest, ApplyWithoutPreviewMatchesPreviewAndCreatesOneUndoStep) {
        const auto original = transform();
        action(Services::AlignUiAction::TogglePreview);
        const auto preview = transform();
        action(Services::AlignUiAction::TogglePreview);
        ASSERT_EQ(transform(), original);
        EXPECT_EQ(action(Services::AlignUiAction::Apply), OperatorResult::FINISHED);
        EXPECT_EQ(transform(), preview);
        EXPECT_FALSE(services().getAlignPreviewEnabled());
        EXPECT_EQ(undoHistory().undoCount(), 1u);
        EXPECT_TRUE(undoHistory().undo().success);
        EXPECT_EQ(transform(), original);
    }

    TEST_F(AlignPreviewTest, ApplyCommitsOneUndoStepMatchingThePreview) {
        const auto original = transform();
        action(Services::AlignUiAction::TogglePreview);
        const auto preview = transform();
        EXPECT_EQ(action(Services::AlignUiAction::Apply), OperatorResult::FINISHED);
        EXPECT_EQ(transform(), preview);
        EXPECT_EQ(undoHistory().undoCount(), 1u);
        EXPECT_TRUE(undoHistory().undo().success);
        EXPECT_EQ(transform(), original);
        EXPECT_TRUE(undoHistory().redo().success);
        EXPECT_EQ(transform(), preview);
    }

} // namespace lfs::vis::op
