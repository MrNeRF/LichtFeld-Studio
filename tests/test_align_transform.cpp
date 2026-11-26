/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/event_bridge/event_bridge.hpp"
#include "core/point_cloud.hpp"
#include "core/services.hpp"
#include "operation/undo_history.hpp"
#include "operator/operator_registry.hpp"
#include "operator/ops/align_ops.hpp"
#include "operator/ops/edit_ops.hpp"
#include "operator/ops/model_alignment_ops.hpp"
#include "rendering/point_cloud_vulkan_renderer.hpp"
#include "rendering/viewport_artifact_service.hpp"
#include "rendering/viewport_request_builder.hpp"
#include "scene/scene_manager.hpp"
#include "visualizer/app_store.hpp"
#include "visualizer/scene_coordinate_utils.hpp"

#include <SDL3/SDL.h>
#include <cmath>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>
#include <limits>

#include "gui/gizmo_manager.hpp"
#include "gui/ui_context.hpp"
#include "licht_test_support.hpp"
#include "tools/align_tool.hpp"
#include "tools/unified_tool_registry.hpp"
#include "visualizer/visualizer_impl.hpp"
#include <cstdlib>

namespace lfs::vis {
    class AlignDepthIntegrationTest : public ::testing::Test {
    protected:
        void SetUp() override {
            if (const auto* home = std::getenv("LFS_HOME"))
                old_home_ = home;
            setHome(temporary_.path.string());
        }
        void TearDown() override {
            setHome(old_home_.value_or(""));
            lfs::event::EventBridge::instance().clear_all();
        }
        static void setHome(const std::string& home) {
#ifdef _WIN32
            (void)_putenv_s("LFS_HOME", home.c_str());
#else
            if (home.empty())
                (void)unsetenv("LFS_HOME");
            else
                (void)setenv("LFS_HOME", home.c_str(), 1);
#endif
        }
        void initialize(VisualizerImpl& viewer) { viewer.initializeTools(); }
        static void clearDirty(RenderingManager& manager) { manager.dirty_mask_.store(0); }
        lfs::test::licht::TemporaryDirectory temporary_{"lfs-align-enabled-depth"};
        std::optional<std::string> old_home_;
    };

    TEST_F(AlignDepthIntegrationTest, EnabledToolPublishesPickingAndInvalidatesViewport) {
        ViewerOptions options;
        options.show_startup_overlay = false;
        options.project_lifecycle_settings_path = temporary_.path / "lifecycle.json";
        VisualizerImpl viewer(options);
        initialize(viewer);
        auto* rm = viewer.getRenderingManager();
        auto* sm = viewer.getSceneManager();
        ASSERT_TRUE(viewer.getAlignTool());
        gui::GizmoManager gizmos(&viewer);
        gui::UIContext ui{.viewer = &viewer};
        Viewport viewport(640, 480);
        const ViewportRegion region{0, 0, 640, 480};
        FrameContext frame{.viewport = viewport, .viewport_region = &region, .scene_manager = sm, .settings = {}, .render_size = {640, 480}};
        int wakes = 0;
        rm->setWakeCallback([&] { ++wakes; });
        auto check = [&](const bool enabled, const bool hidden = false) {
            SCOPED_TRACE(::testing::Message() << "tool=" << UnifiedToolRegistry::instance().getActiveTool()
                                              << " selected=" << sm->getSelectedNodeName() << " hidden=" << hidden);
            gizmos.updateToolState(ui, hidden);
            EXPECT_EQ(viewer.getAlignTool()->isEnabled(), enabled);
            rm->configureSurfacePicking(frame);
            EXPECT_EQ(frame.exact_surface_picking, enabled);
            EXPECT_EQ(buildViewportRenderRequest(frame, {640, 480}).require_exact_depth, enabled);
        };
        auto& registry = UnifiedToolRegistry::instance();
        registry.setActiveTool("builtin.align");
        check(false); // Active tool, but no selected node.
        const auto group = sm->addGroupNode("align-target");
        sm->selectNode(group);
        clearDirty(*rm);
        const int before_enable = wakes;
        check(true);
        EXPECT_GT(wakes, before_enable);
        EXPECT_NE(rm->pendingDirtyMask() & DirtyFlag::VIEWPORT, 0u);
        clearDirty(*rm);
        const int settled = wakes;
        check(true);
        EXPECT_EQ(wakes, settled); // An unchanged tool does not request work.
        EXPECT_EQ(rm->pendingDirtyMask(), 0u);
        frame.frame_dirty = DirtyFlag::CAMERA | DirtyFlag::OVERLAY;
        rm->configureSurfacePicking(frame);
        EXPECT_FALSE(buildViewportRenderRequest(frame, {640, 480}).require_exact_depth);
        frame.frame_dirty |= DirtyFlag::VIEWPORT; // Explicit point request during motion.
        rm->configureSurfacePicking(frame);
        EXPECT_TRUE(buildViewportRenderRequest(frame, {640, 480}).require_exact_depth);
        frame.frame_dirty = 0;
        rm->configureSurfacePicking(frame);
        EXPECT_TRUE(buildViewportRenderRequest(frame, {640, 480}).require_exact_depth);
        const auto crop = sm->getScene().addCropBox("crop", sm->getScene().getNode(group)->id);
        sm->selectNode(crop);
        registry.setActiveTool("builtin.align");
        check(false);
        sm->selectNode(group);
        registry.setActiveTool("builtin.align");
        check(true);
        check(false, true); // Hidden UI disables picking as well.
        check(true);
        registry.setActiveTool("builtin.select");
        check(false);
        registry.setActiveTool("builtin.align");
        check(true);
        sm->clearSelection();
        check(false);
        rm->setWakeCallback({});
    }
    TEST_F(AlignDepthIntegrationTest, ModelModalTicksWithoutGroundToolEnabled) {
        ViewerOptions options;
        options.show_startup_overlay = false;
        options.project_lifecycle_settings_path = temporary_.path / "lifecycle.json";
        VisualizerImpl viewer(options);
        initialize(viewer);
        auto* manager = viewer.getSceneManager();
        auto* renderer = viewer.getRenderingManager();
        auto& scene = manager->getScene();
        scene.addSplat("reference", lfs::test::licht::make_splat(1));
        scene.addSplat("moving", lfs::test::licht::make_splat(1));
        scene.setNodeLocked("reference", true); // Only the moving model must be editable.
        manager->selectNodes({"moving", "reference"});
        EXPECT_EQ(op::operators().invoke(op::BuiltinOp::ModelAlignment).status, op::OperatorResult::RUNNING_MODAL);
        ASSERT_TRUE(op::operators().isModal(op::BuiltinOp::ModelAlignment));
        ASSERT_NE(op::ModelAlignmentOperator::active(), nullptr);
        EXPECT_EQ(op::ModelAlignmentOperator::active()->session()->nodes[0], scene.getNode("reference")->uuid);
        manager->selectNodes({"reference"});
        EXPECT_TRUE(op::operators().poll(op::BuiltinOp::ModelAlignment));
        ToolContext context(renderer, manager, &viewer.getViewport(), nullptr);
        ToolBase* tool = viewer.getAlignTool();
        ASSERT_NE(tool, nullptr);
        ASSERT_FALSE(tool->isEnabled());
        scene.setNodeVisibility("moving", false);
        tool->update(context);
        EXPECT_FALSE(op::operators().isModal(op::BuiltinOp::ModelAlignment));
        EXPECT_EQ(op::ModelAlignmentOperator::active(), nullptr);
        EXPECT_FALSE(renderer->independentPanelOverrides());
    }

} // namespace lfs::vis

namespace {

    [[nodiscard]] glm::vec3 applyRot(const glm::mat4& m, const glm::vec3& v) {
        return glm::mat3(m) * v;
    }

    TEST(AlignDepth, RequestsExactDepthOutsideFrustumCoverage) {
        using namespace lfs::vis;
        struct EventGuard {
            EventGuard() { clear(); }
            ~EventGuard() { clear(); }
            static void clear() {
                lfs::event::EventBridge::instance().clear_all();
            }
        } event_guard;
        Viewport viewport(640, 480);
        const ViewportRegion region{0, 0, 640, 480};
        SceneManager manager;
        FrameContext ctx{.viewport = viewport,
                         .viewport_region = &region,
                         .scene_manager = &manager,
                         .settings = {},
                         .render_size = {640, 480}};
        auto& scene = manager.getScene();
        const auto volume = collision_test::box({}, glm::ivec3(1), .05f);
        scene.addCollision("First", volume);
        scene.addCollision("Second", volume);
        scene.addGroup("Other");
        const auto transform = glm::scale(glm::rotate(glm::translate(glm::mat4(1), {3, -2, 5}), .7f, {0, 1, 0}), {-2, 3, .5f});
        scene.setNodeTransform("First", transform);
        EXPECT_FLOAT_EQ(ctx.settings.collision_preview_opacity, .4f);
        for (const auto [name, visible, opacity] : {std::tuple{"First", true, .4f}, {"Other", true, .4f}, {"", true, .4f}, {"Other", false, 0.f}, {"Other", false, .25f}, {"Other", false, 1.f}, {"", true, 1.f}}) {
            manager.clearSelection();
            if (*name)
                manager.selectNode(name);
            scene.setNodeVisibility("First", visible);
            ctx.settings.collision_preview_opacity = opacity;
            ctx.scene_state = manager.buildRenderState({.metadata_only = true});
            const auto items = buildViewportMeshDrawItems(ctx.scene_state, ctx.settings, {0, 0, 1});
            ASSERT_EQ(items.size(), visible ? 2 : 1);
            if (visible)
                EXPECT_EQ(items[0].model, lfs::rendering::dataWorldTransformToVisualizerWorld(transform));
            for (const auto& item : items) {
                EXPECT_EQ(item.collision, volume);
                EXPECT_FLOAT_EQ(item.collision_opacity, opacity);
            }
        }
        EXPECT_FALSE(buildViewportRenderRequest(ctx, {640, 480}).require_exact_depth);
        ctx.settings.show_camera_frustums = true;
        auto request = buildViewportRenderRequest(ctx, {640, 480});
        EXPECT_TRUE(request.require_exact_depth);
        ASSERT_FALSE(request.exact_depth_sample_mask.empty());
        ctx.exact_surface_picking = true;
        request = buildViewportRenderRequest(ctx, {640, 480});
        EXPECT_TRUE(request.require_exact_depth);
        EXPECT_TRUE(request.exact_depth_sample_mask.empty());
        ctx.settings.show_camera_frustums = false;
        EXPECT_TRUE(buildViewportRenderRequest(ctx, {640, 480}).require_exact_depth);
        ctx.exact_surface_picking = false;
        EXPECT_FALSE(buildViewportRenderRequest(ctx, {640, 480}).require_exact_depth);
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

    TEST(AlignDepth, VulkanPointCloudDepthMatchesSurfaceWithoutGaussians) {
        if (!SDL_InitSubSystem(SDL_INIT_VIDEO))
            GTEST_SKIP() << "SDL video unavailable: " << SDL_GetError();
        struct VideoGuard {
            ~VideoGuard() { SDL_QuitSubSystem(SDL_INIT_VIDEO); }
        } video_guard;
        std::unique_ptr<SDL_Window, decltype(&SDL_DestroyWindow)> window(
            SDL_CreateWindow("Alignment depth test", 64, 64, SDL_WINDOW_VULKAN | SDL_WINDOW_HIDDEN),
            SDL_DestroyWindow);
        if (!window)
            GTEST_SKIP() << "Vulkan window unavailable: " << SDL_GetError();
        lfs::vis::VulkanContext context;
        if (!context.init(window.get(), 64, 64))
            GTEST_SKIP() << "Vulkan unavailable: " << context.lastError();
        lfs::vis::PointCloudVulkanRenderer renderer;
        auto positions = lfs::core::Tensor::from_vector(std::vector<float>{0, 0, -5},
                                                        {std::size_t{1}, std::size_t{3}}, lfs::core::Device::CPU);
        auto colors = lfs::core::Tensor::from_vector(std::vector<float>{1, 1, 1},
                                                     {std::size_t{1}, std::size_t{3}}, lfs::core::Device::CPU);
        const lfs::vis::PointCloudVulkanRenderer::OutputSlot target = lfs::vis::PointCloudVulkanRenderer::OutputSlot::Main;
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
            const auto depth = renderer.sampleDepthAtPixel(context, {.pixel = {32, 32}, .source_size = {64, 64}, .output_slot = target});
            ASSERT_TRUE(depth) << lfs::format_for_developer(depth.error());
            EXPECT_NEAR(*depth, 5.0f, 1e-3f);
            lfs::vis::PointCloudVulkanRenderer::DepthSampleRequest async_request{
                .pixel = {32, 32},
                .source_size = {64, 64},
                .output_slot = target,
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
            const auto background = renderer.sampleDepthAtPixel(context, {.pixel = {0, 0}, .source_size = {64, 64}, .output_slot = target});
            ASSERT_TRUE(background);
            EXPECT_LT(*background, 0.0f);
            EXPECT_FALSE(renderer.sampleDepthAtPixel(context, {.pixel = {64, 0}, .source_size = {64, 64}, .output_slot = target}));
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
            const auto temporary_target = lfs::vis::PointCloudVulkanRenderer::OutputSlot::SplitLeft;
            ASSERT_TRUE(renderer.render(context, request, temporary_target));
            async_request.output_slot = temporary_target;
            EXPECT_EQ(renderer.sampleDepthAtPixel(context, async_request).value(), lfs::vis::PointCloudVulkanRenderer::kDepthSamplePending);
            renderer.reset();
            EXPECT_FALSE(renderer.sampleDepthAtPixel(context, async_request));
            ASSERT_TRUE(context.deviceWaitIdle());
            (void)renderer.takeRefinementRequest();
        }
    }

    TEST(AlignPanelOrthoScale, PanelZoomOverridesGlobalScale) {
        lfs::vis::RenderSettings settings;
        settings.orthographic = true;
        settings.ortho_scale = 100.0f;
        Viewport panel(800, 600);
        EXPECT_FLOAT_EQ(lfs::vis::op::pointPickPanelOrthoScale(panel, settings), 100.0f);

        // A split-view panel zoomed on its own must draw and pick markers at its own scale.
        panel.ortho_scale_override = 400.0f;
        EXPECT_FLOAT_EQ(lfs::vis::op::pointPickPanelOrthoScale(panel, settings), 400.0f);
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

    TEST(PairAlignment, RecoversUniformScaleAndHalfTurnFromPlanarPoints) {
        const std::vector<glm::dvec3> source{{0, 0, 0}, {3, 0, 0}, {1, 2, 0}, {-1, 3, 0}};
        std::vector<glm::dvec3> reference;
        for (const auto p : source)
            reference.emplace_back(7.0 - 2.5 * p.x, -4.0 - 2.5 * p.y, 3.0);
        const auto fit = lfs::vis::op::computePairAlignment(source, reference, true);
        ASSERT_TRUE(fit);
        for (size_t i = 0; i < source.size(); ++i)
            EXPECT_LT(glm::distance(glm::dvec3(*fit * glm::dvec4(source[i], 1.0)), reference[i]), 1e-10);
        const auto rigid = lfs::vis::op::computePairAlignment(source, reference, false);
        ASSERT_TRUE(rigid);
        EXPECT_NEAR(glm::length(glm::dvec3((*rigid)[0])), 1.0, 1e-12);
    }

    TEST(PairAlignment, RejectsCollinearAndNonFiniteCorrespondences) {
        const std::vector<glm::dvec3> source{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
        std::vector<glm::dvec3> reference{{0, 0, 0}, {1, 0, 0}, {2, 0, 0}};
        EXPECT_FALSE(lfs::vis::op::computePairAlignment(source, reference, true));
        reference.back().y = 1e-5;
        EXPECT_FALSE(lfs::vis::op::computePairAlignment(source, reference, true));
        reference.back().y = std::numeric_limits<double>::quiet_NaN();
        EXPECT_FALSE(lfs::vis::op::computePairAlignment(source, reference, true));
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
        }

        OperatorResult action(PointPickState::UiAction action) {
            services().alignPointPick().requestUiAction(action);
            OperatorProperties properties;
            return operation_.modal(*context_, properties);
        }

        OperatorResult send(ModalEvent event) {
            context_->setModalEvent(event);
            OperatorProperties properties;
            return operation_.modal(*context_, properties);
        }

        OperatorResult mouse(const int action, const glm::dvec2 position, const bool over_gui = false) {
            return send({.type = ModalEvent::Type::MOUSE_BUTTON,
                         .over_gui = over_gui,
                         .data = MouseButtonEvent{.button = operation_.pick_button_, .action = action, .mods = 0, .position = position}});
        }

        OperatorResult key(const int key) {
            return send({.type = ModalEvent::Type::KEY, .data = KeyEvent{key, 0, input::ACTION_PRESS, 0}});
        }

        OperatorResult bound(const input::Action bound_action) {
            return send({.type = ModalEvent::Type::ACTION, .data = ActionEvent{.action = bound_action}});
        }

        void selectPoint(const int index) {
            operation_.selected_point_ = index;
            operation_.syncPickedPointsToServices();
        }

        glm::mat4 transform() const { return manager_->getScene().getNode("target")->transform(); }

        std::unique_ptr<SceneManager> manager_;
        std::unique_ptr<OperatorContext> context_;
        AlignPickPointOperator operation_;
    };

    TEST_F(AlignPreviewTest, MousePressPassesThroughAndClickReleaseIsConsumed) {
        EXPECT_EQ(mouse(input::ACTION_PRESS, {10, 20}), OperatorResult::PASS_THROUGH);
        EXPECT_EQ(mouse(input::ACTION_RELEASE, {10, 20}), OperatorResult::RUNNING_MODAL);
        EXPECT_EQ(services().alignPointPick().getPickedPoints().size(), 3u);
        EXPECT_EQ(mouse(input::ACTION_PRESS, {10, 20}), OperatorResult::PASS_THROUGH);
        EXPECT_EQ(mouse(input::ACTION_RELEASE, {30, 20}), OperatorResult::PASS_THROUGH);
    }

    TEST_F(AlignPreviewTest, PreviewAndGuiMousePressesPassThrough) {
        EXPECT_EQ(mouse(input::ACTION_PRESS, {10, 20}, true), OperatorResult::PASS_THROUGH);
        action(PointPickState::UiAction::TogglePreview);
        EXPECT_TRUE(services().getAlignPreviewEnabled());
        EXPECT_EQ(mouse(input::ACTION_PRESS, {10, 20}), OperatorResult::PASS_THROUGH);
        EXPECT_EQ(mouse(input::ACTION_RELEASE, {10, 20}), OperatorResult::PASS_THROUGH);
        EXPECT_EQ(services().alignPointPick().getPickedPoints().size(), 3u);
    }

    TEST_F(AlignPreviewTest, AlignmentChangesNotifyUiOncePerFrame) {
        auto& store = app_store();
        (void)store.store().drain_dirty_into_frame();
        int notifications = 0;
        auto token = store.align_state_generation.subscribe([&](const auto&) { ++notifications; });
        const auto generation = store.align_state_generation.get();

        // Repeated synchronization of the same triangle must not keep the UI awake.
        services().alignPointPick().setPickedPoints(services().alignPointPick().getPickedPoints());
        services().setAlignPreviewEnabled(services().getAlignPreviewEnabled());
        services().setAlignAxisSnapEnabled(services().getAlignAxisSnapEnabled());
        services().setAlignEdgeToAxisEnabled(services().getAlignEdgeToAxisEnabled());
        EXPECT_EQ(store.align_state_generation.get(), generation);

        services().alignPointPick().setPickedPoints({{0, 0, 0}, {1, 0, 0}, {0, 0, 1}});
        services().setAlignPreviewEnabled(true);
        services().setAlignAxisSnapEnabled(true);
        services().setAlignEdgeToAxisEnabled(true);
        EXPECT_EQ(notifications, 0);
        EXPECT_TRUE(store.store().drain_dirty_into_frame());
        EXPECT_EQ(notifications, 1);

        services().alignPointPick().clearPickedPoints();
        EXPECT_TRUE(store.store().drain_dirty_into_frame());
        EXPECT_EQ(notifications, 2);
        services().alignPointPick().clearPickedPoints();
        EXPECT_FALSE(store.store().drain_dirty_into_frame());
        EXPECT_EQ(notifications, 2);
    }

    TEST_F(AlignPreviewTest, ClearingStatusWithoutPickedPointsNotifiesUi) {
        services().alignPointPick().clearPickedPoints();
        services().alignPointPick().setStatusMessage("No surface at the selected point", 5.0);
        auto& store = app_store();
        (void)store.store().drain_dirty_into_frame();
        const auto generation = store.align_state_generation.get();
        ASSERT_NE(services().alignPointPick().getStatusMessage(), nullptr);

        services().alignPointPick().clearPickedPoints();
        EXPECT_EQ(services().alignPointPick().getStatusMessage(), nullptr);
        EXPECT_EQ(store.align_state_generation.get(), generation + 1);
        EXPECT_TRUE(store.store().drain_dirty_into_frame());

        services().alignPointPick().clearPickedPoints();
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
        EXPECT_EQ(action(PointPickState::UiAction::Apply), OperatorResult::FINISHED);
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
        action(PointPickState::UiAction::TogglePreview);
        ASSERT_NE(transform(), original);
        manager_->getScene().setNodeLocked("target", true);
        operation_.cancel(*context_);
        EXPECT_EQ(transform(), original);
        EXPECT_TRUE(static_cast<bool>(manager_->getScene().getNode("target")->locked));
        EXPECT_FALSE(services().getAlignPreviewEnabled());
    }

    TEST_F(AlignPreviewTest, OptionTogglesUseOriginalTransformWithoutDrift) {
        const auto original = transform();
        action(PointPickState::UiAction::TogglePreview);
        const auto without_axis = transform();
        services().setAlignEdgeToAxisEnabled(true);
        action(PointPickState::UiAction::RefreshPreview);
        EXPECT_NE(transform(), without_axis);
        services().setAlignEdgeToAxisEnabled(false);
        action(PointPickState::UiAction::RefreshPreview);
        EXPECT_EQ(transform(), without_axis);
        action(PointPickState::UiAction::TogglePreview);
        EXPECT_EQ(transform(), original);
        EXPECT_EQ(undoHistory().undoCount(), 0u);
    }

    TEST_F(AlignPreviewTest, ApplyWithoutPreviewMatchesPreviewAndCreatesOneUndoStep) {
        const auto original = transform();
        action(PointPickState::UiAction::TogglePreview);
        const auto preview = transform();
        action(PointPickState::UiAction::TogglePreview);
        ASSERT_EQ(transform(), original);
        EXPECT_EQ(action(PointPickState::UiAction::Apply), OperatorResult::FINISHED);
        EXPECT_EQ(transform(), preview);
        EXPECT_FALSE(services().getAlignPreviewEnabled());
        EXPECT_EQ(undoHistory().undoCount(), 1u);
        EXPECT_TRUE(undoHistory().undo().success);
        EXPECT_EQ(transform(), original);
    }

    TEST_F(AlignPreviewTest, PointControlsFollowTheKeyMap) {
        const auto original = transform();
        for (const int fixed_key : {input::KEY_BACKSPACE, input::KEY_DELETE, input::KEY_ENTER,
                                    input::KEY_KP_ENTER, input::KEY_ESCAPE}) {
            EXPECT_EQ(key(fixed_key), OperatorResult::PASS_THROUGH) << fixed_key;
        }
        EXPECT_EQ(services().alignPointPick().getPickedPoints().size(), 3u);
        EXPECT_EQ(transform(), original);

        selectPoint(0);
        EXPECT_EQ(bound(input::Action::DELETE_SELECTED), OperatorResult::RUNNING_MODAL);
        EXPECT_EQ(services().alignPointPick().getPickedPoints().size(), 2u);
        EXPECT_EQ(bound(input::Action::REMOVE_LAST_POINT), OperatorResult::RUNNING_MODAL);
        EXPECT_EQ(services().alignPointPick().getPickedPoints().size(), 1u);
        EXPECT_EQ(bound(input::Action::APPLY_ALIGNMENT), OperatorResult::RUNNING_MODAL);
        EXPECT_EQ(bound(input::Action::UNDO), OperatorResult::PASS_THROUGH);
        EXPECT_EQ(bound(input::Action::CANCEL_POLYGON), OperatorResult::CANCELLED);
        EXPECT_EQ(transform(), original);
    }

    TEST_F(AlignPreviewTest, BoundApplyCommitsOneUndoStep) {
        const auto original = transform();
        EXPECT_EQ(bound(input::Action::APPLY_ALIGNMENT), OperatorResult::FINISHED);
        EXPECT_NE(transform(), original);
        EXPECT_EQ(undoHistory().undoCount(), 1u);
    }

    TEST_F(AlignPreviewTest, ApplyCommitsOneUndoStepMatchingThePreview) {
        const auto original = transform();
        action(PointPickState::UiAction::TogglePreview);
        const auto preview = transform();
        EXPECT_EQ(action(PointPickState::UiAction::Apply), OperatorResult::FINISHED);
        EXPECT_EQ(transform(), preview);
        EXPECT_EQ(undoHistory().undoCount(), 1u);
        EXPECT_TRUE(undoHistory().undo().success);
        EXPECT_EQ(transform(), original);
        EXPECT_TRUE(undoHistory().redo().success);
        EXPECT_EQ(transform(), preview);
    }

    class ModelAlignmentTest : public ::testing::Test {
    protected:
        void SetUp() override {
            services().clear();
            undoHistory().clear();
            manager_ = std::make_unique<SceneManager>();
            services().set(manager_.get());
            operators().setSceneManager(manager_.get());
            registerEditOperators();
            operators().registerOperator(BuiltinOp::ModelAlignment,
                                         OperatorDescriptor{.builtin_id = BuiltinOp::ModelAlignment,
                                                            .label = "Test alignment",
                                                            .flags = OperatorFlags::REGISTER},
                                         [this] {
                                             auto op = std::make_unique<ModelAlignmentOperator>();
                                             if (pair_rendering_) {
                                                 auto& scene = manager_->getScene();
                                                 (void)op->begin(pair_primary_, scene, *scene.getNode("reference"), *scene.getNode("mobile"));
                                             }
                                             return op;
                                         });
            manager_->getScene().addGroup("target");
            manager_->selectNode("target");
            context_ = std::make_unique<OperatorContext>(*manager_);
            services().setAlignCameraPosition({0, 5, 5});
            services().setAlignAxisSnapEnabled(false);
            services().setAlignEdgeToAxisEnabled(false);
        }

        void TearDown() override {
            operators().cancelModalOperator();
            operation_.cancel(*context_);
            operators().unregisterOperator(BuiltinOp::ModelAlignment);
            unregisterEditOperators();
            operators().setSceneManager(nullptr);
            undoHistory().clear();
            services().clear();
            context_.reset();
            manager_.reset();
            lfs::event::EventBridge::instance().clear_all();
        }

        OperatorResult action(PointPickState::UiAction action) {
            operation_.pointState().requestUiAction(action);
            return send({});
        }

        bool previewEnabled() const {
            const auto* view = operation_.session();
            return view ? view->preview.has_value() : services().getAlignPreviewEnabled();
        }

        OperatorResult send(ModalEvent event) {
            context_->setModalEvent(event);
            OperatorProperties properties;
            return operation_.modal(*context_, properties);
        }

        OperatorResult mouse(const int action, const glm::dvec2 position, const bool over_gui = false) {
            return send({.type = ModalEvent::Type::MOUSE_BUTTON,
                         .over_gui = over_gui,
                         .data = MouseButtonEvent{.button = operation_.pick_button_, .action = action, .mods = 0, .position = position}});
        }

        OperatorResult key(const int key) {
            return send({.type = ModalEvent::Type::KEY, .data = KeyEvent{key, 0, input::ACTION_PRESS, 0}});
        }

        OperatorResult bound(const input::Action bound_action) {
            return send({.type = ModalEvent::Type::ACTION, .data = ActionEvent{.action = bound_action}});
        }

        void selectPoint(const int index) {
            operation_.selected_point_ = index;
            operation_.syncPickedPointsToServices();
        }

        void addPairPoint(const glm::vec3 point) {
            operation_.picked_points_.push_back(point);
            operation_.syncPickedPointsToServices();
        }

        void startPairAlignment(RenderingManager& rendering, const bool with_points = true) {
            using core::Device;
            using core::Tensor;
            const auto make_splat = [] {
                return std::make_unique<core::SplatData>(
                    0, Tensor::from_vector({0.0f, 0.0f, 2.0f}, {1, 3}, Device::CPU),
                    Tensor::zeros({1, 1, 3}, Device::CPU),
                    Tensor::zeros({1, 0, 3}, Device::CPU),
                    Tensor::zeros({1, 3}, Device::CPU),
                    Tensor::from_vector({1.0f, 0.0f, 0.0f, 0.0f}, {1, 4}, Device::CPU),
                    Tensor::zeros({1, 1}, Device::CPU), 1.0f);
            };
            auto& scene = manager_->getScene();
            scene.addSplat("reference", make_splat());
            scene.addSplat("mobile", make_splat());
            manager_->selectNodes({"reference", "mobile"});
            services().set(&rendering);
            ASSERT_TRUE(operation_.begin(pair_primary_, scene, *scene.getNode("reference"), *scene.getNode("mobile")));
            OperatorProperties properties;
            ASSERT_EQ(operation_.invoke(*context_, properties), OperatorResult::RUNNING_MODAL);
            ASSERT_TRUE(operation_.session());
            if (with_points) {
                operation_.picked_points_ = {{0, 0, 0}, {0, 0, 0}, {2, 0, 0}, {1, 0, 0}, {0, 2, 0}, {0, 1, 0}};
                operation_.syncPickedPointsToServices();
            }
        }

        bool sample(const glm::vec3 point, SplitViewPanelId panel, std::optional<int> index = {}) {
            return operation_.applyPointSample({.world = point, .panel = panel}, index);
        }

        bool sampleFor(ModelAlignmentOperator& operation, glm::vec3 point, SplitViewPanelId panel, std::optional<int> index = {}) {
            return operation.applyPointSample({.world = point, .panel = panel}, index);
        }

        void selectFor(ModelAlignmentOperator& operation, int index) {
            operation.selected_point_ = index;
            operation.syncPickedPointsToServices();
        }

        std::array<core::Uuid, 2> pairTargets() const { return operation_.session()->nodes; }

        glm::mat4 transform() const { return manager_->getScene().getNode("target")->transform(); }

        std::unique_ptr<SceneManager> manager_;
        std::unique_ptr<OperatorContext> context_;
        Viewport pair_primary_;
        ModelAlignmentOperator operation_;
        std::unique_ptr<RenderingManager> pair_rendering_;
    };

    TEST_F(ModelAlignmentTest, PollUsesRunningBuiltinInstanceUntilCancellation) {
        constexpr auto id = BuiltinOp::ScaleDistancePickPoint;
        class TestModal final : public Operator {
        public:
            const OperatorDescriptor& descriptor() const override {
                static const OperatorDescriptor value{.builtin_id = BuiltinOp::ScaleDistancePickPoint};
                return value;
            }

            bool poll(const OperatorContext&, const OperatorProperties*) const override { return !running_; }

            OperatorResult invoke(OperatorContext&, OperatorProperties&) override {
                running_ = true;
                return OperatorResult::RUNNING_MODAL;
            }

        private:
            bool running_ = false;
        };

        auto& registry = operators();
        registry.registerOperator(id, TestModal{}.descriptor(), [] { return std::make_unique<TestModal>(); });
        EXPECT_TRUE(registry.poll(id)); // Warm the idle poll cache.
        EXPECT_EQ(registry.invoke(id).status, OperatorResult::RUNNING_MODAL);
        EXPECT_FALSE(registry.poll(id));
        EXPECT_FALSE(registry.poll(to_string(id)));

        const std::string callback_id = "test.modal_poll.precedence";
        registry.registerCallbackOperator(
            OperatorDescriptor{.python_class_id = callback_id},
            CallbackOperator{.invoke = [](OperatorProperties&) { return OperatorResult::RUNNING_MODAL; },
                             .modal = [](const ModalEvent&, OperatorProperties&) { return OperatorResult::FINISHED; }});
        EXPECT_EQ(registry.invoke(callback_id).status, OperatorResult::RUNNING_MODAL);
        EXPECT_EQ(registry.activeModalId(), callback_id);
        EXPECT_TRUE(registry.poll(id));
        registry.cancelModalOperator();
        registry.unregisterOperator(callback_id);
        EXPECT_FALSE(registry.poll(id));
        registry.cancelModalOperator();
        EXPECT_TRUE(registry.poll(id));
        registry.unregisterOperator(id);
    }

    TEST_F(ModelAlignmentTest, PairedPreviewRestoresAndApplyOnlyMovesTheMobileSplat) {
        pair_rendering_ = std::make_unique<RenderingManager>();
        startPairAlignment(*pair_rendering_);
        ASSERT_FALSE(HasFatalFailure());
        const auto original = manager_->getNodeTransform("mobile");
        const auto reference = manager_->getNodeTransform("reference");
        action(PointPickState::UiAction::TogglePreview);
        EXPECT_TRUE(previewEnabled());
        EXPECT_EQ(manager_->getNodeTransform("mobile"), original);
        EXPECT_TRUE(operation_.session()->preview);
        EXPECT_EQ(manager_->getNodeTransform("reference"), reference);
        action(PointPickState::UiAction::TogglePreview);
        EXPECT_FALSE(previewEnabled());
        EXPECT_EQ(manager_->getNodeTransform("mobile"), original);
        EXPECT_EQ(action(PointPickState::UiAction::Apply), OperatorResult::FINISHED);
        EXPECT_NEAR(glm::length(glm::vec3(manager_->getNodeTransform("mobile")[0])), 2.0f, 1e-5f);
        EXPECT_EQ(manager_->getNodeTransform("reference"), reference);
        EXPECT_EQ(undoHistory().undoCount(), 1u);
        ASSERT_TRUE(undoHistory().undo().success);
        EXPECT_EQ(manager_->getNodeTransform("mobile"), original);
        EXPECT_EQ(manager_->getNodeTransform("reference"), reference);
    }

    TEST_F(ModelAlignmentTest, DedicatedViewRestoresCamerasAndKeepsPreviewOutOfScene) {
        pair_rendering_ = std::make_unique<RenderingManager>();
        startPairAlignment(*pair_rendering_);
        ASSERT_FALSE(HasFatalFailure());
        operation_.cancel(*context_);

        auto& scene = manager_->getScene();
        const auto& reference = *scene.getNode("reference");
        const auto& mobile = *scene.getNode("mobile");
        const auto original = mobile.transform();
        Viewport primary;
        auto& renderer = *pair_rendering_;
        renderer.projectSecondaryViewport().camera.t = {17, 18, 19};
        const auto saved_left = primary.camera.t;
        const auto saved_right = renderer.projectSecondaryViewport().camera.t;
        const auto saved_mode = renderer.getSplitViewMode();

        ASSERT_TRUE(operation_.begin(primary, scene, reference, mobile));
        ASSERT_TRUE(renderer.isIndependentSplitViewActive());
        ASSERT_NE(operation_.session(), nullptr);
        EXPECT_EQ(primary.camera.t, saved_left);
        EXPECT_EQ(renderer.projectSecondaryViewport().camera.t, saved_left);

        const auto similarity = glm::scale(glm::mat4(1), glm::vec3(2));
        operation_.setPreview(similarity);
        EXPECT_EQ(renderer.projectSecondaryViewport().camera.t, saved_left * 2.0f);
        EXPECT_EQ(mobile.transform(), original);
        renderer.setFocusedSplitPanel(SplitViewPanelId::Right);
        operation_.setComposite(SplitViewPanelId::Left);
        EXPECT_EQ(renderer.getFocusedSplitPanel(), SplitViewPanelId::Right);
        EXPECT_EQ(primary.camera.t, saved_left);
        EXPECT_EQ(renderer.projectSecondaryViewport().camera.t, saved_left * 2.0f);
        operation_.setComposite(std::nullopt);
        EXPECT_FALSE(operation_.session()->composite);
        operation_.setComposite(SplitViewPanelId::Right);
        EXPECT_EQ(operation_.session()->composite, SplitViewPanelId::Right);
        renderer.projectSecondaryViewport().camera.t += glm::vec3(5, 6, 7);
        const auto focused_position = renderer.projectSecondaryViewport().camera.t;
        operation_.toggleCameraLink();
        EXPECT_EQ(primary.camera.t, focused_position);
        primary.camera.t += glm::vec3(1, 2, 3);
        operation_.syncCamera(primary);
        const auto linked_position = primary.camera.t;
        operation_.toggleCameraLink();
        EXPECT_EQ(primary.camera.t, linked_position);
        EXPECT_EQ(renderer.projectSecondaryViewport().camera.t, linked_position);
        primary.camera.t += glm::vec3(2, 3, 4);
        operation_.syncCamera(primary);
        EXPECT_EQ(renderer.projectSecondaryViewport().camera.t, linked_position);
        operation_.setPreview(std::nullopt);
        EXPECT_FALSE(operation_.session()->preview);
        EXPECT_EQ(primary.camera.t, saved_left);
        EXPECT_EQ(renderer.projectSecondaryViewport().camera.t, saved_left);

        const glm::ivec2 new_size{512, 384};
        primary.windowSize = new_size;
        operation_.end();
        EXPECT_EQ(primary.windowSize, new_size);
        EXPECT_EQ(operation_.session(), nullptr);
        EXPECT_EQ(renderer.getSplitViewMode(), saved_mode);
        EXPECT_EQ(primary.camera.t, saved_left);
        EXPECT_EQ(renderer.projectSecondaryViewport().camera.t, saved_right);
        EXPECT_EQ(mobile.transform(), original);
        EXPECT_EQ(undoHistory().undoCount(), 0u);

        renderer.restoreSplitViewMode(SplitViewMode::IndependentDual, primary);
        renderer.projectSecondaryViewport().camera.t = saved_right;
        ASSERT_TRUE(operation_.begin(primary, scene, reference, mobile));
        EXPECT_EQ(primary.camera.t, saved_left);
        EXPECT_EQ(renderer.projectSecondaryViewport().camera.t, saved_right);
        operation_.end();
        EXPECT_EQ(renderer.getSplitViewMode(), SplitViewMode::IndependentDual);
    }

    TEST_F(ModelAlignmentTest, DedicatedApplyResolvesParentTransformAndRestoresBothViews) {
        pair_rendering_ = std::make_unique<RenderingManager>();
        startPairAlignment(*pair_rendering_);
        ASSERT_FALSE(HasFatalFailure());
        operation_.cancel(*context_);
        auto& scene = manager_->getScene();
        const auto parent = scene.addGroup("parent");
        ASSERT_TRUE(scene.reparent(scene.getNode("mobile")->id, parent));
        const auto parent_world = glm::translate(glm::mat4(1), glm::vec3(7, -2, 3)) *
                                  glm::rotate(glm::mat4(1), 0.7f, glm::vec3(0, 1, 0)) *
                                  glm::scale(glm::mat4(1), glm::vec3(1.5f));
        ASSERT_TRUE(manager_->setNodeTransform("parent", parent_world));
        const auto mobile_local = manager_->getNodeTransform("mobile");
        const auto mobile_world = scene_coords::nodeVisualizerWorldTransform(scene, scene.getNode("mobile")->id);
        const auto reference_local = manager_->getNodeTransform("reference");
        const auto fit = glm::translate(glm::mat4(1), glm::vec3(-3, 4, 1)) *
                         glm::rotate(glm::mat4(1), -0.4f, glm::vec3(1, 0, 0)) *
                         glm::scale(glm::mat4(1), glm::vec3(2.0f));
        const auto mode = pair_rendering_->getSplitViewMode();
        pair_primary_.camera.t = {3, 4, 5};
        pair_rendering_->projectSecondaryViewport().camera.t = {8, 9, 10};
        ASSERT_TRUE(operation_.begin(pair_primary_, scene, *scene.getNode("reference"), *scene.getNode("mobile")));
        OperatorProperties props;
        ASSERT_EQ(operation_.invoke(*context_, props), OperatorResult::RUNNING_MODAL);
        for (const auto local : {glm::vec3(0), glm::vec3(1, 0, 0), glm::vec3(0, 1, 1), glm::vec3(0, 0, 2)}) {
            const auto source = glm::vec3(mobile_world * glm::vec4(local, 1));
            ASSERT_TRUE(sample(source, SplitViewPanelId::Right));
            ASSERT_TRUE(sample(glm::vec3(fit * glm::vec4(source, 1)), SplitViewPanelId::Left));
        }
        ASSERT_EQ(action(PointPickState::UiAction::TogglePreview), OperatorResult::RUNNING_MODAL);
        ASSERT_TRUE(previewEnabled());
        EXPECT_EQ(manager_->getNodeTransform("mobile"), mobile_local);
        EXPECT_EQ(manager_->getNodeTransform("reference"), reference_local);
        EXPECT_EQ(action(PointPickState::UiAction::Apply), OperatorResult::FINISHED);
        EXPECT_EQ(operation_.session(), nullptr);
        EXPECT_EQ(pair_rendering_->getSplitViewMode(), mode);
        EXPECT_EQ(pair_primary_.camera.t, glm::vec3(3, 4, 5));
        EXPECT_EQ(pair_rendering_->projectSecondaryViewport().camera.t, glm::vec3(8, 9, 10));
        const auto expected = fit * mobile_world;
        const auto actual = scene_coords::nodeVisualizerWorldTransform(scene, scene.getNode("mobile")->id);
        for (int c = 0; c < 4; ++c)
            for (int r = 0; r < 4; ++r)
                EXPECT_NEAR(actual[c][r], expected[c][r], 2e-4f);
        EXPECT_EQ(manager_->getNodeTransform("reference"), reference_local);
        EXPECT_EQ(undoHistory().undoCount(), 1u);
        ASSERT_TRUE(undoHistory().undo().success);
        EXPECT_EQ(manager_->getNodeTransform("mobile"), mobile_local);
        EXPECT_EQ(manager_->getNodeTransform("reference"), reference_local);
    }

    TEST_F(ModelAlignmentTest, DedicatedCancelAndOperatorDestructionRestoreTheSession) {
        pair_rendering_ = std::make_unique<RenderingManager>();
        startPairAlignment(*pair_rendering_);
        ASSERT_FALSE(HasFatalFailure());
        const auto original = manager_->getNodeTransform("mobile");
        ASSERT_EQ(action(PointPickState::UiAction::TogglePreview), OperatorResult::RUNNING_MODAL);
        operation_.cancel(*context_);
        EXPECT_EQ(operation_.session(), nullptr);
        EXPECT_EQ(manager_->getNodeTransform("mobile"), original);
        auto& scene = manager_->getScene();
        std::shared_ptr<const ModelAlignmentOperator::Session> saved_view;
        {
            ModelAlignmentOperator replacement;
            ASSERT_TRUE(replacement.begin(pair_primary_, scene, *scene.getNode("reference"), *scene.getNode("mobile")));
            saved_view = ModelAlignmentOperator::savedViewSession();
            ASSERT_TRUE(saved_view);
            OperatorProperties props;
            ASSERT_EQ(replacement.invoke(*context_, props), OperatorResult::RUNNING_MODAL);
        }
        EXPECT_EQ(operation_.session(), nullptr);
        EXPECT_FALSE(ModelAlignmentOperator::savedViewSession());
        ASSERT_TRUE(saved_view);
        EXPECT_EQ(saved_view->previous_mode, SplitViewMode::Disabled);
        EXPECT_EQ(manager_->getNodeTransform("mobile"), original);
        EXPECT_EQ(undoHistory().undoCount(), 0u);
    }

    TEST_F(ModelAlignmentTest, DedicatedSwapRefreshesTargetsAndFitFailuresHaveFeedback) {
        pair_rendering_ = std::make_unique<RenderingManager>();
        startPairAlignment(*pair_rendering_, false);
        ASSERT_FALSE(HasFatalFailure());
        auto& scene = manager_->getScene();
        operation_.swapRoles(scene);
        ASSERT_EQ(send({}), OperatorResult::RUNNING_MODAL);
        EXPECT_EQ(pairTargets()[0], scene.getNode("mobile")->uuid);
        EXPECT_EQ(pairTargets()[1], scene.getNode("reference")->uuid);
        EXPECT_EQ(bound(input::Action::APPLY_ALIGNMENT), OperatorResult::RUNNING_MODAL);
        ASSERT_NE(operation_.pointState().getStatusMessage(), nullptr);
        EXPECT_EQ(*operation_.pointState().getStatusMessage(), LOC("align.pair_incomplete"));
        operation_.pointState().clearStatusMessage();
        for (int i = 0; i < 3; ++i) {
            ASSERT_TRUE(sample({i, 0, 0}, SplitViewPanelId::Left));
            ASSERT_TRUE(sample({i, 0, 0}, SplitViewPanelId::Right));
        }
        EXPECT_EQ(bound(input::Action::APPLY_ALIGNMENT), OperatorResult::RUNNING_MODAL);
        ASSERT_NE(operation_.pointState().getStatusMessage(), nullptr);
        EXPECT_EQ(*operation_.pointState().getStatusMessage(), LOC("align.pair_degenerate"));
        EXPECT_FALSE(previewEnabled());
        EXPECT_EQ(undoHistory().undoCount(), 0u);
    }

    TEST_F(ModelAlignmentTest, MovedReferenceInvalidatesPairsBeforeApply) {
        pair_rendering_ = std::make_unique<RenderingManager>();
        startPairAlignment(*pair_rendering_);
        ASSERT_FALSE(HasFatalFailure());
        const auto mobile = manager_->getNodeTransform("mobile");
        const auto moved = glm::translate(glm::mat4(1.0f), glm::vec3(2.0f, 0.0f, 0.0f));
        ASSERT_TRUE(manager_->setNodeTransform("reference", moved));

        EXPECT_EQ(action(PointPickState::UiAction::Apply), OperatorResult::RUNNING_MODAL);
        EXPECT_TRUE(operation_.pointState().getPickedPoints().empty());
        EXPECT_FALSE(previewEnabled());
        EXPECT_EQ(manager_->getNodeTransform("reference"), moved);
        EXPECT_EQ(manager_->getNodeTransform("mobile"), mobile);
        EXPECT_EQ(undoHistory().undoCount(), 0u);
    }

    TEST_F(ModelAlignmentTest, PreviewFocusDoesNotChangeSelectionOrAllowRemoval) {
        pair_rendering_ = std::make_unique<RenderingManager>();
        startPairAlignment(*pair_rendering_);
        ASSERT_FALSE(HasFatalFailure());
        selectPoint(3);
        action(PointPickState::UiAction::TogglePreview);
        ASSERT_TRUE(previewEnabled());
        EXPECT_FALSE(operation_.feedback().focus);
        action(PointPickState::UiAction::TogglePreview);

        addPairPoint({2, 2, 0});
        addPairPoint({1, 1, 0});
        action(PointPickState::UiAction::TogglePreview);
        ASSERT_TRUE(previewEnabled());
        EXPECT_FALSE(operation_.feedback().focus);
        action(PointPickState::UiAction::TogglePreview);

        addPairPoint({1, 1, 1});
        addPairPoint({0.5f, 0.5f, 0});
        const auto original_points = operation_.pointState().getPickedPoints();
        const auto original_mobile = manager_->getNodeTransform("mobile");
        action(PointPickState::UiAction::TogglePreview);
        ASSERT_TRUE(previewEnabled());
        EXPECT_EQ(operation_.feedback().focus, 8);
        EXPECT_EQ(operation_.pointState().getSelectedPoint(), 3);
        const auto preview_points = operation_.pointState().getPickedPoints();
        const auto preview_mobile = manager_->getNodeTransform("mobile");
        EXPECT_EQ(bound(input::Action::DELETE_SELECTED), OperatorResult::RUNNING_MODAL);
        EXPECT_EQ(bound(input::Action::REMOVE_LAST_POINT), OperatorResult::RUNNING_MODAL);
        EXPECT_TRUE(previewEnabled());
        EXPECT_EQ(operation_.pointState().getPickedPoints(), preview_points);
        EXPECT_EQ(manager_->getNodeTransform("mobile"), preview_mobile);

        send({});
        action(PointPickState::UiAction::TogglePreview);
        EXPECT_FALSE(operation_.feedback().focus);
        EXPECT_TRUE(operation_.feedback().summary.empty());
        EXPECT_EQ(operation_.pointState().getSelectedPoint(), 3);
        EXPECT_EQ(operation_.pointState().getPickedPoints(), original_points);
        EXPECT_EQ(manager_->getNodeTransform("mobile"), original_mobile);
    }

    TEST_F(ModelAlignmentTest, ExtraPairsParticipateInFitAndCanBeRemovedTogether) {
        pair_rendering_ = std::make_unique<RenderingManager>();
        startPairAlignment(*pair_rendering_);
        ASSERT_FALSE(HasFatalFailure());
        const auto original = manager_->getNodeTransform("mobile");
        addPairPoint({0, 0, 3});
        EXPECT_EQ(action(PointPickState::UiAction::Apply), OperatorResult::RUNNING_MODAL);
        EXPECT_EQ(manager_->getNodeTransform("mobile"), original);

        addPairPoint({0, 0, 1});
        action(PointPickState::UiAction::TogglePreview);
        ASSERT_TRUE(previewEnabled());
        EXPECT_GT(glm::length(glm::vec3((*operation_.session()->preview)[0])), 2.1f);
        action(PointPickState::UiAction::TogglePreview);
        addPairPoint({2, 2, 0});
        addPairPoint({1, 1, 0});
        selectPoint(7);
        EXPECT_EQ(bound(input::Action::DELETE_SELECTED), OperatorResult::RUNNING_MODAL);
        EXPECT_EQ(operation_.pointState().getPickedPoints().size(), 8u);
        EXPECT_EQ(action(PointPickState::UiAction::Apply), OperatorResult::FINISHED);
        EXPECT_NEAR(glm::length(glm::vec3(manager_->getNodeTransform("mobile")[0])), 2.0f, 1e-5f);
        EXPECT_EQ(undoHistory().undoCount(), 1u);
    }

    TEST_F(ModelAlignmentTest, ExternalMobileEditSupersedesThePreview) {
        pair_rendering_ = std::make_unique<RenderingManager>();
        startPairAlignment(*pair_rendering_);
        ASSERT_FALSE(HasFatalFailure());
        action(PointPickState::UiAction::TogglePreview);
        ASSERT_TRUE(previewEnabled());
        const auto moved = glm::translate(glm::mat4(1.0f), glm::vec3(2.0f, 0.0f, 0.0f));
        ASSERT_TRUE(manager_->setNodeTransform("mobile", moved));

        EXPECT_EQ(action(PointPickState::UiAction::Apply), OperatorResult::RUNNING_MODAL);
        EXPECT_TRUE(operation_.pointState().getPickedPoints().empty());
        EXPECT_FALSE(previewEnabled());
        EXPECT_EQ(manager_->getNodeTransform("mobile"), moved);
        EXPECT_EQ(undoHistory().undoCount(), 0u);
    }

    TEST_F(ModelAlignmentTest, ComparisonRequiresAnEditableMobileSplat) {
        pair_rendering_ = std::make_unique<RenderingManager>();
        startPairAlignment(*pair_rendering_);
        ASSERT_FALSE(HasFatalFailure());
        EXPECT_TRUE(operation_.poll(*context_));
        manager_->getScene().setNodeLocked("reference", true);
        EXPECT_TRUE(operation_.poll(*context_));
        ModelAlignmentOperator idle;
        EXPECT_TRUE(idle.poll(*context_));
        OperatorProperties reversed;
        reversed.set("reference", std::string("mobile"));
        reversed.set("moving", std::string("reference"));
        EXPECT_FALSE(idle.poll(*context_, &reversed));
        manager_->selectNodes({"reference"});
        EXPECT_FALSE(idle.poll(*context_));
        EXPECT_TRUE(operation_.poll(*context_));
        manager_->getScene().setNodeLocked("mobile", true);
        EXPECT_FALSE(operation_.poll(*context_));
        EXPECT_EQ(action(PointPickState::UiAction::Apply), OperatorResult::CANCELLED);
        EXPECT_EQ(undoHistory().undoCount(), 0u);
    }

    TEST_F(ModelAlignmentTest, PointsPreviewAndSceneHistoryHaveIndependentUndo) {
        pair_rendering_ = std::make_unique<RenderingManager>();
        startPairAlignment(*pair_rendering_, false);
        ASSERT_FALSE(HasFatalFailure());
        operation_.cancel(*context_);
        const auto original = manager_->getNodeTransform("target");
        auto entry = std::make_unique<SceneSnapshot>(*manager_, "transform.translate");
        ASSERT_TRUE(entry->captureTransformsBefore({"target"}, {original}));
        ASSERT_TRUE(manager_->setNodeTransform("target", glm::translate(glm::mat4(1), glm::vec3(2, 0, 0))));
        entry->captureAfter();
        undoHistory().push(std::move(entry));
        ASSERT_TRUE(operators().poll(BuiltinOp::Undo));
        const auto generation = undoHistory().generation();
        (void)operators().invoke(BuiltinOp::ModelAlignment);
        ASSERT_TRUE(operators().isModal(BuiltinOp::ModelAlignment));
        auto* active = ModelAlignmentOperator::active();
        ASSERT_NE(active, nullptr);
        const auto sample = [&](glm::vec3 point, SplitViewPanelId panel) {
            return sampleFor(*active, point, panel);
        };
        const auto undo = [&] {
            return operators().dispatchModalEvent({.type = ModalEvent::Type::ACTION,
                                                   .data = ActionEvent{.action = input::Action::UNDO}});
        };
        ASSERT_TRUE(sample({0, 0, 0}, SplitViewPanelId::Right));
        EXPECT_TRUE(active->pendingPairOnRight());
        ASSERT_TRUE(sample({1, 0, 0}, SplitViewPanelId::Left));
        EXPECT_EQ(active->pointState().getPickedPoints().front(), glm::vec3(1, 0, 0));
        EXPECT_FALSE(undoHistory().canUndo());
        EXPECT_FALSE(operators().poll(BuiltinOp::Undo));
        EXPECT_FALSE(operators().poll(BuiltinOp::Redo));
        EXPECT_FALSE(undoHistory().undo().success);
        EXPECT_FALSE(undoHistory().redo().success);
        EXPECT_EQ(undo(), OperatorResult::RUNNING_MODAL);
        EXPECT_EQ(active->pointState().getPickedPoints().size(), 1u);
        EXPECT_TRUE(active->pendingPairOnRight());
        EXPECT_EQ(undo(), OperatorResult::RUNNING_MODAL);
        EXPECT_TRUE(active->pointState().getPickedPoints().empty());
        for (auto point : {glm::vec3(0), glm::vec3(1, 0, 0), glm::vec3(0, 1, 0)}) {
            ASSERT_TRUE(sample(point * 2.0f, SplitViewPanelId::Left));
            ASSERT_TRUE(sample(point, SplitViewPanelId::Right));
        }
        const auto points = active->pointState().getPickedPoints();
        const auto colors = active->pairColorSlots();
        ASSERT_TRUE(sampleFor(*active, {3, 0, 0}, SplitViewPanelId::Left, 2));
        ASSERT_TRUE(sampleFor(*active, {4, 0, 0}, SplitViewPanelId::Left, 2));
        EXPECT_EQ(undo(), OperatorResult::RUNNING_MODAL);
        EXPECT_EQ(active->pointState().getPickedPoints(), points);
        selectFor(*active, 3);
        active->pointState().requestUiAction(PointPickState::UiAction::DeleteSelected);
        EXPECT_EQ(operators().dispatchModalEvent({}), OperatorResult::RUNNING_MODAL);
        EXPECT_EQ(active->pointState().getPickedPoints().size(), 4u);
        EXPECT_EQ(active->pairColorSlots(), (std::vector<size_t>{colors[0], colors[2]}));
        EXPECT_EQ(undo(), OperatorResult::RUNNING_MODAL);
        EXPECT_EQ(active->pointState().getPickedPoints(), points);
        EXPECT_EQ(active->pairColorSlots(), colors);
        active->pointState().requestUiAction(PointPickState::UiAction::Clear);
        EXPECT_EQ(operators().dispatchModalEvent({}), OperatorResult::RUNNING_MODAL);
        EXPECT_TRUE(active->pointState().getPickedPoints().empty());
        EXPECT_TRUE(active->canUndoPointEdit());
        EXPECT_EQ(undo(), OperatorResult::RUNNING_MODAL);
        EXPECT_EQ(active->pointState().getPickedPoints(), points);
        active->pointState().requestUiAction(PointPickState::UiAction::TogglePreview);
        EXPECT_EQ(operators().dispatchModalEvent({}), OperatorResult::RUNNING_MODAL);
        ASSERT_TRUE(active->session()->preview);
        EXPECT_EQ(undo(), OperatorResult::RUNNING_MODAL);
        EXPECT_FALSE(active->session()->preview);
        EXPECT_EQ(active->pointState().getPickedPoints().size(), 6u);
        EXPECT_EQ(operators().dispatchModalEvent({.type = ModalEvent::Type::ACTION,
                                                  .data = ActionEvent{.action = input::Action::REDO}}),
                  OperatorResult::RUNNING_MODAL);
        EXPECT_EQ(undoHistory().generation(), generation);
        operators().cancelModalOperator();
        EXPECT_EQ(ModelAlignmentOperator::active(), nullptr);
        EXPECT_FALSE(pair_rendering_->independentPanelOverrides());
        EXPECT_EQ(undoHistory().generation(), generation);
        EXPECT_TRUE(operators().poll(BuiltinOp::Undo));
        ASSERT_TRUE(undoHistory().undo().success);
        EXPECT_TRUE(operators().poll(BuiltinOp::Redo));
        EXPECT_EQ(manager_->getNodeTransform("target"), original);
    }

} // namespace lfs::vis::op
