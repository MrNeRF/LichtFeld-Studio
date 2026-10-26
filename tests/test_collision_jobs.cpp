/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "collision_fixture.hpp"
#include "core/event_bridge/event_bridge.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/events.hpp"
#include "core/services.hpp"
#include "core/splat_data.hpp"
#include "gui/async_task_manager.hpp"
#include "gui/gui_manager.hpp"
#include "gui/rml_menu_bar.hpp"
#include "input/input_bindings.hpp"
#include "input/input_controller.hpp"
#include "io/project_document.hpp"
#include "licht_test_support.hpp"
#include "operation/undo_history.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "rendering/viewport_request_builder.hpp"
#include "scene/scene_manager.hpp"
#include "visualizer/app_store.hpp"
#include "visualizer/visualizer_impl.hpp"
#include <algorithm>
#include <atomic>
#include <cuda_runtime.h>
#include <fstream>
#include <future>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>
#include <tuple>

namespace lfs::vis::gui {
    using namespace core;
    class CollisionJobsTest : public ::testing::Test {
    protected:
        lfs::test::licht::TemporaryDirectory temporary_;
        std::optional<lfs::test::licht::ScopedEnvironmentVariable> home_;
        std::unique_ptr<VisualizerImpl> viewer_;
        std::promise<void> release_;
        enum class Outcome { Generate,
                             Fail,
                             PublishedExport };
        void SetUp() override {
            int devices = 0;
            if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices)
                GTEST_SKIP() << "Project replacement resets the CUDA arena; a device is required";
            home_.emplace("LFS_HOME", (temporary_.path / "home").string());
            resetServices();
            ViewerOptions options;
            options.show_startup_overlay = false;
            options.project_lifecycle_settings_path = temporary_.path / "lifecycle.json";
            viewer_ = std::make_unique<VisualizerImpl>(options);
        }
        void resetServices() {
            lfs::event::EventBridge::instance().clear_all();
            services().clear();
            op::undoHistory().clear();
        }
        void TearDown() override {
            if (!viewer_)
                return;
            if (tasks().collision_worker_.valid()) {
                tasks().cancelCollision();
                finish();
            }
            viewer_.reset();
            resetServices();
        }
        AsyncTaskManager& tasks() { return viewer_->getGuiManager()->asyncTasks(); }
        Scene& scene() { return viewer_->getScene(); }
        std::vector<const SceneNode*> colliders() {
            auto result = scene().getNodes();
            std::erase_if(result, [](const auto* node) { return node->type != NodeType::COLLISION; });
            return result;
        }
        void start(NodeId source, Outcome outcome = Outcome::Generate) {
            release_ = std::promise<void>{};
            tasks().collision_source_ = scene().collisionSourceIdentity(source);
            tasks().collision_resolved_size_ = .25f;
            tasks().launchCollision([this, release = release_.get_future().share(), outcome](std::stop_token, const ProgressCallback&) {
                release.wait();
                if (outcome == Outcome::Fail)
                    throw std::runtime_error("fixture generation failure");
                if (outcome == Outcome::PublishedExport)
                    return true;
                tasks().collision_result_ = collision_test::box({0, 0, 0}, {4, 1, 4});
                return false;
            });
        }
        std::stop_token launch(std::function<bool(std::stop_token, const ProgressCallback&)> work) {
            tasks().launchCollision(std::move(work));
            return tasks().collision_stop_.get_token();
        }
        void finish() {
            release_.set_value();
            if (tasks().collision_worker_.valid())
                tasks().collision_worker_.wait();
            tasks().pollCollision();
            ASSERT_TRUE(tasks().collisionJob());
            EXPECT_FALSE(tasks().collisionJob()->running());
            EXPECT_FALSE(app_store().collision_state.get().active);
        }
        void expectMatrix(const glm::mat4& actual, const glm::mat4& expected) {
            for (int column = 0; column < 4; ++column)
                for (int row = 0; row < 4; ++row)
                    EXPECT_NEAR(actual[column][row], expected[column][row], 1e-5f);
        }
    };

    TEST_F(CollisionJobsTest, EligibilityAndAllOutcomesSettle) {
        for (int edit = 0; edit < 10; ++edit) {
            SCOPED_TRACE(edit);
            const auto parent = scene().addGroup("Parent");
            scene().setNodeTransform(parent, glm::scale(glm::mat4(1), glm::vec3(2, 3, 4)));
            const auto source = scene().addSplat("Source", lfs::test::licht::make_matrix_splat(), parent);
            scene().setNodeTransform(source, glm::translate(glm::mat4(1), glm::vec3(5, 6, 7)));
            const auto local = glm::inverse(scene().getWorldTransform(source));
            scene().setNodeVisibility(source, false);
            viewer_->getSceneManager()->selectNode(source);
            const auto outcome = edit == 8 ? Outcome::Fail : edit == 9 ? Outcome::PublishedExport
                                                                       : Outcome::Generate;
            start(source, outcome);
            if (edit == 0) {
                scene().setNodeVisibility(source, true);
                scene().renameNode(source, "Renamed");
                scene().invalidateCache();
                scene().invalidateTransformCache();
            } else if (edit == 1)
                scene().removeNodeById(source);
            else if (edit == 2)
                scene().setNodeTransform(source, glm::translate(glm::mat4(1), glm::vec3(1)));
            else if (edit == 3)
                scene().setNodeTransform(parent, glm::scale(glm::mat4(1), glm::vec3(2)));
            else if (edit == 4)
                scene().replaceNodeModel(scene().getNodeById(source)->name, lfs::test::licht::make_matrix_splat());
            else if (edit == 5)
                scene().invalidateBounds(); // Trainer's step-boundary content revision.
            else if (edit == 6) {
                auto& model = *scene().getNodeById(source)->model;
                model.deleted() = Tensor::ones_bool({static_cast<size_t>(model.size())}, Device::CPU);
                model.notify_deleted_mask_changed();
            } else if (edit == 7 || edit == 9)
                tasks().cancelCollision();
            if (edit == 9)
                ASSERT_TRUE(viewer_->clearScene());
            finish();
            const auto expected = edit == 0 || edit == 9 ? JobStatus::Completed
                                  : edit == 7            ? JobStatus::Canceled
                                                         : JobStatus::Failed;
            EXPECT_EQ(tasks().collisionJob()->status, expected);
            ASSERT_EQ(colliders().size(), edit == 0 ? 1 : 0);
            if (edit > 0 && edit < 7)
                EXPECT_EQ(tasks().collisionJob()->error, LOC("collision.source_changed"));
            if (edit == 0) {
                EXPECT_FLOAT_EQ(tasks().collisionResolvedSize(), .25f);
                const auto uuid = colliders().front()->uuid;
                const auto volume = colliders().front()->collision;
                expectMatrix(scene().getWorldTransform(colliders().front()->id), glm::mat4(1));
                const auto unrelated = scene().addGroup("Unrelated later edit");
                (void)op::undoHistory().undo();
                EXPECT_TRUE(colliders().empty());
                EXPECT_NE(scene().getNodeById(unrelated), nullptr);
                (void)op::undoHistory().redo();
                const auto* restored = scene().getNodeByUuid(uuid);
                ASSERT_NE(restored, nullptr);
                EXPECT_EQ(restored->parent_id, source);
                EXPECT_EQ(restored->collision, volume);
                expectMatrix(restored->transform(), local);
                scene().setNodeTransform(parent, glm::translate(glm::mat4(1), glm::vec3(8, 9, 10)));
                expectMatrix(scene().getWorldTransform(restored->id), scene().getWorldTransform(source) * local);
            }
            scene().removeNodeById(parent);
        }
    }

    TEST_F(CollisionJobsTest, ActualOpenReopenClearAndNewRejectEvenReusedSourceIdentity) {
        const auto path = temporary_.path / "other.licht";
        ASSERT_TRUE(lfs::test::licht::make_empty_document(generate_uuid_v4())->save(path));
        for (int replacement = 0; replacement < 4; ++replacement) {
            SCOPED_TRACE(replacement);
            const auto source = scene().addSplat("Source", lfs::test::licht::make_matrix_splat());
            Scene::RestoreNodeDesc reused;
            reused.uuid = scene().getNodeUuid(source);
            reused.preferred_id = source;
            reused.type = NodeType::SPLAT;
            reused.name = "Reused source identity";
            reused.model = lfs::test::licht::make_matrix_splat();
            if (replacement == 0) {
                scene().addSplat("Large unselected source", lfs::test::licht::make_splat(100000));
                scene().consolidateNodeModels();
            }
            auto owned = scene().captureCollisionSource(source);
            EXPECT_EQ(owned.means.size(0), 4);
            EXPECT_FALSE(owned.means.is_view());
            EXPECT_EQ(owned.means.cpu().ptr<float>()[0], 1.f);
            auto dispatched = std::make_shared<std::promise<void>>();
            auto ready = dispatched->get_future();
            release_ = {};
            auto release = release_.get_future().share();
            auto batches = std::make_shared<std::atomic<int>>(0);
            const auto token = launch([dispatched, batches, release, owned = std::move(owned)](std::stop_token stop, const ProgressCallback& progress) {
                (void)generateCollision(owned, {.01f, .1f, 0, false, false}, stop, [&](float value, const std::string& stage) {
                    if (stage == "Voxelizing collision" && ++*batches == 2) {
                        dispatched->set_value();
                        if (release.wait_for(std::chrono::seconds(10)) != std::future_status::ready)
                            throw std::runtime_error("Collision stage release timed out");
                    }
                    return progress(value, stage);
                });
                return false;
            });
            ASSERT_EQ(ready.wait_for(std::chrono::seconds(10)), std::future_status::ready) << tasks().collisionJob()->error << " batches=" << batches->load();
            if (replacement == 0) {
                for (const auto* name : {"missing.licht", "corrupt.licht"}) {
                    const auto invalid = temporary_.path / name;
                    if (*name == 'c')
                        std::ofstream(invalid) << "not a project";
                    EXPECT_FALSE(viewer_->projectOpen(invalid, ProjectSwitchDisposition::DiscardChanges));
                    EXPECT_FALSE(token.stop_requested());
                    EXPECT_TRUE(tasks().collisionWorkerActive());
                    EXPECT_NE(scene().getNodeById(source), nullptr);
                }
            }
            const auto started = std::chrono::steady_clock::now();
            if (replacement < 2) {
                const auto opened = viewer_->projectOpen(path, ProjectSwitchDisposition::DiscardChanges);
                ASSERT_TRUE(opened);
                EXPECT_EQ(*opened, ProjectOpenOutcome::Deferred);
            } else if (replacement == 2)
                ASSERT_TRUE(viewer_->clearScene());
            else
                events::cmd::NewProject{.discard_changes = true}.emit();
            EXPECT_LT(std::chrono::steady_clock::now() - started, std::chrono::milliseconds(250));
            EXPECT_NE(scene().getNodeById(source), nullptr);
            EXPECT_TRUE(tasks().sceneReplacementPending());
            int callbacks = 0;
            for (int frame = 0; frame < 3; ++frame) {
                ASSERT_TRUE(viewer_->postWork({.run = [&] { ++callbacks; }, .cancel = [] {}}));
                viewer_->pumpPostedWorkForProjectWrite();
                tasks().pollImportCompletion();
                EXPECT_NE(scene().getNodeById(source), nullptr);
            }
            EXPECT_EQ(callbacks, 3);
            finish();
            EXPECT_FALSE(tasks().sceneReplacementPending());
            EXPECT_EQ(scene().getNodeById(source), nullptr);
            ASSERT_NE(scene().restoreNodeWithUuid(std::move(reused)), NULL_NODE);
            EXPECT_TRUE(token.stop_requested());
            EXPECT_EQ(batches->load(), 2);
            EXPECT_EQ(tasks().collisionJob()->status, JobStatus::Canceled);
            EXPECT_TRUE(colliders().empty());
        }
    }

    class CollisionUiTest : public ::testing::Test {
    protected:
        SceneManager manager;
        Scene& scene() { return manager.getScene(); }
        void SetUp() override {
            services().set(&manager);
            input::InputBindings::setPersistenceEnabled(false);
        }
        void TearDown() override {
            services().clear();
            input::InputBindings::setPersistenceEnabled(true);
        }
        void enterWalk(RmlMenuBar& bar) { bar.dispatchToolbarAction("set_camera_navigation_mode", "walk"); }
        bool walkEnabled(RmlMenuBar& bar) {
            bar.rebuildToolbarButtons();
            return bar.camera_buttons_.back().enabled;
        }
    };

    TEST_F(CollisionUiTest, WalkRejectionDoesNotEscapeNativeDispatch) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        controller.initialize();
        controller.setCameraNavigationMode(InputController::CameraNavigationMode::Orbit);
        RmlMenuBar bar;
        EXPECT_FALSE(walkEnabled(bar));
        EXPECT_NO_THROW(enterWalk(bar));
        EXPECT_EQ(controller.walk_error, LOC("collision.walk_no_collision"));
        const auto floor = scene().addCollision("Floor", collision_test::box({-8, -1, -8}, {16, 1, 16}));
        scene().setNodeTransform(floor, lfs::rendering::VISUALIZER_TO_DATA_WORLD_AXES_4);
        scene().setNodeVisibility(floor, false);
        EXPECT_TRUE(walkEnabled(bar));
        viewport.camera.t = {0, 20, 0};
        EXPECT_NO_THROW(enterWalk(bar));
        EXPECT_EQ(controller.walk_error, LOC("collision.walk_no_support"));
        EXPECT_EQ(controller.cameraNavigationMode(), InputController::CameraNavigationMode::Orbit);
        EXPECT_EQ(viewport.camera.t, glm::vec3(0, 20, 0));
    }

    TEST_F(CollisionUiTest, DrawRequestsTrackVisibilityTransformsAndOpacityNotSelection) {
        const auto volume = collision_test::box({}, glm::ivec3(1), .05f);
        scene().addCollision("First", volume);
        scene().addCollision("Second", volume);
        scene().addGroup("Other");
        const auto transform = glm::scale(glm::rotate(glm::translate(glm::mat4(1), {3, -2, 5}), .7f, {0, 1, 0}), {-2, 3, .5f});
        scene().setNodeTransform("First", transform);
        RenderSettings settings;
        EXPECT_FLOAT_EQ(settings.collision_preview_opacity, .4f);
        Viewport viewport(640, 480);
        const ViewportRegion region{0, 0, 640, 480};
        FrameContext context{.viewport = viewport, .viewport_region = &region, .scene_manager = &manager, .settings = settings, .render_size = {640, 480}};
        for (const auto [name, visible, opacity] : {std::tuple{"First", true, .4f}, {"Other", true, .4f}, {"", true, .4f}, {"Other", false, 0.f}, {"Other", false, .25f}, {"Other", false, 1.f}, {"", true, 1.f}}) {
            manager.clearSelection();
            if (*name)
                manager.selectNode(name);
            scene().setNodeVisibility("First", visible);
            settings.collision_preview_opacity = opacity;
            const auto items = buildViewportMeshDrawItems({}, settings, {0, 0, 1});
            ASSERT_EQ(items.size(), visible ? 2 : 1);
            if (visible)
                EXPECT_EQ(items[0].model, lfs::rendering::dataWorldTransformToVisualizerWorld(transform));
            for (const auto& item : items) {
                EXPECT_EQ(item.collision, volume);
                EXPECT_FLOAT_EQ(item.collision_opacity, opacity);
            }
        }
        for (bool frustums : {false, true}) {
            context.settings.show_camera_frustums = frustums;
            EXPECT_EQ(buildViewportRenderRequest(context, {640, 480}).require_exact_depth, frustums);
        }
    }
} // namespace lfs::vis::gui
