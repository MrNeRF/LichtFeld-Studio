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
#include "io/project/collision_serialization.hpp"
#include "io/project/project_container_internal.hpp"
#include "io/project_document.hpp"
#include "licht_test_support.hpp"
#include "operation/undo_history.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "rendering/viewport_request_builder.hpp"
#include "scene/scene_manager.hpp"
#include "training/project_snapshot_chapters.hpp"
#include "visualizer/visualizer_impl.hpp"
#include <cuda_runtime.h>
#include <fstream>
#include <glm/ext/matrix_relational.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>
#include <semaphore>
#include <string_view>
#include <thread>
#include <tuple>

namespace lfs::vis::gui {
    using namespace core;
    class CollisionJobsTest : public ::testing::Test {
    protected:
        lfs::test::licht::TemporaryDirectory temporary_;
        std::optional<lfs::test::licht::ScopedEnvironmentVariable> home_;
        std::unique_ptr<VisualizerImpl> viewer_;
        std::binary_semaphore release_{0}, dispatched_{0};
        int batches_ = 0;
        int exit_prompts_ = 0;
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
            events::cmd::ShowExitConfirmation::when([this](const auto&) { ++exit_prompts_; });
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
        project::ProjectLifecycle& lifecycle() { return *viewer_->project_lifecycle_; }
        void queueOpen(const std::filesystem::path& path) {
            EXPECT_EQ(lfs::test::licht::require_result(viewer_->projectOpen(path, ProjectSwitchDisposition::DiscardChanges)), ProjectOpenOutcome::Deferred);
        }
        Scene& scene() { return viewer_->getScene(); }
        std::vector<const SceneNode*> colliders() {
            auto result = scene().getNodes();
            std::erase_if(result, [](const auto* node) { return node->type != NodeType::COLLISION; });
            return result;
        }
        void finish() {
            release_.release();
            if (tasks().collision_worker_.valid())
                tasks().collision_worker_.wait();
            tasks().pollCollision();
            ASSERT_TRUE(tasks().collisionJob());
            EXPECT_FALSE(tasks().collisionJob()->running());
            finishOpen();
        }
        void finishOpen() {
            const auto open = viewer_->jobs().active(JobType::ProjectOpen);
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            while (viewer_->jobs().anyRunning(JobType::ProjectOpen) && std::chrono::steady_clock::now() < deadline) {
                viewer_->pumpPostedWorkForProjectWrite();
                std::this_thread::yield();
            }
            EXPECT_FALSE(viewer_->jobs().active(JobType::ProjectOpen));
            if (open)
                EXPECT_FALSE(viewer_->jobs().peek(open->handle));
        }
        void EligibilityAndAllOutcomesSettle() {
            for (int edit = 0; edit < 10; ++edit) {
                SCOPED_TRACE(edit);
                const auto parent = scene().addGroup("Parent");
                scene().setNodeTransform(parent, glm::scale(glm::mat4(1), glm::vec3(2, 3, 4)));
                const auto source = scene().addSplat("Source", lfs::test::licht::make_matrix_splat(), parent);
                scene().setNodeTransform(source, glm::translate(glm::mat4(1), glm::vec3(5, 6, 7)));
                const auto local = glm::inverse(scene().getWorldTransform(source));
                scene().setNodeVisibility(source, false);
                viewer_->getSceneManager()->selectNode(source);
                tasks().collision_source_ = scene().collisionSourceIdentity(source);
                tasks().collision_resolved_size_ = .25f;
                tasks().launchCollision([this, edit](std::stop_token, const ProgressCallback&) {
                    release_.acquire();
                    if (edit == 8)
                        throw std::runtime_error("fixture generation failure");
                    if (edit == 9)
                        return true;
                    tasks().collision_result_ = collision_test::box({0, 0, 0}, {4, 1, 4});
                    return false;
                });
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
                EXPECT_EQ(tasks().collisionJob()->status, edit == 0 || edit == 9 ? JobStatus::Completed : edit == 7 ? JobStatus::Canceled
                                                                                                                    : JobStatus::Failed);
                ASSERT_EQ(colliders().size(), edit == 0 ? 1 : 0);
                if (edit > 0 && edit < 7)
                    EXPECT_EQ(tasks().collisionJob()->error, LOC("collision.source_changed"));
                if (edit == 0) {
                    EXPECT_FLOAT_EQ(tasks().collisionResolvedSize(), .25f);
                    const auto uuid = colliders().front()->uuid;
                    const auto volume = colliders().front()->collision;
                    EXPECT_TRUE(glm::all(glm::equal(scene().getWorldTransform(colliders().front()->id), glm::mat4(1), 1e-5f)));
                    const auto unrelated = scene().addGroup("Unrelated later edit");
                    (void)op::undoHistory().undo();
                    EXPECT_TRUE(colliders().empty());
                    EXPECT_NE(scene().getNodeById(unrelated), nullptr);
                    (void)op::undoHistory().redo();
                    const auto* restored = scene().getNodeByUuid(uuid);
                    ASSERT_NE(restored, nullptr);
                    EXPECT_EQ(restored->parent_id, source);
                    EXPECT_EQ(restored->collision, volume);
                    EXPECT_TRUE(glm::all(glm::equal(restored->transform(), local, 1e-5f)));
                    scene().setNodeTransform(parent, glm::translate(glm::mat4(1), glm::vec3(8, 9, 10)));
                    EXPECT_TRUE(glm::all(glm::equal(scene().getWorldTransform(restored->id), scene().getWorldTransform(source) * local, 1e-5f)));
                }
                scene().removeNodeById(parent);
            }
        }

        void ActualOpenReopenClearAndNewRejectEvenReusedSourceIdentity() {
            const auto path = temporary_.path / "other.licht";
            const auto second_path = temporary_.path / "second.licht";
            const auto first = generate_uuid_v4(), second = generate_uuid_v4();
            for (const auto& [destination, uuid] : {std::pair{path, first}, std::pair{second_path, second}}) {
                auto document = lfs::test::licht::make_empty_document(uuid);
                ASSERT_TRUE(document->edit_scene_graph().upsert_node({.uuid = uuid, .type = "group", .name = "Prepared shell"}));
                ASSERT_TRUE(document->save(destination));
            }
            const auto original = lfs::test::licht::read_file_bytes(path);
            for (const std::string_view replacement : {"Open", "Reopen", "Clear", "New", "Prepared", "Deleted", "Replaced", "SameCommitCopy", "NewGeneration", "Supersede", "InvalidSecond", "StaleEpoch", "CancelClose", "DiscardClose"}) {
                SCOPED_TRACE(replacement);
                const bool reject_prepared = replacement == "Prepared" || replacement == "Deleted" || replacement == "Replaced" || replacement == "SameCommitCopy" || replacement == "NewGeneration";
                const bool retain_source = reject_prepared || replacement == "StaleEpoch" || replacement == "CancelClose" || replacement == "DiscardClose";
                const auto current_path = temporary_.path / (std::string(replacement) + "-current.licht");
                if (reject_prepared) {
                    auto current = lfs::test::licht::make_empty_document(generate_uuid_v4());
                    (void)lfs::test::licht::require_result(current->save(current_path));
                    ASSERT_EQ(lfs::test::licht::require_result(viewer_->projectOpen(current_path, ProjectSwitchDisposition::DiscardChanges)), ProjectOpenOutcome::Opened);
                    ASSERT_NO_FATAL_FAILURE(finishOpen());
                }
                const auto source = scene().addSplat("Source", lfs::test::licht::make_matrix_splat());
                const auto source_uuid = scene().getNodeUuid(source);
                const auto document = lifecycle().boundDocument();
                const auto epoch = lifecycle().documentEpoch();
                const auto sidecar = lfs::io::project::autosave_sidecar_path(current_path);
                std::vector<std::byte> recovery;
                if (reject_prepared) {
                    ASSERT_TRUE(document && document->source_commit_uuid());
                    scene().setNodeVisibility(source, false);
                    op::undoHistory().push(std::make_unique<op::PropertyChangeUndoEntry>("node.visible", true, false, [this, source](const std::any& value) { scene().setNodeVisibility(source, std::any_cast<bool>(value)); }, viewer_->getSceneManager()));
                    ASSERT_TRUE(document->edit_view().dom().set("retained_recovery", true));
                    ASSERT_TRUE(document->save_autosave(sidecar, {.file_uuid = generate_uuid_v4(), .base_explicit_commit_uuid = *document->source_commit_uuid(), .autosave_sequence = 1, .snapshot_uuid = generate_uuid_v4(), .disk_reserve_bytes = 0}));
                    ASSERT_EQ(lfs::test::licht::require_result(lfs::io::project::inspect_autosave_recovery(current_path)).disposition, lfs::io::project::RecoveryDisposition::Offer);
                    recovery = lfs::test::licht::read_file_bytes(sidecar);
                }
                const auto undo_generation = op::undoHistory().generation();
                const auto old_scene = std::tuple{scene().getNodes(), scene().collisionSourceIdentity(source), bool(scene().getNodeById(source)->visible)};
                if (replacement == "Open") {
                    scene().addSplat("Large unselected source", lfs::test::licht::make_splat(100000));
                    scene().consolidateNodeModels();
                }
                auto owned = scene().captureCollisionSource(source);
                EXPECT_EQ(owned.means.size(0), 4);
                EXPECT_EQ(owned.means.cpu().ptr<float>()[0], 1.f);
                batches_ = 0;
                tasks().launchCollision([this, owned = std::move(owned)](std::stop_token stop, const ProgressCallback& progress) {
                    (void)generateCollision(owned, {.01f, .1f, 0, false, false}, stop, [&](float value, const std::string& stage) {
                        if (stage == "Voxelizing collision" && ++batches_ == 2) {
                            dispatched_.release();
                            if (!release_.try_acquire_for(std::chrono::seconds(10)))
                                throw std::runtime_error("Collision stage release timed out");
                        }
                        return progress(value, stage);
                    });
                    return false;
                });
                const auto token = tasks().collision_stop_.get_token();
                ASSERT_TRUE(dispatched_.try_acquire_for(std::chrono::seconds(10))) << tasks().collisionJob()->error;
                if (replacement == "Open" || replacement == "InvalidSecond") {
                    if (replacement == "InvalidSecond")
                        ASSERT_NO_FATAL_FAILURE(queueOpen(path));
                    for (const auto* name : {"missing.licht", "corrupt.licht"}) {
                        const auto invalid = temporary_.path / name;
                        if (*name == 'c')
                            std::ofstream(invalid) << "not a project";
                        EXPECT_FALSE(viewer_->projectOpen(invalid, ProjectSwitchDisposition::DiscardChanges));
                        EXPECT_EQ(token.stop_requested(), replacement == "InvalidSecond");
                    }
                }
                const auto started = std::chrono::steady_clock::now();
                if (replacement != "Clear" && replacement != "New" && replacement != "InvalidSecond")
                    ASSERT_NO_FATAL_FAILURE(queueOpen(path));
                else if (replacement == "Clear")
                    ASSERT_TRUE(viewer_->clearScene());
                else if (replacement == "New")
                    events::cmd::NewProject{.discard_changes = true}.emit();
                if (replacement == "Supersede")
                    ASSERT_NO_FATAL_FAILURE(queueOpen(second_path));
                EXPECT_LT(std::chrono::steady_clock::now() - started, std::chrono::milliseconds(250));
                if (replacement == "Prepared")
                    std::filesystem::rename(path, temporary_.path / "prepared.licht");
                if (replacement == "Deleted" || replacement == "Replaced" || replacement == "SameCommitCopy")
                    ASSERT_TRUE(std::filesystem::remove(path));
                if (replacement == "Replaced")
                    ASSERT_TRUE(std::filesystem::copy_file(second_path, path));
                if (replacement == "SameCommitCopy")
                    lfs::test::licht::write_file_bytes(path, original);
                if (replacement == "NewGeneration") {
                    auto newer = lfs::test::licht::require_result(lfs::io::project::ProjectDocument::open(path));
                    const auto generation = newer.generation();
                    const auto commit = newer.source_commit_uuid();
                    ASSERT_TRUE(newer.edit_view().dom().set("new_generation", true));
                    ASSERT_TRUE(newer.save(path));
                    EXPECT_EQ(newer.project_uuid(), first);
                    EXPECT_GT(newer.generation(), generation);
                    EXPECT_NE(newer.source_commit_uuid(), commit);
                }
                if (replacement == "StaleEpoch")
                    ++tasks().splat_simplify_scene_epoch_;
                if (replacement == "CancelClose" || replacement == "DiscardClose") {
                    ASSERT_TRUE(lifecycle().setAutoSaveOnClose(false));
                    const auto prompts = exit_prompts_;
                    events::cmd::RequestExit{}.emit();
                    EXPECT_FALSE(viewer_->allowclose());
                    EXPECT_EQ(exit_prompts_, prompts + 1);
                    EXPECT_TRUE(lifecycle().isApplicationClosePending());
                    EXPECT_FALSE(tasks().sceneReplacementPending());
                    if (replacement == "DiscardClose")
                        events::cmd::ForceExit{.discard_autosave = true}.emit();
                }
                int callbacks = 0;
                for (int frame = 0; frame < 3; ++frame) {
                    ASSERT_TRUE(viewer_->postWork({.run = [&] { ++callbacks; }, .cancel = [] {}}));
                    viewer_->pumpPostedWorkForProjectWrite();
                    tasks().pollImportCompletion();
                    EXPECT_NE(scene().getNodeById(source), nullptr);
                    EXPECT_FALSE(viewer_->jobs().active(JobType::ProjectOpen));
                }
                EXPECT_EQ(callbacks, 3);
                finish();
                EXPECT_FALSE(tasks().sceneReplacementPending());
                if (!retain_source) {
                    EXPECT_EQ(scene().getNodeByUuid(source_uuid), nullptr);
                    if (replacement != "Clear" && replacement != "New") {
                        const auto expected = replacement == "Supersede" ? second : first;
                        ASSERT_TRUE(lifecycle().boundDocument());
                        EXPECT_EQ(lifecycle().boundDocument()->project_uuid(), expected);
                        EXPECT_NE(scene().getNodeByUuid(expected), nullptr);
                        EXPECT_EQ(lfs::test::licht::require_result(viewer_->projectGetInfo()).hydration_state, "complete");
                    }
                    ASSERT_NE(scene().restoreNodeWithUuid({.uuid = source_uuid, .preferred_id = source, .type = NodeType::SPLAT, .name = "Reused source identity", .model = lfs::test::licht::make_matrix_splat()}), NULL_NODE);
                } else {
                    if (reject_prepared) {
                        EXPECT_EQ(lifecycle().boundDocument(), document);
                        EXPECT_EQ(lifecycle().documentEpoch(), epoch);
                        ASSERT_NE(scene().getNodeById(source), nullptr);
                        EXPECT_EQ((std::tuple{scene().getNodes(), scene().collisionSourceIdentity(source), bool(scene().getNodeById(source)->visible)}), old_scene);
                        EXPECT_EQ(op::undoHistory().generation(), undo_generation);
                        EXPECT_EQ(op::undoHistory().undoCount(), 1u);
                        EXPECT_EQ(lfs::test::licht::read_file_bytes(sidecar), recovery);
                        ASSERT_TRUE(op::undoHistory().undo().success);
                        EXPECT_TRUE(scene().getNodeById(source)->visible);
                        ASSERT_TRUE(op::undoHistory().redo().success);
                        EXPECT_FALSE(scene().getNodeById(source)->visible);
                    }
                    if (replacement == "CancelClose") {
                        events::cmd::CancelExit{}.emit();
                        EXPECT_FALSE(lifecycle().isApplicationClosePending());
                        EXPECT_FALSE(viewer_->getGuiManager()->isExitConfirmationPending());
                    }
                    tasks().pollImportCompletion();
                    EXPECT_NE(scene().getNodeById(source), nullptr);
                    scene().removeNodeById(source);
                }
                if (reject_prepared)
                    lfs::test::licht::write_file_bytes(path, original);
                EXPECT_TRUE(token.stop_requested());
                EXPECT_EQ(batches_, 2);
                EXPECT_EQ(tasks().collisionJob()->status, JobStatus::Canceled);
                EXPECT_TRUE(colliders().empty());
            }
        }
    };

    TEST_F(CollisionJobsTest, EligibilityAndAllOutcomesSettle) {
        EligibilityAndAllOutcomesSettle();
    }

    TEST_F(CollisionJobsTest, ActualOpenReopenClearAndNewRejectEvenReusedSourceIdentity) {
        ActualOpenReopenClearAndNewRejectEvenReusedSourceIdentity();
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
            context.scene_state = manager.buildRenderState({.metadata_only = true});
            const auto items = buildViewportMeshDrawItems(context.scene_state, settings, {0, 0, 1});
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
    using namespace lfs::io::project;
    using namespace lfs::test::licht;
    using collision_test::box;

    class CollisionArchive : public ::testing::Test {
    protected:
        TemporaryDirectory temporary;
        const std::filesystem::path path = temporary.path / "collision.licht";
        Scene scene;
        std::shared_ptr<CollisionVolume> volume = std::make_shared<CollisionVolume>(*box({1, 2, 3}, {3, 1, 2}, .0125f));
        void SetUp() override {
            volume->settings = {.0125f, .35f, 2.5f, true, true};
            const auto parent = scene.addGroup("Parent");
            scene.setNodeTransform(parent, glm::translate(glm::mat4(1), {4, 5, 6}));
            for (int i = 0; i < 13; ++i) {
                const auto id = scene.addCollision(std::to_string(i), i < 12 ? volume : box({0, 0, 0}, {1, 1, 1}), parent);
                scene.setNodeTransform(id, glm::translate(glm::mat4(1), {float(i), 2, 3}));
                scene.getNodeById(id)->visible = i != 0;
            }
            auto document = make_empty_document(generate_uuid_v4());
            auto captured = require_result(capture_scene_graph_state(scene, {}));
            for (const auto& [uuid, payload] : captured.collisions)
                require_status(document->set_collision(uuid, payload));
            document->edit_scene_graph() = require_result(materialize_scene_graph_chapter(std::move(captured)));
            (void)require_result(document->save(path));
        }
    };

    TEST_F(CollisionArchive, BinaryDedupDeferredHydrationAndSaveAsPreserveNodes) {
        auto reader = require_result(ProjectReader::open(path));
        EXPECT_EQ(std::ranges::count_if(reader.chunks(), [](const auto& row) {
                      return row.is_live() && row.key.fourcc == FOURCC_COLV;
                  }),
                  2);
        EXPECT_TRUE(reader.commit().required_reader_capabilities.contains(COLLISION_VOLUMES_V1));
        ReaderOptions old_reader;
        old_reader.reader_capabilities.set(11, false);
        auto incompatible = ProjectReader::open(path, old_reader);
        ASSERT_FALSE(incompatible);
        EXPECT_EQ(incompatible.error().code(), lfs::ErrorCode::Unsupported);
        EXPECT_NE(incompatible.error().user_message().find("newer LichtFeld"), std::string_view::npos);
        ASSERT_TRUE(reader.validate_source_identity());
        auto newer = require_result(ProjectDocument::open(path));
        ASSERT_TRUE(newer.set_collision(scene.getNode("0")->uuid, box({0, 0, 0}, {1, 1, 1})));
        ASSERT_TRUE(newer.save(path));
        EXPECT_FALSE(reader.validate_source_identity());
        std::filesystem::rename(path, temporary.path / "retained.licht");
        auto document = require_result(ProjectDocument::open(path, {.defer_geometry_payloads = true, .retained_source = &reader}));
        EXPECT_EQ(document.source_commit_uuid(), reader.commit().commit_uuid);
        const auto verify_capabilities = [](const ProjectReader& saved) {
            for (const auto& bits : {saved.commit().required_reader_capabilities, saved.commit().required_writer_capabilities}) {
                EXPECT_TRUE(bits.contains(11));
                EXPECT_FALSE(bits.contains(10));
            }
        };
        verify_capabilities(reader);
        const auto expected = require_result(capture_scene_graph(scene, {})).to_bytes();
        for (bool deferred : {true, false}) {
            SCOPED_TRACE(deferred);
            const auto copy = temporary.path / (deferred ? "unloaded.licht" : "loaded.licht");
            ASSERT_TRUE(document.save_as(copy));
            verify_capabilities(require_result(ProjectReader::open(copy)));
            for (const auto& state : document.payload_states())
                EXPECT_EQ(state.loaded, !deferred);
            document = require_result(ProjectDocument::open(copy, {.defer_geometry_payloads = deferred}));
            Scene target;
            auto shell = require_result(document.stage_shell(target));
            ASSERT_TRUE(document.hydrate(target));
            for (const auto* restored : {shell.get(), &target}) {
                // Recapture checks local/parent transforms, visibility, node type and shared payload references.
                EXPECT_EQ(require_result(capture_scene_graph(*restored, {})).to_bytes(), expected);
                const auto* first = restored->getNode("0");
                ASSERT_NE(first, nullptr);
                for (const auto* node : restored->getNodes()) {
                    if (!node->collision)
                        continue;
                    if (node->name != "12")
                        EXPECT_EQ(node->collision, first->collision);
                    EXPECT_EQ(encodeCollision(*node->collision), encodeCollision(*scene.getNodeByUuid(node->uuid)->collision));
                }
            }
        }
    }

    TEST_F(CollisionArchive, ReadAndSaveFailuresAreTransactional) {
        auto document = require_result(ProjectDocument::open(path));
        const auto baseline = read_file_bytes(path);
        lfs::io::project::detail::set_max_payload_materialized_bytes_for_testing(1);
        auto limited = ProjectDocument::open(path);
        lfs::io::project::detail::set_max_payload_materialized_bytes_for_testing(std::nullopt);
        ASSERT_FALSE(limited);
        EXPECT_EQ(limited.error().code(), lfs::ErrorCode::ResourceExhausted);
        const auto bytes = encodeCollision(*volume);
        EXPECT_THROW(decodeCollision(std::span(bytes).first(bytes.size() - 1)), std::invalid_argument);
        CollisionVolume padded{.dimensions = {1, 3, 3}, .edge = 4, .occupied_count = 4, .words = {CollisionVolume::brick_tag | 1, 0x00110011, 0}};
        ASSERT_NO_THROW(padded.validate());
        padded.words[1] |= 2;
        EXPECT_THROW(padded.validate(), std::invalid_argument);
        auto invalid = std::make_shared<CollisionVolume>(*volume);
        invalid->words.front() = 0xffffffff;
        EXPECT_FALSE(document.set_collision(document.payload_states().front().instance_uuid, invalid));
        std::filesystem::create_directory(temporary.path / "directory");
        EXPECT_FALSE(document.save(temporary.path / "directory"));
        EXPECT_EQ(read_file_bytes(path), baseline);
        for (const bool erase : {false, true}) {
            auto reader = require_result(ProjectReader::open(path));
            auto writer = require_result(ProjectWriter::append(path));
            require_status(writer.plan_commit());
            auto corrupt = encodeCollision(*volume);
            corrupt[0] = std::byte{0};
            require_status(writer.preflight(corrupt.size()));
            for (const auto& row : reader.chunks()) {
                if (!row.is_live())
                    continue;
                if (row.key.fourcc == FOURCC_COLV)
                    require_status(erase ? writer.erase(row.key) : writer.write_chunk(row.key, corrupt));
                else
                    require_status(writer.reuse_if_clean(require_result(reader.make_clean_proof(row, 1)), 1));
            }
            require_status(writer.commit());
            auto failed = ProjectDocument::open(path);
            ASSERT_FALSE(failed);
            EXPECT_EQ(failed.error().code(), lfs::ErrorCode::DataLoss);
        }
        Scene unchanged;
        const auto retained = unchanged.addGroup("Retained");
        EXPECT_FALSE(hydrate_scene_graph(require_result(capture_scene_graph(scene, {})), unchanged, {}));
        EXPECT_EQ(unchanged.getNodes().size(), 1);
        EXPECT_NE(unchanged.getNodeById(retained), nullptr);
    }

    TEST(CollisionScene, CpuSourceAndSafePointSnapshotOwnDataAfterDeletion) {
        Scene scene;
        const auto parent = scene.addGroup("Parent");
        scene.setNodeTransform(parent, glm::scale(glm::mat4(1), glm::vec3(10)));
        const auto source = scene.addSplat("Selected hidden source", make_splat(4), parent);
        scene.setTrainingModelNode(source);
        scene.setNodeVisibility(source, false);
        const auto volume = box({1, 2, 3}, {3, 1, 2});
        const auto first = scene.addCollision("First", volume, source);
        const auto second = scene.addCollision("Second", volume, source);
        const auto transform = glm::translate(glm::mat4(1), {-4, 5, -6});
        scene.setNodeTransform(source, transform);
        scene.setNodeTransform(first, transform);
        const auto captured_source = scene.captureCollisionSource(source);
        EXPECT_EQ(captured_source.transform, scene.getWorldTransform(source));
        lfs::training::ProjectSnapshotCpuState captured;
        ASSERT_TRUE(lfs::training::capture_project_snapshot_cpu_state(scene, ParameterManagerSnapshot{}, generate_uuid_v4(), 3, captured));
        const auto first_uuid = scene.getNodeUuid(first), second_uuid = scene.getNodeUuid(second);
        scene.removeNodeById(source);
        EXPECT_EQ(captured_source.means.ptr<float>()[0], 0);
        EXPECT_THROW((void)scene.captureCollisionSource(source), lfs::Exception);
        lfs::training::ProjectSnapshotChapters chapters;
        ASSERT_TRUE(lfs::training::materialize_project_snapshot_cpu_chapters(std::move(captured), chapters));
        ASSERT_EQ(chapters.collisions.size(), 1);
        EXPECT_EQ(chapters.collisions.begin()->second, volume);
        const auto one = chapters.scene_graph.find(first_uuid), two = chapters.scene_graph.find(second_uuid);
        ASSERT_TRUE(one && *one && two && *two);
        for (int axis = 0; axis < 3; ++axis)
            EXPECT_EQ((**one).local_transform[12 + axis], transform[3][axis]);
    }
} // namespace lfs::vis::gui
