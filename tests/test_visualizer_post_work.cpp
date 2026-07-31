/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <SDL3/SDL.h>

#include "core/event_bridge/event_bridge.hpp"
#include "core/event_bus.hpp"
#include "core/guarded_task.hpp"
#include "core/scene.hpp"
#include "core/services.hpp"
#include "input/input_controller.hpp"
#include "io/project_document.hpp"
#include "operation/undo_history.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "visualizer/core/data_loading_service.hpp"
#include "visualizer/include/visualizer/visualizer.hpp"
#include "visualizer/post_work_utils.hpp"
#include "visualizer/visualizer_impl.hpp"

#include <chrono>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <fstream>
#include <future>
#include <gtest/gtest.h>
#include <mutex>
#include <vector>

namespace {

    class NoopUndoEntry final
        : public lfs::vis::op::UndoEntry {
    public:
        void undo() override {}
        void redo() override {}
        [[nodiscard]] std::string name()
            const override {
            return "test.noop";
        }
    };

    lfs::Error posted_work_cancelled_error() {
        return lfs::make_error(lfs::ErrorInit{
            .code = lfs::ErrorCode::Cancelled,
            .domain = lfs::ErrorDomain::Core,
            .operation_id = lfs::OperationId::generate(),
            .detail = "Viewer is shutting down",
            .detection = LFS_SOURCE_SITE_CURRENT(),
        });
    }

    lfs::core::TaskContext posted_work_context() {
        return {
            .name = "test.posted-work",
            .domain = lfs::ErrorDomain::Core,
            .operation_id = lfs::OperationId::generate(),
            .site = LFS_SOURCE_SITE_CURRENT(),
        };
    }

    class PostedWorkTestVisualizer final : public lfs::vis::Visualizer {
    public:
        void run() override {}
        void setParameters(const lfs::core::param::TrainingParameters&) override {}
        std::expected<void, std::string> loadPLY(const std::filesystem::path&) override { return {}; }
        std::expected<void, std::string> addSplatFile(const std::filesystem::path&) override { return {}; }
        std::expected<void, std::string> loadDataset(const std::filesystem::path&) override { return {}; }
        std::expected<void, std::string> loadCheckpointForTraining(const std::filesystem::path&) override { return {}; }
        void consolidateModels() override {}
        std::expected<void, std::string> clearScene() override { return {}; }
        lfs::core::Scene& getScene() override { return scene_; }
        lfs::vis::SceneManager* getSceneManager() override { return nullptr; }
        lfs::vis::RenderingManager* getRenderingManager() override { return nullptr; }

        bool postWork(WorkItem work) override {
            if (!accepts_work_) {
                return false;
            }
            {
                std::lock_guard lock(mutex_);
                work_.push_back(std::move(work));
            }
            cv_.notify_one();
            return true;
        }

        [[nodiscard]] bool acceptsPostedWork() const override { return accepts_work_; }
        void setShutdownRequestedCallback(std::function<void()>) override {}
        std::expected<void, std::string> startTraining() override { return {}; }
        lfs::Result<void> projectSave(bool) override { return {}; }
        lfs::Result<void> projectSaveAs(
            const std::filesystem::path&, bool) override {
            return {};
        }
        lfs::Result<void> projectOpen(
            const std::filesystem::path&,
            lfs::vis::ProjectSwitchDisposition) override {
            return {};
        }
        lfs::Result<lfs::vis::ProjectInfo>
        projectGetInfo() override {
            return lfs::vis::ProjectInfo{};
        }

        void rejectPostedWork() { accepts_work_ = false; }

        [[nodiscard]] bool waitForWork(const std::chrono::milliseconds timeout) {
            std::unique_lock lock(mutex_);
            return cv_.wait_for(lock, timeout, [this] { return !work_.empty(); });
        }

        void cancelNext() {
            WorkItem work;
            {
                std::lock_guard lock(mutex_);
                work = std::move(work_.front());
                work_.pop_front();
            }
            work.cancel();
        }

    private:
        lfs::core::Scene scene_;
        bool accepts_work_ = true;
        std::mutex mutex_;
        std::condition_variable cv_;
        std::deque<WorkItem> work_;
    };

} // namespace

TEST(VisualizerPostWorkTest, QueuedWorkWakesEventLoop) {
    ASSERT_TRUE(SDL_Init(SDL_INIT_EVENTS));
    SDL_FlushEvents(SDL_EVENT_USER, SDL_EVENT_USER);

    lfs::vis::ViewerOptions options;
    options.show_startup_overlay = false;

    bool ran = false;
    {
        auto viewer = lfs::vis::Visualizer::create(options);
        SDL_FlushEvents(SDL_EVENT_USER, SDL_EVENT_USER);

        EXPECT_FALSE(SDL_HasEvents(SDL_EVENT_USER, SDL_EVENT_USER));
        EXPECT_TRUE(viewer->postWork({
            .run = [&ran]() { ran = true; },
            .cancel = nullptr,
        }));

        EXPECT_FALSE(ran);
        EXPECT_TRUE(SDL_HasEvents(SDL_EVENT_USER, SDL_EVENT_USER));
    }
}

TEST(VisualizerPostedWorkTest, GuardedFastPathSettlesAgainstRealViewer) {
    lfs::vis::ViewerOptions options;
    options.show_startup_overlay = false;
    auto viewer = lfs::vis::Visualizer::create(options);

    auto result = lfs::vis::post_guarded_and_wait<int>(
        *viewer, posted_work_context(), [] { return lfs::Result<int>(17); },
        posted_work_cancelled_error());

    ASSERT_TRUE(result);
    EXPECT_EQ(*result, 17);
}

TEST(VisualizerPostedWorkTest, GuardedQueueRejectionReturnsCancellationWithoutWaiting) {
    PostedWorkTestVisualizer viewer;
    viewer.rejectPostedWork();

    auto future = std::async(std::launch::async, [&viewer] {
        return lfs::vis::post_guarded_and_wait<void>(
            viewer, posted_work_context(), [] { return lfs::Result<void>{}; },
            posted_work_cancelled_error());
    });

    ASSERT_EQ(future.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    auto result = future.get();
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().code(), lfs::ErrorCode::Cancelled);
}

TEST(VisualizerPostedWorkTest, GuardedShutdownCancellationMakesWaitingFutureReady) {
    PostedWorkTestVisualizer viewer;
    auto future = std::async(std::launch::async, [&viewer] {
        return lfs::vis::post_guarded_and_wait<void>(
            viewer, posted_work_context(), [] { return lfs::Result<void>{}; },
            posted_work_cancelled_error());
    });

    ASSERT_TRUE(viewer.waitForWork(std::chrono::seconds(1)));
    viewer.cancelNext();
    ASSERT_EQ(future.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    auto result = future.get();
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().code(), lfs::ErrorCode::Cancelled);
}

class VisualizerImplResetTest : public ::testing::Test {
protected:
    void SetUp() override {
        lfs::event::EventBridge::instance().clear_all();
        lfs::core::event::bus().clear_all();
        lfs::vis::services().clear();
        lfs::vis::op::undoHistory().clear();
    }

    void TearDown() override {
        lfs::vis::op::undoHistory().clear();
        lfs::vis::services().clear();
        lfs::core::event::bus().clear_all();
        lfs::event::EventBridge::instance().clear_all();
    }
};

namespace {

    void write_text_file(const std::filesystem::path& path, const std::string& contents) {
        std::filesystem::create_directories(path.parent_path());
        std::ofstream out(path, std::ios::binary);
        ASSERT_TRUE(out.is_open()) << "Failed to open " << path;
        out << contents;
        out.close();
        ASSERT_TRUE(out.good()) << "Failed to write " << path;
    }

    void write_png(const std::filesystem::path& path) {
        static const std::vector<unsigned char> png_1x1 = {
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
            0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4,
            0x89, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x44, 0x41,
            0x54, 0x78, 0x9C, 0x63, 0xF8, 0xCF, 0xC0, 0xF0,
            0x1F, 0x00, 0x05, 0x00, 0x01, 0xFF, 0x89, 0x99,
            0x3D, 0x1D, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45,
            0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82};

        std::filesystem::create_directories(path.parent_path());
        std::ofstream out(path, std::ios::binary);
        ASSERT_TRUE(out.is_open()) << "Failed to open " << path;
        out.write(reinterpret_cast<const char*>(png_1x1.data()),
                  static_cast<std::streamsize>(png_1x1.size()));
        out.close();
        ASSERT_TRUE(out.good()) << "Failed to write " << path;
    }

    void write_minimal_transforms_dataset(const std::filesystem::path& dataset_path) {
        write_png(dataset_path / "frame_0001.png");
        write_text_file(
            dataset_path / "transforms.json",
            R"({
  "fl_x": 1.0,
  "fl_y": 1.0,
  "cx": 0.5,
  "cy": 0.5,
  "w": 1,
  "h": 1,
  "frames": [
    {
      "file_path": "frame_0001.png",
      "transform_matrix": [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
      ]
    }
  ]
}
)");
    }

    void write_empty_project(
        const std::filesystem::path& path,
        const std::optional<float>
            focal_length_mm = std::nullopt) {
        auto document =
            lfs::io::project::ProjectDocument::create(
                lfs::core::generate_uuid_v4(), 1);
        ASSERT_TRUE(document)
            << lfs::format_for_developer(
                   document.error());
        if (focal_length_mm) {
            const auto updated =
                document->edit_view().dom().set_json(
                    "render_settings.focal_length_mm",
                    *focal_length_mm);
            ASSERT_TRUE(updated)
                << lfs::format_for_developer(
                       updated.error());
        }
        auto saved = document->save(
            path,
            lfs::io::project::
                ProjectDocumentSaveOptions{
                    .commit =
                        {
                            .kind =
                                lfs::io::project::
                                    CommitKind::Explicit,
                            .commit_uuid =
                                lfs::core::
                                    generate_uuid_v4(),
                            .snapshot_uuid = {},
                            .wallclock_unix_ns = 2,
                        },
                    .file_uuid =
                        lfs::core::generate_uuid_v4(),
                    .index_compression =
                        lfs::io::project::
                            IndexCompression::
                                StoredForDeterministicTests,
                    .disk_reserve_bytes = 0,
                });
        ASSERT_TRUE(saved)
            << lfs::format_for_developer(
                   saved.error());
    }

    void write_invalid_phase_a_project(
        const std::filesystem::path& path,
        const bool corrupt_session) {
        write_empty_project(path);
        auto reader =
            lfs::io::project::ProjectReader::open(
                path);
        ASSERT_TRUE(reader)
            << lfs::format_for_developer(
                   reader.error());
        auto writer =
            lfs::io::project::ProjectWriter::append(
                path,
                {
                    .compatibility = {},
                    .index_compression =
                        lfs::io::project::
                            IndexCompression::
                                StoredForDeterministicTests,
                    .disk_reserve_bytes = 0,
                    .boundary_observer = {},
                });
        ASSERT_TRUE(writer)
            << lfs::format_for_developer(
                   writer.error());
        ASSERT_TRUE(writer->plan_commit(
            {
                .kind =
                    lfs::io::project::
                        CommitKind::Explicit,
                .commit_uuid =
                    lfs::core::
                        generate_uuid_v4(),
                .snapshot_uuid = {},
                .wallclock_unix_ns = 3,
            }));

        const std::string invalid_json =
            corrupt_session
                ? R"({"version":1,"render_settings":{"focal_length_mm":"invalid"}})"
                : R"({"version":1,"active_strategy":"not-a-strategy","presets":{},"dataset":{}})";
        const auto invalid_bytes =
            std::as_bytes(
                std::span(invalid_json));
        ASSERT_TRUE(writer->preflight(
            invalid_bytes.size()));
        const std::string target =
            corrupt_session ? "VIEW" : "PRMS";
        bool replaced = false;
        for (const auto& chunk :
             reader->chunks()) {
            if (!chunk.is_live()) {
                continue;
            }
            if (chunk.key.fourcc.to_string() ==
                target) {
                replaced = true;
                const auto written =
                    writer->write_chunk(
                        chunk.key,
                        invalid_bytes,
                        {
                            .chunk_version =
                                chunk.chunk_version,
                            .compression =
                                lfs::io::project::
                                    Compression::
                                        Stored,
                        });
                ASSERT_TRUE(written)
                    << lfs::format_for_developer(
                           written.error());
                continue;
            }
            auto proof =
                reader->make_clean_proof(
                    chunk, 1);
            ASSERT_TRUE(proof)
                << lfs::format_for_developer(
                       proof.error());
            const auto reused =
                writer->reuse_if_clean(
                    *proof, 1);
            ASSERT_TRUE(reused)
                << lfs::format_for_developer(
                       reused.error());
        }
        ASSERT_TRUE(replaced);
        const auto committed =
            writer->commit();
        ASSERT_TRUE(committed)
            << lfs::format_for_developer(
                   committed.error());
    }

} // namespace

namespace lfs::vis {

    TEST_F(VisualizerImplResetTest, DestructorClearsSharedEventBridgeHandlers) {
        ViewerOptions options;
        options.show_startup_overlay = false;

        {
            VisualizerImpl viewer(options);
            EXPECT_GT(lfs::event::EventBridge::instance().handler_count(
                          typeid(lfs::core::events::cmd::ResetTraining)),
                      0u);
        }

        EXPECT_EQ(lfs::event::EventBridge::instance().handler_count(
                      typeid(lfs::core::events::cmd::ResetTraining)),
                  0u);
    }

    TEST_F(VisualizerImplResetTest,
           SuccessfulProjectOpenClearsUndoHistory) {
        const auto temporary =
            std::filesystem::temp_directory_path() /
            ("lfs-p6-open-" +
             lfs::core::generate_uuid_v4()
                 .to_string());
        std::filesystem::create_directories(
            temporary);
        const auto project_path =
            temporary / "empty.licht";
        write_empty_project(project_path);

        ViewerOptions options;
        options.show_startup_overlay = false;
        options.project_lifecycle_settings_path =
            temporary / "lifecycle.json";
        {
            VisualizerImpl viewer(options);
            ASSERT_NE(
                viewer.getScene().addGroup("Current"),
                lfs::core::NULL_NODE);
            op::undoHistory().push(
                std::make_unique<NoopUndoEntry>());
            ASSERT_EQ(
                op::undoHistory().undoCount(), 1u);

            const auto opened =
                viewer.projectOpen(
                    project_path,
                    ProjectSwitchDisposition::
                        DiscardChanges);
            ASSERT_TRUE(opened)
                << lfs::format_for_developer(
                       opened.error());
            EXPECT_EQ(
                op::undoHistory().undoCount(), 0u);
            EXPECT_EQ(
                viewer.getScene().getNodeCount(), 0u);
        }

        std::error_code ignored;
        std::filesystem::remove_all(
            temporary, ignored);
    }

    TEST_F(VisualizerImplResetTest,
           FailedOpenOverOpenPreservesCurrentSceneAndUndoHistory) {
        const auto temporary =
            std::filesystem::temp_directory_path() /
            ("lfs-p6-switch-" +
             lfs::core::generate_uuid_v4()
                 .to_string());
        std::filesystem::create_directories(
            temporary);
        const auto project_path =
            temporary / "first.licht";
        write_empty_project(project_path);

        ViewerOptions options;
        options.show_startup_overlay = false;
        options.project_lifecycle_settings_path =
            temporary / "lifecycle.json";
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(
                viewer.projectOpen(project_path));
            ASSERT_NE(
                viewer.getScene().addGroup(
                    "Current after first open"),
                lfs::core::NULL_NODE);
            op::undoHistory().push(
                std::make_unique<NoopUndoEntry>());

            const auto failed = viewer.projectOpen(
                temporary / "missing.licht",
                ProjectSwitchDisposition::
                    DiscardChanges);
            ASSERT_FALSE(failed);
            EXPECT_NE(
                viewer.getScene().getNode(
                    "Current after first open"),
                nullptr);
            EXPECT_EQ(
                op::undoHistory().undoCount(), 1u);
        }

        std::error_code ignored;
        std::filesystem::remove_all(
            temporary, ignored);
    }

    TEST_F(VisualizerImplResetTest,
           PhaseAParameterAndSessionFailuresPreserveCurrentProject) {
        const auto temporary =
            std::filesystem::temp_directory_path() /
            ("lfs-p6-phase-a-" +
             lfs::core::generate_uuid_v4()
                 .to_string());
        std::filesystem::create_directories(
            temporary);
        const auto bad_parameters =
            temporary / "bad-parameters.licht";
        const auto bad_session =
            temporary / "bad-session.licht";
        write_invalid_phase_a_project(
            bad_parameters, false);
        write_invalid_phase_a_project(
            bad_session, true);

        ViewerOptions options;
        options.show_startup_overlay = false;
        options.project_lifecycle_settings_path =
            temporary / "lifecycle.json";
        {
            VisualizerImpl viewer(options);
            ASSERT_NE(
                viewer.getScene().addGroup(
                    "Current survives Phase A"),
                lfs::core::NULL_NODE);
            op::undoHistory().push(
                std::make_unique<NoopUndoEntry>());

            for (const auto& candidate :
                 {bad_parameters, bad_session}) {
                const auto failed =
                    viewer.projectOpen(
                        candidate,
                        ProjectSwitchDisposition::
                            DiscardChanges);
                ASSERT_FALSE(failed);
                EXPECT_NE(
                    viewer.getScene().getNode(
                        "Current survives Phase A"),
                    nullptr);
                EXPECT_EQ(
                    op::undoHistory().undoCount(),
                    1u);
            }
        }

        std::error_code ignored;
        std::filesystem::remove_all(
            temporary, ignored);
    }

    TEST_F(VisualizerImplResetTest,
           DirtyProjectSwitchRequiresExplicitDiscardAuthorization) {
        const auto temporary =
            std::filesystem::temp_directory_path() /
            ("lfs-p6-dirty-switch-" +
             lfs::core::generate_uuid_v4()
                 .to_string());
        std::filesystem::create_directories(
            temporary);
        const auto project_path =
            temporary / "candidate.licht";
        write_empty_project(project_path);

        ViewerOptions options;
        options.show_startup_overlay = false;
        options.project_lifecycle_settings_path =
            temporary / "lifecycle.json";
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(
                viewer.getParameterManager()
                    ->ensureLoaded());
            viewer.input_controller_ =
                std::make_unique<InputController>(
                    nullptr,
                    viewer.getViewport());
            ASSERT_NE(
                viewer.getScene().addGroup(
                    "Unsaved current project"),
                lfs::core::NULL_NODE);

            bool drag_drop_prompted = false;
            std::filesystem::path prompted_path;
            lfs::core::events::cmd::
                ShowProjectSwitchConfirmation::
                    when([&](const auto& event) {
                        drag_drop_prompted = true;
                        prompted_path =
                            event.path;
                    });
            lfs::core::events::cmd::
                ProjectOpen{
                    .path = project_path}
                    .emit();
            EXPECT_TRUE(drag_drop_prompted);
            EXPECT_EQ(
                prompted_path, project_path);
            EXPECT_NE(
                viewer.getScene().getNode(
                    "Unsaved current project"),
                nullptr);

            const auto blocked =
                viewer.projectOpen(project_path);
            ASSERT_FALSE(blocked);
            EXPECT_EQ(
                blocked.error().code(),
                lfs::ErrorCode::
                    FailedPrecondition);
            EXPECT_EQ(
                blocked.error().user_message(),
                "The current project has unsaved changes.");
            EXPECT_NE(
                viewer.getScene().getNode(
                    "Unsaved current project"),
                nullptr);

            ASSERT_TRUE(viewer.projectOpen(
                project_path,
                ProjectSwitchDisposition::
                    DiscardChanges));
            EXPECT_EQ(
                viewer.getScene().getNodeCount(),
                0u);
        }

        std::error_code ignored;
        std::filesystem::remove_all(
            temporary, ignored);
    }

    TEST_F(VisualizerImplResetTest,
           NewProjectDirtyGateRunsBelowEveryCommandEntry) {
        const auto temporary =
            std::filesystem::temp_directory_path() /
            ("lfs-p6-dirty-new-" +
             lfs::core::generate_uuid_v4()
                 .to_string());
        std::filesystem::create_directories(
            temporary);

        ViewerOptions options;
        options.show_startup_overlay = false;
        options.project_lifecycle_settings_path =
            temporary / "lifecycle.json";
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(
                viewer.getParameterManager()
                    ->ensureLoaded());
            viewer.input_controller_ =
                std::make_unique<InputController>(
                    nullptr,
                    viewer.getViewport());
            ASSERT_NE(
                viewer.getScene().addGroup(
                    "Unsaved current project"),
                lfs::core::NULL_NODE);

            lfs::core::events::cmd::
                NewProject{}
                    .emit();
            EXPECT_NE(
                viewer.getScene().getNode(
                    "Unsaved current project"),
                nullptr);

            lfs::core::events::cmd::
                NewProject{
                    .discard_changes = true}
                    .emit();
            EXPECT_EQ(
                viewer.getScene().getNodeCount(),
                0u);
        }

        std::error_code ignored;
        std::filesystem::remove_all(
            temporary, ignored);
    }

    TEST_F(VisualizerImplResetTest,
           FileExitRoutesThroughCloseSaveStateMachine) {
        const auto temporary =
            std::filesystem::temp_directory_path() /
            ("lfs-p6-file-exit-" +
             lfs::core::generate_uuid_v4()
                 .to_string());
        std::filesystem::create_directories(
            temporary);
        const auto project_path =
            temporary / "session.licht";
        constexpr float restored_focal_length =
            73.0f;
        write_empty_project(
            project_path, restored_focal_length);

        ViewerOptions options;
        options.show_startup_overlay = false;
        options.project_lifecycle_settings_path =
            temporary / "lifecycle.json";
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(
                viewer.getParameterManager()
                    ->ensureLoaded());
            viewer.input_controller_ =
                std::make_unique<InputController>(
                    nullptr,
                    viewer.getViewport());
            const auto opened =
                viewer.projectOpen(project_path);
            ASSERT_TRUE(opened)
                << lfs::format_for_developer(
                       opened.error());
            ASSERT_TRUE(
                viewer
                    .isProjectSessionRestorePending());
            const auto before =
                viewer.projectGetInfo();
            ASSERT_TRUE(before);
            EXPECT_FALSE(before->dirty);

            ASSERT_NE(
                viewer.getScene().addGroup(
                    "Unsaved before File Exit"),
                lfs::core::NULL_NODE);
            auto dirty = viewer.projectGetInfo();
            ASSERT_TRUE(dirty);
            ASSERT_TRUE(dirty->dirty);

            lfs::core::events::cmd::
                RequestExit{}
                    .emit();
            ASSERT_TRUE(
                viewer.getWindowManager()
                    ->shouldClose());
            ASSERT_FALSE(
                viewer.getGuiManager()
                    ->isForceExit());

            EXPECT_FALSE(viewer.allowclose());
            EXPECT_FALSE(
                viewer.getWindowManager()
                    ->shouldClose());
            EXPECT_EQ(
                viewer.project_lifecycle_
                    ->beginOrPollCloseSave(),
                project::ProjectLifecycle::
                    CloseSaveStatus::Saving);

            const auto deadline =
                std::chrono::steady_clock::now() +
                std::chrono::seconds(10);
            while (!viewer.getWindowManager()
                        ->shouldClose() &&
                   std::chrono::steady_clock::now() <
                       deadline) {
                std::vector<Visualizer::WorkItem>
                    work;
                {
                    std::lock_guard lock(
                        viewer.work_queue_mutex_);
                    work.swap(viewer.work_queue_);
                }
                for (auto& item : work) {
                    if (item.run) {
                        item.run();
                    }
                }
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(2));
            }
            ASSERT_TRUE(
                viewer.getWindowManager()
                    ->shouldClose());
            ASSERT_EQ(
                viewer.project_lifecycle_
                    ->beginOrPollCloseSave(),
                project::ProjectLifecycle::
                    CloseSaveStatus::Succeeded);

            const auto after =
                viewer.projectGetInfo();
            ASSERT_TRUE(after);
            EXPECT_GT(
                after->generation,
                before->generation);
            EXPECT_FALSE(after->dirty);
            EXPECT_TRUE(viewer.allowclose());

            auto reopened =
                lfs::io::project::
                    ProjectDocument::open(
                        project_path);
            ASSERT_TRUE(reopened)
                << lfs::format_for_developer(
                       reopened.error());
            const auto focal_length =
                reopened->view().dom().get_json(
                    "render_settings.focal_length_mm");
            ASSERT_TRUE(focal_length);
            ASSERT_TRUE(
                focal_length->is_number());
            EXPECT_FLOAT_EQ(
                focal_length->get<float>(),
                restored_focal_length);
        }

        std::error_code ignored;
        std::filesystem::remove_all(
            temporary, ignored);
    }

    TEST_F(VisualizerImplResetTest,
           CancelExitAndNextWindowAttemptRecoverFromFailedCloseSave) {
        const auto temporary =
            std::filesystem::temp_directory_path() /
            ("lfs-p6-close-retry-" +
             lfs::core::generate_uuid_v4()
                 .to_string());
        std::filesystem::create_directories(
            temporary);
        const auto project_path =
            temporary / "session.licht";
        const auto backup_path =
            temporary / "session.backup";

        ViewerOptions options;
        options.show_startup_overlay = false;
        options.project_lifecycle_settings_path =
            temporary / "lifecycle.json";
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(
                viewer.getParameterManager()
                    ->ensureLoaded());
            viewer.input_controller_ =
                std::make_unique<InputController>(
                    nullptr,
                    viewer.getViewport());
            ASSERT_NE(
                viewer.getScene().addGroup("Saved"),
                lfs::core::NULL_NODE);
            const auto initial_save =
                viewer.projectSaveAs(
                    project_path, false);
            ASSERT_TRUE(initial_save)
                << lfs::format_for_developer(
                       initial_save.error());
            std::filesystem::rename(
                project_path, backup_path);
            std::filesystem::create_directory(
                project_path);
            ASSERT_NE(
                viewer.getScene().addGroup(
                    "Unsaved before failed close"),
                lfs::core::NULL_NODE);

            lfs::core::events::cmd::
                RequestExit{}
                    .emit();
            ASSERT_FALSE(viewer.allowclose());

            const auto drain_until_close =
                [&viewer] {
                    const auto deadline =
                        std::chrono::
                            steady_clock::now() +
                        std::chrono::seconds(10);
                    while (!viewer
                                .getWindowManager()
                                ->shouldClose() &&
                           std::chrono::
                                   steady_clock::now() <
                               deadline) {
                        std::vector<
                            Visualizer::WorkItem>
                            work;
                        {
                            std::lock_guard lock(
                                viewer
                                    .work_queue_mutex_);
                            work.swap(
                                viewer.work_queue_);
                        }
                        for (auto& item : work) {
                            if (item.run) {
                                item.run();
                            }
                        }
                        std::this_thread::sleep_for(
                            std::chrono::
                                milliseconds(2));
                    }
                };
            drain_until_close();
            ASSERT_TRUE(
                viewer.getWindowManager()
                    ->shouldClose());
            ASSERT_EQ(
                viewer.project_lifecycle_
                    ->beginOrPollCloseSave(),
                project::ProjectLifecycle::
                    CloseSaveStatus::Failed);

            std::filesystem::remove_all(
                project_path);
            std::filesystem::rename(
                backup_path, project_path);
            lfs::core::events::cmd::
                CancelExit{}
                    .emit();
            EXPECT_FALSE(
                viewer.getWindowManager()
                    ->shouldClose());

            viewer.getWindowManager()
                ->requestClose();
            EXPECT_FALSE(viewer.allowclose());
            EXPECT_EQ(
                viewer.project_lifecycle_
                    ->beginOrPollCloseSave(),
                project::ProjectLifecycle::
                    CloseSaveStatus::Saving);

            drain_until_close();
            ASSERT_TRUE(
                viewer.getWindowManager()
                    ->shouldClose());
            EXPECT_EQ(
                viewer.project_lifecycle_
                    ->beginOrPollCloseSave(),
                project::ProjectLifecycle::
                    CloseSaveStatus::Succeeded);
            EXPECT_TRUE(viewer.allowclose());
        }

        std::error_code ignored;
        std::filesystem::remove_all(
            temporary, ignored);
    }

    TEST_F(VisualizerImplResetTest, ResetTrainingPreservesExplicitInitPath) {
        ViewerOptions options;
        options.show_startup_overlay = false;

        const auto dataset_path = std::filesystem::temp_directory_path() / "lfs_reset_preserves_init_dataset";
        std::filesystem::create_directories(dataset_path);

        VisualizerImpl viewer(options);
        viewer.getSceneManager()->changeContentType(SceneManager::ContentType::Dataset);
        viewer.getSceneManager()->setDatasetPath(dataset_path);

        lfs::core::param::TrainingParameters params;
        params.init_path = "seed_points.ply";
        viewer.getDataLoader()->setParameters(params);

        lfs::core::events::cmd::ResetTraining{}.emit();

        ASSERT_TRUE(viewer.getDataLoader()->getParameters().init_path.has_value());
        EXPECT_EQ(*viewer.getDataLoader()->getParameters().init_path, "seed_points.ply");

        std::error_code ec;
        std::filesystem::remove_all(dataset_path, ec);
    }

    TEST_F(VisualizerImplResetTest, ResetTrainingPreservesViewportCameraAfterSuccessfulReload) {
        ViewerOptions options;
        options.show_startup_overlay = false;

        const auto dataset_path = std::filesystem::temp_directory_path() / "lfs_reset_preserves_camera_dataset";
        std::error_code ec;
        std::filesystem::remove_all(dataset_path, ec);
        write_minimal_transforms_dataset(dataset_path);

        VisualizerImpl viewer(options);
        InputController controller(nullptr, viewer.getViewport());

        viewer.getSceneManager()->changeContentType(SceneManager::ContentType::Dataset);
        viewer.getSceneManager()->setDatasetPath(dataset_path);

        lfs::core::param::TrainingParameters params;
        params.dataset.data_path = dataset_path;
        viewer.getDataLoader()->setParameters(params);

        const glm::vec3 preserved_eye(2.0f, 3.0f, 4.0f);
        const glm::vec3 preserved_target(-1.0f, 0.5f, 1.5f);
        viewer.getViewport().camera.R = lfs::rendering::makeVisualizerLookAtRotation(preserved_eye, preserved_target);
        viewer.getViewport().camera.t = preserved_eye;
        viewer.getViewport().camera.pivot = preserved_target;

        viewer.getViewport().camera.home_t = glm::vec3(100.0f, 200.0f, 300.0f);
        viewer.getViewport().camera.home_pivot = glm::vec3(10.0f, 20.0f, 30.0f);
        viewer.getViewport().camera.home_R = glm::mat3(1.0f);

        const auto preserved_camera = viewer.getViewport().camera;

        lfs::core::events::cmd::ResetTraining{}.emit();

        ASSERT_NE(viewer.getTrainer(), nullptr);
        EXPECT_EQ(viewer.getSceneManager()->getScene().getAllCameras().size(), 1u);

        EXPECT_EQ(viewer.getViewport().camera.t, preserved_camera.t);
        EXPECT_EQ(viewer.getViewport().camera.pivot, preserved_camera.pivot);
        EXPECT_EQ(viewer.getViewport().camera.home_t, preserved_camera.home_t);
        EXPECT_EQ(viewer.getViewport().camera.home_pivot, preserved_camera.home_pivot);
        EXPECT_EQ(viewer.getViewport().camera.R[0], preserved_camera.R[0]);
        EXPECT_EQ(viewer.getViewport().camera.R[1], preserved_camera.R[1]);
        EXPECT_EQ(viewer.getViewport().camera.R[2], preserved_camera.R[2]);
        EXPECT_EQ(viewer.getViewport().camera.home_R[0], preserved_camera.home_R[0]);
        EXPECT_EQ(viewer.getViewport().camera.home_R[1], preserved_camera.home_R[1]);
        EXPECT_EQ(viewer.getViewport().camera.home_R[2], preserved_camera.home_R[2]);

        std::filesystem::remove_all(dataset_path, ec);
    }

} // namespace lfs::vis
