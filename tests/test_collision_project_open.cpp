/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/event_bridge/event_bridge.hpp"
#include "core/parameter_manager.hpp"
#include "core/scene.hpp"
#include "core/services.hpp"
#include "io/project_document.hpp"
#include "licht_test_support.hpp"
#include "operation/undo_history.hpp"
#include "preferences.hpp"
#include "visualizer_impl.hpp"
#include <condition_variable>
#include <cstdlib>
#include <gtest/gtest.h>

namespace lfs::vis {
    // Gate the real lifecycle's worker settlement without a large file, GPU,
    // driver timing or sleeps determining whether the race is exercised.
    class CollisionProjectOpenTest : public ::testing::Test {
    protected:
        struct Gate {
            std::mutex mutex;
            std::condition_variable changed;
            bool released = false;
        };
        lfs::test::licht::TemporaryDirectory temporary;
        std::optional<std::string> previous_home;
        std::unique_ptr<VisualizerImpl> viewer;
        std::shared_ptr<Gate> gate;
        bool installed = false;
        std::optional<JobHandle> open_job;

        void SetUp() override {
            if (const auto* value = std::getenv("LFS_HOME"))
                previous_home = value;
            setHome((temporary.path / "home").string());
            event::EventBridge::instance().clear_all();
            services().clear();
            op::undoHistory().clear();
            static_cast<void>(setProjectLocationPreference(temporary.path / "projects"));
            ViewerOptions options;
            options.show_startup_overlay = false;
            options.project_lifecycle_settings_path = temporary.path / "lifecycle.json";
            viewer = std::make_unique<VisualizerImpl>(options);
        }

        void TearDown() override {
            release(); // also release a worker if an assertion ended a test early
            viewer.reset();
            op::undoHistory().clear();
            services().clear();
            event::EventBridge::instance().clear_all();
            clearProjectLocationPreference();
            setHome(previous_home.value_or(""));
        }

        static void setHome(const std::string& value) {
#ifdef _WIN32
            (void)_putenv_s("LFS_HOME", value.c_str());
#else
            if (value.empty())
                (void)unsetenv("LFS_HOME");
            else
                (void)setenv("LFS_HOME", value.c_str(), 1);
#endif
        }

        project::ProjectLifecycle& lifecycle() { return *viewer->project_lifecycle_; }

        void begin(bool fail = false) {
            auto pending = std::make_unique<project::ProjectLifecycle::PendingCollisionOpen>();
            open_job = viewer->jobs().init(JobType::ProjectOpen, "Gated collision read");
            if (!open_job)
                throw std::runtime_error("Could not reserve project-open job");
            pending->job = *open_job;
            pending->epoch = lifecycle().epoch_.load();
            pending->scene_serial = lifecycle().scene_mutation_serial_.load();
            pending->parameter_serial = viewer->getParameterManager()->dirtySerial();
            pending->install = [this]() -> lfs::Result<void> {
                auto valid = lifecycle().preflightSwitch(ProjectSwitchDisposition::DiscardChanges);
                if (!valid)
                    return valid;
                installed = true;
                return {};
            };
            gate = std::make_shared<Gate>();
            auto* state = pending.get();
            pending->worker = std::jthread([this, state, gate = gate, fail] {
                viewer->jobs().work(state->job);
                std::unique_lock lock(gate->mutex);
                gate->changed.wait(lock, [&] { return gate->released; });
                if (fail)
                    state->result = lfs::Status::failure(lfs::make_error({.code = lfs::ErrorCode::DataLoss,
                                                                          .domain = lfs::ErrorDomain::IO,
                                                                          .user_message = "Injected collision decode failure",
                                                                          .detection = LFS_SOURCE_SITE_CURRENT()}));
                state->done.store(true, std::memory_order_release);
            });
            lifecycle().pending_collision_open_ = std::move(pending);
        }

        void release() {
            if (gate) {
                {
                    std::lock_guard lock(gate->mutex);
                    gate->released = true;
                }
                gate->changed.notify_all();
            }
        }

        void settle() {
            release();
            // Waiting here belongs to the test, never the GUI's polling path.
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (!lifecycle().pending_collision_open_->done.load(std::memory_order_acquire) &&
                   std::chrono::steady_clock::now() < deadline)
                std::this_thread::yield();
            ASSERT_TRUE(lifecycle().pending_collision_open_->done.load(std::memory_order_acquire));
            lifecycle().settleCollisionOpen();
        }

        void makeOldProjectDirtyAndAutosaveDue() {
            const auto path = temporary.path / "old.licht";
            auto document = lfs::io::project::ProjectDocument::create(core::generate_uuid_v4());
            ASSERT_TRUE(document);
            ASSERT_TRUE(document->save(path));
            ASSERT_TRUE(viewer->projectOpen(path, ProjectSwitchDisposition::DiscardChanges));
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (lifecycle().hydration_.load() != project::ProjectLifecycle::Hydration::Complete &&
                   std::chrono::steady_clock::now() < deadline) {
                std::vector<Visualizer::WorkItem> work;
                {
                    std::lock_guard lock(viewer->work_queue_mutex_);
                    work.swap(viewer->work_queue_);
                }
                for (auto& item : work)
                    if (item.run)
                        item.run();
                std::this_thread::yield();
            }
            ASSERT_EQ(lifecycle().hydration_.load(), project::ProjectLifecycle::Hydration::Complete);
            viewer->getScene().addGroup("Keep old edit");
            lifecycle().markSceneMutation(static_cast<uint32_t>(core::Scene::MutationType::NODE_ADDED));
            lifecycle().settings_.autosave_interval_seconds = 1;
            lifecycle().settings_.autosave_dirty_epoch_threshold = 1;
            lifecycle().settings_.autosave_quiet_seconds = 0;
            lifecycle().last_autosave_at_ = std::chrono::steady_clock::now() - std::chrono::hours(1);
            lifecycle().last_mutation_at_ = lifecycle().last_autosave_at_;
        }

        void changeEpoch() { lifecycle().epoch_.fetch_add(1); }
        void changeParameters() { viewer->getParameterManager()->markDirty(); }
        void queueStopThenOpen() { viewer->pending_training_action_ = VisualizerImpl::PendingTrainingAction::OpenProject; }
    };

    TEST_F(CollisionProjectOpenTest, DueAutosaveWaitsForPendingOpenAndDoesNotDefeatInstallation) {
        makeOldProjectDirtyAndAutosaveDue();
        ASSERT_FALSE(HasFatalFailure());
        begin();
        lifecycle().updateMaintenance();
        EXPECT_FALSE(viewer->jobs().anyRunning(JobType::ProjectWrite));
        EXPECT_FALSE(installed);
        settle();
        EXPECT_TRUE(installed);
        EXPECT_FALSE(viewer->jobs().anyRunning(JobType::ProjectOpen));
    }

    TEST_F(CollisionProjectOpenTest, AutosaveResumesAfterCanceledOpenAndKeepsOldEdits) {
        makeOldProjectDirtyAndAutosaveDue();
        ASSERT_FALSE(HasFatalFailure());
        begin();
        viewer->jobs().requestCancel(*open_job);
        settle();
        EXPECT_FALSE(installed);
        EXPECT_NE(viewer->getScene().getNode("Keep old edit"), nullptr);
        lifecycle().updateMaintenance();
        EXPECT_TRUE(viewer->jobs().anyRunning(JobType::ProjectWrite));
        lifecycle().joinPendingWrite();
    }

    TEST_F(CollisionProjectOpenTest, TrainingPreparationAndViewerStartRejectPendingOpenImmediately) {
        begin();
        bool started = false;
        auto async = lifecycle().prepareTrainingStartProjectAsync([&]() -> lfs::Result<void> {
            started = true;
            return {};
        });
        EXPECT_FALSE(async);
        EXPECT_FALSE(lifecycle().prepareTrainingStartProject());
        EXPECT_FALSE(viewer->startTraining());
        EXPECT_FALSE(started);
        EXPECT_FALSE(lifecycle().isTrainingStartPending());
        EXPECT_FALSE(viewer->jobs().anyRunning(JobType::ProjectWrite));
        settle();
        EXPECT_TRUE(installed);
    }

    TEST_F(CollisionProjectOpenTest, PublicPendingQueryAndProjectInfoAgreeUntilWorkerPublishes) {
        EXPECT_FALSE(viewer->projectOpenPending());
        begin();
        EXPECT_TRUE(viewer->projectOpenPending());
        auto info = viewer->projectGetInfo();
        ASSERT_TRUE(info);
        EXPECT_EQ(info->hydration_state, "preparing");
        EXPECT_FALSE(installed);
        settle();
        EXPECT_TRUE(installed);
        EXPECT_FALSE(viewer->projectOpenPending());
    }

    TEST_F(CollisionProjectOpenTest, StopThenOpenAlsoReportsPending) {
        queueStopThenOpen();
        EXPECT_TRUE(viewer->projectOpenPending());
    }

    TEST_F(CollisionProjectOpenTest, StaleSceneEditDoesNotInstallOrLoseTheEdit) {
        begin();
        viewer->getScene().addGroup("Edit after request");
        lifecycle().markSceneMutation(static_cast<uint32_t>(core::Scene::MutationType::NODE_ADDED));
        settle();
        EXPECT_FALSE(installed);
        EXPECT_NE(viewer->getScene().getNode("Edit after request"), nullptr);
        EXPECT_FALSE(viewer->jobs().anyRunning(JobType::ProjectOpen));
    }

    TEST_F(CollisionProjectOpenTest, StaleDocumentEpochDoesNotInstall) {
        begin();
        changeEpoch();
        settle();
        EXPECT_FALSE(installed);
        EXPECT_FALSE(viewer->jobs().anyRunning(JobType::ProjectOpen));
    }

    TEST_F(CollisionProjectOpenTest, StaleParametersDoNotInstall) {
        begin();
        changeParameters();
        settle();
        EXPECT_FALSE(installed);
    }

    TEST_F(CollisionProjectOpenTest, DecodeFailureSettlesJobWithoutInstalling) {
        begin(true);
        settle();
        EXPECT_FALSE(installed);
        EXPECT_FALSE(viewer->jobs().anyRunning(JobType::ProjectOpen));
        EXPECT_FALSE(viewer->projectOpenPending());
    }

    TEST_F(CollisionProjectOpenTest, ClosePollKeepsWorkerOwnedUntilItsCurrentReadFinishes) {
        begin();
        EXPECT_EQ(lifecycle().beginOrPollCloseSave(), project::ProjectLifecycle::CloseSaveStatus::Saving);
        EXPECT_TRUE(viewer->projectOpenPending());
        EXPECT_FALSE(installed);
        settle();
        EXPECT_FALSE(installed);
        EXPECT_FALSE(viewer->jobs().anyRunning(JobType::ProjectOpen));
        EXPECT_FALSE(viewer->projectOpenPending());
    }

    TEST(CollisionProjectOpenFixture, WritesAProjectWithDeferredCollisionPayload) {
        lfs::test::licht::TemporaryDirectory temporary;
        const auto* requested = std::getenv("LFS_TEST_COLLISION_OPEN_FIXTURE");
        const auto path = requested ? std::filesystem::path(requested) : temporary.path / "collision.licht";
        const auto uuid = core::generate_uuid_v4();
        auto document = lfs::io::project::ProjectDocument::create(core::generate_uuid_v4());
        ASSERT_TRUE(document);
        auto volume = std::make_shared<core::CollisionVolume>();
        volume->dimensions = {4, 4, 4};
        volume->edge = 4;
        volume->words = {1};
        volume->occupied_count = 64;
        ASSERT_TRUE(document->set_collision(uuid, volume));
        ASSERT_TRUE(document->edit_scene_graph().upsert_node({.uuid = uuid, .type = "collision", .name = "Deferred collision", .visible = false, .collision = uuid}));
        ASSERT_TRUE(document->save(path));
        auto reopened = lfs::io::project::ProjectDocument::open(path, {.defer_geometry_payloads = true});
        ASSERT_TRUE(reopened);
        EXPECT_TRUE(reopened->has_deferred_collision_payloads());
    }
} // namespace lfs::vis
