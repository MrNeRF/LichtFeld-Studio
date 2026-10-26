/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "collision_fixture.hpp"
#include "cuda_backend_test.hpp"
#include "core/event_bridge/event_bridge.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/events.hpp"
#include "core/services.hpp"
#include "core/splat_data.hpp"
#include "gui/async_task_manager.hpp"
#include "gui/gui_manager.hpp"
#include "io/project_document.hpp"
#include "licht_test_support.hpp"
#include "operation/undo_history.hpp"
#include "scene/scene_manager.hpp"
#include "visualizer/app_store.hpp"
#include "visualizer/visualizer_impl.hpp"
#include <cuda_runtime.h>
#include <future>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>

namespace lfs::vis::gui {
    using namespace core;
    class CollisionJobsTest : public lfs::test::CudaBackendTest {
    protected:
        lfs::test::licht::TemporaryDirectory temporary_;
        std::optional<lfs::test::licht::ScopedEnvironmentVariable> home_;
        std::unique_ptr<VisualizerImpl> viewer_;
        std::promise<void> release_;
        enum class Outcome { Generate,
                             Fail,
                             Cancel,
                             PublishedExport };
        void SetUp() override {
            lfs::test::CudaBackendTest::SetUp();
            if (IsSkipped())
                return;
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
            if (tasks().collision_thread_ && tasks().collision_thread_->joinable())
                finish();
            viewer_.reset();
            resetServices();
        }
        AsyncTaskManager& tasks() { return viewer_->getGuiManager()->asyncTasks(); }
        Scene& scene() { return viewer_->getScene(); }
        std::vector<const SceneNode*> colliders() {
            std::vector<const SceneNode*> result;
            for (const auto* node : scene().getNodes())
                if (node->type == NodeType::COLLISION)
                    result.push_back(node);
            return result;
        }
        void start(NodeId source, Outcome outcome = Outcome::Generate) {
            release_ = std::promise<void>{};
            Scene::CollisionSourceIdentity identity;
            ASSERT_EQ(scene().captureCollisionSources(source, &identity).size(), 1);
            tasks().collision_source_ = identity;
            tasks().collision_resolved_size_ = .25f;
            tasks().launchCollision([this, release = release_.get_future().share(), outcome](std::stop_token, const ProgressCallback&) {
                release.wait();
                if (outcome == Outcome::Fail)
                    throw std::runtime_error("fixture generation failure");
                if (outcome == Outcome::Cancel)
                    throw OperationCancelled{};
                if (outcome == Outcome::PublishedExport)
                    return true;
                tasks().collision_result_ = collision_test::box({0, 0, 0}, {4, 1, 4});
                return false;
            });
        }
        void finish() {
            release_.set_value();
            tasks().collision_thread_->join();
            tasks().pollCollision();
            ASSERT_TRUE(tasks().collisionJob());
            EXPECT_FALSE(tasks().collisionJob()->running());
            EXPECT_FALSE(viewer_->jobs().anyRunning(JobType::Collision));
            EXPECT_FALSE(app_store().collision_state.get().active);
        }
        void expectMatrix(const glm::mat4& actual, const glm::mat4& expected) {
            for (int column = 0; column < 4; ++column)
                for (int row = 0; row < 4; ++row)
                    EXPECT_NEAR(actual[column][row], expected[column][row], 1e-5f);
        }
    };

    TEST_F(CollisionJobsTest, EligibilityAndAllOutcomesSettle) {
        for (int edit = 0; edit < 11; ++edit) {
            SCOPED_TRACE(edit);
            const auto parent = scene().addGroup("Parent");
            scene().setNodeTransform(parent, glm::scale(glm::mat4(1), glm::vec3(2, 3, 4)));
            const auto source = scene().addSplat("Source", lfs::test::licht::make_splat(4), parent);
            scene().setNodeTransform(source, glm::translate(glm::mat4(1), glm::vec3(5, 6, 7)));
            const auto local = glm::inverse(scene().getWorldTransform(source));
            scene().setNodeVisibility(source, false);
            viewer_->getSceneManager()->selectNode(source);
            const auto outcome = edit == 8 ? Outcome::Fail : edit == 9 ? Outcome::Cancel
                                                         : edit == 10  ? Outcome::PublishedExport
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
                scene().replaceNodeModel(scene().getNodeById(source)->name, lfs::test::licht::make_splat(4));
            else if (edit == 5)
                scene().invalidateBounds(); // Trainer's step-boundary content revision.
            else if (edit == 6) {
                auto& model = *scene().getNodeById(source)->model;
                model.deleted() = Tensor::ones_bool({static_cast<size_t>(model.size())}, Device::CPU);
                model.notify_deleted_mask_changed();
            } else if (edit == 7 || edit == 10)
                tasks().cancelCollision();
            finish();
            const auto expected = edit == 0 || edit == 10  ? JobStatus::Completed
                                  : edit == 7 || edit == 9 ? JobStatus::Canceled
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
            const auto source = scene().addSplat("Source", lfs::test::licht::make_splat(4));
            Scene::RestoreNodeDesc reused;
            reused.uuid = scene().getNodeUuid(source);
            reused.preferred_id = source;
            reused.type = NodeType::SPLAT;
            reused.name = "Reused source identity";
            reused.model = lfs::test::licht::make_splat(4);
            start(source);
            if (replacement < 2)
                ASSERT_TRUE(viewer_->projectOpen(path, ProjectSwitchDisposition::DiscardChanges));
            else if (replacement == 2)
                ASSERT_TRUE(viewer_->clearScene());
            else
                events::cmd::NewProject{.discard_changes = true}.emit();
            ASSERT_NE(scene().restoreNodeWithUuid(std::move(reused)), NULL_NODE);
            finish();
            EXPECT_EQ(tasks().collisionJob()->status, JobStatus::Canceled);
            EXPECT_TRUE(colliders().empty());
        }
    }

} // namespace lfs::vis::gui
