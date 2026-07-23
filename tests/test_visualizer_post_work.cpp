/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <SDL3/SDL.h>

#include "core/event_bridge/event_bridge.hpp"
#include "core/event_bus.hpp"
#include "core/guarded_task.hpp"
#include "core/scene.hpp"
#include "core/services.hpp"
#include "input/input_controller.hpp"
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
        std::expected<std::filesystem::path, std::string> saveCheckpoint(
            const std::optional<std::filesystem::path>&) override {
            return std::filesystem::path{};
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
    }

    void TearDown() override {
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
