/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "app/include/app/mcp_app_utils.hpp"
#include "core/error.hpp"
#include "core/scene.hpp"
#include "mcp/mcp_tools.hpp"
#include "visualizer/visualizer.hpp"

#include <chrono>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <future>
#include <gtest/gtest.h>
#include <mutex>
#include <thread>

namespace {

    class ScopedToolRegistration {
    public:
        explicit ScopedToolRegistration(std::string name) : name_(std::move(name)) {}
        ~ScopedToolRegistration() {
            lfs::mcp::ToolRegistry::instance().unregister_tool(name_);
        }

    private:
        std::string name_;
    };

    class FakeVisualizer final : public lfs::vis::Visualizer {
    public:
        FakeVisualizer() : viewer_thread_id_(std::this_thread::get_id()) {}

        void run() override {}
        void setParameters(const lfs::core::param::TrainingParameters&) override {}
        std::expected<void, std::string> loadPLY(const std::filesystem::path&) override {
            return std::unexpected("not implemented");
        }
        std::expected<void, std::string> addSplatFile(const std::filesystem::path&) override {
            return std::unexpected("not implemented");
        }
        std::expected<void, std::string> loadDataset(const std::filesystem::path&) override {
            return std::unexpected("not implemented");
        }
        std::expected<void, std::string> loadCheckpointForTraining(const std::filesystem::path&) override {
            return std::unexpected("not implemented");
        }
        void consolidateModels() override {}
        std::expected<void, std::string> clearScene() override { return {}; }
        lfs::core::Scene& getScene() override { return scene_; }
        lfs::vis::SceneManager* getSceneManager() override { return nullptr; }
        lfs::vis::RenderingManager* getRenderingManager() override { return nullptr; }

        bool postWork(WorkItem work) override {
            {
                std::lock_guard lock(mutex_);
                if (!accept_work_) {
                    return false;
                }
                ++post_count_;
                work_queue_.push_back(std::move(work));
            }
            cv_.notify_all();
            return true;
        }

        [[nodiscard]] bool isOnViewerThread() const override {
            return std::this_thread::get_id() == viewer_thread_id_;
        }
        [[nodiscard]] bool acceptsPostedWork() const override {
            std::lock_guard lock(mutex_);
            return accept_work_;
        }

        void setShutdownRequestedCallback(std::function<void()>) override {}
        std::expected<void, std::string> startTraining() override {
            return std::unexpected("not implemented");
        }
        lfs::Result<void> projectSave(bool) override {
            return {};
        }
        lfs::Result<void> projectSaveAs(
            const std::filesystem::path&, bool) override {
            return {};
        }
        lfs::Result<lfs::vis::ProjectOpenOutcome> projectOpen(
            const std::filesystem::path&,
            lfs::vis::ProjectSwitchDisposition) override {
            return lfs::vis::ProjectOpenOutcome::Opened;
        }
        lfs::Result<void> projectCompact() override {
            return {};
        }
        lfs::Result<bool> projectIsDirty() override {
            return false;
        }
        lfs::Result<bool> projectHasPath() override {
            return false;
        }
        lfs::Result<lfs::vis::ProjectInfo>
        projectGetInfo() override {
            ++info_calls_;
            lfs::vis::ProjectInfo info;
            info.path = path_;
            info.generation = published_generation_;
            info.project_write_running = false;
            return info;
        }

        lfs::Result<lfs::vis::ProjectWritePoll>
        projectPollWrite() override {
            ++poll_calls_;
            lfs::vis::ProjectWritePoll poll;
            poll.running = remaining_running_ > 0;
            if (remaining_running_ > 0) {
                --remaining_running_;
            }
            poll.generation = poll.running
                                  ? previous_generation_
                                  : published_generation_;
            poll.path = path_;
            return poll;
        }

        void setWaitState(
            std::filesystem::path path,
            const std::uint64_t previous_generation,
            const std::uint64_t published_generation,
            const int remaining_running) {
            path_ = std::move(path);
            previous_generation_ = previous_generation;
            published_generation_ = published_generation;
            remaining_running_ = remaining_running;
            info_calls_ = 0;
            poll_calls_ = 0;
        }

        [[nodiscard]] int infoCalls() const {
            return info_calls_;
        }
        [[nodiscard]] int pollCalls() const {
            return poll_calls_;
        }

        [[nodiscard]] int postCount() const {
            std::lock_guard lock(mutex_);
            return post_count_;
        }

        bool waitForPostedWork(const std::chrono::milliseconds timeout = std::chrono::seconds(1)) {
            std::unique_lock lock(mutex_);
            return cv_.wait_for(lock, timeout, [this] { return !work_queue_.empty(); });
        }

        bool runNextWorkItem() {
            WorkItem item;
            {
                std::lock_guard lock(mutex_);
                if (work_queue_.empty()) {
                    return false;
                }
                item = std::move(work_queue_.front());
                work_queue_.pop_front();
            }

            if (item.run) {
                item.run();
            }
            return true;
        }

        void setAcceptWork(const bool accept_work) {
            std::lock_guard lock(mutex_);
            accept_work_ = accept_work;
        }

    private:
        lfs::core::Scene scene_;
        std::thread::id viewer_thread_id_;
        mutable std::mutex mutex_;
        std::condition_variable cv_;
        std::deque<WorkItem> work_queue_;
        int post_count_ = 0;
        bool accept_work_ = true;
        std::filesystem::path path_;
        std::uint64_t previous_generation_ = 0;
        std::uint64_t published_generation_ = 0;
        int remaining_running_ = 0;
        int info_calls_ = 0;
        int poll_calls_ = 0;
    };

} // namespace

TEST(McpAppUtilsTest, PostAndWaitExecutesInlineOnViewerThread) {
    FakeVisualizer viewer;
    bool ran = false;

    const auto result = lfs::app::post_and_wait(&viewer, [&]() {
        ran = true;
        return nlohmann::json{{"success", true}, {"mode", "inline"}};
    });

    EXPECT_TRUE(ran);
    EXPECT_EQ(viewer.postCount(), 0);
    EXPECT_TRUE(result["success"].get<bool>());
    EXPECT_EQ(result["mode"], "inline");
}

TEST(McpAppUtilsTest, PostAndWaitQueuesAndWaitsOffViewerThread) {
    FakeVisualizer viewer;
    std::promise<nlohmann::json> result_promise;
    auto result_future = result_promise.get_future();

    std::jthread worker([&](std::stop_token) {
        try {
            result_promise.set_value(lfs::app::post_and_wait(&viewer, []() {
                return nlohmann::json{{"success", true}, {"mode", "queued"}};
            }));
        } catch (...) {
            result_promise.set_exception(std::current_exception());
        }
    });

    ASSERT_TRUE(viewer.waitForPostedWork());
    ASSERT_EQ(viewer.postCount(), 1);
    ASSERT_TRUE(viewer.runNextWorkItem());

    const auto result = result_future.get();
    EXPECT_TRUE(result["success"].get<bool>());
    EXPECT_EQ(result["mode"], "queued");
}

TEST(McpAppUtilsTest, PostAndWaitReturnsQueuedExceptionWithoutUnwindingWorkQueue) {
    FakeVisualizer viewer;
    std::promise<void> result_promise;
    auto result_future = result_promise.get_future();

    std::jthread worker([&](std::stop_token) {
        try {
            (void)lfs::app::post_and_wait(&viewer, []() -> nlohmann::json {
                throw std::runtime_error("posted work failed");
            });
            result_promise.set_value();
        } catch (...) {
            result_promise.set_exception(std::current_exception());
        }
    });

    ASSERT_TRUE(viewer.waitForPostedWork());
    EXPECT_NO_THROW(EXPECT_TRUE(viewer.runNextWorkItem()));
    EXPECT_THROW(result_future.get(), std::runtime_error);
}

TEST(McpAppUtilsTest, ToolRegistryHandlerUsingPostAndWaitExecutesInlineOnViewerThread) {
    static constexpr const char* kToolName = "test.mcp.viewer_thread.inline";
    ScopedToolRegistration cleanup(kToolName);
    FakeVisualizer viewer;
    bool handler_ran = false;

    lfs::mcp::ToolRegistry::instance().register_tool(
        lfs::mcp::McpTool{
            .name = kToolName,
            .description = "Viewer-thread inline execution regression test",
            .input_schema = {.type = "object", .properties = nlohmann::json::object(), .required = {}},
            .metadata = {.category = "test", .kind = "command", .runtime = "gui", .thread_affinity = "gui_thread"}},
        [&viewer, &handler_ran](const nlohmann::json&) -> nlohmann::json {
            return lfs::app::post_and_wait(&viewer, [&handler_ran]() {
                handler_ran = true;
                return nlohmann::json{{"success", true}, {"mode", "inline_tool"}};
            });
        });

    const auto result = lfs::mcp::ToolRegistry::instance().call_tool(kToolName, nlohmann::json::object());

    EXPECT_TRUE(handler_ran);
    EXPECT_EQ(viewer.postCount(), 0);
    EXPECT_TRUE(result["success"].get<bool>());
    EXPECT_EQ(result["mode"], "inline_tool");
}

TEST(McpAppUtilsTest, PostAndWaitReturnsShutdownErrorOnViewerThreadWhenWorkRejected) {
    FakeVisualizer viewer;
    viewer.setAcceptWork(false);

    const auto result = lfs::app::post_and_wait(&viewer, []() -> std::expected<void, std::string> {
        return {};
    });

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), "Viewer is shutting down");
}

TEST(McpAppUtilsTest, WaitForProjectGenerationPollsUntilTerminalThenInfosOnce) {
    FakeVisualizer viewer;
    const std::filesystem::path path{"/tmp/wait-generation.licht"};
    viewer.setWaitState(path, 4, 5, 3);

    const auto info = lfs::app::wait_for_project_generation(
        &viewer, 4, path, true);

    ASSERT_TRUE(info) << lfs::format_for_developer(info.error());
    EXPECT_EQ(info->generation, 5u);
    ASSERT_TRUE(info->path.has_value());
    EXPECT_EQ(*info->path, path);
    EXPECT_EQ(viewer.infoCalls(), 1);
    EXPECT_GE(viewer.pollCalls(), 4);
}

TEST(McpAppUtilsTest, WaitForProjectWritePollsUntilTerminalThenInfosOnce) {
    FakeVisualizer viewer;
    const std::filesystem::path path{"/tmp/wait-write.licht"};
    viewer.setWaitState(path, 2, 2, 2);

    const auto info = lfs::app::wait_for_project_write(
        &viewer, "Project compaction");

    ASSERT_TRUE(info) << lfs::format_for_developer(info.error());
    EXPECT_EQ(info->generation, 2u);
    EXPECT_EQ(viewer.infoCalls(), 1);
    EXPECT_GE(viewer.pollCalls(), 3);
}
