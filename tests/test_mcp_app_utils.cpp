/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "app/include/app/mcp_app_utils.hpp"
#include "core/scene.hpp"
#include "mcp/mcp_tools.hpp"
#include "visualizer/visualizer.hpp"

#include <chrono>
#include <condition_variable>
#include <deque>
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
        void clearScene() override {}
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
        std::expected<std::filesystem::path, std::string> saveCheckpoint(
            const std::optional<std::filesystem::path>&) override {
            return std::unexpected("not implemented");
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
