/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "app/include/app/mcp_sequencer_tools.hpp"
#include "core/scene.hpp"
#include "mcp/mcp_tools.hpp"
#include "python/python_runtime.hpp"
#include "sequencer/animation_clip.hpp"
#include "sequencer/keyframe.hpp"
#include "visualizer/ipc/view_context.hpp"
#include "visualizer/sequencer/sequencer_controller.hpp"
#include "visualizer/visualizer.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <future>
#include <gtest/gtest.h>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

namespace {

    using json = nlohmann::json;

    constexpr std::array<const char*, 12> kSequencerToolNames = {
        "sequencer.get",
        "sequencer.add_keyframe",
        "sequencer.update_keyframe",
        "sequencer.select_keyframe",
        "sequencer.go_to_keyframe",
        "sequencer.delete_keyframe",
        "sequencer.set_easing",
        "sequencer.play_pause",
        "sequencer.clear",
        "sequencer.save_path",
        "sequencer.load_path",
        "sequencer.set_playback_speed",
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
                work_queue_.push_back(std::move(work));
            }
            cv_.notify_all();
            return true;
        }

        [[nodiscard]] bool isOnViewerThread() const override {
            return std::this_thread::get_id() == viewer_thread_id_;
        }

        void setShutdownRequestedCallback(std::function<void()>) override {}
        std::expected<void, std::string> startTraining() override {
            return std::unexpected("not implemented");
        }
        std::expected<std::filesystem::path, std::string> saveCheckpoint(
            const std::optional<std::filesystem::path>&) override {
            return std::unexpected("not implemented");
        }

    private:
        lfs::core::Scene scene_;
        std::thread::id viewer_thread_id_;
        std::mutex mutex_;
        std::condition_variable cv_;
        std::deque<WorkItem> work_queue_;
    };

    struct FakeSequencerBackend {
        struct CameraState {
            glm::vec3 eye{0.0f, 0.0f, 5.0f};
            glm::vec3 target{0.0f, 0.0f, 0.0f};
            glm::vec3 up{0.0f, 1.0f, 0.0f};
            float fov_degrees = 60.0f;
        };

        lfs::vis::SequencerController controller;
        lfs::python::SequencerUIStateData ui_state;
        CameraState camera;
        bool visible = false;

        lfs::app::SequencerToolBackend tool_backend() {
            return lfs::app::SequencerToolBackend{
                .ensure_ready = []() -> std::expected<void, std::string> { return {}; },
                .controller = [this]() -> lfs::vis::SequencerController* { return &controller; },
                .is_visible = [this]() { return visible; },
                .set_visible = [this](const bool value) { visible = value; },
                .ui_state = [this]() { return &ui_state; },
                .add_keyframe = [this]() { add_keyframe(); },
                .update_selected_keyframe = [this]() { update_selected_keyframe(); },
                .select_keyframe = [this](const size_t index) { select_keyframe(index); },
                .go_to_keyframe = [this](const size_t index) { go_to_keyframe(index); },
                .delete_keyframe = [this](const size_t index) { delete_keyframe(index); },
                .set_keyframe_easing = [this](const size_t index, const int easing) { set_keyframe_easing(index, easing); },
                .play_pause = [this]() { controller.togglePlayPause(); },
                .clear = [this]() {
                    controller.clear();
                    sync_selected_index();
                },
                .save_path = [this](const std::string& path) { return controller.saveToJson(path); },
                .load_path = [this](const std::string& path) {
                    const bool loaded = controller.loadFromJson(path);
                    sync_selected_index();
                    return loaded;
                },
                .set_playback_speed = [this](const float speed) {
                    controller.setPlaybackSpeed(speed);
                    ui_state.playback_speed = controller.playbackSpeed();
                },
            };
        }

        lfs::sequencer::KeyframeId add_manual_keyframe(const float time, const glm::vec3 position) {
            const auto id = controller.addKeyframeAtTime(
                lfs::sequencer::Keyframe{
                    .time = time,
                    .position = position,
                    .rotation = lfs::sequencer::IDENTITY_ROTATION,
                    .focal_length_mm = camera.fov_degrees,
                },
                time);
            sync_selected_index();
            return id;
        }

        void set_camera(const lfs::vis::SetViewParams& params) {
            camera.eye = glm::vec3(params.eye[0], params.eye[1], params.eye[2]);
            camera.target = glm::vec3(params.target[0], params.target[1], params.target[2]);
            camera.up = glm::vec3(params.up[0], params.up[1], params.up[2]);
        }

        void set_fov(const float fov_degrees) {
            camera.fov_degrees = fov_degrees;
        }

        std::optional<lfs::vis::ViewInfo> current_view_info() const {
            return lfs::vis::ViewInfo{
                .rotation = {1.0f, 0.0f, 0.0f,
                             0.0f, 1.0f, 0.0f,
                             0.0f, 0.0f, 1.0f},
                .translation = {camera.eye.x, camera.eye.y, camera.eye.z},
                .pivot = {camera.target.x, camera.target.y, camera.target.z},
                .width = 1280,
                .height = 720,
                .fov = camera.fov_degrees,
            };
        }

    private:
        void add_keyframe() {
            const float time = static_cast<float>(controller.timeline().realKeyframeCount());
            controller.addKeyframeAtTime(
                lfs::sequencer::Keyframe{
                    .time = time,
                    .position = camera.eye,
                    .rotation = lfs::sequencer::IDENTITY_ROTATION,
                    .focal_length_mm = camera.fov_degrees,
                },
                time);
            sync_selected_index();
        }

        void update_selected_keyframe() {
            controller.updateSelectedKeyframe(camera.eye, lfs::sequencer::IDENTITY_ROTATION, camera.fov_degrees);
            sync_selected_index();
        }

        void select_keyframe(const size_t index) {
            controller.selectKeyframe(index);
            sync_selected_index();
        }

        void go_to_keyframe(const size_t index) {
            auto& timeline = controller.timeline();
            if (index >= timeline.size())
                return;
            if (!controller.selectKeyframe(index))
                return;
            const auto* const keyframe = timeline.getKeyframe(index);
            if (!keyframe)
                return;
            controller.seek(keyframe->time);
            camera.eye = keyframe->position;
            camera.target = keyframe->position + glm::vec3(0.0f, 0.0f, -1.0f);
            camera.up = glm::vec3(0.0f, 1.0f, 0.0f);
            camera.fov_degrees = keyframe->focal_length_mm;
            sync_selected_index();
        }

        void delete_keyframe(const size_t index) {
            if (index == 0)
                return;
            if (!controller.selectKeyframe(index))
                return;
            controller.removeSelectedKeyframe();
            sync_selected_index();
        }

        void set_keyframe_easing(const size_t index, const int easing) {
            controller.setKeyframeEasing(index, static_cast<lfs::sequencer::EasingType>(easing));
            sync_selected_index();
        }

        void sync_selected_index() {
            ui_state.selected_keyframe = controller.selectedKeyframe().has_value()
                                             ? static_cast<int>(controller.selectedKeyframe().value())
                                             : -1;
        }
    };

    class McpSequencerToolsTest : public ::testing::Test {
    protected:
        void SetUp() override {
            unregister_tools();
            lfs::vis::set_view_callback([this]() { return backend_.current_view_info(); });
            lfs::vis::set_set_view_callback([this](const lfs::vis::SetViewParams& params) {
                backend_.set_camera(params);
            });
            lfs::vis::set_set_fov_callback([this](const float fov) {
                backend_.set_fov(fov);
            });
            lfs::app::register_gui_sequencer_tools(
                lfs::mcp::ToolRegistry::instance(),
                &viewer_,
                backend_.tool_backend());
        }

        void TearDown() override {
            lfs::vis::set_view_callback(nullptr);
            lfs::vis::set_set_view_callback(nullptr);
            lfs::vis::set_set_fov_callback(nullptr);
            unregister_tools();
        }

        static void unregister_tools() {
            for (const auto* name : kSequencerToolNames)
                lfs::mcp::ToolRegistry::instance().unregister_tool(name);
        }

        lfs::mcp::McpTool find_tool(const std::string& name) {
            const auto tools = lfs::mcp::ToolRegistry::instance().list_tools();
            const auto it = std::find_if(tools.begin(), tools.end(), [&](const auto& tool) {
                return tool.name == name;
            });
            EXPECT_NE(it, tools.end());
            return *it;
        }

        std::filesystem::path temp_json_path() const {
            return std::filesystem::temp_directory_path() /
                   ("mcp_sequencer_tools_" +
                    std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
                    ".json");
        }

        FakeVisualizer viewer_;
        FakeSequencerBackend backend_;
    };

} // namespace

TEST_F(McpSequencerToolsTest, RegisteredSchemasAreIdOnly) {
    const auto& select_tool = find_tool("sequencer.select_keyframe");
    EXPECT_TRUE(select_tool.input_schema.properties.contains("keyframe_id"));
    EXPECT_FALSE(select_tool.input_schema.properties.contains("keyframe_index"));
    EXPECT_EQ(select_tool.input_schema.required, std::vector<std::string>({"keyframe_id"}));

    const auto& delete_tool = find_tool("sequencer.delete_keyframe");
    EXPECT_TRUE(delete_tool.input_schema.properties.contains("keyframe_id"));
    EXPECT_FALSE(delete_tool.input_schema.properties.contains("keyframe_index"));

    const auto& easing_tool = find_tool("sequencer.set_easing");
    EXPECT_TRUE(easing_tool.input_schema.properties.contains("keyframe_id"));
    EXPECT_FALSE(easing_tool.input_schema.properties.contains("keyframe_index"));

    const auto result = lfs::mcp::ToolRegistry::instance().call_tool(
        "sequencer.select_keyframe",
        json{{"keyframe_index", 0}});
    ASSERT_TRUE(result.contains("error"));
    EXPECT_EQ(result["error"], "Missing required parameter: keyframe_id");
}

TEST_F(McpSequencerToolsTest, GetUsesStableIdsAndSkipsLoopPoint) {
    const auto id_a = backend_.add_manual_keyframe(0.0f, {0.0f, 0.0f, 0.0f});
    const auto id_b = backend_.add_manual_keyframe(1.0f, {1.0f, 2.0f, 3.0f});
    backend_.controller.selectKeyframeById(id_b);
    backend_.controller.setLoopMode(lfs::vis::LoopMode::LOOP);
    backend_.visible = true;

    const auto result = lfs::mcp::ToolRegistry::instance().call_tool("sequencer.get", json::object());

    ASSERT_TRUE(result["success"].get<bool>());
    EXPECT_EQ(result["selected_keyframe_id"], id_b);
    EXPECT_FALSE(result.contains("selected_keyframe"));
    EXPECT_EQ(result["keyframe_count"], 2);
    ASSERT_EQ(result["keyframes"].size(), 2);
    EXPECT_EQ(result["keyframes"][0]["id"], id_a);
    EXPECT_EQ(result["keyframes"][1]["id"], id_b);
    EXPECT_FALSE(result["keyframes"][0].contains("index"));
    EXPECT_FALSE(result["keyframes"][1].contains("index"));
}

TEST_F(McpSequencerToolsTest, GetTreatsClipOnlyTimelineAsNonEmpty) {
    auto clip = std::make_unique<lfs::sequencer::AnimationClip>("clip-only");
    clip->addTrack(lfs::sequencer::ValueType::Float, "camera.exposure");
    backend_.controller.timeline().setAnimationClip(std::move(clip));

    const auto result = lfs::mcp::ToolRegistry::instance().call_tool("sequencer.get", json::object());

    ASSERT_TRUE(result["success"].get<bool>());
    EXPECT_TRUE(result["has_keyframes"].get<bool>());
    EXPECT_EQ(result["keyframe_count"], 0);
    EXPECT_TRUE(result["keyframes"].empty());
}

TEST_F(McpSequencerToolsTest, SelectKeyframeResolvesByIdAfterReorder) {
    const auto id_a = backend_.add_manual_keyframe(0.0f, {0.0f, 0.0f, 0.0f});
    const auto id_b = backend_.add_manual_keyframe(1.0f, {1.0f, 0.0f, 0.0f});
    const auto id_c = backend_.add_manual_keyframe(2.0f, {2.0f, 0.0f, 0.0f});
    backend_.controller.setKeyframeTimeById(id_a, 5.0f);
    backend_.controller.setKeyframeTimeById(id_c, -1.0f);

    const auto result = lfs::mcp::ToolRegistry::instance().call_tool(
        "sequencer.select_keyframe",
        json{{"keyframe_id", id_a}});

    ASSERT_TRUE(result["success"].get<bool>());
    EXPECT_EQ(result["selected_keyframe_id"], id_a);
    EXPECT_TRUE(backend_.visible);
    ASSERT_TRUE(backend_.controller.selectedKeyframeId().has_value());
    EXPECT_EQ(*backend_.controller.selectedKeyframeId(), id_a);
    EXPECT_NE(id_b, id_c);
}

TEST_F(McpSequencerToolsTest, SetEasingAndDeleteResolveByIdAfterReorder) {
    const auto id_a = backend_.add_manual_keyframe(0.0f, {0.0f, 0.0f, 0.0f});
    const auto id_b = backend_.add_manual_keyframe(1.0f, {1.0f, 0.0f, 0.0f});
    const auto id_c = backend_.add_manual_keyframe(2.0f, {2.0f, 0.0f, 0.0f});
    backend_.controller.setKeyframeTimeById(id_a, 3.0f);
    backend_.controller.setKeyframeTimeById(id_c, -2.0f);

    const auto easing_result = lfs::mcp::ToolRegistry::instance().call_tool(
        "sequencer.set_easing",
        json{{"keyframe_id", id_a}, {"easing", "ease_out"}});
    ASSERT_TRUE(easing_result["success"].get<bool>());
    ASSERT_NE(backend_.controller.timeline().getKeyframeById(id_a), nullptr);
    EXPECT_EQ(
        backend_.controller.timeline().getKeyframeById(id_a)->easing,
        lfs::sequencer::EasingType::EASE_OUT);

    const auto delete_result = lfs::mcp::ToolRegistry::instance().call_tool(
        "sequencer.delete_keyframe",
        json{{"keyframe_id", id_a}});
    ASSERT_TRUE(delete_result["success"].get<bool>());
    EXPECT_EQ(backend_.controller.timeline().getKeyframeById(id_a), nullptr);
    EXPECT_EQ(delete_result["keyframe_count"], 2);
    EXPECT_EQ(delete_result["keyframes"][0]["id"], id_c);
    EXPECT_EQ(delete_result["keyframes"][1]["id"], id_b);
}

TEST_F(McpSequencerToolsTest, AddUpdateAndGoToUseCurrentCameraAndStableIds) {
    const auto add_result = lfs::mcp::ToolRegistry::instance().call_tool(
        "sequencer.add_keyframe",
        json{
            {"eye", json::array({1.0f, 2.0f, 3.0f})},
            {"target", json::array({0.0f, 0.0f, 0.0f})},
            {"fov_degrees", 55.0f},
            {"show_sequencer", false},
        });
    ASSERT_TRUE(add_result["success"].get<bool>());
    ASSERT_EQ(add_result["keyframe_count"], 1);
    const auto keyframe_id = add_result["keyframes"][0]["id"].get<lfs::sequencer::KeyframeId>();
    EXPECT_FALSE(backend_.visible);

    const auto update_result = lfs::mcp::ToolRegistry::instance().call_tool(
        "sequencer.update_keyframe",
        json{
            {"keyframe_id", keyframe_id},
            {"eye", json::array({7.0f, 8.0f, 9.0f})},
            {"target", json::array({1.0f, 1.0f, 1.0f})},
            {"fov_degrees", 42.0f},
            {"show_sequencer", false},
        });
    ASSERT_TRUE(update_result["success"].get<bool>());
    const auto* const keyframe = backend_.controller.timeline().getKeyframeById(keyframe_id);
    ASSERT_NE(keyframe, nullptr);
    EXPECT_FLOAT_EQ(keyframe->position.x, 7.0f);
    EXPECT_FLOAT_EQ(keyframe->position.y, 8.0f);
    EXPECT_FLOAT_EQ(keyframe->position.z, 9.0f);
    EXPECT_FLOAT_EQ(keyframe->focal_length_mm, 42.0f);

    const auto go_to_result = lfs::mcp::ToolRegistry::instance().call_tool(
        "sequencer.go_to_keyframe",
        json{{"keyframe_id", keyframe_id}, {"show_sequencer", false}});
    ASSERT_TRUE(go_to_result["success"].get<bool>());
    EXPECT_EQ(go_to_result["selected_keyframe_id"], keyframe_id);
    EXPECT_EQ(go_to_result["camera"]["eye"], json::array({7.0f, 8.0f, 9.0f}));
    EXPECT_FLOAT_EQ(go_to_result["camera"]["fov_degrees"].get<float>(), 42.0f);
}

TEST_F(McpSequencerToolsTest, PlaybackAndPersistenceToolsRoundTripState) {
    backend_.add_manual_keyframe(0.0f, {0.0f, 0.0f, 0.0f});
    backend_.add_manual_keyframe(1.0f, {1.0f, 1.0f, 1.0f});

    const auto speed_result = lfs::mcp::ToolRegistry::instance().call_tool(
        "sequencer.set_playback_speed",
        json{{"speed", 2.75f}});
    ASSERT_TRUE(speed_result["success"].get<bool>());
    EXPECT_FLOAT_EQ(speed_result["playback_speed"].get<float>(), 2.75f);
    EXPECT_FLOAT_EQ(backend_.controller.playbackSpeed(), 2.75f);

    const auto play_result = lfs::mcp::ToolRegistry::instance().call_tool("sequencer.play_pause", json::object());
    ASSERT_TRUE(play_result["success"].get<bool>());
    EXPECT_TRUE(play_result["toggled"].get<bool>());
    EXPECT_TRUE(backend_.controller.isPlaying());

    const auto path = temp_json_path();
    const auto save_result = lfs::mcp::ToolRegistry::instance().call_tool(
        "sequencer.save_path",
        json{{"path", path.string()}});
    ASSERT_TRUE(save_result["success"].get<bool>());
    EXPECT_TRUE(std::filesystem::exists(path));

    const auto clear_result = lfs::mcp::ToolRegistry::instance().call_tool("sequencer.clear", json::object());
    ASSERT_TRUE(clear_result["success"].get<bool>());
    EXPECT_EQ(clear_result["keyframe_count"], 0);

    const auto load_result = lfs::mcp::ToolRegistry::instance().call_tool(
        "sequencer.load_path",
        json{{"path", path.string()}, {"show_sequencer", false}});
    ASSERT_TRUE(load_result["success"].get<bool>());
    EXPECT_EQ(load_result["keyframe_count"], 2);

    std::filesystem::remove(path);
}
