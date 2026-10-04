/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "app/mcp_screen_tools.hpp"
#include "input/injected_pointer.hpp"
#include "mcp/mcp_protocol.hpp"
#include "mcp/mcp_tools.hpp"
#include "visualizer/visualizer_impl.hpp"

#include <SDL3/SDL_init.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <string>

namespace {

    using json = nlohmann::json;

    TEST(McpInjectedPointer, PositionSurvivesFramesAndNativeEventsCannotStealGesture) {
        using namespace lfs::vis::input;
        ASSERT_TRUE(SDL_InitSubSystem(SDL_INIT_EVENTS));
        injectPointerMove(415, 628);
        SDL_Event move{};
        move.type = SDL_EVENT_MOUSE_MOTION;
        pushInjectedPointerEvent(move);
        SDL_Event native = move;
        native.common.timestamp = move.common.timestamp + 1;
        EXPECT_FALSE(prepareInjectedPointerEvent(native));
        ASSERT_TRUE(prepareInjectedPointerEvent(move));
        finishInjectedPointerFrame();
        EXPECT_FALSE(injectedPointer());
        ASSERT_TRUE(lastInjectedPointer());
        EXPECT_EQ(lastInjectedPointer()->x, 415);
        EXPECT_EQ(lastInjectedPointer()->y, 628);
        injectPointerButton(SDL_BUTTON_LEFT, true);
        SDL_Event down{};
        down.type = SDL_EVENT_MOUSE_BUTTON_DOWN;
        pushInjectedPointerEvent(down);
        ASSERT_TRUE(prepareInjectedPointerEvent(down));
        finishInjectedPointerFrame();
        ASSERT_TRUE(injectedPointer());
        EXPECT_EQ(injectedPointer()->x, 415);
        EXPECT_FALSE(prepareInjectedPointerEvent(native));
        injectPointerButton(SDL_BUTTON_LEFT, false);
        SDL_Event up{};
        up.type = SDL_EVENT_MOUSE_BUTTON_UP;
        pushInjectedPointerEvent(up);
        ASSERT_TRUE(prepareInjectedPointerEvent(up));
        finishInjectedPointerFrame();
        EXPECT_FALSE(injectedPointer());
        EXPECT_TRUE(prepareInjectedPointerEvent(native));
        SDL_QuitSubSystem(SDL_INIT_EVENTS);
    }

    TEST(McpInjectedPointer, HoverSurvivesFramesUntilDeadlineOrNextGesture) {
        using namespace lfs::vis::input;
        ASSERT_TRUE(SDL_InitSubSystem(SDL_INIT_EVENTS));
        const auto move = []() {
            injectPointerMove(250, 300);
            SDL_Event event{};
            event.type = SDL_EVENT_MOUSE_MOTION;
            pushInjectedPointerEvent(event);
            EXPECT_TRUE(prepareInjectedPointerEvent(event));
        };
        move();
        retainInjectedPointerFor(std::chrono::seconds(1));
        for (int frame = 0; frame < 5; ++frame) {
            finishInjectedPointerFrame();
            ASSERT_TRUE(injectedPointer());
            EXPECT_EQ(injectedPointer()->buttons, 0u);
        }
        // An expired lease restores real input, even without another event.
        retainInjectedPointerFor(std::chrono::milliseconds(0));
        finishInjectedPointerFrame();
        EXPECT_FALSE(injectedPointer());
        move();
        retainInjectedPointerFor(std::chrono::seconds(1));
        move();
        finishInjectedPointerFrame();
        EXPECT_FALSE(injectedPointer());
        SDL_QuitSubSystem(SDL_INIT_EVENTS);
    }

    constexpr std::array<const char*, 14> kScreenToolNames = {
        "screen.get",
        "screen.split",
        "screen.join",
        "screen.close",
        "screen.swap",
        "screen.set_editor",
        "screen.maximize",
        "screen.reset",
        "view.command",
        "view.get_camera",
        "view.set_camera",
        "view.get_settings",
        "view.set_settings",
        "ui.pointer",
    };

    class McpScreenToolsTest : public ::testing::Test {
    protected:
        void SetUp() override {
            unregister_tools();
            lfs::vis::ViewerOptions options;
            options.show_startup_overlay = false;
            viewer_ = std::make_unique<lfs::vis::VisualizerImpl>(options);
            lfs::app::register_gui_screen_tools(lfs::mcp::ToolRegistry::instance(), viewer_.get());
            lfs::app::register_gui_screen_resources(lfs::mcp::ResourceRegistry::instance(), viewer_.get());
        }

        void TearDown() override {
            unregister_tools();
            lfs::mcp::ResourceRegistry::instance().unregister_resource_prefix("lichtfeld://ui/screen");
            viewer_.reset();
        }

        static void unregister_tools() {
            for (const auto* name : kScreenToolNames)
                lfs::mcp::ToolRegistry::instance().unregister_tool(name);
        }

        json call(const std::string& name, const json& args = json::object()) {
            return lfs::mcp::ToolRegistry::instance().call_tool(name, args);
        }

        std::unique_ptr<lfs::vis::VisualizerImpl> viewer_;
    };

    TEST_F(McpScreenToolsTest, ScreenGetListsDefaultAreas) {
        const auto result = call("screen.get");
        ASSERT_TRUE(result.value("success", false)) << result.dump();
        ASSERT_TRUE(result["areas"].is_array());
        ASSERT_GE(result["areas"].size(), 3u);
        bool has_view = false;
        for (const auto& area : result["areas"]) {
            if (area.value("is_view", false))
                has_view = true;
        }
        EXPECT_TRUE(has_view);
        EXPECT_GT(result.value("active_view", 0), 0);
    }

    TEST_F(McpScreenToolsTest, SplitAndResetRoundTrip) {
        const auto before = call("screen.get");
        const int view = before["active_view"].get<int>();
        const auto split = call("screen.split", json{{"area", view}, {"direction", "vertical"}, {"factor", 0.4}});
        ASSERT_TRUE(split.value("success", false)) << split.dump();
        EXPECT_GT(split["areas"].size(), before["areas"].size());
        EXPECT_GT(split.value("area", 0), 0);

        const auto reset = call("screen.reset");
        ASSERT_TRUE(reset.value("success", false)) << reset.dump();
        EXPECT_EQ(reset["areas"].size(), before["areas"].size());
    }

    TEST_F(McpScreenToolsTest, ViewSettingsRejectUnknownFields) {
        const auto get = call("screen.get");
        const int view = get["active_view"].get<int>();
        const auto result = call("view.set_settings",
                                 json{{"view", view}, {"settings", {{"not_a_field", true}}}});
        EXPECT_TRUE(result.contains("error")) << result.dump();
    }

    TEST_F(McpScreenToolsTest, ViewCameraAndSettingsRoundTrip) {
        const auto get = call("screen.get");
        const int view = get["active_view"].get<int>();
        const auto set_cam = call("view.set_camera",
                                  json{{"view", view},
                                       {"eye", json::array({4.0, 3.0, 4.0})},
                                       {"target", json::array({0.0, 0.0, 0.0})}});
        ASSERT_TRUE(set_cam.value("success", false)) << set_cam.dump();
        const auto cam = call("view.get_camera", json{{"view", view}});
        ASSERT_TRUE(cam.value("success", false)) << cam.dump();
        EXPECT_NEAR(cam["camera"]["eye"][0].get<double>(), 4.0, 1e-4);

        const auto set_settings =
            call("view.set_settings", json{{"view", view}, {"settings", {{"show_grid", false}}}});
        ASSERT_TRUE(set_settings.value("success", false)) << set_settings.dump();
        const auto settings = call("view.get_settings", json{{"view", view}});
        EXPECT_FALSE(settings["settings"].value("show_grid", true));
    }

    TEST_F(McpScreenToolsTest, ToolsAdvertiseGuiThreadMetadata) {
        const auto tools = lfs::mcp::ToolRegistry::instance().list_tools();
        for (const auto* name : kScreenToolNames) {
            const auto it = std::find_if(tools.begin(), tools.end(),
                                         [&](const auto& tool) { return tool.name == name; });
            ASSERT_NE(it, tools.end()) << name;
            EXPECT_EQ(it->metadata.runtime, "gui");
            EXPECT_EQ(it->metadata.thread_affinity, "gui_thread");
            const auto listed = lfs::mcp::tool_to_json(*it);
            EXPECT_TRUE(listed.contains("annotations"));
            EXPECT_TRUE(listed.contains("_meta"));
        }
    }

    TEST_F(McpScreenToolsTest, ScreenResourceMatchesGet) {
        const auto tool = call("screen.get");
        const auto resource = lfs::mcp::ResourceRegistry::instance().read_resource("lichtfeld://ui/screen");
        ASSERT_TRUE(resource) << resource.error();
        ASSERT_FALSE(resource->empty());
        const auto body = json::parse(std::get<std::string>(resource->front().content));
        EXPECT_EQ(body["areas"].size(), tool["areas"].size());
        EXPECT_EQ(body["active_view"], tool["active_view"]);
    }

} // namespace
