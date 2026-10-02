/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "input/input_controller.hpp"
#include "licht_test_support.hpp"
#include "project/session_state.hpp"
#include "screen/screen.hpp"
#include "screen/view3d_space.hpp"
#include "visualizer_impl.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

namespace lfs::vis {
    class P5SessionCaptureTestAccess {
    public:
        static void initializeInputController(VisualizerImpl& viewer) {
            viewer.input_controller_ = std::make_unique<InputController>(nullptr, viewer);
        }
    };
}

namespace {
    using Json = lfs::io::JsonChapterDom::Json;
    using namespace lfs::io::project;
    using namespace lfs::vis::project;
    using lfs::test::licht::json_root;
    using lfs::test::licht::require_result;

    Json find_space_payload_for_test(const Json& root, const std::string_view type) {
        for (const auto& layout : root.at("layouts"))
            for (const auto& area : layout.at("areas"))
                for (const auto& space : area.at("spaces"))
                    if (space.value("type", std::string{}) == type)
                        return space.at("opaque_payload");
        return Json::object();
    }

    TEST(GuiScreenSessionTest, RecreatedViewCanBeSavedAndRestoredWithoutRetainingClosedAreas) {
        lfs::vis::ViewerOptions options;
        options.show_startup_overlay = false;
        lfs::vis::VisualizerImpl viewer(options);
        lfs::vis::P5SessionCaptureTestAccess::initializeInputController(viewer);

        auto& screen = viewer.screens().screen();
        const auto left = screen.activeView();
        const auto right = screen.split(left, lfs::vis::screen::SplitAxis::Columns, 0.5f);
        const auto removed = screen.split(left, lfs::vis::screen::SplitAxis::Rows, 0.5f);
        ASSERT_TRUE(screen.split(right, lfs::vis::screen::SplitAxis::Rows, 0.5f).valid());
        ASSERT_TRUE(screen.setActiveView(removed));
        ASSERT_TRUE(screen.toggleMaximized(removed));
        auto retained = require_result(captureGuiSession(viewer, ProjectSessionChapters{}, {}));

        Json gui = json_root(retained.gui_layout.dom());
        for (auto& layout : gui["layouts"]) {
            for (auto& area : layout["areas"]) {
                for (auto& space : area["spaces"]) {
                    if (space.value("type", std::string{}) != "screen")
                        continue;
                    auto& payload = space["opaque_payload"];
                    payload["vendor_extension"] = "retain-screen";
                    for (auto& saved_area : payload["areas"]) {
                        if (saved_area["id"] == left.value)
                            saved_area["vendor_extension"] = "retain-live-area";
                    }
                }
            }
        }
        Json inactive_layout = gui["layouts"][0];
        inactive_layout["active"] = false;
        gui["layouts"].push_back(inactive_layout);
        retained.gui_layout = require_result(GuiLayoutChapter::parse(gui.dump()));
        ASSERT_TRUE(screen.toggleMaximized(removed));
        ASSERT_TRUE(screen.join(left, removed));
        const auto added = screen.split(left, lfs::vis::screen::SplitAxis::Rows, 0.5f);
        ASSERT_TRUE(added.valid());
        ASSERT_NE(added, removed);
        ASSERT_TRUE(screen.setActiveView(added));
        screen.view(added)->camera.setViewMatrix(glm::mat3(1.0f), glm::vec3(9.0f, 8.0f, 7.0f));
        screen.view(added)->settings.focal_length_mm = 85.0f;

        auto captured = require_result(captureGuiSession(viewer, retained, {}));
        const auto captured_gui = json_root(captured.gui_layout.dom());
        EXPECT_EQ(captured_gui["layouts"][1], inactive_layout);
        const auto payload = find_space_payload_for_test(captured_gui, "screen");
        EXPECT_EQ(payload["vendor_extension"], "retain-screen");
        EXPECT_FALSE(payload.contains("maximized"));
        ASSERT_EQ(payload["areas"].size(), screen.areas().size());
        for (const auto& area : payload["areas"]) {
            EXPECT_NE(area["id"], removed.value);
            if (area["id"] == left.value)
                EXPECT_EQ(area["vendor_extension"], "retain-live-area");
        }
        auto prepared = prepareGuiSessionRestore(std::move(captured));
        ASSERT_TRUE(prepared);
        viewer.screens().resetToDefault();
        std::vector<CameraBookmarkProjectState> bookmarks;
        applyGuiSession(viewer, *prepared, bookmarks);
        EXPECT_EQ(viewer.screens().screen().views().size(), 4u);
        EXPECT_EQ(viewer.screens().screen().activeView(), added);
        EXPECT_EQ(viewer.screens().screen().maximized(), lfs::vis::screen::AreaId{});
        ASSERT_NE(viewer.screens().view3D(added), nullptr);
        EXPECT_EQ(viewer.screens().view3D(added)->camera.camera.t, glm::vec3(9.0f, 8.0f, 7.0f));
        EXPECT_FLOAT_EQ(viewer.screens().view3D(added)->settings.focal_length_mm, 85.0f);
    }

}
