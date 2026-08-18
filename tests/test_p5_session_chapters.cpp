/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "core/camera.hpp"
#include "core/event_bridge/event_bridge.hpp"
#include "core/event_bridge/scoped_handler.hpp"
#include "core/events.hpp"
#include "core/parameters.hpp"
#include "core/scene.hpp"
#include "core/uuid.hpp"
#include "gui/editor/python_editor.hpp"
#include "gui/gui_manager.hpp"
#include "gui/panels/python_console_panel.hpp"
#include "gui/scene_tree_session.hpp"
#include "gui/sequencer_ui_state.hpp"
#include "io/project_document.hpp"
#include "io/video/video_export_options.hpp"
#include "licht_matrix_test_data.hpp"
#include "licht_test_support.hpp"
#include "project/session_state.hpp"
#include "rendering/rendering_types.hpp"
#include "sequencer/timeline.hpp"
#include "tools/unified_tool_registry.hpp"
#include "training/control/command_api.hpp"
#include "training/project_snapshot_chapters.hpp"
#include "training/trainer.hpp"
#include "training/training_manager.hpp"
#include "training/training_state.hpp"
#include "visualizer_impl.hpp"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <limits>
#include <optional>
#include <ranges>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <glm/gtc/constants.hpp>
#include <glm/gtc/quaternion.hpp>

namespace {

    namespace fs = std::filesystem;
    using Json = lfs::io::JsonChapterDom::Json;
    using namespace lfs::io::project;
    using lfs::test::licht::json_root;
    using lfs::test::licht::make_populated_session_chapters;
    using lfs::test::licht::require_result;
    using lfs::test::licht::require_status;
    using lfs::test::licht::rolled_panel_camera;
    using lfs::test::licht::TemporaryDirectory;
    using namespace lfs::vis::project;

    void overwrite_f32_le(
        std::vector<std::byte>& bytes,
        const std::size_t offset,
        const float value) {
        lfs::test::licht::write_u32_le(
            bytes, offset,
            std::bit_cast<std::uint32_t>(value));
    }

    void overwrite_f64_le(
        std::vector<std::byte>& bytes,
        const std::size_t offset,
        const double value) {
        lfs::test::licht::write_u64_le(
            bytes, offset,
            std::bit_cast<std::uint64_t>(value));
    }

    PreparedGuiSessionRestore
    make_select_rectangle_restore() {
        auto session = make_populated_session_chapters();
        auto view = json_root(session.view.dom());
        view["tools"]["active_tool_id"] = "builtin.select";
        view["tools"]["selection_submode"] = "rectangle";
        session.view = require_result(
            ViewSessionChapter::parse(view.dump()));
        return require_result(
            prepareGuiSessionRestore(std::move(session)));
    }

    lfs::core::NodeId seed_splat_selection(
        lfs::vis::VisualizerImpl& viewer) {
        const auto node =
            viewer.getScene().addSplatPlaceholder(
                "Restore splat");
        if (node == lfs::core::NULL_NODE)
            return node;
        viewer.getSceneManager()->changeContentType(
            lfs::vis::SceneManager::ContentType::
                SplatFiles);
        viewer.getSceneManager()->selectNode(node);
        viewer.getEditorContext().update(
            viewer.getSceneManager(),
            viewer.getTrainerManager());
        return node;
    }

    TEST(P5SessionChapterTest,
         GuilUsesFrozenAreaTreeAndStripsExcludedStateOnLoad) {
        GuiLayoutChapter chapter;
        const Json root = json_root(chapter.dom());
        ASSERT_EQ(root["layouts"].size(), 1u);
        const auto& layout = root["layouts"][0];
        ASSERT_TRUE(layout["active"].get<bool>());
        ASSERT_EQ(layout["areas"].size(), 1u);
        const auto& area = layout["areas"][0];
        EXPECT_TRUE(
            area.contains(
                "rect_or_split_position"));
        EXPECT_TRUE(area.contains("active_space"));
        ASSERT_EQ(area["spaces"].size(), 3u);
        for (const auto& space : area["spaces"]) {
            EXPECT_TRUE(space.contains("type"));
            EXPECT_TRUE(space.contains("version"));
            EXPECT_TRUE(
                space.contains("opaque_payload"));
        }

        Json imgui = root;
        imgui["layouts"][0]["areas"][0]
             ["spaces"][0]["opaque_payload"]
             ["imgui_ini"] = "dock-id";
        auto rejected_imgui =
            GuiLayoutChapter::parse(imgui.dump());
        ASSERT_FALSE(rejected_imgui);
        EXPECT_EQ(
            rejected_imgui.error().code(),
            lfs::ErrorCode::DataLoss);

        for (const auto key : {
                 "theme",
                 "language",
                 "scale",
                 "ui_scale",
                 "hud",
                 "vram_hud",
                 "vram_hud_visible",
             }) {
            Json user_global = root;
            user_global["layouts"][0]["areas"][0]
                       ["spaces"][0]
                       ["opaque_payload"][key] =
                           "forbidden";
            auto loaded =
                GuiLayoutChapter::parse(
                    user_global.dump());
            ASSERT_TRUE(loaded)
                << key;
            const auto sanitized =
                json_root(loaded->dom());
            EXPECT_FALSE(
                sanitized["layouts"][0]["areas"][0]
                         ["spaces"][0]
                         ["opaque_payload"]
                             .contains(key))
                << key;
        }

        for (const auto key : {"dpi", "dpi_scale"}) {
            Json invalid = root;
            invalid["layouts"][0]["areas"][0]["spaces"][0]["opaque_payload"][key] = 2.0;
            const auto loaded = GuiLayoutChapter::parse(invalid.dump());
            ASSERT_FALSE(loaded) << key;
            EXPECT_EQ(loaded.error().code(), lfs::ErrorCode::DataLoss) << key;
        }

        Json plugin_fields = root;
        auto& registry_payload =
            plugin_fields["layouts"][0]["areas"][0]
                         ["spaces"][1]["opaque_payload"];
        registry_payload["panel_payloads"]["plugin.example"] = {
            {"scale", 2.0},
            {"theme", "plugin-local"},
            {"nested", {{"dpi", 144}}},
        };
        auto loaded_plugin_fields =
            GuiLayoutChapter::parse(plugin_fields.dump());
        ASSERT_TRUE(loaded_plugin_fields);
        const auto preserved_plugin_payload =
            json_root(loaded_plugin_fields->dom())
                ["layouts"][0]["areas"][0]
                ["spaces"][1]["opaque_payload"]
                ["panel_payloads"]["plugin.example"];
        EXPECT_EQ(preserved_plugin_payload["scale"], 2.0);
        EXPECT_EQ(preserved_plugin_payload["theme"], "plugin-local");
        EXPECT_EQ(preserved_plugin_payload["nested"]["dpi"], 144);

        Json legacy_window = root;
        legacy_window["layouts"][0]["areas"][0]
                     ["spaces"][0]["opaque_payload"]
                     ["window"] = {
                         {"x", 101},
                         {"y", 202},
                         {"width", 1440},
                         {"height", 900},
                         {"maximized", true},
                     };
        auto loaded_legacy_window =
            GuiLayoutChapter::parse(
                legacy_window.dump());
        ASSERT_TRUE(loaded_legacy_window)
            << lfs::format_for_developer(
                   loaded_legacy_window.error());
        const auto sanitized_legacy_window =
            json_root(loaded_legacy_window->dom());
        EXPECT_FALSE(
            sanitized_legacy_window["layouts"][0]["areas"][0]
                                   ["spaces"][0]["opaque_payload"]
                                       .contains("window"));
    }

    TEST(P5SessionChapterTest,
         DefaultViewToolIdsAreRegisteredDefaults) {
        const auto root = json_root(
            default_session_chapter_dom(
                SessionJsonChapterKind::View));
        EXPECT_EQ(
            root["tools"]["active_tool_id"],
            "builtin.select");
        EXPECT_EQ(
            root["tools"]["active_submode_id"],
            "centers");
        EXPECT_EQ(
            root["tools"]["selection_submode"],
            "centers");
        EXPECT_EQ(
            root["tools"]["crop_operation"],
            "translate");
    }

    TEST(P5SessionChapterTest,
         GuilSaveRefusesExcludedUserGlobalState) {
        TemporaryDirectory temporary{"licht-p5"};
        auto document = ProjectDocument::create(
            lfs::core::generate_uuid_v4(), 100);
        ASSERT_TRUE(document)
            << lfs::format_for_developer(
                   document.error());
        require_status(
            document->edit_gui_layout()
                .dom()
                .set("theme", "must-not-save"));

        auto saved = document->save(
            temporary.path /
                "excluded-user-global.licht",
            ProjectDocumentSaveOptions{
                .disk_reserve_bytes = 0,
            });
        ASSERT_FALSE(saved);
        EXPECT_EQ(
            saved.error().code(),
            lfs::ErrorCode::DataLoss);
        EXPECT_FALSE(fs::exists(
            temporary.path /
            "excluded-user-global.licht"));
    }

    TEST(P5SessionChapterTest,
         GuilRetainsUnknownSpaceTypeRoundTrip) {
        GuiLayoutChapter chapter;
        Json root = json_root(chapter.dom());
        const Json unknown{
            {"type", "vendor.future_graph"},
            {"version", 7},
            {"opaque_payload",
             {
                 {"opaque_token", "retain-me"},
                 {"nodes",
                  Json::array({
                      {
                          {"id", 41},
                          {"weight", 0.75f},
                      },
                  })},
             }},
        };
        root["layouts"][0]["areas"][0]["spaces"]
            .push_back(unknown);

        auto loaded =
            GuiLayoutChapter::parse(root.dump());
        ASSERT_TRUE(loaded)
            << lfs::format_for_developer(
                   loaded.error());
        const Json known_update{
            {"layouts",
             Json::array({
                 {
                     {"areas",
                      Json::array({
                          {
                              {"spaces",
                               Json::array({
                                   {
                                       {"type",
                                        "fixed_arrangement"},
                                       {"opaque_payload",
                                        {
                                            {"right_panel_width",
                                             492.0f},
                                        }},
                                   },
                               })},
                          },
                      })},
                 },
             })},
        };
        require_status(
            loaded->merge_known_state(
                known_update));

        auto reopened =
            GuiLayoutChapter::from_bytes(
                loaded->to_bytes());
        ASSERT_TRUE(reopened)
            << lfs::format_for_developer(
                   reopened.error());
        const auto retained =
            json_root(reopened->dom());
        const auto& spaces =
            retained["layouts"][0]["areas"][0]
                    ["spaces"];
        const auto found =
            std::ranges::find_if(
                spaces,
                [](const Json& space) {
                    return space.value(
                               "type",
                               std::string{}) ==
                           "vendor.future_graph";
                });
        ASSERT_NE(found, spaces.end());
        EXPECT_EQ(*found, unknown);

        ProjectSessionChapters session =
            make_populated_session_chapters();
        session.gui_layout =
            std::move(*reopened);
        EXPECT_TRUE(
            prepareGuiSessionRestore(
                std::move(session)));
    }

    TEST(P5SessionChapterTest,
         GuilAreaTreeRejectsCyclesMissingParentsTwoRootsAndIncompleteOpaqueSpaces) {
        const auto area_with_identity =
            [](Json area, std::string id,
               std::optional<std::string> parent) {
                area["id"] = std::move(id);
                area["parent_id"] = parent
                                        ? Json(*parent)
                                        : Json(nullptr);
                return area;
            };
        const auto expect_data_loss = [](const Json& candidate) {
            auto parsed = GuiLayoutChapter::parse(candidate.dump());
            ASSERT_FALSE(parsed);
            EXPECT_EQ(parsed.error().code(),
                      lfs::ErrorCode::DataLoss);
        };

        GuiLayoutChapter chapter;
        const Json baseline = json_root(chapter.dom());
        const Json implicit_area = baseline["layouts"][0]["areas"][0];

        Json cycle = baseline;
        cycle["layouts"][0]["areas"] = Json::array({
            area_with_identity(implicit_area, "root", std::nullopt),
            area_with_identity(implicit_area, "loop-a", "loop-b"),
            area_with_identity(implicit_area, "loop-b", "loop-a"),
        });
        expect_data_loss(cycle);

        Json missing_parent = baseline;
        missing_parent["layouts"][0]["areas"] = Json::array({
            area_with_identity(implicit_area, "root", std::nullopt),
            area_with_identity(implicit_area, "child", "absent"),
        });
        expect_data_loss(missing_parent);

        Json two_roots = baseline;
        two_roots["layouts"][0]["areas"] = Json::array({
            area_with_identity(implicit_area, "root-a", std::nullopt),
            area_with_identity(implicit_area, "root-b", std::nullopt),
        });
        expect_data_loss(two_roots);

        Json missing_opaque = baseline;
        missing_opaque["layouts"][0]["areas"][0]["spaces"].push_back(
            Json{{"type", "vendor.missing_payload"}, {"version", 1}});
        expect_data_loss(missing_opaque);
    }

    TEST(P5SessionChapterTest,
         MissingViewCameraIsDocumentedDegradedStateButSequencerCameraRefusesHydration) {
        const auto missing = lfs::core::generate_uuid_v4();

        auto view_document = ProjectDocument::create(
            lfs::core::generate_uuid_v4(), 100);
        ASSERT_TRUE(view_document);
        require_status(view_document->edit_view().dom().set(
            "active_camera_uuid", missing.to_string()));
        lfs::core::Scene view_scene;
        auto view_plan = view_document->stage_hydration(view_scene);
        ASSERT_TRUE(view_plan)
            << lfs::format_for_developer(view_plan.error());
        EXPECT_NE(std::ranges::find(
                      view_document->degraded_states(),
                      ProjectDocumentDegradedState::MissingActiveCamera),
                  view_document->degraded_states().end());

        auto sequence_document = ProjectDocument::create(
            lfs::core::generate_uuid_v4(), 100);
        ASSERT_TRUE(sequence_document);
        require_status(sequence_document->edit_sequencer().dom().set_json(
            "timeline.keyframes",
            Json::array({Json{{"camera_uuid", missing.to_string()}}})));
        lfs::core::Scene sequence_scene;
        auto sequence_plan =
            sequence_document->stage_hydration(sequence_scene);
        ASSERT_FALSE(sequence_plan);
        EXPECT_EQ(sequence_plan.error().code(),
                  lfs::ErrorCode::DataLoss);
    }

    TEST(P5SessionChapterTest,
         EditorWorkspaceCaptureUsesEveryRealOpenBuffer) {
        using namespace lfs::vis::editor;

        PythonEditor editor;
        PythonEditorWorkspaceSessionState
            restored;
        restored.open_files = {
            {
                .locator =
                    "/tmp/p5-editor-a.py",
                .text =
                    "alpha = 1\nbeta = 2\n",
                .modified = false,
                .editor =
                    {
                        .cursor_byte = 8,
                        .selection_anchor_byte =
                            2,
                        .scroll_x = 11.0f,
                        .scroll_y = 23.0f,
                        .folds =
                            {
                                {
                                    .start_byte =
                                        10,
                                    .end_byte =
                                        18,
                                    .start_line =
                                        1,
                                    .end_line = 2,
                                    .kind =
                                        "block",
                                    .collapsed =
                                        true,
                                },
                            },
                    },
            },
            {
                .locator =
                    "untitled://p5-editor-b",
                .text =
                    "secret = 'embedded'\n",
                .modified = true,
                .editor =
                    {
                        .cursor_byte = 18,
                        .selection_anchor_byte =
                            9,
                        .scroll_x = 3.5f,
                        .scroll_y = 71.0f,
                        .folds = {},
                    },
            },
        };
        restored.active_file =
            "untitled://p5-editor-b";
        restored.vim_mode = true;

        editor.restoreWorkspaceSessionState(
            restored);
        const auto captured =
            editor.captureWorkspaceSessionState(
                "/tmp/stale-console-path.py",
                false);

        ASSERT_EQ(
            captured.open_files.size(), 2u);
        EXPECT_EQ(
            captured.active_file,
            "untitled://p5-editor-b");
        EXPECT_TRUE(captured.vim_mode);
        for (std::size_t index = 0;
             index < restored.open_files.size();
             ++index) {
            const auto& expected =
                restored.open_files[index];
            const auto& actual =
                captured.open_files[index];
            EXPECT_EQ(
                actual.locator,
                expected.locator)
                << index;
            EXPECT_EQ(actual.text, expected.text)
                << index;
            EXPECT_EQ(
                actual.modified,
                expected.modified)
                << index;
            EXPECT_EQ(
                actual.editor.cursor_byte,
                expected.editor.cursor_byte)
                << index;
            EXPECT_EQ(
                actual.editor
                    .selection_anchor_byte,
                expected.editor
                    .selection_anchor_byte)
                << index;
            EXPECT_FLOAT_EQ(
                actual.editor.scroll_x,
                expected.editor.scroll_x)
                << index;
            EXPECT_FLOAT_EQ(
                actual.editor.scroll_y,
                expected.editor.scroll_y)
                << index;
            EXPECT_EQ(
                actual.editor.folds,
                expected.editor.folds)
                << index;
        }
    }

    TEST(P5SessionChapterTest,
         EditorDefaultBufferDropsOnlyEmptyStateAndKeepsTypedCode) {
        using namespace lfs::vis::editor;
        PythonEditor editor;
        editor.restoreWorkspaceSessionState({});
        EXPECT_TRUE(
            editor.captureWorkspaceSessionState(
                      "untitled://empty", false)
                .open_files.empty());

        editor.setText("typed = 1\n");
        editor.setActiveSessionLocator(
            "untitled://typed");
        const auto captured =
            editor.captureWorkspaceSessionState(
                "/tmp/stale-locator.py", true);
        ASSERT_EQ(captured.open_files.size(), 1u);
        EXPECT_EQ(
            captured.open_files.front().locator,
            "untitled://typed");
        EXPECT_EQ(
            captured.open_files.front().text,
            "typed = 1\n");
        EXPECT_TRUE(
            captured.open_files.front().modified);
    }

    TEST(P5SessionChapterTest,
         RetainedDomPreservesUnknownFieldsOnKnownPanel) {
        auto session = make_populated_session_chapters();
        const Json known{
            {"version", 1},
            {"layouts",
             Json::array({
                 {
                     {"areas",
                      Json::array({
                          {
                              {"rect_or_split_position",
                               {
                                   {"kind", "rect"},
                               }},
                              {"active_space",
                               "viewport"},
                              {"spaces",
                               Json::array({
                                   {
                                       {"type",
                                        "panel_registry"},
                                       {"version", 1},
                                       {"opaque_payload",
                                        {
                                            {"panels",
                                             Json::array({
                                                 {
                                                     {"id",
                                                      "plugin.matrix"},
                                                     {"enabled",
                                                      false},
                                                 },
                                             })},
                                        }},
                                   },
                               })},
                          },
                      })},
                     {"active", true},
                 },
             })},
        };
        require_status(
            session.gui_layout
                .merge_known_state(known));
        const auto merged = json_root(
            session.gui_layout.dom());
        const auto& panels =
            merged["layouts"][0]["areas"][0]
                  ["spaces"][1]["opaque_payload"]
                  ["panels"];
        ASSERT_EQ(panels.size(), 1u);
        EXPECT_FALSE(
            panels[0]["enabled"].get<bool>());
        EXPECT_EQ(
            panels[0]["vendor_extension"],
            "retained");
    }

    TEST(P5SessionChapterTest,
         MissingViewOrSequencerReferenceFailsDocumentValidation) {
        const auto session = make_populated_session_chapters();
        TemporaryDirectory temporary{"licht-p5"};

        auto missing_environment =
            ProjectDocument::create(
                lfs::core::generate_uuid_v4(),
                100);
        ASSERT_TRUE(missing_environment);
        missing_environment->edit_view() =
            session.view;
        auto environment_save =
            missing_environment->save(
                temporary.path /
                    "missing-environment.licht",
                ProjectDocumentSaveOptions{
                    .disk_reserve_bytes = 0,
                });
        ASSERT_FALSE(environment_save);
        EXPECT_EQ(
            environment_save.error().code(),
            lfs::ErrorCode::DataLoss);

        auto missing_sequence =
            ProjectDocument::create(
                lfs::core::generate_uuid_v4(),
                100);
        ASSERT_TRUE(missing_sequence);
        missing_sequence->edit_sequencer() =
            session.sequencer;
        auto sequence_save =
            missing_sequence->save(
                temporary.path /
                    "missing-sequence.licht",
                ProjectDocumentSaveOptions{
                    .disk_reserve_bytes = 0,
                });
        ASSERT_FALSE(sequence_save);
        EXPECT_EQ(
            sequence_save.error().code(),
            lfs::ErrorCode::DataLoss);
    }

    TEST(P5SessionChapterTest,
         RenderBackendAndPanelRotationRoundTripCanonically) {
        lfs::vis::RenderSettings settings;
        settings.raster_backend =
            lfs::rendering::
                GaussianRasterBackend::ThreeDgut;
        settings.gut = false;
        const auto json =
            renderSettingsToProjectJson(settings);
        EXPECT_FALSE(json.contains("gut"));
        EXPECT_EQ(
            json["raster_backend"], "3dgut");
        auto restored =
            renderSettingsFromProjectJson(json);
        ASSERT_TRUE(restored)
            << lfs::format_for_developer(
                   restored.error());
        EXPECT_EQ(
            restored->raster_backend,
            lfs::rendering::
                GaussianRasterBackend::ThreeDgut);
        EXPECT_TRUE(restored->gut);

        Viewport viewport;
        const auto expected = rolled_panel_camera(7.0f);
        applyPanelCameraProjectState(
            viewport, expected);
        EXPECT_EQ(
            capturePanelCameraProjectState(
                viewport),
            expected);
    }

    TEST(P5SessionChapterTest,
         RestoreCoordinatorRequiresBothEventGates) {
        auto session = make_populated_session_chapters();
        GuiSessionRestoreCoordinator coordinator;
        require_status(
            coordinator.stage(session));
        EXPECT_TRUE(coordinator.hasPending());
        EXPECT_FALSE(coordinator.ready());

        coordinator.onPanelsReady(73);
        EXPECT_FALSE(coordinator.ready());
        EXPECT_FALSE(coordinator.takeReady());

        coordinator.onFirstGuiFrame();
        ASSERT_TRUE(coordinator.ready());
        auto prepared = coordinator.takeReady();
        ASSERT_TRUE(prepared);
        EXPECT_EQ(
            coordinator
                .panelsRegistrationRevision(),
            73u);
        EXPECT_FALSE(coordinator.hasPending());
        EXPECT_EQ(
            prepared->chapters.editor.dom().dump(),
            session.editor.dom().dump());
    }

    TEST(P5SessionChapterTest,
         RestoreCoordinatorUsesDistinctTicketsForRepeatedOpens) {
        GuiSessionRestoreCoordinator coordinator;
        const auto first = coordinator.stagePrepared(
            prepareGuiSessionRestore(
                make_populated_session_chapters())
                .value());
        const auto second = coordinator.stagePrepared(
            prepareGuiSessionRestore(
                make_populated_session_chapters())
                .value());
        EXPECT_NE(first, second);
        EXPECT_FALSE(coordinator.isCurrent(first));
        EXPECT_TRUE(coordinator.isCurrent(second));
        coordinator.clear();
        EXPECT_FALSE(coordinator.isCurrent(second));
    }

    TEST(P5SessionChapterTest,
         RestoreCoordinatorSecondOpenIsReadyWithoutRegating) {
        GuiSessionRestoreCoordinator coordinator;
        coordinator.onFirstGuiFrame();
        coordinator.onPanelsReady(3);
        const auto first = coordinator.stagePrepared(
            prepareGuiSessionRestore(
                make_populated_session_chapters())
                .value());
        ASSERT_TRUE(coordinator.ready());
        ASSERT_TRUE(coordinator.takeReady());
        EXPECT_FALSE(coordinator.isCurrent(first));

        const auto second = coordinator.stagePrepared(
            prepareGuiSessionRestore(
                make_populated_session_chapters())
                .value());
        EXPECT_NE(first, second);
        EXPECT_TRUE(coordinator.ready());
        EXPECT_TRUE(coordinator.isCurrent(second));
        ASSERT_TRUE(coordinator.takeReady());
        EXPECT_FALSE(coordinator.hasPending());
    }

    TEST(P5SessionChapterTest,
         NativeToolRestoreActivatesSelectionAndSubmode) {
        lfs::event::EventBridge::instance().clear_all();
        lfs::vis::ViewerOptions options;
        options.show_startup_overlay = false;
        lfs::vis::VisualizerImpl viewer(options);
        const auto node =
            viewer.getScene().addSplatPlaceholder("Restore splat");
        ASSERT_NE(node, lfs::core::NULL_NODE);
        viewer.getSceneManager()->changeContentType(
            lfs::vis::SceneManager::ContentType::SplatFiles);
        viewer.getSceneManager()->selectNode(node);
        viewer.getEditorContext().update(
            viewer.getSceneManager(),
            viewer.getTrainerManager());

        auto session = make_populated_session_chapters();
        auto view = json_root(session.view.dom());
        view["tools"]["active_tool_id"] = "builtin.select";
        view["tools"]["selection_submode"] = "rectangle";
        session.view = require_result(
            ViewSessionChapter::parse(view.dump()));
        auto prepared = require_result(
            prepareGuiSessionRestore(std::move(session)));

        viewer.getGuiManager()->panelLayout().setShowSequencer(true);
        applyGuiSessionTools(viewer, prepared);

        EXPECT_EQ(
            viewer.getEditorContext().getActiveTool(),
            lfs::vis::ToolType::Selection);
        EXPECT_EQ(
            lfs::vis::UnifiedToolRegistry::instance()
                .getActiveTool(),
            "builtin.select");
        EXPECT_EQ(
            lfs::vis::UnifiedToolRegistry::instance()
                .getActiveSubmode(),
            "rectangle");
        EXPECT_EQ(
            viewer.getGuiManager()->gizmo().getSelectionSubMode(),
            lfs::vis::SelectionSubMode::Rectangle);
        EXPECT_TRUE(
            viewer.getGuiManager()->panelLayout().isShowSequencer());
    }

    TEST(P5SessionChapterTest,
         NativeToolRestoreWithoutSavedToolDeactivatesPreviousTool) {
        lfs::event::EventBridge::instance().clear_all();
        lfs::vis::ViewerOptions options;
        options.show_startup_overlay = false;
        lfs::vis::VisualizerImpl viewer(options);
        const auto node =
            viewer.getScene().addSplatPlaceholder("Restore splat");
        ASSERT_NE(node, lfs::core::NULL_NODE);
        viewer.getSceneManager()->changeContentType(
            lfs::vis::SceneManager::ContentType::SplatFiles);
        viewer.getSceneManager()->selectNode(node);
        viewer.getEditorContext().update(
            viewer.getSceneManager(),
            viewer.getTrainerManager());

        auto first = make_populated_session_chapters();
        auto first_view = json_root(first.view.dom());
        first_view["tools"]["active_tool_id"] = "builtin.select";
        first.view = require_result(
            ViewSessionChapter::parse(first_view.dump()));
        applyGuiSessionTools(
            viewer,
            require_result(
                prepareGuiSessionRestore(std::move(first))));
        ASSERT_EQ(
            viewer.getEditorContext().getActiveTool(),
            lfs::vis::ToolType::Selection);

        auto second = make_populated_session_chapters();
        auto second_view = json_root(second.view.dom());
        second_view["tools"]["active_tool_id"] = "";
        second.view = require_result(
            ViewSessionChapter::parse(second_view.dump()));
        applyGuiSessionTools(
            viewer,
            require_result(
                prepareGuiSessionRestore(std::move(second))));
        EXPECT_EQ(
            viewer.getEditorContext().getActiveTool(),
            lfs::vis::ToolType::None);
        EXPECT_TRUE(
            lfs::vis::UnifiedToolRegistry::instance()
                .getActiveTool()
                .empty());
    }

    TEST(P5SessionChapterTest,
         ApplyGuiSessionDefersToolActivationUntilToolsApply) {
        lfs::event::EventBridge::instance().clear_all();
        lfs::vis::ViewerOptions options;
        options.show_startup_overlay = false;
        lfs::vis::VisualizerImpl viewer(options);
        const auto node =
            viewer.getScene().addSplatPlaceholder("Restore splat");
        ASSERT_NE(node, lfs::core::NULL_NODE);
        viewer.getSceneManager()->changeContentType(
            lfs::vis::SceneManager::ContentType::SplatFiles);
        viewer.getSceneManager()->selectNode(node);
        viewer.getEditorContext().update(
            viewer.getSceneManager(),
            viewer.getTrainerManager());

        auto session = make_populated_session_chapters();
        auto view = json_root(session.view.dom());
        view["tools"]["active_tool_id"] = "builtin.select";
        view["tools"]["selection_submode"] = "rectangle";
        session.view = require_result(
            ViewSessionChapter::parse(view.dump()));
        auto prepared = require_result(
            prepareGuiSessionRestore(std::move(session)));

        std::vector<CameraBookmarkProjectState> bookmarks;
        viewer.deactivateProjectTools();
        applyGuiSession(viewer, prepared, bookmarks);
        EXPECT_EQ(
            viewer.getEditorContext().getActiveTool(),
            lfs::vis::ToolType::None);

        viewer.getEditorContext().update(
            viewer.getSceneManager(),
            viewer.getTrainerManager());
        applyGuiSessionTools(viewer, prepared);
        EXPECT_EQ(
            viewer.getEditorContext().getActiveTool(),
            lfs::vis::ToolType::Selection);
        EXPECT_EQ(
            viewer.getGuiManager()->gizmo().getSelectionSubMode(),
            lfs::vis::SelectionSubMode::Rectangle);
    }

    TEST(P5SessionChapterTest,
         HydrationTerminalThenApplyGuiSessionActivatesEditorTool) {
        lfs::event::EventBridge::instance().clear_all();
        lfs::vis::ViewerOptions options;
        options.show_startup_overlay = false;
        lfs::vis::VisualizerImpl viewer(options);
        ASSERT_NE(
            seed_splat_selection(viewer),
            lfs::core::NULL_NODE);
        ASSERT_TRUE(
            viewer.getSceneManager()->hasSelectedNode());

        auto prepared = make_select_rectangle_restore();
        const auto ticket =
            viewer.stagePreparedProjectSessionRestore(
                std::move(prepared));
        ASSERT_NE(ticket, 0u);

        lfs::event::ScopedHandler handlers;
        int selection_applies = 0;
        handlers.subscribe<
            lfs::core::events::tools::SetToolbarTool>(
            [&](const auto& event) {
                if (event.tool_mode ==
                    static_cast<int>(
                        lfs::vis::ToolType::Selection)) {
                    ++selection_applies;
                }
            });

        viewer.noteHydrationTerminalForRestoreTicket(
            ticket);
        EXPECT_EQ(
            viewer.getEditorContext().getActiveTool(),
            lfs::vis::ToolType::None);
        EXPECT_EQ(selection_applies, 0);

        viewer.noteGuiSessionRestoreOwnerReady(1);
        EXPECT_EQ(
            viewer.getEditorContext().getActiveTool(),
            lfs::vis::ToolType::Selection);
        EXPECT_EQ(
            lfs::vis::UnifiedToolRegistry::instance()
                .getActiveTool(),
            "builtin.select");
        EXPECT_EQ(selection_applies, 1);
        EXPECT_FALSE(
            viewer.isProjectSessionRestorePending());

        viewer.noteHydrationTerminalForRestoreTicket(
            ticket);
        viewer.noteGuiSessionRestoreOwnerReady(1);
        viewer.tryApplyProjectSessionTools(ticket);
        EXPECT_EQ(selection_applies, 1);
        EXPECT_EQ(
            viewer.getEditorContext().getActiveTool(),
            lfs::vis::ToolType::Selection);
    }

    TEST(P5SessionChapterTest,
         ToolsApplyOnceForBothHydrationAndOwnerReadyOrders) {
        struct OrderCase {
            const char* name;
            bool hydration_first;
        };
        for (const auto& test : {
                 OrderCase{"hydration-terminal-then-apply",
                           true},
                 OrderCase{"apply-then-hydration-terminal",
                           false},
             }) {
            SCOPED_TRACE(test.name);
            lfs::event::EventBridge::instance().clear_all();
            lfs::vis::ViewerOptions options;
            options.show_startup_overlay = false;
            lfs::vis::VisualizerImpl viewer(options);
            ASSERT_NE(
                seed_splat_selection(viewer),
                lfs::core::NULL_NODE);
            ASSERT_TRUE(
                viewer.getSceneManager()
                    ->hasSelectedNode());
            viewer.deactivateProjectTools();
            ASSERT_EQ(
                viewer.getEditorContext().getActiveTool(),
                lfs::vis::ToolType::None);

            auto prepared = make_select_rectangle_restore();
            const auto ticket =
                viewer.stagePreparedProjectSessionRestore(
                    std::move(prepared));
            ASSERT_NE(ticket, 0u);

            lfs::event::ScopedHandler handlers;
            int selection_applies = 0;
            handlers.subscribe<
                lfs::core::events::tools::SetToolbarTool>(
                [&](const auto& event) {
                    if (event.tool_mode ==
                        static_cast<int>(
                            lfs::vis::ToolType::
                                Selection)) {
                        ++selection_applies;
                    }
                });

            if (test.hydration_first) {
                viewer.noteHydrationTerminalForRestoreTicket(
                    ticket);
                EXPECT_EQ(
                    viewer.getEditorContext()
                        .getActiveTool(),
                    lfs::vis::ToolType::None);
                EXPECT_EQ(selection_applies, 0);
                viewer.noteGuiSessionRestoreOwnerReady(1);
            } else {
                viewer.noteGuiSessionRestoreOwnerReady(1);
                EXPECT_EQ(
                    viewer.getEditorContext()
                        .getActiveTool(),
                    lfs::vis::ToolType::None);
                EXPECT_EQ(selection_applies, 0);
                EXPECT_TRUE(
                    viewer
                        .isProjectSessionRestorePending());
                viewer.noteHydrationTerminalForRestoreTicket(
                    ticket);
            }

            EXPECT_EQ(
                viewer.getEditorContext().getActiveTool(),
                lfs::vis::ToolType::Selection);
            EXPECT_EQ(selection_applies, 1);
            EXPECT_FALSE(
                viewer.isProjectSessionRestorePending());

            viewer.noteHydrationTerminalForRestoreTicket(
                ticket);
            viewer.noteGuiSessionRestoreOwnerReady(1);
            viewer.tryApplyProjectSessionTools(ticket);
            EXPECT_EQ(selection_applies, 1);
        }
    }

    TEST(P5SessionChapterTest,
         EmptyToolRestoreKeepsSequencerVisibility) {
        lfs::event::EventBridge::instance().clear_all();
        lfs::vis::ViewerOptions options;
        options.show_startup_overlay = false;
        lfs::vis::VisualizerImpl viewer(options);
        viewer.getGuiManager()->panelLayout().setShowSequencer(true);

        auto session = make_populated_session_chapters();
        auto view = json_root(session.view.dom());
        view["tools"]["active_tool_id"] = "";
        session.view = require_result(
            ViewSessionChapter::parse(view.dump()));
        applyGuiSessionTools(
            viewer,
            require_result(
                prepareGuiSessionRestore(std::move(session))));
        EXPECT_EQ(
            viewer.getEditorContext().getActiveTool(),
            lfs::vis::ToolType::None);
        EXPECT_TRUE(
            viewer.getGuiManager()->panelLayout().isShowSequencer());
    }

    TEST(P5SessionChapterTest,
         SplitViewOffsetSurvivesPlyComparisonRestore) {
        lfs::event::EventBridge::instance().clear_all();
        lfs::vis::ViewerOptions options;
        options.show_startup_overlay = false;
        lfs::vis::VisualizerImpl viewer(options);
        auto* rendering = viewer.getRenderingManager();
        ASSERT_NE(rendering, nullptr);

        auto staged = rendering->getSettings();
        const auto saved_offset = std::size_t{7};
        staged.split_view_mode =
            rendering->getSettings().split_view_mode;
        staged.split_view_offset = saved_offset;
        rendering->updateSettings(staged);
        rendering->restoreSplitViewMode(
            lfs::vis::SplitViewMode::PLYComparison,
            viewer.getViewport());
        EXPECT_EQ(
            rendering->getSettings().split_view_offset,
            0u);

        auto restored = rendering->getSettings();
        restored.split_view_offset = saved_offset;
        rendering->updateSettings(restored);
        EXPECT_EQ(
            rendering->getSettings().split_view_mode,
            lfs::vis::SplitViewMode::PLYComparison);
        EXPECT_EQ(
            rendering->getSettings().split_view_offset,
            saved_offset);
    }

    TEST(P5SessionChapterTest,
         MissingEditorFileDoesNotLeaveCleanPathBuffer) {
        lfs::event::EventBridge::instance().clear_all();
        lfs::vis::ViewerOptions options;
        options.show_startup_overlay = false;
        lfs::vis::VisualizerImpl viewer(options);

        auto session = make_populated_session_chapters();
        Json editor{
            {"version", 2},
            {"open_files",
             Json::array({
                 {
                     {"locator",
                      "/no/such/restore-editor.py"},
                     {"modified", false},
                 },
             })},
            {"active_file",
             "/no/such/restore-editor.py"},
            {"vim_mode", false},
            {"contains_embedded_secrets", false},
        };
        session.editor = require_result(
            EditorSessionChapter::parse(editor.dump()));
        std::vector<CameraBookmarkProjectState> bookmarks;
        applyGuiSession(
            viewer,
            require_result(
                prepareGuiSessionRestore(std::move(session))),
            bookmarks);

        auto& console =
            lfs::vis::gui::panels::PythonConsoleState::
                getInstance();
        auto* python_editor = console.getEditor();
        ASSERT_NE(python_editor, nullptr);
        const auto captured =
            python_editor->captureWorkspaceSessionState(
                "untitled://missing", false);
        EXPECT_TRUE(std::ranges::none_of(
            captured.open_files, [](const auto& file) {
                return file.locator ==
                       "/no/such/restore-editor.py";
            }));
    }

    TEST(P5SessionChapterTest,
         AutoloadOffNotStartedTerminalAppliesStagedRestore) {
        GuiSessionRestoreCoordinator coordinator;
        require_status(
            coordinator.stage(
                make_populated_session_chapters()));
        coordinator.onFirstGuiFrame();

        struct PreloadCase {
            bool autoload;
            std::string_view status;
            bool terminal;
        };
        for (const auto& test : std::to_array<PreloadCase>({
                 {false, "not_started", false},
                 {true, "discovering", false},
                 {true, "loading", false},
                 {true, "completed", true},
                 {true, "cancelled", true},
                 {true, "not_started", true},
             })) {
            SCOPED_TRACE(test.status);
            EXPECT_EQ(pluginPreloadTerminalForGuiPanels(test.autoload, test.status),
                      test.terminal);
        }

        coordinator.onPanelsReady(91);
        auto restored = coordinator.takeReady();
        ASSERT_TRUE(restored);
        EXPECT_EQ(
            restored->chapters.editor.dom().dump(),
            make_populated_session_chapters()
                .editor.dom()
                .dump());
    }

    TEST(P5SessionChapterTest,
         MetricsHostileBinaryIsRejectedBeforeAllocation) {
        constexpr std::size_t LOSS_COUNT_OFFSET = 16;
        constexpr std::size_t ACCUMULATED_OFFSET =
            32;
        constexpr std::size_t LAST_PSNR_OFFSET =
            44;
        constexpr std::size_t LOSS_VALUE_OFFSET =
            56;
        constexpr std::size_t PSNR_VALUE_OFFSET =
            64;

        MetricsChapter metrics;
        metrics.loss_history = {
            {.iteration = 3, .value = 0.5f},
        };
        metrics.psnr_history = {
            {.iteration = 3, .value = 20.0f},
        };
        metrics.accumulated_training_seconds =
            1.5;
        metrics.last_evaluation = {
            .iteration = 3,
            .psnr = 20.0f,
            .ssim = 0.9f,
        };
        const auto valid =
            require_result(metrics.to_bytes());
        ASSERT_EQ(valid.size(), 68u);

        MetricsChapter finished = metrics;
        finished.finish_reason =
            TrainingFinishReason::Completed;
        const auto finished_bytes =
            require_result(finished.to_bytes());
        ASSERT_EQ(finished_bytes.size(), 72u);
        const auto parsed_finished =
            require_result(
                MetricsChapter::from_bytes(
                    finished_bytes));
        EXPECT_EQ(
            parsed_finished.finish_reason,
            TrainingFinishReason::Completed);
        const auto parsed_legacy =
            require_result(
                MetricsChapter::from_bytes(valid));
        EXPECT_EQ(
            parsed_legacy.finish_reason,
            TrainingFinishReason::None);

        struct InvalidMetricsCase {
            std::string_view name;
            lfs::ErrorCode code;
            std::function<void(std::vector<std::byte>&)> mutate;
        };
        const auto cases = std::to_array<InvalidMetricsCase>({
            {"truncated header", lfs::ErrorCode::DataLoss,
             [](auto& bytes) { bytes.resize(31); }},
            {"truncated payload", lfs::ErrorCode::DataLoss,
             [](auto& bytes) { bytes.pop_back(); }},
            {"oversized history", lfs::ErrorCode::ResourceExhausted,
             [](auto& bytes) {
                 lfs::test::licht::write_u64_le(bytes, LOSS_COUNT_OFFSET,
                                                MetricsChapter::MAX_HISTORY_SAMPLES + 1ull);
             }},
            {"NaN accumulated time", lfs::ErrorCode::DataLoss,
             [](auto& bytes) {
                 overwrite_f64_le(bytes, ACCUMULATED_OFFSET,
                                  std::numeric_limits<double>::quiet_NaN());
             }},
            {"infinite last PSNR", lfs::ErrorCode::DataLoss,
             [](auto& bytes) {
                 overwrite_f32_le(bytes, LAST_PSNR_OFFSET,
                                  std::numeric_limits<float>::infinity());
             }},
            {"NaN loss", lfs::ErrorCode::DataLoss,
             [](auto& bytes) {
                 overwrite_f32_le(bytes, LOSS_VALUE_OFFSET,
                                  std::numeric_limits<float>::quiet_NaN());
             }},
            {"infinite PSNR", lfs::ErrorCode::DataLoss,
             [](auto& bytes) {
                 overwrite_f32_le(bytes, PSNR_VALUE_OFFSET,
                                  -std::numeric_limits<float>::infinity());
             }},
            {"misaligned tail", lfs::ErrorCode::DataLoss,
             [](auto& bytes) { bytes.push_back(std::byte{0x7f}); }},
        });
        for (const auto& test : cases) {
            SCOPED_TRACE(test.name);
            auto bytes = valid;
            test.mutate(bytes);
            const auto parsed = MetricsChapter::from_bytes(bytes);
            ASSERT_FALSE(parsed);
            EXPECT_EQ(parsed.error().code(), test.code);
        }
    }

    class P5MetricsRestoreTest
        : public ::testing::Test {
    protected:
        void SetUp() override {
            lfs::event::EventBridge::instance()
                .clear_all();
            auto& command_center =
                lfs::training::CommandCenter::instance();
            command_center.clear_snapshot(
                command_center.snapshot().trainer);
            command_center.clear_loss_history();
        }

        void TearDown() override {
            auto& command_center =
                lfs::training::CommandCenter::instance();
            command_center.clear_snapshot(
                command_center.snapshot().trainer);
            command_center.clear_loss_history();
            lfs::event::EventBridge::instance()
                .clear_all();
        }
    };

    TEST_F(P5MetricsRestoreTest,
           ResumeRebuildsPopulatedGraphBuffers) {
        MetricsChapter metrics;
        metrics.loss_history = {
            {.iteration = 3, .value = 0.8f},
            {.iteration = 9, .value = 0.4f},
        };
        metrics.psnr_history = {
            {.iteration = 3, .value = 18.0f},
            {.iteration = 9, .value = 23.0f},
        };
        metrics.accumulated_training_seconds =
            125.5;
        metrics.last_evaluation = {
            .iteration = 9,
            .psnr = 23.0f,
            .ssim = 0.93f,
        };

        lfs::vis::TrainerManager manager;
        manager.restoreProjectMetrics(metrics);
        const auto loss = manager.getLossBuffer();
        const auto psnr = manager.getPSNRBuffer();
        ASSERT_EQ(loss.size(), 2u);
        ASSERT_EQ(psnr.size(), 2u);
        EXPECT_FLOAT_EQ(loss.back(), 0.4f);
        EXPECT_FLOAT_EQ(psnr.back(), 23.0f);
        EXPECT_NEAR(
            manager.getElapsedSeconds(),
            125.5f, 0.001f);
        const auto last =
            manager.getLastEvaluationMetrics();
        ASSERT_TRUE(last);
        EXPECT_EQ(last->iteration, 9);
        EXPECT_FLOAT_EQ(last->psnr, 23.0f);
        EXPECT_FLOAT_EQ(last->ssim, 0.93f);

        const auto recaptured =
            manager.captureProjectMetrics();
        EXPECT_EQ(recaptured, metrics);

        lfs::core::Scene checkpoint_scene;
        const auto cameras =
            checkpoint_scene.addGroup("Cameras");
        const auto training_cameras =
            checkpoint_scene.addCameraGroup(
                "Training", cameras, 1);
        checkpoint_scene.addCamera(
            "train.png",
            training_cameras,
            std::make_shared<lfs::core::Camera>());
        manager.setTrainerFromCheckpoint(
            std::make_unique<
                lfs::training::Trainer>(
                checkpoint_scene),
            9);
        EXPECT_NEAR(
            manager.getElapsedSeconds(),
            125.5f, 0.001f);
    }

    TEST_F(P5MetricsRestoreTest,
           ElapsedSurvivesCheckpointInstallThenMetrics) {
        MetricsChapter metrics;
        metrics.accumulated_training_seconds =
            88.25;
        lfs::core::Scene checkpoint_scene;
        const auto cameras =
            checkpoint_scene.addGroup("Cameras");
        const auto training_cameras =
            checkpoint_scene.addCameraGroup(
                "Training", cameras, 1);
        checkpoint_scene.addCamera(
            "train.png",
            training_cameras,
            std::make_shared<lfs::core::Camera>());
        lfs::vis::TrainerManager manager;
        manager.setTrainerFromCheckpoint(
            std::make_unique<
                lfs::training::Trainer>(
                checkpoint_scene),
            4);
        EXPECT_FLOAT_EQ(
            manager.getElapsedSeconds(), 0.0f);
        manager.restoreProjectMetrics(metrics);
        EXPECT_NEAR(
            manager.getElapsedSeconds(),
            88.25f, 0.001f);
    }

    TEST_F(P5MetricsRestoreTest,
           FinishedRestoreDisablesResumeBothOrders) {
        MetricsChapter metrics;
        metrics.loss_history = {
            {.iteration = 30, .value = 0.12f},
        };
        metrics.accumulated_training_seconds =
            12.0;
        metrics.finish_reason =
            TrainingFinishReason::Completed;

        const auto make_scene = [] {
            auto scene = std::make_unique<
                lfs::core::Scene>();
            const auto cameras =
                scene->addGroup("Cameras");
            const auto training_cameras =
                scene->addCameraGroup(
                    "Training", cameras, 1);
            scene->addCamera(
                "train.png",
                training_cameras,
                std::make_shared<
                    lfs::core::Camera>());
            return scene;
        };

        {
            auto scene = make_scene();
            lfs::vis::TrainerManager manager;
            manager.restoreProjectMetrics(metrics);
            manager.setTrainerFromCheckpoint(
                std::make_unique<
                    lfs::training::Trainer>(
                    *scene),
                30);
            EXPECT_EQ(
                manager.getState(),
                lfs::vis::TrainingState::Finished);
            EXPECT_EQ(
                manager.getStateMachine()
                    .getFinishReason(),
                lfs::vis::FinishReason::Completed);
            EXPECT_FALSE(manager.canResume());
            EXPECT_NEAR(
                manager.getElapsedSeconds(),
                12.0f, 0.001f);
            EXPECT_FLOAT_EQ(
                manager.getCurrentLoss(), 0.12f);
        }
        {
            auto scene = make_scene();
            lfs::vis::TrainerManager manager;
            manager.setTrainerFromCheckpoint(
                std::make_unique<
                    lfs::training::Trainer>(
                    *scene),
                30);
            EXPECT_EQ(
                manager.getState(),
                lfs::vis::TrainingState::Paused);
            manager.restoreProjectMetrics(metrics);
            EXPECT_EQ(
                manager.getState(),
                lfs::vis::TrainingState::Finished);
            EXPECT_EQ(
                manager.getStateMachine()
                    .getFinishReason(),
                lfs::vis::FinishReason::Completed);
            EXPECT_FALSE(manager.canResume());
            EXPECT_NEAR(
                manager.getElapsedSeconds(),
                12.0f, 0.001f);
        }
    }

    TEST_F(P5MetricsRestoreTest,
           UserStoppedRestoreStaysPausedAndResumableBothOrders) {
        MetricsChapter metrics;
        metrics.loss_history = {
            {.iteration = 20, .value = 0.21f},
        };
        metrics.accumulated_training_seconds =
            7.5;
        metrics.finish_reason =
            TrainingFinishReason::UserStopped;

        const auto make_scene = [] {
            auto scene = std::make_unique<
                lfs::core::Scene>();
            const auto cameras =
                scene->addGroup("Cameras");
            const auto training_cameras =
                scene->addCameraGroup(
                    "Training", cameras, 1);
            scene->addCamera(
                "train.png",
                training_cameras,
                std::make_shared<
                    lfs::core::Camera>());
            return scene;
        };

        const auto expect_resumable =
            [](lfs::vis::TrainerManager& manager) {
                EXPECT_EQ(
                    manager.getState(),
                    lfs::vis::TrainingState::Paused);
                EXPECT_EQ(
                    manager.getStateMachine()
                        .getFinishReason(),
                    lfs::vis::FinishReason::None);
                EXPECT_TRUE(manager.canResume());
                EXPECT_TRUE(manager.canPerform(
                    lfs::vis::TrainingAction::Resume));
                EXPECT_FALSE(manager.canStart());
                EXPECT_FALSE(manager.isFinished());
                EXPECT_NEAR(
                    manager.getElapsedSeconds(),
                    7.5f, 0.001f);
                const auto snapshot =
                    lfs::training::CommandCenter::
                        instance()
                            .snapshot();
                EXPECT_EQ(snapshot.iteration, 20);
                EXPECT_TRUE(snapshot.is_paused);
                EXPECT_FALSE(snapshot.is_running);
            };

        {
            bool completed_emitted = false;
            auto scene = make_scene();
            lfs::vis::TrainerManager manager;
            lfs::core::events::state::TrainingCompleted::
                when([&](const auto&) {
                    completed_emitted = true;
                });
            manager.restoreProjectMetrics(metrics);
            manager.setTrainerFromCheckpoint(
                std::make_unique<
                    lfs::training::Trainer>(
                    *scene),
                20);
            expect_resumable(manager);
            EXPECT_FALSE(completed_emitted);
        }
        {
            bool completed_emitted = false;
            auto scene = make_scene();
            lfs::vis::TrainerManager manager;
            lfs::core::events::state::TrainingCompleted::
                when([&](const auto&) {
                    completed_emitted = true;
                });
            manager.setTrainerFromCheckpoint(
                std::make_unique<
                    lfs::training::Trainer>(
                    *scene),
                20);
            EXPECT_EQ(
                manager.getState(),
                lfs::vis::TrainingState::Paused);
            manager.restoreProjectMetrics(metrics);
            expect_resumable(manager);
            EXPECT_FALSE(completed_emitted);
        }
    }

    TEST_F(P5MetricsRestoreTest,
           ErrorRestoreStaysPausedAndResumableBothOrders) {
        MetricsChapter metrics;
        metrics.loss_history = {
            {.iteration = 8, .value = 0.31f},
        };
        metrics.accumulated_training_seconds =
            3.0;
        metrics.finish_reason =
            TrainingFinishReason::Error;

        const auto make_scene = [] {
            auto scene = std::make_unique<
                lfs::core::Scene>();
            const auto cameras =
                scene->addGroup("Cameras");
            const auto training_cameras =
                scene->addCameraGroup(
                    "Training", cameras, 1);
            scene->addCamera(
                "train.png",
                training_cameras,
                std::make_shared<
                    lfs::core::Camera>());
            return scene;
        };

        const auto expect_resumable =
            [](lfs::vis::TrainerManager& manager) {
                EXPECT_EQ(
                    manager.getState(),
                    lfs::vis::TrainingState::Paused);
                EXPECT_EQ(
                    manager.getStateMachine()
                        .getFinishReason(),
                    lfs::vis::FinishReason::None);
                EXPECT_TRUE(manager.canResume());
                EXPECT_TRUE(manager.canPerform(
                    lfs::vis::TrainingAction::Resume));
                EXPECT_FALSE(manager.canStart());
                EXPECT_FALSE(manager.isFinished());
                EXPECT_NEAR(
                    manager.getElapsedSeconds(),
                    3.0f, 0.001f);
                const auto snapshot =
                    lfs::training::CommandCenter::
                        instance()
                            .snapshot();
                EXPECT_EQ(snapshot.iteration, 8);
                EXPECT_TRUE(snapshot.is_paused);
                EXPECT_FALSE(snapshot.is_running);
            };

        {
            bool completed_emitted = false;
            auto scene = make_scene();
            lfs::vis::TrainerManager manager;
            lfs::core::events::state::TrainingCompleted::
                when([&](const auto&) {
                    completed_emitted = true;
                });
            manager.restoreProjectMetrics(metrics);
            manager.setTrainerFromCheckpoint(
                std::make_unique<
                    lfs::training::Trainer>(
                    *scene),
                8);
            expect_resumable(manager);
            EXPECT_FALSE(completed_emitted);
        }
        {
            bool completed_emitted = false;
            auto scene = make_scene();
            lfs::vis::TrainerManager manager;
            lfs::core::events::state::TrainingCompleted::
                when([&](const auto&) {
                    completed_emitted = true;
                });
            manager.setTrainerFromCheckpoint(
                std::make_unique<
                    lfs::training::Trainer>(
                    *scene),
                8);
            EXPECT_EQ(
                manager.getState(),
                lfs::vis::TrainingState::Paused);
            manager.restoreProjectMetrics(metrics);
            expect_resumable(manager);
            EXPECT_FALSE(completed_emitted);
        }
    }

    TEST_F(P5MetricsRestoreTest,
           UserStoppedAtTotalRestoresCompleted) {
        MetricsChapter metrics;
        metrics.accumulated_training_seconds =
            4.0;
        metrics.finish_reason =
            TrainingFinishReason::UserStopped;
        lfs::core::Scene scene;
        const auto cameras = scene.addGroup("Cameras");
        const auto training_cameras =
            scene.addCameraGroup(
                "Training", cameras, 1);
        scene.addCamera(
            "train.png",
            training_cameras,
            std::make_shared<lfs::core::Camera>());
        auto trainer = std::make_unique<
            lfs::training::Trainer>(scene);
        auto params = trainer->getParams();
        params.optimization.iterations = 10;
        params.optimization.enable_sparsity =
            false;
        trainer->setParams(params);
        lfs::vis::TrainerManager manager;
        manager.restoreProjectMetrics(metrics);
        manager.setTrainerFromCheckpoint(
            std::move(trainer), 10);
        EXPECT_EQ(
            manager.getState(),
            lfs::vis::TrainingState::Finished);
        EXPECT_EQ(
            manager.getStateMachine()
                .getFinishReason(),
            lfs::vis::FinishReason::Completed);
        EXPECT_FALSE(manager.canResume());
        EXPECT_FALSE(manager.canStart());
    }

    TEST_F(P5MetricsRestoreTest,
           IterationAtTotalRestoresCompletedWithoutFinishField) {
        MetricsChapter metrics;
        metrics.accumulated_training_seconds =
            4.0;
        lfs::core::Scene scene;
        const auto cameras = scene.addGroup("Cameras");
        const auto training_cameras =
            scene.addCameraGroup(
                "Training", cameras, 1);
        scene.addCamera(
            "train.png",
            training_cameras,
            std::make_shared<lfs::core::Camera>());
        auto trainer = std::make_unique<
            lfs::training::Trainer>(scene);
        auto params = trainer->getParams();
        params.optimization.iterations = 10;
        params.optimization.enable_sparsity =
            false;
        trainer->setParams(params);
        lfs::vis::TrainerManager manager;
        manager.restoreProjectMetrics(metrics);
        manager.setTrainerFromCheckpoint(
            std::move(trainer), 10);
        EXPECT_EQ(
            manager.getState(),
            lfs::vis::TrainingState::Finished);
        EXPECT_EQ(
            manager.getStateMachine()
                .getFinishReason(),
            lfs::vis::FinishReason::Completed);
        EXPECT_FALSE(manager.canResume());
    }

    TEST_F(P5MetricsRestoreTest,
           ProjectCheckpointLocksSaveSteps) {
        lfs::core::Scene scene;
        const auto cameras = scene.addGroup("Cameras");
        const auto training_cameras =
            scene.addCameraGroup(
                "Training", cameras, 1);
        scene.addCamera(
            "train.png",
            training_cameras,
            std::make_shared<lfs::core::Camera>());
        auto trainer = std::make_unique<
            lfs::training::Trainer>(scene);
        auto params = trainer->getParams();
        params.resume_project =
            std::filesystem::path{"/tmp/live.licht"};
        trainer->setParams(params);
        lfs::vis::TrainerManager manager;
        manager.setTrainerFromCheckpoint(
            std::move(trainer), 3);
        EXPECT_FALSE(manager.canEditSaveSteps());
    }

    TEST_F(P5MetricsRestoreTest,
           IdleManagerShutdownCannotMissReaperStopWake) {
        for (int attempt = 0; attempt < 64; ++attempt) {
            lfs::vis::TrainerManager manager;
        }
    }

    TEST(P5MatrixProof,
         EveryAssignedRowSurvivesSaveLoadAndStagesBeforeMutation) {
        const auto session = make_populated_session_chapters();
        TemporaryDirectory temporary{"licht-p5"};
        const auto project_path =
            temporary.path / "p5-matrix.licht";

        auto document = ProjectDocument::create(
            lfs::core::generate_uuid_v4(), 100);
        ASSERT_TRUE(document)
            << lfs::format_for_developer(
                   document.error());
        const Json session_view =
            json_root(session.view.dom());
        const auto environment_reference =
            lfs::core::Uuid::from_string(
                session_view["render_settings"]
                            ["environment_reference_uuid"]
                                .get<std::string>());
        const Json session_sequencer =
            json_root(session.sequencer.dom());
        const auto sequence_reference =
            lfs::core::Uuid::from_string(
                session_sequencer["ply_sequences"][0]
                                 ["directory_reference_uuid"]
                                     .get<std::string>());
        ASSERT_TRUE(environment_reference);
        ASSERT_TRUE(sequence_reference);
        require_status(
            document->edit_references().upsert(
                ReferenceRecord{
                    .uuid = *environment_reference,
                    .key = "view.environment",
                    .kind = "environment_map",
                    .locator =
                        {
                            .preferred =
                                "assets/matrix.hdr",
                            .base =
                                LocatorBase::Project,
                        },
                    .fingerprint =
                        lfs::test::licht::fingerprint(
                            31, FingerprintKind::File),
                    .unresolved = true,
                }));
        require_status(
            document->edit_references().upsert(
                ReferenceRecord{
                    .uuid = *sequence_reference,
                    .key =
                        "sequencer.matrix_frames",
                    .kind =
                        "ply_sequence_directory",
                    .locator =
                        {
                            .preferred =
                                "matrix-frames",
                            .base =
                                LocatorBase::Project,
                        },
                    .fingerprint =
                        lfs::test::licht::fingerprint(
                            32, FingerprintKind::Directory),
                    .unresolved = true,
                }));
        document->edit_gui_layout() =
            session.gui_layout;
        document->edit_editor() = session.editor;
        document->edit_view() = session.view;
        document->edit_sequencer() =
            session.sequencer;
        document->edit_metrics() =
            session.metrics;
        auto saved = document->save(
            project_path,
            ProjectDocumentSaveOptions{
                .disk_reserve_bytes = 0,
            });
        ASSERT_TRUE(saved)
            << lfs::format_for_developer(
                   saved.error());

        auto reopened =
            ProjectDocument::open(project_path);
        ASSERT_TRUE(reopened)
            << lfs::format_for_developer(
                   reopened.error());
        auto reverse =
            reopened->reverse_reference_index();
        ASSERT_TRUE(reverse)
            << lfs::format_for_developer(
                   reverse.error());
        EXPECT_TRUE(std::ranges::any_of(
            reverse->at(*environment_reference),
            [](const ReferenceOwnerBinding& binding) {
                return binding.chapter == "VIEW" &&
                       binding.field ==
                           "render_settings.environment_reference_uuid";
            }));
        EXPECT_TRUE(std::ranges::any_of(
            reverse->at(*sequence_reference),
            [](const ReferenceOwnerBinding& binding) {
                return binding.chapter == "SEQR" &&
                       binding.owner_uuid.has_value() &&
                       binding.field ==
                           "ply_sequences.directory_reference_uuid";
            }));

        lfs::core::Scene live_scene;
        ASSERT_NE(
            live_scene.addGroup("Live sentinel"),
            lfs::core::NULL_NODE);
        auto staged =
            reopened->stage_hydration(live_scene);
        ASSERT_TRUE(staged)
            << lfs::format_for_developer(
                   staged.error());
        EXPECT_EQ(live_scene.getNodeCount(), 1u);
        EXPECT_EQ(
            staged->report()
                .pending_session.gui_layout
                .dom()
                .dump(),
            reopened->gui_layout().dom().dump());
        EXPECT_EQ(
            staged->report()
                .pending_session.view.dom()
                .dump(),
            reopened->view().dom().dump());
        EXPECT_EQ(
            staged->report()
                .pending_session.editor.dom()
                .dump(),
            reopened->editor().dom().dump());
        EXPECT_EQ(
            staged->report()
                .pending_session.sequencer
                .dom()
                .dump(),
            reopened->sequencer().dom().dump());
        EXPECT_EQ(
            staged->report().pending_session.metrics,
            reopened->metrics());

        GuiSessionRestoreCoordinator coordinator;
        require_status(coordinator.stage(
            staged->report().pending_session));
        coordinator.onFirstGuiFrame();
        EXPECT_FALSE(coordinator.takeReady());
        coordinator.onPanelsReady(88);
        ASSERT_TRUE(coordinator.takeReady());
        EXPECT_EQ(live_scene.getNodeCount(), 1u);

        const auto hydration =
            ProjectDocument::commit_hydration(
                live_scene,
                std::move(*staged));
        (void)hydration;
        EXPECT_EQ(live_scene.getNodeCount(), 0u);

        std::set<std::string> proven;
        const auto prove = [&proven](const std::string_view row) {
            EXPECT_TRUE(proven.emplace(row).second) << row;
        };
        const auto at = [](const Json& root, const std::string_view path)
            -> const Json& {
            return root.at(Json::json_pointer(std::string(path)));
        };
        const auto expect_json = [&](const Json& root, const std::string_view path,
                                     Json expected) {
            EXPECT_EQ(at(root, path), expected) << path;
        };
        const auto expect_float = [&](const Json& root, const std::string_view path,
                                      const float expected) {
            EXPECT_FLOAT_EQ(at(root, path).get<float>(), expected) << path;
        };
        const auto expect_bool = [&](const Json& root, const std::string_view path,
                                     const bool expected) {
            EXPECT_EQ(at(root, path).get<bool>(), expected) << path;
        };

        const Json gui = json_root(reopened->gui_layout().dom());
        const auto& spaces = gui["layouts"][0]["areas"][0]["spaces"];
        const auto& fixed = spaces[0]["opaque_payload"];
        const auto& registry = spaces[1]["opaque_payload"];
        const auto& panel = registry["panels"][0];
        const auto& console = spaces[2]["opaque_payload"];

        expect_float(fixed, "/right_panel_width", 417.0f);
        expect_float(fixed, "/scene_panel_ratio", 0.61f);
        expect_float(fixed, "/python_console_width", 511.0f);
        expect_float(fixed, "/bottom_dock_height", 288.0f);
        expect_float(fixed, "/left_dock_width", 271.0f);
        prove("GUIL-166");
        expect_bool(fixed, "/sequencer_visible", true);
        expect_bool(fixed, "/python_console_visible", true);
        expect_bool(panel, "/enabled", true);
        prove("GUIL-167");
        expect_json(panel, "/parent_id", "main");
        expect_json(panel, "/space", "floating");
        expect_json(panel, "/order", 7);
        expect_json(registry, "/active_tabs/main_panel", "training");
        expect_json(registry, "/active_tabs/scene_panel", "history");
        prove("GUIL-168");
        expect_float(panel, "/float_x", 31.0f);
        expect_float(panel, "/float_last_w", 640.0f);
        expect_json(panel, "/float_stack_order", 12);
        expect_json(panel, "/vendor_extension", "retained");
        prove("GUIL-169");
        EXPECT_FALSE(fixed.contains("window"));
        prove("GUIL-170");
        expect_json(console, "/active_tab", 1);
        expect_float(console, "/font_scale", 1.7f);
        prove("GUIL-171");

        const Json editor = json_root(reopened->editor().dom());
        ASSERT_EQ(editor["open_files"].size(), 2u);
        expect_json(editor, "/open_files/0/locator", "project://scripts/a.py");
        expect_json(editor, "/open_files/1/locator", "project://scripts/b.py");
        prove("EDTR-179");
        expect_json(editor, "/active_file", "project://scripts/b.py");
        prove("EDTR-180");
        expect_bool(editor, "/open_files/1/modified", true);
        expect_json(editor, "/open_files/1/embedded_buffer", "token = 'secret'\\n");
        expect_bool(editor, "/open_files/1/share_warning", true);
        expect_bool(editor, "/contains_embedded_secrets", true);
        prove("EDTR-181");
        expect_json(editor, "/open_files/0/cursor_byte", 4);
        expect_json(editor, "/open_files/0/selection_anchor_byte", 1);
        prove("EDTR-182");
        expect_float(editor, "/open_files/0/scroll_y", 44.0f);
        expect_bool(editor, "/open_files/0/folds/0/collapsed", true);
        prove("EDTR-183");
        expect_bool(editor, "/vim_mode", true);
        prove("EDTR-184");

        const Json view = json_root(reopened->view().dom());
        const auto& render =
            view["render_settings"];
        expect_float(render, "/focal_length_mm", 73.0f);
        expect_bool(render, "/antialiasing", true);
        expect_bool(render, "/mip_filter", true);
        expect_json(render, "/sh_degree", 2);
        expect_json(render, "/camera_metrics_mode", 2);
        prove("VIEW-192");
        expect_bool(render, "/show_crop_box", true);
        expect_bool(render, "/use_ellipsoid", true);
        expect_bool(render, "/desaturate_cropping", true);
        expect_bool(render, "/crop_filter_for_selection", true);
        prove("VIEW-193");
        expect_bool(render, "/apply_appearance_correction", true);
        expect_json(render, "/ppisp_mode", 0);
        expect_float(render, "/ppisp_overrides/exposure_offset", 1.25f);
        expect_float(render, "/ppisp_overrides/gamma_multiplier", 1.3f);
        prove("VIEW-194");
        expect_json(render, "/background_color", Json::array({0.1f, 0.2f, 0.3f}));
        EXPECT_TRUE(at(render, "/environment_reference_uuid").is_string());
        EXPECT_TRUE(at(render, "/environment_builtin").is_null());
        expect_float(render, "/environment_exposure", 1.75f);
        prove("VIEW-195");
        expect_bool(render, "/show_coord_axes", true);
        expect_json(render, "/axes_visibility", Json::array({true, false, true}));
        expect_json(render, "/grid_plane", 2);
        prove("VIEW-196");
        expect_bool(render, "/point_cloud_mode", true);
        expect_bool(render, "/show_rings", true);
        expect_bool(render, "/show_camera_frustums", true);
        expect_bool(render, "/show_pivot", true);
        prove("VIEW-197");
        expect_json(render, "/split_view_mode", 3);
        expect_json(render, "/gt_comparison_mode", 2);
        expect_json(render, "/raster_backend", "3dgut");
        EXPECT_FALSE(render.contains("gut"));
        expect_bool(render, "/orthographic", true);
        expect_bool(render, "/depth_view", true);
        prove("VIEW-198");
        expect_json(render, "/selection_color_committed", Json::array({0.11f, 0.22f, 0.33f}));
        expect_bool(render, "/depth_clip_enabled", true);
        expect_float(render, "/depth_clip_far", 34.0f);
        prove("VIEW-199");
        expect_bool(render, "/mesh_wireframe", true);
        expect_float(render, "/mesh_wireframe_width", 2.5f);
        expect_bool(render, "/mesh_shadow_enabled", true);
        expect_json(render, "/mesh_shadow_resolution", 4096);
        prove("VIEW-200");
        expect_bool(render, "/depth_filter_enabled", true);
        expect_json(render, "/depth_filter_transform/translation",
                    Json::array({1.0f, 2.0f, 3.0f}));
        EXPECT_EQ(at(render, "/depth_filter_transform/rotation").size(), 9u);
        prove("VIEW-201");
        expect_bool(render, "/lod_enabled", true);
        expect_json(render, "/lod_max_splats", 1'234'567);
        expect_json(render, "/lod_page_pool_splats", 765'432);
        expect_bool(render, "/lod_debug_colors", true);
        prove("VIEW-202");
        expect_json(view, "/split/panel_grid_planes", Json::array({0, 2}));
        prove("VIEW-203");
        ASSERT_EQ(at(view, "/panel_cameras").size(), 2u);
        expect_json(view, "/panel_cameras/0/R",
                    Json::array({0.0f, 1.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f}));
        EXPECT_TRUE(at(view, "/panel_cameras/1/ortho_scale").is_number());
        prove("VIEW-204");
        expect_json(view, "/navigation/mode", "drone");
        expect_bool(view, "/navigation/view_snap", true);
        prove("VIEW-205");
        expect_json(view, "/split/focused_panel", "right");
        expect_json(view, "/split/gt_camera_id", 41);
        prove("VIEW-206");
        ASSERT_EQ(at(view, "/camera_bookmarks").size(), 1u);
        expect_json(view, "/camera_bookmarks/0/id", "bookmark.matrix");
        expect_json(view, "/camera_bookmarks/0/name", "Rolled view");
        prove("VIEW-207");
        expect_json(view, "/tools/active_tool_id", "crop");
        expect_json(view, "/tools/active_submode_id", "brush");
        expect_json(view, "/tools/gizmo_operation", "rotate");
        expect_json(view, "/tools/transform_space", "local");
        prove("VIEW-208");
        expect_float(view, "/tools/selection/brush_radius", 37.0f);
        expect_bool(view, "/tools/selection/restrict_to_selected_nodes", true);
        prove("VIEW-209");
        expect_bool(view, "/sequencer_view/show_camera_path", false);
        prove("VIEW-210");

        const Json sequencer = json_root(reopened->sequencer().dom());
        expect_float(sequencer, "/timeline/clip_duration", 48.0f);
        ASSERT_EQ(at(sequencer, "/timeline/keyframes").size(), 1u);
        expect_json(sequencer, "/timeline/keyframes/0/position",
                    Json::array({1.0f, 2.0f, 3.0f}));
        prove("SEQR-218");
        expect_json(sequencer, "/timeline/animation_clip/name", "Matrix animation");
        ASSERT_EQ(at(sequencer, "/timeline/animation_clip/tracks").size(), 1u);
        expect_json(sequencer, "/timeline/animation_clip/tracks/0/target", "node.opacity");
        prove("SEQR-219");
        ASSERT_EQ(at(sequencer, "/ply_sequences").size(), 1u);
        expect_json(sequencer, "/ply_sequences/0/frames/0/locator", "frame_0007.ply");
        expect_float(sequencer, "/ply_sequences/0/fps", 17.5f);
        EXPECT_FALSE(at(sequencer, "/ply_sequences/0").contains("sequence_fps"));
        prove("SEQR-220");
        expect_float(sequencer, "/playhead", 3.5f);
        expect_json(sequencer, "/loop_mode", "ping_pong");
        expect_float(sequencer, "/playback_speed", 1.75f);
        prove("SEQR-221");
        expect_bool(sequencer, "/preferences/snap_to_grid", true);
        expect_bool(sequencer, "/preferences/follow_playback", true);
        expect_bool(sequencer, "/preferences/show_film_strip", false);
        prove("SEQR-222");

        const auto& metrics = reopened->metrics();
        ASSERT_EQ(metrics.loss_history.size(), 2u);
        EXPECT_EQ(metrics.loss_history[1].iteration, 20);
        EXPECT_FLOAT_EQ(metrics.loss_history[1].value, 0.21f);
        prove("METR-230");
        ASSERT_EQ(metrics.psnr_history.size(), 2u);
        EXPECT_FLOAT_EQ(metrics.psnr_history[1].value, 24.75f);
        prove("METR-231");
        EXPECT_DOUBLE_EQ(metrics.accumulated_training_seconds, 37.5);
        prove("METR-232");
        ASSERT_TRUE(metrics.last_evaluation);
        EXPECT_EQ(metrics.last_evaluation->iteration, 20);
        EXPECT_FLOAT_EQ(metrics.last_evaluation->psnr, 24.75f);
        EXPECT_FLOAT_EQ(metrics.last_evaluation->ssim, 0.91f);
        prove("METR-233");

        const auto registered = lfs::test::licht::word_set(
            lfs::test::licht::P5_MATRIX_ROW_DATA);
        EXPECT_EQ(proven, registered)
            << "Every registered P5 row must reach an explicit "
               "save->load->Phase-A assertion";
    }

    TEST(P5SessionChapterTest, SceneTreeCollapseUsesStableUuidsNotNodeIds) {
        lfs::core::Scene scene;
        const auto cameras = scene.addGroup("Cameras");
        const auto models = scene.addGroup("Models");
        ASSERT_NE(cameras, lfs::core::NULL_NODE);
        ASSERT_NE(models, lfs::core::NULL_NODE);
        const auto cameras_uuid = scene.getNodeUuid(cameras).to_string();
        const auto models_uuid = scene.getNodeUuid(models).to_string();
        EXPECT_FALSE(cameras_uuid.empty());
        EXPECT_NE(cameras_uuid, models_uuid);

        const auto collapsed = lfs::vis::gui::collapsedIdsFromUuids(
            scene, {cameras_uuid});
        EXPECT_TRUE(collapsed.contains(cameras));
        EXPECT_FALSE(collapsed.contains(models));

        const auto written = lfs::vis::gui::collapsedUuidsFromIds(
            scene, collapsed);
        ASSERT_EQ(written.size(), 1u);
        EXPECT_EQ(written.front(), cameras_uuid);
    }

    TEST(P5SessionChapterTest, GuilPanelPayloadsAreAdditiveAndRetained) {
        GuiLayoutChapter chapter;
        Json root = json_root(chapter.dom());
        auto& registry =
            root["layouts"][0]["areas"][0]["spaces"][1]["opaque_payload"];
        registry["panel_payloads"] = {
            {"lfs.histogram",
             {{"metric_id", "opacity"},
              {"log_scale", true},
              {"bin_count", 64},
              {"vendor_extra", "keep"}}},
        };
        auto loaded = GuiLayoutChapter::parse(root.dump());
        ASSERT_TRUE(loaded) << lfs::format_for_developer(loaded.error());

        const Json known{
            {"layouts",
             Json::array({
                 {
                     {"areas",
                      Json::array({
                          {
                              {"spaces",
                               Json::array({
                                   {
                                       {"type", "panel_registry"},
                                       {"opaque_payload",
                                        {
                                            {"panel_payloads",
                                             {
                                                 {"lfs.histogram",
                                                  {{"metric_id", "scale"},
                                                   {"log_scale", false}}},
                                             }},
                                        }},
                                   },
                               })},
                          },
                      })},
                 },
             })},
        };
        require_status(loaded->merge_known_state(known));
        const Json merged = json_root(loaded->dom());
        const auto& payload =
            merged["layouts"][0]["areas"][0]["spaces"][1]
                  ["opaque_payload"]["panel_payloads"]["lfs.histogram"];
        EXPECT_EQ(payload["metric_id"], "scale");
        EXPECT_EQ(payload["log_scale"], false);
        EXPECT_EQ(payload["bin_count"], 64);
        EXPECT_EQ(payload["vendor_extra"], "keep");
    }

    TEST(P5SessionChapterTest, SequencerExportPrefsAndViewRoundTrip) {
        lfs::vis::ViewerOptions options;
        options.show_startup_overlay = false;
        lfs::vis::VisualizerImpl viewer(options);
        ASSERT_NE(viewer.getGuiManager(), nullptr);

        auto session = make_populated_session_chapters();
        auto seq = json_root(session.sequencer.dom());
        seq["preferences"]["preset"] = "tiktok_hd";
        seq["preferences"]["custom_width"] = 1280;
        seq["preferences"]["custom_height"] = 720;
        seq["preferences"]["framerate"] = 48;
        seq["preferences"]["quality"] = 22;
        seq["view"] = {
            {"zoom", 2.5f},
            {"pan", 1.25f},
            {"selected_keyframe_id", nullptr},
        };
        session.sequencer = require_result(
            SequencerSessionChapter::parse(seq.dump()));
        auto prepared = require_result(
            prepareGuiSessionRestore(std::move(session)));
        std::vector<CameraBookmarkProjectState> bookmarks;
        applyGuiSession(viewer, prepared, bookmarks);

        const auto& ui = viewer.getGuiManager()->getSequencerUIState();
        EXPECT_EQ(ui.preset, lfs::io::video::VideoPreset::TIKTOK_HD);
        EXPECT_EQ(ui.custom_width, 1280);
        EXPECT_EQ(ui.custom_height, 720);
        EXPECT_EQ(ui.framerate, 48);
        EXPECT_EQ(ui.quality, 22);
        EXPECT_NEAR(viewer.getGuiManager()->sequencerUI().timelineZoom(), 2.5f, 1e-4f);
        EXPECT_NEAR(viewer.getGuiManager()->sequencerUI().timelinePan(), 1.25f, 1e-4f);
        if (const auto* rendering = viewer.getRenderingManager()) {
            EXPECT_EQ(ui.equirectangular, rendering->getSettings().equirectangular);
        }
    }

    TEST(P5SessionChapterTest, SnapshotAbsolutizesRelativeDatasetPath) {
        std::filesystem::path relative{"data/bicycle"};
        ASSERT_FALSE(relative.is_absolute());
        lfs::training::absolutize_dataset_path_for_snapshot(relative);
        EXPECT_TRUE(relative.is_absolute());
        EXPECT_NE(relative.generic_string().find("bicycle"), std::string::npos);

        lfs::core::param::TrainingParameters params;
        params.dataset.data_path = "data/bicycle";
        params.optimization.strategy = std::string(lfs::core::param::kStrategyMCMC);
        lfs::core::Scene scene;
        const auto training = scene.addSplatPlaceholder("Train");
        ASSERT_NE(training, lfs::core::NULL_NODE);
        scene.setTrainingModelNode(scene.getNodeUuid(training));
        lfs::training::ProjectSnapshotCpuState cpu;
        auto captured = lfs::training::capture_project_snapshot_cpu_state(
            scene, params, lfs::core::generate_uuid_v4(), 1, cpu);
        ASSERT_TRUE(captured) << lfs::format_for_developer(captured.error());
        EXPECT_TRUE(cpu.parameters.dataset.data_path.is_absolute());
    }

} // namespace
