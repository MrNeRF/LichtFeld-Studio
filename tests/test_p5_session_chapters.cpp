/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "user_shader_test_access.hpp"

#include "core/camera.hpp"
#include "core/error_bus.hpp"
#include "core/event_bridge/command_api.hpp"
#include "core/event_bridge/event_bridge.hpp"
#include "core/event_bridge/scoped_handler.hpp"
#include "core/events.hpp"
#include "core/parameters.hpp"
#include "core/scene.hpp"
#include "core/training_manager.hpp"
#include "core/training_state.hpp"
#include "core/uuid.hpp"
#include "cuda_backend_test.hpp"
#include "gui/editor/python_editor.hpp"
#include "gui/gui_manager.hpp"
#include "gui/panel_registry.hpp"
#include "gui/panels/python_console_panel.hpp"
#include "gui/scene_tree_session.hpp"
#include "gui/sequencer_ui_state.hpp"
#include "input/input_controller.hpp"
#include "io/project_document.hpp"
#include "io/session_chapters.hpp"
#include "io/video/video_export_options.hpp"
#include "licht_matrix_test_data.hpp"
#include "licht_test_support.hpp"
#include "operation/undo_history.hpp"
#include "p5_session_test_access.hpp"
#include "project/session_state.hpp"
#include "rendering/dirty_flags.hpp"
#include "rendering/rendering_types.hpp"
#include "screen/screen.hpp"
#include "screen/view3d_space.hpp"
#include "sequencer/reveal_render_state.hpp"
#include "sequencer/timeline.hpp"
#include "tools/unified_tool_registry.hpp"
#include "training/project_snapshot_chapters.hpp"
#include "training/trainer.hpp"
#include "user_shader_sources.hpp"
#include "visualizer_impl.hpp"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
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

    class ScopedLfsHome {
    public:
        explicit ScopedLfsHome(const fs::path& path) {
            if (const char* previous = std::getenv("LFS_HOME"))
                previous_ = previous;
#ifdef _WIN32
            (void)_putenv_s("LFS_HOME", path.string().c_str());
#else
            (void)setenv("LFS_HOME", path.string().c_str(), 1);
#endif
        }
        ~ScopedLfsHome() {
#ifdef _WIN32
            (void)_putenv_s("LFS_HOME", previous_ ? previous_->c_str() : "");
#else
            if (previous_)
                (void)setenv("LFS_HOME", previous_->c_str(), 1);
            else
                (void)unsetenv("LFS_HOME");
#endif
        }

    private:
        std::optional<std::string> previous_;
    };

    TEST(P5SessionChapterTest, ShaderAssetCountAndAggregateSourceBoundaries) {
        using Json = lfs::io::JsonChapterDom::Json;
        auto root = json_root(ShaderLibraryChapter{}.dom());
        for (std::size_t i = 0; i <= kMaxShaderAssets; ++i)
            root["assets"].push_back({{"uuid", lfs::test::licht::fixed_uuid(30000 + i).to_string()}, {"name", "Generated"}, {"language", "slang"}, {"source", lfs::test::kIdentity}, {"parameters", Json::object()}});
        auto recovered = require_result(ShaderLibraryChapter::parse(root.dump()));
        EXPECT_EQ(recovered.dom().get_json("assets")->size(), kMaxShaderAssets);
        EXPECT_EQ(recovered.warnings.size(), 1u);
        root["assets"].erase(root["assets"].begin() + 17, root["assets"].end());
        for (auto& asset : root["assets"])
            asset["source"] = std::string(kMaxShaderSourceBytes, ' ');
        recovered = require_result(ShaderLibraryChapter::parse(root.dump()));
        EXPECT_EQ(recovered.dom().get_json("assets")->size(), 16u);
        EXPECT_EQ(recovered.warnings.size(), 1u);
        root["assets"] = Json::array({root["assets"][0]});
        const Json binding{{"target", "global"}, {"shader_uuid", root["assets"][0]["uuid"]}};
        root["bindings"] = Json::array({binding, binding});
        recovered = require_result(ShaderLibraryChapter::parse(root.dump()));
        EXPECT_TRUE(recovered.dom().get_json("bindings")->empty());
        EXPECT_EQ(recovered.warnings.size(), 2u);
    }

    TEST(P5SessionChapterTest, ShaderRecoveryPreservesExtensionsAndSkipsOnlyDamagedEntries) {
        using Json = lfs::io::JsonChapterDom::Json;
        auto root = json_root(ShaderLibraryChapter{}.dom());
        const auto id = lfs::core::generate_uuid_v4().to_string();
        Json valid{{"uuid", id}, {"name", "Generated"}, {"language", "future"}, {"source", lfs::test::kIdentity}, {"parameters", {{"valid", Json::array({1, 2, 3})}, {"wrong", Json::array({1})}, {"bool", true}}}, {"extension", 7}};
        root["assets"] = Json::array({valid, Json{{"uuid", "invalid"}}});
        root["bindings"] = Json::array({Json{{"target", "global"}, {"shader_uuid", id}, {"extension", 8}},
                                        Json{{"target", "node"}, {"node_uuid", "invalid"}, {"shader_uuid", id}}});
        root["extension"] = "kept";
        auto recovered = ShaderLibraryChapter::parse(root.dump());
        ASSERT_TRUE(recovered);
        EXPECT_EQ(recovered->warnings.size(), 4u);
        auto saved = json_root(recovered->dom());
        ASSERT_EQ(saved["assets"].size(), 1u);
        EXPECT_EQ(saved["assets"][0]["parameters"].size(), 1u);
        EXPECT_EQ(saved["assets"][0]["extension"], 7);
        EXPECT_EQ(saved["bindings"][0]["extension"], 8);
        EXPECT_EQ(saved["extension"], "kept");
        root["assets"].push_back(valid);
        recovered = ShaderLibraryChapter::parse(root.dump());
        ASSERT_TRUE(recovered);
        EXPECT_TRUE(recovered->dom().get_json("assets")->empty());
        EXPECT_TRUE(recovered->dom().get_json("bindings")->empty());
        root["version"] = 2;
        EXPECT_FALSE(ShaderLibraryChapter::parse(root.dump()));
        EXPECT_FALSE(ShaderLibraryChapter::parse("{"));
    }

    TEST(P5SessionChapterTest, ShaderLimitsAreBoundedAndDraftWireStaysLegacyValid) {
        using Json = lfs::io::JsonChapterDom::Json;
        auto root = json_root(ShaderLibraryChapter{}.dom());
        root["assets"] = Json::array({{{"uuid", lfs::core::generate_uuid_v4().to_string()}, {"name", "Boundary"}, {"language", "slang"}, {"source", std::string(kMaxShaderSourceBytes, ' ')}, {"parameters", Json::object()}}});
        auto parsed = ShaderLibraryChapter::parse(root.dump());
        ASSERT_TRUE(parsed);
        EXPECT_TRUE(parsed->warnings.empty());
        root["assets"][0]["source"] = std::string(kMaxShaderSourceBytes + 1, ' ');
        parsed = ShaderLibraryChapter::parse(root.dump());
        ASSERT_TRUE(parsed);
        EXPECT_TRUE(parsed->dom().get_json("assets")->empty());
        EXPECT_FALSE(ShaderLibraryChapter::parse(std::string(kMaxShaderChapterBytes + 1, ' ')));
        auto editor = json_root(EditorSessionChapter{}.dom());
        editor["open_files"] = Json::array({{{"locator", "untitled://shader"}, {"kind", "shader"}, {"shader_id", lfs::core::generate_uuid_v4().to_string()}, {"target", {{"kind", "global"}}}, {"modified", false}, {"file_modified", true}}});
        EXPECT_TRUE(EditorSessionChapter::parse(editor.dump()));
        editor["open_files"][0].erase("kind");
        EXPECT_TRUE(EditorSessionChapter::parse(editor.dump()));
    }

    TEST(P5SessionChapterTest, ShaderLibraryPrecedesCustomAndEditorRestore) {
        lfs::event::EventBridge::instance().clear_all();
        lfs::vis::ViewerOptions options;
        options.show_startup_overlay = false;
        lfs::vis::VisualizerImpl viewer(options);
        lfs::vis::P5SessionCaptureTestAccess::initializeInputController(viewer);
        auto* service = viewer.getShaderService();
        ASSERT_NE(service, nullptr);
        const auto id = require_result(lfs::vis::ShaderCommands{*service}.createAsset("Generated", lfs::test::kIdentity));
        ProjectSessionChapters retained;
        retained.shaders = require_result(service->captureProject(retained.shaders)).chapter;
        auto& controller = viewer.getGuiManager()->sequencer();
        controller.addKeyframe(lfs::sequencer::Keyframe{});
        lfs::sequencer::RevealClip custom;
        custom.effect = lfs::sequencer::RevealEffect::Custom;
        custom.shader_id = id;
        ASSERT_TRUE(controller.setRevealClip(custom));
        auto* editor = lfs::vis::gui::panels::PythonConsoleState::getInstance().getEditor();
        ASSERT_NE(editor, nullptr);
        ASSERT_TRUE(editor->setDocumentMode(editor->document().id, lfs::vis::editor::DocumentKind::Shader, service->assets().front()));
        editor->setText(lfs::test::kIdentity);
        ASSERT_TRUE(editor->setDocumentTarget(editor->document().id, {lfs::vis::shaders::ShaderSlot::Kind::Reveal, {}}, *service));
        ASSERT_TRUE(editor->runShader(*service));
        ASSERT_TRUE(editor->hasShaderBinding(*service));
        std::vector<CameraBookmarkProjectState> bookmarks;
        auto captured = require_result(captureGuiSession(viewer, retained, bookmarks));
        GuiSessionRestoreCoordinator coordinator;
        ASSERT_TRUE(coordinator.stage(captured));
        coordinator.onFirstGuiFrame();
        coordinator.onPanelsReady(1);
        EXPECT_FALSE(coordinator.takeReady(false));
        EXPECT_TRUE(coordinator.hasPending());
        EXPECT_TRUE(coordinator.takeReady(true));
        const auto files = captured.editor.dom().get_json("open_files");
        const auto associated = std::ranges::find_if(*files, [&](const auto& file) { return file.contains("shader_id") && file["shader_id"] == id.to_string(); });
        ASSERT_NE(associated, files->end());
        EXPECT_FALSE(associated->contains("embedded_buffer"));
        ASSERT_TRUE(controller.setRevealClip(lfs::sequencer::RevealClip{}));
        auto builtin = require_result(captureGuiSession(viewer, captured, bookmarks));
        EXPECT_FALSE(json_root(builtin.sequencer.dom())["timeline"]["reveal_clip"].contains("shader_id"));
        service->clear();
        applyGuiSession(viewer, require_result(prepareGuiSessionRestore(captured)), bookmarks);
        ASSERT_TRUE(controller.timeline().revealClip());
        EXPECT_EQ(controller.timeline().revealClip()->shader_id, id);
        EXPECT_EQ(editor->document().shader_asset, id);
        EXPECT_EQ(editor->getText(), lfs::test::kIdentity);
        EXPECT_TRUE(editor->inspectDocument(editor->document().id, service)->owned_bindings.empty());
        EXPECT_TRUE(editor->stopShader(*service).empty());
        EXPECT_EQ(controller.timeline().revealClip()->shader_id, id);
        auto unavailable = captured;
        auto unavailable_assets = *unavailable.shaders.dom().get_json("assets");
        unavailable_assets[0]["language"] = "future";
        ASSERT_TRUE(unavailable.shaders.dom().set_json("assets", unavailable_assets));
        applyGuiSession(viewer, require_result(prepareGuiSessionRestore(unavailable)), bookmarks);
        ASSERT_TRUE(controller.timeline().revealClip());
        EXPECT_EQ(controller.timeline().revealClip()->shader_id, id);
        EXPECT_EQ(service->status(id)->state, lfs::vis::shaders::ShaderState::Failed);
        captured.shaders = ShaderLibraryChapter{};
        applyGuiSession(viewer, require_result(prepareGuiSessionRestore(captured)), bookmarks);
        EXPECT_FALSE(controller.timeline().revealClip());
        EXPECT_TRUE(service->assets().empty());
        EXPECT_EQ(controller.timeline().keyframes().size(), 1u);
    }

    TEST(P5SessionChapterTest, RealProjectRevealRemovalSurvivesSaveAndReopen) {
        const char* source = std::getenv("LFS_REVEAL_TEST_PROJECT");
        if (!source || !fs::exists(source))
            GTEST_SKIP() << "Set LFS_REVEAL_TEST_PROJECT to an existing .licht project";

        TemporaryDirectory temporary;
        const auto path = temporary.path / "reveal-removal.licht";
        fs::copy_file(source, path);
        auto document = require_result(ProjectDocument::open(path));
        ProjectSessionChapters retained{
            .gui_layout = document.gui_layout(),
            .editor = document.editor(),
            .view = document.view(),
            .sequencer = document.sequencer(),
            .metrics = document.metrics(),
        };
        ASSERT_TRUE(retained.sequencer.dom().set_json("timeline.review_extension", "preserve"));
        lfs::vis::ViewerOptions options;
        options.show_startup_overlay = false;
        lfs::vis::VisualizerImpl viewer(options);
        lfs::vis::P5SessionCaptureTestAccess::initializeInputController(viewer);
        std::vector<CameraBookmarkProjectState> bookmarks;
        ASSERT_TRUE(viewer.getGuiManager()->sequencer().setRevealClip(lfs::sequencer::RevealClip{}));
        auto with_reveal = require_result(captureGuiSession(viewer, retained, bookmarks));
        ASSERT_TRUE(json_root(with_reveal.sequencer.dom())["timeline"].contains("reveal_clip"));
        document.edit_sequencer() = with_reveal.sequencer;
        ASSERT_TRUE(document.save(path));

        auto reopened = require_result(ProjectDocument::open(path));
        with_reveal.sequencer = reopened.sequencer();
        ASSERT_TRUE(viewer.getGuiManager()->sequencer().setRevealClip(std::nullopt));
        auto removed = require_result(captureGuiSession(viewer, with_reveal, bookmarks));
        document.edit_sequencer() = removed.sequencer;
        ASSERT_TRUE(document.save(path));
        auto final = require_result(ProjectDocument::open(path));
        const auto timeline = json_root(final.sequencer().dom())["timeline"];
        EXPECT_FALSE(timeline.contains("reveal_clip"));
        EXPECT_EQ(timeline["review_extension"], "preserve");
    }

    TEST(P5SessionChapterTest, CustomRevealRunHistoryAndOwnershipUseOneTransaction) {
        namespace shaders = lfs::vis::shaders;
        lfs::vis::ViewerOptions options;
        options.show_startup_overlay = false;
        lfs::vis::VisualizerImpl viewer(options);
        auto& controller = viewer.getGuiManager()->sequencer();
        auto& service = *viewer.getShaderService();
        auto& history = lfs::vis::op::undoHistory();
        const lfs::sequencer::RevealClip builtin;
        ASSERT_TRUE(controller.setRevealClip(builtin));
        history.clear();
        lfs::vis::editor::PythonEditor editor;
        editor.newDocument("draft", lfs::vis::editor::DocumentKind::Shader);
        const shaders::ShaderSlot target{shaders::ShaderSlot::Kind::Reveal, {}};
        ASSERT_TRUE(editor.setDocumentTarget(editor.document().id, target, service));
        ASSERT_TRUE(editor.runShader(service));
        const auto asset = *editor.document().shader_asset;
        ASSERT_EQ(controller.timeline().revealClip()->shader_id, asset);
        ASSERT_TRUE(editor.hasShaderBinding(service));
        ASSERT_EQ(history.undoCount(), 1u);
        ASSERT_TRUE(history.undo().success);
        EXPECT_EQ(controller.timeline().revealClip(), builtin);
        EXPECT_FALSE(service.status(asset));
        EXPECT_FALSE(editor.hasShaderBinding(service));
        ASSERT_TRUE(history.redo().success);
        EXPECT_EQ(controller.timeline().revealClip()->shader_id, asset);
        EXPECT_EQ(service.boundAsset(target), asset);
        EXPECT_TRUE(editor.hasShaderBinding(service));
        ASSERT_TRUE(editor.runShader(service));
        auto timing = *controller.timeline().revealClip();
        timing.end += 2;
        ASSERT_TRUE(controller.setRevealClip(timing));
        EXPECT_TRUE(editor.hasShaderBinding(service));
        EXPECT_FALSE(service.command(shaders::RemoveShader{asset}));
        ASSERT_TRUE(service.command(shaders::BindShader{target, asset}));
        EXPECT_TRUE(editor.hasShaderBinding(service));
        EXPECT_EQ(editor.stopShader(service), std::vector{target});
        EXPECT_FALSE(controller.timeline().revealClip());
        EXPECT_FALSE(service.boundAsset(target));
        ASSERT_TRUE(history.undo().success);
        EXPECT_EQ(controller.timeline().revealClip(), timing);
        EXPECT_FALSE(editor.hasShaderBinding(service));
        history.clear();
        service.clear();
        EXPECT_FALSE(controller.timeline().revealClip());
    }

    TEST(P5SessionChapterTest, CustomRevealLoadAndStaleHistoryLeaveBindingsIntact) {
        namespace shaders = lfs::vis::shaders;
        lfs::vis::ViewerOptions options;
        options.show_startup_overlay = false;
        lfs::vis::VisualizerImpl viewer(options);
        auto& controller = viewer.getGuiManager()->sequencer();
        auto& service = *viewer.getShaderService();
        auto& history = lfs::vis::op::undoHistory();
        const auto asset = lfs::vis::ShaderCommands{service}.createAsset("Reveal", "draft");
        ASSERT_TRUE(asset);
        lfs::sequencer::RevealClip clip{1, 4, .2f, lfs::sequencer::RevealEffect::Custom, *asset};
        ASSERT_TRUE(controller.setRevealClip(clip));
        const auto saved = controller.saveToJson();
        auto invalid = saved;
        invalid["reveal_clip"]["shader_id"] = lfs::core::generate_uuid_v4().to_string();
        EXPECT_FALSE(controller.loadFromJson(invalid));
        EXPECT_EQ(controller.saveToJson(), saved);
        EXPECT_EQ(service.boundAsset({shaders::ShaderSlot::Kind::Reveal, {}}), *asset);
        history.clear();
        ASSERT_TRUE(controller.setRevealClip(std::nullopt));
        ASSERT_TRUE(lfs::vis::ShaderCommands{service}.removeAsset(*asset));
        EXPECT_FALSE(history.undo().success);
        EXPECT_FALSE(controller.timeline().revealClip());
        EXPECT_FALSE(service.boundAsset({shaders::ShaderSlot::Kind::Reveal, {}}));
        history.clear();
    }

    TEST(P5SessionChapterTest, RevealControllerEditsAreUndoable) {
        lfs::vis::ViewerOptions options;
        options.show_startup_overlay = false;
        lfs::vis::VisualizerImpl viewer(options);
        auto& controller = viewer.getGuiManager()->sequencer();
        auto& history = lfs::vis::op::undoHistory();
        history.clear();
        const float original_duration = controller.clipDuration();
        lfs::sequencer::RevealClip clip;
        clip.end = original_duration + 5.0f;
        ASSERT_TRUE(controller.setRevealClip(clip));
        ASSERT_TRUE(controller.timeline().revealClip());
        EXPECT_FLOAT_EQ(controller.clipDuration(), clip.end);
        ASSERT_TRUE(history.canUndo());
        const auto undo = history.undo();
        EXPECT_TRUE(undo.success);
        EXPECT_TRUE(undo.changed);
        EXPECT_FALSE(controller.timeline().revealClip());
        EXPECT_FLOAT_EQ(controller.clipDuration(), original_duration);
        const auto redo = history.redo();
        EXPECT_TRUE(redo.success);
        EXPECT_TRUE(redo.changed);
        EXPECT_TRUE(controller.timeline().revealClip());
        EXPECT_FLOAT_EQ(controller.clipDuration(), clip.end);
        history.clear();
        auto changed = clip;
        changed.effect = lfs::sequencer::RevealEffect::Unveil;
        changed.softness = 0.4f;
        changed.end -= 1.0f;
        ASSERT_TRUE(controller.setRevealClip(changed));
        ASSERT_EQ(history.undoCount(), 1u);
        const auto undo_change = history.undo();
        EXPECT_TRUE(undo_change.success);
        EXPECT_TRUE(undo_change.changed);
        ASSERT_TRUE(controller.timeline().revealClip());
        EXPECT_EQ(controller.timeline().revealClip()->effect, clip.effect);
        EXPECT_FLOAT_EQ(controller.timeline().revealClip()->softness, clip.softness);
        EXPECT_FLOAT_EQ(controller.timeline().revealClip()->end, clip.end);
        const auto redo_change = history.redo();
        EXPECT_TRUE(redo_change.success);
        EXPECT_TRUE(redo_change.changed);
        ASSERT_TRUE(controller.timeline().revealClip());
        EXPECT_EQ(controller.timeline().revealClip()->effect, changed.effect);
        EXPECT_FLOAT_EQ(controller.timeline().revealClip()->softness, changed.softness);
        EXPECT_FLOAT_EQ(controller.timeline().revealClip()->end, changed.end);
        ASSERT_TRUE(controller.setRevealClip(clip));
        history.clear();
        controller.beginRevealEdit();
        clip.start = 1.0f;
        ASSERT_TRUE(controller.setRevealClip(clip));
        clip.start = 2.0f;
        ASSERT_TRUE(controller.setRevealClip(clip));
        controller.endRevealEdit();
        EXPECT_EQ(history.undoCount(), 1u);
        const auto grouped_undo = history.undo();
        EXPECT_TRUE(grouped_undo.success);
        EXPECT_FLOAT_EQ(controller.timeline().revealClip()->start, 0.0f);
        const auto grouped_redo = history.redo();
        EXPECT_TRUE(grouped_redo.success);
        EXPECT_FLOAT_EQ(controller.timeline().revealClip()->start, 2.0f);
        history.clear();
        ASSERT_TRUE(controller.setRevealClip(std::nullopt));
        EXPECT_FALSE(controller.timeline().revealClip());
        EXPECT_TRUE(history.undo().success);
        EXPECT_TRUE(controller.timeline().revealClip());
        history.clear();
    }

    TEST(P5SessionChapterTest, RealProjectRevealBoundsSurviveConsolidation) {
        const char* source = std::getenv("LFS_REVEAL_TEST_PROJECT");
        if (!source || !fs::exists(source))
            GTEST_SKIP() << "Set LFS_REVEAL_TEST_PROJECT to an existing .licht project";
        auto document = require_result(ProjectDocument::open(source));
        lfs::core::Scene scene;
        auto staged = document.stage_hydration(scene);
        ASSERT_TRUE(staged) << lfs::format_for_developer(staged.error());
        (void)ProjectDocument::commit_hydration(scene, std::move(*staged));
        auto slots = scene.getVisibleSplatNodeSlots();
        ASSERT_FALSE(slots.empty()) << "Project must contain a visible splat model";
        ASSERT_NE(slots.front().node, nullptr);
        ASSERT_NE(slots.front().node->model, nullptr);
        const auto name = slots.front().node->name;
        const auto duplicate = scene.duplicateNode(name);
        ASSERT_FALSE(duplicate.empty());
        slots = scene.getVisibleSplatNodeSlots();
        ASSERT_GE(slots.size(), 2u);
        glm::vec3 before_min, before_max;
        ASSERT_TRUE(scene.getNodeRevealBounds(slots.front().node->id, before_min, before_max));
        ASSERT_GE(scene.consolidateNodeModels(), 2u);
        glm::vec3 after_min, after_max;
        ASSERT_TRUE(scene.getNodeRevealBounds(slots.front().node->id, after_min, after_max));
        EXPECT_EQ(after_min, before_min);
        EXPECT_EQ(after_max, before_max);
        EXPECT_TRUE(lfs::vis::makeRevealRenderState(scene, lfs::sequencer::RevealClip{}, 0.0f));
        EXPECT_FALSE(lfs::vis::makeRevealRenderState(scene, lfs::sequencer::RevealClip{}, 3.0f));
        EXPECT_FALSE(lfs::vis::makeRevealRenderState(scene, lfs::sequencer::RevealClip{}, 4.0f));
    }

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

    TEST(P5SessionChapterTest, NavigationVerticalSurvivesLichtSaveOpenAndSaveAs) {
        TemporaryDirectory temporary;
        for (const std::string axis : {"legacy_y", "data_z", "absent", "future_up"}) {
            SCOPED_TRACE(axis);
            auto document = lfs::test::licht::make_empty_document(lfs::core::generate_uuid_v4());
            ProjectSessionChapters session;
            session.view = document->view();
            session.gui_layout = document->gui_layout();
            session.editor = document->editor();
            session.sequencer = document->sequencer();
            session.metrics = document->metrics();
            auto view = json_root(session.view.dom());
            if (axis == "absent")
                view["navigation"].erase("up_axis");
            else
                view["navigation"]["up_axis"] = axis;
            session.view = require_result(ViewSessionChapter::parse(view.dump()));
            document->edit_view() = session.view;
            document->edit_gui_layout() = session.gui_layout;
            document->edit_editor() = session.editor;
            document->edit_sequencer() = session.sequencer;
            const auto source = temporary.path / (axis + ".licht");
            const auto destination = temporary.path / (axis + "-copy.licht");
            auto saved = document->save(source);
            ASSERT_TRUE(saved) << lfs::format_for_developer(saved.error());
            auto reopened = require_result(ProjectDocument::open(source));
            EXPECT_EQ(json_root(reopened.view().dom()), view);
            ASSERT_TRUE(reopened.save_as(destination));
            auto copied = require_result(ProjectDocument::open(destination));
            EXPECT_EQ(json_root(copied.view().dom()), view);
            session.view = copied.view();
            EXPECT_TRUE(prepareGuiSessionRestore(session));
        }
    }

    TEST(P5SessionChapterTest, NavigationVerticalIsOptionalAndPreservesUnknownValues) {
        auto session = make_populated_session_chapters();
        auto view = json_root(session.view.dom());
        view["navigation"].erase("up_axis");
        session.view = require_result(ViewSessionChapter::parse(view.dump()));
        EXPECT_TRUE(prepareGuiSessionRestore(session));
        view["navigation"]["up_axis"] = "data_z";
        session.view = require_result(ViewSessionChapter::parse(view.dump()));
        EXPECT_TRUE(prepareGuiSessionRestore(session));
        // An earlier writer knows mode/view_snap only. Its normal merge must
        // preserve the additive vertical field without changing chapter version.
        Json earlier_state = view;
        earlier_state["navigation"].erase("up_axis");
        require_status(session.view.merge_known_state(earlier_state));
        auto reopened = require_result(ViewSessionChapter::from_bytes(session.view.to_bytes()));
        EXPECT_EQ(json_root(reopened.dom())["navigation"]["up_axis"], "data_z");
        EXPECT_EQ(json_root(reopened.dom())["version"], view["version"]);
        for (const Json unknown : {Json("future_up"), Json(2), Json(nullptr), Json(true), Json::array(), Json::object()}) {
            view["navigation"]["up_axis"] = unknown;
            session.view = require_result(ViewSessionChapter::parse(view.dump()));
            EXPECT_TRUE(prepareGuiSessionRestore(session));
            EXPECT_EQ(json_root(session.view.dom())["navigation"]["up_axis"], unknown);
        }
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
         MissingPlySequenceOwnerIsDocumentedDegradedState) {
        const auto missing = lfs::core::generate_uuid_v4();

        auto document = ProjectDocument::create(
            lfs::core::generate_uuid_v4(), 100);
        ASSERT_TRUE(document);
        require_status(document->edit_sequencer().dom().set_json(
            "ply_sequences",
            Json::array({Json{
                {"node_name", "legacy clip"},
                {"node_uuid", missing.to_string()},
                {"fps", 24.0},
            }})));
        lfs::core::Scene scene;
        auto plan = document->stage_hydration(scene);
        ASSERT_TRUE(plan)
            << lfs::format_for_developer(plan.error());
        EXPECT_NE(
            std::ranges::find(
                document->degraded_states(),
                ProjectDocumentDegradedState::MissingPlySequenceNode),
            document->degraded_states().end());

        TemporaryDirectory temporary;
        const auto path =
            temporary.path / "dangling-seqr.licht";
        auto saved = document->save(
            path,
            ProjectDocumentSaveOptions{
                .disk_reserve_bytes = 0,
            });
        ASSERT_TRUE(saved)
            << lfs::format_for_developer(saved.error());
        auto reopened = ProjectDocument::open(path);
        ASSERT_TRUE(reopened)
            << lfs::format_for_developer(reopened.error());
        EXPECT_NE(
            std::ranges::find(
                reopened->degraded_states(),
                ProjectDocumentDegradedState::MissingPlySequenceNode),
            reopened->degraded_states().end());
    }

    TEST(P5SessionChapterTest,
         SequencerClipOwnerPresentInSceneGraphIsNotDegraded) {
        const auto owner = lfs::core::generate_uuid_v4();
        auto document = ProjectDocument::create(
            lfs::core::generate_uuid_v4(), 100);
        ASSERT_TRUE(document);
        require_status(document->edit_scene_graph().upsert_node(
            SceneNodeRecord{
                .uuid = owner,
                .type = "group",
                .name = "clip owner",
                .child_order = 0,
            }));
        require_status(document->edit_sequencer().dom().set_json(
            "ply_sequences",
            Json::array({Json{
                {"node_name", "owned clip"},
                {"node_uuid", owner.to_string()},
                {"fps", 24.0},
            }})));
        lfs::core::Scene scene;
        auto plan = document->stage_hydration(scene);
        ASSERT_TRUE(plan)
            << lfs::format_for_developer(plan.error());
        EXPECT_EQ(
            std::ranges::find(
                document->degraded_states(),
                ProjectDocumentDegradedState::MissingPlySequenceNode),
            document->degraded_states().end());
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
        EXPECT_EQ(PythonEditorSessionFile{}, PythonEditorSessionFile{});
        PythonEditor editor;
        EXPECT_FALSE(editor.document().id.is_nil());
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

    TEST(P5SessionChapterTest, EmptyShaderDraftRetainsIdentityWithoutMakingDefaultStateNonempty) {
        using namespace lfs::vis::editor;
        PythonEditor editor;
        editor.setDocumentKind(DocumentKind::Shader);
        const auto id = editor.document().id;
        const auto captured = editor.captureWorkspaceSessionState({}, false);
        ASSERT_EQ(captured.open_files.size(), 1u);
        EXPECT_EQ(captured.open_files.front().document.id, id);
        EXPECT_TRUE(captured.open_files.front().text.empty());
        editor.restoreWorkspaceSessionState(captured);
        EXPECT_EQ(editor.document().id, id);
        EXPECT_EQ(editor.document().kind, DocumentKind::Shader);
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

    TEST(P5SessionChapterTest, BuiltinEnvironmentsRoundTripWithoutExternalReferences) {
        for (const auto path : {lfs::vis::kDefaultEnvironmentMapPath,
                                std::string_view("environments/alps_field_1k.hdr")}) {
            SCOPED_TRACE(path);
            lfs::vis::RenderSettings settings;
            settings.environment_map_path = path;
            const auto json = renderSettingsToProjectJson(settings);
            EXPECT_EQ(json["environment_builtin"].get<std::string>(), path);
            EXPECT_TRUE(json["environment_reference_uuid"].is_null());
            const auto restored = renderSettingsFromProjectJson(json);
            ASSERT_TRUE(restored);
            EXPECT_EQ(restored->environment_map_path, path);
        }
    }

    TEST(P5SessionChapterTest, ExternalEnvironmentsKeepReferenceEncoding) {
        for (const auto path : {"", "custom.hdr", "/tmp/custom.hdr",
                                "environments/custom.hdr", "/tmp/environments/alps_field_1k.hdr"}) {
            SCOPED_TRACE(path);
            lfs::vis::RenderSettings settings;
            settings.environment_map_path = path;
            const auto json = renderSettingsToProjectJson(settings);
            EXPECT_TRUE(json["environment_builtin"].is_null());
            EXPECT_TRUE(json["environment_reference_uuid"].is_null());
        }
    }

    TEST(P5SessionChapterTest, BuiltinEnvironmentUsesExistingReaderContract) {
        auto json = renderSettingsToProjectJson(lfs::vis::RenderSettings{});
        json["environment_builtin"] = "environments/alps_field_1k.hdr";
        auto restored = renderSettingsFromProjectJson(json);
        ASSERT_TRUE(restored);
        EXPECT_EQ(restored->environment_map_path, "environments/alps_field_1k.hdr");
        json.erase("environment_builtin");
        restored = renderSettingsFromProjectJson(json);
        ASSERT_TRUE(restored);
        EXPECT_EQ(restored->environment_map_path, lfs::vis::kDefaultEnvironmentMapPath);
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
        lfs::vis::ViewSettings view_settings;
        auto expected = rolled_panel_camera(7.0f);
        expected.ortho_extent_world = static_cast<float>(viewport.windowSize.y) / *expected.ortho_scale;
        applyPanelCameraProjectState(
            viewport, view_settings, expected);
        EXPECT_EQ(
            capturePanelCameraProjectState(
                viewport, view_settings.ortho_scale),
            expected);
    }

    TEST(P5SessionChapterTest,
         EquirectangularSessionRoundTripKeepsSelectedBackendWithoutNewFields) {
        using Backend = lfs::rendering::GaussianRasterBackend;

        lfs::vis::RenderSettings settings;
        settings.equirectangular = true;
        settings.raster_backend = Backend::ThreeDgs;
        settings.gut = false;

        auto json = renderSettingsToProjectJson(settings);
        EXPECT_FALSE(json.contains("raster_backend_before_equirectangular"));
        EXPECT_EQ(json["raster_backend"], "3dgs");

        auto restored = renderSettingsFromProjectJson(json);
        ASSERT_TRUE(restored) << lfs::format_for_developer(restored.error());
        EXPECT_EQ(restored->raster_backend, Backend::ThreeDgs);
        EXPECT_FALSE(restored->gut);
        auto render_settings = *restored;
        lfs::vis::enforceProjectionBackend(render_settings);
        EXPECT_EQ(render_settings.raster_backend, Backend::ThreeDgut);
        restored->equirectangular = false;
        EXPECT_EQ(restored->raster_backend, Backend::ThreeDgs);
    }

    TEST(P5SessionChapterTest, GalleryOrthographicExtentSurvivesViewportSizes) {
        auto state = rolled_panel_camera(7.0f);
        state.ortho_extent_world = 6.25f;
        auto json = panelCameraProjectStateToJson("primary", state);
        auto restored = panelCameraProjectStateFromJson(json);
        ASSERT_TRUE(restored);
        EXPECT_EQ(restored->ortho_extent_world, state.ortho_extent_world);
        for (const int height : {480, 1080}) {
            Viewport viewport(1280, height);
            lfs::vis::ViewSettings settings;
            applyPanelCameraProjectState(viewport, settings, *restored);
            EXPECT_FLOAT_EQ(height / settings.ortho_scale, 6.25f);
            const auto saved = capturePanelCameraProjectState(viewport, settings.ortho_scale);
            ASSERT_TRUE(saved.ortho_extent_world);
            EXPECT_FLOAT_EQ(*saved.ortho_extent_world, 6.25f);
            Viewport reopened(1280, height * 2);
            applyPanelCameraProjectState(reopened, settings, saved);
            EXPECT_FLOAT_EQ(reopened.windowSize.y / settings.ortho_scale, 6.25f);
        }
        json["ortho_extent_world"] = -1;
        EXPECT_FALSE(panelCameraProjectStateFromJson(json));
        json.erase("ortho_extent_world");
        EXPECT_TRUE(panelCameraProjectStateFromJson(json));
    }

    TEST(P5SessionChapterTest, CaptureUsesViewSettingsOrthoScale) {
        Viewport viewport(1280, 720);
        const float fallback = 720.0f / 6.25f;
        auto captured = capturePanelCameraProjectState(viewport, fallback);
        EXPECT_EQ(captured.ortho_scale, fallback);
        ASSERT_TRUE(captured.ortho_extent_world.has_value());
        EXPECT_FLOAT_EQ(*captured.ortho_extent_world, 6.25f);

        Viewport empty(1280, 0);
        captured = capturePanelCameraProjectState(empty, fallback);
        EXPECT_FALSE(captured.ortho_extent_world.has_value());

        Viewport invalid(1280, 720);
        captured = capturePanelCameraProjectState(invalid, -1.0f);
        EXPECT_FALSE(captured.ortho_extent_world.has_value());
        captured = capturePanelCameraProjectState(invalid, std::nullopt);
        EXPECT_FALSE(captured.ortho_extent_world.has_value());
    }

    TEST(P5SessionChapterTest, DefaultViewKeepsBothPanelCamerasForEarlierReaders) {
        const auto cameras =
            lfs::io::project::default_session_chapter_dom(lfs::io::project::SessionJsonChapterKind::View)
                .get_json("panel_cameras");
        ASSERT_TRUE(cameras);
        ASSERT_EQ(cameras->size(), 2u);
        EXPECT_EQ((*cameras)[0]["panel"], "primary");
        EXPECT_EQ((*cameras)[1]["panel"], "secondary");
        for (const auto& camera : *cameras)
            EXPECT_TRUE(panelCameraProjectStateFromJson(camera));
    }

    TEST(P5SessionChapterTest, ViewRenderSettingsKeepEveryKeyEarlierReadersRequire) {
        // Every VIEW render-settings key earlier readers require; a project written without one
        // of them does not open there. Removed settings keep being written.
        static constexpr std::array<std::string_view, 76> earlier_reader_keys{
            "focal_length_mm",
            "scaling_modifier",
            "antialiasing",
            "mip_filter",
            "sh_degree",
            "render_scale",
            "camera_metrics_mode",
            "show_crop_box",
            "use_crop_box",
            "show_ellipsoid",
            "use_ellipsoid",
            "desaturate_unselected",
            "desaturate_cropping",
            "hide_outside_depth_box",
            "crop_filter_for_selection",
            "apply_appearance_correction",
            "ppisp_mode",
            "background_color",
            "environment_mode",
            "environment_exposure",
            "environment_rotation_degrees",
            "show_coord_axes",
            "axes_size",
            "show_grid",
            "grid_plane",
            "grid_opacity",
            "point_cloud_mode",
            "voxel_size",
            "show_rings",
            "ring_width",
            "show_center_markers",
            "show_camera_frustums",
            "camera_frustum_scale",
            "train_camera_color",
            "eval_camera_color",
            "show_pivot",
            "split_view_mode",
            "gt_comparison_mode",
            "split_position",
            "split_view_offset",
            "equirectangular",
            "orthographic",
            "ortho_scale",
            "depth_view",
            "depth_view_min",
            "depth_view_max",
            "depth_visualization_mode",
            "selection_color_committed",
            "selection_color_preview",
            "selection_color_center_marker",
            "depth_clip_enabled",
            "depth_clip_far",
            "mesh_wireframe",
            "mesh_wireframe_color",
            "mesh_wireframe_width",
            "mesh_light_dir",
            "mesh_light_intensity",
            "mesh_ambient",
            "mesh_backface_culling",
            "mesh_shadow_enabled",
            "mesh_shadow_resolution",
            "depth_filter_enabled",
            "depth_filter_min",
            "depth_filter_max",
            "lod_enabled",
            "lod_auto_enable_rad",
            "lod_max_splats",
            "lod_render_scale",
            "lod_behind_camera_penalty",
            "lod_cone_foveation",
            "lod_cone_inner_degrees",
            "lod_cone_outer_degrees",
            "lod_page_pool_splats",
            "lod_pool_vram_fraction",
            "lod_fade_frames",
            "lod_debug_colors",
        };
        const auto written = renderSettingsToProjectJson(lfs::vis::RenderSettings{});
        const auto default_view =
            lfs::io::project::default_session_chapter_dom(lfs::io::project::SessionJsonChapterKind::View)
                .get_json("render_settings");
        ASSERT_TRUE(default_view);
        for (const auto key : earlier_reader_keys) {
            EXPECT_TRUE(written.contains(std::string(key))) << key;
            EXPECT_TRUE(default_view->contains(std::string(key))) << key;
        }

        const auto& committed = written.at("selection_color_committed");
        ASSERT_TRUE(committed.is_array());
        ASSERT_EQ(committed.size(), 3u);
        EXPECT_FLOAT_EQ(committed[0].get<float>(), 0.859f);
        EXPECT_FLOAT_EQ(committed[1].get<float>(), 0.325f);
        EXPECT_FLOAT_EQ(committed[2].get<float>(), 0.325f);

        auto without_removed = written;
        without_removed.erase("selection_color_committed");
        EXPECT_TRUE(renderSettingsFromProjectJson(without_removed));
        EXPECT_TRUE(renderSettingsFromProjectJson(written));
    }

    TEST(P5SessionChapterTest,
         GalleryColorSettingsPersistAndOldProjectsKeepTheirAppearance) {
        lfs::vis::RenderSettings settings;
        settings.color_exposure = 2.5f;
        settings.color_tonemapping = 5;
        settings.splat_render_profile = 1;
        auto json = renderSettingsToProjectJson(settings);
        const auto restored = renderSettingsFromProjectJson(json);
        ASSERT_TRUE(restored);
        EXPECT_FLOAT_EQ(restored->color_exposure, 2.5f);
        EXPECT_EQ(restored->color_tonemapping, 5);
        EXPECT_EQ(restored->splat_render_profile, 1);

        // Projects written before gallery color controls have no such fields.
        // Loading them must preserve the existing untone-mapped display.
        json.erase("color_exposure");
        json.erase("color_tonemapping");
        json.erase("splat_render_profile");
        const auto legacy = renderSettingsFromProjectJson(json);
        ASSERT_TRUE(legacy);
        EXPECT_FLOAT_EQ(legacy->color_exposure, 1.0f);
        EXPECT_EQ(legacy->color_tonemapping, 0);
        EXPECT_EQ(legacy->splat_render_profile, 0);
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

        viewer.getGuiManager()->setSequencerVisible(true);
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
            viewer.getGuiManager()->isSequencerVisible());
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
        viewer.getGuiManager()->setSequencerVisible(true);

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
            viewer.getGuiManager()->isSequencerVisible());
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
        : public lfs::test::CudaBackendTest {
    protected:
        void SetUp() override {
            LFS_CUDA_BACKEND_OR_RETURN();
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
        ASSERT_TRUE(trainer->setParams(params));
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
        ASSERT_TRUE(trainer->setParams(params));
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
           ProjectCheckpointAcceptsSaveStepEdits) {
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
        ASSERT_TRUE(trainer->setParams(params));
        lfs::vis::TrainerManager manager;
        manager.setTrainerFromCheckpoint(
            std::move(trainer), 3);
        manager.setSaveSteps({5});
        EXPECT_EQ(manager.getSaveSteps(), (std::vector<size_t>{5}));
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
        expect_json(render, "/split_view_mode", 1);
        expect_json(render, "/gt_comparison_mode", 2);
        expect_json(render, "/raster_backend", "3dgut");
        EXPECT_FALSE(render.contains("gut"));
        expect_bool(render, "/orthographic", true);
        expect_bool(render, "/depth_view", true);
        prove("VIEW-198");
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
        EXPECT_FALSE(at(view, "/split").contains("panel_grid_planes"));
        prove("VIEW-203");
        ASSERT_EQ(at(view, "/panel_cameras").size(), 1u);
        expect_json(view, "/panel_cameras/0/R",
                    Json::array({0.0f, 1.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f}));
        EXPECT_TRUE(at(view, "/panel_cameras/0/ortho_scale").is_number());
        prove("VIEW-204");
        expect_json(view, "/navigation/mode", "drone");
        expect_bool(view, "/navigation/view_snap", true);
        prove("VIEW-205");
        EXPECT_FALSE(at(view, "/split").contains("focused_panel"));
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
        seq["preferences"]["reconstruction"] = {
            {"version", 1},
            {"backend_id", "com.example.reconstruction"},
            {"preset_id", "quality"},
            {"fallback", "native"},
        };
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
        EXPECT_EQ(ui.reconstruction.backend_id, "com.example.reconstruction");
        EXPECT_EQ(ui.reconstruction.preset_id, "quality");
        EXPECT_EQ(
            ui.reconstruction.fallback,
            lfs::io::video::VideoReconstructionFallback::Native);
        EXPECT_NEAR(viewer.getGuiManager()->sequencerUI().timelineZoom(), 2.5f, 1e-4f);
        EXPECT_NEAR(viewer.getGuiManager()->sequencerUI().timelinePan(), 1.25f, 1e-4f);
        if (const auto* rendering = viewer.getRenderingManager()) {
            EXPECT_EQ(ui.equirectangular, rendering->getSettings().equirectangular);
        }
    }

    TEST(P5SessionChapterTest, LegacySequencerPrefsResetVideoReconstructionToNative) {
        lfs::vis::ViewerOptions options;
        options.show_startup_overlay = false;
        lfs::vis::VisualizerImpl viewer(options);
        ASSERT_NE(viewer.getGuiManager(), nullptr);

        auto& ui = viewer.getGuiManager()->getSequencerUIState();
        ui.reconstruction = {
            .backend_id = "com.example.previous",
            .preset_id = "quality",
            .fallback = lfs::io::video::VideoReconstructionFallback::Native,
        };

        auto prepared = require_result(
            prepareGuiSessionRestore(make_populated_session_chapters()));
        std::vector<CameraBookmarkProjectState> bookmarks;
        applyGuiSession(viewer, prepared, bookmarks);

        EXPECT_EQ(ui.reconstruction, lfs::io::video::VideoReconstructionSelection{});
    }

    TEST(P5SessionChapterTest, FutureVideoReconstructionVersionOpensWithNativeWarningAndResaves) {
        lfs::vis::ViewerOptions options;
        options.show_startup_overlay = false;
        lfs::vis::VisualizerImpl viewer(options);
        lfs::vis::P5SessionCaptureTestAccess::initializeInputController(viewer);
        ASSERT_NE(viewer.getGuiManager(), nullptr);
        ASSERT_NE(viewer.getRenderingManager(), nullptr);
        ASSERT_NE(viewer.getInputController(), nullptr);
        auto& ui = viewer.getGuiManager()->getSequencerUIState();
        ui.reconstruction = {.backend_id = "previous", .preset_id = "quality"};

        struct WarningConsumer final : lfs::NativeErrorConsumer {
            int warnings = 0;
            void on_error(const lfs::ErrorNotification& notification, const lfs::ErrorDeliveryInfo&) noexcept override {
                if (notification.error.code() == lfs::ErrorCode::Unsupported &&
                    notification.error.severity() == lfs::Severity::Warning &&
                    notification.surface == lfs::ErrorSurface::Toast)
                    ++warnings;
            }
        } consumer;
        auto subscription = lfs::ErrorBus::instance().subscribe(consumer);
        auto session = make_populated_session_chapters();
        auto seq = json_root(session.sequencer.dom());
        seq["preferences"]["reconstruction"] = {
            {"version", 2},
            {"future_schema", true},
        };
        seq["preferences"]["quality"] = 22;
        seq["preferences"]["vendor_other_preference"] = "preserve";
        session.sequencer = require_result(
            SequencerSessionChapter::parse(seq.dump()));
        const auto prepared = require_result(prepareGuiSessionRestore(session));
        EXPECT_EQ(consumer.warnings, 0); // Validation alone must not publish UI notifications.
        std::vector<CameraBookmarkProjectState> bookmarks;
        applyGuiSession(viewer, prepared, bookmarks);
        EXPECT_EQ(ui.reconstruction, lfs::io::video::VideoReconstructionSelection{});
        EXPECT_EQ(ui.quality, 22);
        EXPECT_EQ(consumer.warnings, 1);

        const auto saved = require_result(captureGuiSession(viewer, session, bookmarks));
        const auto canonical = lfs::io::video::serializeVideoReconstructionSelection({});
        ASSERT_TRUE(canonical);
        EXPECT_EQ(json_root(saved.sequencer.dom())["preferences"]["reconstruction"], Json::parse(*canonical));
        EXPECT_EQ(json_root(saved.sequencer.dom())["preferences"]["vendor_other_preference"], "preserve");
        EXPECT_TRUE(prepareGuiSessionRestore(saved));
        EXPECT_EQ(consumer.warnings, 1); // Re-saving does not repeat the warning.
    }

    TEST(P5SessionChapterTest, CorruptVideoReconstructionSelectionStillFailsBeforeApply) {
        for (const auto& reconstruction : std::vector<Json>{
                 Json::array(),
                 Json::object(),
                 {{"version", 0}},
                 {{"version", -1}},
                 {{"version", "2"}},
                 {{"version", 2.0}},
                 {{"version", true}},
                 {{"version", 1}, {"backend_id", "native"}, {"preset_id", "native"}, {"fallback", "unknown"}},
                 {{"version", 1}, {"backend_id", "native"}, {"preset_id", "quality"}, {"fallback", "abort"}},
                 {{"version", 1}, {"backend_id", "INVALID"}, {"preset_id", "quality"}, {"fallback", "native"}}}) {
            SCOPED_TRACE(reconstruction.dump());
            auto session = make_populated_session_chapters();
            require_status(session.sequencer.dom().set_json("preferences.reconstruction", reconstruction));
            const auto prepared = prepareGuiSessionRestore(std::move(session));
            ASSERT_FALSE(prepared);
            EXPECT_EQ(prepared.error().code(), lfs::ErrorCode::DataLoss);
        }
    }

    TEST(P5SessionChapterTest, CaptureValidatesAndUsesCanonicalVideoReconstructionSerializer) {
        lfs::vis::ViewerOptions options;
        options.show_startup_overlay = false;
        lfs::vis::VisualizerImpl viewer(options);
        lfs::vis::P5SessionCaptureTestAccess::initializeInputController(viewer);
        ASSERT_NE(viewer.getGuiManager(), nullptr);
        ASSERT_NE(viewer.getRenderingManager(), nullptr);
        ASSERT_NE(viewer.getInputController(), nullptr);
        auto& ui = viewer.getGuiManager()->getSequencerUIState();
        auto session = make_populated_session_chapters();
        ui.reconstruction.fallback = static_cast<lfs::io::video::VideoReconstructionFallback>(255);
        const auto invalid = captureGuiSession(viewer, session, {});
        ASSERT_FALSE(invalid);
        EXPECT_EQ(invalid.error().code(), lfs::ErrorCode::InvalidArgument);

        ui.reconstruction = {.backend_id = "example", .preset_id = "quality", .fallback = lfs::io::video::VideoReconstructionFallback::Native};
        const auto saved = require_result(captureGuiSession(viewer, session, {}));
        const auto canonical = lfs::io::video::serializeVideoReconstructionSelection(ui.reconstruction);
        ASSERT_TRUE(canonical);
        EXPECT_EQ(json_root(saved.sequencer.dom())["preferences"]["reconstruction"], Json::parse(*canonical));
        EXPECT_TRUE(prepareGuiSessionRestore(saved));

        auto retained_selection = Json::parse(*canonical);
        retained_selection["vendor_extra"] = {{"keep", true}};
        require_status(session.sequencer.dom().set_json("preferences.reconstruction", retained_selection));
        ui.reconstruction = {};
        const auto resaved = require_result(captureGuiSession(viewer, session, {}));
        const auto resaved_selection = json_root(resaved.sequencer.dom())["preferences"]["reconstruction"];
        EXPECT_EQ(resaved_selection["backend_id"], "native");
        EXPECT_EQ(resaved_selection["preset_id"], "native");
        EXPECT_EQ(resaved_selection["fallback"], "abort");
        EXPECT_EQ(resaved_selection["vendor_extra"], retained_selection["vendor_extra"]);
    }

} // namespace
namespace {

    class UnknownSessionValueFixture {
    public:
        struct TestPanel final : lfs::vis::gui::IPanel {
            void draw(const lfs::vis::gui::PanelDrawContext&) override {}
            lfs::vis::gui::PanelRenderCapabilities renderCapabilities() const override { return {.direct = true}; }
        };

        UnknownSessionValueFixture() {
            lfs::event::EventBridge::instance().clear_all();
            lfs::vis::ViewerOptions options;
            options.show_startup_overlay = false;
            viewer = std::make_unique<lfs::vis::VisualizerImpl>(options);
            lfs::vis::P5SessionCaptureTestAccess::initializeInputController(*viewer);
            if (seed_splat_selection(*viewer) == lfs::core::NULL_NODE ||
                !lfs::vis::gui::PanelRegistry::instance().register_panel({
                    .panel = std::make_shared<TestPanel>(),
                    .label = "Session test panel",
                    .id = panel_id,
                    .space = lfs::vis::gui::PanelSpace::SidePanel,
                }))
                throw std::runtime_error("Failed to initialize unknown-session test fixture");
        }

        ~UnknownSessionValueFixture() {
            lfs::vis::gui::PanelRegistry::instance().unregister_panel(panel_id);
        }

        PreparedGuiSessionRestore restore(
            const std::string& name,
            const std::function<void(Json&, Json&, Json&)>& mutate = {}) {
            auto document = lfs::test::licht::make_empty_document(lfs::core::generate_uuid_v4());
            auto view = json_root(document->view().dom());
            auto gui = json_root(document->gui_layout().dom());
            auto sequencer = json_root(document->sequencer().dom());
            if (mutate)
                mutate(view, gui, sequencer);
            document->edit_view() = require_result(ViewSessionChapter::parse(view.dump()));
            document->edit_gui_layout() = require_result(GuiLayoutChapter::parse(gui.dump()));
            document->edit_sequencer() = require_result(SequencerSessionChapter::parse(sequencer.dump()));
            const auto path = temporary.path / (name + ".licht");
            (void)require_result(document->save(path));
            auto reopened = require_result(ProjectDocument::open(path));
            ProjectSessionChapters chapters{
                .gui_layout = reopened.gui_layout(),
                .editor = reopened.editor(),
                .view = reopened.view(),
                .sequencer = reopened.sequencer(),
                .metrics = reopened.metrics(),
            };
            auto prepared = require_result(prepareGuiSessionRestore(std::move(chapters)));
            applyGuiSession(*viewer, prepared, bookmarks);
            applyGuiSessionTools(*viewer, prepared);
            return prepared;
        }

        const std::string panel_id = "test.session.unknown-space";
        TemporaryDirectory temporary;
        std::unique_ptr<lfs::vis::VisualizerImpl> viewer;
        std::vector<CameraBookmarkProjectState> bookmarks;
    };

    TEST(P5SessionChapterTest, UnknownProjectValuesUseRuntimeDefaults) {
        UnknownSessionValueFixture test;
        auto prepared = test.restore("fallbacks", [](Json& view, Json&, Json& sequencer) {
            view["split"]["focused_panel"] = "future_panel";
            view["navigation"]["mode"] = "future_navigation";
            view["navigation"]["up_axis"] = "future_axis";
            view["render_settings"]["raster_backend"] = "future_backend";
            view["render_settings"]["camera_metrics_mode"] = 99;
            view["tools"]["gizmo_operation"] = "future_operation";
            sequencer["loop_mode"] = "future_loop";
            sequencer["preferences"]["preset"] = "future_preset";
        });
        EXPECT_EQ(test.viewer->getRenderingManager()->getSettings().split_view_mode,
                  lfs::vis::SplitViewMode::Disabled);
        EXPECT_EQ(test.viewer->getInputController()->cameraNavigationMode(),
                  lfs::vis::InputController::CameraNavigationMode::Orbit);
        EXPECT_EQ(test.viewer->getInputController()->navigationUpAxis(), lfs::vis::NavigationUpAxis::LegacyY);
        EXPECT_EQ(test.viewer->getRenderingManager()->getSettings().raster_backend,
                  lfs::vis::RenderSettings{}.raster_backend);
        EXPECT_EQ(test.viewer->getGuiManager()->gizmo().getOperation(), lfs::vis::gui::GizmoOperation::Translate);
        EXPECT_EQ(test.viewer->getGuiManager()->sequencer().loopMode(), lfs::vis::LoopMode::ONCE);
        EXPECT_EQ(test.viewer->getGuiManager()->getSequencerUIState().preset, lfs::io::video::VideoPreset::CUSTOM);
        EXPECT_EQ(prepared.unsupported_settings.size(), 8u);
    }

    TEST(P5SessionChapterTest, PolygonCropShapeRestoresAndIsCapturedAgain) {
        UnknownSessionValueFixture test;
        auto prepared = test.restore("crop-polygon", [](Json& view, Json&, Json&) {
            view["tools"]["crop_shape"] = "polygon";
        });
        EXPECT_TRUE(prepared.unsupported_settings.empty());
        EXPECT_EQ(test.viewer->getGuiManager()->gizmo().cropToolShape(), "polygon");
        auto captured = require_result(captureGuiSession(*test.viewer, ProjectSessionChapters{}, test.bookmarks));
        EXPECT_EQ(json_root(captured.view.dom())["tools"]["crop_shape"], "polygon");
    }

    TEST(P5SessionChapterTest, UnknownNavigationAxisWarnsAndUsesLegacyY) {
        UnknownSessionValueFixture test;
        auto prepared = test.restore("axis", [](Json& view, Json&, Json&) {
            view["navigation"]["up_axis"] = "future_axis";
        });
        EXPECT_EQ(test.viewer->getInputController()->navigationUpAxis(), lfs::vis::NavigationUpAxis::LegacyY);
        ASSERT_EQ(prepared.unsupported_settings.size(), 1u);
        EXPECT_EQ(prepared.unsupported_settings.front().value, "future_axis");
        EXPECT_EQ(prepared.unsupported_settings.front().fallback, "legacy_y");
    }

    TEST(P5SessionChapterTest, NoActiveToolIsNotReportedAsANewerSetting) {
        UnknownSessionValueFixture test;
        auto prepared = test.restore("no-tool", [](Json& view, Json&, Json&) {
            view["tools"]["active_tool_id"] = "";
        });
        EXPECT_TRUE(prepared.unsupported_settings.empty());
    }

    TEST(P5SessionChapterTest, UnknownSelectionSubmodesResetIndependently) {
        UnknownSessionValueFixture test;
        auto prepared = test.restore("submodes", [](Json& view, Json&, Json&) {
            view["tools"]["selection_submode"] = "future_selection";
            view["tools"]["active_submode_id"] = "future_active";
        });
        EXPECT_EQ(test.viewer->getGuiManager()->gizmo().getSelectionSubMode(), lfs::vis::SelectionSubMode::Centers);
        // The active submode belongs to whichever tool was active, so only the selection mode is checked.
        ASSERT_EQ(prepared.unsupported_settings.size(), 1u);
        EXPECT_EQ(prepared.unsupported_settings[0].value, "future_selection");
    }

    TEST(P5SessionChapterTest, SavedActiveSubmodesRestoreWithoutNewerSettingWarning) {
        UnknownSessionValueFixture test;
        (void)test.restore("active-submodes");
        for (const char* value : {"", "centers", "rectangle", "polygon", "lasso", "rings", "color", "box", "sphere"}) {
            lfs::vis::UnifiedToolRegistry::instance().setActiveSubmode(value);
            auto captured = require_result(captureGuiSession(*test.viewer, ProjectSessionChapters{}, test.bookmarks));
            EXPECT_EQ(json_root(captured.view.dom())["tools"]["active_submode_id"], value);
            const auto prepared = require_result(prepareGuiSessionRestore(std::move(captured)));
            EXPECT_TRUE(prepared.unsupported_settings.empty()) << "active_submode_id=\"" << value << '"';
        }
    }

    TEST(P5SessionChapterTest, DeprecatedScenePanelTabIsIgnoredAndNoLongerWritten) {
        UnknownSessionValueFixture test;
        for (const char* value : {"history", "logging", "future_tab"}) {
            auto prepared = test.restore(value, [&](Json&, Json& gui, Json&) {
                gui["layouts"][0]["areas"][0]["spaces"][1]["opaque_payload"]["active_tabs"]["scene_panel"] = value;
            });
            EXPECT_TRUE(prepared.unsupported_settings.empty()) << value;
        }
        auto captured = require_result(captureGuiSession(*test.viewer, ProjectSessionChapters{}, test.bookmarks));
        const auto gui = json_root(captured.gui_layout.dom());
        EXPECT_FALSE(gui["layouts"][0]["areas"][0]["spaces"][1]["opaque_payload"]["active_tabs"].contains("scene_panel"));
    }

    // Fails if a submode this version saves, from any tool or none, reads back as a newer-version setting.
    TEST(P5SessionChapterTest, SavedToolSubmodesAreNotReportedAsNewerSettings) {
        UnknownSessionValueFixture test;
        for (const char* value : {"", "local", "x", "plugin.paint"}) {
            auto prepared = test.restore(std::string("submode_") + value, [&](Json& view, Json&, Json&) {
                view["tools"]["active_submode_id"] = value;
            });
            EXPECT_TRUE(prepared.unsupported_settings.empty()) << value;
        }
    }

    // Fails if a plugin's tool or tab, missing only because the plugin is not loaded,
    // reads back as a newer-version setting.
    TEST(P5SessionChapterTest, ItemsFromUnloadedPluginsAreNotReportedAsNewerSettings) {
        UnknownSessionValueFixture test;
        const auto restore = [&](const char* name, const char* tool, const char* tab) {
            return test.restore(name, [&](Json& view, Json& gui, Json&) {
                view["tools"]["active_tool_id"] = tool;
                gui["layouts"][0]["areas"][0]["spaces"][1]["opaque_payload"]["active_tabs"]["main_panel"] = tab;
            });
        };
        EXPECT_TRUE(restore("plugin", "camera_selector.pick", "lfs_plugins.camera_selector.Panel")
                        .unsupported_settings.empty());
        EXPECT_EQ(restore("builtin", "builtin.future_tool", "lfs.future_panel").unsupported_settings.size(), 2u);
    }

    TEST(P5SessionChapterTest, UnknownValuesSurviveCaptureAndResave) {
        UnknownSessionValueFixture test;
        auto prepared = test.restore("preserve", [&](Json& view, Json& gui, Json& sequencer) {
            view["navigation"]["up_axis"] = "future_axis";
            view["panel_cameras"][0]["panel"] = "future_camera_role";
            view["render_settings"]["camera_metrics_mode"] = 99;
            auto& payload = gui["layouts"][0]["areas"][0]["spaces"][1]["opaque_payload"];
            payload["panels"] = Json::array({{{"id", test.panel_id}, {"space", "future_space"}}});
            payload["active_tabs"]["main_panel"] = "future_main_panel";
            sequencer["preferences"]["preset"] = "future_preset";
        });
        auto captured = require_result(captureGuiSession(*test.viewer, prepared.chapters, test.bookmarks));
        auto document = require_result(ProjectDocument::open(test.temporary.path / "preserve.licht"));
        document.edit_view() = captured.view;
        document.edit_gui_layout() = captured.gui_layout;
        document.edit_sequencer() = captured.sequencer;
        (void)require_result(document.save(test.temporary.path / "preserve.licht"));
        auto reopened = require_result(ProjectDocument::open(test.temporary.path / "preserve.licht"));
        const auto view = json_root(reopened.view().dom());
        const auto gui = json_root(reopened.gui_layout().dom());
        const auto sequencer = json_root(reopened.sequencer().dom());
        EXPECT_EQ(view["navigation"]["up_axis"], "future_axis");
        EXPECT_EQ(view["panel_cameras"][0]["panel"], "future_camera_role");
        EXPECT_EQ(view["render_settings"]["camera_metrics_mode"], 99);
        EXPECT_EQ(gui["layouts"][0]["areas"][0]["spaces"][1]["opaque_payload"]["panels"][0]["space"],
                  "future_space");
        EXPECT_EQ(gui["layouts"][0]["areas"][0]["spaces"][1]["opaque_payload"]["active_tabs"]["main_panel"],
                  "future_main_panel");
        EXPECT_EQ(sequencer["preferences"]["preset"], "future_preset");
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

    Json find_space_payload_for_test(const Json& root, const std::string_view type) {
        for (const auto& layout : root.at("layouts")) {
            for (const auto& area : layout.at("areas")) {
                for (const auto& space : area.at("spaces")) {
                    if (space.value("type", std::string{}) == type)
                        return space.at("opaque_payload");
                }
            }
        }
        return Json::object();
    }

    TEST(P5SessionChapterTest, MultiAreaScreenRoundTripsCamerasAndViewSettings) {
        lfs::vis::ViewerOptions options;
        options.show_startup_overlay = false;
        lfs::vis::VisualizerImpl viewer(options);
        lfs::vis::P5SessionCaptureTestAccess::initializeInputController(viewer);
        ASSERT_NE(viewer.getGuiManager(), nullptr);
        ASSERT_NE(viewer.getRenderingManager(), nullptr);

        lfs::vis::screen::AreaId view;
        lfs::vis::screen::AreaId other;
        viewer.screens().edit([&](lfs::vis::screen::Screen& screen) {
            view = screen.activeView();
            other = screen.split(view, lfs::vis::screen::SplitAxis::Columns, 0.5f);
        });
        ASSERT_TRUE(view.valid());
        ASSERT_TRUE(other.valid());
        auto* first = viewer.screens().view3D(view);
        auto* second = viewer.screens().view3D(other);
        ASSERT_NE(first, nullptr);
        ASSERT_NE(second, nullptr);
        first->camera.setViewMatrix(glm::mat3(1.0f), glm::vec3(1.0f, 2.0f, 3.0f));
        first->camera.camera.pivot = glm::vec3(0.25f, 0.5f, 0.75f);
        first->settings.show_grid = false;
        first->settings.orthographic = true;
        first->settings.focal_length_mm = 85.0f;
        second->camera.setViewMatrix(glm::mat3(1.0f), glm::vec3(9.0f, 8.0f, 7.0f));
        second->camera.camera.pivot = glm::vec3(4.0f, 5.0f, 6.0f);
        second->settings.point_cloud_mode = true;
        second->settings.show_camera_frustums = true;
        second->settings.focal_length_mm = 24.0f;

        auto& properties = viewer.getGuiManager()->screenHost().properties();
        properties.setActiveTab("lfs.training");
        properties.setScroll(0.375f);

        auto session = make_populated_session_chapters();
        const auto captured = require_result(captureGuiSession(viewer, session, {}));
        const Json gui = json_root(captured.gui_layout.dom());
        const Json screen_json = find_space_payload_for_test(gui, "screen");
        ASSERT_TRUE(screen_json.contains("layout"));
        ASSERT_TRUE(screen_json.contains("areas"));
        const Json fixed = find_space_payload_for_test(gui, "fixed_arrangement");
        EXPECT_TRUE(fixed.contains("right_panel_width"));
        EXPECT_TRUE(fixed.contains("sequencer_visible"));
        EXPECT_EQ(fixed["scene_tree"], screen_json["scene_tree"]);
        EXPECT_FALSE(find_space_payload_for_test(gui, "panel_registry").empty());
        EXPECT_FALSE(find_space_payload_for_test(gui, "python_console").empty());

        auto screen_only = captured;
        Json screen_only_gui = gui;
        auto& spaces = screen_only_gui["layouts"][0]["areas"][0]["spaces"];
        for (auto it = spaces.begin(); it != spaces.end();) {
            if (it->value("type", std::string{}) == "fixed_arrangement")
                it = spaces.erase(it);
            else
                ++it;
        }
        screen_only.gui_layout = require_result(GuiLayoutChapter::parse(screen_only_gui.dump()));
        const auto recaptured = require_result(captureGuiSession(viewer, screen_only, {}));
        EXPECT_FALSE(find_space_payload_for_test(json_root(recaptured.gui_layout.dom()), "fixed_arrangement").empty());
        EXPECT_EQ(screen_json["properties"]["active_tab"], "lfs.training");
        EXPECT_FLOAT_EQ(screen_json["properties"]["scroll"].get<float>(), 0.375f);
        int view_count = 0;
        for (const auto& area : screen_json["areas"]) {
            if (area.value("editor", std::string{}) == "view3d")
                ++view_count;
        }
        EXPECT_EQ(view_count, 2);

        const Json view_json = json_root(captured.view.dom());
        ASSERT_TRUE(view_json.contains("panel_cameras"));
        ASSERT_EQ(view_json["panel_cameras"].size(), 2u);
        EXPECT_EQ(view_json["panel_cameras"][0]["panel"], "primary");
        EXPECT_EQ(view_json["panel_cameras"][1]["panel"], "secondary");
        EXPECT_EQ(view_json["panel_cameras"][0]["t"], Json::array({1.0f, 2.0f, 3.0f}));
        EXPECT_EQ(view_json["panel_cameras"][1]["t"], Json::array({9.0f, 8.0f, 7.0f}));
        EXPECT_TRUE(view_json["render_settings"].contains("focal_length_mm"));
        EXPECT_TRUE(view_json["render_settings"].contains("antialiasing"));

        viewer.screens().resetToDefault();
        properties.setActiveTab("lfs.preferences");
        properties.setScroll(0.0f);
        auto prepared = require_result(prepareGuiSessionRestore(captured));
        std::vector<CameraBookmarkProjectState> bookmarks;
        applyGuiSession(viewer, prepared, bookmarks);

        EXPECT_EQ(properties.activeTab(), "lfs.training");
        EXPECT_FLOAT_EQ(properties.scroll(), 0.375f);

        auto* restored_first = viewer.screens().view3D(view);
        auto* restored_second = viewer.screens().view3D(other);
        ASSERT_NE(restored_first, nullptr);
        ASSERT_NE(restored_second, nullptr);
        EXPECT_EQ(restored_first->camera.camera.t, glm::vec3(1.0f, 2.0f, 3.0f));
        EXPECT_EQ(restored_first->camera.camera.pivot, glm::vec3(0.25f, 0.5f, 0.75f));
        EXPECT_FALSE(restored_first->settings.show_grid);
        EXPECT_TRUE(restored_first->settings.orthographic);
        EXPECT_FLOAT_EQ(restored_first->settings.focal_length_mm, 85.0f);
        EXPECT_EQ(restored_second->camera.camera.t, glm::vec3(9.0f, 8.0f, 7.0f));
        EXPECT_EQ(restored_second->camera.camera.pivot, glm::vec3(4.0f, 5.0f, 6.0f));
        EXPECT_TRUE(restored_second->settings.point_cloud_mode);
        EXPECT_TRUE(restored_second->settings.show_camera_frustums);
        EXPECT_FLOAT_EQ(restored_second->settings.focal_length_mm, 24.0f);
        EXPECT_EQ(viewer.screens().screen().views().size(), 2u);
    }

    TEST(P5SessionChapterTest, ResetLayoutRestoresDefaultScreenAndMarksRenderingDirty) {
        TemporaryDirectory temporary{"licht-screen-reset"};
        const ScopedLfsHome home(temporary.path);
        lfs::vis::ViewerOptions options;
        options.show_startup_overlay = false;
        lfs::vis::VisualizerImpl viewer(options);
        auto* gui = viewer.getGuiManager();
        auto* rendering = viewer.getRenderingManager();
        ASSERT_NE(gui, nullptr);
        ASSERT_NE(rendering, nullptr);

        const auto initial_area_count = viewer.screens().screen().areas().size();
        const auto active = viewer.screens().screen().activeView();
        ASSERT_TRUE(viewer.screens().screen().split(active, lfs::vis::screen::SplitAxis::Columns, 0.5f).valid());
        ASSERT_GT(viewer.screens().screen().areas().size(), initial_area_count);
        (void)rendering->pollDirtyState();

        const auto reset = gui->resetLayout();
        ASSERT_TRUE(reset) << reset.error();
        EXPECT_EQ(viewer.screens().screen().areas().size(), initial_area_count);
        EXPECT_NE(rendering->pendingDirtyMask() & lfs::vis::DirtyFlag::ALL, 0u);
    }

    TEST(P5SessionChapterTest, OldFormatProjectOpensWithDefaultScreenAndLegacyView) {
        lfs::vis::ViewerOptions options;
        options.show_startup_overlay = false;
        lfs::vis::VisualizerImpl viewer(options);
        lfs::vis::P5SessionCaptureTestAccess::initializeInputController(viewer);
        ASSERT_NE(viewer.getGuiManager(), nullptr);
        ASSERT_NE(viewer.getRenderingManager(), nullptr);

        auto prepared = require_result(
            prepareGuiSessionRestore(make_populated_session_chapters()));
        std::vector<CameraBookmarkProjectState> bookmarks;
        applyGuiSession(viewer, prepared, bookmarks);

        const auto& screen = viewer.screens().screen();
        EXPECT_EQ(screen.areas().size(), 3u);
        EXPECT_TRUE(screen.findEditor(lfs::vis::screen::editors::kView3D).valid());
        EXPECT_TRUE(screen.findEditor(lfs::vis::screen::editors::kScene).valid());
        EXPECT_TRUE(screen.findEditor(lfs::vis::screen::editors::kProperties).valid());
        auto* view = viewer.screens().view3D(screen.activeView().value);
        ASSERT_NE(view, nullptr);
        const auto expected = rolled_panel_camera(1.0f);
        EXPECT_EQ(view->camera.camera.t, glm::vec3(expected.translation[0],
                                                   expected.translation[1],
                                                   expected.translation[2]));
        EXPECT_FLOAT_EQ(view->settings.focal_length_mm, 73.0f);
        EXPECT_TRUE(view->settings.equirectangular);
        EXPECT_FALSE(view->settings.orthographic);
        EXPECT_TRUE(view->settings.point_cloud_mode);
        EXPECT_TRUE(view->settings.depth_view);
        EXPECT_TRUE(view->settings.show_camera_frustums);
        EXPECT_EQ(view->settings.grid_plane, 2);
        EXPECT_TRUE(viewer.getRenderingManager()->getSettings().antialiasing);
        EXPECT_EQ(viewer.getRenderingManager()->getSettings().sh_degree, 2);
    }

    TEST(P5SessionChapterTest, MalformedScreenJsonIsRejectedWithoutPartialState) {
        lfs::vis::ViewerOptions options;
        options.show_startup_overlay = false;
        lfs::vis::VisualizerImpl viewer(options);
        lfs::vis::P5SessionCaptureTestAccess::initializeInputController(viewer);
        const auto before = viewer.screens().read(
            [](const lfs::vis::screen::Screen& screen) { return screen.save(); });
        const auto before_generation = viewer.screens().screen().generation();

        auto session = make_populated_session_chapters();
        auto gui = json_root(session.gui_layout.dom());
        gui["layouts"][0]["areas"][0]["spaces"][0] = Json{
            {"type", "screen"},
            {"version", 1},
            {"opaque_payload",
             Json{
                 {"version", 1},
                 {"layout", {{"area", 1}}},
                 {"areas", Json::array()},
             }},
        };
        session.gui_layout = require_result(GuiLayoutChapter::parse(gui.dump()));
        const auto prepared = prepareGuiSessionRestore(std::move(session));
        ASSERT_FALSE(prepared);
        EXPECT_EQ(prepared.error().code(), lfs::ErrorCode::DataLoss);

        EXPECT_EQ(viewer.screens().screen().generation(), before_generation);
        const auto after = viewer.screens().read(
            [](const lfs::vis::screen::Screen& screen) { return screen.save(); });
        EXPECT_EQ(after, before);
    }

    // A transient preview has no payload binding and is never saved, so a
    // training snapshot leaves it out instead of failing on it.
    TEST(P5SessionChapterTest, SnapshotLeavesTransientNodesOut) {
        lfs::core::param::TrainingParameters params;
        params.dataset.data_path = "data/bicycle";
        params.optimization.strategy = std::string(lfs::core::param::kStrategyMCMC);
        lfs::core::Scene scene;
        const auto training = scene.addSplatPlaceholder("Train");
        ASSERT_NE(training, lfs::core::NULL_NODE);
        scene.setTrainingModelNode(scene.getNodeUuid(training));
        ASSERT_NE(scene.addSplatPlaceholder("Preview"), lfs::core::NULL_NODE);
        scene.getMutableNode("Preview")->transient = true;
        lfs::training::ProjectSnapshotCpuState cpu;
        auto captured = lfs::training::capture_project_snapshot_cpu_state(
            scene, params, lfs::core::generate_uuid_v4(), 1, cpu);
        ASSERT_TRUE(captured) << lfs::format_for_developer(captured.error());
        EXPECT_TRUE(std::ranges::any_of(cpu.scene_graph.nodes, [](const auto& node) { return node.name == "Train"; }));
        EXPECT_TRUE(std::ranges::none_of(cpu.scene_graph.nodes, [](const auto& node) { return node.name == "Preview"; }));
    }

} // namespace
