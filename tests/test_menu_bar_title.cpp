/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/event_bridge/localization_manager.hpp"
#include "core/path_utils.hpp"
#include "gui/rml_menu_bar.hpp"
#include "input/input_bindings.hpp"
#include "python/python_runtime.hpp"
#include "visualizer/app_store.hpp"
#include "visualizer/visualizer.hpp"

#include <RmlUi/Core.h>
#include <RmlUi/Core/ElementDocument.h>
#include <RmlUi/Core/ElementText.h>
#include <RmlUi/Core/Factory.h>
#include <RmlUi/Core/RenderInterface.h>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <sstream>
#include <thread>

namespace lfs::vis::gui {
    // Attach a CPU Rml context instead of creating the Vulkan-backed GUI manager.
    // Model registration, resources, layout, updates and hit tests are production code.
    class RmlMenuBarTestAccess {
    public:
        static void bind(RmlMenuBar& bar, Rml::Context* context) {
            bar.rml_context_ = context;
            bar.bindModel();
            bar.camera_buttons_.resize(3);
            bar.render_buttons_.resize(3);
            bar.projection_buttons_.resize(2);
            for (const auto* name : {"menu_camera_buttons", "menu_render_buttons", "menu_projection_buttons"})
                bar.menu_model_.DirtyVariable(name);
        }
        static void attach(RmlMenuBar& bar, Rml::ElementDocument* doc, RmlUIManager& manager) {
            bar.rml_manager_ = &manager;
            bar.document_ = doc;
            bar.menu_items_ = doc->GetElementById("menu-items");
            bar.menu_toolbar_ = doc->GetElementById("menu-toolbar");
            bar.menu_window_controls_ = doc->GetElementById("menu-window-controls");
            bar.project_title_container_ = doc->GetElementById("project-title");
            bar.project_title_el_ = doc->GetElementById("project-title-content");
        }
        static RmlTooltipController& tooltip(RmlMenuBar& bar) { return bar.tooltip_; }
        static void rebuildPortalStatus(RmlMenuBar& bar) { bar.rebuildPortalStatus(); }
        static void layout(RmlMenuBar& bar, int width, float dp) {
            bar.updateProjectTitleLayout(width, dp);
        }
        static bool hit(const RmlMenuBar& bar, float x, float y) {
            return bar.projectTitleAtPoint(x, y);
        }
        static void toolbar(RmlMenuBar& bar, bool visible, float right) {
            bar.toolbar_fits_ = visible;
            bar.applied_toolbar_right_ = right;
            bar.menu_toolbar_->SetClass("hidden", !visible);
            bar.menu_toolbar_->SetProperty("right", std::to_string(right) + "px");
        }
    };
} // namespace lfs::vis::gui

namespace {
    using lfs::vis::ProjectDisplayInfo;
    using lfs::vis::gui::resolveRmlTooltip;
    using lfs::vis::gui::RmlMenuBarTestAccess;

    class TitleRenderInterface final : public Rml::RenderInterface {
    public:
        Rml::CompiledGeometryHandle CompileGeometry(Rml::Span<const Rml::Vertex>, Rml::Span<const int>) override { return 1; }
        void RenderGeometry(Rml::CompiledGeometryHandle, Rml::Vector2f, Rml::TextureHandle) override {}
        void ReleaseGeometry(Rml::CompiledGeometryHandle) override {}
        Rml::TextureHandle LoadTexture(Rml::Vector2i& size, const Rml::String&) override {
            size = {16, 16};
            return 1;
        }
        Rml::TextureHandle GenerateTexture(Rml::Span<const Rml::byte>, Rml::Vector2i) override { return 1; }
        void ReleaseTexture(Rml::TextureHandle) override {}
        void EnableScissorRegion(bool) override {}
        void SetScissorRegion(Rml::Rectanglei) override {}
    };

    std::string resource(const char* name) {
        std::ifstream file(std::filesystem::path(PROJECT_ROOT_PATH) / "src/visualizer/gui/rmlui/resources" / name);
        std::ostringstream result;
        result << file.rdbuf();
        return result.str();
    }

    std::string textContent(Rml::Element* element) {
        if (auto* text = dynamic_cast<Rml::ElementText*>(element))
            return text->GetText();
        std::string result;
        for (int i = 0; i < element->GetNumChildren(); ++i)
            result += textContent(element->GetChild(i));
        return result;
    }

    struct Rect {
        float left, top, right, bottom;
    };
    Rect bounds(Rml::Element* el) {
        const auto offset = el->GetAbsoluteOffset(Rml::BoxArea::Border);
        const auto size = el->GetBox().GetSize(Rml::BoxArea::Border);
        return {offset.x, offset.y, offset.x + size.x, offset.y + size.y};
    }

    class MenuBarTitleTest : public ::testing::Test {
    protected:
        static void SetUpTestSuite() {
            ASSERT_TRUE(Rml::Initialise());
            ASSERT_TRUE(lfs::event::LocalizationManager::getInstance().initialize(
                (std::filesystem::path(PROJECT_ROOT_PATH) / "src/visualizer/gui/resources/locales").string()));
            ASSERT_TRUE(Rml::LoadFontFace((std::filesystem::path(PROJECT_ROOT_PATH) /
                                           "src/visualizer/gui/assets/fonts/Inter-Regular.ttf")
                                              .string()));
        }
        static void TearDownTestSuite() {
            lfs::event::LocalizationManager::getInstance().reset();
            Rml::Shutdown();
        }
        void SetUp() override {
            context_ = Rml::CreateContext("menu_bar_title_test", {1600, 300}, &renderer_);
            ASSERT_NE(context_, nullptr);
            RmlMenuBarTestAccess::bind(bar_, context_);
            document_ = context_->LoadDocument((std::filesystem::path(PROJECT_ROOT_PATH) /
                                                "src/visualizer/gui/rmlui/resources/menubar.rml")
                                                   .string());
            ASSERT_NE(document_, nullptr);
            document_->SetStyleSheetContainer(Rml::Factory::InstanceStyleSheetString(
                resource("components.rcss") + "\n" + resource("menubar.rcss")));
            for (const auto* id : {"project-title", "project-title-content", "menu-items", "menu-toolbar", "menu-window-controls"})
                ASSERT_NE(document_->GetElementById(id), nullptr) << id;
            RmlMenuBarTestAccess::attach(bar_, document_, manager_);
            bar_.updateLabels({"File", "Edit", "View", "Help"}, {"file", "edit", "view", "help"});
            bar_.updateProjectDisplay(ProjectDisplayInfo{.path = "/projects/scene.licht", .title = "Scene"});
            document_->Show();
            resize(1600);
        }
        void TearDown() override {
            // RmlMenuBar owns no context in this fixture; remove it before Rml shutdown.
            ASSERT_TRUE(Rml::RemoveContext("menu_bar_title_test"));
        }
        Rml::Element* el(const char* id) { return document_->GetElementById(id); }
        void resize(int width, float dp = 1.0f, bool toolbar = true) {
            context_->SetDensityIndependentPixelRatio(dp);
            context_->SetDimensions({width, 300});
            context_->Update();
            const float right = width - bounds(el("menu-window-controls")).left + 12 * dp;
            RmlMenuBarTestAccess::toolbar(bar_, toolbar, right);
            context_->Update();
            RmlMenuBarTestAccess::layout(bar_, width, dp);
            context_->Update();
        }
        std::string titleText() { return textContent(el("project-title-content")->GetChild(0)); }
        inline static TitleRenderInterface renderer_;
        Rml::Context* context_ = nullptr;
        Rml::ElementDocument* document_ = nullptr;
        lfs::vis::gui::RmlUIManager manager_;
        lfs::vis::gui::RmlMenuBar bar_;
    };

    TEST_F(MenuBarTitleTest, ConstrainsAndCentersTitleBetweenMenusAndControls) {
        for (float dp : {1.0f, 1.5f}) {
            for (int width : {1600, 1200, 1000}) {
                for (bool toolbar : {true, false}) {
                    SCOPED_TRACE(::testing::Message() << width << " dp=" << dp << " toolbar=" << toolbar);
                    bar_.updateProjectDisplay(std::string(250, 'W'), "/projects/scene.licht", false);
                    resize(width * dp, dp, toolbar);
                    ASSERT_TRUE(el("project-title")->IsVisible());
                    auto title = bounds(el("project-title-content"));
                    const auto menus = bounds(el("menu-items"));
                    const auto controls = bounds(el(toolbar ? "menu-toolbar" : "menu-window-controls"));
                    EXPECT_GE(title.left, menus.right + 12 * dp - 0.5f);
                    EXPECT_LE(title.right, controls.left - 12 * dp + 0.5f);
                    EXPECT_NEAR((title.left + title.right) / 2, (menus.right + controls.left) / 2, 0.5f);
                    bar_.updateProjectDisplay("Short", "/projects/scene.licht", false);
                    context_->Update();
                    title = bounds(el("project-title-content"));
                    EXPECT_NEAR((title.left + title.right) / 2, (menus.right + controls.left) / 2, 0.5f);
                }
            }
        }
    }

    TEST_F(MenuBarTitleTest, ClippingStyleContractForThePaintBackend) {
        // The geometry sink does not paint glyphs. These computed-style checks
        // verify that the real stylesheet asks Rml to clip/ellipsize; they cannot
        // prove the backend's final pixels.
        for (auto* element : {el("project-title-content"), el("project-title-content")->GetChild(0)}) {
            EXPECT_EQ(element->GetComputedValues().overflow_x(), Rml::Style::Overflow::Hidden);
            EXPECT_EQ(element->GetComputedValues().text_overflow(), Rml::Style::TextOverflow::Ellipsis);
        }
    }

    TEST_F(MenuBarTitleTest, PortalStatusKeepsLongNameInTooltipAtNarrowWidth) {
        auto& store = lfs::vis::app_store();
        const auto previous = store.account_state.get();
        const std::string name = "Katharina Theodora Extremely Long Display Name Example";
        store.account_state.set(lfs::vis::AppStore::AccountState{
            .signed_in = true,
            .authorized = true,
            .label = "KE",
            .display_name = name,
        });
        RmlMenuBarTestAccess::rebuildPortalStatus(bar_);
        resize(1280);
        EXPECT_EQ(textContent(el("menu-portal-connection")), "KE · Portal");
        const auto tooltip = el("menu-portal-connection")->GetAttribute<Rml::String>("title", "");
        EXPECT_NE(tooltip.find("Portal connected as " + name), std::string::npos);
        EXPECT_EQ(textContent(el("menu-portal-connection")).find(name), std::string::npos);
        store.account_state.set(previous);
    }

    TEST_F(MenuBarTitleTest, PortalStatusShowsSpecificTransitionLabels) {
        auto& store = lfs::vis::app_store();
        const auto previous_account = store.account_state.get();
        const auto previous_gallery = store.gallery_state.get();
        store.gallery_state.set({});
        const auto label_for = [&](lfs::vis::AppStore::AccountState state) {
            store.account_state.set(std::move(state));
            RmlMenuBarTestAccess::rebuildPortalStatus(bar_);
            context_->Update();
            return textContent(el("menu-portal-connection"));
        };
        EXPECT_EQ(label_for({}), "Portal: Not connected");
        EXPECT_EQ(label_for({.authorized = true}), "Portal connected, switched off");
        EXPECT_EQ(label_for({.linking = true, .label = "ABCD-EFGH"}), "Portal: Connecting… ABCD-EFGH");
        EXPECT_EQ(label_for({.signed_in = true, .authorized = true, .disconnecting = true, .label = "KT", .display_name = "Kay Test"}),
                  "Portal: Disconnecting…");
        auto approval_gallery = previous_gallery;
        approval_gallery.relink_required = true;
        store.gallery_state.set(approval_gallery);
        EXPECT_EQ(label_for({.signed_in = true, .authorized = true, .label = "KT", .display_name = "Kay Test"}),
                  "Portal: Approval needed");
        store.gallery_state.set({});
        EXPECT_EQ(label_for({.signed_in = true, .authorized = true, .label = "KT", .display_name = "Kay Test"}),
                  "KT · Portal");
        store.account_state.set(previous_account);
        store.gallery_state.set(previous_gallery);
    }

    TEST_F(MenuBarTitleTest, AccountsForPendingToolbarPlacementInTheSameFrame) {
        // SetProperty is pending until Update. The layout code must use the new
        // placement selected by the caller, not the toolbar's stale Rml box.
        RmlMenuBarTestAccess::toolbar(bar_, true, 600);
        RmlMenuBarTestAccess::layout(bar_, 1600, 1.0f);
        context_->Update();
        const auto title = bounds(el("project-title"));
        EXPECT_LE(title.right, bounds(el("menu-toolbar")).left - 12 + 0.5f);
    }

    TEST_F(MenuBarTitleTest, HidesWhenNarrowAndReappearsAfterExpansion) {
        for (float dp : {1.0f, 1.5f}) {
            resize(1600 * dp, dp);
            const auto old = bounds(el("project-title-content"));
            resize(550 * dp, dp);
            EXPECT_FALSE(el("project-title")->IsVisible());
            EXPECT_FALSE(RmlMenuBarTestAccess::hit(bar_, (old.left + old.right) / 2, 15 * dp));
            resize(1600 * dp, dp);
            EXPECT_TRUE(el("project-title")->IsVisible());
            const auto title = bounds(el("project-title-content"));
            EXPECT_TRUE(RmlMenuBarTestAccess::hit(bar_, (title.left + title.right) / 2, 15 * dp));
        }
    }

    TEST_F(MenuBarTitleTest, HitTestUsesVisibleTextBoundsAndRequiresAPath) {
        auto title = bounds(el("project-title-content"));
        const float x = (title.left + title.right) / 2, y = (title.top + title.bottom) / 2;
        EXPECT_TRUE(RmlMenuBarTestAccess::hit(bar_, x, y));
        EXPECT_FALSE(RmlMenuBarTestAccess::hit(bar_, title.left - 1, y));
        EXPECT_FALSE(RmlMenuBarTestAccess::hit(bar_, title.right, y));
        EXPECT_FALSE(RmlMenuBarTestAccess::hit(bar_, x, title.top - 1));
        EXPECT_FALSE(RmlMenuBarTestAccess::hit(bar_, x, title.bottom));
        EXPECT_FALSE(RmlMenuBarTestAccess::hit(bar_, 10, y));
        EXPECT_FALSE(RmlMenuBarTestAccess::hit(bar_, 1590, y));
        bar_.updateProjectDisplay("Scene", "", false);
        context_->Update();
        EXPECT_FALSE(RmlMenuBarTestAccess::hit(bar_, x, y));
        bar_.updateProjectDisplay("", "/projects/scene.licht", false);
        resize(1600);
        EXPECT_FALSE(el("project-title")->IsVisible());
        EXPECT_FALSE(RmlMenuBarTestAccess::hit(bar_, x, y));
    }

    TEST_F(MenuBarTitleTest, UsesMetadataThenFileStemThenUntitled) {
        const std::vector<std::pair<ProjectDisplayInfo, std::string>> cases = {
            {{.path = "/projects/saved.name.licht", .title = "Metadata title"}, "Metadata title"},
            {{.path = "/projects/saved.name.licht"}, "saved.name"},
            {{.path = "/projects/saved.name.licht", .title = ""}, "saved.name"},
            {{.title = "Unsaved title"}, "Unsaved title"},
            {{}, "Untitled"},
            {{.title = ""}, "Untitled"},
        };
        for (const auto& [info, expected] : cases) {
            SCOPED_TRACE(expected);
            bar_.updateProjectDisplay(info);
            context_->Update();
            EXPECT_EQ(titleText(), expected);
        }
    }

    TEST_F(MenuBarTitleTest, DirtyMarkerTracksBothTransitionsWithoutChangingTitle) {
        for (bool dirty : {false, true, false}) {
            bar_.updateProjectDisplay(ProjectDisplayInfo{.path = "/projects/scene.licht", .title = "Scene", .dirty = dirty});
            context_->Update();
            auto* marker = el("project-title-content")->QuerySelector(".project-dirty");
            ASSERT_NE(marker, nullptr);
            EXPECT_EQ(marker->IsVisible(), dirty);
            if (dirty)
                EXPECT_NE(textContent(marker).find('*'), std::string::npos);
            EXPECT_EQ(titleText(), "Scene");
        }
    }

    TEST_F(MenuBarTitleTest, ShortcutTooltipTracksBindingAndUnbinding) {
        using namespace lfs::vis::input;
        EXPECT_EQ(toolModeFromName("global"), ToolMode::GLOBAL);
        EXPECT_EQ(toolModeFromName("selection"), ToolMode::SELECTION);
        EXPECT_EQ(toolModeFromName("translate"), ToolMode::TRANSLATE);
        EXPECT_EQ(toolModeFromName("rotate"), ToolMode::ROTATE);
        EXPECT_EQ(toolModeFromName("scale"), ToolMode::SCALE);
        EXPECT_EQ(toolModeFromName("align"), ToolMode::ALIGN);
        EXPECT_EQ(toolModeFromName("crop_box"), ToolMode::CROP_BOX);
        EXPECT_EQ(toolModeFromName("SeLeCtIoN"), ToolMode::SELECTION);
        EXPECT_EQ(toolModeFromName("unknown"), ToolMode::GLOBAL);
        InputBindings bindings;
        lfs::python::set_keymap_bindings(&bindings);
        auto button = document_->CreateElement("button");
        button->SetAttribute("title", "Home");
        button->SetAttribute("data-action", "camera_reset_home");
        button->SetAttribute("data-shortcut", "stale");
        auto* element = document_->AppendChild(std::move(button));
        const auto initial = bindings.getLocalizedTriggerDescription(Action::CAMERA_RESET_HOME);
        EXPECT_EQ(resolveRmlTooltip(element), "Home (" + initial + ")");
        bindings.setBinding(ToolMode::GLOBAL, Action::CAMERA_RESET_HOME, KeyTrigger{KEY_F6});
        const auto rebound = bindings.getLocalizedTriggerDescription(Action::CAMERA_RESET_HOME);
        EXPECT_EQ(resolveRmlTooltip(element), "Home (" + rebound + ")");
        bindings.clearBinding(ToolMode::GLOBAL, Action::CAMERA_RESET_HOME);
        EXPECT_EQ(resolveRmlTooltip(element), "Home");

        auto selection_button = document_->CreateElement("button");
        selection_button->SetAttribute("title", "Depth filter");
        selection_button->SetAttribute("data-action", "toggle_depth_view");
        selection_button->SetAttribute("data-keymap-action", "toggle_selection_depth_filter");
        selection_button->SetAttribute("data-keymap-mode", "selection");
        auto* selection_element = document_->AppendChild(std::move(selection_button));
        const auto selection_shortcut = bindings.getLocalizedTriggerDescription(
            Action::TOGGLE_SELECTION_DEPTH_FILTER, ToolMode::SELECTION);
        EXPECT_EQ(resolveRmlTooltip(selection_element), "Depth filter (" + selection_shortcut + ")");
        bindings.clearBinding(ToolMode::SELECTION, Action::TOGGLE_SELECTION_DEPTH_FILTER);
        EXPECT_EQ(resolveRmlTooltip(selection_element), "Depth filter");

        auto& locale = lfs::event::LocalizationManager::getInstance();
        ASSERT_TRUE(locale.initialize((std::filesystem::path(PROJECT_ROOT_PATH) /
                                       "src/visualizer/gui/resources/locales")
                                          .string()));
        ASSERT_TRUE(locale.setLanguage("de"));
        auto orbit_button = document_->CreateElement("button");
        orbit_button->SetAttribute("title", "Orbit");
        orbit_button->SetAttribute("data-action", "camera_orbit");
        auto* orbit_element = document_->AppendChild(std::move(orbit_button));
        const auto localized = bindings.getLocalizedTriggerDescription(Action::CAMERA_ORBIT);
        EXPECT_NE(localized.find("Ziehen"), std::string::npos);
        EXPECT_EQ(resolveRmlTooltip(orbit_element), "Orbit (" + localized + ")");
        EXPECT_TRUE(locale.setLanguage("en"));
        lfs::python::set_keymap_bindings(nullptr);
    }

    // Catches a visible tooltip that never hides once the pointer leaves its
    // context for another one (the next hover then skips the show delay): a
    // pointer move over the viewport must still produce a frame while any
    // context has an active tooltip, and must not when none has.
    TEST_F(MenuBarTitleTest, PointerMoveElsewhereRendersWhileATooltipIsActive) {
        Rml::Context* viewport = Rml::CreateContext("menu_bar_title_viewport", {1600, 260}, &renderer_);
        ASSERT_NE(viewport, nullptr);
        ASSERT_NE(viewport->LoadDocumentFromMemory("<rml><body></body></rml>"), nullptr);
        viewport->Update();
        viewport->ProcessMouseMove(800, 160, 0);
        viewport->Update();
        lfs::vis::gui::RmlUIManager manager;
        manager.trackContextFrame(context_, 0, 0);
        manager.trackContextFrame(viewport, 0, 40);

        EXPECT_FALSE(manager.passiveMouseMoveNeedsRender(800.0f, 200.0f));
        manager.setContextNeedsPassiveMouseMoveFrames(context_, true);
        EXPECT_TRUE(manager.passiveMouseMoveNeedsRender(800.0f, 200.0f));
        manager.setContextNeedsPassiveMouseMoveFrames(context_, false);
        EXPECT_FALSE(manager.passiveMouseMoveNeedsRender(800.0f, 200.0f));
        ASSERT_TRUE(Rml::RemoveContext("menu_bar_title_viewport"));
    }

    TEST_F(MenuBarTitleTest, TooltipPreservesFullPathAsTextIncludingMarkupCharacters) {
        const std::filesystem::path path =
            std::filesystem::path{"/projects"} / "long folder" / "<b>scan &amp; capture.licht";
        const std::string expected_path = lfs::core::path_to_utf8(path.lexically_normal());
        bar_.updateProjectDisplay(ProjectDisplayInfo{.path = path, .title = "Scan"});
        context_->Update();
        EXPECT_EQ(el("project-title-content")->GetAttribute<Rml::String>("title", ""), expected_path);
        resize(1600);
        const auto title = bounds(el("project-title-content"));
        bar_.processInput({.mouse_x = (title.left + title.right) / 2, .mouse_y = 15, .screen_w = 1600, .screen_h = 300});
        auto& tooltip = RmlMenuBarTestAccess::tooltip(bar_);
        const auto deadline = tooltip.revealDeadline();
        ASSERT_TRUE(deadline);
        std::this_thread::sleep_until(*deadline);
        ASSERT_TRUE(tooltip.apply(document_, 700, 15, 1600, 300));
        context_->Update();
        auto* rendered = el("frame-tooltip");
        ASSERT_NE(rendered, nullptr);
        EXPECT_TRUE(rendered->IsVisible());
        EXPECT_EQ(rendered->QuerySelector("b"), nullptr);
        EXPECT_EQ(textContent(rendered), expected_path);
    }
} // namespace
