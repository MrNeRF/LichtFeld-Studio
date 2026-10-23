/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/panel_registry.hpp"
#include "gui/rmlui/elements/form_control_layout.hpp"
#include "gui/screen_host.hpp"
#include "input/input_bindings.hpp"
#include "keymap_history_helpers.hpp"
#include "screen/screen_service.hpp"

#include <RmlUi/Core.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

namespace lfs::vis::gui {
    class PropertiesTabPickerTestAccess {
    public:
        static void bind(ScreenHost& host, Rml::Context* const context) { host.bindChromeModel(context); }
        static void attach(ScreenHost& host, Rml::ElementDocument* const document) { host.chrome_document_ = document; }
        static void watchPicks(ScreenHost& host, std::vector<std::string>* const picks) {
            host.on_tab_picked_ = [picks](const std::string& id) { picks->push_back(id); };
        }
        static void sync(ScreenHost& host, const std::vector<ChromeItem>& tabs, const std::string& active) {
            host.properties_.setActiveTab(active);
            host.properties_tab_selected_ = active;
            ChromeArea area;
            area.id = 1;
            area.is_properties = true;
            area.show_tab_picker = true;
            area.picker_dom_id = "tab-picker";
            area.active_tab = active;
            area.tabs = tabs;
            host.chrome_areas_ = {std::move(area)};
            host.chrome_model_.DirtyVariable("areas");
            host.reconcileTabPickerSelection();
        }
        static void showAreas(ScreenHost& host, const int count) {
            host.chrome_areas_.clear();
            for (int i = 0; i < count; ++i) {
                ChromeArea area;
                area.id = i + 1;
                area.left = std::to_string(i * 50) + "px";
                area.top = "0px";
                area.width = "50px";
                area.height = "300px";
                area.header_height = "24px";
                host.chrome_areas_.push_back(std::move(area));
            }
            host.chrome_model_.DirtyVariable("areas");
        }
        static bool tooltipPending(const ScreenHost& host) { return host.chrome_tooltip_.needsFrame(); }
        static std::string active(const ScreenHost& host) {
            return host.chrome_areas_.empty() ? std::string{} : std::string(host.chrome_areas_.front().active_tab);
        }
        static void coverPointer(ScreenHost& host) {
            AreaFrame frame;
            frame.header = {0.0f, 0.0f, 4000.0f, 4000.0f};
            host.frames_ = {frame};
            host.work_ = {0.0f, 0.0f, 400.0f, 600.0f};
        }
        static void release(ScreenHost& host) {
            host.on_tab_picked_ = {};
            host.chrome_tooltip_.setHover({}, nullptr);
            host.chrome_model_ = {};
            host.chrome_document_ = nullptr;
            host.chrome_context_ = nullptr;
            host.chrome_areas_.clear();
        }
    };
} // namespace lfs::vis::gui

namespace {
    using namespace lfs::vis::gui;

    class PickerRenderer final : public Rml::RenderInterface {
        Rml::CompiledGeometryHandle CompileGeometry(Rml::Span<const Rml::Vertex>, Rml::Span<const int>) override { return 1; }
        void RenderGeometry(Rml::CompiledGeometryHandle, Rml::Vector2f, Rml::TextureHandle) override {}
        void ReleaseGeometry(Rml::CompiledGeometryHandle) override {}
        Rml::TextureHandle LoadTexture(Rml::Vector2i&, const Rml::String&) override { return 0; }
        Rml::TextureHandle GenerateTexture(Rml::Span<const Rml::byte>, Rml::Vector2i) override { return 0; }
        void ReleaseTexture(Rml::TextureHandle) override {}
        void EnableScissorRegion(bool) override {}
        void SetScissorRegion(Rml::Rectanglei) override {}
    };

    class RightPanelTabPickerTest : public ::testing::Test {
    protected:
        PickerRenderer renderer;
        Rml::ElementInstancerGeneric<FormControlLayout<Rml::ElementFormControlSelect>> select;
        lfs::vis::screen::ScreenService screens;
        ScreenHost host;
        Rml::Context* context = nullptr;
        std::vector<std::string> picks;

        RightPanelTabPickerTest() : host(screens) {}

        void SetUp() override {
            ASSERT_TRUE(Rml::Initialise());
            Rml::Factory::RegisterElementInstancer("select", &select);
            context = Rml::CreateContext("tab-picker-review", {400, 600}, &renderer);
            ASSERT_NE(context, nullptr);
            PropertiesTabPickerTestAccess::bind(host, context);
            auto* doc = context->LoadDocumentFromMemory(
                "<rml><body data-model='screen_chrome'><div data-for='area : areas'>"
                "<div id='tab-strip'><div class='tab' data-for='item : area.tabs' data-attr-id='item.dom_id' "
                "data-class-active='item.id == area.active_tab'>{{item.label}}</div></div>"
                "<select id='tab-picker' data-attr-id='area.picker_dom_id' data-value='area.active_tab' "
                "data-event-change='pick_tab'><option data-for='tab : area.tabs' data-attr-value='tab.id'>"
                "{{tab.label}}</option></select></div></body></rml>");
            ASSERT_NE(doc, nullptr);
            PropertiesTabPickerTestAccess::attach(host, doc);
            PropertiesTabPickerTestAccess::watchPicks(host, &picks);
            doc->Show();
        }
        Rml::ElementFormControlSelect* picker() {
            auto* const element = context->GetDocument(0)->GetElementById("tab-picker");
            return dynamic_cast<Rml::ElementFormControlSelect*>(element);
        }
        void TearDown() override {
            PropertiesTabPickerTestAccess::release(host);
            Rml::RemoveContext("tab-picker-review");
            Rml::Shutdown();
        }
        static std::vector<ChromeItem> tabs(std::initializer_list<const char*> ids) {
            std::vector<ChromeItem> result;
            for (const char* id : ids) {
                ChromeItem tab;
                tab.id = id;
                tab.label = id;
                tab.dom_id = std::string("tab-") + id;
                result.push_back(std::move(tab));
            }
            return result;
        }
    };

    TEST_F(RightPanelTabPickerTest, PickerHoverPreparesHelpAndOpenPopupSuppressesIt) {
        PropertiesTabPickerTestAccess::sync(host, tabs({"A", "B"}), "A");
        auto* const picker = this->picker();
        ASSERT_NE(picker, nullptr);
        picker->SetAttribute("title", "All panels");
        context->Update();
        PropertiesTabPickerTestAccess::coverPointer(host);
        PanelInputState input;
        const auto position = picker->GetAbsoluteOffset(Rml::BoxArea::Border);
        const auto size = picker->GetBox().GetSize(Rml::BoxArea::Border);
        input.mouse_x = position.x + size.x * 0.5f;
        input.mouse_y = position.y + size.y * 0.5f;
        host.processInput(input, true);
        EXPECT_TRUE(PropertiesTabPickerTestAccess::tooltipPending(host));
        picker->Click();
        input.mouse_wheel = 1;
        host.processInput(input, true);
        EXPECT_FALSE(PropertiesTabPickerTestAccess::tooltipPending(host));
    }

    // Fails if the help that was pending when the picker opened shows over its list.
    TEST_F(RightPanelTabPickerTest, ClickOpeningThePickerDropsItsPendingHelp) {
        PropertiesTabPickerTestAccess::sync(host, tabs({"A", "B"}), "A");
        auto* const picker = this->picker();
        ASSERT_NE(picker, nullptr);
        picker->SetAttribute("title", "All panels");
        context->Update();
        PropertiesTabPickerTestAccess::coverPointer(host);
        PanelInputState input;
        const auto position = picker->GetAbsoluteOffset(Rml::BoxArea::Border);
        const auto size = picker->GetBox().GetSize(Rml::BoxArea::Border);
        input.mouse_x = position.x + size.x * 0.5f;
        input.mouse_y = position.y + size.y * 0.5f;
        host.processInput(input, true);
        ASSERT_TRUE(PropertiesTabPickerTestAccess::tooltipPending(host));
        input.mouse_clicked[0] = true;
        input.mouse_down[0] = true;
        host.processInput(input, true);
        input.mouse_clicked[0] = false;
        input.mouse_down[0] = false;
        input.mouse_released[0] = true;
        host.processInput(input, true);
        ASSERT_TRUE(picker->IsSelectBoxVisible());
        EXPECT_FALSE(PropertiesTabPickerTestAccess::tooltipPending(host));
    }

    TEST_F(RightPanelTabPickerTest, RemovingEarlierTabPreservesActiveSelection) {
        for (const std::string active : {"C", "D"}) {
            PropertiesTabPickerTestAccess::sync(host, tabs({"A", "B", "C", "D"}), active);
            PropertiesTabPickerTestAccess::sync(host, tabs({"A", "C", "D"}), active);
            auto* const picker = this->picker();
            ASSERT_NE(picker, nullptr);
            EXPECT_EQ(picker->GetValue(), active);
            EXPECT_EQ(PropertiesTabPickerTestAccess::active(host), active);
            auto* strip = context->GetDocument(0)->GetElementById("tab-strip");
            int highlighted = 0;
            for (int i = 0; i < strip->GetNumChildren(); ++i) {
                auto* tab = strip->GetChild(i);
                if (tab->IsClassSet("active")) {
                    ++highlighted;
                    EXPECT_EQ(tab->GetId(), "tab-" + active);
                }
            }
            EXPECT_EQ(highlighted, 1);
            EXPECT_TRUE(picks.empty());
        }
    }

    TEST_F(RightPanelTabPickerTest, ReorderingPreservesTheCheckedOptionAndNextUserPick) {
        PropertiesTabPickerTestAccess::sync(host, tabs({"A", "B", "C", "D"}), "A");
        auto* picker = this->picker();
        ASSERT_NE(picker, nullptr);
        picker->Focus();
        picker->Click();
        PropertiesTabPickerTestAccess::sync(host, tabs({"B", "A", "C", "D"}), "A");
        picker = this->picker();
        ASSERT_NE(picker, nullptr);
        ASSERT_NE(picker->GetOption(picker->GetSelection()), nullptr);
        EXPECT_EQ(picker->GetOption(picker->GetSelection())->GetAttribute<Rml::String>("value", ""), "A");
        EXPECT_TRUE(picks.empty());
        context->ProcessKeyDown(Rml::Input::KI_DOWN, 0);
        ASSERT_EQ(picks.size(), 1);
        EXPECT_EQ(picks.front(), "C");
    }

    TEST_F(RightPanelTabPickerTest, InsertingEarlierTabPreservesActiveSelection) {
        PropertiesTabPickerTestAccess::sync(host, tabs({"A", "B", "C", "D"}), "C");
        PropertiesTabPickerTestAccess::sync(host, tabs({"A", "X", "B", "C", "D"}), "C");
        auto* const picker = this->picker();
        ASSERT_NE(picker, nullptr);
        EXPECT_EQ(picker->GetValue(), "C");
        EXPECT_EQ(PropertiesTabPickerTestAccess::active(host), "C");
        EXPECT_TRUE(picks.empty());
    }

    TEST_F(RightPanelTabPickerTest, OpenPopupModelUpdatesDoNotBecomeUserPicks) {
        PropertiesTabPickerTestAccess::sync(host, tabs({"A", "B", "C", "D"}), "C");
        auto* picker = this->picker();
        ASSERT_NE(picker, nullptr);
        picker->Focus();
        picker->Click();
        ASSERT_TRUE(picker->IsSelectBoxVisible());
        PropertiesTabPickerTestAccess::sync(host, tabs({"A", "C", "D"}), "C");
        picker = this->picker();
        ASSERT_NE(picker, nullptr);
        EXPECT_EQ(picker->GetValue(), "C");
        EXPECT_TRUE(picks.empty());
        context->ProcessKeyDown(Rml::Input::KI_DOWN, 0);
        ASSERT_EQ(picks.size(), 1);
        EXPECT_EQ(picks.front(), "D");
    }

    class InlineStyleErrorLog final : public Rml::SystemInterface {
    public:
        bool LogMessage(const Rml::Log::Type, const Rml::String& message) override {
            if (message.find("inline property declaration") != Rml::String::npos)
                errors.push_back(message);
            return true;
        }
        std::vector<std::string> errors;
    };

    class ScreenChromeAreasTest : public ::testing::Test {
    protected:
        PickerRenderer renderer;
        InlineStyleErrorLog log;
        lfs::vis::screen::ScreenService screens;
        ScreenHost host;
        Rml::Context* context = nullptr;

        ScreenChromeAreasTest() : host(screens) {}

        void SetUp() override {
            Rml::SetSystemInterface(&log);
            ASSERT_TRUE(Rml::Initialise());
            context = Rml::CreateContext("screen-chrome-areas", {800, 600}, &renderer);
            ASSERT_NE(context, nullptr);
            PropertiesTabPickerTestAccess::bind(host, context);
            auto* const doc = context->LoadDocument(PROJECT_ROOT_PATH "/src/visualizer/gui/rmlui/resources/screen_chrome.rml");
            ASSERT_NE(doc, nullptr);
            PropertiesTabPickerTestAccess::attach(host, doc);
            doc->Show();
        }
        void TearDown() override {
            PropertiesTabPickerTestAccess::release(host);
            Rml::RemoveContext("screen-chrome-areas");
            Rml::Shutdown();
        }
    };

    // RmlUi updates a data-for view and the views on the elements it repeats in
    // one unordered pass, so geometry bound there could be read from a removed
    // area and written as "left: ;" when a project restores fewer areas.
    TEST_F(ScreenChromeAreasTest, RemovedAreasNeverWriteEmptyGeometry) {
        for (int round = 0; round < 20; ++round) {
            PropertiesTabPickerTestAccess::showAreas(host, 8);
            context->Update();
            PropertiesTabPickerTestAccess::showAreas(host, 1);
            context->Update();
        }
        EXPECT_EQ(log.errors, std::vector<std::string>{});
    }

    // Fails if the sidebar category keys stop following the key map, step on a held key whose
    // binding does not repeat, or take keys that type into a focused text field.
    class SidebarCategoryKeys : public lfs::vis::input::test::KeymapTest {};

    TEST_F(SidebarCategoryKeys, FollowTheKeyMapAndLeaveTypedKeysToFields) {
        using lfs::vis::FrameInputEvent;
        using lfs::vis::FrameInputEventKind;
        namespace input = lfs::vis::input;
        input::InputBindings bindings;
        const auto press = [](const SDL_Scancode scancode, const SDL_Keycode keycode, const SDL_Keymod mods,
                              const bool repeat = false) {
            return FrameInputEvent{.kind = FrameInputEventKind::KeyDown,
                                   .scancode = scancode,
                                   .keycode = keycode,
                                   .modifiers = mods,
                                   .repeat = repeat};
        };
        const auto ctrl_tab = press(SDL_SCANCODE_TAB, SDLK_TAB, SDL_KMOD_LCTRL);
        EXPECT_EQ(ScreenHost::sidebarCategoryStep(bindings, ctrl_tab, false), 1);
        EXPECT_EQ(ScreenHost::sidebarCategoryStep(bindings, press(SDL_SCANCODE_TAB, SDLK_TAB, SDL_KMOD_LCTRL | SDL_KMOD_LSHIFT), false), -1);
        EXPECT_FALSE(ScreenHost::sidebarCategoryStep(bindings, press(SDL_SCANCODE_TAB, SDLK_TAB, SDL_KMOD_NONE), false));
        // A Ctrl chord still steps while a field has focus.
        EXPECT_EQ(ScreenHost::sidebarCategoryStep(bindings, ctrl_tab, true), 1);
        // A held key is consumed without stepping unless its binding repeats.
        EXPECT_EQ(ScreenHost::sidebarCategoryStep(bindings, press(SDL_SCANCODE_TAB, SDLK_TAB, SDL_KMOD_LCTRL, true), false), 0);
        lfs::vis::InputEventDispatch consumed;
        consumed.consumed = true;
        auto handled = ctrl_tab;
        handled.dispatch = &consumed;
        EXPECT_FALSE(ScreenHost::sidebarCategoryStep(bindings, handled, false));

        bindings.setBinding(input::ToolMode::GLOBAL, input::Action::NEXT_SIDEBAR_CATEGORY,
                            input::KeyTrigger{input::KEY_N});
        const auto n = press(SDL_SCANCODE_N, SDLK_N, SDL_KMOD_NONE);
        EXPECT_FALSE(ScreenHost::sidebarCategoryStep(bindings, ctrl_tab, false));
        EXPECT_EQ(ScreenHost::sidebarCategoryStep(bindings, n, false), 1);
        // N types into a focused field instead.
        EXPECT_FALSE(ScreenHost::sidebarCategoryStep(bindings, n, true));
    }

    // Fails if cycling stops wrapping, or if closing a tab disables its panel instead of handing
    // the tab to the owner, which unloads the plugin once the input dispatch returns.
    TEST(PropertiesEditorTabs, CycleWrapsAndClosedTabsGoToTheOwner) {
        class Panel final : public IPanel {
            void draw(const PanelDrawContext&) override {}
        };
        static constexpr std::array ids{"test.sidebar_cycle_a", "test.sidebar_cycle_b", "test.sidebar_cycle_c"};
        auto& registry = PanelRegistry::instance();
        for (const char* id : ids) {
            PanelInfo info;
            info.id = id;
            info.label = id;
            info.space = PanelSpace::MainPanelTab;
            info.tab_closeable = true;
            info.panel = std::make_shared<Panel>();
            ASSERT_TRUE(registry.register_panel(std::move(info)));
        }
        struct Unregister {
            ~Unregister() {
                for (const char* id : ids)
                    PanelRegistry::instance().unregister_panel(id);
            }
        } unregister;
        const auto tabs = registry.get_panels_for_space(PanelSpace::MainPanelTab);
        ASSERT_GE(tabs.size(), ids.size());
        const auto index_of = [&](const std::string& id) {
            return static_cast<int>(std::ranges::find(tabs, id, &PanelSummary::id) - tabs.begin());
        };
        const int count = static_cast<int>(tabs.size());

        PropertiesEditor editor;
        editor.setActiveTab(ids[0]);
        const int start = index_of(ids[0]);
        ASSERT_TRUE(editor.cycleTab(1));
        EXPECT_EQ(editor.activeTab(), tabs[static_cast<size_t>((start + 1) % count)].id);
        ASSERT_TRUE(editor.cycleTab(-1));
        EXPECT_EQ(editor.activeTab(), ids[0]);
        editor.setActiveTab(tabs.front().id);
        ASSERT_TRUE(editor.cycleTab(-1));
        EXPECT_EQ(editor.activeTab(), tabs.back().id);
        EXPECT_FALSE(editor.cycleTab(0));

        lfs::vis::screen::ScreenService screens;
        editor.setActiveTab(ids[1]);
        editor.headerAction(AreaFrame{}, screens.screen(), std::string("close:") + ids[1], 0.0f, 0.0f);
        EXPECT_EQ(editor.takeClosedTab(), ids[1]);
        EXPECT_TRUE(editor.takeClosedTab().empty());
        EXPECT_TRUE(editor.activeTab().empty());
        EXPECT_TRUE(std::ranges::any_of(registry.get_panels_for_space(PanelSpace::MainPanelTab),
                                        [&](const PanelSummary& tab) { return tab.id == ids[1]; }));
    }

} // namespace
