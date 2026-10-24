/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/panel_registry.hpp"
#include "gui/rmlui/rml_theme.hpp"
#include "gui/screen_host.hpp"
#include "input/input_bindings.hpp"
#include "keymap_history_helpers.hpp"
#include "python/python_runtime.hpp"
#include "screen/screen_service.hpp"

#include <RmlUi/Core.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <format>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace lfs::vis::gui {
    class PropertiesTabRailTestAccess {
    public:
        static void bind(ScreenHost& host, Rml::Context* const context) { host.bindChromeModel(context); }
        static void attach(ScreenHost& host, Rml::ElementDocument* const document) { host.chrome_document_ = document; }
        static void show(ScreenHost& host, std::vector<ChromeArea> areas) {
            host.chrome_areas_ = std::move(areas);
            host.chrome_model_.DirtyVariable("areas");
            settle(host);
        }
        static void settle(ScreenHost& host) {
            host.chrome_context_->Update();
            host.syncTabRails();
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
        static void release(ScreenHost& host) {
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

    // Records the screen bounds of textured (glyph) geometry, its scissor and its text direction.
    class TabRenderer final : public Rml::RenderInterface {
    public:
        struct TextDraw {
            Rml::Rectanglef bounds;
            Rml::Rectanglei clip;
            float direction_y;
        };
        std::vector<TextDraw> text_draws;

    private:
        std::unordered_map<Rml::CompiledGeometryHandle, std::vector<Rml::Vertex>> geometry;
        Rml::CompiledGeometryHandle next_handle = 1;
        Rml::Matrix4f transform = Rml::Matrix4f::Identity();
        Rml::Rectanglei clip;
        bool clipping = false;
        Rml::CompiledGeometryHandle CompileGeometry(Rml::Span<const Rml::Vertex> vertices, Rml::Span<const int>) override {
            const auto handle = next_handle++;
            geometry[handle] = {vertices.begin(), vertices.end()};
            return handle;
        }
        void RenderGeometry(Rml::CompiledGeometryHandle handle, Rml::Vector2f translation, Rml::TextureHandle texture) override {
            if (!texture)
                return;
            auto bounds = Rml::Rectanglef::MakeInvalid();
            for (const auto& vertex : geometry.at(handle)) {
                const auto p = transform * Rml::Vector4f(vertex.position.x + translation.x, vertex.position.y + translation.y, 0, 1);
                const Rml::Vector2f point(p.x / p.w, p.y / p.w);
                bounds = bounds.Valid() ? bounds.Join(point) : Rml::Rectanglef::FromPosition(point);
            }
            const auto direction = transform * Rml::Vector4f(1, 0, 0, 0);
            text_draws.push_back({bounds, clipping ? clip : Rml::Rectanglei::FromSize({600, 2400}), direction.y});
        }
        void ReleaseGeometry(Rml::CompiledGeometryHandle handle) override { geometry.erase(handle); }
        Rml::TextureHandle LoadTexture(Rml::Vector2i&, const Rml::String&) override { return 0; }
        Rml::TextureHandle GenerateTexture(Rml::Span<const Rml::byte>, Rml::Vector2i) override { return 1; }
        void ReleaseTexture(Rml::TextureHandle) override {}
        void EnableScissorRegion(bool enabled) override { clipping = enabled; }
        void SetScissorRegion(Rml::Rectanglei region) override { clip = region; }
        void SetTransform(const Rml::Matrix4f* matrix) override {
            transform = matrix ? *matrix : Rml::Matrix4f::Identity();
        }
    };

    Rml::Vector2f position(Rml::Element* element) { return element->GetAbsoluteOffset(Rml::BoxArea::Border); }
    Rml::Vector2f size(Rml::Element* element) { return element->GetBox().GetSize(Rml::BoxArea::Border); }

    // The category rail of a Properties area in the real screen chrome document, at 1x to 2x.
    class PropertiesTabRailTest : public ::testing::TestWithParam<float> {
    protected:
        TabRenderer renderer;
        lfs::vis::screen::ScreenService screens;
        ScreenHost host;
        Rml::Context* context = nullptr;
        Rml::ElementDocument* doc = nullptr;

        PropertiesTabRailTest() : host(screens) {}

        void SetUp() override {
            ASSERT_TRUE(Rml::Initialise());
            const auto fonts = std::filesystem::path(PROJECT_ROOT_PATH) / "src/visualizer/gui/assets/fonts";
            ASSERT_TRUE(Rml::LoadFontFace((fonts / "Inter-Regular.ttf").generic_string()));
            ASSERT_TRUE(Rml::LoadFontFace((fonts / "Inter-SemiBold.ttf").generic_string()));
            context = Rml::CreateContext("tab-rail", {600, 2400}, &renderer);
            ASSERT_NE(context, nullptr);
            context->SetDensityIndependentPixelRatio(GetParam());
            context->SetDefaultScrollBehavior(Rml::ScrollBehavior::Instant, 1.0f);
            PropertiesTabRailTestAccess::bind(host, context);
            doc = context->LoadDocument(PROJECT_ROOT_PATH "/src/visualizer/gui/rmlui/resources/screen_chrome.rml");
            ASSERT_NE(doc, nullptr);
            PropertiesTabRailTestAccess::attach(host, doc);
            rml_theme::applyTheme(doc, rml_theme::loadBaseRCSS("rmlui/screen_chrome.rcss"),
                                  rml_theme::loadBaseRCSS("rmlui/screen_chrome.theme.rcss"));
            doc->Show();
        }
        void TearDown() override {
            PropertiesTabRailTestAccess::release(host);
            Rml::RemoveContext("tab-rail");
            Rml::Shutdown();
        }
        float dp(const float value) const { return value * GetParam(); }
        // A Properties area at (20, 10), `height` dp tall, with these tabs.
        void show(const std::vector<std::string>& labels, const int active, const float height, const int closeable = -1) {
            ChromeArea area;
            area.id = 1;
            area.left = "20px";
            area.top = "10px";
            area.width = std::format("{:.0f}px", dp(240));
            area.height = std::format("{:.0f}px", dp(height));
            area.header_height = std::format("{:.0f}px", std::round(dp(28)));
            area.is_properties = true;
            area.rail_id = "tab-rail-1";
            for (int i = 0; i < static_cast<int>(labels.size()); ++i) {
                ChromeItem tab;
                tab.id = std::format("tab:plugin-{}", i);
                tab.label = labels[i];
                tab.dom_id = std::format("tab-plugin-{}", i);
                tab.active = i == active;
                tab.closeable = i == closeable;
                area.tabs.push_back(std::move(tab));
            }
            PropertiesTabRailTestAccess::show(host, {std::move(area)});
        }
        Rml::Element* rail() { return doc->GetElementById("tab-rail-1"); }
        Rml::Element* strip() { return rail()->QuerySelector(".tab-strip"); }
        Rml::Element* tab(const int index) { return doc->GetElementById(std::format("tab-plugin-{}", index)); }
        bool inView(const int index) {
            const float top = position(tab(index)).y;
            const float strip_top = position(strip()).y;
            return top >= strip_top - 0.5f && top + size(tab(index)).y <= strip_top + strip()->GetClientHeight() + 0.5f;
        }
    };

    // Fails if the tabs leave the rail below the header, stop stacking, or stop sizing to their labels.
    TEST_P(PropertiesTabRailTest, TabsStackBelowTheHeaderAndLabelsSetTheirHeight) {
        show({"Rendering", "Training", "A plugin with a much longer translated label"}, 0, 900);
        const auto rail_position = position(rail());
        EXPECT_NEAR(rail_position.x, 20 + dp(4), 0.5f);
        EXPECT_NEAR(rail_position.y, 10 + std::round(dp(28)) + dp(4), 0.5f);
        EXPECT_NEAR(position(tab(0)).y, position(strip()).y, 0.5f);
        EXPECT_FALSE(rail()->IsClassSet("overflow"));
        float last_bottom = 0;
        for (int i = 0; i < 3; ++i) {
            SCOPED_TRACE(i);
            const bool active = i == 0;
            const auto tab_position = position(tab(i));
            const auto tab_size = size(tab(i));
            // Inactive tabs are recessed; the active one fills the rail.
            EXPECT_NEAR(tab_position.x, rail_position.x + dp(active ? 0 : 3), 0.5f);
            EXPECT_NEAR(tab_size.x, dp(active ? 28 : 25), 0.5f);
            EXPECT_GE(tab_position.y, last_bottom);
            last_bottom = tab_position.y + tab_size.y;
            EXPECT_NEAR(tab_size.y, size(tab(i)->GetFirstChild()).x + dp(18), 1.0f);
        }
        EXPECT_GT(size(tab(2)).y, size(tab(1)).y);
    }

    // Fails if a rotated label stops reading bottom to top, leaves its tab or the strip's clip,
    // drifts off centre, or sits on a fractional pixel.
    TEST_P(PropertiesTabRailTest, RotatedLabelsStayInsideTheirTabsAtFractionalScales) {
        const std::vector<std::string> labels{"Rendering", "Training", "Plugin"};
        for (int active = 0; active < 3; ++active) {
            show(labels, active, 900, 2);
            renderer.text_draws.clear();
            context->Render();
            for (int i = 0; i < 3; ++i) {
                SCOPED_TRACE(std::format("active {} tab {}", active, i));
                auto* const label = tab(i)->GetFirstChild();
                const auto tab_position = position(tab(i));
                const auto tab_size = size(tab(i));
                const float close = i == 2 ? dp(18) : 0.0f;
                auto ink = Rml::Rectanglef::MakeInvalid();
                for (const auto& draw : renderer.text_draws) {
                    const auto centre = draw.bounds.Center();
                    // Glyphs rotated to run up the screen; the close cross stays upright.
                    if (std::abs(draw.direction_y + 1.0f) > 0.001f || centre.x < tab_position.x ||
                        centre.x > tab_position.x + tab_size.x || centre.y < tab_position.y ||
                        centre.y > tab_position.y + tab_size.y)
                        continue;
                    ink = ink.Valid() ? ink.Join(draw.bounds) : draw.bounds;
                    EXPECT_GE(draw.bounds.Top(), draw.clip.Top() - 1.0f);
                    EXPECT_LE(draw.bounds.Bottom(), draw.clip.Bottom() + 1.0f);
                    EXPECT_GE(draw.bounds.Left(), draw.clip.Left() - 1.0f);
                    EXPECT_LE(draw.bounds.Right(), draw.clip.Right() + 1.0f);
                }
                ASSERT_TRUE(ink.Valid());
                EXPECT_GE(ink.Left(), tab_position.x + dp(1));
                EXPECT_LE(ink.Right(), tab_position.x + tab_size.x - dp(1));
                EXPECT_GE(ink.Top(), tab_position.y + dp(8));
                EXPECT_LE(ink.Bottom(), tab_position.y + tab_size.y - close - dp(8));
                EXPECT_NEAR(ink.Center().y, tab_position.y + (tab_size.y - close) * 0.5f, 1.5f);
                EXPECT_FLOAT_EQ(label->GetAbsoluteLeft(), std::round(label->GetAbsoluteLeft()));
                EXPECT_FLOAT_EQ(label->GetAbsoluteTop(), std::round(label->GetAbsoluteTop()));
            }
        }
    }

    // Fails if the active tab stops joining the content where the panels start, or the rail edge
    // does not open beside it alone.
    TEST_P(PropertiesTabRailTest, ActiveTabJoinsTheContentThroughTheOpenRailEdge) {
        show({"Rendering", "Training", "Plugin"}, 1, 900);
        auto* const before = rail()->QuerySelector(".tab-edge-before");
        auto* const after = rail()->QuerySelector(".tab-edge-after");
        const auto active = position(tab(1));
        const auto active_size = size(tab(1));
        const float content_x = 20 + dp(PropertiesEditor::kTabRailDp);
        EXPECT_NEAR(position(rail()).x + size(rail()).x, content_x, 0.5f);
        EXPECT_NEAR(active.x + active_size.x, content_x, 0.5f);
        EXPECT_EQ(tab(1)->GetBox().GetEdge(Rml::BoxArea::Border, Rml::BoxEdge::Right), 0.0f);
        EXPECT_GT(tab(0)->GetBox().GetEdge(Rml::BoxArea::Border, Rml::BoxEdge::Right), 0.0f);
        EXPECT_NEAR(position(before).x + size(before).x, content_x, 0.5f);
        EXPECT_NEAR(position(before).y, position(strip()).y, 0.5f);
        EXPECT_NEAR(position(before).y + size(before).y, active.y, 0.5f);
        EXPECT_NEAR(position(after).y, active.y + active_size.y, 0.5f);
        EXPECT_NEAR(position(after).y + size(after).y, position(strip()).y + strip()->GetClientHeight(), 0.5f);
        show({"Rendering", "Training", "Plugin"}, 0, 900);
        EXPECT_FLOAT_EQ(size(before).y, 0.0f);
    }

    // Fails if overflowing tabs lose their scroll buttons, the buttons stop paging or tracking the
    // limits, a tab selected elsewhere stays out of view, or a short rail lets buttons cover it.
    TEST_P(PropertiesTabRailTest, OverflowButtonsPageTheStripAndSelectedTabsComeIntoView) {
        std::vector<std::string> labels;
        for (int i = 0; i < 12; ++i)
            labels.push_back(std::format("Plugin {}", i));
        show(labels, 0, 300);
        auto* const up = rail()->QuerySelector(".tab-scroll-up");
        auto* const down = rail()->QuerySelector(".tab-scroll-down");
        ASSERT_TRUE(rail()->IsClassSet("overflow"));
        EXPECT_TRUE(up->IsVisible());
        EXPECT_TRUE(down->IsVisible());
        EXPECT_GE(position(strip()).y, position(up).y + size(up).y);
        EXPECT_LE(position(strip()).y + strip()->GetClientHeight(), position(down).y);
        EXPECT_TRUE(up->HasAttribute("disabled"));
        EXPECT_FALSE(down->HasAttribute("disabled"));
        down->Click();
        PropertiesTabRailTestAccess::settle(host);
        EXPECT_GT(strip()->GetScrollTop(), 0.0f);
        EXPECT_FALSE(up->HasAttribute("disabled"));
        for (int i = 0; i < 20; ++i)
            down->Click();
        PropertiesTabRailTestAccess::settle(host);
        EXPECT_TRUE(down->HasAttribute("disabled"));
        const float bottom = strip()->GetScrollTop();
        up->Click();
        PropertiesTabRailTestAccess::settle(host);
        EXPECT_LT(strip()->GetScrollTop(), bottom);
        EXPECT_FALSE(down->HasAttribute("disabled"));

        show(labels, 11, 300);
        EXPECT_TRUE(inView(11));
        EXPECT_TRUE(down->HasAttribute("disabled"));
        // The user's own scrolling stays until the active tab moves.
        strip()->SetScrollTop(0.0f);
        PropertiesTabRailTestAccess::settle(host);
        EXPECT_EQ(strip()->GetScrollTop(), 0.0f);
        EXPECT_FALSE(inView(11));
        show(labels, 6, 300);
        EXPECT_TRUE(inView(6));

        // Too short for buttons beside a tab: the whole rail scrolls.
        show(labels, 6, 28 + 8 + 70);
        EXPECT_FALSE(rail()->IsClassSet("overflow"));
        EXPECT_FALSE(up->IsVisible());
        EXPECT_NEAR(strip()->GetClientHeight(), rail()->GetClientHeight(), 0.5f);
        EXPECT_GT(strip()->GetScrollHeight(), strip()->GetClientHeight());
        EXPECT_TRUE(inView(6));

        // A selected tab taller than the strip shows the start of its label and its close button.
        show({"Rendering", "Training", "A longer plugin label"}, 2, 150, 2);
        ASSERT_TRUE(rail()->IsClassSet("overflow"));
        ASSERT_GT(size(tab(2)).y, strip()->GetClientHeight());
        auto* const close = tab(2)->QuerySelector(".tab-close");
        EXPECT_GE(position(close).y, position(strip()).y);
        EXPECT_LE(position(close).y + size(close).y, position(strip()).y + strip()->GetClientHeight() + 0.5f);

        show({"Rendering", "Training"}, 0, 300);
        EXPECT_FALSE(rail()->IsClassSet("overflow"));
        EXPECT_EQ(strip()->GetScrollTop(), 0.0f);
    }

    INSTANTIATE_TEST_SUITE_P(UiScale, PropertiesTabRailTest, ::testing::Values(1.0f, 1.25f, 1.5f, 1.75f, 2.0f));

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
        TabRenderer renderer;
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
            PropertiesTabRailTestAccess::bind(host, context);
            auto* const doc = context->LoadDocument(PROJECT_ROOT_PATH "/src/visualizer/gui/rmlui/resources/screen_chrome.rml");
            ASSERT_NE(doc, nullptr);
            PropertiesTabRailTestAccess::attach(host, doc);
            doc->Show();
        }
        void TearDown() override {
            PropertiesTabRailTestAccess::release(host);
            Rml::RemoveContext("screen-chrome-areas");
            Rml::Shutdown();
        }
    };

    // RmlUi updates a data-for view and the views on the elements it repeats in
    // one unordered pass, so geometry bound there could be read from a removed
    // area and written as "left: ;" when a project restores fewer areas.
    TEST_F(ScreenChromeAreasTest, RemovedAreasNeverWriteEmptyGeometry) {
        for (int round = 0; round < 20; ++round) {
            PropertiesTabRailTestAccess::showAreas(host, 8);
            context->Update();
            PropertiesTabRailTestAccess::showAreas(host, 1);
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

    // Fails if the active tab's panels draw under the tab rail, or the wheel over the rail scrolls
    // them instead of leaving it to the tabs.
    TEST(PropertiesEditorTabs, PanelsDrawRightOfTheRailWhichKeepsItsWheel) {
        class Panel final : public IPanel {
        public:
            void draw(const PanelDrawContext&) override {}
            PanelRenderCapabilities renderCapabilities() const override { return {.direct = true}; }
            PanelDirectRenderResult renderDirect(const PanelDirectRenderRequest& request,
                                                 const PanelDrawContext&) override {
                if (request.mode == PanelDirectRenderMode::Draw) {
                    x = request.x;
                    width = request.width;
                }
                return {.handled = true, .height = 2000.0f};
            }
            float x = 0.0f;
            float width = 0.0f;
        };
        auto panel = std::make_shared<Panel>();
        PanelInfo info;
        info.id = "test.rail_content";
        info.label = "Rail content";
        info.space = PanelSpace::MainPanelTab;
        info.panel = panel;
        ASSERT_TRUE(PanelRegistry::instance().register_panel(std::move(info)));
        const float previous_scale = lfs::python::get_shared_dpi_scale();
        struct Restore {
            float scale;
            ~Restore() {
                PanelRegistry::instance().unregister_panel("test.rail_content");
                lfs::python::set_shared_dpi_scale(scale);
            }
        } restore{previous_scale};
        for (const float scale : {1.0f, 1.25f, 1.5f, 1.75f, 2.0f}) {
            SCOPED_TRACE(scale);
            lfs::python::set_shared_dpi_scale(scale);
            PropertiesEditor editor;
            editor.setActiveTab("test.rail_content");
            AreaFrame frame;
            frame.content = {100.0f, 50.0f, 300.0f, 400.0f};
            const UIContext ui;
            const PanelDrawContext draw{.ui = &ui};
            PanelInputState input;
            input.mouse_x = 100.0f + 10.0f * scale;
            input.mouse_y = 200.0f;
            input.mouse_wheel = -1.0f;
            editor.draw({.area = frame, .ui = ui, .draw = draw, .input = input});
            EXPECT_FLOAT_EQ(panel->x, 100.0f + PropertiesEditor::kTabRailDp * scale);
            EXPECT_FLOAT_EQ(panel->width, 300.0f - (PropertiesEditor::kTabRailDp + 8.0f) * scale);
            EXPECT_EQ(editor.scroll(), 0.0f);
            input.mouse_x = panel->x + 1.0f;
            editor.draw({.area = frame, .ui = ui, .draw = draw, .input = input});
            EXPECT_GT(editor.scroll(), 0.0f);
        }
    }

} // namespace
