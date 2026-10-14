/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/event_bridge/event_bridge.hpp"
#include "core/events.hpp"
#include "gui/layout_state.hpp"
#include "gui/rml_viewport_overlay.hpp"
#include "gui/vram_hud_overlay.hpp"

#include <RmlUi/Core.h>
#include <RmlUi/Core/Factory.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <typeinfo>

namespace lfs::vis::gui {

    namespace {
        class NullRenderInterface final : public Rml::RenderInterface {
        public:
            Rml::CompiledGeometryHandle CompileGeometry(Rml::Span<const Rml::Vertex>,
                                                        Rml::Span<const int>) override {
                return Rml::CompiledGeometryHandle(1);
            }
            void RenderGeometry(Rml::CompiledGeometryHandle, Rml::Vector2f,
                                Rml::TextureHandle) override {}
            void ReleaseGeometry(Rml::CompiledGeometryHandle) override {}
            Rml::TextureHandle LoadTexture(Rml::Vector2i& dimensions, const Rml::String&) override {
                dimensions = Rml::Vector2i(1, 1);
                return Rml::TextureHandle(1);
            }
            Rml::TextureHandle GenerateTexture(Rml::Span<const Rml::byte>,
                                               Rml::Vector2i) override {
                return Rml::TextureHandle(1);
            }
            void ReleaseTexture(Rml::TextureHandle) override {}
            void EnableScissorRegion(bool) override {}
            void SetScissorRegion(Rml::Rectanglei) override {}
        };

        class ScopedEnvironmentVariable {
        public:
            ScopedEnvironmentVariable(const char* name, const std::string& value) : name_(name) {
                if (const char* previous = std::getenv(name))
                    previous_ = previous;
                set(value);
            }
            ~ScopedEnvironmentVariable() { set(previous_); }
            ScopedEnvironmentVariable(const ScopedEnvironmentVariable&) = delete;
            ScopedEnvironmentVariable& operator=(const ScopedEnvironmentVariable&) = delete;

        private:
            void set(const std::optional<std::string>& value) const {
#ifdef _WIN32
                (void)_putenv_s(name_.c_str(), value ? value->c_str() : "");
#else
                if (value)
                    (void)setenv(name_.c_str(), value->c_str(), 1);
                else
                    (void)unsetenv(name_.c_str());
#endif
            }

            std::string name_;
            std::optional<std::string> previous_;
        };

        // The ids, toggle attributes and drag handles the overlay binds to in
        // viewport_overlay.rml, with fixed sizes so no font is needed.
        constexpr const char* kHudDocument = R"(<rml><head><style>
            body { width: 800px; height: 600px; }
            #vram-hud-overlay { position: absolute; left: 100px; top: 100px; width: 300px; }
            #perf-hud-strip { display: block; width: 300px; height: 60px; }
            #perf-hud-strip-header { display: block; width: 300px; height: 20px; drag: drag; }
            #perf-hud-card { display: block; width: 300px; height: 200px; }
            #vram-hud-header { display: block; position: relative; width: 300px; height: 20px; drag: drag; }
            #perf-hud-collapse { position: absolute; right: 0px; top: 0px; width: 20px; height: 20px; }
            #vram-hud-resize { position: absolute; right: 0; bottom: 0; width: 16px; height: 16px; drag: drag; }
            #vram-hud-summary, .vram-hud-summary-row { display: block; } .hidden { display: none; }
        </style></head><body>
            <div id="vram-hud-overlay">
                <div id="perf-hud-strip">
                    <div id="perf-hud-strip-header" data-perf-toggle-expanded="true"></div>
                </div>
                <div id="perf-hud-card">
                    <div id="vram-hud-header">
                        <div id="perf-hud-collapse" data-perf-toggle-expanded="true"></div>
                    </div>
                    <div id="perf-hud-tracking-off" class="hidden"></div>
                    <div id="vram-hud-panel-overview"><div id="vram-hud-summary"></div></div>
                    <div id="vram-hud-panel-ledger" class="hidden"></div>
                </div>
                <div id="vram-hud-resize"></div>
            </div>
        </body></rml>)";
    } // namespace

    class VramHudHeaderClickTest : public ::testing::Test {
    protected:
        static void SetUpTestSuite() {
            render_interface_ = new NullRenderInterface();
            ASSERT_TRUE(Rml::Initialise());
        }

        static void TearDownTestSuite() {
            Rml::Shutdown();
            delete render_interface_;
            render_interface_ = nullptr;
        }

        void SetUp() override {
            home_root_ = std::filesystem::temp_directory_path() /
                         ("lfs_vram_hud_header_click_" +
                          std::to_string(reinterpret_cast<std::uintptr_t>(this)));
            std::error_code ec;
            std::filesystem::remove_all(home_root_, ec);
            home_ = std::make_unique<ScopedEnvironmentVariable>("LFS_HOME", home_root_.string());

            toggle_handler_ = core::events::ui::TogglePerfHudExpanded::when(
                [this](const auto&) { ++toggles_; });

            context_ = Rml::CreateContext("vram_hud_header_click", Rml::Vector2i(800, 600),
                                          render_interface_);
            ASSERT_NE(context_, nullptr);
            document_ = context_->LoadDocumentFromMemory(kHudDocument);
            ASSERT_NE(document_, nullptr);
            document_->Show();

            overlay_ = std::make_unique<VramHudOverlay>();
            overlay_->setViewportGeometry(0.0f, 0.0f, 800.0f, 600.0f);
            overlay_->onDocumentLoaded(document_);
            setExpanded(false);
            context_->Update();
            (void)overlay_->initializeGeometryAfterLayout();
            context_->Update();
        }

        void TearDown() override {
            // The document owns elements that hold the overlay's listeners;
            // it must go before the overlay.
            if (overlay_)
                overlay_->onDocumentDestroyed();
            if (context_)
                Rml::RemoveContext(context_->GetName());
            context_ = nullptr;
            document_ = nullptr;
            overlay_.reset();
            lfs::event::EventBridge::instance().unsubscribe(
                typeid(core::events::ui::TogglePerfHudExpanded), toggle_handler_);
            home_.reset();
            std::error_code ec;
            std::filesystem::remove_all(home_root_, ec);
        }

        void setExpanded(const bool expanded) {
            VramHudOverlay::State state;
            state.perf_hud.visible = true;
            state.perf_hud.expanded = expanded;
            (void)overlay_->setState(std::move(state));
            context_->Update();
            (void)overlay_->initializeGeometryAfterLayout();
            context_->Update();
        }

        [[nodiscard]] bool publish(VramHudOverlay::State state) {
            RmlViewportOverlay viewport;
            viewport.vram_hud_ = std::move(overlay_);
            viewport.render_needed_ = false;
            viewport.render_reason_bits_ = 0;
            viewport.setVramHudOverlay(std::move(state));
            context_->Update();
            overlay_ = std::move(viewport.vram_hud_);
            return viewport.render_needed_;
        }

        [[nodiscard]] std::string summaryValue(const char* label) const {
            auto* const root = document_->GetElementById("vram-hud-summary");
            if (root == nullptr)
                return {};
            for (int i = 0; i < root->GetNumChildren(); ++i) {
                auto* const row = root->GetChild(i);
                if (row->GetNumChildren() < 2 || std::string(row->GetChild(0)->GetInnerRML()) != label)
                    continue;
                return std::string(row->GetChild(1)->GetInnerRML());
            }
            return {};
        }

        [[nodiscard]] Rml::Vector2i centerOf(const char* id) const {
            auto* const element = document_->GetElementById(id);
            EXPECT_NE(element, nullptr) << id;
            if (!element)
                return {};
            const auto offset = element->GetAbsoluteOffset(Rml::BoxArea::Border);
            const auto size = element->GetBox().GetSize(Rml::BoxArea::Border);
            EXPECT_GT(size.x, 0.0f) << id;
            EXPECT_GT(size.y, 0.0f) << id;
            return {static_cast<int>(offset.x + size.x * 0.5f),
                    static_cast<int>(offset.y + size.y * 0.5f)};
        }

        void press(const Rml::Vector2i at) {
            context_->ProcessMouseMove(at.x, at.y, 0);
            context_->ProcessMouseButtonDown(0, 0);
            context_->Update();
        }

        void moveTo(const Rml::Vector2i at) {
            context_->ProcessMouseMove(at.x, at.y, 0);
            context_->Update();
        }

        void release() {
            context_->ProcessMouseButtonUp(0, 0);
            context_->Update();
        }

        void dragBy(const Rml::Vector2i from, const Rml::Vector2i delta) {
            press(from);
            // Small steps like real pointer motion, so the dragged header stays
            // under the pointer, then one settled frame before the release.
            constexpr int kStepPx = 3;
            const int steps = std::max(std::abs(delta.x), std::abs(delta.y)) / kStepPx;
            for (int step = 1; step <= steps; ++step)
                moveTo({from.x + delta.x * step / steps, from.y + delta.y * step / steps});
            context_->Update();
            release();
        }

        void click(const Rml::Vector2i at) {
            press(at);
            release();
        }

        static inline NullRenderInterface* render_interface_ = nullptr;
        std::filesystem::path home_root_;
        std::unique_ptr<ScopedEnvironmentVariable> home_;
        lfs::event::HandlerId toggle_handler_ = 0;
        int toggles_ = 0;
        Rml::Context* context_ = nullptr;
        Rml::ElementDocument* document_ = nullptr;
        std::unique_ptr<VramHudOverlay> overlay_;
    };

    TEST_F(VramHudHeaderClickTest, ResizingAfterBottomCornerSnapDoesNotShrinkEveryFrame) {
        setExpanded(true);
        dragBy(centerOf("vram-hud-resize"), {180, 160});
        for (const bool right : {false, true}) {
            setExpanded(false);
            press(centerOf("perf-hud-strip-header"));
            moveTo({right ? 2000 : -2000, 2000});
            release();
            setExpanded(true);
            auto* root = document_->GetElementById("vram-hud-overlay");
            const auto origin = root->GetAbsoluteOffset(Rml::BoxArea::Border);
            const auto before = root->GetBox().GetSize(Rml::BoxArea::Border);
            const auto from = centerOf("vram-hud-resize");
            press(from);
            for (int delta = 4; delta <= 40; delta += 4) {
                moveTo({from.x + delta, from.y + delta});
                (void)overlay_->initializeGeometryAfterLayout();
                context_->Update();
                const auto size = root->GetBox().GetSize(Rml::BoxArea::Border);
                EXPECT_EQ(root->GetAbsoluteOffset(Rml::BoxArea::Border), origin);
                EXPECT_GE(size.x, before.x - 0.1f);
                EXPECT_GE(size.y, before.y - 0.1f);
            }
            release();
            LayoutState saved;
            saved.load();
            EXPECT_EQ(saved.vram_hud_snap_corner, right ? 4 : 3);
            const auto before_inward = root->GetBox().GetSize(Rml::BoxArea::Border);
            const auto handle = centerOf("vram-hud-resize");
            dragBy(handle, {-40, -40});
            const auto resized = root->GetBox().GetSize(Rml::BoxArea::Border);
            EXPECT_LT(resized.x, before_inward.x);
            EXPECT_LT(resized.y, before_inward.y);
            const auto anchored = root->GetAbsoluteOffset(Rml::BoxArea::Border);
            EXPECT_NEAR(anchored.y + resized.y, 600, 0.1f);
            EXPECT_NEAR(right ? anchored.x + resized.x : anchored.x, right ? 800 : 0, 0.1f);
            setExpanded(false);
            const auto compact = root->GetBox().GetSize(Rml::BoxArea::Border);
            const auto compact_origin = root->GetAbsoluteOffset(Rml::BoxArea::Border);
            EXPECT_NEAR(compact_origin.y + compact.y, 600, 0.1f);
            EXPECT_NEAR(right ? compact_origin.x + compact.x : compact_origin.x, right ? 800 : 0, 0.1f);
            setExpanded(true);
            EXPECT_EQ(root->GetBox().GetSize(Rml::BoxArea::Border), resized);
            saved.load();
            EXPECT_EQ(saved.vram_hud_snap_corner, right ? 4 : 3);
        }
    }

    TEST_F(VramHudHeaderClickTest, BottomCornersSurviveExpansionAndViewportChanges) {
        for (const bool right : {false, true}) {
            press(centerOf("perf-hud-strip-header"));
            moveTo({right ? 2000 : -2000, 2000});
            release();
            auto* root = document_->GetElementById("vram-hud-overlay");
            const auto compact_position = root->GetAbsoluteOffset(Rml::BoxArea::Border);
            LayoutState saved;
            saved.load();
            EXPECT_EQ(saved.vram_hud_snap_corner, right ? 4 : 3);
            for (int i = 0; i < 2; ++i) {
                setExpanded(true);
                const auto expanded_position = root->GetAbsoluteOffset(Rml::BoxArea::Border);
                EXPECT_LE(expanded_position.x, compact_position.x);
                EXPECT_LT(expanded_position.y, compact_position.y);
                setExpanded(false);
                EXPECT_EQ(root->GetAbsoluteOffset(Rml::BoxArea::Border), compact_position);
            }
            overlay_->setViewportGeometry(200, 0, 340, 180);
            EXPECT_TRUE(overlay_->initializeGeometryAfterLayout());
            context_->Update();
            EXPECT_LE(root->GetAbsoluteOffset(Rml::BoxArea::Border).x, compact_position.x);
            setExpanded(true);
            EXPECT_LE(root->GetBox().GetSize(Rml::BoxArea::Border).x, 340);
            setExpanded(false);
            overlay_->setViewportGeometry(0, 0, 800, 600);
            EXPECT_TRUE(overlay_->initializeGeometryAfterLayout());
            context_->Update();
            EXPECT_EQ(root->GetAbsoluteOffset(Rml::BoxArea::Border), compact_position);
            setExpanded(true);
            EXPECT_FLOAT_EQ(root->GetBox().GetSize(Rml::BoxArea::Border).x, saved.vram_hud_width);
            EXPECT_FLOAT_EQ(root->GetBox().GetSize(Rml::BoxArea::Border).y, saved.vram_hud_height);
            setExpanded(false);
            LayoutState after;
            after.load();
            EXPECT_FLOAT_EQ(after.vram_hud_x, saved.vram_hud_x);
            EXPECT_FLOAT_EQ(after.vram_hud_y, saved.vram_hud_y);
            EXPECT_FLOAT_EQ(after.vram_hud_width, saved.vram_hud_width);
            EXPECT_FLOAT_EQ(after.vram_hud_height, saved.vram_hud_height);
            EXPECT_FALSE(overlay_->initializeGeometryAfterLayout());
        }
    }

    TEST_F(VramHudHeaderClickTest, UnsnappedPositionReturnsAfterExpandingAndCollapsing) {
        const auto start = centerOf("perf-hud-strip-header");
        press(start);
        moveTo({start.x + 380, start.y + 400});
        release();
        auto* root = document_->GetElementById("vram-hud-overlay");
        const auto position = root->GetAbsoluteOffset(Rml::BoxArea::Border);
        LayoutState saved;
        saved.load();
        EXPECT_EQ(saved.vram_hud_snap_corner, 0);
        setExpanded(true);
        EXPECT_LT(root->GetAbsoluteOffset(Rml::BoxArea::Border).y, position.y);
        setExpanded(false);
        EXPECT_EQ(root->GetAbsoluteOffset(Rml::BoxArea::Border), position);
    }

    TEST_F(VramHudHeaderClickTest, SnappingUsesTheSameBorderExtentAsDisplayGeometry) {
        auto* strip = document_->GetElementById("perf-hud-strip");
        strip->SetProperty("padding", "5px");
        strip->SetProperty("border-width", "2px");
        context_->Update();
        EXPECT_TRUE(overlay_->initializeGeometryAfterLayout());
        context_->Update();
        press(centerOf("perf-hud-strip-header"));
        moveTo({2000, 2000});
        release();
        auto* root = document_->GetElementById("vram-hud-overlay");
        const auto before = root->GetAbsoluteOffset(Rml::BoxArea::Border);
        const auto extent = strip->GetBox().GetSize(Rml::BoxArea::Border);
        EXPECT_FLOAT_EQ(before.x + extent.x, 800);
        EXPECT_FLOAT_EQ(before.y + extent.y, 600);
        setExpanded(true);
        setExpanded(false);
        EXPECT_EQ(root->GetAbsoluteOffset(Rml::BoxArea::Border), before);
    }

    TEST_F(VramHudHeaderClickTest, CompactStripUsesItsContainersLeftEdge) {
        std::ifstream file(std::filesystem::path(PROJECT_ROOT_PATH) /
                           "src/visualizer/gui/rmlui/resources/viewport_overlay.rcss");
        ASSERT_TRUE(file);
        const std::string css{std::istreambuf_iterator<char>(file), {}};
        document_->SetStyleSheetContainer(Rml::Factory::InstanceStyleSheetString(css));
        auto* root = document_->GetElementById("vram-hud-overlay");
        auto* strip = document_->GetElementById("perf-hud-strip");
        root->SetClass("vram-hud-overlay", true);
        strip->SetClass("perf-hud-strip", true);
        root->SetProperty("width", "620px");
        root->SetProperty("left", "0px");
        root->SetProperty("top", "0px");
        context_->Update();
        EXPECT_NEAR(strip->GetAbsoluteOffset(Rml::BoxArea::Border).x, root->GetAbsoluteOffset(Rml::BoxArea::Border).x, 0.1f);
        EXPECT_NEAR(strip->GetAbsoluteOffset(Rml::BoxArea::Border).y, root->GetAbsoluteOffset(Rml::BoxArea::Border).y, 0.1f);
    }

    TEST_F(VramHudHeaderClickTest, DraggingCompactStripMovesItWithoutExpanding) {
        auto* const root = document_->GetElementById("vram-hud-overlay");
        ASSERT_NE(root, nullptr);
        const auto start = root->GetAbsoluteOffset(Rml::BoxArea::Border);

        dragBy(centerOf("perf-hud-strip-header"), {0, 120});

        const auto moved = root->GetAbsoluteOffset(Rml::BoxArea::Border);
        EXPECT_NEAR(moved.y - start.y, 120.0f, 1.0f);
        EXPECT_EQ(toggles_, 0);
    }

    TEST_F(VramHudHeaderClickTest, ClickingCompactStripStillExpands) {
        click(centerOf("perf-hud-strip-header"));
        EXPECT_EQ(toggles_, 1);

        // A press that wobbles inside the click slop is still a click.
        const auto at = centerOf("perf-hud-strip-header");
        press(at);
        moveTo({at.x + 1, at.y + 1});
        release();
        EXPECT_EQ(toggles_, 2);
    }

    TEST_F(VramHudHeaderClickTest, CollapseClickAfterExpandedHeaderDragIsNotSwallowed) {
        setExpanded(true);

        dragBy(centerOf("vram-hud-header"), {0, 80});
        EXPECT_EQ(toggles_, 0);

        click(centerOf("perf-hud-collapse"));
        EXPECT_EQ(toggles_, 1);
    }

    TEST_F(VramHudHeaderClickTest, DisabledTrackingShowsProcessMemoryAndNotice) {
        VramHudOverlay::State state;
        state.perf_hud.visible = true;
        state.perf_hud.expanded = true;
        state.perf_hud.snapshot = std::make_shared<AppStore::PerfHudSnapshot>();
        EXPECT_TRUE(publish(state));

        EXPECT_FALSE(document_->GetElementById("perf-hud-tracking-off")->IsClassSet("hidden"));
        EXPECT_FALSE(document_->GetElementById("vram-hud-panel-overview")->IsClassSet("hidden"));
        EXPECT_TRUE(document_->GetElementById("vram-hud-panel-ledger")->IsClassSet("hidden"));
        EXPECT_FALSE(summaryValue("Process").empty());
        EXPECT_FALSE(summaryValue("GPU used (all procs)").empty());
        EXPECT_EQ(summaryValue("CUDA pool used"), "—");

        int idle_requests = 0;
        for (int tick = 0; tick < 12; ++tick) {
            auto repeated = state;
            auto snapshot = std::make_shared<AppStore::PerfHudSnapshot>(*state.perf_hud.snapshot);
            snapshot->ram_used_bytes += static_cast<std::size_t>(tick + 1);
            repeated.perf_hud.snapshot = std::move(snapshot);
            idle_requests += publish(std::move(repeated));
        }
        EXPECT_EQ(idle_requests, 0);

        auto labeled = state;
        labeled.snapshot.process.device_name = "Snapshot GPU";
        EXPECT_TRUE(publish(labeled));
        EXPECT_EQ(std::string(document_->GetElementById("vram-hud-summary")->GetLastChild()->GetInnerRML()),
                  "Snapshot GPU");

        auto changed = state;
        changed.snapshot.enabled = true;
        EXPECT_TRUE(publish(std::move(changed)));
    }

} // namespace lfs::vis::gui
