/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/event_bridge/event_bridge.hpp"
#include "core/events.hpp"
#include "gui/vram_hud_overlay.hpp"

#include <RmlUi/Core.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <gtest/gtest.h>
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
        </style></head><body>
            <div id="vram-hud-overlay">
                <div id="perf-hud-strip">
                    <div id="perf-hud-strip-header" data-perf-toggle-expanded="true"></div>
                </div>
                <div id="perf-hud-card">
                    <div id="vram-hud-header">
                        <div id="perf-hud-collapse" data-perf-toggle-expanded="true"></div>
                    </div>
                </div>
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
            overlay_->setState(std::move(state));
            context_->Update();
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

} // namespace lfs::vis::gui
