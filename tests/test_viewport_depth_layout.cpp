/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rml_viewport_overlay.hpp"
#include <RmlUi/Core.h>
#include <RmlUi/Core/ElementDocument.h>
#include <RmlUi/Core/RenderInterface.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <sstream>

namespace lfs::vis::gui {
    class ViewportDepthLayoutTestAccess {
    public:
        static void attach(RmlViewportOverlay& overlay, Rml::ElementDocument* document) {
            overlay.document_ = document;
            overlay.toolbar_position_preference_dirty_ = false;
        }
        static bool update(RmlViewportOverlay& overlay) { return overlay.updateToolbarRoots(); }
    };
} // namespace lfs::vis::gui

namespace {
    class DepthLayoutRenderer final : public Rml::RenderInterface {
    public:
        Rml::CompiledGeometryHandle CompileGeometry(Rml::Span<const Rml::Vertex>, Rml::Span<const int>) override { return 1; }
        void RenderGeometry(Rml::CompiledGeometryHandle, Rml::Vector2f, Rml::TextureHandle) override {}
        void ReleaseGeometry(Rml::CompiledGeometryHandle) override {}
        Rml::TextureHandle LoadTexture(Rml::Vector2i&, const Rml::String&) override { return 0; }
        Rml::TextureHandle GenerateTexture(Rml::Span<const Rml::byte>, Rml::Vector2i) override { return 0; }
        void ReleaseTexture(Rml::TextureHandle) override {}
        void EnableScissorRegion(bool) override {}
        void SetScissorRegion(Rml::Rectanglei) override {}
    };

    class ViewportDepthLayoutTest : public ::testing::Test {
    protected:
        static void SetUpTestSuite() { ASSERT_TRUE(Rml::Initialise()); }
        static void TearDownTestSuite() { Rml::Shutdown(); }
        void SetUp() override {
            context_ = Rml::CreateContext("viewport_depth_layout", {920, 840}, &renderer_);
            ASSERT_NE(context_, nullptr);
            const auto resources = std::filesystem::path(PROJECT_ROOT_PATH) / "src/visualizer/gui/rmlui/resources";
            std::ifstream input(resources / "viewport_overlay.rcss");
            ASSERT_TRUE(input.is_open());
            std::ostringstream css;
            css << input.rdbuf();
            document_ = context_->LoadDocumentFromMemory(
                "<rml><head><style>" + css.str() + "</style></head><body>"
                                                   "<div id='viewport-content'>"
                                                   "<div id='primary-toolbar-root' class='panel-toolbar-root'></div>"
                                                   "<div id='secondary-toolbar-root' class='panel-toolbar-root hidden'>"
                                                   "<div id='secondary-utility-toolbar' class='toolbar-container toolbar-vertical' style='height:400dp; width:44dp;'></div></div>"
                                                   "<div id='depth-view-block' class='viewport-depth-overlay'>"
                                                   "<div id='depth-panel' style='width:605dp; height:34dp; flex-shrink:0; pointer-events:auto;'></div>"
                                                   "</div></div></body></rml>");
            ASSERT_NE(document_, nullptr);
            lfs::vis::gui::ViewportDepthLayoutTestAccess::attach(overlay_, document_);
            document_->Show();
        }
        void TearDown() override { Rml::RemoveContext("viewport_depth_layout"); }

        void layout(bool split, float scale = 1.0f, float divider = 0.5f, int width = 920, int height = 840) {
            scale_ = scale;
            width_ = width;
            divider_ = divider;
            context_->SetDensityIndependentPixelRatio(scale);
            context_->SetDimensions({int(width * scale), int(height * scale)});
            const float primary_width = width * scale * (split ? divider : 1.0f);
            overlay_.setToolbarPanels(0, primary_width, 5 * scale, split,
                                      primary_width, width * scale - primary_width);
            context_->Update();
        }
        Rml::Element* panel() { return document_->GetElementById("depth-panel"); }
        float top() { return panel()->GetAbsoluteOffset(Rml::BoxArea::Border).y; }
        void expectClearOfGizmo() {
            const auto pos = panel()->GetAbsoluteOffset(Rml::BoxArea::Border);
            const auto size = panel()->GetBox().GetSize(Rml::BoxArea::Border);
            const float right = width_ * scale_ * divider_ - 10 * scale_;
            const float left = right - 95 * scale_;
            EXPECT_TRUE(pos.x + size.x <= left || pos.x >= right ||
                        pos.y >= 105 * scale_ || pos.y + size.y <= 10 * scale_);
            // The top (+Y) circle must fall through to the viewport, not the depth field.
            EXPECT_NE(context_->GetElementAtPoint({right - 47.5f * scale_, 25 * scale_}), panel());
        }
        DepthLayoutRenderer renderer_;
        lfs::vis::gui::RmlViewportOverlay overlay_;
        Rml::Context* context_ = nullptr;
        Rml::ElementDocument* document_ = nullptr;
        float scale_ = 1.0f;
        float divider_ = 0.5f;
        int width_ = 920;
    };

    TEST_F(ViewportDepthLayoutTest, SplitDepthBarLeavesAxisMarkersReachable) {
        for (const float scale : {1.0f, 1.5f, 2.0f}) {
            SCOPED_TRACE(scale);
            layout(true, scale);
            expectClearOfGizmo();
        }
    }

    TEST_F(ViewportDepthLayoutTest, SingleViewPlacementIsUnchanged) {
        for (const float scale : {1.0f, 1.5f, 2.0f}) {
            layout(false, scale);
            EXPECT_FLOAT_EQ(top(), 5 * scale);
            const auto pos = panel()->GetAbsoluteOffset(Rml::BoxArea::Border);
            const auto size = panel()->GetBox().GetSize(Rml::BoxArea::Border);
            EXPECT_NEAR(pos.x + size.x * 0.5f, 460 * scale, 1.0f);
        }
    }

    TEST_F(ViewportDepthLayoutTest, SelectionAndTransformBarsAlsoLeaveAxisMarkersReachable) {
        auto* bar = document_->GetElementById("depth-view-block");
        bar->SetId("shared-bar-test");
        for (const auto* classes : {"viewport-selection-overlay", "viewport-transform-overlay"}) {
            bar->SetClassNames(classes);
            for (const float scale : {1.0f, 1.5f, 2.0f}) {
                SCOPED_TRACE(classes);
                SCOPED_TRACE(scale);
                layout(true, scale);
                expectClearOfGizmo();
                layout(false, scale);
                EXPECT_FLOAT_EQ(top(), 5 * scale);
            }
        }
    }

    TEST_F(ViewportDepthLayoutTest, SharedBarsAvoidCenteredToolRailsAtHighUiScale) {
        auto* bar = document_->GetElementById("depth-view-block");
        for (const auto* classes : {"viewport-depth-overlay", "viewport-selection-overlay", "viewport-transform-overlay"}) {
            bar->SetId(std::string(classes) == "viewport-depth-overlay" ? "depth-view-block" : "shared-bar-test");
            bar->SetClassNames(classes);
            layout(true, 1.5f, 0.5f, 800, 668);
            auto* rail = document_->GetElementById("secondary-utility-toolbar");
            const auto rail_pos = rail->GetAbsoluteOffset(Rml::BoxArea::Border);
            const auto rail_size = rail->GetBox().GetSize(Rml::BoxArea::Border);
            const auto bar_pos = panel()->GetAbsoluteOffset(Rml::BoxArea::Border);
            const auto bar_size = panel()->GetBox().GetSize(Rml::BoxArea::Border);
            EXPECT_TRUE(bar_pos.x + bar_size.x <= rail_pos.x || bar_pos.x >= rail_pos.x + rail_size.x ||
                        bar_pos.y + bar_size.y <= rail_pos.y || bar_pos.y >= rail_pos.y + rail_size.y);
            EXPECT_LE(bar_pos.y + bar_size.y, 668 * 1.5f);
            expectClearOfGizmo();
        }
    }

    TEST_F(ViewportDepthLayoutTest, ComparisonAndExportOverlaysKeepExistingPlacement) {
        auto* bar = document_->GetElementById("depth-view-block");
        for (const auto* id : {"gt-compare-mode-block", "viewport-export-block"}) {
            bar->SetId(id);
            layout(true);
            EXPECT_FLOAT_EQ(top(), 5);
        }
    }

    TEST_F(ViewportDepthLayoutTest, SplitResizeAndExitRestoreLayoutWithoutRepeatedInvalidation) {
        for (const int width : {700, 920, 1400}) {
            for (const float divider : {0.3f, 0.5f, 0.7f}) {
                layout(true, 1.0f, divider, width);
                expectClearOfGizmo();
                EXPECT_FALSE(lfs::vis::gui::ViewportDepthLayoutTestAccess::update(overlay_));
                layout(false, 1.0f, divider, width);
                EXPECT_FLOAT_EQ(top(), 5);
            }
        }
    }

    TEST_F(ViewportDepthLayoutTest, DISABLED_PerUpdateCost) {
        // Run explicitly before/after under the shared benchmark lock.
        constexpr int updates = 200;
        std::array<double, 9> timings{};
        auto* rail = document_->GetElementById("secondary-utility-toolbar");
        rail->GetParentNode()->RemoveChild(rail);
        layout(true);
        for (auto& elapsed : timings) {
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < updates; ++i)
                layout(true, 1.0f, i % 2 ? 0.49f : 0.51f);
            elapsed = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / updates;
        }
        std::sort(timings.begin(), timings.end());
        RecordProperty("median_update_us", std::to_string(timings[timings.size() / 2]));
        EXPECT_FALSE(lfs::vis::gui::ViewportDepthLayoutTestAccess::update(overlay_));
    }
} // namespace
