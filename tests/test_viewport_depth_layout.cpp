/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/gui_focus_state.hpp"
#include "gui/panel_input_utils.hpp"
#include "gui/rml_viewport_overlay.hpp"
#include "input/frame_input_buffer.hpp"
#include "visualizer/app_store.hpp"
#include <RmlUi/Core.h>
#include <RmlUi/Core/ElementDocument.h>
#include <RmlUi/Core/RenderInterface.h>
#include <SDL3/SDL_scancode.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace lfs::vis::gui {
    class ViewportDepthLayoutTestAccess {
    public:
        static void attach(RmlViewportOverlay& overlay, Rml::ElementDocument* document) {
            overlay.document_ = document;
            overlay.rml_context_ = document->GetContext();
            overlay.vp_size_ = {920, 840};
            overlay.toolbar_position_preference_dirty_ = false;
        }
        static bool update(RmlViewportOverlay& overlay) { return overlay.updateToolbarRoots(); }
        static void bindStore(RmlViewportOverlay& overlay) { overlay.bindReactiveStore(); }
        static void releaseStore(RmlViewportOverlay& overlay) {
            overlay.gt_metrics_config_subscription_.reset();
            overlay.camera_metrics_subscription_.reset();
            overlay.document_sync_subscriptions_.clear();
        }
        static void setToolbarPosition(RmlViewportOverlay& overlay, std::string position) {
            overlay.viewport_toolbar_position_ = std::move(position);
            overlay.applied_viewport_toolbar_position_.clear();
            overlay.applyToolbarPosition();
        }
        static bool updateOverlayPanels(RmlViewportOverlay& overlay) {
            return overlay.updateViewportOverlayPanelLayout();
        }
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
        static void SetUpTestSuite() {
            ASSERT_TRUE(Rml::Initialise());
            ASSERT_TRUE(Rml::LoadFontFace((std::filesystem::path(PROJECT_ROOT_PATH) /
                                           "src/visualizer/gui/assets/fonts/Inter-Regular.ttf")
                                              .string()));
        }
        static void TearDownTestSuite() { Rml::Shutdown(); }
        void SetUp() override {
            context_ = Rml::CreateContext("viewport_depth_layout", {920, 840}, &renderer_);
            ASSERT_NE(context_, nullptr);
            const auto resources = std::filesystem::path(PROJECT_ROOT_PATH) / "src/visualizer/gui/rmlui/resources";
            std::ifstream components(resources / "components.rcss");
            ASSERT_TRUE(components.is_open());
            std::ostringstream component_css;
            component_css << components.rdbuf();
            std::ifstream input(resources / "viewport_overlay.rcss");
            ASSERT_TRUE(input.is_open());
            std::ostringstream css;
            css << input.rdbuf();
            document_ = context_->LoadDocumentFromMemory(
                "<rml><head><style>" + component_css.str() + css.str() + "</style></head><body>"
                                                                         "<div id='viewport-content'>"
                                                                         "<div id='primary-toolbar-root' class='panel-toolbar-root'>"
                                                                         "<div id='primary-utility-toolbar' class='toolbar-container toolbar-vertical' style='height:400dp; width:44dp;'>"
                                                                         "<button id='toolbar-select' class='icon-btn'></button>"
                                                                         "<button id='toolbar-translate' class='icon-btn'></button>"
                                                                         "<button id='toolbar-rotate' class='icon-btn'></button>"
                                                                         "<button id='toolbar-scale' class='icon-btn'></button>"
                                                                         "</div></div>"
                                                                         "<div id='secondary-toolbar-root' class='panel-toolbar-root hidden'>"
                                                                         "<div id='secondary-utility-toolbar' class='toolbar-container toolbar-vertical' style='height:400dp; width:44dp;'>"
                                                                         "<button id='secondary-toolbar-select' class='icon-btn'></button>"
                                                                         "<button id='secondary-toolbar-translate' class='icon-btn'></button>"
                                                                         "</div></div>"
                                                                         "<div id='depth-view-block' class='viewport-depth-overlay'>"
                                                                         "<div id='depth-panel' style='width:605dp; height:34dp; flex-shrink:0; pointer-events:auto;'></div>"
                                                                         "</div></div></body></rml>");
            ASSERT_NE(document_, nullptr);
            lfs::vis::gui::ViewportDepthLayoutTestAccess::attach(overlay_, document_);
            document_->Show();
        }
        void TearDown() override {
            if (real_context_)
                Rml::RemoveContext("viewport_depth_layout_real");
            Rml::RemoveContext("viewport_depth_layout");
        }

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
        Rml::Context* real_context_ = nullptr;
        Rml::ElementDocument* document_ = nullptr;
        float scale_ = 1.0f;
        float divider_ = 0.5f;
        int width_ = 920;
    };

    struct OverlayKeyListener : Rml::EventListener {
        int presses = 0;
        void ProcessEvent(Rml::Event& event) override {
            ++presses;
            event.StopPropagation();
        }
    };

    TEST_F(ViewportDepthLayoutTest, NonTextControlReceivesEscapeAwayFromPointer) {
        layout(false);
        auto* field = panel()->AppendChild(document_->CreateElement("div"));
        context_->Update();
        ASSERT_TRUE(field->Focus());
        OverlayKeyListener listener;
        document_->AddEventListener("keydown", &listener, true);
        lfs::vis::gui::guiFocusState() = {};
        lfs::vis::gui::PanelInputState input{};
        input.mouse_x = 800;
        input.mouse_y = 500;
        input.input_events.push_back({.kind = lfs::vis::FrameInputEventKind::KeyDown,
                                      .scancode = SDL_SCANCODE_ESCAPE});
        overlay_.processInput(input);
        EXPECT_EQ(listener.presses, 1);
        EXPECT_FALSE(lfs::vis::gui::guiFocusState().want_text_input);
        document_->RemoveEventListener("keydown", &listener, true);
    }

    TEST_F(ViewportDepthLayoutTest, TextControlStillReceivesKeysAwayFromPointer) {
        layout(false);
        auto* field = panel()->AppendChild(document_->CreateElement("input"));
        field->SetAttribute("type", "text");
        context_->Update();
        ASSERT_TRUE(field->Focus());
        OverlayKeyListener listener;
        document_->AddEventListener("keydown", &listener, true);
        lfs::vis::gui::guiFocusState() = {};
        lfs::vis::gui::PanelInputState input{};
        input.mouse_x = 800;
        input.mouse_y = 500;
        input.input_events.push_back({.kind = lfs::vis::FrameInputEventKind::KeyDown,
                                      .scancode = SDL_SCANCODE_A});
        overlay_.processInput(input);
        EXPECT_EQ(listener.presses, 1);
        EXPECT_TRUE(lfs::vis::gui::guiFocusState().want_text_input);
        document_->RemoveEventListener("keydown", &listener, true);
    }

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

    TEST_F(ViewportDepthLayoutTest, MainToolbarAndViewportPanelsStayReachableAtUiScale) {
        const auto path = std::filesystem::path(PROJECT_ROOT_PATH) /
                          "src/visualizer/gui/rmlui/resources/viewport_overlay.rml";
        std::ifstream input(path);
        std::string markup((std::istreambuf_iterator<char>(input)), {});
        markup.replace(markup.find("<head>"), 6,
                       "<head><link type=\"text/rcss\" href=\"components.rcss\"/>");
        real_context_ = Rml::CreateContext("viewport_depth_layout_real", {1280, 720}, &renderer_);
        ASSERT_NE(real_context_, nullptr);
        auto* const real_document = real_context_->LoadDocumentFromMemory(
            markup, path.string());
        ASSERT_NE(real_document, nullptr);
        real_document->Show();
        for (const char* inactive : {"selection_depth_mode_active", "transform_show_rotate", "transform_show_scale"})
            if (auto* element = real_document->QuerySelector(std::format("[data-if='{}']", inactive)))
                element->SetProperty("display", "none");
        lfs::vis::gui::ViewportDepthLayoutTestAccess::attach(overlay_, real_document);
        const std::array cases{
            std::pair{"transform-block", ".viewport-transform-panel"},
            std::pair{"selection-block", ".viewport-transform-panel"},
            std::pair{"depth-view-block", ".viewport-transform-panel"},
            std::pair{"gt-compare-mode-block", ".viewport-transform-panel"},
            std::pair{"viewport-export-block", ".viewport-transform-panel"},
        };
        const auto intersects = [](Rml::Element* a, Rml::Element* b) {
            const auto ap = a->GetAbsoluteOffset(Rml::BoxArea::Border), as = a->GetBox().GetSize(Rml::BoxArea::Border);
            const auto bp = b->GetAbsoluteOffset(Rml::BoxArea::Border), bs = b->GetBox().GetSize(Rml::BoxArea::Border);
            return ap.x < bp.x + bs.x && ap.x + as.x > bp.x && ap.y < bp.y + bs.y && ap.y + as.y > bp.y;
        };
        const auto hit = [&](Rml::Element* element) {
            const auto p = element->GetAbsoluteOffset(Rml::BoxArea::Border), s = element->GetBox().GetSize(Rml::BoxArea::Border);
            auto* found = real_context_->GetElementAtPoint({p.x + s.x * 0.5f, p.y + s.y * 0.5f});
            for (; found && found != element; found = found->GetParentNode()) {}
            EXPECT_EQ(found, element) << element->GetId() << " " << element->GetClassNames()
                                      << " @ " << p.x << "," << p.y << " size " << s.x << "," << s.y;
        };
        auto* const primary = real_document->GetElementById("primary-utility-toolbar");
        auto* const secondary = real_document->GetElementById("secondary-utility-toolbar");
        auto* const content = real_document->GetElementById("viewport-content");
        ASSERT_NE(primary, nullptr);
        ASSERT_NE(secondary, nullptr);
        ASSERT_NE(content, nullptr);
        for (const float scale : {1.0f, 1.25f, 1.5f, 2.0f})
            for (const char* position : {"centered", "top", "free"})
                for (const int width : {1280, 720})
                    for (const bool split : {false, true}) {
                        SCOPED_TRACE(std::format("scale={} position={} width={} split={}", scale, position, width, split));
                        real_context_->SetDensityIndependentPixelRatio(scale);
                        real_context_->SetDimensions({int(width * scale), int(720 * scale)});
                        content->SetClass("split-view", split);
                        const float half = width * scale * 0.5f;
                        overlay_.setToolbarPanels(0, split ? half : width * scale, 5 * scale,
                                                  split, half, half);
                        lfs::vis::gui::ViewportDepthLayoutTestAccess::setToolbarPosition(overlay_, position);
                        for (const auto& [id, selector] : cases) {
                            auto* block = real_document->GetElementById(id);
                            ASSERT_NE(block, nullptr);
                            for (const auto& item : cases)
                                real_document->GetElementById(item.first)->SetClass("hidden", item.first != id);
                            block->SetClass("hidden", false);
                            real_context_->Update();
                            if (lfs::vis::gui::ViewportDepthLayoutTestAccess::updateOverlayPanels(overlay_))
                                real_context_->Update();
                            auto* panel = block->QuerySelector(selector);
                            ASSERT_NE(panel, nullptr);
                            EXPECT_FALSE(intersects(panel, primary)) << id;
                            if (split)
                                EXPECT_FALSE(intersects(panel, secondary)) << id;
                            Rml::ElementList controls;
                            panel->QuerySelectorAll(controls, "button, input, select");
                            ASSERT_FALSE(controls.empty()) << id;
                            for (auto* control : controls)
                                if (control->IsVisible(true))
                                    hit(control);
                        }
                        Rml::ElementList controls;
                        real_document->QuerySelectorAll(controls, "#primary-utility-toolbar .icon-btn, #primary-utility-toolbar .toolbar-drag-handle, #secondary-utility-toolbar .icon-btn, #secondary-utility-toolbar .toolbar-drag-handle");
                        for (auto* control : controls)
                            if (control->IsVisible(true))
                                hit(control);
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

    TEST_F(ViewportDepthLayoutTest, DISABLED_InputPerUpdateCost) {
        layout(false);
        auto* field = panel()->AppendChild(document_->CreateElement("div"));
        context_->Update();
        ASSERT_TRUE(field->Focus());
        lfs::vis::gui::PanelInputState input{};
        input.mouse_x = 800;
        input.mouse_y = 500;
        for (const bool drag : {false, true}) {
            input.mouse_down[0] = drag;
            std::array<double, 9> timings{};
            for (auto& elapsed : timings) {
                const auto start = std::chrono::steady_clock::now();
                for (int i = 0; i < 20000; ++i) {
                    lfs::vis::gui::guiFocusState() = {};
                    overlay_.processInput(input);
                }
                elapsed = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / 20000;
            }
            std::sort(timings.begin(), timings.end());
            RecordProperty(drag ? "drag_update_us" : "idle_update_us", std::to_string(timings[4]));
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

    std::string viewportGtMetricsMarkup() {
        const auto path = std::filesystem::path(PROJECT_ROOT_PATH) /
                          "src/visualizer/gui/rmlui/resources/viewport_overlay.rml";
        std::ifstream input(path);
        if (!input.is_open())
            return {};
        const std::string file((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
        const std::string opener = "<div id=\"gt-metrics-overlay\"";
        const auto start = file.find(opener);
        if (start == std::string::npos)
            return {};
        int depth = 0;
        for (std::size_t i = start; i < file.size();) {
            const bool div_open = file.compare(i, 4, "<div") == 0 &&
                                  (i + 4 >= file.size() || file[i + 4] == ' ' || file[i + 4] == '>' ||
                                   file[i + 4] == '\n' || file[i + 4] == '\r' || file[i + 4] == '\t');
            if (div_open) {
                ++depth;
                i += 4;
                continue;
            }
            if (file.compare(i, 6, "</div>") == 0) {
                --depth;
                i += 6;
                if (depth == 0)
                    return file.substr(start, i - start);
                continue;
            }
            ++i;
        }
        return {};
    }

    TEST_F(ViewportDepthLayoutTest, PublishedUnavailableCameraMetricsShowsReason) {
        const auto markup = viewportGtMetricsMarkup();
        ASSERT_NE(markup.find("id=\"gt-metrics-training-required\""), std::string::npos);
        ASSERT_NE(markup.find("gt_metrics_training_required"), std::string::npos);
        auto* const metrics_document = context_->LoadDocumentFromMemory(
            "<rml><head></head><body>" + markup + "</body></rml>");
        ASSERT_NE(metrics_document, nullptr);
        metrics_document->Show();
        lfs::vis::gui::ViewportDepthLayoutTestAccess::attach(overlay_, metrics_document);

        struct StoreGuard {
            lfs::vis::gui::RmlViewportOverlay* overlay = nullptr;
            ~StoreGuard() {
                if (overlay)
                    lfs::vis::gui::ViewportDepthLayoutTestAccess::releaseStore(*overlay);
                lfs::vis::app_store().camera_metrics.set(std::nullopt);
                lfs::vis::app_store().gt_metrics_overlay_config.set({});
            }
        } store_guard{&overlay_};

        lfs::vis::app_store().camera_metrics.set(std::nullopt);
        lfs::vis::app_store().gt_metrics_overlay_config.set({});
        (void)lfs::vis::app_store().store().drain_dirty_into_frame();
        lfs::vis::gui::ViewportDepthLayoutTestAccess::bindStore(overlay_);

        lfs::vis::AppStore::GTMetricsOverlayConfig config;
        config.visible = true;
        config.show_ssim = true;
        config.current_camera_id = 17;
        config.x = 24.0f;
        config.y = 32.0f;
        ASSERT_TRUE(lfs::vis::app_store().gt_metrics_overlay_config.set(config));
        ASSERT_TRUE(lfs::vis::app_store().camera_metrics.set(
            lfs::vis::AppStore::CameraMetrics::unavailable(17, 7000, false)));
        ASSERT_TRUE(lfs::vis::app_store().store().drain_dirty_into_frame());

        auto* const training_reason = metrics_document->GetElementById("gt-metrics-training-required");
        auto* const failed_reason = metrics_document->GetElementById("gt-metrics-calculation-failed");
        auto* const psnr = metrics_document->GetElementById("gt-metrics-psnr");
        auto* const overlay = metrics_document->GetElementById("gt-metrics-overlay");
        ASSERT_NE(training_reason, nullptr);
        ASSERT_NE(failed_reason, nullptr);
        ASSERT_NE(psnr, nullptr);
        ASSERT_NE(overlay, nullptr);
        EXPECT_FALSE(overlay->IsClassSet("hidden"));
        EXPECT_FALSE(training_reason->IsClassSet("hidden"));
        EXPECT_TRUE(failed_reason->IsClassSet("hidden"));
        EXPECT_NE(std::string(training_reason->GetInnerRML()).find("gt_metrics_training_required"), std::string::npos);
        EXPECT_EQ(std::string(psnr->GetInnerRML()), "--");

        lfs::vis::AppStore::CameraMetrics available;
        available.camera_id = 17;
        available.iteration = 7000;
        available.psnr = 28.5f;
        available.ssim = 0.91f;
        ASSERT_TRUE(lfs::vis::app_store().camera_metrics.set(available));
        ASSERT_TRUE(lfs::vis::app_store().store().drain_dirty_into_frame());
        EXPECT_TRUE(training_reason->IsClassSet("hidden"));
        EXPECT_TRUE(failed_reason->IsClassSet("hidden"));
        EXPECT_EQ(std::string(psnr->GetInnerRML()), "28.50");
        auto* const ssim = metrics_document->GetElementById("gt-metrics-ssim");
        auto* const ssim_row = metrics_document->GetElementById("gt-metrics-ssim-row");
        ASSERT_NE(ssim, nullptr);
        ASSERT_NE(ssim_row, nullptr);
        EXPECT_FALSE(ssim_row->IsClassSet("hidden"));
        EXPECT_EQ(std::string(ssim->GetInnerRML()), "0.9100");

        ASSERT_TRUE(lfs::vis::app_store().camera_metrics.set(
            lfs::vis::AppStore::CameraMetrics::unavailable(17, 7000, true)));
        ASSERT_TRUE(lfs::vis::app_store().store().drain_dirty_into_frame());
        EXPECT_TRUE(training_reason->IsClassSet("hidden"));
        EXPECT_FALSE(failed_reason->IsClassSet("hidden"));
        EXPECT_NE(std::string(failed_reason->GetInnerRML()).find("gt_metrics_calculation_failed"), std::string::npos);
        EXPECT_EQ(std::string(psnr->GetInnerRML()), "--");
    }
} // namespace
