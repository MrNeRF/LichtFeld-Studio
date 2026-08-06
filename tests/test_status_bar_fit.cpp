/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "gui/rml_status_bar.hpp"

#include <RmlUi/Core.h>
#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/ElementDocument.h>
#include <RmlUi/Core/Factory.h>
#include <RmlUi/Core/RenderInterface.h>

#include <cassert>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <vector>

namespace lfs::vis::gui {

    class RmlStatusBarTestAccess {
    public:
        static void attach(RmlStatusBar& status_bar,
                           Rml::Context* context,
                           Rml::ElementDocument* document) {
            assert(context);
            assert(document);
            status_bar.rml_context_ = context;
            status_bar.document_ = document;
            status_bar.fit_level_ = 0;
            status_bar.applyFitLevel(0);
        }

        static int fit(RmlStatusBar& status_bar, const bool allow_expand = false) {
            status_bar.fitToAvailableWidth(allow_expand);
            return status_bar.fit_level_;
        }
    };

} // namespace lfs::vis::gui

namespace {

    class StubRenderInterface final : public Rml::RenderInterface {
    public:
        Rml::CompiledGeometryHandle CompileGeometry(Rml::Span<const Rml::Vertex>,
                                                    Rml::Span<const int>) override {
            return 1;
        }

        void RenderGeometry(Rml::CompiledGeometryHandle,
                            Rml::Vector2f,
                            Rml::TextureHandle) override {}

        void ReleaseGeometry(Rml::CompiledGeometryHandle) override {}

        Rml::TextureHandle LoadTexture(Rml::Vector2i& dimensions,
                                       const Rml::String&) override {
            dimensions = {16, 16};
            return 1;
        }

        Rml::TextureHandle GenerateTexture(Rml::Span<const Rml::byte>,
                                           Rml::Vector2i) override {
            return 1;
        }

        void ReleaseTexture(Rml::TextureHandle) override {}
        void EnableScissorRegion(bool) override {}
        void SetScissorRegion(Rml::Rectanglei) override {}
    };

    struct StatusBarModel {
        std::string mode_text = "Training (Default/3DGS)";
        std::string mode_color = "#ffffff";
        bool show_training = true;
        std::string progress_width = "50%";
        std::string progress_text = "50%";
        std::string step_label = "Step:";
        std::string step_value = "15000/30000";
        std::string loss_label = "Loss:";
        std::string loss_value = "0.1234";
        bool show_eval_metrics = true;
        std::string eval_metrics_value = "PSNR 31.25 / SSIM 0.9876";
        std::string gaussians_label = "Gaussians";
        std::string gaussians_value = "1.25M/1.50M";
        std::string time_value = "1:23:45";
        std::string eta_label = "ETA:";
        std::string eta_value = "2:34:56";
        bool show_splats = false;
        std::string splat_text = "1.25M Gaussians";
        std::string splat_color = "#ffffff";
        bool show_split = false;
        std::string split_mode = "Ground Truth Comparison";
        std::string split_mode_color = "#ffffff";
        std::string split_detail = "Camera 123 / 500";
        bool show_wasd = true;
        std::string wasd_text = "WASD: 100";
        std::string wasd_color = "#ffffff";
        std::string wasd_sep_color = "#ffffff";
        bool show_zoom = true;
        std::string zoom_text = "Zoom: 100";
        std::string zoom_color = "#ffffff";
        std::string zoom_sep_color = "#ffffff";
        std::string account_label = "LichtFeld Account";
        std::string account_tier = "Professional";
        std::string account_tooltip;
        std::string account_color = "#ffffff";
        bool account_show_tier = true;
        bool account_membership_required = false;
        std::string lfs_mem_text = "LFS 12.34 GiB";
        std::string lfs_mem_color = "#ffffff";
        bool show_gpu_model = true;
        bool gpu_panel_active = false;
        std::string gpu_model_text = "NVIDIA GeForce RTX 5090";
        std::string gpu_mem_text = "GPU 18.75/31.99 GiB";
        std::string gpu_mem_color = "#ffffff";
        std::string fps_value = "144";
        std::string fps_color = "#ffffff";
        std::string fps_label = " FPS";
        std::string git_commit = "abcdef12";
        bool show_status_message = false;
        std::string status_message_text = "A long transient status message that must remain on one line";
        std::string status_message_color = "#ffffff";
    };

    std::string readResource(const std::string& name) {
        const std::ifstream file(std::filesystem::path(PROJECT_ROOT_PATH) /
                                 "src/visualizer/gui/rmlui/resources" / name);
        std::ostringstream contents;
        contents << file.rdbuf();
        return contents.str();
    }

    void assertNoVerticalOverflow(Rml::Element* element) {
        ASSERT_TRUE(element);
        SCOPED_TRACE(element->GetAddress());
        EXPECT_LE(element->GetScrollHeight(), element->GetClientHeight() + 0.5f);
        for (int i = 0; i < element->GetNumChildren(); ++i)
            assertNoVerticalOverflow(element->GetChild(i));
    }

    void assertFlexSiblingsDoNotOverlap(Rml::Element* element) {
        ASSERT_TRUE(element);
        if (element->IsVisible(true) && element->GetDisplay() == Rml::Style::Display::Flex) {
            bool have_previous = false;
            float previous_right = 0.0f;
            for (int i = 0; i < element->GetNumChildren(); ++i) {
                auto* const child = element->GetChild(i);
                if (!child || !child->IsVisible(true) || child->GetOffsetWidth() <= 0.0f)
                    continue;

                const float left = child->GetAbsoluteOffset(Rml::BoxArea::Border).x;
                SCOPED_TRACE(child->GetAddress());
                if (have_previous)
                    EXPECT_GE(left, previous_right - 0.5f);
                previous_right = left + child->GetOffsetWidth();
                have_previous = true;
            }
        }

        for (int i = 0; i < element->GetNumChildren(); ++i)
            assertFlexSiblingsDoNotOverlap(element->GetChild(i));
    }

    class StatusBarFitTest : public ::testing::Test {
    protected:
        static void SetUpTestSuite() {
            ASSERT_TRUE(Rml::Initialise());
            const auto font_path = std::filesystem::path(PROJECT_ROOT_PATH) /
                                   "src/visualizer/gui/assets/fonts/Inter-Regular.ttf";
            ASSERT_TRUE(Rml::LoadFontFace(font_path.string()));
        }

        static void TearDownTestSuite() {
            Rml::Shutdown();
        }

        void SetUp() override {
            context_ = Rml::CreateContext("status_bar_fit", {2400, 22}, &render_interface_);
            ASSERT_TRUE(context_);
            context_->SetDensityIndependentPixelRatio(1.0f);

            auto constructor = context_->CreateDataModel("status_bar");
            ASSERT_TRUE(static_cast<bool>(constructor));
            bool bound = true;
            bound &= constructor.Bind("mode_text", &model_.mode_text);
            bound &= constructor.Bind("mode_color", &model_.mode_color);
            bound &= constructor.Bind("show_training", &model_.show_training);
            bound &= constructor.Bind("progress_width", &model_.progress_width);
            bound &= constructor.Bind("progress_text", &model_.progress_text);
            bound &= constructor.Bind("step_label", &model_.step_label);
            bound &= constructor.Bind("step_value", &model_.step_value);
            bound &= constructor.Bind("loss_label", &model_.loss_label);
            bound &= constructor.Bind("loss_value", &model_.loss_value);
            bound &= constructor.Bind("show_eval_metrics", &model_.show_eval_metrics);
            bound &= constructor.Bind("eval_metrics_value", &model_.eval_metrics_value);
            bound &= constructor.Bind("gaussians_label", &model_.gaussians_label);
            bound &= constructor.Bind("gaussians_value", &model_.gaussians_value);
            bound &= constructor.Bind("time_value", &model_.time_value);
            bound &= constructor.Bind("eta_label", &model_.eta_label);
            bound &= constructor.Bind("eta_value", &model_.eta_value);
            bound &= constructor.Bind("show_splats", &model_.show_splats);
            bound &= constructor.Bind("splat_text", &model_.splat_text);
            bound &= constructor.Bind("splat_color", &model_.splat_color);
            bound &= constructor.Bind("show_split", &model_.show_split);
            bound &= constructor.Bind("split_mode", &model_.split_mode);
            bound &= constructor.Bind("split_mode_color", &model_.split_mode_color);
            bound &= constructor.Bind("split_detail", &model_.split_detail);
            bound &= constructor.Bind("show_wasd", &model_.show_wasd);
            bound &= constructor.Bind("wasd_text", &model_.wasd_text);
            bound &= constructor.Bind("wasd_color", &model_.wasd_color);
            bound &= constructor.Bind("wasd_sep_color", &model_.wasd_sep_color);
            bound &= constructor.Bind("show_zoom", &model_.show_zoom);
            bound &= constructor.Bind("zoom_text", &model_.zoom_text);
            bound &= constructor.Bind("zoom_color", &model_.zoom_color);
            bound &= constructor.Bind("zoom_sep_color", &model_.zoom_sep_color);
            bound &= constructor.Bind("account_label", &model_.account_label);
            bound &= constructor.Bind("account_tier", &model_.account_tier);
            bound &= constructor.Bind("account_tooltip", &model_.account_tooltip);
            bound &= constructor.Bind("account_color", &model_.account_color);
            bound &= constructor.Bind("account_show_tier", &model_.account_show_tier);
            bound &= constructor.Bind("account_membership_required", &model_.account_membership_required);
            bound &= constructor.Bind("lfs_mem_text", &model_.lfs_mem_text);
            bound &= constructor.Bind("lfs_mem_color", &model_.lfs_mem_color);
            bound &= constructor.Bind("show_gpu_model", &model_.show_gpu_model);
            bound &= constructor.Bind("gpu_panel_active", &model_.gpu_panel_active);
            bound &= constructor.Bind("gpu_model_text", &model_.gpu_model_text);
            bound &= constructor.Bind("gpu_mem_text", &model_.gpu_mem_text);
            bound &= constructor.Bind("gpu_mem_color", &model_.gpu_mem_color);
            bound &= constructor.Bind("fps_value", &model_.fps_value);
            bound &= constructor.Bind("fps_color", &model_.fps_color);
            bound &= constructor.Bind("fps_label", &model_.fps_label);
            bound &= constructor.Bind("git_commit", &model_.git_commit);
            bound &= constructor.Bind("show_status_message", &model_.show_status_message);
            bound &= constructor.Bind("status_message_text", &model_.status_message_text);
            bound &= constructor.Bind("status_message_color", &model_.status_message_color);
            ASSERT_TRUE(bound);

            const auto document_path = std::filesystem::path(PROJECT_ROOT_PATH) /
                                       "src/visualizer/gui/rmlui/resources/statusbar.rml";
            document_ = context_->LoadDocument(document_path.string());
            ASSERT_TRUE(document_);
            // The application composes the shared components sheet ahead of the status bar sheet
            // (rml_theme::applyTheme), so class collisions between the two only appear when both
            // are present.
            auto sheet = Rml::Factory::InstanceStyleSheetString(
                readResource("components.rcss") + "\n" + readResource("statusbar.rcss"));
            ASSERT_TRUE(sheet);
            document_->SetStyleSheetContainer(std::move(sheet));
            ASSERT_TRUE(document_->SetProperty("height", "22px"));
            document_->Show();
            context_->Update();
            lfs::vis::gui::RmlStatusBarTestAccess::attach(status_bar_, context_, document_);
        }

        void TearDown() override {
            ASSERT_TRUE(Rml::RemoveContext("status_bar_fit"));
            context_ = nullptr;
            document_ = nullptr;
        }

        inline static StubRenderInterface render_interface_;
        Rml::Context* context_ = nullptr;
        Rml::ElementDocument* document_ = nullptr;
        StatusBarModel model_;
        lfs::vis::gui::RmlStatusBar status_bar_;
    };

    TEST_F(StatusBarFitTest, KeepsSingleLineNonOverlappingLayoutAcrossWidths) {
        const std::vector<int> widths = {2400, 1600, 1200, 900, 700, 500, 320};
        int previous_fit_level = 0;

        for (const int width : widths) {
            SCOPED_TRACE(width);
            context_->SetDimensions({width, 22});
            context_->Update();
            const int fit_level = lfs::vis::gui::RmlStatusBarTestAccess::fit(status_bar_);

            if (width == widths.front())
                EXPECT_EQ(fit_level, 0);
            EXPECT_GE(fit_level, previous_fit_level);
            assertNoVerticalOverflow(document_);
            assertFlexSiblingsDoNotOverlap(document_);
            previous_fit_level = fit_level;
        }

        context_->SetDimensions({widths.front(), 22});
        context_->Update();
        const int expanded_fit_level =
            lfs::vis::gui::RmlStatusBarTestAccess::fit(status_bar_, true);
        EXPECT_LT(expanded_fit_level, previous_fit_level);
        assertNoVerticalOverflow(document_);
        assertFlexSiblingsDoNotOverlap(document_);
    }

} // namespace
