/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "core/event_bridge/localization_manager.hpp"
#include "core/tensor_backend.hpp"
#include "gui/gui_input.hpp"
#include "gui/rml_status_bar.hpp"
#include "gui/rmlui/rmlui_manager.hpp"
#include "gui/rmlui/rmlui_system_interface.hpp"
#include "gui/status_bar_training.hpp"
#include "rendering/viewport_artifact_service.hpp"
#include "visualizer/app_store.hpp"
#include "visualizer/preferences.hpp"

#include <RmlUi/Core.h>
#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/ElementDocument.h>
#include <RmlUi/Core/Factory.h>
#include <RmlUi/Core/RenderInterface.h>

#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace lfs::vis::gui {

    TEST(StatusBarBackendTest, UsesActiveTrainerBeforeStoredSession) {
        using lfs::core::param::RasterBackendId;
        EXPECT_EQ(trainingBackendStatusLabel(RasterBackendId::ThreeDGUT, "3dgs"), "3DGUT");
        EXPECT_EQ(trainingBackendStatusLabel(RasterBackendId::ThreeDGS, "3dgut"), "3DGS");
    }

    TEST(StatusBarBackendTest, UsesStoredBackendWithoutInventingDefault) {
        EXPECT_EQ(trainingBackendStatusLabel(std::nullopt, "3dgut"), "3DGUT");
        EXPECT_EQ(trainingBackendStatusLabel(std::nullopt, "3dgs"), "3DGS");
        EXPECT_TRUE(trainingBackendStatusLabel(std::nullopt, "").empty());
        EXPECT_TRUE(trainingBackendStatusLabel(std::nullopt, "unknown").empty());
    }

    class RmlStatusBarTestAccess {
    public:
        using ModelState = RmlStatusBar::ModelState;
        static ModelState& model(RmlStatusBar& status_bar) { return status_bar.model_; }
        static void setModelHandle(RmlStatusBar& status_bar, Rml::DataModelHandle handle) {
            status_bar.model_handle_ = handle;
        }
        static void updateBackends(RmlStatusBar& status_bar) {
            status_bar.updateBackendContent();
        }
        static void updateBackends(RmlStatusBar& status_bar, std::optional<lfs::rendering::ViewerBackend> active_backend) {
            status_bar.updateBackendContent(active_backend);
        }
        static void updateUpscaler(RmlStatusBar& bar, const RenderSettings& settings, SceneUpscalerSelection selection) {
            bar.updateUpscalerContent(settings, selection);
        }
        static void setUpscalerExpanded(RmlStatusBar& bar, bool expanded) {
            bar.model_.upscaler_menu_expanded = expanded;
        }
        static bool applyTooltip(RmlStatusBar& status_bar, int width = 2400, int bar_height = 22) {
            return status_bar.applyHoverTooltip(width, bar_height, 700);
        }
        static auto tooltipDeadline(const RmlStatusBar& status_bar) {
            return status_bar.tooltip_.revealDeadline();
        }
        static void detach(RmlStatusBar& status_bar) {
            status_bar.model_handle_ = {};
            status_bar.rml_context_ = nullptr;
            status_bar.document_ = nullptr;
            status_bar.rml_manager_ = nullptr;
        }
        static void attach(RmlStatusBar& status_bar,
                           Rml::Context* context,
                           Rml::ElementDocument* document) {
            assert(context);
            assert(document);
            status_bar.rml_context_ = context;
            status_bar.document_ = document;
            status_bar.fit_level_ = 0;
            status_bar.applyFitLevel(0);
            status_bar.updateFpsReservedWidths();
        }

        static int fit(RmlStatusBar& status_bar, const bool allow_expand = false) {
            status_bar.fitToAvailableWidth(allow_expand);
            return status_bar.fit_level_;
        }

        static void setMcpExpanded(RmlStatusBar& status_bar, const bool expanded) {
            status_bar.model_.mcp_details_expanded = expanded;
        }

        static void bindStore(RmlStatusBar& status_bar) { status_bar.bindReactiveStore(); }
        static void clearRedraw(RmlStatusBar& status_bar) { status_bar.model_dirty_ = false; }
        [[nodiscard]] static bool redrawPending(const RmlStatusBar& status_bar) {
            return status_bar.model_dirty_;
        }

        static bool tooltipPending(const RmlStatusBar& status_bar) {
            return status_bar.tooltip_.revealDeadline().has_value();
        }

        static void prepareTooltip(RmlStatusBar& status_bar, const std::string& text) {
            // Match the available overlay space supplied by render in a 900px window.
            status_bar.tooltip_max_overlay_height_ = 878.0f;
            status_bar.tooltip_target_ = status_bar.document_;
            status_bar.tooltip_text_ = text;
            status_bar.tooltip_.setHover(text, status_bar.document_);
        }

        static void revealTooltip(RmlStatusBar& status_bar, const bool layout = true) {
            if (const auto deadline = status_bar.tooltip_.revealDeadline())
                std::this_thread::sleep_until(*deadline + std::chrono::milliseconds(2));
            status_bar.tooltip_.apply(status_bar.document_, 1200, 889, 2400, 900);
            if (layout)
                status_bar.rml_context_->Update();
        }

        static void positionTooltip(RmlStatusBar& status_bar, const float before, const float after) {
            status_bar.positionTooltip(before, after, false);
        }

        static void trackRenderedFrame(RmlStatusBar& status_bar,
                                       RmlUIManager& manager,
                                       const float bar_x,
                                       const float bar_y) {
            status_bar.rml_manager_ = &manager;
            status_bar.trackRenderedContextFrame(bar_x, bar_y,
                                                 status_bar.overlayHeight());
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

    void populateStatusBarModel(lfs::vis::gui::RmlStatusBarTestAccess::ModelState& model) {
        model.safe_mode = false;
        model.safe_mode_text = "Safe Mode";
        model.mode_text = "Training (Default/3DGS)";
        model.mode_color = "#ffffff";
        model.show_training = true;
        model.progress_miner = false;
        model.miner_raised = false;
        model.miner_step_a = false;
        model.miner_strike = false;
        model.miner_step_b = false;
        model.miner_smoke_1 = false;
        model.miner_smoke_2 = false;
        model.miner_smoke_3 = false;
        model.miner_smoke_4 = false;
        model.miner_smoke_5 = false;
        model.miner_smoke_6 = false;
        model.progress_width = "50%";
        model.progress_text_left = "0dp";
        model.progress_text = "50%";
        model.step_label = "Step:";
        model.step_value = "15000/30000";
        model.loss_label = "Loss:";
        model.loss_value = "0.1234";
        model.show_eval_metrics = true;
        model.eval_metrics_value = "PSNR 31.25 / SSIM 0.9876";
        model.gaussians_label = "Gaussians";
        model.gaussians_value = "1.25M/1.50M";
        model.time_value = "1:23:45";
        model.eta_label = "ETA:";
        model.eta_value = "2:34:56";
        model.show_splats = false;
        model.splat_text = "1.25M Gaussians";
        model.splat_color = "#ffffff";
        model.show_split = false;
        model.split_mode = "Ground Truth Comparison";
        model.split_mode_color = "#ffffff";
        model.split_detail = "Camera 123 / 500";
        model.show_wasd = true;
        model.wasd_text = "WASD: 100";
        model.wasd_color = "#ffffff";
        model.wasd_sep_color = "#ffffff";
        model.show_zoom = true;
        model.zoom_text = "Zoom: 100";
        model.zoom_color = "#ffffff";
        model.zoom_sep_color = "#ffffff";
        model.show_lfs_memory = true;
        model.preview_reduced = false;
        model.preview_reduced_text = "Reduced preview resolution";
        model.input_device = "mouse";
        model.input_device_tooltip = "Mouse navigation";
        model.lfs_mem_text = "LFS 12.34 GiB";
        model.lfs_mem_color = "#ffffff";
        model.show_gpu_model = true;
        model.gpu_panel_active = false;
        model.gpu_model_text = "NVIDIA GeForce RTX 5090";
        model.gpu_mem_text = "GPU 18.75/31.99 GiB";
        model.gpu_mem_color = "#ffffff";
        model.fps_value = "144";
        model.fps_view_value = "144";
        model.fps_ui_label = "UI";
        model.fps_view_label = "3D";
        model.fps_color = "#ffffff";
        model.fps_label = "FPS";
        model.renderer_label = "R";
        model.renderer_value = "Vulkan";
        model.renderer_tooltip = "Scene renderer";
        model.upscaler_label = "U";
        model.upscaler_value = "AMD FSR 3.1";
        model.upscaler_tooltip = "Scene reconstruction";
        model.tensor_label = "T";
        model.tensor_value = "CUDA";
        model.tensor_tooltip = "Tensor compute backend";
        model.git_commit = "abcdef12";
        model.mcp_details_expanded = false;
        model.mcp_summary = "MCP Local";
        model.mcp_details = "http://127.0.0.1:45677/mcp";
        model.mcp_tooltip = "MCP is listening only on this computer";
        model.mcp_color = "#ffffff";
        model.mcp_preferences_label = "Edit";
        model.mcp_server_enabled = true;
        model.mcp_toggle_label = "Turn off";
        model.mcp_total_text = "2 requests";
        model.mcp_success_text = "2 successful";
        model.mcp_error_text = "0 errors";
        model.show_status_message = false;
        model.status_message_text = "A long transient status message that must remain on one line";
        model.status_message_color = "#ffffff";
    }

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

    class ScopedStatusBarHome {
        std::optional<std::string> previous_;
        std::filesystem::path path_;

    public:
        ScopedStatusBarHome() {
            if (const char* value = std::getenv("LFS_HOME"))
                previous_ = value;
            path_ = std::filesystem::temp_directory_path() /
                    ("lfs_status_bar_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
            std::filesystem::create_directories(path_);
#ifdef _WIN32
            (void)_putenv_s("LFS_HOME", path_.string().c_str());
#else
            (void)setenv("LFS_HOME", path_.string().c_str(), 1);
#endif
        }
        ~ScopedStatusBarHome() {
#ifdef _WIN32
            (void)_putenv_s("LFS_HOME", previous_ ? previous_->c_str() : "");
#else
            if (previous_)
                (void)setenv("LFS_HOME", previous_->c_str(), 1);
            else
                (void)unsetenv("LFS_HOME");
#endif
            std::error_code error;
            std::filesystem::remove_all(path_, error);
        }
    };

    class StatusBarFitTest : public ::testing::Test {
    protected:
        static inline lfs::vis::gui::RmlSystemInterface system_interface_{nullptr};
        static inline bool had_localization_ = false;
        static inline std::string previous_language_;

        static void SetUpTestSuite() {
            auto& localization = lfs::event::LocalizationManager::getInstance();
            had_localization_ = localization.hasKey("status_bar.mcp_name");
            previous_language_ = localization.getCurrentLanguage();
            if (!had_localization_)
                ASSERT_TRUE(localization.initialize((std::filesystem::path(PROJECT_ROOT_PATH) /
                                                     "src/visualizer/gui/resources/locales")
                                                        .string()));
            ASSERT_TRUE(localization.setLanguage("en"));
            Rml::SetSystemInterface(&system_interface_);
            ASSERT_TRUE(Rml::Initialise());
            const auto font_path = std::filesystem::path(PROJECT_ROOT_PATH) /
                                   "src/visualizer/gui/assets/fonts/Inter-Regular.ttf";
            ASSERT_TRUE(Rml::LoadFontFace(font_path.string()));
            for (const char* name : {"NotoSansJP-Regular.ttf", "NotoSansKR-Regular.ttf"})
                ASSERT_TRUE(Rml::LoadFontFace((font_path.parent_path() / name).string(), true));
        }

        static void TearDownTestSuite() {
            Rml::Shutdown();
            Rml::SetSystemInterface(nullptr);
            auto& localization = lfs::event::LocalizationManager::getInstance();
            if (had_localization_)
                EXPECT_TRUE(localization.setLanguage(previous_language_));
            else
                localization.reset();
        }

        void SetUp() override {
            populateStatusBarModel(model_);
            context_ = Rml::CreateContext("status_bar_fit", {2400, 22}, &render_interface_);
            ASSERT_TRUE(context_);
            context_->SetDensityIndependentPixelRatio(1.0f);

            auto constructor = context_->CreateDataModel("status_bar");
            ASSERT_TRUE(static_cast<bool>(constructor));
            bool bound = true;
            bound &= constructor.Bind("safe_mode", &model_.safe_mode);
            bound &= constructor.Bind("safe_mode_text", &model_.safe_mode_text);
            bound &= constructor.Bind("mode_text", &model_.mode_text);
            bound &= constructor.Bind("mode_color", &model_.mode_color);
            bound &= constructor.Bind("show_training", &model_.show_training);
            bound &= constructor.Bind("progress_miner", &model_.progress_miner);
            bound &= constructor.Bind("miner_raised", &model_.miner_raised);
            bound &= constructor.Bind("miner_step_a", &model_.miner_step_a);
            bound &= constructor.Bind("miner_strike", &model_.miner_strike);
            bound &= constructor.Bind("miner_step_b", &model_.miner_step_b);
            bound &= constructor.Bind("miner_smoke_1", &model_.miner_smoke_1);
            bound &= constructor.Bind("miner_smoke_2", &model_.miner_smoke_2);
            bound &= constructor.Bind("miner_smoke_3", &model_.miner_smoke_3);
            bound &= constructor.Bind("miner_smoke_4", &model_.miner_smoke_4);
            bound &= constructor.Bind("miner_smoke_5", &model_.miner_smoke_5);
            bound &= constructor.Bind("miner_smoke_6", &model_.miner_smoke_6);
            bound &= constructor.Bind("progress_width", &model_.progress_width);
            bound &= constructor.Bind("progress_text_left", &model_.progress_text_left);
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
            bound &= constructor.Bind("show_lfs_memory", &model_.show_lfs_memory);
            bound &= constructor.Bind("preview_reduced", &model_.preview_reduced);
            bound &= constructor.Bind("preview_reduced_text", &model_.preview_reduced_text);
            bound &= constructor.Bind("input_device", &model_.input_device);
            bound &= constructor.Bind("input_device_tooltip", &model_.input_device_tooltip);
            bound &= constructor.Bind("lfs_mem_text", &model_.lfs_mem_text);
            bound &= constructor.Bind("lfs_mem_color", &model_.lfs_mem_color);
            bound &= constructor.Bind("show_gpu_model", &model_.show_gpu_model);
            bound &= constructor.Bind("gpu_panel_active", &model_.gpu_panel_active);
            bound &= constructor.Bind("gpu_model_text", &model_.gpu_model_text);
            bound &= constructor.Bind("gpu_mem_text", &model_.gpu_mem_text);
            bound &= constructor.Bind("gpu_mem_color", &model_.gpu_mem_color);
            bound &= constructor.Bind("fps_value", &model_.fps_value);
            bound &= constructor.Bind("fps_view_value", &model_.fps_view_value);
            bound &= constructor.Bind("fps_ui_label", &model_.fps_ui_label);
            bound &= constructor.Bind("fps_view_label", &model_.fps_view_label);
            bound &= constructor.Bind("fps_color", &model_.fps_color);
            bound &= constructor.Bind("fps_label", &model_.fps_label);
            bound &= constructor.Bind("gpu_restart_required", &model_.gpu_restart_required);
            bound &= constructor.Bind("gpu_restart_text", &model_.gpu_restart_text);
            bound &= constructor.Bind("gpu_restart_tooltip", &model_.gpu_restart_tooltip);
            bound &= constructor.BindEventCallback("open_gpu_preferences", [](auto, auto&, const auto&) {});
            bound &= constructor.Bind("renderer_label", &model_.renderer_label);
            bound &= constructor.Bind("renderer_value", &model_.renderer_value);
            bound &= constructor.Bind("renderer_tooltip", &model_.renderer_tooltip);
            bound &= constructor.Bind("upscaler_label", &model_.upscaler_label);
            bound &= constructor.Bind("upscaler_value", &model_.upscaler_value);
            bound &= constructor.Bind("upscaler_tooltip", &model_.upscaler_tooltip);
            bound &= constructor.Bind("upscaler_menu", &model_.upscaler_menu);
            bound &= constructor.Bind("upscaler_menu_expanded", &model_.upscaler_menu_expanded);
            bound &= constructor.BindEventCallback("toggle_upscaler_menu", [](auto, auto&, const auto&) {});
            bound &= constructor.BindEventCallback("choose_upscaler", [](auto, auto&, const auto&) {});
            bound &= constructor.BindEventCallback("choose_upscaler_preset", [](auto, auto&, const auto&) {});
            bound &= constructor.Bind("tensor_label", &model_.tensor_label);
            bound &= constructor.Bind("tensor_value", &model_.tensor_value);
            bound &= constructor.Bind("tensor_tooltip", &model_.tensor_tooltip);
            bound &= constructor.Bind("git_commit", &model_.git_commit);
            bound &= constructor.Bind("mcp_details_expanded", &model_.mcp_details_expanded);
            bound &= constructor.Bind("mcp_summary", &model_.mcp_summary);
            bound &= constructor.Bind("mcp_details", &model_.mcp_details);
            bound &= constructor.Bind("mcp_tooltip", &model_.mcp_tooltip);
            bound &= constructor.Bind("mcp_color", &model_.mcp_color);
            bound &= constructor.Bind("mcp_preferences_label", &model_.mcp_preferences_label);
            bound &= constructor.Bind("mcp_server_enabled", &model_.mcp_server_enabled);
            bound &= constructor.Bind("mcp_toggle_label", &model_.mcp_toggle_label);
            bound &= constructor.Bind("mcp_total_text", &model_.mcp_total_text);
            bound &= constructor.Bind("mcp_success_text", &model_.mcp_success_text);
            bound &= constructor.Bind("mcp_error_text", &model_.mcp_error_text);
            bound &= constructor.Bind("show_status_message", &model_.show_status_message);
            bound &= constructor.Bind("status_message_text", &model_.status_message_text);
            bound &= constructor.Bind("status_message_color", &model_.status_message_color);
            ASSERT_TRUE(bound);
            model_handle_ = constructor.GetModelHandle();
            lfs::vis::gui::RmlStatusBarTestAccess::setModelHandle(status_bar_, model_handle_);

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
            lfs::vis::gui::RmlStatusBarTestAccess::detach(status_bar_);
            model_handle_ = {};
            ASSERT_TRUE(Rml::RemoveContext("status_bar_fit"));
            context_ = nullptr;
            document_ = nullptr;
        }

        inline static StubRenderInterface render_interface_;
        Rml::Context* context_ = nullptr;
        Rml::ElementDocument* document_ = nullptr;
        Rml::DataModelHandle model_handle_;
        lfs::vis::gui::RmlStatusBar status_bar_;
        lfs::vis::gui::RmlStatusBarTestAccess::ModelState& model_ =
            lfs::vis::gui::RmlStatusBarTestAccess::model(status_bar_);
    };

    TEST_F(StatusBarFitTest, TooltipHoverStartsAndClearsForBothFpsMetrics) {
        using lfs::vis::gui::RmlStatusBarTestAccess;
        for (const char* id : {"fps-value", "fps-view-value"}) {
            auto* value = document_->GetElementById(id);
            ASSERT_TRUE(value);
            auto* metric = value->GetParentNode();
            EXPECT_FALSE(metric->GetAttribute<Rml::String>("data-tooltip", "").empty());
            metric->SetAttribute("title", "Frame rate details");
            const auto offset = value->GetAbsoluteOffset(Rml::BoxArea::Border);
            lfs::vis::gui::PanelInputState input{};
            input.mouse_x = offset.x + value->GetOffsetWidth() / 2;
            input.mouse_y = offset.y + value->GetOffsetHeight() / 2;
            status_bar_.processInput(input, 0, 0, 2400, 22);
            EXPECT_TRUE(RmlStatusBarTestAccess::tooltipPending(status_bar_));
            status_bar_.clearTooltip();
            EXPECT_FALSE(RmlStatusBarTestAccess::tooltipPending(status_bar_));
            input.mouse_y = -100;
            status_bar_.processInput(input, 0, 0, 2400, 22);
            EXPECT_FALSE(RmlStatusBarTestAccess::tooltipPending(status_bar_));
        }
    }

    TEST_F(StatusBarFitTest, TooltipOverlayReservesOnlyMeasuredHeight) {
        using lfs::vis::gui::RmlStatusBarTestAccess;
        EXPECT_FLOAT_EQ(status_bar_.overlayHeight(), 0.0f);
        for (const float scale : {1.0f, 2.0f, 4.0f}) {
            SCOPED_TRACE(scale);
            context_->SetDensityIndependentPixelRatio(scale);
            context_->SetDimensions({2400, 900});
            document_->SetProperty("height", "900px");
            context_->Update();
            RmlStatusBarTestAccess::prepareTooltip(status_bar_,
                                                   "Frame rate details\nA second line");
            EXPECT_FLOAT_EQ(status_bar_.overlayHeight(), 0.0f);
            RmlStatusBarTestAccess::revealTooltip(status_bar_);
            auto* tooltip = document_->GetElementById("frame-tooltip");
            ASSERT_NE(tooltip, nullptr);
            ASSERT_TRUE(tooltip->IsVisible());
            const float measured = tooltip->GetOffsetHeight();
            EXPECT_GE(status_bar_.overlayHeight(), measured);
            EXPECT_LE(status_bar_.overlayHeight(), measured + 20.0f);
            status_bar_.clearTooltip();
            RmlStatusBarTestAccess::revealTooltip(status_bar_);
            EXPECT_FLOAT_EQ(status_bar_.overlayHeight(), 0.0f);
        }
    }

    TEST_F(StatusBarFitTest, TooltipAndMcpPopupShareOnlyTheirRequiredOverlay) {
        using lfs::vis::gui::RmlStatusBarTestAccess;
        RmlStatusBarTestAccess::setMcpExpanded(status_bar_, true);
        model_.mcp_details_expanded = true;
        model_handle_.DirtyVariable("mcp_details_expanded");
        context_->SetDimensions({2400, 900});
        document_->SetProperty("height", "900px");
        context_->Update();
        const float popup_height = status_bar_.overlayHeight();
        ASSERT_GE(popup_height, 150.0f);
        RmlStatusBarTestAccess::prepareTooltip(status_bar_,
                                               "Frame rate details\nA second line");
        EXPECT_FLOAT_EQ(status_bar_.overlayHeight(), popup_height);
        RmlStatusBarTestAccess::revealTooltip(status_bar_);
        EXPECT_FLOAT_EQ(status_bar_.overlayHeight(), popup_height);
        RmlStatusBarTestAccess::setMcpExpanded(status_bar_, false);
        EXPECT_LT(status_bar_.overlayHeight(), popup_height);
        status_bar_.clearTooltip();
        RmlStatusBarTestAccess::revealTooltip(status_bar_);
        EXPECT_FLOAT_EQ(status_bar_.overlayHeight(), 0.0f);
    }

    TEST_F(StatusBarFitTest, TooltipTranslationUsesPendingPositionAfterMeasurement) {
        using lfs::vis::gui::RmlStatusBarTestAccess;
        context_->SetDimensions({2400, 900});
        document_->SetProperty("height", "900px");
        context_->Update();
        RmlStatusBarTestAccess::prepareTooltip(status_bar_, "Frame rate details");
        RmlStatusBarTestAccess::revealTooltip(status_bar_, false);
        auto* tooltip = document_->GetElementById("frame-tooltip");
        ASSERT_NE(tooltip, nullptr);
        ASSERT_LT(tooltip->GetOffsetTop(), 0.0f);
        const float pending_top = tooltip->GetLocalProperty(Rml::PropertyId::Top)->Get<float>();
        ASSERT_GE(pending_top, 0.0f);
        RmlStatusBarTestAccess::positionTooltip(status_bar_, 150, 175);
        context_->Update();
        EXPECT_FLOAT_EQ(tooltip->GetOffsetTop(), pending_top + 25);
        // Changing the surface again preserves the same screen-space anchor.
        RmlStatusBarTestAccess::positionTooltip(status_bar_, 175, 150);
        context_->Update();
        EXPECT_FLOAT_EQ(tooltip->GetOffsetTop(), pending_top);
    }

    TEST_F(StatusBarFitTest, LayoutAndDigitReservationFollowDensityChanges) {
        for (const float scale : {1.0f, 1.25f, 1.5f, 1.75f, 2.0f, 2.5f, 3.0f, 4.0f}) {
            SCOPED_TRACE(scale);
            context_->SetDensityIndependentPixelRatio(scale);
            const int height = static_cast<int>(22 * scale + 0.5f);
            document_->SetProperty("height", std::format("{}px", height));
            context_->SetDimensions({static_cast<int>(2400 * scale), height});
            context_->Update();
            lfs::vis::gui::RmlStatusBarTestAccess::attach(status_bar_, context_, document_);
            auto* ui = document_->GetElementById("fps-value");
            model_.fps_value = "144";
            model_handle_.DirtyVariable("fps_value");
            context_->Update();
            const float width = ui->GetOffsetWidth();
            model_.fps_value = "0";
            model_handle_.DirtyVariable("fps_value");
            context_->Update();
            EXPECT_FLOAT_EQ(ui->GetOffsetWidth(), width);
            for (const int dp_width : {2400, 1600, 1200, 900, 700, 500, 320}) {
                SCOPED_TRACE(dp_width);
                const int px_width = static_cast<int>(dp_width * scale);
                context_->SetDimensions({px_width, height});
                context_->Update();
                lfs::vis::gui::RmlStatusBarTestAccess::fit(status_bar_);
                assertNoVerticalOverflow(document_);
                assertFlexSiblingsDoNotOverlap(document_);
                EXPECT_LE(ui->GetAbsoluteOffset().x + ui->GetOffsetWidth(), px_width + 0.5f);
            }
        }
    }

    TEST_F(StatusBarFitTest, LocalizedTooltipsStayInsideScaledNarrowWindow) {
        auto& loc = lfs::event::LocalizationManager::getInstance();
        const auto locales = std::filesystem::path(PROJECT_ROOT_PATH) /
                             "src/visualizer/gui/resources/locales";
        ASSERT_TRUE(loc.initialize(locales.string()));
        lfs::vis::gui::RmlTooltipController tooltip;
        for (const float scale : {1.0f, 1.25f, 1.5f, 2.0f, 3.0f, 4.0f}) {
            SCOPED_TRACE(scale);
            const int width = static_cast<int>(400 * scale);
            const int height = static_cast<int>(300 * scale);
            context_->SetDensityIndependentPixelRatio(scale);
            context_->SetDimensions({width, height});
            document_->SetProperty("height", std::format("{}px", height));
            context_->Update();
            for (const auto& language : loc.getAvailableLanguages()) {
                SCOPED_TRACE(language);
                ASSERT_TRUE(loc.setLanguage(language));
                for (const char* id : {"fps-value", "fps-view-value"}) {
                    auto* value = document_->GetElementById(id);
                    const std::string text = lfs::vis::gui::resolveRmlTooltip(value);
                    ASSERT_FALSE(text.empty());
                    EXPECT_NE(text, value->GetParentNode()->GetAttribute<Rml::String>("data-tooltip", ""));
                    tooltip.setHover(text, value);
                    if (const auto deadline = tooltip.revealDeadline())
                        std::this_thread::sleep_until(*deadline + std::chrono::milliseconds(2));
                    ASSERT_TRUE(tooltip.apply(document_, width - 5, height - 5, width, height));
                    context_->Update();
                    auto* element = document_->GetElementById("frame-tooltip");
                    ASSERT_TRUE(element);
                    ASSERT_TRUE(element->IsVisible());
                    const auto offset = element->GetAbsoluteOffset(Rml::BoxArea::Border);
                    EXPECT_GE(offset.x, 0);
                    EXPECT_GE(offset.y, 0);
                    EXPECT_LE(offset.x + element->GetOffsetWidth(), width + 0.5f);
                    EXPECT_LE(offset.y + element->GetOffsetHeight(), height + 0.5f);
                    tooltip.setHover({}, nullptr);
                    EXPECT_TRUE(tooltip.apply(document_, 0, 0, width, height));
                    context_->Update();
                    EXPECT_FALSE(element->IsVisible());
                }
            }
        }
        EXPECT_TRUE(loc.setLanguage("en"));
    }

    TEST_F(StatusBarFitTest, PendingGpuRestartStaysVisibleWithoutOverlappingBackendBadges) {
        model_.gpu_restart_required = true;
        model_.gpu_restart_text = "Restart needed";
        model_.gpu_restart_tooltip = "Restart to apply the saved GPU selection";
        model_handle_.DirtyVariable("gpu_restart_required");
        model_handle_.DirtyVariable("gpu_restart_text");
        model_handle_.DirtyVariable("gpu_restart_tooltip");
        auto* chip = document_->GetElementById("gpu-restart-chip");
        ASSERT_NE(chip, nullptr);
        auto* label = chip->QuerySelector(".gpu-restart-label");
        ASSERT_NE(label, nullptr);
        for (const int width : {1280, 960, 640, 480, 320}) {
            SCOPED_TRACE(width);
            context_->SetDimensions({width, 22});
            context_->Update();
            lfs::vis::gui::RmlStatusBarTestAccess::fit(status_bar_, true);
            context_->Update();
            EXPECT_TRUE(chip->IsVisible(true));
            EXPECT_GT(chip->GetOffsetWidth(), 0.0f);
            EXPECT_EQ(label->IsVisible(true), width > 640);
            EXPECT_LE(chip->GetAbsoluteOffset().x + chip->GetOffsetWidth(), width + 0.5f);
            assertNoVerticalOverflow(document_);
            assertFlexSiblingsDoNotOverlap(document_);
        }
        model_.gpu_restart_required = false;
        model_handle_.DirtyVariable("gpu_restart_required");
        context_->Update();
        EXPECT_FALSE(chip->IsVisible(true));
    }

    TEST_F(StatusBarFitTest, KeepsSingleLineNonOverlappingLayoutAcrossWidths) {
        const std::vector<int> widths = {3000, 2400, 1600, 1200, 900, 700, 500, 320};
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
            auto* fps = document_->GetElementById("fps-value");
            EXPECT_LE(fps->GetAbsoluteOffset().x + fps->GetOffsetWidth(), width);
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

    TEST_F(StatusBarFitTest, BackendBadgesKeepDistinctRolesAndUpdateTooltips) {
        auto* renderer = document_->GetElementById("renderer-backend-chip");
        auto* tensor = document_->GetElementById("tensor-backend-chip");
        auto* fps = document_->GetElementById("fps-group");
        ASSERT_NE(renderer, nullptr);
        ASSERT_NE(tensor, nullptr);
        ASSERT_NE(fps, nullptr);
        EXPECT_EQ(renderer->GetAttribute<Rml::String>("title", ""), model_.renderer_tooltip);
        EXPECT_EQ(tensor->GetAttribute<Rml::String>("title", ""), model_.tensor_tooltip);
        auto* upscaler = document_->GetElementById("upscaler-backend-chip");
        ASSERT_NE(upscaler, nullptr);
        EXPECT_LT(renderer->GetAbsoluteOffset().x, upscaler->GetAbsoluteOffset().x);
        EXPECT_LT(upscaler->GetAbsoluteOffset().x, tensor->GetAbsoluteOffset().x);
        EXPECT_LT(fps->GetAbsoluteOffset().x, renderer->GetAbsoluteOffset().x);
        EXPECT_NE(renderer->GetInnerRML().find(">R<"), Rml::String::npos);
        EXPECT_NE(tensor->GetInnerRML().find(">T<"), Rml::String::npos);

        model_.renderer_value = "Vulkan";
        model_.renderer_tooltip = "Scene renderer: Vulkan";
        model_handle_.DirtyVariable("renderer_value");
        model_handle_.DirtyVariable("renderer_tooltip");
        context_->Update();
        EXPECT_EQ(renderer->GetAttribute<Rml::String>("title", ""), model_.renderer_tooltip);
        EXPECT_NE(renderer->GetInnerRML().find("Vulkan"), Rml::String::npos);
        EXPECT_EQ(renderer->GetInnerRML().find("Metal / Vulkan"), Rml::String::npos);

        context_->SetDimensions({320, 22});
        context_->Update();
        lfs::vis::gui::RmlStatusBarTestAccess::fit(status_bar_);
        EXPECT_TRUE(renderer->IsVisible(true));
        EXPECT_TRUE(tensor->IsVisible(true));
        EXPECT_TRUE(renderer->GetChild(0)->IsVisible(true));
        EXPECT_TRUE(tensor->GetChild(0)->IsVisible(true));
        EXPECT_LE(tensor->GetAbsoluteOffset().x + tensor->GetOffsetWidth(), 320.5f);
        for (const char* id : {"fps-value", "fps-view-value", "fps-label"}) {
            auto* element = document_->GetElementById(id);
            ASSERT_NE(element, nullptr);
            EXPECT_TRUE(element->IsVisible(true));
            EXPECT_GT(element->GetOffsetWidth(), 0.0f);
        }
        EXPECT_EQ(renderer->GetAttribute<Rml::String>("title", ""), model_.renderer_tooltip);
        assertNoVerticalOverflow(document_);
        assertFlexSiblingsDoNotOverlap(document_);
    }

    TEST_F(StatusBarFitTest, UpscalerShowsEffectiveBackendAndExplainsPendingAndFallback) {
        using namespace lfs::vis;
        RenderSettings settings;
        settings.scene_upscaler = "temporal";
        settings.scene_upscaler_preset = "quality";
        auto& state = gui::RmlStatusBarTestAccess::model(status_bar_);
        gui::RmlStatusBarTestAccess::updateUpscaler(status_bar_, settings,
                                                    {SceneUpscalerBackend::Temporal, SceneUpscalerBackend::Temporal, SceneUpscalerFallback::None});
        const auto active_label = state.upscaler_value;
        EXPECT_FALSE(active_label.empty());
        gui::RmlStatusBarTestAccess::updateUpscaler(status_bar_, settings,
                                                    {SceneUpscalerBackend::Temporal, SceneUpscalerBackend::Native, SceneUpscalerFallback::UnsupportedMode});
        EXPECT_NE(state.upscaler_value, active_label);
        EXPECT_NE(state.upscaler_tooltip.find(LOC("status_bar.upscaler_unsupported")), std::string::npos);
        const auto native_label = state.upscaler_value;
        gui::RmlStatusBarTestAccess::updateUpscaler(status_bar_, settings, {});
        EXPECT_EQ(state.upscaler_value, native_label);
        EXPECT_NE(state.upscaler_tooltip.find(LOC("status_bar.upscaler_pending")), std::string::npos);
        for (const auto backend : {SceneUpscalerBackend::Native, SceneUpscalerBackend::Spatial,
                                   SceneUpscalerBackend::Temporal, SceneUpscalerBackend::MetalFxSpatial,
                                   SceneUpscalerBackend::MetalFxTemporal}) {
            const auto marker = "choose_upscaler('" + std::string(sceneUpscalerBackendId(backend)) + "')";
            EXPECT_EQ(state.upscaler_menu.find(marker) != std::string::npos, sceneUpscalerBackendAvailable(backend));
        }
    }

    TEST_F(StatusBarFitTest, BackendBadgeFollowsActiveViewInsteadOfLastPublishedView) {
        auto& store = lfs::vis::app_store();
        const auto previous = store.viewer_backend.get();
        struct Restore {
            std::optional<lfs::rendering::ViewerBackend> value;
            ~Restore() { lfs::vis::app_store().viewer_backend.set(value); }
        } restore{previous};
        store.viewer_backend.set(lfs::rendering::ViewerBackend::Vulkan);
        lfs::vis::gui::RmlStatusBarTestAccess::updateBackends(
            status_bar_, lfs::rendering::ViewerBackend::Metal);
        EXPECT_EQ(lfs::vis::gui::RmlStatusBarTestAccess::model(status_bar_).renderer_value, "Metal");
        store.viewer_backend.set(lfs::rendering::ViewerBackend::Metal);
        lfs::vis::gui::RmlStatusBarTestAccess::updateBackends(
            status_bar_, lfs::rendering::ViewerBackend::Vulkan);
        EXPECT_EQ(lfs::vis::gui::RmlStatusBarTestAccess::model(status_bar_).renderer_value, "Vulkan");
        store.viewer_backend.set(lfs::rendering::ViewerBackend::Cuda);
        lfs::vis::gui::RmlStatusBarTestAccess::updateBackends(status_bar_, std::nullopt);
        EXPECT_EQ(lfs::vis::gui::RmlStatusBarTestAccess::model(status_bar_).renderer_value,
                  lfs::rendering::viewerBackendDisplayName(lfs::rendering::desktopViewerBackend()));
        lfs::vis::gui::RmlStatusBarTestAccess::updateBackends(status_bar_);
        EXPECT_EQ(lfs::vis::gui::RmlStatusBarTestAccess::model(status_bar_).renderer_value, "CUDA");
    }

    TEST_F(StatusBarFitTest, BackendTooltipRevealsAboveBarAndClearsOnPointerLeave) {
        auto* chip = document_->GetElementById("renderer-backend-chip");
        ASSERT_NE(chip, nullptr);
        // Exercise the real input -> delayed reveal -> tooltip layout path, not
        // just the presence of a title attribute. Rich text must be escaped.
        model_.renderer_tooltip = "Scene renderer <actual> & requested\n" +
                                  std::string(500, 'W');
        model_handle_.DirtyVariable("renderer_tooltip");
        context_->Update();
        lfs::vis::gui::RmlUIManager manager;
        manager.beginFrameCursorTracking();
        lfs::vis::gui::RmlStatusBarTestAccess::trackRenderedFrame(status_bar_, manager, 0.0f, 700.0f);
        const auto offset = chip->GetAbsoluteOffset(Rml::BoxArea::Border);
        lfs::vis::gui::PanelInputState input{};
        input.mouse_x = offset.x + chip->GetOffsetWidth() * 0.5f;
        input.mouse_y = 711.0f;
        status_bar_.processInput(input, 0.0f, 700.0f, 2400.0f, 22.0f);
        const auto deadline = lfs::vis::gui::RmlStatusBarTestAccess::tooltipDeadline(status_bar_);
        ASSERT_TRUE(deadline);
        EXPECT_EQ(status_bar_.overlayHeight(), 0.0f); // no enlarged render surface during delay
        EXPECT_TRUE(manager.secondsUntilTooltipReveal().has_value());
        EXPECT_FALSE(status_bar_.animationFrameDue(std::chrono::steady_clock::now()));
        std::this_thread::sleep_until(*deadline);
        EXPECT_TRUE(status_bar_.animationFrameDue(std::chrono::steady_clock::now()));
        EXPECT_TRUE(lfs::vis::gui::RmlStatusBarTestAccess::applyTooltip(status_bar_));
        context_->Update();
        auto* tooltip = document_->GetElementById("frame-tooltip");
        ASSERT_NE(tooltip, nullptr);
        ASSERT_TRUE(tooltip->IsVisible());
        EXPECT_NE(tooltip->GetInnerRML().find("&lt;actual&gt;"), Rml::String::npos);
        const auto position = tooltip->GetAbsoluteOffset(Rml::BoxArea::Border);
        EXPECT_GE(position.x, 0.0f);
        EXPECT_GE(position.y, 0.0f);
        EXPECT_LE(position.x + tooltip->GetOffsetWidth(), 2400.5f);
        EXPECT_LE(position.y + tooltip->GetOffsetHeight(), status_bar_.overlayHeight() + 0.5f);
        EXPECT_FALSE(manager.secondsUntilTooltipReveal().has_value());
        EXPECT_FALSE(status_bar_.animationFrameDue(std::chrono::steady_clock::now()));
        EXPECT_FALSE(status_bar_.secondsUntilAnimationFrame(std::chrono::steady_clock::now()).has_value());
        EXPECT_FALSE(status_bar_.isOverlayPoint(input.mouse_x, -5.0f, 2400.0f));
        EXPECT_FALSE(lfs::vis::gui::RmlStatusBarTestAccess::applyTooltip(status_bar_));

        input.mouse_y = 200.0f;
        status_bar_.processInput(input, 0.0f, 700.0f, 2400.0f, 22.0f);
        EXPECT_TRUE(lfs::vis::gui::RmlStatusBarTestAccess::applyTooltip(status_bar_));
        context_->Update();
        EXPECT_FALSE(tooltip->IsVisible());
        EXPECT_EQ(status_bar_.overlayHeight(), 0.0f);
        EXPECT_EQ(context_->GetDimensions().y, 22);
    }

    TEST_F(StatusBarFitTest, TensorTooltipUsesOneTargetAcrossLetterAndValue) {
        auto* chip = document_->GetElementById("tensor-backend-chip");
        ASSERT_NE(chip, nullptr);
        ASSERT_GE(chip->GetNumChildren(), 2);
        lfs::vis::gui::PanelInputState input{};
        input.mouse_y = 711.0f;
        const auto hover_child = [&](int child) {
            auto* element = chip->GetChild(child);
            input.mouse_x = element->GetAbsoluteOffset(Rml::BoxArea::Border).x + element->GetOffsetWidth() * 0.5f;
            status_bar_.processInput(input, 0.0f, 700.0f, 2400.0f, 22.0f);
        };
        hover_child(0);
        const auto deadline = lfs::vis::gui::RmlStatusBarTestAccess::tooltipDeadline(status_bar_);
        ASSERT_TRUE(deadline);
        hover_child(1);
        EXPECT_EQ(deadline, lfs::vis::gui::RmlStatusBarTestAccess::tooltipDeadline(status_bar_));
        EXPECT_FALSE(status_bar_.animationFrameDue(std::chrono::steady_clock::now()));
        std::this_thread::sleep_until(*deadline);
        EXPECT_TRUE(status_bar_.animationFrameDue(std::chrono::steady_clock::now()));
        EXPECT_TRUE(lfs::vis::gui::RmlStatusBarTestAccess::applyTooltip(status_bar_));
        context_->Update();
        auto* tooltip = document_->GetElementById("frame-tooltip");
        ASSERT_NE(tooltip, nullptr);
        EXPECT_TRUE(tooltip->IsVisible());
        EXPECT_EQ(tooltip->GetInnerRML(), model_.tensor_tooltip);
    }

    TEST_F(StatusBarFitTest, BadgesTrackPublishedRendererAndSceneClose) {
        const ScopedStatusBarHome scoped_home;
        lfs::vis::ViewportArtifactService artifacts;
        lfs::vis::gui::RmlStatusBarTestAccess::bindStore(status_bar_);
        auto& store = lfs::vis::app_store();
        store.viewer_backend.set(std::nullopt);
        (void)store.store().drain_dirty_into_frame();
        const auto publish = [&](lfs::rendering::ViewerBackend backend, const char* expected) {
            lfs::rendering::FrameMetadata frame;
            frame.viewer_backend = backend;
            artifacts.setLazyCapture({}, frame, {64, 48});
            (void)store.store().drain_dirty_into_frame();
            EXPECT_TRUE(lfs::vis::gui::RmlStatusBarTestAccess::redrawPending(status_bar_));
            lfs::vis::gui::RmlStatusBarTestAccess::updateBackends(status_bar_);
            context_->Update();
            EXPECT_EQ(model_.renderer_value, expected);
            auto* chip = document_->GetElementById("renderer-backend-chip");
            EXPECT_NE(chip->GetInnerRML().find(expected), Rml::String::npos);
            EXPECT_EQ(model_.tensor_value,
                      lfs::core::gpu_backend_name(lfs::core::configured_gpu_backend()));
            lfs::vis::gui::RmlStatusBarTestAccess::clearRedraw(status_bar_);
            artifacts.setLazyCaptureForCurrentOutput({}, frame, {64, 48});
            EXPECT_FALSE(store.store().has_dirty());
            EXPECT_FALSE(lfs::vis::gui::RmlStatusBarTestAccess::redrawPending(status_bar_));
        };
        publish(lfs::rendering::ViewerBackend::Metal, "Metal");
        publish(lfs::rendering::ViewerBackend::Vulkan, "Vulkan");
        publish(lfs::rendering::ViewerBackend::Cuda, "CUDA");
        artifacts.clearViewportOutput();
        (void)store.store().drain_dirty_into_frame();
        EXPECT_TRUE(lfs::vis::gui::RmlStatusBarTestAccess::redrawPending(status_bar_));
        lfs::vis::gui::RmlStatusBarTestAccess::updateBackends(status_bar_);
        context_->Update();
#ifdef LFS_TENSOR_METAL
        EXPECT_EQ(model_.renderer_value,
                  "Metal");
#else
        EXPECT_EQ(model_.renderer_value, "Vulkan");
#endif // Configured renderer, no scene frame.
        EXPECT_FALSE(store.viewer_backend.get());
    }

    TEST_F(StatusBarFitTest, FpsValuesReserveIndependentWidthsAndKeepUnitVisible) {
        context_->SetDimensions({2400, 22});
        context_->Update();
        auto* ui = document_->GetElementById("fps-value");
        auto* view = document_->GetElementById("fps-view-value");
        auto* unit = document_->GetElementById("fps-label");
        ASSERT_NE(ui, nullptr);
        ASSERT_NE(view, nullptr);
        ASSERT_NE(unit, nullptr);
        const auto ui_width = ui->GetOffsetWidth();
        const auto view_position = view->GetAbsoluteOffset().x;
        model_.fps_value = "0";
        model_.fps_view_value = "9";
        model_handle_.DirtyVariable("fps_value");
        model_handle_.DirtyVariable("fps_view_value");
        context_->Update();
        EXPECT_FLOAT_EQ(ui->GetOffsetWidth(), ui_width);
        EXPECT_FLOAT_EQ(view->GetAbsoluteOffset().x, view_position);
        // The reservation is a minimum: four-digit rates must remain visible.
        model_.fps_value = "1234";
        model_handle_.DirtyVariable("fps_value");
        context_->Update();
        EXPECT_GT(ui->GetOffsetWidth(), ui_width);
        assertNoVerticalOverflow(document_);
        assertFlexSiblingsDoNotOverlap(document_);
        document_->SetClass("fit-8", true);
        context_->Update();
        EXPECT_GT(unit->GetOffsetWidth(), 0.0f);
    }

    TEST_F(StatusBarFitTest, FpsSpacingFollowsDensityChanges) {
        for (const float scale : {1.0f, 1.25f, 2.0f, 4.0f}) {
            SCOPED_TRACE(scale);
            context_->SetDensityIndependentPixelRatio(scale);
            context_->SetDimensions({static_cast<int>(2400 * scale), static_cast<int>(22 * scale)});
            document_->SetProperty("height", std::format("{}px", 22 * scale));
            context_->Update();
            lfs::vis::gui::RmlStatusBarTestAccess::attach(status_bar_, context_, document_);
            for (const char* id : {"fps-value", "fps-view-value"}) {
                auto* value = document_->GetElementById(id);
                auto* caption = value->GetParentNode()->QuerySelector(".fps-caption");
                ASSERT_NE(caption, nullptr);
                EXPECT_NEAR(value->GetAbsoluteOffset().x - caption->GetAbsoluteOffset().x -
                                caption->GetOffsetWidth(),
                            3 * scale, 0.5f);
            }
            auto* unit = document_->GetElementById("fps-label");
            auto* ui_caption = document_->GetElementById("fps-value")->GetParentNode()->QuerySelector(".fps-caption");
            EXPECT_NEAR(ui_caption->GetAbsoluteOffset().x - unit->GetAbsoluteOffset().x - unit->GetOffsetWidth(),
                        8 * scale, 0.5f);
        }
    }

    TEST_F(StatusBarFitTest, McpDetailsReserveOnlyTheirMeasuredOverlayArea) {
        EXPECT_EQ(status_bar_.overlayHeight(), 0.0f);
        EXPECT_FALSE(status_bar_.isOverlayPoint(2300.0f, -20.0f, 2400.0f));

        lfs::vis::gui::RmlStatusBarTestAccess::setMcpExpanded(status_bar_, true);
        model_.mcp_details_expanded = true;
        model_handle_.DirtyVariable("mcp_details_expanded");
        context_->SetDimensions({2400, static_cast<int>(22.0f + status_bar_.overlayHeight())});
        ASSERT_TRUE(document_->SetProperty(
            "height", std::format("{}px", 22.0f + status_bar_.overlayHeight())));
        context_->Update();

        auto* const popup = document_->GetElementById("mcp-popup");
        ASSERT_NE(popup, nullptr);
        const auto offset = popup->GetAbsoluteOffset(Rml::BoxArea::Border);
        const float center_x = offset.x + popup->GetOffsetWidth() * 0.5f;
        const float center_y = offset.y + popup->GetOffsetHeight() * 0.5f -
                               status_bar_.overlayHeight();
        EXPECT_TRUE(status_bar_.isOverlayPoint(center_x, center_y, 2400.0f));
        EXPECT_FALSE(status_bar_.isOverlayPoint(100.0f, -20.0f, 2400.0f));

        lfs::vis::gui::RmlUIManager manager;
        manager.beginFrameCursorTracking();
        constexpr float bar_x = 30.0f;
        constexpr float bar_y = 700.0f;
        lfs::vis::gui::RmlStatusBarTestAccess::trackRenderedFrame(
            status_bar_, manager, bar_x, bar_y);
        const float context_y = bar_y - status_bar_.overlayHeight();
        EXPECT_TRUE(manager.activeOverlayContainsPoint(
            bar_x + center_x,
            context_y + offset.y + popup->GetOffsetHeight() * 0.5f));
        EXPECT_FALSE(manager.activeOverlayContainsPoint(bar_x + 100.0f,
                                                        context_y + 5.0f));

        lfs::vis::gui::RmlStatusBarTestAccess::setMcpExpanded(status_bar_, false);
        EXPECT_EQ(status_bar_.overlayHeight(), 0.0f);
    }

    TEST_F(StatusBarFitTest, UpscalerMenuUsesMeasuredOverlayAndRemainsClickable) {
        EXPECT_EQ(status_bar_.overlayHeight(), 0.0f);
        EXPECT_FALSE(status_bar_.isOverlayPoint(2300.0f, -20.0f, 2400.0f));

        lfs::vis::gui::RmlStatusBarTestAccess::setUpscalerExpanded(status_bar_, true);
        model_.upscaler_menu = "<button class='upscaler-choice'>MetalFX Spatial</button><button class='upscaler-choice'>MetalFX Temporal</button>";
        model_handle_.DirtyVariable("upscaler_menu");
        model_.upscaler_menu_expanded = true;
        model_handle_.DirtyVariable("upscaler_menu_expanded");
        context_->SetDimensions({2400, static_cast<int>(22.0f + status_bar_.overlayHeight())});
        ASSERT_TRUE(document_->SetProperty(
            "height", std::format("{}px", 22.0f + status_bar_.overlayHeight())));
        context_->Update();

        auto* const popup = document_->GetElementById("upscaler-popup");
        ASSERT_NE(popup, nullptr);
        const auto offset = popup->GetAbsoluteOffset(Rml::BoxArea::Border);
        const float center_x = offset.x + popup->GetOffsetWidth() * 0.5f;
        const float center_y = offset.y + popup->GetOffsetHeight() * 0.5f -
                               status_bar_.overlayHeight();
        EXPECT_TRUE(status_bar_.isOverlayPoint(center_x, center_y, 2400.0f));
        EXPECT_FALSE(status_bar_.isOverlayPoint(100.0f, -20.0f, 2400.0f));

        lfs::vis::gui::RmlUIManager manager;
        manager.beginFrameCursorTracking();
        constexpr float bar_x = 30.0f;
        constexpr float bar_y = 700.0f;
        lfs::vis::gui::RmlStatusBarTestAccess::trackRenderedFrame(
            status_bar_, manager, bar_x, bar_y);
        const float context_y = bar_y - status_bar_.overlayHeight();
        EXPECT_TRUE(manager.activeOverlayContainsPoint(
            bar_x + center_x,
            context_y + offset.y + popup->GetOffsetHeight() * 0.5f));
        EXPECT_FALSE(manager.activeOverlayContainsPoint(bar_x + 100.0f,
                                                        context_y + 5.0f));

        lfs::vis::gui::RmlStatusBarTestAccess::setUpscalerExpanded(status_bar_, false);
        EXPECT_EQ(status_bar_.overlayHeight(), 0.0f);
    }

    TEST(RuntimeServiceControlsTest, DispatchesMcpActionsThroughVisualizerBoundary) {
        int enabled_toggles = 0;
        int binding_toggles = 0;
        lfs::vis::setRuntimeServiceControls({
            .toggle_mcp_enabled = [&] {
                ++enabled_toggles;
                return true; },
            .toggle_mcp_binding = [&] {
                ++binding_toggles;
                return true; },
        });

        EXPECT_TRUE(lfs::vis::toggleMcpRuntimeEnabled());
        EXPECT_TRUE(lfs::vis::toggleMcpRuntimeBinding());
        EXPECT_EQ(enabled_toggles, 1);
        EXPECT_EQ(binding_toggles, 1);

        lfs::vis::setRuntimeServiceControls({});
        EXPECT_FALSE(lfs::vis::toggleMcpRuntimeEnabled());
        EXPECT_FALSE(lfs::vis::toggleMcpRuntimeBinding());
    }

} // namespace

namespace {
    class StatusBarTrainingTest : public ::testing::Test {
    protected:
        lfs::vis::ThemePalette palette_{};

        void SetUp() override {
            palette_.success = {0.1f, 0.8f, 0.2f, 1.0f};
            palette_.text_dim = {0.4f, 0.4f, 0.4f, 1.0f};
            palette_.error = {0.9f, 0.1f, 0.1f, 1.0f};
        }

        void expectStatus(lfs::vis::FinishReason reason, const char* key,
                          const lfs::vis::ThemeColor& color) {
            const auto status = lfs::vis::gui::finishedTrainingStatus(reason, palette_);
            EXPECT_STREQ(status.label_key, key);
            EXPECT_EQ(status.color.x, color.x);
            EXPECT_EQ(status.color.y, color.y);
            EXPECT_EQ(status.color.z, color.z);
            EXPECT_EQ(status.color.w, color.w);
        }
    };

    TEST_F(StatusBarTrainingTest, UserStopIsNotCompletion) {
        expectStatus(lfs::vis::FinishReason::UserStopped, "status.stopped", palette_.text_dim);
    }

    TEST_F(StatusBarTrainingTest, ErrorIsNotCompletion) {
        expectStatus(lfs::vis::FinishReason::Error, "status.error", palette_.error);
    }

    TEST_F(StatusBarTrainingTest, CompletedRunKeepsSuccessPresentation) {
        expectStatus(lfs::vis::FinishReason::Completed, "status.complete", palette_.success);
    }

    TEST_F(StatusBarTrainingTest, StoredCompletedSessionWithoutTrainerKeepsSuccessPresentation) {
        expectStatus(lfs::vis::FinishReason::None, "status.complete", palette_.success);
    }
} // namespace
