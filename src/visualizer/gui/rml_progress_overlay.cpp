/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rml_progress_overlay.hpp"

#include "core/event_bridge/localization_manager.hpp"
#include "core/logger.hpp"
#include "gui/gui_focus_state.hpp"
#include "gui/panel_layout.hpp"
#include "gui/rmlui/rml_document_utils.hpp"
#include "gui/rmlui/rml_theme.hpp"
#include "gui/rmlui/sdl_rml_key_mapping.hpp"
#include "gui/string_keys.hpp"
#include "internal/resource_paths.hpp"

#include <RmlUi/Core.h>
#include <RmlUi/Core/Element.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <format>
#include <utility>

namespace lfs::vis::gui {

    namespace {
        namespace Common = lichtfeld::Strings::Common;
        namespace Progress = lichtfeld::Strings::Progress;

        constexpr float kDialogWidthDp = 420.0f;
        constexpr float kScreenMarginDp = 16.0f;

        float clampProgress(const float value) {
            return std::clamp(value, 0.0f, 1.0f);
        }

        void setText(Rml::Element* element, const std::string& text) {
            if (element)
                element->SetInnerRML(Rml::StringUtilities::EncodeRml(text));
        }

        void setVisible(Rml::Element* element, const bool visible,
                        const char* visible_display = "block") {
            if (element)
                element->SetProperty("display", visible ? visible_display : "none");
        }
    } // namespace

    ProgressOverlayPresentation makeProgressOverlayPresentation(
        const AppStore::ImportOverlayState& import_state,
        const AppStore::VideoExportOverlayState& video_state) {
        ProgressOverlayPresentation result;

        if (import_state.active || import_state.show_completion) {
            result.kind = ProgressOverlayPresentation::Kind::Import;
            result.path = import_state.path;
            result.stage = import_state.active ? import_state.stage : std::string{};
            result.error = import_state.error;
            result.success = !import_state.active && import_state.show_completion && import_state.success;
            result.show_progress = import_state.active || result.success;
            result.progress = result.success ? 1.0f : clampProgress(import_state.progress);

            if (import_state.active) {
                if (import_state.dataset_type == "project") {
                    result.title = LOC(Progress::OPENING_PROJECT);
                } else {
                    std::string title = LOC(Progress::IMPORTING);
                    const auto placeholder = title.find("%s");
                    if (placeholder != std::string::npos) {
                        title.replace(placeholder, 2,
                                      import_state.dataset_type.empty() ? "dataset"
                                                                        : import_state.dataset_type);
                    }
                    result.title = std::move(title);
                }
            } else {
                result.title = LOC(import_state.success ? Progress::IMPORT_COMPLETE_TITLE
                                                        : Progress::IMPORT_FAILED_TITLE);
                if (import_state.num_images > 0 || import_state.num_points > 0) {
                    result.detail = LOCF(Progress::IMPORT_COUNTS,
                                         import_state.num_images,
                                         import_state.num_points);
                }
                if (!import_state.success) {
                    result.action = ProgressOverlayPresentation::Action::DismissImport;
                    result.action_label = LOC(Common::OK);
                }
            }
            return result;
        }

        if (video_state.active) {
            result.kind = ProgressOverlayPresentation::Kind::VideoExport;
            result.action = ProgressOverlayPresentation::Action::CancelVideoExport;
            result.title = LOC(Progress::EXPORTING_VIDEO);
            result.stage = video_state.stage;
            result.detail = LOCF(Progress::VIDEO_FRAME,
                                 video_state.current_frame,
                                 video_state.total_frames);
            result.action_label = LOC(Common::CANCEL);
            result.progress = clampProgress(video_state.progress);
            result.show_progress = true;
        }

        return result;
    }

    struct RmlProgressOverlay::OverlayEventListener final : Rml::EventListener {
        RmlProgressOverlay* overlay = nullptr;

        void ProcessEvent(Rml::Event& event) override {
            assert(overlay);
            auto* current = event.GetCurrentElement();
            if (!current)
                return;

            const auto& id = current->GetId();
            if (id == "progress-backdrop") {
                event.StopPropagation();
                return;
            }
            if (id == "progress-action") {
                overlay->invokeAction();
                event.StopPropagation();
            }
        }
    };

    RmlProgressOverlay::RmlProgressOverlay(RmlUIManager* rml_manager,
                                           std::function<void()> dismiss_import,
                                           std::function<void()> cancel_video_export)
        : rml_manager_(rml_manager),
          dismiss_import_(std::move(dismiss_import)),
          cancel_video_export_(std::move(cancel_video_export)),
          listener_(std::make_unique<OverlayEventListener>()) {
        assert(rml_manager_);
        listener_->overlay = this;
    }

    RmlProgressOverlay::~RmlProgressOverlay() {
        if (rml_manager_ && rml_manager_->isInitialized())
            rml_manager_->releaseCachedVulkanContext(direct_cache_);
        if (rml_context_ && rml_manager_ && rml_manager_->isInitialized())
            rml_manager_->destroyContext("progress_overlay");
    }

    bool RmlProgressOverlay::isVisible() const {
        const auto import_state = app_store().import_overlay_state.get();
        if (import_state.active || import_state.show_completion)
            return true;
        return app_store().video_export_overlay_state.get().active;
    }

    void RmlProgressOverlay::initContext() {
        if (rml_context_)
            return;

        rml_context_ = rml_manager_->createContext("progress_overlay", 800, 600);
        if (!rml_context_) {
            LOG_ERROR("RmlProgressOverlay: failed to create context");
            return;
        }

        try {
            const auto rml_path = lfs::vis::getAssetPath("rmlui/progress_overlay.rml");
            document_ = rml_documents::loadDocument(rml_context_, rml_path);
            if (!document_) {
                LOG_ERROR("RmlProgressOverlay: failed to load progress_overlay.rml");
                return;
            }
            document_->Show();
            cacheElements();
        } catch (const std::exception& e) {
            LOG_ERROR("RmlProgressOverlay: resource not found: {}", e.what());
        }
    }

    void RmlProgressOverlay::cacheElements() {
        assert(document_);
        el_backdrop_ = document_->GetElementById("progress-backdrop");
        el_dialog_ = document_->GetElementById("progress-dialog");
        el_title_ = document_->GetElementById("progress-title");
        el_path_ = document_->GetElementById("progress-path");
        el_progress_row_ = document_->GetElementById("progress-row");
        el_progress_ = document_->GetElementById("progress-value");
        el_progress_text_ = document_->GetElementById("progress-text");
        el_stage_ = document_->GetElementById("progress-stage");
        el_detail_ = document_->GetElementById("progress-detail");
        el_error_ = document_->GetElementById("progress-error");
        el_actions_ = document_->GetElementById("progress-actions");
        el_action_ = document_->GetElementById("progress-action");

        elements_cached_ = el_backdrop_ && el_dialog_ && el_title_ && el_path_ &&
                           el_progress_row_ && el_progress_ && el_progress_text_ && el_stage_ &&
                           el_detail_ && el_error_ && el_actions_ && el_action_;
        if (!elements_cached_) {
            LOG_ERROR("RmlProgressOverlay: missing DOM elements");
            return;
        }

        el_backdrop_->AddEventListener(Rml::EventId::Click, listener_.get());
        el_action_->AddEventListener(Rml::EventId::Click, listener_.get());
    }

    bool RmlProgressOverlay::syncTheme() {
        if (!document_)
            return false;

        const std::size_t signature = rml_theme::currentThemeSignature();
        if (has_theme_signature_ && signature == last_theme_signature_)
            return false;
        last_theme_signature_ = signature;
        has_theme_signature_ = true;

        if (base_rcss_.empty())
            base_rcss_ = rml_theme::loadBaseRCSS("rmlui/progress_overlay.rcss");
        rml_theme::applyTheme(document_, base_rcss_,
                              rml_theme::loadBaseRCSS("rmlui/progress_overlay.theme.rcss"));
        return true;
    }

    void RmlProgressOverlay::applyPresentation(
        const ProgressOverlayPresentation& presentation) {
        if (!elements_cached_)
            return;

        setText(el_title_, presentation.title);
        setText(el_path_, presentation.path.empty() ? std::string{}
                                                    : LOCF(Progress::PATH, presentation.path));
        setVisible(el_path_, !presentation.path.empty());

        setVisible(el_progress_row_, presentation.show_progress);
        if (presentation.show_progress) {
            el_progress_->SetAttribute("value", std::format("{:.4f}", presentation.progress));
            setText(el_progress_text_, std::format("{:.0f}%", presentation.progress * 100.0f));
        }

        setText(el_stage_, presentation.stage);
        setVisible(el_stage_, !presentation.stage.empty());
        setText(el_detail_, presentation.detail);
        setVisible(el_detail_, !presentation.detail.empty());
        el_detail_->SetClass("status-success",
                             presentation.success && !presentation.detail.empty());
        setText(el_error_, presentation.error);
        setVisible(el_error_, !presentation.error.empty());

        const bool has_action = presentation.action != ProgressOverlayPresentation::Action::None;
        setText(el_action_, presentation.action_label);
        setVisible(el_actions_, has_action, "flex");
        el_action_->SetClass("btn--primary",
                             presentation.action == ProgressOverlayPresentation::Action::DismissImport);
        el_action_->SetClass("btn--secondary",
                             presentation.action == ProgressOverlayPresentation::Action::CancelVideoExport);

        el_title_->SetClass("status-success", presentation.success);
        el_title_->SetClass(
            "status-error",
            presentation.action == ProgressOverlayPresentation::Action::DismissImport);
        setVisible(el_backdrop_, true);
        setVisible(el_dialog_, true, "flex");
    }

    void RmlProgressOverlay::invokeAction() {
        const auto action = presentation_.action;
        if (action == ProgressOverlayPresentation::Action::DismissImport && dismiss_import_)
            dismiss_import_();
        else if (action == ProgressOverlayPresentation::Action::CancelVideoExport && cancel_video_export_)
            cancel_video_export_();
        render_needed_ = true;
    }

    void RmlProgressOverlay::processInput(const PanelInputState& input) {
        if (!isVisible() || !rml_context_ || !elements_cached_)
            return;
        if (rml_manager_)
            rml_manager_->trackContextFrame(rml_context_, 0, 0);

        auto& focus = guiFocusState();
        focus.want_capture_mouse = true;
        focus.want_capture_keyboard = true;

        const int mods = sdlModsToRml(input.key_ctrl, input.key_shift,
                                      input.key_alt, input.key_super);
        const int rml_mx = static_cast<int>(input.mouse_x - input.screen_x);
        const int rml_my = static_cast<int>(input.mouse_y - input.screen_y);
        if (!last_mouse_valid_ || rml_mx != last_mouse_x_ || rml_my != last_mouse_y_) {
            rml_context_->ProcessMouseMove(rml_mx, rml_my, mods);
            last_mouse_valid_ = true;
            last_mouse_x_ = rml_mx;
            last_mouse_y_ = rml_my;
            render_needed_ = true;
        }

        if (input.mouse_clicked[0]) {
            rml_context_->ProcessMouseButtonDown(0, mods);
            render_needed_ = true;
        }
        if (input.mouse_released[0]) {
            rml_context_->ProcessMouseButtonUp(0, mods);
            render_needed_ = true;
        }
        if (input.mouse_wheel != 0.0f || input.mouse_wheel_x != 0.0f) {
            rml_context_->ProcessMouseWheel(
                Rml::Vector2f(-input.mouse_wheel_x, -input.mouse_wheel), mods);
            render_needed_ = true;
        }

        for (const int scancode : input.keys_pressed) {
            const auto key = sdlScancodeToRml(static_cast<SDL_Scancode>(scancode));
            if (key != Rml::Input::KI_UNKNOWN) {
                rml_context_->ProcessKeyDown(key, mods);
                render_needed_ = true;
            }
        }
        for (const int scancode : input.keys_released) {
            const auto key = sdlScancodeToRml(static_cast<SDL_Scancode>(scancode));
            if (key != Rml::Input::KI_UNKNOWN) {
                rml_context_->ProcessKeyUp(key, mods);
                render_needed_ = true;
            }
        }
    }

    void RmlProgressOverlay::render(int screen_w, int screen_h,
                                    float screen_x, float screen_y,
                                    float vp_x, float vp_y, float vp_w, float vp_h) {
        const auto next = makeProgressOverlayPresentation(
            app_store().import_overlay_state.get(),
            app_store().video_export_overlay_state.get());
        if (next.kind == ProgressOverlayPresentation::Kind::None)
            return;

        if (!rml_context_) {
            initContext();
            if (!rml_context_)
                return;
        }
        if (!document_ || !elements_cached_ || !rml_manager_ ||
            !rml_manager_->getVulkanRenderInterface())
            return;
        if (screen_w <= 0 || screen_h <= 0)
            return;

        rml_manager_->trackContextFrame(rml_context_, 0, 0);
        const bool theme_changed = syncTheme();
        bool needs_update = render_needed_ || theme_changed || next != presentation_;

        if (screen_w != width_ || screen_h != height_) {
            width_ = screen_w;
            height_ = screen_h;
            rml_context_->SetDimensions(Rml::Vector2i(width_, height_));
            dialog_position_valid_ = false;
            last_mouse_valid_ = false;
            needs_update = true;
        }

        if (next != presentation_) {
            presentation_ = next;
            applyPresentation(presentation_);
            needs_update = true;
        }

        const float dp_ratio = std::max(rml_manager_->getDpRatio(), 0.01f);
        const float margin = kScreenMarginDp * dp_ratio;
        float dialog_content_width = std::min(kDialogWidthDp * dp_ratio,
                                              std::max(1.0f, static_cast<float>(width_) - 2.0f * margin));
        if (!dialog_position_valid_ ||
            std::abs(dialog_content_width - last_dialog_content_width_) > 0.5f) {
            el_dialog_->SetProperty("width", std::format("{}px", dialog_content_width));
            last_dialog_content_width_ = dialog_content_width;
            needs_update = true;
        }

        if (needs_update)
            rml_context_->Update();

        auto dialog_size = el_dialog_->GetBox().GetSize(Rml::BoxArea::Border);
        const float available_width = std::max(1.0f, static_cast<float>(width_) - 2.0f * margin);
        if (dialog_size.x > available_width + 0.5f) {
            const float chrome_width = std::max(0.0f, dialog_size.x - dialog_content_width);
            dialog_content_width = std::max(1.0f, available_width - chrome_width);
            el_dialog_->SetProperty("width", std::format("{}px", dialog_content_width));
            last_dialog_content_width_ = dialog_content_width;
            rml_context_->Update();
            dialog_size = el_dialog_->GetBox().GetSize(Rml::BoxArea::Border);
            needs_update = true;
        }

        const float dialog_width = dialog_size.x;
        const float dialog_height = dialog_size.y;
        const float viewport_center_x = (vp_x - screen_x) + vp_w * 0.5f;
        const float viewport_center_y = (vp_y - screen_y) + vp_h * 0.5f;
        const float max_left = std::max(margin, static_cast<float>(width_) - margin - dialog_width);
        const float max_top = std::max(margin, static_cast<float>(height_) - margin - dialog_height);
        const float dialog_left = std::clamp(viewport_center_x - dialog_width * 0.5f, margin, max_left);
        const float dialog_top = std::clamp(viewport_center_y - dialog_height * 0.5f, margin, max_top);
        if (!dialog_position_valid_ || std::abs(dialog_left - last_dialog_left_) > 0.5f ||
            std::abs(dialog_top - last_dialog_top_) > 0.5f) {
            el_dialog_->SetProperty("left", std::format("{}px", dialog_left));
            el_dialog_->SetProperty("top", std::format("{}px", dialog_top));
            last_dialog_left_ = dialog_left;
            last_dialog_top_ = dialog_top;
            dialog_position_valid_ = true;
            needs_update = true;
            rml_context_->Update();
        }

        const bool refresh_cache = needs_update || direct_cache_.texture == 0;
        render_needed_ = false;
        rml_manager_->queueCachedVulkanContext({
            .context = rml_context_,
            .cache = &direct_cache_,
            .cache_width = width_,
            .cache_height = height_,
            .offset_x = 0.0f,
            .offset_y = 0.0f,
            .draw_width = static_cast<float>(width_),
            .draw_height = static_cast<float>(height_),
            .refresh = refresh_cache,
            .foreground = true,
            .clip = {},
        });
    }

    void RmlProgressOverlay::reloadResources() {
        if (!rml_context_)
            return;

        if (document_) {
            rml_context_->UnloadDocument(document_);
            rml_context_->Update();
        }
        document_ = nullptr;
        el_backdrop_ = nullptr;
        el_dialog_ = nullptr;
        el_title_ = nullptr;
        el_path_ = nullptr;
        el_progress_row_ = nullptr;
        el_progress_ = nullptr;
        el_progress_text_ = nullptr;
        el_stage_ = nullptr;
        el_detail_ = nullptr;
        el_error_ = nullptr;
        el_actions_ = nullptr;
        el_action_ = nullptr;
        elements_cached_ = false;
        presentation_ = {};
        base_rcss_.clear();
        has_theme_signature_ = false;
        width_ = 0;
        height_ = 0;
        last_dialog_content_width_ = 0.0f;
        dialog_position_valid_ = false;
        last_mouse_valid_ = false;
        if (rml_manager_)
            rml_manager_->releaseCachedVulkanContext(direct_cache_);
        render_needed_ = true;

        try {
            const auto rml_path = lfs::vis::getAssetPath("rmlui/progress_overlay.rml");
            document_ = rml_documents::loadDocument(rml_context_, rml_path);
            if (!document_) {
                LOG_ERROR("RmlProgressOverlay: failed to reload progress_overlay.rml");
                return;
            }
            document_->Show();
            cacheElements();
        } catch (const std::exception& e) {
            LOG_ERROR("RmlProgressOverlay: resource not found during reload: {}", e.what());
            return;
        }
        syncTheme();
    }

    void RmlProgressOverlay::preload() {
        initContext();
        syncTheme();
    }

    void RmlProgressOverlay::releaseRendererResources() {
        if (rml_manager_)
            rml_manager_->releaseCachedVulkanContext(direct_cache_);
    }

} // namespace lfs::vis::gui
