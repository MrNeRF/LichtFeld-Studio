/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rmlui/rml_panel_host.hpp"
#include "core/logger.hpp"
#include "gui/panel_layout.hpp"
#include "gui/rmlui/rml_document_utils.hpp"
#include "gui/rmlui/rml_input_utils.hpp"
#include "gui/rmlui/rml_text_input_handler.hpp"
#include "gui/rmlui/rml_theme.hpp"
#include "gui/rmlui/rml_tooltip.hpp"
#include "gui/rmlui/rmlui_manager.hpp"
#include "gui/ui_widgets.hpp"
#include "internal/resource_paths.hpp"
#include "theme/theme.hpp"

#include "gui/rmlui/sdl_rml_key_mapping.hpp"

#include <RmlUi/Core.h>
#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/Input.h>
#include <SDL3/SDL_keyboard.h>
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <format>
#include <imgui_internal.h>
#include <string_view>
#include <unordered_set>

namespace lfs::vis::gui {

    constexpr int kMaxFboSize = 8192;

    static std::string s_frame_tooltip;
    static const void* s_frame_tooltip_target = nullptr;
    static std::chrono::steady_clock::time_point s_frame_tooltip_hover_started_at{};
    static bool s_frame_tooltip_updated_this_frame = false;
    static bool s_frame_wants_keyboard = false;
    static bool s_frame_wants_text_input = false;

    std::string RmlPanelHost::consumeFrameTooltip() {
        const bool updated = s_frame_tooltip_updated_this_frame;
        s_frame_tooltip_updated_this_frame = false;
        if (!updated) {
            s_frame_tooltip.clear();
            s_frame_tooltip_target = nullptr;
            s_frame_tooltip_hover_started_at = {};
            return {};
        }

        if (s_frame_tooltip.empty() || !s_frame_tooltip_target)
            return {};

        const auto now = std::chrono::steady_clock::now();
        if (s_frame_tooltip_hover_started_at == std::chrono::steady_clock::time_point{} ||
            now - s_frame_tooltip_hover_started_at < kRmlTooltipShowDelay)
            return {};

        return s_frame_tooltip;
    }

    void RmlPanelHost::setFrameTooltip(const std::string& tip, const void* hover_target) {
        s_frame_tooltip_updated_this_frame = true;
        if (tip.empty() || !hover_target) {
            s_frame_tooltip.clear();
            s_frame_tooltip_target = nullptr;
            s_frame_tooltip_hover_started_at = {};
            return;
        }

        if (s_frame_tooltip_target != hover_target || s_frame_tooltip != tip) {
            s_frame_tooltip = tip;
            s_frame_tooltip_target = hover_target;
            s_frame_tooltip_hover_started_at = std::chrono::steady_clock::now();
            return;
        }

        s_frame_tooltip = tip;
    }

    bool RmlPanelHost::consumeFrameWantsKeyboard() {
        bool result = s_frame_wants_keyboard;
        s_frame_wants_keyboard = false;
        return result;
    }

    bool RmlPanelHost::consumeFrameWantsTextInput() {
        const bool result = s_frame_wants_text_input;
        s_frame_wants_text_input = false;
        return result;
    }

    namespace {
        bool pointInRoundedRect(const float x, const float y, const float w, const float h,
                                const Rml::CornerSizes& radii) {
            if (x < 0.0f || y < 0.0f || x >= w || y >= h)
                return false;

            const float max_radius = 0.5f * std::min(w, h);
            const float top_left = std::clamp(radii[0], 0.0f, max_radius);
            const float top_right = std::clamp(radii[1], 0.0f, max_radius);
            const float bottom_right = std::clamp(radii[2], 0.0f, max_radius);
            const float bottom_left = std::clamp(radii[3], 0.0f, max_radius);

            const auto inside_corner = [x, y](const float min_x, const float min_y,
                                              const float radius, const float center_x,
                                              const float center_y) {
                if (radius <= 0.0f)
                    return true;
                if (x < min_x || x > min_x + radius || y < min_y || y > min_y + radius)
                    return true;

                const float dx = x - center_x;
                const float dy = y - center_y;
                return (dx * dx + dy * dy) <= (radius * radius);
            };

            return inside_corner(0.0f, 0.0f, top_left, top_left, top_left) &&
                   inside_corner(w - top_right, 0.0f, top_right, w - top_right, top_right) &&
                   inside_corner(w - bottom_right, h - bottom_right, bottom_right,
                                 w - bottom_right, h - bottom_right) &&
                   inside_corner(0.0f, h - bottom_left, bottom_left, bottom_left,
                                 h - bottom_left);
        }

        float maxCornerRadius(const Rml::CornerSizes& radii) {
            return std::max({radii[0], radii[1], radii[2], radii[3]});
        }

        std::filesystem::path resolveDocumentPath(const std::string& rml_path) {
            const auto requested_path = std::filesystem::path(rml_path);
            return requested_path.is_absolute() ? requested_path : lfs::vis::getAssetPath(rml_path);
        }

        bool isThemeProvidedStaticStylesheet(const std::filesystem::path& rcss_path) {
            const auto filename = rcss_path.filename().string();
            return filename == "components.rcss" || filename == "font_fallback.rcss";
        }

        void appendBaseRCSS(std::string& out,
                            std::unordered_set<std::string>& loaded_paths,
                            const std::filesystem::path& rcss_path) {
            if (isThemeProvidedStaticStylesheet(rcss_path))
                return;

            const auto normalized_path = rcss_path.lexically_normal();
            std::error_code ec;
            if (!std::filesystem::exists(normalized_path, ec))
                return;

            const std::string key = normalized_path.generic_string();
            if (!loaded_paths.insert(key).second)
                return;

            const std::string rcss = rml_theme::loadBaseRCSS(normalized_path.string());
            if (rcss.empty())
                return;

            if (!out.empty())
                out += "\n";
            out += rcss;
        }

    } // namespace

    RmlPanelHost::RmlPanelHost(RmlUIManager* manager, std::string context_name,
                               std::string rml_path, std::string inline_rcss)
        : manager_(manager),
          context_name_(std::move(context_name)),
          rml_path_(std::move(rml_path)),
          inline_rcss_(std::move(inline_rcss)) {
        assert(manager_);
    }

    RmlPanelHost::~RmlPanelHost() {
        if (manager_ && manager_->isInitialized())
            manager_->destroyContext(context_name_);
        rml_context_ = nullptr;
        document_ = nullptr;
    }

    bool RmlPanelHost::syncThemeProperties() {
        if (!document_)
            return false;

        const std::size_t theme_signature = rml_theme::currentThemeSignature();
        if (has_theme_signature_ && last_theme_signature_ == theme_signature)
            return false;

        last_theme_signature_ = theme_signature;
        has_theme_signature_ = true;

        if (!base_rcss_loaded_) {
            try {
                const auto document_path = resolveDocumentPath(rml_path_);
                std::unordered_set<std::string> loaded_rcss;
                for (const auto& linked_rcss :
                     rml_documents::loadLinkedStylesheetPaths(document_path)) {
                    appendBaseRCSS(base_rcss_, loaded_rcss, linked_rcss);
                }

                auto sibling_rcss = document_path;
                sibling_rcss.replace_extension(".rcss");
                appendBaseRCSS(base_rcss_, loaded_rcss, sibling_rcss);
            } catch (const std::exception& e) {
                LOG_INFO("RCSS load failed for '{}': {}", rml_path_, e.what());
            }
            if (!inline_rcss_.empty()) {
                if (!base_rcss_.empty())
                    base_rcss_ += "\n";
                base_rcss_ += inline_rcss_;
            }
            base_rcss_loaded_ = true;
        }

        std::string panel_theme = rml_theme::loadBaseRCSS("rmlui/panel_host.theme.rcss");
        try {
            auto theme_path = resolveDocumentPath(rml_path_);
            theme_path.replace_extension(".theme.rcss");
            std::error_code ec;
            if (std::filesystem::exists(theme_path, ec)) {
                if (!panel_theme.empty())
                    panel_theme += "\n";
                panel_theme += rml_theme::loadBaseRCSS(theme_path.string());
            }
        } catch (const std::exception&) {
            // Sibling theme files are optional; missing ones should not produce startup noise.
        }

        rml_theme::applyTheme(document_, base_rcss_, panel_theme);
        content_dirty_ = true;
        return true;
    }

    bool RmlPanelHost::ensureContext() {
        if (rml_context_)
            return true;
        rml_context_ = manager_->createContext(context_name_, 100, 100);
        return rml_context_ != nullptr;
    }

    bool RmlPanelHost::ensureDocumentLoaded() {
        return ensureContext() && loadDocument();
    }

    bool RmlPanelHost::reloadDocument() {
        if (!ensureContext())
            return false;

        const float scroll_top = scroll_el_ ? scroll_el_->GetScrollTop() : 0.0f;
        if (document_) {
            rml_context_->UnloadDocument(document_);
            rml_context_->Update();
        }

        document_ = nullptr;
        frame_el_ = nullptr;
        content_wrap_el_ = nullptr;
        content_el_ = nullptr;
        scroll_el_ = nullptr;
        base_rcss_.clear();
        base_rcss_loaded_ = false;
        has_text_focus_ = false;
        wants_keyboard_ = false;
        has_theme_signature_ = false;
        render_needed_ = true;
        content_dirty_ = true;
        last_forwarded_mx_ = -1;
        last_forwarded_my_ = -1;
        last_hovered_ = false;

        if (!loadDocument())
            return false;

        if (scroll_top > 0.0f)
            restoreScrollTop(scroll_top);
        return true;
    }

    bool RmlPanelHost::loadDocument() {
        if (document_)
            return true;
        try {
            const auto requested_path = std::filesystem::path(rml_path_);
            const auto full_path = requested_path.is_absolute()
                                       ? requested_path
                                       : lfs::vis::getAssetPath(rml_path_);
            document_ = rml_documents::loadDocument(rml_context_, full_path);
            if (document_) {
                syncThemeProperties();
                document_->Show();
                cacheContentElements();
                render_needed_ = true;
            } else {
                LOG_ERROR("RmlUI: failed to load {}", rml_path_);
            }
        } catch (const std::exception& e) {
            LOG_ERROR("RmlUI: resource not found: {}", e.what());
        }
        return document_ != nullptr;
    }

    void RmlPanelHost::cacheContentElements() {
        assert(document_);
        frame_el_ = document_->GetElementById("window-frame");
        content_wrap_el_ = frame_el_ ? frame_el_ : document_->GetElementById("content-wrap");
        content_el_ = document_->GetElementById("content");
        scroll_el_ = document_->GetElementById("content-wrap");
    }

    float RmlPanelHost::computeScrollHeightCap() const {
        if (!scroll_el_)
            return 0.0f;

        const auto& scroll_computed = scroll_el_->GetComputedValues();
        const bool is_scroll_container =
            scroll_computed.overflow_y() != Rml::Style::Overflow::Visible;
        const auto max_height = scroll_computed.max_height();
        if (!is_scroll_container ||
            max_height.type != Rml::Style::LengthPercentage::Length ||
            max_height.value >= (FLT_MAX * 0.5f)) {
            return 0.0f;
        }

        float scroll_box_h = max_height.value;
        if (scroll_computed.box_sizing() != Rml::Style::BoxSizing::BorderBox) {
            const auto& scroll_box = scroll_el_->GetBox();
            scroll_box_h += scroll_box.GetEdge(Rml::BoxArea::Padding, Rml::BoxEdge::Top);
            scroll_box_h += scroll_box.GetEdge(Rml::BoxArea::Padding, Rml::BoxEdge::Bottom);
            scroll_box_h += scroll_box.GetEdge(Rml::BoxArea::Border, Rml::BoxEdge::Top);
            scroll_box_h += scroll_box.GetEdge(Rml::BoxArea::Border, Rml::BoxEdge::Bottom);
        }

        return scroll_box_h;
    }

    float RmlPanelHost::computeContentHeight() const {
        if (content_el_) {
            const float chrome_above =
                content_el_->GetAbsoluteOffset(Rml::BoxArea::Border).y -
                document_->GetAbsoluteOffset(Rml::BoxArea::Border).y;
            float chrome_below = 0.0f;
            if (scroll_el_)
                chrome_below = scroll_el_->GetBox().GetEdge(Rml::BoxArea::Padding, Rml::BoxEdge::Bottom);
            float measured = chrome_above + content_el_->GetOffsetHeight() + chrome_below;

            const float scroll_height_cap = computeScrollHeightCap();
            if (scroll_height_cap > 0.0f && scroll_el_) {
                const float chrome_above_scroll =
                    scroll_el_->GetAbsoluteOffset(Rml::BoxArea::Border).y -
                    document_->GetAbsoluteOffset(Rml::BoxArea::Border).y;
                measured = std::min(measured, chrome_above_scroll + scroll_height_cap);
            }

            return measured;
        }

        if (content_wrap_el_)
            return content_wrap_el_->GetOffsetHeight();

        return document_->GetOffsetHeight();
    }

    float RmlPanelHost::clampScrollTop(const float scroll_top) const {
        if (!scroll_el_)
            return 0.0f;

        const float max_scroll =
            std::max(0.0f, scroll_el_->GetScrollHeight() - scroll_el_->GetClientHeight());
        return std::clamp(scroll_top, 0.0f, max_scroll);
    }

    void RmlPanelHost::restoreScrollTop(const float scroll_top) {
        if (!scroll_el_ || scroll_top <= 0.0f)
            return;

        scroll_el_->SetScrollTop(clampScrollTop(scroll_top));
    }

    void RmlPanelHost::syncDirectLayout(float w, float h) {
        if (w <= 0 || h <= 0)
            return;

        if (!ensureDocumentLoaded())
            return;

        const bool theme_dirty = syncThemeProperties();

        const int pw = static_cast<int>(w);
        int ph = 0;
        float display_h = 0.0f;
        resolveDirectRenderHeight(h, ph, display_h);

        const bool size_dirty = (pw != last_layout_w_ || ph != last_layout_h_);
        const bool need_layout =
            theme_dirty || size_dirty || content_dirty_ || render_needed_ || animation_active_;
        if (!need_layout)
            return;

        const float saved_scroll = scroll_el_ ? scroll_el_->GetScrollTop() : 0.0f;
        if (!updateContextLayout(pw, ph))
            return;

        restoreScrollTop(saved_scroll);

        last_layout_w_ = pw;
        last_layout_h_ = ph;

        if (height_mode_ == PanelHeightMode::Content) {
            last_content_height_ = computeContentHeight();
            if (content_el_)
                last_content_el_height_ = content_el_->GetOffsetHeight();
        }
    }

    bool RmlPanelHost::updateContextLayout(const int pw, const int ph) {
        const bool dims_changed = (pw != last_layout_w_ || ph != last_layout_h_);
        if (dims_changed)
            rml_context_->SetDimensions(Rml::Vector2i(pw, ph));
        if (!dims_changed && !content_dirty_ && !render_needed_ && !animation_active_)
            return false;
        rml_context_->Update();
        last_layout_w_ = pw;
        last_layout_h_ = ph;
        return true;
    }

    void RmlPanelHost::renderIfDirty(int pw, int ph, float& display_h) {
        if (!manager_ || !manager_->getVulkanRenderInterface())
            return;

        const bool theme_dirty = syncThemeProperties();
        const bool size_dirty = (pw != last_fbo_w_ || ph != last_fbo_h_);
        const bool externally_clipped =
            (clip_y_min_ >= 0.0f && clip_y_max_ > clip_y_min_);

        const bool dirty = render_needed_ || content_dirty_ || theme_dirty ||
                           size_dirty || animation_active_;
        if (!dirty)
            return;

        const bool need_content_measure =
            height_mode_ == PanelHeightMode::Content &&
            (pw != last_measure_w_ || ph != last_layout_h_ || content_dirty_ ||
             last_content_height_ <= 0.0f);
        const float saved_scroll = scroll_el_ ? scroll_el_->GetScrollTop() : 0.0f;

        if (need_content_measure) {
            last_measure_w_ = pw;

            int layout_h = ph;
            if (last_content_height_ > 0.0f)
                layout_h = std::max(layout_h, static_cast<int>(std::ceil(last_content_height_)));
            else if (last_content_el_height_ > 0.0f)
                layout_h = std::max(layout_h, static_cast<int>(std::ceil(last_content_el_height_)));
            else if (last_fbo_h_ > 0)
                layout_h = std::max(layout_h, last_fbo_h_);

            layout_h = std::clamp(layout_h, 1, kMaxFboSize);

            float content_h = 0.0f;
            for (int pass = 0; pass < 3; ++pass) {
                const bool dims_changed =
                    (pw != last_layout_w_ || layout_h != last_layout_h_);
                if (dims_changed)
                    rml_context_->SetDimensions(Rml::Vector2i(pw, layout_h));
                rml_context_->Update();
                last_layout_w_ = pw;
                last_layout_h_ = layout_h;
                content_h = computeContentHeight();

                const int measured = std::clamp(
                    static_cast<int>(std::ceil(content_h)), 1, kMaxFboSize);
                if (measured <= layout_h || layout_h == kMaxFboSize)
                    break;

                layout_h = measured;
            }

            last_content_height_ = content_h;
            if (content_el_)
                last_content_el_height_ = content_el_->GetOffsetHeight();
            const int measured = std::clamp(
                static_cast<int>(std::ceil(content_h)), 1, kMaxFboSize);
            if (externally_clipped) {
                ph = measured;
                display_h = content_h;
            } else if (ph > 0 && ph < measured) {
                display_h = static_cast<float>(ph);
            } else if (forced_height_ > 0 && ph > 0) {
                display_h = static_cast<float>(ph);
            } else {
                ph = measured;
                display_h = static_cast<float>(ph);
            }

            if (pw != last_layout_w_ || ph != last_layout_h_)
                updateContextLayout(pw, ph);

            restoreScrollTop(saved_scroll);
        } else {
            updateContextLayout(pw, ph);
            restoreScrollTop(saved_scroll);
        }

        content_dirty_ = false;
        if (height_mode_ != PanelHeightMode::Content)
            last_content_height_ = display_h;

        animation_active_ = (rml_context_->GetNextUpdateDelay() == 0);
        last_fbo_w_ = pw;
        last_fbo_h_ = ph;
        render_needed_ = false;

        if (height_mode_ == PanelHeightMode::Content) {
            const float prev_content_h = last_content_height_;
            const float actual_content_h = computeContentHeight();
            last_content_height_ = actual_content_h;

            if (content_el_)
                last_content_el_height_ = content_el_->GetOffsetHeight();

            if (std::abs(actual_content_h - prev_content_h) > 2.0f) {
                content_dirty_ = true;
                last_measure_w_ = 0;
            }
        }
    }

    void RmlPanelHost::draw(const PanelDrawContext& ctx) {
        draw(ctx, 0, 0, 0, 0);
    }

    void RmlPanelHost::draw(const PanelDrawContext& ctx,
                            float avail_w, float avail_h,
                            float pos_x, float pos_y) {
        (void)ctx;

        if (avail_w <= 0 || avail_h <= 0)
            return;

        if (!ensureDocumentLoaded())
            return;

        const int w = static_cast<int>(avail_w);

        int h;
        float display_h;
        if (height_mode_ == PanelHeightMode::Content) {
            h = std::max(1, static_cast<int>(std::ceil(last_content_height_)));
            display_h = last_content_height_;
        } else {
            h = static_cast<int>(avail_h);
            display_h = avail_h;
        }

        if (forwardInput(pos_x, pos_y))
            render_needed_ = true;

        renderIfDirty(w, h, display_h);

        const ImVec2 panel_screen_pos = ImGui::GetCursorScreenPos();
        if (!manager_ || !manager_->getVulkanRenderInterface())
            return;

        const auto* vp = ImGui::GetMainViewport();
        const float screen_x = vp ? vp->Pos.x : 0.0f;
        const float screen_y = vp ? vp->Pos.y : 0.0f;
        const ImVec2 clip_min = ImGui::GetWindowDrawList()->GetClipRectMin();
        const ImVec2 clip_max = ImGui::GetWindowDrawList()->GetClipRectMax();
        const float clip_x1 = std::max(clip_min.x, panel_screen_pos.x);
        const float clip_y1 = std::max(clip_min.y, panel_screen_pos.y);
        const float clip_x2 = std::min(clip_max.x, panel_screen_pos.x + avail_w);
        const float clip_y2 = std::min(clip_max.y, panel_screen_pos.y + display_h);
        ImGui::Dummy(ImVec2(avail_w, display_h));
        if (clip_x2 <= clip_x1 || clip_y2 <= clip_y1)
            return;
        manager_->queueVulkanContext(rml_context_,
                                     panel_screen_pos.x - screen_x,
                                     panel_screen_pos.y - screen_y,
                                     foreground_,
                                     true,
                                     clip_x1 - screen_x,
                                     clip_y1 - screen_y,
                                     clip_x2 - screen_x,
                                     clip_y2 - screen_y);
        if (auto* vp = ImGui::GetMainViewport()) {
            const auto popup_shadow =
                collectVisibleColorPickerPopupShadow(panel_screen_pos.x, panel_screen_pos.y);
            if (popup_shadow) {
                const auto& shadow = *popup_shadow;
                auto* fg = ImGui::GetForegroundDrawList(vp);
                widgets::DrawPopoverShadowOverlay(fg,
                                                  {shadow.x, shadow.y},
                                                  {shadow.w, shadow.h},
                                                  shadow.rounding);
            }
        }
    }

    void RmlPanelHost::resolveDirectRenderHeight(float requested_h, int& ph, float& display_h) const {
        if (height_mode_ == PanelHeightMode::Content) {
            const float ch = last_content_height_;
            if (ch > 0 && requested_h < ch) {
                ph = static_cast<int>(requested_h);
                display_h = requested_h;
            } else if (ch > 0) {
                const float eff = (forced_height_ > 0) ? std::max(forced_height_, ch) : ch;
                ph = std::max(1, static_cast<int>(std::ceil(eff)));
                display_h = eff;
            } else {
                float initial_h = requested_h;
                if (clip_y_min_ >= 0.0f && clip_y_max_ > clip_y_min_)
                    initial_h = std::min(initial_h, clip_y_max_ - clip_y_min_);
                if (input_ && input_->screen_h > 0)
                    initial_h = std::min(initial_h, static_cast<float>(input_->screen_h));
                if (last_fbo_h_ > 0)
                    initial_h = std::min(initial_h, static_cast<float>(last_fbo_h_));
                if (!std::isfinite(initial_h) || initial_h <= 0.0f)
                    initial_h = std::min(requested_h, 1024.0f);

                ph = std::clamp(static_cast<int>(std::ceil(initial_h)), 1, kMaxFboSize);
                display_h = static_cast<float>(ph);
            }
        } else {
            float effective_h = requested_h;
            if (clip_y_min_ >= 0.0f && clip_y_max_ > clip_y_min_)
                effective_h = std::min(effective_h, clip_y_max_ - clip_y_min_);
            ph = std::min(kMaxFboSize, static_cast<int>(effective_h));
            display_h = static_cast<float>(ph);
        }
    }

    void RmlPanelHost::prepareDirect(float w, float h) {
        if (w <= 0 || h <= 0)
            return;

        if (!ensureDocumentLoaded())
            return;

        const int pw = static_cast<int>(w);
        int ph = 0;
        float display_h = 0.0f;
        resolveDirectRenderHeight(h, ph, display_h);

        renderIfDirty(pw, ph, display_h);
    }

    void RmlPanelHost::drawDirect(float x, float y, float w, float h) {
        if (w <= 0 || h <= 0)
            return;

        if (!ensureDocumentLoaded())
            return;

        const int pw = static_cast<int>(w);
        int ph;
        float display_h;
        resolveDirectRenderHeight(h, ph, display_h);

        if (forwardInput(x, y))
            render_needed_ = true;

        renderIfDirty(pw, ph, display_h);
        compositeDirectToScreen(x, y, w, display_h);
    }

    std::optional<RmlPanelHost::ShadowRect> RmlPanelHost::collectVisibleColorPickerPopupShadow(
        const float panel_screen_x, const float panel_screen_y) const {
        if (!document_)
            return std::nullopt;

        auto* popup = document_->GetElementById("color-picker-popup");
        if (!popup || !popup->IsClassSet("visible"))
            return std::nullopt;

        float popup_x = 0.0f;
        float popup_y = 0.0f;
        float popup_w = 0.0f;
        float popup_h = 0.0f;

        if (auto* picker = popup->GetElementById("color-picker-el")) {
            const auto picker_size = picker->GetBox().GetSize(Rml::BoxArea::Border);
            if (picker_size.x > 0.0f && picker_size.y > 0.0f) {
                const auto picker_pos = picker->GetAbsoluteOffset(Rml::BoxArea::Border);
                const auto& popup_box = popup->GetBox();
                const float extra_left =
                    popup_box.GetEdge(Rml::BoxArea::Padding, Rml::BoxEdge::Left) +
                    popup_box.GetEdge(Rml::BoxArea::Border, Rml::BoxEdge::Left);
                const float extra_top =
                    popup_box.GetEdge(Rml::BoxArea::Padding, Rml::BoxEdge::Top) +
                    popup_box.GetEdge(Rml::BoxArea::Border, Rml::BoxEdge::Top);
                const float extra_right =
                    popup_box.GetEdge(Rml::BoxArea::Padding, Rml::BoxEdge::Right) +
                    popup_box.GetEdge(Rml::BoxArea::Border, Rml::BoxEdge::Right);
                const float extra_bottom =
                    popup_box.GetEdge(Rml::BoxArea::Padding, Rml::BoxEdge::Bottom) +
                    popup_box.GetEdge(Rml::BoxArea::Border, Rml::BoxEdge::Bottom);

                popup_x = picker_pos.x - extra_left;
                popup_y = picker_pos.y - extra_top;
                popup_w = picker_size.x + extra_left + extra_right;
                popup_h = picker_size.y + extra_top + extra_bottom;
            }
        }

        if (popup_w <= 0.0f || popup_h <= 0.0f) {
            const auto popup_size = popup->GetBox().GetSize(Rml::BoxArea::Border);
            if (popup_size.x <= 0.0f || popup_size.y <= 0.0f)
                return std::nullopt;

            const auto popup_pos = popup->GetAbsoluteOffset(Rml::BoxArea::Border);
            popup_x = popup_pos.x;
            popup_y = popup_pos.y;
            popup_w = popup_size.x;
            popup_h = popup_size.y;
        }

        if (popup_w <= 0.0f || popup_h <= 0.0f)
            return std::nullopt;

        return ShadowRect{
            .x = panel_screen_x + popup_x,
            .y = panel_screen_y + popup_y,
            .w = popup_w,
            .h = popup_h,
            .rounding = maxCornerRadius(popup->GetComputedValues().border_radius()),
        };
    }

    void RmlPanelHost::compositeDirectToScreen(const float x, const float y,
                                               const float w, const float h) const {
        if (!input_ || !manager_ || !manager_->getVulkanRenderInterface() ||
            w <= 0.0f || h <= 0.0f)
            return;

        float clip_x1 = x;
        float clip_y1 = y;
        float clip_x2 = x + w;
        float clip_y2 = y + h;

        if (clip_y_min_ >= 0.0f && clip_y_max_ > clip_y_min_) {
            clip_y1 = std::max(clip_y1, clip_y_min_);
            clip_y2 = std::min(clip_y2, clip_y_max_);
        }

        if (clip_x2 <= clip_x1 || clip_y2 <= clip_y1)
            return;

        const float screen_x = x - input_->screen_x;
        const float screen_y = y - input_->screen_y;
        const float screen_clip_x1 = clip_x1 - input_->screen_x;
        const float screen_clip_y1 = clip_y1 - input_->screen_y;
        const float screen_clip_x2 = clip_x2 - input_->screen_x;
        const float screen_clip_y2 = clip_y2 - input_->screen_y;
        manager_->queueVulkanContext(rml_context_, screen_x, screen_y, foreground_,
                                     true,
                                     screen_clip_x1, screen_clip_y1,
                                     screen_clip_x2, screen_clip_y2);

        if (const auto popover_shadow = collectVisibleColorPickerPopupShadow(screen_x, screen_y)) {
            const auto& shadow = *popover_shadow;
            widgets::DrawPopoverShadowOverlay(ImGui::GetForegroundDrawList(),
                                              {shadow.x, shadow.y},
                                              {shadow.w, shadow.h},
                                              shadow.rounding);
        }
    }

    bool RmlPanelHost::hitTestPanelShape(const float local_x, const float local_y,
                                         const float logical_w, const float logical_h) {
        if (local_x < 0.0f || local_y < 0.0f || local_x >= logical_w || local_y >= logical_h)
            return false;

        if (!frame_el_)
            return true;

        const auto frame_pos = frame_el_->GetAbsoluteOffset(Rml::BoxArea::Border);
        const auto frame_size = frame_el_->GetBox().GetSize(Rml::BoxArea::Border);
        if (frame_size.x <= 0.0f || frame_size.y <= 0.0f)
            return true;

        return pointInRoundedRect(local_x - frame_pos.x, local_y - frame_pos.y,
                                  frame_size.x, frame_size.y,
                                  frame_el_->GetComputedValues().border_radius());
    }

    bool RmlPanelHost::forwardInput(float panel_x, float panel_y) {
        assert(rml_context_);

        if (!input_ || !manager_ || !manager_->getVulkanRenderInterface())
            return false;

        bool had_input = false;
        const auto& input = *input_;
        auto* const text_input_handler = manager_ ? manager_->getTextInputHandler() : nullptr;
        if (manager_) {
            manager_->trackContextFrame(rml_context_,
                                        static_cast<int>(panel_x - input.screen_x),
                                        static_cast<int>(panel_y - input.screen_y));
        }
        const float mouse_x = input.mouse_x;
        const float mouse_y = input.mouse_y;
        const auto sync_text_focus = [&]() {
            const bool want_text = rml_input::wantsTextInput(rml_context_->GetFocusElement());
            if (want_text == has_text_focus_)
                return;

            has_text_focus_ = want_text;
        };
        const auto flush_pending_text_input = [&]() {
            if (!has_text_focus_)
                return;

            auto* const focused = rml_context_->GetFocusElement();
            const bool focused_editable = rml_input::isTextEditableElement(focused);

            if (focused_editable && text_input_handler && input.has_text_editing) {
                had_input |= text_input_handler->handleTextEditing(
                    input.text_editing, input.text_editing_start, input.text_editing_length);
            }

            bool forward_text_codepoints = input.text_inputs.empty();
            for (const auto& text_input : input.text_inputs) {
                had_input = true;
                if (focused_editable && text_input_handler &&
                    text_input_handler->handleTextInput(text_input)) {
                    continue;
                }
                if (focused && rml_input::isCustomTextInputElement(focused)) {
                    rml_context_->ProcessTextInput(text_input);
                } else {
                    forward_text_codepoints = true;
                }
            }

            if (forward_text_codepoints) {
                if (!input.text_codepoints.empty())
                    had_input = true;
                for (const uint32_t cp : input.text_codepoints)
                    rml_context_->ProcessTextInput(static_cast<Rml::Character>(cp));
            }
        };
        const auto blur_focused_element = [&]() {
            auto* const focused = rml_context_->GetFocusElement();
            if (!focused)
                return;

            if (rml_input::wantsTextInput(focused))
                flush_pending_text_input();
            focused->Blur();
            sync_text_focus();
        };

        float local_x = mouse_x - panel_x;
        float local_y = mouse_y - panel_y;

        const float logical_w = static_cast<float>(last_fbo_w_);
        const float logical_h = static_cast<float>(last_fbo_h_);

        bool hovered = hitTestPanelShape(local_x, local_y, logical_w, logical_h);

        if (hovered && clip_y_min_ >= 0 && clip_y_max_ > clip_y_min_) {
            if (mouse_y < clip_y_min_ || mouse_y > clip_y_max_)
                hovered = false;
        }

        const bool hover_changed = (hovered != last_hovered_);
        if (hover_changed) {
            last_hovered_ = hovered;
            had_input = true;
            if (!hovered) {
                last_forwarded_mx_ = -1;
                last_forwarded_my_ = -1;
                rml_context_->ProcessMouseLeave();
            }
        }

        const int rml_mx = static_cast<int>(local_x);
        const int rml_my = static_cast<int>(local_y);
        const bool mouse_moved = hovered &&
                                 (rml_mx != last_forwarded_mx_ || rml_my != last_forwarded_my_);
        if (mouse_moved) {
            had_input = true;
        }

        if (input.mouse_clicked[0] || input.mouse_released[0] ||
            input.mouse_clicked[1] || input.mouse_released[1] ||
            input.mouse_wheel != 0.0f)
            had_input = true;

        const int mods = sdlModsToRml(input.key_ctrl, input.key_shift,
                                      input.key_alt, input.key_super);

        if (mouse_moved) {
            last_forwarded_mx_ = rml_mx;
            last_forwarded_my_ = rml_my;
            rml_context_->ProcessMouseMove(rml_mx, rml_my, mods);
        }
        if (hovered) {
            if (input.mouse_clicked[0])
                rml_context_->ProcessMouseButtonDown(0, mods);
            if (input.mouse_released[0])
                rml_context_->ProcessMouseButtonUp(0, mods);

            if (input.mouse_clicked[1])
                rml_context_->ProcessMouseButtonDown(1, mods);
            if (input.mouse_released[1])
                rml_context_->ProcessMouseButtonUp(1, mods);

            if (input.mouse_wheel != 0.0f)
                rml_context_->ProcessMouseWheel(Rml::Vector2f(0, -input.mouse_wheel), mods);

            if (input.mouse_clicked[0])
                sync_text_focus();
        } else if (input.mouse_clicked[0]) {
            blur_focused_element();
        }

        if (hovered) {
            auto* hover = rml_context_->GetHoverElement();
            if (hover)
                setFrameTooltip(resolveRmlTooltip(hover), hover);
        }

        if (input.viewport_keyboard_focus) {
            blur_focused_element();
        }

        bool forward_keys =
            rml_input::hasFocusedKeyboardTarget(rml_context_->GetFocusElement()) &&
            !input.viewport_keyboard_focus;
        bool commit_requested = false;
        bool escape_requested = false;
        const bool composing = text_input_handler && text_input_handler->isComposing();
        auto isNumpadTextKey = [](int sc) {
            return (sc >= SDL_SCANCODE_KP_1 && sc <= SDL_SCANCODE_KP_0) ||
                   sc == SDL_SCANCODE_KP_PERIOD;
        };

        if (forward_keys) {
            const auto process_key_down = [&](const int sc) {
                if (!composing && sc == SDL_SCANCODE_ESCAPE) {
                    if (auto* const focused = rml_context_->GetFocusElement();
                        focused && (rml_input::isTextEditableElement(focused) ||
                                    rml_input::isSelectRelatedElement(focused))) {
                        escape_requested = true;
                        had_input = true;
                        return;
                    }
                }
                const bool is_submit_key =
                    (sc == SDL_SCANCODE_RETURN || sc == SDL_SCANCODE_KP_ENTER);
                if (composing && (is_submit_key || sc == SDL_SCANCODE_ESCAPE))
                    return;
                if (has_text_focus_ && isNumpadTextKey(sc))
                    return;
                auto rml_key = sdlScancodeToRml(static_cast<SDL_Scancode>(sc));
                if (rml_key != Rml::Input::KI_UNKNOWN) {
                    if (text_input_handler && text_input_handler->handleKeyDown(rml_key, mods)) {
                        had_input = true;
                        return;
                    }
                    rml_context_->ProcessKeyDown(rml_key, mods);
                    had_input = true;
                }
                if (is_submit_key)
                    commit_requested = true;
            };

            for (int sc : input.keys_pressed)
                process_key_down(sc);
            for (int sc : input.keys_repeated)
                process_key_down(sc);
            for (int sc : input.keys_released) {
                if (escape_requested && sc == SDL_SCANCODE_ESCAPE)
                    continue;
                if (composing && (sc == SDL_SCANCODE_RETURN || sc == SDL_SCANCODE_KP_ENTER ||
                                  sc == SDL_SCANCODE_ESCAPE))
                    continue;
                if (has_text_focus_ && isNumpadTextKey(sc))
                    continue;
                auto rml_key = sdlScancodeToRml(static_cast<SDL_Scancode>(sc));
                if (rml_key != Rml::Input::KI_UNKNOWN) {
                    rml_context_->ProcessKeyUp(rml_key, mods);
                    had_input = true;
                }
            }
        }

        if (!composing && escape_requested) {
            if (rml_input::cancelFocusedElement(*rml_context_)) {
                sync_text_focus();
                had_input = true;
            }
        }

        if (!composing && commit_requested &&
            rml_input::isSingleLineTextInput(rml_context_->GetFocusElement())) {
            blur_focused_element();
        }

        sync_text_focus();

        auto* const focused = rml_context_->GetFocusElement();
        wants_keyboard_ = rml_input::hasFocusedKeyboardTarget(focused);
        if (wants_keyboard_)
            s_frame_wants_keyboard = true;

        if (has_text_focus_) {
            s_frame_wants_text_input = true;
            flush_pending_text_input();
        }

        return had_input;
    }

} // namespace lfs::vis::gui
