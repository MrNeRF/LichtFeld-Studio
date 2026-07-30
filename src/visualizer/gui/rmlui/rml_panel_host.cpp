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
#include "internal/resource_paths.hpp"
#include "theme/theme.hpp"

#include "gui/rmlui/sdl_rml_key_mapping.hpp"

#include <RmlUi/Core.h>
#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/Elements/ElementFormControlSelect.h>
#include <RmlUi/Core/Input.h>
#include <SDL3/SDL_keyboard.h>
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <format>
#include <string_view>
#include <unordered_set>

namespace lfs::vis::gui {

    constexpr int kMaxFboSize = 8192;

    static bool s_frame_wants_keyboard = false;
    static bool s_frame_wants_text_input = false;

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

        bool pointInRmlRect(const RmlRect& rect, const float x, const float y) {
            return x >= rect.x1 && y >= rect.y1 && x < rect.x2 && y < rect.y2;
        }

        std::optional<RmlRect> elementBorderRect(Rml::Element* const element) {
            if (!element)
                return std::nullopt;

            const auto size = element->GetBox().GetSize(Rml::BoxArea::Border);
            if (size.x <= 0.0f || size.y <= 0.0f)
                return std::nullopt;

            const auto pos = element->GetAbsoluteOffset(Rml::BoxArea::Border);
            return RmlRect{
                .x1 = pos.x,
                .y1 = pos.y,
                .x2 = pos.x + size.x,
                .y2 = pos.y + size.y,
            };
        }

        Rml::Element* selectboxElement(Rml::ElementFormControlSelect* const select) {
            if (!select)
                return nullptr;

            for (int i = 0; i < select->GetNumChildren(true); ++i) {
                auto* const child = select->GetChild(i);
                if (child && child->GetTagName() == "selectbox")
                    return child;
            }
            return nullptr;
        }

        Rml::ElementFormControlSelect* asOpenSelect(Rml::Element* const element) {
            auto* const select = dynamic_cast<Rml::ElementFormControlSelect*>(element);
            return (select && select->IsSelectBoxVisible()) ? select : nullptr;
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
        if (manager_ && manager_->isInitialized()) {
            manager_->releaseCachedVulkanContext(direct_cache_);
            manager_->destroyContext(context_name_);
        }
        rml_context_ = nullptr;
        document_ = nullptr;
    }

    void RmlPanelHost::releaseRendererResources() {
        if (manager_ && manager_->isInitialized())
            manager_->releaseCachedVulkanContext(direct_cache_);
        direct_cache_dirty_ = true;
    }

    void RmlPanelHost::setFloating(const bool floating) {
        if (floating_ == floating)
            return;

        floating_ = floating;
        last_layout_padding_ = -1;
        render_needed_ = true;
        content_dirty_ = true;
        direct_cache_dirty_ = true;
        applyPanelSpaceClass();
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
        direct_cache_dirty_ = true;
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
        direct_cache_dirty_ = true;
        if (manager_ && manager_->isInitialized())
            manager_->releaseCachedVulkanContext(direct_cache_);
        last_forwarded_mx_ = -1;
        last_forwarded_my_ = -1;
        last_hovered_ = false;
        manual_dropdown_hover_ = nullptr;
        manual_dropdown_mouse_captured_ = false;
        for (auto& captured : mouse_captured_)
            captured = false;

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
                applyPanelSpaceClass();
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

    int RmlPanelHost::renderPadding() const {
        if (!floating_ || !manager_)
            return 0;
        return static_cast<int>(std::ceil(
            rml_theme::layeredShadowPadding(lfs::vis::theme(), 4) *
            manager_->getDpRatio()));
    }

    void RmlPanelHost::applyPanelSpaceClass() {
        if (document_)
            document_->SetClass("floating-panel", floating_);
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
            const float root_y = frame_el_
                                     ? frame_el_->GetAbsoluteOffset(Rml::BoxArea::Border).y
                                     : document_->GetAbsoluteOffset(Rml::BoxArea::Border).y;
            const float chrome_above =
                content_el_->GetAbsoluteOffset(Rml::BoxArea::Border).y -
                root_y;
            float chrome_below = 0.0f;
            if (scroll_el_)
                chrome_below = scroll_el_->GetBox().GetEdge(Rml::BoxArea::Padding, Rml::BoxEdge::Bottom);
            float measured = chrome_above + content_el_->GetOffsetHeight() + chrome_below;

            const float scroll_height_cap = computeScrollHeightCap();
            if (scroll_height_cap > 0.0f && scroll_el_) {
                const float chrome_above_scroll =
                    scroll_el_->GetAbsoluteOffset(Rml::BoxArea::Border).y -
                    root_y;
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

        const bool size_dirty =
            pw != last_layout_w_ || ph != last_layout_h_ ||
            renderPadding() != last_layout_padding_;
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
        const int padding = renderPadding();
        const bool dims_changed =
            pw != last_layout_w_ || ph != last_layout_h_ ||
            padding != last_layout_padding_;
        if (dims_changed) {
            rml_context_->SetDimensions(
                Rml::Vector2i(pw + 2 * padding, ph + 2 * padding));
            applyPanelSpaceClass();
            last_layout_padding_ = padding;
        }
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
        const int padding = renderPadding();
        const bool size_dirty =
            pw != last_fbo_w_ || ph != last_fbo_h_ ||
            padding != last_fbo_padding_;
        const bool externally_clipped =
            (clip_y_min_ >= 0.0f && clip_y_max_ > clip_y_min_);

        const bool dirty = render_needed_ || content_dirty_ || theme_dirty ||
                           size_dirty || animation_active_;
        if (!dirty)
            return;
        direct_cache_dirty_ = true;
        const bool needs_post_layout_update =
            content_dirty_ || render_needed_ || theme_dirty || size_dirty;

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
                    pw != last_layout_w_ || layout_h != last_layout_h_ ||
                    padding != last_layout_padding_;
                if (dims_changed) {
                    rml_context_->SetDimensions(
                        Rml::Vector2i(pw + 2 * padding, layout_h + 2 * padding));
                    applyPanelSpaceClass();
                    last_layout_padding_ = padding;
                }
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

        if (needs_post_layout_update)
            rml_context_->Update();

        content_dirty_ = false;
        if (height_mode_ != PanelHeightMode::Content)
            last_content_height_ = display_h;

        animation_active_ = (rml_context_->GetNextUpdateDelay() == 0);
        last_fbo_w_ = pw;
        last_fbo_h_ = ph;
        last_fbo_padding_ = padding;
        render_needed_ = false;

        if (height_mode_ == PanelHeightMode::Content) {
            const float prev_content_h = last_content_height_;
            const float actual_content_h = computeContentHeight();
            last_content_height_ = actual_content_h;

            if (content_el_)
                last_content_el_height_ = content_el_->GetOffsetHeight();

            if (std::abs(actual_content_h - prev_content_h) > 2.0f) {
                content_dirty_ = true;
                direct_cache_dirty_ = true;
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
        applyHoverTooltip(w, pos_y, display_h);

        renderIfDirty(w, h, display_h);
        trackFrame(pos_x, pos_y);

        if (!manager_ || !manager_->getVulkanRenderInterface())
            return;

        const float screen_origin_x = input_ ? input_->screen_x : 0.0f;
        const float screen_origin_y = input_ ? input_->screen_y : 0.0f;
        const float clip_x1 = pos_x;
        const float clip_y1 = pos_y;
        const float clip_x2 = pos_x + avail_w;
        const float clip_y2 = pos_y + display_h;
        if (clip_x2 <= clip_x1 || clip_y2 <= clip_y1)
            return;
        manager_->queueVulkanContext(rml_context_,
                                     pos_x - screen_origin_x,
                                     pos_y - screen_origin_y,
                                     foreground_,
                                     true,
                                     clip_x1 - screen_origin_x,
                                     clip_y1 - screen_origin_y,
                                     clip_x2 - screen_origin_x,
                                     clip_y2 - screen_origin_y);
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
        applyHoverTooltip(pw, y, display_h);

        renderIfDirty(pw, ph, display_h);
        trackFrame(x, y);
        compositeDirectToScreen(x, y, w, display_h);
    }

    bool RmlPanelHost::drawDirectCached(float x, float y, float w, float h) {
        if (w <= 0 || h <= 0)
            return false;
        if (!document_ || !rml_context_ || last_fbo_w_ <= 0 || last_fbo_h_ <= 0)
            return false;
        if (render_needed_ || content_dirty_ || animation_active_ || tooltip_.revealDue())
            return false;
        if (!has_theme_signature_ || rml_theme::currentThemeSignature() != last_theme_signature_)
            return false;

        const int pw = static_cast<int>(w);
        if (pw != last_fbo_w_)
            return false;
        if (renderPadding() != last_fbo_padding_)
            return false;

        int ph = 0;
        float display_h = 0.0f;
        resolveDirectRenderHeight(h, ph, display_h);
        if (ph <= 0 || display_h <= 0.0f || ph > last_fbo_h_)
            return false;

        trackFrame(x, y);

        if (input_ && manager_ &&
            manager_->activeOverlayOccludesContext(rml_context_, input_->mouse_x, input_->mouse_y)) {
            compositeDirectToScreen(x, y, w, display_h);
            return true;
        }

        const auto dropdown_bounds = openDropdownBounds();
        if ((tooltip_.hasActiveState() || dropdown_bounds) && input_) {
            const float local_x = input_->mouse_x - x + last_fbo_padding_;
            const float local_y = input_->mouse_y - y + last_fbo_padding_;
            const bool dropdown_hovered =
                dropdown_bounds && pointInRmlRect(*dropdown_bounds, local_x, local_y);
            bool hovered = dropdown_hovered ||
                           hitTestPanelShape(local_x, local_y,
                                             static_cast<float>(last_fbo_w_),
                                             static_cast<float>(last_fbo_h_));

            if (!dropdown_hovered &&
                hovered && clip_y_min_ >= 0.0f && clip_y_max_ > clip_y_min_) {
                if (input_->mouse_y < clip_y_min_ || input_->mouse_y > clip_y_max_)
                    hovered = false;
            }

            const bool any_capture =
                manual_dropdown_mouse_captured_ ||
                mouse_captured_[0] || mouse_captured_[1] || mouse_captured_[2];
            const bool effective_hovered = hovered || any_capture;
            const int rml_mx = static_cast<int>(local_x);
            const int rml_my = static_cast<int>(local_y);
            const bool pointer_event =
                input_->mouse_clicked[0] || input_->mouse_released[0] ||
                input_->mouse_clicked[1] || input_->mouse_released[1] ||
                input_->mouse_wheel != 0.0f;

            if (pointer_event ||
                effective_hovered != last_hovered_ ||
                (effective_hovered &&
                 (rml_mx != last_forwarded_mx_ || rml_my != last_forwarded_my_))) {
                return false;
            }
        }

        compositeDirectToScreen(x, y, w, display_h);
        return true;
    }

    std::optional<RmlRect> RmlPanelHost::openDropdownBounds() const {
        if (!document_)
            return std::nullopt;

        Rml::ElementList selects;
        document_->GetElementsByTagName(selects, "select");
        for (auto* const element : selects) {
            auto* const select = asOpenSelect(element);
            if (!select)
                continue;

            if (auto bounds = elementBorderRect(selectboxElement(select)))
                return bounds;
        }

        return std::nullopt;
    }

    bool RmlPanelHost::openDropdownContainsPoint(const float local_x,
                                                 const float local_y) const {
        const auto bounds = openDropdownBounds();
        return bounds && pointInRmlRect(*bounds, local_x, local_y);
    }

    Rml::Element* RmlPanelHost::openDropdownOptionAtPoint(const float local_x,
                                                          const float local_y) const {
        if (!document_)
            return nullptr;

        Rml::ElementList selects;
        document_->GetElementsByTagName(selects, "select");
        for (auto* const element : selects) {
            auto* const select = asOpenSelect(element);
            if (!select)
                continue;

            auto* const selectbox = selectboxElement(select);
            const auto box_rect = elementBorderRect(selectbox);
            if (!box_rect || !pointInRmlRect(*box_rect, local_x, local_y))
                continue;

            for (int i = select->GetNumOptions() - 1; i >= 0; --i) {
                auto* const option = select->GetOption(i);
                const auto option_rect = elementBorderRect(option);
                if (!option_rect)
                    continue;

                if (pointInRmlRect(*option_rect, local_x, local_y))
                    return option;
            }

            return selectbox;
        }

        return nullptr;
    }

    void RmlPanelHost::setManualDropdownHover(Rml::Element* const option) {
        if (manual_dropdown_hover_ == option)
            return;

        if (manual_dropdown_hover_)
            manual_dropdown_hover_->SetPseudoClass("hover", false);
        manual_dropdown_hover_ = option;
        if (manual_dropdown_hover_)
            manual_dropdown_hover_->SetPseudoClass("hover", true);
    }

    void RmlPanelHost::trackFrame(const float panel_x, const float panel_y) {
        if (!manager_ || !rml_context_ || !input_)
            return;

        manager_->trackContextFrame(rml_context_,
                                    static_cast<int>(panel_x - input_->screen_x -
                                                     last_fbo_padding_),
                                    static_cast<int>(panel_y - input_->screen_y -
                                                     last_fbo_padding_),
                                    openDropdownBounds());
    }

    void RmlPanelHost::applyHoverTooltip(const int pw, const float panel_y,
                                         const float display_h) {
        if (!document_)
            return;
        Rml::Element* body = document_->GetElementById("body");
        if (!body)
            body = document_;
        float visible_h = display_h;
        if (clip_y_min_ >= 0.0f && clip_y_max_ > clip_y_min_)
            visible_h = std::min(visible_h, clip_y_max_ - panel_y);
        const int clamp_h = std::max(1, static_cast<int>(std::floor(visible_h)));
        if (manager_)
            manager_->setContextNeedsPassiveMouseMoveFrames(rml_context_,
                                                            tooltip_.hasActiveState());
        if (tooltip_.apply(body,
                           last_forwarded_mx_ - last_fbo_padding_,
                           last_forwarded_my_ - last_fbo_padding_,
                           pw, clamp_h))
            render_needed_ = true;
        if (manager_) {
            manager_->setContextNeedsPassiveMouseMoveFrames(rml_context_,
                                                            tooltip_.hasActiveState());
            manager_->setContextTooltipRevealDeadline(rml_context_, tooltip_.revealDeadline());
        }
    }

    void RmlPanelHost::compositeDirectToScreen(const float x, const float y,
                                               const float w, const float h) {
        if (!input_ || !manager_ || !manager_->getVulkanRenderInterface() ||
            w <= 0.0f || h <= 0.0f)
            return;

        const float padding = static_cast<float>(last_fbo_padding_);
        float clip_x1 = x - padding;
        float clip_y1 = y - padding;
        float clip_x2 = x + w + padding;
        float clip_y2 = y + h + padding;

        if (clip_y_min_ >= 0.0f && clip_y_max_ > clip_y_min_) {
            clip_y1 = std::max(clip_y1, clip_y_min_);
            clip_y2 = std::min(clip_y2, clip_y_max_);
        }

        if (clip_x2 <= clip_x1 || clip_y2 <= clip_y1)
            return;

        const float screen_x = x - input_->screen_x - padding;
        const float screen_y = y - input_->screen_y - padding;
        const float screen_clip_x1 = clip_x1 - input_->screen_x;
        const float screen_clip_y1 = clip_y1 - input_->screen_y;
        const float screen_clip_x2 = clip_x2 - input_->screen_x;
        const float screen_clip_y2 = clip_y2 - input_->screen_y;
        if (animation_active_) {
            manager_->queueVulkanContext(rml_context_,
                                         screen_x,
                                         screen_y,
                                         foreground_,
                                         true,
                                         screen_clip_x1,
                                         screen_clip_y1,
                                         screen_clip_x2,
                                         screen_clip_y2);
            direct_cache_dirty_ = true;
        } else {
            const float draw_w =
                last_fbo_w_ > 0
                    ? static_cast<float>(last_fbo_w_ + 2 * last_fbo_padding_)
                    : w;
            const float draw_h =
                last_fbo_h_ > 0
                    ? static_cast<float>(last_fbo_h_ + 2 * last_fbo_padding_)
                    : h;
            manager_->queueCachedVulkanContext({
                .context = rml_context_,
                .cache = &direct_cache_,
                .cache_width = last_fbo_w_ + 2 * last_fbo_padding_,
                .cache_height = last_fbo_h_ + 2 * last_fbo_padding_,
                .offset_x = screen_x,
                .offset_y = screen_y,
                .draw_width = draw_w,
                .draw_height = draw_h,
                .refresh = direct_cache_dirty_,
                .foreground = foreground_,
                .clip_enabled = true,
                .cache_visible_region = true,
                .clip = {
                    .x1 = screen_clip_x1,
                    .y1 = screen_clip_y1,
                    .x2 = screen_clip_x2,
                    .y2 = screen_clip_y2,
                },
            });
            direct_cache_dirty_ = false;
        }
    }

    bool RmlPanelHost::hitTestPanelShape(const float local_x, const float local_y,
                                         const float logical_w, const float logical_h) const {
        const float panel_x = local_x - static_cast<float>(last_fbo_padding_);
        const float panel_y = local_y - static_cast<float>(last_fbo_padding_);
        if (panel_x < 0.0f || panel_y < 0.0f ||
            panel_x >= logical_w || panel_y >= logical_h)
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
        trackFrame(panel_x, panel_y);
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
        const auto blur_focused_element = [&]() -> bool {
            auto* const focused = rml_context_->GetFocusElement();
            if (!focused)
                return false;

            if (rml_input::wantsTextInput(focused))
                flush_pending_text_input();
            focused->Blur();
            sync_text_focus();
            return true;
        };

        float local_x = mouse_x - panel_x + last_fbo_padding_;
        float local_y = mouse_y - panel_y + last_fbo_padding_;

        const float logical_w = static_cast<float>(last_fbo_w_);
        const float logical_h = static_cast<float>(last_fbo_h_);

        const bool blocked_by_other_overlay =
            manager_ && manager_->activeOverlayOccludesContext(rml_context_, mouse_x, mouse_y);
        const bool dropdown_hovered =
            !blocked_by_other_overlay && openDropdownContainsPoint(local_x, local_y);
        bool hovered =
            !blocked_by_other_overlay &&
            (dropdown_hovered || hitTestPanelShape(local_x, local_y, logical_w, logical_h));

        if (!dropdown_hovered && hovered && clip_y_min_ >= 0 && clip_y_max_ > clip_y_min_) {
            if (mouse_y < clip_y_min_ || mouse_y > clip_y_max_)
                hovered = false;
        }

        // While a button is captured the panel stays active for input
        // forwarding so an in-progress drag survives the cursor leaving.
        const bool any_capture =
            manual_dropdown_mouse_captured_ ||
            mouse_captured_[0] || mouse_captured_[1] || mouse_captured_[2];
        const bool effective_hovered = hovered || any_capture;

        const bool hover_changed = (effective_hovered != last_hovered_);
        if (hover_changed) {
            last_hovered_ = effective_hovered;
            had_input = true;
            if (!effective_hovered) {
                last_forwarded_mx_ = -1;
                last_forwarded_my_ = -1;
                setManualDropdownHover(nullptr);
                manual_dropdown_mouse_captured_ = false;
                rml_context_->ProcessMouseLeave();
            }
        }

        const int rml_mx = static_cast<int>(local_x);
        const int rml_my = static_cast<int>(local_y);
        const bool mouse_moved = effective_hovered &&
                                 (rml_mx != last_forwarded_mx_ || rml_my != last_forwarded_my_);

        const bool pointer_event =
            input.mouse_clicked[0] || input.mouse_released[0] ||
            input.mouse_clicked[1] || input.mouse_released[1] ||
            input.mouse_wheel != 0.0f;
        const bool pointer_active =
            pointer_event ||
            input.mouse_down[0] || input.mouse_down[1] || input.mouse_down[2];
        if (effective_hovered && pointer_event)
            had_input = true;

        const int mods = sdlModsToRml(input.key_ctrl, input.key_shift,
                                      input.key_alt, input.key_super);

        const bool manual_dropdown_route = dropdown_hovered || manual_dropdown_mouse_captured_;
        Rml::Element* manual_dropdown_target =
            manual_dropdown_route ? openDropdownOptionAtPoint(local_x, local_y) : nullptr;
        Rml::Element* manual_dropdown_option =
            (manual_dropdown_target && manual_dropdown_target->GetTagName() == "option")
                ? manual_dropdown_target
                : nullptr;
        Rml::Element* manual_dropdown_box = nullptr;
        if (manual_dropdown_target) {
            if (manual_dropdown_target->GetTagName() == "selectbox")
                manual_dropdown_box = manual_dropdown_target;
            else if (auto* const parent = manual_dropdown_target->GetParentNode();
                     parent && parent->GetTagName() == "selectbox") {
                manual_dropdown_box = parent;
            }
        }
        const bool manual_dropdown_option_route =
            manual_dropdown_option != nullptr || manual_dropdown_mouse_captured_;

        if (mouse_moved) {
            last_forwarded_mx_ = rml_mx;
            last_forwarded_my_ = rml_my;
            if (manual_dropdown_option_route) {
                if (rml_context_->GetHoverElement())
                    rml_context_->ProcessMouseLeave();
                setManualDropdownHover(manual_dropdown_option);
                had_input = true;
            } else {
                auto* const prev_hover = rml_context_->GetHoverElement();
                rml_context_->ProcessMouseMove(rml_mx, rml_my, mods);
                auto* const next_hover = rml_context_->GetHoverElement();
                if (pointer_active || next_hover != prev_hover)
                    had_input = true;
            }
        } else if (manual_dropdown_option_route) {
            if (rml_context_->GetHoverElement())
                rml_context_->ProcessMouseLeave();
            setManualDropdownHover(manual_dropdown_option);
            if (pointer_active)
                had_input = true;
        }

        if (manual_dropdown_option_route) {
            setManualDropdownHover(manual_dropdown_option);
            had_input = true;
        } else {
            setManualDropdownHover(nullptr);
        }

        const auto deliver_button_down = [&](const int button) {
            rml_context_->ProcessMouseButtonDown(button, mods);
            mouse_captured_[button] = true;
        };
        const auto deliver_button_up = [&](const int button) {
            rml_context_->ProcessMouseButtonUp(button, mods);
            mouse_captured_[button] = false;
            had_input = true;
        };

        if (manual_dropdown_option_route) {
            if (input.mouse_clicked[0]) {
                manual_dropdown_mouse_captured_ = true;
                had_input = true;
            }
            if (input.mouse_clicked[1])
                had_input = true;
            if (input.mouse_wheel != 0.0f) {
                if (manual_dropdown_box) {
                    const float max_scroll = std::max(
                        0.0f,
                        manual_dropdown_box->GetScrollHeight() - manual_dropdown_box->GetClientHeight());
                    manual_dropdown_box->SetScrollTop(std::clamp(
                        manual_dropdown_box->GetScrollTop() - input.mouse_wheel * 30.0f,
                        0.0f,
                        max_scroll));
                }
                had_input = true;
            }
        } else if (hovered) {
            if (input.mouse_clicked[0])
                deliver_button_down(0);
            if (input.mouse_clicked[1])
                deliver_button_down(1);
            if (input.mouse_wheel != 0.0f) {
                rml_context_->ProcessMouseWheel(Rml::Vector2f(0, -input.mouse_wheel), mods);
                // Re-resolve hover against the new scroll offset so row text
                // doesn't render against a stale layout for one frame.
                rml_context_->ProcessMouseMove(rml_mx, rml_my, mods);
                had_input = true;
            }
            if (input.mouse_clicked[0])
                sync_text_focus();
        } else if (input.mouse_clicked[0]) {
            had_input |= blur_focused_element();
        }

        if (manual_dropdown_mouse_captured_ &&
            (input.mouse_released[0] || !input.mouse_down[0])) {
            auto* click_option = manual_dropdown_option;
            if (!click_option) {
                auto* const target = openDropdownOptionAtPoint(local_x, local_y);
                if (target && target->GetTagName() == "option")
                    click_option = target;
            }
            if (click_option)
                click_option->Click();
            manual_dropdown_mouse_captured_ = false;
            setManualDropdownHover(nullptr);
            sync_text_focus();
            had_input = true;
        }

        // Forward release regardless of hover so a drag begun on the scrollbar
        // ends when the user lets go anywhere on screen.
        for (int button = 0; button < 2; ++button) {
            if (mouse_captured_[button] &&
                (input.mouse_released[button] || !input.mouse_down[button])) {
                deliver_button_up(button);
            }
        }

        if (hovered) {
            if (auto* const hover = rml_context_->GetHoverElement())
                tooltip_.setHover(resolveRmlTooltip(hover), hover);
            else
                tooltip_.setHover({}, nullptr);
        } else {
            tooltip_.setHover({}, nullptr);
        }

        if (input.viewport_keyboard_focus)
            had_input |= blur_focused_element();

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
