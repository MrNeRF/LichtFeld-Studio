/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// clang-format off
#include <glad/glad.h>
// clang-format on

#include "gui/rmlui/rml_panel_host.hpp"
#include "core/logger.hpp"
#include "gui/panel_layout.hpp"
#include "gui/rmlui/rml_theme.hpp"
#include "gui/rmlui/rml_tooltip.hpp"
#include "gui/rmlui/rmlui_manager.hpp"
#include "gui/rmlui/rmlui_render_interface.hpp"
#include "internal/resource_paths.hpp"
#include "theme/theme.hpp"

#include <RmlUi/Core.h>
#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/Input.h>
#include <SDL3/SDL_keyboard.h>
#include <SDL3/SDL_scancode.h>
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <format>

namespace lfs::vis::gui {

    constexpr int kMaxFboSize = 8192;

    static std::mutex s_text_mutex;
    static std::vector<uint32_t> s_text_queue;

    static std::string s_frame_tooltip;
    static bool s_frame_wants_keyboard = false;

    void RmlPanelHost::pushTextInput(const std::string& text) {
        std::lock_guard lock(s_text_mutex);
        for (size_t i = 0; i < text.size();) {
            uint32_t cp = 0;
            auto c = static_cast<unsigned char>(text[i]);
            if (c < 0x80) {
                cp = c;
                i += 1;
            } else if ((c >> 5) == 0x06) {
                cp = (c & 0x1F) << 6;
                if (i + 1 < text.size())
                    cp |= static_cast<unsigned char>(text[i + 1]) & 0x3F;
                i += 2;
            } else if ((c >> 4) == 0x0E) {
                cp = (c & 0x0F) << 12;
                if (i + 1 < text.size())
                    cp |= (static_cast<unsigned char>(text[i + 1]) & 0x3F) << 6;
                if (i + 2 < text.size())
                    cp |= static_cast<unsigned char>(text[i + 2]) & 0x3F;
                i += 3;
            } else if ((c >> 3) == 0x1E) {
                cp = (c & 0x07) << 18;
                if (i + 1 < text.size())
                    cp |= (static_cast<unsigned char>(text[i + 1]) & 0x3F) << 12;
                if (i + 2 < text.size())
                    cp |= (static_cast<unsigned char>(text[i + 2]) & 0x3F) << 6;
                if (i + 3 < text.size())
                    cp |= static_cast<unsigned char>(text[i + 3]) & 0x3F;
                i += 4;
            } else {
                i += 1;
                continue;
            }
            s_text_queue.push_back(cp);
        }
    }

    std::vector<uint32_t> RmlPanelHost::drainTextInput() {
        std::lock_guard lock(s_text_mutex);
        std::vector<uint32_t> result;
        result.swap(s_text_queue);
        return result;
    }

    std::string RmlPanelHost::consumeFrameTooltip() {
        std::string result;
        result.swap(s_frame_tooltip);
        return result;
    }

    void RmlPanelHost::setFrameTooltip(const std::string& tip) {
        s_frame_tooltip = tip;
    }

    bool RmlPanelHost::consumeFrameWantsKeyboard() {
        bool result = s_frame_wants_keyboard;
        s_frame_wants_keyboard = false;
        return result;
    }

    using rml_theme::colorToRml;
    using rml_theme::colorToRmlAlpha;

    namespace {
        ImVec4 brighten(const ImVec4& color, float factor) {
            return {
                color.x + (1.0f - color.x) * factor,
                color.y + (1.0f - color.y) * factor,
                color.z + (1.0f - color.z) * factor,
                color.w};
        }

        bool isTextEditableElement(Rml::Element* element) {
            if (!element)
                return false;

            const auto tag = element->GetTagName();
            if (tag == "textarea")
                return true;
            if (tag != "input")
                return false;

            const auto input_type = element->GetAttribute<Rml::String>("type", "text");
            return input_type.empty() || input_type == "text" || input_type == "password" ||
                   input_type == "search" || input_type == "email" || input_type == "url";
        }

        bool isSingleLineTextInput(Rml::Element* element) {
            if (!element || element->GetTagName() != "input")
                return false;

            const auto input_type = element->GetAttribute<Rml::String>("type", "text");
            return input_type.empty() || input_type == "text" || input_type == "password" ||
                   input_type == "search" || input_type == "email" || input_type == "url";
        }

        Rml::Input::KeyIdentifier sdlScancodeToRml(int scancode) {
            // clang-format off
            switch (scancode) {
            case SDL_SCANCODE_SPACE:     return Rml::Input::KI_SPACE;
            case SDL_SCANCODE_BACKSPACE: return Rml::Input::KI_BACK;
            case SDL_SCANCODE_TAB:       return Rml::Input::KI_TAB;
            case SDL_SCANCODE_RETURN:    return Rml::Input::KI_RETURN;
            case SDL_SCANCODE_ESCAPE:    return Rml::Input::KI_ESCAPE;
            case SDL_SCANCODE_DELETE:    return Rml::Input::KI_DELETE;
            case SDL_SCANCODE_INSERT:    return Rml::Input::KI_INSERT;
            case SDL_SCANCODE_HOME:      return Rml::Input::KI_HOME;
            case SDL_SCANCODE_END:       return Rml::Input::KI_END;
            case SDL_SCANCODE_PAGEUP:    return Rml::Input::KI_PRIOR;
            case SDL_SCANCODE_PAGEDOWN:  return Rml::Input::KI_NEXT;
            case SDL_SCANCODE_LEFT:      return Rml::Input::KI_LEFT;
            case SDL_SCANCODE_UP:        return Rml::Input::KI_UP;
            case SDL_SCANCODE_RIGHT:     return Rml::Input::KI_RIGHT;
            case SDL_SCANCODE_DOWN:      return Rml::Input::KI_DOWN;
            case SDL_SCANCODE_F1:  return Rml::Input::KI_F1;
            case SDL_SCANCODE_F2:  return Rml::Input::KI_F2;
            case SDL_SCANCODE_F3:  return Rml::Input::KI_F3;
            case SDL_SCANCODE_F4:  return Rml::Input::KI_F4;
            case SDL_SCANCODE_F5:  return Rml::Input::KI_F5;
            case SDL_SCANCODE_F6:  return Rml::Input::KI_F6;
            case SDL_SCANCODE_F7:  return Rml::Input::KI_F7;
            case SDL_SCANCODE_F8:  return Rml::Input::KI_F8;
            case SDL_SCANCODE_F9:  return Rml::Input::KI_F9;
            case SDL_SCANCODE_F10: return Rml::Input::KI_F10;
            case SDL_SCANCODE_F11: return Rml::Input::KI_F11;
            case SDL_SCANCODE_F12: return Rml::Input::KI_F12;
            case SDL_SCANCODE_EQUALS:   return Rml::Input::KI_OEM_PLUS;
            case SDL_SCANCODE_MINUS:    return Rml::Input::KI_OEM_MINUS;
            case SDL_SCANCODE_KP_PLUS:  return Rml::Input::KI_ADD;
            case SDL_SCANCODE_KP_MINUS: return Rml::Input::KI_SUBTRACT;
            case SDL_SCANCODE_KP_ENTER: return Rml::Input::KI_NUMPADENTER;
            default: break;
            }
            // clang-format on

            if (scancode >= SDL_SCANCODE_A && scancode <= SDL_SCANCODE_Z)
                return static_cast<Rml::Input::KeyIdentifier>(
                    Rml::Input::KI_A + (scancode - SDL_SCANCODE_A));

            if (scancode == SDL_SCANCODE_0)
                return Rml::Input::KI_0;
            if (scancode >= SDL_SCANCODE_1 && scancode <= SDL_SCANCODE_9)
                return static_cast<Rml::Input::KeyIdentifier>(
                    Rml::Input::KI_1 + (scancode - SDL_SCANCODE_1));

            return Rml::Input::KI_UNKNOWN;
        }

        int buildRmlModifiers(const PanelInputState& input) {
            int mods = 0;
            if (input.key_ctrl)
                mods |= Rml::Input::KM_CTRL;
            if (input.key_shift)
                mods |= Rml::Input::KM_SHIFT;
            if (input.key_alt)
                mods |= Rml::Input::KM_ALT;
            if (input.key_super)
                mods |= Rml::Input::KM_META;
            return mods;
        }
    } // namespace

    RmlPanelHost::RmlPanelHost(RmlUIManager* manager, std::string context_name,
                               std::string rml_path)
        : manager_(manager),
          context_name_(std::move(context_name)),
          rml_path_(std::move(rml_path)) {
        assert(manager_);
    }

    RmlPanelHost::~RmlPanelHost() {
        if (rml_context_ && manager_) {
            manager_->destroyContext(context_name_);
            rml_context_ = nullptr;
            document_ = nullptr;
        }
    }

    std::string RmlPanelHost::generateThemeRCSS(const lfs::vis::Theme& t) const {
        const auto& p = t.palette;
        const auto text = colorToRml(p.text);
        const auto text_dim = colorToRml(p.text_dim);
        const auto surface = colorToRml(p.surface);
        const auto primary = colorToRml(p.primary);
        const auto primary_dim = colorToRml(p.primary_dim);
        const auto border = colorToRml(p.border);
        const auto row_even = colorToRml(p.row_even);
        const auto row_odd = colorToRml(p.row_odd);
        const auto row_hover = colorToRmlAlpha(p.primary, 0.12f);
        const auto row_hover_border = colorToRml(p.primary);
        const auto row_hover_border_selected = colorToRml(p.primary_dim);
        const auto row_selected = colorToRml(brighten(p.primary, 0.14f));
        const auto row_selected_hover = colorToRml(brighten(p.primary, 0.24f));

        return std::format(
            "body {{ color: {0}; background-color: {2}; }}\n"
            "#search-container {{ background-color: {2}; border-color: {4}; }}\n"
            "#filter-input {{ color: {0}; }}\n"
            ".tree-row.even {{ background-color: {5}; }}\n"
            ".tree-row.odd {{ background-color: {6}; }}\n"
            ".tree-row:hover {{ background-color: {7}; border-left-color: {8}; }}\n"
            ".tree-row.selected {{ background-color: {9}; }}\n"
            ".tree-row.selected:hover {{ background-color: {10}; border-left-color: {11}; }}\n"
            ".tree-row.drop-target {{ border-width: 1dp; border-color: {3}; }}\n"
            ".expand-toggle {{ color: {1}; }}\n"
            ".expand-toggle:hover {{ color: {0}; }}\n"
            ".node-name {{ color: {0}; }}\n"
            ".node-name.training-disabled {{ color: {1}; }}\n"
            ".rename-input {{ color: {0}; background-color: {2}; border-width: 1dp; border-color: {3}; }}\n"
            ".row-icon {{ image-color: {0}; }}\n",
            text, text_dim, surface, primary, border, row_even, row_odd,
            row_hover, row_hover_border, row_selected, row_selected_hover, row_hover_border_selected);
    }

    bool RmlPanelHost::syncThemeProperties() {
        if (!document_)
            return false;

        const std::size_t theme_signature = rml_theme::currentThemeSignature();
        if (has_theme_signature_ && last_theme_signature_ == theme_signature)
            return false;

        last_theme_signature_ = theme_signature;
        has_theme_signature_ = true;

        if (base_rcss_.empty()) {
            auto rcss_name = std::filesystem::path(rml_path_).replace_extension(".rcss").string();
            base_rcss_ = rml_theme::loadBaseRCSS(rcss_name);
        }

        rml_theme::applyTheme(document_, base_rcss_, rml_theme::generateAllThemeMedia([this](const auto& th) { return generateThemeRCSS(th); }));
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

    bool RmlPanelHost::loadDocument() {
        if (document_)
            return true;
        try {
            const auto full_path = lfs::vis::getAssetPath(rml_path_);
            document_ = rml_context_->LoadDocument(full_path.string());
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
        auto* frame = document_->GetElementById("window-frame");
        content_wrap_el_ = frame ? frame : document_->GetElementById("content-wrap");
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

        return content_wrap_el_ ? content_wrap_el_->GetOffsetHeight() : 100.0f;
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

        rml_context_->SetDimensions(Rml::Vector2i(pw, ph));
        rml_context_->Update();

        restoreScrollTop(saved_scroll);

        last_layout_w_ = pw;
        last_layout_h_ = ph;

        if (height_mode_ == HeightMode::Content) {
            last_content_height_ = computeContentHeight();
            if (content_el_)
                last_content_el_height_ = content_el_->GetOffsetHeight();
        }
    }

    void RmlPanelHost::renderIfDirty(int pw, int ph, float& display_h) {
        if (manager_ && manager_->shouldDeferFboUpdate(fbo_))
            return;

        const bool theme_dirty = syncThemeProperties();
        const bool size_dirty = (pw != last_fbo_w_ || ph != last_fbo_h_);
        const bool externally_clipped =
            (clip_y_min_ >= 0.0f && clip_y_max_ > clip_y_min_);

        fbo_.ensure(pw, std::min(ph, kMaxFboSize));
        if (!fbo_.valid())
            return;

        const bool dirty = render_needed_ || content_dirty_ || theme_dirty ||
                           size_dirty || animation_active_;
        if (!dirty)
            return;

        const bool need_content_measure =
            height_mode_ == HeightMode::Content &&
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
                rml_context_->SetDimensions(Rml::Vector2i(pw, layout_h));
                rml_context_->Update();
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

            fbo_.ensure(pw, ph);
            if (!fbo_.valid())
                return;

            rml_context_->SetDimensions(Rml::Vector2i(pw, ph));
            rml_context_->Update();
            last_layout_w_ = pw;
            last_layout_h_ = ph;

            restoreScrollTop(saved_scroll);
        } else {
            rml_context_->SetDimensions(Rml::Vector2i(pw, ph));
            rml_context_->Update();
            last_layout_w_ = pw;
            last_layout_h_ = ph;
            restoreScrollTop(saved_scroll);
        }
        content_dirty_ = false;
        if (height_mode_ != HeightMode::Content)
            last_content_height_ = display_h;

        auto* render = manager_->getRenderInterface();
        assert(render);
        render->SetViewport(pw, ph);

        GLint prev_fbo = 0;
        fbo_.bind(&prev_fbo);
        render->SetTargetFramebuffer(fbo_.fbo());

        render->BeginFrame();
        rml_context_->Render();
        render->EndFrame();

        render->SetTargetFramebuffer(0);
        fbo_.unbind(prev_fbo);

        animation_active_ = (rml_context_->GetNextUpdateDelay() == 0);
        last_fbo_w_ = pw;
        last_fbo_h_ = ph;
        render_needed_ = false;

        if (height_mode_ == HeightMode::Content) {
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
        if (height_mode_ == HeightMode::Content) {
            h = std::max(1, static_cast<int>(std::ceil(last_content_height_)));
            display_h = last_content_height_;
        } else {
            h = static_cast<int>(avail_h);
            display_h = avail_h;
        }

        if (forwardInput(pos_x, pos_y))
            render_needed_ = true;

        renderIfDirty(w, h, display_h);

        fbo_.blitAsImage(avail_w, display_h);
    }

    void RmlPanelHost::resolveDirectRenderHeight(float requested_h, int& ph, float& display_h) const {
        if (height_mode_ == HeightMode::Content) {
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
            ph = std::min(kMaxFboSize, static_cast<int>(requested_h));
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

        assert(input_ && input_->bg_draw_list);
        auto* dl = static_cast<ImDrawList*>(input_->bg_draw_list);
        dl->PushClipRect(ImVec2(x, y), ImVec2(x + w, y + h), true);
        fbo_.blitToDrawListOpaque(input_->bg_draw_list, x, y, w, display_h);
        dl->PopClipRect();
    }

    bool RmlPanelHost::forwardInput(float panel_x, float panel_y) {
        assert(rml_context_);

        if (!input_ || !fbo_.valid())
            return false;

        bool had_input = false;
        const auto& input = *input_;
        const float mouse_x = input.mouse_x;
        const float mouse_y = input.mouse_y;
        const auto sync_text_focus = [&]() {
            const bool want_text = isTextEditableElement(rml_context_->GetFocusElement());
            if (want_text == has_text_focus_)
                return;

            has_text_focus_ = want_text;
            auto* const win = manager_->getWindow();
            if (has_text_focus_)
                SDL_StartTextInput(win);
            else
                SDL_StopTextInput(win);
        };
        const auto flush_pending_text_input = [&]() {
            if (!has_text_focus_)
                return;

            auto chars = drainTextInput();
            if (!chars.empty())
                had_input = true;
            for (uint32_t cp : chars)
                rml_context_->ProcessTextInput(static_cast<Rml::Character>(cp));
        };
        const auto blur_focused_text = [&]() {
            if (auto* const focused = rml_context_->GetFocusElement();
                isTextEditableElement(focused)) {
                flush_pending_text_input();
                focused->Blur();
            }
            sync_text_focus();
        };

        float local_x = mouse_x - panel_x;
        float local_y = mouse_y - panel_y;

        const float logical_w = static_cast<float>(fbo_.width());
        const float logical_h = static_cast<float>(fbo_.height());

        bool hovered = local_x >= 0 && local_y >= 0 && local_x < logical_w && local_y < logical_h;

        if (hovered && clip_y_min_ >= 0 && clip_y_max_ > clip_y_min_) {
            if (mouse_y < clip_y_min_ || mouse_y > clip_y_max_)
                hovered = false;
        }

        if (hovered != last_hovered_) {
            last_hovered_ = hovered;
            had_input = true;
            if (!hovered)
                rml_context_->ProcessMouseLeave();
        }

        const int rml_mx = static_cast<int>(local_x);
        const int rml_my = static_cast<int>(local_y);
        if (hovered &&
            (rml_mx != last_forwarded_mx_ || rml_my != last_forwarded_my_)) {
            last_forwarded_mx_ = rml_mx;
            last_forwarded_my_ = rml_my;
            had_input = true;
        }

        if (input.mouse_clicked[0] || input.mouse_released[0] ||
            input.mouse_clicked[1] || input.mouse_released[1] ||
            input.mouse_wheel != 0.0f)
            had_input = true;

        if (hovered) {
            rml_context_->ProcessMouseMove(rml_mx, rml_my, 0);

            if (input.mouse_clicked[0])
                rml_context_->ProcessMouseButtonDown(0, 0);
            if (input.mouse_released[0])
                rml_context_->ProcessMouseButtonUp(0, 0);

            if (input.mouse_clicked[1])
                rml_context_->ProcessMouseButtonDown(1, 0);
            if (input.mouse_released[1])
                rml_context_->ProcessMouseButtonUp(1, 0);

            if (input.mouse_wheel != 0.0f)
                rml_context_->ProcessMouseWheel(Rml::Vector2f(0, -input.mouse_wheel), 0);

            if (input.mouse_clicked[0])
                sync_text_focus();
        } else if (input.mouse_clicked[0]) {
            if (has_text_focus_)
                blur_focused_text();
        }

        if (hovered) {
            auto* hover = rml_context_->GetHoverElement();
            if (hover) {
                s_frame_tooltip = resolveRmlTooltip(hover);
            }
        }

        bool forward_keys = has_text_focus_ || hovered;
        bool commit_requested = false;
        if (forward_keys) {
            int mods = buildRmlModifiers(input);
            for (int sc : input.keys_pressed) {
                auto rml_key = sdlScancodeToRml(sc);
                if (rml_key != Rml::Input::KI_UNKNOWN) {
                    rml_context_->ProcessKeyDown(rml_key, mods);
                    had_input = true;
                }
                if (sc == SDL_SCANCODE_RETURN || sc == SDL_SCANCODE_KP_ENTER)
                    commit_requested = true;
            }
            for (int sc : input.keys_released) {
                auto rml_key = sdlScancodeToRml(sc);
                if (rml_key != Rml::Input::KI_UNKNOWN) {
                    rml_context_->ProcessKeyUp(rml_key, mods);
                    had_input = true;
                }
            }
        }

        if (commit_requested && isSingleLineTextInput(rml_context_->GetFocusElement()))
            blur_focused_text();

        sync_text_focus();

        wants_keyboard_ = has_text_focus_ || (foreground_ && hovered);
        if (wants_keyboard_)
            s_frame_wants_keyboard = true;

        if (has_text_focus_) {
            flush_pending_text_input();
        }

        return had_input;
    }

} // namespace lfs::vis::gui
