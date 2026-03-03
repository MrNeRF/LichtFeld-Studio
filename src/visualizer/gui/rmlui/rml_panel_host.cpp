/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// clang-format off
#include <glad/glad.h>
// clang-format on

#include "gui/rmlui/rml_panel_host.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/logger.hpp"
#include "gui/panel_layout.hpp"
#include "gui/rmlui/rml_theme.hpp"
#include "gui/rmlui/rmlui_manager.hpp"
#include "gui/rmlui/rmlui_render_interface.hpp"
#include "internal/resource_paths.hpp"
#include "theme/theme.hpp"

#include <RmlUi/Core.h>
#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/Input.h>
#include <SDL3/SDL_keyboard.h>
#include <SDL3/SDL_scancode.h>
#include <cassert>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <format>

namespace lfs::vis::gui {

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

    bool RmlPanelHost::consumeFrameWantsKeyboard() {
        bool result = s_frame_wants_keyboard;
        s_frame_wants_keyboard = false;
        return result;
    }

    using rml_theme::colorToRml;

    namespace {
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

    RmlPanelHost::~RmlPanelHost() = default;

    std::string RmlPanelHost::generateThemeRCSS() const {
        const auto& p = lfs::vis::theme().palette;
        const auto text = colorToRml(p.text);
        const auto text_dim = colorToRml(p.text_dim);
        const auto surface = colorToRml(p.surface);
        const auto surface_bright = colorToRml(p.surface_bright);
        const auto primary = colorToRml(p.primary);
        const auto border = colorToRml(p.border);
        const auto row_even = colorToRml(p.row_even);
        const auto row_odd = colorToRml(p.row_odd);

        return std::format(
            "body {{ color: {0}; background-color: {2}; }}\n"
            "#search-container {{ background-color: {2}; border-color: {5}; }}\n"
            "#filter-input {{ color: {0}; }}\n"
            ".tree-row.even {{ background-color: {6}; }}\n"
            ".tree-row.odd {{ background-color: {7}; }}\n"
            ".tree-row:hover {{ background-color: {3}; }}\n"
            ".tree-row.selected {{ background-color: {4}; }}\n"
            ".tree-row.selected:hover {{ background-color: {4}; }}\n"
            ".tree-row.drop-target {{ border-width: 1dp; border-color: {4}; }}\n"
            ".expand-toggle {{ color: {1}; }}\n"
            ".expand-toggle:hover {{ color: {0}; }}\n"
            ".node-name {{ color: {0}; }}\n"
            ".node-name.training-disabled {{ color: {1}; }}\n"
            ".node-count {{ color: {1}; }}\n"
            ".rename-input {{ color: {0}; background-color: {2}; border-width: 1dp; border-color: {4}; }}\n"
            ".row-icon {{ image-color: {0}; }}\n",
            text, text_dim, surface, surface_bright, primary, border, row_even, row_odd);
    }

    bool RmlPanelHost::syncThemeProperties() {
        if (!document_)
            return false;

        const auto& p = lfs::vis::theme().palette;
        if (std::memcmp(last_synced_text_, &p.text, sizeof(last_synced_text_)) == 0)
            return false;
        std::memcpy(last_synced_text_, &p.text, sizeof(last_synced_text_));

        if (base_rcss_.empty()) {
            auto rcss_name = std::filesystem::path(rml_path_).replace_extension(".rcss").string();
            base_rcss_ = rml_theme::loadBaseRCSS(rcss_name);
        }

        rml_theme::applyTheme(document_, base_rcss_, generateThemeRCSS());
        content_dirty_ = true;
        return true;
    }

    bool RmlPanelHost::ensureContext() {
        if (rml_context_)
            return true;
        rml_context_ = manager_->createContext(context_name_, 100, 100);
        return rml_context_ != nullptr;
    }

    bool RmlPanelHost::loadDocument() {
        if (document_)
            return true;
        try {
            const auto full_path = lfs::vis::getAssetPath(rml_path_);
            document_ = rml_context_->LoadDocument(full_path.string());
            if (document_) {
                document_->Show();
                cacheContentElements();
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
    }

    void RmlPanelHost::renderIfDirty(int pw, int ph, float& display_h) {
        const bool theme_dirty = syncThemeProperties();
        const bool size_dirty = (pw != last_fbo_w_ || ph != last_fbo_h_);

        fbo_.ensure(pw, ph);
        if (!fbo_.valid())
            return;

        const bool dirty = render_needed_ || content_dirty_ || theme_dirty ||
                           size_dirty || animation_active_;
        if (!dirty)
            return;

        const float dp_ratio = manager_->getDpRatio();

        if (height_mode_ == HeightMode::Content &&
            (content_dirty_ || pw != last_measure_w_)) {
            last_measure_w_ = pw;
            const int layout_h = static_cast<int>(10000.0f * dp_ratio);
            rml_context_->SetDimensions(Rml::Vector2i(pw, layout_h));
            rml_context_->Update();

            const float content_h =
                content_wrap_el_ ? content_wrap_el_->GetOffsetHeight() : 100.0f;
            ph = std::max(1, static_cast<int>(std::ceil(content_h)));
            display_h = static_cast<float>(ph) / dp_ratio;
            last_content_height_ = display_h;

            fbo_.ensure(pw, ph);
            if (!fbo_.valid())
                return;
        }
        content_dirty_ = false;
        last_content_height_ = display_h;

        rml_context_->SetDimensions(Rml::Vector2i(pw, ph));
        rml_context_->Update();

        auto* render = manager_->getRenderInterface();
        assert(render);
        render->SetViewport(pw, ph);

        GLint prev_fbo = 0;
        fbo_.bind(&prev_fbo);

        render->BeginFrame();
        rml_context_->Render();
        render->EndFrame();

        fbo_.unbind(prev_fbo);

        animation_active_ = (rml_context_->GetNextUpdateDelay() == 0);
        last_fbo_w_ = pw;
        last_fbo_h_ = ph;
        render_needed_ = false;

        if (height_mode_ == HeightMode::Content && content_wrap_el_) {
            const float actual_h = content_wrap_el_->GetOffsetHeight();
            if (std::abs(actual_h - static_cast<float>(ph)) > 2.0f)
                content_dirty_ = true;
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

        if (!ensureContext() || !loadDocument())
            return;

        const float dp_ratio = manager_->getDpRatio();
        const int w = static_cast<int>(avail_w * dp_ratio);

        int h;
        float display_h;
        if (height_mode_ == HeightMode::Content) {
            h = std::max(1, static_cast<int>(std::ceil(last_content_height_ * dp_ratio)));
            display_h = last_content_height_;
        } else {
            h = static_cast<int>(avail_h * dp_ratio);
            display_h = avail_h;
        }

        if (forwardInput(pos_x, pos_y))
            render_needed_ = true;

        renderIfDirty(w, h, display_h);

        fbo_.blitAsImage(avail_w, display_h);
    }

    void RmlPanelHost::drawDirect(float x, float y, float w, float h) {
        if (w <= 0 || h <= 0)
            return;

        if (!ensureContext() || !loadDocument())
            return;

        const float dp_ratio = manager_->getDpRatio();
        const int pw = static_cast<int>(w * dp_ratio);

        int ph;
        float display_h;
        if (height_mode_ == HeightMode::Content) {
            ph = std::max(1, static_cast<int>(std::ceil(last_content_height_ * dp_ratio)));
            display_h = last_content_height_;
        } else {
            ph = static_cast<int>(h * dp_ratio);
            display_h = h;
        }

        if (forwardInput(x, y))
            render_needed_ = true;

        renderIfDirty(pw, ph, display_h);

        assert(input_ && input_->bg_draw_list);
        fbo_.blitToDrawListOpaque(input_->bg_draw_list, x, y, w, display_h);
    }

    bool RmlPanelHost::forwardInput(float panel_x, float panel_y) {
        assert(rml_context_);

        if (!input_ || !fbo_.valid())
            return false;

        bool had_input = false;
        const auto& input = *input_;
        const float mouse_x = input.mouse_x;
        const float mouse_y = input.mouse_y;

        float local_x = mouse_x - panel_x;
        float local_y = mouse_y - panel_y;

        const float dp_ratio = manager_->getDpRatio();
        const float logical_w = static_cast<float>(fbo_.width()) / dp_ratio;
        const float logical_h = static_cast<float>(fbo_.height()) / dp_ratio;

        bool hovered = local_x >= 0 && local_y >= 0 && local_x < logical_w && local_y < logical_h;

        if (hovered && clip_y_min_ >= 0 && clip_y_max_ > clip_y_min_) {
            if (mouse_y < clip_y_min_ || mouse_y > clip_y_max_)
                hovered = false;
        }

        if (hovered != last_hovered_) {
            last_hovered_ = hovered;
            had_input = true;
        }

        const int rml_mx = static_cast<int>(local_x * dp_ratio);
        const int rml_my = static_cast<int>(local_y * dp_ratio);
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

            if (input.mouse_clicked[0]) {
                auto* focused = rml_context_->GetFocusElement();
                bool want_text = focused && focused->GetTagName() == "input";
                if (want_text != has_text_focus_) {
                    has_text_focus_ = want_text;
                    auto* win = manager_->getWindow();
                    if (has_text_focus_)
                        SDL_StartTextInput(win);
                    else
                        SDL_StopTextInput(win);
                }
            }
        } else if (input.mouse_clicked[0]) {
            if (has_text_focus_) {
                drainTextInput();
                has_text_focus_ = false;
                SDL_StopTextInput(manager_->getWindow());
            }
        }

        if (hovered) {
            auto* hover = rml_context_->GetHoverElement();
            if (hover) {
                Rml::String tip;
                for (auto* el = hover; el; el = el->GetParentNode()) {
                    auto key = el->GetAttribute<Rml::String>("data-tooltip", "");
                    if (!key.empty()) {
                        auto& loc = lfs::event::LocalizationManager::getInstance();
                        tip = loc.get(key);
                        if (tip == key)
                            tip.clear();
                        break;
                    }
                    tip = el->GetAttribute<Rml::String>("title", "");
                    if (!tip.empty())
                        break;
                }
                if (!tip.empty())
                    s_frame_tooltip = std::string(tip.c_str(), tip.size());
            }
        }

        wants_keyboard_ = has_text_focus_;
        if (has_text_focus_)
            s_frame_wants_keyboard = true;

        bool forward_keys = has_text_focus_ || hovered;
        if (forward_keys) {
            int mods = buildRmlModifiers(input);
            for (int sc : input.keys_pressed) {
                auto rml_key = sdlScancodeToRml(sc);
                if (rml_key != Rml::Input::KI_UNKNOWN) {
                    rml_context_->ProcessKeyDown(rml_key, mods);
                    had_input = true;
                }
            }
            for (int sc : input.keys_released) {
                auto rml_key = sdlScancodeToRml(sc);
                if (rml_key != Rml::Input::KI_UNKNOWN) {
                    rml_context_->ProcessKeyUp(rml_key, mods);
                    had_input = true;
                }
            }
        }

        if (has_text_focus_) {
            auto chars = drainTextInput();
            if (!chars.empty())
                had_input = true;
            for (uint32_t cp : chars)
                rml_context_->ProcessTextInput(static_cast<Rml::Character>(cp));
        }

        return had_input;
    }

} // namespace lfs::vis::gui
