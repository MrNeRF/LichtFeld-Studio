/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// clang-format off
#include <glad/glad.h>
// clang-format on

#include "gui/rmlui/rml_panel_host.hpp"
#include "core/logger.hpp"
#include "gui/rmlui/rml_theme.hpp"
#include "gui/rmlui/rmlui_manager.hpp"
#include "gui/rmlui/rmlui_render_interface.hpp"
#include "internal/resource_paths.hpp"
#include "theme/theme.hpp"

#include <RmlUi/Core.h>
#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/Input.h>
#include <SDL3/SDL_keyboard.h>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <format>
#include <imgui.h>

namespace lfs::vis::gui {

    static std::mutex s_text_mutex;
    static std::vector<uint32_t> s_text_queue;

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

    using rml_theme::colorToRml;

    namespace {
        Rml::Input::KeyIdentifier imguiKeyToRml(ImGuiKey key) {
            // clang-format off
            switch (key) {
            case ImGuiKey_Space:      return Rml::Input::KI_SPACE;
            case ImGuiKey_Backspace:  return Rml::Input::KI_BACK;
            case ImGuiKey_Tab:        return Rml::Input::KI_TAB;
            case ImGuiKey_Enter:      return Rml::Input::KI_RETURN;
            case ImGuiKey_Escape:     return Rml::Input::KI_ESCAPE;
            case ImGuiKey_Delete:     return Rml::Input::KI_DELETE;
            case ImGuiKey_Insert:     return Rml::Input::KI_INSERT;
            case ImGuiKey_Home:       return Rml::Input::KI_HOME;
            case ImGuiKey_End:        return Rml::Input::KI_END;
            case ImGuiKey_PageUp:     return Rml::Input::KI_PRIOR;
            case ImGuiKey_PageDown:   return Rml::Input::KI_NEXT;
            case ImGuiKey_LeftArrow:  return Rml::Input::KI_LEFT;
            case ImGuiKey_UpArrow:    return Rml::Input::KI_UP;
            case ImGuiKey_RightArrow: return Rml::Input::KI_RIGHT;
            case ImGuiKey_DownArrow:  return Rml::Input::KI_DOWN;
            case ImGuiKey_F1:  return Rml::Input::KI_F1;
            case ImGuiKey_F2:  return Rml::Input::KI_F2;
            case ImGuiKey_F3:  return Rml::Input::KI_F3;
            case ImGuiKey_F4:  return Rml::Input::KI_F4;
            case ImGuiKey_F5:  return Rml::Input::KI_F5;
            case ImGuiKey_F6:  return Rml::Input::KI_F6;
            case ImGuiKey_F7:  return Rml::Input::KI_F7;
            case ImGuiKey_F8:  return Rml::Input::KI_F8;
            case ImGuiKey_F9:  return Rml::Input::KI_F9;
            case ImGuiKey_F10: return Rml::Input::KI_F10;
            case ImGuiKey_F11: return Rml::Input::KI_F11;
            case ImGuiKey_F12: return Rml::Input::KI_F12;
            default: break;
            }
            // clang-format on

            if (key >= ImGuiKey_A && key <= ImGuiKey_Z)
                return static_cast<Rml::Input::KeyIdentifier>(
                    Rml::Input::KI_A + (key - ImGuiKey_A));

            if (key >= ImGuiKey_0 && key <= ImGuiKey_9)
                return static_cast<Rml::Input::KeyIdentifier>(
                    Rml::Input::KI_0 + (key - ImGuiKey_0));

            return Rml::Input::KI_UNKNOWN;
        }

        int buildRmlModifiers() {
            ImGuiIO& io = ImGui::GetIO();
            int mods = 0;
            if (io.KeyCtrl)
                mods |= Rml::Input::KM_CTRL;
            if (io.KeyShift)
                mods |= Rml::Input::KM_SHIFT;
            if (io.KeyAlt)
                mods |= Rml::Input::KM_ALT;
            if (io.KeySuper)
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

    void RmlPanelHost::syncThemeProperties() {
        if (!document_)
            return;

        const auto& p = lfs::vis::theme().palette;
        if (std::memcmp(&last_synced_text_, &p.text, sizeof(ImVec4)) == 0)
            return;
        last_synced_text_ = p.text;

        if (base_rcss_.empty()) {
            auto rcss_name = std::filesystem::path(rml_path_).replace_extension(".rcss").string();
            base_rcss_ = rml_theme::loadBaseRCSS(rcss_name);
        }

        rml_theme::applyTheme(document_, base_rcss_, generateThemeRCSS());
        content_dirty_ = true;
    }

    bool RmlPanelHost::ensureContext() {
        if (rml_context_)
            return true;
        rml_context_ = manager_->createContext(context_name_, 100, 100);
        return rml_context_ != nullptr;
    }

    void RmlPanelHost::draw(const PanelDrawContext& ctx) {
        (void)ctx;

        const float avail_w = ImGui::GetContentRegionAvail().x;
        const float avail_h = ImGui::GetContentRegionAvail().y;
        if (avail_w <= 0 || avail_h <= 0)
            return;

        const float dp_ratio = manager_->getDpRatio();
        const int w = static_cast<int>(avail_w * dp_ratio);

        if (!ensureContext())
            return;

        if (!document_) {
            try {
                const auto full_path = lfs::vis::getAssetPath(rml_path_);
                document_ = rml_context_->LoadDocument(full_path.string());
                if (document_)
                    document_->Show();
                else
                    LOG_ERROR("RmlUI: failed to load {}", rml_path_);
            } catch (const std::exception& e) {
                LOG_ERROR("RmlUI: resource not found: {}", e.what());
            }
        }
        if (!document_)
            return;

        syncThemeProperties();

        int h;
        float display_h;
        if (height_mode_ == HeightMode::Content) {
            if (w != last_measure_w_ || content_dirty_) {
                last_measure_w_ = w;
                content_dirty_ = false;
                const int layout_h = static_cast<int>(10000.0f * dp_ratio);
                rml_context_->SetDimensions(Rml::Vector2i(w, layout_h));
                rml_context_->Update();

                auto* frame = document_->GetElementById("window-frame");
                auto* wrap = frame ? frame : document_->GetElementById("content-wrap");
                const float content_h = wrap ? wrap->GetOffsetHeight() : 100.0f;
                h = std::max(1, static_cast<int>(std::ceil(content_h)));
                display_h = static_cast<float>(h) / dp_ratio;
                last_content_height_ = display_h;
            } else {
                h = std::max(1, static_cast<int>(std::ceil(last_content_height_ * dp_ratio)));
                display_h = last_content_height_;
            }
        } else {
            h = static_cast<int>(avail_h * dp_ratio);
            display_h = avail_h;
        }

        rml_context_->SetDimensions(Rml::Vector2i(w, h));
        rml_context_->Update();

        fbo_.ensure(w, h);
        if (!fbo_.valid())
            return;

        ImVec2 panel_pos = ImGui::GetCursorScreenPos();
        forwardInput(panel_pos.x, panel_pos.y);

        auto* render = manager_->getRenderInterface();
        assert(render);
        render->SetViewport(w, h);

        GLint prev_fbo = 0;
        fbo_.bind(&prev_fbo);

        render->BeginFrame();
        rml_context_->Render();
        render->EndFrame();

        fbo_.unbind(prev_fbo);

        fbo_.blitAsImage(avail_w, display_h);

        if (height_mode_ == HeightMode::Content && !content_dirty_) {
            auto* frame = document_->GetElementById("window-frame");
            auto* wrap = frame ? frame : document_->GetElementById("content-wrap");
            if (wrap) {
                const float actual_h = wrap->GetOffsetHeight();
                if (std::abs(actual_h - static_cast<float>(h)) > 2.0f)
                    content_dirty_ = true;
            }
        }
    }

    void RmlPanelHost::drawDirect(float x, float y, float w, float h) {
        if (w <= 0 || h <= 0)
            return;

        const float dp_ratio = manager_->getDpRatio();
        const int pw = static_cast<int>(w * dp_ratio);

        if (!ensureContext())
            return;

        if (!document_) {
            try {
                const auto full_path = lfs::vis::getAssetPath(rml_path_);
                document_ = rml_context_->LoadDocument(full_path.string());
                if (document_)
                    document_->Show();
                else
                    LOG_ERROR("RmlUI: failed to load {}", rml_path_);
            } catch (const std::exception& e) {
                LOG_ERROR("RmlUI: resource not found: {}", e.what());
            }
        }
        if (!document_)
            return;

        syncThemeProperties();

        int ph;
        float display_h;
        if (height_mode_ == HeightMode::Content) {
            if (pw != last_measure_w_ || content_dirty_) {
                last_measure_w_ = pw;
                content_dirty_ = false;
                const int layout_h = static_cast<int>(10000.0f * dp_ratio);
                rml_context_->SetDimensions(Rml::Vector2i(pw, layout_h));
                rml_context_->Update();

                auto* frame = document_->GetElementById("window-frame");
                auto* wrap = frame ? frame : document_->GetElementById("content-wrap");
                const float content_h = wrap ? wrap->GetOffsetHeight() : 100.0f;
                ph = std::max(1, static_cast<int>(std::ceil(content_h)));
                display_h = static_cast<float>(ph) / dp_ratio;
            } else {
                ph = std::max(1, static_cast<int>(std::ceil(last_content_height_ * dp_ratio)));
                display_h = last_content_height_;
            }
        } else {
            ph = static_cast<int>(h * dp_ratio);
            display_h = h;
        }

        last_content_height_ = display_h;

        rml_context_->SetDimensions(Rml::Vector2i(pw, ph));
        rml_context_->Update();

        fbo_.ensure(pw, ph);
        if (!fbo_.valid())
            return;

        forwardInput(x, y);

        auto* render = manager_->getRenderInterface();
        assert(render);
        render->SetViewport(pw, ph);

        GLint prev_fbo = 0;
        fbo_.bind(&prev_fbo);

        render->BeginFrame();
        rml_context_->Render();
        render->EndFrame();

        fbo_.unbind(prev_fbo);

        auto* vp = ImGui::GetMainViewport();
        auto* draw_list = foreground_ ? ImGui::GetForegroundDrawList(vp)
                                      : ImGui::GetBackgroundDrawList(vp);
        fbo_.blitToDrawList(draw_list, {x, y}, {w, display_h});

        if (height_mode_ == HeightMode::Content && !content_dirty_) {
            auto* frame = document_->GetElementById("window-frame");
            auto* wrap = frame ? frame : document_->GetElementById("content-wrap");
            if (wrap) {
                const float actual_h = wrap->GetOffsetHeight();
                if (std::abs(actual_h - static_cast<float>(ph)) > 2.0f)
                    content_dirty_ = true;
            }
        }
    }

    void RmlPanelHost::forwardInput(float panel_x, float panel_y) {
        assert(rml_context_);

        ImGuiIO& io = ImGui::GetIO();
        ImVec2 mouse = io.MousePos;

        float local_x = mouse.x - panel_x;
        float local_y = mouse.y - panel_y;

        const float dp_ratio = manager_->getDpRatio();
        const float logical_w = static_cast<float>(fbo_.width()) / dp_ratio;
        const float logical_h = static_cast<float>(fbo_.height()) / dp_ratio;

        bool hovered = local_x >= 0 && local_y >= 0 && local_x < logical_w && local_y < logical_h;

        if (hovered) {
            rml_context_->ProcessMouseMove(static_cast<int>(local_x * dp_ratio),
                                           static_cast<int>(local_y * dp_ratio), 0);

            if (ImGui::IsMouseClicked(ImGuiMouseButton_Left))
                rml_context_->ProcessMouseButtonDown(0, 0);
            if (ImGui::IsMouseReleased(ImGuiMouseButton_Left))
                rml_context_->ProcessMouseButtonUp(0, 0);

            if (ImGui::IsMouseClicked(ImGuiMouseButton_Right))
                rml_context_->ProcessMouseButtonDown(1, 0);
            if (ImGui::IsMouseReleased(ImGuiMouseButton_Right))
                rml_context_->ProcessMouseButtonUp(1, 0);

            float wheel = io.MouseWheel;
            if (wheel != 0.0f)
                rml_context_->ProcessMouseWheel(Rml::Vector2f(0, -wheel), 0);

            if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
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
        } else if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            if (has_text_focus_) {
                drainTextInput();
                has_text_focus_ = false;
                SDL_StopTextInput(manager_->getWindow());
            }
        }

        bool forward_keys = has_text_focus_ || hovered;
        if (has_text_focus_) {
            io.WantCaptureKeyboard = true;
            io.WantTextInput = true;
        }

        if (forward_keys) {
            int mods = buildRmlModifiers();
            for (int k = ImGuiKey_NamedKey_BEGIN; k < ImGuiKey_NamedKey_END; ++k) {
                auto imgui_key = static_cast<ImGuiKey>(k);
                auto rml_key = imguiKeyToRml(imgui_key);
                if (rml_key == Rml::Input::KI_UNKNOWN)
                    continue;
                if (ImGui::IsKeyPressed(imgui_key, false))
                    rml_context_->ProcessKeyDown(rml_key, mods);
                if (ImGui::IsKeyReleased(imgui_key))
                    rml_context_->ProcessKeyUp(rml_key, mods);
            }
        }

        if (has_text_focus_) {
            auto chars = drainTextInput();
            for (uint32_t cp : chars)
                rml_context_->ProcessTextInput(static_cast<Rml::Character>(cp));
        }
    }

} // namespace lfs::vis::gui
