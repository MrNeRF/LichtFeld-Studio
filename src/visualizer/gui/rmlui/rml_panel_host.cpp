/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// clang-format off
#include <glad/glad.h>
// clang-format on

#include "gui/rmlui/rml_panel_host.hpp"
#include "core/logger.hpp"
#include "gui/rmlui/rmlui_manager.hpp"
#include "gui/rmlui/rmlui_render_interface.hpp"
#include "internal/resource_paths.hpp"
#include "theme/theme.hpp"

#include <RmlUi/Core.h>
#include <RmlUi/Core/Factory.h>
#include <RmlUi/Core/Input.h>
#include <cassert>
#include <filesystem>
#include <format>
#include <fstream>
#include <imgui.h>

namespace lfs::vis::gui {

    namespace {
        std::string colorToRml(const ImVec4& c) {
            const auto r = static_cast<int>(c.x * 255.0f);
            const auto g = static_cast<int>(c.y * 255.0f);
            const auto b = static_cast<int>(c.z * 255.0f);
            const auto a = static_cast<int>(c.w * 255.0f);
            return std::format("rgba({},{},{},{})", r, g, b, a);
        }

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

    RmlPanelHost::~RmlPanelHost() { destroyFBO(); }

    void RmlPanelHost::initFBO(int width, int height) {
        if (fbo_ && fbo_width_ == width && fbo_height_ == height)
            return;

        destroyFBO();

        fbo_width_ = width;
        fbo_height_ = height;

        glGenFramebuffers(1, &fbo_);
        glGenTextures(1, &fbo_texture_);
        glGenRenderbuffers(1, &fbo_depth_stencil_);

        glBindTexture(GL_TEXTURE_2D, fbo_texture_);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                     nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glBindTexture(GL_TEXTURE_2D, 0);

        glBindRenderbuffer(GL_RENDERBUFFER, fbo_depth_stencil_);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height);
        glBindRenderbuffer(GL_RENDERBUFFER, 0);

        glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fbo_texture_,
                               0);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER,
                                  fbo_depth_stencil_);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            LOG_ERROR("RmlUI panel FBO incomplete");
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            destroyFBO();
            return;
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void RmlPanelHost::destroyFBO() {
        if (fbo_texture_) {
            glDeleteTextures(1, &fbo_texture_);
            fbo_texture_ = 0;
        }
        if (fbo_depth_stencil_) {
            glDeleteRenderbuffers(1, &fbo_depth_stencil_);
            fbo_depth_stencil_ = 0;
        }
        if (fbo_) {
            glDeleteFramebuffers(1, &fbo_);
            fbo_ = 0;
        }
        fbo_width_ = 0;
        fbo_height_ = 0;
    }

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
            "#filter-input {{ color: {0}; background-color: {2}; border-width: 1dp; border-color: {5}; }}\n"
            "#filter-input:focus {{ border-color: {4}; }}\n"
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
            ".context-menu {{ background-color: {2}; border-width: 1dp; border-color: {5}; }}\n"
            ".context-menu-item {{ color: {0}; }}\n"
            ".context-menu-item:hover {{ background-color: {4}; }}\n"
            ".context-menu-separator {{ background-color: {5}; }}\n"
            ".section-header {{ color: {0}; }}\n"
            ".section-header:hover {{ background-color: {3}; }}\n"
            ".empty-message {{ color: {1}; }}\n"
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
            try {
                auto rcss_path = lfs::vis::getAssetPath(
                    std::filesystem::path(rml_path_).replace_extension(".rcss").string());
                std::ifstream f(rcss_path);
                if (f) {
                    base_rcss_.assign(std::istreambuf_iterator<char>(f),
                                      std::istreambuf_iterator<char>());
                }
            } catch (...) {
            }
        }

        const std::string combined = base_rcss_ + "\n" + generateThemeRCSS();
        auto sheet = Rml::Factory::InstanceStyleSheetString(combined);
        if (sheet)
            document_->SetStyleSheetContainer(std::move(sheet));
    }

    void RmlPanelHost::draw(const PanelDrawContext& ctx) {
        (void)ctx;

        const float avail_w = ImGui::GetContentRegionAvail().x;
        const float avail_h = ImGui::GetContentRegionAvail().y;
        if (avail_w <= 0 || avail_h <= 0)
            return;

        const float dp_ratio = manager_->getDpRatio();
        const int w = static_cast<int>(avail_w * dp_ratio);
        const int h = static_cast<int>(avail_h * dp_ratio);

        if (!rml_context_) {
            rml_context_ = manager_->createContext(context_name_, w, h);
            if (!rml_context_)
                return;

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
        if (!rml_context_ || !document_)
            return;

        syncThemeProperties();

        rml_context_->SetDimensions(Rml::Vector2i(w, h));
        rml_context_->Update();

        initFBO(w, h);
        if (!fbo_)
            return;

        ImVec2 panel_pos = ImGui::GetCursorScreenPos();
        forwardInput(panel_pos.x, panel_pos.y);

        auto* render = manager_->getRenderInterface();
        assert(render);
        render->SetViewport(fbo_width_, fbo_height_);

        GLint prev_fbo = 0;
        glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prev_fbo);

        glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

        render->BeginFrame();
        rml_context_->Render();
        render->EndFrame();

        glBindFramebuffer(GL_FRAMEBUFFER, prev_fbo);

        ImGui::Image(static_cast<ImTextureID>(static_cast<uintptr_t>(fbo_texture_)),
                     ImVec2(avail_w, avail_h), ImVec2(0, 1), ImVec2(1, 0));
    }

    void RmlPanelHost::forwardInput(float panel_x, float panel_y) {
        assert(rml_context_);

        ImGuiIO& io = ImGui::GetIO();
        ImVec2 mouse = io.MousePos;

        float local_x = mouse.x - panel_x;
        float local_y = mouse.y - panel_y;

        const float dp_ratio = manager_->getDpRatio();
        const float logical_w = static_cast<float>(fbo_width_) / dp_ratio;
        const float logical_h = static_cast<float>(fbo_height_) / dp_ratio;

        bool hovered = local_x >= 0 && local_y >= 0 && local_x < logical_w && local_y < logical_h;

        if (!hovered)
            return;

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

        for (int i = 0; i < io.InputQueueCharacters.Size; ++i) {
            rml_context_->ProcessTextInput(static_cast<Rml::Character>(io.InputQueueCharacters[i]));
        }
    }

} // namespace lfs::vis::gui
