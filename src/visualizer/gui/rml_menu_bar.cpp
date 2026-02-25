/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// clang-format off
#include <glad/glad.h>
// clang-format on

#include "gui/rml_menu_bar.hpp"
#include "core/logger.hpp"
#include "gui/rmlui/rmlui_manager.hpp"
#include "gui/rmlui/rmlui_render_interface.hpp"
#include "internal/resource_paths.hpp"
#include "theme/theme.hpp"

#include <RmlUi/Core.h>
#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/Factory.h>
#include <cassert>
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

        ImVec4 darkenImVec4(const ImVec4& c, float amount) {
            return {c.x - amount, c.y - amount, c.z - amount, c.w};
        }

        void setPremultipliedBlend(const ImDrawList*, const ImDrawCmd*) {
            glBlendFuncSeparate(GL_ONE, GL_ONE_MINUS_SRC_ALPHA,
                                GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
        }

        void restoreStandardBlend(const ImDrawList*, const ImDrawCmd*) {
            glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
                                GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
        }
    } // namespace

    void RmlMenuBar::init(RmlUIManager* mgr) {
        assert(mgr);
        rml_manager_ = mgr;

        rml_context_ = rml_manager_->createContext("menu_bar", 800, 30);
        if (!rml_context_) {
            LOG_ERROR("RmlMenuBar: failed to create RML context");
            return;
        }

        try {
            const auto rml_path = lfs::vis::getAssetPath("rmlui/menubar.rml");
            document_ = rml_context_->LoadDocument(rml_path.string());
            if (!document_) {
                LOG_ERROR("RmlMenuBar: failed to load menubar.rml");
                return;
            }
            document_->Show();
        } catch (const std::exception& e) {
            LOG_ERROR("RmlMenuBar: resource not found: {}", e.what());
            return;
        }

        menu_items_ = document_->GetElementById("menu-items");
        bottom_border_ = document_->GetElementById("bottom-border");

        updateTheme();
    }

    void RmlMenuBar::shutdown() {
        destroyFBO();
        if (rml_context_ && rml_manager_)
            rml_manager_->destroyContext("menu_bar");
        rml_context_ = nullptr;
        document_ = nullptr;
        menu_items_ = nullptr;
        bottom_border_ = nullptr;
    }

    void RmlMenuBar::updateLabels(const std::vector<std::string>& labels) {
        if (labels == current_labels_)
            return;
        current_labels_ = labels;
        rebuildLabels();
    }

    void RmlMenuBar::rebuildLabels() {
        if (!menu_items_)
            return;

        std::string rml;
        for (const auto& label : current_labels_) {
            rml += std::format("<span class=\"menu-label\">{}</span>", label);
        }
        menu_items_->SetInnerRML(rml);
    }

    void RmlMenuBar::setActiveIndex(int index) {
        if (index == active_index_)
            return;

        if (!menu_items_)
            return;

        const int count = menu_items_->GetNumChildren();

        if (active_index_ >= 0 && active_index_ < count)
            menu_items_->GetChild(active_index_)->SetClass("active", false);

        active_index_ = index;

        if (active_index_ >= 0 && active_index_ < count)
            menu_items_->GetChild(active_index_)->SetClass("active", true);
    }

    std::string RmlMenuBar::generateThemeRCSS() const {
        const auto& t = lfs::vis::theme();

        const auto bg = colorToRml(t.menu_background());
        const auto text = colorToRml(t.palette.text);
        const auto hover = colorToRml(t.menu_hover());
        const auto active = colorToRml(t.menu_active());
        const auto border = colorToRml(darkenImVec4(t.palette.surface, t.menu.bottom_border_darken));

        return std::format(
            "body {{ background-color: {}; color: {}; }}\n"
            ".menu-label:hover {{ background-color: {}; }}\n"
            ".menu-label.active {{ background-color: {}; }}\n"
            "#bottom-border {{ background-color: {}; }}\n",
            bg, text, hover, active, border);
    }

    void RmlMenuBar::updateTheme() {
        if (!document_)
            return;

        const auto& t = lfs::vis::theme();
        if (t.name == last_theme_)
            return;
        last_theme_ = t.name;

        if (base_rcss_.empty()) {
            try {
                auto rcss_path = lfs::vis::getAssetPath("rmlui/menubar.rcss");
                std::ifstream f(rcss_path);
                if (f) {
                    base_rcss_.assign(std::istreambuf_iterator<char>(f),
                                      std::istreambuf_iterator<char>());
                }
            } catch (const std::exception& e) {
                LOG_ERROR("RmlMenuBar: RCSS not found: {}", e.what());
            }
        }

        const std::string combined = base_rcss_ + "\n" + generateThemeRCSS();
        auto sheet = Rml::Factory::InstanceStyleSheetString(combined);
        if (sheet)
            document_->SetStyleSheetContainer(std::move(sheet));
    }

    void RmlMenuBar::draw() {
        if (!rml_context_ || !document_)
            return;

        const ImVec2 win_pos = ImGui::GetWindowPos();
        const ImVec2 win_size = ImGui::GetWindowSize();
        if (win_size.x <= 0 || win_size.y <= 0)
            return;

        updateTheme();

        const float dp_ratio = rml_manager_->getDpRatio();
        const int w = static_cast<int>(win_size.x * dp_ratio);
        const int h = static_cast<int>(win_size.y * dp_ratio);

        rml_context_->SetDimensions(Rml::Vector2i(w, h));
        document_->SetProperty("height", std::format("{}px", h));
        rml_context_->Update();

        initFBO(w, h);
        if (!fbo_)
            return;

        auto* render = rml_manager_->getRenderInterface();
        assert(render);
        render->SetViewport(w, h);

        GLint prev_fbo = 0;
        glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prev_fbo);

        glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

        render->BeginFrame();
        rml_context_->Render();
        render->EndFrame();

        glBindFramebuffer(GL_FRAMEBUFFER, prev_fbo);

        auto* dl = ImGui::GetWindowDrawList();
        dl->AddCallback(setPremultipliedBlend, nullptr);
        const ImVec2 p0 = win_pos;
        const ImVec2 p1 = {win_pos.x + win_size.x, win_pos.y + win_size.y};
        dl->AddImage(static_cast<ImTextureID>(static_cast<uintptr_t>(fbo_texture_)),
                     p0, p1, {0, 1}, {1, 0});
        dl->AddCallback(restoreStandardBlend, nullptr);
    }

    void RmlMenuBar::initFBO(int w, int h) {
        if (fbo_ && fbo_w_ == w && fbo_h_ == h)
            return;

        destroyFBO();
        fbo_w_ = w;
        fbo_h_ = h;

        glGenFramebuffers(1, &fbo_);
        glGenTextures(1, &fbo_texture_);
        glGenRenderbuffers(1, &fbo_depth_stencil_);

        glBindTexture(GL_TEXTURE_2D, fbo_texture_);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glBindTexture(GL_TEXTURE_2D, 0);

        glBindRenderbuffer(GL_RENDERBUFFER, fbo_depth_stencil_);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, w, h);
        glBindRenderbuffer(GL_RENDERBUFFER, 0);

        glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fbo_texture_, 0);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER,
                                  fbo_depth_stencil_);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            LOG_ERROR("RmlMenuBar: FBO incomplete");
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            destroyFBO();
            return;
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void RmlMenuBar::destroyFBO() {
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
        fbo_w_ = 0;
        fbo_h_ = 0;
    }

} // namespace lfs::vis::gui
