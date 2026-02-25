/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// clang-format off
#include <glad/glad.h>
// clang-format on

#include "gui/rml_menu_bar.hpp"
#include "core/logger.hpp"
#include "gui/rmlui/rml_theme.hpp"
#include "gui/rmlui/rmlui_manager.hpp"
#include "gui/rmlui/rmlui_render_interface.hpp"
#include "internal/resource_paths.hpp"
#include "theme/theme.hpp"

#include <RmlUi/Core.h>
#include <RmlUi/Core/Element.h>
#include <cassert>
#include <format>
#include <imgui.h>

namespace lfs::vis::gui {

    namespace {
        ImVec4 darkenImVec4(const ImVec4& c, float amount) {
            return {c.x - amount, c.y - amount, c.z - amount, c.w};
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
        fbo_.destroy();
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
        using rml_theme::colorToRml;
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

        if (base_rcss_.empty())
            base_rcss_ = rml_theme::loadBaseRCSS("rmlui/menubar.rcss");

        rml_theme::applyTheme(document_, base_rcss_, generateThemeRCSS());
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

        fbo_.ensure(w, h);
        if (!fbo_.valid())
            return;

        auto* render = rml_manager_->getRenderInterface();
        assert(render);
        render->SetViewport(w, h);

        GLint prev_fbo = 0;
        fbo_.bind(&prev_fbo);

        render->BeginFrame();
        rml_context_->Render();
        render->EndFrame();

        fbo_.unbind(prev_fbo);

        fbo_.blitToDrawList(ImGui::GetWindowDrawList(), win_pos, win_size);
    }

} // namespace lfs::vis::gui
