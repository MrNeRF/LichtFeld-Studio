/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// clang-format off
#include <glad/glad.h>
// clang-format on

#include "gui/rml_shell_frame.hpp"
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

    void RmlShellFrame::init(RmlUIManager* mgr) {
        assert(mgr);
        rml_manager_ = mgr;

        rml_context_ = rml_manager_->createContext("shell_frame", 800, 600);
        if (!rml_context_) {
            LOG_ERROR("RmlShellFrame: failed to create RML context");
            return;
        }

        try {
            const auto rml_path = lfs::vis::getAssetPath("rmlui/shell.rml");
            document_ = rml_context_->LoadDocument(rml_path.string());
            if (!document_) {
                LOG_ERROR("RmlShellFrame: failed to load shell.rml");
                return;
            }
            document_->Show();
        } catch (const std::exception& e) {
            LOG_ERROR("RmlShellFrame: resource not found: {}", e.what());
            return;
        }

        menu_region_ = document_->GetElementById("menu-region");
        right_panel_region_ = document_->GetElementById("right-panel-region");
        status_region_ = document_->GetElementById("status-region");

        updateTheme();
    }

    void RmlShellFrame::shutdown() {
        fbo_.destroy();
        if (rml_context_ && rml_manager_)
            rml_manager_->destroyContext("shell_frame");
        rml_context_ = nullptr;
        document_ = nullptr;
        menu_region_ = nullptr;
        right_panel_region_ = nullptr;
        status_region_ = nullptr;
    }

    std::string RmlShellFrame::generateThemeRCSS() const {
        using rml_theme::colorToRml;
        const auto& t = lfs::vis::theme();

        const auto shell_bg = colorToRml(t.menu_background());

        return std::format(
            "#menu-region {{ background-color: {}; }}\n"
            "#right-panel-region {{ background-color: {}; }}\n"
            "#status-region {{ background-color: {}; }}\n",
            shell_bg, shell_bg, shell_bg);
    }

    void RmlShellFrame::updateTheme() {
        if (!document_)
            return;

        const auto& t = lfs::vis::theme();
        if (t.name == last_theme_)
            return;
        last_theme_ = t.name;

        if (base_rcss_.empty())
            base_rcss_ = rml_theme::loadBaseRCSS("rmlui/shell.rcss");

        rml_theme::applyTheme(document_, base_rcss_, generateThemeRCSS());
    }

    void RmlShellFrame::render(const ShellRegions& regions) {
        if (!rml_context_ || !document_)
            return;

        auto* vp = ImGui::GetMainViewport();
        const float full_w = vp->Size.x;
        const float full_h = vp->Size.y;
        if (full_w <= 0 || full_h <= 0)
            return;

        updateTheme();

        const int w = static_cast<int>(full_w);
        const int h = static_cast<int>(full_h);

        const float work_y = vp->WorkPos.y - vp->Pos.y;

        if (menu_region_) {
            menu_region_->SetProperty("top", std::format("{:.0f}px", regions.menu_pos.y - vp->Pos.y));
            menu_region_->SetProperty("height", std::format("{:.0f}px", regions.menu_size.y));
        }
        if (right_panel_region_) {
            right_panel_region_->SetProperty("top", std::format("{:.0f}px", work_y));
            right_panel_region_->SetProperty("right", "0px");
            right_panel_region_->SetProperty("width", std::format("{:.0f}px", regions.right_panel_size.x));
            right_panel_region_->SetProperty("height", std::format("{:.0f}px", regions.right_panel_size.y));
        }
        if (status_region_) {
            status_region_->SetProperty("height", std::format("{:.0f}px", regions.status_size.y));
        }

        if (!rml_manager_->shouldDeferFboUpdate(fbo_)) {
            rml_context_->SetDimensions(Rml::Vector2i(w, h));
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
        }

        if (fbo_.valid())
            fbo_.blitToDrawList(ImGui::GetBackgroundDrawList(vp),
                                vp->Pos, vp->Size);
    }

} // namespace lfs::vis::gui
