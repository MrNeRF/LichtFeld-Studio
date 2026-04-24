/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// clang-format off
#include <glad/glad.h>
// clang-format on

#include "gui/rml_shell_frame.hpp"
#include "core/logger.hpp"
#include "gui/rmlui/rml_document_utils.hpp"
#include "gui/rmlui/rml_theme.hpp"
#include "gui/rmlui/rmlui_manager.hpp"
#include "gui/rmlui/rmlui_render_interface.hpp"
#include "internal/resource_paths.hpp"
#include "theme/theme.hpp"

#include <RmlUi/Core.h>
#include <RmlUi/Core/Element.h>
#include <cassert>
#include <format>

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
            document_ = rml_documents::loadDocument(rml_context_, rml_path);
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

    void RmlShellFrame::reloadResources() {
        if (!rml_context_)
            return;

        if (document_) {
            rml_context_->UnloadDocument(document_);
            rml_context_->Update();
        }

        document_ = nullptr;
        menu_region_ = nullptr;
        right_panel_region_ = nullptr;
        status_region_ = nullptr;
        base_rcss_.clear();
        has_theme_signature_ = false;

        try {
            const auto rml_path = lfs::vis::getAssetPath("rmlui/shell.rml");
            document_ = rml_documents::loadDocument(rml_context_, rml_path);
            if (!document_) {
                LOG_ERROR("RmlShellFrame: failed to reload shell.rml");
                return;
            }
            document_->Show();
        } catch (const std::exception& e) {
            LOG_ERROR("RmlShellFrame: resource not found during reload: {}", e.what());
            return;
        }

        menu_region_ = document_->GetElementById("menu-region");
        right_panel_region_ = document_->GetElementById("right-panel-region");
        status_region_ = document_->GetElementById("status-region");
        updateTheme();
    }

    void RmlShellFrame::updateTheme() {
        if (!document_)
            return;

        const std::size_t theme_signature = rml_theme::currentThemeSignature();
        if (has_theme_signature_ && theme_signature == last_theme_signature_)
            return;
        last_theme_signature_ = theme_signature;
        has_theme_signature_ = true;

        if (base_rcss_.empty())
            base_rcss_ = rml_theme::loadBaseRCSS("rmlui/shell.rcss");

        rml_theme::applyTheme(document_, base_rcss_, rml_theme::loadBaseRCSS("rmlui/shell.theme.rcss"));
    }

    void RmlShellFrame::render(const ShellRegions& regions) {
        if (!rml_context_ || !document_)
            return;

        const float full_w = regions.screen.w;
        const float full_h = regions.screen.h;
        if (full_w <= 0 || full_h <= 0)
            return;

        updateTheme();

        const int w = static_cast<int>(full_w);
        const int h = static_cast<int>(full_h);

        const float work_y = regions.menu.y + regions.menu.h - regions.screen.y;

        if (menu_region_) {
            menu_region_->SetProperty("top", std::format("{:.0f}px", regions.menu.y - regions.screen.y));
            menu_region_->SetProperty("height", std::format("{:.0f}px", regions.menu.h));
        }
        if (right_panel_region_) {
            right_panel_region_->SetProperty("top", std::format("{:.0f}px", work_y));
            right_panel_region_->SetProperty("right", "0px");
            right_panel_region_->SetProperty("width", std::format("{:.0f}px", regions.right_panel.w));
            right_panel_region_->SetProperty("height", std::format("{:.0f}px", regions.right_panel.h));
        }
        if (status_region_) {
            status_region_->SetProperty("height", std::format("{:.0f}px", regions.status.h));
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
            render->SetTargetFramebuffer(fbo_.fbo());

            render->BeginFrame();
            rml_context_->Render();
            render->EndFrame();

            render->SetTargetFramebuffer(0);
            fbo_.unbind(prev_fbo);
        }

        if (fbo_.valid())
            fbo_.blitToScreen(0.0f, 0.0f, full_w, full_h, w, h);
    }

} // namespace lfs::vis::gui
