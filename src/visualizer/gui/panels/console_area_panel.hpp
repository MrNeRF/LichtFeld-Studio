/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/panel_registry.hpp"
#include "gui/panels/python_console_panel.hpp"

#include <memory>
#include <string>
#include <string_view>

namespace lfs::vis::gui {

    class LFS_VIS_API PythonConsoleAreaPanel final : public IPanel {
    public:
        explicit PythonConsoleAreaPanel(RmlUIManager* manager,
                                        std::string instance_id = "python_console_area");

        [[nodiscard]] bool supportsAreaInstances() const override { return true; }
        [[nodiscard]] std::shared_ptr<IPanel> createAreaInstance(
            std::string_view instance_id) const override;

        void draw(const PanelDrawContext&) override {}
        PanelRenderCapabilities renderCapabilities() const override { return {.direct = true}; }
        PanelDirectRenderResult renderDirect(const PanelDirectRenderRequest& request,
                                             const PanelDrawContext& ctx) override;
        bool needsAnimationFrame() const override;
        void releaseRendererResources() override;
        [[nodiscard]] std::string captureChromeJson() const override;
        void applyChromeJson(std::string_view json) override;

    private:
        RmlUIManager* manager_ = nullptr;
        std::string instance_id_;
        panels::PythonConsolePane pane_;
    };

} // namespace lfs::vis::gui
