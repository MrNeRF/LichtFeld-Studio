/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/panels/console_area_panel.hpp"

#include <utility>

namespace lfs::vis::gui {

    PythonConsoleAreaPanel::PythonConsoleAreaPanel(RmlUIManager* manager,
                                                   std::string instance_id)
        : manager_(manager),
          instance_id_(std::move(instance_id)),
          pane_(manager_, instance_id_) {}

    std::shared_ptr<IPanel> PythonConsoleAreaPanel::createAreaInstance(
        const std::string_view instance_id) const {
        if (instance_id.empty())
            return nullptr;
        return std::make_shared<PythonConsoleAreaPanel>(manager_, std::string(instance_id));
    }

    PanelDirectRenderResult PythonConsoleAreaPanel::renderDirect(
        const PanelDirectRenderRequest& request, const PanelDrawContext& ctx) {
        if (request.mode == PanelDirectRenderMode::Measure)
            return {.handled = true, .height = request.height};
        if (request.mode == PanelDirectRenderMode::Cached)
            return {.handled = false};
        if (request.mode == PanelDirectRenderMode::Preload) {
            if (!ctx.ui)
                return {};
            pane_.render(*ctx.ui, request.x, request.y, request.width, request.height, nullptr);
            return {.handled = true, .height = request.height};
        }
        if (!ctx.ui)
            return {};
        return {.handled = pane_.render(*ctx.ui, request.x, request.y,
                                        request.width, request.height, request.input),
                .height = request.height};
    }

    bool PythonConsoleAreaPanel::needsAnimationFrame() const {
        return pane_.needsAnimationFrame();
    }

    void PythonConsoleAreaPanel::releaseRendererResources() {
        pane_.releaseRendererResources();
    }

    std::string PythonConsoleAreaPanel::captureChromeJson() const {
        return pane_.captureChromeJson();
    }

    void PythonConsoleAreaPanel::applyChromeJson(const std::string_view json) {
        pane_.applyChromeJson(json);
    }

} // namespace lfs::vis::gui
