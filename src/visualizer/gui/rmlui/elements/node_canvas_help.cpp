/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "node_canvas_element.hpp"

#include "core/logger.hpp"
#include "gui/rmlui/rml_path_utils.hpp"
#include "visualizer/nodes/modifier_manager.hpp"

#include <RmlUi/Core/Context.h>
#include <RmlUi/Core/Event.h>
#include <SDL3/SDL.h>

namespace lfs::vis::gui {
    bool NodeCanvasElement::processHelpEvent(Rml::Event& event) {
        if (event.GetType() == "focus" || event.GetType() == "blur") {
            auto* focused = event.GetType() == "focus" ? event.GetTargetElement() : nullptr;
            Rml::ElementList rows;
            if (sidebar_element_)
                sidebar_element_->QuerySelectorAll(rows, ".node-setting-help");
            for (auto* row : rows) {
                bool within = false;
                for (auto* element = focused; element; element = element->GetParentNode())
                    within |= element == row;
                row->SetClass("show-help", within);
            }
        }
        if (event.GetType() != "click")
            return false;
        for (auto* element = event.GetTargetElement(); element && element != this; element = element->GetParentNode()) {
            const auto action = element->GetAttribute<Rml::String>("data-action", "");
            if (action != "node-help" && action != "node-learn-more")
                continue;
            const auto id = element->GetAttribute<Rml::String>("data-type", "");
            if (action == "node-help") {
                expanded_help_[id] = !expanded_help_[id];
                updateSidebar();
            } else {
                const auto descriptor = manager_ ? manager_->registry().find(id) : nullptr;
                if (descriptor && !descriptor->localization_key.empty()) {
                    // Documentation links use the user's default browser, as
                    // the Getting Started panel does. SDL also supports macOS.
                    auto url = "https://github.com/MrNeRF/LichtFeld-Studio/blob/dev/docs/docs/development/node-graph/nodes/" + id + ".md";
#if defined(PROJECT_ROOT_PATH) && !defined(LFS_MACOS_PORTABLE_APP)
                    const auto local = std::filesystem::path(PROJECT_ROOT_PATH) /
                                       "docs/docs/development/node-graph/nodes" / (id + ".md");
                    if (std::filesystem::exists(local))
                        url = rml_paths::filesystemPathToFileUri(local);
#endif
                    if (!SDL_OpenURL(url.c_str()))
                        LOG_WARN("Could not open node reference '{}': {}", url, SDL_GetError());
                }
            }
            event.StopPropagation();
            return true;
        }
        return false;
    }
} // namespace lfs::vis::gui
