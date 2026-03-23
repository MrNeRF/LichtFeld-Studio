/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/event_bridge/localization_manager.hpp"

#include <RmlUi/Core/Element.h>
#include <chrono>
#include <string>
#include <string_view>

namespace lfs::vis::gui {

    inline constexpr auto kRmlTooltipShowDelay = std::chrono::milliseconds(700);

    inline std::string resolveRmlTooltip(Rml::Element* hover) {
        for (auto* el = hover; el; el = el->GetParentNode()) {
            const auto key = el->GetAttribute<Rml::String>("data-tooltip", "");
            if (!key.empty()) {
                auto& loc = lfs::event::LocalizationManager::getInstance();
                const char* resolved = loc.get(key.c_str());
                if (!resolved || std::string_view(resolved) == key.c_str())
                    return {};
                return resolved;
            }

            const auto title = el->GetAttribute<Rml::String>("title", "");
            if (!title.empty())
                return std::string(title.c_str(), title.size());
        }
        return {};
    }

} // namespace lfs::vis::gui
