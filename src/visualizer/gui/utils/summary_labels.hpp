/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/event_bridge/localization_manager.hpp"
#include "core/number_format.hpp"
#include <string>
#include <string_view>

namespace lfs::vis::gui {

    [[nodiscard]] inline bool sceneCountUsesFew(const size_t count) {
        return count % 10 >= 2 && count % 10 <= 4 &&
               (count % 100 < 12 || count % 100 > 14) &&
               lfs::event::LocalizationManager::getInstance().getCurrentLanguage() == "pl";
    }

    [[nodiscard]] inline std::string formatSceneModelCount(const size_t count) {
        const auto text = lfs::core::format_count(count);
        if (count == 1)
            return LOCF("scene.summary_model_one", text);
        if (sceneCountUsesFew(count))
            return LOCF("scene.summary_model_few", text);
        return LOCF("scene.summary_model_other", text);
    }

    [[nodiscard]] inline std::string formatSceneNodeCount(const size_t count) {
        const auto text = lfs::core::format_count(count);
        if (count == 1)
            return LOCF("scene.summary_node_one", text);
        if (sceneCountUsesFew(count))
            return LOCF("scene.summary_node_few", text);
        return LOCF("scene.summary_node_other", text);
    }

    [[nodiscard]] inline std::string stripLabelColon(const std::string& text) {
        std::string_view label(text);
        while (!label.empty()) {
            if (label.back() == ':' || label.back() == ' ')
                label.remove_suffix(1);
            else if (label.ends_with("："))
                label.remove_suffix(std::string_view("：").size());
            else
                break;
        }
        return label.empty() ? text : std::string(label);
    }

} // namespace lfs::vis::gui
