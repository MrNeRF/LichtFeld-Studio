/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/event_bridge/localization_manager.hpp"
#include "property_system.hpp"

#include <string>

namespace lfs::core::prop {

    [[nodiscard]] inline std::string property_display_name(const PropertyMeta& meta) {
        if (!meta.ui_locale_key.empty()) {
            std::string label = LOC(meta.ui_locale_key);
            if (!label.empty() && label != meta.ui_locale_key) {
                while (!label.empty() && (label.back() == ':' || label.back() == ' ' || label.back() == '\t')) {
                    label.pop_back();
                }
                if (label.size() >= 3 &&
                    static_cast<unsigned char>(label[label.size() - 3]) == 0xEF &&
                    static_cast<unsigned char>(label[label.size() - 2]) == 0xBC &&
                    static_cast<unsigned char>(label[label.size() - 1]) == 0x9A) {
                    label.erase(label.size() - 3);
                }
                while (!label.empty() && (label.back() == ' ' || label.back() == '\t')) {
                    label.pop_back();
                }
                if (!label.empty())
                    return label;
            }
        }
        return meta.name.empty() ? meta.id : meta.name;
    }

} // namespace lfs::core::prop
