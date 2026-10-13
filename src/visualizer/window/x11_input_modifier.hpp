/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace lfs::vis::window {

    struct X11InputModifierPlan {
        std::optional<std::string> override_value;
        std::optional<std::string> restore_value;
    };

    inline X11InputModifierPlan selectX11InputModifier(std::optional<std::string_view> current,
                                                       bool ibus_xim_available) {
        if (current && *current == "@im=ibus" && !ibus_xim_available)
            return {.override_value = "@im=none", .restore_value = std::string(*current)};
        return {};
    }

    template <typename Getter, typename Setter, typename Operation>
    auto withX11InputModifier(Getter&& get, Setter&& set, bool ibus_xim_available, Operation&& operation) {
        const char* const current = std::invoke(std::forward<Getter>(get));
        const auto plan = selectX11InputModifier(current ? std::optional<std::string_view>(current) : std::nullopt,
                                                 ibus_xim_available);
        const bool changed = plan.override_value &&
                             std::invoke(std::forward<Setter>(set), *plan.override_value) == 0;

        struct Restore {
            Setter& set;
            const std::optional<std::string>& value;
            bool active;
            ~Restore() {
                if (active)
                    std::invoke(set, *value);
            }
        } restore{set, plan.restore_value, changed};
        return std::invoke(std::forward<Operation>(operation));
    }

} // namespace lfs::vis::window
