/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "input/input_bindings.hpp"
#include <algorithm>
#include <nlohmann/json.hpp>
#include <set>
#include <type_traits>

namespace lfs::vis::input::detail {
    // Keep one override per thread across the application/DLL boundary.
    LFS_VIS_API const Profile*& defaultProfileOverride();

    class ScopedDefaultProfileOverride {
    public:
        explicit ScopedDefaultProfileOverride(const Profile& profile)
            : previous_(defaultProfileOverride()) {
            defaultProfileOverride() = &profile;
        }
        ~ScopedDefaultProfileOverride() { defaultProfileOverride() = previous_; }
        ScopedDefaultProfileOverride(const ScopedDefaultProfileOverride&) = delete;
        ScopedDefaultProfileOverride& operator=(const ScopedDefaultProfileOverride&) = delete;

    private:
        const Profile* previous_;
    };

    inline Profile defaultProfile() {
        const auto* profile = defaultProfileOverride();
        return profile ? *profile : InputBindings::createDefaultProfile();
    }

    inline nlohmann::json profileBindingJson(const Binding& binding) {
        nlohmann::json value;
        value["mode"] = static_cast<int>(binding.mode);
        value["action"] = static_cast<int>(binding.action);
        value["description"] = binding.description;
        std::visit([&value](const auto& trigger) {
            using T = std::decay_t<decltype(trigger)>;
            if constexpr (std::is_same_v<T, KeyTrigger>) {
                value["trigger_type"] = "key";
                value["key"] = trigger.key;
                value["modifiers"] = trigger.modifiers;
                value["on_repeat"] = trigger.on_repeat;
            } else if constexpr (std::is_same_v<T, MouseButtonTrigger>) {
                value["trigger_type"] = "mouse_button";
                value["button"] = static_cast<int>(trigger.button);
                value["modifiers"] = trigger.modifiers;
                value["double_click"] = trigger.double_click;
            } else if constexpr (std::is_same_v<T, MouseScrollTrigger>) {
                value["trigger_type"] = "scroll";
                value["modifiers"] = trigger.modifiers;
                if (trigger.chord_key)
                    value["chord_key"] = *trigger.chord_key;
            } else {
                value["trigger_type"] = "drag";
                value["button"] = static_cast<int>(trigger.button);
                value["modifiers"] = trigger.modifiers;
                if (trigger.chord_key)
                    value["chord_key"] = *trigger.chord_key;
            }
        },
                   binding.trigger);
        return value;
    }

    inline std::string bindingContentKey(const Binding& binding) {
        auto value = profileBindingJson(binding);
        value.erase("description");
        return value.dump();
    }

    inline void applyRemovedBindingOverrides(std::vector<Binding>& defaults,
                                             const std::set<std::string>& removed_keys) {
        std::erase_if(defaults, [&](const Binding& binding) {
            return removed_keys.contains(bindingContentKey(binding));
        });
    }
} // namespace lfs::vis::input::detail
