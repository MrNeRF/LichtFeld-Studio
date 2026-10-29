/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "input/input_bindings.hpp"
#include "core/error_bus.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/logger.hpp"
#include "core/native_fs.hpp"
#include "core/path_utils.hpp"
#include "core/source_site.hpp"
#include "core/user_paths.hpp"
#include "input/generated_keymap_defaults.hpp"
#include "input/keymap_profile_overrides.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <charconv>
#include <chrono>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <optional>
#include <ranges>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace lfs::vis::input {

    const Profile*& detail::defaultProfileOverride() {
        static thread_local const Profile* profile_override = nullptr;
        return profile_override;
    }

    namespace {

        std::atomic<bool> g_persistence_enabled{true};

        void publishKeymapNotice(const lfs::ErrorCode code, std::string message,
                                 const lfs::ErrorSurface surface = lfs::ErrorSurface::Toast,
                                 std::vector<lfs::ErrorAction> actions = {}) {
            lfs::ErrorBus::instance().publish(lfs::ErrorNotification{
                .error = lfs::make_error({
                    .code = code,
                    .domain = lfs::ErrorDomain::App,
                    .severity = lfs::Severity::Warning,
                    .user_message = std::move(message),
                    .detection = LFS_SOURCE_SITE_CURRENT(),
                }),
                .surface = surface,
                .actions = std::move(actions),
                .operation_id = lfs::OperationId::generate(),
            });
        }

        void notifyNewerProfile() {
            publishKeymapNotice(lfs::ErrorCode::Unsupported,
                                lfs::event::LocalizationManager::getInstance().get(
                                    "input_settings.newer_profile_unchanged"));
        }

        void notifyInvalidProfileEntries(const std::vector<size_t>& entry_indices) {
            std::ostringstream details;
            constexpr size_t kListedEntryLimit = 24;
            for (size_t i = 0; i < std::min(entry_indices.size(), kListedEntryLimit); ++i) {
                if (i != 0)
                    details << ", ";
                details << entry_indices[i];
            }
            if (entry_indices.size() > kListedEntryLimit)
                details << ", ... (" << entry_indices.size() << " entries)";
            std::string message = lfs::event::LocalizationManager::getInstance().get(
                "input_settings.invalid_entries_skipped");
            message += details.str();
            publishKeymapNotice(lfs::ErrorCode::InvalidArgument, std::move(message));
        }

        constexpr int PROFILE_VERSION = 40; // Editor document commands.

        [[nodiscard]] std::optional<Binding> parseStoredBinding(const nlohmann::json& value);
        [[nodiscard]] std::string triggerIdentity(const Binding& binding);
        [[nodiscard]] bool hasSemanticallyUsableBindingArray(const nlohmann::json& bindings);

        enum class StoredProfileVersionKind { LegacyOrCurrent,
                                              Newer,
                                              Invalid };

        struct StoredProfileVersion {
            StoredProfileVersionKind kind = StoredProfileVersionKind::Invalid;
            uint64_t value = 0;
        };

        [[nodiscard]] StoredProfileVersion readStoredProfileVersion(const nlohmann::json& stored) {
            const auto it = stored.find("version");
            if (it == stored.end())
                return {StoredProfileVersionKind::LegacyOrCurrent, 0};
            if (!it->is_number_integer())
                return {};
            if (it->is_number_unsigned()) {
                const auto value = it->get<uint64_t>();
                return {value > static_cast<uint64_t>(PROFILE_VERSION)
                            ? StoredProfileVersionKind::Newer
                            : StoredProfileVersionKind::LegacyOrCurrent,
                        value};
            }
            const auto value = it->get<int64_t>();
            if (value < 0)
                return {};
            return {value > PROFILE_VERSION ? StoredProfileVersionKind::Newer
                                            : StoredProfileVersionKind::LegacyOrCurrent,
                    static_cast<uint64_t>(value)};
        }

        bool hasValidStoredBindings(const nlohmann::json& stored) {
            if (!stored.is_object() ||
                readStoredProfileVersion(stored).kind == StoredProfileVersionKind::Invalid ||
                !stored.contains("bindings") || !stored["bindings"].is_array())
                return false;
            if (!hasSemanticallyUsableBindingArray(stored["bindings"]))
                return false;
            if (stored.contains("overrides")) {
                const auto& overrides = stored["overrides"];
                if (!overrides.is_object() || !overrides.contains("bindings") ||
                    !hasSemanticallyUsableBindingArray(overrides["bindings"]) ||
                    !overrides.contains("removed") || !overrides["removed"].is_array())
                    return false;
            }
            return true;
        }

        void notifyDamagedProfileBackedUp() {
            publishKeymapNotice(lfs::ErrorCode::InvalidArgument,
                                lfs::event::LocalizationManager::getInstance().get(
                                    "input_settings.damaged_profile_backed_up"));
        }

        static_assert(kLatestHistoricalProfileVersion == PROFILE_VERSION - 1,
                      "Snapshot the previous keymap defaults before bumping PROFILE_VERSION");
        constexpr Action LAST_ACTION = Action::EDITOR_STOP;
        constexpr int REMOVED_TOOL_MODE_2 = 2;
        constexpr int REMOVED_ACTION_39 = 39;
        constexpr int REMOVED_ACTION_66 = 66;
        constexpr std::array<ToolMode, 4> NODE_PICK_MODES = {
            ToolMode::GLOBAL,
            ToolMode::TRANSLATE,
            ToolMode::ROTATE,
            ToolMode::SCALE,
        };
        constexpr std::array<ToolMode, 4> DELETE_NODE_MODES = {
            ToolMode::GLOBAL,
            ToolMode::TRANSLATE,
            ToolMode::ROTATE,
            ToolMode::SCALE,
        };
        constexpr std::array<ToolMode, 3> DELETE_GAUSSIANS_MODES = {
            ToolMode::SELECTION,
            ToolMode::ALIGN,
            ToolMode::CROP_BOX,
        };

        [[nodiscard]] std::string toLowerCopy(std::string_view s) {
            std::string out(s);
            std::transform(out.begin(), out.end(), out.begin(),
                           [](unsigned char c) { return std::tolower(c); });
            return out;
        }

        // Look up an Action by case-insensitive description match. Used to
        // migrate stored profiles whose integer action IDs no longer line up
        // with the current enum (when entries were inserted/removed between
        // versions, the IDs shift but the descriptions stay stable).
        [[nodiscard]] std::optional<Action> findActionByDescription(std::string_view description) {
            static const auto* const table = [] {
                auto* const m = new std::unordered_map<std::string, Action>();
                constexpr int kActionCount = static_cast<int>(LAST_ACTION) + 1;
                for (int i = 0; i < kActionCount; ++i) {
                    const auto a = static_cast<Action>(i);
                    m->emplace(toLowerCopy(getActionName(a)), a);
                }
                return m;
            }();
            const auto it = table->find(toLowerCopy(description));
            if (it == table->end()) {
                return std::nullopt;
            }
            return it->second;
        }

        [[nodiscard]] bool isSelectionDepthAction(const Action action) {
            switch (action) {
            case Action::TOGGLE_DEPTH_MODE:
            case Action::DEPTH_ADJUST_NEAR:
            case Action::DEPTH_ADJUST_FAR:
            case Action::DEPTH_ADJUST_SIDE:
            case Action::TOGGLE_SELECTION_DEPTH_FILTER:
            case Action::TOGGLE_SELECTION_CROP_FILTER:
                return true;
            default:
                return false;
            }
        }

        [[nodiscard]] bool triggerUsesDefaultRedoBinding(const std::optional<InputTrigger>& trigger) {
            const auto* key_trigger = trigger ? std::get_if<KeyTrigger>(&*trigger) : nullptr;
            return key_trigger && key_trigger->key == KEY_Y &&
                   key_trigger->modifiers == MODIFIER_CTRL;
        }

        [[nodiscard]] bool triggersOverlap(const InputTrigger& a, const InputTrigger& b) {
            if (a.index() != b.index()) {
                return false;
            }
            return std::visit([&](const auto& lhs) -> bool {
                using T = std::decay_t<decltype(lhs)>;
                const auto& rhs = std::get<T>(b);
                if constexpr (std::is_same_v<T, KeyTrigger>) {
                    return lhs.key == rhs.key && lhs.modifiers == rhs.modifiers;
                } else if constexpr (std::is_same_v<T, MouseButtonTrigger>) {
                    return lhs.button == rhs.button && lhs.modifiers == rhs.modifiers &&
                           lhs.double_click == rhs.double_click;
                } else if constexpr (std::is_same_v<T, MouseScrollTrigger>) {
                    return lhs.modifiers == rhs.modifiers && lhs.chord_key == rhs.chord_key;
                } else if constexpr (std::is_same_v<T, MouseDragTrigger>) {
                    return lhs.button == rhs.button && lhs.modifiers == rhs.modifiers &&
                           lhs.chord_key == rhs.chord_key;
                } else {
                    return false;
                }
            },
                              a);
        }

        [[nodiscard]] std::string bindingKey(nlohmann::json binding) {
            binding.erase("description");
            binding.erase("action_symbol");
            return binding.dump();
        }

        [[nodiscard]] std::optional<Action> findActionBySymbol(std::string_view symbol) {
            for (const auto& [name, value] : kCurrentActionSymbols) {
                if (name == symbol)
                    return static_cast<Action>(value);
            }
            return std::nullopt;
        }

        [[nodiscard]] Binding normalizeLoadedBinding(Binding binding) {
            if (!isSelectionDepthAction(binding.action)) {
                return binding;
            }

            if (binding.action == Action::TOGGLE_DEPTH_MODE) {
                binding.action = Action::TOGGLE_SELECTION_DEPTH_FILTER;
            } else if (binding.action == Action::DEPTH_ADJUST_NEAR ||
                       binding.action == Action::DEPTH_ADJUST_SIDE) {
                binding.action = Action::DEPTH_ADJUST_FAR;
            }

            binding.mode = ToolMode::SELECTION;
            binding.description = getActionName(binding.action);
            return binding;
        }

        template <size_t N>
        size_t mirrorLegacyBindingToModes(std::vector<Binding>& bindings,
                                          const Binding& source,
                                          const Action target_action,
                                          const std::array<ToolMode, N>& target_modes) {
            size_t added = 0;
            for (const auto mode : target_modes) {
                const bool already_present = std::ranges::any_of(
                    bindings,
                    [&](const Binding& current) {
                        return current.mode == mode && current.action == target_action;
                    });
                if (already_present) {
                    continue;
                }

                Binding mirrored = source;
                mirrored.mode = mode;
                mirrored.action = target_action;
                mirrored.description = getActionName(target_action);
                bindings.push_back(std::move(mirrored));
                ++added;
            }
            return added;
        }

        size_t projectLegacyGlobalBindings(std::vector<Binding>& bindings, const int version) {
            if (version >= 2) {
                return 0;
            }

            const auto legacy_bindings = bindings;
            size_t added = 0;
            for (const auto& binding : legacy_bindings) {
                if (binding.mode != ToolMode::GLOBAL) {
                    continue;
                }

                switch (binding.action) {
                case Action::NONE:
                    break;
                case Action::DELETE_NODE:
                    added += mirrorLegacyBindingToModes(bindings, binding, Action::DELETE_NODE, DELETE_NODE_MODES);
                    added += mirrorLegacyBindingToModes(bindings, binding, Action::DELETE_SELECTED, DELETE_GAUSSIANS_MODES);
                    break;
                case Action::DELETE_SELECTED:
                    added += mirrorLegacyBindingToModes(bindings, binding, Action::DELETE_SELECTED, DELETE_GAUSSIANS_MODES);
                    break;
                case Action::NODE_PICK:
                    added += mirrorLegacyBindingToModes(bindings, binding, Action::NODE_PICK, NODE_PICK_MODES);
                    break;
                case Action::NODE_RECT_SELECT:
                    added += mirrorLegacyBindingToModes(bindings, binding, Action::NODE_RECT_SELECT, NODE_PICK_MODES);
                    break;
                case Action::TOGGLE_SELECTION_DEPTH_FILTER:
                case Action::TOGGLE_SELECTION_CROP_FILTER:
                case Action::DEPTH_ADJUST_FAR:
                    added += mirrorLegacyBindingToModes(bindings, binding, binding.action, std::array<ToolMode, 1>{ToolMode::SELECTION});
                    break;
                default:
                    if (!describe(binding.action).inherits_from_global) {
                        added += mirrorLegacyBindingToModes(bindings, binding, binding.action, kAllToolModes);
                    }
                    break;
                }
            }
            return added;
        }

        [[nodiscard]] bool isSupportedKeyCode(const int key) noexcept {
            return (key >= KEY_SPACE && key <= KEY_GRAVE_ACCENT) ||
                   (key >= KEY_ESCAPE && key <= KEY_END) ||
                   (key >= KEY_CAPS_LOCK && key <= KEY_PAUSE) ||
                   (key >= KEY_F1 && key <= KEY_F24) ||
                   (key >= KEY_KP_0 && key <= KEY_KP_EQUAL) ||
                   (key >= KEY_LEFT_SHIFT && key <= KEY_MENU);
        }

        [[nodiscard]] bool isKnownToolMode(const int value) noexcept {
            return std::ranges::any_of(kAllKeymapScopes, [value](const ToolMode mode) {
                return static_cast<int>(mode) == value;
            });
        }

        [[nodiscard]] bool isKnownActionId(const int value) noexcept {
            return std::ranges::any_of(kCurrentActionSymbols, [value](const auto& action) {
                return action.second == value;
            });
        }

        [[nodiscard]] std::optional<int> integerValue(const nlohmann::json& value) {
            if (!value.is_number_integer())
                return std::nullopt;
            if (value.is_number_unsigned()) {
                const auto number = value.get<uint64_t>();
                if (number > static_cast<uint64_t>(std::numeric_limits<int>::max()))
                    return std::nullopt;
                return static_cast<int>(number);
            }
            const auto number = value.get<int64_t>();
            if (number < std::numeric_limits<int>::min() ||
                number > std::numeric_limits<int>::max())
                return std::nullopt;
            return static_cast<int>(number);
        }

        [[nodiscard]] std::optional<Binding> parseStoredBinding(const nlohmann::json& value) {
            if (!value.is_object() || !value.contains("action") ||
                !value.contains("trigger_type") || !value["trigger_type"].is_string() ||
                (value.contains("description") && !value["description"].is_string()))
                return std::nullopt;
            const auto action_value = integerValue(value["action"]);
            const auto mode_value = value.contains("mode") ? integerValue(value["mode"])
                                                           : std::optional<int>(0);
            const auto modifier_value = value.contains("modifiers")
                                            ? integerValue(value["modifiers"])
                                            : std::optional<int>(0);
            if (!action_value || !mode_value || !modifier_value ||
                !isKnownToolMode(*mode_value) || !isKnownActionId(*action_value) ||
                *action_value <= static_cast<int>(Action::NONE) ||
                (*modifier_value & ~(MODIFIER_SHIFT | MODIFIER_CTRL | MODIFIER_ALT | MODIFIER_SUPER)))
                return std::nullopt;

            const auto action = static_cast<Action>(*action_value);
            const int modifiers = *modifier_value;
            const auto type = value["trigger_type"].get<std::string>();
            InputTrigger trigger;
            if (type == "key") {
                const auto key = value.contains("key") ? integerValue(value["key"])
                                                       : std::nullopt;
                if (!key || !isSupportedKeyCode(*key) ||
                    (value.contains("on_repeat") && !value["on_repeat"].is_boolean()))
                    return std::nullopt;
                trigger = KeyTrigger{*key, modifiers, value.value("on_repeat", false)};
            } else if (type == "mouse_button" || type == "drag") {
                const auto button = value.contains("button") ? integerValue(value["button"])
                                                             : std::nullopt;
                if (!button || *button < static_cast<int>(MouseButton::LEFT) ||
                    *button > static_cast<int>(MouseButton::FORWARD))
                    return std::nullopt;
                std::optional<int> chord;
                if (value.contains("chord_key")) {
                    const auto key = integerValue(value["chord_key"]);
                    if (!key || !isSupportedKeyCode(*key))
                        return std::nullopt;
                    chord = *key;
                }
                if (type == "mouse_button") {
                    if (value.contains("double_click") && !value["double_click"].is_boolean())
                        return std::nullopt;
                    trigger = MouseButtonTrigger{static_cast<MouseButton>(*button), modifiers,
                                                 value.value("double_click", false)};
                } else {
                    trigger = MouseDragTrigger{static_cast<MouseButton>(*button), modifiers, chord};
                }
            } else if (type == "scroll") {
                std::optional<int> chord;
                if (value.contains("chord_key")) {
                    const auto key = integerValue(value["chord_key"]);
                    if (!key || !isSupportedKeyCode(*key))
                        return std::nullopt;
                    chord = *key;
                }
                trigger = MouseScrollTrigger{modifiers, chord};
            } else {
                return std::nullopt;
            }
            if (!InputBindings::isValidTriggerForAction(action, trigger))
                return std::nullopt;
            return Binding{static_cast<ToolMode>(*mode_value), trigger, action,
                           value.value("description", getActionName(action))};
        }

        [[nodiscard]] std::string triggerIdentity(const Binding& binding) {
            auto value = detail::profileBindingJson(binding);
            value.erase("action");
            value.erase("description");
            value.erase("on_repeat");
            return bindingKey(std::move(value));
        }

        [[nodiscard]] std::string storedTriggerIdentity(const nlohmann::json& value) {
            auto identity = value;
            identity.erase("action");
            identity.erase("description");
            identity.erase("on_repeat");
            return bindingKey(std::move(identity));
        }

        [[nodiscard]] bool hasSemanticallyUsableBindingArray(const nlohmann::json& bindings) {
            if (!bindings.is_array())
                return false;
            std::unordered_map<std::string, int> actions_by_trigger;
            for (const auto& binding : bindings) {
                const auto parsed = parseStoredBinding(binding);
                if (!parsed)
                    return false;
                const auto [seen, inserted] = actions_by_trigger.try_emplace(
                    storedTriggerIdentity(binding), static_cast<int>(parsed->action));
                if (!inserted && seen->second != static_cast<int>(parsed->action))
                    return false;
            }
            return true;
        }

        struct HistoricalProfileMatch {
            std::set<std::string> keys, stale_keys, malformed_default_keys;
            std::set<std::pair<int, int>> stale_actions;
            std::unordered_map<int, Action> actions;
            nlohmann::json bindings = nlohmann::json::array();
        };

        [[nodiscard]] HistoricalProfileMatch matchHistoricalProfile(
            const int version, const bool override_format, const nlohmann::json& stored_bindings) {
            HistoricalProfileMatch match;
            auto& historical_keys = match.keys;
            auto& stale_default_keys = match.stale_keys;
            auto& stale_default_actions = match.stale_actions;
            auto& malformed_default_keys = match.malformed_default_keys;
            auto& historical_actions = match.actions;
            auto& historical_bindings = match.bindings;
            nlohmann::json historical_action_symbols = nlohmann::json::object();
            if (!override_format && version < PROFILE_VERSION && version >= 2) {
                size_t best_matches = 0;
                size_t best_difference = std::numeric_limits<size_t>::max();
                std::unordered_set<std::string> stored_keys;
                std::unordered_set<std::string> malformed_default_ids;
                std::unordered_map<std::string, std::string> stored_descriptions;
                for (const auto& b : stored_bindings) {
                    if (!b.is_object())
                        continue;
                    const auto key = bindingKey(b);
                    stored_keys.insert(key);
                    if (b.contains("description") && b["description"].is_string())
                        stored_descriptions.emplace(key, b.value("description", ""));
                    if (!parseStoredBinding(b)) {
                        const auto mode = b.contains("mode") ? integerValue(b["mode"])
                                                             : std::optional<int>(0);
                        if (!mode)
                            continue;
                        if (b.contains("action")) {
                            if (const auto action = integerValue(b["action"]))
                                malformed_default_ids.insert("a:" + std::to_string(*mode) + ":" +
                                                             std::to_string(*action));
                        }
                        if (b.contains("description") && b["description"].is_string()) {
                            malformed_default_ids.insert("d:" + std::to_string(*mode) + ":" +
                                                         toLowerCopy(b["description"].get<std::string>()));
                        }
                    }
                }
                const auto hasInvalidRecordFor = [&](const nlohmann::json& candidate) {
                    const auto mode = candidate.value("mode", 0);
                    const auto action = candidate.value("action", 0);
                    const auto description = toLowerCopy(
                        candidate.value("description", std::string{}));
                    return malformed_default_ids.contains("a:" + std::to_string(mode) + ":" +
                                                          std::to_string(action)) ||
                           malformed_default_ids.contains("d:" + std::to_string(mode) + ":" +
                                                          description);
                };
                const auto& historical = historicalKeymapDefaults();
                for (const auto& snapshot : historical["profiles"]) {
                    if (snapshot.at("version").get<int>() != version)
                        continue;
                    const auto& snapshot_data = historical["data"].at(
                        snapshot.at("data").get<size_t>());
                    const auto& candidate = snapshot_data["bindings"];
                    size_t matches = 0;
                    for (const auto& b : candidate)
                        matches += stored_keys.contains(bindingKey(b)) || hasInvalidRecordFor(b);
                    const size_t difference = candidate.size() + stored_keys.size() - 2 * matches;
                    if (matches > best_matches ||
                        (matches == best_matches && difference < best_difference)) {
                        best_matches = matches;
                        best_difference = difference;
                        historical_bindings = candidate;
                        historical_action_symbols = snapshot_data.value("actions", nlohmann::json::object());
                    }
                }
                // Sparse hand-written files predate complete saved profiles and
                // still need the explicit compatibility migrations below.
                if (historical_bindings.empty() ||
                    (stored_bindings.size() < 10 &&
                     best_matches * 2 < historical_bindings.size())) {
                    historical_bindings = nlohmann::json::array();
                } else {
                    for (const auto& old : historical_bindings) {
                        if (hasInvalidRecordFor(old)) {
                            stored_keys.insert(bindingKey(old));
                            malformed_default_keys.insert(bindingKey(old));
                        }
                    }
                    for (auto it = historical_action_symbols.begin();
                         it != historical_action_symbols.end(); ++it) {
                        if (const auto mapped = findActionBySymbol(it.value().get<std::string>()))
                            historical_actions.emplace(std::stoi(it.key()), *mapped);
                    }
                    const auto current_defaults = detail::defaultProfile().bindings;
                    for (const auto& b : historical_bindings) {
                        historical_keys.insert(bindingKey(b));
                        const auto old_mode = static_cast<ToolMode>(b.value("mode", 0));
                        const auto old_description = toLowerCopy(b.value("description", ""));
                        std::optional<Action> named_default;
                        for (const auto& def : current_defaults) {
                            if (def.mode != old_mode ||
                                toLowerCopy(def.description) != old_description)
                                continue;
                            if (named_default && *named_default != def.action) {
                                named_default.reset();
                                break;
                            }
                            named_default = def.action;
                        }
                        const int old_id = b.value("action", 0);
                        if (named_default)
                            historical_actions[old_id] = *named_default;
                        else if (const auto mapped = findActionBySymbol(b.value("action_symbol", "")))
                            historical_actions.try_emplace(old_id, *mapped);
                    }
                    for (const auto& snapshot : historicalKeymapDefaults()["profiles"]) {
                        if (snapshot.at("version").get<int>() >= version)
                            continue;
                        // Earlier loaders could resave old defaults with a newer version.
                        // Only discard residues with a shifted action ID or a mode
                        // binding now inherited globally; keep ambiguous rebinds.
                        const auto& prior_profile = historicalKeymapDefaults()["data"].at(
                            snapshot.at("data").get<size_t>());
                        for (const auto& old : prior_profile["bindings"]) {
                            const auto key = bindingKey(old);
                            if (historical_keys.contains(key) || !stored_keys.contains(key))
                                continue;
                            const auto action = findActionBySymbol(old.value("action_symbol", ""));
                            if (!action)
                                continue;
                            const auto old_mode = static_cast<ToolMode>(old.value("mode", 0));
                            const bool moved_to_global = old_mode != ToolMode::GLOBAL &&
                                                         std::ranges::any_of(current_defaults, [&](const Binding& def) {
                                                             return def.action == *action && def.mode == ToolMode::GLOBAL;
                                                         }) &&
                                                         !std::ranges::any_of(current_defaults, [&](const Binding& def) {
                                                             return def.action == *action && def.mode == old_mode;
                                                         });
                            const bool shifted_identity =
                                static_cast<int>(*action) != old.value("action", 0);
                            if (!moved_to_global && !shifted_identity)
                                continue;
                            const auto saved = stored_descriptions.find(key);
                            if (saved == stored_descriptions.end() ||
                                saved->second != old.value("description", ""))
                                continue;
                            stale_default_keys.insert(key);
                            stale_default_actions.insert({old.value("mode", 0),
                                                          static_cast<int>(*action)});
                        }
                    }
                }
            }
            return match;
        }

    } // namespace

    InputBindings::InputBindings() {
        const auto config_dir = getConfigDir();
        if (g_persistence_enabled.load(std::memory_order_acquire) &&
            config_dir) {
            std::string selected = "Default";
            std::ifstream marker;
            const auto marker_path = *config_dir / ".active-profile";
            if (lfs::core::native_fs::exists(marker_path) &&
                lfs::core::open_file_for_read(marker_path, marker)) {
                std::getline(marker, selected);
                if (!isValidProfileName(selected))
                    selected = "Default";
            }
            for (const auto& name : {selected, std::string("Default")}) {
                const auto saved_path = *config_dir / (name + ".json");
                if (!lfs::core::native_fs::is_symlink(saved_path) &&
                    lfs::core::native_fs::exists(saved_path) && loadProfileFromFile(saved_path)) {
                    current_profile_name_ = name;
                    return;
                }
            }
        }

        auto profile = detail::defaultProfile();
        current_profile_name_ = profile.name;
        bindings_ = std::move(profile.bindings);
        disabled_bindings_.clear();
        rebuildLookupMaps();
        persistActiveProfileName();
    }

    void InputBindings::loadProfile(const std::string& name) {
        if (!isValidProfileName(name)) {
            LOG_WARN("Refusing invalid input profile name");
            return;
        }
        if (!g_persistence_enabled.load(std::memory_order_acquire)) {
            auto profile = detail::defaultProfile();
            current_profile_name_ = profile.name;
            bindings_ = std::move(profile.bindings);
            disabled_bindings_.clear();
            rebuildLookupMaps();
            notifyBindingsChanged();
            return;
        }
        const auto path = profilePath(name);
        if (!path)
            return;
        if (lfs::core::native_fs::exists(*path) && loadProfileFromFile(*path)) {
            const bool corrected_name = current_profile_name_ != name;
            current_profile_name_ = name;
            persistActiveProfileName();
            if (corrected_name)
                notifyBindingsChanged();
            return;
        }

        if (name == "default" || name == "Default") {
            auto profile = detail::defaultProfile();
            current_profile_name_ = profile.name;
            bindings_ = std::move(profile.bindings);
            disabled_bindings_.clear();
            rebuildLookupMaps();
            notifyBindingsChanged();
            persistActiveProfileName();
        } else {
            LOG_WARN("Unknown profile '{}', using default", name);
            loadProfile("Default");
        }
    }

    bool InputBindings::saveProfile(const std::string& name) {
        const auto path = profilePath(name);
        if (!path)
            return false;
        if (!saveProfileToFile(*path, name))
            return false;
        current_profile_name_ = name;
        persistActiveProfileName();
        notifyBindingsChanged();
        return true;
    }

    bool InputBindings::saveProfile(const std::string& name) const {
        const auto path = profilePath(name);
        return path && saveProfileToFile(*path, name);
    }

    std::optional<std::filesystem::path> InputBindings::profilePath(
        const std::string_view name) const {
        if (!isValidProfileName(name)) {
            LOG_WARN("Refusing invalid input profile name");
            return std::nullopt;
        }
        if (!g_persistence_enabled.load(std::memory_order_acquire))
            return std::nullopt;
        const auto config_dir = getConfigDir();
        if (!config_dir) {
            LOG_WARN("Input profile directory is unavailable");
            return std::nullopt;
        }
        auto path = *config_dir / (std::string(name) + ".json");
        if (lfs::core::native_fs::is_symlink(path)) {
            LOG_WARN("Refusing symbolic link input profile '{}': file is outside the keymap directory", name);
            return std::nullopt;
        }
        return path;
    }

    void InputBindings::persistActiveProfileName() const {
        if (!g_persistence_enabled.load(std::memory_order_acquire) ||
            !isValidProfileName(current_profile_name_))
            return;
        const auto config_dir = getConfigDir();
        if (!config_dir)
            return;
        const auto written = lfs::core::writeTextFileAtomically(
            *config_dir / ".active-profile", current_profile_name_ + "\n");
        if (!written)
            LOG_WARN("Could not remember active input profile '{}': {}",
                     current_profile_name_, lfs::format_for_developer(written.error()));
    }

    bool InputBindings::isValidProfileName(const std::string_view name) noexcept {
        if (name.empty() || name.size() > 128 || name == "." ||
            name.find("..") != std::string_view::npos ||
            name.find(" .") != std::string_view::npos ||
            name.front() == ' ' || name.back() == ' ' || name.back() == '.')
            return false;
        for (const unsigned char c : name) {
            if (c < 0x20 || c == 0x7f || c == '/' || c == '\\' || c == '<' ||
                c == '>' || c == ':' || c == '"' || c == '|' || c == '?' || c == '*')
                return false;
        }
        const auto stem = name.substr(0, name.find('.'));
        if (stem.size() >= 8)
            return true;
        std::string device(stem);
        std::ranges::transform(device, device.begin(), [](const unsigned char c) {
            return static_cast<char>(std::toupper(c));
        });
        static constexpr std::array<std::string_view, 33> reserved = {
            "CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$", "CLOCK$",
            "COM0", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
            "LPT0", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
            "COM¹", "COM²", "COM³", "LPT¹", "LPT²", "LPT³"};
        return std::ranges::find(reserved, device) == reserved.end();
    }

    bool InputBindings::isValidTriggerForAction(const Action action,
                                                const InputTrigger& trigger) noexcept {
        const int modifiers = std::visit([](const auto& value) { return value.modifiers; }, trigger);
        if ((modifiers & ~MODIFIER_MASK) != 0)
            return false;
        const auto allowed = describe(action).allowed_kinds;
        if (const auto* key = std::get_if<KeyTrigger>(&trigger)) {
            if (!(allowed & TRIGGER_KIND_KEY) || key->key == KEY_UNKNOWN)
                return false;
            switch (action) {
            case Action::CAMERA_MOVE_FORWARD:
            case Action::CAMERA_MOVE_BACKWARD:
            case Action::CAMERA_MOVE_LEFT:
            case Action::CAMERA_MOVE_RIGHT:
            case Action::CAMERA_MOVE_UP:
            case Action::CAMERA_MOVE_DOWN:
                return (modifiers & (MODIFIER_CTRL | MODIFIER_ALT | MODIFIER_SUPER)) == 0;
            default: return true;
            }
        }
        if (std::holds_alternative<MouseButtonTrigger>(trigger))
            return (allowed & TRIGGER_KIND_MOUSE_BUTTON) != 0;
        if (std::holds_alternative<MouseScrollTrigger>(trigger))
            return (allowed & TRIGGER_KIND_MOUSE_SCROLL) != 0;
        return (allowed & TRIGGER_KIND_MOUSE_DRAG) != 0;
    }

    std::optional<std::filesystem::path> InputBindings::getConfigDir() {
        const auto paths = lfs::core::UserPaths::resolve();
        if (paths)
            return paths->keymapDir();
        LOG_WARN("Unable to resolve input profile path: {}; persistence is disabled",
                 lfs::format_for_developer(paths.error()));
        return std::nullopt;
    }

    bool InputBindings::isPersistenceEnabled() noexcept {
        return g_persistence_enabled.load(std::memory_order_acquire);
    }

    void InputBindings::setPersistenceEnabled(const bool enabled) noexcept {
        g_persistence_enabled.store(enabled, std::memory_order_release);
    }

    bool InputBindings::saveProfileToFile(const std::filesystem::path& path) const {
        return saveProfileToFile(path, current_profile_name_);
    }

    bool InputBindings::saveProfileToFile(const std::filesystem::path& path,
                                          const std::string_view profile_name) const {
        using json = nlohmann::json;

        std::error_code file_error;
        if (lfs::core::native_fs::exists(path, file_error)) {
            if (lfs::core::native_fs::is_symlink(path, file_error) || file_error) {
                LOG_WARN("Refusing to overwrite symbolic link input profile '{}': file is unchanged",
                         lfs::core::path_to_utf8(path));
                return false;
            }
            std::ifstream existing;
            if (!lfs::core::open_file_for_read(path, existing)) {
                LOG_WARN("Cannot read existing input profile '{}'; file is unchanged",
                         lfs::core::path_to_utf8(path));
                return false;
            }
            const json stored = json::parse(existing, nullptr, false);
            if (stored.is_object() &&
                readStoredProfileVersion(stored).kind == StoredProfileVersionKind::Newer) {
                LOG_WARN("Refusing to overwrite newer input profile '{}'; file is unchanged",
                         lfs::core::path_to_utf8(path));
                notifyNewerProfile();
                return false;
            }
            if (!hasValidStoredBindings(stored)) {
                const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
                auto backup = path;
                backup += ".damaged-" + std::to_string(stamp) + ".bak";
                const bool backed_up = lfs::core::native_fs::copy_file(
                    path, backup, std::filesystem::copy_options::none, file_error);
                if (!backed_up || file_error) {
                    LOG_ERROR("Could not preserve damaged input profile '{}'; file is unchanged",
                              lfs::core::path_to_utf8(path));
                    return false;
                }
                LOG_WARN("Damaged input profile preserved at '{}' before saving",
                         lfs::core::path_to_utf8(backup));
                notifyDamagedProfileBackedUp();
            }
        }

        json j;
        j["name"] = std::string(profile_name);
        j["version"] = PROFILE_VERSION;

        json bindings_array = nlohmann::json::array();
        for (const auto& binding : bindings_)
            bindings_array.push_back(detail::profileBindingJson(binding));
        for (const auto& entry : preserved_profile_entries_)
            bindings_array.push_back(entry);
        // Older builds read this full list and ignore the override record below.
        j["bindings"] = bindings_array;

        const auto defaults = detail::defaultProfile().bindings;
        json overrides = {{"bindings", nlohmann::json::array()},
                          {"removed", nlohmann::json::array()},
                          {"disabled", nlohmann::json::array()}};
        for (const auto& def : defaults) {
            if (std::ranges::any_of(bindings_, [&](const Binding& b) {
                    return detail::bindingContentKey(b) == detail::bindingContentKey(def);
                }))
                continue;
            overrides["removed"].push_back(detail::profileBindingJson(def));
        }
        for (size_t i = 0; i < bindings_.size(); ++i) {
            const auto& b = bindings_[i];
            if (!std::ranges::any_of(defaults, [&](const Binding& def) {
                    return detail::bindingContentKey(b) == detail::bindingContentKey(def);
                }))
                overrides["bindings"].push_back(bindings_array[i]);
        }
        for (const auto& [mode, action] : disabled_bindings_) {
            if (std::ranges::any_of(bindings_, [&](const Binding& b) {
                    return b.mode == mode && b.action == action;
                }))
                continue;
            overrides["disabled"].push_back({{"mode", static_cast<int>(mode)},
                                             {"action", static_cast<int>(action)},
                                             {"description", getActionName(action)}});
        }
        for (const auto& entry : preserved_profile_entries_)
            overrides["bindings"].push_back(entry);
        j["overrides"] = std::move(overrides);

        try {
            const std::string serialized = j.dump(4) + '\n';
            const auto written = lfs::core::writeTextFileAtomically(path, serialized);
            if (!written) {
                LOG_ERROR("Failed to save profile '{}': {}",
                          lfs::core::path_to_utf8(path),
                          lfs::format_for_developer(written.error()));
                return false;
            }
            return true;
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to save profile: {}", e.what());
            return false;
        }
    }

    bool InputBindings::loadProfileFromFile(const std::filesystem::path& path) {
        using json = nlohmann::json;
        InputBindings previous = *this;

        try {
            // Parse in a nested scope so the source handle is closed before
            // any migration save. Windows cannot replace a file that this
            // stream still has open.
            json j;
            {
                std::ifstream file;
                if (!lfs::core::open_file_for_read(path, file)) {
                    LOG_ERROR("Failed to open profile file: {}", lfs::core::path_to_utf8(path));
                    return false;
                }
                j = json::parse(file);
            }
            const auto stored_version = readStoredProfileVersion(j);
            if (stored_version.kind == StoredProfileVersionKind::Newer) {
                LOG_WARN("Cannot load newer input profile version {} from '{}'; file is unchanged",
                         stored_version.value, lfs::core::path_to_utf8(path));
                notifyNewerProfile();
                return false;
            }
            if (stored_version.kind == StoredProfileVersionKind::Invalid) {
                LOG_WARN("Cannot load input profile with an invalid version from '{}'; file is unchanged",
                         lfs::core::path_to_utf8(path));
                return false;
            }
            const int version = static_cast<int>(stored_version.value);
            const std::string profile_name = j.value("name", "Custom");
            const bool override_format = version == PROFILE_VERSION &&
                                         j.contains("overrides") && j["overrides"].is_object() &&
                                         j["overrides"].contains("bindings") &&
                                         j["overrides"].contains("removed");
            const json& stored_bindings = override_format ? j["overrides"]["bindings"] : j["bindings"];
            if (!stored_bindings.is_array())
                return false;
            auto historical = matchHistoricalProfile(version, override_format, stored_bindings);
            const auto& historical_keys = historical.keys;
            const auto& stale_default_keys = historical.stale_keys;
            const auto& stale_default_actions = historical.stale_actions;
            const auto& historical_actions = historical.actions;
            const auto& historical_bindings = historical.bindings;

            if (version < 1) {
                LOG_WARN("Unknown profile version: {}", version);
            }

            current_profile_name_ = profile_name;
            bindings_.clear();
            preserved_profile_entries_.clear();
            disabled_bindings_.clear();
            size_t rewritten = 0;
            bool legacy_window_profile = false;
            std::vector<Binding> user_additions;
            std::vector<size_t> skipped_entry_indices;
            // File order is the tie-breaker: keep the first binding for each
            // mode and trigger, and report later conflicting actions.
            std::unordered_map<std::string, Action> seen_trigger_actions;
            bool conversion_error_logged = false;

            for (size_t entry_index = 0; entry_index < stored_bindings.size(); ++entry_index) {
                const auto& b = stored_bindings[entry_index];
                const auto skip_entry = [&] {
                    skipped_entry_indices.push_back(entry_index + 1);
                    preserved_profile_entries_.push_back(b);
                    ++rewritten;
                };
                try {
                    if (!b.is_object()) {
                        skip_entry();
                        continue;
                    }
                    const auto stored_key = bindingKey(b);
                    // Inherited old defaults are supplied by the current profile.
                    // Some used trigger kinds that an action no longer accepts, so
                    // do not validate or dispatch them as user overrides.
                    if (!historical_bindings.empty() &&
                        (historical_keys.contains(stored_key) ||
                         stale_default_keys.contains(stored_key)))
                        continue;
                    const bool user_added = !historical_bindings.empty() &&
                                            !historical_keys.contains(stored_key) &&
                                            !stale_default_keys.contains(stored_key);
                    if (!b.contains("action") || !integerValue(b["action"]) ||
                        (b.contains("mode") && !integerValue(b["mode"])) ||
                        !b.contains("trigger_type") || !b["trigger_type"].is_string() ||
                        (b.contains("description") && !b["description"].is_string())) {
                        skip_entry();
                        continue;
                    }
                    const int mode_value = b.contains("mode") ? *integerValue(b["mode"]) : 0;
                    if (mode_value == REMOVED_TOOL_MODE_2) {
                        ++rewritten;
                        continue;
                    }
                    if (!isKnownToolMode(mode_value)) {
                        skip_entry();
                        continue;
                    }
                    int action_value = *integerValue(b["action"]);
                    const bool has_historical_action = historical_actions.contains(action_value);
                    if (const auto mapped = historical_actions.find(action_value);
                        mapped != historical_actions.end())
                        action_value = static_cast<int>(mapped->second);
                    const std::string stored_description = b.value("description", "");
                    if (stored_description == "Window size" || stored_description == "Window drag" ||
                        stored_description == "Adjust Window Size" || stored_description == "Drag Depth Window" ||
                        (version == 28 && (action_value == 85 || action_value == 86))) {
                        legacy_window_profile = true;
                        ++rewritten;
                        continue;
                    }
                    // Version 28 inserted two window controls before grouping. They
                    // are unavailable here; their IDs must not invoke grouping.
                    if (version == 28) {
                        if (action_value == 87) {
                            action_value = static_cast<int>(Action::GROUP_SELECTED_SCENE_NODES);
                            ++rewritten;
                        } else if (action_value == 88) {
                            action_value = static_cast<int>(Action::UNGROUP_SELECTED_SCENE_NODE);
                            ++rewritten;
                        }
                    }
                    const bool transient_scene_graph_binding =
                        version == 24 &&
                        (stored_description == "Select Scene Hierarchy" ||
                         stored_description == "Select All Scene Nodes" ||
                         stored_description == "Toggle Scene Cursor Selection" ||
                         stored_description == "Toggle Scene Selection Visibility" ||
                         stored_description == "Toggle Scene Selection Training");
                    if (transient_scene_graph_binding) {
                        LOG_INFO("Replacing transient version 24 Scene Graph binding: '{}'",
                                 stored_description);
                        ++rewritten;
                        continue;
                    }
                    if (action_value == REMOVED_ACTION_39 ||
                        action_value == REMOVED_ACTION_66) {
                        ++rewritten;
                        continue;
                    }

                    Binding binding;
                    // Version 1 had no mode field, default to GLOBAL
                    binding.mode = static_cast<ToolMode>(mode_value);
                    binding.action = static_cast<Action>(action_value);
                    binding.description = b.value("description", getActionName(binding.action));

                    // Cross-version safeguard: if the stored description doesn't
                    // match the current name for that integer action, the enum was
                    // reshuffled between profile saves — re-resolve by description
                    // so the binding still drives the intended action.
                    if (b.contains("description") && !has_historical_action) {
                        const auto stored_desc = b["description"].get<std::string>();
                        const auto current_name = getActionName(binding.action);
                        if (toLowerCopy(stored_desc) != toLowerCopy(current_name)) {
                            if (const auto remapped = findActionByDescription(stored_desc)) {
                                binding.action = *remapped;
                                binding.description = getActionName(*remapped);
                                ++rewritten;
                            }
                        }
                    }
                    if (binding.action == Action::NONE ||
                        getActionName(binding.action) == "Unknown") {
                        const auto remapped = version < PROFILE_VERSION
                                                  ? findActionByDescription(stored_description)
                                                  : std::nullopt;
                        if (!remapped) {
                            skip_entry();
                            continue;
                        }
                        binding.action = *remapped;
                        binding.description = getActionName(*remapped);
                    }

                    json adjusted = b;
                    adjusted["mode"] = mode_value;
                    adjusted["action"] = static_cast<int>(binding.action);
                    adjusted["description"] = binding.description;
                    auto parsed = parseStoredBinding(adjusted);
                    if (!parsed) {
                        skip_entry();
                        continue;
                    }
                    binding = normalizeLoadedBinding(std::move(*parsed));
                    if (!isValidTriggerForAction(binding.action, binding.trigger)) {
                        skip_entry();
                        continue;
                    }
                    const auto [seen, inserted] = seen_trigger_actions.try_emplace(
                        triggerIdentity(binding), binding.action);
                    if (!inserted) {
                        if (seen->second != binding.action)
                            skip_entry();
                        continue;
                    }
                    if (user_added)
                        user_additions.push_back(binding);

                    bindings_.push_back(binding);
                } catch (const std::exception&) {
                    if (!conversion_error_logged) {
                        LOG_WARN("Skipping one or more input profile entries that could not be converted");
                        conversion_error_logged = true;
                    }
                    skip_entry();
                }
            }

            if (override_format || !historical_bindings.empty()) {
                if (override_format)
                    user_additions = std::move(bindings_);
                bindings_ = detail::defaultProfile().bindings;
                std::set<std::string> removed_keys;
                if (override_format) {
                    for (const auto& removed : j["overrides"]["removed"]) {
                        if (removed.contains("trigger_type")) {
                            removed_keys.insert(bindingKey(removed));
                        } else if (removed.contains("slot")) {
                            // Read v31 ordinal tombstones written before stable
                            // content identities were introduced.
                            const auto mode = static_cast<ToolMode>(removed.at("mode").get<int>());
                            const auto action = static_cast<Action>(removed.at("action").get<int>());
                            int slot = 0;
                            for (const auto& def : bindings_) {
                                if (def.mode != mode || def.action != action)
                                    continue;
                                if (slot++ == removed.at("slot").get<int>()) {
                                    removed_keys.insert(bindingKey(detail::profileBindingJson(def)));
                                    break;
                                }
                            }
                        }
                    }
                    if (j["overrides"].contains("disabled")) {
                        for (const auto& disabled : j["overrides"]["disabled"]) {
                            disabled_bindings_.insert(
                                {static_cast<ToolMode>(disabled.at("mode").get<int>()),
                                 static_cast<Action>(disabled.at("action").get<int>())});
                        }
                    }
                } else {
                    std::unordered_set<std::string> stored_keys;
                    for (const auto& b : stored_bindings)
                        stored_keys.insert(bindingKey(b));
                    stored_keys.insert(historical.malformed_default_keys.begin(),
                                       historical.malformed_default_keys.end());
                    for (const auto& old : historical_bindings) {
                        const int old_mode = old.value("mode", 0);
                        const int old_action = old.value("action", 0);
                        if (stored_keys.contains(bindingKey(old)))
                            continue;
                        Action action = static_cast<Action>(old_action);
                        if (const auto mapped = historical_actions.find(old_action);
                            mapped != historical_actions.end())
                            action = mapped->second;
                        else if (const auto named = findActionBySymbol(old.value("action_symbol", "")))
                            action = *named;
                        else if (const auto named = findActionByDescription(old.value("description", "")))
                            action = *named;
                        if (stale_default_actions.contains({old_mode, static_cast<int>(action)}))
                            continue;
                        Binding normalized{static_cast<ToolMode>(old_mode), KeyTrigger{KEY_UNKNOWN},
                                           action, getActionName(action)};
                        normalized = normalizeLoadedBinding(std::move(normalized));
                        auto removed_identity = old;
                        removed_identity["mode"] = static_cast<int>(normalized.mode);
                        removed_identity["action"] = static_cast<int>(normalized.action);
                        removed_keys.insert(bindingKey(std::move(removed_identity)));
                        if (normalized.mode != ToolMode::GLOBAL &&
                            std::ranges::any_of(bindings_, [&](const Binding& def) {
                                return def.mode == ToolMode::GLOBAL &&
                                       def.action == normalized.action &&
                                       describe(def.action).inherits_from_global;
                            }) &&
                            !std::ranges::any_of(bindings_, [&](const Binding& def) {
                                return def.mode == normalized.mode &&
                                       def.action == normalized.action;
                            }))
                            disabled_bindings_.insert({normalized.mode, normalized.action});
                    }
                }
                detail::applyRemovedBindingOverrides(bindings_, removed_keys);
                std::erase_if(bindings_, [&](const Binding& b) {
                    return disabled_bindings_.contains({b.mode, b.action});
                });
                for (const auto& added : user_additions) {
                    std::erase_if(bindings_, [&](const Binding& b) {
                        return b.mode == added.mode && triggersOverlap(b.trigger, added.trigger);
                    });
                    bindings_.push_back(added);
                }
                rewritten += 1;
            } else {
                if (const size_t added_bindings = projectLegacyGlobalBindings(bindings_, version);
                    added_bindings > 0) {
                    LOG_INFO("Projected {} legacy global bindings into mode-specific shortcuts for profile '{}'",
                             added_bindings, current_profile_name_);
                }

                rewritten += collapseRedundantModeBindings(version);
                rewritten += migrateLoadedProfile(legacy_window_profile && version == 27 ? 28 : version);
            }

            rebuildLookupMaps();
            warnAboutUnreachableCoreActions();
            if (!skipped_entry_indices.empty())
                notifyInvalidProfileEntries(skipped_entry_indices);

            // Auto-persist the canonical Default profile so disk stays current
            // after a versioned migration. User-imported files are left untouched;
            // the migration still applies in memory.
            if (rewritten > 0 && version >= 1 && version < PROFILE_VERSION) {
                std::error_code ec;
                const auto config_dir = getConfigDir();
                const auto config_default = config_dir
                                                ? std::optional<std::filesystem::path>(*config_dir / "Default.json")
                                                : std::nullopt;
                if (g_persistence_enabled.load(std::memory_order_acquire) &&
                    config_default && lfs::core::native_fs::equivalent(path, *config_default, ec)) {
                    saveProfileToFile(*config_default);
                }
            }
            notifyBindingsChanged();
            return true;
        } catch (const std::exception& e) {
            *this = std::move(previous);
            LOG_ERROR("Failed to load profile: {}", e.what());
            return false;
        }
    }

    size_t InputBindings::migrateLoadedProfile(const int version) {
        if (version >= PROFILE_VERSION) {
            return 0;
        }

        const Profile defaults = detail::defaultProfile();
        size_t added = 0;
        for (const auto& def : defaults.bindings) {
            // Version 12 adds Shift+scroll as a *parallel* trigger for
            // BRUSH_RESIZE — the existing Ctrl+scroll binding stays, so the
            // usual "skip if the action is already mapped" guard doesn't
            // apply here and we only need to ensure the Shift+scroll trigger
            // itself is free.
            const bool brush_resize_shift_scroll =
                def.action == Action::BRUSH_RESIZE &&
                std::holds_alternative<MouseScrollTrigger>(def.trigger) &&
                std::get<MouseScrollTrigger>(def.trigger).modifiers == MODIFIER_SHIFT;
            const auto* key_trigger = std::get_if<KeyTrigger>(&def.trigger);
            const bool crop_apply_num_enter =
                def.action == Action::APPLY_CROP_BOX &&
                key_trigger &&
                key_trigger->key == KEY_KP_ENTER;
            const bool selection_volume_shortcut =
                def.action == Action::SELECT_MODE_BOX ||
                def.action == Action::SELECT_MODE_SPHERE;
            const bool should_add =
                (version < 37 && def.action == Action::EXIT_WALK) ||
                (version < 6 && def.action == Action::CAMERA_ROLL) ||
                (version < 7 && def.action == Action::BRUSH_RESIZE && !brush_resize_shift_scroll) ||
                (version < 9 && def.action == Action::CONFIRM_POLYGON) ||
                (version < 10 && def.action == Action::UNDO_POLYGON_VERTEX) ||
                (version < 12 && brush_resize_shift_scroll) ||
                (version < 13 && def.action == Action::CAMERA_SET_HOME) ||
                (version < 14 && def.action == Action::HISTOGRAM_ZOOM_MARKED) ||
                (version < 15 && def.action == Action::APPLY_CROP_BOX) ||
                (version < 16 && def.action == Action::TOGGLE_CAMERA_FRUSTUMS) ||
                (version < 17 && def.action == Action::SELECTION_INTERSECT) ||
                (version < 18 && selection_volume_shortcut) ||
                (version < 19 && def.action == Action::CUT_SELECTION) ||
                (version < 20 && def.action == Action::TOGGLE_PERFORMANCE_HUD) ||
                (version < 21 && def.action == Action::OPEN_PREFERENCES) ||
                (version < 22 && def.action == Action::TOGGLE_MCP_SERVER) ||
                (version < 22 && def.action == Action::TOGGLE_MCP_BINDING) ||
                (version < 23 && def.action == Action::TOGGLE_GRID) ||
                (version < 25 &&
                 (def.action == Action::SELECT_ALL_SCENE_NODES ||
                  def.action == Action::TOGGLE_SCENE_SELECTION_VISIBILITY ||
                  def.action == Action::TOGGLE_SCENE_SELECTION_TRAINING)) ||
                (version < 26 &&
                 (def.action == Action::GROUP_SELECTED_SCENE_NODES ||
                  def.action == Action::UNGROUP_SELECTED_SCENE_NODE)) ||
                ((version < 27 || version == 28) &&
                 (def.action == Action::ASSET_GALLERY_PRIMARY ||
                  def.action == Action::ASSET_GALLERY_COPY_LINK ||
                  def.action == Action::ASSET_REFRESH));
            if (!should_add) {
                continue;
            }
            if (!brush_resize_shift_scroll && !crop_apply_num_enter) {
                const bool action_already_bound = std::ranges::any_of(
                    bindings_, [&](const Binding& current) {
                        return current.mode == def.mode && current.action == def.action;
                    });
                if (action_already_bound) {
                    continue;
                }
            }
            const bool trigger_in_use = std::ranges::any_of(
                bindings_, [&](const Binding& current) {
                    return current.mode == def.mode &&
                           triggersOverlap(current.trigger, def.trigger);
                });
            if (trigger_in_use) {
                continue;
            }
            LOG_INFO("Migrating profile '{}' with default binding: action={} mode={}",
                     current_profile_name_,
                     getActionName(def.action), static_cast<int>(def.mode));
            bindings_.push_back(def);
            ++added;
        }
        return added;
    }

    size_t InputBindings::collapseRedundantModeBindings(const int version) {
        if (version >= 8) {
            return 0;
        }

        const auto collect_global_triggers = [](const std::vector<Binding>& bindings) {
            std::map<Action, InputTrigger> result;
            for (const auto& binding : bindings) {
                if (binding.mode == ToolMode::GLOBAL) {
                    result[binding.action] = binding.trigger;
                }
            }
            return result;
        };
        const Profile defaults = detail::defaultProfile();
        const auto current_globals = collect_global_triggers(bindings_);
        const auto default_globals = collect_global_triggers(defaults.bindings);
        std::map<std::pair<ToolMode, Action>, InputTrigger> default_local_bindings;
        std::map<Action, InputTrigger> default_action_triggers;
        for (const auto& binding : defaults.bindings) {
            default_local_bindings[{binding.mode, binding.action}] = binding.trigger;
            default_action_triggers.try_emplace(binding.action, binding.trigger);
        }

        size_t collapsed = 0;
        std::erase_if(bindings_, [&](const Binding& binding) {
            if (binding.mode == ToolMode::GLOBAL) {
                if (default_local_bindings.find({binding.mode, binding.action}) != default_local_bindings.end()) {
                    return false;
                }
                const auto default_action = default_action_triggers.find(binding.action);
                if (default_action == default_action_triggers.end() ||
                    !triggersOverlap(binding.trigger, default_action->second)) {
                    return false;
                }
                ++collapsed;
                return true;
            }

            if (default_local_bindings.find({binding.mode, binding.action}) != default_local_bindings.end()) {
                return false;
            }

            const auto current_global = current_globals.find(binding.action);
            const auto default_global = default_globals.find(binding.action);
            const auto default_action = default_action_triggers.find(binding.action);
            const bool duplicates_current_global =
                describe(binding.action).inherits_from_global &&
                current_global != current_globals.end() &&
                triggersOverlap(binding.trigger, current_global->second);
            const bool duplicates_default_global =
                describe(binding.action).inherits_from_global &&
                default_global != default_globals.end() &&
                triggersOverlap(binding.trigger, default_global->second);
            const bool duplicates_default_action =
                default_action != default_action_triggers.end() &&
                triggersOverlap(binding.trigger, default_action->second);
            if (!duplicates_current_global &&
                !duplicates_default_global &&
                !duplicates_default_action) {
                return false;
            }

            ++collapsed;
            return true;
        });

        if (collapsed > 0) {
            LOG_INFO("Collapsed {} redundant mode bindings for profile '{}'",
                     collapsed, current_profile_name_);
        }
        return collapsed;
    }

    std::vector<std::string> InputBindings::getAvailableProfiles() const {
        std::vector<std::string> profiles = {"Default"};

        if (!g_persistence_enabled.load(std::memory_order_acquire))
            return profiles;

        const auto config_dir = getConfigDir();
        if (config_dir && lfs::core::native_fs::exists(*config_dir)) {
            for (const auto& entry : lfs::core::native_fs::directory_iterator(*config_dir)) {
                if (entry.path().extension() == ".json") {
                    const std::string name = lfs::core::path_to_utf8(entry.path().stem());
                    if (name != "Default" && isValidProfileName(name)) {
                        profiles.push_back(name);
                    }
                }
            }
        }

        return profiles;
    }

    Action InputBindings::getActionForKey(ToolMode mode, int key, int modifiers, bool* const on_repeat) const {
        if (on_repeat)
            *on_repeat = false;
        // AltGr is event-only and never stored on a shortcut. Masking it off would
        // match the bare digit or letter (German AltGr+4 is U+00BC, still key 4).
        if ((modifiers & KEYMOD_ALTGR) != 0)
            return Action::NONE;

        const int mods = modifiers & MODIFIER_MASK;
        const auto find_in_mode = [&](ToolMode query_mode, bool allow_extra) -> Action {
            const auto matched = [&](const KeyMapEntry& entry) {
                if (on_repeat)
                    *on_repeat = entry.on_repeat;
                return entry.action;
            };
            if (auto it = key_map_.find({query_mode, key, mods}); it != key_map_.end()) {
                return matched(it->second);
            }
            if (allow_extra && mods != MODIFIER_NONE) {
                if (auto it = key_map_.find({query_mode, key, MODIFIER_NONE});
                    it != key_map_.end() &&
                    describe(it->second.action).allows_extra_modifiers) {
                    return matched(it->second);
                }
            }
            return Action::NONE;
        };

        if (const auto local_action = find_in_mode(mode, false); local_action != Action::NONE) {
            return local_action;
        }

        if (mode != ToolMode::GLOBAL) {
            const auto global_action = find_in_mode(ToolMode::GLOBAL, false);
            if (global_action != Action::NONE &&
                describe(global_action).inherits_from_global &&
                !isDisabled(mode, global_action)) {
                return global_action;
            }
        }
        if (const auto local_action = find_in_mode(mode, true); local_action != Action::NONE)
            return local_action;
        if (mode != ToolMode::GLOBAL) {
            const auto global_action = find_in_mode(ToolMode::GLOBAL, true);
            if (global_action != Action::NONE &&
                describe(global_action).inherits_from_global &&
                !isDisabled(mode, global_action))
                return global_action;
        }

        // Support the common redo alias when the profile still uses the default Ctrl+Y binding.
        const auto redo_trigger = [&]() -> std::optional<InputTrigger> {
            if (auto trigger = getTriggerForAction(Action::REDO, mode)) {
                return trigger;
            }
            if (mode != ToolMode::GLOBAL && !isDisabled(mode, Action::REDO)) {
                return getTriggerForAction(Action::REDO, ToolMode::GLOBAL);
            }
            return std::nullopt;
        }();
        if (key == KEY_Z &&
            mods == (MODIFIER_CTRL | MODIFIER_SHIFT) &&
            triggerUsesDefaultRedoBinding(redo_trigger)) {
            return Action::REDO;
        }

        return Action::NONE;
    }

    Action InputBindings::getActionForShortcut(const ToolMode mode, const int physical_key,
                                               const int logical_key, const int modifiers,
                                               const int printed_punctuation,
                                               const bool shift_needed_for_printed_key,
                                               bool* const on_repeat) const {
        // Try the exact printed chord before a number-row position. When Shift
        // changes the produced symbol, retry without that layout-only Shift if
        // no exact printed chord is bound (for example, Ctrl+4 on AZERTY).
        if (printed_punctuation != KEY_UNKNOWN) {
            if (const auto action = getActionForKey(mode, printed_punctuation, modifiers, on_repeat);
                action != Action::NONE)
                return action;
            if (shift_needed_for_printed_key) {
                if (const auto action = getActionForKey(mode, printed_punctuation, modifiers & ~MODIFIER_SHIFT, on_repeat);
                    action != Action::NONE)
                    return action;
            }
        }
        // A non-number-row position may print a digit on another layout.
        // Digit shortcuts reserve the physical top row, so that glyph must
        // not hide a punctuation binding at this position.
        if (physical_key != KEY_UNKNOWN && physical_key != logical_key &&
            logical_key >= KEY_0 && logical_key <= KEY_9 &&
            !(physical_key >= KEY_0 && physical_key <= KEY_9)) {
            if (const auto action = getActionForKey(mode, physical_key, modifiers, on_repeat);
                action != Action::NONE)
                return action;
        }
        if (logical_key != KEY_UNKNOWN) {
            if (const auto action = getActionForKey(mode, logical_key, modifiers, on_repeat);
                action != Action::NONE)
                return action;
        }
        if (physical_key != KEY_UNKNOWN && physical_key != logical_key) {
            if (const auto action = getActionForKey(mode, physical_key, modifiers, on_repeat);
                action != Action::NONE)
                return action;
        }
        return Action::NONE;
    }

    Action InputBindings::getActionForMouseButton(ToolMode mode, MouseButton button, int modifiers, bool is_double_click) const {
        const int mods = modifiers & MODIFIER_MASK;
        const auto find_in_mode = [&](ToolMode query_mode, const bool double_click) -> Action {
            if (auto it = mouse_button_map_.find({query_mode, button, mods, double_click}); it != mouse_button_map_.end()) {
                return it->second;
            }
            if (mods != MODIFIER_NONE) {
                if (auto it = mouse_button_map_.find({query_mode, button, MODIFIER_NONE, double_click});
                    it != mouse_button_map_.end() &&
                    describe(it->second).allows_extra_modifiers) {
                    return it->second;
                }
            }
            return Action::NONE;
        };

        if (const auto local_action = find_in_mode(mode, is_double_click); local_action != Action::NONE) {
            return local_action;
        }
        if (mode != ToolMode::GLOBAL) {
            const auto global_action = find_in_mode(ToolMode::GLOBAL, is_double_click);
            if (global_action != Action::NONE &&
                describe(global_action).inherits_from_global &&
                !isDisabled(mode, global_action)) {
                return global_action;
            }
        }

        // A double-click is its own gesture. Only fall back to single-click
        // bindings after exact local and inherited global double-click bindings
        // have both been considered. This keeps global viewport gestures such as
        // Set Pivot available in every tool mode, even when that mode binds the
        // same button to a single-click action.
        if (is_double_click) {
            if (const auto local_action = find_in_mode(mode, false); local_action != Action::NONE) {
                return local_action;
            }
            if (mode != ToolMode::GLOBAL) {
                const auto global_action = find_in_mode(ToolMode::GLOBAL, false);
                if (global_action != Action::NONE &&
                    describe(global_action).inherits_from_global &&
                    !isDisabled(mode, global_action)) {
                    return global_action;
                }
            }
        }
        return Action::NONE;
    }

    Action InputBindings::getActionForScroll(ToolMode mode, int modifiers,
                                             const std::vector<int>& held_keys) const {
        const int mods = modifiers & MODIFIER_MASK;
        const auto find_in_mode = [&](ToolMode query_mode) -> Action {
            for (auto it_key = held_keys.rbegin(); it_key != held_keys.rend(); ++it_key) {
                const int chord = *it_key;
                if (auto it = scroll_chord_map_.find({query_mode, mods, chord}); it != scroll_chord_map_.end()) {
                    return it->second;
                }
            }
            if (auto it = scroll_map_.find({query_mode, mods}); it != scroll_map_.end()) {
                return it->second;
            }
            return Action::NONE;
        };

        if (const auto local_action = find_in_mode(mode); local_action != Action::NONE) {
            return local_action;
        }
        if (mode != ToolMode::GLOBAL) {
            const auto global_action = find_in_mode(ToolMode::GLOBAL);
            if (global_action != Action::NONE &&
                describe(global_action).inherits_from_global &&
                !isDisabled(mode, global_action)) {
                return global_action;
            }
        }
        return Action::NONE;
    }

    Action InputBindings::getActionForDrag(ToolMode mode, MouseButton button, int modifiers,
                                           const std::vector<int>& held_keys) const {
        const int mods = modifiers & MODIFIER_MASK;
        const auto find_in_mode = [&](ToolMode query_mode) -> Action {
            for (auto it_key = held_keys.rbegin(); it_key != held_keys.rend(); ++it_key) {
                const int chord = *it_key;
                if (auto it = drag_chord_map_.find({query_mode, button, mods, chord}); it != drag_chord_map_.end()) {
                    return it->second;
                }
            }
            if (auto it = drag_map_.find({query_mode, button, mods}); it != drag_map_.end()) {
                return it->second;
            }
            if (mods != MODIFIER_NONE) {
                if (auto it = drag_map_.find({query_mode, button, MODIFIER_NONE});
                    it != drag_map_.end() &&
                    describe(it->second).allows_extra_modifiers) {
                    return it->second;
                }
            }
            return Action::NONE;
        };

        if (const auto local_action = find_in_mode(mode); local_action != Action::NONE) {
            return local_action;
        }
        if (mode != ToolMode::GLOBAL) {
            const auto global_action = find_in_mode(ToolMode::GLOBAL);
            if (global_action != Action::NONE &&
                describe(global_action).inherits_from_global &&
                !isDisabled(mode, global_action)) {
                return global_action;
            }
        }
        return Action::NONE;
    }

    std::optional<InputTrigger> InputBindings::getTriggerForAction(Action action, ToolMode mode) const {
        for (const auto& binding : bindings_) {
            if (binding.action == action && binding.mode == mode) {
                return binding.trigger;
            }
        }
        return std::nullopt;
    }

    std::optional<InputTrigger> InputBindings::getEffectiveTriggerForAction(Action action, ToolMode mode) const {
        if (auto trigger = getTriggerForAction(action, mode)) {
            return trigger;
        }
        if (mode != ToolMode::GLOBAL && describe(action).inherits_from_global &&
            !isDisabled(mode, action)) {
            return getTriggerForAction(action, ToolMode::GLOBAL);
        }
        return std::nullopt;
    }

    std::string InputBindings::getTriggerDescription(Action action, ToolMode mode) const {
        const auto trigger = getEffectiveTriggerForAction(action, mode);
        if (!trigger) {
            return "Unbound";
        }

        return std::visit([](const auto& t) -> std::string {
            using T = std::decay_t<decltype(t)>;

            std::string result = getModifierString(t.modifiers);
            if (!result.empty())
                result += "+";

            if constexpr (std::is_same_v<T, KeyTrigger>) {
                return result + getKeyName(t.key);
            } else if constexpr (std::is_same_v<T, MouseButtonTrigger>) {
                std::string btn = getMouseButtonName(t.button);
                if (t.double_click) {
                    btn += " Double-Click";
                }
                return result + btn;
            } else if constexpr (std::is_same_v<T, MouseScrollTrigger>) {
                std::string chord = t.chord_key.has_value() ? getKeyName(*t.chord_key) + " + " : "";
                return chord + result + "Scroll";
            } else if constexpr (std::is_same_v<T, MouseDragTrigger>) {
                std::string chord = t.chord_key.has_value() ? getKeyName(*t.chord_key) + " + " : "";
                return chord + result + getMouseButtonName(t.button) + " Drag";
            } else {
                return "Unknown";
            }
        },
                          *trigger);
    }

    int InputBindings::getKeyForAction(Action action, ToolMode mode) const {
        const auto trigger = getEffectiveTriggerForAction(action, mode);
        if (!trigger)
            return -1;

        if (const auto* key_trigger = std::get_if<KeyTrigger>(&*trigger)) {
            return key_trigger->key;
        }
        return -1;
    }

    void InputBindings::setBinding(ToolMode mode, Action action, const InputTrigger& trigger) {
        if (!isValidTriggerForAction(action, trigger)) {
            LOG_WARN("Refusing invalid or unreachable input binding for '{}'", getActionName(action));
            return;
        }
        std::erase_if(bindings_, [mode, action](const Binding& b) {
            return b.mode == mode && b.action == action;
        });
        bindings_.push_back({mode, trigger, action, getActionName(action)});
        disabled_bindings_.erase({mode, action});
        rebuildLookupMaps();
        notifyBindingsChanged();
    }

    void InputBindings::clearBinding(ToolMode mode, Action action) {
        std::erase_if(bindings_, [mode, action](const Binding& b) {
            return b.mode == mode && b.action == action;
        });
        disabled_bindings_.insert({mode, action});
        rebuildLookupMaps();
        notifyBindingsChanged();
    }

    bool InputBindings::isDisabled(ToolMode mode, Action action) const {
        return disabled_bindings_.contains({mode, action});
    }

    std::vector<std::pair<ToolMode, Action>> InputBindings::unreachableCoreActions() const {
        constexpr std::array core_actions = {
            Action::CAMERA_ORBIT,
            Action::CAMERA_PAN,
            Action::CAMERA_ZOOM,
            Action::SELECTION_REPLACE,
            Action::SELECTION_ADD,
            Action::SELECTION_REMOVE,
            Action::DELETE_SELECTED,
            Action::DELETE_NODE,
            Action::UNDO,
            Action::REDO,
            Action::TOOL_SELECT,
            Action::TOOL_TRANSLATE,
            Action::TOOL_ROTATE,
            Action::TOOL_SCALE,
            Action::TOOL_MIRROR,
            Action::TOOL_ALIGN,
            Action::TOOL_SCALE_DISTANCE,
            Action::CANCEL_POLYGON,
        };
        const auto defaults = createDefaultProfile().bindings;
        const auto applies = [](const Binding& binding, const ToolMode mode) {
            return binding.mode == mode ||
                   (binding.mode == ToolMode::GLOBAL &&
                    describe(binding.action).inherits_from_global);
        };
        const auto dispatches = [this](const Binding& binding, const ToolMode mode) {
            return std::visit([&](const auto& trigger) {
                using T = std::decay_t<decltype(trigger)>;
                if constexpr (std::is_same_v<T, KeyTrigger>)
                    return getActionForKey(mode, trigger.key, trigger.modifiers) == binding.action;
                else if constexpr (std::is_same_v<T, MouseButtonTrigger>)
                    return getActionForMouseButton(mode, trigger.button, trigger.modifiers,
                                                   trigger.double_click) == binding.action;
                else if constexpr (std::is_same_v<T, MouseScrollTrigger>)
                    return getActionForScroll(mode, trigger.modifiers,
                                              trigger.chord_key
                                                  ? std::vector<int>{*trigger.chord_key}
                                                  : std::vector<int>{}) == binding.action;
                else
                    return getActionForDrag(mode, trigger.button, trigger.modifiers,
                                            trigger.chord_key
                                                ? std::vector<int>{*trigger.chord_key}
                                                : std::vector<int>{}) == binding.action;
            },
                              binding.trigger);
        };

        std::vector<std::pair<ToolMode, Action>> missing;
        for (const auto mode : kAllToolModes) {
            for (const auto action : core_actions) {
                // A mode's own default on the same trigger shadows a global one,
                // as Escape exits walk and VR instead of cancelling a polygon.
                const bool required = std::ranges::any_of(defaults, [&](const Binding& b) {
                    return b.action == action && applies(b, mode) &&
                           (b.mode == mode || std::ranges::none_of(defaults, [&](const Binding& local) {
                                return local.mode == mode && triggersOverlap(local.trigger, b.trigger);
                            }));
                });
                if (!required)
                    continue;
                const bool reachable = std::ranges::any_of(bindings_, [&](const Binding& b) {
                    return b.action == action && applies(b, mode) &&
                           (b.mode == mode || !isDisabled(mode, action)) &&
                           dispatches(b, mode);
                });
                if (!reachable)
                    missing.emplace_back(mode, action);
            }
        }
        return missing;
    }

    void InputBindings::restoreUnreachableCoreActions() {
        auto missing = unreachableCoreActions();
        bool changed = false;
        for (const auto& item : missing)
            changed |= disabled_bindings_.erase(item) > 0;
        missing = unreachableCoreActions();
        std::ranges::stable_sort(missing, [](const auto& lhs, const auto& rhs) {
            return lhs.first == ToolMode::GLOBAL && rhs.first != ToolMode::GLOBAL;
        });
        const auto defaults = createDefaultProfile().bindings;
        for (const auto& [mode, action] : missing) {
            const auto still_missing = unreachableCoreActions();
            if (std::ranges::find(still_missing, std::pair{mode, action}) == still_missing.end())
                continue;
            for (const auto& def : defaults) {
                if (def.action != action ||
                    (def.mode != mode &&
                     !(def.mode == ToolMode::GLOBAL &&
                       describe(action).inherits_from_global)))
                    continue;
                Binding restored = def;
                restored.mode = mode;
                bool added_for_action = false;
                const auto try_add = [&](const InputTrigger& trigger) {
                    if (findConflict(mode, trigger, action))
                        return false;
                    restored.trigger = trigger;
                    bindings_.push_back(restored);
                    changed = true;
                    rebuildLookupMaps();
                    added_for_action = true;
                    return true;
                };
                if (try_add(restored.trigger))
                    break;
                for (const int modifiers : {MODIFIER_CTRL | MODIFIER_ALT,
                                            MODIFIER_SHIFT | MODIFIER_ALT,
                                            MODIFIER_CTRL | MODIFIER_SHIFT | MODIFIER_ALT}) {
                    auto alternate = restored.trigger;
                    std::visit([&](auto& t) { t.modifiers = modifiers; }, alternate);
                    if (try_add(alternate))
                        break;
                }
                if (added_for_action)
                    break;
            }
        }
        if (!changed)
            return;
        rebuildLookupMaps();
        notifyBindingsChanged();
        if (const auto still_missing = unreachableCoreActions(); !still_missing.empty())
            LOG_WARN("{} core input actions remain unreachable after restore", still_missing.size());
        if (isValidProfileName(current_profile_name_))
            (void)saveProfile(current_profile_name_);
    }

    void InputBindings::warnAboutUnreachableCoreActions() {
        const auto missing = unreachableCoreActions();
        if (missing.empty())
            return;
        LOG_WARN("Input profile '{}' leaves {} core actions unreachable",
                 current_profile_name_, missing.size());
        const std::weak_ptr<int> lifetime = callback_lifetime_;
        auto& localization = lfs::event::LocalizationManager::getInstance();
        publishKeymapNotice(
            lfs::ErrorCode::FailedPrecondition,
            localization.get("input_settings.core_actions_unreachable"),
            lfs::ErrorSurface::Modal,
            {lfs::ErrorAction{
                 .kind = lfs::ErrorActionKind::Custom,
                 .label = localization.get("input_settings.restore_core_actions"),
                 .on_invoke = [this, lifetime](lfs::OperationId) {
                     if (lifetime.lock())
                         restoreUnreachableCoreActions();
                 },
             },
             lfs::ErrorAction{.kind = lfs::ErrorActionKind::Dismiss}});
    }

    std::optional<BindingConflict> InputBindings::findConflict(
        ToolMode mode, const InputTrigger& trigger, Action ignore_action) const {
        for (const auto& b : bindings_) {
            if (b.action == ignore_action) {
                continue;
            }
            if (b.mode != mode) {
                continue;
            }
            if (triggersOverlap(b.trigger, trigger)) {
                return BindingConflict{b.action, b.mode};
            }
        }

        if (mode == ToolMode::GLOBAL) {
            return std::nullopt;
        }

        for (const auto& b : bindings_) {
            if (b.action == ignore_action) {
                continue;
            }
            if (b.mode != ToolMode::GLOBAL || !describe(b.action).inherits_from_global ||
                isDisabled(mode, b.action)) {
                continue;
            }
            if (triggersOverlap(b.trigger, trigger)) {
                return BindingConflict{b.action, b.mode};
            }
        }
        return std::nullopt;
    }

    void InputBindings::notifyBindingsChanged() {
        ++bindings_revision_;
        if (bindings_revision_ == 0)
            ++bindings_revision_;
        if (on_bindings_changed_) {
            on_bindings_changed_();
        }
    }

    void InputBindings::rebuildLookupMaps() {
        key_map_.clear();
        mouse_button_map_.clear();
        scroll_map_.clear();
        drag_map_.clear();
        scroll_chord_map_.clear();
        drag_chord_map_.clear();

        for (const auto& binding : bindings_) {
            std::visit([&](auto&& t) {
                using T = std::decay_t<decltype(t)>;

                if constexpr (std::is_same_v<T, KeyTrigger>) {
                    key_map_[{binding.mode, t.key, t.modifiers}] = {binding.action, t.on_repeat};
                } else if constexpr (std::is_same_v<T, MouseButtonTrigger>) {
                    mouse_button_map_[{binding.mode, t.button, t.modifiers, t.double_click}] = binding.action;
                } else if constexpr (std::is_same_v<T, MouseScrollTrigger>) {
                    if (t.chord_key.has_value()) {
                        scroll_chord_map_[{binding.mode, t.modifiers, *t.chord_key}] = binding.action;
                    } else {
                        scroll_map_[{binding.mode, t.modifiers}] = binding.action;
                    }
                } else if constexpr (std::is_same_v<T, MouseDragTrigger>) {
                    if (t.chord_key.has_value()) {
                        drag_chord_map_[{binding.mode, t.button, t.modifiers, *t.chord_key}] = binding.action;
                    } else {
                        drag_map_[{binding.mode, t.button, t.modifiers}] = binding.action;
                    }
                }
            },
                       binding.trigger);
        }
    }

    Profile InputBindings::createDefaultProfile() {
        Profile profile;
        profile.name = "Default";
        profile.description = "Default LichtFeld Studio controls";

        // Global bindings are inherited by tool modes unless a mode provides a
        // local override for the same action.
        struct BaseBind {
            InputTrigger trigger;
            Action action;
            std::string desc;
        };
        const std::vector<BaseBind> global = {
            // Camera
            {MouseDragTrigger{MouseButton::MIDDLE, MODIFIER_NONE}, Action::CAMERA_ORBIT, "Orbit"},
            {MouseDragTrigger{MouseButton::RIGHT, MODIFIER_NONE}, Action::CAMERA_PAN, "Pan"},
            {MouseScrollTrigger{MODIFIER_NONE}, Action::CAMERA_ZOOM, "Zoom"},
            {MouseScrollTrigger{MODIFIER_NONE, KEY_R}, Action::CAMERA_ROLL, "Roll"},
            {MouseButtonTrigger{MouseButton::RIGHT, MODIFIER_NONE, true}, Action::CAMERA_SET_PIVOT, "Set pivot"},
            {KeyTrigger{KEY_W, MODIFIER_NONE, true}, Action::CAMERA_MOVE_FORWARD, "Forward"},
            {KeyTrigger{KEY_S, MODIFIER_NONE, true}, Action::CAMERA_MOVE_BACKWARD, "Backward"},
            {KeyTrigger{KEY_A, MODIFIER_NONE, true}, Action::CAMERA_MOVE_LEFT, "Left"},
            {KeyTrigger{KEY_D, MODIFIER_NONE, true}, Action::CAMERA_MOVE_RIGHT, "Right"},
            {KeyTrigger{KEY_Q, MODIFIER_NONE, true}, Action::CAMERA_MOVE_UP, "Up"},
            {KeyTrigger{KEY_E, MODIFIER_NONE, true}, Action::CAMERA_MOVE_DOWN, "Down"},
            {KeyTrigger{KEY_H, MODIFIER_NONE}, Action::CAMERA_RESET_HOME, "Home"},
            {KeyTrigger{KEY_H, MODIFIER_SHIFT}, Action::CAMERA_SET_HOME, "Set home"},
            {KeyTrigger{KEY_H, MODIFIER_CTRL}, Action::CAMERA_RESTORE_HOME, "Reset home"},
            {KeyTrigger{KEY_F, MODIFIER_NONE}, Action::CAMERA_FOCUS_SELECTION, "Focus selection"},
            {KeyTrigger{KEY_RIGHT, MODIFIER_NONE, true}, Action::CAMERA_NEXT_VIEW, "Next view"},
            {KeyTrigger{KEY_LEFT, MODIFIER_NONE, true}, Action::CAMERA_PREV_VIEW, "Prev view"},
            {KeyTrigger{KEY_EQUAL, MODIFIER_CTRL}, Action::CAMERA_SPEED_UP, "Speed up"},
            {KeyTrigger{KEY_MINUS, MODIFIER_CTRL}, Action::CAMERA_SPEED_DOWN, "Speed down"},
            {KeyTrigger{KEY_KP_ADD, MODIFIER_CTRL}, Action::CAMERA_SPEED_UP, "Speed up"},
            {KeyTrigger{KEY_KP_SUBTRACT, MODIFIER_CTRL}, Action::CAMERA_SPEED_DOWN, "Speed down"},
            {KeyTrigger{KEY_EQUAL, MODIFIER_CTRL | MODIFIER_SHIFT}, Action::ZOOM_SPEED_UP, "Zoom speed up"},
            {KeyTrigger{KEY_MINUS, MODIFIER_CTRL | MODIFIER_SHIFT}, Action::ZOOM_SPEED_DOWN, "Zoom speed down"},
            {KeyTrigger{KEY_KP_ADD, MODIFIER_CTRL | MODIFIER_SHIFT}, Action::ZOOM_SPEED_UP, "Zoom speed up"},
            {KeyTrigger{KEY_KP_SUBTRACT, MODIFIER_CTRL | MODIFIER_SHIFT}, Action::ZOOM_SPEED_DOWN, "Zoom speed down"},
            // View
            {KeyTrigger{KEY_V, MODIFIER_NONE}, Action::TOGGLE_SPLIT_VIEW, "Split view"},
            {KeyTrigger{KEY_V, MODIFIER_SHIFT}, Action::TOGGLE_INDEPENDENT_SPLIT_VIEW, "Independent split"},
            {KeyTrigger{KEY_G, MODIFIER_NONE}, Action::TOGGLE_GT_COMPARISON, "GT comparison"},
            {KeyTrigger{KEY_C, MODIFIER_ALT}, Action::TOGGLE_CAMERA_FRUSTUMS, "Camera frustums"},
            {KeyTrigger{KEY_G, MODIFIER_ALT}, Action::TOGGLE_GRID, "Grid"},
            {KeyTrigger{KEY_T, MODIFIER_NONE}, Action::CYCLE_PLY, "Cycle PLY"},
            // Editing (Delete is mode-specific, added below)
            {KeyTrigger{KEY_Z, MODIFIER_CTRL}, Action::UNDO, "Undo"},
            {KeyTrigger{KEY_Y, MODIFIER_CTRL}, Action::REDO, "Redo"},
            {KeyTrigger{KEY_I, MODIFIER_CTRL}, Action::INVERT_SELECTION, "Invert"},
            {KeyTrigger{KEY_D, MODIFIER_CTRL}, Action::DESELECT_ALL, "Deselect"},
            {KeyTrigger{KEY_A, MODIFIER_CTRL}, Action::SELECT_ALL, "Select all"},
            {KeyTrigger{KEY_C, MODIFIER_CTRL}, Action::COPY_SELECTION, "Copy"},
            {KeyTrigger{KEY_X, MODIFIER_CTRL}, Action::CUT_SELECTION, "Cut"},
            {KeyTrigger{KEY_V, MODIFIER_CTRL}, Action::PASTE_SELECTION, "Paste"},
            {KeyTrigger{KEY_ENTER, MODIFIER_CTRL}, Action::ASSET_GALLERY_PRIMARY, "Gallery Primary Action"},
            {KeyTrigger{KEY_C, MODIFIER_CTRL | MODIFIER_SHIFT}, Action::ASSET_GALLERY_COPY_LINK, "Copy Gallery Link"},
            {KeyTrigger{KEY_F5, MODIFIER_NONE}, Action::ASSET_REFRESH, "Refresh Assets"},
            // Selection mode shortcuts
            {KeyTrigger{KEY_T, MODIFIER_CTRL}, Action::CYCLE_SELECTION_VIS, "Sel vis"},
            {KeyTrigger{KEY_1, MODIFIER_CTRL}, Action::SELECT_MODE_CENTERS, "Centers"},
            {KeyTrigger{KEY_2, MODIFIER_CTRL}, Action::SELECT_MODE_RECTANGLE, "Rectangle"},
            {KeyTrigger{KEY_3, MODIFIER_CTRL}, Action::SELECT_MODE_POLYGON, "Polygon"},
            {KeyTrigger{KEY_4, MODIFIER_CTRL}, Action::SELECT_MODE_LASSO, "Lasso"},
            {KeyTrigger{KEY_5, MODIFIER_CTRL}, Action::SELECT_MODE_RINGS, "Rings"},
            {KeyTrigger{KEY_6, MODIFIER_CTRL}, Action::SELECT_MODE_COLOR, "Color"},
            {KeyTrigger{KEY_7, MODIFIER_CTRL}, Action::SELECT_MODE_BOX, "Box"},
            {KeyTrigger{KEY_8, MODIFIER_CTRL}, Action::SELECT_MODE_SPHERE, "Sphere"},
            {KeyTrigger{KEY_ESCAPE, MODIFIER_NONE}, Action::CANCEL_POLYGON, "Cancel"},
            // UI
            {KeyTrigger{KEY_TAB, MODIFIER_CTRL}, Action::NEXT_SIDEBAR_CATEGORY,
             getActionName(Action::NEXT_SIDEBAR_CATEGORY)},
            {KeyTrigger{KEY_TAB, MODIFIER_CTRL | MODIFIER_SHIFT}, Action::PREVIOUS_SIDEBAR_CATEGORY,
             getActionName(Action::PREVIOUS_SIDEBAR_CATEGORY)},
            {KeyTrigger{KEY_F12, MODIFIER_NONE}, Action::TOGGLE_UI, "Hide UI"},
            {KeyTrigger{KEY_F11, MODIFIER_NONE}, Action::TOGGLE_FULLSCREEN, "Fullscreen"},
            {KeyTrigger{KEY_F10, MODIFIER_NONE}, Action::TOGGLE_PERFORMANCE_HUD, "Performance HUD"},
            {KeyTrigger{KEY_COMMA, MODIFIER_CTRL}, Action::OPEN_PREFERENCES, "Preferences"},
            {KeyTrigger{KEY_GRAVE_ACCENT, MODIFIER_CTRL}, Action::TOGGLE_PYTHON_CONSOLE,
             getActionName(Action::TOGGLE_PYTHON_CONSOLE)},
            {KeyTrigger{KEY_A, MODIFIER_CTRL | MODIFIER_SHIFT}, Action::SELECT_ALL_SCENE_NODES,
             getActionName(Action::SELECT_ALL_SCENE_NODES)},
            {KeyTrigger{KEY_H, MODIFIER_CTRL | MODIFIER_SHIFT}, Action::TOGGLE_SCENE_SELECTION_VISIBILITY,
             getActionName(Action::TOGGLE_SCENE_SELECTION_VISIBILITY)},
            {KeyTrigger{KEY_T, MODIFIER_CTRL | MODIFIER_SHIFT}, Action::TOGGLE_SCENE_SELECTION_TRAINING,
             getActionName(Action::TOGGLE_SCENE_SELECTION_TRAINING)},
            {KeyTrigger{KEY_G, MODIFIER_CTRL}, Action::GROUP_SELECTED_SCENE_NODES,
             getActionName(Action::GROUP_SELECTED_SCENE_NODES)},
            {KeyTrigger{KEY_G, MODIFIER_CTRL | MODIFIER_SHIFT}, Action::UNGROUP_SELECTED_SCENE_NODE,
             getActionName(Action::UNGROUP_SELECTED_SCENE_NODE)},
            {KeyTrigger{KEY_M, MODIFIER_CTRL | MODIFIER_SHIFT}, Action::TOGGLE_MCP_SERVER,
             getActionName(Action::TOGGLE_MCP_SERVER)},
            {KeyTrigger{KEY_N, MODIFIER_CTRL | MODIFIER_SHIFT}, Action::TOGGLE_MCP_BINDING,
             getActionName(Action::TOGGLE_MCP_BINDING)},
            {MouseScrollTrigger{MODIFIER_CTRL}, Action::HISTOGRAM_ZOOM_MARKED, "Zoom histogram at cursor"},
            // Sequencer
            {KeyTrigger{KEY_K, MODIFIER_NONE}, Action::SEQUENCER_ADD_KEYFRAME, "Add keyframe"},
            {KeyTrigger{KEY_U, MODIFIER_NONE}, Action::SEQUENCER_UPDATE_KEYFRAME, "Update keyframe"},
            {KeyTrigger{KEY_SPACE, MODIFIER_NONE}, Action::SEQUENCER_PLAY_PAUSE, "Play/Pause"},
            // Tool shortcuts
            {KeyTrigger{KEY_1}, Action::TOOL_SELECT, "Select"},
            {KeyTrigger{KEY_2}, Action::TOOL_TRANSLATE, "Translate"},
            {KeyTrigger{KEY_3}, Action::TOOL_ROTATE, "Rotate"},
            {KeyTrigger{KEY_4}, Action::TOOL_SCALE, "Scale"},
            {KeyTrigger{KEY_5}, Action::TOOL_MIRROR, "Mirror"},
            {KeyTrigger{KEY_6}, Action::TOOL_ALIGN, "Align"},
            {KeyTrigger{KEY_7}, Action::TOOL_SCALE_DISTANCE, "Scale Distance"},
            {KeyTrigger{KEY_GRAVE_ACCENT}, Action::PIE_MENU, "Pie Menu"},
        };

        for (const auto& b : global) {
            profile.bindings.push_back({ToolMode::GLOBAL, b.trigger, b.action, b.desc});
        }

        const std::vector<BaseBind> selection_drags = {
            {MouseDragTrigger{MouseButton::LEFT, MODIFIER_NONE}, Action::SELECTION_REPLACE, "Select"},
            {MouseDragTrigger{MouseButton::LEFT, MODIFIER_SHIFT}, Action::SELECTION_ADD, "Add sel"},
            {MouseDragTrigger{MouseButton::LEFT, MODIFIER_CTRL}, Action::SELECTION_REMOVE, "Remove sel"},
            {MouseDragTrigger{MouseButton::LEFT, MODIFIER_ALT}, Action::SELECTION_INTERSECT, "Intersect sel"},
        };
        for (const auto& b : selection_drags) {
            profile.bindings.push_back({ToolMode::SELECTION, b.trigger, b.action, b.desc});
        }

        profile.bindings.push_back({ToolMode::SELECTION,
                                    KeyTrigger{KEY_ENTER, MODIFIER_NONE},
                                    Action::CONFIRM_POLYGON,
                                    "Confirm polygon"});
        profile.bindings.push_back({ToolMode::SELECTION,
                                    MouseButtonTrigger{MouseButton::RIGHT, MODIFIER_NONE},
                                    Action::UNDO_POLYGON_VERTEX,
                                    "Undo polygon/cancel selection"});
        profile.bindings.push_back({ToolMode::SELECTION,
                                    KeyTrigger{KEY_X, MODIFIER_NONE},
                                    Action::TOGGLE_SELECTION_DEPTH_FILTER,
                                    "Depth box"});
        profile.bindings.push_back({ToolMode::SELECTION,
                                    MouseScrollTrigger{MODIFIER_ALT},
                                    Action::DEPTH_ADJUST_FAR,
                                    "Depth"});
        profile.bindings.push_back({ToolMode::SELECTION,
                                    MouseScrollTrigger{MODIFIER_CTRL},
                                    Action::BRUSH_RESIZE,
                                    "Brush size"});
        profile.bindings.push_back({ToolMode::SELECTION,
                                    MouseScrollTrigger{MODIFIER_SHIFT},
                                    Action::BRUSH_RESIZE,
                                    "Brush size"});
        profile.bindings.push_back({ToolMode::SELECTION,
                                    KeyTrigger{KEY_C, MODIFIER_CTRL | MODIFIER_ALT},
                                    Action::TOGGLE_SELECTION_CROP_FILTER,
                                    "Crop filter"});
        profile.bindings.push_back({ToolMode::EDITOR, KeyTrigger{KEY_F5}, Action::EDITOR_RUN, "Run Document"});
        profile.bindings.push_back({ToolMode::EDITOR, KeyTrigger{KEY_ENTER, MODIFIER_CTRL}, Action::EDITOR_RUN, "Run Document"});
        profile.bindings.push_back({ToolMode::EDITOR, KeyTrigger{KEY_KP_ENTER, MODIFIER_CTRL}, Action::EDITOR_RUN, "Run Document"});
        profile.bindings.push_back({ToolMode::EDITOR, KeyTrigger{KEY_C, MODIFIER_CTRL}, Action::EDITOR_STOP, "Stop Document"});
        profile.bindings.push_back({ToolMode::WALK, KeyTrigger{KEY_ESCAPE, MODIFIER_NONE}, Action::EXIT_WALK, "Exit Walk"});
        profile.bindings.push_back({ToolMode::VR, KeyTrigger{KEY_LEFT, MODIFIER_NONE}, Action::VR_SNAP_TURN_LEFT, "Snap Turn Left"});
        profile.bindings.push_back({ToolMode::VR, KeyTrigger{KEY_RIGHT, MODIFIER_NONE}, Action::VR_SNAP_TURN_RIGHT, "Snap Turn Right"});
        profile.bindings.push_back({ToolMode::VR, KeyTrigger{KEY_R, MODIFIER_NONE}, Action::VR_RECENTER, "Recenter VR"});
        profile.bindings.push_back({ToolMode::VR, KeyTrigger{KEY_ESCAPE, MODIFIER_NONE}, Action::EXIT_VR, "Exit VR"});
        profile.bindings.push_back({ToolMode::CROP_BOX,
                                    KeyTrigger{KEY_ENTER, MODIFIER_NONE},
                                    Action::APPLY_CROP_BOX,
                                    "Apply/confirm"});
        profile.bindings.push_back({ToolMode::CROP_BOX,
                                    KeyTrigger{KEY_KP_ENTER, MODIFIER_NONE},
                                    Action::APPLY_CROP_BOX,
                                    "Apply/confirm"});
        // Closed crop polygon editing, as on the closed selection polygon.
        profile.bindings.push_back({ToolMode::CROP_BOX,
                                    MouseButtonTrigger{MouseButton::LEFT, MODIFIER_SHIFT},
                                    Action::INSERT_CROP_POLYGON_VERTEX,
                                    "Insert polygon vertex"});
        profile.bindings.push_back({ToolMode::CROP_BOX,
                                    MouseButtonTrigger{MouseButton::LEFT, MODIFIER_CTRL},
                                    Action::REMOVE_CROP_POLYGON_VERTEX,
                                    "Remove polygon vertex"});
        profile.bindings.push_back({ToolMode::ALIGN,
                                    KeyTrigger{KEY_BACKSPACE, MODIFIER_NONE},
                                    Action::REMOVE_LAST_POINT,
                                    "Remove last point"});
        profile.bindings.push_back({ToolMode::ALIGN,
                                    KeyTrigger{KEY_ENTER, MODIFIER_NONE},
                                    Action::APPLY_ALIGNMENT,
                                    "Apply alignment"});
        profile.bindings.push_back({ToolMode::ALIGN,
                                    KeyTrigger{KEY_KP_ENTER, MODIFIER_NONE},
                                    Action::APPLY_ALIGNMENT,
                                    "Apply alignment"});

        // Node picking only for transform modes (not selection/cropbox/align)
        for (const auto mode : NODE_PICK_MODES) {
            profile.bindings.push_back({mode, MouseButtonTrigger{MouseButton::LEFT, MODIFIER_NONE}, Action::NODE_PICK, "Pick node"});
            profile.bindings.push_back({mode, MouseDragTrigger{MouseButton::LEFT, MODIFIER_NONE}, Action::NODE_RECT_SELECT, "Rectangle select nodes"});
        }

        // Delete key: GLOBAL/transform modes delete node, selection-like modes delete Gaussians.
        for (const auto mode : DELETE_NODE_MODES) {
            profile.bindings.push_back({mode, KeyTrigger{KEY_DELETE, MODIFIER_NONE}, Action::DELETE_NODE, "Delete node"});
        }

        for (const auto mode : DELETE_GAUSSIANS_MODES) {
            profile.bindings.push_back({mode, KeyTrigger{KEY_DELETE, MODIFIER_NONE}, Action::DELETE_SELECTED, "Delete Gaussians"});
        }

        return profile;
    }

    std::string getActionName(const Action action) {
        switch (action) {
        case Action::NONE: return "None";
        case Action::CAMERA_ORBIT: return "Camera Orbit";
        case Action::CAMERA_PAN: return "Camera Pan";
        case Action::CAMERA_ZOOM: return "Camera Zoom";
        case Action::CAMERA_ROLL: return "Camera Roll";
        case Action::CAMERA_MOVE_FORWARD: return "Move Forward";
        case Action::CAMERA_MOVE_BACKWARD: return "Move Backward";
        case Action::CAMERA_MOVE_LEFT: return "Move Left";
        case Action::CAMERA_MOVE_RIGHT: return "Move Right";
        case Action::CAMERA_MOVE_UP: return "Move Up";
        case Action::CAMERA_MOVE_DOWN: return "Move Down";
        case Action::CAMERA_RESET_HOME: return "Go to Home";
        case Action::CAMERA_SET_HOME: return "Set Home";
        case Action::CAMERA_RESTORE_HOME: return "Reset Home";
        case Action::CAMERA_FOCUS_SELECTION: return "Focus Selection";
        case Action::CAMERA_SET_PIVOT: return "Set Pivot";
        case Action::CAMERA_NEXT_VIEW: return "Next Camera View";
        case Action::CAMERA_PREV_VIEW: return "Previous Camera View";
        case Action::CAMERA_SPEED_UP: return "Increase Move Speed";
        case Action::CAMERA_SPEED_DOWN: return "Decrease Move Speed";
        case Action::ZOOM_SPEED_UP: return "Increase Zoom Speed";
        case Action::ZOOM_SPEED_DOWN: return "Decrease Zoom Speed";
        case Action::TOGGLE_SPLIT_VIEW: return "Toggle Split View";
        case Action::TOGGLE_INDEPENDENT_SPLIT_VIEW: return "Toggle Independent Split View";
        case Action::TOGGLE_GT_COMPARISON: return "Toggle GT Comparison";
        case Action::TOGGLE_DEPTH_MODE: return "Toggle Depth Box";
        case Action::CYCLE_PLY: return "Cycle PLY";
        case Action::DELETE_SELECTED: return "Delete Selected Gaussians";
        case Action::DELETE_NODE: return "Delete Node";
        case Action::UNDO: return "Undo";
        case Action::REDO: return "Redo";
        case Action::INVERT_SELECTION: return "Invert Selection";
        case Action::DESELECT_ALL: return "Deselect All";
        case Action::COPY_SELECTION: return "Copy Selection";
        case Action::CUT_SELECTION: return "Cut Selection";
        case Action::PASTE_SELECTION: return "Paste Selection";
        case Action::DEPTH_ADJUST_NEAR: return "Adjust Depth Box";
        case Action::DEPTH_ADJUST_FAR: return "Adjust Depth Box";
        case Action::DEPTH_ADJUST_SIDE: return "Adjust Depth Box";
        case Action::TOGGLE_SELECTION_DEPTH_FILTER: return "Toggle Depth Box";
        case Action::TOGGLE_SELECTION_CROP_FILTER: return "Toggle Selection Crop Filter";
        case Action::BRUSH_RESIZE: return "Resize Brush";
        case Action::CONFIRM_POLYGON: return "Confirm Polygon";
        case Action::CANCEL_POLYGON: return "Cancel Polygon";
        case Action::UNDO_POLYGON_VERTEX: return "Undo Polygon Vertex / Cancel Selection";
        case Action::CYCLE_SELECTION_VIS: return "Cycle Selection Visualization";
        case Action::SELECTION_REPLACE: return "Selection: Replace";
        case Action::SELECTION_ADD: return "Selection: Add";
        case Action::SELECTION_REMOVE: return "Selection: Remove";
        case Action::SELECTION_INTERSECT: return "Selection: Intersect";
        case Action::SELECT_MODE_CENTERS: return "Selection: Centers";
        case Action::SELECT_MODE_RECTANGLE: return "Selection: Rectangle";
        case Action::SELECT_MODE_POLYGON: return "Selection: Polygon";
        case Action::SELECT_MODE_LASSO: return "Selection: Lasso";
        case Action::SELECT_MODE_RINGS: return "Selection: Rings";
        case Action::SELECT_MODE_COLOR: return "Selection: Color";
        case Action::SELECT_MODE_BOX: return "Selection: Box";
        case Action::SELECT_MODE_SPHERE: return "Selection: Sphere";
        case Action::APPLY_CROP_BOX: return "Apply Crop Box";
        case Action::NODE_PICK: return "Pick Node";
        case Action::NODE_RECT_SELECT: return "Rectangle Select Nodes";
        case Action::TOGGLE_UI: return "Toggle UI";
        case Action::TOGGLE_FULLSCREEN: return "Toggle Fullscreen";
        case Action::TOGGLE_PERFORMANCE_HUD: return "Toggle Performance HUD";
        case Action::SEQUENCER_ADD_KEYFRAME: return "Add Keyframe";
        case Action::SEQUENCER_UPDATE_KEYFRAME: return "Update Keyframe";
        case Action::SEQUENCER_PLAY_PAUSE: return "Play/Pause";
        case Action::TOOL_SELECT: return "Select Tool";
        case Action::TOOL_TRANSLATE: return "Translate Tool";
        case Action::TOOL_ROTATE: return "Rotate Tool";
        case Action::TOOL_SCALE: return "Scale Tool";
        case Action::TOOL_MIRROR: return "Mirror Tool";
        case Action::TOOL_ALIGN: return "Align Tool";
        case Action::TOOL_SCALE_DISTANCE: return "Scale Distance Tool";
        case Action::EDITOR_TOGGLE_KIND: return "Toggle Document Kind";
        case Action::EDITOR_RUN: return "Run Document";
        case Action::EDITOR_STOP: return "Stop Document";
        case Action::PIE_MENU: return "Pie Menu";
        case Action::HISTOGRAM_ZOOM_MARKED: return "Zoom Histogram at Cursor";
        case Action::NEXT_SIDEBAR_CATEGORY: return "Next Sidebar Category";
        case Action::PREVIOUS_SIDEBAR_CATEGORY: return "Previous Sidebar Category";
        case Action::REMOVE_LAST_POINT: return "Remove Last Point";
        case Action::APPLY_ALIGNMENT: return "Apply Alignment";
        case Action::INSERT_CROP_POLYGON_VERTEX: return "Insert Crop Polygon Vertex";
        case Action::REMOVE_CROP_POLYGON_VERTEX: return "Remove Crop Polygon Vertex";
        case Action::EXIT_WALK: return "Exit Walk";
        case Action::VR_SNAP_TURN_LEFT: return "Snap Turn Left";
        case Action::VR_SNAP_TURN_RIGHT: return "Snap Turn Right";
        case Action::VR_RECENTER: return "Recenter VR";
        case Action::EXIT_VR: return "Exit VR";
        case Action::TOGGLE_CAMERA_FRUSTUMS: return "Toggle Camera Frustums";
        case Action::OPEN_PREFERENCES: return "Open Preferences";
        case Action::TOGGLE_MCP_SERVER: return "Toggle MCP Server";
        case Action::TOGGLE_MCP_BINDING: return "Toggle MCP Local/Network Binding";
        case Action::TOGGLE_GRID: return "Toggle Grid";
        case Action::SELECT_ALL_SCENE_NODES: return "Select All Scene Nodes";
        case Action::TOGGLE_SCENE_SELECTION_VISIBILITY: return "Toggle Scene Selection Visibility";
        case Action::TOGGLE_SCENE_SELECTION_TRAINING: return "Toggle Scene Selection Training";
        case Action::GROUP_SELECTED_SCENE_NODES: return "Group Selected Scene Nodes";
        case Action::ASSET_GALLERY_PRIMARY: return "Gallery Primary Action";
        case Action::ASSET_GALLERY_COPY_LINK: return "Copy Gallery Link";
        case Action::ASSET_REFRESH: return "Refresh Assets";
        case Action::TOGGLE_PYTHON_CONSOLE: return "Toggle Python Console";
        case Action::UNGROUP_SELECTED_SCENE_NODE: return "Ungroup Selected Scene Node";
        default: return "Unknown";
        }
    }

    std::string_view actionNameKey(const Action action) {
        switch (action) {
        case Action::NONE: return "none";
        case Action::CAMERA_ORBIT: return "camera_orbit";
        case Action::CAMERA_PAN: return "camera_pan";
        case Action::CAMERA_ZOOM: return "camera_zoom";
        case Action::CAMERA_ROLL: return "camera_roll";
        case Action::CAMERA_MOVE_FORWARD: return "camera_move_forward";
        case Action::CAMERA_MOVE_BACKWARD: return "camera_move_backward";
        case Action::CAMERA_MOVE_LEFT: return "camera_move_left";
        case Action::CAMERA_MOVE_RIGHT: return "camera_move_right";
        case Action::CAMERA_MOVE_UP: return "camera_move_up";
        case Action::CAMERA_MOVE_DOWN: return "camera_move_down";
        case Action::CAMERA_RESET_HOME: return "camera_reset_home";
        case Action::CAMERA_SET_HOME: return "camera_set_home";
        case Action::CAMERA_RESTORE_HOME: return "camera_restore_home";
        case Action::CAMERA_FOCUS_SELECTION: return "camera_focus_selection";
        case Action::CAMERA_SET_PIVOT: return "camera_set_pivot";
        case Action::CAMERA_NEXT_VIEW: return "camera_next_view";
        case Action::CAMERA_PREV_VIEW: return "camera_prev_view";
        case Action::CAMERA_SPEED_UP: return "camera_speed_up";
        case Action::CAMERA_SPEED_DOWN: return "camera_speed_down";
        case Action::ZOOM_SPEED_UP: return "zoom_speed_up";
        case Action::ZOOM_SPEED_DOWN: return "zoom_speed_down";
        case Action::TOGGLE_SPLIT_VIEW: return "toggle_split_view";
        case Action::TOGGLE_INDEPENDENT_SPLIT_VIEW: return "toggle_independent_split_view";
        case Action::TOGGLE_GT_COMPARISON: return "toggle_gt_comparison";
        case Action::TOGGLE_DEPTH_MODE: return "toggle_depth_mode";
        case Action::CYCLE_PLY: return "cycle_ply";
        case Action::DELETE_SELECTED: return "delete_selected";
        case Action::DELETE_NODE: return "delete_node";
        case Action::UNDO: return "undo";
        case Action::REDO: return "redo";
        case Action::SELECT_ALL: return "select_all";
        case Action::INVERT_SELECTION: return "invert_selection";
        case Action::DESELECT_ALL: return "deselect_all";
        case Action::COPY_SELECTION: return "copy_selection";
        case Action::CUT_SELECTION: return "cut_selection";
        case Action::PASTE_SELECTION: return "paste_selection";
        case Action::DEPTH_ADJUST_NEAR: return "depth_adjust_near";
        case Action::DEPTH_ADJUST_FAR: return "depth_adjust_far";
        case Action::DEPTH_ADJUST_SIDE: return "depth_adjust_side";
        case Action::TOGGLE_SELECTION_DEPTH_FILTER: return "toggle_selection_depth_filter";
        case Action::TOGGLE_SELECTION_CROP_FILTER: return "toggle_selection_crop_filter";
        case Action::BRUSH_RESIZE: return "brush_resize";
        case Action::CONFIRM_POLYGON: return "confirm_polygon";
        case Action::CANCEL_POLYGON: return "cancel_polygon";
        case Action::UNDO_POLYGON_VERTEX: return "undo_polygon_vertex";
        case Action::CYCLE_SELECTION_VIS: return "cycle_selection_vis";
        case Action::SELECTION_REPLACE: return "selection_replace";
        case Action::SELECTION_ADD: return "selection_add";
        case Action::SELECTION_REMOVE: return "selection_remove";
        case Action::SELECTION_INTERSECT: return "selection_intersect";
        case Action::SELECT_MODE_CENTERS: return "select_mode_centers";
        case Action::SELECT_MODE_RECTANGLE: return "select_mode_rectangle";
        case Action::SELECT_MODE_POLYGON: return "select_mode_polygon";
        case Action::SELECT_MODE_LASSO: return "select_mode_lasso";
        case Action::SELECT_MODE_RINGS: return "select_mode_rings";
        case Action::SELECT_MODE_COLOR: return "select_mode_color";
        case Action::SELECT_MODE_BOX: return "select_mode_box";
        case Action::SELECT_MODE_SPHERE: return "select_mode_sphere";
        case Action::APPLY_CROP_BOX: return "apply_crop_box";
        case Action::NODE_PICK: return "node_pick";
        case Action::NODE_RECT_SELECT: return "node_rect_select";
        case Action::TOGGLE_UI: return "toggle_ui";
        case Action::TOGGLE_FULLSCREEN: return "toggle_fullscreen";
        case Action::TOGGLE_PERFORMANCE_HUD: return "toggle_performance_hud";
        case Action::SEQUENCER_ADD_KEYFRAME: return "sequencer_add_keyframe";
        case Action::SEQUENCER_UPDATE_KEYFRAME: return "sequencer_update_keyframe";
        case Action::SEQUENCER_PLAY_PAUSE: return "sequencer_play_pause";
        case Action::TOOL_SELECT: return "tool_select";
        case Action::TOOL_TRANSLATE: return "tool_translate";
        case Action::TOOL_ROTATE: return "tool_rotate";
        case Action::TOOL_SCALE: return "tool_scale";
        case Action::TOOL_MIRROR: return "tool_mirror";
        case Action::TOOL_ALIGN: return "tool_align";
        case Action::TOOL_SCALE_DISTANCE: return "tool_scale_distance";
        case Action::EDITOR_TOGGLE_KIND: return "editor_toggle_kind";
        case Action::EDITOR_RUN: return "editor_run";
        case Action::EDITOR_STOP: return "editor_stop";
        case Action::PIE_MENU: return "pie_menu";
        case Action::HISTOGRAM_ZOOM_MARKED: return "histogram_zoom_marked";
        case Action::NEXT_SIDEBAR_CATEGORY: return "next_sidebar_category";
        case Action::PREVIOUS_SIDEBAR_CATEGORY: return "previous_sidebar_category";
        case Action::REMOVE_LAST_POINT: return "remove_last_point";
        case Action::APPLY_ALIGNMENT: return "apply_alignment";
        case Action::INSERT_CROP_POLYGON_VERTEX: return "insert_crop_polygon_vertex";
        case Action::REMOVE_CROP_POLYGON_VERTEX: return "remove_crop_polygon_vertex";
        case Action::EXIT_WALK: return "exit_walk";
        case Action::VR_SNAP_TURN_LEFT: return "vr_snap_turn_left";
        case Action::VR_SNAP_TURN_RIGHT: return "vr_snap_turn_right";
        case Action::VR_RECENTER: return "vr_recenter";
        case Action::EXIT_VR: return "exit_vr";
        case Action::TOGGLE_CAMERA_FRUSTUMS: return "toggle_camera_frustums";
        case Action::OPEN_PREFERENCES: return "open_preferences";
        case Action::TOGGLE_MCP_SERVER: return "toggle_mcp_server";
        case Action::TOGGLE_MCP_BINDING: return "toggle_mcp_binding";
        case Action::TOGGLE_GRID: return "toggle_grid";
        case Action::SELECT_ALL_SCENE_NODES: return "select_all_scene_nodes";
        case Action::TOGGLE_SCENE_SELECTION_VISIBILITY: return "toggle_scene_selection_visibility";
        case Action::TOGGLE_SCENE_SELECTION_TRAINING: return "toggle_scene_selection_training";
        case Action::GROUP_SELECTED_SCENE_NODES: return "group_selected_scene_nodes";
        case Action::ASSET_GALLERY_PRIMARY: return "asset_gallery_primary";
        case Action::ASSET_GALLERY_COPY_LINK: return "asset_gallery_copy_link";
        case Action::ASSET_REFRESH: return "asset_refresh";
        case Action::TOGGLE_PYTHON_CONSOLE: return "toggle_python_console";
        case Action::UNGROUP_SELECTED_SCENE_NODE: return "ungroup_selected_scene_node";
        default: return {};
        }
    }

    namespace {
        std::string lookupLocale(std::string_view key, std::string_view fallback) {
            if (key.empty())
                return std::string(fallback);
            const std::string key_str(key);
            const char* const localized = lfs::event::LocalizationManager::getInstance().get(key_str);
            if (localized && std::string_view(localized) != key_str)
                return localized;
            return std::string(fallback);
        }

        struct ToolModeEntry {
            ToolMode mode;
            std::string_view suffix;
            std::string_view english;
        };
        constexpr ToolModeEntry kToolModeEntries[] = {
            {ToolMode::GLOBAL, "global", "Global"},
            {ToolMode::SELECTION, "selection", "Selection"},
            {ToolMode::TRANSLATE, "translate", "Translate"},
            {ToolMode::ROTATE, "rotate", "Rotate"},
            {ToolMode::SCALE, "scale", "Scale"},
            {ToolMode::ALIGN, "align", "Align"},
            {ToolMode::CROP_BOX, "crop_box", "Crop Box"},
            {ToolMode::WALK, "walk", "Walk"},
            {ToolMode::VR, "vr", "VR"},
            {ToolMode::EDITOR, "editor", "Editor"},
        };
    } // namespace

    std::optional<Action> actionFromName(std::string_view name) {
        static const auto table = [] {
            std::unordered_map<std::string, Action> m;
            for (int i = 0; i <= static_cast<int>(LAST_ACTION); ++i) {
                const auto action = static_cast<Action>(i);
                const auto key = actionNameKey(action);
                if (!key.empty())
                    m.emplace(key, action);
            }
            return m;
        }();
        std::string normalized(name);
        std::ranges::transform(normalized, normalized.begin(),
                               [](unsigned char c) { return std::tolower(c); });
        const auto it = table.find(normalized);
        return it == table.end() ? std::nullopt : std::optional<Action>(it->second);
    }

    ToolMode toolModeFromName(std::string_view name) {
        std::string normalized(name);
        std::ranges::transform(normalized, normalized.begin(),
                               [](unsigned char c) { return std::tolower(c); });
        for (const auto& [mode, suffix, english] : kToolModeEntries) {
            if (suffix == normalized)
                return mode;
        }
        return ToolMode::GLOBAL;
    }

    std::string getLocalizedActionName(const Action action) {
        const auto suffix = actionNameKey(action);
        if (suffix.empty())
            return getActionName(action);
        return lookupLocale(
            std::string("input_settings.action.").append(suffix),
            getActionName(action));
    }

    std::string getLocalizedToolModeName(const ToolMode mode) {
        if (mode == ToolMode::WALK)
            return lookupLocale("collision.walk", "Walk");
        for (const auto& [m, suffix, english] : kToolModeEntries) {
            if (m == mode)
                return lookupLocale(std::string("input_settings.mode.").append(suffix), english);
        }
        return lookupLocale("input_settings.mode.unknown", "Unknown");
    }

    std::string localizeTriggerDescription(std::string desc) {
        if (desc.empty())
            return desc;
        if (desc == "Unbound")
            return lookupLocale("input_settings.unbound", desc);
        if (desc == "Unknown")
            return lookupLocale("input_settings.trigger.unknown", desc);

        static constexpr std::pair<std::string_view, std::string_view> kSubstitutions[] = {
            {" Double-Click", "input_settings.trigger.double_click"},
            {" Drag", "input_settings.trigger.drag"},
            {"Scroll", "input_settings.trigger.scroll"},
        };
        auto& loc = lfs::event::LocalizationManager::getInstance();
        for (const auto& [needle, key] : kSubstitutions) {
            const auto pos = desc.find(needle);
            if (pos == std::string::npos)
                continue;
            const std::string key_str(key);
            const char* const localized = loc.get(key_str);
            if (localized && std::string_view(localized) != key_str)
                desc.replace(pos, needle.size(), localized);
        }
        return desc;
    }

    std::string InputBindings::getLocalizedTriggerDescription(const Action action,
                                                              const ToolMode mode) const {
        return localizeTriggerDescription(getTriggerDescription(action, mode));
    }

    std::string getKeyName(const int key) {
        if (key >= KEY_F13 && key <= KEY_F24)
            return "F" + std::to_string(13 + key - KEY_F13);
        switch (key) {
        case KEY_A: return "A";
        case KEY_B: return "B";
        case KEY_C: return "C";
        case KEY_D: return "D";
        case KEY_E: return "E";
        case KEY_F: return "F";
        case KEY_G: return "G";
        case KEY_H: return "H";
        case KEY_I: return "I";
        case KEY_J: return "J";
        case KEY_K: return "K";
        case KEY_L: return "L";
        case KEY_M: return "M";
        case KEY_N: return "N";
        case KEY_O: return "O";
        case KEY_P: return "P";
        case KEY_Q: return "Q";
        case KEY_R: return "R";
        case KEY_S: return "S";
        case KEY_T: return "T";
        case KEY_U: return "U";
        case KEY_V: return "V";
        case KEY_W: return "W";
        case KEY_X: return "X";
        case KEY_Y: return "Y";
        case KEY_Z: return "Z";
        case KEY_0: return "0";
        case KEY_1: return "1";
        case KEY_2: return "2";
        case KEY_3: return "3";
        case KEY_4: return "4";
        case KEY_5: return "5";
        case KEY_6: return "6";
        case KEY_7: return "7";
        case KEY_8: return "8";
        case KEY_9: return "9";
        case KEY_SPACE: return "Space";
        case KEY_ENTER: return "Enter";
        case KEY_ESCAPE: return "Escape";
        case KEY_TAB: return "Tab";
        case KEY_BACKSPACE: return "Backspace";
        case KEY_INSERT: return "Insert";
        case KEY_DELETE: return "Delete";
        case KEY_HOME: return "Home";
        case KEY_END: return "End";
        case KEY_PAGE_UP: return "Page Up";
        case KEY_PAGE_DOWN: return "Page Down";
        case KEY_LEFT: return "Left";
        case KEY_RIGHT: return "Right";
        case KEY_UP: return "Up";
        case KEY_DOWN: return "Down";
        case KEY_F1: return "F1";
        case KEY_F2: return "F2";
        case KEY_F3: return "F3";
        case KEY_F4: return "F4";
        case KEY_F5: return "F5";
        case KEY_F6: return "F6";
        case KEY_F7: return "F7";
        case KEY_F8: return "F8";
        case KEY_F9: return "F9";
        case KEY_F10: return "F10";
        case KEY_F11: return "F11";
        case KEY_F12: return "F12";
        case KEY_MINUS: return "-";
        case KEY_EQUAL: return "=";
        case KEY_LEFT_BRACKET: return "[";
        case KEY_RIGHT_BRACKET: return "]";
        case KEY_BACKSLASH: return "\\";
        case KEY_SEMICOLON: return ";";
        case KEY_APOSTROPHE: return "'";
        case KEY_GRAVE_ACCENT: return "`";
        case KEY_COMMA: return ",";
        case KEY_PERIOD: return ".";
        case KEY_SLASH: return "/";
        case KEY_KP_ADD: return "Num+";
        case KEY_KP_SUBTRACT: return "Num-";
        case KEY_KP_MULTIPLY: return "Num*";
        case KEY_KP_DIVIDE: return "Num/";
        case KEY_KP_ENTER: return "NumEnter";
        case KEY_KP_0: return "Num0";
        case KEY_KP_1: return "Num1";
        case KEY_KP_2: return "Num2";
        case KEY_KP_3: return "Num3";
        case KEY_KP_4: return "Num4";
        case KEY_KP_5: return "Num5";
        case KEY_KP_6: return "Num6";
        case KEY_KP_7: return "Num7";
        case KEY_KP_8: return "Num8";
        case KEY_KP_9: return "Num9";
        default: return "Key" + std::to_string(key);
        }
    }

    std::string getMouseButtonName(const MouseButton button) {
        switch (button) {
        case MouseButton::LEFT: return "LMB";
        case MouseButton::RIGHT: return "RMB";
        case MouseButton::MIDDLE: return "MMB";
        case MouseButton::BACK:
            return lfs::event::LocalizationManager::getInstance().get("input_settings.mouse_back");
        case MouseButton::FORWARD:
            return lfs::event::LocalizationManager::getInstance().get("input_settings.mouse_forward");
        default: return "Mouse?";
        }
    }

    std::string getModifierString(const int modifiers) {
        std::string result;
        if (modifiers & MODIFIER_CTRL) {
            result += "Ctrl";
        }
        if (modifiers & MODIFIER_ALT) {
            if (!result.empty())
                result += "+";
            result += "Alt";
        }
        if (modifiers & MODIFIER_SHIFT) {
            if (!result.empty())
                result += "+";
            result += "Shift";
        }
        if (modifiers & MODIFIER_SUPER) {
            if (!result.empty())
                result += "+";
            result += "Super";
        }
        return result;
    }

    std::optional<SelectionOp> selectionOpForModifiers(
        const InputBindings& bindings,
        const ToolMode mode,
        const int modifiers,
        const std::vector<int>& held_keys) {
        switch (bindings.getActionForDrag(mode, MouseButton::LEFT, modifiers, held_keys)) {
        case Action::SELECTION_ADD: return SelectionOp::Add;
        case Action::SELECTION_REMOVE: return SelectionOp::Remove;
        case Action::SELECTION_INTERSECT: return SelectionOp::Intersect;
        default: return std::nullopt;
        }
    }

    void InputBindings::startCapture(ToolMode mode, Action action) {
        capture_state_ = CaptureState{};
        capture_state_.active = true;
        capture_state_.mode = mode;
        capture_state_.action = action;
    }

    void InputBindings::cancelCapture() {
        capture_state_ = CaptureState{};
    }

    void InputBindings::captureKey(int key, int mods) {
        captureKey(key, key, mods);
    }

    void InputBindings::captureKey(const int physical_key, const int logical_key, const int mods) {
        if (!capture_state_.active)
            return;

        if (mods & KEYMOD_ALTGR) {
            capture_state_.altgr_rejected = true;
            return;
        }

        const auto& descriptor = describe(capture_state_.action);
        int key = descriptor.prefers_physical_key ? physical_key : logical_key;
        if (key == KEY_UNKNOWN) {
            key = (logical_key != KEY_UNKNOWN) ? logical_key : physical_key;
        }
        if (key == KEY_UNKNOWN)
            return;

        // Escape cancels capture except when capturing its own Cancel action.
        if (key == KEY_ESCAPE && capture_state_.action != Action::CANCEL_POLYGON &&
            capture_state_.action != Action::EXIT_WALK && capture_state_.action != Action::EXIT_VR) {
            cancelCapture();
            return;
        }

        if (key == KEY_LEFT_SHIFT || key == KEY_RIGHT_SHIFT ||
            key == KEY_LEFT_CONTROL || key == KEY_RIGHT_CONTROL ||
            key == KEY_LEFT_ALT || key == KEY_RIGHT_ALT ||
            key == KEY_LEFT_SUPER || key == KEY_RIGHT_SUPER) {
            return;
        }

        if (!(descriptor.allowed_kinds & TRIGGER_KIND_KEY)) {
            return;
        }

        // OS-reserved combos: don't let users shadow window-manager shortcuts.
        const bool alt_held = (mods & MODIFIER_ALT) != 0;
        const bool super_held = (mods & MODIFIER_SUPER) != 0;
        if (super_held) {
            return;
        }
        if (alt_held && (key == KEY_TAB || key == KEY_F4)) {
            return;
        }

        const auto repeat_for_action = [&](const std::vector<Binding>& candidates)
            -> std::optional<bool> {
            for (const auto& binding : candidates) {
                if (binding.mode == capture_state_.mode &&
                    binding.action == capture_state_.action) {
                    if (const auto* prior = std::get_if<KeyTrigger>(&binding.trigger))
                        return prior->on_repeat;
                }
            }
            return std::nullopt;
        };
        auto repeat = repeat_for_action(bindings_);
        if (!repeat)
            repeat = repeat_for_action(detail::defaultProfile().bindings);
        const bool on_repeat = repeat.value_or(false);
        const KeyTrigger trigger{key, mods, on_repeat};
        if (!isValidTriggerForAction(capture_state_.action, trigger))
            return;
        setBinding(capture_state_.mode, capture_state_.action, trigger);
        capture_state_.captured = trigger;
        capture_state_.active = false;
    }

    bool InputBindings::takeAltGrCaptureRejection() {
        const bool rejected = capture_state_.altgr_rejected;
        capture_state_.altgr_rejected = false;
        return rejected;
    }

    void InputBindings::captureMouseButton(int button, int mods, std::optional<int> chord_key) {
        captureMouseButton(button, mods, 0.0, 0.0, chord_key);
        capture_state_.has_pending_mouse_position = false;
    }

    void InputBindings::captureMouseButton(int button, int mods, double x, double y, std::optional<int> chord_key) {
        if (!capture_state_.active)
            return;

        const uint8_t allowed = describe(capture_state_.action).allowed_kinds;
        if (!(allowed & (TRIGGER_KIND_MOUSE_BUTTON | TRIGGER_KIND_MOUSE_DRAG))) {
            return;
        }

        if (capture_state_.waiting_for_double_click) {
            if (button == capture_state_.pending_button && mods == capture_state_.pending_mods) {
                if (!(allowed & TRIGGER_KIND_MOUSE_BUTTON)) {
                    return;
                }
                const auto mouse_btn = static_cast<MouseButton>(button);
                const MouseButtonTrigger trigger{mouse_btn, mods, true};
                setBinding(capture_state_.mode, capture_state_.action, trigger);
                capture_state_.captured = trigger;
                capture_state_.active = false;
                capture_state_.waiting_for_double_click = false;
                capture_state_.pending_button = -1;
                capture_state_.pending_button_down = false;
                capture_state_.has_pending_mouse_position = false;
                capture_state_.pending_chord_key.reset();
                return;
            }
        }

        capture_state_.waiting_for_double_click = true;
        capture_state_.pending_button = button;
        capture_state_.pending_mods = mods;
        capture_state_.pending_chord_key = chord_key;
        capture_state_.pending_button_down = true;
        capture_state_.has_pending_mouse_position = true;
        capture_state_.pending_mouse_x = x;
        capture_state_.pending_mouse_y = y;
        capture_state_.first_click_time = std::chrono::steady_clock::now();
    }

    void InputBindings::captureMouseButtonRelease(int button) {
        if (!capture_state_.active || !capture_state_.waiting_for_double_click)
            return;

        if (button == capture_state_.pending_button) {
            capture_state_.pending_button_down = false;
        }
    }

    void InputBindings::captureMouseMove(double x, double y) {
        if (!capture_state_.active ||
            !capture_state_.waiting_for_double_click ||
            !capture_state_.pending_button_down ||
            !capture_state_.has_pending_mouse_position) {
            return;
        }

        const auto& descriptor = describe(capture_state_.action);
        if (!(descriptor.allowed_kinds & TRIGGER_KIND_MOUSE_DRAG)) {
            return;
        }

        const double dx = x - capture_state_.pending_mouse_x;
        const double dy = y - capture_state_.pending_mouse_y;
        const double threshold = CaptureState::DRAG_CAPTURE_THRESHOLD_PX;
        if (dx * dx + dy * dy < threshold * threshold) {
            return;
        }

        const auto mouse_btn = static_cast<MouseButton>(capture_state_.pending_button);
        const MouseDragTrigger trigger{mouse_btn, capture_state_.pending_mods,
                                       capture_state_.pending_chord_key};
        setBinding(capture_state_.mode, capture_state_.action, trigger);
        capture_state_.captured = trigger;
        capture_state_.active = false;
        capture_state_.waiting_for_double_click = false;
        capture_state_.pending_button = -1;
        capture_state_.pending_button_down = false;
        capture_state_.has_pending_mouse_position = false;
        capture_state_.pending_chord_key.reset();
    }

    void InputBindings::captureScroll(int mods, std::optional<int> chord_key) {
        if (!capture_state_.active)
            return;
        if (!(describe(capture_state_.action).allowed_kinds & TRIGGER_KIND_MOUSE_SCROLL)) {
            return;
        }
        MouseScrollTrigger trigger{mods & MODIFIER_MASK, chord_key};
        setBinding(capture_state_.mode, capture_state_.action, trigger);
        capture_state_.captured = trigger;
        capture_state_.active = false;
    }

    void InputBindings::updateCapture() {
        if (!capture_state_.active || !capture_state_.waiting_for_double_click)
            return;

        const auto now = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(now - capture_state_.first_click_time).count();

        if (elapsed >= CaptureState::DOUBLE_CLICK_WAIT_TIME) {
            const auto mouse_btn = static_cast<MouseButton>(capture_state_.pending_button);
            const auto& descriptor = describe(capture_state_.action);
            const bool can_button = descriptor.allowed_kinds & TRIGGER_KIND_MOUSE_BUTTON;
            const bool can_drag = descriptor.allowed_kinds & TRIGGER_KIND_MOUSE_DRAG;
            if (capture_state_.pending_button_down &&
                capture_state_.has_pending_mouse_position &&
                can_drag) {
                return;
            }
            const bool produce_button = can_button && (!can_drag || descriptor.prefers_single_click);

            if (produce_button) {
                const MouseButtonTrigger trigger{mouse_btn, capture_state_.pending_mods, false};
                setBinding(capture_state_.mode, capture_state_.action, trigger);
                capture_state_.captured = trigger;
            } else if (can_drag) {
                const MouseDragTrigger trigger{mouse_btn, capture_state_.pending_mods,
                                               capture_state_.pending_chord_key};
                setBinding(capture_state_.mode, capture_state_.action, trigger);
                capture_state_.captured = trigger;
            }
            capture_state_.active = false;
            capture_state_.waiting_for_double_click = false;
            capture_state_.pending_button = -1;
            capture_state_.pending_button_down = false;
            capture_state_.has_pending_mouse_position = false;
            capture_state_.pending_chord_key.reset();
        }
    }

    std::optional<InputTrigger> InputBindings::getAndClearCaptured() {
        const auto result = capture_state_.captured;
        capture_state_.captured.reset();
        return result;
    }

    std::vector<std::pair<Action, std::string>> InputBindings::getBindingsForMode(ToolMode mode) const {
        std::vector<std::pair<Action, std::string>> result;
        for (const auto& binding : bindings_) {
            if (binding.mode == mode) {
                result.emplace_back(binding.action, getTriggerDescription(binding.action, mode));
            }
        }
        return result;
    }

    const ActionDescriptor& describe(const Action action) {
        using K = TriggerKindFlag;

        static constexpr ActionDescriptor d_none{};

        // Camera continuous-axis (scroll only).
        static constexpr ActionDescriptor d_zoom{
            .allowed_kinds = K::TRIGGER_KIND_MOUSE_SCROLL,
            .inherits_from_global = true,
            .ui_section = ActionSection::Navigation,
        };
        static constexpr ActionDescriptor d_roll{
            .allowed_kinds = K::TRIGGER_KIND_MOUSE_SCROLL,
            .inherits_from_global = true,
            .ui_section = ActionSection::Navigation,
        };
        // Orbit / pan: prefer drag, but accept a single mouse-button capture and
        // tolerate extra modifiers when nothing more specific is bound.
        static constexpr ActionDescriptor d_orbit_pan{
            .allowed_kinds = K::TRIGGER_KIND_MOUSE_DRAG | K::TRIGGER_KIND_MOUSE_BUTTON,
            .allows_extra_modifiers = true,
            .prefers_single_click = true,
            .inherits_from_global = true,
            .ui_section = ActionSection::Navigation,
        };
        static constexpr ActionDescriptor d_set_pivot{
            .allowed_kinds = K::TRIGGER_KIND_MOUSE_BUTTON,
            .allows_extra_modifiers = true,
            .prefers_single_click = true,
            .inherits_from_global = true,
            .ui_section = ActionSection::Navigation,
        };
        static constexpr ActionDescriptor d_movement{
            .allowed_kinds = K::TRIGGER_KIND_KEY,
            .prefers_physical_key = true,
            .inherits_from_global = true,
            .ui_section = ActionSection::Navigation,
        };
        static constexpr ActionDescriptor d_camera_key{
            .allowed_kinds = K::TRIGGER_KIND_KEY,
            .inherits_from_global = true,
            .repeats_on_hold = true,
            .ui_section = ActionSection::Navigation,
        };
        static constexpr ActionDescriptor d_camera_global_key{
            .allowed_kinds = K::TRIGGER_KIND_KEY,
            .inherits_from_global = true,
            .ui_section = ActionSection::NavigationGlobal,
        };

        // Selection drags (LMB by default).
        static constexpr ActionDescriptor d_selection_drag{
            .allowed_kinds = K::TRIGGER_KIND_MOUSE_DRAG | K::TRIGGER_KIND_MOUSE_BUTTON,
            .ui_section = ActionSection::Selection,
        };
        static constexpr ActionDescriptor d_selection_mode_key{
            .allowed_kinds = K::TRIGGER_KIND_KEY,
            .inherits_from_global = true,
            .ui_section = ActionSection::SelectionGlobal,
        };

        // Depth filter actions.
        static constexpr ActionDescriptor d_depth_scroll{
            .allowed_kinds = K::TRIGGER_KIND_MOUSE_SCROLL,
            .ui_section = ActionSection::Depth,
        };
        static constexpr ActionDescriptor d_depth_key{
            .allowed_kinds = K::TRIGGER_KIND_KEY,
            .ui_section = ActionSection::Depth,
        };

        // Editing — these inherit from GLOBAL when active mode lacks the binding.
        static constexpr ActionDescriptor d_editing_inherit{
            .allowed_kinds = K::TRIGGER_KIND_KEY,
            .inherits_from_global = true,
            .ui_section = ActionSection::Editing,
        };
        static constexpr ActionDescriptor d_editing_local{
            .allowed_kinds = K::TRIGGER_KIND_KEY,
            .ui_section = ActionSection::Editing,
        };
        static constexpr ActionDescriptor d_selection_modal_local_control{
            .allowed_kinds = K::TRIGGER_KIND_KEY |
                             K::TRIGGER_KIND_MOUSE_BUTTON |
                             K::TRIGGER_KIND_MOUSE_DRAG,
            .allows_extra_modifiers = true,
            .prefers_single_click = true,
            .ui_section = ActionSection::SelectionGlobal,
        };
        static constexpr ActionDescriptor d_selection_modal_inherit_control{
            .allowed_kinds = K::TRIGGER_KIND_KEY |
                             K::TRIGGER_KIND_MOUSE_BUTTON |
                             K::TRIGGER_KIND_MOUSE_DRAG,
            .allows_extra_modifiers = true,
            .inherits_from_global = true,
            .ui_section = ActionSection::SelectionGlobal,
        };
        static constexpr ActionDescriptor d_polygon_confirm{
            .allowed_kinds = K::TRIGGER_KIND_KEY,
            .allows_extra_modifiers = true,
            .ui_section = ActionSection::SelectionGlobal,
        };

        static constexpr ActionDescriptor d_brush_scroll{
            .allowed_kinds = K::TRIGGER_KIND_MOUSE_SCROLL,
            .ui_section = ActionSection::Selection,
        };

        static constexpr ActionDescriptor d_crop_box_key{
            .allowed_kinds = K::TRIGGER_KIND_KEY,
            .ui_section = ActionSection::CropBox,
        };
        static constexpr ActionDescriptor d_crop_box_click{
            .allowed_kinds = K::TRIGGER_KIND_MOUSE_BUTTON,
            .prefers_single_click = true,
            .ui_section = ActionSection::CropBox,
        };
        static constexpr ActionDescriptor d_align_key{
            .allowed_kinds = K::TRIGGER_KIND_KEY,
            .ui_section = ActionSection::Align,
        };

        static constexpr ActionDescriptor d_view_global_key{
            .allowed_kinds = K::TRIGGER_KIND_KEY,
            .inherits_from_global = true,
            .ui_section = ActionSection::ViewGlobal,
        };
        static constexpr ActionDescriptor d_tools_key{
            .allowed_kinds = K::TRIGGER_KIND_KEY,
            .inherits_from_global = true,
            .ui_section = ActionSection::Tools,
        };
        static constexpr ActionDescriptor d_sequencer_key{
            .allowed_kinds = K::TRIGGER_KIND_KEY,
            .inherits_from_global = true,
            .ui_section = ActionSection::Sequencer,
        };
        static constexpr ActionDescriptor d_ui_key{
            .allowed_kinds = K::TRIGGER_KIND_KEY,
            .inherits_from_global = true,
            .ui_section = ActionSection::UI,
        };
        static constexpr ActionDescriptor d_ui_scroll{
            .allowed_kinds = K::TRIGGER_KIND_MOUSE_SCROLL,
            .inherits_from_global = true,
            .ui_section = ActionSection::UI,
        };

        static constexpr ActionDescriptor d_node_pick{
            .allowed_kinds = K::TRIGGER_KIND_MOUSE_BUTTON,
            .allows_extra_modifiers = true,
            .ui_section = ActionSection::NodePicking,
        };
        static constexpr ActionDescriptor d_node_rect{
            .allowed_kinds = K::TRIGGER_KIND_MOUSE_DRAG,
            .allows_extra_modifiers = true,
            .ui_section = ActionSection::NodePicking,
        };

        switch (action) {
        case Action::NONE: return d_none;

        case Action::CAMERA_ORBIT:
        case Action::CAMERA_PAN:
            return d_orbit_pan;
        case Action::CAMERA_ZOOM:
            return d_zoom;
        case Action::CAMERA_ROLL:
            return d_roll;
        case Action::CAMERA_SET_PIVOT:
            return d_set_pivot;

        case Action::CAMERA_MOVE_FORWARD:
        case Action::CAMERA_MOVE_BACKWARD:
        case Action::CAMERA_MOVE_LEFT:
        case Action::CAMERA_MOVE_RIGHT:
        case Action::CAMERA_MOVE_UP:
        case Action::CAMERA_MOVE_DOWN:
            return d_movement;

        case Action::CAMERA_RESET_HOME:
        case Action::CAMERA_SET_HOME:
        case Action::CAMERA_RESTORE_HOME:
        case Action::EXIT_WALK:
        case Action::VR_SNAP_TURN_LEFT:
        case Action::VR_SNAP_TURN_RIGHT:
        case Action::VR_RECENTER:
        case Action::EXIT_VR:
        case Action::CAMERA_FOCUS_SELECTION:
        case Action::CAMERA_NEXT_VIEW:
        case Action::CAMERA_PREV_VIEW:
            return d_camera_global_key;
        case Action::CAMERA_SPEED_UP:
        case Action::CAMERA_SPEED_DOWN:
        case Action::ZOOM_SPEED_UP:
        case Action::ZOOM_SPEED_DOWN:
            return d_camera_key;

        case Action::TOGGLE_SPLIT_VIEW:
        case Action::TOGGLE_INDEPENDENT_SPLIT_VIEW:
        case Action::TOGGLE_GT_COMPARISON:
        case Action::TOGGLE_CAMERA_FRUSTUMS:
        case Action::TOGGLE_GRID:
        case Action::CYCLE_PLY:
        case Action::CYCLE_SELECTION_VIS:
            return d_view_global_key;

        case Action::DELETE_SELECTED:
        case Action::DELETE_NODE:
            return d_editing_local;
        case Action::UNDO:
        case Action::REDO:
        case Action::INVERT_SELECTION:
        case Action::DESELECT_ALL:
        case Action::SELECT_ALL:
        case Action::COPY_SELECTION:
        case Action::CUT_SELECTION:
        case Action::PASTE_SELECTION:
            return d_editing_inherit;

        case Action::TOGGLE_DEPTH_MODE:
        case Action::TOGGLE_SELECTION_DEPTH_FILTER:
        case Action::TOGGLE_SELECTION_CROP_FILTER:
            return d_depth_key;
        case Action::DEPTH_ADJUST_FAR:
        case Action::DEPTH_ADJUST_NEAR:
        case Action::DEPTH_ADJUST_SIDE:
            return d_depth_scroll;

        case Action::BRUSH_RESIZE:
            return d_brush_scroll;

        case Action::CONFIRM_POLYGON:
            return d_polygon_confirm;
        case Action::UNDO_POLYGON_VERTEX:
            return d_selection_modal_local_control;
        case Action::CANCEL_POLYGON:
            return d_selection_modal_inherit_control;

        case Action::SELECTION_REPLACE:
        case Action::SELECTION_ADD:
        case Action::SELECTION_REMOVE:
        case Action::SELECTION_INTERSECT:
            return d_selection_drag;
        case Action::SELECT_MODE_CENTERS:
        case Action::SELECT_MODE_RECTANGLE:
        case Action::SELECT_MODE_POLYGON:
        case Action::SELECT_MODE_LASSO:
        case Action::SELECT_MODE_RINGS:
        case Action::SELECT_MODE_COLOR:
        case Action::SELECT_MODE_BOX:
        case Action::SELECT_MODE_SPHERE:
            return d_selection_mode_key;

        case Action::APPLY_CROP_BOX:
            return d_crop_box_key;
        case Action::INSERT_CROP_POLYGON_VERTEX:
        case Action::REMOVE_CROP_POLYGON_VERTEX:
            return d_crop_box_click;
        case Action::REMOVE_LAST_POINT:
        case Action::APPLY_ALIGNMENT:
            return d_align_key;

        case Action::NODE_PICK:
            return d_node_pick;
        case Action::NODE_RECT_SELECT:
            return d_node_rect;

        case Action::TOGGLE_UI:
        case Action::NEXT_SIDEBAR_CATEGORY:
        case Action::PREVIOUS_SIDEBAR_CATEGORY:
        case Action::TOGGLE_FULLSCREEN:
        case Action::TOGGLE_PERFORMANCE_HUD:
        case Action::OPEN_PREFERENCES:
        case Action::TOGGLE_MCP_SERVER:
        case Action::TOGGLE_MCP_BINDING:
        case Action::SELECT_ALL_SCENE_NODES:
        case Action::TOGGLE_SCENE_SELECTION_VISIBILITY:
        case Action::TOGGLE_SCENE_SELECTION_TRAINING:
        case Action::GROUP_SELECTED_SCENE_NODES:
        case Action::ASSET_GALLERY_PRIMARY:
        case Action::ASSET_GALLERY_COPY_LINK:
        case Action::ASSET_REFRESH:
        case Action::TOGGLE_PYTHON_CONSOLE:
        case Action::UNGROUP_SELECTED_SCENE_NODE:
        case Action::EDITOR_TOGGLE_KIND:
        case Action::EDITOR_RUN:
        case Action::EDITOR_STOP:
            return d_ui_key;
        case Action::HISTOGRAM_ZOOM_MARKED:
            return d_ui_scroll;

        case Action::SEQUENCER_ADD_KEYFRAME:
        case Action::SEQUENCER_UPDATE_KEYFRAME:
        case Action::SEQUENCER_PLAY_PAUSE:
            return d_sequencer_key;

        case Action::TOOL_SELECT:
        case Action::TOOL_TRANSLATE:
        case Action::TOOL_ROTATE:
        case Action::TOOL_SCALE:
        case Action::TOOL_MIRROR:
        case Action::TOOL_ALIGN:
        case Action::TOOL_SCALE_DISTANCE:
            return d_tools_key;

        case Action::PIE_MENU:
            return d_ui_key;
        }
        return d_none;
    }

    ShortcutScope shortcutScopeForAction(const Action action) {
        switch (action) {
        case Action::TOGGLE_MCP_SERVER:
        case Action::TOGGLE_MCP_BINDING:
            return ShortcutScope::Global;

        case Action::CAMERA_SET_PIVOT:
        case Action::CAMERA_NEXT_VIEW:
        case Action::CAMERA_PREV_VIEW:
        case Action::PIE_MENU:
        case Action::EXIT_WALK:
            return ShortcutScope::Viewport;

        default:
            // Scene, tool, edit, framing, home and camera navigation shortcuts
            // yield to a focused text widget.
            return ShortcutScope::GlobalWhenNotTextEditing;
        }
    }

} // namespace lfs::vis::input
