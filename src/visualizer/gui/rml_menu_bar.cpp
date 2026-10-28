/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rml_menu_bar.hpp"
#include "core/event_bridge/event_bridge.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/services.hpp"
#include "gui/panel_registry.hpp"
#include "gui/rmlui/rml_document_utils.hpp"
#include "gui/rmlui/rml_theme.hpp"
#include "gui/rmlui/rml_tooltip.hpp"
#include "gui/rmlui/rmlui_manager.hpp"
#include "input/input_controller.hpp"
#include "input/sdl_key_mapping.hpp"
#include "internal/resource_paths.hpp"
#include "ipc/view_context.hpp"
#include "operator/operator_registry.hpp"
#include "preferences.hpp"
#include "python/python_runtime.hpp"
#include "rendering/dirty_flags.hpp"
#include "rendering/rendering_manager.hpp"
#include "rendering/rendering_types.hpp"
#include "scene/scene_manager.hpp"
#include "theme/theme.hpp"
#include "training/training_manager.hpp"
#include "visualizer/app_store.hpp"
#include "visualizer/visualizer.hpp"
#include "vr/openxr_service.hpp"
#include "window/vulkan_context.hpp"
#include "window/window_manager.hpp"

#include <RmlUi/Core.h>
#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/ElementText.h>
#include <RmlUi/Core/ElementUtilities.h>
#include <SDL3/SDL_keycode.h>
#include <SDL3/SDL_video.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <format>
#include <glm/glm.hpp>
#include <string_view>

namespace lfs::vis::gui {

    namespace {
        constexpr std::string_view kVrMenu = "lichtfeld.native_vr";

        vr::OpenXrService* vrService() {
            auto* window = services().windowOrNull();
            auto* context = window ? window->getVulkanContext() : nullptr;
            return context ? &context->vr() : nullptr;
        }

        std::vector<MenuItemDesc> vrMenuItems() {
            std::vector<MenuItemDesc> items;
            auto* vr = vrService();
            auto& preferences = UserPreferences::instance();
            const auto add = [&](std::string label, int callback, bool enabled = true,
                                 bool selected = false, std::string tooltip = {}) {
                MenuItemDesc item;
                item.type = callback == 0 ? MenuItemDesc::Type::Toggle : MenuItemDesc::Type::Item;
                item.label = std::move(label);
                item.callback_index = callback;
                item.enabled = enabled;
                item.selected = selected;
                item.tooltip = std::move(tooltip);
                items.push_back(std::move(item));
            };
            add("Enable VR Support (restart required)", 0, true, preferences.vrEnabled(),
                "Connect the headset and select VDXR or SteamVR as your OpenXR runtime before restarting.");
            auto* scene = services().sceneOrNull();
            auto* trainer = services().trainerOrNull();
            const bool eligible = scene && !scene->hasDataset() &&
                                  scene->getScene().getTotalGaussianCount() > 0 &&
                                  (!trainer || (!trainer->isTrainingActive() && !trainer->isCompletionPending())) &&
                                  !op::operators().hasModalOperator();
            const bool active = vr && (vr->active() || vr->requested());
            add(active ? "Exit VR" : "Enter VR", 1, vr && (active || (vr->prepared() && eligible)),
                false, "View-only navigation. Switch the trained model to edit mode first. Escape exits VR.");
            add("Recenter", 2, vr && vr->active());
            add(vr ? vr->status() : "VR unavailable", -1, false);
            add("Resolution: 50%", 3, !active, false, "Applies on the next Enter VR.");
            add("Resolution: 70%", 4, !active);
            add("Resolution: 100%", 5, !active);
            add(std::format("Scene units per meter: {:.3f}", preferences.vrSceneScale()), -1, false);
            add("Scene scale: halve", 6, !active);
            add("Scene scale: double", 7, !active);
            add(std::format("Move speed: {:.2f} m/s", preferences.vrMoveSpeed()), -1, false);
            add("Move slower", 8);
            add("Move faster", 9);
            return items;
        }

        void executeVrMenuCallback(const int index) {
            auto& preferences = UserPreferences::instance();
            auto* vr = vrService();
            switch (index) {
            case 0: preferences.setVrEnabled(!preferences.vrEnabled()); break;
            case 1:
                if (vr)
                    vr->request(!(vr->active() || vr->requested()));
                break;
            case 2:
                if (vr)
                    vr->recenter();
                break;
            case 3: preferences.setVrResolutionScale(0.5f); break;
            case 4: preferences.setVrResolutionScale(0.7f); break;
            case 5: preferences.setVrResolutionScale(1.0f); break;
            case 6: preferences.setVrSceneScale(preferences.vrSceneScale() * 0.5f); break;
            case 7: preferences.setVrSceneScale(preferences.vrSceneScale() * 2.0f); break;
            case 8: preferences.setVrMoveSpeed(preferences.vrMoveSpeed() * 0.5f); break;
            case 9: preferences.setVrMoveSpeed(preferences.vrMoveSpeed() * 2.0f); break;
            default: break;
            }
            if (auto* window = services().windowOrNull())
                window->wakeEventLoop();
        }
        float untruncatedContentWidth(Rml::Element* element) {
            assert(element);

            float children_width = 0.0f;
            for (int i = 0; i < element->GetNumChildren(); ++i) {
                auto* const child = element->GetChild(i);
                if (!child || child->GetDisplay() == Rml::Style::Display::None)
                    continue;
                // Text nodes have no box of their own; an access-key label is text, span, text.
                if (auto* const text = dynamic_cast<Rml::ElementText*>(child)) {
                    children_width += static_cast<float>(Rml::ElementUtilities::GetStringWidth(text, text->GetText()));
                    continue;
                }

                const auto position = child->GetComputedValues().position();
                if (position == Rml::Style::Position::Absolute ||
                    position == Rml::Style::Position::Fixed) {
                    continue;
                }

                const auto& child_box = child->GetBox();
                const float outer_width = child_box.GetSize(Rml::BoxArea::Margin).x;
                const float frame_width = outer_width - child_box.GetSize().x;
                children_width +=
                    std::max(outer_width, untruncatedContentWidth(child) + frame_width);
            }

            const bool clips =
                element->GetComputedValues().overflow_x() != Rml::Style::Overflow::Visible;
            const float overflow_width =
                clips ? std::max(0.0f, element->GetScrollWidth() - element->GetClientWidth())
                      : 0.0f;
            const float own_width =
                children_width > 0.0f ? children_width : element->GetBox().GetSize().x;
            return own_width + overflow_width;
        }

        void collectDirectMenuRows(Rml::Element* element, Rml::ElementList& rows) {
            assert(element);

            for (int i = 0; i < element->GetNumChildren(); ++i) {
                auto* const child = element->GetChild(i);
                if (!child || child->GetDisplay() == Rml::Style::Display::None)
                    continue;
                if (child->IsClassSet("submenu-popup"))
                    continue;
                if (child->IsClassSet("menu-item")) {
                    rows.push_back(child);
                    continue;
                }
                collectDirectMenuRows(child, rows);
            }
        }

        float menuRowContentWidth(Rml::Element* row) {
            assert(row);

            const auto& row_box = row->GetBox();
            float width =
                row_box.GetEdge(Rml::BoxArea::Padding, Rml::BoxEdge::Left) +
                row_box.GetEdge(Rml::BoxArea::Padding, Rml::BoxEdge::Right) +
                row_box.GetEdge(Rml::BoxArea::Border, Rml::BoxEdge::Left) +
                row_box.GetEdge(Rml::BoxArea::Border, Rml::BoxEdge::Right);

            for (int i = 0; i < row->GetNumChildren(); ++i) {
                auto* const child = row->GetChild(i);
                if (!child || child->GetDisplay() == Rml::Style::Display::None)
                    continue;
                if (!child->IsClassSet("label") && !child->IsClassSet("shortcut") &&
                    !child->IsClassSet("checkmark") && !child->IsClassSet("submenu-arrow")) {
                    continue;
                }

                const auto& child_box = child->GetBox();
                const float frame_width = child_box.GetSize(Rml::BoxArea::Margin).x -
                                          child_box.GetSize().x;
                float content_width = untruncatedContentWidth(child);
                // A shortcut cell keeps the popup's shared shortcut width.
                if (child->IsClassSet("shortcut"))
                    content_width = std::max(content_width, child_box.GetSize().x);
                width += content_width + frame_width;
            }
            return width;
        }

        // Gives the shortcut cells of one popup the widest shortcut's width, so the shortcuts form
        // one left-aligned column at the popup edge and the checkmarks before them line up.
        bool shareShortcutWidth(Rml::Element* popup) {
            Rml::ElementList rows;
            collectDirectMenuRows(popup, rows);
            std::vector<Rml::Element*> cells;
            float width = 0.0f;
            for (auto* const row : rows) {
                for (int i = 0; i < row->GetNumChildren(); ++i) {
                    auto* const child = row->GetChild(i);
                    if (!child || !child->IsClassSet("shortcut") || child->GetDisplay() == Rml::Style::Display::None)
                        continue;
                    cells.push_back(child);
                    // An empty cell reports its own box, which already carries the shared width.
                    if (child->GetNumChildren() > 0)
                        width = std::max(width, untruncatedContentWidth(child));
                }
            }
            width = std::ceil(width);
            bool changed = false;
            for (auto* const cell : cells) {
                if (const auto* const current = cell->GetLocalProperty(Rml::PropertyId::MinWidth);
                    current && current->unit == Rml::Unit::PX && current->Get<float>() == width)
                    continue;
                cell->SetProperty(Rml::PropertyId::MinWidth, Rml::Property(width, Rml::Unit::PX));
                changed = true;
            }
            return changed;
        }

        float popupContentWidth(Rml::Element* popup) {
            Rml::ElementList rows;
            collectDirectMenuRows(popup, rows);

            float width = 0.0f;
            for (auto* const row : rows)
                width = std::max(width, menuRowContentWidth(row));
            return width;
        }

        // Toggle rows show a checkmark before the shortcut. In a popup with a toggle, every action
        // row gets the checkmark slot and, when the popup has shortcuts, a shortcut cell, so the
        // checkmarks share a column and the shortcuts still end at the popup edge.
        template <typename View>
        std::vector<View> withSharedCheckmarkColumn(std::vector<View> views) {
            if (std::none_of(views.begin(), views.end(), [](const View& view) { return view.show_checkmark; }))
                return views;
            const bool has_shortcuts =
                std::any_of(views.begin(), views.end(), [](const View& view) { return view.has_shortcut; });
            for (auto& view : views) {
                if constexpr (requires { view.is_label; }) {
                    if (view.is_label)
                        continue;
                }
                if constexpr (requires { view.has_children; }) {
                    if (view.has_children)
                        continue;
                }
                view.show_checkmark = true;
                view.has_shortcut = view.has_shortcut || has_shortcuts;
            }
            return views;
        }

        MenuDropdownLeafView makeLeafItemView(const MenuItemDesc& item) {
            MenuDropdownLeafView view;
            view.label = item.label;
            view.tooltip = item.tooltip;
            view.enabled = item.enabled;
            view.callback_index = item.callback_index;

            switch (item.type) {
            case MenuItemDesc::Type::Operator:
                view.action = "operator";
                view.operator_id = item.operator_id;
                view.has_shortcut = !item.shortcut.empty();
                view.shortcut = item.shortcut;
                break;
            case MenuItemDesc::Type::Toggle:
                view.action = "callback";
                view.has_shortcut = !item.shortcut.empty();
                view.shortcut = item.shortcut;
                view.show_checkmark = true;
                view.checkmark = item.selected ? "\u2713" : "";
                break;
            case MenuItemDesc::Type::ShortcutItem:
            case MenuItemDesc::Type::Item:
                view.action = "callback";
                view.has_shortcut = !item.shortcut.empty();
                view.shortcut = item.shortcut;
                break;
            case MenuItemDesc::Type::Separator:
            case MenuItemDesc::Type::SubMenuBegin:
            case MenuItemDesc::Type::SubMenuEnd:
                break;
            }

            return view;
        }

        MenuDropdownLeafView makeSubmenuLabelView(const MenuItemDesc& item) {
            MenuDropdownLeafView view;
            view.label = item.label;
            view.tooltip = item.tooltip;
            view.enabled = false;
            view.is_label = true;
            return view;
        }

        MenuDropdownChildView makeChildItemView(const MenuItemDesc& item) {
            const auto leaf = makeLeafItemView(item);
            return MenuDropdownChildView{
                .label = leaf.label,
                .action = leaf.action,
                .operator_id = leaf.operator_id,
                .shortcut = leaf.shortcut,
                .checkmark = leaf.checkmark,
                .tooltip = leaf.tooltip,
                .enabled = leaf.enabled,
                .has_shortcut = leaf.has_shortcut,
                .show_checkmark = leaf.show_checkmark,
                .callback_index = leaf.callback_index,
            };
        }

        std::vector<MenuDropdownLeafView> buildLeafMenuItems(
            const std::vector<MenuItemDesc>& items,
            std::size_t& pos,
            bool& warned_excess_depth) {
            std::vector<MenuDropdownLeafView> result;
            bool separator_before = false;

            while (pos < items.size()) {
                const auto& item = items[pos];
                switch (item.type) {
                case MenuItemDesc::Type::Separator:
                    separator_before = true;
                    ++pos;
                    break;
                case MenuItemDesc::Type::SubMenuBegin: {
                    if (!warned_excess_depth) {
                        warned_excess_depth = true;
                        LOG_WARN("RmlMenuBar: nested submenu depth > 3 is flattened in the retained Rml menu model.");
                    }
                    MenuDropdownLeafView label_view = makeSubmenuLabelView(item);
                    label_view.index = static_cast<int>(result.size());
                    label_view.separator_before = separator_before;
                    separator_before = false;
                    result.push_back(std::move(label_view));
                    ++pos;

                    auto nested_items = buildLeafMenuItems(items, pos, warned_excess_depth);
                    for (auto& nested : nested_items)
                        result.push_back(std::move(nested));
                    break;
                }
                case MenuItemDesc::Type::SubMenuEnd:
                    ++pos;
                    return result;
                case MenuItemDesc::Type::Operator:
                case MenuItemDesc::Type::Toggle:
                case MenuItemDesc::Type::ShortcutItem:
                case MenuItemDesc::Type::Item: {
                    MenuDropdownLeafView view = makeLeafItemView(item);
                    view.index = static_cast<int>(result.size());
                    view.separator_before = separator_before;
                    separator_before = false;
                    result.push_back(std::move(view));
                    ++pos;
                    break;
                }
                }
            }

            return result;
        }

        MenuDropdownGrandchildView makeGrandchildItemView(const MenuItemDesc& item) {
            const auto leaf = makeLeafItemView(item);
            return MenuDropdownGrandchildView{
                .label = leaf.label,
                .action = leaf.action,
                .operator_id = leaf.operator_id,
                .shortcut = leaf.shortcut,
                .checkmark = leaf.checkmark,
                .tooltip = leaf.tooltip,
                .enabled = leaf.enabled,
                .has_shortcut = leaf.has_shortcut,
                .show_checkmark = leaf.show_checkmark,
                .callback_index = leaf.callback_index,
            };
        }

        std::vector<MenuDropdownGrandchildView> buildGrandchildMenuItems(
            const std::vector<MenuItemDesc>& items,
            std::size_t& pos,
            bool& warned_excess_depth) {
            std::vector<MenuDropdownGrandchildView> result;
            bool separator_before = false;

            while (pos < items.size()) {
                const auto& item = items[pos];
                switch (item.type) {
                case MenuItemDesc::Type::Separator:
                    separator_before = true;
                    ++pos;
                    break;
                case MenuItemDesc::Type::SubMenuBegin: {
                    MenuDropdownGrandchildView view;
                    view.index = static_cast<int>(result.size());
                    view.label = item.label;
                    view.tooltip = item.tooltip;
                    view.has_children = true;
                    view.separator_before = separator_before;
                    separator_before = false;
                    ++pos;
                    view.children = withSharedCheckmarkColumn(buildLeafMenuItems(items, pos, warned_excess_depth));
                    result.push_back(std::move(view));
                    break;
                }
                case MenuItemDesc::Type::SubMenuEnd:
                    ++pos;
                    return result;
                case MenuItemDesc::Type::Operator:
                case MenuItemDesc::Type::Toggle:
                case MenuItemDesc::Type::ShortcutItem:
                case MenuItemDesc::Type::Item: {
                    MenuDropdownGrandchildView view = makeGrandchildItemView(item);
                    view.index = static_cast<int>(result.size());
                    view.separator_before = separator_before;
                    separator_before = false;
                    result.push_back(std::move(view));
                    ++pos;
                    break;
                }
                }
            }

            return result;
        }

        std::vector<MenuDropdownChildView> buildChildMenuItems(
            const std::vector<MenuItemDesc>& items,
            std::size_t& pos,
            bool& warned_excess_depth) {
            std::vector<MenuDropdownChildView> result;
            bool separator_before = false;

            while (pos < items.size()) {
                const auto& item = items[pos];
                switch (item.type) {
                case MenuItemDesc::Type::Separator:
                    separator_before = true;
                    ++pos;
                    break;
                case MenuItemDesc::Type::SubMenuBegin: {
                    MenuDropdownChildView view;
                    view.index = static_cast<int>(result.size());
                    view.label = item.label;
                    view.tooltip = item.tooltip;
                    view.has_children = true;
                    view.separator_before = separator_before;
                    separator_before = false;
                    ++pos;
                    view.children = withSharedCheckmarkColumn(buildGrandchildMenuItems(items, pos, warned_excess_depth));
                    result.push_back(std::move(view));
                    break;
                }
                case MenuItemDesc::Type::SubMenuEnd:
                    ++pos;
                    return result;
                case MenuItemDesc::Type::Operator:
                case MenuItemDesc::Type::Toggle:
                case MenuItemDesc::Type::ShortcutItem:
                case MenuItemDesc::Type::Item: {
                    MenuDropdownChildView view = makeChildItemView(item);
                    view.index = static_cast<int>(result.size());
                    view.separator_before = separator_before;
                    separator_before = false;
                    result.push_back(std::move(view));
                    ++pos;
                    break;
                }
                }
            }

            return result;
        }

        std::vector<MenuDropdownRootView> buildRootMenuItems(const std::vector<MenuItemDesc>& items) {
            std::vector<MenuDropdownRootView> result;
            std::size_t pos = 0;
            bool separator_before = false;
            bool warned_excess_depth = false;

            while (pos < items.size()) {
                const auto& item = items[pos];
                switch (item.type) {
                case MenuItemDesc::Type::Separator:
                    separator_before = true;
                    ++pos;
                    break;
                case MenuItemDesc::Type::SubMenuEnd:
                    ++pos;
                    break;
                case MenuItemDesc::Type::SubMenuBegin: {
                    MenuDropdownRootView view;
                    view.index = static_cast<int>(result.size());
                    view.label = item.label;
                    view.tooltip = item.tooltip;
                    view.has_children = true;
                    view.separator_before = separator_before;
                    separator_before = false;
                    ++pos;
                    view.children = withSharedCheckmarkColumn(buildChildMenuItems(items, pos, warned_excess_depth));
                    result.push_back(std::move(view));
                    break;
                }
                case MenuItemDesc::Type::Operator:
                case MenuItemDesc::Type::Toggle:
                case MenuItemDesc::Type::ShortcutItem:
                case MenuItemDesc::Type::Item: {
                    MenuDropdownRootView view;
                    view.index = static_cast<int>(result.size());
                    const auto leaf = makeLeafItemView(item);
                    view.label = leaf.label;
                    view.action = leaf.action;
                    view.operator_id = leaf.operator_id;
                    view.shortcut = leaf.shortcut;
                    view.checkmark = leaf.checkmark;
                    view.tooltip = leaf.tooltip;
                    view.enabled = leaf.enabled;
                    view.separator_before = separator_before;
                    view.has_shortcut = leaf.has_shortcut;
                    view.show_checkmark = leaf.show_checkmark;
                    view.callback_index = leaf.callback_index;
                    separator_before = false;
                    result.push_back(std::move(view));
                    ++pos;
                    break;
                }
                }
            }

            return result;
        }

        bool isAccessWordChar(char32_t codepoint) {
            if (codepoint < 128)
                return (codepoint >= '0' && codepoint <= '9') || (codepoint >= 'A' && codepoint <= 'Z') ||
                       (codepoint >= 'a' && codepoint <= 'z');
            return codepoint != 0;
        }

        char normalizeAsciiKey(char32_t codepoint) {
            if (codepoint >= 'A' && codepoint <= 'Z')
                return static_cast<char>(codepoint);
            if (codepoint >= 'a' && codepoint <= 'z')
                return static_cast<char>(codepoint - 'a' + 'A');
            if (codepoint >= '0' && codepoint <= '9')
                return static_cast<char>(codepoint);
            return 0;
        }

        struct AccessCandidate {
            std::size_t offset = 0;
            std::size_t length = 0;
            char key = 0;
            bool word_start = false;
        };

        struct AccessChoice {
            bool present = false;
            std::size_t offset = 0;
            std::size_t length = 0;
            char key = 0;
        };

        std::vector<AccessCandidate> accessCandidates(std::string_view label) {
            std::vector<AccessCandidate> result;
            const Rml::String text(label.data(), label.size());
            char32_t previous = 0;
            bool have_previous = false;
            for (Rml::StringIteratorU8 it(text); it; ++it) {
                const std::size_t start = static_cast<std::size_t>(it.offset());
                const char32_t codepoint = static_cast<char32_t>(*it);
                Rml::StringIteratorU8 next = it;
                ++next;
                const std::size_t end = next ? static_cast<std::size_t>(next.offset()) : text.size();
                const char key = normalizeAsciiKey(codepoint);
                if (key != 0) {
                    const bool word_start = !have_previous || !isAccessWordChar(previous);
                    result.push_back({start, end - start, key, word_start});
                }
                previous = codepoint;
                have_previous = true;
            }
            return result;
        }

        AccessChoice explicitAccessChoice(std::string_view label, char key) {
            for (const auto& candidate : accessCandidates(label)) {
                if (candidate.key == key)
                    return {true, candidate.offset, candidate.length, key};
            }
            return {};
        }

        AccessChoice automaticAccessChoice(std::string_view label, std::array<bool, 128>& used) {
            const auto candidates = accessCandidates(label);
            const auto free = [&](char key) { return !used[static_cast<unsigned char>(key)]; };
            for (const auto& candidate : candidates) {
                if (candidate.word_start && free(candidate.key)) {
                    used[static_cast<unsigned char>(candidate.key)] = true;
                    return {true, candidate.offset, candidate.length, candidate.key};
                }
            }
            for (const auto& candidate : candidates) {
                if (free(candidate.key)) {
                    used[static_cast<unsigned char>(candidate.key)] = true;
                    return {true, candidate.offset, candidate.length, candidate.key};
                }
            }
            // Every ASCII letter and digit is taken. Share the preferred word-start so a
            // repeated label cycles instead of growing a second unique key.
            for (const auto& candidate : candidates) {
                if (candidate.word_start)
                    return {true, candidate.offset, candidate.length, candidate.key};
            }
            return {};
        }

        template <typename View>
        void clearAccessKey(View& view) {
            view.access.label_before = view.label;
            view.access.access_mark.clear();
            view.access.label_after.clear();
            view.access.access_key = 0;
        }

        template <typename View>
        void applyAccessChoice(View& view, const AccessChoice& choice, bool parentheses_if_missing, char fallback_key) {
            if (choice.present) {
                view.access.label_before.assign(view.label, 0, choice.offset);
                view.access.access_mark.assign(view.label, choice.offset, choice.length);
                view.access.label_after.assign(view.label, choice.offset + choice.length, std::string::npos);
                view.access.access_key = choice.key;
                return;
            }
            if (parentheses_if_missing && fallback_key != 0) {
                view.access.label_before = view.label + "(";
                view.access.access_mark.assign(1, fallback_key);
                view.access.label_after = ")";
                view.access.access_key = fallback_key;
                return;
            }
            clearAccessKey(view);
        }

        template <typename View>
        void assignSiblingAccessKeys(std::vector<View>& items) {
            std::array<bool, 128> used{};
            for (auto& item : items) {
                if constexpr (requires { item.is_label; }) {
                    if (item.is_label) {
                        clearAccessKey(item);
                        if constexpr (requires { item.children; })
                            assignSiblingAccessKeys(item.children);
                        continue;
                    }
                }
                applyAccessChoice(item, automaticAccessChoice(item.label, used), false, 0);
                if constexpr (requires { item.children; })
                    assignSiblingAccessKeys(item.children);
            }
        }

        const char* builtinMenuToken(std::string_view idname) {
            if (idname == "lfs_plugins.file_menu.FileMenu")
                return "file";
            if (idname == "lfs_plugins.edit_menu.EditMenu")
                return "edit";
            if (idname == "lfs_plugins.select_menu.SelectMenu")
                return "select";
            if (idname == "lfs_plugins.tools_menu.ToolsMenu")
                return "tools";
            if (idname == "lfs_plugins.view_menu.ViewMenu")
                return "view";
            return nullptr;
        }

        char explicitMenuLetter(const char* token) {
            const std::string key = std::string("menu.access_key.") + token;
            auto& locales = lfs::event::LocalizationManager::getInstance();
            if (!locales.hasKey(key))
                return 0;
            const char* value = locales.get(key);
            if (value == nullptr || value[0] == '\0' || value[1] != '\0')
                return 0;
            return normalizeAsciiKey(static_cast<unsigned char>(value[0]));
        }

        void assignTopLevelAccessKeys(std::vector<MenuLabelView>& labels, const std::vector<std::string>& idnames) {
            std::array<bool, 128> used{};
            std::vector<char> reserved(labels.size(), 0);
            for (std::size_t i = 0; i < labels.size(); ++i) {
                const char* token = builtinMenuToken(i < idnames.size() ? std::string_view{idnames[i]} : std::string_view{});
                if (token == nullptr)
                    continue;
                const char key = explicitMenuLetter(token);
                if (key == 0 || used[static_cast<unsigned char>(key)])
                    continue;
                reserved[i] = key;
                used[static_cast<unsigned char>(key)] = true;
            }
            for (std::size_t i = 0; i < labels.size(); ++i) {
                if (reserved[i] != 0) {
                    applyAccessChoice(labels[i], explicitAccessChoice(labels[i].label, reserved[i]), true, reserved[i]);
                    continue;
                }
                applyAccessChoice(labels[i], automaticAccessChoice(labels[i].label, used), false, 0);
            }
        }

        bool altLetterBoundInActiveMap(int app_key) {
            using lfs::vis::input::Action;
            using lfs::vis::input::kAllToolModes;
            using lfs::vis::input::KEYMOD_ALT;
            const auto* controller = lfs::vis::InputController::instance();
            if (controller == nullptr)
                return false;
            const auto& bindings = controller->getBindings();
            for (const auto mode : kAllToolModes) {
                if (bindings.getActionForKey(mode, app_key, KEYMOD_ALT) != Action::NONE)
                    return true;
            }
            return false;
        }

        bool eventHasAltGr(const FrameInputEvent& event) {
            const int mods = lfs::vis::input::sdlModsToAppMods(static_cast<SDL_Keymod>(event.modifiers));
            return (mods & lfs::vis::input::KEYMOD_ALTGR) != 0;
        }
    } // namespace

    float RmlMenuBar::barHeight() const {
        const float dp = rml_manager_ ? rml_manager_->getDpRatio() : 1.0f;
        return bar_height_ * dp;
    }

    void RmlMenuBar::bindModel() {
        auto ctor = rml_context_->CreateDataModel("menu_bar");
        assert(ctor);

        if (auto handle = ctor.RegisterStruct<AccessKeyLabel>()) {
            handle.RegisterMember("label_before", &AccessKeyLabel::label_before);
            handle.RegisterMember("access_mark", &AccessKeyLabel::access_mark);
            handle.RegisterMember("label_after", &AccessKeyLabel::label_after);
        }
        if (auto handle = ctor.RegisterStruct<MenuLabelView>()) {
            handle.RegisterMember("index", &MenuLabelView::index);
            handle.RegisterMember("label", &MenuLabelView::label);
            handle.RegisterMember("access", &MenuLabelView::access);
            handle.RegisterMember("active", &MenuLabelView::active);
            handle.RegisterMember("logo", &MenuLabelView::logo);
        }
        if (auto handle = ctor.RegisterStruct<MenuDropdownLeafView>()) {
            handle.RegisterMember("index", &MenuDropdownLeafView::index);
            handle.RegisterMember("label", &MenuDropdownLeafView::label);
            handle.RegisterMember("access", &MenuDropdownLeafView::access);
            handle.RegisterMember("action", &MenuDropdownLeafView::action);
            handle.RegisterMember("operator_id", &MenuDropdownLeafView::operator_id);
            handle.RegisterMember("shortcut", &MenuDropdownLeafView::shortcut);
            handle.RegisterMember("checkmark", &MenuDropdownLeafView::checkmark);
            handle.RegisterMember("tooltip", &MenuDropdownLeafView::tooltip);
            handle.RegisterMember("enabled", &MenuDropdownLeafView::enabled);
            handle.RegisterMember("separator_before", &MenuDropdownLeafView::separator_before);
            handle.RegisterMember("has_shortcut", &MenuDropdownLeafView::has_shortcut);
            handle.RegisterMember("show_checkmark", &MenuDropdownLeafView::show_checkmark);
            handle.RegisterMember("is_label", &MenuDropdownLeafView::is_label);
            handle.RegisterMember("keyboard_active", &MenuDropdownLeafView::keyboard_active);
            handle.RegisterMember("callback_index", &MenuDropdownLeafView::callback_index);
        }
        ctor.RegisterArray<std::vector<MenuLabelView>>();
        ctor.RegisterArray<std::vector<MenuDropdownLeafView>>();
        if (auto handle = ctor.RegisterStruct<MenuDropdownGrandchildView>()) {
            handle.RegisterMember("index", &MenuDropdownGrandchildView::index);
            handle.RegisterMember("label", &MenuDropdownGrandchildView::label);
            handle.RegisterMember("access", &MenuDropdownGrandchildView::access);
            handle.RegisterMember("action", &MenuDropdownGrandchildView::action);
            handle.RegisterMember("operator_id", &MenuDropdownGrandchildView::operator_id);
            handle.RegisterMember("shortcut", &MenuDropdownGrandchildView::shortcut);
            handle.RegisterMember("checkmark", &MenuDropdownGrandchildView::checkmark);
            handle.RegisterMember("tooltip", &MenuDropdownGrandchildView::tooltip);
            handle.RegisterMember("enabled", &MenuDropdownGrandchildView::enabled);
            handle.RegisterMember("separator_before", &MenuDropdownGrandchildView::separator_before);
            handle.RegisterMember("has_shortcut", &MenuDropdownGrandchildView::has_shortcut);
            handle.RegisterMember("show_checkmark", &MenuDropdownGrandchildView::show_checkmark);
            handle.RegisterMember("has_children", &MenuDropdownGrandchildView::has_children);
            handle.RegisterMember("submenu_open", &MenuDropdownGrandchildView::submenu_open);
            handle.RegisterMember("keyboard_active", &MenuDropdownGrandchildView::keyboard_active);
            handle.RegisterMember("callback_index", &MenuDropdownGrandchildView::callback_index);
            handle.RegisterMember("children", &MenuDropdownGrandchildView::children);
        }
        ctor.RegisterArray<std::vector<MenuDropdownGrandchildView>>();
        if (auto handle = ctor.RegisterStruct<MenuDropdownChildView>()) {
            handle.RegisterMember("index", &MenuDropdownChildView::index);
            handle.RegisterMember("label", &MenuDropdownChildView::label);
            handle.RegisterMember("access", &MenuDropdownChildView::access);
            handle.RegisterMember("action", &MenuDropdownChildView::action);
            handle.RegisterMember("operator_id", &MenuDropdownChildView::operator_id);
            handle.RegisterMember("shortcut", &MenuDropdownChildView::shortcut);
            handle.RegisterMember("checkmark", &MenuDropdownChildView::checkmark);
            handle.RegisterMember("tooltip", &MenuDropdownChildView::tooltip);
            handle.RegisterMember("enabled", &MenuDropdownChildView::enabled);
            handle.RegisterMember("separator_before", &MenuDropdownChildView::separator_before);
            handle.RegisterMember("has_shortcut", &MenuDropdownChildView::has_shortcut);
            handle.RegisterMember("show_checkmark", &MenuDropdownChildView::show_checkmark);
            handle.RegisterMember("has_children", &MenuDropdownChildView::has_children);
            handle.RegisterMember("submenu_open", &MenuDropdownChildView::submenu_open);
            handle.RegisterMember("keyboard_active", &MenuDropdownChildView::keyboard_active);
            handle.RegisterMember("callback_index", &MenuDropdownChildView::callback_index);
            handle.RegisterMember("children", &MenuDropdownChildView::children);
        }
        ctor.RegisterArray<std::vector<MenuDropdownChildView>>();
        if (auto handle = ctor.RegisterStruct<MenuDropdownRootView>()) {
            handle.RegisterMember("label", &MenuDropdownRootView::label);
            handle.RegisterMember("access", &MenuDropdownRootView::access);
            handle.RegisterMember("index", &MenuDropdownRootView::index);
            handle.RegisterMember("action", &MenuDropdownRootView::action);
            handle.RegisterMember("operator_id", &MenuDropdownRootView::operator_id);
            handle.RegisterMember("shortcut", &MenuDropdownRootView::shortcut);
            handle.RegisterMember("checkmark", &MenuDropdownRootView::checkmark);
            handle.RegisterMember("tooltip", &MenuDropdownRootView::tooltip);
            handle.RegisterMember("enabled", &MenuDropdownRootView::enabled);
            handle.RegisterMember("separator_before", &MenuDropdownRootView::separator_before);
            handle.RegisterMember("has_shortcut", &MenuDropdownRootView::has_shortcut);
            handle.RegisterMember("show_checkmark", &MenuDropdownRootView::show_checkmark);
            handle.RegisterMember("has_children", &MenuDropdownRootView::has_children);
            handle.RegisterMember("submenu_open", &MenuDropdownRootView::submenu_open);
            handle.RegisterMember("keyboard_active", &MenuDropdownRootView::keyboard_active);
            handle.RegisterMember("callback_index", &MenuDropdownRootView::callback_index);
            handle.RegisterMember("children", &MenuDropdownRootView::children);
        }
        ctor.RegisterArray<std::vector<MenuDropdownRootView>>();
        if (auto handle = ctor.RegisterStruct<MenuToolbarButtonView>()) {
            handle.RegisterMember("button_id", &MenuToolbarButtonView::button_id);
            handle.RegisterMember("action", &MenuToolbarButtonView::action);
            handle.RegisterMember("value", &MenuToolbarButtonView::value);
            handle.RegisterMember("icon_src", &MenuToolbarButtonView::icon_src);
            handle.RegisterMember("tooltip_key", &MenuToolbarButtonView::tooltip_key);
            handle.RegisterMember("tooltip_text", &MenuToolbarButtonView::tooltip_text);
            handle.RegisterMember("selected", &MenuToolbarButtonView::selected);
            handle.RegisterMember("enabled", &MenuToolbarButtonView::enabled);
        }
        ctor.RegisterArray<std::vector<MenuToolbarButtonView>>();
        ctor.Bind("menu_labels", &menu_labels_);
        ctor.Bind("brand_logo_src", &brand_logo_src_);
        ctor.Bind("dropdown_items", &dropdown_items_);
        ctor.Bind("menu_camera_buttons", &camera_buttons_);
        ctor.Bind("menu_render_buttons", &render_buttons_);
        ctor.Bind("menu_projection_buttons", &projection_buttons_);
        ctor.Bind("portal_connection_label", &portal_connection_label_);
        ctor.Bind("portal_connection_tooltip", &portal_connection_tooltip_);
        ctor.Bind("portal_connection_icon", &portal_connection_icon_);
        ctor.Bind("portal_connection_tone", &portal_connection_tone_);
        ctor.Bind("gallery_progress_label", &gallery_progress_label_);
        ctor.Bind("gallery_progress_detail", &gallery_progress_detail_);
        ctor.Bind("gallery_progress_tooltip", &gallery_progress_tooltip_);
        ctor.Bind("gallery_progress_width", &gallery_progress_width_);
        ctor.Bind("gallery_has_progress", &gallery_has_progress_);
        ctor.Bind("gallery_progress_indeterminate", &gallery_progress_indeterminate_);
        ctor.Bind("project_title", &project_title_);
        ctor.Bind("project_tooltip", &project_tooltip_);
        ctor.Bind("project_dirty", &project_dirty_);
        menu_model_ = ctor.GetModelHandle();
    }

    void RmlMenuBar::init(RmlUIManager* mgr) {
        assert(mgr);
        rml_manager_ = mgr;

        rml_context_ = rml_manager_->createContext("menu_bar", 800, 30);
        if (!rml_context_) {
            LOG_ERROR("RmlMenuBar: failed to create RML context");
            return;
        }

        bindModel();

        try {
            const auto rml_path = lfs::vis::getAssetPath("rmlui/menubar.rml");
            document_ = rml_documents::loadDocument(rml_context_, rml_path);
            if (!document_) {
                LOG_ERROR("RmlMenuBar: failed to load menubar.rml");
                return;
            }
            document_->Show();
        } catch (const std::exception& e) {
            LOG_ERROR("RmlMenuBar: resource not found: {}", e.what());
            return;
        }

        menu_items_ = document_->GetElementById("menu-items");
        dropdown_overlay_ = document_->GetElementById("dropdown-overlay");
        dropdown_container_ = document_->GetElementById("dropdown-container");
        dropdown_popup_ = document_->GetElementById("dropdown-popup");
        project_title_container_ = document_->GetElementById("project-title");
        project_title_el_ = document_->GetElementById("project-title-content");
        menu_toolbar_ = document_->GetElementById("menu-toolbar");
        menu_window_controls_ = document_->GetElementById("menu-window-controls");
        menu_window_split_view_ = document_->GetElementById("menu-window-split-view");
        menu_window_toggle_ui_ = document_->GetElementById("menu-window-toggle-ui");
        menu_window_maximize_ = document_->GetElementById("menu-window-maximize");
        body_el_ = document_->GetElementById("body");

        render_needed_ = true;
        updateTheme();
        window_focus_lost_handler_id_ = lfs::core::events::internal::WindowFocusLost::when(
            [this](const auto&) { onWindowFocusLost(); });
    }

    void RmlMenuBar::shutdown() {
        if (window_focus_lost_handler_id_ != 0) {
            lfs::event::EventBridge::instance().unsubscribe(
                typeid(lfs::core::events::internal::WindowFocusLost), window_focus_lost_handler_id_);
            window_focus_lost_handler_id_ = 0;
        }
        menu_model_ = {};
        menu_labels_.clear();
        dropdown_items_.clear();
        camera_buttons_.clear();
        render_buttons_.clear();
        projection_buttons_.clear();
        open_menu_idname_.clear();
        if (rml_manager_)
            rml_manager_->releaseCachedVulkanContext(direct_cache_);
        if (rml_context_ && rml_manager_)
            rml_manager_->destroyContext("menu_bar");
        rml_context_ = nullptr;
        document_ = nullptr;
        menu_items_ = nullptr;
        dropdown_container_ = nullptr;
        dropdown_popup_ = nullptr;
        dropdown_overlay_ = nullptr;
        project_title_container_ = nullptr;
        project_title_el_ = nullptr;
        menu_toolbar_ = nullptr;
        menu_window_controls_ = nullptr;
        menu_window_split_view_ = nullptr;
        menu_window_toggle_ui_ = nullptr;
        menu_window_maximize_ = nullptr;
        body_el_ = nullptr;
        project_title_has_room_ = false;
        applied_project_title_left_ = -1.0f;
        applied_project_title_width_ = -1.0f;
        last_window_split_view_ = false;
        last_ui_hidden_ = false;
        last_window_maximized_ = false;
        clearTitlebarDragRegion();
    }

    void RmlMenuBar::suspend() {
        portal_transfer_animation_active_ = false;
        wants_input_ = false;
        mouse_pos_valid_ = false;
        last_mouse_x_ = 0;
        last_mouse_y_ = 0;
        last_hovered_label_ = -1;
        last_toolbar_hovered_ = false;
        clearTitlebarDragRegion();
        tooltip_.setHover({}, nullptr);

        if (open_menu_index_ >= 0)
            closeDropdown();
    }

    void RmlMenuBar::reloadResources() {
        if (!rml_context_)
            return;

        if (open_menu_index_ >= 0)
            closeDropdown();

        if (document_) {
            rml_context_->UnloadDocument(document_);
            rml_context_->Update();
        }

        document_ = nullptr;
        menu_items_ = nullptr;
        dropdown_container_ = nullptr;
        dropdown_popup_ = nullptr;
        dropdown_overlay_ = nullptr;
        project_title_container_ = nullptr;
        project_title_el_ = nullptr;
        menu_toolbar_ = nullptr;
        menu_window_controls_ = nullptr;
        menu_window_split_view_ = nullptr;
        menu_window_toggle_ui_ = nullptr;
        menu_window_maximize_ = nullptr;
        body_el_ = nullptr;
        project_title_has_room_ = false;
        applied_project_title_left_ = -1.0f;
        applied_project_title_width_ = -1.0f;
        tooltip_.setHover({}, nullptr);
        clearTitlebarDragRegion();
        base_rcss_.clear();
        has_theme_signature_ = false;
        wants_input_ = false;
        render_needed_ = true;
        mouse_pos_valid_ = false;
        last_hovered_label_ = -1;
        last_ctx_w_ = 0;
        last_ctx_h_ = 0;
        last_document_h_ = 0;
        if (rml_manager_)
            rml_manager_->releaseCachedVulkanContext(direct_cache_);

        try {
            const auto rml_path = lfs::vis::getAssetPath("rmlui/menubar.rml");
            document_ = rml_documents::loadDocument(rml_context_, rml_path);
            if (!document_) {
                LOG_ERROR("RmlMenuBar: failed to reload menubar.rml");
                return;
            }
            document_->Show();
        } catch (const std::exception& e) {
            LOG_ERROR("RmlMenuBar: resource not found during reload: {}", e.what());
            return;
        }

        menu_items_ = document_->GetElementById("menu-items");
        dropdown_overlay_ = document_->GetElementById("dropdown-overlay");
        dropdown_container_ = document_->GetElementById("dropdown-container");
        dropdown_popup_ = document_->GetElementById("dropdown-popup");
        project_title_container_ = document_->GetElementById("project-title");
        project_title_el_ = document_->GetElementById("project-title-content");
        menu_toolbar_ = document_->GetElementById("menu-toolbar");
        menu_window_controls_ = document_->GetElementById("menu-window-controls");
        menu_window_split_view_ = document_->GetElementById("menu-window-split-view");
        menu_window_toggle_ui_ = document_->GetElementById("menu-window-toggle-ui");
        menu_window_maximize_ = document_->GetElementById("menu-window-maximize");
        body_el_ = document_->GetElementById("body");
        applied_toolbar_right_ = -1.0f;
        applied_project_title_left_ = -1.0f;
        applied_project_title_width_ = -1.0f;
        toolbar_fits_ = true;
        toolbar_squeezed_ = false;
        toolbar_squeeze_screen_w_ = -1;
        toolbar_squeeze_menus_right_ = -1.0f;
        bar_height_ = 30.0f;
        last_window_split_view_ = false;
        last_ui_hidden_ = false;
        last_window_maximized_ = false;

        rebuildLabels();
        menu_model_.DirtyVariable("dropdown_items");
        menu_model_.DirtyVariable("menu_camera_buttons");
        menu_model_.DirtyVariable("menu_render_buttons");
        menu_model_.DirtyVariable("menu_projection_buttons");
        updateTheme();
    }

    void RmlMenuBar::updateLabels(const std::vector<std::string>& labels,
                                  const std::vector<std::string>& idnames) {
        assert(labels.size() == idnames.size());
        auto all_labels = labels;
        auto all_idnames = idnames;
        if (vr::OpenXrService::compiled()) {
            all_labels.push_back("VR");
            all_idnames.emplace_back(kVrMenu);
        }
        if (all_labels == current_labels_ && all_idnames == current_idnames_)
            return;
        current_labels_ = std::move(all_labels);
        current_idnames_ = std::move(all_idnames);
        render_needed_ = true;

        if (open_menu_index_ >= static_cast<int>(current_idnames_.size())) {
            closeDropdown();
        }

        rebuildLabels();
    }

    void RmlMenuBar::updateProjectDisplay(const ProjectDisplayInfo& project_display) {
        std::string project_title = project_display.title.value_or(std::string{});
        if (project_title.empty() && project_display.path) {
            project_title = lfs::core::path_to_utf8(project_display.path->stem());
        }
        if (project_title.empty()) {
            project_title = "Untitled";
        }
        updateProjectDisplay(
            std::move(project_title),
            project_display.path
                ? lfs::core::path_to_utf8(project_display.path->lexically_normal())
                : std::string{},
            project_display.dirty);
    }

    void RmlMenuBar::updateProjectDisplay(std::string title, std::string tooltip, const bool dirty) {
        if (project_title_ == title && project_tooltip_ == tooltip && project_dirty_ == dirty)
            return;

        project_title_ = std::move(title);
        project_tooltip_ = std::move(tooltip);
        project_dirty_ = dirty;
        menu_model_.DirtyVariable("project_title");
        menu_model_.DirtyVariable("project_tooltip");
        menu_model_.DirtyVariable("project_dirty");
        render_needed_ = true;
    }

    void RmlMenuBar::rebuildLabels() {
        menu_labels_.clear();
        menu_labels_.reserve(current_labels_.size());
        for (size_t i = 0; i < current_labels_.size(); ++i) {
            // Help has no label of its own: the brand logo is its entry.
            const bool logo = current_idnames_[i] == "lfs_plugins.help_menu.HelpMenu";
            menu_labels_.push_back(MenuLabelView{
                .index = static_cast<int>(i),
                .label = logo ? std::string{} : current_labels_[i],
                .active = static_cast<int>(i) == active_index_,
                .logo = logo,
            });
        }
        assignTopLevelAccessKeys(menu_labels_, current_idnames_);
        menu_model_.DirtyVariable("menu_labels");
        render_needed_ = true;
    }

    void RmlMenuBar::syncActiveLabelState() {
        const int display_active = open_menu_index_ >= 0 ? open_menu_index_ : keyboard_focus_menu_index_;
        if (display_active == active_index_)
            return;

        active_index_ = display_active;
        for (size_t i = 0; i < menu_labels_.size(); ++i)
            menu_labels_[i].active = static_cast<int>(i) == active_index_;
        menu_model_.DirtyVariable("menu_labels");
        render_needed_ = true;
    }

    void RmlMenuBar::processInput(const PanelInputState& input) {
        if (!menu_items_ || !document_)
            return;

        // Resolve row changes before the shell reads barHeight() and before hit testing.
        // Vulkan draws this surface after the panels have already been laid out.
        const float layout_dp = rml_manager_->getDpRatio();
        if (input.screen_w != last_ctx_w_ || layout_dp != last_dp_ratio_) {
            updateCompactLayout(input.screen_w, layout_dp);
            rml_context_->SetDimensions({input.screen_w, std::max(1, last_ctx_h_)});
            rml_context_->Update();
            updateToolbarLayout(input.screen_w, layout_dp);
            rml_context_->Update();
        }

        const bool mouse_button_active = input.mouse_clicked[0] || input.mouse_clicked[1] || input.mouse_clicked[2] ||
                                         input.mouse_down[0] || input.mouse_down[1] || input.mouse_down[2];
        if (alt_pressed_ && mouse_button_active)
            alt_chord_ = true;
        // SDL dispatch and the following frame pass both carry the same key. Handle it
        // at dispatch so Escape takes effect before the next event, and skip that same
        // key in the frame pass. A receipt is one delivered event: a key that never
        // reached dispatch (the bar is closed and not the keyboard owner) is still
        // handled here, including when the menu opens or closes mid-batch.
        const bool dispatching = rml_manager_->isDispatchingInput();
        struct ClearDispatchReceipts {
            std::vector<DispatchKeyReceipt>& receipts;
            bool armed = false;
            ~ClearDispatchReceipts() {
                if (armed && !receipts.empty())
                    receipts.clear();
            }
        } clear_receipts{dispatch_key_receipts_, !dispatching};
        const auto already_dispatched = [&](const FrameInputEvent& event) {
            if (event.kind != FrameInputEventKind::KeyDown && event.kind != FrameInputEventKind::KeyUp)
                return false;
            for (auto& receipt : dispatch_key_receipts_) {
                if (receipt.matched || receipt.timestamp != event.timestamp || receipt.kind != event.kind ||
                    receipt.scancode != event.scancode || receipt.repeat != event.repeat)
                    continue;
                receipt.matched = true;
                return true;
            }
            return false;
        };
        for (const auto& event : input.input_events) {
            if (!dispatching && !dispatch_key_receipts_.empty() && already_dispatched(event))
                continue;
            if (dispatching && event.dispatch != nullptr &&
                (event.kind == FrameInputEventKind::KeyDown || event.kind == FrameInputEventKind::KeyUp)) {
                if (dispatch_key_receipts_.capacity() == 0)
                    dispatch_key_receipts_.reserve(8);
                dispatch_key_receipts_.push_back(DispatchKeyReceipt{
                    .timestamp = event.timestamp,
                    .scancode = event.scancode,
                    .kind = event.kind,
                    .repeat = event.repeat,
                });
            }
            const bool alt_key = event.scancode == SDL_SCANCODE_LALT || event.scancode == SDL_SCANCODE_RALT;
            bool window_has_keyboard_focus = true;
            const bool check_window_focus = alt_pressed_ || keyboard_focus_menu_index_ >= 0 || alt_key;
            if (check_window_focus && rml_manager_ && rml_manager_->getWindow()) {
                window_has_keyboard_focus =
                    (SDL_GetWindowFlags(rml_manager_->getWindow()) & SDL_WINDOW_INPUT_FOCUS) != 0;
            }
            if (!window_has_keyboard_focus) {
                onWindowFocusLost();
            }
            if (handleMenuKey(event, mouse_button_active, window_has_keyboard_focus))
                return;
        }
        if (rml_manager_->routeInput(rml_context_, input, [this](const PanelInputState& event) { processInput(event); }, isOpen()))
            return;
        wants_input_ = false;
        if (pointer_suspended_) {
            wants_input_ = isOpen();
            return;
        }
        if (rml_manager_)
            rml_manager_->trackContextFrame(rml_context_, 0, 0);

        const float mx = input.mouse_x - input.screen_x;
        const float my = input.mouse_y - input.screen_y;
        const bool is_open = open_menu_index_ >= 0;
        const int ctx_w = input.screen_w;
        const int ctx_h = (is_open || tooltip_.hasActiveState())
                              ? input.screen_h
                              : static_cast<int>(bar_height_ * rml_manager_->getDpRatio());
        const int rml_mx = static_cast<int>(mx);
        const int rml_my = static_cast<int>(my);
        const bool was_in_context = mouse_pos_valid_ &&
                                    last_mouse_x_ >= 0 && last_mouse_x_ < ctx_w &&
                                    last_mouse_y_ >= 0 && last_mouse_y_ < ctx_h;
        const bool is_in_context = rml_mx >= 0 && rml_mx < ctx_w &&
                                   rml_my >= 0 && rml_my < ctx_h;
        const bool mouse_moved =
            !mouse_pos_valid_ || rml_mx != last_mouse_x_ || rml_my != last_mouse_y_;
        const bool pointer_event =
            input.mouse_clicked[0] || input.mouse_released[0] ||
            input.mouse_clicked[1] || input.mouse_released[1] ||
            input.mouse_clicked[2] || input.mouse_released[2] ||
            input.mouse_wheel != 0.0f;
        const bool pointer_down =
            input.mouse_down[0] || input.mouse_down[1] || input.mouse_down[2];
        const bool context_size_unchanged = ctx_w == last_ctx_w_ && ctx_h == last_ctx_h_;
        if (mouse_pos_valid_ && !mouse_moved && !pointer_event && !pointer_down &&
            context_size_unchanged && !render_needed_) {
            wants_input_ = is_open || last_hovered_label_ >= 0 || last_toolbar_hovered_;
            return;
        }
        if (mouse_moved && (is_open || was_in_context || is_in_context)) {
            mouse_pos_valid_ = true;
            last_mouse_x_ = rml_mx;
            last_mouse_y_ = rml_my;
            render_needed_ = true;
            if (is_in_context)
                rml_context_->ProcessMouseMove(rml_mx, rml_my, 0);
            else
                rml_context_->ProcessMouseLeave();
        }

        const int count = menu_items_->GetNumChildren();
        int hovered_label = -1;
        for (int i = 0; i < count; ++i) {
            auto* child = menu_items_->GetChild(i);
            if (!child)
                continue;
            const auto box = child->GetAbsoluteOffset(Rml::BoxArea::Border);
            const auto size = child->GetBox().GetSize(Rml::BoxArea::Border);
            if (mx >= box.x && mx < box.x + size.x && my >= box.y && my < box.y + size.y) {
                hovered_label = i;
                break;
            }
        }
        last_hovered_label_ = hovered_label;

        Rml::Element* hovered_toolbar_btn =
            (!is_open && hovered_label < 0) ? toolbarButtonAtPoint(mx, my) : nullptr;
        last_toolbar_hovered_ = hovered_toolbar_btn != nullptr;
        if (hovered_toolbar_btn)
            tooltip_.setHover(resolveRmlTooltip(hovered_toolbar_btn), hovered_toolbar_btn);
        else if (!is_open && hovered_label < 0 && projectTitleAtPoint(mx, my))
            tooltip_.setHover(project_tooltip_, project_title_el_);
        else
            tooltip_.setHover({}, nullptr);

        const float dp_ratio = rml_manager_ ? rml_manager_->getDpRatio() : 1.0f;
        const bool in_free_titlebar =
            !is_open &&
            hovered_label < 0 &&
            !hovered_toolbar_btn &&
            my >= 0.0f &&
            my < bar_height_ * dp_ratio;
        if (in_free_titlebar && (input.mouse_clicked[0] || input.mouse_down[0])) {
            wants_input_ = true;
            return;
        }

        if (!is_open && keyboard_focus_menu_index_ >= 0 &&
            (input.mouse_clicked[0] || input.mouse_clicked[1] || input.mouse_clicked[2]) &&
            (my < 0.0f || my >= bar_height_ * dp_ratio)) {
            keyboard_focus_menu_index_ = -1;
            syncActiveLabelState();
            syncAccessKeyVisibility();
        }

        if (is_open) {
            wants_input_ = true;

            if (hovered_label >= 0 && hovered_label != open_menu_index_) {
                openDropdown(hovered_label, keyboard_opened_menu_);
                return;
            }

            Rml::Element* hit_element = nullptr;
            if (hovered_label < 0 && dropdown_container_)
                hit_element = dropdownElementAtPoint(mx, my);
            tooltip_.setHover(resolveRmlTooltip(hit_element), hit_element);
            setOpenSubmenu(
                submenuIndexForElement(hit_element),
                childSubmenuIndexForElement(hit_element),
                grandchildSubmenuIndexForElement(hit_element));
            std::array<int, 4> hovered_path{{-1, -1, -1, -1}};
            int hovered_depth = -1;
            for (auto* row = hit_element; row; row = row->GetParentNode()) {
                if (!row->HasAttribute("data-nav-level"))
                    continue;
                const int level = row->GetAttribute<int>("data-nav-level", -1);
                const int selected = row->GetAttribute<int>("data-nav-index", -1);
                if (level >= 0 && level < 4 && selected >= 0) {
                    hovered_depth = level;
                    hovered_path[level] = selected;
                    for (auto* parent = row; parent; parent = parent->GetParentNode()) {
                        if (parent->HasAttribute("data-root-index"))
                            hovered_path[0] = parent->GetAttribute<int>("data-root-index", -1);
                        if (parent->HasAttribute("data-child-index"))
                            hovered_path[1] = parent->GetAttribute<int>("data-child-index", -1);
                        if (parent->HasAttribute("data-grandchild-index"))
                            hovered_path[2] = parent->GetAttribute<int>("data-grandchild-index", -1);
                    }
                    break;
                }
            }
            if (hovered_depth >= 0 &&
                (keyboard_path_ != hovered_path || keyboard_depth_ != hovered_depth)) {
                keyboard_path_ = hovered_path;
                keyboard_depth_ = hovered_depth;
                updateKeyboardHighlight();
            }

            if (input.mouse_clicked[0]) {
                if (hovered_label >= 0 && hovered_label == open_menu_index_) {
                    closeDropdown();
                    return;
                }

                if (hovered_label < 0 && dropdown_container_) {
                    auto* hit = dropdown_container_;
                    {
                        Rml::Element* clicked = hit_element ? hit_element : dropdownElementAtPoint(mx, my);
                        const bool clicked_submenu =
                            submenuIndexForElement(clicked) >= 0 ||
                            childSubmenuIndexForElement(clicked) >= 0;
                        if (clicked) {
                            while (clicked && clicked != hit) {
                                if (clicked->HasAttribute("data-action")) {
                                    const std::string action = clicked->GetAttribute<Rml::String>("data-action", "");
                                    if (action == "operator") {
                                        const std::string op_id = clicked->GetAttribute<Rml::String>("data-operator-id", "");
                                        if (!op_id.empty() && !clicked->HasAttribute("disabled")) {
                                            op::operators().invoke(op_id);
                                        }
                                        closeDropdown();
                                        return;
                                    } else if (action == "callback") {
                                        const int cb_idx = clicked->GetAttribute<int>("data-callback-index", -1);
                                        if (cb_idx >= 0 && !clicked->HasAttribute("disabled")) {
                                            if (open_menu_idname_ == kVrMenu)
                                                executeVrMenuCallback(cb_idx);
                                            else
                                                python::execute_menu_callback(open_menu_idname_, cb_idx);
                                        }
                                        closeDropdown();
                                        return;
                                    }
                                }
                                clicked = clicked->GetParentNode();
                            }
                        }
                        if (clicked_submenu)
                            return;
                    }

                    closeDropdown();
                    return;
                }
            }
        } else {
            if (hovered_toolbar_btn) {
                wants_input_ = true;
                if (input.mouse_clicked[0] && !hovered_toolbar_btn->IsClassSet("disabled")) {
                    dispatchToolbarAction(
                        hovered_toolbar_btn->GetAttribute<Rml::String>("data-action", ""),
                        hovered_toolbar_btn->GetAttribute<Rml::String>("data-value", ""));
                    return;
                }
            } else if (hovered_label >= 0) {
                wants_input_ = true;
                if (input.mouse_clicked[0]) {
                    openDropdown(hovered_label, false);
                    return;
                }
            }
        }

        // Update active label highlighting
        syncActiveLabelState();
    }

    bool RmlMenuBar::handleMenuKey(const FrameInputEvent& event, const bool mouse_button_active,
                                   const bool window_has_keyboard_focus) {
        const bool alt = event.scancode == SDL_SCANCODE_LALT || event.scancode == SDL_SCANCODE_RALT;
        const bool access_alt = access_alt_held_ ||
                                ((event.modifiers & SDL_KMOD_ALT) != 0 && !eventHasAltGr(event));
        if (!isOpen() && keyboard_focus_menu_index_ < 0 && !alt && !alt_pressed_ && !access_alt)
            return false;
        if (rml_manager_ && rml_manager_->wantsTextInput())
            return false;
        if (event.kind == FrameInputEventKind::KeyDown && alt && !event.repeat) {
            alt_pressed_ = true;
            alt_chord_ = mouse_button_active || !window_has_keyboard_focus;
            if (!eventHasAltGr(event)) {
                access_alt_held_ = true;
                syncAccessKeyVisibility();
            }
            return false;
        }
        if (event.kind == FrameInputEventKind::KeyDown && alt_pressed_ && !alt)
            alt_chord_ = true;
        if (event.kind == FrameInputEventKind::KeyUp && alt) {
            alt_pressed_ = false;
            access_alt_held_ = false;
            const bool focus_menu = !alt_chord_ && window_has_keyboard_focus;
            alt_chord_ = false;
            if (focus_menu && !current_labels_.empty()) {
                keyboard_focus_menu_index_ = 0;
                syncActiveLabelState();
                syncAccessKeyVisibility();
                wants_input_ = true;
                if (event.dispatch)
                    event.dispatch->consumed = true;
                return true;
            }
            syncAccessKeyVisibility();
            return false;
        }
        if (event.kind != FrameInputEventKind::KeyDown)
            return false;
        // A held arrow keeps stepping through entries; every other key acts once per press.
        const bool arrow = event.scancode == SDL_SCANCODE_UP || event.scancode == SDL_SCANCODE_DOWN ||
                           event.scancode == SDL_SCANCODE_LEFT || event.scancode == SDL_SCANCODE_RIGHT;
        if (event.repeat && !arrow)
            return false;

        switch (handleAccessKey(event)) {
        case AccessKeyResult::Handled:
            wants_input_ = true;
            if (event.dispatch)
                event.dispatch->consumed = true;
            return true;
        case AccessKeyResult::Decline:
            return false;
        case AccessKeyResult::Ignore:
            break;
        }

        if (!isOpen()) {
            if (keyboard_focus_menu_index_ < 0)
                return false;
            switch (event.scancode) {
            case SDL_SCANCODE_DOWN:
            case SDL_SCANCODE_RETURN:
            case SDL_SCANCODE_SPACE:
                openDropdown(keyboard_focus_menu_index_, true);
                break;
            case SDL_SCANCODE_LEFT:
                keyboard_focus_menu_index_ = (keyboard_focus_menu_index_ + static_cast<int>(current_labels_.size()) - 1) %
                                             static_cast<int>(current_labels_.size());
                syncActiveLabelState();
                break;
            case SDL_SCANCODE_RIGHT:
                keyboard_focus_menu_index_ = (keyboard_focus_menu_index_ + 1) % static_cast<int>(current_labels_.size());
                syncActiveLabelState();
                break;
            case SDL_SCANCODE_ESCAPE:
                keyboard_focus_menu_index_ = -1;
                syncActiveLabelState();
                syncAccessKeyVisibility();
                break;
            default:
                keyboard_focus_menu_index_ = -1;
                syncActiveLabelState();
                syncAccessKeyVisibility();
                return false;
            }
            wants_input_ = true;
            if (event.dispatch)
                event.dispatch->consumed = true;
            return true;
        }

        // Hover stores the row under the pointer. That row is the parent of a
        // submenu the pointer just opened, so keyboard_depth_ is one short of
        // the open stack. Dismiss one real submenu before the menu itself.
        const auto open_submenu_depth = [this] {
            if (open_submenu_index_ < 0 || open_submenu_index_ >= static_cast<int>(dropdown_items_.size()) ||
                !dropdown_items_[open_submenu_index_].submenu_open)
                return 0;
            const auto& root = dropdown_items_[open_submenu_index_];
            if (open_child_submenu_index_ < 0 || open_child_submenu_index_ >= static_cast<int>(root.children.size()) ||
                !root.children[open_child_submenu_index_].submenu_open)
                return 1;
            const auto& child = root.children[open_child_submenu_index_];
            if (open_grandchild_submenu_index_ < 0 ||
                open_grandchild_submenu_index_ >= static_cast<int>(child.children.size()) ||
                !child.children[open_grandchild_submenu_index_].submenu_open)
                return 2;
            return 3;
        };
        const auto close_one_level = [&] {
            const int next = std::max(open_submenu_depth(), keyboard_depth_) - 1;
            if (keyboard_path_[0] < 0)
                keyboard_path_[0] = open_submenu_index_;
            if (next >= 1 && keyboard_path_[1] < 0)
                keyboard_path_[1] = open_child_submenu_index_;
            if (next >= 2 && keyboard_path_[2] < 0)
                keyboard_path_[2] = open_grandchild_submenu_index_;
            keyboard_depth_ = std::max(next, 0);
            for (int level = keyboard_depth_ + 1; level < 4; ++level)
                keyboard_path_[level] = -1;
            setOpenSubmenu(keyboard_depth_ == 0 ? -1 : keyboard_path_[0],
                           keyboard_depth_ < 2 ? -1 : keyboard_path_[1],
                           keyboard_depth_ < 3 ? -1 : keyboard_path_[2]);
            updateKeyboardHighlight();
        };

        switch (event.scancode) {
        case SDL_SCANCODE_UP: moveKeyboardHighlight(-1); break;
        case SDL_SCANCODE_DOWN: moveKeyboardHighlight(1); break;
        case SDL_SCANCODE_HOME: moveKeyboardHighlight(1, true); break;
        case SDL_SCANCODE_END: moveKeyboardHighlight(-1, true); break;
        case SDL_SCANCODE_RIGHT:
            if (keyboard_depth_ == 0) {
                if (!activateKeyboardHighlight(true)) {
                    const int next = (open_menu_index_ + 1) % static_cast<int>(current_labels_.size());
                    openDropdown(next, true);
                }
            } else {
                activateKeyboardHighlight(true);
            }
            break;
        case SDL_SCANCODE_LEFT:
            if (keyboard_depth_ == 0) {
                const int previous = (open_menu_index_ + static_cast<int>(current_labels_.size()) - 1) %
                                     static_cast<int>(current_labels_.size());
                openDropdown(previous, true);
            } else {
                close_one_level();
            }
            break;
        case SDL_SCANCODE_RETURN:
        case SDL_SCANCODE_SPACE: activateKeyboardHighlight(); break;
        case SDL_SCANCODE_ESCAPE:
            if (keyboard_depth_ > 0 || open_submenu_depth() > 0)
                close_one_level();
            else
                closeDropdown();
            break;
        default: return false;
        }
        wants_input_ = true;
        if (event.dispatch)
            event.dispatch->consumed = true;
        return true;
    }

    void RmlMenuBar::onWindowPointerEvent(const SDL_Event& event, const std::uint32_t main_window_id) {
        if (event.type == SDL_EVENT_WINDOW_MOUSE_LEAVE && event.window.windowID == main_window_id)
            onPointerLeave();
        else if ((event.type == SDL_EVENT_MOUSE_MOTION && event.motion.windowID == main_window_id) ||
                 (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN && event.button.windowID == main_window_id))
            onPointerMotion();
    }

    void RmlMenuBar::onPointerMotion() {
        if (pointer_suspended_) {
            pointer_suspended_ = false;
            mouse_pos_valid_ = false;
        }
    }

    void RmlMenuBar::onPointerLeave() {
        pointer_suspended_ = true;
        if (rml_context_)
            rml_context_->ProcessMouseLeave();
        mouse_pos_valid_ = false;
        last_hovered_label_ = -1;
        last_toolbar_hovered_ = false;
        tooltip_.setHover({}, nullptr);
        render_needed_ = true;
    }

    void RmlMenuBar::onWindowFocusLost() {
        onPointerLeave();
        if (alt_pressed_)
            alt_chord_ = true;
        access_alt_held_ = false;
        if (!isOpen() && keyboard_focus_menu_index_ >= 0) {
            keyboard_focus_menu_index_ = -1;
            syncActiveLabelState();
        }
        syncAccessKeyVisibility();
    }

    void RmlMenuBar::updateKeyboardHighlight() {
        for (auto& root : dropdown_items_) {
            root.keyboard_active = false;
            for (auto& child : root.children) {
                child.keyboard_active = false;
                for (auto& grandchild : child.children) {
                    grandchild.keyboard_active = false;
                    for (auto& leaf : grandchild.children)
                        leaf.keyboard_active = false;
                }
            }
        }
        if (keyboard_path_[0] >= 0 && keyboard_path_[0] < static_cast<int>(dropdown_items_.size())) {
            auto& root = dropdown_items_[keyboard_path_[0]];
            root.keyboard_active = true;
            if (keyboard_depth_ >= 1 && keyboard_path_[1] >= 0 && keyboard_path_[1] < static_cast<int>(root.children.size())) {
                auto& child = root.children[keyboard_path_[1]];
                child.keyboard_active = true;
                if (keyboard_depth_ >= 2 && keyboard_path_[2] >= 0 && keyboard_path_[2] < static_cast<int>(child.children.size())) {
                    auto& grandchild = child.children[keyboard_path_[2]];
                    grandchild.keyboard_active = true;
                    if (keyboard_depth_ >= 3 && keyboard_path_[3] >= 0 && keyboard_path_[3] < static_cast<int>(grandchild.children.size()))
                        grandchild.children[keyboard_path_[3]].keyboard_active = true;
                }
            }
        }
        menu_model_.DirtyVariable("dropdown_items");
        render_needed_ = true;
    }

    void RmlMenuBar::moveKeyboardHighlight(const int direction, const bool to_edge) {
        auto count = [&]() -> int {
            if (keyboard_depth_ == 0)
                return static_cast<int>(dropdown_items_.size());
            if (keyboard_depth_ == 1 && keyboard_path_[0] >= 0)
                return static_cast<int>(dropdown_items_[keyboard_path_[0]].children.size());
            if (keyboard_depth_ == 2 && keyboard_path_[0] >= 0 && keyboard_path_[1] >= 0)
                return static_cast<int>(dropdown_items_[keyboard_path_[0]].children[keyboard_path_[1]].children.size());
            if (keyboard_depth_ == 3 && keyboard_path_[0] >= 0 && keyboard_path_[1] >= 0 && keyboard_path_[2] >= 0)
                return static_cast<int>(dropdown_items_[keyboard_path_[0]].children[keyboard_path_[1]].children[keyboard_path_[2]].children.size());
            return 0;
        };
        auto enabled = [&](const int index) {
            if (keyboard_depth_ == 0)
                return dropdown_items_[index].enabled;
            if (keyboard_depth_ == 1)
                return dropdown_items_[keyboard_path_[0]].children[index].enabled;
            if (keyboard_depth_ == 2)
                return dropdown_items_[keyboard_path_[0]].children[keyboard_path_[1]].children[index].enabled;
            const auto& item = dropdown_items_[keyboard_path_[0]].children[keyboard_path_[1]].children[keyboard_path_[2]].children[index];
            return item.enabled && !item.is_label;
        };
        const int n = count();
        if (n == 0)
            return;
        int index = keyboard_path_[keyboard_depth_];
        if (to_edge)
            index = direction > 0 ? -1 : 0;
        for (int tried = 0; tried < n; ++tried) {
            index = (index + direction + n) % n;
            if (enabled(index)) {
                keyboard_path_[keyboard_depth_] = index;
                for (int i = keyboard_depth_ + 1; i < 4; ++i)
                    keyboard_path_[i] = -1;
                updateKeyboardHighlight();
                return;
            }
        }
    }

    bool RmlMenuBar::activateKeyboardHighlight(const bool submenu_only) {
        const int index = keyboard_path_[keyboard_depth_];
        if (index < 0)
            return false;
        bool has_children = false;
        std::string action, operator_id;
        int callback_index = -1;
        bool enabled = true;
        if (keyboard_depth_ == 0) {
            const auto& item = dropdown_items_[index];
            has_children = item.has_children;
            enabled = item.enabled;
            action = item.action;
            operator_id = item.operator_id;
            callback_index = item.callback_index;
        } else if (keyboard_depth_ == 1) {
            const auto& item = dropdown_items_[keyboard_path_[0]].children[index];
            has_children = item.has_children;
            enabled = item.enabled;
            action = item.action;
            operator_id = item.operator_id;
            callback_index = item.callback_index;
        } else if (keyboard_depth_ == 2) {
            const auto& item = dropdown_items_[keyboard_path_[0]].children[keyboard_path_[1]].children[index];
            has_children = item.has_children;
            enabled = item.enabled;
            action = item.action;
            operator_id = item.operator_id;
            callback_index = item.callback_index;
        } else {
            const auto& item = dropdown_items_[keyboard_path_[0]].children[keyboard_path_[1]].children[keyboard_path_[2]].children[index];
            enabled = item.enabled && !item.is_label;
            action = item.action;
            operator_id = item.operator_id;
            callback_index = item.callback_index;
        }
        if (has_children && keyboard_depth_ < 3) {
            if (keyboard_depth_ == 0)
                setOpenSubmenu(index, -1, -1);
            if (keyboard_depth_ == 1)
                setOpenSubmenu(keyboard_path_[0], index, -1);
            if (keyboard_depth_ == 2)
                setOpenSubmenu(keyboard_path_[0], keyboard_path_[1], index);
            ++keyboard_depth_;
            keyboard_path_[keyboard_depth_] = -1;
            moveKeyboardHighlight(1, true);
            return true;
        }
        if (submenu_only)
            return false;
        if (!enabled)
            return true;
        if (action == "operator") {
            if (!operator_id.empty())
                op::operators().invoke(operator_id);
        } else if (action == "callback") {
            if (callback_index >= 0) {
                if (open_menu_idname_ == kVrMenu)
                    executeVrMenuCallback(callback_index);
                else
                    python::execute_menu_callback(open_menu_idname_, callback_index);
            }
        } else {
            return true;
        }
        closeDropdown();
        return true;
    }

    void RmlMenuBar::syncAccessKeyVisibility() {
        const bool show = access_alt_held_ || (keyboard_focus_menu_index_ >= 0 && open_menu_index_ < 0) ||
                          (open_menu_index_ >= 0 && keyboard_opened_menu_);
        if (show == access_keys_visible_)
            return;
        access_keys_visible_ = show;
        Rml::Element* body = body_el_;
        if (!body && document_)
            body = document_->GetElementById("body");
        if (body)
            body->SetClass("access-keys-visible", show);
        render_needed_ = true;
    }

    RmlMenuBar::AccessKeyResult RmlMenuBar::handleAccessKey(const FrameInputEvent& event) {
        const int app_mods = lfs::vis::input::sdlModsToAppMods(static_cast<SDL_Keymod>(event.modifiers));
        constexpr int blocked = lfs::vis::input::KEYMOD_CTRL | lfs::vis::input::KEYMOD_SHIFT |
                                lfs::vis::input::KEYMOD_SUPER | lfs::vis::input::KEYMOD_ALTGR;
        if ((app_mods & blocked) != 0)
            return AccessKeyResult::Ignore;
        const int app_key = lfs::vis::input::sdlKeycodeToAppKey(event.keycode);
        char letter = 0;
        if (app_key >= lfs::vis::input::KEY_A && app_key <= lfs::vis::input::KEY_Z)
            letter = static_cast<char>(app_key);
        else if (app_key >= lfs::vis::input::KEY_0 && app_key <= lfs::vis::input::KEY_9)
            letter = static_cast<char>(app_key);
        else
            return AccessKeyResult::Ignore;

        if (!isOpen()) {
            const bool plain_alt = access_alt_held_ || (app_mods & lfs::vis::input::KEYMOD_ALT) != 0;
            if (!plain_alt && keyboard_focus_menu_index_ < 0)
                return AccessKeyResult::Ignore;
            if (plain_alt && altLetterBoundInActiveMap(app_key))
                return AccessKeyResult::Decline;
            int found = -1;
            for (std::size_t i = 0; i < menu_labels_.size(); ++i) {
                if (menu_labels_[i].access.access_key == letter) {
                    found = static_cast<int>(i);
                    break;
                }
            }
            if (found < 0)
                return AccessKeyResult::Ignore;
            openDropdown(found, true);
            return AccessKeyResult::Handled;
        }

        const auto level_count = [&]() -> int {
            if (keyboard_depth_ == 0)
                return static_cast<int>(dropdown_items_.size());
            if (keyboard_depth_ == 1 && keyboard_path_[0] >= 0 && keyboard_path_[0] < static_cast<int>(dropdown_items_.size()))
                return static_cast<int>(dropdown_items_[keyboard_path_[0]].children.size());
            if (keyboard_depth_ == 2 && keyboard_path_[0] >= 0 && keyboard_path_[1] >= 0 &&
                keyboard_path_[0] < static_cast<int>(dropdown_items_.size()) &&
                keyboard_path_[1] < static_cast<int>(dropdown_items_[keyboard_path_[0]].children.size()))
                return static_cast<int>(dropdown_items_[keyboard_path_[0]].children[keyboard_path_[1]].children.size());
            if (keyboard_depth_ == 3 && keyboard_path_[0] >= 0 && keyboard_path_[1] >= 0 && keyboard_path_[2] >= 0 &&
                keyboard_path_[0] < static_cast<int>(dropdown_items_.size()) &&
                keyboard_path_[1] < static_cast<int>(dropdown_items_[keyboard_path_[0]].children.size()) &&
                keyboard_path_[2] < static_cast<int>(dropdown_items_[keyboard_path_[0]].children[keyboard_path_[1]].children.size()))
                return static_cast<int>(
                    dropdown_items_[keyboard_path_[0]].children[keyboard_path_[1]].children[keyboard_path_[2]].children.size());
            return 0;
        };
        const auto matches_letter = [&](int index) {
            if (keyboard_depth_ == 0) {
                const auto& item = dropdown_items_[index];
                return item.enabled && item.access.access_key == letter;
            }
            if (keyboard_depth_ == 1) {
                const auto& item = dropdown_items_[keyboard_path_[0]].children[index];
                return item.enabled && item.access.access_key == letter;
            }
            if (keyboard_depth_ == 2) {
                const auto& item = dropdown_items_[keyboard_path_[0]].children[keyboard_path_[1]].children[index];
                return item.enabled && item.access.access_key == letter;
            }
            const auto& item =
                dropdown_items_[keyboard_path_[0]].children[keyboard_path_[1]].children[keyboard_path_[2]].children[index];
            return item.enabled && !item.is_label && item.access.access_key == letter;
        };

        std::vector<int> matches;
        const int count = level_count();
        for (int index = 0; index < count; ++index) {
            if (matches_letter(index))
                matches.push_back(index);
        }
        if (matches.empty())
            return AccessKeyResult::Ignore;
        int chosen = matches.front();
        if (matches.size() > 1) {
            const int current = keyboard_path_[keyboard_depth_];
            for (const int index : matches) {
                if (index > current) {
                    chosen = index;
                    break;
                }
            }
        }
        keyboard_path_[keyboard_depth_] = chosen;
        for (int level = keyboard_depth_ + 1; level < 4; ++level)
            keyboard_path_[level] = -1;
        updateKeyboardHighlight();
        if (matches.size() == 1)
            activateKeyboardHighlight();
        return AccessKeyResult::Handled;
    }

    void RmlMenuBar::openDropdown(int index, const bool from_keyboard) {
        assert(index >= 0 && index < static_cast<int>(current_idnames_.size()));

        if (open_menu_index_ < 0)
            rml_manager_->activateInput(rml_context_, [this](const PanelInputState& event) { processInput(event); });
        open_menu_index_ = index;
        open_submenu_index_ = -1;
        open_child_submenu_index_ = -1;
        open_grandchild_submenu_index_ = -1;
        keyboard_path_.fill(-1);
        keyboard_depth_ = 0;
        keyboard_focus_menu_index_ = -1;
        keyboard_opened_menu_ = from_keyboard;
        open_menu_idname_ = current_idnames_[index];
        clearTitlebarDragRegion();

        MenuDropdownContent content;
        content.menu_idname = current_idnames_[index];

        if (content.menu_idname == kVrMenu) {
            content.items = vrMenuItems();
        } else {
            python::collect_menu_content(
                current_idnames_[index],
                [](const python::MenuItemInfo* info, void* ctx) {
                    auto* c = static_cast<MenuDropdownContent*>(ctx);
                    MenuItemDesc item;
                    item.type = static_cast<MenuItemDesc::Type>(info->type);
                    item.label = info->label ? info->label : "";
                    item.operator_id = info->operator_id ? info->operator_id : "";
                    item.shortcut = info->shortcut ? info->shortcut : "";
                    item.tooltip = info->tooltip ? info->tooltip : "";
                    item.enabled = info->enabled;
                    item.selected = info->selected;
                    item.callback_index = info->callback_index;
                    c->items.push_back(std::move(item));
                },
                &content);
        }

        dropdown_items_ = buildDropdownItems(content.items);
        rebuildDropdownDOM();
        moveKeyboardHighlight(1, true);
        syncActiveLabelState();
        syncAccessKeyVisibility();
        render_needed_ = true;
    }

    void RmlMenuBar::closeDropdown() {
        open_menu_index_ = -1;
        keyboard_opened_menu_ = false;
        rml_manager_->deactivateInput(rml_context_, true);
        open_submenu_index_ = -1;
        open_child_submenu_index_ = -1;
        open_grandchild_submenu_index_ = -1;
        keyboard_path_.fill(-1);
        keyboard_depth_ = 0;
        open_menu_idname_.clear();
        dropdown_items_.clear();
        tooltip_.setHover({}, nullptr);
        menu_model_.DirtyVariable("dropdown_items");

        if (dropdown_container_) {
            dropdown_container_->SetClass("visible", false);
        }
        if (dropdown_overlay_)
            dropdown_overlay_->SetClass("visible", false);

        syncActiveLabelState();
        syncAccessKeyVisibility();
        render_needed_ = true;
    }

    std::vector<MenuDropdownRootView> RmlMenuBar::buildDropdownItems(const std::vector<MenuItemDesc>& items) {
        auto built = withSharedCheckmarkColumn(buildRootMenuItems(items));
        assignSiblingAccessKeys(built);
        return built;
    }

    void RmlMenuBar::setOpenSubmenu(const int root_index, const int child_index, const int grandchild_index) {
        if (root_index == open_submenu_index_ && child_index == open_child_submenu_index_ &&
            grandchild_index == open_grandchild_submenu_index_)
            return;

        open_submenu_index_ = root_index;
        open_child_submenu_index_ = child_index;
        open_grandchild_submenu_index_ = grandchild_index;
        bool changed = false;
        for (auto& item : dropdown_items_) {
            const bool open = item.has_children && item.index == open_submenu_index_;
            if (item.submenu_open != open) {
                item.submenu_open = open;
                changed = true;
            }
            for (auto& child : item.children) {
                const bool child_open =
                    open && child.has_children && child.index == open_child_submenu_index_;
                if (child.submenu_open != child_open) {
                    child.submenu_open = child_open;
                    changed = true;
                }
                for (auto& grandchild : child.children) {
                    const bool grandchild_open = child_open && grandchild.has_children &&
                                                 grandchild.index == open_grandchild_submenu_index_;
                    if (grandchild.submenu_open != grandchild_open) {
                        grandchild.submenu_open = grandchild_open;
                        changed = true;
                    }
                }
            }
        }
        if (!changed)
            return;

        menu_model_.DirtyVariable("dropdown_items");
        // Apply the retained class change before another pointer event can be
        // tested against the previous submenu geometry.
        if (rml_context_)
            rml_context_->Update();
        render_needed_ = true;
    }

    Rml::Element* RmlMenuBar::dropdownElementAtPoint(const float x, const float y) const {
        if (!dropdown_container_)
            return nullptr;

        const auto contains = [x, y](Rml::Element* element) {
            const auto box = element->GetAbsoluteOffset(Rml::BoxArea::Border);
            const auto size = element->GetBox().GetSize(Rml::BoxArea::Border);
            return x >= box.x && x < box.x + size.x && y >= box.y && y < box.y + size.y;
        };

        const auto find_deepest = [&](const auto& self, Rml::Element* element) -> Rml::Element* {
            // RmlUi retains layout boxes for display:none descendants. Do not
            // let a closed submenu's stale rectangle win over the menu that is
            // actually visible under the pointer.
            if (!element || element->GetDisplay() == Rml::Style::Display::None)
                return nullptr;
            for (int i = element->GetNumChildren() - 1; i >= 0; --i) {
                if (auto* hit = self(self, element->GetChild(i)))
                    return hit;
            }
            return contains(element) ? element : nullptr;
        };

        return find_deepest(find_deepest, dropdown_container_);
    }

    int RmlMenuBar::submenuIndexForElement(Rml::Element* element) const {
        for (auto* el = element; el; el = el->GetParentNode()) {
            if (!el->HasAttribute("data-root-index"))
                continue;
            return el->GetAttribute<int>("data-root-index", -1);
        }
        return -1;
    }

    int RmlMenuBar::childSubmenuIndexForElement(Rml::Element* element) const {
        for (auto* el = element; el; el = el->GetParentNode()) {
            if (!el->HasAttribute("data-child-index"))
                continue;
            return el->GetAttribute<int>("data-child-index", -1);
        }
        return -1;
    }

    int RmlMenuBar::grandchildSubmenuIndexForElement(Rml::Element* element) const {
        for (auto* el = element; el; el = el->GetParentNode()) {
            if (!el->HasAttribute("data-grandchild-index"))
                continue;
            return el->GetAttribute<int>("data-grandchild-index", -1);
        }
        return -1;
    }

    void RmlMenuBar::rebuildToolbarButtons() {
        std::vector<MenuToolbarButtonView> camera_buttons;
        std::vector<MenuToolbarButtonView> render_buttons;
        std::vector<MenuToolbarButtonView> projection_buttons;

        const auto make = [](std::string id, std::string action, std::string value,
                             std::string icon, std::string tooltip_key,
                             std::string tooltip_text, bool selected) {
            return MenuToolbarButtonView{
                .button_id = std::move(id),
                .action = std::move(action),
                .value = std::move(value),
                .icon_src = "../icon/" + std::move(icon) + ".png",
                .tooltip_key = std::move(tooltip_key),
                .tooltip_text = std::move(tooltip_text),
                .selected = selected,
            };
        };

        const auto language_generation = lfs::vis::app_store().language_generation.get();
        if (!has_navigation_tooltip_language_generation_ ||
            language_generation != navigation_tooltip_language_generation_) {
            auto& localization = lfs::event::LocalizationManager::getInstance();
            navigation_tooltips_ = {
                localization.get("toolbar.orbit_camera"),
                localization.get("toolbar.free_orbit_camera"),
                localization.get("toolbar.fly_camera"),
                localization.get("toolbar.drone_camera"),
                localization.get("collision.walk"),
            };
            navigation_tooltip_language_generation_ = language_generation;
            has_navigation_tooltip_language_generation_ = true;
        }

        if (const auto* ic = lfs::vis::InputController::instance()) {
            using NavMode = lfs::vis::InputController::CameraNavigationMode;
            struct NavButtonSpec {
                NavMode mode;
                const char* icon;
                size_t tooltip_index;
            };
            const NavButtonSpec kNavButtons[] = {
                {NavMode::Orbit, "camera-orbit", 0},
                {NavMode::Trackball, "world", 1},
                {NavMode::FPV, "camera-fpv", 2},
                {NavMode::Drone, "drone", 3},
                {NavMode::Walk, "walk", 4},
            };
            const auto mode = ic->cameraNavigationMode();
            for (const auto& spec : kNavButtons) {
                const std::string name = lfs::vis::InputController::cameraNavigationModeName(spec.mode);
                camera_buttons.push_back(make("menu-camera-" + name, "set_camera_navigation_mode", name,
                                              spec.icon, "", navigation_tooltips_[spec.tooltip_index],
                                              mode == spec.mode));
                if (spec.mode == NavMode::Walk) {
                    const auto* manager = services().sceneOrNull();
                    camera_buttons.back().enabled = manager && std::ranges::any_of(manager->getScene().getNodes(),
                                                                                   [](const auto* node) { return node && bool(node->collision); });
                }
            }
        }

        if (const auto* rm = lfs::vis::services().renderingOrNull()) {
            const auto settings = rm->getSettings();

            std::string active_mode = "splats";
            if (settings.point_cloud_mode)
                active_mode = "points";
            else if (settings.show_rings)
                active_mode = "rings";
            else if (settings.show_center_markers)
                active_mode = "centers";

            render_buttons.push_back(make("menu-render-splats", "set_render_mode", "splats", "splats",
                                          "toolbar.splat_rendering", "Splat Rendering",
                                          active_mode == "splats"));
            render_buttons.push_back(make("menu-render-points", "set_render_mode", "points",
                                          "points", "toolbar.point_cloud", "Point Cloud",
                                          active_mode == "points"));
            render_buttons.push_back(make("menu-render-rings", "set_render_mode", "rings", "ring",
                                          "toolbar.gaussian_rings", "Gaussian Rings",
                                          active_mode == "rings"));
            render_buttons.push_back(make("menu-render-centers", "set_render_mode", "centers",
                                          "circle-dot", "toolbar.center_markers", "Center Markers",
                                          active_mode == "centers"));

            const bool ortho = settings.orthographic;
            projection_buttons.push_back(make("menu-projection", "toggle_projection", "",
                                              ortho ? "orthographic" : "perspective",
                                              ortho ? "toolbar.orthographic" : "toolbar.perspective",
                                              ortho ? "Orthographic" : "Perspective", ortho));
            projection_buttons.push_back(make("menu-depth-view", "toggle_depth_view", "", "depth-map",
                                              "toolbar.depth_map", "Depth Map", settings.depth_view));

            bool view_snap = false;
            if (const auto* ic = lfs::vis::InputController::instance())
                view_snap = ic->cameraViewSnapEnabled();
            projection_buttons.push_back(make("menu-view-snap", "toggle_camera_view_snap", "", "magnet",
                                              "", "Snap Axis Views", view_snap));
        }

        if (render_buttons != render_buttons_) {
            render_buttons_ = std::move(render_buttons);
            if (menu_model_)
                menu_model_.DirtyVariable("menu_render_buttons");
            render_needed_ = true;
        }
        if (camera_buttons != camera_buttons_) {
            camera_buttons_ = std::move(camera_buttons);
            if (menu_model_)
                menu_model_.DirtyVariable("menu_camera_buttons");
            render_needed_ = true;
        }
        if (projection_buttons != projection_buttons_) {
            projection_buttons_ = std::move(projection_buttons);
            if (menu_model_)
                menu_model_.DirtyVariable("menu_projection_buttons");
            render_needed_ = true;
        }
    }

    void RmlMenuBar::rebuildPortalStatus() {
        if (!menu_model_)
            return;

        const auto account = lfs::vis::app_store().account_state.get();
        const auto gallery = lfs::vis::app_store().gallery_state.get();
        const auto& localization = lfs::event::LocalizationManager::getInstance();
        const bool needs_approval = account.signed_in && gallery.relink_required;
        const std::string connection = account.disconnecting ? "disconnecting"
                                       : account.linking     ? "linking"
                                       : needs_approval      ? "approval_needed"
                                       : account.signed_in   ? "connected"
                                       : account.authorized  ? "switched_off"
                                                             : "not_connected";
        const bool connected = account.signed_in;
        const bool checking = account.linking || account.disconnecting;
        const bool transferring = connected && (gallery.active_uploads > 0 || gallery.active_downloads > 0);
        const std::string tone = checking ? "connecting" : needs_approval || !account.error.empty() ? "error"
                                                       : transferring                               ? "transferring"
                                                       : connected                                  ? "connected"
                                                                                                    : "disconnected";
        const std::string icon = checking         ? "ring"
                                 : needs_approval ? "cloud-bang"
                                 : transferring   ? (gallery.active_uploads > 0 && gallery.active_downloads > 0 ? "cloud-updown"
                                                     : gallery.active_uploads > 0                               ? "cloud-up"
                                                                                                                : "cloud-down")
                                 : connected      ? "cloud-check"
                                                  : "cloud-strike";
        const auto set = [this](const char* name, auto& current, auto value) {
            if (current != value) {
                current = std::move(value);
                menu_model_.DirtyVariable(name);
                render_needed_ = true;
            }
        };
        const std::string activity = transferring && connection == "connected"
                                         ? (gallery.active_uploads > 0 && gallery.active_downloads > 0 ? "transferring"
                                            : gallery.active_uploads > 0                               ? "uploading"
                                                                                                       : "downloading")
                                         : connection;
        const auto connection_key = "portal.status." + (activity == "not_connected" ? "disconnected" : activity);
        std::string label = localization.get(connection_key);
        if (connection == "connected" && !account.label.empty()) {
            label = localization.get("portal.status.toolbar_connected");
            constexpr std::string_view kInitialsToken = "{initials}";
            if (const auto initials = label.find(kInitialsToken); initials != std::string::npos)
                label.replace(initials, kInitialsToken.size(), account.label);
        } else if (account.linking && !account.label.empty())
            label += " " + account.label;
        set("portal_connection_label", portal_connection_label_, std::move(label));
        std::string tooltip = localization.get(needs_approval       ? "portal.status.reauthorize"
                                               : connected          ? "portal.status.disconnect"
                                               : account.linking    ? "portal.status.cancel"
                                               : account.authorized ? "portal.status.turn_on"
                                                                    : "portal.status.connect");
        if (connected && !account.display_name.empty()) {
            std::string connected_as = localization.get("portal.status.connected_as");
            constexpr std::string_view kNameToken = "{name}";
            if (const auto name = connected_as.find(kNameToken); name != std::string::npos)
                connected_as.replace(name, kNameToken.size(), account.display_name);
            tooltip += "\n" + connected_as;
        } else if (!account.tooltip.empty())
            tooltip += "\n" + account.tooltip;
        if (transferring)
            tooltip += "\n" + gallery.tooltip;
        if (!account.error.empty()) {
            const std::string error_key = account.error == "sign_in_unavailable" ? "account.error.unavailable"
                                          : account.error == "sign_in_failed"    ? "account.error.generic"
                                          : account.error == "unsafe_portal_url" ? "projects.gallery.error.unsafe_url"
                                                                                 : "account.error." + account.error;
            tooltip += "\n" + std::string(localization.hasKey(error_key) ? localization.get(error_key)
                                                                         : localization.get("account.error.generic"));
        }
        set("portal_connection_tooltip", portal_connection_tooltip_, std::move(tooltip));
        set("portal_connection_icon", portal_connection_icon_, "../icon/gallery-" + icon + ".png");
        set("portal_connection_tone", portal_connection_tone_, tone);
        portal_transfer_animation_active_ = tone == "transferring";
        set("gallery_has_progress", gallery_has_progress_, gallery.active_uploads > 0 || gallery.active_downloads > 0);
        set("gallery_progress_label", gallery_progress_label_, gallery.label);
        set("gallery_progress_detail", gallery_progress_detail_, gallery.detail);
        set("gallery_progress_tooltip", gallery_progress_tooltip_, gallery.tooltip);
        set("gallery_progress_indeterminate", gallery_progress_indeterminate_, gallery.percent < 0);
        set("gallery_progress_width", gallery_progress_width_,
            std::format("{}%", gallery.percent < 0 ? 100 : std::clamp(gallery.percent, 0, 100)));
    }

    void RmlMenuBar::dispatchToolbarAction(const std::string& action, const std::string& value) {
        auto* rm = lfs::vis::services().renderingOrNull();

        if (action == "set_render_mode") {
            if (!rm)
                return;
            rm->setViewportRenderMode(value);
        } else if (action == "set_camera_navigation_mode") {
            auto* ic = lfs::vis::InputController::instance();
            if (!ic)
                return;
            if (const auto mode = lfs::vis::InputController::cameraNavigationModeFromName(value)) {
                if (!ic->setCameraNavigationMode(*mode))
                    return;
                if (auto* const viewer = lfs::python::get_visualizer()) {
                    viewer->projectMarkSessionChapterCommitted("VIEW");
                }
                lfs::vis::saveCameraNavigationPreference(value);
            }
        } else if (action == "toggle_projection") {
            if (!rm)
                return;
            const bool ortho = rm->getSettings().orthographic;
            float viewport_height = 0.0f;
            float distance_to_pivot = 0.0f;
            if (const auto view = lfs::vis::get_current_view_info(); view.has_value()) {
                viewport_height = static_cast<float>(view->height);
                const glm::vec3 eye(view->translation[0], view->translation[1], view->translation[2]);
                const glm::vec3 pivot(view->pivot[0], view->pivot[1], view->pivot[2]);
                distance_to_pivot = glm::length(pivot - eye);
            }
            rm->setOrthographic(!ortho, viewport_height, distance_to_pivot);
            lfs::vis::apply_set_ortho_scale(std::nullopt);
            if (auto* const viewer = lfs::python::get_visualizer())
                viewer->projectMarkSessionChapterCommitted("VIEW");
        } else if (action == "toggle_depth_view") {
            if (!rm)
                return;
            auto settings = rm->getSettings();
            settings.depth_view = !settings.depth_view;
            rm->updateSettings(settings, lfs::vis::DirtyFlag::ALL);
            if (auto* const viewer = lfs::python::get_visualizer())
                viewer->projectMarkSessionChapterCommitted("VIEW");
        } else if (action == "toggle_camera_view_snap") {
            if (auto* ic = lfs::vis::InputController::instance()) {
                ic->setCameraViewSnapEnabled(!ic->cameraViewSnapEnabled());
                if (auto* const viewer = lfs::python::get_visualizer()) {
                    viewer->projectMarkSessionChapterCommitted("VIEW");
                }
            }
            if (const auto* ic = lfs::vis::InputController::instance())
                lfs::vis::saveCameraViewSnapPreference(ic->cameraViewSnapEnabled());
        } else if (action == "toggle_independent_split_view") {
            if (auto* ic = lfs::vis::InputController::instance())
                ic->toggleIndependentSplitView();
        } else if (action == "portal_connection") {
            python::invoke_operator("lfs_plugins.help_menu.PortalConnectionOperator");
        } else if (action == "gallery_transfers") {
            python::invoke_operator("lfs_plugins.help_menu.GalleryTransfersOperator");
        } else if (action == "window_toggle_ui") {
            lfs::core::events::ui::ToggleUI{}.emit();
        } else if (action == "window_minimize") {
            if (auto* wm = lfs::vis::services().windowOrNull())
                wm->minimize();
        } else if (action == "window_toggle_maximize") {
            if (auto* wm = lfs::vis::services().windowOrNull())
                wm->toggleMaximized();
        } else if (action == "window_close") {
            if (auto* wm = lfs::vis::services().windowOrNull()) {
                wm->requestClose();
                wm->wakeEventLoop();
            }
        }

        rebuildToolbarButtons();
        render_needed_ = true;
    }

    void RmlMenuBar::setUiHidden(const bool hidden) {
        if (ui_hidden_ == hidden)
            return;
        ui_hidden_ = hidden;
        render_needed_ = true;
    }

    Rml::Element* RmlMenuBar::toolbarButtonAtPoint(const float x, const float y) const {
        const auto find_button = [x, y](Rml::Element* root) -> Rml::Element* {
            if (!root)
                return nullptr;
            for (int i = 0; i < root->GetNumChildren(); ++i) {
                auto* child = root->GetChild(i);
                if (!child || child->GetDisplay() == Rml::Style::Display::None || !child->HasAttribute("data-action"))
                    continue;
                const auto box = child->GetAbsoluteOffset(Rml::BoxArea::Border);
                const auto size = child->GetBox().GetSize(Rml::BoxArea::Border);
                if (x >= box.x && x < box.x + size.x && y >= box.y && y < box.y + size.y)
                    return child;
            }
            return nullptr;
        };

        if (auto* button = find_button(menu_toolbar_))
            return button;
        return find_button(menu_window_controls_);
    }

    bool RmlMenuBar::projectTitleAtPoint(const float x, const float y) const {
        if (!project_title_el_ || !project_title_has_room_ ||
            project_title_.empty() || project_tooltip_.empty()) {
            return false;
        }
        const auto offset = project_title_el_->GetAbsoluteOffset(Rml::BoxArea::Border);
        const auto size = project_title_el_->GetBox().GetSize(Rml::BoxArea::Border);
        return x >= offset.x && x < offset.x + size.x &&
               y >= offset.y && y < offset.y + size.y;
    }

    void RmlMenuBar::updateCompactLayout(const int screen_w, const float dp_ratio) {
        if (auto* body = document_->GetElementById("body")) {
            const bool compact = screen_w < 760.0f * dp_ratio;
            if (body->IsClassSet("compact") != compact) {
                body->SetClass("compact", compact);
                render_needed_ = true;
            }
        }
    }

    void RmlMenuBar::updateToolbarLayout(const int screen_w, const float dp_ratio) {
        // Right-align the render/projection toolbar to the viewport edge, but
        // keep it clear of the window-control cluster when there is no dock panel.
        if (!menu_toolbar_)
            return;

        const auto measure = [&] {
            struct Fit {
                float min_right_px = 0;
                float right_px = 0;
                float menus_right_px = 0;
                bool toolbar_fits = false;
            };
            const float inset = 8.0f * dp_ratio;
            constexpr float kFallbackRightClusterReserveDp = 184.0f;
            Fit fit;
            fit.min_right_px = kFallbackRightClusterReserveDp * dp_ratio;
            if (menu_window_controls_) {
                const auto offset = menu_window_controls_->GetAbsoluteOffset(Rml::BoxArea::Border);
                if (offset.x > 0.0f)
                    fit.min_right_px = static_cast<float>(screen_w) - offset.x + 4.0f * dp_ratio;
            }
            fit.right_px = fit.min_right_px;
            if (viewport_right_edge_ > 0.0f)
                fit.right_px = std::max(fit.right_px, static_cast<float>(screen_w) - viewport_right_edge_ + inset);

            // The toolbar is out of flow, so viewport alignment alone would let it slide over the
            // menu labels. Move it to a second row once even the cap collides.
            fit.menus_right_px = menu_items_ ? menu_items_->GetAbsoluteOffset(Rml::BoxArea::Border).x +
                                                   menu_items_->GetOffsetWidth()
                                             : 0.0f;
            const float clear_of_menus_px = static_cast<float>(screen_w) -
                                            menu_toolbar_->GetOffsetWidth() - fit.menus_right_px -
                                            12.0f * dp_ratio;
            // Asymmetric threshold so a window parked on the boundary cannot flicker.
            const float show_slack_px = toolbar_fits_ ? 0.0f : 8.0f * dp_ratio;
            fit.toolbar_fits = clear_of_menus_px >= fit.min_right_px + show_slack_px;
            if (fit.toolbar_fits)
                fit.right_px = std::min(fit.right_px, clear_of_menus_px);
            else
                fit.right_px = inset;
            return fit;
        };

        // The centred project title does not consume this gap. A long portal status does.
        // Drop its words, keep the icon, and only then wrap the viewport tools.
        auto fit = measure();
        auto* body = document_->GetElementById("body");
        const auto set_squeeze = [&](bool squeeze) {
            if (!body || body->IsClassSet("toolbar-squeeze") == squeeze)
                return;
            body->SetClass("toolbar-squeeze", squeeze);
            rml_context_->Update();
            render_needed_ = true;
        };
        const bool content_changed = screen_w != toolbar_squeeze_screen_w_ ||
                                     std::abs(fit.menus_right_px - toolbar_squeeze_menus_right_) > 0.5f;
        if (!toolbar_squeezed_ && !fit.toolbar_fits) {
            set_squeeze(true);
            toolbar_squeezed_ = body && body->IsClassSet("toolbar-squeeze");
            toolbar_squeeze_screen_w_ = screen_w;
            toolbar_squeeze_menus_right_ = fit.menus_right_px;
            fit = measure();
        } else if (toolbar_squeezed_ && content_changed) {
            set_squeeze(false);
            toolbar_squeezed_ = false;
            auto released = measure();
            if (!released.toolbar_fits) {
                set_squeeze(true);
                toolbar_squeezed_ = body && body->IsClassSet("toolbar-squeeze");
                toolbar_squeeze_screen_w_ = screen_w;
                toolbar_squeeze_menus_right_ = released.menus_right_px;
                fit = measure();
            } else {
                fit = released;
            }
        }

        if (fit.toolbar_fits != toolbar_fits_) {
            menu_toolbar_->SetClass("second-row", !fit.toolbar_fits);
            if (auto* row = document_->GetElementById("menu-row"))
                row->SetClass("two-rows", !fit.toolbar_fits);
            bar_height_ = fit.toolbar_fits ? 30.0f : 60.0f;
            toolbar_fits_ = fit.toolbar_fits;
            render_needed_ = true;
            LOG_DEBUG("Menu toolbar {} at {} px", fit.toolbar_fits ? "inline" : "second row", screen_w);
        }
        if (std::abs(fit.right_px - applied_toolbar_right_) > 0.5f) {
            menu_toolbar_->SetProperty("right", std::format("{:.1f}px", fit.right_px));
            applied_toolbar_right_ = fit.right_px;
            render_needed_ = true;
        }
    }

    void RmlMenuBar::updateProjectTitleLayout(const int screen_w, const float dp_ratio) {
        if (!project_title_container_ || !project_title_el_)
            return;

        const float padding = 12.0f * dp_ratio;
        const float menu_right = menu_items_
                                     ? menu_items_->GetAbsoluteOffset(Rml::BoxArea::Border).x +
                                           menu_items_->GetOffsetWidth() + padding
                                     : padding;
        float controls_left = static_cast<float>(screen_w) - padding;
        if (menu_window_controls_)
            controls_left = menu_window_controls_->GetAbsoluteOffset(Rml::BoxArea::Border).x - padding;
        if (toolbar_fits_ && menu_toolbar_) {
            // SetProperty("right") takes effect at the following Rml layout
            // update. Use the value selected in this frame so the title never
            // spends one resize frame underneath the viewport toolbar.
            const float toolbar_width = std::max(
                menu_toolbar_->GetOffsetWidth(),
                menu_toolbar_->GetScrollWidth());
            const float toolbar_left = applied_toolbar_right_ >= 0.0f
                                           ? static_cast<float>(screen_w) -
                                                 applied_toolbar_right_ - toolbar_width
                                           : menu_toolbar_->GetAbsoluteOffset(
                                                              Rml::BoxArea::Border)
                                                 .x;
            controls_left = std::min(controls_left,
                                     toolbar_left - padding);
        }

        const float width = std::max(0.0f, controls_left - menu_right);
        const bool has_room = !project_title_.empty() && width >= 96.0f * dp_ratio;
        if (has_room != project_title_has_room_) {
            project_title_container_->SetClass("no-room", !has_room);
            project_title_has_room_ = has_room;
            render_needed_ = true;
        }
        if (!has_room) {
            return;
        }

        if (std::abs(menu_right - applied_project_title_left_) > 0.5f) {
            project_title_container_->SetProperty("left", std::format("{:.1f}px", menu_right));
            applied_project_title_left_ = menu_right;
            render_needed_ = true;
        }
        if (std::abs(width - applied_project_title_width_) > 0.5f) {
            project_title_container_->SetProperty("width", std::format("{:.1f}px", width));
            applied_project_title_width_ = width;
            render_needed_ = true;
        }
    }

    void RmlMenuBar::clearTitlebarDragRegion() {
        if (auto* wm = lfs::vis::services().windowOrNull())
            wm->clearTitlebarDragRegion();
    }

    void RmlMenuBar::updateTitlebarDragRegion(const int bar_height_px) {
        auto* wm = lfs::vis::services().windowOrNull();
        if (!wm)
            return;
        if (open_menu_index_ >= 0 || bar_height_px <= 0) {
            wm->clearTitlebarDragRegion();
            return;
        }

        const auto append_element = [](std::vector<lfs::vis::WindowManager::HitTestRect>& rects,
                                       Rml::Element* element) {
            if (!element)
                return;

            const auto offset = element->GetAbsoluteOffset(Rml::BoxArea::Border);
            const auto size = element->GetBox().GetSize(Rml::BoxArea::Border);
            const int x1 = static_cast<int>(std::floor(offset.x));
            const int y1 = static_cast<int>(std::floor(offset.y));
            const int x2 = static_cast<int>(std::ceil(offset.x + size.x));
            const int y2 = static_cast<int>(std::ceil(offset.y + size.y));
            if (x2 <= x1 || y2 <= y1)
                return;

            rects.push_back({
                .x = x1,
                .y = y1,
                .w = x2 - x1,
                .h = y2 - y1,
            });
        };

        std::vector<lfs::vis::WindowManager::HitTestRect> excluded_rects;
        excluded_rects.reserve(3);
        append_element(excluded_rects, menu_items_);
        append_element(excluded_rects, menu_toolbar_);
        append_element(excluded_rects, menu_window_controls_);
        wm->setTitlebarDragRegion(bar_height_px, std::move(excluded_rects));
    }

    void RmlMenuBar::rebuildDropdownDOM() {
        if (!dropdown_container_ || !dropdown_popup_ || !dropdown_overlay_ || !menu_items_)
            return;

        const int count = menu_items_->GetNumChildren();
        if (open_menu_index_ < 0 || open_menu_index_ >= count)
            return;

        auto* label_el = menu_items_->GetChild(open_menu_index_);
        if (!label_el)
            return;
        const auto label_offset = label_el->GetAbsoluteOffset(Rml::BoxArea::Border);
        const auto label_size = label_el->GetBox().GetSize(Rml::BoxArea::Border);

        menu_model_.DirtyVariable("dropdown_items");
        dropdown_container_->SetProperty("left", "0px");
        dropdown_container_->SetProperty("top", "0px");
        dropdown_popup_->SetProperty("left", std::format("{}px", label_offset.x));
        dropdown_popup_->SetProperty("top", std::format("{}px", label_offset.y + label_size.y));
        dropdown_container_->SetClass("visible", true);
        dropdown_overlay_->SetClass("visible", true);
        // Top-level menus can switch on hover. Materialize the new records now
        // so a fast following click cannot hit the previous menu's rows.
        if (rml_context_)
            rml_context_->Update();
        render_needed_ = true;
    }

    void RmlMenuBar::sizeOpenDropdowns() {
        if (!rml_context_ || !dropdown_popup_ || open_menu_index_ < 0)
            return;

        struct PopupLimits {
            Rml::Element* element;
            float min_width;
            float max_width;
        };

        const float dp_ratio = rml_manager_ ? rml_manager_->getDpRatio() : 1.0f;
        std::vector<PopupLimits> popups{
            {dropdown_popup_, 200.0f * dp_ratio, 360.0f * dp_ratio},
        };
        Rml::ElementList submenus;
        dropdown_popup_->GetElementsByClassName(submenus, "submenu-popup");
        for (auto* const submenu : submenus) {
            if (submenu && submenu->GetDisplay() != Rml::Style::Display::None)
                popups.push_back({submenu, 200.0f * dp_ratio, 420.0f * dp_ratio});
        }

        constexpr float tolerance_px = 0.5f;
        // Measure from min-width. Flex labels grow to the popup width, and
        // GetScrollWidth() is at least the allocated client width, so measuring
        // at max-width would report max for every popup.
        bool collapsed = false;
        for (const auto& popup : popups) {
            const float current_width =
                popup.element->GetBox().GetSize(Rml::BoxArea::Content).x;
            if (std::abs(current_width - popup.min_width) <= tolerance_px)
                continue;
            popup.element->SetProperty("width", std::format("{:.1f}px", popup.min_width));
            collapsed = true;
        }
        if (collapsed)
            rml_context_->Update();

        bool shared = false;
        for (const auto& popup : popups)
            shared = shareShortcutWidth(popup.element) || shared;
        if (shared)
            rml_context_->Update();

        bool fitted = false;
        for (const auto& popup : popups) {
            const float content_width = std::ceil(popupContentWidth(popup.element));
            const float fitted_width =
                std::clamp(content_width, popup.min_width, popup.max_width);
            const float current_width =
                popup.element->GetBox().GetSize(Rml::BoxArea::Content).x;
            if (std::abs(current_width - fitted_width) <= tolerance_px)
                continue;
            popup.element->SetProperty("width", std::format("{:.1f}px", fitted_width));
            fitted = true;
        }
        if (fitted)
            rml_context_->Update();

        // Fit every visible cascade independently. Additional theme levels can
        // reach the right edge even when the root menu has plenty of room.
        const auto dimensions = rml_context_->GetDimensions();
        for (const auto& popup : popups) {
            auto* element = popup.element;
            if (element == dropdown_popup_)
                continue;
            element->SetProperty("left", "100%");
            element->SetProperty("top", "-4dp");
            rml_context_->Update();
            auto offset = element->GetAbsoluteOffset(Rml::BoxArea::Border);
            const auto size = element->GetBox().GetSize(Rml::BoxArea::Border);
            if (offset.x + size.x > dimensions.x) {
                element->SetProperty("left", std::format("{}px", -size.x));
                rml_context_->Update();
                offset = element->GetAbsoluteOffset(Rml::BoxArea::Border);
            }
            if (offset.x < 0) {
                const auto parent_offset = element->GetParentNode()->GetAbsoluteOffset(Rml::BoxArea::Border);
                element->SetProperty("left", std::format("{}px", -parent_offset.x));
            }
            const float fitted_y = std::clamp(offset.y, 0.0f, std::max(0.0f, dimensions.y - size.y));
            if (std::abs(fitted_y - offset.y) > tolerance_px)
                element->SetProperty("top", std::format("{}px", -4.0f * dp_ratio + fitted_y - offset.y));
            rml_context_->Update();
        }
    }

    bool RmlMenuBar::updateTheme() {
        if (!document_)
            return false;

        const std::size_t theme_signature = rml_theme::currentThemeSignature();
        if (has_theme_signature_ && theme_signature == last_theme_signature_)
            return false;
        last_theme_signature_ = theme_signature;
        has_theme_signature_ = true;

        if (base_rcss_.empty())
            base_rcss_ = rml_theme::loadBaseRCSS("rmlui/menubar.rcss");

        rml_theme::applyTheme(document_, base_rcss_, rml_theme::loadBaseRCSS("rmlui/menubar.theme.rcss"));

        brand_logo_src_ = rml_theme::pathToRmlImageSource(lfs::vis::getAssetPath("lichtfeld-icon.png"));
        menu_model_.DirtyVariable("brand_logo_src");
        return true;
    }

    void RmlMenuBar::draw(int screen_w, int screen_h) {
        if (!rml_context_ || !document_)
            return;
        if (!rml_manager_ || !rml_manager_->getVulkanRenderInterface())
            return;
        const bool theme_changed = updateTheme();
        rebuildToolbarButtons();
        rebuildPortalStatus();

        if (menu_window_split_view_) {
            const bool split_view = [&] {
                if (auto* rm = lfs::vis::services().renderingOrNull())
                    return rm->getSettings().split_view_mode == lfs::vis::SplitViewMode::IndependentDual;
                return false;
            }();
            if (split_view != last_window_split_view_) {
                menu_window_split_view_->SetClass("selected", split_view);
                menu_window_split_view_->SetAttribute(
                    "title", lfs::event::LocalizationManager::getInstance().get(
                                 split_view ? "ui.exit_independent_split_view" : "ui.independent_split_view"));
                last_window_split_view_ = split_view;
                render_needed_ = true;
            }
        }

        if (menu_window_toggle_ui_ && ui_hidden_ != last_ui_hidden_) {
            menu_window_toggle_ui_->SetClass("selected", ui_hidden_);
            menu_window_toggle_ui_->SetAttribute(
                "title", lfs::event::LocalizationManager::getInstance().get(ui_hidden_ ? "ui.show_ui" : "ui.hide_ui"));
            last_ui_hidden_ = ui_hidden_;
            render_needed_ = true;
        }

        if (menu_window_maximize_) {
            const bool maximized = [&] {
                if (auto* wm = lfs::vis::services().windowOrNull())
                    return wm->isMaximized();
                return false;
            }();
            if (maximized != last_window_maximized_) {
                menu_window_maximize_->SetClass("maximized", maximized);
                menu_window_maximize_->SetAttribute(
                    "title", lfs::event::LocalizationManager::getInstance().get(
                                 maximized ? "ui.restore_window" : "ui.maximize_window"));
                last_window_maximized_ = maximized;
                render_needed_ = true;
            }
        }

        const float dp_ratio = rml_manager_->getDpRatio();
        int bar_h = static_cast<int>(bar_height_ * dp_ratio);
        updateCompactLayout(screen_w, dp_ratio);
        if (dp_ratio != last_dp_ratio_)
            render_needed_ = true;
        last_dp_ratio_ = dp_ratio;

        // Portal status and transfer progress can change the right cluster's width.
        // Lay it out before reserving space for the viewport toolbar.
        if (render_needed_ || screen_w != last_ctx_w_) {
            rml_context_->SetDimensions(Rml::Vector2i(screen_w, std::max(bar_h, last_ctx_h_)));
            rml_context_->Update();
        }

        updateToolbarLayout(screen_w, dp_ratio);
        bar_h = static_cast<int>(bar_height_ * dp_ratio);
        updateProjectTitleLayout(screen_w, dp_ratio);

        int ctx_w = screen_w;
        // A closed menu bar only occupies the bar strip, but a dropdown or a
        // tooltip needs the full height so it is not clipped below the bar.
        int ctx_h = (open_menu_index_ >= 0 || tooltip_.hasActiveState()) ? screen_h : bar_h;

        bool tooltip_changed = false;
        if (tooltip_.hasActiveState()) {
            if (!body_el_)
                body_el_ = document_->GetElementById("body");
            tooltip_changed = tooltip_.apply(body_el_, last_mouse_x_, last_mouse_y_, ctx_w, ctx_h);
        }
        rml_manager_->setContextNeedsPassiveMouseMoveFrames(rml_context_, tooltip_.hasActiveState());
        rml_manager_->setContextTooltipRevealDeadline(rml_context_, tooltip_.revealDeadline());

        const bool size_changed = (ctx_w != last_ctx_w_ || ctx_h != last_ctx_h_);
        const bool refresh_cache = render_needed_ || theme_changed || size_changed ||
                                   tooltip_changed || direct_cache_.texture == 0 || portal_transfer_animation_active_;

        if (refresh_cache) {
            rml_context_->SetDimensions(Rml::Vector2i(ctx_w, ctx_h));
            if (ctx_h != last_document_h_) {
                document_->SetProperty("height", std::format("{}px", ctx_h));
                last_document_h_ = ctx_h;
            }
            rml_context_->Update();
            if (open_menu_index_ >= 0)
                sizeOpenDropdowns();
        }
        updateTitlebarDragRegion(bar_h);

        rml_manager_->queueCachedVulkanContext({
            .context = rml_context_,
            .cache = &direct_cache_,
            .cache_width = ctx_w,
            .cache_height = ctx_h,
            .offset_x = 0.0f,
            .offset_y = 0.0f,
            .draw_width = static_cast<float>(screen_w),
            .draw_height = static_cast<float>(ctx_h),
            .refresh = refresh_cache,
            .foreground = true,
            .clip_enabled = true,
            .clip = {
                .x1 = 0.0f,
                .y1 = 0.0f,
                .x2 = static_cast<float>(screen_w),
                .y2 = static_cast<float>(ctx_h),
            },
        });
        last_ctx_w_ = ctx_w;
        last_ctx_h_ = ctx_h;
        render_needed_ = false;
    }

} // namespace lfs::vis::gui
