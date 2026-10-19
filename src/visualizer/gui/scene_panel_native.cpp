/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/scene_panel_native.hpp"

#include "core/event_bridge/localization_manager.hpp"
#include "core/number_format.hpp"
#include "core/parameter_manager.hpp"
#include "gui/gui_manager.hpp"
#include "gui/rmlui/elements/scene_graph_element.hpp"
#include "gui/rmlui/rml_theme.hpp"
#include "gui/string_keys.hpp"
#include "gui/utils/summary_labels.hpp"
#include "internal/resource_paths.hpp"
#include "preferences.hpp"
#include "visualizer/app_store.hpp"
#include "visualizer/core/services.hpp"

#include <RmlUi/Core.h>
#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/Elements/ElementFormControlInput.h>

#include <cmath>
#include <format>
#include <limits>
#include <optional>
#include <string_view>

namespace lfs::vis::gui {

    namespace {

        [[nodiscard]] std::string tr(const char* key) {
            const std::string value = LOC(key);
            return value.empty() ? std::string(key) : value;
        }

        [[nodiscard]] std::string cacheAttrName(std::string_view kind, std::string_view name) {
            return std::format("data-lfs-{}-{}", kind, name);
        }

        [[nodiscard]] std::string resolveRmlImageSource(const std::string_view asset_path) {
            try {
                return rml_theme::pathToRmlImageSource(
                    lfs::vis::getAssetPath(std::string(asset_path)));
            } catch (const std::exception& e) {
                LOG_WARN("NativeScenePanel: failed to resolve icon '{}': {}", asset_path, e.what());
                return {};
            }
        }

        [[nodiscard]] bool setCachedInnerRml(Rml::Element* el, const std::string& value) {
            if (!el)
                return false;

            const std::string attr_name = cacheAttrName("sync", "rml");
            if (el->GetAttribute<Rml::String>(attr_name.c_str(), "") == value)
                return false;

            el->SetInnerRML(value);
            el->SetAttribute(attr_name, value);
            return true;
        }

        [[nodiscard]] bool setCachedText(Rml::Element* el, const std::string& value) {
            return setCachedInnerRml(el, Rml::StringUtilities::EncodeRml(value));
        }

        [[nodiscard]] bool setCachedProperty(Rml::Element* el, std::string_view name,
                                             const std::string& value) {
            if (!el)
                return false;

            const std::string attr_name = cacheAttrName("prop", name);
            if (el->GetAttribute<Rml::String>(attr_name.c_str(), "") == value)
                return false;

            el->SetProperty(std::string(name), value);
            el->SetAttribute(attr_name, value);
            return true;
        }

        [[nodiscard]] bool setCachedAttribute(Rml::Element* el, std::string_view name,
                                              const std::string& value) {
            if (!el)
                return false;

            const std::string attr_name = cacheAttrName("attr", name);
            if (el->GetAttribute<Rml::String>(attr_name.c_str(), "") == value)
                return false;

            el->SetAttribute(std::string(name), value);
            el->SetAttribute(attr_name, value);
            return true;
        }

        [[nodiscard]] bool setCachedDisabled(Rml::Element* el, const bool disabled) {
            if (!el)
                return false;

            const std::string attr_name = cacheAttrName("attr", "disabled");
            const char* const next = disabled ? "1" : "0";
            if (el->GetAttribute<Rml::String>(attr_name.c_str(), "") == next)
                return false;

            if (disabled)
                el->SetAttribute("disabled", "disabled");
            else
                el->RemoveAttribute("disabled");
            el->SetAttribute(attr_name, next);
            return true;
        }

        [[nodiscard]] bool setCachedClass(Rml::Element* el, std::string_view cls,
                                          const bool enabled) {
            if (!el)
                return false;

            const std::string attr_name = cacheAttrName("class", cls);
            const char* const next = enabled ? "1" : "0";
            if (el->GetAttribute<Rml::String>(attr_name.c_str(), "") == next)
                return false;

            el->SetClass(std::string(cls), enabled);
            el->SetAttribute(attr_name, next);
            return true;
        }

        [[nodiscard]] int optionalMetricMilli(const std::optional<float>& value) {
            if (!value || !std::isfinite(*value))
                return std::numeric_limits<int>::min();
            return static_cast<int>(std::lround(*value * 1000.0f));
        }

        [[nodiscard]] bool hasDiscreteInputActivity(const PanelInputState* input) {
            if (!input)
                return false;

            for (int i = 0; i < 3; ++i) {
                if (input->mouse_clicked[i] || input->mouse_released[i] || input->mouse_down[i])
                    return true;
            }
            return input->mouse_wheel != 0.0f ||
                   !input->keys_pressed.empty() ||
                   !input->input_events.empty();
        }

    } // namespace

    NativeScenePanel::NativeScenePanel(RmlUIManager* manager)
        : manager_(manager),
          host_(manager, "scene_panel_native", "rmlui/scene_tree.rml") {
        listener_.owner = this;
    }

    bool NativeScenePanel::needsAnimationFrame() const {
        return host_.needsAnimationFrame() ||
               (tree_el_ && tree_el_->needsAnimationFrame());
    }

    void NativeScenePanel::EventListener::ProcessEvent(Rml::Event& event) {
        if (owner)
            owner->handleEvent(event);
    }

    NativeScenePanel::~NativeScenePanel() {
        clearElementCache();
    }

    void NativeScenePanel::clearElementCache() {
        if (manager_)
            manager_->setActiveSceneGraphElement(nullptr);
        tree_el_ = nullptr;
        scene_tab_el_ = nullptr;
        chip_row_el_ = nullptr;
        summary_model_chip_el_ = nullptr;
        summary_node_chip_el_ = nullptr;
        summary_filter_chip_el_ = nullptr;
        scene_view_el_ = nullptr;
        search_container_el_ = nullptr;
        filter_input_el_ = nullptr;
        filter_clear_el_ = nullptr;
        selection_action_bar_el_ = nullptr;
        selection_action_count_el_ = nullptr;
        selection_clear_el_ = nullptr;
        selection_visibility_el_ = nullptr;
        selection_visibility_icon_el_ = nullptr;
        selection_training_el_ = nullptr;
        selection_training_icon_el_ = nullptr;
        selection_delete_el_ = nullptr;
        visible_icon_source_.clear();
        hidden_icon_source_.clear();
        locked_icon_source_.clear();
        unlocked_icon_source_.clear();
        empty_state_el_ = nullptr;
        empty_primary_el_ = nullptr;
        empty_secondary_el_ = nullptr;
    }

    void NativeScenePanel::reloadRmlResources() {
        filter_input_revert_.clear();
        host_.reloadDocument();
        document_ = nullptr;
        clearElementCache();
        last_language_.clear();
        last_prepare_frame_ = 0;
        has_last_sync_stamp_ = false;
    }

    void NativeScenePanel::preload(const PanelDrawContext& ctx) {
        if (!ensureInitialized())
            return;
        syncPanel(ctx);
    }

    PanelDirectRenderResult NativeScenePanel::renderDirect(
        const PanelDirectRenderRequest& request,
        const PanelDrawContext& ctx) {
        host_.setFloating(request.space == PanelSpace::Floating);
        if (request.mode == PanelDirectRenderMode::Measure)
            return {.handled = true, .height = host_.getContentHeight()};

        host_.setInputClipY(request.clip_y_min, request.clip_y_max);
        host_.setInput(request.input);
        host_.setForcedHeight(request.forced_height);

        bool handled = true;
        try {
            switch (request.mode) {
            case PanelDirectRenderMode::Measure:
                break;
            case PanelDirectRenderMode::Draw:
                drawDirect(request.x, request.y, request.width, request.height, ctx);
                break;
            case PanelDirectRenderMode::Cached:
                handled = drawDirectCached(request.x, request.y, request.width,
                                           request.height, ctx);
                break;
            case PanelDirectRenderMode::Preload:
                preloadDirect(request.width, request.height, ctx,
                              request.clip_y_min, request.clip_y_max, request.input);
                break;
            }
        } catch (...) {
            host_.setForcedHeight(0.0f);
            host_.setInput(nullptr);
            host_.setInputClipY(-1.0f, -1.0f);
            throw;
        }

        const float height = host_.getContentHeight();
        host_.setForcedHeight(0.0f);
        host_.setInput(nullptr);
        host_.setInputClipY(-1.0f, -1.0f);
        return {.handled = handled, .height = height};
    }

    void NativeScenePanel::preloadDirect(const float w, const float h,
                                         const PanelDrawContext& ctx,
                                         const float clip_y_min,
                                         const float clip_y_max,
                                         const PanelInputState* input) {
        host_.setInputClipY(clip_y_min, clip_y_max);
        host_.setInput(input);
        if (!ensureInitialized())
            return;

        if (shouldSyncPanel(input))
            syncPanel(ctx);
        last_prepare_frame_ = ctx.frame_serial;
        host_.prepareDirect(w, h);
    }

    void NativeScenePanel::draw(const PanelDrawContext& ctx) {
        if (!ensureInitialized())
            return;

        syncPanel(ctx);
        host_.draw(ctx);
    }

    void NativeScenePanel::drawDirect(const float x, const float y,
                                      const float w, const float h,
                                      const PanelDrawContext& ctx) {
        if (!ensureInitialized())
            return;

        if (tree_el_)
            tree_el_->setPanelScreenOffset(x, y);

        if (tree_el_ && tree_el_->needsAnimationFrame())
            host_.markContentDirty();

        if (last_prepare_frame_ != ctx.frame_serial)
            syncPanel(ctx);

        host_.drawDirect(x, y, w, h);
    }

    bool NativeScenePanel::drawDirectCached(const float x, const float y,
                                            const float w, const float h,
                                            const PanelDrawContext& ctx) {
        if (!ensureInitialized())
            return false;

        if (tree_el_)
            tree_el_->setPanelScreenOffset(x, y);

        // A context-menu result is normally consumed in syncPanel(), which only
        // runs on the live draw paths. Under render-on-demand an idle panel uses
        // this cached path, so poll the result here too — otherwise a menu action
        // (e.g. "Go to camera view") sits unconsumed until the panel next goes
        // live, i.e. until the next mouse move (a multi-second perceived lag).
        if (auto* gui = services().guiOrNull()) {
            const std::string action = gui->globalContextMenu().pollResult();
            if (!action.empty() && tree_el_ && tree_el_->executeContextMenuAction(action)) {
                syncPanel(ctx);
                host_.markContentDirty();
                host_.drawDirect(x, y, w, h);
                return true;
            }
        }

        if (shouldSyncPanel(nullptr)) {
            syncPanel(ctx);
            host_.drawDirect(x, y, w, h);
            return true;
        }

        return host_.drawDirectCached(x, y, w, h);
    }

    bool NativeScenePanel::ensureInitialized() {
        if (!host_.ensureDocumentLoaded())
            return false;

        if (document_)
            return true;

        document_ = host_.getDocument();
        if (!document_) {
            LOG_ERROR("NativeScenePanel: missing Rml document");
            return false;
        }

        cacheElements();
        return tree_el_ != nullptr;
    }

    void NativeScenePanel::cacheElements() {
        filter_input_revert_.clear();
        clearElementCache();
        tree_el_ = dynamic_cast<SceneGraphElement*>(document_->GetElementById("tree-container"));
        if (manager_)
            manager_->setActiveSceneGraphElement(tree_el_);
        scene_tab_el_ = document_->GetElementById("scene-tab");
        chip_row_el_ = document_->GetElementById("scene-chip-row");
        summary_model_chip_el_ = document_->GetElementById("summary-model-chip");
        summary_node_chip_el_ = document_->GetElementById("summary-node-chip");
        summary_filter_chip_el_ = document_->GetElementById("summary-filter-chip");
        scene_view_el_ = document_->GetElementById("scene-view");
        search_container_el_ = document_->GetElementById("search-container");
        filter_input_el_ = document_->GetElementById("filter-input");
        filter_clear_el_ = document_->GetElementById("filter-clear");
        selection_action_bar_el_ = document_->GetElementById("selection-action-bar");
        selection_action_count_el_ = document_->GetElementById("selection-action-count");
        selection_clear_el_ = document_->GetElementById("selection-clear");
        selection_visibility_el_ = document_->GetElementById("selection-visibility");
        selection_visibility_icon_el_ = document_->GetElementById("selection-visibility-icon");
        selection_training_el_ = document_->GetElementById("selection-training");
        selection_training_icon_el_ = document_->GetElementById("selection-training-icon");
        selection_delete_el_ = document_->GetElementById("selection-delete");
        empty_state_el_ = document_->GetElementById("empty-state");
        empty_primary_el_ = document_->GetElementById("empty-primary");
        empty_secondary_el_ = document_->GetElementById("empty-secondary");

        if (auto* search_icon = document_->GetElementById("search-icon")) {
            const std::string search_icon_source = resolveRmlImageSource("icon/scene/search.png");
            if (!search_icon_source.empty())
                search_icon->SetAttribute("src", search_icon_source);
        }

        if (auto* clear_icon = filter_clear_el_ ? filter_clear_el_->GetChild(0) : nullptr) {
            const std::string clear_icon_source = resolveRmlImageSource("icon/scene/x.png");
            if (!clear_icon_source.empty())
                clear_icon->SetAttribute("src", clear_icon_source);
        }
        if (auto* clear_icon = document_->GetElementById("selection-clear-icon")) {
            const std::string clear_icon_source = resolveRmlImageSource("icon/scene/x.png");
            if (!clear_icon_source.empty())
                clear_icon->SetAttribute("src", clear_icon_source);
        }

        visible_icon_source_ = resolveRmlImageSource("icon/scene/visible.png");
        hidden_icon_source_ = resolveRmlImageSource("icon/scene/hidden.png");
        unlocked_icon_source_ = resolveRmlImageSource("icon/scene/unlocked.png");
        locked_icon_source_ = resolveRmlImageSource("icon/scene/locked.png");
        if (selection_visibility_icon_el_ && !visible_icon_source_.empty())
            selection_visibility_icon_el_->SetAttribute("src", visible_icon_source_);
        if (selection_training_icon_el_ && !unlocked_icon_source_.empty())
            selection_training_icon_el_->SetAttribute("src", unlocked_icon_source_);
        if (auto* icon = document_->GetElementById("selection-delete-icon")) {
            const std::string source = resolveRmlImageSource("icon/scene/trash.png");
            if (!source.empty())
                icon->SetAttribute("src", source);
        }

        if (!tree_el_ || !scene_tab_el_ || !chip_row_el_ ||
            !summary_model_chip_el_ || !summary_node_chip_el_ ||
            !summary_filter_chip_el_ || !scene_view_el_ || !search_container_el_ ||
            !filter_input_el_ || !filter_clear_el_ || !selection_action_bar_el_ ||
            !selection_action_count_el_ || !selection_clear_el_ ||
            !selection_visibility_el_ || !selection_visibility_icon_el_ ||
            !selection_training_el_ || !selection_training_icon_el_ ||
            !selection_delete_el_ || !empty_state_el_ || !empty_primary_el_ ||
            !empty_secondary_el_) {
            LOG_ERROR("NativeScenePanel: missing required DOM elements");
            return;
        }

        filter_clear_el_->AddEventListener(Rml::EventId::Click, &listener_);
        filter_input_el_->AddEventListener("input", &listener_);
        selection_visibility_el_->AddEventListener(Rml::EventId::Click, &listener_);
        selection_clear_el_->AddEventListener(Rml::EventId::Click, &listener_);
        selection_training_el_->AddEventListener(Rml::EventId::Click, &listener_);
        selection_delete_el_->AddEventListener(Rml::EventId::Click, &listener_);
        filter_input_revert_.bind(filter_input_el_, [this](Rml::Element&) {
            applyFilterInputValue();
        });
        applyPendingTreeChrome();
    }

    void NativeScenePanel::syncPanel(const PanelDrawContext& ctx) {
        if (!tree_el_)
            return;

        bool changed = false;

        if (auto* gui = services().guiOrNull()) {
            const std::string action = gui->globalContextMenu().pollResult();
            if (!action.empty() && tree_el_->executeContextMenuAction(action))
                changed = true;
        }

        // Menu actions can mutate the scene after the frame context was captured.
        // Synchronize the tree with the generation recorded in the panel stamp.
        auto current_ctx = ctx;
        current_ctx.scene_generation = app_store().scene_generation.get();
        changed |= syncLocale();
        changed |= syncSceneState(current_ctx);
        changed |= syncSummaryChips();
        changed |= syncSelectionActions();
        changed |= syncSceneVisibility();

        if (changed)
            host_.markContentDirty();

        last_sync_stamp_ = makeSyncStamp();
        has_last_sync_stamp_ = true;
    }

    bool NativeScenePanel::shouldSyncPanel(const PanelInputState* input) const {
        if (!tree_el_ || !has_last_sync_stamp_)
            return true;
        if (hasDiscreteInputActivity(input))
            return true;
        return makeSyncStamp() != last_sync_stamp_;
    }

    NativeScenePanel::SyncStamp NativeScenePanel::makeSyncStamp() const {
        SyncStamp stamp;
        stamp.scene_graph_selection_markers = loadSceneGraphSelectionMarkersPreference();
        auto& store = app_store();
        stamp.language_generation = store.language_generation.get();
        stamp.dp_ratio_milli = manager_
                                   ? static_cast<int>(std::lround(manager_->getDpRatio() * 1000.0f))
                                   : 1000;

        stamp.scene_generation = store.scene_generation.get();
        stamp.selection_generation = store.selection_generation.get();
        stamp.render_settings_generation = store.render_settings_generation.get();
        if (auto* params = services().paramsOrNull())
            stamp.invert_masks = params->getActiveParams().invert_masks;

        stamp.num_gaussians = store.num_gaussians.get();
        stamp.training_running = store.training_running.get();
        stamp.training_state = store.training_state.get();
        stamp.eval_psnr_milli = optionalMetricMilli(store.eval_psnr.get());
        stamp.eval_ssim_milli = optionalMetricMilli(store.eval_ssim.get());

        return stamp;
    }

    bool NativeScenePanel::syncSceneState(const PanelDrawContext& ctx) {
        if (!tree_el_)
            return false;

        const auto* filter_input =
            dynamic_cast<const Rml::ElementFormControlInput*>(filter_input_el_);
        const std::string filter_text = filter_input ? filter_input->GetValue() : "";
        tree_el_->setFilterText(filter_text);
        tree_el_->setSelectionMarkersVisible(loadSceneGraphSelectionMarkersPreference());
        return tree_el_->syncFromScene(ctx);
    }

    bool NativeScenePanel::syncLocale() {
        const std::string language = lfs::event::LocalizationManager::getInstance().getCurrentLanguage();
        if (language == last_language_)
            return false;

        last_language_ = language;

        bool changed = false;
        changed |= setCachedText(scene_tab_el_, tr("window.scene"));
        changed |= setCachedAttribute(filter_input_el_, "placeholder", tr("scene.search"));
        changed |= setCachedText(empty_primary_el_, tr("scene.no_data_loaded"));
        changed |= setCachedText(empty_secondary_el_, tr("scene.use_file_menu"));
        return changed;
    }

    bool NativeScenePanel::syncSelectionActions() {
        if (!tree_el_)
            return false;

        const auto state = tree_el_->selectionActionState();
        const bool visible = state.count > 0;
        bool changed = false;
        changed |= setCachedProperty(selection_action_bar_el_, "display", visible ? "flex" : "none");
        if (!visible)
            return changed;

        changed |= setCachedText(selection_action_count_el_,
                                 LOCF(lichtfeld::Strings::Scene::SELECTED_COUNT, lfs::core::format_count(state.count)));
        changed |= setCachedAttribute(selection_visibility_el_, "data-tooltip",
                                      state.all_visible ? "scene.hide_selected" : "scene.show_selected");
        changed |= setCachedAttribute(selection_visibility_icon_el_, "src",
                                      state.all_visible ? visible_icon_source_ : hidden_icon_source_);
        changed |= setCachedClass(selection_visibility_icon_el_, "selection-hidden", !state.all_visible);
        changed |= setCachedClass(selection_training_el_, "disabled", !state.all_training_compatible);
        changed |= setCachedAttribute(selection_training_el_, "data-tooltip",
                                      state.all_training_enabled ? "scene.disable_for_training"
                                                                 : "scene.enable_for_training");
        changed |= setCachedAttribute(selection_training_icon_el_, "src",
                                      state.all_training_enabled ? unlocked_icon_source_ : locked_icon_source_);
        changed |= setCachedClass(selection_training_icon_el_, "training-disabled",
                                  !state.all_training_enabled);
        changed |= setCachedClass(selection_delete_el_, "disabled", !state.all_delete_enabled);
        return changed;
    }

    bool NativeScenePanel::syncSummaryChips() {
        if (!tree_el_)
            return false;

        bool changed = false;
        changed |= setCachedText(summary_model_chip_el_, formatSceneModelCount(tree_el_->modelCount()));
        changed |= setCachedText(summary_node_chip_el_, formatSceneNodeCount(tree_el_->nodeCount()));

        const bool show_filter = !tree_el_->filterText().empty();
        changed |= setCachedText(summary_filter_chip_el_,
                                 show_filter ? LOCF("runtime.scene_filter", tree_el_->filterText()) : std::string{});
        changed |= setCachedProperty(summary_filter_chip_el_, "display", show_filter ? "inline-block" : "none");
        changed |= setCachedProperty(chip_row_el_, "display",
                                     tree_el_->hasNodes() ? "flex" : "none");
        return changed;
    }

    bool NativeScenePanel::syncSceneVisibility() {
        if (!tree_el_)
            return false;

        const bool show_tree = tree_el_->hasNodes();
        const bool show_filter_clear = !tree_el_->filterText().empty();

        bool changed = false;
        changed |= setCachedProperty(empty_state_el_, "display",
                                     !show_tree ? "block" : "none");
        changed |= setCachedProperty(tree_el_, "display",
                                     show_tree ? "block" : "none");
        changed |= setCachedProperty(filter_clear_el_, "display",
                                     show_filter_clear ? "inline-block" : "none");
        return changed;
    }

    bool NativeScenePanel::handleEvent(Rml::Event& event) {
        const auto type = event.GetType();
        auto* current = event.GetCurrentElement();
        auto* target = event.GetTargetElement();
        if (!current && !target)
            return false;

        if (type == "input" && current == filter_input_el_) {
            applyFilterInputValue();
            return true;
        }

        if (type != "click" || !target)
            return false;

        const Rml::String id = target->GetId();
        const Rml::String current_id = current ? current->GetId() : "";
        if (id == "filter-clear" || current_id == "filter-clear") {
            if (auto* input = dynamic_cast<Rml::ElementFormControlInput*>(filter_input_el_))
                input->SetValue("");
            applyFilterInputValue();
            event.StopPropagation();
            return true;
        }
        if (id == "selection-visibility" || id == "selection-visibility-icon" ||
            current_id == "selection-visibility") {
            const auto state = tree_el_->selectionActionState();
            tree_el_->setSelectedVisibility(!state.all_visible);
            event.StopPropagation();
            return true;
        }
        if (id == "selection-clear" || id == "selection-clear-icon" ||
            current_id == "selection-clear") {
            tree_el_->clearSelectedNodes();
            event.StopPropagation();
            return true;
        }
        if (id == "selection-training" || id == "selection-training-icon" ||
            current_id == "selection-training") {
            const auto state = tree_el_->selectionActionState();
            if (state.all_training_compatible)
                tree_el_->setSelectedTrainingEnabled(!state.all_training_enabled);
            event.StopPropagation();
            return true;
        }
        if (id == "selection-delete" || current_id == "selection-delete") {
            tree_el_->requestDeleteSelection();
            event.StopPropagation();
            return true;
        }
        return false;
    }

    void NativeScenePanel::applyFilterInputValue() {
        const auto* input = dynamic_cast<const Rml::ElementFormControlInput*>(filter_input_el_);
        if (tree_el_)
            tree_el_->setFilterText(input ? input->GetValue() : "");
        syncSummaryChips();
        syncSelectionActions();
        syncSceneVisibility();
        host_.markContentDirty();
    }

    std::unordered_set<int> NativeScenePanel::visibleCameraUids() const {
        return tree_el_ ? tree_el_->visibleCameraUids() : std::unordered_set<int>{};
    }

    SceneTreeSessionChrome NativeScenePanel::captureTreeChrome(
        const core::Scene& scene) const {
        SceneTreeSessionChrome chrome;
        if (tree_el_) {
            chrome.collapsed_uuids =
                collapsedUuidsFromIds(scene, tree_el_->collapsedIds());
            chrome.models_collapsed = tree_el_->modelsCollapsed();
            chrome.filter_text = tree_el_->filterText();
        } else if (pending_tree_chrome_) {
            chrome = *pending_tree_chrome_;
        }
        if (const auto* input =
                dynamic_cast<const Rml::ElementFormControlInput*>(filter_input_el_))
            chrome.filter_text = input->GetValue();
        return chrome;
    }

    void NativeScenePanel::applyTreeChrome(const SceneTreeSessionChrome& chrome) {
        pending_tree_chrome_ = chrome;
        applyPendingTreeChrome();
    }

    void NativeScenePanel::resetTreeChrome() {
        pending_tree_chrome_.reset();
        if (tree_el_) {
            tree_el_->clearSessionCollapseUuids();
            tree_el_->setModelsCollapsed(false);
            tree_el_->setFilterText({});
        }
        if (auto* input = dynamic_cast<Rml::ElementFormControlInput*>(filter_input_el_))
            input->SetValue("");
        host_.markContentDirty();
    }

    bool NativeScenePanel::selectAllIfFocused() {
        return tree_el_ && tree_el_->selectAllIfFocused();
    }

    bool NativeScenePanel::toggleSelectionVisibilityIfFocused() {
        return tree_el_ && tree_el_->toggleSelectedVisibilityIfFocused();
    }

    bool NativeScenePanel::toggleSelectionTrainingIfFocused() {
        return tree_el_ && tree_el_->toggleSelectedTrainingIfFocused();
    }

    bool NativeScenePanel::groupSelectedNodesIfFocused() {
        if (!tree_el_ || !tree_el_->IsPseudoClassSet("focus") || tree_el_->selectedCount() < 2)
            return false;
        (void)tree_el_->executeContextMenuAction("scene_panel:group_selected");
        return true;
    }

    bool NativeScenePanel::ungroupSelectedNodeIfFocused() {
        if (!tree_el_ || !tree_el_->IsPseudoClassSet("focus") || tree_el_->selectedCount() != 1)
            return false;
        (void)tree_el_->executeContextMenuAction("scene_panel:ungroup_selected");
        return true;
    }

    bool NativeScenePanel::requestDeleteSelectionIfAvailable() {
        if (!tree_el_ || tree_el_->selectedCount() == 0)
            return false;
        tree_el_->requestDeleteSelection();
        return true;
    }

    void NativeScenePanel::applyPendingTreeChrome() {
        if (!pending_tree_chrome_)
            return;
        const auto& chrome = *pending_tree_chrome_;
        if (tree_el_) {
            tree_el_->applySessionCollapseUuids(chrome.collapsed_uuids);
            tree_el_->setModelsCollapsed(chrome.models_collapsed);
            tree_el_->setFilterText(chrome.filter_text);
        }
        if (auto* input = dynamic_cast<Rml::ElementFormControlInput*>(filter_input_el_))
            input->SetValue(chrome.filter_text);
        host_.markContentDirty();
    }

} // namespace lfs::vis::gui
