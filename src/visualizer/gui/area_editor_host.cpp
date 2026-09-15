/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/area_editor_host.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/logger.hpp"

#include "gui/rmlui/rml_panel_host.hpp"
#include "gui/rmlui/rmlui_manager.hpp"

#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/ElementDocument.h>
#include <RmlUi/Core/Elements/ElementFormControlSelect.h>
#include <RmlUi/Core/EventListener.h>

#include <algorithm>
#include <cfloat>
#include <format>
#include <limits>
#include <map>
#include <unordered_set>
#include <utility>

namespace lfs::vis::gui {

    namespace {
        constexpr std::string_view kViewportEditor = "viewport";

        struct HeaderListener final : Rml::EventListener {
            AreaEditorHost* owner = nullptr;
            ViewId area = kInvalidViewId;

            void ProcessEvent(Rml::Event& event) override {
                if (!owner)
                    return;
                auto* const target = event.GetTargetElement();
                if (!target)
                    return;

                if (event == Rml::EventId::Change && target->GetId() == "editor-select") {
                    const auto* const select =
                        dynamic_cast<Rml::ElementFormControlSelect*>(target);
                    if (select)
                        owner->handleAction(area, AreaEditorAction::SetEditor,
                                            select->GetValue());
                    return;
                }
                if (event != Rml::EventId::Click)
                    return;
                // Find the button when the click targets its child icon.
                auto* action_target = target;
                while (action_target &&
                       action_target->GetAttribute<Rml::String>("data-action", "").empty())
                    action_target = action_target->GetParentNode();
                if (!action_target)
                    return;
                const auto action = action_target->GetAttribute<Rml::String>("data-action", "");
                if (action == "close") {
                    owner->handleAction(area, AreaEditorAction::Close, {});
                } else if (action == "maximize") {
                    owner->handleAction(area, AreaEditorAction::ToggleMaximize, {});
                } else if (action == "projection") {
                    owner->handleAction(area, AreaEditorAction::ToggleProjection, {});
                }
            }
        };
    } // namespace

    struct AreaEditorHost::Header {
        // The document detaches listeners during destruction, so the listener
        // must outlive the host that owns that document.
        HeaderListener listener;
        std::unique_ptr<RmlPanelHost> host;
        bool listener_bound = false;
        uint64_t options_revision = 0;
        std::string options_markup;
        std::string maximize_icon;
        std::string maximize_tooltip;
        std::string projection_tooltip;
        std::string responsive_mode;

        Header() = default;
        Header(const Header&) = delete;
        Header& operator=(const Header&) = delete;
        Header(Header&&) noexcept = default;
        Header& operator=(Header&&) noexcept = default;
    };

    struct AreaEditorHost::Area {
        ViewId id = kInvalidViewId;
        std::string editor_id = std::string(kViewportEditor);
        uint64_t editor_generation = 0;
        std::shared_ptr<IPanel> panel;
        std::unique_ptr<Header> header;
        uint64_t factory_attempt_revision = std::numeric_limits<uint64_t>::max();
        int consecutive_errors = 0;

        Area() = default;
        Area(const Area&) = delete;
        Area& operator=(const Area&) = delete;
        Area(Area&&) noexcept = default;
        Area& operator=(Area&&) noexcept = default;
    };

    AreaEditorHost::AreaEditorHost(PanelRegistry& registry, ViewportWorkspace& workspace,
                                   RmlUIManager* rml_manager)
        : registry_(registry), workspace_(workspace), rml_manager_(rml_manager) {}

    AreaEditorHost::~AreaEditorHost() {
        // Destruction is expected on the GUI thread. Releasing the remaining
        // hosts here also makes shutdown deterministic when no further frame
        // retirement callback is available.
        retired_.clear();
        areas_.clear();
    }

    std::string AreaEditorHost::instanceName(const ViewId id, const uint64_t generation,
                                             const std::string_view editor) {
        std::string safe_editor;
        safe_editor.reserve(editor.size());
        for (const char c : editor)
            safe_editor.push_back((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                                          (c >= '0' && c <= '9')
                                      ? c
                                      : '_');
        return std::format("lfs.area.{}.{}.{}", id, generation, safe_editor);
    }

    std::string AreaEditorHost::editorLabel(const std::string_view editor) {
        if (editor == "viewport")
            return "Viewport";
        if (editor == "lfs.scene")
            return "Scene";
        if (editor == "lfs.rendering")
            return "Rendering";
        if (editor == "lfs.training")
            return "Training";
        return std::string(editor);
    }

    namespace {
        [[nodiscard]] std::string escapeRml(const std::string_view value) {
            std::string escaped;
            escaped.reserve(value.size());
            for (const char c : value) {
                switch (c) {
                case '&': escaped += "&amp;"; break;
                case '<': escaped += "&lt;"; break;
                case '>': escaped += "&gt;"; break;
                case '"': escaped += "&quot;"; break;
                default: escaped.push_back(c); break;
                }
            }
            return escaped;
        }
    } // namespace

    std::string AreaEditorHost::editorOptionsMarkup(const std::string_view current_editor) const {
        // Editor types remain available when their legacy dock is closed.
        // A child panel is part of its parent editor, not a standalone area.
        constexpr PanelSpace spaces[] = {
            PanelSpace::MainPanelTab, PanelSpace::SceneHeader, PanelSpace::SidePanel,
            PanelSpace::BottomDock, PanelSpace::LeftDock, PanelSpace::Floating,
            PanelSpace::ViewportOverlay, PanelSpace::StatusBar};
        std::map<std::string, std::string> editors;
        editors.emplace("viewport", "Viewport");
        for (const auto space : spaces) {
            for (const auto& id : registry_.get_panel_names(space)) {
                const auto details = registry_.get_panel(id);
                const auto panel = registry_.get_panel_instance(id);
                if (details && details->parent_id.empty() && panel && panel->supportsAreaInstances())
                    editors.emplace(id, details->label.empty() ? editorLabel(id) : details->label);
            }
        }
        if (!current_editor.empty() && !editors.contains(std::string(current_editor)))
            editors.emplace(std::string(current_editor), editorLabel(current_editor));

        std::string markup;
        for (const auto& [id, label] : editors)
            markup += std::format("<option value=\"{}\">{}</option>", escapeRml(id),
                                  escapeRml(label));
        return markup;
    }

    uint64_t AreaEditorHost::editorOptionsRevision() const {
        return registry_.registration_revision() ^
               (registry_.visibility_revision() + 0x9e3779b97f4a7c15ULL +
                (registry_.registration_revision() << 6) +
                (registry_.registration_revision() >> 2));
    }

    void AreaEditorHost::tryCreatePanel(Area& area, const AreaSnapshot& snapshot) {
        if (snapshot.editor_id == kViewportEditor || area.panel)
            return;
        const auto revision = editorOptionsRevision();
        if (area.factory_attempt_revision == revision)
            return;
        area.factory_attempt_revision = revision;
        const auto instance_id = instanceName(snapshot.id, area.editor_generation,
                                              snapshot.editor_id);
        try {
            area.panel = registry_.create_area_instance(snapshot.editor_id, instance_id);
        } catch (const std::exception& error) {
            LOG_ERROR("Area editor '{}' factory failed for view {}: {}",
                      snapshot.editor_id, snapshot.id, error.what());
            return;
        } catch (...) {
            LOG_ERROR("Area editor '{}' factory failed for view {}",
                      snapshot.editor_id, snapshot.id);
            return;
        }
        if (area.panel) {
            if (const auto* record = workspace_.findView(snapshot.id)) {
                const auto chrome = record->chrome.find(snapshot.editor_id);
                if (chrome != record->chrome.end())
                    area.panel->applyChromeJson(chrome->second);
            }
        }
    }

    void AreaEditorHost::retireArea(Area&& area) {
        if (area.panel || area.header)
            retired_.push_back(std::move(area));
    }

    void AreaEditorHost::synchronizeArea(const AreaSnapshot& snapshot) {
        auto [it, inserted] = areas_.try_emplace(snapshot.id);
        auto& area = it->second;
        if (inserted) {
            area.id = snapshot.id;
            area.editor_id = std::string(kViewportEditor);
        }

        if (area.editor_id == snapshot.editor_id && (!rml_manager_ || area.header) &&
            (snapshot.editor_id == kViewportEditor || area.panel))
            return;

        // A Python panel may register after the area was first displayed. Keep
        // its header/context and retry only the missing factory instance.
        if (area.editor_id == snapshot.editor_id && snapshot.editor_id != kViewportEditor &&
            !area.panel) {
            tryCreatePanel(area, snapshot);
            return;
        }

        if (area.panel || area.header) {
            if (area.panel)
                captureChrome();
            Area retired;
            retired.id = area.id;
            retired.editor_id = std::move(area.editor_id);
            retired.editor_generation = area.editor_generation;
            retired.panel = std::move(area.panel);
            retired.header = std::move(area.header);
            retireArea(std::move(retired));
        }

        area.editor_generation = next_editor_generation_[snapshot.id]++;
        area.editor_id = snapshot.editor_id;
        // Retry failed factories after an editor switch, even if the registry is unchanged.
        area.factory_attempt_revision = std::numeric_limits<uint64_t>::max();
        area.consecutive_errors = 0;
        if (rml_manager_) {
            area.header = std::make_unique<Header>();
            area.header->listener.owner = this;
            area.header->listener.area = snapshot.id;
            area.header->host = std::make_unique<RmlPanelHost>(
                rml_manager_, instanceName(snapshot.id, area.editor_generation, area.editor_id) + ".header",
                "rmlui/area_editor_header.rml");
            area.header->host->setForeground(true);
        }

        if (snapshot.editor_id == kViewportEditor)
            return;
        tryCreatePanel(area, snapshot);
    }

    void AreaEditorHost::synchronize(const WorkspaceFrameSnapshot& snapshot) {
        std::unordered_set<ViewId> desired;
        desired.reserve(snapshot.areas.size() + snapshot.live_ids.size());
        for (const ViewId id : snapshot.live_ids)
            if (isValidViewId(id))
                desired.insert(id);
        for (const auto& area : snapshot.areas) {
            if (!isValidViewId(area.id))
                continue;
            desired.insert(area.id);
            synchronizeArea(area);
        }

        for (auto it = areas_.begin(); it != areas_.end();) {
            if (desired.contains(it->first)) {
                ++it;
                continue;
            }
            retireArea(std::move(it->second));
            it = areas_.erase(it);
        }
    }

    void AreaEditorHost::captureChrome() {
        for (auto& [id, area] : areas_) {
            if (!area.panel)
                continue;
            auto* const record = workspace_.findView(id);
            if (!record)
                continue;
            const auto chrome = area.panel->captureChromeJson();
            if (chrome.empty())
                continue;
            const auto current = record->chrome.find(area.editor_id);
            if (current != record->chrome.end() && current->second == chrome)
                continue;
            record->chrome[area.editor_id] = chrome;
            static_cast<void>(workspace_.bumpViewState(id));
        }
    }

    PanelInputState AreaEditorHost::filteredInput(
        const PanelInputState& input, const AreaSnapshot& area,
        const std::optional<ViewId> pointer_owner,
        const std::optional<ViewId> keyboard_owner) const {
        PanelInputState result = input;
        const bool pointer_allowed = pointer_owner ? *pointer_owner == area.id : area.focused;
        const bool keyboard_allowed = keyboard_owner ? *keyboard_owner == area.id : area.focused;

        if (!pointer_allowed) {
            // Keep screen origin intact: RmlPanelHost uses it when composing
            // the panel texture. Only the local cursor position is moved out
            // of the panel so hover/capture cannot leak between areas.
            result.mouse_x = -FLT_MAX;
            result.mouse_y = -FLT_MAX;
            std::fill(std::begin(result.mouse_down), std::end(result.mouse_down), false);
            std::fill(std::begin(result.mouse_clicked), std::end(result.mouse_clicked), false);
            std::fill(std::begin(result.mouse_released), std::end(result.mouse_released), false);
            result.mouse_wheel = 0.0f;
            result.mouse_wheel_x = 0.0f;
            result.mouse_button_events.clear();
        }
        if (!keyboard_allowed) {
            result.key_ctrl = false;
            result.key_shift = false;
            result.key_alt = false;
            result.key_super = false;
            result.viewport_keyboard_focus = false;
            result.keys_pressed.clear();
            result.keys_repeated.clear();
            result.keys_released.clear();
            result.text_codepoints.clear();
            result.text_inputs.clear();
            result.text_editing.clear();
            result.text_editing_start = -1;
            result.text_editing_length = -1;
            result.has_text_editing = false;
        }
        return result;
    }

    void AreaEditorHost::render(const WorkspaceFrameSnapshot& snapshot,
                                const PanelDrawContext& base_context,
                                const PanelInputState& input,
                                const int header_pixels,
                                const std::optional<ViewId> pointer_owner,
                                const std::optional<ViewId> keyboard_owner) {
        synchronize(snapshot);
        const float header_h = static_cast<float>(std::max(header_pixels, 0));

        for (const auto& snapshot_area : snapshot.areas) {
            const auto it = areas_.find(snapshot_area.id);
            if (it == areas_.end() || !it->second.header)
                continue;
            auto& area = it->second;
            const auto area_input = filteredInput(input, snapshot_area, pointer_owner,
                                                  keyboard_owner);
            auto* header_document = area.header->host->getDocument();
            auto* header_select = header_document
                                      ? dynamic_cast<Rml::ElementFormControlSelect*>(
                                            header_document->GetElementById("editor-select"))
                                      : nullptr;
            const bool header_owns_pointer =
                (header_select && header_select->IsSelectBoxVisible()) ||
                (area_input.mouse_y >= snapshot_area.rect.y &&
                 area_input.mouse_y < snapshot_area.content_rect.y);
            auto content_input = area_input;
            if (header_owns_pointer) {
                content_input = filteredInput(input, snapshot_area, kInvalidViewId, kInvalidViewId);
            }
            if (area.header->host->ensureDocumentLoaded()) {
                if (!area.header->listener_bound) {
                    area.header->host->getDocument()->AddEventListener(Rml::EventId::Click,
                                                                       &area.header->listener);
                    area.header->host->getDocument()->AddEventListener(Rml::EventId::Change,
                                                                       &area.header->listener);
                    area.header->listener_bound = true;
                    header_document = area.header->host->getDocument();
                }

                if (header_document) {
                    bool header_changed = false;
                    const float ui_scale = header_h > 0.0f ? header_h / 26.0f : 1.0f;
                    const float area_width_dp =
                        static_cast<float>(snapshot_area.rect.width) / std::max(ui_scale, 1.0e-3f);
                    std::string responsive_mode;
                    if (area_width_dp < 145.0f)
                        responsive_mode = "narrow";
                    else if (area_width_dp < 190.0f)
                        responsive_mode = "compact";
                    if (auto* const body = header_document->GetElementById("area-header");
                        body && area.header->responsive_mode != responsive_mode) {
                        body->SetClass("compact", !responsive_mode.empty());
                        body->SetClass("narrow", responsive_mode == "narrow");
                        area.header->responsive_mode = responsive_mode;
                        header_changed = true;
                    }
                    const auto options_revision = editorOptionsRevision();
                    if (auto* const select = dynamic_cast<Rml::ElementFormControlSelect*>(
                            header_document->GetElementById("editor-select"))) {
                        if (area.header->options_revision != options_revision ||
                            area.header->options_markup.empty()) {
                            area.header->options_markup = editorOptionsMarkup(snapshot_area.editor_id);
                            select->SetInnerRML(area.header->options_markup);
                            area.header->options_revision = options_revision;
                            header_changed = true;
                        }
                    }
                    const std::string maximize_icon = snapshot_area.maximized
                                                          ? "../icon/arrows-minimize.png"
                                                          : "../icon/arrows-maximize.png";
                    if (auto* const maximize = header_document->GetElementById("maximize-icon");
                        maximize && area.header->maximize_icon != maximize_icon) {
                        maximize->SetAttribute("src", maximize_icon);
                        area.header->maximize_icon = maximize_icon;
                        header_changed = true;
                    }
                    if (auto* const maximize = header_document->GetElementById("maximize")) {
                        const std::string tooltip = LOC(snapshot_area.maximized
                                                            ? "ui.restore_area"
                                                            : "ui.maximize_area");
                        if (area.header->maximize_tooltip != tooltip) {
                            maximize->SetAttribute("title", tooltip);
                            maximize->SetAttribute("aria-label", tooltip);
                            area.header->maximize_tooltip = tooltip;
                            header_changed = true;
                        }
                    }
                    if (auto* const projection = header_document->GetElementById("projection")) {
                        const auto* const record = workspace_.findView(snapshot_area.id);
                        const std::string projection_tooltip =
                            LOC(record && record->projection.orthographic
                                    ? "ui.use_perspective_projection"
                                    : "ui.use_orthographic_projection");
                        if (area.header->projection_tooltip != projection_tooltip) {
                            projection->SetAttribute("title", projection_tooltip);
                            projection->SetAttribute("aria-label", projection_tooltip);
                            area.header->projection_tooltip = projection_tooltip;
                            header_changed = true;
                        }
                        projection->SetClass("hidden", snapshot_area.editor_id != kViewportEditor);
                    }
                    if (auto* const select = dynamic_cast<Rml::ElementFormControlSelect*>(
                            header_document->GetElementById("editor-select"));
                        select && select->GetValue() != snapshot_area.editor_id) {
                        select->SetValue(snapshot_area.editor_id);
                        header_changed = true;
                    }

                    if (header_changed)
                        area.header->host->markContentDirty();
                }

                area.header->host->setInput(&area_input);
                if (snapshot_area.rect.width > 0 && snapshot_area.rect.height > 0 &&
                    header_h > 0.0f) {
                    area.header->host->drawDirect(static_cast<float>(snapshot_area.rect.x),
                                                  static_cast<float>(snapshot_area.rect.y),
                                                  static_cast<float>(snapshot_area.rect.width),
                                                  static_cast<float>(snapshot_area.rect.height));
                }
                area.header->host->setInput(nullptr);
            }

            renderPanel(area, snapshot_area, base_context, PanelDirectRenderMode::Draw, &content_input);
        }
    }

    void AreaEditorHost::renderPanel(Area& area, const AreaSnapshot& snapshot,
                                     const PanelDrawContext& base_context,
                                     const PanelDirectRenderMode mode,
                                     const PanelInputState* input) {
        if (!area.panel || snapshot.content_rect.empty() || base_context.ui_hidden ||
            area.consecutive_errors >= PanelInfo::MAX_CONSECUTIVE_ERRORS)
            return;
        const auto info = registry_.get_panel(area.editor_id);
        if (!info || (base_context.suppress_non_native_panels && !info->is_native))
            return;

        PanelDrawContext context = base_context;
        context.bounds = PanelDrawBounds{
            static_cast<float>(snapshot.content_rect.x),
            static_cast<float>(snapshot.content_rect.y),
            static_cast<float>(snapshot.content_rect.width),
            static_cast<float>(snapshot.content_rect.height)};
        context.screen_bounds = context.bounds;
        try {
            if (!area.panel->poll(context))
                return;
            area.panel->renderDirect(
                PanelDirectRenderRequest{
                    .mode = mode,
                    .space = info->space,
                    .x = context.bounds->x,
                    .y = context.bounds->y,
                    .width = context.bounds->width,
                    .height = context.bounds->height,
                    .clip_y_min = static_cast<float>(snapshot.content_rect.y),
                    .clip_y_max = static_cast<float>(snapshot.content_rect.bottom()),
                    .input = input},
                context);
            area.consecutive_errors = 0;
        } catch (const std::exception& error) {
            ++area.consecutive_errors;
            LOG_ERROR("Area editor '{}' in area {} failed: {}", area.editor_id, area.id, error.what());
        } catch (...) {
            ++area.consecutive_errors;
            LOG_ERROR("Area editor '{}' in area {} failed", area.editor_id, area.id);
        }
    }

    void AreaEditorHost::preloadVisiblePanels(const WorkspaceFrameSnapshot& snapshot,
                                              const PanelDrawContext& context) {
        synchronize(snapshot);
        for (const auto& area : snapshot.areas) {
            if (auto it = areas_.find(area.id); it != areas_.end())
                renderPanel(it->second, area, context, PanelDirectRenderMode::Preload, nullptr);
        }
    }

    void AreaEditorHost::collectRetired() { retired_.clear(); }

    void AreaEditorHost::retireAll() {
        for (auto& [id, area] : areas_) {
            (void)id;
            retireArea(std::move(area));
        }
        areas_.clear();
    }

    std::size_t AreaEditorHost::retiredAreaCount() const noexcept { return retired_.size(); }

    bool AreaEditorHost::needsAnimationFrame() const {
        for (const auto& [id, area] : areas_) {
            (void)id;
            if ((area.panel && (area.panel->needsAnimationFrame() ||
                                area.panel->needsImmediateAnimationFrame())) ||
                (area.header && area.header->host && area.header->host->needsAnimationFrame()))
                return true;
        }
        return false;
    }

    void AreaEditorHost::releaseRendererResources() {
        for (auto& [id, area] : areas_) {
            (void)id;
            if (area.panel)
                area.panel->releaseRendererResources();
            if (area.header && area.header->host)
                area.header->host->releaseRendererResources();
        }
        for (auto& area : retired_) {
            if (area.panel)
                area.panel->releaseRendererResources();
            if (area.header && area.header->host)
                area.header->host->releaseRendererResources();
        }
    }

    void AreaEditorHost::handleAction(const ViewId id, const AreaEditorAction action,
                                      const std::string_view editor) {
        switch (action) {
        case AreaEditorAction::SetEditor:
            if (!editor.empty())
                static_cast<void>(workspace_.setAreaEditor(id, std::string(editor)));
            break;
        case AreaEditorAction::Close:
            static_cast<void>(workspace_.close(id));
            break;
        case AreaEditorAction::ToggleMaximize: {
            const auto snapshot = workspace_.snapshot({0, 0, 1, 1});
            if (snapshot.maximized && *snapshot.maximized == id)
                static_cast<void>(workspace_.restoreMaximized());
            else
                static_cast<void>(workspace_.maximize(id));
            break;
        }
        case AreaEditorAction::ToggleProjection: {
            const auto* const record = workspace_.findView(id);
            if (!record)
                break;
            auto projection = record->projection;
            projection.orthographic = !projection.orthographic;
            static_cast<void>(workspace_.setViewProjection(id, projection));
            break;
        }
        }
    }

} // namespace lfs::vis::gui
