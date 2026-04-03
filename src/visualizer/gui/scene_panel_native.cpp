/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/scene_panel_native.hpp"

#include "core/event_bridge/localization_manager.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "gui/gui_manager.hpp"
#include "gui/rmlui/elements/scene_graph_element.hpp"
#include "gui/string_keys.hpp"
#include "gui/utils/native_file_dialog.hpp"
#include "operation/undo_history.hpp"
#include "visualizer/core/services.hpp"

#include <RmlUi/Core.h>
#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/Elements/ElementFormControlSelect.h>
#include <SDL3/SDL_clipboard.h>

#include <algorithm>
#include <array>
#include <ctime>
#include <format>
#include <fstream>
#include <limits>
#include <string_view>

namespace lfs::vis::gui {

    namespace {
        constexpr size_t MAX_RENDERED_LOG_ENTRIES = 250;

        [[nodiscard]] std::string tr(const char* key) {
            const std::string value = LOC(key);
            return value.empty() ? std::string(key) : value;
        }

        [[nodiscard]] std::string replaceUnderscores(std::string value) {
            std::replace(value.begin(), value.end(), '_', ' ');
            return value;
        }

        [[nodiscard]] std::string pluralize(const size_t count,
                                            std::string_view singular,
                                            std::string_view plural = {}) {
            if (count == 1)
                return std::format("{} {}", count, singular);
            if (!plural.empty())
                return std::format("{} {}", count, plural);
            return std::format("{} {}s", count, singular);
        }

        [[nodiscard]] std::string formatBytes(const size_t value) {
            constexpr std::array units{"B", "KB", "MB", "GB"};
            double amount = static_cast<double>(value);
            size_t unit_index = 0;
            while (amount >= 1024.0 && unit_index + 1 < units.size()) {
                amount /= 1024.0;
                ++unit_index;
            }
            if (unit_index == 0)
                return std::format("{} {}", static_cast<int>(amount), units[unit_index]);
            return std::format("{:.1f} {}", amount, units[unit_index]);
        }

        [[nodiscard]] std::string cacheAttrName(std::string_view kind, std::string_view name) {
            return std::format("data-lfs-{}-{}", kind, name);
        }

        struct LogLevelChoice {
            core::LogLevel level;
            std::string_view label;
            std::string_view css_suffix;
        };

        constexpr std::array LOG_LEVEL_CHOICES{
            LogLevelChoice{core::LogLevel::Trace, "Trace", "trace"},
            LogLevelChoice{core::LogLevel::Debug, "Debug", "debug"},
            LogLevelChoice{core::LogLevel::Info, "Info", "info"},
            LogLevelChoice{core::LogLevel::Performance, "Performance", "perf"},
            LogLevelChoice{core::LogLevel::Warn, "Warn", "warn"},
            LogLevelChoice{core::LogLevel::Error, "Error", "error"},
            LogLevelChoice{core::LogLevel::Critical, "Critical", "critical"},
            LogLevelChoice{core::LogLevel::Off, "Off", "off"},
        };

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

        [[nodiscard]] std::string formatHistorySummary(const op::UndoHistory& history) {
            const auto undo_items = history.undoItems();
            const auto redo_items = history.redoItems();
            const size_t total_bytes = history.totalBytes();
            const size_t total_gpu_bytes = history.totalMemory().gpu_bytes;

            if (undo_items.empty() && redo_items.empty())
                return "No history yet";

            if (total_gpu_bytes < total_bytes) {
                return std::format("{} undo / {} redo · {} total · {} GPU",
                                   undo_items.size(), redo_items.size(),
                                   formatBytes(total_bytes),
                                   formatBytes(total_gpu_bytes));
            }

            return std::format("{} undo / {} redo · {}",
                               undo_items.size(), redo_items.size(),
                               formatBytes(total_bytes));
        }

        [[nodiscard]] std::string formatHistoryTransaction(const op::UndoHistory& history) {
            if (!history.hasActiveTransaction())
                return {};

            const std::string name = history.activeTransactionName().empty()
                                         ? "Grouped changes"
                                         : history.activeTransactionName();
            return std::format("Transaction active: {} (depth {})",
                               name, history.transactionDepth());
        }

        [[nodiscard]] std::string historyRowsHtml(const std::vector<op::UndoStackItem>& items,
                                                  std::string_view kind) {
            std::string html;
            html.reserve(items.size() * 256);

            for (size_t index = 0; index < items.size(); ++index) {
                const auto& item = items[index];
                const size_t estimated_bytes = item.estimated_bytes;
                const size_t gpu_bytes = item.gpu_bytes;
                const std::string size_meta = gpu_bytes < estimated_bytes
                                                  ? std::format("{} · GPU {}",
                                                                formatBytes(estimated_bytes),
                                                                formatBytes(gpu_bytes))
                                                  : formatBytes(estimated_bytes);
                const std::string scope = replaceUnderscores(
                    item.metadata.scope.empty() ? std::string("general") : item.metadata.scope);
                const std::string source =
                    item.metadata.source.empty() ? std::string("system") : item.metadata.source;
                const std::string label =
                    item.metadata.label.empty() ? std::string("Untitled Change") : item.metadata.label;
                const bool is_next = index == 0;
                const std::string stack_line = is_next
                                                   ? std::format("NEXT {} · Top of stack",
                                                                 kind == "undo" ? "UNDO" : "REDO")
                                                   : std::format("{} · {}", scope, source);
                const std::string detail_line = is_next
                                                    ? std::format("{} · {} · Size: {}",
                                                                  scope, source, size_meta)
                                                    : std::format("Size: {}", size_meta);
                const std::string row_classes = std::format(
                    "btn btn--ghost history-row{}{}",
                    kind == "redo" ? " history-row--redo" : "",
                    is_next ? " is-next" : "");

                html += std::format(
                    R"(<button type="button" class="{}" style="display:block; width:100%;" data-kind="{}" data-steps="{}"><span class="history-row__line history-row__line--primary text-default">&#9679; {}</span><br /><span class="history-row__line history-row__line--stack text-muted">{}</span><br /><span class="history-row__line history-row__line--secondary text-muted">{}</span></button>)",
                    row_classes,
                    kind,
                    index + 1,
                    Rml::StringUtilities::EncodeRml(label),
                    Rml::StringUtilities::EncodeRml(stack_line),
                    Rml::StringUtilities::EncodeRml(detail_line));
            }

            return html;
        }

        [[nodiscard]] Rml::Element* findHistoryActionTarget(Rml::Element* target) {
            while (target) {
                const auto kind = target->GetAttribute<Rml::String>("data-kind", "");
                if (!kind.empty())
                    return target;
                target = target->GetParentNode();
            }
            return nullptr;
        }

        [[nodiscard]] int logLevelSelectionIndex(const core::LogLevel level) {
            for (size_t index = 0; index < LOG_LEVEL_CHOICES.size(); ++index) {
                if (LOG_LEVEL_CHOICES[index].level == level)
                    return static_cast<int>(index);
            }
            return 2;
        }

        [[nodiscard]] core::LogLevel logLevelFromSelection(const int selection) {
            if (selection >= 0 && selection < static_cast<int>(LOG_LEVEL_CHOICES.size()))
                return LOG_LEVEL_CHOICES[static_cast<size_t>(selection)].level;
            return core::LogLevel::Info;
        }

        [[nodiscard]] std::string_view logLevelLabel(const core::LogLevel level) {
            return LOG_LEVEL_CHOICES[static_cast<size_t>(logLevelSelectionIndex(level))].label;
        }

        [[nodiscard]] std::string_view logLevelCssSuffix(const core::LogLevel level) {
            return LOG_LEVEL_CHOICES[static_cast<size_t>(logLevelSelectionIndex(level))].css_suffix;
        }

        [[nodiscard]] std::string formatLogTimestamp(const std::chrono::system_clock::time_point& timestamp) {
            const auto time_t_val = std::chrono::system_clock::to_time_t(timestamp);
            std::tm tm{};
#ifdef WIN32
            localtime_s(&tm, &time_t_val);
#else
            localtime_r(&time_t_val, &tm);
#endif
            const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    timestamp.time_since_epoch())
                                    .count() %
                                1000;
            return std::format("{:02}:{:02}:{:02}.{:03}",
                               tm.tm_hour,
                               tm.tm_min,
                               tm.tm_sec,
                               static_cast<int>(millis));
        }

        [[nodiscard]] std::string formatLoggingSummary(const size_t entry_count,
                                                       const size_t displayed_entry_count,
                                                       const core::LogLevel level) {
            const std::string level_label(logLevelLabel(level));
            if (entry_count == 0)
                return std::format("No buffered CLI logs · Level {}", level_label);
            if (displayed_entry_count < entry_count) {
                return std::format("{} buffered log {} · showing latest {} · Level {}",
                                   entry_count,
                                   entry_count == 1 ? "entry" : "entries",
                                   displayed_entry_count,
                                   level_label);
            }
            return std::format("{} buffered log {} · Level {}",
                               entry_count,
                               entry_count == 1 ? "entry" : "entries",
                               level_label);
        }

        [[nodiscard]] std::string loggingRowsHtml(const std::vector<core::LogEntrySnapshot>& entries) {
            const size_t rendered_entry_count = std::min(entries.size(), MAX_RENDERED_LOG_ENTRIES);
            std::string html;
            html.reserve(rendered_entry_count * 256);

            size_t rendered = 0;
            for (auto it = entries.rbegin(); it != entries.rend() && rendered < rendered_entry_count;
                 ++it, ++rendered) {
                const auto& entry = *it;
                const std::string timestamp = formatLogTimestamp(entry.timestamp);
                const std::string source = entry.line > 0
                                               ? std::format("{}:{}",
                                                             entry.file.empty() ? std::string("runtime")
                                                                                : entry.file,
                                                             entry.line)
                                               : (entry.file.empty() ? std::string("runtime")
                                                                     : entry.file);
                html += std::format(
                    R"(<div class="log-entry log-entry--{}"><div class="log-entry__header"><span class="log-entry__level">{}</span><span class="log-entry__meta text-muted">{} · {}</span></div><div class="log-entry__message text-default">{}</div></div>)",
                    logLevelCssSuffix(entry.level),
                    Rml::StringUtilities::EncodeRml(std::string(logLevelLabel(entry.level))),
                    Rml::StringUtilities::EncodeRml(timestamp),
                    Rml::StringUtilities::EncodeRml(source),
                    Rml::StringUtilities::EncodeRml(entry.message));
            }

            return html;
        }

    } // namespace

    NativeScenePanel::NativeScenePanel(RmlUIManager* manager)
        : manager_(manager),
          host_(manager, "scene_panel_native", "rmlui/scene_tree.rml") {
        listener_.owner = this;
        last_history_generation_ = std::numeric_limits<uint64_t>::max();
        last_log_generation_ = std::numeric_limits<uint64_t>::max();
    }

    void NativeScenePanel::EventListener::ProcessEvent(Rml::Event& event) {
        if (owner)
            owner->handleEvent(event);
    }

    void NativeScenePanel::preload(const PanelDrawContext& ctx) {
        if (!ensureInitialized())
            return;
        syncPanel(ctx);
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

        if (last_prepare_frame_ != ctx.frame_serial)
            syncPanel(ctx);

        host_.drawDirect(x, y, w, h);
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
        tree_el_ = dynamic_cast<SceneGraphElement*>(document_->GetElementById("tree-container"));
        scene_tab_el_ = document_->GetElementById("scene-tab");
        history_tab_el_ = document_->GetElementById("history-tab");
        logging_tab_el_ = document_->GetElementById("logging-tab");
        chip_row_el_ = document_->GetElementById("scene-chip-row");
        summary_model_chip_el_ = document_->GetElementById("summary-model-chip");
        summary_node_chip_el_ = document_->GetElementById("summary-node-chip");
        summary_selection_chip_el_ = document_->GetElementById("summary-selection-chip");
        summary_filter_chip_el_ = document_->GetElementById("summary-filter-chip");
        scene_view_el_ = document_->GetElementById("scene-view");
        search_container_el_ = document_->GetElementById("search-container");
        filter_input_el_ = document_->GetElementById("filter-input");
        filter_clear_el_ = document_->GetElementById("filter-clear");
        empty_state_el_ = document_->GetElementById("empty-state");
        empty_primary_el_ = document_->GetElementById("empty-primary");
        empty_secondary_el_ = document_->GetElementById("empty-secondary");
        history_container_el_ = document_->GetElementById("history-container");
        history_summary_label_el_ = document_->GetElementById("history-summary-label");
        history_summary_value_el_ = document_->GetElementById("history-summary-value");
        history_transaction_el_ = document_->GetElementById("history-transaction");
        history_undo_btn_el_ = document_->GetElementById("history-undo-btn");
        history_redo_btn_el_ = document_->GetElementById("history-redo-btn");
        history_clear_btn_el_ = document_->GetElementById("history-clear-btn");
        history_note_el_ = document_->GetElementById("history-note");
        history_undo_title_el_ = document_->GetElementById("history-undo-title");
        history_redo_title_el_ = document_->GetElementById("history-redo-title");
        history_undo_list_el_ = document_->GetElementById("history-undo-list");
        history_redo_list_el_ = document_->GetElementById("history-redo-list");
        history_empty_undo_el_ = document_->GetElementById("history-empty-undo");
        history_empty_redo_el_ = document_->GetElementById("history-empty-redo");
        logging_container_el_ = document_->GetElementById("logging-container");
        logging_summary_label_el_ = document_->GetElementById("logging-summary-label");
        logging_summary_value_el_ = document_->GetElementById("logging-summary-value");
        logging_level_label_el_ = document_->GetElementById("logging-level-label");
        logging_level_select_el_ =
            dynamic_cast<Rml::ElementFormControlSelect*>(document_->GetElementById("logging-level-select"));
        logging_export_btn_el_ = document_->GetElementById("logging-export-btn");
        logging_copy_btn_el_ = document_->GetElementById("logging-copy-btn");
        logging_feedback_el_ = document_->GetElementById("logging-feedback");
        logging_note_el_ = document_->GetElementById("logging-note");
        logging_scroll_el_ = document_->GetElementById("logging-scroll");
        logging_list_el_ = document_->GetElementById("logging-list");
        logging_empty_el_ = document_->GetElementById("logging-empty");

        if (!tree_el_ || !scene_tab_el_ || !history_tab_el_ || !logging_tab_el_ || !chip_row_el_ ||
            !summary_model_chip_el_ || !summary_node_chip_el_ || !summary_selection_chip_el_ ||
            !summary_filter_chip_el_ || !scene_view_el_ || !search_container_el_ ||
            !filter_input_el_ || !filter_clear_el_ || !empty_state_el_ || !empty_primary_el_ ||
            !empty_secondary_el_ || !history_container_el_ || !history_summary_label_el_ ||
            !history_summary_value_el_ || !history_transaction_el_ || !history_undo_btn_el_ ||
            !history_redo_btn_el_ || !history_clear_btn_el_ || !history_note_el_ ||
            !history_undo_title_el_ || !history_redo_title_el_ || !history_undo_list_el_ ||
            !history_redo_list_el_ || !history_empty_undo_el_ || !history_empty_redo_el_ ||
            !logging_container_el_ || !logging_summary_label_el_ || !logging_summary_value_el_ ||
            !logging_level_label_el_ || !logging_level_select_el_ || !logging_export_btn_el_ ||
            !logging_copy_btn_el_ || !logging_feedback_el_ || !logging_note_el_ ||
            !logging_scroll_el_ || !logging_list_el_ || !logging_empty_el_) {
            LOG_ERROR("NativeScenePanel: missing required DOM elements");
            return;
        }

        scene_tab_el_->AddEventListener(Rml::EventId::Click, &listener_);
        history_tab_el_->AddEventListener(Rml::EventId::Click, &listener_);
        logging_tab_el_->AddEventListener(Rml::EventId::Click, &listener_);
        filter_clear_el_->AddEventListener(Rml::EventId::Click, &listener_);
        history_undo_btn_el_->AddEventListener(Rml::EventId::Click, &listener_);
        history_redo_btn_el_->AddEventListener(Rml::EventId::Click, &listener_);
        history_clear_btn_el_->AddEventListener(Rml::EventId::Click, &listener_);
        history_undo_list_el_->AddEventListener(Rml::EventId::Click, &listener_);
        history_redo_list_el_->AddEventListener(Rml::EventId::Click, &listener_);
        logging_level_select_el_->AddEventListener(Rml::EventId::Change, &listener_);
        logging_export_btn_el_->AddEventListener(Rml::EventId::Click, &listener_);
        logging_copy_btn_el_->AddEventListener(Rml::EventId::Click, &listener_);
        filter_input_revert_.bind(filter_input_el_, [this](Rml::Element&) {
            applyFilterInputValue();
        });
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

        changed |= syncLocale();
        changed |= syncSceneState(ctx);
        changed |= syncHistoryState();
        changed |= syncLoggingState();
        changed |= syncTabState();
        changed |= syncSummaryChips();
        changed |= syncSceneVisibility();

        if (changed)
            host_.markContentDirty();
    }

    bool NativeScenePanel::syncSceneState(const PanelDrawContext& ctx) {
        if (!tree_el_)
            return false;

        const std::string filter_text =
            filter_input_el_ ? filter_input_el_->GetAttribute<Rml::String>("value", "") : "";
        tree_el_->setFilterText(filter_text);
        return tree_el_->syncFromScene(ctx);
    }

    bool NativeScenePanel::syncHistoryState() {
        auto& history = op::undoHistory();
        const uint64_t generation = history.generation();
        if (generation == last_history_generation_)
            return false;

        last_history_generation_ = generation;

        const auto undo_items = history.undoItems();
        const auto redo_items = history.redoItems();
        const bool has_undo = !undo_items.empty();
        const bool has_redo = !redo_items.empty();
        const bool can_clear = has_undo || has_redo || history.hasActiveTransaction();

        bool changed = false;
        changed |= setCachedText(history_summary_value_el_, formatHistorySummary(history));
        changed |= setCachedText(history_transaction_el_, formatHistoryTransaction(history));
        changed |= setCachedProperty(history_transaction_el_, "display",
                                     history.hasActiveTransaction() ? "block" : "none");
        changed |= setCachedText(history_undo_btn_el_,
                                 has_undo ? std::format("Undo: {}", undo_items.front().metadata.label)
                                          : std::string("Undo"));
        changed |= setCachedText(history_redo_btn_el_,
                                 has_redo ? std::format("Redo: {}", redo_items.front().metadata.label)
                                          : std::string("Redo"));
        changed |= setCachedDisabled(history_undo_btn_el_, !has_undo);
        changed |= setCachedDisabled(history_redo_btn_el_, !has_redo);
        changed |= setCachedDisabled(history_clear_btn_el_, !can_clear);
        changed |= setCachedInnerRml(history_undo_list_el_, historyRowsHtml(undo_items, "undo"));
        changed |= setCachedInnerRml(history_redo_list_el_, historyRowsHtml(redo_items, "redo"));
        changed |= setCachedProperty(history_empty_undo_el_, "display", has_undo ? "none" : "block");
        changed |= setCachedProperty(history_empty_redo_el_, "display", has_redo ? "none" : "block");
        changed |= setCachedText(history_empty_undo_el_,
                                 has_undo || has_redo ? "No entries in this stack"
                                                      : "Nothing recorded yet");
        changed |= setCachedText(history_empty_redo_el_,
                                 has_undo || has_redo ? "No entries in this stack"
                                                      : "Nothing recorded yet");
        return changed;
    }

    bool NativeScenePanel::syncLoggingState() {
        if (active_tab_ != Tab::Logging && !logging_feedback_dirty_)
            return false;

        auto& logger = core::Logger::get();
        const uint64_t generation = logger.buffered_log_generation();
        const core::LogLevel level = logger.level();
        if (generation == last_log_generation_ &&
            level == last_log_level_ &&
            !logging_feedback_dirty_) {
            return false;
        }

        last_log_generation_ = generation;
        last_log_level_ = level;
        logging_feedback_dirty_ = false;

        const auto entries = logger.buffered_logs();
        const bool has_entries = !entries.empty();
        const size_t displayed_entry_count = std::min(entries.size(), MAX_RENDERED_LOG_ENTRIES);
        const int desired_selection = logLevelSelectionIndex(level);

        bool changed = false;
        changed |= setCachedText(logging_summary_value_el_,
                                 formatLoggingSummary(entries.size(), displayed_entry_count, level));
        changed |= setCachedInnerRml(logging_list_el_, loggingRowsHtml(entries));
        changed |= setCachedProperty(logging_empty_el_, "display", has_entries ? "none" : "block");
        changed |= setCachedProperty(logging_list_el_, "display", has_entries ? "flex" : "none");
        changed |= setCachedText(logging_feedback_el_, logging_feedback_text_);
        changed |= setCachedProperty(logging_feedback_el_, "display",
                                     logging_feedback_text_.empty() ? "none" : "block");
        changed |= setCachedClass(logging_feedback_el_, "status-info",
                                  logging_feedback_tone_ == FeedbackTone::Info);
        changed |= setCachedClass(logging_feedback_el_, "status-success",
                                  logging_feedback_tone_ == FeedbackTone::Success);
        changed |= setCachedClass(logging_feedback_el_, "status-error",
                                  logging_feedback_tone_ == FeedbackTone::Error);

        if (logging_level_select_el_ &&
            logging_level_select_el_->GetSelection() != desired_selection) {
            logging_level_select_el_->SetSelection(desired_selection);
            changed = true;
        }

        return changed;
    }

    bool NativeScenePanel::syncLocale() {
        const std::string language = lfs::event::LocalizationManager::getInstance().getCurrentLanguage();
        if (language == last_language_)
            return false;

        last_language_ = language;

        bool changed = false;
        changed |= setCachedText(scene_tab_el_, tr("window.scene"));
        changed |= setCachedText(history_tab_el_, "History");
        changed |= setCachedText(logging_tab_el_, "Logging");
        changed |= setCachedAttribute(filter_input_el_, "placeholder", tr("scene.search"));
        changed |= setCachedText(empty_primary_el_, tr("scene.no_data_loaded"));
        changed |= setCachedText(empty_secondary_el_, tr("scene.use_file_menu"));
        changed |= setCachedText(history_summary_label_el_, "Shared History");
        changed |= setCachedText(history_note_el_,
                                 "Newest block is at the top. Click a block to jump there. New projects start clean.");
        changed |= setCachedText(history_undo_title_el_, "Undo Stack");
        changed |= setCachedText(history_redo_title_el_, "Redo Stack");
        changed |= setCachedText(history_clear_btn_el_, tr(lichtfeld::Strings::DebugInfo::CLEAR_HISTORY));
        changed |= setCachedText(logging_summary_label_el_, "CLI Log Mirror");
        changed |= setCachedText(logging_level_label_el_, "Log Level");
        changed |= setCachedText(logging_export_btn_el_, "Export .txt");
        changed |= setCachedText(logging_copy_btn_el_, "Copy");
        changed |= setCachedText(logging_note_el_,
                                 std::format("Mirrors console output. The panel shows the latest {} entries to stay responsive; copy and export use the full buffered log history.",
                                             MAX_RENDERED_LOG_ENTRIES));
        changed |= setCachedText(logging_empty_el_, "No logs captured yet");
        last_history_generation_ = std::numeric_limits<uint64_t>::max();
        last_log_generation_ = std::numeric_limits<uint64_t>::max();
        return changed;
    }

    bool NativeScenePanel::syncTabState() {
        bool changed = false;
        changed |= setCachedClass(scene_tab_el_, "active", active_tab_ == Tab::Scene);
        changed |= setCachedClass(history_tab_el_, "active", active_tab_ == Tab::History);
        changed |= setCachedClass(logging_tab_el_, "active", active_tab_ == Tab::Logging);
        changed |= setCachedProperty(scene_view_el_, "display",
                                     active_tab_ == Tab::Scene ? "flex" : "none");
        changed |= setCachedProperty(history_container_el_, "display",
                                     active_tab_ == Tab::History ? "flex" : "none");
        changed |= setCachedProperty(logging_container_el_, "display",
                                     active_tab_ == Tab::Logging ? "flex" : "none");
        return changed;
    }

    bool NativeScenePanel::syncSummaryChips() {
        if (!tree_el_)
            return false;

        bool changed = false;
        changed |= setCachedText(summary_model_chip_el_,
                                 pluralize(tree_el_->rootCount(), "model"));
        changed |= setCachedText(summary_node_chip_el_,
                                 pluralize(tree_el_->nodeCount(), "node"));
        changed |= setCachedText(summary_selection_chip_el_,
                                 pluralize(tree_el_->selectedCount(),
                                           "selected item", "selected items"));

        const bool show_filter = !tree_el_->filterText().empty();
        changed |= setCachedText(summary_filter_chip_el_,
                                 show_filter
                                     ? std::format("Filter: \"{}\"", tree_el_->filterText())
                                     : std::string{});
        changed |= setCachedProperty(summary_filter_chip_el_, "display",
                                     show_filter ? "inline-block" : "none");
        changed |= setCachedProperty(chip_row_el_, "display",
                                     active_tab_ == Tab::Scene && tree_el_->hasNodes()
                                         ? "flex"
                                         : "none");
        return changed;
    }

    bool NativeScenePanel::syncSceneVisibility() {
        if (!tree_el_)
            return false;

        const bool show_tree = tree_el_->hasNodes();
        const bool show_scene = active_tab_ == Tab::Scene;
        const bool show_filter_clear = !tree_el_->filterText().empty();

        bool changed = false;
        changed |= setCachedProperty(empty_state_el_, "display",
                                     show_scene && !show_tree ? "block" : "none");
        changed |= setCachedProperty(tree_el_, "display",
                                     show_scene && show_tree ? "block" : "none");
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

        if (type == "change") {
            const Rml::String current_id = current ? current->GetId() : "";
            if (current_id == "logging-level-select") {
                applyLogLevelSelection();
                event.StopPropagation();
                return true;
            }
            return false;
        }

        if (type != "click" || !target)
            return false;

        const Rml::String id = target->GetId();
        if (id == "scene-tab") {
            setTab(Tab::Scene);
            event.StopPropagation();
            return true;
        }
        if (id == "history-tab") {
            setTab(Tab::History);
            event.StopPropagation();
            return true;
        }
        if (id == "logging-tab") {
            setTab(Tab::Logging);
            event.StopPropagation();
            return true;
        }
        if (id == "filter-clear") {
            if (filter_input_el_)
                filter_input_el_->SetAttribute("value", "");
            applyFilterInputValue();
            event.StopPropagation();
            return true;
        }
        if (id == "history-undo-btn") {
            if (op::undoHistory().canUndo())
                op::undoHistory().undo();
            last_history_generation_ = std::numeric_limits<uint64_t>::max();
            event.StopPropagation();
            return true;
        }
        if (id == "history-redo-btn") {
            if (op::undoHistory().canRedo())
                op::undoHistory().redo();
            last_history_generation_ = std::numeric_limits<uint64_t>::max();
            event.StopPropagation();
            return true;
        }
        if (id == "history-clear-btn") {
            op::undoHistory().clear();
            last_history_generation_ = std::numeric_limits<uint64_t>::max();
            event.StopPropagation();
            return true;
        }
        if (id == "logging-export-btn") {
            exportBufferedLogsToTextFile();
            event.StopPropagation();
            return true;
        }
        if (id == "logging-copy-btn") {
            copyBufferedLogsToClipboard();
            event.StopPropagation();
            return true;
        }

        if (auto* action_target = findHistoryActionTarget(target)) {
            const std::string kind = action_target->GetAttribute<Rml::String>("data-kind", "");
            const int steps = action_target->GetAttribute<int>("data-steps", 0);
            if (steps > 0) {
                if (kind == "undo")
                    op::undoHistory().undoMultiple(static_cast<size_t>(steps));
                else if (kind == "redo")
                    op::undoHistory().redoMultiple(static_cast<size_t>(steps));
                last_history_generation_ = std::numeric_limits<uint64_t>::max();
                event.StopPropagation();
                return true;
            }
        }

        return false;
    }

    void NativeScenePanel::applyFilterInputValue() {
        if (tree_el_)
            tree_el_->setFilterText(filter_input_el_ ? filter_input_el_->GetAttribute<Rml::String>("value", "") : "");
        syncSummaryChips();
        syncSceneVisibility();
        host_.markContentDirty();
    }

    void NativeScenePanel::applyLogLevelSelection() {
        if (!logging_level_select_el_)
            return;

        const core::LogLevel selected_level =
            logLevelFromSelection(logging_level_select_el_->GetSelection());
        core::Logger::get().set_level(selected_level);
        setLoggingFeedback(std::format("Log level set to {}", logLevelLabel(selected_level)),
                           FeedbackTone::Success);
    }

    void NativeScenePanel::copyBufferedLogsToClipboard() {
        auto& logger = core::Logger::get();
        const size_t entry_count = logger.buffered_log_count();
        if (entry_count == 0) {
            setLoggingFeedback("No buffered logs to copy.", FeedbackTone::Info);
            return;
        }

        const std::string log_text = logger.buffered_logs_as_text();
        SDL_SetClipboardText(log_text.c_str());
        setLoggingFeedback(std::format("Copied {} log {} to the clipboard.",
                                       entry_count,
                                       entry_count == 1 ? "entry" : "entries"),
                           FeedbackTone::Success);
    }

    void NativeScenePanel::exportBufferedLogsToTextFile() {
        auto& logger = core::Logger::get();
        const size_t entry_count = logger.buffered_log_count();
        if (entry_count == 0) {
            setLoggingFeedback("No buffered logs to export.", FeedbackTone::Info);
            return;
        }

        const auto path = SaveTextFileDialog("lichtfeld_logs");
        if (path.empty())
            return;

        const std::string log_text = logger.buffered_logs_as_text();
        std::ofstream file;
        if (!core::open_file_for_write(path, file)) {
            setLoggingFeedback("Failed to open the selected log file for writing.",
                               FeedbackTone::Error);
            return;
        }

        file << log_text;
        if (!file.good()) {
            setLoggingFeedback("Failed to write the log export.", FeedbackTone::Error);
            return;
        }

        const std::string exported_count =
            entry_count == 1 ? std::string("1 log entry")
                             : std::format("{} log entries", entry_count);
        setLoggingFeedback(std::format("Exported {} to {}.",
                                       exported_count,
                                       path.filename().string()),
                           FeedbackTone::Success);
    }

    void NativeScenePanel::setLoggingFeedback(std::string message, const FeedbackTone tone) {
        logging_feedback_text_ = std::move(message);
        logging_feedback_tone_ = tone;
        logging_feedback_dirty_ = true;
        host_.markContentDirty();
    }

    void NativeScenePanel::setTab(const Tab tab) {
        if (active_tab_ == tab)
            return;
        active_tab_ = tab;
        host_.markContentDirty();
    }

} // namespace lfs::vis::gui
