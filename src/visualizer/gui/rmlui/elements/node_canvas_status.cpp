/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "node_canvas_element.hpp"

#include "core/event_bridge/localization_manager.hpp"
#include "core/nodes/tree.hpp"
#include "gui/rmlui/elements/node_canvas_dom.hpp"
#include "gui/rmlui/elements/node_canvas_widgets.hpp"
#include "gui/utils/native_file_dialog.hpp"
#include "scene/scene_manager.hpp"
#include "visualizer/nodes/modifier_manager.hpp"

#include <RmlUi/Core/ElementText.h>
#include <algorithm>
#include <cmath>
#include <format>
#include <fstream>
#include <ranges>

namespace lfs::vis::gui {
    namespace {
        std::string duration(const double milliseconds) {
            return milliseconds >= 1000.0 ? std::format("{:.2f} s", milliseconds / 1000.0)
                                          : std::format("{:.0f} ms", milliseconds);
        }

        std::string compactCount(const std::size_t count) {
            if (count >= 1'000'000)
                return std::format("{:.2f}M", static_cast<double>(count) / 1'000'000);
            if (count >= 10'000)
                return std::format("{:.1f}K", static_cast<double>(count) / 1000);
            return std::to_string(count);
        }

        std::string errorSummary(const std::string& message) {
            const auto end = message.find_last_not_of("\r\n");
            if (end == std::string::npos)
                return message;
            const auto begin = message.find_last_of('\n', end);
            return message.substr(begin == std::string::npos ? 0 : begin + 1,
                                  end - (begin == std::string::npos ? 0 : begin + 1) + 1);
        }
    } // namespace

    std::string NodeCanvasElement::nodeStatus(const std::string_view name) const {
        if (busy_node_ == name)
            return LOC("node_editor.running");
        if (queued_nodes_.contains(std::string(name)))
            return LOC("node_editor.queued");
        const auto host = activeHost();
        const auto* result = host && manager_ ? manager_->lastResult(*host) : nullptr;
        const lfs::nodes::NodeEvaluation* status = nullptr;
        if (const auto found = progress_nodes_.find(std::string(name)); found != progress_nodes_.end())
            status = &found->second;
        else if (result) {
            const auto found = result->nodes.find(active_modifier_uuid_ + "/" + std::string(name));
            if (found != result->nodes.end())
                status = &found->second;
        }
        if (!status)
            return "—";
        std::string text;
        const auto append = [&](const std::string& value) {
            if (!text.empty())
                text += " · ";
            text += value;
        };
        if (status->element_count)
            append(compactCount(*status->element_count));
        if (status->selected_share) {
            const auto percentage = static_cast<int>(std::round(*status->selected_share * 100.0));
            append(std::vformat(LOC("node_editor.selected_share"), std::make_format_args(percentage)));
        }
        if (status->cached)
            append(LOC("node_editor.cached"));
        append(duration(status->time_ms));
        return text;
    }

    std::string NodeCanvasElement::statusText() const {
        if (!file_error_.empty())
            return file_error_;
        if (!manager_ || !editableMode())
            return {};
        const auto host = activeHost();
        if (!host)
            return {};
        const auto progress = manager_->progress();
        if (progress.busy && progress.host == *host &&
            std::chrono::steady_clock::now() - progress.started_at >= std::chrono::milliseconds(80)) {
            if (progress.label.empty())
                return LOC("node_editor.updating");
            const auto current = std::min(progress.completed + 1, progress.total);
            return std::vformat(LOC("node_editor.updating_node"),
                                std::make_format_args(current, progress.total, progress.label));
        }
        const auto* result = manager_->lastResult(*host);
        if (!result)
            return "—";
        if (!result->ok) {
            const auto count = result->errors.size();
            return std::vformat(LOC("node_editor.error_count"), std::make_format_args(count));
        }
        if (result->unchanged)
            return LOC("node_editor.unchanged");
        std::size_t count = 0;
        std::string kind;
        if (result->geometry.splats) {
            count = result->geometry.splats->means.shape()[0];
            kind = LOC("node_editor.splats");
        } else if (result->geometry.points) {
            count = result->geometry.points->positions.shape()[0];
            kind = LOC("node_editor.points");
        } else if (result->geometry.mesh && result->geometry.mesh->mesh) {
            count = result->geometry.mesh->mesh->vertex_count();
            kind = LOC("node_editor.vertices");
        }
        const auto elapsed = duration(result->total_time_ms);
        return std::vformat(LOC("node_editor.result"), std::make_format_args(count, kind, elapsed));
    }

    bool NodeCanvasElement::modifiersVisible() const {
        const auto host = activeHost();
        const auto* stack = host && manager_ ? manager_->stack(*host) : nullptr;
        return stack && std::ranges::any_of(stack->modifiers, &Modifier::show_viewport);
    }

    void NodeCanvasElement::updateEvaluationDom() {
        const auto text = [](Rml::Element* element, const std::string& value) {
            if (!element)
                return;
            if (auto* child = dynamic_cast<Rml::ElementText*>(element->GetChild(0))) {
                if (child->GetText() != value)
                    child->SetText(value);
            } else if (!value.empty()) {
                node_widgets::patchMarkup(*element, node_widgets::escape(value));
            }
        };
        const auto host = activeHost();
        const auto* stack = host && manager_ ? manager_->stack(*host) : nullptr;
        if (stack && sidebar_element_) {
            Rml::ElementList rows;
            sidebar_element_->QuerySelectorAll(rows, ".modifier-row");
            for (auto* row : rows) {
                const auto modifier = std::ranges::find(stack->modifiers, row->GetAttribute<Rml::String>("data-modifier", ""), &Modifier::uuid);
                auto* indicator = row->QuerySelector(".modifier-status");
                if (modifier == stack->modifiers.end() || !indicator)
                    continue;
                const auto count = std::ranges::count_if(evaluation_errors_, [&](const auto& error) {
                    return error.first == modifier->name || error.first.starts_with(modifier->name + "/");
                });
                indicator->SetClass("failed", count != 0);
                const auto label = count ? std::vformat(LOC("node_editor.error_count"), std::make_format_args(count))
                                         : std::string(LOC("node_editor.status_ok"));
                const auto title = label + " · " + duration(evaluation_total_ms_);
                if (indicator->GetAttribute<Rml::String>("title", "") != title)
                    indicator->SetAttribute("title", title);
            }
        }
        for (const auto& visual : visuals_) {
            const auto element = node_elements_.find(visual.interaction.id);
            if (element == node_elements_.end())
                continue;
            element->second->SetClass("failed", !visual.error.empty());
            auto* footer = element->second->QuerySelector(".node-footer");
            const bool pending = busy_node_ == visual.interaction.id || queued_nodes_.contains(visual.interaction.id);
            text(footer, visual.error.empty() || pending ? nodeStatus(visual.interaction.id) : errorSummary(visual.error));
            const auto tooltip = visual.error.empty() ? nodeStatus(visual.interaction.id) : visual.error;
            if (footer && footer->GetAttribute<Rml::String>("title", "") != tooltip)
                footer->SetAttribute("title", tooltip);
            if (selected_nodes_.size() == 1 && selected_nodes_.contains(visual.interaction.id)) {
                const auto last_run = nodeStatus(visual.interaction.id);
                text(sidebar_element_->GetElementById("node-inspector-last-run"),
                     node_widgets::nonBreakingStatus(std::vformat(LOC("node_editor.last_run"), std::make_format_args(last_run))));
                auto* error = sidebar_element_->GetElementById("node-inspector-error");
                text(error, visual.error);
                if (error)
                    error->SetProperty("display", visual.error.empty() ? "none" : "block");
            }
        }
    }

    void NodeCanvasElement::updateProgress() {
        const auto progress = manager_ ? manager_->progress() : ModifierWorkerProgress{};
        const auto host = activeHost();
        const bool visible = progress.busy && host && progress.host == *host &&
                             progress.modifier == active_modifier_uuid_ &&
                             std::chrono::steady_clock::now() - progress.started_at >= std::chrono::milliseconds(80);
        const std::string next = visible
                                     ? progress.node
                                     : std::string{};
        const auto generation = visible ? progress.generation : 0;
        if (busy_node_ == next && progress_generation_ == generation &&
            progress_nodes_.size() == (visible ? progress.finished_nodes.size() : 0))
            return;
        busy_node_ = next;
        progress_generation_ = generation;
        queued_nodes_ = visible ? progress.pending_nodes : std::unordered_set<std::string>{};
        queued_nodes_.erase(busy_node_);
        progress_nodes_ = visible ? progress.finished_nodes : decltype(progress_nodes_){};
        for (const auto& [name, element] : node_elements_) {
            element->SetClass("evaluating", name == busy_node_);
            element->SetClass("queued", queued_nodes_.contains(name));
        }
        updateEvaluationDom();
    }

    void NodeCanvasElement::headerAction(const std::string_view action, const float screen_x,
                                         const float screen_y) {
        if (!editableMode())
            return;
        file_error_.clear();
        if (action == "add") {
            openAddMenu(screen_x - panel_screen_offset_.x, screen_y - panel_screen_offset_.y);
        } else if (action == "frame") {
            syncModel();
            frameAll();
        } else if (action == "arrange") {
            syncModel();
            arrange();
        } else if (action == "modifiers-visible") {
            const auto host = activeHost();
            if (auto* stack = host && manager_ ? manager_->stack(*host) : nullptr) {
                const nlohmann::json before = stack->modifiers;
                const bool visible = !modifiersVisible();
                for (auto& modifier : stack->modifiers)
                    modifier.show_viewport = visible;
                manager_->recordStackEdit(*host, before);
            }
        } else if (action == "open" || action == "save") {
            try {
                if (action == "open") {
                    const auto path = OpenJsonFileDialog();
                    if (path.empty())
                        return;
                    std::ifstream stream(path);
                    if (!stream)
                        throw std::runtime_error(LOC("node_editor.file_read_error"));
                    auto json = nlohmann::json::parse(stream);
                    // Import into the library without replacing an existing graph.
                    json["uuid"] = core::generate_uuid_v4().to_string();
                    manager_->loadTree(json);
                } else if (const auto* tree = activeTree()) {
                    const auto path = SaveJsonFileDialog(tree->name + ".json");
                    if (path.empty())
                        return;
                    std::ofstream stream(path);
                    stream << tree->to_json().dump(2) << '\n';
                    stream.close();
                    if (!stream)
                        throw std::runtime_error(LOC("node_editor.file_write_error"));
                }
            } catch (const std::exception& error) {
                // LFS-CENSUS-OK(empty-catch): file-dialog failures are reported in the editor header, preserving the current graph.
                file_error_ = error.what();
            }
        }
    }
} // namespace lfs::vis::gui
