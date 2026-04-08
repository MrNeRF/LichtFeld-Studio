/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rmlui/elements/scene_graph_element.hpp"

#include "core/event_bridge/localization_manager.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "gui/global_context_menu.hpp"
#include "gui/gui_manager.hpp"
#include "gui/panel_registry.hpp"
#include "gui/string_keys.hpp"
#include "gui/utils/native_file_dialog.hpp"
#include "io/exporter.hpp"
#include "scene/scene_manager.hpp"
#include "visualizer/core/parameter_manager.hpp"
#include "visualizer/core/services.hpp"
#include "visualizer/gui_capabilities.hpp"

#include <RmlUi/Core.h>
#include <RmlUi/Core/Event.h>
#include <RmlUi/Core/Input.h>
#include <SDL3/SDL_keyboard.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cmath>
#include <format>
#include <functional>
#include <optional>
#include <ranges>
#include <sstream>

namespace lfs::vis::gui {

    using namespace lfs::core::events;
    namespace string_keys = lichtfeld::Strings;

    namespace {

        constexpr std::string_view kContextActionPrefix = "scene_panel:";
        constexpr std::array kTypeClasses{
            "splat",
            "group",
            "dataset",
            "camera",
            "camera_group",
            "cropbox",
            "ellipsoid",
            "pointcloud",
            "mesh",
            "keyframe_group",
            "keyframe",
        };

        [[nodiscard]] std::string tr(const char* key) {
            const std::string value = LOC(key);
            return value.empty() ? std::string(key) : value;
        }

        [[nodiscard]] std::string formatLocalizedCount(std::string pattern, const size_t count) {
            if (const size_t pos = pattern.find("{}"); pos != std::string::npos) {
                pattern.replace(pos, 2, std::to_string(count));
                return pattern;
            }
            return std::format("{} ({})", pattern, count);
        }

        [[nodiscard]] std::string lowerCopy(std::string value) {
            std::ranges::transform(value, value.begin(), [](const unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
            return value;
        }

        [[nodiscard]] std::string encode(const std::string& value) {
            return Rml::StringUtilities::EncodeRml(value);
        }

        [[nodiscard]] std::string formatDp(const int value) {
            return std::to_string(value) + "dp";
        }

        [[nodiscard]] std::string cacheAttrName(std::string_view kind, std::string_view name) {
            return std::format("data-lfs-tree-{}-{}", kind, name);
        }

        bool setCachedInnerRml(Rml::Element* el, const std::string& value) {
            if (!el)
                return false;

            const std::string attr_name = cacheAttrName("rml", "value");
            if (el->GetAttribute<Rml::String>(attr_name.c_str(), "") == value)
                return false;

            el->SetInnerRML(value);
            el->SetAttribute(attr_name, value);
            return true;
        }

        bool setCachedProperty(Rml::Element* el, std::string_view name, const std::string& value) {
            if (!el)
                return false;

            const std::string attr_name = cacheAttrName("prop", name);
            if (el->GetAttribute<Rml::String>(attr_name.c_str(), "") == value)
                return false;

            el->SetProperty(std::string(name), value);
            el->SetAttribute(attr_name, value);
            return true;
        }

        bool setCachedAttribute(Rml::Element* el, std::string_view name, const std::string& value) {
            if (!el)
                return false;

            const std::string attr_name = cacheAttrName("attr", name);
            if (el->GetAttribute<Rml::String>(attr_name.c_str(), "") == value)
                return false;

            el->SetAttribute(std::string(name), value);
            el->SetAttribute(attr_name, value);
            return true;
        }

        bool setCachedClass(Rml::Element* el, std::string_view name, const bool enabled) {
            if (!el)
                return false;

            const std::string attr_name = cacheAttrName("class", name);
            const char* const encoded = enabled ? "1" : "0";
            if (el->GetAttribute<Rml::String>(attr_name.c_str(), "") == encoded)
                return false;

            el->SetClass(std::string(name), enabled);
            el->SetAttribute(attr_name, encoded);
            return true;
        }

        bool setCachedTypeClass(Rml::Element* el, std::string_view next_class) {
            if (!el)
                return false;

            constexpr std::string_view kAttr = "data-lfs-tree-type-class";
            const std::string current = el->GetAttribute<Rml::String>(kAttr.data(), "");
            if (current == next_class)
                return false;

            if (!current.empty())
                el->SetClass(current, false);
            if (!next_class.empty())
                el->SetClass(std::string(next_class), true);
            el->SetAttribute(kAttr.data(), std::string(next_class));
            return true;
        }

        [[nodiscard]] const char* typeClass(const core::NodeType type) {
            switch (type) {
            case core::NodeType::SPLAT: return "splat";
            case core::NodeType::GROUP: return "group";
            case core::NodeType::DATASET: return "dataset";
            case core::NodeType::CAMERA: return "camera";
            case core::NodeType::CAMERA_GROUP: return "camera_group";
            case core::NodeType::CROPBOX: return "cropbox";
            case core::NodeType::ELLIPSOID: return "ellipsoid";
            case core::NodeType::POINTCLOUD: return "pointcloud";
            case core::NodeType::MESH: return "mesh";
            case core::NodeType::KEYFRAME_GROUP: return "keyframe_group";
            case core::NodeType::KEYFRAME: return "keyframe";
            case core::NodeType::IMAGE_GROUP: return "group";
            case core::NodeType::IMAGE: return "group";
            }
            return "";
        }

        [[nodiscard]] const char* typeIconPath(const core::NodeType type) {
            switch (type) {
            case core::NodeType::SPLAT: return "../icon/scene/splat.png";
            case core::NodeType::POINTCLOUD: return "../icon/scene/pointcloud.png";
            case core::NodeType::GROUP: return "../icon/scene/group.png";
            case core::NodeType::DATASET: return "../icon/scene/dataset.png";
            case core::NodeType::CAMERA:
            case core::NodeType::CAMERA_GROUP: return "../icon/scene/camera.png";
            case core::NodeType::CROPBOX: return "../icon/scene/cropbox.png";
            case core::NodeType::ELLIPSOID: return "../icon/scene/ellipsoid.png";
            case core::NodeType::MESH: return "../icon/scene/mesh.png";
            case core::NodeType::KEYFRAME_GROUP:
            case core::NodeType::KEYFRAME:
            case core::NodeType::IMAGE_GROUP:
            case core::NodeType::IMAGE: return "";
            }
            return "";
        }

        [[nodiscard]] const char* unicodeIcon(const core::NodeType type) {
            switch (type) {
            case core::NodeType::KEYFRAME_GROUP:
            case core::NodeType::KEYFRAME: return "\u25c6";
            case core::NodeType::IMAGE_GROUP:
            case core::NodeType::IMAGE: return "\u25a3";
            default: return "";
            }
        }

        [[nodiscard]] bool isDeletable(const core::NodeType type, const bool parent_is_dataset) {
            return type != core::NodeType::CAMERA &&
                   type != core::NodeType::CAMERA_GROUP &&
                   type != core::NodeType::KEYFRAME &&
                   type != core::NodeType::KEYFRAME_GROUP &&
                   !parent_is_dataset;
        }

        [[nodiscard]] bool canDrag(const core::NodeType type, const bool parent_is_dataset) {
            if (parent_is_dataset)
                return false;
            switch (type) {
            case core::NodeType::SPLAT:
            case core::NodeType::GROUP:
            case core::NodeType::POINTCLOUD:
            case core::NodeType::MESH:
            case core::NodeType::CROPBOX:
            case core::NodeType::ELLIPSOID:
                return true;
            default:
                return false;
            }
        }

        [[nodiscard]] bool ctrlDown() {
            const SDL_Keymod mod = SDL_GetModState();
            return (mod & SDL_KMOD_CTRL) != 0 || (mod & SDL_KMOD_GUI) != 0;
        }

        [[nodiscard]] bool shiftDown() {
            return (SDL_GetModState() & SDL_KMOD_SHIFT) != 0;
        }

        [[nodiscard]] std::vector<std::string> splitAction(std::string_view action) {
            std::vector<std::string> parts;
            std::stringstream stream{std::string(action)};
            std::string segment;
            while (std::getline(stream, segment, ':'))
                parts.push_back(segment);
            return parts;
        }

        [[nodiscard]] bool parseNodeId(std::string_view value, core::NodeId& node_id) {
            core::NodeId parsed = core::NULL_NODE;
            const char* const begin = value.data();
            const char* const end = begin + value.size();
            const auto [ptr, ec] = std::from_chars(begin, end, parsed);
            if (ec != std::errc{} || ptr != end)
                return false;
            node_id = parsed;
            return true;
        }

        void saveNodeToDisk(const core::Scene& scene, const core::NodeId node_id) {
            const auto* node = scene.getNodeById(node_id);
            if (!node)
                return;

            std::string default_name = node->name.empty() ? "scene_node" : node->name;
            const auto path = SavePlyFileDialog(default_name);
            if (path.empty())
                return;

            const io::PlySaveOptions options{
                .output_path = path,
                .binary = true,
                .async = false,
                .extra_attributes = {},
            };

            io::Result<void> result = std::unexpected(
                io::Error{io::ErrorCode::INTERNAL_ERROR, "uninitialized"});

            switch (node->type) {
            case core::NodeType::POINTCLOUD:
                if (!node->point_cloud || node->point_cloud->size() == 0)
                    return;
                result = io::save_ply(*node->point_cloud, options);
                break;
            case core::NodeType::SPLAT:
                if (!node->model || node->model->size() == 0)
                    return;
                result = io::save_ply(*node->model, options);
                break;
            default:
                return;
            }

            if (!result) {
                LOG_ERROR("Failed to save '{}' to {}: {}",
                          node->name,
                          lfs::core::path_to_utf8(path),
                          result.error().message);
            }
        }

    } // namespace

    SceneGraphElement::SceneGraphElement(const Rml::String& tag) : Rml::Element(tag) {}

    bool SceneGraphElement::ownsContextMenuAction(const std::string_view action) {
        return action.starts_with(kContextActionPrefix);
    }

    void SceneGraphElement::setPanelScreenOffset(const float x, const float y) {
        panel_screen_x_ = x;
        panel_screen_y_ = y;
    }

    void SceneGraphElement::setFilterText(const std::string_view text) {
        const std::string next(text);
        if (next == filter_text_)
            return;
        captureRenameBuffer();
        filter_text_ = next;
        tree_rebuild_needed_ = true;
        markStateDirty();
    }

    bool SceneGraphElement::GetIntrinsicDimensions(Rml::Vector2f& dimensions, float& ratio) {
        dimensions = {1.0f, 1.0f};
        ratio = 0.0f;
        return true;
    }

    void SceneGraphElement::OnResize() { dom_dirty_ = true; }

    void SceneGraphElement::OnRender() {
        ensureDom();
        if (tree_rebuild_needed_) {
            if (auto* scene_manager = services().sceneOrNull()) {
                const core::Scene& scene = scene_manager->getScene();
                syncSelectionFromScene(scene, scene_manager);

                if (scene.hasNodes()) {
                    captureRenameBuffer();
                    rebuildFlatRows(scene);
                    markStateDirty();
                } else if (scene_has_nodes_ || !node_snapshots_.empty()) {
                    clear();
                }
            }
        }
        syncVisibleRows(false);
    }

    void SceneGraphElement::RenameInputListener::ProcessEvent(Rml::Event& event) {
        if (!owner)
            return;
        auto* target = event.GetTargetElement();
        if (owner->rename_node_id_ == core::NULL_NODE || !owner->isTextInputTarget(target))
            return;

        const auto type = event.GetType();
        if (type == "escapecancel") {
            owner->cancelRename();
            event.StopPropagation();
        } else if (type == "keydown") {
            const auto key =
                static_cast<Rml::Input::KeyIdentifier>(event.GetParameter("key_identifier", 0));
            if (key == Rml::Input::KI_RETURN) {
                owner->confirmRename();
                event.StopPropagation();
            }
        }
    }

    void SceneGraphElement::ensureDom() {
        if (content_el_)
            return;

        auto* doc = GetOwnerDocument();
        if (!doc)
            return;

        rename_input_listener_.owner = this;
        AddEventListener("escapecancel", &rename_input_listener_);
        AddEventListener("keydown", &rename_input_listener_, true);

        auto content = doc->CreateElement("div");
        content->SetClass("scene-graph-content", true);
        content->SetProperty("position", "relative");
        content->SetProperty("width", "100%");
        content->SetProperty("height", "0dp");
        content_el_ = AppendChild(std::move(content));

        auto header = doc->CreateElement("div");
        header->SetClass("section-header", true);
        header->SetProperty("position", "absolute");
        header->SetProperty("left", "0px");
        header->SetProperty("right", "0px");
        header->SetProperty("top", "0px");
        header->SetProperty("height", std::format("{}dp", kHeaderHeightDp));
        header->SetAttribute("data-role", "models-header");
        header_el_ = content_el_->AppendChild(std::move(header));

        auto arrow = doc->CreateElement("span");
        arrow->SetClass("section-arrow", true);
        header_arrow_el_ = header_el_->AppendChild(std::move(arrow));

        auto label = doc->CreateElement("span");
        header_label_el_ = header_el_->AppendChild(std::move(label));

        dom_dirty_ = true;
    }

    void SceneGraphElement::ensureRowPool(const size_t count) {
        ensureDom();
        auto* doc = GetOwnerDocument();
        if (!content_el_ || !doc)
            return;

        while (row_slots_.size() < count) {
            RowSlot slot;
            auto row = doc->CreateElement("div");
            row->SetClass("tree-row", true);
            row->SetProperty("position", "absolute");
            row->SetProperty("left", "0px");
            row->SetProperty("right", "0px");
            row->SetProperty("display", "none");
            slot.root = content_el_->AppendChild(std::move(row));

            auto content = doc->CreateElement("span");
            content->SetClass("row-content", true);
            slot.content = slot.root->AppendChild(std::move(content));

            auto vis_icon = doc->CreateElement("img");
            vis_icon->SetClass("row-icon", true);
            vis_icon->SetAttribute("data-action", "toggle-vis");
            slot.vis_icon = slot.content->AppendChild(std::move(vis_icon));

            auto trash_icon = doc->CreateElement("img");
            trash_icon->SetClass("row-icon", true);
            trash_icon->SetClass("trash-icon", true);
            trash_icon->SetAttribute("src", "../icon/scene/trash.png");
            trash_icon->SetAttribute("data-action", "delete");
            slot.delete_icon = slot.content->AppendChild(std::move(trash_icon));

            auto type_icon = doc->CreateElement("img");
            type_icon->SetClass("row-icon", true);
            type_icon->SetClass("type-icon", true);
            slot.type_icon = slot.content->AppendChild(std::move(type_icon));

            auto unicode = doc->CreateElement("span");
            unicode->SetClass("node-icon", true);
            slot.unicode_icon = slot.content->AppendChild(std::move(unicode));

            auto mask_icon = doc->CreateElement("img");
            mask_icon->SetClass("row-icon", true);
            mask_icon->SetClass("mask-icon", true);
            mask_icon->SetAttribute("src", "../icon/scene/mask.png");
            slot.mask_icon = slot.content->AppendChild(std::move(mask_icon));

            auto expand = doc->CreateElement("span");
            expand->SetClass("expand-toggle", true);
            slot.expand_toggle = slot.content->AppendChild(std::move(expand));

            auto leaf = doc->CreateElement("span");
            leaf->SetClass("leaf-spacer", true);
            slot.leaf_spacer = slot.content->AppendChild(std::move(leaf));

            auto rename_input = doc->CreateElement("input");
            rename_input->SetAttribute("type", "text");
            rename_input->SetClass("rename-input", true);
            slot.rename_input = slot.content->AppendChild(std::move(rename_input));

            auto node_name = doc->CreateElement("span");
            node_name->SetClass("node-name", true);
            slot.node_name = slot.content->AppendChild(std::move(node_name));

            row_slots_.push_back(slot);
        }
    }

    void SceneGraphElement::clear() {
        node_snapshots_.clear();
        root_ids_.clear();
        flat_rows_.clear();
        flat_index_by_id_.clear();
        collapsed_ids_.clear();
        selected_ids_.clear();
        click_anchor_id_ = core::NULL_NODE;
        rename_node_id_ = core::NULL_NODE;
        rename_buffer_.clear();
        context_menu_node_id_ = core::NULL_NODE;
        drag_source_id_ = core::NULL_NODE;
        drop_target_id_ = core::NULL_NODE;
        scene_has_nodes_ = false;
        root_count_ = 0;
        row_top_dp_cache_.clear();
        last_selection_generation_ = std::numeric_limits<uint32_t>::max();
        last_visible_start_ = kUnsetVisibleRange;
        last_visible_end_ = kUnsetVisibleRange;
        tree_rebuild_needed_ = false;
        markStateDirty();
    }

    void SceneGraphElement::markStateDirty() {
        ++state_revision_;
        dom_dirty_ = true;
    }

    void SceneGraphElement::captureRenameBuffer() {
        if (rename_node_id_ == core::NULL_NODE)
            return;
        for (const RowSlot& slot : row_slots_) {
            if (!slot.visible || slot.bound_id != rename_node_id_ || !slot.rename_input)
                continue;
            rename_buffer_ = slot.rename_input->GetAttribute<Rml::String>("value", rename_buffer_);
            return;
        }
    }

    bool SceneGraphElement::syncSelectionFromScene(const core::Scene& scene,
                                                   SceneManager* scene_manager) {
        const uint32_t selection_generation =
            scene_manager ? scene_manager->selectionState().generation() : 0;
        if (selection_generation == last_selection_generation_)
            return false;

        std::unordered_set<core::NodeId> selected_ids;
        if (scene_manager) {
            for (const std::string& name : scene_manager->getSelectedNodeNames()) {
                const core::NodeId id = scene.getNodeIdByName(name);
                if (id != core::NULL_NODE)
                    selected_ids.insert(id);
            }
        }

        last_selection_generation_ = selection_generation;
        if (selected_ids == selected_ids_)
            return false;

        selected_ids_ = std::move(selected_ids);
        markStateDirty();
        return true;
    }

    void SceneGraphElement::captureSceneSnapshot(
        const core::Scene& scene,
        std::unordered_map<core::NodeId, NodeSnapshot>& snapshots,
        std::vector<core::NodeId>& root_ids) {
        const std::unordered_set<core::NodeId> previous_ids = [&] {
            std::unordered_set<core::NodeId> ids;
            ids.reserve(node_snapshots_.size());
            for (const auto& [id, _] : node_snapshots_)
                ids.insert(id);
            return ids;
        }();

        const auto nodes = scene.getNodes();
        const auto active_gaussian_counts = scene.getActiveGaussianCountsByNode();
        snapshots.reserve(nodes.size());
        root_ids.reserve(nodes.size());

        for (const core::SceneNode* node : nodes) {
            NodeSnapshot snapshot;
            snapshot.id = node->id;
            snapshot.parent_id = node->parent_id;
            snapshot.children = node->children;
            snapshot.type = node->type;
            snapshot.name = node->name;
            snapshot.visible = static_cast<bool>(node->visible);
            snapshot.has_children = !node->children.empty();
            snapshot.training_enabled = node->training_enabled;
            snapshot.has_mask = node->type == core::NodeType::CAMERA && !node->mask_path.empty();

            switch (node->type) {
            case core::NodeType::SPLAT:
                if (const auto it = active_gaussian_counts.find(node->id);
                    it != active_gaussian_counts.end() && it->second > 0) {
                    snapshot.label = std::format("{}  ({})", node->name, it->second);
                } else {
                    snapshot.label = node->name;
                }
                break;
            case core::NodeType::POINTCLOUD:
                snapshot.label = (node->point_cloud && node->point_cloud->size() > 0)
                                     ? std::format("{}  ({})", node->name, node->point_cloud->size())
                                     : node->name;
                break;
            case core::NodeType::MESH:
                snapshot.label = (node->mesh)
                                     ? std::format("{}  ({}V / {}F)", node->name,
                                                   node->mesh->vertex_count(),
                                                   node->mesh->face_count())
                                     : node->name;
                break;
            case core::NodeType::KEYFRAME:
                if (node->keyframe)
                    snapshot.label = std::format("Keyframe {}  ({:.2f}s)",
                                                 node->keyframe->keyframe_index + 1,
                                                 node->keyframe->time);
                else
                    snapshot.label = node->name;
                break;
            default:
                snapshot.label = node->name;
                break;
            }

            snapshots.emplace(snapshot.id, std::move(snapshot));
            if (node->parent_id == core::NULL_NODE)
                root_ids.push_back(node->id);
        }

        auto collapsed_ids = collapsed_ids_;
        collapsed_ids_.clear();
        for (const auto& [id, _] : snapshots) {
            if (collapsed_ids.contains(id))
                collapsed_ids_.insert(id);
        }

        for (auto& [id, snapshot] : snapshots) {
            const auto parent_it = snapshots.find(snapshot.parent_id);
            const bool parent_is_dataset =
                parent_it != snapshots.end() && parent_it->second.type == core::NodeType::DATASET;
            snapshot.draggable = canDrag(snapshot.type, parent_is_dataset);
            snapshot.deletable = isDeletable(snapshot.type, parent_is_dataset);
            if (!previous_ids.contains(id) &&
                snapshot.type == core::NodeType::CAMERA_GROUP &&
                static_cast<int>(snapshot.children.size()) >= kAutoCollapseCameraGroupThreshold) {
                collapsed_ids_.insert(id);
            }
        }
    }

    void SceneGraphElement::appendSnapshotRows(const core::NodeId node_id,
                                               const int depth,
                                               std::vector<FlatRow>& rows,
                                               const std::string& filter_text_lower) const {
        const auto it = node_snapshots_.find(node_id);
        if (it == node_snapshots_.end())
            return;

        const NodeSnapshot& snapshot = it->second;
        std::vector<FlatRow> child_rows;
        for (const core::NodeId child_id : snapshot.children)
            appendSnapshotRows(child_id, depth + 1, child_rows, filter_text_lower);

        if (!filter_text_lower.empty() &&
            lowerCopy(snapshot.name).find(filter_text_lower) == std::string::npos) {
            rows.insert(rows.end(), child_rows.begin(), child_rows.end());
            return;
        }

        rows.push_back(FlatRow{
            .id = snapshot.id,
            .type = snapshot.type,
            .depth = depth,
            .visible = snapshot.visible,
            .has_children = snapshot.has_children,
            .collapsed = collapsed_ids_.contains(snapshot.id),
            .draggable = snapshot.draggable,
            .training_enabled = snapshot.training_enabled,
            .label = snapshot.label,
            .node_id_text = std::to_string(snapshot.id),
            .encoded_label = encode(snapshot.label),
            .padding_left_dp = formatDp(4 + depth * 16),
            .has_mask = snapshot.has_mask,
            .deletable = snapshot.deletable,
        });
        if (snapshot.has_children && !collapsed_ids_.contains(snapshot.id))
            rows.insert(rows.end(), child_rows.begin(), child_rows.end());
    }

    void SceneGraphElement::appendVisibleSubtreeRows(const core::NodeId node_id,
                                                     const int depth,
                                                     std::vector<FlatRow>& rows) const {
        const auto it = node_snapshots_.find(node_id);
        if (it == node_snapshots_.end())
            return;

        const NodeSnapshot& snapshot = it->second;
        rows.push_back(FlatRow{
            .id = snapshot.id,
            .type = snapshot.type,
            .depth = depth,
            .visible = snapshot.visible,
            .has_children = snapshot.has_children,
            .collapsed = collapsed_ids_.contains(snapshot.id),
            .draggable = snapshot.draggable,
            .training_enabled = snapshot.training_enabled,
            .label = snapshot.label,
            .node_id_text = std::to_string(snapshot.id),
            .encoded_label = encode(snapshot.label),
            .padding_left_dp = formatDp(4 + depth * 16),
            .has_mask = snapshot.has_mask,
            .deletable = snapshot.deletable,
        });

        if (snapshot.has_children && !collapsed_ids_.contains(snapshot.id)) {
            for (const core::NodeId child_id : snapshot.children)
                appendVisibleSubtreeRows(child_id, depth + 1, rows);
        }
    }

    void SceneGraphElement::rebuildIndex() {
        flat_index_by_id_.clear();
        flat_index_by_id_.reserve(flat_rows_.size());
        for (size_t i = 0; i < flat_rows_.size(); ++i)
            flat_index_by_id_[flat_rows_[i].id] = i;
    }

    void SceneGraphElement::rebuildFlatRows(const core::Scene& scene) {
        std::unordered_map<core::NodeId, NodeSnapshot> snapshots;
        std::vector<core::NodeId> root_ids;
        captureSceneSnapshot(scene, snapshots, root_ids);

        node_snapshots_ = std::move(snapshots);
        root_ids_ = std::move(root_ids);
        root_count_ = root_ids_.size();
        scene_has_nodes_ = !root_ids_.empty();

        if (!scene_has_nodes_) {
            flat_rows_.clear();
            flat_index_by_id_.clear();
            row_top_dp_cache_.clear();
            return;
        }

        std::vector<FlatRow> rows;
        const std::string filter_text_lower = lowerCopy(filter_text_);
        for (const core::NodeId root_id : root_ids_)
            appendSnapshotRows(root_id, 0, rows, filter_text_lower);

        flat_rows_ = std::move(rows);
        rebuildIndex();
        row_top_dp_cache_.resize(flat_rows_.size());
        for (size_t i = 0; i < row_top_dp_cache_.size(); ++i)
            row_top_dp_cache_[i] = formatDp(kHeaderHeightDpInt + static_cast<int>(i) * kRowHeightDpInt);
        tree_rebuild_needed_ = false;

        std::erase_if(selected_ids_, [this](const core::NodeId id) {
            return !node_snapshots_.contains(id);
        });
        if (click_anchor_id_ != core::NULL_NODE && !node_snapshots_.contains(click_anchor_id_))
            click_anchor_id_ = core::NULL_NODE;
        if (rename_node_id_ != core::NULL_NODE && !node_snapshots_.contains(rename_node_id_)) {
            rename_node_id_ = core::NULL_NODE;
            rename_buffer_.clear();
        }
        if (context_menu_node_id_ != core::NULL_NODE && !node_snapshots_.contains(context_menu_node_id_))
            context_menu_node_id_ = core::NULL_NODE;
        if (drag_source_id_ != core::NULL_NODE && !node_snapshots_.contains(drag_source_id_))
            drag_source_id_ = core::NULL_NODE;
        if (drop_target_id_ != core::NULL_NODE && !node_snapshots_.contains(drop_target_id_))
            drop_target_id_ = core::NULL_NODE;
    }

    bool SceneGraphElement::syncFromScene(const PanelDrawContext& ctx) {
        const auto* scene = ctx.scene;
        auto* scene_manager = services().sceneOrNull();
        const bool invert_masks =
            services().paramsOrNull() ? services().paramsOrNull()->getActiveParams().invert_masks
                                      : false;

        bool changed = false;
        if (invert_masks != invert_masks_) {
            invert_masks_ = invert_masks;
            markStateDirty();
            changed = true;
        }

        if (!scene || !scene->hasNodes()) {
            if (scene_has_nodes_ || !node_snapshots_.empty()) {
                clear();
                changed = true;
            }
            syncVisibleRows(true);
            return changed;
        }

        changed |= syncSelectionFromScene(*scene, scene_manager);

        if (tree_rebuild_needed_ || last_scene_generation_ != ctx.scene_generation) {
            captureRenameBuffer();
            rebuildFlatRows(*scene);
            last_scene_generation_ = ctx.scene_generation;
            markStateDirty();
            changed = true;
        }

        if (changed)
            syncVisibleRows(true);
        return changed;
    }

    void SceneGraphElement::updateHeader() {
        if (!header_el_ || !header_arrow_el_ || !header_label_el_)
            return;

        const bool expanded = !models_collapsed_;
        const bool visible = scene_has_nodes_;
        const std::string header_text =
            encode(formatLocalizedCount(tr(string_keys::Scene::MODELS), root_count_));

        if (expanded != last_header_expanded_) {
            header_el_->SetClass("is-expanded", expanded);
            header_arrow_el_->SetInnerRML(expanded ? "\u25BC" : "\u25B6");
            last_header_expanded_ = expanded;
        }
        if (header_text != last_header_text_) {
            header_label_el_->SetInnerRML(header_text);
            last_header_text_ = header_text;
        }
        if (visible != last_header_visible_) {
            header_el_->SetProperty("display", visible ? "block" : "none");
            last_header_visible_ = visible;
        }
    }

    void SceneGraphElement::updateContentHeight() {
        if (!content_el_)
            return;

        if (!scene_has_nodes_) {
            if (last_content_height_ != 0.0f) {
                content_el_->SetProperty("height", "0dp");
                last_content_height_ = 0.0f;
            }
            return;
        }

        const float total_height = models_collapsed_
                                       ? kHeaderHeightDp
                                       : (kHeaderHeightDp + static_cast<float>(flat_rows_.size()) * kRowHeightDp);
        if (std::abs(total_height - last_content_height_) < 0.01f)
            return;

        content_el_->SetProperty("height", std::format("{:.1f}dp", total_height));
        last_content_height_ = total_height;
    }

    void SceneGraphElement::hideRow(RowSlot& slot) {
        if (!slot.root)
            return;
        if (!slot.visible && slot.bound_id == core::NULL_NODE)
            return;
        setCachedProperty(slot.root, "display", "none");
        slot.bound_id = core::NULL_NODE;
        slot.visible = false;
    }

    void SceneGraphElement::bindRow(RowSlot& slot, const FlatRow& row, const size_t absolute_index) {
        const auto snapshot_it = node_snapshots_.find(row.id);
        if (snapshot_it == node_snapshots_.end()) {
            hideRow(slot);
            return;
        }
        const NodeSnapshot& snapshot = snapshot_it->second;

        setCachedProperty(slot.root, "display", "block");
        setCachedAttribute(slot.root, "data-node-id", row.node_id_text);
        if (absolute_index < row_top_dp_cache_.size())
            setCachedProperty(slot.root, "top", row_top_dp_cache_[absolute_index]);
        else
            setCachedProperty(slot.root, "top",
                              formatDp(kHeaderHeightDpInt + static_cast<int>(absolute_index) * kRowHeightDpInt));
        setCachedProperty(slot.root, "padding-left", row.padding_left_dp);
        setCachedProperty(slot.root, "drag", row.draggable ? "drag-drop" : "none");
        setCachedClass(slot.root, "even", absolute_index % 2 == 0);
        setCachedClass(slot.root, "odd", absolute_index % 2 == 1);
        setCachedClass(slot.root, "selected", selected_ids_.contains(row.id));
        setCachedClass(slot.root, "drop-target", drop_target_id_ == row.id);

        setCachedAttribute(slot.vis_icon, "data-node-id", row.node_id_text);
        setCachedAttribute(slot.vis_icon, "sprite", row.visible ? "icon-visible" : "icon-hidden");
        setCachedClass(slot.vis_icon, "icon-vis-on", row.visible);
        setCachedClass(slot.vis_icon, "icon-vis-off", !row.visible);

        setCachedAttribute(slot.delete_icon, "data-node-id", row.node_id_text);
        setCachedProperty(slot.delete_icon, "display", row.deletable ? "inline" : "none");

        const std::string_view icon_path = typeIconPath(row.type);
        const std::string_view unicode = unicodeIcon(row.type);
        setCachedTypeClass(slot.type_icon, typeClass(row.type));
        setCachedTypeClass(slot.unicode_icon, typeClass(row.type));

        if (!icon_path.empty()) {
            setCachedAttribute(slot.type_icon, "src", std::string(icon_path));
            setCachedProperty(slot.type_icon, "display", "inline");
            setCachedProperty(slot.unicode_icon, "display", "none");
        } else if (!unicode.empty()) {
            setCachedInnerRml(slot.unicode_icon, std::string(unicode));
            setCachedProperty(slot.unicode_icon, "display", "inline");
            setCachedProperty(slot.type_icon, "display", "none");
        } else {
            setCachedProperty(slot.type_icon, "display", "none");
            setCachedProperty(slot.unicode_icon, "display", "none");
        }

        setCachedProperty(slot.mask_icon, "display", row.has_mask ? "inline" : "none");
        setCachedClass(slot.mask_icon, "mask-inverted", row.has_mask && invert_masks_);

        if (row.has_children) {
            setCachedInnerRml(slot.expand_toggle, row.collapsed ? "\u25B6" : "\u25BC");
            setCachedProperty(slot.expand_toggle, "display", "inline");
            setCachedAttribute(slot.expand_toggle, "data-node-id", row.node_id_text);
            setCachedProperty(slot.leaf_spacer, "display", "none");
        } else {
            setCachedProperty(slot.expand_toggle, "display", "none");
            setCachedProperty(slot.leaf_spacer, "display", "inline");
        }

        const bool renaming = rename_node_id_ == row.id;
        setCachedAttribute(slot.rename_input, "data-node-id", row.node_id_text);
        setCachedProperty(slot.rename_input, "display", renaming ? "inline" : "none");
        setCachedProperty(slot.node_name, "display", renaming ? "none" : "block");
        setCachedClass(slot.node_name, "training-disabled",
                       row.type == core::NodeType::CAMERA && !row.training_enabled);
        setCachedInnerRml(slot.node_name, row.encoded_label);
        if (renaming) {
            if (rename_buffer_.empty())
                rename_buffer_ = snapshot.name;
            setCachedAttribute(slot.rename_input, "value", rename_buffer_);
            if (rename_focus_pending_ && slot.rename_input->Focus())
                rename_focus_pending_ = false;
        }

        slot.bound_id = row.id;
        slot.visible = true;
    }

    void SceneGraphElement::syncVisibleRows(const bool force) {
        ensureDom();
        if (!content_el_)
            return;

        captureRenameBuffer();
        updateHeader();
        updateContentHeight();

        // Rml scroll metrics already use the same logical units as our row heights.
        const float row_height = kRowHeightDp;
        const float header_height = kHeaderHeightDp;
        const float client_height = GetClientHeight();
        const float scroll_top = GetScrollTop();
        const bool has_prev_window =
            last_visible_start_ != kUnsetVisibleRange && last_visible_end_ != kUnsetVisibleRange;
        const size_t prev_start = has_prev_window ? last_visible_start_ : 0;
        const size_t prev_end = has_prev_window ? last_visible_end_ : 0;
        const size_t prev_count = has_prev_window ? prev_end - prev_start : 0;

        if (!scene_has_nodes_ || models_collapsed_ || client_height <= 0.0f) {
            for (RowSlot& slot : row_slots_)
                hideRow(slot);
            last_visible_start_ = 0;
            last_visible_end_ = 0;
            last_bound_revision_ = state_revision_;
            last_client_height_ = client_height;
            return;
        }

        const float rows_scroll_top = std::max(0.0f, scroll_top - header_height);
        const size_t start = std::max(0, static_cast<int>(rows_scroll_top / row_height) - kOverscanRows);
        const size_t visible_count = static_cast<size_t>(std::max(
            1, static_cast<int>(std::ceil(client_height / row_height)) + kOverscanRows * 2));
        const size_t end = std::min(flat_rows_.size(), start + visible_count);

        if (!force &&
            last_bound_revision_ == state_revision_ &&
            last_visible_start_ == start &&
            last_visible_end_ == end &&
            std::abs(client_height - last_client_height_) < 0.5f &&
            !dom_dirty_) {
            return;
        }

        ensureRowPool(visible_count);

        bool rebound_incrementally = false;
        const size_t next_count = end - start;
        const bool state_unchanged =
            !force && last_bound_revision_ == state_revision_ && !dom_dirty_ &&
            std::abs(client_height - last_client_height_) < 0.5f;
        if (state_unchanged && prev_count == next_count && prev_count > 0) {
            const ptrdiff_t delta =
                static_cast<ptrdiff_t>(start) - static_cast<ptrdiff_t>(prev_start);
            if (delta > 0 && static_cast<size_t>(delta) < prev_count) {
                const size_t shift = static_cast<size_t>(delta);
                std::rotate(row_slots_.begin(),
                            row_slots_.begin() + static_cast<ptrdiff_t>(shift),
                            row_slots_.begin() + static_cast<ptrdiff_t>(prev_count));
                for (size_t pool_index = prev_count - shift; pool_index < prev_count; ++pool_index) {
                    const size_t absolute_index = start + pool_index;
                    bindRow(row_slots_[pool_index], flat_rows_[absolute_index], absolute_index);
                }
                rebound_incrementally = true;
            } else if (delta < 0 && static_cast<size_t>(-delta) < prev_count) {
                const size_t shift = static_cast<size_t>(-delta);
                std::rotate(row_slots_.begin(),
                            row_slots_.begin() + static_cast<ptrdiff_t>(prev_count - shift),
                            row_slots_.begin() + static_cast<ptrdiff_t>(prev_count));
                for (size_t pool_index = 0; pool_index < shift; ++pool_index) {
                    const size_t absolute_index = start + pool_index;
                    bindRow(row_slots_[pool_index], flat_rows_[absolute_index], absolute_index);
                }
                rebound_incrementally = true;
            }
        }

        if (!rebound_incrementally) {
            size_t pool_index = 0;
            for (size_t absolute_index = start; absolute_index < end && pool_index < row_slots_.size();
                 ++absolute_index, ++pool_index) {
                bindRow(row_slots_[pool_index], flat_rows_[absolute_index], absolute_index);
            }
            for (; pool_index < row_slots_.size(); ++pool_index)
                hideRow(row_slots_[pool_index]);
        } else {
            for (size_t pool_index = next_count; pool_index < row_slots_.size(); ++pool_index)
                hideRow(row_slots_[pool_index]);
        }

        last_visible_start_ = start;
        last_visible_end_ = end;
        last_bound_revision_ = state_revision_;
        last_client_height_ = client_height;
        dom_dirty_ = false;
    }

    void SceneGraphElement::scrollNodeIntoView(const core::NodeId node_id) {
        const auto it = flat_index_by_id_.find(node_id);
        if (it == flat_index_by_id_.end())
            return;

        const float row_top = kHeaderHeightDp + static_cast<float>(it->second) * kRowHeightDp;
        const float row_bottom = row_top + kRowHeightDp;
        const float scroll_top = GetScrollTop();
        const float view_h = GetClientHeight();

        if (row_top < scroll_top)
            SetScrollTop(row_top);
        else if (row_bottom > scroll_top + view_h)
            SetScrollTop(row_bottom - view_h);
    }

    void SceneGraphElement::focusTree() {
        if (rename_node_id_ == core::NULL_NODE)
            Focus();
    }

    void SceneGraphElement::beginRename(const core::NodeId node_id) {
        const auto it = node_snapshots_.find(node_id);
        if (it == node_snapshots_.end() || !it->second.deletable)
            return;
        rename_node_id_ = node_id;
        rename_buffer_ = it->second.name;
        rename_focus_pending_ = true;
        markStateDirty();
        syncVisibleRows(true);
    }

    void SceneGraphElement::confirmRename() {
        if (rename_node_id_ == core::NULL_NODE)
            return;
        captureRenameBuffer();
        auto* scene_manager = services().sceneOrNull();
        auto* scene = scene_manager ? &scene_manager->getScene() : nullptr;
        if (!scene)
            return;

        const auto it = node_snapshots_.find(rename_node_id_);
        if (it == node_snapshots_.end())
            return;

        const std::string next_name = rename_buffer_.empty() ? it->second.name : rename_buffer_;
        if (next_name != it->second.name)
            scene->renameNode(it->second.name, next_name);

        rename_node_id_ = core::NULL_NODE;
        rename_buffer_.clear();
        markStateDirty();
        syncVisibleRows(true);
    }

    void SceneGraphElement::cancelRename() {
        if (rename_node_id_ == core::NULL_NODE)
            return;
        rename_node_id_ = core::NULL_NODE;
        rename_buffer_.clear();
        markStateDirty();
    }

    void SceneGraphElement::handleInlineAction(const std::string& action, const core::NodeId node_id) {
        auto* scene_manager = services().sceneOrNull();
        auto* scene = scene_manager ? &scene_manager->getScene() : nullptr;
        if (!scene)
            return;

        const auto* node = scene->getNodeById(node_id);
        if (!node)
            return;

        if (action == "toggle-vis") {
            if (auto it = node_snapshots_.find(node_id); it != node_snapshots_.end()) {
                it->second.visible = !static_cast<bool>(node->visible);
                markStateDirty();
            }
            scene->setNodeVisibility(node->name, !static_cast<bool>(node->visible));
        } else if (action == "delete") {
            scene_manager->removePLY(node->name, false);
            tree_rebuild_needed_ = true;
            markStateDirty();
        }
    }

    void SceneGraphElement::handlePrimaryClick(const core::NodeId node_id) {
        auto* scene_manager = services().sceneOrNull();
        auto* scene = scene_manager ? &scene_manager->getScene() : nullptr;
        if (!scene || !node_snapshots_.contains(node_id))
            return;

        const bool ctrl = ctrlDown();
        const bool shift = shiftDown();

        if (ctrl) {
            if (selected_ids_.contains(node_id)) {
                selected_ids_.erase(node_id);
                std::vector<std::string> names;
                names.reserve(selected_ids_.size());
                for (const core::NodeId id : selected_ids_) {
                    if (const auto it = node_snapshots_.find(id); it != node_snapshots_.end())
                        names.push_back(it->second.name);
                }
                scene_manager->selectNodes(names);
            } else {
                scene_manager->addToSelection(node_snapshots_[node_id].name);
                selected_ids_.insert(node_id);
            }
            click_anchor_id_ = node_id;
        } else if (shift && click_anchor_id_ != core::NULL_NODE) {
            scene_manager->selectNodes(rangeSelectionNames(click_anchor_id_, node_id));
            selected_ids_.clear();
            const auto lo = flat_index_by_id_.find(click_anchor_id_);
            const auto hi = flat_index_by_id_.find(node_id);
            if (lo != flat_index_by_id_.end() && hi != flat_index_by_id_.end()) {
                const size_t start = std::min(lo->second, hi->second);
                const size_t end = std::max(lo->second, hi->second);
                for (size_t i = start; i <= end; ++i)
                    selected_ids_.insert(flat_rows_[i].id);
            } else {
                selected_ids_.insert(node_id);
            }
        } else {
            if (selected_ids_.size() == 1 && selected_ids_.contains(node_id))
                return;
            scene_manager->selectNode(node_snapshots_[node_id].name);
            selected_ids_.clear();
            selected_ids_.insert(node_id);
            click_anchor_id_ = node_id;
        }
        markStateDirty();
    }

    void SceneGraphElement::handleSecondaryClick(const core::NodeId node_id,
                                                 const float mouse_x,
                                                 const float mouse_y) {
        auto* scene_manager = services().sceneOrNull();
        if (!scene_manager || !node_snapshots_.contains(node_id))
            return;

        focusTree();
        if (!selected_ids_.contains(node_id) || selected_ids_.size() != 1) {
            scene_manager->selectNode(node_snapshots_[node_id].name);
            selected_ids_.clear();
            selected_ids_.insert(node_id);
            click_anchor_id_ = node_id;
            markStateDirty();
        }
        showContextMenu(node_id, mouse_x, mouse_y);
    }

    bool SceneGraphElement::activateNode(const core::NodeId node_id) {
        auto* scene_manager = services().sceneOrNull();
        const auto* scene = scene_manager ? &scene_manager->getScene() : nullptr;
        if (!scene)
            return false;

        const auto* node = scene->getNodeById(node_id);
        if (!node)
            return false;

        if (node->type == core::NodeType::CAMERA) {
            cmd::OpenCameraPreview{.cam_id = node->camera_uid}.emit();
            return true;
        }
        if (node->type == core::NodeType::KEYFRAME && node->keyframe) {
            cmd::SequencerGoToKeyframe{.keyframe_index = node->keyframe->keyframe_index}.emit();
            return true;
        }
        return false;
    }

    std::vector<std::string> SceneGraphElement::rangeSelectionNames(const core::NodeId a,
                                                                    const core::NodeId b) const {
        const auto ita = flat_index_by_id_.find(a);
        const auto itb = flat_index_by_id_.find(b);
        if (ita == flat_index_by_id_.end() || itb == flat_index_by_id_.end()) {
            const auto it = node_snapshots_.find(b);
            return it != node_snapshots_.end() ? std::vector<std::string>{it->second.name}
                                               : std::vector<std::string>{};
        }

        const size_t lo = std::min(ita->second, itb->second);
        const size_t hi = std::max(ita->second, itb->second);
        std::vector<std::string> names;
        names.reserve(hi - lo + 1);
        for (size_t i = lo; i <= hi; ++i) {
            if (const auto it = node_snapshots_.find(flat_rows_[i].id); it != node_snapshots_.end())
                names.push_back(it->second.name);
        }
        return names;
    }

    core::NodeId SceneGraphElement::selectionCursor() const {
        if (click_anchor_id_ != core::NULL_NODE && selected_ids_.contains(click_anchor_id_))
            return click_anchor_id_;
        for (const FlatRow& row : flat_rows_) {
            if (selected_ids_.contains(row.id))
                return row.id;
        }
        return core::NULL_NODE;
    }

    bool SceneGraphElement::moveSelection(const int delta, const bool extend) {
        auto* scene_manager = services().sceneOrNull();
        if (!scene_manager || flat_rows_.empty())
            return false;

        const core::NodeId current = selectionCursor();
        core::NodeId target = core::NULL_NODE;
        if (current == core::NULL_NODE) {
            target = delta > 0 ? flat_rows_.front().id : flat_rows_.back().id;
        } else {
            const size_t index = flat_index_by_id_.contains(current) ? flat_index_by_id_.at(current) : 0;
            const size_t target_index = std::clamp<int>(
                static_cast<int>(index) + delta, 0, static_cast<int>(flat_rows_.size()) - 1);
            target = flat_rows_[target_index].id;
        }
        if (target == core::NULL_NODE)
            return false;

        if (extend) {
            const core::NodeId anchor = click_anchor_id_ != core::NULL_NODE ? click_anchor_id_ : target;
            scene_manager->selectNodes(rangeSelectionNames(anchor, target));
            selected_ids_.clear();
            const auto lo = flat_index_by_id_.find(anchor);
            const auto hi = flat_index_by_id_.find(target);
            if (lo != flat_index_by_id_.end() && hi != flat_index_by_id_.end()) {
                const size_t start = std::min(lo->second, hi->second);
                const size_t end = std::max(lo->second, hi->second);
                for (size_t i = start; i <= end; ++i)
                    selected_ids_.insert(flat_rows_[i].id);
            } else {
                selected_ids_.insert(target);
            }
        } else {
            scene_manager->selectNode(node_snapshots_.at(target).name);
            selected_ids_.clear();
            selected_ids_.insert(target);
            click_anchor_id_ = target;
        }
        scrollNodeIntoView(target);
        markStateDirty();
        return true;
    }

    bool SceneGraphElement::isTextInputTarget(Rml::Element* target) const {
        return target && target->GetTagName() == "input";
    }

    SceneGraphElement::RowSlot* SceneGraphElement::rowSlotFromTarget(Rml::Element* target) {
        while (target && target != this) {
            const auto value = target->GetAttribute<Rml::String>("data-node-id", "");
            if (!value.empty()) {
                core::NodeId node_id = core::NULL_NODE;
                if (parseNodeId(value, node_id)) {
                    for (RowSlot& slot : row_slots_) {
                        if (slot.visible && slot.bound_id == node_id)
                            return &slot;
                    }
                }
            }
            target = target->GetParentNode();
        }
        return nullptr;
    }

    const SceneGraphElement::RowSlot* SceneGraphElement::rowSlotFromTarget(Rml::Element* target) const {
        return const_cast<SceneGraphElement*>(this)->rowSlotFromTarget(target);
    }

    core::NodeId SceneGraphElement::nodeIdFromTarget(Rml::Element* target) const {
        if (const RowSlot* slot = rowSlotFromTarget(target))
            return slot->bound_id;
        return core::NULL_NODE;
    }

    void SceneGraphElement::toggleExpand(const core::NodeId node_id) {
        if (!node_snapshots_.contains(node_id))
            return;
        if (collapsed_ids_.contains(node_id))
            collapsed_ids_.erase(node_id);
        else
            collapsed_ids_.insert(node_id);
        tree_rebuild_needed_ = true;
        markStateDirty();
    }

    void SceneGraphElement::toggleModelsSection() {
        models_collapsed_ = !models_collapsed_;
        markStateDirty();
        syncVisibleRows(true);
    }

    bool SceneGraphElement::setDropTarget(const core::NodeId node_id) {
        if (drop_target_id_ == node_id)
            return false;
        drop_target_id_ = node_id;
        markStateDirty();
        return true;
    }

    std::vector<std::string> SceneGraphElement::deletableSelectedNodeNames() const {
        std::vector<std::string> names;
        names.reserve(selected_ids_.size());
        for (const core::NodeId id : selected_ids_) {
            const auto it = node_snapshots_.find(id);
            if (it != node_snapshots_.end() && it->second.deletable)
                names.push_back(it->second.name);
        }
        return names;
    }

    void SceneGraphElement::deleteSelectedNodes() {
        auto* scene_manager = services().sceneOrNull();
        if (!scene_manager)
            return;
        for (const std::string& name : deletableSelectedNodeNames())
            scene_manager->removePLY(name, false);
    }

    void SceneGraphElement::toggleChildrenTraining(const core::NodeId group_id, const bool enabled) {
        auto* scene_manager = services().sceneOrNull();
        auto* scene = scene_manager ? &scene_manager->getScene() : nullptr;
        if (!scene)
            return;
        const auto* group = scene->getNodeById(group_id);
        if (!group)
            return;
        for (const core::NodeId child_id : group->children) {
            const auto* child = scene->getNodeById(child_id);
            if (child && child->type == core::NodeType::CAMERA)
                scene->setCameraTrainingEnabled(child->name, enabled);
        }
    }

    void SceneGraphElement::toggleSelectedTraining(const bool enabled) {
        auto* scene_manager = services().sceneOrNull();
        auto* scene = scene_manager ? &scene_manager->getScene() : nullptr;
        if (!scene)
            return;
        for (const core::NodeId id : selected_ids_) {
            const auto* node = scene->getNodeById(id);
            if (!node)
                continue;
            if (node->type == core::NodeType::CAMERA) {
                scene->setCameraTrainingEnabled(node->name, enabled);
            } else if (node->type == core::NodeType::CAMERA_GROUP) {
                for (const core::NodeId child_id : node->children) {
                    const auto* child = scene->getNodeById(child_id);
                    if (child && child->type == core::NodeType::CAMERA)
                        scene->setCameraTrainingEnabled(child->name, enabled);
                }
            }
        }
    }

    void SceneGraphElement::showContextMenu(const core::NodeId node_id,
                                            const float mouse_x,
                                            const float mouse_y) {
        auto* gui = services().guiOrNull();
        auto* scene_manager = services().sceneOrNull();
        auto* scene = scene_manager ? &scene_manager->getScene() : nullptr;
        if (!gui || !scene)
            return;

        const auto make_action = [](std::string label, std::string action,
                                    const bool separator_before = false,
                                    const bool submenu = false,
                                    const bool active = false) {
            return ContextMenuItem{
                .label = std::move(label),
                .action = std::move(action),
                .separator_before = separator_before,
                .is_label = false,
                .is_submenu_item = submenu,
                .is_active = active,
            };
        };
        const auto make_label = [](std::string label, const bool separator_before = false) {
            return ContextMenuItem{
                .label = std::move(label),
                .action = {},
                .separator_before = separator_before,
                .is_label = true,
                .is_submenu_item = false,
                .is_active = false,
            };
        };

        auto prefixed = [](const std::string& payload) {
            return std::string(kContextActionPrefix) + payload;
        };

        std::vector<ContextMenuItem> items;
        const auto* node = scene->getNodeById(node_id);
        if (!node)
            return;

        if (selected_ids_.size() > 1) {
            bool all_camera_like = true;
            for (const core::NodeId id : selected_ids_) {
                const auto* selected = scene->getNodeById(id);
                if (!selected)
                    continue;
                if (selected->type != core::NodeType::CAMERA &&
                    selected->type != core::NodeType::CAMERA_GROUP) {
                    all_camera_like = false;
                    break;
                }
            }
            if (all_camera_like) {
                items.push_back(make_action(tr(string_keys::Scene::ENABLE_ALL_TRAINING),
                                            prefixed("enable_all_selected_train")));
                items.push_back(make_action(tr(string_keys::Scene::DISABLE_ALL_TRAINING),
                                            prefixed("disable_all_selected_train")));
            }

            const auto deletable = deletableSelectedNodeNames();
            if (!deletable.empty()) {
                items.push_back(make_action(
                    std::format("{} ({})", tr(string_keys::Scene::DELETE_ITEM), deletable.size()),
                    prefixed("delete_selected"),
                    !items.empty()));
            }
        } else {
            switch (node->type) {
            case core::NodeType::CAMERA:
                items.push_back(make_action(
                    tr(string_keys::Scene::GO_TO_CAMERA_VIEW),
                    prefixed(std::format("go_to_camera:{}", node->camera_uid))));
                items.push_back(make_action(
                    node->training_enabled ? tr(string_keys::Scene::DISABLE_FOR_TRAINING)
                                           : tr(string_keys::Scene::ENABLE_FOR_TRAINING),
                    prefixed(std::format("{}:{}", node->training_enabled ? "disable_train" : "enable_train", node_id)),
                    true));
                break;
            case core::NodeType::KEYFRAME:
                if (node->keyframe) {
                    items.push_back(make_action(
                        tr(string_keys::Scene::GO_TO_KEYFRAME),
                        prefixed(std::format("go_to_kf:{}", node->keyframe->keyframe_index))));
                    items.push_back(make_action(
                        tr(string_keys::Scene::UPDATE_KEYFRAME),
                        prefixed(std::format("update_kf:{}", node->keyframe->keyframe_index))));
                    items.push_back(make_action(
                        tr("scene.select_in_timeline"),
                        prefixed(std::format("select_kf:{}", node->keyframe->keyframe_index))));
                    items.push_back(make_label(tr(string_keys::Scene::KEYFRAME_EASING), true));
                    static constexpr std::array easing_labels{
                        "scene.keyframe_easing.linear",
                        "scene.keyframe_easing.ease_in",
                        "scene.keyframe_easing.ease_out",
                        "scene.keyframe_easing.ease_in_out",
                    };
                    for (size_t i = 0; i < easing_labels.size(); ++i) {
                        items.push_back(make_action(
                            tr(easing_labels[i]),
                            prefixed(std::format("set_easing:{}:{}", node->keyframe->keyframe_index, i)),
                            false, true, node->keyframe->easing == i));
                    }
                    if (node->keyframe->keyframe_index > 0) {
                        items.push_back(make_action(
                            tr(string_keys::Scene::DELETE_ITEM),
                            prefixed(std::format("delete_kf:{}", node->keyframe->keyframe_index)),
                            true));
                    }
                }
                break;
            case core::NodeType::KEYFRAME_GROUP:
                items.push_back(make_action(
                    tr(string_keys::Scene::ADD_KEYFRAME_SCENE),
                    prefixed("add_kf")));
                break;
            case core::NodeType::CAMERA_GROUP:
                items.push_back(make_action(
                    tr(string_keys::Scene::ENABLE_ALL_TRAINING),
                    prefixed(std::format("enable_all_train:{}", node_id))));
                items.push_back(make_action(
                    tr(string_keys::Scene::DISABLE_ALL_TRAINING),
                    prefixed(std::format("disable_all_train:{}", node_id))));
                break;
            case core::NodeType::DATASET:
                items.push_back(make_action(
                    tr(string_keys::Scene::DELETE_ITEM),
                    prefixed(std::format("delete:{}", node_id))));
                break;
            case core::NodeType::CROPBOX:
                items.push_back(make_action(tr("common.apply"), prefixed("apply_cropbox")));
                items.push_back(make_action(
                    tr("scene.fit_to_scene"),
                    prefixed("fit_cropbox:0"),
                    true));
                items.push_back(make_action(
                    tr("scene.fit_to_scene_trimmed"),
                    prefixed("fit_cropbox:1")));
                items.push_back(make_action(tr("scene.reset_crop"), prefixed("reset_cropbox")));
                items.push_back(make_action(
                    tr(string_keys::Scene::DELETE_ITEM),
                    prefixed(std::format("delete:{}", node_id)),
                    true));
                break;
            case core::NodeType::ELLIPSOID:
                items.push_back(make_action(tr("common.apply"), prefixed("apply_ellipsoid")));
                items.push_back(make_action(
                    tr("scene.fit_to_scene"),
                    prefixed("fit_ellipsoid:0"),
                    true));
                items.push_back(make_action(
                    tr("scene.fit_to_scene_trimmed"),
                    prefixed("fit_ellipsoid:1")));
                items.push_back(make_action(tr("scene.reset_crop"), prefixed("reset_ellipsoid")));
                items.push_back(make_action(
                    tr(string_keys::Scene::DELETE_ITEM),
                    prefixed(std::format("delete:{}", node_id)),
                    true));
                break;
            default:
                break;
            }

            if (node->type == core::NodeType::GROUP) {
                items.push_back(make_action(
                    tr("scene.add_group_ellipsis"),
                    prefixed(std::format("add_group:{}", node_id)),
                    !items.empty()));
                items.push_back(make_action(
                    tr("scene.merge_to_single_ply"),
                    prefixed(std::format("merge_group:{}", node_id))));
            }

            if (node->type == core::NodeType::SPLAT || node->type == core::NodeType::POINTCLOUD) {
                items.push_back(make_action(
                    tr("scene.add_crop_box"),
                    prefixed(std::format("add_cropbox:{}", node_id)),
                    !items.empty()));
                items.push_back(make_action(
                    tr("scene.add_crop_ellipsoid"),
                    prefixed(std::format("add_ellipsoid:{}", node_id))));
                items.push_back(make_action(
                    tr("scene.save_to_disk"),
                    prefixed(std::format("save_node:{}", node_id))));
            }

            if (node_snapshots_.at(node_id).deletable) {
                items.push_back(make_action(
                    tr(string_keys::Scene::RENAME),
                    prefixed(std::format("rename:{}", node_id)),
                    !items.empty()));
            }

            items.push_back(make_action(
                tr("scene.duplicate"),
                prefixed(std::format("duplicate:{}", node_id))));

            if (node_snapshots_.at(node_id).draggable) {
                std::vector<std::pair<core::NodeId, std::string>> groups;
                for (const auto& [candidate_id, snapshot] : node_snapshots_) {
                    if (candidate_id != node_id && snapshot.type == core::NodeType::GROUP)
                        groups.emplace_back(candidate_id, snapshot.name);
                }
                if (!groups.empty()) {
                    items.push_back(make_label(tr("scene.move_to"), true));
                    items.push_back(make_action(
                        tr("scene.move_to_root"),
                        prefixed(std::format("reparent:{}:{}", node_id, core::NULL_NODE)),
                        false, true));
                    std::ranges::sort(groups, [](const auto& a, const auto& b) {
                        return a.second < b.second;
                    });
                    for (const auto& [group_id, group_name] : groups) {
                        items.push_back(make_action(
                            group_name,
                            prefixed(std::format("reparent:{}:{}", node_id, group_id)),
                            false, true));
                    }
                }
            }

            if (node_snapshots_.at(node_id).deletable) {
                items.push_back(make_action(
                    tr(string_keys::Scene::DELETE_ITEM),
                    prefixed(std::format("delete:{}", node_id)),
                    true));
            }
        }

        if (items.empty())
            return;

        context_menu_node_id_ = node_id;
        gui->globalContextMenu().request(std::move(items),
                                         panel_screen_x_ + mouse_x,
                                         panel_screen_y_ + mouse_y);
    }

    void SceneGraphElement::executeAction(const std::string& action) {
        auto* scene_manager = services().sceneOrNull();
        auto* scene = scene_manager ? &scene_manager->getScene() : nullptr;
        if (!scene_manager || !scene)
            return;

        const auto parts = splitAction(action);
        if (parts.empty())
            return;

        const std::string& kind = parts[0];
        if (kind == "go_to_camera" && parts.size() >= 2) {
            cmd::GoToCamView{.cam_id = std::stoi(parts[1])}.emit();
        } else if ((kind == "enable_train" || kind == "disable_train") && parts.size() >= 2) {
            core::NodeId node_id = core::NULL_NODE;
            if (parseNodeId(parts[1], node_id)) {
                if (const auto* node = scene->getNodeById(node_id))
                    scene->setCameraTrainingEnabled(node->name, kind == "enable_train");
            }
        } else if (kind == "go_to_kf" && parts.size() >= 2) {
            cmd::SequencerGoToKeyframe{.keyframe_index = static_cast<size_t>(std::stoul(parts[1]))}.emit();
        } else if (kind == "update_kf" && parts.size() >= 2) {
            const size_t index = static_cast<size_t>(std::stoul(parts[1]));
            cmd::SequencerSelectKeyframe{.keyframe_index = index}.emit();
            cmd::SequencerUpdateKeyframe{}.emit();
        } else if (kind == "select_kf" && parts.size() >= 2) {
            cmd::SequencerSelectKeyframe{.keyframe_index = static_cast<size_t>(std::stoul(parts[1]))}.emit();
        } else if (kind == "delete_kf" && parts.size() >= 2) {
            cmd::SequencerDeleteKeyframe{.keyframe_index = static_cast<size_t>(std::stoul(parts[1]))}.emit();
        } else if (kind == "add_kf") {
            cmd::SequencerAddKeyframe{}.emit();
        } else if ((kind == "enable_all_train" || kind == "disable_all_train") && parts.size() >= 2) {
            core::NodeId group_id = core::NULL_NODE;
            if (parseNodeId(parts[1], group_id))
                toggleChildrenTraining(group_id, kind == "enable_all_train");
        } else if (kind == "delete" && parts.size() >= 2) {
            core::NodeId node_id = core::NULL_NODE;
            if (parseNodeId(parts[1], node_id)) {
                if (const auto* node = scene->getNodeById(node_id))
                    scene_manager->removePLY(node->name, false);
            }
        } else if (kind == "rename" && parts.size() >= 2) {
            core::NodeId node_id = core::NULL_NODE;
            if (parseNodeId(parts[1], node_id))
                beginRename(node_id);
        } else if (kind == "duplicate" && parts.size() >= 2) {
            core::NodeId node_id = core::NULL_NODE;
            if (parseNodeId(parts[1], node_id)) {
                if (const auto* node = scene->getNodeById(node_id))
                    cmd::DuplicateNode{.name = node->name}.emit();
            }
        } else if (kind == "add_group" && parts.size() >= 2) {
            core::NodeId parent_id = core::NULL_NODE;
            if (parseNodeId(parts[1], parent_id)) {
                const std::string parent_name =
                    parent_id == core::NULL_NODE ? std::string{} : (scene->getNodeById(parent_id) ? scene->getNodeById(parent_id)->name : std::string{});
                cmd::AddGroup{.name = tr("scene.new_group_name"), .parent_name = parent_name}.emit();
            }
        } else if (kind == "merge_group" && parts.size() >= 2) {
            core::NodeId node_id = core::NULL_NODE;
            if (parseNodeId(parts[1], node_id)) {
                if (const auto* node = scene->getNodeById(node_id))
                    cmd::MergeGroup{.name = node->name}.emit();
            }
        } else if ((kind == "add_cropbox" || kind == "add_ellipsoid" || kind == "save_node") && parts.size() >= 2) {
            core::NodeId node_id = core::NULL_NODE;
            if (!parseNodeId(parts[1], node_id))
                return;
            if (const auto* node = scene->getNodeById(node_id)) {
                if (kind == "add_cropbox")
                    cmd::AddCropBox{.node_name = node->name}.emit();
                else if (kind == "add_ellipsoid")
                    cmd::AddCropEllipsoid{.node_name = node->name}.emit();
                else
                    saveNodeToDisk(*scene, node_id);
            }
        } else if (kind == "apply_cropbox") {
            cmd::ApplyCropBox{}.emit();
        } else if (kind == "fit_cropbox" && parts.size() >= 2) {
            cmd::FitCropBoxToScene{.use_percentile = parts[1] == "1"}.emit();
        } else if (kind == "reset_cropbox") {
            cmd::ResetCropBox{}.emit();
        } else if (kind == "apply_ellipsoid") {
            cmd::ApplyEllipsoid{}.emit();
        } else if (kind == "fit_ellipsoid" && parts.size() >= 2) {
            cmd::FitEllipsoidToScene{.use_percentile = parts[1] == "1"}.emit();
        } else if (kind == "reset_ellipsoid") {
            cmd::ResetEllipsoid{}.emit();
        } else if (kind == "enable_all_selected_train") {
            toggleSelectedTraining(true);
        } else if (kind == "disable_all_selected_train") {
            toggleSelectedTraining(false);
        } else if (kind == "delete_selected") {
            deleteSelectedNodes();
        } else if (kind == "set_easing" && parts.size() >= 3) {
            cmd::SequencerSetKeyframeEasing{
                .keyframe_index = static_cast<size_t>(std::stoul(parts[1])),
                .easing_type = std::stoi(parts[2]),
            }
                .emit();
        } else if (kind == "reparent" && parts.size() >= 3) {
            core::NodeId node_id = core::NULL_NODE;
            core::NodeId parent_id = core::NULL_NODE;
            if (!parseNodeId(parts[1], node_id) || !parseNodeId(parts[2], parent_id))
                return;
            const auto* node = scene->getNodeById(node_id);
            const auto* parent = parent_id == core::NULL_NODE ? nullptr : scene->getNodeById(parent_id);
            if (node)
                cmd::ReparentNode{.node_name = node->name, .new_parent_name = parent ? parent->name : std::string{}}.emit();
        }
    }

    bool SceneGraphElement::executeContextMenuAction(const std::string_view action) {
        if (!ownsContextMenuAction(action))
            return false;
        executeAction(std::string(action.substr(kContextActionPrefix.size())));
        return true;
    }

    void SceneGraphElement::ProcessDefaultAction(Rml::Event& event) {
        Element::ProcessDefaultAction(event);

        ensureDom();

        const std::string type = event.GetType();
        Rml::Element* target = event.GetTargetElement();
        if (!target)
            return;

        if (type == "click") {
            focusTree();

            if (target == header_el_ || target == header_arrow_el_ ||
                target->GetAttribute<Rml::String>("data-role", "") == "models-header") {
                toggleModelsSection();
                event.StopPropagation();
                return;
            }

            const std::string action = target->GetAttribute<Rml::String>("data-action", "");
            if (!action.empty()) {
                const core::NodeId node_id = nodeIdFromTarget(target);
                if (node_id != core::NULL_NODE) {
                    handleInlineAction(action, node_id);
                    event.StopPropagation();
                }
                return;
            }

            if (target->IsClassSet("expand-toggle")) {
                const core::NodeId node_id = nodeIdFromTarget(target);
                if (node_id != core::NULL_NODE) {
                    toggleExpand(node_id);
                    event.StopPropagation();
                }
                return;
            }

            if (isTextInputTarget(target))
                return;

            const core::NodeId node_id = nodeIdFromTarget(target);
            if (node_id != core::NULL_NODE) {
                handlePrimaryClick(node_id);
                event.StopPropagation();
            }
        } else if (type == "dblclick") {
            if (isTextInputTarget(target))
                return;
            const core::NodeId node_id = nodeIdFromTarget(target);
            if (node_id != core::NULL_NODE && activateNode(node_id))
                event.StopPropagation();
        } else if (type == "mousedown") {
            const int button = event.GetParameter("button", 0);
            if (button == 1) {
                const core::NodeId node_id = nodeIdFromTarget(target);
                if (node_id != core::NULL_NODE) {
                    handleSecondaryClick(node_id,
                                         event.GetParameter("mouse_x", 0.0f),
                                         event.GetParameter("mouse_y", 0.0f));
                    event.StopPropagation();
                }
            }
        } else if (type == "keydown") {
            const auto key =
                static_cast<Rml::Input::KeyIdentifier>(event.GetParameter("key_identifier", 0));

            if (isTextInputTarget(target))
                return;

            switch (key) {
            case Rml::Input::KI_UP:
                if (!ctrlDown() && moveSelection(-1, shiftDown()))
                    event.StopPropagation();
                break;
            case Rml::Input::KI_DOWN:
                if (!ctrlDown() && moveSelection(1, shiftDown()))
                    event.StopPropagation();
                break;
            case Rml::Input::KI_RETURN: {
                const core::NodeId node_id = selectionCursor();
                if (node_id != core::NULL_NODE && activateNode(node_id))
                    event.StopPropagation();
                break;
            }
            case Rml::Input::KI_F2:
                if (selected_ids_.size() == 1)
                    beginRename(*selected_ids_.begin());
                event.StopPropagation();
                break;
            case Rml::Input::KI_DELETE:
                if (rename_node_id_ == core::NULL_NODE) {
                    deleteSelectedNodes();
                    event.StopPropagation();
                }
                break;
            case Rml::Input::KI_ESCAPE:
                if (rename_node_id_ != core::NULL_NODE) {
                    cancelRename();
                    event.StopPropagation();
                }
                break;
            default:
                break;
            }
        } else if (type == "scroll") {
            syncVisibleRows(false);
        } else if (type == "blur") {
            if (rename_node_id_ != core::NULL_NODE && target->GetTagName() == "input")
                confirmRename();
        } else if (type == "dragstart") {
            drag_source_id_ = nodeIdFromTarget(target);
            setDropTarget(core::NULL_NODE);
        } else if (type == "dragend") {
            drag_source_id_ = core::NULL_NODE;
            setDropTarget(core::NULL_NODE);
        } else if (type == "dragover") {
            const core::NodeId target_id = nodeIdFromTarget(target);
            if (drag_source_id_ != core::NULL_NODE && target_id != drag_source_id_ && target_id != core::NULL_NODE) {
                setDropTarget(target_id);
                event.StopPropagation();
            }
        } else if (type == "dragdrop") {
            const core::NodeId target_id = nodeIdFromTarget(target);
            auto* scene_manager = services().sceneOrNull();
            auto* scene = scene_manager ? &scene_manager->getScene() : nullptr;
            if (scene && drag_source_id_ != core::NULL_NODE && target_id != core::NULL_NODE &&
                drag_source_id_ != target_id) {
                const auto* src = scene->getNodeById(drag_source_id_);
                const auto* dst = scene->getNodeById(target_id);
                if (src && dst)
                    cmd::ReparentNode{.node_name = src->name, .new_parent_name = dst->name}.emit();
                drag_source_id_ = core::NULL_NODE;
                setDropTarget(core::NULL_NODE);
                event.StopPropagation();
            }
        }
    }

} // namespace lfs::vis::gui
