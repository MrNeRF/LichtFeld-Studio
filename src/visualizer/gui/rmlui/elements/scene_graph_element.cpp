/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rmlui/elements/scene_graph_element.hpp"
#include "gui/scene_tree_session.hpp"

#include "core/event_bridge/localization_manager.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/provenance.hpp"
#include "core/tensor.hpp"
#include "gui/global_context_menu.hpp"
#include "gui/gui_focus_state.hpp"
#include "gui/gui_manager.hpp"
#include "gui/panel_registry.hpp"
#include "gui/string_keys.hpp"
#include "gui/utils/native_file_dialog.hpp"
#include "io/exporter.hpp"
#include "io/formats/colmap.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "training/training_manager.hpp"
#include "visualizer/core/parameter_manager.hpp"
#include "visualizer/core/services.hpp"
#include "visualizer/gui_capabilities.hpp"

#include <RmlUi/Core.h>
#include <RmlUi/Core/Context.h>
#include <RmlUi/Core/Event.h>
#include <RmlUi/Core/Input.h>
#include <SDL3/SDL_keyboard.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <format>
#include <functional>
#include <glm/mat4x4.hpp>
#include <optional>
#include <ranges>
#include <sstream>

namespace lfs::vis::gui {

    using namespace lfs::core::events;
    namespace string_keys = lichtfeld::Strings;

    namespace {

        constexpr std::string_view kContextActionPrefix = "scene_panel:";

        [[nodiscard]] std::string tr(const char* key) {
            const std::string value = LOC(key);
            return value.empty() ? std::string(key) : value;
        }

        [[nodiscard]] std::string formatWithThousands(const uint64_t value) {
            std::string result = std::to_string(value);
            for (int i = static_cast<int>(result.length()) - 3; i > 0; i -= 3)
                result.insert(i, ",");
            return result;
        }

        [[nodiscard]] std::string formatSplatLabel(const std::string& name, const size_t count) {
            if (count > 0)
                return std::format("{}  ({})", name, formatWithThousands(count));
            return name;
        }

        [[nodiscard]] std::string formatCameraGroupLabel(
            const std::string& name,
            const core::Scene::CameraTrainingCounts counts) {
            if (counts.enabled == counts.total)
                return std::format("{}  ({})", name, formatWithThousands(counts.total));
            return std::format("{}  ({} / {})",
                               name,
                               formatWithThousands(counts.enabled),
                               formatWithThousands(counts.total));
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

        [[nodiscard]] int colorChannel(const float value) {
            return std::clamp(static_cast<int>(std::round(value * 255.0f)), 0, 255);
        }

        [[nodiscard]] std::string formatCameraLossIconColor(const std::array<float, 3>& color) {
            constexpr float alpha = 1.0f;
            return std::format("rgba({}, {}, {}, {})",
                               colorChannel(color[0]),
                               colorChannel(color[1]),
                               colorChannel(color[2]),
                               colorChannel(alpha));
        }

        [[nodiscard]] lfs::io::Result<std::filesystem::path> colmapSparseSourcePath(
            const lfs::vis::SceneManager& scene_manager) {
            const auto dataset_path = scene_manager.getDatasetPath();
            if (dataset_path.empty()) {
                return lfs::io::make_error(lfs::io::ErrorCode::PATH_NOT_FOUND,
                                           "COLMAP export requires a source dataset path",
                                           dataset_path);
            }

            return lfs::io::find_colmap_sparse_model_path(dataset_path);
        }

        [[nodiscard]] std::string_view colmapExportExtension(const std::filesystem::path& source_path) {
            std::error_code ec;
            const bool has_binary_cameras = std::filesystem::exists(source_path / "cameras.bin", ec);
            ec.clear();
            const bool has_binary_images = std::filesystem::exists(source_path / "images.bin", ec);
            const bool has_binary_metadata = has_binary_cameras && has_binary_images;
            return has_binary_metadata ? "bin" : "txt";
        }

        [[nodiscard]] bool colmapSparseDataExists(const std::filesystem::path& output_path) {
            constexpr std::array<std::string_view, 6> file_names{
                "cameras.bin",
                "images.bin",
                "points3D.bin",
                "cameras.txt",
                "images.txt",
                "points3D.txt",
            };
            for (const std::string_view file_name : file_names) {
                std::error_code ec;
                if (std::filesystem::exists(output_path / std::string(file_name), ec)) {
                    return true;
                }
            }
            return false;
        }

        [[nodiscard]] float currentDpRatio(const Rml::Element* element) {
            const Rml::Context* context = element ? element->GetContext() : nullptr;
            return context ? std::max(context->GetDensityIndependentPixelRatio(), 0.01f) : 1.0f;
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

        bool setCachedOptionalProperty(Rml::Element* el,
                                       std::string_view name,
                                       const std::optional<std::string>& value) {
            if (!el)
                return false;

            const std::string attr_name = cacheAttrName("prop", name);
            const std::string encoded_value = value.value_or(std::string{});
            if (el->GetAttribute<Rml::String>(attr_name.c_str(), "") == encoded_value)
                return false;

            if (value)
                el->SetProperty(std::string(name), *value);
            else
                el->RemoveProperty(std::string(name));
            el->SetAttribute(attr_name, encoded_value);
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
            case core::NodeType::PLY_SEQUENCE: return "ply_sequence";
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

        [[nodiscard]] const char* typeIconSprite(const core::NodeType type) {
            switch (type) {
            case core::NodeType::SPLAT: return "icon-splat";
            case core::NodeType::POINTCLOUD: return "icon-pointcloud";
            case core::NodeType::GROUP: return "icon-group";
            case core::NodeType::PLY_SEQUENCE: return "icon-group";
            case core::NodeType::DATASET: return "icon-dataset";
            case core::NodeType::CAMERA:
            case core::NodeType::CAMERA_GROUP: return "icon-camera";
            case core::NodeType::CROPBOX: return "icon-cropbox";
            case core::NodeType::ELLIPSOID: return "icon-ellipsoid";
            case core::NodeType::MESH: return "icon-mesh";
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
            return type != core::NodeType::CAMERA_GROUP &&
                   type != core::NodeType::KEYFRAME &&
                   type != core::NodeType::KEYFRAME_GROUP &&
                   type != core::NodeType::CROPBOX &&
                   type != core::NodeType::ELLIPSOID &&
                   !parent_is_dataset;
        }

        [[nodiscard]] bool isRenamable(const core::NodeType type, const bool parent_is_dataset) {
            return type != core::NodeType::CAMERA &&
                   type != core::NodeType::CAMERA_GROUP &&
                   type != core::NodeType::KEYFRAME &&
                   type != core::NodeType::KEYFRAME_GROUP &&
                   type != core::NodeType::CROPBOX &&
                   type != core::NodeType::ELLIPSOID &&
                   !parent_is_dataset;
        }

        [[nodiscard]] bool canSaveAsAsset(const core::SceneNode& node) {
            switch (node.type) {
            case core::NodeType::SPLAT:
            case core::NodeType::POINTCLOUD:
            case core::NodeType::MESH:
            case core::NodeType::GROUP:
            case core::NodeType::PLY_SEQUENCE:
            case core::NodeType::DATASET:
                return true;
            default:
                return false;
            }
        }

        [[nodiscard]] bool canDrag(const core::NodeType type, const bool parent_is_dataset) {
            if (parent_is_dataset)
                return false;
            switch (type) {
            case core::NodeType::SPLAT:
            case core::NodeType::GROUP:
            case core::NodeType::PLY_SEQUENCE:
            case core::NodeType::POINTCLOUD:
            case core::NodeType::MESH:
                return true;
            default:
                return false;
            }
        }

        [[nodiscard]] std::unordered_set<core::NodeId> buildTrainingDeleteBlockedNodeIds(
            const core::Scene& scene,
            const TrainerManager* trainer) {
            std::unordered_set<core::NodeId> blocked;
            if (!trainer || trainer->canPerform(TrainingAction::DeleteTrainingNode)) {
                return blocked;
            }

            const auto mark_with_ancestors = [&](core::NodeId id) {
                for (const core::SceneNode* node = scene.getNodeById(id);
                     node != nullptr;
                     node = node->parent_id == core::NULL_NODE ? nullptr : scene.getNodeById(node->parent_id)) {
                    blocked.insert(node->id);
                }
            };

            const std::string& training_model_name = scene.getTrainingModelNodeName();
            if (!training_model_name.empty()) {
                const core::NodeId training_model_id = scene.getNodeIdByName(training_model_name);
                if (training_model_id != core::NULL_NODE) {
                    mark_with_ancestors(training_model_id);
                }
            }

            for (const core::SceneNode* node : scene.getNodes()) {
                if (node &&
                    node->type == core::NodeType::CAMERA &&
                    node->camera &&
                    node->training_enabled) {
                    mark_with_ancestors(node->id);
                }
            }

            return blocked;
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
                .provenance = core::make_provenance_stamp(),
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

            lfs::core::Tensor::trim_memory_pool();

            if (!result) {
                LOG_ERROR("Failed to save '{}' to {}: {}",
                          node->name,
                          lfs::core::path_to_utf8(path),
                          result.error().message);
            }
        }

        ContextMenuItem makeAction(std::string label,
                                   std::string action,
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
        }

        ContextMenuItem makeLabel(std::string label, const bool separator_before = false) {
            return ContextMenuItem{
                .label = std::move(label),
                .action = {},
                .separator_before = separator_before,
                .is_label = true,
                .is_submenu_item = false,
                .is_active = false,
            };
        }

        std::string prefixedAction(std::string_view payload) {
            return std::string(kContextActionPrefix) + std::string(payload);
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

    void SceneGraphElement::setModelsCollapsed(const bool /*collapsed*/) {
        // The redundant Models header was removed. Keep accepting legacy session
        // state, but never let an old collapsed flag hide the root rows.
        models_collapsed_ = false;
    }

    void SceneGraphElement::applySessionCollapseUuids(
        const std::vector<std::string>& uuids) {
        session_collapsed_uuids_.clear();
        session_collapsed_uuids_.insert(uuids.begin(), uuids.end());
        session_collapse_pending_ = true;
        tree_rebuild_needed_ = true;
        markStateDirty();
    }

    void SceneGraphElement::clearSessionCollapseUuids() {
        session_collapsed_uuids_.clear();
        session_collapse_pending_ = false;
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

    void SceneGraphElement::OnUpdate() {
        claimRenameTextInputFocus();
        ensureDom();
        const bool delete_training_node_allowed = [&] {
            const auto* trainer = services().trainerOrNull();
            return !trainer || trainer->canPerform(TrainingAction::DeleteTrainingNode);
        }();
        if (delete_training_node_allowed != last_delete_training_node_allowed_) {
            last_delete_training_node_allowed_ = delete_training_node_allowed;
            tree_rebuild_needed_ = true;
        }
        bool rebuilt_tree = false;
        if (tree_rebuild_needed_) {
            if (auto* scene_manager = services().sceneOrNull()) {
                const core::Scene& scene = scene_manager->getScene();
                syncSelectionFromScene(scene, scene_manager);

                if (scene.hasNodes()) {
                    captureRenameBuffer();
                    rebuildFlatRows(scene);
                    syncCameraLossIconColors(scene, true);
                    markStateDirty();
                    rebuilt_tree = true;
                } else if (scene_has_nodes_ || !node_snapshots_.empty()) {
                    clear();
                    rebuilt_tree = true;
                }
            }
        }
        const bool frustum_visibility_changed = syncCameraFrustumVisibility();
        syncVisibleRows(rebuilt_tree || frustum_visibility_changed);
    }

    void SceneGraphElement::RenameInputListener::ProcessEvent(Rml::Event& event) {
        if (!owner)
            return;
        auto* target = event.GetTargetElement();
        if (owner->rename_node_id_ == core::NULL_NODE || !owner->isTextInputTarget(target))
            return;

        const auto type = event.GetType();
        if (type == "input" || type == "change") {
            owner->captureRenameBuffer();
            event.StopPropagation();
        } else if (type == "escapecancel") {
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

        drag_listener_.owner = this;
        AddEventListener("dragstart", &drag_listener_);
        AddEventListener("dragover", &drag_listener_);
        AddEventListener("dragdrop", &drag_listener_);
        AddEventListener("dragend", &drag_listener_);

        auto content = doc->CreateElement("div");
        content->SetClass("scene-graph-content", true);
        content_el_ = AppendChild(std::move(content));
        if (!content_el_)
            return;

        auto header = doc->CreateElement("div");
        header->SetClass("section-header", true);
        header->SetClass("scene-graph-models-header", true);
        header->SetAttribute("data-role", "models-header");
        header_el_ = content_el_->AppendChild(std::move(header));
        if (!header_el_)
            return;

        auto arrow = doc->CreateElement("span");
        arrow->SetClass("section-arrow", true);
        header_arrow_el_ = header_el_->AppendChild(std::move(arrow));

        auto label = doc->CreateElement("span");
        header_label_el_ = header_el_->AppendChild(std::move(label));
        header_el_->SetProperty("display", "none");

        auto insert_line = doc->CreateElement("div");
        insert_line->SetClass("tree-insert-line", true);
        insert_line->SetProperty("display", "none");
        insertion_line_ = content_el_->AppendChild(std::move(insert_line));

        auto ghost = doc->CreateElement("div");
        ghost->SetClass("tree-drag-ghost", true);
        ghost->SetProperty("display", "none");
        drag_ghost_ = content_el_->AppendChild(std::move(ghost));

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
            row->SetProperty("display", "none");
            slot.root = content_el_->AppendChild(std::move(row));
            if (!slot.root)
                return;

            auto content = doc->CreateElement("span");
            content->SetClass("row-content", true);
            slot.content = slot.root->AppendChild(std::move(content));
            if (!slot.content)
                return;

            auto selection_checkbox = doc->CreateElement("span");
            selection_checkbox->SetClass("tree-checkbox", true);
            selection_checkbox->SetAttribute("data-action", "toggle-select");
            slot.selection_checkbox = slot.content->AppendChild(std::move(selection_checkbox));

            auto expand = doc->CreateElement("span");
            expand->SetClass("expand-toggle", true);
            slot.expand_toggle = slot.content->AppendChild(std::move(expand));

            auto leaf = doc->CreateElement("span");
            leaf->SetClass("leaf-spacer", true);
            slot.leaf_spacer = slot.content->AppendChild(std::move(leaf));

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
            mask_icon->SetAttribute("sprite", "icon-mask");
            slot.mask_icon = slot.content->AppendChild(std::move(mask_icon));

            auto rename_input = doc->CreateElement("input");
            rename_input->SetAttribute("type", "text");
            rename_input->SetClass("rename-input", true);
            rename_input->AddEventListener("input", &rename_input_listener_);
            rename_input->AddEventListener("change", &rename_input_listener_);
            slot.rename_input = slot.content->AppendChild(std::move(rename_input));

            auto node_name = doc->CreateElement("span");
            node_name->SetClass("node-name", true);
            slot.node_name = slot.content->AppendChild(std::move(node_name));

            auto actions = doc->CreateElement("span");
            actions->SetClass("row-actions", true);
            auto* actions_el = slot.content->AppendChild(std::move(actions));

            auto visibility_slot = doc->CreateElement("span");
            visibility_slot->SetClass("row-action-slot", true);
            auto* visibility_slot_el = actions_el->AppendChild(std::move(visibility_slot));
            auto vis_icon = doc->CreateElement("img");
            vis_icon->SetClass("row-icon", true);
            vis_icon->SetAttribute("data-action", "toggle-vis");
            slot.vis_icon = visibility_slot_el->AppendChild(std::move(vis_icon));

            auto training_slot = doc->CreateElement("span");
            training_slot->SetClass("row-action-slot", true);
            training_slot->SetClass("training-action-slot", true);
            auto* training_slot_el = actions_el->AppendChild(std::move(training_slot));

            auto training_toggle_icon = doc->CreateElement("img");
            training_toggle_icon->SetClass("row-icon", true);
            training_toggle_icon->SetClass("training-toggle-icon", true);
            training_toggle_icon->SetAttribute("data-action", "toggle-training");
            slot.training_toggle_icon = training_slot_el->AppendChild(std::move(training_toggle_icon));

            auto mixed_mark = doc->CreateElement("span");
            mixed_mark->SetClass("training-mixed-mark", true);
            slot.training_mixed_mark = training_slot_el->AppendChild(std::move(mixed_mark));

            auto delete_slot = doc->CreateElement("span");
            delete_slot->SetClass("row-action-slot", true);
            auto* delete_slot_el = actions_el->AppendChild(std::move(delete_slot));

            auto trash_icon = doc->CreateElement("img");
            trash_icon->SetClass("row-icon", true);
            trash_icon->SetClass("trash-icon", true);
            trash_icon->SetAttribute("sprite", "icon-trash");
            trash_icon->SetAttribute("data-action", "delete");
            slot.delete_icon = delete_slot_el->AppendChild(std::move(trash_icon));

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
        rename_conflict_modal_pending_ = false;
        context_menu_node_id_ = core::NULL_NODE;
        drag_source_id_ = core::NULL_NODE;
        drop_into_group_id_ = core::NULL_NODE;
        drop_parent_id_ = core::NULL_NODE;
        drop_index_ = -1;
        drop_valid_ = false;
        pending_reveal_node_id_ = core::NULL_NODE;
        scene_has_nodes_ = false;
        root_count_ = 0;
        last_training_model_node_name_.clear();
        last_training_model_gaussian_count_ = std::numeric_limits<size_t>::max();
        row_top_dp_cache_.clear();
        last_selection_generation_ = std::numeric_limits<uint32_t>::max();
        last_visible_start_ = kUnsetVisibleRange;
        last_visible_end_ = kUnsetVisibleRange;
        last_bound_dp_ratio_ = -1.0f;
        tree_rebuild_needed_ = false;
        markStateDirty();
    }

    void SceneGraphElement::markStateDirty() {
        ++state_revision_;
        dom_dirty_ = true;
    }

    void SceneGraphElement::claimRenameTextInputFocus() const {
        if (rename_node_id_ == core::NULL_NODE)
            return;
        auto& focus = guiFocusState();
        focus.want_capture_keyboard = true;
        focus.want_text_input = true;
        focus.any_item_active = true;
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
        if (selected_ids_.size() == 1)
            pending_reveal_node_id_ = *selected_ids_.begin();
        else
            pending_reveal_node_id_ = core::NULL_NODE;
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
        const auto delete_blocked_ids = buildTrainingDeleteBlockedNodeIds(scene, services().trainerOrNull());
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
            snapshot.has_mask = node->type == core::NodeType::CAMERA &&
                                (!node->mask_path.empty() ||
                                 (node->camera && node->camera->has_in_memory_mask()));

            switch (node->type) {
            case core::NodeType::SPLAT:
                if (const auto it = active_gaussian_counts.find(node->id); it != active_gaussian_counts.end())
                    snapshot.label = formatSplatLabel(node->name, it->second);
                else
                    snapshot.label = node->name;
                break;
            case core::NodeType::POINTCLOUD:
                snapshot.label = (node->point_cloud && node->point_cloud->size() > 0)
                                     ? std::format("{}  ({})", node->name, formatWithThousands(node->point_cloud->size()))
                                     : node->name;
                break;
            case core::NodeType::MESH:
                snapshot.label = (node->mesh)
                                     ? std::format("{}  ({}V / {}F)", node->name,
                                                   formatWithThousands(node->mesh->vertex_count()),
                                                   formatWithThousands(node->mesh->face_count()))
                                     : node->name;
                break;
            case core::NodeType::PLY_SEQUENCE: {
                const size_t frame_count = node->gaussian_count.load(std::memory_order_acquire);
                snapshot.label = LOCF(string_keys::Scene::PLY_SEQUENCE_LABEL,
                                      node->name,
                                      formatWithThousands(frame_count > 0 ? frame_count : node->children.size()));
                break;
            }
            case core::NodeType::CAMERA_GROUP: {
                const auto counts = scene.getCameraTrainingCounts(node->id);
                snapshot.label = formatCameraGroupLabel(
                    node->name,
                    counts);
                snapshot.training_enabled = counts.enabled == counts.total;
                snapshot.training_mixed = counts.enabled > 0 && counts.enabled < counts.total;
                snapshot.training_enabled_count = counts.enabled;
                snapshot.training_total_count = counts.total;
                break;
            }
            case core::NodeType::KEYFRAME:
                if (node->keyframe)
                    snapshot.label = LOCF(string_keys::Scene::KEYFRAME_NODE_LABEL,
                                          node->keyframe->keyframe_index + 1,
                                          node->keyframe->time);
                else
                    snapshot.label = node->name;
                break;
            default:
                snapshot.label = node->name;
                break;
            }
            switch (node->payload_hydration) {
            case core::PayloadHydrationState::Unloaded:
                snapshot.label += "  [Unloaded]";
                break;
            case core::PayloadHydrationState::Hydrating:
                snapshot.label += "  [Loading...]";
                break;
            case core::PayloadHydrationState::Failed:
                snapshot.label += "  [Load failed]";
                break;
            case core::PayloadHydrationState::NotApplicable:
            case core::PayloadHydrationState::Loaded:
                break;
            }

            snapshots.emplace(snapshot.id, std::move(snapshot));
            if (node->parent_id == core::NULL_NODE)
                root_ids.push_back(node->id);
        }

        auto collapsed_ids = collapsed_ids_;
        collapsed_ids_.clear();
        const bool apply_session_collapse =
            session_collapse_pending_ && !snapshots.empty();
        if (apply_session_collapse) {
            const auto session_ids =
                collapsedIdsFromUuids(scene, {session_collapsed_uuids_.begin(),
                                              session_collapsed_uuids_.end()});
            collapsed_ids_ = session_ids;
            session_collapse_pending_ = false;
        } else {
            for (const auto& [id, _] : snapshots) {
                if (collapsed_ids.contains(id))
                    collapsed_ids_.insert(id);
            }
        }

        for (auto& [id, snapshot] : snapshots) {
            const auto parent_it = snapshots.find(snapshot.parent_id);
            const bool parent_is_dataset =
                parent_it != snapshots.end() && parent_it->second.type == core::NodeType::DATASET;
            snapshot.draggable = canDrag(snapshot.type, parent_is_dataset);
            snapshot.can_delete = isDeletable(snapshot.type, parent_is_dataset);
            snapshot.delete_enabled = snapshot.can_delete && !delete_blocked_ids.contains(id);
            snapshot.can_rename = isRenamable(snapshot.type, parent_is_dataset);
            snapshot.rename_enabled = snapshot.can_rename;
            snapshot.can_toggle_training =
                snapshot.type == core::NodeType::CAMERA ||
                snapshot.type == core::NodeType::CAMERA_GROUP;
            snapshot.camera_frustum_container =
                (snapshot.type == core::NodeType::GROUP &&
                 std::ranges::any_of(snapshot.children, [&](const core::NodeId child_id) {
                     const auto child_it = snapshots.find(child_id);
                     return child_it != snapshots.end() &&
                            child_it->second.type == core::NodeType::CAMERA_GROUP;
                 })) ||
                (snapshot.type == core::NodeType::CAMERA_GROUP && (parent_it == snapshots.end() || parent_it->second.type != core::NodeType::GROUP));
            if (snapshot.camera_frustum_container) {
                if (const auto* rendering = services().renderingOrNull())
                    snapshot.visible = rendering->getSettings().show_camera_frustums;
            }
            if (!apply_session_collapse &&
                !previous_ids.contains(id) &&
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
            .training_mixed = snapshot.training_mixed,
            .training_enabled_count = snapshot.training_enabled_count,
            .training_total_count = snapshot.training_total_count,
            .can_toggle_training = snapshot.can_toggle_training,
            .name = snapshot.name,
            .label = snapshot.label,
            .node_id_text = std::to_string(snapshot.id),
            .encoded_label = encode(snapshot.label),
            .padding_left_dp = formatDp((selection_markers_visible_ ? 17 : 4) + depth * 16),
            .has_mask = snapshot.has_mask,
            .can_delete = snapshot.can_delete,
            .delete_enabled = snapshot.delete_enabled,
            .can_rename = snapshot.can_rename,
            .rename_enabled = snapshot.rename_enabled,
            .camera_loss_icon_color = snapshot.camera_loss_icon_color,
        });
        if (snapshot.has_children && !collapsed_ids_.contains(snapshot.id))
            rows.insert(rows.end(), child_rows.begin(), child_rows.end());
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
            row_top_dp_cache_[i] = formatDp(static_cast<int>(i) * kRowHeightDpInt);
        tree_rebuild_needed_ = false;

        std::erase_if(selected_ids_, [this](const core::NodeId id) {
            return !node_snapshots_.contains(id);
        });
        if (click_anchor_id_ != core::NULL_NODE && !node_snapshots_.contains(click_anchor_id_))
            click_anchor_id_ = core::NULL_NODE;
        if (rename_node_id_ != core::NULL_NODE && !node_snapshots_.contains(rename_node_id_)) {
            rename_node_id_ = core::NULL_NODE;
            rename_buffer_.clear();
            rename_conflict_modal_pending_ = false;
        }
        if (context_menu_node_id_ != core::NULL_NODE && !node_snapshots_.contains(context_menu_node_id_))
            context_menu_node_id_ = core::NULL_NODE;
        if (drag_source_id_ != core::NULL_NODE && !node_snapshots_.contains(drag_source_id_))
            drag_source_id_ = core::NULL_NODE;
        if (drop_into_group_id_ != core::NULL_NODE && !node_snapshots_.contains(drop_into_group_id_))
            drop_into_group_id_ = core::NULL_NODE;
    }

    bool SceneGraphElement::syncTrainingTopologyLabel(const core::Scene& scene,
                                                      const bool update_cached_rows) {
        const std::string& training_model_node_name = scene.getTrainingModelNodeName();
        const size_t gaussian_count = training_model_node_name.empty()
                                          ? 0
                                          : scene.getTrainingModelGaussianCount();

        if (training_model_node_name == last_training_model_node_name_ &&
            gaussian_count == last_training_model_gaussian_count_) {
            return false;
        }

        last_training_model_node_name_ = training_model_node_name;
        last_training_model_gaussian_count_ = gaussian_count;

        if (!update_cached_rows || training_model_node_name.empty())
            return false;

        const core::NodeId node_id = scene.getTrainingModelNodeId();
        const auto snapshot_it = node_snapshots_.find(node_id);
        if (node_id == core::NULL_NODE || snapshot_it == node_snapshots_.end() ||
            snapshot_it->second.type != core::NodeType::SPLAT) {
            return false;
        }

        const std::string label = formatSplatLabel(snapshot_it->second.name, gaussian_count);
        if (label == snapshot_it->second.label)
            return false;

        snapshot_it->second.label = label;
        if (const auto flat_it = flat_index_by_id_.find(node_id);
            flat_it != flat_index_by_id_.end() && flat_it->second < flat_rows_.size()) {
            FlatRow& row = flat_rows_[flat_it->second];
            row.label = label;
            row.encoded_label = encode(label);
        }
        markStateDirty();
        return true;
    }

    bool SceneGraphElement::syncCameraLossIconColors(const core::Scene& scene,
                                                     const bool update_cached_rows) {
        std::unordered_map<core::NodeId, std::string> camera_icon_colors;

        const auto* trainer_manager = services().trainerOrNull();
        const auto* trainer = trainer_manager ? trainer_manager->getTrainer() : nullptr;
        if (trainer) {
            std::vector<std::shared_ptr<const core::Camera>> cameras;
            std::vector<core::NodeId> camera_node_ids;
            const auto nodes = scene.getNodes();
            cameras.reserve(nodes.size());
            camera_node_ids.reserve(nodes.size());

            for (const core::SceneNode* node : nodes) {
                if (!node || node->type != core::NodeType::CAMERA || !node->camera ||
                    !node->training_enabled)
                    continue;
                cameras.push_back(node->camera);
                camera_node_ids.push_back(node->id);
            }

            std::vector<std::array<float, 3>> loss_colors;
            if (!cameras.empty() &&
                trainer->fillCameraLossColors(cameras, loss_colors) &&
                loss_colors.size() == cameras.size()) {
                camera_icon_colors.reserve(loss_colors.size());
                for (size_t i = 0; i < loss_colors.size(); ++i) {
                    const auto& color = loss_colors[i];
                    if (!std::isfinite(color[0]) ||
                        !std::isfinite(color[1]) ||
                        !std::isfinite(color[2])) {
                        continue;
                    }

                    camera_icon_colors.emplace(camera_node_ids[i], formatCameraLossIconColor(color));
                }
            }
        }

        bool changed = false;
        for (auto& [id, snapshot] : node_snapshots_) {
            if (snapshot.type != core::NodeType::CAMERA)
                continue;

            std::optional<std::string> next_color;
            if (const auto color_it = camera_icon_colors.find(id); color_it != camera_icon_colors.end())
                next_color = color_it->second;

            if (snapshot.camera_loss_icon_color == next_color)
                continue;

            snapshot.camera_loss_icon_color = std::move(next_color);
            if (update_cached_rows) {
                if (const auto flat_it = flat_index_by_id_.find(id);
                    flat_it != flat_index_by_id_.end() && flat_it->second < flat_rows_.size()) {
                    flat_rows_[flat_it->second].camera_loss_icon_color = snapshot.camera_loss_icon_color;
                }
            }
            changed = true;
        }

        if (changed)
            markStateDirty();
        return changed;
    }

    bool SceneGraphElement::syncCameraFrustumVisibility() {
        const auto* rendering = services().renderingOrNull();
        if (!rendering)
            return false;

        const bool visible = rendering->getSettings().show_camera_frustums;
        bool changed = false;
        for (auto& [id, snapshot] : node_snapshots_) {
            if (!snapshot.camera_frustum_container || snapshot.visible == visible)
                continue;

            snapshot.visible = visible;
            if (const auto flat_it = flat_index_by_id_.find(id);
                flat_it != flat_index_by_id_.end() && flat_it->second < flat_rows_.size()) {
                flat_rows_[flat_it->second].visible = visible;
            }
            changed = true;
        }

        if (changed)
            markStateDirty();
        return changed;
    }

    bool SceneGraphElement::syncFromScene(const PanelDrawContext& ctx) {
        claimRenameTextInputFocus();

        const auto* scene = ctx.scene;
        auto* scene_manager = services().sceneOrNull();
        const bool invert_masks =
            services().paramsOrNull() ? services().paramsOrNull()->getActiveParams().invert_masks
                                      : false;

        bool changed = false;
        const float dp_ratio = currentDpRatio(this);
        if (std::abs(dp_ratio - last_bound_dp_ratio_) > 0.001f) {
            markStateDirty();
            changed = true;
        }

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

        const bool tree_needs_rebuild =
            tree_rebuild_needed_ || last_scene_generation_ != ctx.scene_generation;
        if (tree_needs_rebuild) {
            captureRenameBuffer();
            rebuildFlatRows(*scene);
            last_scene_generation_ = ctx.scene_generation;
            syncTrainingTopologyLabel(*scene, false);
            syncCameraLossIconColors(*scene, true);
            markStateDirty();
            changed = true;
        } else {
            changed |= syncTrainingTopologyLabel(*scene, true);
            changed |= syncCameraLossIconColors(*scene, true);
        }
        changed |= syncCameraFrustumVisibility();

        if (pending_reveal_node_id_ != core::NULL_NODE) {
            const auto target_it = node_snapshots_.find(pending_reveal_node_id_);
            if (target_it == node_snapshots_.end()) {
                pending_reveal_node_id_ = core::NULL_NODE;
            } else {
                if (models_collapsed_) {
                    models_collapsed_ = false;
                    markStateDirty();
                    changed = true;
                }
                if (!flat_index_by_id_.contains(pending_reveal_node_id_)) {
                    bool expanded_any = false;
                    for (core::NodeId ancestor = target_it->second.parent_id;
                         ancestor != core::NULL_NODE;) {
                        if (collapsed_ids_.erase(ancestor) > 0)
                            expanded_any = true;
                        const auto parent_it = node_snapshots_.find(ancestor);
                        if (parent_it == node_snapshots_.end())
                            break;
                        ancestor = parent_it->second.parent_id;
                    }
                    if (expanded_any) {
                        rebuildFlatRows(*scene);
                        markStateDirty();
                        changed = true;
                    }
                }
            }
        }

        if (changed)
            syncVisibleRows(true);

        if (pending_reveal_node_id_ != core::NULL_NODE) {
            if (flat_index_by_id_.contains(pending_reveal_node_id_)) {
                scrollNodeIntoViewCentered(pending_reveal_node_id_);
                syncVisibleRows(true);
            }
            pending_reveal_node_id_ = core::NULL_NODE;
        }

        return changed;
    }

    void SceneGraphElement::updateHeader() {
        // Kept as a no-op while legacy session fields remain readable.
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

        const float total_height = static_cast<float>(flat_rows_.size()) * kRowHeightDp;
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
                              formatDp(static_cast<int>(absolute_index) * kRowHeightDpInt));
        setCachedProperty(slot.root, "padding-left", row.padding_left_dp);
        setCachedProperty(slot.root, "drag", row.draggable ? "drag-drop" : "none");
        setCachedClass(slot.root, "even", absolute_index % 2 == 0);
        setCachedClass(slot.root, "odd", absolute_index % 2 == 1);
        const auto [checkbox_checked, checkbox_mixed] = checkboxState(row.id);
        setCachedClass(slot.root, "selected", selected_ids_.contains(row.id) || checkbox_checked);
        setCachedClass(slot.root, "selection-partial", checkbox_mixed);
        setCachedClass(slot.root, "keyboard-cursor", keyboard_cursor_id_ == row.id);
        setCachedClass(slot.root, "drop-target", drop_into_group_id_ == row.id);
        setCachedClass(slot.root, "dragging",
                       drag_source_id_ != core::NULL_NODE && drag_source_id_ == row.id);

        setCachedAttribute(slot.vis_icon, "data-node-id", row.node_id_text);
        setCachedAttribute(slot.vis_icon, "sprite", row.visible ? "icon-visible" : "icon-hidden");
        setCachedClass(slot.vis_icon, "icon-vis-on", row.visible);
        setCachedClass(slot.vis_icon, "icon-vis-off", !row.visible);

        setCachedAttribute(slot.selection_checkbox, "data-node-id", row.node_id_text);
        setCachedProperty(slot.selection_checkbox, "display",
                          selection_markers_visible_ ? "inline-block" : "none");
        setCachedClass(slot.selection_checkbox, "checked", checkbox_checked);
        setCachedClass(slot.selection_checkbox, "mixed", checkbox_mixed);
        setCachedInnerRml(slot.selection_checkbox, checkbox_mixed ? "\u2212" : "");

        setCachedAttribute(slot.delete_icon, "data-node-id", row.node_id_text);
        setCachedProperty(slot.delete_icon, "display", row.can_delete ? "inline" : "none");
        setCachedClass(slot.delete_icon, "trash-disabled", row.can_delete && !row.delete_enabled);

        setCachedAttribute(slot.training_toggle_icon, "data-node-id", row.node_id_text);
        setCachedProperty(slot.training_toggle_icon, "display", row.can_toggle_training ? "inline" : "none");
        setCachedAttribute(slot.training_toggle_icon, "sprite",
                           row.training_enabled ? "icon-unlocked" : "icon-locked");
        const std::string training_title = row.training_mixed
                                               ? std::format("{} / {}",
                                                             row.training_enabled_count,
                                                             row.training_total_count)
                                           : row.training_enabled
                                               ? tr(string_keys::Scene::DISABLE_FOR_TRAINING)
                                               : tr(string_keys::Scene::ENABLE_FOR_TRAINING);
        setCachedAttribute(slot.training_toggle_icon, "title", training_title);
        setCachedClass(slot.training_toggle_icon, "training-disabled", !row.training_enabled);
        setCachedClass(slot.training_toggle_icon, "training-mixed", row.training_mixed);
        setCachedProperty(slot.training_mixed_mark, "display", row.training_mixed ? "block" : "none");

        const std::string_view icon_sprite = typeIconSprite(row.type);
        const std::string_view unicode = unicodeIcon(row.type);
        setCachedTypeClass(slot.type_icon, typeClass(row.type));
        setCachedTypeClass(slot.unicode_icon, typeClass(row.type));

        if (!icon_sprite.empty()) {
            setCachedAttribute(slot.type_icon, "sprite", std::string(icon_sprite));
            setCachedOptionalProperty(slot.type_icon, "image-color", row.camera_loss_icon_color);
            setCachedProperty(slot.type_icon, "display", "inline");
            setCachedProperty(slot.unicode_icon, "display", "none");
        } else if (!unicode.empty()) {
            setCachedOptionalProperty(slot.type_icon, "image-color", std::nullopt);
            setCachedInnerRml(slot.unicode_icon, std::string(unicode));
            setCachedProperty(slot.unicode_icon, "display", "inline");
            setCachedProperty(slot.type_icon, "display", "none");
        } else {
            setCachedOptionalProperty(slot.type_icon, "image-color", std::nullopt);
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

        claimRenameTextInputFocus();
        captureRenameBuffer();
        updateHeader();
        updateContentHeight();

        const float dp_ratio = currentDpRatio(this);
        const bool dp_ratio_changed = std::abs(dp_ratio - last_bound_dp_ratio_) > 0.001f;
        const float row_height = kRowHeightDp * dp_ratio;
        const float client_height = GetClientHeight();
        const float scroll_top = GetScrollTop();
        const bool has_prev_window =
            last_visible_start_ != kUnsetVisibleRange && last_visible_end_ != kUnsetVisibleRange;
        const size_t prev_start = has_prev_window ? last_visible_start_ : 0;
        const size_t prev_end = has_prev_window ? last_visible_end_ : 0;
        const size_t prev_count = has_prev_window ? prev_end - prev_start : 0;

        if (!scene_has_nodes_ || client_height <= 0.0f) {
            for (RowSlot& slot : row_slots_)
                hideRow(slot);
            last_visible_start_ = 0;
            last_visible_end_ = 0;
            last_bound_revision_ = state_revision_;
            last_bound_dp_ratio_ = dp_ratio;
            last_client_height_ = client_height;
            return;
        }

        const size_t start = std::max(0, static_cast<int>(scroll_top / row_height) - kOverscanRows);
        const size_t visible_count = static_cast<size_t>(std::max(
            1, static_cast<int>(std::ceil(client_height / row_height)) + kOverscanRows * 2));
        const size_t end = std::min(flat_rows_.size(), start + visible_count);

        if (!force &&
            last_bound_revision_ == state_revision_ &&
            last_visible_start_ == start &&
            last_visible_end_ == end &&
            std::abs(client_height - last_client_height_) < 0.5f &&
            !dp_ratio_changed &&
            !dom_dirty_) {
            return;
        }

        ensureRowPool(visible_count);

        bool rebound_incrementally = false;
        const size_t next_count = end - start;
        const bool state_unchanged =
            !force && last_bound_revision_ == state_revision_ && !dom_dirty_ &&
            std::abs(client_height - last_client_height_) < 0.5f &&
            !dp_ratio_changed;
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
        last_bound_dp_ratio_ = dp_ratio;
        last_client_height_ = client_height;
        dom_dirty_ = false;
    }

    void SceneGraphElement::scrollNodeIntoView(const core::NodeId node_id) {
        const auto it = flat_index_by_id_.find(node_id);
        if (it == flat_index_by_id_.end())
            return;

        const float dp_ratio = currentDpRatio(this);
        const float row_height = kRowHeightDp * dp_ratio;
        const float row_top = static_cast<float>(it->second) * row_height;
        const float row_bottom = row_top + row_height;
        const float scroll_top = GetScrollTop();
        const float view_h = GetClientHeight();

        if (row_top < scroll_top)
            SetScrollTop(row_top);
        else if (row_bottom > scroll_top + view_h)
            SetScrollTop(row_bottom - view_h);
    }

    void SceneGraphElement::scrollNodeIntoViewCentered(const core::NodeId node_id) {
        const auto it = flat_index_by_id_.find(node_id);
        if (it == flat_index_by_id_.end())
            return;

        const float view_h = GetClientHeight();
        if (view_h <= 0.0f) {
            scrollNodeIntoView(node_id);
            return;
        }

        const float dp_ratio = currentDpRatio(this);
        const float row_height = kRowHeightDp * dp_ratio;
        const float row_top = static_cast<float>(it->second) * row_height;
        const float content_h = static_cast<float>(flat_rows_.size()) * row_height;
        const float max_scroll = std::max(0.0f, content_h - view_h);
        const float desired = row_top + 0.5f * row_height - 0.5f * view_h;
        SetScrollTop(std::clamp(desired, 0.0f, max_scroll));
    }

    void SceneGraphElement::focusTree() {
        if (rename_node_id_ == core::NULL_NODE)
            Focus();
    }

    void SceneGraphElement::beginRename(const core::NodeId node_id) {
        const auto it = node_snapshots_.find(node_id);
        if (it == node_snapshots_.end() || !it->second.rename_enabled)
            return;
        rename_node_id_ = node_id;
        rename_buffer_ = it->second.name;
        rename_focus_pending_ = true;
        markStateDirty();
        syncVisibleRows(true);
    }

    void SceneGraphElement::confirmRename() {
        if (rename_node_id_ == core::NULL_NODE || rename_conflict_modal_pending_)
            return;
        captureRenameBuffer();
        auto* scene_manager = services().sceneOrNull();
        auto* scene = scene_manager ? &scene_manager->getScene() : nullptr;
        if (!scene)
            return;

        const auto it = node_snapshots_.find(rename_node_id_);
        if (it == node_snapshots_.end())
            return;
        if (!it->second.rename_enabled) {
            cancelRename();
            return;
        }

        const std::string next_name = rename_buffer_.empty() ? it->second.name : rename_buffer_;
        if (next_name != it->second.name) {
            const core::NodeId existing_id = scene->getNodeIdByName(next_name);
            if (existing_id != core::NULL_NODE && existing_id != rename_node_id_) {
                LOG_WARN("Scene node rename rejected: name '{}' is already in use", next_name);
                auto* gui = services().guiOrNull();
                if (!gui)
                    return;

                const core::NodeId rename_node_id = rename_node_id_;
                const std::string rename_button = tr(string_keys::Scene::RENAME);
                rename_conflict_modal_pending_ = true;

                lfs::core::ModalRequest request;
                request.title = tr(string_keys::Scene::RENAME_CONFLICT_TITLE);
                request.style = lfs::core::ModalStyle::Warning;
                request.width_dp = 440;
                request.body_rml = encode(LOCF(string_keys::Scene::RENAME_CONFLICT_MESSAGE, next_name));
                request.buttons = {
                    {tr(string_keys::Common::CANCEL), "secondary"},
                    {rename_button, "primary"},
                };
                request.on_result = [this, rename_node_id, rename_button](
                                        const lfs::core::ModalResult& result) {
                    rename_conflict_modal_pending_ = false;
                    if (result.button_label == rename_button && rename_node_id_ == rename_node_id) {
                        rename_focus_pending_ = true;
                        markStateDirty();
                        syncVisibleRows(true);
                    } else {
                        cancelRename();
                    }
                };
                request.on_cancel = [this] {
                    rename_conflict_modal_pending_ = false;
                    cancelRename();
                };
                gui->enqueueModal(std::move(request));
                return;
            }
            cmd::RenameNodeById{.node_id = static_cast<int32_t>(rename_node_id_), .new_name = next_name}.emit();
        }

        rename_node_id_ = core::NULL_NODE;
        rename_buffer_.clear();
        rename_conflict_modal_pending_ = false;
        markStateDirty();
        syncVisibleRows(true);
    }

    void SceneGraphElement::cancelRename() {
        if (rename_node_id_ == core::NULL_NODE)
            return;
        rename_node_id_ = core::NULL_NODE;
        rename_buffer_.clear();
        rename_conflict_modal_pending_ = false;
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

        if (action == "toggle-select") {
            toggleCheckboxSelection(node_id);
        } else if (action == "toggle-vis") {
            if (const auto snapshot_it = node_snapshots_.find(node_id);
                snapshot_it != node_snapshots_.end() && snapshot_it->second.camera_frustum_container) {
                if (auto* rendering = services().renderingOrNull()) {
                    auto settings = rendering->getSettings();
                    settings.show_camera_frustums = !settings.show_camera_frustums;
                    rendering->updateSettings(settings);
                    syncCameraFrustumVisibility();
                }
                return;
            }
            if (auto it = node_snapshots_.find(node_id); it != node_snapshots_.end()) {
                it->second.visible = !static_cast<bool>(node->visible);
                markStateDirty();
            }
            cmd::SetNodeVisibilityById{
                .node_id = static_cast<int32_t>(node_id),
                .visible = !static_cast<bool>(node->visible)}
                .emit();
        } else if (action == "delete") {
            const auto snapshot_it = node_snapshots_.find(node_id);
            if (snapshot_it == node_snapshots_.end() || !snapshot_it->second.delete_enabled)
                return;
            requestDeleteNodes({node_id});
        } else if (action == "toggle-training") {
            if (node->type == core::NodeType::CAMERA) {
                scene->setCameraTrainingEnabled(node_id, !node->training_enabled);
            } else if (node->type == core::NodeType::CAMERA_GROUP) {
                const auto counts = scene->getCameraTrainingCounts(node_id);
                toggleChildrenTraining(node_id, counts.enabled != counts.total);
            }
            tree_rebuild_needed_ = true;
            markStateDirty();
        }
    }

    std::pair<bool, bool> SceneGraphElement::checkboxState(const core::NodeId node_id) const {
        const auto it = node_snapshots_.find(node_id);
        if (it == node_snapshots_.end())
            return {false, false};
        if (it->second.children.empty())
            return {selected_ids_.contains(node_id), false};

        bool any_selected = false;
        bool all_selected = true;
        for (const core::NodeId child_id : it->second.children) {
            const auto [child_checked, child_mixed] = checkboxState(child_id);
            any_selected = any_selected || child_checked || child_mixed;
            all_selected = all_selected && child_checked && !child_mixed;
        }
        return {all_selected && any_selected, any_selected && !all_selected};
    }

    void SceneGraphElement::collectCheckboxSelectionIds(
        const core::NodeId node_id,
        std::vector<core::NodeId>& ids) const {
        const auto it = node_snapshots_.find(node_id);
        if (it == node_snapshots_.end())
            return;
        if (it->second.children.empty()) {
            ids.push_back(node_id);
            return;
        }
        for (const core::NodeId child_id : it->second.children)
            collectCheckboxSelectionIds(child_id, ids);
    }

    void SceneGraphElement::toggleCheckboxSelection(const core::NodeId node_id) {
        auto* scene_manager = services().sceneOrNull();
        if (!scene_manager)
            return;

        std::vector<core::NodeId> affected;
        collectCheckboxSelectionIds(node_id, affected);
        if (affected.empty())
            return;

        const bool all_checked = checkboxState(node_id).first;
        if (all_checked) {
            for (const core::NodeId id : affected)
                selected_ids_.erase(id);
        } else {
            selected_ids_.insert(affected.begin(), affected.end());
        }
        scene_manager->selectNodesById(
            std::vector<core::NodeId>(selected_ids_.begin(), selected_ids_.end()));
        click_anchor_id_ = node_id;
        keyboard_cursor_id_ = node_id;
        markStateDirty();
    }

    void SceneGraphElement::collectHierarchyIds(
        const core::NodeId node_id,
        std::unordered_set<core::NodeId>& ids) const {
        const auto it = node_snapshots_.find(node_id);
        if (it == node_snapshots_.end() || !ids.insert(node_id).second)
            return;
        for (const core::NodeId child_id : it->second.children)
            collectHierarchyIds(child_id, ids);
    }

    void SceneGraphElement::selectHierarchyFromSelection() {
        auto* scene_manager = services().sceneOrNull();
        if (!scene_manager || selected_ids_.empty())
            return;

        std::unordered_set<core::NodeId> hierarchy_ids;
        for (const core::NodeId id : selected_ids_)
            collectHierarchyIds(id, hierarchy_ids);
        selected_ids_ = std::move(hierarchy_ids);
        scene_manager->selectNodesById(
            std::vector<core::NodeId>(selected_ids_.begin(), selected_ids_.end()));
        markStateDirty();
    }

    void SceneGraphElement::selectHierarchy(const core::NodeId node_id) {
        auto* scene_manager = services().sceneOrNull();
        if (!scene_manager || !node_snapshots_.contains(node_id))
            return;

        std::unordered_set<core::NodeId> hierarchy_ids;
        collectHierarchyIds(node_id, hierarchy_ids);
        bool all_selected = true;
        for (const core::NodeId id : hierarchy_ids)
            all_selected = all_selected && selected_ids_.contains(id);
        if (all_selected) {
            for (const core::NodeId id : hierarchy_ids)
                selected_ids_.erase(id);
        } else {
            selected_ids_.insert(hierarchy_ids.begin(), hierarchy_ids.end());
        }
        scene_manager->selectNodesById(
            std::vector<core::NodeId>(selected_ids_.begin(), selected_ids_.end()));
        click_anchor_id_ = node_id;
        keyboard_cursor_id_ = node_id;
        markStateDirty();
    }

    bool SceneGraphElement::selectAllIfFocused() {
        if (!IsPseudoClassSet("focus"))
            return false;
        auto* scene_manager = services().sceneOrNull();
        if (!scene_manager || node_snapshots_.empty())
            return false;
        selected_ids_.clear();
        for (const auto& [id, _] : node_snapshots_)
            selected_ids_.insert(id);
        scene_manager->selectNodesById(
            std::vector<core::NodeId>(selected_ids_.begin(), selected_ids_.end()));
        markStateDirty();
        return true;
    }

    bool SceneGraphElement::toggleSelectedVisibilityIfFocused() {
        if (!IsPseudoClassSet("focus") || selected_ids_.empty())
            return false;
        const auto state = selectionActionState();
        setSelectedVisibility(!state.all_visible);
        return true;
    }

    bool SceneGraphElement::toggleSelectedTrainingIfFocused() {
        if (!IsPseudoClassSet("focus") || selected_ids_.empty())
            return false;
        const auto state = selectionActionState();
        if (!state.all_training_compatible)
            return false;
        setSelectedTrainingEnabled(!state.all_training_enabled);
        return true;
    }

    void SceneGraphElement::setSelectionMarkersVisible(const bool visible) {
        if (selection_markers_visible_ == visible)
            return;
        selection_markers_visible_ = visible;
        tree_rebuild_needed_ = true;
        markStateDirty();
    }

    void SceneGraphElement::handlePrimaryClick(const core::NodeId node_id) {
        auto* scene_manager = services().sceneOrNull();
        auto* scene = scene_manager ? &scene_manager->getScene() : nullptr;
        if (!scene || !node_snapshots_.contains(node_id))
            return;

        const bool ctrl = ctrlDown();
        const bool shift = shiftDown();
        keyboard_cursor_id_ = node_id;

        if (ctrl && shift) {
            selectHierarchy(node_id);
            return;
        }
        if (shift && click_anchor_id_ != core::NULL_NODE) {
            const auto range_ids = rangeSelectionIds(click_anchor_id_, node_id);
            selected_ids_.clear();
            selected_ids_.insert(range_ids.begin(), range_ids.end());
            scene_manager->selectNodesById(range_ids);
        } else if (ctrl) {
            if (selected_ids_.contains(node_id)) {
                selected_ids_.erase(node_id);
                scene_manager->selectNodesById(std::vector<core::NodeId>(selected_ids_.begin(), selected_ids_.end()));
            } else {
                scene_manager->addToSelection(node_id);
                selected_ids_.insert(node_id);
            }
            click_anchor_id_ = node_id;
        } else {
            if (selected_ids_.size() == 1 && selected_ids_.contains(node_id))
                return;
            scene_manager->selectNode(node_id);
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
        if (!selected_ids_.contains(node_id)) {
            scene_manager->selectNode(node_id);
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
            camera_preview_navigation_active_ = true;
            cmd::OpenCameraPreview{.cam_id = node->camera_uid}.emit();
            return true;
        }
        if (node->type == core::NodeType::KEYFRAME && node->keyframe) {
            camera_preview_navigation_active_ = false;
            cmd::SequencerGoToKeyframe{.keyframe_index = node->keyframe->keyframe_index}.emit();
            return true;
        }
        return false;
    }

    std::vector<core::NodeId> SceneGraphElement::rangeSelectionIds(const core::NodeId a,
                                                                   const core::NodeId b) const {
        const auto ita = flat_index_by_id_.find(a);
        const auto itb = flat_index_by_id_.find(b);
        if (ita == flat_index_by_id_.end() || itb == flat_index_by_id_.end()) {
            return node_snapshots_.contains(b) ? std::vector<core::NodeId>{b}
                                               : std::vector<core::NodeId>{};
        }

        const size_t lo = std::min(ita->second, itb->second);
        const size_t hi = std::max(ita->second, itb->second);
        std::vector<core::NodeId> ids;
        ids.reserve(hi - lo + 1);
        for (size_t i = lo; i <= hi; ++i) {
            const core::NodeId id = flat_rows_[i].id;
            if (node_snapshots_.contains(id))
                ids.push_back(id);
        }
        return ids;
    }

    core::NodeId SceneGraphElement::selectionCursor() const {
        if (keyboard_cursor_id_ != core::NULL_NODE &&
            flat_index_by_id_.contains(keyboard_cursor_id_))
            return keyboard_cursor_id_;
        if (click_anchor_id_ != core::NULL_NODE && selected_ids_.contains(click_anchor_id_))
            return click_anchor_id_;
        for (const FlatRow& row : flat_rows_) {
            if (selected_ids_.contains(row.id))
                return row.id;
        }
        return core::NULL_NODE;
    }

    bool SceneGraphElement::moveSelectionCursor(const int delta,
                                                const bool extend,
                                                const bool toggle) {
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
            const core::NodeId anchor = click_anchor_id_ != core::NULL_NODE
                                            ? click_anchor_id_
                                            : (current != core::NULL_NODE ? current : target);
            click_anchor_id_ = anchor;
            scene_manager->selectNodesById(rangeSelectionIds(anchor, target));
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
        } else if (toggle) {
            const core::NodeId toggled = current != core::NULL_NODE ? current : target;
            if (selected_ids_.contains(toggled))
                selected_ids_.erase(toggled);
            else
                selected_ids_.insert(toggled);
            scene_manager->selectNodesById(
                std::vector<core::NodeId>(selected_ids_.begin(), selected_ids_.end()));
        }
        keyboard_cursor_id_ = target;
        if (camera_preview_navigation_active_ &&
            PanelRegistry::instance().is_panel_enabled("lfs.image_preview")) {
            const auto* node = scene_manager->getScene().getNodeById(target);
            if (node && node->type == core::NodeType::CAMERA)
                cmd::OpenCameraPreview{.cam_id = node->camera_uid}.emit();
        } else {
            camera_preview_navigation_active_ = false;
        }
        scrollNodeIntoView(target);
        markStateDirty();
        return true;
    }

    bool SceneGraphElement::isTextInputTarget(Rml::Element* target) const {
        return target && target->GetTagName() == "input";
    }

    bool SceneGraphElement::isModelsHeaderTarget(Rml::Element* target) const {
        if (!target)
            return false;
        if (target == header_el_ || target == header_arrow_el_ || target == header_label_el_)
            return true;
        return target->GetAttribute<Rml::String>("data-role", "") == "models-header";
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

    void SceneGraphElement::toggleModelsSection() {}

    bool SceneGraphElement::isValidDropContainer(const core::NodeId container_id) const {
        if (drag_source_id_ == core::NULL_NODE)
            return false;
        if (container_id != core::NULL_NODE) {
            const auto it = node_snapshots_.find(container_id);
            if (it == node_snapshots_.end() || it->second.type != core::NodeType::GROUP)
                return false;
        }
        core::NodeId walk = container_id;
        while (walk != core::NULL_NODE) {
            if (walk == drag_source_id_)
                return false;
            const auto it = node_snapshots_.find(walk);
            if (it == node_snapshots_.end())
                break;
            walk = it->second.parent_id;
        }
        return true;
    }

    int SceneGraphElement::siblingIndexOf(const core::NodeId node_id) const {
        const auto snap_it = node_snapshots_.find(node_id);
        if (snap_it == node_snapshots_.end())
            return 0;
        if (snap_it->second.parent_id == core::NULL_NODE) {
            const auto it = std::find(root_ids_.begin(), root_ids_.end(), node_id);
            return it != root_ids_.end() ? static_cast<int>(std::distance(root_ids_.begin(), it)) : 0;
        }
        const auto parent_it = node_snapshots_.find(snap_it->second.parent_id);
        if (parent_it == node_snapshots_.end())
            return 0;
        const auto& siblings = parent_it->second.children;
        const auto it = std::find(siblings.begin(), siblings.end(), node_id);
        return it != siblings.end() ? static_cast<int>(std::distance(siblings.begin(), it)) : 0;
    }

    void SceneGraphElement::clearDropState() {
        drop_parent_id_ = core::NULL_NODE;
        drop_index_ = -1;
        drop_valid_ = false;
        if (drop_into_group_id_ != core::NULL_NODE) {
            drop_into_group_id_ = core::NULL_NODE;
            markStateDirty();
        }
        if (insertion_line_)
            setCachedProperty(insertion_line_, "display", "none");
    }

    void SceneGraphElement::updateDropTarget(RowSlot* const hovered_slot,
                                             const core::NodeId hovered_id,
                                             const float mouse_y) {
        if (drag_source_id_ == core::NULL_NODE || hovered_id == drag_source_id_) {
            clearDropState();
            return;
        }

        core::NodeId parent = core::NULL_NODE;
        int index = -1;
        core::NodeId into_group = core::NULL_NODE;
        bool show_line = false;
        int line_top_dp = 0;
        int line_left_dp = 4;

        if (hovered_id == core::NULL_NODE || !hovered_slot) {
            show_line = true;
            line_top_dp = static_cast<int>(flat_rows_.size()) * kRowHeightDpInt;
        } else {
            const auto snap_it = node_snapshots_.find(hovered_id);
            const auto flat_it = flat_index_by_id_.find(hovered_id);
            if (snap_it == node_snapshots_.end() || flat_it == flat_index_by_id_.end()) {
                clearDropState();
                return;
            }
            const size_t fidx = flat_it->second;
            const int depth = fidx < flat_rows_.size() ? flat_rows_[fidx].depth : 0;
            const float row_h = kRowHeightDp * currentDpRatio(this);
            const float top = hovered_slot->root->GetAbsoluteOffset(Rml::BoxArea::Border).y;
            const float rel = row_h > 0.0f ? (mouse_y - top) / row_h : 0.5f;
            const bool is_group = snap_it->second.type == core::NodeType::GROUP;

            if (is_group && rel > 0.2f && rel < 0.8f) {
                into_group = hovered_id;
                parent = hovered_id;
            } else {
                const bool after = rel >= 0.5f;
                parent = snap_it->second.parent_id;
                index = siblingIndexOf(hovered_id) + (after ? 1 : 0);
                show_line = true;
                line_top_dp = static_cast<int>(fidx) * kRowHeightDpInt +
                              (after ? kRowHeightDpInt : 0);
                line_left_dp = 21 + depth * 16;
            }
        }

        const core::NodeId container = into_group != core::NULL_NODE ? into_group : parent;
        if (!isValidDropContainer(container)) {
            clearDropState();
            return;
        }

        drop_parent_id_ = parent;
        drop_index_ = index;
        drop_valid_ = true;

        if (drop_into_group_id_ != into_group) {
            drop_into_group_id_ = into_group;
            markStateDirty();
        }

        if (insertion_line_) {
            if (show_line) {
                setCachedProperty(insertion_line_, "display", "block");
                setCachedProperty(insertion_line_, "top", formatDp(line_top_dp - 1));
                setCachedProperty(insertion_line_, "left", formatDp(line_left_dp));
            } else {
                setCachedProperty(insertion_line_, "display", "none");
            }
        }
    }

    void SceneGraphElement::commitDrop() {
        if (drop_valid_ && drag_source_id_ != core::NULL_NODE) {
            cmd::MoveNodeById{
                .node_id = static_cast<int32_t>(drag_source_id_),
                .new_parent_id = static_cast<int32_t>(drop_parent_id_),
                .index = drop_index_}
                .emit();
        }
        drag_source_id_ = core::NULL_NODE;
        clearDropState();
        markStateDirty();
    }

    void SceneGraphElement::showDragGhost(const core::NodeId node_id, const float mouse_x, const float mouse_y) {
        if (!drag_ghost_ || node_id == core::NULL_NODE)
            return;
        const auto it = node_snapshots_.find(node_id);
        if (it == node_snapshots_.end())
            return;
        drag_ghost_->SetInnerRML(encode(it->second.name));
        drag_ghost_->SetProperty("display", "block");
        moveDragGhost(mouse_x, mouse_y);
    }

    void SceneGraphElement::moveDragGhost(const float mouse_x, const float mouse_y) {
        if (!drag_ghost_ || !content_el_)
            return;
        const Rml::Vector2f base = content_el_->GetAbsoluteOffset(Rml::BoxArea::Border);
        drag_ghost_->SetProperty("left", std::format("{:.0f}px", mouse_x - base.x + 14.0f));
        drag_ghost_->SetProperty("top", std::format("{:.0f}px", mouse_y - base.y + 10.0f));
    }

    void SceneGraphElement::hideDragGhost() {
        if (drag_ghost_)
            drag_ghost_->SetProperty("display", "none");
    }

    SceneGraphElement::SelectionActionState SceneGraphElement::selectionActionState() const {
        SelectionActionState state;
        state.count = selected_ids_.size();
        if (selected_ids_.empty())
            return state;

        state.all_visible = true;
        state.all_training_compatible = true;
        state.all_training_enabled = true;
        state.all_delete_enabled = true;
        for (const core::NodeId id : selected_ids_) {
            const auto it = node_snapshots_.find(id);
            if (it == node_snapshots_.end()) {
                state.all_visible = false;
                state.all_training_compatible = false;
                state.all_training_enabled = false;
                state.all_delete_enabled = false;
                continue;
            }
            const NodeSnapshot& node = it->second;
            state.all_visible = state.all_visible && node.visible;
            state.any_visible = state.any_visible || node.visible;
            state.all_training_compatible = state.all_training_compatible && node.can_toggle_training;
            state.all_training_enabled = state.all_training_enabled &&
                                         node.can_toggle_training && node.training_enabled;
            state.any_training_enabled = state.any_training_enabled ||
                                         (node.can_toggle_training &&
                                          (node.training_enabled || node.training_mixed));
            state.all_delete_enabled = state.all_delete_enabled && node.delete_enabled;
        }
        return state;
    }

    void SceneGraphElement::setSelectedVisibility(const bool visible) {
        auto* scene_manager = services().sceneOrNull();
        if (!scene_manager)
            return;

        bool update_frustums = false;
        for (const core::NodeId id : selected_ids_) {
            const auto snapshot_it = node_snapshots_.find(id);
            if (snapshot_it == node_snapshots_.end())
                continue;
            if (snapshot_it->second.camera_frustum_container) {
                update_frustums = true;
            } else if (snapshot_it->second.visible != visible) {
                scene_manager->setNodeVisibility(id, visible);
            }
        }
        if (update_frustums) {
            if (auto* rendering = services().renderingOrNull()) {
                auto settings = rendering->getSettings();
                settings.show_camera_frustums = visible;
                rendering->updateSettings(settings);
            }
        }
        tree_rebuild_needed_ = true;
        markStateDirty();
    }

    void SceneGraphElement::setSelectedTrainingEnabled(const bool enabled) {
        if (!selectionActionState().all_training_compatible)
            return;
        toggleSelectedTraining(enabled);
        tree_rebuild_needed_ = true;
        markStateDirty();
    }

    void SceneGraphElement::requestDeleteSelection() {
        if (!selectionActionState().all_delete_enabled)
            return;
        deleteSelectedNodes();
    }

    void SceneGraphElement::clearSelectedNodes() {
        if (selected_ids_.empty())
            return;
        if (auto* scene_manager = services().sceneOrNull())
            scene_manager->clearSelection();
        selected_ids_.clear();
        click_anchor_id_ = core::NULL_NODE;
        markStateDirty();
    }

    void SceneGraphElement::requestDeleteNodes(const std::vector<core::NodeId>& node_ids) {
        auto* gui = services().guiOrNull();
        auto* scene_manager = services().sceneOrNull();
        if (!gui || !scene_manager || node_ids.empty())
            return;

        const auto& scene = scene_manager->getScene();
        std::vector<core::Uuid> node_uuids;
        node_uuids.reserve(node_ids.size());
        std::string single_name;
        for (const core::NodeId node_id : node_ids) {
            const auto snapshot_it = node_snapshots_.find(node_id);
            const auto* node = scene.getNodeById(node_id);
            if (!node || snapshot_it == node_snapshots_.end() || !snapshot_it->second.delete_enabled) {
                LOG_WARN("Scene node deletion request aborted: the complete selection is no longer removable");
                return;
            }
            if (const auto result = scene_manager->canRemoveNode(node_id); !result) {
                LOG_WARN("Scene node deletion request aborted: {}", result.error());
                return;
            }
            node_uuids.push_back(node->uuid);
            if (single_name.empty())
                single_name = node->name;
        }
        if (node_uuids.empty())
            return;

        const std::string delete_button = tr(string_keys::Scene::DELETE_ITEM);
        lfs::core::ModalRequest request;
        request.title = tr(string_keys::Scene::DELETE_CONFIRMATION_TITLE);
        request.style = lfs::core::ModalStyle::Warning;
        request.width_dp = 440;
        const std::string message = node_uuids.size() == 1
                                        ? LOCF(string_keys::Scene::DELETE_CONFIRMATION_SINGLE, single_name)
                                        : LOCF(string_keys::Scene::DELETE_CONFIRMATION_MULTIPLE, node_uuids.size());
        request.body_rml = encode(message);
        request.buttons = {
            {tr(string_keys::Common::CANCEL), "secondary"},
            {delete_button, "warning"},
        };
        request.on_result = [node_uuids = std::move(node_uuids), delete_button](
                                const lfs::core::ModalResult& result) {
            if (result.button_label != delete_button)
                return;
            auto* live_manager = services().sceneOrNull();
            if (!live_manager)
                return;
            auto& live_scene = live_manager->getScene();
            std::vector<core::NodeId> live_ids;
            live_ids.reserve(node_uuids.size());
            for (const core::Uuid& uuid : node_uuids) {
                const core::NodeId live_id = live_scene.getNodeIdByUuid(uuid);
                const auto* live_node = live_scene.getNodeById(live_id);
                if (!live_node) {
                    LOG_WARN("Scene node deletion aborted after confirmation: a selected node no longer exists");
                    return;
                }
                if (const auto can_remove = live_manager->canRemoveNode(live_id); !can_remove) {
                    LOG_WARN("Scene node deletion aborted after confirmation: {}", can_remove.error());
                    return;
                }
                live_ids.push_back(live_id);
            }
            if (live_ids.size() != node_uuids.size())
                return;
            if (const auto delete_result =
                    live_manager->removeNodesByIdsWithResult(live_ids, /*keep_children=*/false);
                !delete_result) {
                LOG_WARN("Scene node deletion failed after confirmation: {}", delete_result.error());
            }
        };
        gui->enqueueModal(std::move(request));
    }

    void SceneGraphElement::deleteSelectedNodes() {
        auto* scene_manager = services().sceneOrNull();
        if (!scene_manager)
            return;

        if (selected_ids_.empty())
            return;
        const std::vector<core::NodeId> ids(selected_ids_.begin(), selected_ids_.end());
        requestDeleteNodes(ids);
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
                scene->setCameraTrainingEnabled(child_id, enabled);
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
                scene->setCameraTrainingEnabled(id, enabled);
            } else if (node->type == core::NodeType::CAMERA_GROUP) {
                for (const core::NodeId child_id : node->children) {
                    const auto* child = scene->getNodeById(child_id);
                    if (child && child->type == core::NodeType::CAMERA)
                        scene->setCameraTrainingEnabled(child_id, enabled);
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

        std::vector<ContextMenuItem> items;
        const auto* node = scene->getNodeById(node_id);
        if (!node)
            return;

        if (selected_ids_.size() > 1) {
            items.push_back(makeAction(tr("scene.select_hierarchy"),
                                       prefixedAction("select_hierarchy")));
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
                items.push_back(makeAction(tr(string_keys::Scene::ENABLE_ALL_TRAINING),
                                           prefixedAction("enable_all_selected_train")));
                items.push_back(makeAction(tr(string_keys::Scene::DISABLE_ALL_TRAINING),
                                           prefixedAction("disable_all_selected_train")));
            }

            const auto selection_state = selectionActionState();
            if (selection_state.all_delete_enabled) {
                items.push_back(makeAction(
                    std::format("{} ({})", tr(string_keys::Scene::DELETE_ITEM), selection_state.count),
                    prefixedAction("delete_selected"),
                    !items.empty()));
            }
        } else {
            if (node_snapshots_.at(node_id).has_children) {
                items.push_back(makeAction(tr("scene.select_hierarchy"),
                                           prefixedAction("select_hierarchy")));
            }
            switch (node->type) {
            case core::NodeType::CAMERA:
                items.push_back(makeAction(
                    tr(string_keys::Scene::GO_TO_CAMERA_VIEW),
                    prefixedAction(std::format("go_to_camera:{}", node->camera_uid))));
                items.push_back(makeAction(
                    tr(string_keys::Scene::GO_TO_IMAGE),
                    prefixedAction(std::format("go_to_image:{}", node->camera_uid))));
                items.push_back(makeAction(
                    tr(string_keys::Scene::OPEN_IN_GT_COMPARE),
                    prefixedAction(std::format("open_in_gt_compare:{}", node->camera_uid))));
                if (!node->image_path.empty()) {
                    items.push_back(makeAction(
                        tr(string_keys::Scene::SHOW_IN_FILE_MANAGER),
                        prefixedAction(std::format("show_in_file_manager:{}", node_id))));
                }
                items.push_back(makeAction(
                    node->training_enabled ? tr(string_keys::Scene::DISABLE_FOR_TRAINING)
                                           : tr(string_keys::Scene::ENABLE_FOR_TRAINING),
                    prefixedAction(std::format("{}:{}", node->training_enabled ? "disable_train" : "enable_train", node_id)),
                    true));
                break;
            case core::NodeType::KEYFRAME:
                if (node->keyframe) {
                    items.push_back(makeAction(
                        tr(string_keys::Scene::GO_TO_KEYFRAME),
                        prefixedAction(std::format("go_to_kf:{}", node->keyframe->keyframe_index))));
                    items.push_back(makeAction(
                        tr(string_keys::Scene::UPDATE_KEYFRAME),
                        prefixedAction(std::format("update_kf:{}", node->keyframe->keyframe_index))));
                    items.push_back(makeAction(
                        tr("scene.select_in_timeline"),
                        prefixedAction(std::format("select_kf:{}", node->keyframe->keyframe_index))));
                    items.push_back(makeLabel(tr(string_keys::Scene::KEYFRAME_EASING), true));
                    static constexpr std::array easing_labels{
                        "scene.keyframe_easing.linear",
                        "scene.keyframe_easing.ease_in",
                        "scene.keyframe_easing.ease_out",
                        "scene.keyframe_easing.ease_in_out",
                    };
                    for (size_t i = 0; i < easing_labels.size(); ++i) {
                        items.push_back(makeAction(
                            tr(easing_labels[i]),
                            prefixedAction(std::format("set_easing:{}:{}", node->keyframe->keyframe_index, i)),
                            false, true, node->keyframe->easing == i));
                    }
                    if (node->keyframe->keyframe_index > 0) {
                        items.push_back(makeAction(
                            tr(string_keys::Scene::DELETE_ITEM),
                            prefixedAction(std::format("delete_kf:{}", node->keyframe->keyframe_index)),
                            true));
                    }
                }
                break;
            case core::NodeType::KEYFRAME_GROUP:
                items.push_back(makeAction(
                    tr(string_keys::Scene::ADD_KEYFRAME_SCENE),
                    prefixedAction("add_kf")));
                break;
            case core::NodeType::CAMERA_GROUP:
                items.push_back(makeAction(
                    tr(string_keys::Scene::ENABLE_ALL_TRAINING),
                    prefixedAction(std::format("enable_all_train:{}", node_id))));
                items.push_back(makeAction(
                    tr(string_keys::Scene::DISABLE_ALL_TRAINING),
                    prefixedAction(std::format("disable_all_train:{}", node_id))));
                break;
            case core::NodeType::DATASET:
                if (colmapSparseSourcePath(*scene_manager)) {
                    items.push_back(makeAction(
                        tr("export.format.colmap_sparse") + "...",
                        prefixedAction("export_colmap")));
                }
                break;
            default:
                break;
            }

            if (node->type == core::NodeType::GROUP) {
                items.push_back(makeAction(
                    tr("scene.add_group_ellipsis"),
                    prefixedAction(std::format("add_group:{}", node_id)),
                    !items.empty()));
                items.push_back(makeAction(
                    tr("scene.merge_to_single_ply"),
                    prefixedAction(std::format("merge_group:{}", node_id))));
            }

            if (node->type == core::NodeType::SPLAT || node->type == core::NodeType::POINTCLOUD) {
                items.push_back(makeAction(
                    tr("scene.save_to_disk"),
                    prefixedAction(std::format("save_node:{}", node_id)),
                    !items.empty()));
            }

            // Add Save Asset for asset-compatible node types
            if (canSaveAsAsset(*node)) {
                items.push_back(makeAction(
                    tr(string_keys::Scene::SAVE_ASSET),
                    prefixedAction(std::format("save_asset:{}", node_id)),
                    !items.empty()));
            }

            if (node_snapshots_.at(node_id).rename_enabled) {
                items.push_back(makeAction(
                    tr(string_keys::Scene::RENAME),
                    prefixedAction(std::format("rename:{}", node_id)),
                    !items.empty()));
            }

            if (node->type != core::NodeType::CAMERA &&
                node->type != core::NodeType::CROPBOX &&
                node->type != core::NodeType::ELLIPSOID) {
                items.push_back(makeAction(
                    tr("scene.duplicate"),
                    prefixedAction(std::format("duplicate:{}", node_id))));
            }

            if (node_snapshots_.at(node_id).draggable) {
                std::vector<std::pair<core::NodeId, std::string>> groups;
                for (const auto& [candidate_id, snapshot] : node_snapshots_) {
                    if (candidate_id != node_id && snapshot.type == core::NodeType::GROUP)
                        groups.emplace_back(candidate_id, snapshot.name);
                }
                if (!groups.empty()) {
                    items.push_back(makeLabel(tr("scene.move_to"), true));
                    items.push_back(makeAction(
                        tr("scene.move_to_root"),
                        prefixedAction(std::format("reparent:{}:{}", node_id, core::NULL_NODE)),
                        false, true));
                    std::ranges::sort(groups, [](const auto& a, const auto& b) {
                        return a.second < b.second;
                    });
                    for (const auto& [group_id, group_name] : groups) {
                        items.push_back(makeAction(
                            group_name,
                            prefixedAction(std::format("reparent:{}:{}", node_id, group_id)),
                            false, true));
                    }
                }
            }

            if (node_snapshots_.at(node_id).delete_enabled) {
                items.push_back(makeAction(
                    tr(string_keys::Scene::DELETE_ITEM),
                    prefixedAction(std::format("delete:{}", node_id)),
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

    void SceneGraphElement::showModelsHeaderContextMenu(const float mouse_x, const float mouse_y) {
        auto* gui = services().guiOrNull();
        if (!gui)
            return;

        std::vector<ContextMenuItem> items;
        items.push_back(makeAction(tr(string_keys::Scene::ADD_PLY), prefixedAction("add_ply_root")));
        items.push_back(makeAction(tr(string_keys::Scene::ADD_GROUP_ELLIPSIS), prefixedAction("add_group_root")));

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
        if (kind == "select_hierarchy") {
            selectHierarchyFromSelection();
            return;
        }
        if (kind == "go_to_camera" && parts.size() >= 2) {
            cmd::GoToCamView{.cam_id = std::stoi(parts[1])}.emit();
            Blur();
        } else if (kind == "go_to_image" && parts.size() >= 2) {
            cmd::OpenCameraPreview{.cam_id = std::stoi(parts[1])}.emit();
        } else if (kind == "open_in_gt_compare" && parts.size() >= 2) {
            const int cam_uid = std::stoi(parts[1]);
            cmd::GoToCamView{.cam_id = cam_uid}.emit();
            auto* const rm = services().renderingOrNull();
            if (!rm || !rm->isGTComparisonActive())
                cmd::ToggleGTComparison{}.emit();
            Blur();
        } else if (kind == "show_in_file_manager" && parts.size() >= 2) {
            core::NodeId node_id = core::NULL_NODE;
            if (parseNodeId(parts[1], node_id)) {
                if (const auto* node = scene->getNodeById(node_id);
                    node && !node->image_path.empty()) {
                    if (!lfs::core::reveal_in_file_manager(
                            lfs::core::utf8_to_path(node->image_path))) {
                        LOG_WARN("Failed to reveal image in file manager: {}", node->image_path);
                    }
                }
            }
        } else if ((kind == "enable_train" || kind == "disable_train") && parts.size() >= 2) {
            core::NodeId node_id = core::NULL_NODE;
            if (parseNodeId(parts[1], node_id))
                scene->setCameraTrainingEnabled(node_id, kind == "enable_train");
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
                const auto snapshot_it = node_snapshots_.find(node_id);
                if (snapshot_it == node_snapshots_.end() || !snapshot_it->second.delete_enabled)
                    return;
                requestDeleteNodes({node_id});
            }
        } else if (kind == "rename" && parts.size() >= 2) {
            core::NodeId node_id = core::NULL_NODE;
            if (parseNodeId(parts[1], node_id)) {
                const auto snapshot_it = node_snapshots_.find(node_id);
                if (snapshot_it == node_snapshots_.end() || !snapshot_it->second.rename_enabled)
                    return;
                beginRename(node_id);
            }
        } else if (kind == "duplicate" && parts.size() >= 2) {
            core::NodeId node_id = core::NULL_NODE;
            if (parseNodeId(parts[1], node_id))
                cmd::DuplicateNodeById{.node_id = static_cast<int32_t>(node_id)}.emit();
        } else if (kind == "add_group" && parts.size() >= 2) {
            core::NodeId parent_id = core::NULL_NODE;
            if (parseNodeId(parts[1], parent_id))
                cmd::AddGroupByParentId{
                    .name = tr("scene.new_group_name"),
                    .parent_id = static_cast<int32_t>(parent_id)}
                    .emit();
        } else if (kind == "merge_group" && parts.size() >= 2) {
            core::NodeId node_id = core::NULL_NODE;
            if (parseNodeId(parts[1], node_id))
                cmd::MergeGroupById{.node_id = static_cast<int32_t>(node_id)}.emit();
        } else if (kind == "add_ply_root") {
            const auto path = OpenPointCloudFileDialog();
            if (!path.empty()) {
                cmd::LoadFile{.path = path, .is_dataset = false}.emit();
            }
        } else if (kind == "add_group_root") {
            cmd::AddGroupByParentId{
                .name = tr("scene.new_group_name"),
                .parent_id = static_cast<int32_t>(core::NULL_NODE)}
                .emit();
        } else if (kind == "export_colmap") {
            auto* gui = services().guiOrNull();
            if (!gui)
                return;

            auto path_result = colmapSparseSourcePath(*scene_manager);
            if (!path_result) {
                const std::string error = path_result.error().format();
                LOG_ERROR("COLMAP export failed: {}", error);
                state::ExportFailed{.error = error}.emit();
                return;
            }

            const auto selected_path = PickColmapSparseFolderDialog(*path_result);
            if (selected_path.empty()) {
                return;
            }

            // The folder picker result is the export destination. Do not rewrite a selected
            // sparse root to sparse/0; overwrite checks and writes must hit the same folder.
            const auto output_path = selected_path;
            const std::string output_path_text = lfs::core::path_to_utf8(output_path);

            if (!colmapSparseDataExists(output_path)) {
                gui->asyncTasks().performExport(core::ExportFormat::COLMAP, output_path, {}, 0);
                return;
            }

            LOG_INFO("Confirming COLMAP sparse overwrite folder: {}", output_path_text);

            const std::string export_button = LOC("export.overwrite");
            const std::string written_files =
                LOCF("export_dialog.colmap_writes_sparse", colmapExportExtension(*path_result));
            lfs::core::ModalRequest request;
            request.title = LOC(lichtfeld::Strings::Window::EXPORT);
            request.style = lfs::core::ModalStyle::Warning;
            request.width_dp = 560;
            request.body_rml =
                std::format("<div>{}</div>", LOC(lichtfeld::Strings::Runtime::COLMAP_OVERWRITE_MESSAGE)) +
                "<div class=\"content-row\" style=\"margin-top: 8dp;\">" +
                std::format("<span class=\"dim-text\">{} </span>", LOC(lichtfeld::Strings::Runtime::FOLDER_LABEL)) +
                encode(output_path_text) +
                "</div>"
                "<div class=\"warning-text\" style=\"margin-top: 8dp;\">" +
                encode(written_files) +
                "</div>";
            request.buttons = {
                {LOC(lichtfeld::Strings::Common::CANCEL), "secondary"},
                {export_button, "warning"},
            };
            request.on_result = [gui, output_path, export_button](const lfs::core::ModalResult& result) {
                if (result.button_label == export_button) {
                    gui->asyncTasks().performExport(core::ExportFormat::COLMAP, output_path, {}, 0);
                }
            };
            gui->enqueueModal(std::move(request));
        } else if (kind == "save_node" && parts.size() >= 2) {
            core::NodeId node_id = core::NULL_NODE;
            if (!parseNodeId(parts[1], node_id))
                return;
            if (scene->getNodeById(node_id))
                saveNodeToDisk(*scene, node_id);
        } else if (kind == "enable_all_selected_train") {
            toggleSelectedTraining(true);
        } else if (kind == "disable_all_selected_train") {
            toggleSelectedTraining(false);
        } else if (kind == "delete_selected") {
            requestDeleteSelection();
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
            cmd::ReparentNodeById{
                .node_id = static_cast<int32_t>(node_id),
                .new_parent_id = static_cast<int32_t>(parent_id)}
                .emit();
        } else if (kind == "save_asset" && parts.size() >= 2) {
            core::NodeId node_id = core::NULL_NODE;
            if (parseNodeId(parts[1], node_id))
                cmd::SaveAssetById{.node_id = static_cast<int32_t>(node_id)}.emit();
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

            if (isModelsHeaderTarget(target)) {
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
            } else if (target == this || target == content_el_) {
                clearSelectedNodes();
                event.StopPropagation();
            }
        } else if (type == "dblclick") {
            if (isTextInputTarget(target))
                return;
            for (auto* current = target; current && current != this; current = current->GetParentNode()) {
                if (!current->GetAttribute<Rml::String>("data-action", "").empty()) {
                    event.StopPropagation();
                    return;
                }
            }
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
                } else if (target == this || target == content_el_ || isModelsHeaderTarget(target)) {
                    showModelsHeaderContextMenu(event.GetParameter("mouse_x", 0.0f),
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
                if (moveSelectionCursor(-1, shiftDown(), ctrlDown() && !shiftDown()))
                    event.StopPropagation();
                break;
            case Rml::Input::KI_DOWN:
                if (moveSelectionCursor(1, shiftDown(), ctrlDown() && !shiftDown()))
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
                    requestDeleteSelection();
                    event.StopPropagation();
                }
                break;
            case Rml::Input::KI_ESCAPE:
                if (rename_node_id_ != core::NULL_NODE) {
                    cancelRename();
                    event.StopPropagation();
                } else if (!selected_ids_.empty()) {
                    clearSelectedNodes();
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
        }
    }

    void SceneGraphElement::DragListener::ProcessEvent(Rml::Event& event) {
        if (owner)
            owner->handleDragEvent(event);
    }

    void SceneGraphElement::handleDragEvent(Rml::Event& event) {
        ensureDom();

        const std::string type = event.GetType();
        Rml::Element* const target = event.GetTargetElement();
        if (!target)
            return;

        if (type == "dragstart") {
            drag_source_id_ = nodeIdFromTarget(target);
            LOG_DEBUG("[scene-graph] dragstart source={}", drag_source_id_);
            clearDropState();
            showDragGhost(drag_source_id_,
                          event.GetParameter("mouse_x", 0.0f),
                          event.GetParameter("mouse_y", 0.0f));
            markStateDirty();
        } else if (type == "dragend") {
            drag_source_id_ = core::NULL_NODE;
            clearDropState();
            hideDragGhost();
            markStateDirty();
        } else if (type == "dragover") {
            if (drag_source_id_ != core::NULL_NODE) {
                moveDragGhost(event.GetParameter("mouse_x", 0.0f),
                              event.GetParameter("mouse_y", 0.0f));
                RowSlot* const slot = rowSlotFromTarget(target);
                const core::NodeId hovered = slot ? slot->bound_id : core::NULL_NODE;
                updateDropTarget(slot, hovered, event.GetParameter("mouse_y", 0.0f));
            }
        } else if (type == "dragdrop") {
            if (services().sceneOrNull() && drag_source_id_ != core::NULL_NODE) {
                RowSlot* const slot = rowSlotFromTarget(target);
                const core::NodeId hovered = slot ? slot->bound_id : core::NULL_NODE;
                updateDropTarget(slot, hovered, event.GetParameter("mouse_y", 0.0f));
                LOG_DEBUG("[scene-graph] dragdrop source={} hovered={} valid={} parent={} index={}",
                          drag_source_id_, hovered, drop_valid_, drop_parent_id_, drop_index_);
                commitDrop();
            } else {
                drag_source_id_ = core::NULL_NODE;
                clearDropState();
            }
            hideDragGhost();
        }
    }

} // namespace lfs::vis::gui
