/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/EventListener.h>
#include <core/scene.hpp>

#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace lfs::vis {
    class SceneManager;
}

namespace lfs::vis::gui {

    struct PanelDrawContext;

    class SceneGraphElement : public Rml::Element {
    public:
        explicit SceneGraphElement(const Rml::String& tag);

        void setPanelScreenOffset(float x, float y);
        void setFilterText(std::string_view text);
        [[nodiscard]] bool syncFromScene(const PanelDrawContext& ctx);
        [[nodiscard]] bool executeContextMenuAction(std::string_view action);

        [[nodiscard]] size_t rootCount() const { return root_count_; }
        [[nodiscard]] size_t nodeCount() const { return node_snapshots_.size(); }
        [[nodiscard]] size_t selectedCount() const { return selected_ids_.size(); }
        [[nodiscard]] const std::string& filterText() const { return filter_text_; }
        [[nodiscard]] bool hasNodes() const { return scene_has_nodes_; }

        [[nodiscard]] static bool ownsContextMenuAction(std::string_view action);

    protected:
        void OnRender() override;
        void OnResize() override;
        bool GetIntrinsicDimensions(Rml::Vector2f& dimensions, float& ratio) override;
        void ProcessDefaultAction(Rml::Event& event) override;

    private:
        struct RenameInputListener : Rml::EventListener {
            SceneGraphElement* owner = nullptr;
            void ProcessEvent(Rml::Event& event) override;
        };

        static constexpr size_t kUnsetVisibleRange = std::numeric_limits<size_t>::max();

        struct NodeSnapshot {
            core::NodeId id = core::NULL_NODE;
            core::NodeId parent_id = core::NULL_NODE;
            std::vector<core::NodeId> children;
            core::NodeType type = core::NodeType::GROUP;
            std::string name;
            bool visible = true;
            bool has_children = false;
            bool training_enabled = true;
            std::string label;
            bool draggable = false;
            bool has_mask = false;
            bool deletable = false;
        };

        struct FlatRow {
            core::NodeId id = core::NULL_NODE;
            core::NodeType type = core::NodeType::GROUP;
            int depth = 0;
            bool visible = true;
            bool has_children = false;
            bool collapsed = false;
            bool draggable = false;
            bool training_enabled = true;
            std::string name;
            std::string label;
            std::string node_id_text;
            std::string encoded_label;
            std::string padding_left_dp;
            bool has_mask = false;
            bool deletable = false;
        };

        struct RowSlot {
            Rml::Element* root = nullptr;
            Rml::Element* content = nullptr;
            Rml::Element* vis_icon = nullptr;
            Rml::Element* delete_icon = nullptr;
            Rml::Element* type_icon = nullptr;
            Rml::Element* unicode_icon = nullptr;
            Rml::Element* mask_icon = nullptr;
            Rml::Element* expand_toggle = nullptr;
            Rml::Element* leaf_spacer = nullptr;
            Rml::Element* rename_input = nullptr;
            Rml::Element* node_name = nullptr;
            core::NodeId bound_id = core::NULL_NODE;
            bool visible = false;
        };

        void ensureDom();
        void ensureRowPool(size_t count);
        void clear();
        void markStateDirty();
        void captureRenameBuffer();
        void rebuildFlatRows(const core::Scene& scene);
        bool syncSelectionFromScene(const core::Scene& scene, lfs::vis::SceneManager* scene_manager);
        void captureSceneSnapshot(const core::Scene& scene,
                                  std::unordered_map<core::NodeId, NodeSnapshot>& snapshots,
                                  std::vector<core::NodeId>& root_ids);
        bool syncTrainingTopologyLabel(const core::Scene& scene, bool update_cached_rows);
        void appendSnapshotRows(core::NodeId node_id, int depth, std::vector<FlatRow>& rows,
                                const std::string& filter_text_lower) const;
        void appendVisibleSubtreeRows(core::NodeId node_id, int depth,
                                      std::vector<FlatRow>& rows) const;
        void rebuildIndex();
        void syncVisibleRows(bool force);
        void bindRow(RowSlot& slot, const FlatRow& row, size_t absolute_index);
        void hideRow(RowSlot& slot);
        void updateHeader();
        void updateContentHeight();
        void scrollNodeIntoView(core::NodeId node_id);
        void scrollNodeIntoViewCentered(core::NodeId node_id);
        void focusTree();
        void beginRename(core::NodeId node_id);
        void confirmRename();
        void cancelRename();
        void handleInlineAction(const std::string& action, core::NodeId node_id);
        void handlePrimaryClick(core::NodeId node_id);
        void handleSecondaryClick(core::NodeId node_id, float mouse_x, float mouse_y);
        bool activateNode(core::NodeId node_id);
        bool moveSelection(int delta, bool extend);
        std::vector<std::string> rangeSelectionNames(core::NodeId a, core::NodeId b) const;
        core::NodeId selectionCursor() const;
        bool isTextInputTarget(Rml::Element* target) const;
        RowSlot* rowSlotFromTarget(Rml::Element* target);
        const RowSlot* rowSlotFromTarget(Rml::Element* target) const;
        core::NodeId nodeIdFromTarget(Rml::Element* target) const;
        void toggleExpand(core::NodeId node_id);
        void toggleModelsSection();
        bool setDropTarget(core::NodeId node_id);
        void showContextMenu(core::NodeId node_id, float mouse_x, float mouse_y);
        void showModelsHeaderContextMenu(float mouse_x, float mouse_y);
        std::vector<std::string> deletableSelectedNodeNames() const;
        void deleteSelectedNodes();
        void toggleChildrenTraining(core::NodeId group_id, bool enabled);
        void toggleSelectedTraining(bool enabled);
        void executeAction(const std::string& action);

        Rml::Element* content_el_ = nullptr;
        Rml::Element* header_el_ = nullptr;
        Rml::Element* header_arrow_el_ = nullptr;
        Rml::Element* header_label_el_ = nullptr;

        std::vector<RowSlot> row_slots_;
        std::unordered_map<core::NodeId, NodeSnapshot> node_snapshots_;
        std::vector<core::NodeId> root_ids_;
        std::vector<FlatRow> flat_rows_;
        std::unordered_map<core::NodeId, size_t> flat_index_by_id_;
        std::unordered_set<core::NodeId> collapsed_ids_;
        std::unordered_set<core::NodeId> selected_ids_;
        core::NodeId pending_reveal_node_id_ = core::NULL_NODE;

        std::string filter_text_;
        std::string last_training_model_node_name_;
        core::NodeId click_anchor_id_ = core::NULL_NODE;
        core::NodeId rename_node_id_ = core::NULL_NODE;
        std::string rename_buffer_;
        RenameInputListener rename_input_listener_;
        core::NodeId context_menu_node_id_ = core::NULL_NODE;
        core::NodeId drag_source_id_ = core::NULL_NODE;
        core::NodeId drop_target_id_ = core::NULL_NODE;
        bool models_collapsed_ = false;
        bool scene_has_nodes_ = false;
        size_t root_count_ = 0;
        bool invert_masks_ = false;
        bool dom_dirty_ = true;
        bool tree_rebuild_needed_ = true;
        bool rename_focus_pending_ = false;
        uint64_t last_scene_generation_ = 0;
        uint64_t state_revision_ = 1;
        uint64_t last_bound_revision_ = 0;
        uint32_t last_selection_generation_ = std::numeric_limits<uint32_t>::max();
        size_t last_training_model_gaussian_count_ = std::numeric_limits<size_t>::max();
        size_t last_visible_start_ = kUnsetVisibleRange;
        size_t last_visible_end_ = kUnsetVisibleRange;
        float last_client_height_ = -1.0f;
        float last_content_height_ = -1.0f;
        std::string last_header_text_;
        std::vector<std::string> row_top_dp_cache_;
        bool last_header_visible_ = false;
        bool last_header_expanded_ = true;
        float panel_screen_x_ = 0.0f;
        float panel_screen_y_ = 0.0f;

        static constexpr int kRowHeightDpInt = 20;
        static constexpr int kHeaderHeightDpInt = 24;
        static constexpr float kRowHeightDp = 20.0f;
        static constexpr float kHeaderHeightDp = 24.0f;
        static constexpr int kOverscanRows = 12;
        static constexpr int kAutoCollapseCameraGroupThreshold = 25;
    };

} // namespace lfs::vis::gui
