/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/nodes/evaluator.hpp"
#include "core/uuid.hpp"
#include "gui/node_canvas_interaction.hpp"

#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/EventListener.h>
#include <RmlUi/Core/Geometry.h>
#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace lfs::nodes {
    class NodeTree;
    struct Node;
    struct NodeTypeInfo;
    class SocketTypeRegistry;
} // namespace lfs::nodes

namespace lfs::vis {
    class ModifierManager;
    class SceneManager;
} // namespace lfs::vis

namespace lfs::vis::gui {
    class GlobalContextMenu;

    class NodeCanvasElement final : public Rml::Element, private Rml::EventListener {
    public:
        explicit NodeCanvasElement(const Rml::String& tag);
        ~NodeCanvasElement() override;

        void setContext(SceneManager* scene_manager, GlobalContextMenu* context_menu);
        void setPanelScreenOffset(float x, float y) { panel_screen_offset_ = {x, y}; }
        void setPanelSize(float width, float height) { panel_size_ = {width, height}; }
        void setSidebarVisible(bool visible);
        void setPreviewSelection(bool enabled);
        [[nodiscard]] bool sidebarVisible() const { return sidebar_visible_; }
        [[nodiscard]] bool previewSelection() const { return preview_selection_; }
        [[nodiscard]] bool paintSelectionSelected() const;
        [[nodiscard]] bool paintModeActive() const;
        void togglePaintMode();
        bool handleKey(int scancode, bool shift, bool control, bool alt);
        void headerAction(std::string_view action, float screen_x, float screen_y);
        [[nodiscard]] std::string statusText() const;
        [[nodiscard]] bool modifiersVisible() const;
        [[nodiscard]] bool needsModelUpdate() const;
        void invalidateView();
        bool arrange(bool selection_only = true,
                     const std::optional<std::unordered_set<std::string>>& nodes = std::nullopt);
        void setView(CanvasPoint pan, float zoom);
        [[nodiscard]] nlohmann::json viewState();
        void refresh() { syncModel(); }
        bool showModifier(std::string_view uuid);
        bool selectNodes(const std::unordered_set<std::string>& nodes, const std::optional<lfs::nodes::Link>& link);

    protected:
        void OnRender() override;
        void OnUpdate() override;
        void OnResize() override;
        void ProcessDefaultAction(Rml::Event& event) override;
        void ProcessEvent(Rml::Event& event) override;
        bool GetIntrinsicDimensions(Rml::Vector2f& dimensions, float& ratio) override;

    private:
        struct NodeVisual {
            CanvasNode interaction;
            std::string title;
            std::string category;
            std::string error;
            bool muted = false;
        };

        void ensureDom();
        void syncModel();
        void rebuildModel();
        void rebuildDom();
        void rebuildGeometry();
        void updateNodePositions();
        void updateSelectedClasses();
        void refreshEvaluation();
        void updateEvaluationDom();
        void updateProgress();
        [[nodiscard]] std::string nodeStatus(std::string_view name) const;
        bool processAddMenuEvent(Rml::Event& event);
        void filterAddMenu(std::string_view search);
        void highlightAddType(std::string_view id);
        void moveAddHighlight(int direction);
        void closeAddMenu();
        bool processHelpEvent(Rml::Event& event);
        void updateSidebar();
        void updateSelectionPreview();
        bool processFieldEvent(Rml::Event& event);
        void processControlEvent(Rml::Event& event);
        void finishFieldEdit(bool cancel);
        void stepField();
        void repeatFieldStep();
        void commitControl(Rml::Element* target, const Rml::Event* event, bool record_undo = true);
        void openModifierMenu(std::string_view uuid, float x, float y);
        void openModifierLibrary(float x, float y);
        void executeCommands(const std::vector<CanvasCommand>& commands);
        void openAddMenu(float screen_x, float screen_y);
        void openCanvasMenu(CanvasPoint pointer);
        [[nodiscard]] bool isPanPress(const Rml::Event& event) const;
        void addNode(std::string_view type_id, CanvasPoint graph_position);
        void addModifier();
        void removeSelected();
        void duplicateSelected();
        void toggleMuted();
        void frameAll();
        void focusCanvas();
        [[nodiscard]] CanvasPoint localPointer(const Rml::Event& event);
        [[nodiscard]] lfs::nodes::NodeTree* activeTree();
        [[nodiscard]] const lfs::nodes::NodeTree* activeTree() const;
        [[nodiscard]] bool editableMode() const;
        [[nodiscard]] std::array<float, 4> socketColor(std::string_view type) const;
        [[nodiscard]] std::string categoryColor(std::string_view category) const;
        [[nodiscard]] std::optional<core::Uuid> activeHost() const;

        SceneManager* scene_manager_ = nullptr;
        ModifierManager* manager_ = nullptr;
        GlobalContextMenu* context_menu_ = nullptr;
        std::unique_ptr<lfs::nodes::SocketTypeRegistry> socket_types_;
        Rml::Element* nodes_element_ = nullptr;
        Rml::Element* viewport_element_ = nullptr;
        Rml::Element* sidebar_element_ = nullptr;
        Rml::Element* notice_element_ = nullptr;
        Rml::Element* add_menu_ = nullptr;
        CanvasPoint add_position_;
        std::string first_add_type_;
        std::unordered_map<std::string, bool> expanded_help_;
        std::uint64_t language_generation_ = 0;
        std::string busy_node_;
        std::unordered_set<std::string> queued_nodes_;
        std::unordered_map<std::string, lfs::nodes::NodeEvaluation> progress_nodes_;
        std::uint64_t progress_generation_ = 0;
        std::string file_error_;
        std::vector<NodeVisual> visuals_;
        std::unordered_map<std::string, Rml::Element*> node_elements_;
        std::unordered_map<std::string, std::string> node_markup_;
        std::unordered_map<std::string, nlohmann::json> node_content_state_;
        std::unordered_map<std::string, CanvasRect> node_layout_;
        std::string sidebar_markup_;
        CanvasPoint layout_pan_;
        float canvas_width_ = -1.0f;
        std::unordered_set<std::string> selected_nodes_;
        std::unordered_set<std::string> seen_trees_;
        bool frame_pending_ = false;
        std::optional<CanvasLink> selected_link_;
        NodeCanvasInteraction interaction_;
        std::string active_tree_uuid_;
        std::string active_modifier_uuid_;
        std::string host_uuid_;
        std::uint64_t last_generation_ = 0;
        std::uint64_t last_result_generation_ = 0;
        std::uint64_t last_selection_generation_ = 0;
        std::unordered_map<std::string, std::string> evaluation_errors_;
        double evaluation_total_ms_ = 0.0;
        bool sidebar_visible_ = true;
        bool preview_selection_ = true;
        bool preview_selection_opt_out_ = false;
        bool dom_dirty_ = true;
        bool paint_mode_shown_ = false;
        bool geometry_dirty_ = true;
        bool pointer_down_ = false;
        bool pending_context_menu_ = false;
        CanvasPoint context_press_;
        float dp_ratio_ = 1.0f;
        float layout_zoom_ = -1.0f;
        CanvasPoint panel_screen_offset_;
        CanvasPoint panel_size_;
        std::size_t theme_signature_ = 0;
        Rml::Element* active_field_ = nullptr;
        float field_start_x_ = 0.0f;
        double field_start_value_ = 0.0;
        bool field_dragged_ = false;
        int field_step_direction_ = 0;
        double field_step_multiplier_ = 1.0;
        std::chrono::steady_clock::time_point field_repeat_at_;
        std::optional<nlohmann::json> field_before_;
        Rml::Geometry grid_geometry_;
        Rml::Geometry wire_geometry_;
        Rml::Geometry live_wire_geometry_;
        Rml::Geometry overlay_geometry_;
        CanvasPoint geometry_pan_;
        Rml::Vector2f geometry_size_;
        float geometry_zoom_ = -1.0f;
        std::size_t geometry_theme_ = 0;
        std::uint64_t geometry_route_generation_ = 0;
        std::optional<CanvasLink> geometry_selected_link_;
        std::optional<CanvasLink> geometry_highlighted_link_;
        std::vector<CanvasNode> geometry_nodes_;
        std::optional<CanvasSocket> geometry_snap_;
        bool geometry_dragging_wire_ = false;
    };

} // namespace lfs::vis::gui
