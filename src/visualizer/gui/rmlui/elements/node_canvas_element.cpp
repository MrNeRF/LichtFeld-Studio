/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rmlui/elements/node_canvas_element.hpp"

#include "core/event_bridge/localization_manager.hpp"
#include "core/nodes/registry.hpp"
#include "core/nodes/tree.hpp"
#include "gui/global_context_menu.hpp"
#include "gui/rmlui/elements/node_canvas_dom.hpp"
#include "gui/rmlui/elements/node_canvas_widgets.hpp"
#include "gui/rmlui/rml_theme.hpp"
#include "input/input_controller.hpp"
#include "scene/scene_manager.hpp"
#include "theme/theme.hpp"
#include "visualizer/app_store.hpp"
#include "visualizer/nodes/modifier_manager.hpp"
#include "window/window_manager.hpp"

#include <RmlUi/Core/Context.h>
#include <RmlUi/Core/ElementDocument.h>
#include <RmlUi/Core/ElementInstancer.h>
#include <RmlUi/Core/Elements/ElementFormControl.h>
#include <RmlUi/Core/Elements/ElementFormControlInput.h>
#include <RmlUi/Core/Elements/ElementFormControlSelect.h>
#include <RmlUi/Core/Input.h>
#include <RmlUi/Core/RenderManager.h>
#include <SDL3/SDL_scancode.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <format>
#include <ranges>

namespace lfs::vis::gui {
    namespace {
        constexpr float kNodeWidth = 224.0f;
        constexpr float kTitleHeight = 26.0f;
        constexpr float kSidebarWidth = 280.0f;
        constexpr int kBezierSegments = 64;

        class CanvasCard final : public Rml::Element {
        public:
            explicit CanvasCard(const Rml::String& tag) : Rml::Element(tag) {}
            Rml::Geometry sockets;

        protected:
            void OnRender() override {
                sockets.Render(GetAbsoluteOffset(Rml::BoxArea::Border));
            }
        };

        class CanvasOverlay final : public Rml::Element {
        public:
            explicit CanvasOverlay(Rml::Geometry& selection)
                : Rml::Element("div"), selection_(selection) {
                SetProperty("position", "absolute");
                SetProperty("left", "0dp");
                SetProperty("top", "0dp");
                SetProperty("width", "100%");
                SetProperty("height", "100%");
                SetProperty("pointer-events", "none");
            }

        protected:
            void OnRender() override {
                const auto offset = GetAbsoluteOffset(Rml::BoxArea::Content);
                selection_.Render(offset);
            }

        private:
            Rml::Geometry& selection_;
        };

        std::string px(const float value) { return std::format("{:.1f}px", value); }

        using node_widgets::escape;
        using node_widgets::valuePayload;

        float currentDpRatio(const Rml::Element* element) {
            const auto* context = element ? element->GetContext() : nullptr;
            return context ? std::max(context->GetDensityIndependentPixelRatio(), 0.01f) : 1.0f;
        }

        std::array<float, 4> themeColor(const ThemeColor& value) {
            return {value.x, value.y, value.z, value.w};
        }

        Rml::ColourbPremultiplied color(const std::array<float, 4>& value, const float opacity = 1.0f) {
            const auto byte = [](const float component) {
                return static_cast<Rml::byte>(std::clamp(component * 255.0f, 0.0f, 255.0f));
            };
            const float alpha = value[3] * opacity;
            return {byte(value[0] * alpha), byte(value[1] * alpha), byte(value[2] * alpha), byte(alpha)};
        }

        void quad(Rml::Mesh& mesh, const float x0, const float y0, const float x1, const float y1,
                  const Rml::ColourbPremultiplied value) {
            const int base = static_cast<int>(mesh.vertices.size());
            mesh.vertices.push_back({{x0, y0}, value, {0, 0}});
            mesh.vertices.push_back({{x1, y0}, value, {0, 0}});
            mesh.vertices.push_back({{x1, y1}, value, {0, 0}});
            mesh.vertices.push_back({{x0, y1}, value, {0, 0}});
            mesh.indices.insert(mesh.indices.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
        }

        void circle(Rml::Mesh& mesh, const CanvasPoint center, const float radius,
                    const Rml::ColourbPremultiplied value) {
            constexpr int segments = 16;
            const int center_index = static_cast<int>(mesh.vertices.size());
            mesh.vertices.push_back({{center.x, center.y}, value, {0, 0}});
            for (int index = 0; index <= segments; ++index) {
                const float angle = static_cast<float>(index) * 2.0f * 3.14159265f /
                                    static_cast<float>(segments);
                mesh.vertices.push_back({{center.x + std::cos(angle) * radius,
                                          center.y + std::sin(angle) * radius},
                                         value,
                                         {0, 0}});
                if (index > 0)
                    mesh.indices.insert(mesh.indices.end(),
                                        {center_index, center_index + index, center_index + index + 1});
            }
        }

        void gridDot(Rml::Mesh& mesh, const CanvasPoint center, const float radius,
                     const Rml::ColourbPremultiplied value) {
            // At this subpixel size a feathered four-triangle dot is smoother
            // and substantially smaller than a tessellated circle.
            const int base = static_cast<int>(mesh.vertices.size());
            const Rml::ColourbPremultiplied clear(0, 0, 0, 0);
            mesh.vertices.push_back({{center.x, center.y}, value, {0, 0}});
            mesh.vertices.push_back({{center.x - radius, center.y - radius}, clear, {0, 0}});
            mesh.vertices.push_back({{center.x + radius, center.y - radius}, clear, {0, 0}});
            mesh.vertices.push_back({{center.x + radius, center.y + radius}, clear, {0, 0}});
            mesh.vertices.push_back({{center.x - radius, center.y + radius}, clear, {0, 0}});
            mesh.indices.insert(mesh.indices.end(), {base, base + 1, base + 2,
                                                     base, base + 2, base + 3,
                                                     base, base + 3, base + 4,
                                                     base, base + 4, base + 1});
        }

        void pathRibbon(Rml::Mesh& mesh, const std::vector<CanvasPoint>& points,
                        const Rml::ColourbPremultiplied value, const float width, const float dp_ratio) {
            const int base = static_cast<int>(mesh.vertices.size());
            const Rml::ColourbPremultiplied transparent(0, 0, 0, 0);
            for (std::size_t index = 0; index < points.size(); ++index) {
                const auto point = points[index];
                const CanvasPoint tangent_vector = points[std::min(index + 1, points.size() - 1)] - points[index == 0 ? 0 : index - 1];
                const float length = std::max(std::hypot(tangent_vector.x, tangent_vector.y), 0.001f);
                const CanvasPoint normal{-tangent_vector.y / length, tangent_vector.x / length};
                const float offsets[] = {-width * 0.5f - dp_ratio, -width * 0.5f,
                                         width * 0.5f, width * 0.5f + dp_ratio};
                for (int edge = 0; edge < 4; ++edge) {
                    const auto vertex = point + normal * offsets[edge];
                    mesh.vertices.push_back({{vertex.x, vertex.y},
                                             edge == 0 || edge == 3 ? transparent : value,
                                             {0, 0}});
                }
                if (index == 0)
                    continue;
                const int row = base + static_cast<int>(index) * 4;
                for (int edge = 0; edge < 3; ++edge)
                    mesh.indices.insert(mesh.indices.end(),
                                        {row - 4 + edge, row + edge, row + edge + 1,
                                         row - 4 + edge, row + edge + 1, row - 3 + edge});
            }
        }

        void ribbon(Rml::Mesh& mesh, const CanvasPoint from, const CanvasPoint to,
                    const Rml::ColourbPremultiplied value, const float width, const float dp_ratio) {
            const float tangent = std::max(std::abs(to.x - from.x) * 0.5f, 36.0f * dp_ratio);
            const CanvasPoint b{from.x + tangent, from.y};
            const CanvasPoint c{to.x - tangent, to.y};
            std::vector<CanvasPoint> points;
            for (int index = 0; index <= kBezierSegments; ++index) {
                const float t = static_cast<float>(index) / kBezierSegments;
                const float u = 1.0f - t;
                points.push_back(from * (u * u * u) + b * (3.0f * u * u * t) + c * (3.0f * u * t * t) + to * (t * t * t));
            }
            pathRibbon(mesh, points, value, width, dp_ratio);
        }

        void straightRibbon(Rml::Mesh& mesh, const CanvasPoint from, const CanvasPoint to,
                            const Rml::ColourbPremultiplied value, const float width) {
            const CanvasPoint delta = to - from;
            const float length = std::max(std::sqrt(delta.x * delta.x + delta.y * delta.y), 0.001f);
            const CanvasPoint normal{-delta.y / length * width * 0.5f,
                                     delta.x / length * width * 0.5f};
            const int base = static_cast<int>(mesh.vertices.size());
            mesh.vertices.push_back({{from.x + normal.x, from.y + normal.y}, value, {0, 0}});
            mesh.vertices.push_back({{from.x - normal.x, from.y - normal.y}, value, {0, 0}});
            mesh.vertices.push_back({{to.x + normal.x, to.y + normal.y}, value, {0, 0}});
            mesh.vertices.push_back({{to.x - normal.x, to.y - normal.y}, value, {0, 0}});
            mesh.indices.insert(mesh.indices.end(),
                                {base, base + 1, base + 3, base, base + 3, base + 2});
        }

        std::string sceneTypeLabel(const core::NodeType type) {
            switch (type) {
            case core::NodeType::SPLAT: return LOC("node_editor.type_splat");
            case core::NodeType::POINTCLOUD: return LOC("node_editor.type_point_cloud");
            case core::NodeType::MESH: return LOC("node_editor.type_mesh");
            default: return LOC("node_editor.type_unsupported");
            }
        }

        std::vector<lfs::nodes::SocketDecl>
        canvasInputs(const lfs::nodes::NodeTree& tree, const lfs::nodes::Node& node,
                     const std::shared_ptr<const lfs::nodes::NodeTypeInfo>& type) {
            if (node.type_id == "lfs.group_input")
                return {};
            if (node.type_id != "lfs.group_output")
                return type ? type->inputs : std::vector<lfs::nodes::SocketDecl>{};
            std::vector<lfs::nodes::SocketDecl> result;
            result.reserve(tree.interface.outputs.size());
            for (const auto& socket : tree.interface.outputs) {
                result.push_back({socket.identifier, socket.label, socket.type, socket.default_value,
                                  socket.min, socket.max, socket.step});
                if (type) {
                    const auto declared = std::ranges::find(type->inputs, socket.identifier, &lfs::nodes::SocketDecl::identifier);
                    if (declared != type->inputs.end()) {
                        result.back().label = declared->label;
                        result.back().description = declared->description;
                    }
                }
            }
            return result;
        }

        std::vector<lfs::nodes::SocketDecl>
        canvasOutputs(const lfs::nodes::NodeTree& tree, const lfs::nodes::Node& node,
                      const std::shared_ptr<const lfs::nodes::NodeTypeInfo>& type) {
            if (node.type_id == "lfs.group_output")
                return {};
            if (node.type_id != "lfs.group_input")
                return type ? type->outputs : std::vector<lfs::nodes::SocketDecl>{};
            std::vector<lfs::nodes::SocketDecl> result;
            result.reserve(tree.interface.inputs.size());
            for (const auto& socket : tree.interface.inputs) {
                result.push_back({socket.identifier, socket.label, socket.type, socket.default_value,
                                  socket.min, socket.max, socket.step});
                if (type) {
                    const auto declared = std::ranges::find(type->outputs, socket.identifier, &lfs::nodes::SocketDecl::identifier);
                    if (declared != type->outputs.end()) {
                        result.back().label = declared->label;
                        result.back().description = declared->description;
                    }
                }
            }
            return result;
        }
    } // namespace

    NodeCanvasElement::NodeCanvasElement(const Rml::String& tag)
        : Rml::Element(tag), socket_types_(std::make_unique<lfs::nodes::SocketTypeRegistry>()) {
        SetProperty("drag", "drag");
        SetAttribute("tab-index", 0);
        AddEventListener("mousescroll", this, true);
        AddEventListener("pinch", this, true);
        AddEventListener("mousemove", this, true);
        AddEventListener("mouseover", this, true);
        AddEventListener("drag", this, true);
        AddEventListener("dblclick", this, true);
        AddEventListener("blur", this, true);
        AddEventListener("focus", this, true);
        AddEventListener("click", this, true);
        AddEventListener("keydown", this, true);
        AddEventListener("change", this, true);
        AddEventListener("mousedown", this, true);
    }

    NodeCanvasElement::~NodeCanvasElement() {
        RemoveEventListener("mousescroll", this, true);
        RemoveEventListener("pinch", this, true);
        RemoveEventListener("mousemove", this, true);
        RemoveEventListener("mouseover", this, true);
        RemoveEventListener("drag", this, true);
        RemoveEventListener("dblclick", this, true);
        RemoveEventListener("blur", this, true);
        RemoveEventListener("keydown", this, true);
        RemoveEventListener("change", this, true);
        RemoveEventListener("mousedown", this, true);
        if (scene_manager_)
            scene_manager_->setModifierSelectionPreview({}, std::nullopt);
    }

    void NodeCanvasElement::setContext(SceneManager* scene_manager, GlobalContextMenu* context_menu) {
        scene_manager_ = scene_manager;
        manager_ = scene_manager ? &scene_manager->modifierManager() : nullptr;
        context_menu_ = context_menu;
        last_generation_ = 0;
        dom_dirty_ = true;
        geometry_dirty_ = true;
    }

    void NodeCanvasElement::setSidebarVisible(const bool visible) {
        if (sidebar_visible_ == visible)
            return;
        sidebar_visible_ = visible;
        dom_dirty_ = true;
        geometry_dirty_ = true;
    }

    void NodeCanvasElement::invalidateView() {
        dom_dirty_ = true;
        geometry_dirty_ = true;
        geometry_zoom_ = -1.0f;
        frame_pending_ = true;
    }

    void NodeCanvasElement::setView(const CanvasPoint pan, const float zoom) {
        interaction_.setView(pan, zoom);
        rebuildModel();
        updateNodePositions();
        geometry_dirty_ = true;
    }

    nlohmann::json NodeCanvasElement::viewState() {
        nlohmann::json result = {{"target", host_uuid_}, {"tree", active_tree_uuid_}, {"modifier", active_modifier_uuid_}, {"selected_nodes", selected_nodes_}, {"pan", {interaction_.pan().x, interaction_.pan().y}}, {"zoom", interaction_.zoom()}, {"preview_selection", preview_selection_}, {"nodes", nlohmann::json::array()}, {"links", interaction_.links().size()}};
        result["gesture_active"] = interaction_.active();
        result["pointer_down"] = pointer_down_;
        result["pointer"] = {interaction_.pointer().x, interaction_.pointer().y};
        const auto size = GetBox().GetSize(Rml::BoxArea::Content);
        result["size"] = {size.x, size.y};
        result["screen_offset"] = {panel_screen_offset_.x, panel_screen_offset_.y};
        result["sidebar_visible"] = sidebar_visible_;
        result["selected_links"] = nlohmann::json::array();
        if (selected_link_)
            result["selected_links"].push_back({{"from_node", selected_link_->from.node}, {"from_socket", selected_link_->from.identifier}, {"to_node", selected_link_->to.node}, {"to_socket", selected_link_->to.identifier}});
        for (const auto& node : interaction_.nodes()) {
            nlohmann::json sockets = nlohmann::json::array();
            for (const auto& socket : node.sockets) {
                const auto point = interaction_.graphToScreen(socket.position);
                sockets.push_back({{"identifier", socket.identifier}, {"type", socket.type}, {"output", socket.direction == CanvasSocketDirection::Output}, {"position", {point.x + panel_screen_offset_.x, point.y + panel_screen_offset_.y}}});
            }
            const auto origin = interaction_.graphToScreen({node.bounds.x, node.bounds.y});
            result["nodes"].push_back({{"name", node.id}, {"bounds", {node.bounds.x, node.bounds.y, node.bounds.width, node.bounds.height}}, {"screen_bounds", {origin.x + panel_screen_offset_.x, origin.y + panel_screen_offset_.y, node.bounds.width * interaction_.zoom(), node.bounds.height * interaction_.zoom()}}, {"sockets", sockets}});
        }
        return result;
    }

    bool NodeCanvasElement::showModifier(const std::string_view uuid) {
        const auto host = activeHost();
        const auto* stack = host && manager_ ? manager_->stack(*host) : nullptr;
        if (!stack || std::ranges::find(stack->modifiers, uuid, &Modifier::uuid) == stack->modifiers.end())
            return false;
        active_modifier_uuid_ = uuid;
        dom_dirty_ = true;
        syncModel();
        return true;
    }

    bool NodeCanvasElement::selectNodes(const std::unordered_set<std::string>& nodes,
                                        const std::optional<lfs::nodes::Link>& link) {
        syncModel();
        const auto* tree = activeTree();
        if (!tree || std::ranges::any_of(nodes, [&](const auto& name) { return !tree->find_node(name); }))
            return false;
        std::optional<CanvasLink> selected;
        if (link) {
            const auto found = std::ranges::find_if(interaction_.links(), [&](const auto& item) {
                return item.from.node == link->from_node && item.from.identifier == link->from_socket &&
                       item.to.node == link->to_node && item.to.identifier == link->to_socket;
            });
            if (found == interaction_.links().end())
                return false;
            selected = *found;
        }
        selected_nodes_ = nodes;
        selected_link_ = selected;
        interaction_.setSelectedNodes(nodes);
        updateSelectedClasses();
        updateSidebar();
        updateSelectionPreview();
        geometry_dirty_ = true;
        return true;
    }

    void NodeCanvasElement::setPreviewSelection(const bool enabled) {
        preview_selection_ = enabled;
        preview_selection_opt_out_ = !enabled;
        updateSelectionPreview();
    }

    bool NodeCanvasElement::GetIntrinsicDimensions(Rml::Vector2f& dimensions, float& ratio) {
        dimensions = {640.0f, 320.0f};
        ratio = 0.0f;
        return true;
    }

    std::optional<core::Uuid> NodeCanvasElement::activeHost() const {
        if (!scene_manager_)
            return std::nullopt;
        const auto ids = scene_manager_->getSelectedNodeIds();
        if (ids.size() != 1)
            return std::nullopt;
        const auto* node = scene_manager_->getScene().getNodeById(ids.front());
        if (!node || (node->type != core::NodeType::SPLAT && node->type != core::NodeType::POINTCLOUD &&
                      node->type != core::NodeType::MESH))
            return std::nullopt;
        return node->uuid;
    }

    bool NodeCanvasElement::editableMode() const {
        return scene_manager_ && scene_manager_->getContentType() == SceneManager::ContentType::SplatFiles;
    }

    lfs::nodes::NodeTree* NodeCanvasElement::activeTree() {
        return manager_ && !active_tree_uuid_.empty() ? manager_->tree(active_tree_uuid_) : nullptr;
    }

    const lfs::nodes::NodeTree* NodeCanvasElement::activeTree() const {
        return const_cast<NodeCanvasElement*>(this)->activeTree();
    }

    void NodeCanvasElement::ensureDom() {
        if (nodes_element_)
            return;
        auto* document = GetOwnerDocument();
        if (!document)
            return;
        auto nodes = document->CreateElement("div");
        nodes->SetId("node-canvas-nodes");
        auto viewport = document->CreateElement("div");
        viewport->SetId("node-canvas-viewport");
        viewport_element_ = AppendChild(std::move(viewport));
        nodes_element_ = viewport_element_->AppendChild(std::move(nodes));
        static Rml::ElementInstancerGeneric<Rml::Element> overlay_instancer;
        Rml::ElementPtr overlay(new CanvasOverlay(overlay_geometry_));
        overlay->SetInstancer(&overlay_instancer);
        AppendChild(std::move(overlay));
        auto notice = document->CreateElement("div");
        notice->SetId("node-canvas-notice");
        notice_element_ = AppendChild(std::move(notice));
        auto sidebar = document->CreateElement("div");
        sidebar->SetId("node-editor-sidebar");
        sidebar_element_ = AppendChild(std::move(sidebar));
    }

    std::array<float, 4> NodeCanvasElement::socketColor(const std::string_view type) const {
        if (const auto info = socket_types_->find(type))
            return info->color;
        return {0.65f, 0.65f, 0.65f, 1.0f};
    }

    std::string NodeCanvasElement::categoryColor(const std::string_view category) const {
        const auto& palette = theme().palette;
        const ThemeColor tint = category == "Selection"    ? palette.warning
                                : category == "Input"      ? palette.success
                                : category == "Splat"      ? palette.secondary
                                : category == "Colour"     ? palette.warning
                                : category == "Clean-up"   ? palette.success
                                : category == "Conversion" ? palette.primary
                                : category == "Utilities"  ? palette.text_dim
                                : category == "Output"     ? palette.secondary
                                                           : palette.info;
        constexpr float strength = 0.32f;
        return rml_theme::colorToRml({palette.surface.x * (1.0f - strength) + tint.x * strength,
                                      palette.surface.y * (1.0f - strength) + tint.y * strength,
                                      palette.surface.z * (1.0f - strength) + tint.z * strength, 1.0f});
    }

    void NodeCanvasElement::syncModel() {
        ensureDom();
        updateProgress();
        const auto language_generation = event::LocalizationManager::getInstance().getCurrentLanguageGeneration();
        if (language_generation_ != language_generation) {
            language_generation_ = language_generation;
            dom_dirty_ = true;
            closeAddMenu();
        }
        const float ratio = currentDpRatio(this);
        const auto signature = rml_theme::currentThemeSignature();
        if (dp_ratio_ != ratio || theme_signature_ != signature) {
            dp_ratio_ = ratio;
            theme_signature_ = signature;
            interaction_.setDpRatio(ratio);
            dom_dirty_ = true;
            geometry_dirty_ = true;
        }
        if (active_field_)
            return;
        const auto selection_generation = app_store().selection_generation.get();
        const auto host = activeHost();
        const std::string next_host = host ? host->to_string() : std::string{};
        const std::uint64_t generation = manager_ ? manager_->generation() : 0;
        // A result may change during a gesture, but its live layout belongs to
        // the gesture until release. Do not replace it with stored locations.
        if (interaction_.active() && next_host == host_uuid_) {
            if (manager_ && last_result_generation_ != manager_->resultGeneration()) {
                last_result_generation_ = manager_->resultGeneration();
                refreshEvaluation();
                updateEvaluationDom();
            }
            return;
        }
        if (selection_generation == last_selection_generation_ && generation == last_generation_ &&
            next_host == host_uuid_ && !dom_dirty_ && !frame_pending_) {
            if (manager_ && last_result_generation_ != manager_->resultGeneration()) {
                last_result_generation_ = manager_->resultGeneration();
                refreshEvaluation();
                updateEvaluationDom();
                updateSelectionPreview();
            }
            return;
        }
        last_selection_generation_ = selection_generation;
        last_generation_ = generation;
        last_result_generation_ = manager_ ? manager_->resultGeneration() : 0;
        host_uuid_ = next_host;

        const auto previous_tree = active_tree_uuid_;
        if (!host) {
            active_tree_uuid_.clear();
            active_modifier_uuid_.clear();
        } else if (auto* stack = manager_->stack(*host); stack && !stack->modifiers.empty()) {
            auto found = std::ranges::find(stack->modifiers, active_modifier_uuid_, &Modifier::uuid);
            if (found == stack->modifiers.end() || !manager_->tree(found->tree_uuid))
                found = std::ranges::find_if(stack->modifiers, [&](const auto& modifier) {
                    return manager_->tree(modifier.tree_uuid) != nullptr;
                });
            active_modifier_uuid_ = found == stack->modifiers.end() ? std::string{} : found->uuid;
            active_tree_uuid_ = found == stack->modifiers.end() ? std::string{} : found->tree_uuid;
        } else {
            active_tree_uuid_.clear();
            active_modifier_uuid_.clear();
        }
        if (previous_tree != active_tree_uuid_) {
            interaction_.cancel();
            selected_nodes_.clear();
            selected_link_.reset();
            closeAddMenu();
            geometry_zoom_ = -1.0f;
            frame_pending_ = true;
        }
        rebuildModel();
        if (activeTree() && seen_trees_.insert(active_tree_uuid_).second) {
            const auto& nodes = interaction_.nodes();
            bool overlap = false;
            for (std::size_t i = 0; i < nodes.size() && !overlap; ++i)
                for (std::size_t j = i + 1; j < nodes.size() && !overlap; ++j)
                    overlap = nodes[i].bounds.intersects(nodes[j].bounds);
            if (overlap)
                arrange(false);
        }
        rebuildDom();
        const auto size = GetBox().GetSize(Rml::BoxArea::Content);
        if (frame_pending_ && size.x > 1.0f &&
            (panel_size_.x == 0.0f || (std::abs(size.x - panel_size_.x) < 2.0f && std::abs(size.y - panel_size_.y) < 2.0f))) {
            frame_pending_ = false;
            frameAll();
        }
        geometry_dirty_ = true;
        dom_dirty_ = false;
    }

    bool NodeCanvasElement::needsModelUpdate() const {
        return dom_dirty_ || geometry_dirty_ || frame_pending_ || pointer_down_ || field_step_direction_ != 0 ||
               language_generation_ != event::LocalizationManager::getInstance().getCurrentLanguageGeneration() ||
               (manager_ && (manager_->generation() != last_generation_ || manager_->resultGeneration() != last_result_generation_ || manager_->progress().busy)) ||
               app_store().selection_generation.get() != last_selection_generation_;
    }

    void NodeCanvasElement::rebuildModel() {
        visuals_.clear();
        std::vector<CanvasNode> nodes;
        std::vector<CanvasLink> links;
        auto* tree = activeTree();
        if (!tree) {
            interaction_.setGraph({}, {});
            return;
        }
        ModifierEvaluation evaluation;
        if (const auto host = activeHost())
            if (const auto* result = manager_->lastResult(*host))
                evaluation = *result;
        evaluation_errors_ = evaluation.errors;
        evaluation_total_ms_ = evaluation.total_time_ms;
        std::string active_modifier_name;
        if (const auto host = activeHost()) {
            if (const auto* stack = manager_->stack(*host)) {
                const auto modifier =
                    std::ranges::find(stack->modifiers, active_modifier_uuid_, &Modifier::uuid);
                if (modifier != stack->modifiers.end())
                    active_modifier_name = modifier->name;
            }
        }
        for (const auto& node : tree->nodes) {
            const auto type = manager_->registry().find_localized(node.type_id);
            const auto inputs = canvasInputs(*tree, node, type);
            const auto outputs = canvasOutputs(*tree, node, type);
            NodeVisual visual;
            visual.interaction.id = node.name;
            visual.title = type ? type->label : node.type_id;
            visual.category = type ? type->category : "Missing";
            visual.muted = node.muted;
            const std::string error_key = active_modifier_name.empty()
                                              ? node.name
                                              : active_modifier_name + "/" + node.name;
            if (const auto error = evaluation.errors.find(error_key); error != evaluation.errors.end())
                visual.error = error->second;
            else if (const auto error = evaluation.errors.find(node.name); error != evaluation.errors.end())
                visual.error = error->second;
            const CanvasPoint origin{node.location[0] * dp_ratio_, node.location[1] * dp_ratio_};
            constexpr float row_height = 22.0f;
            float row_y = std::max(kTitleHeight, 18.0f / interaction_.zoom()) + 6.0f;
            for (const auto& socket : outputs) {
                visual.interaction.sockets.push_back({.node = node.name, .identifier = socket.identifier, .type = socket.type, .direction = CanvasSocketDirection::Output, .position = {origin.x + kNodeWidth * dp_ratio_, origin.y + (row_y + row_height * 0.5f) * dp_ratio_}});
                row_y += row_height;
            }
            const bool expanded = node_widgets::settingsExpanded(node);
            const auto settings = std::ranges::count_if(inputs, node_widgets::singleValue) +
                                  (type ? std::ranges::count_if(type->properties, [](const auto& property) {
                                      return property.kind != lfs::nodes::PropertyKind::Data;
                                  })
                                        : 0);
            if (settings > 0)
                row_y += row_height;
            if (type && expanded)
                for (const auto& property : type->properties)
                    if (property.kind != lfs::nodes::PropertyKind::Data)
                        row_y += row_height;
            for (const auto& socket : inputs) {
                if (!node_widgets::inputVisible(*tree, node, socket))
                    continue;
                visual.interaction.sockets.push_back({.node = node.name, .identifier = socket.identifier, .type = socket.type, .direction = CanvasSocketDirection::Input, .position = {origin.x, origin.y + (row_y + row_height * 0.5f) * dp_ratio_}, .multi_input = socket.multi_input});
                row_y += row_height * node_widgets::inputRows(*tree, node, socket);
            }
            const float borders = 2.0f * std::max(1.0f, 0.75f / interaction_.zoom());
            visual.interaction.bounds = {origin.x, origin.y, kNodeWidth * dp_ratio_,
                                         (row_y + 6.0f + 20.0f + borders) * dp_ratio_};
            nodes.push_back(visual.interaction);
            visuals_.push_back(std::move(visual));
        }
        const auto find_socket = [&](const std::string& node, const std::string& identifier,
                                     const CanvasSocketDirection direction) -> std::optional<CanvasSocket> {
            const auto visual = std::ranges::find_if(visuals_, [&](const NodeVisual& item) {
                return item.interaction.id == node;
            });
            if (visual == visuals_.end())
                return std::nullopt;
            const auto socket = std::ranges::find_if(visual->interaction.sockets, [&](const CanvasSocket& item) {
                return item.identifier == identifier && item.direction == direction;
            });
            return socket == visual->interaction.sockets.end() ? std::nullopt
                                                               : std::optional(*socket);
        };
        std::unordered_map<std::string, std::unordered_map<std::string, size_t>> input_counts;
        std::unordered_map<std::string, std::unordered_map<std::string, size_t>> input_indices;
        for (const auto& link : tree->links)
            ++input_counts[link.to_node][link.to_socket];
        for (const auto& link : tree->links) {
            const auto from = find_socket(link.from_node, link.from_socket, CanvasSocketDirection::Output);
            auto to = find_socket(link.to_node, link.to_socket, CanvasSocketDirection::Input);
            if (from && to) {
                const auto count = input_counts[link.to_node][link.to_socket];
                if (to->multi_input && count > 1) {
                    const auto index = input_indices[link.to_node][link.to_socket]++;
                    to->position.y += (static_cast<float>(index) / static_cast<float>(count - 1) - 0.5f) * 6.0f * dp_ratio_;
                }
                links.push_back({*from, *to});
            }
        }
        interaction_.setGraph(std::move(nodes), std::move(links));
        interaction_.setSelectedNodes(selected_nodes_);
        updateSelectionPreview();
    }

    void NodeCanvasElement::updateSelectionPreview() {
        if (!scene_manager_ || !manager_)
            return;
        if (!preview_selection_opt_out_ && selected_nodes_.size() == 1) {
            const auto* tree = activeTree();
            const auto* node = tree ? tree->find_node(*selected_nodes_.begin()) : nullptr;
            const auto type = node ? manager_->registry().find(node->type_id) : nullptr;
            if (type && std::ranges::any_of(type->outputs, [](const auto& socket) {
                    return socket.type == lfs::nodes::BOOL_SOCKET ||
                           (socket.identifier == "Selection" && socket.type != lfs::nodes::GEOMETRY_SOCKET);
                }))
                preview_selection_ = true;
        }
        const auto host = activeHost();
        if (!preview_selection_ || !host || selected_nodes_.size() != 1 ||
            active_modifier_uuid_.empty()) {
            scene_manager_->setModifierSelectionPreview({}, std::nullopt);
            return;
        }
        scene_manager_->setModifierSelectionPreview(
            *host, manager_->selectionPreview(*host, active_modifier_uuid_,
                                              *selected_nodes_.begin()));
    }

    void NodeCanvasElement::refreshEvaluation() {
        const auto host = activeHost();
        const auto* result = host && manager_ ? manager_->lastResult(*host) : nullptr;
        if (!result)
            return;
        evaluation_errors_ = result->errors;
        evaluation_total_ms_ = result->total_time_ms;
        std::string prefix;
        if (const auto* stack = manager_->stack(*host)) {
            const auto modifier = std::ranges::find(stack->modifiers, active_modifier_uuid_, &Modifier::uuid);
            if (modifier != stack->modifiers.end())
                prefix = modifier->name + "/";
        }
        for (auto& visual : visuals_) {
            const auto error = result->errors.find(prefix + visual.interaction.id);
            visual.error = error == result->errors.end() ? std::string{} : error->second;
        }
    }

    void NodeCanvasElement::rebuildDom() {
        if (!nodes_element_ || !notice_element_ || !sidebar_element_)
            return;
        std::erase_if(node_elements_, [&](const auto& entry) {
            if (activeTree() && activeTree()->find_node(entry.first))
                return false;
            nodes_element_->RemoveChild(entry.second);
            node_markup_.erase(entry.first);
            node_content_state_.erase(entry.first);
            node_layout_.erase(entry.first);
            selected_nodes_.erase(entry.first);
            return true;
        });
        const auto host = activeHost();
        if (!editableMode()) {
            notice_element_->SetInnerRML(escape(LOC("node_editor.edit_mode_notice")));
            notice_element_->SetProperty("display", "flex");
        } else if (!host) {
            notice_element_->SetInnerRML(escape(LOC("node_editor.select_node_notice")));
            notice_element_->SetProperty("display", "flex");
        } else if (!activeTree()) {
            notice_element_->SetInnerRML(escape(LOC("node_editor.add_modifier_notice")));
            notice_element_->SetProperty("display", "flex");
        } else {
            notice_element_->SetProperty("display", "none");
        }
        auto* document = GetOwnerDocument();
        auto* tree = activeTree();
        if (document && tree) {
            for (const auto& visual : visuals_) {
                const auto* node = tree->find_node(visual.interaction.id);
                const auto type = node ? manager_->registry().find_localized(node->type_id) : nullptr;
                const auto inputs = node ? canvasInputs(*tree, *node, type)
                                         : std::vector<lfs::nodes::SocketDecl>{};
                const auto outputs = node ? canvasOutputs(*tree, *node, type)
                                          : std::vector<lfs::nodes::SocketDecl>{};
                auto*& element = node_elements_[visual.interaction.id];
                if (!element) {
                    static Rml::ElementInstancerGeneric<CanvasCard> card_instancer;
                    auto card = card_instancer.InstanceElement(nodes_element_, "div", {});
                    card->SetInstancer(&card_instancer);
                    element = nodes_element_->AppendChild(std::move(card));
                }
                element->SetClass("node-box", true);
                element->SetClass("selected", selected_nodes_.contains(visual.interaction.id));
                element->SetClass("muted", visual.muted);
                element->SetClass("failed", !visual.error.empty());
                element->SetClass("evaluating", busy_node_ == visual.interaction.id);
                element->SetAttribute("data-node", visual.interaction.id);
                if (!visual.error.empty())
                    element->SetAttribute("title", visual.error);
                else if (type && !type->description.empty())
                    element->SetAttribute("title", type->description);
                nlohmann::json content_state = {
                    {"values", node->input_values},
                    {"properties", node->properties},
                    {"descriptor", reinterpret_cast<std::uintptr_t>(type.get())},
                    {"theme", theme_signature_},
                    {"language", language_generation_},
                    {"optional", LOC("node_editor.optional_selection")}};
                for (const auto& input : inputs)
                    content_state["inputs"].push_back({input.identifier, input.label, input.type});
                for (const auto& output : outputs)
                    content_state["outputs"].push_back({output.identifier, output.label, output.type});
                if (node_content_state_[visual.interaction.id] != content_state) {
                    std::string title = "<div class=\"node-title\" title=\"" + escape(type ? type->description : "") + "\" style=\"background-color:" +
                                        categoryColor(visual.category) + "\">" + node_widgets::categoryIcon(visual.category) +
                                        "<span class=\"node-title-label\">" + escape(visual.title) +
                                        "</span>";
                    title += "</div><div class=\"socket-rows\">";
                    for (const auto& output : outputs)
                        title += "<div class=\"socket-row output\" title=\"" + escape(output.description) + "\"><span class=\"socket-label\">" +
                                 escape(output.label) + "</span></div>";
                    const auto settings = std::ranges::count_if(inputs, node_widgets::singleValue) +
                                          (type ? std::ranges::count_if(type->properties, [](const auto& property) {
                                              return property.kind != lfs::nodes::PropertyKind::Data;
                                          })
                                                : 0);
                    if (settings > 0)
                        title += "<button class=\"node-settings\" data-action=\"node-settings\" data-node=\"" +
                                 escape(node->name) + "\"><span class=\"settings-arrow\" data-preserve-content></span> " +
                                 escape(std::vformat(LOC(settings == 1 ? "node_editor.settings_count_one"
                                                                      : "node_editor.settings_count_other"),
                                                     std::make_format_args(settings))) +
                                 "</button>";
                    if (type)
                        for (const auto& property : type->properties)
                            if (property.kind != lfs::nodes::PropertyKind::Data)
                                title += "<div class=\"socket-row node-property\" title=\"" +
                                         escape(property.description) + "\">" +
                                         node_widgets::property(*node, property) + "</div>";
                    for (const auto& input : inputs) {
                        const bool has_widget = !input.hide_value &&
                                                input.type != lfs::nodes::GEOMETRY_SOCKET;
                        title += "<div class=\"socket-input\" title=\"" + escape(input.description) + "\"><span class=\"socket-label" +
                                 std::string(has_widget ? " value-fallback" : "") + "\">" +
                                 escape(input.label) +
                                 (input.identifier == "Selection"
                                      ? "<span class=\"socket-optional\" title=\"" + escape(LOC("node_editor.optional_selection")) + "\"> ?</span>"
                                      : "") +
                                 "</span>";
                        if (has_widget)
                            title += "<div class=\"inline-value\">" + node_widgets::input(*node, input, true) +
                                     "</div>";
                        title += "</div>";
                    }
                    title += "</div><div class=\"node-footer\" data-preserve-content>—</div><div class=\"node-progress\"></div>";
                    if (node_markup_[visual.interaction.id] != title) {
                        node_widgets::patchMarkup(*element, title);
                        node_markup_[visual.interaction.id] = std::move(title);
                        node_layout_.erase(visual.interaction.id);
                    }
                    node_content_state_[visual.interaction.id] = std::move(content_state);
                }
                Rml::ElementList rows;
                const bool expanded = node_widgets::settingsExpanded(*node);
                if (auto* arrow = element->QuerySelector(".settings-arrow"))
                    arrow->SetClass("expanded", expanded);
                element->QuerySelectorAll(rows, ".node-property");
                for (auto* row : rows)
                    row->SetProperty("display", expanded ? "flex" : "none");
                rows.clear();
                element->QuerySelectorAll(rows, ".socket-input");
                for (std::size_t index = 0; index < inputs.size(); ++index) {
                    auto* row = rows[index];
                    row->SetProperty("display", node_widgets::inputVisible(*tree, *node, inputs[index]) ? "block" : "none");
                    row->SetClass("linked", node_widgets::linked(*tree, *node, inputs[index]));
                    const int count = node_widgets::inputRows(*tree, *node, inputs[index]);
                    if (row->GetAttribute<int>("data-rows", 0) != count) {
                        row->SetAttribute("data-rows", count);
                        row->SetProperty("height", px(count * 22.0f * dp_ratio_ * interaction_.zoom()));
                    }
                }
            }
        }
        updateSelectedClasses();
        updateNodePositions();
        updateSidebar();
    }

    void NodeCanvasElement::updateSelectedClasses() {
        for (const auto& [name, element] : node_elements_)
            element->SetClass("selected", selected_nodes_.contains(name));
        int layer = 0;
        for (const auto& node : interaction_.nodes()) {
            if (const auto found = node_elements_.find(node.id); found != node_elements_.end()) {
                auto* card = found->second;
                if (card->GetAttribute<int>("data-layer", -1) != layer) {
                    card->SetAttribute("data-layer", layer);
                    card->SetProperty("z-index", std::to_string(layer));
                }
            }
            ++layer;
        }
    }

    void NodeCanvasElement::updateNodePositions() {
        const auto size = GetBox().GetSize(Rml::BoxArea::Content);
        const float width = std::max(0.0f, size.x - (sidebar_visible_ ? kSidebarWidth * dp_ratio_ : 0.0f));
        if (canvas_width_ != width) {
            viewport_element_->SetProperty("width", px(width));
            canvas_width_ = width;
        }
        const auto pan = interaction_.pan();
        if (pan != layout_pan_) {
            nodes_element_->SetProperty("left", px(pan.x));
            nodes_element_->SetProperty("top", px(pan.y));
            layout_pan_ = pan;
        }
        const float zoom = interaction_.zoom();
        for (const auto& node : interaction_.nodes()) {
            const auto found = node_elements_.find(node.id);
            if (found == node_elements_.end())
                continue;
            const CanvasRect rect{node.bounds.x * zoom, node.bounds.y * zoom,
                                  node.bounds.width * zoom, node.bounds.height * zoom};
            const auto previous = node_layout_.find(node.id);
            const bool fresh = previous == node_layout_.end();
            if (fresh || previous->second.x != rect.x)
                found->second->SetProperty("left", px(rect.x));
            if (fresh || previous->second.y != rect.y)
                found->second->SetProperty("top", px(rect.y));
            if (fresh || layout_zoom_ != zoom || previous->second.width != rect.width ||
                previous->second.height != rect.height) {
                found->second->SetProperty("width", px(rect.width));
                found->second->SetProperty("height", px(rect.height));
                node_widgets::layoutCard(*found->second, zoom, dp_ratio_);
            }
            node_layout_[node.id] = rect;
        }
        layout_zoom_ = zoom;
    }

    void NodeCanvasElement::updateSidebar() {
        if (!sidebar_element_ || !scene_manager_ || !manager_)
            return;
        sidebar_element_->SetProperty("display", sidebar_visible_ ? "block" : "none");
        if (!sidebar_visible_) {
            updateEvaluationDom();
            return;
        }
        const auto host_uuid = activeHost();
        const auto* host = host_uuid ? scene_manager_->getScene().getNodeByUuid(*host_uuid) : nullptr;
        std::string html = "<div class=\"sidebar-section\"><div class=\"sidebar-heading\">" +
                           escape(LOC("node_editor.target")) + "</div>";
        if (host)
            html += "<div class=\"target-name\">" + escape(host->name) +
                    "</div><div class=\"target-type\">" + escape(sceneTypeLabel(host->type)) + "</div>";
        html += "</div><div class=\"sidebar-section\"><div class=\"sidebar-heading\">" +
                escape(LOC("node_editor.modifiers")) + "</div>";
        if (const auto* stack = host_uuid ? manager_->stack(*host_uuid) : nullptr) {
            for (const auto& modifier : stack->modifiers) {
                const std::string attributes = " data-modifier=\"" + escape(modifier.uuid) + "\"";
                const bool failed = std::ranges::any_of(evaluation_errors_, [&](const auto& error) {
                    return error.first == modifier.name || error.first.starts_with(modifier.name + "/");
                });
                const bool disabled = !modifier.enabled || !modifier.show_viewport;
                const std::string status = disabled ? LOC("node_editor.status_disabled")
                                           : failed ? LOC("node_editor.status_errors")
                                                    : LOC("node_editor.status_ok");
                html += "<div class=\"modifier-row" +
                        std::string(modifier.uuid == active_modifier_uuid_ ? " active" : "") +
                        "\" data-action=\"modifier\"" + attributes + ">"
                                                                     "<input type=\"checkbox\" data-action=\"modifier-enabled\"" +
                        attributes +
                        (modifier.enabled ? " checked" : "") + "/>"
                                                               "<div class=\"modifier-name\"><input class=\"modifier-rename\" type=\"text\" data-action=\"modifier-rename\"" +
                        attributes + " title=\"" + escape(modifier.name) + "\" value=\"" + escape(modifier.name) + "\"/>"
                                                                                                                   "<span class=\"modifier-name-label\">" +
                        escape(modifier.name) + "</span></div>"
                                                "<span class=\"modifier-status" +
                        std::string(disabled ? " disabled" : failed ? " failed"
                                                                    : "") +
                        "\" title=\"" + escape(status) + " · " +
                        std::format("{:.2f} ms", evaluation_total_ms_) + "\"></span>"
                                                                         "<button class=\"modifier-eye" +
                        std::string(modifier.show_viewport ? "" : " disabled") +
                        "\" data-action=\"modifier-eye\"" + attributes + " title=\"" +
                        escape(LOC("node_editor.show_viewport")) +
                        "\"><img sprite=\"icon-visible\"/></button>"
                        "<button class=\"modifier-menu\" data-action=\"modifier-menu\"" +
                        attributes +
                        " title=\"" + escape(LOC("node_editor.modifier_actions")) + "\">⋯</button></div>";
            }
        }
        html += "<button class=\"btn btn--secondary sidebar-button\" data-action=\"add-modifier\">+ " +
                escape(LOC("node_editor.add_modifier")) + "</button></div>";
        if (const auto* tree = activeTree(); tree && selected_nodes_.size() == 1) {
            if (const auto* node = tree->find_node(*selected_nodes_.begin())) {
                const auto type = manager_->registry().find_localized(node->type_id);
                html += "<div class=\"sidebar-section\"><div class=\"selected-node-name\">" +
                        escape(type ? type->label : node->type_id) + "</div>";
                if (type)
                    html += "<div class=\"node-description\">" + escape(type->description) + "</div>";
                html += "<div class=\"target-type\">" + escape(type ? type->category : node->type_id) + "</div>";
                const auto last_run = nodeStatus(node->name);
                html += "<div id=\"node-inspector-last-run\" class=\"node-eval-time\" data-preserve-content>" +
                        escape(node_widgets::nonBreakingStatus(std::vformat(LOC("node_editor.last_run"), std::make_format_args(last_run)))) + "</div>";
                html += "<label class=\"setting-row\"><span class=\"prop-label\">" +
                        escape(LOC("node_editor.node_on")) + "</span><input type=\"checkbox\" data-action=\"node-on\" data-node=\"" +
                        escape(node->name) + "\"" + (node->muted ? "" : " checked") + "/></label>";
                if (type) {
                    if (!type->help.empty()) {
                        const bool expanded = expanded_help_.try_emplace(type->id, true).first->second;
                        html += "<div class=\"node-help" + std::string(expanded ? " expanded" : "") +
                                "\"><button class=\"node-help-toggle\" data-action=\"node-help\" data-type=\"" +
                                escape(type->id) + "\"><span class=\"settings-arrow" + std::string(expanded ? " expanded" : "") +
                                "\"></span> " + escape(LOC("node_editor.how_to_use")) +
                                "</button><div class=\"node-help-text\">" + escape(type->help) + "</div></div>";
                    }
                    for (const auto& socket : type->inputs) {
                        if (socket.hide_value || socket.type == lfs::nodes::GEOMETRY_SOCKET ||
                            node_widgets::linked(*tree, *node, socket))
                            continue;
                        html += "<div class=\"node-setting-help\" title=\"" + escape(socket.description) + "\"><div class=\"setting-row node-setting\"><span class=\"prop-label\">" +
                                escape(socket.label) + "</span><div class=\"node-field-control\">" +
                                node_widgets::input(*node, socket, false) + "</div></div><div class=\"node-field-help\">" +
                                escape(socket.description) + "</div></div>";
                    }
                    for (const auto& property : type->properties) {
                        if (property.kind == lfs::nodes::PropertyKind::Data) {
                            if (node->type_id == "lfs.stored_selection") {
                                html += "<button class=\"btn sidebar-button\" data-action=\"capture-selection\" title=\"" + escape(property.description) + "\">" +
                                        escape(LOC("node_editor.capture_selection")) + "</button>";
                                const nlohmann::json* data = &node->properties;
                                if (const auto* stack = host_uuid ? manager_->stack(*host_uuid) : nullptr) {
                                    const auto modifier = std::ranges::find(stack->modifiers, active_modifier_uuid_, &Modifier::uuid);
                                    if (modifier != stack->modifiers.end())
                                        if (const auto stored = modifier->stored_selections.find(node->name); stored != modifier->stored_selections.end())
                                            data = &stored->second;
                                }
                                const auto count = data->value("selected_count", std::size_t{0});
                                html += "<div class=\"node-eval-time\">" + escape(std::vformat(LOC("node_editor.captured_count"), std::make_format_args(count))) + "</div>";
                            }
                            continue;
                        }
                        html += "<div class=\"node-setting-help\" title=\"" + escape(property.description) + "\"><div class=\"setting-row node-setting\"><span class=\"prop-label\">" +
                                escape(property.label) + "</span><div class=\"node-field-control\">" +
                                node_widgets::property(*node, property) + "</div></div><div class=\"node-field-help\">" +
                                escape(property.description) + "</div></div>";
                    }
                    if (!type->localization_key.empty())
                        html += "<button class=\"node-learn-more\" data-action=\"node-learn-more\" data-type=\"" +
                                escape(type->id) + "\">" + escape(LOC("node_editor.learn_more")) + "</button>";
                }
                const auto visual = std::ranges::find_if(visuals_, [&](const NodeVisual& item) {
                    return item.interaction.id == node->name;
                });
                html += "<div id=\"node-inspector-error\" class=\"node-error\" data-preserve-content>" +
                        escape(visual != visuals_.end() ? visual->error : "") + "</div>";
                html += "</div>";
            }
        }
        if (sidebar_markup_ != html) {
            node_widgets::patchMarkup(*sidebar_element_, html);
            sidebar_markup_ = std::move(html);
        }
        updateEvaluationDom();
    }

    void NodeCanvasElement::rebuildGeometry() {
        auto* renderer = GetRenderManager();
        if (!renderer)
            return;
        const auto size = GetBox().GetSize(Rml::BoxArea::Content);
        const float canvas_width = std::max(0.0f, size.x - (sidebar_visible_ ? kSidebarWidth * dp_ratio_ : 0.0f));
        interaction_.setViewport({0.0f, 0.0f, canvas_width, size.y});
        const auto& palette = theme().palette;
        const bool view_changed = geometry_pan_ != interaction_.pan() ||
                                  geometry_zoom_ != interaction_.zoom() || geometry_size_ != size ||
                                  geometry_theme_ != theme_signature_;
        if (view_changed) {
            Rml::Mesh grid;
            const float spacing = 24.0f * dp_ratio_ * interaction_.zoom();
            const float fade = std::clamp((interaction_.zoom() - 0.3f) / 0.2f, 0.0f, 1.0f);
            const auto minor = color(themeColor(palette.text), 0.10f * fade);
            const auto major = color(themeColor(palette.text), 0.17f * fade);
            const int first_column = static_cast<int>(std::floor(-interaction_.pan().x / spacing));
            const int first_row = static_cast<int>(std::floor(-interaction_.pan().y / spacing));
            for (int column = first_column; column * spacing + interaction_.pan().x < canvas_width; ++column)
                for (int row = first_row; row * spacing + interaction_.pan().y < size.y; ++row) {
                    const CanvasPoint point{column * spacing + interaction_.pan().x,
                                            row * spacing + interaction_.pan().y};
                    if (point.x >= 0.0f && point.y >= 0.0f)
                        gridDot(grid, point, 0.75f * dp_ratio_,
                                column % 5 == 0 && row % 5 == 0 ? major : minor);
                }
            grid_geometry_ = renderer->MakeGeometry(std::move(grid));
        }

        const auto route_generation = interaction_.routeGeneration();
        if (view_changed || geometry_route_generation_ != route_generation ||
            geometry_selected_link_ != selected_link_ || geometry_highlighted_link_ != interaction_.highlightedLink()) {
            Rml::Mesh wires;
            const auto& paths = interaction_.wirePaths();
            for (std::size_t index = 0; index < interaction_.links().size(); ++index) {
                const auto& link = interaction_.links()[index];
                const bool selected = selected_link_ && *selected_link_ == link;
                const bool highlighted = interaction_.highlightedLink() &&
                                         *interaction_.highlightedLink() == link;
                std::vector<CanvasPoint> points;
                points.reserve(paths[index].size());
                for (const auto point : paths[index])
                    points.push_back(interaction_.graphToScreen(point));
                pathRibbon(wires, points,
                           color(socketColor(link.from.type), 0.85f),
                           (selected || highlighted ? 3.5f : 2.5f) * dp_ratio_, dp_ratio_);
            }
            wire_geometry_ = renderer->MakeGeometry(std::move(wires));
            geometry_route_generation_ = route_generation;
            geometry_selected_link_ = selected_link_;
            geometry_highlighted_link_ = interaction_.highlightedLink();
        }
        Rml::Mesh live_wire;
        if (interaction_.draggingWire() && interaction_.wireSource()) {
            const auto& source = *interaction_.wireSource();
            const CanvasPoint end = interaction_.snappedSocket()
                                        ? interaction_.graphToScreen(interaction_.snappedSocket()->position)
                                        : interaction_.pointer();
            ribbon(live_wire, interaction_.graphToScreen(source.position), end,
                   color(socketColor(source.type), 0.85f), 2.5f * dp_ratio_, dp_ratio_);
        }
        live_wire_geometry_ = renderer->MakeGeometry(std::move(live_wire));

        if (view_changed || geometry_nodes_ != interaction_.nodes() ||
            geometry_snap_ != interaction_.snappedSocket() || geometry_dragging_wire_ != interaction_.draggingWire()) {
            for (const auto& node : interaction_.nodes()) {
                Rml::Mesh sockets;
                for (const auto& socket : node.sockets) {
                    const bool snapped = interaction_.snappedSocket() && *interaction_.snappedSocket() == socket;
                    const bool compatible = !interaction_.draggingWire() || interaction_.canSnapTo(socket);
                    const auto position = (socket.position - CanvasPoint{node.bounds.x, node.bounds.y}) * interaction_.zoom();
                    const float radius = (snapped ? 6.0f : 4.5f) * dp_ratio_ * interaction_.zoom();
                    const auto shape = [&](const float r, const Rml::ColourbPremultiplied fill) {
                        const float span = socket.multi_input ? 3.0f * dp_ratio_ * interaction_.zoom() : 0.0f;
                        circle(sockets, {position.x, position.y - span}, r, fill);
                        if (span > 0.0f) {
                            circle(sockets, {position.x, position.y + span}, r, fill);
                            quad(sockets, position.x - r, position.y - span, position.x + r, position.y + span, fill);
                        }
                    };
                    shape(radius + 1.5f * dp_ratio_ * interaction_.zoom(), color(themeColor(palette.background)));
                    shape(radius, color(socketColor(socket.type), snapped ? 1.0f : compatible ? 0.9f
                                                                                              : 0.2f));
                }
                if (const auto card = node_elements_.find(node.id); card != node_elements_.end())
                    static_cast<CanvasCard*>(card->second)->sockets = renderer->MakeGeometry(std::move(sockets));
            }
            geometry_nodes_ = interaction_.nodes();
            geometry_snap_ = interaction_.snappedSocket();
            geometry_dragging_wire_ = interaction_.draggingWire();
        }
        Rml::Mesh overlay;
        if (const auto box = interaction_.selectionBox()) {
            const auto fill = color(themeColor(palette.primary), 0.14f);
            const auto border = color(themeColor(palette.primary), 0.8f);
            quad(overlay, box->x, box->y, box->x + box->width, box->y + box->height, fill);
            quad(overlay, box->x, box->y, box->x + box->width, box->y + dp_ratio_, border);
            quad(overlay, box->x, box->y + box->height - dp_ratio_, box->x + box->width,
                 box->y + box->height, border);
            quad(overlay, box->x, box->y, box->x + dp_ratio_, box->y + box->height, border);
            quad(overlay, box->x + box->width - dp_ratio_, box->y, box->x + box->width,
                 box->y + box->height, border);
        }
        const auto& cut = interaction_.cutPoints();
        for (size_t index = 1; index < cut.size(); ++index)
            straightRibbon(overlay, cut[index - 1], cut[index],
                           color(themeColor(palette.error), 0.9f), 2.0f * dp_ratio_);
        overlay_geometry_ = renderer->MakeGeometry(std::move(overlay));
        geometry_pan_ = interaction_.pan();
        geometry_zoom_ = interaction_.zoom();
        geometry_size_ = size;
        geometry_theme_ = theme_signature_;
        geometry_dirty_ = false;
    }

    void NodeCanvasElement::OnRender() {
        // Stacking/render lists have already been built. Patching DOM here
        // invalidates them mid-traversal and can omit node cards from a frame.
        if (geometry_dirty_)
            rebuildGeometry();
        const auto offset = GetAbsoluteOffset(Rml::BoxArea::Content);
        grid_geometry_.Render(offset);
        wire_geometry_.Render(offset);
        live_wire_geometry_.Render(offset);
    }

    void NodeCanvasElement::OnUpdate() {
        repeatFieldStep();
        syncModel();
    }

    void NodeCanvasElement::OnResize() {
        geometry_dirty_ = true;
        dom_dirty_ = true;
    }

    CanvasPoint NodeCanvasElement::localPointer(const Rml::Event& event) {
        const auto offset = GetAbsoluteOffset(Rml::BoxArea::Content);
        return {event.GetParameter("mouse_x", 0.0f) - offset.x,
                event.GetParameter("mouse_y", 0.0f) - offset.y};
    }

    void NodeCanvasElement::focusCanvas() {
        Focus();
    }

    void NodeCanvasElement::executeCommands(const std::vector<CanvasCommand>& commands) {
        auto* tree = activeTree();
        const auto host = activeHost();
        if (!tree || !host || commands.empty())
            return;
        const bool selection_only = std::ranges::all_of(commands, [](const CanvasCommand& command) {
            return command.kind == CanvasCommandKind::Select || command.kind == CanvasCommandKind::SelectLink;
        });
        const auto before = selection_only ? nlohmann::json{} : tree->to_json();
        bool edited = false;
        bool selection_changed = false;
        for (const auto& command : commands) {
            if (command.kind == CanvasCommandKind::Select) {
                selection_changed = true;
                selected_nodes_.clear();
                selected_nodes_.insert(command.nodes.begin(), command.nodes.end());
                selected_link_.reset();
                continue;
            }
            if (command.kind == CanvasCommandKind::SelectLink && command.before) {
                selection_changed = true;
                selected_link_ = command.before;
                selected_nodes_.clear();
                continue;
            }
            if (command.kind == CanvasCommandKind::Move) {
                for (const auto& name : command.nodes)
                    if (auto* node = tree->find_node(name)) {
                        node->location[0] += command.delta.x / dp_ratio_;
                        node->location[1] += command.delta.y / dp_ratio_;
                        edited = true;
                    }
            } else if (command.kind == CanvasCommandKind::Connect && command.after) {
                const auto& link = *command.after;
                edited |= tree->add_link({link.from.node, link.from.identifier, link.to.node, link.to.identifier});
            } else if (command.kind == CanvasCommandKind::Disconnect && command.before) {
                const auto& link = *command.before;
                edited |= tree->remove_link({link.from.node, link.from.identifier, link.to.node, link.to.identifier});
            } else if (command.kind == CanvasCommandKind::ReRoute && command.before && command.after) {
                const auto& old = *command.before;
                const auto& next = *command.after;
                edited |= tree->remove_link({old.from.node, old.from.identifier, old.to.node, old.to.identifier});
                edited |= tree->add_link({next.from.node, next.from.identifier, next.to.node, next.to.identifier});
            } else if (command.kind == CanvasCommandKind::DeleteLinks) {
                for (const auto& link : command.links)
                    edited |= tree->remove_link({link.from.node, link.from.identifier, link.to.node, link.to.identifier});
            } else if (command.kind == CanvasCommandKind::Splice && command.before && !command.nodes.empty()) {
                auto* node = tree->find_node(command.nodes.front());
                const auto type = node ? manager_->registry().find_localized(node->type_id) : nullptr;
                if (!node || !type)
                    continue;
                const auto input = std::ranges::find_if(type->inputs, [&](const lfs::nodes::SocketDecl& socket) {
                    return lfs::nodes::can_convert_socket(command.before->from.type, socket.type);
                });
                const auto output = std::ranges::find_if(type->outputs, [&](const lfs::nodes::SocketDecl& socket) {
                    return lfs::nodes::can_convert_socket(socket.type, command.before->to.type);
                });
                if (input == type->inputs.end() || output == type->outputs.end())
                    continue;
                const auto old = *command.before;
                tree->remove_link({old.from.node, old.from.identifier, old.to.node, old.to.identifier});
                tree->add_link({old.from.node, old.from.identifier, node->name, input->identifier});
                tree->add_link({node->name, output->identifier, old.to.node, old.to.identifier});
                edited = true;
            }
        }
        if (edited) {
            manager_->recordTreeEdit(tree->uuid, before);
        }
        const bool layout_only = std::ranges::all_of(commands, [](const CanvasCommand& command) {
            return command.kind == CanvasCommandKind::Move || command.kind == CanvasCommandKind::Select ||
                   command.kind == CanvasCommandKind::SelectLink;
        });
        if (edited && !layout_only)
            dom_dirty_ = true;
        if (layout_only)
            last_generation_ = manager_->generation();
        if (selection_changed) {
            updateSelectedClasses();
            updateSidebar();
            updateSelectionPreview();
        }
        geometry_dirty_ = true;
    }

    void NodeCanvasElement::addNode(const std::string_view type_id, const CanvasPoint graph_position) {
        auto* tree = activeTree();
        const auto host = activeHost();
        if (!tree || !host)
            return;
        const auto before = tree->to_json();
        auto& node = tree->add_node(std::string(type_id));
        node.location = {graph_position.x / dp_ratio_, graph_position.y / dp_ratio_};
        selected_nodes_ = {node.name};
        manager_->recordTreeEdit(tree->uuid, before);
        dom_dirty_ = true;
    }

    void NodeCanvasElement::addModifier() {
        const auto host = activeHost();
        if (!host || !manager_)
            return;
        auto& tree = manager_->newTree(LOC("node_editor.default_tree_name"));
        auto& modifier = manager_->addModifier(*host, tree.uuid);
        active_tree_uuid_ = tree.uuid;
        active_modifier_uuid_ = modifier.uuid;
        dom_dirty_ = true;
    }

    void NodeCanvasElement::removeSelected() {
        auto* tree = activeTree();
        const auto host = activeHost();
        if (!tree || !host)
            return;
        const auto before = tree->to_json();
        bool changed = false;
        if (selected_link_) {
            const auto& link = *selected_link_;
            changed |= tree->remove_link({link.from.node, link.from.identifier, link.to.node, link.to.identifier});
            selected_link_.reset();
        }
        for (const auto& node : selected_nodes_)
            changed |= tree->remove_node(node);
        if (changed) {
            manager_->recordTreeEdit(tree->uuid, before);
            selected_nodes_.clear();
            dom_dirty_ = true;
        }
    }

    void NodeCanvasElement::duplicateSelected() {
        auto* tree = activeTree();
        const auto host = activeHost();
        if (!tree || !host || selected_nodes_.empty())
            return;
        const auto before = tree->to_json();
        std::unordered_set<std::string> duplicated;
        const auto original_nodes = tree->nodes;
        for (const auto& source : original_nodes) {
            if (!selected_nodes_.contains(source.name) || source.type_id == "lfs.group_input" ||
                source.type_id == "lfs.group_output")
                continue;
            auto& copy = tree->add_node(source.type_id, source.name);
            copy.location = {source.location[0] + 30.0f, source.location[1] + 30.0f};
            copy.input_values = source.input_values;
            copy.properties = source.properties;
            copy.muted = source.muted;
            duplicated.insert(copy.name);
        }
        if (!duplicated.empty()) {
            selected_nodes_ = std::move(duplicated);
            manager_->recordTreeEdit(tree->uuid, before);
            dom_dirty_ = true;
        }
    }

    void NodeCanvasElement::toggleMuted() {
        auto* tree = activeTree();
        const auto host = activeHost();
        if (!tree || !host || selected_nodes_.empty())
            return;
        const auto before = tree->to_json();
        for (const auto& name : selected_nodes_)
            if (auto* node = tree->find_node(name))
                node->muted = !node->muted;
        manager_->recordTreeEdit(tree->uuid, before);
        dom_dirty_ = true;
    }

    void NodeCanvasElement::frameAll() {
        if (visuals_.empty())
            return;
        float min_x = visuals_.front().interaction.bounds.x;
        float min_y = visuals_.front().interaction.bounds.y;
        float max_x = min_x + visuals_.front().interaction.bounds.width;
        float max_y = min_y + visuals_.front().interaction.bounds.height;
        for (const auto& visual : visuals_) {
            min_x = std::min(min_x, visual.interaction.bounds.x);
            min_y = std::min(min_y, visual.interaction.bounds.y);
            max_x = std::max(max_x, visual.interaction.bounds.x + visual.interaction.bounds.width);
            max_y = std::max(max_y, visual.interaction.bounds.y + visual.interaction.bounds.height);
        }
        const auto size = GetBox().GetSize(Rml::BoxArea::Content);
        const float width = std::max(1.0f, size.x - (sidebar_visible_ ? kSidebarWidth * dp_ratio_ : 0.0f));
        const float zoom = std::clamp(std::min((width - 80.0f * dp_ratio_) / (max_x - min_x),
                                               (size.y - 80.0f * dp_ratio_) / (max_y - min_y)),
                                      0.3f, 2.5f);
        interaction_.setView({width * 0.5f - (min_x + max_x) * 0.5f * zoom,
                              size.y * 0.5f - (min_y + max_y) * 0.5f * zoom},
                             zoom);
        rebuildModel();
        updateNodePositions();
        geometry_dirty_ = true;
    }

    bool NodeCanvasElement::arrange(const bool selection_only,
                                    const std::optional<std::unordered_set<std::string>>& nodes) {
        auto* tree = activeTree();
        if (!tree)
            return false;
        const auto subset = nodes.value_or(selection_only ? selected_nodes_ : std::unordered_set<std::string>{});
        for (const auto& name : subset)
            if (!tree->find_node(name))
                return false;
        if (nodes && nodes->empty())
            return true;
        const auto before = tree->to_json();
        const auto positions = interaction_.arrangedPositions(tree->input_node().name, tree->output_node().name, subset);
        for (auto& node : tree->nodes) {
            const auto found = positions.find(node.name);
            if (found != positions.end())
                node.location = {found->second.x / dp_ratio_, found->second.y / dp_ratio_};
        }
        manager_->recordTreeEdit(tree->uuid, before);
        rebuildModel();
        updateNodePositions();
        frame_pending_ = subset.empty();
        dom_dirty_ = true;
        geometry_dirty_ = true;
        return true;
    }

    bool NodeCanvasElement::handleKey(const int scancode, const bool shift, const bool control,
                                      const bool alt) {
        if (!editableMode())
            return false;
        if (scancode == SDL_SCANCODE_L && shift) {
            arrange();
            return true;
        }
        if (add_menu_) {
            if (scancode == SDL_SCANCODE_UP || scancode == SDL_SCANCODE_DOWN) {
                moveAddHighlight(scancode == SDL_SCANCODE_UP ? -1 : 1);
                return true;
            }
            if (scancode == SDL_SCANCODE_ESCAPE) {
                closeAddMenu();
                return true;
            }
            if (scancode == SDL_SCANCODE_RETURN || scancode == SDL_SCANCODE_KP_ENTER) {
                const auto id = first_add_type_;
                closeAddMenu();
                if (!id.empty())
                    addNode(id, add_position_);
                return true;
            }
            return false;
        }
        if (active_field_) {
            if (scancode == SDL_SCANCODE_ESCAPE) {
                finishFieldEdit(true);
                return true;
            }
            return false;
        }
        if (auto* context = GetContext()) {
            const auto* focused = context->GetFocusElement();
            if (focused && (focused->GetTagName() == "input" || focused->GetTagName() == "select"))
                return false;
        }
        switch (scancode) {
        case SDL_SCANCODE_ESCAPE:
            interaction_.cancel();
            pointer_down_ = false;
            selected_nodes_ = interaction_.selectedNodes();
            updateSelectedClasses();
            updateNodePositions();
            geometry_dirty_ = true;
            return true;
        case SDL_SCANCODE_DELETE:
        case SDL_SCANCODE_X:
            removeSelected();
            return true;
        case SDL_SCANCODE_M:
            toggleMuted();
            return true;
        case SDL_SCANCODE_D:
            if (shift) {
                duplicateSelected();
                return true;
            }
            break;
        case SDL_SCANCODE_A:
            if (shift) {
                const auto offset = GetAbsoluteOffset(Rml::BoxArea::Content);
                openAddMenu(offset.x + interaction_.pointer().x, offset.y + interaction_.pointer().y);
                return true;
            }
            break;
        case SDL_SCANCODE_HOME:
            frameAll();
            return true;
        case SDL_SCANCODE_N:
            setSidebarVisible(!sidebar_visible_);
            return true;
        default: break;
        }
        (void)control;
        (void)alt;
        return false;
    }

    void NodeCanvasElement::ProcessEvent(Rml::Event& event) {
        // Colour elements change their value in the target's default action,
        // before the canvas default action. Capture the undo snapshot first.
        if (event.GetType() == "mousedown") {
            auto* target = event.GetTargetElement();
            if (editableMode() && event.GetParameter("button", 0) == 0 &&
                (target->GetTagName() == "colour-offset" || target->GetTagName() == "color-picker")) {
                active_field_ = target;
                if (const auto* tree = activeTree())
                    field_before_ = tree->to_json();
                field_step_direction_ = 0;
            }
            return;
        }
        if (!editableMode() || processAddMenuEvent(event) || processHelpEvent(event) || processFieldEvent(event))
            return;
        // RmlUi mousemove has no bubbling default action. Live gestures must
        // consume motion as listeners, not wait for ProcessDefaultAction/up.
        if ((event.GetType() == "mousemove" || event.GetType() == "drag") && pointer_down_) {
            const auto delta = localPointer(event) - context_press_;
            if (std::abs(delta.x) + std::abs(delta.y) >= 4.0f)
                pending_context_menu_ = false;
            (void)interaction_.pointerMove(localPointer(event));
            selected_nodes_ = interaction_.selectedNodes();
            updateSelectedClasses();
            updateNodePositions();
            geometry_dirty_ = true;
            event.StopPropagation();
            return;
        }
        if (event.GetType() == "change" || event.GetType() == "blur") {
            processControlEvent(event);
            return;
        }
        if (event.GetType() != "mousescroll" && event.GetType() != "pinch")
            return;
        const auto pointer = localPointer(event);
        if (sidebar_visible_ && pointer.x >= GetBox().GetSize(Rml::BoxArea::Content).x - kSidebarWidth * dp_ratio_)
            return;
        auto* window = services().windowOrNull();
        auto* controller = window ? window->inputController() : nullptr;
        const auto preferences = controller ? controller->trackpadPreferences() : TrackpadPreferenceState{};
        const auto previous_zoom = interaction_.zoom();
        if (event.GetType() == "pinch")
            interaction_.pinch(localPointer(event), event.GetParameter("scale", 1.0f), preferences.zoom_speed);
        else
            interaction_.scroll(localPointer(event), {-event.GetParameter("wheel_delta_x", 0.0f), -event.GetParameter("wheel_delta_y", 0.0f)},
                                preferences, controller ? controller->trackpadTouchCount() : 0,
                                event.GetParameter<int>("ctrl_key", 0) != 0 || event.GetParameter<int>("meta_key", 0) != 0,
                                controller ? controller->wheelZoomSpeed() : 11.0f);
        if (interaction_.zoom() != previous_zoom)
            rebuildModel();
        updateNodePositions();
        geometry_dirty_ = true;
        event.StopPropagation();
    }

    void NodeCanvasElement::processControlEvent(Rml::Event& event) {
        const std::string type = event.GetType();
        Rml::Element* target = event.GetTargetElement();
        if (!target)
            return;
        const std::string action = target->GetAttribute<Rml::String>("data-action", "");
        if ((type == "change" || type == "blur") && target) {
            if (action == "node-on" && type == "change") {
                auto* tree = activeTree();
                auto* node = tree ? tree->find_node(target->GetAttribute<Rml::String>("data-node", "")) : nullptr;
                if (node) {
                    const auto before = tree->to_json();
                    node->muted = !target->HasAttribute("checked");
                    manager_->recordTreeEdit(tree->uuid, before);
                }
                event.StopPropagation();
                return;
            }
            const auto* text_input = dynamic_cast<Rml::ElementFormControlInput*>(target);
            const bool text_field = text_input && target->GetAttribute<Rml::String>("type", "") == "text";
            if (text_field && type == "change" && !event.GetParameter("linebreak", false))
                return;
            if (type == "blur" && !text_field)
                return;
            if (action == "modifier-enabled") {
                const auto host = activeHost();
                auto* stack = host ? manager_->stack(*host) : nullptr;
                const std::string uuid = target->GetAttribute<Rml::String>("data-modifier", "");
                const auto found = stack ? std::ranges::find(stack->modifiers, uuid, &Modifier::uuid)
                                         : std::vector<Modifier>::iterator{};
                if (host && stack && found != stack->modifiers.end()) {
                    const nlohmann::json before = stack->modifiers;
                    const bool checked = target->HasAttribute("checked");
                    found->enabled = checked;
                    manager_->recordStackEdit(*host, before);
                    dom_dirty_ = true;
                }
                event.StopPropagation();
                return;
            }
            if (action == "modifier-rename") {
                const auto host = activeHost();
                auto* stack = host ? manager_->stack(*host) : nullptr;
                const std::string uuid = target->GetAttribute<Rml::String>("data-modifier", "");
                const auto found = stack ? std::ranges::find(stack->modifiers, uuid, &Modifier::uuid)
                                         : std::vector<Modifier>::iterator{};
                if (host && stack && found != stack->modifiers.end()) {
                    if (const auto* input = dynamic_cast<Rml::ElementFormControlInput*>(target))
                        (void)manager_->renameModifier(*host, found->name, input->GetValue());
                    dom_dirty_ = true;
                }
                event.StopPropagation();
                return;
            }
            commitControl(target, &event, active_field_ != target);
            event.StopPropagation();
            return;
        }
    }

    bool NodeCanvasElement::isPanPress(const Rml::Event& event) const {
        auto* window = services().windowOrNull();
        auto* controller = window ? window->inputController() : nullptr;
        if (!controller)
            return false;
        const int button = event.GetParameter("button", 0);
        const auto mouse_button = button == 1 ? input::MouseButton::RIGHT : button == 2 ? input::MouseButton::MIDDLE
                                                                                        : input::MouseButton::LEFT;
        int modifiers = input::MODIFIER_NONE;
        if (event.GetParameter<int>("shift_key", 0))
            modifiers |= input::MODIFIER_SHIFT;
        if (event.GetParameter<int>("ctrl_key", 0))
            modifiers |= input::MODIFIER_CTRL;
        if (event.GetParameter<int>("meta_key", 0))
            modifiers |= input::MODIFIER_SUPER;
        if (event.GetParameter<int>("alt_key", 0))
            modifiers |= input::MODIFIER_ALT;
        return controller->isPanDrag(mouse_button, modifiers);
    }

    void NodeCanvasElement::openCanvasMenu(const CanvasPoint pointer) {
        const auto offset = GetAbsoluteOffset(Rml::BoxArea::Content);
        const auto graph_pointer = interaction_.screenToGraph(pointer);
        const bool over_node = std::ranges::any_of(interaction_.nodes(), [&](const auto& node) { return node.bounds.contains(graph_pointer); });
        if (over_node && context_menu_) {
            context_menu_->request({{.label = LOC("node_editor.arrange"), .action = "arrange", .shortcut = LOC("node_editor.arrange_shortcut")},
                                    {.label = LOC("node_editor.add"), .action = "add"}},
                                   panel_screen_offset_.x + pointer.x, panel_screen_offset_.y + pointer.y,
                                   [this, pointer, offset](const std::string_view action) {
                                       if (action == "arrange")
                                           arrange();
                                       else if (action == "add")
                                           openAddMenu(offset.x + pointer.x, offset.y + pointer.y);
                                   });
        } else {
            openAddMenu(offset.x + pointer.x, offset.y + pointer.y);
        }
    }

    void NodeCanvasElement::ProcessDefaultAction(Rml::Event& event) {
        Element::ProcessDefaultAction(event);
        if (processAddMenuEvent(event) || processFieldEvent(event))
            return;
        const std::string type = event.GetType();
        Rml::Element* target = event.GetTargetElement();
        std::string action;
        for (auto* element = target; element && element != this; element = element->GetParentNode()) {
            action = element->GetAttribute<Rml::String>("data-action", "");
            if (!action.empty()) {
                target = element;
                break;
            }
        }
        if (type == "click" && target) {
            if (action == "node-settings") {
                if (auto* tree = activeTree()) {
                    if (auto* node = tree->find_node(target->GetAttribute<Rml::String>("data-node", ""))) {
                        const auto before = tree->to_json();
                        node->ui["settings_expanded"] = !node_widgets::settingsExpanded(*node);
                        manager_->recordTreeEdit(tree->uuid, before);
                    }
                }
                event.StopPropagation();
                return;
            }
            if (action == "add-modifier") {
                openModifierLibrary(event.GetParameter("mouse_x", 0.0f), event.GetParameter("mouse_y", 0.0f));
                event.StopPropagation();
                return;
            }
            if (action == "modifier") {
                const std::string uuid = target->GetAttribute<Rml::String>("data-modifier", "");
                if (const auto host = activeHost()) {
                    if (const auto* stack = manager_->stack(*host)) {
                        const auto found = std::ranges::find(stack->modifiers, uuid, &Modifier::uuid);
                        if (found != stack->modifiers.end()) {
                            active_modifier_uuid_ = found->uuid;
                            active_tree_uuid_ = found->tree_uuid;
                            dom_dirty_ = true;
                        }
                    }
                }
                event.StopPropagation();
                return;
            }
            if (action == "modifier-menu") {
                openModifierMenu(target->GetAttribute<Rml::String>("data-modifier", ""),
                                 event.GetParameter("mouse_x", 0.0f), event.GetParameter("mouse_y", 0.0f));
                event.StopPropagation();
                return;
            }
            if (action == "modifier-eye") {
                const auto host = activeHost();
                auto* stack = host ? manager_->stack(*host) : nullptr;
                if (stack) {
                    const auto found = std::ranges::find(stack->modifiers,
                                                         target->GetAttribute<Rml::String>("data-modifier", ""), &Modifier::uuid);
                    if (found != stack->modifiers.end()) {
                        const nlohmann::json before = stack->modifiers;
                        found->show_viewport = !found->show_viewport;
                        manager_->recordStackEdit(*host, before);
                        dom_dirty_ = true;
                    }
                }
                event.StopPropagation();
                return;
            }
            if (action == "capture-selection") {
                const auto host = activeHost();
                auto* stack = host ? manager_->stack(*host) : nullptr;
                const auto modifier = stack ? std::ranges::find(stack->modifiers, active_modifier_uuid_,
                                                                &Modifier::uuid)
                                            : std::vector<Modifier>::iterator{};
                if (host && stack && modifier != stack->modifiers.end() &&
                    selected_nodes_.size() == 1)
                    (void)manager_->captureSelection(*host, modifier->name,
                                                     *selected_nodes_.begin());
                dom_dirty_ = true;
                event.StopPropagation();
                return;
            }
        }
        if (!editableMode())
            return;
        if (type == "mousedown") {
            if (action == "node-settings") {
                event.StopPropagation();
                return;
            }
            focusCanvas();
            const int button = event.GetParameter("button", 0);
            const auto pointer = localPointer(event);
            const auto size = GetBox().GetSize(Rml::BoxArea::Content);
            if (sidebar_visible_ && pointer.x >= size.x - kSidebarWidth * dp_ratio_)
                return;
            CanvasPointerButton canvas_button = CanvasPointerButton::Left;
            if (button == 1)
                canvas_button = CanvasPointerButton::Right;
            else if (button == 2)
                canvas_button = CanvasPointerButton::Middle;
            const CanvasModifiers modifiers{
                .shift = event.GetParameter<int>("shift_key", 0) != 0,
                .control = event.GetParameter<int>("ctrl_key", 0) != 0 || event.GetParameter<int>("meta_key", 0) != 0,
                .alt = event.GetParameter<int>("alt_key", 0) != 0,
            };
            const bool pan_drag = isPanPress(event);
            pending_context_menu_ = button == 1 && pan_drag && !modifiers.control;
            context_press_ = pointer;
            executeCommands(interaction_.pointerDown(pointer, canvas_button, modifiers, pan_drag));
            pointer_down_ = interaction_.active();
            if (button == 1 && !modifiers.control && !pan_drag)
                openCanvasMenu(pointer);
            event.StopPropagation();
        } else if ((type == "mouseup" || type == "dragend") && pointer_down_) {
            executeCommands(interaction_.pointerUp(localPointer(event)));
            selected_nodes_ = interaction_.selectedNodes();
            pointer_down_ = false;
            updateNodePositions();
            geometry_dirty_ = true;
            if (pending_context_menu_) {
                pending_context_menu_ = false;
                openCanvasMenu(localPointer(event));
            }
            event.StopPropagation();
        } else if (type == "keydown") {
            const auto key = static_cast<Rml::Input::KeyIdentifier>(event.GetParameter("key_identifier", 0));
            int scancode = SDL_SCANCODE_UNKNOWN;
            if (key == Rml::Input::KI_ESCAPE)
                scancode = SDL_SCANCODE_ESCAPE;
            else if (key == Rml::Input::KI_DELETE)
                scancode = SDL_SCANCODE_DELETE;
            else if (key == Rml::Input::KI_X)
                scancode = SDL_SCANCODE_X;
            else if (key == Rml::Input::KI_M)
                scancode = SDL_SCANCODE_M;
            else if (key == Rml::Input::KI_D)
                scancode = SDL_SCANCODE_D;
            else if (key == Rml::Input::KI_A)
                scancode = SDL_SCANCODE_A;
            else if (key == Rml::Input::KI_HOME)
                scancode = SDL_SCANCODE_HOME;
            else if (key == Rml::Input::KI_N)
                scancode = SDL_SCANCODE_N;
            else if (key == Rml::Input::KI_L)
                scancode = SDL_SCANCODE_L;
            if (handleKey(scancode, event.GetParameter<int>("shift_key", 0) != 0,
                          event.GetParameter<int>("ctrl_key", 0) != 0,
                          event.GetParameter<int>("alt_key", 0) != 0))
                event.StopPropagation();
        }
    }

} // namespace lfs::vis::gui
