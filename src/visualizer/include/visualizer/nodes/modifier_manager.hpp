/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/nodes/nodes.hpp"
#include "core/scene.hpp"
#include "visualizer/nodes/modifier_error.hpp"

#include <nlohmann/json.hpp>

#include <array>
#include <expected>
#include <glm/glm.hpp>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace lfs::vis {

    class SceneManager;
    class SequencerController;
    class ModifierEvaluationWorker;
    struct ModifierEvaluationRequest;
    struct ModifierHostResult;
    struct ModifierWorkerProgress {
        bool busy = false;
        core::Uuid host;
        std::string modifier;
        std::string node;
        std::string label;
        std::size_t completed = 0;
        std::size_t total = 0;
        std::uint64_t generation = 0;
        std::chrono::steady_clock::time_point started_at;
        std::unordered_set<std::string> pending_nodes;
        std::unordered_map<std::string, lfs::nodes::NodeEvaluation> finished_nodes;
    };

    struct Modifier {
        std::string uuid;
        std::string name;
        std::string tree_uuid;
        bool enabled = true;
        bool show_viewport = true;
        std::unordered_map<std::string, lfs::nodes::Value> input_overrides;
        std::unordered_map<std::string, nlohmann::json> stored_selections;
    };

    struct ModifierStack {
        std::vector<Modifier> modifiers;
    };

    struct ModifierEvaluation {
        bool ok = true;
        bool unchanged = false;
        lfs::nodes::Geometry geometry;
        std::unordered_map<std::string, std::string> errors;
        std::unordered_map<std::string, double> time_ms;
        double total_time_ms = 0.0;
        std::unordered_map<std::string, lfs::nodes::NodeEvaluation> nodes;
    };

    struct PasteNodesResult {
        std::vector<std::string> nodes;
        std::size_t dropped_links = 0;
    };

    struct MakeGroupResult {
        std::string group_node;
        std::string graph;
    };

    enum class NodeViewportGizmoKind {
        Box,
        Ellipsoid,
        Transform,
    };

    struct NodeViewportGizmo {
        core::Uuid host;
        core::NodeId host_id = core::NULL_NODE;
        std::string tree_uuid;
        std::string node;
        NodeViewportGizmoKind kind = NodeViewportGizmoKind::Box;
        bool editable = false;
        glm::mat4 world_transform{1.0f};
        glm::mat4 local_transform{1.0f};
        glm::vec3 local_translation{0.0f};
        glm::vec3 local_rotation{0.0f};
        glm::vec3 local_scale{1.0f};
        float falloff = 0.0f;
    };

    struct PaintStrokeSample {
        glm::vec3 position{0.0f};
        float radius = 0.0f;
        float value = 1.0f;
    };

    struct HsvPickBands {
        float hue = 0.0f;
        float saturation_min = 0.0f;
        float saturation_max = 1.0f;
        float value_min = 0.0f;
        float value_max = 1.0f;
    };

    LFS_VIS_API HsvPickBands centreHsvPickBands(glm::vec3 rgb, float saturation_width,
                                                float value_width);

    class LFS_VIS_API ModifierManager final {
    public:
        explicit ModifierManager(SceneManager& scene_manager);
        ~ModifierManager();

        ModifierManager(const ModifierManager&) = delete;
        ModifierManager& operator=(const ModifierManager&) = delete;

        [[nodiscard]] lfs::nodes::NodeTypeRegistry& registry() noexcept;
        [[nodiscard]] const lfs::nodes::NodeTypeRegistry& registry() const noexcept;

        lfs::nodes::NodeTree& newTree(std::string name);
        lfs::nodes::NodeTree& loadTree(const nlohmann::json& json);
        [[nodiscard]] lfs::nodes::NodeTree* tree(std::string_view uuid_or_name);
        [[nodiscard]] const lfs::nodes::NodeTree* tree(std::string_view uuid_or_name) const;
        [[nodiscard]] std::vector<lfs::nodes::NodeTree*> trees();
        bool removeTree(std::string_view uuid_or_name);
        [[nodiscard]] std::string uniqueTreeName(std::string name, std::string_view except_uuid = {}) const;
        [[nodiscard]] std::expected<std::string, ModifierError>
        copyNodes(std::string_view tree_uuid, const std::vector<std::string>& nodes) const;
        [[nodiscard]] std::expected<PasteNodesResult, ModifierError>
        pasteNodes(std::string_view tree_uuid, std::string_view clipboard,
                   std::optional<std::array<float, 2>> location = std::nullopt);
        [[nodiscard]] std::expected<MakeGroupResult, ModifierError>
        makeGroup(std::string_view tree_uuid, const std::vector<std::string>& nodes,
                  std::string name = "Group");
        [[nodiscard]] std::expected<std::vector<std::string>, ModifierError>
        ungroup(std::string_view tree_uuid, std::string_view node);
        [[nodiscard]] ModifierResult setGroupGraph(std::string_view tree_uuid,
                                                   std::string_view node,
                                                   std::string_view graph);
        [[nodiscard]] ModifierResult makeGroupSingleUser(std::string_view tree_uuid,
                                                         std::string_view node);
        [[nodiscard]] std::expected<std::string, ModifierError>
        interfaceAdd(std::string_view tree_uuid, bool output, std::string type,
                     std::string label, lfs::nodes::Value default_value = {},
                     std::optional<double> min = {}, std::optional<double> max = {},
                     std::optional<double> step = {});
        [[nodiscard]] ModifierResult interfaceRemove(std::string_view tree_uuid, bool output,
                                                     std::string_view identifier);
        [[nodiscard]] ModifierResult interfaceUpdate(std::string_view tree_uuid, bool output,
                                                     std::string_view identifier,
                                                     const nlohmann::json& changes);
        [[nodiscard]] ModifierResult interfaceMove(std::string_view tree_uuid, bool output,
                                                   std::string_view identifier, std::size_t index);
        [[nodiscard]] std::expected<std::string, ModifierError>
        frameWrap(std::string_view tree_uuid, const std::vector<std::string>& nodes,
                  std::string label = "Frame");
        [[nodiscard]] ModifierResult frameSetMembers(std::string_view tree_uuid,
                                                     std::string_view frame,
                                                     const std::vector<std::string>& nodes);
        [[nodiscard]] std::expected<std::string, ModifierError>
        rerouteInsert(std::string_view tree_uuid, const lfs::nodes::Link& link,
                      std::optional<std::array<float, 2>> location = std::nullopt);
        [[nodiscard]] ModifierResult setNodeInput(std::string_view tree_uuid, std::string_view node_name,
                                                  std::string_view input, lfs::nodes::Value value);
        [[nodiscard]] ModifierResult captureViewportCamera(std::string_view tree_uuid, std::string_view node_name);
        [[nodiscard]] ModifierResult renameNode(std::string_view tree_uuid, std::string_view node, std::string name);
        [[nodiscard]] ModifierResult keyframeSet(std::string_view tree_uuid, std::string_view node,
                                                 std::string_view input, std::optional<float> time = {},
                                                 std::optional<lfs::nodes::Value> value = {}, int easing = 0);
        [[nodiscard]] ModifierResult keyframeRemove(std::string_view tree_uuid, std::string_view node,
                                                    std::string_view input, std::optional<float> time = {});
        [[nodiscard]] SequencerController* sequencer() const;
        // Explicit binding supports headless controllers; the GUI uses its registered sequencer.
        void setSequencer(SequencerController* controller) { sequencer_ = controller; }
        [[nodiscard]] float animationTime() const;
        [[nodiscard]] nlohmann::json animationJson() const;
        [[nodiscard]] bool timeDependent(const core::Uuid& host = {}) const;
        [[nodiscard]] ModifierResult evaluateAtTime(float seconds, float fps);

        Modifier& addModifier(const core::Uuid& node_uuid, std::string tree_uuid,
                              std::string name = {});
        bool renameModifier(const core::Uuid& node_uuid, std::string_view modifier_name,
                            std::string new_name);
        bool removeModifier(const core::Uuid& node_uuid, std::string_view modifier_name);
        bool moveModifier(const core::Uuid& node_uuid, std::string_view modifier_name, size_t index);
        [[nodiscard]] ModifierStack* stack(const core::Uuid& node_uuid);
        [[nodiscard]] const ModifierStack* stack(const core::Uuid& node_uuid) const;

        void markDirty(const core::Uuid& node_uuid = {});
        void tick();
        [[nodiscard]] bool trainingSuspended() const noexcept { return training_suspended_; }
        void recordCanvasFrame(double milliseconds);
        void recordViewerFrame(double milliseconds, bool rendered_viewport);
        [[nodiscard]] nlohmann::json performance(bool reset = false);
        [[nodiscard]] const ModifierEvaluation* lastResult(const core::Uuid& node_uuid) const;
        [[nodiscard]] std::uint64_t resultGeneration() const { return result_generation_; }
        [[nodiscard]] ModifierWorkerProgress progress() const;
        ModifierEvaluation evaluate(const core::Uuid& node_uuid);
        [[nodiscard]] std::optional<lfs::nodes::Geometry> evaluated(const core::Uuid& node_uuid) const;
        [[nodiscard]] std::optional<core::Tensor>
        selectionPreview(const core::Uuid& node_uuid, std::string_view modifier_uuid,
                         std::string_view node_name);
        [[nodiscard]] ModifierResult
        captureSelection(const core::Uuid& node_uuid, std::string_view modifier_name,
                         std::string_view stored_selection_node);
        [[nodiscard]] ModifierResult
        applyModifier(const core::Uuid& node_uuid, std::string_view modifier_name);

        [[nodiscard]] nlohmann::json toJson(bool existing_nodes_only = true) const;
        [[nodiscard]] ModifierResult restoreJson(const nlohmann::json& json);
        void clear();

        void recordTreeEdit(std::string_view tree_uuid, nlohmann::json before,
                            std::string merge_key = {}, bool reevaluate = true);
        void recordStackEdit(const core::Uuid& node_uuid, nlohmann::json before,
                             std::string merge_key = {});
        void recordLibraryEdit(nlohmann::json before, std::string merge_key = {});

        std::uint64_t generation() const;

        // The node canvas publishes only its current selection. Viewport tools
        // consume this shared state and clear it when the editor is hidden.
        void setViewportNodeSelection(const core::Uuid& host, std::string tree_uuid,
                                      std::string node_name, bool editor_visible = true);
        void setViewportEditorVisible(bool visible);
        void clearViewportNodeSelection();
        [[nodiscard]] std::optional<NodeViewportGizmo> viewportNodeGizmo() const;
        bool beginViewportNodeGizmoDrag();
        bool updateViewportNodeGizmo(const glm::mat4& world_transform);
        void endViewportNodeGizmoDrag(bool cancel = false);

        [[nodiscard]] bool paintModeActive() const noexcept { return paint_mode_; }
        [[nodiscard]] float paintRadius() const noexcept { return paint_radius_; }
        bool setPaintMode(bool enabled);
        void adjustPaintRadius(float factor);
        bool beginPaintStroke();
        bool appendPaintSample(const PaintStrokeSample& sample, bool position_is_world = true);
        void endPaintStroke(bool cancel = false);
        [[nodiscard]] ModifierResult addPaintStroke(const std::vector<PaintStrokeSample>& samples,
                                                    bool positions_are_world = false);
        [[nodiscard]] ModifierResult clearPaintStrokes();

        bool beginColourPick(std::string node, std::string input, bool widen_hue = false);
        void cancelViewportMode();
        [[nodiscard]] bool colourPickActive() const noexcept { return colour_pick_.has_value(); }
        [[nodiscard]] ModifierResult applyPickedColour(glm::vec3 colour, bool widen_hue = false);

    private:
        struct RuntimeState {
            ModifierEvaluation evaluation;
            bool dirty = true;
            std::unordered_map<std::string, core::Tensor> previews;
        };

        [[nodiscard]] ModifierHostResult evaluateForApply(const core::Uuid& node_uuid, size_t last_modifier);
        void clearEvaluatedPayloads();
        [[nodiscard]] Modifier* findModifier(const core::Uuid& node_uuid,
                                             std::string_view name);
        void registerVisualizerNodes();
        [[nodiscard]] ModifierEvaluationRequest captureRequest() const;
        void installReady();
        void updateAnimationTime();

        SceneManager* scene_manager_ = nullptr;
        SequencerController* sequencer_ = nullptr;
        std::optional<float> export_time_;
        float export_fps_ = 24.0f;
        float last_animation_time_ = 0.0f;
        float last_animation_fps_ = 24.0f;
        std::uint64_t last_animation_revision_ = 0;
        bool last_time_dependent_ = false;
        bool animation_only_request_ = false;
        lfs::nodes::NodeTypeRegistry registry_;
        std::unordered_map<std::string, std::unique_ptr<lfs::nodes::NodeTree>> trees_;
        std::unordered_map<core::Uuid, ModifierStack> stacks_;
        std::unordered_map<core::Uuid, RuntimeState> runtime_;
        std::uint64_t generation_ = 1;
        std::uint64_t last_scene_generation_ = 0;
        bool training_suspended_ = false;
        bool restoring_ = false;
        std::vector<double> canvas_frame_ms_;
        std::vector<double> evaluation_latency_ms_;
        std::vector<double> viewer_frame_ms_;
        std::vector<double> busy_viewport_frame_ms_;
        std::vector<double> idle_viewport_frame_ms_;
        bool measure_canvas_ = false;
        std::optional<double> canvas_work_ms_;
        std::unique_ptr<ModifierEvaluationWorker> worker_;
        std::uint64_t output_generation_ = 1;
        std::uint64_t source_generation_ = 1;
        std::uint64_t requested_generation_ = 0;
        std::uint64_t result_generation_ = 0;
        std::uint64_t installed_count_ = 0;
        std::uint64_t last_installed_generation_ = 0;
        std::uint64_t progress_event_generation_ = 0;
        std::string progress_event_node_;
        std::thread::id viewer_thread_;

        struct ViewportSelection {
            core::Uuid host;
            std::string tree_uuid;
            std::string node;
            bool editor_visible = false;
        };
        struct ColourPick {
            std::string node;
            std::string input;
            bool widen_hue = false;
        };
        std::optional<ViewportSelection> viewport_selection_;
        std::optional<ColourPick> colour_pick_;
        bool paint_mode_ = false;
        // Viewport brush radius in logical screen pixels. Samples convert this
        // to world and then host-local units at the picked surface depth.
        float paint_radius_ = 25.0f;
        std::optional<nlohmann::json> gizmo_before_;
        std::optional<nlohmann::json> paint_before_;
    };

    LFS_VIS_API void to_json(nlohmann::json& json, const Modifier& modifier);
    LFS_VIS_API void from_json(const nlohmann::json& json, Modifier& modifier);

} // namespace lfs::vis
