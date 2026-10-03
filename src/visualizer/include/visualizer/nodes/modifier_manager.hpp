/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/nodes/nodes.hpp"
#include "core/scene.hpp"

#include <nlohmann/json.hpp>

#include <expected>
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

    struct ModifierError {
        std::string message;
    };

    using ModifierResult = std::expected<void, ModifierError>;

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
        [[nodiscard]] ModifierResult setNodeInput(std::string_view tree_uuid, std::string_view node_name,
                                                  std::string_view input, lfs::nodes::Value value);

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

        std::uint64_t generation() const;

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

        SceneManager* scene_manager_ = nullptr;
        lfs::nodes::NodeTypeRegistry registry_;
        std::unordered_map<std::string, std::unique_ptr<lfs::nodes::NodeTree>> trees_;
        std::unordered_map<core::Uuid, ModifierStack> stacks_;
        std::unordered_map<core::Uuid, RuntimeState> runtime_;
        std::uint64_t generation_ = 1;
        std::uint64_t last_scene_generation_ = 0;
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
        std::uint64_t progress_event_generation_ = 0;
        std::string progress_event_node_;
        std::thread::id viewer_thread_;
    };

    LFS_VIS_API void to_json(nlohmann::json& json, const Modifier& modifier);
    LFS_VIS_API void from_json(const nlohmann::json& json, Modifier& modifier);

} // namespace lfs::vis
