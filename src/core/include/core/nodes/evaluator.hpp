/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/nodes/device.hpp"
#include "core/nodes/tree.hpp"

#include <chrono>
#include <functional>
#include <optional>
#include <stdexcept>
#include <unordered_map>

namespace lfs::nodes {

    enum class TransformSpace { Original,
                                Relative };

    struct EvaluationCamera {
        std::string name;
        glm::mat4 world_to_camera{1.0f}; // dataset convention: +Z forward, +Y down
        float focal_x = 1, focal_y = 1, center_x = 0, center_y = 0;
        int width = 1, height = 1;
    };

    class LFS_CORE_API EvalHost {
    public:
        virtual ~EvalHost() = default;
        virtual std::uint64_t generation() const = 0;
        virtual std::optional<Geometry> object_geometry(std::string_view name, TransformSpace space) = 0;
        virtual std::span<const EvaluationCamera> cameras() const { return {}; }
        virtual glm::mat4 object_to_world() const { return glm::mat4(1.0f); }
    };

    class LFS_CORE_API NodeError : public std::runtime_error {
    public:
        using std::runtime_error::runtime_error;
    };

    class LFS_CORE_API FieldNodeError : public NodeError {
    public:
        FieldNodeError(std::string node, std::string message)
            : NodeError(std::move(message)), node_(std::move(node)) {}
        [[nodiscard]] const std::string& node() const noexcept { return node_; }

    private:
        std::string node_;
    };

    struct EvalInputs {
        Geometry geometry;
        std::unordered_map<std::string, Value> interface_overrides;
        std::uint64_t geometry_generation = 0;
        // Explicit CPU execution remains available for tests/offline callers.
        // Unspecified chooses GPU whenever the selected backend is available.
        std::optional<core::Device> device;
        TreeResolver tree_resolver;
        // Internal recursion/cache context is public so headless callers can forward
        // EvalInputs without a second evaluator API. Ordinary callers leave it empty.
        std::vector<std::pair<std::string, std::string>> group_stack;
        std::string cache_namespace;
        // Evaluate only this output and its dependencies. For a field, geometry
        // is the first reachable consumer's input (or the modifier input).
        std::string requested_node;
        std::string requested_socket;
        bool resolve_preview_context = true;
        float seconds = 0.0f;
        float frames_per_second = 24.0f;
    };

    struct NodeEvaluation {
        double time_ms = 0.0;
        bool cached = false;
        std::optional<std::size_t> element_count;
        std::optional<double> selected_share;
    };

    // Thrown by node work that stops because its evaluation was cancelled.
    class LFS_CORE_API EvaluationCancelled : public std::runtime_error {
    public:
        EvaluationCancelled() : std::runtime_error("Evaluation cancelled") {}
    };

    // Long node work calls this between chunks; it does nothing outside an evaluation.
    LFS_CORE_API void throw_if_evaluation_cancelled();

    struct EvalControl {
        std::function<bool()> cancelled;
        std::function<void(const Node&)> started;
        std::function<void(const std::string&, const NodeEvaluation&)> finished;
        // Rethrow device out-of-memory instead of reporting it as a node error,
        // for hosts that can release memory and retry.
        bool propagate_out_of_memory = false;
        // Wait for each node's queued device work before stopping its timer, so node times include the GPU.
        bool synchronize_nodes = false;
    };

    struct EvalResult {
        Geometry geometry;
        bool ok = true;
        std::unordered_map<std::string, std::string> errors;
        std::unordered_map<std::string, double> time_ms;
        std::unordered_map<std::string, NodeEvaluation> nodes;
        std::unordered_map<std::string, Value> output_values;
        std::size_t output_key = 0;
        bool cancelled = false;
    };

    struct ConsumedSelection {
        std::uint64_t context = 0; // FieldContext::identity
        core::Tensor mask;         // Bool
    };

    struct CachedNodeOutput {
        std::size_t key = 0;
        std::unordered_map<std::string, Value> outputs;
        double time_ms = 0;
        std::optional<Geometry> geometry_input;
        // The Selection input as the node consumed it, so previews need not evaluate it again.
        std::optional<ConsumedSelection> selection;
    };

    struct EvalCache {
        std::unordered_map<std::string, CachedNodeOutput> nodes;
        GeometryDeviceCache devices;
        void clear() {
            nodes.clear();
            devices.clear();
        }
    };

    LFS_CORE_API EvalResult evaluate(const NodeTree& tree, EvalInputs inputs, EvalHost* host = nullptr,
                                     EvalCache* cache = nullptr, const EvalControl& control = {});

    class LFS_CORE_API NodeContext {
    public:
        [[nodiscard]] const Value& input(std::string_view identifier) const;
        [[nodiscard]] std::vector<Value> inputs(std::string_view identifier) const;
        [[nodiscard]] Field field(std::string_view identifier, std::string_view type) const;
        [[nodiscard]] core::Tensor evaluate_field(std::string_view identifier, const FieldContext& context,
                                                  std::string_view type = FLOAT_SOCKET) const;
        [[nodiscard]] const nlohmann::json& properties() const;
        [[nodiscard]] const Node& node() const;
        [[nodiscard]] EvalHost* host() const noexcept;
        [[nodiscard]] float seconds() const noexcept { return seconds_; }
        [[nodiscard]] float frame() const noexcept { return seconds_ * frames_per_second_; }
        void set_output(std::string identifier, Value value);
        // Keeps the first Selection mask a node evaluates, for viewport previews.
        void record_selection(const FieldContext& context, const core::Tensor& mask) const;

    private:
        friend EvalResult evaluate(const NodeTree&, EvalInputs, EvalHost*, EvalCache*, const EvalControl&);
        const Node* node_ = nullptr;
        std::unordered_map<std::string, std::vector<Value>> inputs_;
        std::unordered_map<std::string, Value> outputs_;
        mutable std::optional<ConsumedSelection> selection_;
        FieldMemo* memo_ = nullptr;
        EvalHost* host_ = nullptr;
        float seconds_ = 0.0f;
        float frames_per_second_ = 24.0f;
    };

} // namespace lfs::nodes
