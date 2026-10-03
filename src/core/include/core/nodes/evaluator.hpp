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

    class LFS_CORE_API EvalHost {
    public:
        virtual ~EvalHost() = default;
        virtual std::uint64_t generation() const = 0;
        virtual std::optional<Geometry> object_geometry(std::string_view name, TransformSpace space) = 0;
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
    };

    struct NodeEvaluation {
        double time_ms = 0.0;
        bool cached = false;
        std::optional<std::size_t> element_count;
        std::optional<double> selected_share;
    };

    struct EvalControl {
        std::function<bool()> cancelled;
        std::function<void(const Node&)> started;
        std::function<void(const std::string&, const NodeEvaluation&)> finished;
    };

    struct EvalResult {
        Geometry geometry;
        bool ok = true;
        std::unordered_map<std::string, std::string> errors;
        std::unordered_map<std::string, double> time_ms;
        std::unordered_map<std::string, NodeEvaluation> nodes;
        std::unordered_map<std::string, Value> output_values;
        bool cancelled = false;
    };

    struct CachedNodeOutput {
        std::size_t key = 0;
        std::unordered_map<std::string, Value> outputs;
        double time_ms = 0;
        std::optional<Geometry> geometry_input;
    };

    struct EvalCache {
        std::unordered_map<std::string, CachedNodeOutput> nodes;
        GeometryDeviceCache devices;
        void clear() {
            nodes.clear();
            devices.clear();
        }
    };

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
        void set_output(std::string identifier, Value value);

    private:
        friend EvalResult evaluate(const NodeTree&, EvalInputs, EvalHost*, EvalCache*, const EvalControl&);
        const Node* node_ = nullptr;
        std::unordered_map<std::string, std::vector<Value>> inputs_;
        std::unordered_map<std::string, Value> outputs_;
        FieldMemo* memo_ = nullptr;
        EvalHost* host_ = nullptr;
    };

    LFS_CORE_API EvalResult evaluate(const NodeTree& tree, EvalInputs inputs, EvalHost* host = nullptr,
                                     EvalCache* cache = nullptr, const EvalControl& control = {});

} // namespace lfs::nodes
