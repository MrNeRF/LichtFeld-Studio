/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "internal/lazy_ir.hpp"

#include "internal/lazy_config.hpp"
#include "internal/tensor_impl.hpp"
#include <algorithm>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace lfs::core::internal {

    namespace {

        struct LazyExprNode {
            uint64_t node_id = 0;
            LazyOpKind op_kind = LazyOpKind::Leaf;
            std::string op_name;
            std::vector<uint64_t> input_ids;
            Device device = static_cast<Device>(0);
            DataType dtype = static_cast<DataType>(0);
            std::string shape;
            size_t buffer_bytes = 0;
            size_t tensor_owners = 0;
            size_t dependents = 0;
        };

        struct LazyIrRuntime {
            std::mutex mutex;
            uint64_t next_node_id = 1;
            std::unordered_map<uint64_t, LazyExprNode> nodes;
            std::unordered_map<size_t, uint64_t> tensor_to_node;
            std::optional<size_t> node_limit_override;
        };

        LazyIrRuntime& lazy_ir_runtime() {
            static LazyIrRuntime runtime;
            return runtime;
        }

        constexpr size_t kDefaultLazyIrNodeLimit = 65'536;

        size_t configured_node_limit() {
            return kDefaultLazyIrNodeLimit;
        }

        size_t node_limit_locked(const LazyIrRuntime& runtime) {
            return runtime.node_limit_override.value_or(configured_node_limit());
        }

        void publish_registry_counts_locked(const LazyIrRuntime& runtime) {
            telemetry_publish_expr_registry_counts(runtime.nodes.size(),
                                                   runtime.tensor_to_node.size(),
                                                   node_limit_locked(runtime));
        }

        void collect_unreferenced_node_locked(LazyIrRuntime& runtime, const uint64_t initial_node_id) {
            std::vector<uint64_t> pending{initial_node_id};
            while (!pending.empty()) {
                const uint64_t node_id = pending.back();
                pending.pop_back();

                const auto node_it = runtime.nodes.find(node_id);
                if (node_it == runtime.nodes.end() ||
                    node_it->second.tensor_owners != 0 ||
                    node_it->second.dependents != 0) {
                    continue;
                }

                const std::vector<uint64_t> inputs = std::move(node_it->second.input_ids);
                runtime.nodes.erase(node_it);
                for (const uint64_t input_id : inputs) {
                    const auto input_it = runtime.nodes.find(input_id);
                    if (input_it == runtime.nodes.end() || input_it->second.dependents == 0) {
                        continue;
                    }
                    --input_it->second.dependents;
                    if (input_it->second.tensor_owners == 0 && input_it->second.dependents == 0) {
                        pending.push_back(input_id);
                    }
                }
            }
        }

        void unregister_tensor_locked(LazyIrRuntime& runtime, const size_t tensor_id) {
            const auto mapping_it = runtime.tensor_to_node.find(tensor_id);
            if (mapping_it == runtime.tensor_to_node.end()) {
                return;
            }

            const uint64_t node_id = mapping_it->second;
            runtime.tensor_to_node.erase(mapping_it);
            const auto node_it = runtime.nodes.find(node_id);
            if (node_it != runtime.nodes.end() && node_it->second.tensor_owners != 0) {
                --node_it->second.tensor_owners;
            }
            collect_unreferenced_node_locked(runtime, node_id);
        }

        std::vector<uint64_t> normalize_inputs_locked(const LazyIrRuntime& runtime,
                                                      const uint64_t output_node_id,
                                                      const std::vector<uint64_t>& inputs) {
            std::vector<uint64_t> normalized;
            normalized.reserve(inputs.size());
            for (const uint64_t input_id : inputs) {
                if (input_id == 0 || input_id == output_node_id ||
                    runtime.nodes.find(input_id) == runtime.nodes.end() ||
                    std::find(normalized.begin(), normalized.end(), input_id) != normalized.end()) {
                    continue;
                }
                normalized.push_back(input_id);
            }
            return normalized;
        }

        uint64_t register_node_locked(LazyIrRuntime& runtime,
                                      size_t tensor_id,
                                      LazyOpKind kind,
                                      std::string_view op_name,
                                      std::vector<uint64_t> inputs,
                                      const Tensor& tensor) {
            unregister_tensor_locked(runtime, tensor_id);
            LazyIrTensorAccess::set_registered(tensor, false);
            if (runtime.nodes.size() >= node_limit_locked(runtime)) {
                // Lazy execution is an optimization: an unregistered deferred tensor
                // still owns its direct materializer and remains fully functional.
                telemetry_record_expr_node_drop(1);
                publish_registry_counts_locked(runtime);
                return 0;
            }

            const uint64_t node_id = runtime.next_node_id++;
            const size_t bytes = tensor.shape().elements() * dtype_size(tensor.dtype());
            inputs = normalize_inputs_locked(runtime, node_id, inputs);
            for (const uint64_t input_id : inputs) {
                ++runtime.nodes.at(input_id).dependents;
            }
            runtime.nodes.emplace(node_id, LazyExprNode{
                                               node_id,
                                               kind,
                                               std::string(op_name),
                                               std::move(inputs),
                                               tensor.device(),
                                               tensor.dtype(),
                                               tensor.shape().str(),
                                               bytes,
                                               1,
                                               0});
            runtime.tensor_to_node[tensor_id] = node_id;
            LazyIrTensorAccess::set_registered(tensor, true);
            telemetry_record_expr_node(1);
            publish_registry_counts_locked(runtime);
            return node_id;
        }

        uint64_t ensure_leaf_node_locked(LazyIrRuntime& runtime, const Tensor& tensor) {
            const size_t tensor_id = tensor.debug_id();
            if (tensor_id == 0) {
                return 0;
            }
            if (const auto it = runtime.tensor_to_node.find(tensor_id); it != runtime.tensor_to_node.end()) {
                return it->second;
            }
            return register_node_locked(runtime, tensor_id, LazyOpKind::Leaf, "leaf", {}, tensor);
        }

        LazyExprDebugInfo to_debug_info(const LazyExprNode& node) {
            return LazyExprDebugInfo{
                node.node_id,
                node.op_kind,
                node.op_name,
                node.input_ids,
                node.device,
                node.dtype,
                node.shape,
                node.buffer_bytes};
        }

    } // namespace

    void LazyIrTensorAccess::set_registered(const Tensor& tensor, const bool registered) noexcept {
        tensor.lazy_ir_registered_ = registered;
    }

    bool lazy_ir_active() {
        return true;
    }

    void clear_lazy_ir_for_testing() {
        auto& runtime = lazy_ir_runtime();
        std::lock_guard<std::mutex> lock(runtime.mutex);
        runtime.next_node_id = 1;
        runtime.nodes.clear();
        runtime.tensor_to_node.clear();
        publish_registry_counts_locked(runtime);
    }

    void lazy_ir_set_node_limit_override_for_testing(const std::optional<size_t> limit) {
        auto& runtime = lazy_ir_runtime();
        std::lock_guard<std::mutex> lock(runtime.mutex);
        runtime.node_limit_override = limit;
        publish_registry_counts_locked(runtime);
    }

    void lazy_ir_unregister_tensor(const size_t tensor_id) {
        if (tensor_id == 0) {
            return;
        }
        auto& runtime = lazy_ir_runtime();
        std::lock_guard<std::mutex> lock(runtime.mutex);
        unregister_tensor_locked(runtime, tensor_id);
        publish_registry_counts_locked(runtime);
    }

    bool tensor_has_lazy_expr(const Tensor& tensor) {
        return tensor_lazy_expr_id(tensor) != 0;
    }

    uint64_t tensor_lazy_expr_id(const Tensor& tensor) {
        if (!lazy_ir_active()) {
            return 0;
        }
        const size_t tensor_id = tensor.debug_id();
        if (tensor_id == 0) {
            return 0;
        }
        auto& runtime = lazy_ir_runtime();
        std::lock_guard<std::mutex> lock(runtime.mutex);
        if (const auto it = runtime.tensor_to_node.find(tensor_id); it != runtime.tensor_to_node.end()) {
            return it->second;
        }
        return 0;
    }

    std::optional<LazyExprDebugInfo> tensor_lazy_expr_info(const Tensor& tensor) {
        const uint64_t node_id = tensor_lazy_expr_id(tensor);
        return lazy_ir_node_info(node_id);
    }

    std::optional<LazyExprDebugInfo> lazy_ir_node_info(uint64_t node_id) {
        if (node_id == 0) {
            return std::nullopt;
        }
        auto& runtime = lazy_ir_runtime();
        std::lock_guard<std::mutex> lock(runtime.mutex);
        const auto it = runtime.nodes.find(node_id);
        if (it == runtime.nodes.end()) {
            return std::nullopt;
        }
        const auto& node = it->second;
        return to_debug_info(node);
    }

    std::vector<LazyExprDebugInfo> lazy_ir_collect_topological_subgraph(uint64_t root_node_id) {
        if (root_node_id == 0) {
            return {};
        }

        auto& runtime = lazy_ir_runtime();
        std::lock_guard<std::mutex> lock(runtime.mutex);

        std::vector<LazyExprDebugInfo> topo;
        topo.reserve(16);
        std::unordered_set<uint64_t> visited;

        const auto visit = [&](auto&& self, uint64_t node_id) -> void {
            if (node_id == 0 || !visited.insert(node_id).second) {
                return;
            }
            const auto it = runtime.nodes.find(node_id);
            if (it == runtime.nodes.end()) {
                return;
            }
            for (uint64_t input_id : it->second.input_ids) {
                self(self, input_id);
            }
            topo.push_back(to_debug_info(it->second));
        };

        visit(visit, root_node_id);
        return topo;
    }

    bool lazy_ir_set_node_inputs(uint64_t node_id, const std::vector<uint64_t>& input_ids) {
        if (!lazy_ir_active() || node_id == 0) {
            return false;
        }

        auto& runtime = lazy_ir_runtime();
        std::lock_guard<std::mutex> lock(runtime.mutex);
        const auto it = runtime.nodes.find(node_id);
        if (it == runtime.nodes.end()) {
            return false;
        }

        std::vector<uint64_t> normalized_inputs = normalize_inputs_locked(runtime, node_id, input_ids);
        const std::vector<uint64_t> previous_inputs = it->second.input_ids;

        for (const uint64_t input_id : previous_inputs) {
            const auto input_it = runtime.nodes.find(input_id);
            if (input_it != runtime.nodes.end() && input_it->second.dependents != 0) {
                --input_it->second.dependents;
            }
        }
        for (const uint64_t input_id : normalized_inputs) {
            ++runtime.nodes.at(input_id).dependents;
        }
        it->second.input_ids = std::move(normalized_inputs);

        for (const uint64_t input_id : previous_inputs) {
            collect_unreferenced_node_locked(runtime, input_id);
        }
        publish_registry_counts_locked(runtime);
        return true;
    }

    void lazy_ir_record_unary(const Tensor& input,
                              const Tensor& output,
                              std::string_view op_name) {
        if (!lazy_ir_active()) {
            return;
        }
        if (output.debug_id() == 0) {
            return;
        }
        auto& runtime = lazy_ir_runtime();
        std::lock_guard<std::mutex> lock(runtime.mutex);
        std::vector<uint64_t> inputs = {ensure_leaf_node_locked(runtime, input)};
        register_node_locked(runtime, output.debug_id(), LazyOpKind::Unary, op_name, std::move(inputs), output);
    }

    void lazy_ir_record_binary(const Tensor& left,
                               const Tensor& right,
                               const Tensor& output,
                               std::string_view op_name) {
        if (!lazy_ir_active()) {
            return;
        }
        if (output.debug_id() == 0) {
            return;
        }
        auto& runtime = lazy_ir_runtime();
        std::lock_guard<std::mutex> lock(runtime.mutex);
        std::vector<uint64_t> inputs = {
            ensure_leaf_node_locked(runtime, left),
            ensure_leaf_node_locked(runtime, right)};
        register_node_locked(runtime, output.debug_id(), LazyOpKind::Binary, op_name, std::move(inputs), output);
    }

    void lazy_ir_record_scalar_unary(const Tensor& input,
                                     const Tensor& output,
                                     std::string_view op_name) {
        if (!lazy_ir_active()) {
            return;
        }
        if (output.debug_id() == 0) {
            return;
        }
        auto& runtime = lazy_ir_runtime();
        std::lock_guard<std::mutex> lock(runtime.mutex);
        std::vector<uint64_t> inputs = {ensure_leaf_node_locked(runtime, input)};
        register_node_locked(runtime, output.debug_id(), LazyOpKind::ScalarUnary, op_name, std::move(inputs), output);
    }

    void lazy_ir_record_permutation(const Tensor& input,
                                    const Tensor& indices,
                                    const Tensor& output,
                                    std::string_view op_name) {
        if (!lazy_ir_active()) {
            return;
        }
        if (output.debug_id() == 0) {
            return;
        }
        auto& runtime = lazy_ir_runtime();
        std::lock_guard<std::mutex> lock(runtime.mutex);
        std::vector<uint64_t> inputs = {
            ensure_leaf_node_locked(runtime, input),
            ensure_leaf_node_locked(runtime, indices)};
        register_node_locked(runtime, output.debug_id(), LazyOpKind::Permutation, op_name, std::move(inputs), output);
    }

    void lazy_ir_record_reduce(const Tensor& input,
                               const Tensor& output,
                               std::string_view op_name) {
        if (!lazy_ir_active()) {
            return;
        }
        if (output.debug_id() == 0) {
            return;
        }
        auto& runtime = lazy_ir_runtime();
        std::lock_guard<std::mutex> lock(runtime.mutex);
        std::vector<uint64_t> inputs = {ensure_leaf_node_locked(runtime, input)};
        register_node_locked(runtime, output.debug_id(), LazyOpKind::Reduce, op_name, std::move(inputs), output);
    }

    uint64_t lazy_ir_record_deferred(const Tensor& output) {
        return lazy_ir_record_deferred(output, "deferred_expr", {});
    }

    uint64_t lazy_ir_record_deferred(const Tensor& output, const std::string_view op_name) {
        return lazy_ir_record_deferred(output, op_name, {});
    }

    uint64_t lazy_ir_record_deferred(const Tensor& output,
                                     std::string_view op_name,
                                     const std::vector<uint64_t>& input_ids) {
        if (!lazy_ir_active()) {
            return 0;
        }
        if (output.debug_id() == 0) {
            return 0;
        }
        auto& runtime = lazy_ir_runtime();
        std::lock_guard<std::mutex> lock(runtime.mutex);
        return register_node_locked(runtime, output.debug_id(), LazyOpKind::Deferred, op_name, input_ids, output);
    }

} // namespace lfs::core::internal
