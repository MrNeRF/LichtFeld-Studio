/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "internal/lazy_ir.hpp"

#include "internal/lazy_config.hpp"
#include "internal/tensor_impl.hpp"
#include <algorithm>
#include <mutex>
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
        };

        struct LazyIrRuntime {
            std::mutex mutex;
            uint64_t next_node_id = 1;
            std::unordered_map<uint64_t, LazyExprNode> nodes;
            std::unordered_map<size_t, uint64_t> tensor_to_node;
        };

        LazyIrRuntime& lazy_ir_runtime() {
            static LazyIrRuntime runtime;
            return runtime;
        }

        uint64_t register_node_locked(LazyIrRuntime& runtime,
                                      size_t tensor_id,
                                      LazyOpKind kind,
                                      std::string_view op_name,
                                      std::vector<uint64_t> inputs,
                                      const Tensor& tensor) {
            const uint64_t node_id = runtime.next_node_id++;
            const size_t bytes = tensor.shape().elements() * dtype_size(tensor.dtype());
            runtime.nodes.emplace(node_id, LazyExprNode{
                                               node_id,
                                               kind,
                                               std::string(op_name),
                                               std::move(inputs),
                                               tensor.device(),
                                               tensor.dtype(),
                                               tensor.shape().str(),
                                               bytes});
            runtime.tensor_to_node[tensor_id] = node_id;
            telemetry_record_expr_node(1);
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

    bool lazy_ir_active() {
        return true;
    }

    void clear_lazy_ir_for_testing() {
        auto& runtime = lazy_ir_runtime();
        std::lock_guard<std::mutex> lock(runtime.mutex);
        runtime.next_node_id = 1;
        runtime.nodes.clear();
        runtime.tensor_to_node.clear();
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

        std::vector<uint64_t> deduped_inputs;
        deduped_inputs.reserve(input_ids.size());
        for (uint64_t input_id : input_ids) {
            if (input_id == 0) {
                continue;
            }
            if (std::find(deduped_inputs.begin(), deduped_inputs.end(), input_id) == deduped_inputs.end()) {
                deduped_inputs.push_back(input_id);
            }
        }

        auto& runtime = lazy_ir_runtime();
        std::lock_guard<std::mutex> lock(runtime.mutex);
        const auto it = runtime.nodes.find(node_id);
        if (it == runtime.nodes.end()) {
            return false;
        }

        it->second.input_ids = std::move(deduped_inputs);
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
