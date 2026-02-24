/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "internal/lazy_executor.hpp"

#include "core/logger.hpp"
#include "internal/cuda_stream_context.hpp"
#include "internal/lazy_config.hpp"
#include "internal/lazy_ir.hpp"
#include "internal/tensor_impl.hpp"
#include "internal/tensor_ops.hpp"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace lfs::core::internal {

    namespace {

        struct LazyExecutorContext {
            std::unordered_map<uint64_t, Tensor> cached_materializations;
        };

        struct DeferredMaterializerRegistry {
            struct Entry {
                std::function<Tensor()> materializer;
                std::weak_ptr<void> owner;
            };

            std::mutex mutex;
            std::unordered_map<uint64_t, Entry> by_node_id;
        };

        struct PointwiseFusionRegistry {
            struct Entry {
                uint64_t parent_node_id = 0;
                Tensor source;
                std::vector<LazyPointwiseOp> ops;
                std::weak_ptr<void> owner;
            };

            std::mutex mutex;
            std::unordered_map<uint64_t, Entry> by_node_id;
        };

        struct LazyExecutorDiagnosticsCounters {
            std::atomic<uint64_t> planned_nodes{0};
            std::atomic<uint64_t> executed_nodes{0};
            std::atomic<uint64_t> cache_hits{0};
            std::atomic<uint64_t> cache_misses{0};
            std::atomic<uint64_t> root_fallbacks{0};
            std::atomic<uint64_t> fused_launches{0};
            std::atomic<uint64_t> fused_reduce_launches{0};
            std::atomic<uint64_t> max_registry_entries{0};
            std::atomic<uint64_t> max_context_cache_entries{0};
            std::atomic<uint64_t> early_releases{0};
            std::atomic<uint64_t> early_release_bytes{0};
            std::atomic<uint64_t> peak_cache_bytes{0};
        };

        struct LazyExecutorDebugDumpState {
            std::atomic<int> override_enabled{-1}; // -1 unset, 0 false, 1 true
            std::atomic<bool> cached_env_enabled{false};
            std::atomic<bool> has_cached_env_enabled{false};
        };

        struct LazyExecutorPointwiseFusionState {
            std::atomic<int> override_enabled{-1}; // -1 unset, 0 false, 1 true
        };

        struct LazyExecutorMemoryPlannerState {
            std::atomic<int> override_enabled{-1}; // -1 unset, 0 false, 1 true
        };

        DeferredMaterializerRegistry& deferred_materializer_registry() {
            static DeferredMaterializerRegistry registry;
            return registry;
        }

        PointwiseFusionRegistry& pointwise_fusion_registry() {
            static PointwiseFusionRegistry registry;
            return registry;
        }

        LazyExecutorDiagnosticsCounters& lazy_executor_diagnostics_counters() {
            static LazyExecutorDiagnosticsCounters counters;
            return counters;
        }

        LazyExecutorDebugDumpState& lazy_executor_debug_dump_state() {
            static LazyExecutorDebugDumpState state;
            return state;
        }

        LazyExecutorPointwiseFusionState& lazy_executor_pointwise_fusion_state() {
            static LazyExecutorPointwiseFusionState state;
            return state;
        }

        struct LazyExecutorSizeHeuristicState {
            std::atomic<int> override_enabled{-1};       // -1 unset, 0 false, 1 true
            std::atomic<int64_t> override_threshold{-1}; // -1 unset
            std::atomic<bool> has_cached_env{false};
            std::atomic<bool> cached_enabled{true};
            std::atomic<size_t> cached_threshold{4096};
        };

        LazyExecutorMemoryPlannerState& lazy_executor_memory_planner_state() {
            static LazyExecutorMemoryPlannerState state;
            return state;
        }

        LazyExecutorSizeHeuristicState& lazy_executor_size_heuristic_state() {
            static LazyExecutorSizeHeuristicState state;
            return state;
        }

        thread_local LazyExecutorContext* active_context = nullptr;

        class LazyExecutorContextScope {
        public:
            explicit LazyExecutorContextScope(LazyExecutorContext* context)
                : previous_(active_context) {
                active_context = context;
            }

            ~LazyExecutorContextScope() { active_context = previous_; }

            LazyExecutorContextScope(const LazyExecutorContextScope&) = delete;
            LazyExecutorContextScope& operator=(const LazyExecutorContextScope&) = delete;

        private:
            LazyExecutorContext* previous_ = nullptr;
        };

        std::string trim_ascii_lower(std::string_view value) {
            size_t begin = 0;
            while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin]))) {
                ++begin;
            }

            size_t end = value.size();
            while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
                --end;
            }

            std::string result(value.substr(begin, end - begin));
            std::transform(result.begin(), result.end(), result.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
            return result;
        }

        bool parse_debug_dump_bool(std::string_view value, bool fallback) {
            const std::string normalized = trim_ascii_lower(value);
            if (normalized.empty()) {
                return fallback;
            }
            if (normalized == "1" || normalized == "true" || normalized == "yes" ||
                normalized == "on" || normalized == "enable" || normalized == "enabled") {
                return true;
            }
            if (normalized == "0" || normalized == "false" || normalized == "no" ||
                normalized == "off" || normalized == "disable" || normalized == "disabled") {
                return false;
            }
            return fallback;
        }

        bool lazy_executor_debug_dump_enabled() {
            auto& state = lazy_executor_debug_dump_state();
            const int override_enabled = state.override_enabled.load(std::memory_order_acquire);
            if (override_enabled == 0 || override_enabled == 1) {
                return override_enabled == 1;
            }

            if (!state.has_cached_env_enabled.load(std::memory_order_acquire)) {
                const char* env = std::getenv("TENSOR_LAZY_EXEC_DIAGNOSTICS");
                const bool enabled = env ? parse_debug_dump_bool(env, false) : false;
                state.cached_env_enabled.store(enabled, std::memory_order_release);
                state.has_cached_env_enabled.store(true, std::memory_order_release);
            }
            return state.cached_env_enabled.load(std::memory_order_acquire);
        }

        bool lazy_executor_pointwise_fusion_enabled() {
            const int override_enabled = lazy_executor_pointwise_fusion_state()
                                             .override_enabled.load(std::memory_order_acquire);
            if (override_enabled == 0 || override_enabled == 1) {
                return override_enabled == 1;
            }
            return true;
        }

        bool lazy_executor_memory_planner_enabled() {
            const int override_enabled = lazy_executor_memory_planner_state()
                                             .override_enabled.load(std::memory_order_acquire);
            if (override_enabled == 0 || override_enabled == 1) {
                return override_enabled == 1;
            }
            return true;
        }

        void prune_expired_materializers_locked(DeferredMaterializerRegistry& registry) {
            for (auto it = registry.by_node_id.begin(); it != registry.by_node_id.end();) {
                if (it->second.owner.expired()) {
                    it = registry.by_node_id.erase(it);
                    continue;
                }
                ++it;
            }
        }

        void prune_expired_pointwise_fusions_locked(PointwiseFusionRegistry& registry) {
            for (auto it = registry.by_node_id.begin(); it != registry.by_node_id.end();) {
                if (it->second.owner.expired()) {
                    it = registry.by_node_id.erase(it);
                    continue;
                }
                ++it;
            }
        }

        bool lookup_registered_materializer(uint64_t node_id, std::function<Tensor()>& materializer) {
            if (node_id == 0) {
                return false;
            }
            auto& registry = deferred_materializer_registry();
            std::lock_guard<std::mutex> lock(registry.mutex);
            prune_expired_materializers_locked(registry);
            const auto it = registry.by_node_id.find(node_id);
            if (it == registry.by_node_id.end()) {
                return false;
            }
            materializer = it->second.materializer;
            return static_cast<bool>(materializer);
        }

        void record_planned_nodes(size_t count) {
            lazy_executor_diagnostics_counters().planned_nodes.fetch_add(
                static_cast<uint64_t>(count), std::memory_order_relaxed);
        }

        void record_executed_node() {
            lazy_executor_diagnostics_counters().executed_nodes.fetch_add(1, std::memory_order_relaxed);
        }

        void record_cache_hit() {
            lazy_executor_diagnostics_counters().cache_hits.fetch_add(1, std::memory_order_relaxed);
        }

        void record_cache_miss() {
            lazy_executor_diagnostics_counters().cache_misses.fetch_add(1, std::memory_order_relaxed);
        }

        void record_root_fallback() {
            lazy_executor_diagnostics_counters().root_fallbacks.fetch_add(1, std::memory_order_relaxed);
        }

        void record_fused_launch() {
            lazy_executor_diagnostics_counters().fused_launches.fetch_add(1, std::memory_order_relaxed);
        }

        void record_relaxed_max(std::atomic<uint64_t>& target, uint64_t value) {
            uint64_t current = target.load(std::memory_order_relaxed);
            while (current < value &&
                   !target.compare_exchange_weak(current, value, std::memory_order_relaxed)) {
            }
        }

        void record_early_release(size_t bytes) {
            auto& counters = lazy_executor_diagnostics_counters();
            counters.early_releases.fetch_add(1, std::memory_order_relaxed);
            counters.early_release_bytes.fetch_add(
                static_cast<uint64_t>(bytes), std::memory_order_relaxed);
        }

        void record_peak_cache_bytes(size_t bytes) {
            record_relaxed_max(lazy_executor_diagnostics_counters().peak_cache_bytes,
                               static_cast<uint64_t>(bytes));
        }

        // Maps execution step index → list of node_ids to release from cache after that step.
        std::unordered_map<size_t, std::vector<uint64_t>>
        compute_release_schedule(const LazyExecutionPlanDebug& plan,
                                 const std::unordered_set<uint64_t>& internal_fused_nodes) {
            std::unordered_map<size_t, std::vector<uint64_t>> schedule;
            if (plan.topo_nodes.empty() || !plan.has_root) {
                return schedule;
            }

            // Build step index for each node_id.
            std::unordered_map<uint64_t, size_t> node_step;
            node_step.reserve(plan.topo_nodes.size());
            for (size_t i = 0; i < plan.topo_nodes.size(); ++i) {
                node_step[plan.topo_nodes[i].node_id] = i;
            }

            // For each node, find the last step that consumes it as input.
            std::unordered_map<uint64_t, size_t> last_consumer_step;
            for (size_t step = 0; step < plan.topo_nodes.size(); ++step) {
                if (internal_fused_nodes.find(plan.topo_nodes[step].node_id) != internal_fused_nodes.end()) {
                    continue;
                }
                for (uint64_t input_id : plan.topo_nodes[step].input_ids) {
                    last_consumer_step[input_id] = step;
                }
            }

            // Invert: step → [node_ids to release].
            for (const auto& [node_id, step] : last_consumer_step) {
                if (internal_fused_nodes.find(node_id) != internal_fused_nodes.end()) {
                    continue;
                }
                if (node_id == plan.root_node_id) {
                    continue;
                }
                schedule[step].push_back(node_id);
            }

            return schedule;
        }

        LazyExecutorDiagnosticsSnapshot diagnostics_delta(const LazyExecutorDiagnosticsSnapshot& before,
                                                          const LazyExecutorDiagnosticsSnapshot& after) {
            return LazyExecutorDiagnosticsSnapshot{
                after.planned_nodes - before.planned_nodes,
                after.executed_nodes - before.executed_nodes,
                after.cache_hits - before.cache_hits,
                after.cache_misses - before.cache_misses,
                after.root_fallbacks - before.root_fallbacks,
                after.fused_launches - before.fused_launches,
                after.fused_reduce_launches - before.fused_reduce_launches,
                after.max_registry_entries,
                after.max_context_cache_entries,
                after.early_releases - before.early_releases,
                after.early_release_bytes - before.early_release_bytes,
                after.peak_cache_bytes};
        }

        struct PointwiseFusionRecipe {
            uint64_t parent_node_id = 0;
            Tensor source;
            std::vector<LazyPointwiseOp> ops;
        };

        std::unordered_map<uint64_t, PointwiseFusionRecipe>
        collect_pointwise_fusion_recipes_for_plan(const LazyExecutionPlanDebug& plan) {
            std::unordered_map<uint64_t, PointwiseFusionRecipe> recipes;
            if (!lazy_executor_pointwise_fusion_enabled() || plan.topo_nodes.empty()) {
                return recipes;
            }

            auto& registry = pointwise_fusion_registry();
            std::lock_guard<std::mutex> lock(registry.mutex);
            prune_expired_pointwise_fusions_locked(registry);

            recipes.reserve(plan.topo_nodes.size());
            for (const auto& node : plan.topo_nodes) {
                const auto it = registry.by_node_id.find(node.node_id);
                if (it == registry.by_node_id.end()) {
                    continue;
                }
                recipes.emplace(node.node_id, PointwiseFusionRecipe{
                                                  it->second.parent_node_id,
                                                  it->second.source,
                                                  it->second.ops});
            }
            return recipes;
        }

        std::unordered_set<uint64_t> collect_internal_fused_nodes(
            const LazyExecutionPlanDebug& plan,
            const std::unordered_map<uint64_t, PointwiseFusionRecipe>& recipes) {
            std::unordered_map<uint64_t, size_t> usage_count;
            std::unordered_map<uint64_t, uint64_t> last_consumer;
            for (const auto& node : plan.topo_nodes) {
                for (uint64_t input_id : node.input_ids) {
                    usage_count[input_id] += 1;
                    last_consumer[input_id] = node.node_id;
                }
            }

            std::unordered_set<uint64_t> internal_nodes;
            for (const auto& [node_id, recipe] : recipes) {
                const auto usage_it = usage_count.find(node_id);
                if (usage_it == usage_count.end() || usage_it->second != 1) {
                    continue;
                }
                const auto consumer_it = last_consumer.find(node_id);
                if (consumer_it == last_consumer.end()) {
                    continue;
                }
                const auto consumer_recipe_it = recipes.find(consumer_it->second);
                if (consumer_recipe_it == recipes.end()) {
                    continue;
                }
                if (consumer_recipe_it->second.parent_node_id != node_id) {
                    continue;
                }
                internal_nodes.insert(node_id);
            }
            return internal_nodes;
        }

        bool is_pure_scalar_chain(const std::vector<LazyPointwiseOp>& ops) {
            for (const auto& op : ops) {
                if (static_cast<uint8_t>(op.kind) >= 10) {
                    return false;
                }
            }
            return true;
        }

        float apply_pointwise_op_cpu(float x, LazyPointwiseOpKind kind, float scalar) {
            switch (kind) {
            case LazyPointwiseOpKind::AddScalar: return x + scalar;
            case LazyPointwiseOpKind::SubScalar: return x - scalar;
            case LazyPointwiseOpKind::MulScalar: return x * scalar;
            case LazyPointwiseOpKind::DivScalar: return x / scalar;
            case LazyPointwiseOpKind::Abs: return std::abs(x);
            case LazyPointwiseOpKind::Neg: return -x;
            case LazyPointwiseOpKind::Exp: return std::exp(x);
            case LazyPointwiseOpKind::Log: return std::log(std::max(x, 1e-10f));
            case LazyPointwiseOpKind::Sqrt: return std::sqrt(std::max(x, 0.0f));
            case LazyPointwiseOpKind::Sigmoid: return 1.0f / (1.0f + std::exp(-x));
            case LazyPointwiseOpKind::Relu: return std::max(x, 0.0f);
            case LazyPointwiseOpKind::Square: return x * x;
            case LazyPointwiseOpKind::Tanh: return std::tanh(x);
            case LazyPointwiseOpKind::Rsqrt: return 1.0f / std::sqrt(std::max(x, 1e-10f));
            case LazyPointwiseOpKind::Sign: return float((x > 0) - (x < 0));
            case LazyPointwiseOpKind::Reciprocal: return 1.0f / (x + 1e-8f);
            case LazyPointwiseOpKind::Floor: return std::floor(x);
            case LazyPointwiseOpKind::Ceil: return std::ceil(x);
            case LazyPointwiseOpKind::Round: return std::round(x);
            }
            return x;
        }

        struct AffineCoeffs {
            float a = 1.0f;
            float b = 0.0f;
        };

        AffineCoeffs fold_affine(const std::vector<LazyPointwiseOp>& ops) {
            float a = 1.0f;
            float b = 0.0f;
            for (const auto& op : ops) {
                switch (op.kind) {
                case LazyPointwiseOpKind::AddScalar:
                    b += op.scalar;
                    break;
                case LazyPointwiseOpKind::SubScalar:
                    b -= op.scalar;
                    break;
                case LazyPointwiseOpKind::MulScalar:
                    a *= op.scalar;
                    b *= op.scalar;
                    break;
                case LazyPointwiseOpKind::DivScalar:
                    a /= op.scalar;
                    b /= op.scalar;
                    break;
                }
            }
            return {a, b};
        }

        bool execute_pointwise_fusion_recipe(const PointwiseFusionRecipe& recipe, Tensor& materialized) {
            if (recipe.ops.empty()) {
                return false;
            }

            if (recipe.ops.size() > tensor_ops::FUSED_POINTWISE_MAX_OPS) {
                return false;
            }

            Tensor source = recipe.source;
            if (!source.is_valid() ||
                source.dtype() != DataType::Float32 ||
                !source.is_contiguous()) {
                return false;
            }

            const size_t n = source.numel();
            const bool pure_scalar = is_pure_scalar_chain(recipe.ops);

            if (pure_scalar) {
                const auto [a, b] = fold_affine(recipe.ops);

                if (source.device() == Device::CUDA) {
                    const float* in_ptr = source.ptr<float>();
                    assert(in_ptr != nullptr);
                    CUDAStreamGuard guard(source.stream());
                    Tensor out = Tensor::empty(source.shape(), Device::CUDA, DataType::Float32);
                    float* out_ptr = out.ptr<float>();
                    assert(out_ptr != nullptr);
                    tensor_ops::launch_fused_affine_transform(in_ptr, out_ptr, n, a, b, out.stream());
                    materialized = std::move(out);
                    return true;
                }

                const float* in_ptr = source.ptr<float>();
                if (in_ptr == nullptr)
                    return false;
                Tensor out = Tensor::empty(source.shape(), Device::CPU, DataType::Float32);
                float* out_ptr = out.ptr<float>();
                if (out_ptr == nullptr)
                    return false;
                for (size_t i = 0; i < n; ++i) {
                    out_ptr[i] = std::fma(a, in_ptr[i], b);
                }
                materialized = std::move(out);
                return materialized.is_valid();
            }

            // Mixed chain: build op chain and dispatch
            tensor_ops::FusedPointwiseOpChain chain{};
            chain.num_ops = static_cast<int>(recipe.ops.size());
            for (int i = 0; i < chain.num_ops; ++i) {
                chain.ops[i].kind = static_cast<uint8_t>(recipe.ops[i].kind);
                chain.ops[i].scalar = recipe.ops[i].scalar;
            }

            if (source.device() == Device::CUDA) {
                const float* in_ptr = source.ptr<float>();
                assert(in_ptr != nullptr);
                CUDAStreamGuard guard(source.stream());
                Tensor out = Tensor::empty(source.shape(), Device::CUDA, DataType::Float32);
                float* out_ptr = out.ptr<float>();
                assert(out_ptr != nullptr);
                tensor_ops::launch_fused_pointwise_chain(in_ptr, out_ptr, n, chain, out.stream());
                materialized = std::move(out);
                return true;
            }

            const float* in_ptr = source.ptr<float>();
            if (in_ptr == nullptr)
                return false;
            Tensor out = Tensor::empty(source.shape(), Device::CPU, DataType::Float32);
            float* out_ptr = out.ptr<float>();
            if (out_ptr == nullptr)
                return false;
            for (size_t i = 0; i < n; ++i) {
                float val = in_ptr[i];
                for (int j = 0; j < chain.num_ops; ++j) {
                    val = apply_pointwise_op_cpu(val, recipe.ops[j].kind, recipe.ops[j].scalar);
                }
                out_ptr[i] = val;
            }
            materialized = std::move(out);
            return materialized.is_valid();
        }

        void execute_topological_nodes(const LazyExecutionPlanDebug& plan) {
            const auto recipes = collect_pointwise_fusion_recipes_for_plan(plan);
            const auto internal_nodes = collect_internal_fused_nodes(plan, recipes);

            const bool memory_planner_active = lazy_executor_memory_planner_enabled();
            std::unordered_map<size_t, std::vector<uint64_t>> release_schedule;
            std::unordered_map<uint64_t, size_t> node_bytes_map;
            if (memory_planner_active) {
                release_schedule = compute_release_schedule(plan, internal_nodes);
                for (const auto& n : plan.topo_nodes) {
                    node_bytes_map[n.node_id] = n.buffer_bytes;
                }
            }

            size_t current_cache_bytes = 0;

            auto track_cache_insert = [&](size_t bytes) {
                if (!memory_planner_active)
                    return;
                current_cache_bytes += bytes;
                record_peak_cache_bytes(current_cache_bytes);
            };

            auto try_release_dead = [&](size_t step) {
                if (!memory_planner_active)
                    return;
                const auto it = release_schedule.find(step);
                if (it == release_schedule.end())
                    return;
                for (uint64_t dead_id : it->second) {
                    if (active_context->cached_materializations.erase(dead_id)) {
                        const size_t dead_bytes = node_bytes_map[dead_id];
                        record_early_release(dead_bytes);
                        if (current_cache_bytes >= dead_bytes)
                            current_cache_bytes -= dead_bytes;
                    }
                }
            };

            for (size_t step = 0; step < plan.topo_nodes.size(); ++step) {
                const auto& node = plan.topo_nodes[step];

                if (internal_nodes.find(node.node_id) != internal_nodes.end()) {
                    try_release_dead(step);
                    continue;
                }

                Tensor cached;
                if (lazy_executor_lookup_cached_materialization(node.node_id, cached)) {
                    try_release_dead(step);
                    continue;
                }

                const auto recipe_it = recipes.find(node.node_id);
                if (recipe_it != recipes.end()) {
                    Tensor fused_materialized;
                    if (execute_pointwise_fusion_recipe(recipe_it->second, fused_materialized) &&
                        fused_materialized.is_valid()) {
                        record_executed_node();
                        record_fused_launch();
                        lazy_executor_cache_materialization(node.node_id, fused_materialized);
                        track_cache_insert(node.buffer_bytes);
                        try_release_dead(step);
                        continue;
                    }
                }

                std::function<Tensor()> materializer;
                if (!lookup_registered_materializer(node.node_id, materializer)) {
                    try_release_dead(step);
                    continue;
                }

                record_executed_node();
                Tensor materialized = materializer();
                if (materialized.is_valid()) {
                    lazy_executor_cache_materialization(node.node_id, materialized);
                    track_cache_insert(node.buffer_bytes);
                }
                try_release_dead(step);
            }
        }

    } // namespace

    void lazy_executor_register_deferred_materializer(uint64_t node_id,
                                                      std::function<Tensor()> materializer,
                                                      std::weak_ptr<void> owner) {
        if (node_id == 0 || !materializer || owner.expired()) {
            return;
        }
        auto& registry = deferred_materializer_registry();
        std::lock_guard<std::mutex> lock(registry.mutex);
        prune_expired_materializers_locked(registry);
        registry.by_node_id[node_id] = DeferredMaterializerRegistry::Entry{
            std::move(materializer),
            std::move(owner)};
        record_relaxed_max(lazy_executor_diagnostics_counters().max_registry_entries,
                           static_cast<uint64_t>(registry.by_node_id.size()));
    }

    void lazy_executor_register_pointwise_fusion_op(uint64_t node_id,
                                                    uint64_t parent_node_id,
                                                    const Tensor& source_tensor,
                                                    LazyPointwiseOp op,
                                                    std::weak_ptr<void> owner) {
        if (!lazy_executor_pointwise_fusion_enabled() ||
            node_id == 0 ||
            owner.expired()) {
            return;
        }

        auto& registry = pointwise_fusion_registry();
        std::lock_guard<std::mutex> lock(registry.mutex);
        prune_expired_pointwise_fusions_locked(registry);

        PointwiseFusionRegistry::Entry entry;
        entry.parent_node_id = parent_node_id;
        entry.owner = std::move(owner);

        if (parent_node_id != 0) {
            const auto parent_it = registry.by_node_id.find(parent_node_id);
            if (parent_it != registry.by_node_id.end()) {
                entry.source = parent_it->second.source;
                entry.ops = parent_it->second.ops;
            }
        }
        if (!entry.source.is_valid()) {
            entry.source = source_tensor;
        }
        entry.ops.push_back(op);
        registry.by_node_id[node_id] = std::move(entry);
    }

    void lazy_executor_unregister_deferred_materializer(uint64_t node_id) {
        if (node_id == 0) {
            return;
        }
        auto& registry = deferred_materializer_registry();
        std::lock_guard<std::mutex> lock(registry.mutex);
        registry.by_node_id.erase(node_id);

        auto& fusion_registry = pointwise_fusion_registry();
        std::lock_guard<std::mutex> fusion_lock(fusion_registry.mutex);
        fusion_registry.by_node_id.erase(node_id);
    }

    size_t lazy_executor_registered_node_count_for_testing() {
        auto& registry = deferred_materializer_registry();
        std::lock_guard<std::mutex> lock(registry.mutex);
        prune_expired_materializers_locked(registry);
        return registry.by_node_id.size();
    }

    void lazy_executor_clear_registry_for_testing() {
        auto& registry = deferred_materializer_registry();
        std::lock_guard<std::mutex> lock(registry.mutex);
        registry.by_node_id.clear();

        auto& fusion_registry = pointwise_fusion_registry();
        std::lock_guard<std::mutex> fusion_lock(fusion_registry.mutex);
        fusion_registry.by_node_id.clear();
    }

    bool lazy_executor_context_active() {
        return active_context != nullptr;
    }

    bool lazy_executor_lookup_cached_materialization(uint64_t node_id, Tensor& materialized) {
        if (node_id == 0 || active_context == nullptr) {
            return false;
        }
        const auto it = active_context->cached_materializations.find(node_id);
        if (it == active_context->cached_materializations.end()) {
            record_cache_miss();
            return false;
        }
        record_cache_hit();
        materialized = it->second;
        return true;
    }

    void lazy_executor_cache_materialization(uint64_t node_id, const Tensor& materialized) {
        if (node_id == 0 || active_context == nullptr || !materialized.is_valid()) {
            return;
        }
        active_context->cached_materializations.emplace(node_id, materialized);
        record_relaxed_max(
            lazy_executor_diagnostics_counters().max_context_cache_entries,
            static_cast<uint64_t>(active_context->cached_materializations.size()));
    }

    LazyExecutionPlanDebug lazy_planner_build_plan_for_tensor(const Tensor& output) {
        LazyExecutionPlanDebug plan;
        plan.planner_enabled = true;

        const uint64_t root_id = output.lazy_expr_id();
        if (root_id == 0) {
            return plan;
        }

        plan.has_root = true;
        plan.root_node_id = root_id;

        const auto topo = lazy_ir_collect_topological_subgraph(root_id);
        for (const auto& node : topo) {
            plan.topo_nodes.push_back(
                LazyPlanNodeDebug{node.node_id, node.op_name, node.input_ids, node.buffer_bytes});
        }

        if (plan.topo_nodes.empty()) {
            const auto root_info = lazy_ir_node_info(root_id);
            if (root_info.has_value()) {
                plan.topo_nodes.push_back(LazyPlanNodeDebug{
                    root_info->node_id, root_info->op_name, root_info->input_ids, root_info->buffer_bytes});
            }
        }
        return plan;
    }

    Tensor lazy_planner_execute_plan_for_tensor(const Tensor& output,
                                                std::function<Tensor()> materializer) {
        if (!materializer) {
            throw std::runtime_error("lazy_planner_execute_plan_for_tensor: materializer is empty");
        }

        const bool emit_debug_dump = (active_context == nullptr) && lazy_executor_debug_dump_enabled();
        LazyExecutorDiagnosticsSnapshot diagnostics_before{};
        if (emit_debug_dump) {
            diagnostics_before = lazy_executor_diagnostics_snapshot_for_testing();
        }

        // Build plan and execute registered deferred nodes in topological order.
        const LazyExecutionPlanDebug plan = lazy_planner_build_plan_for_tensor(output);
        record_planned_nodes(plan.topo_nodes.size());

        auto maybe_emit_debug_dump = [&](const Tensor& result, const char* root_source) {
            if (!emit_debug_dump) {
                return;
            }
            const auto diagnostics_after = lazy_executor_diagnostics_snapshot_for_testing();
            const auto delta = diagnostics_delta(diagnostics_before, diagnostics_after);
            LOG_INFO(
                "lazy-exec root={} source={} planned={} executed={} fused={} cache_hit={} cache_miss={} root_fallback={} reg_peak={} ctx_peak={} early_rel={} early_rel_bytes={} peak_cache_bytes={} result_valid={}",
                plan.root_node_id,
                root_source,
                delta.planned_nodes,
                delta.executed_nodes,
                delta.fused_launches,
                delta.cache_hits,
                delta.cache_misses,
                delta.root_fallbacks,
                delta.max_registry_entries,
                delta.max_context_cache_entries,
                delta.early_releases,
                delta.early_release_bytes,
                delta.peak_cache_bytes,
                result.is_valid());
        };

        if (active_context != nullptr) {
            execute_topological_nodes(plan);

            if (plan.has_root) {
                Tensor cached;
                if (lazy_executor_lookup_cached_materialization(plan.root_node_id, cached)) {
                    return cached;
                }
            }
            Tensor result = materializer();
            if (plan.has_root && result.is_valid()) {
                lazy_executor_cache_materialization(plan.root_node_id, result);
            }
            record_root_fallback();
            return result;
        }

        LazyExecutorContext local_context;
        local_context.cached_materializations.reserve(
            std::max<size_t>(plan.topo_nodes.size(), static_cast<size_t>(4)));
        LazyExecutorContextScope context_scope(&local_context);

        execute_topological_nodes(plan);
        if (plan.has_root) {
            Tensor cached;
            if (lazy_executor_lookup_cached_materialization(plan.root_node_id, cached)) {
                maybe_emit_debug_dump(cached, "cache");
                return cached;
            }
        }

        Tensor result = materializer();
        if (plan.has_root && result.is_valid()) {
            lazy_executor_cache_materialization(plan.root_node_id, result);
        }
        record_root_fallback();
        maybe_emit_debug_dump(result, "fallback");
        return result;
    }

    void lazy_executor_reset_diagnostics_for_testing() {
        auto& diagnostics = lazy_executor_diagnostics_counters();
        diagnostics.planned_nodes.store(0, std::memory_order_relaxed);
        diagnostics.executed_nodes.store(0, std::memory_order_relaxed);
        diagnostics.cache_hits.store(0, std::memory_order_relaxed);
        diagnostics.cache_misses.store(0, std::memory_order_relaxed);
        diagnostics.root_fallbacks.store(0, std::memory_order_relaxed);
        diagnostics.fused_launches.store(0, std::memory_order_relaxed);
        diagnostics.fused_reduce_launches.store(0, std::memory_order_relaxed);
        diagnostics.max_registry_entries.store(0, std::memory_order_relaxed);
        diagnostics.max_context_cache_entries.store(0, std::memory_order_relaxed);
        diagnostics.early_releases.store(0, std::memory_order_relaxed);
        diagnostics.early_release_bytes.store(0, std::memory_order_relaxed);
        diagnostics.peak_cache_bytes.store(0, std::memory_order_relaxed);
    }

    LazyExecutorDiagnosticsSnapshot lazy_executor_diagnostics_snapshot_for_testing() {
        const auto& diagnostics = lazy_executor_diagnostics_counters();
        return LazyExecutorDiagnosticsSnapshot{
            diagnostics.planned_nodes.load(std::memory_order_relaxed),
            diagnostics.executed_nodes.load(std::memory_order_relaxed),
            diagnostics.cache_hits.load(std::memory_order_relaxed),
            diagnostics.cache_misses.load(std::memory_order_relaxed),
            diagnostics.root_fallbacks.load(std::memory_order_relaxed),
            diagnostics.fused_launches.load(std::memory_order_relaxed),
            diagnostics.fused_reduce_launches.load(std::memory_order_relaxed),
            diagnostics.max_registry_entries.load(std::memory_order_relaxed),
            diagnostics.max_context_cache_entries.load(std::memory_order_relaxed),
            diagnostics.early_releases.load(std::memory_order_relaxed),
            diagnostics.early_release_bytes.load(std::memory_order_relaxed),
            diagnostics.peak_cache_bytes.load(std::memory_order_relaxed)};
    }

    void lazy_executor_set_debug_dump_override_for_testing(std::optional<bool> enabled) {
        auto& state = lazy_executor_debug_dump_state();
        if (enabled.has_value()) {
            state.override_enabled.store(*enabled ? 1 : 0, std::memory_order_release);
            return;
        }
        state.override_enabled.store(-1, std::memory_order_release);
    }

    void lazy_executor_clear_debug_dump_cache_for_testing() {
        auto& state = lazy_executor_debug_dump_state();
        state.cached_env_enabled.store(false, std::memory_order_release);
        state.has_cached_env_enabled.store(false, std::memory_order_release);
    }

    bool lazy_executor_debug_dump_enabled_for_testing() {
        return lazy_executor_debug_dump_enabled();
    }

    void lazy_executor_set_pointwise_fusion_override_for_testing(std::optional<bool> enabled) {
        auto& state = lazy_executor_pointwise_fusion_state();
        if (enabled.has_value()) {
            state.override_enabled.store(*enabled ? 1 : 0, std::memory_order_release);
            return;
        }
        state.override_enabled.store(-1, std::memory_order_release);
    }

    bool lazy_executor_pointwise_fusion_enabled_for_testing() {
        return lazy_executor_pointwise_fusion_enabled();
    }

    void lazy_executor_diagnostics_counters_increment_fused() {
        lazy_executor_diagnostics_counters().fused_launches.fetch_add(1, std::memory_order_relaxed);
    }

    void lazy_executor_diagnostics_counters_increment_fused_reduce() {
        lazy_executor_diagnostics_counters().fused_reduce_launches.fetch_add(1, std::memory_order_relaxed);
    }

    void lazy_executor_set_memory_planner_override_for_testing(std::optional<bool> enabled) {
        auto& state = lazy_executor_memory_planner_state();
        if (enabled.has_value()) {
            state.override_enabled.store(*enabled ? 1 : 0, std::memory_order_release);
            return;
        }
        state.override_enabled.store(-1, std::memory_order_release);
    }

    bool lazy_executor_memory_planner_enabled_for_testing() {
        return lazy_executor_memory_planner_enabled();
    }

    void lazy_executor_set_size_heuristic_override_for_testing(std::optional<bool> enabled) {
        auto& state = lazy_executor_size_heuristic_state();
        if (enabled.has_value()) {
            state.override_enabled.store(*enabled ? 1 : 0, std::memory_order_release);
            return;
        }
        state.override_enabled.store(-1, std::memory_order_release);
    }

    void lazy_executor_set_size_threshold_override_for_testing(std::optional<size_t> threshold) {
        auto& state = lazy_executor_size_heuristic_state();
        if (threshold.has_value()) {
            state.override_threshold.store(static_cast<int64_t>(*threshold), std::memory_order_release);
            return;
        }
        state.override_threshold.store(-1, std::memory_order_release);
    }

    size_t lazy_executor_size_heuristic_threshold() {
        auto& state = lazy_executor_size_heuristic_state();

        const int64_t override_threshold = state.override_threshold.load(std::memory_order_acquire);
        if (override_threshold >= 0) {
            return static_cast<size_t>(override_threshold);
        }

        if (!state.has_cached_env.load(std::memory_order_acquire)) {
            const char* env_threshold = std::getenv("TENSOR_LAZY_SIZE_THRESHOLD");
            if (env_threshold) {
                const std::string normalized = trim_ascii_lower(env_threshold);
                if (!normalized.empty()) {
                    char* end = nullptr;
                    const unsigned long parsed = std::strtoul(normalized.c_str(), &end, 10);
                    if (end != normalized.c_str() && parsed > 0) {
                        state.cached_threshold.store(static_cast<size_t>(parsed), std::memory_order_release);
                    }
                }
            }

            const char* env_enabled = std::getenv("TENSOR_LAZY_SIZE_HEURISTIC");
            if (env_enabled) {
                state.cached_enabled.store(
                    parse_debug_dump_bool(env_enabled, true), std::memory_order_release);
            }

            state.has_cached_env.store(true, std::memory_order_release);
        }

        return state.cached_threshold.load(std::memory_order_acquire);
    }

    bool lazy_size_heuristic_should_defer(size_t byte_count) {
        auto& state = lazy_executor_size_heuristic_state();

        const int override_enabled = state.override_enabled.load(std::memory_order_acquire);
        if (override_enabled == 0) {
            return true;
        }

        bool enabled;
        if (override_enabled == 1) {
            enabled = true;
        } else {
            // Trigger env caching via threshold() call
            lazy_executor_size_heuristic_threshold();
            enabled = state.cached_enabled.load(std::memory_order_acquire);
        }

        if (!enabled) {
            return true;
        }

        return byte_count >= lazy_executor_size_heuristic_threshold();
    }

    bool lazy_executor_try_consume_pointwise_fusion(
        uint64_t node_id, Tensor* out_source, std::vector<LazyPointwiseOp>* out_ops) {
        assert(out_source != nullptr);
        assert(out_ops != nullptr);

        if (node_id == 0 || !lazy_executor_pointwise_fusion_enabled()) {
            return false;
        }

        auto& registry = pointwise_fusion_registry();
        std::lock_guard<std::mutex> lock(registry.mutex);
        prune_expired_pointwise_fusions_locked(registry);

        const auto it = registry.by_node_id.find(node_id);
        if (it == registry.by_node_id.end()) {
            return false;
        }

        *out_source = std::move(it->second.source);
        *out_ops = std::move(it->second.ops);
        registry.by_node_id.erase(it);
        return true;
    }

} // namespace lfs::core::internal
