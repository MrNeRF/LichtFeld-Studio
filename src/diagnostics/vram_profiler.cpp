/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "diagnostics/vram_profiler.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <deque>
#include <iterator>
#include <list>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace lfs::diagnostics {

    namespace {

        struct MetricKey {
            std::string scope;
            std::string label;

            bool operator==(const MetricKey& other) const {
                return scope == other.scope && label == other.label;
            }
        };

        struct MetricKeyHash {
            std::size_t operator()(const MetricKey& key) const {
                const std::size_t h1 = std::hash<std::string>{}(key.scope);
                const std::size_t h2 = std::hash<std::string>{}(key.label);
                return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
            }
        };

        struct Metric {
            std::size_t live_bytes = 0;
            std::size_t peak_bytes = 0;
            std::size_t allocated_bytes = 0;
            std::size_t freed_bytes = 0;
            std::uint64_t allocation_count = 0;
            std::uint64_t free_count = 0;
            VramAllocationMethod method = VramAllocationMethod::Unknown;
            bool current_sample = false;
        };

        struct AllocationRecord {
            MetricKey key;
            std::size_t bytes = 0;
            VramAllocationMethod method = VramAllocationMethod::Unknown;
        };

        constexpr std::size_t kWallRingCapacity = 256;
        constexpr std::size_t kIterRingCapacity = 60;
        constexpr std::size_t kTopNLive = 10;
        constexpr std::size_t kHistRingCapacity = 256;
        constexpr std::size_t kGpuEventPoolSize = 64;
        constexpr std::size_t kAccountedHistoryLength = 64;

        template <std::size_t Capacity>
        struct RingBuffer {
            std::array<double, Capacity> data{};
            std::size_t count = 0;
            std::size_t next = 0;

            void push(double v) {
                data[next] = v;
                next = (next + 1) % Capacity;
                if (count < Capacity)
                    ++count;
            }

            void clear() {
                count = 0;
                next = 0;
            }

            [[nodiscard]] bool empty() const { return count == 0; }
            [[nodiscard]] std::size_t size() const { return count; }
        };

        template <std::size_t Capacity>
        [[nodiscard]] double ring_percentile(const RingBuffer<Capacity>& ring, double q) {
            if (ring.empty())
                return 0.0;
            std::array<double, Capacity> scratch{};
            for (std::size_t i = 0; i < ring.count; ++i)
                scratch[i] = ring.data[i];
            std::sort(scratch.begin(), scratch.begin() + ring.count);
            const double rank = q * static_cast<double>(ring.count - 1);
            const auto lo = static_cast<std::size_t>(std::floor(rank));
            const auto hi = static_cast<std::size_t>(std::ceil(rank));
            const double frac = rank - static_cast<double>(lo);
            return scratch[lo] * (1.0 - frac) + scratch[hi] * frac;
        }

        struct ScopeNodeStats {
            std::string path;
            std::string name;
            std::uint32_t depth = 0;
            bool timer_scope = false;
            bool vram_delta_scope = false;
            bool logical_scope = false;
            std::uint64_t timer_call_count = 0;
            double total_ms = 0.0;
            double last_ms = 0.0;
            double max_ms = 0.0;
            RingBuffer<kWallRingCapacity> wall_ring;
            std::uint64_t gpu_call_count = 0;
            double gpu_total_ms = 0.0;
            double gpu_last_ms = 0.0;
            double gpu_max_ms = 0.0;
            std::uint64_t vram_delta_count = 0;
            std::int64_t last_vram_delta_bytes = 0;
            std::int64_t net_vram_delta_bytes = 0;
            std::int64_t max_vram_increase_bytes = 0;
            std::int64_t max_vram_decrease_bytes = 0;
        };

        struct PendingGpuEvent {
            std::string scope;
            cudaEvent_t start = nullptr;
            cudaEvent_t stop = nullptr;
            bool start_recorded = false;
            bool stop_recorded = false;
        };

        struct HistogramSeries {
            RingBuffer<kHistRingCapacity> ring;
            double min_value = 0.0;
            double max_value = 0.0;
            double sum = 0.0;
            std::uint64_t count = 0;
        };

        struct TimerVramSample {
            std::size_t start_used_bytes = 0;
            bool valid = false;
        };

        thread_local std::vector<std::string> g_scope_stack;
        thread_local std::vector<TimerVramSample> g_timer_vram_stack;

        [[nodiscard]] std::string normalize_scope_component(std::string_view text) {
            std::string out;
            out.reserve(text.size());
            bool previous_space = false;
            for (const char c : text) {
                const bool separator_space = c == '\t' || c == '\n' || c == '\r';
                if (separator_space || c == ' ') {
                    if (!previous_space && !out.empty()) {
                        out.push_back(' ');
                    }
                    previous_space = true;
                    continue;
                }
                out.push_back(c);
                previous_space = false;
            }
            while (!out.empty() && out.back() == ' ') {
                out.pop_back();
            }
            return out.empty() ? std::string("unnamed") : out;
        }

        [[nodiscard]] std::string join_scope_stack() {
            if (g_scope_stack.empty()) {
                return "unscoped";
            }

            std::string path;
            for (const auto& item : g_scope_stack) {
                if (item.empty()) {
                    continue;
                }
                if (!path.empty()) {
                    path.push_back('/');
                }
                path += item;
            }
            return path.empty() ? std::string("unscoped") : path;
        }

        [[nodiscard]] std::vector<std::string> split_scope_path(std::string_view text) {
            std::vector<std::string> parts;
            std::string current;
            const auto flush = [&]() {
                auto normalized = normalize_scope_component(current);
                if (!normalized.empty() && normalized != "unnamed") {
                    parts.push_back(std::move(normalized));
                }
                current.clear();
            };

            for (std::size_t i = 0; i < text.size(); ++i) {
                const char c = text[i];
                if (c == '/' || c == '>' || c == '.') {
                    flush();
                    continue;
                }
                if (c == ':' && i + 1 < text.size() && text[i + 1] == ':') {
                    flush();
                    ++i;
                    continue;
                }
                current.push_back(c);
            }
            flush();
            if (parts.empty()) {
                parts.emplace_back("unscoped");
            }
            return parts;
        }

        [[nodiscard]] std::string join_parts(const std::vector<std::string>& parts,
                                             const std::size_t count) {
            std::string path;
            for (std::size_t i = 0; i < count && i < parts.size(); ++i) {
                if (!path.empty()) {
                    path.push_back('/');
                }
                path += parts[i];
            }
            return path;
        }

        [[nodiscard]] std::string current_scope() {
            return join_scope_stack();
        }

        [[nodiscard]] bool sample_cuda_used_bytes(std::size_t& used_bytes,
                                                  std::size_t* total_bytes = nullptr) {
            std::size_t free_bytes = 0;
            std::size_t total = 0;
            if (cudaMemGetInfo(&free_bytes, &total) != cudaSuccess || total < free_bytes) {
                return false;
            }
            used_bytes = total - free_bytes;
            if (total_bytes) {
                *total_bytes = total;
            }
            return true;
        }

        [[nodiscard]] std::string method_label(const VramAllocationMethod method) {
            switch (method) {
            case VramAllocationMethod::Slab: return "slab";
            case VramAllocationMethod::Bucketed: return "bucketed";
            case VramAllocationMethod::Async: return "async";
            case VramAllocationMethod::Direct: return "direct";
            case VramAllocationMethod::Arena: return "arena";
            case VramAllocationMethod::External: return "external";
            case VramAllocationMethod::Unknown:
            default: return "unknown";
            }
        }

        void upsert_scope_node(std::unordered_map<std::string, ScopeNodeStats>& nodes,
                               std::string_view path,
                               const bool logical,
                               const bool timer,
                               const bool vram_delta) {
            const auto parts = split_scope_path(path);
            for (std::size_t i = 1; i <= parts.size(); ++i) {
                const auto node_path = join_parts(parts, i);
                auto& node = nodes[node_path];
                if (node.path.empty()) {
                    node.path = node_path;
                    node.name = parts[i - 1];
                    node.depth = static_cast<std::uint32_t>(i - 1);
                }
                node.logical_scope = node.logical_scope || logical;
                node.timer_scope = node.timer_scope || (timer && i == parts.size());
                node.vram_delta_scope = node.vram_delta_scope || (vram_delta && i == parts.size());
            }
        }

        [[nodiscard]] bool scope_matches_prefix(const std::string& scope,
                                                const std::string_view prefix) {
            if (prefix.empty()) {
                return false;
            }
            if (scope == prefix) {
                return true;
            }
            if (scope.size() <= prefix.size() || scope.compare(0, prefix.size(), prefix) != 0) {
                return false;
            }
            const char separator = scope[prefix.size()];
            return separator == '.' || separator == '/' || separator == '>';
        }

    } // namespace

    struct VramProfiler::Impl {
        mutable std::mutex mutex;
        std::atomic_bool enabled{false};
        std::atomic<int> iteration{0};
        std::atomic<std::uint64_t> sequence{0};
        std::unordered_map<MetricKey, Metric, MetricKeyHash> metrics;
        std::unordered_map<MetricKey, Metric, MetricKeyHash> static_metrics;
        std::unordered_map<void*, AllocationRecord> allocations;
        std::unordered_map<std::string, ScopeNodeStats> scope_nodes;
        VramProcessSnapshot process;
        std::size_t pinned_host_used = 0;
        std::size_t pinned_host_cached = 0;
        std::size_t pinned_host_peak = 0;
        std::size_t vulkan_vma_used = 0;
        std::size_t vulkan_vma_block_bytes = 0;
        std::size_t exportable_splat_bytes = 0;
        std::size_t cuda_slab_reserved_bytes = 0;
        // Pushed lock-free from the tensor pool's hot path; read into the snapshot
        // under the mutex. A lossy gauge, so no sequence bump on update.
        std::atomic<std::size_t> cuda_pool_bucket_cache_bytes{0};
        std::size_t cuda_context_baseline = 0;
        std::size_t cuda_warmup_bytes = 0;
        std::size_t cuda_device_baseline = 0;
        std::unordered_map<std::string, std::size_t> cuda_phase_bytes;
        std::uint64_t allocation_events = 0;
        std::uint64_t free_events = 0;
        std::uint64_t iter_allocation_events_start = 0;
        std::uint64_t iter_free_events_start = 0;
        std::size_t accounted_live_bytes = 0;
        std::size_t accounted_peak_bytes = 0;

        std::unordered_map<std::string, double> gauges;
        std::uint64_t gauge_generation = 0;
        std::unordered_map<std::string, std::uint64_t> iter_counters;
        std::unordered_map<std::string, std::uint64_t> total_counters;
        std::unordered_map<std::string, HistogramSeries> histograms;

        std::array<PendingGpuEvent, kGpuEventPoolSize> gpu_event_pool{};
        std::array<bool, kGpuEventPoolSize> gpu_event_in_use{};
        std::deque<std::int32_t> gpu_event_pending;

        RingBuffer<kIterRingCapacity> iter_ms_ring;
        double iter_ms_last = 0.0;
        std::chrono::steady_clock::time_point last_iteration_tp{};
        std::chrono::steady_clock::time_point iter_window_origin{};
        std::uint64_t iter_window_count = 0;
        double iter_per_second = 0.0;
        std::deque<std::size_t> accounted_history;
    };

    VramProfiler& VramProfiler::instance() {
        static VramProfiler profiler;
        return profiler;
    }

    VramProfiler::VramProfiler()
        : impl_(new Impl) {}

    VramProfiler::~VramProfiler() {
        delete impl_;
    }

    void VramProfiler::setEnabled(const bool enabled) {
        impl_->enabled.store(enabled, std::memory_order_release);
        if (!enabled) {
            std::lock_guard lock(impl_->mutex);
            impl_->metrics.clear();
            impl_->allocations.clear();
            impl_->scope_nodes.clear();
            impl_->accounted_live_bytes = 0;
            impl_->accounted_peak_bytes = 0;
            impl_->allocation_events = 0;
            impl_->free_events = 0;
            impl_->iter_allocation_events_start = 0;
            impl_->iter_free_events_start = 0;
            impl_->gauges.clear();
            impl_->iter_counters.clear();
            impl_->total_counters.clear();
            impl_->histograms.clear();
            impl_->iter_ms_ring.clear();
            impl_->iter_ms_last = 0.0;
            impl_->iter_per_second = 0.0;
            impl_->last_iteration_tp = std::chrono::steady_clock::time_point{};
            impl_->iter_window_origin = std::chrono::steady_clock::time_point{};
            impl_->iter_window_count = 0;
            impl_->accounted_history.clear();
            impl_->pinned_host_used = 0;
            impl_->pinned_host_cached = 0;
            impl_->pinned_host_peak = 0;
            impl_->vulkan_vma_used = 0;
            impl_->vulkan_vma_block_bytes = 0;
            impl_->exportable_splat_bytes = 0;
            impl_->cuda_slab_reserved_bytes = 0;
            // cuda_context_baseline intentionally preserved: it captures the irreducible
            // runtime overhead once at process startup and is valid for the session.
            impl_->sequence.fetch_add(1, std::memory_order_relaxed);
        }
    }

    bool VramProfiler::enabled() const {
        return impl_->enabled.load(std::memory_order_acquire);
    }

    void VramProfiler::beginIteration(const int iteration) {
        impl_->iteration.store(iteration, std::memory_order_relaxed);
        if (!enabled()) {
            return;
        }

        std::lock_guard lock(impl_->mutex);

        const auto now = std::chrono::steady_clock::now();
        if (impl_->last_iteration_tp != std::chrono::steady_clock::time_point{}) {
            const double iter_ms =
                std::chrono::duration<double, std::milli>(now - impl_->last_iteration_tp).count();
            impl_->iter_ms_last = iter_ms;
            impl_->iter_ms_ring.push(iter_ms);
        } else {
            impl_->iter_window_origin = now;
            impl_->iter_window_count = 0;
        }
        impl_->last_iteration_tp = now;
        ++impl_->iter_window_count;
        const double window_seconds =
            std::chrono::duration<double>(now - impl_->iter_window_origin).count();
        if (window_seconds >= 1.0) {
            impl_->iter_per_second = static_cast<double>(impl_->iter_window_count) / window_seconds;
            impl_->iter_window_origin = now;
            impl_->iter_window_count = 0;
        }

        impl_->accounted_history.push_back(impl_->accounted_live_bytes);
        while (impl_->accounted_history.size() > kAccountedHistoryLength)
            impl_->accounted_history.pop_front();

        impl_->iter_counters.clear();
        impl_->iter_allocation_events_start = impl_->allocation_events;
        impl_->iter_free_events_start = impl_->free_events;

        // live_bytes is derived state: zero it for every allocator-tracked metric, then
        // rebuild from the ground-truth live allocations map below. Sampled (current_sample)
        // metrics own their own live_bytes via recordCurrentBytes and must not be reset here.
        for (auto& [_, metric] : impl_->metrics) {
            if (!metric.current_sample) {
                metric.live_bytes = 0;
            }
        }
        // Per-iteration timer/delta stats reset on every scope (not just a hard-coded
        // prefix list — nested top-level scopes like "Training execution/train.*" never
        // matched the prefix and were silently accumulating call counts across runs).
        for (auto& [_, node] : impl_->scope_nodes) {
            node.timer_call_count = 0;
            node.total_ms = 0.0;
            node.last_ms = 0.0;
            node.wall_ring.clear();
            node.gpu_call_count = 0;
            node.gpu_total_ms = 0.0;
            node.gpu_last_ms = 0.0;
            node.vram_delta_count = 0;
            node.last_vram_delta_bytes = 0;
            node.net_vram_delta_bytes = 0;
            // max_ms / gpu_max_ms / max_vram_increase / max_vram_decrease keep lifetime maxima.
        }
        for (const auto& [_, allocation] : impl_->allocations) {
            auto& metric = impl_->metrics[allocation.key];
            metric.live_bytes += allocation.bytes;
            metric.peak_bytes = std::max(metric.peak_bytes, metric.live_bytes);
            metric.allocated_bytes = std::max(metric.allocated_bytes, metric.live_bytes);
            upsert_scope_node(impl_->scope_nodes, allocation.key.scope, false, false, false);
        }
        impl_->sequence.fetch_add(1, std::memory_order_relaxed);
    }

    void VramProfiler::setIteration(const int iteration) {
        impl_->iteration.store(iteration, std::memory_order_relaxed);
    }

    void VramProfiler::pushScope(std::string_view scope) {
        if (!enabled()) {
            return;
        }
        bool pushed = false;
        try {
            g_scope_stack.emplace_back(normalize_scope_component(scope));
            pushed = true;
            const auto path = current_scope();
            std::lock_guard lock(impl_->mutex);
            upsert_scope_node(impl_->scope_nodes, path, true, false, false);
            impl_->sequence.fetch_add(1, std::memory_order_relaxed);
        } catch (...) {
            if (pushed && !g_scope_stack.empty()) {
                g_scope_stack.pop_back();
            }
            throw;
        }
    }

    void VramProfiler::popScope() {
        if (!g_scope_stack.empty()) {
            g_scope_stack.pop_back();
        }
    }

    void VramProfiler::pushTimerScope(std::string_view scope) {
        if (!enabled()) {
            return;
        }
        bool pushed = false;
        bool timer_sample_pushed = false;
        try {
            g_scope_stack.emplace_back(normalize_scope_component(scope));
            pushed = true;
            auto& sample = g_timer_vram_stack.emplace_back();
            timer_sample_pushed = true;
            sample.valid = sample_cuda_used_bytes(sample.start_used_bytes);
            const auto path = current_scope();
            std::lock_guard lock(impl_->mutex);
            upsert_scope_node(impl_->scope_nodes, path, false, true, false);
            impl_->sequence.fetch_add(1, std::memory_order_relaxed);
        } catch (...) {
            if (timer_sample_pushed && !g_timer_vram_stack.empty()) {
                g_timer_vram_stack.pop_back();
            }
            if (pushed && !g_scope_stack.empty()) {
                g_scope_stack.pop_back();
            }
            throw;
        }
    }

    void VramProfiler::popTimerScope(const double elapsed_ms) {
        const auto path = current_scope();
        TimerVramSample timer_sample;
        const bool has_timer_sample = !g_timer_vram_stack.empty();
        if (has_timer_sample) {
            timer_sample = g_timer_vram_stack.back();
            g_timer_vram_stack.pop_back();
        }
        std::size_t end_used_bytes = 0;
        const bool end_valid = enabled() && has_timer_sample && timer_sample.valid &&
                               sample_cuda_used_bytes(end_used_bytes);
        if (!g_scope_stack.empty()) {
            g_scope_stack.pop_back();
        }
        if (enabled() && !path.empty()) {
            std::lock_guard lock(impl_->mutex);
            upsert_scope_node(impl_->scope_nodes, path, false, true, false);
            const auto parts = split_scope_path(path);
            auto& node = impl_->scope_nodes[join_parts(parts, parts.size())];
            node.timer_scope = true;
            node.timer_call_count += 1;
            node.total_ms += std::max(elapsed_ms, 0.0);
            node.last_ms = std::max(elapsed_ms, 0.0);
            node.max_ms = std::max(node.max_ms, node.last_ms);
            node.wall_ring.push(std::max(elapsed_ms, 0.0));
            if (end_valid) {
                const auto delta = static_cast<std::int64_t>(end_used_bytes) -
                                   static_cast<std::int64_t>(timer_sample.start_used_bytes);
                node.vram_delta_count += 1;
                node.last_vram_delta_bytes = delta;
                node.net_vram_delta_bytes += delta;
                node.max_vram_increase_bytes = std::max(node.max_vram_increase_bytes, delta);
                node.max_vram_decrease_bytes = std::min(node.max_vram_decrease_bytes, delta);
            }
            impl_->sequence.fetch_add(1, std::memory_order_relaxed);
        }
    }

    bool VramProfiler::pushVramDeltaScope(std::string_view scope,
                                          std::size_t& start_used_bytes,
                                          bool& pushed_scope) {
        if (!enabled()) {
            return false;
        }

        pushed_scope = false;
        try {
            const auto normalized = normalize_scope_component(scope);
            if (g_scope_stack.empty() || g_scope_stack.back() != normalized) {
                g_scope_stack.emplace_back(normalized);
                pushed_scope = true;
            }
            const auto start_valid = sample_cuda_used_bytes(start_used_bytes);
            const auto path = current_scope();
            std::lock_guard lock(impl_->mutex);
            upsert_scope_node(impl_->scope_nodes, path, false, false, true);
            impl_->sequence.fetch_add(1, std::memory_order_relaxed);
            return start_valid;
        } catch (...) {
            if (pushed_scope && !g_scope_stack.empty()) {
                g_scope_stack.pop_back();
            }
            pushed_scope = false;
            throw;
        }
    }

    void VramProfiler::popVramDeltaScope(const std::size_t start_used_bytes,
                                         const bool start_valid,
                                         const bool pushed_scope) {
        const auto path = current_scope();
        std::size_t end_used_bytes = 0;
        const bool end_valid = start_valid && sample_cuda_used_bytes(end_used_bytes);
        if (pushed_scope && !g_scope_stack.empty()) {
            g_scope_stack.pop_back();
        }
        if (enabled() && end_valid && !path.empty()) {
            const auto delta = static_cast<std::int64_t>(end_used_bytes) -
                               static_cast<std::int64_t>(start_used_bytes);
            const auto parts = split_scope_path(path);
            const auto node_path = join_parts(parts, parts.size());

            std::lock_guard lock(impl_->mutex);
            upsert_scope_node(impl_->scope_nodes, path, false, false, true);
            auto& node = impl_->scope_nodes[node_path];
            node.vram_delta_scope = true;
            node.vram_delta_count += 1;
            node.last_vram_delta_bytes = delta;
            node.net_vram_delta_bytes += delta;
            node.max_vram_increase_bytes = std::max(node.max_vram_increase_bytes, delta);
            node.max_vram_decrease_bytes = std::min(node.max_vram_decrease_bytes, delta);
            impl_->sequence.fetch_add(1, std::memory_order_relaxed);
        }
    }

    void VramProfiler::recordAllocation(void* ptr,
                                        const std::size_t bytes,
                                        const VramAllocationMethod method) {
        recordAllocation(ptr, bytes, method, {});
    }

    void VramProfiler::recordAllocation(void* ptr,
                                        const std::size_t bytes,
                                        const VramAllocationMethod method,
                                        std::string_view label) {
        if (!enabled() || !ptr || bytes == 0) {
            return;
        }

        MetricKey key{
            .scope = current_scope(),
            .label = label.empty() ? method_label(method) : std::string(label),
        };

        std::lock_guard lock(impl_->mutex);
        auto& metric = impl_->metrics[key];
        metric.live_bytes += bytes;
        metric.allocated_bytes += bytes;
        metric.peak_bytes = std::max(metric.peak_bytes, metric.live_bytes);
        metric.allocation_count += 1;
        metric.method = method;
        upsert_scope_node(impl_->scope_nodes, key.scope, false, false, false);

        impl_->allocations[ptr] = AllocationRecord{
            .key = key,
            .bytes = bytes,
            .method = method,
        };
        impl_->accounted_live_bytes += bytes;
        impl_->accounted_peak_bytes = std::max(impl_->accounted_peak_bytes, impl_->accounted_live_bytes);
        impl_->allocation_events += 1;
        impl_->sequence.fetch_add(1, std::memory_order_relaxed);
    }

    void VramProfiler::relabelAllocation(void* ptr, std::string_view label) {
        if (!enabled() || !ptr || label.empty()) {
            return;
        }
        std::lock_guard lock(impl_->mutex);
        const auto alloc_it = impl_->allocations.find(ptr);
        if (alloc_it == impl_->allocations.end()) {
            return;
        }
        if (alloc_it->second.key.label == label) {
            return;
        }
        const auto bytes = alloc_it->second.bytes;
        const auto old_key = alloc_it->second.key;
        const MetricKey new_key{old_key.scope, std::string(label)};

        // Move live_bytes from old key to new key.
        auto& old_metric = impl_->metrics[old_key];
        old_metric.live_bytes = bytes > old_metric.live_bytes ? 0 : old_metric.live_bytes - bytes;

        auto& new_metric = impl_->metrics[new_key];
        new_metric.live_bytes += bytes;
        new_metric.peak_bytes = std::max(new_metric.peak_bytes, new_metric.live_bytes);
        new_metric.allocated_bytes = std::max(new_metric.allocated_bytes, new_metric.live_bytes);
        if (new_metric.allocation_count == 0)
            new_metric.allocation_count = 1;
        new_metric.method = alloc_it->second.method;

        alloc_it->second.key = new_key;
        upsert_scope_node(impl_->scope_nodes, new_key.scope, false, false, false);
        impl_->sequence.fetch_add(1, std::memory_order_relaxed);
    }

    void VramProfiler::recordDeallocation(void* ptr) {
        if (!enabled() || !ptr) {
            return;
        }

        std::lock_guard lock(impl_->mutex);
        const auto alloc_it = impl_->allocations.find(ptr);
        if (alloc_it == impl_->allocations.end()) {
            return;
        }

        const auto record = std::move(alloc_it->second);
        impl_->allocations.erase(alloc_it);

        auto& metric = impl_->metrics[record.key];
        metric.live_bytes = record.bytes > metric.live_bytes ? 0 : metric.live_bytes - record.bytes;
        metric.freed_bytes += record.bytes;
        metric.free_count += 1;
        upsert_scope_node(impl_->scope_nodes, record.key.scope, false, false, false);

        impl_->accounted_live_bytes =
            record.bytes > impl_->accounted_live_bytes ? 0 : impl_->accounted_live_bytes - record.bytes;
        impl_->free_events += 1;
        impl_->sequence.fetch_add(1, std::memory_order_relaxed);
    }

    void VramProfiler::recordBytes(std::string_view scope,
                                   std::string_view label,
                                   const std::size_t bytes,
                                   const VramAllocationMethod method) {
        if (!enabled() || bytes == 0 || scope.empty() || label.empty()) {
            return;
        }

        std::lock_guard lock(impl_->mutex);
        auto& metric = impl_->metrics[MetricKey{std::string(scope), std::string(label)}];
        metric.allocated_bytes += bytes;
        metric.peak_bytes = std::max(metric.peak_bytes, bytes);
        metric.allocation_count += 1;
        metric.method = method;
        upsert_scope_node(impl_->scope_nodes, scope, false, false, false);
        impl_->sequence.fetch_add(1, std::memory_order_relaxed);
    }

    void VramProfiler::recordCurrentBytes(std::string_view scope,
                                          std::string_view label,
                                          const std::size_t bytes,
                                          const VramAllocationMethod method) {
        if (!enabled() || scope.empty() || label.empty()) {
            return;
        }

        std::lock_guard lock(impl_->mutex);
        auto& metric = impl_->metrics[MetricKey{std::string(scope), std::string(label)}];
        metric.live_bytes = bytes;
        metric.peak_bytes = std::max(metric.peak_bytes, bytes);
        metric.allocated_bytes = std::max(metric.allocated_bytes, bytes);
        metric.allocation_count = std::max<std::uint64_t>(metric.allocation_count, bytes > 0 ? 1 : 0);
        metric.method = method;
        metric.current_sample = true;
        upsert_scope_node(impl_->scope_nodes, scope, false, false, false);
        impl_->sequence.fetch_add(1, std::memory_order_relaxed);
    }

    void VramProfiler::recordStaticBytes(std::string_view scope,
                                         std::string_view label,
                                         const std::size_t bytes,
                                         const VramAllocationMethod method) {
        if (bytes == 0 || scope.empty() || label.empty()) {
            return;
        }

        std::lock_guard lock(impl_->mutex);
        auto& metric = impl_->static_metrics[MetricKey{std::string(scope), std::string(label)}];
        metric.live_bytes = bytes;
        metric.peak_bytes = std::max(metric.peak_bytes, bytes);
        metric.allocated_bytes = std::max(metric.allocated_bytes, bytes);
        metric.allocation_count = 1;
        metric.method = method;
        metric.current_sample = false;
        impl_->sequence.fetch_add(1, std::memory_order_relaxed);
    }

    void VramProfiler::clearStaticScope(std::string_view scope) {
        if (scope.empty()) {
            return;
        }

        bool erased_any = false;
        std::lock_guard lock(impl_->mutex);
        for (auto it = impl_->static_metrics.begin(); it != impl_->static_metrics.end();) {
            if (scope_matches_prefix(it->first.scope, scope)) {
                it = impl_->static_metrics.erase(it);
                erased_any = true;
            } else {
                ++it;
            }
        }
        if (erased_any) {
            impl_->sequence.fetch_add(1, std::memory_order_relaxed);
        }
    }

    void VramProfiler::recordTimerSample(std::string_view scope, const double elapsed_ms) {
        if (!enabled() || scope.empty() || !std::isfinite(elapsed_ms)) {
            return;
        }

        std::lock_guard lock(impl_->mutex);
        const auto parts = split_scope_path(scope);
        if (parts.empty()) {
            return;
        }
        const auto path = join_parts(parts, parts.size());
        upsert_scope_node(impl_->scope_nodes, path, false, true, false);
        auto& node = impl_->scope_nodes[path];
        node.timer_scope = true;
        node.timer_call_count += 1;
        node.total_ms += elapsed_ms;
        node.last_ms = elapsed_ms;
        node.max_ms = std::max(node.max_ms, elapsed_ms);
        node.wall_ring.push(std::max(elapsed_ms, 0.0));
        impl_->sequence.fetch_add(1, std::memory_order_relaxed);
    }

    void VramProfiler::recordGpuTimerSample(std::string_view scope, const double elapsed_ms) {
        if (!enabled() || scope.empty() || !std::isfinite(elapsed_ms) || elapsed_ms < 0.0) {
            return;
        }
        std::lock_guard lock(impl_->mutex);
        const auto parts = split_scope_path(scope);
        if (parts.empty()) {
            return;
        }
        const auto path = join_parts(parts, parts.size());
        upsert_scope_node(impl_->scope_nodes, path, false, true, false);
        auto& node = impl_->scope_nodes[path];
        node.timer_scope = true;
        node.gpu_call_count += 1;
        node.gpu_total_ms += elapsed_ms;
        node.gpu_last_ms = elapsed_ms;
        node.gpu_max_ms = std::max(node.gpu_max_ms, elapsed_ms);
        impl_->sequence.fetch_add(1, std::memory_order_relaxed);
    }

    std::int32_t VramProfiler::acquireGpuEventPair(std::string_view scope, void* stream) {
        if (!enabled()) {
            return -1;
        }
        std::lock_guard lock(impl_->mutex);
        for (std::size_t i = 0; i < kGpuEventPoolSize; ++i) {
            if (impl_->gpu_event_in_use[i]) {
                continue;
            }
            auto& slot = impl_->gpu_event_pool[i];
            if (!slot.start) {
                if (cudaEventCreateWithFlags(&slot.start, cudaEventDefault) != cudaSuccess) {
                    return -1;
                }
            }
            if (!slot.stop) {
                if (cudaEventCreateWithFlags(&slot.stop, cudaEventDefault) != cudaSuccess) {
                    return -1;
                }
            }
            slot.scope = std::string(scope);
            slot.start_recorded = false;
            slot.stop_recorded = false;
            if (cudaEventRecord(slot.start, static_cast<cudaStream_t>(stream)) != cudaSuccess) {
                return -1;
            }
            slot.start_recorded = true;
            impl_->gpu_event_in_use[i] = true;
            return static_cast<std::int32_t>(i);
        }
        return -1;
    }

    void VramProfiler::releaseGpuEventPair(const std::int32_t pair, void* stream) {
        if (pair < 0 || static_cast<std::size_t>(pair) >= kGpuEventPoolSize) {
            return;
        }
        std::lock_guard lock(impl_->mutex);
        auto& slot = impl_->gpu_event_pool[static_cast<std::size_t>(pair)];
        if (!impl_->gpu_event_in_use[static_cast<std::size_t>(pair)]) {
            return;
        }
        if (slot.start_recorded && slot.stop) {
            if (cudaEventRecord(slot.stop, static_cast<cudaStream_t>(stream)) == cudaSuccess) {
                slot.stop_recorded = true;
                impl_->gpu_event_pending.push_back(pair);
                return;
            }
        }
        impl_->gpu_event_in_use[static_cast<std::size_t>(pair)] = false;
    }

    void VramProfiler::drainGpuEvents() {
        if (!enabled()) {
            return;
        }
        std::lock_guard lock(impl_->mutex);
        for (auto it = impl_->gpu_event_pending.begin(); it != impl_->gpu_event_pending.end();) {
            const auto idx = static_cast<std::size_t>(*it);
            auto& slot = impl_->gpu_event_pool[idx];
            const auto status = cudaEventQuery(slot.stop);
            if (status == cudaErrorNotReady) {
                ++it;
                continue;
            }
            float elapsed_ms = 0.0f;
            if (status == cudaSuccess &&
                cudaEventElapsedTime(&elapsed_ms, slot.start, slot.stop) == cudaSuccess) {
                const auto parts = split_scope_path(slot.scope);
                if (!parts.empty()) {
                    const auto path = join_parts(parts, parts.size());
                    upsert_scope_node(impl_->scope_nodes, path, false, true, false);
                    auto& node = impl_->scope_nodes[path];
                    node.timer_scope = true;
                    node.gpu_call_count += 1;
                    node.gpu_total_ms += static_cast<double>(elapsed_ms);
                    node.gpu_last_ms = static_cast<double>(elapsed_ms);
                    node.gpu_max_ms = std::max(node.gpu_max_ms, node.gpu_last_ms);
                    impl_->sequence.fetch_add(1, std::memory_order_relaxed);
                }
            }
            impl_->gpu_event_in_use[idx] = false;
            it = impl_->gpu_event_pending.erase(it);
        }
    }

    void VramProfiler::setGauge(std::string_view key, const double value) {
        if (!enabled() || key.empty()) {
            return;
        }
        std::lock_guard lock(impl_->mutex);
        impl_->gauges[std::string(key)] = value;
        ++impl_->gauge_generation;
        impl_->sequence.fetch_add(1, std::memory_order_relaxed);
    }

    void VramProfiler::addCounter(std::string_view key, const std::uint64_t delta, const bool per_iteration) {
        if (!enabled() || key.empty() || delta == 0) {
            return;
        }
        std::lock_guard lock(impl_->mutex);
        if (per_iteration) {
            impl_->iter_counters[std::string(key)] += delta;
        }
        impl_->total_counters[std::string(key)] += delta;
        impl_->sequence.fetch_add(1, std::memory_order_relaxed);
    }

    void VramProfiler::recordHistogram(std::string_view key, const double value) {
        if (!enabled() || key.empty() || !std::isfinite(value)) {
            return;
        }
        std::lock_guard lock(impl_->mutex);
        auto& hist = impl_->histograms[std::string(key)];
        if (hist.count == 0) {
            hist.min_value = value;
            hist.max_value = value;
        } else {
            hist.min_value = std::min(hist.min_value, value);
            hist.max_value = std::max(hist.max_value, value);
        }
        hist.sum += value;
        ++hist.count;
        hist.ring.push(value);
        impl_->sequence.fetch_add(1, std::memory_order_relaxed);
    }

    void VramProfiler::setPinnedHostMemory(const std::size_t active_bytes,
                                           const std::size_t cached_bytes,
                                           const std::size_t peak_bytes) {
        std::lock_guard lock(impl_->mutex);
        impl_->pinned_host_used = active_bytes;
        impl_->pinned_host_cached = cached_bytes;
        impl_->pinned_host_peak = peak_bytes;
        impl_->process.pinned_host_used = active_bytes;
        impl_->process.pinned_host_cached = cached_bytes;
        impl_->process.pinned_host_peak = peak_bytes;
        impl_->sequence.fetch_add(1, std::memory_order_relaxed);
    }

    void VramProfiler::setVulkanVmaUsed(const std::size_t bytes) {
        std::lock_guard lock(impl_->mutex);
        impl_->vulkan_vma_used = bytes;
        impl_->process.vulkan_vma_used = bytes;
        impl_->sequence.fetch_add(1, std::memory_order_relaxed);
    }

    void VramProfiler::setVulkanVmaBlockBytes(const std::size_t bytes) {
        std::lock_guard lock(impl_->mutex);
        impl_->vulkan_vma_block_bytes = bytes;
        impl_->process.vulkan_vma_block_bytes = bytes;
        impl_->sequence.fetch_add(1, std::memory_order_relaxed);
    }

    void VramProfiler::setCudaPoolBucketCacheBytes(const std::size_t bytes) {
        // Lock-free: called from the tensor allocator's per-allocation path. Read
        // into the process snapshot during the periodic sampleCudaMemory pass.
        impl_->cuda_pool_bucket_cache_bytes.store(bytes, std::memory_order_relaxed);
    }

    void VramProfiler::setCudaSlabReservedBytes(const std::size_t bytes) {
        std::lock_guard lock(impl_->mutex);
        impl_->cuda_slab_reserved_bytes = bytes;
        impl_->process.cuda_slab_reserved_bytes = bytes;
        impl_->sequence.fetch_add(1, std::memory_order_relaxed);
    }

    void VramProfiler::setExportableSplatBytes(const std::size_t bytes) {
        std::lock_guard lock(impl_->mutex);
        impl_->exportable_splat_bytes = bytes;
        impl_->process.exportable_splat_bytes = bytes;
        impl_->sequence.fetch_add(1, std::memory_order_relaxed);
    }

    void VramProfiler::captureCudaContextBaseline() {
        std::size_t used = 0;
        if (!sample_cuda_used_bytes(used))
            return;
        std::lock_guard lock(impl_->mutex);
        if (impl_->cuda_context_baseline == 0) {
            impl_->cuda_context_baseline = used;
            impl_->process.cuda_context_baseline = used;
        }
        impl_->sequence.fetch_add(1, std::memory_order_relaxed);
    }

    void VramProfiler::setCudaContextBaselineBytes(const std::size_t bytes) {
        std::lock_guard lock(impl_->mutex);
        impl_->cuda_context_baseline = bytes;
        impl_->process.cuda_context_baseline = bytes;
        impl_->sequence.fetch_add(1, std::memory_order_relaxed);
    }

    void VramProfiler::recordCudaPhaseBytes(std::string_view phase, std::size_t bytes) {
        std::lock_guard lock(impl_->mutex);
        impl_->cuda_phase_bytes[std::string(phase)] = bytes;
        if (phase == "primary_context")
            impl_->process.cuda_phase_primary_context = bytes;
        else if (phase == "default_pool")
            impl_->process.cuda_phase_default_pool = bytes;
        else if (phase == "printf_fifo")
            impl_->process.cuda_phase_printf_fifo = bytes;
        else if (phase == "stack_reserve")
            impl_->process.cuda_phase_stack_reserve = bytes;
        else if (phase == "malloc_heap")
            impl_->process.cuda_phase_malloc_heap = bytes;
        else if (phase == "curand_load")
            impl_->process.cuda_phase_curand_load = bytes;
        impl_->sequence.fetch_add(1, std::memory_order_relaxed);
    }

    void VramProfiler::captureCudaDeviceBaseline() {
        std::size_t used = 0;
        if (!sample_cuda_used_bytes(used))
            return;
        std::lock_guard lock(impl_->mutex);
        if (impl_->cuda_device_baseline == 0)
            impl_->cuda_device_baseline = used;
        impl_->sequence.fetch_add(1, std::memory_order_relaxed);
    }

    void VramProfiler::captureCudaWarmupDelta() {
        std::size_t used = 0;
        if (!sample_cuda_used_bytes(used))
            return;
        std::lock_guard lock(impl_->mutex);
        if (impl_->cuda_warmup_bytes != 0 || impl_->cuda_device_baseline == 0)
            return;
        // Diff device-wide cudaMemGetInfo against a device-wide baseline captured at
        // the same point — NOT the NVML per-PID context baseline. Mixing the two
        // metrics produced a meaningless offset. This delta is the VRAM the kernel
        // warmup committed (cubin upload + per-launch driver reservations).
        if (used > impl_->cuda_device_baseline) {
            impl_->cuda_warmup_bytes = used - impl_->cuda_device_baseline;
            impl_->process.cuda_warmup_bytes = impl_->cuda_warmup_bytes;
        }
        impl_->sequence.fetch_add(1, std::memory_order_relaxed);
    }

    void VramProfiler::clearScope(std::string_view scope) {
        if (!enabled() || scope.empty()) {
            return;
        }

        std::lock_guard lock(impl_->mutex);
        for (auto it = impl_->metrics.begin(); it != impl_->metrics.end();) {
            if (it->first.scope == scope) {
                it = impl_->metrics.erase(it);
            } else {
                ++it;
            }
        }
        const auto normalized_parts = split_scope_path(scope);
        const auto normalized_scope = join_parts(normalized_parts, normalized_parts.size());
        for (auto it = impl_->scope_nodes.begin(); it != impl_->scope_nodes.end();) {
            if (it->second.path == normalized_scope ||
                it->second.path.rfind(normalized_scope + "/", 0) == 0) {
                it = impl_->scope_nodes.erase(it);
            } else {
                ++it;
            }
        }
        impl_->sequence.fetch_add(1, std::memory_order_relaxed);
    }

    void VramProfiler::sampleCudaMemory() {
        if (!enabled()) {
            return;
        }

        VramProcessSnapshot process;
        {
            std::lock_guard lock(impl_->mutex);
            process = impl_->process;
        }

        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess && total_bytes >= free_bytes) {
            process.cuda_used = total_bytes - free_bytes;
            process.cuda_total = total_bytes;
            process.cuda_memory_valid = true;
        }

#if CUDART_VERSION >= 12080
        int device = 0;
        if (cudaGetDevice(&device) == cudaSuccess) {
            cudaMemPool_t pool = nullptr;
            if (cudaDeviceGetDefaultMemPool(&pool, device) == cudaSuccess) {
                std::uint64_t used = 0;
                std::uint64_t reserved = 0;
                if (cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used) == cudaSuccess &&
                    cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &reserved) == cudaSuccess) {
                    process.cuda_pool_used = static_cast<std::size_t>(used);
                    process.cuda_pool_reserved = static_cast<std::size_t>(reserved);
                    process.cuda_pool_valid = true;
                }
            }
        }
#endif

        {
            std::lock_guard lock(impl_->mutex);
            if (process.cuda_pool_valid) {
                process.cuda_pool_fragmentation =
                    process.cuda_pool_reserved > process.cuda_pool_used
                        ? process.cuda_pool_reserved - process.cuda_pool_used
                        : 0;
            }
            process.pinned_host_used = impl_->pinned_host_used;
            process.pinned_host_cached = impl_->pinned_host_cached;
            process.pinned_host_peak = impl_->pinned_host_peak;
            process.vulkan_vma_used = impl_->vulkan_vma_used;
            process.vulkan_vma_block_bytes = impl_->vulkan_vma_block_bytes;
            process.cuda_pool_bucket_cache_bytes =
                impl_->cuda_pool_bucket_cache_bytes.load(std::memory_order_relaxed);
            process.cuda_slab_reserved_bytes = impl_->cuda_slab_reserved_bytes;
            process.exportable_splat_bytes = impl_->exportable_splat_bytes;
            process.cuda_context_baseline = impl_->cuda_context_baseline;
            process.cuda_warmup_bytes = impl_->cuda_warmup_bytes;
            impl_->process = std::move(process);
            impl_->sequence.fetch_add(1, std::memory_order_relaxed);
        }
        drainGpuEvents();
    }

    void VramProfiler::updateProcessMemory(const std::size_t process_used,
                                           const std::size_t total_used,
                                           const std::size_t total,
                                           std::string device_name) {
        if (!enabled()) {
            return;
        }

        std::lock_guard lock(impl_->mutex);
        impl_->process.process_used = process_used;
        impl_->process.total_used = total_used;
        impl_->process.total = total;
        impl_->process.device_name = std::move(device_name);
        impl_->process.process_memory_valid = process_used > 0 || total_used > 0 || total > 0;
        if (!impl_->process.cuda_memory_valid && total_used > 0 && total > 0) {
            impl_->process.cuda_used = total_used;
            impl_->process.cuda_total = total;
            impl_->process.cuda_memory_valid = true;
        }
        impl_->sequence.fetch_add(1, std::memory_order_relaxed);
    }

    VramProfilerSnapshot VramProfiler::snapshot() const {
        std::lock_guard lock(impl_->mutex);
        VramProfilerSnapshot out;
        out.enabled = enabled();
        out.iteration = impl_->iteration.load(std::memory_order_relaxed);
        out.sequence = impl_->sequence.load(std::memory_order_relaxed);
        out.allocation_events = impl_->allocation_events;
        out.free_events = impl_->free_events;
        out.iter_allocation_events =
            impl_->allocation_events > impl_->iter_allocation_events_start
                ? impl_->allocation_events - impl_->iter_allocation_events_start
                : 0;
        out.iter_free_events = impl_->free_events > impl_->iter_free_events_start
                                   ? impl_->free_events - impl_->iter_free_events_start
                                   : 0;
        out.iter_per_second = impl_->iter_per_second;
        out.iter_ms_p95 = ring_percentile(impl_->iter_ms_ring, 0.95);
        out.iter_ms_last = impl_->iter_ms_last;
        out.accounted_live_bytes = impl_->accounted_live_bytes;
        const auto add_accounted_method = [&](const VramAllocationMethod method,
                                              const std::size_t bytes) {
            switch (method) {
            case VramAllocationMethod::Slab:
                out.accounted_slab_live_bytes += bytes;
                out.accounted_cuda_pool_live_bytes += bytes;
                break;
            case VramAllocationMethod::Bucketed:
                out.accounted_bucketed_live_bytes += bytes;
                out.accounted_cuda_pool_live_bytes += bytes;
                break;
            case VramAllocationMethod::Async:
                out.accounted_async_live_bytes += bytes;
                out.accounted_cuda_pool_live_bytes += bytes;
                break;
            case VramAllocationMethod::Direct:
                out.accounted_direct_live_bytes += bytes;
                break;
            case VramAllocationMethod::Arena:
                out.accounted_arena_live_bytes += bytes;
                break;
            case VramAllocationMethod::External:
                out.accounted_external_live_bytes += bytes;
                break;
            case VramAllocationMethod::Unknown:
            default:
                out.accounted_unknown_live_bytes += bytes;
                break;
            }
        };
        for (const auto& [_, allocation] : impl_->allocations) {
            add_accounted_method(allocation.method, allocation.bytes);
        }
        for (const auto& [_, metric] : impl_->static_metrics) {
            add_accounted_method(metric.method, metric.live_bytes);
        }
        out.accounted_peak_bytes = impl_->accounted_peak_bytes;
        out.accounted_live_history.assign(impl_->accounted_history.begin(),
                                          impl_->accounted_history.end());
        out.process = impl_->process;
        out.rows.reserve(impl_->metrics.size() + impl_->static_metrics.size());
        std::unordered_map<std::string, VramTreeNodeSnapshot> tree_nodes;

        const auto ensure_tree_path = [&](const std::vector<std::string>& parts,
                                          const std::size_t count) -> VramTreeNodeSnapshot& {
            const auto path = join_parts(parts, count);
            auto& node = tree_nodes[path];
            if (node.path.empty()) {
                node.path = path;
                node.name = parts[count - 1];
                node.depth = static_cast<std::uint32_t>(count - 1);
            }
            return node;
        };

        const auto add_metric_to_tree = [&](const MetricKey& key, const Metric& metric) {
            std::vector<std::string> parts = split_scope_path(key.scope);
            if (!key.label.empty()) {
                auto label_parts = split_scope_path(key.label);
                parts.insert(parts.end(),
                             std::make_move_iterator(label_parts.begin()),
                             std::make_move_iterator(label_parts.end()));
            }
            for (std::size_t i = 1; i <= parts.size(); ++i) {
                auto& node = ensure_tree_path(parts, i);
                node.has_metrics = true;
                node.live_bytes += metric.live_bytes;
                node.peak_bytes += metric.peak_bytes;
                node.allocated_bytes += metric.allocated_bytes;
                node.freed_bytes += metric.freed_bytes;
                node.allocation_count += metric.allocation_count;
                node.free_count += metric.free_count;
            }
        };

        const auto add_scope_to_tree = [&](const ScopeNodeStats& stats) {
            const auto parts = split_scope_path(stats.path);
            for (std::size_t i = 1; i <= parts.size(); ++i) {
                (void)ensure_tree_path(parts, i);
            }

            auto& node = ensure_tree_path(parts, parts.size());
            node.logical_scope = node.logical_scope || stats.logical_scope;
            node.timer_scope = node.timer_scope || stats.timer_scope;
            node.vram_delta_scope = node.vram_delta_scope || stats.vram_delta_scope;
            node.timer_call_count += stats.timer_call_count;
            node.total_ms += stats.total_ms;
            node.last_ms = std::max(node.last_ms, stats.last_ms);
            node.max_ms = std::max(node.max_ms, stats.max_ms);
            if (!stats.wall_ring.empty()) {
                node.wall_p50_ms = ring_percentile(stats.wall_ring, 0.50);
                node.wall_p95_ms = ring_percentile(stats.wall_ring, 0.95);
                node.wall_p99_ms = ring_percentile(stats.wall_ring, 0.99);
            }
            node.gpu_call_count += stats.gpu_call_count;
            node.gpu_total_ms += stats.gpu_total_ms;
            node.gpu_last_ms = std::max(node.gpu_last_ms, stats.gpu_last_ms);
            node.gpu_max_ms = std::max(node.gpu_max_ms, stats.gpu_max_ms);
            node.vram_delta_count += stats.vram_delta_count;
            if (stats.vram_delta_count > 0) {
                node.last_vram_delta_bytes = stats.last_vram_delta_bytes;
            }
            node.net_vram_delta_bytes += stats.net_vram_delta_bytes;
            node.max_vram_increase_bytes = std::max(node.max_vram_increase_bytes,
                                                    stats.max_vram_increase_bytes);
            node.max_vram_decrease_bytes = std::min(node.max_vram_decrease_bytes,
                                                    stats.max_vram_decrease_bytes);
        };

        const auto append_metric = [&](const MetricKey& key, const Metric& metric, const bool include_live_sample) {
            if (metric.live_bytes == 0 && metric.peak_bytes == 0 &&
                metric.allocated_bytes == 0 && metric.freed_bytes == 0) {
                return;
            }
            if (metric.current_sample || include_live_sample) {
                out.sampled_live_bytes += metric.live_bytes;
            }
            out.rows.push_back({
                .scope = key.scope,
                .label = key.label,
                .live_bytes = metric.live_bytes,
                .peak_bytes = metric.peak_bytes,
                .allocated_bytes = metric.allocated_bytes,
                .freed_bytes = metric.freed_bytes,
                .allocation_count = metric.allocation_count,
                .free_count = metric.free_count,
                .method = metric.method,
            });
            add_metric_to_tree(key, metric);
        };

        for (const auto& [key, metric] : impl_->metrics) {
            append_metric(key, metric, false);
        }
        for (const auto& [key, metric] : impl_->static_metrics) {
            append_metric(key, metric, true);
        }
        for (const auto& [_, stats] : impl_->scope_nodes) {
            if (stats.path.empty()) {
                continue;
            }
            add_scope_to_tree(stats);
        }

        std::unordered_map<std::string, std::vector<std::string>> children;
        children.reserve(tree_nodes.size());
        for (auto& [path, node] : tree_nodes) {
            const auto slash = path.rfind('/');
            const auto parent = slash == std::string::npos ? std::string{} : path.substr(0, slash);
            children[parent].push_back(path);
        }
        for (auto& [parent, list] : children) {
            for (const auto& child : list) {
                if (!parent.empty()) {
                    tree_nodes[parent].has_children = true;
                }
                (void)child;
            }
            std::sort(list.begin(), list.end(), [&](const auto& lhs_path, const auto& rhs_path) {
                const auto& lhs = tree_nodes[lhs_path];
                const auto& rhs = tree_nodes[rhs_path];
                if (lhs.name != rhs.name) {
                    return lhs.name < rhs.name;
                }
                return lhs.path < rhs.path;
            });
        }

        const auto aggregate_tree = [&](const auto& self,
                                        const std::string& path) -> VramTreeNodeSnapshot {
            auto aggregate = tree_nodes[path];
            const bool inherit_child_delta = aggregate.vram_delta_count == 0;
            const auto child_it = children.find(path);
            if (child_it == children.end()) {
                return aggregate;
            }

            for (const auto& child_path : child_it->second) {
                const auto child = self(self, child_path);
                if (!aggregate.timer_scope) {
                    aggregate.timer_call_count += child.timer_call_count;
                    aggregate.total_ms += child.total_ms;
                    aggregate.last_ms += child.last_ms;
                    aggregate.max_ms = std::max(aggregate.max_ms, child.max_ms);
                }
                if (inherit_child_delta) {
                    aggregate.vram_delta_count += child.vram_delta_count;
                    aggregate.last_vram_delta_bytes += child.last_vram_delta_bytes;
                    aggregate.net_vram_delta_bytes += child.net_vram_delta_bytes;
                    aggregate.max_vram_increase_bytes = std::max(aggregate.max_vram_increase_bytes,
                                                                 child.max_vram_increase_bytes);
                    aggregate.max_vram_decrease_bytes = std::min(aggregate.max_vram_decrease_bytes,
                                                                 child.max_vram_decrease_bytes);
                }
            }

            auto& node = tree_nodes[path];
            if (!node.timer_scope) {
                node.timer_call_count = aggregate.timer_call_count;
                node.total_ms = aggregate.total_ms;
                node.last_ms = aggregate.last_ms;
                node.max_ms = aggregate.max_ms;
            }
            if (node.vram_delta_count == 0) {
                node.vram_delta_count = aggregate.vram_delta_count;
                node.last_vram_delta_bytes = aggregate.last_vram_delta_bytes;
                node.net_vram_delta_bytes = aggregate.net_vram_delta_bytes;
                node.max_vram_increase_bytes = aggregate.max_vram_increase_bytes;
                node.max_vram_decrease_bytes = aggregate.max_vram_decrease_bytes;
            }
            return tree_nodes[path];
        };
        if (const auto root_it = children.find(std::string{}); root_it != children.end()) {
            for (const auto& child_path : root_it->second) {
                (void)aggregate_tree(aggregate_tree, child_path);
            }
        }
        for (auto& [parent, list] : children) {
            std::sort(list.begin(), list.end(), [&](const auto& lhs_path, const auto& rhs_path) {
                const auto& lhs = tree_nodes[lhs_path];
                const auto& rhs = tree_nodes[rhs_path];
                if (lhs.name != rhs.name) {
                    return lhs.name < rhs.name;
                }
                return lhs.path < rhs.path;
            });
        }

        const auto append_tree = [&](const auto& self, const std::string& parent) -> void {
            const auto it = children.find(parent);
            if (it == children.end()) {
                return;
            }
            for (const auto& child_path : it->second) {
                out.tree.push_back(tree_nodes[child_path]);
                self(self, child_path);
            }
        };
        append_tree(append_tree, std::string{});

        std::sort(out.rows.begin(), out.rows.end(), [](const auto& lhs, const auto& rhs) {
            const auto lhs_key = std::max({lhs.live_bytes, lhs.peak_bytes, lhs.allocated_bytes});
            const auto rhs_key = std::max({rhs.live_bytes, rhs.peak_bytes, rhs.allocated_bytes});
            if (lhs_key != rhs_key) {
                return lhs_key > rhs_key;
            }
            if (lhs.scope != rhs.scope) {
                return lhs.scope < rhs.scope;
            }
            return lhs.label < rhs.label;
        });

        out.top_live.reserve(std::min<std::size_t>(kTopNLive, out.rows.size()));
        for (const auto& row : out.rows) {
            if (row.live_bytes == 0)
                continue;
            out.top_live.push_back({row.scope, row.label, row.live_bytes});
            if (out.top_live.size() >= kTopNLive)
                break;
        }

        out.gauges.reserve(impl_->gauges.size());
        for (const auto& [k, v] : impl_->gauges) {
            out.gauges.push_back({k, v, impl_->gauge_generation});
        }
        std::sort(out.gauges.begin(), out.gauges.end(),
                  [](const auto& a, const auto& b) { return a.key < b.key; });

        out.iter_counters.reserve(impl_->iter_counters.size());
        for (const auto& [k, v] : impl_->iter_counters) {
            out.iter_counters.push_back({k, v});
        }
        std::sort(out.iter_counters.begin(), out.iter_counters.end(),
                  [](const auto& a, const auto& b) { return a.key < b.key; });

        out.total_counters.reserve(impl_->total_counters.size());
        for (const auto& [k, v] : impl_->total_counters) {
            out.total_counters.push_back({k, v});
        }
        std::sort(out.total_counters.begin(), out.total_counters.end(),
                  [](const auto& a, const auto& b) { return a.key < b.key; });

        out.histograms.reserve(impl_->histograms.size());
        for (const auto& [k, hist] : impl_->histograms) {
            NamedHistogram h;
            h.key = k;
            h.count = hist.count;
            h.sum = hist.sum;
            h.min_value = hist.min_value;
            h.max_value = hist.max_value;
            h.p50 = ring_percentile(hist.ring, 0.50);
            h.p95 = ring_percentile(hist.ring, 0.95);
            h.p99 = ring_percentile(hist.ring, 0.99);
            out.histograms.push_back(std::move(h));
        }
        std::sort(out.histograms.begin(), out.histograms.end(),
                  [](const auto& a, const auto& b) { return a.key < b.key; });

        return out;
    }

    VramScope::VramScope(std::string_view scope)
        : active_(VramProfiler::instance().enabled()) {
        if (active_) {
            try {
                VramProfiler::instance().pushScope(scope);
            } catch (...) {
                active_ = false;
            }
        }
    }

    VramScope::~VramScope() {
        if (active_) {
            try {
                VramProfiler::instance().popScope();
            } catch (...) {
            }
        }
    }

    VramScope::VramScope(VramScope&& other) noexcept
        : active_(std::exchange(other.active_, false)) {}

    VramScope& VramScope::operator=(VramScope&& other) noexcept {
        if (this != &other) {
            if (active_) {
                try {
                    VramProfiler::instance().popScope();
                } catch (...) {
                }
            }
            active_ = std::exchange(other.active_, false);
        }
        return *this;
    }

    VramDeltaScope::VramDeltaScope(std::string_view scope)
        : active_(VramProfiler::instance().enabled()) {
        if (active_) {
            try {
                start_valid_ = VramProfiler::instance().pushVramDeltaScope(scope, start_used_bytes_, pushed_scope_);
            } catch (...) {
                active_ = false;
                start_valid_ = false;
                pushed_scope_ = false;
            }
        }
    }

    VramDeltaScope::~VramDeltaScope() {
        if (active_) {
            try {
                VramProfiler::instance().popVramDeltaScope(start_used_bytes_, start_valid_, pushed_scope_);
            } catch (...) {
            }
        }
    }

    VramDeltaScope::VramDeltaScope(VramDeltaScope&& other) noexcept
        : active_(std::exchange(other.active_, false)),
          start_valid_(std::exchange(other.start_valid_, false)),
          pushed_scope_(std::exchange(other.pushed_scope_, false)),
          start_used_bytes_(std::exchange(other.start_used_bytes_, 0)) {}

    VramDeltaScope& VramDeltaScope::operator=(VramDeltaScope&& other) noexcept {
        if (this != &other) {
            if (active_) {
                try {
                    VramProfiler::instance().popVramDeltaScope(start_used_bytes_, start_valid_, pushed_scope_);
                } catch (...) {
                }
            }
            active_ = std::exchange(other.active_, false);
            start_valid_ = std::exchange(other.start_valid_, false);
            pushed_scope_ = std::exchange(other.pushed_scope_, false);
            start_used_bytes_ = std::exchange(other.start_used_bytes_, 0);
        }
        return *this;
    }

    TraceScope::TraceScope(std::string_view scope)
        : active_(VramProfiler::instance().enabled()) {
        if (!active_)
            return;
        try {
            VramProfiler::instance().pushTimerScope(scope);
            start_ns_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now().time_since_epoch())
                            .count();
        } catch (...) {
            active_ = false;
        }
    }

    TraceScope::~TraceScope() {
        if (!active_)
            return;
        try {
            const auto end_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now().time_since_epoch())
                                    .count();
            const double elapsed_ms = static_cast<double>(end_ns - start_ns_) / 1.0e6;
            VramProfiler::instance().popTimerScope(elapsed_ms);
        } catch (...) {
        }
    }

    GpuTimeScope::GpuTimeScope(std::string_view scope, void* stream)
        : stream_(stream) {
        if (!VramProfiler::instance().enabled())
            return;
        try {
            event_pair_ = VramProfiler::instance().acquireGpuEventPair(scope, stream);
            active_ = event_pair_ >= 0;
        } catch (...) {
            active_ = false;
            event_pair_ = -1;
        }
    }

    GpuTimeScope::~GpuTimeScope() {
        if (!active_)
            return;
        try {
            VramProfiler::instance().releaseGpuEventPair(event_pair_, stream_);
        } catch (...) {
        }
    }

} // namespace lfs::diagnostics
