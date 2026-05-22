/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "diagnostics/export.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace lfs::diagnostics {

    enum class VramAllocationMethod : std::uint8_t {
        Unknown,
        Slab,
        Bucketed,
        Async,
        Direct,
        Arena,
        External,
    };

    struct VramMetricSnapshot {
        std::string scope;
        std::string label;
        std::size_t live_bytes = 0;
        std::size_t peak_bytes = 0;
        std::size_t allocated_bytes = 0;
        std::size_t freed_bytes = 0;
        std::uint64_t allocation_count = 0;
        std::uint64_t free_count = 0;
    };

    struct VramTreeNodeSnapshot {
        std::string path;
        std::string name;
        std::uint32_t depth = 0;
        bool has_children = false;
        bool has_metrics = false;
        bool timer_scope = false;
        bool vram_delta_scope = false;
        bool logical_scope = false;
        std::size_t live_bytes = 0;
        std::size_t peak_bytes = 0;
        std::size_t allocated_bytes = 0;
        std::size_t freed_bytes = 0;
        std::uint64_t allocation_count = 0;
        std::uint64_t free_count = 0;
        std::uint64_t timer_call_count = 0;
        double total_ms = 0.0;
        double last_ms = 0.0;
        double max_ms = 0.0;
        std::uint64_t vram_delta_count = 0;
        std::int64_t last_vram_delta_bytes = 0;
        std::int64_t net_vram_delta_bytes = 0;
        std::int64_t max_vram_increase_bytes = 0;
        std::int64_t max_vram_decrease_bytes = 0;
    };

    struct VramProcessSnapshot {
        std::size_t cuda_used = 0;
        std::size_t cuda_total = 0;
        std::size_t cuda_pool_used = 0;
        std::size_t cuda_pool_reserved = 0;
        std::size_t process_used = 0;
        std::size_t total_used = 0;
        std::size_t total = 0;
        std::string device_name;
        bool cuda_memory_valid = false;
        bool cuda_pool_valid = false;
        bool process_memory_valid = false;
    };

    struct VramProfilerSnapshot {
        bool enabled = false;
        int iteration = 0;
        std::uint64_t sequence = 0;
        std::uint64_t allocation_events = 0;
        std::uint64_t free_events = 0;
        std::size_t accounted_live_bytes = 0;
        std::size_t accounted_peak_bytes = 0;
        std::size_t sampled_live_bytes = 0;
        VramProcessSnapshot process;
        std::vector<VramMetricSnapshot> rows;
        std::vector<VramTreeNodeSnapshot> tree;
    };

    class LFS_DIAGNOSTICS_API VramScope {
    public:
        explicit VramScope(std::string_view scope);
        ~VramScope();

        VramScope(const VramScope&) = delete;
        VramScope& operator=(const VramScope&) = delete;

        VramScope(VramScope&& other) noexcept;
        VramScope& operator=(VramScope&& other) noexcept;

    private:
        bool active_ = false;
    };

    class LFS_DIAGNOSTICS_API VramDeltaScope {
    public:
        explicit VramDeltaScope(std::string_view scope);
        ~VramDeltaScope();

        VramDeltaScope(const VramDeltaScope&) = delete;
        VramDeltaScope& operator=(const VramDeltaScope&) = delete;

        VramDeltaScope(VramDeltaScope&& other) noexcept;
        VramDeltaScope& operator=(VramDeltaScope&& other) noexcept;

    private:
        bool active_ = false;
        bool start_valid_ = false;
        bool pushed_scope_ = false;
        std::size_t start_used_bytes_ = 0;
    };

    class LFS_DIAGNOSTICS_API VramProfiler {
    public:
        static VramProfiler& instance();

        void setEnabled(bool enabled);
        [[nodiscard]] bool enabled() const;

        void beginIteration(int iteration);
        void setIteration(int iteration);

        void pushScope(std::string_view scope);
        void popScope();
        void pushTimerScope(std::string_view scope);
        void popTimerScope(double elapsed_ms);
        bool pushVramDeltaScope(std::string_view scope,
                                std::size_t& start_used_bytes,
                                bool& pushed_scope);
        void popVramDeltaScope(std::size_t start_used_bytes, bool start_valid, bool pushed_scope);

        void recordAllocation(void* ptr,
                              std::size_t bytes,
                              VramAllocationMethod method,
                              std::string_view label = {});
        void recordDeallocation(void* ptr);
        void recordBytes(std::string_view scope,
                         std::string_view label,
                         std::size_t bytes,
                         VramAllocationMethod method = VramAllocationMethod::External);
        void recordCurrentBytes(std::string_view scope,
                                std::string_view label,
                                std::size_t bytes,
                                VramAllocationMethod method = VramAllocationMethod::External);
        void recordStaticBytes(std::string_view scope,
                               std::string_view label,
                               std::size_t bytes,
                               VramAllocationMethod method = VramAllocationMethod::External);
        void clearStaticScope(std::string_view scope);
        void recordTimerSample(std::string_view scope, double elapsed_ms);
        void clearScope(std::string_view scope);

        void sampleCudaMemory();
        void updateProcessMemory(std::size_t process_used,
                                 std::size_t total_used,
                                 std::size_t total,
                                 std::string device_name);

        [[nodiscard]] VramProfilerSnapshot snapshot() const;

    private:
        VramProfiler();
        ~VramProfiler();
        VramProfiler(const VramProfiler&) = delete;
        VramProfiler& operator=(const VramProfiler&) = delete;

        struct Impl;
        Impl* impl_;
    };

} // namespace lfs::diagnostics

#define LFS_DIAGNOSTICS_CONCAT_INNER(a, b) a##b
#define LFS_DIAGNOSTICS_CONCAT(a, b) LFS_DIAGNOSTICS_CONCAT_INNER(a, b)
#define LFS_VRAM_SCOPE(name) \
    ::lfs::diagnostics::VramScope LFS_DIAGNOSTICS_CONCAT(_lfs_vram_scope_, __LINE__)(name)
#define LOG_VRAM_DIFF(name) \
    ::lfs::diagnostics::VramDeltaScope LFS_DIAGNOSTICS_CONCAT(_lfs_vram_delta_scope_, __LINE__)(name)
