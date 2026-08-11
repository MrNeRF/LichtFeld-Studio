/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

/**
 * @file perf_bench.hpp
 * @brief training-loop measurement harness.
 *
 * Activated via CLI `--perf-bench` (and optional `--perf-bench-warmup=N`).
 * Collects per-iteration wall time, real device allocs (alloc_counter), peak
 * CUDA VRAM, last loss, and the training-state ledger. Writes a JSON report
 * at finalize().
 */

#include "diagnostics/vram_profiler.hpp"

#include <cstdint>
#include <filesystem>
#include <string>

namespace lfs::training {

    namespace detail {

        struct PerfPeakCoverSample {
            std::size_t pool_bucket_cache_bytes = 0;
            std::size_t pool_bucket_live_waste_bytes = 0;
            std::size_t exportable_splat_bytes = 0;
            std::size_t io_external_bytes = 0;
        };

        /// Values that may justify a process peak must come from that same
        /// snapshot. Historical per-row peaks are disclosure only.
        [[nodiscard]] PerfPeakCoverSample collect_perf_peak_cover_sample(
            const diagnostics::VramProfilerSnapshot& snapshot);

    } // namespace detail

    class PerfBenchCollector {
    public:
        static PerfBenchCollector& instance();

        /// Configure from CLI (`--perf-bench` / `--perf-bench-warmup=N`). Call before training.
        static void configure(bool enable, int warmup_iters = 200);

        [[nodiscard]] static bool enabled();

        /// Warmup length for steady-state metrics (default 200).
        [[nodiscard]] static int warmup_iters();

        void on_training_start(int total_iters);
        void on_step_begin(int iter);
        void on_step_end(int iter, float loss, std::size_t live_splats);
        void set_ledger(const diagnostics::TrainingStateLedger& ledger);
        void set_psnr(double psnr);

        /// Accumulate wall time spent blocked in dataloader->next() (outside
        /// the train_step span that feeds steady_ms). @p iter is 1-based.
        void record_dataloader_wait(int iter, double wait_ms);

        // Loss-workspace active requirement and backing allocation for peak attribution.
        void set_loss_workspace_bytes(std::size_t required_bytes,
                                      std::size_t allocated_bytes);

        // Densify child / N-scratch high-water for peak attribution.
        void set_densify_workspace_bytes(std::size_t bytes);

        void set_mrnf_strategy_bytes(std::size_t required_bytes,
                                     std::size_t allocated_bytes);
        void set_mrnf_densify_n_bytes(std::size_t required_bytes,
                                      std::size_t allocated_bytes);
        void set_mrnf_densify_child_bytes(std::size_t required_bytes,
                                          std::size_t allocated_bytes);

        /// Capacity-backed training-state high-water (params+optim reserved, not logical N).
        void set_training_state_reserved_bytes(std::size_t bytes);

        /// FastGS raster live buffers split into disjoint arena and sort owners.
        void set_fastgs_raster_live_bytes(std::size_t arena_bytes,
                                          std::size_t sort_bytes);

        /// Write JSON report to @p path (parent dirs created as needed).
        void finalize(const std::filesystem::path& path);

        // Build the peak ex-cache ledger (owners + justified residuals).
        [[nodiscard]] diagnostics::PeakExCacheLedger peak_ex_cache_ledger() const;

    private:
        PerfBenchCollector() = default;

        void capture_peak_snapshot(int iter, std::size_t used, std::size_t total);

        bool started_ = false;
        int total_iters_ = 0;
        int warmup_ = 200;

        // Per-step bookkeeping
        std::uint64_t step_alloc_snap_ = 0;
        std::int64_t step_start_ns_ = 0;

        // Aggregates
        std::uint64_t warmup_allocs_ = 0;
        std::uint64_t steady_allocs_ = 0;
        std::uint64_t warmup_steps_ = 0;
        std::uint64_t steady_steps_ = 0;
        double warmup_ms_sum_ = 0.0;
        double steady_ms_sum_ = 0.0;
        // Dataloader wait is outside train_step timing (steady_ms is blind to it).
        double dataloader_wait_ms_sum_ = 0.0;
        double steady_dataloader_wait_ms_sum_ = 0.0;
        std::uint64_t dataloader_wait_count_ = 0;
        std::uint64_t steady_dataloader_wait_count_ = 0;
        std::size_t peak_cuda_used_ = 0;
        std::size_t peak_cuda_total_ = 0;
        /// Device-wide used sampled at on_training_start (desktop + CUDA context
        // already resident). Subtracted for process-net ex_cache so
        /// quiet-GPU comparisons are not polluted by concurrent GPU users.
        std::size_t baseline_cuda_used_ = 0;
        std::size_t peak_pool_reserved_ = 0;
        std::size_t peak_pool_used_ = 0;
        std::size_t pool_reserved_at_peak_ = 0;
        std::size_t pool_used_at_peak_ = 0;
        std::size_t peak_pool_bucket_cache_ = 0;
        std::size_t peak_pool_bucket_live_waste_ = 0;
        std::size_t peak_exportable_splat_ = 0;
        std::size_t peak_arena_required_ = 0;
        std::size_t peak_arena_capacity_ = 0;
        std::size_t peak_fastgs_sort_required_ = 0;
        std::size_t peak_fastgs_sort_allocated_ = 0;
        std::size_t peak_fastgs_raster_live_ = 0;
        std::size_t peak_fastgs_raster_arena_live_ = 0;
        std::size_t peak_fastgs_raster_sort_live_ = 0;
        std::size_t peak_io_ring_bytes_ = 0;
        std::size_t peak_io_external_bytes_ = 0;
        std::size_t peak_steady_pinned_host_bytes_ = 0;
        bool peak_cover_captured_ = false;
        int peak_iter_ = 0;
        std::vector<diagnostics::PeakSubsystemLine> peak_rows_;
        std::size_t loss_workspace_required_bytes_ = 0;
        std::size_t loss_workspace_allocated_bytes_ = 0;
        std::size_t densify_workspace_bytes_ = 0;
        std::size_t mrnf_strategy_required_bytes_ = 0;
        std::size_t mrnf_strategy_allocated_bytes_ = 0;
        std::size_t mrnf_densify_n_required_bytes_ = 0;
        std::size_t mrnf_densify_n_allocated_bytes_ = 0;
        std::size_t mrnf_densify_child_required_bytes_ = 0;
        std::size_t mrnf_densify_child_allocated_bytes_ = 0;
        std::size_t training_state_reserved_bytes_ = 0;
        std::size_t fastgs_raster_live_bytes_ = 0;
        std::size_t fastgs_raster_arena_live_bytes_ = 0;
        std::size_t fastgs_raster_sort_live_bytes_ = 0;
        float last_loss_ = 0.0f;
        std::size_t last_live_splats_ = 0;
        double last_psnr_ = -1.0;
        diagnostics::TrainingStateLedger ledger_{};
        std::int64_t train_start_ns_ = 0;
        std::int64_t train_end_ns_ = 0;
    };

} // namespace lfs::training
