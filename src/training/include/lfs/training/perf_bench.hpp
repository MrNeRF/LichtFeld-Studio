/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

/**
 * @file perf_bench.hpp
 * @brief Phase 0.3 training-loop measurement harness.
 *
 * Activated when the environment variable LFS_PERF_BENCH is set to a non-empty
 * non-"0" value. Collects per-iteration wall time, real device allocs
 * (alloc_counter), peak CUDA VRAM, last loss, and the training-state ledger.
 * Writes a JSON report at finalize().
 */

#include "diagnostics/vram_profiler.hpp"

#include <cstdint>
#include <filesystem>
#include <string>

namespace lfs::training {

    class PerfBenchCollector {
    public:
        static PerfBenchCollector& instance();

        /// True when LFS_PERF_BENCH is enabled for this process.
        [[nodiscard]] static bool enabled();

        /// Warmup length for steady-state metrics (default 200; overridable via
        /// LFS_PERF_BENCH_WARMUP).
        [[nodiscard]] static int warmup_iters();

        void on_training_start(int total_iters);
        void on_step_begin(int iter);
        void on_step_end(int iter, float loss, std::size_t live_splats);
        void set_ledger(const diagnostics::TrainingStateLedger& ledger);
        void set_psnr(double psnr);

        /// Write JSON report to @p path (parent dirs created as needed).
        void finalize(const std::filesystem::path& path);

        [[nodiscard]] bool started() const noexcept { return started_; }

    private:
        PerfBenchCollector() = default;

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
        std::size_t peak_cuda_used_ = 0;
        std::size_t peak_cuda_total_ = 0;
        float last_loss_ = 0.0f;
        std::size_t last_live_splats_ = 0;
        double last_psnr_ = -1.0;
        diagnostics::TrainingStateLedger ledger_{};
        std::int64_t train_start_ns_ = 0;
        std::int64_t train_end_ns_ = 0;
    };

} // namespace lfs::training
