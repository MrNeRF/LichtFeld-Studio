/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/perf_bench.hpp"

#include "core/alloc_counter.hpp"
#include "core/environment.hpp"
#include "core/logger.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace lfs::training {
    namespace {

        [[nodiscard]] std::int64_t now_ns() {
            return std::chrono::duration_cast<std::chrono::nanoseconds>(
                       std::chrono::steady_clock::now().time_since_epoch())
                .count();
        }

        [[nodiscard]] bool env_truthy(const char* name) {
            const char* v = std::getenv(name);
            if (v == nullptr || v[0] == '\0' || v[0] == '0') {
                return false;
            }
            // Treat "false" / "off" / "no" as disabled.
            if ((v[0] == 'f' || v[0] == 'F' || v[0] == 'n' || v[0] == 'N') ||
                (v[0] == 'o' && (v[1] == 'f' || v[1] == 'F'))) {
                return false;
            }
            return true;
        }

        void sample_cuda_used(std::size_t& used, std::size_t& total) {
            std::size_t free_b = 0;
            std::size_t total_b = 0;
            if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess && total_b >= free_b) {
                used = total_b - free_b;
                total = total_b;
            }
        }

    } // namespace

    PerfBenchCollector& PerfBenchCollector::instance() {
        static PerfBenchCollector collector;
        return collector;
    }

    bool PerfBenchCollector::enabled() {
        return env_truthy("LFS_PERF_BENCH");
    }

    int PerfBenchCollector::warmup_iters() {
        if (const auto v = lfs::core::environment::unsigned_integer<unsigned long long>(
                "LFS_PERF_BENCH_WARMUP");
            v && *v > 0) {
            return static_cast<int>(*v);
        }
        return 200;
    }

    void PerfBenchCollector::on_training_start(const int total_iters) {
        if (!enabled()) {
            return;
        }
        started_ = true;
        total_iters_ = total_iters;
        warmup_ = warmup_iters();
        warmup_allocs_ = 0;
        steady_allocs_ = 0;
        warmup_steps_ = 0;
        steady_steps_ = 0;
        warmup_ms_sum_ = 0.0;
        steady_ms_sum_ = 0.0;
        peak_cuda_used_ = 0;
        peak_cuda_total_ = 0;
        last_loss_ = 0.0f;
        last_live_splats_ = 0;
        last_psnr_ = -1.0;
        ledger_ = {};
        train_start_ns_ = now_ns();
        train_end_ns_ = train_start_ns_;

        // Ensure the VRAM profiler is on so the ledger is published each step.
        lfs::diagnostics::VramProfiler::instance().setEnabled(true);
        LOG_INFO("PerfBench: enabled (warmup={} iters, total={})", warmup_, total_iters_);
    }

    void PerfBenchCollector::on_step_begin(const int /*iter*/) {
        if (!started_) {
            return;
        }
        step_alloc_snap_ = lfs::core::alloc_counter::snapshot();
        step_start_ns_ = now_ns();
    }

    void PerfBenchCollector::on_step_end(const int iter,
                                         const float loss,
                                         const std::size_t live_splats) {
        if (!started_) {
            return;
        }
        const auto step_end = now_ns();
        const double ms =
            static_cast<double>(step_end - step_start_ns_) / 1.0e6;
        const auto allocs = lfs::core::alloc_counter::delta_since(step_alloc_snap_);

        std::size_t used = 0;
        std::size_t total = 0;
        sample_cuda_used(used, total);
        if (used > peak_cuda_used_) {
            peak_cuda_used_ = used;
            peak_cuda_total_ = total;
        }

        last_loss_ = loss;
        last_live_splats_ = live_splats;
        train_end_ns_ = step_end;

        // iter is 1-based in the trainer.
        if (iter <= warmup_) {
            warmup_allocs_ += allocs;
            warmup_ms_sum_ += ms;
            ++warmup_steps_;
        } else {
            steady_allocs_ += allocs;
            steady_ms_sum_ += ms;
            ++steady_steps_;
        }
    }

    void PerfBenchCollector::set_ledger(const diagnostics::TrainingStateLedger& ledger) {
        if (!started_) {
            return;
        }
        ledger_ = ledger;
    }

    void PerfBenchCollector::set_psnr(const double psnr) {
        if (!started_) {
            return;
        }
        last_psnr_ = psnr;
    }

    void PerfBenchCollector::finalize(const std::filesystem::path& path) {
        if (!started_) {
            return;
        }
        train_end_ns_ = now_ns();

        // Prefer the last published profiler ledger if ours is empty.
        if (ledger_.total_bytes == 0) {
            ledger_ = diagnostics::VramProfiler::instance().trainingStateLedger();
        }

        const double wall_s =
            static_cast<double>(train_end_ns_ - train_start_ns_) / 1.0e9;
        const double warmup_ms_iter =
            warmup_steps_ > 0 ? warmup_ms_sum_ / static_cast<double>(warmup_steps_) : 0.0;
        const double steady_ms_iter =
            steady_steps_ > 0 ? steady_ms_sum_ / static_cast<double>(steady_steps_) : 0.0;
        const double warmup_allocs_iter =
            warmup_steps_ > 0
                ? static_cast<double>(warmup_allocs_) / static_cast<double>(warmup_steps_)
                : 0.0;
        const double steady_allocs_iter =
            steady_steps_ > 0
                ? static_cast<double>(steady_allocs_) / static_cast<double>(steady_steps_)
                : 0.0;

        std::error_code ec;
        std::filesystem::create_directories(path.parent_path(), ec);

        std::ofstream out(path);
        if (!out) {
            LOG_ERROR("PerfBench: failed to write {}", path.string());
            return;
        }

        out << std::setprecision(6) << std::fixed;
        out << "{\n";
        out << "  \"total_iters\": " << total_iters_ << ",\n";
        out << "  \"warmup_iters\": " << warmup_ << ",\n";
        out << "  \"warmup_steps\": " << warmup_steps_ << ",\n";
        out << "  \"steady_steps\": " << steady_steps_ << ",\n";
        out << "  \"wall_seconds\": " << wall_s << ",\n";
        out << "  \"warmup_ms_per_iter\": " << warmup_ms_iter << ",\n";
        out << "  \"steady_ms_per_iter\": " << steady_ms_iter << ",\n";
        out << "  \"warmup_allocs_total\": " << warmup_allocs_ << ",\n";
        out << "  \"steady_allocs_total\": " << steady_allocs_ << ",\n";
        out << "  \"warmup_allocs_per_iter\": " << warmup_allocs_iter << ",\n";
        out << "  \"steady_allocs_per_iter\": " << steady_allocs_iter << ",\n";
        out << "  \"peak_cuda_used_bytes\": " << peak_cuda_used_ << ",\n";
        out << "  \"peak_cuda_total_bytes\": " << peak_cuda_total_ << ",\n";
        out << "  \"last_loss\": " << last_loss_ << ",\n";
        out << "  \"last_psnr\": " << last_psnr_ << ",\n";
        out << "  \"last_live_splats\": " << last_live_splats_ << ",\n";
        out << "  \"alloc_counter_total\": " << lfs::core::alloc_counter::total() << ",\n";
        out << "  \"ledger\": {\n";
        out << "    \"params_bytes\": " << ledger_.params_bytes << ",\n";
        out << "    \"optimizer_bytes\": " << ledger_.optimizer_bytes << ",\n";
        out << "    \"gradients_or_helpers_bytes\": " << ledger_.gradients_or_helpers_bytes << ",\n";
        out << "    \"densify_aux_bytes\": " << ledger_.densify_aux_bytes << ",\n";
        out << "    \"total_bytes\": " << ledger_.total_bytes << ",\n";
        out << "    \"live_splats\": " << ledger_.live_splats << ",\n";
        out << "    \"bytes_per_splat\": " << ledger_.bytes_per_splat << "\n";
        out << "  }\n";
        out << "}\n";
        out.close();

        LOG_INFO("PerfBench: wrote {} (steady {:.2f} ms/iter, {:.1f} allocs/iter, peak VRAM {:.1f} MiB, {:.1f} B/splat)",
                 path.string(),
                 steady_ms_iter,
                 steady_allocs_iter,
                 static_cast<double>(peak_cuda_used_) / (1024.0 * 1024.0),
                 ledger_.bytes_per_splat);
    }

} // namespace lfs::training
