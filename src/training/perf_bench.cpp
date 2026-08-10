/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/perf_bench.hpp"

#include "core/alloc_counter.hpp"
#include "core/logger.hpp"
#include "core/pinned_memory_allocator.hpp"
#include "training/rasterization/fastgs/rasterization/include/forward.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>

namespace lfs::training {
    namespace {

        std::atomic<bool> g_perf_bench_enabled{false};
        std::atomic<int> g_perf_bench_warmup{200};

        [[nodiscard]] std::int64_t now_ns() {
            return std::chrono::duration_cast<std::chrono::nanoseconds>(
                       std::chrono::steady_clock::now().time_since_epoch())
                .count();
        }

        void sample_cuda_used(std::size_t& used, std::size_t& total) {
            std::size_t free_b = 0;
            std::size_t total_b = 0;
            if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess && total_b >= free_b) {
                used = total_b - free_b;
                total = total_b;
            }
        }

        void sample_pool_hwm(std::size_t& used_high, std::size_t& reserved_high,
                             std::size_t& used_cur, std::size_t& reserved_cur) {
            used_high = reserved_high = used_cur = reserved_cur = 0;
#if CUDART_VERSION >= 12080
            int device = 0;
            if (cudaGetDevice(&device) != cudaSuccess) {
                return;
            }
            cudaMemPool_t pool = nullptr;
            if (cudaDeviceGetDefaultMemPool(&pool, device) != cudaSuccess || !pool) {
                return;
            }
            std::uint64_t u = 0;
            std::uint64_t r = 0;
            if (cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &u) == cudaSuccess) {
                used_high = static_cast<std::size_t>(u);
            }
            if (cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemHigh, &r) ==
                cudaSuccess) {
                reserved_high = static_cast<std::size_t>(r);
            }
            u = r = 0;
            if (cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &u) ==
                cudaSuccess) {
                used_cur = static_cast<std::size_t>(u);
            }
            if (cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &r) ==
                cudaSuccess) {
                reserved_cur = static_cast<std::size_t>(r);
            }
#endif
        }

    } // namespace

    PerfBenchCollector& PerfBenchCollector::instance() {
        static PerfBenchCollector collector;
        return collector;
    }

    void PerfBenchCollector::configure(const bool enable, const int warmup) {
        g_perf_bench_enabled.store(enable, std::memory_order_relaxed);
        if (warmup > 0) {
            g_perf_bench_warmup.store(warmup, std::memory_order_relaxed);
        }
        // Prefer main.cpp's early device baseline (post primary-context, pre-model).
        // Fall back to a configure-time sample when the profiler baseline is unset
        // (unit tests / non-main entry points).
        if (enable) {
            auto& c = instance();
            const auto early =
                diagnostics::VramProfiler::instance().cudaDeviceBaselineBytes();
            if (early > 0) {
                c.baseline_cuda_used_ = early;
            } else {
                std::size_t used = 0;
                std::size_t total = 0;
                sample_cuda_used(used, total);
                c.baseline_cuda_used_ = used;
                (void)total;
            }
        }
    }

    bool PerfBenchCollector::enabled() {
        return g_perf_bench_enabled.load(std::memory_order_relaxed);
    }

    int PerfBenchCollector::warmup_iters() {
        return g_perf_bench_warmup.load(std::memory_order_relaxed);
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
        // configure was skipped, fall back to the value already stored (0).
        peak_pool_reserved_ = 0;
        peak_pool_used_ = 0;
        peak_pool_bucket_cache_ = 0;
        peak_pool_bucket_live_waste_ = 0;
        peak_exportable_splat_ = 0;
        peak_arena_required_ = 0;
        peak_arena_capacity_ = 0;
        peak_fastgs_sort_hwm_ = 0;
        peak_fastgs_raster_live_ = 0;
        peak_io_ring_bytes_ = 0;
        peak_io_external_bytes_ = 0;
        peak_steady_pinned_host_bytes_ = 0;
        peak_iter_ = 0;
        peak_rows_.clear();
        loss_workspace_required_bytes_ = 0;
        loss_workspace_allocated_bytes_ = 0;
        densify_workspace_bytes_ = 0;
        training_state_reserved_bytes_ = 0;
        fastgs_raster_live_bytes_ = 0;
        dataloader_wait_ms_sum_ = 0.0;
        steady_dataloader_wait_ms_sum_ = 0.0;
        dataloader_wait_count_ = 0;
        steady_dataloader_wait_count_ = 0;
        last_loss_ = 0.0f;
        last_live_splats_ = 0;
        last_psnr_ = -1.0;
        ledger_ = {};
        train_start_ns_ = now_ns();
        train_end_ns_ = train_start_ns_;
        lfs::core::alloc_counter::reset_site_counts();

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

    void PerfBenchCollector::capture_peak_snapshot(const int iter,
                                                   const std::size_t used,
                                                   const std::size_t total) {
        peak_cuda_used_ = used;
        peak_cuda_total_ = total;
        peak_iter_ = iter;

        std::size_t used_high = 0;
        std::size_t reserved_high = 0;
        std::size_t used_cur = 0;
        std::size_t reserved_cur = 0;
        sample_pool_hwm(used_high, reserved_high, used_cur, reserved_cur);
        peak_pool_used_ = std::max(peak_pool_used_, std::max(used_high, used_cur));
        peak_pool_reserved_ =
            std::max(peak_pool_reserved_, std::max(reserved_high, reserved_cur));

        const auto sort_hwm = fast_lfs::rasterization::sort_workspace_high_water_bytes();
        peak_fastgs_sort_hwm_ = std::max(peak_fastgs_sort_hwm_, sort_hwm);
        peak_fastgs_raster_live_ =
            std::max(peak_fastgs_raster_live_, fastgs_raster_live_bytes_);

        // Refresh process snapshot so pool_bucket_cache / exportable are current.
        auto& profiler = diagnostics::VramProfiler::instance();
        profiler.sampleCudaMemory();
        const auto snap = profiler.snapshot();
        peak_pool_bucket_cache_ =
            std::max(peak_pool_bucket_cache_, snap.process.cuda_pool_bucket_cache_bytes);
        peak_pool_bucket_live_waste_ = std::max(
            peak_pool_bucket_live_waste_,
            snap.process.cuda_pool_bucket_live_waste_bytes);
        peak_exportable_splat_ =
            std::max(peak_exportable_splat_, snap.process.exportable_splat_bytes);
        for (const auto& gauge : snap.gauges) {
            if (gauge.key == "vram.audit.io.decoded_frame_ring.bytes") {
                peak_io_ring_bytes_ = std::max(peak_io_ring_bytes_, static_cast<std::size_t>(gauge.value));
            } else if (gauge.key == "vram.audit.rasterizer_arena.required_bytes") {
                peak_arena_required_ =
                    std::max(peak_arena_required_, static_cast<std::size_t>(gauge.value));
            } else if (gauge.key == "vram.audit.rasterizer_arena.allocated_bytes") {
                peak_arena_capacity_ =
                    std::max(peak_arena_capacity_, static_cast<std::size_t>(gauge.value));
            }
        }

        std::size_t arena_cap = 0;
        std::size_t raster_live = 0;
        std::size_t io_external = 0;
        peak_rows_.clear();
        for (const auto& row : snap.rows) {
            if (row.live_bytes == 0 && row.peak_bytes == 0) {
                continue;
            }
            if (row.label == "arena.capacity" ||
                row.label.find("arena.capacity") != std::string::npos) {
                arena_cap = std::max(arena_cap, std::max(row.live_bytes, row.peak_bytes));
            }
            if (row.label.find("per_primitive_buffers") != std::string::npos ||
                row.label.find("per_tile_buffers") != std::string::npos ||
                row.label.find("sorted_indices") != std::string::npos) {
                raster_live += std::max(row.live_bytes, row.peak_bytes);
            }
            if (row.scope == "io.nvimagecodec" || row.scope == "io.image_loader" ||
                row.label.find("nvimagecodec") != std::string::npos ||
                row.label.find("image_loader") != std::string::npos) {
                io_external += std::max(row.live_bytes, row.peak_bytes);
            }
            diagnostics::PeakSubsystemLine line;
            line.name = row.scope.empty() ? row.label : (row.scope + "." + row.label);
            line.owner = "vram_profiler";
            line.bytes = std::max(row.live_bytes, row.peak_bytes);
            line.justified = true;
            peak_rows_.push_back(std::move(line));
        }
        peak_arena_capacity_ = std::max(peak_arena_capacity_, arena_cap);
        peak_io_external_bytes_ = std::max(peak_io_external_bytes_, io_external);
        if (raster_live > 0) {
            peak_fastgs_raster_live_ = std::max(peak_fastgs_raster_live_, raster_live);
        }

        // Keep the largest rows for the JSON ledger (top 12 by bytes).
        std::sort(peak_rows_.begin(), peak_rows_.end(),
                  [](const auto& a, const auto& b) { return a.bytes > b.bytes; });
        if (peak_rows_.size() > 12) {
            peak_rows_.resize(12);
        }
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
            capture_peak_snapshot(iter, used, total);
        } else {
            // Still track pool / sort high-water even when device-wide free dips.
            std::size_t used_high = 0;
            std::size_t reserved_high = 0;
            std::size_t used_cur = 0;
            std::size_t reserved_cur = 0;
            sample_pool_hwm(used_high, reserved_high, used_cur, reserved_cur);
            peak_pool_used_ = std::max(peak_pool_used_, std::max(used_high, used_cur));
            peak_pool_reserved_ =
                std::max(peak_pool_reserved_, std::max(reserved_high, reserved_cur));
            peak_fastgs_sort_hwm_ = std::max(
                peak_fastgs_sort_hwm_,
                fast_lfs::rasterization::sort_workspace_high_water_bytes());
            peak_fastgs_raster_live_ =
                std::max(peak_fastgs_raster_live_, fastgs_raster_live_bytes_);
        }

        last_loss_ = loss;
        last_live_splats_ = live_splats;
        train_end_ns_ = step_end;

        // iter is 1-based in the trainer.
        if (iter <= warmup_) {
            warmup_allocs_ += allocs;
            warmup_ms_sum_ += ms;
            ++warmup_steps_;
            if (iter == warmup_) {
                // Next step is the first steady-state step — enable alloc trace.
            }
        } else {
            const auto pinned_stats = lfs::core::PinnedMemoryAllocator::instance().get_stats();
            peak_steady_pinned_host_bytes_ = std::max(
                peak_steady_pinned_host_bytes_,
                pinned_stats.allocated_bytes + pinned_stats.cached_bytes);
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

    void PerfBenchCollector::record_dataloader_wait(const int iter, const double wait_ms) {
        if (!started_) {
            return;
        }
        dataloader_wait_ms_sum_ += wait_ms;
        ++dataloader_wait_count_;
        // iter is 1-based in the trainer; mirror on_step_end warmup split.
        if (iter > warmup_) {
            steady_dataloader_wait_ms_sum_ += wait_ms;
            ++steady_dataloader_wait_count_;
        }
    }

    void PerfBenchCollector::set_loss_workspace_bytes(const std::size_t required_bytes,
                                                      const std::size_t allocated_bytes) {
        if (!started_) {
            return;
        }
        loss_workspace_required_bytes_ =
            std::max(loss_workspace_required_bytes_, required_bytes);
        loss_workspace_allocated_bytes_ =
            std::max(loss_workspace_allocated_bytes_, allocated_bytes);
    }

    void PerfBenchCollector::set_densify_workspace_bytes(const std::size_t bytes) {
        if (!started_) {
            return;
        }
        densify_workspace_bytes_ = std::max(densify_workspace_bytes_, bytes);
    }

    void PerfBenchCollector::set_training_state_reserved_bytes(const std::size_t bytes) {
        if (!started_) {
            return;
        }
        training_state_reserved_bytes_ = std::max(training_state_reserved_bytes_, bytes);
    }

    void PerfBenchCollector::set_fastgs_raster_live_bytes(const std::size_t bytes) {
        if (!started_) {
            return;
        }
        fastgs_raster_live_bytes_ = bytes;
        peak_fastgs_raster_live_ = std::max(peak_fastgs_raster_live_, bytes);
    }

    diagnostics::PeakExCacheLedger PerfBenchCollector::peak_ex_cache_ledger() const {
        diagnostics::PeakExCacheLedger out;
        out.peak_cuda_used_bytes = peak_cuda_used_;
        out.baseline_cuda_used_bytes = baseline_cuda_used_;
        out.peak_pool_reserved_bytes = peak_pool_reserved_;
        out.peak_pool_used_bytes = peak_pool_used_;
        out.training_state_bytes = ledger_.total_bytes;
        out.training_state_reserved_bytes = training_state_reserved_bytes_;
        out.loss_workspace_required_bytes = loss_workspace_required_bytes_;
        out.loss_workspace_allocated_bytes = loss_workspace_allocated_bytes_;
        out.densify_workspace_bytes = densify_workspace_bytes_;
        out.pool_bucket_live_rounding_waste_bytes = peak_pool_bucket_live_waste_;
        out.fastgs_sort_hwm_bytes = peak_fastgs_sort_hwm_;
        out.fastgs_raster_live_bytes = peak_fastgs_raster_live_;
        out.arena_required_bytes = peak_arena_required_;
        out.arena_capacity_bytes = peak_arena_capacity_;
        out.peak_iter = peak_iter_;
        out.peak_rows = peak_rows_;

        const auto snap = diagnostics::VramProfiler::instance().snapshot();
        // Prefer peak-moment bucket cache; fall back to finalize snapshot.
        out.pool_bucket_cache_bytes =
            peak_pool_bucket_cache_ > 0 ? peak_pool_bucket_cache_
                                        : snap.process.cuda_pool_bucket_cache_bytes;
        out.exportable_splat_bytes =
            peak_exportable_splat_ > 0 ? peak_exportable_splat_
                                       : snap.process.exportable_splat_bytes;

        out.ex_cache_bytes = peak_cuda_used_;
        // Process-net: also subtract pre-train baseline (desktop + cold CUDA context).
        // On a busy desktop baseline can be 200–300 MiB and must not count as "ours".
        const std::size_t peak_above_baseline =
            peak_cuda_used_ > baseline_cuda_used_ ? peak_cuda_used_ - baseline_cuda_used_
                                                  : 0;
        out.ex_cache_net_bytes = peak_above_baseline;
        // Compare process-net usage with the quiet-GPU baseline.
        out.baseline_ex_cache_bytes = diagnostics::PeakExCacheLedger::kExCacheBaselineBytes;
        out.excess_over_baseline_bytes =
            out.ex_cache_net_bytes > out.baseline_ex_cache_bytes
                ? out.ex_cache_net_bytes - out.baseline_ex_cache_bytes
                : 0;

        auto add = [&](const char* name, const char* owner, std::size_t bytes, bool justified) {
            if (bytes == 0) {
                return;
            }
            diagnostics::PeakSubsystemLine line;
            line.name = name;
            line.owner = owner;
            line.bytes = bytes;
            line.justified = justified;
            out.lines.push_back(std::move(line));
            if (justified) {
                out.justified_excess_bytes += bytes;
            }
        };

        // Attribute retained allocations to their owning subsystem.
        add("baseline_cuda_context", "desktop+ctx", baseline_cuda_used_,
            /*justified=*/true);
        add("training_state", "optimizer", ledger_.total_bytes, /*justified=*/true);
        add("training_state_capacity_overhead",
            "capacity",
            training_state_reserved_bytes_ > ledger_.total_bytes
                ? training_state_reserved_bytes_ - ledger_.total_bytes
                : 0,
            /*justified=*/true);
        add("loss_workspace_required", "loss_workspace",
            loss_workspace_required_bytes_, /*justified=*/true);
        add("loss_workspace_arena", "loss_workspace",
            loss_workspace_allocated_bytes_, /*justified=*/true);
        add("densify_child_workspace", "densification", densify_workspace_bytes_,
            /*justified=*/true);
        add("pool_bucket_cache", "allocator", out.pool_bucket_cache_bytes,
            /*justified=*/true);
        add("pool_bucket_live_rounding_waste", "allocator",
            out.pool_bucket_live_rounding_waste_bytes, /*justified=*/true);
        add("exportable_splat", "viewport", out.exportable_splat_bytes,
            /*justified=*/true);
        add("fastgs_sort_hwm", "fastgs_sort", out.fastgs_sort_hwm_bytes, /*justified=*/true);
        add("fastgs_raster_live", "FastGS", out.fastgs_raster_live_bytes,
            /*justified=*/true);
        add("rasterizer_arena", "rasterizer", out.arena_capacity_bytes,
            /*justified=*/true);
        add("rasterizer_arena_required", "rasterizer", out.arena_required_bytes,
            /*justified=*/true);
        add("io.decoded_frame_ring", "image_loader", peak_io_ring_bytes_, false);
        add("io.external_codec_and_bucketed", "image_loader", peak_io_external_bytes_, true);
        add("pinned_host_active_cached_steady", "pinned_allocator",
            peak_steady_pinned_host_bytes_, false);

        // Only residuals added above the baseline may cover excess_over_baseline;
        // the baseline already includes training state and the raster working set.
        // Eligible residuals are the loss arena, densification workspace, pool
        // free lists, FastGS sort high-water, exportable splat block, and retained
        // tensor capacity above live N. Transient q16 expansion is excluded.
        const std::size_t capacity_overhead =
            training_state_reserved_bytes_ > ledger_.total_bytes
                ? training_state_reserved_bytes_ - ledger_.total_bytes
                : 0;
        const std::size_t new_justified =
            loss_workspace_allocated_bytes_ + densify_workspace_bytes_ + out.pool_bucket_cache_bytes +
            out.fastgs_sort_hwm_bytes + out.exportable_splat_bytes + capacity_overhead +
            peak_io_external_bytes_;

        // Honest residual: do NOT auto-justify the remainder as "no_trim".
        // MRNF currently trims after refine; free-list residual is the
        // measured pool_bucket_cache line above. Anything left is unattributed.
        out.unjustified_excess_bytes =
            out.excess_over_baseline_bytes > new_justified
                ? out.excess_over_baseline_bytes - new_justified
                : 0;
        if (out.unjustified_excess_bytes > 0) {
            add("unattributed_residual",
                "audit",
                out.unjustified_excess_bytes,
                /*justified=*/false);
        }
        // justified_excess_bytes already sums all justified inventory lines;
        // for the gate, track post-baseline cover separately via new_justified.
        out.justified_excess_bytes = new_justified;
        return out;
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

        // Final sort HWM (TLS still alive on this thread at finalize).
        peak_fastgs_sort_hwm_ = std::max(
            peak_fastgs_sort_hwm_,
            fast_lfs::rasterization::sort_workspace_high_water_bytes());

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
        const double dataloader_wait_ms =
            dataloader_wait_ms_sum_;
        const double dataloader_wait_ms_per_iter =
            dataloader_wait_count_ > 0
                ? dataloader_wait_ms_sum_ / static_cast<double>(dataloader_wait_count_)
                : 0.0;
        const double steady_dataloader_wait_ms_per_iter =
            steady_dataloader_wait_count_ > 0
                ? steady_dataloader_wait_ms_sum_ /
                      static_cast<double>(steady_dataloader_wait_count_)
                : 0.0;
        const auto peak_ledger = peak_ex_cache_ledger();
        const double ex_cache_mib =
            static_cast<double>(peak_ledger.ex_cache_bytes) / (1024.0 * 1024.0);
        const double excess_mib =
            static_cast<double>(peak_ledger.excess_over_baseline_bytes) / (1024.0 * 1024.0);
        const double unjustified_mib =
            static_cast<double>(peak_ledger.unjustified_excess_bytes) / (1024.0 * 1024.0);

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
        out << "  \"dataloader_wait_ms\": " << dataloader_wait_ms << ",\n";
        out << "  \"dataloader_wait_ms_per_iter\": " << dataloader_wait_ms_per_iter << ",\n";
        out << "  \"steady_dataloader_wait_ms_per_iter\": " << steady_dataloader_wait_ms_per_iter << ",\n";
        out << "  \"warmup_allocs_total\": " << warmup_allocs_ << ",\n";
        out << "  \"steady_allocs_total\": " << steady_allocs_ << ",\n";
        out << "  \"warmup_allocs_per_iter\": " << warmup_allocs_iter << ",\n";
        out << "  \"steady_allocs_per_iter\": " << steady_allocs_iter << ",\n";
        out << "  \"steady_pinned_active_cached_bytes\": "
            << peak_steady_pinned_host_bytes_ << ",\n";
        out << "  \"peak_cuda_used_bytes\": " << peak_cuda_used_ << ",\n";
        out << "  \"peak_cuda_total_bytes\": " << peak_cuda_total_ << ",\n";
        out << "  \"baseline_cuda_used_bytes\": " << baseline_cuda_used_ << ",\n";
        out << "  \"peak_pool_reserved_bytes\": " << peak_pool_reserved_ << ",\n";
        out << "  \"peak_pool_used_bytes\": " << peak_pool_used_ << ",\n";
        out << "  \"ex_cache_bytes\": " << peak_ledger.ex_cache_bytes << ",\n";
        out << "  \"ex_cache_mib\": " << ex_cache_mib << ",\n";
        out << "  \"ex_cache_net_bytes\": " << peak_ledger.ex_cache_net_bytes << ",\n";
        out << "  \"ex_cache_net_mib\": "
            << (static_cast<double>(peak_ledger.ex_cache_net_bytes) / (1024.0 * 1024.0))
            << ",\n";
        out << "  \"ex_cache_excess_over_baseline_mib\": " << excess_mib << ",\n";
        out << "  \"ex_cache_unjustified_excess_bytes\": " << peak_ledger.unjustified_excess_bytes
            << ",\n";
        out << "  \"ex_cache_unjustified_excess_mib\": " << unjustified_mib << ",\n";
        out << "  \"last_loss\": " << last_loss_ << ",\n";
        out << "  \"last_psnr\": " << last_psnr_ << ",\n";
        out << "  \"last_live_splats\": " << last_live_splats_ << ",\n";
        out << "  \"alloc_counter_total\": " << lfs::core::alloc_counter::total() << ",\n";
        out << "  \"alloc_sites\": {\n";
        {
            using lfs::core::alloc_counter::Site;
            const Site sites[] = {Site::PoolBucket, Site::PoolAsync, Site::PoolDirect,
                                  Site::Slab, Site::ZerosDirect, Site::Arena,
                                  Site::FastgsSort, Site::Unknown};
            for (std::size_t i = 0; i < sizeof(sites) / sizeof(sites[0]); ++i) {
                out << "    \"" << lfs::core::alloc_counter::site_name(sites[i]) << "\": "
                    << lfs::core::alloc_counter::site_count(sites[i]);
                out << (i + 1 < sizeof(sites) / sizeof(sites[0]) ? ",\n" : "\n");
            }
        }
        out << "  },\n";
        out << "  \"ledger\": {\n";
        out << "    \"params_bytes\": " << ledger_.params_bytes << ",\n";
        out << "    \"optimizer_bytes\": " << ledger_.optimizer_bytes << ",\n";
        out << "    \"gradients_or_helpers_bytes\": " << ledger_.gradients_or_helpers_bytes << ",\n";
        out << "    \"densify_aux_bytes\": " << ledger_.densify_aux_bytes << ",\n";
        out << "    \"loss_workspace_bytes\": " << loss_workspace_allocated_bytes_ << ",\n";
        out << "    \"loss_workspace_required_bytes\": "
            << loss_workspace_required_bytes_ << ",\n";
        out << "    \"loss_workspace_allocated_bytes\": "
            << loss_workspace_allocated_bytes_ << ",\n";
        out << "    \"densify_workspace_bytes\": " << densify_workspace_bytes_ << ",\n";
        out << "    \"training_state_required_bytes\": " << ledger_.total_bytes << ",\n";
        out << "    \"training_state_allocated_bytes\": "
            << training_state_reserved_bytes_ << ",\n";
        out << "    \"training_state_reserved_bytes\": " << training_state_reserved_bytes_ << ",\n";
        out << "    \"pool_bucket_cache_required_bytes\": 0,\n";
        out << "    \"pool_bucket_cache_allocated_bytes\": "
            << peak_ledger.pool_bucket_cache_bytes << ",\n";
        out << "    \"pool_bucket_live_rounding_waste_required_bytes\": 0,\n";
        out << "    \"pool_bucket_live_rounding_waste_allocated_bytes\": "
            << peak_ledger.pool_bucket_live_rounding_waste_bytes << ",\n";
        out << "    \"total_bytes\": " << ledger_.total_bytes << ",\n";
        out << "    \"live_splats\": " << ledger_.live_splats << ",\n";
        out << "    \"bytes_per_splat\": " << ledger_.bytes_per_splat << "\n";
        out << "  },\n";
        out << "  \"peak_ex_cache\": {\n";
        out << "    \"ex_cache_bytes\": " << peak_ledger.ex_cache_bytes << ",\n";
        out << "    \"ex_cache_net_bytes\": " << peak_ledger.ex_cache_net_bytes << ",\n";
        out << "    \"baseline_cuda_used_bytes\": " << peak_ledger.baseline_cuda_used_bytes
            << ",\n";
        out << "    \"baseline_ex_cache_bytes\": " << peak_ledger.baseline_ex_cache_bytes << ",\n";
        out << "    \"excess_over_baseline_bytes\": " << peak_ledger.excess_over_baseline_bytes << ",\n";
        out << "    \"justified_new_bytes\": " << peak_ledger.justified_excess_bytes << ",\n";
        out << "    \"unjustified_excess_bytes\": " << peak_ledger.unjustified_excess_bytes
            << ",\n";
        out << "    \"peak_iter\": " << peak_ledger.peak_iter << ",\n";
        out << "    \"peak_pool_reserved_bytes\": " << peak_ledger.peak_pool_reserved_bytes
            << ",\n";
        out << "    \"peak_pool_used_bytes\": " << peak_ledger.peak_pool_used_bytes << ",\n";
        out << "    \"training_state_required_bytes\": "
            << peak_ledger.training_state_bytes << ",\n";
        out << "    \"training_state_allocated_bytes\": "
            << peak_ledger.training_state_reserved_bytes << ",\n";
        out << "    \"loss_workspace_required_bytes\": "
            << peak_ledger.loss_workspace_required_bytes << ",\n";
        out << "    \"loss_workspace_allocated_bytes\": "
            << peak_ledger.loss_workspace_allocated_bytes << ",\n";
        out << "    \"pool_bucket_cache_required_bytes\": 0,\n";
        out << "    \"pool_bucket_cache_allocated_bytes\": "
            << peak_ledger.pool_bucket_cache_bytes << ",\n";
        out << "    \"pool_bucket_live_rounding_waste_required_bytes\": 0,\n";
        out << "    \"pool_bucket_live_rounding_waste_allocated_bytes\": "
            << peak_ledger.pool_bucket_live_rounding_waste_bytes << ",\n";
        out << "    \"fastgs_raster_required_bytes\": "
            << peak_ledger.fastgs_raster_live_bytes << ",\n";
        out << "    \"fastgs_raster_allocated_bytes\": "
            << peak_ledger.fastgs_raster_live_bytes << ",\n";
        out << "    \"rasterizer_arena_required_bytes\": "
            << peak_ledger.arena_required_bytes << ",\n";
        out << "    \"rasterizer_arena_allocated_bytes\": "
            << peak_ledger.arena_capacity_bytes << ",\n";
        out << "    \"steady_pinned_active_cached_bytes\": "
            << peak_steady_pinned_host_bytes_ << ",\n";
        out << "    \"fastgs_sort_hwm_bytes\": " << peak_ledger.fastgs_sort_hwm_bytes << ",\n";
        out << "    \"fastgs_raster_live_bytes\": " << peak_ledger.fastgs_raster_live_bytes
            << ",\n";
        out << "    \"lines\": [\n";
        for (std::size_t i = 0; i < peak_ledger.lines.size(); ++i) {
            const auto& L = peak_ledger.lines[i];
            out << "      {\"name\": \"" << L.name << "\", \"owner\": \"" << L.owner
                << "\", \"bytes\": " << L.bytes
                << ", \"justified\": " << (L.justified ? "true" : "false") << "}";
            out << (i + 1 < peak_ledger.lines.size() ? ",\n" : "\n");
        }
        out << "    ],\n";
        out << "    \"peak_rows\": [\n";
        for (std::size_t i = 0; i < peak_ledger.peak_rows.size(); ++i) {
            const auto& L = peak_ledger.peak_rows[i];
            out << "      {\"name\": \"" << L.name << "\", \"bytes\": " << L.bytes << "}";
            out << (i + 1 < peak_ledger.peak_rows.size() ? ",\n" : "\n");
        }
        out << "    ]\n";
        out << "  }\n";
        out << "}\n";
        out.close();

        LOG_INFO("PerfBench: wrote {} (steady {:.2f} ms/iter, dl_wait {:.2f} ms/iter steady, "
                 "{:.1f} allocs/iter, peak VRAM {:.1f} MiB, baseline {:.1f} MiB, "
                 "ex_cache {:.1f} MiB / net {:.1f} MiB "
                 "(excess {:.1f} vs baseline, unjustified {:.1f}), "
                 "sort_hwm {:.1f} MiB, {:.1f} B/splat)",
                 path.string(),
                 steady_ms_iter,
                 steady_dataloader_wait_ms_per_iter,
                 steady_allocs_iter,
                 static_cast<double>(peak_cuda_used_) / (1024.0 * 1024.0),
                 static_cast<double>(baseline_cuda_used_) / (1024.0 * 1024.0),
                 ex_cache_mib,
                 static_cast<double>(peak_ledger.ex_cache_net_bytes) / (1024.0 * 1024.0),
                 excess_mib,
                 unjustified_mib,
                 static_cast<double>(peak_ledger.fastgs_sort_hwm_bytes) / (1024.0 * 1024.0),
                 ledger_.bytes_per_splat);
    }

} // namespace lfs::training
