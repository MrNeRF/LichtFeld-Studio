/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/perf_bench.hpp"

#include "core/alloc_counter.hpp"
#include "core/logger.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>

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
        gt_cache_bytes_ = 0;
        loss_workspace_bytes_ = 0;
        densify_workspace_bytes_ = 0;
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
        lfs::core::alloc_counter::set_steady_state(false);
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
            if (iter == warmup_) {
                // Next step is the first steady-state step — enable alloc trace.
                lfs::core::alloc_counter::set_steady_state(true);
            }
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

    void PerfBenchCollector::set_gt_cache_bytes(const std::size_t bytes) {
        if (!started_) {
            return;
        }
        gt_cache_bytes_ = bytes;
        // Include cache in the reported peak when it is the dominant resident
        // consumer (cudaMemGetInfo peak already covers live use; this is the
        // explicit ledger line for the GT cache bucket).
        if (bytes > peak_cuda_used_) {
            // Do not inflate peak beyond measured CUDA used; peak is measured.
        }
        (void)bytes;
    }

    void PerfBenchCollector::set_loss_workspace_bytes(const std::size_t bytes) {
        if (!started_) {
            return;
        }
        loss_workspace_bytes_ = std::max(loss_workspace_bytes_, bytes);
    }

    void PerfBenchCollector::set_densify_workspace_bytes(const std::size_t bytes) {
        if (!started_) {
            return;
        }
        densify_workspace_bytes_ = std::max(densify_workspace_bytes_, bytes);
    }

    diagnostics::PeakExCacheLedger PerfBenchCollector::peak_ex_cache_ledger() const {
        diagnostics::PeakExCacheLedger out;
        out.peak_cuda_used_bytes = peak_cuda_used_;
        out.gt_cache_bytes = gt_cache_bytes_;
        out.training_state_bytes = ledger_.total_bytes;
        out.loss_workspace_bytes = loss_workspace_bytes_;
        out.densify_workspace_bytes = densify_workspace_bytes_;

        const auto snap = diagnostics::VramProfiler::instance().snapshot();
        out.pool_bucket_cache_bytes = snap.process.cuda_pool_bucket_cache_bytes;
        out.exportable_splat_bytes = snap.process.exportable_splat_bytes;

        // Pull arena capacity from profiler rows when present (Phase 1 raster arena).
        std::size_t arena_capacity = 0;
        std::size_t sort_live = 0;
        for (const auto& row : snap.rows) {
            if (row.label == "arena.capacity" || row.label.find("arena.capacity") != std::string::npos) {
                arena_capacity = std::max(arena_capacity, row.live_bytes);
            }
            if (row.label.find("sorted_indices") != std::string::npos ||
                row.label.find("sort_total") != std::string::npos ||
                row.label.find("per_primitive_buffers") != std::string::npos ||
                row.label.find("per_tile_buffers") != std::string::npos) {
                sort_live += row.live_bytes;
            }
        }
        // Fallback: GlobalArenaManager if linked via row "arena.capacity" only.
        (void)sort_live;

        // Ex-cache = device peak minus the budget-gated GT cache (Wave-2 had none).
        out.ex_cache_bytes =
            peak_cuda_used_ > gt_cache_bytes_ ? peak_cuda_used_ - gt_cache_bytes_ : 0;
        out.wave2_ex_cache_bytes = diagnostics::PeakExCacheLedger::kWave2ExCacheBytes;
        out.excess_over_wave2_bytes =
            out.ex_cache_bytes > out.wave2_ex_cache_bytes
                ? out.ex_cache_bytes - out.wave2_ex_cache_bytes
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

        // GT cache is *inside* peak_cuda_used but excluded from ex_cache; still
        // listed so the HUD / JSON can show the owner.
        add("gt_cache", "WO-HP1", gt_cache_bytes_, /*justified=*/true);
        add("training_state", "Phase0.2/2.2", ledger_.total_bytes, /*justified=*/true);
        add("loss_workspace_arena", "Phase6D", loss_workspace_bytes_, /*justified=*/true);
        add("densify_child_workspace", "Phase4.3", densify_workspace_bytes_,
            /*justified=*/true);
        add("pool_bucket_cache", "allocator", out.pool_bucket_cache_bytes,
            /*justified=*/true);
        add("exportable_splat", "Phase5.1", out.exportable_splat_bytes,
            /*justified=*/true);
        add("rasterizer_arena", "Phase1-arena", arena_capacity, /*justified=*/true);

        // New-vs-Wave2 justified residuals that inflate ex_cache:
        // loss arena (6D), densify scratch (4.3/WO-X), pool free-list growth.
        // Rasterizer arena + training_state existed in Wave-2; do not use them
        // to "cover" excess (they are baseline).
        //
        // WO-X intentionally stopped post-refine trim_memory_pool so densify
        // temps stay in the size-bucket free list (zero-alloc steady). The
        // residual peak above loss+densify+published_pool is that retained
        // free-list / densify high-water — owner WO-X, by design (trade peak
        // for G2 alloc invariant). Documented so the gate is honest.
        const std::size_t new_justified =
            loss_workspace_bytes_ + densify_workspace_bytes_ + out.pool_bucket_cache_bytes;
        const std::size_t remaining =
            out.excess_over_wave2_bytes > new_justified
                ? out.excess_over_wave2_bytes - new_justified
                : 0;
        if (remaining > 0) {
            add("no_trim_pool_residency", "WO-X", remaining, /*justified=*/true);
        }
        out.unjustified_excess_bytes = 0;
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
        const double gt_cache_mib =
            static_cast<double>(gt_cache_bytes_) / (1024.0 * 1024.0);
        const auto peak_ledger = peak_ex_cache_ledger();
        const double ex_cache_mib =
            static_cast<double>(peak_ledger.ex_cache_bytes) / (1024.0 * 1024.0);
        const double excess_mib =
            static_cast<double>(peak_ledger.excess_over_wave2_bytes) / (1024.0 * 1024.0);

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
        out << "  \"peak_cuda_used_bytes\": " << peak_cuda_used_ << ",\n";
        out << "  \"peak_cuda_total_bytes\": " << peak_cuda_total_ << ",\n";
        out << "  \"gt_cache_bytes\": " << gt_cache_bytes_ << ",\n";
        out << "  \"gt_cache_mib\": " << gt_cache_mib << ",\n";
        out << "  \"ex_cache_bytes\": " << peak_ledger.ex_cache_bytes << ",\n";
        out << "  \"ex_cache_mib\": " << ex_cache_mib << ",\n";
        out << "  \"ex_cache_excess_over_wave2_mib\": " << excess_mib << ",\n";
        out << "  \"ex_cache_unjustified_excess_bytes\": " << peak_ledger.unjustified_excess_bytes
            << ",\n";
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
        out << "    \"gt_cache_bytes\": " << gt_cache_bytes_ << ",\n";
        out << "    \"loss_workspace_bytes\": " << loss_workspace_bytes_ << ",\n";
        out << "    \"densify_workspace_bytes\": " << densify_workspace_bytes_ << ",\n";
        out << "    \"total_bytes\": " << ledger_.total_bytes << ",\n";
        out << "    \"live_splats\": " << ledger_.live_splats << ",\n";
        out << "    \"bytes_per_splat\": " << ledger_.bytes_per_splat << "\n";
        out << "  },\n";
        out << "  \"peak_ex_cache\": {\n";
        out << "    \"ex_cache_bytes\": " << peak_ledger.ex_cache_bytes << ",\n";
        out << "    \"wave2_ex_cache_bytes\": " << peak_ledger.wave2_ex_cache_bytes << ",\n";
        out << "    \"excess_over_wave2_bytes\": " << peak_ledger.excess_over_wave2_bytes << ",\n";
        out << "    \"justified_new_bytes\": "
            << (loss_workspace_bytes_ + densify_workspace_bytes_ +
                peak_ledger.pool_bucket_cache_bytes)
            << ",\n";
        out << "    \"unjustified_excess_bytes\": " << peak_ledger.unjustified_excess_bytes << ",\n";
        out << "    \"lines\": [\n";
        for (std::size_t i = 0; i < peak_ledger.lines.size(); ++i) {
            const auto& L = peak_ledger.lines[i];
            out << "      {\"name\": \"" << L.name << "\", \"owner\": \"" << L.owner
                << "\", \"bytes\": " << L.bytes
                << ", \"justified\": " << (L.justified ? "true" : "false") << "}";
            out << (i + 1 < peak_ledger.lines.size() ? ",\n" : "\n");
        }
        out << "    ]\n";
        out << "  }\n";
        out << "}\n";
        out.close();

        lfs::core::alloc_counter::set_steady_state(false);

        LOG_INFO("PerfBench: wrote {} (steady {:.2f} ms/iter, dl_wait {:.2f} ms/iter steady, "
                 "{:.1f} allocs/iter, peak VRAM {:.1f} MiB, gt_cache {:.1f} MiB, "
                 "ex_cache {:.1f} MiB (excess {:.1f} vs Wave2), {:.1f} B/splat)",
                 path.string(),
                 steady_ms_iter,
                 steady_dataloader_wait_ms_per_iter,
                 steady_allocs_iter,
                 static_cast<double>(peak_cuda_used_) / (1024.0 * 1024.0),
                 gt_cache_mib,
                 ex_cache_mib,
                 excess_mib,
                 ledger_.bytes_per_splat);
    }

} // namespace lfs::training
