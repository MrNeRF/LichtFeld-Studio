/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "strategy_utils.hpp"
#include "core/assert.hpp"
#include "core/cuda/memory_arena.hpp"
#include "core/cuda/sh_layout.cuh"
#include "core/lazy_direct_storage.hpp"
#include "core/logger.hpp"
#include "core/memory_pressure.hpp"
#include "kernels/pruning_kernels.hpp"
#include "lfs/training/sh_value_storage.hpp"
#include "lfs/training/vram_ledger.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

namespace lfs::training {

    namespace {
        std::atomic<std::uint64_t> g_frozen_mask_rebuilds{0};

        [[nodiscard]] std::vector<bool> build_frozen_vector(
            const lfs::core::SplatData& splat_data,
            const size_t n,
            size_t* frozen_count = nullptr) {
            std::vector<bool> mask(n, false);
            size_t count = 0;

            for (const auto& range : splat_data.frozen_ranges()) {
                if (range.count == 0 || range.start >= n) {
                    continue;
                }
                const size_t end = range.count > n - range.start ? n : range.start + range.count;
                for (size_t i = range.start; i < end; ++i) {
                    if (!mask[i]) {
                        mask[i] = true;
                        ++count;
                    }
                }
            }

            if (frozen_count) {
                *frozen_count = count;
            }
            return mask;
        }

        [[nodiscard]] size_t frozen_ranges_fingerprint(const lfs::core::SplatData& splat_data) {
            size_t h = 1469598103934665603ull; // FNV-1a offset
            for (const auto& range : splat_data.frozen_ranges()) {
                h ^= range.start + 0x9e3779b97f4a7c15ull;
                h *= 1099511628211ull;
                h ^= range.count + 0x9e3779b97f4a7c15ull;
                h *= 1099511628211ull;
            }
            return h;
        }

        // This process-wide cache avoids rebuilding the frozen mask when its
        // model, size, ranges, and device are unchanged.
        struct FrozenMaskCache {
            const void* key = nullptr;
            size_t n = 0;
            size_t ranges_fp = 0;
            lfs::core::Device device = lfs::core::Device::CUDA;
            lfs::core::Tensor mask;
        };
        FrozenMaskCache g_frozen_mask_cache;

        void grow_score_preserving_prefix(
            lfs::core::Tensor& scores,
            const size_t desired_capacity,
            const lfs::core::Device device) {
            if (scores.capacity() >= desired_capacity) {
                return;
            }
            const bool direct_cuda_storage =
                device == lfs::core::Device::CUDA &&
                (scores.is_external_storage() || !scores.external_storage_kind().empty());
            if (!direct_cuda_storage) {
                scores.reserve(desired_capacity);
                return;
            }

            const lfs::core::Tensor source = scores;
            auto grown = lfs::core::Tensor::zeros_direct(
                source.shape(), desired_capacity, device, source.dtype());
            if (source.numel() > 0) {
                const auto stream = grown.stream();
                source.sync_to_stream(stream);
                LFS_CUDA_CHECK(cudaMemcpyAsync(
                    grown.data_ptr(), source.data_ptr(), source.bytes(),
                    cudaMemcpyDeviceToDevice, stream));
                LFS_CUDA_CHECK(cudaStreamSynchronize(stream));
            }
            scores = std::move(grown);
        }

    } // namespace

    std::uint64_t frozen_mask_rebuild_count() noexcept {
        return g_frozen_mask_rebuilds.load(std::memory_order_relaxed);
    }

    void reset_frozen_mask_rebuild_count() noexcept {
        g_frozen_mask_rebuilds.store(0, std::memory_order_relaxed);
    }

    namespace {
        std::atomic<size_t> g_vram_limit_logged_n{0};
        std::atomic<size_t> g_vram_extra_reserve{0};

        // Pool allocations made while the arena is at its high-water (loss
        // workspaces, images, densify scratch) and views heavier than any seen.
        constexpr size_t kPoolReserveBytes = size_t{256} << 20;

        // Planned before any refine was measured (a run resumed near the
        // limit); measured refines take over as this decays.
        constexpr size_t kDefaultRefineTransientBytesPerGaussian = 64;
        std::atomic<size_t> g_refine_transient_per_gaussian{kDefaultRefineTransientBytesPerGaussian};
        std::atomic<bool> g_refine_window_open{false};
        std::atomic<size_t> g_refine_window_start{0};
        std::atomic<size_t> g_pool_reserved_high_closed{0};
        std::atomic<size_t> g_pool_used_high_closed{0};
        std::atomic<size_t> g_refine_windows{0};

        constexpr int kFrameProbeInterval = 64;
        constexpr size_t kFrameHeadroomMarginBytes = size_t{256} << 20;
        std::atomic<size_t> g_frame_transient{0};

        [[nodiscard]] cudaMemPool_t default_device_pool() {
            int device = 0;
            cudaMemPool_t pool = nullptr;
            if (cudaGetDevice(&device) != cudaSuccess ||
                cudaDeviceGetDefaultMemPool(&pool, device) != cudaSuccess) {
                (void)cudaGetLastError();
                return nullptr;
            }
            return pool;
        }

        [[nodiscard]] size_t pool_attribute(const cudaMemPool_t pool, const cudaMemPoolAttr attr) {
            std::uint64_t value = 0;
            if (!pool || cudaMemPoolGetAttribute(pool, attr, &value) != cudaSuccess) {
                (void)cudaGetLastError();
                return 0;
            }
            return static_cast<size_t>(value);
        }

        void fold_high(std::atomic<size_t>& closed, const size_t bytes) {
            size_t high = closed.load(std::memory_order_relaxed);
            while (bytes > high && !closed.compare_exchange_weak(high, bytes, std::memory_order_relaxed)) {
            }
        }

        // Pool bytes the open refine window reserved beyond its start.
        [[nodiscard]] size_t refine_window_transient_bytes() {
            if (!g_refine_window_open.load(std::memory_order_relaxed)) {
                return 0;
            }
            const size_t high = pool_attribute(default_device_pool(), cudaMemPoolAttrReservedMemHigh);
            const size_t start = g_refine_window_start.load(std::memory_order_relaxed);
            return high > start ? high - start : 0;
        }
    } // namespace

    RefineVramWindow::RefineVramWindow(const lfs::core::SplatData& splat_data)
        : splat_data_(splat_data),
          pressure_episodes_(lfs::core::MemoryPressureCoordinator::instance().episode_count()) {
        const cudaMemPool_t pool = default_device_pool();
        if (!pool) {
            return;
        }
        fold_high(g_pool_reserved_high_closed, pool_attribute(pool, cudaMemPoolAttrReservedMemHigh));
        std::uint64_t reset = 0;
        if (cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReservedMemHigh, &reset) != cudaSuccess) {
            (void)cudaGetLastError();
            return;
        }
        g_refine_window_start.store(pool_attribute(pool, cudaMemPoolAttrReservedMemCurrent),
                                    std::memory_order_relaxed);
        g_refine_window_open.store(true, std::memory_order_relaxed);
        g_refine_windows.fetch_add(1, std::memory_order_relaxed);
    }

    RefineVramWindow::~RefineVramWindow() {
        const size_t transient = refine_window_transient_bytes();
        g_refine_window_open.store(false, std::memory_order_relaxed);
        // Relief took the idle arena, which the next frame commits again, so
        // the next refine would run short the same way.
        if (lfs::core::MemoryPressureCoordinator::instance().episode_count() != pressure_episodes_) {
            raise_vram_growth_reserve(kPoolReserveBytes);
        }
        const size_t n = static_cast<size_t>(splat_data_.size());
        if (n == 0) {
            return;
        }
        // The reserve growth always keeps covers a refine's fixed-size
        // staging; only the rest scales with the model. Keep the largest
        // recent refine: one that relocates or splits little must not hide the
        // next one that does a lot. Decays by 1/8 a refine.
        const size_t scaling = transient > kPoolReserveBytes ? transient - kPoolReserveBytes : 0;
        const size_t previous = g_refine_transient_per_gaussian.load(std::memory_order_relaxed);
        g_refine_transient_per_gaussian.store(std::max(scaling / n, previous - previous / 8),
                                              std::memory_order_relaxed);
    }

    FrameVramProbe::FrameVramProbe(const int iter) {
        if (iter % kFrameProbeInterval != 0) {
            return;
        }
        const cudaMemPool_t pool = default_device_pool();
        if (!pool) {
            return;
        }
        fold_high(g_pool_used_high_closed, pool_attribute(pool, cudaMemPoolAttrUsedMemHigh));
        std::uint64_t reset = 0;
        if (cudaMemPoolSetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &reset) != cudaSuccess) {
            (void)cudaGetLastError();
            return;
        }
        start_used_ = pool_attribute(pool, cudaMemPoolAttrUsedMemCurrent);
        refines_ = g_refine_windows.load(std::memory_order_relaxed);
        active_ = true;
    }

    FrameVramProbe::~FrameVramProbe() {
        // A step that refined measures the refine; RefineVramWindow has it.
        if (!active_ || g_refine_windows.load(std::memory_order_relaxed) != refines_) {
            return;
        }
        const size_t high = pool_attribute(default_device_pool(), cudaMemPoolAttrUsedMemHigh);
        const size_t transient = high > start_used_ ? high - start_used_ : 0;
        // Keep the largest recent step, decaying by 1/8 a sample.
        const size_t previous = g_frame_transient.load(std::memory_order_relaxed);
        const size_t planned = std::max(transient, previous - previous / 8);
        g_frame_transient.store(planned, std::memory_order_relaxed);
        lfs::core::set_measured_frame_headroom_bytes(planned + kFrameHeadroomMarginBytes);
    }

    size_t refine_transient_bytes_per_gaussian() {
        return g_refine_transient_per_gaussian.load(std::memory_order_relaxed);
    }

    size_t tensor_pool_reserved_high_bytes() {
        return std::max(g_pool_reserved_high_closed.load(std::memory_order_relaxed),
                        pool_attribute(default_device_pool(), cudaMemPoolAttrReservedMemHigh));
    }

    size_t tensor_pool_used_high_bytes() {
        return std::max(g_pool_used_high_closed.load(std::memory_order_relaxed),
                        pool_attribute(default_device_pool(), cudaMemPoolAttrUsedMemHigh));
    }

    void initialize_gaussians(lfs::core::SplatData& splat_data, int max_cap) {
        g_vram_extra_reserve.store(0, std::memory_order_relaxed);
        g_frame_transient.store(0, std::memory_order_relaxed);
        lfs::core::set_measured_frame_headroom_bytes(0);
        g_vram_limit_logged_n.store(0, std::memory_order_relaxed);
        g_refine_transient_per_gaussian.store(kDefaultRefineTransientBytesPerGaussian, std::memory_order_relaxed);
        // Tensors are already on GPU in the new framework (created with Device::CUDA by default)
        // Gradients are now owned by AdamOptimizer, not SplatData

        // Pre-allocate tensor capacity to avoid reallocations during MCMC operations
        // This eliminates memory fragmentation from varying tensor sizes
        if (max_cap > 0) {
            const size_t capacity = static_cast<size_t>(max_cap);
            LOG_INFO("Pre-allocating tensor capacity for {} Gaussians (parameters)", capacity);

            splat_data.reserve_capacity(capacity);
        }

        // Apply quantization after reserving capacity so the quantized layout
        // inherits the correct capacity.
        lfs::training::sh_value::apply_shN_value_quant(splat_data);
    }

    std::unique_ptr<AdamOptimizer> create_optimizer(
        lfs::core::SplatData& splat_data,
        const lfs::core::param::OptimizationParameters& params) {

        // Create Adam config with per-parameter learning rates
        AdamConfig config;
        config.lr = params.means_lr * splat_data.get_scene_scale(); // Default LR (for means)
        // Use double literals (not float!) to match legacy precision
        config.beta1 = 0.9;
        config.beta2 = 0.999;
        config.eps = 1e-15;

        // Set per-parameter learning rates (matching legacy MCMC strategy)
        config.param_lrs["means"] = params.means_lr * splat_data.get_scene_scale();
        config.param_lrs["sh0"] = params.shs_lr;
        config.param_lrs["shN"] = params.shs_lr / 20.0f; // ShN uses reduced LR (1/20 of SH0)
        config.param_lrs["scaling"] = params.scaling_lr;
        config.param_lrs["rotation"] = params.rotation_lr;
        config.param_lrs["opacity"] = params.opacity_lr;

        // Pre-allocate optimizer state capacity to avoid reallocations during training
        // This dramatically reduces peak memory usage by avoiding double-buffering during growth
        if (params.max_cap > 0) {
            config.initial_capacity = static_cast<size_t>(params.max_cap);
            config.growth_factor = 1.5f; // Still allow growth beyond max_cap if needed
        }

        LOG_DEBUG("Creating optimizer with per-parameter LRs:");
        LOG_DEBUG("  means: {:.2e}", config.param_lrs["means"]);
        LOG_DEBUG("  sh0: {:.2e}", config.param_lrs["sh0"]);
        LOG_DEBUG("  shN: {:.2e}", config.param_lrs["shN"]);
        LOG_DEBUG("  scaling: {:.2e}", config.param_lrs["scaling"]);
        LOG_DEBUG("  rotation: {:.2e}", config.param_lrs["rotation"]);
        LOG_DEBUG("  opacity: {:.2e}", config.param_lrs["opacity"]);

        auto optimizer = std::make_unique<AdamOptimizer>(splat_data, config);

        return optimizer;
    }

    std::unique_ptr<ExponentialLR> create_scheduler(
        const lfs::core::param::OptimizationParameters& params,
        AdamOptimizer& optimizer) {

        // Python: gamma = 0.01^(1/max_steps)
        // This means after max_steps, lr will be 0.01 * initial_lr
        const double gamma = std::pow(0.01, 1.0 / params.iterations);

        return std::make_unique<ExponentialLR>(optimizer, gamma, std::vector<ParamType>{ParamType::Means});
    }

    namespace {
        // Per-Gaussian strategy buffers the ledger does not count (refine
        // weights, screen share, edge scores, densify scratch).
        constexpr size_t kStrategyBytesPerGaussian = 32;
        // The arena's per-Gaussian bytes vary with the view.
        constexpr double kArenaViewVariance = 1.05;

    } // namespace

    void raise_vram_growth_reserve(const size_t bytes) {
        const size_t reserve = g_vram_extra_reserve.fetch_add(bytes, std::memory_order_relaxed) + bytes;
        LOG_WARN("Training ran out of VRAM; densification now keeps {} MiB more free", reserve >> 20);
    }

    size_t vram_growth_limit(const lfs::core::SplatData& splat_data,
                             const AdamOptimizer* optimizer,
                             const size_t transient_bytes_per_gaussian) {
        // Blocks the pool caches without using would otherwise look taken.
        lfs::core::Tensor::trim_device_memory_pool();
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess || total_bytes == 0) {
            (void)cudaGetLastError();
            return std::numeric_limits<size_t>::max();
        }
        const size_t n = static_cast<size_t>(splat_data.size());
        if (n == 0) {
            return std::numeric_limits<size_t>::max();
        }

        size_t arena_bytes = 0;
        size_t arena_recommit = 0;
        if (auto* arena = lfs::core::GlobalArenaManager::instance().try_get_arena()) {
            const auto info = arena->get_memory_info();
            arena_bytes = std::max(info.arena_capacity + info.recommit_bytes, info.peak_usage);
            arena_recommit = info.recommit_bytes;
        }
        const auto ledger = compute_training_state_ledger(splat_data, optimizer);
        // Refines peak above the steady state; the grown model's refines
        // must fit too. A refine already under way counts what it took so far.
        const size_t transient = std::max({transient_bytes_per_gaussian,
                                           g_refine_transient_per_gaussian.load(std::memory_order_relaxed),
                                           refine_window_transient_bytes() / n});
        const double per_gaussian =
            static_cast<double>(ledger.total_bytes) / static_cast<double>(n) +
            static_cast<double>(kStrategyBytesPerGaussian) +
            kArenaViewVariance * static_cast<double>(arena_bytes) / static_cast<double>(n);

        // The arena's growth headroom is needed while it commits in forward;
        // a refine's transient comes after, with the arena already grown. The
        // two never coexist, so the larger one is kept free, not their sum.
        const double reserve =
            static_cast<double>(kPoolReserveBytes + g_vram_extra_reserve.load(std::memory_order_relaxed));
        const double arena_headroom =
            static_cast<double>(lfs::core::RasterizerMemoryArena::growth_headroom_bytes(total_bytes));
        // Arena memory a refine took under pressure is free only until the next frame.
        double available = -static_cast<double>(arena_recommit);
        // What the open refine holds is already part of its transient.
        if (g_refine_window_open.load(std::memory_order_relaxed)) {
            const size_t held = pool_attribute(default_device_pool(), cudaMemPoolAttrReservedMemCurrent);
            const size_t start = g_refine_window_start.load(std::memory_order_relaxed);
            available += static_cast<double>(held > start ? held - start : 0);
        }
        // Rows the viewer-shared parameter block keeps committed past live N
        // need no new memory.
        if (const auto& means = splat_data.means(); means.external_storage_kind() == "splat.exportable" &&
                                                    means.capacity() > n) {
            available += static_cast<double>(means.capacity() - n) *
                         static_cast<double>(ledger.params_bytes) / static_cast<double>(n);
        }
        const auto rooms = [&](const size_t free_now) {
            const double usable = static_cast<double>(free_now) + available - reserve;
            return std::pair{usable - arena_headroom, usable - static_cast<double>(transient) * static_cast<double>(n)};
        };
        // Lazy tensors' commit-ahead pages serve only the tensor that mapped
        // them, and a small growth leaves most of them unused, so they are not
        // counted as free. Once they matter for the room left, hand them back.
        auto [room_after_arena, room_after_refine] = rooms(free_bytes);
        if (const size_t ahead = lfs::core::lazy_direct_storage_ahead_bytes();
            ahead > 0 && std::min(room_after_arena, room_after_refine) < 2.0 * static_cast<double>(ahead)) {
            lfs::core::release_lazy_direct_storage_ahead();
            if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
                (void)cudaGetLastError();
                return 0;
            }
            std::tie(room_after_arena, room_after_refine) = rooms(free_bytes);
        }
        // Growth by d needs room - d * per_gaussian to stay positive: the
        // reserve plus both the arena headroom and the grown model's refine
        // transient remain free.
        if (room_after_arena <= 0.0 || room_after_refine <= 0.0) {
            return 0;
        }
        return static_cast<size_t>(std::min(room_after_arena / per_gaussian,
                                            room_after_refine / (per_gaussian + static_cast<double>(transient))));
    }

    void note_vram_growth_limit(const lfs::core::SplatData& splat_data,
                                const AdamOptimizer* optimizer,
                                const size_t granted,
                                const size_t requested) {
        // One line per 1% of model growth, not one per refine.
        const size_t n = static_cast<size_t>(splat_data.size());
        const size_t logged = g_vram_limit_logged_n.load(std::memory_order_relaxed);
        if (n <= logged + logged / 100 && n + n / 100 >= logged) {
            return;
        }
        g_vram_limit_logged_n.store(n, std::memory_order_relaxed);
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        (void)cudaMemGetInfo(&free_bytes, &total_bytes);
        size_t arena_bytes = 0;
        if (auto* arena = lfs::core::GlobalArenaManager::instance().try_get_arena()) {
            arena_bytes = arena->get_memory_info().arena_capacity;
        }
        const size_t state_bytes = compute_training_state_ledger(splat_data, optimizer).total_bytes;
        // Every per-Gaussian buffer (model, optimizer, strategy) is direct storage.
        const size_t direct_bytes = std::max(lfs::core::Tensor::cuda_direct_storage_live_bytes(), state_bytes);
        const size_t pool_bytes = pool_attribute(default_device_pool(), cudaMemPoolAttrReservedMemCurrent);
        const size_t used_bytes = total_bytes - std::min(free_bytes, total_bytes);
        const size_t accounted = direct_bytes + arena_bytes + pool_bytes;
        constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
        const size_t step_headroom = lfs::core::RasterizerMemoryArena::growth_headroom_bytes(total_bytes);
        const size_t refine_plan = g_refine_transient_per_gaussian.load(std::memory_order_relaxed) * n;
        LOG_INFO("VRAM limits densification at {} Gaussians (+{} of {} requested): Gaussians {:.2f} GiB "
                 "({:.2f} GiB model and optimizer), rasterizer {:.2f} GiB, tensor pool {:.2f} GiB, "
                 "other {:.2f} GiB, free {:.2f} GiB (keeps the larger of {:.2f} GiB step headroom and "
                 "{:.2f} GiB refine peak, plus {:.2f} GiB reserve)",
                 n, granted, requested, static_cast<double>(direct_bytes) / kGiB,
                 static_cast<double>(state_bytes) / kGiB, static_cast<double>(arena_bytes) / kGiB,
                 static_cast<double>(pool_bytes) / kGiB,
                 static_cast<double>(used_bytes - std::min(used_bytes, accounted)) / kGiB,
                 static_cast<double>(free_bytes) / kGiB, static_cast<double>(step_headroom) / kGiB,
                 static_cast<double>(refine_plan) / kGiB,
                 static_cast<double>(kPoolReserveBytes + g_vram_extra_reserve.load(std::memory_order_relaxed)) / kGiB);
    }

    size_t clamp_growth_to_vram(const size_t requested,
                                const lfs::core::SplatData& splat_data,
                                const AdamOptimizer* optimizer,
                                const size_t transient_bytes_per_gaussian) {
        if (requested == 0) {
            return 0;
        }
        const size_t limit = vram_growth_limit(splat_data, optimizer, transient_bytes_per_gaussian);
        if (limit >= requested) {
            return requested;
        }
        note_vram_growth_limit(splat_data, optimizer, limit, requested);
        return limit;
    }

    lfs::core::Tensor make_frozen_mask(
        const lfs::core::SplatData& splat_data,
        const size_t n,
        const lfs::core::Device device) {
        if (!splat_data.has_frozen_ranges() || n == 0) {
            return {};
        }

        const size_t ranges_fp = frozen_ranges_fingerprint(splat_data);
        const void* key = static_cast<const void*>(&splat_data);
        if (g_frozen_mask_cache.key == key &&
            g_frozen_mask_cache.n == n &&
            g_frozen_mask_cache.ranges_fp == ranges_fp &&
            g_frozen_mask_cache.device == device &&
            g_frozen_mask_cache.mask.is_valid() &&
            g_frozen_mask_cache.mask.numel() == n) {
            return g_frozen_mask_cache.mask; // share persistent GPU tensor
        }

        size_t frozen_count = 0;
        auto mask = build_frozen_vector(splat_data, n, &frozen_count);
        if (frozen_count == 0) {
            g_frozen_mask_cache = {};
            return {};
        }

        auto tensor = lfs::core::Tensor::from_vector(
            mask,
            lfs::core::TensorShape({n}),
            device);
        tensor.set_name("splat.frozen_mask");
        g_frozen_mask_cache.key = key;
        g_frozen_mask_cache.n = n;
        g_frozen_mask_cache.ranges_fp = ranges_fp;
        g_frozen_mask_cache.device = device;
        g_frozen_mask_cache.mask = tensor;
        g_frozen_mask_rebuilds.fetch_add(1, std::memory_order_relaxed);
        return tensor;
    }

    lfs::core::Tensor make_trainable_mask(
        const lfs::core::SplatData& splat_data,
        const size_t n,
        const lfs::core::Device device) {
        auto frozen_mask = make_frozen_mask(splat_data, n, device);
        return frozen_mask.is_valid()
                   ? frozen_mask.logical_not()
                   : lfs::core::Tensor();
    }

    lfs::core::Tensor exclude_frozen_from_mask(
        const lfs::core::SplatData& splat_data,
        const lfs::core::Tensor& mask) {
        if (!mask.is_valid() || mask.numel() == 0 || !splat_data.has_frozen_ranges()) {
            return mask;
        }

        auto frozen_mask = make_frozen_mask(splat_data, mask.numel(), mask.device());
        return frozen_mask.is_valid()
                   ? mask.logical_and(frozen_mask.logical_not())
                   : mask;
    }

    lfs::core::Tensor zero_frozen_scores(
        const lfs::core::SplatData& splat_data,
        const lfs::core::Tensor& scores) {
        if (!scores.is_valid() || scores.numel() == 0 || !splat_data.has_frozen_ranges()) {
            return scores;
        }

        auto frozen_mask = make_frozen_mask(splat_data, scores.numel(), scores.device());
        return frozen_mask.is_valid()
                   ? scores.masked_fill(frozen_mask, 0.0f)
                   : scores;
    }

    void zero_frozen_scores_inplace(
        const lfs::core::SplatData& splat_data,
        lfs::core::Tensor& scores) {
        if (!scores.is_valid() || scores.numel() == 0 || !splat_data.has_frozen_ranges()) {
            return;
        }

        auto frozen_mask = make_frozen_mask(splat_data, scores.numel(), scores.device());
        if (frozen_mask.is_valid()) {
            scores.masked_fill_(frozen_mask, 0.0f);
        }
    }

    lfs::core::Tensor apply_crop_damping_to_scores(
        const AdamOptimizer& optimizer,
        const lfs::core::Tensor& scores) {
        const auto& crop_mask = optimizer.crop_score_mask();
        const float scale = optimizer.cropbox_lr_scale();
        if (!crop_mask.is_valid() || crop_mask.numel() == 0 || scale == 1.0f) {
            return scores;
        }

        LFS_ASSERT_MSG(scores.is_valid(), "crop-damped scores must be valid");
        LFS_ASSERT_MSG(scores.ndim() == 1, "crop-damped scores must be one-dimensional");
        LFS_ASSERT_MSG(
            scores.device() == crop_mask.device(),
            "crop damping mask and scores must use the same device");

        // Rows appended by this refine after the mask was built are not damped.
        lfs::core::Tensor active_mask;
        if (scores.numel() == crop_mask.numel()) {
            active_mask = crop_mask;
        } else if (scores.numel() < crop_mask.numel()) {
            active_mask = crop_mask.slice(0, 0, scores.numel());
        } else {
            active_mask = lfs::core::Tensor::cat(
                {crop_mask, lfs::core::Tensor::zeros_bool({scores.numel() - crop_mask.numel()}, crop_mask.device())}, 0);
        }
        if (scale == 0.0f) {
            return scores.masked_fill(active_mask, 0.0f);
        }
        return lfs::core::Tensor::where(active_mask, scores * scale, scores);
    }

    namespace {
        // ROI rows sized to n: rows appended after the mask was built are outside the ROI.
        lfs::core::Tensor roi_rows_for(const lfs::core::Tensor& rows, const size_t n) {
            if (rows.numel() == n)
                return rows;
            if (rows.numel() > n)
                return rows.slice(0, 0, n);
            return lfs::core::Tensor::cat(
                {rows, lfs::core::Tensor::zeros_bool({n - rows.numel()}, rows.device())}, 0);
        }
    } // namespace

    std::optional<RoiGrowthSplit> split_growth_by_roi(
        const AdamOptimizer& optimizer,
        const lfs::core::Tensor& weights,
        const int64_t k) {
        const auto& rows = optimizer.crop_roi_rows();
        if (!rows.is_valid() || k <= 0 || !weights.is_valid() || weights.numel() == 0)
            return std::nullopt;
        const float share = optimizer.cropbox_growth_inside();
        const auto in_roi = roi_rows_for(rows, weights.numel());
        RoiGrowthSplit split;
        split.roi_weights = weights.masked_fill(in_roi.logical_not(), 0.0f);
        split.rest_weights = weights.masked_fill(in_roi, 0.0f);
        // Count only what Gumbel top-k can select: NaN and negative weights are not candidates.
        split.roi_nnz = static_cast<int64_t>(split.roi_weights.gt(0.0f).count_nonzero());
        split.rest_nnz = static_cast<int64_t>(split.rest_weights.gt(0.0f).count_nonzero());
        const int64_t roi_quota = std::llround(static_cast<double>(share) * static_cast<double>(k));
        const int64_t rest_quota = k - roi_quota;
        const int64_t roi_first = std::min(roi_quota, split.roi_nnz);
        const int64_t rest_first = std::min(rest_quota, split.rest_nnz);
        split.roi_count = share > 0.0f ? std::min(split.roi_nnz, roi_first + (rest_quota - rest_first)) : roi_first;
        split.rest_count = share < 1.0f ? std::min(split.rest_nnz, rest_first + (roi_quota - roi_first)) : rest_first;
        return split;
    }

    bool reshapes_sampling_scores(const lfs::core::SplatData& splat_data,
                                  const AdamOptimizer& optimizer) {
        const auto& crop_mask = optimizer.crop_score_mask();
        return splat_data.has_frozen_ranges() ||
               (crop_mask.is_valid() && crop_mask.numel() > 0 && optimizer.cropbox_lr_scale() != 1.0f) ||
               optimizer.crop_roi_rows().is_valid();
    }

    lfs::core::Tensor roi_share_sampling_weights(
        const AdamOptimizer& optimizer,
        const lfs::core::Tensor& weights) {
        const auto& rows = optimizer.crop_roi_rows();
        if (!rows.is_valid() || !weights.is_valid() || weights.numel() == 0)
            return weights;
        const float share = optimizer.cropbox_growth_inside();
        const auto in_roi = roi_rows_for(rows, weights.numel());
        const float roi_sum = weights.masked_fill(in_roi.logical_not(), 0.0f).sum().item();
        const float rest_sum = weights.masked_fill(in_roi, 0.0f).sum().item();
        // A strict share never draws from its excluded side, even when the other side is empty.
        const float roi_scale = roi_sum > 0.0f ? (rest_sum > 0.0f ? share / roi_sum : (share > 0.0f ? 1.0f : 0.0f)) : 0.0f;
        const float rest_scale = rest_sum > 0.0f ? (roi_sum > 0.0f ? (1.0f - share) / rest_sum : (share < 1.0f ? 1.0f : 0.0f)) : 0.0f;
        return lfs::core::Tensor::where(in_roi, weights * roi_scale, weights * rest_scale);
    }

    size_t frozen_row_count(const lfs::core::SplatData& splat_data, const size_t n) {
        size_t frozen_count = 0;
        (void)build_frozen_vector(splat_data, n, &frozen_count);
        return frozen_count;
    }

    namespace {
        void apply_frozen_ranges_to_optimizer_impl(
            const lfs::core::SplatData& splat_data,
            AdamOptimizer& optimizer,
            const float* freeze_lr_scale) {
            if (freeze_lr_scale != nullptr) {
                optimizer.set_frozen_lr_scale(*freeze_lr_scale);
            }

            const size_t n = static_cast<size_t>(splat_data.size());
            auto mask = make_frozen_mask(splat_data, n, lfs::core::Device::CUDA);
            if (!mask.is_valid()) {
                optimizer.set_frozen_mask({});
                return;
            }

            const size_t frozen_count = frozen_row_count(splat_data, n);
            optimizer.set_frozen_mask(std::move(mask));
            if (freeze_lr_scale != nullptr && *freeze_lr_scale > 0.0f) {
                LOG_INFO("Frozen training mask: {} / {} Gaussians frozen (LR scale: {})",
                         frozen_count,
                         n,
                         *freeze_lr_scale);
            } else {
                LOG_INFO("Frozen training mask: {} / {} Gaussians frozen", frozen_count, n);
            }
        }
    } // namespace

    void apply_frozen_ranges_to_optimizer(
        const lfs::core::SplatData& splat_data,
        AdamOptimizer& optimizer) {
        apply_frozen_ranges_to_optimizer_impl(splat_data, optimizer, nullptr);
    }

    void apply_frozen_ranges_to_optimizer(
        const lfs::core::SplatData& splat_data,
        AdamOptimizer& optimizer,
        const float freeze_lr_scale) {
        apply_frozen_ranges_to_optimizer_impl(splat_data, optimizer, &freeze_lr_scale);
    }

    void remap_frozen_ranges_after_compaction(
        lfs::core::SplatData& splat_data,
        const lfs::core::Tensor& kept_old_indices,
        const size_t old_size) {
        if (!splat_data.has_frozen_ranges()) {
            return;
        }
        if (!kept_old_indices.is_valid() || kept_old_indices.numel() == 0) {
            splat_data.clear_frozen_ranges();
            return;
        }

        auto kept_i64 = kept_old_indices.dtype() == lfs::core::DataType::Int64
                            ? kept_old_indices
                            : kept_old_indices.to(lfs::core::DataType::Int64);
        splat_data.remap_frozen_ranges_after_keep(old_size, kept_i64.to_vector_int64());
    }

    lfs::core::Tensor compute_dead_mask_from_opacity_and_rotation(
        const lfs::core::Tensor& opacities,
        const lfs::core::Tensor& rotations,
        const float min_opacity) {
        using namespace lfs::core;

        Tensor flat_opacities = opacities;
        if (flat_opacities.ndim() == 2 && flat_opacities.shape()[1] == 1) {
            flat_opacities = flat_opacities.squeeze(-1);
        }

        const size_t n = flat_opacities.numel();
        assert(flat_opacities.ndim() == 1);
        assert(rotations.ndim() == 2 && rotations.shape()[1] == 4);
        assert(rotations.shape()[0] == n);

        auto dead_mask = Tensor::empty({n}, Device::CUDA, DataType::Bool);
        pruning::launch_compute_dead_mask(
            flat_opacities.ptr<float>(),
            rotations.ptr<float>(),
            dead_mask.ptr<uint8_t>(),
            n,
            min_opacity);
        return dead_mask;
    }

    lfs::core::Tensor compute_near_zero_rotation_mask(
        const lfs::core::Tensor& rotations) {
        using namespace lfs::core;

        const size_t n = rotations.shape()[0];
        assert(rotations.ndim() == 2 && rotations.shape()[1] == 4);

        auto near_zero_mask = Tensor::empty({n}, Device::CUDA, DataType::Bool);
        pruning::launch_compute_near_zero_rotation_mask(
            rotations.ptr<float>(),
            near_zero_mask.ptr<uint8_t>(),
            n);
        return near_zero_mask;
    }

    void DensifyNScratch::ensure_n(const size_t n, const lfs::core::Device device) {
        using namespace lfs::core;
        if (n == 0) {
            return;
        }
        n_required = std::max(n_required, n);
        if (n_capacity >= n) {
            return;
        }
        // Match the model reservation on first use; retain headroom only for
        // subsequent growth beyond that reservation.
        const size_t new_cap = n_capacity == 0
                                   ? n
                                   : std::max(
                                         n, static_cast<size_t>(static_cast<double>(std::max(n_capacity, n)) * 1.2) + 1);
        LFS_ASSERT_MSG(device == Device::CUDA, "DensifyNScratch requires CUDA storage");
        f32_a = Tensor::empty_exact({new_cap}, DataType::Float32);
        bool_a = Tensor::empty_exact({new_cap}, DataType::Bool);
        f32_a.zero_();
        bool_a.zero_();
        n_capacity = new_cap;
    }

    lfs::core::Tensor DensifyNScratch::f32_a_view(const size_t n) const {
        return f32_a.slice(0, 0, n);
    }
    lfs::core::Tensor DensifyNScratch::bool_a_view(const size_t n) const {
        return bool_a.slice(0, 0, n);
    }
    void DensifyChildWorkspace::ensure(
        const size_t K,
        const size_t sh_rest_in,
        const bool use_shN,
        const bool sh0_flat_layout,
        const lfs::core::Device device) {

        using namespace lfs::core;
        if (K == 0)
            return;

        const size_t need = K;
        constexpr std::size_t base_floats_per_row = 3 + 4 + 3 + 1 + 3;
        required_bytes_high_water = std::max(
            required_bytes_high_water,
            K * (base_floats_per_row + (use_shN ? sh_rest_in * 3 : 0)) * sizeof(float));
        const bool layout_changed =
            (sh0_as_flat != sh0_flat_layout) ||
            (use_shN && sh_rest != sh_rest_in) ||
            (use_shN && (!shN.is_valid() || shN.ndim() != 3));

        if (capacity < need || layout_changed) {
            const size_t new_cap = capacity == 0
                                       ? need
                                       : std::max(
                                             need,
                                             static_cast<size_t>(static_cast<double>(std::max(capacity, need)) * 1.2) + 1);
            means = Tensor::empty({new_cap, 3}, device);
            rotations = Tensor::empty({new_cap, 4}, device);
            scales = Tensor::empty({new_cap, 3}, device);
            opacities = Tensor::empty({new_cap}, device);
            sh0_as_flat = sh0_flat_layout;
            if (sh0_flat_layout) {
                sh0_flat = Tensor::empty({new_cap, 3}, device);
                sh0 = Tensor();
            } else {
                sh0 = Tensor::empty({new_cap, 1, 3}, device);
                sh0_flat = Tensor();
            }
            sh_rest = sh_rest_in;
            if (use_shN && sh_rest_in > 0) {
                shN = Tensor::empty({new_cap, sh_rest_in, 3}, device);
            } else {
                shN = Tensor();
            }
            capacity = new_cap;
        } else if (use_shN && sh_rest_in > 0 &&
                   (!shN.is_valid() || shN.shape()[0] < capacity || shN.shape()[1] != sh_rest_in)) {
            sh_rest = sh_rest_in;
            shN = Tensor::empty({capacity, sh_rest_in, 3}, device);
        }
    }

    namespace {
        [[nodiscard]] std::size_t densify_child_bytes(
            const std::size_t rows,
            const std::size_t sh_rest) noexcept {
            // means + rotations + scales + opacity + sh0 + optional rest SH.
            constexpr std::size_t base_floats_per_row = 3 + 4 + 3 + 1 + 3;
            return rows * (base_floats_per_row + sh_rest * 3) * sizeof(float);
        }
    } // namespace

    std::size_t DensifyChildWorkspace::required_bytes() const noexcept {
        return required_bytes_high_water;
    }

    std::size_t DensifyChildWorkspace::resident_bytes() const noexcept {
        return densify_child_bytes(capacity, shN.is_valid() ? sh_rest : 0);
    }

    lfs::core::Tensor DensifyChildWorkspace::means_view(const size_t K) const {
        return means.slice(0, 0, K);
    }
    lfs::core::Tensor DensifyChildWorkspace::rotations_view(const size_t K) const {
        return rotations.slice(0, 0, K);
    }
    lfs::core::Tensor DensifyChildWorkspace::scales_view(const size_t K) const {
        return scales.slice(0, 0, K);
    }
    lfs::core::Tensor DensifyChildWorkspace::sh0_view(const size_t K) const {
        if (sh0_as_flat)
            return sh0_flat.slice(0, 0, K);
        return sh0.slice(0, 0, K);
    }
    lfs::core::Tensor DensifyChildWorkspace::shN_view(const size_t K) const {
        if (!shN.is_valid())
            return lfs::core::Tensor();
        return shN.slice(0, 0, K);
    }
    lfs::core::Tensor DensifyChildWorkspace::opacities_view(const size_t K) const {
        return opacities.slice(0, 0, K);
    }

    void ensure_densification_info_shape_inplace(
        lfs::core::Tensor& densification_info,
        const size_t n,
        const lfs::core::Device device,
        const size_t n_rows) {
        using namespace lfs::core;
        // densification_info is row-major [n_rows, N]: row k starts at offset k*N.
        // One row is kept as [N]: the error row alone. Growing N cannot use
        // append_zeros (that grows dim0). When shape already matches, reuse the
        // buffer (caller zeros). On mismatch, allocate exact [n_rows,n] or [n].
        // Pre-sizing columns would break the rasterizer's row offsets.
        const bool error_row_only = n_rows == 1;
        const size_t rows = error_row_only ? 1 : std::max<size_t>(n_rows, 2);
        auto& info = densification_info;
        if (info.is_valid() &&
            (error_row_only ? info.ndim() == 1 && info.shape()[0] == n
                            : info.ndim() == 2 && info.shape()[0] == rows && info.shape()[1] == n)) {
            return;
        }
        // The error row over the same splats survives a layout change: a
        // resumed checkpoint can hold the last view before it was folded.
        const float* error_row = nullptr;
        if (info.is_valid() && info.dtype() == lfs::core::DataType::Float32 && n > 0) {
            if (info.ndim() == 2 && info.shape()[0] >= 2 && info.shape()[1] == n)
                error_row = info.ptr<float>() + n;
            else if (info.ndim() == 1 && info.shape()[0] == n)
                error_row = info.ptr<float>();
        }
        if (error_row == nullptr) {
            // Release the old buffer first so both never hold VRAM at once.
            info = Tensor();
        }
        Tensor resized = error_row_only ? Tensor::zeros({n}, device) : Tensor::zeros({rows, n}, device);
        if (error_row != nullptr) {
            const auto stream = resized.stream();
            info.sync_to_stream(stream);
            LFS_CUDA_CHECK(cudaMemcpyAsync(resized.ptr<float>() + (error_row_only ? 0 : n), error_row,
                                           n * sizeof(float), cudaMemcpyDeviceToDevice, stream));
            LFS_CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        info = std::move(resized);
        info.set_name("splat.densification_info");
    }

    void ensure_score_buffer_inplace(
        lfs::core::Tensor& scores,
        const size_t n,
        const lfs::core::Device device,
        const size_t reserve_capacity) {
        using namespace lfs::core;
        const size_t desired_cap = reserve_capacity > 0 ? std::max(reserve_capacity, n) : n;

        if (!scores.is_valid() || scores.ndim() != 1 || scores.device() != device ||
            scores.dtype() != DataType::Float32) {
            if (desired_cap > n) {
                scores = Tensor::zeros_direct(TensorShape({n}), desired_cap, device);
            } else {
                scores = Tensor::zeros({n}, device);
            }
            return;
        }

        const size_t cur = scores.numel();
        if (cur == n) {
            if (desired_cap > n && scores.capacity() < desired_cap) {
                grow_score_preserving_prefix(scores, desired_cap, device);
            }
            return;
        }
        if (cur < n) {
            if (scores.capacity() < desired_cap) {
                if (scores.capacity() == 0 || scores.is_external_storage() ||
                    !scores.external_storage_kind().empty()) {
                    // Slow path: realloc with headroom.
                    auto fresh = desired_cap > n
                                     ? Tensor::zeros_direct(TensorShape({n}), desired_cap, device)
                                     : Tensor::zeros({n}, device);
                    if (cur > 0) {
                        cudaMemcpy(fresh.ptr<float>(), scores.ptr<float>(),
                                   cur * sizeof(float), cudaMemcpyDeviceToDevice);
                    }
                    scores = std::move(fresh);
                    return;
                }
                scores.reserve(desired_cap);
            }
            scores.append_zeros(n - cur);
            return;
        }
        // Shrink logical size: reallocate exact (rare; refine reset uses zero_ at same n).
        if (desired_cap > n) {
            scores = Tensor::zeros_direct(TensorShape({n}), desired_cap, device);
        } else {
            scores = Tensor::zeros({n}, device);
        }
    }

    int collect_adam_scale_ptrs(AdamOptimizer& /*optimizer*/, float* /*out_ptrs*/[12]) {
        return 0;
    }

    void zero_adam_grads_at_indices(
        AdamOptimizer& optimizer,
        const lfs::core::Tensor& indices,
        const uint32_t shN_layout_rest) {
        using namespace lfs::core;
        if (!indices.is_valid() || indices.numel() == 0)
            return;

        const ParamType types[] = {
            ParamType::Means, ParamType::Sh0, ParamType::ShN,
            ParamType::Scaling, ParamType::Rotation, ParamType::Opacity};

        for (ParamType pt : types) {
            auto* state = optimizer.get_state_mutable(pt);
            if (!state || !state->grad.is_valid() || state->grad.numel() == 0)
                continue;

            if (pt == ParamType::ShN) {
                if (shN_layout_rest != 0) {
                    auto idx_i32 = indices.dtype() == DataType::Int32
                                       ? indices
                                       : indices.to(DataType::Int32);
                    shN_swizzled_zero_at_indices(
                        state->grad.ptr<float>(), idx_i32.ptr<int>(),
                        idx_i32.numel(), shN_layout_rest);
                }
                continue;
            }

            const auto& shape = state->grad.shape();
            if (shape.rank() == 0)
                continue;
            bool any_zero = false;
            for (size_t d = 0; d < shape.rank(); ++d) {
                if (shape[d] == 0) {
                    any_zero = true;
                    break;
                }
            }
            if (any_zero)
                continue;

            std::vector<size_t> dims = {indices.numel()};
            for (size_t i = 1; i < shape.rank(); ++i)
                dims.push_back(shape[i]);
            auto zeros = Tensor::zeros(TensorShape(dims), state->grad.device());
            state->grad.index_put_(indices, zeros);
        }
    }

} // namespace lfs::training
