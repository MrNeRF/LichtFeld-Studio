/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/sh_value_storage.hpp"

#include "core/cuda/sh_layout.cuh"
#include "core/cuda_error.hpp"
#include "core/logger.hpp"
#include "core/sh_value_quant.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "lfs/training/sh_value_codec.hpp"
#include "lfs/training/sh_value_quant_kernels.hpp"

#include <algorithm>
#include <atomic>
#include <cuda_runtime.h>
#include <stdexcept>

namespace lfs::training::sh_value {

    namespace {
        using core::DataType;
        using core::Device;
        using core::Tensor;
        using core::TensorShape;

        constexpr std::size_t kDefaultStagingBytes = 96ull << 20; // 96 MiB
        constexpr std::size_t kMinStagingBytes = 4ull << 20;      // 4 MiB floor
        // Quant-block alignment (encode kernel + bounds float2 stride).
        constexpr std::size_t kQuantBlockPrims =
            static_cast<std::size_t>(core::sh_value_quant::kBlockSize);
        // Reorder swizzle alignment for contiguous float/u16 subranges.
        constexpr std::size_t kReorderPrims =
            static_cast<std::size_t>(core::kShReorderSize);

        [[nodiscard]] std::uint32_t layout_rest(const core::SplatData& splat) {
            return static_cast<std::uint32_t>(splat.max_sh_coeffs_rest());
        }

        /// Primitive capacity for quant buffers: means capacity (max_cap), never exact-N only.
        [[nodiscard]] std::size_t prim_capacity(const core::SplatData& splat) {
            const auto n = static_cast<std::size_t>(splat.size());
            if (!splat.means().is_valid())
                return n;
            const auto cap = splat.means().capacity();
            return std::max(cap > 0 ? cap : n, n);
        }

        /// Full device barrier after encode/decode. Stream-only sync is not enough:
        /// densify runs on the strategy stream while the next forward may launch on
        /// the training stream without an intervening wait (ISS-2.1 multi-stream UAF).
        void sync_codec_stream(cudaStream_t /*stream*/) {
            LFS_CUDA_CHECK_MSG(cudaDeviceSynchronize(), "sh_value quant codec device barrier");
        }

        [[nodiscard]] bool is_exportable_backed(const core::SplatData& splat) {
            if (!splat.has_tensor_allocator()) {
                return false;
            }
            if (!splat.means().is_valid() || !splat.means().is_external_storage()) {
                // Allocator present but means not external yet (cold migrate) —
                // still prefer chunked path so codes land via allocate_named_param.
                return true;
            }
            const auto kind = splat.means().external_storage_kind();
            return kind == "vulkan_external_buffer" || kind == "splat.exportable";
        }

        struct PublishStaging {
            Tensor codes;  // u16 cells for max_chunk_prims
            Tensor bounds; // float2 × ceil(max_chunk/256)
            std::size_t max_chunk_prims = 0;
            std::uint32_t rest = 0;
            std::uint64_t generation = 0;
        };

        PublishStaging g_staging;
        std::atomic<std::int64_t> g_staging_budget_override{-1};

        [[nodiscard]] std::size_t staging_budget_bytes() {
            const auto o = g_staging_budget_override.load(std::memory_order_relaxed);
            if (o >= 0) {
                return static_cast<std::size_t>(o);
            }
            return kDefaultStagingBytes;
        }

        [[nodiscard]] std::size_t max_chunk_prims_for_budget(std::uint32_t rest,
                                                             std::size_t budget_bytes) {
            if (rest == 0 || budget_bytes < kMinStagingBytes) {
                return kQuantBlockPrims;
            }
            const auto n_cells =
                static_cast<std::size_t>(core::sh_value_quant::n_value_cells_per_prim(rest));
            // codes + bounds: C * n_cells * 2 + ceil(C/256) * 8 ≈ C * (2*n_cells) + small
            const std::size_t bytes_per_prim = n_cells * sizeof(std::uint16_t) + 1;
            std::size_t max_c = budget_bytes / std::max<std::size_t>(bytes_per_prim, 1);
            // Align down to quant-block (also multiple of reorder R=32).
            max_c = (max_c / kQuantBlockPrims) * kQuantBlockPrims;
            if (max_c < kQuantBlockPrims) {
                max_c = kQuantBlockPrims;
            }
            return max_c;
        }

        bool ensure_staging(std::uint32_t rest) {
            const std::size_t budget = staging_budget_bytes();
            const std::size_t want_prims = max_chunk_prims_for_budget(rest, budget);
            if (g_staging.max_chunk_prims >= want_prims && g_staging.rest == rest &&
                g_staging.codes.is_valid() && g_staging.bounds.is_valid()) {
                return true;
            }

            const auto n_cells = core::sh_value_quant::n_value_cells_per_prim(rest);
            const std::size_t code_cells =
                core::sh_value_quant::sh_value_u16_count(want_prims, rest);
            const std::size_t n_bounds =
                core::sh_value_quant::n_bounds_for_prims(want_prims);

            Tensor codes = Tensor::zeros_direct(TensorShape({code_cells}),
                                                code_cells,
                                                Device::CUDA,
                                                DataType::Float16);
            codes.set_name("shN.q16_publish_staging.codes");
            Tensor bounds = Tensor::zeros_direct(TensorShape({n_bounds * 2}),
                                                 n_bounds * 2,
                                                 Device::CUDA,
                                                 DataType::Float32);
            bounds.set_name("shN.q16_publish_staging.bounds");

            g_staging.codes = std::move(codes);
            g_staging.bounds = std::move(bounds);
            g_staging.max_chunk_prims = want_prims;
            g_staging.rest = rest;

            LOG_INFO("SH q16 publish staging: budget={} MiB chunk_prims={} rest={} "
                     "codes={} MiB bounds={} KiB",
                     budget >> 20,
                     want_prims,
                     rest,
                     (code_cells * sizeof(std::uint16_t)) >> 20,
                     (n_bounds * 2 * sizeof(float)) >> 10);
            return true;
        }

        /// Copy one reorder-block group of u16 codes from relative staging into live.
        void publish_codes_chunk(std::uint16_t* live,
                                 const std::uint16_t* staging,
                                 std::size_t prim_begin,
                                 std::size_t prim_count,
                                 std::uint32_t n_cells,
                                 cudaStream_t stream) {
            // Layout [ceil(N/R), n_cells, R]: each reorder-block of R prims is
            // contiguous for all cells.
            const std::size_t n_blocks = prim_count / kReorderPrims;
            const std::size_t live_block0 = prim_begin / kReorderPrims;
            const std::size_t cells_per_block =
                static_cast<std::size_t>(n_cells) * kReorderPrims;
            for (std::size_t b = 0; b < n_blocks; ++b) {
                const std::size_t src_off = b * cells_per_block;
                const std::size_t dst_off = (live_block0 + b) * cells_per_block;
                LFS_CUDA_CHECK_MSG(
                    cudaMemcpyAsync(live + dst_off,
                                    staging + src_off,
                                    cells_per_block * sizeof(std::uint16_t),
                                    cudaMemcpyDeviceToDevice,
                                    stream),
                    "sh_value chunked publish codes memcpy");
            }
        }

        void publish_bounds_chunk(float* live_bounds_f2_as_float,
                                  const float* staging_bounds,
                                  std::size_t prim_begin,
                                  std::size_t prim_count,
                                  cudaStream_t stream) {
            const std::size_t qb0 = prim_begin / kQuantBlockPrims;
            const std::size_t n_qb = (prim_count + kQuantBlockPrims - 1) / kQuantBlockPrims;
            LFS_CUDA_CHECK_MSG(
                cudaMemcpyAsync(live_bounds_f2_as_float + qb0 * 2,
                                staging_bounds,
                                n_qb * 2 * sizeof(float),
                                cudaMemcpyDeviceToDevice,
                                stream),
                "sh_value chunked publish bounds memcpy");
        }

        /// Race-free re-encode for exportable: encode into bounded staging, then
        /// copy each chunk into the live q16 region under a device barrier
        /// (frame-boundary publish). Viewer + training never read a chunk mid-write.
        bool commit_exportable_chunked(core::SplatData& splat) {
            auto& shN = splat.shN();
            if (!shN.is_valid() || shN.dtype() != DataType::Float32)
                return false;

            const auto n = static_cast<std::size_t>(splat.size());
            const auto rest = layout_rest(splat);
            if (n == 0 || rest == 0)
                return false;

            if (!ensure_staging(rest))
                return false;

            const auto cap = prim_capacity(splat);
            const auto n_cells_total = core::sh_value_quant::sh_value_u16_count(n, rest);
            const auto capacity_cells = core::sh_value_quant::sh_value_u16_count(cap, rest);
            const auto n_bounds = core::sh_value_quant::n_bounds_for_prims(n);
            const auto n_bounds_cap = core::sh_value_quant::n_bounds_for_prims(cap);
            const auto n_cells_per_prim = core::sh_value_quant::n_value_cells_per_prim(rest);

            // Live region: single resident q16 codes + bounds in the exportable block.
            Tensor live_u16 = splat.allocate_named_param(
                TensorShape({n_cells_total}),
                std::max(n_cells_total, capacity_cells),
                DataType::Float16,
                "SplatData.shN");
            Tensor live_bounds = splat.allocate_named_param(
                TensorShape({n_bounds * 2}),
                std::max(n_bounds, n_bounds_cap) * 2,
                DataType::Float32,
                "SplatData.shN_value_bounds");
            live_u16.set_name("splat.shN");
            live_bounds.set_name("splat.shN_value_bounds");

            Tensor float_src = shN;
            if (float_src.device() != Device::CUDA)
                float_src = float_src.cuda();
            if (!float_src.is_contiguous())
                float_src = float_src.contiguous();

            const cudaStream_t stream = core::getCurrentCUDAStream();
            if (live_u16.stream() != stream)
                live_u16.set_stream(stream);
            if (live_bounds.stream() != stream)
                live_bounds.set_stream(stream);
            if (float_src.stream() != stream)
                float_src.set_stream(stream);
            if (g_staging.codes.stream() != stream)
                g_staging.codes.set_stream(stream);
            if (g_staging.bounds.stream() != stream)
                g_staging.bounds.set_stream(stream);

            auto* live_codes =
                reinterpret_cast<std::uint16_t*>(live_u16.data_ptr());
            auto* live_b = live_bounds.ptr<float>();
            auto* staging_codes =
                reinterpret_cast<std::uint16_t*>(g_staging.codes.data_ptr());
            auto* staging_b = g_staging.bounds.ptr<float>();
            const float* float_ptr = float_src.ptr<float>();

            // Drain concurrent GPU readers of the live region before first write.
            // Viewport zero-copy and training share the exportable VMM block;
            // mid-frame swaps are forbidden — publish only at this barrier.
            sync_codec_stream(stream);

            const std::size_t chunk = g_staging.max_chunk_prims;
            std::size_t published_prims = 0;
            std::size_t n_chunks = 0;
            for (std::size_t p0 = 0; p0 < n; p0 += chunk) {
                std::size_t count = std::min(chunk, n - p0);
                // Align chunk end up to reorder boundary only for the last partial
                // range is handled by encode's n_primitives cap; keep count as-is
                // but ensure encode sees a contiguous float subrange. Float layout
                // for [p0, p0+count) is contiguous when p0 % R == 0.
                if (p0 % kReorderPrims != 0) {
                    LOG_ERROR("SH chunked publish: prim_begin {} not reorder-aligned", p0);
                    return false;
                }

                // Float offset for reorder-aligned p0.
                const std::size_t float_off = core::sh_swizzled_float_count(p0, rest);
                const float* src_chunk = float_ptr + float_off;

                // Encode into staging as a mini-buffer of `count` prims (relative indices).
                encode_shN_float4_to_u16(
                    src_chunk,
                    staging_codes,
                    staging_b,
                    count,
                    rest,
                    stream);
                // Finish encode before publishing this chunk into the live region.
                sync_codec_stream(stream);

                publish_codes_chunk(live_codes,
                                    staging_codes,
                                    p0,
                                    // Publish full reorder-padded span covered by encode.
                                    core::sh_swizzled_padded_n(count),
                                    n_cells_per_prim,
                                    stream);
                // Bounds: only real quant-blocks for the live prim range.
                const std::size_t prim_end = std::min(p0 + count, n);
                const std::size_t bounds_prims = prim_end - p0;
                publish_bounds_chunk(live_b, staging_b, p0, bounds_prims, stream);
                sync_codec_stream(stream);

                published_prims = prim_end;
                ++n_chunks;
            }
            (void)published_prims;

            ++g_staging.generation;

            shN = std::move(live_u16);
            splat.shN_value_bounds() = std::move(live_bounds);

            LOG_DEBUG("SH q16 chunked publish: N={} cap={} rest={} chunks={} gen={} "
                      "staging_chunk_prims={}",
                      n,
                      cap,
                      rest,
                      n_chunks,
                      g_staging.generation,
                      g_staging.max_chunk_prims);
            return true;
        }

    } // namespace

    bool apply_shN_value_quant(core::SplatData& splat) {
        if (!sh_value_quant_enabled())
            return false;
        auto& shN = splat.shN();
        if (!shN.is_valid() || shN.numel() == 0)
            return false;
        // Already pad-dropped q16 (codes + bounds). IEEE f16 float4-swizzle is also
        // Float16 but has no bounds — that path expands to float then re-encodes.
        if (shN.dtype() == DataType::Float16 && splat.shN_value_quantized())
            return false;

        const auto n = static_cast<std::size_t>(splat.size());
        const auto rest = layout_rest(splat);
        if (n == 0 || rest == 0)
            return false;

        // Capacity must track means capacity (max_cap), not exact-N.
        const auto cap = prim_capacity(splat);
        const auto n_cells = core::sh_value_quant::sh_value_u16_count(n, rest);
        const auto capacity_cells = core::sh_value_quant::sh_value_u16_count(cap, rest);
        const auto n_bounds = core::sh_value_quant::n_bounds_for_prims(n);
        const auto n_bounds_cap = core::sh_value_quant::n_bounds_for_prims(cap);

        // Prefer the model's backing allocator (exportable / Vulkan-external) so
        // codes + bounds land in the shared block the viewer zero-copies.
        Tensor u16 = splat.allocate_named_param(
            TensorShape({n_cells}),
            std::max(n_cells, capacity_cells),
            DataType::Float16,
            "SplatData.shN");
        Tensor bounds = splat.allocate_named_param(
            TensorShape({n_bounds * 2}),
            std::max(n_bounds, n_bounds_cap) * 2,
            DataType::Float32,
            "SplatData.shN_value_bounds");
        u16.set_name("splat.shN");
        bounds.set_name("splat.shN_value_bounds");

        // Source may be fp32 float4-swizzle or IEEE f16 float4-swizzle (standalone
        // / pre-quant). Stage to float for the encode kernel.
        Tensor float_src = shN;
        if (float_src.dtype() == DataType::Float16) {
            float_src = float_src.to(DataType::Float32);
        }
        if (float_src.device() != Device::CUDA)
            float_src = float_src.cuda();
        if (!float_src.is_contiguous())
            float_src = float_src.contiguous();

        // Encode on the current training stream, then sync before releasing the float
        // source. Null-stream launch + immediate free was the densify re-encode UAF
        // (next FastGS preprocess illegal address — ISS-2.1).
        const cudaStream_t stream = core::getCurrentCUDAStream();
        if (u16.stream() != stream)
            u16.set_stream(stream);
        if (bounds.stream() != stream)
            bounds.set_stream(stream);
        if (float_src.stream() != stream)
            float_src.set_stream(stream);

        encode_shN_float4_to_u16(
            float_src.ptr<float>(),
            reinterpret_cast<std::uint16_t*>(u16.data_ptr()),
            bounds.ptr<float>(),
            n,
            rest,
            stream);
        sync_codec_stream(stream);

        shN = std::move(u16);
        splat.shN_value_bounds() = std::move(bounds);
        LOG_DEBUG("SH value quant applied: N={} cap={} rest={} cells={} bounds={}",
                  n, cap, rest, n_cells, n_bounds);
        return true;
    }

    bool ensure_shN_fp32_for_mutation(core::SplatData& splat) {
        auto& shN = splat.shN();
        if (!shN.is_valid() || shN.dtype() != DataType::Float16)
            return false;

        const auto n = static_cast<std::size_t>(splat.size());
        const auto rest = layout_rest(splat);
        if (n == 0 || rest == 0)
            return false;

        const auto cap = prim_capacity(splat);
        const auto logical_floats = core::sh_swizzled_float_count(n, rest);
        const auto capacity_floats = core::sh_swizzled_float_count(cap, rest);

        // IEEE f16 float4-swizzle (standalone PLY/SOG viewer path, no bounds):
        // element-wise half→float cast. Training exportable is pad-dropped q16
        // (has bounds) and takes the decode path below.
        if (splat.shN_ieee_f16()) {
            Tensor fp32 = shN.to(DataType::Float32);
            if (fp32.device() != Device::CUDA)
                fp32 = fp32.cuda();
            if (!fp32.is_contiguous())
                fp32 = fp32.contiguous();
            // Preserve capacity headroom when possible.
            if (fp32.capacity() < capacity_floats) {
                Tensor room = Tensor::zeros_direct(TensorShape({logical_floats}),
                                                   capacity_floats, Device::CUDA);
                room.set_name("splat.shN");
                room.copy_from(fp32);
                fp32 = std::move(room);
            } else {
                fp32.set_name("splat.shN");
            }
            shN = std::move(fp32);
            splat.shN_value_bounds() = Tensor{};
            return true;
        }

        if (!sh_value_quant_enabled())
            return false;

        Tensor fp32 = Tensor::zeros_direct(TensorShape({logical_floats}),
                                           std::max(logical_floats, capacity_floats),
                                           Device::CUDA);
        fp32.set_name("splat.shN");

        const cudaStream_t stream = core::getCurrentCUDAStream();
        if (fp32.stream() != stream)
            fp32.set_stream(stream);
        if (shN.stream() != stream)
            shN.set_stream(stream);
        auto& bounds = splat.shN_value_bounds();
        if (!bounds.is_valid() ||
            bounds.numel() < core::sh_value_quant::n_bounds_for_prims(n) * 2) {
            // MN-2: rebuilding empty/zero bounds makes decode emit all zeros — a silent
            // SH wipe. Fail loud so densify/relocate never zero SH-rest by accident.
            LOG_ERROR("MN-2: SH value quant expand refused — bounds short for N={} "
                      "(have={} need={}). Refusing silent SH wipe; restore bounds or "
                      "dequant via a known-good checkpoint.",
                      n,
                      bounds.is_valid() ? bounds.numel() : 0,
                      core::sh_value_quant::n_bounds_for_prims(n) * 2);
            throw std::runtime_error(
                "ensure_shN_fp32_for_mutation: shN_value_bounds short/missing — "
                "refusing silent SH wipe (MN-2)");
        }
        if (bounds.stream() != stream)
            bounds.set_stream(stream);

        decode_shN_u16_to_float4(
            reinterpret_cast<const std::uint16_t*>(shN.data_ptr()),
            bounds.ptr<float>(),
            fp32.ptr<float>(),
            n,
            rest,
            stream);
        sync_codec_stream(stream);

        shN = std::move(fp32);
        // ISS-027: drop bounds once codes are expanded. Leaving pad-dropped bounds
        // attached to a Float32 float4-swizzle buffer is a dual-representation
        // footgun (FastGS/viewer may treat Float16+bounds as q16; a later
        // partial rebind can re-install codes without matching bounds).
        splat.shN_value_bounds() = Tensor{};
        return true;
    }

    bool commit_shN_after_mutation(core::SplatData& splat) {
        auto& shN = splat.shN();
        if (!shN.is_valid() || shN.dtype() != DataType::Float32)
            return false;

        if (!sh_value_quant_enabled())
            return false;

        // Exportable/GUI: chunked staging publish into the single live q16 region
        // so peak SH residency is q16 + staging (not fp32-for-the-whole-refine-window
        // and not 2× full q16). Headless pool path keeps the direct full encode.
        if (is_exportable_backed(splat)) {
            return commit_exportable_chunked(splat);
        }

        // Rebuild codes+bounds from float after densify (heal-vs-rebuild: always rebuild
        // value storage so block min/max match post-growth N). Adam moments stay on the
        // float4-swizzle layout and are healed in prepare_fastgs_fused_adam.
        return apply_shN_value_quant(splat);
    }

    void release_shN_publish_staging() {
        g_staging.codes = Tensor{};
        g_staging.bounds = Tensor{};
        g_staging.max_chunk_prims = 0;
        g_staging.rest = 0;
        // Keep generation monotonic across refine windows so consumers never
        // confuse pre/post-release publishes.
        LOG_DEBUG("SH q16 publish staging released (gen={})", g_staging.generation);
    }

    std::uint64_t shN_publish_generation() noexcept {
        return g_staging.generation;
    }

    std::size_t shN_publish_staging_budget_bytes() noexcept {
        return staging_budget_bytes();
    }

    void set_shN_publish_staging_budget_for_testing(std::optional<std::size_t> bytes) {
        if (!bytes.has_value()) {
            g_staging_budget_override.store(-1, std::memory_order_relaxed);
            return;
        }
        g_staging_budget_override.store(static_cast<std::int64_t>(*bytes),
                                        std::memory_order_relaxed);
        // Force reallocate on next ensure.
        g_staging.codes = Tensor{};
        g_staging.bounds = Tensor{};
        g_staging.max_chunk_prims = 0;
        g_staging.rest = 0;
    }

    void bind_shN_quant_for_raster(const core::SplatData& splat) {
        (void)splat;
    }

    void clear_shN_quant_for_raster() {
    }

} // namespace lfs::training::sh_value
