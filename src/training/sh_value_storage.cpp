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
#include <cuda_runtime.h>
#include <stdexcept>

namespace lfs::training::sh_value {

    namespace {
        using core::DataType;
        using core::Device;
        using core::Tensor;
        using core::TensorShape;

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
    } // namespace

    bool apply_shN_value_quant(core::SplatData& splat) {
        if (!sh_value_quant_enabled())
            return false;
        auto& shN = splat.shN();
        if (!shN.is_valid() || shN.numel() == 0)
            return false;
        if (shN.dtype() == DataType::Float16)
            return false; // already quantized
        // Exportable/viewer path uses IEEE f16 float4-swizzle in the packed block
        // (half the fp32 bytes). Never apply pad-dropped q16 there — the Vulkan
        // projection shader binds half slots, and quant would detach shN from the
        // exportable VA range.
        const auto is_exportable_kind = [](const Tensor& t) {
            if (!t.is_valid() || !t.is_external_storage())
                return false;
            const auto kind = t.external_storage_kind();
            return kind == "vulkan_external_buffer" || kind == "splat.exportable";
        };
        if (is_exportable_kind(shN) || is_exportable_kind(splat.means())) {
            LOG_DEBUG("SH value quant skipped: exportable/viewer-backed model (IEEE f16 path)");
            return false;
        }

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

        Tensor u16 = Tensor::zeros_direct(TensorShape({n_cells}),
                                          std::max(n_cells, capacity_cells), Device::CUDA,
                                          DataType::Float16);
        Tensor bounds = Tensor::zeros_direct(TensorShape({n_bounds * 2}),
                                             std::max(n_bounds, n_bounds_cap) * 2, Device::CUDA,
                                             DataType::Float32);
        u16.set_name("splat.shN");
        bounds.set_name("splat.shN_value_bounds");

        // Encode on the current training stream, then sync before releasing the float
        // source. Null-stream launch + immediate free was the densify re-encode UAF
        // (next FastGS preprocess illegal address — ISS-2.1).
        const cudaStream_t stream = core::getCurrentCUDAStream();
        if (u16.stream() != stream)
            u16.set_stream(stream);
        if (bounds.stream() != stream)
            bounds.set_stream(stream);
        if (shN.stream() != stream)
            shN.set_stream(stream);

        encode_shN_float4_to_u16(
            shN.ptr<float>(),
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

        // IEEE f16 float4-swizzle (exportable GUI): element-wise half→float cast.
        // Same topology as fp32; densify float-native ops then commit back to f16.
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
        return true;
    }

    bool commit_shN_after_mutation(core::SplatData& splat) {
        auto& shN = splat.shN();
        if (!shN.is_valid() || shN.dtype() != DataType::Float32)
            return false;

        // GUI exportable path: means live in the exportable block, so shN must
        // stay IEEE f16 float4-swizzle (not pad-dropped q16). ensure_* may have
        // expanded to a float temp; convert back to half. Densify grow rebind
        // reinstalls the view into the packed block.
        const auto is_exportable_kind = [](const Tensor& t) {
            if (!t.is_valid() || !t.is_external_storage())
                return false;
            const auto kind = t.external_storage_kind();
            return kind == "vulkan_external_buffer" || kind == "splat.exportable";
        };
        if (is_exportable_kind(splat.means_raw()) || is_exportable_kind(shN)) {
            Tensor half = shN.to(DataType::Float16);
            if (half.device() != Device::CUDA)
                half = half.cuda();
            if (!half.is_contiguous())
                half = half.contiguous();
            half.set_name("splat.shN");
            shN = std::move(half);
            splat.shN_value_bounds() = Tensor{};
            return true;
        }

        if (!sh_value_quant_enabled())
            return false;
        // Rebuild codes+bounds from float after densify (heal-vs-rebuild: always rebuild
        // value storage so block min/max match post-growth N). Adam moments stay on the
        // float4-swizzle layout and are healed in prepare_fastgs_fused_adam.
        return apply_shN_value_quant(splat);
    }

    void bind_shN_quant_for_raster(const core::SplatData& splat) {
        (void)splat;
    }

    void clear_shN_quant_for_raster() {
    }

} // namespace lfs::training::sh_value
