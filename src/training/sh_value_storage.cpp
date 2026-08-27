/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/sh_value_storage.hpp"

#include "core/cuda/sh_layout.cuh"
#include "core/cuda_error.hpp"
#include "core/logger.hpp"
#include "core/sh_value_quant.hpp"
#include "core/sh_value_quant_kernels.hpp"
#include "core/splat_exportable_storage.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "lfs/training/live_model_mutation_guard.hpp"
#include "lfs/training/sh_value_codec.hpp"

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
        // the training stream without an intervening wait (multi-stream UAF).
        void sync_codec_stream(cudaStream_t /*stream*/) {
            LFS_CUDA_CHECK_MSG(cudaDeviceSynchronize(), "sh_value quant codec device barrier");
        }
    } // namespace

    bool apply_shN_value_quant(core::SplatData& splat) {
        return splat.apply_shN_value_quant();
    }

    bool ensure_shN_fp32_for_mutation(core::SplatData& splat) {
        LFS_ASSERT_LIVE_MODEL_MUTATION_LOCK_HELD();
        LiveModelMutationGuard mutation_guard("ensure_shN_fp32_for_mutation");
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
            // Rebuilding empty or zero bounds makes decode emit all zeros — a silent
            // SH wipe. Fail loud so densify/relocate never zero SH-rest by accident.
            LOG_ERROR("SH value quant expand refused: bounds are short for N={} "
                      "(have={} need={}). Refusing silent SH wipe; restore bounds or "
                      "dequant via a known-good checkpoint.",
                      n,
                      bounds.is_valid() ? bounds.numel() : 0,
                      core::sh_value_quant::n_bounds_for_prims(n) * 2);
            throw std::runtime_error(
                "ensure_shN_fp32_for_mutation: shN_value_bounds short/missing — "
                "refusing silent SH wipe");
        }
        if (bounds.stream() != stream)
            bounds.set_stream(stream);

        lfs::core::sh_value_quant::decode_shN_u16_to_float4(
            reinterpret_cast<const std::uint16_t*>(
                lfs::core::resolve_exportable_device_ptr(shN)),
            static_cast<const float*>(
                lfs::core::resolve_exportable_device_ptr(bounds)),
            fp32.ptr<float>(),
            n,
            rest,
            stream);
        sync_codec_stream(stream);

        shN = std::move(fp32);
        // drop bounds once codes are expanded. Leaving pad-dropped bounds
        // attached to a Float32 float4-swizzle buffer is a dual-representation
        // footgun (FastGS/viewer may treat Float16+bounds as q16; a later
        // partial rebind can re-install codes without matching bounds).
        splat.shN_value_bounds() = Tensor{};
        return true;
    }

    bool commit_shN_after_mutation(core::SplatData& splat) {
        LFS_ASSERT_LIVE_MODEL_MUTATION_LOCK_HELD();
        LiveModelMutationGuard mutation_guard("commit_shN_after_mutation");
        auto& shN = splat.shN();
        if (!shN.is_valid() || shN.dtype() != DataType::Float32)
            return false;

        if (!sh_value_quant_enabled())
            return false;
        // Single-buffer: rebuild codes+bounds into the live exportable q16 region
        // (allocate_named_param). The caller must already own the mutation guard or
        // trainer render exclusive; this helper's marker only covers nested work.
        return splat.apply_shN_value_quant();
    }

    ShNCommitGuard::~ShNCommitGuard() noexcept {
        if (!expanded_ || !splat_) {
            return;
        }
        try {
            (void)commit_shN_after_mutation(*splat_);
        } catch (const std::exception& e) {
            try {
                LOG_ERROR("{}: SH value commit failed during scope exit: {}", site_, e.what());
            } catch (...) {
            }
        } catch (...) {
            try {
                LOG_ERROR("{}: SH value commit failed during scope exit with unknown exception", site_);
            } catch (...) {
            }
        }
    }

} // namespace lfs::training::sh_value
