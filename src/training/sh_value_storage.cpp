/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/sh_value_storage.hpp"

#include "core/cuda/sh_layout.cuh"
#include "core/sh_value_quant.hpp"
#include "lfs/training/sh_value_codec.hpp"
#include "lfs/training/sh_value_quant_kernels.hpp"

#include "core/logger.hpp"

namespace lfs::training::sh_value {

    namespace {
        using core::DataType;
        using core::Device;
        using core::Tensor;
        using core::TensorShape;

        [[nodiscard]] std::uint32_t layout_rest(const core::SplatData& splat) {
            return static_cast<std::uint32_t>(splat.max_sh_coeffs_rest());
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
        // Viewer/exportable interop is float4-layout sized; keep fp32 there.
        // Headless training (zeros_direct / cuda.direct) is the VRAM prize path.
        if (shN.is_external_storage()) {
            const auto kind = shN.external_storage_kind();
            if (kind == "vulkan_external_buffer" || kind == "splat.exportable") {
                LOG_DEBUG("SH value quant skipped: shN is exportable/viewer-backed ({})", kind);
                return false;
            }
        }

        const auto n = static_cast<std::size_t>(splat.size());
        const auto rest = layout_rest(splat);
        if (n == 0 || rest == 0)
            return false;

        const auto n_cells = core::sh_value_quant::sh_value_u16_count(n, rest);
        const auto n_bounds = core::sh_value_quant::n_bounds_for_prims(n);
        const auto cap = std::max(shN.capacity() > 0 ? shN.capacity() / std::max<std::size_t>(1, shN.numel() / n_cells)
                                                     : n,
                                  n);
        // capacity_cells based on param capacity rows
        const auto means_cap = splat.means().is_valid() ? splat.means().capacity() : n;
        const auto capacity_cells =
            core::sh_value_quant::sh_value_u16_count(std::max(means_cap, n), rest);

        Tensor u16 = Tensor::zeros_direct(TensorShape({n_cells}), capacity_cells, Device::CUDA,
                                          DataType::Float16);
        Tensor bounds = Tensor::zeros_direct(TensorShape({n_bounds * 2}), n_bounds * 2, Device::CUDA,
                                             DataType::Float32);
        u16.set_name("splat.shN");
        bounds.set_name("splat.shN_value_bounds");

        encode_shN_float4_to_u16(
            shN.ptr<float>(),
            reinterpret_cast<std::uint16_t*>(u16.data_ptr()),
            bounds.ptr<float>(),
            n,
            rest,
            nullptr);

        shN = std::move(u16);
        splat.shN_value_bounds() = std::move(bounds);
        LOG_DEBUG("SH value quant applied: N={} rest={} cells={} bounds={}",
                  n, rest, n_cells, n_bounds);
        return true;
    }

    bool ensure_shN_fp32_for_mutation(core::SplatData& splat) {
        if (!sh_value_quant_enabled())
            return false;
        auto& shN = splat.shN();
        if (!shN.is_valid() || shN.dtype() != DataType::Float16)
            return false;

        const auto n = static_cast<std::size_t>(splat.size());
        const auto rest = layout_rest(splat);
        if (n == 0 || rest == 0)
            return false;

        const auto means_cap = splat.means().is_valid() ? splat.means().capacity() : n;
        const auto logical_floats = core::sh_swizzled_float_count(n, rest);
        const auto capacity_floats = core::sh_swizzled_float_count(std::max(means_cap, n), rest);
        Tensor fp32 = Tensor::zeros_direct(TensorShape({logical_floats}), capacity_floats, Device::CUDA);
        fp32.set_name("splat.shN");

        decode_shN_u16_to_float4(
            reinterpret_cast<const std::uint16_t*>(shN.data_ptr()),
            splat.shN_value_bounds().ptr<float>(),
            fp32.ptr<float>(),
            n,
            rest,
            nullptr);

        shN = std::move(fp32);
        // keep bounds tensor for re-encode
        return true;
    }

    bool commit_shN_after_mutation(core::SplatData& splat) {
        if (!sh_value_quant_enabled())
            return false;
        auto& shN = splat.shN();
        if (!shN.is_valid() || shN.dtype() != DataType::Float32)
            return false;
        // Re-apply quant from current float storage.
        // Clear Float16 check by temporarily... apply_shN_value_quant refuses Float16 only.
        return apply_shN_value_quant(splat);
    }

    void bind_shN_quant_for_raster(const core::SplatData& splat) {
        // Bounds are passed explicitly through forward_raw / fused Adam (no device
        // constant table). Keep as no-op so densify bridges stay the only materialize path.
        (void)splat;
    }

    void clear_shN_quant_for_raster() {
    }

} // namespace lfs::training::sh_value
