/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vksplat_input_packer.hpp"

#include "core/sh_value_quant.hpp"

#include <algorithm>
#include <cstdint>

namespace lfs::vis::vksplat {
    namespace {
        using lfs::core::DataType;
        using lfs::core::Tensor;

        [[nodiscard]] bool validOpacityShape(const Tensor& opacity, const std::size_t n) {
            return opacity.ndim() != 0 &&
                   ((opacity.ndim() == 1 && opacity.size(0) == n) ||
                    (opacity.ndim() == 2 && opacity.size(0) == n && opacity.size(1) == 1));
        }

        [[nodiscard]] bool validSh0Shape(const Tensor& sh0, const std::size_t n) {
            if (!sh0.is_valid() || sh0.numel() == 0) {
                return false;
            }
            const std::size_t dims = sh0.ndim();
            if (dims != 2 && dims != 3) {
                return false;
            }
            return sh0.size(0) == n &&
                   sh0.size(dims - 1) == 3 &&
                   (dims == 2 || sh0.size(1) == 1);
        }
    } // namespace

    std::expected<RawDeviceInputLayout, std::string> rawDeviceInputLayout(
        const lfs::core::SplatData& splat_data,
        const int upload_sh_degree) {
        const std::size_t n = static_cast<std::size_t>(splat_data.size());
        if (n == 0) {
            return std::unexpected("VkSplat cannot render an empty model");
        }

        const Tensor& means_raw = splat_data.means_raw();
        const Tensor& rotation_raw = splat_data.rotation_raw();
        const Tensor& scaling_raw = splat_data.scaling_raw();
        const Tensor& opacity_raw = splat_data.opacity_raw();
        const Tensor& sh0_raw = splat_data.sh0_raw();
        const Tensor& shN_raw = splat_data.shN_raw();
        if (means_raw.ndim() != 2 || means_raw.size(0) != n || means_raw.size(1) != 3 ||
            rotation_raw.ndim() != 2 || rotation_raw.size(0) != n || rotation_raw.size(1) != 4 ||
            scaling_raw.ndim() != 2 || scaling_raw.size(0) != n || scaling_raw.size(1) != 3) {
            return std::unexpected("VkSplat raw input tensor shapes do not match [N,3]/[N,4]/[N,3]");
        }
        if (!validOpacityShape(opacity_raw, n)) {
            return std::unexpected("VkSplat opacity tensor must be [N] or [N, 1]");
        }
        if (!validSh0Shape(sh0_raw, n)) {
            return std::unexpected("VkSplat expected SH DC coefficients shaped [N, 1, 3] or [N, 3]");
        }
        const auto layout_rest = static_cast<std::uint32_t>(splat_data.max_sh_coeffs_rest());
        const int effective_upload_sh_degree =
            upload_sh_degree < 0
                ? splat_data.get_max_sh_degree()
                : std::clamp(upload_sh_degree, 0, splat_data.get_max_sh_degree());
        const auto upload_layout_rest = std::min<std::uint32_t>(
            layout_rest,
            lfs::core::sh_rest_coefficients_for_degree(effective_upload_sh_degree));
        const bool omit_shN_upload = upload_layout_rest == 0;
        // The raw split path accepts three resident SH formats:
        // - pad-dropped q16 with u16 cells and float2 bounds
        // - IEEE f16 float4-swizzle with the same topology as fp32
        // - fp32 float4-swizzle
        const bool shN_q16 = splat_data.shN_value_quantized() && !omit_shN_upload;
        const bool shN_f16 = splat_data.shN_ieee_f16() && !omit_shN_upload && !shN_q16;
        const std::uint32_t n_cells =
            shN_q16 ? lfs::core::sh_value_quant::n_value_cells_per_prim(upload_layout_rest) : 0u;
        std::size_t shN_bytes = 4 * sizeof(float);
        std::size_t element_bytes = sizeof(float);
        std::size_t bounds_bytes = 0;
        if (!omit_shN_upload) {
            if (shN_q16) {
                element_bytes = sizeof(std::uint16_t);
                shN_bytes = lfs::core::sh_value_quant::sh_value_u16_count(n, upload_layout_rest) *
                            sizeof(std::uint16_t);
                bounds_bytes = lfs::core::sh_value_quant::n_bounds_for_prims(n) * 2u * sizeof(float);
            } else {
                element_bytes = shN_f16 ? sizeof(std::uint16_t) : sizeof(float);
                shN_bytes = lfs::core::sh_swizzled_byte_count_for_element(
                    n, upload_layout_rest, element_bytes);
            }
        }
        if (!omit_shN_upload && shN_raw.is_valid() && shN_raw.numel() > 0) {
            if (shN_raw.ndim() != 1) {
                return std::unexpected("VkSplat expected swizzled SH rest coefficients as a 1D tensor");
            }
            const std::size_t src_elem =
                shN_raw.dtype() == DataType::Float16 ? sizeof(std::uint16_t) : sizeof(float);
            const std::size_t resident_bytes =
                shN_q16
                    ? lfs::core::sh_value_quant::sh_value_u16_count(n, layout_rest) *
                          sizeof(std::uint16_t)
                    : lfs::core::sh_swizzled_byte_count_for_element(n, layout_rest, element_bytes);
            if (static_cast<std::size_t>(shN_raw.numel()) * src_elem < resident_bytes) {
                return std::unexpected("VkSplat swizzled SH rest tensor is smaller than expected");
            }
            if (shN_q16) {
                const auto& bounds = splat_data.shN_value_bounds();
                const std::size_t need_bounds =
                    lfs::core::sh_value_quant::n_bounds_for_prims(n) * 2u;
                if (!bounds.is_valid() || static_cast<std::size_t>(bounds.numel()) < need_bounds) {
                    return std::unexpected(
                        "VkSplat q16 SH rest requires per-256 float2 bounds");
                }
            }
        } else if (!omit_shN_upload && splat_data.max_sh_coeffs_rest() > 0) {
            return std::unexpected("VkSplat expected swizzled SH rest coefficients for max SH degree");
        }

        // Non-SH display attrs: IEEE f16 when exportable/allocator forces half
        // for rotation + scaling + opacity (means stay fp32). Byte sizes match
        // lodq pool packing so projection can reuse the quant-style f16tof32
        // loads (rot uint2, scale uint2 with pad, opacity packed halfs).
        const bool attrs_f16 = splat_data.non_sh_attrs_f16();
        const std::size_t rotations_bytes =
            attrs_f16 ? n * 8u : n * 4u * sizeof(float);
        const std::size_t scaling_bytes =
            attrs_f16 ? n * 8u : n * 3u * sizeof(float);
        const std::size_t opacity_bytes =
            attrs_f16 ? n * 2u : n * sizeof(float);
        const std::size_t xyz_bytes = n * 3u * sizeof(float);
        return RawDeviceInputLayout{
            .num_splats = n,
            .xyz_bytes = xyz_bytes,
            .sh0_bytes = n * 3 * sizeof(float),
            .shN_bytes = shN_bytes,
            .rotations_bytes = rotations_bytes,
            .scaling_bytes = scaling_bytes,
            .opacity_bytes = opacity_bytes,
            .shN_layout_rest = upload_layout_rest,
            .omits_shN = omit_shN_upload,
            .shN_f16 = shN_f16,
            .shN_q16 = shN_q16,
            .shN_element_bytes = omit_shN_upload ? sizeof(float) : element_bytes,
            .shN_n_cells = n_cells,
            .shN_bounds_bytes = bounds_bytes,
            .attrs_f16 = attrs_f16,
            .non_sh_bytes = xyz_bytes + rotations_bytes + scaling_bytes + opacity_bytes,
        };
    }

} // namespace lfs::vis::vksplat
