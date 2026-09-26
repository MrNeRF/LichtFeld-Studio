/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace lfs::core {

    // SH coefficient swizzle layout (vksplat float4 packing, ported from
    // vksplat/vksplat/slang/spherical_harmonics.slang).
    //
    // Canonical layout (input/export):
    //     [N, K, 3] row-major, K = SH-rest coefficient count (0 / 3 / 8 / 15 for SH 0-3).
    //
    // Swizzled layout (resident GPU storage):
    //     [ceil(N/R), slots_for_layout_rest, R] of float4, where R = kShReorderSize.
    //     For a warp of R lanes the float4 reads/writes coalesce into a single 16-byte vector
    //     load per coefficient slot — fixing the stride-12 misaligned loads of the old float3
    //     layout.
    //
    // Packing of 15 float3 coefficients (c0..c14) into 12 float4 slots, identical to vksplat
    // but with the DC term (which we keep in a separate sh0 tensor) replaced by tail padding:
    //
    //     slot 0  : (c0.x, c0.y, c0.z, c1.x)         slot 6  : (c8.x,  c8.y,  c8.z,  c9.x)
    //     slot 1  : (c1.y, c1.z, c2.x, c2.y)         slot 7  : (c9.y,  c9.z,  c10.x, c10.y)
    //     slot 2  : (c2.z, c3.x, c3.y, c3.z)         slot 8  : (c10.z, c11.x, c11.y, c11.z)
    //     slot 3  : (c4.x, c4.y, c4.z, c5.x)         slot 9  : (c12.x, c12.y, c12.z, c13.x)
    //     slot 4  : (c5.y, c5.z, c6.x, c6.y)         slot 10 : (c13.y, c13.z, c14.x, c14.y)
    //     slot 5  : (c6.z, c7.x, c7.y, c7.z)         slot 11 : (c14.z, 0,     0,     0    )
    //
    // shAt(p, k) returns the float4 slot index (multiply by 4 to get the float offset).
    //
    //     shAt(p, k, layout_rest) = (p / R) * (slots_for_layout_rest * R) + k * R + (p % R)
    //
    // Storage is compact for the selected/max SH degree:
    //     degree 0 -> 0 slots, degree 1 -> 3 slots, degree 2 -> 6 slots, degree 3 -> 12 slots.
    // Active degree controls which coefficients are used/updated, not how wide resident storage is.

    inline constexpr std::uint32_t kShReorderSize = 32u;
    inline constexpr std::uint32_t kShMaxCoeffsRest = 15u;          // canonical float3 coeff count (shN)
    inline constexpr std::uint32_t kShRestFloat4PerPrimitive = 12u; // packed float4 slot count per primitive
    inline constexpr std::uint32_t kShChannels = 3u;                // canonical layout channel count

    [[nodiscard]] inline constexpr std::uint32_t sh_rest_coefficients_for_degree(int degree) noexcept {
        if (degree <= 0) {
            return 0u;
        }
        const auto d = std::min<std::uint32_t>(static_cast<std::uint32_t>(degree), 3u);
        return std::min<std::uint32_t>(kShMaxCoeffsRest, (d + 1u) * (d + 1u) - 1u);
    }

    [[nodiscard]] inline constexpr int sh_degree_for_rest_coefficients(std::uint32_t active_coeffs_rest) noexcept {
        if (active_coeffs_rest >= 15u)
            return 3;
        if (active_coeffs_rest >= 8u)
            return 2;
        if (active_coeffs_rest >= 3u)
            return 1;
        return 0;
    }

    // Number of primitives in the swizzled buffer (rounded up to multiple of kShReorderSize).
    [[nodiscard]] inline constexpr std::size_t sh_swizzled_block_count(std::size_t n) noexcept {
        return (n + kShReorderSize - 1) / kShReorderSize;
    }

    [[nodiscard]] inline constexpr std::size_t sh_swizzled_padded_n(std::size_t n) noexcept {
        return sh_swizzled_block_count(n) * kShReorderSize;
    }

    [[nodiscard]] inline constexpr std::uint32_t sh_float4_slots_for_rest(std::uint32_t coeffs_rest) noexcept {
        return coeffs_rest == 0u
                   ? 0u
                   : std::min<std::uint32_t>(
                         kShRestFloat4PerPrimitive,
                         (std::min<std::uint32_t>(coeffs_rest, kShMaxCoeffsRest) * kShChannels + 3u) / 4u);
    }

    // Total float count in the compact swizzled SH buffer for n primitives and layout rest coeffs.
    // (ceil(n/R) * slots_for_layout_rest * R * 4) — packed float4 layout.
    // The DC term (sh0) is stored in a separate tensor; this buffer only holds shN-rest.
    [[nodiscard]] inline constexpr std::size_t sh_swizzled_float_count(
        std::size_t n,
        std::uint32_t coeffs_rest) noexcept {
        return sh_swizzled_block_count(n) *
               static_cast<std::size_t>(sh_float4_slots_for_rest(coeffs_rest)) *
               kShReorderSize * 4u;
    }

    // Full degree-3 layout helper used by VkSplat export paths.
    [[nodiscard]] inline constexpr std::size_t sh_swizzled_float_count(std::size_t n) noexcept {
        return sh_swizzled_float_count(n, kShMaxCoeffsRest);
    }

    // Total float4 slot count in the swizzled SH buffer for n primitives.
    [[nodiscard]] inline constexpr std::size_t sh_swizzled_float4_count(
        std::size_t n,
        std::uint32_t coeffs_rest) noexcept {
        return sh_swizzled_block_count(n) *
               static_cast<std::size_t>(sh_float4_slots_for_rest(coeffs_rest)) *
               kShReorderSize;
    }

    [[nodiscard]] inline constexpr std::size_t sh_swizzled_float4_count(std::size_t n) noexcept {
        return sh_swizzled_float4_count(n, kShMaxCoeffsRest);
    }

    [[nodiscard]] inline constexpr std::size_t sh_swizzled_byte_count(
        std::size_t n,
        std::uint32_t coeffs_rest) noexcept {
        return sh_swizzled_float_count(n, coeffs_rest) * sizeof(float);
    }

    [[nodiscard]] inline constexpr std::size_t sh_swizzled_byte_count(std::size_t n) noexcept {
        return sh_swizzled_byte_count(n, kShMaxCoeffsRest);
    }

    // IEEE f16 float4-swizzle (same topology as fp32, 2 bytes per component).
    // Used by the GUI exportable/viewer path to cut resident SH ~in half vs fp32.
    [[nodiscard]] inline constexpr std::size_t sh_swizzled_f16_byte_count(
        std::size_t n,
        std::uint32_t coeffs_rest) noexcept {
        return sh_swizzled_float_count(n, coeffs_rest) * sizeof(std::uint16_t);
    }

    [[nodiscard]] inline constexpr std::size_t sh_swizzled_f16_byte_count(std::size_t n) noexcept {
        return sh_swizzled_f16_byte_count(n, kShMaxCoeffsRest);
    }

    // Element size for resident SH rest storage: 4 = fp32, 2 = IEEE f16.
    [[nodiscard]] inline constexpr std::size_t sh_swizzled_byte_count_for_element(
        std::size_t n,
        std::uint32_t coeffs_rest,
        std::size_t element_bytes) noexcept {
        return sh_swizzled_float_count(n, coeffs_rest) * element_bytes;
    }

    // Host index helper. Returns the float4-slot index (multiply by 4 to get the float offset).
    [[nodiscard]] inline std::uint32_t sh_swizzled_index(
        std::uint32_t primitive_idx,
        std::uint32_t float4_slot,
        std::uint32_t layout_coeffs_rest) noexcept {
        const std::uint32_t block = primitive_idx / kShReorderSize;
        const std::uint32_t lane = primitive_idx % kShReorderSize;
        const std::uint32_t slots = sh_float4_slots_for_rest(layout_coeffs_rest);
        return block * (slots * kShReorderSize) + float4_slot * kShReorderSize + lane;
    }

    [[nodiscard]] inline std::uint32_t sh_swizzled_index(std::uint32_t primitive_idx, std::uint32_t float4_slot) noexcept {
        return sh_swizzled_index(primitive_idx, float4_slot, kShMaxCoeffsRest);
    }

    // Reorder canonical [N, K, 3] (K = active_coeffs_rest, contiguous, row-major) into the

} // namespace lfs::core
