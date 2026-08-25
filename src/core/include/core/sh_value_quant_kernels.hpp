/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace lfs::core::sh_value_quant {

    /// Encode float4-swizzled SH-rest into pad-dropped u16 + float2 bounds / 256.
    void encode_shN_float4_to_u16(
        const float* src_float4_swizzled,
        std::uint16_t* dst_u16,
        float* bounds_float2,
        std::size_t n_primitives,
        std::uint32_t coeffs_rest,
        cudaStream_t stream = nullptr);

    /// Decode u16 + bounds into float4-swizzled (zeros float4 tail pad).
    void decode_shN_u16_to_float4(
        const std::uint16_t* src_u16,
        const float* bounds_float2,
        float* dst_float4_swizzled,
        std::size_t n_primitives,
        std::uint32_t coeffs_rest,
        cudaStream_t stream = nullptr);

    void decode_shN_u16_range_to_canonical(
        const std::uint16_t* src_u16,
        const float* bounds_float2,
        float* dst_canonical,
        std::uint64_t canonical_float_offset,
        std::uint64_t float_count,
        std::size_t n_primitives,
        std::uint32_t dst_coeffs_rest,
        std::uint32_t layout_coeffs_rest,
        cudaStream_t stream = nullptr);

    void decode_shN_f16_range_to_canonical(
        const std::uint16_t* src_f16,
        float* dst_canonical,
        std::uint64_t canonical_float_offset,
        std::uint64_t float_count,
        std::size_t n_primitives,
        std::uint32_t dst_coeffs_rest,
        std::uint32_t layout_coeffs_rest,
        cudaStream_t stream = nullptr);

} // namespace lfs::core::sh_value_quant
