/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/cuda/sh_layout.cuh"
#include "core/splat_data.hpp"

#include <cstddef>
#include <cuda_runtime_api.h>
#include <expected>
#include <map>
#include <string>
#include <vector>

#include "rendering/rasterizer/vulkan/src/buffer.h"

#include "visualizer/visualizer.hpp"

namespace lfs::vis::vksplat {

    // CPU-side reference packer for VkSplat forward inputs (tests / offline).
    //
    // The LIVE training viewport does NOT use this packer: prepareInputs
    // zero-copies exportable raw tensors (means/rot/scale/opacity/sh) via
    // a Vulkan-external borrow, avoiding a separate 44 B/splat packed non-SH
    // residency of approximately 210 MiB at 5M splats.
    //
    // Host packer layouts (tests/test_vksplat_input_packer.cpp):
    //   xyz_ws       : (N*3)  contiguous float32 row-major copy of means_raw
    //   rotations    : (N*4)  unit-norm quaternions (rotation_raw / |rotation_raw|)
    //   scales_opacs : (N*4)  interleaved [exp(s0), exp(s1), exp(s2), sigmoid(o)]
    //   sh_coeffs    : padded to N*16*3 with sh0 in slot 0 and shN in slots 1..,
    //                  then permuted by VulkanGSPipelineBuffers::reorderSH
    //                  (SH_REORDER_SIZE=SUBGROUP_SIZE) for warp-coalesced reads.
    //
    // The buffers preserve their existing deviceBuffer fields so callers can
    // drop the result straight into VulkanGSPipelineBuffers without a reupload.
    LFS_VIS_API [[nodiscard]] std::expected<void, std::string> packHostInputs(
        const lfs::core::SplatData& splat_data,
        Buffer<float>& xyz_ws,
        Buffer<float>& rotations,
        Buffer<float>& scales_opacs,
        Buffer<float>& sh_coeffs);

    // Padded SH layout produced by packHostInputs *before* reorderSH is applied.
    // Returns sh_coeffs as a freshly allocated vector with size num_splats*16*3.
    // Slot 0 is the DC (sh0) component, slots 1..rest hold shN coefficients,
    // the remainder is zero.
    LFS_VIS_API [[nodiscard]] std::expected<std::vector<float>, std::string> buildPaddedShReference(
        const lfs::core::SplatData& splat_data);

    // GPU-resident packed inputs. Each tensor is contiguous Float32 on CUDA and
    // matches the host packer byte-for-byte; downstream code uploads them with
    // a single cudaMemcpyAsync(D2D) into a Vulkan-imported buffer.
    struct LFS_VIS_API DevicePackedInputs {
        lfs::core::Tensor xyz_ws;       // [N, 3]
        lfs::core::Tensor rotations;    // [N, 4]
        lfs::core::Tensor scales_opacs; // [N, 4]
        lfs::core::Tensor sh_coeffs;    // padded ceil(N/SH_REORDER_SIZE)*SH_REORDER_SIZE*16*3 floats, reordered
        std::size_t num_splats = 0;
        std::size_t sh_padded_floats = 0;
    };

    // GPU-only packer. Uses the tensor library to compose activations, padding,
    // and the SH reorder via permute+contiguous. Produces output tensors whose
    // raw byte layout matches packHostInputs's host buffers exactly.
    LFS_VIS_API [[nodiscard]] std::expected<DevicePackedInputs, std::string> packDeviceInputs(
        const lfs::core::SplatData& splat_data);

    struct LFS_VIS_API DeviceInputLayout {
        std::size_t num_splats = 0;
        std::size_t xyz_bytes = 0;
        std::size_t rotations_bytes = 0;
        std::size_t scales_opacs_bytes = 0;
        std::size_t sh_coeffs_bytes = 0;
        std::size_t sh_padded_floats = 0;
    };

    // Zero-intermediate GPU packer for the live Vulkan viewer path. The caller
    // supplies CUDA pointers to a Vulkan-imported buffer; this copies means and
    // writes activated rotation/scale/opacity/SH directly into those regions.
    LFS_VIS_API [[nodiscard]] std::expected<DeviceInputLayout, std::string> deviceInputLayout(
        const lfs::core::SplatData& splat_data);

    struct LFS_VIS_API RawDeviceInputLayout {
        std::size_t num_splats = 0;
        std::size_t xyz_bytes = 0;
        std::size_t sh0_bytes = 0;
        std::size_t shN_bytes = 0;
        std::size_t rotations_bytes = 0;
        std::size_t scaling_bytes = 0;
        std::size_t opacity_bytes = 0;
        std::uint32_t shN_layout_rest = 0;
        bool omits_shN = false;
        // True when resident shN is IEEE f16 float4-swizzle (2 B/component).
        // False = fp32 float4-swizzle (or omit placeholder). Mutually exclusive
        // with shN_q16.
        bool shN_f16 = false;
        // True when resident shN is pad-dropped q16 (uint16 cells + float2
        // bounds / 256). Training exportable zero-copy path.
        bool shN_q16 = false;
        // Bytes per shN component element (4=fp32, 2=f16/q16).
        std::size_t shN_element_bytes = sizeof(float);
        // q16 only: number of pad-dropped u16 cells per primitive.
        std::uint32_t shN_n_cells = 0;
        // q16 only: bytes of the per-256 float2 bounds buffer.
        std::size_t shN_bounds_bytes = 0;
        // Non-SH display attrs (rotation + log-scale + logit opacity) stored as
        // IEEE f16 for the exportable/viewer path. Means/xyz stay fp32 (large
        // scenes shimmer under f16 positions). Matches lodq pool packing:
        //   rotation f16x4 = 8 B, scaling f16x3+pad = 8 B, opacity f16 = 2 B.
        bool attrs_f16 = false;
        // Sum of xyz+rotations+scaling+opacity bytes (before/after f16 accounting).
        std::size_t non_sh_bytes = 0;
    };

    // Raw split SplatData layout for the live Vulkan viewer. Unlike the packed
    // path above, this keeps log-scale/logit opacity and split SH untouched so
    // shaders can consume the training tensors directly when they are Vulkan
    // external buffers.
    //  - shN_value_quantized → pad-dropped q16 (+ bounds bytes)
    //  - shN_ieee_f16        → IEEE f16 float4-swizzle (standalone PLY/SOG)
    //  - else                → fp32 float4-swizzle
    LFS_VIS_API [[nodiscard]] std::expected<RawDeviceInputLayout, std::string> rawDeviceInputLayout(
        const lfs::core::SplatData& splat_data,
        int upload_sh_degree = -1);

    // Copy just raw opacity, baking SplatData::deleted() into the destination
    // when present. This lets the live renderer borrow all other raw tensors
    // directly instead of allocating a full raw-model copy only to honor deletes.
    LFS_VIS_API [[nodiscard]] std::expected<void, std::string> copyRawOpacityToBuffer(
        const lfs::core::SplatData& splat_data,
        void* opacity_dst,
        cudaStream_t stream);

} // namespace lfs::vis::vksplat
