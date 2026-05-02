/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/splat_data.hpp"

#include <cstddef>
#include <expected>
#include <map>
#include <string>
#include <vector>

#include "rendering/rasterizer/vksplat_fwd/src/buffer.h"

namespace lfs::vis::vksplat {

    // CPU-side reference packer for VkSplat forward inputs.
    //
    // Produces the four host buffers that the rasterizer consumes after a
    // copyToDevice. The layouts are fixed contracts checked by
    // tests/test_vksplat_input_packer.cpp:
    //
    //   xyz_ws       : (N*3)  contiguous float32 row-major copy of means_raw
    //   rotations    : (N*4)  unit-norm quaternions (rotation_raw / |rotation_raw|)
    //   scales_opacs : (N*4)  interleaved [exp(s0), exp(s1), exp(s2), sigmoid(o)]
    //   sh_coeffs    : padded to N*16*3 with sh0 in slot 0 and shN in slots 1..,
    //                  then permuted by VulkanGSPipelineBuffers::reorderSH
    //                  (SH_REORDER_SIZE=SUBGROUP_SIZE) for warp-coalesced reads.
    //
    // The buffers preserve their existing deviceBuffer fields so callers can
    // drop the result straight into VulkanGSPipelineBuffers without a reupload.
    [[nodiscard]] std::expected<void, std::string> packHostInputs(
        const lfs::core::SplatData& splat_data,
        Buffer<float>& xyz_ws,
        Buffer<float>& rotations,
        Buffer<float>& scales_opacs,
        Buffer<float>& sh_coeffs);

    // Padded SH layout produced by packHostInputs *before* reorderSH is applied.
    // Returns sh_coeffs as a freshly allocated vector with size num_splats*16*3.
    // Slot 0 is the DC (sh0) component, slots 1..rest hold shN coefficients,
    // the remainder is zero.
    [[nodiscard]] std::expected<std::vector<float>, std::string> buildPaddedShReference(
        const lfs::core::SplatData& splat_data);

    // GPU-resident packed inputs. Each tensor is contiguous Float32 on CUDA and
    // matches the host packer byte-for-byte; downstream code uploads them with
    // a single cudaMemcpyAsync(D2D) into a Vulkan-imported buffer.
    struct DevicePackedInputs {
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
    [[nodiscard]] std::expected<DevicePackedInputs, std::string> packDeviceInputs(
        const lfs::core::SplatData& splat_data);

} // namespace lfs::vis::vksplat
