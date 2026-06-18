/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "io/formats/rad_packed_page.hpp"
#include "lod_pool_quant.hpp"

#include <cstdint>
#include <cuda_runtime.h>

namespace lfs::vis {

    // Device-pointer view of the canonical quantized page-pool regions (see
    // lod_pool_quant.hpp). Pointers are pre-offset region bases inside the
    // CUDA-imported pool buffers; indexing is by global pool splat
    // (page * page_splats + i).
    struct LodPoolDeviceView {
        float* means = nullptr;       // f32x3
        uint2* sh0 = nullptr;         // f16x3 + pad
        std::uint32_t* shN = nullptr; // s8x4 per swizzled float4 slot
        uint2* rotation = nullptr;    // f16x4, (w,x,y,z)
        uint2* scaling = nullptr;     // log-domain f16x3 + pad
        std::uint16_t* opacity = nullptr;
        float4* page_frames = nullptr;       // lodq::PageFrame per page
        uint2* meta_bounds = nullptr;        // RadMetaBoundsQ passthrough
        std::uint32_t* meta_links = nullptr; // RadMetaLinksQ, 3 words/node
        std::uint32_t dst_rest = 0;
        std::uint32_t dst_slots = 0;
    };

    // Dequantizes one packed page (planes staged by decode_rad_chunk_packed,
    // already copied to `device_slot`) into the canonical pool regions and
    // expands the sidecar bounds/links planes. s8 SH bands and log-f16 scales
    // pass through bit-exact; everything else lands as f16 — no new loss over
    // the file. Launched on the upload engine's stream between the slot copy
    // and the timeline signal.
    LFS_VIS_API cudaError_t launchLodPageDequant(const std::uint8_t* device_slot,
                                                 const lfs::io::RadPagePackedDesc& desc,
                                                 const LodPoolDeviceView& pool,
                                                 std::uint32_t page,
                                                 std::uint32_t page_splats,
                                                 cudaStream_t stream);

    // fp32 source tensors for the in-core / pinned-root D2D fill path.
    // Pointers are pre-offset to the page's first splat; layouts match the
    // resident SplatData tensors (means [n,3], sh0_raw [n,3] post-transform,
    // shN canonical [n, rest*3], rotation_raw [n,4] (w,x,y,z), scaling_raw
    // [n,3] log domain, opacity_raw [n]).
    struct LodPageTensorSources {
        const float* means = nullptr;
        const float* sh0 = nullptr;
        const float* shN = nullptr; // null when the model has no SH rest
        const float* rotation = nullptr;
        const float* scaling = nullptr;
        const float* opacity = nullptr;
        std::uint32_t src_rest = 0; // SH-rest coefficients in the shN tensor
        std::uint32_t count = 0;    // splats in this page
        std::uint32_t lod_opacity = 0;
    };

    // Quantizes resident fp32 tensors into one canonical pool page (two
    // passes: per-band |max| reduction into the page frame, then the
    // quantizing scatter). Same canonical writers as the streamed path.
    LFS_VIS_API cudaError_t launchLodPageQuantizeFromTensors(const LodPageTensorSources& src,
                                                             const LodPoolDeviceView& pool,
                                                             std::uint32_t page,
                                                             std::uint32_t page_splats,
                                                             cudaStream_t stream);

} // namespace lfs::vis
