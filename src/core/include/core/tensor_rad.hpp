/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/rad_packed_page.hpp"
#include "core/rad_pool_quant.hpp"
#include "core/tensor.hpp"
#include <array>

namespace lfs::core {
    // Byte tensor views in order: xyz, sh0, shN, rotation, log scale, opacity,
    // page frames, metadata bounds, metadata links. Metadata is optional.
    struct RadPagePool {
        std::array<Tensor, 9> regions;
        uint32_t page_splats = 65536;
        uint32_t sh_slots = 0;
    };
    struct RadPageSources {
        Tensor means, sh0, shN, rotation, scaling, opacity, shN_bounds;
        uint32_t offset = 0, count = 0, sh_rest = 0;
        bool sh_q16 = false;
    };
    // Worst-case degree-three f32 planes: xyz + alpha + rgb + scale +
    // rotation xyz + SH-rest (3 + 1 + 3 + 3 + 3 + 45) floats per splat.
    inline constexpr size_t kRadMaxPayloadBytesPerSplat = (3 + 1 + 3 + 3 + 3 + 45) * sizeof(float);
    inline constexpr size_t kRadMetadataBytesPerSplat = radq::kMetaBoundsBytes + radq::kMetaLinksBytes;
    inline constexpr size_t kRadPlaneAlignment = 16;
    inline constexpr size_t rad_page_staging_bytes(size_t splats) {
        return splats * (kRadMaxPayloadBytesPerSplat + kRadMetadataBytesPerSplat) +
               kRadPlaneAlignment * (kRadPackedMaxProps + 2);
    }
    // Enqueue on the current tensor stream/recorder. The packed tensor starts
    // with a RadPagePackedDesc followed by its planes; all layout fields come
    // from the shared POD rather than manually numbered descriptor words.
    LFS_CORE_API void rad_page_validate(const RadPagePool& pool, const RadPagePackedDesc& desc,
                                        size_t packed_bytes, uint32_t page);
    LFS_CORE_API void rad_page_dequant(const Tensor& packed, const RadPagePackedDesc& desc,
                                       const RadPagePool& pool, uint32_t page);
    LFS_CORE_API void rad_page_quantize(const RadPageSources& source, const RadPagePool& pool, uint32_t page);
} // namespace lfs::core
