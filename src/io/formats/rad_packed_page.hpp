/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// POD descriptor for a RAD chunk inflated-but-still-quantized into an upload
// staging slot. Produced by decode_rad_chunk_packed (CPU, rad.cpp), consumed
// by the page-dequant CUDA kernel as a by-value launch argument. Kept free of
// any non-CUDA-safe includes: this header is shared with .cu translation
// units (rad.hpp itself uses std::expected, which nvcc's C++20 cannot see).

#pragma once

#include <cstdint>

namespace lfs::io {

    enum class RadPackedEncoding : std::uint32_t {
        F32 = 0,
        F32LeBytes,
        F16,
        F16LeBytes,
        R8, // r8_delta is undeltaed on the CPU and arrives here as plain R8
        S8, // likewise s8_delta
        Ln0R8,
        LnF16,
        Oct88R8, // 3 bytes per splat, splat-major (not dimension-major)
    };

    enum class RadPackedKind : std::uint32_t {
        Means = 0, // 3 dims
        Alpha,     // 1 dim
        Sh0,       // 3 dims (display RGB; sh0 transform applies after dequant)
        Scales,    // 3 dims (linear domain; log applies after dequant)
        Rotation,  // f32/f16: 3 dims xyz (w derived); oct88r8: 3 B/splat
        Sh1,       // 9 dims  (coeffs 0-2 x 3 channels)
        Sh2,       // 15 dims (coeffs 3-7)
        Sh3,       // 21 dims (coeffs 8-14)
    };

    inline constexpr std::uint32_t kRadPackedMaxProps = 8;

    struct RadPackedProperty {
        std::uint32_t kind = 0;     // RadPackedKind
        std::uint32_t encoding = 0; // RadPackedEncoding
        std::uint32_t plane_offset = 0;
        std::uint32_t plane_bytes = 0;
        float min_val = 0.0f;
        float max_val = 1.0f;
        float base = 0.0f;
        float scale = 1.0f;
    };

    // Mirror of lfs::core::RadMetaChunkRecord's dequant frame (the kernel must
    // not include splat_data.hpp); layout pinned by static_asserts in rad.cpp.
    struct RadPackedDequantFrame {
        float bbox_min[3] = {0.0f, 0.0f, 0.0f};
        float bbox_extent[3] = {0.0f, 0.0f, 0.0f};
        float log_size_min = 0.0f;
        float log_size_range = 0.0f;
    };

    struct RadPagePackedDesc {
        std::uint32_t count = 0;          // splats in the chunk
        std::uint32_t sh_coeffs_rest = 0; // file's SH-rest coefficient count
        std::uint32_t lod_opacity = 0;
        std::uint32_t property_count = 0;
        std::uint32_t meta_bounds_offset = 0; // RadMetaBoundsQ plane (8 B/node)
        std::uint32_t meta_links_offset = 0;  // RadMetaLinksQ plane (12 B/node)
        std::uint32_t meta_node_count = 0;
        std::uint32_t used_bytes = 0; // total slot bytes to copy to the device
        std::uint32_t chunk = 0;      // logical chunk index ('logical' derivation)
        std::uint32_t reserved[3] = {};
        RadPackedDequantFrame frame{};
        RadPackedProperty props[kRadPackedMaxProps]{};
    };

} // namespace lfs::io
