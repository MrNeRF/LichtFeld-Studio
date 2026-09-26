/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// POD descriptor for a RAD chunk inflated-but-still-quantized into an upload
// staging slot. The CPU decoder and both tensor backends share this layout.
// Keep it free of headers unavailable to CUDA and Slang.

#pragma once

#if defined(__SLANG__)
#define RAD_U32 uint
#define RAD_DEFAULT(...)
#else
#include <cstdint>
#define RAD_U32          std::uint32_t
#define RAD_DEFAULT(...) = __VA_ARGS__
#endif

namespace lfs::core {

    enum class RadPackedEncoding : RAD_U32 {
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

    enum class RadPackedKind : RAD_U32 {
        Means = 0, // 3 dims
        Alpha,     // 1 dim
        Sh0,       // 3 dims (display RGB; sh0 transform applies after dequant)
        Scales,    // 3 dims (linear domain; log applies after dequant)
        Rotation,  // f32/f16: 3 dims xyz (w derived); oct88r8: 3 B/splat
        Sh1,       // 9 dims  (coeffs 0-2 x 3 channels)
        Sh2,       // 15 dims (coeffs 3-7)
        Sh3,       // 21 dims (coeffs 8-14)
    };

#if defined(__SLANG__)
    static const uint kRadPackedMaxProps = 8;
#else
    inline constexpr RAD_U32 kRadPackedMaxProps = 8;
#endif

    struct RadPackedProperty {
        RAD_U32 kind RAD_DEFAULT(0);     // RadPackedKind
        RAD_U32 encoding RAD_DEFAULT(0); // RadPackedEncoding
        RAD_U32 plane_offset RAD_DEFAULT(0);
        RAD_U32 plane_bytes RAD_DEFAULT(0);
        float min_val RAD_DEFAULT(0.0f);
        float max_val RAD_DEFAULT(1.0f);
        float scale RAD_DEFAULT(1.0f);
    };

    // Dequant frame copied so kernels do not include splat_data.hpp.
    struct RadPackedDequantFrame {
        float bbox_min[3] RAD_DEFAULT({0.0f, 0.0f, 0.0f});
        float bbox_extent[3] RAD_DEFAULT({0.0f, 0.0f, 0.0f});
        float log_size_min RAD_DEFAULT(0.0f);
        float log_size_range RAD_DEFAULT(0.0f);
    };

    struct RadPagePackedDesc {
        RAD_U32 count RAD_DEFAULT(0);          // splats in the chunk
        RAD_U32 sh_coeffs_rest RAD_DEFAULT(0); // file's SH-rest coefficient count
        RAD_U32 lod_opacity RAD_DEFAULT(0);
        RAD_U32 property_count RAD_DEFAULT(0);
        RAD_U32 meta_bounds_offset RAD_DEFAULT(0); // RadMetaBoundsQ plane (8 B/node)
        RAD_U32 meta_links_offset RAD_DEFAULT(0);  // RadMetaLinksQ plane (12 B/node)
        RAD_U32 meta_node_count RAD_DEFAULT(0);
        RAD_U32 used_bytes RAD_DEFAULT(0); // total slot bytes to copy to the device
        RAD_U32 chunk RAD_DEFAULT(0);      // logical chunk index ('logical' derivation)
        RAD_U32 reserved[3] RAD_DEFAULT({});
        RadPackedDequantFrame frame RAD_DEFAULT({});
        RadPackedProperty props[kRadPackedMaxProps] RAD_DEFAULT({});
    };

} // namespace lfs::core

#undef RAD_U32
#undef RAD_DEFAULT
