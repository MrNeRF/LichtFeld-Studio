/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/tensor_image.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace lfs::core::internal {
    // Scalar layout shared with image_warp.slang. A mode bit carries the input
    // type without changing the merged warp block's 56-byte ABI.
    constexpr int32_t IMAGE_WARP_UINT8 = 0x100;
    struct ImageWarpCameraParams {
        float src_fx, src_fy, src_cx, src_cy, dst_fx, dst_fy, dst_cx, dst_cy;
        int32_t src_width, src_height, dst_width, dst_height, model_type;
        float distortion[12];
        int32_t num_distortion;
    };
    struct ImageWarpDispatchParams {
        uint64_t input = 0, output = 0, validity = 0, camera = 0;
        int32_t width, height, channels, mode, inverse, quadrature;
    };
    static_assert(sizeof(ImageWarpCameraParams) == 104);
    static_assert(offsetof(ImageWarpCameraParams, distortion) == 52);
    static_assert(offsetof(ImageWarpCameraParams, num_distortion) == 100);
    static_assert(sizeof(ImageWarpDispatchParams) == 56);
    static_assert(offsetof(ImageWarpDispatchParams, width) == 32);
    static_assert(offsetof(ImageWarpDispatchParams, mode) == 44);
    static_assert(offsetof(ImageWarpDispatchParams, quadrature) == 52);

    inline ImageWarpCameraParams imageWarpCameraParams(const UndistortParams& p) {
        ImageWarpCameraParams camera{p.src_fx, p.src_fy, p.src_cx, p.src_cy, p.dst_fx, p.dst_fy, p.dst_cx, p.dst_cy, p.src_width, p.src_height, p.dst_width, p.dst_height, int32_t(p.model_type), {}, p.num_distortion};
        std::copy_n(p.distortion, 12, camera.distortion);
        return camera;
    }
} // namespace lfs::core::internal
