// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "core/error.hpp"
#include "core/tensor.hpp"
#include "core/video_color_math.h"
namespace lfs::core {
    // Packed UInt8 component storage -> contiguous HWC Float32 linear RGB.
    // Output preserves the input device/backend; all component ranges are checked.
    // Decode/resize share the sampler used by the HDR tensor tonemapper.
    LFS_CORE_API Result<Tensor> video_to_linear_rgb(const Tensor& bytes, const color::VideoColorParameters&, uint32_t width, uint32_t height);

    struct Yuv420Planes {
        Tensor y, u, v;
    };
    // Float32 HWC RGB -> BT.601 limited-range UInt8. Even dimensions; accepts
    // strided inputs. Output remains on the input device/backend and timeline.
    // The into variant permits producers to reuse plane storage. Inputs and
    // outputs must not alias; output planes have the exact visible 2D shapes.
    LFS_CORE_API Result<void> rgb_to_yuv420p_into(const Tensor& rgb, Yuv420Planes& output);
    LFS_CORE_API Result<Yuv420Planes> rgb_to_yuv420p(const Tensor& rgb);
} // namespace lfs::core
