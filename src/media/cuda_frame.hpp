// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "core/video_color_math.h"
#include <array>
#include <cstdint>

namespace lfs::media::detail {
    // Native CUDA handles, not decoder objects. The host adapter owns the driver
    // calls; media owns the decoded frame until finishHardware() has returned.
    struct CudaDecodeContext {
        void* context = nullptr;
        void* stream = nullptr;
    };
    enum class CudaVideoLayout { NV12,
                                 P010 };
    struct CudaVideoFrame {
        CudaDecodeContext decoder;
        std::array<const std::uint8_t*, 2> planes;
        std::array<int, 2> row_stride;
        bool full_range = false;
        CudaVideoLayout layout = CudaVideoLayout::NV12;
        core::color::VideoColorParameters color{};
        bool legacy_bt601 = true;
    };
} // namespace lfs::media::detail
