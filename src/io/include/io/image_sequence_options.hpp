// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "media/image_sequence.hpp"

namespace lfs::io {
    struct TensorImageSequenceOptions {
        media::ImageSequenceOptions output;
        // Caller-declared HWC float RGB/RGBA color; never inferred from a file,
        // image dimensions or Tensor device.
        media::FrameColor input_color;
        std::size_t readback_budget = 1024ULL * 1024 * 1024;
    };
} // namespace lfs::io
