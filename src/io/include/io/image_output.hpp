// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "core/tensor.hpp"
#include "media/image_output.hpp"

namespace lfs::io {
    // Host adapter, not another writer. Input is explicitly HWC Float16/Float32
    // RGB/RGBA with declared linear color/alpha. No quantization, transfer
    // conversion, guessed layout or synthetic video timing. The synchronous
    // call retains CPU storage until the shared EXR writer has returned; callers
    // must not mutate input storage during this synchronous operation.
    [[nodiscard]] media::SinkResult writeExrImage(const std::filesystem::path&, const core::Tensor&,
                                                  media::FrameColor, const media::ExrOutputOptions& = {},
                                                  const media::FrameInfo& = {});
} // namespace lfs::io
