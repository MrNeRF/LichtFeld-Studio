/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/tensor.hpp"

namespace lfs::io {
    // Keep policy for a source- or output-grid mask, applied while preparing a derived cache entry.
    struct MaskedImageFill {
        lfs::core::Tensor mask;
        bool alpha = false;
        bool invert = false;
        float threshold = 0.f;
        bool keep_high_band = false;
        int resize_factor = 1;
        int max_width = 0;
    };
    // Mutates only ignored RGB in a raw HWC staging tensor; RGBA alpha is preserved.
    const char* masked_image_fill_error();
    void fill_masked_image(lfs::core::Tensor& image, const MaskedImageFill& policy, cudaStream_t stream);
} // namespace lfs::io
