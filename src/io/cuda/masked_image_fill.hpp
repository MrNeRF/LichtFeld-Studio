/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/tensor.hpp"
#include "io/masked_image_fill.hpp"

namespace lfs::io {
    // Mutates only ignored RGB in a raw HWC staging tensor; RGBA alpha is preserved.
    void fill_masked_image(lfs::core::Tensor& image, const MaskedImageFill& policy, cudaStream_t stream);
} // namespace lfs::io
