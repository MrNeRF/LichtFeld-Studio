/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/error.hpp"
#include "core/tensor.hpp"

namespace lfs::core {
    // Linear RGBA [height,width,4] image: black RGB, vignette coverage in alpha.
    // Uses the current tensor allocation backend; no presentation dependency.
    [[nodiscard]] LFS_CORE_API Result<Tensor> vignette_image(
        uint32_t width, uint32_t height, float intensity, float radius, float softness);
} // namespace lfs::core
