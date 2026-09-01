/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"

namespace lfs::io {

    using lfs::core::Tensor;

    /**
     * @brief Compute 63-bit Morton codes and sort indices on the GPU.
     *
     * SOG export uses 21 bits per axis over the global bounds. This finer grid
     * avoids the dense-cell scrambling caused by a single 10-bit pass while
     * retaining the source order for equal keys.
     *
     * @param positions Tensor of shape [N, 3] containing 3D positions (Float32, CUDA)
     * @return Tensor of sorted indices (Int32, CUDA)
     */
    Tensor morton_sort_indices_for_positions(const Tensor& positions);

} // namespace lfs::io
