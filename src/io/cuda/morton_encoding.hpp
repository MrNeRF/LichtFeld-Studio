/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"

namespace lfs::io {

    using lfs::core::Tensor;

    /**
     * @brief Compute Morton codes and sort indices using compact 32-bit buffers.
     *
     * SOG export only needs 30-bit Morton keys and supports at most INT_MAX
     * splats on the GPU sort path, so this avoids the 64-bit key/index buffers
     * used by the generic API.
     *
     * @param positions Tensor of shape [N, 3] containing 3D positions (Float32, CUDA)
     * @return Tensor of sorted indices (Int32, CUDA)
     */
    Tensor morton_sort_indices_for_positions(const Tensor& positions);

} // namespace lfs::io
