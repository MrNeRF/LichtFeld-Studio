/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"

namespace lfs::io {

    using lfs::core::Tensor;

    /**
     * @brief Compute Morton codes for 3D positions
     *
     * Encodes 3D positions into Morton codes (Z-order curve) for spatial sorting.
     * Improves cache locality during rendering and compression.
     *
     * @param positions Tensor of shape [N, 3] containing 3D positions (Float32, CUDA)
     * @return Tensor of shape [N] containing Morton codes as Int64
     */
    Tensor morton_encode(const Tensor& positions);

    /**
     * @brief Sort indices by Morton codes
     *
     * @param morton_codes Tensor of Morton codes (Int64, CUDA)
     * @return Tensor of indices that would sort the Morton codes (Int64)
     */
    Tensor morton_sort_indices(const Tensor& morton_codes);

    /**
     * @brief Sort indices by Morton codes, consuming the key buffer.
     *
     * This avoids cloning the Morton code tensor when the caller does not need
     * the unsorted codes after the sort.
     *
     * @param morton_codes Tensor of Morton codes (Int64, CUDA), sorted in-place
     * @return Tensor of indices that would sort the original Morton codes (Int64)
     */
    Tensor morton_sort_indices_inplace(Tensor& morton_codes);

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
