/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "tensor_impl.hpp"
#include <algorithm>
#include <ranges>

namespace lfs::core {

    // Pure functions for broadcasting - no classes!
    namespace broadcast {

        /**
         * @brief Compute broadcast shape for two shapes following NumPy rules
         * @param a First shape
         * @param b Second shape
         * @return Broadcast result shape, or empty vector if incompatible
         * @details
         * - Aligns shapes from the right
         * - Dimensions must either match or one must be 1
         * - Result takes the maximum of each dimension
         * - Handles zero dimensions: 0 can only broadcast with another 0
         */
        inline std::vector<size_t> shape(std::span<const size_t> a, std::span<const size_t> b) {
            size_t max_rank = std::max(a.size(), b.size());
            std::vector<size_t> result(max_rank);

            // Work backwards (numpy-style broadcasting)
            for (size_t i = 0; i < max_rank; ++i) {
                size_t dim_a = (i < a.size()) ? a[a.size() - 1 - i] : 1;
                size_t dim_b = (i < b.size()) ? b[b.size() - 1 - i] : 1;

                // Handle zero dimensions
                if (dim_a == 0 && dim_b == 0) {
                    result[max_rank - 1 - i] = 0;
                } else if (dim_a == 0 || dim_b == 0) {
                    // Can't broadcast 0 with non-zero (except matching 0)
                    return {}; // Incompatible
                } else if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
                    return {}; // Incompatible
                } else {
                    result[max_rank - 1 - i] = std::max(dim_a, dim_b);
                }
            }
            return result;
        }

        /**
         * @brief Check if two shapes are broadcast-compatible
         * @param a First shape
         * @param b Second shape
         * @return True if shapes can be broadcast together, false otherwise
         */
        inline bool can_broadcast(std::span<const size_t> a, std::span<const size_t> b) {
            auto result = shape(a, b);
            return !result.empty();
        }

        /**
         * @brief Map output index to input index for broadcasting (CPU fallback)
         * @param out_idx Linear index in output array
         * @param out_shape Shape of output array
         * @param in_shape Shape of input array
         * @return Linear index in input array
         * @details Used for CPU fallback when CUDA is not available
         */
        inline size_t index(size_t out_idx, std::span<const size_t> out_shape,
                            std::span<const size_t> in_shape) {
            size_t in_idx = 0;
            size_t stride = 1;

            for (int i = out_shape.size() - 1; i >= 0; --i) {
                size_t coord = (out_idx / stride) % out_shape[i];
                stride *= out_shape[i];

                int in_dim = i - (out_shape.size() - in_shape.size());
                if (in_dim >= 0) {
                    size_t in_coord = (in_shape[in_dim] == 1) ? 0 : coord;
                    size_t in_stride = 1;
                    for (size_t j = in_dim + 1; j < in_shape.size(); ++j) {
                        in_stride *= in_shape[j];
                    }
                    in_idx += in_coord * in_stride;
                }
            }
            return in_idx;
        }

    } // namespace broadcast

    // Single function to handle all broadcasting
    Tensor broadcast_to(const Tensor& src, const TensorShape& target);

} // namespace lfs::core