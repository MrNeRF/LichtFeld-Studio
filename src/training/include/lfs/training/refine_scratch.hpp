/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace lfs::training {

    namespace detail {
        [[nodiscard]] inline size_t grow_only_capacity(const size_t current, const size_t need) {
            if (need == 0 || current >= need) {
                return current;
            }
            // The first reservation is supplied by the model allocator. Do
            // not add another headroom multiplier to it; later refinements
            // retain the bounded growth pattern.
            if (current == 0) {
                return need;
            }
            return std::max(
                need,
                static_cast<size_t>(static_cast<double>(std::max(current, need)) * 1.2) + 1);
        }
    } // namespace detail

    // Grow-only Gumbel-top-k sort buffers + CUB workspace, sized to the
    // actual sort length. Sparse selections can use less than N slots.
    struct GumbelTopKScratch {
        lfs::core::Tensor keys;
        lfs::core::Tensor indices;
        lfs::core::Tensor keys_sorted;
        lfs::core::Tensor indices_sorted;
        lfs::core::Tensor cub;
        size_t n_capacity = 0;
        size_t cub_bytes = 0;

        void ensure_n(const size_t n, const lfs::core::Device device) {
            using namespace lfs::core;
            if (n == 0 || n_capacity >= n) {
                return;
            }
            if (device != Device::CUDA) {
                throw std::invalid_argument("GumbelTopKScratch requires CUDA storage");
            }
            const size_t new_cap = detail::grow_only_capacity(n_capacity, n);
            keys = Tensor::empty_exact({new_cap}, DataType::Float32);
            keys_sorted = Tensor::empty_exact({new_cap}, DataType::Float32);
            indices = Tensor::empty_exact({new_cap}, DataType::UInt32);
            indices_sorted = Tensor::empty_exact({new_cap}, DataType::UInt32);
            keys.zero_();
            keys_sorted.zero_();
            n_capacity = new_cap;
        }

        void ensure_cub(const size_t bytes, const lfs::core::Device device) {
            using namespace lfs::core;
            if (bytes == 0 || cub_bytes >= bytes) {
                return;
            }
            if (device != Device::CUDA) {
                throw std::invalid_argument("GumbelTopKScratch requires CUDA storage");
            }
            const size_t new_cap = detail::grow_only_capacity(cub_bytes, bytes);
            cub = Tensor::empty_exact({new_cap}, DataType::UInt8);
            cub_bytes = new_cap;
        }

        [[nodiscard]] std::size_t resident_bytes() const noexcept {
            return n_capacity * 4 * sizeof(float) + cub_bytes;
        }

        void release() noexcept {
            keys = {};
            indices = {};
            keys_sorted = {};
            indices_sorted = {};
            cub = {};
            n_capacity = 0;
            cub_bytes = 0;
        }
    };

} // namespace lfs::training
