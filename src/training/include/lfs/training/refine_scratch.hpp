/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"

#include <algorithm>
#include <cstddef>

namespace lfs::training {

    namespace detail {
        [[nodiscard]] inline size_t grow_only_capacity(const size_t current, const size_t need) {
            if (need == 0 || current >= need) {
                return current;
            }
            return std::max(
                need,
                static_cast<size_t>(static_cast<double>(std::max(current, need)) * 1.2) + 1);
        }
    } // namespace detail

    // Grow-only Gumbel-top-k sort buffers + CUB workspace, pre-sized to max_cap.
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
            const size_t new_cap = detail::grow_only_capacity(n_capacity, n);
            keys = Tensor::zeros_direct(TensorShape({new_cap}), new_cap, device, DataType::Float32);
            keys_sorted = Tensor::zeros_direct(TensorShape({new_cap}), new_cap, device, DataType::Float32);
            indices = Tensor::empty({new_cap}, device, DataType::Int64);
            indices_sorted = Tensor::empty({new_cap}, device, DataType::Int64);
            n_capacity = new_cap;
        }

        void ensure_cub(const size_t bytes, const lfs::core::Device device) {
            using namespace lfs::core;
            if (bytes == 0 || cub_bytes >= bytes) {
                return;
            }
            const size_t new_cap = detail::grow_only_capacity(cub_bytes, bytes);
            cub = Tensor::empty({new_cap}, device, DataType::UInt8);
            cub_bytes = new_cap;
        }
    };

    // Grow-only positive-median workspace; omitting scratch keeps the malloc path.
    struct PositiveMedianScratch {
        lfs::core::Tensor selected;
        lfs::core::Tensor sorted;
        lfs::core::Tensor count;
        lfs::core::Tensor select_temp;
        lfs::core::Tensor sort_temp;
        size_t n_capacity = 0;
        size_t select_temp_bytes = 0;
        size_t sort_temp_bytes = 0;

        void ensure_n(const size_t n, const lfs::core::Device device) {
            using namespace lfs::core;
            if (n == 0 || n_capacity >= n) {
                return;
            }
            const size_t new_cap = detail::grow_only_capacity(n_capacity, n);
            selected = Tensor::zeros_direct(TensorShape({new_cap}), new_cap, device, DataType::Float32);
            sorted = Tensor::zeros_direct(TensorShape({new_cap}), new_cap, device, DataType::Float32);
            if (!count.is_valid() || count.numel() < 1 ||
                count.device() != device || count.dtype() != DataType::Int32) {
                count = Tensor::zeros({1}, device, DataType::Int32);
            }
            n_capacity = new_cap;
        }

        void ensure_temps(const size_t select_bytes,
                          const size_t sort_bytes,
                          const lfs::core::Device device) {
            using namespace lfs::core;
            if (select_bytes > select_temp_bytes) {
                const size_t new_cap = detail::grow_only_capacity(select_temp_bytes, select_bytes);
                select_temp = Tensor::empty({new_cap}, device, DataType::UInt8);
                select_temp_bytes = new_cap;
            }
            if (sort_bytes > sort_temp_bytes) {
                const size_t new_cap = detail::grow_only_capacity(sort_temp_bytes, sort_bytes);
                sort_temp = Tensor::empty({new_cap}, device, DataType::UInt8);
                sort_temp_bytes = new_cap;
            }
        }
    };

} // namespace lfs::training
