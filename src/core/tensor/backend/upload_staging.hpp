/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "../internal/tensor_impl.hpp"
#include <cstring>

namespace lfs::core::internal {
    // Used by implementations whose native copy retains the host pointer.
    // Immediate-copy implementations need no extra host staging allocation.
    class UploadStaging {
        Tensor staging_;

    public:
        Tensor snapshot(std::span<const std::byte> bytes, DataType dtype) {
            const size_t count = bytes.size() / dtype_size(dtype);
            if (!staging_.is_valid() || staging_.dtype() != dtype || staging_.numel() < count)
                staging_ = Tensor::empty({count}, Device::CPU, dtype);
            auto view = staging_.slice(0, 0, count);
            if (!bytes.empty())
                std::memcpy(view.data_ptr(), bytes.data(), bytes.size());
            return view;
        }
    };
} // namespace lfs::core::internal
