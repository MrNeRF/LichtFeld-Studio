/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/tensor.hpp"
#include "core/tensor_upload.hpp"

#include <algorithm>
#include <cstddef>
#include <span>
#include <vector>

namespace lfs::vis {
    // Per-frame host data for GPU programs. Tensor::to(Device::GPU) waits for
    // the whole queue; these uploads are ordered on the tensor timeline
    // instead, and the host bytes may be reused as soon as upload() returns.
    class TensorFrameUploads {
    public:
        lfs::core::Tensor upload(std::span<const std::byte> bytes, lfs::core::TensorShape shape,
                                 lfs::core::DataType dtype) {
            std::erase_if(slots_, [](lfs::core::TensorUpload& slot) { return slot.poll(); });
            auto destination = lfs::core::Tensor::empty(shape, lfs::core::Device::GPU, dtype);
            if (bytes.empty())
                return destination;
            slots_.emplace_back().enqueue_in_batch(destination, bytes);
            return destination;
        }

        template <typename T>
        lfs::core::Tensor upload(std::span<const T> values, lfs::core::TensorShape shape,
                                 lfs::core::DataType dtype) {
            return upload(std::as_bytes(values), shape, dtype);
        }

    private:
        std::vector<lfs::core::TensorUpload> slots_;
    };
} // namespace lfs::vis
