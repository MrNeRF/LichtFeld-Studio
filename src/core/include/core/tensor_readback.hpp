/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/export.hpp"

#include <cstddef>
#include <memory>
#include <span>

namespace lfs::core {
    class Tensor;

    // One reusable asynchronous download slot. The source and staging memory
    // remain owned until completion, including when the slot is destroyed.
    // Calls on a slot must be externally serialized. CPU inputs complete inline.
    class LFS_CORE_API TensorReadback {
    public:
        TensorReadback();
        ~TensorReadback();
        TensorReadback(TensorReadback&&) noexcept;
        TensorReadback& operator=(TensorReadback&&) noexcept;
        TensorReadback(const TensorReadback&) = delete;
        TensorReadback& operator=(const TensorReadback&) = delete;

        // Accepts strided tensors and follows the source's storage backend.
        // Complete the previous download before reusing this slot.
        void enqueue(const Tensor& source);
        // Byte range is measured in the source's contiguous logical layout.
        void enqueue_range(const Tensor& source, std::size_t byte_offset,
                           std::size_t byte_count);
        [[nodiscard]] bool pending() const noexcept;

        // Copies all bytes and consumes the pending download when ready.
        // The destination must have exactly source.bytes() bytes. poll leaves it
        // untouched when not ready. Both functions require a pending download.
        [[nodiscard]] bool poll(std::span<std::byte> destination);
        void wait(std::span<std::byte> destination);

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace lfs::core
