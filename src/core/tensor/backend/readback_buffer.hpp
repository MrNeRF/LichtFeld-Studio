/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/tensor/internal/private_access.hpp"

#include "descriptors.hpp"

namespace lfs::core::internal {
    // Backend-owned reusable staging. Destruction waits for outstanding work.
    class ReadbackBuffer {
    public:
        virtual ~ReadbackBuffer() = default;
        virtual void enqueue(StorageRef source, size_t bytes, ExecContext context) = 0;
        virtual void wait() = 0;
        // A null destination discards a completed result.
        virtual bool poll(void* destination) = 0;
    };
} // namespace lfs::core::internal
