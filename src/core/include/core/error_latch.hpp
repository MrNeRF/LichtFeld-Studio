/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/error.hpp"
#include "core/export.hpp"

#include <mutex>
#include <optional>

namespace lfs::core {

    // Thread-safe most-recent-error latch. set() and get() cross threads (a
    // worker's terminal completion vs an MCP/TCP query thread); the Error copy
    // get() returns is a thread-safe intrusive add-ref, so the caller always
    // sees a stable, complete snapshot rather than a torn read.
    class LFS_CORE_API ErrorLatch {
    public:
        void set(Error error) noexcept;
        void clear() noexcept;
        [[nodiscard]] std::optional<Error> get() const;

    private:
        mutable std::mutex mutex_;
        std::optional<Error> error_;
    };

} // namespace lfs::core
