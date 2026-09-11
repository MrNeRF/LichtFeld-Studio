/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/error_latch.hpp"

#include <utility>

namespace lfs::core {

    void ErrorLatch::set(Error error) noexcept {
        std::lock_guard lock(mutex_);
        error_ = std::move(error);
    }

    void ErrorLatch::clear() noexcept {
        std::lock_guard lock(mutex_);
        error_.reset();
    }

    std::optional<Error> ErrorLatch::get() const {
        std::lock_guard lock(mutex_);
        return error_;
    }

} // namespace lfs::core
