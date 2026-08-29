/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "core/export.hpp"

#include <cstddef>
#include <span>
#include <streambuf>

namespace lfs::core {

    // Memory-backed input streambuf that can expose unread bytes without a copy.
    // Tensor deserialize uses this to cudaMemcpyAsync from a visit_materialized
    // span. Concrete buffers (SpanStreambuf) live in lfs_io.
    class LFS_CORE_API ContiguousStreambuf : public std::streambuf {
    public:
        ~ContiguousStreambuf() override;

        [[nodiscard]] virtual std::span<const std::byte>
        remaining_bytes() const noexcept = 0;
    };

} // namespace lfs::core
