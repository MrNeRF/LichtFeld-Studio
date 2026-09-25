/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace tinyply {
    struct PlyFile;
}

namespace lfs::io {

    // tinyply reads a body without checking for a short read: a truncated body loads as
    // zeros and an oversized row count spins on the failed stream. Returns why a body of
    // `body_bytes` cannot hold the rows the parsed header declares.
    [[nodiscard]] std::optional<std::string> tinyply_body_shortfall(const tinyply::PlyFile& ply,
                                                                    std::uint64_t body_bytes);

} // namespace lfs::io
