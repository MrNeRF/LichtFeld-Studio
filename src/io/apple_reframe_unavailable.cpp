/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "io/apple_reframe.hpp"

namespace lfs::io {
    bool appleReframeAvailable() { return false; }
    Result<LoadResult> createAppleReframeSplat(const std::filesystem::path& photo, const LoadOptions&) {
        return make_error(ErrorCode::UNSUPPORTED_FORMAT, "Photo reconstruction is unavailable in this build", photo);
    }
} // namespace lfs::io
