/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/error.hpp"
#include <string>
#include <utility>

namespace lfs::vis {
    inline lfs::Error viewportError(
        std::string message,
        const lfs::ErrorCode code = lfs::ErrorCode::Internal,
        const lfs::core::SourceSite source = LFS_SOURCE_SITE_CURRENT()) {
        return lfs::make_error(lfs::ErrorInit{
            .code = code,
            .domain = lfs::ErrorDomain::Rendering,
            .user_message = message,
            .detail = std::move(message),
            .detection = source,
        });
    }

    inline lfs::Error viewportError(lfs::Error error) {
        return error;
    }
    inline std::string viewportErrorText(const lfs::Error& error) {
        return std::string(error.user_message().empty() ? error.detail() : error.user_message());
    }

    inline const std::string& viewportErrorText(const std::string& error) {
        return error;
    }
} // namespace lfs::vis
