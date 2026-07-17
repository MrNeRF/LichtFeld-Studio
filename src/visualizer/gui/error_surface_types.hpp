/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <core/export.hpp>

#include <cstdint>
#include <string>
#include <string_view>

// Shared light types for the native error surfaces (Phase 8, packet P2). No
// RmlUi dependency, so the policy cores (ToastStack, StatusMessageState) and
// their unit tests can use them headlessly.
namespace lfs::vis::gui {

    enum class ErrorNoticeLevel : std::uint8_t { Info,
                                                 Warning,
                                                 Error };

    // One transient notification. `title` and `message` are plain text (the
    // overlay escapes them before insertion). `fingerprint` collapses repeats of
    // the same fault into one visible toast with a counter.
    struct ToastRequest {
        std::string title;
        std::string message;
        ErrorNoticeLevel level = ErrorNoticeLevel::Error;
        std::uint64_t fingerprint = 0;
    };

    // Escapes RML metacharacters and turns newlines into breaks so arbitrary
    // error text cannot corrupt the document. Shared by the modal consumer and
    // the toast overlay.
    [[nodiscard]] LFS_VIS_API std::string escapeRmlText(std::string_view text);

} // namespace lfs::vis::gui
