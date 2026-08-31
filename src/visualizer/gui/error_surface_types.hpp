/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <core/error.hpp>
#include <core/export.hpp>
#include <core/resource_messages.hpp>

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

    // Host-RAM exhaustion (e.g. a refused project save) shares
    // ResourceExhausted/Training with GPU OOM; the snapshot service prefixes
    // its user message with HOST_MEMORY_SAVE_ERROR_PREFIX so the surfaces can
    // keep the GPU-OOM presentation for actual GPU failures.
    [[nodiscard]] inline bool isHostMemorySaveMessage(const std::string_view message) noexcept {
        return message.find(lfs::core::HOST_MEMORY_SAVE_ERROR_PREFIX) != std::string_view::npos;
    }

    [[nodiscard]] inline bool isHostMemoryExhaustion(const lfs::Error& error) noexcept {
        return isHostMemorySaveMessage(error.user_message());
    }

} // namespace lfs::vis::gui
