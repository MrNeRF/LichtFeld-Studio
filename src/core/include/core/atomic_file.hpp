// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "core/error.hpp"
#include "core/export.hpp"
#include <cstdio>
#include <filesystem>
#include <functional>
#include <string_view>

namespace lfs::core {
    struct AtomicFileOptions {
        bool overwrite = true;
        bool durable = true;
        bool create_directories = true;
        std::function<bool()> cancelled;
    };
    // The stream is binary, seekable and exclusively owned by this call. The
    // callback must not close or retain it. Failure/cancel before publication
    // leaves the destination intact. No-replace also protects against races.
    using AtomicFileWriter = std::function<Status(FILE*)>;
    [[nodiscard]] LFS_FILE_IO_API Status writeFileAtomically(
        const std::filesystem::path&, const AtomicFileWriter&, const AtomicFileOptions& = {});
    [[nodiscard]] LFS_FILE_IO_API Status writeTextFileAtomically(
        const std::filesystem::path&, std::string_view, const AtomicFileOptions& = {});
} // namespace lfs::core
