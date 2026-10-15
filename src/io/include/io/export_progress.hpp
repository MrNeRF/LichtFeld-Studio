/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <functional>
#include <string>
namespace lfs::io {
    // Returning false requests cancellation before publication.
    using ExportProgressCallback = std::function<bool(float progress, const std::string& stage)>;
} // namespace lfs::io
