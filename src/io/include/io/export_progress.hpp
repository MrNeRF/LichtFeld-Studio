/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/progress.hpp"
namespace lfs::io {
    // Returning false requests cancellation before publication.
    using ExportProgressCallback = core::ProgressCallback;
} // namespace lfs::io
