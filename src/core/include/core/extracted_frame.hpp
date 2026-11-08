/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <filesystem>

namespace lfs::core {
    // A finalized extractor output. timestamp is the extractor's observed relative
    // time, not an exact source PTS. Identity additionally requires its extraction run.
    struct ExtractedFrame {
        std::filesystem::path path;
        int width = 0, height = 0;
        double timestamp = 0;
        int source_frame = 0;
        bool operator==(const ExtractedFrame&) const = default;
    };
} // namespace lfs::core
