/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace lfs::vis::gui {
    struct RmlImageFile {
        std::string path;
        int width = 0;
        int height = 0;
        std::vector<std::uint8_t> rgba; // premultiplied
    };

    // Decodes an RmlUi image source to premultiplied RGBA8. The source is tried
    // as given, then as a bundled asset, then (POSIX) as an absolute path whose
    // root slash RmlUi dropped. Single-channel images are masks: gray drives alpha.
    [[nodiscard]] std::optional<RmlImageFile> loadRmlImageFile(const std::string& source);
} // namespace lfs::vis::gui
