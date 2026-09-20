/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <filesystem>
#include <iosfwd>

namespace lfs::app {

    enum class OverwriteChoice {
        YES,
        NO,
        ALL,
    };

    OverwriteChoice ask_overwrite(
        const std::filesystem::path& path,
        std::istream& input,
        std::ostream& output);

} // namespace lfs::app
