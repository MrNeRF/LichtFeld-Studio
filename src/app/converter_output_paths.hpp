/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/parameters.hpp"

#include <filesystem>

namespace lfs::app {

    [[nodiscard]] std::filesystem::path generate_converter_output_path(
        const std::filesystem::path& input,
        const std::filesystem::path& output_template,
        lfs::core::param::OutputFormat format,
        const char* suffix,
        bool replace_output_extension = false);

    [[nodiscard]] std::filesystem::path generate_ssog_batch_output_path(
        const std::filesystem::path& output_directory,
        const std::filesystem::path& input);

} // namespace lfs::app
