/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "io/error.hpp"

#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace lfs::io {

    // Internal helper shared by the PLY exporter and Python bindings.
    [[nodiscard]] std::vector<std::string> make_ply_extra_attribute_names(
        std::string_view base_name, size_t count);

    // Internal helper for early Python-boundary feedback on reserved exporter names.
    [[nodiscard]] Result<void> validate_reserved_ply_extra_attribute_names(
        std::span<const std::string> names, const std::filesystem::path& output_path);

} // namespace lfs::io
