/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/splat_data.hpp"
#include "io/exporter.hpp"
#include <expected>
#include <filesystem>
#include <string>

namespace lfs::vis::gui {

    struct HtmlViewerExportOptions {
        std::filesystem::path output_path;
        lfs::io::ExportProgressCallback progress_callback;
    };

    LFS_VIS_API std::expected<void, std::string> export_html_viewer(
        const lfs::core::SplatData& splat_data,
        const HtmlViewerExportOptions& options);

} // namespace lfs::vis::gui
