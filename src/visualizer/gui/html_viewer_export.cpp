/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "html_viewer_export.hpp"
#include "io/exporter.hpp"

namespace lfs::vis::gui {

    std::expected<void, std::string> export_html_viewer(
        const lfs::core::SplatData& splat_data,
        const HtmlViewerExportOptions& options) {

        const lfs::io::HtmlExportOptions io_options{
            .output_path = options.output_path,
            .kmeans_iterations = 10,
            .progress_callback = options.progress_callback,
            .provenance = options.provenance};

        auto result = lfs::io::export_html(splat_data, io_options);
        if (!result) {
            return std::unexpected(result.error().format());
        }
        return {};
    }

} // namespace lfs::vis::gui
