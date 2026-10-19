/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "html_viewer_export.hpp"
#include "io/exporter.hpp"
#include "training/trainer.hpp"
#include "training/training_manager.hpp"
#include "visualizer/core/parameter_manager.hpp"
#include "visualizer/core/services.hpp"
#include "visualizer/rendering/rendering_manager.hpp"

namespace lfs::vis::gui {

    std::optional<glm::vec3> resolve_html_export_background_color(
        const std::optional<glm::vec3>& viewport_bg,
        const std::optional<lfs::core::param::OptimizationParameters>& opt_params) {
        if (viewport_bg && *viewport_bg != glm::vec3(0.0f)) {
            return *viewport_bg;
        }
        if (opt_params && opt_params->bg_mode == lfs::core::param::BackgroundMode::SolidColor) {
            return glm::vec3(opt_params->bg_color[0], opt_params->bg_color[1], opt_params->bg_color[2]);
        }
        if (viewport_bg) {
            return *viewport_bg;
        }
        return std::nullopt;
    }

    std::optional<glm::vec3> resolve_html_export_background_color() {
        std::optional<glm::vec3> viewport_bg;
        if (auto* rm = services().renderingOrNull()) {
            viewport_bg = rm->getSettings().background_color;
        }

        std::optional<lfs::core::param::OptimizationParameters> opt_params;
        if (auto* tm = services().trainerOrNull(); tm && tm->hasTrainer()) {
            if (auto* trainer = tm->getTrainer()) {
                opt_params = trainer->getParams().optimization;
            }
        } else if (auto* pm = services().paramsOrNull(); pm && pm->isLoaded()) {
            opt_params = pm->getActiveParams();
        }

        return resolve_html_export_background_color(viewport_bg, opt_params);
    }

    std::expected<void, std::string> export_html_viewer(
        const lfs::core::SplatData& splat_data,
        const HtmlViewerExportOptions& options) {

        const lfs::io::HtmlExportOptions io_options{
            .output_path = options.output_path,
            .kmeans_iterations = 10,
            .progress_callback = options.progress_callback,
            .provenance = options.provenance,
            .background_color = options.background_color};

        auto result = lfs::io::export_html(splat_data, io_options);
        if (!result) {
            return std::unexpected(result.error().format());
        }
        return {};
    }

} // namespace lfs::vis::gui
