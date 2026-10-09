/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "app/compare.hpp"
#include "app/application.hpp"
#include "core/path_utils.hpp"
#include "training/metrics/compare_images.hpp"

#include <print>

namespace lfs::app {

    int run_compare(const lfs::core::param::CompareParameters& params) {
        install_image_loader(false);
        const auto result = lfs::training::compare_image_sets(params, prepare_lpips_weights(!params.no_download));
        if (!result) {
            std::println(stderr, "Error: {}", result.error().user_message());
            return 1;
        }
        std::print("\n{}", lfs::training::format_compare_report(params, *result));
        if (!result->metrics.valid) {
            std::println(stderr, "Error: no image could be compared");
            return 1;
        }
        std::println("\nReports written to {}", lfs::core::path_to_utf8(params.output_path));
        return 0;
    }

} // namespace lfs::app
