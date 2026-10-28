/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/parameters.hpp"

#include <string>
#include <string_view>
#include <vector>

namespace lfs::app {

    // Expects manifests to have been loaded by the CLI parser; throws on invalid input paths.
    std::vector<lfs::core::param::BatchJobItem> resolve_batch_jobs(
        const lfs::core::param::BatchTrainingParameters& params);

    std::vector<std::string> build_child_args(
        const lfs::core::param::BatchJobItem& job,
        const lfs::core::param::BatchTrainingParameters& params);

    bool validate_job_arguments(
        const lfs::core::param::BatchJobItem& job,
        const lfs::core::param::BatchTrainingParameters& params,
        std::string& error);

    bool validate_batch_report_paths(const lfs::core::param::BatchTrainingParameters& params, std::string& error);

    bool write_batch_report(const std::filesystem::path& destination, std::string_view contents);

    std::string batch_timeout_message(int minutes);

    int run_batch_training(const lfs::core::param::BatchTrainingParameters& params);

} // namespace lfs::app
