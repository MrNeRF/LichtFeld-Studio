/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/parameters.hpp"

#include <string>
#include <vector>

namespace lfs::app {

    std::vector<lfs::core::param::BatchJobItem> resolve_batch_jobs(
        const lfs::core::param::BatchTrainingParameters& params);

    std::vector<std::string> build_child_args(
        const lfs::core::param::BatchJobItem& job,
        const lfs::core::param::BatchTrainingParameters& params);

    bool validate_job_arguments(
        const lfs::core::param::BatchJobItem& job,
        const lfs::core::param::BatchTrainingParameters& params,
        std::string& error);

    std::string batch_timeout_message(int minutes);

    int run_batch_training(const lfs::core::param::BatchTrainingParameters& params);

} // namespace lfs::app
