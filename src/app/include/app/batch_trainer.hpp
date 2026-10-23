/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/parameters.hpp"

#include <vector>

namespace lfs::app {

    std::vector<lfs::core::param::BatchJobItem> resolve_batch_jobs(
        const lfs::core::param::BatchTrainingParameters& params);

    int run_batch_training(const lfs::core::param::BatchTrainingParameters& params);

} // namespace lfs::app

