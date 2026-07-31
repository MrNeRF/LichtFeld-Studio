/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "core/error.hpp"

namespace lfs::core::param {
    struct TrainingParameters;
}

namespace lfs::app {

    [[nodiscard]] lfs::Result<void>
    validate_application_startup(
        const lfs::core::param::TrainingParameters& params);

} // namespace lfs::app
