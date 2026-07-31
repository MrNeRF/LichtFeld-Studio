/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "app/application_startup.hpp"

#include "core/parameters.hpp"

namespace lfs::app {

    lfs::Result<void> validate_application_startup(
        const lfs::core::param::TrainingParameters&) {
        return {};
    }

} // namespace lfs::app
