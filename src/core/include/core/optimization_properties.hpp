/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

namespace lfs::core::param {

    struct OptimizationParameters;

    LFS_CORE_API void register_optimization_properties();
    LFS_CORE_API void ensure_optimization_properties_registered();

    // Keep initialization-only settings in the active run. Other settings remain
    // editable for the next run, including independently configured strategies.
    LFS_CORE_API void apply_live_optimization_updates(
        OptimizationParameters& current, const OptimizationParameters& edited);

} // namespace lfs::core::param
