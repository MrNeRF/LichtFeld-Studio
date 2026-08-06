/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

namespace lfs::core::param {

    LFS_CORE_API void register_optimization_properties();
    LFS_CORE_API void ensure_optimization_properties_registered();

} // namespace lfs::core::param
