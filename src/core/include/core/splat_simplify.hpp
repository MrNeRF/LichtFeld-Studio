/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/splat_simplify_types.hpp"

#include <expected>
#include <memory>

namespace lfs::core {

    class SplatData;

    LFS_CORE_API std::expected<std::unique_ptr<SplatData>, std::string> simplify_splats(
        const SplatData& input,
        const SplatSimplifyOptions& options = {},
        SplatSimplifyProgressCallback progress = {});

} // namespace lfs::core
