/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/export.hpp"

#include <filesystem>
#include <memory>
#include <utility>

namespace lfs::core::nn {

    // Reserves an archive through the handoff to a loader, including deferred
    // consumers. Loaded models retain the same token after closing the mapping.
    struct LFS_CORE_API WeightSource {
        WeightSource(std::filesystem::path value);
        WeightSource(std::filesystem::path value, std::shared_ptr<const void> retained)
            : path(std::move(value)), source(std::move(retained)) {}

        std::filesystem::path path;
        std::shared_ptr<const void> source;
    };

} // namespace lfs::core::nn
