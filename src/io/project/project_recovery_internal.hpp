/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "io/project_container.hpp"

#include <filesystem>
#include <vector>

namespace lfs::io::project::detail {

    struct ValidBoundAutosave {
        std::filesystem::path path;
        std::uint64_t sequence = 0;
        lfs::core::Uuid snapshot_uuid;
    };

    // The caller must already hold the master writer lock. Candidates that do
    // not open, bind, verify, or form a complete overlay are simply not valid.
    [[nodiscard]] lfs::Result<std::vector<ValidBoundAutosave>>
    valid_bound_autosaves_locked(
        const std::filesystem::path& master_path,
        const ProjectReader& master);

} // namespace lfs::io::project::detail
