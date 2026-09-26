/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "io/ppisp_file.hpp"
#include <cstdint>
#include <expected>
#include <filesystem>
#include <string>
#include <vector>

namespace lfs::training {

    class PPISP;
    class PPISPControllerPool;

    /// Import PPISP state from a standalone legacy file.
    /// @param path Input file path
    /// @param ppisp PPISP instance to load into (must be pre-constructed with matching dimensions)
    /// @param controller_pool Optional controller pool to load into
    /// @param metadata Optional metadata output for trainer-side reuse
    [[nodiscard]] std::expected<void, std::string> load_ppisp_file(
        const std::filesystem::path& path,
        PPISP& ppisp,
        PPISPControllerPool* controller_pool = nullptr,
        PPISPFileMetadata* metadata = nullptr);

} // namespace lfs::training
