/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <filesystem>

namespace lfs::core {
    class SplatData;
}

namespace lfs::vis {

    void quantizeViewerLoadedPlyShN(const std::filesystem::path& path, lfs::core::SplatData& model);

} // namespace lfs::vis
