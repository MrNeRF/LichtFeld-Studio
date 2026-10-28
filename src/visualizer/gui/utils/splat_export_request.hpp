/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <core/export.hpp>
#include <string>
#include <vector>
namespace lfs::vis::gui {
    LFS_VIS_API void requestSplatExport(std::vector<std::string> node_names);
    LFS_VIS_API std::vector<std::string> takeSplatExportRequest();
} // namespace lfs::vis::gui
