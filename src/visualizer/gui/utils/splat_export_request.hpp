/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <core/export.hpp>
#include <core/uuid.hpp>
#include <string>
#include <vector>
namespace lfs::vis::gui {
    LFS_VIS_API void requestSplatExport(std::vector<std::string> node_names);
    LFS_VIS_API std::vector<std::string> takeSplatExportRequest();
    LFS_VIS_API void requestCollisionExport(core::Uuid node);
    LFS_VIS_API core::Uuid takeCollisionExportRequest();
} // namespace lfs::vis::gui
