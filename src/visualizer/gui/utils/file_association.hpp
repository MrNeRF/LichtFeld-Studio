/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <core/export.hpp>

#include <string>
#include <string_view>
#include <vector>

namespace lfs::vis::gui {

    struct FileAssociationStatus {
        std::string extension;
        bool registered = false;
    };

    LFS_VIS_API bool registerFileAssociations();
    LFS_VIS_API bool unregisterFileAssociations();
    LFS_VIS_API bool areFileAssociationsRegistered();
    LFS_VIS_API bool openFileAssociationSettings();
    LFS_VIS_API bool registerFileAssociation(std::string_view extension);
    LFS_VIS_API bool unregisterFileAssociation(std::string_view extension);
    LFS_VIS_API std::vector<FileAssociationStatus> fileAssociationsStatus();

} // namespace lfs::vis::gui
