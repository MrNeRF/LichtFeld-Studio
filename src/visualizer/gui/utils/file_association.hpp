/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <core/export.hpp>
#include <filesystem>
#include <optional>

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

    // Registers .licht for the current user when it is not registered or its registered executable no longer
    // exists, unless the user removed the association. Shows no UI.
    LFS_VIS_API void ensureProjectFileAssociation();

    namespace detail {
        [[nodiscard]] LFS_VIS_API std::optional<std::filesystem::path>
        executableFromOpenCommand(std::wstring_view command);
        [[nodiscard]] LFS_VIS_API bool
        projectAssociationNeedsRegistration(bool opted_out,
                                            const std::optional<std::filesystem::path>& registered_executable);
    } // namespace detail

} // namespace lfs::vis::gui
