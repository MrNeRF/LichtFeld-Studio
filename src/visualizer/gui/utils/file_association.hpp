/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <core/export.hpp>

namespace lfs::vis::gui {

    LFS_VIS_API bool registerFileAssociations();
    LFS_VIS_API bool unregisterFileAssociations();
    LFS_VIS_API bool areFileAssociationsRegistered();
    LFS_VIS_API bool openFileAssociationSettings();

} // namespace lfs::vis::gui
