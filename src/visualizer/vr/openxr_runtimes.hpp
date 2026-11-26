/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include <string>
#include <vector>

namespace lfs::vis::vr {

    struct OpenXrRuntime {
        std::string name;     // from its manifest, else the manifest's file name
        std::string manifest; // UTF-8 path of the runtime's JSON manifest
        bool operator==(const OpenXrRuntime&) const = default;
    };

    struct OpenXrRuntimes {
        std::string active;                   // name of the system's active runtime; empty when there is none
        std::vector<OpenXrRuntime> installed; // by name, the active one included
    };

    // The OpenXR runtimes this computer declares. Only reads; a computer can
    // hold several, one for each headset software, and the system names one of
    // them the active runtime.
    [[nodiscard]] LFS_VIS_API OpenXrRuntimes installedOpenXrRuntimes();

    // Makes this process use the runtime of this manifest instead of the
    // system's active one, which is not changed. Call before the first OpenXR
    // call. False, and nothing changes, when the manifest is not a file.
    [[nodiscard]] LFS_VIS_API bool useOpenXrRuntime(const std::string& manifest);

} // namespace lfs::vis::vr
