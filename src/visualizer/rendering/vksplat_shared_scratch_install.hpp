/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

namespace lfs::vis {

    // Called for a retained viewer block BEFORE capacity reuse or growth. B3
    // can detach the arena backing without updating the renderer's cached flag.
    // Keep the ownership decision independent of Vulkan import/allocation work.
    template <typename IsInstalled, typename TryInstall>
    [[nodiscard]] bool ensureRetainedSharedScratchInstalled(
        bool& installed, IsInstalled&& is_installed, TryInstall&& try_install) {
        installed = is_installed();
        if (!installed) {
            if (!try_install()) {
                return false;
            }
            installed = true;
        }
        return true;
    }

} // namespace lfs::vis
