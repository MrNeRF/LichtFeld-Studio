/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering_manager.hpp"

namespace lfs::vis {

    void RenderingManager::renderFrame(const RenderContext& context) {
        (void)renderVulkanFrame(context);
    }

} // namespace lfs::vis
