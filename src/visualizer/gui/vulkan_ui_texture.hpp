/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/ui_texture.hpp"
#include <vulkan/vulkan.h>

namespace lfs::vis {
    class VulkanContext;
}

namespace lfs::vis::gui {

    LFS_VIS_API void setUiTextureContext(VulkanContext* context);
    [[nodiscard]] LFS_VIS_API VulkanContext* getUiTextureContext();
    [[nodiscard]] LFS_VIS_API VkDescriptorSet referenceUiTextureDescriptor(const lfs::core::Tensor&);

} // namespace lfs::vis::gui
