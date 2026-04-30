/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include <cstdint>

namespace lfs::core {
    class Tensor;
}

namespace lfs::vis {
    class VulkanContext;
}

namespace lfs::vis::gui {

    LFS_VIS_API void setVulkanUiTextureContext(VulkanContext* context);
    [[nodiscard]] LFS_VIS_API VulkanContext* getVulkanUiTextureContext();

    class LFS_VIS_API VulkanUiTexture {
    public:
        VulkanUiTexture() = default;
        ~VulkanUiTexture();

        VulkanUiTexture(const VulkanUiTexture&) = delete;
        VulkanUiTexture& operator=(const VulkanUiTexture&) = delete;

        VulkanUiTexture(VulkanUiTexture&& other) noexcept;
        VulkanUiTexture& operator=(VulkanUiTexture&& other) noexcept;

        [[nodiscard]] bool upload(const std::uint8_t* pixels, int width, int height, int channels);
        [[nodiscard]] bool uploadRegion(const std::uint8_t* pixels,
                                         int texture_width,
                                         int texture_height,
                                         int x,
                                         int y,
                                         int width,
                                         int height,
                                         int channels);
        [[nodiscard]] bool upload(const lfs::core::Tensor& image, int expected_width, int expected_height);
        [[nodiscard]] std::uintptr_t textureId() const;
        [[nodiscard]] bool valid() const;
        void reset();

    private:
        struct Impl;
        Impl* impl_ = nullptr;
    };

} // namespace lfs::vis::gui
