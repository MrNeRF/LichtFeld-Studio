/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include <cstdint>
#include <string>

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
        // flip_y: vertically mirror the image during upload. Set when the source tensor uses the
        // rasterizer's OpenGL (bottom-left origin) convention but the Vulkan view samples
        // top-left, e.g., the sequencer's RmlUi-bound preview textures.
        [[nodiscard]] bool upload(const lfs::core::Tensor& image,
                                  int expected_width,
                                  int expected_height,
                                  bool flip_y = false);
        [[nodiscard]] bool uploadCudaTensor(const lfs::core::Tensor& image,
                                            int expected_width,
                                            int expected_height,
                                            bool flip_y = false);
        [[nodiscard]] std::uintptr_t textureId() const;
        [[nodiscard]] bool valid() const;
        // Build a Rml::Image src URL referencing this texture's image view and sampler.
        // The returned URL is stable as long as the underlying VkImage is not torn down
        // (a same-size re-upload preserves the view; a resize destroys and recreates it).
        [[nodiscard]] std::string rmlSrcUrl(int width, int height) const;
        void reset();

    private:
        struct Impl;
        Impl* impl_ = nullptr;
    };

} // namespace lfs::vis::gui
