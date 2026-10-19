/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include <cstdint>
#include <span>
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

    class ScopedVulkanUiTextureContext {
    public:
        explicit ScopedVulkanUiTextureContext(VulkanContext* target)
            : previous_(getVulkanUiTextureContext()), changed_(target && target != previous_) {
            if (changed_)
                setVulkanUiTextureContext(target);
        }
        ~ScopedVulkanUiTextureContext() {
            if (changed_)
                setVulkanUiTextureContext(previous_);
        }
        ScopedVulkanUiTextureContext(const ScopedVulkanUiTextureContext&) = delete;
        ScopedVulkanUiTextureContext& operator=(const ScopedVulkanUiTextureContext&) = delete;

    private:
        VulkanContext* previous_;
        bool changed_;
    };

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
        struct Region {
            const std::uint8_t* pixels = nullptr;
            int texture_width = 0;
            int texture_height = 0;
            int x = 0;
            int y = 0;
            int width = 0;
            int height = 0;
            int channels = 0;
        };
        [[nodiscard]] bool uploadRegions(std::span<const Region> regions);
        // flip_y: vertically mirror the image during upload. Set when the source tensor uses the
        // rasterizer's OpenGL (bottom-left origin) convention but the Vulkan view samples
        // top-left, e.g., the sequencer's RmlUi-bound preview textures.
        [[nodiscard]] bool upload(const lfs::core::Tensor& image,
                                  int expected_width,
                                  int expected_height,
                                  bool flip_y = false);
        [[nodiscard]] std::uintptr_t textureId() const;
        [[nodiscard]] bool valid() const;
        // Cached Rml::Image source, unique to this image lifetime even if handles are recycled.
        // Same-size uploads preserve it. The reference is borrowed until the image lifetime
        // or requested display dimensions change; copy it if it must outlive those changes.
        [[nodiscard]] const std::string& rmlSrcUrl(int width, int height) const;
        void reset();

    private:
        struct Impl;
        Impl* impl_ = nullptr;
    };

} // namespace lfs::vis::gui
