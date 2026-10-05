/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/export.hpp"

#include <RmlUi/Core/RenderInterface.h>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace lfs::vis {
    class GraphicsContext;
    struct GraphicsFrame;
} // namespace lfs::vis

namespace lfs::vis::gui {

    struct UiPixelRect {
        int x = 0;
        int y = 0;
        std::uint32_t width = 0;
        std::uint32_t height = 0;
    };

    struct UiFrostedGlassRegion {
        float x = 0.0f;
        float y = 0.0f;
        float width = 0.0f;
        float height = 0.0f;
        float radius = 0.0f;
    };

    struct UiRendererMemoryStatistics {
        std::size_t block_bytes = 0;
        std::size_t allocation_bytes = 0;
    };

    // UI frame contract. Implementations keep native presentation access private.
    class LFS_VIS_API UiRenderer : public Rml::RenderInterface {
    public:
        ~UiRenderer() override = default;

        [[nodiscard]] virtual bool initialize(GraphicsContext& graphics) = 0;
        virtual void shutdown() = 0;
        [[nodiscard]] virtual bool beginFrame(const GraphicsFrame& frame) = 0;
        virtual void endFrame() = 0;

        virtual void resetContextRenderState() = 0;
        virtual void setContextOffset(float x, float y) = 0;
        virtual void setContextClipRect(float x1, float y1, float x2, float y2) = 0;
        virtual void renderTextureQuad(Rml::TextureHandle texture, float x, float y,
                                       float width, float height) = 0;
        [[nodiscard]] virtual bool
        renderFrostedGlass(std::span<const UiFrostedGlassRegion> regions) = 0;
        virtual void beginCacheCapture(int x, int y, int width, int height) = 0;
        virtual void endCacheCapture() = 0;
        [[nodiscard]] virtual Rml::TextureHandle
        saveLayerRegionAsTexture(UiPixelRect region,
                                 Rml::TextureHandle reuse_texture) = 0;
        virtual void setTextureDebugName(Rml::TextureHandle texture,
                                         std::string_view name) const = 0;
        [[nodiscard]] virtual std::uint64_t previewTextureGeneration() const = 0;
        [[nodiscard]] virtual bool currentContextUsedPreviewTexture() const = 0;
        [[nodiscard]] virtual UiRendererMemoryStatistics memoryStatistics() const = 0;
    };

} // namespace lfs::vis::gui
