/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "gui/ui_renderer.hpp"

#include <memory>

namespace lfs::vis::gui {
    // Backend-neutral RmlUi renderer. Geometry, textures, clip masks and layers
    // are tensors; all raster and filter work is authored once in Slang.
    class TensorRmlUiRenderer final : public UiRenderer {
    public:
        TensorRmlUiRenderer();
        ~TensorRmlUiRenderer() override;

        bool initialize(GraphicsContext& graphics) override;
        void shutdown() override;
        bool beginFrame(const GraphicsFrame& frame) override;
        void endFrame() override;
        void resetContextRenderState() override;
        void setContextOffset(float x, float y) override;
        void setContextClipRect(float x1, float y1, float x2, float y2) override;
        void renderTextureQuad(Rml::TextureHandle texture, float x, float y,
                               float width, float height) override;
        bool renderFrostedGlass(std::span<const UiFrostedGlassRegion> regions) override;
        void beginCacheCapture(int x, int y, int width, int height) override;
        void endCacheCapture() override;
        Rml::TextureHandle saveLayerRegionAsTexture(UiPixelRect region,
                                                    Rml::TextureHandle reuse) override;
        void setTextureDebugName(Rml::TextureHandle, std::string_view) const override {}
        uint64_t previewTextureGeneration() const override;
        bool currentContextUsedPreviewTexture() const override;
        UiRendererMemoryStatistics memoryStatistics() const override;

        Rml::CompiledGeometryHandle CompileGeometry(Rml::Span<const Rml::Vertex> vertices,
                                                    Rml::Span<const int> indices) override;
        void RenderGeometry(Rml::CompiledGeometryHandle geometry,
                            Rml::Vector2f translation,
                            Rml::TextureHandle texture) override;
        void ReleaseGeometry(Rml::CompiledGeometryHandle geometry) override;
        Rml::TextureHandle LoadTexture(Rml::Vector2i& dimensions,
                                       const Rml::String& source) override;
        Rml::TextureHandle GenerateTexture(Rml::Span<const Rml::byte> source,
                                           Rml::Vector2i dimensions) override;
        void ReleaseTexture(Rml::TextureHandle texture) override;
        void EnableScissorRegion(bool enable) override;
        void SetScissorRegion(Rml::Rectanglei region) override;
        void EnableClipMask(bool enable) override;
        void RenderToClipMask(Rml::ClipMaskOperation operation,
                              Rml::CompiledGeometryHandle geometry,
                              Rml::Vector2f translation) override;
        Rml::LayerHandle PushLayer() override;
        void CompositeLayers(Rml::LayerHandle source, Rml::LayerHandle destination,
                             Rml::BlendMode blend_mode,
                             Rml::Span<const Rml::CompiledFilterHandle> filters) override;
        void PopLayer() override;
        Rml::TextureHandle SaveLayerAsTexture() override;
        void SetTransform(const Rml::Matrix4f* transform) override;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace lfs::vis::gui
