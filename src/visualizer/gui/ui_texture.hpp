/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/error.hpp"
#include "core/export.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace lfs::core {
    class Tensor;
}

namespace lfs::vis {
    class GraphicsContext;
}
namespace lfs::vis::gui {
    LFS_VIS_API void connectUiTextureGraphics(GraphicsContext* context);

    // Expands `count` gray, gray+alpha, RGB or RGBA pixels to RGBA8.
    inline void expandToRgba8(const std::uint8_t* source, std::uint8_t* target,
                              const std::size_t count, const int channels) {
        for (std::size_t i = 0; i < count; ++i, source += channels, target += 4) {
            const bool gray = channels <= 2;
            target[0] = source[0];
            target[1] = gray ? source[0] : source[1];
            target[2] = gray ? source[0] : source[2];
            target[3] = channels == 2 ? source[1] : channels == 4 ? source[3]
                                                                  : 255;
        }
    }

    // Backend-neutral UI texture. The active graphics backend owns the image,
    // upload scheduling, RmlUi URL and retirement policy behind this facade.
    class LFS_VIS_API UiTexture {
    public:
        UiTexture() = default;
        ~UiTexture();

        UiTexture(const UiTexture&) = delete;
        UiTexture& operator=(const UiTexture&) = delete;
        UiTexture(UiTexture&& other) noexcept;
        UiTexture& operator=(UiTexture&& other) noexcept;

        [[nodiscard]] bool upload(const std::uint8_t* pixels, int width, int height,
                                  int channels);
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
        [[nodiscard]] bool upload(const lfs::core::Tensor& image,
                                  int expected_width,
                                  int expected_height,
                                  bool flip_y = false);
        // Preserves Float32 RGBA values for tensor-rendered overlays.
        [[nodiscard]] lfs::Result<void> uploadLinearRgba(const lfs::core::Tensor& image);
        [[nodiscard]] std::shared_ptr<const lfs::core::Tensor> image() const;
        [[nodiscard]] std::uintptr_t textureId() const;
        [[nodiscard]] bool valid() const;
        [[nodiscard]] std::string rmlSrcUrl(int width, int height) const;
        void reset();

        // Public so the separately compiled backend can retire resources
        // without exposing its native handles.
        struct Impl;

    private:
        Impl* impl_ = nullptr;
    };

    // Compatibility at existing UI-command boundaries. New viewport descriptors
    // retain the tensor itself, never the legacy Rml texture identifier.
    [[nodiscard]] LFS_VIS_API std::shared_ptr<const lfs::core::Tensor>
    uiTextureImage(std::uintptr_t legacy_id);
    // Changes whenever a UiTexture publishes new content; cached UI that drew
    // a texture re-renders when it differs.
    [[nodiscard]] LFS_VIS_API std::uint64_t uiTextureGeneration();

} // namespace lfs::vis::gui
