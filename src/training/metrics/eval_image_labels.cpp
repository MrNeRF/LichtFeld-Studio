/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "eval_image_labels.hpp"
#include "core/executable_path.hpp"
#include "core/logger.hpp"
#include "core/native_fs.hpp"
#include "core/path_utils.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#define STB_TRUETYPE_IMPLEMENTATION
#define STBTT_STATIC
#include <stb_truetype.h>

namespace lfs::training {

    namespace {
        constexpr float kBandValue = 0.12f;
        constexpr float kTextValue = 0.95f;

        struct CaptionFont {
            std::vector<unsigned char> bytes;
            stbtt_fontinfo info{};
        };

        const CaptionFont* load_caption_font(const std::filesystem::path& path) {
            static std::mutex mutex;
            static std::map<std::filesystem::path, std::unique_ptr<CaptionFont>> fonts;
            const std::lock_guard lock(mutex);
            if (const auto found = fonts.find(path); found != fonts.end())
                return found->second.get();

            auto font = std::make_unique<CaptionFont>();
            std::ifstream in;
            if (lfs::core::open_file_for_read(path, std::ios::in | std::ios::binary, in))
                font->bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
            if (font->bytes.empty() ||
                !stbtt_InitFont(&font->info, font->bytes.data(), stbtt_GetFontOffsetForIndex(font->bytes.data(), 0))) {
                LOG_WARN("Eval: caption font '{}' is unreadable; evaluation images are saved without captions",
                         lfs::core::path_to_utf8(path));
                font.reset();
            }
            return (fonts[path] = std::move(font)).get();
        }
    } // namespace

    std::filesystem::path default_caption_font() {
        return lfs::core::getFontsDir() / "Inter-SemiBold.ttf";
    }

    int caption_band_height(const int tile_height) {
        return std::clamp(tile_height / 20, 20, 128);
    }

    lfs::core::Tensor caption_band(const std::span<const ImageCaption> captions, const int width, const int height,
                                   const std::filesystem::path& font_path) {
        assert(width > 0 && height > 0);
        std::vector<float> pixels(static_cast<size_t>(3) * height * width, kBandValue);
        if (const auto* font = load_caption_font(font_path)) {
            const float scale = stbtt_ScaleForPixelHeight(&font->info, 0.6f * static_cast<float>(height));
            int ascent = 0, descent = 0, line_gap = 0;
            stbtt_GetFontVMetrics(&font->info, &ascent, &descent, &line_gap);
            const int baseline = static_cast<int>(std::lround(
                0.5f * (static_cast<float>(height) - static_cast<float>(ascent - descent) * scale) +
                static_cast<float>(ascent) * scale));
            const float padding = 0.3f * static_cast<float>(height);
            std::vector<unsigned char> glyph;
            for (const auto& caption : captions) {
                int limit = width;
                for (const auto& next : captions) {
                    if (next.x > caption.x)
                        limit = std::min(limit, next.x - static_cast<int>(padding));
                }
                float pen = static_cast<float>(caption.x) + padding;
                for (size_t i = 0; i < caption.text.size() && pen < static_cast<float>(limit); ++i) {
                    const int codepoint = static_cast<unsigned char>(caption.text[i]);
                    const float shift = pen - std::floor(pen);
                    int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
                    stbtt_GetCodepointBitmapBoxSubpixel(&font->info, codepoint, scale, scale, shift, 0.0f,
                                                        &x0, &y0, &x1, &y1);
                    const int glyph_width = x1 - x0;
                    const int glyph_height = y1 - y0;
                    if (glyph_width > 0 && glyph_height > 0) {
                        glyph.assign(static_cast<size_t>(glyph_width) * glyph_height, 0);
                        stbtt_MakeCodepointBitmapSubpixel(&font->info, glyph.data(), glyph_width, glyph_height,
                                                          glyph_width, scale, scale, shift, 0.0f, codepoint);
                        const int left = static_cast<int>(std::floor(pen)) + x0;
                        const int top = baseline + y0;
                        for (int gy = 0; gy < glyph_height; ++gy) {
                            const int y = top + gy;
                            if (y < 0 || y >= height)
                                continue;
                            for (int gx = 0; gx < glyph_width; ++gx) {
                                const int x = left + gx;
                                if (x < 0 || x >= limit)
                                    continue;
                                const float coverage =
                                    static_cast<float>(glyph[static_cast<size_t>(gy) * glyph_width + gx]) / 255.0f;
                                for (int c = 0; c < 3; ++c) {
                                    float& value = pixels[(static_cast<size_t>(c) * height + y) * width + x];
                                    value += (kTextValue - value) * coverage;
                                }
                            }
                        }
                    }
                    int advance = 0, left_bearing = 0;
                    stbtt_GetCodepointHMetrics(&font->info, codepoint, &advance, &left_bearing);
                    pen += static_cast<float>(advance) * scale;
                    if (i + 1 < caption.text.size())
                        pen += scale * static_cast<float>(stbtt_GetCodepointKernAdvance(
                                           &font->info, codepoint, static_cast<unsigned char>(caption.text[i + 1])));
                }
            }
        }
        return lfs::core::Tensor::from_vector(
            pixels, lfs::core::TensorShape({size_t{3}, static_cast<size_t>(height), static_cast<size_t>(width)}),
            lfs::core::Device::CUDA);
    }

} // namespace lfs::training
