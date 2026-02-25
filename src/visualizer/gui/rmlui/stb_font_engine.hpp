/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <RmlUi/Core/CallbackTexture.h>
#include <RmlUi/Core/FontEngineInterface.h>
#include <RmlUi/Core/FontMetrics.h>
#include <stb_truetype.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace lfs::vis::gui {

    class StbFontEngine : public Rml::FontEngineInterface {
    public:
        StbFontEngine();
        ~StbFontEngine() override;

        void Initialize() override;
        void Shutdown() override;

        bool LoadFontFace(const Rml::String& file_name, int face_index, bool fallback_face,
                          Rml::Style::FontWeight weight) override;

        bool LoadFontFace(Rml::Span<const Rml::byte> data, int face_index, const Rml::String& family,
                          Rml::Style::FontStyle style, Rml::Style::FontWeight weight,
                          bool fallback_face) override;

        Rml::FontFaceHandle GetFontFaceHandle(const Rml::String& family, Rml::Style::FontStyle style,
                                              Rml::Style::FontWeight weight, int size) override;

        Rml::FontEffectsHandle PrepareFontEffects(Rml::FontFaceHandle handle,
                                                  const Rml::FontEffectList& effects) override;

        const Rml::FontMetrics& GetFontMetrics(Rml::FontFaceHandle handle) override;

        int GetStringWidth(Rml::FontFaceHandle handle, Rml::StringView string,
                           const Rml::TextShapingContext& ctx,
                           Rml::Character prior_character) override;

        int GenerateString(Rml::RenderManager& render_manager, Rml::FontFaceHandle handle,
                           Rml::FontEffectsHandle effects, Rml::StringView string,
                           Rml::Vector2f position, Rml::ColourbPremultiplied colour, float opacity,
                           const Rml::TextShapingContext& ctx,
                           Rml::TexturedMeshList& mesh_list) override;

        int GetVersion(Rml::FontFaceHandle handle) override;
        void ReleaseFontResources() override;

    private:
        struct FontFace {
            std::string family;
            Rml::Style::FontStyle style;
            Rml::Style::FontWeight weight;
            std::vector<unsigned char> ttf_data;
            stbtt_fontinfo info;
            bool fallback = false;
        };

        struct PackedGlyph {
            float u0, v0, u1, v1;
            float x_offset, y_offset, x_offset2, y_offset2;
            float x_advance;
        };

        struct GlyphAtlas {
            FontFace* face = nullptr;
            int atlas_w = 0, atlas_h = 0;
            std::vector<unsigned char> atlas_pixels;
            Rml::CallbackTextureSource texture_source;
            std::unordered_map<uint32_t, PackedGlyph> glyphs;
        };

        struct FontInstance {
            FontFace* face = nullptr;
            int size_px = 0;
            Rml::FontMetrics metrics{};
            int atlas_w = 0, atlas_h = 0;
            std::vector<unsigned char> atlas_pixels;
            Rml::CallbackTextureSource texture_source;
            std::unordered_map<uint32_t, PackedGlyph> glyphs;
            int version = 1;
            std::vector<std::unique_ptr<GlyphAtlas>> fallback_atlases;
            std::unordered_map<uint32_t, FontFace*> pending_fallback;
        };

        FontFace* findFace(const Rml::String& family, Rml::Style::FontStyle style,
                           Rml::Style::FontWeight weight);
        FontFace* findFallbackFace(uint32_t cp) const;
        bool buildAtlas(FontInstance& inst);
        bool buildFallbackAtlas(GlyphAtlas& atlas, int size_px,
                                const std::vector<uint32_t>& codepoints);
        void flushPendingFallbacks(FontInstance& inst);

        std::vector<std::unique_ptr<FontFace>> faces_;
        std::vector<std::unique_ptr<FontInstance>> instances_;
        std::vector<FontFace*> fallback_faces_;
    };

} // namespace lfs::vis::gui
