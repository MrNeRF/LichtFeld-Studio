/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// clang-format off
#define STB_RECT_PACK_IMPLEMENTATION
#include <stb_rect_pack.h>
#define STB_TRUETYPE_IMPLEMENTATION
#include "gui/rmlui/stb_font_engine.hpp"
// clang-format on
#include "core/logger.hpp"

#include <RmlUi/Core/MeshUtilities.h>
#include <RmlUi/Core/RenderManager.h>
#include <RmlUi/Core/StringUtilities.h>
#include <RmlUi/Core/TextShapingContext.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>

namespace lfs::vis::gui {

    namespace {
        constexpr int ATLAS_INITIAL_SIZE = 512;
        constexpr int ATLAS_MAX_SIZE = 4096;
        constexpr int GLYPH_RANGE_START = 32;
        constexpr int GLYPH_RANGE_END = 0x024F;
        constexpr int GLYPH_PADDING = 1;
        constexpr unsigned char OVERSAMPLE_H = 2;
        constexpr unsigned char OVERSAMPLE_V = 2;
    } // namespace

    StbFontEngine::StbFontEngine() = default;
    StbFontEngine::~StbFontEngine() = default;

    void StbFontEngine::Initialize() {}
    void StbFontEngine::Shutdown() {
        instances_.clear();
        faces_.clear();
        fallback_faces_.clear();
    }

    bool StbFontEngine::LoadFontFace(const Rml::String& file_name, int face_index,
                                     bool fallback_face, Rml::Style::FontWeight weight) {
        std::ifstream file(file_name, std::ios::binary | std::ios::ate);
        if (!file)
            return false;

        const auto size = file.tellg();
        file.seekg(0, std::ios::beg);

        auto face = std::make_unique<FontFace>();
        face->ttf_data.resize(static_cast<size_t>(size));
        if (!file.read(reinterpret_cast<char*>(face->ttf_data.data()), size))
            return false;

        int offset = stbtt_GetFontOffsetForIndex(face->ttf_data.data(), face_index);
        if (offset < 0)
            offset = 0;

        if (!stbtt_InitFont(&face->info, face->ttf_data.data(), offset)) {
            LOG_ERROR("StbFontEngine: failed to init font from {}", file_name);
            return false;
        }

        int name_len = 0;
        const char* name = stbtt_GetFontNameString(
            &face->info, &name_len, STBTT_PLATFORM_ID_MICROSOFT, STBTT_MS_EID_UNICODE_BMP,
            STBTT_MS_LANG_ENGLISH, 4);
        if (name && name_len > 0) {
            face->family.clear();
            for (int i = 1; i < name_len; i += 2) {
                char c = name[i];
                if (c == '\0')
                    break;
                face->family.push_back(c);
            }
        }
        if (face->family.empty()) {
            face->family = std::filesystem::path(file_name).stem().string();
        }

        face->style = Rml::Style::FontStyle::Normal;
        face->weight = weight;
        face->fallback = fallback_face;

        if (fallback_face)
            fallback_faces_.push_back(face.get());

        LOG_INFO("StbFontEngine: loaded '{}' family='{}' weight={}", file_name, face->family,
                 static_cast<int>(weight));

        faces_.push_back(std::move(face));
        return true;
    }

    bool StbFontEngine::LoadFontFace(Rml::Span<const Rml::byte> data, int face_index,
                                     const Rml::String& family, Rml::Style::FontStyle style,
                                     Rml::Style::FontWeight weight, bool fallback_face) {
        auto face = std::make_unique<FontFace>();
        face->ttf_data.assign(data.begin(), data.end());

        int offset = stbtt_GetFontOffsetForIndex(face->ttf_data.data(), face_index);
        if (offset < 0)
            offset = 0;

        if (!stbtt_InitFont(&face->info, face->ttf_data.data(), offset)) {
            LOG_ERROR("StbFontEngine: failed to init font from memory");
            return false;
        }

        face->family = family;
        face->style = style;
        face->weight = weight;
        face->fallback = fallback_face;

        if (fallback_face)
            fallback_faces_.push_back(face.get());

        faces_.push_back(std::move(face));
        return true;
    }

    StbFontEngine::FontFace* StbFontEngine::findFallbackFace(uint32_t cp) const {
        for (auto* f : fallback_faces_) {
            if (stbtt_FindGlyphIndex(&f->info, static_cast<int>(cp)) != 0)
                return f;
        }
        return nullptr;
    }

    StbFontEngine::FontFace* StbFontEngine::findFace(const Rml::String& family,
                                                     Rml::Style::FontStyle style,
                                                     Rml::Style::FontWeight weight) {
        FontFace* best = nullptr;
        int best_weight_dist = INT_MAX;

        for (auto& f : faces_) {
            if (f->family != family)
                continue;
            if (f->style != style)
                continue;

            int dist =
                std::abs(static_cast<int>(f->weight) - static_cast<int>(weight));
            if (dist < best_weight_dist) {
                best_weight_dist = dist;
                best = f.get();
            }
        }

        if (!best && !fallback_faces_.empty())
            best = fallback_faces_.front();

        return best;
    }

    bool StbFontEngine::buildAtlas(FontInstance& inst) {
        assert(inst.face);

        int atlas_w = ATLAS_INITIAL_SIZE;
        int atlas_h = ATLAS_INITIAL_SIZE;

        const int num_glyphs = GLYPH_RANGE_END - GLYPH_RANGE_START + 1;

        for (;;) {
            std::vector<unsigned char> pixels(static_cast<size_t>(atlas_w * atlas_h));
            stbtt_pack_context pack_ctx;

            if (!stbtt_PackBegin(&pack_ctx, pixels.data(), atlas_w, atlas_h, 0, GLYPH_PADDING,
                                 nullptr)) {
                LOG_ERROR("StbFontEngine: PackBegin failed");
                return false;
            }

            stbtt_PackSetOversampling(&pack_ctx, OVERSAMPLE_H, OVERSAMPLE_V);

            std::vector<stbtt_packedchar> packed(static_cast<size_t>(num_glyphs));
            stbtt_pack_range range{};
            range.font_size = static_cast<float>(inst.size_px);
            range.first_unicode_codepoint_in_range = GLYPH_RANGE_START;
            range.num_chars = num_glyphs;
            range.chardata_for_range = packed.data();

            int result = stbtt_PackFontRanges(&pack_ctx, inst.face->ttf_data.data(), 0, &range, 1);
            stbtt_PackEnd(&pack_ctx);

            if (result) {
                inst.atlas_w = atlas_w;
                inst.atlas_h = atlas_h;
                inst.atlas_pixels = std::move(pixels);

                float inv_w = 1.0f / static_cast<float>(atlas_w);
                float inv_h = 1.0f / static_cast<float>(atlas_h);

                for (int i = 0; i < num_glyphs; ++i) {
                    const auto& pc = packed[static_cast<size_t>(i)];
                    if (pc.x0 == 0 && pc.x1 == 0)
                        continue;

                    auto cp = static_cast<uint32_t>(GLYPH_RANGE_START + i);
                    PackedGlyph g{};
                    g.u0 = static_cast<float>(pc.x0) * inv_w;
                    g.v0 = static_cast<float>(pc.y0) * inv_h;
                    g.u1 = static_cast<float>(pc.x1) * inv_w;
                    g.v1 = static_cast<float>(pc.y1) * inv_h;
                    g.x_offset = pc.xoff;
                    g.y_offset = pc.yoff;
                    g.x_offset2 = pc.xoff2;
                    g.y_offset2 = pc.yoff2;
                    g.x_advance = pc.xadvance;
                    inst.glyphs[cp] = g;
                }

                auto ellipsis_it = inst.glyphs.find(0x2026);
                inst.metrics.has_ellipsis = (ellipsis_it != inst.glyphs.end());

                inst.texture_source = Rml::CallbackTextureSource(
                    [&inst](const Rml::CallbackTextureInterface& iface) -> bool {
                        const int w = inst.atlas_w;
                        const int h = inst.atlas_h;
                        std::vector<Rml::byte> rgba(static_cast<size_t>(w * h * 4));
                        for (int j = 0; j < w * h; ++j) {
                            rgba[static_cast<size_t>(j * 4 + 0)] = 255;
                            rgba[static_cast<size_t>(j * 4 + 1)] = 255;
                            rgba[static_cast<size_t>(j * 4 + 2)] = 255;
                            rgba[static_cast<size_t>(j * 4 + 3)] = inst.atlas_pixels[static_cast<size_t>(j)];
                        }
                        return iface.GenerateTexture(
                            Rml::Span<const Rml::byte>(rgba.data(), rgba.size()),
                            Rml::Vector2i(w, h));
                    });

                return true;
            }

            if (atlas_w >= ATLAS_MAX_SIZE && atlas_h >= ATLAS_MAX_SIZE) {
                LOG_ERROR("StbFontEngine: atlas overflow at {}x{}", atlas_w, atlas_h);
                return false;
            }

            if (atlas_w <= atlas_h)
                atlas_w *= 2;
            else
                atlas_h *= 2;
        }
    }

    bool StbFontEngine::buildFallbackAtlas(GlyphAtlas& atlas, int size_px,
                                           const std::vector<uint32_t>& codepoints) {
        assert(atlas.face);
        if (codepoints.empty())
            return true;

        const int num_glyphs = static_cast<int>(codepoints.size());
        std::vector<int> cp_array(codepoints.begin(), codepoints.end());

        int atlas_w = ATLAS_INITIAL_SIZE;
        int atlas_h = ATLAS_INITIAL_SIZE;

        for (;;) {
            std::vector<unsigned char> pixels(static_cast<size_t>(atlas_w * atlas_h));
            stbtt_pack_context pack_ctx;

            if (!stbtt_PackBegin(&pack_ctx, pixels.data(), atlas_w, atlas_h, 0, GLYPH_PADDING,
                                 nullptr))
                return false;

            stbtt_PackSetOversampling(&pack_ctx, OVERSAMPLE_H, OVERSAMPLE_V);

            std::vector<stbtt_packedchar> packed(static_cast<size_t>(num_glyphs));
            stbtt_pack_range range{};
            range.font_size = static_cast<float>(size_px);
            range.array_of_unicode_codepoints = cp_array.data();
            range.num_chars = num_glyphs;
            range.chardata_for_range = packed.data();

            int result =
                stbtt_PackFontRanges(&pack_ctx, atlas.face->ttf_data.data(), 0, &range, 1);
            stbtt_PackEnd(&pack_ctx);

            if (result) {
                atlas.atlas_w = atlas_w;
                atlas.atlas_h = atlas_h;
                atlas.atlas_pixels = std::move(pixels);
                atlas.glyphs.clear();

                float inv_w = 1.0f / static_cast<float>(atlas_w);
                float inv_h = 1.0f / static_cast<float>(atlas_h);

                for (int i = 0; i < num_glyphs; ++i) {
                    const auto& pc = packed[static_cast<size_t>(i)];
                    if (pc.x0 == 0 && pc.x1 == 0)
                        continue;

                    PackedGlyph g{};
                    g.u0 = static_cast<float>(pc.x0) * inv_w;
                    g.v0 = static_cast<float>(pc.y0) * inv_h;
                    g.u1 = static_cast<float>(pc.x1) * inv_w;
                    g.v1 = static_cast<float>(pc.y1) * inv_h;
                    g.x_offset = pc.xoff;
                    g.y_offset = pc.yoff;
                    g.x_offset2 = pc.xoff2;
                    g.y_offset2 = pc.yoff2;
                    g.x_advance = pc.xadvance;
                    atlas.glyphs[codepoints[static_cast<size_t>(i)]] = g;
                }

                auto* raw = &atlas;
                atlas.texture_source = Rml::CallbackTextureSource(
                    [raw](const Rml::CallbackTextureInterface& iface) -> bool {
                        const int w = raw->atlas_w;
                        const int h = raw->atlas_h;
                        std::vector<Rml::byte> rgba(static_cast<size_t>(w * h * 4));
                        for (int j = 0; j < w * h; ++j) {
                            rgba[static_cast<size_t>(j * 4 + 0)] = 255;
                            rgba[static_cast<size_t>(j * 4 + 1)] = 255;
                            rgba[static_cast<size_t>(j * 4 + 2)] = 255;
                            rgba[static_cast<size_t>(j * 4 + 3)] =
                                raw->atlas_pixels[static_cast<size_t>(j)];
                        }
                        return iface.GenerateTexture(
                            Rml::Span<const Rml::byte>(rgba.data(), rgba.size()),
                            Rml::Vector2i(w, h));
                    });

                return true;
            }

            if (atlas_w >= ATLAS_MAX_SIZE && atlas_h >= ATLAS_MAX_SIZE) {
                LOG_ERROR("StbFontEngine: fallback atlas overflow at {}x{}", atlas_w, atlas_h);
                return false;
            }

            if (atlas_w <= atlas_h)
                atlas_w *= 2;
            else
                atlas_h *= 2;
        }
    }

    void StbFontEngine::flushPendingFallbacks(FontInstance& inst) {
        if (inst.pending_fallback.empty())
            return;

        std::unordered_map<FontFace*, std::vector<uint32_t>> by_face;
        for (auto& [cp, face] : inst.pending_fallback)
            by_face[face].push_back(cp);
        inst.pending_fallback.clear();

        for (auto& [face, cps] : by_face) {
            GlyphAtlas* atlas = nullptr;
            for (auto& fa : inst.fallback_atlases) {
                if (fa->face == face) {
                    atlas = fa.get();
                    break;
                }
            }

            std::vector<uint32_t> all_cps;
            if (atlas) {
                for (auto& [cp, _] : atlas->glyphs)
                    all_cps.push_back(cp);
            }
            for (auto cp : cps)
                all_cps.push_back(cp);

            std::sort(all_cps.begin(), all_cps.end());
            all_cps.erase(std::unique(all_cps.begin(), all_cps.end()), all_cps.end());

            if (!atlas) {
                auto new_atlas = std::make_unique<GlyphAtlas>();
                new_atlas->face = face;
                atlas = new_atlas.get();
                inst.fallback_atlases.push_back(std::move(new_atlas));
            }

            buildFallbackAtlas(*atlas, inst.size_px, all_cps);
        }

        ++inst.version;
    }

    Rml::FontFaceHandle StbFontEngine::GetFontFaceHandle(const Rml::String& family,
                                                         Rml::Style::FontStyle style,
                                                         Rml::Style::FontWeight weight, int size) {
        for (auto& inst : instances_) {
            if (inst->face && inst->face->family == family && inst->face->style == style &&
                inst->size_px == size) {
                int wdist = std::abs(static_cast<int>(inst->face->weight) -
                                     static_cast<int>(weight));
                if (wdist == 0)
                    return reinterpret_cast<Rml::FontFaceHandle>(inst.get());
            }
        }

        FontFace* face = findFace(family, style, weight);
        if (!face) {
            LOG_WARN("StbFontEngine: no face for family='{}' size={}", family, size);
            return 0;
        }

        auto inst = std::make_unique<FontInstance>();
        inst->face = face;
        inst->size_px = size;

        float scale = stbtt_ScaleForPixelHeight(&face->info, static_cast<float>(size));

        int ascent_raw = 0, descent_raw = 0, line_gap_raw = 0;
        stbtt_GetFontVMetrics(&face->info, &ascent_raw, &descent_raw, &line_gap_raw);

        inst->metrics.size = size;
        inst->metrics.ascent = std::round(static_cast<float>(ascent_raw) * scale);
        inst->metrics.descent = std::round(std::abs(static_cast<float>(descent_raw) * scale));
        inst->metrics.line_spacing =
            std::round(static_cast<float>(ascent_raw - descent_raw + line_gap_raw) * scale);

        int x_advance = 0, x_lsb = 0;
        stbtt_GetCodepointHMetrics(&face->info, 'x', &x_advance, &x_lsb);
        int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
        stbtt_GetCodepointBitmapBox(&face->info, 'x', scale, scale, &x0, &y0, &x1, &y1);
        inst->metrics.x_height = static_cast<float>(y1 - y0);
        if (inst->metrics.x_height <= 0)
            inst->metrics.x_height = inst->metrics.ascent * 0.5f;

        inst->metrics.underline_position = std::round(inst->metrics.ascent * 0.15f);
        inst->metrics.underline_thickness = std::max(1.0f, std::round(static_cast<float>(size) / 14.0f));

        if (!buildAtlas(*inst)) {
            LOG_ERROR("StbFontEngine: failed to build atlas for '{}' size={}", family, size);
            return 0;
        }

        auto* ptr = inst.get();
        instances_.push_back(std::move(inst));
        return reinterpret_cast<Rml::FontFaceHandle>(ptr);
    }

    Rml::FontEffectsHandle StbFontEngine::PrepareFontEffects(Rml::FontFaceHandle /*handle*/,
                                                             const Rml::FontEffectList& /*effects*/) {
        return 0;
    }

    const Rml::FontMetrics& StbFontEngine::GetFontMetrics(Rml::FontFaceHandle handle) {
        assert(handle != 0);
        auto* inst = reinterpret_cast<FontInstance*>(handle);
        return inst->metrics;
    }

    int StbFontEngine::GetStringWidth(Rml::FontFaceHandle handle, Rml::StringView str,
                                      const Rml::TextShapingContext& ctx,
                                      Rml::Character prior_character) {
        assert(handle != 0);
        auto* inst = reinterpret_cast<FontInstance*>(handle);

        float width = 0.0f;
        float scale =
            stbtt_ScaleForPixelHeight(&inst->face->info, static_cast<float>(inst->size_px));
        uint32_t prev_cp = static_cast<uint32_t>(prior_character);

        const char* p = str.begin();
        const char* end = str.end();

        while (p < end) {
            uint32_t cp = 0;
            if ((*p & 0x80) == 0) {
                cp = static_cast<uint32_t>(*p++);
            } else if ((*p & 0xE0) == 0xC0) {
                cp = static_cast<uint32_t>(*p++ & 0x1F) << 6;
                if (p < end)
                    cp |= static_cast<uint32_t>(*p++ & 0x3F);
            } else if ((*p & 0xF0) == 0xE0) {
                cp = static_cast<uint32_t>(*p++ & 0x0F) << 12;
                if (p < end)
                    cp |= static_cast<uint32_t>(*p++ & 0x3F) << 6;
                if (p < end)
                    cp |= static_cast<uint32_t>(*p++ & 0x3F);
            } else if ((*p & 0xF8) == 0xF0) {
                cp = static_cast<uint32_t>(*p++ & 0x07) << 18;
                if (p < end)
                    cp |= static_cast<uint32_t>(*p++ & 0x3F) << 12;
                if (p < end)
                    cp |= static_cast<uint32_t>(*p++ & 0x3F) << 6;
                if (p < end)
                    cp |= static_cast<uint32_t>(*p++ & 0x3F);
            } else {
                ++p;
                continue;
            }

            auto it = inst->glyphs.find(cp);
            if (it != inst->glyphs.end()) {
                width += it->second.x_advance;
            } else {
                bool found = false;
                for (auto& fa : inst->fallback_atlases) {
                    auto fb_it = fa->glyphs.find(cp);
                    if (fb_it != fa->glyphs.end()) {
                        width += fb_it->second.x_advance;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    FontFace* fb_face = findFallbackFace(cp);
                    if (fb_face) {
                        inst->pending_fallback[cp] = fb_face;
                        float fb_scale = stbtt_ScaleForPixelHeight(
                            &fb_face->info, static_cast<float>(inst->size_px));
                        int advance = 0, lsb = 0;
                        stbtt_GetCodepointHMetrics(&fb_face->info, static_cast<int>(cp),
                                                   &advance, &lsb);
                        width += static_cast<float>(advance) * fb_scale;
                    } else {
                        int advance = 0, lsb = 0;
                        stbtt_GetCodepointHMetrics(&inst->face->info, static_cast<int>(cp),
                                                   &advance, &lsb);
                        width += static_cast<float>(advance) * scale;
                    }
                }
            }

            if (prev_cp != 0 && ctx.font_kerning != Rml::Style::FontKerning::None) {
                int kern = stbtt_GetCodepointKernAdvance(&inst->face->info,
                                                         static_cast<int>(prev_cp),
                                                         static_cast<int>(cp));
                width += static_cast<float>(kern) * scale;
            }

            width += ctx.letter_spacing;
            prev_cp = cp;
        }

        return static_cast<int>(std::ceil(width));
    }

    int StbFontEngine::GenerateString(Rml::RenderManager& render_manager,
                                      Rml::FontFaceHandle handle,
                                      Rml::FontEffectsHandle /*effects*/, Rml::StringView str,
                                      Rml::Vector2f position, Rml::ColourbPremultiplied colour,
                                      float opacity, const Rml::TextShapingContext& ctx,
                                      Rml::TexturedMeshList& mesh_list) {
        assert(handle != 0);
        auto* inst = reinterpret_cast<FontInstance*>(handle);

        if (str.empty())
            return 0;

        flushPendingFallbacks(*inst);

        Rml::ColourbPremultiplied final_colour = colour;
        if (opacity < 1.0f) {
            final_colour.alpha =
                static_cast<Rml::byte>(static_cast<float>(colour.alpha) * opacity);
            final_colour.red =
                static_cast<Rml::byte>(static_cast<float>(colour.red) * opacity);
            final_colour.green =
                static_cast<Rml::byte>(static_cast<float>(colour.green) * opacity);
            final_colour.blue =
                static_cast<Rml::byte>(static_cast<float>(colour.blue) * opacity);
        }

        Rml::Texture texture = inst->texture_source.GetTexture(render_manager);
        if (!texture)
            return 0;

        Rml::Mesh primary_mesh;
        std::unordered_map<size_t, Rml::Mesh> fb_meshes;

        float cursor_x = position.x;
        float baseline_y = position.y;

        float scale =
            stbtt_ScaleForPixelHeight(&inst->face->info, static_cast<float>(inst->size_px));
        uint32_t prev_cp = 0;

        const char* p = str.begin();
        const char* end = str.end();
        int glyph_count = 0;

        while (p < end) {
            uint32_t cp = 0;
            if ((*p & 0x80) == 0) {
                cp = static_cast<uint32_t>(*p++);
            } else if ((*p & 0xE0) == 0xC0) {
                cp = static_cast<uint32_t>(*p++ & 0x1F) << 6;
                if (p < end)
                    cp |= static_cast<uint32_t>(*p++ & 0x3F);
            } else if ((*p & 0xF0) == 0xE0) {
                cp = static_cast<uint32_t>(*p++ & 0x0F) << 12;
                if (p < end)
                    cp |= static_cast<uint32_t>(*p++ & 0x3F) << 6;
                if (p < end)
                    cp |= static_cast<uint32_t>(*p++ & 0x3F);
            } else if ((*p & 0xF8) == 0xF0) {
                cp = static_cast<uint32_t>(*p++ & 0x07) << 18;
                if (p < end)
                    cp |= static_cast<uint32_t>(*p++ & 0x3F) << 12;
                if (p < end)
                    cp |= static_cast<uint32_t>(*p++ & 0x3F) << 6;
                if (p < end)
                    cp |= static_cast<uint32_t>(*p++ & 0x3F);
            } else {
                ++p;
                continue;
            }

            if (prev_cp != 0 && ctx.font_kerning != Rml::Style::FontKerning::None) {
                int kern = stbtt_GetCodepointKernAdvance(&inst->face->info,
                                                         static_cast<int>(prev_cp),
                                                         static_cast<int>(cp));
                cursor_x += static_cast<float>(kern) * scale;
            }

            auto emit_quad = [&](Rml::Mesh& m, const PackedGlyph& g) {
                int base_idx = static_cast<int>(m.vertices.size());
                m.vertices.push_back(Rml::Vertex{
                    Rml::Vector2f(cursor_x + g.x_offset, baseline_y + g.y_offset),
                    final_colour, Rml::Vector2f(g.u0, g.v0)});
                m.vertices.push_back(Rml::Vertex{
                    Rml::Vector2f(cursor_x + g.x_offset2, baseline_y + g.y_offset),
                    final_colour, Rml::Vector2f(g.u1, g.v0)});
                m.vertices.push_back(Rml::Vertex{
                    Rml::Vector2f(cursor_x + g.x_offset2, baseline_y + g.y_offset2),
                    final_colour, Rml::Vector2f(g.u1, g.v1)});
                m.vertices.push_back(Rml::Vertex{
                    Rml::Vector2f(cursor_x + g.x_offset, baseline_y + g.y_offset2),
                    final_colour, Rml::Vector2f(g.u0, g.v1)});
                m.indices.push_back(base_idx);
                m.indices.push_back(base_idx + 1);
                m.indices.push_back(base_idx + 2);
                m.indices.push_back(base_idx);
                m.indices.push_back(base_idx + 2);
                m.indices.push_back(base_idx + 3);
            };

            auto it = inst->glyphs.find(cp);
            if (it != inst->glyphs.end()) {
                emit_quad(primary_mesh, it->second);
                cursor_x += it->second.x_advance;
                ++glyph_count;
            } else {
                bool found = false;
                for (size_t fi = 0; fi < inst->fallback_atlases.size(); ++fi) {
                    auto fb_it = inst->fallback_atlases[fi]->glyphs.find(cp);
                    if (fb_it != inst->fallback_atlases[fi]->glyphs.end()) {
                        emit_quad(fb_meshes[fi], fb_it->second);
                        cursor_x += fb_it->second.x_advance;
                        ++glyph_count;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    int advance = 0, lsb = 0;
                    stbtt_GetCodepointHMetrics(&inst->face->info, static_cast<int>(cp), &advance,
                                               &lsb);
                    cursor_x += static_cast<float>(advance) * scale;
                }
            }

            cursor_x += ctx.letter_spacing;
            prev_cp = cp;
        }

        if (primary_mesh)
            mesh_list.push_back(Rml::TexturedMesh{std::move(primary_mesh), texture});

        for (auto& [fi, mesh] : fb_meshes) {
            if (mesh) {
                Rml::Texture fb_tex =
                    inst->fallback_atlases[fi]->texture_source.GetTexture(render_manager);
                if (fb_tex)
                    mesh_list.push_back(Rml::TexturedMesh{std::move(mesh), fb_tex});
            }
        }

        return static_cast<int>(std::ceil(cursor_x - position.x));
    }

    int StbFontEngine::GetVersion(Rml::FontFaceHandle handle) {
        if (handle == 0)
            return 0;
        return reinterpret_cast<FontInstance*>(handle)->version;
    }

    void StbFontEngine::ReleaseFontResources() {}

} // namespace lfs::vis::gui
