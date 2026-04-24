/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/editor/zep_rml_display.hpp"

#include <RmlUi/Core/Core.h>
#include <RmlUi/Core/FontEngineInterface.h>
#include <RmlUi/Core/Geometry.h>
#include <RmlUi/Core/Mesh.h>
#include <RmlUi/Core/RenderManager.h>
#include <RmlUi/Core/StringUtilities.h>

#include <algorithm>
#include <cmath>
#include <cstring>

namespace lfs::vis::editor {

    namespace {
        constexpr float kUiScale = 0.95f;
        constexpr float kHeading1Scale = 1.35f;
        constexpr float kHeading2Scale = 1.20f;
        constexpr float kHeading3Scale = 1.10f;

        Rml::ColourbPremultiplied toRmlColor(const Zep::NVec4f& color) {
            const auto to_byte = [](float value) {
                return static_cast<Rml::byte>(std::clamp(value, 0.0f, 1.0f) * 255.0f);
            };
            const float alpha = std::clamp(color.w, 0.0f, 1.0f);
            return {
                to_byte(color.x * alpha),
                to_byte(color.y * alpha),
                to_byte(color.z * alpha),
                to_byte(alpha),
            };
        }

        void addQuad(Rml::Mesh& mesh,
                     float x0,
                     float y0,
                     float x1,
                     float y1,
                     const Rml::ColourbPremultiplied& color) {
            const int base = static_cast<int>(mesh.vertices.size());
            mesh.vertices.push_back({{x0, y0}, color, {0, 0}});
            mesh.vertices.push_back({{x1, y0}, color, {0, 0}});
            mesh.vertices.push_back({{x1, y1}, color, {0, 0}});
            mesh.vertices.push_back({{x0, y1}, color, {0, 0}});
            mesh.indices.insert(mesh.indices.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
        }

        const Rml::String& emptyLanguage() {
            static const Rml::String language;
            return language;
        }
    } // namespace

    class ZepDisplay_Rml::Font final : public Zep::ZepFont {
    public:
        Font(ZepDisplay_Rml& display, const float pixel_height)
            : Zep::ZepFont(display) {
            SetPixelHeight(std::max(1, static_cast<int>(std::lround(pixel_height))));
        }

        void SetPixelHeight(const int pixel_height) override {
            InvalidateCharCache();
            m_pixelHeight = std::max(1, pixel_height);
            face_handle_ = {};
            effects_handle_ = {};
            ensureFontHandles();
        }

        Zep::NVec2f GetTextSize(const uint8_t* begin, const uint8_t* end = nullptr) const override {
            if (!begin)
                return {};
            if (!end)
                end = begin + std::strlen(reinterpret_cast<const char*>(begin));
            if (end <= begin)
                return {};

            const auto byte_count = static_cast<int>(end - begin);
            ensureFontHandles();
            if (!face_handle_) {
                return {std::max(1.0f, byte_count * m_pixelHeight * 0.58f),
                        static_cast<float>(m_pixelHeight)};
            }

            auto* font_engine = Rml::GetFontEngineInterface();
            if (!font_engine)
                return {std::max(1.0f, byte_count * m_pixelHeight * 0.58f),
                        static_cast<float>(m_pixelHeight)};

            Rml::TextShapingContext shaping{emptyLanguage()};
            const int width = font_engine->GetStringWidth(
                face_handle_,
                Rml::StringView(reinterpret_cast<const char*>(begin), reinterpret_cast<const char*>(end)),
                shaping);
            return {static_cast<float>(std::max(width, 1)), static_cast<float>(m_pixelHeight)};
        }

        Rml::FontFaceHandle faceHandle() const {
            ensureFontHandles();
            return face_handle_;
        }
        Rml::FontEffectsHandle effectsHandle() const {
            ensureFontHandles();
            return effects_handle_;
        }

    private:
        void ensureFontHandles() const {
            if (face_handle_)
                return;

            face_handle_ = {};
            effects_handle_ = {};

            auto* font_engine = Rml::GetFontEngineInterface();
            if (!font_engine)
                return;

            face_handle_ = font_engine->GetFontFaceHandle(
                "JetBrains Mono",
                Rml::Style::FontStyle::Normal,
                Rml::Style::FontWeight::Normal,
                m_pixelHeight);
            if (!face_handle_) {
                face_handle_ = font_engine->GetFontFaceHandle(
                    "Inter",
                    Rml::Style::FontStyle::Normal,
                    Rml::Style::FontWeight::Normal,
                    m_pixelHeight);
            }
            if (face_handle_) {
                const Rml::FontEffectList effects;
                effects_handle_ = font_engine->PrepareFontEffects(face_handle_, effects);
            }
        }

        mutable Rml::FontFaceHandle face_handle_ = {};
        mutable Rml::FontEffectsHandle effects_handle_ = {};
    };

    ZepDisplay_Rml::ZepDisplay_Rml() : Zep::ZepDisplay() {
        SetPixelScale(Zep::NVec2f(1.0f));
        ensureDefaultFonts();
    }

    void ZepDisplay_Rml::beginFrame(Rml::Element& element) {
        element_ = &element;
        ensureDefaultFonts();
    }

    void ZepDisplay_Rml::endFrame() {
        element_ = nullptr;
        clip_rect_ = {};
    }

    void ZepDisplay_Rml::setBaseFontSize(const float pixel_height) {
        const float next = std::max(1.0f, pixel_height);
        if (std::abs(next - base_font_size_) < 0.5f)
            return;

        base_font_size_ = next;
        m_fonts = {};
        ensureDefaultFonts();
        SetLayoutDirty(true);
    }

    void ZepDisplay_Rml::ensureDefaultFonts() {
        auto make_font = [this](const float scale) {
            return std::make_shared<Font>(*this, base_font_size_ * scale);
        };

        if (!m_fonts[static_cast<int>(Zep::ZepTextType::UI)])
            m_fonts[static_cast<int>(Zep::ZepTextType::UI)] = make_font(kUiScale);
        if (!m_fonts[static_cast<int>(Zep::ZepTextType::Text)])
            m_fonts[static_cast<int>(Zep::ZepTextType::Text)] = make_font(1.0f);
        if (!m_fonts[static_cast<int>(Zep::ZepTextType::Heading1)])
            m_fonts[static_cast<int>(Zep::ZepTextType::Heading1)] = make_font(kHeading1Scale);
        if (!m_fonts[static_cast<int>(Zep::ZepTextType::Heading2)])
            m_fonts[static_cast<int>(Zep::ZepTextType::Heading2)] = make_font(kHeading2Scale);
        if (!m_fonts[static_cast<int>(Zep::ZepTextType::Heading3)])
            m_fonts[static_cast<int>(Zep::ZepTextType::Heading3)] = make_font(kHeading3Scale);
    }

    Zep::ZepFont& ZepDisplay_Rml::GetFont(const Zep::ZepTextType type) {
        ensureDefaultFonts();
        return *m_fonts[static_cast<int>(type)];
    }

    Rml::Vector2f ZepDisplay_Rml::offset() const {
        if (!element_)
            return {};
        return element_->GetAbsoluteOffset(Rml::BoxArea::Content);
    }

    bool ZepDisplay_Rml::hasClip() const {
        return clip_rect_.Width() > 0.0f && clip_rect_.Height() > 0.0f;
    }

    Rml::Rectanglei ZepDisplay_Rml::absoluteClip() const {
        const auto off = offset();
        const int x0 = static_cast<int>(std::floor(off.x + clip_rect_.topLeftPx.x));
        const int y0 = static_cast<int>(std::floor(off.y + clip_rect_.topLeftPx.y));
        const int x1 = static_cast<int>(std::ceil(off.x + clip_rect_.bottomRightPx.x));
        const int y1 = static_cast<int>(std::ceil(off.y + clip_rect_.bottomRightPx.y));
        return Rml::Rectanglei::FromCorners({x0, y0}, {x1, y1});
    }

    void ZepDisplay_Rml::DrawRectFilled(const Zep::NRectf& rc,
                                        const Zep::NVec4f& color,
                                        float /*rounding*/) const {
        if (!element_)
            return;

        auto* render_manager = element_->GetRenderManager();
        if (!render_manager)
            return;

        Rml::Mesh mesh;
        addQuad(mesh,
                rc.topLeftPx.x,
                rc.topLeftPx.y,
                rc.bottomRightPx.x,
                rc.bottomRightPx.y,
                toRmlColor(color));

        const auto state = render_manager->GetState();
        if (hasClip())
            render_manager->SetScissorRegion(absoluteClip());
        auto geometry = render_manager->MakeGeometry(std::move(mesh));
        geometry.Render(offset());
        render_manager->SetState(state);
    }

    void ZepDisplay_Rml::DrawLine(const Zep::NVec2f& start,
                                  const Zep::NVec2f& end,
                                  const Zep::NVec4f& color,
                                  const float width) const {
        if (!element_)
            return;

        const float dx = end.x - start.x;
        const float dy = end.y - start.y;
        const float len = std::sqrt(dx * dx + dy * dy);
        if (len <= 0.0f)
            return;

        const float half_width = std::max(width, 1.0f) * 0.5f;
        const float nx = -dy / len * half_width;
        const float ny = dx / len * half_width;

        auto* render_manager = element_->GetRenderManager();
        if (!render_manager)
            return;

        Rml::Mesh mesh;
        const auto c = toRmlColor(color);
        const int base = static_cast<int>(mesh.vertices.size());
        mesh.vertices.push_back({{start.x + nx, start.y + ny}, c, {0, 0}});
        mesh.vertices.push_back({{start.x - nx, start.y - ny}, c, {0, 0}});
        mesh.vertices.push_back({{end.x - nx, end.y - ny}, c, {0, 0}});
        mesh.vertices.push_back({{end.x + nx, end.y + ny}, c, {0, 0}});
        mesh.indices.insert(mesh.indices.end(), {base, base + 1, base + 2, base, base + 2, base + 3});

        const auto state = render_manager->GetState();
        if (hasClip())
            render_manager->SetScissorRegion(absoluteClip());
        auto geometry = render_manager->MakeGeometry(std::move(mesh));
        geometry.Render(offset());
        render_manager->SetState(state);
    }

    void ZepDisplay_Rml::DrawChars(Zep::ZepFont& font,
                                   const Zep::NVec2f& pos,
                                   const Zep::NVec4f& color,
                                   const uint8_t* text_begin,
                                   const uint8_t* text_end) const {
        if (!element_ || !text_begin)
            return;
        if (!text_end)
            text_end = text_begin + std::strlen(reinterpret_cast<const char*>(text_begin));
        if (text_end <= text_begin)
            return;

        auto* render_manager = element_->GetRenderManager();
        auto* font_engine = Rml::GetFontEngineInterface();
        if (!render_manager || !font_engine)
            return;

        auto* rml_font = dynamic_cast<Font*>(&font);
        Rml::FontFaceHandle face_handle = rml_font ? rml_font->faceHandle() : Rml::FontFaceHandle{};
        Rml::FontEffectsHandle effects_handle = rml_font ? rml_font->effectsHandle() : Rml::FontEffectsHandle{};
        if (!face_handle && element_) {
            face_handle = element_->GetFontFaceHandle();
            if (face_handle) {
                const Rml::FontEffectList effects;
                effects_handle = font_engine->PrepareFontEffects(face_handle, effects);
            }
        }
        if (!face_handle)
            return;

        Rml::TexturedMeshList meshes;
        Rml::TextShapingContext shaping{emptyLanguage()};
        font_engine->GenerateString(
            *render_manager,
            face_handle,
            effects_handle,
            Rml::StringView(reinterpret_cast<const char*>(text_begin), reinterpret_cast<const char*>(text_end)),
            {pos.x, pos.y + static_cast<float>(font.GetPixelHeight())},
            toRmlColor(color),
            1.0f,
            shaping,
            meshes);

        const auto state = render_manager->GetState();
        if (hasClip())
            render_manager->SetScissorRegion(absoluteClip());
        const auto off = offset();
        for (auto& mesh : meshes) {
            auto geometry = render_manager->MakeGeometry(std::move(mesh.mesh));
            geometry.Render(off, mesh.texture);
        }
        render_manager->SetState(state);
    }

    void ZepDisplay_Rml::SetClipRect(const Zep::NRectf& rc) {
        clip_rect_ = rc;
    }

} // namespace lfs::vis::editor
