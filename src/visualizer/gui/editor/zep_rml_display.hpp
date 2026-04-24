/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <RmlUi/Core/Element.h>
#include <zep/display.h>

namespace lfs::vis::editor {

    class ZepDisplay_Rml final : public Zep::ZepDisplay {
    public:
        ZepDisplay_Rml();

        void beginFrame(Rml::Element& element);
        void endFrame();
        void setBaseFontSize(float pixel_height);

        void DrawLine(const Zep::NVec2f& start,
                      const Zep::NVec2f& end,
                      const Zep::NVec4f& color = Zep::NVec4f(1.0f),
                      float width = 1.0f) const override;
        void DrawChars(Zep::ZepFont& font,
                       const Zep::NVec2f& pos,
                       const Zep::NVec4f& color,
                       const uint8_t* text_begin,
                       const uint8_t* text_end = nullptr) const override;
        void DrawRectFilled(const Zep::NRectf& rc,
                            const Zep::NVec4f& color = Zep::NVec4f(1.0f),
                            float rounding = 0.0f) const override;
        void SetClipRect(const Zep::NRectf& rc) override;
        Zep::ZepFont& GetFont(Zep::ZepTextType type) override;

    private:
        class Font;

        void ensureDefaultFonts();
        Rml::Vector2f offset() const;
        bool hasClip() const;
        Rml::Rectanglei absoluteClip() const;

        Rml::Element* element_ = nullptr;
        Zep::NRectf clip_rect_;
        float base_font_size_ = Zep::DefaultTextSize;
    };

} // namespace lfs::vis::editor
