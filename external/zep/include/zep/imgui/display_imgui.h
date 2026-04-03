#pragma once
#include "../display.h"
#include "../syntax.h"
#include <string>
#include <imgui.h>

// Can't include this publicly
// #include "zep/mcommon/logger.h"

namespace Zep {

    inline NVec2f toNVec2f(const ImVec2& im) {
        return NVec2f(im.x, im.y);
    }

    inline ImVec2 toImVec2(const NVec2f& im) {
        return ImVec2(im.x, im.y);
    }

    inline NVec4f toNVec4f(const ImVec4& im) {
        return NVec4f(im.x, im.y, im.z, im.w);
    }

    inline ImVec4 toImVec4(const NVec4f& im) {
        return ImVec4(im.x, im.y, im.z, im.w);
    }

    static ImWchar greek_range[] = {0x300, 0x52F, 0x1f00, 0x1fff, 0, 0};

    class ZepFont_ImGui : public ZepFont {
    public:
        ZepFont_ImGui(ZepDisplay& display, ImFont* pFont, int pixelHeight)
            : ZepFont(display), m_pFont(pFont) {
            SetPixelHeight(pixelHeight);
        }

        virtual void SetPixelHeight(int pixelHeight) override {
            InvalidateCharCache();
            m_pixelHeight = pixelHeight;
        }

        virtual NVec2f GetTextSize(const uint8_t* pBegin, const uint8_t* pEnd = nullptr) const override {
            // This is the code from ImGui internals; we can't call GetTextSize, because it doesn't return the correct 'advance' formula, which we
            // need as we draw one character at a time...
            ImVec2 text_size = m_pFont->CalcTextSizeA(float(GetPixelHeight()), FLT_MAX, FLT_MAX, (const char*)pBegin, (const char*)pEnd, NULL);
            if (text_size.x == 0.0) {
                // Make invalid characters a default fixed_size
                const char chDefault = 'A';
                text_size = m_pFont->CalcTextSizeA(float(GetPixelHeight()), FLT_MAX, FLT_MAX, &chDefault, (&chDefault + 1), NULL);
            }

            return toNVec2f(text_size);
        }

        ImFont* GetImFont() {
            return m_pFont;
        }

    private:
        ImFont* m_pFont;
        float m_fontScale = 1.0f;
    };

    class ZepDisplay_ImGui : public ZepDisplay {
    public:
        ZepDisplay_ImGui()
            : ZepDisplay() {
        }

        void DrawChars(ZepFont& font, const NVec2f& pos, const NVec4f& col, const uint8_t* text_begin, const uint8_t* text_end) const override {
            auto imFont = static_cast<ZepFont_ImGui&>(font).GetImFont();
            ImDrawList* drawList = ImGui::GetWindowDrawList();
            if (text_end == nullptr) {
                text_end = text_begin + strlen((const char*)text_begin);
            }
            const auto modulatedColor = GetStyleModulatedColor(col);
            if (m_clipRect.Width() == 0) {
                drawList->AddText(imFont, float(font.GetPixelHeight()), toImVec2(pos), modulatedColor, (const char*)text_begin, (const char*)text_end);
            } else {
                drawList->PushClipRect(toImVec2(m_clipRect.topLeftPx), toImVec2(m_clipRect.bottomRightPx));
                drawList->AddText(imFont, float(font.GetPixelHeight()), toImVec2(pos), modulatedColor, (const char*)text_begin, (const char*)text_end);
                drawList->PopClipRect();
            }
        }

        void DrawLine(const NVec2f& start, const NVec2f& end, const NVec4f& color, float width) const override {
            ImDrawList* drawList = ImGui::GetWindowDrawList();
            const auto modulatedColor = GetStyleModulatedColor(color);
            // Background rect for numbers
            if (m_clipRect.Width() == 0) {
                drawList->AddLine(toImVec2(start), toImVec2(end), modulatedColor, width);
            } else {
                drawList->PushClipRect(toImVec2(m_clipRect.topLeftPx), toImVec2(m_clipRect.bottomRightPx));
                drawList->AddLine(toImVec2(start), toImVec2(end), modulatedColor, width);
                drawList->PopClipRect();
            }
        }

        void DrawRectFilled(const NRectf& rc, const NVec4f& color, float rounding = 0.0f) const override {
            ImDrawList* drawList = ImGui::GetWindowDrawList();
            const auto modulatedColor = GetStyleModulatedColor(color);
            // Background rect for numbers
            if (m_clipRect.Width() == 0) {
                drawList->AddRectFilled(toImVec2(rc.topLeftPx), toImVec2(rc.bottomRightPx), modulatedColor, rounding);
            } else {
                drawList->PushClipRect(toImVec2(m_clipRect.topLeftPx), toImVec2(m_clipRect.bottomRightPx));
                drawList->AddRectFilled(toImVec2(rc.topLeftPx), toImVec2(rc.bottomRightPx), modulatedColor, rounding);
                drawList->PopClipRect();
            }
        }

        virtual void SetClipRect(const NRectf& rc) override {
            m_clipRect = rc;
        }

        virtual ZepFont& GetFont(ZepTextType type) override {
            if (m_fonts[(int)type] == nullptr) {
                m_fonts[(int)type] = std::make_shared<ZepFont_ImGui>(*this, ImGui::GetIO().Fonts[0].Fonts[0], int(DefaultTextSize * GetPixelScale().y));
            }
            return *m_fonts[(int)type];
        }

    private:
        static ImU32 GetStyleModulatedColor(const NVec4f& color) {
            return ToPackedABGR(NVec4f(color.x, color.y, color.z, color.w * ImGui::GetStyle().Alpha));
        }

        NRectf m_clipRect;
    }; // namespace Zep

} // namespace Zep
