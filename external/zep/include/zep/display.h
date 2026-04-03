#pragma once

#include <array>

#include "buffer.h"

namespace Zep {

    class ZepTabWindow;

    // A region inside the text for selections
    struct SelectRegion {
        // For vertical select, we will have a list of spans...
        GlyphIterator start;
        GlyphIterator end;
        bool visible = true;
        bool vertical = false; // Not yet supported
    };

    using ZepFontHandle = uint32_t;

    enum class ZepTextType {
        UI,
        Text,
        Heading1,
        Heading2,
        Heading3,
        Count
    };

    static const inline float DefaultTextSize = 16.0;

    class ZepFont {
    public:
        ZepFont(ZepDisplay& display)
            : m_display(display) {
        }

        // Implemented in API specific ways
        virtual void SetPixelHeight(int height) = 0;
        virtual NVec2f GetTextSize(const uint8_t* pBegin, const uint8_t* pEnd = nullptr) const = 0;

        virtual int GetPixelHeight() const {
            return m_pixelHeight;
        }

        virtual const NVec2f& GetDefaultCharSize();
        virtual const NVec2f& GetDotSize();
        virtual void BuildCharCache();
        virtual void InvalidateCharCache();
        virtual NVec2f GetCharSize(const uint8_t* pChar);

    protected:
        int m_pixelHeight;
        std::string m_filePath;
        bool m_charCacheDirty = true;
        std::unordered_map<uint32_t, NVec2f> m_charCache;
        NVec2f m_charCacheASCII[256];
        NVec2f m_dotSize;
        NVec2f m_defaultCharSize;
        ZepDisplay& m_display;
    };

    // Display interface
    class ZepDisplay {
    public:
        virtual ~ZepDisplay(){};
        ZepDisplay();

        // Renderer specific overrides
        // Implement these to draw the buffer using whichever system you prefer
        virtual void DrawLine(const NVec2f& start, const NVec2f& end, const NVec4f& color = NVec4f(1.0f), float width = 1.0f) const = 0;
        virtual void DrawChars(ZepFont& font, const NVec2f& pos, const NVec4f& col, const uint8_t* text_begin, const uint8_t* text_end = nullptr) const = 0;
        virtual void DrawRectFilled(const NRectf& rc, const NVec4f& col = NVec4f(1.0f), float rounding = 0.0f) const = 0;
        virtual void SetClipRect(const NRectf& rc) = 0;

        virtual uint32_t GetCodePointCount(const uint8_t* pCh, const uint8_t* pEnd) const;
        virtual void DrawRect(const NRectf& rc, const NVec4f& col = NVec4f(1.0f)) const;
        virtual bool LayoutDirty() const;
        virtual void SetLayoutDirty(bool changed = true);

        virtual void SetFont(ZepTextType type, std::shared_ptr<ZepFont> spFont);
        virtual ZepFont& GetFont(ZepTextType type) = 0;
        const NVec2f& GetPixelScale() const;
        void SetPixelScale(const NVec2f& scale);

        void Bigger();
        void Smaller();

        float GetDefaultTextSize() const;

    protected:
        bool m_bRebuildLayout = false;
        std::array<std::shared_ptr<ZepFont>, (int)ZepTextType::Count> m_fonts;
        std::shared_ptr<ZepFont> m_spDefaultFont;
        NVec2f m_pixelScale;
        float m_defaultTextSize = DefaultTextSize;
        float m_fontScale = 1.0f;
    };

    class ZepFontNull : public ZepFont {
    public:
        ZepFontNull(ZepDisplay& display)
            : ZepFont(display) {
        }

        virtual void SetPixelHeight(int val) override {
            ZEP_UNUSED(val);
        }

        virtual NVec2f GetTextSize(const uint8_t* pBegin, const uint8_t* pEnd = nullptr) const override {
            return NVec2f(float(pEnd - pBegin), 10.0f);
        }
    };

    // A NULL renderer, used for testing
    // Discards all drawing, and returns text fixed_size of 1 pixel per char, 10 height!
    // This is the only work you need to do to make a new renderer type for the editor
    class ZepDisplayNull : public ZepDisplay {
    public:
        ZepDisplayNull()
            : ZepDisplay() {
        }

        virtual void DrawLine(const NVec2f& start, const NVec2f& end, const NVec4f& color = NVec4f(1.0f), float width = 1.0f) const override {
            (void)start;
            (void)end;
            (void)color;
            (void)width;
        };
        virtual void DrawChars(ZepFont&, const NVec2f& pos, const NVec4f& col, const uint8_t* text_begin, const uint8_t* text_end = nullptr) const override {
            (void)pos;
            (void)col;
            (void)text_begin;
            (void)text_end;
        }
        virtual void DrawRectFilled(const NRectf& a, const NVec4f& col = NVec4f(1.0f), float rounding = 0.0f) const override {
            (void)a;
            (void)col;
            (void)rounding;
        };
        virtual void SetClipRect(const NRectf& rc) override {
            (void)rc;
        }

        virtual ZepFont& GetFont(ZepTextType type) override {
            if (m_fonts[(int)type] == nullptr) {
                if (m_spDefaultFont == nullptr) {
                    m_spDefaultFont = std::make_shared<ZepFontNull>(*this);
                }
                return *m_spDefaultFont;
            }
            return *m_fonts[(int)type];
        }
    };

} // namespace Zep
