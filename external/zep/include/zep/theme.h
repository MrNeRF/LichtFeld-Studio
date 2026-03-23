#pragma once

#include <map>
#include <vector>

#include "zep/mcommon/math/math.h"

namespace Zep {

    enum class ThemeColor {
        None,
        TabBorder,
        HiddenText,
        Text,
        TextDim,
        Background,
        TabInactive,
        TabActive,
        LineNumberBackground,
        LineNumber,
        LineNumberActive,
        CursorNormal,
        CursorInsert,
        Light,
        Dark,
        VisualSelectBackground,
        CursorLineBackground,
        AirlineBackground,
        Mode,
        Normal,
        Keyword,
        Identifier,
        Number,
        String,
        Comment,
        Whitespace,
        HiddenChar,
        Parenthesis,
        Error,
        Warning,
        Info,
        WidgetBorder,
        WidgetBackground,
        WidgetActive,
        WidgetInactive,
        FlashColor,
        Custom,

        // Last in the list!
        // A set of pregenerated, easy to differentiate unique colors
        UniqueColor0,
        UniqueColor1,
        UniqueColor2,
        UniqueColor3,
        UniqueColor4,
        UniqueColor5,
        UniqueColor6,
        UniqueColor7,
        UniqueColor8,
        UniqueColor9,
        UniqueColor10,
        UniqueColor11,
        UniqueColor12,
        UniqueColor13,
        UniqueColor14,
        UniqueColor15,
        UniqueColorLast,
    };

    enum class ThemeType {
        Dark,
        Light
    };

    class ZepTheme {
    public:
        ZepTheme();
        virtual ~ZepTheme() {}

        virtual const NVec4f& GetColor(ThemeColor themeColor) const;
        virtual NVec4f GetComplement(const NVec4f& col, const NVec4f& adjust = NVec4f(0.0f)) const;
        virtual ThemeColor GetUniqueColor(uint32_t id) const;
        virtual void SetColor(ThemeColor themeColor, const NVec4f& col);

        void SetThemeType(ThemeType type);
        ThemeType GetThemeType() const;

    private:
        void SetDarkTheme();
        void SetLightTheme();

    private:
        std::vector<NVec4f> m_uniqueColors;
        std::map<ThemeColor, NVec4f> m_colors;
        ThemeType m_currentTheme = ThemeType::Dark;
    };

} // namespace Zep
