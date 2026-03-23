#pragma once

#include <map>
#include <set>

#include "zep/mcommon/animation/timer.h"
#include "zep/mcommon/signals.h"

#include "zep/glyph_iterator.h"
#include "zep/theme.h"

// Range Markers are adornments over the text; they represent any additional marks over the existing text buffer.
// For example, tooltips, underlines, inline widgets, etc.
// Try :ZTestMarkers 5 or :ZTestMarkers 3 after selecting a region of text
namespace Zep {
    struct IWidget;
    class ZepBuffer;

    namespace RangeMarkerType {
        enum {
            Mark = (1 << 0),
            Search = (1 << 1),
            Widget = (1 << 2),
            LineWidget = (1 << 3),
            All = (Mark | Search | LineWidget | Widget)
        };
    };

    enum class FlashType {
        Flash
    };

    namespace RangeMarkerDisplayType {
        enum {
            Hidden = 0,
            Underline = (1 << 0),       // Underline the range
            Background = (1 << 1),      // Add a background to the range
            Tooltip = (1 << 2),         // Show a tooltip using the name/description
            TooltipAtLine = (1 << 3),   // Tooltip shown if the user hovers the line
            CursorTip = (1 << 4),       // Tooltip shown if the user cursor is on the Mark
            CursorTipAtLine = (1 << 5), // Tooltip shown if the user cursor is on the Mark line
            Indicator = (1 << 6),       // Show an indicator on the left side
            Timed = (1 << 7),
            All = Underline | Tooltip | TooltipAtLine | CursorTip | CursorTipAtLine | Indicator | Background,
            CompileError = Tooltip | CursorTip | Indicator | Background,
            BackgroundMark = Background
        };
    };

    enum class ToolTipPos {
        AboveLine = 0,
        BelowLine = 1,
        RightLine = 2,
        Count = 3
    };

    struct RangeMarker : std::enable_shared_from_this<RangeMarker> {
        RangeMarker(ZepBuffer& buffer);

        bool ContainsLocation(GlyphIterator loc) const;
        bool IntersectsRange(const ByteRange& i) const;
        virtual ThemeColor GetBackgroundColor(const GlyphIterator& itr = GlyphIterator()) const;
        virtual ThemeColor GetTextColor(const GlyphIterator& itr = GlyphIterator()) const;
        virtual ThemeColor GetHighlightColor(const GlyphIterator& itr = GlyphIterator()) const;
        virtual float GetAlpha(const GlyphIterator& itr = GlyphIterator()) const;
        virtual const std::string& GetName() const;
        virtual const std::string& GetDescription() const;
        virtual const NVec2f& GetInlineSize() const;

        virtual void SetRange(ByteRange range);
        virtual const ByteRange& GetRange() const;
        virtual void SetBackgroundColor(ThemeColor color);
        virtual void SetTextColor(ThemeColor color);
        virtual void SetHighlightColor(ThemeColor color);
        virtual void SetColors(ThemeColor back = ThemeColor::None, ThemeColor text = ThemeColor::Text, ThemeColor highlight = ThemeColor::Text);
        virtual void SetAlpha(float a);
        virtual void SetName(const std::string& name);
        virtual void SetDescription(const std::string& desc);
        virtual void SetEnabled(bool enabled);
        virtual void SetInlineSize(const NVec2f& size);

        void HandleBufferInsert(ZepBuffer& buffer, const GlyphIterator& itrStart, const std::string& str);
        void HandleBufferDelete(ZepBuffer& buffer, const GlyphIterator& itr, const GlyphIterator& itrEnd);

        ZepBuffer& GetBuffer();

    public:
        // TODO: Move to accessors.
        uint32_t displayType = RangeMarkerDisplayType::All;
        uint32_t markerType = RangeMarkerType::Mark;
        uint32_t displayRow = 0;
        ToolTipPos tipPos = ToolTipPos::AboveLine;
        std::shared_ptr<IWidget> spWidget;
        float duration = 1.0f;
        Zep::timer timer;
        FlashType flashType = FlashType::Flash;

    protected:
        ZepBuffer& m_buffer;
        ByteRange m_range;
        mutable float alpha = 1.0f;
        std::string m_name;
        std::string m_description;
        bool m_enabled = true;
        NVec2f m_inlineSize;

        Zep::scoped_connection onPreBufferInsert;
        Zep::scoped_connection onPreBufferDelete;

        mutable ThemeColor m_textColor = ThemeColor::Text;
        mutable ThemeColor m_backgroundColor = ThemeColor::Background;
        mutable ThemeColor m_highlightColor = ThemeColor::Background; // Used for lines around tip box, underline, etc.
    };

    using tRangeMarkers = std::map<ByteIndex, std::set<std::shared_ptr<RangeMarker>>>;

}; // namespace Zep
