#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "buffer.h"

namespace Zep {

    class ZepTabWindow;
    class ZepDisplay;
    class Scroller;

    struct Region;

    struct LineCharInfo {
        NVec2f pos;
        NVec2f size;
        GlyphIterator iterator;
    };

    class ZepFont;
    // Line information, calculated during display update.
    // A collection of spans that show split lines on the display
    struct SpanInfo {
        ByteRange lineByteRange;                  // Begin/end range of the text buffer for this line, as always end is one beyond the end.
        std::vector<LineCharInfo> lineCodePoints; // Codepoints
        long bufferLineNumber = 0;                // Line in the original buffer, not the screen line
        float yOffsetPx = 0.0f;                   // Position in the buffer in pixels, if the screen was as big as the buffer.
        NVec2f lineTextSizePx = NVec2f(0.0f);     // Pixel size of the text
        int spanLineIndex = 0;                    // The index of this line in spans; might be more than buffer index
        NVec2f padding = NVec2f(1.0f, 1.0f);      // Padding above and below the line
        bool isSplitContinuation = false;
        NVec2f lineWidgetHeights;
        ZepFont* pFont = nullptr;

        float FullLineHeightPx() const {
            return padding.x + padding.y + lineTextSizePx.y;
        }

        // The byte length, not code point length
        // TODO: This is not the way to measure text lengths
        long ByteLength() const {
            return lineByteRange.second - lineByteRange.first;
        }

        bool BufferCursorInside(GlyphIterator offset) const {
            return offset.Index() >= lineByteRange.first && offset.Index() < lineByteRange.second;
        }
    };

    inline bool operator<(const SpanInfo& lhs, const SpanInfo& rhs) {
        if (lhs.lineByteRange.first != rhs.lineByteRange.first) {
            return lhs.lineByteRange.first < rhs.lineByteRange.first;
        }
        return lhs.lineByteRange.second < rhs.lineByteRange.second;
    }

    enum class CursorType {
        None,
        Normal,
        Insert,
        Visual,
        LineMarker
    };

    namespace WindowFlags {
        enum {
            None = (0),
            ShowWhiteSpace = (1 << 0),
            ShowCR = (1 << 1),
            ShowLineNumbers = (1 << 2),
            ShowIndicators = (1 << 3),
            HideScrollBar = (1 << 4),
            Modal = (1 << 5),
            WrapText = (1 << 6), // Warning: this is not for general use yet. Has issues
            HideSplitMark = (1 << 7),
            GridStyle = (1 << 8)
        };
    }

    struct AirBox {
        std::string text;
        NVec4f background;
    };

    struct Airline {
        std::vector<AirBox> leftBoxes;
        std::vector<AirBox> rightBoxes;
    };

    // Message to request a tooltip.
    // Clients should return the marker information if appropriate
    struct ToolTipMessage : public ZepMessage {
        ToolTipMessage(ZepBuffer* pBuff, const NVec2f& pos, const GlyphIterator& loc = GlyphIterator())
            : ZepMessage(Msg::ToolTip, pos), pBuffer(pBuff), location(loc) {
        }

        ZepBuffer* pBuffer;
        GlyphIterator location;
        std::shared_ptr<RangeMarker> spMarker;
    };

    // Display state for a single pane of text.
    // Window shows a buffer, and is parented by a TabWindow
    // The buffer can change, but the window must always have an active buffer
    // Editor operations such as select and change are local to a displayed pane
    class ZepWindow : public ZepComponent {
    public:
        ZepWindow(ZepTabWindow& window, ZepBuffer* buffer);
        virtual ~ZepWindow();

        virtual void Notify(std::shared_ptr<ZepMessage> message) override;

        // Display
        virtual void SetDisplayRegion(const NRectf& region);
        virtual void Display();

        // Cursor
        virtual GlyphIterator GetBufferCursor();
        virtual void SetBufferCursor(GlyphIterator location);
        virtual void MoveCursorY(int yDistance, LineLocation clampLocation = LineLocation::LineLastNonCR);
        virtual NVec2i BufferToDisplay();
        virtual NRectf GetCursorRect();

        // Flags
        virtual void SetWindowFlags(uint32_t windowFlags);
        virtual uint32_t GetWindowFlags() const;
        virtual void ToggleFlag(uint32_t flag);

        virtual long GetMaxDisplayLines();
        virtual long GetNumDisplayedLines();

        virtual ZepBuffer& GetBuffer() const;
        virtual void SetBuffer(ZepBuffer* pBuffer);

        ZepTabWindow& GetTabWindow() const;
        NVec4f FilterActiveColor(const NVec4f& col, float atten = 1.0f);

        void DirtyLayout();

    private:
        void UpdateLayout(bool force = false);
        void UpdateMarkers();
        void UpdateAirline();
        void UpdateScrollers();
        void UpdateLineSpans();
        void EnsureCursorVisible();
        void UpdateVisibleLineRange();

        NVec2i BufferToDisplay(const GlyphIterator& location);

        void ScrollToCursor();
        bool IsInsideVisibleText(NVec2i pos) const;

        enum class SpecialChar {
            None,
            Hidden,
            Tab,
            Space
        };
        void GetCharPointer(GlyphIterator loc, const uint8_t*& pBegin, const uint8_t*& pEnd, SpecialChar& specialChar);
        const SpanInfo& GetCursorLineInfo(long y);

        float ToWindowY(float pos) const;
        float ToWindowX(float pos) const;
        float TipBoxShadowWidth() const;

        // Display
        void DisplayToolTip(const NVec2f& pos, const RangeMarker& marker) const;
        bool DisplayLine(SpanInfo& lineInfo, int displayPass);
        void DisplayLineBackground(SpanInfo& lineInfo, ZepSyntax* pSyntax);
        void DisplayScrollers();
        void DisplayGridMarkers();
        void DisplayMarkerHints();
        void DisplayLineNumbers();

        void DisableToolTipTillMove();

        NVec4f ModifyBackgroundColor(ThemeColor color) const;
        void GetCursorInfo(NVec2f& pos, NVec2f& size);
        float GetCursorTextX(const SpanInfo& cursorBufferLine, long cursorColumn, NVec2f* size = nullptr) const;

        void PlaceToolTip(const NVec2f& pos, ToolTipPos location, uint32_t lineGap, const std::shared_ptr<RangeMarker> spMarker);

        void DrawAboveLineWidgets(SpanInfo& lineInfo);

        NVec2f ArrangeLineMarkers(tRangeMarkers& markers);

        bool IsActiveWindow() const;

        NVec2f GetSpanPixelRange(SpanInfo& span) const;

    private:
        NRectf m_displayRect;
        std::shared_ptr<Region> m_bufferRegion;        // region of the display we are showing on.
        std::shared_ptr<Region> m_editRegion;          // region of the window buffer editing
        std::shared_ptr<Region> m_textRegion;          // region of the display for text.
        std::shared_ptr<Region> m_airlineRegion;       // Airline
        std::shared_ptr<Region> m_numberRegion;        // Numbers
        std::shared_ptr<Region> m_indicatorRegion;     // Indicators (between numbers and text)
        std::shared_ptr<Region> m_vScrollRegion;       // Vertical scroller
        std::shared_ptr<Region> m_scrollFooterRegion;  // Row containing the horizontal scroller
        std::shared_ptr<Region> m_hScrollOffsetRegion; // Spacer so the bottom scroller lines up with text
        std::shared_ptr<Region> m_hScrollRegion;       // Horizontal scroller
        std::shared_ptr<Region> m_scrollCornerRegion;  // Fill under the vertical scroller
        std::shared_ptr<Region> m_expandingEditRegion; // Region containing the text sub-box
        Airline m_airline;

        // Owner tab
        ZepTabWindow& m_tabWindow;

        // Buffer
        ZepBuffer* m_pBuffer = nullptr;

        // Scroll bar, if visible
        std::shared_ptr<Scroller> m_vScroller;
        std::shared_ptr<Scroller> m_hScroller;

        // Wrap ; need horizontal offset for this to be turned on.
        // This will indeed stop lines wrapping though!  You just can't move to the far right and have
        // the text scroll; which isn't a big deal, but a work item.
        // TODO Use flags instead
        bool m_layoutDirty = true;
        bool m_scrollVisibilityChanged = true;
        bool m_cursorMoved = true;
        uint32_t m_windowFlags = WindowFlags::ShowWhiteSpace | WindowFlags::ShowIndicators | WindowFlags::ShowLineNumbers | WindowFlags::WrapText;

        // Cursor
        GlyphIterator m_bufferCursor;  // Location in buffer coordinates.  Each window has a different buffer cursor
        long m_lastCursorColumn = 0;   // The last cursor column (could be removed and recalculated)
        NVec2f m_mousePos;             // Current mouse location
        GlyphIterator m_mouseIterator; // Current iterator for the mouse cursor

        // Visual stuff
        std::vector<std::string> m_statusLines; // Status information, shown under the buffer

        // Setup of displayed lines
        std::vector<SpanInfo*> m_windowLines; // Information about the currently displayed lines
        float m_textOffsetPx = 0.0f;          // The Scroll position within the text
        float m_textOffsetXPx = 0.0f;         // The horizontal scroll position within the text
        NVec2f m_textSizePx;                  // The calculated size of the buffer text, containing just the text
        NVec2i m_visibleLineIndices = {0, 0}; // Index of the line spans that are visible
        long m_maxDisplayLines = 0;
        int m_defaultLineSize = 0;
        float m_xPad = 0.0f;

        // Tooltips
        timer m_toolTipTimer;                                      // Timer for when the tip is shown
        NVec2f m_mouseHoverPos;                                    // Current location for the tip
        GlyphIterator m_mouseBufferLocation;                       // The character in the buffer the tip pos is over, or -1
        NVec2f m_lastTipQueryPos;                                  // last query location for the tip
        bool m_tipDisabledTillMove = false;                        // Certain operations will stop the tip until the mouse is moved
        std::map<NVec2f, std::shared_ptr<RangeMarker>> m_toolTips; // All tooltips for a given position, currently only 1 at a time

        NVec2f lastLinePx;
    };

} // namespace Zep
