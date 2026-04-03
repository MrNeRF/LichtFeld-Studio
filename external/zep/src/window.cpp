#include <algorithm>
#include <cctype>
#include <cmath>
#include <sstream>

#include "zep/buffer.h"
#include "zep/display.h"
#include "zep/mode.h"
#include "zep/scroller.h"
#include "zep/syntax.h"
#include "zep/tab_window.h"
#include "zep/theme.h"
#include "zep/window.h"

#include "zep/mcommon/logger.h"
#include "zep/mcommon/string/stringutils.h"
#include "zep/mcommon/utf8.h"

namespace {
    struct WindowPass {
        enum Pass {
            Background = 0,
            Text,
            Max
        };
    };
} // namespace

// A 'window' is like a vim window; i.e. a region inside a tab
namespace Zep {

    const float UnderlineMargin = 1.0f;

    ZepWindow::ZepWindow(ZepTabWindow& window, ZepBuffer* buffer)
        : ZepComponent(window.GetEditor()), m_tabWindow(window), m_pBuffer(buffer) {
        m_bufferRegion = std::make_shared<Region>();
        m_numberRegion = std::make_shared<Region>();
        m_indicatorRegion = std::make_shared<Region>();
        m_textRegion = std::make_shared<Region>();
        m_airlineRegion = std::make_shared<Region>();
        m_vScrollRegion = std::make_shared<Region>();
        m_scrollFooterRegion = std::make_shared<Region>();
        m_hScrollOffsetRegion = std::make_shared<Region>();
        m_hScrollRegion = std::make_shared<Region>();
        m_scrollCornerRegion = std::make_shared<Region>();

        m_bufferRegion->flags = RegionFlags::Expanding;
        m_bufferRegion->layoutType = RegionLayoutType::VBox;

        m_numberRegion->flags = RegionFlags::Fixed;
        m_indicatorRegion->flags = RegionFlags::Fixed;
        m_vScrollRegion->flags = RegionFlags::Fixed;
        m_textRegion->flags = RegionFlags::Expanding;
        m_scrollFooterRegion->flags = RegionFlags::Fixed;
        m_hScrollOffsetRegion->flags = RegionFlags::Fixed;
        m_hScrollRegion->flags = RegionFlags::Expanding;
        m_scrollCornerRegion->flags = RegionFlags::Fixed;
        m_airlineRegion->flags = RegionFlags::Fixed;
        m_scrollFooterRegion->layoutType = RegionLayoutType::HBox;

        m_editRegion = std::make_shared<Region>();
        m_editRegion->flags = RegionFlags::Expanding;
        m_editRegion->layoutType = RegionLayoutType::HBox;

        // A little daylight between the indicators
        m_textRegion->padding = NVec2f(DPI_X(8), 0);

        // Ensure that the main area with text, numbers, indicators fills the remaining space
        m_expandingEditRegion = std::make_shared<Region>();
        m_expandingEditRegion->flags = RegionFlags::Expanding;
        m_expandingEditRegion->layoutType = RegionLayoutType::HBox;
        m_expandingEditRegion->children.push_back(m_editRegion);

        m_bufferRegion->children.push_back(m_expandingEditRegion);
        m_bufferRegion->children.push_back(m_scrollFooterRegion);
        m_editRegion->children.push_back(m_numberRegion);
        m_editRegion->children.push_back(m_indicatorRegion);
        m_editRegion->children.push_back(m_textRegion);
        m_editRegion->children.push_back(m_vScrollRegion);
        m_scrollFooterRegion->children.push_back(m_hScrollOffsetRegion);
        m_scrollFooterRegion->children.push_back(m_hScrollRegion);
        m_scrollFooterRegion->children.push_back(m_scrollCornerRegion);

        m_bufferRegion->children.push_back(m_airlineRegion);

        m_vScroller = std::make_shared<Scroller>(GetEditor(), *m_vScrollRegion, true);
        m_hScroller = std::make_shared<Scroller>(GetEditor(), *m_hScrollRegion, false);

        SetBuffer(m_pBuffer);

        timer_start(m_toolTipTimer);
    }

    ZepWindow::~ZepWindow() {
        std::for_each(m_windowLines.begin(), m_windowLines.end(), [](SpanInfo* pInfo) { delete pInfo; });
    }

    void ZepWindow::UpdateScrollers() {
        m_scrollVisibilityChanged = false;
        const auto pixelScale = GetEditor().GetDisplay().GetPixelScale();
        const float scrollBarWidthPx = GetEditor().GetConfig().scrollBarSize * pixelScale.x;
        const float scrollBarHeightPx = GetEditor().GetConfig().scrollBarSize * pixelScale.y;
        const float oldVScrollWidth = m_vScrollRegion->fixed_size.x;
        const float oldFooterHeight = m_scrollFooterRegion->fixed_size.y;
        const float oldHScrollOffset = m_hScrollOffsetRegion->fixed_size.x;
        const float oldCornerWidth = m_scrollCornerRegion->fixed_size.x;

        if (m_maxDisplayLines == 0) {
            m_vScroller->scrollVisiblePercent = 1.0f;
            m_vScroller->scrollPosition = 0.0f;
            m_hScroller->scrollVisiblePercent = 1.0f;
            m_hScroller->scrollPosition = 0.0f;
            m_vScrollRegion->fixed_size = NVec2f(0.0f);
            m_scrollFooterRegion->fixed_size = NVec2f(0.0f);
            m_hScrollOffsetRegion->fixed_size = NVec2f(0.0f);
            m_scrollCornerRegion->fixed_size = NVec2f(0.0f);
            if (oldVScrollWidth != 0.0f || oldFooterHeight != 0.0f ||
                oldHScrollOffset != 0.0f || oldCornerWidth != 0.0f) {
                m_scrollVisibilityChanged = true;
            }
            return;
        }

        const bool scrollBarsAllowed =
            GetEditor().GetConfig().showScrollBar != 0 &&
            !ZTestFlags(GetWindowFlags(), WindowFlags::HideScrollBar);
        const float visibleHeight = std::max(m_textRegion->rect.Height(), 0.0f);
        const float visibleWidth = std::max(m_textRegion->rect.Width(), 0.0f);
        const float contentHeight = std::max(m_textSizePx.y, 0.0f);
        const float contentWidth = std::max(m_textSizePx.x, 0.0f);

        m_textOffsetPx = std::clamp(m_textOffsetPx, 0.0f, std::max(0.0f, contentHeight - visibleHeight));
        m_textOffsetXPx = std::clamp(m_textOffsetXPx, 0.0f, std::max(0.0f, contentWidth - visibleWidth));

        if (contentHeight > 0.0f) {
            m_vScroller->scrollVisiblePercent = std::min(visibleHeight / contentHeight, 1.0f);
            m_vScroller->scrollPosition = m_textOffsetPx / contentHeight;
        } else {
            m_vScroller->scrollVisiblePercent = 1.0f;
            m_vScroller->scrollPosition = 0.0f;
        }
        m_vScroller->scrollLinePercent = m_windowLines.empty() ? 0.0f : (1.0f / m_windowLines.size());
        m_vScroller->scrollPagePercent = m_vScroller->scrollVisiblePercent;

        if (contentWidth > 0.0f) {
            m_hScroller->scrollVisiblePercent = std::min(visibleWidth / contentWidth, 1.0f);
            m_hScroller->scrollPosition = m_textOffsetXPx / contentWidth;
        } else {
            m_hScroller->scrollVisiblePercent = 1.0f;
            m_hScroller->scrollPosition = 0.0f;
        }
        const float charWidth = GetEditor().GetDisplay().GetFont(ZepTextType::Text).GetDefaultCharSize().x;
        m_hScroller->scrollLinePercent = std::min(charWidth / std::max(contentWidth, 1.0f), 1.0f);
        m_hScroller->scrollPagePercent = m_hScroller->scrollVisiblePercent;

        const bool showVertical =
            scrollBarsAllowed &&
            (GetEditor().GetConfig().showScrollBar == 2 || m_vScroller->scrollVisiblePercent < 1.0f);
        if (!showVertical) {
            m_vScrollRegion->fixed_size = NVec2f(0.0f);
        } else {
            m_vScrollRegion->fixed_size = NVec2f(scrollBarWidthPx, 0.0f);
        }

        const bool showHorizontal =
            scrollBarsAllowed &&
            !ZTestFlags(GetWindowFlags(), WindowFlags::WrapText) &&
            (GetEditor().GetConfig().showScrollBar == 2 || m_hScroller->scrollVisiblePercent < 1.0f);
        if (!showHorizontal) {
            m_scrollFooterRegion->fixed_size = NVec2f(0.0f);
            m_hScrollOffsetRegion->fixed_size = NVec2f(0.0f);
            m_scrollCornerRegion->fixed_size = NVec2f(0.0f);
        } else {
            m_scrollFooterRegion->fixed_size = NVec2f(0.0f, scrollBarHeightPx);
            m_hScrollOffsetRegion->fixed_size =
                NVec2f(std::max(0.0f, m_textRegion->rect.Left() - m_scrollFooterRegion->rect.Left()), 0.0f);
            m_scrollCornerRegion->fixed_size = NVec2f(m_vScrollRegion->fixed_size.x, 0.0f);
        }

        if (oldVScrollWidth != m_vScrollRegion->fixed_size.x ||
            oldFooterHeight != m_scrollFooterRegion->fixed_size.y ||
            oldHScrollOffset != m_hScrollOffsetRegion->fixed_size.x ||
            oldCornerWidth != m_scrollCornerRegion->fixed_size.x) {
            m_scrollVisibilityChanged = true;
        }
    }

    void ZepWindow::UpdateAirline() {
        m_airline.leftBoxes.clear();
        m_airline.rightBoxes.clear();

        if (IsActiveWindow()) {
            m_airline.leftBoxes.push_back(AirBox{GetBuffer().GetMode()->Name(), FilterActiveColor(m_pBuffer->GetTheme().GetColor(ThemeColor::Mode))});
            switch (GetBuffer().GetMode()->GetEditorMode()) {
                /*case EditorMode::Hidden:
                m_airline.leftBoxes.push_back(AirBox{ "HIDDEN", m_pBuffer->GetTheme().GetColor(ThemeColor::HiddenText) });
                break;
                */
            default:
                break;
            case EditorMode::Insert:
                m_airline.leftBoxes.push_back(AirBox{"INSERT", FilterActiveColor(m_pBuffer->GetTheme().GetColor(ThemeColor::CursorInsert))});
                break;
            case EditorMode::None:
            case EditorMode::Normal:
                m_airline.leftBoxes.push_back(AirBox{"NORMAL", FilterActiveColor(m_pBuffer->GetTheme().GetColor(ThemeColor::CursorNormal))});
                break;
            case EditorMode::Visual:
                m_airline.leftBoxes.push_back(AirBox{"VISUAL", FilterActiveColor(m_pBuffer->GetTheme().GetColor(ThemeColor::VisualSelectBackground))});
                break;
            };
        }

        auto& font = GetEditor().GetDisplay().GetFont(ZepTextType::Text);
        m_airline.leftBoxes.push_back(AirBox{std::to_string(int(((float)font.GetPixelHeight() / GetEditor().GetDisplay().GetDefaultTextSize()) * 100.0f)) + "%", m_pBuffer->GetTheme().GetColor(ThemeColor::TabActive)});

        auto cursor = BufferToDisplay();
        m_airline.leftBoxes.push_back(AirBox{m_pBuffer->GetDisplayName(), FilterActiveColor(m_pBuffer->GetTheme().GetColor(ThemeColor::AirlineBackground))});
        m_airline.leftBoxes.push_back(AirBox{std::to_string(cursor.x) + ":" + std::to_string(cursor.y), m_pBuffer->GetTheme().GetColor(ThemeColor::TabActive)});

#ifdef _SHOW_SCALE
        m_airline.leftBoxes.push_back(AirBox{"(" + std::to_string(GetEditor().GetDisplay().GetPixelScale().x) + "," + std::to_string(GetEditor().GetDisplay().GetPixelScale().y) + ")", m_pBuffer->GetTheme().GetColor(ThemeColor::Error)});
#endif

        auto extra = GetBuffer().GetMode()->GetAirlines(*this);

        auto lastSize = m_airlineRegion->fixed_size;
        m_airlineRegion->fixed_size = NVec2f(0.0f, float(GetEditor().GetDisplay().GetFont(ZepTextType::UI).GetPixelHeight() * (1 + extra.size())));
        if (m_airlineRegion->fixed_size != lastSize) {
            m_layoutDirty = true;
        }
    }

    void ZepWindow::Notify(std::shared_ptr<ZepMessage> payload) {
        if (payload->messageId == Msg::Buffer) {
            auto pMsg = std::static_pointer_cast<BufferMessage>(payload);

            if (pMsg->pBuffer != m_pBuffer) {
                return;
            }

            m_layoutDirty = true;

            if (pMsg->type != BufferMessageType::PreBufferChange) {
                // Make sure the cursor is on its 'display' part of the flash cycle after an edit.
                GetEditor().ResetCursorTimer();
            }

            // Remove tooltips that might be present
            DisableToolTipTillMove();
        } else if (payload->messageId == Msg::ComponentChanged) {
            if (payload->pComponent == m_vScroller.get()) {
                auto pScroller = dynamic_cast<Scroller*>(payload->pComponent);
                m_textOffsetPx = pScroller->scrollPosition * m_textSizePx.y;
                UpdateVisibleLineRange();
                EnsureCursorVisible();
                DisableToolTipTillMove();
            } else if (payload->pComponent == m_hScroller.get()) {
                auto pScroller = dynamic_cast<Scroller*>(payload->pComponent);
                m_textOffsetXPx = std::clamp(
                    pScroller->scrollPosition * m_textSizePx.x,
                    0.0f,
                    std::max(0.0f, m_textSizePx.x - m_textRegion->rect.Width()));
                DisableToolTipTillMove();
            }
        } else if (payload->messageId == Msg::MouseMove) {
            m_mousePos = payload->pos;
            if (!m_toolTips.empty()) {
                if (ManhattanDistance(m_mouseHoverPos, payload->pos) > 4.0f) {
                    timer_restart(m_toolTipTimer);
                    m_toolTips.clear();
                }
            } else {
                timer_restart(m_toolTipTimer);
                m_mouseHoverPos = payload->pos;

                // Can now show tooltip again, due to mouse hover
                m_tipDisabledTillMove = false;
            }
        } else if (payload->messageId == Msg::ConfigChanged) {
            m_layoutDirty = true;
        } else if (payload->messageId == Msg::MouseDown) {
            if (payload->button == ZepMouseButton::Left && m_textRegion->rect.Contains(m_mousePos) && m_mouseIterator.Valid()) {
                SetBufferCursor(m_mouseIterator);
            }
        } else if (payload->messageId == Msg::MouseWheel) {
            if (!m_bufferRegion->rect.Contains(payload->pos)) {
                return;
            }

            payload->handled = true;

            const float maxScrollOffset =
                std::max(0.0f, m_textSizePx.y - m_textRegion->rect.Height());
            if (maxScrollOffset <= 0.0f) {
                return;
            }

            const float scrollDistance = std::stof(payload->str);
            if (scrollDistance == 0.0f) {
                return;
            }

            const float lineHeight =
                GetEditor().GetDisplay().GetFont(ZepTextType::Text).GetPixelHeight();
            constexpr float wheelLines = 4.0f;
            m_textOffsetPx = std::clamp(
                m_textOffsetPx - wheelLines * scrollDistance * lineHeight,
                0.0f,
                maxScrollOffset);
            UpdateVisibleLineRange();
            DisableToolTipTillMove();
        }
    }

    void ZepWindow::SetDisplayRegion(const NRectf& region) {
        if (m_bufferRegion->rect == region) {
            return;
        }

        m_displayRect = region;

        m_layoutDirty = true;
        m_bufferRegion->rect = region;

        m_airlineRegion->fixed_size = NVec2f(0.0f, float(GetEditor().GetDisplay().GetFont(ZepTextType::UI).GetPixelHeight()));

        m_defaultLineSize = GetEditor().GetDisplay().GetFont(ZepTextType::Text).GetPixelHeight();
    }

    void ZepWindow::EnsureCursorVisible() {
        UpdateLayout();
        GlyphIterator loc = m_bufferCursor;
        for (auto& line : m_windowLines) {
            if (line->lineByteRange.first <= loc.Index() && line->lineByteRange.second > loc.Index()) {
                auto cursorLine = line->spanLineIndex;
                if (cursorLine < m_visibleLineIndices.x) {
                    MoveCursorY(std::abs(m_visibleLineIndices.x - cursorLine));
                } else if (cursorLine >= m_visibleLineIndices.y) {
                    MoveCursorY((long(m_visibleLineIndices.y) - cursorLine) - 1);
                }
                m_cursorMoved = false;
                return;
            }
        }
    }

    void ZepWindow::ScrollToCursor() {
        if (!m_cursorMoved) {
            return;
        }

        auto old_offset = m_textOffsetPx;
        auto two_lines = (GetEditor().GetDisplay().GetFont(ZepTextType::Text).GetPixelHeight() * 2); // +(lineMargins.x + lineMargins.y) * 2;
        auto cursorPos = BufferToDisplay();
        auto& cursorLine = GetCursorLineInfo(cursorPos.y);

        // If the buffer is beyond two lines above the cursor position, move it back by the difference
        if (m_textOffsetPx > (cursorLine.yOffsetPx - two_lines)) {
            m_textOffsetPx -= (m_textOffsetPx - (cursorLine.yOffsetPx - two_lines));
        } else if ((m_textOffsetPx + m_textRegion->rect.Height() - two_lines) < cursorLine.yOffsetPx) {
            m_textOffsetPx += cursorLine.yOffsetPx - (m_textOffsetPx + m_textRegion->rect.Height() - two_lines);
        }

        m_textOffsetPx = std::min(m_textOffsetPx, m_textSizePx.y - m_textRegion->rect.Height());
        m_textOffsetPx = std::max(0.f, m_textOffsetPx);

        if (old_offset != m_textOffsetPx) {
            UpdateVisibleLineRange();
        }

        NVec2f cursorSize;
        const float cursorX = GetCursorTextX(cursorLine, cursorPos.x, &cursorSize);
        const float horizontalMargin = GetEditor().GetDisplay().GetFont(ZepTextType::Text).GetDefaultCharSize().x * 2.0f;
        if (m_textOffsetXPx > std::max(0.0f, cursorX - horizontalMargin)) {
            m_textOffsetXPx = std::max(0.0f, cursorX - horizontalMargin);
        } else if ((m_textOffsetXPx + m_textRegion->rect.Width() - horizontalMargin) < (cursorX + cursorSize.x)) {
            m_textOffsetXPx =
                cursorX + cursorSize.x - std::max(horizontalMargin, m_textRegion->rect.Width() - horizontalMargin);
        }

        m_textOffsetXPx = std::clamp(
            m_textOffsetXPx,
            0.0f,
            std::max(0.0f, m_textSizePx.x - m_textRegion->rect.Width()));
        m_cursorMoved = false;
    }

    void ZepWindow::GetCharPointer(GlyphIterator loc, const uint8_t*& pBegin, const uint8_t*& pEnd, SpecialChar& special) {
        static char invalidChar;
        static const char blankSpace = ' ';

        pBegin = &m_pBuffer->GetWorkingBuffer()[loc.Index()];

        // Shown only one char for end of line
        special = SpecialChar::None;
        if (*pBegin == '\n' || *pBegin == 0) {
            invalidChar = '@' + *pBegin;
            if (GetWindowFlags() & WindowFlags::ShowCR) {
                pBegin = (const uint8_t*)&invalidChar;
            } else {
                pBegin = (const uint8_t*)&blankSpace;
            }
            special = SpecialChar::Hidden;
        } else if (*pBegin == '\t') {
            special = SpecialChar::Tab;
        } else if (*pBegin == ' ') {
            special = SpecialChar::Space;
        }

        pEnd = pBegin + utf8_codepoint_length(*pBegin);
    }

    NVec2f ZepWindow::ArrangeLineMarkers(tRangeMarkers& markers) {
        // Account for markers
        auto margins = DPI_VEC2(GetEditor().GetConfig().widgetMargins);
        auto underlineHeight = DPI_Y(GetEditor().GetConfig().underlineHeight) + DPI_Y(UnderlineMargin * 2.0f);
        NVec2f height(0.0f);

        uint32_t lineWidgetCount = 0;
        bool underPad = false;
        std::vector<ByteIndex> markerStack;
        for (auto& [index, markerSet] : markers) {
            for (auto& spMarker : markerSet) {
                if (spMarker->markerType == RangeMarkerType::LineWidget) {
                    auto size = DPI_VEC2(spMarker->spWidget->GetSize());

                    // Each widget has a margin then its height then the bottom
                    height.x += margins.x;
                    height.x += size.y;
                    height.x += margins.y;

                    spMarker->displayRow = lineWidgetCount++;
                }

                if (spMarker->displayType & RangeMarkerDisplayType::Underline) {
                    // Stack the markers packed
                    uint32_t row = 0;
                    bool found = false;
                    for (auto& stack : markerStack) {
                        if (stack <= spMarker->GetRange().first) {
                            stack = spMarker->GetRange().second;
                            found = true;
                            break;
                        }
                        row++;
                    }

                    if (!found) {
                        markerStack.push_back(spMarker->GetRange().second);
                        row = uint32_t(markerStack.size()) - 1;

                        // Make the height bigger due to new row depth
                        height.y += underlineHeight;
                    }

                    // Underlines get an extra space underneath to make it clear they are under and not over!
                    if (!underPad) {
                        height.y += 1.0f;
                        underPad = true;
                    }

                    spMarker->displayRow = row;
                }
            }
        }

        return height;
    }

    // This is the most expensive part of window update; applying line span generation for wrapped text and unicode
    // character sizes which may vary in byte count and physical pixel width
    // It can take about a millisecond to do during editing on release buildj; but this is fast enough for now.
    // There are several ways in which this function can be optimized:
    // - Only regenerate spans which have changed, since only the edited line range will change.
    // - Generate blocks of text, based on syntax highlighting, instead of single characters.
    // - Have a no-wrap text mode and save a lot of the wrapping work.
    // - Do some threading
    void ZepWindow::UpdateLineSpans() {
        TIME_SCOPE(UpdateLineSpans);

        m_maxDisplayLines = (long)std::max(0.0f, std::floor(m_textRegion->rect.Height() / m_defaultLineSize));

        const auto& textBuffer = m_pBuffer->GetWorkingBuffer();

        long bufferLine = 0;
        long spanLine = 0;
        float bufferPosYPx = 0.0f;
        float xOffset = m_xPad;

        bool isMarkdown = m_pBuffer->GetFileExtension() == ".md";

        // Nuke the existing spans
        // In future we can in-place modify for speed
        std::for_each(m_windowLines.begin(), m_windowLines.end(), [](SpanInfo* pInfo) { delete pInfo; });
        m_windowLines.clear();

        auto widgetMarkers = m_pBuffer->GetRangeMarkers(RangeMarkerType::Widget);
        auto itrWidgetMarkers = widgetMarkers.begin();

        // Process every buffer line
        for (;;) {
            // We haven't processed this line yet, so we can't display anything
            // else
            if (m_pBuffer->GetLineCount() <= bufferLine)
                break;

            ByteRange lineByteRange;
            if (!m_pBuffer->GetLineOffsets(bufferLine, lineByteRange))
                break;

            // Padding at the top of the line
            NVec2f topPadding = NVec2f(DPI_Y((float)GetEditor().GetConfig().lineMargins.x), DPI_Y((float)GetEditor().GetConfig().lineMargins.y));

            auto markersOnLine = m_pBuffer->GetRangeMarkersOnLine(RangeMarkerType::All, bufferLine);
            auto lineWidgetHeight = ArrangeLineMarkers(markersOnLine);

            // Move the line down by the height of the widget
            bufferPosYPx += lineWidgetHeight.x;

            // TODO: Find a clean way to do this extra work during layout for extensions that need it
            ZepTextType type = ZepTextType::Text;
            if (isMarkdown) {
                uint32_t headerCount = 0;
                // Markdown experiment
                for (auto ch = lineByteRange.first; ch < lineByteRange.second; ch += utf8_codepoint_length(textBuffer[ch])) {
                    if (textBuffer[ch] != '#')
                        break;
                    headerCount++;
                }

                switch (headerCount) {
                case 0:
                    break;
                case 1:
                    type = ZepTextType::Heading1;
                    break;
                case 2:
                    type = ZepTextType::Heading2;
                    break;
                case 3:
                    type = ZepTextType::Heading3;
                    break;
                }
                // !Markdown experiment
            }

            auto& font = GetEditor().GetDisplay().GetFont(type);
            int textHeight = font.GetPixelHeight();

            // text line height is top/bottom pad
            float fullLineHeight = textHeight + topPadding.x + topPadding.y;

            // Start a new line
            SpanInfo* lineInfo = new SpanInfo();
            lineInfo->pFont = &font;
            lineInfo->lineWidgetHeights = lineWidgetHeight;
            lineInfo->bufferLineNumber = bufferLine;
            lineInfo->spanLineIndex = spanLine;
            lineInfo->lineByteRange.first = lineByteRange.first;
            lineInfo->lineByteRange.second = lineByteRange.first;
            lineInfo->yOffsetPx = bufferPosYPx;
            lineInfo->padding = topPadding;
            lineInfo->lineTextSizePx.x = xOffset;
            lineInfo->lineTextSizePx.y = float(textHeight);
            lineInfo->isSplitContinuation = false;

            auto inlineMargins = DPI_VEC2(GetEditor().GetConfig().inlineWidgetMargins);

            // These offsets are 0 -> n + 1, i.e. the last offset the buffer returns is 1 beyond the current
            // Note: Must not use pointers into the character buffer!
            for (auto ch = lineByteRange.first; ch < lineByteRange.second; ch += utf8_codepoint_length(textBuffer[ch])) {
                const uint8_t* pCh = &textBuffer[ch];
                auto textSize = font.GetCharSize(pCh);

                // Skip to current marker
                while (itrWidgetMarkers != widgetMarkers.end() && itrWidgetMarkers->first < ch) {
                    itrWidgetMarkers++;
                }

                if (itrWidgetMarkers != widgetMarkers.end()) {
                    if (itrWidgetMarkers->first == ch) {
                        for (auto& pWidget : itrWidgetMarkers->second) {
                            NVec2f inlineSize = pWidget->GetInlineSize();
                            inlineSize.x = inlineMargins.x * 2 + textHeight;
                            xOffset += inlineSize.x;
                            pWidget->SetInlineSize(inlineSize);
                        }
                        lineInfo->lineTextSizePx.x = xOffset;
                    }
                }

                // Wrap if we have displayed at least one char, and we are wrapping.
                // Don't wrap just for the CR
                if (ZTestFlags(GetWindowFlags(), WindowFlags::WrapText) && ch != lineByteRange.first && *pCh != '\n' && *pCh != 0) {
                    // At least a single char has wrapped; close the old line, start a new one
                    if (((xOffset + textSize.x) + textSize.x) >= (m_textRegion->rect.Width())) {
                        // Remember the offset beyond the end of the line
                        lineInfo->lineByteRange.second = ch;
                        lineInfo->lineTextSizePx.x = xOffset;
                        m_windowLines.push_back(lineInfo);

                        // Next line
                        lineInfo = new SpanInfo();
                        spanLine++;
                        bufferPosYPx += fullLineHeight + lineWidgetHeight.y;

                        // Reset the line margin and height, because when we split a line we don't include a
                        // custom widget space above it.  That goes just above the first part of the line
                        topPadding.x = (float)GetEditor().GetConfig().lineMargins.x;
                        fullLineHeight = textHeight + topPadding.x + topPadding.y;

                        // Now jump to the next 'screen line' for the rest of this 'buffer line'
                        lineInfo->lineByteRange = ByteRange(ch, ch + utf8_codepoint_length(textBuffer[ch]));
                        lineInfo->spanLineIndex = spanLine;
                        lineInfo->bufferLineNumber = bufferLine;
                        lineInfo->yOffsetPx = bufferPosYPx;
                        lineInfo->padding = topPadding;
                        lineInfo->lineTextSizePx.y = float(textHeight);
                        lineInfo->lineTextSizePx.x = xOffset;
                        lineInfo->isSplitContinuation = true;
                        lineInfo->pFont = &font;

                        xOffset = m_xPad;
                    } else {
                        xOffset += textSize.x + m_xPad;
                    }
                } else {
                    xOffset += textSize.x + m_xPad;
                }

                if (*pCh == '\n' && !ZTestFlags(GetWindowFlags(), WindowFlags::ShowCR)) {
                    xOffset -= (textSize.x + m_xPad);
                }

                if (*pCh == 0) {
                    xOffset -= (textSize.x + m_xPad);
                }

                lineInfo->yOffsetPx = bufferPosYPx;
                lineInfo->lineByteRange.second = ch + utf8_codepoint_length(textBuffer[ch]);
                lineInfo->lineTextSizePx.x = std::max(lineInfo->lineTextSizePx.x, xOffset);
            }

            // Complete the line
            m_windowLines.push_back(lineInfo);

            // Next time round - down a buffer line, down a span line
            bufferLine++;
            spanLine++;
            xOffset = m_xPad;
            bufferPosYPx += fullLineHeight + lineWidgetHeight.y;
        }

        // Sanity
        if (m_windowLines.empty()) {
            SpanInfo* lineInfo = new SpanInfo();
            lineInfo->lineByteRange.first = 0;
            lineInfo->lineByteRange.second = 0;
            lineInfo->padding = NVec2f(0.0f);
            lineInfo->lineTextSizePx = NVec2f(0.0f);
            lineInfo->bufferLineNumber = 0;
            m_windowLines.push_back(lineInfo);
        }

        // Now build the codepoint offsets
        for (auto& line : m_windowLines) {
            auto ch = line->lineByteRange.first;

            // TODO: Optimize
            line->lineCodePoints.clear();
            uint32_t points = 0;
            while (ch < line->lineByteRange.second) {
                LineCharInfo info;

                // Important note: We can't navigate the text buffer by pointers!
                // The gap buffer will get in the way; so need to be careful to use [] or an iterator
                // GetCharSize is cached for speed on debug builds.
                info.iterator = GlyphIterator(m_pBuffer, ch);
                info.size = line->pFont->GetCharSize(&textBuffer[ch]);
                line->lineCodePoints.push_back(info);
                ch += utf8_codepoint_length(textBuffer[ch]);
                points++;
            }
        }

        UpdateVisibleLineRange();
        m_layoutDirty = true;
    }

    void ZepWindow::UpdateVisibleLineRange() {
        TIME_SCOPE(UpdateVisibleLineRange);

        m_visibleLineIndices.x = (long)m_windowLines.size();
        m_visibleLineIndices.y = 0;
        m_textSizePx.x = 0;
        for (long line = 0; line < long(m_windowLines.size()); line++) {
            auto& windowLine = *m_windowLines[line];
            m_textSizePx.x = std::max(m_textSizePx.x, windowLine.lineTextSizePx.x);

            if (windowLine.yOffsetPx < m_textOffsetPx) {
                continue;
            }

            if ((windowLine.yOffsetPx) > (m_textOffsetPx + m_textRegion->rect.Height())) {
                break;
            }

            m_visibleLineIndices.x = std::min(m_visibleLineIndices.x, long(line));
            m_visibleLineIndices.y = long(line);
        }

        // This seems
        m_visibleLineIndices.y++;

        m_textSizePx.y = m_windowLines[m_windowLines.size() - 1]->yOffsetPx + GetEditor().GetDisplay().GetFont(ZepTextType::Text).GetPixelHeight() + DPI_Y(GetEditor().GetConfig().lineMargins.y) + DPI_Y(GetEditor().GetConfig().lineMargins.x);
        m_textOffsetXPx = std::clamp(
            m_textOffsetXPx,
            0.0f,
            std::max(0.0f, m_textSizePx.x - m_textRegion->rect.Width()));

        UpdateScrollers();
    }

    const SpanInfo& ZepWindow::GetCursorLineInfo(long y) {
        UpdateLayout();
        y = std::max(0l, y);
        y = std::min(y, long(m_windowLines.size() - 1));
        return *m_windowLines[y];
    }

    // Convert a normalized y coordinate to the window region
    float ZepWindow::ToWindowY(float pos) const {
        return pos - m_textOffsetPx + m_textRegion->rect.topLeftPx.y;
    }

    float ZepWindow::ToWindowX(float pos) const {
        return pos - m_textOffsetXPx + m_textRegion->rect.topLeftPx.x;
    }

    float ZepWindow::TipBoxShadowWidth() const {
        return DPI_X(4.0f);
    }

    void ZepWindow::DisplayToolTip(const NVec2f& pos, const RangeMarker& marker) const {
        auto textSize = GetEditor().GetDisplay().GetFont(ZepTextType::Text).GetTextSize((const uint8_t*)marker.GetDescription().c_str(), (const uint8_t*)(marker.GetDescription().c_str() + marker.GetDescription().size()));

        auto boxShadowWidth = TipBoxShadowWidth();

        // Draw a black area a little wider than the tip box.
        NRectf tipBox(pos.x, pos.y, textSize.x, textSize.y);
        tipBox.Adjust(0, 0, (textBorder + boxShadowWidth) * 2, (textBorder + boxShadowWidth) * 2);

        auto& display = GetEditor().GetDisplay();

        // Don't clip the scroll bar
        auto clip = m_expandingEditRegion->rect;
        clip.SetSize(clip.Size() - NVec2f(m_vScrollRegion->rect.Width(), 0.0f));
        display.SetClipRect(clip);
        display.DrawRectFilled(tipBox, m_pBuffer->GetTheme().GetColor(ThemeColor::Background));

        // Draw a lighter inner and a border the same color as the marker theme
        tipBox.Adjust(boxShadowWidth, boxShadowWidth, -boxShadowWidth, -boxShadowWidth);
        display.DrawRectFilled(tipBox, m_pBuffer->GetTheme().GetColor(marker.GetBackgroundColor()));
        display.DrawLine(tipBox.topLeftPx, tipBox.TopRight(), m_pBuffer->GetTheme().GetColor(marker.GetHighlightColor()));
        display.DrawLine(tipBox.BottomLeft(), tipBox.bottomRightPx, m_pBuffer->GetTheme().GetColor(marker.GetHighlightColor()));
        display.DrawLine(tipBox.topLeftPx, tipBox.BottomLeft(), m_pBuffer->GetTheme().GetColor(marker.GetHighlightColor()));
        display.DrawLine(tipBox.TopRight(), tipBox.bottomRightPx, m_pBuffer->GetTheme().GetColor(marker.GetHighlightColor()));

        // Draw the text in the box
        display.DrawChars(display.GetFont(ZepTextType::Text), tipBox.topLeftPx + NVec2f(textBorder, textBorder), m_pBuffer->GetTheme().GetColor(marker.GetTextColor()), (const uint8_t*)marker.GetDescription().c_str());
    }

    NVec4f ZepWindow::ModifyBackgroundColor(ThemeColor color) const {
        auto col = m_pBuffer->GetTheme().GetColor(color);
        return GetEditor().ModifyBackgroundColor(col);
    }

    void ZepWindow::DrawAboveLineWidgets(SpanInfo& lineInfo) {
        if (lineInfo.isSplitContinuation)
            return;

        auto markers = m_pBuffer->GetRangeMarkersOnLine(RangeMarkerType::LineWidget, lineInfo.bufferLineNumber);
        if (markers.empty()) {
            return;
        }

        auto widgetMargins = DPI_VEC2(GetEditor().GetConfig().widgetMargins);

        float widgetHeight = lineInfo.lineWidgetHeights.x;
        NVec2f linePx = GetSpanPixelRange(lineInfo);
        float currentY = 0.0f; // lineMargins.x;

        for (auto& [index, markerSet] : markers) {
            for (auto& spMarker : markerSet) {
                if (spMarker->spWidget == nullptr) {
                    continue;
                }

                auto widgetSize = DPI_VEC2(spMarker->spWidget->GetSize());

                currentY += widgetMargins.x;
                spMarker->spWidget->Draw(*m_pBuffer, NVec2f(linePx.x, ToWindowY(currentY + lineInfo.yOffsetPx - widgetHeight)));
                currentY += widgetSize.y;
                currentY += widgetMargins.y;
            }
        }
    }

    NRectf SquareRect(const NRectf& rc) {
        auto minSize = std::min(rc.Width(), rc.Height());
        auto xMargin = (rc.Width() - minSize) / 2;
        auto yMargin = (rc.Height() - minSize) / 2;
        return NRectf(rc.topLeftPx + NVec2f(xMargin, yMargin), rc.bottomRightPx - NVec2f(xMargin, yMargin));
    }

    void ZepWindow::UpdateMarkers() {
        bool foundFlash = false;

        std::vector<std::shared_ptr<RangeMarker>> victims;
        m_pBuffer->ForEachMarker(RangeMarkerType::All, Direction::Forward, GlyphIterator(m_pBuffer, 0), GlyphIterator(m_pBuffer, m_pBuffer->End().Index()), [&](const std::shared_ptr<RangeMarker>& marker) {
            // Don't show hidden markers
            if (!(marker->displayType & RangeMarkerDisplayType::Timed)) {
                return true;
            }

            auto elapsed = timer_get_elapsed_seconds(marker->timer);
            if (elapsed < marker->duration) {
                // Swap it out for our custom flash color
                float time = float(elapsed) / marker->duration;
                marker->SetAlpha(sin(time * ZPI));

                foundFlash = true;
                marker->displayType &= ~RangeMarkerDisplayType::Hidden;
            } else {
                marker->SetAlpha(0.0f);
                marker->displayType |= RangeMarkerDisplayType::Hidden;
                victims.push_back(marker);
            }
            return true;
        });

        for (auto& pVictim : victims) {
            m_pBuffer->ClearRangeMarker(pVictim);
        }

        if (!foundFlash) {
            GetEditor().SetFlags(ZClearFlags(GetEditor().GetFlags(), ZepEditorFlags::FastUpdate));
        }
    }

    void ZepWindow::DisplayLineBackground(SpanInfo& lineInfo, ZepSyntax* pSyntax) {
        auto& display = GetEditor().GetDisplay();

        auto underlineHeight = DPI_Y(GetEditor().GetConfig().underlineHeight);
        auto inlineMargins = DPI_VEC2(GetEditor().GetConfig().inlineWidgetMargins);
        auto screenPosX = ToWindowX(m_xPad);
        auto widgetMarkers = m_pBuffer->GetRangeMarkers(RangeMarkerType::Widget);
        auto itrWidgetMarkers = widgetMarkers.begin();
        auto tipTimeSeconds = timer_get_elapsed_seconds(m_toolTipTimer);

        NVec2f linePx = GetSpanPixelRange(lineInfo);

        NVec4f backColor;
        // Fill entire line background
        if (lineInfo.lineByteRange.ContainsLocation(GetBufferCursor().Index()) && IsActiveWindow()) {
            backColor = ModifyBackgroundColor(ThemeColor::CursorLineBackground);
        } else {
            backColor = ModifyBackgroundColor(ThemeColor::Background);
        }

        if (GetEditor().GetConfig().style == EditorStyle::Minimal) {
            auto copyLinePx = linePx;
            if (lastLinePx.y > linePx.y) {
                linePx.y = lastLinePx.y;
            }
            lastLinePx = copyLinePx;
        }

        // Note; We fill below the line for underlines for now, to make them standout in minimal mode
        display.DrawRectFilled(
            NRectf(
                NVec2f(linePx.x, ToWindowY(lineInfo.yOffsetPx)),
                NVec2f(linePx.y, ToWindowY(lineInfo.yOffsetPx + lineInfo.FullLineHeightPx() + lineInfo.lineWidgetHeights.y))),
            backColor);

        // Walk from the start of the line to the end of the line (in buffer chars)
        for (auto& cp : lineInfo.lineCodePoints) {
            NRectf charRect(NVec2f(screenPosX, ToWindowY(lineInfo.yOffsetPx)), NVec2f(screenPosX + cp.size.x, ToWindowY(lineInfo.yOffsetPx + lineInfo.FullLineHeightPx())));

            // If the syntax overrides the background, show it first, and underneath a marker or char that might come next
            if (pSyntax) {
                auto syntaxResult = pSyntax->GetSyntaxAt(cp.iterator);
                if (syntaxResult.background != ThemeColor::None) {
                    auto syntaxColor = pSyntax->ToBackgroundColor(syntaxResult);
                    display.DrawRectFilled(charRect, syntaxColor);
                }
            }

            // Skip to current marker
            while (itrWidgetMarkers != widgetMarkers.end() && itrWidgetMarkers->first < cp.iterator.Index()) {
                itrWidgetMarkers++;
            }

            // Draw inline widget marks
            if (itrWidgetMarkers != widgetMarkers.end()) {
                if (itrWidgetMarkers->first == cp.iterator.Index()) {
                    for (auto pMarker : itrWidgetMarkers->second) {
                        // Tell the widget to draw inline
                        NRectf inlineRect(NVec2f(screenPosX, ToWindowY(lineInfo.yOffsetPx)), NVec2f(screenPosX + pMarker->GetInlineSize().x, ToWindowY(lineInfo.yOffsetPx + lineInfo.FullLineHeightPx())));

                        // If the syntax overrides the background, show it first
                        if (pSyntax) {
                            auto syntaxResult = pSyntax->GetSyntaxAt(cp.iterator);
                            if (syntaxResult.background != ThemeColor::None) {
                                auto themeCol = pSyntax->ToBackgroundColor(syntaxResult);
                                display.DrawRectFilled(inlineRect, themeCol);
                            }
                        }

                        // Draw the inline marker and advance the text position
                        inlineRect.Adjust(inlineMargins.x, inlineMargins.y, -inlineMargins.x, -inlineMargins.y);
                        inlineRect = SquareRect(inlineRect);
                        pMarker->spWidget->DrawInline(*m_pBuffer, inlineRect);
                        screenPosX += pMarker->GetInlineSize().x;
                    }
                }
            }

            // Store the actual location of the text codepoint
            cp.pos = NVec2f(screenPosX, ToWindowY(lineInfo.yOffsetPx));

            // Background and underlines
            // Track the background color for multiple overlapping markers and blend the alpha correctly by
            // doing a mix between the previous color and the new one.
            NVec4f backgroundColor = backColor;

            m_pBuffer->ForEachMarker(RangeMarkerType::All, Direction::Forward, GlyphIterator(m_pBuffer, lineInfo.lineByteRange.first), GlyphIterator(m_pBuffer, lineInfo.lineByteRange.second), [&](const std::shared_ptr<RangeMarker>& marker) {
                // Don't show hidden markers
                if (marker->displayType & RangeMarkerDisplayType::Hidden) {
                    return true;
                }

                auto sel = marker->GetRange();
                if (marker->ContainsLocation(cp.iterator)) {
                    if (marker->markerType == RangeMarkerType::Mark || marker->markerType == RangeMarkerType::Search) {
                        // Draw lines under the text
                        if (marker->displayType & RangeMarkerDisplayType::Underline) {
                            float offset = lineInfo.yOffsetPx + lineInfo.FullLineHeightPx();
                            offset += marker->displayRow * (DPI_Y(UnderlineMargin * 2) + underlineHeight) + 1.0f; // Margins & an extra line to seperate from background highlight
                            display.DrawRectFilled(
                                NRectf(NVec2f(screenPosX, ToWindowY(offset)),
                                       NVec2f(screenPosX + cp.size.x, ToWindowY(offset + underlineHeight))),
                                m_pBuffer->GetTheme().GetColor(marker->GetHighlightColor()));
                        }

                        // Fill the background of the text with the marker color
                        if (marker->displayType & RangeMarkerDisplayType::Background) {
                            auto markerBack = marker->GetBackgroundColor(cp.iterator);
                            if (markerBack != ThemeColor::None) {
                                auto markerBackColor = m_pBuffer->GetTheme().GetColor(markerBack);
                                backgroundColor = Mix(backgroundColor, markerBackColor, marker->GetAlpha(cp.iterator));
                                display.DrawRectFilled(charRect, backgroundColor);
                            }
                        }
                    }

                    // If this marker has an associated tooltip, pop it up after a time delay
                    // TODO: Make tooltip generation seperate to this display loop
                    if (m_toolTips.empty() && !m_tipDisabledTillMove && (tipTimeSeconds > 0.5f)) {
                        bool showTip = false;
                        if (marker->displayType & RangeMarkerDisplayType::Tooltip) {
                            if (m_mouseBufferLocation == cp.iterator) {
                                showTip = true;
                            }
                        }

                        // If we want the tip showing at anywhere on the line, show it
                        if (marker->displayType & RangeMarkerDisplayType::TooltipAtLine) {
                            // TODO: This should be a helper function
                            // Checks for mouse pos inside a line string
                            if (m_mouseHoverPos.y >= ToWindowY(lineInfo.yOffsetPx) &&
                                m_mouseHoverPos.y < (ToWindowY(lineInfo.yOffsetPx) + cp.size.y) &&
                                m_mouseHoverPos.x < linePx.y) {
                                showTip = true;
                            }
                        }

                        if (showTip) {
                            // Register this tooltip
                            m_toolTips[NVec2f(m_mouseHoverPos.x, m_mouseHoverPos.y + textBorder)] = marker;
                        }
                    }
                }
                return true;
            });

            screenPosX += cp.size.x + m_xPad;
        }
    }

    void ZepWindow::DisplayLineNumbers() {
        auto cursorCL = BufferToDisplay();
        auto& display = GetEditor().GetDisplay();

        if (m_numberRegion->rect.Width() > 0) {
            for (long windowLine = m_visibleLineIndices.x; windowLine < m_visibleLineIndices.y; windowLine++) {
                auto& lineInfo = *m_windowLines[windowLine];

                if (!IsInsideVisibleText(NVec2i(0, lineInfo.spanLineIndex)))
                    return;

                auto cursorBufferLine = GetCursorLineInfo(cursorCL.y).bufferLineNumber;
                std::string strNum;

                auto mode = m_pBuffer->GetMode();

                // In Vim mode show relative lines, unless in Ex mode (with hidden cursor)
                if (mode->UsesRelativeLines() && mode->GetCursorType() != CursorType::None) {
                    strNum = std::to_string(std::abs(lineInfo.bufferLineNumber - cursorBufferLine));
                } else {
                    strNum = std::to_string(lineInfo.bufferLineNumber + 1);
                }

                auto& numFont = display.GetFont(ZepTextType::UI);
                auto textSize = numFont.GetTextSize((const uint8_t*)strNum.c_str(), (const uint8_t*)(strNum.c_str() + strNum.size()));
                auto lineCenter = (lineInfo.FullLineHeightPx() * .5f) + lineInfo.yOffsetPx;

                auto digitCol = m_pBuffer->GetTheme().GetColor(ThemeColor::LineNumber);
                if (lineInfo.BufferCursorInside(m_bufferCursor)) {
                    digitCol = m_pBuffer->GetTheme().GetColor(ThemeColor::CursorNormal);
                }

                if (m_numberRegion->rect.Width() > 0) {
                    // Numbers
                    display.SetClipRect(m_numberRegion->rect);
                    display.DrawChars(numFont,
                                      NVec2f(m_numberRegion->rect.bottomRightPx.x - textSize.x,
                                             ToWindowY(lineCenter - numFont.GetPixelHeight() * .5f)),
                                      digitCol,
                                      (const uint8_t*)strNum.c_str(), (const uint8_t*)(strNum.c_str() + strNum.size()));
                }

                if (m_indicatorRegion->rect.Width() > 0) {
                    // Show any markers in the left indicator region
                    m_pBuffer->ForEachMarker(RangeMarkerType::Mark, Direction::Forward, GlyphIterator(m_pBuffer, lineInfo.lineByteRange.first), GlyphIterator(m_pBuffer, lineInfo.lineByteRange.second), [&](const std::shared_ptr<RangeMarker>& marker) {
                        // >|< Text.  This is the bit between the arrows <-.  A vertical bar in the 'margin'
                        if (marker->displayType & RangeMarkerDisplayType::Indicator) {
                            if (marker->IntersectsRange(lineInfo.lineByteRange)) {
                                display.SetClipRect(m_indicatorRegion->rect);
                                display.DrawRectFilled(
                                    NRectf(
                                        NVec2f(
                                            m_indicatorRegion->rect.Center().x - m_indicatorRegion->rect.Width() / 4,
                                            ToWindowY(lineInfo.yOffsetPx + lineInfo.padding.x)),
                                        NVec2f(
                                            m_indicatorRegion->rect.Center().x + m_indicatorRegion->rect.Width() / 4,
                                            ToWindowY(lineInfo.yOffsetPx + lineInfo.padding.x) + display.GetFont(ZepTextType::Text).GetPixelHeight())),
                                    m_pBuffer->GetTheme().GetColor(marker->GetHighlightColor()));
                            }
                        }
                        return true;
                    });
                }
            }
        }
    }

    // TODO: This function draws one char at a time.  It could be more optimal at the expense of some
    // complexity.  Basically, I don't like the current implementation, but it works for now.
    // The text is displayed acorrding to the region bounds and the display lineData
    // Additionally (and perhaps that should be a seperate function), this code draws line numbers
    bool ZepWindow::DisplayLine(SpanInfo& lineInfo, int displayPass) {
        static const auto blankSpace = ' ';

        auto& buffer = GetBuffer();
        auto pMode = buffer.GetMode();
        if (!pMode) {
            return false;
        }

        auto cursorCL = BufferToDisplay();
        auto& display = GetEditor().GetDisplay();
        auto pSyntax = m_pBuffer->GetSyntax();
        auto cursorBlink = GetEditor().GetCursorBlinkState();
        auto cursorType = GetBuffer().GetMode()->GetCursorType();
        auto defaultCharSize = display.GetFont(ZepTextType::Text).GetDefaultCharSize();
        auto dotSize = display.GetFont(ZepTextType::Text).GetDotSize();
        auto whiteSpaceCol = m_pBuffer->GetTheme().GetColor(ThemeColor::Whitespace);
        auto height = lineInfo.FullLineHeightPx();
        bool isLineHovered = false;

        // Drawing commands for the whole line
        if (displayPass == WindowPass::Background) {
            display.SetClipRect(m_textRegion->rect);
            DisplayLineBackground(lineInfo, pSyntax);
            NRectf lineRect(NVec2f(m_textRegion->rect.Left(), ToWindowY(lineInfo.yOffsetPx)), NVec2f(m_textRegion->rect.Right(), ToWindowY(lineInfo.yOffsetPx + height)));
            isLineHovered = lineRect.Contains(m_mousePos);
        }

        display.SetClipRect(m_textRegion->rect);

        bool lineStart = true;
        bool hasBeenHovered = false;

        // Walk from the start of the line to the end of the line (in buffer chars)
        for (auto& cp : lineInfo.lineCodePoints) {
            const uint8_t* pCh;
            const uint8_t* pEnd;
            SpecialChar special;
            GetCharPointer(cp.iterator, pCh, pEnd, special);
            bool isHovered = false;
            bool isLast = &cp == &lineInfo.lineCodePoints.back();

            // TODO : Cache this for speed - a little sluggish on debug builds.
            if (displayPass == WindowPass::Background) {
                NRectf charRect(NVec2f(cp.pos.x, ToWindowY(lineInfo.yOffsetPx)), NVec2f(cp.pos.x + cp.size.x, ToWindowY(lineInfo.yOffsetPx + height)));
                if (charRect.Contains(m_mousePos) || (isLast && !hasBeenHovered && isLineHovered)) {
                    m_mouseIterator = cp.iterator;
                    isHovered = true;
                    hasBeenHovered = true;
                }

                if (charRect.Contains(m_mouseHoverPos)) {
                    // Record the mouse-over buffer location
                    m_mouseBufferLocation = cp.iterator;
                }

                // Draw the visual selection marker second
                if (IsActiveWindow()) {
                    if (GetBuffer().HasSelection()) {
                        auto sel = m_pBuffer->GetInclusiveSelection();

                        // Visual selection is 'inclusive' - it starts/ends on the cursor
                        if (sel.ContainsInclusiveLocation(cp.iterator)) {
                            display.DrawRectFilled(NRectf(NVec2f(cp.pos.x, ToWindowY(lineInfo.yOffsetPx)), NVec2f(cp.pos.x + cp.size.x, ToWindowY(lineInfo.yOffsetPx + lineInfo.FullLineHeightPx()))), m_pBuffer->GetTheme().GetColor(ThemeColor::VisualSelectBackground));
                        }
                    }
                }

                // If active window and this is the cursor char then display the marker as a priority over what we would have shown
                if (IsActiveWindow() && (cp.iterator == m_bufferCursor) && (!cursorBlink || cursorType == CursorType::LineMarker)) {
                    switch (cursorType) {
                    default:
                    case CursorType::None:
                        break;

                    case CursorType::LineMarker: {
                        display.SetClipRect(NRectf());
                        auto posX = m_indicatorRegion->rect.Right() - DPI_X(2.0f);
                        GetEditor().GetDisplay().DrawRectFilled(NRectf(
                                                                    NVec2f(posX, ToWindowY(lineInfo.yOffsetPx)),
                                                                    NVec2f(posX + DPI_X(2.0f), ToWindowY(lineInfo.yOffsetPx + height))),
                                                                m_pBuffer->GetTheme().GetColor(ThemeColor::CursorNormal));
                        display.SetClipRect(m_textRegion->rect);
                    } break;

                    case CursorType::Insert: {
                        GetEditor().GetDisplay().DrawRectFilled(NRectf(
                                                                    NVec2f(cp.pos.x, ToWindowY(lineInfo.yOffsetPx)),
                                                                    NVec2f(cp.pos.x + DPI_X(1.0f), ToWindowY(lineInfo.yOffsetPx + height))),
                                                                m_pBuffer->GetTheme().GetColor(ThemeColor::CursorInsert));
                    } break;

                    case CursorType::Normal:
                    case CursorType::Visual: {
                        GetEditor().GetDisplay().DrawRectFilled(NRectf(
                                                                    NVec2f(cp.pos.x, ToWindowY(lineInfo.yOffsetPx)),
                                                                    NVec2f(cp.pos.x + cp.size.x, ToWindowY(lineInfo.yOffsetPx + height))),
                                                                m_pBuffer->GetTheme().GetColor(ThemeColor::CursorNormal));
                    } break;
                    }
                } else if (isHovered) {
                    GetEditor().GetDisplay().DrawRectFilled(NRectf(
                                                                NVec2f(cp.pos.x, ToWindowY(lineInfo.yOffsetPx)),
                                                                NVec2f(cp.pos.x + cp.size.x, ToWindowY(lineInfo.yOffsetPx + height))),
                                                            m_pBuffer->GetTheme().GetColor(ThemeColor::AirlineBackground));
                }
            }
            // Second pass, characters
            else {
                if (lineStart) {
                    DrawAboveLineWidgets(lineInfo);
                }

                if ((special != SpecialChar::Hidden) || (GetWindowFlags() & WindowFlags::ShowCR)) {
                    auto centerY = ToWindowY(lineInfo.yOffsetPx) + cp.size.y / 2;
                    auto centerChar = NVec2f(cp.pos.x + cp.size.x / 2, centerY);
                    NVec4f col;
                    if (special == SpecialChar::Hidden) {
                        col = m_pBuffer->GetTheme().GetColor(ThemeColor::HiddenText);
                    } else {
                        if (pSyntax) {
                            auto syntaxResult = pSyntax->GetSyntaxAt(cp.iterator);
                            if (syntaxResult.foreground != ThemeColor::None) {
                                col = pSyntax->ToForegroundColor(syntaxResult);
                            } else {
                                col = m_pBuffer->GetTheme().GetColor(ThemeColor::Text);
                            }
                        } else {
                            col = m_pBuffer->GetTheme().GetColor(ThemeColor::Text);
                        }
                    }

                    // If this is the cursor char we override the colors
                    auto ws = whiteSpaceCol;
                    if (IsActiveWindow() && (cp.iterator == m_bufferCursor) && !cursorBlink && cursorType == CursorType::Normal) {
                        col = m_pBuffer->GetTheme().GetComplement(m_pBuffer->GetTheme().GetColor(ThemeColor::CursorNormal));
                        ws = col;
                    }

                    if (special == SpecialChar::None || special == SpecialChar::Hidden) {
                        display.DrawChars(*lineInfo.pFont, NVec2f(cp.pos.x, ToWindowY(lineInfo.yOffsetPx + lineInfo.padding.x)), col, pCh, pEnd);
                    } else if (special == SpecialChar::Tab) {
                        if (GetWindowFlags() & WindowFlags::ShowWhiteSpace) {
                            // A line and an arrow
                            display.DrawLine(NVec2f(cp.pos.x + defaultCharSize.x / 2, centerY), NVec2f(cp.pos.x + cp.size.x - defaultCharSize.x / 4, centerY), ws, 2);
                            display.DrawLine(NVec2f(cp.pos.x, ToWindowY(lineInfo.yOffsetPx)), NVec2f(cp.pos.x + defaultCharSize.x / 2, centerY), ws, 2);
                            display.DrawLine(NVec2f(cp.pos.x, ToWindowY(lineInfo.yOffsetPx + cp.size.y)), NVec2f(cp.pos.x + defaultCharSize.x / 2, centerY), ws, 2);
                        }
                    } else if (special == SpecialChar::Space) {
                        if (GetWindowFlags() & WindowFlags::ShowWhiteSpace) {
                            // A dot
                            display.DrawRectFilled(NRectf(centerChar - dotSize, centerChar + dotSize), ws);
                        }
                    }
                }
            }

            lineStart = false;
        }

        display.SetClipRect(NRectf{});

        return true;
    }

    bool ZepWindow::IsInsideVisibleText(NVec2i pos) const {
        if (pos.y < m_visibleLineIndices.x || pos.y >= m_visibleLineIndices.y) {
            return false;
        }
        return true;
    }

    ZepTabWindow& ZepWindow::GetTabWindow() const {
        return m_tabWindow;
    }

    void ZepWindow::SetWindowFlags(uint32_t windowFlags) {
        if (windowFlags != m_windowFlags) {
            m_windowFlags = windowFlags;

            m_layoutDirty = true;
        }
    }

    uint32_t ZepWindow::GetWindowFlags() const {
        auto flags = m_windowFlags;

        if (m_pBuffer && m_pBuffer->GetMode()) {
            flags = m_pBuffer->GetMode()->ModifyWindowFlags(flags);
        }

        return flags;
    }

    void ZepWindow::ToggleFlag(uint32_t flag) {
        if (ZTestFlags(m_windowFlags, flag)) {
            SetWindowFlags(m_windowFlags & ~flag);
        } else {
            SetWindowFlags(m_windowFlags | flag);
        }
    }

    long ZepWindow::GetMaxDisplayLines() {
        UpdateLayout();
        return m_maxDisplayLines;
    }

    long ZepWindow::GetNumDisplayedLines() {
        UpdateLayout();
        return std::min((long)m_windowLines.size(), GetMaxDisplayLines());
    }

    void ZepWindow::SetBufferCursor(GlyphIterator location) {
        // Don't move cursor if not necessary
        // This helps preserve 'lastCursorColumn' from being changed all the time
        // during line clamps, etc.
        if (location != m_bufferCursor) {
            m_bufferCursor = location.Clamped();
            m_lastCursorColumn = BufferToDisplay(m_bufferCursor).x;
            m_cursorMoved = true;
            DisableToolTipTillMove();
        }
        assert(!m_pBuffer || m_bufferCursor.Valid());
    }

    void ZepWindow::DisableToolTipTillMove() {
        m_tipDisabledTillMove = true;
        m_toolTips.clear();
    }

    void ZepWindow::SetBuffer(ZepBuffer* pBuffer) {
        assert(pBuffer);

        m_pBuffer = pBuffer;
        m_layoutDirty = true;
        m_textOffsetPx = 0;
        m_textOffsetXPx = 0;
        m_bufferCursor = pBuffer->GetLastEditLocation().Clamped();
        m_lastCursorColumn = 0;
        m_cursorMoved = false;
        if (pBuffer->GetMode()) {
            pBuffer->GetMode()->Begin(this);
        }
        GetEditor().UpdateTabs();
    }

    GlyphIterator ZepWindow::GetBufferCursor() {
        // Ensure cursor is always valid inside the buffer
        m_bufferCursor.Clamp();
        assert(!m_pBuffer || m_bufferCursor.Valid());
        return m_bufferCursor;
    }

    NRectf ZepWindow::GetCursorRect() {
        NVec2f pos;
        NVec2f size;
        GetCursorInfo(pos, size);
        return NRectf(pos, pos + size);
    }

    ZepBuffer& ZepWindow::GetBuffer() const {
        return *m_pBuffer;
    }

    bool ZepWindow::IsActiveWindow() const {
        return m_tabWindow.GetActiveWindow() == this;
    }

    NVec4f ZepWindow::FilterActiveColor(const NVec4f& col, float atten) {
        if (!IsActiveWindow()) {
            return NVec4f(Luminosity(col) * atten);
        }
        return col;
    }

    void ZepWindow::DisplayScrollers() {
        if (!m_vScrollRegion->rect.Empty()) {
            m_vScroller->Display(m_pBuffer->GetTheme());
        }
        if (!m_hScrollRegion->rect.Empty()) {
            m_hScroller->Display(m_pBuffer->GetTheme());
        }
        if (!m_scrollCornerRegion->rect.Empty()) {
            GetEditor().GetDisplay().DrawRectFilled(
                m_scrollCornerRegion->rect,
                m_pBuffer->GetTheme().GetColor(ThemeColor::WidgetBackground));
        }

        GetEditor().GetDisplay().SetClipRect(m_bufferRegion->rect);
    }

    void ZepWindow::DirtyLayout() {
        m_layoutDirty = true;
    }

    void ZepWindow::UpdateLayout(bool force) {
        if (m_layoutDirty || force) {
            // Border, and move the text across a bit
            if (ZTestFlags(GetWindowFlags(), WindowFlags::ShowLineNumbers) && GetEditor().GetConfig().showLineNumbers) {
                m_numberRegion->fixed_size = NVec2f(float(leftBorderChars) * GetEditor().GetDisplay().GetFont(ZepTextType::Text).GetDefaultCharSize().x, 0.0f);
            } else {
                m_numberRegion->fixed_size = NVec2f(0.0f);
            }

            if (ZTestFlags(GetWindowFlags(), WindowFlags::ShowIndicators) && GetEditor().GetConfig().showIndicatorRegion) {
                m_indicatorRegion->fixed_size = NVec2f(GetEditor().GetDisplay().GetFont(ZepTextType::Text).GetDefaultCharSize().x * 1.5f, 0.0f);
            } else {
                m_indicatorRegion->fixed_size = NVec2f(0.0f);
            }

            // When wrapping text, we fit the text to the available window space
            if (ZTestFlags(GetWindowFlags(), WindowFlags::WrapText)) {
                m_editRegion->flags = RegionFlags::Expanding;
                m_editRegion->fixed_size = NVec2f(0.0f);

                // First layout
                LayoutRegion(*m_bufferRegion);

                // Then update the text alignment
                UpdateLineSpans();
            } else {
                // First update the text, since it is always the same size without wrapping
                UpdateLineSpans();

                // Non-wrapped editors should still occupy the full pane and scroll within it.
                // Centering a fixed-size edit region creates large dead areas and misplaces the
                // horizontal scrollbar under the content instead of the viewport.
                m_editRegion->flags = RegionFlags::Expanding;
                m_editRegion->fixed_size = NVec2f(0.0f);

                LayoutRegion(*m_bufferRegion);

                // Finally, we have to update the line visibility again because the layout has changed!
                UpdateVisibleLineRange();
            }

            m_layoutDirty = false;
        }
    }

    NVec2f ZepWindow::GetSpanPixelRange(SpanInfo& span) const {
        // Need to take account of the text rect offset
        return NVec2f(ToWindowX(0.0f), ToWindowX(span.lineTextSizePx.x));
    }

    float ZepWindow::GetCursorTextX(const SpanInfo& cursorBufferLine, long cursorColumn, NVec2f* size) const {
        NVec2f cursorSize;
        bool found = false;
        float xPos = m_xPad;
        auto cursor = m_bufferCursor;
        cursor.Clamp();

        int count = 0;
        for (auto ch : cursorBufferLine.lineCodePoints) {
            if (count == cursorColumn) {
                found = true;
                cursorSize = ch.size;
                break;
            }
            count++;
            xPos += ch.size.x + m_xPad;
        }

        if (cursor.Char() == '\t') {
            cursorSize = GetEditor().GetDisplay().GetFont(ZepTextType::Text).GetDefaultCharSize();
        } else if (!found) {
            cursorSize = GetEditor().GetDisplay().GetFont(ZepTextType::Text).GetDefaultCharSize();
            xPos += cursorSize.x;
        }

        if (size != nullptr) {
            *size = cursorSize;
            size->y = cursorBufferLine.lineTextSizePx.y;
        }

        return xPos;
    }

    void ZepWindow::GetCursorInfo(NVec2f& pos, NVec2f& size) {
        auto cursorCL = BufferToDisplay();
        auto cursorBufferLine = GetCursorLineInfo(cursorCL.y);

        pos = NVec2f(ToWindowX(GetCursorTextX(cursorBufferLine, cursorCL.x, &size)),
                     cursorBufferLine.yOffsetPx + cursorBufferLine.padding.x - m_textOffsetPx + m_textRegion->rect.topLeftPx.y);
        size.y = cursorBufferLine.lineTextSizePx.y;
    }

    void ZepWindow::PlaceToolTip(const NVec2f& pos, ToolTipPos location, uint32_t lineGap, const std::shared_ptr<RangeMarker> spMarker) {
        auto textSize = GetEditor().GetDisplay().GetFont(ZepTextType::Text).GetTextSize((const uint8_t*)spMarker->GetDescription().c_str(), (const uint8_t*)(spMarker->GetDescription().c_str() + spMarker->GetDescription().size()));
        float boxShadowWidth = TipBoxShadowWidth();

        NRectf tipBox;
        float currentLineGap = lineGap + .5f;

        for (int i = 0; i < int(ToolTipPos::Count); i++) {
            auto genBox = [&]() {
                // Draw a black area a little wider than the tip box.
                tipBox = NRectf(pos.x, pos.y, textSize.x, textSize.y);
                tipBox.Adjust(textBorder + boxShadowWidth, textBorder + boxShadowWidth, textBorder + boxShadowWidth, textBorder + boxShadowWidth);

                float dist = currentLineGap * (GetEditor().GetDisplay().GetFont(ZepTextType::Text).GetPixelHeight() + textBorder * 2);
                if (location == ToolTipPos::AboveLine) {
                    dist += textSize.y;
                    tipBox.Adjust(0.0f, -dist);
                } else if (location == ToolTipPos::BelowLine) {
                    tipBox.Adjust(0.0f, dist);
                }
            };

            genBox();
            if (!NRectFits(m_textRegion->rect, tipBox, FitCriteria::X)) {
                // If it is above or below, slide it to the left to fit
                if (location != ToolTipPos::RightLine) {
                    // Move in X along to the left
                    genBox();
                    tipBox.Move(std::max(m_textRegion->rect.Left() + textBorder, (m_textRegion->rect.Right() - (tipBox.Width() + textBorder))), tipBox.Top());
                }
            }

            // Swap above below
            if (!NRectFits(m_textRegion->rect, tipBox, FitCriteria::Y)) {
                switch (location) {
                case ToolTipPos::AboveLine:
                    location = ToolTipPos::BelowLine;
                    break;
                case ToolTipPos::BelowLine:
                    location = ToolTipPos::AboveLine;
                    break;
                case ToolTipPos::RightLine:
                    location = ToolTipPos::AboveLine;
                    break;
                default:
                    break;
                }
            } else {
                break;
            }
        }

        m_toolTips[tipBox.topLeftPx] = spMarker;
    }

    void ZepWindow::DisplayMarkerHints() {
        if ((m_visibleLineIndices.x >= m_windowLines.size()) || (m_visibleLineIndices.y >= m_windowLines.size())) {
            return;
        }

        const auto lastLineInfo = m_windowLines[m_visibleLineIndices.y];
        const auto firstLineInfo = m_windowLines[m_visibleLineIndices.x];
        const auto firstIndex = firstLineInfo->lineByteRange.first;
        const auto lastIndex = lastLineInfo->lineByteRange.second;
        const auto lineHeight = m_windowLines[m_visibleLineIndices.x]->FullLineHeightPx();
        const float indicatorWidth = 10.0f;
        auto& theme = m_pBuffer->GetTheme();
        auto& display = GetEditor().GetDisplay();
        auto& font = display.GetFont(ZepTextType::Text);

        m_pBuffer->ForEachMarker(RangeMarkerType::Mark, Direction::Forward, m_pBuffer->Begin(), m_pBuffer->End(), [&](const std::shared_ptr<RangeMarker>& marker) {
            if (!(marker->displayType & RangeMarkerDisplayType::Background)) {
                return true;
            }

            auto indicatorWidth = font.GetTextSize((const uint8_t*)"More...").x;
            if (marker->GetRange().second > lastIndex) {
                auto rc = NRectf(m_textRegion->rect.Right() - indicatorWidth, m_textRegion->rect.Bottom() - lineHeight, indicatorWidth, lineHeight);
                display.DrawRectFilled(rc, ModifyBackgroundColor(marker->GetBackgroundColor()));
                display.DrawChars(font, rc.topLeftPx, theme.GetColor(marker->GetTextColor()), (const uint8_t*)"More...");
            } else if (marker->GetRange().first < firstIndex) {
                auto rc = NRectf(m_textRegion->rect.Right() - indicatorWidth, m_textRegion->rect.Top(), indicatorWidth, lineHeight);
                display.DrawRectFilled(rc, ModifyBackgroundColor(marker->GetBackgroundColor()));
                display.DrawChars(font, rc.topLeftPx, theme.GetColor(marker->GetTextColor()), (const uint8_t*)"More...");
            }

            return true;
        });
    }

    void ZepWindow::DisplayGridMarkers() {
        auto& display = GetEditor().GetDisplay();

        auto rc = m_textRegion->rect;
        rc.Adjust(-1, -1, 1, 1);

        // Border around the edge
        display.DrawRect(rc, ModifyBackgroundColor(ThemeColor::TabActive));

        /*
        for (long windowLine = m_visibleLineIndices.x; windowLine < m_visibleLineIndices.y; windowLine++)
        {
            auto& lineInfo = *m_windowLines[windowLine];
            auto pos = m_textRegion->rect.topLeftPx + NVec2f(m_xPad, 0.0f);
            for (int i = 0; i < lineInfo.lineCodePoints.size(); i++)
            {
                auto cp = lineInfo.lineCodePoints[i];

                if (i != 0 && i % 8 == 0)
                {
                    display.DrawLine(NVec2f(pos.x - m_xPad / 2, pos.y), NVec2f(pos.x - m_xPad / 2, pos.y + DPI_Y(3.0f)), NVec4f(1.0f));
                }
                pos.x += cp.size.x + m_xPad;
            }
        }
        */
    }

    void ZepWindow::Display() {
        TIME_SCOPE(Display);

        auto pMode = GetBuffer().GetMode();
        pMode->PreDisplay(*this);

        // Ensure line spans are valid; updated if the text is changed or the window dimensions change
        UpdateLayout();
        ScrollToCursor();
        UpdateScrollers();
        UpdateMarkers();

        // Second pass if the scroller visibility changed, since this can change the whole layout!
        if (m_scrollVisibilityChanged) {
            m_layoutDirty = true;
            m_cursorMoved = true;
            UpdateLayout();
            ScrollToCursor();
            UpdateScrollers();

            m_scrollVisibilityChanged = false;
        }

        auto& display = GetEditor().GetDisplay();
        auto cursorCL = BufferToDisplay(m_bufferCursor);
        m_mouseBufferLocation = GlyphIterator();

        // Always update
        UpdateAirline();

        UpdateLayout();

        if (GetEditor().GetConfig().style == EditorStyle::Normal) {
            // Fill the background color for the whole area, only in normal mode.
            display.DrawRectFilled(m_textRegion->rect, ModifyBackgroundColor(ThemeColor::Background));
        }

        if (m_numberRegion->rect.Width() > 0) {
            display.DrawRectFilled(m_numberRegion->rect, ModifyBackgroundColor(ThemeColor::LineNumberBackground));
        }

        if (m_indicatorRegion->rect.Width() > 0) {
            display.DrawRectFilled(m_indicatorRegion->rect, ModifyBackgroundColor(ThemeColor::LineNumberBackground));
        }

        DisplayScrollers();

        // This is a line down the middle of a split
        if (GetEditor().GetConfig().style == EditorStyle::Normal && !ZTestFlags(GetWindowFlags(), WindowFlags::HideSplitMark)) {
            display.DrawRectFilled(
                NRectf(NVec2f(m_expandingEditRegion->rect.topLeftPx.x, m_expandingEditRegion->rect.topLeftPx.y), NVec2f(m_expandingEditRegion->rect.topLeftPx.x + 1, m_expandingEditRegion->rect.bottomRightPx.y)), ModifyBackgroundColor(ThemeColor::TabInactive));
        }

        DisplayLineNumbers();

        {
            // Reset the last line pixel size
            lastLinePx = NVec2f(0.0f);

            TIME_SCOPE(DrawLine);
            for (int displayPass = 0; displayPass < WindowPass::Max; displayPass++) {
                for (long windowLine = m_visibleLineIndices.x; windowLine < m_visibleLineIndices.y; windowLine++) {
                    auto& lineInfo = *m_windowLines[windowLine];
                    if (!DisplayLine(lineInfo, displayPass)) {
                        break;
                    }
                }
            }
        }

        DisplayMarkerHints();

        if (ZTestFlags(GetWindowFlags(), WindowFlags::GridStyle)) {
            DisplayGridMarkers();
        }

        // Is the cursor on a tooltip row or mark?
        if (m_toolTips.empty()) {
            auto cursorLine = GetCursorLineInfo(BufferToDisplay().y);

            // If this marker has an associated tooltip, pop it up after a time delay
            NVec2f pos, size;
            GetCursorInfo(pos, size);

            // Calculate our desired location for the tip.
            auto tipPos = [&](RangeMarker& marker) {
                NVec2f ret;
                NVec2f linePx = GetSpanPixelRange(cursorLine);
                if (marker.tipPos == ToolTipPos::RightLine) {
                    ret = NVec2f(linePx.y, pos.y);
                } else {
                    ret = NVec2f(linePx.x, pos.y);
                }
                return ret;
            };

            m_pBuffer->ForEachMarker(RangeMarkerType::All, Direction::Forward, GlyphIterator(m_pBuffer, cursorLine.lineByteRange.first), GlyphIterator(m_pBuffer, cursorLine.lineByteRange.second), [&](const std::shared_ptr<RangeMarker>& marker) {
                if (marker->displayType == RangeMarkerDisplayType::Hidden) {
                    return true;
                }

                auto sel = marker->GetRange();
                if (marker->displayType & RangeMarkerDisplayType::CursorTip) {
                    if (m_bufferCursor.Index() >= sel.first && m_bufferCursor.Index() < sel.second) {
                        PlaceToolTip(tipPos(*marker), marker->tipPos, 2, marker);
                    }
                }

                if (marker->displayType & RangeMarkerDisplayType::CursorTipAtLine) {
                    if ((cursorLine.lineByteRange.first <= sel.first && cursorLine.lineByteRange.second > sel.first) || (cursorLine.lineByteRange.first <= sel.second && cursorLine.lineByteRange.second > sel.second)) {
                        PlaceToolTip(tipPos(*marker), marker->tipPos, 2, marker);
                    }
                }
                return true;
            });
        } else {
            // No hanging tooltips if the markers on the page have gone
            if (m_pBuffer->GetRangeMarkers(RangeMarkerType::Mark).empty()) {
                m_toolTips.clear();
            }
        }

        // No tooltip, and we can show one, then ask for tooltips from any client that wants to show them
        if (!m_tipDisabledTillMove && (timer_get_elapsed_seconds(m_toolTipTimer) > 0.5f) && m_toolTips.empty() && m_lastTipQueryPos != m_mouseHoverPos) {
            auto spMsg = std::make_shared<ToolTipMessage>(m_pBuffer, m_mouseHoverPos, m_mouseBufferLocation);
            GetEditor().Broadcast(spMsg);
            if (spMsg->handled && spMsg->spMarker != nullptr) {
                PlaceToolTip(NVec2f(m_mouseHoverPos.x, m_mouseHoverPos.y), spMsg->spMarker->tipPos, 1, spMsg->spMarker);
            }
            m_lastTipQueryPos = m_mouseHoverPos;
        }

        for (auto& toolTip : m_toolTips) {
            DisplayToolTip(toolTip.first, *toolTip.second);
        }

        display.SetClipRect(NRectf{});

        if (!GetEditor().GetCommandText().empty() || (GetEditor().GetConfig().autoHideCommandRegion == false)) {
            auto modeAirlines = GetBuffer().GetMode()->GetAirlines(*this);

            // Airline and underline
            display.DrawRectFilled(m_airlineRegion->rect, ModifyBackgroundColor(ThemeColor::AirlineBackground));

            auto airHeight = GetEditor().GetDisplay().GetFont(ZepTextType::UI).GetPixelHeight();
            auto border = 12.0f;

            auto& uiFont = display.GetFont(ZepTextType::UI);

            NVec2f screenPosYPx = m_airlineRegion->rect.topLeftPx;

            auto drawAirline = [&](Airline& airline) {
                display.SetClipRect(NRectf{});
                for (int i = 0; i < (int)airline.leftBoxes.size(); i++) {
                    auto pText = (const uint8_t*)airline.leftBoxes[i].text.c_str();
                    auto textSize = uiFont.GetTextSize(pText, pText + airline.leftBoxes[i].text.size());
                    textSize.x += border * 2;

                    auto col = GetEditor().ModifyBackgroundColor(airline.leftBoxes[i].background);
                    display.DrawRectFilled(NRectf(screenPosYPx, NVec2f(textSize.x + screenPosYPx.x, screenPosYPx.y + airHeight)), col);

                    NVec4f textCol = m_pBuffer->GetTheme().GetComplement(airline.leftBoxes[i].background, IsActiveWindow() ? NVec4f(0.0f) : NVec4f(.5f, .5f, .5f, 0.0f));
                    display.DrawChars(uiFont, screenPosYPx + NVec2f(border, 0.0f), textCol, (const uint8_t*)(airline.leftBoxes[i].text.c_str()));
                    screenPosYPx.x += textSize.x;
                }

                // Clip to the remaining space
                auto clipRect = NRectf(screenPosYPx.x, screenPosYPx.y, m_airlineRegion->rect.Right() - screenPosYPx.x, float(airHeight));
                if (clipRect.Width() > 0 && clipRect.Height() > 0) {
                    display.SetClipRect(clipRect);

                    float totalRightSize = 0.0f;
                    for (int i = 0; i < (int)airline.rightBoxes.size(); i++) {
                        auto pText = (const uint8_t*)airline.rightBoxes[i].text.c_str();
                        totalRightSize += uiFont.GetTextSize(pText, pText + airline.rightBoxes[i].text.size()).x + border * 2;
                    }

                    screenPosYPx.x = m_airlineRegion->rect.Right() - totalRightSize;
                    for (int i = 0; i < (int)airline.rightBoxes.size(); i++) {
                        auto pText = (const uint8_t*)airline.rightBoxes[i].text.c_str();
                        auto textSize = uiFont.GetTextSize(pText, pText + airline.rightBoxes[i].text.size());
                        textSize.x += border * 2;

                        auto col = GetEditor().ModifyBackgroundColor(airline.rightBoxes[i].background);
                        display.DrawRectFilled(NRectf(screenPosYPx, NVec2f(textSize.x + screenPosYPx.x, screenPosYPx.y + float(airHeight))), col);

                        NVec4f textCol = m_pBuffer->GetTheme().GetComplement(airline.rightBoxes[i].background, IsActiveWindow() ? NVec4f(0.0f) : NVec4f(.5f, .5f, .5f, 0.0f));
                        display.DrawChars(uiFont, screenPosYPx + NVec2f(border, 0.0f), textCol, (const uint8_t*)(airline.rightBoxes[i].text.c_str()));
                        screenPosYPx.x += textSize.x;
                    }
                }
            };

            for (auto& line : modeAirlines) {
                drawAirline(line);
                screenPosYPx.y += airHeight;
                screenPosYPx.x = m_airlineRegion->rect.Left();
            }
            drawAirline(m_airline);
        }

        display.SetClipRect(NRectf{});
    }

    void ZepWindow::MoveCursorY(int yDistance, LineLocation clampLocation) {
        UpdateLayout();

        timer_restart(m_toolTipTimer);
        m_toolTips.clear();

        // Get the cursor
        auto cursorCL = BufferToDisplay();
        if (cursorCL.x == -1)
            return;

        // Find the screen line relative target
        auto target = cursorCL + NVec2i(0, yDistance);
        target.y = std::max(0l, target.y);
        target.y = std::min(target.y, long(m_windowLines.size() - 1));

        auto& line = *m_windowLines[target.y];

        // Snap to the new vertical column if necessary (see comment below)
        if (target.x < m_lastCursorColumn)
            target.x = m_lastCursorColumn;

        // TODO; this was an assert
        if (line.lineCodePoints.empty()) {
            return;
        }

        // Move to the same codepoint offset on the line below
        target.x = std::min(target.x, long(line.lineCodePoints.size() - 1));
        target.x = std::max(target.x, long(0));

        GlyphIterator cursorItr = line.lineCodePoints[target.x].iterator;

        // We can't call the buffer's LineLocation code, because when moving in span lines,
        // we are technically not moving in buffer lines; we are stepping in wrapped buffer lines.
        switch (clampLocation) {
        default:
        case LineLocation::LineBegin:
        case LineLocation::LineFirstGraphChar:
        case LineLocation::BeyondLineEnd:
            assert(!"Not supported Y motion line clamp!");
            break;
        case LineLocation::LineLastNonCR: {
            // Don't skip back if we are right at the start of the line
            // (i.e. an empty line)
            if (target.x != 0 && (cursorItr.Char() == '\n' || cursorItr.Char() == 0)) {
                cursorItr.MoveClamped(-1, LineLocation::LineLastNonCR);
            }
        } break;
        case LineLocation::LineCRBegin:
            // We already clamped to here above by testing for max codepoint
            // Last codepoint is the carriage return
            break;
        }

        m_bufferCursor = cursorItr;
        m_cursorMoved = true;

        GetEditor().ResetCursorTimer();

        m_pBuffer->SetLastEditLocation(m_bufferCursor);
    }

    NVec2i ZepWindow::BufferToDisplay() {
        return BufferToDisplay(m_bufferCursor);
    }

    NVec2i ZepWindow::BufferToDisplay(const GlyphIterator& loc) {
        UpdateLayout();

        NVec2i ret(0, 0);
        int line_number = 0;

        // TODO: Performance; quick lookup for line
        for (auto& line : m_windowLines) {
            // If inside the line...
            if (line->lineByteRange.first <= loc.Index() && line->lineByteRange.second > loc.Index()) {
                ret.y = line_number;
                ret.x = 0;

                // Scan the code points for where we are
                for (auto& ch : line->lineCodePoints) {
                    if (ch.iterator == loc) {
                        return ret;
                    }
                    ret.x++;
                }
            }
            line_number++;
        }

        assert(!m_windowLines.empty());
        if (m_windowLines.empty()) {
            return NVec2i(0, 0);
        }

        // Max Last line, last code point offset
        ret.y = long(m_windowLines.size() - 1);
        ret.x = long(m_windowLines[m_windowLines.size() - 1]->lineCodePoints.size() - 1);
        return ret;
    }

} // namespace Zep

#if 0
    // Ensure we can see the cursor
    NVec2i cursor(0, 0);
    cursor.x = m_pBuffer->GetBufferColumn(m_bufferCursor);
    cursor.y = m_pBuffer->GetBufferLine(m_bufferCursor) - m_pBuffer->GetBufferLine(m_dvisibleLineRange.x);

    // Handle the case where there is no need to scroll, since the visible lines are inside
    // The current screen rectangle.
    if (cursor.y >= m_nvisibleLineRange.y && m_linesFillScreen)
    {
        m_visibleLineRange.x = cursor.y - (m_nvisibleLineRange.y - m_vnisibleLineRange.x) + 1;
        m_linesChanged = true;
    }
    else if (cursor.y < m_nvisibleLineRange.x)
    {
        m_nvisibleLineRange.x = cursor.y;
        m_linesChanged = true;
    }

    // Clamp
    m_nvisibleLineRange.x = std::max(0l, (long)m_nvisibleLineRange.x);
#endif
