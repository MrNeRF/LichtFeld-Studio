#include "zep/scroller.h"
#include "zep/display.h"
#include "zep/editor.h"
#include "zep/theme.h"

#include "zep/mcommon/logger.h"

// A scrollbar that is manually drawn and implemented.  This means it is independent of the backend and can be drawn the
// same in Qt and ImGui
namespace Zep {

    Scroller::Scroller(ZepEditor& editor, Region& parent)
        : ZepComponent(editor) {
        m_region = std::make_shared<Region>();
        m_topButtonRegion = std::make_shared<Region>();
        m_bottomButtonRegion = std::make_shared<Region>();
        m_mainRegion = std::make_shared<Region>();

        m_region->flags = RegionFlags::Expanding;
        m_topButtonRegion->flags = RegionFlags::Fixed;
        m_bottomButtonRegion->flags = RegionFlags::Fixed;
        m_mainRegion->flags = RegionFlags::Expanding;

        m_region->layoutType = RegionLayoutType::VBox;

        const float scrollButtonMargin = 3.0f * editor.GetDisplay().GetPixelScale().x;
        m_topButtonRegion->padding = NVec2f(scrollButtonMargin, scrollButtonMargin);
        m_bottomButtonRegion->padding = NVec2f(scrollButtonMargin, scrollButtonMargin);
        m_mainRegion->padding = NVec2f(scrollButtonMargin, 0.0f);

        const float scrollButtonSize = DefaultTextSize * editor.GetDisplay().GetPixelScale().x;
        m_topButtonRegion->fixed_size = NVec2f(0.0f, scrollButtonSize);
        m_bottomButtonRegion->fixed_size = NVec2f(0.0f, scrollButtonSize);

        m_region->children.push_back(m_topButtonRegion);
        m_region->children.push_back(m_mainRegion);
        m_region->children.push_back(m_bottomButtonRegion);

        parent.children.push_back(m_region);
    }

    void Scroller::CheckState() {
        if (m_scrollState == ScrollState::None) {
            return;
        }

        if (timer_get_elapsed_seconds(m_start_delay_timer) < 0.5f) {
            return;
        }

        switch (m_scrollState) {
        default:
        case ScrollState::None:
            break;
        case ScrollState::ScrollUp:
            ClickUp();
            break;
        case ScrollState::ScrollDown:
            ClickDown();
            break;
        case ScrollState::PageUp:
            PageUp();
            break;
        case ScrollState::PageDown:
            PageDown();
            break;
        }

        GetEditor().RequestRefresh();
    }

    void Scroller::ClickUp() {
        vScrollPosition -= vScrollLinePercent;
        vScrollPosition = std::max(0.0f, vScrollPosition);
        GetEditor().Broadcast(std::make_shared<ZepMessage>(Msg::ComponentChanged, this));
        m_scrollState = ScrollState::ScrollUp;
    }

    void Scroller::ClickDown() {
        vScrollPosition += vScrollLinePercent;
        vScrollPosition = std::min(1.0f - vScrollVisiblePercent, vScrollPosition);
        GetEditor().Broadcast(std::make_shared<ZepMessage>(Msg::ComponentChanged, this));
        m_scrollState = ScrollState::ScrollDown;
    }

    void Scroller::PageUp() {
        vScrollPosition -= vScrollPagePercent;
        vScrollPosition = std::max(0.0f, vScrollPosition);
        GetEditor().Broadcast(std::make_shared<ZepMessage>(Msg::ComponentChanged, this));
        m_scrollState = ScrollState::PageUp;
    }

    void Scroller::PageDown() {
        vScrollPosition += vScrollPagePercent;
        vScrollPosition = std::min(1.0f - vScrollVisiblePercent, vScrollPosition);
        GetEditor().Broadcast(std::make_shared<ZepMessage>(Msg::ComponentChanged, this));
        m_scrollState = ScrollState::PageDown;
    }

    void Scroller::DoMove(NVec2f pos) {
        if (m_scrollState == ScrollState::Drag) {
            float dist = pos.y - m_mouseDownPos.y;

            float totalMove = m_mainRegion->rect.Height() - ThumbSize();
            float percentPerPixel = (1.0f - vScrollVisiblePercent) / totalMove;
            vScrollPosition = m_mouseDownPercent + (percentPerPixel * dist);
            vScrollPosition = std::min(1.0f - vScrollVisiblePercent, vScrollPosition);
            vScrollPosition = std::max(0.0f, vScrollPosition);
            GetEditor().Broadcast(std::make_shared<ZepMessage>(Msg::ComponentChanged, this));
        }
    }

    float Scroller::ThumbSize() const {
        return std::max(10.0f, m_mainRegion->rect.Height() * vScrollVisiblePercent);
    }

    float Scroller::ThumbExtra() const {
        return std::max(0.0f, ThumbSize() - (m_mainRegion->rect.Height() * vScrollVisiblePercent));
    }

    NRectf Scroller::ThumbRect() const {
        auto thumbSize = ThumbSize();
        return NRectf(NVec2f(m_mainRegion->rect.topLeftPx.x, m_mainRegion->rect.topLeftPx.y + m_mainRegion->rect.Height() * vScrollPosition), NVec2f(m_mainRegion->rect.bottomRightPx.x, m_mainRegion->rect.topLeftPx.y + m_mainRegion->rect.Height() * vScrollPosition + thumbSize));
    }

    void Scroller::Notify(std::shared_ptr<ZepMessage> message) {
        switch (message->messageId) {
        case Msg::Tick: {
            CheckState();
        } break;

        case Msg::MouseDown:
            if (message->button == ZepMouseButton::Left) {
                if (m_bottomButtonRegion->rect.Contains(message->pos)) {
                    ClickDown();
                    timer_start(m_start_delay_timer);
                    message->handled = true;
                } else if (m_topButtonRegion->rect.Contains(message->pos)) {
                    ClickUp();
                    timer_start(m_start_delay_timer);
                    message->handled = true;
                } else if (m_mainRegion->rect.Contains(message->pos)) {
                    auto thumbRect = ThumbRect();
                    if (thumbRect.Contains(message->pos)) {
                        m_mouseDownPos = message->pos;
                        m_mouseDownPercent = vScrollPosition;
                        m_scrollState = ScrollState::Drag;
                        message->handled = true;
                    } else if (message->pos.y > thumbRect.BottomLeft().y) {
                        PageDown();
                        timer_start(m_start_delay_timer);
                        message->handled = true;
                    } else if (message->pos.y < thumbRect.TopRight().y) {
                        PageUp();
                        timer_start(m_start_delay_timer);
                        message->handled = true;
                    }
                }
            }
            break;
        case Msg::MouseUp: {
            m_scrollState = ScrollState::None;
        } break;
        case Msg::MouseMove:
            DoMove(message->pos);
            break;
        default:
            break;
        }
    }

    void Scroller::Display(ZepTheme& theme) {
        auto& display = GetEditor().GetDisplay();

        display.SetClipRect(m_region->rect);

        auto mousePos = GetEditor().GetMousePos();
        auto activeColor = theme.GetColor(ThemeColor::WidgetActive);
        auto inactiveColor = theme.GetColor(ThemeColor::WidgetInactive);

        // Scroller background
        display.DrawRectFilled(m_region->rect, theme.GetColor(ThemeColor::WidgetBackground));

        bool onTop = m_topButtonRegion->rect.Contains(mousePos) && m_scrollState != ScrollState::Drag;
        bool onBottom = m_bottomButtonRegion->rect.Contains(mousePos) && m_scrollState != ScrollState::Drag;

        if (m_scrollState == ScrollState::ScrollUp) {
            onTop = true;
        }
        if (m_scrollState == ScrollState::ScrollDown) {
            onBottom = true;
        }

        display.DrawRectFilled(m_topButtonRegion->rect, onTop ? activeColor : inactiveColor);
        display.DrawRectFilled(m_bottomButtonRegion->rect, onBottom ? activeColor : inactiveColor);

        auto thumbRect = ThumbRect();

        // Thumb
        display.DrawRectFilled(thumbRect, thumbRect.Contains(mousePos) || m_scrollState == ScrollState::Drag ? activeColor : inactiveColor);
    }

}; // namespace Zep
