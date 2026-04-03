#include "zep/scroller.h"
#include "zep/display.h"
#include "zep/editor.h"
#include "zep/theme.h"

#include "zep/mcommon/logger.h"
#include <algorithm>

// A scrollbar that is manually drawn and implemented.  This means it is independent of the backend and can be drawn the
// same in Qt and ImGui
namespace Zep {

    Scroller::Scroller(ZepEditor& editor, Region& parent, bool vertical)
        : ZepComponent(editor), m_vertical(vertical) {
        m_region = std::make_shared<Region>();
        m_topButtonRegion = std::make_shared<Region>();
        m_bottomButtonRegion = std::make_shared<Region>();
        m_mainRegion = std::make_shared<Region>();

        m_region->flags = RegionFlags::Expanding;
        m_topButtonRegion->flags = RegionFlags::Fixed;
        m_bottomButtonRegion->flags = RegionFlags::Fixed;
        m_mainRegion->flags = RegionFlags::Expanding;

        m_region->layoutType = m_vertical ? RegionLayoutType::VBox : RegionLayoutType::HBox;

        m_topButtonRegion->padding = NVec2f(0.0f, 0.0f);
        m_bottomButtonRegion->padding = NVec2f(0.0f, 0.0f);
        m_mainRegion->padding = NVec2f(0.0f, 0.0f);
        m_topButtonRegion->fixed_size = NVec2f(0.0f, 0.0f);
        m_bottomButtonRegion->fixed_size = NVec2f(0.0f, 0.0f);

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
        scrollPosition -= scrollLinePercent;
        scrollPosition = std::max(0.0f, scrollPosition);
        GetEditor().Broadcast(std::make_shared<ZepMessage>(Msg::ComponentChanged, this));
        m_scrollState = ScrollState::ScrollUp;
    }

    void Scroller::ClickDown() {
        scrollPosition += scrollLinePercent;
        scrollPosition = std::min(1.0f - scrollVisiblePercent, scrollPosition);
        GetEditor().Broadcast(std::make_shared<ZepMessage>(Msg::ComponentChanged, this));
        m_scrollState = ScrollState::ScrollDown;
    }

    void Scroller::PageUp() {
        scrollPosition -= scrollPagePercent;
        scrollPosition = std::max(0.0f, scrollPosition);
        GetEditor().Broadcast(std::make_shared<ZepMessage>(Msg::ComponentChanged, this));
        m_scrollState = ScrollState::PageUp;
    }

    void Scroller::PageDown() {
        scrollPosition += scrollPagePercent;
        scrollPosition = std::min(1.0f - scrollVisiblePercent, scrollPosition);
        GetEditor().Broadcast(std::make_shared<ZepMessage>(Msg::ComponentChanged, this));
        m_scrollState = ScrollState::PageDown;
    }

    void Scroller::DoMove(NVec2f pos) {
        if (m_scrollState == ScrollState::Drag) {
            float dist = PrimaryCoord(pos) - PrimaryCoord(m_mouseDownPos);

            float totalMove = MainSize() - ThumbSize();
            if (totalMove <= 0.0f) {
                scrollPosition = 0.0f;
            } else {
                float percentPerPixel = (1.0f - scrollVisiblePercent) / totalMove;
                scrollPosition = m_mouseDownPercent + (percentPerPixel * dist);
            }
            scrollPosition = std::min(1.0f - scrollVisiblePercent, scrollPosition);
            scrollPosition = std::max(0.0f, scrollPosition);
            GetEditor().Broadcast(std::make_shared<ZepMessage>(Msg::ComponentChanged, this));
        }
    }

    float Scroller::MainSize() const {
        return m_vertical ? m_mainRegion->rect.Height() : m_mainRegion->rect.Width();
    }

    float Scroller::PrimaryCoord(const NVec2f& pos) const {
        return m_vertical ? pos.y : pos.x;
    }

    float Scroller::ThumbSize() const {
        const auto pixelScale = GetEditor().GetDisplay().GetPixelScale();
        const float minSize = std::max(
            1.0f,
            GetEditor().GetConfig().scrollBarMinSize * (m_vertical ? pixelScale.y : pixelScale.x));
        return std::max(minSize, MainSize() * scrollVisiblePercent);
    }

    NRectf Scroller::ThumbRect() const {
        auto thumbSize = ThumbSize();
        if (m_vertical) {
            return NRectf(
                NVec2f(m_mainRegion->rect.topLeftPx.x,
                       m_mainRegion->rect.topLeftPx.y + MainSize() * scrollPosition),
                NVec2f(m_mainRegion->rect.bottomRightPx.x,
                       m_mainRegion->rect.topLeftPx.y + MainSize() * scrollPosition + thumbSize));
        }

        return NRectf(
            NVec2f(m_mainRegion->rect.topLeftPx.x + MainSize() * scrollPosition,
                   m_mainRegion->rect.topLeftPx.y),
            NVec2f(m_mainRegion->rect.topLeftPx.x + MainSize() * scrollPosition + thumbSize,
                   m_mainRegion->rect.bottomRightPx.y));
    }

    void Scroller::Notify(std::shared_ptr<ZepMessage> message) {
        switch (message->messageId) {
        case Msg::Tick: {
            CheckState();
        } break;

        case Msg::MouseDown:
            if (message->button == ZepMouseButton::Left) {
                if (m_mainRegion->rect.Contains(message->pos)) {
                    auto thumbRect = ThumbRect();
                    if (thumbRect.Contains(message->pos)) {
                        m_mouseDownPos = message->pos;
                        m_mouseDownPercent = scrollPosition;
                        m_scrollState = ScrollState::Drag;
                        message->handled = true;
                    } else if (PrimaryCoord(message->pos) >
                               (m_vertical ? thumbRect.BottomLeft().y : thumbRect.TopRight().x)) {
                        PageDown();
                        timer_start(m_start_delay_timer);
                        message->handled = true;
                    } else if (PrimaryCoord(message->pos) <
                               (m_vertical ? thumbRect.TopRight().y : thumbRect.topLeftPx.x)) {
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
        const float thickness =
            m_vertical ? m_region->rect.Width() : m_region->rect.Height();
        const float rounding = std::max(0.0f, thickness * 0.5f);

        // Scroller background
        display.DrawRectFilled(m_region->rect, theme.GetColor(ThemeColor::WidgetBackground), rounding);

        auto thumbRect = ThumbRect();

        // Thumb
        display.DrawRectFilled(
            thumbRect,
            thumbRect.Contains(mousePos) || m_scrollState == ScrollState::Drag ? activeColor : inactiveColor,
            rounding);
    }

}; // namespace Zep
