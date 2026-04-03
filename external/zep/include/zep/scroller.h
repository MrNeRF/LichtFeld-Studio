#pragma once

#include "editor.h"
#include "splits.h"
#include "zep/mcommon/animation/timer.h"

namespace Zep {
    class ZepTheme;
    class ZepEditor;

    class Scroller : public ZepComponent {
    public:
        Scroller(ZepEditor& editor, Region& parent, bool vertical);

        virtual void Display(ZepTheme& theme);
        virtual void Notify(std::shared_ptr<ZepMessage> message) override;

        float scrollVisiblePercent = 1.0f;
        float scrollPosition = 0.0f;
        float scrollLinePercent = 0.0f;
        float scrollPagePercent = 0.0f;

    private:
        void CheckState();
        void ClickUp();
        void ClickDown();
        void PageUp();
        void PageDown();
        void DoMove(NVec2f pos);
        float MainSize() const;
        float PrimaryCoord(const NVec2f& pos) const;

        float ThumbSize() const;
        NRectf ThumbRect() const;

    private:
        bool m_vertical = true;
        std::shared_ptr<Region> m_region;
        std::shared_ptr<Region> m_topButtonRegion;
        std::shared_ptr<Region> m_bottomButtonRegion;
        std::shared_ptr<Region> m_mainRegion;
        timer m_start_delay_timer;
        timer m_reclick_timer;
        enum class ScrollState {
            None,
            ScrollDown,
            ScrollUp,
            PageUp,
            PageDown,
            Drag
        };
        ScrollState m_scrollState = ScrollState::None;
        NVec2f m_mouseDownPos;
        float m_mouseDownPercent;
    };

}; // namespace Zep
