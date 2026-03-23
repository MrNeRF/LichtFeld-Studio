#pragma once

#include "buffer.h"
#include <deque>

namespace Zep {

    class ZepWindow;
    class ZepDisplay;
    struct Region;

    enum class WindowMotion {
        Left,
        Right,
        Up,
        Down
    };

    // Display state for a single pane of text.
    // Editor operations such as select and change are local to a displayed pane
    class ZepTabWindow : public ZepComponent {
    public:
        ZepTabWindow(ZepEditor& editor);
        virtual ~ZepTabWindow();

        virtual void Notify(std::shared_ptr<ZepMessage> message) override;

        ZepWindow* DoMotion(WindowMotion motion);
        ZepWindow* AddWindow(ZepBuffer* pBuffer, ZepWindow* pParent = nullptr, RegionLayoutType layoutType = RegionLayoutType::HBox);
        void RemoveWindow(ZepWindow* pWindow);
        // void WalkRegions();
        void SetActiveWindow(ZepWindow* pBuffer);
        ZepWindow* GetActiveWindow() const {
            return m_pActiveWindow;
        }
        void CloseActiveWindow();
        std::string GetName() const;

        using tWindows = std::vector<ZepWindow*>;
        using tWindowRegions = std::map<ZepWindow*, std::shared_ptr<Region>>;
        const tWindows& GetWindows() const {
            return m_windows;
        }

        void SetDisplayRegion(const NRectf& region, bool force = false);

        void Display();

    private:
        ZepEditor& m_editor; // Editor that owns this window
        NRectf m_lastRegionRect;

        tWindows m_windows;
        tWindowRegions m_windowRegions;
        std::shared_ptr<Region> m_spRootRegion;
        ZepWindow* m_pActiveWindow = nullptr;
    };

} // namespace Zep
