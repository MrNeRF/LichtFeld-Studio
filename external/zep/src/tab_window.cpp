#include "zep/tab_window.h"
#include "zep/buffer.h"
#include "zep/display.h"
#include "zep/editor.h"
#include "zep/mode.h"
#include "zep/window.h"

#include "zep/mcommon/logger.h"

// A 'window' is like a Vim Tab
namespace Zep {

    ZepTabWindow::ZepTabWindow(ZepEditor& editor)
        : ZepComponent(editor), m_editor(editor) {
        m_spRootRegion = std::make_shared<Region>();
        m_spRootRegion->flags = RegionFlags::Expanding;
    }

    ZepTabWindow::~ZepTabWindow() {
        std::for_each(m_windows.begin(), m_windows.end(), [](ZepWindow* w) { delete w; });
        m_spRootRegion.reset();
        m_windowRegions.clear();
    }

    ZepWindow* ZepTabWindow::DoMotion(WindowMotion motion) {
        if (!m_pActiveWindow)
            return nullptr;

        auto pCurrentRegion = m_windowRegions[m_pActiveWindow];
        auto center = pCurrentRegion->rect.Center();
        auto minDistance = std::numeric_limits<float>::max();
        Region* pBestRegion = nullptr;

        auto overlap_x = [&](const NRectf& lhs, const NRectf& rhs) {
            if ((lhs.topLeftPx.x < rhs.bottomRightPx.x) && (lhs.bottomRightPx.x > rhs.topLeftPx.x)) {
                return true;
            }
            return false;
        };
        auto overlap_y = [&](const NRectf& lhs, const NRectf& rhs) {
            if ((lhs.topLeftPx.y < rhs.bottomRightPx.y) && (lhs.bottomRightPx.y > rhs.topLeftPx.y)) {
                return true;
            }
            return false;
        };

        for (auto& r : m_windowRegions) {
            auto& pRegion = r.second;
            if (pCurrentRegion == pRegion)
                continue;

            auto thisCenter = pRegion->rect.Center();
            float dist = 0.0f;
            switch (motion) {
            case WindowMotion::Right: {
                if (overlap_x(pRegion->rect, pCurrentRegion->rect))
                    continue;
                if (!overlap_y(pRegion->rect, pCurrentRegion->rect))
                    continue;
                dist = thisCenter.x - center.x;
            } break;
            case WindowMotion::Left: {
                if (overlap_x(pRegion->rect, pCurrentRegion->rect))
                    continue;
                if (!overlap_y(pRegion->rect, pCurrentRegion->rect))
                    continue;
                dist = center.x - thisCenter.x;
            } break;
            case WindowMotion::Up: {
                if (overlap_y(pRegion->rect, pCurrentRegion->rect))
                    continue;
                if (!overlap_x(pRegion->rect, pCurrentRegion->rect))
                    continue;
                dist = center.y - thisCenter.y;
            } break;
            case WindowMotion::Down: {
                if (overlap_y(pRegion->rect, pCurrentRegion->rect))
                    continue;
                if (!overlap_x(pRegion->rect, pCurrentRegion->rect))
                    continue;
                dist = thisCenter.y - center.y;
            } break;
            }
            if (dist > 0.0f && minDistance > dist) {
                minDistance = dist;
                pBestRegion = pRegion.get();
            }
        }

        if (pBestRegion != nullptr) {
            auto itrFound = std::find_if(m_windowRegions.begin(), m_windowRegions.end(), [pBestRegion](auto& val) { return (val.second.get() == pBestRegion); });
            if (itrFound != m_windowRegions.end()) {
                SetActiveWindow(itrFound->first);
            }
        }
        return m_pActiveWindow;
    }

    void ZepTabWindow::SetActiveWindow(ZepWindow* pBuffer) {
        m_pActiveWindow = pBuffer;
        if (m_pActiveWindow) {
            m_pActiveWindow->GetBuffer().GetMode()->Begin(m_pActiveWindow);
        }
        GetEditor().UpdateTabs();
    }

    // The region management is in theory dirt-simple.
    // Each region contains a list of either vertical or horizontally split regions.
    // Any region can have child regions of the same type.
    // If you split a window, you effectively create a stacked region within the parent region.
    // Sometimes this code will create extra regions to hold children regions.  The RemoveWindow code has a special
    // case to deal with that.
    ZepWindow* ZepTabWindow::AddWindow(ZepBuffer* pBuffer, ZepWindow* pParent, RegionLayoutType layoutType) {
        // If we are replacing a default/unmodified start buffer, then this new window will replace it
        // This makes for nice behavior where adding a top-level window to the tab will nuke the default buffer
        if (m_windows.size() == 1 && pParent == nullptr) {
            auto& buffer = m_windows[0]->GetBuffer();
            if (buffer.HasFileFlags(FileFlags::DefaultBuffer) && !buffer.HasFileFlags(FileFlags::Dirty)) {
                m_windows[0]->SetBuffer(pBuffer);
                return m_windows[0];
            }
        }

        // Make a new window
        auto pWin = new ZepWindow(*this, pBuffer);
        m_windows.push_back(pWin);

        // This new window is going to introduce a new region
        auto r = std::make_shared<Region>();
        r->flags = RegionFlags::Expanding;
        r->debugName = pBuffer->GetName();

        SetActiveWindow(pWin);

        // No parent, inserting our window on top of whatever is already in the window
        if (pParent == nullptr) {
            // There is something in the child, so make a good
            if (m_spRootRegion->children.size() > 1) {
                // Make a new parent to hold the children at the root, and replace the root region
                // with the new parent, putting the old root inside it!
                auto r1 = std::make_shared<Region>();
                r1->flags = RegionFlags::Expanding;
                r1->layoutType = layoutType;
                r1->children.push_back(m_spRootRegion);
                r1->debugName = "Parented Root";
                m_spRootRegion->pParent = r1.get();
                m_spRootRegion = r1;
                r1->children.push_back(r);
            } else {
                // The root region has 0 or 1 children, so just set its split type and add our child
                m_spRootRegion->children.push_back(r);
                m_spRootRegion->layoutType = layoutType;
            }

            // New region has root as the parent.
            r->pParent = m_spRootRegion.get();
            m_windowRegions[pWin] = r;
        } else {
            // Get the parent region that holds the parent window!
            auto pParentRegion = m_windowRegions[pParent]->pParent;

            // Fix to ensure that a parent region isn't unnecessarily split if it doesn't have a split layout,
            // and has only one child
            if (pParentRegion->children.size() == 1) {
                pParentRegion->layoutType = layoutType;
            }

            if (pParentRegion->layoutType == layoutType) {
                // Add our newly created region into the parent region, since it is splitting in the same way as the parent's children
                // Try to find the right spot, so we effectively split the correct parent
                auto itrFound = std::find_if(pParentRegion->children.begin(), pParentRegion->children.end(), [&](auto pCurrent) {
                    if (pCurrent == m_windowRegions[pParent]) {
                        return true;
                    }
                    return false;
                });

                // Insertion point should be _after_ the location we want
                if (itrFound != pParentRegion->children.end()) {
                    itrFound++;
                }
                pParentRegion->children.insert(itrFound, r);

                r->pParent = pParentRegion;
                m_windowRegions[pWin] = r;
            } else {
                // We are adding ourselves to a child that isn't the same orientation as us
                auto pSplitRegion = m_windowRegions[pParent];
                assert(pSplitRegion->children.empty());

                // Force the region to be of the new split type
                pSplitRegion->layoutType = layoutType;

                // Make a new region and put the existing region in it
                auto r1 = std::make_shared<Region>();
                r1->debugName = "New Sub Region";
                r1->flags = RegionFlags::Expanding;
                pSplitRegion->children.push_back(r1);
                r1->pParent = pSplitRegion.get();
                m_windowRegions[pParent] = r1;

                // Put our new window on the end
                pSplitRegion->children.push_back(r);
                r->pParent = pSplitRegion.get();
                m_windowRegions[pWin] = r;
            }
        }

        SetActiveWindow(pWin);

        SetDisplayRegion(m_lastRegionRect, true);

        return pWin;
    }

    void ZepTabWindow::CloseActiveWindow() {
        if (m_pActiveWindow) {
            if (m_pActiveWindow->GetBuffer().GetMode()) {
                m_pActiveWindow->GetBuffer().GetMode()->Begin(nullptr);
            }

            // Note: cannot do anything after this call if this is the last window to close!
            RemoveWindow(m_pActiveWindow);
        }
    }

    /*
    void ZepTabWindow::WalkRegions()
    {
        // Not currently used
        for (auto& winRegion : m_windowRegions)
        {
            assert(std::find(m_windows.begin(), m_windows.end(), winRegion.first) != m_windows.end());

            using fnCb = std::function<void(std::shared_ptr<Region>)>;
            using fnWalk = std::function<void(std::shared_ptr<Region>, fnCb)>;
            fnWalk walk = [&](std::shared_ptr<Region> spRegion, fnCb cb) {
                cb(spRegion);
                for (auto& child : spRegion->children)
                {
                    walk(child, cb);
                }
            };

            walk(m_spRootRegion, [&](std::shared_ptr<Region> spRegion) {
                assert(!spRegion->children.empty());
            });
        }
    }
    */

    // See AddWindow for comments..
    void ZepTabWindow::RemoveWindow(ZepWindow* pWindow) {
        assert(pWindow);
        if (!pWindow) {
            assert(!"No window?");
            return;
        }

        // Find the window
        auto itrFound = std::find(m_windows.begin(), m_windows.end(), pWindow);
        if (itrFound == m_windows.end()) {
            assert(!"Not found?");
            return;
        }

        // Find its region
        auto pRegion = m_windowRegions[pWindow];
        assert(pRegion->pParent);

        // Find the parent that owns its region
        auto pParentRegion = pRegion->pParent;

        // Now erase the region from the parent
        auto itrFoundRegion = std::find_if(pParentRegion->children.begin(), pParentRegion->children.end(), [pRegion](std::shared_ptr<Region> pCurrent) {
            return pCurrent == pRegion;
        });
        assert(itrFoundRegion != pParentRegion->children.end());
        pParentRegion->children.erase(itrFoundRegion);

        // Sometimes, adding windows creates an intermediate region to hold child regions.  This happens
        // when adding a window to a split that is in the other direction.
        // When later removing the window, there is the problem that an owner split might now be empty because it
        // isn't associated with any remaining windows.  This function walks up the parents and cleans out empty ones
        std::function<void(Region*)> fnRemoveEmptyParent = [&](Region* pParent) {
            // If the parent of the region we are deleting has no more children...
            if (pParent->children.empty()) {
                // ... and it is not associated with a window (i.e. it was created as a container)
                auto itrFoundWin = std::find_if(m_windowRegions.begin(), m_windowRegions.end(), [pParent](auto currentPair) {
                    return currentPair.second.get() == pParent;
                });

                // ... then we remove it.
                if (itrFoundWin == m_windowRegions.end()) {
                    auto pDeleteRegion = pParent;
                    auto pOwnerRegion = pParent->pParent;
                    if (pOwnerRegion) {
                        auto itrFoundChild = std::find_if(pOwnerRegion->children.begin(), pOwnerRegion->children.end(),
                                                          [pDeleteRegion](std::shared_ptr<Region> pChild) { return pChild.get() == pDeleteRegion; });
                        if (itrFoundChild != pOwnerRegion->children.end()) {
                            pOwnerRegion->children.erase(itrFoundChild);
                        }

                        // Walk up
                        fnRemoveEmptyParent(pOwnerRegion);
                    }
                }
            }
        };

        fnRemoveEmptyParent(pParentRegion);

        // Now delete the window, erase it from the region info
        delete pWindow;
        m_windows.erase(itrFound);
        m_windowRegions.erase(pWindow);

        /*if (pParentRegion)
            CleanEmptyRegions();
            */

        if (m_windows.empty()) {
            SetActiveWindow(nullptr);
            m_spRootRegion.reset();
            GetEditor().RemoveTabWindow(this);
        } else {
            if (m_pActiveWindow == pWindow) {
                // TODO: Active window ordering - remember the last active and switch to it when this one is closed
                assert(!m_windows.empty());
                SetActiveWindow(m_windows[m_windows.size() - 1]);
            }
            SetDisplayRegion(m_lastRegionRect, true);
            assert(!m_spRootRegion->children.empty());
        }
    }

    void ZepTabWindow::Notify(std::shared_ptr<ZepMessage> pMsg) {
        if (pMsg->messageId == Msg::MouseDown) {
            for (auto& region : m_windowRegions) {
                if (region.second->rect.Contains(pMsg->pos)) {
                    if (m_pActiveWindow != region.first) {
                        SetActiveWindow(region.first);
                        pMsg->handled = true;
                        return;
                    }
                }
            }
        }
    }

    void ZepTabWindow::SetDisplayRegion(const NRectf& region, bool force) {
        if (m_lastRegionRect != region || force) {
            m_lastRegionRect = region;

            if (m_spRootRegion) {
                m_spRootRegion->rect = region;
                LayoutRegion(*m_spRootRegion);
            }

            for (auto& w : m_windows) {
                // TODO: Crash here rect is NULL?
                w->SetDisplayRegion(m_windowRegions[w]->rect);
            }
        }
    }

    void ZepTabWindow::Display() {
        for (auto& w : m_windows) {
            w->Display();
        }
    }

    std::string ZepTabWindow::GetName() const {
        std::string name;

        // Show active buffer in tab as tab name
        if (GetActiveWindow()) {
            auto& buffer = GetActiveWindow()->GetBuffer();
            name = buffer.GetName();
            if (GetEditor().GetConfig().shortTabNames) {
                auto pos = name.find_last_of('.');
                if (pos != std::string::npos) {
                    name = name.substr(0, pos);
                }
            }
        }
        return name;
    }

} // namespace Zep
