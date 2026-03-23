#include <iterator>
#include <random>

#include "zep/editor.h"
#include "zep/mcommon/animation/timer.h"
#include "zep/regress.h"
#include "zep/tab_window.h"
#include "zep/window.h"

namespace Zep {

    template <typename Iter, typename RandomGenerator>
    Iter select_randomly(Iter start, Iter end, RandomGenerator& g) {
        std::uniform_int_distribution<> dis(0, (int)std::distance(start, end) - 1);
        std::advance(start, dis(g));
        return start;
    }

    template <typename Iter>
    Iter select_randomly(Iter start, Iter end) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        return select_randomly(start, end, gen);
    }

    ZepRegressExCommand::ZepRegressExCommand(ZepEditor& editor)
        : ZepExCommand(editor) {
        timer_start(m_timer);
    }

    void ZepRegressExCommand::Register(ZepEditor& editor) {
        editor.RegisterExCommand(std::make_shared<ZepRegressExCommand>(editor));
    }

    const char* ZepRegressExCommand::ExCommandName() const {
        return "ZRegress";
    }

    void ZepRegressExCommand::Run(const std::vector<std::string>& tokens) {
        ZEP_UNUSED(tokens);
        m_enable = !m_enable;
        if (m_enable) {
            GetEditor().RegisterCallback(this);
            m_windowOperationCount = 150;
        } else {
            GetEditor().UnRegisterCallback(this);
        }
    }

    void ZepRegressExCommand::Notify(std::shared_ptr<ZepMessage> message) {
        if (message->messageId == Msg::Tick) {
            Tick();
        }
    }

    void ZepRegressExCommand::Tick() {
        const int MaxTabWindows = 10;
        if (!m_enable) {
            return;
        }

        auto seconds = timer_get_elapsed_seconds(m_timer);
        if (seconds < .05f) {
            return;
        }

        m_windowOperationCount--;
        if (m_windowOperationCount == 0) {
            m_enable = false;
        }

        timer_restart(m_timer);

        float fRand1 = rand() / (float)RAND_MAX;
        float fRand2 = rand() / (float)RAND_MAX;
        float fRand3 = rand() / (float)RAND_MAX;

        auto& tabWindows = GetEditor().GetTabWindows();
        auto& buffer = GetEditor().GetActiveTabWindow()->GetActiveWindow()->GetBuffer();

        if (fRand3 > .5f && tabWindows.size() < MaxTabWindows) {
            auto pNewTab = GetEditor().AddTabWindow();
            pNewTab->AddWindow(&buffer, nullptr, RegionLayoutType::HBox);
            GetEditor().SetCurrentTabWindow(pNewTab);
        } else {
            if (tabWindows.size() > 1) {
                GetEditor().RemoveTabWindow(*select_randomly(tabWindows.begin(), tabWindows.end()));
            }
        }

        auto pTab = GetEditor().GetActiveTabWindow();
        auto& windows = pTab->GetWindows();
        auto pActiveWindow = pTab->GetActiveWindow();

        if (fRand1 > .5f && windows.size() > 1) {
            pTab->RemoveWindow(*select_randomly(windows.begin(), windows.end()));
        } else if (windows.size() < 10) {
            pTab->AddWindow(&pActiveWindow->GetBuffer(), *select_randomly(windows.begin(), windows.end()), fRand2 > .5f ? RegionLayoutType::HBox : RegionLayoutType::VBox);
        }
        GetEditor().RequestRefresh();
    }

} // namespace Zep
