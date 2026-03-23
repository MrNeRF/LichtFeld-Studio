#pragma once

#include "zep/mcommon/animation/timer.h"

namespace Zep {

    class ZepEditor;
    class ZepRegressExCommand : public ZepExCommand {
    public:
        ZepRegressExCommand(ZepEditor& editor);

        static void Register(ZepEditor& editor);

        virtual void Tick();
        virtual void Run(const std::vector<std::string>& tokens) override;
        virtual void Notify(std::shared_ptr<ZepMessage> message) override;
        virtual const char* ExCommandName() const override;

    private:
        timer m_timer;
        bool m_enable = false;
        uint32_t m_windowOperationCount = 0;
    };

} // namespace Zep