#pragma once

#include "mode.h"
#include <cstdint>

namespace Zep {

    class ZepMode_Standard : public ZepMode {
    public:
        ZepMode_Standard(ZepEditor& editor);
        ~ZepMode_Standard();

        virtual void Init() override;
        virtual void Begin(ZepWindow* pWindow) override;
        void AddKeyPress(uint32_t key, uint32_t modifierKeys = ModifierKey::None) override;
        virtual EditorMode DefaultMode() const override {
            return EditorMode::Insert;
        }

        static const char* StaticName() {
            return "Standard";
        }
        virtual const char* Name() const override {
            return StaticName();
        }

    private:
        std::string keyCache;
    };

} // namespace Zep
