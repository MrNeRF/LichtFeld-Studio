#pragma once

#include "mode.h"
#include "zep/keymap.h"

class Timer;

namespace Zep {

    struct SpanInfo;

    enum class VimMotion {
        LineBegin,
        LineEnd,
        NonWhiteSpaceBegin,
        NonWhiteSpaceEnd
    };

    class ZepMode_Vim : public ZepMode {
    public:
        ZepMode_Vim(ZepEditor& editor);
        virtual ~ZepMode_Vim();

        static const char* StaticName() {
            return "Vim";
        }

        // Zep Mode
        virtual void Init() override;
        virtual void Begin(ZepWindow* pWindow) override;
        virtual const char* Name() const override {
            return StaticName();
        }
        virtual EditorMode DefaultMode() const override {
            return EditorMode::Normal;
        }
        virtual void PreDisplay(ZepWindow& win) override;
        virtual void SetupKeyMaps();
        virtual void AddOverStrikeMaps();
        virtual void AddCopyMaps();
        virtual void AddPasteMaps();
        virtual void AddGlobalKeyMaps();
        virtual void AddNavigationKeyMaps(bool allowInVisualMode = true);
        virtual void AddSearchKeyMaps();
        virtual bool UsesRelativeLines() const override {
            return true;
        }
    };

} // namespace Zep
