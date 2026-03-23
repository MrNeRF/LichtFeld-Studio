#pragma once

#include "zep/buffer.h"

namespace Zep {

    class ZepCommand {
    public:
        ZepCommand(ZepBuffer& currentMode, const GlyphIterator& cursorBefore = GlyphIterator(), const GlyphIterator& cursorAfter = GlyphIterator())
            : m_buffer(currentMode), m_cursorBefore(cursorBefore), m_cursorAfter(cursorAfter) {
        }

        virtual ~ZepCommand() {
        }

        virtual void Redo() = 0;
        virtual void Undo() = 0;

        virtual GlyphIterator GetCursorAfter() const {
            return m_cursorAfter;
        }
        virtual GlyphIterator GetCursorBefore() const {
            return m_cursorBefore;
        }

    protected:
        ZepBuffer& m_buffer;
        GlyphIterator m_cursorBefore;
        GlyphIterator m_cursorAfter;
        ChangeRecord m_changeRecord;
    };

    class ZepCommand_GroupMarker : public ZepCommand {
    public:
        ZepCommand_GroupMarker(ZepBuffer& currentMode)
            : ZepCommand(currentMode) {
        }
        virtual void Redo() override {};
        virtual void Undo() override {};
    };

    class ZepCommand_EndGroup : public ZepCommand {
    public:
        ZepCommand_EndGroup(ZepBuffer& currentMode)
            : ZepCommand(currentMode) {
        }
        virtual void Redo() override {};
        virtual void Undo() override {};
    };

    class ZepCommand_DeleteRange : public ZepCommand {
    public:
        ZepCommand_DeleteRange(ZepBuffer& buffer, const GlyphIterator& startIndex, const GlyphIterator& endIndex, const GlyphIterator& cursor = GlyphIterator(), const GlyphIterator& cursorAfter = GlyphIterator());
        virtual ~ZepCommand_DeleteRange(){};

        virtual void Redo() override;
        virtual void Undo() override;

        GlyphIterator m_startIndex;
        GlyphIterator m_endIndex;
    };

    class ZepCommand_ReplaceRange : public ZepCommand {
    public:
        ZepCommand_ReplaceRange(ZepBuffer& buffer, ReplaceRangeMode replaceMode, const GlyphIterator& startIndex, const GlyphIterator& endIndex, const std::string& ch, const GlyphIterator& cursor = GlyphIterator(), const GlyphIterator& cursorAfter = GlyphIterator());
        virtual ~ZepCommand_ReplaceRange(){};

        virtual void Redo() override;
        virtual void Undo() override;

        GlyphIterator m_startIndex;
        GlyphIterator m_endIndex;

        std::string m_strReplace;
        ReplaceRangeMode m_mode;
        ChangeRecord m_deleteStepChange;
    };

    class ZepCommand_Insert : public ZepCommand {
    public:
        ZepCommand_Insert(ZepBuffer& buffer, const GlyphIterator& startIndex, const std::string& str, const GlyphIterator& cursor = GlyphIterator(), const GlyphIterator& cursorAfter = GlyphIterator());
        virtual ~ZepCommand_Insert(){};

        virtual void Redo() override;
        virtual void Undo() override;

        GlyphIterator m_startIndex;
        std::string m_strInsert;
        GlyphIterator m_endIndexInserted;
    };

} // namespace Zep
