#include "zep/commands.h"

namespace Zep {

    // Delete Range of chars
    ZepCommand_DeleteRange::ZepCommand_DeleteRange(ZepBuffer& buffer, const GlyphIterator& start, const GlyphIterator& end, const GlyphIterator& cursor, const GlyphIterator& cursorAfter)
        : ZepCommand(buffer, cursor, cursorAfter.Valid() ? cursorAfter : start), m_startIndex(start), m_endIndex(end) {
        assert(m_startIndex.Valid());
        assert(m_endIndex.Valid());

        // We never allow deletion of the '0' at the end of the buffer
        if (buffer.GetWorkingBuffer().empty()) {
            m_endIndex = m_startIndex;
        } else {
            m_endIndex.Clamp();
        }
    }

    void ZepCommand_DeleteRange::Redo() {
        if (m_startIndex != m_endIndex) {
            m_changeRecord.Clear();
            m_buffer.Delete(m_startIndex, m_endIndex, m_changeRecord);
        }
    }

    void ZepCommand_DeleteRange::Undo() {
        if (m_changeRecord.strDeleted.empty())
            return;

        ChangeRecord tempRecord;
        m_buffer.Insert(m_startIndex, m_changeRecord.strDeleted, tempRecord);
    }

    // Insert a string
    ZepCommand_Insert::ZepCommand_Insert(ZepBuffer& buffer, const GlyphIterator& start, const std::string& str, const GlyphIterator& cursor, const GlyphIterator& cursorAfter)
        : ZepCommand(buffer, cursor, cursorAfter.Valid() ? cursorAfter : (start.PeekByteOffset(long(str.size())))), m_startIndex(start), m_strInsert(str) {
        m_startIndex.Clamp();
    }

    void ZepCommand_Insert::Redo() {
        m_changeRecord.Clear();
        bool ret = m_buffer.Insert(m_startIndex, m_strInsert, m_changeRecord);
        assert(ret);
        if (ret == true) {
            m_endIndexInserted = m_startIndex.PeekByteOffset(long(m_strInsert.size()));
        } else {
            m_endIndexInserted.Invalidate();
        }
    }

    void ZepCommand_Insert::Undo() {
        if (m_endIndexInserted.Valid()) {
            ChangeRecord tempRecord;
            m_buffer.Delete(m_startIndex, m_endIndexInserted, tempRecord);
        }
    }

    // Replace
    ZepCommand_ReplaceRange::ZepCommand_ReplaceRange(ZepBuffer& buffer, ReplaceRangeMode currentMode, const GlyphIterator& startIndex, const GlyphIterator& endIndex, const std::string& strReplace, const GlyphIterator& cursor, const GlyphIterator& cursorAfter)
        : ZepCommand(buffer, cursor.Valid() ? cursor : endIndex, cursorAfter.Valid() ? cursorAfter : startIndex), m_startIndex(startIndex), m_endIndex(endIndex), m_strReplace(strReplace), m_mode(currentMode) {
        m_startIndex.Clamp();
    }

    void ZepCommand_ReplaceRange::Redo() {
        if (m_startIndex != m_endIndex) {
            m_changeRecord.Clear();
            m_buffer.Replace(m_startIndex, m_endIndex, m_strReplace, m_mode, m_changeRecord);
        }
    }

    void ZepCommand_ReplaceRange::Undo() {
        if (m_startIndex != m_endIndex) {
            // Replace the range we replaced previously with the old thing
            ChangeRecord temp;
            m_buffer.Replace(m_startIndex, m_mode == ReplaceRangeMode::Fill ? m_endIndex : m_startIndex.PeekByteOffset((long)m_strReplace.length()), m_changeRecord.strDeleted, ReplaceRangeMode::Replace, temp);
        }
    }

} // namespace Zep