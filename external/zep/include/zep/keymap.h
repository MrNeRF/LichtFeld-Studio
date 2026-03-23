#pragma once

#include <functional>
#include <map>
#include <numeric>
#include <regex>
#include <set>
#include <string>
#include <unordered_map>

#include "mcommon/string/stringutils.h"

namespace Zep {

    class ZepMode;

#define DECLARE_COMMANDID(name) const StringId id_##name(#name);

    DECLARE_COMMANDID(YankLine)
    DECLARE_COMMANDID(Yank)

    DECLARE_COMMANDID(NormalMode)
    DECLARE_COMMANDID(VisualMode)
    DECLARE_COMMANDID(InsertMode)
    DECLARE_COMMANDID(ExMode)

    DECLARE_COMMANDID(Find)
    DECLARE_COMMANDID(FindBackwards)

    DECLARE_COMMANDID(VisualSelectInnerWORD)
    DECLARE_COMMANDID(VisualSubstitute)
    DECLARE_COMMANDID(VisualSelectInnerWord)
    DECLARE_COMMANDID(VisualSelectAWord)
    DECLARE_COMMANDID(VisualSelectAWORD)
    DECLARE_COMMANDID(VisualDelete)
    DECLARE_COMMANDID(VisualChange)
    DECLARE_COMMANDID(VisualAppendToLine)
    DECLARE_COMMANDID(VisualAppend)
    DECLARE_COMMANDID(VisualInsertAtFirstChar)
    DECLARE_COMMANDID(FindNext)
    DECLARE_COMMANDID(FindNextDelimiter)
    DECLARE_COMMANDID(NextMarker)
    DECLARE_COMMANDID(PreviousMarker)
    DECLARE_COMMANDID(MotionNextFirstChar)

    DECLARE_COMMANDID(JoinLines)
    DECLARE_COMMANDID(OpenLineBelow)
    DECLARE_COMMANDID(OpenLineAbove)

    DECLARE_COMMANDID(ListBuffers)
    DECLARE_COMMANDID(ListRegisters)

    DECLARE_COMMANDID(Delete)
    DECLARE_COMMANDID(DeleteToLineEnd)
    DECLARE_COMMANDID(DeleteLine)
    DECLARE_COMMANDID(DeleteWord)
    DECLARE_COMMANDID(DeleteWORD)
    DECLARE_COMMANDID(DeleteAWord)
    DECLARE_COMMANDID(DeleteAWORD)
    DECLARE_COMMANDID(DeleteInnerWord)
    DECLARE_COMMANDID(DeleteInnerWORD)
    DECLARE_COMMANDID(DeleteToChar)

    DECLARE_COMMANDID(SubstituteLine)
    DECLARE_COMMANDID(Substitute)

    DECLARE_COMMANDID(ChangeToLineEnd)
    DECLARE_COMMANDID(ChangeLine)
    DECLARE_COMMANDID(ChangeWord)
    DECLARE_COMMANDID(ChangeWORD)
    DECLARE_COMMANDID(ChangeAWord)
    DECLARE_COMMANDID(ChangeAWORD)
    DECLARE_COMMANDID(ChangeInnerWord)
    DECLARE_COMMANDID(ChangeInnerWORD)
    DECLARE_COMMANDID(ChangeIn)
    DECLARE_COMMANDID(ChangeToChar)

    DECLARE_COMMANDID(Replace)

    DECLARE_COMMANDID(PasteAfter)
    DECLARE_COMMANDID(PasteBefore)

    DECLARE_COMMANDID(Append)
    DECLARE_COMMANDID(AppendToLine)
    DECLARE_COMMANDID(InsertAtFirstChar)

    DECLARE_COMMANDID(VisualLineMode)

    DECLARE_COMMANDID(ExBackspace)

    DECLARE_COMMANDID(SwitchToAlternateFile)

    DECLARE_COMMANDID(FontBigger)
    DECLARE_COMMANDID(FontSmaller)

    DECLARE_COMMANDID(QuickSearch)

    DECLARE_COMMANDID(Undo)
    DECLARE_COMMANDID(Redo)

    DECLARE_COMMANDID(MotionNextMarker)
    DECLARE_COMMANDID(MotionPreviousMarker)

    DECLARE_COMMANDID(MotionNextSearch)
    DECLARE_COMMANDID(MotionPreviousSearch)

    DECLARE_COMMANDID(MotionLineEnd)
    DECLARE_COMMANDID(MotionLineBeyondEnd)
    DECLARE_COMMANDID(MotionLineBegin)
    DECLARE_COMMANDID(MotionLineFirstChar)
    DECLARE_COMMANDID(MotionLineHomeToggle)

    DECLARE_COMMANDID(MotionRightSplit)
    DECLARE_COMMANDID(MotionLeftSplit)
    DECLARE_COMMANDID(MotionUpSplit)
    DECLARE_COMMANDID(MotionDownSplit)

    DECLARE_COMMANDID(MotionDown)
    DECLARE_COMMANDID(MotionUp)
    DECLARE_COMMANDID(MotionLeft)
    DECLARE_COMMANDID(MotionRight)

    DECLARE_COMMANDID(MotionWord)
    DECLARE_COMMANDID(MotionBackWord)
    DECLARE_COMMANDID(MotionWORD)
    DECLARE_COMMANDID(MotionBackWORD)
    DECLARE_COMMANDID(MotionEndWord)
    DECLARE_COMMANDID(MotionEndWORD)
    DECLARE_COMMANDID(MotionBackEndWord)
    DECLARE_COMMANDID(MotionBackEndWORD)
    DECLARE_COMMANDID(MotionGotoBeginning)

    DECLARE_COMMANDID(MotionPageForward)
    DECLARE_COMMANDID(MotionPageBackward)

    DECLARE_COMMANDID(MotionHalfPageForward)
    DECLARE_COMMANDID(MotionHalfPageBackward)

    DECLARE_COMMANDID(MotionGotoLine)

    DECLARE_COMMANDID(PreviousTabWindow)
    DECLARE_COMMANDID(NextTabWindow)

    DECLARE_COMMANDID(Save)

    // Standard mode commands
    DECLARE_COMMANDID(StandardCopy)
    DECLARE_COMMANDID(StandardPaste)

    DECLARE_COMMANDID(StandardSelectAll)

    DECLARE_COMMANDID(MotionStandardDown)
    DECLARE_COMMANDID(MotionStandardUp)
    DECLARE_COMMANDID(MotionStandardLeft)
    DECLARE_COMMANDID(MotionStandardRight)

    DECLARE_COMMANDID(MotionStandardLeftWord)
    DECLARE_COMMANDID(MotionStandardRightWord)

    DECLARE_COMMANDID(MotionStandardLeftSelect)
    DECLARE_COMMANDID(MotionStandardRightSelect)
    DECLARE_COMMANDID(MotionStandardUpSelect)
    DECLARE_COMMANDID(MotionStandardDownSelect)

    DECLARE_COMMANDID(MotionStandardLeftWordSelect)
    DECLARE_COMMANDID(MotionStandardRightWordSelect)

    DECLARE_COMMANDID(InsertCarriageReturn)
    DECLARE_COMMANDID(InsertTab)

    // Insert Mode
    DECLARE_COMMANDID(Backspace)
    struct CommandNode {
        // <D> = digits
        // <R> = register
        // <.> = any char
        std::string token;
        StringId commandId;
        std::unordered_map<std::string, std::shared_ptr<CommandNode>> children;
    };

    struct KeyMap {
        bool ignoreFinalDigit = false;
        std::shared_ptr<CommandNode> spRoot = std::make_shared<CommandNode>();
    };

    struct KeyMapResult {
        std::vector<int> captureNumbers;
        std::vector<char> captureChars;
        std::vector<char> captureRegisters;
        std::string commandWithoutGroups;
        bool needMoreChars = false;

        StringId foundMapping;
        std::string searchPath;

        int TotalCount() const {
            if (captureNumbers.empty()) {
                return 1;
            }
            return std::accumulate(captureNumbers.begin(), captureNumbers.end(), 0);
        }

        // Return the first register for commands that only want 1
        char RegisterName() const {
            if (captureRegisters.empty()) {
                return 0;
            }
            return captureRegisters[0];
        }
    };

    enum class KeyMapAdd {
        New,
        Replace
    };

    bool keymap_add(const std::vector<KeyMap*>& maps, const std::vector<std::string>& strCommand, const StringId& commandId, KeyMapAdd opt = KeyMapAdd::Replace);
    bool keymap_add(KeyMap& map, const std::string& strCommand, const StringId& commandId, KeyMapAdd opt = KeyMapAdd::Replace);
    void keymap_find(const KeyMap& map, const std::string& strCommand, KeyMapResult& result);
    void keymap_dump(const KeyMap& map, std::ostringstream& str);

} // namespace Zep
