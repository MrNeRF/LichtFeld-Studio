#include <cctype>
#include <sstream>

#include "zep/keymap.h"
#include "zep/mode_search.h"
#include "zep/mode_vim.h"
#include "zep/tab_window.h"
#include "zep/theme.h"

#include "zep/mcommon/animation/timer.h"
#include "zep/mcommon/logger.h"
#include "zep/mcommon/string/stringutils.h"

// Note:
// This is a very basic implementation of the common Vim commands that I use: the bare minimum I can live with.
// I do use more, and depending on how much pain I suffer, will add them over time.
// My aim is to make it easy to add commands, so if you want to put something in, please send me a PR.
// The buffer/display search and find support makes it easy to gather the info you need, and the basic insert/delete undo redo commands
// make it easy to find the locations in the buffer
// Important to note: I'm not trying to beat/better Vim here.  Just make an editor I can use in a viewport without feeling pain!
// See further down for what is implemented, and what's on my todo list

// IMPLEMENTED VIM:
// Command counts
// hjkl Motions
// . dot command
// TAB
// w,W,e,E,ge,gE,b,B WORD motions
// u,CTRL+r  Undo, Redo
// i,I,a,A Insert mode (pending undo/redo fix)
// DELETE/BACKSPACE in insert and normal mode; match vim
// Command status bar
// Arrow keys
// '$'
// 'jk' to insert mode
// gg Jump to end
// G Jump to beginning
// CTRL+F/B/D/U page and have page moves
// 'J' join
// D
// dd,d$,x  Delete line, to end of line, chars
// 'v' + 'x'/'d'
// 'y'
// 'p'/'P'
// a-z&a-Z, 0->9, _ " registers
// r Replace with char
// '$'
// 'yy'
// cc
// c$  Change to end of line
// C
// S, s, with visual modes
// ^
// 'O', 'o'
// 'V' (linewise v)
// Y, D, linewise yank/paste
// d[a]<count>w/e  Delete words
// di[({})]"'
// c[a]<count>w/e  Change word
// ci[({})]"'
// ct[char]/dt[char] Change to and delete to
// vi[Ww], va[Ww] Visual inner and word selections
// f[char] find on line
// /[string] find in file, 'n' find next

namespace Zep {

    ZepMode_Vim::ZepMode_Vim(ZepEditor& editor)
        : ZepMode(editor) {
    }

    ZepMode_Vim::~ZepMode_Vim() {
    }

    void ZepMode_Vim::AddOverStrikeMaps() {
        AddKeyMapWithCountRegisters({&m_normalMap, &m_visualMap}, {"r<.>"}, id_Replace);
    }

    void ZepMode_Vim::AddCopyMaps() {
        AddKeyMapWithCountRegisters({&m_normalMap}, {"v"}, id_VisualMode);
        AddKeyMapWithCountRegisters({&m_normalMap}, {"V"}, id_VisualLineMode);
        AddKeyMapWithCountRegisters({&m_visualMap}, {"y"}, id_Yank);
        AddKeyMapWithCountRegisters({&m_normalMap, &m_visualMap}, {"Y"}, id_YankLine);
        AddKeyMapWithCountRegisters({&m_normalMap}, {"yy"}, id_YankLine);

        // Visual mode
        AddKeyMapWithCountRegisters({&m_visualMap}, {"aW"}, id_VisualSelectAWORD);
        AddKeyMapWithCountRegisters({&m_visualMap}, {"aw"}, id_VisualSelectAWord);
        AddKeyMapWithCountRegisters({&m_visualMap}, {"iW"}, id_VisualSelectInnerWORD);
        AddKeyMapWithCountRegisters({&m_visualMap}, {"iw"}, id_VisualSelectInnerWord);
    }

    void ZepMode_Vim::AddPasteMaps() {
    }

    void ZepMode_Vim::Init() {
        for (int i = 0; i <= 9; i++) {
            GetEditor().SetRegister('0' + (const char)i, "");
        }
        GetEditor().SetRegister('"', "");

        SetupKeyMaps();
    }

    void ZepMode_Vim::AddNavigationKeyMaps(bool allowInVisualMode) {
        std::vector<KeyMap*> navigationMaps = {&m_normalMap};
        if (allowInVisualMode) {
            navigationMaps.push_back(&m_visualMap);
        }

        // Up/Down/Left/Right
        AddKeyMapWithCountRegisters(navigationMaps, {"j", "<Down>"}, id_MotionDown);
        AddKeyMapWithCountRegisters(navigationMaps, {"k", "<Up>"}, id_MotionUp);
        AddKeyMapWithCountRegisters(navigationMaps, {"l", "<Right>"}, id_MotionRight);
        AddKeyMapWithCountRegisters(navigationMaps, {"h", "<Left>"}, id_MotionLeft);

        // Page Motions
        AddKeyMapWithCountRegisters(navigationMaps, {"<C-f>", "<PageDown>"}, id_MotionPageForward);
        AddKeyMapWithCountRegisters(navigationMaps, {"<C-b>", "<PageUp>"}, id_MotionPageBackward);
        AddKeyMapWithCountRegisters(navigationMaps, {"<C-d>"}, id_MotionHalfPageForward);
        AddKeyMapWithCountRegisters(navigationMaps, {"<C-u>"}, id_MotionHalfPageBackward);
        AddKeyMapWithCountRegisters(navigationMaps, {"G"}, id_MotionGotoLine);

        // Line Motions
        AddKeyMapWithCountRegisters(navigationMaps, {"$", "<End>"}, id_MotionLineEnd);
        AddKeyMapWithCountRegisters(navigationMaps, {"^"}, id_MotionLineFirstChar);
        keymap_add(navigationMaps, {"0", "<Home>"}, id_MotionLineBegin);

        // Word motions
        AddKeyMapWithCountRegisters(navigationMaps, {"w"}, id_MotionWord);
        AddKeyMapWithCountRegisters(navigationMaps, {"b"}, id_MotionBackWord);
        AddKeyMapWithCountRegisters(navigationMaps, {"W"}, id_MotionWORD);
        AddKeyMapWithCountRegisters(navigationMaps, {"B"}, id_MotionBackWORD);
        AddKeyMapWithCountRegisters(navigationMaps, {"e"}, id_MotionEndWord);
        AddKeyMapWithCountRegisters(navigationMaps, {"E"}, id_MotionEndWORD);
        AddKeyMapWithCountRegisters(navigationMaps, {"ge"}, id_MotionBackEndWord);
        AddKeyMapWithCountRegisters(navigationMaps, {"gE"}, id_MotionBackEndWORD);
        AddKeyMapWithCountRegisters(navigationMaps, {"gg"}, id_MotionGotoBeginning);

        // Navigate between splits
        keymap_add(navigationMaps, {"<C-j>"}, id_MotionDownSplit);
        keymap_add(navigationMaps, {"<C-l>"}, id_MotionRightSplit);
        keymap_add(navigationMaps, {"<C-k>"}, id_MotionUpSplit);
        keymap_add(navigationMaps, {"<C-h>"}, id_MotionLeftSplit);

        // Arrows always navigate in insert mode
        keymap_add({&m_insertMap}, {"<Down>"}, id_MotionDown);
        keymap_add({&m_insertMap}, {"<Up>"}, id_MotionUp);
        keymap_add({&m_insertMap}, {"<Right>"}, id_MotionRight);
        keymap_add({&m_insertMap}, {"<Left>"}, id_MotionLeft);

        keymap_add({&m_insertMap}, {"<End>"}, id_MotionLineBeyondEnd);
        keymap_add({&m_insertMap}, {"<Home>"}, id_MotionLineBegin);
    }

    void ZepMode_Vim::AddSearchKeyMaps() {
        // Normal mode searching
        AddKeyMapWithCountRegisters({&m_normalMap}, {"f<.>"}, id_Find);
        AddKeyMapWithCountRegisters({&m_normalMap}, {"F<.>"}, id_FindBackwards);
        AddKeyMapWithCountRegisters({&m_normalMap}, {";"}, id_FindNext);
        AddKeyMapWithCountRegisters({&m_normalMap}, {"%"}, id_FindNextDelimiter);
        AddKeyMapWithCountRegisters({&m_normalMap}, {"n"}, id_MotionNextSearch);
        AddKeyMapWithCountRegisters({&m_normalMap}, {"N"}, id_MotionPreviousSearch);
        keymap_add({&m_normalMap}, {"<F8>"}, id_MotionNextMarker);
        keymap_add({&m_normalMap}, {"<S-F8>"}, id_MotionPreviousMarker);
    }

    void ZepMode_Vim::AddGlobalKeyMaps() {
        // Global bits
        keymap_add({&m_normalMap, &m_insertMap}, {"<C-p>", "<C-,>"}, id_QuickSearch);
        keymap_add({&m_normalMap}, {":", "/", "?"}, id_ExMode);
        keymap_add({&m_normalMap}, {"H"}, id_PreviousTabWindow);
        keymap_add({&m_normalMap}, {"L"}, id_NextTabWindow);
        keymap_add({&m_normalMap}, {"<C-i><C-o>"}, id_SwitchToAlternateFile);
        keymap_add({&m_normalMap}, {"+"}, id_FontBigger);
        keymap_add({&m_normalMap}, {"-"}, id_FontSmaller);

        // CTRL + =/- to zoom fonts while in non-normal mode
        keymap_add({&m_normalMap, &m_visualMap, &m_insertMap}, {"<C-=>"}, id_FontBigger);
        keymap_add({&m_normalMap, &m_visualMap, &m_insertMap}, {"<C-->"}, id_FontSmaller);
    }

    void ZepMode_Vim::SetupKeyMaps() {
        // Standard choices
        AddGlobalKeyMaps();
        AddNavigationKeyMaps(true);
        AddSearchKeyMaps();
        AddCopyMaps();
        AddPasteMaps();
        AddOverStrikeMaps();

        // Mode switching
        AddKeyMapWithCountRegisters({&m_normalMap, &m_visualMap}, {"<Escape>"}, id_NormalMode);
        keymap_add({&m_insertMap}, {"jk"}, id_NormalMode);
        keymap_add({&m_insertMap}, {"<Escape>"}, id_NormalMode);

        AddKeyMapWithCountRegisters({&m_normalMap}, {"i"}, id_InsertMode);
        AddKeyMapWithCountRegisters({&m_normalMap}, {"S"}, id_SubstituteLine);
        AddKeyMapWithCountRegisters({&m_normalMap}, {"s"}, id_Substitute);
        AddKeyMapWithCountRegisters({&m_normalMap}, {"A"}, id_AppendToLine);
        AddKeyMapWithCountRegisters({&m_normalMap}, {"a"}, id_Append);
        AddKeyMapWithCountRegisters({&m_normalMap}, {"I"}, id_InsertAtFirstChar);
        AddKeyMapWithCountRegisters({&m_visualMap}, {":", "/", "?"}, id_ExMode);

        // Copy/paste
        AddKeyMapWithCountRegisters({&m_normalMap, &m_visualMap}, {"p"}, id_PasteAfter);
        AddKeyMapWithCountRegisters({&m_normalMap, &m_visualMap}, {"P"}, id_PasteBefore);
        AddKeyMapWithCountRegisters({&m_normalMap, &m_visualMap}, {"x", "<Del>"}, id_Delete);

        // Visual changes
        AddKeyMapWithCountRegisters({&m_visualMap}, {"d"}, id_VisualDelete);
        AddKeyMapWithCountRegisters({&m_visualMap}, {"c"}, id_VisualChange);
        AddKeyMapWithCountRegisters({&m_visualMap}, {"s"}, id_VisualSubstitute);

        // Line modifications
        AddKeyMapWithCountRegisters({&m_normalMap, &m_visualMap}, {"J"}, id_JoinLines);
        AddKeyMapWithCountRegisters({&m_visualMap}, {"C"}, id_ChangeLine);
        AddKeyMapWithCountRegisters({&m_normalMap}, {"o"}, id_OpenLineBelow);
        AddKeyMapWithCountRegisters({&m_normalMap}, {"O"}, id_OpenLineAbove);

        // Word modification/text
        AddKeyMapWithCountRegisters({&m_normalMap}, {"d<D>w", "dw"}, id_DeleteWord);
        AddKeyMapWithCountRegisters({&m_normalMap}, {"dW"}, id_DeleteWORD);
        AddKeyMapWithCountRegisters({&m_normalMap}, {"daw"}, id_DeleteAWord);
        AddKeyMapWithCountRegisters({&m_normalMap}, {"daW"}, id_DeleteAWORD);
        AddKeyMapWithCountRegisters({&m_normalMap}, {"diw"}, id_DeleteInnerWord);
        AddKeyMapWithCountRegisters({&m_normalMap}, {"diW"}, id_DeleteInnerWORD);
        AddKeyMapWithCountRegisters({&m_normalMap}, {"D", "d$"}, id_DeleteToLineEnd);
        AddKeyMapWithCountRegisters({&m_normalMap}, {"d<D>d", "dd"}, id_DeleteLine);
        AddKeyMapWithCountRegisters({&m_normalMap}, {"dt<.>"}, id_DeleteToChar);

        AddKeyMapWithCountRegisters({&m_normalMap}, {"cw"}, id_ChangeWord);
        AddKeyMapWithCountRegisters({&m_normalMap}, {"cW"}, id_ChangeWORD);
        AddKeyMapWithCountRegisters({&m_normalMap}, {"ciw"}, id_ChangeInnerWord);
        AddKeyMapWithCountRegisters({&m_normalMap}, {"ciW"}, id_ChangeInnerWORD);
        AddKeyMapWithCountRegisters({&m_normalMap}, {"ci<.>"}, id_ChangeIn);
        AddKeyMapWithCountRegisters({&m_normalMap}, {"caw"}, id_ChangeAWord);
        AddKeyMapWithCountRegisters({&m_normalMap}, {"caW"}, id_ChangeAWORD);
        AddKeyMapWithCountRegisters({&m_normalMap}, {"C", "c$"}, id_ChangeToLineEnd);
        AddKeyMapWithCountRegisters({&m_normalMap}, {"cc"}, id_ChangeLine);
        AddKeyMapWithCountRegisters({&m_normalMap}, {"ct<.>"}, id_ChangeToChar);

        AddKeyMapWithCountRegisters({&m_normalMap}, {"<Return>"}, id_MotionNextFirstChar);

        AddKeyMapWithCountRegisters({&m_normalMap, &m_visualMap}, {"<C-r>"}, id_Redo);
        AddKeyMapWithCountRegisters({&m_normalMap, &m_visualMap}, {"<C-z>", "u"}, id_Undo);

        keymap_add({&m_normalMap}, {"<Backspace>"}, id_MotionStandardLeft);

        // No count allowed on backspace in insert mode, or that would interfere with text.
        keymap_add({&m_insertMap}, {"<Backspace>"}, id_Backspace);
        keymap_add({&m_insertMap}, {"<Del>"}, id_Delete);

        keymap_add({&m_insertMap}, {"<Return>"}, id_InsertCarriageReturn);
        keymap_add({&m_insertMap}, {"<Tab>"}, id_InsertTab);
    }

    void ZepMode_Vim::Begin(ZepWindow* pWindow) {
        ZepMode::Begin(pWindow);

        GetEditor().SetCommandText(m_currentCommand);
        m_currentMode = EditorMode::Normal;
        m_currentCommand.clear();
        m_dotCommand.clear();
    }

    void ZepMode_Vim::PreDisplay(ZepWindow& window) {
        // After .25 seconds of not pressing the 'k' escape code after j,
        // put the j in.
        // We can do better than this and fix the keymapper to handle timed key events.
        // This is an easier fix for now
        if (timer_get_elapsed_seconds(m_lastKeyPressTimer) > .25f && m_currentMode == EditorMode::Insert && m_currentCommand == "j") {
            auto cmd = std::make_shared<ZepCommand_Insert>(
                window.GetBuffer(),
                window.GetBufferCursor(),
                m_currentCommand);
            AddCommand(cmd);

            m_currentCommand = "";
        }
    }

} // namespace Zep
