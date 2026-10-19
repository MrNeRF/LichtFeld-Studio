#include "zep/mode_standard.h"
#include "zep/mcommon/string/stringutils.h"

// Note:
// This is a version of the buffer that behaves like notepad.
// It is basic, but can easily be extended

// STANDARD:
//
// DONE:
// -----
// CTRLZ/Y Undo Redo
// Insert
// Delete/Backspace
// TAB
// Arrows - up,down, left, right
// Home (+Ctrl) move top/startline
// End (+Ctrol) move bottom/endline
// Shift == Select
// control+Shift == select word
// CTRL - CVX (copy paste, cut) + Delete Selection

namespace Zep {

    ZepMode_Standard::ZepMode_Standard(ZepEditor& editor)
        : ZepMode(editor) {
    }

    ZepMode_Standard::~ZepMode_Standard() {
    }

    void ZepMode_Standard::Init() {
        // In standard mode, we always show the insert cursor type
        m_visualCursorType = CursorType::Insert;

        m_modeFlags |= ModeFlags::InsertModeGroupUndo | ModeFlags::StayInInsertMode;

        for (int i = 0; i <= 9; i++) {
            GetEditor().SetRegister('0' + (const char)i, "");
        }
        GetEditor().SetRegister('"', "");

        // Insert Mode
        keymap_add({&m_insertMap}, {"<Backspace>"}, id_Backspace);
        keymap_add({&m_insertMap, &m_visualMap}, {"<Return>"}, id_InsertCarriageReturn);
        keymap_add({&m_insertMap, &m_visualMap}, {"<Tab>"}, id_InsertTab);
        keymap_add({&m_insertMap, &m_visualMap}, {"<Del>"}, id_Delete);
        keymap_add({&m_insertMap, &m_visualMap}, {"<C-S-z>", "<C-y>"}, id_Redo);
        keymap_add({&m_insertMap, &m_visualMap}, {"<C-z>", "<C-u>"}, id_Undo);

        keymap_add({&m_insertMap, &m_visualMap}, {"<Left>"}, id_MotionStandardLeft);
        keymap_add({&m_insertMap, &m_visualMap}, {"<Right>"}, id_MotionStandardRight);
        keymap_add({&m_insertMap, &m_visualMap}, {"<Up>"}, id_MotionStandardUp);
        keymap_add({&m_insertMap, &m_visualMap}, {"<Down>"}, id_MotionStandardDown);
        keymap_add({&m_insertMap, &m_visualMap}, {"<C-Left>"}, id_MotionStandardLeftWord);
        keymap_add({&m_insertMap, &m_visualMap}, {"<C-Right>"}, id_MotionStandardRightWord);
        keymap_add({&m_insertMap, &m_visualMap}, {"<PageDown>"}, id_MotionPageForward);
        keymap_add({&m_insertMap, &m_visualMap}, {"<PageUp>"}, id_MotionPageBackward);

        keymap_add({&m_insertMap, &m_visualMap}, {"<C-S-Left>"}, id_MotionStandardLeftWordSelect);
        keymap_add({&m_insertMap, &m_visualMap}, {"<C-S-Right>"}, id_MotionStandardRightWordSelect);

        keymap_add({&m_insertMap, &m_visualMap}, {"<S-Left>"}, id_MotionStandardLeftSelect);
        keymap_add({&m_insertMap, &m_visualMap}, {"<S-Right>"}, id_MotionStandardRightSelect);
        keymap_add({&m_insertMap, &m_visualMap}, {"<S-Up>"}, id_MotionStandardUpSelect);
        keymap_add({&m_insertMap, &m_visualMap}, {"<S-Down>"}, id_MotionStandardDownSelect);

        keymap_add({&m_visualMap}, {"<C-x>"}, id_Delete);
        keymap_add({&m_visualMap}, {"<Backspace>"}, id_Delete);

        keymap_add({&m_insertMap, &m_visualMap}, {"<C-v>"}, id_StandardPaste);
        keymap_add({&m_visualMap}, {"<C-c>"}, id_StandardCopy);

        keymap_add({&m_insertMap, &m_visualMap}, {"<C-a>"}, id_StandardSelectAll);

        keymap_add({&m_normalMap, &m_visualMap, &m_insertMap}, {"<Escape>"}, id_InsertMode);

        keymap_add({&m_visualMap, &m_insertMap}, {"<C-=>"}, id_FontBigger);
        keymap_add({&m_visualMap, &m_insertMap}, {"<C-->"}, id_FontSmaller);

        keymap_add({&m_visualMap, &m_insertMap}, {"<C-s>"}, id_Save);
    }

    void ZepMode_Standard::AddKeyPress(uint32_t key, uint32_t modifierKeys) {
        auto* window = GetCurrentWindow();
        if (window && (key == ExtKeys::HOME || key == ExtKeys::END) &&
            !(modifierKeys & ModifierKey::Alt)) {
            auto& buffer = window->GetBuffer();
            const auto cursor = window->GetBufferCursor();
            const bool home = key == ExtKeys::HOME;
            auto target = cursor;
            if (modifierKeys & ModifierKey::Ctrl) {
                target = home ? buffer.Begin() : buffer.End();
            } else if (home) {
                target = buffer.GetLinePos(cursor, LineLocation::LineFirstGraphChar);
                if (target == cursor || (modifierKeys & ModifierKey::Shift))
                    target = buffer.GetLinePos(cursor, LineLocation::LineBegin);
            } else {
                target = buffer.GetLinePos(cursor, LineLocation::LineCRBegin);
            }
            if (modifierKeys & ModifierKey::Shift) {
                SetSelection(GetEditorMode() == EditorMode::Visual ? m_visualBegin : cursor, target);
            } else {
                SwitchMode(EditorMode::Insert);
                window->SetBufferCursor(target);
            }
            GetEditor().RequestRefresh();
            return;
        }
        ZepMode::AddKeyPress(key, modifierKeys);
    }

    void ZepMode_Standard::Begin(ZepWindow* pWindow) {
        ZepMode::Begin(pWindow);

        // This will also set the cursor type
        SwitchMode(EditorMode::Insert);
    }

} // namespace Zep
