#include "zep/mode.h"
#include "zep/buffer.h"
#include "zep/editor.h"
#include "zep/filesystem.h"
#include "zep/mcommon/logger.h"
#include "zep/mode_search.h"
#include "zep/regress.h"
#include "zep/syntax.h"
#include "zep/tab_window.h"

namespace Zep {
    CommandContext::CommandContext(const std::string& commandIn, ZepMode& md, EditorMode editorMode)
        : owner(md), fullCommand(commandIn), buffer(md.GetCurrentWindow()->GetBuffer()), bufferCursor(md.GetCurrentWindow()->GetBufferCursor()), tempReg("", false), currentMode(editorMode) {
        registers.push('"');
        pRegister = &tempReg;

        bool needMore = false;
        auto extraMaps = md.GetEditor().GetGlobalKeyMaps(md);
        for (auto& extra : extraMaps) {
            keymap_find(*extra, fullCommand, keymap);
            if (keymap.foundMapping.id != 0) {
                break;
            }
        }

        if (keymap.foundMapping.id == 0) {
            keymap_find(owner.GetKeyMappings(currentMode), fullCommand, keymap);
            if (keymap.foundMapping.id == 0 && needMore) {
                keymap.needMoreChars = true;
            }
        }

        GetCommandRegisters();
    }

    void CommandContext::UpdateRegisters() {
        // Store in a register
        if (registers.empty())
            return;

        if (op == CommandOperation::Delete || op == CommandOperation::DeleteLines) {
            beginRange.Clamp();
            endRange.Clamp();
            if (beginRange > endRange) {
                std::swap(beginRange, endRange);
            }

            // TODO: Make a helper for this
            std::string str = std::string(buffer.GetWorkingBuffer().begin() + beginRange.Index(), buffer.GetWorkingBuffer().begin() + endRange.Index());

            // Delete commands fill up 1-9 registers
            if (keymap.commandWithoutGroups[0] == 'd' || keymap.commandWithoutGroups[0] == 'D') {
                for (int i = 9; i > 1; i--) {
                    owner.GetEditor().SetRegister('0' + (char)i, owner.GetEditor().GetRegister('0' + (char)i - 1));
                }
                owner.GetEditor().SetRegister('1', Register(str, (op == CommandOperation::DeleteLines)));
            }

            // Fill up any other required registers
            while (!registers.empty()) {
                owner.GetEditor().SetRegister(registers.top(), Register(str, (op == CommandOperation::DeleteLines)));
                registers.pop();
            }
        } else if (op == CommandOperation::Copy || op == CommandOperation::CopyLines) {
            beginRange = beginRange.Clamp();
            endRange = endRange.Clamp();
            if (beginRange > endRange) {
                std::swap(beginRange, endRange);
            }

            std::string str = std::string(buffer.GetWorkingBuffer().begin() + beginRange.Index(), buffer.GetWorkingBuffer().begin() + endRange.Index());
            while (!registers.empty()) {
                auto& ed = owner.GetEditor();

                // Capital letters append to registers instead of replacing them
                if (registers.top() >= 'A' && registers.top() <= 'Z') {
                    auto chlow = (char)std::tolower(registers.top());
                    ed.SetRegister(chlow, Register(ed.GetRegister(chlow).text + str, (op == CommandOperation::CopyLines)));
                } else {
                    ed.SetRegister(registers.top(), Register(str, op == CommandOperation::CopyLines));
                }
                registers.pop();
            }
        }
    }

    void CommandContext::GetCommandRegisters() {
        // No specified register, so use the default
        if (keymap.RegisterName() == 0) {
            registers.push('+');
            registers.push('*');
        } else {
            if (keymap.RegisterName() == '_') {
                std::stack<char> temp;
                registers.swap(temp);
            } else {
                registers.push(keymap.RegisterName());
                char reg = keymap.RegisterName();

                // Demote capitals to lower registers when pasting (all both)
                if (reg >= 'A' && reg <= 'Z') {
                    reg = (char)std::tolower((char)reg);
                }

                if (owner.GetEditor().GetRegisters().find(std::string({reg})) != owner.GetEditor().GetRegisters().end()) {
                    pRegister = &owner.GetEditor().GetRegister(reg);
                }
            }
        }

        // Default register
        if (!pRegister || pRegister->text.empty()) {
            pRegister = &owner.GetEditor().GetRegister('"');
        }
    }

    ZepMode::ZepMode(ZepEditor& editor)
        : ZepComponent(editor) {
    }

    ZepMode::~ZepMode() {
    }

    ZepWindow* ZepMode::GetCurrentWindow() const {
        // Mode begin should always set this and we should always have a valid window associated with the mode
        assert(m_pCurrentWindow != nullptr);
        return m_pCurrentWindow;
    }

    EditorMode ZepMode::GetEditorMode() const {
        return m_currentMode;
    }

    void ZepMode::AddCommandText(std::string strText) {
        if (m_pCurrentWindow == nullptr) {
            return;
        }

        for (auto& ch : strText) {
            AddKeyPress(ch);
        }
    }

    void ZepMode::ClampCursorForMode() {
        if (m_pCurrentWindow == nullptr) {
            return;
        }

        // Normal mode cursor is never on a CR/0
        // This stops an edit, such as an undo from leaving the cursor on the CR.
        if (m_currentMode == EditorMode::Normal) {
            GetCurrentWindow()->SetBufferCursor(GetCurrentWindow()->GetBuffer().ClampToVisibleLine(GetCurrentWindow()->GetBufferCursor()));
        }
    }

    // Clear our active selection markers and whatever selection we have set in the buffer
    void ZepMode::ClearSelection() {
        auto& buffer = GetCurrentWindow()->GetBuffer();

        buffer.ClearSelection();

        m_selectionStartPos.Invalidate();
        m_visualBegin.Invalidate();
        m_visualEnd.Invalidate();
    }

    void ZepMode::SwitchMode(EditorMode currentMode) {
        if (m_pCurrentWindow == nullptr) {
            return;
        }

        // Don't switch to invalid mode
        if (currentMode == EditorMode::None)
            return;

        // Don't switch to the same thing again
        if (currentMode == m_currentMode)
            return;

        auto pWindow = GetCurrentWindow();
        auto& buffer = pWindow->GetBuffer();
        auto cursor = pWindow->GetBufferCursor();

        // Force normal mode if the file is read only
        if (currentMode == EditorMode::Insert && buffer.HasFileFlags(FileFlags::ReadOnly)) {
            currentMode = DefaultMode();
        }

        // When leaving Ex mode, reset search markers
        if (m_currentMode == EditorMode::Ex) {
            buffer.HideMarkers(RangeMarkerType::Search);
        } else if (m_currentMode == EditorMode::Insert) {
            // When switching back to normal mode, put the cursor on the last character typed
            if (currentMode == EditorMode::Normal) {
                // Move back, but not to the previous line
                GlyphIterator itr(cursor);
                itr.MoveClamped(-1);
                GetCurrentWindow()->SetBufferCursor(itr);
            }
        }

        m_currentMode = currentMode;

        switch (currentMode) {
        case EditorMode::Normal: {
            ClearSelection();
            ClampCursorForMode();
            ResetCommand();
        } break;
        case EditorMode::Insert: {
            ClearSelection();
            ResetCommand();
        } break;
        case EditorMode::Visual: {
            ResetCommand();
        } break;
        case EditorMode::Ex: {
            m_exCommandStartLocation = cursor;
            // Ensure we show the command at the bottom
            GetEditor().SetCommandText(m_currentCommand);
        } break;
        default:
        case EditorMode::None:
            break;
        }
    }

    std::string ZepMode::ConvertInputToMapString(uint32_t key, uint32_t modifierKeys) {
        bool brackets = false;
        std::string str;
        bool closeBracket = false;
        if (modifierKeys & ModifierKey::Ctrl) {
            str += "C-";
        }

        if (modifierKeys & ModifierKey::Alt) {
            str += "A-";
        }

        if (modifierKeys & ModifierKey::Shift) {
            if (modifierKeys != ModifierKey::Shift || key < ' ') {
                str += "S-";
            }
        }
        brackets = !str.empty();

        std::string mapped;

#define COMPARE_STR(a, b) \
    if (key == b)         \
        mapped = #a;

        COMPARE_STR(Return, ExtKeys::RETURN)
        COMPARE_STR(Escape, ExtKeys::ESCAPE)
        COMPARE_STR(Backspace, ExtKeys::BACKSPACE)
        COMPARE_STR(Left, ExtKeys::LEFT)
        COMPARE_STR(Right, ExtKeys::RIGHT)
        COMPARE_STR(Up, ExtKeys::UP)
        COMPARE_STR(Down, ExtKeys::DOWN)
        COMPARE_STR(Tab, ExtKeys::TAB)
        COMPARE_STR(Del, ExtKeys::DEL)
        COMPARE_STR(Home, ExtKeys::HOME)
        COMPARE_STR(End, ExtKeys::END)
        COMPARE_STR(PageDown, ExtKeys::PAGEDOWN)
        COMPARE_STR(PageUp, ExtKeys::PAGEUP)
        COMPARE_STR(F1, ExtKeys::F1)
        COMPARE_STR(F2, ExtKeys::F2)
        COMPARE_STR(F3, ExtKeys::F3)
        COMPARE_STR(F4, ExtKeys::F4)
        COMPARE_STR(F5, ExtKeys::F5)
        COMPARE_STR(F6, ExtKeys::F6)
        COMPARE_STR(F7, ExtKeys::F7)
        COMPARE_STR(F8, ExtKeys::F8)
        COMPARE_STR(F9, ExtKeys::F9)
        COMPARE_STR(F10, ExtKeys::F10)
        COMPARE_STR(F11, ExtKeys::F11)
        COMPARE_STR(F12, ExtKeys::F12)

        if (!mapped.empty()) {
            brackets = true;
            str += mapped;
        } else {
            str += std::string((const char*)&key);
        }

        return brackets ? "<" + str + ">" : str;
    }

    // Handle a key press, convert it to an input command and context, and return it.
    void ZepMode::AddKeyPress(uint32_t key, uint32_t modifierKeys) {
        if (m_pCurrentWindow == nullptr) {
            return;
        }

        // Temporarily accept up to 255; this is not fully allowing UTF8 input yet (though display and management of buffers with
        // utf8 is just fine)
        key &= 0xFF;

        // Keys in this range converted to UTF8.  I need to figure out how to generically receive UTF8 here, but this
        // temporary fix enables £-sign and other specials to display and work correctly
        if (key >= 127 && key <= 255) {

            key = (0x00C2 | ((key & 0xFF) << 8));
        }

        m_lastKey = key;

        // Get the new command by parsing out the keys
        // We convert CTRL + f to a string: "<C-f>"
        HandleMappedInput(ConvertInputToMapString(key, modifierKeys));

        if (m_pCurrentWindow) {
            auto notifier = m_pCurrentWindow->GetBuffer().GetPostKeyNotifier();
            if (notifier != nullptr) {
                notifier(key, modifierKeys);
            }
        }

        timer_restart(m_lastKeyPressTimer);
    }

    void ZepMode::HandleMappedInput(const std::string& input) {
        if (input.empty()) {
            return;
        }

        // Special case, dot command (do last edit again)
        // Dot command is complicated, this is my second attempt at implementing it and is less
        // complex.  The approach is to store relevent key strokes for the last edit operation,
        // and replay them when the user uses the dot.
        if (m_currentMode == EditorMode::Normal && input[input.size() - 1] == '.') {
            // Save and restore the last command while doing it.
            auto lastCommand = m_dotCommand;
            for (auto& last : lastCommand) {
                HandleMappedInput(std::string(1, last));
            }
            m_dotCommand = lastCommand;

            SwitchMode(EditorMode::Normal);

            return;
        }

        // The current command is our currently typed multi-key operation
        m_currentCommand += input;

        // Reset the timer for the last edit, for time-sensitive key strokes
        GetEditor().ResetLastEditTimer();

        // Reset the cursor to keep it visible during typing, and not flashing
        GetEditor().ResetCursorTimer();

        // Reset command text - it may get updated later.
        GetEditor().SetCommandText("");

        // Figure out the command we have typed. foundCommand means that the command was interpreted and understood.
        // If spCommand is returned, then there is an atomic command operation that needs to be done.
        auto spContext = std::make_shared<CommandContext>(m_currentCommand, *this, m_currentMode);

        // Before handling the command, change the command text, since the command might override it
        if (GetEditor().GetConfig().showNormalModeKeyStrokes && (m_currentMode == EditorMode::Normal || m_currentMode == EditorMode::Visual)) {
            GetEditor().SetCommandText(spContext->keymap.searchPath);
        }

        spContext->foundCommand = GetCommand(*spContext);

        // Stay in insert mode unless commanded otherwise
        if (spContext->commandResult.modeSwitch == EditorMode::None && spContext->foundCommand) {
            if (m_modeFlags & ModeFlags::StayInInsertMode) {
                spContext->commandResult.modeSwitch = EditorMode::Insert;
            }
        }

        // A lambda to check for a pending mode switch after the command
        auto enteringMode = [=](auto mode) {
            if (m_currentMode != spContext->commandResult.modeSwitch && spContext->commandResult.modeSwitch == mode) {
                return true;
            }
            return false;
        };

        // Escape Nukes the current command - we handle it in the keyboard mappings after that
        // TODO: This feels awkward
        if (m_lastKey == ExtKeys::ESCAPE) {
            m_currentCommand.clear();
        }

        // Did we find something to do?
        if (spContext->foundCommand) {
            // It's an undoable command  - add it
            // Note: a command here is something that modifies the text.  It can be something like a delete
            // or a simple insert
            if (spContext->commandResult.spCommand) {
                // If not in insert mode, begin the group, because we have started a new operation
                // i.e. one group before adding characters and typing stuff.
                if (m_currentMode != EditorMode::Insert || ZTestFlags(spContext->commandResult.flags, CommandResultFlags::BeginUndoGroup)) {
                    AddCommand(std::make_shared<ZepCommand_GroupMarker>(spContext->buffer));

                    // Record for the dot command
                    m_dotCommand = m_currentCommand;
                } else {
                    // In insert mode keep the text for the dot command.  An insert adds a command too!
                    // Accumulate
                    m_dotCommand += input;
                }

                // Do the command
                AddCommand(spContext->commandResult.spCommand);
            } else {
                // This command didn't change anything, but switched into insert mode, so
                // remember the dot command that did it
                if (enteringMode(EditorMode::Insert)) {
                    AddCommand(std::make_shared<ZepCommand_GroupMarker>(spContext->buffer));
                    m_dotCommand = m_currentCommand;
                }
            }

            // If the command can't manage the count, we do it
            // Maybe all commands should handle the count?  What are the implications of that?  This bit is a bit messy
            if (!(spContext->commandResult.flags & CommandResultFlags::HandledCount)) {
                // Ignore count == 1, we already did it
                for (int i = 1; i < spContext->keymap.TotalCount(); i++) {
                    // May immediate execute and not return a command...
                    // Create a new 'inner' spContext-> for the next command, because we need to re-initialize the command
                    // spContext-> for 'after' what just happened!
                    CommandContext contextInner(m_currentCommand, *this, m_currentMode);
                    if (GetCommand(contextInner) && contextInner.commandResult.spCommand) {
                        // Actually queue/do command
                        AddCommand(contextInner.commandResult.spCommand);
                    }
                }
            }

            // A mode to switch to after the command is done
            SwitchMode(spContext->commandResult.modeSwitch);

            // If not in ex mode, wait for a new command
            // Can this be cleaner?
            if (m_currentMode != EditorMode::Ex) {
                ResetCommand();
            }

            // Motions can update the visual selection
            UpdateVisualSelection();
        } else {
            // If not found, and there was no request for more characters, and we aren't in Ex mode
            if (m_currentMode != EditorMode::Ex) {
                if (HandleIgnoredInput(*spContext) || !spContext->keymap.needMoreChars) {
                    ResetCommand();
                }
            }
        }
        ClampCursorForMode();
    }

    void ZepMode::AddCommand(std::shared_ptr<ZepCommand> spCmd) {
        if (m_pCurrentWindow == nullptr) {
            return;
        }

        if (GetCurrentWindow()->GetBuffer().HasFileFlags(FileFlags::Locked)) {
            // Ignore commands on buffers because we are view only,
            // and all commands currently modify the buffer!
            return;
        }

        auto& undoStack = m_pCurrentWindow->GetBuffer().GetUndoStack();
        auto& redoStack = m_pCurrentWindow->GetBuffer().GetRedoStack();

        spCmd->Redo();
        undoStack.push(spCmd);

        // Can't redo anything beyond this point
        std::stack<std::shared_ptr<ZepCommand>> empty;
        redoStack.swap(empty);

        if (spCmd->GetCursorAfter().Valid()) {
            GetCurrentWindow()->SetBufferCursor(spCmd->GetCursorAfter());
        }
    }

    void ZepMode::Redo() {
        if (m_pCurrentWindow == nullptr) {
            return;
        }

        auto& undoStack = m_pCurrentWindow->GetBuffer().GetUndoStack();
        auto& redoStack = m_pCurrentWindow->GetBuffer().GetRedoStack();

        if (redoStack.empty())
            return;

        if (std::dynamic_pointer_cast<ZepCommand_GroupMarker>(redoStack.top()) != nullptr) {
            undoStack.push(redoStack.top());
            redoStack.pop();
        }

        while (!redoStack.empty()) {
            auto& spCommand = redoStack.top();
            spCommand->Redo();

            if (spCommand->GetCursorAfter().Valid()) {
                GetCurrentWindow()->SetBufferCursor(spCommand->GetCursorAfter());
            }

            undoStack.push(spCommand);
            redoStack.pop();

            if (std::dynamic_pointer_cast<ZepCommand_GroupMarker>(spCommand) != nullptr) {
                break;
            }
        };
    }

    void ZepMode::Undo() {
        if (m_pCurrentWindow == nullptr) {
            return;
        }

        auto& undoStack = m_pCurrentWindow->GetBuffer().GetUndoStack();
        auto& redoStack = m_pCurrentWindow->GetBuffer().GetRedoStack();
        if (undoStack.empty())
            return;

        if (std::dynamic_pointer_cast<ZepCommand_GroupMarker>(undoStack.top()) != nullptr) {
            redoStack.push(undoStack.top());
            undoStack.pop();
        }

        while (!undoStack.empty()) {
            auto& spCommand = undoStack.top();
            spCommand->Undo();

            if (spCommand->GetCursorBefore().Valid()) {
                GetCurrentWindow()->SetBufferCursor(spCommand->GetCursorBefore());
            }

            redoStack.push(spCommand);
            undoStack.pop();

            if (std::dynamic_pointer_cast<ZepCommand_GroupMarker>(spCommand) != nullptr) {
                break;
            }
        };
    }

    GlyphRange ZepMode::GetInclusiveVisualRange() const {
        // Clamp and orient the correct way around
        auto startOffset = m_visualBegin.Clamped();
        auto endOffset = m_visualEnd.Clamped();

        if (DefaultMode() == EditorMode::Insert) {
            if (startOffset == endOffset) {
                // In standard mode, if the cursor has not moved, this is not a selection
                startOffset = endOffset = GlyphIterator();
            } else {
                // In standard/insert mode, selections exclude the last character,
                // depending on direction.  Starting from an insert mode cursor,
                // selections are not made until at least one step
                // (remembering that an insert mode cursor is effectively sitting on the character in front of it.)
                if (endOffset > startOffset) {
                    endOffset--;
                } else {
                    startOffset--;
                }
            }
        }

        if (startOffset > endOffset) {
            std::swap(startOffset, endOffset);
        }

        return GlyphRange(startOffset, endOffset);
    }

    const std::string& ZepMode::GetLastCommand() const {
        return m_dotCommand;
    }

    bool ZepMode::GetCommand(CommandContext& context) {
        auto bufferCursor = GetCurrentWindow()->GetBufferCursor();
        auto& buffer = GetCurrentWindow()->GetBuffer();

        if (m_currentMode == EditorMode::Ex) {
            // TODO: Is it possible extend our key mapping to better process ex commands?  Or are these
            // too specialized?
            if (HandleExCommand(context.fullCommand)) {
                // buffer.GetMode()->Begin(GetCurrentWindow());
                SwitchMode(DefaultMode());
                ResetCommand();
                return true;
            }
            GetEditor().SetCommandText(m_currentCommand);

            return false;
        }

        // The keymapper is waiting for more input
        if (context.keymap.needMoreChars) {
            return false;
        }

        // This flag is for non-modal editors which like to break insertions into undo groups.
        // Vim, for example, doesn't do that; an insert mode operation is a single 'group'
        bool shouldGroupInserts = ZTestFlags(m_modeFlags, ModeFlags::InsertModeGroupUndo);

        GlyphIterator cursorItr = bufferCursor;

        auto mappedCommand = context.keymap.foundMapping;

        // Using movement in insert mode vs normal mode is different; should be able to move the cursor beyond the end of the line in insert mode.
        auto movementLineEndLocation = (context.currentMode == EditorMode::Insert ? LineLocation::LineCRBegin : LineLocation::LineLastNonCR);

        auto pEx = GetEditor().FindExCommand(mappedCommand);
        if (pEx) {
            pEx->Run();
            return true;
        }

        // Inherited modes can handle extra commands this way
        if (HandleSpecialCommand(context)) {
            return true;
        }

        if (mappedCommand == id_NormalMode) {
            // TODO: I think this should be a 'command' which would get replayed with dot;
            // instead of special casing it later, we could just insert it into the stream of commands
            context.commandResult.modeSwitch = EditorMode::Normal;
            return true;
        } else if (mappedCommand == id_ExMode) {
            context.commandResult.modeSwitch = EditorMode::Ex;
            return true;
        }
        // Control
        else if (mappedCommand == id_MotionNextMarker) {
            auto pFound = buffer.FindNextMarker(GetCurrentWindow()->GetBufferCursor(), Direction::Forward, RangeMarkerType::Mark);
            if (pFound) {
                GetCurrentWindow()->SetBufferCursor(GlyphIterator(&context.buffer, pFound->GetRange().first));
            }
            return true;
        } else if (mappedCommand == id_MotionPreviousMarker) {
            auto pFound = buffer.FindNextMarker(GetCurrentWindow()->GetBufferCursor(), Direction::Backward, RangeMarkerType::Mark);
            if (pFound) {
                GetCurrentWindow()->SetBufferCursor(GlyphIterator(&context.buffer, pFound->GetRange().first));
            }
            return true;
        } else if (mappedCommand == id_MotionNextSearch) {
            auto pFound = buffer.FindNextMarker(GetCurrentWindow()->GetBufferCursor(), m_lastSearchDirection, RangeMarkerType::Search);
            if (pFound) {
                GetCurrentWindow()->SetBufferCursor(GlyphIterator(&context.buffer, pFound->GetRange().first));
            }
            return true;
        } else if (mappedCommand == id_MotionPreviousSearch) {
            auto pFound = buffer.FindNextMarker(GetCurrentWindow()->GetBufferCursor(), m_lastSearchDirection == Direction::Forward ? Direction::Backward : Direction::Forward, RangeMarkerType::Search);
            if (pFound) {
                GetCurrentWindow()->SetBufferCursor(GlyphIterator(&context.buffer, pFound->GetRange().first));
            }
            return true;
        } else if (mappedCommand == id_SwitchToAlternateFile) {
            // This is a quick and easy 'alternate file swap'.
            // It searches a preset list of useful folder targets around the current file.
            // A better alternative might be a wildcard list of relations, but this works well enough
            // It also only looks for a file with the same name and different extension!
            // it is good enough for my current needs...
            auto path = buffer.GetFilePath();
            if (!path.empty() && GetEditor().GetFileSystem().Exists(path)) {
                auto ext = path.extension();
                auto searchPaths = std::vector<fs::path>{
                    path.parent_path(),
                    path.parent_path().parent_path(),
                    path.parent_path().parent_path().parent_path()};

                auto ignoreFolders = std::vector<std::string>{"build", ".git", "obj", "debug", "release"};

                auto priorityFolders = std::vector<std::string>{"source", "include", "src", "inc", "lib"};

                // Search the paths, starting near and widening
                for (auto& p : searchPaths) {
                    if (p.empty())
                        continue;

                    bool found = false;

                    // Look for the priority folder locations
                    std::vector<fs::path> searchFolders{path.parent_path()};
                    for (auto& priorityFolder : priorityFolders) {
                        GetEditor().GetFileSystem().ScanDirectory(p, [&](const fs::path& currentPath, bool& recurse) {
                            recurse = false;
                            if (GetEditor().GetFileSystem().IsDirectory(currentPath)) {
                                auto lower = string_tolower(currentPath.filename().string());
                                if (std::find(ignoreFolders.begin(), ignoreFolders.end(), lower) != ignoreFolders.end()) {
                                    return true;
                                }

                                if (priorityFolder == lower) {
                                    searchFolders.push_back(currentPath);
                                }
                            }
                            return true;
                        });
                    }

                    for (auto& folder : searchFolders) {
                        ZLOG(INFO, "Searching: " << folder.string());

                        GetEditor().GetFileSystem().ScanDirectory(folder, [&](const fs::path& currentPath, bool& recurse) {
                            recurse = true;
                            if (path.stem() == currentPath.stem() && !(currentPath.extension() == path.extension())) {
                                auto load = GetEditor().GetFileBuffer(currentPath, 0, true);
                                if (load != nullptr) {
                                    GetCurrentWindow()->SetBuffer(load);
                                    found = true;
                                    return false;
                                }
                            }
                            return true;
                        });
                        if (found)
                            return true;
                    }
                }
            }
        } else if (mappedCommand == id_FontBigger) {
            GetEditor().GetDisplay().Bigger();
            return true;
        } else if (mappedCommand == id_FontSmaller) {
            GetEditor().GetDisplay().Smaller();
            return true;
        }
        // Moving between splits
        else if (mappedCommand == id_MotionDownSplit) {
            GetCurrentWindow()->GetTabWindow().DoMotion(WindowMotion::Down);
            return true;
        } else if (mappedCommand == id_MotionUpSplit) {
            GetCurrentWindow()->GetTabWindow().DoMotion(WindowMotion::Up);
            return true;
        } else if (mappedCommand == id_MotionLeftSplit) {
            GetCurrentWindow()->GetTabWindow().DoMotion(WindowMotion::Left);
            return true;
        } else if (mappedCommand == id_MotionRightSplit) {
            GetCurrentWindow()->GetTabWindow().DoMotion(WindowMotion::Right);
            return true;
        }
        // global search
        else if (mappedCommand == id_QuickSearch) {
            GetEditor().AddSearch();
            return true;
        } else if (mappedCommand == id_Redo) {
            context.commandResult.modeSwitch = DefaultMode();
            Redo();
            return true;
        } else if (mappedCommand == id_Undo) {
            context.commandResult.modeSwitch = DefaultMode();
            Undo();
            return true;
        } else if (mappedCommand == id_MotionLineEnd) {
            GetCurrentWindow()->SetBufferCursor(context.buffer.GetLinePos(bufferCursor, LineLocation::LineLastNonCR));
            return true;
        } else if (mappedCommand == id_MotionLineBeyondEnd) {
            GetCurrentWindow()->SetBufferCursor(context.buffer.GetLinePos(bufferCursor, LineLocation::LineCRBegin));
            return true;
        } else if (mappedCommand == id_MotionLineBegin) {
            GetCurrentWindow()->SetBufferCursor(context.buffer.GetLinePos(bufferCursor, LineLocation::LineBegin));
            return true;
        } else if (mappedCommand == id_MotionLineFirstChar) {
            GetCurrentWindow()->SetBufferCursor(context.buffer.GetLinePos(bufferCursor, LineLocation::LineFirstGraphChar));
            return true;
        } else if (mappedCommand == id_MotionLineHomeToggle) {
            auto newCursorPos = context.buffer.GetLinePos(bufferCursor, LineLocation::LineFirstGraphChar);
            if (bufferCursor == newCursorPos) {
                newCursorPos = context.buffer.GetLinePos(bufferCursor, LineLocation::LineBegin);
            }
            GetCurrentWindow()->SetBufferCursor(newCursorPos);
            return true;
        }
        // Moving between tabs
        else if (mappedCommand == id_PreviousTabWindow) {
            GetEditor().PreviousTabWindow();
            return true;
        } else if (mappedCommand == id_NextTabWindow) {
            GetEditor().NextTabWindow();
            return true;
        } else if (mappedCommand == id_MotionDown) {
            GetCurrentWindow()->MoveCursorY(context.keymap.TotalCount(), movementLineEndLocation);
            context.commandResult.flags |= CommandResultFlags::HandledCount;
            return true;
        } else if (mappedCommand == id_MotionUp) {
            GetCurrentWindow()->MoveCursorY(-context.keymap.TotalCount(), movementLineEndLocation);
            context.commandResult.flags |= CommandResultFlags::HandledCount;
            return true;
        } else if (mappedCommand == id_MotionRight) {
            GetCurrentWindow()->SetBufferCursor(cursorItr.MoveClamped(context.keymap.TotalCount(), movementLineEndLocation));
            context.commandResult.flags |= CommandResultFlags::HandledCount;
            return true;
        } else if (mappedCommand == id_MotionLeft) {
            GetCurrentWindow()->SetBufferCursor(cursorItr.MoveClamped(-context.keymap.TotalCount()));
            context.commandResult.flags |= CommandResultFlags::HandledCount;
            return true;
        } else if (mappedCommand == id_MotionStandardRight) {
            GetCurrentWindow()->SetBufferCursor(cursorItr.Move(context.keymap.TotalCount()));
            context.commandResult.flags |= CommandResultFlags::HandledCount;
            return true;
        } else if (mappedCommand == id_MotionStandardLeft) {
            GetCurrentWindow()->SetBufferCursor(cursorItr.Move(-context.keymap.TotalCount()));
            context.commandResult.flags |= CommandResultFlags::HandledCount;
            return true;
        } else if (mappedCommand == id_MotionStandardUp) {
            GetCurrentWindow()->MoveCursorY(-1, LineLocation::LineCRBegin);
            return true;
        } else if (mappedCommand == id_MotionStandardDown) {
            GetCurrentWindow()->MoveCursorY(1, LineLocation::LineCRBegin);
            return true;
        } else if (mappedCommand == id_StandardSelectAll) {
            m_visualBegin = context.buffer.Begin();
            m_visualEnd = context.buffer.End();
            auto range = GetInclusiveVisualRange();
            if (!range.Valid()) {
                return true;
            }
            // In standard mode, we don't normally select the last character, but here we want it.
            range.second++;

            context.commandResult.modeSwitch = EditorMode::Visual;
            GetCurrentWindow()->GetBuffer().SetSelection(range);
            GetCurrentWindow()->SetBufferCursor(range.second);
            return true;
        } else if (mappedCommand == id_MotionStandardRightSelect) {
            context.commandResult.modeSwitch = EditorMode::Visual;
            if (m_currentMode != EditorMode::Visual) {
                m_visualBegin = GetCurrentWindow()->GetBufferCursor();
            }
            GetCurrentWindow()->SetBufferCursor(bufferCursor + 1);
            UpdateVisualSelection();
            return true;
        } else if (mappedCommand == id_MotionStandardLeftSelect) {
            context.commandResult.modeSwitch = EditorMode::Visual;
            if (m_currentMode != EditorMode::Visual) {
                m_visualBegin = GetCurrentWindow()->GetBufferCursor();
            }
            GetCurrentWindow()->SetBufferCursor(bufferCursor - 1);
            UpdateVisualSelection();
            return true;
        } else if (mappedCommand == id_MotionStandardUpSelect) {
            context.commandResult.modeSwitch = EditorMode::Visual;
            if (m_currentMode != EditorMode::Visual) {
                m_visualBegin = GetCurrentWindow()->GetBufferCursor();
            }
            GetCurrentWindow()->MoveCursorY(-1, LineLocation::LineCRBegin);
            UpdateVisualSelection();
            return true;
        } else if (mappedCommand == id_MotionStandardDownSelect) {
            context.commandResult.modeSwitch = EditorMode::Visual;
            if (m_currentMode != EditorMode::Visual) {
                m_visualBegin = GetCurrentWindow()->GetBufferCursor();
            }
            GetCurrentWindow()->MoveCursorY(1, LineLocation::LineCRBegin);
            UpdateVisualSelection();
            return true;
        } else if (mappedCommand == id_MotionStandardRightWord) {
            auto target = buffer.StandardCtrlMotion(bufferCursor, Direction::Forward);
            GetCurrentWindow()->SetBufferCursor(target.second);
            return true;
        } else if (mappedCommand == id_MotionStandardLeftWord) {
            auto target = buffer.StandardCtrlMotion(bufferCursor, Direction::Backward);
            GetCurrentWindow()->SetBufferCursor(target.second);
            return true;
        } else if (mappedCommand == id_MotionStandardRightWordSelect) {
            context.commandResult.modeSwitch = EditorMode::Visual;
            if (m_currentMode != EditorMode::Visual) {
                m_visualBegin = GetCurrentWindow()->GetBufferCursor();
            }
            auto target = buffer.StandardCtrlMotion(bufferCursor, Direction::Forward);
            GetCurrentWindow()->SetBufferCursor(target.second);
            UpdateVisualSelection();
            return true;
        } else if (mappedCommand == id_MotionStandardLeftWordSelect) {
            context.commandResult.modeSwitch = EditorMode::Visual;
            if (m_currentMode != EditorMode::Visual) {
                m_visualBegin = GetCurrentWindow()->GetBufferCursor();
            }
            auto target = buffer.StandardCtrlMotion(bufferCursor, Direction::Backward);
            GetCurrentWindow()->SetBufferCursor(target.second);
            UpdateVisualSelection();
            return true;
        } else if (mappedCommand == id_MotionPageForward) {
            // Note: the vim spec says 'visible lines - 2' for a 'page'.
            // We jump the max possible lines, which might hit the end of the text; this matches observed vim behavior
            GetCurrentWindow()->MoveCursorY((GetCurrentWindow()->GetMaxDisplayLines() - 2) * context.keymap.TotalCount());
            context.commandResult.flags |= CommandResultFlags::HandledCount;
            return true;
        } else if (mappedCommand == id_MotionHalfPageForward) {
            // Note: the vim spec says 'half visible lines' for up/down
            GetCurrentWindow()->MoveCursorY((GetCurrentWindow()->GetNumDisplayedLines() / 2) * context.keymap.TotalCount());
            context.commandResult.flags |= CommandResultFlags::HandledCount;
            return true;
        } else if (mappedCommand == id_MotionPageBackward) {
            // Note: the vim spec says 'visible lines - 2' for a 'page'
            GetCurrentWindow()->MoveCursorY(-(GetCurrentWindow()->GetMaxDisplayLines() - 2) * context.keymap.TotalCount());
            context.commandResult.flags |= CommandResultFlags::HandledCount;
            return true;
        } else if (mappedCommand == id_MotionHalfPageBackward) {
            GetCurrentWindow()->MoveCursorY(-(GetCurrentWindow()->GetNumDisplayedLines() / 2) * context.keymap.TotalCount());
            context.commandResult.flags |= CommandResultFlags::HandledCount;
            return true;
        } else if (mappedCommand == id_MotionGotoLine) {
            if (!context.keymap.captureNumbers.empty()) {
                // In Vim, 0G means go to end!  1G is the first line...
                long count = context.keymap.TotalCount() - 1;
                count = std::min(context.buffer.GetLineCount() - 1, count);
                if (count < 0)
                    count = context.buffer.GetLineCount() - 1;

                ByteRange range;
                if (context.buffer.GetLineOffsets(count, range)) {
                    GetCurrentWindow()->SetBufferCursor(GlyphIterator(&context.buffer, range.first));
                }
            } else {
                // Move right to the end
                auto lastLine = context.buffer.GetLinePos(context.buffer.End(), LineLocation::LineBegin);
                GetCurrentWindow()->SetBufferCursor(lastLine);
                context.commandResult.flags |= CommandResultFlags::HandledCount;
            }
            return true;
        } else if (mappedCommand == id_Backspace) {
            // In insert mode, we are 'on' the character after the one we want to delete
            context.beginRange = cursorItr.Peek(-1);
            context.endRange = cursorItr;
            context.op = CommandOperation::Delete;
        } else if (mappedCommand == id_MotionWord) {
            auto target = context.buffer.WordMotion(context.bufferCursor, SearchType::Word, Direction::Forward);
            GetCurrentWindow()->SetBufferCursor(target);
            return true;
        } else if (mappedCommand == id_MotionWORD) {
            auto target = context.buffer.WordMotion(context.bufferCursor, SearchType::WORD, Direction::Forward);
            GetCurrentWindow()->SetBufferCursor(target);
            return true;
        } else if (mappedCommand == id_MotionBackWord) {
            auto target = context.buffer.WordMotion(context.bufferCursor, SearchType::Word, Direction::Backward);
            GetCurrentWindow()->SetBufferCursor(target);
            return true;
        } else if (mappedCommand == id_MotionBackWORD) {
            auto target = context.buffer.WordMotion(context.bufferCursor, SearchType::WORD, Direction::Backward);
            GetCurrentWindow()->SetBufferCursor(target);
            return true;
        } else if (mappedCommand == id_MotionEndWord) {
            auto target = context.buffer.EndWordMotion(context.bufferCursor, SearchType::Word, Direction::Forward);
            GetCurrentWindow()->SetBufferCursor(target);
            return true;
        } else if (mappedCommand == id_MotionEndWORD) {
            auto target = context.buffer.EndWordMotion(context.bufferCursor, SearchType::WORD, Direction::Forward);
            GetCurrentWindow()->SetBufferCursor(target);
            return true;
        } else if (mappedCommand == id_MotionBackEndWord) {
            auto target = context.buffer.EndWordMotion(context.bufferCursor, SearchType::Word, Direction::Backward);
            GetCurrentWindow()->SetBufferCursor(target);
            return true;
        } else if (mappedCommand == id_MotionBackEndWORD) {
            auto target = context.buffer.EndWordMotion(context.bufferCursor, SearchType::WORD, Direction::Backward);
            GetCurrentWindow()->SetBufferCursor(target);
            return true;
        } else if (mappedCommand == id_MotionGotoBeginning) {
            GetCurrentWindow()->SetBufferCursor(context.buffer.Begin());
            return true;
        } else if (mappedCommand == id_JoinLines) {
            // Special case, join on empy line, just pull out the newline
            if (context.bufferCursor.Char() == '\n') {
                context.beginRange = context.bufferCursor;
                context.endRange = context.bufferCursor.PeekByteOffset(1);
                context.op = CommandOperation::Delete;
            } else {
                // Replace the CR (and thus join lines)
                context.beginRange = context.buffer.GetLinePos(context.bufferCursor, LineLocation::LineCRBegin);
                context.endRange = context.buffer.GetLinePos(context.bufferCursor, LineLocation::BeyondLineEnd);

                // Replace all white space (as the J append command does)
                context.tempReg.text = " ";
                context.pRegister = &context.tempReg;
                context.endRange = std::max(context.endRange, buffer.GetLinePos(context.endRange, LineLocation::LineFirstGraphChar));
                context.replaceRangeMode = ReplaceRangeMode::Replace;

                context.op = CommandOperation::Replace;
                context.commandResult.flags = ZSetFlags(context.commandResult.flags, CommandResultFlags::BeginUndoGroup);
            }
        } else if (mappedCommand == id_VisualMode || mappedCommand == id_VisualLineMode) {
            if (m_currentMode == EditorMode::Visual) {
                context.commandResult.modeSwitch = DefaultMode();
            } else {
                if (mappedCommand == id_VisualLineMode) {
                    m_visualBegin = context.buffer.GetLinePos(context.bufferCursor, LineLocation::LineBegin);
                    m_visualEnd = context.buffer.GetLinePos(context.bufferCursor, LineLocation::LineCRBegin);
                } else {
                    m_visualBegin = context.bufferCursor;
                    m_visualEnd = m_visualBegin;
                }
                context.commandResult.modeSwitch = EditorMode::Visual;
            }
            m_lineWise = (mappedCommand == id_VisualLineMode ? true : false);
            return true;
        } else if (mappedCommand == id_Delete) {
            if (m_currentMode == EditorMode::Visual) {
                auto range = GetInclusiveVisualRange();
                if (!range.Valid()) {
                    return true;
                }
                context.beginRange = range.first;
                context.endRange = range.second.Peek(1);
                context.op = CommandOperation::Delete;
                context.commandResult.modeSwitch = DefaultMode();
            } else {
                context.beginRange = cursorItr;
                if (m_currentMode == EditorMode::Normal) {
                    // Normal/Vim mode clamped to end of line
                    context.endRange = cursorItr.PeekLineClamped(context.keymap.TotalCount(), LineLocation::LineCRBegin);
                } else {
                    context.endRange = cursorItr.Peek(context.keymap.TotalCount());
                }
                context.op = CommandOperation::Delete;
                context.commandResult.flags |= CommandResultFlags::HandledCount;
            }
        } else if (mappedCommand == id_OpenLineBelow) {
            context.beginRange = context.buffer.GetLinePos(context.bufferCursor, LineLocation::LineCRBegin);
            context.tempReg.text = "\n";
            context.pRegister = &context.tempReg;
            context.op = CommandOperation::Insert;
            context.commandResult.modeSwitch = EditorMode::Insert;
            context.commandResult.flags = ZSetFlags(context.commandResult.flags, CommandResultFlags::BeginUndoGroup);
        } else if (mappedCommand == id_InsertCarriageReturn) {
            context.beginRange = context.bufferCursor;
            context.tempReg.text = "\n";
            context.pRegister = &context.tempReg;
            context.op = CommandOperation::Insert;
            context.commandResult.modeSwitch = EditorMode::Insert;
            context.commandResult.flags = ZSetFlags(context.commandResult.flags, CommandResultFlags::BeginUndoGroup, shouldGroupInserts);
        } else if (mappedCommand == id_InsertTab) {
            context.beginRange = context.bufferCursor;
            if (buffer.HasFileFlags(FileFlags::InsertTabs)) {
                context.tempReg.text = "\t";
            } else {
                context.tempReg.text = "    ";
            }
            context.pRegister = &context.tempReg;
            context.op = CommandOperation::Insert;
            context.commandResult.modeSwitch = EditorMode::Insert;
            context.commandResult.flags = ZSetFlags(context.commandResult.flags, CommandResultFlags::BeginUndoGroup, shouldGroupInserts);
        } else if (mappedCommand == id_OpenLineAbove) {
            context.beginRange = context.buffer.GetLinePos(context.bufferCursor, LineLocation::LineBegin);
            context.tempReg.text = "\n";
            context.pRegister = &context.tempReg;
            context.op = CommandOperation::Insert;
            context.commandResult.modeSwitch = EditorMode::Insert;
            context.cursorAfterOverride = context.bufferCursor;
            context.commandResult.flags = ZSetFlags(context.commandResult.flags, CommandResultFlags::BeginUndoGroup, shouldGroupInserts);
        } else if (mappedCommand == id_YankLine) {
            // Copy the whole line, including the CR
            context.registers.push('0');
            context.registers.push('*');
            context.registers.push('+');
            context.beginRange = context.buffer.GetLinePos(context.bufferCursor, LineLocation::LineBegin);
            context.endRange = context.buffer.GetLinePos(context.bufferCursor, LineLocation::BeyondLineEnd);
            context.op = CommandOperation::CopyLines;
            context.commandResult.modeSwitch = DefaultMode();
            context.cursorAfterOverride = context.beginRange;
        } else if (mappedCommand == id_Yank) {
            context.registers.push('0');
            context.registers.push('*');
            context.registers.push('+');
            auto range = GetInclusiveVisualRange();
            if (!range.Valid()) {
                return true;
            }
            context.beginRange = range.first;
            context.endRange = range.second.Peek(1);
            // Note: select line wise yank if we started in linewise copy mode
            context.op = m_lineWise ? CommandOperation::CopyLines : CommandOperation::Copy;
            context.commandResult.modeSwitch = DefaultMode();
            context.cursorAfterOverride = context.beginRange;
        } else if (mappedCommand == id_StandardCopy) {
            // Ignore empty copy
            auto range = GetInclusiveVisualRange();
            if (!range.Valid()) {
                return true;
            }
            context.beginRange = range.first;
            context.endRange = range.second.Peek(1);
            if (context.beginRange == context.endRange) {
                return true;
            }

            // Copy in standard mode stays in visual mode
            context.commandResult.modeSwitch = EditorMode::Visual;
            context.registers.push('0');
            context.registers.push('*');
            context.registers.push('+');
            context.cursorAfterOverride = context.bufferCursor;

            // Note: select line wise yank if we started in linewise copy mode
            context.op = CommandOperation::Copy;
        } else if (mappedCommand == id_StandardPaste) {
            GetEditor().ReadClipboard();

            if (context.currentMode == EditorMode::Visual) {
                auto range = GetInclusiveVisualRange();
                if (!range.Valid()) {
                    return true;
                }

                context.replaceRangeMode = ReplaceRangeMode::Replace;
                context.op = CommandOperation::Replace;
                context.pRegister = &GetEditor().GetRegister('"');
                context.beginRange = range.first;
                context.endRange = range.second.Peek(1);
                context.cursorAfterOverride = context.beginRange.PeekByteOffset(long(context.pRegister->text.size()));
                context.commandResult.modeSwitch = EditorMode::Insert;
            } else {
                context.beginRange = context.bufferCursor;
                context.op = CommandOperation::Insert;
            }
            context.commandResult.flags = ZSetFlags(context.commandResult.flags, CommandResultFlags::BeginUndoGroup);
        } else if (mappedCommand == id_PasteAfter) {
            GetEditor().ReadClipboard();

            if (!context.pRegister->text.empty()) {
                // Already in visual mode, so replace the selection
                if (context.currentMode == EditorMode::Visual) {
                    auto range = GetInclusiveVisualRange();
                    if (!range.Valid()) {
                        return true;
                    }
                    context.replaceRangeMode = ReplaceRangeMode::Replace;
                    context.op = CommandOperation::Replace;
                    context.pRegister = &GetEditor().GetRegister('"');
                    context.beginRange = range.first;
                    context.endRange = range.second.Peek(1);
                    context.cursorAfterOverride = context.beginRange.PeekByteOffset(long(context.pRegister->text.size()));
                    context.commandResult.modeSwitch = EditorMode::Insert;
                } else {
                    if (context.pRegister->lineWise) {
                        context.beginRange = context.buffer.GetLinePos(context.bufferCursor, LineLocation::BeyondLineEnd);
                        context.cursorAfterOverride = context.beginRange;
                    } else {
                        context.beginRange = cursorItr.PeekLineClamped(1, LineLocation::LineCRBegin);
                    }
                    context.op = CommandOperation::Insert;
                }
            }
            context.commandResult.flags = ZSetFlags(context.commandResult.flags, CommandResultFlags::BeginUndoGroup);
        } else if (mappedCommand == id_PasteBefore) {
            if (!context.pRegister->text.empty()) {
                // Already in visual mode, so replace the selection with whatever we copied
                if (context.currentMode == EditorMode::Visual) {
                    auto range = GetInclusiveVisualRange();
                    if (!range.Valid()) {
                        return true;
                    }
                    context.pRegister = &GetEditor().GetRegister('"');
                    context.beginRange = range.first;
                    context.endRange = range.second.Peek(1);
                    context.cursorAfterOverride = context.beginRange.PeekByteOffset(long(context.pRegister->text.size()));
                    context.commandResult.modeSwitch = EditorMode::Insert;
                    context.replaceRangeMode = ReplaceRangeMode::Replace;
                    context.op = CommandOperation::Replace;
                } else {
                    if (context.pRegister->lineWise) {
                        context.beginRange = context.buffer.GetLinePos(context.bufferCursor, LineLocation::LineBegin);
                    } else {
                        context.beginRange = context.bufferCursor;
                    }
                    context.op = CommandOperation::Insert;
                }
            }
            context.commandResult.flags = ZSetFlags(context.commandResult.flags, CommandResultFlags::BeginUndoGroup);
        } else if (mappedCommand == id_InsertMode) {
            // TODO: I think this should be a 'command' which would get replayed with dot;
            // instead of special casing it later, we could just insert it into the stream of commands
            // Then undo/redo would replay the insert operation neatly
            context.commandResult.modeSwitch = EditorMode::Insert;
            return true;
        } else if (mappedCommand == id_VisualSelectInnerWORD) {
            if (GetOperationRange("iW", context.currentMode, context.beginRange, context.endRange)) {
                m_visualBegin = context.beginRange;
                GetCurrentWindow()->SetBufferCursor(context.endRange - 1);
                UpdateVisualSelection();
                return true;
            }
            return true;
        } else if (mappedCommand == id_VisualSelectInnerWord) {
            if (GetOperationRange("iw", context.currentMode, context.beginRange, context.endRange)) {
                m_visualBegin = context.beginRange;
                GetCurrentWindow()->SetBufferCursor(context.endRange - 1);
                UpdateVisualSelection();
                return true;
            }
        } else if (mappedCommand == id_VisualSelectAWord) {
            if (GetOperationRange("aw", context.currentMode, context.beginRange, context.endRange)) {
                m_visualBegin = context.beginRange;
                GetCurrentWindow()->SetBufferCursor(context.endRange - 1);
                UpdateVisualSelection();
                return true;
            }
        } else if (mappedCommand == id_VisualSelectAWORD) {
            if (GetOperationRange("aW", context.currentMode, context.beginRange, context.endRange)) {
                m_visualBegin = context.beginRange;
                GetCurrentWindow()->SetBufferCursor(context.endRange - 1);
                UpdateVisualSelection();
                return true;
            }
            return true;
        } else if (mappedCommand == id_DeleteToLineEnd) {
            if (GetOperationRange("$", context.currentMode, context.beginRange, context.endRange)) {
                context.op = CommandOperation::Delete;
            }
        } else if (mappedCommand == id_VisualDelete) {
            // Only in visual mode; delete selected block
            if (GetOperationRange("visual", context.currentMode, context.beginRange, context.endRange)) {
                context.op = CommandOperation::Delete;
                context.commandResult.modeSwitch = DefaultMode();
            }
        } else if (mappedCommand == id_DeleteLine) {
            if (GetOperationRange("line", context.currentMode, context.beginRange, context.endRange)) {
                context.op = CommandOperation::DeleteLines;
                context.commandResult.modeSwitch = DefaultMode();
                context.cursorAfterOverride = context.buffer.GetLinePos(context.beginRange, LineLocation::LineBegin);
            }
        } else if (mappedCommand == id_DeleteWord) {
            if (GetOperationRange("w", context.currentMode, context.beginRange, context.endRange)) {
                context.op = CommandOperation::Delete;
            }
        } else if (mappedCommand == id_DeleteWORD) {
            if (GetOperationRange("W", context.currentMode, context.beginRange, context.endRange)) {
                context.op = CommandOperation::Delete;
            }
        } else if (mappedCommand == id_DeleteAWord) {
            if (GetOperationRange("aw", context.currentMode, context.beginRange, context.endRange)) {
                context.op = CommandOperation::Delete;
            }
        } else if (mappedCommand == id_DeleteAWORD) {
            if (GetOperationRange("aW", context.currentMode, context.beginRange, context.endRange)) {
                context.op = CommandOperation::Delete;
            }
        } else if (mappedCommand == id_DeleteInnerWord) {
            if (GetOperationRange("iw", context.currentMode, context.beginRange, context.endRange)) {
                context.op = CommandOperation::Delete;
            }
        } else if (mappedCommand == id_DeleteInnerWORD) {
            if (GetOperationRange("iW", context.currentMode, context.beginRange, context.endRange)) {
                context.op = CommandOperation::Delete;
            }
        } else if (mappedCommand == id_ChangeToLineEnd) {
            if (GetOperationRange("$", context.currentMode, context.beginRange, context.endRange)) {
                context.op = CommandOperation::Delete;
                context.commandResult.modeSwitch = EditorMode::Insert;
            }
        } else if (mappedCommand == id_VisualChange) {
            // Only in visual mode; delete selected block
            if (GetOperationRange("visual", context.currentMode, context.beginRange, context.endRange)) {
                context.op = CommandOperation::Delete;
                context.commandResult.modeSwitch = EditorMode::Insert;
            }
        } else if (mappedCommand == id_ChangeLine) {
            if (GetOperationRange("line", context.currentMode, context.beginRange, context.endRange)) {
                context.op = CommandOperation::DeleteLines;
                context.commandResult.modeSwitch = EditorMode::Insert;
            }
        } else if (mappedCommand == id_ChangeWord) {
            if (GetOperationRange("cw", context.currentMode, context.beginRange, context.endRange)) {
                context.op = CommandOperation::Delete;
                context.commandResult.modeSwitch = EditorMode::Insert;
            }
        } else if (mappedCommand == id_ChangeWORD) {
            if (GetOperationRange("cW", context.currentMode, context.beginRange, context.endRange)) {
                context.op = CommandOperation::Delete;
                context.commandResult.modeSwitch = EditorMode::Insert;
            }
        } else if (mappedCommand == id_ChangeAWord) {
            if (GetOperationRange("aw", context.currentMode, context.beginRange, context.endRange)) {
                context.op = CommandOperation::Delete;
                context.commandResult.modeSwitch = EditorMode::Insert;
            }
        } else if (mappedCommand == id_ChangeAWORD) {
            if (GetOperationRange("aW", context.currentMode, context.beginRange, context.endRange)) {
                context.op = CommandOperation::Delete;
                context.commandResult.modeSwitch = EditorMode::Insert;
            }
        } else if (mappedCommand == id_ChangeInnerWord) {
            if (GetOperationRange("iw", context.currentMode, context.beginRange, context.endRange)) {
                context.op = CommandOperation::Delete;
                context.commandResult.modeSwitch = EditorMode::Insert;
            }
        } else if (mappedCommand == id_ChangeInnerWORD) {
            if (GetOperationRange("iW", context.currentMode, context.beginRange, context.endRange)) {
                context.op = CommandOperation::Delete;
                context.commandResult.modeSwitch = EditorMode::Insert;
            }
        } else if (mappedCommand == id_ChangeIn) {
            if (!context.keymap.captureChars.empty()) {
                auto range = buffer.FindMatchingPair(bufferCursor, context.keymap.captureChars[0]);
                if (range.first.Valid() && range.second.Valid()) {
                    if ((range.first + 1) == range.second) {
                        // A closed pair (); so insert between them
                        GetCurrentWindow()->SetBufferCursor(range.first + 1);
                        context.commandResult.modeSwitch = EditorMode::Insert;
                        return true;
                    } else {
                        GlyphIterator lineEnd = context.buffer.GetLinePos(range.first, LineLocation::LineCRBegin);
                        if (lineEnd.Valid() && lineEnd < range.second) {
                            GlyphIterator lineStart = context.buffer.GetLinePos(range.first, LineLocation::LineBegin);
                            auto offsetStart = (range.first.Index() - lineStart.Index());

                            // If change in a pair of delimeters that are on seperate lines, then
                            // we remove everything and replace with 2 CRs and an indent based on the start bracket
                            // Since Zep doesn't auto indent, this is the best we can do for now.
                            context.replaceRangeMode = ReplaceRangeMode::Replace;
                            context.op = CommandOperation::Replace;

                            auto offsetText = std::string(offsetStart + 4, ' ');
                            auto offsetBracket = std::string(offsetStart, ' ');
                            context.tempReg.text = std::string("\n") + offsetText + "\n" + offsetBracket;
                            context.pRegister = &context.tempReg;
                            context.beginRange = range.first + 1;
                            context.endRange = range.second;
                            context.cursorAfterOverride = range.first + (long)offsetText.length() + 2;
                            context.commandResult.modeSwitch = EditorMode::Insert;
                        } else {
                            context.beginRange = range.first + 1; // returned range is inclusive
                            context.endRange = range.second;
                            context.op = CommandOperation::Delete;
                            context.commandResult.modeSwitch = EditorMode::Insert;
                        }
                    }
                }
            }
        } else if (mappedCommand == id_SubstituteLine) {
            // Delete whole line and go to insert mode
            context.beginRange = context.buffer.GetLinePos(context.bufferCursor, LineLocation::LineBegin);
            context.endRange = context.buffer.GetLinePos(context.bufferCursor, LineLocation::LineCRBegin);
            context.op = CommandOperation::Delete;
            context.commandResult.modeSwitch = EditorMode::Insert;
        } else if (mappedCommand == id_Substitute) {
            // Only in visual mode; delete selected block and go to insert mode
            // Just delete under m_bufferCursor and insert
            if (GetOperationRange("cursor", context.currentMode, context.beginRange, context.endRange)) {
                context.op = CommandOperation::Delete;
                context.commandResult.modeSwitch = EditorMode::Insert;
            }
        } else if (mappedCommand == id_VisualSubstitute) {
            if (GetOperationRange("visual", context.currentMode, context.beginRange, context.endRange)) {
                context.op = CommandOperation::Delete;
                context.commandResult.modeSwitch = EditorMode::Insert;
            }
        } else if (mappedCommand == id_Find) {
            if (!context.keymap.captureChars.empty()) {
                GetCurrentWindow()->SetBufferCursor(context.buffer.FindOnLineMotion(bufferCursor, (const uint8_t*)&context.keymap.captureChars[0], Direction::Forward));
                m_lastFind = context.keymap.captureChars[0];
                m_lastFindDirection = Direction::Forward;
            }
            return true;
        } else if (mappedCommand == id_FindBackwards) {
            if (!context.keymap.captureChars.empty()) {
                GetCurrentWindow()->SetBufferCursor(context.buffer.FindOnLineMotion(bufferCursor, (const uint8_t*)&context.keymap.captureChars[0], Direction::Backward));
                m_lastFind = context.keymap.captureChars[0];
                m_lastFindDirection = Direction::Backward;
            }
            return true;
        } else if (mappedCommand == id_FindNext) {
            GetCurrentWindow()->SetBufferCursor(context.buffer.FindOnLineMotion(bufferCursor, (const uint8_t*)m_lastFind.c_str(), m_lastFindDirection));
            return true;
        } else if (mappedCommand == id_FindNextDelimiter) {
            auto loc = context.buffer.FindMatchingParen(bufferCursor);
            if (loc.Valid()) {
                GetCurrentWindow()->SetBufferCursor(loc);
                return true;
            }
            return false;
        } else if (mappedCommand == id_Append) {
            // Cursor append
            cursorItr.MoveClamped(1, LineLocation::LineCRBegin);
            GetCurrentWindow()->SetBufferCursor(cursorItr);
            context.commandResult.modeSwitch = EditorMode::Insert;
            return true;
        } else if (mappedCommand == id_AppendToLine) {
            GlyphIterator appendItr = context.buffer.GetLinePos(bufferCursor, LineLocation::LineLastNonCR);
            appendItr.MoveClamped(1, LineLocation::LineCRBegin);
            GetCurrentWindow()->SetBufferCursor(appendItr);
            context.commandResult.modeSwitch = EditorMode::Insert;
            return true;
        } else if (mappedCommand == id_InsertAtFirstChar) {
            GetCurrentWindow()->SetBufferCursor(context.buffer.GetLinePos(bufferCursor, LineLocation::LineFirstGraphChar));
            context.commandResult.modeSwitch = EditorMode::Insert;
            return true;
        } else if (mappedCommand == id_MotionNextFirstChar) {
            GetCurrentWindow()->MoveCursorY(1);
            GetCurrentWindow()->SetBufferCursor(context.buffer.GetLinePos(GetCurrentWindow()->GetBufferCursor(), LineLocation::LineBegin));
            return true;
        } else if (mappedCommand == id_Replace) {
            if (!context.keymap.captureChars.empty()) {
                context.commandResult.flags |= CommandResultFlags::HandledCount;

                if (!bufferCursor.PeekByteOffset(context.keymap.TotalCount()).Valid()) {
                    // Outside the valid buffer; an invalid replace with count!
                    return true;
                }

                context.replaceRangeMode = ReplaceRangeMode::Fill;
                context.op = CommandOperation::Replace;
                context.tempReg.text = context.keymap.captureChars[0];
                context.pRegister = &context.tempReg;

                // Get the range from visual, or use the cursor location
                if (!GetOperationRange("visual", context.currentMode, context.beginRange, context.endRange)) {
                    context.beginRange = cursorItr;
                    context.endRange = cursorItr.PeekLineClamped(context.keymap.TotalCount(), LineLocation::LineCRBegin);
                }

                context.commandResult.modeSwitch = DefaultMode();
            }
        } else if (mappedCommand == id_ChangeToChar) {
            if (!context.keymap.captureChars.empty()) {
                context.beginRange = bufferCursor;
                context.endRange = buffer.FindOnLineMotion(bufferCursor, (const uint8_t*)&context.keymap.captureChars[0], Direction::Forward);
                context.op = CommandOperation::Delete;
                context.commandResult.modeSwitch = EditorMode::Insert;
            }
        } else if (mappedCommand == id_DeleteToChar) {
            if (!context.keymap.captureChars.empty()) {
                context.beginRange = bufferCursor;
                context.endRange = buffer.FindOnLineMotion(bufferCursor, (const uint8_t*)&context.keymap.captureChars[0], Direction::Forward);
                context.op = CommandOperation::Delete;
            }
        } else if (mappedCommand == id_Save) {
            GetEditor().SaveBuffer(GetCurrentWindow()->GetBuffer());
        } else if (m_currentMode == EditorMode::Insert) {
            auto& text = context.keymap.commandWithoutGroups;

            // Detect attempted insertion of a 'special', such as <C-u> which may not have been mapped
            if (text.size() > 3 && text[0] == '<' && text[text.size() - 1] == '>') {
                return false;
            }

            context.beginRange = context.bufferCursor;
            context.tempReg.text = text;
            context.pRegister = &context.tempReg;
            context.op = CommandOperation::Insert;
            context.commandResult.modeSwitch = EditorMode::Insert;
            context.commandResult.flags |= CommandResultFlags::HandledCount;

            // Insert grouping command if necessary
            if (context.fullCommand == " ") {
                context.commandResult.flags = ZSetFlags(context.commandResult.flags, CommandResultFlags::BeginUndoGroup, shouldGroupInserts);
            }
        }

        GetEditor().Broadcast(std::make_shared<ModifyCommandMessage>(context));

        // Update the registers based on context state
        context.UpdateRegisters();

        // Setup command, if any
        if (context.op == CommandOperation::Delete || context.op == CommandOperation::DeleteLines) {
            auto cmd = std::make_shared<ZepCommand_DeleteRange>(
                context.buffer,
                context.beginRange,
                context.endRange,
                context.bufferCursor,
                context.cursorAfterOverride);
            context.commandResult.spCommand = std::static_pointer_cast<ZepCommand>(cmd);
            context.commandResult.flags = ZSetFlags(context.commandResult.flags, CommandResultFlags::BeginUndoGroup);
            return true;
        } else if (context.op == CommandOperation::Insert && !context.pRegister->text.empty()) {
            auto cmd = std::make_shared<ZepCommand_Insert>(
                context.buffer,
                context.beginRange,
                context.pRegister->text,
                context.bufferCursor,
                context.cursorAfterOverride);
            context.commandResult.spCommand = std::static_pointer_cast<ZepCommand>(cmd);
            return true;
        } else if (context.op == CommandOperation::Replace && !context.pRegister->text.empty()) {
            auto cmd = std::make_shared<ZepCommand_ReplaceRange>(
                context.buffer,
                context.replaceRangeMode,
                context.beginRange,
                context.endRange,
                context.pRegister->text,
                context.bufferCursor,
                context.cursorAfterOverride);
            context.commandResult.spCommand = std::static_pointer_cast<ZepCommand>(cmd);
            return true;
        } else if (context.op == CommandOperation::Copy || context.op == CommandOperation::CopyLines) {
            // Put the cursor where the command says it should be
            GetCurrentWindow()->SetBufferCursor(context.cursorAfterOverride);
            return true;
        }

        return false;
    } // namespace Zep

    void ZepMode::ResetCommand() {
        m_currentCommand.clear();
    }

    bool ZepMode::GetOperationRange(const std::string& op, EditorMode currentMode, GlyphIterator& beginRange, GlyphIterator& endRange) const {
        auto& buffer = GetCurrentWindow()->GetBuffer();
        const auto bufferCursor = GetCurrentWindow()->GetBufferCursor();

        beginRange = GlyphIterator();
        if (op == "visual") {
            if (currentMode == EditorMode::Visual) {
                auto range = GetInclusiveVisualRange();
                if (!range.Valid()) {
                    return false;
                }
                beginRange = range.first;
                endRange = range.second.Peek(1);
            }
        }
        // Whole line
        else if (op == "line") {
            beginRange = buffer.GetLinePos(bufferCursor, LineLocation::LineBegin);
            endRange = buffer.GetLinePos(bufferCursor, LineLocation::BeyondLineEnd);

            // Special case; if this is the last line, remove the previous newline in favour of the terminator
            if (endRange.Char() == 0) {
                beginRange.Move(-1);
            }
        } else if (op == "$") {
            beginRange = bufferCursor;
            endRange = buffer.GetLinePos(bufferCursor, LineLocation::LineCRBegin);
        } else if (op == "w") {
            beginRange = bufferCursor;
            endRange = buffer.WordMotion(bufferCursor, SearchType::Word, Direction::Forward);
        } else if (op == "cw") {
            // Change word doesn't extend over the next space
            beginRange = bufferCursor;
            endRange = buffer.ChangeWordMotion(bufferCursor, SearchType::Word, Direction::Forward);
        } else if (op == "cW") {
            beginRange = bufferCursor;
            endRange = buffer.ChangeWordMotion(bufferCursor, SearchType::WORD, Direction::Forward);
        } else if (op == "W") {
            beginRange = bufferCursor;
            endRange = buffer.WordMotion(bufferCursor, SearchType::WORD, Direction::Forward);
        } else if (op == "aw") {
            auto range = buffer.AWordMotion(bufferCursor, SearchType::Word);
            beginRange = range.first;
            endRange = range.second;
        } else if (op == "aW") {
            auto range = buffer.AWordMotion(bufferCursor, SearchType::WORD);
            beginRange = range.first;
            endRange = range.second;
        } else if (op == "iw") {
            auto range = buffer.InnerWordMotion(bufferCursor, SearchType::Word);
            beginRange = range.first;
            endRange = range.second;
        } else if (op == "iW") {
            auto range = buffer.InnerWordMotion(bufferCursor, SearchType::WORD);
            beginRange = range.first;
            endRange = range.second;
        } else if (op == "cursor") {
            auto cursorItr = bufferCursor;
            beginRange = cursorItr;
            endRange = cursorItr.PeekLineClamped(1);
        }
        return beginRange.Valid();
    }

    void ZepMode::UpdateVisualSelection() {
        // Visual mode update - after a command
        if (m_currentMode == EditorMode::Visual) {
            auto pos = GetCurrentWindow()->GetBufferCursor();

            // Update the visual range
            if (m_lineWise) {
                // Record the first selected character position
                if (!m_selectionStartPos.Valid()) {
                    m_selectionStartPos = pos;
                }

                if (m_selectionStartPos > pos) {
                    m_visualBegin = GetCurrentWindow()->GetBuffer().GetLinePos(pos, LineLocation::LineBegin);
                    m_visualEnd = GetCurrentWindow()->GetBuffer().GetLinePos(m_selectionStartPos, LineLocation::LineCRBegin);
                } else {
                    m_visualBegin = GetCurrentWindow()->GetBuffer().GetLinePos(m_selectionStartPos, LineLocation::LineBegin);
                    m_visualEnd = GetCurrentWindow()->GetBuffer().GetLinePos(pos, LineLocation::LineCRBegin);
                }
            } else {
                m_visualEnd = pos;
            }

            auto range = GetInclusiveVisualRange();

            if (range.Valid()) {
                GetCurrentWindow()->GetBuffer().SetSelection(range);
            } else {
                GetCurrentWindow()->GetBuffer().ClearSelection();
            }
        }
    }

    bool ZepMode::HandleExCommand(std::string strCommand) {
        if (strCommand.empty())
            return false;

        auto eraseExtKey = [](std::string& str) {
            auto pos = str.find_last_of('<');
            if (pos != std::string::npos) {
                str.erase(pos, str.size() - pos);
            }
        };

        if (m_lastKey == ExtKeys::BACKSPACE) {
            eraseExtKey(strCommand);

            // Remove the previous character
            if (!strCommand.empty())
                strCommand.pop_back();

            if (strCommand.empty()) {
                GetCurrentWindow()->SetBufferCursor(m_exCommandStartLocation);
                return true;
            }

            m_currentCommand = strCommand;
            return false;
        }

        if (m_lastKey == ExtKeys::ESCAPE) {
            GetCurrentWindow()->SetBufferCursor(m_exCommandStartLocation);
            return true;
        }

        if (m_lastKey == ExtKeys::RETURN) {
            assert(GetCurrentWindow());

            auto& buffer = GetCurrentWindow()->GetBuffer();
            auto bufferCursor = GetCurrentWindow()->GetBufferCursor();

            if (strCommand[0] == '/' || strCommand[0] == '?') {
                // Just exit Ex mode when finished the search
                return true;
            }

            // Remove the return
            eraseExtKey(strCommand);
            if (strCommand == "") {
                return false;
            }

            if (GetEditor().Broadcast(std::make_shared<ZepMessage>(Msg::HandleCommand, strCommand))) {
                return true;
            }

            auto pCommand = GetEditor().FindExCommand(strCommand.substr(1));
            if (pCommand) {
                auto strTok = string_split(strCommand, " ");
                pCommand->Run(strTok);
            } else if (strCommand == ":reg") {
                std::ostringstream str;
                str << "--- Registers ---" << '\n';
                for (auto& reg : GetEditor().GetRegisters()) {
                    if (!reg.second.text.empty()) {
                        std::string displayText = reg.second.text;
                        displayText = string_replace(displayText, "\n", "^J");
                        displayText = string_replace(displayText, "\r", "");
                        str << "\"" << reg.first << "   " << displayText << '\n';
                    }
                }
                GetEditor().SetCommandText(str.str());
            } else if (strCommand == ":map") {
                std::ostringstream str;

                // TODO: CM: this overflows; need to page the output
                str << "--- Mappings ---" << std::endl;
                str << "Normal Maps:" << std::endl;
                keymap_dump(m_normalMap, str);
                str << "Insert Maps:" << std::endl;
                keymap_dump(m_insertMap, str);
                str << "Visual Maps:" << std::endl;
                keymap_dump(m_visualMap, str);

                auto pMapBuffer = GetEditor().GetEmptyBuffer("Mappings");

                pMapBuffer->SetFileFlags(FileFlags::Locked | FileFlags::ReadOnly);
                pMapBuffer->SetText(str.str());
                GetEditor().GetActiveTabWindow()->AddWindow(pMapBuffer, nullptr, RegionLayoutType::VBox);
            } else if (strCommand.find(":tabedit") == 0) {
                auto pTab = GetEditor().AddTabWindow();
                auto strTok = string_split(strCommand, " ");
                if (strTok.size() > 1) {
                    if (strTok[1] == "%") {
                        pTab->AddWindow(&buffer, nullptr, RegionLayoutType::HBox);
                    } else {
                        auto fname = strTok[1];
                        auto pBuffer = GetEditor().GetFileBuffer(fname);
                        pTab->AddWindow(pBuffer, nullptr, RegionLayoutType::HBox);
                    }
                }
                GetEditor().SetCurrentTabWindow(pTab);
            } else if (strCommand.find(":tree") == 0) {
                // Note this is a work in progress; not done yet.
                GetEditor().AddTree();
            } else if (strCommand.find(":vsplit") == 0) {
                auto pTab = GetEditor().GetActiveTabWindow();
                auto strTok = string_split(strCommand, " ");
                if (strTok.size() > 1) {
                    if (strTok[1] == "%") {
                        pTab->AddWindow(&GetCurrentWindow()->GetBuffer(), GetCurrentWindow(), RegionLayoutType::HBox);
                    } else {
                        auto fname = strTok[1];
                        auto pBuffer = GetEditor().GetFileBuffer(fname);
                        pTab->AddWindow(pBuffer, GetCurrentWindow(), RegionLayoutType::HBox);
                    }
                } else {
                    pTab->AddWindow(&GetCurrentWindow()->GetBuffer(), GetCurrentWindow(), RegionLayoutType::HBox);
                }
            } else if (strCommand.find(":hsplit") == 0 || strCommand.find(":split") == 0) {
                auto pTab = GetEditor().GetActiveTabWindow();
                auto strTok = string_split(strCommand, " ");
                if (strTok.size() > 1) {
                    if (strTok[1] == "%") {
                        pTab->AddWindow(&GetCurrentWindow()->GetBuffer(), GetCurrentWindow(), RegionLayoutType::VBox);
                    } else {
                        auto fname = strTok[1];
                        auto pBuffer = GetEditor().GetFileBuffer(fname);
                        pTab->AddWindow(pBuffer, GetCurrentWindow(), RegionLayoutType::VBox);
                    }
                } else {
                    pTab->AddWindow(&GetCurrentWindow()->GetBuffer(), GetCurrentWindow(), RegionLayoutType::VBox);
                }
            } else if (strCommand.find(":e") == 0) {
                auto strTok = string_split(strCommand, " ");
                if (strTok.size() > 1) {
                    auto fname = strTok[1];
                    auto pBuffer = GetEditor().GetFileBuffer(fname);
                    GetCurrentWindow()->SetBuffer(pBuffer);
                }
            } else if (strCommand.find(":w") == 0) {
                auto strTok = string_split(strCommand, " ");
                if (strTok.size() > 1) {
                    auto fname = strTok[1];
                    GetEditor().SaveBufferAs(GetCurrentWindow()->GetBuffer(), fs::path(fname));
                } else {
                    GetEditor().SaveBuffer(GetCurrentWindow()->GetBuffer());
                }
            } else if (strCommand == ":close" || strCommand == ":clo") {
                GetEditor().GetActiveTabWindow()->CloseActiveWindow();
            } else if (strCommand[1] == 'q') {
                if (strCommand == ":q") {
                    GetEditor().GetActiveTabWindow()->CloseActiveWindow();
                }
            } else if (strCommand.find(":ZConfigPath") == 0) {
                GetEditor().SetCommandText(GetEditor().GetFileSystem().GetConfigPath().string());
            } else if (strCommand.find(":ZConfig") == 0) {
                auto pBuffer = GetEditor().GetFileBuffer(GetEditor().GetFileSystem().GetConfigPath() / "zep.cfg");
                GetCurrentWindow()->SetBuffer(pBuffer);
            } else if (strCommand.find(":cd") == 0) {
                GetEditor().SetCommandText(GetEditor().GetFileSystem().GetWorkingDirectory().string());
            } else if (strCommand.find(":ZTestFloatSlider") == 0) {
                // auto line = buffer.GetBufferLine(bufferCursor);
                auto pSlider = std::make_shared<FloatSlider>(GetEditor(), 4);

                auto spMarker = std::make_shared<RangeMarker>(buffer);
                spMarker->SetRange(ByteRange(bufferCursor.Index(), bufferCursor.Peek(1).Index()));
                spMarker->spWidget = pSlider;
                spMarker->markerType = RangeMarkerType::LineWidget;
                spMarker->displayType = RangeMarkerDisplayType::Hidden;
            } else if (strCommand.find(":ZTestColorPicker") == 0) {
                // auto line = buffer.GetBufferLine(bufferCursor);
                auto pPicker = std::make_shared<ColorPicker>(GetEditor());
                auto spMarker = std::make_shared<RangeMarker>(buffer);
                spMarker->SetRange(ByteRange(bufferCursor.Index(), bufferCursor.Peek(1).Index()));
                spMarker->spWidget = pPicker;
                spMarker->markerType = RangeMarkerType::Widget;
                spMarker->displayType = RangeMarkerDisplayType::Background;
            } else if (strCommand.find(":ZTestFlash") == 0) {
                if (buffer.GetSyntax()) {
                    FlashType flashType = FlashType::Flash;
                    float time = 1.0f;
                    auto strTok = string_split(strCommand, " ");
                    if (strTok.size() > 1) {
                        if (std::stoi(strTok[1]) > 0) {
                            flashType = FlashType::Flash;
                        }
                    }
                    if (strTok.size() > 2) {
                        try {
                            time = std::stof(strTok[2]);
                        } catch (std::exception&) {
                        }
                    }
                    buffer.BeginFlash(time, flashType, GlyphRange(buffer.Begin(), buffer.End()));
                }
            } else if (strCommand.find(":ZTestMarkers") == 0) {
                static uint32_t unique = 0;
                int markerSelection = 0;
                auto strTok = string_split(strCommand, " ");
                if (strTok.size() > 1) {
                    markerSelection = std::stoi(strTok[1]);
                }
                auto spMarker = std::make_shared<RangeMarker>(buffer);
                GlyphIterator start;
                GlyphIterator end;

                if (GetCurrentWindow()->GetBuffer().HasSelection()) {
                    // Markers are exclusive
                    auto range = GetCurrentWindow()->GetBuffer().GetInclusiveSelection();
                    start = range.first;
                    end = range.second.Peek(1);
                } else {
                    start = buffer.GetLinePos(bufferCursor, LineLocation::LineFirstGraphChar);
                    end = buffer.GetLinePos(bufferCursor, LineLocation::LineLastGraphChar) + 1;
                }
                spMarker->SetRange(ByteRange(start.Index(), end.Index()));
                switch (markerSelection) {
                case 5:
                    spMarker->SetColors(ThemeColor::Error, ThemeColor::Text, ThemeColor::Error);
                    spMarker->SetName("All Marker");
                    spMarker->SetDescription("This is an example tooltip\nThey can be added to any range of characters");
                    spMarker->displayType = RangeMarkerDisplayType::All;
                    break;
                case 4:
                    spMarker->SetColors(ThemeColor::Error, ThemeColor::Text, ThemeColor::Error);
                    spMarker->SetName("Filled Marker");
                    spMarker->SetDescription("This is an example tooltip\nThey can be added to any range of characters");
                    spMarker->displayType = RangeMarkerDisplayType::Tooltip | RangeMarkerDisplayType::Underline | RangeMarkerDisplayType::Indicator | RangeMarkerDisplayType::Background;
                    break;
                case 3:
                    spMarker->SetColors(ThemeColor::Background, ThemeColor::Text, GetEditor().GetTheme().GetUniqueColor(unique++));
                    spMarker->SetName("Underline Marker");
                    spMarker->SetDescription("This is an example tooltip\nThey can be added to any range of characters");
                    spMarker->displayType = RangeMarkerDisplayType::Tooltip | RangeMarkerDisplayType::Underline | RangeMarkerDisplayType::CursorTip;
                    break;
                case 2:
                    spMarker->SetColors(ThemeColor::Background, ThemeColor::Text, ThemeColor::Warning);
                    spMarker->SetName("Tooltip");
                    spMarker->SetDescription("This is an example tooltip\nThey can be added to any range of characters");
                    spMarker->displayType = RangeMarkerDisplayType::Tooltip;
                    break;
                case 1:
                    spMarker->SetColors(ThemeColor::Background, ThemeColor::Text, ThemeColor::Warning);
                    spMarker->SetName("Warning");
                    spMarker->SetDescription("This is an example warning mark");
                    break;
                case 0:
                default:
                    spMarker->SetColors(ThemeColor::Background, ThemeColor::Text, ThemeColor::Error);
                    spMarker->SetName("Error");
                    spMarker->SetDescription("This is an example error mark");
                }
                SwitchMode(DefaultMode());
            } else if (strCommand == ":ZTabs") {
                buffer.ToggleFileFlag(FileFlags::InsertTabs);
            } else if (strCommand == ":ZShowCR") {
                GetCurrentWindow()->ToggleFlag(WindowFlags::ShowCR);
            } else if (strCommand == ":ZShowLineNumbers") {
                GetCurrentWindow()->ToggleFlag(WindowFlags::ShowLineNumbers);
            } else if (strCommand == ":ZWrapText") {
                // Wrapping is not fully supported yet, but useful for the Orca optional mode.
                // To enable wrapping fully, the editor needs to scroll in X as well as Y...
                GetCurrentWindow()->ToggleFlag(WindowFlags::WrapText);
            } else if (strCommand == ":ZShowIndicators") {
                GetCurrentWindow()->ToggleFlag(WindowFlags::ShowIndicators);
            } else if (strCommand == ":ZShowInput") {
                GetEditor().GetConfig().showNormalModeKeyStrokes = !GetEditor().GetConfig().showNormalModeKeyStrokes;
            } else if (strCommand == ":ZThemeToggle") {
                // An easy test command to check per-buffer themeing
                // Just gets the current theme on the buffer and makes a new
                // one that is opposite
                auto theme = GetCurrentWindow()->GetBuffer().GetTheme();
                auto spNewTheme = std::make_shared<ZepTheme>();
                if (theme.GetThemeType() == ThemeType::Dark) {
                    spNewTheme->SetThemeType(ThemeType::Light);
                } else {
                    spNewTheme->SetThemeType(ThemeType::Dark);
                }
                GetCurrentWindow()->GetBuffer().SetTheme(spNewTheme);
            } else if (strCommand == ":ls") {
                std::ostringstream str;
                str << "--- buffers ---" << '\n';
                int index = 0;
                for (auto& editor_buffer : GetEditor().GetBuffers()) {
                    if (!editor_buffer->GetName().empty()) {
                        std::string displayText = editor_buffer->GetName();
                        displayText = string_replace(displayText, "\n", "^J");
                        if (editor_buffer->IsHidden()) {
                            str << "h";
                        } else {
                            str << " ";
                        }

                        if (&GetCurrentWindow()->GetBuffer() == editor_buffer.get()) {
                            str << "*";
                        } else {
                            str << " ";
                        }

                        if (editor_buffer->HasFileFlags(FileFlags::Dirty)) {
                            str << "+";
                        } else {
                            str << " ";
                        }
                        str << index++ << " : " << displayText << '\n';
                    }
                }
                GetEditor().SetCommandText(str.str());
            } else if (strCommand.find(":bu") == 0) {
                auto strTok = string_split(strCommand, " ");

                if (strTok.size() > 1) {
                    try {
                        auto index = std::stoi(strTok[1]);
                        auto current = 0;
                        for (auto& editor_buffer : GetEditor().GetBuffers()) {
                            if (index == current) {
                                GetCurrentWindow()->SetBuffer(editor_buffer.get());
                            }
                            current++;
                        }
                    } catch (std::exception&) {
                    }
                }
            } else {
                GetEditor().SetCommandText("Not a command");
            }
            return true;
        } else if (!m_currentCommand.empty() && (m_currentCommand[0] == '/' || m_currentCommand[0] == '?')) {
            // TODO: Busy editing the search string; do the search
            if (m_currentCommand.length() > 0) {
                auto pWindow = GetCurrentWindow();
                auto& buffer = pWindow->GetBuffer();
                auto searchString = m_currentCommand.substr(1);

                buffer.ClearRangeMarkers(RangeMarkerType::Search);

                uint32_t numMarkers = 0;
                GlyphIterator start = buffer.Begin();

                if (!searchString.empty()) {
                    static const uint32_t MaxMarkers = 1000;
                    while (numMarkers < MaxMarkers) {
                        auto found = buffer.Find(start, (uint8_t*)&searchString[0], (uint8_t*)&searchString[searchString.length()]);
                        if (!found.Valid()) {
                            break;
                        }

                        start = found + 1;

                        auto spMarker = std::make_shared<RangeMarker>(buffer);
                        spMarker->SetColors(ThemeColor::VisualSelectBackground, ThemeColor::Text);
                        spMarker->SetRange(ByteRange(found.Index(), found.PeekByteOffset(long(searchString.size())).Index()));
                        spMarker->displayType = RangeMarkerDisplayType::Background;
                        spMarker->markerType = RangeMarkerType::Search;

                        numMarkers++;
                    }
                }

                Direction dir = (m_currentCommand[0] == '/') ? Direction::Forward : Direction::Backward;
                m_lastSearchDirection = dir;

                // Find the one on or in front of the cursor, in either direction.
                auto startLocation = m_exCommandStartLocation;
                if (dir == Direction::Forward)
                    startLocation--;
                else
                    startLocation++;

                auto pMark = buffer.FindNextMarker(startLocation, dir, RangeMarkerType::Search);
                if (pMark) {
                    pWindow->SetBufferCursor(GlyphIterator(&buffer, pMark->GetRange().first));
                    pMark->SetBackgroundColor(ThemeColor::Info);
                } else {
                    pWindow->SetBufferCursor(m_exCommandStartLocation);
                }
            }
        }
        return false;
    }

    const KeyMap& ZepMode::GetKeyMappings(EditorMode mode) const {
        if (mode == EditorMode::Visual) {
            return m_visualMap;
        } else if (mode == EditorMode::Normal) {
            return m_normalMap;
        }
        return m_insertMap;
    }

    void ZepMode::AddKeyMapWithCountRegisters(const std::vector<KeyMap*>& maps, const std::vector<std::string>& commands, const StringId& id) {
        for (auto& m : maps) {
            for (auto& c : commands) {
                keymap_add({m}, {"<D><R>" + c}, id);
                keymap_add({m}, {"<R>" + c}, id);
                keymap_add({m}, {"<D>" + c}, id);
                keymap_add({m}, {c}, id);
            }
        }
    }

    CursorType ZepMode::GetCursorType() const {
        switch (m_currentMode) {
        default:
        case EditorMode::None:
        case EditorMode::Ex:
            return CursorType::None;
        case EditorMode::Insert:
            return CursorType::Insert;
        case EditorMode::Normal:
            return CursorType::Normal;
        case EditorMode::Visual:
            return m_visualCursorType;
        }
    }

    void ZepMode::Begin(ZepWindow* pWindow) {
        timer_restart(m_lastKeyPressTimer);

        m_pCurrentWindow = pWindow;

        if (pWindow) {
            ClearSelection();
        }

        // If we are an overlay mode, make sure that the global mode is also begun on the new window
        if (GetEditor().GetGlobalMode() != this) {
            GetEditor().GetGlobalMode()->Begin(pWindow);
        }
    }

} // namespace Zep
