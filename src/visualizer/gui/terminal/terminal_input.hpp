/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/terminal/terminal_widget.hpp"
#include "input/frame_input_buffer.hpp"
#include <SDL3/SDL_clipboard.h>
#include <optional>

namespace lfs::vis::terminal {

    inline std::optional<TerminalKey> terminalKeyFromScancode(int scancode) {
        switch (scancode) {
        case SDL_SCANCODE_RETURN:
        case SDL_SCANCODE_KP_ENTER:
            return TerminalKey::Enter;
        case SDL_SCANCODE_BACKSPACE:
            return TerminalKey::Backspace;
        case SDL_SCANCODE_TAB:
            return TerminalKey::Tab;
        case SDL_SCANCODE_ESCAPE:
            return TerminalKey::Escape;
        case SDL_SCANCODE_UP:
            return TerminalKey::Up;
        case SDL_SCANCODE_DOWN:
            return TerminalKey::Down;
        case SDL_SCANCODE_RIGHT:
            return TerminalKey::Right;
        case SDL_SCANCODE_LEFT:
            return TerminalKey::Left;
        case SDL_SCANCODE_HOME:
            return TerminalKey::Home;
        case SDL_SCANCODE_END:
            return TerminalKey::End;
        case SDL_SCANCODE_PAGEUP:
            return TerminalKey::PageUp;
        case SDL_SCANCODE_PAGEDOWN:
            return TerminalKey::PageDown;
        case SDL_SCANCODE_DELETE:
            return TerminalKey::Delete;
        case SDL_SCANCODE_INSERT:
            return TerminalKey::Insert;
        case SDL_SCANCODE_F1:
            return TerminalKey::F1;
        case SDL_SCANCODE_F2:
            return TerminalKey::F2;
        case SDL_SCANCODE_F3:
            return TerminalKey::F3;
        case SDL_SCANCODE_F4:
            return TerminalKey::F4;
        case SDL_SCANCODE_F5:
            return TerminalKey::F5;
        case SDL_SCANCODE_F6:
            return TerminalKey::F6;
        case SDL_SCANCODE_F7:
            return TerminalKey::F7;
        case SDL_SCANCODE_F8:
            return TerminalKey::F8;
        case SDL_SCANCODE_F9:
            return TerminalKey::F9;
        case SDL_SCANCODE_F10:
            return TerminalKey::F10;
        case SDL_SCANCODE_F11:
            return TerminalKey::F11;
        case SDL_SCANCODE_F12:
            return TerminalKey::F12;
        default:
            return std::nullopt;
        }
    }

    inline void processInputEvent(TerminalWidget& terminal, const FrameInputEvent& event) {
        if (terminal.isReadOnly())
            return;
        if (event.kind == FrameInputEventKind::Text) {
            terminal.sendText(event.text);
            return;
        }
        if (event.kind != FrameInputEventKind::KeyDown)
            return;
        const auto sc = event.scancode;
        if ((event.modifiers & SDL_KMOD_CTRL) && sc >= SDL_SCANCODE_A && sc <= SDL_SCANCODE_Z) {
            const char letter = static_cast<char>('A' + (sc - SDL_SCANCODE_A));
            if (letter == 'V' && (event.modifiers & SDL_KMOD_SHIFT)) {
                if (char* text = SDL_GetClipboardText()) {
                    terminal.paste(text);
                    SDL_free(text);
                }
            } else if (letter == 'C' && terminal.hasSelection()) {
                if (!event.repeat)
                    SDL_SetClipboardText(terminal.getSelection().c_str());
            } else {
                terminal.sendControl(letter);
            }
            return;
        }
        if (const auto key = terminalKeyFromScancode(sc))
            terminal.sendKey(*key);
    }

} // namespace lfs::vis::terminal
