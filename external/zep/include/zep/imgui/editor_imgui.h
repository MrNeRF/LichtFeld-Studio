#pragma once
#include <string>

#include "zep/imgui/display_imgui.h"

#include "zep/editor.h"
#include "zep/mode_standard.h"
#include "zep/mode_vim.h"
#include "zep/syntax.h"
#include "zep/tab_window.h"
#include "zep/window.h"

namespace Zep {

    class ZepDisplay_ImGui;
    class ZepTabWindow;
    class ZepEditor_ImGui : public ZepEditor {
    public:
        ZepEditor_ImGui(const fs::path& root, const NVec2f& pixelScale, uint32_t flags = 0, IZepFileSystem* pFileSystem = nullptr)
            : ZepEditor(new ZepDisplay_ImGui(), root, flags, pFileSystem) {
        }

        void HandleInput() {
            auto& io = ImGui::GetIO();

            bool handled = false;

            uint32_t mod = 0;

            static std::map<int, int> MapUSBKeys = {
                {ImGuiKey_F1, ExtKeys::F1},
                {ImGuiKey_F2, ExtKeys::F2},
                {ImGuiKey_F3, ExtKeys::F3},
                {ImGuiKey_F4, ExtKeys::F4},
                {ImGuiKey_F5, ExtKeys::F5},
                {ImGuiKey_F6, ExtKeys::F6},
                {ImGuiKey_F7, ExtKeys::F7},
                {ImGuiKey_F8, ExtKeys::F8},
                {ImGuiKey_F9, ExtKeys::F9},
                {ImGuiKey_F10, ExtKeys::F10},
                {ImGuiKey_F11, ExtKeys::F11},
                {ImGuiKey_F12, ExtKeys::F12},
                {ImGuiKey_Tab, ExtKeys::TAB},
                {ImGuiKey_Escape, ExtKeys::ESCAPE},
                {ImGuiKey_Enter, ExtKeys::RETURN},
                {ImGuiKey_Delete, ExtKeys::DEL},
                {ImGuiKey_Home, ExtKeys::HOME},
                {ImGuiKey_End, ExtKeys::END},
                {ImGuiKey_Backspace, ExtKeys::BACKSPACE},
                {ImGuiKey_RightArrow, ExtKeys::RIGHT},
                {ImGuiKey_LeftArrow, ExtKeys::LEFT},
                {ImGuiKey_UpArrow, ExtKeys::UP},
                {ImGuiKey_DownArrow, ExtKeys::DOWN},
                {ImGuiKey_PageDown, ExtKeys::PAGEDOWN},
                {ImGuiKey_PageUp, ExtKeys::PAGEUP}};
            static std::map<int, int> MapShiftableUSBKeys = {
                {ImGuiKey_Apostrophe, '\''},
                {ImGuiKey_Comma, ','},
                {ImGuiKey_Minus, '-'},
                {ImGuiKey_Period, '.'},
                {ImGuiKey_Slash, '/'},
                {ImGuiKey_Semicolon, ';'},
                {ImGuiKey_Equal, '='},
                {ImGuiKey_LeftBracket, '['},
                {ImGuiKey_Backslash, '\\'},
                {ImGuiKey_RightBracket, ']'},
                {ImGuiKey_GraveAccent, '`'}};

            if (io.MouseDelta.x != 0 || io.MouseDelta.y != 0) {
                OnMouseMove(toNVec2f(io.MousePos));
            }

            if (io.MouseClicked[0]) {
                if (OnMouseDown(toNVec2f(io.MousePos), ZepMouseButton::Left)) {
                    // Hide the mouse click from imgui if we handled it
                    io.MouseClicked[0] = false;
                }
            }

            if (io.MouseClicked[1]) {
                if (OnMouseDown(toNVec2f(io.MousePos), ZepMouseButton::Right)) {
                    // Hide the mouse click from imgui if we handled it
                    io.MouseClicked[0] = false;
                }
            }

            if (io.MouseReleased[0]) {
                if (OnMouseUp(toNVec2f(io.MousePos), ZepMouseButton::Left)) {
                    // Hide the mouse click from imgui if we handled it
                    io.MouseClicked[0] = false;
                }
            }

            if (io.MouseReleased[1]) {
                if (OnMouseUp(toNVec2f(io.MousePos), ZepMouseButton::Right)) {
                    // Hide the mouse click from imgui if we handled it
                    io.MouseClicked[0] = false;
                }
            }

            if (io.MouseWheel) {
                if (OnMouseWheel(toNVec2f(io.MousePos), io.MouseWheel)) {
                    // Hide the mouse scroll from imgui if we handled it
                    io.MouseWheel = 0;
                }
            }

            if (io.KeyCtrl) {
                mod |= ModifierKey::Ctrl;
            }

            if (io.KeyAlt) {
                mod |= ModifierKey::Alt;
            }

            if (io.KeyShift) {
                mod |= ModifierKey::Shift;
            }

            auto pBuffer = GetActiveBuffer();
            if (!pBuffer) {
                return;
            }

            // Check USB Keys
            for (auto& usbKey : MapUSBKeys) {
                if (ImGui::IsKeyPressed(ImGuiKey(usbKey.first))) {
                    pBuffer->GetMode()->AddKeyPress(usbKey.second, mod);
                    return;
                }
            }

            if (io.KeyCtrl) {
                // Check Shiftable USB Keys
                for (auto& usbKey : MapShiftableUSBKeys) {
                    if (ImGui::IsKeyPressed(ImGuiKey(usbKey.first))) {
                        pBuffer->GetMode()->AddKeyPress(usbKey.second, mod);
                        return;
                    }
                }

                static constexpr std::pair<ImGuiKey, char> ctrlDigits[] = {
                    {ImGuiKey_0, '0'},
                    {ImGuiKey_1, '1'},
                    {ImGuiKey_2, '2'},
                    {ImGuiKey_3, '3'},
                    {ImGuiKey_4, '4'},
                    {ImGuiKey_5, '5'},
                    {ImGuiKey_6, '6'},
                    {ImGuiKey_7, '7'},
                    {ImGuiKey_8, '8'},
                    {ImGuiKey_9, '9'},
                };
                static constexpr std::pair<ImGuiKey, char> ctrlLetters[] = {
                    {ImGuiKey_A, 'a'},
                    {ImGuiKey_B, 'b'},
                    {ImGuiKey_C, 'c'},
                    {ImGuiKey_D, 'd'},
                    {ImGuiKey_E, 'e'},
                    {ImGuiKey_F, 'f'},
                    {ImGuiKey_G, 'g'},
                    {ImGuiKey_H, 'h'},
                    {ImGuiKey_I, 'i'},
                    {ImGuiKey_J, 'j'},
                    {ImGuiKey_K, 'k'},
                    {ImGuiKey_L, 'l'},
                    {ImGuiKey_M, 'm'},
                    {ImGuiKey_N, 'n'},
                    {ImGuiKey_O, 'o'},
                    {ImGuiKey_P, 'p'},
                    {ImGuiKey_Q, 'q'},
                    {ImGuiKey_R, 'r'},
                    {ImGuiKey_S, 's'},
                    {ImGuiKey_T, 't'},
                    {ImGuiKey_U, 'u'},
                    {ImGuiKey_V, 'v'},
                    {ImGuiKey_W, 'w'},
                    {ImGuiKey_X, 'x'},
                    {ImGuiKey_Y, 'y'},
                    {ImGuiKey_Z, 'z'},
                };

                for (auto& ctrlDigit : ctrlDigits) {
                    if (ImGui::IsKeyPressed(ctrlDigit.first)) {
                        pBuffer->GetMode()->AddKeyPress(ctrlDigit.second, mod);
                        handled = true;
                    }
                }
                for (auto& ctrlLetter : ctrlLetters) {
                    if (ImGui::IsKeyPressed(ctrlLetter.first)) {
                        pBuffer->GetMode()->AddKeyPress(ctrlLetter.second, mod);
                        handled = true;
                    }
                }

                if (ImGui::IsKeyPressed(ImGuiKey_Space)) {
                    pBuffer->GetMode()->AddKeyPress(' ', mod);
                    handled = true;
                }
            }

            if (!handled) {
                for (int n = 0; n < io.InputQueueCharacters.Size && io.InputQueueCharacters[n]; n++) {
                    // Ignore '\r' - sometimes ImGui generates it!
                    if (io.InputQueueCharacters[n] == '\r')
                        continue;

                    pBuffer->GetMode()->AddKeyPress(io.InputQueueCharacters[n], mod);
                }
            }
        }

    private:
    };

} // namespace Zep
