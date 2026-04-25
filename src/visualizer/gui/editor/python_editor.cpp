/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "python_editor.hpp"

#include "python_lsp_client.hpp"

#include "gui/editor/zep_rml_display.hpp"
#include "gui/gui_focus_state.hpp"
#include "python/python_buffer_analysis.hpp"
#include "theme/theme.hpp"

#include <RmlUi/Core/Context.h>
#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/ElementDocument.h>
#include <RmlUi/Core/Event.h>
#include <RmlUi/Core/Input.h>
#include <RmlUi/Core/StringUtilities.h>
#include <SDL3/SDL_clipboard.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <format>
#include <memory>
#include <optional>
#include <set>
#include <string_view>
#include <vector>

#include <zep/buffer.h>
#include <zep/commands.h>
#include <zep/editor.h>
#include <zep/mode.h>
#include <zep/mode_standard.h>
#include <zep/mode_vim.h>
#include <zep/syntax_python.h>
#include <zep/tab_window.h>
#include <zep/theme.h>
#include <zep/window.h>

namespace lfs::vis::editor {

    namespace {

        using Clock = std::chrono::steady_clock;
        constexpr auto AUTO_COMPLETION_DEBOUNCE = std::chrono::milliseconds(8);
        constexpr auto ACTIVE_COMPLETION_DEBOUNCE = std::chrono::milliseconds(0);
        constexpr auto SYNTAX_ANALYSIS_DEBOUNCE = std::chrono::milliseconds(160);
        constexpr auto SEMANTIC_TOKENS_WORD_DELAY = std::chrono::milliseconds(800);
        constexpr auto SEMANTIC_TOKENS_BOUNDARY_DELAY = std::chrono::milliseconds(90);
        constexpr int COMPLETION_POPUP_MAX_ITEMS = 8;

        struct EditorColor {
            float x = 0.0f;
            float y = 0.0f;
            float z = 0.0f;
            float w = 1.0f;

            EditorColor() = default;

            EditorColor(float red, float green, float blue, float alpha)
                : x(red), y(green), z(blue), w(alpha) {}

            template <typename Color>
            EditorColor(const Color& color)
                : x(color.x), y(color.y), z(color.z), w(color.w) {}

            template <typename Color>
            EditorColor& operator=(const Color& color) {
                x = color.x;
                y = color.y;
                z = color.z;
                w = color.w;
                return *this;
            }
        };

        Zep::NVec4f to_zep(const EditorColor& color) {
            return {color.x, color.y, color.z, color.w};
        }

        template <typename ColorA, typename ColorB>
        Zep::NVec4f mix(const ColorA& a_in, const ColorB& b_in, float t) {
            const EditorColor a = a_in;
            const EditorColor b = b_in;
            return {
                a.x + (b.x - a.x) * t,
                a.y + (b.y - a.y) * t,
                a.z + (b.z - a.z) * t,
                a.w + (b.w - a.w) * t,
            };
        }

        template <typename ColorA, typename ColorB>
        EditorColor mix_color(const ColorA& a_in, const ColorB& b_in, float t) {
            const EditorColor a = a_in;
            const EditorColor b = b_in;
            return {
                a.x + (b.x - a.x) * t,
                a.y + (b.y - a.y) * t,
                a.z + (b.z - a.z) * t,
                a.w + (b.w - a.w) * t,
            };
        }

        template <typename Color>
        EditorColor with_alpha(const Color& color_in, float alpha) {
            const EditorColor color = color_in;
            return {color.x, color.y, color.z, alpha};
        }

        template <typename Color>
        uint64_t color_to_u32(const Color& color_in) {
            const EditorColor color = color_in;
            const auto to_byte = [](float value) {
                return static_cast<uint64_t>(std::clamp(value, 0.0f, 1.0f) * 255.0f + 0.5f);
            };
            return to_byte(color.x) |
                   (to_byte(color.y) << 8u) |
                   (to_byte(color.z) << 16u) |
                   (to_byte(color.w) << 24u);
        }

        uint32_t zep_modifiers_from_rml(const Rml::Event& event) {
            uint32_t modifiers = 0;
            if (event.GetParameter<int>("ctrl_key", 0) != 0)
                modifiers |= Zep::ModifierKey::Ctrl;
            if (event.GetParameter<int>("shift_key", 0) != 0)
                modifiers |= Zep::ModifierKey::Shift;
            if (event.GetParameter<int>("alt_key", 0) != 0)
                modifiers |= Zep::ModifierKey::Alt;
            return modifiers;
        }

        std::optional<uint32_t> zep_key_from_rml(const Rml::Input::KeyIdentifier key,
                                                 const uint32_t modifiers) {
            switch (key) {
            case Rml::Input::KI_RETURN:
            case Rml::Input::KI_NUMPADENTER:
                return Zep::ExtKeys::RETURN;
            case Rml::Input::KI_BACK:
                return Zep::ExtKeys::BACKSPACE;
            case Rml::Input::KI_TAB:
                return Zep::ExtKeys::TAB;
            case Rml::Input::KI_ESCAPE:
                return Zep::ExtKeys::ESCAPE;
            case Rml::Input::KI_DELETE:
                return Zep::ExtKeys::DEL;
            case Rml::Input::KI_HOME:
                return Zep::ExtKeys::HOME;
            case Rml::Input::KI_END:
                return Zep::ExtKeys::END;
            case Rml::Input::KI_RIGHT:
                return Zep::ExtKeys::RIGHT;
            case Rml::Input::KI_LEFT:
                return Zep::ExtKeys::LEFT;
            case Rml::Input::KI_UP:
                return Zep::ExtKeys::UP;
            case Rml::Input::KI_DOWN:
                return Zep::ExtKeys::DOWN;
            case Rml::Input::KI_NEXT:
                return Zep::ExtKeys::PAGEDOWN;
            case Rml::Input::KI_PRIOR:
                return Zep::ExtKeys::PAGEUP;
            case Rml::Input::KI_INSERT:
                return std::nullopt;
            case Rml::Input::KI_F1:
            case Rml::Input::KI_F2:
            case Rml::Input::KI_F3:
            case Rml::Input::KI_F4:
            case Rml::Input::KI_F5:
            case Rml::Input::KI_F6:
            case Rml::Input::KI_F7:
            case Rml::Input::KI_F8:
            case Rml::Input::KI_F9:
            case Rml::Input::KI_F10:
            case Rml::Input::KI_F11:
            case Rml::Input::KI_F12:
                return Zep::ExtKeys::F1 + (static_cast<int>(key) - static_cast<int>(Rml::Input::KI_F1));
            default:
                break;
            }

            if ((modifiers & Zep::ModifierKey::Ctrl) == 0)
                return std::nullopt;

            if (key >= Rml::Input::KI_A && key <= Rml::Input::KI_Z)
                return static_cast<uint32_t>('a' + (static_cast<int>(key) - static_cast<int>(Rml::Input::KI_A)));
            if (key >= Rml::Input::KI_0 && key <= Rml::Input::KI_9)
                return static_cast<uint32_t>('0' + (static_cast<int>(key) - static_cast<int>(Rml::Input::KI_0)));

            switch (key) {
            case Rml::Input::KI_SPACE:
                return static_cast<uint32_t>(' ');
            case Rml::Input::KI_OEM_1:
                return static_cast<uint32_t>(';');
            case Rml::Input::KI_OEM_PLUS:
                return static_cast<uint32_t>('=');
            case Rml::Input::KI_OEM_COMMA:
                return static_cast<uint32_t>(',');
            case Rml::Input::KI_OEM_MINUS:
                return static_cast<uint32_t>('-');
            case Rml::Input::KI_OEM_PERIOD:
                return static_cast<uint32_t>('.');
            case Rml::Input::KI_OEM_2:
                return static_cast<uint32_t>('/');
            case Rml::Input::KI_OEM_3:
                return static_cast<uint32_t>('`');
            case Rml::Input::KI_OEM_4:
                return static_cast<uint32_t>('[');
            case Rml::Input::KI_OEM_5:
                return static_cast<uint32_t>('\\');
            case Rml::Input::KI_OEM_6:
                return static_cast<uint32_t>(']');
            case Rml::Input::KI_OEM_7:
                return static_cast<uint32_t>('\'');
            default:
                return std::nullopt;
            }
        }

        Zep::ZepMouseButton zep_mouse_button_from_rml(const int button) {
            switch (button) {
            case 0:
                return Zep::ZepMouseButton::Left;
            case 1:
                return Zep::ZepMouseButton::Right;
            case 2:
                return Zep::ZepMouseButton::Middle;
            default:
                return Zep::ZepMouseButton::Unknown;
            }
        }

        bool point_in_rect(const Zep::NRectf& rect, const float x, const float y) {
            return x >= rect.topLeftPx.x && x < rect.bottomRightPx.x &&
                   y >= rect.topLeftPx.y && y < rect.bottomRightPx.y;
        }

        float zep_text_width(Zep::ZepFont& font, const std::string& text) {
            if (text.empty()) {
                return 0.0f;
            }
            const auto* begin = reinterpret_cast<const uint8_t*>(text.data());
            return font.GetTextSize(begin, begin + text.size()).x;
        }

        std::string get_system_clipboard_text() {
            char* clipboard = SDL_GetClipboardText();
            if (clipboard == nullptr) {
                return {};
            }

            std::string text = clipboard;
            SDL_free(clipboard);
            return text;
        }

        void set_system_clipboard_text(const std::string& text) {
            SDL_SetClipboardText(text.c_str());
        }

        std::string rstrip_lines(const std::string& text) {
            std::string result;
            result.reserve(text.size());

            size_t line_start = 0;
            while (line_start < text.size()) {
                const size_t newline = text.find('\n', line_start);
                const bool has_newline = newline != std::string::npos;
                size_t line_end = has_newline ? newline : text.size();

                while (line_end > line_start &&
                       (text[line_end - 1] == ' ' || text[line_end - 1] == '\t')) {
                    --line_end;
                }

                result.append(text, line_start, line_end - line_start);
                if (has_newline) {
                    result.push_back('\n');
                    line_start = newline + 1;
                } else {
                    break;
                }
            }

            return result;
        }

        std::string_view trim_right(std::string_view text) {
            while (!text.empty() &&
                   (text.back() == ' ' || text.back() == '\t' || text.back() == '\r')) {
                text.remove_suffix(1);
            }
            return text;
        }

        bool is_identifier_char(const char ch) {
            const auto byte = static_cast<unsigned char>(ch);
            return std::isalnum(byte) != 0 || ch == '_';
        }

        bool is_semantic_boundary_char(const char ch) {
            switch (ch) {
            case ' ':
            case '\t':
            case '\n':
            case '\r':
            case '.':
            case ':':
            case ',':
            case ';':
            case '(':
            case ')':
            case '[':
            case ']':
            case '{':
            case '}':
            case '=':
            case '+':
            case '-':
            case '*':
            case '/':
                return true;
            default:
                return false;
            }
        }

        char lower_ascii(const char ch) {
            return static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        }

        size_t common_prefix_length_ci(std::string_view lhs, std::string_view rhs) {
            const size_t count = std::min(lhs.size(), rhs.size());
            size_t index = 0;
            while (index < count && lower_ascii(lhs[index]) == lower_ascii(rhs[index])) {
                ++index;
            }
            return index;
        }

        struct CompletionMatchScore {
            bool matched = false;
            int score = 0;
            size_t highlighted_prefix_length = 0;
        };

        CompletionMatchScore score_completion_match(std::string_view candidate,
                                                    std::string_view typed_prefix) {
            if (typed_prefix.empty()) {
                return {.matched = true, .score = 1, .highlighted_prefix_length = 0};
            }

            const size_t prefix_length = common_prefix_length_ci(candidate, typed_prefix);
            if (prefix_length == typed_prefix.size()) {
                return {
                    .matched = true,
                    .score = 400 - static_cast<int>(std::min(candidate.size(), static_cast<size_t>(300))),
                    .highlighted_prefix_length = prefix_length,
                };
            }

            size_t candidate_index = 0;
            int gaps = 0;
            for (const char ch : typed_prefix) {
                const char needle = lower_ascii(ch);
                bool found = false;
                while (candidate_index < candidate.size()) {
                    if (lower_ascii(candidate[candidate_index]) == needle) {
                        ++candidate_index;
                        found = true;
                        break;
                    }
                    ++candidate_index;
                    ++gaps;
                }
                if (!found) {
                    return {};
                }
            }

            return {
                .matched = true,
                .score = 150 - gaps -
                         static_cast<int>(std::min(candidate.size(), static_cast<size_t>(120))),
                .highlighted_prefix_length = prefix_length,
            };
        }

        uint32_t decode_utf8(std::string_view text, size_t& index) {
            if (index >= text.size()) {
                return 0;
            }

            const unsigned char lead = static_cast<unsigned char>(text[index]);
            if (lead < 0x80) {
                ++index;
                return lead;
            }

            auto continuation = [&](size_t offset) -> unsigned char {
                if (index + offset >= text.size()) {
                    return 0;
                }
                return static_cast<unsigned char>(text[index + offset]);
            };

            if ((lead & 0xE0) == 0xC0) {
                const unsigned char b1 = continuation(1);
                if ((b1 & 0xC0) == 0x80) {
                    index += 2;
                    return ((lead & 0x1F) << 6) | (b1 & 0x3F);
                }
            } else if ((lead & 0xF0) == 0xE0) {
                const unsigned char b1 = continuation(1);
                const unsigned char b2 = continuation(2);
                if ((b1 & 0xC0) == 0x80 && (b2 & 0xC0) == 0x80) {
                    index += 3;
                    return ((lead & 0x0F) << 12) | ((b1 & 0x3F) << 6) | (b2 & 0x3F);
                }
            } else if ((lead & 0xF8) == 0xF0) {
                const unsigned char b1 = continuation(1);
                const unsigned char b2 = continuation(2);
                const unsigned char b3 = continuation(3);
                if ((b1 & 0xC0) == 0x80 && (b2 & 0xC0) == 0x80 && (b3 & 0xC0) == 0x80) {
                    index += 4;
                    return ((lead & 0x07) << 18) | ((b1 & 0x3F) << 12) |
                           ((b2 & 0x3F) << 6) | (b3 & 0x3F);
                }
            }

            ++index;
            return lead;
        }

        int utf16_units_between(std::string_view text, const size_t start, const size_t end) {
            int units = 0;
            size_t index = start;
            while (index < end && index < text.size()) {
                const uint32_t codepoint = decode_utf8(text, index);
                units += codepoint > 0xFFFF ? 2 : 1;
            }
            return units;
        }

        size_t line_start_offset(std::string_view text, int line) {
            size_t offset = 0;
            while (line > 0 && offset < text.size()) {
                const size_t newline = text.find('\n', offset);
                if (newline == std::string_view::npos) {
                    return text.size();
                }
                offset = newline + 1;
                --line;
            }
            return offset;
        }

        size_t byte_offset_from_lsp_position(std::string_view text,
                                             const int line,
                                             const int character_utf16) {
            size_t offset = line_start_offset(text, line);
            const size_t line_end = text.find('\n', offset);
            const size_t limit = line_end == std::string_view::npos ? text.size() : line_end;

            int remaining = std::max(character_utf16, 0);
            while (offset < limit && remaining > 0) {
                const size_t before = offset;
                const uint32_t codepoint = decode_utf8(text, offset);
                remaining -= codepoint > 0xFFFF ? 2 : 1;
                if (remaining < 0) {
                    offset = before;
                    break;
                }
            }

            return std::min(offset, limit);
        }

        const char* completion_kind_badge(const int kind) {
            switch (kind) {
            case 2:
                return "M";
            case 3:
                return "F";
            case 4:
                return "C";
            case 5:
                return "F";
            case 6:
                return "V";
            case 7:
                return "T";
            case 8:
                return "I";
            case 9:
                return "M";
            case 10:
                return "P";
            case 11:
                return "U";
            case 12:
                return "V";
            case 14:
                return "K";
            case 17:
                return "F";
            default:
                return "A";
            }
        }

        const char* syntax_symbol_prefix(const lfs::python::PythonSymbolKind kind) {
            switch (kind) {
            case lfs::python::PythonSymbolKind::Function:
                return "def";
            case lfs::python::PythonSymbolKind::Class:
                return "class";
            case lfs::python::PythonSymbolKind::Import:
                return "import";
            case lfs::python::PythonSymbolKind::Variable:
                return "var";
            }
            return "sym";
        }

        std::string range_preview(std::string_view text, const size_t start, const size_t end) {
            if (start >= end || start >= text.size()) {
                return {};
            }

            const size_t clamped_end = std::min(end, text.size());
            std::string preview(text.substr(start, clamped_end - start));
            if (const auto newline = preview.find('\n'); newline != std::string::npos) {
                preview.erase(newline);
            }
            return std::string(trim_right(preview));
        }

        struct CursorLocation {
            size_t byte_index = 0;
            size_t line_start = 0;
            int line = 0;
            int character = 0;
        };

        struct ResolvedEdit {
            size_t start = 0;
            size_t end = 0;
            std::string new_text;
            bool primary = false;
        };

        struct SemanticDirtyRange {
            size_t start = std::string::npos;
            size_t end = 0;

            [[nodiscard]] bool valid() const {
                return start != std::string::npos;
            }

            void include(const size_t range_start, const size_t range_end) {
                const size_t normalized_start = std::min(range_start, range_end);
                const size_t normalized_end = std::max(range_start, range_end);
                if (!valid()) {
                    start = normalized_start;
                    end = normalized_end;
                    return;
                }

                start = std::min(start, normalized_start);
                end = std::max(end, normalized_end);
            }

            void clear() {
                start = std::string::npos;
                end = 0;
            }
        };

        struct SemanticLineRange {
            int start_line = 0;
            int end_line = 0;
            size_t start_byte = 0;
            size_t end_byte = 0;
        };

        struct DiagnosticByteRange {
            size_t start = 0;
            size_t end = 0;
        };

        struct PendingSyntaxPreEdit {
            size_t start_byte = 0;
            size_t end_byte = 0;
            lfs::python::PythonBufferPoint start_point;
            lfs::python::PythonBufferPoint end_point;
        };

    } // namespace

    struct PythonEditor::Impl {
        struct CompletionPopupState {
            struct DisplayItem {
                PythonLspClient::CompletionItem item;
                int score = 0;
                size_t highlighted_prefix_length = 0;
            };

            bool visible = false;
            bool hovered = false;
            int selected_index = 0;
            bool scroll_to_selected = false;
            bool keyboard_navigation_active = false;
            int document_version = 0;
            int line = 0;
            int character = 0;
            size_t replacement_start = 0;
            std::string typed_prefix;
            std::vector<PythonLspClient::CompletionItem> all_items;
            std::vector<DisplayItem> items;

            void clear() {
                visible = false;
                hovered = false;
                selected_index = 0;
                scroll_to_selected = false;
                keyboard_navigation_active = false;
                document_version = 0;
                line = 0;
                character = 0;
                replacement_start = 0;
                typed_prefix.clear();
                all_items.clear();
                items.clear();
            }
        };

        struct SemanticHighlightState {
            int document_version = 0;
            std::vector<PythonLspClient::SemanticToken> tokens;

            void clear() {
                document_version = 0;
                tokens.clear();
            }
        };

        struct Host final : Zep::IZepComponent {
            explicit Host(Impl& owner)
                : owner_(owner) {
            }

            void Notify(std::shared_ptr<Zep::ZepMessage> message) override {
                if (message->messageId == Zep::Msg::GetClipBoard) {
                    message->str = get_system_clipboard_text();
                    message->handled = true;
                    return;
                }

                if (message->messageId == Zep::Msg::SetClipBoard) {
                    set_system_clipboard_text(message->str);
                    message->handled = true;
                    return;
                }

                if (message->messageId != Zep::Msg::Buffer || owner_.suppress_buffer_events ||
                    owner_.buffer == nullptr) {
                    return;
                }

                auto buffer_message = std::static_pointer_cast<Zep::BufferMessage>(message);
                if (buffer_message->pBuffer != owner_.buffer) {
                    return;
                }

                switch (buffer_message->type) {
                case Zep::BufferMessageType::PreBufferChange:
                    owner_.recordSyntaxPreEdit(
                        static_cast<size_t>(std::max(0l, buffer_message->startLocation.Index())),
                        static_cast<size_t>(std::max(0l, buffer_message->endLocation.Index())));
                    break;
                case Zep::BufferMessageType::TextAdded:
                case Zep::BufferMessageType::TextChanged:
                case Zep::BufferMessageType::TextDeleted:
                    owner_.text_changed = true;
                    owner_.noteSemanticDirtyRange(
                        static_cast<size_t>(std::max(0l, buffer_message->startLocation.Index())),
                        static_cast<size_t>(std::max(0l, buffer_message->endLocation.Index())));
                    owner_.recordSyntaxPostEdit(
                        buffer_message->type,
                        static_cast<size_t>(std::max(0l, buffer_message->startLocation.Index())),
                        static_cast<size_t>(std::max(0l, buffer_message->endLocation.Index())));
                    owner_.scheduleSyntaxAnalysis();
                    break;
                default:
                    break;
                }
            }

            Zep::ZepEditor& GetEditor() const override {
                return *owner_.editor;
            }

        private:
            Impl& owner_;
        };

        Impl()
            : editor(std::make_unique<Zep::ZepEditor>(
                  new ZepDisplay_Rml(),
                  std::filesystem::path(PROJECT_ROOT_PATH) / "external" / "zep",
                  Zep::ZepEditorFlags::DisableThreads)),
              host(*this) {
            editor->RegisterCallback(&host);
            editor->GetDisplay().SetPixelScale(Zep::NVec2f(1.0f));
            editor->SetGlobalMode(Zep::ZepMode_Standard::StaticName());

            auto& config = editor->GetConfig();
            config.showLineNumbers = true;
            config.showIndicatorRegion = true;
            config.autoHideCommandRegion = true;
            config.cursorLineSolid = true;
            config.showNormalModeKeyStrokes = false;
            config.shortTabNames = true;
            config.showScrollBar = 2;
            config.lineMargins = Zep::NVec2f(1.0f, 1.0f);

            buffer = editor->InitWithText("script.py", "");
            if (auto* tab = editor->GetActiveTabWindow()) {
                if (auto* window = tab->GetActiveWindow()) {
                    window->SetWindowFlags(window->GetWindowFlags() & ~Zep::WindowFlags::WrapText);
                }
            }
            if (buffer != nullptr) {
                buffer->SetFileFlags(Zep::FileFlags::InsertTabs, false);
                buffer->SetPostKeyNotifier([this](uint32_t key, uint32_t modifier) {
                    handlePostKey(key, modifier);
                    return false;
                });
            }

            applyTheme(theme());
            semantic_highlight_palette_signature = semanticPaletteSignature();
        }

        ~Impl() {
            if (editor != nullptr) {
                editor->UnRegisterCallback(&host);
            }
        }

        void applyTheme(const Theme& app_theme) {
            auto& zep_theme = editor->GetTheme();
            auto& config = editor->GetConfig();
            config.scrollBarSize = app_theme.sizes.scrollbar_size;
            config.scrollBarMinSize = app_theme.sizes.grab_min_size;

            const EditorColor scroll_track = with_alpha(app_theme.palette.background, 0.5f);
            const EditorColor scroll_thumb = with_alpha(app_theme.palette.text_dim, 0.63f);
            const EditorColor scroll_hover = with_alpha(app_theme.palette.primary, 0.78f);
            zep_theme.SetThemeType(app_theme.isLightTheme() ? Zep::ThemeType::Light
                                                            : Zep::ThemeType::Dark);

            zep_theme.SetColor(Zep::ThemeColor::Background, to_zep(app_theme.palette.surface));
            zep_theme.SetColor(Zep::ThemeColor::AirlineBackground,
                               mix(app_theme.palette.surface, app_theme.palette.background, 0.5f));
            zep_theme.SetColor(Zep::ThemeColor::Text, to_zep(app_theme.palette.text));
            zep_theme.SetColor(Zep::ThemeColor::TextDim, to_zep(app_theme.palette.text_dim));
            zep_theme.SetColor(Zep::ThemeColor::Comment,
                               mix(app_theme.palette.text_dim, app_theme.palette.surface_bright, 0.25f));
            zep_theme.SetColor(Zep::ThemeColor::Keyword, to_zep(app_theme.palette.primary));
            zep_theme.SetColor(Zep::ThemeColor::Identifier, to_zep(app_theme.palette.text));
            zep_theme.SetColor(Zep::ThemeColor::Number, to_zep(app_theme.palette.warning));
            zep_theme.SetColor(Zep::ThemeColor::String, to_zep(app_theme.palette.success));
            zep_theme.SetColor(Zep::ThemeColor::Parenthesis, to_zep(app_theme.palette.text));
            zep_theme.SetColor(Zep::ThemeColor::Whitespace, to_zep(app_theme.palette.border));
            zep_theme.SetColor(Zep::ThemeColor::CursorLineBackground,
                               mix(app_theme.palette.surface, app_theme.palette.surface_bright, 0.7f));
            zep_theme.SetColor(Zep::ThemeColor::VisualSelectBackground,
                               mix(app_theme.palette.primary_dim, app_theme.palette.primary, 0.45f));
            zep_theme.SetColor(Zep::ThemeColor::CursorInsert, to_zep(app_theme.palette.text));
            zep_theme.SetColor(Zep::ThemeColor::CursorNormal, to_zep(app_theme.palette.primary));
            zep_theme.SetColor(Zep::ThemeColor::LineNumberBackground,
                               mix(app_theme.palette.surface, app_theme.palette.background, 0.35f));
            zep_theme.SetColor(Zep::ThemeColor::LineNumber, to_zep(app_theme.palette.text_dim));
            zep_theme.SetColor(Zep::ThemeColor::LineNumberActive, to_zep(app_theme.palette.text));
            zep_theme.SetColor(Zep::ThemeColor::TabInactive,
                               mix(app_theme.palette.surface, app_theme.palette.background, 0.55f));
            zep_theme.SetColor(Zep::ThemeColor::TabActive,
                               mix(app_theme.palette.surface_bright, app_theme.palette.primary_dim, 0.2f));
            zep_theme.SetColor(Zep::ThemeColor::TabBorder, to_zep(app_theme.palette.border));
            zep_theme.SetColor(Zep::ThemeColor::WidgetBorder, to_zep(app_theme.palette.border));
            zep_theme.SetColor(Zep::ThemeColor::WidgetBackground, to_zep(scroll_track));
            zep_theme.SetColor(Zep::ThemeColor::WidgetActive, to_zep(scroll_hover));
            zep_theme.SetColor(Zep::ThemeColor::WidgetInactive, to_zep(scroll_thumb));
            zep_theme.SetColor(Zep::ThemeColor::Error, to_zep(app_theme.palette.error));
            zep_theme.SetColor(Zep::ThemeColor::Warning, to_zep(app_theme.palette.warning));
            zep_theme.SetColor(Zep::ThemeColor::Info, to_zep(app_theme.palette.info));
            zep_theme.SetColor(Zep::ThemeColor::FlashColor, to_zep(app_theme.palette.primary));

            editor->RequestRefresh();
        }

        ZepDisplay_Rml& rmlDisplay() {
            return static_cast<ZepDisplay_Rml&>(editor->GetDisplay());
        }

        void syncFontsToRml(const float font_size_px) {
            const float font_size = font_size_px > 0.0f ? font_size_px : 14.0f;
            if (std::abs(font_size - bound_font_size) < 0.5f) {
                return;
            }

            bound_font_size = font_size;
            rmlDisplay().setBaseFontSize(font_size);
            editor->RequestRefresh();
        }

        void ensureLspStarted() {
            if (lsp != nullptr) {
                return;
            }

            lsp = std::make_unique<PythonLspClient>();
            document_version = lsp->updateDocument(getText());
            pending_semantic_tokens.reset();
            semantic_tokens_request_pending = true;
            semantic_full_refresh_required = true;
            semantic_dirty_range.clear();
            last_text_change_at = Clock::now();
            semantic_tokens_idle_delay = SEMANTIC_TOKENS_BOUNDARY_DELAY;
            next_semantic_tokens_request_at = last_text_change_at;
            last_semantic_tokens_requested_version = -1;
        }

        PythonLspClient& ensureLsp() {
            ensureLspStarted();
            return *lsp;
        }

        void clearCompletionState() {
            completion.clear();
            completion_request_pending = false;
            manual_completion_requested = false;
            last_requested_version = -1;
            last_requested_line = -1;
            last_requested_character = -1;
        }

        void setTextSilently(const std::string& text,
                             const std::optional<size_t> cursor_byte_index = std::nullopt) {
            if (buffer == nullptr) {
                return;
            }

            suppress_buffer_events = true;
            buffer->SetText(text);
            suppress_buffer_events = false;
            text_changed = false;
            if (lsp) {
                document_version = lsp->updateDocument(text);
            }
            clearCompletionState();
            clearSyntaxDiagnosticMarkers();
            buffer->ClearFoldRanges();
            pending_syntax_pre_edit.reset();
            pending_syntax_edits.clear();
            syntax_full_reparse_required = true;
            clearSemanticHighlighting();
            semantic_dirty_range.clear();
            semantic_full_refresh_required = true;
            semantic_tokens_request_pending = true;
            last_text_change_at = Clock::now();
            semantic_tokens_idle_delay = SEMANTIC_TOKENS_BOUNDARY_DELAY;
            next_semantic_tokens_request_at = last_text_change_at + semantic_tokens_idle_delay;
            last_semantic_tokens_requested_version = -1;
            pending_semantic_tokens.reset();
            scheduleSyntaxAnalysis(std::chrono::milliseconds(0));

            if (auto* window = editor->GetActiveWindow()) {
                const size_t target = cursor_byte_index.value_or(0);
                window->SetBufferCursor(Zep::GlyphIterator(buffer, static_cast<long>(target)));
            }

            last_cursor_byte_index = std::string::npos;
            editor->RequestRefresh();
        }

        std::string getText() const {
            if (buffer == nullptr) {
                return {};
            }
            return buffer->GetBufferText(buffer->Begin(), buffer->End());
        }

        bool isInsertMode() const {
            return buffer == nullptr || buffer->GetMode() == nullptr ||
                   buffer->GetMode()->GetEditorMode() == Zep::EditorMode::Insert;
        }

        CursorLocation getCursorLocation(const std::string& text) const {
            CursorLocation location;
            if (buffer == nullptr) {
                return location;
            }

            auto* window = editor->GetActiveWindow();
            if (window == nullptr) {
                return location;
            }

            const auto cursor = window->GetBufferCursor();
            if (!cursor.Valid()) {
                return location;
            }

            location.byte_index = std::min(static_cast<size_t>(cursor.Index()), text.size());
            location.line = buffer->GetBufferLine(Zep::GlyphIterator(buffer, static_cast<long>(location.byte_index)));

            const size_t previous_line_break =
                location.byte_index == 0
                    ? std::string::npos
                    : text.rfind('\n', std::min(location.byte_index - 1, text.size() - 1));
            location.line_start =
                previous_line_break == std::string::npos ? 0 : previous_line_break + 1;
            location.character = utf16_units_between(text, location.line_start, location.byte_index);
            return location;
        }

        std::string_view completionMatchText(const PythonLspClient::CompletionItem& item) const {
            if (!item.filter_text.empty()) {
                return item.filter_text;
            }
            if (!item.label.empty()) {
                return item.label;
            }
            return item.insert_text;
        }

        bool isSuppressedCompletionContext(const std::string& text,
                                           const CursorLocation& cursor) const {
            if (buffer == nullptr || cursor.byte_index == 0 || cursor.byte_index > text.size()) {
                return false;
            }

            auto* syntax = buffer->GetSyntax();
            if (syntax == nullptr) {
                return false;
            }

            const auto syntax_index = static_cast<long>(cursor.byte_index - 1);
            const auto syntax_result =
                syntax->GetSyntaxAt(Zep::GlyphIterator(buffer, syntax_index));
            return syntax_result.foreground == Zep::ThemeColor::String ||
                   syntax_result.foreground == Zep::ThemeColor::Comment;
        }

        bool shouldOfferCompletions(const std::string& text, const CursorLocation& cursor) const {
            if (!isInsertMode() || cursor.byte_index == 0 || cursor.byte_index > text.size() ||
                isSuppressedCompletionContext(text, cursor)) {
                return false;
            }

            const char previous = text[cursor.byte_index - 1];
            return previous == '.' || is_identifier_char(previous);
        }

        std::string currentCompletionPrefix(const std::string& text,
                                            const CursorLocation& cursor,
                                            size_t* start_out = nullptr) const {
            const size_t start = fallbackCompletionStart(text, cursor);
            if (start_out != nullptr) {
                *start_out = start;
            }
            if (start >= cursor.byte_index || cursor.byte_index > text.size()) {
                return {};
            }
            return text.substr(start, cursor.byte_index - start);
        }

        bool refilterVisibleCompletion(const std::string& text, const CursorLocation& cursor) {
            if (completion.all_items.empty()) {
                completion.visible = false;
                completion.items.clear();
                completion.typed_prefix.clear();
                return false;
            }

            size_t replacement_start = cursor.byte_index;
            std::string typed_prefix = currentCompletionPrefix(text, cursor, &replacement_start);
            std::string selected_label;
            if (completion.selected_index >= 0 &&
                completion.selected_index < static_cast<int>(completion.items.size())) {
                selected_label = completion.items[completion.selected_index].item.label;
            }

            std::vector<CompletionPopupState::DisplayItem> filtered;
            filtered.reserve(completion.all_items.size());
            for (const auto& item : completion.all_items) {
                const auto match = score_completion_match(completionMatchText(item), typed_prefix);
                if (!match.matched) {
                    continue;
                }

                filtered.push_back({
                    .item = item,
                    .score = match.score,
                    .highlighted_prefix_length = match.highlighted_prefix_length,
                });
            }

            std::stable_sort(filtered.begin(), filtered.end(),
                             [](const CompletionPopupState::DisplayItem& lhs,
                                const CompletionPopupState::DisplayItem& rhs) {
                                 if (lhs.score != rhs.score) {
                                     return lhs.score > rhs.score;
                                 }
                                 if (lhs.item.sort_text != rhs.item.sort_text) {
                                     return lhs.item.sort_text < rhs.item.sort_text;
                                 }
                                 return lhs.item.label < rhs.item.label;
                             });

            completion.visible = !filtered.empty();
            completion.document_version = document_version;
            completion.line = cursor.line;
            completion.character = cursor.character;
            completion.replacement_start = replacement_start;
            completion.typed_prefix = std::move(typed_prefix);
            completion.items = std::move(filtered);

            if (!selected_label.empty()) {
                const auto it =
                    std::find_if(completion.items.begin(), completion.items.end(),
                                 [&](const CompletionPopupState::DisplayItem& item) {
                                     return item.item.label == selected_label;
                                 });
                if (it != completion.items.end()) {
                    completion.selected_index = static_cast<int>(
                        std::distance(completion.items.begin(), it));
                } else {
                    completion.selected_index = 0;
                }
            } else {
                completion.selected_index = 0;
            }

            completion.selected_index =
                std::clamp(completion.selected_index, 0,
                           std::max(static_cast<int>(completion.items.size()) - 1, 0));
            return completion.visible;
        }

        bool isCompatibleCompletionResult(const PythonLspClient::CompletionList& result,
                                          const std::string& text,
                                          const CursorLocation& cursor) const {
            if (result.document_version != document_version || result.line != cursor.line ||
                result.character > cursor.character) {
                return false;
            }

            const size_t result_offset =
                byte_offset_from_lsp_position(text, result.line, result.character);
            if (result_offset > cursor.byte_index || cursor.byte_index > text.size()) {
                return false;
            }

            for (size_t index = result_offset; index < cursor.byte_index; ++index) {
                if (!is_identifier_char(text[index])) {
                    return false;
                }
            }

            return true;
        }

        uint64_t semanticPaletteSignature() const {
            const auto& palette = theme().palette;
            const auto mix_u32 = [](const auto& color) {
                return color_to_u32(color);
            };

            uint64_t signature = 1469598103934665603ull;
            auto fold = [&](const uint64_t value) {
                signature ^= value;
                signature *= 1099511628211ull;
            };

            fold(mix_u32(palette.primary));
            fold(mix_u32(palette.primary_dim));
            fold(mix_u32(palette.secondary));
            fold(mix_u32(palette.info));
            fold(mix_u32(palette.warning));
            fold(mix_u32(palette.success));
            fold(mix_u32(palette.text));
            fold(mix_u32(palette.text_dim));
            return signature;
        }

        void clearSemanticHighlighting() {
            semantic_highlights.clear();
            semantic_highlight_palette_signature = semanticPaletteSignature();
            semantic_highlighting_from_tree_sitter = false;
            semantic_full_refresh_required = true;
            semantic_dirty_range.clear();

            if (buffer == nullptr) {
                return;
            }

            if (auto* syntax = dynamic_cast<Zep::ZepSyntax_Python*>(buffer->GetSyntax())) {
                syntax->ClearSemanticHighlighting();
            }
        }

        void scheduleSyntaxAnalysis(
            const std::chrono::milliseconds delay = SYNTAX_ANALYSIS_DEBOUNCE) {
            syntax_analysis_pending = true;
            next_syntax_analysis_at = Clock::now() + delay;
        }

        void recordSyntaxPreEdit(const size_t start, const size_t end) {
            const std::string text = getText();
            const size_t clamped_start = std::min(start, text.size());
            const size_t clamped_end = std::min(std::max(start, end), text.size());
            pending_syntax_pre_edit = PendingSyntaxPreEdit{
                .start_byte = clamped_start,
                .end_byte = clamped_end,
                .start_point = lfs::python::python_buffer_point_at_byte(text, clamped_start),
                .end_point = lfs::python::python_buffer_point_at_byte(text, clamped_end),
            };
        }

        void recordSyntaxPostEdit(const Zep::BufferMessageType type, const size_t start, const size_t end) {
            const std::string text = getText();
            if (!pending_syntax_pre_edit.has_value()) {
                syntax_full_reparse_required = true;
                return;
            }

            lfs::python::PythonBufferEdit edit;
            edit.start_byte = pending_syntax_pre_edit->start_byte;
            edit.start_point = pending_syntax_pre_edit->start_point;

            switch (type) {
            case Zep::BufferMessageType::TextAdded: {
                const size_t new_end = std::min(std::max(start, end), text.size());
                edit.old_end_byte = pending_syntax_pre_edit->start_byte;
                edit.old_end_point = pending_syntax_pre_edit->start_point;
                edit.new_end_byte = new_end;
                edit.new_end_point = lfs::python::python_buffer_point_at_byte(text, new_end);
                break;
            }
            case Zep::BufferMessageType::TextDeleted: {
                const size_t new_end = std::min(start, text.size());
                edit.old_end_byte = pending_syntax_pre_edit->end_byte;
                edit.old_end_point = pending_syntax_pre_edit->end_point;
                edit.new_end_byte = new_end;
                edit.new_end_point = lfs::python::python_buffer_point_at_byte(text, new_end);
                break;
            }
            case Zep::BufferMessageType::TextChanged: {
                const size_t new_end = std::min(std::max(start, end), text.size());
                edit.old_end_byte = pending_syntax_pre_edit->end_byte;
                edit.old_end_point = pending_syntax_pre_edit->end_point;
                edit.new_end_byte = new_end;
                edit.new_end_point = lfs::python::python_buffer_point_at_byte(text, new_end);
                break;
            }
            default:
                syntax_full_reparse_required = true;
                pending_syntax_pre_edit.reset();
                return;
            }

            pending_syntax_edits.push_back(edit);
            pending_syntax_pre_edit.reset();
        }

        void clearSyntaxDiagnosticMarkers() {
            if (buffer != nullptr && !syntax_markers.empty()) {
                buffer->ClearRangeMarkers(syntax_markers);
            }
            syntax_markers.clear();
        }

        [[nodiscard]] std::optional<DiagnosticByteRange> diagnosticRangeForIssue(
            const lfs::python::PythonBufferIssue& issue,
            std::string_view text) const {
            size_t start = std::min(issue.start_byte, text.size());
            size_t end = std::min(issue.end_byte, text.size());
            if (end < start) {
                std::swap(start, end);
            }

            if (start < end) {
                return DiagnosticByteRange{.start = start, .end = end};
            }

            if (text.empty()) {
                return std::nullopt;
            }

            const size_t line_start = line_start_offset(text, static_cast<int>(issue.line));
            const size_t newline = text.find('\n', line_start);
            const size_t line_end = newline == std::string_view::npos ? text.size() : newline;

            if (line_start < line_end) {
                start = std::clamp(start, line_start, line_end - 1);
                end = start;
                decode_utf8(text, end);
                end = std::clamp(end, start + 1, line_end);
            } else {
                start = std::min(start, text.size() - 1);
                end = start + 1;
            }

            if (start >= end) {
                return std::nullopt;
            }
            return DiagnosticByteRange{.start = start, .end = end};
        }

        void applySyntaxDiagnostics(const std::string& text) {
            clearSyntaxDiagnosticMarkers();
            if (buffer == nullptr || editor == nullptr ||
                syntax_document.analysis().status != lfs::python::PythonBufferStatus::SyntaxError) {
                return;
            }

            for (const auto& issue : syntax_document.analysis().issues) {
                const auto range = diagnosticRangeForIssue(issue, text);
                if (!range.has_value()) {
                    continue;
                }

                auto marker = std::make_shared<Zep::RangeMarker>(*buffer);
                marker->markerType = Zep::RangeMarkerType::Mark;
                marker->displayType = Zep::RangeMarkerDisplayType::Underline |
                                      Zep::RangeMarkerDisplayType::Tooltip |
                                      Zep::RangeMarkerDisplayType::Indicator;
                marker->tipPos = Zep::ToolTipPos::RightLine;
                marker->SetName("Python syntax");
                marker->SetDescription(issue.message);
                marker->SetColors(Zep::ThemeColor::Background, Zep::ThemeColor::Text, Zep::ThemeColor::Error);
                marker->SetAlpha(0.95f);
                marker->SetRange(Zep::ByteRange(static_cast<Zep::ByteIndex>(range->start),
                                                static_cast<Zep::ByteIndex>(range->end)));
                syntax_markers.insert(std::move(marker));
            }

            editor->RequestRefresh();
        }

        void syncSyntaxFolds(const std::string& text) {
            if (buffer == nullptr) {
                return;
            }

            if (text.empty() || !syntax_document.hasTree()) {
                buffer->ClearFoldRanges();
                return;
            }

            std::vector<Zep::FoldRange> folds;
            folds.reserve(syntax_document.foldRanges().size());
            for (const auto& fold : syntax_document.foldRanges()) {
                if (fold.end_byte > text.size() || fold.start_byte >= fold.end_byte ||
                    fold.end_line <= fold.line) {
                    continue;
                }

                folds.push_back(Zep::FoldRange{
                    .range = Zep::ByteRange(static_cast<Zep::ByteIndex>(fold.start_byte),
                                            static_cast<Zep::ByteIndex>(fold.end_byte)),
                    .startLine = static_cast<long>(fold.line),
                    .endLine = static_cast<long>(fold.end_line),
                    .kind = fold.kind,
                    .collapsed = false,
                });
            }

            buffer->SetFoldRanges(std::move(folds));
        }

        void updateSyntaxDiagnostics(const std::string& text, const CursorLocation& cursor) {
            current_syntax_scope = syntax_document.scopeAt(cursor.byte_index);
            if (!syntax_analysis_pending || Clock::now() < next_syntax_analysis_at) {
                return;
            }

            syntax_analysis_pending = false;
            if (syntax_full_reparse_required || !syntax_document.hasTree()) {
                syntax_document.reset(text);
            } else {
                syntax_document.applyEditsAndReparse(text, pending_syntax_edits);
            }
            pending_syntax_edits.clear();
            pending_syntax_pre_edit.reset();
            syntax_full_reparse_required = false;
            current_syntax_scope = syntax_document.scopeAt(cursor.byte_index);
            applySyntaxDiagnostics(text);
            syncSyntaxFolds(text);
            if (shouldUseTreeSitterHighlightingFallback()) {
                applyTreeSitterHighlightingFallback(text);
            }
        }

        std::optional<Zep::ZepSemanticHighlight> mapSemanticToken(
            const PythonLspClient::SemanticToken& token,
            const std::string& text) const {
            if (token.length <= 0) {
                return std::nullopt;
            }

            const size_t start = byte_offset_from_lsp_position(text, token.line, token.start_character);
            const size_t end =
                byte_offset_from_lsp_position(text, token.line, token.start_character + token.length);
            if (start >= end || end > text.size()) {
                return std::nullopt;
            }

            const auto& palette = theme().palette;
            EditorColor color = palette.text;
            bool use_custom = true;
            bool underline = (token.modifiers & (1u << 4)) != 0;

            if (token.type == "class" || token.type == "type" || token.type == "enum" ||
                token.type == "interface" || token.type == "struct" ||
                token.type == "typeParameter") {
                color = palette.info;
            } else if (token.type == "function" || token.type == "method") {
                color = mix_color(palette.primary, palette.info, 0.35f);
            } else if (token.type == "decorator") {
                color = mix_color(palette.primary, palette.secondary, 0.25f);
            } else if (token.type == "namespace" || token.type == "module") {
                color = palette.secondary;
            } else if (token.type == "property" || token.type == "enumMember") {
                color = mix_color(palette.secondary, palette.text, 0.35f);
            } else if (token.type == "parameter") {
                color = mix_color(palette.warning, palette.text, 0.25f);
            } else if (token.type == "keyword") {
                color = palette.primary;
            } else if (token.type == "comment") {
                color = palette.text_dim;
            } else if (token.type == "string") {
                color = palette.success;
            } else if (token.type == "number") {
                color = palette.warning;
            } else {
                return std::nullopt;
            }

            return Zep::ZepSemanticHighlight{
                .start = static_cast<long>(start),
                .end = static_cast<long>(end),
                .foreground = Zep::ThemeColor::Custom,
                .custom_foreground = use_custom,
                .custom_foreground_color = to_zep(color),
                .underline = underline,
            };
        }

        [[nodiscard]] bool shouldUseTreeSitterHighlightingFallback() const {
            return lsp == nullptr || !lsp->isAvailable();
        }

        [[nodiscard]] Zep::NVec4f treeSitterHighlightColor(
            const lfs::python::PythonHighlightKind kind) const {
            const auto& palette = theme().palette;
            switch (kind) {
            case lfs::python::PythonHighlightKind::Keyword:
                return to_zep(palette.primary);
            case lfs::python::PythonHighlightKind::Comment:
                return to_zep(palette.text_dim);
            case lfs::python::PythonHighlightKind::String:
                return to_zep(palette.success);
            case lfs::python::PythonHighlightKind::Number:
            case lfs::python::PythonHighlightKind::Constant:
                return to_zep(palette.warning);
            case lfs::python::PythonHighlightKind::Decorator:
                return to_zep(mix_color(palette.primary, palette.secondary, 0.25f));
            case lfs::python::PythonHighlightKind::Function:
                return to_zep(mix_color(palette.primary, palette.info, 0.35f));
            case lfs::python::PythonHighlightKind::Type:
                return to_zep(palette.info);
            case lfs::python::PythonHighlightKind::Property:
                return to_zep(mix_color(palette.secondary, palette.text, 0.35f));
            }
            return to_zep(palette.text);
        }

        void applyTreeSitterHighlightingFallback(const std::string& text) {
            if (buffer == nullptr) {
                return;
            }

            auto* syntax = dynamic_cast<Zep::ZepSyntax_Python*>(buffer->GetSyntax());
            if (syntax == nullptr) {
                return;
            }

            std::vector<Zep::ZepSemanticHighlight> highlights;
            highlights.reserve(syntax_document.highlights().size());
            for (const auto& highlight : syntax_document.highlights()) {
                if (highlight.start_byte >= highlight.end_byte || highlight.end_byte > text.size()) {
                    continue;
                }

                highlights.push_back(Zep::ZepSemanticHighlight{
                    .start = static_cast<long>(highlight.start_byte),
                    .end = static_cast<long>(highlight.end_byte),
                    .foreground = Zep::ThemeColor::Custom,
                    .custom_foreground = true,
                    .custom_foreground_color = treeSitterHighlightColor(highlight.kind),
                    .underline = false,
                });
            }

            syntax->SetSemanticHighlighting(highlights);
            semantic_highlighting_from_tree_sitter = true;
            semantic_highlight_palette_signature = semanticPaletteSignature();
            semantic_full_refresh_required = false;
        }

        void applySemanticHighlighting(const std::string& text) {
            if (buffer == nullptr) {
                return;
            }

            auto* syntax = dynamic_cast<Zep::ZepSyntax_Python*>(buffer->GetSyntax());
            if (syntax == nullptr) {
                return;
            }

            std::vector<Zep::ZepSemanticHighlight> highlights;
            highlights.reserve(semantic_highlights.tokens.size());
            for (const auto& token : semantic_highlights.tokens) {
                if (auto mapped = mapSemanticToken(token, text)) {
                    highlights.push_back(std::move(*mapped));
                }
            }

            syntax->SetSemanticHighlighting(highlights);
            semantic_highlighting_from_tree_sitter = false;
            semantic_highlight_palette_signature = semanticPaletteSignature();
            semantic_full_refresh_required = false;
        }

        void applySemanticHighlightingRange(const std::string& text,
                                            const SemanticLineRange& range) {
            if (buffer == nullptr || range.start_byte >= range.end_byte) {
                return;
            }

            auto* syntax = dynamic_cast<Zep::ZepSyntax_Python*>(buffer->GetSyntax());
            if (syntax == nullptr) {
                return;
            }

            std::vector<Zep::ZepSemanticHighlight> highlights;
            for (const auto& token : semantic_highlights.tokens) {
                if (token.line < range.start_line || token.line > range.end_line) {
                    continue;
                }
                if (auto mapped = mapSemanticToken(token, text)) {
                    highlights.push_back(std::move(*mapped));
                }
            }

            syntax->ReplaceSemanticHighlighting(static_cast<long>(range.start_byte),
                                                static_cast<long>(range.end_byte), highlights);
            semantic_highlighting_from_tree_sitter = false;
            semantic_highlight_palette_signature = semanticPaletteSignature();
            semantic_full_refresh_required = false;
        }

        void scheduleSemanticTokens() {
            semantic_tokens_request_pending = true;
            next_semantic_tokens_request_at = last_text_change_at + semantic_tokens_idle_delay;
        }

        std::chrono::milliseconds semanticTokenIdleDelay(const std::string& text,
                                                         const CursorLocation& cursor) const {
            if (cursor.byte_index == 0 || cursor.byte_index > text.size()) {
                return SEMANTIC_TOKENS_BOUNDARY_DELAY;
            }

            const char previous = text[cursor.byte_index - 1];
            if (is_semantic_boundary_char(previous) || !is_identifier_char(previous)) {
                return SEMANTIC_TOKENS_BOUNDARY_DELAY;
            }

            return SEMANTIC_TOKENS_WORD_DELAY;
        }

        void noteSemanticDirtyRange(const size_t start, const size_t end) {
            semantic_dirty_range.include(start, end);
        }

        [[nodiscard]] int lineFromByteOffset(std::string_view text, size_t offset) const {
            offset = std::min(offset, text.size());
            int line = 0;
            for (size_t index = 0; index < offset; ++index) {
                if (text[index] == '\n') {
                    ++line;
                }
            }
            return line;
        }

        [[nodiscard]] size_t lineEndOffset(std::string_view text, const int line) const {
            const size_t start = line_start_offset(text, std::max(line, 0));
            const size_t newline = text.find('\n', start);
            return newline == std::string_view::npos ? text.size() : newline + 1;
        }

        [[nodiscard]] SemanticLineRange semanticDirtyLineRange(const std::string& text) const {
            if (!semantic_dirty_range.valid()) {
                const int last_line = lineFromByteOffset(text, text.size());
                return {
                    .start_line = 0,
                    .end_line = last_line,
                    .start_byte = 0,
                    .end_byte = text.size(),
                };
            }

            const size_t dirty_start = std::min(semantic_dirty_range.start, text.size());
            const size_t dirty_end = std::min(std::max(semantic_dirty_range.start, semantic_dirty_range.end),
                                              text.size());

            int start_line = lineFromByteOffset(text, dirty_start);
            int end_line = lineFromByteOffset(text, dirty_end);
            start_line = std::max(0, start_line - 1);
            end_line += 1;

            return {
                .start_line = start_line,
                .end_line = end_line,
                .start_byte = line_start_offset(text, start_line),
                .end_byte = lineEndOffset(text, end_line),
            };
        }

        void issueSemanticTokensRequest() {
            if (!lsp || !lsp->isAvailable()) {
                semantic_tokens_request_pending = false;
                return;
            }

            if (last_semantic_tokens_requested_version == document_version) {
                semantic_tokens_request_pending = false;
                return;
            }

            lsp->requestSemanticTokens(document_version);
            last_semantic_tokens_requested_version = document_version;
            semantic_tokens_request_pending = false;
        }

        void scheduleAutoCompletion(const std::string& text, const CursorLocation& cursor) {
            if (!shouldOfferCompletions(text, cursor)) {
                completion.clear();
                completion_request_pending = false;
                return;
            }

            completion_request_pending = true;
            const char previous = text[cursor.byte_index - 1];
            next_completion_request_at =
                Clock::now() + (previous == '.' || !completion.all_items.empty()
                                    ? ACTIVE_COMPLETION_DEBOUNCE
                                    : AUTO_COMPLETION_DEBOUNCE);
        }

        void issueCompletionRequest(const std::string& text,
                                    const CursorLocation& cursor,
                                    const bool manual) {
            if (!lsp || !lsp->isAvailable() || !isInsertMode()) {
                completion_request_pending = false;
                return;
            }

            if (!manual && last_requested_version == document_version &&
                last_requested_line == cursor.line &&
                last_requested_character == cursor.character) {
                completion_request_pending = false;
                return;
            }

            std::string trigger_character;
            if (!manual && cursor.byte_index > 0) {
                const char previous = text[cursor.byte_index - 1];
                if (previous == '.') {
                    trigger_character = ".";
                }
            }

            lsp->requestCompletion(document_version, cursor.line, cursor.character, manual,
                                   std::move(trigger_character));
            last_requested_version = document_version;
            last_requested_line = cursor.line;
            last_requested_character = cursor.character;
            completion_request_pending = false;
        }

        void updateLanguageServerState(const std::string& text, const CursorLocation& cursor) {
            const bool cursor_changed =
                cursor.byte_index != last_cursor_byte_index || cursor.line != last_cursor_line ||
                cursor.character != last_cursor_character;

            if (text_changed) {
                last_text_change_at = Clock::now();
                document_version = ensureLsp().updateDocument(text);
                if (!completion.all_items.empty()) {
                    refilterVisibleCompletion(text, cursor);
                }
                pending_semantic_tokens.reset();
                if (!semantic_dirty_range.valid()) {
                    noteSemanticDirtyRange(cursor.line_start, cursor.byte_index);
                }
                semantic_tokens_idle_delay = semanticTokenIdleDelay(text, cursor);
                scheduleAutoCompletion(text, cursor);
                scheduleSemanticTokens();
            } else if (cursor_changed && !shouldOfferCompletions(text, cursor)) {
                completion.clear();
            } else if (cursor_changed && !completion.all_items.empty()) {
                refilterVisibleCompletion(text, cursor);
                scheduleAutoCompletion(text, cursor);
            } else if (cursor_changed && completion.visible) {
                scheduleAutoCompletion(text, cursor);
            }

            if (manual_completion_requested) {
                issueCompletionRequest(text, cursor, true);
                manual_completion_requested = false;
            } else if (completion_request_pending && Clock::now() >= next_completion_request_at) {
                issueCompletionRequest(text, cursor, false);
            }

            if (semantic_tokens_request_pending && !completion_request_pending &&
                Clock::now() >= next_semantic_tokens_request_at) {
                issueSemanticTokensRequest();
            }

            if (lsp == nullptr) {
                last_cursor_byte_index = cursor.byte_index;
                last_cursor_line = cursor.line;
                last_cursor_character = cursor.character;
                return;
            }

            if (auto result = lsp->takeLatestCompletion()) {
                if (isCompatibleCompletionResult(*result, text, cursor)) {
                    completion.all_items = std::move(result->items);
                    refilterVisibleCompletion(text, cursor);
                } else if (result->document_version == document_version &&
                           result->line == cursor.line &&
                           result->character == cursor.character) {
                    completion.clear();
                }
            }

            if (auto semantic_tokens = lsp->takeLatestSemanticTokens()) {
                if (semantic_tokens->document_version == document_version) {
                    pending_semantic_tokens = std::move(*semantic_tokens);
                }
            }

            if (pending_semantic_tokens.has_value() &&
                pending_semantic_tokens->document_version == document_version &&
                !completion_request_pending &&
                Clock::now() >= last_text_change_at + semantic_tokens_idle_delay) {
                semantic_highlights.document_version = pending_semantic_tokens->document_version;
                semantic_highlights.tokens = std::move(pending_semantic_tokens->tokens);
                pending_semantic_tokens.reset();
                if (semantic_full_refresh_required || !semantic_dirty_range.valid()) {
                    applySemanticHighlighting(text);
                } else {
                    applySemanticHighlightingRange(text, semanticDirtyLineRange(text));
                }
                semantic_dirty_range.clear();
            }

            if (shouldUseTreeSitterHighlightingFallback() &&
                !syntax_analysis_pending &&
                (!semantic_highlighting_from_tree_sitter || text_changed ||
                 semantic_highlight_palette_signature != semanticPaletteSignature())) {
                applyTreeSitterHighlightingFallback(text);
            }

            last_cursor_byte_index = cursor.byte_index;
            last_cursor_line = cursor.line;
            last_cursor_character = cursor.character;
        }

        std::optional<ResolvedEdit> resolveTextEdit(const PythonLspClient::TextEdit& edit,
                                                    const std::string& text,
                                                    const bool primary) const {
            const size_t start = byte_offset_from_lsp_position(text, edit.start_line, edit.start_character);
            const size_t end = byte_offset_from_lsp_position(text, edit.end_line, edit.end_character);
            if (start > end || end > text.size()) {
                return std::nullopt;
            }

            return ResolvedEdit{
                .start = start,
                .end = end,
                .new_text = edit.new_text,
                .primary = primary,
            };
        }

        size_t fallbackCompletionStart(const std::string& text, const CursorLocation& cursor) const {
            size_t start = cursor.byte_index;
            while (start > cursor.line_start && is_identifier_char(text[start - 1])) {
                --start;
            }
            return start;
        }

        bool applyCompletion(const std::string& text,
                             const CursorLocation& cursor,
                             const PythonLspClient::CompletionItem& item) {
            if (buffer == nullptr) {
                return false;
            }

            std::vector<ResolvedEdit> edits;
            edits.reserve(item.additional_text_edits.size() + 1);

            for (const auto& additional : item.additional_text_edits) {
                if (auto resolved = resolveTextEdit(additional, text, false)) {
                    edits.push_back(std::move(*resolved));
                }
            }

            if (item.text_edit.has_value()) {
                if (auto resolved = resolveTextEdit(*item.text_edit, text, true)) {
                    edits.push_back(std::move(*resolved));
                }
            } else {
                edits.push_back({
                    .start = fallbackCompletionStart(text, cursor),
                    .end = cursor.byte_index,
                    .new_text = item.insert_text.empty() ? item.label : item.insert_text,
                    .primary = true,
                });
            }

            std::ranges::sort(edits, [](const ResolvedEdit& lhs, const ResolvedEdit& rhs) {
                if (lhs.start != rhs.start) {
                    return lhs.start < rhs.start;
                }
                return lhs.end < rhs.end;
            });

            size_t cursor_offset = cursor.byte_index;
            size_t validated_until = 0;
            bool primary_seen = false;
            long size_delta = 0;
            for (const auto& edit : edits) {
                if (edit.start < validated_until || edit.end > text.size()) {
                    return false;
                }

                if (edit.primary) {
                    cursor_offset = static_cast<size_t>(std::max<long>(
                        0, static_cast<long>(edit.start) + size_delta +
                               static_cast<long>(edit.new_text.size())));
                    primary_seen = true;
                }
                size_delta += static_cast<long>(edit.new_text.size()) -
                              static_cast<long>(edit.end - edit.start);
                validated_until = edit.end;
            }

            if (!primary_seen) {
                cursor_offset = static_cast<size_t>(
                    std::max<long>(0, static_cast<long>(text.size()) + size_delta));
            }

            bool changed = false;
            for (auto it = edits.rbegin(); it != edits.rend(); ++it) {
                const auto& edit = *it;
                const auto start = Zep::GlyphIterator(buffer, static_cast<long>(edit.start));
                const auto end = Zep::GlyphIterator(buffer, static_cast<long>(edit.end));

                if (edit.start != edit.end) {
                    Zep::ChangeRecord delete_record;
                    if (!buffer->Delete(start, end, delete_record)) {
                        return false;
                    }
                    changed = true;
                }

                if (!edit.new_text.empty()) {
                    Zep::ChangeRecord insert_record;
                    if (!buffer->Insert(Zep::GlyphIterator(buffer, static_cast<long>(edit.start)),
                                        edit.new_text, insert_record)) {
                        return false;
                    }
                    changed = true;
                }
            }

            if (auto* window = editor->GetActiveWindow()) {
                const size_t clamped_cursor = std::min(cursor_offset, getText().size());
                window->SetBufferCursor(
                    Zep::GlyphIterator(buffer, static_cast<long>(clamped_cursor)));
            }

            if (changed) {
                text_changed = true;
                editor->RequestRefresh();
            }

            request_focus = true;
            force_unfocused = false;
            clearCompletionState();
            return true;
        }

        bool handleCompletionKey(const Rml::Input::KeyIdentifier key,
                                 const uint32_t modifiers,
                                 const std::string& text,
                                 const CursorLocation& cursor) {
            if (read_only) {
                return false;
            }

            if ((modifiers & Zep::ModifierKey::Ctrl) != 0 && key == Rml::Input::KI_SPACE) {
                manual_completion_requested = true;
                return true;
            }

            if (!completion.visible || completion.items.empty()) {
                return false;
            }

            switch (key) {
            case Rml::Input::KI_ESCAPE:
                completion.clear();
                return true;
            case Rml::Input::KI_DOWN:
                completion.selected_index =
                    (completion.selected_index + 1) % static_cast<int>(completion.items.size());
                completion.scroll_to_selected = true;
                completion.keyboard_navigation_active = true;
                return true;
            case Rml::Input::KI_UP:
                completion.selected_index =
                    (completion.selected_index + static_cast<int>(completion.items.size()) - 1) %
                    static_cast<int>(completion.items.size());
                completion.scroll_to_selected = true;
                completion.keyboard_navigation_active = true;
                return true;
            case Rml::Input::KI_NEXT:
                completion.selected_index = std::min(
                    completion.selected_index + COMPLETION_POPUP_MAX_ITEMS,
                    static_cast<int>(completion.items.size()) - 1);
                completion.scroll_to_selected = true;
                completion.keyboard_navigation_active = true;
                return true;
            case Rml::Input::KI_PRIOR:
                completion.selected_index =
                    std::max(completion.selected_index - COMPLETION_POPUP_MAX_ITEMS, 0);
                completion.scroll_to_selected = true;
                completion.keyboard_navigation_active = true;
                return true;
            case Rml::Input::KI_TAB:
                applyCompletion(text, cursor, completion.items[completion.selected_index].item);
                return true;
            case Rml::Input::KI_RETURN:
            case Rml::Input::KI_NUMPADENTER:
                if ((modifiers & Zep::ModifierKey::Ctrl) == 0) {
                    applyCompletion(text, cursor, completion.items[completion.selected_index].item);
                    return true;
                }
                return false;
            default:
                return false;
            }
        }

        float completionAnchorX(const std::string& text,
                                const CursorLocation& cursor,
                                const Zep::NRectf& cursor_rect) const {
            float anchor_x = cursor_rect.topLeftPx.x;
            const size_t line_end = text.find('\n', cursor.line_start);
            const size_t clamped_line_end =
                line_end == std::string::npos ? text.size() : line_end;
            if (cursor.byte_index >= clamped_line_end) {
                anchor_x -= std::max(cursor_rect.Width(), 1.0f);
            }
            return anchor_x;
        }

        std::string completionInsertPreviewText(const PythonLspClient::CompletionItem& item) const {
            std::string_view inserted = item.label;
            if (item.text_edit.has_value() && !item.text_edit->new_text.empty()) {
                inserted = item.text_edit->new_text;
            } else if (!item.insert_text.empty()) {
                inserted = item.insert_text;
            }

            std::string preview;
            preview.reserve(std::min<size_t>(inserted.size(), 48));
            for (const char ch : inserted) {
                if (ch == '\n' || ch == '\r' || ch == '\t') {
                    if (preview.empty() || preview.back() != ' ') {
                        preview.push_back(' ');
                    }
                } else {
                    preview.push_back(ch);
                }

                if (preview.size() >= 48) {
                    break;
                }
            }

            if (inserted.size() > preview.size()) {
                preview += "...";
            }
            return preview;
        }

        bool handleCompletionMouseDown(const std::string& text,
                                       const CursorLocation& cursor,
                                       const float x,
                                       const float y,
                                       const int button) {
            if (!completion.visible || completion.items.empty() ||
                !point_in_rect(completion_popup_rect, x, y)) {
                return false;
            }

            completion.hovered = true;
            completion.keyboard_navigation_active = false;
            if (button != 0) {
                return true;
            }

            const int row = static_cast<int>(
                std::floor((y - completion_popup_rect.topLeftPx.y - 6.0f) /
                           std::max(completion_popup_row_height, 1.0f)));
            const int index = completion_popup_first_index + row;
            if (index >= 0 && index < static_cast<int>(completion.items.size())) {
                completion.selected_index = index;
                completion.scroll_to_selected = false;
                applyCompletion(text, cursor, completion.items[index].item);
            }
            return true;
        }

        void renderCompletionPopup(const std::string& text,
                                   const CursorLocation& cursor,
                                   const float editor_width,
                                   const float editor_height) {
            completion.hovered = false;
            completion_popup_rect = {};
            completion_popup_first_index = 0;
            completion_popup_row_height = 0.0f;
            if (!completion.visible || completion.items.empty()) {
                return;
            }

            auto* window = editor->GetActiveWindow();
            if (window == nullptr) {
                completion.clear();
                return;
            }

            auto& display = rmlDisplay();
            auto& ui_font = display.GetFont(Zep::ZepTextType::UI);
            auto& text_font = display.GetFont(Zep::ZepTextType::Text);
            const auto& palette = theme().palette;

            const auto cursor_rect = window->GetCursorRect();
            const float row_height = std::max(18.0f, static_cast<float>(ui_font.GetPixelHeight()) + 6.0f);
            const int visible_count =
                std::min<int>(static_cast<int>(completion.items.size()), COMPLETION_POPUP_MAX_ITEMS);
            float desired_width = 160.0f;
            for (int i = 0; i < visible_count; ++i) {
                const auto& item = completion.items[static_cast<size_t>(i)].item;
                const std::string detail = item.detail.empty() ? item.description : item.detail;
                const float label_width = zep_text_width(text_font, item.label);
                const float detail_width = detail.empty() ? 0.0f : zep_text_width(ui_font, detail);
                desired_width = std::max(desired_width,
                                         48.0f + label_width +
                                             (detail_width > 0.0f ? detail_width + 36.0f : 18.0f));
            }
            const float popup_width = std::clamp(
                desired_width,
                160.0f,
                std::min(440.0f, std::max(160.0f, editor_width - 12.0f)));
            const float popup_height = std::min(
                visible_count * row_height + 12.0f,
                std::max(row_height + 12.0f, editor_height - 12.0f));

            float popup_x = completionAnchorX(text, cursor, cursor_rect);
            float popup_y = cursor_rect.bottomRightPx.y + 4.0f;
            if (popup_y + popup_height > editor_height - 6.0f) {
                popup_y = std::max(6.0f, cursor_rect.topLeftPx.y - popup_height - 4.0f);
            }
            popup_x = std::clamp(popup_x, 6.0f, std::max(6.0f, editor_width - popup_width - 6.0f));
            popup_y = std::clamp(popup_y, 6.0f, std::max(6.0f, editor_height - popup_height - 6.0f));

            completion_popup_rect = Zep::NRectf(popup_x, popup_y, popup_width, popup_height);
            completion_popup_row_height = row_height;
            completion.hovered = mouse_pos_valid &&
                                 point_in_rect(completion_popup_rect, mouse_x, mouse_y);
            if (completion.hovered) {
                gui::guiFocusState().want_capture_mouse = true;
            }

            const int item_count = static_cast<int>(completion.items.size());
            int first_index = std::clamp(
                completion.selected_index - visible_count / 2,
                0,
                std::max(0, item_count - visible_count));
            if (!completion.scroll_to_selected) {
                first_index = std::clamp(
                    completion_popup_first_index,
                    0,
                    std::max(0, item_count - visible_count));
            }
            completion_popup_first_index = first_index;
            completion.scroll_to_selected = false;

            display.DrawRectFilled(completion_popup_rect, to_zep(palette.background));
            const auto border = to_zep(palette.border);
            display.DrawRectFilled(Zep::NRectf(popup_x, popup_y, popup_width, 1.0f), border);
            display.DrawRectFilled(Zep::NRectf(popup_x, popup_y + popup_height - 1.0f, popup_width, 1.0f), border);
            display.DrawRectFilled(Zep::NRectf(popup_x, popup_y, 1.0f, popup_height), border);
            display.DrawRectFilled(Zep::NRectf(popup_x + popup_width - 1.0f, popup_y, 1.0f, popup_height), border);

            auto draw_text = [&](Zep::ZepFont& font,
                                 const float x,
                                 const float y,
                                 const Zep::NVec4f& color,
                                 std::string_view value) {
                if (value.empty()) {
                    return;
                }
                const auto* begin = reinterpret_cast<const uint8_t*>(value.data());
                display.DrawChars(font, {x, y}, color, begin, begin + value.size());
            };

            const float row_left = popup_x + 1.0f;
            const float row_right = popup_x + popup_width - 1.0f;
            for (int visible = 0; visible < visible_count; ++visible) {
                const int index = first_index + visible;
                const auto& entry = completion.items[index];
                const auto& item = entry.item;
                const bool active_row = index == completion.selected_index;
                const float row_y = popup_y + 6.0f + visible * row_height;
                const Zep::NRectf row_rect(row_left, row_y, row_right - row_left, row_height);
                const bool hovered_row = mouse_pos_valid && point_in_rect(row_rect, mouse_x, mouse_y);

                if (hovered_row && !completion.keyboard_navigation_active) {
                    completion.selected_index = index;
                }

                if (active_row || hovered_row) {
                    EditorColor fill = palette.primary;
                    fill.w = active_row ? 0.32f : 0.18f;
                    display.DrawRectFilled(row_rect, to_zep(fill));
                }

                const auto active_foreground = palette.text;
                const Zep::NVec4f badge_color =
                    to_zep(active_row ? active_foreground : palette.text_dim);
                const Zep::NVec4f text_color =
                    to_zep(active_row ? active_foreground
                                      : (item.deprecated ? palette.text_dim : palette.text));
                const Zep::NVec4f highlight_color =
                    to_zep(active_row ? active_foreground : palette.primary);
                const Zep::NVec4f detail_color =
                    to_zep(active_row ? active_foreground : palette.text_dim);
                const float text_y = row_y + 3.0f;

                draw_text(ui_font, row_left + 9.0f, text_y, badge_color,
                          completion_kind_badge(item.kind));

                const float label_x = row_left + 30.0f;
                const size_t label_highlight =
                    common_prefix_length_ci(item.label, completion.typed_prefix);
                if (label_highlight > 0 && label_highlight < item.label.size()) {
                    const std::string highlight_text = item.label.substr(0, label_highlight);
                    draw_text(text_font, label_x, text_y, highlight_color, highlight_text);
                    draw_text(text_font,
                              label_x + zep_text_width(text_font, highlight_text),
                              text_y,
                              text_color,
                              std::string_view(item.label).substr(label_highlight));
                } else {
                    draw_text(text_font, label_x, text_y,
                              label_highlight > 0 ? highlight_color : text_color,
                              item.label);
                }

                std::string detail = item.detail.empty() ? item.description : item.detail;
                const std::string insert_preview = completionInsertPreviewText(item);
                if (active_row && !insert_preview.empty() && insert_preview != item.label) {
                    detail = "-> " + insert_preview;
                }

                if (!detail.empty()) {
                    const float detail_width = zep_text_width(ui_font, detail);
                    const float detail_x = row_right - detail_width - 10.0f;
                    if (detail_x > label_x + 150.0f) {
                        draw_text(ui_font, detail_x, text_y, detail_color, detail);
                    }
                }
            }
        }

        void handlePostKey(uint32_t key, uint32_t modifier) {
            if (buffer == nullptr || (modifier & (Zep::ModifierKey::Ctrl | Zep::ModifierKey::Alt)) != 0 ||
                key != Zep::ExtKeys::RETURN) {
                return;
            }

            auto* window = editor->GetActiveWindow();
            if (window == nullptr) {
                return;
            }

            const auto cursor = window->GetBufferCursor();
            if (!cursor.Valid() || cursor.Index() <= 0) {
                return;
            }

            const std::string text = getText();
            size_t cursor_index = static_cast<size_t>(cursor.Index());
            cursor_index = std::min(cursor_index, text.size());
            if (cursor_index == 0 || text[cursor_index - 1] != '\n') {
                return;
            }

            const size_t previous_line_end = cursor_index - 1;
            const size_t previous_line_start =
                previous_line_end == 0 ? std::string::npos
                                       : text.rfind('\n', previous_line_end - 1);
            const size_t line_start =
                previous_line_start == std::string::npos ? 0 : previous_line_start + 1;

            std::string_view previous_line(text.data() + line_start, previous_line_end - line_start);
            const std::string_view trimmed_line = trim_right(previous_line);

            std::string indent;
            for (const char ch : previous_line) {
                if (ch == ' ' || ch == '\t') {
                    indent.push_back(ch);
                    continue;
                }
                break;
            }

            if (!trimmed_line.empty() && trimmed_line.back() == ':') {
                indent.append(4, ' ');
            }

            if (indent.empty() || buffer->GetMode() == nullptr) {
                return;
            }

            for (const char ch : indent) {
                if (ch == '\t') {
                    buffer->GetMode()->AddKeyPress(Zep::ExtKeys::TAB, 0);
                } else {
                    buffer->GetMode()->AddKeyPress(static_cast<uint32_t>(ch), 0);
                }
            }
        }

        void setVimModeEnabled(const bool enabled) {
            vim_mode_enabled = enabled;
            editor->SetGlobalMode(enabled ? Zep::ZepMode_Vim::StaticName()
                                          : Zep::ZepMode_Standard::StaticName());
            editor->RequestRefresh();
        }

        std::optional<Zep::GlyphRange> currentSelectionRange() const {
            if (buffer == nullptr || !buffer->HasSelection()) {
                return std::nullopt;
            }

            auto range = buffer->GetInclusiveSelection();
            if (!range.first.Valid() || !range.second.Valid()) {
                return std::nullopt;
            }

            if (range.second < range.first) {
                std::swap(range.first, range.second);
            }

            return range;
        }

        Zep::GlyphIterator selectionEndExclusive(const Zep::GlyphRange& range) const {
            auto end = range.second;
            if (buffer == nullptr) {
                return end;
            }

            const auto buffer_end = buffer->End();
            if (end < buffer_end) {
                end = end.Peek(1);
            }
            return end;
        }

        bool hasEditableSelection() const {
            const auto range = currentSelectionRange();
            if (!range.has_value()) {
                return false;
            }

            const auto end = selectionEndExclusive(*range);
            return range->first.Valid() && end.Valid() && range->first < end;
        }

        void syncRegistersToClipboard(const std::string& text) {
            if (editor == nullptr) {
                set_system_clipboard_text(text);
                return;
            }

            editor->SetRegister('"', Zep::Register(text));
            editor->SetRegister('*', Zep::Register(text));
            editor->SetRegister('+', Zep::Register(text));
        }

        void focusEditor() {
            request_focus = true;
            force_unfocused = false;
        }

        void copySelectionToClipboard() {
            if (buffer == nullptr) {
                return;
            }

            const auto range = currentSelectionRange();
            if (!range.has_value()) {
                return;
            }

            const auto end = selectionEndExclusive(*range);
            if (!(range->first < end)) {
                return;
            }

            syncRegistersToClipboard(buffer->GetBufferText(range->first, end));
        }

        void clearSelectionAndReturnToDefaultMode() {
            if (buffer != nullptr) {
                buffer->ClearSelection();
            }
            if (auto* mode = buffer != nullptr ? buffer->GetMode() : nullptr) {
                mode->SwitchMode(mode->DefaultMode());
            }
            if (editor != nullptr) {
                editor->RequestRefresh();
            }
        }

        [[nodiscard]] std::optional<lfs::python::PythonByteRange> currentSelectionByteRange() const {
            const auto selection = currentSelectionRange();
            if (!selection.has_value()) {
                return std::nullopt;
            }

            const auto end = selectionEndExclusive(*selection);
            if (!selection->first.Valid() || !end.Valid() || !(selection->first < end)) {
                return std::nullopt;
            }

            return lfs::python::PythonByteRange{
                .start_byte = static_cast<size_t>(std::max(0l, selection->first.Index())),
                .end_byte = static_cast<size_t>(std::max(0l, end.Index())),
            };
        }

        bool ensureSyntaxDocumentCurrent(const std::string& text, const size_t cursor_byte) {
            if (!syntax_analysis_pending && syntax_document.hasTree()) {
                current_syntax_scope = syntax_document.scopeAt(cursor_byte);
                return true;
            }

            if (!syntax_full_reparse_required && syntax_document.hasTree()) {
                syntax_document.applyEditsAndReparse(text, pending_syntax_edits);
            } else {
                syntax_document.reset(text);
            }
            pending_syntax_edits.clear();
            pending_syntax_pre_edit.reset();
            syntax_full_reparse_required = false;
            syntax_analysis_pending = false;
            current_syntax_scope = syntax_document.scopeAt(cursor_byte);
            applySyntaxDiagnostics(text);
            syncSyntaxFolds(text);
            if (shouldUseTreeSitterHighlightingFallback()) {
                applyTreeSitterHighlightingFallback(text);
            }
            return syntax_document.hasTree();
        }

        bool moveCursorToByte(const size_t byte_offset) {
            if (buffer == nullptr || editor == nullptr) {
                return false;
            }

            auto* window = editor->GetActiveWindow();
            if (window == nullptr) {
                return false;
            }

            const std::string text = getText();
            const auto cursor =
                Zep::GlyphIterator(buffer, static_cast<long>(std::min(byte_offset, text.size())));
            if (!cursor.Valid()) {
                return false;
            }

            buffer->ClearSelection();
            if (auto* mode = buffer->GetMode()) {
                mode->SwitchMode(mode->DefaultMode());
            }
            window->SetBufferCursor(cursor);
            current_syntax_scope = syntax_document.scopeAt(std::min(byte_offset, text.size()));
            editor->RequestRefresh();
            focusEditor();
            return true;
        }

        bool selectSyntaxByteRange(const lfs::python::PythonByteRange& range) {
            if (buffer == nullptr || editor == nullptr || range.start_byte >= range.end_byte) {
                return false;
            }

            auto* window = editor->GetActiveWindow();
            if (window == nullptr) {
                return false;
            }

            const std::string text = getText();
            if (range.end_byte > text.size()) {
                return false;
            }

            const auto begin = Zep::GlyphIterator(buffer, static_cast<long>(range.start_byte));
            const auto end = Zep::GlyphIterator(buffer, static_cast<long>(range.end_byte));
            if (!begin.Valid() || !end.Valid() || !(begin < end)) {
                return false;
            }

            buffer->SetSelection(Zep::GlyphRange(begin, end));
            window->SetBufferCursor(end);
            if (auto* mode = buffer->GetMode()) {
                mode->SwitchMode(Zep::EditorMode::Visual);
            }
            current_syntax_scope = syntax_document.scopeAt(range.start_byte);
            editor->RequestRefresh();
            focusEditor();
            return true;
        }

        void cutSelectionToClipboard() {
            if (read_only || buffer == nullptr || editor == nullptr) {
                return;
            }

            const auto range = currentSelectionRange();
            if (!range.has_value()) {
                return;
            }

            const auto end = selectionEndExclusive(*range);
            if (!(range->first < end)) {
                return;
            }

            copySelectionToClipboard();

            auto* window = editor->GetActiveWindow();
            auto* mode = buffer->GetMode();
            if (window == nullptr || mode == nullptr) {
                return;
            }

            const auto cursor_before = window->GetBufferCursor();
            mode->AddCommand(std::make_shared<Zep::ZepCommand_GroupMarker>(*buffer));
            mode->AddCommand(std::make_shared<Zep::ZepCommand_DeleteRange>(
                *buffer, range->first, end, cursor_before, range->first));

            clearSelectionAndReturnToDefaultMode();
            focusEditor();
        }

        void pasteFromClipboard() {
            if (read_only || buffer == nullptr || editor == nullptr) {
                return;
            }

            const std::string text = get_system_clipboard_text();
            if (text.empty()) {
                return;
            }

            syncRegistersToClipboard(text);

            auto* window = editor->GetActiveWindow();
            auto* mode = buffer->GetMode();
            if (window == nullptr || mode == nullptr) {
                return;
            }

            const auto cursor_before = window->GetBufferCursor();
            mode->AddCommand(std::make_shared<Zep::ZepCommand_GroupMarker>(*buffer));

            const auto range = currentSelectionRange();
            if (range.has_value()) {
                const auto end = selectionEndExclusive(*range);
                mode->AddCommand(std::make_shared<Zep::ZepCommand_ReplaceRange>(
                    *buffer,
                    Zep::ReplaceRangeMode::Replace,
                    range->first,
                    end,
                    text,
                    cursor_before,
                    range->first.PeekByteOffset(static_cast<long>(text.size()))));
            } else {
                mode->AddCommand(std::make_shared<Zep::ZepCommand_Insert>(
                    *buffer,
                    cursor_before,
                    text,
                    cursor_before,
                    cursor_before.PeekByteOffset(static_cast<long>(text.size()))));
            }

            clearSelectionAndReturnToDefaultMode();
            focusEditor();
        }

        void selectAll() {
            if (buffer == nullptr || editor == nullptr) {
                return;
            }

            const auto begin = buffer->Begin();
            const auto end = buffer->End();
            if (!begin.Valid() || !end.Valid() || !(begin < end)) {
                return;
            }

            buffer->SetSelection(Zep::GlyphRange(begin, end));
            if (auto* window = editor->GetActiveWindow()) {
                window->SetBufferCursor(end);
            }
            if (auto* mode = buffer->GetMode()) {
                mode->SwitchMode(Zep::EditorMode::Visual);
            }
            editor->RequestRefresh();
            focusEditor();
        }

        bool selectEnclosingSyntaxBlock() {
            if (buffer == nullptr || editor == nullptr) {
                return false;
            }

            const std::string text = getText();
            const CursorLocation cursor = getCursorLocation(text);
            if (!ensureSyntaxDocumentCurrent(text, cursor.byte_index)) {
                return false;
            }

            const auto range = syntax_document.enclosingBlockRange(cursor.byte_index);
            if (!range.has_value()) {
                return false;
            }

            return selectSyntaxByteRange(*range);
        }

        [[nodiscard]] std::optional<lfs::python::PythonByteRange> innermostFoldRangeAt(
            const size_t byte_offset) const {
            std::optional<lfs::python::PythonByteRange> best;
            for (const auto& fold : syntax_document.foldRanges()) {
                if (fold.start_byte > byte_offset || byte_offset > fold.end_byte) {
                    continue;
                }
                if (!best.has_value() ||
                    (fold.end_byte - fold.start_byte) < (best->end_byte - best->start_byte)) {
                    best = lfs::python::PythonByteRange{
                        .start_byte = fold.start_byte,
                        .end_byte = fold.end_byte,
                    };
                }
            }
            return best;
        }

        bool expandSyntaxSelection() {
            if (buffer == nullptr || editor == nullptr) {
                return false;
            }

            const std::string text = getText();
            const CursorLocation cursor = getCursorLocation(text);
            if (!ensureSyntaxDocumentCurrent(text, cursor.byte_index)) {
                return false;
            }

            const auto selection = currentSelectionByteRange();
            if (!selection.has_value()) {
                return selectEnclosingSyntaxBlock();
            }

            const auto ranges = syntax_document.enclosingBlockRanges(selection->start_byte);
            for (const auto& range : ranges) {
                if (range.start_byte <= selection->start_byte && range.end_byte >= selection->end_byte &&
                    (range.start_byte < selection->start_byte || range.end_byte > selection->end_byte)) {
                    return selectSyntaxByteRange(range);
                }
            }

            return false;
        }

        bool selectCurrentSyntaxFold() {
            if (buffer == nullptr || editor == nullptr) {
                return false;
            }

            const std::string text = getText();
            const CursorLocation cursor = getCursorLocation(text);
            if (!ensureSyntaxDocumentCurrent(text, cursor.byte_index)) {
                return false;
            }

            const auto range = innermostFoldRangeAt(cursor.byte_index);
            if (!range.has_value()) {
                return false;
            }

            return selectSyntaxByteRange(*range);
        }

        bool toggleCurrentSyntaxFold() {
            if (buffer == nullptr || editor == nullptr) {
                return false;
            }

            const std::string text = getText();
            const CursorLocation cursor = getCursorLocation(text);
            if (!ensureSyntaxDocumentCurrent(text, cursor.byte_index)) {
                return false;
            }

            const auto range = innermostFoldRangeAt(cursor.byte_index);
            if (!range.has_value()) {
                return false;
            }

            const bool changed =
                buffer->ToggleFoldAtByte(static_cast<Zep::ByteIndex>(range->start_byte));
            if (changed) {
                editor->RequestRefresh();
                focusEditor();
            }
            return changed;
        }

        bool foldAllSyntaxBlocks() {
            if (buffer == nullptr || editor == nullptr) {
                return false;
            }

            const std::string text = getText();
            const CursorLocation cursor = getCursorLocation(text);
            if (!ensureSyntaxDocumentCurrent(text, cursor.byte_index)) {
                return false;
            }

            const bool changed = buffer->SetAllFoldsCollapsed(true);
            if (changed) {
                editor->RequestRefresh();
                focusEditor();
            }
            return changed;
        }

        bool unfoldAllSyntaxBlocks() {
            if (buffer == nullptr || editor == nullptr) {
                return false;
            }

            const bool changed = buffer->SetAllFoldsCollapsed(false);
            if (changed) {
                editor->RequestRefresh();
                focusEditor();
            }
            return changed;
        }

        bool jumpToParentSyntaxBlock() {
            if (buffer == nullptr || editor == nullptr) {
                return false;
            }

            const std::string text = getText();
            const CursorLocation cursor = getCursorLocation(text);
            if (!ensureSyntaxDocumentCurrent(text, cursor.byte_index)) {
                return false;
            }

            const auto ranges = syntax_document.enclosingBlockRanges(cursor.byte_index);
            if (ranges.empty()) {
                return false;
            }

            const auto& target = ranges.front().start_byte == cursor.byte_index && ranges.size() > 1
                                     ? ranges[1]
                                     : ranges.front();
            return moveCursorToByte(target.start_byte);
        }

        bool jumpToChildSyntaxBlock() {
            if (buffer == nullptr || editor == nullptr) {
                return false;
            }

            const std::string text = getText();
            const CursorLocation cursor = getCursorLocation(text);
            if (!ensureSyntaxDocumentCurrent(text, cursor.byte_index)) {
                return false;
            }

            const auto parent = syntax_document.enclosingBlockRange(cursor.byte_index).value_or(lfs::python::PythonByteRange{.start_byte = 0, .end_byte = text.size()});
            std::optional<lfs::python::PythonByteRange> first_child;
            std::optional<lfs::python::PythonByteRange> next_child;

            for (const auto& fold : syntax_document.foldRanges()) {
                if (fold.start_byte <= parent.start_byte || fold.end_byte > parent.end_byte) {
                    continue;
                }

                const lfs::python::PythonByteRange range{
                    .start_byte = fold.start_byte,
                    .end_byte = fold.end_byte,
                };
                if (!first_child.has_value() || range.start_byte < first_child->start_byte) {
                    first_child = range;
                }
                if (range.start_byte > cursor.byte_index &&
                    (!next_child.has_value() || range.start_byte < next_child->start_byte)) {
                    next_child = range;
                }
            }

            if (next_child.has_value()) {
                return moveCursorToByte(next_child->start_byte);
            }
            if (first_child.has_value()) {
                return moveCursorToByte(first_child->start_byte);
            }
            return false;
        }

        bool jumpToSyntaxSymbol(const size_t index) {
            if (buffer == nullptr || editor == nullptr || index >= syntax_document.symbols().size()) {
                return false;
            }

            const auto& symbol = syntax_document.symbols()[index];
            return moveCursorToByte(symbol.start_byte);
        }

        [[nodiscard]] std::vector<PythonEditorSymbol> syntaxBreadcrumbs() const {
            std::vector<PythonEditorSymbol> breadcrumbs;
            const std::string text = getText();
            const CursorLocation cursor = getCursorLocation(text);

            for (const auto& symbol : syntax_document.symbols()) {
                if ((symbol.kind != lfs::python::PythonSymbolKind::Class &&
                     symbol.kind != lfs::python::PythonSymbolKind::Function) ||
                    symbol.start_byte > cursor.byte_index || cursor.byte_index > symbol.end_byte) {
                    continue;
                }

                breadcrumbs.push_back(PythonEditorSymbol{
                    .label = symbol.name.empty() ? symbol.detail : symbol.name,
                    .detail = symbol.detail,
                    .byte_offset = symbol.start_byte,
                    .line = symbol.line,
                    .depth = symbol.depth,
                });
            }

            return breadcrumbs;
        }

        bool jumpToSyntaxBreadcrumb(const size_t index) {
            const auto breadcrumbs = syntaxBreadcrumbs();
            if (index >= breadcrumbs.size()) {
                return false;
            }
            return moveCursorToByte(breadcrumbs[index].byte_offset);
        }

        bool jumpToSyntaxFold(const size_t index) {
            if (buffer == nullptr || index >= buffer->GetFoldRanges().size()) {
                return false;
            }
            return moveCursorToByte(static_cast<size_t>(buffer->GetFoldRanges()[index].range.first));
        }

        bool toggleSyntaxFold(const size_t index) {
            if (buffer == nullptr || editor == nullptr || index >= buffer->GetFoldRanges().size()) {
                return false;
            }

            const bool changed = buffer->ToggleFoldAtByte(buffer->GetFoldRanges()[index].range.first);
            if (changed) {
                editor->RequestRefresh();
                focusEditor();
            }
            return changed;
        }

        void undo() {
            if (buffer == nullptr) {
                return;
            }
            if (auto* mode = buffer->GetMode()) {
                mode->Undo();
            }
            clearSelectionAndReturnToDefaultMode();
            focusEditor();
        }

        void redo() {
            if (buffer == nullptr) {
                return;
            }
            if (auto* mode = buffer->GetMode()) {
                mode->Redo();
            }
            clearSelectionAndReturnToDefaultMode();
            focusEditor();
        }

        [[nodiscard]] bool needsRmlFrame() const {
            return request_focus ||
                   is_focused ||
                   completion.visible ||
                   completion.hovered ||
                   manual_completion_requested ||
                   completion_request_pending ||
                   semantic_tokens_request_pending ||
                   pending_semantic_tokens.has_value() ||
                   syntax_analysis_pending;
        }

        std::unique_ptr<Zep::ZepEditor> editor;
        Zep::ZepBuffer* buffer = nullptr;
        Host host;
        std::unique_ptr<PythonLspClient> lsp;
        CompletionPopupState completion;
        SemanticHighlightState semantic_highlights;
        lfs::python::PythonSyntaxDocument syntax_document;
        std::set<std::shared_ptr<Zep::RangeMarker>> syntax_markers;
        std::optional<PendingSyntaxPreEdit> pending_syntax_pre_edit;
        std::vector<lfs::python::PythonBufferEdit> pending_syntax_edits;
        std::string current_syntax_scope;

        bool request_focus = false;
        bool is_focused = false;
        bool force_unfocused = false;
        bool read_only = false;
        bool text_changed = false;
        bool suppress_buffer_events = false;
        bool completion_request_pending = false;
        bool manual_completion_requested = false;
        bool syntax_analysis_pending = true;
        bool syntax_full_reparse_required = true;
        bool semantic_tokens_request_pending = false;
        bool semantic_highlighting_from_tree_sitter = false;
        std::optional<PythonLspClient::SemanticTokenList> pending_semantic_tokens;
        SemanticDirtyRange semantic_dirty_range;
        Zep::NRectf completion_popup_rect;
        int completion_popup_first_index = 0;
        float completion_popup_row_height = 0.0f;
        float mouse_x = 0.0f;
        float mouse_y = 0.0f;
        bool mouse_pos_valid = false;
        bool semantic_full_refresh_required = true;
        int document_version = 1;
        int last_requested_version = -1;
        int last_requested_line = -1;
        int last_requested_character = -1;
        int last_semantic_tokens_requested_version = -1;
        int last_cursor_line = -1;
        int last_cursor_character = -1;
        size_t last_cursor_byte_index = std::string::npos;
        Clock::time_point next_completion_request_at = Clock::now();
        Clock::time_point next_syntax_analysis_at = Clock::now();
        Clock::time_point next_semantic_tokens_request_at = Clock::now();
        Clock::time_point last_text_change_at = Clock::now();
        std::chrono::milliseconds semantic_tokens_idle_delay = SEMANTIC_TOKENS_WORD_DELAY;
        uint64_t semantic_highlight_palette_signature = 0;
        float bound_font_size = 0.0f;
        bool vim_mode_enabled = false;
    };

    PythonEditor::PythonEditor()
        : impl_(std::make_unique<Impl>()) {
    }

    PythonEditor::~PythonEditor() = default;

    bool PythonEditor::renderRml(Rml::Element& element,
                                 const float width,
                                 const float height,
                                 const float font_size_px) {
        impl_->ensureLspStarted();
        impl_->applyTheme(theme());
        if (impl_->semantic_highlighting_from_tree_sitter &&
            impl_->semantic_highlight_palette_signature != impl_->semanticPaletteSignature()) {
            impl_->applyTreeSitterHighlightingFallback(impl_->getText());
        } else if (!impl_->semantic_highlights.tokens.empty() &&
                   impl_->semantic_highlight_palette_signature != impl_->semanticPaletteSignature()) {
            impl_->applySemanticHighlighting(impl_->getText());
        }

        if (impl_->request_focus) {
            element.Focus();
            impl_->force_unfocused = false;
        }

        bool element_focused = false;
        if (auto* document = element.GetOwnerDocument()) {
            if (auto* context = document->GetContext()) {
                element_focused = context->GetFocusElement() == &element;
            }
        }
        impl_->is_focused =
            !impl_->force_unfocused && (element_focused || impl_->request_focus);

        impl_->syncFontsToRml(font_size_px);

        const float editor_width = std::max(width, 1.0f);
        const float editor_height = std::max(height, 1.0f);
        impl_->rmlDisplay().beginFrame(element);
        impl_->editor->SetDisplayRegion({0.0f, 0.0f}, {editor_width, editor_height});
        impl_->editor->Display();

        const std::string updated_text = impl_->getText();
        const CursorLocation updated_cursor = impl_->getCursorLocation(updated_text);
        impl_->updateLanguageServerState(updated_text, updated_cursor);
        impl_->updateSyntaxDiagnostics(updated_text, updated_cursor);
        impl_->renderCompletionPopup(updated_text, updated_cursor, editor_width, editor_height);
        impl_->rmlDisplay().endFrame();

        const bool editor_has_keyboard_focus =
            impl_->is_focused || impl_->request_focus || impl_->completion.visible;
        if (editor_has_keyboard_focus) {
            gui::guiFocusState().want_capture_keyboard = true;
            gui::guiFocusState().any_item_active = true;
            if (!impl_->read_only) {
                gui::guiFocusState().want_text_input = true;
            }
        }
        if (impl_->completion.visible || impl_->completion.hovered) {
            gui::guiFocusState().want_capture_keyboard = true;
            gui::guiFocusState().want_text_input = !impl_->read_only;
        }

        impl_->request_focus = false;

        return execute_requested_;
    }

    void PythonEditor::processRmlEvent(Rml::Element& element, Rml::Event& event) {
        if (!impl_ || !impl_->editor) {
            return;
        }

        const std::string type = event.GetType();
        const auto local_mouse = [&]() {
            const auto offset = element.GetAbsoluteOffset(Rml::BoxArea::Content);
            return Zep::NVec2f{
                event.GetParameter("mouse_x", 0.0f) - offset.x,
                event.GetParameter("mouse_y", 0.0f) - offset.y,
            };
        };

        if (type == "focus") {
            impl_->is_focused = true;
            impl_->force_unfocused = false;
            return;
        }
        if (type == "blur") {
            impl_->is_focused = false;
            impl_->mouse_pos_valid = false;
            impl_->completion.clear();
            return;
        }

        if (type == "mousemove" || type == "drag") {
            const auto mouse = local_mouse();
            impl_->mouse_x = mouse.x;
            impl_->mouse_y = mouse.y;
            impl_->mouse_pos_valid = true;
            impl_->editor->OnMouseMove(mouse);
            event.StopPropagation();
            return;
        }

        if (type == "mousedown") {
            element.Focus();
            impl_->is_focused = true;
            impl_->force_unfocused = false;
            const auto mouse = local_mouse();
            impl_->mouse_x = mouse.x;
            impl_->mouse_y = mouse.y;
            impl_->mouse_pos_valid = true;

            const std::string text = impl_->getText();
            const CursorLocation cursor = impl_->getCursorLocation(text);
            const int button = event.GetParameter("button", 0);
            if (impl_->handleCompletionMouseDown(text, cursor, mouse.x, mouse.y, button)) {
                event.StopPropagation();
                return;
            }

            impl_->editor->OnMouseDown(mouse, zep_mouse_button_from_rml(button));
            event.StopPropagation();
            return;
        }

        if (type == "mouseup") {
            const auto mouse = local_mouse();
            impl_->mouse_x = mouse.x;
            impl_->mouse_y = mouse.y;
            impl_->mouse_pos_valid = true;
            impl_->editor->OnMouseUp(mouse, zep_mouse_button_from_rml(event.GetParameter("button", 0)));
            event.StopPropagation();
            return;
        }

        if (type == "mousewheel") {
            const auto mouse = local_mouse();
            impl_->mouse_x = mouse.x;
            impl_->mouse_y = mouse.y;
            impl_->mouse_pos_valid = true;
            const float wheel = event.GetParameter("wheel_delta_y",
                                                   event.GetParameter("wheel_delta", 0.0f));
            impl_->editor->OnMouseWheel(mouse, -wheel);
            event.StopPropagation();
            return;
        }

        if (type == "keydown") {
            if (!impl_->is_focused && !impl_->completion.visible) {
                return;
            }

            const auto key = static_cast<Rml::Input::KeyIdentifier>(
                event.GetParameter("key_identifier", static_cast<int>(Rml::Input::KI_UNKNOWN)));
            const uint32_t modifiers = zep_modifiers_from_rml(event);
            const std::string text = impl_->getText();
            const CursorLocation cursor = impl_->getCursorLocation(text);

            if ((modifiers & Zep::ModifierKey::Ctrl) != 0 &&
                (key == Rml::Input::KI_RETURN || key == Rml::Input::KI_NUMPADENTER)) {
                execute_requested_ = true;
                event.StopPropagation();
                return;
            }
            if (key == Rml::Input::KI_F5) {
                execute_requested_ = true;
                event.StopPropagation();
                return;
            }

            if (impl_->handleCompletionKey(key, modifiers, text, cursor)) {
                event.StopPropagation();
                return;
            }

            if (!impl_->read_only) {
                if (auto mapped = zep_key_from_rml(key, modifiers)) {
                    if (auto* buffer = impl_->buffer) {
                        if (auto* mode = buffer->GetMode()) {
                            mode->AddKeyPress(*mapped, modifiers);
                            event.StopPropagation();
                        }
                    }
                }
            }
            return;
        }

        if (type == "textinput") {
            if (impl_->read_only || !impl_->is_focused || impl_->buffer == nullptr) {
                return;
            }
            auto* mode = impl_->buffer->GetMode();
            if (!mode) {
                return;
            }

            Rml::String input = event.GetParameter<Rml::String>("text", "");
            if (input.empty())
                input = event.GetParameter<Rml::String>("data", "");

            const auto add_codepoint = [&](const uint32_t codepoint) {
                if (codepoint != '\r' && codepoint != 0)
                    mode->AddKeyPress(codepoint, 0);
            };

            size_t index = 0;
            while (index < input.size()) {
                add_codepoint(decode_utf8(input, index));
            }

            if (input.empty()) {
                uint32_t codepoint = static_cast<uint32_t>(event.GetParameter("character", 0));
                if (codepoint == 0)
                    codepoint = static_cast<uint32_t>(event.GetParameter("codepoint", 0));
                add_codepoint(codepoint);
            }
            event.StopPropagation();
        }
    }

    std::string PythonEditor::getText() const {
        return impl_->getText();
    }

    std::string PythonEditor::getTextStripped() const {
        return rstrip_lines(getText());
    }

    void PythonEditor::setText(const std::string& text) {
        impl_->setTextSilently(text, std::nullopt);
    }

    void PythonEditor::clear() {
        impl_->setTextSilently("", std::nullopt);
        history_index_ = -1;
        current_input_.clear();
    }

    bool PythonEditor::consumeExecuteRequested() {
        const bool requested = execute_requested_;
        execute_requested_ = false;
        return requested;
    }

    bool PythonEditor::consumeTextChanged() {
        const bool changed = impl_->text_changed;
        impl_->text_changed = false;
        return changed;
    }

    bool PythonEditor::hasSyntaxErrors() const {
        return impl_->syntax_document.analysis().status == lfs::python::PythonBufferStatus::SyntaxError;
    }

    bool PythonEditor::syntaxDiagnosticsAvailable() const {
        return impl_->syntax_document.analysis().status != lfs::python::PythonBufferStatus::ParserUnavailable;
    }

    std::string PythonEditor::syntaxSummary() const {
        if (!impl_->syntax_document.analysis().summary.empty()) {
            return impl_->syntax_document.analysis().summary;
        }
        return "Python syntax analysis pending";
    }

    std::string PythonEditor::syntaxStructureSummary() const {
        int class_count = 0;
        int function_count = 0;
        int import_count = 0;
        int variable_count = 0;
        for (const auto& symbol : impl_->syntax_document.symbols()) {
            switch (symbol.kind) {
            case lfs::python::PythonSymbolKind::Class:
                ++class_count;
                break;
            case lfs::python::PythonSymbolKind::Function:
                ++function_count;
                break;
            case lfs::python::PythonSymbolKind::Import:
                ++import_count;
                break;
            case lfs::python::PythonSymbolKind::Variable:
                ++variable_count;
                break;
            }
        }

        std::string summary = std::format("{} import{}, {} class{}, {} function{}, {} variable{}, {} fold{}",
                                          import_count,
                                          import_count == 1 ? "" : "s",
                                          class_count,
                                          class_count == 1 ? "" : "es",
                                          function_count,
                                          function_count == 1 ? "" : "s",
                                          variable_count,
                                          variable_count == 1 ? "" : "s",
                                          impl_->syntax_document.foldRanges().size(),
                                          impl_->syntax_document.foldRanges().size() == 1 ? "" : "s");
        if (!impl_->syntax_document.structureCurrent()) {
            summary += " (partial)";
        }
        return summary;
    }

    std::vector<PythonEditorSymbol> PythonEditor::syntaxSymbols() const {
        std::vector<PythonEditorSymbol> symbols;
        symbols.reserve(impl_->syntax_document.symbols().size());

        for (const auto& symbol : impl_->syntax_document.symbols()) {
            std::string name = symbol.name.empty() ? symbol.detail : symbol.name;
            if (name.empty()) {
                name = "symbol";
            }

            const std::string indent(static_cast<size_t>(std::max(symbol.depth, 0)) * 2, ' ');
            symbols.push_back(PythonEditorSymbol{
                .label = std::format("{}{} {}  L{}",
                                     indent,
                                     syntax_symbol_prefix(symbol.kind),
                                     name,
                                     symbol.line + 1),
                .detail = symbol.detail,
                .byte_offset = symbol.start_byte,
                .line = symbol.line,
                .depth = symbol.depth,
            });
        }

        return symbols;
    }

    std::vector<PythonEditorSymbol> PythonEditor::syntaxBreadcrumbs() const {
        return impl_->syntaxBreadcrumbs();
    }

    std::vector<PythonEditorFold> PythonEditor::syntaxFolds() const {
        const std::string text = impl_->getText();
        std::vector<PythonEditorFold> folds;
        if (impl_->buffer == nullptr) {
            return folds;
        }

        folds.reserve(impl_->buffer->GetFoldRanges().size());
        for (const auto& fold : impl_->buffer->GetFoldRanges()) {
            const std::size_t start = static_cast<std::size_t>(std::max(0l, fold.range.first));
            const std::size_t end = static_cast<std::size_t>(std::max(0l, fold.range.second));
            const std::string preview = range_preview(text, start, end);
            folds.push_back(PythonEditorFold{
                .label = std::format("{} {} block  L{}-{}",
                                     fold.collapsed ? "[+]" : "[-]",
                                     fold.kind.empty() ? "syntax" : fold.kind,
                                     fold.startLine + 1,
                                     fold.endLine + 1),
                .detail = preview,
                .byte_offset = start,
                .line = static_cast<std::size_t>(std::max(0l, fold.startLine)),
                .end_line = static_cast<std::size_t>(std::max(0l, fold.endLine)),
                .collapsed = fold.collapsed,
            });
        }

        return folds;
    }

    bool PythonEditor::syntaxStructureCurrent() const {
        return impl_->syntax_document.structureCurrent();
    }

    std::size_t PythonEditor::syntaxFoldCount() const {
        return impl_->syntax_document.foldRanges().size();
    }

    std::string PythonEditor::currentSyntaxScope() const {
        return impl_->current_syntax_scope;
    }

    void PythonEditor::refreshSyntaxDiagnostics() {
        impl_->scheduleSyntaxAnalysis(std::chrono::milliseconds(0));
        if (impl_->editor != nullptr) {
            impl_->editor->RequestRefresh();
        }
    }

    bool PythonEditor::selectEnclosingSyntaxBlock() {
        return impl_->selectEnclosingSyntaxBlock();
    }

    bool PythonEditor::expandSyntaxSelection() {
        return impl_->expandSyntaxSelection();
    }

    bool PythonEditor::selectCurrentSyntaxFold() {
        return impl_->selectCurrentSyntaxFold();
    }

    bool PythonEditor::toggleCurrentSyntaxFold() {
        return impl_->toggleCurrentSyntaxFold();
    }

    bool PythonEditor::foldAllSyntaxBlocks() {
        return impl_->foldAllSyntaxBlocks();
    }

    bool PythonEditor::unfoldAllSyntaxBlocks() {
        return impl_->unfoldAllSyntaxBlocks();
    }

    bool PythonEditor::jumpToParentSyntaxBlock() {
        return impl_->jumpToParentSyntaxBlock();
    }

    bool PythonEditor::jumpToChildSyntaxBlock() {
        return impl_->jumpToChildSyntaxBlock();
    }

    bool PythonEditor::jumpToSyntaxSymbol(const std::size_t index) {
        return impl_->jumpToSyntaxSymbol(index);
    }

    bool PythonEditor::jumpToSyntaxBreadcrumb(const std::size_t index) {
        return impl_->jumpToSyntaxBreadcrumb(index);
    }

    bool PythonEditor::jumpToSyntaxFold(const std::size_t index) {
        return impl_->jumpToSyntaxFold(index);
    }

    bool PythonEditor::toggleSyntaxFold(const std::size_t index) {
        return impl_->toggleSyntaxFold(index);
    }

    void PythonEditor::updateTheme(const Theme& theme) {
        impl_->applyTheme(theme);
    }

    void PythonEditor::addToHistory(const std::string& cmd) {
        if (cmd.empty()) {
            return;
        }
        if (!history_.empty() && history_.back() == cmd) {
            return;
        }
        history_.push_back(cmd);
        history_index_ = -1;
    }

    void PythonEditor::historyUp() {
        if (history_.empty()) {
            return;
        }

        if (history_index_ == -1) {
            current_input_ = getText();
            history_index_ = static_cast<int>(history_.size()) - 1;
        } else if (history_index_ > 0) {
            --history_index_;
        }

        impl_->setTextSilently(history_[history_index_], history_[history_index_].size());
        focus();
    }

    void PythonEditor::historyDown() {
        if (history_index_ < 0) {
            return;
        }

        if (history_index_ + 1 < static_cast<int>(history_.size())) {
            ++history_index_;
            impl_->setTextSilently(history_[history_index_], history_[history_index_].size());
        } else {
            history_index_ = -1;
            impl_->setTextSilently(current_input_, current_input_.size());
        }

        focus();
    }

    void PythonEditor::focus() {
        impl_->request_focus = true;
        impl_->force_unfocused = false;
    }

    void PythonEditor::unfocus() {
        impl_->request_focus = false;
        impl_->force_unfocused = true;
        impl_->is_focused = false;
        impl_->completion.clear();
    }

    bool PythonEditor::isFocused() const {
        return impl_->is_focused && !impl_->force_unfocused;
    }

    bool PythonEditor::hasActiveCompletion() const {
        return impl_->completion.visible && !impl_->completion.items.empty();
    }

    bool PythonEditor::needsRmlFrame() const {
        return impl_ && impl_->needsRmlFrame();
    }

    void PythonEditor::setVimModeEnabled(const bool enabled) {
        impl_->setVimModeEnabled(enabled);
    }

    bool PythonEditor::isVimModeEnabled() const {
        return impl_->vim_mode_enabled;
    }

    void PythonEditor::setReadOnly(bool readonly) {
        impl_->read_only = readonly;
        if (impl_->buffer != nullptr) {
            impl_->buffer->SetFileFlags(Zep::FileFlags::ReadOnly, readonly);
        }
        if (readonly) {
            impl_->completion.clear();
        }
    }

    bool PythonEditor::isReadOnly() const {
        return impl_->read_only;
    }

} // namespace lfs::vis::editor
