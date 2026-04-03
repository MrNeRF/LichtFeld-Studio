/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "python_editor.hpp"

#include "python_lsp_client.hpp"

#include "gui/gui_focus_state.hpp"
#include "theme/theme.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

#include <imgui.h>

#include <zep/buffer.h>
#include <zep/imgui/display_imgui.h>
#include <zep/imgui/editor_imgui.h>
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
        constexpr auto SEMANTIC_TOKENS_WORD_DELAY = std::chrono::milliseconds(800);
        constexpr auto SEMANTIC_TOKENS_BOUNDARY_DELAY = std::chrono::milliseconds(90);
        constexpr int COMPLETION_POPUP_MAX_ITEMS = 8;

        Zep::NVec4f to_zep(const ImVec4& color) {
            return {color.x, color.y, color.z, color.w};
        }

        Zep::NVec4f mix(const ImVec4& a, const ImVec4& b, float t) {
            return {
                a.x + (b.x - a.x) * t,
                a.y + (b.y - a.y) * t,
                a.z + (b.z - a.z) * t,
                a.w + (b.w - a.w) * t,
            };
        }

        ImVec4 mix_imgui(const ImVec4& a, const ImVec4& b, float t) {
            return {
                a.x + (b.x - a.x) * t,
                a.y + (b.y - a.y) * t,
                a.z + (b.z - a.z) * t,
                a.w + (b.w - a.w) * t,
            };
        }

        ImVec4 with_alpha(const ImVec4& color, float alpha) {
            return {color.x, color.y, color.z, alpha};
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
                    if (const char* clipboard = ImGui::GetClipboardText()) {
                        message->str = clipboard;
                    } else {
                        message->str.clear();
                    }
                    message->handled = true;
                    return;
                }

                if (message->messageId == Zep::Msg::SetClipBoard) {
                    ImGui::SetClipboardText(message->str.c_str());
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
                case Zep::BufferMessageType::TextAdded:
                case Zep::BufferMessageType::TextChanged:
                case Zep::BufferMessageType::TextDeleted:
                    owner_.text_changed = true;
                    owner_.noteSemanticDirtyRange(
                        static_cast<size_t>(std::max(0l, buffer_message->startLocation.Index())),
                        static_cast<size_t>(std::max(0l, buffer_message->endLocation.Index())));
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
            : editor(std::make_unique<Zep::ZepEditor_ImGui>(
                  std::filesystem::path(PROJECT_ROOT_PATH) / "external" / "zep",
                  Zep::NVec2f(1.0f),
                  Zep::ZepEditorFlags::DisableThreads)),
              host(*this) {
            editor->RegisterCallback(&host);
            editor->GetDisplay().SetPixelScale(Zep::NVec2f(1.0f));
            editor->SetGlobalMode(Zep::ZepMode_Standard::StaticName());

            auto& config = editor->GetConfig();
            config.showLineNumbers = true;
            config.showIndicatorRegion = false;
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

            const ImVec4 scroll_track = with_alpha(app_theme.palette.background, 0.5f);
            const ImVec4 scroll_thumb = with_alpha(app_theme.palette.text_dim, 0.63f);
            const ImVec4 scroll_hover = with_alpha(app_theme.palette.primary, 0.78f);
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

        void syncFontsToImGui() {
            ImFont* const font = ImGui::GetFont();
            const float font_size = ImGui::GetFontSize();
            if (font == nullptr) {
                return;
            }

            if (font == bound_font && std::abs(font_size - bound_font_size) < 0.5f) {
                return;
            }

            auto& display = static_cast<Zep::ZepDisplay_ImGui&>(editor->GetDisplay());
            auto make_font = [&](float scale) {
                return std::make_shared<Zep::ZepFont_ImGui>(
                    display,
                    font,
                    std::max(1, static_cast<int>(std::lround(font_size * scale))));
            };

            display.SetFont(Zep::ZepTextType::UI, make_font(0.95f));
            display.SetFont(Zep::ZepTextType::Text, make_font(1.0f));
            display.SetFont(Zep::ZepTextType::Heading1, make_font(1.35f));
            display.SetFont(Zep::ZepTextType::Heading2, make_font(1.2f));
            display.SetFont(Zep::ZepTextType::Heading3, make_font(1.1f));

            bound_font = font;
            bound_font_size = font_size;
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
            clearSemanticHighlighting();
            semantic_dirty_range.clear();
            semantic_full_refresh_required = true;
            semantic_tokens_request_pending = true;
            last_text_change_at = Clock::now();
            semantic_tokens_idle_delay = SEMANTIC_TOKENS_BOUNDARY_DELAY;
            next_semantic_tokens_request_at = last_text_change_at + semantic_tokens_idle_delay;
            last_semantic_tokens_requested_version = -1;
            pending_semantic_tokens.reset();

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
            const auto mix_u32 = [](const ImVec4& color) {
                return static_cast<uint64_t>(ImGui::ColorConvertFloat4ToU32(color));
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
            semantic_full_refresh_required = true;
            semantic_dirty_range.clear();

            if (buffer == nullptr) {
                return;
            }

            if (auto* syntax = dynamic_cast<Zep::ZepSyntax_Python*>(buffer->GetSyntax())) {
                syntax->ClearSemanticHighlighting();
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
            ImVec4 color = palette.text;
            bool use_custom = true;
            bool underline = (token.modifiers & (1u << 4)) != 0;

            if (token.type == "class" || token.type == "type" || token.type == "enum" ||
                token.type == "interface" || token.type == "struct" ||
                token.type == "typeParameter") {
                color = palette.info;
            } else if (token.type == "function" || token.type == "method") {
                color = mix_imgui(palette.primary, palette.info, 0.35f);
            } else if (token.type == "decorator") {
                color = mix_imgui(palette.primary, palette.secondary, 0.25f);
            } else if (token.type == "namespace" || token.type == "module") {
                color = palette.secondary;
            } else if (token.type == "property" || token.type == "enumMember") {
                color = mix_imgui(palette.secondary, palette.text, 0.35f);
            } else if (token.type == "parameter") {
                color = mix_imgui(palette.warning, palette.text, 0.25f);
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

        bool handleCompletionKeys(const std::string& text, const CursorLocation& cursor) {
            if (!is_focused || read_only) {
                if (!completion.visible) {
                    return false;
                }
            }

            if (read_only) {
                return false;
            }

            bool consumed = false;
            if (ImGui::GetIO().KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_Space, false)) {
                manual_completion_requested = true;
                consumed = true;
            }

            if (!completion.visible || completion.items.empty()) {
                return consumed;
            }

            if (ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
                completion.clear();
                return true;
            }

            if (ImGui::IsKeyPressed(ImGuiKey_DownArrow, false)) {
                completion.selected_index =
                    (completion.selected_index + 1) % static_cast<int>(completion.items.size());
                completion.scroll_to_selected = true;
                completion.keyboard_navigation_active = true;
                return true;
            }

            if (ImGui::IsKeyPressed(ImGuiKey_UpArrow, false)) {
                completion.selected_index =
                    (completion.selected_index + static_cast<int>(completion.items.size()) - 1) %
                    static_cast<int>(completion.items.size());
                completion.scroll_to_selected = true;
                completion.keyboard_navigation_active = true;
                return true;
            }

            if (ImGui::IsKeyPressed(ImGuiKey_PageDown, false)) {
                completion.selected_index = std::min(
                    completion.selected_index + COMPLETION_POPUP_MAX_ITEMS,
                    static_cast<int>(completion.items.size()) - 1);
                completion.scroll_to_selected = true;
                completion.keyboard_navigation_active = true;
                return true;
            }

            if (ImGui::IsKeyPressed(ImGuiKey_PageUp, false)) {
                completion.selected_index =
                    std::max(completion.selected_index - COMPLETION_POPUP_MAX_ITEMS, 0);
                completion.scroll_to_selected = true;
                completion.keyboard_navigation_active = true;
                return true;
            }

            if (ImGui::IsKeyPressed(ImGuiKey_Tab, false) ||
                (!ImGui::GetIO().KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_Enter, false))) {
                applyCompletion(text, cursor, completion.items[completion.selected_index].item);
                return true;
            }

            return consumed;
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

        void renderCompletionPopup(const std::string& text, const CursorLocation& cursor) {
            completion.hovered = false;
            if (!completion.visible || completion.items.empty()) {
                return;
            }

            auto* window = editor->GetActiveWindow();
            if (window == nullptr) {
                completion.clear();
                return;
            }

            const auto cursor_rect = window->GetCursorRect();
            const float row_height = ImGui::GetTextLineHeightWithSpacing() + 2.0f;
            const ImVec2 editor_window_pos = ImGui::GetWindowPos();
            const ImVec2 editor_window_size = ImGui::GetWindowSize();
            const float popup_width =
                std::max(180.0f, std::min(440.0f, editor_window_size.x - 12.0f));
            const float popup_height = std::min(
                std::min<int>(static_cast<int>(completion.items.size()), COMPLETION_POPUP_MAX_ITEMS) *
                        row_height +
                    12.0f,
                std::max(row_height + 12.0f, editor_window_size.y - 12.0f));

            ImGuiViewport* const viewport = ImGui::GetWindowViewport();
            ImVec2 popup_pos(completionAnchorX(text, cursor, cursor_rect),
                             cursor_rect.bottomRightPx.y + 4.0f);
            if (popup_pos.y + popup_height > editor_window_pos.y + editor_window_size.y - 6.0f) {
                popup_pos.y =
                    std::max(editor_window_pos.y + 6.0f,
                             cursor_rect.topLeftPx.y - popup_height - 4.0f);
            }
            popup_pos.x = std::clamp(
                popup_pos.x,
                editor_window_pos.x + 6.0f,
                editor_window_pos.x + editor_window_size.x - popup_width - 6.0f);
            popup_pos.y = std::clamp(
                popup_pos.y,
                editor_window_pos.y + 6.0f,
                editor_window_pos.y + editor_window_size.y - popup_height - 6.0f);

            const auto& palette = theme().palette;

            ImGui::SetNextWindowViewport(viewport->ID);
            ImGui::SetNextWindowPos(popup_pos);
            ImGui::SetNextWindowSize(ImVec2(popup_width, popup_height));
            ImGui::SetNextWindowBgAlpha(1.0f);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 6.0f));
            ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 6.0f);
            ImVec4 header = palette.primary;
            header.w *= 0.25f;
            ImVec4 header_hovered = palette.primary;
            header_hovered.w *= 0.5f;
            ImVec4 header_active = palette.primary;
            header_active.w *= 0.7f;
            ImGui::PushStyleColor(ImGuiCol_WindowBg, palette.background);
            ImGui::PushStyleColor(ImGuiCol_Border, palette.border);
            ImGui::PushStyleColor(ImGuiCol_Header, header);
            ImGui::PushStyleColor(ImGuiCol_HeaderHovered, header_hovered);
            ImGui::PushStyleColor(ImGuiCol_HeaderActive, header_active);

            constexpr ImGuiWindowFlags POPUP_FLAGS =
                ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings |
                ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav |
                ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_Tooltip;

            if (ImGui::Begin("##python_completion_popup", nullptr, POPUP_FLAGS)) {
                completion.hovered = ImGui::IsWindowHovered(ImGuiHoveredFlags_RootAndChildWindows);
                if (completion.hovered) {
                    gui::guiFocusState().want_capture_mouse = true;
                }

                if (std::abs(ImGui::GetIO().MouseDelta.x) > 0.0f ||
                    std::abs(ImGui::GetIO().MouseDelta.y) > 0.0f ||
                    ImGui::IsMouseClicked(ImGuiMouseButton_Left) ||
                    ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
                    completion.keyboard_navigation_active = false;
                }

                ImDrawList* const draw_list = ImGui::GetWindowDrawList();
                const float row_left =
                    ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMin().x;
                const float row_right =
                    ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMax().x;
                ImGuiListClipper clipper;
                clipper.Begin(static_cast<int>(completion.items.size()), row_height);
                while (clipper.Step()) {
                    for (int index = clipper.DisplayStart; index < clipper.DisplayEnd; ++index) {
                        const auto& entry = completion.items[index];
                        const auto& item = entry.item;
                        const bool selected = index == completion.selected_index;

                        ImGui::PushID(index);
                        const float row_width = std::max(1.0f, row_right - row_left);
                        if (ImGui::Selectable("##completion_item", selected,
                                              ImGuiSelectableFlags_AllowOverlap,
                                              ImVec2(row_width, row_height))) {
                            completion.selected_index = index;
                            completion.scroll_to_selected = false;
                            completion.keyboard_navigation_active = false;
                            applyCompletion(text, cursor, item);
                        }

                        const bool hovered = ImGui::IsItemHovered();
                        if (hovered && !completion.keyboard_navigation_active) {
                            completion.selected_index = index;
                            completion.scroll_to_selected = false;
                        }

                        if (selected && completion.scroll_to_selected) {
                            ImGui::SetScrollHereY(0.35f);
                            completion.scroll_to_selected = false;
                        }

                        const ImVec2 min = ImGui::GetItemRectMin();
                        const ImVec2 max = ImGui::GetItemRectMax();
                        const bool active_row = index == completion.selected_index;
                        if (active_row || hovered) {
                            draw_list->AddRectFilled(
                                ImVec2(row_left, min.y),
                                ImVec2(row_right, max.y),
                                active_row ? theme().selection_fill_u32()
                                           : ImGui::GetColorU32(ImGuiCol_HeaderHovered));
                        }

                        const ImU32 badge_color = ImGui::ColorConvertFloat4ToU32(
                            active_row ? palette.primary : palette.text_dim);
                        const ImU32 text_color = ImGui::ColorConvertFloat4ToU32(
                            item.deprecated ? palette.text_dim : palette.text);
                        const ImU32 highlight_color =
                            ImGui::ColorConvertFloat4ToU32(palette.primary);
                        const ImU32 detail_color =
                            ImGui::ColorConvertFloat4ToU32(palette.text_dim);

                        draw_list->AddText(ImVec2(min.x + 10.0f, min.y + 3.0f), badge_color,
                                           completion_kind_badge(item.kind));
                        const ImVec2 label_pos(min.x + 30.0f, min.y + 3.0f);
                        const size_t label_highlight =
                            common_prefix_length_ci(item.label, completion.typed_prefix);
                        if (label_highlight > 0) {
                            const std::string highlight_text =
                                item.label.substr(0, label_highlight);
                            draw_list->AddText(label_pos, highlight_color, highlight_text.c_str());

                            const ImVec2 highlight_size =
                                ImGui::CalcTextSize(highlight_text.c_str());
                            draw_list->AddText(ImVec2(label_pos.x + highlight_size.x, label_pos.y),
                                               text_color,
                                               item.label.c_str() + static_cast<ptrdiff_t>(label_highlight));
                        } else {
                            draw_list->AddText(label_pos, text_color, item.label.c_str());
                        }

                        std::string detail = item.detail.empty() ? item.description : item.detail;
                        const std::string insert_preview = completionInsertPreviewText(item);
                        if (active_row && !insert_preview.empty() && insert_preview != item.label) {
                            detail = "-> " + insert_preview;
                        }

                        if (!detail.empty()) {
                            const ImVec2 detail_size = ImGui::CalcTextSize(detail.c_str());
                            draw_list->AddText(
                                ImVec2(std::max(min.x + 150.0f, max.x - detail_size.x - 10.0f),
                                       min.y + 3.0f),
                                active_row ? highlight_color : detail_color,
                                detail.c_str());
                        }

                        ImGui::PopID();
                    }
                }
            }
            ImGui::End();

            ImGui::PopStyleColor(5);
            ImGui::PopStyleVar(2);
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

        std::unique_ptr<Zep::ZepEditor_ImGui> editor;
        Zep::ZepBuffer* buffer = nullptr;
        Host host;
        std::unique_ptr<PythonLspClient> lsp;
        CompletionPopupState completion;
        SemanticHighlightState semantic_highlights;

        bool request_focus = false;
        bool is_focused = false;
        bool force_unfocused = false;
        bool read_only = false;
        bool text_changed = false;
        bool suppress_buffer_events = false;
        bool completion_request_pending = false;
        bool manual_completion_requested = false;
        bool semantic_tokens_request_pending = false;
        std::optional<PythonLspClient::SemanticTokenList> pending_semantic_tokens;
        SemanticDirtyRange semantic_dirty_range;
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
        Clock::time_point next_semantic_tokens_request_at = Clock::now();
        Clock::time_point last_text_change_at = Clock::now();
        std::chrono::milliseconds semantic_tokens_idle_delay = SEMANTIC_TOKENS_WORD_DELAY;
        uint64_t semantic_highlight_palette_signature = 0;
        ImFont* bound_font = nullptr;
        float bound_font_size = 0.0f;
        bool vim_mode_enabled = false;
    };

    PythonEditor::PythonEditor()
        : impl_(std::make_unique<Impl>()) {
    }

    PythonEditor::~PythonEditor() = default;

    bool PythonEditor::render(const ImVec2& size) {
        execute_requested_ = false;
        impl_->ensureLspStarted();
        impl_->applyTheme(theme());
        if (!impl_->semantic_highlights.tokens.empty() &&
            impl_->semantic_highlight_palette_signature != impl_->semanticPaletteSignature()) {
            impl_->applySemanticHighlighting(impl_->getText());
        }

        ImGuiIO& io = ImGui::GetIO();

        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
        constexpr ImGuiWindowFlags CHILD_FLAGS =
            ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse;

        if (ImGui::BeginChild("##python_input", size, true, CHILD_FLAGS)) {
            if (impl_->request_focus) {
                ImGui::SetWindowFocus();
                impl_->force_unfocused = false;
            }

            impl_->syncFontsToImGui();

            const ImVec2 min = ImGui::GetCursorScreenPos();
            const ImVec2 avail = ImGui::GetContentRegionAvail();
            const ImVec2 max(min.x + std::max(avail.x, 1.0f), min.y + std::max(avail.y, 1.0f));

            impl_->editor->SetDisplayRegion(Zep::NVec2f(min.x, min.y), Zep::NVec2f(max.x, max.y));
            impl_->editor->Display();

            const bool child_hovered = ImGui::IsWindowHovered(ImGuiHoveredFlags_RootAndChildWindows);
            const bool child_focused = ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows);
            const bool activated_by_click =
                child_hovered && (ImGui::IsMouseClicked(ImGuiMouseButton_Left) ||
                                  ImGui::IsMouseClicked(ImGuiMouseButton_Right));
            const bool mouse_interaction =
                child_hovered && (activated_by_click || io.MouseReleased[0] || io.MouseReleased[1] ||
                                  io.MouseWheel != 0.0f || io.MouseDelta.x != 0.0f ||
                                  io.MouseDelta.y != 0.0f);

            if (activated_by_click) {
                impl_->force_unfocused = false;
            }

            impl_->is_focused =
                !impl_->force_unfocused && (child_focused || impl_->request_focus || activated_by_click);

            const std::string text = impl_->getText();
            const CursorLocation cursor = impl_->getCursorLocation(text);
            const bool completion_key_consumed = impl_->handleCompletionKeys(text, cursor);

            const bool editor_has_keyboard_focus =
                impl_->is_focused || impl_->request_focus || impl_->completion.visible;
            if (editor_has_keyboard_focus) {
                execute_requested_ = io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_Enter, false);
                ImGui::GetIO().WantCaptureKeyboard = true;
                gui::guiFocusState().want_capture_keyboard = true;
                gui::guiFocusState().any_item_active = true;
                if (!impl_->read_only) {
                    ImGui::GetIO().WantTextInput = true;
                    gui::guiFocusState().want_text_input = true;
                }
            }

            const bool should_handle_input =
                !impl_->read_only && ((impl_->is_focused || impl_->request_focus) || mouse_interaction);
            if (should_handle_input && !execute_requested_ && !completion_key_consumed) {
                impl_->editor->HandleInput();
            }

            const std::string updated_text = impl_->getText();
            const CursorLocation updated_cursor = impl_->getCursorLocation(updated_text);
            impl_->updateLanguageServerState(updated_text, updated_cursor);
            impl_->renderCompletionPopup(updated_text, updated_cursor);

            if (impl_->completion.visible || impl_->completion.hovered) {
                gui::guiFocusState().want_capture_keyboard = true;
                gui::guiFocusState().want_text_input = true;
            }

            if (impl_->completion.hovered || (impl_->completion.visible && child_hovered)) {
                gui::guiFocusState().want_capture_mouse = true;
            }

            if (impl_->completion.visible && !impl_->completion.hovered &&
                !child_hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                impl_->completion.clear();
            }

            impl_->request_focus = false;
        }
        ImGui::EndChild();
        ImGui::PopStyleVar();

        return execute_requested_;
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

    bool PythonEditor::consumeTextChanged() {
        const bool changed = impl_->text_changed;
        impl_->text_changed = false;
        return changed;
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
