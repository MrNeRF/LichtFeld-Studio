/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rmlui/elements/terminal_element.hpp"

#include <RmlUi/Core/ElementDocument.h>

#include <algorithm>
#include <format>

namespace lfs::vis::gui {

    namespace {
        constexpr terminal::TerminalColor CURSOR_COLOR = 0xB4C8C8C8u;
        constexpr int DEFAULT_COLS = 80;
        constexpr int DEFAULT_ROWS = 24;
        constexpr float DEFAULT_CHAR_WIDTH = 8.0f;
        constexpr float DEFAULT_LINE_HEIGHT = 16.0f;

        struct CellStyle {
            terminal::TerminalColor fg = 0;
            terminal::TerminalColor bg = 0;
            bool bold = false;
            bool underline = false;

            bool operator==(const CellStyle& other) const {
                return fg == other.fg && bg == other.bg && bold == other.bold && underline == other.underline;
            }
        };

        std::string terminalColorToCss(terminal::TerminalColor color) {
            const auto r = static_cast<unsigned>(color & 0xFFu);
            const auto g = static_cast<unsigned>((color >> 8u) & 0xFFu);
            const auto b = static_cast<unsigned>((color >> 16u) & 0xFFu);
            const auto a = static_cast<unsigned>((color >> 24u) & 0xFFu);
            return std::format("rgba({},{},{},{})", r, g, b, a);
        }

        void appendRun(std::string& out, const CellStyle& style, const std::string& text) {
            if (text.empty())
                return;

            out += "<span style=\"";
            if (style.fg != 0)
                out += "color: " + terminalColorToCss(style.fg) + ";";
            if (style.bg != 0)
                out += "background-color: " + terminalColorToCss(style.bg) + ";";
            if (style.bold)
                out += "font-weight: bold;";
            if (style.underline)
                out += "text-decoration: underline;";
            out += "\">";
            out += text;
            out += "</span>";
        }
    } // namespace

    TerminalElement::TerminalElement(const Rml::String& tag) : Rml::Element(tag) {
        SetProperty("display", "block");
        SetProperty("overflow", "hidden");
        SetProperty("background-color", "rgba(30,30,30,255)");
        SetProperty("color", "rgba(229,229,229,255)");
        SetProperty("font-family", "\"JetBrains Mono\"");
        SetProperty("white-space", "pre");
    }

    bool TerminalElement::GetIntrinsicDimensions(Rml::Vector2f& dimensions, float& ratio) {
        const int cols = cols_ > 0 ? cols_ : DEFAULT_COLS;
        const int rows = rows_count_ > 0 ? rows_count_ : DEFAULT_ROWS;
        dimensions = {
            static_cast<float>(cols) * DEFAULT_CHAR_WIDTH,
            static_cast<float>(rows) * DEFAULT_LINE_HEIGHT,
        };
        ratio = 0.0f;
        return true;
    }

    void TerminalElement::setSnapshot(const terminal::TerminalSnapshot& snapshot) {
        cols_ = snapshot.cols;
        rows_count_ = snapshot.rows;
        ensureRows(snapshot.rows);

        for (int row = 0; row < snapshot.rows; ++row) {
            auto& slot = rows_[static_cast<size_t>(row)];
            std::string next = rowToRml(snapshot, row);
            if (slot.rml == next)
                continue;
            slot.rml = std::move(next);
            if (slot.element)
                slot.element->SetInnerRML(slot.rml);
        }
    }

    void TerminalElement::ensureRows(int count) {
        if (count < 0)
            count = 0;

        while (static_cast<int>(rows_.size()) > count) {
            if (rows_.back().element)
                RemoveChild(rows_.back().element);
            rows_.pop_back();
        }

        auto* doc = GetOwnerDocument();
        if (!doc)
            return;

        while (static_cast<int>(rows_.size()) < count) {
            auto row = doc->CreateElement("div");
            row->SetClass("terminal-row", true);
            row->SetProperty("display", "block");
            row->SetProperty("white-space", "pre");
            row->SetProperty("line-height", "1em");
            row->SetProperty("height", "1em");
            row->SetProperty("min-height", "1em");
            row->SetProperty("overflow", "hidden");
            rows_.push_back({AppendChild(std::move(row)), {}});
        }
    }

    std::string TerminalElement::rowToRml(const terminal::TerminalSnapshot& snapshot, int row_index) {
        if (row_index < 0 || row_index >= static_cast<int>(snapshot.visible_rows.size()))
            return {};

        const auto& row = snapshot.visible_rows[static_cast<size_t>(row_index)];
        std::string out;
        out.reserve(static_cast<size_t>(snapshot.cols) * 48);

        CellStyle current;
        std::string run_text;
        auto flush = [&]() {
            appendRun(out, current, run_text);
            run_text.clear();
        };

        for (int col = 0; col < snapshot.cols; ++col) {
            const auto* cell = col < static_cast<int>(row.cells.size())
                                   ? &row.cells[static_cast<size_t>(col)]
                                   : nullptr;
            CellStyle style;
            std::string text = " ";

            if (cell) {
                style.fg = cell->foreground;
                style.bg = cell->background;
                style.bold = cell->bold;
                style.underline = cell->underline;
                if (!cell->text.empty())
                    text = cell->text;
            }

            if (snapshot.focused && snapshot.scroll_offset == 0 && snapshot.cursor_visible &&
                row_index == snapshot.cursor_row && col == snapshot.cursor_col) {
                style.bg = CURSOR_COLOR;
            }

            if (!run_text.empty() && !(style == current))
                flush();
            current = style;
            run_text += escapeText(text);
        }

        flush();
        return out;
    }

    std::string TerminalElement::escapeText(std::string_view text) {
        std::string out;
        out.reserve(text.size());
        for (char c : text) {
            switch (c) {
            case '&':
                out += "&amp;";
                break;
            case '<':
                out += "&lt;";
                break;
            case '>':
                out += "&gt;";
                break;
            default:
                out += c;
                break;
            }
        }
        return out;
    }

} // namespace lfs::vis::gui
