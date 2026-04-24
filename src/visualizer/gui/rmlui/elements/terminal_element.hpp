/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "gui/terminal/terminal_widget.hpp"

#include <RmlUi/Core/Element.h>

#include <string>
#include <string_view>
#include <vector>

namespace lfs::vis::gui {

    class TerminalElement : public Rml::Element {
    public:
        explicit TerminalElement(const Rml::String& tag);

        LFS_VIS_API void setSnapshot(const terminal::TerminalSnapshot& snapshot);

    protected:
        bool GetIntrinsicDimensions(Rml::Vector2f& dimensions, float& ratio) override;

    private:
        struct RowSlot {
            Rml::Element* element = nullptr;
            std::string rml;
        };

        void ensureRows(int count);
        static std::string rowToRml(const terminal::TerminalSnapshot& snapshot, int row_index);
        static std::string escapeText(std::string_view text);

        std::vector<RowSlot> rows_;
        int cols_ = 0;
        int rows_count_ = 0;
    };

} // namespace lfs::vis::gui
