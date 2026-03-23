/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "syntax.h"

namespace Zep {

    struct ZepSemanticHighlight {
        long start = 0;
        long end = 0;
        ThemeColor foreground = ThemeColor::None;
        bool custom_foreground = false;
        NVec4f custom_foreground_color = {};
        bool underline = false;
    };

    class ZepSyntax_Python : public ZepSyntax {
    public:
        ZepSyntax_Python(ZepBuffer& buffer,
                         const std::unordered_set<std::string>& keywords = std::unordered_set<std::string>{},
                         const std::unordered_set<std::string>& identifiers = std::unordered_set<std::string>{},
                         uint32_t flags = 0);

        SyntaxResult GetSyntaxAt(const GlyphIterator& index) const override;
        void Notify(std::shared_ptr<ZepMessage> message) override;
        void UpdateSyntax() override;
        void ClearSemanticHighlighting();
        void ReplaceSemanticHighlighting(long start,
                                         long end,
                                         const std::vector<ZepSemanticHighlight>& highlights);
        void SetSemanticHighlighting(const std::vector<ZepSemanticHighlight>& highlights);

    private:
        struct SemanticCell {
            ThemeColor foreground = ThemeColor::None;
            bool custom_foreground = false;
            NVec4f custom_foreground_color = {};
            bool underline = false;
            bool active = false;
        };

        std::vector<SemanticCell> semantic_cells_;
    };

} // namespace Zep
