#include "zep/syntax_markdown.h"
#include "zep/editor.h"
#include "zep/theme.h"

#include "zep/mcommon/logger.h"
#include "zep/mcommon/string/stringutils.h"

#include <string>
#include <vector>

namespace Zep {

    ZepSyntax_Markdown::ZepSyntax_Markdown(ZepBuffer& buffer,
                                           const std::unordered_set<std::string>& keywords,
                                           const std::unordered_set<std::string>& identifiers,
                                           uint32_t flags)
        : ZepSyntax(buffer, keywords, identifiers, flags) {
        // Don't need default
        m_adornments.clear();
    }

    void ZepSyntax_Markdown::UpdateSyntax() {
        auto& buffer = m_buffer.GetWorkingBuffer();
        auto itrCurrent = buffer.begin();
        auto itrEnd = buffer.end();

        assert(std::distance(itrCurrent, itrEnd) <= int(m_syntax.size()));
        assert(m_syntax.size() == buffer.size());

        // Mark a region of the syntax buffer with the correct marker
        auto mark = [&](GapBuffer<uint8_t>::const_iterator itrA, GapBuffer<uint8_t>::const_iterator itrB, ThemeColor type, ThemeColor background) {
            std::fill(m_syntax.begin() + (itrA - buffer.begin()), m_syntax.begin() + (itrB - buffer.begin()), SyntaxData{type, background});
        };

        auto markSingle = [&](GapBuffer<uint8_t>::const_iterator itrA, ThemeColor type, ThemeColor background) {
            (m_syntax.begin() + (itrA - buffer.begin()))->foreground = type;
            (m_syntax.begin() + (itrA - buffer.begin()))->background = background;
        };

        bool lineBegin = true;

        // Walk backwards to previous delimiter
        while (itrCurrent != itrEnd) {
            if (m_stop == true) {
                return;
            }

            // Update start location
            m_processedChar = long(itrCurrent - buffer.begin());

            if (*itrCurrent == '#' && lineBegin) {
                lineBegin = false;
                auto itrStart = itrCurrent;
                while (itrCurrent != itrEnd && *itrCurrent != '\n' && *itrCurrent != 0) {
                    itrCurrent++;
                }
                mark(itrStart, itrCurrent, ThemeColor::Identifier, ThemeColor::None);
            } else {
                if (*itrCurrent == '[') {
                    int inCount = 0;
                    auto itrStart = itrCurrent;
                    while (itrCurrent != itrEnd && *itrCurrent != '\n' && *itrCurrent != 0) {
                        if (*itrCurrent == '[') {
                            inCount++;
                        } else if (*itrCurrent == ']') {
                            inCount--;
                        }
                        itrCurrent++;
                        if (inCount == 0)
                            break;
                    }
                    mark(itrStart, itrCurrent, ThemeColor::Keyword, ThemeColor::None);
                }
            }

            if (*itrCurrent == '\n') {
                lineBegin = true;
            }

            itrCurrent++;
        }

        // If we got here, we sucessfully completed
        // Reset the target to the beginning
        m_targetChar = long(0);
        m_processedChar = long(buffer.size() - 1);
    }

} // namespace Zep
