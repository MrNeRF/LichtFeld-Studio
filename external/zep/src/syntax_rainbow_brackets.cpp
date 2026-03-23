#include "zep/syntax_rainbow_brackets.h"
#include "zep/theme.h"

#include "zep/mcommon/logger.h"
#include "zep/mcommon/string/stringutils.h"

// A Simple adornment to add rainbow brackets to the syntax
namespace Zep {

    ZepSyntaxAdorn_RainbowBrackets::ZepSyntaxAdorn_RainbowBrackets(ZepSyntax& syntax, ZepBuffer& buffer)
        : ZepSyntaxAdorn(syntax, buffer) {
        syntax.GetEditor().RegisterCallback(this);

        Update(buffer.Begin(), buffer.End());
    }

    ZepSyntaxAdorn_RainbowBrackets::~ZepSyntaxAdorn_RainbowBrackets() {
    }

    void ZepSyntaxAdorn_RainbowBrackets::Notify(std::shared_ptr<ZepMessage> spMsg) {
        // Handle any interesting buffer messages
        if (spMsg->messageId == Msg::Buffer) {
            auto spBufferMsg = std::static_pointer_cast<BufferMessage>(spMsg);
            if (spBufferMsg->pBuffer != &m_buffer) {
                return;
            } else if (spBufferMsg->type == BufferMessageType::TextDeleted) {
                Clear(spBufferMsg->startLocation, spBufferMsg->endLocation);
            } else if (spBufferMsg->type == BufferMessageType::TextAdded || spBufferMsg->type == BufferMessageType::Loaded) {
                Insert(spBufferMsg->startLocation, spBufferMsg->endLocation);
                Update(spBufferMsg->startLocation, spBufferMsg->endLocation);
            } else if (spBufferMsg->type == BufferMessageType::TextChanged) {
                Update(spBufferMsg->startLocation, spBufferMsg->endLocation);
            }
        }
    }

    SyntaxResult ZepSyntaxAdorn_RainbowBrackets::GetSyntaxAt(const GlyphIterator& offset, bool& found) const {
        SyntaxResult data;
        auto itr = m_brackets.find(offset.Index());
        if (itr == m_brackets.end()) {
            found = false;
            return data;
        }

        found = true;
        if (!itr->second.valid) {
            data.foreground = ThemeColor::Text;
            data.background = ThemeColor::Error;
        } else {
            data.foreground = (ThemeColor)(((int32_t)ThemeColor::UniqueColor0 + itr->second.indent) % (int32_t)ThemeColor::UniqueColorLast);
            data.background = ThemeColor::None;
        }

        return data;
    }

    void ZepSyntaxAdorn_RainbowBrackets::Insert(const GlyphIterator& start, const GlyphIterator& end) {
        // Adjust all the brackets after us by the same distance
        auto diff = ByteDistance(start, end);
        std::map<long, Bracket> replace;
        for (auto& b : m_brackets) {
            if (b.first < start.Index())
                replace[b.first] = b.second;
            else
                replace[b.first + diff] = b.second;
        }
        std::swap(m_brackets, replace);

        RefreshBrackets();
    }

    void ZepSyntaxAdorn_RainbowBrackets::Clear(const GlyphIterator& start, const GlyphIterator& end) {
        // Remove brackets in the erased section
        // Note that we can't iterate here if the buffer has changed
        // TODO: this should happen on a message before delete! not after!
        auto itr = m_brackets.begin();
        while (itr != m_brackets.end() && itr->first < start.Index()) {
            itr++;
        }
        while (itr != m_brackets.end() && itr->first < end.Index()) {
            itr = m_brackets.erase(itr);
        }

        // Adjust remaining brackets by the difference
        auto diff = ByteDistance(start, end);
        std::map<long, Bracket> replace;
        for (auto& b : m_brackets) {
            if (b.first < start.Index())
                replace[b.first] = b.second;
            else
                replace[b.first - diff] = b.second;
        }
        std::swap(m_brackets, replace);

        RefreshBrackets();
    }

    void ZepSyntaxAdorn_RainbowBrackets::Update(const GlyphIterator& start, const GlyphIterator& end) {
        auto itrStart = start;
        auto itrEnd = end;

        for (auto itrBracket = itrStart; itrBracket < itrEnd; itrBracket++) {
            if (*itrBracket == '(') {
                m_brackets[itrBracket.Index()] = Bracket{0, BracketType::Bracket, true};
            } else if (*itrBracket == ')') {
                m_brackets[itrBracket.Index()] = Bracket{0, BracketType::Bracket, false};
            } else if (*itrBracket == '[') {
                m_brackets[itrBracket.Index()] = Bracket{0, BracketType::Group, true};
            } else if (*itrBracket == ']') {
                m_brackets[itrBracket.Index()] = Bracket{0, BracketType::Group, false};
            } else if (*itrBracket == '{') {
                m_brackets[itrBracket.Index()] = Bracket{0, BracketType::Brace, true};
            } else if (*itrBracket == '}') {
                m_brackets[itrBracket.Index()] = Bracket{0, BracketType::Brace, false};
            } else {
                auto itr = m_brackets.find(itrBracket.Index());
                if (itr != std::end(m_brackets)) {
                    m_brackets.erase(itr);
                }
            }
        }
        RefreshBrackets();
    }

    void ZepSyntaxAdorn_RainbowBrackets::RefreshBrackets() {
        std::vector<int32_t> indents((int)BracketType::Max, 0);
        for (auto& b : m_brackets) {
            auto& bracket = b.second;
            if (!bracket.is_open) {
                indents[int(bracket.type)]--;
            }
            bracket.indent = indents[int(bracket.type)];
            // Allow one bracket error, before going back to normal
            bracket.valid = (indents[int(bracket.type)] < 0) ? false : true;
            if (!bracket.valid) {
                indents[int(bracket.type)] = 0;
            }
            if (bracket.is_open) {
                indents[int(bracket.type)]++;
            }
        }

        auto MarkTails = [&](auto type) {
            if (indents[int(type)] > 0) {
                for (auto& b : m_brackets) {
                    if (b.second.type == type) {
                        b.second.valid = false;
                        return;
                    }
                }
            }
        };
        MarkTails(BracketType::Brace);
        MarkTails(BracketType::Bracket);
        MarkTails(BracketType::Group);
    }

} // namespace Zep
