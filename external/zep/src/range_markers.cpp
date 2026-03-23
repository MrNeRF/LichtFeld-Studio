#include "zep/range_markers.h"
#include "zep/buffer.h"
#include "zep/editor.h"

namespace Zep {

    RangeMarker::RangeMarker(ZepBuffer& buffer)
        : m_buffer(buffer) {
        onPreBufferInsert = buffer.sigPreInsert.connect([=](ZepBuffer& buffer, const GlyphIterator& itrStart, const std::string& str) {
            HandleBufferInsert(buffer, itrStart, str);
        });
        onPreBufferDelete = buffer.sigPreDelete.connect([=](ZepBuffer& buffer, const GlyphIterator& itrStart, const GlyphIterator itrEnd) {
            HandleBufferDelete(buffer, itrStart, itrEnd);
        });
    }

    bool RangeMarker::ContainsLocation(GlyphIterator loc) const {
        return m_range.ContainsLocation(loc.Index());
    }

    bool RangeMarker::IntersectsRange(const ByteRange& i) const {
        return i.first < m_range.second && i.second > m_range.first;
    }

    ThemeColor RangeMarker::GetBackgroundColor(const GlyphIterator& itr) const {
        ZEP_UNUSED(itr);
        return m_backgroundColor;
    }

    ThemeColor RangeMarker::GetTextColor(const GlyphIterator& itr) const {
        ZEP_UNUSED(itr);
        return m_textColor;
    }

    ThemeColor RangeMarker::GetHighlightColor(const GlyphIterator& itr) const {
        ZEP_UNUSED(itr);
        return m_highlightColor;
    }

    float RangeMarker::GetAlpha(const GlyphIterator& itr) const {
        ZEP_UNUSED(itr);
        return alpha;
    }

    void RangeMarker::SetBackgroundColor(ThemeColor color) {
        m_backgroundColor = color;
    }

    void RangeMarker::SetTextColor(ThemeColor color) {
        m_textColor = color;
    }

    void RangeMarker::SetHighlightColor(ThemeColor color) {
        m_highlightColor = color;
    }

    void RangeMarker::SetColors(ThemeColor back, ThemeColor text, ThemeColor highlight) {
        m_backgroundColor = back;
        m_textColor = text;
        m_highlightColor = highlight;
    }

    void RangeMarker::SetAlpha(float a) {
        alpha = a;
    }

    void RangeMarker::SetRange(ByteRange range) {
        auto spMarker = shared_from_this();
        m_buffer.ClearRangeMarker(spMarker);

        m_range = range;
        m_buffer.AddRangeMarker(spMarker);
    }

    const ByteRange& RangeMarker::GetRange() const {
        return m_range;
    }

    ZepBuffer& RangeMarker::GetBuffer() {
        return m_buffer;
    }

    // By Default Markers will:
    // - Move down if text is inserted before them.
    // - Move up if text is deleted before them.
    // - Remove themselves from the buffer if text is edited _inside_ them.
    // Derived markers can modify this behavior.
    // Its up to marker owners to update this behavior if necessary
    // Markers do not act inside the undo/redo system.  They live on the buffer but are not stored with it.  They are adornments that
    // must be managed externally
    void RangeMarker::HandleBufferInsert(ZepBuffer& buffer, const GlyphIterator& itrStart, const std::string& str) {
        if (!m_enabled) {
            return;
        }

        if (itrStart.Index() > GetRange().second) {
            return;
        } else {
            auto itrEnd = itrStart + long(str.size());
            if (itrEnd.Index() <= (GetRange().first + 1)) {
                auto distance = itrEnd.Index() - itrStart.Index();
                auto currentRange = GetRange();
                SetRange(ByteRange(currentRange.first + distance, currentRange.second + distance));
            } else {
                buffer.ClearRangeMarker(shared_from_this());
                m_enabled = false;
            }
        }
    }

    void RangeMarker::HandleBufferDelete(ZepBuffer& buffer, const GlyphIterator& itrStart, const GlyphIterator& itrEnd) {
        if (!m_enabled) {
            return;
        }

        if (itrStart.Index() > GetRange().second) {
            return;
        } else {
            ZLOG(INFO, "Range: " << itrStart.Index() << ", " << itrEnd.Index() << " : mark: " << GetRange().first);

            // It's OK to move on the first char; since that is like a shove
            if (itrEnd.Index() < (GetRange().first + 1)) {
                auto distance = std::min(itrEnd.Index(), GetRange().first) - itrStart.Index();
                auto currentRange = GetRange();
                SetRange(ByteRange(currentRange.first - distance, currentRange.second - distance));
            } else {
                buffer.ClearRangeMarker(shared_from_this());
                m_enabled = false;
            }
        }
    }

    const std::string& RangeMarker::GetName() const {
        return m_name;
    }

    const std::string& RangeMarker::GetDescription() const {
        return m_description;
    }

    void RangeMarker::SetName(const std::string& strName) {
        m_name = strName;
    }

    void RangeMarker::SetDescription(const std::string& strDescription) {
        m_description = strDescription;
    }

    void RangeMarker::SetEnabled(bool enabled) {
        m_enabled = enabled;
    }

    const NVec2f& RangeMarker::GetInlineSize() const {
        return m_inlineSize;
    }

    void RangeMarker::SetInlineSize(const NVec2f& size) {
        m_inlineSize = size;
    }

}; // namespace Zep
