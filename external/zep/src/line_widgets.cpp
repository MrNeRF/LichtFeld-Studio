#include "zep/line_widgets.h"
#include "zep/display.h"

namespace Zep {

    NVec2f FloatSlider::GetSize() const {
        // Make the slider as high as the font, but return non-dpi scale
        return NVec2f((60.0f * m_dimension) + (m_sliderGap * (m_dimension - 1)), m_editor.GetDisplay().GetFont(ZepTextType::Text).GetPixelHeight() / m_editor.GetDisplay().GetPixelScale().y);
    }

    void FloatSlider::MouseDown(const NVec2f& pos, ZepMouseButton button) {
        ZEP_UNUSED(pos);
        ZEP_UNUSED(button);
    }

    void FloatSlider::MouseUp(const NVec2f& pos, ZepMouseButton button) {
        ZEP_UNUSED(pos);
        ZEP_UNUSED(button);
    }

    void FloatSlider::MouseMove(const NVec2f& pos) {
        ZEP_UNUSED(pos);
    }

    void FloatSlider::DrawInline(const ZepBuffer& buffer, const NRectf& location) {
        // Nothing inline
        ZEP_UNUSED(buffer);
        ZEP_UNUSED(location);
        return;
    }

    void FloatSlider::Draw(const ZepBuffer& buffer, const NVec2f& loc) {
        auto& display = m_editor.GetDisplay();

        for (uint32_t slider = 0; slider < m_dimension; slider++) {
            // Convert to low DPI, then double up on submit
            // We should do it this way more.
            auto location = loc / m_editor.GetDisplay().GetPixelScale().x;
            location = NVec2f(location.x + (slider * (60.0f + m_sliderGap)), location.y);

            NVec2f size = GetSize();
            size.x -= ((m_dimension - 1) * m_sliderGap);
            size.x /= m_dimension;
            auto rc = NRectf(
                NVec2f(location.x, location.y),
                NVec2f(location.x + size.x, location.y + size.y));

            NVec2f padding = NVec2f(1, 1);
            NRectf rcInner = rc;
            rcInner.Adjust(padding.x, padding.y, -padding.x, -padding.y);

            display.DrawRectFilled(DPI_RECT(rc), buffer.GetTheme().GetColor(ThemeColor::WidgetBorder));
            display.DrawRectFilled(DPI_RECT(rcInner), buffer.GetTheme().GetColor(ThemeColor::WidgetBackground));

            NRectf rcThumb = rcInner;
            rcThumb.Adjust(padding.x, padding.y, -padding.x, -padding.y);
            rcThumb = NRectf(rcThumb.Left() + 10.0f, rcThumb.Top(), 10.0f, rcThumb.Size().y);
            display.DrawRectFilled(DPI_RECT(rcThumb), buffer.GetTheme().GetColor(ThemeColor::WidgetActive));
        }
    }

    void FloatSlider::Set(const NVec4f& value) {
        m_value = value;
        if (m_fnChanged) {
            m_fnChanged(this);
        }
    }

    const NVec4f& FloatSlider::Get() const {
        return m_value;
    }

    NVec2f ColorPicker::GetSize() const {
        return NVec2f(0.0f, 0.0f);
    }

    void ColorPicker::MouseDown(const NVec2f& pos, ZepMouseButton button) {
        ZEP_UNUSED(pos);
        ZEP_UNUSED(button);
    }

    void ColorPicker::MouseUp(const NVec2f& pos, ZepMouseButton button) {
        ZEP_UNUSED(pos);
        ZEP_UNUSED(button);
    }

    void ColorPicker::MouseMove(const NVec2f& pos) {
        ZEP_UNUSED(pos);
    }

    void ColorPicker::Draw(const ZepBuffer& buffer, const NVec2f& location) {
        ZEP_UNUSED(buffer);
        ZEP_UNUSED(location);
    }

    void ColorPicker::DrawInline(const ZepBuffer& buffer, const NRectf& location) {
        ZEP_UNUSED(buffer);
        ZEP_UNUSED(location);

        auto& display = m_editor.GetDisplay();
        display.DrawRectFilled(location);
    }

    void ColorPicker::Set(const NVec4f& value) {
        m_color = value;
    }

    const NVec4f& ColorPicker::Get() const {
        return m_color;
    }

} // namespace Zep
