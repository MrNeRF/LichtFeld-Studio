#pragma once

#include "zep/editor.h"

namespace Zep {

    struct IWidget {
        virtual NVec2f GetSize() const = 0; // Required size of the widget
        virtual void MouseDown(const NVec2f& pos, ZepMouseButton button) = 0;
        virtual void MouseUp(const NVec2f& pos, ZepMouseButton button) = 0;
        virtual void MouseMove(const NVec2f& pos) = 0;
        virtual void Draw(const ZepBuffer& buffer, const NVec2f& location) = 0;
        virtual void DrawInline(const ZepBuffer& buffer, const NRectf& location) = 0;
        virtual void Set(const NVec4f& value) = 0;
        virtual const NVec4f& Get() const = 0;
    };

    using fnWidgetValueChanged = std::function<void(IWidget* pWidget)>;
    class FloatSlider : public IWidget {
    public:
        FloatSlider(ZepEditor& editor, uint32_t dimension, fnWidgetValueChanged fnChanged = nullptr)
            : m_editor(editor), m_dimension(dimension), m_fnChanged(fnChanged) {
        }
        virtual NVec2f GetSize() const override;
        virtual void MouseDown(const NVec2f& pos, ZepMouseButton button) override;
        virtual void MouseUp(const NVec2f& pos, ZepMouseButton button) override;
        virtual void MouseMove(const NVec2f& pos) override;
        virtual void Draw(const ZepBuffer& buffer, const NVec2f& location) override;
        virtual void DrawInline(const ZepBuffer& buffer, const NRectf& location) override;
        virtual void Set(const NVec4f& value) override;
        virtual const NVec4f& Get() const override;

    private:
        virtual ZepEditor& GetEditor() const {
            return m_editor;
        };

    private:
        ZepEditor& m_editor;
        uint32_t m_dimension = 1;
        NVec2f m_range = NVec2f(0.0f, 1.0f);
        NVec4f m_value = NVec4f(0.0f, 0.0f, 0.0f, 0.0f);
        float m_sliderGap = 5.0f;
        fnWidgetValueChanged m_fnChanged = nullptr;
    };

    class ColorPicker : public IWidget {
    public:
        ColorPicker(ZepEditor& editor, fnWidgetValueChanged fnChanged = nullptr)
            : m_editor(editor), m_fnChanged(fnChanged) {
        }
        virtual NVec2f GetSize() const override;
        virtual void MouseDown(const NVec2f& pos, ZepMouseButton button) override;
        virtual void MouseUp(const NVec2f& pos, ZepMouseButton button) override;
        virtual void MouseMove(const NVec2f& pos) override;
        virtual void Draw(const ZepBuffer& buffer, const NVec2f& location) override;
        virtual void DrawInline(const ZepBuffer& buffer, const NRectf& location) override;
        virtual void Set(const NVec4f& value) override;
        virtual const NVec4f& Get() const override;

    private:
        virtual ZepEditor& GetEditor() const {
            return m_editor;
        };

    private:
        ZepEditor& m_editor;
        fnWidgetValueChanged m_fnChanged = nullptr;
        NVec4f m_color;
    };
} // namespace Zep
