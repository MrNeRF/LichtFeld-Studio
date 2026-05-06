/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/EventListener.h>

namespace lfs::vis::editor {
    class PythonEditor;
}

namespace lfs::vis::gui {

    class PythonEditorElement final : public Rml::Element {
    public:
        explicit PythonEditorElement(const Rml::String& tag);

        void setEditor(editor::PythonEditor* editor);
        void setFontSizePx(float font_size_px);

        void OnRender() override;
        void ProcessDefaultAction(Rml::Event& event) override;

    protected:
        bool GetIntrinsicDimensions(Rml::Vector2f& dimensions, float& ratio) override;

    private:
        class DragEndListener final : public Rml::EventListener {
        public:
            explicit DragEndListener(PythonEditorElement& owner) : owner_(owner) {}
            void ProcessEvent(Rml::Event& event) override;

        private:
            PythonEditorElement& owner_;
        };

        editor::PythonEditor* editor_ = nullptr;
        float font_size_px_ = 14.0f;
        DragEndListener drag_end_listener_;
    };

} // namespace lfs::vis::gui
