/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rmlui/elements/python_editor_element.hpp"

#include "gui/editor/python_editor.hpp"

#include <RmlUi/Core/Event.h>

#include <algorithm>

namespace lfs::vis::gui {

    PythonEditorElement::PythonEditorElement(const Rml::String& tag)
        : Rml::Element(tag),
          drag_end_listener_(*this) {
        SetProperty("display", "block");
        SetProperty("overflow", "hidden");
        SetProperty("width", "100%");
        SetProperty("height", "100%");
        SetProperty("drag", "drag");
        SetAttribute("tabindex", "0");
        SetAttribute("data-text-input", "true");
        AddEventListener(Rml::EventId::Dragend, &drag_end_listener_);
    }

    void PythonEditorElement::DragEndListener::ProcessEvent(Rml::Event& event) {
        if (owner_.editor_) {
            owner_.editor_->processRmlEvent(owner_, event);
        }
    }

    void PythonEditorElement::setEditor(editor::PythonEditor* editor) {
        editor_ = editor;
    }

    void PythonEditorElement::setFontSizePx(const float font_size_px) {
        font_size_px_ = std::max(1.0f, font_size_px);
    }

    void PythonEditorElement::OnRender() {
        if (!editor_) {
            return;
        }

        const auto size = GetBox().GetSize(Rml::BoxArea::Content);
        if (size.x <= 0.0f || size.y <= 0.0f) {
            return;
        }

        editor_->renderRml(*this, size.x, size.y, font_size_px_);
    }

    void PythonEditorElement::ProcessDefaultAction(Rml::Event& event) {
        Element::ProcessDefaultAction(event);
        if (editor_) {
            editor_->processRmlEvent(*this, event);
        }
    }

    bool PythonEditorElement::GetIntrinsicDimensions(Rml::Vector2f& dimensions, float& ratio) {
        dimensions = {1.0f, 1.0f};
        ratio = 0.0f;
        return true;
    }

} // namespace lfs::vis::gui
