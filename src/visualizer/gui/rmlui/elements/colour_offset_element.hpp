/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/EventListener.h>
#include <RmlUi/Core/Geometry.h>
#include <array>

namespace lfs::vis::gui {
    // A signed, zero-centred RGB opponent plane. Achromatic offsets remain
    // editable through the standard colour picker; wheel movement preserves
    // their mean until a channel reaches its declared -1..1 bound.
    class ColourOffsetElement final : public Rml::Element, private Rml::EventListener {
    public:
        explicit ColourOffsetElement(const Rml::String& tag);

    protected:
        void OnRender() override;
        void OnResize() override;
        void OnAttributeChange(const Rml::ElementAttributes& changed) override;
        void ProcessDefaultAction(Rml::Event& event) override;
        void ProcessEvent(Rml::Event& event) override;

    private:
        void change(Rml::Vector2f offset, bool reset);
        void updatePointer(const Rml::Event& event);
        void rebuild();
        std::array<float, 3> value_{};
        bool dragging_ = false;
        bool dirty_ = true;
        std::size_t theme_signature_ = 0;
        Rml::Geometry geometry_;
    };
} // namespace lfs::vis::gui
