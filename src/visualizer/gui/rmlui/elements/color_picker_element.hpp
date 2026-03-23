/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/EventListener.h>
#include <RmlUi/Core/Geometry.h>

namespace lfs::vis::gui {

    class ColorPickerElement : public Rml::Element {
    public:
        explicit ColorPickerElement(const Rml::String& tag);

    protected:
        void OnRender() override;
        void OnResize() override;
        void OnAttributeChange(const Rml::ElementAttributes& changed) override;
        void ProcessDefaultAction(Rml::Event& event) override;
        bool GetIntrinsicDimensions(Rml::Vector2f& dimensions, float& ratio) override;

    private:
        void rebuildSVPlane();
        void rebuildHueBar();
        void rebuildCursors();
        void readAttributes();
        void dispatchChange();
        void endDrag();

        struct DragEndListener : Rml::EventListener {
            ColorPickerElement& owner;
            explicit DragEndListener(ColorPickerElement& o) : owner(o) {}
            void ProcessEvent(Rml::Event& event) override;
        };

        static constexpr int GRID_RES = 32;
        static constexpr float HUE_BAR_WIDTH = 20.0f;
        static constexpr float GAP = 4.0f;
        static constexpr float CURSOR_SIZE = 6.0f;
        static constexpr float CURSOR_THICKNESS = 1.5f;

        float h_ = 0.f, s_ = 1.f, v_ = 1.f;

        int drag_target_ = -1;
        DragEndListener drag_end_listener_{*this};

        Rml::Geometry sv_geom_;
        Rml::Geometry hue_geom_;
        Rml::Geometry cursor_geom_;
        bool sv_dirty_ = true;
        bool hue_dirty_ = true;
        bool cursor_dirty_ = true;
    };

} // namespace lfs::vis::gui
