/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/EventListener.h>
#include <RmlUi/Core/Geometry.h>

namespace lfs::vis::gui {

    class ChromaticityElement : public Rml::Element {
    public:
        explicit ChromaticityElement(const Rml::String& tag);

    protected:
        void OnRender() override;
        void OnResize() override;
        void OnAttributeChange(const Rml::ElementAttributes& changed) override;
        void ProcessDefaultAction(Rml::Event& event) override;
        bool GetIntrinsicDimensions(Rml::Vector2f& dimensions, float& ratio) override;

    private:
        void rebuildGrid();
        void rebuildPoints();
        void readAttributes();
        void dispatchChange();
        void endDrag();

        struct DragEndListener : Rml::EventListener {
            ChromaticityElement& owner;
            explicit DragEndListener(ChromaticityElement& o) : owner(o) {}
            void ProcessEvent(Rml::Event& event) override;
        };

        static constexpr int GRID_RES = 24;
        static constexpr float RANGE = 1.0f;
        static constexpr float POINT_RADIUS = 6.0f;
        static constexpr float HIT_RADIUS = 10.0f;
        static constexpr int CIRCLE_SEGMENTS = 16;

        float red_x_ = 0.f, red_y_ = 0.f;
        float green_x_ = 0.f, green_y_ = 0.f;
        float blue_x_ = 0.f, blue_y_ = 0.f;
        float wb_temp_ = 0.f, wb_tint_ = 0.f;

        int dragging_ = -1;
        DragEndListener drag_end_listener_{*this};

        Rml::Geometry grid_geom_;
        Rml::Geometry points_geom_;
        Rml::Geometry border_geom_;
        bool grid_dirty_ = true;
        bool points_dirty_ = true;
    };

} // namespace lfs::vis::gui
