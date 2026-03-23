/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/Geometry.h>

namespace lfs::vis::gui {

    class CRFCurveElement : public Rml::Element {
    public:
        explicit CRFCurveElement(const Rml::String& tag);

    protected:
        void OnRender() override;
        void OnResize() override;
        void OnAttributeChange(const Rml::ElementAttributes& changed) override;
        bool GetIntrinsicDimensions(Rml::Vector2f& dimensions, float& ratio) override;

    private:
        void rebuild();
        static float applyCrf(float x, float gamma, float toe, float shoulder);

        static constexpr int NUM_POINTS = 128;
        static constexpr float LINE_WIDTH = 2.5f;
        static constexpr float TOE_FACTOR = 0.5f;
        static constexpr float SHOULDER_FACTOR = 0.3f;
        static constexpr float MIDPOINT = 0.5f;

        float gamma_ = 1.0f;
        float toe_ = 0.0f;
        float shoulder_ = 0.0f;
        float gamma_r_ = 0.0f;
        float gamma_g_ = 0.0f;
        float gamma_b_ = 0.0f;

        Rml::Geometry bg_geom_;
        Rml::Geometry diag_geom_;
        Rml::Geometry curve_geom_;
        bool dirty_ = true;
    };

} // namespace lfs::vis::gui
