/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/Geometry.h>
#include <core/export.hpp>

#include <deque>

namespace lfs::vis::gui {

    class LossGraphElement : public Rml::Element {
    public:
        explicit LossGraphElement(const Rml::String& tag);

        LFS_VIS_API void setData(const std::deque<float>& data);

        float getDataMin() const { return data_min_; }
        float getDataMax() const { return data_max_; }

    protected:
        void OnRender() override;
        void OnResize() override;
        bool GetIntrinsicDimensions(Rml::Vector2f& dimensions, float& ratio) override;

    private:
        void rebuild();

        static constexpr int MAX_POINTS = 512;
        static constexpr float LINE_WIDTH = 2.0f;
        static constexpr float MARGIN_FACTOR = 0.05f;
        static constexpr float VERT_PAD = 6.0f;
        static constexpr int TICK_COUNT = 3;

        std::deque<float> data_;
        float data_min_ = 0.0f;
        float data_max_ = 1.0f;

        Rml::Geometry bg_geom_;
        Rml::Geometry grid_geom_;
        Rml::Geometry line_geom_;
        bool dirty_ = true;
    };

} // namespace lfs::vis::gui
