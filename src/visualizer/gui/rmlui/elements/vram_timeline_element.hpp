/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "diagnostics/vram_timeline.hpp"

#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/Geometry.h>
#include <core/export.hpp>

#include <vector>

namespace lfs::vis::gui {

    class LFS_VIS_API VramTimelineElement : public Rml::Element {
    public:
        explicit VramTimelineElement(const Rml::String& tag);
        void setData(std::vector<lfs::diagnostics::VramTimelinePoint> points);
        void setMarkers(std::vector<lfs::diagnostics::VramMarker> markers);
        void setVisibleMask(std::uint16_t mask);
        void setWindowSeconds(int seconds);
        void setIterationAxis(bool iteration);
        void setDeviceScale(bool device);
        void setHighlightedOwner(int owner);
        void setCompact(bool compact);

    protected:
        void OnRender() override;
        void OnResize() override;
        bool GetIntrinsicDimensions(Rml::Vector2f& dimensions, float& ratio) override;

    private:
        void rebuild();
        std::vector<lfs::diagnostics::VramTimelinePoint> points_;
        std::vector<lfs::diagnostics::VramMarker> markers_;
        std::uint16_t mask_ = 0x3ff;
        int window_seconds_ = 300;
        bool iteration_axis_ = false;
        bool device_scale_ = false;
        bool compact_ = false;
        int highlighted_owner_ = -1;
        bool dirty_ = true;
        double render_total_ms_ = 0.0;
        double render_max_ms_ = 0.0;
        unsigned int render_samples_ = 0;
        Rml::Geometry grid_;
        Rml::Geometry areas_;
        Rml::Geometry lines_;
    };

} // namespace lfs::vis::gui
