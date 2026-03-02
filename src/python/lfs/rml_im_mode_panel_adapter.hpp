/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/panel_registry.hpp"
#include "rml_im_mode_layout.hpp"

#include <cstdint>
#include <memory>
#include <nanobind/nanobind.h>
#include <string>

namespace nb = nanobind;

namespace lfs::vis::gui {

    class RmlImModePanelAdapter : public IPanel {
    public:
        RmlImModePanelAdapter(void* manager, nb::object panel_instance, bool has_poll);
        ~RmlImModePanelAdapter() override;

        void draw(const PanelDrawContext& ctx) override;
        bool poll(const PanelDrawContext& ctx) override;
        bool supportsDirectDraw() const override { return true; }
        void drawDirect(float x, float y, float w, float h, const PanelDrawContext& ctx) override;
        float getDirectDrawHeight() const override;

    private:
        void ensureHost();
        void drawLayout();

        void* host_ = nullptr;
        void* manager_;
        nb::object panel_instance_;
        bool has_poll_;
        lfs::python::RmlImModeLayout layout_;
        uint64_t last_scene_gen_ = 0;
    };

} // namespace lfs::vis::gui
