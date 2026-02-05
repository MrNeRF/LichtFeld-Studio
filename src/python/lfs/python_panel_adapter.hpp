/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "py_ui.hpp"
#include "python/gil.hpp"
#include "python/python_runtime.hpp"
#include "visualizer/gui/panel_registry.hpp"

#include <cassert>
#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace lfs::python {

    namespace gui = lfs::vis::gui;

    inline gui::PanelSpace to_gui_space(PanelSpace ps) {
        switch (ps) {
        case PanelSpace::SidePanel: return gui::PanelSpace::SidePanel;
        case PanelSpace::Floating: return gui::PanelSpace::Floating;
        case PanelSpace::ViewportOverlay: return gui::PanelSpace::ViewportOverlay;
        case PanelSpace::Dockable: return gui::PanelSpace::Dockable;
        case PanelSpace::MainPanelTab: return gui::PanelSpace::MainPanelTab;
        case PanelSpace::SceneHeader: return gui::PanelSpace::SceneHeader;
        case PanelSpace::StatusBar: return gui::PanelSpace::StatusBar;
        }
        assert(false && "Unknown PanelSpace");
        return gui::PanelSpace::Floating;
    }

    class PythonPanelAdapter : public gui::IPanel {
        nb::object panel_instance_;
        bool has_poll_;

    public:
        PythonPanelAdapter(nb::object inst, bool has_poll)
            : panel_instance_(std::move(inst)),
              has_poll_(has_poll) {}

        void draw(const gui::PanelDrawContext& ctx) override {
            if (!can_acquire_gil())
                return;
            if (bridge().prepare_ui)
                bridge().prepare_ui();
            const SceneContextGuard scene_guard(ctx.scene);
            const GilAcquire gil;
            PyUILayout layout;
            panel_instance_.attr("draw")(layout);
        }

        bool poll(const gui::PanelDrawContext& ctx) override {
            (void)ctx;
            if (!has_poll_)
                return true;
            if (!can_acquire_gil())
                return false;
            if (bridge().prepare_ui)
                bridge().prepare_ui();
            const GilAcquire gil;
            return nb::cast<bool>(panel_instance_.attr("poll")(get_app_context()));
        }
    };

} // namespace lfs::python
