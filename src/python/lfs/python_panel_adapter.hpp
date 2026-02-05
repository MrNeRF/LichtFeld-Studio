/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "py_ui.hpp"
#include "python/gil.hpp"
#include "python/python_runtime.hpp"
#include "visualizer/gui/panel_registry.hpp"

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace lfs::python {

    namespace gui = lfs::vis::gui;

    inline gui::PanelSpace to_gui_space(PanelSpace ps) {
        return static_cast<gui::PanelSpace>(static_cast<int>(ps));
    }

    static_assert(static_cast<int>(PanelSpace::SidePanel) == static_cast<int>(gui::PanelSpace::SidePanel));
    static_assert(static_cast<int>(PanelSpace::Floating) == static_cast<int>(gui::PanelSpace::Floating));
    static_assert(static_cast<int>(PanelSpace::ViewportOverlay) == static_cast<int>(gui::PanelSpace::ViewportOverlay));
    static_assert(static_cast<int>(PanelSpace::Dockable) == static_cast<int>(gui::PanelSpace::Dockable));
    static_assert(static_cast<int>(PanelSpace::MainPanelTab) == static_cast<int>(gui::PanelSpace::MainPanelTab));
    static_assert(static_cast<int>(PanelSpace::SceneHeader) == static_cast<int>(gui::PanelSpace::SceneHeader));
    static_assert(static_cast<int>(PanelSpace::StatusBar) == static_cast<int>(gui::PanelSpace::StatusBar));

    class PythonPanelAdapter : public gui::IPanel {
        nb::object panel_instance_;
        bool has_poll_;

    public:
        PythonPanelAdapter(nb::object inst, bool has_poll)
            : panel_instance_(std::move(inst)),
              has_poll_(has_poll) {}

        void draw(const gui::PanelDrawContext& ctx) override {
            (void)ctx;
            if (!can_acquire_gil())
                return;
            if (bridge().prepare_ui)
                bridge().prepare_ui();
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
