/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/panel_registry.hpp"

#include <string>

namespace lfs::vis {
    class VisualizerImpl;

    namespace gui {
        class AsyncTaskManager;
    }
} // namespace lfs::vis

namespace lfs::vis::gui::panels {

    class Mesh2SplatPanel : public IPanel {
    public:
        explicit Mesh2SplatPanel(VisualizerImpl* viewer);
        void draw(const PanelDrawContext& ctx) override;
        bool poll(const PanelDrawContext& ctx) override;

    private:
        void drawMeshSelector();
        void drawParameters();
        void drawConvertButton();
        void drawProgress();
        void drawError();

        int computeResolutionTarget() const;
        void triggerConversion();

        VisualizerImpl* viewer_;
        std::string selected_mesh_name_;
        bool has_initial_conversion_ = false;

        float quality_ = 0.5f;
        int resolution_index_ = 3;
        float gaussian_scale_ = 0.65f;

        static constexpr int kResolutionOptions[] = {128, 256, 512, 1024, 2048, 4096};
        static constexpr int kResolutionOptionCount = 6;
    };

} // namespace lfs::vis::gui::panels
