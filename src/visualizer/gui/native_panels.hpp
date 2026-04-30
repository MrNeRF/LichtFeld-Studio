/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/panel_registry.hpp"

namespace lfs::vis::gui {

    class StartupOverlay;
    class GizmoManager;
    class GuiManager;
    class SequencerUIManager;
    class PanelLayoutManager;
    class RmlStatusBar;

} // namespace lfs::vis::gui

namespace lfs::gui {
    class IVideoExtractorWidget;
}

namespace lfs::vis::gui::native_panels {

    class VideoExtractorPanel : public IPanel {
    public:
        explicit VideoExtractorPanel(lfs::gui::IVideoExtractorWidget* widget);
        void draw(const PanelDrawContext& ctx) override;

    private:
        lfs::gui::IVideoExtractorWidget* widget_;
    };

    class StartupOverlayPanel : public IPanel {
    public:
        StartupOverlayPanel(StartupOverlay* overlay, const bool* drag_hovering);
        void draw(const PanelDrawContext& ctx) override;
        bool poll(const PanelDrawContext& ctx) override;

    private:
        StartupOverlay* overlay_;
        const bool* drag_hovering_;
    };

    class SelectionOverlayPanel : public IPanel {
    public:
        explicit SelectionOverlayPanel(GuiManager* gui);
        void draw(const PanelDrawContext& ctx) override;

    private:
        GuiManager* gui_;
    };

    class ViewportDecorationsPanel : public IPanel {
    public:
        explicit ViewportDecorationsPanel(GuiManager* gui);
        void draw(const PanelDrawContext& ctx) override;

    private:
        GuiManager* gui_;
    };

    class SequencerPanel : public IPanel {
    public:
        SequencerPanel(SequencerUIManager* seq, const PanelLayoutManager* layout);
        void draw(const PanelDrawContext& ctx) override;
        bool poll(const PanelDrawContext& ctx) override;
        bool supportsDirectDraw() const override { return true; }
        void preloadDirect(float w, float h, const PanelDrawContext& ctx,
                           float clip_y_min, float clip_y_max,
                           const PanelInputState* input) override;
        void drawDirect(float x, float y, float w, float h, const PanelDrawContext& ctx) override;
        float getDirectDrawHeight() const override { return direct_draw_height_; }
        void setInput(const PanelInputState* input) override { input_ = input; }
        void setForcedHeight(float h) override { forced_height_ = h; }
        bool wantsExternalFloatingShadow() const override { return false; }
        void setPanelSpace(PanelSpace space) override { is_floating_ = space == PanelSpace::Floating; }

    private:
        SequencerUIManager* seq_;
        const PanelLayoutManager* layout_;
        const PanelInputState* input_ = nullptr;
        float direct_draw_height_ = 0.0f;
        float forced_height_ = 0.0f;
        bool is_floating_ = false;
    };

    class NodeTransformGizmoPanel : public IPanel {
    public:
        explicit NodeTransformGizmoPanel(GizmoManager* gizmo);
        void draw(const PanelDrawContext& ctx) override;

    private:
        GizmoManager* gizmo_;
    };

    class CropBoxGizmoPanel : public IPanel {
    public:
        explicit CropBoxGizmoPanel(GizmoManager* gizmo);
        void draw(const PanelDrawContext& ctx) override;

    private:
        GizmoManager* gizmo_;
    };

    class EllipsoidGizmoPanel : public IPanel {
    public:
        explicit EllipsoidGizmoPanel(GizmoManager* gizmo);
        void draw(const PanelDrawContext& ctx) override;

    private:
        GizmoManager* gizmo_;
    };

    class ViewportGizmoPanel : public IPanel {
    public:
        explicit ViewportGizmoPanel(GizmoManager* gizmo);
        void draw(const PanelDrawContext& ctx) override;
        bool poll(const PanelDrawContext& ctx) override;

    private:
        GizmoManager* gizmo_;
    };

    class PieMenuPanel : public IPanel {
    public:
        explicit PieMenuPanel(GizmoManager* gizmo);
        void draw(const PanelDrawContext& ctx) override;
        bool poll(const PanelDrawContext& ctx) override;

    private:
        GizmoManager* gizmo_;
    };

    class PythonOverlayPanel : public IPanel {
    public:
        explicit PythonOverlayPanel(GuiManager* gui);
        void draw(const PanelDrawContext& ctx) override;
        bool poll(const PanelDrawContext& ctx) override;

    private:
        GuiManager* gui_;
    };

} // namespace lfs::vis::gui::native_panels
