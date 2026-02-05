/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/panel_registry.hpp"

struct ImFont;

namespace lfs::vis::gui {

    class FileBrowser;
    class DiskSpaceErrorDialog;
    class StartupOverlay;
    class GizmoManager;
    class GuiManager;
    class SequencerUIManager;
    class PanelLayoutManager;

} // namespace lfs::vis::gui

namespace lfs::gui {
    class VideoExtractorDialog;
}

namespace lfs::vis::gui::native_panels {

    class FileBrowserPanel : public IPanel {
    public:
        FileBrowserPanel(FileBrowser* browser, bool* visible);
        void draw(const PanelDrawContext& ctx) override;
        bool poll(const PanelDrawContext& ctx) override;

    private:
        FileBrowser* browser_;
        bool* visible_;
    };

    class VideoExtractorPanel : public IPanel {
    public:
        VideoExtractorPanel(lfs::gui::VideoExtractorDialog* dialog, bool* visible);
        void draw(const PanelDrawContext& ctx) override;
        bool poll(const PanelDrawContext& ctx) override;

    private:
        lfs::gui::VideoExtractorDialog* dialog_;
        bool* visible_;
    };

    class DiskSpaceErrorPanel : public IPanel {
    public:
        explicit DiskSpaceErrorPanel(DiskSpaceErrorDialog* dialog);
        void draw(const PanelDrawContext& ctx) override;
        bool poll(const PanelDrawContext& ctx) override;

    private:
        DiskSpaceErrorDialog* dialog_;
    };

    class StartupOverlayPanel : public IPanel {
    public:
        StartupOverlayPanel(StartupOverlay* overlay, ImFont* font, const bool* drag_hovering);
        void draw(const PanelDrawContext& ctx) override;
        bool poll(const PanelDrawContext& ctx) override;

    private:
        StartupOverlay* overlay_;
        ImFont* font_;
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

    private:
        SequencerUIManager* seq_;
        const PanelLayoutManager* layout_;
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

} // namespace lfs::vis::gui::native_panels
