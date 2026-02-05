/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/native_panels.hpp"
#include "gui/gizmo_manager.hpp"
#include "gui/gui_manager.hpp"
#include "gui/panel_layout.hpp"
#include "gui/sequencer_ui_manager.hpp"
#include "gui/startup_overlay.hpp"
#include "gui/windows/file_browser.hpp"
#include "windows/disk_space_error_dialog.hpp"
#include "windows/video_extractor_dialog.hpp"

namespace lfs::vis::gui::native_panels {

    FileBrowserPanel::FileBrowserPanel(FileBrowser* browser, bool* visible)
        : browser_(browser),
          visible_(visible) {}

    void FileBrowserPanel::draw(const PanelDrawContext& ctx) {
        (void)ctx;
        browser_->render(visible_);
    }

    bool FileBrowserPanel::poll(const PanelDrawContext& ctx) {
        (void)ctx;
        return visible_ && *visible_;
    }

    VideoExtractorPanel::VideoExtractorPanel(lfs::gui::VideoExtractorDialog* dialog, bool* visible)
        : dialog_(dialog),
          visible_(visible) {}

    void VideoExtractorPanel::draw(const PanelDrawContext& ctx) {
        (void)ctx;
        dialog_->render(visible_);
    }

    bool VideoExtractorPanel::poll(const PanelDrawContext& ctx) {
        (void)ctx;
        return visible_ && *visible_;
    }

    DiskSpaceErrorPanel::DiskSpaceErrorPanel(DiskSpaceErrorDialog* dialog)
        : dialog_(dialog) {}

    void DiskSpaceErrorPanel::draw(const PanelDrawContext& ctx) {
        (void)ctx;
        dialog_->render();
    }

    bool DiskSpaceErrorPanel::poll(const PanelDrawContext& ctx) {
        (void)ctx;
        return dialog_->isOpen();
    }

    StartupOverlayPanel::StartupOverlayPanel(StartupOverlay* overlay, ImFont* font, const bool* drag_hovering)
        : overlay_(overlay),
          font_(font),
          drag_hovering_(drag_hovering) {}

    void StartupOverlayPanel::draw(const PanelDrawContext& ctx) {
        if (ctx.viewport)
            overlay_->render(*ctx.viewport, font_, drag_hovering_ ? *drag_hovering_ : false);
    }

    bool StartupOverlayPanel::poll(const PanelDrawContext& ctx) {
        (void)ctx;
        return overlay_->isVisible();
    }

    SelectionOverlayPanel::SelectionOverlayPanel(GuiManager* gui)
        : gui_(gui) {}

    void SelectionOverlayPanel::draw(const PanelDrawContext& ctx) {
        if (ctx.ui)
            gui_->renderSelectionOverlays(*ctx.ui);
    }

    ViewportDecorationsPanel::ViewportDecorationsPanel(GuiManager* gui)
        : gui_(gui) {}

    void ViewportDecorationsPanel::draw(const PanelDrawContext& ctx) {
        (void)ctx;
        gui_->renderViewportDecorations();
    }

    SequencerPanel::SequencerPanel(SequencerUIManager* seq, const PanelLayoutManager* layout)
        : seq_(seq),
          layout_(layout) {}

    void SequencerPanel::draw(const PanelDrawContext& ctx) {
        if (ctx.ui && ctx.viewport)
            seq_->render(*ctx.ui, *ctx.viewport);
    }

    bool SequencerPanel::poll(const PanelDrawContext& ctx) {
        return !ctx.ui_hidden && ctx.ui && ctx.ui->editor &&
               !ctx.ui->editor->isToolsDisabled() && layout_->isShowSequencer();
    }

    NodeTransformGizmoPanel::NodeTransformGizmoPanel(GizmoManager* gizmo)
        : gizmo_(gizmo) {}

    void NodeTransformGizmoPanel::draw(const PanelDrawContext& ctx) {
        if (ctx.ui && ctx.viewport)
            gizmo_->renderNodeTransformGizmo(*ctx.ui, *ctx.viewport);
    }

    CropBoxGizmoPanel::CropBoxGizmoPanel(GizmoManager* gizmo)
        : gizmo_(gizmo) {}

    void CropBoxGizmoPanel::draw(const PanelDrawContext& ctx) {
        if (ctx.ui && ctx.viewport)
            gizmo_->renderCropBoxGizmo(*ctx.ui, *ctx.viewport);
    }

    EllipsoidGizmoPanel::EllipsoidGizmoPanel(GizmoManager* gizmo)
        : gizmo_(gizmo) {}

    void EllipsoidGizmoPanel::draw(const PanelDrawContext& ctx) {
        if (ctx.ui && ctx.viewport)
            gizmo_->renderEllipsoidGizmo(*ctx.ui, *ctx.viewport);
    }

    ViewportGizmoPanel::ViewportGizmoPanel(GizmoManager* gizmo)
        : gizmo_(gizmo) {}

    void ViewportGizmoPanel::draw(const PanelDrawContext& ctx) {
        if (ctx.viewport)
            gizmo_->renderViewportGizmo(*ctx.viewport);
    }

    bool ViewportGizmoPanel::poll(const PanelDrawContext& ctx) {
        return !ctx.ui_hidden && ctx.viewport &&
               ctx.viewport->size.x > 0 && ctx.viewport->size.y > 0;
    }

} // namespace lfs::vis::gui::native_panels
