/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/native_panels.hpp"
#include "gui/gizmo_manager.hpp"
#include "gui/gui_manager.hpp"
#include "gui/panel_layout.hpp"
#include "gui/panel_registry.hpp"
#include "gui/sequencer_ui_manager.hpp"
#include "gui/startup_overlay.hpp"
#include "gui/windows/file_browser.hpp"
#include "internal/viewport.hpp"
#include "python/python_runtime.hpp"
#include "rendering/rendering_manager.hpp"
#include "visualizer_impl.hpp"
#include "windows/disk_space_error_dialog.hpp"
#include "windows/video_extractor_dialog.hpp"

#include <glm/gtc/type_ptr.hpp>
#include <imgui.h>

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

    VideoExtractorPanel::VideoExtractorPanel(lfs::gui::VideoExtractorDialog* dialog)
        : dialog_(dialog) {}

    void VideoExtractorPanel::draw(const PanelDrawContext& ctx) {
        (void)ctx;
        if (!dialog_->render())
            PanelRegistry::instance().set_panel_enabled("native.video_extractor", false);
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

    PythonOverlayPanel::PythonOverlayPanel(GuiManager* gui)
        : gui_(gui) {}

    bool PythonOverlayPanel::poll(const PanelDrawContext& ctx) {
        return ctx.viewport && ctx.viewport->size.x > 0 && ctx.viewport->size.y > 0 &&
               python::has_viewport_draw_handlers();
    }

    void PythonOverlayPanel::draw(const PanelDrawContext& ctx) {
        if (!ctx.ui || !ctx.ui->viewer || !ctx.viewport)
            return;

        const auto& vp = ctx.ui->viewer->getViewport();
        const auto view = vp.getViewMatrix();
        auto* rm = ctx.ui->viewer->getRenderingManager();
        const float focal_mm = rm ? rm->getFocalLengthMm() : lfs::rendering::DEFAULT_FOCAL_LENGTH_MM;
        const auto proj = vp.getProjectionMatrix(focal_mm);
        const float vp_pos[] = {ctx.viewport->pos.x, ctx.viewport->pos.y};
        const float vp_size[] = {ctx.viewport->size.x, ctx.viewport->size.y};
        const float cam_pos[] = {vp.camera.t.x, vp.camera.t.y, vp.camera.t.z};
        const float cam_fwd[] = {vp.camera.R[2].x, vp.camera.R[2].y, vp.camera.R[2].z};

        python::invoke_viewport_overlay(glm::value_ptr(view), glm::value_ptr(proj),
                                        vp_pos, vp_size, cam_pos, cam_fwd,
                                        ImGui::GetBackgroundDrawList());
    }

} // namespace lfs::vis::gui::native_panels
