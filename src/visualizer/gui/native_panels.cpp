/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/native_panels.hpp"
#include "gui/gizmo_manager.hpp"
#include "gui/gui_manager.hpp"
#include "gui/panel_layout.hpp"
#include "gui/panel_registry.hpp"
#include "gui/rml_status_bar.hpp"
#include "gui/sequencer_ui_manager.hpp"
#include "gui/startup_overlay.hpp"
#include "internal/viewport.hpp"
#include "python/python_runtime.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "rendering/rendering_manager.hpp"
#include "visualizer/gui/video_widget_interface.hpp"
#include "visualizer_impl.hpp"

#include <glm/gtc/type_ptr.hpp>
#include <imgui.h>

namespace lfs::vis::gui::native_panels {

    VideoExtractorPanel::VideoExtractorPanel(lfs::gui::IVideoExtractorWidget* widget)
        : widget_(widget) {}

    void VideoExtractorPanel::draw(const PanelDrawContext& ctx) {
        (void)ctx;
        if (!widget_ || !widget_->render())
            PanelRegistry::instance().set_panel_enabled("native.video_extractor", false);
    }

    StartupOverlayPanel::StartupOverlayPanel(StartupOverlay* overlay, const bool* drag_hovering)
        : overlay_(overlay),
          drag_hovering_(drag_hovering) {}

    void StartupOverlayPanel::draw(const PanelDrawContext& ctx) {
        if (ctx.viewport)
            overlay_->render(*ctx.viewport, drag_hovering_ ? *drag_hovering_ : false);
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
        const bool is_enabled = !ctx.ui_hidden && ctx.ui && ctx.ui->editor &&
                                !ctx.ui->editor->isToolsDisabled() && layout_->isShowSequencer();
        if (!is_enabled && seq_)
            seq_->setSequencerEnabled(false);
        return is_enabled;
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

    PieMenuPanel::PieMenuPanel(GizmoManager* gizmo)
        : gizmo_(gizmo) {}

    void PieMenuPanel::draw(const PanelDrawContext&) {
        gizmo_->renderPieMenu();
    }

    bool PieMenuPanel::poll(const PanelDrawContext&) {
        return gizmo_->isPieMenuOpen();
    }

    PythonOverlayPanel::PythonOverlayPanel(GuiManager* gui)
        : gui_(gui) {}

    bool PythonOverlayPanel::poll(const PanelDrawContext& ctx) {
        if (gui_ && gui_->isStartupVisible()) {
            return false;
        }
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
        const glm::vec3 forward = lfs::rendering::cameraForward(vp.camera.R);
        const float cam_fwd[] = {forward.x, forward.y, forward.z};

        python::invoke_viewport_overlay(glm::value_ptr(view), glm::value_ptr(proj),
                                        vp_pos, vp_size, cam_pos, cam_fwd,
                                        ImGui::GetBackgroundDrawList());
    }

} // namespace lfs::vis::gui::native_panels
