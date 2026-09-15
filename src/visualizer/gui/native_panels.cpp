/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/native_panels.hpp"
#include "gui/gizmo_manager.hpp"
#include "gui/gui_manager.hpp"
#include "gui/line_renderer.hpp"
#include "gui/panel_input_utils.hpp"
#include "gui/panel_layout.hpp"
#include "gui/panel_registry.hpp"
#include "gui/rml_status_bar.hpp"
#include "gui/sequencer_ui_manager.hpp"
#include "gui/startup_overlay.hpp"
#include "internal/viewport.hpp"
#include "python/python_runtime.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "rendering/rendering.hpp"
#include "rendering/rendering_manager.hpp"
#include "rendering/screen_overlay_renderer.hpp"
#include "theme/theme.hpp"
#include "visualizer/gui/video_widget_interface.hpp"
#include "visualizer_impl.hpp"

#include <algorithm>
#include <glm/gtc/type_ptr.hpp>
#include <nlohmann/json.hpp>

namespace lfs::vis::gui::native_panels {

    VideoExtractorPanel::VideoExtractorPanel(lfs::gui::IVideoExtractorWidget* widget)
        : widget_(widget) {}

    void VideoExtractorPanel::draw(const PanelDrawContext& ctx) {
        (void)ctx;
        if (!widget_)
            PanelRegistry::instance().set_panel_enabled("native.video_extractor", false);
    }

    PanelRenderCapabilities VideoExtractorPanel::renderCapabilities() const {
        return {.direct = widget_ && widget_->supportsDirectDraw()};
    }

    PanelDirectRenderResult VideoExtractorPanel::renderDirect(
        const PanelDirectRenderRequest& request,
        const PanelDrawContext& ctx) {
        if (!widget_)
            return {};
        setPanelSpace(request.space);
        if (request.mode == PanelDirectRenderMode::Measure)
            return {.handled = true, .height = getDirectDrawHeight()};

        setInputClipY(request.clip_y_min, request.clip_y_max);
        setInput(request.input);
        setForcedHeight(request.forced_height);

        bool handled = true;
        try {
            switch (request.mode) {
            case PanelDirectRenderMode::Measure:
                break;
            case PanelDirectRenderMode::Draw:
                drawDirect(request.x, request.y, request.width, request.height, ctx);
                break;
            case PanelDirectRenderMode::Cached:
                handled = drawDirectCached(request.x, request.y, request.width,
                                           request.height, ctx);
                break;
            case PanelDirectRenderMode::Preload:
                preloadDirect(request.width, request.height, ctx,
                              request.clip_y_min, request.clip_y_max, request.input);
                break;
            }
        } catch (...) {
            setForcedHeight(0.0f);
            setInput(nullptr);
            setInputClipY(-1.0f, -1.0f);
            throw;
        }

        const float height = getDirectDrawHeight();
        setForcedHeight(0.0f);
        setInput(nullptr);
        setInputClipY(-1.0f, -1.0f);
        return {.handled = handled, .height = height};
    }

    void VideoExtractorPanel::preloadDirect(const float w, const float h,
                                            const PanelDrawContext& ctx,
                                            const float clip_y_min,
                                            const float clip_y_max,
                                            const PanelInputState* input) {
        if (widget_)
            widget_->preloadDirect(w, h, ctx, clip_y_min, clip_y_max, input);
    }

    void VideoExtractorPanel::drawDirect(const float x, const float y,
                                         const float w, const float h,
                                         const PanelDrawContext& ctx) {
        if (widget_)
            widget_->drawDirect(x, y, w, h, ctx);
    }

    bool VideoExtractorPanel::drawDirectCached(const float x, const float y,
                                               const float w, const float h,
                                               const PanelDrawContext& ctx) {
        return widget_ && widget_->drawDirectCached(x, y, w, h, ctx);
    }

    float VideoExtractorPanel::getDirectDrawHeight() const {
        return widget_ ? widget_->getDirectDrawHeight() : 0.0f;
    }

    void VideoExtractorPanel::setInputClipY(const float y_min, const float y_max) {
        if (widget_)
            widget_->setInputClipY(y_min, y_max);
    }

    void VideoExtractorPanel::setInput(const PanelInputState* input) {
        if (widget_)
            widget_->setInput(input);
    }

    void VideoExtractorPanel::setForcedHeight(const float h) {
        if (widget_)
            widget_->setForcedHeight(h);
    }

    void VideoExtractorPanel::setPanelSpace(const PanelSpace space) {
        if (widget_)
            widget_->setFloating(space == PanelSpace::Floating);
    }

    bool VideoExtractorPanel::needsAnimationFrame() const {
        return widget_ && widget_->needsAnimationFrame();
    }

    void VideoExtractorPanel::reloadRmlResources() {
        if (widget_)
            widget_->reloadRmlResources();
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
        gui_->renderViewportDecorations();
        (void)ctx;
    }

    namespace {
        // Area instances deliberately share only the sequencer controller. RML
        // context, timeline view, input state, film-strip textures, and panel
        // cache belong to this retained instance.
        class SequencerAreaPanel final : public IPanel {
        public:
            SequencerAreaPanel(SequencerUIManager* manager, const std::string_view instance_id)
                : manager_(manager),
                  ui_state_(manager->uiState()),
                  panel_(manager->controller(), ui_state_, manager->rmlManager(),
                         std::string(instance_id) + ".sequencer") {
                panel_.setAreaHosted(true);
                owner_ = manager_->createAreaOverlayOwner(&panel_, &ui_state_);
            }

            ~SequencerAreaPanel() override {
                if (manager_)
                    manager_->releaseAreaOverlayOwner(owner_);
            }

            void draw(const PanelDrawContext& /*ctx*/) override {}

            bool poll(const PanelDrawContext& ctx) override {
                return !ctx.ui_hidden && manager_ != nullptr;
            }

            PanelRenderCapabilities renderCapabilities() const override { return {.direct = true}; }

            PanelDirectRenderResult renderDirect(const PanelDirectRenderRequest& request,
                                                 const PanelDrawContext& /*ctx*/) override {
                if (request.mode == PanelDirectRenderMode::Measure)
                    return {.handled = true, .height = request.height};
                if (request.mode == PanelDirectRenderMode::Cached)
                    return {};
                if (!manager_ || !request.input || request.width <= 0.0f || request.height <= 0.0f)
                    return {.handled = true, .height = request.height};

                panel_.setFloating(request.space == PanelSpace::Floating);
                panel_.setFilmStripAttached(ui_state_.show_film_strip);
                if (request.mode == PanelDirectRenderMode::Draw) {
                    panel_.render(request.x, request.y, request.width, request.height,
                                  toSequencerPanelInput(*request.input), manager_->viewer()->getRenderingManager(),
                                  manager_->viewer()->getSceneManager(), film_strip_);
                    manager_->processAreaPanelRequests(panel_, ui_state_, owner_,
                                                       request.input->mouse_x,
                                                       request.input->mouse_y);
                }
                return {.handled = true, .height = request.height};
            }

            bool needsAnimationFrame() const override {
                return manager_ && (manager_->controller().isPlaying() ||
                                    panel_.needsLocalizationFrame());
            }

            void reloadRmlResources() override { panel_.reloadResources(); }

            void releaseRendererResources() override {
                panel_.destroyGraphicsResources();
                film_strip_.destroyGraphicsResources();
            }

            [[nodiscard]] std::string captureChromeJson() const override {
                return nlohmann::json{
                    {"timeline_zoom", panel_.zoomLevel()},
                    {"timeline_pan", panel_.panOffset()},
                    {"show_film_strip", ui_state_.show_film_strip},
                }
                    .dump();
            }

            void applyChromeJson(const std::string_view json) override {
                if (json.empty())
                    return;
                try {
                    const auto chrome = nlohmann::json::parse(json);
                    panel_.setTimelineView(chrome.value("timeline_zoom", 1.0f),
                                           chrome.value("timeline_pan", 0.0f));
                    ui_state_.show_film_strip = chrome.value(
                        "show_film_strip", ui_state_.show_film_strip);
                } catch (const std::exception&) {
                    // Chrome is optional presentation state; malformed data
                    // must leave the freshly constructed instance usable.
                }
            }

        private:
            SequencerUIManager* manager_ = nullptr;
            panels::SequencerUIState ui_state_;
            RmlSequencerPanel panel_;
            FilmStripRenderer film_strip_;
            SequencerUIManager::AreaOverlayOwnerPtr owner_;
        };
    } // namespace

    SequencerPanel::SequencerPanel(SequencerUIManager* seq, const PanelLayoutManager* layout)
        : seq_(seq),
          layout_(layout) {}

    std::shared_ptr<IPanel> SequencerPanel::createAreaInstance(
        const std::string_view instance_id) const {
        if (!seq_ || !seq_->rmlManager())
            return nullptr;
        return std::make_shared<SequencerAreaPanel>(seq_, instance_id);
    }

    void SequencerPanel::draw(const PanelDrawContext& ctx) {
        (void)ctx;
    }

    PanelDirectRenderResult SequencerPanel::renderDirect(
        const PanelDirectRenderRequest& request,
        const PanelDrawContext& ctx) {
        is_floating_ = request.space == PanelSpace::Floating;
        if (request.mode == PanelDirectRenderMode::Measure)
            return {.handled = true, .height = direct_draw_height_};

        input_ = request.input;
        forced_height_ = request.forced_height;

        bool handled = true;
        switch (request.mode) {
        case PanelDirectRenderMode::Measure:
            break;
        case PanelDirectRenderMode::Draw:
            drawDirect(request.x, request.y, request.width, request.height, ctx);
            break;
        case PanelDirectRenderMode::Cached:
            handled = false;
            break;
        case PanelDirectRenderMode::Preload:
            preloadDirect(request.width, request.height, ctx,
                          request.clip_y_min, request.clip_y_max, request.input);
            break;
        }

        input_ = nullptr;
        forced_height_ = 0.0f;
        return {.handled = handled, .height = direct_draw_height_};
    }

    void SequencerPanel::preloadDirect(const float w, const float h,
                                       const PanelDrawContext& ctx,
                                       const float clip_y_min,
                                       const float clip_y_max,
                                       const PanelInputState* input) {
        (void)w;
        (void)ctx;
        (void)clip_y_min;
        (void)clip_y_max;
        input_ = input;

        if (seq_)
            seq_->setFloating(is_floating_);

        if (is_floating_) {
            const float preferred_h = seq_ ? seq_->preferredFloatingHeight() : 0.0f;
            direct_draw_height_ = forced_height_ > 0.0f
                                      ? forced_height_
                                      : std::min(h, std::max(0.0f, preferred_h));
        } else {
            direct_draw_height_ = h;
        }
    }

    void SequencerPanel::drawDirect(const float x, const float y,
                                    const float w, const float h,
                                    const PanelDrawContext& ctx) {
        if (seq_)
            seq_->setFloating(is_floating_);

        if (is_floating_) {
            direct_draw_height_ = seq_ ? std::max(0.0f, seq_->preferredFloatingHeight()) : h;
        } else {
            direct_draw_height_ = h;
        }

        if (seq_ && ctx.ui && ctx.viewport && input_ && h > 0.0f)
            seq_->render(*ctx.ui, *ctx.viewport, x, y, w, h, *input_);
    }

    bool SequencerPanel::poll(const PanelDrawContext& ctx) {
        // The sequencer is a camera/animation timeline, not a scene-editing
        // tool, so it must not share the editing gizmos' isToolsDisabled() gate
        // (true for TRAINING/PAUSED/FINISHED). The gizmos are disabled across all
        // of those because the trainer owns the model tensors, but the sequencer
        // has no such dependency. Disable it only while training is actively
        // running; once training is paused/finished it should be usable without
        // having to switch to Edit mode (which tears the trainer down).
        const bool training_active = ctx.ui && ctx.ui->editor && ctx.ui->editor->isTraining();
        const bool is_enabled = !ctx.ui_hidden && ctx.ui && ctx.ui->editor &&
                                !training_active && layout_->isShowSequencer();
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
        return ctx.viewport &&
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

        auto* rm = ctx.ui->viewer->getRenderingManager();
        auto* overlay = rm ? rm->getScreenOverlayRenderer() : nullptr;
        const auto draw = [&](const glm::mat4& view, const glm::mat4& projection,
                              glm::vec2 position, glm::vec2 size,
                              glm::vec3 camera_position, glm::vec3 camera_forward) {
            const float vp_pos[] = {position.x, position.y};
            const float vp_size[] = {size.x, size.y};
            NativeOverlayDrawList draw_list;
            draw_list.PushClipRect(position, position + size);
            python::invoke_viewport_overlay(glm::value_ptr(view), glm::value_ptr(projection),
                                            vp_pos, vp_size, glm::value_ptr(camera_position),
                                            glm::value_ptr(camera_forward), overlay, &draw_list);
        };
        if (gui_ && gui_->usesAreaWorkspace()) {
            for (const auto& pane : gui_->workspaceSnapshot().panes) {
                const auto* vp = ctx.ui->viewer->getViewportWorkspace()->findCamera(pane.id);
                if (!vp)
                    continue;
                const auto& projection = pane.projection;
                draw(vp->getViewMatrix(),
                     lfs::rendering::createProjectionMatrixFromFocal(
                         pane.window_size, projection.focal_length_mm,
                         projection.orthographic, projection.ortho_scale,
                         projection.near_plane, projection.far_plane),
                     {pane.rect.x, pane.rect.y}, {pane.rect.width, pane.rect.height},
                     vp->camera.t, lfs::rendering::cameraForward(vp->camera.R));
            }
        } else {
            const auto& vp = ctx.ui->viewer->getViewport();
            const float focal_mm = rm ? rm->getFocalLengthMm() : lfs::rendering::DEFAULT_FOCAL_LENGTH_MM;
            draw(vp.getViewMatrix(), vp.getProjectionMatrix(focal_mm),
                 ctx.viewport->pos, ctx.viewport->size,
                 vp.camera.t, lfs::rendering::cameraForward(vp.camera.R));
        }
    }

} // namespace lfs::vis::gui::native_panels
