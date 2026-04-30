/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/sequencer_ui_manager.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "gui/gui_focus_state.hpp"
#include "gui/panel_input_utils.hpp"
#include "gui/rml_sequencer_overlay.hpp"
#include "gui/rotation_gizmo.hpp"
#include "gui/string_keys.hpp"
#include "gui/translation_gizmo.hpp"
#include "gui/utils/native_file_dialog.hpp"
#include "io/video/video_export_options.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "rendering/rendering.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "sequencer/interpolation.hpp"
#include "sequencer/keyframe.hpp"
#include "sequencer/timeline_view_math.hpp"
#include "theme/theme.hpp"
#include "visualizer_impl.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <string_view>
#include <vector>

namespace lfs::vis::gui {

    namespace {
        constexpr size_t MIN_PATH_RENDER_SAMPLES = 128;
        constexpr size_t MAX_PATH_RENDER_SAMPLES = 4096;
        constexpr float PATH_SAMPLES_PER_VIEWPORT_PIXEL = 2.0f;

    } // namespace

    SequencerUIManager::SequencerUIManager(VisualizerImpl* viewer, panels::SequencerUIState& ui_state,
                                           gui::RmlUIManager* rml_manager)
        : viewer_(viewer),
          ui_state_(ui_state),
          panel_(std::make_unique<RmlSequencerPanel>(controller_, ui_state_, rml_manager)),
          overlay_(std::make_unique<RmlSequencerOverlay>(controller_, rml_manager)),
          scene_sync_(std::make_unique<KeyframeSceneSync>(controller_, viewer)) {}

    SequencerUIManager::~SequencerUIManager() = default;

    void SequencerUIManager::destroyGraphicsResources() {
        if (panel_)
            panel_->destroyGraphicsResources();
        if (overlay_)
            overlay_->destroyGraphicsResources();
        pip_texture_.reset();
        pip_initialized_ = false;
        line_renderer_.destroyResources();
        film_strip_.destroyGraphicsResources();
    }

    void SequencerUIManager::reloadRmlResources() {
        if (panel_)
            panel_->reloadResources();
        if (overlay_)
            overlay_->reloadResources();
        pip_needs_update_ = true;
    }

    void SequencerUIManager::setSequencerEnabled(const bool enabled) {
        if (enabled)
            return;

        if (panel_)
            panel_->clearPendingComposite();

        if (ui_state_.show_pip_preview)
            ui_state_.show_pip_preview = false;

        viewport_edit_mode_ = SequencerViewportEditMode::None;
        keyframe_gizmo_active_ = false;
        pip_last_keyframe_ = std::nullopt;
        pip_needs_update_ = true;
        last_panel_frame_time_ = std::chrono::steady_clock::now();
        endViewportKeyframeEdit();
    }

    void SequencerUIManager::beginViewportKeyframeEdit(const size_t keyframe_index) {
        const auto* const keyframe = controller_.timeline().getKeyframe(keyframe_index);
        if (!keyframe || keyframe->is_loop_point)
            return;

        controller_.selectKeyframe(keyframe_index);
        if (auto* sm = viewer_->getSceneManager())
            sm->clearSelection();
        viewport_keyframe_edit_snapshot_ = *keyframe;
        viewport_edit_mode_ = SequencerViewportEditMode::None;
        keyframe_gizmo_active_ = false;
        edit_entered_mouse_down_ = true;
    }

    void SequencerUIManager::endViewportKeyframeEdit() {
        viewport_keyframe_edit_snapshot_ = std::nullopt;
        if (overlay_)
            overlay_->hideEditOverlay();
    }

    sequencer::CameraState SequencerUIManager::currentViewportCameraState() const {
        const auto& cam = viewer_->getViewport().camera;
        auto* const rm = viewer_->getRenderingManager();

        return {
            .position = cam.t,
            .rotation = glm::quat_cast(cam.R),
            .focal_length_mm = rm ? rm->getFocalLengthMm() : lfs::rendering::DEFAULT_FOCAL_LENGTH_MM};
    }

    void SequencerUIManager::restoreViewportCameraState(const sequencer::CameraState& state) const {
        auto& vp = viewer_->getViewport();
        vp.camera.R = glm::mat3_cast(state.rotation);
        vp.camera.t = state.position;

        if (auto* const rm = viewer_->getRenderingManager()) {
            rm->setFocalLength(state.focal_length_mm);
            rm->markDirty(DirtyFlag::CAMERA);
        }
    }

    void SequencerUIManager::setupEvents() {
        using namespace lfs::core::events;

        ui::RenderSettingsChanged::when([this](const auto& event) {
            if (event.equirectangular)
                ui_state_.equirectangular = *event.equirectangular;
        });

        cmd::SequencerAddKeyframe::when([this](const auto&) {
            const auto& cam = viewer_->getViewport().camera;

            const float interval = ui_state_.snap_to_grid ? ui_state_.snap_interval : 1.0f;
            const float time = controller_.timeline().realKeyframeCount() == 0
                                   ? 0.0f
                                   : controller_.timeline().realEndTime() + interval;

            auto* const rm = viewer_->getRenderingManager();
            const float focal_mm = rm ? rm->getFocalLengthMm() : lfs::rendering::DEFAULT_FOCAL_LENGTH_MM;

            lfs::sequencer::Keyframe kf;
            kf.time = time;
            kf.position = cam.t;
            kf.rotation = glm::quat_cast(cam.R);
            kf.focal_length_mm = focal_mm;
            controller_.addKeyframeAtTime(kf, time);
            controller_.seek(time);
            state::KeyframeListChanged{.count = controller_.timeline().realKeyframeCount()}.emit();
        });

        cmd::SequencerUpdateKeyframe::when([this](const auto&) {
            if (!controller_.hasSelection())
                return;
            const auto& cam = viewer_->getViewport().camera;
            auto* const rm = viewer_->getRenderingManager();
            const float focal_mm = rm ? rm->getFocalLengthMm() : lfs::rendering::DEFAULT_FOCAL_LENGTH_MM;
            controller_.updateSelectedKeyframe(
                cam.t,
                glm::quat_cast(cam.R),
                focal_mm);
            if (viewport_keyframe_edit_snapshot_.has_value() &&
                controller_.selectedKeyframeId().has_value() &&
                *controller_.selectedKeyframeId() ==
                    viewport_keyframe_edit_snapshot_->id) {
                viewport_keyframe_edit_snapshot_->position = cam.t;
                viewport_keyframe_edit_snapshot_->rotation =
                    glm::quat_cast(cam.R);
                viewport_keyframe_edit_snapshot_->focal_length_mm = focal_mm;
            } else {
                endViewportKeyframeEdit();
            }
            state::KeyframeListChanged{.count = controller_.timeline().realKeyframeCount()}.emit();
        });

        cmd::SequencerGoToKeyframe::when([this](const auto&) {
            endViewportKeyframeEdit();
        });

        cmd::SequencerPlayPause::when([this](const auto&) {
            controller_.togglePlayPause();
        });

        state::KeyframeListChanged::when([this](const auto&) {
            film_strip_.invalidateAll();
        });

        ui::NodeSelected::when([this](const auto& e) {
            if (e.type != "KEYFRAME") {
                viewport_edit_mode_ = SequencerViewportEditMode::None;
                keyframe_gizmo_active_ = false;
                endViewportKeyframeEdit();
            }
        });

        scene_sync_->setupEvents();
    }

    void SequencerUIManager::setFloating(const bool floating) {
        if (panel_)
            panel_->setFloating(floating);
    }

    float SequencerUIManager::preferredFloatingHeight() const {
        const float dp = std::max(1.0f, getThemeDpiScale());
        const float strip_h = ui_state_.show_film_strip ? FilmStripRenderer::STRIP_HEIGHT : 0.0f;
        return (panel_config::HEIGHT + panel_config::EASING_STRIPE_HEIGHT) * dp + strip_h;
    }

    void SequencerUIManager::render(const UIContext& ctx, const ViewportLayout& viewport,
                                    const float panel_x, const float panel_y,
                                    const float panel_width, const float panel_height,
                                    const PanelInputState& panel_input) {
        const auto* const gui = viewer_->getGuiManager();
        const bool sequencer_enabled = gui && gui->panelLayout().isShowSequencer();
        if (!sequencer_enabled) {
            setSequencerEnabled(false);
            return;
        }

        if (ui_state_.equirectangular != last_equirectangular_) {
            last_equirectangular_ = ui_state_.equirectangular;
            pip_needs_update_ = true;
            film_strip_.invalidateAll();
        }

        const auto& sdl_buf = viewer_->getWindowManager()->frameInput();
        lfs::vis::PanelInputState overlay_input;
        overlay_input.mouse_x = sdl_buf.mouse_x;
        overlay_input.mouse_y = sdl_buf.mouse_y;
        overlay_input.mouse_down[0] = sdl_buf.mouse_down[0];
        overlay_input.mouse_down[1] = sdl_buf.mouse_down[1];
        overlay_input.mouse_clicked[0] = sdl_buf.mouse_clicked[0];
        overlay_input.mouse_clicked[1] = sdl_buf.mouse_clicked[1];
        overlay_input.mouse_released[0] = sdl_buf.mouse_released[0];
        overlay_input.mouse_released[1] = sdl_buf.mouse_released[1];
        overlay_input.key_ctrl = (sdl_buf.key_mods & SDL_KMOD_CTRL) != 0;
        overlay_input.key_shift = (sdl_buf.key_mods & SDL_KMOD_SHIFT) != 0;
        overlay_input.key_alt = (sdl_buf.key_mods & SDL_KMOD_ALT) != 0;
        overlay_input.key_super = (sdl_buf.key_mods & SDL_KMOD_GUI) != 0;
        for (auto sc : sdl_buf.keys_pressed)
            overlay_input.keys_pressed.push_back(static_cast<int>(sc));
        for (auto sc : sdl_buf.keys_released)
            overlay_input.keys_released.push_back(static_cast<int>(sc));
        overlay_input.text_codepoints = sdl_buf.text_codepoints;
        overlay_input.text_inputs = sdl_buf.text_inputs;
        overlay_input.text_editing = sdl_buf.text_editing;
        overlay_input.text_editing_start = sdl_buf.text_editing_start;
        overlay_input.text_editing_length = sdl_buf.text_editing_length;
        overlay_input.has_text_editing = sdl_buf.has_text_editing;

        renderKeyframeEditOverlay(viewport);
        overlay_->processInput(overlay_input);
        handleOverlayActions();

        const bool overlay_active = overlay_->wantsInput() ||
                                    overlay_->isMouseOverEditOverlay(sdl_buf.mouse_x, sdl_buf.mouse_y);
        if (edit_entered_mouse_down_ && !sdl_buf.mouse_down[0])
            edit_entered_mouse_down_ = false;
        if (overlay_active || edit_entered_mouse_down_)
            guiFocusState().want_capture_mouse = true;

        const bool actively_following =
            ui_state_.follow_playback && controller_.isPlaying() &&
            controller_.timeline().realKeyframeCount() > 0;

        if (ui_state_.show_camera_path && !actively_following) {
            renderCameraPath(viewport);
            renderKeyframeGizmo(ctx, viewport);
        }
        renderKeyframePreview(ctx);
        renderSequencerPanel(ctx, viewport, panel_x, panel_y, panel_width, panel_height, panel_input);
        syncPipPreviewWindow(viewport);

        overlay_->render(sdl_buf.window_w, sdl_buf.window_h);
    }

    void SequencerUIManager::compositeOverlays(const int screen_w, const int screen_h) {
        if (panel_)
            panel_->compositeToScreen(screen_w, screen_h);
        if (overlay_)
            overlay_->compositeToScreen(screen_w, screen_h);
    }

    bool SequencerUIManager::blocksPointer(const double x, const double y) const {
        return overlay_ &&
               (overlay_->wantsInput() || overlay_->isMouseOverEditOverlay(static_cast<float>(x),
                                                                           static_cast<float>(y)));
    }

    bool SequencerUIManager::blocksKeyboard() const {
        return overlay_ && (overlay_->isContextMenuOpen() || overlay_->isPopupOpen());
    }

    void SequencerUIManager::renderSequencerPanel(const UIContext& /*ctx*/, const ViewportLayout& viewport,
                                                  const float panel_x, const float panel_y,
                                                  const float panel_width, const float panel_height,
                                                  const PanelInputState& panel_input) {
        (void)viewport;
        const auto now = std::chrono::steady_clock::now();
        float delta_time = std::chrono::duration<float>(now - last_panel_frame_time_).count();
        last_panel_frame_time_ = now;
        if (!std::isfinite(delta_time) || delta_time < 0.0f)
            delta_time = 0.0f;
        delta_time = std::min(delta_time, 0.1f);
        panel_elapsed_time_ += delta_time;

        controller_.update(delta_time);

        const bool is_playing = controller_.isPlaying() && controller_.timeline().realKeyframeCount() > 0;

        if (auto* const rm = viewer_->getRenderingManager()) {
            rm->setOverlayAnimationActive(is_playing);
            if (is_playing && ui_state_.follow_playback && !ui_state_.show_pip_preview) {
                rm->markDirty(DirtyFlag::CAMERA);
                const auto state = controller_.currentCameraState();
                auto& vp = viewer_->getViewport();
                vp.camera.R = glm::mat3_cast(state.rotation);
                vp.camera.t = state.position;
                rm->setFocalLength(state.focal_length_mm);
            }
        }

        panel_->setFilmStripAttached(ui_state_.show_film_strip);

        panel_input_ = toSequencerPanelInput(panel_input);
        panel_input_.time = panel_elapsed_time_;
        panel_input_.delta_time = delta_time;
        panel_input_.want_capture_mouse = guiFocusState().want_capture_mouse;

        panel_->render(panel_x, panel_y, panel_width, panel_height, panel_input_,
                       viewer_->getRenderingManager(), viewer_->getSceneManager(), film_strip_);

        if (panel_->isHovered())
            guiFocusState().want_capture_mouse = true;
        if (panel_->wantsKeyboard())
            guiFocusState().want_capture_keyboard = true;

        const auto timeline_menu = panel_->consumeContextMenu();
        if (timeline_menu.open) {
            overlay_->showContextMenu(panel_input_.mouse_x, panel_input_.mouse_y,
                                      timeline_menu.keyframe, timeline_menu.time, viewport_edit_mode_);
        }

        const auto time_req = panel_->consumeTimeEditRequest();
        if (time_req.active)
            overlay_->showTimeEdit(time_req.keyframe_index, time_req.current_time);

        const auto focal_req = panel_->consumeFocalEditRequest();
        if (focal_req.active)
            overlay_->showFocalEdit(focal_req.keyframe_index, focal_req.current_focal_mm);

        if (panel_->consumeSavePathRequest()) {
            const auto path = gui::SaveJsonFileDialog("camera_path");
            if (!path.empty()) {
                const std::string path_utf8 = lfs::core::path_to_utf8(path);
                if (controller_.saveToJson(path_utf8))
                    LOG_INFO("Camera path saved to {}", path_utf8);
                else
                    LOG_ERROR("Failed to save camera path to {}", path_utf8);
            }
        }

        if (panel_->consumeLoadPathRequest()) {
            const auto path = gui::OpenJsonFileDialog();
            if (!path.empty()) {
                const std::string path_utf8 = lfs::core::path_to_utf8(path);
                if (controller_.loadFromJson(path_utf8)) {
                    LOG_INFO("Camera path loaded from {}", path_utf8);
                    lfs::core::events::state::KeyframeListChanged{
                        .count = controller_.timeline().realKeyframeCount()}
                        .emit();
                    pip_needs_update_ = true;
                } else {
                    LOG_ERROR("Failed to load camera path from {}", path_utf8);
                }
            }
        }

        if (panel_->consumeDockToggleRequest()) {
            const PanelSpace target = panel_->isFloating() ? PanelSpace::BottomDock : PanelSpace::Floating;
            if (!PanelRegistry::instance().set_panel_space("native.sequencer", target)) {
                LOG_ERROR("Failed to move sequencer panel to {}",
                          target == PanelSpace::Floating ? "floating" : "bottom dock");
            }
        }

        if (panel_->consumeClosePanelRequest()) {
            if (auto* const gui = viewer_->getGuiManager())
                gui->panelLayout().setShowSequencer(false);
            setSequencerEnabled(false);
        }

        if (panel_->consumeExportRequest() && controller_.timeline().realKeyframeCount() > 0) {
            const auto info = lfs::io::video::getPresetInfo(ui_state_.preset);
            const int w = ui_state_.preset == lfs::io::video::VideoPreset::CUSTOM
                              ? ui_state_.custom_width
                              : info.width;
            const int h = ui_state_.preset == lfs::io::video::VideoPreset::CUSTOM
                              ? ui_state_.custom_height
                              : info.height;
            lfs::core::events::cmd::SequencerExportVideo{
                .width = w,
                .height = h,
                .framerate = ui_state_.framerate,
                .crf = ui_state_.quality}
                .emit();
        }

        if (panel_->consumeClearRequest() &&
            (controller_.timeline().realKeyframeCount() > 0 || controller_.timeline().hasAnimationClip())) {
            controller_.clear();
            lfs::core::events::state::KeyframeListChanged{.count = 0}.emit();
            LOG_INFO("All keyframes cleared");
        }

        auto ctx_req = panel_->consumeTransportContextMenu();
        if (ctx_req.target != TransportContextMenuRequest::Target::NONE) {
            auto& cm = viewer_->getGuiManager()->globalContextMenu();
            std::vector<gui::ContextMenuItem> items;

            using Target = TransportContextMenuRequest::Target;
            switch (ctx_req.target) {
            case Target::SNAP: {
                items.push_back({LOC("context_menu.snap_interval"), "", false, true});
                constexpr std::array<float, 4> snap_values = {0.25f, 0.5f, 1.0f, 2.0f};
                constexpr std::array<const char*, 4> snap_labels = {"0.25s", "0.5s", "1s", "2s"};
                for (size_t i = 0; i < snap_values.size(); ++i) {
                    bool active = std::abs(ui_state_.snap_interval - snap_values[i]) < 0.01f;
                    items.push_back({snap_labels[i],
                                     std::format("snap_{}", snap_values[i]),
                                     false, false, false, active});
                }
                break;
            }
            case Target::PREVIEW: {
                items.push_back({LOC("context_menu.preview_scale"), "", false, true});
                constexpr std::array<float, 5> scale_values = {0.5f, 0.75f, 1.0f, 1.5f, 2.0f};
                constexpr std::array<const char*, 5> scale_labels = {"0.5x", "0.75x", "1.0x", "1.5x", "2.0x"};
                for (size_t i = 0; i < scale_values.size(); ++i) {
                    bool active = std::abs(ui_state_.pip_preview_scale - scale_values[i]) < 0.01f;
                    items.push_back({scale_labels[i],
                                     std::format("scale_{}", scale_values[i]),
                                     false, false, false, active});
                }
                break;
            }
            case Target::FORMAT: {
                items.push_back({LOC("context_menu.video_format"), "", false, true});
                using lfs::io::video::VideoPreset;
                for (int p = 0; p <= static_cast<int>(VideoPreset::CUSTOM); ++p) {
                    const auto preset = static_cast<VideoPreset>(p);
                    const auto info = lfs::io::video::getPresetInfo(preset);
                    bool active = ui_state_.preset == preset;
                    items.push_back({info.name,
                                     std::format("preset_{}", p),
                                     false, false, false, active});
                }
                break;
            }
            case Target::CLEAR: {
                items.push_back({LOC("context_menu.clear_confirm"), "", false, true});
                items.push_back({LOC("context_menu.confirm"), "clear_confirm"});
                items.push_back({LOC("context_menu.cancel"), "clear_cancel"});
                break;
            }
            default:
                break;
            }

            if (!items.empty()) {
                const auto target = ctx_req.target;
                cm.request(std::move(items), ctx_req.screen_x, ctx_req.screen_y,
                           [this, target](std::string_view action) {
                               switch (target) {
                               case Target::SNAP:
                                   if (action.starts_with("snap_"))
                                       ui_state_.snap_interval = std::stof(std::string(action.substr(5)));
                                   break;
                               case Target::PREVIEW:
                                   if (action.starts_with("scale_"))
                                       ui_state_.pip_preview_scale = std::stof(std::string(action.substr(6)));
                                   break;
                               case Target::FORMAT:
                                   if (action.starts_with("preset_")) {
                                       using lfs::io::video::VideoPreset;
                                       const int idx = std::stoi(std::string(action.substr(7)));
                                       ui_state_.preset = static_cast<VideoPreset>(idx);
                                       const auto info = lfs::io::video::getPresetInfo(ui_state_.preset);
                                       ui_state_.custom_width = info.width;
                                       ui_state_.custom_height = info.height;
                                       ui_state_.framerate = info.framerate;
                                   }
                                   break;
                               case Target::CLEAR:
                                   if (action == "clear_confirm" &&
                                       (controller_.timeline().realKeyframeCount() > 0 || controller_.timeline().hasAnimationClip())) {
                                       controller_.clear();
                                       lfs::core::events::state::KeyframeListChanged{.count = 0}.emit();
                                       LOG_INFO("All keyframes cleared");
                                   }
                                   break;
                               case Target::NONE:
                                   break;
                               }
                           });
            }
        }
    }

    void SequencerUIManager::renderCameraPath(const ViewportLayout& viewport) {
        constexpr float PATH_THICKNESS = 2.0f;
        constexpr float PATH_SAMPLE_RADIUS = 2.5f;
        constexpr float FRUSTUM_THICKNESS = 1.5f;
        constexpr float NDC_CULL_MARGIN = 1.5f;
        constexpr size_t MAX_PATH_SAMPLE_MARKERS = 2000;
        constexpr float FRUSTUM_DEPTH = 0.25f;
        constexpr float SENSOR_ASPECT = rendering::SENSOR_WIDTH_35MM / rendering::SENSOR_HEIGHT_35MM;
        constexpr float HIT_RADIUS = 15.0f;

        const auto& timeline = controller_.timeline();
        if (timeline.empty())
            return;

        const auto& vp = viewer_->getViewport();
        auto* const rm = viewer_->getRenderingManager();
        if (!rm)
            return;
        const auto& settings = rm->getSettings();
        const glm::ivec2 vp_size(static_cast<int>(viewport.size.x), static_cast<int>(viewport.size.y));
        const auto* const rendering_manager = static_cast<const RenderingManager*>(rm);

        struct CameraPathPanel {
            SplitViewPanelId panel_id = SplitViewPanelId::Left;
            const Viewport* viewport = nullptr;
            glm::vec2 projection_pos{0.0f};
            glm::vec2 projection_size{0.0f};
            glm::ivec2 render_size{0};
            gui::ClipRect clip_rect{};

            [[nodiscard]] bool valid() const {
                return viewport != nullptr &&
                       projection_size.x > 0.0f &&
                       projection_size.y > 0.0f &&
                       render_size.x > 0 &&
                       render_size.y > 0 &&
                       clip_rect.width > 0 &&
                       clip_rect.height > 0;
            }

            [[nodiscard]] bool contains(const float x, const float y) const {
                return x >= static_cast<float>(clip_rect.x) &&
                       x <= static_cast<float>(clip_rect.x + clip_rect.width) &&
                       y >= static_cast<float>(clip_rect.y) &&
                       y <= static_cast<float>(clip_rect.y + clip_rect.height);
            }
        };

        std::vector<CameraPathPanel> panels;
        panels.reserve(2);

        const auto add_viewer_panel = [&](const std::optional<RenderingManager::ViewerPanelInfo>& info_opt) {
            if (!info_opt || !info_opt->valid())
                return;
            const auto& info = *info_opt;
            panels.push_back(CameraPathPanel{
                .panel_id = info.panel,
                .viewport = info.viewport,
                .projection_pos = {info.x, info.y},
                .projection_size = {info.width, info.height},
                .render_size = {info.render_width, info.render_height},
                .clip_rect = {
                    static_cast<int>(std::round(info.x)),
                    static_cast<int>(std::round(info.y)),
                    static_cast<int>(std::round(info.width)),
                    static_cast<int>(std::round(info.height)),
                },
            });
        };

        if (rm->isIndependentSplitViewActive()) {
            add_viewer_panel(rendering_manager->resolveViewerPanel(
                vp, viewport.pos, viewport.size, std::nullopt, SplitViewPanelId::Left));
            add_viewer_panel(rendering_manager->resolveViewerPanel(
                vp, viewport.pos, viewport.size, std::nullopt, SplitViewPanelId::Right));
        }

        if (panels.empty()) {
            const int clip_x = static_cast<int>(std::round(viewport.pos.x));
            const int clip_y = static_cast<int>(std::round(viewport.pos.y));
            const int clip_w = static_cast<int>(std::round(viewport.size.x));
            const int clip_h = static_cast<int>(std::round(viewport.size.y));
            std::vector<gui::ClipRect> clip_rects;
            clip_rects.reserve(2);

            if (const auto divider_x = rm->getSplitDividerScreenX(viewport.pos, viewport.size);
                divider_x.has_value()) {
                const int divider =
                    std::clamp(static_cast<int>(std::round(*divider_x)), clip_x, clip_x + clip_w);
                if (divider > clip_x)
                    clip_rects.push_back({clip_x, clip_y, divider - clip_x, clip_h});
                if (divider < clip_x + clip_w)
                    clip_rects.push_back({divider, clip_y, clip_x + clip_w - divider, clip_h});
            }

            if (clip_rects.empty())
                clip_rects.push_back({clip_x, clip_y, clip_w, clip_h});

            for (size_t i = 0; i < clip_rects.size(); ++i) {
                panels.push_back(CameraPathPanel{
                    .panel_id = (i == 0) ? SplitViewPanelId::Left : SplitViewPanelId::Right,
                    .viewport = &vp,
                    .projection_pos = viewport.pos,
                    .projection_size = viewport.size,
                    .render_size = vp_size,
                    .clip_rect = clip_rects[i],
                });
            }
        }

        if (panels.empty())
            return;

        if (viewport_edit_mode_ != SequencerViewportEditMode::None) {
            const auto selected = controller_.selectedKeyframe();
            const auto* const selected_keyframe =
                selected.has_value() && *selected < timeline.size()
                    ? timeline.getKeyframe(*selected)
                    : nullptr;
            if (!selected_keyframe || selected_keyframe->is_loop_point) {
                viewport_edit_mode_ = SequencerViewportEditMode::None;
                keyframe_gizmo_active_ = false;
            }
        }

        const auto projectToScreen = [&](const CameraPathPanel& panel,
                                         const glm::vec3& pos) -> glm::vec2 {
            const auto projected = lfs::rendering::projectWorldPoint(
                panel.viewport->getRotationMatrix(),
                panel.viewport->getTranslation(),
                panel.render_size,
                pos,
                settings.focal_length_mm,
                settings.orthographic,
                settings.ortho_scale);
            if (!projected)
                return {-10000.0f, -10000.0f};
            const float scale_x =
                panel.projection_size.x / static_cast<float>(std::max(panel.render_size.x, 1));
            const float scale_y =
                panel.projection_size.y / static_cast<float>(std::max(panel.render_size.y, 1));
            return {
                panel.projection_pos.x + projected->x * scale_x,
                panel.projection_pos.y + projected->y * scale_y,
            };
        };

        const auto isVisible = [&](const CameraPathPanel& panel,
                                   const glm::vec3& pos) -> bool {
            const auto projected = lfs::rendering::projectWorldPoint(
                panel.viewport->getRotationMatrix(),
                panel.viewport->getTranslation(),
                panel.render_size,
                pos,
                settings.focal_length_mm,
                settings.orthographic,
                settings.ortho_scale);
            if (!projected)
                return false;
            const float margin_x =
                (NDC_CULL_MARGIN - 1.0f) * 0.5f * static_cast<float>(panel.render_size.x);
            const float margin_y =
                (NDC_CULL_MARGIN - 1.0f) * 0.5f * static_cast<float>(panel.render_size.y);
            return projected->x >= -margin_x &&
                   projected->x <= static_cast<float>(panel.render_size.x) + margin_x &&
                   projected->y >= -margin_y &&
                   projected->y <= static_cast<float>(panel.render_size.y) + margin_y;
        };

        const auto toColor = [](const ImVec4& c, const float alpha) -> glm::vec4 {
            return {c.x, c.y, c.z, alpha};
        };

        const auto& t = theme();
        const auto* const wm = viewer_->getWindowManager();
        const glm::ivec2 screen_size = wm ? wm->getWindowSize() : glm::ivec2{};
        const glm::ivec2 framebuffer_size = wm ? wm->getFramebufferSize() : glm::ivec2{};
        const int screen_w = screen_size.x;
        const int screen_h = screen_size.y;
        const int fb_w = framebuffer_size.x;
        const int fb_h = framebuffer_size.y;

        const int path_framerate = std::max(ui_state_.framerate, 1);
        const float base_path_time_step = 1.0f / static_cast<float>(path_framerate);
        const float path_duration = std::max(timeline.endTime() - timeline.startTime(), 0.0f);
        const size_t target_render_samples = std::clamp<size_t>(
            static_cast<size_t>(
                std::ceil(std::max(viewport.size.x, 1.0f) * PATH_SAMPLES_PER_VIEWPORT_PIXEL)),
            MIN_PATH_RENDER_SAMPLES, MAX_PATH_RENDER_SAMPLES);
        const float capped_path_time_step =
            (path_duration > 0.0f && target_render_samples > 1)
                ? path_duration / static_cast<float>(target_render_samples - 1)
                : base_path_time_step;
        const float path_time_step = std::max(base_path_time_step, capped_path_time_step);
        const auto path_points = timeline.generatePathAtTimeStep(path_time_step);

        const auto& input = viewer_->getWindowManager()->frameInput();
        const float mouse_x = input.mouse_x;
        const float mouse_y = input.mouse_y;
        const CameraPathPanel* mouse_panel = nullptr;
        for (const auto& panel : panels) {
            if (panel.contains(mouse_x, mouse_y)) {
                mouse_panel = &panel;
                break;
            }
        }

        std::optional<size_t> hovered_keyframe;
        float closest_dist = HIT_RADIUS;

        const glm::vec4 frustum_color = toColor(t.palette.primary, 0.7f);
        const glm::vec4 hovered_frustum_color = toColor(lighten(t.palette.primary, 0.15f), 0.85f);
        const glm::vec4 selected_frustum_color = toColor(lighten(t.palette.primary, 0.3f), 0.9f);

        if (mouse_panel) {
            for (size_t i = 0; i < timeline.keyframes().size(); ++i) {
                const auto& kf = timeline.keyframes()[i];
                if (kf.is_loop_point)
                    continue;
                if (!isVisible(*mouse_panel, kf.position))
                    continue;

                const glm::vec2 s_apex = projectToScreen(*mouse_panel, kf.position);
                const float dx = mouse_x - s_apex.x;
                const float dy = mouse_y - s_apex.y;
                const float dist = std::sqrt(dx * dx + dy * dy);
                if (dist < closest_dist) {
                    closest_dist = dist;
                    hovered_keyframe = i;
                }
            }
        }

        const auto drawOverlay = [&](const CameraPathPanel& panel) {
            if (path_points.size() >= 2) {
                const glm::vec4 path_color = toColor(t.palette.primary, 0.8f);
                const glm::vec4 sample_color = toColor(t.palette.primary, 0.45f);
                for (size_t i = 0; i + 1 < path_points.size(); ++i) {
                    if (!isVisible(panel, path_points[i]) && !isVisible(panel, path_points[i + 1]))
                        continue;
                    line_renderer_.addLine(projectToScreen(panel, path_points[i]),
                                           projectToScreen(panel, path_points[i + 1]),
                                           path_color, PATH_THICKNESS);
                }

                const size_t marker_stride =
                    std::max<size_t>(path_points.size() / MAX_PATH_SAMPLE_MARKERS, 1);
                for (size_t i = 0; i < path_points.size(); i += marker_stride) {
                    if (!isVisible(panel, path_points[i]))
                        continue;
                    line_renderer_.addCircleFilled(projectToScreen(panel, path_points[i]),
                                                   PATH_SAMPLE_RADIUS,
                                                   sample_color, 10);
                }
            }

            for (size_t i = 0; i < timeline.keyframes().size(); ++i) {
                const auto& kf = timeline.keyframes()[i];
                if (kf.is_loop_point)
                    continue;
                if (!isVisible(panel, kf.position))
                    continue;

                const glm::vec2 s_apex = projectToScreen(panel, kf.position);
                const bool selected = controller_.selectedKeyframe() == i;
                const bool hovered = mouse_panel == &panel && hovered_keyframe == i;
                glm::vec4 color = frustum_color;
                if (selected)
                    color = selected_frustum_color;
                else if (hovered)
                    color = hovered_frustum_color;
                const float thickness = selected ? FRUSTUM_THICKNESS * 1.5f : FRUSTUM_THICKNESS;

                const float half_vfov = rendering::focalLengthToVFovRad(kf.focal_length_mm) * 0.5f;
                const float half_h = std::tan(half_vfov) * FRUSTUM_DEPTH;
                const float half_w = half_h * SENSOR_ASPECT;

                const glm::mat3 rot_mat = glm::mat3_cast(kf.rotation);
                const glm::vec3 forward = rendering::cameraForward(rot_mat);
                const glm::vec3 up = rendering::cameraUp(rot_mat);
                const glm::vec3 right = rendering::cameraRight(rot_mat);

                const glm::vec3 apex = kf.position;
                const glm::vec3 base_center = apex + forward * FRUSTUM_DEPTH;
                const glm::vec3 tl = base_center + up * half_h - right * half_w;
                const glm::vec3 tr = base_center + up * half_h + right * half_w;
                const glm::vec3 bl = base_center - up * half_h - right * half_w;
                const glm::vec3 br = base_center - up * half_h + right * half_w;

                const glm::vec2 s_tl = projectToScreen(panel, tl);
                const glm::vec2 s_tr = projectToScreen(panel, tr);
                const glm::vec2 s_bl = projectToScreen(panel, bl);
                const glm::vec2 s_br = projectToScreen(panel, br);

                line_renderer_.addLine(s_apex, s_tl, color, thickness);
                line_renderer_.addLine(s_apex, s_tr, color, thickness);
                line_renderer_.addLine(s_apex, s_bl, color, thickness);
                line_renderer_.addLine(s_apex, s_br, color, thickness);

                line_renderer_.addLine(s_tl, s_tr, color, thickness);
                line_renderer_.addLine(s_tr, s_br, color, thickness);
                line_renderer_.addLine(s_br, s_bl, color, thickness);
                line_renderer_.addLine(s_bl, s_tl, color, thickness);

                const glm::vec3 up_tip = base_center + up * half_h * 1.3f;
                const glm::vec2 s_up = projectToScreen(panel, up_tip);
                line_renderer_.addTriangleFilled(s_up, s_tl, s_tr, color);
            }

            if (!controller_.isStopped()) {
                const auto state = controller_.currentCameraState();
                if (isVisible(panel, state.position)) {
                    const glm::vec4 playhead_color = toColor(t.palette.error, 1.0f);
                    constexpr float PLAYHEAD_FRUSTUM_DEPTH = 0.20f;

                    const float ph_half_vfov = rendering::focalLengthToVFovRad(state.focal_length_mm) * 0.5f;
                    const float ph_half_h = std::tan(ph_half_vfov) * PLAYHEAD_FRUSTUM_DEPTH;
                    const float ph_half_w = ph_half_h * SENSOR_ASPECT;

                    const glm::mat3 rot_mat = glm::mat3_cast(state.rotation);
                    const glm::vec3 forward = rendering::cameraForward(rot_mat);
                    const glm::vec3 up = rendering::cameraUp(rot_mat);
                    const glm::vec3 right = rendering::cameraRight(rot_mat);

                    const glm::vec3 apex = state.position;
                    const glm::vec3 base_center = apex + forward * PLAYHEAD_FRUSTUM_DEPTH;
                    const glm::vec3 tl = base_center + up * ph_half_h - right * ph_half_w;
                    const glm::vec3 tr = base_center + up * ph_half_h + right * ph_half_w;
                    const glm::vec3 bl = base_center - up * ph_half_h - right * ph_half_w;
                    const glm::vec3 br = base_center - up * ph_half_h + right * ph_half_w;

                    const glm::vec2 s_apex = projectToScreen(panel, apex);
                    const glm::vec2 s_tl = projectToScreen(panel, tl);
                    const glm::vec2 s_tr = projectToScreen(panel, tr);
                    const glm::vec2 s_bl = projectToScreen(panel, bl);
                    const glm::vec2 s_br = projectToScreen(panel, br);

                    line_renderer_.addLine(s_apex, s_tl, playhead_color, FRUSTUM_THICKNESS);
                    line_renderer_.addLine(s_apex, s_tr, playhead_color, FRUSTUM_THICKNESS);
                    line_renderer_.addLine(s_apex, s_bl, playhead_color, FRUSTUM_THICKNESS);
                    line_renderer_.addLine(s_apex, s_br, playhead_color, FRUSTUM_THICKNESS);

                    line_renderer_.addLine(s_tl, s_tr, playhead_color, FRUSTUM_THICKNESS);
                    line_renderer_.addLine(s_tr, s_br, playhead_color, FRUSTUM_THICKNESS);
                    line_renderer_.addLine(s_br, s_bl, playhead_color, FRUSTUM_THICKNESS);
                    line_renderer_.addLine(s_bl, s_tl, playhead_color, FRUSTUM_THICKNESS);

                    const glm::vec3 up_tip = base_center + up * ph_half_h * 1.3f;
                    const glm::vec2 s_up = projectToScreen(panel, up_tip);
                    line_renderer_.addTriangleFilled(s_up, s_tl, s_tr, playhead_color);
                }
            }
        };

        for (const auto& panel : panels) {
            line_renderer_.begin(screen_w, screen_h, fb_w, fb_h, panel.clip_rect);
            drawOverlay(panel);
            line_renderer_.end();
        }

        const bool overlay_blocks_mouse =
            overlay_->wantsInput() || overlay_->isMouseOverEditOverlay(mouse_x, mouse_y);
        const bool mouse_blocked_by_ui =
            overlay_blocks_mouse ||
            guiFocusState().want_capture_mouse;

        if (mouse_panel && !mouse_blocked_by_ui && hovered_keyframe.has_value()) {
            const auto* const hovered = timeline.getKeyframe(*hovered_keyframe);
            if (hovered && !hovered->is_loop_point) {
                if (input.mouse_clicked[0] &&
                    !isRotationGizmoHovered() &&
                    !isTranslationGizmoHovered()) {
                    beginViewportKeyframeEdit(*hovered_keyframe);
                    guiFocusState().want_capture_mouse = true;
                }
                if (input.mouse_clicked[1]) {
                    overlay_->showContextMenu(mouse_x, mouse_y, hovered_keyframe,
                                              hovered->time, viewport_edit_mode_);
                    guiFocusState().want_capture_mouse = true;
                }
            }
        }
    }

    void SequencerUIManager::renderKeyframeGizmo(const UIContext& ctx, const ViewportLayout& viewport) {
        if (viewport_edit_mode_ == SequencerViewportEditMode::None)
            return;

        const auto selected = controller_.selectedKeyframe();
        const auto selected_id = controller_.selectedKeyframeId();
        if (!selected.has_value() || !selected_id.has_value()) {
            viewport_edit_mode_ = SequencerViewportEditMode::None;
            keyframe_gizmo_active_ = false;
            return;
        }

        const auto& timeline = controller_.timeline();
        const auto* const kf = timeline.getKeyframe(*selected);
        if (!kf || kf->is_loop_point) {
            viewport_edit_mode_ = SequencerViewportEditMode::None;
            keyframe_gizmo_active_ = false;
            return;
        }

        auto* const rendering_manager = viewer_ ? viewer_->getRenderingManager() : nullptr;
        if (!rendering_manager)
            return;

        auto& primary_viewport = ctx.viewer ? ctx.viewer->getViewport() : viewer_->getViewport();
        const auto& settings = rendering_manager->getSettings();

        const auto& input = viewer_->getWindowManager()->frameInput();
        std::optional<glm::vec2> screen_point;
        if (input.mouse_x >= viewport.pos.x &&
            input.mouse_x <= viewport.pos.x + viewport.size.x &&
            input.mouse_y >= viewport.pos.y &&
            input.mouse_y <= viewport.pos.y + viewport.size.y) {
            screen_point = glm::vec2(input.mouse_x, input.mouse_y);
        }

        const Viewport* gizmo_viewport = &primary_viewport;
        glm::vec2 rect_pos = viewport.pos;
        glm::vec2 rect_size = viewport.size;
        glm::ivec2 render_size(static_cast<int>(std::round(viewport.size.x)),
                               static_cast<int>(std::round(viewport.size.y)));

        if (rendering_manager->isIndependentSplitViewActive()) {
            auto panel = rendering_manager->resolveViewerPanel(
                primary_viewport, viewport.pos, viewport.size, screen_point, std::nullopt);
            if (!panel || !panel->valid()) {
                panel = rendering_manager->resolveViewerPanel(
                    primary_viewport,
                    viewport.pos,
                    viewport.size,
                    std::nullopt,
                    rendering_manager->getFocusedSplitPanel());
            }
            if (panel && panel->valid()) {
                gizmo_viewport = panel->viewport;
                rect_pos = {panel->x, panel->y};
                rect_size = {panel->width, panel->height};
                render_size = {panel->render_width, panel->render_height};
            }
        }

        if (!gizmo_viewport || rect_size.x <= 0.0f || rect_size.y <= 0.0f ||
            render_size.x <= 0 || render_size.y <= 0) {
            return;
        }

        const glm::mat4 view = gizmo_viewport->getViewMatrix();
        const glm::mat4 projection = lfs::rendering::createProjectionMatrixFromFocal(
            render_size,
            settings.focal_length_mm,
            settings.orthographic,
            settings.ortho_scale);

        const glm::mat3 rot_mat = glm::mat3_cast(kf->rotation);
        glm::mat4 gizmo_matrix(rot_mat);
        gizmo_matrix[3] = glm::vec4(kf->position, 1.0f);

        const bool rotate_mode = viewport_edit_mode_ == SequencerViewportEditMode::Rotate;

        NativeOverlayDrawList draw_list;
        const glm::vec2 clip_min(rect_pos.x, rect_pos.y);
        const glm::vec2 clip_max(rect_pos.x + rect_size.x, rect_pos.y + rect_size.y);
        draw_list.PushClipRect(clip_min, clip_max, true);
        const auto& frame_input = viewer_->getWindowManager()->frameInput();
        const NativeGizmoInput gizmo_input{
            .mouse_pos = {frame_input.mouse_x, frame_input.mouse_y},
            .mouse_left_down = frame_input.mouse_down[0],
            .mouse_left_clicked = frame_input.mouse_clicked[0],
        };
        const bool snap_modifier = (frame_input.key_mods & SDL_KMOD_CTRL) != 0;

        bool changed = false;
        bool is_using = false;
        glm::mat3 rotation_delta(1.0f);

        if (rotate_mode) {
            RotationGizmoConfig rotation_config;
            rotation_config.id = 4000;
            rotation_config.viewport_pos = rect_pos;
            rotation_config.viewport_size = rect_size;
            rotation_config.view = view;
            rotation_config.projection = projection;
            rotation_config.pivot_world = kf->position;
            rotation_config.orientation_world = rot_mat;
            rotation_config.draw_list = &draw_list;
            rotation_config.input = gizmo_input;
            rotation_config.snap = snap_modifier;
            rotation_config.snap_degrees = 5.0f;

            const auto rotation_result = drawRotationGizmo(rotation_config);
            changed = rotation_result.changed;
            is_using = rotation_result.active;
            rotation_delta = rotation_result.delta_rotation;
            if (rotation_result.hovered || rotation_result.active)
                guiFocusState().want_capture_mouse = true;
        } else {
            TranslationGizmoConfig translation_config;
            translation_config.id = 4000;
            translation_config.viewport_pos = rect_pos;
            translation_config.viewport_size = rect_size;
            translation_config.view = view;
            translation_config.projection = projection;
            translation_config.pivot_world = kf->position;
            translation_config.orientation_world = glm::mat3(1.0f);
            translation_config.draw_list = &draw_list;
            translation_config.input = gizmo_input;
            translation_config.snap = snap_modifier;
            translation_config.snap_units = 0.1f;

            const auto translation_result = drawTranslationGizmo(translation_config);
            changed = translation_result.changed;
            is_using = translation_result.active;
            if (translation_result.active) {
                const glm::vec3 translated_position = kf->position + translation_result.delta_translation;
                gizmo_matrix[3] = glm::vec4(translated_position, 1.0f);
            }
            if (translation_result.hovered || translation_result.active)
                guiFocusState().want_capture_mouse = true;
        }

        if (is_using)
            guiFocusState().want_capture_mouse = true;

        if (is_using && !keyframe_gizmo_active_)
            keyframe_gizmo_active_ = true;

        if (changed) {
            const glm::vec3 new_pos(gizmo_matrix[3]);
            const glm::quat new_rot = rotate_mode
                                          ? glm::normalize(glm::quat_cast(rotation_delta * rot_mat))
                                          : glm::normalize(glm::quat_cast(glm::mat3(gizmo_matrix)));
            if (controller_.updateKeyframeById(
                    *selected_id,
                    new_pos,
                    new_rot,
                    kf->focal_length_mm)) {
                pip_needs_update_ = true;
                rendering_manager->markDirty(DirtyFlag::OVERLAY);
            }
        }

        if (!is_using && keyframe_gizmo_active_) {
            keyframe_gizmo_active_ = false;
            lfs::core::events::state::KeyframeListChanged{
                .count = controller_.timeline().realKeyframeCount()}
                .emit();
        }

        draw_list.PopClipRect();
    }

    void SequencerUIManager::handleOverlayActions() {
        using Action = RmlSequencerOverlay::Action;
        using namespace lfs::core::events;

        while (auto action = overlay_->consumeAction()) {
            switch (action->action) {
            case Action::ADD_KEYFRAME: {
                const auto& cam = viewer_->getViewport().camera;
                auto* const rm = viewer_->getRenderingManager();
                const float focal_mm = rm ? rm->getFocalLengthMm() : lfs::rendering::DEFAULT_FOCAL_LENGTH_MM;
                const float time = std::max(0.0f, action->time);

                lfs::sequencer::Keyframe kf;
                kf.position = cam.t;
                kf.rotation = glm::quat_cast(cam.R);
                kf.focal_length_mm = focal_mm;
                controller_.addKeyframeAtTime(kf, time);
                controller_.seek(time);
                state::KeyframeListChanged{.count = controller_.timeline().realKeyframeCount()}.emit();
                pip_needs_update_ = true;
            } break;
            case Action::UPDATE_KEYFRAME:
                viewport_edit_mode_ = SequencerViewportEditMode::None;
                keyframe_gizmo_active_ = false;
                endViewportKeyframeEdit();
                cmd::SequencerSelectKeyframe{.keyframe_index = action->keyframe_index}.emit();
                cmd::SequencerUpdateKeyframe{}.emit();
                break;
            case Action::GOTO_KEYFRAME:
                viewport_edit_mode_ = SequencerViewportEditMode::None;
                keyframe_gizmo_active_ = false;
                endViewportKeyframeEdit();
                cmd::SequencerGoToKeyframe{.keyframe_index = action->keyframe_index}.emit();
                break;
            case Action::EDIT_FOCAL_LENGTH:
                keyframe_gizmo_active_ = false;
                endViewportKeyframeEdit();
                panel_->openFocalLengthEdit(
                    action->keyframe_index,
                    controller_.timeline().keyframes()[action->keyframe_index].focal_length_mm);
                break;
            case Action::SET_TRANSLATE:
                keyframe_gizmo_active_ = false;
                endViewportKeyframeEdit();
                cmd::SequencerSelectKeyframe{.keyframe_index = action->keyframe_index}.emit();
                viewport_edit_mode_ = (viewport_edit_mode_ == SequencerViewportEditMode::Translate)
                                          ? SequencerViewportEditMode::None
                                          : SequencerViewportEditMode::Translate;
                if (auto* const rm = viewer_->getRenderingManager())
                    rm->markDirty(DirtyFlag::OVERLAY);
                break;
            case Action::SET_ROTATE:
                keyframe_gizmo_active_ = false;
                endViewportKeyframeEdit();
                cmd::SequencerSelectKeyframe{.keyframe_index = action->keyframe_index}.emit();
                viewport_edit_mode_ = (viewport_edit_mode_ == SequencerViewportEditMode::Rotate)
                                          ? SequencerViewportEditMode::None
                                          : SequencerViewportEditMode::Rotate;
                if (auto* const rm = viewer_->getRenderingManager())
                    rm->markDirty(DirtyFlag::OVERLAY);
                break;
            case Action::SET_EASING: {
                const auto easing = static_cast<sequencer::EasingType>(action->easing_value);
                controller_.setKeyframeEasing(action->keyframe_index, easing);
                state::KeyframeListChanged{.count = controller_.timeline().realKeyframeCount()}.emit();
                break;
            }
            case Action::DELETE_KEYFRAME:
                viewport_edit_mode_ = SequencerViewportEditMode::None;
                keyframe_gizmo_active_ = false;
                endViewportKeyframeEdit();
                cmd::SequencerSelectKeyframe{.keyframe_index = action->keyframe_index}.emit();
                controller_.removeSelectedKeyframe();
                state::KeyframeListChanged{.count = controller_.timeline().realKeyframeCount()}.emit();
                break;
            case Action::CLOSE_EDIT_PANEL:
                viewport_edit_mode_ = SequencerViewportEditMode::None;
                keyframe_gizmo_active_ = false;
                endViewportKeyframeEdit();
                break;
            case Action::APPLY_EDIT:
                if (viewport_keyframe_edit_snapshot_.has_value()) {
                    const auto view_state = currentViewportCameraState();
                    if (controller_.updateKeyframeById(
                            viewport_keyframe_edit_snapshot_->id,
                            view_state.position,
                            view_state.rotation,
                            view_state.focal_length_mm)) {
                        state::KeyframeListChanged{.count = controller_.timeline().realKeyframeCount()}.emit();
                        pip_needs_update_ = true;
                    }
                    if (const auto* const keyframe =
                            controller_.timeline().getKeyframeById(
                                viewport_keyframe_edit_snapshot_->id)) {
                        viewport_keyframe_edit_snapshot_ = *keyframe;
                    }
                }
                break;
            case Action::REVERT_EDIT: {
                if (viewport_keyframe_edit_snapshot_.has_value()) {
                    restoreViewportCameraState({.position = viewport_keyframe_edit_snapshot_->position,
                                                .rotation = viewport_keyframe_edit_snapshot_->rotation,
                                                .focal_length_mm = viewport_keyframe_edit_snapshot_->focal_length_mm});
                }
                break;
            }
            }
        }

        if (auto time_result = overlay_->consumeTimeEdit()) {
            if (controller_.setKeyframeTime(time_result->index, time_result->value)) {
                state::KeyframeListChanged{.count = controller_.timeline().realKeyframeCount()}.emit();
                pip_needs_update_ = true;
            }
        }

        if (auto focal_result = overlay_->consumeFocalEdit()) {
            if (controller_.setKeyframeFocalLength(focal_result->index, focal_result->value)) {
                state::KeyframeListChanged{.count = controller_.timeline().realKeyframeCount()}.emit();
                pip_needs_update_ = true;
            }
        }
    }

    void SequencerUIManager::initPipPreview() {
        pip_initialized_ = true;
    }

    void SequencerUIManager::renderKeyframePreview(const UIContext& ctx) {
        if (!ui_state_.show_pip_preview)
            return;

        const bool is_playing = !controller_.isStopped();
        const auto selected = controller_.selectedKeyframe();

        const auto now = std::chrono::steady_clock::now();
        if (is_playing) {
            const float elapsed = std::chrono::duration<float>(now - pip_last_render_time_).count();
            if (elapsed < 1.0f / PREVIEW_TARGET_FPS)
                return;
        }

        auto* const rm = ctx.viewer->getRenderingManager();
        auto* const sm = ctx.viewer->getSceneManager();
        if (!rm || !sm)
            return;

        if (!pip_initialized_)
            initPipPreview();

        glm::mat3 cam_rot;
        glm::vec3 cam_pos;
        float cam_focal_length_mm;
        auto& vp = ctx.viewer->getViewport();

        if (is_playing) {
            const auto state = controller_.currentCameraState();
            cam_rot = glm::mat3_cast(state.rotation);
            cam_pos = state.position;
            cam_focal_length_mm = state.focal_length_mm;
        } else {
            if (selected.has_value()) {
                if (pip_last_keyframe_ == selected && !pip_needs_update_)
                    return;

                const auto& timeline = controller_.timeline();
                if (*selected >= timeline.size())
                    return;

                const auto* const kf = timeline.getKeyframe(*selected);
                if (!kf)
                    return;

                cam_rot = glm::mat3_cast(kf->rotation);
                cam_pos = kf->position;
                cam_focal_length_mm = kf->focal_length_mm;
            } else {
                if (pip_last_keyframe_.has_value()) {
                    pip_needs_update_ = true;
                    pip_last_keyframe_ = std::nullopt;
                }
                if (!pip_needs_update_)
                    return;

                cam_rot = vp.camera.R;
                cam_pos = vp.camera.t;
                cam_focal_length_mm = rm ? rm->getFocalLengthMm()
                                         : lfs::rendering::DEFAULT_FOCAL_LENGTH_MM;
            }
        }

        const auto image = rm->renderPreviewImage(sm, cam_rot, cam_pos, cam_focal_length_mm,
                                                  PREVIEW_WIDTH, PREVIEW_HEIGHT);
        if (image && pip_texture_.upload(*image, PREVIEW_WIDTH, PREVIEW_HEIGHT)) {
            pip_last_render_time_ = now;
            if (!is_playing) {
                pip_last_keyframe_ = selected;
                pip_needs_update_ = false;
            }
        }
    }

    void SequencerUIManager::syncPipPreviewWindow(const ViewportLayout& viewport) {
        if (!overlay_)
            return;

        if (!ui_state_.show_pip_preview) {
            overlay_->hidePreviewWindow();
            return;
        }

        const bool is_playing = !controller_.isStopped();
        const auto selected = controller_.selectedKeyframe();

        if (!pip_initialized_ || !pip_texture_.valid()) {
            overlay_->hidePreviewWindow();
            return;
        }

        if (!is_playing && selected.has_value()) {
            const auto& timeline = controller_.timeline();
            if (*selected >= timeline.size()) {
                overlay_->hidePreviewWindow();
                return;
            }
            const auto* const kf = timeline.getKeyframe(*selected);
            if (!kf || kf->is_loop_point) {
                overlay_->hidePreviewWindow();
                return;
            }
        }

        const float scale = ui_state_.pip_preview_scale;
        constexpr float MARGIN = 16.0f;
        constexpr float TITLE_HEIGHT = 18.0f;
        const float scaled_width = static_cast<float>(PREVIEW_WIDTH) * scale;
        const float scaled_height = static_cast<float>(PREVIEW_HEIGHT) * scale;
        const float total_height = scaled_height + TITLE_HEIGHT + 8.0f;

        const float left = viewport.pos.x + MARGIN;
        const float top = panel_->cachedPanelY() - total_height - MARGIN;

        const float playhead = controller_.playhead();
        const std::string title = (is_playing || !selected.has_value())
                                      ? std::vformat(LOC(lichtfeld::Strings::Sequencer::PLAYBACK_TIME),
                                                     std::make_format_args(playhead))
                                      : [&selected]() {
                                            const size_t kf_num = *selected + 1;
                                            return std::vformat(LOC(lichtfeld::Strings::Sequencer::KEYFRAME_PREVIEW),
                                                                std::make_format_args(kf_num));
                                        }();

        overlay_->showPreviewWindow(left, top, scaled_width, scaled_height,
                                    title, is_playing, pip_texture_.textureId());
    }

    void SequencerUIManager::renderKeyframeEditOverlay(const ViewportLayout& viewport) {
        if (!viewport_keyframe_edit_snapshot_.has_value()) {
            overlay_->hideEditOverlay();
            return;
        }

        const auto& timeline = controller_.timeline();
        const auto selected = controller_.selectedKeyframe();
        if (!selected.has_value() || !controller_.selectedKeyframeId().has_value() ||
            *controller_.selectedKeyframeId() != viewport_keyframe_edit_snapshot_->id) {
            endViewportKeyframeEdit();
            overlay_->hideEditOverlay();
            return;
        }

        const auto keyframe_index = timeline.findKeyframeIndex(viewport_keyframe_edit_snapshot_->id);
        if (!keyframe_index.has_value()) {
            endViewportKeyframeEdit();
            overlay_->hideEditOverlay();
            return;
        }

        const auto* const keyframe = timeline.getKeyframe(*keyframe_index);
        if (!keyframe || keyframe->is_loop_point) {
            endViewportKeyframeEdit();
            overlay_->hideEditOverlay();
            return;
        }

        const auto cam = currentViewportCameraState();
        const float pos_delta = glm::length(cam.position - viewport_keyframe_edit_snapshot_->position);
        const float dot = std::clamp(std::abs(glm::dot(cam.rotation, viewport_keyframe_edit_snapshot_->rotation)), 0.0f, 1.0f);
        const float rot_delta = glm::degrees(2.0f * std::acos(dot));

        overlay_->updateEditOverlay(*keyframe_index, pos_delta, rot_delta,
                                    viewport.pos.x + viewport.size.x, viewport.pos.y);
    }

} // namespace lfs::vis::gui
