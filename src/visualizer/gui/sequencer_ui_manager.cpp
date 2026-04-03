/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// glad must be included before OpenGL headers
// clang-format off
#include <glad/glad.h>
// clang-format on

#include "gui/sequencer_ui_manager.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "gui/gui_focus_state.hpp"
#include "gui/panel_input_utils.hpp"
#include "gui/rml_sequencer_overlay.hpp"
#include "gui/string_keys.hpp"
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
#include <glm/gtc/type_ptr.hpp>
#include <string_view>
#include <vector>
#include <imgui.h>
#include <ImGuizmo.h>

namespace lfs::vis::gui {

    namespace {
        constexpr size_t MIN_PATH_RENDER_SAMPLES = 128;
        constexpr size_t MAX_PATH_RENDER_SAMPLES = 4096;
        constexpr float PATH_SAMPLES_PER_VIEWPORT_PIXEL = 2.0f;

        [[nodiscard]] std::string formatTimelineTime(const float seconds) {
            const int mins = static_cast<int>(seconds) / 60;
            const float secs = seconds - static_cast<float>(mins * 60);
            return std::format("{}:{:05.2f}", mins, secs);
        }

        void drawGuideLine(ImDrawList* const dl, const float x, const float top, const float bottom,
                           const ImU32 color, const float thickness) {
            dl->AddLine({std::round(x), top}, {std::round(x), bottom}, color, thickness);
        }

    } // namespace

    SequencerUIManager::SequencerUIManager(VisualizerImpl* viewer, panels::SequencerUIState& ui_state,
                                           gui::RmlUIManager* rml_manager)
        : viewer_(viewer),
          ui_state_(ui_state),
          panel_(std::make_unique<RmlSequencerPanel>(controller_, ui_state_, rml_manager)),
          overlay_(std::make_unique<RmlSequencerOverlay>(controller_, rml_manager)),
          scene_sync_(std::make_unique<KeyframeSceneSync>(controller_, viewer)) {}

    SequencerUIManager::~SequencerUIManager() = default;

    void SequencerUIManager::destroyGLResources() {
        pip_fbo_ = {};
        pip_texture_ = {};
        pip_depth_rbo_ = {};
        pip_initialized_ = false;
        line_renderer_.destroyGLResources();
        film_strip_.destroyGLResources();
        if (panel_)
            panel_->destroyGLResources();
        if (overlay_)
            overlay_->destroyGLResources();
    }

    void SequencerUIManager::setSequencerEnabled(const bool enabled) {
        if (enabled)
            return;

        if (ui_state_.show_pip_preview)
            ui_state_.show_pip_preview = false;

        pip_last_keyframe_ = std::nullopt;
        pip_needs_update_ = true;
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
        keyframe_gizmo_op_ = ImGuizmo::OPERATION(0);
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
                keyframe_gizmo_op_ = ImGuizmo::OPERATION(0);
                endViewportKeyframeEdit();
            }
        });

        scene_sync_->setupEvents();
    }

    void SequencerUIManager::render(const UIContext& ctx, const ViewportLayout& viewport) {
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
        renderSequencerPanel(ctx, viewport);
        {
            const float dp = panel_->cachedDpRatio();
            const float px = panel_->cachedPanelX();
            const float pw = panel_->cachedPanelWidth();
            tl_geo_ = {
                px + panel_config::INNER_PADDING_H * dp,
                pw - panel_config::INNER_PADDING_H * 2.0f * dp,
                px, pw, panel_->cachedPanelY(), dp};
        }
        timeline_tooltip_active_ = false;
        timeline_tooltip_text_.clear();
        renderFilmStrip(ctx);
        drawEasingCurves();
        drawTimelineGuides();
        drawTimelineTooltip();
        drawPipPreviewWindow(viewport);

        overlay_->render(sdl_buf.window_w, sdl_buf.window_h);
    }

    void SequencerUIManager::compositeOverlays(const int screen_w, const int screen_h) const {
        if (!overlay_)
            return;
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

    void SequencerUIManager::renderSequencerPanel(const UIContext& /*ctx*/, const ViewportLayout& viewport) {
        const auto& io = ImGui::GetIO();
        controller_.update(io.DeltaTime);

        const bool is_playing = controller_.isPlaying() && controller_.timeline().realKeyframeCount() > 0;

        if (auto* const rm = viewer_->getRenderingManager()) {
            rm->setOverlayAnimationActive(is_playing);
            if (is_playing && ui_state_.follow_playback) {
                rm->markDirty(DirtyFlag::CAMERA);
                const auto state = controller_.currentCameraState();
                auto& vp = viewer_->getViewport();
                vp.camera.R = glm::mat3_cast(state.rotation);
                vp.camera.t = state.position;
                rm->setFocalLength(state.focal_length_mm);
            }
        }

        panel_->setFilmStripAttached(ui_state_.show_film_strip);

        lfs::vis::PanelInputState input =
            buildSequencerPanelInputFromSDL(viewer_->getWindowManager()->frameInput());
        if (const ImGuiViewport* const main_viewport = ImGui::GetMainViewport()) {
            input.screen_x = main_viewport->Pos.x;
            input.screen_y = main_viewport->Pos.y;
        }
        input.time = static_cast<float>(ImGui::GetTime());
        input.delta_time = io.DeltaTime;
        input.want_capture_mouse = guiFocusState().want_capture_mouse;

        const float strip_offset = ui_state_.show_film_strip ? FilmStripRenderer::STRIP_HEIGHT : 0.0f;
        panel_->render(viewport.pos.x, viewport.size.x,
                       viewport.pos.y + viewport.size.y - strip_offset, input);

        if (panel_->isHovered())
            guiFocusState().want_capture_mouse = true;
        if (panel_->wantsKeyboard())
            guiFocusState().want_capture_keyboard = true;

        const auto timeline_menu = panel_->consumeContextMenu();
        if (timeline_menu.open) {
            overlay_->showContextMenu(input.mouse_x, input.mouse_y,
                                      timeline_menu.keyframe, timeline_menu.time, keyframe_gizmo_op_);
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
                active_transport_menu_ = TransportMenuType::SNAP;
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
                active_transport_menu_ = TransportMenuType::PREVIEW;
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
                active_transport_menu_ = TransportMenuType::FORMAT;
                break;
            }
            case Target::CLEAR: {
                items.push_back({LOC("context_menu.clear_confirm"), "", false, true});
                items.push_back({LOC("context_menu.confirm"), "clear_confirm"});
                items.push_back({LOC("context_menu.cancel"), "clear_cancel"});
                active_transport_menu_ = TransportMenuType::CLEAR_CONFIRM;
                break;
            }
            default:
                break;
            }

            if (!items.empty())
                cm.request(std::move(items), ctx_req.screen_x, ctx_req.screen_y);
        }

        if (active_transport_menu_ != TransportMenuType::NONE) {
            auto& cm = viewer_->getGuiManager()->globalContextMenu();
            auto action = cm.pollResult();
            if (!action.empty()) {
                switch (active_transport_menu_) {
                case TransportMenuType::SNAP: {
                    if (action.starts_with("snap_")) {
                        float val = std::stof(action.substr(5));
                        ui_state_.snap_interval = val;
                    }
                    break;
                }
                case TransportMenuType::PREVIEW: {
                    if (action.starts_with("scale_")) {
                        float val = std::stof(action.substr(6));
                        ui_state_.pip_preview_scale = val;
                    }
                    break;
                }
                case TransportMenuType::FORMAT: {
                    if (action.starts_with("preset_")) {
                        using lfs::io::video::VideoPreset;
                        int idx = std::stoi(action.substr(7));
                        ui_state_.preset = static_cast<VideoPreset>(idx);
                        const auto info = lfs::io::video::getPresetInfo(ui_state_.preset);
                        ui_state_.custom_width = info.width;
                        ui_state_.custom_height = info.height;
                        ui_state_.framerate = info.framerate;
                    }
                    break;
                }
                case TransportMenuType::CLEAR_CONFIRM: {
                    if (action == "clear_confirm" &&
                        (controller_.timeline().realKeyframeCount() > 0 || controller_.timeline().hasAnimationClip())) {
                        controller_.clear();
                        lfs::core::events::state::KeyframeListChanged{.count = 0}.emit();
                        LOG_INFO("All keyframes cleared");
                    }
                    break;
                }
                default:
                    break;
                }
                active_transport_menu_ = TransportMenuType::NONE;
            } else if (!cm.isOpen()) {
                active_transport_menu_ = TransportMenuType::NONE;
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
        const auto& vp = viewer_->getViewport();
        auto* const rm = viewer_->getRenderingManager();
        if (!rm)
            return;
        const auto& settings = rm->getSettings();
        const glm::ivec2 vp_size(static_cast<int>(viewport.size.x), static_cast<int>(viewport.size.y));

        const auto projectToScreen = [&](const glm::vec3& pos) -> glm::vec2 {
            const auto projected = lfs::rendering::projectWorldPoint(
                vp.getRotationMatrix(),
                vp.getTranslation(),
                vp_size,
                pos,
                settings.focal_length_mm);
            if (!projected)
                return {-10000.0f, -10000.0f};
            return {viewport.pos.x + projected->x,
                    viewport.pos.y + projected->y};
        };

        const auto isVisible = [&](const glm::vec3& pos) -> bool {
            const auto projected = lfs::rendering::projectWorldPoint(
                vp.getRotationMatrix(),
                vp.getTranslation(),
                vp_size,
                pos,
                settings.focal_length_mm);
            if (!projected)
                return false;
            const float margin_x = (NDC_CULL_MARGIN - 1.0f) * 0.5f * viewport.size.x;
            const float margin_y = (NDC_CULL_MARGIN - 1.0f) * 0.5f * viewport.size.y;
            return projected->x >= -margin_x &&
                   projected->x <= viewport.size.x + margin_x &&
                   projected->y >= -margin_y &&
                   projected->y <= viewport.size.y + margin_y;
        };

        const auto toColor = [](const ImVec4& c, float alpha) -> glm::vec4 {
            return {c.x, c.y, c.z, alpha};
        };

        const auto& t = theme();

        if (timeline.empty())
            return;

        const auto* const wm = viewer_->getWindowManager();
        const glm::ivec2 screen_size = wm ? wm->getWindowSize() : glm::ivec2{};
        const glm::ivec2 framebuffer_size = wm ? wm->getFramebufferSize() : glm::ivec2{};
        const int screen_w = screen_size.x;
        const int screen_h = screen_size.y;
        const int fb_w = framebuffer_size.x;
        const int fb_h = framebuffer_size.y;
        line_renderer_.begin(
            screen_w, screen_h, fb_w, fb_h,
            gui::ClipRect{
                static_cast<int>(std::round(viewport.pos.x)),
                static_cast<int>(std::round(viewport.pos.y)),
                static_cast<int>(std::round(viewport.size.x)),
                static_cast<int>(std::round(viewport.size.y))});

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
        if (path_points.size() >= 2) {
            const glm::vec4 path_color = toColor(t.palette.primary, 0.8f);
            const glm::vec4 sample_color = toColor(t.palette.primary, 0.45f);
            for (size_t i = 0; i + 1 < path_points.size(); ++i) {
                if (!isVisible(path_points[i]) && !isVisible(path_points[i + 1]))
                    continue;
                line_renderer_.addLine(projectToScreen(path_points[i]), projectToScreen(path_points[i + 1]),
                                       path_color, PATH_THICKNESS);
            }

            const size_t marker_stride =
                std::max<size_t>(path_points.size() / MAX_PATH_SAMPLE_MARKERS, 1);
            for (size_t i = 0; i < path_points.size(); i += marker_stride) {
                if (!isVisible(path_points[i]))
                    continue;
                line_renderer_.addCircleFilled(projectToScreen(path_points[i]),
                                               PATH_SAMPLE_RADIUS,
                                               sample_color, 10);
            }
        }

        const auto& input = viewer_->getWindowManager()->frameInput();
        const float mouse_x = input.mouse_x;
        const float mouse_y = input.mouse_y;
        const bool mouse_in_viewport = mouse_x >= viewport.pos.x &&
                                       mouse_x <= viewport.pos.x + viewport.size.x &&
                                       mouse_y >= viewport.pos.y &&
                                       mouse_y <= viewport.pos.y + viewport.size.y;

        std::optional<size_t> hovered_keyframe;
        float closest_dist = HIT_RADIUS;

        const glm::vec4 frustum_color = toColor(t.palette.primary, 0.7f);
        const glm::vec4 hovered_frustum_color = toColor(lighten(t.palette.primary, 0.15f), 0.85f);
        const glm::vec4 selected_frustum_color = toColor(lighten(t.palette.primary, 0.3f), 0.9f);

        for (size_t i = 0; i < timeline.keyframes().size(); ++i) {
            const auto& kf = timeline.keyframes()[i];
            if (kf.is_loop_point)
                continue;
            if (!isVisible(kf.position))
                continue;

            const glm::vec2 s_apex = projectToScreen(kf.position);

            if (mouse_in_viewport) {
                const float dx = mouse_x - s_apex.x;
                const float dy = mouse_y - s_apex.y;
                const float dist = std::sqrt(dx * dx + dy * dy);
                if (dist < closest_dist) {
                    closest_dist = dist;
                    hovered_keyframe = i;
                }
            }

            const bool selected = controller_.selectedKeyframe() == i;
            const bool hovered = hovered_keyframe == i;
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

            const glm::vec2 s_tl = projectToScreen(tl);
            const glm::vec2 s_tr = projectToScreen(tr);
            const glm::vec2 s_bl = projectToScreen(bl);
            const glm::vec2 s_br = projectToScreen(br);

            line_renderer_.addLine(s_apex, s_tl, color, thickness);
            line_renderer_.addLine(s_apex, s_tr, color, thickness);
            line_renderer_.addLine(s_apex, s_bl, color, thickness);
            line_renderer_.addLine(s_apex, s_br, color, thickness);

            line_renderer_.addLine(s_tl, s_tr, color, thickness);
            line_renderer_.addLine(s_tr, s_br, color, thickness);
            line_renderer_.addLine(s_br, s_bl, color, thickness);
            line_renderer_.addLine(s_bl, s_tl, color, thickness);

            const glm::vec3 up_tip = base_center + up * half_h * 1.3f;
            const glm::vec2 s_up = projectToScreen(up_tip);
            line_renderer_.addTriangleFilled(s_up, s_tl, s_tr, color);
        }

        if (!controller_.isStopped()) {
            const auto state = controller_.currentCameraState();
            if (isVisible(state.position)) {
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

                const glm::vec2 s_apex = projectToScreen(apex);
                const glm::vec2 s_tl = projectToScreen(tl);
                const glm::vec2 s_tr = projectToScreen(tr);
                const glm::vec2 s_bl = projectToScreen(bl);
                const glm::vec2 s_br = projectToScreen(br);

                line_renderer_.addLine(s_apex, s_tl, playhead_color, FRUSTUM_THICKNESS);
                line_renderer_.addLine(s_apex, s_tr, playhead_color, FRUSTUM_THICKNESS);
                line_renderer_.addLine(s_apex, s_bl, playhead_color, FRUSTUM_THICKNESS);
                line_renderer_.addLine(s_apex, s_br, playhead_color, FRUSTUM_THICKNESS);

                line_renderer_.addLine(s_tl, s_tr, playhead_color, FRUSTUM_THICKNESS);
                line_renderer_.addLine(s_tr, s_br, playhead_color, FRUSTUM_THICKNESS);
                line_renderer_.addLine(s_br, s_bl, playhead_color, FRUSTUM_THICKNESS);
                line_renderer_.addLine(s_bl, s_tl, playhead_color, FRUSTUM_THICKNESS);

                const glm::vec3 up_tip = base_center + up * ph_half_h * 1.3f;
                const glm::vec2 s_up = projectToScreen(up_tip);
                line_renderer_.addTriangleFilled(s_up, s_tl, s_tr, playhead_color);
            }
        }

        line_renderer_.end();

        if (mouse_in_viewport && !ImGui::IsAnyItemHovered() &&
            !overlay_->wantsInput() && hovered_keyframe.has_value() && !ImGuizmo::IsOver()) {
            const auto* const hovered = timeline.getKeyframe(*hovered_keyframe);
            if (hovered && !hovered->is_loop_point) {
                if (input.mouse_clicked[0]) {
                    beginViewportKeyframeEdit(*hovered_keyframe);
                    guiFocusState().want_capture_mouse = true;
                }
                if (input.mouse_clicked[1]) {
                    overlay_->showContextMenu(mouse_x, mouse_y, hovered_keyframe,
                                              hovered->time, keyframe_gizmo_op_);
                    guiFocusState().want_capture_mouse = true;
                }
            }
        }
    }

    void SequencerUIManager::renderKeyframeGizmo(const UIContext& ctx, const ViewportLayout& viewport) {
        if (keyframe_gizmo_op_ == ImGuizmo::OPERATION(0))
            return;

        const auto selected = controller_.selectedKeyframe();
        if (!selected.has_value()) {
            keyframe_gizmo_op_ = ImGuizmo::OPERATION(0);
            return;
        }

        const auto& timeline = controller_.timeline();
        if (*selected >= timeline.size())
            return;

        const auto* kf = timeline.getKeyframe(*selected);
        if (!kf || kf->is_loop_point) {
            keyframe_gizmo_op_ = ImGuizmo::OPERATION(0);
            return;
        }

        auto* const rendering_manager = ctx.viewer->getRenderingManager();
        if (!rendering_manager)
            return;

        const auto& settings = rendering_manager->getSettings();
        auto& vp = ctx.viewer->getViewport();
        const glm::mat4 view = vp.getViewMatrix();
        const glm::ivec2 vp_size(static_cast<int>(viewport.size.x), static_cast<int>(viewport.size.y));
        const glm::mat4 projection = lfs::rendering::createProjectionMatrix(
            vp_size, lfs::rendering::focalLengthToVFov(settings.focal_length_mm), settings.orthographic, settings.ortho_scale);

        const glm::mat3 rot_mat = glm::mat3_cast(kf->rotation);
        glm::mat4 gizmo_matrix(rot_mat);
        gizmo_matrix[3] = glm::vec4(kf->position, 1.0f);

        ImGuizmo::SetOrthographic(settings.orthographic);
        ImGuizmo::SetRect(viewport.pos.x, viewport.pos.y, viewport.size.x, viewport.size.y);

        ImDrawList* const dl = ImGui::GetForegroundDrawList();
        const ImVec2 clip_min(viewport.pos.x, viewport.pos.y);
        const ImVec2 clip_max(clip_min.x + viewport.size.x, clip_min.y + viewport.size.y);
        dl->PushClipRect(clip_min, clip_max, true);
        ImGuizmo::SetDrawlist(dl);

        const ImGuizmo::MODE mode = (keyframe_gizmo_op_ == ImGuizmo::ROTATE) ? ImGuizmo::LOCAL : ImGuizmo::WORLD;
        glm::mat4 delta;
        const bool changed = ImGuizmo::Manipulate(
            glm::value_ptr(view), glm::value_ptr(projection),
            keyframe_gizmo_op_, mode,
            glm::value_ptr(gizmo_matrix), glm::value_ptr(delta), nullptr);

        const bool is_using = ImGuizmo::IsUsing();

        if (is_using && !keyframe_gizmo_active_) {
            keyframe_gizmo_active_ = true;
        }

        if (changed) {
            const glm::vec3 new_pos(gizmo_matrix[3]);
            const glm::quat new_rot = glm::quat_cast(glm::mat3(gizmo_matrix));
            controller_.updateKeyframe(*selected, new_pos, new_rot, kf->focal_length_mm);
            pip_needs_update_ = true;
        }

        if (!is_using && keyframe_gizmo_active_) {
            keyframe_gizmo_active_ = false;
            lfs::core::events::state::KeyframeListChanged{.count = controller_.timeline().realKeyframeCount()}.emit();
        }

        dl->PopClipRect();
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
                endViewportKeyframeEdit();
                cmd::SequencerSelectKeyframe{.keyframe_index = action->keyframe_index}.emit();
                cmd::SequencerUpdateKeyframe{}.emit();
                break;
            case Action::GOTO_KEYFRAME:
                endViewportKeyframeEdit();
                cmd::SequencerGoToKeyframe{.keyframe_index = action->keyframe_index}.emit();
                break;
            case Action::EDIT_FOCAL_LENGTH:
                endViewportKeyframeEdit();
                panel_->openFocalLengthEdit(
                    action->keyframe_index,
                    controller_.timeline().keyframes()[action->keyframe_index].focal_length_mm);
                break;
            case Action::SET_TRANSLATE:
                endViewportKeyframeEdit();
                cmd::SequencerSelectKeyframe{.keyframe_index = action->keyframe_index}.emit();
                keyframe_gizmo_op_ = (keyframe_gizmo_op_ == ImGuizmo::TRANSLATE)
                                         ? ImGuizmo::OPERATION(0)
                                         : ImGuizmo::TRANSLATE;
                break;
            case Action::SET_ROTATE:
                endViewportKeyframeEdit();
                cmd::SequencerSelectKeyframe{.keyframe_index = action->keyframe_index}.emit();
                keyframe_gizmo_op_ = (keyframe_gizmo_op_ == ImGuizmo::ROTATE)
                                         ? ImGuizmo::OPERATION(0)
                                         : ImGuizmo::ROTATE;
                break;
            case Action::SET_EASING: {
                const auto easing = static_cast<sequencer::EasingType>(action->easing_value);
                controller_.setKeyframeEasing(action->keyframe_index, easing);
                state::KeyframeListChanged{.count = controller_.timeline().realKeyframeCount()}.emit();
                break;
            }
            case Action::DELETE_KEYFRAME:
                endViewportKeyframeEdit();
                cmd::SequencerSelectKeyframe{.keyframe_index = action->keyframe_index}.emit();
                controller_.removeSelectedKeyframe();
                state::KeyframeListChanged{.count = controller_.timeline().realKeyframeCount()}.emit();
                break;
            case Action::CLOSE_EDIT_PANEL:
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

    void SequencerUIManager::renderFilmStrip(const UIContext& ctx) {
        if (!ui_state_.show_film_strip) {
            if (film_strip_scrubbing_) {
                film_strip_scrubbing_ = false;
                controller_.endScrub();
            }
            return;
        }

        auto* const rm = ctx.viewer->getRenderingManager();
        auto* const sm = ctx.viewer->getSceneManager();

        const float timeline_x = tl_geo_.timeline_x;
        const float timeline_width = tl_geo_.timeline_width;
        const float px = tl_geo_.panel_x;
        const float pw = tl_geo_.panel_width;
        if (timeline_width <= 0.0f)
            return;

        const float strip_y = tl_geo_.panel_y + (panel_config::HEIGHT + panel_config::EASING_STRIPE_HEIGHT - panel_config::BORDER_OVERLAP) * tl_geo_.dp;

        std::optional<float> selected_keyframe_time;
        if (const auto selected = controller_.selectedKeyframe(); selected.has_value()) {
            if (const auto* const keyframe = controller_.timeline().getKeyframe(*selected))
                selected_keyframe_time = keyframe->time;
        }

        std::optional<float> hovered_keyframe_time;
        if (const auto hovered_id = panel_->hoveredKeyframeId(); hovered_id.has_value()) {
            if (const auto* const keyframe = controller_.timeline().getKeyframeById(*hovered_id))
                hovered_keyframe_time = keyframe->time;
        }

        FilmStripRenderer::RenderOptions options;
        options.panel_x = px;
        options.panel_width = pw;
        options.timeline_x = timeline_x;
        options.timeline_width = timeline_width;
        options.strip_y = strip_y;
        const auto& input = viewer_->getWindowManager()->frameInput();
        options.mouse_x = input.mouse_x;
        options.mouse_y = input.mouse_y;
        options.zoom_level = panel_->zoomLevel();
        options.pan_offset = panel_->panOffset();
        options.display_end_time = panel_->getDisplayEndTime();
        options.selected_keyframe_id = controller_.selectedKeyframeId();
        options.hovered_keyframe_id = panel_->hoveredKeyframeId();
        options.selected_keyframe_time = selected_keyframe_time;
        options.hovered_keyframe_time = hovered_keyframe_time;
        film_strip_.render(controller_, rm, sm, options);

        const bool can_scrub = controller_.timeline().size() >= 2;
        const float scrub_time = can_scrub
                                     ? std::clamp(
                                           sequencer_ui::screenXToTime(input.mouse_x, timeline_x, timeline_width,
                                                                       panel_->getDisplayEndTime(), panel_->panOffset()),
                                           controller_.timeline().startTime(), controller_.timeline().endTime())
                                     : 0.0f;

        if (film_strip_scrubbing_) {
            if (input.mouse_down[0] && can_scrub) {
                controller_.scrub(scrub_time);
            } else {
                film_strip_scrubbing_ = false;
                controller_.endScrub();
            }
        }

        if (const auto& hover = film_strip_.hoverState(); hover.has_value()) {
            guiFocusState().want_capture_mouse = true;

            if (can_scrub && !overlay_->wantsInput() && !film_strip_scrubbing_ && input.mouse_clicked[0]) {
                film_strip_scrubbing_ = true;
                controller_.beginScrub();
                controller_.scrub(scrub_time);
            }

            std::string tooltip = std::format("Time {}", formatTimelineTime(hover->exact_time));
            if (hover->over_thumbnail) {
                tooltip += std::format("\nSample {}", formatTimelineTime(hover->sample_time));
                tooltip += std::format("\nCovers {} - {}",
                                       formatTimelineTime(hover->interval_start_time),
                                       formatTimelineTime(hover->interval_end_time));
            }
            timeline_tooltip_active_ = true;
            timeline_tooltip_pos_ = {input.mouse_x, input.mouse_y};
            timeline_tooltip_text_ = std::move(tooltip);
        }
    }

    void SequencerUIManager::drawEasingCurves() {
        const float dp = tl_geo_.dp;
        const float px = tl_geo_.panel_x;
        const float pw = tl_geo_.panel_width;
        const float panel_y = tl_geo_.panel_y;
        const float timeline_x = tl_geo_.timeline_x;
        const float timeline_width = tl_geo_.timeline_width;
        if (timeline_width <= 0.0f)
            return;

        const float stripe_y = panel_y + panel_config::HEIGHT * dp;
        const float stripe_h = panel_config::EASING_STRIPE_HEIGHT * dp;
        const float y_center = stripe_y + stripe_h * 0.5f;

        auto* dl = ImGui::GetForegroundDrawList();

        const auto& t = theme();
        dl->AddRectFilled({px, stripe_y}, {px + pw, stripe_y + stripe_h},
                          toU32WithAlpha(t.palette.surface, 0.85f),
                          0.0f);
        dl->AddLine({px, stripe_y}, {px + pw, stripe_y},
                    toU32WithAlpha(t.palette.border, 0.3f));

        const auto& timeline = controller_.timeline();
        const auto& keyframes = timeline.keyframes();
        if (keyframes.size() < 2)
            return;

        constexpr int CURVE_SAMPLES = 20;
        constexpr float CURVE_THICKNESS = 1.5f;
        constexpr float DOT_RADIUS = 3.0f;
        constexpr float INDICATOR_SIZE = 4.0f;

        const float pan = panel_->panOffset();
        const float display_end = panel_->getDisplayEndTime();
        const float amplitude = stripe_h * 0.35f;

        const auto localTimeToX = [&](float time) -> float {
            return sequencer_ui::timeToScreenX(time, timeline_x, timeline_width, display_end, pan);
        };

        dl->PushClipRect({timeline_x, stripe_y}, {timeline_x + timeline_width, stripe_y + stripe_h}, true);
        const ImU32 colors[2] = {
            toU32WithAlpha(t.palette.primary, 0.8f),
            toU32WithAlpha(t.palette.secondary, 0.8f),
        };
        const ImU32 segment_fills[2] = {
            toU32WithAlpha(t.palette.primary, 0.25f),
            toU32WithAlpha(t.palette.secondary, 0.25f),
        };
        const ImU32 curve_color = toU32WithAlpha(t.palette.primary, 0.5f);

        for (size_t i = 0; i + 1 < keyframes.size(); ++i) {
            const float x0 = localTimeToX(keyframes[i].time);
            const float x1 = localTimeToX(keyframes[i + 1].time);
            dl->AddRectFilled({x0, stripe_y}, {x1, stripe_y + stripe_h}, segment_fills[i % 2]);
        }

        for (size_t i = 0; i + 1 < keyframes.size(); ++i) {
            const auto& kf_a = keyframes[i];
            const auto& kf_b = keyframes[i + 1];
            const auto easing = kf_a.easing;
            const float x0 = localTimeToX(kf_a.time);
            const float x1 = localTimeToX(kf_b.time);

            if (easing == sequencer::EasingType::LINEAR) {
                dl->AddLine({x0, y_center}, {x1, y_center}, curve_color, CURVE_THICKNESS);
                continue;
            }

            ImVec2 points[CURVE_SAMPLES + 1];
            for (int s = 0; s <= CURVE_SAMPLES; ++s) {
                const float t_norm = static_cast<float>(s) / static_cast<float>(CURVE_SAMPLES);
                const float eased = sequencer::applyEasing(t_norm, easing);
                const float x = x0 + t_norm * (x1 - x0);
                const float y = y_center - (eased - t_norm) * amplitude;
                points[s] = {x, y};
            }
            dl->AddPolyline(points, CURVE_SAMPLES + 1, curve_color, ImDrawFlags_None, CURVE_THICKNESS);
        }

        for (size_t i = 0; i < keyframes.size(); ++i) {
            const float kx = localTimeToX(keyframes[i].time);
            const ImU32 kf_color = colors[i % 2];
            dl->AddCircleFilled({kx, y_center}, DOT_RADIUS, kf_color);

            const auto easing = keyframes[i].easing;
            if (easing == sequencer::EasingType::LINEAR)
                continue;

            const float iy = y_center - stripe_h * 0.3f;
            switch (easing) {
            case sequencer::EasingType::EASE_IN:
                dl->AddTriangleFilled(
                    {kx, iy},
                    {kx + INDICATOR_SIZE, iy - INDICATOR_SIZE},
                    {kx - INDICATOR_SIZE, iy - INDICATOR_SIZE},
                    kf_color);
                break;
            case sequencer::EasingType::EASE_OUT:
                dl->AddTriangleFilled(
                    {kx - INDICATOR_SIZE, iy},
                    {kx + INDICATOR_SIZE, iy},
                    {kx, iy - INDICATOR_SIZE},
                    kf_color);
                break;
            case sequencer::EasingType::EASE_IN_OUT:
                dl->AddQuadFilled(
                    {kx, iy - INDICATOR_SIZE},
                    {kx + INDICATOR_SIZE, iy - INDICATOR_SIZE * 0.5f},
                    {kx, iy},
                    {kx - INDICATOR_SIZE, iy - INDICATOR_SIZE * 0.5f},
                    kf_color);
                break;
            default:
                break;
            }
        }

        dl->PopClipRect();

        const auto& input = viewer_->getWindowManager()->frameInput();
        const float mx = input.mouse_x;
        const float my = input.mouse_y;
        if (mx >= timeline_x && mx <= timeline_x + timeline_width &&
            my >= stripe_y && my <= stripe_y + stripe_h) {
            guiFocusState().want_capture_mouse = true;

            if (input.mouse_clicked[1]) {
                std::optional<size_t> nearest;
                float best_dist = panel_config::KEYFRAME_RADIUS * 3.0f * dp;
                for (size_t i = 0; i < keyframes.size(); ++i) {
                    const float dist = std::abs(mx - localTimeToX(keyframes[i].time));
                    if (dist < best_dist) {
                        best_dist = dist;
                        nearest = i;
                    }
                }
                overlay_->showContextMenu(mx, my, nearest,
                                          nearest.has_value() ? keyframes[*nearest].time : controller_.playhead(),
                                          keyframe_gizmo_op_);
            }

            if (input.mouse_clicked[0]) {
                std::optional<size_t> nearest;
                float best_dist = panel_config::KEYFRAME_RADIUS * 2.0f * dp;
                for (size_t i = 0; i < keyframes.size(); ++i) {
                    const float dist = std::abs(mx - localTimeToX(keyframes[i].time));
                    if (dist < best_dist) {
                        best_dist = dist;
                        nearest = i;
                    }
                }
                if (nearest.has_value())
                    lfs::core::events::cmd::SequencerSelectKeyframe{.keyframe_index = *nearest}.emit();
            }
        }
    }

    void SequencerUIManager::drawTimelineGuides() {
        if (tl_geo_.timeline_width <= 0.0f)
            return;

        const float dp = tl_geo_.dp;
        const float panel_y = tl_geo_.panel_y;
        const float line_top = panel_y + (panel_config::TRANSPORT_ROW_HEIGHT + panel_config::INNER_PADDING) * dp;
        const float strip_offset = ui_state_.show_film_strip ? FilmStripRenderer::STRIP_HEIGHT : 0.0f;
        const float line_bottom = panel_y + (panel_config::HEIGHT + panel_config::EASING_STRIPE_HEIGHT - panel_config::BORDER_OVERLAP) * dp + strip_offset;

        auto* dl = ImGui::GetForegroundDrawList();
        const auto& t = theme();
        const float display_end = panel_->getDisplayEndTime();
        const float pan = panel_->panOffset();

        const auto timeToX = [&](const float time) -> float {
            return sequencer_ui::timeToScreenX(time, tl_geo_.timeline_x, tl_geo_.timeline_width, display_end, pan);
        };
        const auto drawTimedGuide = [&](const float time, const ImU32 color, const float thickness) {
            const float x = timeToX(time);
            if (x < tl_geo_.timeline_x || x > tl_geo_.timeline_x + tl_geo_.timeline_width)
                return;
            drawGuideLine(dl, x, line_top, line_bottom, color, thickness);
        };

        if (ui_state_.show_film_strip) {
            if (const auto& hover = film_strip_.hoverState(); hover.has_value()) {
                drawGuideLine(dl, hover->guide_x, line_top, line_bottom,
                              toU32WithAlpha(t.palette.text_dim, 0.55f), 1.0f);
            }
        }

        if (const auto hovered_id = panel_->hoveredKeyframeId(); hovered_id.has_value()) {
            if (const auto* const keyframe = controller_.timeline().getKeyframeById(*hovered_id)) {
                drawTimedGuide(keyframe->time, toU32WithAlpha(t.palette.secondary, 0.75f), 1.5f);
            }
        }

        if (const auto selected = controller_.selectedKeyframe(); selected.has_value()) {
            if (const auto* const keyframe = controller_.timeline().getKeyframe(*selected)) {
                drawTimedGuide(keyframe->time, toU32WithAlpha(t.palette.primary, 0.85f), 2.0f);
            }
        }

        if (panel_->isPlayheadInRange()) {
            drawGuideLine(dl, panel_->cachedPlayheadScreenX(), line_top, line_bottom,
                          theme().error_u32(), panel_config::PLAYHEAD_WIDTH);
        }
    }

    void SequencerUIManager::drawTimelineTooltip() {
        if (!timeline_tooltip_active_ || timeline_tooltip_text_.empty())
            return;

        auto* const dl = ImGui::GetForegroundDrawList();
        const auto& t = theme();
        const float dp = std::max(tl_geo_.dp, 1.0f);
        const float pad_x = 10.0f * dp;
        const float pad_y = 7.0f * dp;
        const float line_gap = 2.0f * dp;
        const float offset_x = 14.0f * dp;
        const float offset_y = -10.0f * dp;

        std::vector<std::string_view> lines;
        size_t start = 0;
        while (start <= timeline_tooltip_text_.size()) {
            const size_t end = timeline_tooltip_text_.find('\n', start);
            if (end == std::string::npos) {
                lines.emplace_back(timeline_tooltip_text_.data() + start, timeline_tooltip_text_.size() - start);
                break;
            }
            lines.emplace_back(timeline_tooltip_text_.data() + start, end - start);
            start = end + 1;
        }

        float max_width = 0.0f;
        float total_height = pad_y * 2.0f;
        for (size_t i = 0; i < lines.size(); ++i) {
            const ImVec2 size = ImGui::CalcTextSize(lines[i].data(), lines[i].data() + lines[i].size());
            max_width = std::max(max_width, size.x);
            total_height += size.y;
            if (i + 1 < lines.size())
                total_height += line_gap;
        }

        const glm::ivec2 display_size = viewer_->getWindowManager()->getWindowSize();
        const ImVec2 display(static_cast<float>(display_size.x),
                             static_cast<float>(display_size.y));
        ImVec2 box_min(timeline_tooltip_pos_.x + offset_x, timeline_tooltip_pos_.y + offset_y - total_height);
        ImVec2 box_max(box_min.x + max_width + pad_x * 2.0f, box_min.y + total_height);

        if (box_max.x > display.x - 8.0f * dp) {
            const float shift = box_max.x - (display.x - 8.0f * dp);
            box_min.x -= shift;
            box_max.x -= shift;
        }
        if (box_min.x < 8.0f * dp) {
            const float shift = 8.0f * dp - box_min.x;
            box_min.x += shift;
            box_max.x += shift;
        }
        if (box_min.y < 8.0f * dp) {
            const float shift = (timeline_tooltip_pos_.y + 18.0f * dp) - box_min.y;
            box_min.y += shift;
            box_max.y += shift;
        }

        dl->AddRectFilled({box_min.x + 2.0f * dp, box_min.y + 3.0f * dp},
                          {box_max.x + 2.0f * dp, box_max.y + 3.0f * dp},
                          IM_COL32(0, 0, 0, 60), 8.0f * dp);
        dl->AddRectFilled(box_min, box_max,
                          toU32WithAlpha(t.palette.surface, 0.96f), 8.0f * dp);
        dl->AddRect(box_min, box_max,
                    toU32WithAlpha(t.palette.border, 0.75f), 8.0f * dp, 0, 1.0f);

        float text_y = box_min.y + pad_y;
        for (size_t i = 0; i < lines.size(); ++i) {
            const ImVec2 size = ImGui::CalcTextSize(lines[i].data(), lines[i].data() + lines[i].size());
            const ImU32 color = (i == 0)
                                    ? t.text_u32()
                                    : toU32WithAlpha(t.palette.text_dim, 0.95f);
            dl->AddText({box_min.x + pad_x, text_y}, color,
                        lines[i].data(), lines[i].data() + lines[i].size());
            text_y += size.y + line_gap;
        }
    }

    void SequencerUIManager::initPipPreview() {
        if (pip_initialized_ || pip_init_failed_)
            return;

        glGenFramebuffers(1, pip_fbo_.ptr());
        glGenTextures(1, pip_texture_.ptr());
        glGenRenderbuffers(1, pip_depth_rbo_.ptr());

        glBindTexture(GL_TEXTURE_2D, pip_texture_.get());
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, PREVIEW_WIDTH, PREVIEW_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glBindTexture(GL_TEXTURE_2D, 0);

        glBindRenderbuffer(GL_RENDERBUFFER, pip_depth_rbo_.get());
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, PREVIEW_WIDTH, PREVIEW_HEIGHT);
        glBindRenderbuffer(GL_RENDERBUFFER, 0);

        glBindFramebuffer(GL_FRAMEBUFFER, pip_fbo_.get());
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, pip_texture_.get(), 0);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, pip_depth_rbo_.get());

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            LOG_ERROR("PiP preview FBO incomplete");
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            pip_init_failed_ = true;
            return;
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

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

        if (rm->renderPreviewFrame(sm, cam_rot, cam_pos, cam_focal_length_mm,
                                   pip_fbo_, pip_texture_, PREVIEW_WIDTH, PREVIEW_HEIGHT)) {
            pip_last_render_time_ = now;
            if (!is_playing) {
                pip_last_keyframe_ = selected;
                pip_needs_update_ = false;
            }
        }
    }

    void SequencerUIManager::drawPipPreviewWindow(const ViewportLayout& viewport) {
        if (!ui_state_.show_pip_preview)
            return;

        const bool is_playing = !controller_.isStopped();
        const auto selected = controller_.selectedKeyframe();

        if (!pip_initialized_ || pip_texture_ == 0)
            return;

        if (!is_playing && selected.has_value()) {
            const auto& timeline = controller_.timeline();
            if (*selected >= timeline.size())
                return;
            const auto* const kf = timeline.getKeyframe(*selected);
            if (!kf || kf->is_loop_point)
                return;
        }

        const auto& t = theme();
        const float scale = ui_state_.pip_preview_scale;
        constexpr float MARGIN = 16.0f;
        const float dp = panel_->cachedDpRatio();
        const float panel_height = (panel_config::HEIGHT + panel_config::PADDING_BOTTOM +
                                    panel_config::EASING_STRIPE_HEIGHT) *
                                       dp +
                                   (ui_state_.show_film_strip ? FilmStripRenderer::STRIP_HEIGHT : 0.0f);
        constexpr float PADDING = 4.0f;
        constexpr float TITLE_HEIGHT = 18.0f;
        const float scaled_width = static_cast<float>(PREVIEW_WIDTH) * scale;
        const float scaled_height = static_cast<float>(PREVIEW_HEIGHT) * scale;
        const float total_height = scaled_height + TITLE_HEIGHT + PADDING * 2.0f;

        const ImVec2 pos(
            viewport.pos.x + MARGIN,
            viewport.pos.y + viewport.size.y - panel_height - total_height - MARGIN);
        const ImVec2 size(scaled_width + PADDING * 2.0f, total_height);

        const ImU32 bg_color = toU32WithAlpha(t.palette.surface, 0.95f);
        const ImU32 border_color = is_playing
                                       ? t.error_u32()
                                       : toU32WithAlpha(t.palette.primary, 0.6f);
        const ImU32 text_color = toU32WithAlpha(t.palette.text, 0.8f);

        auto* dl = ImGui::GetForegroundDrawList();
        const ImVec2 p1(pos.x + size.x, pos.y + size.y);
        dl->AddRectFilled(pos, p1, bg_color, t.sizes.window_rounding);
        dl->AddRect(pos, p1, border_color, t.sizes.window_rounding, 0, 2.0f);

        const float playhead = controller_.playhead();
        const std::string title = (is_playing || !selected.has_value())
                                      ? std::vformat(LOC(lichtfeld::Strings::Sequencer::PLAYBACK_TIME),
                                                     std::make_format_args(playhead))
                                      : [&selected]() {
                                            const size_t kf_num = *selected + 1;
                                            return std::vformat(LOC(lichtfeld::Strings::Sequencer::KEYFRAME_PREVIEW),
                                                                std::make_format_args(kf_num));
                                        }();
        dl->AddText({pos.x + PADDING, pos.y + PADDING}, text_color, title.c_str());

        const ImVec2 img_pos(pos.x + PADDING, pos.y + PADDING + TITLE_HEIGHT);
        const ImVec2 img_end(img_pos.x + scaled_width, img_pos.y + scaled_height);
        dl->AddImage(static_cast<ImTextureID>(static_cast<uintptr_t>(pip_texture_.get())),
                     img_pos, img_end, {0, 1}, {1, 0});
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
