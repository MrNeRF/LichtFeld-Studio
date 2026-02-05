/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// glad must be included before OpenGL headers
// clang-format off
#include <glad/glad.h>
// clang-format on

#include "gui/sequencer_ui_manager.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "rendering/rendering.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "sequencer/keyframe.hpp"
#include "theme/theme.hpp"
#include "visualizer_impl.hpp"

#include <cmath>
#include <format>
#include <glm/gtc/type_ptr.hpp>
#include <imgui_internal.h>
#include <imgui.h>
#include <ImGuizmo.h>

namespace lfs::vis::gui {

    SequencerUIManager::SequencerUIManager(VisualizerImpl* viewer, panels::SequencerUIState& ui_state)
        : viewer_(viewer),
          ui_state_(ui_state),
          panel_(std::make_unique<SequencerPanel>(controller_)) {}

    SequencerUIManager::~SequencerUIManager() = default;

    void SequencerUIManager::destroyGLResources() {
        pip_fbo_ = {};
        pip_texture_ = {};
        pip_depth_rbo_ = {};
        pip_initialized_ = false;
    }

    void SequencerUIManager::setupEvents() {
        using namespace lfs::core::events;

        cmd::SequencerAddKeyframe::when([this](const auto&) {
            const auto& cam = viewer_->getViewport().camera;
            auto& timeline = controller_.timeline();

            const float interval = ui_state_.snap_to_grid ? ui_state_.snap_interval : 1.0f;
            const float time = timeline.empty() ? 0.0f : timeline.endTime() + interval;

            lfs::sequencer::Keyframe kf;
            kf.time = time;
            kf.position = cam.t;
            kf.rotation = glm::quat_cast(cam.R);
            kf.fov = lfs::sequencer::DEFAULT_FOV;
            timeline.addKeyframe(kf);
            controller_.seek(time);
        });

        cmd::SequencerUpdateKeyframe::when([this](const auto&) {
            if (!controller_.hasSelection())
                return;
            const auto& cam = viewer_->getViewport().camera;
            controller_.updateSelectedKeyframe(
                cam.t,
                glm::quat_cast(cam.R),
                lfs::sequencer::DEFAULT_FOV);
        });

        cmd::SequencerPlayPause::when([this](const auto&) {
            controller_.togglePlayPause();
        });
    }

    void SequencerUIManager::render(const UIContext& ctx, const ViewportLayout& viewport) {
        if (ui_state_.show_camera_path) {
            renderCameraPath(viewport);
            renderKeyframeGizmo(ctx, viewport);
            renderKeyframePreview(ctx);
        }
        renderSequencerPanel(ctx, viewport);
        drawPipPreviewWindow(viewport);
        renderContextMenu();
    }

    void SequencerUIManager::renderSequencerPanel(const UIContext& /*ctx*/, const ViewportLayout& viewport) {
        controller_.update(ImGui::GetIO().DeltaTime);

        const bool is_playing = controller_.isPlaying() && !controller_.timeline().empty();

        if (auto* const rm = viewer_->getRenderingManager()) {
            rm->setOverlayAnimationActive(is_playing);
            if (is_playing && ui_state_.follow_playback) {
                rm->markDirty();
                const auto state = controller_.currentCameraState();
                auto& vp = viewer_->getViewport();
                vp.camera.R = glm::mat3_cast(state.rotation);
                vp.camera.t = state.position;
            }
        }

        panel_->setSnapEnabled(ui_state_.snap_to_grid);
        panel_->setSnapInterval(ui_state_.snap_interval);
        panel_->render(viewport.pos.x, viewport.size.x, viewport.pos.y + viewport.size.y);
    }

    void SequencerUIManager::renderCameraPath(const ViewportLayout& viewport) {
        constexpr float PATH_THICKNESS = 2.0f;
        constexpr float FRUSTUM_THICKNESS = 1.5f;
        constexpr float NDC_CULL_MARGIN = 1.5f;
        constexpr int PATH_SAMPLES = 20;
        constexpr float FRUSTUM_SIZE = 0.15f;
        constexpr float FRUSTUM_DEPTH = 0.25f;
        constexpr float HIT_RADIUS = 15.0f;

        const auto& timeline = controller_.timeline();
        const auto& vp = viewer_->getViewport();
        const glm::mat4 view_proj = vp.getProjectionMatrix() * vp.getViewMatrix();

        const auto projectToScreen = [&](const glm::vec3& pos) -> ImVec2 {
            const glm::vec4 clip = view_proj * glm::vec4(pos, 1.0f);
            if (clip.w <= 0.0f)
                return {-10000.0f, -10000.0f};
            const glm::vec3 ndc = glm::vec3(clip) / clip.w;
            return {viewport.pos.x + (ndc.x * 0.5f + 0.5f) * viewport.size.x,
                    viewport.pos.y + (1.0f - (ndc.y * 0.5f + 0.5f)) * viewport.size.y};
        };

        const auto isVisible = [&](const glm::vec3& pos) -> bool {
            const glm::vec4 clip = view_proj * glm::vec4(pos, 1.0f);
            if (clip.w <= 0.0f)
                return false;
            const glm::vec3 ndc = glm::vec3(clip) / clip.w;
            return std::abs(ndc.x) <= NDC_CULL_MARGIN && std::abs(ndc.y) <= NDC_CULL_MARGIN;
        };

        ImDrawList* const dl = ImGui::GetBackgroundDrawList();
        const auto& t = theme();

        if (timeline.empty())
            return;

        const auto path_points = timeline.generatePath(PATH_SAMPLES);
        if (path_points.size() >= 2) {
            const ImU32 path_color = toU32WithAlpha(t.palette.primary, 0.8f);
            for (size_t i = 0; i + 1 < path_points.size(); ++i) {
                if (!isVisible(path_points[i]) && !isVisible(path_points[i + 1]))
                    continue;
                dl->AddLine(projectToScreen(path_points[i]), projectToScreen(path_points[i + 1]), path_color, PATH_THICKNESS);
            }
        }

        const ImVec2 mouse = ImGui::GetMousePos();
        const bool mouse_in_viewport = mouse.x >= viewport.pos.x && mouse.x <= viewport.pos.x + viewport.size.x &&
                                       mouse.y >= viewport.pos.y && mouse.y <= viewport.pos.y + viewport.size.y;

        std::optional<size_t> hovered_keyframe;
        float closest_dist = HIT_RADIUS;

        const ImU32 frustum_color = toU32WithAlpha(t.palette.primary, 0.7f);
        const ImU32 hovered_frustum_color = toU32WithAlpha(lighten(t.palette.primary, 0.15f), 0.85f);
        const ImU32 selected_frustum_color = toU32WithAlpha(lighten(t.palette.primary, 0.3f), 0.9f);

        for (size_t i = 0; i < timeline.keyframes().size(); ++i) {
            const auto& kf = timeline.keyframes()[i];
            if (!isVisible(kf.position))
                continue;

            const ImVec2 s_apex = projectToScreen(kf.position);

            if (mouse_in_viewport) {
                const float dist = std::sqrt((mouse.x - s_apex.x) * (mouse.x - s_apex.x) +
                                             (mouse.y - s_apex.y) * (mouse.y - s_apex.y));
                if (dist < closest_dist) {
                    closest_dist = dist;
                    hovered_keyframe = i;
                }
            }

            const bool selected = controller_.selectedKeyframe() == i;
            const bool hovered = hovered_keyframe == i;
            ImU32 color = frustum_color;
            if (selected)
                color = selected_frustum_color;
            else if (hovered)
                color = hovered_frustum_color;
            const float thickness = selected ? FRUSTUM_THICKNESS * 1.5f : FRUSTUM_THICKNESS;

            const glm::mat3 rot_mat = glm::mat3_cast(kf.rotation);
            const glm::vec3 forward = rot_mat[2];
            const glm::vec3 up = -rot_mat[1];
            const glm::vec3 right = rot_mat[0];

            const glm::vec3 apex = kf.position;

            const glm::vec3 base_center = apex + forward * FRUSTUM_DEPTH;
            const glm::vec3 tl = base_center + up * FRUSTUM_SIZE - right * FRUSTUM_SIZE;
            const glm::vec3 tr = base_center + up * FRUSTUM_SIZE + right * FRUSTUM_SIZE;
            const glm::vec3 bl = base_center - up * FRUSTUM_SIZE - right * FRUSTUM_SIZE;
            const glm::vec3 br = base_center - up * FRUSTUM_SIZE + right * FRUSTUM_SIZE;

            const ImVec2 s_tl = projectToScreen(tl);
            const ImVec2 s_tr = projectToScreen(tr);
            const ImVec2 s_bl = projectToScreen(bl);
            const ImVec2 s_br = projectToScreen(br);

            dl->AddLine(s_apex, s_tl, color, thickness);
            dl->AddLine(s_apex, s_tr, color, thickness);
            dl->AddLine(s_apex, s_bl, color, thickness);
            dl->AddLine(s_apex, s_br, color, thickness);

            dl->AddLine(s_tl, s_tr, color, thickness);
            dl->AddLine(s_tr, s_br, color, thickness);
            dl->AddLine(s_br, s_bl, color, thickness);
            dl->AddLine(s_bl, s_tl, color, thickness);

            const glm::vec3 up_tip = base_center + up * FRUSTUM_SIZE * 1.3f;
            const ImVec2 s_up = projectToScreen(up_tip);
            dl->AddTriangleFilled(s_up, s_tl, s_tr, color);
        }

        if (mouse_in_viewport && !ImGui::IsAnyItemHovered()) {
            if (hovered_keyframe.has_value() && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !ImGuizmo::IsOver()) {
                controller_.selectKeyframe(*hovered_keyframe);
            }
            if (ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
                context_menu_keyframe_ = hovered_keyframe;
                context_menu_open_ = true;
                ImGui::OpenPopup("KeyframeContextMenu");
            }
        }

        if (!controller_.isStopped()) {
            const auto state = controller_.currentCameraState();
            if (isVisible(state.position)) {
                const ImU32 playhead_color = t.error_u32();
                constexpr float PLAYHEAD_FRUSTUM_SIZE = 0.12f;
                constexpr float PLAYHEAD_FRUSTUM_DEPTH = 0.20f;

                const glm::mat3 rot_mat = glm::mat3_cast(state.rotation);
                const glm::vec3 forward = rot_mat[2];
                const glm::vec3 up = -rot_mat[1];
                const glm::vec3 right = rot_mat[0];

                const glm::vec3 apex = state.position;
                const glm::vec3 base_center = apex + forward * PLAYHEAD_FRUSTUM_DEPTH;
                const glm::vec3 tl = base_center + up * PLAYHEAD_FRUSTUM_SIZE - right * PLAYHEAD_FRUSTUM_SIZE;
                const glm::vec3 tr = base_center + up * PLAYHEAD_FRUSTUM_SIZE + right * PLAYHEAD_FRUSTUM_SIZE;
                const glm::vec3 bl = base_center - up * PLAYHEAD_FRUSTUM_SIZE - right * PLAYHEAD_FRUSTUM_SIZE;
                const glm::vec3 br = base_center - up * PLAYHEAD_FRUSTUM_SIZE + right * PLAYHEAD_FRUSTUM_SIZE;

                const ImVec2 s_apex = projectToScreen(apex);
                const ImVec2 s_tl = projectToScreen(tl);
                const ImVec2 s_tr = projectToScreen(tr);
                const ImVec2 s_bl = projectToScreen(bl);
                const ImVec2 s_br = projectToScreen(br);

                dl->AddLine(s_apex, s_tl, playhead_color, FRUSTUM_THICKNESS);
                dl->AddLine(s_apex, s_tr, playhead_color, FRUSTUM_THICKNESS);
                dl->AddLine(s_apex, s_bl, playhead_color, FRUSTUM_THICKNESS);
                dl->AddLine(s_apex, s_br, playhead_color, FRUSTUM_THICKNESS);

                dl->AddLine(s_tl, s_tr, playhead_color, FRUSTUM_THICKNESS);
                dl->AddLine(s_tr, s_br, playhead_color, FRUSTUM_THICKNESS);
                dl->AddLine(s_br, s_bl, playhead_color, FRUSTUM_THICKNESS);
                dl->AddLine(s_bl, s_tl, playhead_color, FRUSTUM_THICKNESS);

                const glm::vec3 up_tip = base_center + up * PLAYHEAD_FRUSTUM_SIZE * 1.3f;
                const ImVec2 s_up = projectToScreen(up_tip);
                dl->AddTriangleFilled(s_up, s_tl, s_tr, playhead_color);
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
            keyframe_pos_before_drag_ = kf->position;
            keyframe_rot_before_drag_ = kf->rotation;
        }

        if (changed) {
            const glm::vec3 new_pos(gizmo_matrix[3]);
            const glm::quat new_rot = glm::quat_cast(glm::mat3(gizmo_matrix));
            controller_.timeline().updateKeyframe(*selected, new_pos, new_rot, kf->fov);
            controller_.updateLoopKeyframe();
            pip_needs_update_ = true;
        }

        if (!is_using && keyframe_gizmo_active_) {
            keyframe_gizmo_active_ = false;
        }

        dl->PopClipRect();
    }

    void SequencerUIManager::renderContextMenu() {
        if (!context_menu_open_)
            return;

        const auto& timeline = controller_.timeline();
        if (ImGui::BeginPopup("KeyframeContextMenu")) {
            if (ImGui::MenuItem("Add Keyframe Here", "K")) {
                lfs::core::events::cmd::SequencerAddKeyframe{}.emit();
            }
            if (context_menu_keyframe_.has_value() && *context_menu_keyframe_ < timeline.size()) {
                ImGui::Separator();
                if (ImGui::MenuItem("Update to Current View", "U")) {
                    controller_.selectKeyframe(*context_menu_keyframe_);
                    lfs::core::events::cmd::SequencerUpdateKeyframe{}.emit();
                }
                if (ImGui::MenuItem("Go to Keyframe")) {
                    controller_.selectKeyframe(*context_menu_keyframe_);
                    controller_.seek(timeline.keyframes()[*context_menu_keyframe_].time);
                }
                ImGui::Separator();
                const bool translate_active = keyframe_gizmo_op_ == ImGuizmo::TRANSLATE;
                const bool rotate_active = keyframe_gizmo_op_ == ImGuizmo::ROTATE;
                if (ImGui::MenuItem("Move (Translate)", nullptr, translate_active)) {
                    controller_.selectKeyframe(*context_menu_keyframe_);
                    keyframe_gizmo_op_ = translate_active ? ImGuizmo::OPERATION(0) : ImGuizmo::TRANSLATE;
                }
                if (ImGui::MenuItem("Rotate", nullptr, rotate_active)) {
                    controller_.selectKeyframe(*context_menu_keyframe_);
                    keyframe_gizmo_op_ = rotate_active ? ImGuizmo::OPERATION(0) : ImGuizmo::ROTATE;
                }
                ImGui::Separator();
                const size_t idx = *context_menu_keyframe_;
                const bool is_last = (idx == timeline.size() - 1);
                if (ImGui::BeginMenu("Easing", !is_last)) {
                    static constexpr const char* EASING_NAMES[] = {"Linear", "Ease In", "Ease Out", "Ease In-Out"};
                    const auto current_easing = timeline.keyframes()[idx].easing;
                    for (int e = 0; e < 4; ++e) {
                        const auto easing = static_cast<sequencer::EasingType>(e);
                        if (ImGui::MenuItem(EASING_NAMES[e], nullptr, current_easing == easing)) {
                            if (current_easing != easing) {
                                controller_.timeline().setKeyframeEasing(idx, easing);
                            }
                        }
                    }
                    ImGui::EndMenu();
                }
                if (is_last && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                    ImGui::SetTooltip("Easing controls outgoing motion\n(last keyframe has no outgoing segment)");
                }
                ImGui::Separator();
                const bool is_first = (*context_menu_keyframe_ == 0);
                if (ImGui::MenuItem("Delete Keyframe", "Del", false, !is_first)) {
                    controller_.selectKeyframe(*context_menu_keyframe_);
                    controller_.removeSelectedKeyframe();
                }
                if (is_first && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                    ImGui::SetTooltip("Cannot delete first keyframe");
                }
            }
            ImGui::EndPopup();
        } else {
            context_menu_open_ = false;
            context_menu_keyframe_ = std::nullopt;
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
        const bool is_playing = !controller_.isStopped();
        const auto selected = controller_.selectedKeyframe();

        if (!is_playing && !selected.has_value()) {
            pip_last_keyframe_ = std::nullopt;
            return;
        }

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
        float cam_fov;

        if (is_playing) {
            const auto state = controller_.currentCameraState();
            cam_rot = glm::mat3_cast(state.rotation);
            cam_pos = state.position;
            cam_fov = state.fov;
        } else {
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
            cam_fov = kf->fov;
        }

        if (rm->renderPreviewFrame(sm, cam_rot, cam_pos, cam_fov, pip_fbo_, pip_texture_, PREVIEW_WIDTH, PREVIEW_HEIGHT)) {
            pip_last_render_time_ = now;
            if (!is_playing) {
                pip_last_keyframe_ = selected;
                pip_needs_update_ = false;
            }
        }
    }

    void SequencerUIManager::drawPipPreviewWindow(const ViewportLayout& viewport) {
        const bool is_playing = !controller_.isStopped();
        const auto selected = controller_.selectedKeyframe();

        if (!is_playing && !selected.has_value())
            return;
        if (!pip_initialized_ || pip_texture_ == 0)
            return;

        if (!is_playing) {
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
        constexpr float PANEL_HEIGHT = 90.0f;
        const float scaled_width = static_cast<float>(PREVIEW_WIDTH) * scale;
        const float scaled_height = static_cast<float>(PREVIEW_HEIGHT) * scale;
        const ImVec2 window_size(scaled_width, scaled_height + 24.0f);
        const ImVec2 window_pos(
            viewport.pos.x + MARGIN,
            viewport.pos.y + viewport.size.y - PANEL_HEIGHT - window_size.y - MARGIN);

        ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always);
        ImGui::SetNextWindowSize(window_size, ImGuiCond_Always);
        ImGui::SetNextWindowBgAlpha(0.95f);

        const ImU32 border_color = is_playing
                                       ? t.error_u32()
                                       : toU32WithAlpha(t.palette.primary, 0.6f);

        ImGui::PushStyleColor(ImGuiCol_WindowBg, toU32(t.palette.surface));
        ImGui::PushStyleColor(ImGuiCol_Border, border_color);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, t.sizes.window_rounding);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 2.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {4.0f, 4.0f});

        constexpr ImGuiWindowFlags flags =
            ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus |
            ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoScrollbar;

        if (ImGui::Begin("##KeyframePreview", nullptr, flags)) {
            const std::string title = is_playing
                                          ? std::format("Playback {:.2f}s", controller_.playhead())
                                          : std::format("Keyframe {} Preview", *selected + 1);
            ImGui::TextColored({t.palette.text.x, t.palette.text.y, t.palette.text.z, 0.8f}, "%s", title.c_str());
            ImGui::Image(static_cast<ImTextureID>(pip_texture_),
                         {scaled_width - 8.0f, scaled_height - 8.0f}, {0, 1}, {1, 0});
        }
        ImGui::End();
        ImGui::PopStyleVar(3);
        ImGui::PopStyleColor(2);
    }

} // namespace lfs::vis::gui
