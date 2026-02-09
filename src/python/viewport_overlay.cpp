/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "viewport_overlay.hpp"
#include "core/logger.hpp"
#include "gil.hpp"
#include "lfs/py_viewport.hpp"
#include "python_runtime.hpp"

#include <algorithm>
#include <cassert>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <imgui.h>

namespace lfs::python {

    namespace {

        bool has_handlers_impl() {
            return PyViewportDrawRegistry::instance().has_handlers();
        }

        void invoke_overlay_impl(const float* view_matrix, const float* proj_matrix,
                                 const float* vp_pos, const float* vp_size,
                                 const float* cam_pos, const float* cam_fwd,
                                 void* draw_list_ptr) {
            assert(view_matrix && proj_matrix && vp_pos && vp_size && cam_pos && cam_fwd);
            assert(draw_list_ptr);

            const auto view = glm::make_mat4(view_matrix);
            const auto proj = glm::make_mat4(proj_matrix);
            const glm::vec2 vp_p(vp_pos[0], vp_pos[1]);
            const glm::vec2 vp_s(vp_size[0], vp_size[1]);
            const glm::vec3 cp(cam_pos[0], cam_pos[1], cam_pos[2]);
            const glm::vec3 cf(cam_fwd[0], cam_fwd[1], cam_fwd[2]);

            PyViewportDrawContext draw_ctx;
            draw_ctx.set_camera_state(view, proj, vp_p, vp_s, cp, cf);

            if (!can_acquire_gil()) {
                LOG_DEBUG("Viewport overlay skipped: Python not ready");
                return;
            }

            auto& registry = PyViewportDrawRegistry::instance();
            registry.invoke_handlers(DrawHandlerTiming::PreView, draw_ctx);
            registry.invoke_handlers(DrawHandlerTiming::PostView, draw_ctx);
            registry.invoke_handlers(DrawHandlerTiming::PostUI, draw_ctx);

            auto* dl = static_cast<ImDrawList*>(draw_list_ptr);
            const float ox = vp_p.x;
            const float oy = vp_p.y;

            dl->PushClipRect({ox, oy}, {ox + vp_s.x, oy + vp_s.y}, true);

            for (const auto& cmd : draw_ctx.get_draw_commands()) {
                const auto to_u8 = [](float v) -> int {
                    return static_cast<int>(std::clamp(v * 255.0f + 0.5f, 0.0f, 255.0f));
                };
                const ImU32 color = IM_COL32(to_u8(cmd.r), to_u8(cmd.g), to_u8(cmd.b), to_u8(cmd.a));

                switch (cmd.type) {
                case PyViewportDrawContext::DrawCommand::LINE_2D:
                    dl->AddLine({cmd.x1, cmd.y1}, {cmd.x2, cmd.y2}, color, cmd.thickness);
                    break;
                case PyViewportDrawContext::DrawCommand::CIRCLE_2D:
                    dl->AddCircle({cmd.x1, cmd.y1}, cmd.radius, color, 0, cmd.thickness);
                    break;
                case PyViewportDrawContext::DrawCommand::RECT_2D:
                    dl->AddRect({cmd.x1, cmd.y1}, {cmd.x2, cmd.y2}, color, 0.0f, 0, cmd.thickness);
                    break;
                case PyViewportDrawContext::DrawCommand::FILLED_RECT_2D:
                    dl->AddRectFilled({cmd.x1, cmd.y1}, {cmd.x2, cmd.y2}, color);
                    break;
                case PyViewportDrawContext::DrawCommand::FILLED_CIRCLE_2D:
                    dl->AddCircleFilled({cmd.x1, cmd.y1}, cmd.radius, color);
                    break;
                case PyViewportDrawContext::DrawCommand::TEXT_2D:
                    if (cmd.font_size > 0.0f)
                        dl->AddText(ImGui::GetFont(), cmd.font_size, {cmd.x1, cmd.y1}, color, cmd.text.c_str());
                    else
                        dl->AddText({cmd.x1, cmd.y1}, color, cmd.text.c_str());
                    break;
                case PyViewportDrawContext::DrawCommand::LINE_3D: {
                    auto s = draw_ctx.world_to_screen({cmd.x1, cmd.y1, cmd.z1});
                    auto e = draw_ctx.world_to_screen({cmd.x2, cmd.y2, cmd.z2});
                    if (s && e) {
                        auto [sx, sy] = *s;
                        auto [ex, ey] = *e;
                        dl->AddLine({sx, sy}, {ex, ey}, color, cmd.thickness);
                    }
                    break;
                }
                case PyViewportDrawContext::DrawCommand::POINT_3D: {
                    auto p = draw_ctx.world_to_screen({cmd.x1, cmd.y1, cmd.z1});
                    if (p) {
                        auto [px, py] = *p;
                        dl->AddCircleFilled({px, py}, cmd.radius, color);
                    }
                    break;
                }
                case PyViewportDrawContext::DrawCommand::TEXT_3D: {
                    auto p = draw_ctx.world_to_screen({cmd.x1, cmd.y1, cmd.z1});
                    if (p) {
                        auto [px, py] = *p;
                        if (cmd.font_size > 0.0f)
                            dl->AddText(ImGui::GetFont(), cmd.font_size, {px, py}, color, cmd.text.c_str());
                        else
                            dl->AddText({px, py}, color, cmd.text.c_str());
                    }
                    break;
                }
                default:
                    assert(false && "Unknown DrawCommand type");
                    break;
                }
            }

            dl->PopClipRect();
        }

    } // namespace

    void register_viewport_overlay_bridge() {
        set_viewport_overlay_callbacks(has_handlers_impl, invoke_overlay_impl);
    }

} // namespace lfs::python
