/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rml_im_mode_panel_adapter.hpp"
#include "core/logger.hpp"
#include "py_ui.hpp"
#include "python/gil.hpp"
#include "python/python_runtime.hpp"

#include <RmlUi/Core/ElementDocument.h>
#include <cassert>
#include <imgui.h>

namespace lfs::vis::gui {

    static constexpr const char* IM_MODE_RML = "rmlui/im_mode_panel.rml";

    RmlImModePanelAdapter::RmlImModePanelAdapter(void* manager, nb::object panel_instance, bool has_poll)
        : manager_(manager),
          panel_instance_(std::move(panel_instance)),
          has_poll_(has_poll) {
        assert(manager_);
    }

    RmlImModePanelAdapter::~RmlImModePanelAdapter() {
        if (host_) {
            const auto& ops = lfs::python::get_rml_panel_host_ops();
            assert(ops.destroy);
            ops.destroy(host_);
        }
    }

    void RmlImModePanelAdapter::ensureHost() {
        if (host_)
            return;
        const auto& ops = lfs::python::get_rml_panel_host_ops();
        assert(ops.create);

        static int ctx_counter = 0;
        std::string ctx_name = "im_mode_" + std::to_string(ctx_counter++);
        host_ = ops.create(manager_, ctx_name.c_str(), IM_MODE_RML);

        if (host_ && ops.set_height_mode)
            ops.set_height_mode(host_, 1);
    }

    void RmlImModePanelAdapter::drawLayout(const PanelDrawContext* ctx) {
        const auto& ops = lfs::python::get_rml_panel_host_ops();
        if (ops.ensure_document && !ops.ensure_document(host_))
            return;

        auto* doc = static_cast<Rml::ElementDocument*>(ops.get_document(host_));
        if (!doc)
            return;

        if (!lfs::python::can_acquire_gil())
            return;

        const uint64_t frame_serial = ctx ? ctx->frame_serial : 0;
        if (frame_serial != 0 && last_layout_frame_ == frame_serial)
            return;

        if (lfs::python::bridge().prepare_ui)
            lfs::python::bridge().prepare_ui();

        const lfs::python::GilAcquire gil;

        lfs::python::MouseState mouse;
        auto& io = ImGui::GetIO();
        mouse.pos_x = io.MousePos.x;
        mouse.pos_y = io.MousePos.y;
        mouse.delta_x = io.MouseDelta.x;
        mouse.delta_y = io.MouseDelta.y;
        mouse.wheel = io.MouseWheel;
        mouse.double_clicked = ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left);
        mouse.dragging = ImGui::IsMouseDragging(ImGuiMouseButton_Left);

        layout_.begin_frame(doc, mouse);
        try {
            panel_instance_.attr("draw")(nb::cast(layout_, nb::rv_policy::reference));
        } catch (const std::exception& e) {
            LOG_ERROR("RmlImMode draw error: {}", e.what());
        }
        layout_.end_frame();

        if (ops.mark_content_dirty)
            ops.mark_content_dirty(host_);
        if (frame_serial != 0)
            last_layout_frame_ = frame_serial;
    }

    void RmlImModePanelAdapter::draw(const PanelDrawContext& ctx) {
        ensureHost();
        if (!host_)
            return;

        const auto& ops = lfs::python::get_rml_panel_host_ops();

        const lfs::python::SceneContextGuard scene_guard(ctx.scene);
        drawLayout(&ctx);

        ops.draw(host_, &ctx);
    }

    void RmlImModePanelAdapter::preloadDirect(float w, float h, const PanelDrawContext& ctx,
                                              float clip_y_min, float clip_y_max,
                                              const PanelInputState* input) {
        ensureHost();
        if (!host_)
            return;

        const auto& ops = lfs::python::get_rml_panel_host_ops();
        if (!ops.prepare_direct)
            return;

        if (ops.set_input_clip_y)
            ops.set_input_clip_y(host_, clip_y_min, clip_y_max);
        if (ops.set_input)
            ops.set_input(host_, input);

        const lfs::python::SceneContextGuard scene_guard(ctx.scene);
        drawLayout(&ctx);
        ops.prepare_direct(host_, w, h);

        if (ops.set_input)
            ops.set_input(host_, nullptr);
        if (ops.set_input_clip_y)
            ops.set_input_clip_y(host_, -1.0f, -1.0f);
    }

    void RmlImModePanelAdapter::drawDirect(float x, float y, float w, float h,
                                           const PanelDrawContext& ctx) {
        ensureHost();
        if (!host_)
            return;

        const auto& ops = lfs::python::get_rml_panel_host_ops();

        const lfs::python::SceneContextGuard scene_guard(ctx.scene);
        drawLayout(&ctx);

        ops.draw_direct(host_, x, y, w, h);
    }

    float RmlImModePanelAdapter::getDirectDrawHeight() const {
        if (!host_)
            return 0.0f;
        const auto& ops = lfs::python::get_rml_panel_host_ops();
        return ops.get_content_height ? ops.get_content_height(host_) : 0.0f;
    }

    void RmlImModePanelAdapter::setInputClipY(float y_min, float y_max) {
        if (host_) {
            const auto& ops = lfs::python::get_rml_panel_host_ops();
            if (ops.set_input_clip_y)
                ops.set_input_clip_y(host_, y_min, y_max);
        }
    }

    void RmlImModePanelAdapter::setInput(const PanelInputState* input) {
        if (host_) {
            const auto& ops = lfs::python::get_rml_panel_host_ops();
            if (ops.set_input)
                ops.set_input(host_, input);
        }
    }

    void RmlImModePanelAdapter::setForcedHeight(float h) {
        if (host_) {
            const auto& ops = lfs::python::get_rml_panel_host_ops();
            if (ops.set_forced_height)
                ops.set_forced_height(host_, h);
        }
    }

    bool RmlImModePanelAdapter::needsAnimationFrame() const {
        if (!host_)
            return false;
        const auto& ops = lfs::python::get_rml_panel_host_ops();
        return ops.needs_animation ? ops.needs_animation(host_) : false;
    }

    bool RmlImModePanelAdapter::poll(const PanelDrawContext& ctx) {
        (void)ctx;
        if (!has_poll_)
            return true;
        if (!lfs::python::can_acquire_gil())
            return false;
        if (lfs::python::bridge().prepare_ui)
            lfs::python::bridge().prepare_ui();
        const lfs::python::GilAcquire gil;
        return nb::cast<bool>(panel_instance_.attr("poll")(lfs::python::get_app_context()));
    }

} // namespace lfs::vis::gui
