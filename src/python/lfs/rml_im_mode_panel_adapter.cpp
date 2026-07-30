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

namespace lfs::vis::gui {
    namespace {
        lfs::python::MouseState makeMouseState(const std::optional<PanelInputState>& input,
                                               float prev_mouse_x, float prev_mouse_y,
                                               bool have_prev_mouse,
                                               bool& have_left_click_time,
                                               std::chrono::steady_clock::time_point& last_left_click_at) {
            lfs::python::MouseState mouse;
            if (!input) {
                mouse.pos_x = prev_mouse_x;
                mouse.pos_y = prev_mouse_y;
                return mouse;
            }

            mouse.pos_x = input->mouse_x;
            mouse.pos_y = input->mouse_y;
            if (have_prev_mouse) {
                mouse.delta_x = mouse.pos_x - prev_mouse_x;
                mouse.delta_y = mouse.pos_y - prev_mouse_y;
            }
            mouse.wheel = input->mouse_wheel;
            mouse.right_clicked = input->mouse_clicked[1];
            if (input->mouse_clicked[0]) {
                constexpr auto kDoubleClickWindow = std::chrono::milliseconds(350);
                const auto now = std::chrono::steady_clock::now();
                mouse.double_clicked =
                    have_left_click_time && (now - last_left_click_at) <= kDoubleClickWindow;
                last_left_click_at = now;
                have_left_click_time = true;
            }
            mouse.dragging = input->mouse_down[0];
            return mouse;
        }
    } // namespace

    RmlImModePanelAdapter::RmlImModePanelAdapter(void* manager, nb::object panel_instance,
                                                 const bool has_poll,
                                                 const std::string& rml_path)
        : manager_(manager),
          rml_path_(rml_path),
          panel_instance_(std::move(panel_instance)),
          has_poll_(has_poll) {
        assert(manager_);
    }

    RmlImModePanelAdapter::~RmlImModePanelAdapter() {
        layout_.release_elements();
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
        host_ = ops.create(manager_, ctx_name.c_str(), rml_path_.c_str(), "");

        if (host_ && ops.set_height_mode)
            ops.set_height_mode(host_, 1);
        if (host_ && ops.set_floating)
            ops.set_floating(host_, floating_);
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

        const lfs::python::MouseState mouse = makeMouseState(
            current_input_, prev_mouse_x_, prev_mouse_y_, have_prev_mouse_,
            have_left_click_time_, last_left_click_at_);
        prev_mouse_x_ = mouse.pos_x;
        prev_mouse_y_ = mouse.pos_y;
        have_prev_mouse_ = true;

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

    bool RmlImModePanelAdapter::drawDirectCached(float x, float y, float w, float h,
                                                 const PanelDrawContext& ctx) {
        (void)ctx;
        if (!host_)
            return false;
        if (needsAnimationFrame())
            return false;

        const auto& ops = lfs::python::get_rml_panel_host_ops();
        return ops.draw_direct_cached ? ops.draw_direct_cached(host_, x, y, w, h) : false;
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
        if (input)
            current_input_ = *input;
        else
            current_input_.reset();

        if (!host_)
            return;

        const auto& ops = lfs::python::get_rml_panel_host_ops();
        if (ops.set_input)
            ops.set_input(host_, input);
    }

    void RmlImModePanelAdapter::setForcedHeight(float h) {
        if (host_) {
            const auto& ops = lfs::python::get_rml_panel_host_ops();
            if (ops.set_forced_height)
                ops.set_forced_height(host_, h);
        }
    }

    void RmlImModePanelAdapter::setPanelSpace(const PanelSpace space) {
        const bool floating = space == PanelSpace::Floating;
        if (floating_ == floating)
            return;

        floating_ = floating;
        if (host_) {
            const auto& ops = lfs::python::get_rml_panel_host_ops();
            if (ops.set_floating)
                ops.set_floating(host_, floating_);
        }
    }

    bool RmlImModePanelAdapter::needsAnimationFrame() const {
        if (!host_)
            return false;
        const auto& ops = lfs::python::get_rml_panel_host_ops();
        return ops.needs_animation ? ops.needs_animation(host_) : false;
    }

    void RmlImModePanelAdapter::reloadRmlResources() {
        if (!host_)
            return;

        const auto& ops = lfs::python::get_rml_panel_host_ops();
        if (!ops.reload_document)
            return;

        layout_.release_elements();
        if (!ops.reload_document(host_)) {
            LOG_ERROR("RmlImMode reload_document failed for '{}'", rml_path_);
            return;
        }
        if (ops.mark_content_dirty)
            ops.mark_content_dirty(host_);
        last_layout_frame_ = 0;
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
