/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rml_python_panel_adapter.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/logger.hpp"
#include "py_rml.hpp"
#include "py_ui.hpp"
#include "python/gil.hpp"
#include "python/python_runtime.hpp"

#include <algorithm>
#include <cassert>
#include <imgui.h>

namespace lfs::vis::gui {

    bool RmlPythonPanelAdapter::isModelBound() const {
        return lifecycle_state_ == LifecycleState::ModelBound ||
               lifecycle_state_ == LifecycleState::Mounted;
    }

    bool RmlPythonPanelAdapter::isMounted() const {
        return lifecycle_state_ == LifecycleState::Mounted;
    }

    void RmlPythonPanelAdapter::setLifecycleState(const LifecycleState next_state) {
        switch (lifecycle_state_) {
        case LifecycleState::AwaitingModelBind:
            assert(next_state == LifecycleState::AwaitingModelBind ||
                   next_state == LifecycleState::ModelBound);
            break;
        case LifecycleState::ModelBound:
            assert(next_state == LifecycleState::AwaitingModelBind ||
                   next_state == LifecycleState::ModelBound ||
                   next_state == LifecycleState::Mounted);
            break;
        case LifecycleState::Mounted:
            assert(next_state == LifecycleState::AwaitingModelBind ||
                   next_state == LifecycleState::ModelBound ||
                   next_state == LifecycleState::Mounted);
            break;
        }
        lifecycle_state_ = next_state;
    }

    void RmlPythonPanelAdapter::resetLifecycle() {
        setLifecycleState(LifecycleState::AwaitingModelBind);
        content_dirty_ = true;
        last_prepare_frame_ = 0;
        next_update_at_ = std::chrono::steady_clock::time_point{};
    }

    bool RmlPythonPanelAdapter::ensureHost() {
        if (host_)
            return true;

        const auto& ops = lfs::python::get_rml_panel_host_ops();
        assert(ops.create);

        host_ = ops.create(manager_, context_name_.c_str(), rml_path_.c_str(), style_.c_str());
        if (!host_)
            return false;

        if (height_mode_ != 0 && ops.set_height_mode)
            ops.set_height_mode(host_, height_mode_);
        if (foreground_ && ops.set_foreground)
            ops.set_foreground(host_, true);
        return true;
    }

    void RmlPythonPanelAdapter::cachePythonCapabilities() {
        if (!bind_model_checked_) {
            has_bind_model_ = nb::hasattr(panel_instance_, "on_bind_model");
            bind_model_checked_ = true;
        }
        if (!has_update_interval_) {
            has_update_interval_ = true;
            try {
                if (nb::hasattr(panel_instance_, "update_interval_ms"))
                    update_interval_ms_ = std::max(0, nb::cast<int>(panel_instance_.attr("update_interval_ms")));
            } catch (const std::exception& e) {
                LOG_ERROR("Panel update_interval_ms error: {}", e.what());
            }
        }
    }

    void RmlPythonPanelAdapter::bindModelIfNeeded() {
        if (isModelBound())
            return;

        const auto& ops = lfs::python::get_rml_panel_host_ops();
        if (!ops.ensure_context || !ops.get_context || !lfs::python::can_acquire_gil())
            return;

        const lfs::python::GilAcquire gil;
        cachePythonCapabilities();
        if (!has_bind_model_) {
            setLifecycleState(LifecycleState::ModelBound);
            return;
        }

        if (!ops.ensure_context(host_))
            return;

        auto* rml_ctx = static_cast<Rml::Context*>(ops.get_context(host_));
        assert(rml_ctx);
        try {
            auto py_ctx = lfs::python::PyRmlContext(rml_ctx);
            panel_instance_.attr("on_bind_model")(py_ctx);
            setLifecycleState(LifecycleState::ModelBound);
        } catch (const std::exception& e) {
            LOG_ERROR("Panel on_bind_model error: {}", e.what());
        }
    }

    void RmlPythonPanelAdapter::callOnUnload(Rml::ElementDocument* doc) {
        if (!doc || !isMounted() || !lfs::python::can_acquire_gil())
            return;

        const lfs::python::GilAcquire gil;
        if (!nb::hasattr(panel_instance_, "on_unmount"))
            return;

        try {
            auto py_doc = lfs::python::PyRmlDocument(doc);
            panel_instance_.attr("on_unmount")(py_doc);
        } catch (const std::exception& e) {
            LOG_ERROR("Panel on_unmount error: {}", e.what());
        }
        setLifecycleState(LifecycleState::ModelBound);
    }

    void RmlPythonPanelAdapter::callOnLoad(Rml::ElementDocument* doc) {
        if (!doc || isMounted() || !lfs::python::can_acquire_gil())
            return;

        const lfs::python::GilAcquire gil;
        cachePythonCapabilities();
        lfs::python::RmlDocumentRegistry::instance().register_document(context_name_, doc);
        try {
            auto py_doc = lfs::python::PyRmlDocument(doc);
            panel_instance_.attr("on_mount")(py_doc);
            content_dirty_ = true;
            setLifecycleState(LifecycleState::Mounted);
        } catch (const std::exception& e) {
            LOG_ERROR("Panel on_mount error: {}", e.what());
        }
    }

    Rml::ElementDocument* RmlPythonPanelAdapter::ensureDocumentInitialized() {
        if (!ensureHost())
            return nullptr;

        bindModelIfNeeded();

        const auto& ops = lfs::python::get_rml_panel_host_ops();
        if (ops.ensure_document && !ops.ensure_document(host_))
            return nullptr;

        auto* doc = static_cast<Rml::ElementDocument*>(ops.get_document(host_));
        if (!doc)
            return nullptr;

        if (!isMounted()) {
            assert(isModelBound());
            callOnLoad(doc);
        }

        if (isMounted() && last_language_.empty())
            last_language_ = lfs::event::LocalizationManager::getInstance().getCurrentLanguage();

        return doc;
    }

    bool RmlPythonPanelAdapter::reloadDocumentForLanguage(const std::string& language) {
        if (!isMounted() || language.empty() || !lfs::python::can_acquire_gil())
            return false;

        const auto& ops = lfs::python::get_rml_panel_host_ops();
        if (!ops.reload_document || !ops.get_document || !ops.get_context)
            return false;

        const lfs::python::GilAcquire gil;

        auto* current_doc = static_cast<Rml::ElementDocument*>(ops.get_document(host_));
        callOnUnload(current_doc);
        lfs::python::RmlDocumentRegistry::instance().unregister_document(context_name_);

        resetLifecycle();
        bindModelIfNeeded();

        if (!ops.reload_document(host_)) {
            LOG_ERROR("Panel reload_document failed for '{}'", context_name_);
            resetLifecycle();
            return false;
        }

        auto* new_doc = static_cast<Rml::ElementDocument*>(ops.get_document(host_));
        if (!new_doc)
            return false;

        callOnLoad(new_doc);
        last_language_ = language;
        return true;
    }

    void RmlPythonPanelAdapter::reloadRmlResources() {
        if (!host_)
            return;

        const std::string current_language =
            lfs::event::LocalizationManager::getInstance().getCurrentLanguage();
        const std::string language = current_language.empty() ? last_language_ : current_language;
        if (!language.empty() && reloadDocumentForLanguage(language)) {
            layout_.release_elements();
            content_dirty_ = true;
            last_prepare_frame_ = 0;
            next_update_at_ = {};
            return;
        }

        if (!lfs::python::can_acquire_gil())
            return;

        const auto& ops = lfs::python::get_rml_panel_host_ops();
        if (!ops.reload_document || !ops.get_document)
            return;

        const lfs::python::GilAcquire gil;
        auto* current_doc = static_cast<Rml::ElementDocument*>(ops.get_document(host_));
        callOnUnload(current_doc);
        lfs::python::RmlDocumentRegistry::instance().unregister_document(context_name_);
        resetLifecycle();
        layout_.release_elements();

        if (!ops.reload_document(host_)) {
            LOG_ERROR("Panel reload_document failed for '{}'", context_name_);
            resetLifecycle();
            return;
        }

        bindModelIfNeeded();
        auto* new_doc = static_cast<Rml::ElementDocument*>(ops.get_document(host_));
        if (new_doc)
            callOnLoad(new_doc);

        last_language_ = language;
        content_dirty_ = true;
        last_prepare_frame_ = 0;
        next_update_at_ = {};
    }

    void RmlPythonPanelAdapter::syncDirectLayout(float w, float h) {
        if (!ensureDocumentInitialized())
            return;

        const auto& ops = lfs::python::get_rml_panel_host_ops();
        if (ops.prepare_layout)
            ops.prepare_layout(host_, w, h);
    }

    void RmlPythonPanelAdapter::drawImmediateLayout(Rml::ElementDocument* doc,
                                                    const PanelDrawContext* ctx) {
        if (!has_draw_ || !doc || !lfs::python::can_acquire_gil())
            return;

        assert(isMounted());

        if (lfs::python::bridge().prepare_ui)
            lfs::python::bridge().prepare_ui();

        lfs::python::MouseState mouse;
        if (current_input_) {
            mouse.pos_x = current_input_->mouse_x;
            mouse.pos_y = current_input_->mouse_y;
            if (have_prev_mouse_) {
                mouse.delta_x = mouse.pos_x - prev_mouse_x_;
                mouse.delta_y = mouse.pos_y - prev_mouse_y_;
            }
            mouse.wheel = current_input_->mouse_wheel;
            if (current_input_->mouse_clicked[0]) {
                constexpr auto kDoubleClickWindow = std::chrono::milliseconds(350);
                const auto now = std::chrono::steady_clock::now();
                mouse.double_clicked =
                    have_left_click_time_ && (now - last_left_click_at_) <= kDoubleClickWindow;
                last_left_click_at_ = now;
                have_left_click_time_ = true;
            }
            mouse.dragging = current_input_->mouse_down[0];
        } else {
            auto& io = ImGui::GetIO();
            mouse.pos_x = io.MousePos.x;
            mouse.pos_y = io.MousePos.y;
            mouse.delta_x = io.MouseDelta.x;
            mouse.delta_y = io.MouseDelta.y;
            mouse.wheel = io.MouseWheel;
            mouse.double_clicked = ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left);
            mouse.dragging = ImGui::IsMouseDragging(ImGuiMouseButton_Left);
        }
        prev_mouse_x_ = mouse.pos_x;
        prev_mouse_y_ = mouse.pos_y;
        have_prev_mouse_ = true;

        const lfs::python::SceneContextGuard scene_guard(ctx ? ctx->scene : nullptr);
        const lfs::python::GilAcquire gil;

        layout_.begin_frame(doc, mouse);
        try {
            panel_instance_.attr("draw")(nb::cast(layout_, nb::rv_policy::reference));
        } catch (const std::exception& e) {
            LOG_ERROR("Panel draw error: {}", e.what());
        }
        layout_.end_frame();

        const auto& ops = lfs::python::get_rml_panel_host_ops();
        if (ops.mark_content_dirty)
            ops.mark_content_dirty(host_);
    }

    std::chrono::milliseconds RmlPythonPanelAdapter::updateInterval() const {
        return std::chrono::milliseconds(update_interval_ms_);
    }

    Rml::ElementDocument* RmlPythonPanelAdapter::prepareForRender(const PanelDrawContext* ctx) {
        auto* doc = ensureDocumentInitialized();
        if (!doc || !lfs::python::can_acquire_gil())
            return doc;

        const auto& ops = lfs::python::get_rml_panel_host_ops();
        const auto& current_language =
            lfs::event::LocalizationManager::getInstance().getCurrentLanguage();
        if (!current_language.empty() && current_language != last_language_) {
            reloadDocumentForLanguage(current_language);
            doc = ops.get_document ? static_cast<Rml::ElementDocument*>(ops.get_document(host_))
                                   : nullptr;
            if (!doc)
                return nullptr;
        }

        const uint64_t frame_serial = ctx ? ctx->frame_serial : 0;
        if (frame_serial != 0 && last_prepare_frame_ == frame_serial)
            return doc;

        bool pending_dirty = content_dirty_ || lfs::python::consume_document_dirty(doc);
        const bool scene_changed = ctx && ctx->scene && ctx->scene_generation != last_scene_gen_;
        const auto now = std::chrono::steady_clock::now();
        cachePythonCapabilities();
        const bool update_due =
            next_update_at_ == std::chrono::steady_clock::time_point{} || now >= next_update_at_;
        const bool should_run_update = scene_changed || pending_dirty || update_due;

        if (should_run_update) {
            assert(isMounted());
            const lfs::python::GilAcquire gil;
            auto py_doc = lfs::python::PyRmlDocument(doc);

            if (scene_changed) {
                try {
                    panel_instance_.attr("on_scene_changed")(py_doc);
                    pending_dirty = true;
                } catch (const std::exception& e) {
                    LOG_ERROR("Panel on_scene_changed error: {}", e.what());
                }
                last_scene_gen_ = ctx->scene_generation;
            }

            try {
                nb::object result = panel_instance_.attr("on_update")(py_doc);
                pending_dirty |= !result.is_none() && nb::cast<bool>(result);
            } catch (const std::exception& e) {
                LOG_ERROR("Panel on_update error: {}", e.what());
            }
            pending_dirty |= lfs::python::consume_document_dirty(doc);
            next_update_at_ = now + updateInterval();
        }

        pending_dirty |= lfs::python::consume_document_dirty(doc);

        if (pending_dirty && ops.mark_content_dirty)
            ops.mark_content_dirty(host_);

        drawImmediateLayout(doc, ctx);

        if (frame_serial != 0)
            last_prepare_frame_ = frame_serial;
        content_dirty_ = false;
        return doc;
    }

    RmlPythonPanelAdapter::RmlPythonPanelAdapter(void* manager, nb::object panel_instance,
                                                 const std::string& context_name,
                                                 const std::string& rml_path,
                                                 const std::string& style, bool has_poll,
                                                 const int height_mode, const bool has_draw)
        : manager_(manager),
          context_name_(context_name),
          rml_path_(rml_path),
          style_(style),
          panel_instance_(std::move(panel_instance)),
          has_poll_(has_poll),
          has_draw_(has_draw),
          height_mode_(height_mode) {
    }

    RmlPythonPanelAdapter::~RmlPythonPanelAdapter() {
        layout_.release_elements();
        if (!host_)
            return;

        const auto& ops = lfs::python::get_rml_panel_host_ops();
        if (ops.get_document)
            callOnUnload(static_cast<Rml::ElementDocument*>(ops.get_document(host_)));
        lfs::python::RmlDocumentRegistry::instance().unregister_document(context_name_);
        assert(ops.destroy);
        ops.destroy(host_);
    }

    void RmlPythonPanelAdapter::draw(const PanelDrawContext& ctx) {
        const auto& ops = lfs::python::get_rml_panel_host_ops();
        assert(ops.create && ops.draw && ops.get_document && ops.is_loaded);

        if (!prepareForRender(&ctx))
            return;

        ops.draw(host_, &ctx);
    }

    void RmlPythonPanelAdapter::drawDirect(float x, float y, float w, float h,
                                           const PanelDrawContext& ctx) {
        const auto& ops = lfs::python::get_rml_panel_host_ops();
        assert(ops.create && ops.draw_direct && ops.get_document && ops.is_loaded);

        if (ctx.frame_serial == 0 || last_prepare_frame_ != ctx.frame_serial)
            syncDirectLayout(w, h);

        if (!prepareForRender(&ctx))
            return;

        ops.draw_direct(host_, x, y, w, h);
    }

    bool RmlPythonPanelAdapter::poll(const PanelDrawContext& ctx) {
        (void)ctx;
        if (!has_poll_)
            return true;
        if (!lfs::python::can_acquire_gil())
            return false;
        if (lfs::python::bridge().prepare_ui)
            lfs::python::bridge().prepare_ui();

        const lfs::python::GilAcquire gil;
        try {
            return nb::cast<bool>(panel_instance_.attr("poll")(lfs::python::get_app_context()));
        } catch (const std::exception& e) {
            LOG_ERROR("Panel poll error: {}", e.what());
            return false;
        }
    }

    void RmlPythonPanelAdapter::preload(const PanelDrawContext& ctx) {
        if (isMounted())
            return;
        prepareForRender(&ctx);
    }

    void RmlPythonPanelAdapter::preloadDirect(float w, float h, const PanelDrawContext& ctx,
                                              float clip_y_min, float clip_y_max,
                                              const PanelInputState* input) {
        if (!prepareForRender(&ctx))
            return;

        const auto& ops = lfs::python::get_rml_panel_host_ops();
        if (!ops.prepare_direct)
            return;

        if (ops.set_input_clip_y)
            ops.set_input_clip_y(host_, clip_y_min, clip_y_max);
        if (ops.set_input)
            ops.set_input(host_, input);

        ops.prepare_direct(host_, w, h);

        if (ops.set_input)
            ops.set_input(host_, nullptr);
        if (ops.set_input_clip_y)
            ops.set_input_clip_y(host_, -1.0f, -1.0f);
    }

    float RmlPythonPanelAdapter::getDirectDrawHeight() const {
        if (!host_)
            return 0.0f;
        const auto& ops = lfs::python::get_rml_panel_host_ops();
        return ops.get_content_height ? ops.get_content_height(host_) : 0.0f;
    }

    void RmlPythonPanelAdapter::setInputClipY(float y_min, float y_max) {
        if (host_) {
            const auto& ops = lfs::python::get_rml_panel_host_ops();
            if (ops.set_input_clip_y)
                ops.set_input_clip_y(host_, y_min, y_max);
        }
    }

    void RmlPythonPanelAdapter::setInput(const PanelInputState* input) {
        if (input)
            current_input_ = *input;
        else
            current_input_.reset();

        if (host_) {
            const auto& ops = lfs::python::get_rml_panel_host_ops();
            if (ops.set_input)
                ops.set_input(host_, input);
        }
    }

    void RmlPythonPanelAdapter::setForcedHeight(float h) {
        if (host_) {
            const auto& ops = lfs::python::get_rml_panel_host_ops();
            if (ops.set_forced_height)
                ops.set_forced_height(host_, h);
        }
    }

    bool RmlPythonPanelAdapter::wantsKeyboard() const {
        return false;
    }

    bool RmlPythonPanelAdapter::needsAnimationFrame() const {
        if (content_dirty_)
            return true;
        if (!host_)
            return false;

        const auto& ops = lfs::python::get_rml_panel_host_ops();
        if (ops.get_document) {
            auto* doc = static_cast<Rml::ElementDocument*>(ops.get_document(host_));
            if (lfs::python::is_document_dirty(doc))
                return true;
        }

        return ops.needs_animation ? ops.needs_animation(host_) : false;
    }

    void RmlPythonPanelAdapter::setForeground(bool fg) {
        foreground_ = fg;
        if (host_) {
            const auto& ops = lfs::python::get_rml_panel_host_ops();
            if (ops.set_foreground)
                ops.set_foreground(host_, fg);
        }
    }

} // namespace lfs::vis::gui
