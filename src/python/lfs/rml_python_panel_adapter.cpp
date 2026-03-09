/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rml_python_panel_adapter.hpp"
#include "core/logger.hpp"
#include "py_rml.hpp"
#include "py_ui.hpp"
#include "python/gil.hpp"
#include "python/python_runtime.hpp"

#include <algorithm>
#include <cassert>
#include <mutex>
#include <unordered_set>

namespace lfs::vis::gui {

    namespace {
        void warnLegacyRmlImguiPathOnce(const char* feature) {
            static std::mutex mutex;
            static std::unordered_set<std::string> warned_features;
            std::lock_guard lock(mutex);
            if (warned_features.emplace(feature).second) {
                LOG_WARN("Rml transition: '{}' is a legacy ImGui compatibility path. "
                         "Keep it for compatibility, but do not add new usage.",
                         feature);
            }
        }
    } // namespace

    bool RmlPythonPanelAdapter::ensureHost() {
        if (host_)
            return true;

        const auto& ops = lfs::python::get_rml_panel_host_ops();
        assert(ops.create);

        host_ = ops.create(manager_, context_name_.c_str(), rml_path_.c_str());
        if (!host_)
            return false;

        if (height_mode_ != 0 && ops.set_height_mode)
            ops.set_height_mode(host_, height_mode_);
        if (foreground_ && ops.set_foreground)
            ops.set_foreground(host_, true);
        return true;
    }

    void RmlPythonPanelAdapter::cachePythonCapabilities() {
        if (!draw_imgui_checked_) {
            has_draw_imgui_ = nb::hasattr(panel_instance_, "draw_imgui");
            draw_imgui_checked_ = true;
            if (has_draw_imgui_)
                warnLegacyRmlImguiPathOnce("RmlPanel.draw_imgui");
        }
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
                LOG_ERROR("RmlPanel update_interval_ms error: {}", e.what());
            }
        }
    }

    void RmlPythonPanelAdapter::bindModelIfNeeded() {
        if (model_bound_)
            return;

        const auto& ops = lfs::python::get_rml_panel_host_ops();
        if (!ops.ensure_context || !ops.get_context || !lfs::python::can_acquire_gil())
            return;

        const lfs::python::GilAcquire gil;
        cachePythonCapabilities();
        if (!has_bind_model_) {
            model_bound_ = true;
            return;
        }

        if (!ops.ensure_context(host_))
            return;

        auto* rml_ctx = static_cast<Rml::Context*>(ops.get_context(host_));
        assert(rml_ctx);
        try {
            auto py_ctx = lfs::python::PyRmlContext(rml_ctx);
            panel_instance_.attr("on_bind_model")(py_ctx);
            model_bound_ = true;
        } catch (const std::exception& e) {
            LOG_ERROR("RmlPanel on_bind_model error: {}", e.what());
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

        if (!loaded_ && lfs::python::can_acquire_gil()) {
            const lfs::python::GilAcquire gil;
            cachePythonCapabilities();
            lfs::python::RmlDocumentRegistry::instance().register_document(context_name_, doc);
            try {
                auto py_doc = lfs::python::PyRmlDocument(doc);
                panel_instance_.attr("on_load")(py_doc);
                content_dirty_ = true;
            } catch (const std::exception& e) {
                LOG_ERROR("RmlPanel on_load error: {}", e.what());
            }
            loaded_ = true;
        }

        return doc;
    }

    void RmlPythonPanelAdapter::syncDirectLayout(float w, float h) {
        if (!ensureDocumentInitialized())
            return;

        const auto& ops = lfs::python::get_rml_panel_host_ops();
        if (ops.prepare_layout)
            ops.prepare_layout(host_, w, h);
    }

    std::chrono::milliseconds RmlPythonPanelAdapter::updateInterval() const {
        return std::chrono::milliseconds(update_interval_ms_);
    }

    Rml::ElementDocument* RmlPythonPanelAdapter::prepareForRender(const PanelDrawContext* ctx) {
        auto* doc = ensureDocumentInitialized();
        if (!doc || !lfs::python::can_acquire_gil())
            return doc;

        const auto& ops = lfs::python::get_rml_panel_host_ops();
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
            const lfs::python::GilAcquire gil;
            auto py_doc = lfs::python::PyRmlDocument(doc);

            if (scene_changed) {
                try {
                    panel_instance_.attr("on_scene_changed")(py_doc);
                    pending_dirty = true;
                } catch (const std::exception& e) {
                    LOG_ERROR("RmlPanel on_scene_changed error: {}", e.what());
                }
                last_scene_gen_ = ctx->scene_generation;
            }

            try {
                nb::object result = panel_instance_.attr("on_update")(py_doc);
                pending_dirty |= !result.is_none() && nb::cast<bool>(result);
            } catch (const std::exception& e) {
                LOG_ERROR("RmlPanel on_update error: {}", e.what());
            }
            pending_dirty |= lfs::python::consume_document_dirty(doc);
            next_update_at_ = now + updateInterval();
        }

        pending_dirty |= lfs::python::consume_document_dirty(doc);

        if (pending_dirty && ops.mark_content_dirty)
            ops.mark_content_dirty(host_);
        if (frame_serial != 0)
            last_prepare_frame_ = frame_serial;
        content_dirty_ = false;
        return doc;
    }

    RmlPythonPanelAdapter::RmlPythonPanelAdapter(void* manager, nb::object panel_instance,
                                                 const std::string& context_name,
                                                 const std::string& rml_path,
                                                 bool has_poll, int height_mode)
        : manager_(manager),
          context_name_(context_name),
          rml_path_(rml_path),
          panel_instance_(std::move(panel_instance)),
          has_poll_(has_poll),
          height_mode_(height_mode) {
    }

    RmlPythonPanelAdapter::~RmlPythonPanelAdapter() {
        if (!host_)
            return;

        const auto& ops = lfs::python::get_rml_panel_host_ops();
        if (lfs::python::can_acquire_gil()) {
            const lfs::python::GilAcquire gil;
            if (loaded_ && nb::hasattr(panel_instance_, "on_unload")) {
                try {
                    auto* doc = static_cast<Rml::ElementDocument*>(
                        ops.get_document(host_));
                    if (doc) {
                        auto py_doc = lfs::python::PyRmlDocument(doc);
                        panel_instance_.attr("on_unload")(py_doc);
                    }
                } catch (const std::exception& e) {
                    LOG_ERROR("RmlPanel on_unload error: {}", e.what());
                }
            }
            assert(ops.destroy);
            ops.destroy(host_);
        } else {
            ops.destroy(host_);
        }
    }

    void RmlPythonPanelAdapter::draw(const PanelDrawContext& ctx) {
        const auto& ops = lfs::python::get_rml_panel_host_ops();
        assert(ops.create && ops.draw && ops.get_document && ops.is_loaded);

        if (!prepareForRender(&ctx))
            return;

        ops.draw(host_, &ctx);

        if (has_draw_imgui_) {
            try {
                lfs::python::PyUILayout layout;
                panel_instance_.attr("draw_imgui")(layout);
            } catch (const std::exception& e) {
                LOG_ERROR("RmlPanel draw_imgui error: {}", e.what());
            }
        }
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
            LOG_ERROR("RmlPanel poll error: {}", e.what());
            return false;
        }
    }

    void RmlPythonPanelAdapter::preload(const PanelDrawContext& ctx) {
        if (loaded_)
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
