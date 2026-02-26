/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rml_python_panel_adapter.hpp"
#include "core/logger.hpp"
#include "py_rml.hpp"
#include "py_ui.hpp"
#include "python/gil.hpp"
#include "python/python_runtime.hpp"

#include <cassert>

namespace lfs::vis::gui {

    RmlPythonPanelAdapter::RmlPythonPanelAdapter(void* manager, nb::object panel_instance,
                                                 const std::string& context_name,
                                                 const std::string& rml_path,
                                                 int height_mode)
        : manager_(manager),
          context_name_(context_name),
          rml_path_(rml_path),
          panel_instance_(std::move(panel_instance)),
          height_mode_(height_mode) {
    }

    RmlPythonPanelAdapter::~RmlPythonPanelAdapter() {
        if (host_) {
            if (loaded_ && lfs::python::can_acquire_gil()) {
                const lfs::python::GilAcquire gil;
                if (nb::hasattr(panel_instance_, "on_unload")) {
                    try {
                        const auto& ops = lfs::python::get_rml_panel_host_ops();
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
            }
            const auto& ops = lfs::python::get_rml_panel_host_ops();
            assert(ops.destroy);
            ops.destroy(host_);
        }
    }

    void RmlPythonPanelAdapter::draw(const PanelDrawContext& ctx) {
        const auto& ops = lfs::python::get_rml_panel_host_ops();
        assert(ops.create && ops.draw && ops.get_document && ops.is_loaded);

        if (!host_) {
            host_ = ops.create(manager_, context_name_.c_str(), rml_path_.c_str());
            if (!host_)
                return;

            if (height_mode_ != 0 && ops.set_height_mode)
                ops.set_height_mode(host_, height_mode_);
            if (foreground_ && ops.set_foreground)
                ops.set_foreground(host_, true);
        }

        if (!model_bound_ && ops.ensure_context && ops.get_context && lfs::python::can_acquire_gil()) {
            const lfs::python::GilAcquire gil;
            has_bind_model_ = nb::hasattr(panel_instance_, "on_bind_model");
            if (!draw_imgui_checked_) {
                has_draw_imgui_ = nb::hasattr(panel_instance_, "draw_imgui");
                draw_imgui_checked_ = true;
            }

            if (has_bind_model_) {
                if (ops.ensure_context(host_)) {
                    auto* rml_ctx = static_cast<Rml::Context*>(ops.get_context(host_));
                    assert(rml_ctx);
                    try {
                        auto py_ctx = lfs::python::PyRmlContext(rml_ctx);
                        panel_instance_.attr("on_bind_model")(py_ctx);
                    } catch (const std::exception& e) {
                        LOG_ERROR("RmlPanel on_bind_model error: {}", e.what());
                    }
                }
            }
            model_bound_ = true;
        }

        ops.draw(host_, &ctx);

        auto* doc = static_cast<Rml::ElementDocument*>(ops.get_document(host_));
        if (!doc)
            return;

        if (!lfs::python::can_acquire_gil())
            return;

        const lfs::python::GilAcquire gil;

        if (!loaded_) {
            lfs::python::RmlDocumentRegistry::instance().register_document(
                context_name_, doc);

            if (!draw_imgui_checked_) {
                has_draw_imgui_ = nb::hasattr(panel_instance_, "draw_imgui");
                draw_imgui_checked_ = true;
            }

            try {
                auto py_doc = lfs::python::PyRmlDocument(doc);
                panel_instance_.attr("on_load")(py_doc);
            } catch (const std::exception& e) {
                LOG_ERROR("RmlPanel on_load error: {}", e.what());
            }
            loaded_ = true;
        }

        try {
            auto py_doc = lfs::python::PyRmlDocument(doc);
            panel_instance_.attr("on_update")(py_doc);
        } catch (const std::exception& e) {
            LOG_ERROR("RmlPanel on_update error: {}", e.what());
        }

        if (ctx.scene && ctx.scene_generation != last_scene_gen_) {
            try {
                auto py_doc = lfs::python::PyRmlDocument(doc);
                panel_instance_.attr("on_scene_changed")(py_doc);
            } catch (const std::exception& e) {
                LOG_ERROR("RmlPanel on_scene_changed error: {}", e.what());
            }
            last_scene_gen_ = ctx.scene_generation;
            if (ops.mark_content_dirty)
                ops.mark_content_dirty(host_);
        }

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

        if (!host_) {
            host_ = ops.create(manager_, context_name_.c_str(), rml_path_.c_str());
            if (!host_)
                return;
            if (height_mode_ != 0 && ops.set_height_mode)
                ops.set_height_mode(host_, height_mode_);
            if (foreground_ && ops.set_foreground)
                ops.set_foreground(host_, true);
        }

        if (!model_bound_ && ops.ensure_context && ops.get_context && lfs::python::can_acquire_gil()) {
            const lfs::python::GilAcquire gil;
            has_bind_model_ = nb::hasattr(panel_instance_, "on_bind_model");
            if (!draw_imgui_checked_) {
                has_draw_imgui_ = nb::hasattr(panel_instance_, "draw_imgui");
                draw_imgui_checked_ = true;
            }
            if (has_bind_model_) {
                if (ops.ensure_context(host_)) {
                    auto* rml_ctx = static_cast<Rml::Context*>(ops.get_context(host_));
                    assert(rml_ctx);
                    try {
                        auto py_ctx = lfs::python::PyRmlContext(rml_ctx);
                        panel_instance_.attr("on_bind_model")(py_ctx);
                    } catch (const std::exception& e) {
                        LOG_ERROR("RmlPanel on_bind_model error: {}", e.what());
                    }
                }
            }
            model_bound_ = true;
        }

        ops.draw_direct(host_, x, y, w, h);

        auto* doc = static_cast<Rml::ElementDocument*>(ops.get_document(host_));
        if (!doc)
            return;
        if (!lfs::python::can_acquire_gil())
            return;

        const lfs::python::GilAcquire gil;

        if (!loaded_) {
            lfs::python::RmlDocumentRegistry::instance().register_document(context_name_, doc);
            if (!draw_imgui_checked_) {
                has_draw_imgui_ = nb::hasattr(panel_instance_, "draw_imgui");
                draw_imgui_checked_ = true;
            }
            try {
                auto py_doc = lfs::python::PyRmlDocument(doc);
                panel_instance_.attr("on_load")(py_doc);
            } catch (const std::exception& e) {
                LOG_ERROR("RmlPanel on_load error: {}", e.what());
            }
            loaded_ = true;
        }

        try {
            auto py_doc = lfs::python::PyRmlDocument(doc);
            panel_instance_.attr("on_update")(py_doc);
        } catch (const std::exception& e) {
            LOG_ERROR("RmlPanel on_update error: {}", e.what());
        }

        if (ctx.scene && ctx.scene_generation != last_scene_gen_) {
            try {
                auto py_doc = lfs::python::PyRmlDocument(doc);
                panel_instance_.attr("on_scene_changed")(py_doc);
            } catch (const std::exception& e) {
                LOG_ERROR("RmlPanel on_scene_changed error: {}", e.what());
            }
            last_scene_gen_ = ctx.scene_generation;
            if (ops.mark_content_dirty)
                ops.mark_content_dirty(host_);
        }
    }

    float RmlPythonPanelAdapter::getDirectDrawHeight() const {
        if (!host_)
            return 0.0f;
        const auto& ops = lfs::python::get_rml_panel_host_ops();
        return ops.get_content_height ? ops.get_content_height(host_) : 0.0f;
    }

    bool RmlPythonPanelAdapter::hasImguiOverlay() const {
        return has_draw_imgui_;
    }

    void RmlPythonPanelAdapter::drawImguiOverlay(const PanelDrawContext& ctx) {
        (void)ctx;
        if (!has_draw_imgui_ || !lfs::python::can_acquire_gil())
            return;
        const lfs::python::GilAcquire gil;
        try {
            lfs::python::PyUILayout layout;
            panel_instance_.attr("draw_imgui")(layout);
        } catch (const std::exception& e) {
            LOG_ERROR("RmlPanel draw_imgui error: {}", e.what());
        }
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
