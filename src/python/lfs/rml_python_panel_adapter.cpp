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

} // namespace lfs::vis::gui
