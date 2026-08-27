/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/logger.hpp"
#include "py_ui.hpp"
#include "python/python_runtime.hpp"
#include "visualizer/gui/gui_manager.hpp"
#include "visualizer/gui/rml_modal_overlay.hpp"
#include "visualizer/post_work_utils.hpp"
#include "visualizer/visualizer.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

namespace lfs::python {

    namespace {

        using SafePyFunc = std::shared_ptr<nb::object>;

        SafePyFunc make_safe_py_func(nb::object func) {
            return {new nb::object(std::move(func)), [](nb::object* p) {
                        nb::gil_scoped_acquire gil;
                        delete p;
                    }};
        }

        std::string escapeRml(const std::string& text) {
            std::string result;
            result.reserve(text.size() + text.size() / 8);
            for (char c : text) {
                switch (c) {
                case '<': result += "&lt;"; break;
                case '>': result += "&gt;"; break;
                case '&': result += "&amp;"; break;
                case '"': result += "&quot;"; break;
                case '\n': result += "<br/>"; break;
                default: result += c;
                }
            }
            return result;
        }

        lfs::core::ModalStyle convertStyle(MessageStyle style) {
            switch (style) {
            case MessageStyle::Warning: return lfs::core::ModalStyle::Warning;
            case MessageStyle::Error: return lfs::core::ModalStyle::Error;
            default: return lfs::core::ModalStyle::Info;
            }
        }

        struct ModalView {
            std::optional<vis::gui::ModalSnapshot> snap;
            std::size_t pending = 0;
        };

        template <typename F>
        auto invoke_on_viewer(F&& fn, std::invoke_result_t<F> fallback) {
            auto* const viewer = get_visualizer();
            if (!viewer || viewer->isOnViewerThread())
                return std::invoke(std::forward<F>(fn));
            if (!viewer->acceptsPostedWork())
                return fallback;
            nb::gil_scoped_release release;
            return vis::post_work_and_wait(
                [viewer](vis::Visualizer::WorkItem work) { return viewer->postWork(std::move(work)); },
                std::forward<F>(fn),
                [fallback]() { return fallback; });
        }

        ModalView read_modal_view() {
            auto* const gui = get_gui_manager();
            auto* const overlay = gui ? gui->modalOverlay() : nullptr;
            if (!overlay)
                return {};
            return ModalView{overlay->current(), overlay->pending_count()};
        }

        bool press_modal_button(const std::string& label) {
            auto* const gui = get_gui_manager();
            auto* const overlay = gui ? gui->modalOverlay() : nullptr;
            if (!overlay)
                return false;
            return overlay->dismiss(label);
        }

        std::optional<nb::dict> modal_view_to_dict(const ModalView& view) {
            if (!view.snap)
                return std::nullopt;
            nb::dict result;
            result["title"] = view.snap->title;
            result["body"] = view.snap->body_text;
            nb::list buttons;
            const auto n = view.snap->button_labels.size() < view.snap->button_enabled.size()
                               ? view.snap->button_labels.size()
                               : view.snap->button_enabled.size();
            for (std::size_t i = 0; i < n; ++i) {
                nb::dict button;
                button["label"] = view.snap->button_labels[i];
                button["enabled"] = static_cast<bool>(view.snap->button_enabled[i]);
                buttons.append(button);
            }
            result["buttons"] = buttons;
            result["has_input"] = view.snap->has_input;
            result["pending"] = view.pending;
            return result;
        }
    } // namespace

    void PyModalRegistry::draw_modals() {
        std::vector<PyModalDialog> local_modals;

        {
            std::lock_guard lock(mutex_);
            if (modals_.empty())
                return;
            local_modals = std::move(modals_);
            modals_.clear();
        }

        if (!enqueue_cb_)
            return;

        for (auto& modal : local_modals) {
            lfs::core::ModalRequest req;
            req.title = modal.title;
            req.body_rml = escapeRml(modal.message);
            req.style = convertStyle(modal.style);

            switch (modal.type) {
            case ModalDialogType::Confirm: {
                for (size_t i = 0; i < modal.buttons.size(); ++i) {
                    const std::string style = (i == 0) ? "primary" : "secondary";
                    req.buttons.push_back({modal.buttons[i], style});
                }

                if (modal.cpp_callback) {
                    auto cpp_cb = std::move(modal.cpp_callback);
                    req.on_result = [cpp_cb = std::move(cpp_cb)](const lfs::core::ModalResult& result) {
                        cpp_cb(result.button_label);
                    };
                } else if (modal.callback.is_valid() && !modal.callback.is_none()) {
                    auto py_cb = make_safe_py_func(std::move(modal.callback));
                    req.on_result = [py_cb](const lfs::core::ModalResult& result) {
                        nb::gil_scoped_acquire gil;
                        try {
                            (*py_cb)(result.button_label);
                        } catch (const std::exception& e) {
                            LOG_ERROR("Modal callback error: {}", e.what());
                        }
                    };
                    // Cancellation is state cleanup, not an implicit click on
                    // whichever button happens to be last.
                    req.on_cancel = [py_cb]() {
                        nb::gil_scoped_acquire gil;
                        try {
                            (*py_cb)("");
                        } catch (const std::exception& e) {
                            LOG_ERROR("Modal cancel callback error: {}", e.what());
                        }
                    };
                }
                break;
            }
            case ModalDialogType::Input: {
                req.has_input = true;
                req.input_default = modal.input_value;
                req.buttons = {{"OK", "primary"}, {"Cancel", "secondary"}};

                if (modal.callback.is_valid() && !modal.callback.is_none()) {
                    auto py_cb = make_safe_py_func(std::move(modal.callback));
                    req.on_result = [py_cb](const lfs::core::ModalResult& result) {
                        nb::gil_scoped_acquire gil;
                        try {
                            if (result.button_label == "OK")
                                (*py_cb)(nb::str(result.input_value.c_str()));
                            else
                                (*py_cb)(nb::none());
                        } catch (const std::exception& e) {
                            LOG_ERROR("Modal callback error: {}", e.what());
                        }
                    };
                    req.on_cancel = [py_cb]() {
                        nb::gil_scoped_acquire gil;
                        try {
                            (*py_cb)(nb::none());
                        } catch (const std::exception& e) {
                            LOG_ERROR("Modal callback error: {}", e.what());
                        }
                    };
                }
                break;
            }
            case ModalDialogType::Message: {
                req.buttons = {{"OK", "primary"}};

                if (modal.callback.is_valid() && !modal.callback.is_none()) {
                    auto py_cb = make_safe_py_func(std::move(modal.callback));
                    req.on_result = [py_cb](const lfs::core::ModalResult&) {
                        nb::gil_scoped_acquire gil;
                        try {
                            (*py_cb)();
                        } catch (const std::exception& e) {
                            LOG_ERROR("Modal callback error: {}", e.what());
                        }
                    };
                }
                break;
            }
            }

            enqueue_cb_(std::move(req));
        }
    }

    void register_ui_modals(nb::module_& m) {
        m.def(
            "confirm_dialog",
            [](const std::string& title, const std::string& message,
               const std::vector<std::string>& buttons, nb::object callback) {
                PyModalRegistry::instance().show_confirm(title, message, buttons, callback);
            },
            nb::arg("title"), nb::arg("message"),
            nb::arg("buttons") = std::vector<std::string>{"OK", "Cancel"},
            nb::arg("callback") = nb::none(),
            "Show a confirmation dialog with custom buttons");

        m.def(
            "input_dialog",
            [](const std::string& title, const std::string& message,
               const std::string& default_value, nb::object callback) {
                PyModalRegistry::instance().show_input(title, message, default_value, callback);
            },
            nb::arg("title"), nb::arg("message"),
            nb::arg("default_value") = "",
            nb::arg("callback") = nb::none(),
            "Show an input dialog");

        m.def(
            "message_dialog",
            [](const std::string& title, const std::string& message,
               const std::string& style, nb::object callback) {
                MessageStyle msg_style = MessageStyle::Info;
                if (style == "warning")
                    msg_style = MessageStyle::Warning;
                else if (style == "error")
                    msg_style = MessageStyle::Error;
                PyModalRegistry::instance().show_message(title, message, msg_style, callback);
            },
            nb::arg("title"), nb::arg("message"),
            nb::arg("style") = "info",
            nb::arg("callback") = nb::none(),
            "Show a message dialog (style: 'info', 'warning', or 'error')");

        m.def(
            "modal_get",
            []() -> std::optional<nb::dict> {
                return modal_view_to_dict(invoke_on_viewer([] { return read_modal_view(); }, ModalView{}));
            },
            "Return the currently shown modal dialog as a dict, or None if none is open");

        m.def(
            "modal_press",
            [](const std::string& label) {
                return invoke_on_viewer([label] { return press_modal_button(label); }, false);
            },
            nb::arg("label"),
            "Press an enabled modal button by label. Returns False if no matching enabled button.");
    }

} // namespace lfs::python
