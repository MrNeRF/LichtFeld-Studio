/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/logger.hpp"
#include "py_ui.hpp"

#include <algorithm>
#include <cassert>

namespace lfs::python {

    namespace {
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
                    nb::object py_cb = std::move(modal.callback);
                    req.on_result = [py_cb = std::move(py_cb)](const lfs::core::ModalResult& result) {
                        nb::gil_scoped_acquire gil;
                        try {
                            py_cb(result.button_label);
                        } catch (const std::exception& e) {
                            LOG_ERROR("Modal callback error: {}", e.what());
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
                    nb::object py_cb = std::move(modal.callback);
                    req.on_result = [py_cb = std::move(py_cb)](const lfs::core::ModalResult& result) {
                        nb::gil_scoped_acquire gil;
                        try {
                            if (result.button_label == "OK")
                                py_cb(nb::str(result.input_value.c_str()));
                            else
                                py_cb(nb::none());
                        } catch (const std::exception& e) {
                            LOG_ERROR("Modal callback error: {}", e.what());
                        }
                    };
                    req.on_cancel = [py_cb_cancel = nb::object(py_cb)]() {
                        nb::gil_scoped_acquire gil;
                        try {
                            py_cb_cancel(nb::none());
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
                    nb::object py_cb = std::move(modal.callback);
                    req.on_result = [py_cb = std::move(py_cb)](const lfs::core::ModalResult&) {
                        nb::gil_scoped_acquire gil;
                        try {
                            py_cb();
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
    }

} // namespace lfs::python
