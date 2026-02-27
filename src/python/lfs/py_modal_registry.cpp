/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_ui.hpp"

namespace lfs::python {

    PyModalRegistry& PyModalRegistry::instance() {
        static PyModalRegistry registry;
        return registry;
    }

    void PyModalRegistry::set_enqueue_callback(EnqueueCallback cb) {
        std::lock_guard lock(mutex_);
        enqueue_cb_ = std::move(cb);
    }

    void PyModalRegistry::show_confirm(const std::string& title, const std::string& message,
                                       const std::vector<std::string>& buttons, nb::object callback) {
        std::lock_guard lock(mutex_);
        PyModalDialog modal;
        modal.id = "modal_" + std::to_string(next_id_++);
        modal.title = title;
        modal.message = message;
        modal.buttons = buttons.empty() ? std::vector<std::string>{"OK", "Cancel"} : buttons;
        modal.callback = callback;
        modal.type = ModalDialogType::Confirm;
        modal.is_open = true;
        modal.needs_open = true;
        modals_.push_back(std::move(modal));
    }

    void PyModalRegistry::show_confirm(const std::string& title, const std::string& message,
                                       const std::vector<std::string>& buttons,
                                       std::function<void(const std::string&)> callback) {
        std::lock_guard lock(mutex_);
        PyModalDialog modal;
        modal.id = "modal_" + std::to_string(next_id_++);
        modal.title = title;
        modal.message = message;
        modal.buttons = buttons.empty() ? std::vector<std::string>{"OK", "Cancel"} : buttons;
        modal.cpp_callback = std::move(callback);
        modal.type = ModalDialogType::Confirm;
        modal.is_open = true;
        modal.needs_open = true;
        modals_.push_back(std::move(modal));
    }

    void PyModalRegistry::show_input(const std::string& title, const std::string& message,
                                     const std::string& default_value, nb::object callback) {
        std::lock_guard lock(mutex_);
        PyModalDialog modal;
        modal.id = "modal_" + std::to_string(next_id_++);
        modal.title = title;
        modal.message = message;
        modal.buttons = {"OK", "Cancel"};
        modal.callback = callback;
        modal.type = ModalDialogType::Input;
        modal.input_value = default_value;
        modal.is_open = true;
        modal.needs_open = true;
        modals_.push_back(std::move(modal));
    }

    void PyModalRegistry::show_message(const std::string& title, const std::string& message,
                                       MessageStyle style, nb::object callback) {
        std::lock_guard lock(mutex_);
        PyModalDialog modal;
        modal.id = "modal_" + std::to_string(next_id_++);
        modal.title = title;
        modal.message = message;
        modal.buttons = {"OK"};
        modal.callback = callback;
        modal.type = ModalDialogType::Message;
        modal.style = style;
        modal.is_open = true;
        modal.needs_open = true;
        modals_.push_back(std::move(modal));
    }

    bool PyModalRegistry::has_open_modals() const {
        std::lock_guard lock(mutex_);
        return !modals_.empty();
    }

    void PyModalRegistry::clear_for_test() {
        std::lock_guard lock(mutex_);
        modals_.clear();
        next_id_ = 0;
    }

    bool PyModalRegistry::can_lock_mutex_for_test() const {
        if (!mutex_.try_lock()) {
            return false;
        }
        mutex_.unlock();
        return true;
    }

    void PyModalRegistry::run_pending_callback_for_test(std::function<void()> callback) {
        std::vector<ModalCallbackAction> pending_callbacks;
        {
            std::lock_guard lock(mutex_);
            pending_callbacks.push_back(std::move(callback));
        }

        for (auto& pending_callback : pending_callbacks) {
            pending_callback();
        }
    }

} // namespace lfs::python
