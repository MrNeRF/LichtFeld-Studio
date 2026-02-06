/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/logger.hpp"
#include "gui/ui_widgets.hpp"
#include "py_ui.hpp"
#include "theme/theme.hpp"

#include <cstring>
#include <imgui.h>

namespace lfs::python {

    using lfs::vis::gui::widgets::ButtonStyle;
    using lfs::vis::gui::widgets::ColoredButton;

    static void draw_text_centered(const std::string& text) {
        const float region_width = ImGui::GetContentRegionAvail().x;
        const char* p = text.c_str();
        const char* end = p + text.size();
        while (p < end) {
            const char* nl = static_cast<const char*>(std::memchr(p, '\n', end - p));
            if (!nl)
                nl = end;
            if (nl == p) {
                ImGui::Spacing();
            } else {
                const float w = ImGui::CalcTextSize(p, nl).x;
                if (w <= region_width) {
                    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (region_width - w) * 0.5f);
                    ImGui::TextUnformatted(p, nl);
                } else {
                    ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + region_width);
                    ImGui::TextUnformatted(p, nl);
                    ImGui::PopTextWrapPos();
                }
            }
            p = nl + 1;
        }
    }

    static constexpr float BUTTON_WIDTH = 100.0f;
    static constexpr float POPUP_WIDTH = 440.0f;

    static constexpr ImGuiWindowFlags MODAL_FLAGS = ImGuiWindowFlags_NoCollapse |
                                                    ImGuiWindowFlags_NoDocking |
                                                    ImGuiWindowFlags_NoResize |
                                                    ImGuiWindowFlags_NoSavedSettings;

    PyModalRegistry& PyModalRegistry::instance() {
        static PyModalRegistry registry;
        return registry;
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

    void PyModalRegistry::draw_confirm_dialog(PyModalDialog& modal, const float scale) {
        const float btn_w = BUTTON_WIDTH * scale;

        draw_text_centered(modal.message);
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        std::string clicked_button;

        if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
            clicked_button = modal.buttons.back();
            modal.is_open = false;
        } else if (ImGui::IsKeyPressed(ImGuiKey_Enter)) {
            clicked_button = modal.buttons.front();
            modal.is_open = false;
        }

        const auto& im_style = ImGui::GetStyle();
        const float total_btns_width =
            btn_w * static_cast<float>(modal.buttons.size()) +
            im_style.ItemSpacing.x * static_cast<float>(modal.buttons.size() - 1);
        ImGui::SetCursorPosX((ImGui::GetContentRegionAvail().x - total_btns_width) * 0.5f +
                             im_style.WindowPadding.x);

        for (size_t i = 0; i < modal.buttons.size(); ++i) {
            if (i > 0)
                ImGui::SameLine();
            const auto btn_style = (i == 0) ? ButtonStyle::Primary : ButtonStyle::Secondary;
            if (ColoredButton(modal.buttons[i].c_str(), btn_style, ImVec2(btn_w, 0))) {
                clicked_button = modal.buttons[i];
                modal.is_open = false;
            }
        }

        if (!clicked_button.empty()) {
            if (modal.cpp_callback) {
                modal.cpp_callback(clicked_button);
            } else if (modal.callback.is_valid() && !modal.callback.is_none()) {
                nb::gil_scoped_acquire gil;
                try {
                    modal.callback(clicked_button);
                } catch (const std::exception& e) {
                    LOG_ERROR("Modal callback error: {}", e.what());
                }
            }
        }
    }

    void PyModalRegistry::draw_input_dialog(PyModalDialog& modal, const float scale) {
        const float btn_w = BUTTON_WIDTH * scale;

        draw_text_centered(modal.message);
        ImGui::Spacing();

        static char input_buf[1024];
        strncpy(input_buf, modal.input_value.c_str(), sizeof(input_buf) - 1);
        input_buf[sizeof(input_buf) - 1] = '\0';

        ImGui::SetNextItemWidth(-1);
        if (ImGui::InputText("##input", input_buf, sizeof(input_buf))) {
            modal.input_value = input_buf;
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        bool submitted = false;
        bool cancelled = false;

        if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
            cancelled = true;
            modal.is_open = false;
        } else if (ImGui::IsKeyPressed(ImGuiKey_Enter)) {
            submitted = true;
            modal.is_open = false;
        }

        const auto& im_style = ImGui::GetStyle();
        const float total_btns_width = btn_w * 2.0f + im_style.ItemSpacing.x;
        ImGui::SetCursorPosX((ImGui::GetContentRegionAvail().x - total_btns_width) * 0.5f +
                             im_style.WindowPadding.x);

        if (ColoredButton("OK", ButtonStyle::Primary, ImVec2(btn_w, 0))) {
            submitted = true;
            modal.is_open = false;
        }
        ImGui::SameLine();
        if (ColoredButton("Cancel", ButtonStyle::Secondary, ImVec2(btn_w, 0))) {
            cancelled = true;
            modal.is_open = false;
        }

        if (modal.callback.is_valid() && !modal.callback.is_none()) {
            nb::gil_scoped_acquire gil;
            try {
                if (submitted) {
                    modal.callback(nb::str(modal.input_value.c_str()));
                } else if (cancelled) {
                    modal.callback(nb::none());
                }
            } catch (const std::exception& e) {
                LOG_ERROR("Modal callback error: {}", e.what());
            }
        }
    }

    void PyModalRegistry::draw_message_dialog(PyModalDialog& modal, const float scale) {
        const float btn_w = BUTTON_WIDTH * scale;

        draw_text_centered(modal.message);
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        bool close = ImGui::IsKeyPressed(ImGuiKey_Escape) || ImGui::IsKeyPressed(ImGuiKey_Enter);

        ImGui::SetCursorPosX((ImGui::GetContentRegionAvail().x - btn_w) * 0.5f +
                             ImGui::GetStyle().WindowPadding.x);

        if (ColoredButton("OK", ButtonStyle::Primary, ImVec2(btn_w, 0)))
            close = true;

        if (close) {
            modal.is_open = false;
            if (modal.callback.is_valid() && !modal.callback.is_none()) {
                nb::gil_scoped_acquire gil;
                try {
                    modal.callback();
                } catch (const std::exception& e) {
                    LOG_ERROR("Modal callback error: {}", e.what());
                }
            }
        }
    }

    void PyModalRegistry::draw_modals() {
        std::lock_guard lock(mutex_);

        const auto& t = lfs::vis::theme();
        const float scale = get_shared_dpi_scale();

        for (auto it = modals_.begin(); it != modals_.end();) {
            auto& modal = *it;

            if (!modal.is_open) {
                it = modals_.erase(it);
                continue;
            }

            const ImVec4 border_color = [&]() -> ImVec4 {
                switch (modal.style) {
                case MessageStyle::Warning: return t.palette.warning;
                case MessageStyle::Error: return t.palette.error;
                default: return t.palette.success;
                }
            }();

            if (modal.needs_open) {
                ImGui::OpenPopup(modal.title.c_str());
                modal.needs_open = false;
            }

            ImGui::SetNextWindowSize(ImVec2(POPUP_WIDTH * scale, 0), ImGuiCond_Always);
            ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing,
                                    ImVec2(0.5f, 0.5f));

            t.pushModalStyle();
            ImGui::PushStyleColor(ImGuiCol_Border, border_color);

            if (ImGui::BeginPopupModal(modal.title.c_str(), nullptr, MODAL_FLAGS)) {
                switch (modal.type) {
                case ModalDialogType::Confirm:
                    draw_confirm_dialog(modal, scale);
                    break;
                case ModalDialogType::Input:
                    draw_input_dialog(modal, scale);
                    break;
                case ModalDialogType::Message:
                    draw_message_dialog(modal, scale);
                    break;
                }

                if (!modal.is_open)
                    ImGui::CloseCurrentPopup();

                ImGui::EndPopup();
            }

            ImGui::PopStyleColor();
            lfs::vis::Theme::popModalStyle();

            if (!modal.is_open) {
                it = modals_.erase(it);
            } else {
                ++it;
            }
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
