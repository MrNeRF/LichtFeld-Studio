/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace lfs::core {

    struct ModalButtonSpec {
        std::string label;
        std::string style; // "primary", "secondary", "error", "warning", "success"
        bool disabled = false;
    };

    enum class ModalStyle : uint8_t { Info,
                                      Warning,
                                      Error };

    struct ModalResult {
        std::string button_label;
        std::string input_value;
        std::unordered_map<std::string, std::string> form_values;
    };

    struct ModalRequest {
        std::string title;
        std::string body_rml;
        ModalStyle style = ModalStyle::Info;
        std::vector<ModalButtonSpec> buttons;
        bool has_input = false;
        std::string input_default;
        int width_dp = 440;
        std::function<void(const ModalResult&)> on_result;
        std::function<void()> on_cancel;
    };

} // namespace lfs::core
