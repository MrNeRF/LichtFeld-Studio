/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/panel_registry.hpp"

#include <cstdint>
#include <nanobind/nanobind.h>
#include <string>

namespace nb = nanobind;

namespace lfs::vis::gui {

    class RmlPythonPanelAdapter : public IPanel {
    public:
        RmlPythonPanelAdapter(void* manager, nb::object panel_instance,
                              const std::string& context_name, const std::string& rml_path);
        ~RmlPythonPanelAdapter() override;

        void draw(const PanelDrawContext& ctx) override;

    private:
        void* host_ = nullptr;
        void* manager_;
        std::string context_name_;
        std::string rml_path_;
        nb::object panel_instance_;
        bool loaded_ = false;
        uint64_t last_scene_gen_ = 0;
    };

} // namespace lfs::vis::gui
