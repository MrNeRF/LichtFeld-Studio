/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "tool_base.hpp"

namespace lfs::vis::tools {

    class ScaleDistanceTool : public ToolBase {
    public:
        ScaleDistanceTool();
        ~ScaleDistanceTool() override = default;

        std::string_view getName() const override { return "Distance Scale Tool"; }
        std::string_view getDescription() const override { return "Scale scene by measured 2-point distance"; }

        bool initialize(const ToolContext& ctx) override;
        void shutdown() override;
        void update(const ToolContext& ctx) override;
        void renderUI(const lfs::vis::gui::UIContext& ui_ctx, bool* p_open) override;

    protected:
        void onEnabledChanged(bool enabled) override;

    private:
        const ToolContext* tool_context_ = nullptr;
        bool had_status_ = false;
    };

} // namespace lfs::vis::tools
