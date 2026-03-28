/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "render_pass.hpp"

namespace lfs::vis {

    struct SplitViewPanelPlan {
        std::string label;
        lfs::rendering::SplitViewPanel panel;
    };

    struct SplitViewCompositionPlan {
        std::string mode_label;
        std::string detail_label;
        std::array<SplitViewPanelPlan, 2> panels;
        lfs::rendering::SplitViewCompositeState composite;
        lfs::rendering::SplitViewPresentationState presentation;
        bool prefer_batched_gaussian_render = false;

        [[nodiscard]] lfs::rendering::SplitViewRequest toRequest() const;
        [[nodiscard]] SplitViewInfo toInfo() const;
    };

    [[nodiscard]] std::optional<SplitViewCompositionPlan> buildSplitViewCompositionPlan(
        const FrameContext& ctx,
        const FrameResources& res);

} // namespace lfs::vis
