/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/rmlui/rmlui_manager.hpp"
#include "workspace/corner_split_interaction.hpp"

#include <cstddef>
#include <optional>
#include <string>

namespace Rml {
    class Context;
    class ElementDocument;
} // namespace Rml

namespace lfs::vis::gui {

    // A single pointer-transparent Rml context for workspace decorations. It
    // deliberately renders after area content so grips and split previews are
    // visible over native and RML editors alike.
    class RmlWorkspaceChrome {
    public:
        void init(RmlUIManager* manager);
        void shutdown();
        void reloadResources();
        void render(const WorkspaceFrameSnapshot& snapshot,
                    std::optional<CornerSplitPreview> preview,
                    bool ui_hidden);

    private:
        [[nodiscard]] std::string makeMarkup(const WorkspaceFrameSnapshot& snapshot,
                                             const std::optional<CornerSplitPreview>& preview) const;
        bool updateTheme();

        RmlUIManager* rml_manager_ = nullptr;
        Rml::Context* rml_context_ = nullptr;
        Rml::ElementDocument* document_ = nullptr;
        std::size_t last_theme_signature_ = 0;
        bool has_theme_signature_ = false;
        std::string last_markup_;
        CachedVulkanContextRender direct_cache_;
    };

} // namespace lfs::vis::gui
