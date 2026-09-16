/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/panel_layout.hpp"
#include "gui/panel_registry.hpp"
#include "workspace/viewport_workspace.hpp"

#include <core/export.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace lfs::vis::gui {

    class RmlPanelHost;
    class RmlUIManager;

    enum class AreaEditorAction : uint8_t {
        SetEditor,
        Close,
        ToggleMaximize,
        ToggleProjection,
    };

    // Owns the retained UI for screen areas. Viewport areas only receive a
    // header; other editors receive a factory-created panel instance and are
    // rendered into the area's content rectangle.
    class LFS_VIS_API AreaEditorHost {
    public:
        AreaEditorHost(PanelRegistry& registry, ViewportWorkspace& workspace,
                       RmlUIManager* rml_manager);
        ~AreaEditorHost();

        AreaEditorHost(const AreaEditorHost&) = delete;
        AreaEditorHost& operator=(const AreaEditorHost&) = delete;

        // Synchronize area identity/editor changes before rendering. Areas
        // absent from the snapshot are retired and held until collectRetired().
        void synchronize(const WorkspaceFrameSnapshot& snapshot);

        // Render headers and editor content. pointer_owner and keyboard_owner
        // are the areas allowed to receive those input classes for this frame.
        // An omitted owner falls back to the area's focused bit.
        void render(const WorkspaceFrameSnapshot& snapshot,
                    const PanelDrawContext& base_context,
                    const PanelInputState& input,
                    int header_pixels,
                    std::optional<ViewId> pointer_owner = {},
                    std::optional<ViewId> keyboard_owner = {});

        // Refresh the visible editor instances after modal Python changes,
        // without recording GUI composition or delivering input.
        void preloadVisiblePanels(const WorkspaceFrameSnapshot& snapshot,
                                  const PanelDrawContext& context);

        // The caller invokes this after submitted frame work has retired.
        void collectRetired();

        // Retire all retained instances before replacing workspace state (for
        // example during project restore). Chrome is intentionally not
        // captured here: the restored records are the source of truth.
        void retireAll();

        // Capture chrome from live instances before project serialization.
        void captureChrome();

        [[nodiscard]] bool needsAnimationFrame() const;
        void releaseRendererResources();

        [[nodiscard]] std::size_t activeAreaCount() const noexcept { return areas_.size(); }
        [[nodiscard]] std::size_t retiredAreaCount() const noexcept;

    private:
        struct Header;
        struct Area;

        [[nodiscard]] static std::string instanceName(ViewId id, uint64_t generation,
                                                      std::string_view editor);
        [[nodiscard]] static std::string editorLabel(std::string_view editor);
        [[nodiscard]] std::string editorOptionsMarkup(std::string_view current_editor) const;
        [[nodiscard]] uint64_t editorOptionsRevision() const;
        void synchronizeArea(const AreaSnapshot& snapshot);
        void tryCreatePanel(Area& area, const AreaSnapshot& snapshot);
        void retireArea(Area&& area);
        void renderPanel(Area& area, const AreaSnapshot& snapshot,
                         const PanelDrawContext& context,
                         PanelDirectRenderMode mode, const PanelInputState* input);

    public:
        // Called by the retained header's event listener. Public so the
        // listener remains a small independent RmlUi object in the .cpp.
        void handleAction(ViewId id, AreaEditorAction action, std::string_view editor);

    private:
        [[nodiscard]] PanelInputState filteredInput(const PanelInputState& input,
                                                    const AreaSnapshot& area,
                                                    std::optional<ViewId> pointer_owner,
                                                    std::optional<ViewId> keyboard_owner) const;

        PanelRegistry& registry_;
        ViewportWorkspace& workspace_;
        RmlUIManager* rml_manager_ = nullptr;
        std::unordered_map<ViewId, Area> areas_;
        std::vector<Area> retired_;
        std::unordered_map<ViewId, uint64_t> next_editor_generation_;
    };

} // namespace lfs::vis::gui
