/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/error.hpp"
#include "core/export.hpp"
#include "internal/viewport.hpp"
#include "rendering/rendering_types.hpp"
#include "workspace/viewport_workspace.hpp"

#include <nlohmann/json.hpp>

namespace lfs::vis {

    using WorkspaceJson = nlohmann::json;

    // Workspace state is an additive VIEW payload. The caller owns retention of
    // unknown VIEW fields; these functions only encode/decode the workspace
    // object itself.
    [[nodiscard]] LFS_VIS_API lfs::Result<WorkspaceJson>
    workspaceStateToJson(const WorkspacePersistentState& state);

    [[nodiscard]] LFS_VIS_API lfs::Result<WorkspacePersistentState>
    workspaceStateFromJson(const WorkspaceJson& json);

    // Import the legacy panel_cameras pose into a single-pane workspace.
    // Preserve the ID high-water mark and commit only a valid state.
    [[nodiscard]] LFS_VIS_API lfs::Status
    initializeWorkspacePrimaryFromLegacy(ViewportWorkspace& workspace,
                                         const Viewport& legacy_viewport,
                                         const RenderSettings& settings);

} // namespace lfs::vis
