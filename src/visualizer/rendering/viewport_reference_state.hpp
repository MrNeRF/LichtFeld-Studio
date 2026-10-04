/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/export.hpp"
#include <cstdint>
#include <optional>
namespace lfs::vis {
    struct ViewRenderState;
    // Opaque lifetime storage for the temporary reference path, not a backend
    // resource handle or an extensible compositor implementation contract.
    struct ViewportReferenceState;
    [[nodiscard]] LFS_VIS_API std::optional<std::uint64_t>
    referenceSceneOutputGeneration(const ViewRenderState&);
    LFS_VIS_API void clearViewportReferenceOutput(ViewRenderState&);
}
