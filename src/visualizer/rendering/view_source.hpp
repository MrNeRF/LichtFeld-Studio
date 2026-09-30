/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "rendering/rendering_types.hpp"

#include <cstdint>
#include <functional>
#include <optional>

namespace lfs::vis {

    // Identity of a 3D view: the id of the screen area showing it.
    using ViewId = std::uint32_t;
    inline constexpr ViewId kNoView = 0;

    // How the renderer reaches the per-view settings it does not own. The
    // screen owns them; the renderer owns only the scene-wide half.
    class ViewSource {
    public:
        virtual ~ViewSource() = default;

        // The 3D view the user last worked in; the one "the viewport" means
        // in panels, Python and MCP.
        [[nodiscard]] virtual ViewId activeView() const = 0;
        [[nodiscard]] virtual std::optional<ViewSettings> viewSettings(ViewId view) const = 0;
        // Returns false when the view does not exist.
        virtual bool editViewSettings(ViewId view, const std::function<void(ViewSettings&)>& edit) = 0;
    };

} // namespace lfs::vis
