/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "internal/viewport.hpp"
#include "rendering/rendering_types.hpp"
#include "screen/editor_type.hpp"

#include <string>

namespace lfs::vis::screen {

    // The state of one 3D viewport: where it looks from and how it draws the
    // scene. Blender splits this into View3D and RegionView3D; with one view
    // region per area they are one object here.
    class View3DSpace final : public SpaceData {
    public:
        View3DSpace() = default;

        Viewport camera;
        ViewSettings settings;
        // Set when an axis view switched the projection to orthographic, so
        // orbiting away returns to perspective (Blender's auto perspective).
        bool auto_orthographic = false;

        [[nodiscard]] std::unique_ptr<SpaceData> clone() const override;
        [[nodiscard]] nlohmann::json save() const override;
        bool load(const nlohmann::json& json) override;
    };

    // Which axis-aligned direction a view looks along, if any.
    enum class ViewAxis : std::uint8_t { None,
                                         Top,
                                         Bottom,
                                         Front,
                                         Back,
                                         Right,
                                         Left };

    [[nodiscard]] ViewAxis alignedViewAxis(const glm::mat3& rotation);

    // Blender-style view name, e.g. "Top Orthographic" or "User Perspective".
    [[nodiscard]] std::string viewLabel(const View3DSpace& view);

    // Switches projection, keeping the apparent size of what is at the pivot
    // when entering orthographic. `viewport_height` is in pixels.
    void setOrthographic(View3DSpace& view, bool enabled, float viewport_height);
    // Looks along `axis` at the pivot; like Blender's numpad views this also
    // switches to orthographic.
    void setAxisView(View3DSpace& view, ViewAxis axis, float viewport_height);
    // Leaves an axis view: back to perspective if the axis view chose ortho.
    void leaveAxisView(View3DSpace& view);

    [[nodiscard]] nlohmann::json viewSettingsToJson(const ViewSettings& settings);
    // Fields missing from `json` keep their value from `base`; present fields
    // must be valid or the whole read fails.
    [[nodiscard]] std::optional<ViewSettings> viewSettingsFromJson(const nlohmann::json& json,
                                                                   const ViewSettings& base);

} // namespace lfs::vis::screen
