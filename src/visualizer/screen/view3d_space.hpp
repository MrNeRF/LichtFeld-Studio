/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "internal/viewport.hpp"
#include "rendering/rendering_types.hpp"
#include "screen/editor_type.hpp"

#include <cstdint>
#include <string>
#include <string_view>

namespace lfs::vis::screen {

    // The state of one 3D viewport: its camera and display settings.
    class LFS_VIS_API View3DSpace final : public SpaceData {
    public:
        View3DSpace() = default;

        Viewport camera;
        ViewSettings settings;
        // Set when an axis view switches to orthographic, so orbiting away
        // returns to perspective.
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

    [[nodiscard]] LFS_VIS_API ViewAxis alignedViewAxis(const glm::mat3& rotation);

    // Human-readable view name, e.g. "Top Orthographic" or "User Perspective".
    [[nodiscard]] LFS_VIS_API std::string viewLabel(const View3DSpace& view);

    // Switches projection, keeping the apparent size of what is at the pivot
    // when entering orthographic. `viewport_height` is in pixels.
    void setOrthographic(View3DSpace& view, bool enabled, float viewport_height);
    // Looks along `axis` at the pivot and switches to orthographic.
    void setAxisView(View3DSpace& view, ViewAxis axis, float viewport_height);
    // Leaves an axis view: back to perspective if the axis view chose ortho.

    // Applies a view command that only touches the view itself:
    //   display:splats|points|rings|centers, depth, projection,
    //   axis:top|bottom|front|back|right|left, overlay:grid|axes|pivot|frustums,
    //   overlay:grid_plane:<0..2>.
    // Returns false for commands it does not know.
    bool applyViewCommand(View3DSpace& view, std::string_view command, float viewport_height);

    [[nodiscard]] LFS_VIS_API nlohmann::json viewSettingsToJson(const ViewSettings& settings);
    // Fields missing from `json` keep their value from `base`; present fields
    // must be valid or the whole read fails.
    [[nodiscard]] LFS_VIS_API std::optional<ViewSettings> viewSettingsFromJson(const nlohmann::json& json,
                                                                               const ViewSettings& base);

} // namespace lfs::vis::screen
