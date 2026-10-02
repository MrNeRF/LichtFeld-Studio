/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "screen/screen_layout.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <nlohmann/json_fwd.hpp>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

// Each area shows one editor type. The registry defines its label, menu icon,
// instance rules, default placement and per-area state.
namespace lfs::vis::screen {

    // An area retains each editor's state while other editors are shown.
    class SpaceData {
    public:
        virtual ~SpaceData() = default;
        [[nodiscard]] virtual std::unique_ptr<SpaceData> clone() const = 0;
        [[nodiscard]] virtual nlohmann::json save() const = 0;
        // Returns false and leaves the state unchanged on malformed input.
        virtual bool load(const nlohmann::json& json) = 0;
    };

    // Where an editor opens when it is shown without a target area.
    struct EditorPlacement {
        enum class Anchor : std::uint8_t {
            ScreenEdge, // a new full-height or full-width area at the screen edge
            ActiveView, // split the active 3D view on `side`
            Editor,     // split the area showing `editor`
        };
        Anchor anchor = Anchor::ScreenEdge;
        Side side = Side::Right;
        float fraction = 0.25f;
        std::string editor{};
        // Used for ScreenEdge, and when the anchor area does not exist.
        Side edge_side = Side::Right;
        float edge_fraction = 0.22f;
    };

    namespace editors {
        inline constexpr std::string_view kView3D = "view3d";
        inline constexpr std::string_view kScene = "scene";
        inline constexpr std::string_view kProperties = "properties";
        inline constexpr std::string_view kConsole = "console";
    } // namespace editors

    struct EditorType {
        std::string id;
        std::string label;     // untranslated fallback
        std::string label_key; // localization key, empty for plugin panels
        std::string icon;      // icon asset name without extension
        bool multi_instance = false;
        EditorPlacement placement;
        // Null for editors without per-area state.
        std::function<std::unique_ptr<SpaceData>()> create_space{};
    };

    class LFS_VIS_API EditorTypeRegistry {
    public:
        // Resolves ids that are not registered types, e.g. registered UI
        // panels that can be shown as an editor. Called on every lookup, so
        // it reflects panels that come and go at runtime.
        using DynamicProvider = std::function<std::optional<EditorType>(std::string_view id)>;
        using DynamicLister = std::function<std::vector<EditorType>()>;

        // Replaces a type with the same id.
        void add(EditorType type);
        void setDynamicProvider(DynamicProvider provider, DynamicLister lister);

        [[nodiscard]] std::optional<EditorType> find(std::string_view id) const;
        [[nodiscard]] bool contains(std::string_view id) const { return find(id).has_value(); }
        // Registered types in registration order, then dynamic ones.
        [[nodiscard]] std::vector<EditorType> list() const;

        [[nodiscard]] std::unique_ptr<SpaceData> createSpace(std::string_view id) const;

    private:
        std::vector<EditorType> types_;
        DynamicProvider dynamic_provider_;
        DynamicLister dynamic_lister_;
    };

} // namespace lfs::vis::screen
