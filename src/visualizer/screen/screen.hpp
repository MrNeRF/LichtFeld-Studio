/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "screen/editor_type.hpp"
#include "screen/screen_layout.hpp"

#include <map>
#include <memory>
#include <nlohmann/json_fwd.hpp>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace lfs::vis::screen {

    class View3DSpace;

    struct Area {
        AreaId id;
        std::string editor;
        // State of every editor this area has shown, by editor id.
        std::map<std::string, std::unique_ptr<SpaceData>, std::less<>> spaces;

        [[nodiscard]] SpaceData* space(std::string_view editor_id) const;
        [[nodiscard]] SpaceData* activeSpace() const { return space(editor); }
    };

    // The window's work area as a set of areas, each showing one editor.
    // Mutations are all-or-nothing and keep two rules: at least one area is
    // a 3D viewport, and an editor that is not multi-instance is shown by at
    // most one area.
    class Screen {
    public:
        explicit Screen(const EditorTypeRegistry& registry);
        Screen(const Screen& other);
        Screen& operator=(const Screen& other);
        Screen(Screen&&) noexcept = default;
        Screen& operator=(Screen&&) noexcept = default;

        // A 3D viewport with the scene tree above the properties on the right.
        [[nodiscard]] static Screen makeDefault(const EditorTypeRegistry& registry);

        [[nodiscard]] const EditorTypeRegistry& registry() const { return *registry_; }
        [[nodiscard]] const ScreenLayout& layout() const { return layout_; }
        [[nodiscard]] std::vector<AreaId> areas() const { return layout_.areas(); }
        [[nodiscard]] const Area* area(AreaId id) const;
        [[nodiscard]] Area* area(AreaId id);

        // First area showing `editor`, in layout order.
        [[nodiscard]] AreaId findEditor(std::string_view editor) const;
        // Areas showing a 3D viewport, in layout order.
        [[nodiscard]] std::vector<AreaId> views() const;
        [[nodiscard]] View3DSpace* view(AreaId id) const;

        // The 3D viewport the user last worked in. Tools and properties that
        // edit "the viewport" act on it. Always valid on a valid screen.
        [[nodiscard]] AreaId activeView() const { return active_view_; }
        bool setActiveView(AreaId id);

        [[nodiscard]] AreaId maximized() const { return maximized_; }
        // Maximizes `id`, or restores when it is already maximized or invalid.
        bool toggleMaximized(AreaId id);

        // Splits `id` and returns the new area, which shows the same editor
        // with a copy of its state. Single-instance editors cannot be shown
        // twice, so their new half shows a copy of the active 3D view.
        AreaId split(AreaId id, SplitAxis axis, float fraction, bool new_first = false);
        bool join(AreaId kept, AreaId absorbed);
        [[nodiscard]] bool canJoin(AreaId kept, AreaId absorbed) const;
        bool close(AreaId id);
        [[nodiscard]] bool canClose(AreaId id) const;
        bool swap(AreaId a, AreaId b);
        // Shows `editor` in `id`. A single-instance editor already shown
        // elsewhere trades places with this area instead of duplicating.
        bool setEditor(AreaId id, std::string_view editor);
        // Shows `editor` somewhere: an area already showing it (single
        // instance) or a new area at the editor's placement.
        AreaId openEditor(std::string_view editor);
        // Closes every area showing `editor`, keeping the last 3D viewport.
        bool closeEditor(std::string_view editor);
        bool moveDivider(const DividerGeometry& divider, float position);

        [[nodiscard]] LayoutGeometry solve(const Rect& bounds, const LayoutMetrics& metrics) const {
            return layout_.solve(bounds, metrics, maximized_);
        }

        // Bumped by every change of layout, editors, active view or
        // maximized state (not by edits inside a space).
        [[nodiscard]] std::uint64_t generation() const { return generation_; }

        [[nodiscard]] nlohmann::json save() const;
        [[nodiscard]] static std::optional<Screen> load(const nlohmann::json& json,
                                                        const EditorTypeRegistry& registry);

    private:
        [[nodiscard]] bool isView(AreaId id) const;
        [[nodiscard]] bool isLastView(AreaId id) const;
        [[nodiscard]] bool isMultiInstance(std::string_view editor) const;
        [[nodiscard]] std::unique_ptr<SpaceData> newSpace(std::string_view editor) const;
        AreaId addArea(std::string editor);
        void eraseArea(AreaId id);
        void repair();
        void touch() { ++generation_; }

        const EditorTypeRegistry* registry_;
        ScreenLayout layout_;
        std::map<AreaId, Area> areas_;
        AreaId active_view_;
        AreaId maximized_;
        std::uint32_t next_area_ = 1;
        std::uint64_t generation_ = 1;
    };

    // The editor types every build has: the 3D viewport, the scene tree, the
    // properties and the Python console.
    void registerBuiltinEditorTypes(EditorTypeRegistry& registry);

} // namespace lfs::vis::screen
