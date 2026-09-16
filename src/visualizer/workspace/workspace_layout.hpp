/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/error.hpp"
#include "core/export.hpp"
#include "workspace/view_id.hpp"
#include "workspace/view_types.hpp"

#include <cstddef>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

namespace lfs::vis {

    struct LayoutPane {
        ViewId id = kInvalidViewId;
        LayoutNodeId leaf = kInvalidLayoutNodeId;
        ViewRect rect{};
    };

    struct SplitterRect {
        LayoutNodeId split = kInvalidLayoutNodeId;
        SplitAxis axis = SplitAxis::Horizontal;
        ViewRect rect{};
        // Bounds of the layout node containing this splitter. Input uses the
        // matching axis to normalize pointer movement against the split's
        // local extent, including nested splits.
        ViewRect parent{};
    };

    struct WorkspaceLayoutSnapshot {
        LayoutNodeId root = kInvalidLayoutNodeId;
        std::vector<LayoutPane> panes;       // tree order; visible panes only
        std::vector<SplitterRect> splitters; // tree order; positive-area only
        std::optional<ViewId> focused;
        std::optional<ViewId> maximized;
        std::uint64_t generation = 0;
        ViewRect outer{};
    };

    // Plain serializable tree node. Children are IDs, never pointers.
    struct LayoutNodeState {
        LayoutNodeId id = kInvalidLayoutNodeId;
        LayoutNodeKind kind = LayoutNodeKind::Leaf;
        ViewId view_id = kInvalidViewId;
        SplitAxis axis = SplitAxis::Horizontal;
        float ratio = 0.5f;
        LayoutNodeId first = kInvalidLayoutNodeId;
        LayoutNodeId second = kInvalidLayoutNodeId;
    };

    struct WorkspaceLayoutState {
        LayoutNodeId root = kInvalidLayoutNodeId;
        std::vector<LayoutNodeState> nodes;
        std::optional<ViewId> focused;
        std::optional<ViewId> maximized;
        LayoutNodeId next_node_id = 1;
        std::uint64_t generation = 0;
    };

    // Binary split tree of leaves (ViewId) and splitters. Owns no cameras.
    class LFS_VIS_API WorkspaceLayout {
    public:
        WorkspaceLayout() = default;

        [[nodiscard]] static WorkspaceLayout single(ViewId view);

        [[nodiscard]] bool empty() const noexcept { return root_ == kInvalidLayoutNodeId; }
        [[nodiscard]] bool contains(ViewId id) const noexcept;
        [[nodiscard]] bool containsNode(LayoutNodeId id) const noexcept;
        [[nodiscard]] std::size_t leafCount() const noexcept;
        [[nodiscard]] LayoutNodeId root() const noexcept { return root_; }
        [[nodiscard]] LayoutNodeId nextNodeId() const noexcept { return next_node_id_; }
        [[nodiscard]] std::uint64_t generation() const noexcept { return generation_; }
        [[nodiscard]] std::optional<ViewId> focused() const noexcept { return focused_; }
        [[nodiscard]] std::optional<ViewId> maximized() const noexcept { return maximized_; }

        [[nodiscard]] std::vector<ViewId> leafIds() const;
        [[nodiscard]] std::optional<float> splitRatio(LayoutNodeId split) const;
        [[nodiscard]] std::optional<SplitAxis> splitAxis(LayoutNodeId split) const;

        [[nodiscard]] WorkspaceLayoutSnapshot solve(
            ViewRect outer,
            int min_pane_pixels = kDefaultMinPanePixels,
            int divider_pixels = kDefaultDividerPixels) const;

        // Caller supplies the already-allocated registry ID. These methods only
        // mutate tree membership; they never allocate views.
        [[nodiscard]] bool split(ViewId source, ViewId new_view, SplitAxis axis,
                                 float ratio = 0.5f);
        [[nodiscard]] bool close(ViewId id);
        [[nodiscard]] bool focus(ViewId id);
        [[nodiscard]] bool maximize(ViewId id);
        [[nodiscard]] bool restoreMaximized();
        [[nodiscard]] bool resize(LayoutNodeId split, float ratio);

        // Existing IDs to keep for a preset, in tree order, with focused
        // surviving when the preset shrinks the leaf count.
        [[nodiscard]] std::vector<ViewId> requiredPresetIds(LayoutPreset preset) const;
        [[nodiscard]] bool matchesPreset(LayoutPreset preset) const noexcept;
        [[nodiscard]] bool applyPreset(LayoutPreset preset, std::span<const ViewId> ids);

        [[nodiscard]] lfs::Status validate() const;
        [[nodiscard]] WorkspaceLayoutState exportState() const;
        [[nodiscard]] static lfs::Result<WorkspaceLayout> fromState(const WorkspaceLayoutState& state);

        // Raises the monotonic node counter. Never decreases it.
        void bumpNextNodeId(LayoutNodeId at_least) noexcept;

        // Keeps frame generations monotonic when importing persisted state.
        void bumpGenerationAtLeast(std::uint64_t at_least) noexcept;

    private:
        struct Node {
            LayoutNodeId id = kInvalidLayoutNodeId;
            LayoutNodeKind kind = LayoutNodeKind::Leaf;
            ViewId view = kInvalidViewId;
            SplitAxis axis = SplitAxis::Horizontal;
            float ratio = 0.5f;
            LayoutNodeId first = kInvalidLayoutNodeId;
            LayoutNodeId second = kInvalidLayoutNodeId;
        };

        [[nodiscard]] LayoutNodeId allocateNode();
        void reindex();
        void bumpGeneration();
        void collectLeaves(LayoutNodeId id, std::vector<ViewId>& out,
                           std::unordered_map<LayoutNodeId, int>& visiting) const;
        void solveNode(LayoutNodeId id, ViewRect rect, int min_pane, int divider,
                       WorkspaceLayoutSnapshot& out,
                       std::unordered_map<LayoutNodeId, int>& visiting) const;
        [[nodiscard]] const Node* node(LayoutNodeId id) const;
        Node* node(LayoutNodeId id);

        LayoutNodeId root_ = kInvalidLayoutNodeId;
        LayoutNodeId next_node_id_ = 1;
        bool node_id_space_exhausted_ = false;
        std::uint64_t generation_ = 0;
        std::optional<ViewId> focused_;
        std::optional<ViewId> maximized_;
        std::unordered_map<LayoutNodeId, Node> nodes_;
        std::unordered_map<ViewId, LayoutNodeId> leaf_by_view_;
        std::unordered_map<LayoutNodeId, LayoutNodeId> parent_;
    };

} // namespace lfs::vis
