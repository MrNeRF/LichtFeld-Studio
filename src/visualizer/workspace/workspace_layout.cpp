/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "workspace/workspace_layout.hpp"

#include <algorithm>
#include <cmath>
#include <span>
#include <string>
#include <unordered_set>
#include <utility>

namespace lfs::vis {
    namespace {

        lfs::Error layoutError(const lfs::ErrorCode code, std::string message,
                               const lfs::core::SourceSite site) {
            return lfs::make_error(lfs::ErrorInit{
                .code = code,
                .domain = lfs::ErrorDomain::App,
                .severity = lfs::Severity::Error,
                .retryability = lfs::Retryability::NotRetryable,
                .user_message = message,
                .detail = message,
                .detection = site,
            });
        }

        [[nodiscard]] bool finiteRatio(const float ratio) noexcept {
            return std::isfinite(ratio);
        }

        [[nodiscard]] std::optional<float> clampRatio(const float ratio) {
            // Require an interior ratio; partitionRect enforces the physical pane minimum.
            if (!finiteRatio(ratio) || ratio <= 0.0f || ratio >= 1.0f)
                return std::nullopt;
            return ratio;
        }

        [[nodiscard]] ViewRect sanitized(ViewRect rect) noexcept {
            if (rect.width < 0)
                rect.width = 0;
            if (rect.height < 0)
                rect.height = 0;
            return rect;
        }

        struct Partition {
            ViewRect first{};
            ViewRect splitter{};
            ViewRect second{};
        };

        [[nodiscard]] Partition partitionRect(const ViewRect rect, const SplitAxis axis,
                                              const float ratio, const int min_pane,
                                              const int divider_px) {
            Partition out;
            const float r = finiteRatio(ratio) ? std::clamp(ratio, 0.0f, 1.0f) : 0.5f;
            const int min_px = std::max(min_pane, 0);
            int divider = std::max(divider_px, 0);

            if (axis == SplitAxis::Horizontal) {
                const int w = std::max(rect.width, 0);
                const int h = std::max(rect.height, 0);
                if (divider >= w)
                    divider = 0;
                const int avail = w - divider;
                int first = static_cast<int>(std::lround(static_cast<double>(avail) * static_cast<double>(r)));
                first = std::clamp(first, 0, avail);
                if (min_px > 0 && avail >= min_px * 2)
                    first = std::clamp(first, min_px, avail - min_px);
                const int second = avail - first;
                out.first = {rect.x, rect.y, first, h};
                out.splitter = {rect.x + first, rect.y, divider, h};
                out.second = {rect.x + first + divider, rect.y, second, h};
                return out;
            }

            const int w = std::max(rect.width, 0);
            const int h = std::max(rect.height, 0);
            if (divider >= h)
                divider = 0;
            const int avail = h - divider;
            int first = static_cast<int>(std::lround(static_cast<double>(avail) * static_cast<double>(r)));
            first = std::clamp(first, 0, avail);
            if (min_px > 0 && avail >= min_px * 2)
                first = std::clamp(first, min_px, avail - min_px);
            const int second = avail - first;
            out.first = {rect.x, rect.y, w, first};
            out.splitter = {rect.x, rect.y + first, w, divider};
            out.second = {rect.x, rect.y + first + divider, w, second};
            return out;
        }

        [[nodiscard]] bool uniqueNonZeroIds(std::span<const ViewId> ids) {
            std::unordered_set<ViewId> seen;
            seen.reserve(ids.size());
            for (const ViewId id : ids) {
                if (!isValidViewId(id) || !seen.insert(id).second)
                    return false;
            }
            return true;
        }

    } // namespace

    WorkspaceLayout WorkspaceLayout::single(const ViewId view) {
        WorkspaceLayout layout;
        if (!isValidViewId(view))
            return layout;
        Node leaf;
        leaf.id = layout.allocateNode();
        if (!isValidLayoutNodeId(leaf.id))
            return layout;
        leaf.kind = LayoutNodeKind::Leaf;
        leaf.view = view;
        layout.nodes_.emplace(leaf.id, leaf);
        layout.root_ = leaf.id;
        layout.focused_ = view;
        layout.maximized_.reset();
        layout.reindex();
        layout.bumpGeneration();
        return layout;
    }

    bool WorkspaceLayout::contains(const ViewId id) const noexcept {
        return isValidViewId(id) && leaf_by_view_.contains(id);
    }

    bool WorkspaceLayout::containsNode(const LayoutNodeId id) const noexcept {
        return isValidLayoutNodeId(id) && nodes_.contains(id);
    }

    std::size_t WorkspaceLayout::leafCount() const noexcept {
        return leaf_by_view_.size();
    }

    std::vector<ViewId> WorkspaceLayout::leafIds() const {
        std::vector<ViewId> ids;
        ids.reserve(leaf_by_view_.size());
        std::unordered_map<LayoutNodeId, int> visiting;
        collectLeaves(root_, ids, visiting);
        return ids;
    }

    std::optional<float> WorkspaceLayout::splitRatio(const LayoutNodeId split) const {
        const Node* n = node(split);
        if (n == nullptr || n->kind != LayoutNodeKind::Split)
            return std::nullopt;
        return n->ratio;
    }

    std::optional<SplitAxis> WorkspaceLayout::splitAxis(const LayoutNodeId split) const {
        const Node* n = node(split);
        if (n == nullptr || n->kind != LayoutNodeKind::Split)
            return std::nullopt;
        return n->axis;
    }

    WorkspaceLayoutSnapshot WorkspaceLayout::solve(ViewRect outer, const int min_pane_pixels,
                                                   const int divider_pixels) const {
        WorkspaceLayoutSnapshot snap;
        snap.root = root_;
        snap.focused = focused_;
        snap.maximized = maximized_;
        snap.generation = generation_;
        snap.outer = sanitized(outer);
        snap.panes.reserve(leaf_by_view_.size());
        snap.splitters.reserve(nodes_.size());

        if (root_ == kInvalidLayoutNodeId || nodes_.empty()) {
            return snap;
        }

        if (maximized_ && contains(*maximized_)) {
            LayoutPane pane;
            pane.id = *maximized_;
            pane.leaf = leaf_by_view_.at(*maximized_);
            pane.rect = snap.outer;
            snap.panes.push_back(pane);
            return snap;
        }

        std::unordered_map<LayoutNodeId, int> visiting;
        solveNode(root_, snap.outer, min_pane_pixels, divider_pixels, snap, visiting);
        return snap;
    }

    bool WorkspaceLayout::split(const ViewId source, const ViewId new_view, const SplitAxis axis,
                                const float ratio) {
        const auto clamped = clampRatio(ratio);
        if (!clamped)
            return false;
        if (!contains(source) || !isValidViewId(new_view) || contains(new_view) || source == new_view)
            return false;

        const LayoutNodeId leaf_id = leaf_by_view_[source];
        Node split_node;
        split_node.id = allocateNode();
        if (!isValidLayoutNodeId(split_node.id))
            return false;
        split_node.kind = LayoutNodeKind::Split;
        split_node.axis = axis;
        split_node.ratio = *clamped;
        split_node.first = leaf_id;

        Node new_leaf;
        new_leaf.id = allocateNode();
        if (!isValidLayoutNodeId(new_leaf.id))
            return false;
        new_leaf.kind = LayoutNodeKind::Leaf;
        new_leaf.view = new_view;
        split_node.second = new_leaf.id;

        const auto parent_it = parent_.find(leaf_id);
        if (parent_it == parent_.end()) {
            root_ = split_node.id;
        } else {
            Node* parent = node(parent_it->second);
            if (parent == nullptr)
                return false;
            if (parent->first == leaf_id)
                parent->first = split_node.id;
            else if (parent->second == leaf_id)
                parent->second = split_node.id;
            else
                return false;
        }

        nodes_.emplace(split_node.id, split_node);
        nodes_.emplace(new_leaf.id, new_leaf);
        reindex();
        if (focused_ && !contains(*focused_))
            focused_ = source;
        if (maximized_ && !contains(*maximized_))
            maximized_.reset();
        maximized_.reset();
        bumpGeneration();
        return true;
    }

    bool WorkspaceLayout::close(const ViewId id) {
        if (!contains(id) || leafCount() <= 1)
            return false;

        const LayoutNodeId leaf_id = leaf_by_view_[id];
        const auto parent_it = parent_.find(leaf_id);
        if (parent_it == parent_.end())
            return false;

        const LayoutNodeId parent_id = parent_it->second;
        Node* parent = node(parent_id);
        if (parent == nullptr || parent->kind != LayoutNodeKind::Split)
            return false;

        const LayoutNodeId sibling_id = parent->first == leaf_id ? parent->second : parent->first;
        if (!containsNode(sibling_id))
            return false;

        const auto grand_it = parent_.find(parent_id);
        if (grand_it == parent_.end()) {
            root_ = sibling_id;
        } else {
            Node* grand = node(grand_it->second);
            if (grand == nullptr)
                return false;
            if (grand->first == parent_id)
                grand->first = sibling_id;
            else if (grand->second == parent_id)
                grand->second = sibling_id;
            else
                return false;
        }

        nodes_.erase(leaf_id);
        nodes_.erase(parent_id);
        reindex();

        if (focused_ && *focused_ == id) {
            std::vector<ViewId> remaining;
            std::unordered_map<LayoutNodeId, int> visiting;
            collectLeaves(sibling_id, remaining, visiting);
            focused_ = remaining.empty() ? std::optional<ViewId>{} : remaining.front();
        } else if (focused_ && !contains(*focused_)) {
            const auto ids = leafIds();
            focused_ = ids.empty() ? std::optional<ViewId>{} : ids.front();
        }
        if (maximized_ && (*maximized_ == id || !contains(*maximized_)))
            maximized_.reset();

        bumpGeneration();
        return true;
    }

    bool WorkspaceLayout::focus(const ViewId id) {
        if (!contains(id))
            return false;
        if (focused_ == id)
            return true;
        focused_ = id;
        bumpGeneration();
        return true;
    }

    bool WorkspaceLayout::maximize(const ViewId id) {
        if (!contains(id))
            return false;
        if (maximized_ == id)
            return true;
        maximized_ = id;
        bumpGeneration();
        return true;
    }

    bool WorkspaceLayout::restoreMaximized() {
        if (!maximized_)
            return true;
        maximized_.reset();
        bumpGeneration();
        return true;
    }

    bool WorkspaceLayout::resize(const LayoutNodeId split, const float ratio) {
        const auto clamped = clampRatio(ratio);
        if (!clamped)
            return false;
        Node* n = node(split);
        if (n == nullptr || n->kind != LayoutNodeKind::Split)
            return false;
        if (n->ratio == *clamped)
            return true;
        n->ratio = *clamped;
        bumpGeneration();
        return true;
    }

    std::vector<ViewId> WorkspaceLayout::requiredPresetIds(const LayoutPreset preset) const {
        const std::size_t n = presetLeafCount(preset);
        auto ids = leafIds();
        if (ids.size() <= n)
            return ids;

        const bool focused_in_prefix =
            focused_ && std::find(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(n),
                                  *focused_) != ids.begin() + static_cast<std::ptrdiff_t>(n);
        if (focused_ && contains(*focused_) && !focused_in_prefix) {
            ids.resize(n - 1);
            ids.push_back(*focused_);
            return ids;
        }
        ids.resize(n);
        return ids;
    }

    bool WorkspaceLayout::matchesPreset(const LayoutPreset preset) const noexcept {
        const Node* root = node(root_);
        if (root == nullptr)
            return false;

        const auto isLeaf = [this](const LayoutNodeId id) {
            const Node* child = node(id);
            return child != nullptr && child->kind == LayoutNodeKind::Leaf;
        };
        const auto isSplit = [this](const LayoutNodeId id, const SplitAxis axis) {
            const Node* child = node(id);
            return child != nullptr && child->kind == LayoutNodeKind::Split &&
                   child->axis == axis;
        };

        switch (preset) {
        case LayoutPreset::Single:
            return root->kind == LayoutNodeKind::Leaf && leafCount() == 1;
        case LayoutPreset::DualHorizontal:
            return root->kind == LayoutNodeKind::Split &&
                   root->axis == SplitAxis::Horizontal && leafCount() == 2 &&
                   isLeaf(root->first) && isLeaf(root->second);
        case LayoutPreset::DualVertical:
            return root->kind == LayoutNodeKind::Split &&
                   root->axis == SplitAxis::Vertical && leafCount() == 2 &&
                   isLeaf(root->first) && isLeaf(root->second);
        case LayoutPreset::Quad: {
            if (root->kind != LayoutNodeKind::Split || root->axis != SplitAxis::Vertical ||
                leafCount() != 4 || !isSplit(root->first, SplitAxis::Horizontal) ||
                !isSplit(root->second, SplitAxis::Horizontal))
                return false;
            const Node* first = node(root->first);
            const Node* second = node(root->second);
            return isLeaf(first->first) && isLeaf(first->second) &&
                   isLeaf(second->first) && isLeaf(second->second);
        }
        }
        return false;
    }

    bool WorkspaceLayout::applyPreset(const LayoutPreset preset, const std::span<const ViewId> ids) {
        const std::size_t n = presetLeafCount(preset);
        if (ids.size() != n || !uniqueNonZeroIds(ids))
            return false;

        WorkspaceLayout built;
        built.next_node_id_ = next_node_id_;
        built.node_id_space_exhausted_ = node_id_space_exhausted_;
        built.generation_ = generation_ + 1;

        auto make_leaf = [&built](const ViewId view) {
            Node leaf;
            leaf.id = built.allocateNode();
            leaf.kind = LayoutNodeKind::Leaf;
            leaf.view = view;
            built.nodes_.emplace(leaf.id, leaf);
            return leaf.id;
        };
        auto make_split = [&built](const SplitAxis axis, const LayoutNodeId first,
                                   const LayoutNodeId second) {
            Node split;
            split.id = built.allocateNode();
            split.kind = LayoutNodeKind::Split;
            split.axis = axis;
            split.ratio = 0.5f;
            split.first = first;
            split.second = second;
            built.nodes_.emplace(split.id, split);
            return split.id;
        };

        switch (preset) {
        case LayoutPreset::Single:
            built.root_ = make_leaf(ids[0]);
            break;
        case LayoutPreset::DualHorizontal:
            built.root_ = make_split(SplitAxis::Horizontal, make_leaf(ids[0]), make_leaf(ids[1]));
            break;
        case LayoutPreset::DualVertical:
            built.root_ = make_split(SplitAxis::Vertical, make_leaf(ids[0]), make_leaf(ids[1]));
            break;
        case LayoutPreset::Quad: {
            const LayoutNodeId top =
                make_split(SplitAxis::Horizontal, make_leaf(ids[0]), make_leaf(ids[1]));
            const LayoutNodeId bottom =
                make_split(SplitAxis::Horizontal, make_leaf(ids[2]), make_leaf(ids[3]));
            built.root_ = make_split(SplitAxis::Vertical, top, bottom);
            break;
        }
        }

        if (focused_ && std::find(ids.begin(), ids.end(), *focused_) != ids.end())
            built.focused_ = focused_;
        else
            built.focused_ = ids.front();
        built.maximized_.reset();
        if (!isValidLayoutNodeId(built.root_))
            return false;
        built.reindex();
        if (!built.validate())
            return false;
        *this = std::move(built);
        return true;
    }

    lfs::Status WorkspaceLayout::validate() const {
        if (root_ == kInvalidLayoutNodeId) {
            if (!nodes_.empty() || !leaf_by_view_.empty()) {
                return lfs::Status::failure(layoutError(
                    lfs::ErrorCode::ContractViolation, "Layout has nodes without a root",
                    LFS_SOURCE_SITE_CURRENT()));
            }
            return {};
        }
        if (!nodes_.contains(root_)) {
            return lfs::Status::failure(layoutError(
                lfs::ErrorCode::ContractViolation, "Layout root is missing from the node map",
                LFS_SOURCE_SITE_CURRENT()));
        }

        std::unordered_set<LayoutNodeId> reachable;
        std::unordered_map<LayoutNodeId, int> visiting;
        std::unordered_map<LayoutNodeId, int> incoming;
        auto walk = [&](auto& self, const LayoutNodeId id) -> lfs::Status {
            if (!isValidLayoutNodeId(id) || !nodes_.contains(id)) {
                return lfs::Status::failure(layoutError(
                    lfs::ErrorCode::ContractViolation, "Layout references a missing node",
                    LFS_SOURCE_SITE_CURRENT()));
            }
            int& state = visiting[id];
            if (state == 1) {
                return lfs::Status::failure(layoutError(
                    lfs::ErrorCode::ContractViolation, "Layout tree contains a cycle",
                    LFS_SOURCE_SITE_CURRENT()));
            }
            if (state == 2)
                return {};
            state = 1;
            const Node& n = nodes_.at(id);
            if (n.id != id) {
                return lfs::Status::failure(layoutError(
                    lfs::ErrorCode::ContractViolation, "Layout node id does not match map key",
                    LFS_SOURCE_SITE_CURRENT()));
            }
            if (n.kind == LayoutNodeKind::Leaf) {
                if (!isValidViewId(n.view) || n.first != kInvalidLayoutNodeId ||
                    n.second != kInvalidLayoutNodeId) {
                    return lfs::Status::failure(layoutError(
                        lfs::ErrorCode::ContractViolation, "Layout leaf is malformed",
                        LFS_SOURCE_SITE_CURRENT()));
                }
            } else if (n.kind == LayoutNodeKind::Split) {
                if (n.axis != SplitAxis::Horizontal && n.axis != SplitAxis::Vertical) {
                    return lfs::Status::failure(layoutError(
                        lfs::ErrorCode::ContractViolation, "Layout split axis is unknown",
                        LFS_SOURCE_SITE_CURRENT()));
                }
                if (!finiteRatio(n.ratio) || n.ratio <= 0.0f || n.ratio >= 1.0f) {
                    return lfs::Status::failure(layoutError(
                        lfs::ErrorCode::InvalidArgument,
                        "Layout split ratio must be finite and between zero and one",
                        LFS_SOURCE_SITE_CURRENT()));
                }
                if (n.first == kInvalidLayoutNodeId || n.second == kInvalidLayoutNodeId ||
                    n.first == n.second || n.first == id || n.second == id) {
                    return lfs::Status::failure(layoutError(
                        lfs::ErrorCode::ContractViolation, "Layout split children are invalid",
                        LFS_SOURCE_SITE_CURRENT()));
                }
                incoming[n.first]++;
                incoming[n.second]++;
                if (auto status = self(self, n.first); !status)
                    return status;
                if (auto status = self(self, n.second); !status)
                    return status;
            } else {
                return lfs::Status::failure(layoutError(
                    lfs::ErrorCode::ContractViolation, "Layout node has an unknown kind",
                    LFS_SOURCE_SITE_CURRENT()));
            }
            state = 2;
            reachable.insert(id);
            return {};
        };
        if (auto status = walk(walk, root_); !status)
            return status;
        if (incoming[root_] != 0) {
            return lfs::Status::failure(layoutError(
                lfs::ErrorCode::ContractViolation, "Layout root has a parent",
                LFS_SOURCE_SITE_CURRENT()));
        }
        for (const auto& [id, _] : nodes_) {
            if (id == root_)
                continue;
            if (incoming[id] != 1) {
                return lfs::Status::failure(layoutError(
                    lfs::ErrorCode::ContractViolation,
                    "Layout node does not have exactly one parent",
                    LFS_SOURCE_SITE_CURRENT()));
            }
        }

        if (reachable.size() != nodes_.size()) {
            return lfs::Status::failure(layoutError(
                lfs::ErrorCode::ContractViolation, "Layout contains unreachable nodes",
                LFS_SOURCE_SITE_CURRENT()));
        }

        std::unordered_set<ViewId> views;
        views.reserve(leaf_by_view_.size());
        for (const auto& [id, n] : nodes_) {
            if (n.kind != LayoutNodeKind::Leaf)
                continue;
            if (!views.insert(n.view).second) {
                return lfs::Status::failure(layoutError(
                    lfs::ErrorCode::AlreadyExists, "Layout assigns the same view to two leaves",
                    LFS_SOURCE_SITE_CURRENT()));
            }
            const auto leaf_it = leaf_by_view_.find(n.view);
            if (leaf_it == leaf_by_view_.end() || leaf_it->second != id) {
                return lfs::Status::failure(layoutError(
                    lfs::ErrorCode::ContractViolation, "Layout leaf index is inconsistent",
                    LFS_SOURCE_SITE_CURRENT()));
            }
        }
        if (views.size() != leaf_by_view_.size()) {
            return lfs::Status::failure(layoutError(
                lfs::ErrorCode::ContractViolation, "Layout leaf index has extra entries",
                LFS_SOURCE_SITE_CURRENT()));
        }
        if (focused_ && !contains(*focused_)) {
            return lfs::Status::failure(layoutError(
                lfs::ErrorCode::NotFound, "Focused view is not a live leaf",
                LFS_SOURCE_SITE_CURRENT()));
        }
        if (maximized_ && !contains(*maximized_)) {
            return lfs::Status::failure(layoutError(
                lfs::ErrorCode::NotFound, "Maximized view is not a live leaf",
                LFS_SOURCE_SITE_CURRENT()));
        }
        return {};
    }

    WorkspaceLayoutState WorkspaceLayout::exportState() const {
        WorkspaceLayoutState state;
        state.root = root_;
        state.focused = focused_;
        state.maximized = maximized_;
        state.next_node_id = next_node_id_;
        state.generation = generation_;
        state.nodes.reserve(nodes_.size());
        std::vector<LayoutNodeId> order;
        order.reserve(nodes_.size());
        for (const auto& [id, _] : nodes_)
            order.push_back(id);
        std::sort(order.begin(), order.end());
        for (const LayoutNodeId id : order) {
            const Node& n = nodes_.at(id);
            LayoutNodeState item;
            item.id = n.id;
            item.kind = n.kind;
            item.view_id = n.view;
            item.axis = n.axis;
            item.ratio = n.ratio;
            item.first = n.first;
            item.second = n.second;
            state.nodes.push_back(item);
        }
        return state;
    }

    lfs::Result<WorkspaceLayout> WorkspaceLayout::fromState(const WorkspaceLayoutState& state) {
        WorkspaceLayout layout;
        if (state.root == kInvalidLayoutNodeId && state.nodes.empty())
            return layout;

        LayoutNodeId max_id = 0;
        std::unordered_set<LayoutNodeId> ids;
        ids.reserve(state.nodes.size());
        for (const LayoutNodeState& item : state.nodes) {
            if (!isValidLayoutNodeId(item.id) || !ids.insert(item.id).second) {
                return layoutError(lfs::ErrorCode::DataLoss, "Layout state has invalid or duplicate node ids",
                                   LFS_SOURCE_SITE_CURRENT());
            }
            max_id = std::max(max_id, item.id);
            Node n;
            n.id = item.id;
            n.kind = item.kind;
            n.view = item.view_id;
            n.axis = item.axis;
            n.ratio = item.ratio;
            n.first = item.first;
            n.second = item.second;
            layout.nodes_.emplace(n.id, n);
        }
        layout.root_ = state.root;
        layout.focused_ = state.focused;
        layout.maximized_ = state.maximized;
        layout.next_node_id_ = 1;
        const LayoutNodeId minimum_next =
            max_id == ~LayoutNodeId{0} ? max_id : static_cast<LayoutNodeId>(max_id + 1);
        layout.bumpNextNodeId(std::max(state.next_node_id, minimum_next));
        layout.generation_ = state.generation;
        layout.reindex();
        if (auto status = layout.validate(); !status)
            return std::move(status).error();
        return layout;
    }

    void WorkspaceLayout::bumpNextNodeId(const LayoutNodeId at_least) noexcept {
        if (at_least > next_node_id_)
            next_node_id_ = at_least;
        if (next_node_id_ == kInvalidLayoutNodeId)
            next_node_id_ = 1;
        if (next_node_id_ == ~LayoutNodeId{0} && nodes_.contains(next_node_id_))
            node_id_space_exhausted_ = true;
    }

    LayoutNodeId WorkspaceLayout::allocateNode() {
        if (node_id_space_exhausted_)
            return kInvalidLayoutNodeId;
        if (next_node_id_ == kInvalidLayoutNodeId)
            next_node_id_ = 1;
        while (nodes_.contains(next_node_id_)) {
            if (next_node_id_ == ~LayoutNodeId{0}) {
                node_id_space_exhausted_ = true;
                return kInvalidLayoutNodeId;
            }
            ++next_node_id_;
        }
        const LayoutNodeId id = next_node_id_;
        if (next_node_id_ == ~LayoutNodeId{0})
            node_id_space_exhausted_ = true;
        else
            ++next_node_id_;
        return id;
    }

    void WorkspaceLayout::reindex() {
        parent_.clear();
        leaf_by_view_.clear();
        for (const auto& [id, n] : nodes_) {
            if (n.kind == LayoutNodeKind::Split) {
                if (n.first != kInvalidLayoutNodeId)
                    parent_[n.first] = id;
                if (n.second != kInvalidLayoutNodeId)
                    parent_[n.second] = id;
            } else if (isValidViewId(n.view)) {
                leaf_by_view_[n.view] = id;
            }
        }
    }

    void WorkspaceLayout::bumpGeneration() {
        ++generation_;
    }

    void WorkspaceLayout::bumpGenerationAtLeast(const std::uint64_t at_least) noexcept {
        if (generation_ < at_least)
            generation_ = at_least;
    }

    void WorkspaceLayout::collectLeaves(const LayoutNodeId id, std::vector<ViewId>& out,
                                        std::unordered_map<LayoutNodeId, int>& visiting) const {
        if (!isValidLayoutNodeId(id))
            return;
        int& state = visiting[id];
        if (state != 0)
            return;
        state = 1;
        const Node* n = node(id);
        if (n == nullptr)
            return;
        if (n->kind == LayoutNodeKind::Leaf) {
            if (isValidViewId(n->view))
                out.push_back(n->view);
            return;
        }
        collectLeaves(n->first, out, visiting);
        collectLeaves(n->second, out, visiting);
    }

    void WorkspaceLayout::solveNode(const LayoutNodeId id, const ViewRect rect, const int min_pane,
                                    const int divider, WorkspaceLayoutSnapshot& out,
                                    std::unordered_map<LayoutNodeId, int>& visiting) const {
        if (!isValidLayoutNodeId(id))
            return;
        int& state = visiting[id];
        if (state != 0)
            return;
        state = 1;
        const Node* n = node(id);
        if (n == nullptr)
            return;
        if (n->kind == LayoutNodeKind::Leaf) {
            LayoutPane pane;
            pane.id = n->view;
            pane.leaf = n->id;
            pane.rect = rect;
            out.panes.push_back(pane);
            return;
        }
        const Partition parts = partitionRect(rect, n->axis, n->ratio, min_pane, divider);
        if (!parts.splitter.empty()) {
            SplitterRect splitter;
            splitter.split = n->id;
            splitter.axis = n->axis;
            splitter.rect = parts.splitter;
            splitter.parent = rect;
            out.splitters.push_back(splitter);
        }
        solveNode(n->first, parts.first, min_pane, divider, out, visiting);
        solveNode(n->second, parts.second, min_pane, divider, out, visiting);
    }

    const WorkspaceLayout::Node* WorkspaceLayout::node(const LayoutNodeId id) const {
        const auto it = nodes_.find(id);
        return it == nodes_.end() ? nullptr : &it->second;
    }

    WorkspaceLayout::Node* WorkspaceLayout::node(const LayoutNodeId id) {
        const auto it = nodes_.find(id);
        return it == nodes_.end() ? nullptr : &it->second;
    }

} // namespace lfs::vis
