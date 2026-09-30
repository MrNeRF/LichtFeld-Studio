/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "screen/screen_layout.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <limits>
#include <nlohmann/json.hpp>
#include <set>

namespace lfs::vis::screen {

    namespace {

        using Node = ScreenLayout::Node;

        constexpr float kMinWeight = 1e-4f;
        constexpr int kMaxJsonDepth = 32;

        Node makeLeaf(const AreaId area) {
            Node node;
            node.area = area;
            return node;
        }

        // Parent of the leaf holding `area`, plus the leaf's index in it.
        struct LeafLocation {
            Node* parent = nullptr;
            std::size_t index = 0;
            Node* leaf = nullptr;
        };

        LeafLocation locate(Node& node, const AreaId area, Node* parent = nullptr, std::size_t index = 0) {
            if (node.isArea())
                return node.area == area ? LeafLocation{parent, index, &node} : LeafLocation{};
            for (std::size_t i = 0; i < node.children.size(); ++i) {
                if (auto found = locate(node.children[i], area, &node, i); found.leaf)
                    return found;
            }
            return {};
        }

        Node* findSplitNode(Node& node, const SplitId split) {
            if (node.isArea())
                return nullptr;
            if (node.split == split)
                return &node;
            for (auto& child : node.children) {
                if (auto* found = findSplitNode(child, split))
                    return found;
            }
            return nullptr;
        }

        void collectAreas(const Node& node, std::vector<AreaId>& out) {
            if (node.isArea()) {
                out.push_back(node.area);
                return;
            }
            for (const auto& child : node.children)
                collectAreas(child, out);
        }

        void renormalize(std::vector<float>& weights) {
            float sum = 0.0f;
            for (float& w : weights) {
                if (!std::isfinite(w) || w < kMinWeight)
                    w = kMinWeight;
                sum += w;
            }
            for (float& w : weights)
                w /= sum;
        }

        // Restores the tree invariants below `node`: single-child splits
        // collapse into their child, same-axis child splits are flattened into
        // their parent with scaled weights, and weights sum to one.
        void normalizeNode(Node& node) {
            if (node.isArea())
                return;
            for (auto& child : node.children)
                normalizeNode(child);

            std::vector<Node> children;
            std::vector<float> weights;
            children.reserve(node.children.size());
            for (std::size_t i = 0; i < node.children.size(); ++i) {
                Node& child = node.children[i];
                const float weight = node.weights[i];
                if (!child.isArea() && child.axis == node.axis) {
                    for (std::size_t j = 0; j < child.children.size(); ++j) {
                        children.push_back(std::move(child.children[j]));
                        weights.push_back(weight * child.weights[j]);
                    }
                } else {
                    children.push_back(std::move(child));
                    weights.push_back(weight);
                }
            }
            node.children = std::move(children);
            node.weights = std::move(weights);
            renormalize(node.weights);

            if (node.children.size() == 1) {
                Node only = std::move(node.children.front());
                node = std::move(only);
            }
        }

        // Smallest extent a subtree can take along an axis without shrinking
        // any area below the minimum.
        float minExtent(const Node& node, const SplitAxis axis, const LayoutMetrics& metrics) {
            if (node.isArea())
                return axis == SplitAxis::Columns ? metrics.min_width : metrics.min_height;
            if (node.axis == axis) {
                float total = metrics.divider * static_cast<float>(node.children.size() - 1);
                for (const auto& child : node.children)
                    total += minExtent(child, axis, metrics);
                return total;
            }
            float largest = 0.0f;
            for (const auto& child : node.children)
                largest = std::max(largest, minExtent(child, axis, metrics));
            return largest;
        }

        // Distributes `available` over children by weight, then pushes up the
        // children that fall below their minimum by taking space from the ones
        // above theirs. When the minimums cannot all be met the children share
        // the space in proportion to their minimums.
        std::vector<float> distribute(const std::vector<float>& weights, const std::vector<float>& minimums,
                                      const float available) {
            const std::size_t n = weights.size();
            std::vector<float> sizes(n);
            float min_total = 0.0f;
            for (const float m : minimums)
                min_total += m;
            if (available <= min_total) {
                for (std::size_t i = 0; i < n; ++i)
                    sizes[i] = min_total > 0.0f ? available * minimums[i] / min_total : available / static_cast<float>(n);
                return sizes;
            }
            std::vector<bool> pinned(n, false);
            for (;;) {
                float free_space = available;
                float free_weight = 0.0f;
                for (std::size_t i = 0; i < n; ++i) {
                    if (pinned[i])
                        free_space -= minimums[i];
                    else
                        free_weight += weights[i];
                }
                bool changed = false;
                for (std::size_t i = 0; i < n; ++i) {
                    if (pinned[i]) {
                        sizes[i] = minimums[i];
                        continue;
                    }
                    sizes[i] = free_weight > 0.0f ? free_space * weights[i] / free_weight : 0.0f;
                    if (sizes[i] < minimums[i]) {
                        pinned[i] = true;
                        changed = true;
                    }
                }
                if (!changed)
                    return sizes;
            }
        }

        void solveNode(const Node& node, const Rect& rect, const LayoutMetrics& metrics, LayoutGeometry& out) {
            if (node.isArea()) {
                out.areas.push_back({node.area, rect});
                return;
            }
            const bool columns = node.axis == SplitAxis::Columns;
            const float origin = columns ? rect.x : rect.y;
            const float extent = columns ? rect.w : rect.h;
            const std::size_t n = node.children.size();
            const float gaps = metrics.divider * static_cast<float>(n - 1);

            std::vector<float> minimums(n);
            for (std::size_t i = 0; i < n; ++i)
                minimums[i] = minExtent(node.children[i], node.axis, metrics);
            const auto sizes = distribute(node.weights, minimums, std::max(0.0f, extent - gaps));

            // Edges are rounded to whole units from the running total, so the
            // children tile the rect exactly.
            std::vector<float> edges(n + 1);
            edges[0] = origin;
            float running = origin;
            for (std::size_t i = 0; i < n; ++i) {
                running += sizes[i];
                edges[i + 1] = i + 1 == n ? origin + extent : std::round(running + metrics.divider * static_cast<float>(i));
            }

            float lead = origin;
            for (std::size_t i = 0; i < n; ++i) {
                const float trail = i + 1 == n ? origin + extent : edges[i + 1];
                const float child_extent = std::max(0.0f, trail - lead);
                const Rect child_rect = columns ? Rect{lead, rect.y, child_extent, rect.h}
                                                : Rect{rect.x, lead, rect.w, child_extent};
                solveNode(node.children[i], child_rect, metrics, out);

                if (i + 1 < n) {
                    DividerGeometry divider;
                    divider.split = node.split;
                    divider.index = static_cast<std::uint32_t>(i);
                    divider.axis = node.axis;
                    divider.rect = columns ? Rect{trail, rect.y, metrics.divider, rect.h}
                                           : Rect{rect.x, trail, rect.w, metrics.divider};
                    divider.parent = rect;
                    // The gap may move between the lead of child i plus its
                    // minimum and the trail of child i+1 minus its minimum.
                    const float next_trail = i + 2 == n ? origin + extent : edges[i + 2];
                    divider.min_position = lead + minimums[i];
                    divider.max_position = next_trail - minimums[i + 1] - metrics.divider;
                    out.dividers.push_back(divider);
                }
                lead = trail + (i + 1 < n ? metrics.divider : 0.0f);
            }
        }

        bool validateNode(const Node& node, std::set<std::uint32_t>& areas, std::set<std::uint32_t>& splits) {
            if (node.isArea())
                return node.children.empty() && areas.insert(node.area.value).second;
            if (!node.split.valid() || !splits.insert(node.split.value).second)
                return false;
            if (node.children.size() < 2 || node.children.size() != node.weights.size())
                return false;
            for (const float w : node.weights) {
                if (!std::isfinite(w) || w <= 0.0f)
                    return false;
            }
            return std::all_of(node.children.begin(), node.children.end(),
                               [&](const Node& child) { return validateNode(child, areas, splits); });
        }

        nlohmann::json nodeToJson(const Node& node) {
            if (node.isArea())
                return {{"area", node.area.value}};
            nlohmann::json children = nlohmann::json::array();
            for (const auto& child : node.children)
                children.push_back(nodeToJson(child));
            return {{"split", node.split.value},
                    {"axis", node.axis == SplitAxis::Columns ? "columns" : "rows"},
                    {"weights", node.weights},
                    {"children", std::move(children)}};
        }

        std::optional<Node> nodeFromJson(const nlohmann::json& json, const int depth) {
            if (depth > kMaxJsonDepth || !json.is_object())
                return std::nullopt;
            if (const auto it = json.find("area"); it != json.end()) {
                if (!it->is_number_unsigned() || it->get<std::uint64_t>() == 0 ||
                    it->get<std::uint64_t>() > std::numeric_limits<std::uint32_t>::max())
                    return std::nullopt;
                return makeLeaf(AreaId{it->get<std::uint32_t>()});
            }
            const auto split = json.find("split");
            const auto axis = json.find("axis");
            const auto weights = json.find("weights");
            const auto children = json.find("children");
            if (split == json.end() || axis == json.end() || weights == json.end() || children == json.end())
                return std::nullopt;
            if (!split->is_number_unsigned() || !axis->is_string() || !weights->is_array() || !children->is_array())
                return std::nullopt;
            if (split->get<std::uint64_t>() > std::numeric_limits<std::uint32_t>::max())
                return std::nullopt;
            Node node;
            node.split = SplitId{split->get<std::uint32_t>()};
            const auto axis_name = axis->get<std::string>();
            if (axis_name == "columns")
                node.axis = SplitAxis::Columns;
            else if (axis_name == "rows")
                node.axis = SplitAxis::Rows;
            else
                return std::nullopt;
            for (const auto& w : *weights) {
                if (!w.is_number())
                    return std::nullopt;
                node.weights.push_back(w.get<float>());
            }
            for (const auto& child_json : *children) {
                auto child = nodeFromJson(child_json, depth + 1);
                if (!child)
                    return std::nullopt;
                node.children.push_back(std::move(*child));
            }
            return node;
        }

        std::uint32_t maxSplitId(const Node& node) {
            if (node.isArea())
                return 0;
            std::uint32_t largest = node.split.value;
            for (const auto& child : node.children)
                largest = std::max(largest, maxSplitId(child));
            return largest;
        }

        bool nodesEqual(const Node& a, const Node& b) {
            if (a.area != b.area || a.split != b.split)
                return false;
            if (a.isArea())
                return true;
            if (a.axis != b.axis || a.children.size() != b.children.size())
                return false;
            for (std::size_t i = 0; i < a.children.size(); ++i) {
                if (std::abs(a.weights[i] - b.weights[i]) > 1e-5f || !nodesEqual(a.children[i], b.children[i]))
                    return false;
            }
            return true;
        }

    } // namespace

    SplitAxis axisAcross(const Side side) {
        return side == Side::Left || side == Side::Right ? SplitAxis::Columns : SplitAxis::Rows;
    }

    const AreaGeometry* LayoutGeometry::find(const AreaId area) const {
        const auto it = std::find_if(areas.begin(), areas.end(),
                                     [area](const AreaGeometry& g) { return g.area == area; });
        return it != areas.end() ? &*it : nullptr;
    }

    AreaId LayoutGeometry::areaAt(const float x, const float y) const {
        for (const auto& g : areas) {
            if (g.rect.contains(x, y))
                return g.area;
        }
        return {};
    }

    const DividerGeometry* LayoutGeometry::dividerAt(const float x, const float y, const float slop) const {
        const DividerGeometry* best = nullptr;
        float best_distance = std::numeric_limits<float>::max();
        for (const auto& d : dividers) {
            const Rect& r = d.rect;
            const bool columns = d.axis == SplitAxis::Columns;
            const float along_min = columns ? r.y : r.x;
            const float along_max = columns ? r.bottom() : r.right();
            const float along = columns ? y : x;
            if (along < along_min || along >= along_max)
                continue;
            const float across_lead = columns ? r.x : r.y;
            const float across_trail = columns ? r.right() : r.bottom();
            const float across = columns ? x : y;
            if (across < across_lead - slop || across >= across_trail + slop)
                continue;
            const float distance = std::abs(across - (across_lead + across_trail) * 0.5f);
            if (distance < best_distance) {
                best_distance = distance;
                best = &d;
            }
        }
        return best;
    }

    ScreenLayout::ScreenLayout(const AreaId only) {
        if (only.valid())
            root_ = makeLeaf(only);
    }

    std::vector<AreaId> ScreenLayout::areas() const {
        std::vector<AreaId> out;
        if (root_)
            collectAreas(*root_, out);
        return out;
    }

    bool ScreenLayout::contains(const AreaId area) const {
        if (!root_ || !area.valid())
            return false;
        return locate(const_cast<Node&>(*root_), area).leaf != nullptr;
    }

    std::size_t ScreenLayout::areaCount() const { return areas().size(); }

    bool ScreenLayout::split(const AreaId target, const AreaId added, const SplitAxis axis, const float fraction,
                             const bool new_first) {
        if (!root_ || !added.valid() || contains(added) || !(fraction > 0.0f && fraction < 1.0f))
            return false;
        auto location = locate(*root_, target);
        if (!location.leaf)
            return false;

        const float kept_share = 1.0f - fraction;
        if (location.parent && location.parent->axis == axis) {
            // Same direction as the parent: become a sibling instead of
            // nesting, so dividers along one line stay in one split.
            Node& parent = *location.parent;
            const float weight = parent.weights[location.index];
            const std::size_t insert_at = new_first ? location.index : location.index + 1;
            parent.weights[location.index] = weight * kept_share;
            parent.children.insert(parent.children.begin() + static_cast<std::ptrdiff_t>(insert_at), makeLeaf(added));
            parent.weights.insert(parent.weights.begin() + static_cast<std::ptrdiff_t>(insert_at), weight * fraction);
        } else {
            Node split_node;
            split_node.split = allocateSplit();
            split_node.axis = axis;
            if (new_first) {
                split_node.children = {makeLeaf(added), makeLeaf(target)};
                split_node.weights = {fraction, kept_share};
            } else {
                split_node.children = {makeLeaf(target), makeLeaf(added)};
                split_node.weights = {kept_share, fraction};
            }
            *location.leaf = std::move(split_node);
        }
        normalize();
        return true;
    }

    bool ScreenLayout::insertAtEdge(const AreaId added, const Side side, const float fraction) {
        if (!root_ || !added.valid() || contains(added) || !(fraction > 0.0f && fraction < 1.0f))
            return false;
        const SplitAxis axis = axisAcross(side);
        const bool at_start = side == Side::Left || side == Side::Top;
        if (root_->isArea() || root_->axis != axis) {
            Node wrapper;
            wrapper.split = allocateSplit();
            wrapper.axis = axis;
            wrapper.children.push_back(std::move(*root_));
            wrapper.weights.push_back(1.0f);
            root_ = std::move(wrapper);
        }
        for (float& w : root_->weights)
            w *= 1.0f - fraction;
        const auto at = at_start ? root_->children.begin() : root_->children.end();
        root_->children.insert(at, makeLeaf(added));
        root_->weights.insert(at_start ? root_->weights.begin() : root_->weights.end(), fraction);
        normalize();
        return true;
    }

    bool ScreenLayout::remove(const AreaId area) {
        if (!root_ || areaCount() <= 1)
            return false;
        auto location = locate(*root_, area);
        if (!location.leaf || !location.parent)
            return false;
        Node& parent = *location.parent;
        const std::size_t heir = location.index > 0 ? location.index - 1 : 1;
        parent.weights[heir] += parent.weights[location.index];
        parent.children.erase(parent.children.begin() + static_cast<std::ptrdiff_t>(location.index));
        parent.weights.erase(parent.weights.begin() + static_cast<std::ptrdiff_t>(location.index));
        normalize();
        return true;
    }

    bool ScreenLayout::canJoin(const AreaId kept, const AreaId absorbed) const {
        if (!root_ || kept == absorbed || !kept.valid() || !absorbed.valid())
            return false;
        auto& root = const_cast<Node&>(*root_);
        const auto a = locate(root, kept);
        const auto b = locate(root, absorbed);
        if (!a.leaf || !b.leaf || !a.parent || a.parent != b.parent)
            return false;
        return a.index + 1 == b.index || b.index + 1 == a.index;
    }

    bool ScreenLayout::join(const AreaId kept, const AreaId absorbed) {
        if (!canJoin(kept, absorbed))
            return false;
        auto a = locate(*root_, kept);
        const auto b = locate(*root_, absorbed);
        Node& parent = *a.parent;
        parent.weights[a.index] += parent.weights[b.index];
        parent.children.erase(parent.children.begin() + static_cast<std::ptrdiff_t>(b.index));
        parent.weights.erase(parent.weights.begin() + static_cast<std::ptrdiff_t>(b.index));
        normalize();
        return true;
    }

    AreaId ScreenLayout::joinableNeighbour(const AreaId area, const Side side) const {
        if (!root_)
            return {};
        const auto location = locate(const_cast<Node&>(*root_), area);
        if (!location.leaf || !location.parent || location.parent->axis != axisAcross(side))
            return {};
        const Node& parent = *location.parent;
        const bool forward = side == Side::Right || side == Side::Bottom;
        if (forward && location.index + 1 >= parent.children.size())
            return {};
        if (!forward && location.index == 0)
            return {};
        const Node& neighbour = parent.children[forward ? location.index + 1 : location.index - 1];
        return neighbour.isArea() ? neighbour.area : AreaId{};
    }

    bool ScreenLayout::swap(const AreaId a, const AreaId b) {
        if (!root_ || a == b)
            return false;
        auto la = locate(*root_, a);
        auto lb = locate(*root_, b);
        if (!la.leaf || !lb.leaf)
            return false;
        std::swap(la.leaf->area, lb.leaf->area);
        return true;
    }

    bool ScreenLayout::moveDivider(const DividerGeometry& divider, const float position) {
        if (!root_ || !std::isfinite(position))
            return false;
        Node* node = findSplitNode(*root_, divider.split);
        if (!node || divider.index + 1 >= node->children.size() || node->axis != divider.axis)
            return false;
        const bool columns = divider.axis == SplitAxis::Columns;
        const float lo = std::min(divider.min_position, divider.max_position);
        const float hi = std::max(divider.min_position, divider.max_position);
        const float clamped = std::clamp(position, lo, hi);
        const float current = columns ? divider.rect.x : divider.rect.y;
        const float delta = clamped - current;
        if (std::abs(delta) < 0.5f)
            return false;

        // Convert the pixel delta into weight using the parent's extent minus
        // its gaps, which is what the weights are distributed over.
        const float extent = columns ? divider.parent.w : divider.parent.h;
        const float divider_size = columns ? divider.rect.w : divider.rect.h;
        const float usable = extent - divider_size * static_cast<float>(node->children.size() - 1);
        if (usable <= 0.0f)
            return false;
        const float dw = delta / usable;
        float& lead = node->weights[divider.index];
        float& trail = node->weights[divider.index + 1];
        const float pair = lead + trail;
        lead = std::clamp(lead + dw, kMinWeight, pair - kMinWeight);
        trail = pair - lead;
        return true;
    }

    bool ScreenLayout::setWeights(const SplitId split, const std::vector<float>& weights) {
        if (!root_)
            return false;
        Node* node = findSplitNode(*root_, split);
        if (!node || weights.size() != node->children.size())
            return false;
        for (const float w : weights) {
            if (!std::isfinite(w) || w <= 0.0f)
                return false;
        }
        node->weights = weights;
        renormalize(node->weights);
        return true;
    }

    const ScreenLayout::Node* ScreenLayout::findSplit(const SplitId split) const {
        return root_ ? findSplitNode(const_cast<Node&>(*root_), split) : nullptr;
    }

    LayoutGeometry ScreenLayout::solve(const Rect& bounds, const LayoutMetrics& metrics, const AreaId maximized) const {
        LayoutGeometry out;
        out.bounds = bounds;
        if (!root_ || bounds.empty())
            return out;
        if (maximized.valid() && contains(maximized)) {
            out.maximized = maximized;
            out.areas.push_back({maximized, bounds});
            return out;
        }
        solveNode(*root_, bounds, metrics, out);
        return out;
    }

    nlohmann::json ScreenLayout::toJson() const {
        if (!root_)
            return nullptr;
        return nodeToJson(*root_);
    }

    std::optional<ScreenLayout> ScreenLayout::fromJson(const nlohmann::json& json) {
        auto root = nodeFromJson(json, 0);
        if (!root)
            return std::nullopt;
        std::set<std::uint32_t> areas;
        std::set<std::uint32_t> splits;
        if (!validateNode(*root, areas, splits))
            return std::nullopt;
        ScreenLayout layout;
        layout.root_ = std::move(root);
        layout.next_split_id_ = maxSplitId(*layout.root_) + 1;
        layout.normalize();
        return layout;
    }

    bool ScreenLayout::operator==(const ScreenLayout& other) const {
        if (root_.has_value() != other.root_.has_value())
            return false;
        return !root_ || nodesEqual(*root_, *other.root_);
    }

    void ScreenLayout::normalize() {
        if (root_)
            normalizeNode(*root_);
    }

} // namespace lfs::vis::screen
