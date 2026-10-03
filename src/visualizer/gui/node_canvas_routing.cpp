/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "gui/node_canvas_interaction.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <queue>
#include <ranges>

namespace lfs::vis::gui {
    namespace {
        using Path = std::vector<CanvasPoint>;

        bool crosses(const CanvasPoint a, const CanvasPoint b, const CanvasRect& box) {
            // Open rectangle: travelling along a clearance boundary is safe.
            float low = 0.0f;
            float high = 1.0f;
            const auto axis = [&](const float origin, const float delta, const float first, const float last) {
                if (std::abs(delta) < 1e-6f)
                    return origin > first && origin < last;
                const float t0 = (first - origin) / delta;
                const float t1 = (last - origin) / delta;
                low = std::max(low, std::min(t0, t1));
                high = std::min(high, std::max(t0, t1));
                return low < high;
            };
            return axis(a.x, b.x - a.x, box.x, box.x + box.width) &&
                   axis(a.y, b.y - a.y, box.y, box.y + box.height) && low < high;
        }

        bool clear(const Path& path, const std::vector<CanvasRect>& boxes) {
            for (std::size_t i = 1; i < path.size(); ++i)
                for (const auto& box : boxes)
                    if (crosses(path[i - 1], path[i], box))
                        return false;
            return true;
        }

        Path curve(const CanvasPoint a, const CanvasPoint d, const float dp) {
            const float tangent = std::max(std::abs(d.x - a.x) * 0.5f, 24.0f * dp);
            const CanvasPoint b{a.x + tangent, a.y};
            const CanvasPoint c{d.x - tangent, d.y};
            Path result;
            for (int i = 0; i <= 48; ++i) {
                const float t = static_cast<float>(i) / 48;
                const float u = 1.0f - t;
                result.push_back(a * (u * u * u) + b * (3 * u * u * t) + c * (3 * u * t * t) + d * (t * t * t));
            }
            return result;
        }

        Path rounded(const Path& input, const float radius) {
            Path corners;
            for (const auto p : input) {
                if (!corners.empty() && p == corners.back())
                    continue;
                if (corners.size() > 1) {
                    const auto a = corners.back() - corners[corners.size() - 2];
                    const auto b = p - corners.back();
                    if (std::abs(a.x * b.y - a.y * b.x) < 0.01f && a.x * b.x + a.y * b.y >= 0)
                        corners.pop_back();
                }
                corners.push_back(p);
            }
            if (corners.size() < 3)
                return corners;
            Path result{corners.front()};
            for (std::size_t i = 1; i + 1 < corners.size(); ++i) {
                const auto a = corners[i - 1] - corners[i];
                const auto b = corners[i + 1] - corners[i];
                const float la = std::hypot(a.x, a.y);
                const float lb = std::hypot(b.x, b.y);
                const float r = std::min({radius, la * 0.45f, lb * 0.45f});
                const auto start = corners[i] + a * (r / la);
                const auto end = corners[i] + b * (r / lb);
                result.push_back(start);
                for (int step = 1; step <= 8; ++step) {
                    const float t = static_cast<float>(step) / 8;
                    const float u = 1.0f - t;
                    result.push_back(start * (u * u) + corners[i] * (2 * u * t) + end * (t * t));
                }
            }
            result.push_back(corners.back());
            return result;
        }

        // Coordinate-compressed rectilinear visibility grid. Rectangle interiors
        // block edges as well as vertices, including very thin obstacles. This
        // fallback is only needed when a simple above/below lane cannot connect.
        Path gridRoute(const CanvasPoint start, const CanvasPoint end,
                       const std::vector<CanvasRect>& boxes, const float turn_cost) {
            std::vector<float> xs{start.x, end.x}, ys{start.y, end.y};
            for (const auto& box : boxes) {
                xs.insert(xs.end(), {box.x, box.x + box.width});
                ys.insert(ys.end(), {box.y, box.y + box.height});
            }
            const auto unique = [](auto& values) {
                std::ranges::sort(values);
                values.erase(std::unique(values.begin(), values.end()), values.end());
            };
            unique(xs);
            unique(ys);
            const auto index = [](const auto& values, const float value) {
                return static_cast<int>(std::lower_bound(values.begin(), values.end(), value) - values.begin());
            };
            const int nx = static_cast<int>(xs.size());
            const int ny = static_cast<int>(ys.size());
            const int count = nx * ny;
            std::vector<bool> horizontal(count), vertical(count);
            for (const auto& box : boxes) {
                const int x0 = index(xs, box.x);
                const int x1 = index(xs, box.x + box.width);
                const int y0 = index(ys, box.y);
                const int y1 = index(ys, box.y + box.height);
                for (int y = y0 + 1; y < y1; ++y)
                    for (int x = x0; x < x1; ++x)
                        horizontal[y * nx + x] = true;
                for (int y = y0; y < y1; ++y)
                    for (int x = x0 + 1; x < x1; ++x)
                        vertical[y * nx + x] = true;
            }
            const int first = index(ys, start.y) * nx + index(xs, start.x);
            const int last = index(ys, end.y) * nx + index(xs, end.x);
            const auto point = [&](const int i) { return CanvasPoint{xs[i % nx], ys[i / nx]}; };
            const auto heuristic = [&](const int i) {
                const auto p = point(i);
                return std::abs(end.x - p.x) + std::abs(end.y - p.y);
            };
            std::vector<float> cost(count * 2, std::numeric_limits<float>::infinity());
            std::vector<int> previous(count * 2, -1);
            using Entry = std::pair<float, int>;
            std::priority_queue<Entry, std::vector<Entry>, std::greater<>> open;
            cost[first * 2] = 0;
            open.emplace(heuristic(first), first * 2);
            int finish = -1;
            while (!open.empty()) {
                const auto [score, state] = open.top();
                open.pop();
                const int current = state / 2;
                if (score > cost[state] + heuristic(current) + 0.01f)
                    continue;
                if (current == last) {
                    finish = state;
                    break;
                }
                const int x = current % nx;
                const int y = current / nx;
                const auto visit = [&](const int next, const int direction, const bool blocked) {
                    if (blocked)
                        return;
                    const auto delta = point(next) - point(current);
                    const float candidate = cost[state] + std::abs(delta.x) + std::abs(delta.y) +
                                            (direction == state % 2 ? 0.0f : turn_cost);
                    const int next_state = next * 2 + direction;
                    if (candidate < cost[next_state]) {
                        cost[next_state] = candidate;
                        previous[next_state] = state;
                        open.emplace(candidate + heuristic(next), next_state);
                    }
                };
                if (x + 1 < nx)
                    visit(current + 1, 0, horizontal[current]);
                if (x > 0)
                    visit(current - 1, 0, horizontal[current - 1]);
                if (y + 1 < ny)
                    visit(current + nx, 1, vertical[current]);
                if (y > 0)
                    visit(current - nx, 1, vertical[current - nx]);
            }
            Path result;
            for (int state = finish; state >= 0; state = previous[state])
                result.push_back(point(state / 2));
            std::ranges::reverse(result);
            return result;
        }

        Path route(const CanvasLink& link, const std::vector<CanvasNode>& nodes,
                   const float dp, const std::size_t lane, const bool narrow = false) {
            const auto a = link.from.position;
            const auto d = link.to.position;
            // Successive links occupy separate clearance lanes. Compress large
            // bundles within 26 dp so Arrange's 64 dp corridors remain open.
            const float clearance = narrow ? 4.0f * dp : (12.0f + 14.0f * (1.0f - std::exp(-0.3f * lane))) * dp;
            std::vector<CanvasRect> boxes;
            std::vector<CanvasRect> cards;
            for (const auto& node : nodes) {
                const auto& r = node.bounds;
                cards.push_back(r);
                boxes.push_back({r.x - clearance, r.y - clearance,
                                 r.width + clearance * 2, r.height + clearance * 2});
            }
            auto direct = curve(a, d, dp);
            if (d.x > a.x && clear(direct, cards))
                return direct;
            const CanvasPoint start{a.x + clearance, a.y};
            const CanvasPoint end{d.x - clearance, d.y};
            Path middle;
            std::vector<float> lanes{start.y, end.y};
            for (const auto& box : boxes)
                lanes.insert(lanes.end(), {box.y, box.y + box.height});
            std::ranges::stable_sort(lanes, [&](const float x, const float y) {
                return std::abs(x - start.y) + std::abs(x - end.y) < std::abs(y - start.y) + std::abs(y - end.y);
            });
            for (const float y : lanes) {
                Path candidate{start, {start.x, y}, {end.x, y}, end};
                if (clear(candidate, boxes)) {
                    middle = std::move(candidate);
                    break;
                }
            }
            if (middle.empty())
                middle = gridRoute(start, end, boxes, 16.0f * dp);
            // Overlapping cards can cover a socket completely: no unobstructed
            // path exists until the user moves/arranges them. Keep the link.
            if (middle.empty())
                return narrow ? direct : route(link, nodes, dp, lane, true);
            middle.insert(middle.begin(), a);
            middle.push_back(d);
            auto smooth = rounded(middle, (narrow ? 2.0f : 8.0f) * dp);
            return clear(smooth, cards) ? smooth : middle;
        }
    } // namespace

    const std::vector<std::vector<CanvasPoint>>& NodeCanvasInteraction::wirePaths() const {
        const bool same_bounds = nodes_.size() == routed_nodes_.size() && std::ranges::all_of(nodes_, [&](const auto& node) {
                                     const auto found = std::ranges::find(routed_nodes_, node.id, &CanvasNode::id);
                                     return found != routed_nodes_.end() && found->bounds == node.bounds;
                                 });
        if (same_bounds && links_ == routed_links_ && dp_ratio_ == routed_dp_ratio_)
            return wire_paths_;
        wire_paths_.clear();
        std::unordered_map<std::string, std::size_t> lanes;
        for (const auto& link : links_) {
            // Fan-out shares a departure corridor, fan-in shares an arrival
            // corridor. Use the busier side's next lane, without relying on ids
            // being contiguous or normalising plugin identifiers.
            auto& departure = lanes["out:" + link.from.node];
            auto& arrival = lanes["in:" + link.to.node];
            const auto lane = std::max(departure++, arrival++);
            wire_paths_.push_back(route(link, nodes_, dp_ratio_, lane));
        }
        routed_nodes_ = nodes_;
        routed_links_ = links_;
        routed_dp_ratio_ = dp_ratio_;
        ++route_generation_;
        return wire_paths_;
    }
} // namespace lfs::vis::gui
