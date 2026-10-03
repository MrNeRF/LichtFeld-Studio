/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/node_canvas_interaction.hpp"

#include "core/nodes/types.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <ranges>
#include <unordered_map>

namespace lfs::vis::gui {
    namespace {
        constexpr float kSocketHitRadius = 12.0f;
        constexpr float kSocketSnapRadius = 20.0f;
        constexpr float kLinkHitRadius = 7.0f;

        float distanceSquared(const CanvasPoint a, const CanvasPoint b) {
            const float x = a.x - b.x;
            const float y = a.y - b.y;
            return x * x + y * y;
        }

        float pointSegmentDistanceSquared(const CanvasPoint point, const CanvasPoint a,
                                          const CanvasPoint b) {
            const CanvasPoint segment = b - a;
            const float length_squared = segment.x * segment.x + segment.y * segment.y;
            if (length_squared <= 1.0e-6f)
                return distanceSquared(point, a);
            const CanvasPoint relative = point - a;
            const float t = std::clamp((relative.x * segment.x + relative.y * segment.y) /
                                           length_squared,
                                       0.0f, 1.0f);
            return distanceSquared(point, a + segment * t);
        }

        float orientation(const CanvasPoint a, const CanvasPoint b, const CanvasPoint c) {
            return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
        }

        bool segmentsIntersect(const CanvasPoint a, const CanvasPoint b, const CanvasPoint c,
                               const CanvasPoint d) {
            const float ab_c = orientation(a, b, c);
            const float ab_d = orientation(a, b, d);
            const float cd_a = orientation(c, d, a);
            const float cd_b = orientation(c, d, b);
            return ((ab_c <= 0.0f && ab_d >= 0.0f) || (ab_c >= 0.0f && ab_d <= 0.0f)) &&
                   ((cd_a <= 0.0f && cd_b >= 0.0f) || (cd_a >= 0.0f && cd_b <= 0.0f));
        }

        std::vector<CanvasPoint> bezierPolyline(const CanvasLink& link, const float minimum_tangent,
                                                const int segments = 64) {
            std::vector<CanvasPoint> points;
            points.reserve(static_cast<size_t>(segments + 1));
            const CanvasPoint a = link.from.position;
            const CanvasPoint d = link.to.position;
            const float tangent = std::max(std::abs(d.x - a.x) * 0.5f, minimum_tangent);
            const CanvasPoint b{a.x + tangent, a.y};
            const CanvasPoint c{d.x - tangent, d.y};
            for (int index = 0; index <= segments; ++index) {
                const float t = static_cast<float>(index) / static_cast<float>(segments);
                const float u = 1.0f - t;
                points.push_back(a * (u * u * u) + b * (3.0f * u * u * t) +
                                 c * (3.0f * u * t * t) + d * (t * t * t));
            }
            return points;
        }
    } // namespace

    bool CanvasRect::contains(const CanvasPoint point) const {
        return point.x >= x && point.y >= y && point.x <= x + width && point.y <= y + height;
    }

    bool CanvasRect::intersects(const CanvasRect& other) const {
        return x <= other.x + other.width && x + width >= other.x && y <= other.y + other.height &&
               y + height >= other.y;
    }

    void NodeCanvasInteraction::setGraph(std::vector<CanvasNode> nodes,
                                         std::vector<CanvasLink> links) {
        std::unordered_map<std::string, std::size_t> order;
        for (std::size_t index = 0; index < nodes_.size(); ++index)
            order.emplace(nodes_[index].id, index);
        const auto rank = [&](const CanvasNode& node) {
            const auto found = order.find(node.id);
            return found == order.end() ? nodes_.size() : found->second;
        };
        std::ranges::stable_sort(nodes, [&](const auto& a, const auto& b) { return rank(a) < rank(b); });
        nodes_ = std::move(nodes);
        links_ = std::move(links);
        if (mode_ == Mode::Idle)
            highlighted_link_.reset();
    }

    void NodeCanvasInteraction::setView(const CanvasPoint pan, const float zoom) {
        pan_ = pan;
        zoom_ = std::clamp(zoom, 0.3f, 2.5f);
    }

    void NodeCanvasInteraction::setDpRatio(const float ratio) {
        dp_ratio_ = std::max(ratio, 0.01f);
    }

    void NodeCanvasInteraction::setSelectedNodes(std::unordered_set<std::string> selected) {
        selected_nodes_ = std::move(selected);
        std::stable_partition(nodes_.begin(), nodes_.end(), [&](const CanvasNode& item) {
            return !selected_nodes_.contains(item.id);
        });
    }

    CanvasPoint NodeCanvasInteraction::screenToGraph(const CanvasPoint screen) const {
        return {(screen.x - pan_.x) / zoom_, (screen.y - pan_.y) / zoom_};
    }

    CanvasPoint NodeCanvasInteraction::graphToScreen(const CanvasPoint graph) const {
        return {graph.x * zoom_ + pan_.x, graph.y * zoom_ + pan_.y};
    }

    void NodeCanvasInteraction::zoomAbout(const CanvasPoint screen, const float factor) {
        const CanvasPoint graph = screenToGraph(screen);
        zoom_ = std::clamp(zoom_ * factor, 0.3f, 2.5f);
        pan_ = screen - graph * zoom_;
    }

    void NodeCanvasInteraction::scroll(const CanvasPoint cursor, const CanvasPoint delta,
                                       const TrackpadPreferenceState& preferences, const int touches,
                                       const bool control, const float wheel_speed) {
        if (input::trackpadSwipe(preferences, touches)) {
            if (control)
                zoomAbout(cursor, input::swipeZoomFactor(delta.y, preferences.zoom_speed));
            else
                pan_ = pan_ + CanvasPoint{-delta.x, delta.y} * (input::swipePixels(preferences.swipe_speed) * dp_ratio_);
        } else {
            zoomAbout(cursor, input::wheelZoomFactor(delta.y, wheel_speed));
        }
    }

    void NodeCanvasInteraction::pinch(const CanvasPoint cursor, const float scale, const float speed) {
        if (std::isfinite(scale) && scale > 0.0f)
            zoomAbout(cursor, input::pinchZoomFactor(scale, speed));
    }

    const CanvasNode* NodeCanvasInteraction::nodeAt(const CanvasPoint screen) const {
        const CanvasPoint graph = screenToGraph(screen);
        const auto found = std::ranges::find_if(nodes_ | std::views::reverse,
                                                [&](const CanvasNode& node) {
                                                    return node.bounds.contains(graph);
                                                });
        return found == (nodes_ | std::views::reverse).end() ? nullptr : &*found;
    }

    std::optional<CanvasSocket> NodeCanvasInteraction::socketAt(const CanvasPoint screen) const {
        std::optional<CanvasSocket> result;
        float best = kSocketHitRadius * kSocketHitRadius * dp_ratio_ * dp_ratio_;
        for (const auto& node : nodes_)
            for (const auto& socket : node.sockets) {
                const auto* front = nodeAt(screen);
                const auto* covering = nodeAt(graphToScreen(socket.position));
                if ((front && front->id != node.id) || (covering && covering->id != node.id))
                    continue;
                const float distance = distanceSquared(screen, graphToScreen(socket.position));
                if (distance <= best) {
                    best = distance;
                    result = socket;
                }
            }
        return result;
    }

    bool NodeCanvasInteraction::compatible(const CanvasSocket& from, const CanvasSocket& to) const {
        return from.direction == CanvasSocketDirection::Output &&
               to.direction == CanvasSocketDirection::Input && from.node != to.node &&
               lfs::nodes::can_convert_socket(from.type, to.type);
    }

    bool NodeCanvasInteraction::inputOccupied(const CanvasSocket& socket) const {
        return std::ranges::any_of(links_, [&](const CanvasLink& link) {
            return link.to.node == socket.node && link.to.identifier == socket.identifier;
        });
    }

    std::optional<CanvasSocket> NodeCanvasInteraction::snapSocket(const CanvasPoint screen) const {
        if (!wire_source_)
            return std::nullopt;
        std::optional<CanvasSocket> result;
        float best = kSocketSnapRadius * kSocketSnapRadius * dp_ratio_ * dp_ratio_;
        for (const auto& node : nodes_)
            for (const auto& socket : node.sockets) {
                const auto* covering = nodeAt(graphToScreen(socket.position));
                if (covering && covering->id != node.id)
                    continue;
                CanvasSocket from = *wire_source_;
                CanvasSocket to = socket;
                if (from.direction == CanvasSocketDirection::Input)
                    std::swap(from, to);
                if (!compatible(from, to))
                    continue;
                const float distance = distanceSquared(screen, graphToScreen(socket.position));
                if (distance <= best) {
                    best = distance;
                    result = socket;
                }
            }
        return result;
    }

    std::optional<CanvasSocket> NodeCanvasInteraction::compatibleBodyInput(
        const CanvasPoint screen) const {
        const auto* node = nodeAt(screen);
        if (!node || !wire_source_)
            return std::nullopt;
        for (const auto& socket : node->sockets) {
            CanvasSocket from = *wire_source_;
            CanvasSocket to = socket;
            if (from.direction == CanvasSocketDirection::Input)
                std::swap(from, to);
            if (compatible(from, to) && (socket.multi_input || !inputOccupied(socket)))
                return socket;
        }
        for (const auto& socket : node->sockets) {
            CanvasSocket from = *wire_source_;
            CanvasSocket to = socket;
            if (from.direction == CanvasSocketDirection::Input)
                std::swap(from, to);
            if (compatible(from, to))
                return socket;
        }
        return std::nullopt;
    }

    std::optional<CanvasLink> NodeCanvasInteraction::linkAt(const CanvasPoint screen,
                                                            const float radius) const {
        const CanvasPoint graph = screenToGraph(screen);
        const float graph_radius_squared = radius * radius / (zoom_ * zoom_);
        std::optional<CanvasLink> result;
        float best = graph_radius_squared;
        for (const auto& link : links_) {
            const auto points = bezierPolyline(link, 36.0f * dp_ratio_ / zoom_);
            for (size_t index = 1; index < points.size(); ++index) {
                const float distance = pointSegmentDistanceSquared(graph, points[index - 1], points[index]);
                if (distance <= best) {
                    best = distance;
                    result = link;
                }
            }
        }
        return result;
    }

    bool NodeCanvasInteraction::nodeHasLinks(const std::string_view node) const {
        return std::ranges::any_of(links_, [&](const CanvasLink& link) {
            return link.from.node == node || link.to.node == node;
        });
    }

    std::optional<CanvasLink> NodeCanvasInteraction::spliceTarget(const CanvasNode& node) const {
        if (nodeHasLinks(node.id))
            return std::nullopt;
        for (const auto& link : links_) {
            const auto type_input = std::ranges::find_if(node.sockets, [&](const CanvasSocket& socket) {
                return socket.direction == CanvasSocketDirection::Input && compatible(link.from, socket);
            });
            const auto type_output = std::ranges::find_if(node.sockets, [&](const CanvasSocket& socket) {
                return socket.direction == CanvasSocketDirection::Output && compatible(socket, link.to);
            });
            if (type_input == node.sockets.end() || type_output == node.sockets.end())
                continue;
            CanvasPoint center{node.bounds.x + node.bounds.width * 0.5f,
                               node.bounds.y + node.bounds.height * 0.5f};
            const auto points = bezierPolyline(link, 36.0f * dp_ratio_ / zoom_);
            for (size_t index = 1; index < points.size(); ++index)
                if (pointSegmentDistanceSquared(center, points[index - 1], points[index]) <=
                    12.0f * 12.0f * dp_ratio_ * dp_ratio_ / (zoom_ * zoom_))
                    return link;
        }
        return std::nullopt;
    }

    std::vector<CanvasLink> NodeCanvasInteraction::cutLinks() const {
        std::vector<CanvasLink> result;
        if (cut_points_.size() < 2)
            return result;
        for (const auto& link : links_) {
            const auto wire = bezierPolyline(link, 36.0f * dp_ratio_ / zoom_);
            bool crossed = false;
            for (size_t cut = 1; cut < cut_points_.size() && !crossed; ++cut) {
                const CanvasPoint a = screenToGraph(cut_points_[cut - 1]);
                const CanvasPoint b = screenToGraph(cut_points_[cut]);
                for (size_t wire_index = 1; wire_index < wire.size(); ++wire_index)
                    if (segmentsIntersect(a, b, wire[wire_index - 1], wire[wire_index])) {
                        crossed = true;
                        break;
                    }
            }
            if (crossed)
                result.push_back(link);
        }
        return result;
    }

    std::vector<CanvasCommand> NodeCanvasInteraction::pointerDown(const CanvasPoint screen,
                                                                  const CanvasPointerButton button,
                                                                  const CanvasModifiers modifiers, const bool pan_drag) {
        if (mode_ != Mode::Idle)
            return {};
        start_ = last_ = pointer_ = screen;
        modifiers_ = modifiers;
        snapped_.reset();
        highlighted_link_.reset();

        if (pan_drag || button == CanvasPointerButton::Middle) {
            mode_ = Mode::Pan;
            return {};
        }

        if ((button == CanvasPointerButton::Right && modifiers.control) ||
            (button == CanvasPointerButton::Left && modifiers.control && modifiers.alt)) {
            mode_ = Mode::Cut;
            cut_points_ = {screen};
            return {};
        }
        if (button != CanvasPointerButton::Left)
            return {};
        if (const auto socket = socketAt(screen)) {
            mode_ = Mode::Wire;
            wire_source_ = socket;
            if (socket->direction == CanvasSocketDirection::Input) {
                const auto found = std::ranges::find_if(links_, [&](const CanvasLink& link) {
                    return link.to.node == socket->node && link.to.identifier == socket->identifier;
                });
                if (found != links_.end()) {
                    detached_link_ = *found;
                    wire_source_ = found->from;
                }
            }
            return {};
        }
        if (const auto* node = nodeAt(screen)) {
            if (!selected_nodes_.contains(node->id)) {
                if (!modifiers.shift)
                    selected_nodes_.clear();
                selected_nodes_.insert(node->id);
            }
            std::stable_partition(nodes_.begin(), nodes_.end(), [&](const CanvasNode& item) {
                return !selected_nodes_.contains(item.id);
            });
            move_start_nodes_ = nodes_;
            move_start_links_ = links_;
            mode_ = Mode::MoveNodes;
            return {{.kind = CanvasCommandKind::Select,
                     .nodes = std::vector<std::string>(selected_nodes_.begin(), selected_nodes_.end()),
                     .additive = modifiers.shift}};
        }
        if (const auto link = linkAt(screen, kLinkHitRadius * dp_ratio_)) {
            highlighted_link_ = link;
            return {{.kind = CanvasCommandKind::SelectLink, .before = link}};
        }
        mode_ = Mode::BoxSelect;
        if (!modifiers.shift)
            selected_nodes_.clear();
        return {};
    }

    std::vector<CanvasCommand> NodeCanvasInteraction::pointerMove(const CanvasPoint screen) {
        if (mode_ == Mode::Idle)
            return {};
        pointer_ = screen;
        CanvasPoint pan_delta;
        if ((mode_ == Mode::MoveNodes || mode_ == Mode::Wire) && viewport_.width > 0.0f &&
            viewport_.height > 0.0f) {
            const float edge = 24.0f * dp_ratio_;
            const float speed = 8.0f * dp_ratio_;
            if (screen.x < viewport_.x + edge)
                pan_delta.x = speed;
            else if (screen.x > viewport_.x + viewport_.width - edge)
                pan_delta.x = -speed;
            if (screen.y < viewport_.y + edge)
                pan_delta.y = speed;
            else if (screen.y > viewport_.y + viewport_.height - edge)
                pan_delta.y = -speed;
            pan_ = pan_ + pan_delta;
        }
        const CanvasPoint delta = screen - last_;
        last_ = screen;
        if (mode_ == Mode::Pan) {
            pan_ = pan_ + delta;
        } else if (mode_ == Mode::Wire) {
            snapped_ = snapSocket(screen);
            if (!snapped_)
                snapped_ = compatibleBodyInput(screen);
        } else if (mode_ == Mode::Cut) {
            cut_points_.push_back(screen);
        } else if (mode_ == Mode::MoveNodes) {
            const CanvasPoint graph_delta = (delta - pan_delta) * (1.0f / zoom_);
            for (auto& node : nodes_)
                if (selected_nodes_.contains(node.id)) {
                    node.bounds.x += graph_delta.x;
                    node.bounds.y += graph_delta.y;
                    for (auto& socket : node.sockets)
                        socket.position = socket.position + graph_delta;
                }
            for (auto& link : links_) {
                if (selected_nodes_.contains(link.from.node))
                    link.from.position = link.from.position + graph_delta;
                if (selected_nodes_.contains(link.to.node))
                    link.to.position = link.to.position + graph_delta;
            }
            if (selected_nodes_.size() == 1) {
                const auto found = std::ranges::find_if(nodes_, [&](const CanvasNode& node) {
                    return selected_nodes_.contains(node.id);
                });
                highlighted_link_ = found == nodes_.end() ? std::nullopt : spliceTarget(*found);
            }
        }
        return {};
    }

    std::vector<CanvasCommand> NodeCanvasInteraction::pointerUp(const CanvasPoint screen) {
        if (mode_ == Mode::Idle)
            return {};
        (void)pointerMove(screen);
        std::vector<CanvasCommand> commands;
        if (mode_ == Mode::Wire && wire_source_) {
            if (snapped_) {
                CanvasSocket from = *wire_source_;
                CanvasSocket to = *snapped_;
                if (from.direction == CanvasSocketDirection::Input)
                    std::swap(from, to);
                const CanvasLink connected{from, to};
                if (compatible(from, to)) {
                    if (detached_link_)
                        commands.push_back({.kind = CanvasCommandKind::ReRoute,
                                            .before = detached_link_,
                                            .after = connected});
                    else
                        commands.push_back({.kind = CanvasCommandKind::Connect, .after = connected});
                }
            } else if (detached_link_) {
                commands.push_back({.kind = CanvasCommandKind::Disconnect, .before = detached_link_});
            }
        } else if (mode_ == Mode::Cut) {
            auto crossed = cutLinks();
            if (!crossed.empty())
                commands.push_back({.kind = CanvasCommandKind::DeleteLinks,
                                    .links = std::move(crossed)});
        } else if (mode_ == Mode::BoxSelect) {
            const CanvasPoint a = screenToGraph(start_);
            const CanvasPoint b = screenToGraph(screen);
            const CanvasRect box{std::min(a.x, b.x), std::min(a.y, b.y), std::abs(b.x - a.x),
                                 std::abs(b.y - a.y)};
            for (const auto& node : nodes_)
                if (box.intersects(node.bounds))
                    selected_nodes_.insert(node.id);
            commands.push_back({.kind = CanvasCommandKind::Select,
                                .nodes = std::vector<std::string>(selected_nodes_.begin(),
                                                                  selected_nodes_.end()),
                                .additive = modifiers_.shift});
        } else if (mode_ == Mode::MoveNodes) {
            CanvasPoint graph_delta;
            for (size_t index = 0; index < nodes_.size(); ++index)
                if (selected_nodes_.contains(nodes_[index].id)) {
                    graph_delta = {nodes_[index].bounds.x - move_start_nodes_[index].bounds.x,
                                   nodes_[index].bounds.y - move_start_nodes_[index].bounds.y};
                    break;
                }
            if (highlighted_link_ && selected_nodes_.size() == 1) {
                commands.push_back({.kind = CanvasCommandKind::Splice,
                                    .nodes = {*selected_nodes_.begin()},
                                    .before = highlighted_link_});
            } else if (std::abs(graph_delta.x) > 0.001f || std::abs(graph_delta.y) > 0.001f) {
                commands.push_back({.kind = CanvasCommandKind::Move,
                                    .nodes = std::vector<std::string>(selected_nodes_.begin(),
                                                                      selected_nodes_.end()),
                                    .delta = graph_delta});
            }
        }
        mode_ = Mode::Idle;
        wire_source_.reset();
        detached_link_.reset();
        snapped_.reset();
        highlighted_link_.reset();
        cut_points_.clear();
        move_start_nodes_.clear();
        move_start_links_.clear();
        return commands;
    }

    void NodeCanvasInteraction::cancel() {
        if (mode_ == Mode::MoveNodes && !move_start_nodes_.empty()) {
            nodes_ = move_start_nodes_;
            links_ = move_start_links_;
        }
        mode_ = Mode::Idle;
        wire_source_.reset();
        detached_link_.reset();
        snapped_.reset();
        highlighted_link_.reset();
        cut_points_.clear();
        move_start_nodes_.clear();
        move_start_links_.clear();
    }

    bool NodeCanvasInteraction::draggingWire() const { return mode_ == Mode::Wire; }
    bool NodeCanvasInteraction::active() const { return mode_ != Mode::Idle; }

    std::optional<CanvasRect> NodeCanvasInteraction::selectionBox() const {
        if (mode_ != Mode::BoxSelect)
            return std::nullopt;
        return CanvasRect{std::min(start_.x, pointer_.x), std::min(start_.y, pointer_.y),
                          std::abs(pointer_.x - start_.x), std::abs(pointer_.y - start_.y)};
    }

    bool NodeCanvasInteraction::canSnapTo(const CanvasSocket& socket) const {
        if (!wire_source_)
            return true;
        CanvasSocket from = *wire_source_;
        CanvasSocket to = socket;
        if (from.direction == CanvasSocketDirection::Input)
            std::swap(from, to);
        return compatible(from, to);
    }

} // namespace lfs::vis::gui
