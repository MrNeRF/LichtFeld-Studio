/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "input/navigation_gestures.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace lfs::vis::gui {

    struct CanvasPoint {
        float x = 0.0f;
        float y = 0.0f;

        CanvasPoint operator+(const CanvasPoint& other) const { return {x + other.x, y + other.y}; }
        CanvasPoint operator-(const CanvasPoint& other) const { return {x - other.x, y - other.y}; }
        CanvasPoint operator*(float scalar) const { return {x * scalar, y * scalar}; }
        bool operator==(const CanvasPoint&) const = default;
    };

    struct CanvasRect {
        float x = 0.0f;
        float y = 0.0f;
        float width = 0.0f;
        float height = 0.0f;

        [[nodiscard]] bool contains(CanvasPoint point) const;
        [[nodiscard]] bool intersects(const CanvasRect& other) const;
        bool operator==(const CanvasRect&) const = default;
    };

    enum class CanvasSocketDirection { Input,
                                       Output };

    struct CanvasSocket {
        std::string node;
        std::string identifier;
        std::string type;
        CanvasSocketDirection direction = CanvasSocketDirection::Input;
        CanvasPoint position;
        bool multi_input = false;

        bool operator==(const CanvasSocket&) const = default;
    };

    struct CanvasNode {
        std::string id;
        CanvasRect bounds;
        std::vector<CanvasSocket> sockets;
        bool operator==(const CanvasNode&) const = default;
    };

    struct CanvasLink {
        CanvasSocket from;
        CanvasSocket to;

        bool operator==(const CanvasLink&) const = default;
    };

    enum class CanvasCommandKind { Select,
                                   SelectLink,
                                   Move,
                                   Connect,
                                   ReRoute,
                                   Disconnect,
                                   DeleteLinks,
                                   Splice };

    struct CanvasCommand {
        CanvasCommandKind kind = CanvasCommandKind::Select;
        std::vector<std::string> nodes;
        std::vector<CanvasLink> links;
        std::optional<CanvasLink> before;
        std::optional<CanvasLink> after;
        CanvasPoint delta;
        bool additive = false;
    };

    struct CanvasModifiers {
        bool shift = false;
        bool control = false;
        bool alt = false;
    };

    enum class CanvasPointerButton { Left,
                                     Right,
                                     Middle };

    class LFS_VIS_API NodeCanvasInteraction {
    public:
        void setGraph(std::vector<CanvasNode> nodes, std::vector<CanvasLink> links);
        void setView(CanvasPoint pan, float zoom);
        void setDpRatio(float ratio);
        void setViewport(CanvasRect viewport) { viewport_ = viewport; }
        void setSelectedNodes(std::unordered_set<std::string> selected);

        [[nodiscard]] std::vector<CanvasCommand> pointerDown(CanvasPoint screen,
                                                             CanvasPointerButton button,
                                                             CanvasModifiers modifiers = {}, bool pan_drag = false);
        [[nodiscard]] std::vector<CanvasCommand> pointerMove(CanvasPoint screen);
        [[nodiscard]] std::vector<CanvasCommand> pointerUp(CanvasPoint screen);
        void cancel();

        [[nodiscard]] CanvasPoint pan() const { return pan_; }
        [[nodiscard]] float zoom() const { return zoom_; }
        void zoomAbout(CanvasPoint screen, float factor);
        void scroll(CanvasPoint cursor, CanvasPoint delta, const TrackpadPreferenceState& preferences,
                    int touches, bool control, float wheel_speed);
        void pinch(CanvasPoint cursor, float scale, float speed);
        [[nodiscard]] CanvasPoint screenToGraph(CanvasPoint screen) const;
        [[nodiscard]] CanvasPoint graphToScreen(CanvasPoint graph) const;
        [[nodiscard]] const std::optional<CanvasSocket>& snappedSocket() const { return snapped_; }
        [[nodiscard]] const std::optional<CanvasLink>& highlightedLink() const { return highlighted_link_; }
        [[nodiscard]] const std::optional<CanvasSocket>& wireSource() const { return wire_source_; }
        [[nodiscard]] const std::unordered_set<std::string>& selectedNodes() const { return selected_nodes_; }
        [[nodiscard]] const std::vector<CanvasNode>& nodes() const { return nodes_; }
        [[nodiscard]] const std::vector<CanvasLink>& links() const { return links_; }
        // Graph-space routes are shared by drawing, selection and knife gestures.
        // View-only changes do not invalidate them.
        [[nodiscard]] const std::vector<std::vector<CanvasPoint>>& wirePaths() const;
        [[nodiscard]] std::uint64_t routeGeneration() const {
            (void)wirePaths();
            return route_generation_;
        }
        [[nodiscard]] bool draggingWire() const;
        [[nodiscard]] bool active() const;
        [[nodiscard]] CanvasPoint pointer() const { return pointer_; }
        [[nodiscard]] std::optional<CanvasRect> selectionBox() const;
        [[nodiscard]] const std::vector<CanvasPoint>& cutPoints() const { return cut_points_; }
        [[nodiscard]] bool canSnapTo(const CanvasSocket& socket) const;
        [[nodiscard]] std::unordered_map<std::string, CanvasPoint>
        arrangedPositions(std::string_view input, std::string_view output,
                          const std::unordered_set<std::string>& subset = {}) const;

    private:
        enum class Mode { Idle,
                          Pan,
                          BoxSelect,
                          MoveNodes,
                          Wire,
                          Cut };

        [[nodiscard]] const CanvasNode* nodeAt(CanvasPoint screen) const;
        [[nodiscard]] std::optional<CanvasSocket> socketAt(CanvasPoint screen) const;
        [[nodiscard]] std::optional<CanvasSocket> snapSocket(CanvasPoint screen) const;
        [[nodiscard]] std::optional<CanvasSocket> compatibleBodyInput(CanvasPoint screen) const;
        [[nodiscard]] std::optional<CanvasLink> linkAt(CanvasPoint screen, float radius) const;
        [[nodiscard]] bool compatible(const CanvasSocket& from, const CanvasSocket& to) const;
        [[nodiscard]] bool inputOccupied(const CanvasSocket& socket) const;
        [[nodiscard]] bool nodeHasLinks(std::string_view node) const;
        [[nodiscard]] std::optional<CanvasLink> spliceTarget(const CanvasNode& node) const;
        [[nodiscard]] std::vector<CanvasLink> cutLinks() const;

        std::vector<CanvasNode> nodes_;
        std::vector<CanvasLink> links_;
        std::unordered_set<std::string> selected_nodes_;
        Mode mode_ = Mode::Idle;
        CanvasPoint pan_;
        float zoom_ = 1.0f;
        float dp_ratio_ = 1.0f;
        CanvasPoint start_;
        CanvasPoint pointer_;
        CanvasPoint last_;
        CanvasModifiers modifiers_;
        std::optional<CanvasSocket> wire_source_;
        std::optional<CanvasLink> detached_link_;
        std::optional<CanvasSocket> snapped_;
        std::optional<CanvasLink> highlighted_link_;
        std::vector<CanvasPoint> cut_points_;
        std::vector<CanvasNode> move_start_nodes_;
        std::vector<CanvasLink> move_start_links_;
        std::vector<std::vector<CanvasPoint>> move_start_paths_;
        mutable std::vector<CanvasNode> routed_nodes_;
        mutable std::vector<CanvasLink> routed_links_;
        mutable std::vector<std::vector<CanvasPoint>> wire_paths_;
        mutable float routed_dp_ratio_ = 0.0f;
        mutable std::uint64_t route_generation_ = 0;
        CanvasRect viewport_;
    };

} // namespace lfs::vis::gui
