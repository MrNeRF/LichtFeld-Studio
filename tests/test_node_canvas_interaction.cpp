/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/node_canvas_interaction.hpp"
#include "input/input_bindings.hpp"

#include <chrono>
#include <cmath>
#include <gtest/gtest.h>
#include <iostream>

namespace {
    using namespace lfs::vis::gui;

    CanvasSocket output(std::string node, std::string type, CanvasPoint position) {
        return {.node = std::move(node),
                .identifier = "out",
                .type = std::move(type),
                .direction = CanvasSocketDirection::Output,
                .position = position};
    }

    CanvasSocket input(std::string node, std::string type, CanvasPoint position,
                       const bool multi = false) {
        return {.node = std::move(node),
                .identifier = "in",
                .type = std::move(type),
                .direction = CanvasSocketDirection::Input,
                .position = position,
                .multi_input = multi};
    }

    NodeCanvasInteraction graph(const bool multi = false) {
        NodeCanvasInteraction interaction;
        interaction.setGraph(
            {{.id = "A", .bounds = {0, 0, 100, 80}, .sockets = {output("A", "lfs.float", {100, 40})}},
             {.id = "B", .bounds = {260, 0, 100, 80}, .sockets = {input("B", "lfs.float", {260, 40}, multi)}},
             {.id = "C", .bounds = {260, 160, 100, 80}, .sockets = {input("C", "lfs.geometry", {260, 200})}}},
            {});
        return interaction;
    }

    TEST(NodeCanvasInteraction, ConnectsAndSnapsWithinScreenRadius) {
        auto interaction = graph();
        EXPECT_TRUE(interaction.pointerDown({100, 40}, CanvasPointerButton::Left).empty());
        (void)interaction.pointerMove({279, 40});
        ASSERT_TRUE(interaction.snappedSocket());
        const auto commands = interaction.pointerUp({279, 40});
        ASSERT_EQ(commands.size(), 1u);
        EXPECT_EQ(commands[0].kind, CanvasCommandKind::Connect);
        EXPECT_EQ(commands[0].after->from.node, "A");
        EXPECT_EQ(commands[0].after->to.node, "B");
    }

    TEST(NodeCanvasInteraction, NavigationUsesViewportDeviceClassificationAndSpeeds) {
        using namespace lfs::vis;
        auto interaction = graph();
        const CanvasPoint cursor{120, 90};
        const auto anchor = interaction.screenToGraph(cursor);
        TrackpadPreferenceState preferences;
        interaction.scroll(cursor, {0, 1}, preferences, 0, false, 11);
        EXPECT_NEAR(interaction.zoom(), 1.0f / 0.89f, 1e-6f);
        EXPECT_NEAR(interaction.screenToGraph(cursor).x, anchor.x, 1e-5f);
        EXPECT_NEAR(interaction.screenToGraph(cursor).y, anchor.y, 1e-5f);
        preferences.device = NavigationDevice::Automatic;
        interaction.setView({}, 1.0f);
        interaction.scroll(cursor, {2, 3}, preferences, 1, false, 11);
        EXPECT_GT(interaction.zoom(), 1.0f);
        interaction.setView({}, 1.0f);
        interaction.setDpRatio(2.0f);
        interaction.scroll(cursor, {2, 3}, preferences, 2, false, 11);
        EXPECT_EQ(interaction.pan(), (CanvasPoint{-40, 60}));
        EXPECT_EQ(interaction.zoom(), 1.0f);
        preferences.device = NavigationDevice::Trackpad;
        preferences.swipe_speed = 75;
        interaction.setView({}, 1.0f);
        interaction.scroll(cursor, {2, 3}, preferences, 0, false, 11);
        EXPECT_EQ(interaction.pan(), (CanvasPoint{-80, 120}));
        interaction.setView({}, 1.0f);
        interaction.scroll(cursor, {0, 2}, preferences, 0, true, 11);
        EXPECT_NEAR(interaction.zoom(), std::exp(0.1f), 1e-6f);
        interaction.setView({}, 1.0f);
        interaction.pinch(cursor, 1.1f, 75);
        EXPECT_NEAR(interaction.zoom(), std::pow(1.1f, 4.0f), 1e-6f);
        EXPECT_NEAR(interaction.screenToGraph(cursor).x, anchor.x, 1e-5f);
    }

    TEST(NodeCanvasInteraction, OccludedSocketsDoNotWinHitTestsAndSelectedCardsComeForward) {
        auto interaction = graph();
        auto nodes = interaction.nodes();
        nodes.push_back({.id = "Front", .bounds = {70, 10, 100, 80}});
        interaction.setGraph(nodes, {});
        const auto selected = interaction.pointerDown({100, 40}, CanvasPointerButton::Left);
        ASSERT_EQ(selected.size(), 1u);
        EXPECT_EQ(selected.front().nodes.front(), "Front");
        EXPECT_FALSE(interaction.draggingWire());
        (void)interaction.pointerUp({100, 40});
        (void)interaction.pointerDown({20, 20}, CanvasPointerButton::Left);
        EXPECT_EQ(interaction.nodes().back().id, "A");
        (void)interaction.pointerUp({20, 20});
        (void)interaction.pointerDown({100, 40}, CanvasPointerButton::Left);
        EXPECT_TRUE(interaction.draggingWire());
        interaction.cancel();
        interaction.setSelectedNodes({"Front"});
        EXPECT_EQ(interaction.nodes().back().id, "Front");
    }

    TEST(NodeCanvasInteraction, ArrangeUsesLongestPathAndRealSizesWithoutOverlap) {
        NodeCanvasInteraction interaction;
        std::vector<CanvasNode> nodes;
        for (const auto* id : {"Input", "A", "B", "Helper", "Output"})
            nodes.push_back({.id = id, .bounds = {0, 0, 224, std::string_view(id) == "A" ? 350.0f : 120.0f}, .sockets = {output(id, std::string_view(id) == "Helper" ? "lfs.float" : "lfs.geometry", {224, 30})}});
        const auto link = [](const char* from, const char* to) {
            return CanvasLink{output(from, "lfs.geometry", {}), input(to, "lfs.geometry", {})};
        };
        interaction.setGraph(nodes, {link("Input", "A"), link("A", "B"), link("Input", "B"), link("Helper", "B"), link("B", "Output")});
        const auto positions = interaction.arrangedPositions("Input", "Output");
        EXPECT_LT(positions.at("Input").x, positions.at("A").x);
        EXPECT_LT(positions.at("A").x, positions.at("B").x);
        EXPECT_EQ(positions.at("Helper").x, positions.at("A").x);
        EXPECT_LT(positions.at("B").x, positions.at("Output").x);
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            auto a = nodes[i].bounds;
            a.x = positions.at(nodes[i].id).x;
            a.y = positions.at(nodes[i].id).y;
            for (std::size_t j = i + 1; j < nodes.size(); ++j) {
                auto b = nodes[j].bounds;
                b.x = positions.at(nodes[j].id).x;
                b.y = positions.at(nodes[j].id).y;
                EXPECT_FALSE(a.intersects(b));
            }
        }
    }

    TEST(NodeCanvasInteraction, RefusesIncompatibleSocket) {
        auto interaction = graph();
        (void)interaction.pointerDown({100, 40}, CanvasPointerButton::Left);
        (void)interaction.pointerMove({260, 200});
        EXPECT_FALSE(interaction.snappedSocket());
        EXPECT_TRUE(interaction.pointerUp({260, 200}).empty());
    }

    TEST(NodeCanvasInteraction, ArrangeSelectionPreservesCentreAndOtherNodes) {
        auto interaction = graph();
        const auto before = interaction.nodes();
        const auto positions = interaction.arrangedPositions("A", "C", {"A", "B"});
        ASSERT_EQ(positions.size(), 2u);
        EXPECT_FALSE(positions.contains("C"));
        EXPECT_EQ(interaction.nodes(), before);
        const float low_x = std::min(positions.at("A").x, positions.at("B").x);
        const float high_x = std::max(positions.at("A").x + 100, positions.at("B").x + 100);
        const float low_y = std::min(positions.at("A").y, positions.at("B").y);
        const float high_y = std::max(positions.at("A").y + 80, positions.at("B").y + 80);
        EXPECT_FLOAT_EQ((low_x + high_x) * 0.5f, 180);
        EXPECT_FLOAT_EQ((low_y + high_y) * 0.5f, 40);
        const auto single = interaction.arrangedPositions("A", "C", {"B"});
        EXPECT_EQ(single.at("B"), (CanvasPoint{260, 0}));
    }

    void expectVisiblePaths(const NodeCanvasInteraction& interaction) {
        const auto& paths = interaction.wirePaths();
        ASSERT_EQ(paths.size(), interaction.links().size());
        for (std::size_t i = 0; i < paths.size(); ++i) {
            const auto& path = paths[i];
            ASSERT_GE(path.size(), 2u);
            EXPECT_EQ(path.front(), interaction.links()[i].from.position);
            EXPECT_EQ(path.back(), interaction.links()[i].to.position);
            for (std::size_t j = 1; j < path.size(); ++j) {
                const auto delta = path[j] - path[j - 1];
                const int steps = static_cast<int>(std::hypot(delta.x, delta.y)) + 1;
                for (int step = 0; step <= steps; ++step) {
                    const auto p = path[j - 1] + delta * (static_cast<float>(step) / steps);
                    for (const auto& node : interaction.nodes()) {
                        const auto& r = node.bounds;
                        ASSERT_FALSE(p.x > r.x + 0.01f && p.x < r.x + r.width - 0.01f &&
                                     p.y > r.y + 0.01f && p.y < r.y + r.height - 0.01f)
                            << "Link " << i << " crosses " << node.id << " at " << p.x << "," << p.y;
                    }
                }
            }
        }
    }

    TEST(NodeCanvasInteraction, ForwardBackwardAndParallelWiresAvoidCards) {
        NodeCanvasInteraction interaction;
        const CanvasLink forward{output("A", "lfs.float", {100, 40}), input("B", "lfs.float", {600, 40})};
        const CanvasLink backward{output("B", "lfs.float", {700, 60}), input("A", "lfs.float", {0, 60})};
        interaction.setGraph({{.id = "A", .bounds = {0, 0, 100, 100}},
                              {.id = "B", .bounds = {600, 0, 100, 100}},
                              {.id = "Obstacle", .bounds = {220, -30, 240, 180}}},
                             {forward, forward, backward});
        expectVisiblePaths(interaction);
        EXPECT_NE(interaction.wirePaths()[0], interaction.wirePaths()[1]);
        const auto generation = interaction.routeGeneration();
        const auto routes = interaction.wirePaths();
        interaction.setView({300, 200}, 0.5f);
        interaction.setSelectedNodes({"A"});
        EXPECT_EQ(interaction.routeGeneration(), generation);
        EXPECT_EQ(interaction.wirePaths(), routes);
        auto moved = interaction.nodes();
        std::ranges::find(moved, "Obstacle", &CanvasNode::id)->bounds.y = 300;
        interaction.setGraph(std::move(moved), interaction.links());
        EXPECT_GT(interaction.routeGeneration(), generation);
        expectVisiblePaths(interaction);
    }

    TEST(NodeCanvasInteraction, DetouredWireHitAndKnifeUseVisiblePath) {
        NodeCanvasInteraction interaction;
        const CanvasLink link{output("A", "lfs.float", {100, 40}), input("B", "lfs.float", {600, 40})};
        interaction.setGraph({{.id = "A", .bounds = {0, 0, 100, 100}},
                              {.id = "B", .bounds = {600, 0, 100, 100}},
                              {.id = "Obstacle", .bounds = {220, -30, 240, 180}}},
                             {link});
        const auto& path = interaction.wirePaths()[0];
        const auto corner = *std::ranges::min_element(path, {}, &CanvasPoint::y);
        const CanvasPoint point{340, corner.y};
        const auto selected = interaction.pointerDown(point, CanvasPointerButton::Left);
        ASSERT_EQ(selected.size(), 1u);
        EXPECT_EQ(selected[0].kind, CanvasCommandKind::SelectLink);
        (void)interaction.pointerUp(point);
        (void)interaction.pointerDown(point + CanvasPoint{0, -10}, CanvasPointerButton::Right, {.control = true});
        const auto commands = interaction.pointerUp(point + CanvasPoint{0, 10});
        ASSERT_EQ(commands.size(), 1u);
        EXPECT_EQ(commands[0].kind, CanvasCommandKind::DeleteLinks);
    }

    TEST(NodeCanvasInteraction, KnifeDoesNotCutDisjointCollinearLane) {
        auto interaction = graph();
        const CanvasLink link{output("A", "lfs.float", {100, 40}), input("B", "lfs.float", {260, 40})};
        interaction.setGraph(interaction.nodes(), {link});
        (void)interaction.pointerDown({800, 40}, CanvasPointerButton::Right, {.control = true});
        EXPECT_TRUE(interaction.pointerUp({900, 40}).empty());
    }

    TEST(NodeCanvasInteraction, RoutingFindsNarrowAndMultiTurnCorridors) {
        NodeCanvasInteraction interaction;
        const CanvasLink link{output("A", "lfs.float", {100, 40}), input("B", "lfs.float", {600, 40})};
        interaction.setGraph({{.id = "A", .bounds = {0, 0, 100, 100}},
                              {.id = "B", .bounds = {600, 0, 100, 100}},
                              {.id = "Middle", .bounds = {240, -90, 200, 260}},
                              {.id = "AboveDeparture", .bounds = {90, -150, 100, 100}},
                              {.id = "BelowDeparture", .bounds = {90, 150, 100, 100}},
                              {.id = "AboveArrival", .bounds = {510, -150, 100, 100}},
                              {.id = "BelowArrival", .bounds = {510, 150, 100, 100}}},
                             {link});
        expectVisiblePaths(interaction);
        interaction.setGraph({{.id = "A", .bounds = {0, 0, 100, 100}},
                              {.id = "B", .bounds = {600, 0, 100, 100}},
                              {.id = "Close", .bounds = {116, -30, 460, 180}}},
                             {link});
        expectVisiblePaths(interaction);
    }

    TEST(NodeCanvasInteraction, FiftyNodeRoutingCost) {
        NodeCanvasInteraction interaction;
        std::vector<CanvasNode> nodes;
        std::vector<CanvasLink> links;
        for (int i = 0; i < 50; ++i) {
            const auto name = std::to_string(i);
            const float x = (i / 5) * 300.0f;
            const float y = (i % 5) * 180.0f;
            nodes.push_back({.id = name, .bounds = {x, y, 224, 130}, .sockets = {input(name, "lfs.float", {x, y + 70}), output(name, "lfs.float", {x + 224, y + 40})}});
            if (i >= 5)
                links.push_back({nodes[i - 5].sockets[1], nodes[i].sockets[0]});
            if (i >= 10 && i % 3 == 0)
                links.push_back({nodes[i - 10].sockets[1], nodes[i].sockets[0]});
        }
        interaction.setGraph(nodes, links);
        expectVisiblePaths(interaction);
        double maximum = 0;
        double sum = 0;
        for (int frame = 0; frame < 60; ++frame) {
            nodes[17].bounds.y += 0.2f;
            const auto start = std::chrono::steady_clock::now();
            interaction.setGraph(nodes, links);
            (void)interaction.wirePaths();
            const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            maximum = std::max(maximum, ms);
            sum += ms;
        }
        std::cout << "50-node routing: mean " << sum / 60 << " ms, max " << maximum << " ms\n";
        EXPECT_LT(maximum, 20.0); // Regression ceiling, not the live 2 ms frame target.
    }

    TEST(NodeCanvasInteraction, ReRoutesConnectedInput) {
        auto interaction = graph();
        const CanvasLink original{output("A", "lfs.float", {100, 40}),
                                  input("B", "lfs.float", {260, 40})};
        interaction.setGraph(
            {{.id = "A", .bounds = {0, 0, 100, 80}, .sockets = {original.from}},
             {.id = "B", .bounds = {260, 0, 100, 80}, .sockets = {original.to}},
             {.id = "D", .bounds = {260, 160, 100, 80}, .sockets = {input("D", "lfs.float", {260, 200})}}},
            {original});
        interaction.pointerDown({260, 40}, CanvasPointerButton::Left);
        interaction.pointerMove({260, 200});
        const auto commands = interaction.pointerUp({260, 200});
        ASSERT_EQ(commands.size(), 1u);
        EXPECT_EQ(commands[0].kind, CanvasCommandKind::ReRoute);
        EXPECT_EQ(commands[0].before, original);
        EXPECT_EQ(commands[0].after->to.node, "D");
    }

    TEST(NodeCanvasInteraction, EmptyDropDisconnectsAndEscapeCancels) {
        auto interaction = graph();
        const CanvasLink original{output("A", "lfs.float", {100, 40}),
                                  input("B", "lfs.float", {260, 40})};
        interaction.setGraph(
            {{.id = "A", .bounds = {0, 0, 100, 80}, .sockets = {original.from}},
             {.id = "B", .bounds = {260, 0, 100, 80}, .sockets = {original.to}}},
            {original});
        interaction.pointerDown({260, 40}, CanvasPointerButton::Left);
        interaction.cancel();
        EXPECT_FALSE(interaction.active());
        interaction.pointerDown({260, 40}, CanvasPointerButton::Left);
        const auto commands = interaction.pointerUp({500, 500});
        ASSERT_EQ(commands.size(), 1u);
        EXPECT_EQ(commands[0].kind, CanvasCommandKind::Disconnect);
        EXPECT_EQ(commands[0].before, original);
    }

    TEST(NodeCanvasInteraction, KnifeCutsCrossedLink) {
        auto interaction = graph();
        const CanvasLink link{output("A", "lfs.float", {100, 40}),
                              input("B", "lfs.float", {260, 40})};
        interaction.setGraph(
            {{.id = "A", .bounds = {0, 0, 100, 80}, .sockets = {link.from}},
             {.id = "B", .bounds = {260, 0, 100, 80}, .sockets = {link.to}}},
            {link});
        interaction.pointerDown({180, 0}, CanvasPointerButton::Right, {.control = true});
        interaction.pointerMove({180, 80});
        const auto commands = interaction.pointerUp({180, 80});
        ASSERT_EQ(commands.size(), 1u);
        EXPECT_EQ(commands[0].kind, CanvasCommandKind::DeleteLinks);
        EXPECT_EQ(commands[0].links, std::vector<CanvasLink>{link});
    }

    TEST(NodeCanvasInteraction, MultiInputAcceptsAdditionalLinks) {
        auto interaction = graph(true);
        const CanvasLink existing{output("X", "lfs.float", {100, 120}),
                                  input("B", "lfs.float", {260, 40}, true)};
        interaction.setGraph(
            {{.id = "A", .bounds = {0, 0, 100, 80}, .sockets = {output("A", "lfs.float", {100, 40})}},
             {.id = "B", .bounds = {260, 0, 100, 80}, .sockets = {existing.to}}},
            {existing});
        interaction.pointerDown({100, 40}, CanvasPointerButton::Left);
        interaction.pointerMove({260, 40});
        const auto commands = interaction.pointerUp({260, 40});
        ASSERT_EQ(commands.size(), 1u);
        EXPECT_EQ(commands[0].kind, CanvasCommandKind::Connect);
    }

    TEST(NodeCanvasInteraction, SocketHitRadiusDoesNotScaleWithZoom) {
        auto interaction = graph();
        interaction.setView({0, 0}, 0.5f);
        interaction.pointerDown({61, 20}, CanvasPointerButton::Left);
        EXPECT_TRUE(interaction.draggingWire());
        interaction.cancel();
        interaction.setView({0, 0}, 2.5f);
        interaction.pointerDown({261, 100}, CanvasPointerButton::Left);
        EXPECT_TRUE(interaction.draggingWire());
    }

    TEST(NodeCanvasInteraction, RetinaHitAndSnapRadiiRemainZoomIndependent) {
        for (const float zoom : {0.5f, 1.0f, 2.5f}) {
            auto interaction = graph();
            interaction.setDpRatio(2.0f);
            interaction.setView({0, 0}, zoom);
            (void)interaction.pointerDown({100 * zoom + 23, 40 * zoom}, CanvasPointerButton::Left);
            ASSERT_TRUE(interaction.draggingWire());
            (void)interaction.pointerMove({260 * zoom - 39, 40 * zoom});
            ASSERT_TRUE(interaction.snappedSocket());
            EXPECT_EQ(interaction.snappedSocket()->node, "B");
            interaction.cancel();
            (void)interaction.pointerDown({100 * zoom + 25, 40 * zoom}, CanvasPointerButton::Left);
            EXPECT_FALSE(interaction.draggingWire());
        }
    }

    TEST(NodeCanvasInteraction, MovingNodeUpdatesWiresAndCancelRestoresBoth) {
        auto interaction = graph();
        const CanvasLink link{output("A", "lfs.float", {100, 40}), input("B", "lfs.float", {260, 40})};
        interaction.setGraph(interaction.nodes(), {link});
        (void)interaction.pointerDown({50, 20}, CanvasPointerButton::Left);
        (void)interaction.pointerMove({80, 50});
        EXPECT_EQ(interaction.links().front().from.position, (CanvasPoint{130, 70}));
        EXPECT_EQ(interaction.links().front().to.position, link.to.position);
        interaction.cancel();
        EXPECT_EQ(interaction.links().front(), link);
        EXPECT_EQ(std::ranges::find(interaction.nodes(), "A", &CanvasNode::id)->bounds.x, 0);
    }

    TEST(NodeCanvasInteraction, ReboundViewportPanUsesSameBindingTable) {
        lfs::vis::input::InputBindings bindings;
        using namespace lfs::vis::input;
        bindings.setBinding(ToolMode::GLOBAL, Action::CAMERA_PAN, MouseDragTrigger{MouseButton::LEFT, MODIFIER_ALT});
        auto interaction = graph();
        const bool pan = bindings.getActionForDrag(ToolMode::GLOBAL, MouseButton::LEFT, MODIFIER_ALT) == Action::CAMERA_PAN;
        ASSERT_TRUE(pan);
        EXPECT_NE(bindings.getActionForDrag(ToolMode::GLOBAL, MouseButton::RIGHT, MODIFIER_NONE), Action::CAMERA_PAN);
        const auto before = interaction.nodes();
        EXPECT_TRUE(interaction.pointerDown({50, 20}, CanvasPointerButton::Left, {.alt = true}, pan).empty());
        EXPECT_TRUE(interaction.pointerMove({150, 70}).empty());
        EXPECT_EQ(interaction.pan(), (CanvasPoint{100, 50}));
        EXPECT_EQ(interaction.nodes(), before);
        EXPECT_TRUE(interaction.pointerUp({150, 70}).empty());
    }

    TEST(NodeCanvasInteraction, SplicesUnlinkedNodeAcrossCompatibleLink) {
        NodeCanvasInteraction interaction;
        const CanvasLink link{output("A", "lfs.float", {100, 40}),
                              input("B", "lfs.float", {300, 40})};
        interaction.setGraph(
            {{.id = "A", .bounds = {0, 0, 100, 80}, .sockets = {link.from}},
             {.id = "B", .bounds = {300, 0, 100, 80}, .sockets = {link.to}},
             {.id = "C", .bounds = {150, 120, 100, 80}, .sockets = {input("C", "lfs.float", {150, 160}), output("C", "lfs.float", {250, 160})}}},
            {link});
        interaction.pointerDown({200, 150}, CanvasPointerButton::Left);
        interaction.pointerMove({200, 40});
        const auto commands = interaction.pointerUp({200, 40});
        ASSERT_EQ(commands.size(), 1u);
        EXPECT_EQ(commands[0].kind, CanvasCommandKind::Splice);
        EXPECT_EQ(commands[0].nodes, std::vector<std::string>{"C"});
    }
} // namespace
