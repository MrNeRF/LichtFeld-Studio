/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/node_canvas_interaction.hpp"
#include "input/input_bindings.hpp"

#include <gtest/gtest.h>

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
