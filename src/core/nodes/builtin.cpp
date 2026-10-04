/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "builtin_common.hpp"

namespace lfs::nodes {
    void register_builtin_nodes(NodeTypeRegistry& registry) {
        builtin::register_input(registry);
        builtin::register_utilities(registry);
        builtin::register_selection(registry);
        builtin::register_geometry(registry);
        builtin::register_splat(registry);
        builtin::register_cleanup(registry);
        builtin::register_conversion(registry);
        builtin::register_colour(registry);
        builtin::register_texture(registry);
        builtin::register_instances(registry);

        NodeTypeInfo group;
        group.id = "lfs.group";
        group.label = "Group";
        group.category = "Group";
        group.description = "Evaluates a reusable node graph.";
        group.help = "Choose a graph in the sidebar. Inputs and outputs follow that graph's interface.";
        group.properties.push_back({"tree", "Graph", PropertyKind::String, "", {}, {}, {}, "Referenced graph UUID."});
        builtin::register_type(registry, std::move(group));

        NodeTypeInfo frame;
        frame.id = "lfs.frame";
        frame.label = "Frame";
        frame.category = "Layout";
        frame.description = "Keeps related nodes visually grouped.";
        frame.help = "Drop nodes into the frame. Moving the frame moves its members.";
        frame.properties = {
            {"label", "Label", PropertyKind::String, "Frame", {}, {}, {}, "Frame heading."},
            {"colour", "Colour", PropertyKind::String, "neutral", {}, {}, {}, "Theme colour preset or RGB value."},
            {"note", "Note", PropertyKind::String, "", {}, {}, {}, "Optional text shown under the heading."},
        };
        builtin::register_type(registry, std::move(frame));

        NodeTypeInfo reroute;
        reroute.id = "lfs.reroute";
        reroute.label = "Reroute";
        reroute.category = "Layout";
        reroute.description = "Redirects a link without changing its value.";
        reroute.help = "Connect any socket type; the reroute adopts its upstream type.";
        reroute.inputs.push_back({.identifier = "Input", .label = "", .type = std::string(ANY_SOCKET), .default_value = {}, .description = "Value to pass through."});
        reroute.outputs.push_back({.identifier = "Output", .label = "", .type = std::string(ANY_SOCKET), .default_value = {}, .description = "Unchanged input value."});
        builtin::register_type(registry, std::move(reroute));

        NodeTypeInfo note;
        note.id = "lfs.note";
        note.label = "Note";
        note.category = "Layout";
        note.description = "Adds an editable text card to the graph.";
        note.help = "Double-click the card or edit its text in the sidebar.";
        note.properties.push_back({"text", "Text", PropertyKind::String, "Note", {}, {}, {}, "Text displayed on the card."});
        note.properties.push_back({"width", "Width", PropertyKind::Float, 220.0f, {}, 100.0, 1200.0, "Card width in canvas units; the height fits the text."});
        builtin::register_type(registry, std::move(note));
    }
} // namespace lfs::nodes
