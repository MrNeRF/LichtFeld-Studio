/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "builtin_common.hpp"
namespace lfs::nodes::builtin {

    void evaluate_math(NodeContext& context) {
        const auto op = property_string(context, "operation", "add");
        context.set_output("Value",
                           operation(FLOAT_SOCKET,
                                     {context.field("A", FLOAT_SOCKET), context.field("B", FLOAT_SOCKET)},
                                     [op](const std::vector<Tensor>& values) {
                                         const auto& a = values[0];
                                         const auto& b = values[1];
                                         if (op == "absolute")
                                             return a.abs();
                                         if (op == "sqrt")
                                             return a.maximum(0).sqrt();
                                         if (op == "floor")
                                             return a.floor();
                                         if (op == "fraction")
                                             return a - a.floor();
                                         if (op == "sine")
                                             return a.sin();
                                         if (op == "cosine")
                                             return a.cos();
                                         if (op == "subtract")
                                             return a - b;
                                         if (op == "multiply")
                                             return a * b;
                                         if (op == "divide")
                                             return safe_divide(a, b);
                                         if (op == "power")
                                             return a.pow(b);
                                         if (op == "minimum")
                                             return a.minimum(b);
                                         if (op == "maximum")
                                             return a.maximum(b);
                                         if (op == "greater_than")
                                             return a.gt(b).to(DataType::Float32);
                                         if (op == "less_than")
                                             return a.lt(b).to(DataType::Float32);
                                         if (op == "clamp")
                                             return a.maximum(0).minimum(b);
                                         return a + b;
                                     }));
    }

    void evaluate_vector_math(NodeContext& context) {
        const auto op = property_string(context, "operation", "add");
        const bool scalar = op == "length" || op == "distance" || op == "dot";
        context.set_output(scalar ? "Vector" : "Value", scalar ? Value(glm::vec3(0)) : Value(0.0f));
        context.set_output(scalar ? "Value" : "Vector",
                           operation(scalar ? FLOAT_SOCKET : VECTOR_SOCKET,
                                     {context.field("A", VECTOR_SOCKET), context.field("B", VECTOR_SOCKET),
                                      context.field("Scale", FLOAT_SOCKET)},
                                     [op](const std::vector<Tensor>& values) {
                                         const auto& a = values[0];
                                         const auto& b = values[1];
                                         if (op == "length")
                                             return (a * a).sum(1).sqrt();
                                         if (op == "distance")
                                             return ((a - b) * (a - b)).sum(1).sqrt();
                                         if (op == "dot")
                                             return (a * b).sum(1);
                                         if (op == "normalise")
                                             return safe_divide(a, (a * a).sum(1, true).sqrt());
                                         if (op == "scale")
                                             return a * values[2].unsqueeze(1);
                                         if (op == "subtract")
                                             return a - b;
                                         if (op == "multiply")
                                             return a * b;
                                         return a + b;
                                     }));
    }

    void evaluate_compare(NodeContext& context) {
        const auto op = property_string(context, "operation", "equal");
        context.set_output("Result",
                           operation(BOOL_SOCKET,
                                     {context.field("A", FLOAT_SOCKET), context.field("B", FLOAT_SOCKET),
                                      context.field("Epsilon", FLOAT_SOCKET)},
                                     [op](const std::vector<Tensor>& values) {
                                         const auto& a = values[0];
                                         const auto& b = values[1];
                                         if (op == "less_than")
                                             return a.lt(b);
                                         if (op == "less_equal")
                                             return a.le(b);
                                         if (op == "greater_than")
                                             return a.gt(b);
                                         if (op == "greater_equal")
                                             return a.ge(b);
                                         if (op == "not_equal")
                                             return (a - b).abs().gt(values[2]);
                                         return (a - b).abs().le(values[2]);
                                     }));
    }

    void evaluate_boolean_math(NodeContext& context) {
        const auto op = property_string(context, "operation", "and");
        context.set_output("Result",
                           operation(BOOL_SOCKET,
                                     {context.field("A", BOOL_SOCKET), context.field("B", BOOL_SOCKET)},
                                     [op](const std::vector<Tensor>& values) {
                                         if (op == "not")
                                             return values[0].logical_not();
                                         if (op == "or")
                                             return values[0].logical_or(values[1]);
                                         if (op == "xor")
                                             return values[0].logical_xor(values[1]);
                                         return values[0].logical_and(values[1]);
                                     }));
    }

    void evaluate_combine_xyz(NodeContext& context, bool colour) {
        const bool hsv = colour && property_string(context, "mode", "rgb") == "hsv";
        context.set_output(colour ? "Colour" : "Vector",
                           operation(colour ? COLOUR_SOCKET : VECTOR_SOCKET,
                                     {context.field(colour ? "R" : "X", FLOAT_SOCKET),
                                      context.field(colour ? "G" : "Y", FLOAT_SOCKET),
                                      context.field(colour ? "B" : "Z", FLOAT_SOCKET)},
                                     [hsv](const std::vector<Tensor>& values) {
                                         auto result = Tensor::stack(values, 1);
                                         return hsv ? hsv_to_rgb(result) : result;
                                     }));
    }

    void evaluate_separate(NodeContext& context, bool colour) {
        const bool hsv = colour && property_string(context, "mode", "rgb") == "hsv";
        auto source = context.field(colour ? "Colour" : "Vector", colour ? COLOUR_SOCKET : VECTOR_SOCKET);
        if (hsv)
            source = operation(COLOUR_SOCKET, {source}, [](const std::vector<Tensor>& values) {
                return rgb_to_hsv(values[0]);
            });
        const std::array<std::string, 3> names =
            colour ? std::array<std::string, 3>{"R", "G", "B"} : std::array<std::string, 3>{"X", "Y", "Z"};
        for (int axis = 0; axis < 3; ++axis)
            context.set_output(names[axis],
                               operation(FLOAT_SOCKET, {source}, [axis](const std::vector<Tensor>& values) {
                                   return channel(values[0], axis);
                               }));
    }

    void evaluate_map_range(NodeContext& context) {
        const bool clamp = property_bool(context, "clamp", true);
        context.set_output(
            "Result",
            operation(FLOAT_SOCKET,
                      {context.field("Value", FLOAT_SOCKET), context.field("From Min", FLOAT_SOCKET),
                       context.field("From Max", FLOAT_SOCKET), context.field("To Min", FLOAT_SOCKET),
                       context.field("To Max", FLOAT_SOCKET)},
                      [clamp](const std::vector<Tensor>& values) {
                          auto t = safe_divide(values[0] - values[1], values[2] - values[1]);
                          if (clamp)
                              t = t.clamp(0, 1);
                          return values[3] + t * (values[4] - values[3]);
                      }));
    }

    void evaluate_mix_colour(NodeContext& context) {
        const auto mode = property_string(context, "mode", "mix");
        context.set_output("Colour",
                           operation(COLOUR_SOCKET,
                                     {context.field("A", COLOUR_SOCKET), context.field("B", COLOUR_SOCKET),
                                      context.field("Factor", FLOAT_SOCKET)},
                                     [mode](const std::vector<Tensor>& values) {
                                         const auto& a = values[0];
                                         const auto& b = values[1];
                                         Tensor mixed = b;
                                         if (mode == "multiply")
                                             mixed = a * b;
                                         if (mode == "add")
                                             mixed = a + b;
                                         if (mode == "subtract")
                                             mixed = a - b;
                                         return blend(a, mixed, values[2].clamp(0, 1));
                                     }));
    }
    void register_utilities(NodeTypeRegistry& registry) {
        const auto f = std::string(FLOAT_SOCKET);
        const auto b = std::string(BOOL_SOCKET);
        const auto v = std::string(VECTOR_SOCKET);
        const auto c = std::string(COLOUR_SOCKET);
        register_type(registry,
                      type("lfs.math", "Math", "Utilities",
                           "Apply a scalar arithmetic operation to two fields.",
                           {in("A", f, 0.0f, true).step_size(0.01),
                            in("B", f, 0.0f, true).step_size(0.01)},
                           {out("Value", f)}, evaluate_math,
                           {prop("operation", PropertyKind::Enum, "add",
                                 {"add", "subtract", "multiply", "divide", "power", "minimum", "maximum",
                                  "absolute", "sqrt", "floor", "fraction", "sine", "cosine", "greater_than",
                                  "less_than", "clamp"})}));
        register_type(
            registry,
            type("lfs.vector_math", "Vector Math", "Utilities",
                 "Apply a vector arithmetic operation to vector fields.",
                 {in("A", v, glm::vec3(0), true).step_size(0.01),
                  in("B", v, glm::vec3(0), true).step_size(0.01),
                  in("Scale", f, 1.0f, true).step_size(0.01)},
                 {out("Vector", v), out("Value", f)}, evaluate_vector_math,
                 {prop("operation", PropertyKind::Enum, "add",
                       {"add", "subtract", "multiply", "scale", "length", "distance", "dot", "normalise"})}));
        register_type(
            registry,
            type("lfs.compare", "Compare", "Utilities",
                 "Compare scalar fields using an optional equality tolerance.",
                 {in("A", f, 0.0f, true).step_size(0.01),
                  in("B", f, 0.0f, true).step_size(0.01),
                  in("Epsilon", f, 0.001f, true).minimum(0).step_size(0.001)},
                 {out("Result", b)}, evaluate_compare,
                 {prop("operation", PropertyKind::Enum, "equal",
                       {"less_than", "less_equal", "greater_than", "greater_equal", "equal", "not_equal"})}));
        register_type(registry,
                      type("lfs.boolean_math", "Boolean Math", "Utilities",
                           "Combine boolean fields with a logical operation.",
                           {in("A", b, false, true), in("B", b, false, true)}, {out("Result", b)},
                           evaluate_boolean_math,
                           {prop("operation", PropertyKind::Enum, "and", {"and", "or", "not", "xor"})}));
        register_type(
            registry,
            type("lfs.map_range", "Map Range", "Utilities",
                 "Remap a scalar field between two numeric ranges.",
                 {in("Value", f, 0.0f, true).step_size(0.01),
                  in("From Min", f, 0.0f, true).step_size(0.01),
                  in("From Max", f, 1.0f, true).step_size(0.01),
                  in("To Min", f, 0.0f, true).step_size(0.01),
                  in("To Max", f, 1.0f, true).step_size(0.01)},
                 {out("Result", f)}, evaluate_map_range, {prop("clamp", PropertyKind::Bool, true)}));
        register_type(registry, type("lfs.separate_xyz", "Separate XYZ", "Utilities",
                                     "Separate a vector field into its three coordinates.",
                                     {in("Vector", v, glm::vec3(0), true).step_size(0.01)},
                                     {out("X", f), out("Y", f), out("Z", f)}, [](NodeContext& x) {
                                         evaluate_separate(x, false);
                                     }));
        register_type(registry, type("lfs.combine_xyz", "Combine XYZ", "Utilities",
                                     "Combine three scalar fields into a vector field.",
                                     {in("X", f, 0.0f, true).step_size(0.01),
                                      in("Y", f, 0.0f, true).step_size(0.01),
                                      in("Z", f, 0.0f, true).step_size(0.01)},
                                     {out("Vector", v)}, [](NodeContext& x) {
                                         evaluate_combine_xyz(x, false);
                                     }));
        register_type(registry,
                      type("lfs.separate_colour", "Separate Colour", "Utilities",
                           "Separate a colour field into RGB or HSV channels.",
                           {in("Colour", c, glm::vec3(0), true).step_size(0.01)},
                           {out("R", f), out("G", f), out("B", f)},
                           [](NodeContext& x) {
                               evaluate_separate(x, true);
                           },
                           {prop("mode", PropertyKind::Enum, "rgb", {"rgb", "hsv"})}));
        register_type(registry, type("lfs.combine_colour", "Combine Colour", "Utilities",
                                     "Combine RGB or HSV channels into a colour field.",
                                     {in("R", f, 0.0f, true).step_size(0.01),
                                      in("G", f, 0.0f, true).step_size(0.01),
                                      in("B", f, 0.0f, true).step_size(0.01)},
                                     {out("Colour", c)},
                                     [](NodeContext& x) {
                                         evaluate_combine_xyz(x, true);
                                     },
                                     {prop("mode", PropertyKind::Enum, "rgb", {"rgb", "hsv"})}));
        register_type(registry, type("lfs.mix_colour", "Mix Colour", "Utilities",
                                     "Blend two colour fields using a selected mixing operation.",
                                     {in("A", c, glm::vec3(0), true).step_size(0.01),
                                      in("B", c, glm::vec3(0), true).step_size(0.01),
                                      in("Factor", f, 0.5f, true).range(0, 1).step_size(0.01)},
                                     {out("Colour", c)}, evaluate_mix_colour,
                                     {prop("mode", PropertyKind::Enum, "mix",
                                           {"mix", "multiply", "add", "subtract"})}));
    }

} // namespace lfs::nodes::builtin
