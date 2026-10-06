/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "builtin_common.hpp"
#include "builtin_curve.hpp"
#include "core/tensor_spatial.hpp"
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
    void evaluate_float_curve(NodeContext& context) {
        const auto points = curve_points(context, "points");
        const bool clamp = property_bool(context, "clamp", true);
        context.set_output("Value", operation(FLOAT_SOCKET, {context.field("Value", FLOAT_SOCKET)},
                                              [points, clamp](const std::vector<Tensor>& values) {
                                                  return evaluate_curve(values[0], points, clamp);
                                              }));
    }

    static Tensor normalized(const Tensor& vector) {
        return safe_divide(vector, (vector * vector).sum(1, true).sqrt());
    }

    void evaluate_distance(NodeContext& context) {
        const auto mode = property_string(context, "mode", "point");
        context.set_output("Distance", operation(FLOAT_SOCKET,
                                                 {context.field("Vector", VECTOR_SOCKET), context.field("Point", VECTOR_SOCKET),
                                                  context.field("Direction", VECTOR_SOCKET), context.field("Normal", VECTOR_SOCKET)},
                                                 [mode](const auto& values) {
                                                     const auto delta = values[0] - values[1];
                                                     if (mode == "plane")
                                                         return (delta * normalized(values[3])).sum(1);
                                                     if (mode == "line") {
                                                         const auto direction = normalized(values[2]);
                                                         const auto perpendicular = delta - direction * (delta * direction).sum(1, true);
                                                         return (perpendicular * perpendicular).sum(1).sqrt();
                                                     }
                                                     return (delta * delta).sum(1).sqrt();
                                                 }));
    }

    void evaluate_geometry_proximity(NodeContext& context) {
        const auto target = geometry_input(context, "Target");
        std::vector<Tensor> positions;
        if (target.splats)
            positions.push_back(target.splats->means);
        if (target.points)
            positions.push_back(target.points->positions);
        if (target.mesh && target.mesh->mesh)
            positions.push_back(target.mesh->mesh->vertices);
        if (positions.empty())
            throw NodeError("Geometry Proximity needs a non-empty target geometry");
        const auto targets = Tensor::cat(positions, 0);
        if (!targets.size(0))
            throw NodeError("Geometry Proximity needs a non-empty target geometry");
        const auto position = position_field();
        const auto nearest = Field(std::string(VECTOR_SOCKET), [targets, position](const FieldContext& domain, FieldMemo& memo) {
            const auto queries = memo.evaluate(position, domain);
            const auto reference = targets.to(domain.device());
            const auto indices = core::nearest_point_indices(queries, reference);
            if (indices.lt(0).count_nonzero())
                throw NodeError("Geometry Proximity requires finite query and target positions");
            return reference.index_select(0, indices);
        });
        context.set_output("Position", nearest);
        context.set_output("Distance", operation(FLOAT_SOCKET, {position, nearest}, [](const auto& values) {
                               const auto delta = values[0] - values[1];
                               return (delta * delta).sum(1).sqrt();
                           }));
    }

    void register_utilities(NodeTypeRegistry& registry) {
        const auto f = std::string(FLOAT_SOCKET);
        const auto b = std::string(BOOL_SOCKET);
        const auto v = std::string(VECTOR_SOCKET);
        const auto c = std::string(COLOUR_SOCKET);
        register_type(registry, type("lfs.geometry_proximity", "Utilities",
                                     {in("Target", std::string(GEOMETRY_SOCKET))},
                                     {out("Distance", f), out("Position", v)}, evaluate_geometry_proximity));
        register_type(registry,
                      type("lfs.math", "Utilities",
                           {in("A", f, 0.0f, true).step_size(0.01),
                            in("B", f, 0.0f, true).step_size(0.01)},
                           {out("Value", f)}, evaluate_math,
                           {prop("operation", PropertyKind::Enum, "add",
                                 {"add", "subtract", "multiply", "divide", "power", "minimum", "maximum",
                                  "absolute", "sqrt", "floor", "fraction", "sine", "cosine", "greater_than",
                                  "less_than", "clamp"})}));
        register_type(
            registry,
            type("lfs.vector_math", "Utilities",
                 {in("A", v, glm::vec3(0), true).step_size(0.01),
                  in("B", v, glm::vec3(0), true).step_size(0.01),
                  in("Scale", f, 1.0f, true).step_size(0.01)},
                 {out("Vector", v), out("Value", f)}, evaluate_vector_math,
                 {prop("operation", PropertyKind::Enum, "add",
                       {"add", "subtract", "multiply", "scale", "length", "distance", "dot", "normalise"})}));
        register_type(
            registry,
            type("lfs.compare", "Utilities",
                 {in("A", f, 0.0f, true).step_size(0.01),
                  in("B", f, 0.0f, true).step_size(0.01),
                  in("Epsilon", f, 0.001f, true).minimum(0).step_size(0.001)},
                 {out("Result", b)}, evaluate_compare,
                 {prop("operation", PropertyKind::Enum, "equal",
                       {"less_than", "less_equal", "greater_than", "greater_equal", "equal", "not_equal"})}));
        register_type(registry,
                      type("lfs.boolean_math", "Utilities",
                           {in("A", b, false, true), in("B", b, false, true)}, {out("Result", b)},
                           evaluate_boolean_math,
                           {prop("operation", PropertyKind::Enum, "and", {"and", "or", "not", "xor"})}));
        register_type(
            registry,
            type("lfs.map_range", "Utilities",
                 {in("Value", f, 0.0f, true).step_size(0.01),
                  in("From Min", f, 0.0f, true).step_size(0.01),
                  in("From Max", f, 1.0f, true).step_size(0.01),
                  in("To Min", f, 0.0f, true).step_size(0.01),
                  in("To Max", f, 1.0f, true).step_size(0.01)},
                 {out("Result", f)}, evaluate_map_range, {prop("clamp", PropertyKind::Bool, true)}));
        register_type(registry, type("lfs.separate_xyz", "Utilities",
                                     {in("Vector", v, glm::vec3(0), true).step_size(0.01)},
                                     {out("X", f), out("Y", f), out("Z", f)}, [](NodeContext& x) {
                                         evaluate_separate(x, false);
                                     }));
        register_type(registry, type("lfs.combine_xyz", "Utilities",
                                     {in("X", f, 0.0f, true).step_size(0.01),
                                      in("Y", f, 0.0f, true).step_size(0.01),
                                      in("Z", f, 0.0f, true).step_size(0.01)},
                                     {out("Vector", v)}, [](NodeContext& x) {
                                         evaluate_combine_xyz(x, false);
                                     }));
        register_type(registry,
                      type("lfs.separate_colour", "Utilities",
                           {in("Colour", c, glm::vec3(0), true).step_size(0.01)},
                           {out("R", f), out("G", f), out("B", f)},
                           [](NodeContext& x) {
                               evaluate_separate(x, true);
                           },
                           {prop("mode", PropertyKind::Enum, "rgb", {"rgb", "hsv"})}));
        register_type(registry, type("lfs.combine_colour", "Utilities",
                                     {in("R", f, 0.0f, true).step_size(0.01),
                                      in("G", f, 0.0f, true).step_size(0.01),
                                      in("B", f, 0.0f, true).step_size(0.01)},
                                     {out("Colour", c)},
                                     [](NodeContext& x) {
                                         evaluate_combine_xyz(x, true);
                                     },
                                     {prop("mode", PropertyKind::Enum, "rgb", {"rgb", "hsv"})}));
        register_type(registry, type("lfs.mix_colour", "Utilities",
                                     {in("A", c, glm::vec3(0), true).step_size(0.01),
                                      in("B", c, glm::vec3(0), true).step_size(0.01),
                                      in("Factor", f, 0.5f, true).range(0, 1).step_size(0.01)},
                                     {out("Colour", c)}, evaluate_mix_colour,
                                     {prop("mode", PropertyKind::Enum, "mix",
                                           {"mix", "multiply", "add", "subtract"})}));
        const auto curve = nlohmann::json::array({{0, 0}, {1, 1}});
        register_type(registry, type("lfs.float_curve", "Utilities",
                                     {in("Value", f, 0.0f, true).step_size(0.01)}, {out("Value", f)},
                                     evaluate_float_curve,
                                     {prop("points", PropertyKind::Data, curve), prop("clamp", PropertyKind::Bool, true)}));
        register_type(registry, type("lfs.distance", "Utilities",
                                     {in("Vector", v, glm::vec3(0), true).step_size(0.01),
                                      in("Point", v, glm::vec3(0), true).step_size(0.01),
                                      in("Direction", v, glm::vec3(1, 0, 0), true).step_size(0.01),
                                      in("Normal", v, glm::vec3(0, 0, 1), true).step_size(0.01)},
                                     {out("Distance", f)}, evaluate_distance,
                                     {prop("mode", PropertyKind::Enum, "point", {"point", "line", "plane"})}));
    }

} // namespace lfs::nodes::builtin
