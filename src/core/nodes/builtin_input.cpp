/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "builtin_common.hpp"
#include "core/base64.hpp"
#include "core/tensor_backend.hpp"
#include <cstring>
#include <mutex>

namespace lfs::nodes {
    void set_stored_selection(Node& node, const core::Tensor& selection) {
        if (selection.dtype() != core::DataType::Bool && selection.dtype() != core::DataType::UInt8)
            throw NodeError("Stored Selection requires a Bool or UInt8 tensor");
        // Capturing a user selection is a one-time serialization boundary.
        const auto cpu = selection.cpu().contiguous();
        const auto* values = cpu.ptr<std::uint8_t>();
        std::vector<std::uint8_t> packed((cpu.numel() + 7) / 8, 0);
        std::size_t selected_count = 0;
        for (std::size_t index = 0; index < cpu.numel(); ++index)
            if (values[index]) {
                packed[index / 8] |= static_cast<std::uint8_t>(1U << (index % 8));
                ++selected_count;
            }
        node.properties["data"] = core::base64_encode(packed);
        node.properties["size"] = cpu.numel();
        node.properties["selected_count"] = selected_count;
    }

    Field stored_selection_field(const Node& node) {
        using namespace builtin;
        const auto packed = core::base64_decode(node.properties.value("data", ""));
        const std::size_t size = node.properties.value("size", std::size_t{0});
        if (packed.size() != (size + 7) / 8)
            throw NodeError("Stored Selection bitmask length does not match its element count");
        const bool invert = node.properties.value("invert", false);
        // Decode once, then retain the uploaded tensor across evaluations of this field.
        auto decoded = Tensor::empty({size}, Device::CPU, DataType::Bool);
        auto* values = decoded.ptr<std::uint8_t>();
        for (std::size_t index = 0; index < size; ++index)
            values[index] = ((packed[index / 8] >> (index % 8)) & 1U) != 0;
        struct Cache {
            Tensor device_values;
            std::mutex mutex;
        };
        const auto cache = std::make_shared<Cache>();
        return Field(std::string(BOOL_SOCKET), [decoded, cache, invert, size](const FieldContext& context,
                                                                              FieldMemo& memo) {
            std::lock_guard lock(cache->mutex);
            const auto positions = position_field().evaluate(context, memo);
            if (!cache->device_values.is_valid() || cache->device_values.device() != context.device() ||
                core::gpu_backend_of(cache->device_values) != core::gpu_backend_of(positions))
                cache->device_values = decoded.to(context.device());
            const auto count = context.size();
            Tensor result;
            if (count == size)
                result = cache->device_values;
            else if (count < size)
                result = cache->device_values.slice(0, 0, count);
            else
                result = Tensor::cat(
                    {cache->device_values, Tensor::full_bool({count - size}, false, context.device())}, 0);
            return invert ? result.logical_not() : result;
        });
    }
} // namespace lfs::nodes

namespace lfs::nodes::builtin {
    std::string named_type(std::string_view identifier) {
        return "lfs." + std::string(identifier);
    }

    void evaluate_random(NodeContext& context) {
        const float seed = static_cast<float>(property_int(context, "seed"));
        context.set_output("Value",
                           operation(FLOAT_SOCKET,
                                     {convert_field(index_field(), FLOAT_SOCKET),
                                      context.field("Min", FLOAT_SOCKET), context.field("Max", FLOAT_SOCKET)},
                                     [seed](const std::vector<Tensor>& values) {
                                         // Float hashing is repeatable per backend, not bit-identical across
                                         // backends' sine implementations.
                                         const auto hash =
                                             (values[0] * 12.9898f + seed * 78.233f).sin() * 43758.5453f;
                                         const auto unit = hash - hash.floor();
                                         return values[1] + unit * (values[2] - values[1]);
                                     }));
    }
    void register_input(NodeTypeRegistry& registry) {
        const auto geo = std::string(GEOMETRY_SOCKET);
        const auto f = std::string(FLOAT_SOCKET);
        const auto i = std::string(INT_SOCKET);
        const auto b = std::string(BOOL_SOCKET);
        const auto v = std::string(VECTOR_SOCKET);
        const auto c = std::string(COLOUR_SOCKET);
        const auto s = std::string(STRING_SOCKET);
        register_type(registry, type("lfs.group_input", "Group Input", "Input",
                                     "Expose the geometry and interface values supplied to this node graph.",
                                     {}, {out("Geometry", geo)}, {}));
        register_type(registry, type("lfs.group_output", "Group Output", "Output",
                                     "Return the evaluated geometry from this node graph.",
                                     {in("Geometry", geo)}, {out("Geometry", geo)}, [](NodeContext& x) {
                                         x.set_output("Geometry", x.input("Geometry"));
                                     }));
        register_type(registry,
                      type("lfs.value", "Value", "Input", "Provide a constant floating-point value.",
                           {in("Value", f, 0.0f).step_size(0.01)}, {out("Value", f)}, [](NodeContext& x) {
                               x.set_output("Value", x.input("Value"));
                           }));
        register_type(registry,
                      type("lfs.integer", "Integer", "Input", "Provide a constant integer value.",
                           {in("Value", i, std::int64_t(0)).step_size(1)}, {out("Value", i)}, [](NodeContext& x) {
                               x.set_output("Value", x.input("Value"));
                           }));
        register_type(registry, type("lfs.boolean", "Boolean", "Input", "Provide a constant boolean value.",
                                     {in("Value", b, false)}, {out("Value", b)}, [](NodeContext& x) {
                                         x.set_output("Value", x.input("Value"));
                                     }));
        register_type(registry,
                      type("lfs.vector", "Vector", "Input", "Provide a constant three-dimensional vector.",
                           {in("Vector", v, glm::vec3(0)).step_size(0.01)}, {out("Vector", v)}, [](NodeContext& x) {
                               x.set_output("Vector", x.input("Vector"));
                           }));
        register_type(registry,
                      type("lfs.colour", "Colour", "Input", "Provide a constant linear RGB colour.",
                           {in("Colour", c, glm::vec3(0.5f)).step_size(0.01)}, {out("Colour", c)}, [](NodeContext& x) {
                               x.set_output("Colour", x.input("Colour"));
                           }));
        register_type(registry, type("lfs.position", "Position", "Input",
                                     "Read each element's position in its geometry domain.", {},
                                     {out("Position", v)}, [](NodeContext& x) {
                                         x.set_output("Position", position_field());
                                     }));
        register_type(registry, type("lfs.colour_attribute", "Colour Attribute", "Input",
                                     "Read each element's linear base colour.", {}, {out("Colour", c)},
                                     [](NodeContext& x) {
                                         x.set_output("Colour", colour_field());
                                     }));
        register_type(registry,
                      type("lfs.opacity", "Opacity", "Input", "Read each splat's activated opacity.", {},
                           {out("Opacity", f)}, [](NodeContext& x) {
                               x.set_output("Opacity", opacity_field());
                           }));
        register_type(registry, type("lfs.scale", "Scale", "Input",
                                     "Read each splat's activated scale along its three axes.", {},
                                     {out("Scale", v)}, [](NodeContext& x) {
                                         x.set_output("Scale", scale_field());
                                     }));
        register_type(registry, type("lfs.index", "Index", "Input",
                                     "Read each element's zero-based index in its geometry domain.", {},
                                     {out("Index", i)}, [](NodeContext& x) {
                                         x.set_output("Index", index_field());
                                     }));
        register_type(
            registry,
            type("lfs.named_attribute", "Named Attribute", "Input",
                 "Read a named attribute from the current geometry domain.", {in("Name", s, std::string{})},
                 {out("Attribute", f)},
                 [](NodeContext& x) {
                     std::string name;
                     if (auto* p = x.input("Name").get_if<std::string>())
                         name = *p;
                     x.set_output("Attribute", named_attribute_field(
                                                   name, named_type(property_string(x, "type", "float"))));
                 },
                 {prop("type", PropertyKind::Enum, "float", {"float", "int", "bool", "vector", "colour"})}));
        register_type(registry, type("lfs.random_value", "Random Value", "Input",
                                     "Generate repeatable per-index random values within a range.",
                                     {in("Min", f, 0.0f, true).step_size(0.01),
                                      in("Max", f, 1.0f, true).step_size(0.01)},
                                     {out("Value", f)},
                                     evaluate_random, {prop("seed", PropertyKind::Int, 0)}));
        register_type(registry,
                      type("lfs.stored_selection", "Stored Selection", "Input",
                           "Read a captured selection bitmask, optionally inverted.", {},
                           {out("Selection", b)},
                           [](NodeContext& x) {
                               x.set_output("Selection", stored_selection_field(x.node()));
                           },
                           {prop("data", PropertyKind::Data, ""), prop("size", PropertyKind::Int, 0),
                            prop("invert", PropertyKind::Bool, false)}));
    }

} // namespace lfs::nodes::builtin
