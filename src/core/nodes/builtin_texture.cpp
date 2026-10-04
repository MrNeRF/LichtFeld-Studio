/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "builtin_common.hpp"
#include "core/tensor_procedural.hpp"

namespace lfs::nodes::builtin {
    namespace {
        void evaluate_gradient(NodeContext& context) {
            const auto name = property_string(context, "type", "linear");
            using core::GradientType;
            const auto mode = name == "quadratic" ? GradientType::Quadratic : name == "easing"  ? GradientType::Easing
                                                                          : name == "diagonal"  ? GradientType::Diagonal
                                                                          : name == "spherical" ? GradientType::Spherical
                                                                          : name == "radial"    ? GradientType::Radial
                                                                                                : GradientType::Linear;
            context.set_output("Fac", operation(FLOAT_SOCKET, {context.field("Vector", VECTOR_SOCKET)},
                                                [mode](const auto& values) { return core::procedural_gradient(values[0], mode); }));
        }

        void evaluate_noise(NodeContext& context) {
            const auto fac = operation(FLOAT_SOCKET,
                                       {context.field("Vector", VECTOR_SOCKET), context.field("Scale", FLOAT_SOCKET),
                                        context.field("Detail", FLOAT_SOCKET), context.field("Roughness", FLOAT_SOCKET),
                                        context.field("Distortion", FLOAT_SOCKET), context.field("Seed", FLOAT_SOCKET)},
                                       [](const auto& values) {
                                           return core::procedural_noise(values[0], values[1], values[2], values[3], values[4], values[5]);
                                       });
            context.set_output("Fac", fac);
            context.set_output("Colour", operation(COLOUR_SOCKET, {fac}, [](const auto& values) {
                                   return values[0].unsqueeze(1).expand({static_cast<int>(values[0].size(0)), 3});
                               }));
        }
    } // namespace

    void register_texture(NodeTypeRegistry& registry) {
        const auto f = std::string(FLOAT_SOCKET);
        const auto v = std::string(VECTOR_SOCKET);
        const auto c = std::string(COLOUR_SOCKET);
        register_type(registry, type("lfs.gradient_texture", "Texture",
                                     {in("Vector", v, glm::vec3(0), true).step_size(0.01)}, {out("Fac", f)},
                                     evaluate_gradient,
                                     {prop("type", PropertyKind::Enum, "linear",
                                           {"linear", "quadratic", "easing", "diagonal", "spherical", "radial"})}));
        register_type(registry, type("lfs.noise_texture", "Texture",
                                     {in("Vector", v, position_field(), true).step_size(0.01),
                                      in("Scale", f, 5.0f, true).minimum(0).step_size(0.1),
                                      in("Detail", f, 2.0f, true).range(0, 15).step_size(0.1),
                                      in("Roughness", f, 0.5f, true).range(0, 1).step_size(0.01),
                                      in("Distortion", f, 0.0f, true).minimum(0).step_size(0.1),
                                      in("Seed", f, 0.0f, true).step_size(1)},
                                     {out("Fac", f), out("Colour", c)}, evaluate_noise));
    }
} // namespace lfs::nodes::builtin
