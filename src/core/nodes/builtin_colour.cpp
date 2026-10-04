/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "builtin_common.hpp"
#include "builtin_curve.hpp"

#include <array>
#include <numbers>

namespace lfs::nodes::builtin {
    namespace {
        struct RampStop {
            float position;
            std::array<float, 4> colour;
        };

        std::vector<RampStop> ramp_stops(const NodeContext& context) {
            std::vector<RampStop> result;
            const auto found = context.properties().find("stops");
            if (found != context.properties().end() && found->is_array()) {
                for (const auto& stop : *found) {
                    if (stop.is_array() && stop.size() >= 5) {
                        bool valid = true;
                        for (size_t i = 0; i < 5; ++i)
                            valid = valid && stop[i].is_number() && std::isfinite(stop[i].get<float>());
                        if (valid)
                            result.push_back({stop[0].get<float>(), {stop[1].get<float>(), stop[2].get<float>(), stop[3].get<float>(), stop[4].get<float>()}});
                    }
                }
            }
            if (result.empty())
                result = {{0, {0, 0, 0, 1}}, {1, {1, 1, 1, 1}}};
            std::ranges::sort(result, {}, &RampStop::position);
            return result;
        }

        Tensor evaluate_ramp(const Tensor& fac, const std::vector<RampStop>& stops, std::string_view interpolation,
                             bool alpha) {
            std::vector<core::ColourStop> controls;
            controls.reserve(stops.size());
            for (const auto& stop : stops)
                controls.push_back({stop.position, stop.colour[0], stop.colour[1], stop.colour[2], stop.colour[3]});
            using core::RampInterpolation;
            const auto mode = interpolation == "constant" ? RampInterpolation::Constant : interpolation == "ease" ? RampInterpolation::Ease
                                                                                                                  : RampInterpolation::Linear;
            return core::interpolate_colour_ramp(fac, controls, mode, alpha);
        }

        void evaluate_colour_ramp(NodeContext& context) {
            const auto stops = ramp_stops(context);
            const auto interpolation = property_string(context, "interpolation", "linear");
            const auto fac = context.field("Fac", FLOAT_SOCKET);
            context.set_output("Colour", operation(COLOUR_SOCKET, {fac}, [stops, interpolation](const auto& values) {
                                   return evaluate_ramp(values[0], stops, interpolation, false);
                               }));
            context.set_output("Alpha", operation(FLOAT_SOCKET, {fac}, [stops, interpolation](const auto& values) {
                                   return evaluate_ramp(values[0], stops, interpolation, true);
                               }));
        }

        void evaluate_rgb_curves(NodeContext& context) {
            auto geometry = geometry_input(context);
            const std::array curves{curve_knots(curve_points(context, "combined")), curve_knots(curve_points(context, "r")),
                                    curve_knots(curve_points(context, "g")), curve_knots(curve_points(context, "b"))};
            const auto apply = [&](auto& component, const Tensor& old_colour, bool sh_dc = false) {
                const auto fc = field_context(component);
                return core::interpolate_rgb_curves(old_colour, selection(context, "Selection", fc),
                                                    context.evaluate_field("Factor", fc, FLOAT_SOCKET),
                                                    {curves[0], curves[1], curves[2], curves[3]}, sh_dc);
            };
            if (geometry.splats) {
                auto& splats = *geometry.splats;
                splats.sh0 = apply(splats, splats.sh0, true);
            }
            if (geometry.points) {
                auto& points = *geometry.points;
                points.colors = apply(points, points.colors);
            }
            if (geometry.mesh && geometry.mesh->mesh) {
                auto& component = *geometry.mesh;
                const auto old_colour = component.mesh->has_colors()
                                            ? component.mesh->colors.slice(1, 0, 3)
                                            : Tensor::ones({static_cast<size_t>(component.mesh->vertex_count()), 3},
                                                           component.mesh->vertices.device());
                auto mesh = copy_mesh(*component.mesh, component.mesh->vertices, component.mesh->indices);
                const auto rgb = apply(component, old_colour);
                const auto alpha = component.mesh->has_colors()
                                       ? component.mesh->colors.slice(1, 3, 4)
                                       : Tensor::ones({rgb.shape()[0], 1}, rgb.device());
                mesh->colors = Tensor::cat({rgb, alpha}, 1);
                component.mesh = std::move(mesh);
            }
            context.set_output("Geometry", std::move(geometry));
        }

    } // namespace
    void register_colour(NodeTypeRegistry& registry) {
        const auto geo = std::string(GEOMETRY_SOCKET);
        const auto f = std::string(FLOAT_SOCKET);
        const auto c = std::string(COLOUR_SOCKET);
        const auto curve = nlohmann::json::array({{0, 0}, {1, 1}});
        register_type(registry, type("lfs.rgb_curves", "Colour",
                                     geometry_inputs({in("Selection", f, 1.0f, true).range(0, 1).step_size(0.01),
                                                      in("Factor", f, 1.0f, true).range(0, 1).step_size(0.01)}),
                                     {out("Geometry", geo)}, evaluate_rgb_curves,
                                     {prop("combined", PropertyKind::Data, curve), prop("r", PropertyKind::Data, curve),
                                      prop("g", PropertyKind::Data, curve), prop("b", PropertyKind::Data, curve)}));
        const auto stops = nlohmann::json::array({{0, 0, 0, 0, 1}, {1, 1, 1, 1, 1}});
        register_type(registry, type("lfs.colour_ramp", "Colour",
                                     {in("Fac", f, 0.5f, true).step_size(0.01)},
                                     {out("Colour", c), out("Alpha", f)}, evaluate_colour_ramp,
                                     {prop("stops", PropertyKind::Data, stops),
                                      prop("interpolation", PropertyKind::Enum, "linear", {"constant", "linear", "ease"})}));
    }
} // namespace lfs::nodes::builtin
