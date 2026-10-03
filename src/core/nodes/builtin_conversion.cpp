/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "builtin_common.hpp"
namespace lfs::nodes::builtin {

    void evaluate_points_to_splats(NodeContext& context) {
        auto geometry = geometry_input(context);
        if (geometry.points) {
            const auto& points = *geometry.points;
            const auto count = points.positions.shape()[0];
            const auto device = points.positions.device();
            float radius = input_float(context, "Radius");
            const float opacity = std::clamp(input_float(context, "Opacity", 1), 1e-6f, 1 - 1e-6f);
            if (radius <= 0 && count) {
                const auto extent = points.positions.max(0) - points.positions.min(0);
                const float volume = extent.slice(0, 0, 1).item<float>() *
                                     extent.slice(0, 1, 2).item<float>() *
                                     extent.slice(0, 2, 3).item<float>();
                radius = 0.5f * std::cbrt(std::max(volume, 1e-12f) / static_cast<float>(count));
            }
            radius = std::max(radius, 1e-6f);
            const auto rotation =
                Tensor::cat({Tensor::ones({count, 1}, device), Tensor::zeros({count, 3}, device)}, 1);
            geometry.splats =
                SplatsComponent{points.positions,
                                (points.colors - 0.5f) / kShC0,
                                Tensor::zeros({count, 0, 3}, device),
                                Tensor::full({count, 3}, std::log(radius), device),
                                rotation,
                                Tensor::full({count}, std::log(opacity / (1 - opacity)), device),
                                0,
                                1,
                                points.attributes};
        }
        context.set_output("Geometry", std::move(geometry));
    }

    void evaluate_splats_to_points(NodeContext& context) {
        auto geometry = geometry_input(context);
        if (geometry.splats) {
            const auto& splats = *geometry.splats;
            geometry.points =
                PointsComponent{splats.means, (splats.sh0 * kShC0 + 0.5f).clamp(0, 1), splats.attributes};
        }
        context.set_output("Geometry", std::move(geometry));
    }

    void evaluate_mesh_to_points(NodeContext& context) {
        auto geometry = geometry_input(context);
        if (geometry.mesh && geometry.mesh->mesh) {
            const auto& mesh = *geometry.mesh->mesh;
            const auto colours = mesh.has_colors()
                                     ? mesh.colors.slice(1, 0, 3)
                                     : Tensor::ones({static_cast<std::size_t>(mesh.vertex_count()), 3},
                                                    mesh.vertices.device());
            geometry.points = PointsComponent{mesh.vertices, colours, {}};
        }
        context.set_output("Geometry", std::move(geometry));
    }
    void register_conversion(NodeTypeRegistry& registry) {
        const auto geo = std::string(GEOMETRY_SOCKET);
        const auto f = std::string(FLOAT_SOCKET);
        register_type(registry, type("lfs.points_to_splats", "Points to Splats", "Conversion",
                                     "Create isotropic degree-zero splats from point positions and colours.",
                                     geometry_inputs({in("Radius", f, 0.0f).minimum(0).step_size(0.01),
                                                      in("Opacity", f, 0.9f).range(0, 1).step_size(0.01)}),
                                     {out("Geometry", geo)}, evaluate_points_to_splats));
        register_type(registry, type("lfs.splats_to_points", "Splats to Points", "Conversion",
                                     "Create points from splat positions and clamped base colours.",
                                     geometry_inputs({}), {out("Geometry", geo)}, evaluate_splats_to_points));
        register_type(registry, type("lfs.mesh_to_points", "Mesh to Points", "Conversion",
                                     "Create points from mesh vertices and vertex colours.",
                                     geometry_inputs({}), {out("Geometry", geo)}, evaluate_mesh_to_points));
    }

} // namespace lfs::nodes::builtin
