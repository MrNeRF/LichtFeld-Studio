/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "builtin_common.hpp"
#include "core/splat_simplify.hpp"
namespace lfs::nodes::builtin {

    void evaluate_remove_floaters(NodeContext& context) {
        auto geometry = geometry_input(context);
        if (geometry.splats) {
            auto& splats = *geometry.splats;
            auto candidate = splats.opacity.sigmoid().lt(input_float(context, "Min Opacity"));
            const float maximum_size = input_float(context, "Max Size");
            const float radius = input_float(context, "Isolation Radius");
            const int neighbours = input_int(context, "Min Neighbours", 1);
            const bool relative = property_bool(context, "relative_to_size", true);
            if (maximum_size > 0)
                candidate = candidate.logical_or(splats.scaling.exp().max(1).gt(maximum_size));
            if (radius > 0 && neighbours > 0) {
                const auto activated_scale = splats.scaling.exp();
                const auto isolated = (relative ? relative_neighbour_counts(splats.means, activated_scale, radius, neighbours)
                                                : neighbour_counts(splats.means, radius, neighbours))
                                          .lt(static_cast<float>(neighbours));
                candidate = candidate.logical_or(isolated);
            }
            candidate = candidate.logical_and(selection(context, "Selection", field_context(splats), true));
            if (property_bool(context, "preview", false)) {
                splats = filter_splats(splats, candidate);
            } else {
                splats = filter_splats(splats, candidate.logical_not());
            }
        }
        context.set_output("Geometry", std::move(geometry));
    }

    void evaluate_simplify(NodeContext& context) {
        auto geometry = geometry_input(context);
        if (geometry.splats) {
            const auto data = splat_data_from_geometry(geometry);
            core::SplatSimplifyOptions options;
            options.ratio = input_float(context, "Ratio", 0.5f);
            auto simplified = core::simplify_splats(*data, options);
            if (!simplified)
                throw NodeError(simplified.error());
            geometry.splats = geometry_from_splat_data(**simplified).splats;
        }
        context.set_output("Geometry", std::move(geometry));
    }

    void evaluate_decimate(NodeContext& context) {
        auto geometry = geometry_input(context);
        if (geometry.splats) {
            auto& splats = *geometry.splats;
            const float keep_fraction = input_float(context, "Keep Fraction", 0.5f);
            if (keep_fraction < 1 && splats.means.shape()[0] > 0) {
                const auto selected = selection(context, "Selection", field_context(splats), true);
                const size_t selected_count = selected.count_nonzero();
                if (selected_count > 0) {
                    const auto importance = splats.opacity.sigmoid() * splats.scaling.exp().max(1);
                    const auto selected_importance = importance.masked_select(selected).to(DataType::Float32);
                    const auto sorted = selected_importance.sort(0, false).first;
                    const size_t keep_count =
                        std::max<size_t>(static_cast<size_t>(selected_count * keep_fraction), 1);
                    const size_t threshold_index = selected_count - keep_count;
                    const auto threshold = sorted.slice(0, threshold_index, threshold_index + 1);
                    const auto keep = selected.logical_not().logical_or(importance.ge(threshold));
                    splats = filter_splats(splats, keep);
                }
            }
        }
        context.set_output("Geometry", std::move(geometry));
    }

    void register_cleanup(NodeTypeRegistry& registry) {
        const auto geo = std::string(GEOMETRY_SOCKET);
        const auto f = std::string(FLOAT_SOCKET);
        const auto i = std::string(INT_SOCKET);
        register_type(registry,
                      type("lfs.remove_floaters", "Clean-up",
                           geometry_inputs({in("Selection", f, 1.0f, true)
                                                .range(0, 1)
                                                .step_size(0.01),
                                            in("Min Opacity", f, 0.02f).range(0, 1).step_size(0.01),
                                            in("Max Size", f, 0.0f).minimum(0).step_size(0.01),
                                            in("Isolation Radius", f, 3.0f).minimum(0).step_size(0.1),
                                            in("Min Neighbours", i, std::int64_t(1)).minimum(0).step_size(1)}),
                           {out("Geometry", geo)}, evaluate_remove_floaters,
                           {prop("preview", PropertyKind::Bool, false),
                            prop("relative_to_size", PropertyKind::Bool, true)}));
        register_type(registry, type("lfs.simplify", "Clean-up",
                                     geometry_inputs({in("Ratio", f, 0.5f).range(0.01, 1).step_size(0.01)}),
                                     {out("Geometry", geo)},
                                     evaluate_simplify));
        register_type(registry,
                      type("lfs.decimate", "Clean-up",
                           geometry_inputs({in("Selection", f, 1.0f, true)
                                                .range(0, 1)
                                                .step_size(0.01),
                                            in("Keep Fraction", f, 0.5f)
                                                .range(0.001, 1)
                                                .step_size(0.001)}),
                           {out("Geometry", geo)}, evaluate_decimate));
    }

} // namespace lfs::nodes::builtin
