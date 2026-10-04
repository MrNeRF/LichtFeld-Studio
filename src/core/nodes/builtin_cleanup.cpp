/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "builtin_common.hpp"
#include "core/splat_simplify.hpp"
#include "core/tensor_spatial.hpp"
namespace lfs::nodes::builtin {

    namespace {
        Tensor local_spacing(const Tensor& positions) {
            const size_t count = positions.shape()[0];
            if (count < 2)
                return Tensor::full({count}, 1e-6f, positions.device());
            const size_t sampled = std::min<size_t>(4096, count);
            const auto indices = (Tensor::linspace(0, static_cast<float>(sampled - 1), sampled, positions.device()) *
                                  (static_cast<float>(count) / sampled))
                                     .to(DataType::Int32);
            const auto sample = positions.index_select(0, indices).sort(0).first;
            const auto extent = sample.slice(0, sampled * 95 / 100, sampled * 95 / 100 + 1) -
                                sample.slice(0, sampled * 5 / 100, sampled * 5 / 100 + 1);
            std::array<float, 3> widths;
            for (int axis = 0; axis < 3; ++axis)
                widths[axis] = channel(extent, axis).item<float>();
            std::ranges::sort(widths);
            const float width = std::max(1e-6f, widths[1] < widths[2] * 1e-3f
                                                    ? 2 * widths[2] / count
                                                : widths[0] < widths[2] * 1e-3f
                                                    ? 2 * std::sqrt(widths[1] * widths[2] / count)
                                                    : 2 * std::cbrt(widths[0] * widths[1] * widths[2] / count));
            return core::point_neighbor_spacing(positions, width);
        }

        struct Components {
            Tensor labels;
            Tensor sizes;
            int iterations = 0;
        };

        Components connected_components(const Tensor& positions, float multiple, int minimum_size) {
            const size_t count = positions.shape()[0];
            // Three local neighbours keep isolated small groups local. A distant
            // eighth neighbour could otherwise inflate their radii into a surface.
            const auto radii = (local_spacing(positions) * multiple).clamp_min(1e-6f);
            const int first_octave = static_cast<int>(std::ceil(std::log2(radii.min().item<float>())));
            const int last_octave = static_cast<int>(std::ceil(std::log2(radii.max().item<float>())));
            auto labels = (Tensor::ones({count}, positions.device(), DataType::Int32).cumsum(0) - 1)
                              .to(DataType::Int32);
            const auto ones = Tensor::ones({count}, positions.device(), DataType::Int32);
            Tensor sizes;
            int iterations = 0;
            constexpr int maximum_iterations = 64;
            for (; iterations < maximum_iterations; ++iterations) {
                auto next = labels;
                for (int octave = first_octave; octave <= last_octave; ++octave)
                    next = next.minimum(core::radius_neighbor_min(positions, labels, std::ldexp(1.0f, octave), &radii));
                next = next.minimum(next.index_select(0, next.clamp_min(0)));
                auto histogram = Tensor::zeros({count + 1}, positions.device(), DataType::Int32);
                histogram.index_add_(0, next + 1, ones);
                sizes = histogram.index_select(0, next + 1);
                // Once a connected subset reaches Min Size its exact root no
                // longer matters. Propagate -1 to certify the rest of that
                // component, avoiding convergence across an entire large scene.
                next = Tensor::where(next.lt(0).logical_or(sizes.ge(minimum_size)),
                                     Tensor::full({count}, -1, positions.device(), DataType::Int32), next);
                const bool stable = next.ne(labels).count_nonzero() == 0;
                labels = std::move(next);
                if (stable) {
                    ++iterations;
                    break;
                }
            }
            return {labels, Tensor::where(labels.lt(0), Tensor::full({count}, minimum_size, positions.device(), DataType::Int32), sizes), iterations};
        }

        Field captured_selection(Tensor mask) {
            return Field(std::string(BOOL_SOCKET), [mask = std::move(mask)](const FieldContext& domain, FieldMemo&) {
                if (domain.size() != mask.numel())
                    throw NodeError("Remove Clumps selection is evaluated on geometry with a different element count");
                return mask.device() == domain.device() ? mask : mask.to(domain.device());
            });
        }
    } // namespace

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
            const auto selected = selection(context, "Selection", field_context(splats), true);
            if (radius > 0 && neighbours > 0) {
                // Only selected splats that no other test removes need their neighbours counted.
                const auto queries = selected.logical_and(candidate.logical_not());
                const auto counts = relative ? relative_neighbour_counts(splats.means, splats.scaling.exp(), radius, neighbours, &queries)
                                             : neighbour_counts(splats.means, radius, neighbours, &queries);
                candidate = candidate.logical_or(counts.lt(static_cast<float>(neighbours)).logical_and(queries));
            }
            candidate = candidate.logical_and(selected);
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
        if (geometry.splats && geometry.splats->means.shape()[0] != 0 &&
            input_float(context, "Ratio", 0.5f) < 1.0f) {
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

    void evaluate_remove_clumps(NodeContext& context) {
        auto geometry = geometry_input(context);
        Tensor remove;
        const int minimum_size = std::max(1, input_int(context, "Min Size", 16));
        const float radius_multiple = input_float(context, "Radius", 2.0f);
        if (geometry.splats) {
            auto& component = *geometry.splats;
            const size_t count = component.means.shape()[0];
            remove = Tensor::full_bool({count}, false, component.means.device());
            if (count && radius_multiple > 0) {
                const auto components = connected_components(component.means, radius_multiple, minimum_size);
                remove = components.sizes.lt(minimum_size).logical_and(selection(context, "Selection", field_context(component), true));
                if (property_bool(context, "delete", true))
                    component = filter_splats(component, remove.logical_not());
            }
        } else if (geometry.points) {
            auto& component = *geometry.points;
            const size_t count = component.positions.shape()[0];
            remove = Tensor::full_bool({count}, false, component.positions.device());
            if (count && radius_multiple > 0) {
                const auto components = connected_components(component.positions, radius_multiple, minimum_size);
                remove = components.sizes.lt(minimum_size).logical_and(selection(context, "Selection", field_context(component), true));
                if (property_bool(context, "delete", true))
                    component = filter_points(component, remove.logical_not());
            }
        }
        if (remove.is_valid())
            context.set_output("Selection", captured_selection(remove));
        else
            context.set_output("Selection", constant_field(false, BOOL_SOCKET));
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
        register_type(registry,
                      type("lfs.remove_clumps", "Clean-up",
                           geometry_inputs({in("Selection", f, 1.0f, true).range(0, 1).step_size(0.01),
                                            in("Radius", f, 2.0f).minimum(0).step_size(0.1),
                                            in("Min Size", i, std::int64_t(16)).minimum(1).step_size(1)}),
                           {out("Geometry", geo), out("Selection", std::string(BOOL_SOCKET))},
                           evaluate_remove_clumps,
                           {prop("delete", PropertyKind::Bool, true)}));
    }

} // namespace lfs::nodes::builtin
