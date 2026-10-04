/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "builtin_common.hpp"
#include "core/splat_data_transform.hpp"
#include <numbers>
namespace lfs::nodes::builtin {
    namespace {
        Tensor trs_matrices(const Tensor& translation, const Tensor& degrees, const Tensor& scale) {
            const auto a = degrees * (std::numbers::pi_v<float> / 180.0f);
            const auto x = channel(a, 0), y = channel(a, 1), z = channel(a, 2);
            const auto cx = x.cos(), sx = x.sin(), cy = y.cos(), sy = y.sin(), cz = z.cos(), sz = z.sin();
            const auto s0 = channel(scale, 0), s1 = channel(scale, 1), s2 = channel(scale, 2);
            const auto zero = Tensor::zeros_like(x), one = Tensor::ones_like(x);
            return Tensor::stack({cz * cy * s0, (cz * sy * sx - sz * cx) * s1, (cz * sy * cx + sz * sx) * s2, channel(translation, 0),
                                  sz * cy * s0, (sz * sy * sx + cz * cx) * s1, (sz * sy * cx - cz * sx) * s2, channel(translation, 1),
                                  sy.neg() * s0, cy * sx * s1, cy * cx * s2, channel(translation, 2), zero, zero, zero, one},
                                 1)
                .reshape({int(translation.size(0)), 4, 4});
        }
        void evaluate_instances(NodeContext& context) {
            const auto anchors = geometry_input(context, "Points"), instance = geometry_input(context, "Instance");
            if (!instance.splats)
                throw NodeError("Instance on Points requires splat geometry for Instance");
            const size_t instance_count = instance.splats->means.size(0);
            std::vector<Tensor> translations, rotations, scales;
            size_t count = 0;
            const auto collect = [&](const auto& component, const Tensor& positions) {
                const auto domain = field_context(component);
                const auto indices = selection(context, "Selection", domain, true).nonzero().reshape({-1}).to(DataType::Int32);
                count += indices.numel();
                if (instance_count && count > 50'000'000 / instance_count)
                    throw NodeError("Instance on Points exceeds the 50 million splat limit");
                translations.push_back(positions.index_select(0, indices));
                rotations.push_back(context.evaluate_field("Rotation", domain, VECTOR_SOCKET).index_select(0, indices));
                scales.push_back(context.evaluate_field("Scale", domain, VECTOR_SOCKET).index_select(0, indices));
            };
            if (anchors.points)
                collect(*anchors.points, anchors.points->positions);
            if (anchors.mesh && anchors.mesh->mesh)
                collect(*anchors.mesh, anchors.mesh->mesh->vertices);
            if (anchors.splats)
                collect(*anchors.splats, anchors.splats->means);
            Geometry result;
            if (!count || !instance_count) {
                result.splats = filter_splats(*instance.splats, Tensor::full_bool({instance_count}, false, instance.splats->means.device()));
                context.set_output("Geometry", std::move(result));
                return;
            }
            const auto location = Tensor::cat(translations, 0), angles = Tensor::cat(rotations, 0), size = Tensor::cat(scales, 0);
            if (!size.isfinite().all().item<bool>() || !angles.isfinite().all().item<bool>())
                throw NodeError("Instance scales and rotations must be finite");
            const auto transforms = trs_matrices(location, angles, size);
            const auto device = instance.splats->means.device();
            // Integer cumsum preserves every index through the 50M output cap.
            const auto indices = (Tensor::ones({count * instance_count}, device, DataType::Int32).cumsum(0) - 1).to(DataType::Int32);
            const auto source = indices.mod(int(instance_count)).to(DataType::Int32);
            // Cumulative anchor starts avoid float division losing indices above 2^24.
            const auto anchor = (source.eq(0).to(DataType::Int32).cumsum(0) - 1).to(DataType::Int32);
            auto splats = *instance.splats;
            splats.means = splats.means.index_select(0, source);
            splats.sh0 = splats.sh0.index_select(0, source);
            splats.shN = splats.shN.index_select(0, source);
            splats.scaling = splats.scaling.index_select(0, source);
            splats.rotation = splats.rotation.index_select(0, source);
            splats.opacity = splats.opacity.index_select(0, source);
            for (auto& [_, value] : splats.attributes)
                value = value.index_select(0, source);
            auto attributes = std::move(splats.attributes);
            result.splats = std::move(splats);
            auto data = splat_data_from_geometry(result);
            core::transform(*data, transforms, anchor);
            result = geometry_from_splat_data(*data);
            result.splats->attributes = std::move(attributes);
            context.set_output("Geometry", std::move(result));
        }
    } // namespace
    void register_instances(NodeTypeRegistry& registry) {
        const auto geo = std::string(GEOMETRY_SOCKET), v = std::string(VECTOR_SOCKET), f = std::string(FLOAT_SOCKET);
        register_type(registry, type("lfs.instance_on_points", "Instances",
                                     {in("Points", geo), in("Instance", geo), in("Selection", f, 1.0f, true).range(0, 1).step_size(0.01),
                                      in("Rotation", v, glm::vec3(0), true).step_size(1), in("Scale", v, glm::vec3(1), true).step_size(0.01)},
                                     {out("Geometry", geo)}, evaluate_instances));
    }
} // namespace lfs::nodes::builtin
